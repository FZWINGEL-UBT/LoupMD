"""PolarMACE finite-difference stress with its unstrained reciprocal index set.

The model's hard reciprocal cutoff is discontinuous under cell changes. Stress differentiates the selected
terms, not the cutoff mask. Only finite-difference probes hold that mask fixed; model evaluations are unchanged.
The scoped MACE module patch requires serial reference evaluations, as used by test_gate.py.
"""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np


@contextmanager
def frozen_kset(cutoff, cell):
    """Hold integer reciprocal indices, while allowing their vectors/norms to follow each strained cell."""
    import torch
    import mace.modules.extensions as extensions

    original = extensions.compute_k_vectors_flat
    cell_tensor = torch.as_tensor(np.asarray(cell), dtype=cutoff.dtype, device=cutoff.device)[None]
    reciprocal = 2 * np.pi * torch.linalg.inv(cell_tensor.transpose(-1, -2))
    with torch.no_grad():
        vectors, _, batch, zero = original(cutoff, cell_tensor, reciprocal)
        indices = (vectors @ torch.linalg.inv(reciprocal[0])).round()

    def fixed(cutoff, cell_vectors, r_cell_vectors):
        if cell_vectors.shape[0] != 1:
            raise ValueError("the frozen reciprocal set belongs to one configuration")
        vectors = indices @ r_cell_vectors[0]
        return vectors, (vectors * vectors).sum(-1), batch, zero

    with patch.object(extensions, "compute_k_vectors_flat", fixed):
        yield len(indices)


def fd_stress(model_path, configs, *, device, h):
    """Corrected float64 stress from energy differences; no k-vector may enter or leave a probe's sum."""
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float64")
    out = []
    for _, atoms in configs:
        def energy(eps):
            at = atoms.copy()
            transform = np.eye(3) + eps
            at.set_cell(atoms.cell.array @ transform, scale_atoms=False)
            at.positions = atoms.positions @ transform
            at.calc = calc
            return at.get_potential_energy()

        volume = atoms.get_volume()
        stress = np.zeros((3, 3))
        with frozen_kset(calc.models[0].kspace_cutoff, atoms.cell.array):
            for i in range(3):
                for j in range(i, 3):
                    eps = np.zeros((3, 3))
                    eps[i, j] = eps[j, i] = h
                    value = (energy(eps) - energy(-eps)) / (2 * h * volume * (1 if i == j else 2))
                    stress[i, j] = stress[j, i] = value
        out.append(stress)
    return out
