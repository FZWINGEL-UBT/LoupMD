"""Checks on the generic path's model families that the single-point gate does not make.

Each test skips when its model file is absent (LOUP_MODELS_DIR, or a workstation's model directory).
"""
from __future__ import annotations

import numpy as np
import pytest
from conftest import find_model

from loupmd import LoupEngine, ensure_loup_bundle


def _rattled_nacl(rep: int, sigma: float, seed: int):
    from ase.build import bulk

    at = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((rep, rep, rep))
    at.positions += np.random.default_rng(seed).normal(0.0, sigma, at.positions.shape)
    return at


def test_polar_energy_follows_a_small_cell_change_on_the_same_engine():
    """A second evaluate with a slightly different cell must give what a fresh engine gives for that cell (the
    volume enters the long-range terms; a volume kept from the first call would scale them wrongly)."""
    bundle = ensure_loup_bundle(find_model("MACE-POLAR-1-S.model"), [11, 17])
    at = _rattled_nacl(2, 0.05, 21)  # 64 atoms
    z, m = at.get_atomic_numbers(), at.get_masses()
    eng = LoupEngine(bundle)
    eng.set_system(z, m)
    eng.evaluate(at.positions, at.cell.array)
    # Strains small enough to keep the number of k-vectors of this cell (measured: it changes near 1e-3). With the
    # first cell's volume kept, the differences were 3e-7 / 1e-6 eV/atom, 3e-5 / 9e-5 eV/A and 8e-8 / 3e-7 eV/A^3.
    for strain in (1e-4, 3e-4):
        s = 1.0 + strain
        e_same, f_same, s_same = eng.evaluate(at.positions * s, at.cell.array * s)
        fresh = LoupEngine(bundle)
        fresh.set_system(z, m)
        e_fresh, f_fresh, s_fresh = fresh.evaluate(at.positions * s, at.cell.array * s)
        assert abs(e_same - e_fresh) / len(at) < 2e-7, f"strain {strain}: {(e_same - e_fresh) / len(at):.3e} eV/atom"
        assert np.abs(f_same - f_fresh).max() < 1e-5, f"strain {strain}"
        assert np.abs(s_same - s_fresh).max() < 3e-8, f"strain {strain}"
        eng.evaluate(at.positions, at.cell.array)  # back to the first cell for the next strain


@pytest.mark.parametrize("model,symbols", [("mace-mh-1.model", "OTiSr"), ("MACE-POLAR-1-S.model", "NaCl")])
def test_generic_model_without_neighbour_pairs_matches_mace(model, symbols):
    """Atoms further apart than the cutoff have no edges; a generic model can still give each a non-zero
    interaction energy (residual first block, readout biases, long-range terms), which must not be dropped.
    Measured before the fix: MACE-MH-1 off by 1.67 eV for three atoms, MACE-POLAR-1-S by 1.27 eV for two."""
    import torch
    from ase import Atoms
    from mace.calculators import MACECalculator

    path = find_model(model)
    nat = len(Atoms(symbols))
    at = Atoms(symbols, positions=[(2.0 + 14.0 * i,) * 3 for i in range(nat)], cell=[14.0 * nat] * 3, pbc=True)
    eng = LoupEngine(ensure_loup_bundle(path, sorted(set(at.get_atomic_numbers().tolist()))))
    eng.set_system(at.get_atomic_numbers(), at.get_masses())
    e, f, _ = eng.evaluate(at.positions, at.cell.array)
    heads = list(getattr(torch.load(path, map_location="cpu", weights_only=False), "heads", None) or [])
    head = {"head": next((h for h in ("omat_pbe", "default", "Default") if h in heads), heads[-1])} if len(heads) > 1 else {}
    ref = at.copy()
    ref.calc = MACECalculator(model_paths=path, device="cpu", default_dtype="float64", **head)
    assert abs(e - ref.get_potential_energy()) < 1e-4 * nat
    assert np.abs(f).max() < 1e-6
