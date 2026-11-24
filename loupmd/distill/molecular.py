"""Geometry and reference scales for compact, explicitly identified periodic molecules."""
from itertools import product

import numpy as np
from ase.geometry import find_mic, minkowski_reduce


def grouped_geometry(atoms, key):
    if key not in atoms.arrays:
        raise ValueError(f'missing required molecule ID array {key}')
    ids = np.asarray(atoms.arrays[key])
    if ids.dtype.kind not in 'iu' or ids.shape != (len(atoms),) or np.any(ids < 0):
        raise ValueError(f'{key} must contain one nonnegative integer molecule ID per atom')
    order = np.argsort(ids, kind='stable')
    ids = ids[order]
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(ids)])
    masses = atoms.get_masses()[order]
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError('molecular COM requires finite positive masses')

    # IDs without topology/image flags only determine the images of compact molecules.
    pos = atoms.positions[order]
    cell, _ = minkowski_reduce(atoms.cell)
    displ, _ = find_mic(pos - np.repeat(pos[starts], sizes, axis=0), cell)
    # Extrema along each Voronoi-relevant vector check all intramolecular pairs.
    for indices in product((-1, 0, 1), repeat=3):
        if indices <= (0, 0, 0):
            continue
        vec = np.asarray(indices) @ cell
        projection = displ @ vec
        extent = np.maximum.reduceat(projection, starts) - np.minimum.reduceat(projection, starts)
        if np.any(extent >= (0.5 - 1e-12) * (vec @ vec)):
            raise ValueError('ambiguous periodic molecule images; compact molecules required')
    com = np.add.reduceat(masses[:, None] * displ, starts, axis=0) / np.add.reduceat(masses, starts)[:, None]
    return order, starts, displ - np.repeat(com, sizes, axis=0)


def molecular_geometry(atoms, key):
    """Per-atom local anchor index and periodic COM displacement in original order."""
    order, starts, arms = grouped_geometry(atoms, key)
    anchor = np.empty_like(order)
    anchor[order] = np.repeat(order[starts], np.diff(np.r_[starts, len(atoms)]))
    original = np.empty_like(arms)
    original[order] = arms
    return anchor, original


class ReferenceScales:
    """Relative Cartesian component MSEs, from raw base-train forces in float64 only."""

    def __init__(self):
        self.sums = np.zeros(3)
        self.counts = np.zeros(3, dtype=np.int64)

    def observe(self, atoms, grouped):
        order, starts, arms = grouped
        f = np.asarray(atoms.arrays['REF_forces'][order], dtype=np.float64)
        for i, value in enumerate((f, np.add.reduceat(f, starts, axis=0),
                                   np.add.reduceat(np.cross(arms, f), starts, axis=0))):
            self.sums[i] += np.vdot(value, value)
            self.counts[i] += value.size

    def result(self):
        atom, net, torque = self.sums / self.counts
        if not np.all(np.isfinite([atom, net, torque])) or min(atom, net, torque) <= 0:
            raise ValueError('molecular loss normalization requires positive train force and torque scales')
        return dict(atomic_force_component_mse=atom, net_force_component_mse=net, torque_component_mse=torque,
                    net_scale=atom/net, torque_scale=atom/torque, molecule_key='mol_id')
