"""Assign water IDs once from unique periodic O-H covalent bonds; audit labelled frames.

python -m loupmd.distill.water label water_216.extxyz initial.xyz
python -m loupmd.distill.water inspect INITIAL.xyz SPLIT.xyz [SPLIT.xyz ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import iread, read, write
from ase.neighborlist import neighbor_list


def assign_water_ids(atoms):
    if set(atoms.numbers) != {1, 8} or not atoms.pbc.all() or atoms.cell.rank != 3:
        raise ValueError('expected fully periodic water')
    oxygens = np.flatnonzero(atoms.numbers == 8)
    hydrogens = np.flatnonzero(atoms.numbers == 1)
    i, j = neighbor_list('ij', atoms, {(1, 8): 1.30})
    take = atoms.numbers[i] == 8
    o, h = i[take], j[take]
    n = np.bincount(np.r_[o, h], minlength=len(atoms))
    if np.any(n[oxygens] != 2) or np.any(n[hydrogens] != 1):
        raise ValueError('expected exactly two unique covalent O-H bonds per O and one per H')
    ids = np.empty(len(atoms), dtype=np.int64)
    ids[oxygens] = np.arange(len(oxygens))
    ids[h] = ids[o]
    atoms.set_array('mol_id', ids)
    return inspect_water(atoms)


def inspect_water(atoms):
    # IDs, not spatial nearest neighbors, define the triplets in every saved frame.
    ids = atoms.arrays['mol_id']
    if ids.dtype.kind not in 'iu' or ids.shape != (len(atoms),) or not atoms.pbc.all():
        raise ValueError('invalid periodic water molecule IDs')
    order = np.lexsort((-atoms.numbers, ids))
    _, sizes = np.unique(ids, return_counts=True)
    if (sizes != 3).any() or not np.all(atoms.numbers[order].reshape(-1, 3) == [8, 1, 1]):
        raise ValueError('invalid water molecule composition')
    pos = atoms.positions[order].reshape(-1, 3, 3)
    vectors, bonds = find_mic((pos[:, 1:] - pos[:, :1]).reshape(-1, 3), atoms.cell)
    v = vectors.reshape(-1, 2, 3)
    cosine = (v[:, 0] * v[:, 1]).sum(axis=1) / bonds.reshape(-1, 2).prod(axis=1)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    if not (np.isfinite(bonds).all() and bonds.min() > .65 and bonds.max() < 1.35):
        raise ValueError('invalid intramolecular O-H bond length')
    if not (angle.min() > 65 and angle.max() < 150):
        raise ValueError('invalid intramolecular H-O-H angle')
    return dict(molecules=len(v), bond_min_A=float(bonds.min()), bond_max_A=float(bonds.max()),
                angle_min_deg=float(angle.min()), angle_max_deg=float(angle.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    label = subs.add_parser('label')
    label.add_argument('source', type=Path)
    label.add_argument('output', type=Path)
    inspect = subs.add_parser('inspect')
    inspect.add_argument('initial', type=Path)
    inspect.add_argument('splits', type=Path, nargs='+')
    args = parser.parse_args()
    if args.command == 'label':
        a = read(args.source)
        stats = assign_water_ids(a)
        with args.output.open('x') as f:
            write(f, a, format='extxyz')
        print(json.dumps(stats))
        return
    base = read(args.initial)
    seen, geometry, trajectories, seeds = set(), set(), {}, {}
    from .train import file_digest
    reports = {}
    for path in args.splits:
        rows = []
        for a in iread(path, ':'):
            if not (np.array_equal(a.numbers, base.numbers) and np.array_equal(a.arrays['mol_id'], base.arrays['mol_id'])):
                raise ValueError('molecule membership or atom order changed')
            info = a.info
            fid, tid, seed, role = (info[k] for k in ('frame_id', 'trajectory_id', 'md_seed', 'split'))
            geom = a.positions.tobytes(), a.cell.array.tobytes()
            if fid in seen or geom in geometry or trajectories.setdefault(tid, role) != role or seeds.setdefault(seed, tid) != tid:
                raise ValueError('duplicate geometry, frame ID or shared random stream/trajectory')
            seen.add(fid)
            geometry.add(geom)
            xyz, _ = find_mic(a.positions[a.numbers == 8] - base.positions[base.numbers == 8] @
                              np.linalg.solve(base.cell.array, a.cell.array), a.cell)
            rows.append(dict(frame_id=fid, trajectory_id=tid, md_seed=int(seed), regime=info['regime'],
                             kinetic_K=a.get_temperature(), oxygen_displacement_rms_A=np.sqrt((xyz*xyz).sum(1).mean()),
                             **inspect_water(a)))
        if not rows:
            raise ValueError(f'empty split: {path}')
        reports[str(path)] = dict(sha256=file_digest(path), frames=len(rows), rows=rows)
    print(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
