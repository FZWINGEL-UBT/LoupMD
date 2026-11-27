"""Qualify MH-1 omat_pbe on one frozen extxyz split in its original order.

Run under the GPU reservation: qualify_frozen_cells.py ORIGINAL.xyz NEW_REPORT.json.
No MD, rattling, cell substitution, training, or writes to the source. The report
contains the exact sequential labels that were gated, for offline reissue.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import repeat
import json
import multiprocessing
import os
from pathlib import Path

from ase.io import iread
import numpy as np

import loupmd as lm
from loupmd import _core, config
from loupgym.distill.frames import _check_geometry, _labels
from loupgym.distill.reissue import NATIVE_AGREEMENT_LIMITS, native_paths_agree
from loupmd.models import sha256_of
from loupmd.validate import gate, references

HEAD = 'omat_pbe'


def stored(at):
    return _labels(at, 'stored', at.info['REF_energy'], at.arrays['REF_forces'], at.info['REF_stress'])


def delta(a, b, n):
    return dict(energy_abs_eV_atom=abs(a[0] - b[0]) / n,
                force_max_abs_eV_A=float(np.max(np.abs(a[1] - b[1]))),
                stress_max_abs_eV_A3=float(np.max(np.abs(a[2] - b[2]))))


def first_call(bundle, at):
    # Same exported weights as ForceField.evaluate. Never hash/reload the Torch
    # checkpoint per frame. This handle sees ONLY the geometry being checked.
    engine = _core.LoupEngine(bundle)
    engine.set_system(at.numbers, at.get_masses(), skin=0.5)
    return engine.evaluate(at.positions, at.cell.array, compute_stress=True)


def qualify(source, checkpoint, fresh_workers=1):
    if not 1 <= fresh_workers <= 4:
        raise ValueError('fresh workers must be 1..4')
    source_sha = sha256_of(source)
    library = _core.library_path().resolve()
    library_sha = sha256_of(library)
    teacher = lm.load('mh1', head=HEAD, checkpoint=checkpoint)
    configs = []
    seen = set()
    for at in iread(source, index=':', format='extxyz'):
        name = at.info['frame_id']
        if not name or name in seen:
            raise ValueError(f'{source}: missing or duplicate frame_id {name!r}')
        if (at.info['teacher_sha256'], at.info['teacher_head']) != (teacher.sha256, HEAD):
            raise ValueError(f'{name}: wrong teacher checkpoint or head')
        if configs and not np.array_equal(at.numbers, configs[0][1].numbers):
            raise ValueError(f'{name}: expected the frozen split atom order/species throughout')
        _check_geometry(at)
        if at.constraints:
            raise ValueError(f'{name}: constrained geometry is not a frozen Stage 0 frame')
        stored(at)
        seen.add(name)
        configs.append((name, at))
    if not configs:
        raise ValueError(f'{source}: empty split')

    # Stock gate protocol: one model per precision, one structure at a time on
    # GPU, before any native engine is allocated. No retained autograd graphs.
    print(f'Running stock FP64/FP32 on {len(configs)} serialized geometries', flush=True)
    r64, r32, stress_refs = references.compute(teacher.path, configs, HEAD, 'cuda', None, None)
    # Do not use run_gate's per-frame set_system or reset on a trajectory boundary:
    # this must be the same cached ForceField path as the original label generator.
    seq = [teacher.evaluate(at, stress=True) for _, at in configs]
    bundle = teacher.bundle
    teacher_sha = teacher.sha256
    del teacher  # the sequential native handle is no longer needed

    # Loading and certifying the full spline table is CPU-bound on each *fresh*
    # handle. Separate spawned processes share no handle or inherited CUDA state.
    # The guard follows all descendants. Never fan out the sequential/stock paths.
    atoms = (at for _, at in configs)
    if fresh_workers == 1:
        fresh_values = map(first_call, repeat(bundle), atoms)
    else:
        with ProcessPoolExecutor(fresh_workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            fresh_values = list(pool.map(first_call, repeat(bundle), atoms))
    rows = []
    results = zip(configs, seq, r64, r32, stress_refs, fresh_values, strict=True)
    for i, ((name, at), reused, ref64, ref32, sref, fresh) in enumerate(results):
        historical = stored(at)
        row = dict(frame_id=name, trajectory_id=at.info['trajectory_id'], step=int(at.info['step']),
                   replica=int(at.info['replica']), regime=at.info['regime'], cell_A=at.cell.array.tolist(),
                   cell_changed=i > 0 and not np.array_equal(at.cell.array, configs[i-1][1].cell.array),
                   sequential_vs_fresh=delta(reused, fresh, len(at)))
        for tag, value in [('sequential', reused), ('fresh', fresh), ('historical', historical)]:
            checked = _labels(at, tag, *value)
            metric = gate.compare(name, {'LOUP_MODE': tag}, len(at), checked, ref64, ref32, sref)
            row[tag] = asdict(metric)
        row['labels'] = dict(REF_energy=reused[0], REF_forces=reused[1].tolist(), REF_stress=reused[2].tolist())
        if at.info['split'] == 'valid' and (row['regime'], row['replica'], row['step']) == ('hot', 0, 700):
            if at.symbols[138] != 'O':
                raise ValueError(f'{name}: expected diagnostic oxygen at index 138')
            row['O138_Fy_eV_A'] = {tag: float(value[1][138, 1]) for tag, value in (
                ('stock64', ref64), ('stock32', ref32), ('fresh', fresh), ('sequential', reused), ('historical', historical))}
        rows.append(row)

    if sha256_of(source) != source_sha or sha256_of(library) != library_sha:
        raise ValueError('source or native library changed during qualification')
    mace_version, torch_version = references.versions()
    return dict(schema='loupmd.frozen-cell-gate.v1', source=str(source.resolve()), source_sha256=source_sha,
                library=str(library), library_sha256=library_sha, library_version=lm.library_version(),
                bundle=str(bundle), bundle_sha256=sha256_of(bundle), teacher_head=HEAD, teacher_sha256=teacher_sha,
                native_environment={k: v for k, v in os.environ.items() if k.startswith('LOUP_') and k != 'LOUP_LIB'},
                fresh_workers=fresh_workers,
                mace_version=mace_version, torch_version=torch_version, reference_device=references.device_tag('cuda'),
                rule=dict(factor=config.GATE_FACTOR, energy_floor_meV_per_atom=config.GATE_ENERGY_FLOOR_MEV_PER_ATOM,
                          force_floor_eV_per_A=config.GATE_FORCE_FLOOR_EV_PER_A, stress_floor_eV_per_A3=config.GATE_STRESS_FLOOR_EV_PER_A3),
                native_agreement_limits=NATIVE_AGREEMENT_LIMITS,
                passed=all(r['sequential']['passed'] and r['fresh']['passed'] and
                           native_paths_agree(r['sequential_vs_fresh']) for r in rows), frames=len(rows), rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('report', type=Path, help='new file; historical failures do not count against corrected labels')
    ap.add_argument('--checkpoint', type=Path, help='explicit MH-1 checkpoint (verified against the model table)')
    ap.add_argument('--fresh-workers', type=int, default=1, choices=range(1, 5),
                    help='process-isolated first calls under the same GPU reservation, after sequential evaluation')
    args = ap.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = qualify(args.source, args.checkpoint, args.fresh_workers)
    with args.report.open('x') as f:
        json.dump(result, f, allow_nan=False, separators=(',', ':'))
        f.write('\n')
    print(f"{result['frames']} frames: " + ', '.join(f"{k} {sum(r[k]['passed'] for r in result['rows'])} passed"
          for k in ('sequential', 'fresh', 'historical')) + f'; {args.report}')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
