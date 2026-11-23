"""Audit a frozen, unaugmented FP64 Stage 0 fit, without fitting or scoring either test set.

From the checkout, with mace-torch==0.3.16 (use local_gpu.py run on WSL):
  PYTHONPATH=. python tools/dev/audit_distill_fit.py SPLIT_DIR JOB_OUT report.json
JOB_OUT must be trusted: student.model is a Python pickle. Predict every train and valid
frame, in file order, with the stock FP64 CPU model. Stress RMSE uses all nine tensor
components, like the training log. Force RMSE pools Cartesian components, not vectors.
"""
import argparse
import ast
from collections import defaultdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from ase.io import iread

from loupmd.distill import train


def teacher_replay(raw, calc, student_forces, atom):
    """Independent stock teacher on the serialized geometry; never use replay to relabel a split."""
    raw.calc = calc
    e = raw.get_potential_energy()
    f = raw.get_forces(apply_constraint=False)
    s = raw.get_stress(voigt=False, apply_constraint=False)
    df = f - raw.arrays['REF_forces']
    return dict(energy_delta_eV_atom=(e-raw.info['REF_energy'])/len(raw), force_delta_sse=float(np.square(df).sum()),
                max_abs_force_delta=float(np.abs(df).max()), max_abs_stress_delta=float(np.abs(s-raw.info['REF_stress']).max()),
                student_vs_replayed_force_sse=float(np.square(student_forces-f).sum()), replayed_zero_sse=float(np.square(f).sum()),
                replayed_force_at_worst_student_atom=f[atom].tolist())


def frame_record(raw, graph, prediction, index):
    n = len(raw)
    ref = raw.arrays['REF_forces']
    pred = prediction['forces'].detach().numpy()
    err = pred - ref
    atom_sse = np.square(err).sum(axis=1)
    worst = int(atom_sse.argmax())
    i, j = graph.edge_index.numpy()
    dr = raw.positions[j] - raw.positions[i] + graph.shifts.numpy()
    distance = np.linalg.norm(dr, axis=1)
    info = raw.info
    species = {}
    for z in np.unique(raw.numbers):
        mask = raw.numbers == z
        species[str(z)] = [int(mask.sum()) * 3, float(atom_sse[mask].sum()), float(np.square(ref[mask]).sum())]
    return dict(index=index, frame_id=info['frame_id'], trajectory_id=info['trajectory_id'], regime=info['regime'],
                replica=int(info['replica']), temperature_K=float(info['temperature_K']), step=int(info['step']),
                cell_edge_equiv_A=raw.get_volume() ** (1/3), deformation=np.asarray(info['deformation']).tolist(),
                min_pair_A=float(distance.min()), n=n, force_sse=float(atom_sse.sum()), zero_sse=float(np.square(ref).sum()),
                force_by_Z_ncomp_sse_zero=species, energy_err_eV_atom=(prediction['energy'].item()-info['REF_energy'])/n,
                stress_mse=float(np.square(prediction['stress'].detach().numpy()[0] - info['REF_stress']).mean()),
                worst_atom=dict(index=worst, Z=int(raw.numbers[worst]), sse=float(atom_sse[worst]),
                                nearest_A=float(distance[i == worst].min()), pred=pred[worst].tolist(), ref=ref[worst].tolist()))


def summary(rows):
    ncomp = sum(r['n'] for r in rows) * 3
    sse = sum(r['force_sse'] for r in rows)
    zero = sum(r['zero_sse'] for r in rows)
    worst = max(rows, key=lambda r: r['force_sse'])
    species = defaultdict(lambda: np.zeros(3))
    for row in rows:
        for z, stats in row['force_by_Z_ncomp_sse_zero'].items():
            species[z] += stats
    other = [r for r in rows if r is not worst]
    return dict(frames=len(rows), force_rmse=(sse/ncomp)**0.5, zero_force_rmse=(zero/ncomp)**0.5,
                energy_rmse_eV_atom=np.mean([r['energy_err_eV_atom']**2 for r in rows])**0.5,
                stress_rmse=np.mean([r['stress_mse'] for r in rows])**0.5,
                force_by_Z={z: dict(rmse=(s/n)**0.5, zero_rmse=(b/n)**0.5) for z, (n, s, b) in species.items()},
                worst_frame=worst['frame_id'], worst_frame_sse_fraction=worst['force_sse']/sse if sse else None,
                without_worst_force_rmse=(sum(r['force_sse'] for r in other)/(3*sum(r['n'] for r in other)))**0.5 if other else None,
                frames_worse_than_zero=sum(r['force_sse'] > r['zero_sse'] for r in rows))


def groups(rows, key):
    parts = defaultdict(list)
    for r in rows:
        parts[r[key]].append(r)
    return {k: summary(v) for k, v in parts.items()}


def audit(args):
    # No CUDA discovery or model migration; hide devices before importing torch/MACE.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if version('mace-torch') != train.MACE_VERSION:
        raise ValueError(f'requires mace-torch=={train.MACE_VERSION}')
    import torch
    from mace import data
    from mace.tools import torch_geometric, utils

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    manifest = args.job / 'manifest.json'
    job = json.loads(manifest.read_text())
    if (job['mace_torch'] != train.MACE_VERSION or job['centered_before_cast'] or job['recipe']['dtype'] != 'float64'
            or [s['role'] for s in job['sources']] != list(train.BASE_SPLITS) or job['resume']):
        raise ValueError('expected an unaugmented, uncentered FP64 base recipe')
    for s in job['sources']:
        s['path'] = str(args.splits / (s['role'] + '.xyz'))
    path = args.job / 'student.model'
    if train.file_digest(path) != job['student_sha256']:
        raise ValueError('student digest mismatch')
    model = torch.load(path, map_location='cpu', weights_only=False).eval().requires_grad_(False)
    epoch = job['student_epoch']
    if args.pre_stage_two:
        path = args.job / 'pre_stage_two.pt'
        if train.file_digest(path) != job['pre_stage_two_sha256']:
            raise ValueError('pre-stage-two checkpoint digest mismatch')
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        del state
        epoch = job['pre_stage_two_epoch']
    if any(p.dtype != torch.float64 for p in model.parameters()) or model.heads != ['Default']:
        raise ValueError('expected a stock FP64 single-head student')
    zs = model.atomic_numbers.tolist()
    ztable = utils.AtomicNumberTable(zs)
    e0 = job['energy_offsets_eV']
    options = job['mace_args']
    def option(name):
        return options[options.index('--' + name) + 1]
    np.testing.assert_allclose(model.atomic_energies_fn.atomic_energies.numpy().ravel(), [e0[str(z)] for z in zs], rtol=0, atol=1e-12)
    np.testing.assert_equal(ast.literal_eval(option('E0s')), {int(z): e for z, e in e0.items()})
    spec = data.KeySpecification(info_keys={k: option(k+'_key') for k in ('energy', 'stress')},
                                 arrays_keys={'forces': option('forces_key')})
    log = (args.job / 'train.log').read_text()
    epochs = {int(ep): float(f)/1000 for ep, f in re.findall(r'Epoch (\d+): head: Default,[^\n]*?RMSE_F=\s*([\d.]+) meV / A', log)}
    if sorted(epochs) != list(range(job['recipe']['epochs'])):
        raise ValueError('incomplete log: expected one validation record every epoch')
    report = dict(schema=1, manifest_sha256=train.file_digest(manifest), student_sha256=job['student_sha256'],
                  train_module_sha256=train.file_digest(train.__file__), split_sha256={s['role']: s['sha256'] for s in job['sources']},
                  versions={p: version(p) for p in ('mace-torch', 'torch', 'e3nn', 'ase', 'numpy')}, device='cpu',
                  dtype='float64', threads=args.threads, units=dict(energy='eV/atom error', forces='eV/angstrom',
                  stress='eV/angstrom^3, tensile-positive potential stress (3x3)', positions='angstrom'),
                  E0_eV=e0, centered_before_cast=False, teacher_head=job['teacher_head'], teacher_sha256=job['teacher_sha256'],
                  loss_weights={p: {k: float(option(p+k+'_weight')) for k in ('energy', 'forces', 'stress')} for p in ('', 'stage_two_')},
                  logged_validation_force_rmse=epochs, student_epoch=job['student_epoch'], splits={})
    if args.pre_stage_two:
        report['pre_stage_two_sha256'] = job['pre_stage_two_sha256']
        report['evaluated_epoch'] = epoch
    teacher = None
    if args.teacher:
        from mace.calculators import MACECalculator

        if train.file_digest(args.teacher) != job['teacher_sha256']:
            raise ValueError('teacher digest mismatch')
        teacher = MACECalculator(model_paths=str(args.teacher), device='cpu', default_dtype='float64', head=job['teacher_head'])
        if teacher.head != job['teacher_head']:
            raise ValueError('wrong stock teacher head')
        report['teacher_replay_sampling'] = 'all validation; first two in file order per training trajectory'
    counts, energies = [], []
    with TemporaryDirectory() as tmp:
        dest = Path(tmp) / 'inputs'
        train.write_inputs(job, dest)  # verifies all raw-file digests before writing each split
        for s in job['sources'][:2]:  # no test predictions, breakdowns or checkpoint selection
            role = s['role']
            rows = []
            seen = defaultdict(int)
            inputs = dest / (role + '.xyz')
            for idx, (raw, material) in enumerate(zip(iread(s['path'], ':'), iread(inputs, ':'), strict=True)):
                if any(raw.info[k] != material.info[k] for k in ('frame_id', 'trajectory_id', 'split', 'sampler', 'teacher_sha256', 'teacher_head')):
                    raise ValueError(f'{role}[{idx}]: provenance or frame order changed')
                if raw.info['split'] != role or (raw.info['teacher_sha256'], raw.info['teacher_head']) != (job['teacher_sha256'], job['teacher_head']):
                    raise ValueError(f'{role}[{idx}]: unexpected split or teacher')
                cfg = data.config_from_atoms(material, key_specification=spec)
                for x, y in ((cfg.atomic_numbers, raw.numbers), (cfg.positions, raw.positions), (cfg.cell, raw.cell.array), (cfg.pbc, raw.pbc)):
                    np.testing.assert_array_equal(x, y)
                graph = data.AtomicData.from_config(cfg, ztable, float(model.r_max), heads=model.heads)
                for k, v in (('energy', raw.info['REF_energy']), ('forces', raw.arrays['REF_forces']), ('stress', raw.info['REF_stress'])):
                    target = getattr(graph, k).numpy()
                    if k == 'stress':
                        target = target[0]
                    np.testing.assert_array_equal(target, v)
                    np.testing.assert_equal(getattr(graph, k+'_weight').item(), 1.)
                np.testing.assert_equal(graph.weight.item(), 1.)
                if role == 'train':
                    counts.append([np.count_nonzero(raw.numbers == z) for z in zs])
                    energies.append(raw.info['REF_energy'])
                batch = torch_geometric.Batch.from_data_list([graph])
                out = model(batch.to_dict(), compute_force=True, compute_stress=True, training=False)
                rows.append(frame_record(raw, graph, out, idx))
                traj = raw.info['trajectory_id']
                if teacher is not None and (role == 'valid' or seen[traj] < 2):
                    rows[-1]['teacher_replay'] = teacher_replay(raw, teacher, out['forces'].detach().numpy(), rows[-1]['worst_atom']['index'])
                seen[traj] += 1
            if len(rows) != s['frames']:
                raise ValueError(f'{role}: frame count differs from archived run')
            report['splits'][role] = dict(materialized_sha256=train.file_digest(inputs), rows=rows,
                                          overall=summary(rows), by_regime=groups(rows, 'regime'), by_trajectory=groups(rows, 'trajectory_id'))
    np.testing.assert_allclose(np.linalg.lstsq(counts, energies, rcond=None)[0], [e0[str(z)] for z in zs], atol=1e-12, rtol=0)
    logged = epochs[epoch]
    observed = report['splits']['valid']['overall']['force_rmse']
    report['validation_final'] = dict(logged=logged, cpu=observed, delta=observed-logged, log_rounding_half_width=5e-6,
                                      matches_at_printed_precision=abs(observed-logged) <= 5e-6)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('splits', type=Path, help='directory of the three original frozen XYZ files')
    p.add_argument('job', type=Path, help='trusted directory with manifest.json, student.model and train.log')
    p.add_argument('output', type=Path, help='new JSON file: identities, log, metrics and per-frame evidence')
    p.add_argument('--threads', type=int, default=4, help='CPU intra-op threads (default: 4)')
    p.add_argument('--pre-stage-two', action='store_true', help='load the archived boundary EMA instead of final weights')
    p.add_argument('--teacher', type=Path, help='trusted teacher pickle: fresh CPU E/F/stress on all valid and two train frames/trajectory')
    args = p.parse_args()
    report = audit(args)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(report['validation_final']))
    if not report['validation_final']['matches_at_printed_precision']:
        raise SystemExit('validation differs from the archived rounded value; see report')


if __name__ == '__main__':
    main()
