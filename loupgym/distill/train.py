"""Train one periodic student with mace-torch 0.3.16. Also runnable as a standalone script.

--plan checks data and prints the exact job without importing torch/MACE or writing files.
Only load trusted local checkpoints: a .model is a Python pickle. student.model is the
physical-energy export/gate input; work/ contains MACE's native training-space outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from ase.io import iread, write


MACE_VERSION = '0.3.16'
BASE_SPLITS = ('train', 'valid', 'teacher-test')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in BASE_SPLITS:
        parser.add_argument('--' + role, required=True, type=Path, help=f'labelled {role} extxyz')
    parser.add_argument('--out', required=True, type=Path, help='new job directory (never overwritten)')
    parser.add_argument('--width', type=int, default=64, help='scalar node channels')
    parser.add_argument('--radial-head', type=int, default=64, help='last radial MLP width: 64 for s64l0, 8 for s64r8')
    parser.add_argument('--cutoff', type=float, default=5.0, help='radial cutoff in angstroms')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    parser.add_argument('--epochs', type=int, default=150, help='total epochs across both phases')
    parser.add_argument('--stage-two-start', type=int, help='first stage-two epoch (zero based; default MACE 3/4 rule)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--valid-batch-size', type=int, default=8)
    parser.add_argument('--molecular-loss', choices=('control', 'net-torque'),
                        help='water pair: same mol_id-labelled inputs and train-only scales. Control uses stock stress loss.')
    parser.add_argument('--resume', type=Path, help='completed parent job with preserved pre_stage_two.pt')
    parser.add_argument('--augment', type=Path, action='append', default=[],
                        help='selected student harvest extxyz, repeat for all rounds; requires --resume')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args(argv)
    if args.stage_two_start is None:
        args.stage_two_start = max(1, args.epochs // 4 * 3)
    if not 0 < args.stage_two_start < args.epochs:
        parser.error('both phases need at least one epoch')
    if not all(x > 0 for x in (args.width, args.radial_head, args.batch_size, args.valid_batch_size,
                               args.cutoff)) or not np.isfinite(args.cutoff):
        parser.error('widths, batches and cutoff must be finite and positive')
    if not 0 <= args.seed < 2**32:
        parser.error('seed must fit in an unsigned 32-bit integer')
    if bool(args.resume) != bool(args.augment):
        parser.error('--resume and --augment must be specified together')
    return args


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _molecular():
    # The standalone upload only needs the sibling files for molecular jobs.
    if __package__:
        from . import molecular
    else:
        import molecular
    return molecular


def _inspect_sources(args):
    """Validate before MACE, which otherwise silently zero-weights missing labels."""
    sources = []
    frame_ids, geometries, trajectories = set(), set(), {}
    teacher = None
    train_counts, train_energies = [], []
    all_z = set()
    scales = _molecular().ReferenceScales() if args.molecular_loss else None
    files = [(role, getattr(args, role.replace('-', '_'))) for role in BASE_SPLITS]
    for role, path in files + [('harvest', p) for p in args.augment]:
        n = 0
        for a in iread(path, index=':', format='extxyz'):
            n += 1
            info = a.info
            label = f'{path}, frame {n}'
            if info.get('split') != role or info.get('sampler') != ('student' if role == 'harvest' else 'teacher'):
                raise ValueError(f'{label}: expected {role} split and its sampler')
            uid, trajectory = info.get('frame_id'), info.get('trajectory_id')
            if not isinstance(uid, str) or not uid or uid in frame_ids:
                raise ValueError(f'{label}: missing or duplicate frame_id {uid!r}')
            if not isinstance(trajectory, str) or not trajectory or trajectories.setdefault(trajectory, role) != role:
                raise ValueError(f'{label}: missing or shared cross-split trajectory_id')
            frame_ids.add(uid)
            ident = (info.get('teacher_sha256'), info.get('teacher_head'))
            if not all(isinstance(k, str) and k for k in ident) or (teacher is not None and teacher != ident):
                raise ValueError(f'{label}: missing or inconsistent teacher identity')
            teacher = ident
            if (not len(a) or not a.pbc.all() or not np.isfinite(a.cell).all() or a.cell.rank != 3
                    or not a.get_volume() > 0 or not np.isfinite(a.positions).all()):
                raise ValueError(f'{label}: expected finite fully periodic geometry with nonzero volume')
            digest = hashlib.sha256(a.numbers.astype('<i4').tobytes())
            digest.update(np.asarray(a.cell, dtype='<f8').tobytes())
            digest.update(np.asarray(a.positions, dtype='<f8').tobytes())
            geom = digest.digest()
            if geom in geometries:
                raise ValueError(f'{label}: duplicate geometry')
            geometries.add(geom)
            for key, value, shape in [('REF_energy', info.get('REF_energy'), ()),
                                      ('REF_forces', a.arrays.get('REF_forces'), (len(a), 3)),
                                      ('REF_stress', info.get('REF_stress'), (3, 3))]:
                if value is None or np.shape(value) != shape or not np.isfinite(value).all():
                    raise ValueError(f'{label}: missing or invalid {key}, expected shape {shape}')
            if not np.allclose(info['REF_stress'], np.asarray(info['REF_stress']).T, atol=1e-10, rtol=1e-7):
                raise ValueError(f'{label}: non-symmetric REF_stress')
            if scales is not None:
                grouped = _molecular().grouped_geometry(a, 'mol_id')
                if role == 'train':
                    scales.observe(a, grouped)
            z, counts = np.unique(a.numbers, return_counts=True)
            all_z.update(z.tolist())
            if role == 'train':
                train_counts.append(dict(zip(z.tolist(), counts.tolist())))
                train_energies.append(float(info['REF_energy']))
        if not n:
            raise ValueError(f'{path}: empty {role} split')
        sources.append(dict(role=role, path=str(path.resolve()), frames=n, sha256=file_digest(path)))
    if sources[0]['frames'] < args.batch_size:
        raise ValueError('base train split smaller than batch-size (MACE drops partial training batches)')
    zs = sorted(set().union(*train_counts))
    if set(zs) != all_z:
        raise ValueError('species outside the original train split')
    counts = np.array([[row.get(z, 0) for z in zs] for row in train_counts], dtype=np.float64)
    # Same minimum-norm fit as MACE --E0s average, including fixed stoichiometry.
    e0 = np.linalg.lstsq(counts, np.asarray(train_energies, dtype=np.float64), rcond=None)[0]
    return sources, {str(z): float(e) for z, e in zip(zs, e0)}, teacher, scales.result() if scales else None


def _arguments(job):
    r = job['recipe']
    out = Path(job['out'])
    inputs = out / 'inputs'
    zeros = r['dtype'] == 'float32'
    e0 = {int(z): 0. if zeros else e for z, e in job['energy_offsets_eV'].items()}
    options = dict(name='student', seed=r['seed'], device='cuda', default_dtype=r['dtype'], work_dir=out / 'work',
                   train_file=inputs / 'train.xyz', valid_file=inputs / 'valid.xyz', test_file=inputs / 'teacher-test.xyz',
                   energy_key='REF_energy', forces_key='REF_forces', stress_key='REF_stress', E0s=repr(e0),
                   model='MACE', hidden_irreps=f"{r['width']}x0e", num_interactions=2, correlation=3, max_ell=3,
                   interaction_first='RealAgnosticInteractionBlock', interaction='RealAgnosticResidualInteractionBlock',
                   MLP_irreps='16x0e', radial_MLP=json.dumps(job['radial_input']['mlp_widths']), r_max=r['cutoff'],
                   radial_type='bessel', num_radial_basis=8, num_cutoff_basis=5, distance_transform='None', apply_cutoff='True',
                   loss='stress', compute_stress='True', error_table='PerAtomRMSEstressvirials',
                   energy_weight=1, forces_weight=100, stress_weight=1,
                   stage_two_energy_weight=1000, stage_two_forces_weight=100, stage_two_stress_weight=10,
                   lr=.01, stage_two_lr=.001, ema_decay=.99, batch_size=r['batch_size'], valid_batch_size=r['valid_batch_size'],
                   scaling='rms_forces_scaling', scheduler_patience=5, patience=r['epochs'] + 1,
                   eval_interval=1, max_num_epochs=r['epochs'], start_stage_two=r['stage_two_start'])
    if 'molecular' in job:
        # 0.3.16 forwards explicitly mapped extra arrays to AtomicData. Use
        # the normal single Default head and retain the held-out test file.
        options['heads'] = repr({'Default': {'arrays_keys': {'mol_anchor': 'mol_anchor', 'mol_arm': 'mol_arm'},
                                             'test_file': str(options['test_file'])}})
    if job['resume']:
        options.update(avg_num_neighbors=job['avg_num_neighbors'], compute_avg_num_neighbors='False')
    args = [item for name, value in options.items() for item in ('--' + name, str(value))]
    # At every epoch (including a non-improving boundary), 0.3.16 saves the model
    # within ema.average_parameters(). Both paired runs select the final stage-two EMA.
    args += ['--pair_repulsion', '--amsgrad', '--ema', '--stage_two', '--save_cpu', '--save_all_checkpoints']
    if job['resume']:
        args.append('--restart_latest')
    return args


def plan_job(args):
    sources, offsets, teacher, scales = _inspect_sources(args)
    recipe = {key: getattr(args, key) for key in ('width', 'radial_head', 'cutoff', 'seed', 'dtype', 'epochs',
                                                    'stage_two_start', 'batch_size', 'valid_batch_size')}
    job = dict(schema=1, mace_torch=MACE_VERSION, recipe=recipe, out=str(args.out.resolve()), sources=sources,
               teacher_sha256=teacher[0], teacher_head=teacher[1], energy_offsets_eV=offsets,
               centered_before_cast=args.dtype == 'float32', resume=None,
               radial_input=dict(basis='bessel', num_basis=8, cutoff_polynomial_order=5, species_dependent=False,
                                 distance_transform='None', apply_cutoff_before_mlp=True, mlp_widths=[64, 64, args.radial_head]))
    if scales is not None:
        recipe['molecular_loss'] = args.molecular_loss
        job['molecular'] = scales
    identities = [(s['role'], s['sha256']) for s in sources]
    if args.resume:
        parent = json.loads((args.resume / 'manifest.json').read_text(encoding='utf-8'))
        if parent['schema'] != job['schema'] or parent['mace_torch'] != MACE_VERSION or parent['recipe'] != recipe:
            raise ValueError('resume requires the parent architecture, dtype, seed and schedule')
        previous = [(s['role'], s['sha256']) for s in parent['sources']]
        if previous[:3] != identities[:3] or not set(previous[3:]) < set(identities[3:]):
            raise ValueError('resume requires identical base splits and cumulative augmentation including new frames')
        checkpoint = (args.resume / 'pre_stage_two.pt').resolve()
        digest = file_digest(checkpoint)
        if digest != parent['pre_stage_two_sha256']:
            raise ValueError('resume checkpoint digest mismatch')
        job.update(energy_offsets_eV=parent['energy_offsets_eV'], avg_num_neighbors=parent['avg_num_neighbors'],
                   resume=dict(checkpoint=str(checkpoint), sha256=digest))
    job['mace_args'] = _arguments(job)
    # Order-sensitive data and schedule identity for pairing jobs that differ only in precision.
    job['pair_id'] = hashlib.sha256(json.dumps([MACE_VERSION, identities, {k: v for k, v in recipe.items() if k != 'dtype'}],
                                               sort_keys=True).encode()).hexdigest()
    return job


def write_inputs(job, dest):
    dest.mkdir(parents=True, exist_ok=False)
    offsets = {int(z): e for z, e in job['energy_offsets_eV'].items()}
    for source in job['sources']:
        role = source['role']
        if file_digest(source['path']) != source['sha256']:
            raise ValueError(f"changed split since planning: {source['path']}")
        target = dest / (('train' if role == 'harvest' else role) + '.xyz')
        with target.open('a', encoding='utf-8') as stream:
            for a in iread(source['path'], index=':', format='extxyz'):
                # Strip calculator results, constraints and MACE-specific per-frame loss
                # overrides. All teacher labels have equal weight in this recipe.
                energy = float(a.info['REF_energy'])
                if job['centered_before_cast']:
                    z, counts = np.unique(a.numbers, return_counts=True)
                    energy -= sum(int(n) * offsets[int(element)] for element, n in zip(z, counts))
                keep = ('frame_id', 'trajectory_id', 'split', 'sampler', 'teacher_sha256', 'teacher_head', 'REF_stress')
                a.info = {k: a.info[k] for k in keep}
                a.info['REF_energy'] = energy
                a.set_constraint()
                columns = ['symbols', 'positions', 'REF_forces']
                if 'molecular' in job:
                    a.arrays['mol_anchor'], a.arrays['mol_arm'] = _molecular().molecular_geometry(a, 'mol_id')
                    columns += ['mol_id', 'mol_anchor', 'mol_arm']
                write(stream, a, format='extxyz', columns=columns, write_results=False)


def restart_paths(checkpoints, job):
    r = job['recipe']
    prefix = f"student_run-{r['seed']}_epoch-"
    boundary = r['stage_two_start']
    return checkpoints / f'{prefix}{boundary - 1}.pt', checkpoints / f'{prefix}{boundary}_swa.pt'


def _seed_resume(job, checkpoints):
    import torch

    checkpoints.mkdir(parents=True)
    pre, restart = restart_paths(checkpoints, job)
    shutil.copyfile(job['resume']['checkpoint'], pre)
    # MACE reads the *next* epoch directly from the filename (no +1) and tries
    # _swa first. Leave the real phase-one checkpoint for its final two-table eval.
    state = torch.load(pre, map_location='cpu', weights_only=False)
    for group in state['optimizer']['param_groups']:
        # Native SWALR anneals in one step. On a restart MACE skips that step at
        # start_epoch, so give its first augmented minibatch the phase-two rate.
        group['lr'] = group['swa_lr']
    torch.save(state, restart)


def physical_e0(training_e0, atomic_numbers, offsets):
    """Restore single-head E0s without rounding a large total back to float32."""
    if training_e0.size != len(atomic_numbers) or np.any(training_e0):
        raise ValueError('centered student must have one head with zero training E0s')
    return np.array([offsets[str(z)] for z in atomic_numbers], dtype=np.float64).reshape(training_e0.shape)


def _export_physical(job, out):
    import torch

    model = torch.load(out / 'work/student_stagetwo.model', map_location='cpu', weights_only=False).double()
    if job['centered_before_cast']:
        zs = model.atomic_numbers.tolist()
        e0 = model.atomic_energies_fn.atomic_energies
        restored = physical_e0(e0.detach().numpy(), zs, job['energy_offsets_eV'])
        with torch.no_grad():
            e0.copy_(torch.from_numpy(restored))
    torch.save(model, out / 'student.model')
    return float(model.interactions[0].avg_num_neighbors)


def run_job(job):
    if version('mace-torch') != MACE_VERSION:
        raise RuntimeError(f'training requires mace-torch=={MACE_VERSION}')
    out = Path(job['out'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'plan.json').write_text(json.dumps(job, indent=2) + '\n', encoding='utf-8')
    write_inputs(job, out / 'inputs')
    # MACE/e3nn 0.4.x unpickle training models/constants in both processes.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    checkpoints = out / 'work/checkpoints'
    if job['resume']:
        _seed_resume(job, checkpoints)
    with (out / 'train.log').open('w', encoding='utf-8') as log:
        if job['recipe'].get('molecular_loss') == 'net-torque':
            command = [sys.executable, str(Path(__file__).resolve().with_name('molecular_loss.py')), str(out/'plan.json')]
        else:
            command = [sys.executable, '-m', 'mace.cli.run_train', *job['mace_args']]
        subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
    pre, _ = restart_paths(checkpoints, job)
    r = job['recipe']
    if not (checkpoints / f"student_run-{r['seed']}_epoch-{r['epochs'] - 1}_swa.pt").is_file():
        raise RuntimeError('training failed to complete the requested second phase')
    if job['resume'] and file_digest(pre) != job['resume']['sha256']:
        raise RuntimeError('MACE retrained phase one instead of resuming the requested boundary')
    avg = _export_physical(job, out)
    shutil.copyfile(pre, out / 'pre_stage_two.pt')
    job.update(avg_num_neighbors=avg, pre_stage_two_sha256=file_digest(out / 'pre_stage_two.pt'),
               student_sha256=file_digest(out / 'student.model'), pre_stage_two_epoch=r['stage_two_start'] - 1,
               student_epoch=r['epochs'] - 1)
    # Save-all is needed to get the exact boundary, not 150 retained optimizer copies.
    for path in checkpoints.glob(f"student_run-{r['seed']}_epoch-*.pt"):
        path.unlink()
    (out / 'manifest.json').write_text(json.dumps(job, indent=2) + '\n', encoding='utf-8')


def main(argv=None):
    args = parse_args(argv)
    job = plan_job(args)
    if args.plan:
        print(json.dumps(job, indent=2))
    else:
        run_job(job)


if __name__ == '__main__':
    main()
