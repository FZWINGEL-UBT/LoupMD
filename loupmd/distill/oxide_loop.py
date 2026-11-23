"""Corrected SrTiO3 closed loop. Run in WSL with the repository runtime Python.

First, in Windows Git Bash: bash tools/dev/oxide_loop_freeze.sh
Then, from this checkout in WSL:
python -m loupmd.distill.oxide_loop --fp32-rehearsal prepare 0
python -m loupmd.distill.oxide_loop prepare 0    # publishes round-1 launch.sh, never allocates
python -m loupmd.distill.oxide_loop evaluate 1   # after the authorized launch finishes
python -m loupmd.distill.oxide_loop prepare 1    # then evaluate 2; only if open prepare 2/evaluate 3

Each GPU command takes a separate foreground FIFO reservation. Existing immutable
shards allow continuation after a queue interruption. Round 0 consumes, never
regenerates, the baseline owner's release and independent fixed evaluation sets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

from ase.io import iread
import numpy as np

from . import fidelity, harvest, train

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / 'tools/dev/oxide_loop_protocol.json'
BASE = {'train': (300, 'cfcb2131082b98746dac1ad661b31497de9f6ce50b8616a83c04457eb1773bc4'),
        'valid': (30, 'b01df402109c60094338e078e10af205e54997821f2f0c286d6f62499804e619'),
        'teacher-test': (60, 'e015fe99ec49e6f093ef93f791963302e865391788ab2fffa096da103ed2a7e6')}
RECIPE = dict(width=64, radial_head=64, cutoff=5.0, seed=1, epochs=150, stage_two_start=111,
              batch_size=4, valid_batch_size=8)
LIB_SHA = 'd12e88dee68635570258ef678111637bd859966d942cae25627a09c3992ba048'
BUILD = '0.0.2+27d3ab256b7d8126'
TEACHER_SHA = 'a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47'


def load(path):
    return json.loads(path.read_bytes())


def pin(path, data):
    """Idempotent immutable publication; no partial payload is named as complete."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'changed immutable artifact: {path}')
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(data)
            f.close()
            os.link(tmp, path)
        finally:
            tmp.unlink()


def record(path, data):
    pin(path, (json.dumps(data, indent=2, allow_nan=False) + '\n').encode())


def schedule(parent_round, role):
    p = load(PROTOCOL)
    common = {k: v for k, v in p['trajectory'].items() if k != 'source'}
    if role == 'fixed-student-test':
        return {**common, **p[role]}
    h = p['harvest']
    return dict(common, seed=h['seeds_by_child_round'][parent_round], hot_K=h['hot_K_by_child_round'][parent_round],
                **{k: h[k] for k in ('replicas', 'target_frames', 'hot_frames')})


def decision(r, exceeds):
    if r < 2:
        return 'not_evaluated'
    return ('third_hot_round' if r == 2 else 'stop_open') if exceeds else 'closed'


def verify_parent(job, dtype, augments):
    m = load(job / 'manifest.json')
    if (m['schema'] != 1 or m['mace_torch'] != '0.3.16' or m['recipe'] != dict(RECIPE, dtype=dtype)
            or (m['student_epoch'], m['pre_stage_two_epoch']) != (149, 110)):
        raise ValueError(f'{job}: wrong parent recipe, backend or epoch')
    expected = [(k, n, sha) for k, (n, sha) in BASE.items()]
    expected += [('harvest', load(PROTOCOL)['harvest']['select'], train.file_digest(p)) for p in augments]
    if [(s['role'], s['frames'], s['sha256']) for s in m['sources']] != expected:
        raise ValueError(f'{job}: parent sources differ from corrected release and frozen cumulative harvest')
    if bool(m['resume']) != bool(augments) or (augments and m['resume']['sha256'] != m['pre_stage_two_sha256']):
        raise ValueError('unexpected phase-one restart lineage')
    for name, key in [('student.model', 'student_sha256'), ('pre_stage_two.pt', 'pre_stage_two_sha256')]:
        if train.file_digest(job / name) != m[key]:
            raise ValueError(f'{job}/{name}: digest mismatch')
    return m


def geometry(a):
    h = hashlib.sha256(a.numbers.astype('<i4').tobytes())
    h.update(np.asarray(a.cell, dtype='<f8').tobytes())
    h.update(np.asarray(a.positions, dtype='<f8').tobytes())
    return h.digest()


def audit_streams(streams):
    """Frame, RNG and geometry exclusion over ALL candidates (not just selected frames).

    Models intentionally share only the fixed test initial-condition streams. They
    may not share frames or geometries, even with renamed IDs. No test may overlap
    train/valid/teacher-test/harvest, including in a later round.
    """
    ids, geoms, seeds, trajectories, counts = set(), set(), {}, {}, {}
    for role, frames in streams:
        role_seeds = {}
        counts[role] = 0
        for a in frames:
            counts[role] += 1
            uid, tid, seed = (a.info[k] for k in ('frame_id', 'trajectory_id', 'md_seed'))
            if uid in ids:
                raise ValueError(f'{role}: shared frame_id {uid}')
            ids.add(uid)
            g = geometry(a)
            if g in geoms:
                raise ValueError(f'{role}: shared geometry {uid}')
            geoms.add(g)
            if trajectories.setdefault(tid, (role, seed)) != (role, seed):
                raise ValueError(f'{role}: shared trajectory_id {tid}')
            # Pair only the initial RNG stream between distinct rounds of own tests.
            if role_seeds.setdefault(seed, tid) != tid:
                raise ValueError(f'{role}: shared md_seed {seed}')
            other, traj = seeds.setdefault(seed, (role, tid))
            if (role, tid) != (other, traj) and not (role != other and role.startswith('own:') and other.startswith('own:')):
                raise ValueError(f'{role}: shared md_seed {seed}')
    return counts


def xyz(path):
    return iread(path, index=':', format='extxyz')


def labels_match(scored, reference, model_sha):
    sentinel = object()
    from itertools import zip_longest
    for a, b in zip_longest(xyz(scored), xyz(reference), fillvalue=sentinel):
        if (a is sentinel or b is sentinel or a.info['frame_id'] != b.info['frame_id'] or geometry(a) != geometry(b)
                or a.info['prediction_sha256'] != model_sha or a.info['teacher_sha256'] != TEACHER_SHA
                or a.info['REF_energy'] != b.info['REF_energy']
                or not np.array_equal(a.info['REF_stress'], b.info['REF_stress'])
                # Reissue preserves binary64 REF, whereas ASE writes arrays to eight decimals on rescore.
                or not np.array_equal(a.arrays['REF_forces'], np.char.mod('%.8f', b.arrays['REF_forces']).astype(float))):
            raise ValueError(f'{scored}: not a rescore of the frozen teacher-test')


def check_sample(path, role, r, model_sha, replicas):
    p = schedule(r, role)
    expected = {(i, regime, p['burn_in'] + j * p['stride']) for i in replicas for regime in ('target', 'hot')
                for j in range(1, p[regime + '_frames'] + 1)}
    tids = {}
    for a in xyz(path):
        m = a.info
        ident = (m['replica'], m['regime'], m['step'])
        if ident not in expected:
            raise ValueError(f'{path}: duplicate or unexpected sample {ident}')
        expected.remove(ident)
        checks = dict(split=role, sampler='student', teacher_sha256=TEACHER_SHA, teacher_head='omat_pbe',
                      prediction_sha256=model_sha, sampler_sha256=model_sha, engine_build=BUILD,
                      integrator='langevin', source='SrTiO3-4x4x4', temperature_K=p[m['regime'] + '_K'],
                      **{k:p[k] for k in ('seed', 'stride', 'burn_in', 'timestep_fs', 'friction_per_fs', 'strain', 'shear', 'rattle_A')})
        if (any(m.get(k) != v for k, v in checks.items()) or len(a) != 320
                or tids.setdefault(ident[:2], m['trajectory_id']) != m['trajectory_id']):
            raise ValueError(f'{path}: unexpected sampler or protocol for {ident}')
    if expected or len(set(tids.values())) != len(tids):
        raise ValueError(f'{path}: incomplete or shared trajectories')


def verify_upload(run):
    inputs = (run.parent / 'paired-inputs.sha256').read_text().splitlines()
    expected = {Path(name).name: sha for sha, name in (line.split('  ', 1) for line in inputs)}
    if len(expected) != len(inputs) or load(run / 'allocation.json')['input_sha256'] != expected:
        raise ValueError(f'{run}: uploaded payload differs from the frozen paired inputs')


def collect(run):
    """Only unpack regular known members of the digest-checked authorized child result."""
    out = run / 'checked'
    if out.exists():
        return
    allocation = load(run / 'allocation.json')
    archive = run / 'result.tar.gz'
    if archive.exists():
        sha = allocation['result_sha256']
        if allocation['exit_code'] != 0:
            raise ValueError('child training failed')
    else:
        archive = run / 'recovery/final.tar.gz'
        sha = archive.with_suffix('').with_suffix('.sha256').read_text().split()[0]
    if train.file_digest(archive) != sha:
        raise ValueError(f'{archive}: transport SHA mismatch')
    with tempfile.TemporaryDirectory(dir=run) as tmp, tarfile.open(archive, 'r:gz') as tar:
        checked = Path(tmp) / 'checked'
        checked.mkdir()
        for name in ('student.model', 'pre_stage_two.pt', 'manifest.json', 'train.log', 'job.log', 'plan.json'):
            member = tar.getmember('out/' + name)
            if not member.isfile():
                raise ValueError(f'{archive}: not a regular {member.name}')
            with tar.extractfile(member) as src, (checked / name).open('xb') as dst:
                shutil.copyfileobj(src, dst)
        checked.rename(out)


def check_stored(g, sha, ids):
    # qualify_frozen_cells' top-level flag checks NEW native labels, not these stored bytes.
    if (not g['passed'] or g['teacher_sha256'] != TEACHER_SHA or g['teacher_head'] != 'omat_pbe'
            or g['library_sha256'] != LIB_SHA or g['source_sha256'] != sha or g['frames'] != len(ids)
            or [x['frame_id'] for x in g['rows']] != ids or not all(x['historical']['passed'] for x in g['rows'])):
        raise ValueError('stored selected training labels did not pass their independent 3x gate')


def gpu(label, *command):
    queue = '/mnt/d/Margraf/FastMACE/tools/ci/local_gpu.py'
    subprocess.run(['python3', queue, 'status'], check=True)
    print(f'Reserving {label} (one foreground job; no timing)', flush=True)
    subprocess.run(['python3', queue, 'run', '--purpose', f'Stage 0 oxide loop {label}', '--wait-minutes', '240',
                    '--', sys.executable, *map(str, command)], check=True)


def sample(path, role, r, job, sha, replica):
    if not path.exists():
        options = schedule(r, role)
        options['first_replica'] = replica
        options['replicas'] = 1
        gpu(f'{role} round {r} replica {replica}', '-m', 'loupmd.distill.frames', '--split', role,
            '--teacher', 'mh1', '--teacher-head',
            'omat_pbe', '--student', job / 'student.model', '--student-sha256', sha, '--output', path,
            *(v for k, x in options.items() for v in ('--' + k.replace('_', '-'), x)))
    check_sample(path, role, r, sha, [replica])


def combine(paths, out):
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        merged = Path(tmp) / 'all.xyz'
        with merged.open('wb') as stream:
            for path in paths:
                with path.open('rb') as source:
                    shutil.copyfileobj(source, stream)
        if out.exists():
            if train.file_digest(out) != train.file_digest(merged):
                raise ValueError(f'{out}: changed merged shards')
        else:
            merged.rename(out)


def check_gate(run, sha):
    path = run / 'release'
    g = load(path / 'gate.json')
    if not (g['passed'] and g['model']['sha256'] == sha and g['library'] == BUILD and len(g['rows']) == 2
            and all(row['passed'] for row in g['rows']) and g['reference']['mace'] == train.MACE_VERSION
            and train.file_digest(path/'student.loup') == (path/'student.loup.sha256').read_text().split()[0]):
        raise ValueError(f'{path}: missing or mismatched student export gate')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['evaluate', 'prepare'])
    p.add_argument('round', type=int, choices=range(4), help='completed training rounds for the model to inspect')
    p.add_argument('--campaign', type=Path, default=Path.home() / 'loup-s-stage0/sto320')
    p.add_argument('--fp32-rehearsal', action='store_true', help='plan only; no child launch command')
    args = p.parse_args(argv)
    r = args.round
    if (args.fp32_rehearsal and r != 0) or (r == 3 and args.action == 'prepare'):
        p.error('only round-0 preparation for FP32; no fourth round')
    c = args.campaign.resolve()
    dtype = 'float32' if args.fp32_rehearsal else 'float64'
    baseline = c / ('v2-l4-s64l0-fp32-seed1-r2' if args.fp32_rehearsal else 'v2-l4-s64l0-seed1')
    root = c / 'oxide-loop-v1' / ('fp32-rehearsal' if args.fp32_rehearsal else 'fp64')
    work = root / f'round-{r}'
    work.mkdir(parents=True, exist_ok=True)
    pin(root / 'protocol.json', PROTOCOL.read_bytes())
    lib = REPO / 'build/sm_120/libloupmd.so'
    if train.file_digest(lib) != LIB_SHA:
        raise ValueError(f'{lib}: not the qualified cell-change build')
    os.environ.update(LOUP_LIB=str(lib), PYTHONPATH=str(REPO), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
                      LOUPMD_MODELS=str(Path.home() / 'models'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    base = c / 'base-labels-v2'
    for k, (_, sha) in BASE.items():
        if train.file_digest(base / (k + '.xyz')) != sha:
            raise ValueError(f'{base}/{k}: not the frozen v2 split')
    def run_at(i):
        return root / f'round-{i}/training' if i else baseline
    run = run_at(r)
    selected = [root / f'round-{i}/selected-{i+1}.xyz' for i in range(r)]
    for i, choice in enumerate(selected):
        receipt = load(root / f'round-{i+1}/preparation.json')
        previous = receipt['parent_evaluation']
        if (previous['protocol_sha256'] != train.file_digest(PROTOCOL)
                or previous['own_test_sha256'] != train.file_digest(run_at(i) / 'fixed-student-test.xyz')
                or previous['teacher_test_sha256'] != train.file_digest(run_at(i) / 'teacher-scored.xyz')
                or previous['model_sha256'] != train.file_digest(run_at(i) / 'checked/student.model')
                or receipt['harvest_sha256'] != train.file_digest(root / f'round-{i}/harvest.xyz')
                or receipt['selected_sha256'] != train.file_digest(choice)):
            raise ValueError(f'round {i}: historical model, harvest or test bytes no longer match receipt')
    if r:
        verify_upload(run)
        collect(run)
    job = run / 'checked'
    m = verify_parent(job, dtype, selected)
    sha = m['student_sha256']
    if r and m['pre_stage_two_sha256'] != load(baseline / 'checked/manifest.json')['pre_stage_two_sha256']:
        raise ValueError('child did not preserve the baseline epoch-110 EMA')
    if r and not (run / 'release').exists():
        gpu(f'gate round {r}', '-m', 'loupmd.distill.publish', job, base/'teacher-test.xyz', run/'release',
            '--ref-device', 'cuda')
    check_gate(run, sha)
    own, scored = run / 'fixed-student-test.xyz', run / 'teacher-scored.xyz'
    if r:
        if not scored.exists():
            gpu(f'teacher-test rescore round {r}', '-m', 'loupmd.distill.frames',
                '--rescore', base/'teacher-test.xyz', '--output', scored,
                '--student', job/'student.model', '--student-sha256', sha)
        parts = [work / f'own-{i}.xyz' for i in range(2)]
        for i, path in enumerate(parts):
            sample(path, 'fixed-student-test', r, job, sha, i)
        combine(parts, own)
    check_sample(own, 'fixed-student-test', r, sha, range(2))
    labels_match(scored, base / 'teacher-test.xyz', sha)
    teacher = fidelity.score_frames(xyz(scored), regime='target')
    student = fidelity.score_frames(xyz(own), regime='target')
    comparison = fidelity.compare_round(student, teacher, r)
    outcome = decision(r, comparison['exceeds_1_5x'])
    streams = [(k, base / (k + '.xyz')) for k in BASE]
    streams += [(f'own:{i}', run_at(i) / 'fixed-student-test.xyz') for i in range(r+1)]
    streams += [(f'harvest:{i}', root / f'round-{i}/harvest.xyz') for i in range(r)]
    def audit():
        return audit_streams([(role, xyz(path)) for role, path in streams])
    inventory = audit()
    if r:
        record(run / 'teacher-score.json', teacher)
        record(run / 'fidelity.json', dict(student, round_comparison=comparison))
    else:
        if load(run/'fidelity.json')['round_comparison']['student_to_teacher_rmse_ratio'] != comparison['student_to_teacher_rmse_ratio']:
            raise ValueError('round-0 comparison does not reproduce the baseline owner record')
    result = dict(protocol_sha256=train.file_digest(PROTOCOL), model_sha256=sha, checkpoint=m['pre_stage_two_sha256'],
                  teacher_test_sha256=train.file_digest(scored), own_test_sha256=train.file_digest(own),
                  checked_counts=inventory, comparison=comparison, teacher_label_bar=teacher['label_metrics_passed'],
                  own_label_bar=student['student_visited_label_bar'], falsifier_1=outcome)
    record(work / 'evaluation.json', result)
    print(json.dumps(result), flush=True)
    if args.action == 'evaluate':
        return
    if r == 2 and outcome != 'third_hot_round':
        raise ValueError('falsifier closed after two rounds; do not train a third')
    parts = [work / f'harvest-{i}.xyz' for i in range(schedule(r, 'harvest')['replicas'])]
    for i, path in enumerate(parts):
        sample(path, 'harvest', r, job, sha, i)
    candidates = work / 'harvest.xyz'
    combine(parts, candidates)
    streams.append((f'harvest:{r}', candidates))
    inventory = audit()
    choice = work / f'selected-{r+1}.xyz'
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        new = Path(tmp) / choice.name
        harvest.select(candidates, new, load(PROTOCOL)['harvest']['select'])
        combine([new], choice)
    chosen_ids = [a.info['frame_id'] for a in xyz(choice)]
    label_gate = work / f'selected-{r+1}-gate.json'
    if not label_gate.exists():
        gpu(f'stored harvest labels round {r}', REPO/'tools/dev/qualify_frozen_cells.py', choice, label_gate)
    choice_sha = train.file_digest(choice)
    check_stored(load(label_gate), choice_sha, chosen_ids)
    selected.append(choice)
    target = root / f'round-{r+1}'
    target.mkdir(exist_ok=True)
    recipe = dict(RECIPE, dtype=dtype)
    opts = [str(v) for k, x in recipe.items() for v in ('--' + k.replace('_', '-'), x)]
    files = [base / (k + '.xyz') for k in BASE]
    local = [v for k, path in zip(BASE, files) for v in ('--'+k, str(path))]
    def restarts(parent, aug):
        return ['--resume', str(parent), *(str(v) for x in aug for v in ('--augment', x))]
    stock = root.parent / 'stock-trainer'
    stock_files = [stock/'colab_train.sh', stock/'train.py']
    for path in stock_files:
        if train.file_digest(path) != load(baseline/'allocation.json')['input_sha256'][path.name]:
            raise ValueError(f'{path}: not the payload used by the corrected baseline allocation')
    cmd = [sys.executable, stock/'train.py', *opts, *local, '--out', str(target/'planned-job'),
           *restarts(job, selected), '--plan']
    planned = json.loads(subprocess.check_output(cmd))
    record(target / 'resume-plan.json', planned)
    arguments = target / 'train.args'
    pin(arguments, ('\n'.join(opts + restarts('.', [f.name for f in selected])) + '\n').encode())
    payload = [*stock_files, arguments, *files, job/'manifest.json', job/'pre_stage_two.pt', *selected]
    sums = ''.join(f'{train.file_digest(path)}  {path}\n' for path in payload)
    pin(target / 'paired-inputs.sha256', sums.encode())
    record(target / 'preparation.json', dict(parent_evaluation=result, inventory=inventory,
           harvest_sha256=train.file_digest(candidates), selected_sha256=choice_sha,
           selected_frame_ids=chosen_ids, pair_id=planned['pair_id']))
    if not args.fp32_rehearsal:
        command = [sys.executable, REPO/'tools/dev/colab_distill.py', '--gpu', 'L4', '--account', '1',
                   '--output', target/'training', *payload]
        pin(target/'launch.sh', ('#!/usr/bin/env bash\nset -euo pipefail\nexport PYTHONPATH='+shlex.quote(str(REPO))
            +'\nsha256sum -c '+shlex.quote(str(target/'paired-inputs.sha256'))+'\nexec '+shlex.join(map(str, command))+'\n').encode())
    print(f'Prepared {target}; ' + ('FP32 rehearsal only' if args.fp32_rehearsal else 'allocator: bash launch.sh'), flush=True)


if __name__ == '__main__':
    main()
