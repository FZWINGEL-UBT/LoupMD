"""Recreate the Stage 0 flexible-water experiment (216 H2O, OFF24-medium).

From the repository root, W=$HOME/loup-s-stage0/water, python from the pinned
mace-torch==0.3.16 environment. Use the corrected LOUP_LIB and pinned teacher.
Run EACH sample/rollout/md/rescore under a separate foreground local_gpu.py run.
No reservation is needed for label assignment, planning, assembly or reduction.

  mkdir -p "$W"
  python -m loupmd.distill.water label loupmd/bench/structures/water_216.extxyz "$W/initial.xyz"
  python -m loupmd.distill.water_protocol "$W" sample PART
  # PART in train-0, train-1, train-2, train-3, valid, teacher-test (one reservation each).
  # On the permitted CPU host, for EACH unmodified shard copied there:
  CUDA_VISIBLE_DEVICES='' python -m loupmd.distill.water_gate PART.xyz MACE-OFF24_medium.model PART-gate.json
  # Return the reports to W, then on the data host:
  python -m loupmd.distill.water_protocol "$W" assemble "$W/dispatch"
  # dispatch/ includes the standalone trainer, original pinned colab_train.sh, both .args
  # and all splits. Verify inputs.sha256, then use the account-1 dispatcher.
  # Without allocating anything: tools/dev/water_trial.sh plan control
  # Each cold start consumes its own sealed .args: control or net-torque.

For each trained model after independent `loupmd.distill.publish` (JOB/checked
and JOB/release), `rollout JOB target` and `rollout JOB hard` generate separate
60-frame sets. Freeze the control's files for cross-model scoring and the trigger;
the molecular student ALSO needs its own rollout for §4. Use `frames --rescore`
with the returned checkpoint/SHA on valid and teacher-test; compare the joint
frames with `fidelity --molecule-key mol_id --regime target` (hot hard frames:
`--regime hot`, not eligible for the own target-condition bar). Gate each new
student-visited file with water_gate before interpreting labels; never fit on it.

  python -m loupmd.distill.water_protocol "$W" start-md
  python -m loupmd.distill.water_protocol "$W" md CASE [--job JOB]
  python -m loupmd.distill.water_protocol "$W" reduce [--job JOB]

Each of nve-production, nve-conserving, npt-1, npt-2 is a separate MD reservation.
No --job: teacher into W/observables. --job: matched student into JOB/observables.
Two finite teacher seeds give descriptive spreads, not converged tolerances.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from ase.geometry import find_mic
from ase.io import iread, read, write

from .train import file_digest, parse_args as train_args, plan_job
from .water import inspect_water
from .water_gate import TEACHER

SOURCE_SHA = 'fb1dec83bfff0568a11d174c7785465be5cdff4361d96de7e314c07e331acc3a'
INITIAL_SHA = '348b11bad5df7b57317193d0ae00d7f969d67bc62f5d18937981acbc034f5ee2'
COLAB_SHA = '3d1edd3299bd27f169dedf4ee8b4a0078edf25f69fd76afa0e8393d970a41caf'
PARTS = {**{f'train-{i}': ('train', 91001, i, 50, 25) for i in range(4)},
         'valid': ('valid', 92001, 0, 20, 10), 'teacher-test': ('teacher-test', 93001, 0, 40, 20)}
MD = {'nve-production': ('nve', 96001, 1., 10, 5000), 'nve-conserving': ('nve', 96001, .5, 20, 10000),
      'npt-1': ('npt', 96101, .5, 40, 40000), 'npt-2': ('npt', 96102, .5, 40, 40000)}


def _md_sampling(root):
    return ['--teacher', 'off24-medium', '--input', str(root/'initial.xyz'), '--timestep-fs', '0.5',
            '--burn-in', '10000', '--stride', '400', '--rattle-A', '0.015', '--target-K', '300', '--hot-K', '450']


def sample_args(root, part):
    role, seed, replica, target, hot = PARTS[part]
    # frames.py applies train-only isotropic strain and symmetric shear; pin them explicitly.
    strain = '0.01' if role == 'train' else '0.0'
    return [*_md_sampling(root), '--split', role, '--seed', str(seed), '--first-replica', str(replica),
            '--strain', strain, '--shear', strain, '--target-frames', str(target), '--hot-frames', str(hot),
            '--output', str(root/f'{part}.xyz')]


def rollout_args(root, model, sha, role, output):
    hard = role == 'hard'
    return [*_md_sampling(root), '--student', str(model), '--student-sha256', sha, '--split', 'fixed-student-test',
            '--seed', str(94102 if hard else 94101), '--target-frames', '0' if hard else '60',
            '--hot-frames', '60' if hard else '0', '--strain', '.01' if hard else '0',
            '--shear', '.01' if hard else '0', '--output', str(output)]


def checked_student(job):
    model = job/'checked/student.model'
    manifest = json.loads((job/'checked/manifest.json').read_text())
    gate = json.loads((job/'release/gate.json').read_text())
    sha = file_digest(model)
    if not (manifest['teacher_sha256'] == TEACHER and manifest['student_sha256'] == sha == gate['model']['sha256']
            and gate['passed'] and gate['rows'] and all(r['passed'] for r in gate['rows']) and
            file_digest(job/'release/student.loup') == (job/'release/student.loup.sha256').read_text().split()[0]):
        raise ValueError('student was not independently gate-qualified for this water teacher')
    return model, sha


def _training_files(stage):
    """Single source for the paired standalone upload and both job manifests."""
    distill = Path(__file__).resolve().parent
    wrapper = distill.parents[1]/'tools/dev/colab_train.sh'
    if file_digest(wrapper) != COLAB_SHA:
        raise ValueError('pinned colab_train.sh SHA-256 mismatch')
    for name in ('train.py', 'molecular.py', 'molecular_loss.py'):
        shutil.copyfile(distill/name, stage/name)
    shutil.copyfile(wrapper, stage/'colab_train.sh')
    shutil.copyfile(distill/'water_train.sh', stage/'launch.sh')
    inputs = [arg for role in ('train', 'valid', 'teacher-test') for arg in ('--'+role, str(stage/f'{role}.xyz'))]
    arms = {}
    for mode in ('control', 'net-torque'):
        recipe = ['--width','64','--radial-head','64','--cutoff','5.0','--seed','1','--dtype','float64',
                  '--epochs','150','--stage-two-start','111','--batch-size','2','--valid-batch-size','2',
                  '--molecular-loss', mode]
        (stage/f'{mode}.args').write_text('\n'.join(recipe)+'\n', encoding='utf-8')
        job = plan_job(train_args([*inputs, '--out', str(stage/mode), *recipe]))
        arms[mode] = dict(recipe=job['recipe'], molecular=job['molecular'], pair_id=job['pair_id'],
                          offsets=job['energy_offsets_eV'])
    hashes = {p.name:file_digest(p) for p in sorted(stage.iterdir()) if p.is_file()}
    (stage/'inputs.sha256').write_text(''.join(f'{digest}  {name}\n' for name,digest in hashes.items()))
    return hashes, arms


def assemble(root, output):
    """Publish only all six verified shards; no reads of validation for loss normalization."""
    if output.exists():
        raise FileExistsError(output)
    base = read(root/'initial.xyz')
    if file_digest(root/'initial.xyz') != INITIAL_SHA or inspect_water(base)['molecules'] != 216:
        raise ValueError('unexpected starting geometry/topology; use water label on the pinned input')
    src = Path(__file__).resolve().parents[1]/'bench/structures/water_216.extxyz'
    if file_digest(src) != SOURCE_SHA:
        raise ValueError('repository water_216.extxyz SHA-256 mismatch')
    reports, seeds, conformations = {}, {}, {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/'dataset'
        stage.mkdir()
        for part, (role, seed, replica, target, hot) in PARTS.items():
            source, receipt = root/f'{part}.xyz', root/f'{part}-gate.json'
            g = json.loads(receipt.read_text())
            digest = file_digest(source)
            if not (g['schema'] == 'loupmd.water-stored-label-gate.v1' and g['teacher_sha256'] == TEACHER and
                    g['passed'] and g['source_sha256'] == digest and g['frames'] == target + hot == len(g['rows'])):
                raise ValueError(f'{part}: missing or invalid stored-label gate')
            for i, (a, row) in enumerate(zip(iread(source, ':'), g['rows'], strict=True)):
                meta = a.info
                regime = 'target' if i < target else 'hot'
                expected = dict(split=role, seed=seed, replica=replica, timestep_fs=.5, regime=regime,
                                integrator='langevin', burn_in=10000, stride=400, friction_per_fs=.01,
                                step=10000+400*((i if regime=='target' else i-target)+1),
                                temperature_K=300. if regime == 'target' else 450.,
                                strain=.01 if role=='train' else 0., shear=.01 if role=='train' else 0., rattle_A=.015,
                                teacher_sha256=TEACHER, sampler_sha256=TEACHER, teacher_head='default', sampler='teacher')
                if (not row['passed'] or row['config'] != meta['frame_id'] or any(meta.get(k)!=v for k,v in expected.items())
                        or seeds.setdefault(meta['md_seed'], (part, regime, meta['trajectory_id'])) != (part, regime, meta['trajectory_id'])
                        or not np.array_equal(a.numbers, base.numbers) or not np.array_equal(a.arrays['mol_id'], base.arrays['mol_id'])):
                    raise ValueError(f'{part}: unexpected frame, membership, random stream or provenance')
                oxy = a.numbers == 8
                delta = a.positions[oxy] - base.positions[oxy] @ np.linalg.solve(base.cell.array,a.cell.array)
                _, dist = find_mic(delta, a.cell)
                # Six oxygen sites per source-grid axis; the undisturbed input has unit peaks.
                grid = np.exp(12j*np.pi*a.get_scaled_positions()[oxy]).mean(axis=0)
                conformations.setdefault(role, []).append(dict(
                    frame_id=row['config'], trajectory_id=meta['trajectory_id'], md_seed=int(meta['md_seed']),
                    regime=regime, kinetic_K=a.get_temperature(), oxygen_grid_peak_max=float(np.max(abs(grid)**2)),
                    oxygen_displacement_rms_A=float(np.sqrt(np.mean(dist**2))), **inspect_water(a)))
            reports[part] = dict(source_sha256=digest, gate_sha256=file_digest(receipt), frames=g['frames'],
                                worst_ratio={k:max(r[k] for r in g['rows']) for k in ('energy_ratio','force_ratio','stress_ratio')})
            with (stage/f'{role}.xyz').open('ab') as out, source.open('rb') as inp:
                shutil.copyfileobj(inp, out)
        if len(seeds) != 2*len(PARTS):
            raise ValueError('expected one independent stream per shard and temperature')
        hashes, arms = _training_files(stage)
        (stage/'conformations.json').write_text(json.dumps(conformations, indent=2, allow_nan=False)+'\n')
        record = dict(schema='loupmd.water-release.v1', initial_sha256=INITIAL_SHA, source_sha256=SOURCE_SHA,
                      teacher_sha256=TEACHER, sources=reports, inputs_sha256=hashes, md_seeds=len(seeds),
                      library_sha256=file_digest(os.environ['LOUP_LIB']),
                      conformations_sha256=file_digest(stage/'conformations.json'), **arms)
        (stage/'release.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        stage.rename(output)
    return record


def md_args(root, role, case, directory, model, sha=None):
    ensemble, seed, dt, stride, steps = MD[case]
    return ['md','--role',role,'--model',str(model),*(['--sha256',sha] if sha else []),
            '--input',str(root/'observables/start.xyz'),'--directory',str(directory/case),'--ensemble',ensemble,
            *(['--pressure-GPa','0.000101325'] if ensemble=='npt' else []),'--temperature-K','300','--seed',str(seed),
            '--timestep-fs',str(dt),'--stride',str(stride),'--steps',str(steps)]


def reduce(root, job):
    from .observables import main as observables
    teacher = root/'observables'
    dirs = [teacher] if job is None else [teacher, job/'observables']
    def endpoints(case):
        return [str(d/case/f'segment-{MD[case][-1]}.json') for d in dirs]
    observables(['nve','--production',*endpoints('nve-production'),'--conserving',*endpoints('nve-conserving'),
                 '--blocks','5','--output',str(dirs[-1]/'nve-5ps.json')])
    for block, lags in [(200,[25,50,75,100]), (400,[50,100,150,200])]:
        runs = [arg for i,d in enumerate(dirs) for arg in
                ('--student' if i else '--teacher', str(d/'npt-1/segment-40000.json'), str(d/'npt-2/segment-40000.json'))]
        observables(['statistics',*runs,'--discard-frames','200','--block-frames',str(block),'--lags',*map(str,lags),
                     '--diffuse','O','--rmax-A','6','--bins','60','--output',str(dirs[-1]/f'npt-block-{block}.json')])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', type=Path)
    sub = p.add_subparsers(dest='task', required=True)
    sub.add_parser('sample').add_argument('part', choices=PARTS)
    sub.add_parser('assemble').add_argument('output', type=Path)
    ro = sub.add_parser('rollout'); ro.add_argument('job', type=Path); ro.add_argument('regime', choices=('target','hard'))
    sub.add_parser('start-md')
    md = sub.add_parser('md'); md.add_argument('case', choices=MD); md.add_argument('--job', type=Path)
    sub.add_parser('reduce').add_argument('--job', type=Path)
    a = p.parse_args(argv); root = a.root.resolve()
    if a.task in ('sample','rollout'):
        from .frames import main as frames
        if file_digest(root/'initial.xyz') != INITIAL_SHA:
            raise ValueError('starting water geometry or IDs differ from the pinned source')
        if a.task=='sample':
            frames(sample_args(root, a.part))
        else:
            name = 'fixed-student-test' if a.regime=='target' else 'hard'
            frames(rollout_args(root, *checked_student(a.job), a.regime, a.job/f'{name}.xyz'))
    elif a.task=='assemble':
        print(json.dumps(assemble(root, a.output), indent=2))
    elif a.task=='start-md':
        dest = root/'observables/start.xyz'; dest.parent.mkdir(parents=True,exist_ok=True)
        at=read(root/'valid.xyz',index=19)
        if at.info['regime']!='target' or at.info['step']!=18000 or at.info['seed']!=92001:
            raise ValueError('unexpected teacher calibration start')
        with dest.open('x') as f:
            write(f,at,format='extxyz',write_results=False)
    elif a.task=='md':
        from .observables import main as observables
        directory=(a.job or root)/'observables'
        model,sha=checked_student(a.job) if a.job else ('off24-medium',None)
        observables(md_args(root,'student' if a.job else 'teacher',a.case,directory,model,sha))
    else:
        reduce(root,a.job)


if __name__ == '__main__':
    main()
