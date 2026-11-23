"""Generate teacher-labelled periodic extxyz sets with LoupMD Langevin dynamics.

    python -m loupmd.distill.frames --split train --output train.extxyz --teacher mh1 \
        --teacher-head omat_pbe --seed 1 --target-K 300 --hot-K 600 --target-frames 200 --hot-frames 100
    python -m loupmd.distill.frames --rescore teacher-test.extxyz --output scored.extxyz \
        --student student.model --student-sha256 CHECKSUM

Per replica, each temperature starts independently from the input, not from the preceding
trajectory. Frame i (1-based) is at burn_in + i * stride. Increase burn-in and stride to
suit the system's equilibration and slowest relevant decorrelation time. Use --first-replica
and a new output path for each reservation shard. No set is ever appended to or overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from ase.io import iread, read, write

import loupmd as lm
from loupmd.bench.suites import srtio3
from loupmd.quick import draw_initial_velocities
from loupmd.simulation import fixed_mask

# These indices are part of the RNG stream contract. Only append new splits.
SPLITS = ('train', 'valid', 'teacher-test', 'harvest', 'fixed-student-test')
STUDENT_SPLITS = ('harvest', 'fixed-student-test')


@dataclass(frozen=True)
class Sampling:
    split: str
    seed: int
    target_K: float
    hot_K: float | None = None
    target_frames: int = 1
    hot_frames: int = 0
    replicas: int = 1
    first_replica: int = 0
    burn_in: int = 100
    stride: int = 25
    timestep_fs: float = 1.0
    friction_per_fs: float = 0.01
    strain: float = 0.0
    shear: float = 0.0
    rattle_A: float = 0.02

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f'split must be one of {SPLITS}')
        if not 0 <= self.seed < 2**63:
            raise ValueError('seed must be in [0, 2**63)')
        if self.replicas < 1 or self.first_replica < 0 or self.burn_in < 0 or self.stride < 1:
            raise ValueError('replicas and stride must be positive; first-replica and burn-in must be nonnegative')
        if min(self.target_frames, self.hot_frames) < 0 or self.target_frames + self.hot_frames == 0:
            raise ValueError('frame counts must be nonnegative with at least one frame requested')
        if not all(math.isfinite(x) and x > 0 for x in (self.target_K, self.timestep_fs, self.friction_per_fs)):
            raise ValueError('target-K, timestep-fs and friction-per-fs must be positive and finite')
        if self.hot_frames and self.hot_K is None:
            raise ValueError('hot frames require --hot-K')
        if self.hot_K is not None and not (math.isfinite(self.hot_K) and self.hot_K > self.target_K):
            raise ValueError('hot-K must be finite and greater than target-K')
        if not all(math.isfinite(x) and x >= 0 for x in (self.strain, self.shear, self.rattle_A)):
            raise ValueError('strain, shear and rattle-A must be nonnegative and finite')
        if self.strain + 2 * self.shear >= 1:
            raise ValueError('strain + 2 * shear must be below one so all sampled cells remain physical')


def _check_geometry(atoms):
    if not len(atoms) or not all(atoms.pbc) or atoms.cell.rank != 3 or not atoms.get_volume() > 0:
        raise ValueError('expected a nonempty structure periodic in all three directions with a nonsingular cell')
    if not (np.isfinite(atoms.cell.array).all() and np.isfinite(atoms.positions).all()):
        raise ValueError('nonfinite positions or cell')


def _identity(model, role):
    head = model.head or (model.heads[0] if model.heads and len(model.heads) == 1 else 'default')
    return {role + '_model': model.name, role + '_sha256': model.sha256, role + '_head': head}


def _labels(frame, prefix, energy, forces, stress):
    if energy is None or forces is None or stress is None:
        raise ValueError(f'missing {prefix}_energy/forces/stress')
    energy, forces, stress = np.asarray(energy), np.asarray(forces), np.asarray(stress)
    if energy.shape != () or forces.shape != (len(frame), 3) or stress.shape != (3, 3):
        raise ValueError(f'{prefix}_energy/forces/stress must be scalar, N x 3 and 3 x 3')
    if not all(np.isfinite(x).all() for x in (energy, forces, stress)):
        raise ValueError(f'nonfinite {prefix}_energy/forces/stress')
    return float(energy), forces, stress


def _evaluate(frame, model, prefix):
    # ForceField.evaluate returns the potential stress in ASE's tensile-positive eV/A^3,
    # not a pressure, not a virial, and with no kinetic contribution.
    energy, forces, stress = _labels(frame, prefix, *model.evaluate(frame, stress=True))
    frame.info[prefix + '_energy'] = energy
    frame.info[prefix + '_stress'] = stress
    frame.arrays.pop(prefix + '_forces', None)
    frame.new_array(prefix + '_forces', forces)
    frame.info.update(_identity(model, 'prediction' if prefix == 'STUDENT' else 'teacher'))


def _start(base, plan, regime_idx, replica, temperature):
    # Explicit independent streams: outputs don't depend on loop order, replica sharding,
    # temperature counts, the presence of a preceding hot run, or Python's salted hash().
    streams = np.random.SeedSequence([plan.seed, SPLITS.index(plan.split), regime_idx, replica]).spawn(3)
    strain_rng, rattle_rng = (np.random.default_rng(s) for s in streams[:2])
    md_seed = int(streams[2].generate_state(1, dtype=np.uint64)[0]) % 2**63
    xy, xz, yz = strain_rng.uniform(-plan.shear, plan.shear, size=3)
    scale = 1 + strain_rng.uniform(-plan.strain, plan.strain)
    deformation = np.array([[scale, xy, xz], [xy, scale, yz], [xz, yz, scale]])
    atoms = base.copy()
    # This is a new trajectory, not a continuation of the input file's labels or momenta.
    atoms.info.clear()
    for key in list(atoms.arrays):
        if key.startswith(('REF_', 'STUDENT_')) or key == 'momenta':
            del atoms.arrays[key]
    atoms.set_cell(base.cell.array @ deformation, scale_atoms=True)
    if plan.rattle_A:
        atoms.set_positions(atoms.positions + rattle_rng.normal(0, plan.rattle_A, atoms.positions.shape))
    draw_initial_velocities(atoms, temperature, md_seed)  # its velocity stream is separate from thermostat noise
    return atoms, md_seed, deformation


def _trajectory_id(atoms, metadata):
    # Distinguish trajectories that use the same seeds but different inputs, potentials, or integration settings.
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for values in (atoms.numbers, atoms.cell.array, atoms.positions, atoms.get_masses(), atoms.get_momenta()):
        digest.update(np.asarray(values, dtype='<f8').tobytes())
    mask = fixed_mask(atoms)
    if mask is not None:
        digest.update(mask.tobytes())
    return f"{metadata['split']}-{digest.hexdigest()}"


def sample(base, plan: Sampling, teacher, student=None, *, source='ase.Atoms'):
    """Yield labelled ASE frames. Teacher and student are separately loaded ``ForceField``s."""
    _check_geometry(base)
    if (plan.split in STUDENT_SPLITS) != (student is not None):
        raise ValueError('harvest and fixed-student-test require student MD; train, valid and teacher-test use teacher MD')
    if student is teacher:
        raise ValueError('student MD requires an independently loaded teacher')
    sampler = teacher if student is None else student
    # Resolve heads (including an implicit sole head on an out-of-table checkpoint) before recording identities.
    for model in ((teacher,) if student is None else (teacher, student)):
        model.check_elements(base.numbers)
    build = lm.library_version()
    for replica in range(plan.first_replica, plan.first_replica + plan.replicas):
        regimes = [('target', plan.target_K, plan.target_frames), ('hot', plan.hot_K, plan.hot_frames)]
        for idx, (regime, temperature, count) in enumerate(regimes):
            if not count:
                continue
            atoms, seed, deformation = _start(base, plan, idx, replica, temperature)
            meta = dict(split=plan.split, sampler='teacher' if student is None else 'student',
                        seed=plan.seed, md_seed=seed, replica=replica, regime=regime, temperature_K=temperature,
                        integrator='langevin', timestep_fs=plan.timestep_fs, friction_per_fs=plan.friction_per_fs,
                        engine_build=build, **_identity(sampler, 'sampler'))
            trajectory = _trajectory_id(atoms, meta)
            meta.update(trajectory_id=trajectory, source=source, burn_in=plan.burn_in, stride=plan.stride,
                        strain=plan.strain, shear=plan.shear, rattle_A=plan.rattle_A, deformation=deformation)
            dynamics = lm.integrators.Langevin(timestep_fs=plan.timestep_fs, temperature_K=temperature,
                                               friction_per_fs=plan.friction_per_fs,
                                               fixcm='auto' if len(atoms) > 1 else False)
            with closing(lm.Simulation(atoms, sampler, integrator=dynamics, seed=seed)) as sim:
                for i in range(count):
                    sim.run(steps=plan.stride + (plan.burn_in if i == 0 else 0))
                    frame = sim.atoms
                    frame.calc = None  # never serialize the integrator's stale or unnamespaced results
                    frame.info.update(meta, frame_id=f'{trajectory}:{sim.step}', step=sim.step)
                    _evaluate(frame, teacher, 'REF')
                    if student is not None:
                        _evaluate(frame, student, 'STUDENT')
                    yield frame


def rescore(frames, student):
    """Yield copies with fresh STUDENT_* only. Never integrate or re-evaluate the reference."""
    for frame in frames:
        _check_geometry(frame)
        _labels(frame, 'REF', frame.info.get('REF_energy'), frame.arrays.get('REF_forces'), frame.info.get('REF_stress'))
        scored = frame.copy()
        _evaluate(scored, student, 'STUDENT')
        yield scored


def _write_set(path, frames):
    # A killed reservation can leave a .partial, but never a truncated set under the final name.
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    handle = partial.open('x', encoding='utf-8')
    try:
        with handle, closing(frames):
            if path.exists():
                raise FileExistsError(f'refusing to replace existing set {path}')
            count = 0
            for frame in frames:
                write(handle, frame, format='extxyz', write_results=False)
                count += 1
            if not count:
                raise ValueError('empty frame set')
        partial.rename(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return count


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--split', choices=SPLITS, help='sample a new independent named set')
    mode.add_argument('--rescore', type=Path, help='evaluate a candidate on an existing labelled extxyz; no MD')
    p.add_argument('--output', type=Path, required=True, help='new extxyz path')
    for role in ('teacher', 'student'):
        p.add_argument('--' + role, help='model label or checkpoint path')
        p.add_argument('--' + role + '-head')
        p.add_argument('--' + role + '-sha256', help='verified checkpoint digest for an unlisted model')
    md = p.add_argument_group('sampling only')
    md.add_argument('--input', type=Path, help='periodic extxyz structure (default: 4x4x4 SrTiO3, 320 atoms)')
    md.add_argument('--input-index', type=int, help='frame in --input (default: first)')
    md.add_argument('--seed', type=int, help='required root seed (streams partitioned by split, regime and replica)')
    md.add_argument('--target-K', type=float, help='required target temperature')
    md.add_argument('--hot-K', type=float, help='independent hot trajectory temperature, required for hot frames')
    md.add_argument('--target-frames', type=int, help='target frames per replica (default: 1)')
    md.add_argument('--hot-frames', type=int, help='hot frames per replica (default: 0)')
    md.add_argument('--replicas', type=int, help='number of independent replicas (default: 1)')
    md.add_argument('--first-replica', type=int, help='first replica index for this shard (default: 0)')
    md.add_argument('--burn-in', type=int, help='initial unsampled steps per trajectory (default: 100)')
    md.add_argument('--stride', type=int, help='steps between saved frames (default: 25)')
    md.add_argument('--timestep-fs', type=float, help='Langevin timestep in fs (default: 1)')
    md.add_argument('--friction-per-fs', type=float, help='Langevin friction in 1/fs (default: 0.01)')
    md.add_argument('--strain', type=float, help='uniform isotropic fractional strain half-width (train: 0.01; else: 0)')
    md.add_argument('--shear', type=float, help='uniform symmetric off-diagonal strain half-width (train: 0.01; else: 0)')
    md.add_argument('--rattle-A', type=float, help='startup Gaussian displacement std. dev. per coordinate (default: 0.02 Å)')
    opts = p.parse_args(argv)
    if opts.output.exists():
        raise FileExistsError(f'refusing to replace existing set {opts.output}')

    def load(role):
        return lm.load(getattr(opts, role), head=getattr(opts, role + '_head'), sha256=getattr(opts, role + '_sha256'))

    settings = {k: v for k, v in vars(opts).items() if k in Sampling.__dataclass_fields__ and v is not None}
    if opts.rescore is not None:
        if not opts.student or any((settings, opts.input, opts.input_index is not None, opts.teacher,
                                    opts.teacher_head, opts.teacher_sha256)):
            p.error('--rescore requires --student and accepts no sampling or teacher options')
        result = rescore(iread(opts.rescore, index=':', format='extxyz'), load('student'))
    else:
        if opts.teacher is None or opts.seed is None or opts.target_K is None:
            p.error('sampling requires --teacher, --seed and --target-K')
        if (opts.split in STUDENT_SPLITS) != (opts.student is not None):
            raise ValueError('--student is required only for harvest and fixed-student-test')
        if opts.student is None and (opts.student_sha256 or opts.student_head):
            p.error('student options require --student')
        if opts.input is None and opts.input_index is not None:
            p.error('--input-index requires --input')
        if opts.split == 'train':
            settings.setdefault('strain', 0.01)
            settings.setdefault('shear', 0.01)
        plan = Sampling(**settings)
        index = opts.input_index or 0
        base = read(opts.input, index=index, format='extxyz') if opts.input else srtio3(4)
        _check_geometry(base)
        source = f'{opts.input}:{index}' if opts.input else 'SrTiO3-4x4x4'
        result = sample(base, plan, load('teacher'), load('student') if opts.student else None, source=source)
    count = _write_set(opts.output, result)
    print(f'{opts.output}: {count} labelled frames')


if __name__ == '__main__':
    main()
