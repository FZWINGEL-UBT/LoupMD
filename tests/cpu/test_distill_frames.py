"""Distillation samples engine MD, not the evaluator's last positions. No torch or CUDA."""
import hashlib

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from fake_core import SpringEngine

import loupmd as lm
from loupmd.distill import frames


# Distinguishable analytic single points on the *supplied* geometry. For MD, reuse the
# suite's stateful engine so tests also exercise Simulation, units, velocities and step counts.
def labels(x, cell, k):
    stress = np.array([[1., -.25, .375], [-.25, -2., .5], [.375, .5, 3.]]) * k / abs(np.linalg.det(cell))
    return k * (x * x).sum() / 2, -k * x, stress


class LabelEngine(SpringEngine):
    def evaluate(self, positions, cell, compute_stress=True):
        if not compute_stress:
            return super().evaluate(positions, cell, compute_stress=False)
        return labels(positions, cell, {'teacher': 2, 'student': 3, 'candidate': 5}[self.bundle.stem])


@pytest.fixture
def models(tmp_path, monkeypatch):
    load = lm.load
    made = []
    for key in ('teacher', 'student', 'candidate'):
        (tmp_path / key).write_bytes(key.encode())

    def fake_load(model, **kw):
        made.append(model)
        return load(tmp_path / model, **kw, engine_factory=LabelEngine, probe=lambda _: None,
                    ensure_bundle=lambda path, *args, **kwargs: path)

    monkeypatch.setattr(lm, 'load', fake_load)
    monkeypatch.setattr(lm, 'library_version', lambda: 'cpu-test-engine')
    monkeypatch.setattr(SpringEngine, 'made', [])
    return made


def model_args(key, model):
    return ['--' + key, model, '--' + key + '-sha256', hashlib.sha256(model.encode()).hexdigest()]


def sample_args(split, output):
    return ['--split', split, '--output', str(output), '--seed', '9', '--target-K', '300', '--hot-K', '600',
            '--target-frames', '2', '--hot-frames', '1', '--burn-in', '3', '--stride', '2', '--strain', '.015',
            '--shear', '.012', '--rattle-A', '.02', '--teacher-head', 'omat_pbe', *model_args('teacher', 'teacher')]


def check_labels(frame, prefix, k):
    e, f, s = labels(frame.positions, frame.cell, k)
    np.testing.assert_allclose(frame.info[prefix + '_energy'], e, rtol=1e-7)
    np.testing.assert_allclose(frame.arrays[prefix + '_forces'], f, atol=5e-8)
    np.testing.assert_allclose(frame.info[prefix + '_stress'], s, rtol=1e-10)


def test_startup_rattle_does_not_change_cell_velocities_or_thermostat_stream():
    atoms = Atoms('He2', positions=[[.2, .3, .4], [2., 2., 2.]], cell=[5, 5, 5], pbc=True)
    atoms.set_momenta(np.full((2, 3), 1000.))
    settings = dict(split='train', seed=71, target_K=300., strain=.03, shear=.02)
    plain, seed, deformation = frames._start(atoms, frames.Sampling(**settings, rattle_A=0.), 0, 0, 300.)
    rattled, other_seed, other_deformation = frames._start(atoms, frames.Sampling(**settings, rattle_A=.1), 0, 0, 300.)
    assert seed == other_seed
    np.testing.assert_array_equal(deformation, other_deformation)
    np.testing.assert_allclose(plain.positions, atoms.positions @ deformation)
    assert not np.allclose(rattled.positions, plain.positions)
    np.testing.assert_equal(rattled.get_momenta(), plain.get_momenta())
    assert not np.array_equal(plain.get_momenta(), atoms.get_momenta())


def test_default_oxide_temperatures_independent_reproducible_streams(tmp_path, models):
    def run(split, file, extra=()):
        path = tmp_path / file
        frames.main([*sample_args(split, path), *extra])
        return read(path, ':', format='extxyz')

    first = run('train', 'train.xyz')
    again = run('train', 'repeat.xyz')
    other = run('valid', 'valid.xyz')
    shard = run('train', 'shard.xyz', ['--first-replica', '1'])
    together = run('train', 'both.xyz', ['--replicas', '2'])
    assert len(first) == 3 and all(len(a) == 320 for a in first)
    assert [a.info['temperature_K'] for a in first] == [300, 300, 600]
    assert [a.info['step'] for a in first] == [5, 7, 5]
    assert len({a.info['frame_id'] for a in first + shard + other}) == 9
    assert {a.info['frame_id'] for a in together} == {a.info['frame_id'] for a in first + shard}
    np.testing.assert_equal(first[0].cell.array, first[1].cell.array)
    for a, b in ((first[0], first[2]), (first[0], other[0]), (first[0], shard[0])):
        assert a.info['md_seed'] != b.info['md_seed']
        assert not np.allclose(a.cell, b.cell)
        assert not np.allclose(a.get_momenta(), b.get_momenta())
    for a, b in zip(first, again):
        assert a.info['frame_id'] == b.info['frame_id']
        np.testing.assert_array_equal(a.positions, b.positions)
        assert (a.info['split'], a.info['sampler'], a.info['teacher_head']) == ('train', 'teacher', 'omat_pbe')
        assert not any(key.startswith('STUDENT_') for key in (*a.info, *a.arrays))
        check_labels(a, 'REF', 2)
    runs = [event for engine in SpringEngine.made for event in engine.log
            if isinstance(event, tuple) and event[0] == 'run']
    assert runs and all(event[2] == lm._core.LOUP_LANGEVIN for event in runs)


def test_student_relabel_and_fixed_set_rescore(tmp_path, models):
    # Triclinic input with an extra per-atom field (e.g. a molecule ID), not an oxide.
    source = tmp_path / 'source.extxyz'
    cell = np.array([[4., 0., 0.], [.5, 5., 0.], [.1, -.4, 6.]])
    base = Atoms('HO', positions=[[1., 1., 1.], [2., 2., 2.]], cell=cell, pbc=True)
    base.new_array('mol_id', np.array([1, 1]))
    write(source, base)
    fixed, scored = tmp_path / 'fixed.extxyz', tmp_path / 'candidate.extxyz'
    frames.main([*sample_args('fixed-student-test', fixed), '--input', str(source),
                 *model_args('student', 'student'), '--hot-frames', '0'])
    originals = read(fixed, ':')
    original_bytes = fixed.read_bytes()
    assert sorted(models) == ['student', 'teacher']
    models.clear()
    frames.main(['--rescore', str(fixed), '--output', str(scored), *model_args('student', 'candidate')])
    assert models == ['candidate'] and original_bytes == fixed.read_bytes()
    candidates = read(scored, ':')
    assert len(candidates) == len(originals) == 2
    for a, b in zip(originals, candidates):
        assert a.info['sampler'] == 'student' and a.info['regime'] == 'target'
        for key in ('frame_id', 'step', 'split', 'teacher_sha256', 'sampler_sha256', 'trajectory_id'):
            assert a.info[key] == b.info[key]
        assert b.info['prediction_sha256'] != b.info['sampler_sha256'] == a.info['prediction_sha256']
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.cell.array, b.cell.array)
        np.testing.assert_array_equal(a.numbers, b.numbers)
        np.testing.assert_array_equal(b.arrays['mol_id'], [1, 1])
        check_labels(a, 'STUDENT', 3)
        check_labels(b, 'REF', 2)
        check_labels(b, 'STUDENT', 5)


@pytest.mark.parametrize('args,message', [(['--split', 'harvest'], 'student'),
                                          (['--hot-K', '200'], 'hot'),
                                          (['--stride', '0'], 'stride')])
def test_bad_sampling_requests_do_not_publish(tmp_path, models, args, message):
    out = tmp_path / 'out.extxyz'
    with pytest.raises(ValueError, match=message):
        frames.main([*sample_args('train', out), *args])
    assert not out.exists()


def test_rescore_requires_teacher_labels_and_never_overwrites(tmp_path, models):
    source = tmp_path / 'input.extxyz'
    write(source, Atoms('He2', positions=[[0, 0, 0], [2, 2, 2]], cell=[5]*3, pbc=True))
    args = ['--rescore', str(source), *model_args('student', 'candidate')]
    with pytest.raises(ValueError, match='REF_'):
        frames.main([*args, '--output', str(tmp_path / 'missing-labels.xyz')])
    before = source.read_bytes()
    with pytest.raises(FileExistsError):
        frames.main([*args, '--output', str(source)])
    assert source.read_bytes() == before
