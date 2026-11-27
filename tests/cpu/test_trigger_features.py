"""Torch-free tests of the Stage 0 feature/label boundary."""
import hashlib
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import iread, write

from loupgym.distill import trigger_features as features


SHA = 'a1' * 32
TEACHER = 'b2' * 32


def frame(split='fixed-student-test'):
    atoms = Atoms('OH', positions=[[.1, 0, 0], [3.9, 0, 0]], cell=[4, 6, 8], pbc=True)
    atoms.info.update(frame_id=split + '-frame:1', trajectory_id=split + '-traj', split=split,
                      sampler='student' if split == 'fixed-student-test' else 'teacher', regime='target',
                      teacher_sha256=TEACHER, teacher_head='pbe')
    atoms.new_array('REF_forces', np.array([[0., 3., 4.], [1., 1., 1.]]))
    if split != 'train':
        atoms.info.update(prediction_sha256=SHA, prediction_head='Default')
        atoms.new_array('STUDENT_forces', np.array([[0., 0., 0.], [1., 1., 2.]]))
    return atoms


def numeric(_atoms):
    return dict(energy=np.array([-10., .125]), h=np.array([[1., 2.], [-1., 3.]]),
                hidden=np.arange(32.).reshape(2, 16), edge_index=np.array([[0, 1], [1, 0]]),
                distance=np.array([.2, .2]), cutoff=3., cutoff_p=5)


@pytest.mark.parametrize('split', ['train', 'valid', 'fixed-student-test'])
def test_extxyz_npz_contract(tmp_path, split):
    xyz, dst = tmp_path / 'frames.xyz', tmp_path / split
    write(xyz, frame(split), format='extxyz')
    calls = []

    def extract(atoms):
        calls.append(atoms.info['frame_id'])
        return numeric(atoms)

    assert features.write_split(iread(xyz, index=':'), dst, split, SHA, 'Default', extract) == 1
    assert calls == [split + '-frame:1']
    with np.load(next(dst.glob('*.npz')), allow_pickle=False) as row:
        assert all(row[k].dtype.kind != 'O' for k in row.files)
        for k, value in numeric(None).items():
            np.testing.assert_array_equal(row[k], value)
        np.testing.assert_array_equal(row['numbers'], [8, 1])
        np.testing.assert_array_equal(row['ref_forces'], frame(split).arrays['REF_forces'])
        for k, value in dict(frame(split).info, prediction_head='Default', prediction_sha256=SHA).items():
            assert row[k].shape == () and row[k].item() == value
        if split == 'train':
            assert 'force_error' not in row
        else:
            np.testing.assert_allclose(row['force_error'], [5., 1.])
    with pytest.raises(FileExistsError):
        features.write_split([], dst, split, SHA, 'Default', numeric)


@pytest.mark.parametrize(('key', 'value'), [
    ('split', 'train'), ('sampler', 'teacher'), ('frame_id', ''), ('trajectory_id', ''),
    ('teacher_sha256', 'bad'), ('prediction_sha256', TEACHER), ('prediction_head', 'different'),
    ('STUDENT_forces', None), ('REF_forces', np.full((2, 3), np.nan)),
])
def test_invalid_joint_label_rejected_before_forward(tmp_path, key, value):
    atoms = frame()
    if key in atoms.arrays:
        if value is None:
            del atoms.arrays[key]
        else:
            atoms.arrays[key] = value
    else:
        atoms.info[key] = value
    dst = tmp_path / 'fixed'
    with pytest.raises(ValueError, match=key):
        features.write_split([atoms], dst, 'fixed-student-test', SHA, 'Default',
                             lambda a: pytest.fail('invalid label was passed to the student'))
    assert not dst.exists()


@pytest.mark.parametrize(('key', 'bad'), [('edge_index', np.array([[0, 2], [1, 0]])),
                                        ('distance', np.array([.2])), ('hidden', np.zeros((2, 15))),
                                        ('energy', np.array([np.nan, 0.])), ('h', np.zeros((2, 0)))])
def test_invalid_features_not_published(tmp_path, key, bad):
    with pytest.raises(ValueError, match=key):
        features.write_split([frame()], tmp_path / 'fixed', 'fixed-student-test', SHA, 'Default',
                             lambda a: dict(numeric(a), **{key: bad}))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('problem', ['empty', 'duplicate', 'changed teacher'])
def test_partial_sets_not_published(tmp_path, problem):
    frames = [] if problem == 'empty' else [frame(), frame()]
    if problem == 'changed teacher':
        frames[1].info.update(frame_id='other', teacher_head='other')
    with pytest.raises(ValueError):
        features.write_split(frames, tmp_path / 'fixed', 'fixed-student-test', SHA, 'Default', numeric)
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_bytes_verified_before_deserialization(tmp_path):
    path = tmp_path / 'student.model'
    path.write_bytes(b'not a checkpoint')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert features.trusted_bytes(path, digest) == b'not a checkpoint'
    with pytest.raises(ValueError, match='SHA-256'):
        features.trusted_bytes(path, SHA)


def test_forward_taps_exact_inputs_without_replaying_the_student():
    class Tap:
        @contextmanager
        def register_forward_pre_hook(self, hook):
            self.hook = hook
            try:
                yield
            finally:
                del self.hook

    radial, last_linear = Tap(), Tap()
    data = dict(positions=frame().positions, edge_index=numeric(None)['edge_index'],
                shifts=np.array([[-4., 0., 0.], [4., 0., 0.]]))
    h, hidden, energy = (numeric(None)[k] for k in ('h', 'hidden', 'energy'))
    calls = []

    class Forward:
        radial_embedding = radial
        readouts = [None, SimpleNamespace(linear_2=last_linear)]

        def __call__(self, graph, *, training, compute_force):
            assert not training and not compute_force
            calls.append(1)
            src, dst = graph['edge_index']
            vec = graph['positions'][dst] - graph['positions'][src] + graph['shifts']
            radial.hook(radial, (np.linalg.norm(vec, axis=-1, keepdims=True), None, graph['edge_index'], None))
            last_linear.hook(last_linear, (hidden,))
            return {'node_feats': np.column_stack((h, -h)), 'node_energy': energy}

    row = features._forward_features(Forward(), data, h.shape[1])
    assert calls == [1] and not hasattr(radial, 'hook') and not hasattr(last_linear, 'hook')
    for key in ('h', 'hidden', 'energy', 'edge_index', 'distance'):
        np.testing.assert_allclose(row[key], numeric(None)[key])
