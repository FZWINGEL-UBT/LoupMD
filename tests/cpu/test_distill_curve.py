"""CPU-only learning curve assembly: validate the same bytes we publish."""
import hashlib
import json
import shutil

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from loupmd.distill import curve


SHA = hashlib.sha256(b'teacher').hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sidecar(path):
    path.with_suffix('.sha256').write_text(f'{digest(path)}  shards/{path.name}\n')


def frame(split, sample, replica=0):
    a = Atoms('HeH', positions=[[sample / 100., 0., 0.], [0., 1., 0.]], cell=[5.] * 3, pbc=True)
    a.info.update(split=split, sampler='teacher', frame_id=f'frame-{sample}', trajectory_id=f'traj-{sample}',
                  md_seed=2**60 + sample, teacher_head='omat_pbe', teacher_sha256=SHA, replica=replica,
                  temperature_K=300. if replica < 2 else 450., burn_in=1000, stride=1, step=1001,
                  REF_energy=-1., REF_stress=np.eye(3))
    a.new_array('REF_forces', np.ones((2, 3)))
    return a


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(curve, 'BASE_FRAMES', 2)
    monkeypatch.setattr(curve, 'SHARD_FRAMES', 2)
    monkeypatch.setattr(curve, 'LEVELS', (4, 6, 8))
    base = tmp_path / 'base'
    base.mkdir()
    a, b = frame('train', 10), frame('train', 11)
    # Existing train uses multiple time slices of the same independent MD trajectory.
    b.info.update(trajectory_id=a.info['trajectory_id'], md_seed=a.info['md_seed'], step=1002)
    for name, rows in [('train', [a, b]), ('valid', [frame('valid', 12)]),
                       ('teacher-test', [frame('teacher-test', 13)])]:
        write(base / f'{name}.xyz', rows, format='extxyz')
        sidecar(base / f'{name}.xyz')
    for i, directory in enumerate(('z', 'a')):
        path = tmp_path / directory / f'part-{i:02d}.xyz'
        path.parent.mkdir()
        write(path, [frame('train', 20 + j, j) for j in range(2*i, 2*i + 2)], format='extxyz')
        sidecar(path)
    return tmp_path


def assemble(root, count=6, *extra):
    curve.main(['--base', str(root / 'base'), '--base-sha256', digest(root / 'base/train.xyz'),
                '--shards', str(root / 'a'), '--shards', str(root / 'z'), '--count', str(count),
                '--output', str(root / f'out-{count}'), *extra])
    return root / f'out-{count}'


def test_nested_prefixes_and_immutable_snapshots(study):
    base = study / 'base'
    small, large = assemble(study, 4), assemble(study, 6)
    paths = [base / 'train.xyz', study / 'z/part-00.xyz', study / 'a/part-01.xyz']
    train = (small / 'train.xyz').read_bytes()
    assert train == b''.join(p.read_bytes() for p in paths[:2])
    assert (large / 'train.xyz').read_bytes() == train + paths[-1].read_bytes()
    manifest = json.loads((large / 'manifest.json').read_text())
    assert manifest['frames'] == 6 and manifest['new_independent_trajectories'] == 4
    assert manifest['new_temperature_counts'] == {'300': 2, '450': 2}
    assert manifest['teacher_sha256'] == SHA and manifest['teacher_head'] == 'omat_pbe'
    assert manifest['split_counts'] == {'train': 6, 'valid': 1, 'teacher-test': 1}
    assert manifest['train_sha256'] == digest(large / 'train.xyz')
    for p in paths + [base / 'valid.xyz', base / 'teacher-test.xyz']:
        assert manifest['input_sha256'][p.name] == digest(p)
    for name in ('valid.xyz', 'teacher-test.xyz'):
        (base / name).unlink()
        assert digest(small / name) == digest(large / name) == manifest['input_sha256'][name]
    with pytest.raises(FileExistsError):
        assemble(study, 6)
    assert (large / 'manifest.json').is_file()


@pytest.mark.parametrize('bad, message', [
    ('replica', 'replica'), ('md_seed', 'md_seed'), ('trajectory_id', 'trajectory_id'),
    ('frame_id', 'frame_id'), ('geometry', 'geometry'), ('split', 'split'), ('step', 'terminal'),
    ('teacher_head', 'teacher'), ('teacher_sha256', 'teacher'), ('species', 'composition'),
    ('natoms', 'composition'), ('REF_energy', 'REF_energy'), ('REF_forces', 'REF_forces'),
    ('REF_stress', 'REF_stress'), ('short', 'frames'), ('long', 'frames')])
def test_invalid_final_shard_never_published(study, bad, message):
    path = study / 'a/part-01.xyz'
    rows = read(path, ':')
    a = rows[-1]
    if bad == 'short':
        rows.pop()
    elif bad == 'long':
        rows.append(frame('train', 24, 4))
    elif bad == 'species':
        a.numbers[0] = 3
    elif bad == 'natoms':
        del a[1]
    elif bad == 'geometry':
        a.positions[:] = read(study / 'base/valid.xyz').positions
    elif bad == 'REF_forces':
        a.arrays[bad][1, 0] = np.inf
    else:
        a.info[bad] = dict(replica=4, md_seed=2**60 + 12, trajectory_id='traj-20', frame_id='frame-13',
                           split='valid', step=1002, teacher_head='other', teacher_sha256='b' * 64,
                           REF_energy=np.nan, REF_stress=np.ones(6))[bad]
    write(path, rows, format='extxyz')
    sidecar(path)
    with pytest.raises(ValueError, match=message):
        assemble(study)
    assert not (study / 'out-6').exists()
    assert not list(study.glob('.out-6*'))


@pytest.mark.parametrize('bad', ['ambiguous', 'missing', 'alias', 'range', 'checksum', 'wrong-filename',
                                 'no-checksum', 'truncated', 'base-size', 'valid-overlap'])
def test_incomplete_or_ambiguous_inputs(study, bad):
    p = study / 'a/part-01.xyz'
    if bad == 'ambiguous':
        shutil.copyfile(p, study / 'z' / p.name)
    elif bad == 'missing':
        p.unlink()
    elif bad == 'alias':
        p.rename(p.with_name('part-1.xyz'))
    elif bad == 'range':
        p.with_name('part-388.xyz').touch()
    elif bad == 'checksum':
        p.with_suffix('.sha256').write_text(f'{"f" * 64}  {p.name}\n')
    elif bad == 'wrong-filename':
        p.with_suffix('.sha256').write_text(f'{digest(p)}  part-00.xyz\n')
    elif bad == 'no-checksum':
        p.with_suffix('.sha256').unlink()
    elif bad == 'truncated':
        p.write_bytes(p.read_bytes().rsplit(b'\n', 2)[0] + b'\n')
        sidecar(p)
    elif bad == 'base-size':
        p = study / 'base/train.xyz'
        write(p, frame('train', 10), format='extxyz')
    else:
        p = study / 'base/valid.xyz'
        a = read(p)
        a.info['trajectory_id'] = 'traj-10'
        write(p, a, format='extxyz')
        sidecar(p)
    with pytest.raises((ValueError, FileNotFoundError)):
        assemble(study)
    assert not (study / 'out-6').exists()


@pytest.mark.parametrize('name, flag', [('train', '--base-sha256'), ('valid', '--valid-sha256'),
                                       ('teacher-test', '--teacher-test-sha256')])
def test_frozen_split_pins(study, name, flag):
    with pytest.raises(ValueError, match='sha256'):
        assemble(study, 4, flag, '1' * 64)
    assert not (study / 'out-4').exists()
    (study / f'base/{name}.sha256').unlink()
    assemble(study, 4, flag, digest(study / f'base/{name}.xyz'))


def test_production_levels_select_numeric_prefixes(tmp_path):
    assert curve.LEVELS == (1000, 3000, 10000)
    assert curve.BASE_FRAMES == 300 and curve.SHARD_FRAMES == 25
    for i in reversed(range(388)):
        (tmp_path / f'part-{i:02d}.xyz').touch()
    for count, last in ((1000, 27), (3000, 107), (10000, 387)):
        paths = curve._shards([tmp_path], (count - curve.BASE_FRAMES) // curve.SHARD_FRAMES)
        assert [p.name for p in paths] == [f'part-{i:02d}.xyz' for i in range(last + 1)]


@pytest.mark.parametrize('mutation', ['empty-receipt', 'wrong-corpus', 'wrong-baseline', 'missing-group', 'changed-valid'])
def test_training_admission_rejects_a_receipt_for_different_bytes(study, mutation):
    base, data = study / 'base', assemble(study, 6)
    manifest = json.loads((data / 'manifest.json').read_text())
    (base / 'manifest.json').write_text(json.dumps({'splits': {
        split: {'output_sha256': digest(base / f'{split}.xyz'), 'frames': count}
        for split, count in (('train', 2), ('valid', 1), ('teacher-test', 1))}}))
    receipt = dict(schema='loupmd.curve-release.v2', label_release='curve-labels-v2',
                   base_manifest_sha256=digest(base / 'manifest.json'), split_counts=manifest['split_counts'],
                   train_sha256=manifest['train_sha256'], groups={'0': {'gate_sha256': SHA, 'receipt_sha256': SHA}})
    path = data / 'release.json'
    path.write_text(json.dumps(receipt))
    curve.verify_training_release(base, data, 6)
    if mutation == 'empty-receipt':
        receipt = {}
    elif mutation == 'wrong-corpus':
        receipt['train_sha256'] = digest(base / 'train.xyz')
    elif mutation == 'wrong-baseline':
        receipt['base_manifest_sha256'] = SHA
    elif mutation == 'missing-group':
        receipt['groups'] = {}
    else:
        shutil.copyfile(base / 'teacher-test.xyz', data / 'valid.xyz')
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='release'):
        curve.verify_training_release(base, data, 6)
