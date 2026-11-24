"""The job recipe, resume boundary and energy transforms require no MACE or GPU."""
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write


# Colab can stage just this script and the extxyz files, without installing LoupMD.
train = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'loupmd/distill/train.py'))


def frame(role, i, energy):
    a = Atoms('Si2', positions=[[0, 0, 0], [2 + .01 * i, 0, 0]], cell=[6] * 3, pbc=True)
    a.info.update(split=role, sampler='student' if role == 'harvest' else 'teacher',
                  frame_id=f'f{i}', trajectory_id=role, teacher_sha256='a' * 64,
                  teacher_head='omat_pbe', REF_energy=energy,
                  REF_stress=np.array([[1., 2., 0.], [2., 3., 0.], [0., 0., 4.]]) * .001)
    a.arrays['REF_forces'] = np.arange(6.).reshape(2, 3) * .001
    return a


@pytest.fixture
def splits(tmp_path):
    paths = []
    for role, frames in [('train', [frame('train', 0, 1e8 + .125), frame('train', 1, 1e8 - .125)]),
                         ('valid', [frame('valid', 2, 1e8 + 7.)]),
                         ('teacher-test', [frame('teacher-test', 3, 1e8 - 2.)])]:
        path = tmp_path / (role + '.xyz')
        write(path, frames)
        paths.extend(['--' + role, str(path)])
    return paths


def recipe(tmp_path, splits, *extra):
    return train['plan_job'](train['parse_args']([*splits, '--out', str(tmp_path / 'job'),
                                                   '--batch-size', '2', *extra]))


def test_paired_recipe_has_explicit_stress_architecture_and_partitions(tmp_path, splits):
    a = recipe(tmp_path, splits, '--width', '96', '--radial-head', '8', '--cutoff', '4.5', '--seed', '2')
    b = recipe(tmp_path, splits, '--width', '96', '--radial-head', '8', '--cutoff', '4.5', '--seed', '2',
               '--dtype', 'float32')
    assert a['pair_id'] == b['pair_id']
    cmd = b['mace_args']
    def value(key):
        return cmd[cmd.index('--' + key) + 1]
    assert value('loss') == 'stress' and '--pair_repulsion' in cmd
    assert value('stress_key') == 'REF_stress' and float(value('stage_two_stress_weight')) > 0
    assert '--valid_file' in cmd and '--test_file' in cmd and '--valid_fraction' not in cmd
    assert value('hidden_irreps') == '96x0e' and json.loads(value('radial_MLP')) == [64, 64, 8]
    assert float(value('r_max')) == 4.5 and value('seed') == '2'
    assert value('radial_type') == 'bessel' and value('distance_transform') == 'None'
    assert b['radial_input']['species_dependent'] is False
    assert value('num_radial_basis') == '8' and '--ema' in cmd
    assert int(value('start_stage_two')) < int(value('max_num_epochs')) == 150
    assert float(value('stage_two_energy_weight')) > float(value('energy_weight'))
    assert float(value('stage_two_lr')) < float(value('lr'))
    assert not (tmp_path / 'job').exists()


def test_fp32_residuals_subtract_training_only_fit_before_cast(tmp_path, splits):
    plan = recipe(tmp_path, splits, '--dtype', 'float32')
    train['write_inputs'](plan, tmp_path / 'inputs')
    data = read(tmp_path / 'inputs/train.xyz', ':')
    np.testing.assert_allclose([a.info['REF_energy'] for a in data], [.125, -.125], atol=1e-6)
    assert read(tmp_path / 'inputs/valid.xyz').info['REF_energy'] == pytest.approx(7., abs=1e-6)
    assert read(tmp_path / 'inputs/teacher-test.xyz').info['REF_energy'] == pytest.approx(-2., abs=1e-6)
    assert data[0].info['REF_energy'] + 2 * plan['energy_offsets_eV']['14'] == pytest.approx(1e8 + .125, abs=1e-7)
    ref = read(tmp_path / 'train.xyz', 0)
    np.testing.assert_array_equal(data[0].arrays['REF_forces'], ref.arrays['REF_forces'])
    np.testing.assert_array_equal(data[0].info['REF_stress'], ref.info['REF_stress'])
    assert read(tmp_path / 'train.xyz', 0).info['REF_energy'] == 1e8 + .125


@pytest.mark.parametrize('mistake', ['role', 'trajectory', 'geometry', 'frame_id', 'stress', 'teacher', 'sampler'])
def test_no_test_leakage_or_unlabelled_periodic_frames(tmp_path, splits, mistake):
    a = read(tmp_path / 'valid.xyz')
    if mistake == 'role':
        a.info['split'] = 'fixed-student-test'
    elif mistake == 'trajectory':
        a.info['trajectory_id'] = 'train'
    elif mistake == 'geometry':
        a.positions[:] = read(tmp_path / 'train.xyz', 0).positions
    elif mistake == 'frame_id':
        a.info['frame_id'] = 'f0'
    elif mistake == 'stress':
        del a.info['REF_stress']
    elif mistake == 'teacher':
        a.info['teacher_head'] = 'wrong'
    else:
        a.info['sampler'] = 'student'
    write(tmp_path / 'valid.xyz', a)
    with pytest.raises(ValueError):
        recipe(tmp_path, splits)


def test_augmented_restart_keeps_offsets_and_locked_splits(tmp_path, splits):
    plan = recipe(tmp_path, splits, '--dtype', 'float32', '--epochs', '6', '--stage-two-start', '4')
    parent = tmp_path / 'base'
    parent.mkdir()
    ckpt = parent / 'pre_stage_two.pt'
    ckpt.write_bytes(b'checkpoint identity only')
    plan.update(pre_stage_two_sha256=train['file_digest'](ckpt), avg_num_neighbors=17.75)
    (parent / 'manifest.json').write_text(json.dumps(plan))
    harvest = tmp_path / 'harvest.xyz'
    write(harvest, frame('harvest', 4, 1e8 + 123.))
    args = ('--dtype', 'float32', '--epochs', '6', '--stage-two-start', '4',
            '--resume', str(parent), '--augment', str(harvest))
    child = recipe(tmp_path, splits, *args)
    assert child['energy_offsets_eV'] == plan['energy_offsets_eV']
    assert '--restart_latest' in child['mace_args'] and '17.75' in child['mace_args']
    ckdir = tmp_path / 'new' / 'checkpoints'
    paths = train['restart_paths'](ckdir, child)
    assert [p.name for p in paths] == ['student_run-1_epoch-3.pt', 'student_run-1_epoch-4_swa.pt']
    train['write_inputs'](child, tmp_path / 'augmented')
    assert len(read(tmp_path / 'augmented/train.xyz', ':')) == 3
    assert len(read(tmp_path / 'augmented/valid.xyz', ':')) == 1
    write(tmp_path / 'valid.xyz', frame('valid', 99, 0.))
    with pytest.raises(ValueError, match='split'):
        recipe(tmp_path, splits, *args)


def test_export_restores_species_offsets_in_model_order_at_reference_precision():
    # The model's species order is authoritative, not JSON key order.
    restore = train['physical_e0']
    offsets = {'8': 100000000.125, '1': -1000.25}
    result = restore(np.zeros((1, 2), dtype='float32'), [8, 1], offsets)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [[100000000.125, -1000.25]])
    with pytest.raises(ValueError, match='zero'):
        restore(result, [8, 1], offsets)


def test_molecular_pair_uses_training_only_scales_and_preserves_ids(tmp_path, splits):
    from loupmd.distill import train as module

    for name in ('train', 'valid', 'teacher-test'):
        path = tmp_path / f'{name}.xyz'
        frames = read(path, ':')
        for a in frames:
            a.arrays['mol_id'] = np.array([591, 591])
            if name != 'train':
                a.arrays['REF_forces'] *= 1e5
        write(path, frames)

    def plan(mode, dtype='float64'):
        return module.plan_job(module.parse_args([*splits, '--out', str(tmp_path / 'job'), '--batch-size', '2',
                                                  '--molecular-loss', mode, '--dtype', dtype]))
    control, weighted = plan('control'), plan('net-torque')
    assert control['molecular'] == weighted['molecular'] == plan('net-torque', 'float32')['molecular']
    assert weighted['pair_id'] == plan('net-torque', 'float32')['pair_id']
    assert control['pair_id'] != weighted['pair_id']
    assert control['energy_offsets_eV'] == weighted['energy_offsets_eV']
    assert weighted['molecular']['net_force_component_mse'] == pytest.approx(np.mean((np.array([3, 5, 7])*.001)**2))
    for label, job in [('c', control), ('w', weighted)]:
        dest = tmp_path / label
        module.write_inputs(job, dest)
        atoms = read(dest / 'train.xyz', 0)
        np.testing.assert_array_equal(atoms.arrays['mol_id'], [591, 591])
        np.testing.assert_array_equal(atoms.arrays['mol_anchor'], [0, 0])
        np.testing.assert_allclose(atoms.arrays['mol_arm'], [[-1, 0, 0], [1, 0, 0]])
    assert (tmp_path / 'c/train.xyz').read_bytes() == (tmp_path / 'w/train.xyz').read_bytes()
    write(tmp_path / 'valid.xyz', frame('valid', 2, 50.))
    with pytest.raises(ValueError, match='mol_id'):
        plan('net-torque')
