from pathlib import Path

import pytest

from loupgym.distill import water_protocol as wp


def option(cmd, key):
    return cmd[cmd.index('--'+key)+1]


def test_water_split_common_integrator_and_partitioned_streams(tmp_path):
    args = [wp.sample_args(tmp_path, p) for p in wp.PARTS]
    split_counts = {}
    streams = set()
    for cmd in args:
        assert option(cmd, 'teacher') == 'off24-medium' and float(option(cmd, 'timestep-fs')) == .5
        assert float(option(cmd, 'target-K')) == 300 and float(option(cmd, 'hot-K')) == 450
        assert int(option(cmd, 'burn-in')) == 10000 and int(option(cmd, 'stride')) == 400
        assert option(cmd, 'input') == str(tmp_path/'initial.xyz')
        seed = (option(cmd, 'seed'), option(cmd, 'first-replica'))
        assert seed not in streams
        streams.add(seed)
        role = option(cmd, 'split')
        split_counts[role] = split_counts.get(role, 0) + int(option(cmd, 'target-frames')) + int(option(cmd, 'hot-frames'))
    assert split_counts == {'train': 300, 'valid': 30, 'teacher-test': 60}


def test_water_fixed_and_hard_are_independent_heldouts(tmp_path):
    cmds = [wp.rollout_args(tmp_path, tmp_path/'student.model', 'a'*64, role, tmp_path/(role+'.xyz'))
            for role in ('target', 'hard')]
    assert all(option(cmd, 'split') == 'fixed-student-test' for cmd in cmds)
    assert option(cmds[0], 'target-frames') == option(cmds[1], 'hot-frames') == '60'
    assert option(cmds[1], 'target-frames') == option(cmds[0], 'hot-frames') == '0'
    assert option(cmds[0], 'seed') != option(cmds[1], 'seed')
    assert float(option(cmds[1], 'strain')) == float(option(cmds[1], 'shear')) == .01


def test_water_observables_are_paired_and_use_equal_physical_nve_windows(tmp_path):
    root = Path(tmp_path)
    commands = [wp.md_args(root, 'teacher', case, root/'calibration', 'off24-medium') for case in wp.MD]
    for case, teacher in zip(wp.MD, commands):
        student = wp.md_args(root, 'student', case, root/'observations', root/'student.model', 'a'*64)
        for setting in ('input', 'temperature-K', 'ensemble', 'seed', 'steps', 'stride', 'timestep-fs'):
            assert option(teacher, setting) == option(student, setting)
    nve = commands[:2]
    assert len({int(option(c,'steps'))*float(option(c,'timestep-fs')) for c in nve}) == 1
    assert len({int(option(c,'stride'))*float(option(c,'timestep-fs')) for c in nve}) == 1
    assert len({option(c,'seed') for c in nve}) == 1
    assert len({option(c,'seed') for c in commands[2:]}) == 2
    assert all(float(option(c,'pressure-GPa')) == pytest.approx(.000101325) for c in commands[2:])


def test_release_binds_actual_colab_wrapper_and_both_complete_recipes(tmp_path):
    import numpy as np
    from ase import Atoms
    from ase.io import write
    from loupgym.distill.train import file_digest

    for role, steps in [('train',[1,2]), ('valid',[3]), ('teacher-test',[4])]:
        frames = []
        for i in steps:
            a = Atoms('OH2', positions=[[0,0,0],[.95,.01*i,0],[-.25,.94,0]], pbc=True, cell=[10]*3)
            a.set_array('mol_id', np.zeros(3, dtype=int))
            a.set_array('REF_forces', np.arange(9, dtype=float).reshape(3,3))
            a.info.update(split=role, sampler='teacher', frame_id=f'{role}:{i}', trajectory_id=role,
                          teacher_sha256=wp.TEACHER, teacher_head='default', REF_energy=-1., REF_stress=np.eye(3))
            frames.append(a)
        write(tmp_path/f'{role}.xyz', frames)
    hashes, arms = wp._training_files(tmp_path)
    assert hashes['colab_train.sh'] == wp.COLAB_SHA == file_digest(tmp_path/'colab_train.sh')
    assert arms['control']['molecular'] == arms['net-torque']['molecular']
    assert arms['control']['pair_id'] != arms['net-torque']['pair_id']
    for mode in arms:
        args = (tmp_path/f'{mode}.args').read_text().splitlines()
        assert option(args, 'molecular-loss') == mode and option(args, 'batch-size') == '2'
    assert f'{wp.COLAB_SHA}  colab_train.sh\n' in (tmp_path/'inputs.sha256').read_text()


def test_swapped_independently_gated_students_fail_closed(tmp_path):
    import hashlib
    import json
    from loupgym.distill.train import BASE_SPLITS, file_digest

    data = tmp_path/'dispatch'
    data.mkdir()
    def store(path, value):
        path.write_text(json.dumps(value))
    roles = []
    pins = {}
    for role in BASE_SPLITS:
        path = data/f'{role}.xyz'
        path.write_text(role)
        pins[path.name] = file_digest(path)
        roles.append({'role':role, 'sha256':pins[path.name]})
    release = dict(schema='loupmd.water-release.v1', teacher_sha256=wp.TEACHER, inputs_sha256=pins)
    for mode in ('control','net-torque'):
        run = tmp_path/f'{mode}-s1-v1'
        checked, native = run/'checked', run/'release'
        checked.mkdir(parents=True)
        native.mkdir()
        spec = dict(pair_id=hashlib.sha256(mode.encode()).hexdigest(),recipe={'molecular_loss':mode,'seed':1},
                    molecular={'molecule_key':'mol_id','net_scale':4.},offsets={'8':-42.})
        release[mode] = spec
        model = checked/'student.model'
        model.write_bytes(mode.encode())
        sha = file_digest(model)
        store(checked/'manifest.json',dict(**{k:v for k,v in spec.items() if k!='offsets'},teacher_sha256=wp.TEACHER,
             teacher_head='default',resume=None,mace_torch='0.3.16',student_sha256=sha,sources=roles,energy_offsets_eV=spec['offsets']))
        store(native/'gate.json',dict(model={'sha256':sha},passed=True,rows=[{'passed':True}]))
        bundle = native/'student.loup'
        bundle.write_bytes(b'gate-qualified '+mode.encode())
        (native/'student.loup.sha256').write_text(file_digest(bundle)+'  student.loup\n')
    store(data/'release.json',release)
    for mode in ('control','net-torque'):
        run = tmp_path/f'{mode}-s1-v1'
        assert wp.checked_student(tmp_path,run)[0] == run/'checked/student.model'
    a,b=tmp_path/'control-s1-v1',tmp_path/'net-torque-s1-v1'
    a.rename(tmp_path/'moving'); b.rename(a); (tmp_path/'moving').rename(b)
    for run in (a,b):
        with pytest.raises(ValueError,match='dispatch arm'):
            wp.checked_student(tmp_path,run)
    # Reversing the swap restores each release, but reordered split identities are not valid.
    a.rename(tmp_path/'moving'); b.rename(a); (tmp_path/'moving').rename(b)
    path=a/'checked/manifest.json'
    m=json.loads(path.read_text()); m['sources'].reverse(); store(path,m)
    with pytest.raises(ValueError,match='source'):
        wp.checked_student(tmp_path,a)
