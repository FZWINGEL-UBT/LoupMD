import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from loupgym.distill.harvest import select


def test_harvest_uses_same_geometry_and_never_trains_on_test(tmp_path):
    data = []
    for i, err in enumerate((0.1, 0.4, 0.2)):
        a = Atoms('H', positions=[[i, 1, 1]], cell=[6]*3, pbc=True)
        a.info.update(frame_id=f'h:{i}', trajectory_id='h', sampler='student', split='harvest',
                      teacher_sha256='t', prediction_sha256='s', sampler_sha256='s', teacher_head='default',
                      prediction_head='default', sampler_head='default', REF_energy=0., STUDENT_energy=0.,
                      REF_stress=np.eye(3), STUDENT_stress=np.eye(3))
        a.arrays['REF_forces'] = np.ones((1, 3))
        a.arrays['STUDENT_forces'] = np.full((1, 3), 1. + err)
        data.append(a)
    source, chosen = tmp_path/'h.xyz', tmp_path/'chosen.xyz'
    write(source, data)
    select(source, chosen, 2)
    got = read(chosen, ':')
    assert [a.info['frame_id'] for a in got] == ['h:1', 'h:2']
    assert all('STUDENT_forces' not in a.arrays and a.info['split'] == 'harvest' for a in got)
    np.testing.assert_array_equal(got[0].positions, data[1].positions)
    np.testing.assert_array_equal(got[1].arrays['REF_forces'], data[2].arrays['REF_forces'])
    data[2].info['split'] = 'fixed-student-test'
    write(source, data)
    with pytest.raises(ValueError):
        select(source, tmp_path/'leaked.xyz', 1)
    assert not (tmp_path/'leaked.xyz').exists()
