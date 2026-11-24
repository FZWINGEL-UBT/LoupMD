import numpy as np
import pytest
from ase import Atoms

from loupmd.distill.salt import paired, transport


def test_paired_balances_each_regime_and_relabels_only_changed():
    frames = []
    for i in range(8):
        a = Atoms('LiCl', positions=[[0, 0, 0], [1, 1, 1]], cell=[8]*3, pbc=True)
        a.info.update(frame_id=str(i), trajectory_id=str(i % 2), split='train')
        frames.append(a)
    mixed = list(paired(frames, 31, .04, lambda a: a.info.update(relabel=True)))
    assert [f.info['parent_frame_id'] for f in mixed if 'relabel' in f.info] == ['2', '3', '6', '7']
    assert all(np.array_equal(a.positions, b.positions) == ('relabel' not in b.info) for a, b in zip(frames, mixed))
    with pytest.raises(ValueError):
        list(paired(frames[:-1], 31, .04, lambda a: None))


def test_transport_removes_global_drift_and_unwraps():
    frames = []
    for i in range(40):
        a = Atoms('LiCl', positions=[[i*.8%5, 0, 0], [(1+i*.8)%5, 1, 0]], cell=[5]*3, pbc=True)
        a.info.update(step=i, timestep_fs=1, trajectory_id='same')
        frames.append(a)
    assert max(abs(v['D_A2_ps']) for v in transport(frames)['species'].values()) < 1e-15
