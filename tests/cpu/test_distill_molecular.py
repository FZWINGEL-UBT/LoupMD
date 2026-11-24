import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from loupmd.distill.water import assign_water_ids, inspect_water
from loupmd.distill.molecular import molecular_geometry


def wrapped_water():
    # Two different orientations; no OHH ordering, molecule zero crosses x=0.
    return Atoms('HOOHHH', positions=[[.76, 0, 0], [9.8, 0, 0], [4, 4, 4],
                                       [9.56, .93, 0], [4, 4, 4.96], [4.93, 4, 3.76]], cell=[10]*3, pbc=True)


def test_water_membership_and_periodic_com(tmp_path):
    a = wrapped_water()
    assign_water_ids(a)
    np.testing.assert_array_equal(a.arrays['mol_id'], [0, 0, 1, 0, 1, 1])
    write(tmp_path / 'water.xyz', a)
    a = read(tmp_path / 'water.xyz')
    geometry = inspect_water(a)
    assert geometry['molecules'] == 2 and 100 < geometry['angle_min_deg'] < 110
    anchor, arms = molecular_geometry(a, 'mol_id')
    np.testing.assert_array_equal(anchor, [0, 0, 2, 0, 2, 2])
    np.testing.assert_allclose(arms[0]-arms[1], [.96, 0, 0], atol=1e-12)
    for i in (0, 2):
        j = anchor == i
        np.testing.assert_allclose(np.average(arms[j], weights=a.get_masses()[j], axis=0), 0, atol=1e-12)
    a.positions[0] += 2*a.cell[0] - a.cell[1]
    np.testing.assert_allclose(molecular_geometry(a, 'mol_id')[1], arms, atol=1e-12)
    # Same stoichiometry, no chemically valid assignment.
    a.positions[0] = (2, 1, 1)
    with pytest.raises(ValueError, match='bond'):
        assign_water_ids(a)
    with pytest.raises(ValueError, match='bond'):
        inspect_water(a)


def test_molecular_loss_no_cross_frame_cancellation():
    torch = pytest.importorskip('torch')
    pytest.importorskip('mace')
    from loupmd.distill.molecular_loss import MolecularStressLoss
    from mace.modules.loss import WeightedEnergyForcesStressLoss
    from mace import data
    from mace.data.atomic_data import get_data_loader
    from mace.tools import AtomicNumberTable
    # Opposite errors on two graphs with identically numbered local molecules.
    a = wrapped_water()
    assign_water_ids(a)
    roots, arms = molecular_geometry(a, 'mol_id')
    err = torch.zeros(12, 3, dtype=torch.float64)
    err[0, 1], err[6, 1] = 2., -2.
    err.requires_grad_()
    t = torch.float64
    a.set_array('mol_anchor', roots)
    a.set_array('mol_arm', arms)
    a.set_array('REF_forces', np.zeros((6, 3)))
    a.info.update(REF_energy=1., REF_stress=np.eye(3))
    spec = data.KeySpecification(info_keys={'energy': 'REF_energy', 'stress': 'REF_stress'},
                                 arrays_keys={k: k for k in ('mol_anchor', 'mol_arm')})
    spec.arrays_keys['forces'] = 'REF_forces'
    cfg = data.config_from_atoms(a, key_specification=spec)
    graph = data.AtomicData.from_config(cfg, z_table=AtomicNumberTable([1, 8]), cutoff=2.)
    ref = next(iter(get_data_loader([graph, graph], batch_size=2, shuffle=False)))
    ref.mol_arm = ref.mol_arm.to(t)
    pred = dict(energy=ref.energy, stress=ref.stress, forces=err)
    for ew, sw in [(1, 1), (1000, 10)]:
        native = WeightedEnergyForcesStressLoss(energy_weight=ew, forces_weight=100., stress_weight=sw)
        loss = MolecularStressLoss(native, net_scale=2., torque_scale=3.)(ref, pred)
        expected = (2*2*4 + 3*2*np.linalg.norm(np.cross(arms[0], [0, 2, 0]))**2)/12
        assert (loss-native(ref, pred)).item() == pytest.approx(100*expected)
        assert torch.isfinite(torch.autograd.grad(loss, err)[0]).all()
