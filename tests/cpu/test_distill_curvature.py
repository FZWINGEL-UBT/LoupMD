"""Unit/one-update checks, never train a student. Optional torch/MACE tests are CPU-only."""
from importlib.metadata import version

from ase import Atoms
import numpy as np
import pytest

from loupgym.distill import curvature as c
from loupgym.distill.curvature_train import auxiliary, paired_batch


def test_pair_shuffle_is_reproducible_complete_and_epoch_local():
    first = c.schedule(111)
    assert first == c.schedule(111) and first != c.schedule(112)
    for e in range(111, 150):
        s = c.schedule(e)
        assert len(s) == 75 and sorted(s[2::3]) == list(range(25))
        assert s[::3] == s[1::3] == [None] * 25


def test_identity_binds_geometry_atom_order_side_and_force():
    a = Atoms('OTi', positions=[[1, 0, 0], [0, 1, 0]], cell=[9]*3, pbc=True, info={'frame_id': 'train:a'})
    pos, neg = a.copy(), a.copy()
    pos.positions += c.EPS; neg.positions -= c.EPS
    f = -pos.positions
    k = c.side_id(a, pos, f)
    assert k != c.side_id(a, neg, f) and k != c.side_id(a, pos[::-1], f)
    assert k != c.side_id(a, pos, -neg.positions)
    other = a.copy(); other.info['frame_id'] = 'train:b'
    assert k != c.side_id(other, pos, f)


def test_reject_wrong_digest_and_nontraining_file(tmp_path):
    p = tmp_path / 'teacher-test.xyz'; p.write_text('not a corrected train file')
    with pytest.raises(ValueError, match='corrected oxide train'):
        c.probes(p)
    with pytest.raises(ValueError, match='pair file SHA256'):
        c.checked(p, 'wrong', p)


def test_mixed_gradient_negative_hessian_units_and_bias_cancellation():
    torch = pytest.importorskip('torch')
    # E=1/2 k x^T A x + b^T x. Central FORCE difference is -k A v (not +Hv).
    A = torch.tensor([[2., .5, 0.], [.5, 1., -.4], [0., -.4, 3.]], dtype=torch.float64)
    k = torch.tensor(2., dtype=A.dtype, requires_grad=True)
    v = torch.tensor([[1., -1., 1.], [-1., 1., 1.]], dtype=A.dtype)
    x = torch.tensor([[1., 2., 3.], [-2., 0., 1.]], dtype=A.dtype)
    def force(r):
        r = r.requires_grad_()
        energy = .5*k*((r@A)*r).sum() + 4*r.sum()
        return -torch.autograd.grad(energy, r, create_graph=True)[0]
    D = c.difference(force(x+c.EPS*v), force(x-c.EPS*v))
    torch.testing.assert_close(D, -k*(v@A))
    ref = -3.5*(v@A)
    loss = (D-ref).square().mean()
    torch.testing.assert_close(torch.autograd.grad(loss, k)[0], 2*(k-3.5)*(v@A).square().mean())
    torch.testing.assert_close(c.difference(force(x+c.EPS*v)+7, force(x-c.EPS*v)+7), D)


def test_pinned_cpu_one_native_update_with_paired_auxiliary():
    torch = pytest.importorskip('torch')
    pytest.importorskip('mace')
    if version('mace-torch') != '0.3.16':
        pytest.skip('pinned Stage 0 adapter')
    from e3nn import o3
    from mace import data, modules
    from mace.tools import AtomicNumberTable
    from mace.tools.torch_geometric.batch import Batch
    from mace.tools.train import take_step
    from torch_ema import ExponentialMovingAverage

    torch.manual_seed(5)
    at = Atoms('OTiO', positions=[[0, 0, 0], [1.9, .2, 0], [5, 0, 0]], cell=[14]*3, pbc=True)
    table = AtomicNumberTable([8, 22])
    pos, neg = at.copy(), at.copy()
    v = np.array([[1, -1, 1], [-1, 1, 1], [-1, -1, 1]])
    pos.positions += c.EPS*v; neg.positions -= c.EPS*v
    pair = paired_batch(pos, neg, table)
    # The O--O edge crosses 5 Å: sharing topology across signs would be incorrect.
    sender = pair['edge_index'][0]; assert int((sender < 3).sum()) == int((sender >= 3).sum()) + 2
    dtype = torch.get_default_dtype()
    model = modules.MACE(r_max=5., num_bessel=4, num_polynomial_cutoff=5, max_ell=1,
                         interaction_cls=modules.RealAgnosticResidualInteractionBlock,
                         interaction_cls_first=modules.RealAgnosticInteractionBlock, num_interactions=2,
                         num_elements=2, hidden_irreps=o3.Irreps('4x0e'), MLP_irreps=o3.Irreps('4x0e'),
                         atomic_energies=np.zeros(2), avg_num_neighbors=2., atomic_numbers=table.zs,
                         correlation=2, gate=torch.nn.functional.silu)
    graph = data.AtomicData.from_config(data.Configuration(at.numbers, at.positions,
        dict(energy=.2, forces=np.zeros((3,3)), stress=np.zeros(6)), {}, at.cell.array, tuple(at.pbc)), table, 5.)
    base = Batch.from_data_list([graph] * 4)
    target = torch.tensor(v, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=.001, amsgrad=True)
    ema = ExponentialMovingAverage(model.parameters(), decay=.99)
    native = modules.WeightedEnergyForcesStressLoss(1000., 100., 10.)
    param = next(model.parameters()); before = param.detach().clone()
    # The auxiliary has an independent nonzero mixed coordinate/parameter gradient.
    grad, = torch.autograd.grad(auxiliary(model, pair, target, 'cpu'), param)
    assert torch.isfinite(grad).all() and grad.abs().max() > 0
    def loss_fn(ref, pred):
        return native(ref, pred) + .01 * auxiliary(model, pair, target, 'cpu')
    loss, _ = take_step(model, loss_fn, base, opt, ema, dict(forces=True, virials=False, stress=True), 10., 'cpu')
    assert torch.isfinite(loss) and not torch.equal(before, param) and ema.num_updates == 1
