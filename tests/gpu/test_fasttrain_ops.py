"""Fused convolution slots: values against the materialised reference, and gradcheck/gradgradcheck."""
import pytest
import torch

triton = pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

from loupgym.fasttrain.data import add_csr  # noqa: E402
from loupgym.fasttrain.model import Batch, conv_reference  # noqa: E402
from loupgym.fasttrain.ops import conv_triton  # noqa: E402


def _graph(dtype, N=7, E=40, C=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    snd = torch.randint(0, N, (E,), generator=g)
    rcv = torch.randint(0, N - 1, (E,), generator=g)     # node N-1 receives nothing
    snd[0] = rcv[0]                                       # a periodic self-image edge
    order = torch.sort(rcv, stable=True).indices
    snd, rcv = snd[order].cuda(), rcv[order].cuda()
    b = Batch(vectors=None, sender=snd, receiver=rcv, species=torch.zeros(N, dtype=torch.long).cuda(),
              species_bounds=(0, N), node_graph=None, edge_graph=None, num_graphs=1, n_atoms=None,
              volume=None, e_ref=None, f_ref=None, s_ref=None, w_energy=None, w_forces=None,
              w_stress=None, core_e=None, core_f=None, core_s=None)
    add_csr(b)
    x = torch.randn(N, C, generator=g, dtype=dtype).cuda().requires_grad_()
    R = torch.randn(E, 4, C, generator=g, dtype=dtype).cuda().requires_grad_()
    Y = torch.randn(E, 16, generator=g, dtype=dtype).cuda().requires_grad_()
    return b, x, R, Y, N


def test_forward_and_first_derivatives_match_reference():
    b, x, R, Y, N = _graph(torch.float64)
    a1 = conv_reference(x, R, Y, b, N)
    a2 = conv_triton(x, R, Y, b, N)
    assert torch.allclose(a1, a2, rtol=1e-12, atol=1e-12)
    assert float(a2[N - 1].abs().max()) == 0.0
    w = torch.randn_like(a1)
    g1 = torch.autograd.grad((a1 * w).sum(), (x, R, Y), create_graph=True)
    g2 = torch.autograd.grad((a2 * w).sum(), (x, R, Y), create_graph=True)
    for u, v in zip(g1, g2):
        assert torch.allclose(u, v, rtol=1e-12, atol=1e-12)
    # mixed second derivative through all first derivatives with arbitrary cotangents
    vs = [torch.randn_like(t) for t in g1]
    h1 = torch.autograd.grad(sum((u * v).sum() for u, v in zip(g1, vs)), (x, R, Y))
    h2 = torch.autograd.grad(sum((u * v).sum() for u, v in zip(g2, vs)), (x, R, Y))
    for u, v in zip(h1, h2):
        assert torch.allclose(u, v, rtol=1e-11, atol=1e-11)


def test_gradcheck_and_gradgradcheck():
    b, x, R, Y, N = _graph(torch.float64, N=5, E=12, C=32)
    fn = lambda x, R, Y: conv_triton(x, R, Y, b, N)
    assert torch.autograd.gradcheck(fn, (x, R, Y), eps=1e-6, atol=1e-7)
    assert torch.autograd.gradgradcheck(fn, (x, R, Y), eps=1e-6, atol=1e-7)


def test_float32_matches_float64_reference():
    b, x, R, Y, N = _graph(torch.float64, N=50, E=2000)
    ref = conv_reference(x, R, Y, b, N)
    a = conv_triton(x.float(), R.float(), Y.float(), b, N)
    assert float((a.double() - ref).abs().max() / ref.abs().max()) < 1e-5


def test_mnc_layout_matches_reference():
    b, x, R, Y, N = _graph(torch.float64)
    b.mnc = True
    a1 = conv_reference(x, R, Y, b, N).permute(2, 0, 1)
    a2 = conv_triton(x, R, Y, b, N)
    assert a2.shape == a1.shape
    assert torch.allclose(a1, a2, rtol=1e-12, atol=1e-12)
    w = torch.randn_like(a1)
    g1 = torch.autograd.grad((a1 * w).sum(), (x, R, Y), create_graph=True)
    g2 = torch.autograd.grad((a2 * w).sum(), (x, R, Y), create_graph=True)
    for u, v in zip(g1, g2):
        assert torch.allclose(u, v, rtol=1e-12, atol=1e-12)
    vs = [torch.randn_like(t) for t in g1]
    h1 = torch.autograd.grad(sum((u * v).sum() for u, v in zip(g1, vs)), (x, R, Y))
    h2 = torch.autograd.grad(sum((u * v).sum() for u, v in zip(g2, vs)), (x, R, Y))
    for u, v in zip(h1, h2):
        assert torch.allclose(u, v, rtol=1e-11, atol=1e-11)
