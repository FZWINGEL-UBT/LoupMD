"""Fused edge basis: values, first derivatives and the parameter gradient through forces."""
import math

import pytest
import torch

triton = pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

from loupgym.fasttrain.ops import edge_basis  # noqa: E402

R_MAX, P, NB = 5.0, 5, 8


def _reference(vec, bw, prefactor):
    from e3nn import o3
    Y = o3.spherical_harmonics(list(range(4)), vec, True, 'component')
    r = torch.linalg.norm(vec, dim=-1, keepdim=True)
    bes = prefactor * torch.sin(bw * r) / r
    u = r / R_MAX
    env = (1.0 - ((P + 1.0) * (P + 2.0) / 2.0) * u ** P + P * (P + 2.0) * u ** (P + 1)
           - (P * (P + 1.0) / 2) * u ** (P + 2)) * (r < R_MAX)
    return Y, bes * env


def _vectors(E=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(E, 3, generator=g, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True) * (0.5 + 5.0 * torch.rand(E, 1, generator=g, dtype=torch.float64))
    v[0] = torch.tensor([0.0, 0.0, 1.3])        # on an axis
    v[1] = torch.tensor([5.2, 0.0, 0.0])        # beyond the cutoff
    return v.cuda()


def test_values_and_first_derivatives_match_e3nn():
    vec = _vectors()
    bw = (math.pi / R_MAX * torch.arange(1, NB + 1, dtype=torch.float64)).cuda()
    pre = math.sqrt(2.0 / R_MAX)
    v1 = vec.clone().requires_grad_()
    v2 = vec.clone().requires_grad_()
    Y1, f1 = _reference(v1, bw, pre)
    Y2, f2 = edge_basis(v2, bw, pre, R_MAX, P)
    assert torch.allclose(Y1, Y2, rtol=1e-12, atol=1e-12)
    assert torch.allclose(f1, f2, rtol=1e-12, atol=1e-12)
    wY, wf = torch.randn_like(Y1), torch.randn_like(f1)
    (g1,) = torch.autograd.grad((Y1 * wY).sum() + (f1 * wf).sum(), v1)
    (g2,) = torch.autograd.grad((Y2 * wY).sum() + (f2 * wf).sum(), v2)
    assert torch.allclose(g1, g2, rtol=1e-10, atol=1e-10)


def test_parameter_gradient_through_the_vector_derivative():
    vec = _vectors(seed=1)
    bw = (math.pi / R_MAX * torch.arange(1, NB + 1, dtype=torch.float64)).cuda()
    pre = math.sqrt(2.0 / R_MAX)
    th = torch.randn(16, 8, dtype=torch.float64, device='cuda', requires_grad=True)
    cot = torch.randn_like(vec)
    out = []
    for fn in (lambda v: _reference(v, bw, pre), lambda v: edge_basis(v, bw, pre, R_MAX, P)):
        v = vec.clone().requires_grad_()
        Y, f = fn(v)
        e = torch.tanh(Y @ th) * f             # nonlinear in both features and parameters
        (g,) = torch.autograd.grad(e.sum(), v, create_graph=True)
        (gth,) = torch.autograd.grad((g * cot).sum(), th)
        out.append(gth)
    assert torch.allclose(out[0], out[1], rtol=1e-10, atol=1e-10)
