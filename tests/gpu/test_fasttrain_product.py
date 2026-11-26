"""Generated product polynomials: values against the coupling tensors, gradcheck and gradgradcheck."""
import pytest
import torch

triton = pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

from loupmd.fasttrain import _product_poly as pp  # noqa: E402
from loupmd.fasttrain.ops import product_poly, product_poly_cm  # noqa: E402


def _dense():
    """Rebuild symmetric dense coefficient tensors from the generated terms."""
    import itertools
    K = pp.K
    c3 = torch.zeros(16, 16, 16, K, dtype=torch.float64)
    c2 = torch.zeros(16, 16, K, dtype=torch.float64)
    c1 = torch.zeros(16, K, dtype=torch.float64)
    for k, m, v in pp.TERMS:
        perms = set(itertools.permutations(m))
        for q in perms:
            if len(m) == 3:
                c3[q + (k,)] += v / len(perms)
            elif len(m) == 2:
                c2[q + (k,)] += v / len(perms)
            else:
                c1[q + (k,)] += v
    return c3.cuda(), c2.cuda(), c1.cuda()


def test_values_match_dense_coupling():
    c3, c2, c1 = _dense()
    x = torch.randn(500, 16, dtype=torch.float64, device='cuda')
    ref = (torch.einsum('nw,ny,ni,wyik->nk', x, x, x, c3) + torch.einsum('nw,ny,wyk->nk', x, x, c2)
           + torch.einsum('nw,wk->nk', x, c1))
    assert torch.allclose(product_poly(x), ref, rtol=1e-12, atol=1e-12)


def test_gradcheck_and_gradgradcheck():
    x = torch.randn(37, 16, dtype=torch.float64, device='cuda', requires_grad=True)
    assert torch.autograd.gradcheck(product_poly, (x,), eps=1e-6, atol=1e-7)
    assert torch.autograd.gradgradcheck(product_poly, (x,), eps=1e-6, atol=1e-7)


def test_component_major_layout():
    x = torch.randn(16, 7, 5, dtype=torch.float64, device='cuda', requires_grad=True)
    P1 = product_poly_cm(x)
    P2 = product_poly(x.permute(1, 2, 0)).permute(2, 0, 1)
    assert torch.allclose(P1, P2, rtol=1e-13, atol=1e-13)
    assert torch.autograd.gradgradcheck(product_poly_cm, (x,), eps=1e-6, atol=1e-7)
