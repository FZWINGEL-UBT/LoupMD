"""CPU FP64 diagnostic for the pre-Stage-1 training design; not a MACE implementation."""
import torch


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    cell = torch.tensor([[2.3, .4, .2], [.1, 1.8, .3], [-.4, .2, 2.6]])
    volume = cell.det().abs()
    x = torch.tensor([[.3, .5, 1.], [1.1, .8, .3]])
    sender, receiver = [0, 0, 1, 0], [1, 0, 0, 0]
    images = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]])
    pair = [0, 1, 0, 1]  # ordinary pair and periodic self-image, then their reverses
    coeff = torch.tensor([.7, 1.2, -.4, 1.6])  # intentionally unequal directed adjoints
    eye = torch.eye(3)
    q = torch.zeros(15, requires_grad=True)  # two displacements and nine strain entries
    w = torch.tensor([.31, -.16, .48], requires_grad=True)

    def parts(q):
        eps = q[6:].reshape(3, 3)
        return q[:6].reshape(2, 3), (eps + eps.T) / 2

    def vectors(q):
        u, eps = parts(q)
        return (x[receiver] + u[receiver] - x[sender] - u[sender] + images @ cell) @ (eye + eps)

    def energy(w, basis, y, c):
        return (c * torch.sin(w[0] * basis[:, 0] + w[1] * basis[:, 1] + w[2] * y
                              + w[1] * basis[:, 0] * y.square())).sum()

    def direct(w, q):
        d = vectors(q)
        r = d.norm(dim=1)
        return energy(w, torch.stack((r.sin(), r.square()), dim=1), d[:, 2] / r, coeff)

    d0 = vectors(q).detach()
    r0 = d0.norm(dim=1)
    unit = d0 / r0[:, None]
    y0 = unit[:, 2]  # one odd angular component
    dy = (eye[2] - y0[:, None] * unit) / r0[:, None]
    basis0 = torch.stack((r0[:2].sin(), r0[:2].square()), dim=1)
    dbasis = torch.stack((r0[:2].cos(), 2 * r0[:2]), dim=1)

    def affine(w, q, c=coeff):
        u, eps = parts(q)
        dd = u[receiver] - u[sender] + d0 @ eps
        dr = (dd[:2] * unit[:2]).sum(1)
        basis = basis0 + dbasis * dr[:, None]
        return energy(w, basis[pair], y0 + (dy * dd).sum(1), c)

    def outputs(fn, w, q):
        e = fn(w, q)
        g, = torch.autograd.grad(e, q, create_graph=True)
        return e, -g[:6], g[6:] / volume

    target = (torch.tensor(.4), torch.linspace(-.5, .6, 6),
              torch.tensor([.01, .04, -.1, .04, .06, .08, -.1, .08, .03]))
    weight = (1., 2., 3.)

    def loss(fn, w, q):
        return sum(k * (p - t).square().sum() for p, t, k in zip(outputs(fn, w, q), target, weight))

    predicted = outputs(direct, w, q)
    for a, b in zip(outputs(affine, w, q), predicted):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=0)
    a, b, c = [(2 * k * (p - t)).detach() for p, t, k in zip(predicted, target, weight)]
    val, dot = torch.func.jvp(lambda z: direct(w, z), (q,), (torch.cat((-b, c / volume)),))
    grads = [torch.autograd.grad(z, w)[0] for z in (loss(direct, w, q), loss(affine, w, q), a * val + dot)]
    for g in grads[1:]:
        torch.testing.assert_close(g, grads[0], atol=1e-11, rtol=0)
    print('Max parameter gradient error (affine, JVP):', *[(g - grads[0]).abs().max().item() for g in grads[1:]])
    assert torch.autograd.gradcheck(lambda p: loss(affine, p, q), (w,))
    assert torch.autograd.gradgradcheck(lambda p, z: affine(p, z), (w, q))

    g, = torch.autograd.grad(affine(w, q, torch.tensor([0., 1., 0., -.6])), q)
    assert g[:6].abs().max() == 0 and g[6:].abs().max() > 1e-3
    hess = torch.autograd.functional.hessian
    err = (hess(lambda z: direct(w, z), q) - hess(lambda z: affine(w, z), q)).abs().max().item()
    assert err > 1e-2  # first-order affine surrogate is not a physical Hessian oracle
    print('Spatial Hessian difference, self-image max strain derivative:', err, g[6:].abs().max().item())

    class ValueOnly(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z):
            return basis0[0, 0].clone()

        @staticmethod
        def backward(ctx, adjoint):
            return adjoint  # negative control: invents a derivative of a constant

    assert not torch.autograd.gradcheck(ValueOnly.apply, (q[0],), raise_exception=False)


if __name__ == '__main__':
    main()
