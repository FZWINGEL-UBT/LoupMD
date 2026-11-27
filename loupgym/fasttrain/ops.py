"""Fused scalar-sender equivariant convolution with derivatives of any order (Triton).

All quantities are slots of one multilinear form over directed edges e = (s -> r):

    T(G, R, Y, X) = sum_e sum_c sum_lm G[r, c, lm] R[e, l(lm), c] Y[e, lm] X[s, c]

with node tensors G [N, C, 16] and X [N, C], and edge tensors R [E, 4, C] (radial weights per
degree) and Y [E, 16] (spherical harmonics). The convolution is dT/dG; every derivative of every
slot is again a slot of T with one argument replaced by the incoming cotangent. Three kernels
cover all slots without materialising per-edge messages:

* ``node_g``: dT/dG for one or two (R, Y) pairs, receiver-owned CSR rows, no atomics;
* ``node_x``: dT/dX for one or two pairs, sender-owned rows through a sender permutation;
* ``edge``:   (dT/dR, dT/dY), receiver-owned so G[r] is loaded once per row.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

NLM = 16


@triton.jit
def _l_of_lm(lm):
    # degrees of the 16 components: 0 | 1 1 1 | 2 x5 | 3 x7
    return (lm >= 1).to(tl.int32) + (lm >= 4).to(tl.int32) + (lm >= 9).to(tl.int32)


@triton.jit
def _node_g_kernel(out, ptr, snd, PIDX, X, R1, Y1, R2, Y2, SN, SC, SM, C: tl.constexpr, TWO: tl.constexpr,
                   BC: tl.constexpr, PAIRED: tl.constexpr):
    i = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BC + tl.arange(0, BC)
    lm = tl.arange(0, 16)
    l = _l_of_lm(lm)
    start = tl.load(ptr + i)
    stop = tl.load(ptr + i + 1)
    acc = tl.zeros((BC, 16), dtype=out.dtype.element_ty)
    for e in range(start, stop):
        s = tl.load(snd + e)
        if PAIRED:
            re = tl.load(PIDX + e)
        else:
            re = e
        x = tl.load(X + s * C + c)
        r = tl.load(R1 + re * 4 * C + l[None, :] * C + c[:, None])
        y = tl.load(Y1 + e * 16 + lm)
        t = r * y[None, :]
        if TWO:
            r2 = tl.load(R2 + re * 4 * C + l[None, :] * C + c[:, None])
            y2 = tl.load(Y2 + e * 16 + lm)
            t += r2 * y2[None, :]
        acc += t * x[:, None]
    tl.store(out + i * SN + c[:, None] * SC + lm[None, :] * SM, acc)


@triton.jit
def _node_x_kernel(out, ptr, perm, rcv, PIDX, G, R1, Y1, R2, Y2, SN, SC, SM, C: tl.constexpr,
                   TWO: tl.constexpr, BC: tl.constexpr, PAIRED: tl.constexpr):
    j = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BC + tl.arange(0, BC)
    lm = tl.arange(0, 16)
    l = _l_of_lm(lm)
    start = tl.load(ptr + j)
    stop = tl.load(ptr + j + 1)
    acc = tl.zeros((BC, 16), dtype=out.dtype.element_ty)
    for k in range(start, stop):
        e = tl.load(perm + k)
        r = tl.load(rcv + e)
        if PAIRED:
            re = tl.load(PIDX + e)
        else:
            re = e
        g = tl.load(G + r * SN + c[:, None] * SC + lm[None, :] * SM)
        w = tl.load(R1 + re * 4 * C + l[None, :] * C + c[:, None])
        y = tl.load(Y1 + e * 16 + lm)
        t = w * y[None, :]
        if TWO:
            w2 = tl.load(R2 + re * 4 * C + l[None, :] * C + c[:, None])
            y2 = tl.load(Y2 + e * 16 + lm)
            t += w2 * y2[None, :]
        acc += g * t
    tl.store(out + j * C + c, tl.sum(acc, axis=1))


@triton.jit
def _edge_kernel(outR, outY, ptr, snd, G, R, Y, X, SN, SC, SM, C: tl.constexpr, NEED_R: tl.constexpr,
                 NEED_Y: tl.constexpr):
    i = tl.program_id(0)
    c = tl.arange(0, C)
    lm = tl.arange(0, 16)
    l = _l_of_lm(lm)
    start = tl.load(ptr + i)
    stop = tl.load(ptr + i + 1)
    g = tl.load(G + i * SN + c[:, None] * SC + lm[None, :] * SM)
    for e in range(start, stop):
        s = tl.load(snd + e)
        x = tl.load(X + s * C + c)
        if NEED_R:
            y = tl.load(Y + e * 16 + lm)
            t = g * y[None, :] * x[:, None]
            for ll in tl.static_range(4):
                v = tl.sum(tl.where(l[None, :] == ll, t, 0.0), axis=1)
                tl.store(outR + e * 4 * C + ll * C + c, v)
        if NEED_Y:
            w = tl.load(R + e * 4 * C + l[None, :] * C + c[:, None])
            v = tl.sum(g * w * x[:, None], axis=0)
            tl.store(outY + e * 16 + lm, v)


@triton.jit
def _edge2_kernel(outR, outY, rcv, snd, PIDX, G, R, Y, X, E, SN, SC, SM, C: tl.constexpr, EB: tl.constexpr,
                  NEED_R: tl.constexpr, NEED_Y: tl.constexpr, PAIRED: tl.constexpr):
    """Edge-parallel slots: outR[e,l,c] = sum_{m in l} G[r,c,m] Y[e,m] X[s,c];
    outY[e,m] = sum_c G[r,c,m] R[e,l(m),c] X[s,c]."""
    e = tl.program_id(0) * EB + tl.arange(0, EB)
    em = e < E
    c = tl.arange(0, C)
    m2 = em[:, None]
    r = tl.load(rcv + e, mask=em, other=0)
    s = tl.load(snd + e, mask=em, other=0)
    if PAIRED:
        re = tl.load(PIDX + e, mask=em, other=0)
    else:
        re = e
    x = tl.load(X + s[:, None] * C + c[None, :], mask=m2, other=0.0)
    for l in tl.static_range(4):
        acc = tl.zeros((EB, C), dtype=x.dtype)
        if NEED_Y:
            rx = tl.load(R + re[:, None] * 4 * C + l * C + c[None, :], mask=m2, other=0.0) * x
        for mm in tl.static_range(2 * l + 1):
            m = l * l + mm
            g = tl.load(G + r[:, None] * SN + c[None, :] * SC + m * SM, mask=m2, other=0.0)
            if NEED_R:
                y = tl.load(Y + e * 16 + m, mask=em, other=0.0)
                acc += g * y[:, None]
            if NEED_Y:
                tl.store(outY + e * 16 + m, tl.sum(g * rx, axis=1), mask=em)
        if NEED_R:
            tl.store(outR + e[:, None] * 4 * C + l * C + c[None, :], acc * x, mask=m2)


@triton.jit
def _edge_pair_kernel(OUT1, OUT2, pe0, pe1, rcv, snd, G, Y1, Y2, X, P, SN, SC, SM, C: tl.constexpr,
                      EB: tl.constexpr, TWO: tl.constexpr):
    """R-slots summed over both directions of each edge pair p = (e0, e1):
    OUTk[p,l,c] = sum_{e in p} sum_{m in l} G[r_e,c,m] Yk[e,m] X[s_e,c]."""
    p = tl.program_id(0) * EB + tl.arange(0, EB)
    pm = p < P
    c = tl.arange(0, C)
    m2 = pm[:, None]
    for l in tl.static_range(4):
        a1 = tl.zeros((EB, C), dtype=OUT1.dtype.element_ty)
        a2 = tl.zeros((EB, C), dtype=OUT1.dtype.element_ty)
        for side in tl.static_range(2):
            if side == 0:
                e = tl.load(pe0 + p, mask=pm, other=0)
            else:
                e = tl.load(pe1 + p, mask=pm, other=0)
            r = tl.load(rcv + e, mask=pm, other=0)
            sn = tl.load(snd + e, mask=pm, other=0)
            x = tl.load(X + sn[:, None] * C + c[None, :], mask=m2, other=0.0)
            i1 = tl.zeros((EB, C), dtype=OUT1.dtype.element_ty)
            i2 = tl.zeros((EB, C), dtype=OUT1.dtype.element_ty)
            for mm in tl.static_range(2 * l + 1):
                m = l * l + mm
                g = tl.load(G + r[:, None] * SN + c[None, :] * SC + m * SM, mask=m2, other=0.0)
                i1 += g * tl.load(Y1 + e * 16 + m, mask=pm, other=0.0)[:, None]
                if TWO:
                    i2 += g * tl.load(Y2 + e * 16 + m, mask=pm, other=0.0)[:, None]
            a1 += i1 * x
            if TWO:
                a2 += i2 * x
        tl.store(OUT1 + p[:, None] * 4 * C + l * C + c[None, :], a1, mask=m2)
        if TWO:
            tl.store(OUT2 + p[:, None] * 4 * C + l * C + c[None, :], a2, mask=m2)


def _paired(meta):
    return getattr(meta, 'pidx', None) is not None


def _pidx(meta):
    return meta.pidx if _paired(meta) else meta.sender


def _k_edge_pairs(meta, G, X, Y1, Y2=None):
    """R-slot outputs per edge pair [P, 4, C] (one or two Y cotangents)."""
    mnc = _mnc(meta)
    N, C = _gshape(G, mnc)
    P = meta.pe0.shape[0]
    out1 = torch.empty(P, 4, C, device=G.device, dtype=G.dtype)
    two = Y2 is not None
    out2 = torch.empty(P, 4, C, device=G.device, dtype=G.dtype) if two else out1
    if P > 0:
        _edge_pair_kernel[(triton.cdiv(P, EB_EDGE),)](out1, out2, meta.pe0, meta.pe1, meta.receiver, meta.sender, G,
                                                      Y1.contiguous(), (Y2 if two else Y1).contiguous(),
                                                      X.contiguous(), P, *_gstrides(G, mnc), C=C, EB=EB_EDGE,
                                                      TWO=two, num_warps=4)
    return (out1, out2) if two else out1


EDGE_PARALLEL = True
EB_EDGE = 32
BC = 32


def _mnc(meta):
    return bool(getattr(meta, 'mnc', False))


def _gstrides(G, mnc):
    """(SN, SC, SM) of a node tensor in either layout."""
    if mnc:
        sm, sn, sc = G.stride()
    else:
        sn, sc, sm = G.stride()
    return sn, sc, sm


def _gshape(G, mnc):
    """(N, C) of a node tensor in either layout."""
    return (G.shape[1], G.shape[2]) if mnc else (G.shape[0], G.shape[1])


def _k_node_g(meta, R1, Y1, X, R2=None, Y2=None):
    N, C = X.shape
    mnc = _mnc(meta)
    out = (torch.empty(NLM, N, C, device=X.device, dtype=X.dtype) if mnc
           else torch.empty(N, C, NLM, device=X.device, dtype=X.dtype))
    two = R2 is not None
    _node_g_kernel[(N, C // BC)](out, meta.receiver_ptr, meta.sender, _pidx(meta), X.contiguous(), R1.contiguous(),
                                Y1.contiguous(), (R2 if two else R1).contiguous(),
                                (Y2 if two else Y1).contiguous(), *_gstrides(out, mnc), C=C, TWO=two, BC=BC,
                                PAIRED=_paired(meta), num_warps=2)
    return out


def _k_node_x(meta, G, R1, Y1, R2=None, Y2=None):
    mnc = _mnc(meta)
    G = G.contiguous()
    N, C = _gshape(G, mnc)
    out = torch.empty(N, C, device=G.device, dtype=G.dtype)
    two = R2 is not None
    _node_x_kernel[(N, C // BC)](out, meta.sender_ptr, meta.sender_perm, meta.receiver, _pidx(meta), G,
                                R1.contiguous(), Y1.contiguous(), (R2 if two else R1).contiguous(),
                                (Y2 if two else Y1).contiguous(), *_gstrides(G, mnc), C=C, TWO=two, BC=BC,
                                PAIRED=_paired(meta), num_warps=2)
    return out


def _k_edge(meta, G, R, Y, X, need_r, need_y):
    mnc = _mnc(meta)
    G = G.contiguous()
    N, C = _gshape(G, mnc)
    E = meta.sender.shape[0]
    Rc = (R if R is not None else G).contiguous()
    Yc = (Y if Y is not None else G).contiguous()
    if _paired(meta):
        outR = _k_edge_pairs(meta, G, X, Yc) if need_r else None
        outY = None
        if need_y:
            outY = torch.empty(E, NLM, device=G.device, dtype=G.dtype)
            if E > 0:
                _edge2_kernel[(triton.cdiv(E, EB_EDGE),)](G, outY, meta.receiver, meta.sender, meta.pidx, G, Rc, Yc,
                                                          X.contiguous(), E, *_gstrides(G, mnc), C=C, EB=EB_EDGE,
                                                          NEED_R=False, NEED_Y=True, PAIRED=True, num_warps=4)
        return outR, outY
    outR = torch.empty(E, 4, C, device=G.device, dtype=G.dtype) if need_r else G
    outY = torch.empty(E, NLM, device=G.device, dtype=G.dtype) if need_y else G
    if EDGE_PARALLEL:
        if E > 0:
            _edge2_kernel[(triton.cdiv(E, EB_EDGE),)](outR, outY, meta.receiver, meta.sender, meta.sender, G, Rc, Yc,
                                                      X.contiguous(), E, *_gstrides(G, mnc), C=C, EB=EB_EDGE,
                                                      NEED_R=need_r, NEED_Y=need_y, PAIRED=False, num_warps=4)
    else:
        _edge_kernel[(N,)](outR, outY, meta.receiver_ptr, meta.sender, G, Rc, Yc,
                           X.contiguous(), *_gstrides(G, mnc), C=C, NEED_R=need_r, NEED_Y=need_y, num_warps=4)
    return (outR if need_r else None), (outY if need_y else None)


def _pairs(*items):
    """Drop (R, Y) pairs that are missing (None cotangents)."""
    return [(r, y) for r, y in items if r is not None and y is not None]


def node_g(meta, pairs, X):
    pairs = _pairs(*pairs)
    if not pairs or X is None:
        return None
    if len(pairs) == 1:
        return _NodeG.apply(meta, X, pairs[0][0], pairs[0][1], None, None)
    return _NodeG.apply(meta, X, pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])


def node_x(meta, G, pairs):
    pairs = _pairs(*pairs)
    if not pairs or G is None:
        return None
    if len(pairs) == 1:
        return _NodeX.apply(meta, G, pairs[0][0], pairs[0][1], None, None)
    return _NodeX.apply(meta, G, pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])


def edge(meta, G, R, Y, X):
    """(dT/dR with this Y, dT/dY with this R); either side may be skipped by passing None."""
    if G is None or X is None or (R is None and Y is None):
        return None, None
    return _Edge.apply(meta, G, R, Y, X)


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


class _NodeG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, meta, X, R1, Y1, R2, Y2):
        ctx.meta = meta
        ctx.save_for_backward(X, R1, Y1, R2, Y2)
        return _k_node_g(meta, R1, Y1, X, R2, Y2)

    @staticmethod
    def backward(ctx, gG):
        X, R1, Y1, R2, Y2 = ctx.saved_tensors
        meta = ctx.meta
        need = ctx.needs_input_grad
        dX = node_x(meta, gG, [(R1, Y1), (R2, Y2)]) if need[1] else None
        dR1 = dY1 = dR2 = dY2 = None
        if need[2] or need[3]:
            dR1, dY1 = edge(meta, gG, R1 if need[3] else None, Y1 if need[2] else None, X)
        if R2 is not None and (need[4] or need[5]):
            dR2, dY2 = edge(meta, gG, R2 if need[5] else None, Y2 if need[4] else None, X)
        return None, dX, dR1, dY1, dR2, dY2


class _NodeX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, meta, G, R1, Y1, R2, Y2):
        ctx.meta = meta
        ctx.save_for_backward(G, R1, Y1, R2, Y2)
        return _k_node_x(meta, G, R1, Y1, R2, Y2)

    @staticmethod
    def backward(ctx, gX):
        G, R1, Y1, R2, Y2 = ctx.saved_tensors
        meta = ctx.meta
        need = ctx.needs_input_grad
        dG = node_g(meta, [(R1, Y1), (R2, Y2)], gX) if need[1] else None
        dR1 = dY1 = dR2 = dY2 = None
        if need[2] or need[3]:
            dR1, dY1 = edge(meta, G, R1 if need[3] else None, Y1 if need[2] else None, gX)
        if R2 is not None and (need[4] or need[5]):
            dR2, dY2 = edge(meta, G, R2 if need[5] else None, Y2 if need[4] else None, gX)
        return None, dG, dR1, dY1, dR2, dY2


class _Edge(torch.autograd.Function):
    """Outputs (A, B) = (dT/dR given Y, dT/dY given R). A needs Y; B needs R."""

    @staticmethod
    def forward(ctx, meta, G, R, Y, X):
        ctx.meta = meta
        ctx.save_for_backward(G, R, Y, X)
        A, B = _k_edge(meta, G, R, Y, X, Y is not None, R is not None)
        return A, B

    @staticmethod
    def backward(ctx, gA, gB):
        G, R, Y, X = ctx.saved_tensors
        meta = ctx.meta
        need = ctx.needs_input_grad
        if Y is None:
            gA = None
        if R is None:
            gB = None
        dG = node_g(meta, [(gA, Y), (R, gB)], X) if need[1] else None
        dX = node_x(meta, G, [(gA, Y), (R, gB)]) if need[4] else None
        # dA/dY contracts gA in the R slot; dB/dR contracts gB in the Y slot.
        dR = dY = None
        if (need[2] and gB is not None) or (need[3] and gA is not None):
            dR, dY = edge(meta, G, gA if need[3] else None, gB if need[2] else None, X)
        return None, dG, dR, dY, dX


def conv_triton(x, R, Y, b, N):
    """Drop-in for :func:`loupmd.fasttrain.model.conv_reference` (needs ``add_csr`` on the batch)."""
    return node_g(b, [(R, Y)], x)


# --------------------------------------------------------------------------- edge basis
_PR_CACHE = {}


def edge_basis_raw(vec, bw, prefactor, r_max, p):
    """(Y [E,16], ef [E,NB], d ef/dr [E,NB], unit vectors U [E,3], dY/dvec JY [E,16,3]); no autograd."""
    from ._edge_basis import edge_basis_kernel
    vec = vec.detach().contiguous()
    E = vec.shape[0]
    nb = bw.numel()
    kw = dict(device=vec.device, dtype=vec.dtype)
    Y = torch.empty(E, NLM, **kw)
    JY = torch.empty(E, NLM, 3, **kw)
    EF = torch.empty(E, nb, **kw)
    DEF = torch.empty(E, nb, **kw)
    U = torch.empty(E, 3, **kw)
    key = (float(prefactor), float(r_max), vec.device, vec.dtype)
    pr = _PR_CACHE.get(key)
    if pr is None:
        pr = _PR_CACHE[key] = torch.tensor([float(prefactor), float(r_max)], **kw)
    BLOCK = 128
    if E > 0:
        edge_basis_kernel[(triton.cdiv(E, BLOCK),)](vec, Y, JY, EF, DEF, U, bw.detach().to(vec.dtype).contiguous(),
                                                     pr, E, P=p, NB=nb, BLOCK=BLOCK, num_warps=4)
    return Y, EF, DEF, U, JY


class _EdgeBasis(torch.autograd.Function):
    """Spherical harmonics Y [E, 16] and enveloped Bessels ef [E, NB] of fixed edge vectors.

    Training needs only the first derivative of the edge features with respect to the vectors
    (forces and stress); the parameter gradient of the loss differentiates that first derivative
    through the features' cotangents only. The backward is therefore the linear map
    g_vec = J_Y^T gY + (dEf/dr . gEf) u with saved Jacobians: exact for every derivative the
    trainer takes, with no second derivative of the basis.
    """

    @staticmethod
    def forward(ctx, vec, bw, prefactor, r_max, p):
        Y, EF, DEF, U, JY = edge_basis_raw(vec, bw, prefactor, r_max, p)
        ctx.save_for_backward(JY, DEF, U)
        return Y, EF

    @staticmethod
    def backward(ctx, gY, gEf):
        JY, DEF, U = ctx.saved_tensors
        g = None
        if gY is not None:
            g = torch.bmm(gY.unsqueeze(1), JY).squeeze(1)
        if gEf is not None:
            t = (gEf * DEF).sum(-1, keepdim=True) * U
            g = t if g is None else g + t
        return g, None, None, None, None


def edge_basis(vec, bw, prefactor, r_max, p):
    return _EdgeBasis.apply(vec, bw, prefactor, r_max, p)


# --------------------------------------------------------------------------- product polynomials
from torch.autograd.function import once_differentiable  # noqa: E402

_PBLOCK = 128


def _poly_k():
    from . import _product_poly as pp
    return pp.K


def _poly_launch(name, rows, cm, *tensors):
    """Launch a generated polynomial kernel; ``cm``: component-major ([16, rows] / [K, rows]) layout."""
    from . import _product_poly as pp
    K = pp.K
    strides = (1, rows, 1, rows) if cm else (16, 1, K, 1)
    if rows > 0:
        getattr(pp, name)[(triton.cdiv(rows, _PBLOCK),)](*tensors, rows, *strides, BLOCK=_PBLOCK, num_warps=4)


def _alloc(rows, width, cm, like):
    shape = (width, rows) if cm else (rows, width)
    return torch.empty(shape, device=like.device, dtype=like.dtype)


class _Poly(torch.autograd.Function):
    """P of x: the correlation-3 invariant polynomials. Row-major x [rows, 16] -> P [rows, K];
    component-major x [16, rows] -> P [K, rows]."""

    @staticmethod
    def forward(ctx, x, cm):
        x = x.contiguous()
        rows = x.shape[1] if cm else x.shape[0]
        P = _alloc(rows, _poly_k(), cm, x)
        _poly_launch('poly_fwd_kernel', rows, cm, x, P)
        ctx.save_for_backward(x)
        ctx.cm = cm
        return P

    @staticmethod
    def backward(ctx, gP):
        (x,) = ctx.saved_tensors
        return _PolyVjp.apply(x, gP, ctx.cm), None


class _PolyVjp(torch.autograd.Function):
    """sum_k V_k grad P_k(x); its own derivatives are the HVP and the JVP (no third derivative)."""

    @staticmethod
    def forward(ctx, x, v, cm):
        x, v = x.contiguous(), v.contiguous()
        rows = x.shape[1] if cm else x.shape[0]
        G = torch.empty_like(x)
        _poly_launch('poly_vjp_kernel', rows, cm, x, v, G)
        ctx.save_for_backward(x, v)
        ctx.cm = cm
        return G

    @staticmethod
    @once_differentiable
    def backward(ctx, gG):
        x, v = ctx.saved_tensors
        cm = ctx.cm
        rows = x.shape[1] if cm else x.shape[0]
        gG = gG.contiguous()
        gx = gv = None
        if ctx.needs_input_grad[0]:
            gx = torch.empty_like(x)
            _poly_launch('poly_hvp_kernel', rows, cm, x, v, gG, gx)
        if ctx.needs_input_grad[1]:
            gv = torch.empty_like(v)
            _poly_launch('poly_jvp_kernel', rows, cm, x, gG, gv)
        return gx, gv, None


def product_poly(x):
    """x [..., 16] -> P [..., K]."""
    shp = x.shape
    return _Poly.apply(x.reshape(-1, shp[-1]), False).view(*shp[:-1], _poly_k())


def product_poly_cm(x):
    """Component-major: x [16, ...] -> P [K, ...]."""
    shp = x.shape
    return _Poly.apply(x.reshape(shp[0], -1), True).view(_poly_k(), *shp[1:])


# --------------------------------------------------------------------------- radial MLP tangent
@triton.jit
def _silu_t_fwd_kernel(ZZ, HH, ACT, half, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    act = tl.load(ACT)
    z = tl.load(ZZ + i, mask=mask, other=0.0)
    dz = tl.load(ZZ + half + i, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-z))
    d1 = s * (1.0 + z * (1.0 - s))
    tl.store(HH + i, act * z * s, mask=mask)
    tl.store(HH + half + i, act * d1 * dz, mask=mask)


@triton.jit
def _silu_t_bwd_kernel(ZZ, GH, GZ, ACT, half, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    act = tl.load(ACT)
    z = tl.load(ZZ + i, mask=mask, other=0.0)
    dz = tl.load(ZZ + half + i, mask=mask, other=0.0)
    gh = tl.load(GH + i, mask=mask, other=0.0)
    gdh = tl.load(GH + half + i, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-z))
    d1 = s * (1.0 + z * (1.0 - s))
    d2 = s * (1.0 - s) * (2.0 + z * (1.0 - 2.0 * s))
    tl.store(GZ + i, act * (d1 * gh + d2 * dz * gdh), mask=mask)
    tl.store(GZ + half + i, act * d1 * gdh, mask=mask)


_ACT_CACHE = {}


def _act_tensor(act, ref):
    key = (float(act), ref.device, ref.dtype)
    t = _ACT_CACHE.get(key)
    if t is None:
        t = _ACT_CACHE[key] = torch.tensor([float(act)], device=ref.device, dtype=ref.dtype)
    return t


class _SiluTangent(torch.autograd.Function):
    """[z; dz] (stacked rows) -> [act silu(z); act silu'(z) dz]; differentiated once only."""

    @staticmethod
    def forward(ctx, zz, act):
        zz = zz.contiguous()
        hh = torch.empty_like(zz)
        half = zz.numel() // 2
        a = _act_tensor(act, zz)
        if half > 0:
            _silu_t_fwd_kernel[(triton.cdiv(half, 1024),)](zz, hh, a, half, half, BLOCK=1024, num_warps=4)
        ctx.save_for_backward(zz)
        ctx.act = act
        return hh

    @staticmethod
    @once_differentiable
    def backward(ctx, gh):
        (zz,) = ctx.saved_tensors
        gh = gh.contiguous()
        gz = torch.empty_like(zz)
        half = zz.numel() // 2
        a = _act_tensor(ctx.act, zz)
        if half > 0:
            _silu_t_bwd_kernel[(triton.cdiv(half, 1024),)](zz, gh, gz, a, half, half, BLOCK=1024, num_warps=4)
        return gz, None


def silu_tangent(zz, act):
    return _SiluTangent.apply(zz, act)


@triton.jit
def _silu_t2_fwd_kernel(Z, DZ, H, DH, ACT, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    act = tl.load(ACT)
    z = tl.load(Z + i, mask=mask, other=0.0)
    dz = tl.load(DZ + i, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-z))
    d1 = s * (1.0 + z * (1.0 - s))
    tl.store(H + i, act * z * s, mask=mask)
    tl.store(DH + i, act * d1 * dz, mask=mask)


@triton.jit
def _silu_t2_bwd_kernel(Z, DZ, GH, GDH, GZ, GDZ, ACT, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    act = tl.load(ACT)
    z = tl.load(Z + i, mask=mask, other=0.0)
    dz = tl.load(DZ + i, mask=mask, other=0.0)
    gh = tl.load(GH + i, mask=mask, other=0.0)
    gdh = tl.load(GDH + i, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-z))
    d1 = s * (1.0 + z * (1.0 - s))
    d2 = s * (1.0 - s) * (2.0 + z * (1.0 - 2.0 * s))
    tl.store(GZ + i, act * (d1 * gh + d2 * dz * gdh), mask=mask)
    tl.store(GDZ + i, act * d1 * gdh, mask=mask)


class _SiluTangent2(torch.autograd.Function):
    """(z, dz) -> (act silu(z), act silu'(z) dz); differentiated once only."""

    @staticmethod
    def forward(ctx, z, dz, act):
        z, dz = z.contiguous(), dz.contiguous()
        h, dh = torch.empty_like(z), torch.empty_like(z)
        n = z.numel()
        a = _act_tensor(act, z)
        if n > 0:
            _silu_t2_fwd_kernel[(triton.cdiv(n, 1024),)](z, dz, h, dh, a, n, BLOCK=1024, num_warps=4)
        ctx.save_for_backward(z, dz)
        ctx.act = act
        return h, dh

    @staticmethod
    @once_differentiable
    def backward(ctx, gh, gdh):
        z, dz = ctx.saved_tensors
        gh = torch.zeros_like(z) if gh is None else gh.contiguous()
        gdh = torch.zeros_like(z) if gdh is None else gdh.contiguous()
        gz, gdz = torch.empty_like(z), torch.empty_like(z)
        n = z.numel()
        a = _act_tensor(ctx.act, z)
        if n > 0:
            _silu_t2_bwd_kernel[(triton.cdiv(n, 1024),)](z, dz, gh, gdh, gz, gdz, a, n, BLOCK=1024, num_warps=4)
        return gz, gdz, None


def silu_tangent2(z, dz, act):
    return _SiluTangent2.apply(z, dz, act)


# --------------------------------------------------------------------------- edge vector gradient
@triton.jit
def _edge_vec_fwd_kernel(GR, GY, U, JY, OUT, E, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = e < E
    gr = tl.load(GR + e, mask=mask, other=0.0)
    gx = gr * tl.load(U + e * 3 + 0, mask=mask, other=0.0)
    gy = gr * tl.load(U + e * 3 + 1, mask=mask, other=0.0)
    gz = gr * tl.load(U + e * 3 + 2, mask=mask, other=0.0)
    for m in tl.static_range(16):
        w = tl.load(GY + e * 16 + m, mask=mask, other=0.0)
        gx += w * tl.load(JY + e * 48 + m * 3 + 0, mask=mask, other=0.0)
        gy += w * tl.load(JY + e * 48 + m * 3 + 1, mask=mask, other=0.0)
        gz += w * tl.load(JY + e * 48 + m * 3 + 2, mask=mask, other=0.0)
    tl.store(OUT + e * 3 + 0, gx, mask=mask)
    tl.store(OUT + e * 3 + 1, gy, mask=mask)
    tl.store(OUT + e * 3 + 2, gz, mask=mask)


@triton.jit
def _edge_vec_bwd_kernel(W, U, JY, DGR, DGY, E, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = e < E
    wx = tl.load(W + e * 3 + 0, mask=mask, other=0.0)
    wy = tl.load(W + e * 3 + 1, mask=mask, other=0.0)
    wz = tl.load(W + e * 3 + 2, mask=mask, other=0.0)
    d = (wx * tl.load(U + e * 3 + 0, mask=mask, other=0.0) + wy * tl.load(U + e * 3 + 1, mask=mask, other=0.0)
         + wz * tl.load(U + e * 3 + 2, mask=mask, other=0.0))
    tl.store(DGR + e, d, mask=mask)
    for m in tl.static_range(16):
        v = (wx * tl.load(JY + e * 48 + m * 3 + 0, mask=mask, other=0.0)
             + wy * tl.load(JY + e * 48 + m * 3 + 1, mask=mask, other=0.0)
             + wz * tl.load(JY + e * 48 + m * 3 + 2, mask=mask, other=0.0))
        tl.store(DGY + e * 16 + m, v, mask=mask)


class _EdgeVec(torch.autograd.Function):
    """dE/dvec = gr u + JY^T gY from the r-derivative gr [E] and the harmonic cotangent gY [E, 16]."""

    @staticmethod
    def forward(ctx, gr, gY, U, JY):
        gr, gY = gr.contiguous(), gY.contiguous()
        E = gr.shape[0]
        out = torch.empty(E, 3, device=gr.device, dtype=gr.dtype)
        if E > 0:
            _edge_vec_fwd_kernel[(triton.cdiv(E, 256),)](gr, gY, U, JY, out, E, BLOCK=256, num_warps=4)
        ctx.save_for_backward(U, JY)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, w):
        U, JY = ctx.saved_tensors
        w = w.contiguous()
        E = w.shape[0]
        dgr = torch.empty(E, device=w.device, dtype=w.dtype)
        dgY = torch.empty(E, 16, device=w.device, dtype=w.dtype)
        if E > 0:
            _edge_vec_bwd_kernel[(triton.cdiv(E, 256),)](w, U, JY, dgr, dgY, E, BLOCK=256, num_warps=4)
        return dgr, dgY, None, None


def edge_vec(gr, gY, U, JY):
    return _EdgeVec.apply(gr, gY, U.contiguous(), JY.contiguous())


# --------------------------------------------------------------------------- both radial MLPs at once
@triton.jit
def _silu_tb_fwd_kernel(Z, H, ACT, HALF, n, BLOCK: tl.constexpr):
    """Z, H [B, 2, HALF] (value rows, tangent rows): H = (act silu(z), act silu'(z) dz)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    base = (i // HALF) * 2 * HALF + i % HALF
    act = tl.load(ACT)
    z = tl.load(Z + base, mask=mask, other=0.0)
    dz = tl.load(Z + base + HALF, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-z))
    d1 = s * (1.0 + z * (1.0 - s))
    tl.store(H + base, act * z * s, mask=mask)
    tl.store(H + base + HALF, act * d1 * dz, mask=mask)


@triton.jit
def _silu_tb_bwd_kernel(Z, GH, GZ, ACT, HALF, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    base = (i // HALF) * 2 * HALF + i % HALF
    act = tl.load(ACT)
    z = tl.load(Z + base, mask=mask, other=0.0)
    dz = tl.load(Z + base + HALF, mask=mask, other=0.0)
    gh = tl.load(GH + base, mask=mask, other=0.0)
    gdh = tl.load(GH + base + HALF, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-z))
    d1 = s * (1.0 + z * (1.0 - s))
    d2 = s * (1.0 - s) * (2.0 + z * (1.0 - 2.0 * s))
    tl.store(GZ + base, act * (d1 * gh + d2 * dz * gdh), mask=mask)
    tl.store(GZ + base + HALF, act * d1 * gdh, mask=mask)


def _silu_tb(z, act, fwd=True, gh=None):
    """z [B, 2E, H] with value rows [0, E) and tangent rows [E, 2E)."""
    B, E2, H = z.shape
    half = (E2 // 2) * H
    n = B * half
    a = _act_tensor(act, z)
    out = torch.empty_like(z)
    if n > 0:
        if fwd:
            _silu_tb_fwd_kernel[(triton.cdiv(n, 1024),)](z, out, a, half, n, BLOCK=1024, num_warps=4)
        else:
            _silu_tb_bwd_kernel[(triton.cdiv(n, 1024),)](z, gh, out, a, half, n, BLOCK=1024, num_warps=4)
    return out


class _RadialMLPs(torch.autograd.Function):
    """B radial MLPs (no biases, normalised SiLU) on the same edge features and their r-tangent.

    x [2E, nb] stacks the features (rows [0, E)) and their r-derivative (rows [E, 2E)); weights
    W1 [B, nb, H], W2, W3 [B, H, H], W4 [B, H, O] (scaled). Returns 2B tensors [E, O]: R_0, T_0,
    R_1, T_1, ... Differentiated once (the trainer never differentiates the radial weights twice).
    """

    @staticmethod
    def forward(ctx, x, W1, W2, W3, W4, act):
        # one GEMM per MLP (cuBLAS picks far better kernels than for strided batches here)
        B = W1.shape[0]
        E = x.shape[0] // 2
        H, O = W2.shape[-1], W4.shape[-1]
        kw = dict(device=x.device, dtype=x.dtype)
        z1 = torch.empty(B, 2 * E, H, **kw)
        for b in range(B):
            torch.mm(x, W1[b], out=z1[b])
        h1 = _silu_tb(z1, act)
        z2 = torch.empty_like(z1)
        for b in range(B):
            torch.mm(h1[b], W2[b], out=z2[b])
        h2 = _silu_tb(z2, act)
        z3 = torch.empty_like(z1)
        for b in range(B):
            torch.mm(h2[b], W3[b], out=z3[b])
        h3 = _silu_tb(z3, act)
        out = torch.empty(B, 2 * E, O, **kw)
        for b in range(B):
            torch.mm(h3[b], W4[b], out=out[b])
        ctx.save_for_backward(x, W1, W2, W3, W4, z1, z2, z3, h1, h2, h3)
        ctx.act = act
        res = []
        for b in range(B):
            res += [out[b, :E], out[b, E:]]
        return tuple(res)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grads):
        x, W1, W2, W3, W4, z1, z2, z3, h1, h2, h3 = ctx.saved_tensors
        act = ctx.act
        B, E2, H = h3.shape
        E = E2 // 2
        dh3 = torch.empty_like(h3)
        dW4 = torch.zeros_like(W4)
        for b in range(B):
            for half, g in enumerate(grads[2 * b:2 * b + 2]):
                rows = slice(half * E, (half + 1) * E)
                if g is None:
                    dh3[b, rows].zero_()
                    continue
                torch.mm(g, W4[b].t(), out=dh3[b, rows])
                dW4[b].addmm_(h3[b, rows].t(), g)
        dWs = []
        dz = _silu_tb(z3, act, fwd=False, gh=dh3)
        for z, h, W in ((z2, h2, W3), (z1, h1, W2)):
            dW = torch.empty_like(W)
            dh = torch.empty_like(h)
            for b in range(B):
                torch.mm(h[b].t(), dz[b], out=dW[b])
                torch.mm(dz[b], W[b].t(), out=dh[b])
            dWs.append(dW)
            dz = _silu_tb(z, act, fwd=False, gh=dh)
        dW3, dW2 = dWs
        dW1 = torch.empty_like(W1)
        for b in range(B):
            torch.mm(x.t(), dz[b], out=dW1[b])
        return None, dW1, dW2, dW3, dW4, None


# --------------------------------------------------------------------------- fused radial MLPs (FP32)
@triton.jit
def _mlp_fwd_kernel(X, W1, W2, W3, W4, OUT, ACT, E, NB: tl.constexpr, KP: tl.constexpr, H: tl.constexpr,
                    O: tl.constexpr, OC: tl.constexpr, BM: tl.constexpr, PREC: tl.constexpr):
    """One radial MLP (program axis 1 picks it) on value rows [0, E) and tangent rows [E, 2E) of X [2E, NB]:
    OUT[b] [2E, O]. The hidden layers stay in registers."""
    pid = tl.program_id(0)
    b = tl.program_id(1)
    rows = pid * BM + tl.arange(0, BM)
    rm = rows < E
    k = tl.arange(0, KP)
    hc = tl.arange(0, H)
    act = tl.load(ACT)
    xm = rm[:, None] & (k[None, :] < NB)
    x = tl.load(X + rows[:, None] * NB + k[None, :], mask=xm, other=0.0)
    dx = tl.load(X + (rows[:, None] + E) * NB + k[None, :], mask=xm, other=0.0)
    w = tl.load(W1 + b * NB * H + k[:, None] * H + hc[None, :], mask=k[:, None] < NB, other=0.0)
    z = tl.dot(x, w, input_precision=PREC)
    dz = tl.dot(dx, w, input_precision=PREC)
    for layer in tl.static_range(2):
        sg = 1.0 / (1.0 + tl.exp(-z))
        h = act * z * sg
        dh = act * sg * (1.0 + z * (1.0 - sg)) * dz
        if layer == 0:
            w = tl.load(W2 + b * H * H + hc[:, None] * H + hc[None, :])
        else:
            w = tl.load(W3 + b * H * H + hc[:, None] * H + hc[None, :])
        z = tl.dot(h, w, input_precision=PREC)
        dz = tl.dot(dh, w, input_precision=PREC)
    sg = 1.0 / (1.0 + tl.exp(-z))
    h = act * z * sg
    dh = act * sg * (1.0 + z * (1.0 - sg)) * dz
    base = OUT + b * 2 * E * O
    for j in tl.static_range(O // OC):
        oc = j * OC + tl.arange(0, OC)
        w4 = tl.load(W4 + b * H * O + hc[:, None] * O + oc[None, :])
        tl.store(base + rows[:, None] * O + oc[None, :], tl.dot(h, w4, input_precision=PREC), mask=rm[:, None])
        tl.store(base + (rows[:, None] + E) * O + oc[None, :], tl.dot(dh, w4, input_precision=PREC),
                 mask=rm[:, None])


@triton.jit
def _mlp_layer_bwd(z, dz, gh, gdh, act):
    sg = 1.0 / (1.0 + tl.exp(-z))
    d1 = sg * (1.0 + z * (1.0 - sg))
    d2 = sg * (1.0 - sg) * (2.0 + z * (1.0 - 2.0 * sg))
    return act * (d1 * gh + d2 * dz * gdh), act * d1 * gdh


@triton.jit
def _mlp_hid(z, dz, act):
    sg = 1.0 / (1.0 + tl.exp(-z))
    return act * z * sg, act * sg * (1.0 + z * (1.0 - sg)) * dz


@triton.jit
def _mlp_bwd_kernel(X, W1, W2, W3, W4, GR, GT, DW1, DW2, DW3, DW4, ACT, E, NB: tl.constexpr, KP: tl.constexpr,
                    H: tl.constexpr, O: tl.constexpr, OC: tl.constexpr, BM: tl.constexpr, PREC: tl.constexpr):
    """Weight gradients of one MLP (the forward recomputed) from GR, GT [E, O] (value and tangent outputs);
    each program adds its rows' share into DW* (atomics)."""
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rm = rows < E
    k = tl.arange(0, KP)
    hc = tl.arange(0, H)
    act = tl.load(ACT)
    xm = rm[:, None] & (k[None, :] < NB)
    x = tl.load(X + rows[:, None] * NB + k[None, :], mask=xm, other=0.0)
    dx = tl.load(X + (rows[:, None] + E) * NB + k[None, :], mask=xm, other=0.0)
    w = tl.load(W1 + k[:, None] * H + hc[None, :], mask=k[:, None] < NB, other=0.0)
    z1 = tl.dot(x, w, input_precision=PREC)
    dz1 = tl.dot(dx, w, input_precision=PREC)
    h, dh = _mlp_hid(z1, dz1, act)
    w = tl.load(W2 + hc[:, None] * H + hc[None, :])
    z2 = tl.dot(h, w, input_precision=PREC)
    dz2 = tl.dot(dh, w, input_precision=PREC)
    h, dh = _mlp_hid(z2, dz2, act)
    w = tl.load(W3 + hc[:, None] * H + hc[None, :])
    z3 = tl.dot(h, w, input_precision=PREC)
    dz3 = tl.dot(dh, w, input_precision=PREC)
    h, dh = _mlp_hid(z3, dz3, act)
    ht = tl.trans(h)
    dht = tl.trans(dh)
    gh = tl.zeros((BM, H), dtype=tl.float32)
    gdh = tl.zeros((BM, H), dtype=tl.float32)
    for j in tl.static_range(O // OC):
        oc = j * OC + tl.arange(0, OC)
        gr = tl.load(GR + rows[:, None] * O + oc[None, :], mask=rm[:, None], other=0.0)
        gt = tl.load(GT + rows[:, None] * O + oc[None, :], mask=rm[:, None], other=0.0)
        dw = tl.dot(ht, gr, input_precision=PREC) + tl.dot(dht, gt, input_precision=PREC)
        tl.atomic_add(DW4 + hc[:, None] * O + oc[None, :], dw, sem='relaxed')
        w4t = tl.load(W4 + oc[:, None] + hc[None, :] * O)                 # [OC, H] = W4[:, oc]^T
        gh += tl.dot(gr, w4t, input_precision=PREC)
        gdh += tl.dot(gt, w4t, input_precision=PREC)
    # layer 3
    gz, gdz = _mlp_layer_bwd(z3, dz3, gh, gdh, act)
    h, dh = _mlp_hid(z2, dz2, act)
    dw = tl.dot(tl.trans(h), gz, input_precision=PREC) + tl.dot(tl.trans(dh), gdz, input_precision=PREC)
    tl.atomic_add(DW3 + hc[:, None] * H + hc[None, :], dw, sem='relaxed')
    w3t = tl.load(W3 + hc[:, None] + hc[None, :] * H)
    gh = tl.dot(gz, w3t, input_precision=PREC)
    gdh = tl.dot(gdz, w3t, input_precision=PREC)
    # layer 2
    gz, gdz = _mlp_layer_bwd(z2, dz2, gh, gdh, act)
    h, dh = _mlp_hid(z1, dz1, act)
    dw = tl.dot(tl.trans(h), gz, input_precision=PREC) + tl.dot(tl.trans(dh), gdz, input_precision=PREC)
    tl.atomic_add(DW2 + hc[:, None] * H + hc[None, :], dw, sem='relaxed')
    w2t = tl.load(W2 + hc[:, None] + hc[None, :] * H)
    gh = tl.dot(gz, w2t, input_precision=PREC)
    gdh = tl.dot(gdz, w2t, input_precision=PREC)
    # layer 1
    gz, gdz = _mlp_layer_bwd(z1, dz1, gh, gdh, act)
    dw = tl.dot(tl.trans(x), gz, input_precision=PREC) + tl.dot(tl.trans(dx), gdz, input_precision=PREC)
    tl.atomic_add(DW1 + k[:, None] * H + hc[None, :], dw, mask=k[:, None] < NB, sem='relaxed')


MLP_FUSED = False
MLP_PREC = 'ieee'
MLP_BM = 32
MLP_BM_BWD = 32
MLP_OC_BWD = 32


class _RadialMLPsFused(torch.autograd.Function):
    """``_RadialMLPs`` in two Triton kernels (FP32): hidden layers never leave the chip; the backward
    recomputes them and accumulates the weight gradients with atomics. Differentiated once."""

    @staticmethod
    def forward(ctx, x, W1, W2, W3, W4, act):
        x, W1, W2, W3, W4 = (t.contiguous() for t in (x, W1, W2, W3, W4))
        B, NB, H = W1.shape
        O = W4.shape[-1]
        E = x.shape[0] // 2
        out = torch.empty(B, 2 * E, O, device=x.device, dtype=x.dtype)
        a = _act_tensor(act, x)
        if E > 0:
            _mlp_fwd_kernel[(triton.cdiv(E, MLP_BM), B)](x, W1, W2, W3, W4, out, a, E, NB=NB, KP=16, H=H, O=O,
                                                         OC=64, BM=MLP_BM, PREC=MLP_PREC, num_warps=4, num_stages=1)
        ctx.save_for_backward(x, W1, W2, W3, W4)
        ctx.act = act
        res = []
        for b in range(B):
            res += [out[b, :E], out[b, E:]]
        return tuple(res)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grads):
        x, W1, W2, W3, W4 = ctx.saved_tensors
        B, NB, H = W1.shape
        O = W4.shape[-1]
        E = x.shape[0] // 2
        a = _act_tensor(ctx.act, x)
        dWs = [torch.zeros_like(W) for W in (W1, W2, W3, W4)]
        for b in range(B):
            gr, gt = grads[2 * b], grads[2 * b + 1]
            if gr is None and gt is None:
                continue
            gr = torch.zeros(E, O, device=x.device, dtype=x.dtype) if gr is None else gr.contiguous()
            gt = torch.zeros(E, O, device=x.device, dtype=x.dtype) if gt is None else gt.contiguous()
            if E > 0:
                _mlp_bwd_kernel[(triton.cdiv(E, MLP_BM_BWD),)](
                    x, W1[b], W2[b], W3[b], W4[b], gr, gt, dWs[0][b], dWs[1][b], dWs[2][b], dWs[3][b], a, E,
                    NB=NB, KP=16, H=H, O=O, OC=MLP_OC_BWD, BM=MLP_BM_BWD, PREC=MLP_PREC, num_warps=4, num_stages=1)
        return (None, *dWs, None)


def radial_mlps(x, W1, W2, W3, W4, act):
    if (MLP_FUSED and x.dtype == torch.float32 and x.is_cuda and W1.shape[1] <= 16 and W1.shape[2] == 64
            and W4.shape[2] % 64 == 0):
        return _RadialMLPsFused.apply(x, W1, W2, W3, W4, act)
    return _RadialMLPs.apply(x, W1, W2, W3, W4, act)


# --------------------------------------------------------------------------- fused force-pass edge slots
@triton.jit
def _force_edge_kernel(S_OUT, GY_OUT, rcv, snd, PIDX, G, R, T, Y, X, E, SN, SC, SM, C: tl.constexpr,
                       EB: tl.constexpr, PAIRED: tl.constexpr):
    """s[e] = sum_{l,c} T[e,l,c] sum_{m in l} G[r,c,m] Y[e,m] X[s,c];  gY[e,m] = sum_c G[r,c,m] R[e,l(m),c] X[s,c]."""
    e = tl.program_id(0) * EB + tl.arange(0, EB)
    em = e < E
    c = tl.arange(0, C)
    m2 = em[:, None]
    r = tl.load(rcv + e, mask=em, other=0)
    sn = tl.load(snd + e, mask=em, other=0)
    if PAIRED:
        re = tl.load(PIDX + e, mask=em, other=0)
    else:
        re = e
    x = tl.load(X + sn[:, None] * C + c[None, :], mask=m2, other=0.0)
    sacc = tl.zeros((EB,), dtype=x.dtype)
    for l in tl.static_range(4):
        rx = tl.load(R + re[:, None] * 4 * C + l * C + c[None, :], mask=m2, other=0.0) * x
        tx = tl.load(T + re[:, None] * 4 * C + l * C + c[None, :], mask=m2, other=0.0) * x
        for mm in tl.static_range(2 * l + 1):
            m = l * l + mm
            g = tl.load(G + r[:, None] * SN + c[None, :] * SC + m * SM, mask=m2, other=0.0)
            y = tl.load(Y + e * 16 + m, mask=em, other=0.0)
            sacc += y * tl.sum(g * tx, axis=1)
            tl.store(GY_OUT + e * 16 + m, tl.sum(g * rx, axis=1), mask=em)
    tl.store(S_OUT + e, sacc, mask=em)


@triton.jit
def _edge_a2_kernel(OUT1, OUT2, rcv, snd, G, Y1, Y2, X, E, SN, SC, SM, C: tl.constexpr, EB: tl.constexpr):
    """Two R-slots at once: OUTk[e,l,c] = sum_{m in l} G[r,c,m] Yk[e,m] X[s,c]."""
    e = tl.program_id(0) * EB + tl.arange(0, EB)
    em = e < E
    c = tl.arange(0, C)
    m2 = em[:, None]
    r = tl.load(rcv + e, mask=em, other=0)
    sn = tl.load(snd + e, mask=em, other=0)
    x = tl.load(X + sn[:, None] * C + c[None, :], mask=m2, other=0.0)
    for l in tl.static_range(4):
        a1 = tl.zeros((EB, C), dtype=x.dtype)
        a2 = tl.zeros((EB, C), dtype=x.dtype)
        for mm in tl.static_range(2 * l + 1):
            m = l * l + mm
            g = tl.load(G + r[:, None] * SN + c[None, :] * SC + m * SM, mask=m2, other=0.0)
            a1 += g * tl.load(Y1 + e * 16 + m, mask=em, other=0.0)[:, None]
            a2 += g * tl.load(Y2 + e * 16 + m, mask=em, other=0.0)[:, None]
        tl.store(OUT1 + e[:, None] * 4 * C + l * C + c[None, :], a1 * x, mask=m2)
        tl.store(OUT2 + e[:, None] * 4 * C + l * C + c[None, :], a2 * x, mask=m2)


class _ForceEdge(torch.autograd.Function):
    """Force-pass edge quantities of one interaction from G = dE/d(conv output):
    s = dE/dr through the radial weights (with T = dR/dr) and gY = dE/dY. Differentiated once."""

    @staticmethod
    def forward(ctx, meta, G, R, T, Y, X):
        G, R, T, Y, X = (t.contiguous() for t in (G, R, T, Y, X))
        mnc = _mnc(meta)
        N, C = _gshape(G, mnc)
        E = meta.sender.shape[0]
        s_out = torch.empty(E, device=G.device, dtype=G.dtype)
        gy = torch.empty(E, NLM, device=G.device, dtype=G.dtype)
        if E > 0:
            _force_edge_kernel[(triton.cdiv(E, EB_EDGE),)](s_out, gy, meta.receiver, meta.sender, _pidx(meta), G, R, T,
                                                           Y, X, E, *_gstrides(G, mnc), C=C, EB=EB_EDGE,
                                                           PAIRED=_paired(meta), num_warps=4)
        ctx.meta = meta
        ctx.save_for_backward(G, R, T, Y, X)
        return s_out, gy

    @staticmethod
    @once_differentiable
    def backward(ctx, ws, wY):
        G, R, T, Y, X = ctx.saved_tensors
        meta = ctx.meta
        mnc = _mnc(meta)
        N, C = _gshape(G, mnc)
        E = meta.sender.shape[0]
        ws = torch.zeros(E, device=G.device, dtype=G.dtype) if ws is None else ws.contiguous()
        wY = torch.zeros(E, NLM, device=G.device, dtype=G.dtype) if wY is None else wY.contiguous()
        Yw = (Y * ws[:, None]).contiguous()
        need = ctx.needs_input_grad
        dG = _k_node_g(meta, T, Yw, X, R, wY) if need[1] else None
        dX = _k_node_x(meta, G, T, Yw, R, wY) if need[5] else None
        dR = dT = None
        if (need[2] or need[3]) and _paired(meta):
            dT, dR = _k_edge_pairs(meta, G, X, Yw, wY)
        elif need[2] or need[3]:
            dT = torch.empty_like(T)
            dR = torch.empty_like(R)
            if E > 0:
                _edge_a2_kernel[(triton.cdiv(E, EB_EDGE),)](dT, dR, meta.receiver, meta.sender, G, Yw, wY, X, E,
                                                            *_gstrides(G, mnc), C=C, EB=EB_EDGE, num_warps=4)
        return None, dG, dR, dT, None, dX


def force_edge(meta, G, R, T, Y, X):
    return _ForceEdge.apply(meta, G, R, T, Y, X)
