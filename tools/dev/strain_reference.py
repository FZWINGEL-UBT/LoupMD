"""Explicit reference of one Loup-S training step (plain torch, no autograd through the model).

The specification of the C++/CUDA training library (src/train): forward energy, force pass (first reverse),
loss, and the parameter gradient by an explicit second reverse sweep. Each stage is a named tensor so the
library's kernels can be compared stage by stage. ``check`` compares the parameter gradients with autograd
through loupmd.fasttrain.engine (FP64).

Notation: nodes n (species s(n)), directed edges e = s(e) -> r(e), pairs p(e) (both directions of an edge share
the radial weights), components m = (l, mu) of degree l(m) <= 3, channels c, d. Conv slots:
    conv_g(R, Y, X)[m, n, c] = sum_{e -> n} R[p(e), l(m), c] Y[e, m] X[s(e), c]
    conv_x(G, R, Y)[j, c]    = sum_{e: s(e) = j} sum_m G[m, r(e), c] R[p(e), l(m), c] Y[e, m]
    slot_r(G, Y, X)[p, l, c] = sum_{e in p} sum_{m in l} G[m, r(e), c] Y[e, m] X[s(e), c]
    slot_y(G, R, X)[e, m]    = sum_c G[m, r(e), c] R[p(e), l(m), c] X[s(e), c]

Usage (local GPU reservation): python tools/dev/strain_reference.py --model M --xyz X
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

L_OF_M = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])


# ----------------------------------------------------------------------------- primitives
class Graph:
    def __init__(self, bt):
        self.snd, self.rcv, self.pidx = bt.sender, bt.receiver, bt.pidx
        self.pe0, self.pe1 = bt.pe0, bt.pe1
        self.N = bt.species.shape[0]
        self.P = bt.pe0.shape[0]
        self.lm = L_OF_M.to(bt.sender.device)

    def conv_g(self, R, Y, X):
        t = R[self.pidx][:, self.lm, :] * Y[:, :, None] * X[self.snd][:, None, :]          # [E, 16, C]
        out = torch.zeros(self.N, 16, X.shape[1], dtype=X.dtype, device=X.device).index_add_(0, self.rcv, t)
        return out.permute(1, 0, 2).contiguous()                                            # [16, N, C]

    def conv_x(self, G, R, Y):
        t = (G[:, self.rcv, :].permute(1, 0, 2) * R[self.pidx][:, self.lm, :] * Y[:, :, None]).sum(1)  # [E, C]
        return torch.zeros(self.N, G.shape[2], dtype=G.dtype, device=G.device).index_add_(0, self.snd, t)

    def slot_r(self, G, Y, X):
        t = G[:, self.rcv, :].permute(1, 0, 2) * Y[:, :, None] * X[self.snd][:, None, :]      # [E, 16, C]
        per_l = torch.zeros(t.shape[0], 4, t.shape[2], dtype=t.dtype, device=t.device).index_add_(1, self.lm, t)
        return torch.zeros(self.P, 4, t.shape[2], dtype=t.dtype, device=t.device).index_add_(0, self.pidx, per_l)

    def slot_y(self, G, R, X):
        return (G[:, self.rcv, :].permute(1, 0, 2) * R[self.pidx][:, self.lm, :] * X[self.snd][:, None, :]).sum(2)


class Poly:
    """P_k(x), x [..., 16], from the generated term list; VJP, JVP, HVP explicitly."""

    def __init__(self, terms, K, device, dtype):
        self.K = K
        self.terms = [(k, list(m), v) for k, m, v in terms]

    def p(self, x):
        out = torch.zeros(*x.shape[:-1], self.K, dtype=x.dtype, device=x.device)
        for k, m, v in self.terms:
            t = v * x[..., m[0]]
            for j in m[1:]:
                t = t * x[..., j]
            out[..., k] += t
        return out

    @staticmethod
    def _d(m, j):
        if j not in m:
            return 0, None
        rest = list(m)
        rest.remove(j)
        return m.count(j), rest

    def _mono(self, x, m):
        t = torch.ones_like(x[..., 0])
        for j in m:
            t = t * x[..., j]
        return t

    def vjp(self, x, V):
        """G_j = sum_k V_k dP_k/dx_j."""
        G = torch.zeros_like(x)
        for k, m, v in self.terms:
            for j in sorted(set(m)):
                n, rest = self._d(m, j)
                G[..., j] += V[..., k] * (n * v) * self._mono(x, rest)
        return G

    def jvp(self, x, T):
        """J_k = sum_j dP_k/dx_j T_j."""
        J = torch.zeros(*x.shape[:-1], self.K, dtype=x.dtype, device=x.device)
        for k, m, v in self.terms:
            for j in sorted(set(m)):
                n, rest = self._d(m, j)
                J[..., k] += (n * v) * self._mono(x, rest) * T[..., j]
        return J

    def hvp(self, x, V, T):
        """H_i = sum_k V_k sum_j d2P_k/dx_i dx_j T_j."""
        H = torch.zeros_like(x)
        for k, m, v in self.terms:
            for i in sorted(set(m)):
                ni, rest = self._d(m, i)
                for j in sorted(set(rest)):
                    nj, rest2 = self._d(rest, j)
                    H[..., i] += V[..., k] * (ni * nj * v) * self._mono(x, rest2) * T[..., j]
        return H


def silu1(u):
    s = torch.sigmoid(u)
    return s * (1 + u * (1 - s))


def silu2(u):
    s = torch.sigmoid(u)
    return s * (1 - s) * (2 + u * (1 - 2 * s))


# ----------------------------------------------------------------------------- parameters
def params(model):
    """Scaled parameter views as the engine uses them (differentiable functions of the raw tensors)."""
    m = model
    i0, i1 = m.interactions
    p0, p1 = m.products
    S = len(m.atomic_numbers)
    C = p0.linear.irreps_out.dim
    rC = math.sqrt(C)
    avg = [float(i.avg_num_neighbors) for i in m.interactions]
    hid = m.readouts[1].linear_1.weight.numel() // C
    q = SimpleNamespace(S=S, C=C, hid=hid)
    q.emb = m.node_embedding.linear.weight.view(S, C) / math.sqrt(S)
    q.wup0 = i0.linear_up.weight.view(C, C) / rC
    q.wl0 = i0.linear.weight.view(4, C, C) / (rC * avg[0])
    q.ws0 = i0.skip_tp.weight.view(4, C, S, C) / math.sqrt(C * S)
    q.wl1 = i1.linear.weight.view(4, C, C) / (rC * avg[1])
    cat = lambda c: torch.cat([c.weights_max, c.weights[0], c.weights[1]], dim=1)          # [S, K, C]
    q.W0 = cat(p0.symmetric_contractions.contractions[0])
    q.W1 = cat(p1.symmetric_contractions.contractions[0])
    q.wp0 = p0.linear.weight.view(C, C) / rC
    q.wr0 = m.readouts[0].linear.weight.view(C, 1) / rC
    q.wu1 = i1.linear_up.weight.view(C, C) / rC
    q.wsk1 = i1.skip_tp.weight.view(C, S * C) / math.sqrt(C * S)
    q.wp1 = p1.linear.weight.view(C, C) / rC
    q.wro1 = m.readouts[1].linear_1.weight.view(C, hid) / rC
    q.wro2 = m.readouts[1].linear_2.weight.view(hid, 1) / math.sqrt(hid)
    q.scale = float(m.scale_shift.scale)
    q.mlp = []
    for inter in m.interactions:
        net = inter.conv_tp_weights
        q.mlp.append([getattr(net, f'layer{k}').weight / math.sqrt(getattr(net, f'layer{k}').weight.shape[0])
                      for k in range(4)])
    return q


def mlp_value_tangent(x, dx, Ws, act):
    """Radial weights and their r-tangent: R = mlp(x), T = J_mlp(x) dx; hidden states returned."""
    h, dh = x, dx
    zs = []
    for i, W in enumerate(Ws):
        z, dz = h @ W, dh @ W
        if i < len(Ws) - 1:
            zs.append((h, dh, z, dz))
            s = torch.sigmoid(z)
            h, dh = act * z * s, act * silu1(z) * dz
        else:
            zs.append((h, dh, None, None))
            h, dh = z, dz
    return h, dh, zs


def mlp_backward(zs, Ws, gR, gT, act):
    """Weight gradients of the MLP (value + tangent outputs) from cotangents gR, gT [P, O]."""
    dWs = [None] * len(Ws)
    gh, gdh = gR, gT
    for i in range(len(Ws) - 1, -1, -1):
        h, dh, z, dz = zs[i]
        if i < len(Ws) - 1:            # through the activation of layer i (output z of W_i)
            gz = act * (silu1(z) * gh + silu2(z) * dz * gdh)
            gdz = act * silu1(z) * gdh
        else:
            gz, gdz = gh, gdh
        dWs[i] = h.T @ gz + dh.T @ gdz
        gh, gdh = gz @ Ws[i].T, gdz @ Ws[i].T
    return dWs


# ----------------------------------------------------------------------------- the step
def step(model, bt, poly, act, lw, keep=None):
    """Returns (loss, dict of gradients w.r.t. the scaled views in params(), stages)."""
    q = params(model)
    q = SimpleNamespace(**{k: (v.detach() if torch.is_tensor(v) else v) for k, v in vars(q).items()})
    q.mlp = [[w.detach() for w in ws] for ws in q.mlp]
    S, C = q.S, q.C
    gr = Graph(bt)
    lm = gr.lm
    sp = bt.species
    N = sp.shape[0]
    st = {}                                   # stages
    from loupmd.fasttrain import ops
    from loupmd.fasttrain.engine import FastStudent
    fs = FastStudent(model)
    Y, ef, dEf, U, JY = ops.edge_basis_raw(bt.vectors, fs.bw, fs.prefactor, fs.r_max, fs.p)
    xe, dxe = ef[bt.pe0], dEf[bt.pe0]
    R, T, Z = [], [], []
    for t in range(2):
        r_, t_, z_ = mlp_value_tangent(xe, dxe, q.mlp[t], act)
        R.append(r_.view(-1, 4, C))
        T.append(t_.view(-1, 4, C))
        Z.append(z_)
    st.update(Y=Y, R0=R[0], T0=T[0], R1=R[1], T1=T[1])
    onehot = (sp[:, None] == torch.arange(S, device=sp.device)).to(Y.dtype)
    # ---------------- forward
    T0 = q.emb @ q.wup0                                           # [S, C]
    x0 = T0[sp]
    A0 = gr.conv_g(R[0], Y, x0)                                   # [16, N, C]
    M0 = torch.einsum('lce,lesd->lcsd', q.wl0, q.ws0)            # [4, C, S, C]
    M0n = M0[:, :, sp, :]                                         # [4, C, N, C]
    B0 = torch.einsum('mnc,mcnd->mnd', A0, M0n[lm])              # [16, N, C]
    x_B0 = B0.permute(1, 2, 0)                                    # [N, C, 16]
    P0 = poly.p(x_B0)                                             # [N, C, K]
    W0n = q.W0[sp].transpose(1, 2)                                # [N, C, K]
    f0 = (P0 * W0n).sum(-1)
    Q = q.wp0 @ torch.cat([q.wr0, q.wu1, q.wsk1], dim=1)          # [C, 1 + C + S C]
    z = f0 @ Q
    e0 = z[:, 0]
    x1 = z[:, 1:1 + C]
    zsc = z[:, 1 + C:].view(N, S, C)
    sc = zsc[torch.arange(N), sp]
    A1 = gr.conv_g(R[1], Y, x1)
    B1 = torch.einsum('mnc,mcd->mnd', A1, q.wl1[lm])
    x_B1 = B1.permute(1, 2, 0)
    P1 = poly.p(x_B1)
    W1n = q.W1[sp].transpose(1, 2)
    f1 = (P1 * W1n).sum(-1)
    f2 = f1 @ q.wp1 + sc
    u = f2 @ q.wro1
    tt = act * torch.nn.functional.silu(u)
    e1 = (tt @ q.wro2)[:, 0]
    ne = q.scale * (e0 + e1)
    E = torch.zeros(bt.num_graphs, dtype=ne.dtype, device=ne.device).index_add_(0, bt.node_graph, ne)
    st.update(x0=x0, A0=A0, B0=B0, f0=f0, x1=x1, sc=sc, A1=A1, B1=B1, f1=f1, f2=f2, u=u, E=E)
    # ---------------- force pass (first reverse): G_t = dE/dA_t
    tb = q.scale * q.wro2[:, 0].expand(N, -1)                     # [N, hid]
    ub = tb * act * silu1(u)
    f2b = ub @ q.wro1.T
    f1b = f2b @ q.wp1.T
    scb = f2b
    V1 = W1n * f1b[:, :, None]                                    # [N, C, K]
    B1b = poly.vjp(x_B1, V1).permute(2, 0, 1)                     # [16, N, C]
    G1 = torch.einsum('mnd,mcd->mnc', B1b, q.wl1[lm])
    x1b = gr.conv_x(G1, R[1], Y)
    zb = torch.zeros(N, 1 + C + S * C, dtype=Y.dtype, device=Y.device)
    zb[:, 0] = q.scale
    zb[:, 1:1 + C] = x1b
    zb[:, 1 + C:].view(N, S, C)[torch.arange(N), sp] = scb
    f0b = zb @ Q.T
    V0 = W0n * f0b[:, :, None]
    B0b = poly.vjp(x_B0, V0).permute(2, 0, 1)
    G0 = torch.einsum('mnd,mcnd->mnc', B0b, M0n[lm])
    st.update(G0=G0, G1=G1, x1b=x1b, f0b=f0b, f1b=f1b, f2b=f2b)
    s_e, gy = 0, 0
    for t, (G, X) in enumerate(((G0, x0), (G1, x1))):
        s_e = s_e + (gr.slot_y(G, T[t], X) * Y).sum(1)      # sum_{l,c} T G Y X
        gy = gy + gr.slot_y(G, R[t], X)
    g = s_e[:, None] * U + torch.einsum('em,emk->ek', gy, JY)   # dE/dvec [E, 3]
    dpos = torch.zeros(N, 3, dtype=g.dtype, device=g.device).index_add_(0, bt.receiver, g).index_add_(0, bt.sender, -g)
    F = -dpos + bt.core_f
    mv = torch.zeros(bt.num_graphs, 3, 3, dtype=g.dtype, device=g.device).index_add_(
        0, bt.edge_graph, bt.vectors[:, :, None] * g[:, None, :])
    Sg = 0.5 * (mv + mv.transpose(1, 2)) / bt.volume[:, None, None] + bt.core_s
    Etot = E + bt.core_e
    st.update(s_e=s_e, gy=gy, g=g, F=F, S=Sg)
    # ---------------- loss
    n_real, g_real = bt.counts[0], bt.counts[1]
    le = (bt.w_e * ((bt.e_ref - Etot) / bt.n_atoms) ** 2).sum() / g_real
    lf = (bt.w_f[:, None] * (bt.f_ref - F) ** 2).sum() / (3 * n_real)
    ls = (bt.w_s[:, None, None] * (bt.s_ref - Sg) ** 2).sum() / (9 * g_real)
    loss = lw[0] * le + lw[1] * lf + lw[2] * ls
    Eb = lw[0] * bt.w_e * 2 * (Etot - bt.e_ref) / bt.n_atoms ** 2 / g_real
    Fb = lw[1] * bt.w_f[:, None] * 2 * (F - bt.f_ref) / (3 * n_real)
    Sb = lw[2] * bt.w_s[:, None, None] * 2 * (Sg - bt.s_ref) / (9 * g_real)
    # ---------------- second reverse
    Ms = 0.5 * (Sb + Sb.transpose(1, 2)) / bt.volume[:, None, None]
    w = Fb[bt.sender] - Fb[bt.receiver] + torch.einsum('ei,eij->ej', bt.vectors, Ms[bt.edge_graph])   # dL/dg
    sb = (w * U).sum(1)                                           # dL/ds_e
    gyb = torch.einsum('ek,emk->em', w, JY)                       # dL/dgy
    st.update(w=w, sb=sb, gyb=gyb)
    grads = {}
    Rb = [None, None]
    Tb = [None, None]
    Gs = [G0, G1]
    Xs = [x0, x1]
    Gst, Xd = [], []
    for t in range(2):
        Yw = Y * sb[:, None]
        Gst.append(gr.conv_g(T[t], Yw, Xs[t]) + gr.conv_g(R[t], gyb, Xs[t]))     # dL/dG_t (force_edge)
        Xd.append(gr.conv_x(Gs[t], T[t], Yw) + gr.conv_x(Gs[t], R[t], gyb))       # dL/dX_t (force_edge)
        Tb[t] = gr.slot_r(Gs[t], Yw, Xs[t])
        Rb[t] = gr.slot_r(Gs[t], gyb, Xs[t])
    st.update(G0s=Gst[0], G1s=Gst[1])
    # (g)* x1b = conv_x(G1, R1, Y): cotangent x1b* comes from f0b* below; do the force-pass reverse top-down
    # (j)* G0 = B0b M0 -> B0b*, M0*
    B0bs = torch.einsum('mnc,mcnd->mnd', Gst[0], M0n[lm])
    M0s = torch.zeros_like(M0)
    for l in range(4):
        ms = (lm == l).nonzero()[:, 0]
        M0s[l] += torch.einsum('mnc,mnd,ns->csd', Gst[0][ms], B0b[ms], onehot)
    # (i)* B0b = vjp(B0, V0)
    B0s = poly.hvp(x_B0, V0, B0bs.permute(1, 2, 0)).permute(2, 0, 1)            # dL/dB0 (part)
    V0s = poly.jvp(x_B0, B0bs.permute(1, 2, 0))                                  # [N, C, K]
    f0bs = (V0s * W0n).sum(-1)
    W0s = torch.einsum('nck,ns->skc', V0s * f0b[:, :, None], onehot)
    # (h)* f0b = zb Q^T
    zbs = f0bs @ Q                                                                # [N, 1 + C + S C]
    Qs = f0bs.T @ zb                                                              # [C, 1 + C + S C]
    x1bs = zbs[:, 1:1 + C]
    scbs = zbs[:, 1 + C:].view(N, S, C)[torch.arange(N), sp]
    # (g)* x1b = conv_x(G1, R1, Y) -> G1 cotangent, R1 cotangent
    G1tot = Gst[1] + gr.conv_g(R[1], Y, x1bs)
    Rb[1] = Rb[1] + gr.slot_r(G1, Y, x1bs)
    # (f)* G1 = B1b wl1^T
    B1bs = torch.einsum('mnc,mcd->mnd', G1tot, q.wl1[lm])
    wl1s = torch.zeros_like(q.wl1)
    for l in range(4):
        ms = (lm == l).nonzero()[:, 0]
        wl1s[l] += torch.einsum('mnc,mnd->cd', G1tot[ms], B1b[ms])
    # (e)* B1b = vjp(B1, V1)
    B1s = poly.hvp(x_B1, V1, B1bs.permute(1, 2, 0)).permute(2, 0, 1)
    V1s = poly.jvp(x_B1, B1bs.permute(1, 2, 0))
    f1bs = (V1s * W1n).sum(-1)
    W1s = torch.einsum('nck,ns->skc', V1s * f1b[:, :, None], onehot)
    # (d)* f1b = f2b wp1^T, scb = f2b
    f2bs = f1bs @ q.wp1 + scbs
    wp1s = f1bs.T @ f2b
    # (c)* f2b = ub wro1^T
    ubs = f2bs @ q.wro1
    wro1s = f2bs.T @ ub
    # (b)* ub = tb act silu'(u)
    tbs = ubs * act * silu1(u)
    us = ubs * tb * act * silu2(u)
    # (a)* tb = scale wro2^T
    wro2s = q.scale * tbs.sum(0)[:, None]
    # ---------------- forward reverse
    qn = q.scale * Eb[bt.node_graph]                             # dL/de0 = dL/de1 per node
    ts = qn[:, None] * q.wro2[:, 0][None, :]
    wro2s = wro2s + tt.T @ qn[:, None]
    us = us + ts * act * silu1(u)
    f2s = us @ q.wro1.T
    wro1s = wro1s + f2.T @ us
    f1s = f2s @ q.wp1.T
    wp1s = wp1s + f1.T @ f2s
    scs = f2s
    B1s = B1s + poly.vjp(x_B1, W1n * f1s[:, :, None]).permute(2, 0, 1)
    W1s = W1s + torch.einsum('nck,ns->skc', P1 * f1s[:, :, None], onehot)
    A1s = torch.einsum('mnd,mcd->mnc', B1s, q.wl1[lm])
    for l in range(4):
        ms = (lm == l).nonzero()[:, 0]
        wl1s[l] += torch.einsum('mnc,mnd->cd', A1[ms], B1s[ms])
    x1s = Xd[1] + gr.conv_x(A1s, R[1], Y)
    Rb[1] = Rb[1] + gr.slot_r(A1s, Y, x1)
    zs = torch.zeros(N, 1 + C + S * C, dtype=Y.dtype, device=Y.device)
    zs[:, 0] = qn
    zs[:, 1:1 + C] = x1s
    zs[:, 1 + C:].view(N, S, C)[torch.arange(N), sp] = scs
    f0s = zs @ Q.T
    Qs = Qs + f0.T @ zs
    B0s = B0s + poly.vjp(x_B0, W0n * f0s[:, :, None]).permute(2, 0, 1)
    W0s = W0s + torch.einsum('nck,ns->skc', P0 * f0s[:, :, None], onehot)
    A0s = torch.einsum('mnd,mcnd->mnc', B0s, M0n[lm])
    for l in range(4):
        ms = (lm == l).nonzero()[:, 0]
        M0s[l] += torch.einsum('mnc,mnd,ns->csd', A0[ms], B0s[ms], onehot)
    x0s = Xd[0] + gr.conv_x(A0s, R[0], Y)
    Rb[0] = Rb[0] + gr.slot_r(A0s, Y, x0)
    T0s = onehot.T @ x0s                                          # [S, C]
    st.update(x0s=x0s, x1s=x1s, A0s=A0s, A1s=A1s, R0b=Rb[0], T0b=Tb[0], R1b=Rb[1], T1b=Tb[1])
    # ---------------- parameters
    grads['emb'] = T0s @ q.wup0.T
    grads['wup0'] = q.emb.T @ T0s
    grads['wl0'] = torch.einsum('lcsd,lesd->lce', M0s, q.ws0)
    grads['ws0'] = torch.einsum('lcsd,lce->lesd', M0s, q.wl0)
    grads['wl1'] = wl1s
    grads['W0'] = W0s
    grads['W1'] = W1s
    Wcat = torch.cat([q.wr0, q.wu1, q.wsk1], dim=1)
    grads['wp0'] = Qs @ Wcat.T
    Wcs = q.wp0.T @ Qs
    grads['wr0'] = Wcs[:, :1]
    grads['wu1'] = Wcs[:, 1:1 + C]
    grads['wsk1'] = Wcs[:, 1 + C:]
    grads['wp1'] = wp1s
    grads['wro1'] = wro1s
    grads['wro2'] = wro2s
    for t in range(2):
        dW = mlp_backward(Z[t], q.mlp[t], Rb[t].reshape(Rb[t].shape[0], -1), Tb[t].reshape(Tb[t].shape[0], -1), act)
        for k in range(4):
            grads[f'mlp{t}_{k}'] = dW[k]
    return loss, grads, st


def check(args):
    from fasttrain_parity import load_items
    from loupmd.fasttrain.data import frames_from_dataset
    from loupmd.fasttrain.engine import PaddedFrames, FastStudent
    from loupmd.fasttrain import _product_poly as pp
    from loupmd.fasttrain.model import silu_norm_constant
    dtype, dev = torch.float64, torch.device('cuda')
    model = torch.load(args.model, map_location='cpu').to(dtype)
    items = load_items(args.xyz, model, dtype, 7)
    model = model.to(dev)
    frames = frames_from_dataset(items, model)
    pf = PaddedFrames(frames, dtype, dev, batch_size=5, r_max=float(model.r_max))
    bt = pf.gather(pf.device_batch(pf.host_batch([5, 1, 3, 0])))
    lw = (1.0, 100.0, 1.0) if args.stage == 1 else (1000.0, 100.0, 10.0)
    act = silu_norm_constant()
    poly = Poly(pp.TERMS, pp.K, dev, dtype)
    loss, grads, _ = step(model, bt, poly, act, lw)
    # autograd through the engine, w.r.t. the same scaled views
    q = params(model)
    views = dict(emb=q.emb, wup0=q.wup0, wl0=q.wl0, ws0=q.ws0, wl1=q.wl1, W0=q.W0, W1=q.W1, wp0=q.wp0, wr0=q.wr0,
                 wu1=q.wu1, wsk1=q.wsk1, wp1=q.wp1, wro1=q.wro1, wro2=q.wro2)
    for t in range(2):
        for k in range(4):
            views[f'mlp{t}_{k}'] = q.mlp[t][k]
    st = FastStudent(model)
    e, f, s = st.evaluate(bt, training=True)
    le, lf, ls = st.loss_terms(bt, e, f, s)
    loss2 = lw[0] * le + lw[1] * lf + lw[2] * ls
    # gradient w.r.t. the raw parameters, mapped to the views by the chain rule through params(): compare the
    # raw-parameter gradients instead (both sides)
    raw = [p for p in model.parameters() if p.requires_grad]
    g_ref = torch.autograd.grad(loss2, raw)
    vals = [views[k] for k in grads]
    g_new = torch.autograd.grad(vals, raw, [grads[k] for k in grads], allow_unused=True)
    print(f'loss explicit {float(loss):.15g} engine {float(loss2):.15g}')
    worst = 0.0
    for (n, _), a, b in zip([(n, p) for n, p in model.named_parameters() if p.requires_grad], g_ref, g_new):
        b = torch.zeros_like(a) if b is None else b
        rel = float((a - b).norm() / a.norm().clamp_min(1e-300))
        worst = max(worst, rel)
        print(f'  {n:58s} {rel:.2e}')
    print('worst grad rel diff', worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--xyz', required=True)
    ap.add_argument('--stage', type=int, default=2)
    check(ap.parse_args())


if __name__ == '__main__':
    main()
