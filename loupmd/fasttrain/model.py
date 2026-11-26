"""Exact re-expression of a stock mace-torch 0.3.16 ``ScaleShiftMACE`` student for training.

Supported signature (checked by :func:`check_supported`): one head, hidden ``Cx0e`` scalars, two
interactions (``RealAgnosticInteractionBlock`` then ``RealAgnosticResidualInteractionBlock``),
spherical harmonics through l = 3, correlation 3, Bessel radial basis with polynomial cutoff applied
before the radial MLP, ZBL pair repulsion, ``NonLinearReadoutBlock`` with SiLU last.

All trainable tensors are *views* of the mace-torch model's own ``Parameter`` objects, so the
optimizer, EMA, SWA and checkpoint code of mace-torch operate on the canonical parameters.
The fixed ZBL core (no trainable parameter) is evaluated once per immutable frame in float64 and
added, scaled, to the learned energy, forces and stress (plans/loup-s-training-engine.md, ZBL rule).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

LMAX = 3
NLM = (LMAX + 1) ** 2
L_OF_LM = torch.tensor([l for l in range(LMAX + 1) for _ in range(2 * l + 1)])


@dataclass
class Batch:
    """A packed batch with nodes sorted by species (contiguous species blocks)."""
    vectors: Tensor        # [E, 3] r_receiver - r_sender + shift, fixed geometry
    sender: Tensor         # [E] int64
    receiver: Tensor       # [E] int64, edges sorted by receiver
    species: Tensor        # [N] int64 species index (sorted ascending)
    species_bounds: tuple  # python ints: node ranges per species, len S + 1
    node_graph: Tensor     # [N] graph index of each node
    edge_graph: Tensor     # [E] graph index of each edge
    num_graphs: int
    n_atoms: Tensor        # [G] float atom counts
    volume: Tensor         # [G]
    e_ref: Tensor          # [G]
    f_ref: Tensor          # [N, 3]
    s_ref: Tensor          # [G, 3, 3]
    w_energy: Tensor       # [G] weight * energy_weight
    w_forces: Tensor       # [N] weight * forces_weight, per atom
    w_stress: Tensor       # [G] weight * stress_weight
    core_e: Tensor         # [G] scaled fixed core energy (+ E0 + shift sum), model dtype
    core_f: Tensor         # [N, 3]
    core_s: Tensor         # [G, 3, 3]
    receiver_ptr: Optional[Tensor] = None   # [N + 1] CSR row pointer of receivers
    sender_perm: Optional[Tensor] = None    # [E] edge ids sorted by sender
    sender_ptr: Optional[Tensor] = None     # [N + 1]
    frame_ids: Optional[list] = None


def check_supported(model) -> None:
    from mace.modules.models import ScaleShiftMACE
    from mace.modules.blocks import (RealAgnosticInteractionBlock, RealAgnosticResidualInteractionBlock,
                                     NonLinearReadoutBlock, LinearReadoutBlock)
    problems = []
    if type(model) is not ScaleShiftMACE:
        problems.append(f'model class {type(model).__name__}')
    if len(model.interactions) != 2:
        problems.append('number of interactions')
    else:
        if type(model.interactions[0]) is not RealAgnosticInteractionBlock:
            problems.append('first interaction class')
        if type(model.interactions[1]) is not RealAgnosticResidualInteractionBlock:
            problems.append('second interaction class')
    if not hasattr(model, 'pair_repulsion'):
        problems.append('no ZBL pair repulsion')
    if len(model.heads) != 1:
        problems.append('multiple heads')
    if model.spherical_harmonics.irreps_out.lmax != LMAX:
        problems.append('spherical harmonics lmax')
    if model.radial_embedding.bessel_fn.bessel_weights.requires_grad:
        problems.append('trainable Bessel weights')
    if hasattr(model.radial_embedding, 'distance_transform'):
        problems.append('distance transform')
    if not getattr(model.radial_embedding, 'apply_cutoff', True):
        problems.append('cutoff not applied before MLP')
    if type(model.readouts[0]) is not LinearReadoutBlock or type(model.readouts[1]) is not NonLinearReadoutBlock:
        problems.append('readout classes')
    for p in model.products:
        sc = p.symmetric_contractions
        if len(sc.contractions) != 1 or sc.contractions[0].correlation != 3:
            problems.append('product correlation/irreps')
    hid = model.products[0].linear.irreps_out
    if hid.lmax != 0 or len(hid) != 1:
        problems.append(f'hidden irreps {hid}')
    if problems:
        raise ValueError('unsupported student for the fast trainer: ' + ', '.join(problems))


_POLY_CHECKED = []          # the coupling tensors (U matrices per product) already found equal to the polynomials


def _u_matrices(model):
    out = []
    for p in model.products:
        c = p.symmetric_contractions.contractions[0]
        out.append(tuple(u.detach().double().cpu() for u in (c.U_matrix_3, c.U_matrix_2, c.U_matrix_1)))
    return out


def check_poly_terms(model) -> None:
    """The generated product polynomials must be this model's coupling tensors (checked once per distinct set of
    U matrices: the trainer, the evaluators and a reloaded checkpoint share them)."""
    from . import _product_terms as pp            # the terms alone: no Triton import on the library's path
    from .polyterms import model_terms
    us = _u_matrices(model)
    same = lambda a, b: len(a) == len(b) and all(
        x.shape == y.shape and torch.equal(x, y) for ta, tb in zip(a, b) for x, y in zip(ta, tb))
    if any(same(us, seen) for seen in _POLY_CHECKED):
        return
    for terms, K in model_terms(model):
        ok = K == pp.K and len(terms) == len(pp.TERMS) and all(
            k1 == k2 and m1 == tuple(m2) and abs(v1 - v2) <= 1e-12 * max(1.0, abs(v2))
            for (k1, m1, v1), (k2, m2, v2) in zip(terms, pp.TERMS))
        if not ok:
            raise ValueError('product coupling tensors differ from the generated polynomials; '
                             'rerun tools/dev/gen_fasttrain_product.py')
    _POLY_CHECKED.append(us)


_SILU_CST = []


def silu_norm_constant() -> float:
    """e3nn's second-moment normalisation of SiLU (``normalize2mom``: a million draws of its own generator seeded
    with 0, the global generator untouched), computed once."""
    if not _SILU_CST:
        from e3nn.math import normalize2mom
        _SILU_CST.append(float(normalize2mom(F.silu).cst))
    return _SILU_CST[0]


class Student:
    """Fast energy/force/stress evaluation sharing parameters with a mace-torch model."""

    def __init__(self, model, conv: Optional[Callable] = None, product: str = 'grouped',
                 basis: Optional[Callable] = None, tangent: bool = False):
        check_supported(model)
        self.basis = basis
        self.tangent = tangent
        self.product_mode = product
        if product == 'poly':
            check_poly_terms(model)
        self.model = model
        self.S = int(len(model.atomic_numbers))
        self.C = int(model.products[0].linear.irreps_out.dim)
        self.avg = [float(i.avg_num_neighbors) for i in model.interactions]
        be = model.radial_embedding.bessel_fn
        self.bessel_w = be.bessel_weights
        self.prefactor = be.prefactor
        self.r_max = float(model.radial_embedding.cutoff_fn.r_max)
        self.p = int(model.radial_embedding.cutoff_fn.p)
        self.scale = model.scale_shift.scale
        self.shift = model.scale_shift.shift
        self.act = silu_norm_constant()
        self.conv = conv if conv is not None else conv_reference
        self.l_of_lm = L_OF_LM.to(be.bessel_weights.device)
        for inter in model.interactions:
            n = len(inter.conv_tp_weights.hs)
            assert n == 5, 'radial MLP depth'

    # ----------------------------------------------------------------- pieces
    def radial(self, r: Tensor) -> Tensor:
        bes = self.prefactor * (torch.sin(self.bessel_w * r) / r)
        u = r / self.r_max
        p = self.p
        env = (1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(u, p) + p * (p + 2.0) * torch.pow(u, p + 1)
               - (p * (p + 1.0) / 2) * torch.pow(u, p + 2))
        env = env * (r < self.r_max)
        return bes * env

    def mlp(self, x: Tensor, net) -> Tensor:
        layers = [getattr(net, f'layer{i}') for i in range(len(net.hs) - 1)]
        for i, layer in enumerate(layers):
            w = layer.weight
            x = x @ (w / math.sqrt(w.shape[0]))
            if i < len(layers) - 1:
                x = F.silu(x) * self.act
        return x

    def lin(self, x: Tensor, w: Tensor, cin: int, cout: int) -> Tensor:
        return x @ (w.view(cin, cout) / math.sqrt(cin))

    def per_l_linear(self, a: Tensor, w: Tensor) -> Tensor:
        """a [N, C, 16]; e3nn Linear 64x(0e+1o+2e+3o) -> same: per-l [C, C] maps, mul-major."""
        C = self.C
        ws = w.view(LMAX + 1, C, C) / math.sqrt(C)
        outs = []
        for l in range(LMAX + 1):
            sl = slice(l * l, (l + 1) * (l + 1))
            outs.append(torch.einsum('ncm,cd->ndm', a[:, :, sl], ws[l]))
        return torch.cat(outs, dim=2)

    def species_blocks(self, x: Tensor, bounds, fn):
        outs = [fn(x[bounds[s]:bounds[s + 1]], s) for s in range(self.S) if bounds[s + 1] > bounds[s]]
        return torch.cat(outs, dim=0)

    def skip_first(self, a: Tensor, w: Tensor, bounds) -> Tensor:
        """FullyConnectedTensorProduct(Cx(0e+1o+2e+3o) x Sx0e -> same) with one-hot species."""
        C, S = self.C, self.S
        ws = w.view(LMAX + 1, C, S, C) / math.sqrt(C * S)

        def block(x, s):
            outs = []
            for l in range(LMAX + 1):
                sl = slice(l * l, (l + 1) * (l + 1))
                outs.append(torch.einsum('ncm,cd->ndm', x[:, :, sl], ws[l, :, s]))
            return torch.cat(outs, dim=2)
        return self.species_blocks(a, bounds, block)

    def skip_scalar(self, h: Tensor, w: Tensor, bounds) -> Tensor:
        C, S = self.C, self.S
        ws = w.view(C, S, C) / math.sqrt(C * S)
        return self.species_blocks(h, bounds, lambda x, s: x @ ws[:, s])

    def product(self, a: Tensor, contraction, bounds) -> Tensor:
        if self.product_mode == 'poly':
            return self.product_poly(a, contraction)
        return self.product_horner(a, contraction, bounds)

    def product_poly(self, a: Tensor, contraction) -> Tensor:
        """out[n, c] = sum_k W[s(n), k, c] P_k(a[n, c, :]) with generated invariant polynomials."""
        from .ops import product_poly
        W = torch.cat([contraction.weights_max, contraction.weights[0], contraction.weights[1]], dim=1)
        P = product_poly(a)                                  # [N, C, K]
        Wn = W.transpose(1, 2)[self._species]                # [N, C, K]
        return (P * Wn).sum(-1)

    def product_horner(self, a: Tensor, contraction, bounds) -> Tensor:
        """MACE order-3 symmetric contraction to scalars, by species-blocked Horner evaluation.

        out[n,c] = sum_w x_w (sum_y x_y (sum_i x_i A3[w,y,i] + A2[w,y]) + A1[w]),
        A_nu[s, c] = U_nu . W_nu[s, :, c], with x = a[n, c, :] (16 components).
        """
        U3 = contraction.U_matrix_3          # [16,16,16,K3]
        U2 = contraction.U_matrix_2          # [16,16,K2]
        U1 = contraction.U_matrix_1          # [16,K1]
        W3 = contraction.weights_max         # [S,K3,C]
        W2 = contraction.weights[0]          # [S,K2,C]
        W1 = contraction.weights[1]          # [S,K1,C]
        n = NLM
        # [S, C, i, (w y)]
        A3 = torch.einsum('wyik,skc->sciwy', U3, W3).reshape(self.S, self.C, n, n * n)
        A2 = torch.einsum('wyk,skc->scwy', U2, W2).reshape(self.S, self.C, 1, n * n)
        A1 = torch.einsum('wk,skc->scw', U1, W1)

        def block(x, s):
            xt = x.transpose(0, 1)                      # [C, n_s, 16]
            m = torch.baddbmm(A2[s], xt, A3[s])         # [C, n_s, 256]
            m = m.view(self.C, -1, n, n)
            v = torch.einsum('cnwy,cny->cnw', m, xt) + A1[s][:, None, :]
            out = (v * xt).sum(-1)                      # [C, n_s]
            return out.transpose(0, 1)
        return self.species_blocks(a, bounds, block)

    # ----------------------------------------------------------------- energy
    def spherical_harmonics(self, vec: Tensor) -> Tensor:
        from e3nn import o3
        return o3.spherical_harmonics(list(range(LMAX + 1)), vec, True, 'component')

    def mlp_tangent(self, ef: Tensor, dEf: Tensor, net):
        """Radial MLP value and its derivative along r (forward-mode tangent of d ef/dr), stacked rows."""
        from .ops import silu_tangent
        layers = [getattr(net, f'layer{i}') for i in range(len(net.hs) - 1)]
        E = ef.shape[0]
        xx = torch.cat([ef, dEf], dim=0)
        for i, layer in enumerate(layers):
            w = layer.weight
            xx = xx @ (w / math.sqrt(w.shape[0]))
            if i < len(layers) - 1:
                xx = silu_tangent(xx, self.act)
        return xx[:E].view(E, LMAX + 1, self.C), xx[E:].view(E, LMAX + 1, self.C)

    def node_energy(self, b: Batch, vec: Tensor) -> Tensor:
        m = self.model
        C = self.C
        if self.basis is not None:
            Y, ef = self.basis(vec, self.bessel_w, float(self.prefactor), self.r_max, self.p)
        else:
            r = torch.linalg.norm(vec, dim=-1, keepdim=True)
            Y = self.spherical_harmonics(vec)
            ef = self.radial(r)
        i0, i1 = m.interactions
        R0 = self.mlp(ef, i0.conv_tp_weights).view(-1, LMAX + 1, C)
        R1 = self.mlp(ef, i1.conv_tp_weights).view(-1, LMAX + 1, C)
        return self.node_energy_from(b, Y, R0, R1)

    def node_energy_from(self, b: Batch, Y: Tensor, R0: Tensor, R1: Tensor) -> Tensor:
        m = self.model
        C, S = self.C, self.S
        N = b.species.shape[0]
        bounds = b.species_bounds
        self._species = b.species

        h = m.node_embedding.linear.weight.view(S, C)[b.species] / math.sqrt(S)
        i0, i1 = m.interactions
        p0, p1 = m.products

        x = self.lin(h, i0.linear_up.weight, C, C)
        a = self.conv(x, R0, Y, b, N)
        a = self.per_l_linear(a, i0.linear.weight) / self.avg[0]
        a = self.skip_first(a, i0.skip_tp.weight, bounds)
        f1 = self.product(a, p0.symmetric_contractions.contractions[0], bounds)
        f1 = self.lin(f1, p0.linear.weight, C, C)
        e_read0 = self.lin(f1, m.readouts[0].linear.weight, C, 1)

        sc = self.skip_scalar(f1, i1.skip_tp.weight, bounds)
        x = self.lin(f1, i1.linear_up.weight, C, C)
        a = self.conv(x, R1, Y, b, N)
        a = self.per_l_linear(a, i1.linear.weight) / self.avg[1]
        f2 = self.product(a, p1.symmetric_contractions.contractions[0], bounds)
        f2 = self.lin(f2, p1.linear.weight, C, C) + sc
        ro = m.readouts[1]
        hid = ro.linear_1.weight.numel() // C
        t = F.silu(self.lin(f2, ro.linear_1.weight, C, hid)) * self.act
        e_read1 = self.lin(t, ro.linear_2.weight, hid, 1)
        return (self.scale * (e_read0 + e_read1)).squeeze(-1)

    def graph_energy(self, b: Batch, vec: Tensor) -> Tensor:
        ne = self.node_energy(b, vec)
        return torch.zeros(b.num_graphs, dtype=ne.dtype, device=ne.device).index_add_(0, b.node_graph, ne)

    def edge_gradient(self, b: Batch, training: bool):
        """(energy [G], dE/dvectors [E, 3]) with the graph kept for the parameter backward if training."""
        if not self.tangent:
            vec = b.vectors.detach().requires_grad_(True)
            with torch.enable_grad():
                e = self.graph_energy(b, vec)
                (g,) = torch.autograd.grad(e.sum(), vec, create_graph=training)
            return e, g
        # Forward mode along r for the radial weights: dE/dvec = (sum_t dE/dR_t . dR_t/dr) u + JY^T dE/dY.
        # The force pass differentiates the energy to R and Y only, so the radial MLP is never
        # differentiated twice.
        from .ops import edge_basis_raw
        Y, ef, dEf, U, JY = edge_basis_raw(b.vectors, self.bessel_w, float(self.prefactor), self.r_max, self.p)
        i0, i1 = self.model.interactions
        with torch.enable_grad():
            Y = Y.requires_grad_(True)
            R0, T0 = self.mlp_tangent(ef, dEf, i0.conv_tp_weights)
            R1, T1 = self.mlp_tangent(ef, dEf, i1.conv_tp_weights)
            ne = self.node_energy_from(b, Y, R0, R1)
            e = torch.zeros(b.num_graphs, dtype=ne.dtype, device=ne.device).index_add_(0, b.node_graph, ne)
            gR0, gR1, gY = torch.autograd.grad(e.sum(), (R0, R1, Y), create_graph=training)
            gr = (gR0 * T0).sum((1, 2)) + (gR1 * T1).sum((1, 2))
            g = gr[:, None] * U + torch.bmm(gY.unsqueeze(1), JY).squeeze(1)
        return e, g

    def evaluate(self, b: Batch, training: bool):
        """Energy, forces and stress; with ``training`` the graph is kept for the parameter backward."""
        e, g = self.edge_gradient(b, training)
        N = b.species.shape[0]
        dpos = torch.zeros(N, 3, dtype=g.dtype, device=g.device)
        dpos = dpos.index_add(0, b.receiver, g).index_add(0, b.sender, -g)
        forces = -dpos + b.core_f
        m = torch.zeros(b.num_graphs, 3, 3, dtype=g.dtype, device=g.device)
        m = m.index_add(0, b.edge_graph, b.vectors[:, :, None] * g[:, None, :])
        stress = 0.5 * (m + m.transpose(1, 2)) / b.volume[:, None, None] + b.core_s
        energy = e + b.core_e
        return energy, forces, stress


def loss_terms(b: Batch, energy: Tensor, forces: Tensor, stress: Tensor):
    """The three reduced terms of mace-torch's WeightedEnergyForcesStressLoss (before weights)."""
    le = (b.w_energy * torch.square((b.e_ref - energy) / b.n_atoms)).mean()
    lf = (b.w_forces[:, None] * torch.square(b.f_ref - forces)).mean()
    ls = (b.w_stress[:, None, None] * torch.square(b.s_ref - stress)).mean()
    return le, lf, ls


# --------------------------------------------------------------------- reference convolution
def conv_reference(x: Tensor, R: Tensor, Y: Tensor, b: Batch, N: int) -> Tensor:
    """A[i,c,lm] = sum over edges e into i of R[e,l(lm),c] Y[e,lm] x[sender(e),c]. Materialises messages."""
    l_of_lm = L_OF_LM.to(R.device)
    msg = R[:, l_of_lm, :].transpose(1, 2) * Y[:, None, :] * x[b.sender][:, :, None]   # [E, C, 16]
    out = torch.zeros(N, x.shape[1], NLM, dtype=msg.dtype, device=msg.device)
    return out.index_add(0, b.receiver, msg)
