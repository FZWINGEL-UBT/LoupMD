"""Static-shape training engine for the supported mace-torch students (GPU, Triton kernels).

Every batch of an epoch is padded to one shape (nodes, edges, graphs), so the whole optimizer step
(energy, forces and stress, the loss, the parameter backward, clipping, Adam and the EMA update)
is captured once in a CUDA graph and replayed. Padding is exact:

* slack edges are self-loops on dummy nodes with a vector beyond the cutoff, so their radial
  features, radial weights and r-tangents are exactly zero and they carry no message and no force;
* dummy nodes have no real neighbours, belong to a dummy graph and carry zero loss weight;
* loss means divide by the real counts of atoms and graphs.

Node features use the component-major layout [16, N, C]; species-dependent weights are gathered
per node, so nothing depends on the species counts of a batch. All weights are folded on the
autograd tape from mace-torch's own ``Parameter`` objects.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor



class _LazyOps:
    """``loupmd.fasttrain.ops`` (the Triton kernels), imported at the first use: only :class:`FastStudent`'s
    evaluation uses it, so a run on the training library (its trainer and evaluators) never imports Triton."""

    def __getattr__(self, name):
        from . import ops as mod
        globals()['ops'] = mod
        return getattr(mod, name)


ops = _LazyOps()
from .model import LMAX, NLM, L_OF_LM, check_supported, check_poly_terms, silu_norm_constant

DUMMY_ROW = 32          # at most this many slack edges per dummy node
PAIRS = True            # radial MLPs per edge pair (both directions share |r|) when every edge has its reverse


# ============================================================================ data
def _host_topology(f):
    """A frame's sender, receiver, receiver row pointers, sender-sorted edge order and its row pointers (int64
    host arrays, read only), computed once per frame."""
    t = f.__dict__.get('_host_topology')
    if t is None:
        snd = f.sender.numpy().astype(np.int64)
        rcv = f.receiver.numpy().astype(np.int64)
        rptr = np.searchsorted(rcv, np.arange(f.n + 1)).astype(np.int64)
        sperm = np.argsort(snd, kind='stable').astype(np.int64)
        sptr = np.searchsorted(snd[sperm], np.arange(f.n + 1)).astype(np.int64)
        t = f.__dict__['_host_topology'] = (snd, rcv, rptr, sperm, sptr)
    return t


def _host_pairs(f):
    """A frame's edge-pair ids, and the canonical edge and its reverse of each pair (one direction per pair)."""
    t = f.__dict__.get('_host_pairs')
    if t is None:
        part = np.asarray(f.partner, dtype=np.int64)
        E = part.shape[0]
        canon = np.nonzero(np.arange(E) < part)[0]
        pid = np.empty(E, dtype=np.int64)
        pid[canon] = np.arange(canon.shape[0])
        pid[part[canon]] = pid[canon]
        t = f.__dict__['_host_pairs'] = (pid, canon, part[canon])
    return t


_DEVICE_ARRAYS = {}        # id(frames list) -> (the list, {(dtype, device, r_max): arrays})


def _device_arrays(frames, n, dtype, device, r_max):
    """The per-node, per-edge and per-frame arrays of a frame list on the device, with one appended dummy row;
    read only, so kept per list (the frame cache hands the trainer and the evaluators of one set the same list)
    and shared by every batch size."""
    entry = _DEVICE_ARRAYS.get(id(frames))
    if entry is None or entry[0] is not frames:
        entry = _DEVICE_ARRAYS[id(frames)] = (frames, {})
    key = (dtype, str(torch.device(device)), float(r_max))
    hit = entry[1].get(key)
    if hit is not None:
        return hit
    kw = dict(device=device)
    cat = lambda xs, dt: torch.cat(xs).to(dtype=dt, **kw)
    far = torch.tensor([[2.0 * r_max + 1.0, 0.0, 0.0]], dtype=torch.float64)
    scal = lambda key, dummy: torch.tensor([float(getattr(f, key)) for f in frames] + [dummy],
                                           dtype=torch.float64).to(dtype=dtype, **kw)
    out = dict(
        species=cat([f.species for f in frames] + [torch.zeros(1, dtype=torch.long)], torch.long),
        f_ref=cat([f.forces for f in frames] + [torch.zeros(1, 3, dtype=torch.float64)], dtype),
        core_f=cat([f.core_f for f in frames] + [torch.zeros(1, 3, dtype=torch.float64)], dtype),
        vectors=cat([f.vectors for f in frames] + [far], dtype),
        e_ref=scal('energy', 0.0), core_e=scal('core_e', 0.0), volume=scal('volume', 1.0),
        w_e=scal('w_energy', 0.0), w_f=scal('w_forces', 0.0), w_s=scal('w_stress', 0.0),
        n_atoms=torch.tensor(list(n) + [1], dtype=dtype, **kw),
        s_ref=cat([f.stress[None] for f in frames] + [torch.zeros(1, 3, 3, dtype=torch.float64)], dtype),
        core_s=cat([f.core_s[None] for f in frames] + [torch.zeros(1, 3, 3, dtype=torch.float64)], dtype))
    entry[1][key] = out
    return out


class PaddedFrames:
    """Frames resident on the device, and the index arrays of padded static batches."""

    def __init__(self, frames, dtype, device, batch_size: int, r_max: float, n_batches_hint=None):
        self.frames = frames
        self.dtype, self.device = dtype, device
        self.B = int(batch_size)
        nf = len(frames)
        n = np.array([f.n for f in frames], dtype=np.int64)
        e = np.array([f.sender.numel() for f in frames], dtype=np.int64)
        self.n, self.e = n, e
        b = min(self.B, nf)
        ns, es = np.sort(n), np.sort(e)
        n_max, e_max = int(ns[-b:].sum()), int(es[-b:].sum())
        # the fewest edges any batch can have (a short last batch included)
        short = nf % self.B or self.B
        e_min = int(es[:min(short, nf)].sum())
        self.D = max(1, math.ceil((e_max - e_min) / DUMMY_ROW) + 1)
        self.N_pad = n_max + self.D
        # edge pairs: every edge has its reverse (same length) in every frame -> the radial MLPs run per pair
        self.paired = PAIRS and all(f.partner is not None for f in frames)
        self.E_pad = e_max + (e_max % 2 if self.paired else 0)
        self.P_pad = self.E_pad // 2
        self.G1 = self.B + 1                       # graph slots plus the dummy graph
        self.node_off = np.concatenate([[0], np.cumsum(n)])
        self.edge_off = np.concatenate([[0], np.cumsum(e)])
        self.__dict__.update(_device_arrays(frames, n, dtype, device, r_max))
        # host-side local topology per frame (receiver-sorted edges), kept on the frames: the trainer and the
        # evaluators of one set (validation, the post-training tables) share them
        topo = [_host_topology(f) for f in frames]
        self.snd_l, self.rcv_l, self.rptr_l, self.sperm_l, self.sptr_l = (list(x) for x in zip(*topo))
        if self.paired:
            pairs = [_host_pairs(f) for f in frames]
            self.pid_l, self.pe0_l, self.pe1_l = (list(x) for x in zip(*pairs))
        self.dummy_node_row = int(self.node_off[-1])
        self.dummy_edge_row = int(self.edge_off[-1])
        self.dummy_frame = nf

    def host_batch(self, ids: Sequence[int]):
        """Index arrays of one padded batch (numpy)."""
        ids = [int(i) for i in ids]
        G = len(ids)
        assert 0 < G <= self.B
        node_rows, edge_rows, snd, rcv, rptr, sperm, sptr = [], [], [], [], [], [], []
        ngraph, egraph = [], []
        pid, pe0, pe1 = [], [], []
        n0 = e0 = p0 = 0
        for g, i in enumerate(ids):
            ni, ei = int(self.n[i]), int(self.e[i])
            node_rows.append(np.arange(self.node_off[i], self.node_off[i + 1]))
            edge_rows.append(np.arange(self.edge_off[i], self.edge_off[i + 1]))
            snd.append(self.snd_l[i] + n0)
            rcv.append(self.rcv_l[i] + n0)
            rptr.append(self.rptr_l[i][:-1] + e0)
            sperm.append(self.sperm_l[i] + e0)
            sptr.append(self.sptr_l[i][:-1] + e0)
            ngraph.append(np.full(ni, g))
            egraph.append(np.full(ei, g))
            if self.paired:
                pid.append(self.pid_l[i] + p0)
                pe0.append(self.pe0_l[i] + e0)
                pe1.append(self.pe1_l[i] + e0)
                p0 += self.pe0_l[i].shape[0]
            n0 += ni
            e0 += ei
        n_real, e_real = n0, e0
        nd = self.N_pad - n_real                  # dummy nodes n_real .. N_pad-1
        ed = self.E_pad - e_real
        assert nd >= 1 and ed >= 0 and math.ceil(ed / DUMMY_ROW) <= nd, (nd, ed)
        # slack edge j -> self-loop on dummy node n_real + j // DUMMY_ROW (receiver-sorted)
        dn = n_real + np.arange(ed) // DUMMY_ROW
        node_rows.append(np.full(nd, self.dummy_node_row))
        edge_rows.append(np.full(ed, self.dummy_edge_row))
        snd.append(dn)
        rcv.append(dn)
        ngraph.append(np.full(nd, self.B))
        egraph.append(np.full(ed, self.B))
        # CSR of the dummy rows: receiver rows and sender rows coincide for self-loops
        dcount = np.bincount(dn - n_real, minlength=nd)
        dptr = e_real + np.concatenate([[0], np.cumsum(dcount)])
        rptr.append(dptr)
        sptr.append(dptr)
        sperm.append(np.arange(e_real, self.E_pad))
        graph_rows = np.array(ids + [self.dummy_frame] * (self.G1 - G))
        cat = np.concatenate
        extra = {}
        if self.paired:
            assert ed % 2 == 0 and p0 * 2 == e_real
            k = np.arange(ed // 2)
            pid.append(p0 + np.repeat(k, 2))                       # slack edges paired consecutively
            pe0.append(e_real + 2 * k)
            pe1.append(e_real + 2 * k + 1)
            extra = dict(pidx=cat(pid), pe0=cat(pe0), pe1=cat(pe1))
            assert extra['pe0'].shape[0] == self.P_pad
        out = dict(node_rows=cat(node_rows), edge_rows=cat(edge_rows), sender=cat(snd), receiver=cat(rcv),
                   receiver_ptr=cat(rptr), sender_perm=cat(sperm), sender_ptr=cat(sptr),
                   node_graph=cat(ngraph), edge_graph=cat(egraph), graph_rows=graph_rows,
                   counts=np.array([n_real, G], dtype=np.float64), **extra)
        assert out['node_rows'].shape[0] == self.N_pad and out['edge_rows'].shape[0] == self.E_pad
        assert out['receiver_ptr'].shape[0] == self.N_pad + 1
        return out

    INT32 = ('receiver_ptr', 'sender_perm', 'sender_ptr')

    def device_batches(self, batches: Sequence[Sequence[int]]):
        """Stack the index arrays of all batches of an epoch on the device: dict of [nb, ...]."""
        hs = [self.host_batch(ids) for ids in batches]
        out = {}
        for k in hs[0]:
            arr = np.stack([h[k] for h in hs])
            if k == 'counts':
                out[k] = torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)
            elif k in self.INT32:
                out[k] = torch.from_numpy(arr.astype(np.int32)).to(self.device)
            else:
                out[k] = torch.from_numpy(arr.astype(np.int64)).to(self.device)
        return out

    def device_batch(self, h):
        """One batch's index arrays (from ``host_batch``) on the device."""
        out = {}
        for k, v in h.items():
            if k == 'counts':
                out[k] = torch.from_numpy(v).to(device=self.device, dtype=self.dtype)
            elif k in self.INT32:
                out[k] = torch.from_numpy(v.astype(np.int32)).to(self.device)
            else:
                out[k] = torch.from_numpy(v.astype(np.int64)).to(self.device)
        return out

    def static_buffers(self):
        """Empty static index buffers of one batch."""
        h = self.host_batch(list(range(min(self.B, len(self.frames)))))
        bufs = {}
        for k, v in h.items():
            if k == 'counts':
                bufs[k] = torch.zeros(v.shape, device=self.device, dtype=self.dtype)
            elif k in self.INT32:
                bufs[k] = torch.zeros(v.shape, device=self.device, dtype=torch.int32)
            else:
                bufs[k] = torch.zeros(v.shape, device=self.device, dtype=torch.int64)
        return bufs

    def gather(self, idx) -> SimpleNamespace:
        """The padded batch from its index buffers (all static shapes; capturable)."""
        gr = idx['graph_rows']
        ng = idx['node_graph']
        nr = idx['node_rows']
        return SimpleNamespace(
            mnc=True, num_graphs=self.G1,
            vectors=self.vectors[idx['edge_rows']], sender=idx['sender'], receiver=idx['receiver'],
            receiver_ptr=idx['receiver_ptr'], sender_perm=idx['sender_perm'], sender_ptr=idx['sender_ptr'],
            species=self.species[nr], node_graph=ng, edge_graph=idx['edge_graph'],
            f_ref=self.f_ref[nr], core_f=self.core_f[nr], w_f=self.w_f[gr][ng],
            e_ref=self.e_ref[gr], core_e=self.core_e[gr], volume=self.volume[gr], n_atoms=self.n_atoms[gr],
            w_e=self.w_e[gr], w_s=self.w_s[gr], s_ref=self.s_ref[gr], core_s=self.core_s[gr],
            counts=idx['counts'], pidx=idx.get('pidx'), pe0=idx.get('pe0'), pe1=idx.get('pe1'))


# ============================================================================ model
class FastStudent:
    """Energy, forces and stress of a padded static batch, sharing mace-torch's parameters."""

    def __init__(self, model):
        check_supported(model)
        check_poly_terms(model)
        self.model = model
        self.S = int(len(model.atomic_numbers))
        self.C = int(model.products[0].linear.irreps_out.dim)
        self.avg = [float(i.avg_num_neighbors) for i in model.interactions]
        be = model.radial_embedding.bessel_fn
        self.bw = be.bessel_weights
        self.prefactor = float(be.prefactor)
        self.r_max = float(model.radial_embedding.cutoff_fn.r_max)
        self.p = int(model.radial_embedding.cutoff_fn.p)
        self.act = silu_norm_constant()
        self.l_of_m = L_OF_LM.to(self.bw.device)
        for inter in model.interactions:
            assert len(inter.conv_tp_weights.hs) == 5, 'radial MLP depth'

    # ------------------------------------------------------------------ pieces
    def mlp_tangent(self, ef: Tensor, dEf: Tensor, net):
        """Radial weights R and their r-derivative T: the MLP and its forward-mode tangent."""
        layers = [getattr(net, f'layer{i}') for i in range(len(net.hs) - 1)]
        E = ef.shape[0]
        h, dh = ef, dEf
        for i, layer in enumerate(layers):
            w = layer.weight / math.sqrt(layer.weight.shape[0])
            h, dh = h @ w, dh @ w
            if i < len(layers) - 1:
                h, dh = ops.silu_tangent2(h, dh, self.act)
        return h.view(E, LMAX + 1, self.C), dh.view(E, LMAX + 1, self.C)

    def radial(self, ef: Tensor, dEf: Tensor):
        """(R_0, T_0, R_1, T_1) [E, 4, C]: both interactions' radial weights and their r-tangents."""
        nets = [i.conv_tp_weights for i in self.model.interactions]
        Ws = []
        for k in range(4):
            ws = [getattr(n, f'layer{k}').weight for n in nets]
            Ws.append(torch.stack([w / math.sqrt(w.shape[0]) for w in ws]))
        E = ef.shape[0]
        outs = ops.radial_mlps(torch.cat([ef, dEf], dim=0), *Ws, self.act)
        return tuple(o.view(E, LMAX + 1, self.C) for o in outs)

    def _per_m(self, w: Tensor) -> Tensor:
        """[4, ...] per-degree weights -> [16, ...] per component (expand + cat: cheap backward)."""
        return torch.cat([w[l:l + 1].expand(2 * l + 1, *w.shape[1:]) for l in range(LMAX + 1)], dim=0)

    def _pick(self, y: Tensor, species: Tensor) -> Tensor:
        """y [..., N, S*C] -> y[..., n, species(n)*C : (species(n)+1)*C]."""
        S, C = self.S, self.C
        shp = y.shape
        y = y.view(*shp[:-1], S, C)
        idx = species.view(*([1] * (len(shp) - 2)), -1, 1, 1).expand(*shp[:-1], 1, C)
        return torch.gather(y, -2, idx).squeeze(-2)

    def product(self, a: Tensor, contraction, species: Tensor) -> Tensor:
        """a [16, N, C] -> [N, C]: sum_k W[s(n), k, c] P_k(a[:, n, c])."""
        W = torch.cat([contraction.weights_max, contraction.weights[0], contraction.weights[1]], dim=1)
        P = ops.product_poly_cm(a)                         # [K, N, C]
        S, K, C = W.shape
        Wn = (self._onehot @ W.reshape(S, K * C)).view(-1, K, C).transpose(0, 1)   # [K, N, C]
        return (P * Wn).sum(0)

    def node_energy(self, bt, Y: Tensor, R0: Tensor, R1: Tensor, conv_io: Optional[list] = None) -> Tensor:
        m = self.model
        S, C = self.S, self.C
        sp = bt.species
        N = sp.shape[0]
        i0, i1 = m.interactions
        p0, p1 = m.products
        rC = math.sqrt(C)
        # layer 0: embedding folded with linear_up
        emb = m.node_embedding.linear.weight.view(S, C) / math.sqrt(S)
        self._onehot = (sp[:, None] == torch.arange(S, device=sp.device)).to(emb.dtype)
        x = self._onehot @ (emb @ (i0.linear_up.weight.view(C, C) / rC))
        a = ops.node_g(bt, [(R0, Y)], x)                                   # [16, N, C]
        if conv_io is not None:
            if not a.requires_grad:        # frozen parameters (mace's final tables): forces still need dE/da
                a.requires_grad_(True)
            conv_io.append((a, x))
        # per-degree linear (with 1/avg) folded with the species skip: [4, C, S*C]
        wl = i0.linear.weight.view(LMAX + 1, C, C) / (rC * self.avg[0])
        ws = i0.skip_tp.weight.view(LMAX + 1, C, S, C) / math.sqrt(C * S)
        M = torch.einsum('lcd,ldse->lcse', wl, ws).reshape(LMAX + 1, C, S * C)
        a = self._pick(torch.bmm(a, self._per_m(M)), sp)                  # [16, N, C]
        f = self.product(a, p0.symmetric_contractions.contractions[0], sp)
        # f1 = f Wp0 feeds the first readout, layer 1's linear_up and its species skip: one GEMM
        wp0 = p0.linear.weight.view(C, C) / rC
        wr0 = m.readouts[0].linear.weight.view(C, 1) / rC
        wu1 = i1.linear_up.weight.view(C, C) / rC
        wsk1 = i1.skip_tp.weight.view(C, S * C) / math.sqrt(C * S)
        z = f @ (wp0 @ torch.cat([wr0, wu1, wsk1], dim=1))                 # [N, 1 + C + S*C]
        e0 = z[:, 0]
        x = z[:, 1:1 + C].contiguous()
        sc = self._pick(z[:, 1 + C:], sp)
        # layer 1
        a = ops.node_g(bt, [(R1, Y)], x)
        if conv_io is not None:
            conv_io.append((a, x))
        wl1 = i1.linear.weight.view(LMAX + 1, C, C) / (rC * self.avg[1])
        a = torch.bmm(a, self._per_m(wl1))
        f = self.product(a, p1.symmetric_contractions.contractions[0], sp)
        f2 = f @ (p1.linear.weight.view(C, C) / rC) + sc
        ro = m.readouts[1]
        hid = ro.linear_1.weight.numel() // C
        t = torch.nn.functional.silu(f2 @ (ro.linear_1.weight.view(C, hid) / rC)) * self.act
        e1 = (t @ (ro.linear_2.weight.view(hid, 1) / math.sqrt(hid))).squeeze(-1)
        return m.scale_shift.scale * (e0 + e1)

    def edge_gradient(self, bt, training: bool):
        Y, ef, dEf, U, JY = ops.edge_basis_raw(bt.vectors, self.bw, self.prefactor, self.r_max, self.p)
        if getattr(bt, 'pidx', None) is not None:          # radial weights per edge pair
            ef, dEf = ef[bt.pe0], dEf[bt.pe0]
        with torch.enable_grad():
            R0, T0, R1, T1 = self.radial(ef, dEf)
            io = []
            ne = self.node_energy(bt, Y, R0, R1, conv_io=io)
            e = torch.zeros(bt.num_graphs, dtype=ne.dtype, device=ne.device).index_add_(0, bt.node_graph, ne)
            (a0, x0), (a1, x1) = io
            G0, G1 = torch.autograd.grad(e.sum(), (a0, a1), create_graph=training)
            s0, gy0 = ops.force_edge(bt, G0, R0, T0, Y, x0)
            s1, gy1 = ops.force_edge(bt, G1, R1, T1, Y, x1)
            g = ops.edge_vec(s0 + s1, gy0 + gy1, U, JY)
        return e, g

    def evaluate(self, bt, training: bool):
        e, g = self.edge_gradient(bt, training)
        N = bt.species.shape[0]
        dpos = torch.zeros(N, 3, dtype=g.dtype, device=g.device)
        dpos = dpos.index_add(0, bt.receiver, g).index_add(0, bt.sender, -g)
        forces = -dpos + bt.core_f
        mv = torch.zeros(bt.num_graphs, 3, 3, dtype=g.dtype, device=g.device)
        mv = mv.index_add(0, bt.edge_graph, bt.vectors[:, :, None] * g[:, None, :])
        stress = 0.5 * (mv + mv.transpose(1, 2)) / bt.volume[:, None, None] + bt.core_s
        return e + bt.core_e, forces, stress

    @staticmethod
    def loss_terms(bt, energy, forces, stress):
        """mace-torch's WeightedEnergyForcesStressLoss terms over the real atoms and graphs."""
        n_real, g_real = bt.counts[0], bt.counts[1]
        le = (bt.w_e * torch.square((bt.e_ref - energy) / bt.n_atoms)).sum() / g_real
        lf = (bt.w_f[:, None] * torch.square(bt.f_ref - forces)).sum() / (3.0 * n_real)
        ls = (bt.w_s[:, None, None] * torch.square(bt.s_ref - stress)).sum() / (9.0 * g_real)
        return le, lf, ls


# ============================================================================ trainer
class GraphTrainer:
    """mace-torch optimizer steps for one model and training set, captured in one CUDA graph."""

    WARMUP = 3

    def __init__(self, model, dataset, batch_size: int, dtype=torch.float32, use_graph: bool = True):
        from .warm import join
        join()                              # the warm-up thread works on the legacy stream: done before captures
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = dtype
        from .data import frames_from_dataset
        t0 = time.time()
        frames = frames_from_dataset(dataset, model)
        self.data = PaddedFrames(frames, dtype, self.device, batch_size, float(model.r_max))
        self.student = FastStudent(model)
        self.idx = self.data.static_buffers()
        self.loss_w = torch.zeros(3, device=self.device, dtype=dtype)
        self.one_minus_decay = torch.zeros((), device=self.device, dtype=dtype)
        self.static_loss = torch.zeros((), device=self.device, dtype=dtype)
        self.use_graph = use_graph
        self.graph = None
        self.steps_done = 0
        self.setup_seconds = time.time() - t0
        self._opt_key = None

    # ------------------------------------------------------------------ the step body
    def _body(self, params, optimizer, ema_pairs, max_grad_norm):
        bt = self.data.gather(self.idx)
        e, f, s = self.student.evaluate(bt, training=True)
        le, lf, ls = self.student.loss_terms(bt, e, f, s)
        loss = self.loss_w[0] * le + self.loss_w[1] * lf + self.loss_w[2] * ls
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
        optimizer.step()
        if ema_pairs is not None:
            shadow, live = ema_pairs
            tmp = torch._foreach_sub(shadow, live)
            torch._foreach_mul_(tmp, self.one_minus_decay)
            torch._foreach_sub_(shadow, tmp)
        self.static_loss.copy_(loss.detach())

    def _set_loss_weights(self, loss_fn):
        w = torch.stack([loss_fn.energy_weight, loss_fn.forces_weight, loss_fn.stress_weight]).to(self.loss_w)
        self.loss_w.copy_(w)

    def _ema_pairs(self, ema):
        if ema is None:
            return None
        live = [r() for r in ema._params_refs] if hasattr(ema, '_params_refs') else None
        if live is None or any(p is None for p in live):
            live = [p for p in self.model.parameters() if p.requires_grad]
        return list(ema.shadow_params), live

    def _ema_tick(self, ema):
        if ema is None:
            return
        decay = ema.decay
        if ema.num_updates is not None:
            ema.num_updates += 1
            decay = min(decay, (1 + ema.num_updates) / (10 + ema.num_updates))
        self.one_minus_decay.fill_(1.0 - decay)

    def _prepare_optimizer(self, optimizer):
        """Device-tensor learning rates (replays read them) and the fused, capturable Adam path."""
        if self._opt_key is optimizer:
            return
        self._opt_key = optimizer
        self.lr_t = []
        for g in optimizer.param_groups:
            g['capturable'] = True
            g['fused'] = True
            g['foreach'] = False
            lr = g['lr']
            t = torch.tensor(float(lr), device=self.device, dtype=torch.float32)
            self.lr_t.append(t)

    def _sync_lr(self, optimizer):
        for g, t in zip(optimizer.param_groups, self.lr_t):
            t.fill_(float(g['lr']))

    def _with_tensor_lr(self, optimizer, fn):
        saved = [g['lr'] for g in optimizer.param_groups]
        for g, t in zip(optimizer.param_groups, self.lr_t):
            g['lr'] = t
        try:
            fn()
        finally:
            for g, lr in zip(optimizer.param_groups, saved):
                g['lr'] = lr

    def run_epoch(self, batches: List[List[int]], loss_fn, optimizer, ema, max_grad_norm):
        """Run one epoch of optimizer steps; returns per-step losses (floats) and seconds."""
        t_start = time.time()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self._prepare_optimizer(optimizer)
        self._sync_lr(optimizer)
        self._set_loss_weights(loss_fn)
        ema_pairs = self._ema_pairs(ema)
        stacks = self.data.device_batches(batches)
        nb = len(batches)
        losses = torch.empty(nb, device=self.device, dtype=self.dtype)
        body = lambda: self._body(params, optimizer, ema_pairs, max_grad_norm)
        for i in range(nb):
            for k, buf in self.idx.items():
                buf.copy_(stacks[k][i])
            self._ema_tick(ema)
            if self.graph is not None:
                self.graph.replay()
            elif not self.use_graph:
                optimizer.zero_grad(set_to_none=True)
                self._with_tensor_lr(optimizer, body)
            elif self.steps_done < self.WARMUP:
                # warm-up steps (real steps) on a side stream, as graph capture requires
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    optimizer.zero_grad(set_to_none=True)
                    self._with_tensor_lr(optimizer, body)
                torch.cuda.current_stream().wait_stream(side)
            else:
                self._capture(optimizer, body)
                self.graph.replay()
            losses[i].copy_(self.static_loss)
            self.steps_done += 1
        out = losses.tolist()
        return out, time.time() - t_start

    def _capture(self, optimizer, body):
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True)
        g = torch.cuda.CUDAGraph()
        saved = [g_['lr'] for g_ in optimizer.param_groups]
        for g_, t in zip(optimizer.param_groups, self.lr_t):
            g_['lr'] = t
        try:
            with torch.cuda.graph(g):
                body()
        finally:
            for g_, lr in zip(optimizer.param_groups, saved):
                g_['lr'] = lr
        self.graph = g


# ============================================================================ evaluation
class LightBatch:
    """The reference fields of a collated mace batch that ``MACELoss`` and mace's energy / forces / stress
    losses read (labels, weights, ``ptr``, ``batch``); any other field reads as None, as on mace's batch."""

    def __init__(self, fields, num_graphs):
        self.__dict__.update(fields)
        self.num_graphs = num_graphs

    def __getattr__(self, name):            # only for fields not set
        if name.startswith('__'):
            raise AttributeError(name)
        return None

    def __getitem__(self, key):
        return getattr(self, key)


_LIGHT_GRAPH = ('energy', 'stress', 'virials', 'weight', 'energy_weight', 'forces_weight', 'stress_weight',
                'virials_weight', 'head')
_LIGHT_NEEDED = ('energy', 'forces', 'stress')


class Evaluator:
    """Energies, forces and stress of mace-torch batches (validation), in mace's atom order."""

    def __init__(self, model, dataset, batch_size: int):
        from .warm import join
        join()                              # the warm-up thread works on the legacy stream: done before captures
        from .data import frames_from_dataset
        self.model = model
        self.dataset = dataset
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        frames = frames_from_dataset(dataset, model)
        self.data = PaddedFrames(frames, self.dtype, self.device, batch_size, float(model.r_max))
        self.student = FastStudent(model)
        # engine rows are sorted by species within a frame: row j is atom order[j]
        self.orders = [torch.sort(item.node_attrs.argmax(dim=1).cpu(), stable=True).indices.numpy()
                       for item in dataset]

    def rebind(self, model):
        """Evaluate ``model`` (same architecture, frames and constants) from now on."""
        self.student = FastStudent(model)
        self.model = model

    def run(self, batches, data_loader, light=False):
        """Yields (ids, collated mace batch on the device, output dict) per batch.

        The collated batches are kept on the device by their frame ids: validation draws the same batches in
        every epoch, and mace's collate and copy cost several times the model's evaluation. ``light=True``
        yields a :class:`LightBatch` (the reference fields mace's ``MACELoss`` and losses read) instead, when
        the items carry energy, forces and stress labels."""
        light = light and self._light_ok()
        cache = self.__dict__.setdefault('_light_batches' if light else '_batches', {})
        for ids in batches:
            key = tuple(int(i) for i in ids)
            hit = cache.get(key)
            if hit is None:
                dev = self.data.device_batch(self.data.host_batch(ids))
                n0, perm = 0, []
                for i in ids:
                    perm.append(self.orders[i] + n0)
                    n0 += int(self.data.n[i])
                perm = torch.from_numpy(np.concatenate(perm)).to(self.device)
                if light:
                    batch = self._light_batch(key)
                else:
                    batch = data_loader.collate_fn([self.dataset[i] for i in ids]).to(self.device)
                hit = cache[key] = (dev, perm, n0, batch)
            dev, perm, n0, batch = hit
            bt = self.data.gather(dev)
            e, f, s = self.student.evaluate(bt, training=False)
            G = len(ids)
            forces = torch.empty(n0, 3, device=self.device, dtype=f.dtype)
            forces[perm] = f[:n0].detach()
            stress = s[:G].detach()
            vol = bt.volume[:G]
            yield ids, batch, dict(energy=e[:G].detach(), forces=forces, stress=stress,
                                   virials=-stress * vol[:, None, None])

    def _light_ok(self):
        ok = self.__dict__.get('_light_ok_v')
        if ok is None:
            # MACELoss reads dipole / polarizability labels only when the output has them (ours never does)
            ok = self._light_ok_v = all(getattr(it, k, None) is not None for it in self.dataset for k in _LIGHT_NEEDED)
        return ok

    def _light_batch(self, ids):
        """mace's collate of these items, restricted to the fields :class:`LightBatch` carries: graph fields
        are the items' values stacked (a 0-d value as one row), forces concatenated in item order."""
        st = self.__dict__.get('_light')
        if st is None:
            graph = {}
            for k in _LIGHT_GRAPH:
                vals = [getattr(it, k, None) for it in self.dataset]
                if all(v is None for v in vals):
                    continue
                graph[k] = torch.cat([v.reshape(1) if v.dim() == 0 else v for v in vals]).to(self.device)
            forces = [it.forces.to(self.device) for it in self.dataset]
            st = self._light = (graph, forces)
        graph, forces = st
        dev = self.device
        idx = torch.tensor(ids, device=dev)
        fields = {k: v.index_select(0, idx) for k, v in graph.items()}
        counts = [int(forces[i].shape[0]) for i in ids]
        fields['forces'] = torch.cat([forces[i] for i in ids])
        fields['ptr'] = torch.tensor(np.concatenate([[0], np.cumsum(counts)]), dtype=torch.long, device=dev)
        fields['batch'] = torch.repeat_interleave(torch.arange(len(ids), device=dev),
                                                  torch.tensor(counts, device=dev))
        return LightBatch(fields, len(ids))
