"""Immutable frames on the device and fast batch packing for the training engine.

Each frame keeps the topology of mace-torch's own ``AtomicData`` (directed ``(sender, receiver)``
edges with integer images, no zero-image self edge), its labels and per-frame weights, and the
float64-evaluated scaled ZBL core (energy, forces, stress). Nodes are relabelled so that a packed
batch has contiguous species blocks; edges are sorted by receiver. Loss terms are permutation
invariant, and reference forces are permuted with the nodes.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .model import Batch


@dataclass
class Frame:
    n: int
    counts: List[int]          # atoms per species
    starts: List[int]          # species block start in the frame's sorted order
    species: Tensor            # [n] sorted
    sender: Tensor             # [E] local sorted ids
    receiver: Tensor
    vectors: Tensor            # [E, 3]
    volume: float
    energy: float
    forces: Tensor             # [n, 3] in sorted order
    stress: Tensor             # [3, 3]
    w_energy: float
    w_forces: float
    w_stress: float
    core_e: float
    core_f: Tensor
    core_s: Tensor
    partner: Optional[np.ndarray] = None   # [E] index of the reverse edge (same length), if all have one
    node_order: Optional[Tensor] = None    # [n] sorted index of each atom in the item's own order


def _core(model, species: Tensor, sender: Tensor, receiver: Tensor, vectors: Tensor, volume: float, zbl=None):
    """Scaled ZBL energy/forces/stress of one frame in float64 at fixed geometry, plus E0 and shift."""
    import copy
    if zbl is None:
        zbl = copy.deepcopy(model.pair_repulsion_fn).double().cpu()
    S = len(model.atomic_numbers)
    node_attrs = torch.nn.functional.one_hot(species, S).double()
    zs = model.atomic_numbers.cpu()
    vec = vectors.clone().double().requires_grad_(True)
    r = torch.linalg.norm(vec, dim=-1, keepdim=True)
    edge_index = torch.stack([sender, receiver])
    scale = float(model.scale_shift.scale.double().item())
    shift = float(model.scale_shift.shift.double().item())
    e_nodes = zbl(r, node_attrs, edge_index, zs)
    e = scale * e_nodes.sum()
    (g,) = torch.autograd.grad(e, vec)
    n = species.shape[0]
    dpos = torch.zeros(n, 3, dtype=torch.float64).index_add(0, receiver, g).index_add(0, sender, -g)
    m = (vectors.double()[:, :, None] * g[:, None, :]).sum(0)
    stress = 0.5 * (m + m.T) / volume
    e0 = model.atomic_energies_fn.atomic_energies.double().cpu().reshape(-1)[species].sum()
    return float(e.detach()) + float(e0) + shift * n, -dpos, stress


_FRAMES = {}


@contextlib.contextmanager
def host_threads():
    """torch's intra-op threads limited to ``LOUP_FT_DATA_THREADS`` (default 1; 0 leaves them) for per-frame host
    work on small tensors: there the OpenMP team (24 threads on the 5060 Ti PC) costs more than it gives and
    stalls beside the driver's library loads (:mod:`loupmd.fasttrain.warm`). Copies, indexing, stable sorts,
    one-hot encodings and 3x3 determinants give the same bits with any thread count."""
    want = int(os.environ.get('LOUP_FT_DATA_THREADS', '1'))
    n0 = torch.get_num_threads()
    if want <= 0 or want >= n0:
        yield
        return
    torch.set_num_threads(want)
    try:
        yield
    finally:
        torch.set_num_threads(n0)


def _model_constants(model):
    """What a frame depends on besides its item: the ZBL core's and E0 / scale / shift values."""
    vals = [model.atomic_energies_fn.atomic_energies, model.scale_shift.scale, model.scale_shift.shift,
            model.atomic_numbers]
    vals += list(model.pair_repulsion_fn.state_dict().values())
    return tuple(tuple(v.detach().double().cpu().reshape(-1).tolist()) for v in vals)


def frames_from_dataset(dataset: Sequence, model) -> List[Frame]:
    """Convert mace-torch ``AtomicData`` items (one head, stress labels) into engine frames.

    Kept per item list (the same objects) and model constants (E0, scale, shift, ZBL): the trainer and the
    evaluators of mace's post-training tables and plots (a loaded checkpoint) convert the training set once."""
    items = list(dataset)
    key = (len(items), id(items[0]) if items else 0)   # a loaded checkpoint shares the training frames
    consts = _model_constants(model)
    hit = _FRAMES.get(key)
    if hit is not None and hit[1] == consts and all(a is b for a, b in zip(hit[0], items)):
        return hit[2]
    frames = _convert(items, model)
    _FRAMES[key] = (items, consts, frames)
    return frames


def _convert(dataset: Sequence, model) -> List[Frame]:
    """Per-frame relabelling on the host, then the ZBL core terms and the reverse-edge pairing of all frames
    at once (chunks of frames, on the model's device); ``LOUP_FT_FRAMES_BATCHED=0`` keeps the per-frame
    host path (``_core`` / ``reverse_edges``)."""
    if os.environ.get('LOUP_FT_FRAMES_BATCHED', '1') == '0':
        return _convert_serial(dataset, model)
    with host_threads():
        return _convert_batched(dataset, model)


def _inverse(order: Tensor) -> Tensor:
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel())
    return inv


def _convert_batched(dataset: Sequence, model) -> List[Frame]:
    dev = next(model.parameters()).device
    preps = []
    for item in dataset:
        species0 = item.node_attrs.argmax(dim=1).cpu()
        order = torch.sort(species0, stable=True).indices
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel())
        species = species0[order]
        S = item.node_attrs.shape[1]
        counts = torch.bincount(species, minlength=S).tolist()
        starts = [0] + list(np.cumsum(counts)[:-1].astype(int))
        sender = inv[item.edge_index[0].cpu()]
        receiver = inv[item.edge_index[1].cpu()]
        pos = item.positions.cpu().double()
        shifts = item.shifts.cpu().double()
        vectors = pos[item.edge_index[1]] - pos[item.edge_index[0]] + shifts
        e_order = torch.sort(receiver, stable=True).indices
        sender, receiver, vectors = sender[e_order], receiver[e_order], vectors[e_order]
        cell = item.cell.cpu().double().view(3, 3)
        volume = float(torch.linalg.det(cell).abs())
        preps.append((item, order, species, counts, starts, sender, receiver, vectors, volume))
    core = []
    partners = []
    CH = 32                                   # frames per chunk (bounds the [E, 3, 3] stress products)
    for c0 in range(0, len(preps), CH):
        chunk = preps[c0:c0 + CH]
        core += _core_frames(model, chunk, os.environ.get('LOUP_FT_CORE_DEV') or dev)
        partners += _pair_frames(chunk, os.environ.get('LOUP_FT_PAIR_DEV') or dev)
    frames = []
    for (item, order, species, counts, starts, sender, receiver, vectors, volume), (ce, cf, cs), partner in zip(
            preps, core, partners):
        w = float(item.weight)
        frames.append(Frame(
            n=species.numel(), counts=counts, starts=starts, species=species, sender=sender,
            receiver=receiver, vectors=vectors, volume=volume, energy=float(item.energy),
            forces=item.forces.cpu().double()[order], stress=item.stress.cpu().double().view(3, 3),
            w_energy=w * float(item.energy_weight), w_forces=w * float(item.forces_weight),
            w_stress=w * float(item.stress_weight), core_e=ce, core_f=cf, core_s=cs, partner=partner,
            node_order=_inverse(order)))
    return frames


def _core_frames(model, chunk, dev):
    """``_core`` of several frames in one float64 ZBL evaluation (the energy is a sum over frames, so the
    gradient of the total is every frame's own)."""
    import copy
    zbl = copy.deepcopy(model.pair_repulsion_fn).double().to(dev)
    S = len(model.atomic_numbers)
    zs = model.atomic_numbers.to(dev)
    n = [c[2].numel() for c in chunk]
    m_e = [c[5].numel() for c in chunk]
    node_off = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
    F = len(chunk)
    species = torch.cat([c[2] for c in chunk]).to(dev)
    snd = torch.cat([c[5] + int(node_off[i]) for i, c in enumerate(chunk)]).to(dev)
    rcv = torch.cat([c[6] + int(node_off[i]) for i, c in enumerate(chunk)]).to(dev)
    vec0 = torch.cat([c[7] for c in chunk]).to(dev).double()
    node_frame = torch.repeat_interleave(torch.arange(F, device=dev), torch.tensor(n, device=dev))
    edge_frame = torch.repeat_interleave(torch.arange(F, device=dev), torch.tensor(m_e, device=dev))
    vol = torch.tensor([c[8] for c in chunk], dtype=torch.float64, device=dev)
    scale = float(model.scale_shift.scale.double().item())
    shift = float(model.scale_shift.shift.double().item())
    vec = vec0.clone().requires_grad_(True)
    r = torch.linalg.norm(vec, dim=-1, keepdim=True)
    node_attrs = torch.nn.functional.one_hot(species, S).double()
    e_nodes = zbl(r, node_attrs, torch.stack([snd, rcv]), zs).reshape(-1)
    e_frame = scale * torch.zeros(F, dtype=torch.float64, device=dev).index_add(0, node_frame, e_nodes)
    (g,) = torch.autograd.grad(e_frame.sum(), vec)
    dpos = torch.zeros(int(node_off[-1]), 3, dtype=torch.float64, device=dev)
    dpos.index_add_(0, rcv, g).index_add_(0, snd, -g)
    m = torch.zeros(F, 3, 3, dtype=torch.float64, device=dev).index_add(
        0, edge_frame, vec0[:, :, None] * g[:, None, :])
    stress = 0.5 * (m + m.transpose(1, 2)) / vol[:, None, None]
    ae = model.atomic_energies_fn.atomic_energies.double().to(dev).reshape(-1)
    e0 = torch.zeros(F, dtype=torch.float64, device=dev).index_add(0, node_frame, ae[species])
    tot = (e_frame.detach() + e0 + shift * torch.tensor(n, dtype=torch.float64, device=dev)).cpu()
    dpos = dpos.cpu()
    stress = stress.cpu()
    return [(float(tot[i]), -dpos[node_off[i]:node_off[i + 1]], stress[i]) for i in range(F)]


def _pair_frames(chunk, dev):
    """``reverse_edges`` of several frames: one lexicographic order per side over (frame, ...) by successive
    stable sorts on the device; a frame with an unmatched or self-paired edge gets None."""
    F = len(chunk)
    m_e = [c[5].numel() for c in chunk]
    edge_off = np.concatenate([[0], np.cumsum(m_e)]).astype(np.int64)
    E = int(edge_off[-1])
    if E == 0:
        return [np.zeros(0, dtype=np.int64) for _ in chunk]
    snd = torch.cat([c[5] for c in chunk]).to(dev)
    rcv = torch.cat([c[6] for c in chunk]).to(dev)
    vec = torch.cat([c[7] for c in chunk]).to(dev).double()
    frame = torch.repeat_interleave(torch.arange(F, device=dev), torch.tensor(m_e, device=dev))
    neg = -vec

    def lexperm(keys):
        idx = torch.arange(E, device=dev)
        for k in reversed(keys):               # least significant first; stable sorts keep the earlier order
            idx = idx[torch.sort(k[idx], stable=True).indices]
        return idx

    oa = lexperm([frame, snd, rcv, vec[:, 0], vec[:, 1], vec[:, 2]])
    ob = lexperm([frame, rcv, snd, neg[:, 0], neg[:, 1], neg[:, 2]])
    ok = ((frame[oa] == frame[ob]) & (snd[oa] == rcv[ob]) & (rcv[oa] == snd[ob])
          & (vec[oa] == neg[ob]).all(dim=1))
    partner = torch.empty(E, dtype=torch.int64, device=dev)
    partner[ob] = oa
    bad = torch.zeros(F, dtype=torch.bool, device=dev)
    bad[frame[oa][~ok]] = True
    bad[frame[partner == torch.arange(E, device=dev)]] = True
    local = (partner - torch.as_tensor(edge_off[:-1], device=dev)[frame]).cpu().numpy()
    bad = bad.cpu().numpy()
    return [None if bad[i] else local[edge_off[i]:edge_off[i + 1]].copy() for i in range(F)]


def _convert_serial(dataset: Sequence, model) -> List[Frame]:
    import copy
    zbl = copy.deepcopy(model.pair_repulsion_fn).double().cpu()
    frames = []
    for item in dataset:
        species0 = item.node_attrs.argmax(dim=1).cpu()
        order = torch.sort(species0, stable=True).indices
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel())
        species = species0[order]
        S = item.node_attrs.shape[1]
        counts = [int((species == s).sum()) for s in range(S)]
        starts = [0] + list(np.cumsum(counts)[:-1].astype(int))
        sender = inv[item.edge_index[0].cpu()]
        receiver = inv[item.edge_index[1].cpu()]
        pos = item.positions.cpu().double()
        shifts = item.shifts.cpu().double()
        vectors = pos[item.edge_index[1]] - pos[item.edge_index[0]] + shifts
        e_order = torch.sort(receiver, stable=True).indices
        sender, receiver, vectors = sender[e_order], receiver[e_order], vectors[e_order]
        cell = item.cell.cpu().double().view(3, 3)
        volume = float(torch.linalg.det(cell).abs())
        core_e, core_f, core_s = _core(model, species, sender, receiver, vectors, volume, zbl)
        partner = reverse_edges(sender.numpy(), receiver.numpy(), vectors.numpy())
        w = float(item.weight)
        frames.append(Frame(
            n=species.numel(), counts=counts, starts=starts, species=species, sender=sender,
            receiver=receiver, vectors=vectors, volume=volume, energy=float(item.energy),
            forces=item.forces.cpu().double()[order], stress=item.stress.cpu().double().view(3, 3),
            w_energy=w * float(item.energy_weight), w_forces=w * float(item.forces_weight),
            w_stress=w * float(item.stress_weight), core_e=core_e, core_f=core_f, core_s=core_s,
            partner=partner, node_order=inv))
    return frames


def reverse_edges(snd: np.ndarray, rcv: np.ndarray, vec: np.ndarray) -> Optional[np.ndarray]:
    """partner[e] = the edge (rcv, snd, -vec) for every edge e, or None if some edge has none.
    The reverse vector is the exact negation (IEEE subtraction and addition are antisymmetric)."""
    E = snd.shape[0]
    if E == 0:
        return np.zeros(0, dtype=np.int64)
    va = vec + 0.0
    vb = -vec + 0.0                       # +0.0 turns -0.0 into 0.0
    oa = np.lexsort((va[:, 2], va[:, 1], va[:, 0], rcv, snd))
    ob = np.lexsort((vb[:, 2], vb[:, 1], vb[:, 0], snd, rcv))
    ka = np.stack([snd[oa], rcv[oa]], 1)
    kb = np.stack([rcv[ob], snd[ob]], 1)
    if not (np.array_equal(ka, kb) and np.array_equal(va[oa], vb[ob])):
        return None
    partner = np.empty(E, dtype=np.int64)
    partner[ob] = oa
    if np.any(partner == np.arange(E)):
        return None
    return partner


class DeviceFrames:
    """All frames resident on the device as concatenated arrays with per-frame offsets."""

    def __init__(self, frames: List[Frame], dtype, device):
        self.frames = frames
        self.dtype, self.device = dtype, device
        self.S = len(frames[0].counts)
        cat = lambda xs, dt=dtype: torch.cat(xs).to(device=device, dtype=dt)
        self.node_off = np.concatenate([[0], np.cumsum([f.n for f in frames])]).astype(np.int64)
        self.edge_off = np.concatenate([[0], np.cumsum([f.sender.numel() for f in frames])]).astype(np.int64)
        self.species = cat([f.species for f in frames], torch.int64)
        self.sender = cat([f.sender for f in frames], torch.int64)
        self.receiver = cat([f.receiver for f in frames], torch.int64)
        self.vectors = cat([f.vectors for f in frames])
        self.forces = cat([f.forces for f in frames])
        self.core_f = cat([f.core_f for f in frames])
        self.stress = cat([f.stress[None] for f in frames])
        self.core_s = cat([f.core_s[None] for f in frames])
        scalar = lambda key: torch.tensor([getattr(f, key) for f in frames], dtype=torch.float64)
        self.energy = scalar('energy').to(device=device, dtype=dtype)
        self.core_e = scalar('core_e').to(device=device, dtype=dtype)
        self.volume = scalar('volume').to(device=device, dtype=dtype)
        self.w_energy = scalar('w_energy').to(device=device, dtype=dtype)
        self.w_forces = scalar('w_forces').to(device=device, dtype=dtype)
        self.w_stress = scalar('w_stress').to(device=device, dtype=dtype)
        self.n_atoms = torch.tensor([f.n for f in frames], dtype=dtype, device=device)
        self.counts = np.array([f.counts for f in frames], dtype=np.int64)       # [F, S]
        self.starts = np.array([f.starts for f in frames], dtype=np.int64)

    def batch(self, ids: Sequence[int], csr: bool = False) -> Batch:
        ids = [int(i) for i in ids]
        dev = self.device
        G = len(ids)
        counts = self.counts[ids]                     # [G, S]
        totals = counts.sum(0)
        sp_off = np.concatenate([[0], np.cumsum(totals)])
        base = sp_off[None, :-1] + np.cumsum(counts, 0) - counts           # [G, S]
        delta = base - self.starts[ids]                                     # new = local + delta[species]
        node_parts, edge_parts, nid, eid = [], [], [], []
        for g, i in enumerate(ids):
            node_parts.append((self.node_off[i], self.node_off[i + 1]))
            edge_parts.append((self.edge_off[i], self.edge_off[i + 1]))
        node_idx = torch.from_numpy(np.concatenate([np.arange(a, b) for a, b in node_parts])).to(dev)
        edge_idx = torch.from_numpy(np.concatenate([np.arange(a, b) for a, b in edge_parts])).to(dev)
        n_per = np.array([b - a for a, b in node_parts])
        e_per = np.array([b - a for a, b in edge_parts])
        node_graph_local = torch.from_numpy(np.repeat(np.arange(G), n_per)).to(dev)
        edge_graph_local = torch.from_numpy(np.repeat(np.arange(G), e_per)).to(dev)
        delta_t = torch.from_numpy(delta).to(dev)                           # [G, S]
        species_local = self.species[node_idx]
        local = node_idx - torch.from_numpy(np.repeat(self.node_off[ids], n_per)).to(dev)
        new_pos = local + delta_t[node_graph_local, species_local]
        N = int(n_per.sum())
        inv = torch.empty(N, dtype=torch.int64, device=dev)
        inv[new_pos] = torch.arange(N, device=dev)                          # batch slot -> concat row
        node_rows = node_idx[inv]
        species = self.species[node_rows]
        node_graph = node_graph_local[inv]
        before = np.concatenate([[0], np.cumsum(n_per)[:-1]])
        e_before = torch.from_numpy(np.repeat(before, e_per)).to(dev)
        # local id -> position in this batch's concatenation -> new (species-sorted) position
        snd = new_pos[self.sender[edge_idx] + e_before]
        rcv = new_pos[self.receiver[edge_idx] + e_before]
        order = torch.sort(rcv, stable=True).indices
        edge_rows = edge_idx[order]
        snd, rcv = snd[order], rcv[order]
        edge_graph = edge_graph_local[order]
        ids_t = torch.tensor(ids, device=dev)
        b = Batch(
            vectors=self.vectors[edge_rows], sender=snd, receiver=rcv, species=species,
            species_bounds=tuple(int(x) for x in sp_off), node_graph=node_graph, edge_graph=edge_graph,
            num_graphs=G, n_atoms=self.n_atoms[ids_t], volume=self.volume[ids_t], e_ref=self.energy[ids_t],
            f_ref=self.forces[node_rows], s_ref=self.stress[ids_t], w_energy=self.w_energy[ids_t],
            w_forces=self.w_forces[ids_t][node_graph], w_stress=self.w_stress[ids_t],
            core_e=self.core_e[ids_t], core_f=self.core_f[node_rows], core_s=self.core_s[ids_t], frame_ids=ids)
        if csr:
            add_csr(b)
        return b


def add_csr(b: Batch) -> Batch:
    N = b.species.shape[0]
    ar = torch.arange(N + 1, device=b.receiver.device)
    b.receiver_ptr = torch.searchsorted(b.receiver, ar).to(torch.int32)
    perm = torch.sort(b.sender, stable=True).indices
    b.sender_perm = perm.to(torch.int32)
    b.sender_ptr = torch.searchsorted(b.sender[perm], ar).to(torch.int32)
    return b
