"""Muon on MACE's matrix blocks, Adam on the rest: a recipe optimizer prototype (``optimizer.kind = "muon"``).

Muon (Jordan et al. 2024; the Moonlight scaling, Liu et al. 2025) replaces Adam's update of a weight matrix by
the orthogonalised momentum: ``B = mu B + G``, ``O = NS5(G + mu B)`` (Nesterov), ``W -= lr * s * O`` with
``s = scale * sqrt(max(m, n))`` so that Adam learning rates carry over (``scale`` 0.2 matches AdamW's update RMS).

Matrix blocks of the student: the radial MLP layers, every per-l block of the e3nn linears (``linear_up``,
``linear``, ``products.*.linear``), ``skip_tp`` as one 64 x 64 block per species, and the symmetric contraction's
``weights_max`` as one 23 x 64 block per species. The species embedding, the readouts and the small contraction
weights (4 x 64, 1 x 64) keep Adam (with the recipe's betas / eps / AMSGrad).

The library runs it inside its fused step (``lst_set_muon``, ABI 5; :func:`library_blocks` gives the block
table). :class:`MuonAdam` / :func:`run_epoch` are the reference: the library's gradient pass step by step
(``lst_debug_step`` without its update, the raw gradient copied out) and the update in torch
(recipe ``optimizer.impl = "torch"``)."""
from __future__ import annotations

import ctypes as ct
import math
import os
from typing import Dict, List

import numpy as np
import torch

NS_COEF = (3.4445, -4.7750, 2.0315)
NS_POLISH = (2.0, -1.5, 0.5)          # DPA4's polishing steps: singular values -> 1 (f(1) = 1, f'(1) = 0)
MAGMA_TAU, MAGMA_RHO, MAGMA_MIN = 2.0, 0.9, 0.1


def newton_schulz(G: torch.Tensor, steps: int = 5, polish: int = 0) -> torch.Tensor:
    """Approximate orthogonalisation U V^T of each matrix of G [b, m, n] (quintic Newton-Schulz, Muon's;
    ``polish`` further steps with DPA4's (2, -1.5, 0.5))."""
    X = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    tr = X.shape[-2] > X.shape[-1]
    if tr:
        X = X.mT
    for k in range(steps + polish):
        a, b, c = NS_COEF if k < steps else NS_POLISH
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if tr else X


def magma_raw(M: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """DPA4's Magma-lite score per matrix of [b, m, n]: the cosine of momentum and gradient through a
    temperature-2 sigmoid, stretched to [0, 1]."""
    chi = (M * G).sum((-2, -1)) / ((M.norm(dim=(-2, -1)) * G.norm(dim=(-2, -1))).clamp_min(1e-12))
    s = lambda x: 1.0 / (1.0 + math.exp(-x))  # noqa: E731
    lo, hi = s(-1.0 / MAGMA_TAU), s(1.0 / MAGMA_TAU)
    return ((torch.sigmoid(chi.clamp(-1, 1) / MAGMA_TAU) - lo) / (hi - lo)).clamp(0.0, 1.0)


def matrix_views(model, name: str, t: torch.Tensor):
    """[b, m, n] views of a tensor laid out as parameter ``name`` (its matrix blocks), or None (Adam)."""
    if name.startswith(('node_embedding', 'readouts')) or '.weights.' in name:
        return None
    if '.conv_tp_weights.layer' in name:
        return [t.unsqueeze(0)]
    if name.endswith('weights_max'):
        return [t]
    mod = model.get_submodule(name.rsplit('.', 1)[0])
    if not hasattr(mod, 'weight_views'):
        return None
    out = []
    for v in mod.weight_views(weight=t):
        if v.dim() == 2:
            out.append(v.unsqueeze(0))
        elif v.dim() == 3:                       # skip_tp: [mul_in, species, mul_out] -> per species
            out.append(v.permute(1, 0, 2))
        else:
            return None
    return out


def library_blocks(model, params) -> list:
    """The library's Muon block table (``LstMuonBlock``) of the matrix views of ``params`` (the trainable tensors
    in the library's order): (param, batch, rows, cols, offset, batch_stride, row_stride, col_stride)."""
    names = {id(p): n for n, p in model.named_parameters()}
    out = []
    for i, p in enumerate(params):
        views = matrix_views(model, names[id(p)], p.data)
        if views is None:
            continue
        for v in views:
            b, r, c = v.shape
            sb, sr, sc = v.stride()
            out.append((i, b, r, c, v.storage_offset() - p.data.storage_offset(), sb if b > 1 else r * c, sr, sc))
    return out


class NodeEnergy:
    """Gradient of  weight * mean_i (e_i - t_i)^2  over a batch's atoms (mace's forward, energies only).
    e0 {Z: eV}: the run's fitted reference energies, subtracted from each atom's absolute target (a centred
    student trains on E - sum e0)."""

    def __init__(self, model, dataset, xyz, weight, e0=None, recenter=False):
        from ase.io import read
        frames = read(xyz, ':')
        if len(frames) != len(dataset):
            raise ValueError(f'{xyz}: {len(frames)} frames, the train set has {len(dataset)}')
        dev = next(model.parameters()).device
        dt = next(model.parameters()).dtype
        self.t = []
        for k, (a, d) in enumerate(zip(frames, dataset)):
            pos = torch.as_tensor(a.positions, dtype=d.positions.dtype)
            if pos.shape != d.positions.shape or float((pos - d.positions.cpu()).abs().max()) > 1e-6:
                raise ValueError(f'{xyz}: frame {k} is not the train set\'s frame {k}')
            t = a.arrays['REF_node_energy'].astype('float64')
            if e0:
                t = t - np.array([float(e0[int(z)]) for z in a.numbers])
            ref = getattr(d, 'energy', None)
            if ref is not None and abs(float(t.sum()) - float(ref)) > 1e-3 * len(t):
                raise ValueError(f'{xyz}: frame {k}: targets sum to {t.sum():.4f} eV, the train energy is '
                                 f'{float(ref):.4f} (wrong e0?)')
            self.t.append(t)
        if recenter:
            comp = {tuple(np.unique(a.numbers, return_counts=True)[1]) for a in frames}
            if len(comp) != 1 or len({tuple(np.unique(a.numbers)) for a in frames}) != 1:
                raise ValueError('recenter needs one composition in every frame')
            zs = frames[0].numbers
            mu = {z: np.mean(np.concatenate([t[a.numbers == z] for a, t in zip(frames, self.t)])) for z in set(zs)}
            k = np.mean([mu[z] for z in zs])
            self.t = [t - np.array([mu[z] - k for z in a.numbers]) for a, t in zip(frames, self.t)]
        self.t = [torch.as_tensor(t, dtype=dt, device=dev) for t in self.t]
        self.model, self.data, self.w = model, dataset, weight
        self.last = 0.0

    def grads(self, idx, params):
        from mace.tools import torch_geometric
        b = torch_geometric.Batch.from_data_list([self.data[j] for j in idx]).to(self.t[0].device)
        with torch.enable_grad():
            out = self.model(b.to_dict(), training=True, compute_force=False, compute_virials=False,
                             compute_stress=False)
            e = out['node_energy']
            t = torch.cat([self.t[j] for j in idx])
            loss = self.w * (e - t).pow(2).mean()
            g = torch.autograd.grad(loss, params, allow_unused=True)
        self.last = float(loss) / self.w
        return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(g, params)])


class MuonAdam:
    def __init__(self, model, named_params, momentum=0.95, nesterov=True, scale=0.2, muon_lr_mult=1.0,
                 betas=(0.9, 0.999), eps=1e-8, amsgrad=True, ns_steps=5, ns_polish=0, magma=False, muon_wd=0.0,
                 matrix='ns', adam_rms=False, soap_betas=(0.95, 0.95), soap_every=10, soap_eps=1e-8, soap_dtype='float64', soap_eig='eigh', soap_max_dim=0,
                 normuon_beta=0.95, cautious=''):
        self.model = model
        self.node_energy = None
        self.norms, self.zclip, self.zstat = [], 0.0, None    # pre-clip gradient norms; ZClip's z and state
        if matrix not in ('ns', 'rms', 'soap', 'normuon'):
            raise ValueError(f'matrix must be ns, rms, soap or normuon, not {matrix!r}')
        if cautious not in ('', 'adam', 'all'):
            raise ValueError(f'cautious must be "", adam or all, not {cautious!r}')
        self.normuon_beta, self.cautious, self.nm = float(normuon_beta), cautious, {}
        self.matrix, self.adam_rms = matrix, bool(adam_rms)
        self.soap_betas, self.soap_every, self.soap_eps = tuple(float(b) for b in soap_betas), int(soap_every), \
            float(soap_eps)
        self.soap_dtype = {'float64': torch.float64, 'float32': torch.float32}[soap_dtype]
        if soap_eig not in ('eigh', 'jacobi'):
            raise ValueError(f'soap_eig must be eigh or jacobi, not {soap_eig!r}')
        self.soap_eig = soap_eig
        self.soap_max_dim = int(soap_max_dim)
        self.soap = {}                        # per (tensor, view): L, R, QL, QR, M, V, t
        self.ns_steps, self.ns_polish, self.magma, self.muon_wd = int(ns_steps), int(ns_polish), bool(magma), \
            float(muon_wd)
        self.score = {}                       # Magma-lite EMA per (tensor, view): [b], starts at 0.5 (deepmd-kit)
        self.momentum, self.nesterov, self.scale, self.mult = float(momentum), bool(nesterov), float(scale), \
            float(muon_lr_mult)
        self.muon, adam = [], []
        for n, p in named_params:
            (self.muon if matrix_views(model, n, p.data) is not None else adam).append((n, p))
        self.buf = {n: torch.zeros_like(p) for n, p in self.muon}
        self.adam_params = [p for _, p in adam]
        self.adam = torch.optim.Adam(self.adam_params, lr=1e-3, betas=tuple(betas), eps=float(eps),
                                     amsgrad=bool(amsgrad))

    def describe(self) -> str:
        return (f'{self.matrix} on {len(self.muon)} tensors ({sum(p.numel() for _, p in self.muon)} weights), Adam on '
                f'{len(self.adam_params)} ({sum(p.numel() for p in self.adam_params)})')

    @torch.no_grad()
    def step(self, grads: Dict[int, torch.Tensor], lr: float):
        for g in self.adam.param_groups:
            g['lr'] = lr
        for p in self.adam_params:
            p.grad = grads[id(p)]
        old = [p.data.clone() for p in self.adam_params] if (self.adam_rms or self.cautious) else None
        self.adam.step()
        for k, p in enumerate(self.adam_params):
            p.grad = None
            if old is not None:                       # cautious mask; block normalisation (RMS = scale * lr)
                d = p.data - old[k]
                if self.cautious:                     # keep components whose step opposes the gradient
                    msk = (d * grads[id(p)] < 0).to(d.dtype)
                    d = d * msk * (msk.numel() / (float(msk.sum()) + 1.0))
                    p.data.copy_(old[k] + d)
                rms = float(d.pow(2).mean().sqrt())
                if self.adam_rms and rms > 0:
                    p.data.copy_(old[k] + d * (lr * self.scale / rms))
        if self.matrix == 'soap':
            for n, p in self.muon:
                for j, (P_, G_) in enumerate(zip(matrix_views(self.model, n, p.data),
                                                 matrix_views(self.model, n, grads[id(p)]))):
                    P_.add_(self._soap(n, j, G_).to(P_.dtype), alpha=-lr * self.mult)
            return
        for n, p in self.muon:
            g = grads[id(p)]
            B = self.buf[n]
            B.mul_(self.momentum).add_(g)
            D = g.add(B, alpha=self.momentum) if self.nesterov else B
            if self.muon_wd > 0:                      # decoupled, before the update (DPA4: 1 - lr wd)
                p.data.mul_(1.0 - lr * self.mult * self.muon_wd)
            views = zip(matrix_views(self.model, n, p.data), matrix_views(self.model, n, D))
            if self.magma:
                views = zip(matrix_views(self.model, n, p.data), matrix_views(self.model, n, D),
                            matrix_views(self.model, n, B), matrix_views(self.model, n, g))
            for j, v in enumerate(views):
                P_, D_ = v[0], v[1]
                m, k = D_.shape[-2:]
                if self.matrix == 'normuon':              # NorMuon (2510.05491): per-row second moment after NS
                    O = newton_schulz(D_, self.ns_steps, self.ns_polish)
                    r2 = O.pow(2).mean(-1, keepdim=True)
                    vr = self.nm.get((n, j))
                    vr = r2.clone() if vr is None else vr.mul_(self.normuon_beta).add_(r2, alpha=1 - self.normuon_beta)
                    self.nm[(n, j)] = vr
                    Oh = O / (vr.sqrt() + 1e-12)
                    O = Oh * (O.norm(dim=(-2, -1), keepdim=True) / (Oh.norm(dim=(-2, -1), keepdim=True) + 1e-12))
                elif self.matrix == 'rms':                # Muon's update RMS, no orthogonalisation
                    O = D_ / (D_.pow(2).mean((-2, -1), keepdim=True).sqrt() + 1e-12) / math.sqrt(max(m, k))
                else:
                    O = newton_schulz(D_, self.ns_steps, self.ns_polish)
                if self.cautious == 'all':                # the step is -O: keep components with O * g > 0
                    gv = matrix_views(self.model, n, g)[j]
                    msk = (O * gv > 0).to(O.dtype)
                    O = O * msk * (msk[0].numel() / (msk.sum((-2, -1), keepdim=True) + 1.0))
                if self.magma:
                    raw = magma_raw(v[2], v[3])
                    u = self.score.get((n, j))
                    u = torch.full_like(raw, 0.5) if u is None else u
                    u = MAGMA_RHO * u + (1.0 - MAGMA_RHO) * raw
                    self.score[(n, j)] = u
                    O = O * (MAGMA_MIN + (1.0 - MAGMA_MIN) * u).view(-1, 1, 1)
                P_.add_(O, alpha=-lr * self.mult * self.scale * math.sqrt(max(m, k)))

    def _soap(self, n, j, G):
        """SOAP's update of one block stack G [b, m, n] (Vyas et al. 2024): Adam's moments with the second one in
        the eigenbasis of Shampoo's L = E[G G^T], R = E[G^T G] (bases refreshed every ``soap_every`` steps;
        the first moment kept in the original basis and rotated each step)."""
        b1, b2 = self.soap_betas
        s = self.soap.get((n, j))
        G = G.to(self.soap_dtype)
        if s is None:
            b, m, k = G.shape
            s = dict(L=torch.zeros(b, m, m, dtype=G.dtype, device=G.device),
                     R=torch.zeros(b, k, k, dtype=G.dtype, device=G.device),
                     M=torch.zeros_like(G), V=torch.zeros_like(G), t=0)
            self.soap[(n, j)] = s
        s['t'] += 1
        t = s['t']
        s['L'].mul_(b2).add_(G @ G.mT, alpha=1 - b2)
        s['R'].mul_(b2).add_(G.mT @ G, alpha=1 - b2)
        md = self.soap_max_dim
        if md and (t - 1) % self.soap_every == 0:
            eye = lambda q: torch.eye(q.shape[-1], dtype=q.dtype, device=q.device).expand_as(q)  # noqa: E731
            QL0 = eye(s['L']) if s['L'].shape[-1] > md else None
            QR0 = eye(s['R']) if s['R'].shape[-1] > md else None
        else:
            QL0 = QR0 = None
        if (t - 1) % self.soap_every == 0:
            eye = lambda q: torch.eye(q.shape[-1], dtype=q.dtype, device=q.device)  # noqa: E731
            if self.soap_eig == 'jacobi':
                QL = QL0 if QL0 is not None else _jacobi(s['L'], s.get('QL'), tol=_SOAP_TOL)
                QR = QR0 if QR0 is not None else _jacobi(s['R'], s.get('QR'), tol=_SOAP_TOL)
            else:
                QL = QL0 if QL0 is not None else torch.linalg.eigh(s['L'] + 1e-12 * eye(s['L']))[1]
                QR = QR0 if QR0 is not None else torch.linalg.eigh(s['R'] + 1e-12 * eye(s['R']))[1]
            s['QL'], s['QR'] = (QL if QL0 is None else QL0), (QR if QR0 is None else QR0)  # V kept as it is
        QL, QR = s['QL'], s['QR']
        s['M'].mul_(b1).add_(G, alpha=1 - b1)
        Gr = QL.mT @ G @ QR
        s['V'].mul_(b2).addcmul_(Gr, Gr, value=1 - b2)
        Mr = QL.mT @ s['M'] @ QR
        Ur = (Mr / (1 - b1 ** t)) / ((s['V'] / (1 - b2 ** t)).sqrt() + self.soap_eps)
        return QL @ Ur @ QR.mT


_SOAP_TOL = float(os.environ.get('LOUP_SOAP_TOL', '1e-7'))   # Jacobi stop (parity checks: 0 = all sweeps)


def _jacobi(A, Q=None, sweeps=30, tol=1e-7):
    """Eigenvectors of the symmetric stack A [b, d, d] by cyclic Jacobi (round-robin pairs, all pairs of a round
    at once), started from the basis Q (identity when None): columns keep their identity across refreshes."""
    b, d, _ = A.shape
    Q = torch.eye(d, dtype=A.dtype, device=A.device).expand(b, d, d).clone() if Q is None else Q.clone()
    if d < 2:
        return Q
    B = Q.mT @ A @ Q
    n = d + (d % 2)
    idx = list(range(n))
    for _ in range(sweeps):
        off = (B - torch.diag_embed(torch.diagonal(B, dim1=-2, dim2=-1))).pow(2).sum((-2, -1))
        if bool((off <= (tol ** 2) * B.pow(2).sum((-2, -1))).all()):
            break
        for _r in range(n - 1):
            pairs = [(idx[i], idx[n - 1 - i]) for i in range(n // 2)]
            pairs = [(min(p, q), max(p, q)) for p, q in pairs if p < d and q < d]
            p = torch.tensor([x for x, _ in pairs], device=A.device)
            q = torch.tensor([y for _, y in pairs], device=A.device)
            app, aqq, apq = B[:, p, p], B[:, q, q], B[:, p, q]
            theta = 0.5 * torch.atan2(2 * apq, aqq - app)
            c, s_ = torch.cos(theta)[..., None], torch.sin(theta)[..., None]
            Bp, Bq = B[:, p, :], B[:, q, :]                      # rows: B <- J^T B
            B[:, p, :], B[:, q, :] = c * Bp - s_ * Bq, s_ * Bp + c * Bq
            c, s_ = c.mT, s_.mT
            Bp, Bq = B[:, :, p], B[:, :, q]                      # columns: B <- B J, Q <- Q J
            B[:, :, p], B[:, :, q] = c * Bp - s_ * Bq, s_ * Bp + c * Bq
            Qp, Qq = Q[:, :, p], Q[:, :, q]
            Q[:, :, p], Q[:, :, q] = c * Qp - s_ * Qq, s_ * Qp + c * Qq
            idx = [idx[0]] + [idx[-1]] + idx[1:-1]
    return Q


def run_epoch(tr, mo: MuonAdam, batches: List[List[int]], lr_steps, w_steps, ema, clip):
    """One epoch: per step the library's loss and raw gradient (no library update), global-norm clipping,
    the Muon / Adam update and the EMA. Returns the steps' losses."""
    from .cuda import _dbl
    n = len(batches)
    bt = tr.stack(batches)
    s = tr._stream()                                    # pylint: disable=protected-access
    G = len(lr_steps[0])
    tr._check(tr.lib.lst_begin_epoch_steps(tr.h, ct.byref(bt), _dbl([0.0] * (n * G)),
                                           _dbl([float(x) for r in w_steps for x in r]), None, s))
    params = tr.params
    for i in range(n):
        tr._check(tr.lib.lst_debug_step(tr.h, i, 0, s))
        flat = tr.stage('graw')
        if mo.node_energy is not None:
            flat = flat + mo.node_energy.grads(batches[i], params).to(flat.dtype)
        norm = float(flat.double().norm())
        mo.norms.append(norm)
        if mo.zclip:                                    # ZClip (2504.02507): mean + z sd of the norm's EMA
            z, a = mo.zclip, 0.97
            if mo.zstat is None:
                mo.zstat = [norm, 0.0]
            mu, var = mo.zstat
            thr = mu + z * var ** 0.5 if var > 0 else max(mu, norm)
            if norm > thr:
                flat.mul_(thr / (norm + 1e-6))
            used = min(norm, thr)
            mo.zstat = [a * mu + (1 - a) * used, a * var + (1 - a) * (used - mu) ** 2]
        elif clip is not None and clip > 0:
            if norm > clip:
                flat.mul_(clip / (norm + 1e-6))
        grads, o = {}, 0
        for p in params:
            grads[id(p)] = flat[o:o + p.numel()].view_as(p)
            o += p.numel()
        mo.step(grads, float(lr_steps[i][0]))
        if ema is not None:
            ema.update()
    out = (ct.c_double * n)()
    tr._check(tr.lib.lst_epoch_losses(tr.h, out, s))
    tr.steps_done += n
    return list(out)
