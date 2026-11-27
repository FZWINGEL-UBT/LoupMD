"""ctypes binding of the Loup-S training library (``libloupstrain.so``, src/train, include/loupstrain.h).

:class:`CudaTrainer` has the interface of :class:`loupmd.fasttrain.engine.GraphTrainer` (``run_epoch``) and runs
each optimizer step as one replay of the library's CUDA graph: batch gather, energy / forces / stress, loss,
parameter gradient, clipping, Adam / AMSGrad and EMA, all on mace's own parameter, optimizer-state and EMA
tensors (updated in place through their device pointers).

The library is looked for at ``$LOUP_STRAIN_LIB``, then ``build/<arch>/libloupstrain.so`` of the checkout
(``make train-lib``).
"""
from __future__ import annotations

import ctypes as ct
import math
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .engine import DUMMY_ROW

ABI = 6
NP = 26
ROOT = Path(__file__).resolve().parents[2]

# mace named_parameters() order (trainable), with each view's shape (S species, C channels, NB, H, hid, K*)
PARAM_NAMES = [
    'node_embedding.linear.weight',
    'interactions.0.linear_up.weight', 'interactions.0.conv_tp_weights.layer0.weight',
    'interactions.0.conv_tp_weights.layer1.weight', 'interactions.0.conv_tp_weights.layer2.weight',
    'interactions.0.conv_tp_weights.layer3.weight', 'interactions.0.linear.weight', 'interactions.0.skip_tp.weight',
    'interactions.1.linear_up.weight', 'interactions.1.conv_tp_weights.layer0.weight',
    'interactions.1.conv_tp_weights.layer1.weight', 'interactions.1.conv_tp_weights.layer2.weight',
    'interactions.1.conv_tp_weights.layer3.weight', 'interactions.1.linear.weight', 'interactions.1.skip_tp.weight',
    'products.0.symmetric_contractions.contractions.0.weights_max',
    'products.0.symmetric_contractions.contractions.0.weights.0',
    'products.0.symmetric_contractions.contractions.0.weights.1', 'products.0.linear.weight',
    'products.1.symmetric_contractions.contractions.0.weights_max',
    'products.1.symmetric_contractions.contractions.0.weights.0',
    'products.1.symmetric_contractions.contractions.0.weights.1', 'products.1.linear.weight',
    'readouts.0.linear.weight', 'readouts.1.linear_1.weight', 'readouts.1.linear_2.weight',
]

P_ = ct.c_void_p


class ModelDesc(ct.Structure):
    _fields_ = [('dtype', ct.c_int), ('S', ct.c_int), ('C', ct.c_int), ('H', ct.c_int), ('NB', ct.c_int),
                ('hid', ct.c_int), ('K', ct.c_int), ('K3', ct.c_int), ('K2', ct.c_int), ('p_cut', ct.c_int),
                ('r_max', ct.c_double), ('prefactor', ct.c_double), ('act', ct.c_double), ('scale', ct.c_double),
                ('avg0', ct.c_double), ('avg1', ct.c_double), ('bessel_w', ct.POINTER(ct.c_double))]


class Shape(ct.Structure):
    _fields_ = [(k, ct.c_int) for k in ('N', 'E', 'P', 'G1', 'B')]


FRAME_KEYS = ('vectors', 'species', 'f_ref', 'core_f', 'e_ref', 'core_e', 'volume', 'n_atoms', 'w_e', 'w_f', 'w_s',
              's_ref', 'core_s')


TOPO_KEYS = ('node_off', 'edge_off', 'pair_off', 'sender', 'receiver', 'receiver_ptr', 'pidx', 'pe0', 'pe1')


class Frames(ct.Structure):
    _fields_ = [(k, P_) for k in FRAME_KEYS] + [('n_frames', ct.c_int), ('dummy_row', ct.c_int)] + \
        [(k, P_) for k in TOPO_KEYS]


class Batches(ct.Structure):
    _fields_ = [('n', ct.c_int), ('frame_ids', ct.POINTER(ct.c_int32))]


class Optim(ct.Structure):
    _fields_ = [('exp_avg', P_ * NP), ('exp_avg_sq', P_ * NP), ('max_exp_avg_sq', P_ * NP), ('step', P_ * NP),
                ('ema', P_ * NP), ('group', ct.c_int * NP), ('n_groups', ct.c_int), ('beta1', ct.c_double),
                ('beta2', ct.c_double), ('eps', ct.c_double), ('weight_decay', ct.c_double * 16),
                ('max_grad_norm', ct.c_double), ('update_clip', ct.c_double)]


_LIB = None


class MuonBlock(ct.Structure):
    _fields_ = [('param', ct.c_int), ('batch', ct.c_int), ('rows', ct.c_int), ('cols', ct.c_int),
                ('offset', ct.c_long), ('batch_stride', ct.c_long), ('row_stride', ct.c_long),
                ('col_stride', ct.c_long)]


def load_library(path: Optional[str] = None):
    global _LIB
    if _LIB is not None and path is None:
        return _LIB
    if path is None:
        path = os.environ.get('LOUP_STRAIN_LIB')
    if path is None:
        major, minor = torch.cuda.get_device_capability()
        path = str(ROOT / 'build' / f'sm_{major}{minor}' / 'libloupstrain.so')
    lib = ct.CDLL(path)
    lib.lst_abi_version.restype = ct.c_int
    if lib.lst_abi_version() != ABI:
        raise RuntimeError(f'{path}: ABI {lib.lst_abi_version()}, binding {ABI}')
    lib.lst_last_error.restype = ct.c_char_p
    lib.lst_last_error.argtypes = [P_]
    lib.lst_create.restype = P_
    lib.lst_create.argtypes = [ct.POINTER(ModelDesc), ct.POINTER(Shape), ct.c_int]
    lib.lst_destroy.argtypes = [P_]
    lib.lst_set_params.argtypes = [P_, ct.POINTER(P_)]
    lib.lst_set_frames.argtypes = [P_, ct.POINTER(Frames)]
    if hasattr(lib, 'lst_set_node_order'):
        lib.lst_set_node_order.argtypes = [P_, P_]
    lib.lst_set_optim.argtypes = [P_, ct.POINTER(Optim)]
    lib.lst_set_muon.argtypes = [P_, ct.POINTER(MuonBlock), ct.c_int, ct.c_double, ct.c_int, ct.c_double]
    lib.lst_set_muon_ns.argtypes = [P_, ct.c_int, ct.c_int]
    if hasattr(lib, 'lst_set_muon_norm'):
        lib.lst_set_muon_norm.argtypes = [P_, ct.c_double]
    if hasattr(lib, 'lst_set_cautious'):
        lib.lst_set_cautious.argtypes = [P_, ct.c_int]
    if hasattr(lib, 'lst_set_muon_wd'):
        lib.lst_set_muon_wd.argtypes = [P_, ct.c_double]
    if hasattr(lib, 'lst_set_math'):
        lib.lst_set_math.argtypes = [P_, ct.c_int, ct.c_int]
    if hasattr(lib, 'lst_set_soap'):
        lib.lst_set_soap.argtypes = [P_, ct.c_double, ct.c_double, ct.c_double, ct.c_int, ct.c_int, ct.c_double]
    if hasattr(lib, 'lst_evaluate'):
        lib.lst_evaluate.argtypes = [P_, ct.POINTER(ct.c_int32), ct.c_int, P_, P_, P_, P_]
    lib.lst_begin_epoch.argtypes = [P_, ct.POINTER(Batches), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                                    ct.POINTER(ct.c_double), P_]
    lib.lst_begin_epoch_steps.argtypes = lib.lst_begin_epoch.argtypes
    lib.lst_run_steps.argtypes = [P_, ct.c_int, P_]
    lib.lst_epoch_losses.argtypes = [P_, ct.POINTER(ct.c_double), P_]
    lib.lst_debug_step.argtypes = [P_, ct.c_int, ct.c_int, P_]
    lib.lst_stage_size.restype = ct.c_long
    lib.lst_stage_size.argtypes = [P_, ct.c_char_p]
    lib.lst_stage_copy.argtypes = [P_, ct.c_char_p, P_, P_]
    if path is None or _LIB is None:
        _LIB = lib
    return lib


def _ptr(t: torch.Tensor):
    return ct.c_void_p(t.data_ptr())


def _dbl(xs):
    return (ct.c_double * len(xs))(*[float(x) for x in xs])


def _named_params(model, by_name: bool):
    if by_name:
        allp = dict(model.named_parameters())
        return [(n, allp[n]) for n in PARAM_NAMES if n in allp]
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


class CudaTrainer:
    """mace-torch optimizer steps for one model and training set in the C++/CUDA training library."""

    def __init__(self, model, dataset, batch_size: int, dtype=torch.float32, lib_path: Optional[str] = None,
                 by_name: bool = False):
        """``by_name``: the parameters named ``PARAM_NAMES`` whether or not they require gradients (evaluation of
        a loaded checkpoint); otherwise the trainable ones, which must be exactly those."""
        from .data import frames_from_dataset
        from .engine import PaddedFrames, FastStudent, DUMMY_ROW
        from .model import silu_norm_constant
        from .warm import join
        join()                              # the warm-up thread works on the legacy stream: done before captures
        t0 = time.time()
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = dtype
        self.lib = load_library(lib_path)
        frames = frames_from_dataset(dataset, model)
        self.data = d = PaddedFrames(frames, dtype, self.device, batch_size, float(model.r_max))
        if not d.paired:
            raise RuntimeError('the training library needs paired edges (every edge with its reverse)')
        fs = FastStudent(model)                       # model checks
        named = _named_params(model, by_name)
        if [n for n, _ in named] != PARAM_NAMES:
            raise RuntimeError(f'unexpected trainable parameters: {[n for n, _ in named]}')
        self.params = [p for _, p in named]
        for n, p in named:
            if p.dtype != dtype or not p.is_contiguous() or p.device != self.device:
                raise RuntimeError(f'{n}: {p.dtype} {p.device} contiguous={p.is_contiguous()}')
        sc = model.products[0].symmetric_contractions.contractions[0]
        K3, K2, K1 = sc.weights_max.shape[1], sc.weights[0].shape[1], sc.weights[1].shape[1]
        net = model.interactions[0].conv_tp_weights
        self._bw = (ct.c_double * fs.bw.numel())(*fs.bw.detach().double().cpu().tolist())
        self.desc = ModelDesc(dtype=0 if dtype == torch.float32 else 1, S=fs.S, C=fs.C,
                              H=int(net.layer0.weight.shape[1]), NB=int(fs.bw.numel()),
                              hid=int(model.readouts[1].linear_1.weight.numel() // fs.C), K=K3 + K2 + K1, K3=K3, K2=K2,
                              p_cut=fs.p, r_max=fs.r_max, prefactor=fs.prefactor, act=float(fs.act),
                              scale=float(model.scale_shift.scale), avg0=fs.avg[0], avg1=fs.avg[1], bessel_w=self._bw)
        self.shape = Shape(N=d.N_pad, E=d.E_pad, P=d.P_pad, G1=d.G1, B=d.B)
        h = self.lib.lst_create(ct.byref(self.desc), ct.byref(self.shape), self.device.index or 0)
        if not h:
            raise RuntimeError('lst_create: ' + self.lib.lst_last_error(None).decode())
        self.h = ct.c_void_p(h)
        self._check(self.lib.lst_set_params(self.h, (P_ * NP)(*[p.data_ptr() for p in self.params])))
        self._param_ptrs = tuple(p.data_ptr() for p in self.params)
        self._topo = self._topology()
        self._frames = Frames(**{k: getattr(d, k).data_ptr() for k in FRAME_KEYS}, n_frames=len(frames),
                              dummy_row=DUMMY_ROW, **{k: self._topo[k].data_ptr() for k in TOPO_KEYS})
        self._check(self.lib.lst_set_frames(self.h, ct.byref(self._frames)))
        self._node_order = None
        if (os.environ.get('LOUP_FT_NODE_ORDER', '1') != '0' and hasattr(self.lib, 'lst_set_node_order')
                and all(f.node_order is not None for f in frames)):
            # the force pass in each frame's own atom order (L2 reuse of a pair's radial rows)
            self._node_order = torch.from_numpy(np.concatenate([f.node_order.numpy() for f in frames])
                                                .astype(np.int32)).to(self.device).contiguous()
            self._check(self.lib.lst_set_node_order(self.h, ct.c_void_p(self._node_order.data_ptr())))
        self._opt_key = None
        self.update_clip = 0.0          # StableAdamW per-tensor update clipping (LstOptim.update_clip)
        self._stacks = None
        self.steps_done = 0
        self.setup_seconds = time.time() - t0

    def __del__(self):
        h = getattr(self, 'h', None)
        if h:
            self.lib.lst_destroy(h)
            self.h = None

    def _check(self, rc):
        if rc != 0:
            raise RuntimeError('loupstrain: ' + self.lib.lst_last_error(self.h).decode())

    def _stream(self):
        return ct.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)

    # ------------------------------------------------------------------ data
    def _topology(self):
        """Frame-local topology of every frame on the device (int32), for the library's batch assembly."""
        d = self.data
        cat = lambda xs: torch.from_numpy(np.concatenate(xs).astype(np.int32)).to(self.device).contiguous()
        npair = np.array([x.shape[0] for x in d.pe0_l], dtype=np.int64)
        t = dict(node_off=cat([d.node_off]), edge_off=cat([d.edge_off]),
                 pair_off=cat([np.concatenate([[0], np.cumsum(npair)])]),
                 sender=cat(d.snd_l), receiver=cat(d.rcv_l), receiver_ptr=cat(d.rptr_l), pidx=cat(d.pid_l),
                 pe0=cat(d.pe0_l), pe1=cat(d.pe1_l))
        self._n_frame, self._e_frame = d.n.astype(np.int64), d.e.astype(np.int64)
        return t

    def stack(self, batches: List[List[int]]):
        """The epoch's table of frame ids [n, B] (-1 padded), after checking that every batch fits the shape."""
        d = self.data
        nb = len(batches)
        ids = np.full((nb, d.B), -1, dtype=np.int32)
        for i, b in enumerate(batches):
            if not 0 < len(b) <= d.B:
                raise ValueError(f'batch {i} has {len(b)} graphs (1 .. {d.B})')
            ids[i, :len(b)] = b
        m = ids >= 0
        safe = np.where(m, ids, 0)
        n_real = (self._n_frame[safe] * m).sum(1)
        e_real = (self._e_frame[safe] * m).sum(1)
        nd, ed = d.N_pad - n_real, d.E_pad - e_real
        bad = (nd < 1) | (ed < 0) | (-(-ed // DUMMY_ROW) > nd)
        if bad.any():
            raise ValueError(f'batch {int(np.argmax(bad))} does not fit the padded shape')
        self._ids = np.ascontiguousarray(ids)
        return Batches(n=nb, frame_ids=self._ids.ctypes.data_as(ct.POINTER(ct.c_int32)))

    # ------------------------------------------------------------------ optimizer
    def _prepare_optimizer(self, optimizer, ema, max_grad_norm):
        # identity plus the storage of every state tensor (a load_state_dict replaces them)
        ptrs = tuple(t.data_ptr() for p in self.params for t in optimizer.state[p].values() if torch.is_tensor(t))
        shadow = tuple(s.data_ptr() for s in ema.shadow_params) if ema is not None else ()
        key = (id(optimizer), id(ema), (max_grad_norm, self.update_clip), ptrs, shadow,
               tuple(p.data_ptr() for p in self.params))
        if self._opt_key == key:
            return
        groups = optimizer.param_groups
        if len(groups) > 16:
            raise RuntimeError('more than 16 parameter groups')
        gid = {}
        for i, g in enumerate(groups):
            for p in g['params']:
                gid[id(p)] = i
        b1, b2 = groups[0]['betas']
        eps = groups[0]['eps']
        ams = bool(groups[0].get('amsgrad', False))
        for g in groups:
            if tuple(g['betas']) != (b1, b2) or g['eps'] != eps or bool(g.get('amsgrad', False)) != ams:
                raise RuntimeError('parameter groups with different betas / eps / amsgrad')
            if g.get('maximize', False):
                raise RuntimeError('maximize')
        o = Optim()
        self._opt_refs = []
        for i, p in enumerate(self.params):
            st = optimizer.state[p]
            if not st:
                st['step'] = torch.zeros((), dtype=torch.float32, device=self.device)
                st['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if ams:
                    st['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if not torch.is_tensor(st['step']) or st['step'].device != self.device or st['step'].dtype != torch.float32:
                st['step'] = torch.tensor(float(st['step']), dtype=torch.float32, device=self.device)
            for k in ('exp_avg', 'exp_avg_sq') + (('max_exp_avg_sq',) if ams else ()):
                if not st[k].is_contiguous() or st[k].dtype != self.dtype:
                    raise RuntimeError(f'optimizer state {k} of {PARAM_NAMES[i]}')
            o.exp_avg[i] = st['exp_avg'].data_ptr()
            o.exp_avg_sq[i] = st['exp_avg_sq'].data_ptr()
            o.max_exp_avg_sq[i] = st['max_exp_avg_sq'].data_ptr() if ams else None
            o.step[i] = st['step'].data_ptr()
            o.group[i] = gid[id(p)]
        # the fused / capturable paths keep the step on the device as well (stock steps after ours stay valid)
        for g in groups:
            g['capturable'] = True
        if ema is not None:
            shadow = list(ema.shadow_params)
            if len(shadow) != NP:
                raise RuntimeError('EMA shadows do not match the trainable parameters')
            for i, s in enumerate(shadow):
                if s.shape != self.params[i].shape or not s.is_contiguous() or s.dtype != self.dtype:
                    raise RuntimeError(f'EMA shadow of {PARAM_NAMES[i]}')
                o.ema[i] = s.data_ptr()
        o.n_groups = len(groups)
        o.beta1, o.beta2, o.eps = float(b1), float(b2), float(eps)
        for i, g in enumerate(groups):
            o.weight_decay[i] = float(g.get('weight_decay', 0.0))
        o.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else 0.0
        o.update_clip = float(self.update_clip or 0.0)
        self._optim = o
        if key[-1] != self._param_ptrs:          # parameters re-allocated (not a mace path, but cheap to follow)
            self._check(self.lib.lst_set_params(self.h, (P_ * NP)(*[p.data_ptr() for p in self.params])))
            self._param_ptrs = key[-1]
        self._check(self.lib.lst_set_optim(self.h, ct.byref(o)))
        ptrs = tuple(t.data_ptr() for p in self.params for t in optimizer.state[p].values() if torch.is_tensor(t))
        self._opt_key = key[:3] + (ptrs,) + key[4:]

    def set_muon(self, blocks, momentum: float = 0.95, nesterov: bool = True, scale: float = 0.2,
                 ns_steps: int = 5, ns_polish: int = 0, normuon_beta=None):
        """Muon on matrix blocks (``muon.library_blocks``: (param index, batch, rows, cols, offset, batch / row /
        column strides)); their tensors' ``exp_avg`` is the momentum and they take no L2 weight decay.
        ``ns_steps`` quintic Newton-Schulz steps, then ``ns_polish`` with (2, -1.5, 0.5). ``blocks=[]`` turns
        it off."""
        arr = (MuonBlock * max(1, len(blocks)))(*[MuonBlock(*b) for b in blocks])
        self._check(self.lib.lst_set_muon(self.h, arr, len(blocks), float(momentum), int(bool(nesterov)),
                                          float(scale)))
        self._check(self.lib.lst_set_muon_ns(self.h, int(ns_steps), int(ns_polish)))
        if normuon_beta is not None or hasattr(self.lib, 'lst_set_muon_norm'):
            if not hasattr(self.lib, 'lst_set_muon_norm'):
                raise RuntimeError('this libloupstrain has no NorMuon (lst_set_muon_norm)')
            self._check(self.lib.lst_set_muon_norm(self.h, -1.0 if normuon_beta is None else float(normuon_beta)))
        self.muon_blocks = list(blocks)

    def set_cautious(self, on: bool = True):
        """Cautious Adam on the tensors Adam updates (muon.py cautious='adam')."""
        if not hasattr(self.lib, 'lst_set_cautious'):
            raise RuntimeError('this libloupstrain has no cautious Adam (lst_set_cautious)')
        self._check(self.lib.lst_set_cautious(self.h, 1 if on else 0))

    def set_muon_wd(self, wd: float, lr_mult: float = 1.0):
        """Decoupled weight decay of the Muon / SOAP blocks (muon.py muon_wd): p *= 1 - lr * lr_mult * wd
        before each step (after ``set_muon``); 0 turns it off."""
        if not hasattr(self.lib, 'lst_set_muon_wd'):
            raise RuntimeError('this libloupstrain has no decoupled Muon weight decay (lst_set_muon_wd)')
        self._check(self.lib.lst_set_muon_wd(self.h, float(wd) * float(lr_mult)))

    def set_math(self, gemm: Optional[str] = None, mlp: Optional[str] = None):
        """The GEMMs' and the radial MLPs' arithmetic from the next step on: 'fp32', 'tf32' or 'bf16'; None keeps
        that part (``LST_MATH`` / ``LST_MLP_TC`` set them at creation)."""
        if not hasattr(self.lib, 'lst_set_math'):
            raise RuntimeError('this libloupstrain cannot change its arithmetic (lst_set_math)')
        modes = {None: -1, 'fp32': 0, 'tf32': 1, 'bf16': 2}
        if gemm not in modes or mlp not in modes:
            raise ValueError(f'math modes: fp32, tf32, bf16 or None, not {gemm!r} / {mlp!r}')
        self._check(self.lib.lst_set_math(self.h, modes[gemm], modes[mlp]))

    def set_soap(self, betas=(0.95, 0.95), eps: float = 1e-8, every: int = 10, max_dim: int = 64,
                 lr_mult: float = 1.0, sweeps: Optional[int] = None, block: Optional[bool] = None):
        """SOAP on the Muon blocks (after ``set_muon``): muon.py matrix='soap' with soap_eig='jacobi' and
        soap_max_dim; ``every=0`` turns it off. ``sweeps`` caps the Jacobi sweeps of the warm-started basis
        refreshes (the first refresh always runs to convergence); None keeps LST_SOAP_SWEEPS or 30. ``block``: a
        matrix whose long side is above 64 gets one two-sided preconditioner per 64-column block (None keeps
        LST_SOAP_BLOCK or off)."""
        if not hasattr(self.lib, 'lst_set_soap'):
            raise RuntimeError('this libloupstrain has no SOAP (lst_set_soap)')
        env = {}
        if sweeps is not None:
            env['LST_SOAP_SWEEPS'] = str(max(1, int(sweeps)))
        if block is not None:
            env['LST_SOAP_BLOCK'] = '1' if block else '0'
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            self._check(self.lib.lst_set_soap(self.h, float(betas[0]), float(betas[1]), float(eps), int(every),
                                              int(max_dim), float(lr_mult)))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    @staticmethod
    def _ema_decays(ema, n):
        if ema is None:
            return None
        out = []
        for _ in range(n):
            decay = ema.decay
            if ema.num_updates is not None:
                ema.num_updates += 1
                decay = min(decay, (1 + ema.num_updates) / (10 + ema.num_updates))
            out.append(decay)
        return out

    # ------------------------------------------------------------------ epochs
    def run_epoch(self, batches: List[List[int]], loss_fn, optimizer, ema, max_grad_norm, lr_steps=None,
                  weight_steps=None):
        """Run one epoch of optimizer steps; returns per-step losses (floats) and seconds.

        ``lr_steps`` ([steps][groups]) and ``weight_steps`` ([steps][3]: energy, forces, stress) give each step its
        own learning rates and loss weights (a recipe's schedule); without them every step takes the optimizer's
        current ``lr`` and ``loss_fn``'s weights. The optimizer's ``lr`` is left as it is."""
        t0 = time.time()
        self._prepare_optimizer(optimizer, ema, max_grad_norm)
        n = len(batches)
        if lr_steps is None and weight_steps is None:
            lw = _dbl([float(loss_fn.energy_weight), float(loss_fn.forces_weight), float(loss_fn.stress_weight)])
            lr = _dbl([g['lr'] for g in optimizer.param_groups])
            begin = self.lib.lst_begin_epoch
        else:
            G = len(optimizer.param_groups)
            if lr_steps is None:
                lr_steps = [[g['lr'] for g in optimizer.param_groups]] * n
            if weight_steps is None:
                weight_steps = [[float(loss_fn.energy_weight), float(loss_fn.forces_weight),
                                 float(loss_fn.stress_weight)]] * n
            if len(lr_steps) != n or any(len(r) != G for r in lr_steps):
                raise ValueError(f'lr_steps must be [{n}][{G}]')
            if len(weight_steps) != n or any(len(r) != 3 for r in weight_steps):
                raise ValueError(f'weight_steps must be [{n}][3]')
            lr = _dbl([float(x) for r in lr_steps for x in r])
            lw = _dbl([float(x) for r in weight_steps for x in r])
            begin = self.lib.lst_begin_epoch_steps
        bt = self.stack(batches)
        dec = self._ema_decays(ema, n)
        s = self._stream()
        t1 = time.time()
        self._check(begin(self.h, ct.byref(bt), lr, lw, _dbl(dec) if dec else None, s))
        self._check(self.lib.lst_run_steps(self.h, len(batches), s))
        out = (ct.c_double * len(batches))()
        self._check(self.lib.lst_epoch_losses(self.h, out, s))
        self.last_times = (t1 - t0, time.time() - t1)
        self.steps_done += len(batches)
        return list(out), time.time() - t0

    # ------------------------------------------------------------------ debugging
    def debug_step(self, batches, loss_weights, which=0, update=False):
        """One step of batch ``which`` of ``batches`` outside the graph (no optimizer update unless asked)."""
        bt = self.stack(batches)
        s = self._stream()
        self._check(self.lib.lst_begin_epoch(self.h, ct.byref(bt), _dbl([0.0] * 16), _dbl(loss_weights), None, s))
        self._check(self.lib.lst_debug_step(self.h, int(which), int(bool(update)), s))

    def stage(self, name: str) -> torch.Tensor:
        n = self.lib.lst_stage_size(self.h, name.encode())
        if n < 0:
            raise KeyError(name)
        out = torch.empty(n, dtype=self.dtype, device=self.device)
        self._check(self.lib.lst_stage_copy(self.h, name.encode(), _ptr(out), self._stream()))
        return out

    def raw_grads(self):
        """The last step's gradient of each trainable parameter (before clipping)."""
        flat = self.stage('graw')
        out, o = [], 0
        for p in self.params:
            out.append(flat[o:o + p.numel()].view_as(p))
            o += p.numel()
        return out


class CudaEvaluator:
    """Energies, forces and stress of mace-torch batches from the training library (``lst_evaluate``: the step's
    forward and force pass), in mace's atom order: :class:`loupmd.fasttrain.engine.Evaluator`'s interface
    (``run``, ``rebind``) without the Triton kernels. Its own library engine on the set's frames holds ``B``
    frames per call (``LOUP_FT_CUDA_EVAL_B``, default 4, at most the loader's batch size); a batch of more frames
    is evaluated ``B`` at a time and yielded whole."""

    def __init__(self, model, dataset, batch_size: int, lib_path: Optional[str] = None):
        from .engine import Evaluator
        self._light_ok = Evaluator._light_ok.__get__(self)
        self._light_batch = Evaluator._light_batch.__get__(self)
        lib = load_library(lib_path)
        if not hasattr(lib, 'lst_evaluate'):
            raise RuntimeError('this libloupstrain has no lst_evaluate')
        self.B = max(1, min(int(batch_size), int(os.environ.get('LOUP_FT_CUDA_EVAL_B', '4'))))
        self.dataset = dataset
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.engine = CudaTrainer(model, dataset, batch_size=self.B, dtype=self.dtype, lib_path=lib_path,
                                  by_name=True)
        d = self.engine.data
        kw = dict(dtype=self.dtype, device=self.device)
        self._e = torch.empty(d.G1, **kw)
        self._f = torch.empty(d.N_pad, 3, **kw)
        self._s = torch.empty(d.G1, 9, **kw)
        self.orders = [torch.sort(item.node_attrs.argmax(dim=1).cpu(), stable=True).indices.numpy()
                       for item in dataset]

    def rebind(self, model):
        """Evaluate ``model`` (same architecture, frames and constants) from now on."""
        from .engine import FastStudent
        FastStudent(model)                             # the model checks
        named = _named_params(model, True)
        if [n for n, _ in named] != PARAM_NAMES:
            raise RuntimeError(f'unexpected trainable parameters: {[n for n, _ in named]}')
        for n, q in named:
            if q.dtype != self.dtype or not q.is_contiguous() or q.device != self.device:
                raise RuntimeError(f'{n}: {q.dtype} {q.device} contiguous={q.is_contiguous()}')
        self.model = model
        self.engine.model = model
        self.engine.params = [q for _, q in named]
        self._sync_params()

    def _sync_params(self):
        tr = self.engine
        ptrs = tuple(q.data_ptr() for q in tr.params)
        if ptrs != tr._param_ptrs:                     # pylint: disable=protected-access
            tr._check(tr.lib.lst_set_params(tr.h, (P_ * NP)(*ptrs)))   # pylint: disable=protected-access
            tr._param_ptrs = ptrs                      # pylint: disable=protected-access

    def run(self, batches, data_loader, light=False):
        """Yields (ids, collated mace batch on the device, output dict) per batch, as ``Evaluator.run``."""
        light = light and self._light_ok()
        cache = self.__dict__.setdefault('_light_batches' if light else '_batches', {})
        tr = self.engine
        d = tr.data
        self._sync_params()
        stream = tr._stream()                           # pylint: disable=protected-access
        e_p, f_p, s_p = _ptr(self._e), _ptr(self._f), _ptr(self._s)
        for ids in batches:
            key = tuple(int(i) for i in ids)
            hit = cache.get(key)
            if hit is None:
                pieces = [list(key[k:k + self.B]) for k in range(0, len(key), self.B)]
                tr.stack(pieces)                        # every piece fits the padded shape
                arrs = [np.ascontiguousarray(q, dtype=np.int32) for q in pieces]
                rows = [int(d.n[q].sum()) for q in pieces]
                n0, perm = 0, []
                for i in key:
                    perm.append(self.orders[i] + n0)
                    n0 += int(d.n[i])
                perm = torch.from_numpy(np.concatenate(perm)).to(self.device)
                idx = torch.tensor(key, device=self.device)
                batch = (self._light_batch(key) if light
                         else data_loader.collate_fn([self.dataset[i] for i in key]).to(self.device))
                hit = cache[key] = (arrs, rows, perm, n0, idx, batch)
            arrs, rows, perm, n0, idx, batch = hit
            G = len(key)
            kw = dict(dtype=self.dtype, device=self.device)
            energy = torch.empty(G, **kw)
            f_rows = torch.empty(n0, 3, **kw)
            stress = torch.empty(G, 3, 3, **kw)
            g0 = r0 = 0
            for a, nr in zip(arrs, rows):
                k = len(a)
                tr._check(tr.lib.lst_evaluate(tr.h, a.ctypes.data_as(ct.POINTER(ct.c_int32)), k, e_p, f_p, s_p,
                                              stream))   # pylint: disable=protected-access
                energy[g0:g0 + k] = self._e[:k]
                f_rows[r0:r0 + nr] = self._f[:nr]
                stress[g0:g0 + k] = self._s[:k].view(k, 3, 3)
                g0 += k
                r0 += nr
            forces = torch.empty(n0, 3, **kw)
            forces[perm] = f_rows
            vol = d.volume.index_select(0, idx)
            yield ids, batch, dict(energy=energy, forces=forces, stress=stress,
                                   virials=-stress * vol[:, None, None])
