#!/usr/bin/env python3
"""Universal General Full Pure-CUDA LRSK Trainer (`src/train/lrsk_full_trainer.py`).

Trains FastLRSKStudent on ANY atomic structure (crystals, liquids, molecules, variable-size
graphs, any species S, any cutoffs r_max/r_max2, lmax<=3, lfeat<=2, taper, n_base/n_base2<=16,
c0/c1/c2<=16, arbitrary readout_depth/width, film_radials=True/False, pol_field/pol_gram,
sw_mp/mp_cap/ef_cap, ZBL switch, stress, per-atom ei_ref, molecular COM/torque, per-species
and per-regime weights) with ZERO Python/PyTorch autograd in the training loop!

All forward, 1st-order force/virial backward, loss, exact 2nd-order dual-number backward,
grad-norm clip, GPU Block-Jacobi SOAP, compact cautious AMSGrad-Adam, and EMA run inside
`build/libloup_lrsk_full_train.so` (`lrsk_ft_run_epoch_f32`).

Usage:
  1. Exact FP64 gradcheck vs PyTorch autograd + FP32 speed check:
     python src/train/lrsk_full_trainer.py --gradcheck
  2. Train any dataset -> standard FastLRSKStudent .pt checkpoint:
     python src/train/lrsk_full_trainer.py --name my_model --train train.xyz --valid valid.xyz --test test.xyz ...
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / ".tmp_o"), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from ase.io import iread
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric import Batch

from p2_lrsk_v4_fast import FastLRSKStudent
from p2_ilmlip_fast import eval_frames_pt, eval_sto_phonons_pt, fit_min_norm_e0, prebuild_data
from loupmd.distill.molecular import ReferenceScales, grouped_geometry
from lrsk_train_ops import _build_csr_single, _compute_zbl_sparse, precompute_G1_cuda


# =============================================================================
# ctypes Struct Mirrors (exact match to src/train/lrsk_full_kernels.cuh)
# =============================================================================
class CgEntryF32(ctypes.Structure):
    _fields_ = [("a", ctypes.c_int), ("b", ctypes.c_int), ("m", ctypes.c_int), ("val", ctypes.c_float)]


class CgEntryF64(ctypes.Structure):
    _fields_ = [("a", ctypes.c_int), ("b", ctypes.c_int), ("m", ctypes.c_int), ("val", ctypes.c_double)]


class Nu2Path(ctypes.Structure):
    _fields_ = [
        ("l1", ctypes.c_int), ("l2", ctypes.c_int), ("L", ctypes.c_int), ("c", ctypes.c_int),
        ("dst_b2_off", ctypes.c_int), ("dst_bcat_off", ctypes.c_int),
        ("cg_start", ctypes.c_int), ("cg_count", ctypes.c_int),
    ]


class Nu3Path(ctypes.Structure):
    _fields_ = [
        ("p2_idx", ctypes.c_int), ("l3", ctypes.c_int), ("L", ctypes.c_int), ("c", ctypes.c_int),
        ("dst_bcat_off", ctypes.c_int),
        ("cg_start", ctypes.c_int), ("cg_count", ctypes.c_int),
    ]


class ModelDims(ctypes.Structure):
    _fields_ = [
        ("S", ctypes.c_int), ("lmax", ctypes.c_int), ("lfeat", ctypes.c_int), ("lmsg", ctypes.c_int), ("n_sh", ctypes.c_int),
        ("K1", ctypes.c_int), ("K2", ctypes.c_int), ("n_tot1", ctypes.c_int), ("D_A1", ctypes.c_int),
        ("n_l", ctypes.c_int * 4),
        ("l1_off", ctypes.c_int * 4),
        ("a1_off", ctypes.c_int * 4),
        ("bcat_dim", ctypes.c_int * 3),
        ("n_nu2", ctypes.c_int), ("n_nu3", ctypes.c_int), ("B2_tot", ctypes.c_int),
        ("c0", ctypes.c_int), ("c1", ctypes.c_int), ("c2", ctypes.c_int),
        ("c01", ctypes.c_int), ("c02", ctypes.c_int), ("c12", ctypes.c_int), ("T_lr", ctypes.c_int),
        ("ro_depth", ctypes.c_int), ("ro_width", ctypes.c_int),
        ("film_radials", ctypes.c_int), ("sym", ctypes.c_int), ("pol_field", ctypes.c_int),
        ("pol_gram", ctypes.c_int), ("sw_mp", ctypes.c_int), ("use_zbl_switch", ctypes.c_int),
        ("pol_width", ctypes.c_int), ("D_pol", ctypes.c_int),
        ("max_c_cg", ctypes.c_int),
        ("r_max", ctypes.c_double), ("r_max2", ctypes.c_double), ("radial_warp", ctypes.c_double),
        ("kappa", ctypes.c_double), ("a0", ctypes.c_double), ("sigma0", ctypes.c_double),
        ("energy_scale", ctypes.c_double), ("z_c", ctypes.c_double),
        ("mp_cap", ctypes.c_double), ("ef_cap", ctypes.c_double),
    ]


_FT_LIB = None


def load_ft_lib() -> ctypes.CDLL:
    global _FT_LIB
    if _FT_LIB is not None:
        return _FT_LIB
    candidates = [
        os.environ.get("LOUP_LRSK_FT_LIB", ""),
        str(_REPO_ROOT / "build" / "libloup_lrsk_full_train_v3.so"),
        str(_THIS_DIR / "libloup_lrsk_full_train_v3.so"),
        "/mnt/homes/zwingel/loups-phase1/libloup_lrsk_full_train_v3.so",
        str(_REPO_ROOT / "build" / "libloup_lrsk_full_train_v2.so"),
        str(_THIS_DIR / "libloup_lrsk_full_train_v2.so"),
        "/mnt/homes/zwingel/loups-phase1/libloup_lrsk_full_train_v2.so",
        str(_REPO_ROOT / "build" / "libloup_lrsk_full_train.so"),
        str(_THIS_DIR / "libloup_lrsk_full_train.so"),
        "/mnt/homes/zwingel/loups-phase1/libloup_lrsk_full_train.so",
    ]
    lib_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if lib_path is None:
        raise FileNotFoundError(f"Could not find libloup_lrsk_full_train.so in {candidates}")
    lib = ctypes.CDLL(lib_path)

    vp = ctypes.c_void_p
    ci = ctypes.c_int
    cl = ctypes.c_long
    cd = ctypes.c_double
    for sfx in ("f32", "f64"):
        getattr(lib, f"lrsk_ft_create_{sfx}").argtypes = [vp, vp, vp, vp, ci, vp, vp, vp, vp, vp, ci]
        getattr(lib, f"lrsk_ft_create_{sfx}").restype = vp
        getattr(lib, f"lrsk_ft_destroy_{sfx}").argtypes = [vp]
        getattr(lib, f"lrsk_ft_destroy_{sfx}").restype = None
        getattr(lib, f"lrsk_ft_num_params_{sfx}").argtypes = [vp]
        getattr(lib, f"lrsk_ft_num_params_{sfx}").restype = ci
        getattr(lib, f"lrsk_ft_param_len_{sfx}").argtypes = [vp, ci]
        getattr(lib, f"lrsk_ft_param_len_{sfx}").restype = cl
        getattr(lib, f"lrsk_ft_set_param_{sfx}").argtypes = [vp, ci, vp, ci]
        getattr(lib, f"lrsk_ft_set_param_{sfx}").restype = ci
        getattr(lib, f"lrsk_ft_get_param_{sfx}").argtypes = [vp, ci, vp, ci]
        getattr(lib, f"lrsk_ft_get_param_{sfx}").restype = ci
        getattr(lib, f"lrsk_ft_get_grad_{sfx}").argtypes = [vp, ci, vp]
        getattr(lib, f"lrsk_ft_get_grad_{sfx}").restype = ci
        getattr(lib, f"lrsk_ft_set_opt_{sfx}").argtypes = [
            vp, cd, cd, ci, cd, cd, cd, cd, cd, cd, cd, cd, cd, vp, ci
        ]
        getattr(lib, f"lrsk_ft_set_opt_{sfx}").restype = ci
        getattr(lib, f"lrsk_ft_upload_dataset_{sfx}").argtypes = [
            vp, ci, ci, ci, ci, ci, ci, ci, ci, ci,
            vp, vp, vp, vp, vp, vp, vp, vp, vp,
            vp, vp, vp, vp, vp, vp, vp, vp, vp,
            vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp
        ]
        getattr(lib, f"lrsk_ft_upload_dataset_{sfx}").restype = ci
        getattr(lib, f"lrsk_ft_run_epoch_{sfx}").argtypes = [
            vp, vp, ci, ci, vp, vp, vp, vp, vp, vp, cd, cd, vp
        ]
        getattr(lib, f"lrsk_ft_run_epoch_{sfx}").restype = ci
        if hasattr(lib, f"lrsk_ft_run_epoch_weighted_{sfx}"):
            getattr(lib, f"lrsk_ft_run_epoch_weighted_{sfx}").argtypes = [
                vp, vp, vp, ci, ci, vp, vp, vp, ci, vp, vp
            ]
            getattr(lib, f"lrsk_ft_run_epoch_weighted_{sfx}").restype = ci
            getattr(lib, f"lrsk_ft_get_frame_err_{sfx}").argtypes = [vp, vp]
            getattr(lib, f"lrsk_ft_get_frame_err_{sfx}").restype = ci
            getattr(lib, f"lrsk_ft_gradcheck_weighted_{sfx}").argtypes = [
                vp, vp, vp, ci, cd, cd, ci, vp, vp, vp
            ]
            getattr(lib, f"lrsk_ft_gradcheck_weighted_{sfx}").restype = ci
        getattr(lib, f"lrsk_ft_gradcheck_{sfx}").argtypes = [
            vp, vp, ci, cd, cd, cd, cd, cd, cd, cd, ci, vp, vp, vp, vp
        ]
        getattr(lib, f"lrsk_ft_gradcheck_{sfx}").restype = ci

    _FT_LIB = lib
    return lib


def canonical_param_list(model: FastLRSKStudent) -> list[tuple[str, torch.nn.Parameter]]:
    """Exact canonical parameter order matching LrskFullTrainer::init in src/train/lrsk_full_train.cu."""
    out: list[tuple[str, torch.nn.Parameter]] = []
    if model.film_radials:
        out.extend([
            ("C1_R", model.C1_R),
            ("C1_B", model.C1_B),
            ("gamma1", model.gamma1),
            ("beta1", model.beta1),
            ("eta1", model.eta1),
        ])
    else:
        out.append(("C1", model.C1))
    for d, lin in enumerate(model.readout_layers):
        out.append((f"readout_layers.{d}.weight", lin.weight))
        out.append((f"readout_layers.{d}.bias", lin.bias))
    out.append(("alpha_Z", model.alpha_Z))
    out.append(("u_Z", model.u_Z))
    if model.c0 > 0:
        out.append(("W_qA", model.W_qA))
        out.append(("b_qA", model.b_qA))
    if model.c1 > 0:
        out.append(("W_p1A", model.W_p1A))
        out.append(("W_g1A", model.W_g1A))
        out.append(("b_g1A", model.b_g1A))
    if model.c2 > 0:
        out.append(("W_p2A", model.W_p2A))
        out.append(("W_g2A", model.W_g2A))
        out.append(("b_g2A", model.b_g2A))
    out.append(("W_lr_rad", model.W_lr_rad))
    if model.pol_field:
        out.append(("pol_W0.weight", model.pol_W0.weight))
        out.append(("pol_W0.bias", model.pol_W0.bias))
        out.append(("pol_u", model.pol_u))
    return out


class CudaFullLRSKTrainer:
    """High-level wrapper around LrskFullTrainer<float|double>."""

    def __init__(self, model: FastLRSKStudent, dtype: str = "f32", soap_max_dim: int = 128):
        assert model.sym, "Pure-CUDA LRSK trainer currently requires sym=True"
        assert model.pair_mlp_width == 0, "Pure-CUDA LRSK trainer currently requires pair_mlp_width=0"
        assert model.lmax <= 3 and model.lfeat <= 2
        self.model = model
        self.sfx = dtype
        self.np_dt = np.float64 if dtype == "f64" else np.float32
        self.th_dt = torch.float64 if dtype == "f64" else torch.float32
        self.lib = load_ft_lib()
        self.param_list = canonical_param_list(model)

        # Build sparse CG tables + nu2/nu3 paths
        CgT = CgEntryF64 if dtype == "f64" else CgEntryF32
        cg_entries = []
        cg_cache: dict[tuple[int, int, int], tuple[int, int]] = {}

        def _get_cg(l1: int, l2: int, L: int) -> tuple[int, int]:
            key = (l1, l2, L)
            if key in cg_cache:
                return cg_cache[key]
            t = getattr(model, f"cg_{l1}_{l2}_{L}").detach().cpu().double().numpy()
            start = len(cg_entries)
            d1, d2, dL = t.shape
            for a in range(d1):
                for b in range(d2):
                    for m in range(dL):
                        v = float(t[a, b, m])
                        if abs(v) > 1e-14:
                            cg_entries.append(CgT(a, b, m, v))
            cnt = len(cg_entries) - start
            cg_cache[key] = (start, cnt)
            return start, cnt

        # Compute exact B_cat offsets matching FastLRSKStudent._compute_l1
        bcat_cur = [model.n_l[l] if l <= model.lmsg else 0 for l in range(3)]
        nu2_arr = (Nu2Path * max(1, len(model.nu2_paths)))()
        b2_off = 0
        for idx, (l1, l2, L, c) in enumerate(model.nu2_paths):
            st, cnt = _get_cg(l1, l2, L)
            if L <= model.lmsg:
                bo = bcat_cur[L]
                bcat_cur[L] += c
            else:
                bo = -1
            nu2_arr[idx] = Nu2Path(l1, l2, L, c, b2_off, bo, st, cnt)
            b2_off += c * (2 * L + 1)

        nu3_arr = (Nu3Path * max(1, len(model.nu3_paths)))()
        for idx, (p2_idx, l3, L, c) in enumerate(model.nu3_paths):
            l12 = model.nu2_paths[p2_idx][2]
            st, cnt = _get_cg(l12, l3, L)
            bo = bcat_cur[L]
            bcat_cur[L] += c
            nu3_arr[idx] = Nu3Path(p2_idx, l3, L, c, bo, st, cnt)

        for l in range(model.lmsg + 1):
            assert bcat_cur[l] == model.bcat_dim[l], f"bcat mismatch l={l}: {bcat_cur[l]} vs {model.bcat_dim[l]}"

        cg_arr = (CgT * max(1, len(cg_entries)))(*cg_entries)

        dims = ModelDims()
        dims.S = model.S
        dims.lmax = model.lmax
        dims.lfeat = model.lfeat
        dims.lmsg = model.lmsg
        dims.n_sh = (model.lmax + 1) ** 2
        dims.K1 = model.n_base
        dims.K2 = model.n_base2
        dims.n_tot1 = model.n_tot1
        dims.D_A1 = model.D_A1
        a_off = 0
        for l in range(4):
            nl = model.n_l[l] if l <= model.lmax else 0
            dims.n_l[l] = nl
            dims.l1_off[l] = model.l1_offsets[l][0] if l <= model.lmax else model.n_tot1
            dims.a1_off[l] = a_off
            a_off += nl * (2 * l + 1)
        for l in range(3):
            dims.bcat_dim[l] = model.bcat_dim[l] if l <= model.lmsg else 0
        dims.n_nu2 = len(model.nu2_paths)
        dims.n_nu3 = len(model.nu3_paths)
        dims.B2_tot = b2_off
        dims.c0, dims.c1, dims.c2 = model.c0, model.c1, model.c2
        dims.c01, dims.c02, dims.c12 = model.c01, model.c02, model.c12
        dims.T_lr = model.n_lr_terms
        dims.ro_depth = len(model.readout_layers)
        dims.ro_width = int(model.readout_layers[0].out_features)
        dims.film_radials = int(model.film_radials)
        dims.sym = int(model.sym)
        dims.pol_field = int(model.pol_field)
        dims.pol_gram = int(model.pol_gram)
        dims.sw_mp = int(model.sw_mp)
        dims.use_zbl_switch = int(model.use_zbl_switch)
        dims.pol_width = int(model.field_mlp_width) if model.pol_field else 0
        dims.D_pol = int(model.pol_W0.in_features) if model.pol_field else 0
        dims.max_c_cg = max(model.n_l)
        dims.r_max = float(model.r_max)
        dims.r_max2 = float(model.r_max2)
        dims.radial_warp = float(model.radial_warp)
        dims.kappa = float(model.kappa)
        dims.a0 = float(model.a0)
        dims.sigma0 = float(model.sigma0.item())
        dims.energy_scale = float(model.energy_scale)
        dims.z_c = float(model.z_c)
        dims.mp_cap = float(model.mp_cap)
        dims.ef_cap = float(model.ef_cap)

        lr_sym = np.ascontiguousarray(model.lr_sym_mask.detach().cpu().numpy().astype(np.int32))
        mu_x = np.ascontiguousarray(model.mu_x.detach().cpu().numpy().astype(self.np_dt))
        std_x = np.ascontiguousarray(model.std_x.detach().cpu().numpy().astype(self.np_dt))
        std_b1 = np.ascontiguousarray(model.std_b1.detach().cpu().numpy().astype(self.np_dt)) if model.c1 > 0 else np.zeros(1, dtype=self.np_dt)
        std_b2 = np.ascontiguousarray(model.std_b2.detach().cpu().numpy().astype(self.np_dt)) if model.c2 > 0 else np.zeros(1, dtype=self.np_dt)

        fn = getattr(self.lib, f"lrsk_ft_create_{self.sfx}")
        self.handle = fn(
            ctypes.byref(dims), ctypes.byref(nu2_arr), ctypes.byref(nu3_arr),
            ctypes.byref(cg_arr), len(cg_entries), lr_sym.ctypes.data,
            mu_x.ctypes.data, std_x.ctypes.data, std_b1.ctypes.data, std_b2.ctypes.data,
            int(soap_max_dim),
        )
        if not self.handle:
            raise RuntimeError("lrsk_ft_create failed!")
        n_c = getattr(self.lib, f"lrsk_ft_num_params_{self.sfx}")(self.handle)
        assert n_c == len(self.param_list), f"Param count mismatch: C={n_c} vs Py={len(self.param_list)}"
        for pid, (name, p) in enumerate(self.param_list):
            clen = getattr(self.lib, f"lrsk_ft_param_len_{self.sfx}")(self.handle, pid)
            assert clen == p.numel(), f"Param {name} len mismatch: C={clen} vs Py={p.numel()}"
        self.upload_params(init_ema=True)

    def close(self):
        if getattr(self, "handle", None):
            getattr(self.lib, f"lrsk_ft_destroy_{self.sfx}")(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def upload_params(self, init_ema: bool = True):
        fn = getattr(self.lib, f"lrsk_ft_set_param_{self.sfx}")
        for pid, (_, p) in enumerate(self.param_list):
            arr = np.ascontiguousarray(p.detach().cpu().numpy().astype(self.np_dt).ravel())
            rc = fn(self.handle, pid, arr.ctypes.data, int(init_ema))
            if rc:
                raise RuntimeError(f"set_param {pid} failed rc={rc}")

    def download_params_to_model(self, from_ema: bool = True):
        fn = getattr(self.lib, f"lrsk_ft_get_param_{self.sfx}")
        with torch.no_grad():
            for pid, (_, p) in enumerate(self.param_list):
                buf = np.empty(p.numel(), dtype=self.np_dt)
                rc = fn(self.handle, pid, buf.ctypes.data, int(from_ema))
                if rc:
                    raise RuntimeError(f"get_param {pid} failed rc={rc}")
                p.data.copy_(torch.from_numpy(buf.reshape(p.shape)).to(device=p.device, dtype=p.dtype))

    def download_grads(self) -> dict[str, np.ndarray]:
        fn = getattr(self.lib, f"lrsk_ft_get_grad_{self.sfx}")
        out = {}
        for pid, (name, p) in enumerate(self.param_list):
            buf = np.empty(p.numel(), dtype=self.np_dt)
            rc = fn(self.handle, pid, buf.ctypes.data)
            if rc:
                raise RuntimeError(f"get_grad {pid} failed rc={rc}")
            out[name] = buf.reshape(p.shape)
        return out

    def set_optimizer(
        self, soap_b1: float, soap_b2: float, soap_every: int, soap_wd: float, soap_mult: float,
        adam_b1: float, adam_b2: float, ema_dmax: float, grad_clip: float,
        w_lr_reg: float, rad_smooth: float, wd_ro: float, sp_w: np.ndarray | None = None,
    ):
        fn = getattr(self.lib, f"lrsk_ft_set_opt_{self.sfx}")
        if sp_w is not None:
            sp_arr = np.ascontiguousarray(sp_w.astype(self.np_dt))
            sp_ptr, has_sp = sp_arr.ctypes.data, 1
        else:
            sp_arr, sp_ptr, has_sp = None, 0, 0
        rc = fn(
            self.handle, soap_b1, soap_b2, int(soap_every), soap_wd, soap_mult,
            adam_b1, adam_b2, ema_dmax, grad_clip, w_lr_reg, rad_smooth, wd_ro, sp_ptr, has_sp,
        )
        if rc:
            raise RuntimeError(f"set_opt failed rc={rc}")

    def upload_dataset(
        self, items: list, mol_list: list | None = None, graph_weights: np.ndarray | None = None,
        has_stress: bool = True, device: torch.device = torch.device("cuda"),
    ):
        """Precomputes CSR, G1, and sparse ZBL on GPU once, then uploads flat arrays into C++ trainer."""
        m = self.model
        G = len(items)
        split_r2 = 1 if (abs(m.r_max2 - m.r_max) > 1e-6) else 0
        has_ei = 1 if (hasattr(items[0], "ei_ref") and items[0].ei_ref is not None) else 0
        has_mol = 1 if (mol_list is not None and any(x is not None for x in mol_list)) else 0
        has_st = 1 if (has_stress and hasattr(items[0], "stress") and items[0].stress is not None) else 0

        n_atoms, n_e1, n_e2, n_zbl, n_mol = [], [], [], [], []
        l_z, l_deg1, l_deg2, l_snd1, l_snd2 = [], [], [], [], []
        l_vec1, l_vec2, l_G1, l_f = [], [], [], []
        l_E, l_vol, l_S = [], [], []
        l_ezbl, l_sw, l_dsw = [], [], []
        l_zsnd, l_zrcv, l_zgv, l_zvec = [], [], [], []
        l_ei, l_mid, l_arms = [], [], []

        for g, ad in enumerate(items):
            pos = ad.positions.to(device=device, dtype=self.th_dt)
            ei = ad.edge_index.to(device=device, dtype=torch.long)
            sh = ad.shifts.to(device=device, dtype=self.th_dt)
            vec = pos[ei[1]] - pos[ei[0]] + sh
            snd, rcv = ei[0], ei[1]
            z_i32 = ad.node_attrs.to(device=device).argmax(dim=-1).to(torch.int32)
            N = int(pos.shape[0])

            r_all = torch.linalg.norm(vec, dim=-1).clamp_min(1e-8)
            if m.r_max < m.r_max2 - 1e-6:
                m1 = r_all < m.r_max
                rp1, s1, v1 = _build_csr_single(snd[m1], rcv[m1], vec[m1], N)
            else:
                rp1, s1, v1 = _build_csr_single(snd, rcv, vec, N)
            G1 = precompute_G1_cuda(rp1, s1, v1, z_i32, m.S, m.n_base, m.lmax, m.r_max, m.radial_warp)
            n_atoms.append(N)
            n_e1.append(int(s1.numel()))
            l_z.append(z_i32.cpu().numpy())
            l_deg1.append((rp1[1:] - rp1[:-1]).cpu().numpy())
            l_snd1.append(s1.cpu().numpy())
            l_vec1.append(v1.cpu().numpy().astype(self.np_dt))
            l_G1.append(G1.cpu().numpy().astype(self.np_dt))

            if split_r2:
                if m.r_max2 < m.r_max - 1e-6:
                    m2 = r_all < m.r_max2
                    rp2, s2, v2 = _build_csr_single(snd[m2], rcv[m2], vec[m2], N)
                else:
                    m2 = r_all < m.r_max2
                    rp2, s2, v2 = _build_csr_single(snd[m2], rcv[m2], vec[m2], N)
                n_e2.append(int(s2.numel()))
                l_deg2.append((rp2[1:] - rp2[:-1]).cpu().numpy())
                l_snd2.append(s2.cpu().numpy())
                l_vec2.append(v2.cpu().numpy().astype(self.np_dt))
            else:
                n_e2.append(int(s1.numel()))

            ez, sw, dsw, zs, zr, zgv, zv = _compute_zbl_sparse(vec, snd, rcv, z_i32, m, N)
            if not m.use_zbl_switch:
                ez = torch.zeros_like(ez); sw = torch.ones_like(sw); dsw = torch.zeros_like(dsw)
                zs = zr = torch.empty(0, dtype=torch.long, device=device)
                zgv = zv = torch.empty((0, 3), dtype=self.th_dt, device=device)
            n_zbl.append(int(zr.numel()))
            l_ezbl.append(ez.cpu().numpy().astype(self.np_dt))
            l_sw.append(sw.cpu().numpy().astype(self.np_dt))
            l_dsw.append(dsw.cpu().numpy().astype(self.np_dt))
            if zr.numel() > 0:
                l_zsnd.append(zs.cpu().numpy().astype(np.int32))
                l_zrcv.append(zr.cpu().numpy().astype(np.int32))
                l_zgv.append(zgv.cpu().numpy().astype(self.np_dt))
                l_zvec.append(zv.cpu().numpy().astype(self.np_dt))

            l_f.append(ad.forces.detach().cpu().numpy().astype(self.np_dt))
            l_E.append(float(ad.energy.reshape(-1)[0].item()))
            cell = ad.cell.detach().cpu().double().reshape(3, 3)
            l_vol.append(float(torch.abs(torch.linalg.det(cell)).item()))
            if has_st:
                l_S.append(ad.stress.detach().cpu().numpy().astype(self.np_dt).reshape(9))
            if has_ei:
                l_ei.append(ad.ei_ref.detach().cpu().numpy().astype(self.np_dt))
            if has_mol:
                mg = mol_list[g]
                if mg is None:
                    n_mol.append(0)
                    l_mid.append(np.full(N, -1, dtype=np.int32))
                    l_arms.append(np.zeros((N, 3), dtype=self.np_dt))
                else:
                    mol_idx, n_m, arms = mg
                    n_mol.append(int(n_m))
                    l_mid.append(mol_idx.detach().cpu().numpy().astype(np.int32))
                    l_arms.append(arms.detach().cpu().numpy().astype(self.np_dt))
            else:
                n_mol.append(0)

        a_off = np.zeros(G + 1, dtype=np.int32); np.cumsum(n_atoms, out=a_off[1:])
        e1_off = np.zeros(G + 1, dtype=np.int32); np.cumsum(n_e1, out=e1_off[1:])
        e2_off = np.zeros(G + 1, dtype=np.int32); np.cumsum(n_e2, out=e2_off[1:])
        ez_off = np.zeros(G + 1, dtype=np.int32); np.cumsum(n_zbl, out=ez_off[1:])

        na_a = np.asarray(n_atoms, dtype=np.int32)
        ne1_a = np.asarray(n_e1, dtype=np.int32)
        ne2_a = np.asarray(n_e2, dtype=np.int32)
        nez_a = np.asarray(n_zbl, dtype=np.int32)
        nm_a = np.asarray(n_mol, dtype=np.int32)

        z_a = np.ascontiguousarray(np.concatenate(l_z), dtype=np.int32)
        d1_a = np.ascontiguousarray(np.concatenate(l_deg1), dtype=np.int32)
        d2_a = np.ascontiguousarray(np.concatenate(l_deg2), dtype=np.int32) if split_r2 else np.zeros(1, dtype=np.int32)
        s1_a = np.ascontiguousarray(np.concatenate(l_snd1), dtype=np.int32)
        s2_a = np.ascontiguousarray(np.concatenate(l_snd2), dtype=np.int32) if split_r2 else np.zeros(1, dtype=np.int32)
        v1_a = np.ascontiguousarray(np.concatenate(l_vec1, axis=0), dtype=self.np_dt)
        v2_a = np.ascontiguousarray(np.concatenate(l_vec2, axis=0), dtype=self.np_dt) if split_r2 else np.zeros(1, dtype=self.np_dt)
        G1_a = np.ascontiguousarray(np.concatenate(l_G1, axis=0), dtype=self.np_dt)
        f_a = np.ascontiguousarray(np.concatenate(l_f, axis=0), dtype=self.np_dt)
        E_a = np.ascontiguousarray(l_E, dtype=self.np_dt)
        vol_a = np.ascontiguousarray(l_vol, dtype=self.np_dt)
        S_a = np.ascontiguousarray(np.concatenate(l_S, axis=0), dtype=self.np_dt) if has_st else np.zeros(1, dtype=self.np_dt)
        gw_a = np.ascontiguousarray(graph_weights.astype(self.np_dt)) if graph_weights is not None else np.ones(G, dtype=self.np_dt)
        ez_a = np.ascontiguousarray(np.concatenate(l_ezbl), dtype=self.np_dt)
        sw_a = np.ascontiguousarray(np.concatenate(l_sw), dtype=self.np_dt)
        dsw_a = np.ascontiguousarray(np.concatenate(l_dsw), dtype=self.np_dt)

        tot_Ez = int(ez_off[-1])
        if tot_Ez > 0:
            zs_a = np.ascontiguousarray(np.concatenate(l_zsnd), dtype=np.int32)
            zr_a = np.ascontiguousarray(np.concatenate(l_zrcv), dtype=np.int32)
            zgv_a = np.ascontiguousarray(np.concatenate(l_zgv, axis=0), dtype=self.np_dt)
            zv_a = np.ascontiguousarray(np.concatenate(l_zvec, axis=0), dtype=self.np_dt)
        else:
            zs_a = zr_a = np.zeros(1, dtype=np.int32)
            zgv_a = zv_a = np.zeros(1, dtype=self.np_dt)

        ei_a = np.ascontiguousarray(np.concatenate(l_ei), dtype=self.np_dt) if has_ei else np.zeros(1, dtype=self.np_dt)
        mid_a = np.ascontiguousarray(np.concatenate(l_mid), dtype=np.int32) if has_mol else np.zeros(1, dtype=np.int32)
        arm_a = np.ascontiguousarray(np.concatenate(l_arms, axis=0), dtype=self.np_dt) if has_mol else np.zeros(1, dtype=self.np_dt)

        fn = getattr(self.lib, f"lrsk_ft_upload_dataset_{self.sfx}")
        rc = fn(
            self.handle, G, int(a_off[-1]), int(e1_off[-1]), int(e2_off[-1]), tot_Ez, split_r2,
            has_st, has_ei, has_mol,
            a_off.ctypes.data, e1_off.ctypes.data, e2_off.ctypes.data, ez_off.ctypes.data,
            na_a.ctypes.data, ne1_a.ctypes.data, ne2_a.ctypes.data, nez_a.ctypes.data, nm_a.ctypes.data,
            z_a.ctypes.data, d1_a.ctypes.data, d2_a.ctypes.data, s1_a.ctypes.data, s2_a.ctypes.data,
            v1_a.ctypes.data, v2_a.ctypes.data, G1_a.ctypes.data, f_a.ctypes.data,
            E_a.ctypes.data, vol_a.ctypes.data, S_a.ctypes.data, gw_a.ctypes.data,
            ez_a.ctypes.data, sw_a.ctypes.data, dsw_a.ctypes.data,
            zs_a.ctypes.data, zr_a.ctypes.data, zgv_a.ctypes.data, zv_a.ctypes.data,
            ei_a.ctypes.data, mid_a.ctypes.data, arm_a.ctypes.data,
        )
        if rc:
            raise RuntimeError(f"upload_dataset failed rc={rc}")
        g1_mb = G1_a.nbytes / (1024 * 1024)
        print(f"  PureCudaLRSKDataset: {G} graphs ({int(a_off[-1]):,} atoms, G1={g1_mb:.1f} MB, split_r2={bool(split_r2)}, "
              f"stress={bool(has_st)}, ei={bool(has_ei)}, mol={bool(has_mol)}, zbl_edges={tot_Ez})", flush=True)
        self._n_atoms = n_atoms

    def run_epoch(
        self, perm: np.ndarray, batch_size: int,
        lr_s: np.ndarray, we_s: np.ndarray, wf_s: np.ndarray,
        ws_s: np.ndarray, wei_s: np.ndarray, wmol_s: np.ndarray,
        net_s: float = 0.0, trq_s: float = 0.0,
    ) -> float:
        n_steps = len(lr_s)
        p_a = np.ascontiguousarray(perm[: n_steps * batch_size], dtype=np.int32)
        lr_a = np.ascontiguousarray(lr_s, dtype=np.float64)
        we_a = np.ascontiguousarray(we_s, dtype=np.float64)
        wf_a = np.ascontiguousarray(wf_s, dtype=np.float64)
        ws_a = np.ascontiguousarray(ws_s, dtype=np.float64)
        wei_a = np.ascontiguousarray(wei_s, dtype=np.float64)
        wm_a = np.ascontiguousarray(wmol_s, dtype=np.float64)
        loss_out = ctypes.c_double(0.0)
        rc = getattr(self.lib, f"lrsk_ft_run_epoch_{self.sfx}")(
            self.handle, p_a.ctypes.data, n_steps, batch_size,
            lr_a.ctypes.data, we_a.ctypes.data, wf_a.ctypes.data, ws_a.ctypes.data, wei_a.ctypes.data, wm_a.ctypes.data,
            float(net_s), float(trq_s), ctypes.byref(loss_out),
        )
        if rc:
            raise RuntimeError(f"run_epoch failed rc={rc}")
        return float(loss_out.value)

    def run_epoch_weighted(
        self, perm: np.ndarray, dyn_gw: np.ndarray, batch_size: int,
        lr_s: np.ndarray, we_s: np.ndarray, wf_s: np.ndarray, update_ferr: bool = True,
    ) -> tuple[float, float]:
        n_steps = len(lr_s)
        p_a = np.ascontiguousarray(perm[: n_steps * batch_size], dtype=np.int32)
        gw_a = np.ascontiguousarray(dyn_gw[: n_steps * batch_size], dtype=np.float64)
        lr_a = np.ascontiguousarray(lr_s, dtype=np.float64)
        we_a = np.ascontiguousarray(we_s, dtype=np.float64)
        wf_a = np.ascontiguousarray(wf_s, dtype=np.float64)
        loss_out = ctypes.c_double(0.0)
        lf_out = ctypes.c_double(0.0)
        rc = getattr(self.lib, f"lrsk_ft_run_epoch_weighted_{self.sfx}")(
            self.handle, p_a.ctypes.data, gw_a.ctypes.data, n_steps, batch_size,
            lr_a.ctypes.data, we_a.ctypes.data, wf_a.ctypes.data,
            int(update_ferr), ctypes.byref(loss_out), ctypes.byref(lf_out),
        )
        if rc:
            raise RuntimeError(f"run_epoch_weighted failed rc={rc}")
        return float(loss_out.value), float(lf_out.value)

    def get_frame_err(self, n_graphs: int) -> np.ndarray:
        buf = np.empty(n_graphs, dtype=np.float32)
        rc = getattr(self.lib, f"lrsk_ft_get_frame_err_{self.sfx}")(self.handle, buf.ctypes.data)
        if rc:
            raise RuntimeError(f"get_frame_err failed rc={rc}")
        return buf


# =============================================================================
# Exact FP64 Gradcheck Suite Across Diverse Structures & Model Configs!
# =============================================================================
def run_gradcheck():
    from ase import Atoms
    print("=== PURE-CUDA LRSK TRAINER: EXACT FP64 GRADCHECK SUITE ===", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    def _make_frames(zs, n_frames, close_contact=False, with_mol=False, seed=42):
        rng = np.random.default_rng(seed)
        frames = []
        for fi in range(n_frames):
            if with_mol:
                # 6 water-like 3-atom molecules in a 9.5 A box
                pos, nums, mids = [], [], []
                for m in range(6):
                    cx, cy, cz = (m % 2) * 4.5 + 1.5, ((m // 2) % 2) * 4.5 + 1.5, (m // 4) * 4.5 + 1.5
                    c = np.array([cx, cy, cz]) + rng.uniform(-0.3, 0.3, 3)
                    pos.extend([c, c + [0.96, 0.0, 0.0], c + [-0.24, 0.93, 0.0]])
                    nums.extend([zs[-1], zs[0], zs[0]])
                    mids.extend([m, m, m])
                at = Atoms(numbers=nums, positions=pos, cell=np.diag([9.5, 9.5, 9.5]), pbc=True)
                at.arrays["mol_id"] = np.array(mids, dtype=np.int64)
            else:
                n_at = 10 + 2 * fi
                cell = np.diag([6.8, 7.1, 6.9]) + rng.uniform(-0.15, 0.15, (3, 3))
                pos = rng.uniform(0.0, 1.0, (n_at, 3)) @ cell
                if close_contact and fi == 0:
                    pos[1] = pos[0] + np.array([0.82, 0.11, -0.07])
                nums = [zs[i % len(zs)] for i in range(n_at)]
                at = Atoms(numbers=nums, positions=pos, cell=cell, pbc=True)
            at.info["REF_energy"] = float(rng.normal(0.0, 5.0))
            at.arrays["REF_forces"] = rng.normal(0.0, 1.5, (len(at), 3))
            st = rng.normal(0.0, 0.2, (3, 3))
            at.info["REF_stress"] = 0.5 * (st + st.T)
            at.arrays["REF_ei"] = rng.normal(0.0, 0.3, len(at))
            frames.append(at)
        return frames

    cases = [
        # (tag, zs, kwargs, close_contact, with_mol, with_stress, with_ei, with_spw, with_gw)
        ("crystal_zbl_ro1_l3_film", [8, 22, 38], dict(
            r_max=4.6, r_max2=4.6, lmax=3, lfeat=1, taper=[8, 6, 4, 4], n_base=6, n_base2=4,
            c0=4, c1=4, c2=0, readout_width=16, readout_depth=1, radial_warp=0.0,
            use_zbl_switch=True, film_radials=True, sym=True, pol_field=True, pol_gram=True,
            sw_mp=True, mp_cap=4.0, ef_cap=4.0, kappa=6.0, a0=0.05, field_mlp_width=12, energy_scale=1.3
        ), True, False, True, True, True, True),
        ("mol_splitr2_c442_ro2_nofilm", [1, 6, 7, 8], dict(
            r_max=5.0, r_max2=4.2, lmax=3, lfeat=2, taper=[6, 6, 4, 4], n_base=8, n_base2=4,
            c0=4, c1=4, c2=2, readout_width=16, readout_depth=2, radial_warp=1.5,
            use_zbl_switch=True, film_radials=False, sym=True, pol_field=True, pol_gram=False,
            sw_mp=True, mp_cap=4.0, ef_cap=0.0, kappa=4.0, a0=0.6, field_mlp_width=12, energy_scale=1.1
        ), False, False, False, False, False, True),
        ("water_mol_com_trq_nopol", [1, 8], dict(
            r_max=5.2, r_max2=4.4, lmax=2, lfeat=2, taper=[8, 6, 4], n_base=8, n_base2=4,
            c0=4, c1=4, c2=0, readout_width=16, readout_depth=2, radial_warp=2.0,
            use_zbl_switch=False, film_radials=True, sym=True, pol_field=True, pol_gram=True,
            sw_mp=False, mp_cap=0.0, ef_cap=0.0, kappa=16.0, a0=0.05, field_mlp_width=12, energy_scale=0.9
        ), False, True, True, True, False, False),
        ("rev_split_r42_58_water", [1, 8], dict(
            r_max=4.2, r_max2=5.8, lmax=2, lfeat=2, taper=[8, 6, 4], n_base=8, n_base2=4,
            c0=4, c1=4, c2=0, readout_width=16, readout_depth=2, radial_warp=2.0,
            use_zbl_switch=True, film_radials=True, sym=True, pol_field=True, pol_gram=True,
            sw_mp=True, mp_cap=4.0, ef_cap=4.0, kappa=16.0, a0=0.05, field_mlp_width=12, energy_scale=0.95
        ), False, True, True, True, False, False),
    ]

    for tag, zs, kw, cc, wmol, wst, wei, wspw, wgw in cases:
        torch.manual_seed(123)
        torch.set_default_dtype(torch.float64)
        frames = _make_frames(zs, 3, close_contact=cc, with_mol=wmol, seed=101 + len(tag))
        zt = AtomicNumberTable(zs)
        e0_map = {z: 0.0 for z in zs}
        r_graph = max(kw["r_max"], kw["r_max2"])
        items, mol_list = prebuild_data(frames, zt, r_graph, e0_map, mol_key=("mol_id" if wmol else None), device=device)
        for ad, at in zip(items, frames):
            ad.ei_ref = torch.tensor(at.arrays["REF_ei"], dtype=torch.float64, device=device)
        model = FastLRSKStudent(zs=zs, **kw).double().to(device)
        model.calibrate_buffers(items, bs=3)
        # Non-zero biases so every bias grad is exercised
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.abs().max() == 0:
                    p.normal_(0.0, 0.15)
        model.e0.zero_()

        sp_w = np.linspace(0.7, 1.4, len(zs)) if wspw else None
        gw = np.array([1.0, 0.35, 1.7], dtype=np.float64) if wgw else np.ones(3, dtype=np.float64)
        w_lr_reg, rad_sm, wd_ro = 2e-4, 1e-4, 5e-4
        we, wf, ws_v, wei_v = 1.5, 100.0, (30.0 if wst else 0.0), (1.2 if wei else 0.0)
        wmol_v, net_s, trq_s = (18.0 if wmol else 0.0), 1.7, 2.3

        # 1) PyTorch FP64 reference
        model.zero_grad(set_to_none=True)
        b_pyg = Batch.from_data_list(items).to(device).to_dict()
        for k in ("positions", "shifts", "cell", "node_attrs", "energy", "forces", "stress"):
            if k in b_pyg and b_pyg[k] is not None:
                b_pyg[k] = b_pyg[k].double()
        out = model(b_pyg, training=True, compute_force=True, compute_stress=wst)
        na_f = torch.tensor([len(f) for f in frames], dtype=torch.float64, device=device)
        gw_t = torch.tensor(gw, dtype=torch.float64, device=device)
        gw_at = gw_t[b_pyg["batch"]]
        le = (gw_t * ((out["energy"] - b_pyg["energy"].view(3)) / na_f) ** 2).mean()
        df = out["forces"] - b_pyg["forces"]
        if sp_w is not None:
            sp_t = torch.tensor(sp_w, dtype=torch.float64, device=device)
            z_i = b_pyg["node_attrs"].argmax(-1)
            lf = ((df ** 2) * (gw_at * sp_t[z_i]).unsqueeze(-1)).mean()
        else:
            lf = ((df ** 2) * gw_at.unsqueeze(-1)).mean()
        loss = we * le + wf * lf
        if wst:
            ls = (gw_t.view(3, 1, 1) * (out["stress"] - b_pyg["stress"].view(3, 3, 3)) ** 2).mean()
            loss = loss + ws_v * ls
        if wei:
            ne = out["node_energy"]
            ei_r = torch.cat([ad.ei_ref for ad in items], dim=0)
            gs = b_pyg["batch"] * model.S + b_pyg["node_attrs"].argmax(-1)
            sums = torch.zeros(3 * model.S, dtype=torch.float64, device=device)
            cnts = torch.zeros(3 * model.S, dtype=torch.float64, device=device)
            sums.index_add_(0, gs, ne); cnts.index_add_(0, gs, torch.ones_like(ne))
            ne_c = ne - (sums / cnts.clamp_min(1.0))[gs]
            loss = loss + wei_v * (gw_at * (ne_c - ei_r) ** 2).mean()
        if wmol:
            off = 0
            lmol = torch.zeros((), dtype=torch.float64, device=device)
            for gi, mg in enumerate(mol_list):
                n_a = len(frames[gi])
                df_g = df[off : off + n_a]; off += n_a
                mid, nm, arms = mg
                arms64 = arms.double()
                dfn = torch.zeros(nm, 3, dtype=torch.float64, device=device); dfn.index_add_(0, mid, df_g)
                dft = torch.zeros(nm, 3, dtype=torch.float64, device=device); dft.index_add_(0, mid, torch.linalg.cross(arms64, df_g, dim=-1))
                lmol = lmol + gw_t[gi] * (net_s * (dfn ** 2).mean() + trq_s * (dft ** 2).mean())
            loss = loss + wmol_v * (lmol / 3.0)

        k4 = (torch.arange(model.n_base2, dtype=torch.float64, device=device) ** 4)[None, None, :]
        k2 = (torch.arange(model.n_base, dtype=torch.float64, device=device) ** 2)[None, :]
        loss_reg = loss + w_lr_reg * ((model.W_lr_rad ** 2) * k4).mean()
        if model.film_radials:
            loss_reg = loss_reg + rad_sm * (((model.C1_R ** 2) * k2).mean() + ((model.C1_B ** 2) * k2).mean())
        else:
            loss_reg = loss_reg + rad_sm * ((model.C1 ** 2) * k2[None, :, :]).mean()
        ro_p = [p for n, p in model.named_parameters() if any(k in n for k in ("readout_layers", "pol_W0", "W_q", "W_g", "u_Z", "pol_u"))]
        loss_reg = loss_reg + wd_ro * sum((p ** 2).mean() for p in ro_p)
        loss_reg.backward()
        ref_grads = {n: p.grad.detach().cpu().numpy().copy() for n, p in canonical_param_list(model)}

        # 2) Pure-CUDA FP64 trainer
        ft = CudaFullLRSKTrainer(model, dtype="f64", soap_max_dim=128)
        ft.set_optimizer(0.98, 0.98, 8, 2e-4, 1.0, 0.85, 0.96, 0.96, 1.0, w_lr_reg, rad_sm, wd_ro, sp_w=sp_w)
        ft.upload_dataset(items, mol_list=mol_list if wmol else None, graph_weights=gw, has_stress=wst, device=device)
        g_ids = np.array([0, 1, 2], dtype=np.int32)
        tot_N = sum(len(f) for f in frames)
        E_cu = np.empty(3, dtype=np.float64)
        F_cu = np.empty((tot_N, 3), dtype=np.float64)
        V_cu = np.empty((3, 3, 3), dtype=np.float64)
        L_cu = ctypes.c_double(0.0)
        rc = ft.lib.lrsk_ft_gradcheck_f64(
            ft.handle, g_ids.ctypes.data, 3, we, wf, ws_v, wei_v, wmol_v, net_s, trq_s, 1,
            E_cu.ctypes.data, F_cu.ctypes.data, V_cu.ctypes.data, ctypes.byref(L_cu),
        )
        assert rc == 0, f"gradcheck rc={rc}"
        cu_grads = ft.download_grads()
        ft.close()

        dE = float(np.max(np.abs(E_cu - out["energy"].detach().cpu().numpy())))
        dF = float(np.max(np.abs(F_cu - out["forces"].detach().cpu().numpy())))
        dS = 0.0
        if wst:
            vols = np.array([abs(np.linalg.det(f.cell.array)) for f in frames]).reshape(3, 1, 1)
            dS = float(np.max(np.abs(V_cu / vols - out["stress"].detach().cpu().numpy())))
        dL = abs(L_cu.value - float(loss.detach().cpu().item()))
        max_rel = 0.0
        worst_name = ""
        for name, rg in ref_grads.items():
            cg = cu_grads[name]
            diff = float(np.max(np.abs(cg - rg)))
            denom = max(float(np.max(np.abs(rg))), 1.0)
            rel = diff / denom
            if rel > max_rel:
                max_rel = rel; worst_name = name
        status = "PASS" if (dE < 1e-11 and dF < 1e-11 and dS < 1e-11 and max_rel < 1e-10) else "FAIL"
        print(f"  [{status}] {tag:30s} | dE={dE:.2e} dF={dF:.2e} dS={dS:.2e} dL={dL:.2e} | max_grad_rel={max_rel:.2e} ({worst_name})", flush=True)
        if status != "PASS":
            for name, rg in ref_grads.items():
                cg = cu_grads[name]
                diff = float(np.max(np.abs(cg - rg)))
                rel = diff / max(float(np.max(np.abs(rg))), 1.0)
                if rel > 1e-9:
                    print(f"      MISMATCH {name:26s}: max_abs={diff:.3e} rel={rel:.3e}", flush=True)
            raise AssertionError(f"Gradcheck FAILED on {tag}")

    # 4th case: n3tbu per-graph force MSE + dynamic dyn_gw gradcheck (if v2 API present)
    lib_check = load_ft_lib()
    if hasattr(lib_check, "lrsk_ft_gradcheck_weighted_f64"):
        torch.manual_seed(777)
        torch.set_default_dtype(torch.float64)
        zs = [1, 6, 7, 8]
        frames = _make_frames(zs, 3, close_contact=True, with_mol=False, seed=303)
        zt = AtomicNumberTable(zs)
        items, _ = prebuild_data(frames, zt, 4.8, {z: 0.0 for z in zs}, device=device)
        model = FastLRSKStudent(
            zs=zs, r_max=4.8, r_max2=4.0, lmax=3, lfeat=2, taper=[6, 6, 4, 4], n_base=8, n_base2=4,
            c0=4, c1=4, c2=0, readout_width=16, readout_depth=1, radial_warp=1.5,
            use_zbl_switch=True, film_radials=True, sym=True, pol_field=True, pol_gram=True,
            sw_mp=True, mp_cap=4.0, ef_cap=4.0, kappa=4.0, a0=0.6, field_mlp_width=12, energy_scale=1.2
        ).double().to(device)
        model.calibrate_buffers(items, bs=3)
        with torch.no_grad():
            for _, p in model.named_parameters():
                if p.abs().max() == 0:
                    p.normal_(0.0, 0.12)
        model.e0.zero_()
        dyn_gw = np.array([1.35, 0.72, 0.93], dtype=np.float64)
        we, wf = 2.0, 100.0
        w_lr_reg, rad_sm, wd_ro = 5e-4, 1e-3, 1.5e-2
        model.zero_grad(set_to_none=True)
        b_pyg = Batch.from_data_list(items).to(device).to_dict()
        for k in ("positions", "shifts", "cell", "node_attrs", "energy", "forces"):
            if k in b_pyg and b_pyg[k] is not None:
                b_pyg[k] = b_pyg[k].double()
        out = model(b_pyg, training=True, compute_force=True, compute_stress=False)
        na_f = torch.tensor([len(f) for f in frames], dtype=torch.float64, device=device)
        wg_t = torch.tensor(dyn_gw, dtype=torch.float64, device=device)
        e_sq = ((out["energy"] - b_pyg["energy"].view(3)) / na_f) ** 2
        f_sq_a = ((out["forces"] - b_pyg["forces"]) ** 2).mean(dim=-1)
        f_sq_g = torch.zeros(3, dtype=torch.float64, device=device)
        f_sq_g.index_add_(0, b_pyg["batch"], f_sq_a)
        f_sq_g = f_sq_g / na_f
        loss = we * (e_sq * wg_t).mean() + wf * (f_sq_g * wg_t).mean()
        k4 = (torch.arange(model.n_base2, dtype=torch.float64, device=device) ** 4)[None, None, :]
        k2 = (torch.arange(model.n_base, dtype=torch.float64, device=device) ** 2)[None, :]
        loss_reg = loss + w_lr_reg * ((model.W_lr_rad ** 2) * k4).mean()
        loss_reg = loss_reg + rad_sm * (((model.C1_R ** 2) * k2).mean() + ((model.C1_B ** 2) * k2).mean())
        ro_p = [p for n, p in model.named_parameters() if any(k in n for k in ("readout_layers", "pol_W0", "W_q", "W_g", "u_Z", "pol_u"))]
        loss_reg = loss_reg + wd_ro * sum((p ** 2).mean() for p in ro_p)
        loss_reg.backward()
        ref_grads = {n: p.grad.detach().cpu().numpy().copy() for n, p in canonical_param_list(model)}

        ft = CudaFullLRSKTrainer(model, dtype="f64", soap_max_dim=128)
        ft.set_optimizer(0.98, 0.98, 8, 2e-3, 1.0, 0.85, 0.96, 0.96, 1.0, w_lr_reg, rad_sm, wd_ro, sp_w=None)
        ft.upload_dataset(items, mol_list=None, graph_weights=None, has_stress=False, device=device)
        g_ids = np.array([0, 1, 2], dtype=np.int32)
        E_cu = np.empty(3, dtype=np.float64)
        F_cu = np.empty((sum(len(f) for f in frames), 3), dtype=np.float64)
        L_cu = ctypes.c_double(0.0)
        rc = ft.lib.lrsk_ft_gradcheck_weighted_f64(
            ft.handle, g_ids.ctypes.data, dyn_gw.ctypes.data, 3, we, wf, 1,
            E_cu.ctypes.data, F_cu.ctypes.data, ctypes.byref(L_cu),
        )
        assert rc == 0, f"weighted gradcheck rc={rc}"
        cu_grads = ft.download_grads()
        ft.close()
        dE = float(np.max(np.abs(E_cu - out["energy"].detach().cpu().numpy())))
        dF = float(np.max(np.abs(F_cu - out["forces"].detach().cpu().numpy())))
        dL = abs(L_cu.value - float(loss.detach().cpu().item()))
        max_rel, worst_name = 0.0, ""
        for name, rg in ref_grads.items():
            rel = float(np.max(np.abs(cu_grads[name] - rg))) / max(float(np.max(np.abs(rg))), 1.0)
            if rel > max_rel:
                max_rel = rel; worst_name = name
        dL_rel = dL / max(abs(float(loss.detach().cpu().item())), 1.0)
        status = "PASS" if (dE < 1e-11 and dF < 1e-11 and dL_rel < 1e-11 and max_rel < 1e-10) else "FAIL"
        print(f"  [{status}] {'n3tbu_pergraph_dyn_gw':30s} | dE={dE:.2e} dF={dF:.2e} dS=0.00e+00 dL={dL:.2e} | max_grad_rel={max_rel:.2e} ({worst_name})", flush=True)
        if status != "PASS":
            raise AssertionError("Gradcheck FAILED on n3tbu_pergraph_dyn_gw")
    print("ALL FP64 GRADCHECKS PASSED (< 1e-10)!", flush=True)


# =============================================================================
# Universal Pure-CUDA Training Driver CLI (for ANY dataset!)
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradcheck", action="store_true", default=False)
    ap.add_argument("--name", default="lrsk_full_cu")
    ap.add_argument("--train", type=Path, default=None)
    ap.add_argument("--valid", type=Path, default=None)
    ap.add_argument("--test", type=Path, default=None)
    ap.add_argument("--extra-eval", nargs="*", default=[])
    ap.add_argument("--pair-mlp-width", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--r-max", type=float, default=5.4)
    ap.add_argument("--r-max2", type=float, default=None)
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--lfeat", type=int, default=2)
    ap.add_argument("--taper", type=str, default="16,16,16")
    ap.add_argument("--n-base", type=int, default=8)
    ap.add_argument("--n-base2", type=int, default=4)
    ap.add_argument("--c0", type=int, default=4)
    ap.add_argument("--c1", type=int, default=4)
    ap.add_argument("--c2", type=int, default=0)
    ap.add_argument("--readout-width", type=int, default=64)
    ap.add_argument("--readout-depth", type=int, default=1)
    ap.add_argument("--radial-warp", type=float, default=0.0)
    ap.add_argument("--sym", action="store_true", default=True)
    ap.add_argument("--pol-field", action="store_true", default=False)
    ap.add_argument("--pol-gram", action="store_true", default=False)
    ap.add_argument("--sw-mp", action="store_true", default=False)
    ap.add_argument("--mp-cap", type=float, default=0.0)
    ap.add_argument("--ef-cap", type=float, default=0.0)
    ap.add_argument("--kappa", type=float, default=16.0)
    ap.add_argument("--a0", type=float, default=0.05)
    ap.add_argument("--field-mlp-width", type=int, default=32)
    ap.add_argument("--no-film-radials", dest="film_radials", action="store_false", default=True)
    ap.add_argument("--no-zbl-switch", dest="use_zbl_switch", action="store_false", default=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr-peak", type=float, default=0.008)
    ap.add_argument("--lr-final", type=float, default=1e-4)
    ap.add_argument("--w-stress", type=float, default=30.0)
    ap.add_argument("--w-ei", type=float, default=1.0)
    ap.add_argument("--mol-key", default=None)
    ap.add_argument("--mol-loss", type=float, default=0.0)
    ap.add_argument("--species-rel-loss", type=float, default=0.0)
    ap.add_argument("--regime-weight", action="append", default=[])
    ap.add_argument("--w-lr-reg", type=float, default=2e-4)
    ap.add_argument("--rad-smooth", type=float, default=0.0)
    ap.add_argument("--wd-ro", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=2e-4)
    ap.add_argument("--soap-every", type=int, default=0)
    ap.add_argument("--soap-max-dim", type=int, default=128)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sto-stage0-phonons", type=Path, default=None)
    args = ap.parse_args()

    if args.gradcheck:
        run_gradcheck()
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda")

    tr_frames = list(iread(args.train, ":", format="extxyz"))
    va_frames = list(iread(args.valid, ":", format="extxyz"))
    te_frames = list(iread(args.test, ":", format="extxyz"))
    zs = sorted({int(z) for a in tr_frames for z in a.numbers})
    z_table = AtomicNumberTable(zs)
    phys_e0 = fit_min_norm_e0(tr_frames, zs)
    e0_map = {z: float(e) for z, e in zip(zs, phys_e0)}
    f_all = np.concatenate([a.arrays["REF_forces"].ravel() for a in tr_frames])
    rms_f = float(np.sqrt(np.mean(f_all ** 2)))

    sp_w_arr = None
    if args.species_rel_loss > 0.0:
        sp_f_sq = {z: [] for z in zs}
        for a in tr_frames:
            ff = a.arrays["REF_forces"]
            for z in zs:
                m = (a.numbers == z)
                if np.any(m):
                    sp_f_sq[z].append((ff[m] ** 2).ravel())
        sp_rms = {z: float(np.sqrt(np.mean(np.concatenate(sp_f_sq[z])))) for z in zs}
        raw_w = np.array([(rms_f / max(sp_rms[z], 1e-6)) ** args.species_rel_loss for z in zs], dtype=np.float64)
        raw_w /= np.mean(raw_w)
        sp_w_arr = raw_w
        print(f"  Species-balanced force loss (alpha={args.species_rel_loss}): w={ {z: round(float(w), 3) for z, w in zip(zs, raw_w)} }", flush=True)

    reg_w_map = {}
    for spec in args.regime_weight:
        rk, rv = spec.split("=", 1)
        reg_w_map[rk.strip()] = float(rv)
    graph_w = np.ones(len(tr_frames), dtype=np.float64)
    if reg_w_map:
        n_nonunit = 0
        for gi, a in enumerate(tr_frames):
            reg = str(a.info.get("regime", ""))
            if reg in reg_w_map:
                graph_w[gi] = reg_w_map[reg]
                n_nonunit += 1
        print(f"  Per-regime loss weights: {reg_w_map} (non-unit frames: {n_nonunit}/{len(tr_frames)})", flush=True)

    mol_scales = None
    if args.mol_key and args.mol_loss > 0.0:
        rs = ReferenceScales()
        for a in tr_frames:
            rs.observe(a, grouped_geometry(a, args.mol_key))
        ms = rs.result()
        mol_scales = (ms["net_scale"], ms["torque_scale"], args.mol_loss)

    r_graph = max(args.r_max, args.r_max2 if args.r_max2 is not None else args.r_max)
    tr_items, tr_mol = prebuild_data(tr_frames, z_table, r_graph, e0_map, mol_key=args.mol_key, device=device)
    taper = [int(x) for x in args.taper.split(",")]
    avg_nb = float(np.mean([ad.edge_index.shape[1] / ad.positions.shape[0] for ad in tr_items]))

    print(f"\n============================================================", flush=True)
    print(f">>> PURE-CUDA LRSK TRAINER: {args.name} (bs={args.batch_size}, 100% GPU C++)", flush=True)
    print(f"    r_max={args.r_max} r_max2={args.r_max2 or args.r_max} lmax={args.lmax}/lfeat={args.lfeat} taper={taper} "
          f"nb=({args.n_base},{args.n_base2}) warp={args.radial_warp} c=({args.c0},{args.c1},{args.c2}) "
          f"pol={args.pol_field} gram={args.pol_gram} sw_mp={args.sw_mp} cap=({args.mp_cap},{args.ef_cap}) "
          f"ro={args.readout_depth}x{args.readout_width} wd_ro={args.wd_ro} rad_sm={args.rad_smooth} avg_nb={avg_nb:.2f}", flush=True)

    model = FastLRSKStudent(
        zs=zs, r_max=args.r_max, r_max2=args.r_max2, lmax=args.lmax, lfeat=args.lfeat,
        taper=taper, n_base=args.n_base, n_base2=args.n_base2,
        c0=args.c0, c1=args.c1, c2=args.c2,
        readout_width=args.readout_width, readout_depth=args.readout_depth,
        radial_warp=args.radial_warp, use_zbl_switch=args.use_zbl_switch,
        kappa=args.kappa, a0=args.a0, energy_scale=rms_f, film_radials=args.film_radials,
        pol_field=args.pol_field, field_mlp_width=args.field_mlp_width, sym=True,
        pol_gram=args.pol_gram, sw_mp=args.sw_mp, mp_cap=args.mp_cap, ef_cap=args.ef_cap,
        pair_mlp_width=0,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    total trainable params: {n_params:,}", flush=True)

    if args.init_from is not None and args.init_from.exists():
        ck0 = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd0 = ck0["state_dict"]
        cur = model.state_dict()
        loaded = 0
        for k, v in sd0.items():
            if k in cur and cur[k].shape == v.shape and k != "e0":
                cur[k].copy_(v.to(cur[k].dtype)); loaded += 1
        print(f"  Warm-started {loaded}/{len(cur)} tensors from {args.init_from.name}", flush=True)
    else:
        model.calibrate_buffers(tr_items, bs=max(4, min(args.batch_size, 8)))

    bs = args.batch_size
    b_soap = max(0.95, 0.995 ** bs)
    soap_ev = args.soap_every if args.soap_every > 0 else max(4, min(16, 12 // max(1, bs // 2)))
    adam_b1 = max(0.85, 0.90 ** math.sqrt(bs))
    adam_b2 = max(0.95, 0.99 ** bs)
    d_ema_max = max(0.94, 0.99 ** bs)

    ft = CudaFullLRSKTrainer(model, dtype="f32", soap_max_dim=args.soap_max_dim)
    ft.set_optimizer(
        b_soap, b_soap, soap_ev, args.weight_decay, 1.0,
        adam_b1, adam_b2, d_ema_max, 1.0,
        args.w_lr_reg, args.rad_smooth, args.wd_ro, sp_w=sp_w_arr,
    )
    compute_stress = args.w_stress > 0.0
    ft.upload_dataset(tr_items, mol_list=tr_mol if mol_scales else None, graph_weights=graph_w, has_stress=compute_stress, device=device)

    N_tr = len(tr_items)
    spe = max(1, N_tr // bs)
    tail_ep = max(float(args.epochs - 10), math.floor(args.epochs * 0.85))
    rng = np.random.default_rng(args.seed)

    def lr_at(t):
        pts = [(0.0, args.lr_peak * 0.05), (2.0, args.lr_peak), (tail_ep, args.lr_peak * 0.12), (float(args.epochs), args.lr_final)]
        if t <= pts[0][0]:
            return pts[0][1]
        for i in range(len(pts) - 1):
            if t <= pts[i + 1][0]:
                e0, v0 = pts[i]; e1, v1 = pts[i + 1]
                return v0 + (v1 - v0) * (t - e0) / max(e1 - e0, 1e-12)
        return pts[-1][1]

    t_tr0 = time.perf_counter()
    for ep in range(args.epochs):
        perm = rng.permutation(N_tr).astype(np.int32)
        lr_s = np.empty(spe, dtype=np.float64)
        we_s = np.empty(spe, dtype=np.float64)
        wf_s = np.empty(spe, dtype=np.float64)
        ws_s = np.empty(spe, dtype=np.float64)
        wei_s = np.empty(spe, dtype=np.float64)
        wm_s = np.empty(spe, dtype=np.float64)
        for st in range(spe):
            t = ep + st / spe
            lr_s[st] = lr_at(t)
            if t < tail_ep:
                we_s[st], wf_s[st], ws_s[st] = 1.0, 100.0, args.w_stress
                wm_s[st] = mol_scales[2] if mol_scales else 0.0
            else:
                we_s[st], wf_s[st], ws_s[st] = 4000.0, 100.0, (max(args.w_stress, 30.0) if compute_stress else 0.0)
                wm_s[st] = (mol_scales[2] * 0.5) if mol_scales else 0.0
            if args.w_ei > 0.0:
                u = t / max(1.0, float(args.epochs))
                wei_s[st] = args.w_ei if u <= 0.5 else (args.w_ei * (1.0 - (u - 0.5) / 0.35) if u < 0.85 else 0.0)
            else:
                wei_s[st] = 0.0
        mean_loss = ft.run_epoch(
            perm, bs, lr_s, we_s, wf_s, ws_s, wei_s, wm_s,
            net_s=(mol_scales[0] if mol_scales else 0.0),
            trq_s=(mol_scales[1] if mol_scales else 0.0),
        )
        if (ep + 1) in (1, 10, 20, 40, 60, 80, 100, 140, 180, 200, 240, 300, 450) or (ep + 1) == args.epochs:
            print(f"    ep {ep+1:3d}/{args.epochs} | elapsed={time.perf_counter()-t_tr0:5.1f}s | loss={mean_loss:.4f}", flush=True)

    train_s = time.perf_counter() - t_tr0
    ft.download_params_to_model(from_ema=True)
    ft.close()
    torch.cuda.empty_cache()

    # Post-EMA E0 calibration (exclude non-unit regime weights if any)
    with torch.no_grad():
        model.e0.zero_()
        model.eval()
        cal_idx = [i for i in range(N_tr) if graph_w[i] >= 0.99] or list(range(N_tr))
        A_e0 = np.zeros((len(cal_idx), len(zs)), dtype=np.float64)
        res_e0 = np.zeros(len(cal_idx), dtype=np.float64)
        z2i = {z: i for i, z in enumerate(zs)}
        for row, k in enumerate(cal_idx):
            for z in tr_frames[k].numbers:
                A_e0[row, z2i[int(z)]] += 1.0
        for s0 in range(0, len(cal_idx), 4):
            sub_idx = cal_idx[s0 : s0 + 4]
            sub = [tr_items[k] for k in sub_idx]
            b = Batch.from_data_list(sub).to(device)
            out = model(b.to_dict(), training=False, compute_force=False, compute_stress=False)
            res_e0[s0 : s0 + len(sub)] = b.energy.view(len(sub)).double().cpu().numpy() - out["energy"].view(len(sub)).double().cpu().numpy()
        delta_e0, *_ = np.linalg.lstsq(A_e0, res_e0, rcond=None)
        model.e0.copy_(torch.tensor(phys_e0 + delta_e0, dtype=torch.float64, device=device))
        print(f"  Post-EMA E0 calibration: delta_e0={np.round(delta_e0 * 1000, 3).tolist()} meV/at (train={train_s:.1f}s)", flush=True)

    eval_list = [("train_sub", args.train, None), ("valid", args.valid, None), ("teacher_test", args.test, None)]
    for item in args.extra_eval:
        k, v = item.split("=", 1)
        reg = None
        if ":" in v:
            v, reg = v.rsplit(":", 1)
        eval_list.append((k, Path(v), reg))

    results = {"name": args.name, "params": n_params, "train_s": train_s, "scores": {}}
    for dname, dpath, reg in eval_list:
        frames = list(iread(dpath, ":", format="extxyz"))
        if dname == "train_sub" and len(frames) > 40:
            frames = frames[:40]
        rep = eval_frames_pt(model, frames, z_table, r_graph, device, mol_key=args.mol_key, regime=reg)
        fo = rep["forces"]["overall"]
        sp = {k: round(v["relative_rmse"] * 100, 2) for k, v in rep["forces"]["per_species"].items()}
        an = rep["forces"]["atom_error_norm"]
        mol_s = ""
        mol_d = None
        if rep["molecular"]:
            nf = rep["molecular"]["net_force"]; tq = rep["molecular"]["torque"]
            mol_s = f" COM={nf['rmse']*1000:.1f}({nf['relative_rmse']*100:.2f}%) TRQ={tq['rmse']*1000:.1f}({tq['relative_rmse']*100:.2f}%)"
            mol_d = {"com_pct": nf["relative_rmse"] * 100, "trq_pct": tq["relative_rmse"] * 100}
        print(f"  [{dname:18s}] F={fo['rmse']*1000:5.2f}meV/A ({fo['relative_rmse']*100:5.2f}%) sp%={sp}{mol_s} "
              f"p99.9={an['p99_9']*1000:5.1f} E={rep['energy']['rmse_meV_per_atom']:.3f}meV/at "
              f"S={rep['stress']['rmse']*1000:.3f}meV/A3 pass={rep['label_metrics_passed']}", flush=True)
        results["scores"][dname] = {
            "f_mev": fo["rmse"] * 1000, "f_pct": fo["relative_rmse"] * 100, "sp_pct": sp,
            "p999_mev": an["p99_9"] * 1000, "e_mev": rep["energy"]["rmse_meV_per_atom"],
            "s_mev": rep["stress"]["rmse"] * 1000, "mol": mol_d, "pass": rep["label_metrics_passed"],
        }

    ckpt_path = args.out_dir / f"{args.name}.pt"
    save_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    save_args["lrsk"] = True
    save_args["pair_mlp_width"] = 0
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "energy_scale": float(rms_f), "args": save_args}, ckpt_path)
    if args.sto_stage0_phonons is not None:
        results["sto_phonons"] = eval_sto_phonons_pt(model, args.sto_stage0_phonons, z_table, r_graph, device)
        print("  STO PHONONS:", results["sto_phonons"], flush=True)
    (args.out_dir / f"{args.name}_summary.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"  Saved -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
