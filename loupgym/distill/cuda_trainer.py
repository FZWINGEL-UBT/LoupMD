"""Self-contained pure-CUDA FastLR-SK training wrapper matching ``src/train/lrsk_full_train.cu`` ABI."""
from __future__ import annotations

import ctypes
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .fast_lrsk import (
    KE_EV_A,
    ZBL_C,
    ZBL_D,
    FastLRSKStudent,
    batch_graph_items,
    build_graph_items,
    fit_min_norm_e0,
)

_LIB_CACHE: dict[str, ctypes.CDLL] = {}
_REPO_ROOT = Path(__file__).resolve().parents[2]


from .._lib import find_native_lib


def _ptr(t: torch.Tensor | None) -> int:
    if t is None or t.numel() == 0:
        return 0
    return int(t.data_ptr())


def find_or_build_lrsk_train_lib(auto_build: bool = True) -> Path:
    """Locate ``libloup_lrsk_full_train_v3.so``, building ``make lrsk-train-lib`` if missing."""
    return find_native_lib(
        "libloup_lrsk_full_train_v3.so",
        env_var="LOUP_LRSK_FT_LIB",
        make_target="lrsk-train-lib",
        auto_build=auto_build,
        extra_candidates=(Path(__file__).resolve().parent / "libloup_lrsk_full_train_v3.so",),
    )


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


def load_ft_lib() -> ctypes.CDLL:
    p = str(find_or_build_lrsk_train_lib())
    if p in _LIB_CACHE:
        return _LIB_CACHE[p]
    lib = ctypes.CDLL(p)

    vp = ctypes.c_void_p
    ci = ctypes.c_int
    cl = ctypes.c_long
    cd = ctypes.c_double
    for sfx in ("f32", "f64"):
        getattr(lib, f"l1_precompute_G1_{sfx}").argtypes = [vp, ci, ci, ci, ci, cd, cd, vp, vp, vp, vp, vp]
        getattr(lib, f"l1_precompute_G1_{sfx}").restype = ci

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
        getattr(lib, f"lrsk_ft_set_opt_{sfx}").argtypes = [vp, cd, cd, ci, cd, cd, cd, cd, cd, cd, cd, cd, cd, vp, ci]
        getattr(lib, f"lrsk_ft_set_opt_{sfx}").restype = ci
        getattr(lib, f"lrsk_ft_upload_dataset_{sfx}").argtypes = [
            vp, ci, ci, ci, ci, ci, ci, ci, ci, ci,
            vp, vp, vp, vp, vp, vp, vp, vp, vp,
            vp, vp, vp, vp, vp, vp, vp, vp, vp,
            vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp,
        ]
        getattr(lib, f"lrsk_ft_upload_dataset_{sfx}").restype = ci
        getattr(lib, f"lrsk_ft_run_epoch_{sfx}").argtypes = [vp, vp, ci, ci, vp, vp, vp, vp, vp, vp, cd, cd, vp]
        getattr(lib, f"lrsk_ft_run_epoch_{sfx}").restype = ci
        if hasattr(lib, f"lrsk_ft_run_epoch_weighted_{sfx}"):
            getattr(lib, f"lrsk_ft_run_epoch_weighted_{sfx}").argtypes = [vp, vp, vp, ci, ci, vp, vp, vp, ci, vp, vp]
            getattr(lib, f"lrsk_ft_run_epoch_weighted_{sfx}").restype = ci
            getattr(lib, f"lrsk_ft_get_frame_err_{sfx}").argtypes = [vp, vp]
            getattr(lib, f"lrsk_ft_get_frame_err_{sfx}").restype = ci

    _LIB_CACHE[p] = lib
    return lib


def canonical_param_list(model: FastLRSKStudent) -> list[tuple[str, torch.nn.Parameter]]:
    """Exact canonical parameter order matching ``LrskFullTrainer::init`` in ``src/train/lrsk_full_train.cu``."""
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


def _build_csr_single(snd: torch.Tensor, rcv: torch.Tensor, vec: torch.Tensor, N: int):
    dev = vec.device
    if rcv.numel() == 0:
        return (
            torch.zeros(N + 1, dtype=torch.int32, device=dev),
            torch.empty(0, dtype=torch.int32, device=dev),
            torch.empty((0, 3), dtype=vec.dtype, device=dev),
        )
    order = torch.argsort(rcv, stable=True)
    rcv_s = rcv[order]
    snd_s = snd[order].to(torch.int32).contiguous()
    vec_s = vec[order].contiguous()
    counts = torch.bincount(rcv_s, minlength=N).to(torch.int32)
    row_ptr = torch.zeros(N + 1, dtype=torch.int32, device=dev)
    row_ptr[1:] = torch.cumsum(counts, dim=0)
    return row_ptr, snd_s, vec_s


def _compute_zbl_sparse(vec: torch.Tensor, snd: torch.Tensor, rcv: torch.Tensor, z_idx: torch.Tensor, model: FastLRSKStudent, N: int):
    dt, dev = vec.dtype, vec.device
    if not model.use_zbl_switch or vec.shape[0] == 0:
        z0 = torch.zeros(N, dtype=dt, device=dev)
        o1 = torch.ones(N, dtype=dt, device=dev)
        ei0 = torch.empty(0, dtype=torch.long, device=dev)
        ev0 = torch.empty((0, 3), dtype=dt, device=dev)
        return z0, o1, z0, ei0, ei0, ev0, ev0

    with torch.enable_grad():
        vec64 = vec.detach().double()
        r64 = torch.linalg.norm(vec64, dim=-1).clamp_min(1e-8)
        pair_idx = z_idx[rcv].long() * model.S + z_idx[snd].long()
        rs = model.rs_pair.to(dev)[pair_idx]
        active = r64 < rs
        if not bool(active.any().item()):
            z0 = torch.zeros(N, dtype=dt, device=dev)
            o1 = torch.ones(N, dtype=dt, device=dev)
            ei0 = torch.empty(0, dtype=torch.long, device=dev)
            ev0 = torch.empty((0, 3), dtype=dt, device=dev)
            return z0, o1, z0, ei0, ei0, ev0, ev0

        va = vec64[active].detach().requires_grad_(True)
        ra = torch.linalg.norm(va, dim=-1).clamp_min(1e-8)
        s_a = snd[active].long()
        r_a = rcv[active].long()
        p_a = pair_idx[active]
        rs_a = rs[active]
        zz_a = model.zz_pair.to(dev)[p_a]
        au_a = model.au_pair.to(dev)[p_a]
        u = ra / rs_a
        u6 = u.pow(6)
        tau = 1.0 - 28.0 * u6 + 48.0 * u6 * u - 21.0 * u6 * u * u
        x = ra / au_a
        phi = (
            ZBL_C[0] * torch.exp(-ZBL_D[0] * x)
            + ZBL_C[1] * torch.exp(-ZBL_D[1] * x)
            + ZBL_C[2] * torch.exp(-ZBL_D[2] * x)
            + ZBL_C[3] * torch.exp(-ZBL_D[3] * x)
        )
        v_pair = 0.5 * (KE_EV_A * zz_a / ra) * phi * tau
        gv_a = torch.autograd.grad(v_pair.sum(), va)[0].detach()

        e_zbl64 = torch.zeros(N, dtype=torch.float64, device=dev)
        e_zbl64.index_add_(0, r_a, v_pair.detach())
        xc = (e_zbl64 / float(model.z_c)).clamp(0.0, 1.0)
        sw64 = 1.0 - 10.0 * xc.pow(3) + 15.0 * xc.pow(4) - 6.0 * xc.pow(5)
        in_sw = (e_zbl64 > 0.0) & (e_zbl64 < float(model.z_c))
        dsw64 = torch.where(
            in_sw,
            (-30.0 * xc * xc * (1.0 - xc) * (1.0 - xc)) / float(model.z_c),
            torch.zeros_like(xc),
        )
        return e_zbl64.to(dt), sw64.to(dt), dsw64.to(dt), s_a, r_a, gv_a.to(dt), va.detach().to(dt)


def _precompute_G1_cuda(row_ptr, snd, vec, z_idx, S: int, K1: int, lmax: int, r_max: float, warp: float, sfx: str = "f32") -> torch.Tensor:
    lib = load_ft_lib()
    N = int(z_idx.shape[0])
    n_sh = (lmax + 1) * (lmax + 1)
    G1 = torch.empty((N, S, K1, n_sh), dtype=vec.dtype, device=vec.device)
    st = int(torch.cuda.current_stream(vec.device).cuda_stream)
    rc = getattr(lib, f"l1_precompute_G1_{sfx}")(
        st, N, int(S), int(K1), int(lmax), float(r_max), float(warp),
        _ptr(row_ptr.contiguous()), _ptr(snd.contiguous()), _ptr(vec.contiguous()), _ptr(z_idx.contiguous()), _ptr(G1),
    )
    if rc != 0:
        raise RuntimeError(f"l1_precompute_G1_{sfx} failed rc={rc}")
    return G1


class CudaFullLRSKTrainer:
    """High-level wrapper around ``LrskFullTrainer<float|double>`` in ``libloup_lrsk_full_train_v3.so``."""

    def __init__(self, model: FastLRSKStudent, dtype: str = "f32", soap_max_dim: int = 128):
        assert model.sym, "Pure-CUDA LRSK trainer requires sym=True"
        assert model.pair_mlp_width == 0, "Pure-CUDA LRSK trainer requires pair_mlp_width=0"
        assert model.lmax <= 3 and model.lfeat <= 2
        self.model = model
        self.sfx = dtype
        self.np_dt = np.float64 if dtype == "f64" else np.float32
        self.th_dt = torch.float64 if dtype == "f64" else torch.float32
        self.lib = load_ft_lib()
        self.param_list = canonical_param_list(model)

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
        m = self.model
        G = len(items)
        split_r2 = 1 if (abs(m.r_max2 - m.r_max) > 1e-6) else 0
        has_ei = 1 if (items and getattr(items[0], "ei_ref", None) is not None) else 0
        has_mol = 1 if (mol_list is not None and any(x is not None for x in mol_list)) else 0
        has_st = 1 if (has_stress and items and getattr(items[0], "stress", None) is not None) else 0

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
            vec = (pos[ei[1]] - pos[ei[0]] + sh) if ei.numel() > 0 else torch.empty((0, 3), dtype=self.th_dt, device=device)
            snd, rcv = ei[0], ei[1]
            z_i32 = ad.node_attrs.to(device=device).argmax(dim=-1).to(torch.int32)
            N = int(pos.shape[0])

            r_all = torch.linalg.norm(vec, dim=-1).clamp_min(1e-8) if vec.numel() > 0 else torch.empty(0, dtype=self.th_dt, device=device)
            if m.r_max < m.r_max2 - 1e-6:
                m1 = r_all < m.r_max
                rp1, s1, v1 = _build_csr_single(snd[m1], rcv[m1], vec[m1], N)
            else:
                rp1, s1, v1 = _build_csr_single(snd, rcv, vec, N)
            G1 = _precompute_G1_cuda(rp1, s1, v1, z_i32, m.S, m.n_base, m.lmax, m.r_max, m.radial_warp, sfx=self.sfx)
            n_atoms.append(N)
            n_e1.append(int(s1.numel()))
            l_z.append(z_i32.cpu().numpy())
            l_deg1.append((rp1[1:] - rp1[:-1]).cpu().numpy())
            l_snd1.append(s1.cpu().numpy())
            l_vec1.append(v1.cpu().numpy().astype(self.np_dt))
            l_G1.append(G1.cpu().numpy().astype(self.np_dt))

            if split_r2:
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
            vol_val = float(torch.abs(torch.linalg.det(cell)).item())
            l_vol.append(vol_val if vol_val > 1e-8 else 1.0)
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
        s1_a = np.ascontiguousarray(np.concatenate(l_snd1), dtype=np.int32) if int(e1_off[-1]) > 0 else np.zeros(1, dtype=np.int32)
        s2_a = np.ascontiguousarray(np.concatenate(l_snd2), dtype=np.int32) if (split_r2 and int(e2_off[-1]) > 0) else np.zeros(1, dtype=np.int32)
        v1_a = np.ascontiguousarray(np.concatenate(l_vec1, axis=0), dtype=self.np_dt) if int(e1_off[-1]) > 0 else np.zeros(3, dtype=self.np_dt)
        v2_a = np.ascontiguousarray(np.concatenate(l_vec2, axis=0), dtype=self.np_dt) if (split_r2 and int(e2_off[-1]) > 0) else np.zeros(3, dtype=self.np_dt)
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


def eval_model_splits(
    model: FastLRSKStudent,
    items: Sequence,
    zs: Sequence[int],
    device: str | torch.device = "cuda",
    has_stress: bool = True,
    batch_size: int = 8,
) -> dict:
    """Evaluate force, energy, per-species force, and optional torque RMSEs on ``items``."""
    model.eval()
    model_dt = next(model.parameters()).dtype
    f_err_sq = []
    e_err_mev = []
    sp_sq: dict[int, list[np.ndarray]] = {int(z): [] for z in zs}
    ct_errs = []

    for s0 in range(0, len(items), batch_size):
        sub = items[s0 : s0 + batch_size]
        b = batch_graph_items(sub, device=device, dtype=model_dt)
        out = model(b, training=False, compute_force=True, compute_stress=False)
        pred_e = out["energy"].detach().cpu().numpy()
        pred_f = out["forces"].detach().cpu().numpy()
        z_all = b["node_attrs"].argmax(dim=-1).detach().cpu().numpy()

        off = 0
        for g, it in enumerate(sub):
            N = int(it.positions.shape[0])
            pf = pred_f[off : off + N]
            rf = it.forces.numpy()
            df = pf - rf
            f_err_sq.append((df ** 2).ravel())
            gz = z_all[off : off + N]
            for zi, z in enumerate(zs):
                mask = gz == zi
                if np.any(mask):
                    sp_sq[int(z)].append((df[mask] ** 2).ravel())

            pe = float(pred_e[g])
            re = float(it.energy.reshape(-1)[0].item())
            e_err_mev.append(((pe - re) / max(1, N)) * 1000.0)

            if it.mol_id is not None:
                mid = it.mol_id.numpy()
                pos = it.positions.numpy()
                for m in range(int(mid.max()) + 1):
                    mask = mid == m
                    if mask.sum() > 1:
                        arm = pos[mask] - pos[mask].mean(axis=0, keepdims=True)
                        tau_p = np.cross(arm, pf[mask]).sum(axis=0)
                        tau_r = np.cross(arm, rf[mask]).sum(axis=0)
                        ct_errs.append(((tau_p - tau_r) ** 2).ravel())
            off += N

    f_mev = float(np.sqrt(np.mean(np.concatenate(f_err_sq))) * 1000.0) if f_err_sq else 0.0
    e_mev = float(np.sqrt(np.mean(np.square(e_err_mev)))) if e_err_mev else 0.0
    sp_mev = {
        int(z): (float(np.sqrt(np.mean(np.concatenate(arrs))) * 1000.0) if arrs else 0.0)
        for z, arrs in sp_sq.items()
    }
    ct_mev = float(np.sqrt(np.mean(np.concatenate(ct_errs))) * 1000.0) if ct_errs else None
    return {"f_mev": f_mev, "e_mev": e_mev, "sp_mev": sp_mev, "ct_mev": ct_mev}


def calibrate_e0_lstsq(
    model: FastLRSKStudent,
    items: Sequence,
    zs: Sequence[int],
    device: str | torch.device = "cuda",
    batch_size: int = 8,
) -> None:
    """Least-squares residual shift of ``model.e0`` after training."""
    if not items:
        return
    model_dt = next(model.parameters()).dtype
    S = len(zs)
    A = np.zeros((len(items), S), dtype=np.float64)
    y = np.zeros(len(items), dtype=np.float64)
    with torch.no_grad():
        model.e0.zero_()
    model.eval()
    with torch.no_grad():
        for s0 in range(0, len(items), batch_size):
            sub = items[s0 : s0 + batch_size]
            b = batch_graph_items(sub, device=device, dtype=model_dt)
            eps = model(b, training=False, compute_force=False, compute_stress=False)["energy"].detach().cpu().numpy()
            for j, it in enumerate(sub):
                k = s0 + j
                y[k] = float(it.energy.reshape(-1)[0].item()) - float(eps[j])
                zi = it.node_attrs.argmax(dim=-1).numpy()
                for idx in zi:
                    A[k, int(idx)] += 1.0
    e0_solved, *_ = np.linalg.lstsq(A, y, rcond=None)
    with torch.no_grad():
        model.e0.copy_(torch.tensor(e0_solved, dtype=torch.float64, device=model.e0.device))


def _cosine_lr(ep: int, epochs: int, peak_lr: float, min_lr: float, warmup: int) -> float:
    if epochs <= 1:
        return peak_lr
    if warmup > 0 and ep <= warmup:
        return peak_lr * (0.2 + 0.8 * ep / warmup)
    denom = max(1, epochs - max(1, warmup))
    frac = min(1.0, max(0.0, (ep - max(1, warmup)) / denom))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * frac))


def _build_mol_geom_list(frames: Sequence, dev: torch.device) -> tuple[list | None, float, float]:
    from .molecular import ReferenceScales, grouped_geometry

    mol_geom: list = []
    any_mol = False
    rs = ReferenceScales()
    has_scales = False
    for at in frames:
        if "mol_id" not in at.arrays:
            mol_geom.append(None)
            continue
        try:
            grouped = grouped_geometry(at, "mol_id")
        except ValueError:
            mol_geom.append(None)
            continue
        order, starts, arms_ord = grouped
        sizes = np.diff(np.r_[starts, len(at)])
        if not np.any(sizes >= 2):
            mol_geom.append(None)
            continue
        any_mol = True
        nm = len(starts)
        mid_orig = np.empty(len(at), dtype=np.int64)
        mid_orig[order] = np.repeat(np.arange(nm, dtype=np.int64), sizes)
        arms_orig = np.empty_like(arms_ord)
        arms_orig[order] = arms_ord
        mol_geom.append((
            torch.from_numpy(mid_orig).to(dev),
            nm,
            torch.from_numpy(np.ascontiguousarray(arms_orig, dtype=np.float64)).to(dev),
        ))
        if "REF_forces" in at.arrays:
            rs.observe(at, grouped)
            has_scales = True
    if not any_mol:
        return None, 1.0, 1.0
    if has_scales:
        try:
            ms = rs.result()
            return mol_geom, float(ms["net_scale"]), float(ms["torque_scale"])
        except ValueError:
            pass
    return mol_geom, 1.0, 1.0


def train_student_cuda(
    model: FastLRSKStudent,
    train_frames: Sequence,
    val_frames: Sequence,
    zs: Sequence[int],
    *,
    epochs: int = 120,
    batch_size: int = 4,
    peak_lr: float = 0.008,
    min_lr: float = 1e-4,
    eval_every: int = 10,
    we: float = 1.0,
    wf: float = 100.0,
    ws: float | None = None,
    wct: float = 0.0,
    sp_w: Sequence[float] | None = None,
    weight_decay: float = 1e-4,
    wd_ro: float = 0.0,
    rad_smooth: float = 0.0,
    w_lr_reg: float = 1e-4,
    we_tail: float = 4000.0,
    e0_frames: Sequence | None = None,
    calibrate: bool = True,
    device: str | torch.device = "cuda",
    seed: int = 42,
    early_stop_patience: int = 6,
    log_fn=None,
) -> dict:
    """Train a ``FastLRSKStudent`` via ``CudaFullLRSKTrainer`` (f32 pure CUDA) with EMA + E0 fit."""
    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)
    model = model.to(device=dev, dtype=torch.float32)
    model.e0 = model.e0.to(device=dev, dtype=torch.float64)
    phys_e0 = fit_min_norm_e0(train_frames, zs)
    e0_map = {int(z): float(e) for z, e in zip(zs, phys_e0)}

    r_graph = float(max(model.r_max, model.r_max2))
    tr_items, _, has_stress = build_graph_items(train_frames, zs, r_graph, e0_map=e0_map)
    e0_src = e0_frames if e0_frames is not None else train_frames
    tr_items_raw, _, _ = build_graph_items(e0_src, zs, r_graph, e0_map=None)
    va_items, _, va_has_stress = build_graph_items(val_frames if val_frames else train_frames[:4], zs, r_graph, e0_map=None)
    has_ei = bool(tr_items and getattr(tr_items[0], "ei_ref", None) is not None)

    model.e0.zero_()
    if calibrate:
        model.calibrate_buffers(tr_items, bs=min(8, max(1, len(tr_items))))
    mol_geom, net_s, trq_s = _build_mol_geom_list(train_frames, dev)
    has_mol = mol_geom is not None

    eff_bs = max(1, min(int(batch_size), len(tr_items), 8))
    ws_eff = (30.0 if has_stress else 0.0) if ws is None else (float(ws) if has_stress else 0.0)
    wei_eff = 1.0 if has_ei else 0.0
    wct_eff = float(wct) if has_mol else 0.0

    b_soap = max(0.95, 0.995 ** eff_bs)
    adam_b1 = max(0.85, 0.90 ** math.sqrt(eff_bs))
    adam_b2 = max(0.95, 0.99 ** eff_bs)
    d_ema_max = max(0.94, 0.99 ** eff_bs)

    sp_arr = np.asarray(sp_w, dtype=np.float32) if sp_w is not None else None
    gw_arr = np.asarray([float(getattr(it, "weight", 1.0)) for it in tr_items], dtype=np.float32)

    trainer = CudaFullLRSKTrainer(model, dtype="f32", soap_max_dim=128)
    trainer.set_optimizer(
        b_soap, b_soap, 8, float(weight_decay), 1.0,
        adam_b1, adam_b2, d_ema_max, 1.0,
        float(w_lr_reg), float(rad_smooth), float(wd_ro), sp_w=sp_arr,
    )
    trainer.upload_dataset(tr_items, mol_list=mol_geom, graph_weights=gw_arr, has_stress=has_stress, device=dev)

    best_f = float("inf")
    best_ep = 0
    best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict] = []
    no_improve = 0
    warmup = max(2, min(10, epochs // 10))
    n_steps = max(1, len(tr_items) // eff_bs)
    tail_ep = max(1, int(math.floor(epochs * 0.85)))
    rng = np.random.default_rng(seed)

    try:
        for ep in range(1, epochs + 1):
            lr_ep = _cosine_lr(ep, epochs, peak_lr, min_lr, warmup)
            in_tail = (epochs >= 20) and (ep > tail_ep)
            we_cur = float(we_tail) if in_tail else we
            wm_cur = (wct_eff * 0.5) if in_tail else wct_eff
            perm = rng.permutation(len(tr_items))
            lr_s = np.full(n_steps, lr_ep, dtype=np.float64)
            we_s = np.full(n_steps, we_cur, dtype=np.float64)
            wf_s = np.full(n_steps, wf, dtype=np.float64)
            ws_s = np.full(n_steps, ws_eff, dtype=np.float64)
            wei_s = np.full(n_steps, wei_eff, dtype=np.float64)
            wm_s = np.full(n_steps, wm_cur, dtype=np.float64)

            loss = trainer.run_epoch(perm, eff_bs, lr_s, we_s, wf_s, ws_s, wei_s, wm_s, net_s=net_s, trq_s=trq_s)

            if (ep % eval_every == 0) or (ep == epochs) or (ep == 1 and epochs >= 10):
                trainer.download_params_to_model(from_ema=True)
                calibrate_e0_lstsq(model, tr_items_raw, zs, device=dev)
                sc = eval_model_splits(model, va_items, zs, device=dev, has_stress=va_has_stress)
                history.append({"epoch": ep, "lr": lr_ep, "loss": loss, **sc})
                if log_fn is not None:
                    log_fn(
                        f"  [CUDA ep {ep:3d}/{epochs}] lr={lr_ep:.5f} loss={loss:.4f} "
                        f"val_F={sc['f_mev']:.2f} meV/A val_E={sc['e_mev']:.3f} meV/at"
                    )
                if sc["f_mev"] < best_f:
                    best_f = sc["f_mev"]
                    best_ep = ep
                    best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if ep >= epochs // 2 and no_improve >= early_stop_patience:
                        break
    finally:
        trainer.close()

    model.load_state_dict({k: v.to(device=dev, dtype=torch.float64 if k in ("e0", "rs_pair", "zz_pair", "au_pair") else torch.float32) for k, v in best_sd.items()})
    calibrate_e0_lstsq(model, tr_items_raw, zs, device=dev)
    final_va = eval_model_splits(model, va_items, zs, device=dev, has_stress=va_has_stress)
    return {
        "model": model,
        "best_ep": best_ep,
        "best_val_f_mev": float(final_va["f_mev"]),
        "final_val": final_va,
        "train_s": float(time.perf_counter() - t0),
        "epochs_run": int(history[-1]["epoch"] if history else epochs),
        "history": history,
    }
