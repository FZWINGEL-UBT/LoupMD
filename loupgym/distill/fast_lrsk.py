"""Self-contained FastLRSKStudent architecture and dataset builder for LoupMD Auto-Distiller.

Ported cleanly from ``FastLRSKStudent`` with zero ``.tmp_o`` or ``mace`` imports so parameter
shapes, buffers, spherical harmonics, radial warps, dual cutoffs, and forward equations match
``src/train/lrsk_full_train.cu``, ``loupmd/export/ilmlip.py``, and ``src/ilmlip_lrsk_fast.cu``.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase.data import covalent_radii
from ase.neighborlist import primitive_neighbor_list
from e3nn import o3

ZBL_C = (0.18175, 0.50986, 0.28022, 0.02817)
ZBL_D = (3.19980, 0.94229, 0.40290, 0.20162)
KE_EV_A = 14.39964547842567

_SH_S3 = math.sqrt(3.0)
_SH_S5 = math.sqrt(5.0)
_SH_S56 = math.sqrt(5.0 / 6.0)
_SH_S38 = math.sqrt(3.0 / 8.0)
_SH_C1 = math.sqrt(3.0)
_SH_C2 = math.sqrt(5.0)
_SH_C3 = math.sqrt(7.0)


def analytical_sh_list(r_hat: torch.Tensor, lmax: int) -> list[torch.Tensor]:
    """Pure-PyTorch e3nn component-normalized real spherical harmonics for ``l=0..lmax`` (``lmax <= 3``)."""
    x, y, z = r_hat[:, 0], r_hat[:, 1], r_hat[:, 2]
    out = [torch.ones((r_hat.shape[0], 1), dtype=r_hat.dtype, device=r_hat.device)]
    if lmax >= 1:
        out.append(r_hat * _SH_C1)
    if lmax >= 2:
        y2 = y * y
        x2z2 = x * x + z * z
        s20 = _SH_S3 * (x * z)
        s21 = _SH_S3 * (x * y)
        s22 = y2 - 0.5 * x2z2
        s23 = _SH_S3 * (y * z)
        s24 = (0.5 * _SH_S3) * (z * z - x * x)
        out.append(torch.stack((s20, s21, s22, s23, s24), dim=-1) * _SH_C2)
    if lmax >= 3:
        t4 = 4.0 * y2 - x2z2
        s30 = _SH_S56 * (s20 * z + s24 * x)
        s31 = _SH_S5 * (s20 * y)
        s32 = _SH_S38 * (t4 * x)
        s33 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
        s34 = _SH_S38 * (z * t4)
        s35 = _SH_S5 * (s24 * y)
        s36 = _SH_S56 * (s24 * z - s20 * x)
        out.append(torch.stack((s30, s31, s32, s33, s34, s35, s36), dim=-1) * _SH_C3)
    return out


def _l2_matvec(q: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv_s2 = 0.70710678118654752440
    inv_s6 = 0.40824829046386301637
    q0, q1, q2, q3, q4 = q.unbind(dim=-1)
    ux, uy, uz = u[..., 0:1], u[..., 1:2], u[..., 2:3]
    m_xx = -inv_s6 * q2 - inv_s2 * q4
    m_yy = (2.0 * inv_s6) * q2
    m_zz = -inv_s6 * q2 + inv_s2 * q4
    m_xy = inv_s2 * q1
    m_xz = inv_s2 * q0
    m_yz = inv_s2 * q3
    mu_x = m_xx * ux + m_xy * uy + m_xz * uz
    mu_y = m_xy * ux + m_yy * uy + m_yz * uz
    mu_z = m_xz * ux + m_yz * uy + m_zz * uz
    s2 = mu_x * ux + mu_y * uy + mu_z * uz
    return torch.stack((mu_x, mu_y, mu_z), dim=-1), s2


def _w3j(l1: int, l2: int, l3: int) -> torch.Tensor:
    return o3.wigner_3j(l1, l2, l3, dtype=torch.float64)


def fit_min_norm_e0(frames: Sequence, zs: Sequence[int], energy_key: str = "REF_energy") -> np.ndarray:
    """Minimum-norm least-squares per-species atomic energy baseline."""
    z2i = {int(z): i for i, z in enumerate(zs)}
    S = len(zs)
    rows = []
    rhs = []
    for at in frames:
        if energy_key not in at.info:
            continue
        cnt = np.zeros(S, dtype=np.float64)
        for z in at.numbers:
            idx = z2i.get(int(z))
            if idx is not None:
                cnt[idx] += 1.0
        rows.append(cnt)
        rhs.append(float(at.info[energy_key]))
    if not rows:
        return np.zeros(S, dtype=np.float64)
    A = np.asarray(rows, dtype=np.float64)
    y = np.asarray(rhs, dtype=np.float64)
    return np.linalg.lstsq(A, y, rcond=None)[0]


class FastLRSKStudent(nn.Module):
    """Exact architectural match to ``FastLRSKStudent`` / ``libloupmd`` ``arch=ilmlip``."""

    def __init__(
        self,
        zs: Sequence[int],
        r_max: float = 5.8,
        r_max2: float | None = None,
        lmax: int = 2,
        lfeat: int = 2,
        taper: Sequence[int] | None = None,
        n_base: int = 12,
        n_base2: int = 8,
        c0: int = 4,
        c1: int = 4,
        c2: int = 0,
        readout_width: int = 64,
        readout_depth: int = 2,
        radial_warp: float = 1.5,
        use_zbl_switch: bool = True,
        kappa: float = 16.0,
        a0: float = 0.05,
        energy_scale: float = 1.0,
        z_c: float = 10.0,
        max_order: int = 3,
        film_radials: bool = True,
        pol_field: bool = False,
        field_mlp_width: int = 32,
        sym: bool = True,
        pol_gram: bool = False,
        sw_mp: bool = False,
        mp_cap: float = 0.0,
        ef_cap: float = 0.0,
        pair_mlp_width: int = 0,
    ):
        super().__init__()
        self.zs = [int(z) for z in zs]
        self.S = len(self.zs)
        self.r_max = float(r_max)
        self.r_max2 = float(self.r_max if r_max2 is None else r_max2)
        self.lmax = int(lmax)
        self.lfeat = int(lfeat)
        self.c0 = int(c0)
        self.c1 = int(c1) if self.lfeat >= 1 else 0
        self.c2 = int(c2) if self.lfeat >= 2 else 0
        self.lmsg = min(2 if self.c2 > 0 else (1 if self.c1 > 0 else 0), self.lfeat)
        self.n_base = int(n_base)
        self.n_base2 = int(n_base2)
        self.radial_warp = float(radial_warp)
        self.use_zbl_switch = bool(use_zbl_switch)
        self.kappa = float(kappa)
        self.a0 = float(a0)
        self.energy_scale = float(energy_scale)
        self.z_c = float(z_c)
        self.max_order = int(max_order)
        self.film_radials = bool(film_radials)
        self.pol_field = bool(pol_field)
        self.field_mlp_width = int(field_mlp_width)
        self.sym = bool(sym)
        self.pol_gram = bool(pol_gram)
        self.sw_mp = bool(sw_mp)
        self.mp_cap = float(mp_cap)
        self.ef_cap = float(ef_cap)
        self.pair_mlp_width = int(pair_mlp_width)
        assert self.lmax <= 3, "analytical_sh_list supports lmax <= 3"

        if taper is None:
            self.n_l = [32, 24, 16, 8][: lmax + 1]
        else:
            assert len(taper) == lmax + 1
            self.n_l = [int(x) for x in taper]

        z_arr = torch.tensor(self.zs, dtype=torch.float64)
        rcov = torch.tensor([float(covalent_radii[z]) for z in self.zs], dtype=torch.float64)
        rs_pair = 0.65 * (rcov[:, None] + rcov[None, :])
        zz_pair = z_arr[:, None] * z_arr[None, :]
        au_pair = 0.46850 / (z_arr[:, None].pow(0.23) + z_arr[None, :].pow(0.23))
        self.register_buffer("rs_pair", rs_pair.reshape(-1))
        self.register_buffer("zz_pair", zz_pair.reshape(-1))
        self.register_buffer("au_pair", au_pair.reshape(-1))
        self.register_buffer("e0", torch.zeros(self.S, dtype=torch.float64))

        self.n_tot1 = sum(self.n_l)
        self.l1_offsets = []
        off = 0
        for nl in self.n_l:
            self.l1_offsets.append((off, off + nl))
            off += nl
        if self.film_radials:
            self.C1_R = nn.Parameter(torch.randn(self.n_tot1, n_base) / math.sqrt(n_base))
            self.C1_B = nn.Parameter(torch.randn(self.n_tot1, n_base) / math.sqrt(n_base))
            self.gamma1 = nn.Parameter(torch.zeros(self.S * self.S, self.n_tot1))
            self.beta1 = nn.Parameter(torch.zeros(self.S * self.S, self.n_tot1))
            self.eta1 = nn.Parameter(torch.randn(self.S, self.n_tot1) * 0.2 + 1.0)
            self.C1 = None
        else:
            self.C1 = nn.Parameter(torch.randn(self.S * self.S, self.n_tot1, n_base) / math.sqrt(n_base))

        self.nu2_paths = []
        for l1 in range(lmax + 1):
            for l2 in range(l1, lmax + 1):
                c = min(self.n_l[l1], self.n_l[l2])
                for L in range(min(lfeat, l1 + l2) + 1):
                    if L >= abs(l1 - l2) and (l1 + l2 + L) % 2 == 0:
                        self.nu2_paths.append((l1, l2, L, c))
                        self._reg_cg(l1, l2, L)

        self.nu3_paths = []
        if self.max_order >= 3:
            cands_by_L: dict[int, list] = {L: [] for L in range(lfeat + 1)}
            for p2_idx, (l1, l2, l12, c2) in enumerate(self.nu2_paths):
                for l3 in range(lmax + 1):
                    if l3 > lfeat:
                        continue
                    c3 = min(c2, self.n_l[l3])
                    for L in range(min(lfeat, l12 + l3) + 1):
                        if L >= abs(l12 - l3) and (l12 + l3 + L) % 2 == 0:
                            cands_by_L[L].append((p2_idx, l3, L, c3))
            gen = torch.Generator().manual_seed(12345)
            A_rand = {l: torch.randn(128, 2 * l + 1, generator=gen, dtype=torch.float64) for l in range(lmax + 1)}
            B2_rand = [
                torch.einsum("abm,ka,kb->km", _w3j(l1, l2, L) * math.sqrt(2 * L + 1), A_rand[l1], A_rand[l2])
                for l1, l2, L, _ in self.nu2_paths
            ]
            for L, cands in cands_by_L.items():
                basis = []
                for p2_idx, l3, L_, c3 in cands:
                    l12 = self.nu2_paths[p2_idx][2]
                    cg = _w3j(l12, l3, L_) * math.sqrt(2 * L_ + 1)
                    v = torch.einsum("abm,ka,kb->km", cg, B2_rand[p2_idx], A_rand[l3]).reshape(-1)
                    vn = float(v.norm())
                    if vn < 1e-10:
                        continue
                    u = v / vn
                    if basis:
                        B_mat = torch.stack(basis, dim=1)
                        rem = u - B_mat @ (B_mat.T @ u)
                        if float(rem.norm()) < 1e-6:
                            continue
                        basis.append(rem / rem.norm())
                    else:
                        basis.append(u)
                    self.nu3_paths.append((p2_idx, l3, L_, c3))
                    self._reg_cg(l12, l3, L_)

        self.bcat_dim = [0] * (self.lmsg + 1)
        for l in range(self.lmsg + 1):
            self.bcat_dim[l] += self.n_l[l]
        for _, _, L, c in self.nu2_paths:
            if L <= self.lmsg:
                self.bcat_dim[L] += c
        for _, _, L, c in self.nu3_paths:
            if L <= self.lmsg:
                self.bcat_dim[L] += c
        self.n_inv1 = self.bcat_dim[0]
        self.D_A1 = sum(self.n_l[l] * (2 * l + 1) for l in range(lmax + 1))

        self.register_buffer("sigma0", torch.tensor(1.0))
        self.register_buffer("mu_x", torch.zeros(self.n_inv1))
        self.register_buffer("std_x", torch.ones(self.n_inv1))
        if self.c1 > 0:
            self.register_buffer("std_b1", torch.ones(self.bcat_dim[1]))
        if self.c2 > 0:
            self.register_buffer("std_b2", torch.ones(self.bcat_dim[2]))

        layers = []
        d_in = self.n_inv1
        for _ in range(readout_depth):
            lin = nn.Linear(d_in, readout_width)
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(d_in))
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            d_in = readout_width
        self.readout_layers = nn.ModuleList(layers)
        self.alpha_Z = nn.Parameter(torch.zeros(self.S, readout_width))
        self.u_Z = nn.Parameter(torch.randn(self.S, readout_width) / math.sqrt(readout_width))

        if self.sym:
            if self.c0 > 0:
                self.W_qA = nn.Parameter(torch.randn(self.S, self.c0, readout_width) / math.sqrt(readout_width))
                self.b_qA = nn.Parameter(torch.zeros(self.S, self.c0))
            if self.c1 > 0:
                self.W_p1A = nn.Parameter(torch.randn(self.c1, self.bcat_dim[1]) / math.sqrt(self.bcat_dim[1]))
                self.W_g1A = nn.Parameter(torch.randn(self.S, self.c1, readout_width) * (0.2 / math.sqrt(readout_width)))
                self.b_g1A = nn.Parameter(torch.zeros(self.S, self.c1))
            if self.c2 > 0:
                self.W_p2A = nn.Parameter(torch.randn(self.c2, self.bcat_dim[2]) / math.sqrt(self.bcat_dim[2]))
                self.W_g2A = nn.Parameter(torch.randn(self.S, self.c2, readout_width) * (0.2 / math.sqrt(readout_width)))
                self.b_g2A = nn.Parameter(torch.zeros(self.S, self.c2))
        else:
            if self.c0 > 0:
                self.W_qA = nn.Parameter(torch.randn(self.S, self.c0, readout_width) / math.sqrt(readout_width))
                self.b_qA = nn.Parameter(torch.zeros(self.S, self.c0))
                self.W_qB = nn.Parameter(torch.randn(self.S, self.c0, readout_width) / math.sqrt(readout_width))
                self.b_qB = nn.Parameter(torch.zeros(self.S, self.c0))
            if self.c1 > 0:
                self.W_p1A = nn.Parameter(torch.randn(self.c1, self.bcat_dim[1]) / math.sqrt(self.bcat_dim[1]))
                self.W_p1B = nn.Parameter(torch.randn(self.c1, self.bcat_dim[1]) / math.sqrt(self.bcat_dim[1]))
                self.W_g1A = nn.Parameter(torch.randn(self.S, self.c1, readout_width) * (0.2 / math.sqrt(readout_width)))
                self.b_g1A = nn.Parameter(torch.zeros(self.S, self.c1))
                self.W_g1B = nn.Parameter(torch.randn(self.S, self.c1, readout_width) * (0.2 / math.sqrt(readout_width)))
                self.b_g1B = nn.Parameter(torch.zeros(self.S, self.c1))
            if self.c2 > 0:
                self.W_p2A = nn.Parameter(torch.randn(self.c2, self.bcat_dim[2]) / math.sqrt(self.bcat_dim[2]))
                self.W_p2B = nn.Parameter(torch.randn(self.c2, self.bcat_dim[2]) / math.sqrt(self.bcat_dim[2]))
                self.W_g2A = nn.Parameter(torch.randn(self.S, self.c2, readout_width) * (0.2 / math.sqrt(readout_width)))
                self.b_g2A = nn.Parameter(torch.zeros(self.S, self.c2))
                self.W_g2B = nn.Parameter(torch.randn(self.S, self.c2, readout_width) * (0.2 / math.sqrt(readout_width)))
                self.b_g2B = nn.Parameter(torch.zeros(self.S, self.c2))

        c01 = min(self.c0, self.c1)
        c02 = min(self.c0, self.c2)
        c12 = min(self.c1, self.c2)
        self.c01, self.c02, self.c12 = c01, c02, c12
        self.n_lr_terms = 1 + self.c0 + c01 + 2 * self.c1 + c02 + 2 * c12 + 3 * self.c2
        scale_lr = 0.25 / math.sqrt(max(self.n_lr_terms * self.n_base2, 1))
        self.W_lr_rad = nn.Parameter(torch.randn(self.S * self.S, self.n_lr_terms, self.n_base2) * scale_lr)

        rev_pair = torch.tensor([zj * self.S + zi for zi in range(self.S) for zj in range(self.S)], dtype=torch.long)
        self.register_buffer("rev_pair_idx", rev_pair)
        sym_mask = torch.zeros(self.n_lr_terms, dtype=torch.bool)
        o = 0
        sym_mask[o] = True; o += 1
        sym_mask[o : o + self.c0] = True; o += self.c0
        o += c01
        sym_mask[o : o + 2 * self.c1] = True; o += 2 * self.c1
        o += c02
        o += 2 * c12
        sym_mask[o : o + 3 * self.c2] = True; o += 3 * self.c2
        assert o == self.n_lr_terms
        self.register_buffer("lr_sym_mask", sym_mask)

        self._triu_r = [a for a in range(self.c1) for b in range(a, self.c1)]
        self._triu_c = [b for a in range(self.c1) for b in range(a, self.c1)]

        if self.pol_field:
            if self.pol_gram:
                f_dim = self.c0 + (self.c1 * (self.c1 + 1)) // 2 + self.c1 * self.c1
            else:
                f_dim = self.c0 + 2 * self.c1
            self.pol_W0 = nn.Linear(f_dim + readout_width, self.field_mlp_width)
            nn.init.normal_(self.pol_W0.weight, std=1.0 / math.sqrt(f_dim + readout_width))
            nn.init.zeros_(self.pol_W0.bias)
            self.pol_u = nn.Parameter(torch.randn(self.S, self.field_mlp_width) * (0.1 / math.sqrt(self.field_mlp_width)))

        if self.pair_mlp_width > 0:
            up_map, sgn_map = [], []
            u_idx = {}
            nxt = 0
            for zi in range(self.S):
                for zj in range(self.S):
                    key = (min(zi, zj), max(zi, zj))
                    if key not in u_idx:
                        u_idx[key] = nxt
                        nxt += 1
                    up_map.append(u_idx[key])
                    sgn_map.append(1.0 if zi > zj else (-1.0 if zi < zj else 0.0))
            self.n_upairs = nxt
            self.register_buffer("pair_to_upair", torch.tensor(up_map, dtype=torch.long))
            self.register_buffer("pair_to_sgn", torch.tensor(sgn_map, dtype=torch.float64))
            self.pair_feat_dim = 3 * self.c0 + 4 * self.c1
            H = self.pair_mlp_width
            self.bo_Wx = nn.Parameter(torch.randn(H, self.pair_feat_dim) / math.sqrt(max(self.pair_feat_dim, 1)))
            self.bo_Wr = nn.Parameter(torch.randn(self.n_upairs, H, self.n_base2) / math.sqrt(max(self.n_base2, 1)))
            self.bo_b = nn.Parameter(torch.zeros(self.n_upairs, H))
            self.bo_v = nn.Parameter(torch.randn(self.n_upairs, H) * (0.05 / math.sqrt(H)))

    def _w_lr_eff(self) -> torch.Tensor:
        if not self.sym:
            return self.W_lr_rad
        w_sym = 0.5 * (self.W_lr_rad + self.W_lr_rad[self.rev_pair_idx])
        return torch.where(self.lr_sym_mask[None, :, None], w_sym, self.W_lr_rad)

    def _c1_eff(self) -> torch.Tensor:
        if not self.film_radials:
            return self.C1
        zj = torch.arange(self.S * self.S, device=self.C1_R.device) % self.S
        eta = self.eta1[zj]
        return ((1.0 + self.gamma1).unsqueeze(-1) * self.C1_R.unsqueeze(0) + self.beta1.unsqueeze(-1) * self.C1_B.unsqueeze(0)) * eta.unsqueeze(-1)

    def _reg_cg(self, l1: int, l2: int, L: int):
        key = f"cg_{l1}_{l2}_{L}"
        if not hasattr(self, key):
            t = _w3j(l1, l2, L) * math.sqrt(2 * L + 1)
            self.register_buffer(key, t.to(torch.get_default_dtype()))

    def _chebyshev(self, r: torch.Tensor, r_cut: float, nb: int, warp: float = 0.0) -> torch.Tensor:
        t = (r / r_cut).clamp(0.0, 1.0)
        t5 = t * t * t * t * t
        env = torch.where(t < 1.0, 1.0 - 21.0 * t5 + 35.0 * t5 * t - 15.0 * t5 * t * t, torch.zeros_like(t))
        u = ((1.0 + warp) * t) / (1.0 + warp * t) if warp > 0.0 else t
        x = 2.0 * u - 1.0
        ts = [torch.ones_like(x)]
        if nb > 1:
            ts.append(x)
        for _ in range(2, nb):
            ts.append(2.0 * x * ts[-1] - ts[-2])
        return torch.stack(ts, dim=-1) * env.unsqueeze(-1)

    def _zbl_per_atom(self, r: torch.Tensor, receivers: torch.Tensor, pair_idx: torch.Tensor, N: int) -> torch.Tensor:
        rs = self.rs_pair[pair_idx].to(r.dtype)
        u = r / rs
        mask = u < 1.0
        u_c = torch.where(mask, u, torch.ones_like(u))
        u6 = u_c.pow(6)
        tau = torch.where(mask, 1.0 - 28.0 * u6 + 48.0 * u6 * u_c - 21.0 * u6 * u_c * u_c, torch.zeros_like(u))
        zz = self.zz_pair[pair_idx].to(r.dtype)
        au = self.au_pair[pair_idx].to(r.dtype)
        x = r / au
        phi = (
            ZBL_C[0] * torch.exp(-ZBL_D[0] * x)
            + ZBL_C[1] * torch.exp(-ZBL_D[1] * x)
            + ZBL_C[2] * torch.exp(-ZBL_D[2] * x)
            + ZBL_C[3] * torch.exp(-ZBL_D[3] * x)
        )
        v_pair = 0.5 * (KE_EV_A * zz / r) * phi * tau
        e_zbl = torch.zeros(N, dtype=r.dtype, device=r.device)
        e_zbl.index_add_(0, receivers, v_pair)
        return e_zbl

    def _compute_l1(self, r: torch.Tensor, r_hat: torch.Tensor, receivers: torch.Tensor, pair_idx: torch.Tensor, N: int):
        dt, dev = r.dtype, r.device
        g1 = self._chebyshev(r, self.r_max, self.n_base, warp=self.radial_warp)
        R1 = torch.einsum("ek,eck->ec", g1, self._c1_eff()[pair_idx].to(dt))
        Y_l = analytical_sh_list(r_hat, self.lmax)

        A1_raw = []
        for l in range(self.lmax + 1):
            s0, s1 = self.l1_offsets[l]
            ph = R1[:, s0:s1].unsqueeze(-1) * Y_l[l].unsqueeze(1)
            a = torch.zeros(N, self.n_l[l], 2 * l + 1, dtype=dt, device=dev)
            a.index_add_(0, receivers, ph)
            A1_raw.append(a)

        sq1 = sum((a * a).sum(dim=(1, 2)) for a in A1_raw) / float(self.D_A1)
        sig0 = self.sigma0.to(dt)
        f0 = (self.a0 / sig0) * torch.rsqrt(1.0 + (sq1 / (sig0 * sig0)) / (self.kappa * self.kappa))
        A1 = [a * f0[:, None, None] for a in A1_raw]

        B2 = []
        for l1, l2, L, c in self.nu2_paths:
            cg = getattr(self, f"cg_{l1}_{l2}_{L}").to(dt)
            B2.append(torch.einsum("abm,nca,ncb->ncm", cg, A1[l1][:, :c], A1[l2][:, :c]))

        B3 = []
        for p2_idx, l3, L, c in self.nu3_paths:
            l12 = self.nu2_paths[p2_idx][2]
            cg = getattr(self, f"cg_{l12}_{l3}_{L}").to(dt)
            B3.append(torch.einsum("abm,nca,ncb->ncm", cg, B2[p2_idx][:, :c], A1[l3][:, :c]))

        B_cat = []
        for l in range(self.lmsg + 1):
            parts = [A1[l]]
            for idx, (_, _, L, _) in enumerate(self.nu2_paths):
                if L == l:
                    parts.append(B2[idx])
            for idx, (_, _, L, _) in enumerate(self.nu3_paths):
                if L == l:
                    parts.append(B3[idx])
            B_cat.append(torch.cat(parts, dim=1))
        return B_cat, sq1, g1

    @torch.no_grad()
    def calibrate_buffers(self, items: list, bs: int = 8):
        self.eval()
        dev = self.mu_x.device
        dt = self.mu_x.dtype
        sq1_sum, n_tot = 0.0, 0
        limit = min(len(items), 64)
        for s0 in range(0, limit, bs):
            d = batch_graph_items(items[s0 : s0 + bs], device=dev, dtype=dt)
            pos = d["positions"]
            snd, rcv = d["edge_index"][0], d["edge_index"][1]
            vec = pos[rcv] - pos[snd] + d["shifts"]
            r = torch.linalg.norm(vec, dim=-1).clamp_min(1e-8)
            z_idx = d["node_attrs"].argmax(-1)
            if self.r_max < self.r_max2 - 1e-6:
                m1 = r < self.r_max
                _, sq1, _ = self._compute_l1(r[m1], (vec[m1] / r[m1].unsqueeze(-1)), rcv[m1], z_idx[rcv[m1]] * self.S + z_idx[snd[m1]], pos.shape[0])
            else:
                _, sq1, _ = self._compute_l1(r, vec / r.unsqueeze(-1), rcv, z_idx[rcv] * self.S + z_idx[snd], pos.shape[0])
            sq1_sum += float(sq1.double().sum())
            n_tot += pos.shape[0]
        self.sigma0.fill_(math.sqrt(max(sq1_sum / max(n_tot, 1), 1e-12)))

        x_sum = torch.zeros(self.n_inv1, dtype=torch.float64, device=dev)
        x_sq = torch.zeros(self.n_inv1, dtype=torch.float64, device=dev)
        b1_sq = torch.zeros(self.bcat_dim[1], dtype=torch.float64, device=dev) if self.c1 > 0 else None
        b2_sq = torch.zeros(self.bcat_dim[2], dtype=torch.float64, device=dev) if self.c2 > 0 else None
        for s0 in range(0, limit, bs):
            d = batch_graph_items(items[s0 : s0 + bs], device=dev, dtype=dt)
            pos = d["positions"]
            snd, rcv = d["edge_index"][0], d["edge_index"][1]
            vec = pos[rcv] - pos[snd] + d["shifts"]
            r = torch.linalg.norm(vec, dim=-1).clamp_min(1e-8)
            z_idx = d["node_attrs"].argmax(-1)
            if self.r_max < self.r_max2 - 1e-6:
                m1 = r < self.r_max
                B_cat, _, _ = self._compute_l1(r[m1], (vec[m1] / r[m1].unsqueeze(-1)), rcv[m1], z_idx[rcv[m1]] * self.S + z_idx[snd[m1]], pos.shape[0])
            else:
                B_cat, _, _ = self._compute_l1(r, vec / r.unsqueeze(-1), rcv, z_idx[rcv] * self.S + z_idx[snd], pos.shape[0])
            x64 = B_cat[0].squeeze(-1).double()
            x_sum += x64.sum(dim=0); x_sq += (x64 * x64).sum(dim=0)
            if self.c1 > 0:
                b1_sq += (B_cat[1].double() ** 2).mean(dim=-1).sum(dim=0)
            if self.c2 > 0:
                b2_sq += (B_cat[2].double() ** 2).mean(dim=-1).sum(dim=0)
        mu = x_sum / max(n_tot, 1)
        var = (x_sq / max(n_tot, 1) - mu * mu).clamp_min(1e-14)
        self.mu_x.copy_(mu.to(self.mu_x.dtype))
        self.std_x.copy_(var.sqrt().to(self.std_x.dtype))
        if self.c1 > 0:
            self.std_b1.copy_((b1_sq / max(n_tot, 1)).clamp_min(1e-14).sqrt().to(self.std_b1.dtype))
        if self.c2 > 0:
            self.std_b2.copy_((b2_sq / max(n_tot, 1)).clamp_min(1e-14).sqrt().to(self.std_b2.dtype))

    def node_energy_from_vec(self, vec: torch.Tensor, snd: torch.Tensor, rcv: torch.Tensor, z_idx: torch.Tensor, N: int) -> torch.Tensor:
        """Pure physics energy per atom as a function of edge vectors ``vec``."""
        dt, dev = vec.dtype, vec.device
        if vec.numel() == 0:
            r = torch.empty(0, dtype=dt, device=dev)
            u = torch.empty((0, 3), dtype=dt, device=dev)
        else:
            r = torch.linalg.norm(vec, dim=-1).clamp_min(1e-8)
            u = vec / r.unsqueeze(-1)
        pair_idx = z_idx[rcv] * self.S + z_idx[snd]

        if self.r_max < self.r_max2 - 1e-6:
            m1 = r < self.r_max
            B_cat, _, g1 = self._compute_l1(r[m1], u[m1], rcv[m1], pair_idx[m1], N)
        else:
            B_cat, _, g1 = self._compute_l1(r, u, rcv, pair_idx, N)
        x1 = B_cat[0].squeeze(-1)

        x_norm = (x1 - self.mu_x.to(dt)) / self.std_x.to(dt).clamp_min(1e-7)
        h = F.silu(self.readout_layers[0](x_norm) + self.alpha_Z[z_idx].to(dt))
        for layer in self.readout_layers[1:]:
            h = F.silu(layer(h))
        e_site = (h * self.u_Z[z_idx].to(dt)).sum(dim=-1) * self.energy_scale

        e_zbl = None
        sw = None
        if self.use_zbl_switch:
            e_zbl = self._zbl_per_atom(r, rcv, pair_idx, N)
            xc = (e_zbl / self.z_c).clamp(0.0, 1.0)
            sw = 1.0 - 10.0 * xc.pow(3) + 15.0 * xc.pow(4) - 6.0 * xc.pow(5)

        inv_cap2 = (1.0 / (self.mp_cap * self.mp_cap)) if self.mp_cap > 0.0 else 0.0

        def _cap_s(v: torch.Tensor) -> torch.Tensor:
            if inv_cap2 > 0.0:
                v = v * torch.rsqrt(1.0 + (v * v) * inv_cap2)
            if self.sw_mp and sw is not None:
                v = v * sw.unsqueeze(-1)
            return v

        def _cap_v(v: torch.Tensor) -> torch.Tensor:
            if inv_cap2 > 0.0:
                v = v * torch.rsqrt(1.0 + (v * v).sum(dim=-1, keepdim=True) * inv_cap2)
            if self.sw_mp and sw is not None:
                v = v * sw[:, None, None]
            return v

        if self.c0 > 0:
            qA = _cap_s(torch.einsum("ncw,nw->nc", self.W_qA[z_idx].to(dt), h) + self.b_qA[z_idx].to(dt))
            qB = qA if self.sym else _cap_s(torch.einsum("ncw,nw->nc", self.W_qB[z_idx].to(dt), h) + self.b_qB[z_idx].to(dt))
        if self.c1 > 0:
            b1n = B_cat[1] / self.std_b1.to(dt).clamp_min(1e-7)[None, :, None]
            g1A = 1.0 + torch.einsum("ncw,nw->nc", self.W_g1A[z_idx].to(dt), h) + self.b_g1A[z_idx].to(dt)
            pA = _cap_v(torch.einsum("ck,nkm->ncm", self.W_p1A.to(dt), b1n) * g1A.unsqueeze(-1))
            if self.sym:
                pB = pA
            else:
                g1B = 1.0 + torch.einsum("ncw,nw->nc", self.W_g1B[z_idx].to(dt), h) + self.b_g1B[z_idx].to(dt)
                pB = _cap_v(torch.einsum("ck,nkm->ncm", self.W_p1B.to(dt), b1n) * g1B.unsqueeze(-1))
        if self.c2 > 0:
            b2n = B_cat[2] / self.std_b2.to(dt).clamp_min(1e-7)[None, :, None]
            g2A = 1.0 + torch.einsum("ncw,nw->nc", self.W_g2A[z_idx].to(dt), h) + self.b_g2A[z_idx].to(dt)
            QA = _cap_v(torch.einsum("ck,nkm->ncm", self.W_p2A.to(dt), b2n) * g2A.unsqueeze(-1))
            if self.sym:
                QB = QA
            else:
                g2B = 1.0 + torch.einsum("ncw,nw->nc", self.W_g2B[z_idx].to(dt), h) + self.b_g2B[z_idx].to(dt)
                QB = _cap_v(torch.einsum("ck,nkm->ncm", self.W_p2B.to(dt), b2n) * g2B.unsqueeze(-1))

        if self.r_max2 < self.r_max - 1e-6 or self.r_max < self.r_max2 - 1e-6:
            m2 = r < self.r_max2
            r2, u2, rcv2, snd2, pair2 = r[m2], u[m2], rcv[m2], snd[m2], pair_idx[m2]
            g2 = self._chebyshev(r2, self.r_max2, self.n_base2, warp=self.radial_warp)
        else:
            r2, u2, rcv2, snd2, pair2 = r, u, rcv, snd, pair_idx
            g2 = g1 if self.n_base2 == self.n_base else self._chebyshev(r2, self.r_max2, self.n_base2, warp=self.radial_warp)

        W_eff = self._w_lr_eff().to(dt)
        W_e = torch.einsum("ek,eck->ec", g2, W_eff[pair2])
        off = 0
        e_ij = W_e[:, 0]
        off += 1

        if self.c0 > 0:
            qAi = qA[rcv2]
            qBj = qB[snd2]
            w0 = W_e[:, off : off + self.c0]; off += self.c0
            e_ij = e_ij + (w0 * qAi * qBj).sum(dim=-1)
        if self.c1 > 0:
            pAi = pA[rcv2]
            pBj = pB[snd2]
            pAi_u = (pAi * u2.unsqueeze(1)).sum(dim=-1)
            pBj_u = (pBj * u2.unsqueeze(1)).sum(dim=-1)
        if self.c01 > 0:
            w01 = W_e[:, off : off + self.c01]; off += self.c01
            e_ij = e_ij + (w01 * qAi[:, : self.c01] * pBj_u[:, : self.c01]).sum(dim=-1)
        if self.c1 > 0:
            w1s = W_e[:, off : off + self.c1]; off += self.c1
            w1p = W_e[:, off : off + self.c1]; off += self.c1
            e_ij = e_ij + (w1s * pAi_u * pBj_u + w1p * (pAi * pBj).sum(dim=-1)).sum(dim=-1)
        if self.c2 > 0:
            QAi = QA[rcv2]
            QBj = QB[snd2]
            Mui, s2i = _l2_matvec(QAi, u2)
            Muj, s2j = _l2_matvec(QBj, u2)
        if self.c02 > 0:
            w02 = W_e[:, off : off + self.c02]; off += self.c02
            e_ij = e_ij + (w02 * qAi[:, : self.c02] * s2j[:, : self.c02]).sum(dim=-1)
        if self.c12 > 0:
            w12s = W_e[:, off : off + self.c12]; off += self.c12
            w12p = W_e[:, off : off + self.c12]; off += self.c12
            e_ij = e_ij + (
                w12s * pAi_u[:, : self.c12] * s2j[:, : self.c12]
                + w12p * (pAi[:, : self.c12] * Muj[:, : self.c12]).sum(dim=-1)
            ).sum(dim=-1)
        if self.c2 > 0:
            w2s = W_e[:, off : off + self.c2]; off += self.c2
            w2p = W_e[:, off : off + self.c2]; off += self.c2
            w2d = W_e[:, off : off + self.c2]; off += self.c2
            e_ij = e_ij + (
                w2s * (s2i * s2j)
                + w2p * (Mui * Muj).sum(dim=-1)
                + w2d * (QAi * QBj).sum(dim=-1)
            ).sum(dim=-1)

        e_lr = torch.zeros(N, dtype=dt, device=dev)
        e_lr.index_add_(0, rcv2, (0.5 * self.energy_scale) * e_ij)

        if self.pol_field:
            phi_e = w0 * qBj
            if self.c01 > 0:
                if self.c01 == self.c0:
                    phi_e = phi_e + w01 * pBj_u
                else:
                    phi_e = torch.cat([phi_e[:, : self.c01] + w01 * pBj_u[:, : self.c01], phi_e[:, self.c01 :]], dim=-1)
            Phi = torch.zeros(N, self.c0, dtype=dt, device=dev)
            Phi.index_add_(0, rcv2, phi_e)

            E_e = (w1s * pBj_u).unsqueeze(-1) * u2.unsqueeze(1) + w1p.unsqueeze(-1) * pBj
            if self.c01 > 0:
                add_e = (w01 * qBj[:, : self.c01]).unsqueeze(-1) * u2.unsqueeze(1)
                if self.c01 == self.c1:
                    E_e = E_e + add_e
                else:
                    E_e = torch.cat([E_e[:, : self.c01] + add_e, E_e[:, self.c01 :]], dim=1)
            Ef = torch.zeros(N, self.c1, 3, dtype=dt, device=dev)
            Ef.index_add_(0, rcv2, E_e)

            if self.ef_cap > 0.0:
                inv_ef2 = 1.0 / (self.ef_cap * self.ef_cap)
                Phi = Phi * torch.rsqrt(1.0 + (Phi * Phi) * inv_ef2)
                Ef = Ef * torch.rsqrt(1.0 + (Ef * Ef).sum(dim=-1, keepdim=True) * inv_ef2)
            if self.sw_mp and sw is not None and self.ef_cap > 0.0:
                Phi = Phi * sw.unsqueeze(-1)
                Ef = Ef * sw[:, None, None]

            if self.pol_gram:
                ee_mat = torch.einsum("nam,nbm->nab", Ef, Ef)
                ee_triu = ee_mat[:, self._triu_r, self._triu_c]
                pe_full = torch.einsum("nam,nbm->nab", pA, Ef).reshape(N, self.c1 * self.c1)
                pol_in = torch.cat([Phi, ee_triu, pe_full, h], dim=-1)
            else:
                E_sq = (Ef * Ef).sum(dim=-1)
                p_dot_E = (pA * Ef).sum(dim=-1)
                pol_in = torch.cat([Phi, E_sq, p_dot_E, h], dim=-1)
            e_pol = (F.silu(self.pol_W0(pol_in)) * self.pol_u[z_idx].to(dt)).sum(dim=-1) * self.energy_scale
            e_lr = e_lr + e_pol

        e_ml = e_site + e_lr
        if self.use_zbl_switch:
            return sw * e_ml + e_zbl
        return e_ml

    def forward(self, d: dict, training: bool = False, compute_force: bool = True, compute_stress: bool = True):
        if "vec" in d:
            vec_in = d["vec"]
            snd, rcv = d["snd"], d["rcv"]
            z_idx = d["z_idx"]
            batch = d["batch"]
            edge_batch = d["edge_batch"]
            N, G = int(d["N"]), int(d["G"])
            vol = d["vol"].view(G, 1, 1) if compute_stress else None
        else:
            pos = d["positions"]
            snd, rcv = d["edge_index"][0], d["edge_index"][1]
            vec_in = pos[rcv] - pos[snd] + d["shifts"]
            z_idx = d["node_attrs"].argmax(dim=-1)
            batch = d["batch"]
            edge_batch = batch[rcv]
            N = pos.shape[0]
            G = int(d["ptr"].shape[0]) - 1
            if compute_stress:
                cell = d["cell"].view(G, 3, 3)
                vol = torch.abs(torch.linalg.det(cell)).view(G, 1, 1)
            else:
                vol = None

        dt, dev = vec_in.dtype, vec_in.device
        vec = vec_in.detach().requires_grad_(True) if compute_force else vec_in
        node_inter = self.node_energy_from_vec(vec, snd, rcv, z_idx, N)

        node_e64 = node_inter.double() + self.e0[z_idx]
        energy = torch.zeros(G, dtype=torch.float64, device=dev)
        energy.index_add_(0, batch, node_e64)

        forces = stress = None
        if compute_force:
            e_diff = node_inter.sum()
            g_vec = torch.autograd.grad([e_diff], [vec], create_graph=training, retain_graph=training)[0]
            forces = torch.zeros(N, 3, dtype=dt, device=dev)
            forces.index_add_(0, snd, g_vec)
            forces.index_add_(0, rcv, -g_vec)
            if compute_stress:
                v_e = vec.unsqueeze(-1) * g_vec.unsqueeze(-2)
                vir_e = 0.5 * (v_e + v_e.transpose(-1, -2))
                vir = torch.zeros(G, 3, 3, dtype=dt, device=dev)
                vir.index_add_(0, edge_batch, vir_e)
                stress = vir / vol.to(dt)
        return {"energy": energy, "node_energy": node_inter, "forces": forces, "stress": stress}


def _neighbor_pairs(atoms, r_graph: float):
    pos = np.asarray(atoms.positions, dtype=np.float64)
    pbc = tuple(bool(x) for x in atoms.pbc)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    if all(pbc) and abs(float(np.linalg.det(cell))) > 1e-8:
        try:
            from matscipy.neighbours import neighbour_list
            return neighbour_list("ijS", atoms, float(r_graph))
        except Exception:
            pass
    return primitive_neighbor_list("ijS", pbc, cell, pos, float(r_graph))


def build_graph_items(
    frames: Sequence,
    zs: Sequence[int],
    r_max: float,
    e0_map: dict[int, float] | None = None,
    mol_key: str | None = "mol_id",
) -> tuple[list[SimpleNamespace], list | None, bool]:
    """Convert ASE ``Atoms`` frames into lightweight graph objects (use ``r_graph=max(r_max, r_max2)``)."""
    z2i = {int(z): i for i, z in enumerate(zs)}
    S = len(zs)
    eye = np.eye(S, dtype=np.float64)
    items: list[SimpleNamespace] = []
    mol_list: list | None = []
    any_mol = False
    has_stress = bool(frames)

    for at in frames:
        i_idx, j_idx, S_arr = _neighbor_pairs(at, float(r_max))
        cell = np.asarray(at.cell.array, dtype=np.float64)
        shifts = (np.asarray(S_arr, dtype=np.float64) @ cell) if len(i_idx) > 0 else np.zeros((0, 3), dtype=np.float64)
        z_idx = np.asarray([z2i[int(z)] for z in at.numbers], dtype=np.int64)
        node_attrs = eye[z_idx]

        energy_val = float(at.info.get("REF_energy", 0.0))
        if e0_map is not None:
            energy_val -= sum(float(e0_map[int(z)]) for z in at.numbers)
        forces_arr = np.asarray(
            at.arrays["REF_forces"] if "REF_forces" in at.arrays else np.zeros((len(at), 3), dtype=np.float64),
            dtype=np.float64,
        )
        raw_stress = at.info.get("REF_stress", None)
        if raw_stress is not None and all(at.pbc):
            stress_arr = np.asarray(raw_stress, dtype=np.float64).reshape(3, 3)
        else:
            has_stress = False
            stress_arr = np.zeros((3, 3), dtype=np.float64)

        ei_ref = torch.from_numpy(np.asarray(at.arrays["REF_ei"], dtype=np.float64)) if "REF_ei" in at.arrays else None
        mol_id = torch.from_numpy(np.asarray(at.arrays[mol_key], dtype=np.int64)) if (mol_key and mol_key in at.arrays) else None
        if mol_id is not None:
            any_mol = True
        mol_list.append(mol_id)

        edge_index = np.stack([np.asarray(i_idx, dtype=np.int64), np.asarray(j_idx, dtype=np.int64)], axis=0)
        items.append(
            SimpleNamespace(
                positions=torch.from_numpy(np.asarray(at.positions, dtype=np.float64)),
                edge_index=torch.from_numpy(edge_index),
                shifts=torch.from_numpy(shifts),
                node_attrs=torch.from_numpy(node_attrs),
                cell=torch.from_numpy(cell.copy()),
                energy=torch.tensor([energy_val], dtype=torch.float64),
                forces=torch.from_numpy(forces_arr),
                stress=torch.from_numpy(stress_arr).unsqueeze(0),
                ei_ref=ei_ref,
                mol_id=mol_id,
                weight=float(at.info.get("distill_weight", 1.0)),
            )
        )

    return items, (mol_list if any_mol else None), has_stress


def batch_graph_items(items: Sequence[SimpleNamespace], device="cuda", dtype=torch.float64) -> dict:
    """Collate lightweight graph items into a batch dict for ``FastLRSKStudent.forward``."""
    pos_l, ei_l, sh_l, na_l, cell_l, batch_l, ebatch_l, ptr_l = [], [], [], [], [], [], [], [0]
    n_off = 0
    for g, it in enumerate(items):
        N = int(it.positions.shape[0])
        E = int(it.edge_index.shape[1])
        pos_l.append(it.positions)
        ei_l.append(it.edge_index + n_off)
        sh_l.append(it.shifts)
        na_l.append(it.node_attrs)
        cell_l.append(it.cell)
        batch_l.append(torch.full((N,), g, dtype=torch.long))
        ebatch_l.append(torch.full((E,), g, dtype=torch.long))
        n_off += N
        ptr_l.append(n_off)

    return {
        "positions": torch.cat(pos_l, dim=0).to(device=device, dtype=dtype),
        "edge_index": torch.cat(ei_l, dim=1).to(device=device, dtype=torch.long),
        "shifts": torch.cat(sh_l, dim=0).to(device=device, dtype=dtype),
        "node_attrs": torch.cat(na_l, dim=0).to(device=device, dtype=dtype),
        "cell": torch.cat(cell_l, dim=0).to(device=device, dtype=dtype),
        "batch": torch.cat(batch_l, dim=0).to(device=device, dtype=torch.long),
        "edge_batch": torch.cat(ebatch_l, dim=0).to(device=device, dtype=torch.long),
        "ptr": torch.tensor(ptr_l, dtype=torch.long, device=device),
    }


def make_student_args_dict(model: FastLRSKStudent, extra: dict | None = None) -> dict:
    """Build canonical checkpoint ``args`` metadata compatible with ``loupmd.export.ilmlip``."""
    d = {
        "lrsk": True,
        "sym": model.sym,
        "pair_mlp_width": 0,
        "zs": list(model.zs),
        "r_max": float(model.r_max),
        "r_max2": float(model.r_max2),
        "lmax": int(model.lmax),
        "lfeat": int(model.lfeat),
        "taper": list(model.n_l),
        "n_base": int(model.n_base),
        "n_base2": int(model.n_base2),
        "c0": int(model.c0),
        "c1": int(model.c1),
        "c2": int(model.c2),
        "readout_depth": len(model.readout_layers),
        "readout_width": int(model.readout_layers[0].out_features),
        "field_mlp_width": int(model.field_mlp_width),
        "pol_gram": bool(model.pol_gram),
        "pol_field": bool(model.pol_field),
        "radial_warp": float(model.radial_warp),
        "use_zbl_switch": bool(model.use_zbl_switch),
        "film_radials": bool(model.film_radials),
        "kappa": float(model.kappa),
        "a0": float(model.a0),
        "z_c": float(model.z_c),
        "max_order": int(model.max_order),
        "sw_mp": bool(model.sw_mp),
        "mp_cap": float(model.mp_cap),
        "ef_cap": float(model.ef_cap),
    }
    if extra:
        d.update(extra)
    return d


def save_student_checkpoint(model: FastLRSKStudent, ckpt_path: str | Path, extra_args: dict | None = None) -> Path:
    p = Path(ckpt_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "energy_scale": float(model.energy_scale),
            "args": make_student_args_dict(model, extra_args),
        },
        p,
    )
    return p
