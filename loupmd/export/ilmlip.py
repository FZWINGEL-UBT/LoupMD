"""Export an ILMlip-Student PyTorch checkpoint (.pt) into a native LoupMD bundle (arch=ilmlip).

Folds pair-FiLM radial parameters (C_R, C_B, gamma, beta, eta) into compact per-ordered-pair
Chebyshev tables C1_eff / C2_eff, pre-contracts exact order-1 back-flow tables C_bf = (-1)^l
W_msg[l][:, :n_l[l]] @ C1_eff[rev_pair, l], packs sparse Clebsch-Gordan & Wigner-3j nonzeros,
and folds ro_b_0 into alpha_Z and energy_scale into u_Z.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from e3nn import o3

from .bundle import write_bundle

EXPORTER_VERSION = 1


def _pack_sparse_tensor(t: np.ndarray, tol: float = 1e-12) -> tuple[list[int], list[float]]:
    """Pack nonzero entries of a 3D tensor [d1, d2, d3] into packed m1|(m2<<4)|(m3<<8) + float32."""
    idx_out: list[int] = []
    val_out: list[float] = []
    d1, d2, d3 = t.shape
    for m1 in range(d1):
        for m2 in range(d2):
            for m3 in range(d3):
                v = float(t[m1, m2, m3])
                if abs(v) > tol:
                    idx_out.append(m1 | (m2 << 4) | (m3 << 8))
                    val_out.append(v)
    return idx_out, val_out


def _compute_c_eff(sd: dict, prefix: str, S: int, film: bool) -> np.ndarray:
    if not film:
        return sd[prefix].detach().cpu().to(torch.float64).numpy()
    cr = sd[f"{prefix}_R"].detach().cpu().to(torch.float64)
    cb = sd[f"{prefix}_B"].detach().cpu().to(torch.float64)
    gam = sd[f"gamma{prefix[1:]}"].detach().cpu().to(torch.float64)
    bet = sd[f"beta{prefix[1:]}"].detach().cpu().to(torch.float64)
    eta = sd[f"eta{prefix[1:]}"].detach().cpu().to(torch.float64)
    zj = torch.arange(S * S) % S
    c_eff = ((1.0 + gam).unsqueeze(-1) * cr.unsqueeze(0) + bet.unsqueeze(-1) * cb.unsqueeze(0)) * eta[zj].unsqueeze(-1)
    return c_eff.numpy()


def _build_paths(cfg: dict):
    lmax = int(cfg["lmax"])
    lfeat = int(cfg["lfeat"])
    lmsg = int(cfg["lmsg"])
    lfeat2 = int(cfg.get("lfeat2", lfeat))
    lout2 = int(cfg.get("lout2", min(lfeat2 + lmsg, max(2, lfeat))))
    n_l = [int(x) for x in cfg["n_l"]]
    n_msg = int(cfg["n_msg"])
    r_gram = int(cfg["r_gram"])
    r_bisp = int(cfg["r_bisp"])
    max_order = int(cfg.get("max_order", 3))

    cg_cache: dict[tuple[int, int, int], tuple[int, int]] = {}
    cg_idx: list[int] = []
    cg_val: list[float] = []

    def reg_cg(l1: int, l2: int, L: int) -> tuple[int, int]:
        key = (l1, l2, L)
        if key not in cg_cache:
            t = (o3.wigner_3j(l1, l2, L, dtype=torch.float64) * math.sqrt(2 * L + 1)).numpy()
            off = len(cg_idx)
            ii, vv = _pack_sparse_tensor(t)
            cg_idx.extend(ii)
            cg_val.extend(vv)
            cg_cache[key] = (off, len(ii))
        return cg_cache[key]

    nu2_paths = []
    for l1 in range(lmax + 1):
        for l2 in range(l1, lmax + 1):
            c = min(n_l[l1], n_l[l2])
            for L in range(min(lfeat, l1 + l2) + 1):
                if L >= abs(l1 - l2) and (l1 + l2 + L) % 2 == 0:
                    off, cnt = reg_cg(l1, l2, L)
                    nu2_paths.append((l1, l2, L, c, off, cnt))

    nu3_paths = []
    has_l2 = n_msg > 0
    is_lrsk = bool(cfg.get("lrsk", False))
    if max_order >= 3:
        max_nu3_L = lfeat if (has_l2 or is_lrsk) else 0
        cands_by_L: dict[int, list[tuple[int, int, int, int]]] = {L: [] for L in range(max_nu3_L + 1)}
        for p2_idx, (l1, l2, l12, c2, _, _) in enumerate(nu2_paths):
            for l3 in range(lmax + 1):
                if l3 > lfeat:
                    continue
                c3 = min(c2, n_l[l3])
                for L in range(min(max_nu3_L, l12 + l3) + 1):
                    if L >= abs(l12 - l3) and (l12 + l3 + L) % 2 == 0:
                        cands_by_L[L].append((p2_idx, l3, L, c3))
        gen = torch.Generator().manual_seed(12345)
        K_test = 128
        A_rand = {l: torch.randn(K_test, 2 * l + 1, generator=gen, dtype=torch.float64) for l in range(lmax + 1)}
        B2_rand = []
        for l1, l2, L, _, _, _ in nu2_paths:
            cg = o3.wigner_3j(l1, l2, L, dtype=torch.float64) * math.sqrt(2 * L + 1)
            B2_rand.append(torch.einsum("abm,ka,kb->km", cg, A_rand[l1], A_rand[l2]))
        for L, cands in cands_by_L.items():
            basis = []
            for p2_idx, l3, L_, c3 in cands:
                l12 = nu2_paths[p2_idx][2]
                cg = o3.wigner_3j(l12, l3, L_, dtype=torch.float64) * math.sqrt(2 * L_ + 1)
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
                off, cnt = reg_cg(l12, l3, L_)
                nu3_paths.append((p2_idx, l3, L_, c3, off, cnt))

    l2_paths = []
    l2_by_L = {L: [] for L in range(lout2 + 1)}
    if has_l2:
        for l1 in range(lfeat2 + 1):
            for l2 in range(lmsg + 1):
                for L in range(min(lout2, l1 + l2) + 1):
                    if L >= abs(l1 - l2) and (l1 + l2 + L) % 2 == 0:
                        p_in_L = len(l2_by_L[L])
                        l2_by_L[L].append((l1, l2))
                        off, cnt = reg_cg(l1, l2, L)
                        l2_paths.append((l1, l2, L, p_in_L, off, cnt))

    bisp_table = []
    bisp_idx: list[int] = []
    bisp_val: list[float] = []
    if has_l2 and r_bisp > 0:
        for L1 in range(lout2 + 1):
            for L2 in range(L1, lout2 + 1):
                for L3 in range(L2, lout2 + 1):
                    if L3 == 0:
                        continue
                    if L3 <= L1 + L2 and (L1 + L2 + L3) % 2 == 0:
                        if all(len(l2_by_L[Lx]) > 0 for Lx in (L1, L2, L3)):
                            w3 = (
                                o3.wigner_3j(L1, L2, L3, dtype=torch.float64)
                                * math.sqrt((2 * L1 + 1) * (2 * L2 + 1) * (2 * L3 + 1)) ** (1.0 / 3.0)
                            ).numpy()
                            off = len(bisp_idx)
                            ii, vv = _pack_sparse_tensor(w3)
                            bisp_idx.extend(ii)
                            bisp_val.extend(vv)
                            bisp_table.append((L1, L2, L3, off, len(ii)))

    return nu2_paths, nu3_paths, l2_paths, l2_by_L, cg_idx, cg_val, bisp_table, bisp_idx, bisp_val


def normalize_ilmlip_cfg(ckpt: dict, energy_scale: float | None = None) -> dict:
    """Normalize either a clean ``cfg`` dict or a Phase-2 probe ``args`` dict into canonical config."""
    sd = ckpt["state_dict"]
    if "cfg" in ckpt:
        cfg = dict(ckpt["cfg"])
    elif "args" in ckpt:
        a = dict(ckpt["args"])
        zz = sd["zz_pair"].detach().cpu().numpy()
        S = int(round(math.sqrt(zz.size)))
        zs = [int(round(math.sqrt(float(zz[i * S + i])))) for i in range(S)]
        taper = a["taper"]
        n_l = [int(x) for x in taper.split(",")] if isinstance(taper, str) else [int(x) for x in taper]
        is_lrsk = bool(a.get("lrsk", False)) or ("W_lr_rad" in sd)
        lfeat = int(a["lfeat"])
        if is_lrsk:
            c0 = int(a.get("c0", 12))
            c1 = int(a.get("c1", 8)) if lfeat >= 1 else 0
            c2 = int(a.get("c2", 4)) if lfeat >= 2 else 0
            lmsg = min(2 if c2 > 0 else (1 if c1 > 0 else 0), lfeat)
            lfeat2 = 0
            lout2 = 0
        else:
            c0 = c1 = c2 = 0
            lmsg = int(a["lmsg"])
            lfeat2 = int(lfeat if a.get("lfeat2") is None else a["lfeat2"])
            lout2 = int(min(lfeat2 + lmsg, max(2, lfeat)) if a.get("lout2") is None else a["lout2"])
        esc = energy_scale if energy_scale is not None else ckpt.get("energy_scale")
        if esc is None and a.get("train") and Path(a["train"]).is_file():
            from ase.io import iread
            f_all = np.concatenate([at.arrays["REF_forces"].ravel() for at in iread(a["train"], ":", format="extxyz")])
            esc = float(np.sqrt(np.mean(f_all ** 2)))
        if esc is None:
            esc = 1.0
        cfg = {
            "zs": zs,
            "r_max": float(a["r_max"]),
            "r_max2": float(a["r_max"] if a.get("r_max2") is None else a["r_max2"]),
            "lmax": int(a["lmax"]),
            "lfeat": lfeat,
            "lmsg": lmsg,
            "lfeat2": lfeat2,
            "lout2": lout2,
            "n_l": n_l,
            "n_base": int(a["n_base"]),
            "n_base2": int(a["n_base"] if a.get("n_base2") is None else a["n_base2"]),
            "n_msg": 0 if is_lrsk else int(a["n_msg"]),
            "r_gram": 0 if is_lrsk else int(a["r_gram"]),
            "r_bisp": 0 if is_lrsk else int(a["r_bisp"]),
            "readout_depth": int(a["readout_depth"]),
            "readout_width": int(a["readout_width"]),
            "decoupled_l2": False if is_lrsk else bool(a.get("decoupled_l2", False)),
            "remove_backflow": False if is_lrsk else bool(a.get("remove_backflow", True)),
            "use_zbl_switch": bool(a.get("use_zbl_switch", True)),
            "film_radials": bool(a.get("film_radials", "C1_R" in sd)),
            "energy_scale": float(esc),
            "kappa": float(a.get("kappa", 16.0)),
            "a0": float(a.get("a0", 0.05)),
            "z_c": float(a.get("z_c", 10.0)),
            "max_order": int(a.get("max_order", 3)),
            "radial_warp": float(a.get("radial_warp", 0.0)),
            "lrsk": is_lrsk,
            "lrsk_c0": c0,
            "lrsk_c1": c1,
            "lrsk_c2": c2,
            "lrsk_sym": bool(a.get("sym", False)) or (is_lrsk and ("W_qA" in sd and "W_qB" not in sd)) or (is_lrsk and ("W_g1A" in sd and "W_g1B" not in sd)),
            "lrsk_pol_field": bool(a.get("pol_field", "pol_W0.weight" in sd)) if is_lrsk else False,
            "lrsk_field_mlp_width": int(a.get("field_mlp_width", 32)) if is_lrsk else 0,
            "lrsk_pol_gram": bool(a.get("pol_gram", False)) if is_lrsk else False,
            "lrsk_sw_mp": bool(a.get("sw_mp", False)) if is_lrsk else False,
            "lrsk_mp_cap": float(a.get("mp_cap", 0.0)) if is_lrsk else 0.0,
            "lrsk_ef_cap": float(a.get("ef_cap", 0.0)) if is_lrsk else 0.0,
            "lrsk_pair_mlp_width": int(a.get("pair_mlp_width", 0)) if is_lrsk else 0,
        }
    else:
        raise KeyError("ILMlip checkpoint missing both 'cfg' and 'args'")
    if energy_scale is not None:
        cfg["energy_scale"] = float(energy_scale)
    return cfg


def export_ilmlip_checkpoint(ckpt_path: str | Path, output_path: str | Path, energy_scale: float | None = None) -> Path:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = normalize_ilmlip_cfg(ckpt, energy_scale=energy_scale)
    sd = ckpt["state_dict"]

    zs = [int(z) for z in cfg["zs"]]
    S = len(zs)
    r_max = float(cfg["r_max"])
    r_max2 = float(cfg.get("r_max2", r_max) if cfg.get("r_max2") is not None else r_max)
    lmax = int(cfg["lmax"])
    lfeat = int(cfg["lfeat"])
    lmsg = int(cfg["lmsg"])
    lfeat2 = int(cfg.get("lfeat2", lfeat))
    lout2 = int(cfg.get("lout2", min(lfeat2 + lmsg, max(2, lfeat))))
    n_l = [int(x) for x in cfg["n_l"]]
    n_base = int(cfg["n_base"])
    n_base2 = int(cfg.get("n_base2", n_base))
    n_msg = int(cfg["n_msg"])
    r_gram = int(cfg["r_gram"])
    r_bisp = int(cfg["r_bisp"])
    ro_depth = int(cfg["readout_depth"])
    ro_width = int(cfg["readout_width"])
    decoupled_l2 = bool(cfg.get("decoupled_l2", False))
    remove_backflow = bool(cfg.get("remove_backflow", True))
    use_zbl_switch = bool(cfg.get("use_zbl_switch", True))
    film = bool(cfg.get("film_radials", "C1_R" in sd))
    energy_scale = float(cfg.get("energy_scale", 1.0))
    kappa = float(cfg.get("kappa", 16.0))
    a0 = float(cfg.get("a0", 0.05))
    z_c = float(cfg.get("z_c", 10.0))
    radial_warp = float(cfg.get("radial_warp", 0.0))
    is_lrsk = bool(cfg.get("lrsk", False)) or ("W_lr_rad" in sd)
    cfg["lrsk"] = is_lrsk

    has_l2 = (n_msg > 0) and (not is_lrsk)
    if not has_l2:
        remove_backflow = False
        r_gram = 0
        r_bisp = 0
    nu2_paths, nu3_paths, l2_paths, l2_by_L, cg_idx, cg_val, bisp_table, bisp_idx, bisp_val = _build_paths(cfg)

    need_bcat = has_l2 or is_lrsk
    msg_in_dim = [n_l[l] for l in range(lmsg + 1)] if need_bcat else []
    if need_bcat:
        for _, _, L, c, _, _ in nu2_paths:
            if L <= lmsg:
                msg_in_dim[L] += c
        for _, _, L, c, _, _ in nu3_paths:
            if L <= lmsg:
                msg_in_dim[L] += c
    n_inv1 = (
        n_l[0]
        + sum(c for _, _, L, c, _, _ in nu2_paths if L == 0)
        + sum(c for _, _, L, c, _, _ in nu3_paths if L == 0)
    )
    n_inv2 = 0
    if has_l2:
        n_inv2 = len(l2_by_L[0]) * n_msg
        for L in range(1, lout2 + 1):
            if len(l2_by_L[L]) > 0 and r_gram > 0:
                n_inv2 += r_gram * (r_gram + 1) // 2
        n_inv2 += len(bisp_table) * r_bisp
    n_feat = n_inv1 + n_inv2
    D_A1 = sum((2 * l + 1) * n_l[l] for l in range(lmax + 1))
    D_A2 = sum((2 * L + 1) * len(l2_by_L[L]) * n_msg for L in range(lout2 + 1)) if has_l2 else 0

    # Fold C1_eff, C2_eff, and exact pre-contracted C_bf in float64 before casting to float32
    C1_eff = _compute_c_eff(sd, "C1", S, film)  # [S*S, n_tot1, n_base]
    l1_offs = []
    off = 0
    for nl in n_l:
        l1_offs.append((off, off + nl))
        off += nl

    tensors: dict[str, np.ndarray] = {
        "r_max": np.array([r_max], dtype=np.float64),
        "e0": sd["e0"].detach().cpu().to(torch.float64).numpy(),
        "scalars": np.array([
            float(sd["sigma0"].item()),
            float(sd["sigma1"].item()) if "sigma1" in sd else 1.0,
            a0,
            kappa,
            energy_scale,
            z_c,
        ], dtype=np.float64),
        "rs_pair": sd["rs_pair"].detach().cpu().to(torch.float64).numpy(),
        "zz_pair": sd["zz_pair"].detach().cpu().to(torch.float64).numpy(),
        "au_pair": sd["au_pair"].detach().cpu().to(torch.float64).numpy(),
        "C1_eff": C1_eff.astype(np.float32),
        "nu2_table": np.array(nu2_paths, dtype=np.int32).reshape(len(nu2_paths), 6),
        "nu3_table": np.array(nu3_paths, dtype=np.int32).reshape(len(nu3_paths), 6) if nu3_paths else np.zeros((0, 6), dtype=np.int32),
        "l2_paths": np.array(l2_paths, dtype=np.int32).reshape(len(l2_paths), 6) if l2_paths else np.zeros((0, 6), dtype=np.int32),
        "cg_idx": np.array(cg_idx, dtype=np.int32),
        "cg_val": np.array(cg_val, dtype=np.float32),
    }
    if abs(r_max2 - r_max) > 1e-6:
        tensors["r_max2"] = np.array([r_max2], dtype=np.float64)
    if radial_warp > 0.0:
        tensors["radial_warp"] = np.array([radial_warp], dtype=np.float64)

    if has_l2:
        if decoupled_l2:
            C2_eff = _compute_c_eff(sd, "C2", S, film)  # [S*S, (lfeat2+1)*n_msg, n_base2]
            n_base2 = int(C2_eff.shape[-1])
        else:
            parts = []
            for l in range(lfeat2 + 1):
                s0, _ = l1_offs[l]
                parts.append(C1_eff[:, s0 : s0 + n_msg, :n_base2])
            C2_eff = np.concatenate(parts, axis=1)
        tensors["C2_eff"] = C2_eff.astype(np.float32)

        W_msg_f64 = [sd[f"W_msg.{l}"].detach().cpu().to(torch.float64).numpy() for l in range(lmsg + 1)]  # [n_msg, msg_in_dim]

        if remove_backflow:
            # C_bf[z_i*S + z_j, l*n_msg + q, k] = (-1)^l * sum_c W_msg[l][q, c] * C1_eff[z_j*S + z_i, s0(l)+c, k]
            C_bf = np.zeros((S * S, (lmsg + 1) * n_msg, n_base), dtype=np.float64)
            for zi in range(S):
                for zj in range(S):
                    p_fwd = zi * S + zj
                    p_rev = zj * S + zi
                    for l in range(lmsg + 1):
                        s0, s1 = l1_offs[l]
                        sign = -1.0 if (l % 2 == 1) else 1.0
                        w1 = W_msg_f64[l][:, : n_l[l]]  # [n_msg, n_l[l]]
                        C_bf[p_fwd, l * n_msg : (l + 1) * n_msg, :] = sign * (w1 @ C1_eff[p_rev, s0:s1, :])
            tensors["C_bf"] = C_bf.astype(np.float32)

        if bisp_table:
            tensors["bisp_table"] = np.array(bisp_table, dtype=np.int32).reshape(len(bisp_table), 5)
            tensors["bisp_idx"] = np.array(bisp_idx, dtype=np.int32)
            tensors["bisp_val"] = np.array(bisp_val, dtype=np.float32)

        for l in range(lmsg + 1):
            # Store transposed [msg_in_dim[l], n_msg] for direct row-major GEMM B_cat @ W_msg_T
            tensors[f"W_msg_{l}"] = np.ascontiguousarray(W_msg_f64[l].T.astype(np.float32))

        for L in range(lout2 + 1):
            if not l2_by_L[L]:
                continue
            if L >= 1 and r_gram > 0:
                tensors[f"W_G_{L}"] = sd[f"W_G.{L}"].detach().cpu().to(torch.float32).numpy()
            if r_bisp > 0:
                tensors[f"W_B_{L}"] = sd[f"W_B.{L}"].detach().cpu().to(torch.float32).numpy()

    mu_x = sd["mu_x"].detach().cpu().to(torch.float64).numpy()
    std_x = sd["std_x"].detach().cpu().to(torch.float64).numpy()
    inv_std_x = 1.0 / np.maximum(std_x, 1e-7)
    tensors["mu_x"] = mu_x.astype(np.float32)
    tensors["inv_std_x"] = inv_std_x.astype(np.float32)

    ro_b0 = sd["readout_layers.0.bias"].detach().cpu().to(torch.float64).numpy()
    alpha_Z = sd["alpha_Z"].detach().cpu().to(torch.float64).numpy() + ro_b0[None, :]
    u_Z = sd["u_Z"].detach().cpu().to(torch.float64).numpy() * energy_scale
    tensors["alpha_Z"] = alpha_Z.astype(np.float32)
    tensors["u_Z"] = u_Z.astype(np.float32)

    for d in range(ro_depth):
        w = sd[f"readout_layers.{d}.weight"].detach().cpu().to(torch.float64).numpy()  # [out, in]
        tensors[f"ro_W_{d}"] = np.ascontiguousarray(w.T.astype(np.float32))  # [in, out]
        if d > 0:
            tensors[f"ro_b_{d}"] = sd[f"readout_layers.{d}.bias"].detach().cpu().to(torch.float32).numpy()

    lrsk_meta: dict[str, int] = {}
    if is_lrsk:
        c0 = int(cfg["lrsk_c0"])
        c1 = int(cfg["lrsk_c1"])
        c2 = int(cfg["lrsk_c2"])
        lrsk_sym = bool(cfg.get("lrsk_sym", False)) or ("W_qA" in sd and "W_qB" not in sd) or ("W_g1A" in sd and "W_g1B" not in sd)
        pol_field = bool(cfg.get("lrsk_pol_field", "pol_W0.weight" in sd))
        pol_width = int(cfg.get("lrsk_field_mlp_width", 32)) if pol_field else 0
        pol_gram = bool(cfg.get("lrsk_pol_gram", False))
        if pol_field and (not pol_gram) and "pol_W0.weight" in sd:
            gram_in = c0 + (c1 * (c1 + 1)) // 2 + c1 * c1 + ro_width
            if int(sd["pol_W0.weight"].shape[1]) == gram_in and gram_in != (c0 + 2 * c1 + ro_width):
                pol_gram = True
        sw_mp = bool(cfg.get("lrsk_sw_mp", False))
        mp_cap = float(cfg.get("lrsk_mp_cap", 0.0))
        ef_cap = float(cfg.get("lrsk_ef_cap", 0.0))
        pair_mlp_width = int(cfg.get("lrsk_pair_mlp_width", 0))
        if pair_mlp_width == 0 and "bo_Wx" in sd:
            pair_mlp_width = int(sd["bo_Wx"].shape[0])
        if c1 > 0:
            std_b1 = np.maximum(sd["std_b1"].detach().cpu().to(torch.float64).numpy(), 1e-7)
            if lrsk_sym:
                wp1 = sd["W_p1A"].detach().cpu().to(torch.float64).numpy() / std_b1[None, :]  # [c1, bcat_dim[1]]
            else:
                wp1 = np.concatenate([
                    sd["W_p1A"].detach().cpu().to(torch.float64).numpy(),
                    sd["W_p1B"].detach().cpu().to(torch.float64).numpy(),
                ], axis=0) / std_b1[None, :]  # [2*c1, bcat_dim[1]]
            tensors["lrsk_W_p1"] = np.ascontiguousarray(wp1.T.astype(np.float32))
        if c2 > 0:
            std_b2 = np.maximum(sd["std_b2"].detach().cpu().to(torch.float64).numpy(), 1e-7)
            if lrsk_sym:
                wp2 = sd["W_p2A"].detach().cpu().to(torch.float64).numpy() / std_b2[None, :]  # [c2, bcat_dim[2]]
            else:
                wp2 = np.concatenate([
                    sd["W_p2A"].detach().cpu().to(torch.float64).numpy(),
                    sd["W_p2B"].detach().cpu().to(torch.float64).numpy(),
                ], axis=0) / std_b2[None, :]  # [2*c2, bcat_dim[2]]
            tensors["lrsk_W_p2"] = np.ascontiguousarray(wp2.T.astype(np.float32))

        # Pack species-indexed multipole heads [S, n_head_out, ro_width] and transposed [S, ro_width, n_head_out]
        w_parts = [u_Z[:, None, :]]  # [S, 1, ro_width], already scaled by energy_scale!
        b_parts = [np.zeros((S, 1), dtype=np.float64)]
        if c0 > 0:
            w_parts.append(sd["W_qA"].detach().cpu().to(torch.float64).numpy())
            b_parts.append(sd["b_qA"].detach().cpu().to(torch.float64).numpy())
            if not lrsk_sym:
                w_parts.append(sd["W_qB"].detach().cpu().to(torch.float64).numpy())
                b_parts.append(sd["b_qB"].detach().cpu().to(torch.float64).numpy())
        if c1 > 0:
            w_parts.append(sd["W_g1A"].detach().cpu().to(torch.float64).numpy())
            b_parts.append(1.0 + sd["b_g1A"].detach().cpu().to(torch.float64).numpy())
            if not lrsk_sym:
                w_parts.append(sd["W_g1B"].detach().cpu().to(torch.float64).numpy())
                b_parts.append(1.0 + sd["b_g1B"].detach().cpu().to(torch.float64).numpy())
        if c2 > 0:
            w_parts.append(sd["W_g2A"].detach().cpu().to(torch.float64).numpy())
            b_parts.append(1.0 + sd["b_g2A"].detach().cpu().to(torch.float64).numpy())
            if not lrsk_sym:
                w_parts.append(sd["W_g2B"].detach().cpu().to(torch.float64).numpy())
                b_parts.append(1.0 + sd["b_g2B"].detach().cpu().to(torch.float64).numpy())
        W_head_T = np.concatenate(w_parts, axis=1)  # [S, n_head_out, ro_width]
        b_head = np.concatenate(b_parts, axis=1)    # [S, n_head_out]
        W_head = np.transpose(W_head_T, (0, 2, 1))  # [S, ro_width, n_head_out]
        tensors["lrsk_W_head"] = np.ascontiguousarray(W_head.astype(np.float32))
        tensors["lrsk_W_head_T"] = np.ascontiguousarray(W_head_T.astype(np.float32))
        tensors["lrsk_b_head"] = np.ascontiguousarray(b_head.astype(np.float32))

        W_rad64 = sd["W_lr_rad"].detach().cpu().to(torch.float64).numpy()  # [S*S, n_lr_terms, n_base2]
        n_lr_terms = int(W_rad64.shape[1])
        n_base2 = int(W_rad64.shape[2])
        if lrsk_sym:
            rev_p = np.array([zj * S + zi for zi in range(S) for zj in range(S)], dtype=np.int64)
            c01, c02, c12 = min(c0, c1), min(c0, c2), min(c1, c2)
            sym_mask = np.zeros(n_lr_terms, dtype=bool)
            o = 0
            sym_mask[o] = True; o += 1
            sym_mask[o : o + c0] = True; o += c0
            o += c01
            sym_mask[o : o + 2 * c1] = True; o += 2 * c1
            o += c02 + 2 * c12
            sym_mask[o : o + 3 * c2] = True; o += 3 * c2
            W_rad64 = np.where(sym_mask[None, :, None], 0.5 * (W_rad64 + W_rad64[rev_p]), W_rad64)
        tensors["lrsk_W_rad"] = np.ascontiguousarray(W_rad64.astype(np.float32))

        if pol_field:
            pw0 = sd["pol_W0.weight"].detach().cpu().to(torch.float64).numpy()  # [pol_width, in_dim]
            tensors["lrsk_pol_W0"] = np.ascontiguousarray(pw0.T.astype(np.float32))
            tensors["lrsk_pol_b0"] = sd["pol_W0.bias"].detach().cpu().to(torch.float32).numpy()
            pu = sd["pol_u"].detach().cpu().to(torch.float64).numpy() * energy_scale
            tensors["lrsk_pol_u"] = np.ascontiguousarray(pu.astype(np.float32))

        if mp_cap > 0.0:
            tensors["lrsk_mp_cap"] = np.array([mp_cap], dtype=np.float64)
        if ef_cap > 0.0:
            tensors["lrsk_ef_cap"] = np.array([ef_cap], dtype=np.float64)
        if pair_mlp_width > 0:
            tensors["lrsk_bo_Wx"] = np.ascontiguousarray(sd["bo_Wx"].detach().cpu().to(torch.float32).numpy())
            tensors["lrsk_bo_Wr"] = np.ascontiguousarray(sd["bo_Wr"].detach().cpu().to(torch.float32).numpy())
            tensors["lrsk_bo_b"]  = np.ascontiguousarray(sd["bo_b"].detach().cpu().to(torch.float32).numpy())
            tensors["lrsk_bo_v"]  = np.ascontiguousarray(sd["bo_v"].detach().cpu().to(torch.float32).numpy())

        lrsk_meta = {
            "lrsk": 1,
            "lrsk_sym": int(lrsk_sym),
            "lrsk_c0": c0,
            "lrsk_c1": c1,
            "lrsk_c2": c2,
            "lrsk_pol_field": int(pol_field),
            "lrsk_field_mlp_width": pol_width,
            "lrsk_pol_gram": int(pol_gram),
            "lrsk_sw_mp": int(sw_mp),
            "lrsk_n_terms": n_lr_terms,
            "lrsk_pair_mlp_width": pair_mlp_width,
        }

    meta = {
        "arch": "ilmlip",
        "exporter_version": EXPORTER_VERSION,
        "S": S,
        "Z": zs,
        "lmax": lmax,
        "lfeat": lfeat,
        "lmsg": lmsg,
        "lfeat2": lfeat2,
        "lout2": lout2,
        "n_l": n_l,
        "n_base": n_base,
        "n_base2": n_base2,
        "n_msg": n_msg,
        "r_gram": r_gram,
        "r_bisp": r_bisp,
        "readout_depth": ro_depth,
        "readout_width": ro_width,
        "decoupled_l2": int(decoupled_l2),
        "remove_backflow": int(remove_backflow),
        "use_zbl_switch": int(use_zbl_switch),
        "n_nu2": len(nu2_paths),
        "n_nu3": len(nu3_paths),
        "n_l2_paths": len(l2_paths),
        "n_bisp": len(bisp_table),
        "n_inv1": n_inv1,
        "n_inv2": n_inv2,
        "n_feat": n_feat,
        "D_A1": D_A1,
        "D_A2": D_A2,
        "msg_in_dim": msg_in_dim,
        **lrsk_meta,
    }

    out = Path(output_path)
    write_bundle(out, meta, tensors)
    return out


def main():
    ap = argparse.ArgumentParser(description="Export an ILMlip-Student .pt checkpoint to a .loup bundle.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = export_ilmlip_checkpoint(a.checkpoint, a.output)
    print(f"Exported arch=ilmlip bundle: {out} ({out.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
