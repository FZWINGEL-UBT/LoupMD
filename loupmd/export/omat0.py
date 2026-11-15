#!/usr/bin/env python3
"""Export mace-omat-0-medium.model + certified radial tables for a given set of atomic numbers into a .loup bundle
(the container of loupmd/export/bundle.py; arch=omat0_medium_spec, read by src/model_loader.cu).

Usage:
  python -m loupmd.export.omat0 --elements 8,22,38 --output /tmp/omat0_SrTiO3.loup
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

FASTMACE_ROOT = next(
    (p for p in (Path("/mnt/d/Margraf/FastMACE"), Path("/mnt/homes/zwingel/FastMACE")) if p.exists()),
    Path("/mnt/d/Margraf/FastMACE"),
)
sys.path.insert(0, str(FASTMACE_ROOT))

import numpy as np
import torch
from mace.cli.convert_e3nn_cueq import run as convert_e3nn_cueq

from fastmace.kernels.cuda.sc_factored import factor, fold_weights
from fastmace.kernels.cuda.sc_generated import merged_terms, shape_of
from fastmace.kernels.triton import compose_skip
from fastmace.rewrites import radial as rad
from fastmace.rewrites.generic.modules import fc_last_weight, fc_split
from fastmace.rewrites.linear.modules import block_linear, raw_blocks
from fastmace.rewrites.radial import _GenericChain, power_basis, spline as rspline


from .bundle import write_bundle  # noqa: E402

EXPORTER_VERSION = 4
ARCH = "omat0_medium_spec"
# What src/generated_kernels.cuh and the specialised engine assume. The full signature (kernel_signature below)
# adds the two factored product programs, so gen_kernels.py and this exporter must agree on them too.
KERNEL_SIGNATURE = "omat0m:C128:h0e+1o/0e:corr3:sh3:paths10"


def product_signature(prog, feats, sh) -> str:
    """Hash of one factored symmetric-contraction program (as export_generic.product_signature): the feature basis the
    generated product kernel evaluates and the folded weights multiply. FastMACE's factor() chooses it; the kernels
    get it at generation time and the weights at export time, so both sides record it."""
    h = hashlib.sha256(repr((sh[0], sh[1], sh[2], prog.nodes, [sorted(f.items()) if f else None for f in feats])).encode())
    return h.hexdigest()[:16]


def kernel_signature(product_sigs) -> str:
    """The string compared by src/model_loader.cu: architecture literal plus the hashes of products 0 and 1."""
    return KERNEL_SIGNATURE + "".join(f":p{i}={s}" for i, s in enumerate(product_sigs))


def check_architecture(m):
    """The specialised engine's assumptions, checked on the model (raises with the first mismatch)."""
    names = [type(it).__name__ for it in m.interactions]
    want = ["RealAgnosticDensityInteractionBlock", "RealAgnosticDensityResidualInteractionBlock"]
    if names != want:
        raise ValueError(f"interactions {names}: the specialised path needs {want}")
    checks = [
        (str(m.interactions[0].hidden_irreps), "128x0e+128x1o"),
        (str(m.interactions[1].hidden_irreps), "128x0e"),
        (str(m.interactions[0].node_feats_irreps), "128x0e"),
        (str(m.spherical_harmonics.irreps_out), "1x0e+1x1o+1x2e+1x3o"),
        (int(m.interactions[1].conv_tp.weight_numel), 1280),
        (int(m.interactions[0].conv_tp.weight_numel), 512),
        (len(m.readouts), 2),
        (str(m.readouts[1].linear_1.irreps_out), "16x0e"),
    ]
    for got, want in checks:
        if got != want:
            raise ValueError(f"model has {got}, the specialised path needs {want}")
    heads = list(getattr(m, "heads", None) or ["default"])
    if len(heads) != 1:
        raise ValueError(f"{len(heads)} heads: the specialised path takes single-head models")


def _extract_skip_blocks(skip_tp, cols, dtype=torch.float64):
    """Extract [S, mul_in, mul_out] weight per (i, j) block via SpeciesSkip."""
    from fastmace.rewrites.generic.modules import species_skip
    rep = species_skip(skip_tp, torch.as_tensor(cols, dtype=torch.long))
    return [(i, j, getattr(rep, f"ws{idx}").detach().to(dtype)) for idx, (i, j) in enumerate(rep.pairs)]
def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_model(model_path: str, atomic_nums: list[int], out_path: Path):
    atomic_nums = sorted({int(z) for z in atomic_nums})
    S = len(atomic_nums)
    sym_pairs = [(s, t) for s in range(S) for t in range(s, S)]
    P_sym = len(sym_pairs)

    m64 = torch.load(model_path, map_location="cpu", weights_only=False).double()
    check_architecture(m64)
    m32 = torch.load(model_path, map_location="cpu", weights_only=False).float()
    mc64 = convert_e3nn_cueq(m64)
    mc32 = convert_e3nn_cueq(m32)

    all_z = [int(z) for z in m64.atomic_numbers.tolist()]
    cols = [all_z.index(z) for z in atomic_nums]
    r_max = float(m64.r_max)
    scale = float(m64.scale_shift.scale.reshape(-1)[0].item())
    # ScaleShiftMACE adds `shift` to every atom's interaction energy: fold it into E0.
    shift = float(m64.scale_shift.shift.reshape(-1)[0].item())
    e0 = m64.atomic_energies_fn.atomic_energies.reshape(-1, len(m64.atomic_numbers))[0][cols]
    e0 = e0.detach().cpu().numpy().astype(np.float64) + shift

    # ------------------------------------------------------------------
    # Interaction 0 radial table (Chebyshev + power basis on symmetric pairs)
    # ------------------------------------------------------------------
    chain0_64 = _GenericChain(m64, m64.interactions[0], cols, torch.float64, torch.device("cpu"))
    chain0_32 = _GenericChain(m32, m32.interactions[0], cols, torch.float32, torch.device("cpu"))
    tables0 = rad.fit(chain0_64, sym_pairs, r_max, table_dtype=torch.float32)
    # FastMACE's acceptance test for this table (its own self-check applies the same): over 12,000 lengths per pair
    # the float32 table must be no further from the float64 network than the float32 network itself. rad.fit alone
    # does not ensure that: it accepts a piece at its deepest level whatever the error, and the real table has such
    # pieces (fit_error 4e-5 against a tolerance of 1e-7), so fit_error is not the criterion.
    cert0 = rad.certify(SimpleNamespace(**tables0), chain0_64, chain0_32, sym_pairs, r_max)
    if not cert0["worst"] <= 1.0:
        raise RuntimeError(f"interaction 0 radial table: certificate {cert0['worst']:.3f} > 1 "
                           "(the table is less accurate than the float32 network)")
    coef0 = tables0["coef"].contiguous()  # [P_sym, pieces0, 6, 66]
    pcoef0 = power_basis(coef0).contiguous()  # [P_sym, pieces0, 6, 66]
    starts0 = tables0["starts"].contiguous()
    scales0 = tables0["scales"].contiguous()
    bins0 = tables0["bins"].contiguous()
    r_lo0 = float(tables0["r_lo"])
    bin_scale0 = float(tables0["bin_scale"])
    pieces0 = int(coef0.shape[1])
    n_bins0 = int(bins0.shape[0])

    # ------------------------------------------------------------------
    # Interaction 0 COMPOSE_SKIP weights C0_l [S, S*64, 128]
    # ------------------------------------------------------------------
    with torch.no_grad():
        eye = torch.zeros((S, len(all_z)), dtype=torch.float64)
        for s_idx, c_idx in enumerate(cols):
            eye[s_idx, c_idx] = 1.0
        # node_embedding + inter[0].linear_up
        emb_w = list(raw_blocks(block_linear(mc64.node_embedding.linear, grouped=True)))[0][2].double()
        lup0_w = list(raw_blocks(block_linear(mc64.interactions[0].linear_up, grouped=True)))[0][2].double()
        x_sp = eye @ emb_w @ lup0_w  # [S, 128]

        w_last0 = fc_last_weight(fc_split(m64.interactions[0].conv_tp_weights)[1]).detach().double()  # [512, 64]
        lin0_blocks = {j: w.double() for _, j, w in raw_blocks(block_linear(mc64.interactions[0].linear, grouped=True))}
        skip0_blocks_list = _extract_skip_blocks(mc64.interactions[0].skip_tp, cols, torch.float64)
        skip0_blocks = {j: w.double() for _, j, w in skip0_blocks_list}

        gammas = [1.0, 1.0, 1.0, 1.0]
        c0_list = []
        for l in range(4):
            right = (gammas[l] * x_sp[:, None, :] * w_last0[l * 128:(l + 1) * 128, :].t()[None, :, :]).reshape(S * 64, 128)
            c_mat, _ = compose_skip.composed(right, skip0_blocks[l], lin0_blocks[l])
            c0_list.append(c_mat.to(torch.float32).contiguous())  # [S, S*64, 128]

    # ------------------------------------------------------------------
    # Product 0 + folded readout 0 + folded inter[1].linear_up / skip_tp
    # ------------------------------------------------------------------
    with torch.no_grad():
        sc0 = mc64.products[0].symmetric_contractions
        sc0_32 = mc32.products[0].symmetric_contractions
        sh0 = shape_of(sc0_32)
        terms0 = merged_terms(sc0_32.etp.polynomial)
        prog0, feats0, K0 = factor(terms0, sh0[0], sh0[1], sh0[2])
        w0_raw = torch.einsum("zau,ab->zbu", sc0.weight.detach(), sc0.projection.detach())[cols].reshape(S, -1)
        w0_factored = fold_weights(w0_raw, K0, sh0[1], sh0[3]).to(torch.float32).contiguous()  # [S, 29, 128]

        p0_lin = {j: w.double() for _, j, w in raw_blocks(block_linear(mc64.products[0].linear, grouped=True))}
        ro0_w = list(raw_blocks(block_linear(mc64.readouts[0].linear, grouped=True)))[0][2].double().squeeze(-1)
        w_ro0 = (p0_lin[0] @ ro0_w * scale).to(torch.float32).contiguous()  # [128]

        lup1_blocks = {j: w.double() for _, j, w in raw_blocks(block_linear(mc64.interactions[1].linear_up, grouped=True))}
        W_up_0 = (p0_lin[0] @ lup1_blocks[0]).to(torch.float32).contiguous()  # [128, 128]
        W_up_1 = (p0_lin[1] @ lup1_blocks[1]).to(torch.float32).contiguous()  # [128, 128]

        # inter[1].skip_tp [S, 128, 128] folded with p0_lin[0] on left AND readout_tail on right!
        skip1_blocks_list = _extract_skip_blocks(mc64.interactions[1].skip_tp, cols, torch.float64)
        W_skip1_raw = skip1_blocks_list[0][2].double()  # [S, 128, 128]

        # Fold prod[1].linear + readout[1].linear_1 + scale_shift + Activation(1.6791767923989418)
        p1_lin = list(raw_blocks(block_linear(mc64.products[1].linear, grouped=True)))[0][2].double()  # [128, 128]
        ro1_lin1 = list(raw_blocks(block_linear(mc64.readouts[1].linear_1, grouped=True)))[0][2].double()  # [128, 16]
        ro1_lin2 = list(raw_blocks(block_linear(mc64.readouts[1].linear_2, grouped=True)))[0][2].double().squeeze(-1)  # [16]
        act_cst = float(mc64.readouts[1].non_linearity.acts[0].cst)

        W_p1_ro1 = (p1_lin @ ro1_lin1).to(torch.float32).contiguous()  # [128, 16]
        W_skip1_16 = torch.einsum("ab,sbc,cd->sad", p0_lin[0], W_skip1_raw, ro1_lin1).to(torch.float32).contiguous()  # [S, 128, 16]
        W_ro1_2 = (ro1_lin2 * (act_cst * scale)).to(torch.float32).contiguous()  # [16]

        # Product 1 factored weights
        sc1 = mc64.products[1].symmetric_contractions
        sc1_32 = mc32.products[1].symmetric_contractions
        sh1 = shape_of(sc1_32)
        terms1 = merged_terms(sc1_32.etp.polynomial)
        prog1, feats1, K1 = factor(terms1, sh1[0], sh1[1], sh1[2])
        w1_raw = torch.einsum("zau,ab->zbu", sc1.weight.detach(), sc1.projection.detach())[cols].reshape(S, -1)
        w1_factored = fold_weights(w1_raw, K1, sh1[1], sh1[3]).to(torch.float32).contiguous()  # [S, 13, 128]

        # inter[1].linear grouped block weights
        lin1 = block_linear(mc64.interactions[1].linear, grouped=True)
        W_lin1 = [getattr(lin1, f"weight_{k}").detach().to(torch.float32).contiguous() for k in range(4)]

        # Pair repulsion (ZBL) if present
        has_zbl = int(bool(getattr(m64, "pair_repulsion", False)))
        zbl_c = np.zeros(4, dtype=np.float32)
        zbl_p = 5.0
        zbl_r_max_max = 0.0
        zbl_r_cut = np.zeros(P_sym, dtype=np.float32)
        zbl_inv_a = np.zeros(P_sym, dtype=np.float32)
        zbl_pref = np.zeros(P_sym, dtype=np.float32)
        if has_zbl:
            zfn = m64.pair_repulsion_fn
            zbl_c = zfn.c.detach().cpu().numpy().astype(np.float32)
            zbl_p = float(zfn.p.item())
            a_exp = float(zfn.a_exp.item())
            a_pref = float(zfn.a_prefactor.item())
            cov = zfn.covalent_radii.detach().cpu().numpy().astype(np.float64)
            for p_idx, (sa, sb) in enumerate(sym_pairs):
                zu = float(atomic_nums[sa])
                zv = float(atomic_nums[sb])
                rc = float(cov[int(atomic_nums[sa])] + cov[int(atomic_nums[sb])])
                zbl_r_cut[p_idx] = rc
                if rc > zbl_r_max_max:
                    zbl_r_max_max = rc
                zbl_inv_a[p_idx] = (zu ** a_exp + zv ** a_exp) / (a_pref * 0.529)
                zbl_pref[p_idx] = scale * 0.5 * 14.3996 * zu * zv

    # ------------------------------------------------------------------
    # Interaction 1 cubic Hermite + PACKED8 + density_1 Hermite spline
    # ------------------------------------------------------------------
    chain1_64 = _GenericChain(m64, m64.interactions[1], cols, torch.float64, torch.device("cpu"))
    chain1_32 = _GenericChain(m32, m32.interactions[1], cols, torch.float32, torch.device("cpu"))
    spline_res, cert1 = rspline.fit(
        chain1_64, chain1_32, chain1_64.folded, sym_pairs,
        rad.R_LO, r_max, rad.CERTIFY_POINTS, rad.R_PHYSICAL, seed=1
    )
    # rspline.fit returns after its last round whether or not it passed (as export_generic, which also raises).
    if not cert1["worst"] <= 1.0:
        raise RuntimeError(f"interaction 1 radial spline: certificate {cert1['worst']:.3f} > 1 "
                           "(the spline is less accurate than the float32 network)")
    grid1 = spline_res["grid"]  # [(a, b, n), ...]
    coef1_raw = spline_res["coef"]  # [P_sym, intervals1, 4, 1280]
    intervals1 = int(coef1_raw.shape[1])
    # q-major [P_sym * intervals1, 1280, 4]
    spline_table_1 = coef1_raw.permute(0, 1, 3, 2).reshape(P_sym * intervals1, 1280, 4).to(torch.float32).contiguous()

    # Fit density_1 on the exact same grid1
    knots1 = rspline.grid_knots(grid1)
    h1 = knots1[1:] - knots1[:-1]
    dens_list = []
    with torch.no_grad():
        for s, t in sym_pairs:
            g, dg = chain1_64.with_derivative(knots1, s, t)
            y = g[:, -1]
            dy = dg[:, -1]
            y0, y1, m0, m1 = y[:-1], y[1:], dy[:-1] * h1, dy[1:] * h1
            dens_list.append(torch.stack([y0, m0, 3 * (y1 - y0) - 2 * m0 - m1, 2 * (y0 - y1) + m0 + m1], dim=-1))
    density_table_1 = torch.stack(dens_list, dim=0).to(torch.float32).contiguous()  # [P_sym, intervals1, 4]

    # Verify density_1 spline max error
    with torch.no_grad():
        r_chk = torch.linspace(rad.R_LO, r_max - 1e-5, 2000, dtype=torch.float32)
        iv_chk, t_chk, inv_chk = rspline.locate(r_chk, grid1)
        for p_idx, (s, t) in enumerate(sym_pairs):
            dc = density_table_1[p_idx, iv_chk]
            val = dc[:, 0] + t_chk * (dc[:, 1] + t_chk * (dc[:, 2] + t_chk * dc[:, 3]))
            slp = (dc[:, 1] + t_chk * (2.0 * dc[:, 2] + 3.0 * t_chk * dc[:, 3])) * inv_chk
            ex_v, ex_d = chain1_64.with_derivative(r_chk.double(), s, t)
            err_v = float((val.double() - ex_v[:, -1]).abs().max())
            err_d = float((slp.double() - ex_d[:, -1]).abs().max())
            assert err_v < 1e-5 and err_d < 5e-5, f"density_1 spline err too large: {err_v}, {err_d}"

    n_bands = len(grid1)
    assert 1 <= n_bands <= 8, f"{n_bands} spline bands (the engine takes at most 8)"
    starts1 = np.zeros(8, dtype=np.float32)
    ends1 = np.zeros(8, dtype=np.float32)
    inv_h1 = np.zeros(8, dtype=np.float32)
    offsets1 = np.zeros(8, dtype=np.int32)
    counts1 = np.zeros(8, dtype=np.int32)
    off = 0
    for idx_b, (a, b, cnt) in enumerate(grid1):
        starts1[idx_b] = a
        ends1[idx_b] = b
        inv_h1[idx_b] = cnt / (b - a)
        offsets1[idx_b] = off
        counts1[idx_b] = cnt
        off += cnt
    assert off == intervals1

    f32 = lambda x: np.asarray(x.numpy() if hasattr(x, "numpy") else x, dtype=np.float32)  # noqa: E731
    meta = dict(
        arch=ARCH, exporter_version=EXPORTER_VERSION,
        kernel_signature=kernel_signature([product_signature(prog0, feats0, sh0), product_signature(prog1, feats1, sh1)]),
        model_sha256=file_sha256(model_path), S=S, P_sym=P_sym, Z=atomic_nums,
        pieces0=pieces0, n_bins0=n_bins0, n_bands1=n_bands, intervals1=intervals1,
        has_zbl=has_zbl,
    )
    tensors = {
        "r_max": np.array([r_max], np.float64), "scale": np.array([scale], np.float64), "e0": e0,
        "r_lo0": np.array([r_lo0], np.float64), "bin_scale0": np.array([bin_scale0], np.float64),
        "starts0": f32(starts0), "scales0": f32(scales0), "bins0": bins0.numpy().astype(np.int32),
        "coef0": f32(coef0), "pcoef0": f32(pcoef0),
        **{f"C0_{l}": f32(c) for l, c in enumerate(c0_list)},
        "w0_factored": f32(w0_factored).reshape(S, -1, 128), "w_ro0": f32(w_ro0), "W_up_0": f32(W_up_0), "W_up_1": f32(W_up_1),
        "W_skip1_16": f32(W_skip1_16),
        "starts1": starts1, "ends1": ends1, "inv_h1": inv_h1, "offsets1": offsets1, "counts1": counts1,
        "spline_table_1": f32(spline_table_1),
        "density_table_1": f32(density_table_1),
        **{f"W_lin1_{l}": f32(w) for l, w in enumerate(W_lin1)},
        "w1_factored": f32(w1_factored).reshape(S, -1, 128), "W_p1_ro1": f32(W_p1_ro1), "W_ro1_2": f32(W_ro1_2),
        "zbl_c": f32(zbl_c), "zbl_p": np.array([zbl_p], np.float64), "zbl_r_max_max": np.array([zbl_r_max_max], np.float64),
        "zbl_r_cut": zbl_r_cut, "zbl_inv_a": zbl_inv_a, "zbl_pref": zbl_pref,
    }
    write_bundle(out_path, meta, tensors)

    print(
        f"Exported {out_path} ({out_path.stat().st_size / 1e6:.2f} MB): "
        f"Z={atomic_nums}, P_sym={P_sym}, pieces0={pieces0}, intervals1={intervals1}, "
        f"table0_cert={cert0['worst']:.3f}, spline1_cert={cert1['worst']:.3f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/fzwin/models/mace-omat-0-medium.model")
    p.add_argument("--elements", required=True, help="Comma-separated atomic numbers, e.g. 8,22,38")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    elems = [int(x) for x in args.elements.split(",") if x.strip()]
    export_model(args.model, elems, Path(args.output))


if __name__ == "__main__":
    main()
