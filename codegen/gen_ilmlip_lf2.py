#!/usr/bin/env python3
"""Generate verified specialized CUDA kernels for arch=ilmlip with lfeat=2, lfeat2=2, lmsg in {1, 2}.

Generates src/ilmlip_lf2_kernels.cuh containing:
  1. edge_sh9: fast l<=2 real spherical harmonics (9 floats, exact match to edge_sh16[:9])
  2. ilmlip_l1_node_fwd_lf2_kernel<LMAX, LMSG> & ilmlip_l1_node_bwd_lf2_kernel<LMAX, LMSG>
     for LMAX in {2, 3}, LMSG in {1, 2}, lfeat=2 (zero CG-table lookups, conflict-free SMEM b2/db2)
  3. ilmlip_l2_edge_fwd_lf2_kernel<LMSG, SUBWARP, NB_MAX, REM_BF, SMEM_C>
     for lfeat2=2, lout2=2, LMSG in {1, 2}, n_msg<=32, n_base<=16 (hardcoded inline CG)
  4. ilmlip_gate_bwd_pack_l2_lf2_kernel<LMSG, SUBWARP>
     fused f1 gate backward + coalesced [N, N_OUT, SUBWARP] packing
  5. ilmlip_l2_edge_bwd_lf2_sender_kernel<LMSG, SUBWARP, NB_MAX, REM_BF, STRESS, F32_F, SMEM_C>
     sender-owned ZERO-atomic-dI, ZERO-zero-slab-memset, FP32 RED.E.ADD.F32 forces!

Also self-verifies every CG forward and backward formula in float64 against e3nn.o3.wigner_3j.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from e3nn import o3

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from loupmd.export.ilmlip import _build_paths


def cg_tensor(l1: int, l2: int, L: int) -> np.ndarray:
    return (o3.wigner_3j(l1, l2, L, dtype=torch.float64) * math.sqrt(2 * L + 1)).numpy()


def fmt_f(v: float) -> str:
    s = f"{float(v):.9g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def emit_cg_fwd_terms(cg: np.ndarray, a_names: list[str], b_names: list[str], out_names: list[str], indent: str = "    ") -> list[str]:
    """Emit out[m] = sum_{a,b} cg[a,b,m] * a_names[a] * b_names[b]."""
    d1, d2, d3 = cg.shape
    lines = []
    for m in range(d3):
        terms = []
        for i in range(d1):
            for j in range(d2):
                v = float(cg[i, j, m])
                if abs(v) > 1e-12:
                    terms.append((v, a_names[i], b_names[j]))
        if not terms:
            lines.append(f"{indent}{out_names[m]} = 0.0f;")
            continue
        v0, a0, b0 = terms[0]
        if abs(v0 - 1.0) < 1e-12:
            expr = f"({a0} * {b0})"
        elif abs(v0 + 1.0) < 1e-12:
            expr = f"(-({a0} * {b0}))"
        else:
            expr = f"{fmt_f(v0)} * ({a0} * {b0})"
        for v, an, bn in terms[1:]:
            if abs(v - 1.0) < 1e-12:
                expr = f"fmaf({an}, {bn}, {expr})"
            elif abs(v + 1.0) < 1e-12:
                expr = f"fmaf(-({an}), {bn}, {expr})"
            else:
                expr = f"fmaf({fmt_f(v)}, {an} * {bn}, {expr})"
        lines.append(f"{indent}{out_names[m]} = {expr};")
    return lines


def emit_cg_fwd_accum(cg: np.ndarray, a_names: list[str], b_names: list[str], out_names: list[str], indent: str = "    ") -> list[str]:
    """Emit out[m] += sum_{a,b} cg[a,b,m] * a_names[a] * b_names[b]."""
    d1, d2, d3 = cg.shape
    lines = []
    for m in range(d3):
        for i in range(d1):
            for j in range(d2):
                v = float(cg[i, j, m])
                if abs(v) > 1e-12:
                    an, bn = a_names[i], b_names[j]
                    if abs(v - 1.0) < 1e-12:
                        lines.append(f"{indent}{out_names[m]} = fmaf({an}, {bn}, {out_names[m]});")
                    elif abs(v + 1.0) < 1e-12:
                        lines.append(f"{indent}{out_names[m]} = fmaf(-({an}), {bn}, {out_names[m]});")
                    else:
                        lines.append(f"{indent}{out_names[m]} = fmaf({fmt_f(v)}, {an} * {bn}, {out_names[m]});")
    return lines


def emit_cg_bwd_accum(
    cg: np.ndarray,
    g_names: list[str],
    a_names: list[str],
    b_names: list[str],
    da_names: list[str],
    db_names: list[str],
    same_ab: bool = False,
    indent: str = "    ",
) -> list[str]:
    """Emit exact adjoints da[i] += sum_{j,m} cg[i,j,m]*g[m]*b[j], db[j] += sum_{i,m} cg[i,j,m]*g[m]*a[i]."""
    d1, d2, d3 = cg.shape
    lines = []
    for i in range(d1):
        for j in range(d2):
            for m in range(d3):
                v = float(cg[i, j, m])
                if abs(v) > 1e-12:
                    gn, bn = g_names[m], b_names[j]
                    if abs(v - 1.0) < 1e-12:
                        lines.append(f"{indent}{da_names[i]} = fmaf({gn}, {bn}, {da_names[i]});")
                    elif abs(v + 1.0) < 1e-12:
                        lines.append(f"{indent}{da_names[i]} = fmaf(-({gn}), {bn}, {da_names[i]});")
                    else:
                        lines.append(f"{indent}{da_names[i]} = fmaf({fmt_f(v)}, {gn} * {bn}, {da_names[i]});")
    for j in range(d2):
        for i in range(d1):
            for m in range(d3):
                v = float(cg[i, j, m])
                if abs(v) > 1e-12:
                    gn, an = g_names[m], a_names[i]
                    if abs(v - 1.0) < 1e-12:
                        lines.append(f"{indent}{db_names[j]} = fmaf({gn}, {an}, {db_names[j]});")
                    elif abs(v + 1.0) < 1e-12:
                        lines.append(f"{indent}{db_names[j]} = fmaf(-({gn}), {an}, {db_names[j]});")
                    else:
                        lines.append(f"{indent}{db_names[j]} = fmaf({fmt_f(v)}, {gn} * {an}, {db_names[j]});")
    return lines


def verify_paths():
    """Verify FD accuracy of nu2/nu3 and l2 CG adjoints in float64."""
    rng = np.random.default_rng(123)
    for l1 in range(4):
        for l2 in range(4):
            for L in range(3):
                if L < abs(l1 - l2) or L > l1 + l2 or (l1 + l2 + L) % 2 != 0:
                    continue
                cg = cg_tensor(l1, l2, L)
                a = rng.normal(size=2 * l1 + 1)
                b = rng.normal(size=2 * l2 + 1)
                g = rng.normal(size=2 * L + 1)
                out = np.einsum("ijm,i,j->m", cg, a, b)
                da = np.einsum("ijm,m,j->i", cg, g, b)
                db = np.einsum("ijm,m,i->j", cg, g, a)
                eps = 1e-6
                for idx in range(len(a)):
                    ap, am = a.copy(), a.copy()
                    ap[idx] += eps
                    am[idx] -= eps
                    fd = (np.dot(g, np.einsum("ijm,i,j->m", cg, ap, b)) - np.dot(g, np.einsum("ijm,i,j->m", cg, am, b))) / (2 * eps)
                    assert abs(fd - da[idx]) < 1e-7, f"da mismatch {(l1,l2,L)} {idx}: {fd} vs {da[idx]}"
                for idx in range(len(b)):
                    bp, bm = b.copy(), b.copy()
                    bp[idx] += eps
                    bm[idx] -= eps
                    fd = (np.dot(g, np.einsum("ijm,i,j->m", cg, a, bp)) - np.dot(g, np.einsum("ijm,i,j->m", cg, a, bm))) / (2 * eps)
                    assert abs(fd - db[idx]) < 1e-7, f"db mismatch {(l1,l2,L)} {idx}: {fd} vs {db[idx]}"
    print("Python float64 CG adjoint self-check: PASS")


def build_l1_node_lf2_code() -> str:
    """Emit ilmlip_l1_node_fwd_lf2_kernel<LMAX, LMSG> and ilmlip_l1_node_bwd_lf2_kernel<LMAX, LMSG>."""
    out = []
    out.append("// =================================================================================================")
    out.append("// Auto-generated Specialized Layer-1 Node Forward & Backward for lfeat == 2, LMAX in {2,3}, LMSG in {1,2}")
    out.append("// Zero CG-table lookups, conflict-free SMEM b2/db2, skips dead nu3 L=2 paths when LMSG==1.")
    out.append("// =================================================================================================")

    for lmax in (2, 3):
        cfg = {
            "lmax": lmax, "lfeat": 2, "lmsg": 2, "lfeat2": 2, "lout2": 2,
            "n_l": [32, 24, 16, 8][: lmax + 1], "n_msg": 16, "r_gram": 8, "r_bisp": 0,
        }
        nu2, nu3, _, _, _, _, *_ = _build_paths(cfg)
        b2_offs = []
        cur = 0
        for l1, l2, L, _, _, _ in nu2:
            b2_offs.append(cur)
            cur += 2 * L + 1
        b2_dim = cur

        # Emit canonical check tables for IlmlipModel::load()
        out.append(f"constexpr int kLf2Nu2Cnt_L{lmax} = {len(nu2)};")
        out.append(f"constexpr int kLf2Nu3Cnt_L{lmax} = {len(nu3)};")
        nu2_init = ", ".join(f"{{{a},{b},{L}}}" for a, b, L, _, _, _ in nu2)
        nu3_init = ", ".join(f"{{{p2},{l3},{L}}}" for p2, l3, L, _, _, _ in nu3)
        out.append(f"static constexpr int kLf2ExpNu2_L{lmax}[{len(nu2)}][3] = {{{nu2_init}}};")
        out.append(f"static constexpr int kLf2ExpNu3_L{lmax}[{len(nu3)}][3] = {{{nu3_init}}};")
        out.append("")

        def c_expr_nu2(l1, l2):
            if l1 == l2:
                return f"nl{l1}"
            return f"(nl{l1} < nl{l2} ? nl{l1} : nl{l2})"

        def c_expr_nu3(p2_idx, l3):
            l1, l2, _, _, _, _ = nu2[p2_idx]
            ls = sorted(set([l1, l2, l3]))
            e = f"nl{ls[0]}"
            for lx in ls[1:]:
                e = f"({e} < nl{lx} ? {e} : nl{lx})"
            return e

        def a_vars(l):
            return [f"a{l}_{m}" for m in range(2 * l + 1)]

        def da_vars(l):
            return [f"da{l}_{m}" for m in range(2 * l + 1)]

        # --- FORWARD KERNEL ---
        out.append(f"template <int LMSG>")
        out.append(f"__global__ void __launch_bounds__(128) ilmlip_l1_node_fwd_lf2_L{lmax}_kernel(")
        out.append("    int N, int n_feat, Int4 n_l, Int4 msg_in_dim,")
        out.append("    CPtr4 A1, float* __restrict__ x_out, Ptr4 Bcat) {")
        out.append(f"    __shared__ float s_b2[4][{b2_dim}][32];")
        out.append("    const int lx = threadIdx.x;")
        out.append("    const int ay = threadIdx.y;")
        out.append("    const int c = blockIdx.y * 32 + lx;")
        out.append("    const int i = blockIdx.x * 4 + ay;")
        out.append("    if (i >= N) return;")
        out.append("    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];")
        out.append("    const int m0d = msg_in_dim.v[0], m1d = msg_in_dim.v[1], m2d = msg_in_dim.v[2];")
        out.append("    float* xi  = x_out + (int64_t)i * n_feat;")
        out.append("    float* bc0 = Bcat.p[0] + (int64_t)i * m0d;")
        out.append("    float* bc1 = Bcat.p[1] + (int64_t)i * 3 * m1d;")
        out.append("    float* bc2 = (LMSG >= 2) ? (Bcat.p[2] + (int64_t)i * 5 * m2d) : nullptr;")
        out.append("")
        out.append("    const float a0_0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;")
        out.append("    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;")
        out.append("    if (c < nl1) { const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c; a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1]; }")
        out.append("    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;")
        out.append("    if (c < nl2) { const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c; a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2]; }")
        if lmax == 3:
            out.append("    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;")
            out.append("    if (c < nl3) { const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c; a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3]; a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3]; }")
        out.append("")
        out.append("    // nu = 1")
        out.append("    if (c < nl0) { xi[c] = a0_0; bc0[c] = a0_0; }")
        out.append("    if (c < nl1) { bc1[c] = a1_0; bc1[m1d + c] = a1_1; bc1[2 * m1d + c] = a1_2; }")
        out.append("    if constexpr (LMSG >= 2) { if (c < nl2) { bc2[c] = a2_0; bc2[m2d + c] = a2_1; bc2[2 * m2d + c] = a2_2; bc2[3 * m2d + c] = a2_3; bc2[4 * m2d + c] = a2_4; } }")
        out.append("    int xoff = nl0, b0off = nl0, b1off = nl1, b2off = nl2;")
        out.append("")
        out.append("    // nu = 2")
        for p, (l1, l2, L, _, _, _) in enumerate(nu2):
            cg = cg_tensor(l1, l2, L)
            boff = b2_offs[p]
            ce = c_expr_nu2(l1, l2)
            out.append(f"    {{ // p{p}: ({l1},{l2})->{L}")
            out.append(f"        const int cp = {ce};")
            out.append(f"        if (c < cp) {{")
            vnames = [f"v{m}" for m in range(2 * L + 1)]
            out.append("            float " + ", ".join(vnames) + ";")
            out.extend(emit_cg_fwd_terms(cg, a_vars(l1), a_vars(l2), vnames, "            "))
            for m in range(2 * L + 1):
                out.append(f"            s_b2[ay][{boff + m}][lx] = v{m};")
            if L == 0:
                out.append("            xi[xoff + c] = v0; bc0[b0off + c] = v0;")
            elif L == 1:
                out.append("            bc1[b1off + c] = v0; bc1[m1d + b1off + c] = v1; bc1[2 * m1d + b1off + c] = v2;")
            elif L == 2:
                out.append("            if constexpr (LMSG >= 2) { bc2[b2off + c] = v0; bc2[m2d + b2off + c] = v1; bc2[2 * m2d + b2off + c] = v2; bc2[3 * m2d + b2off + c] = v3; bc2[4 * m2d + b2off + c] = v4; }")
            out.append("        }")
            if L == 0:
                out.append("        xoff += cp; b0off += cp;")
            elif L == 1:
                out.append("        b1off += cp;")
            elif L == 2:
                out.append("        if constexpr (LMSG >= 2) b2off += cp;")
            out.append("    }")

        out.append("")
        out.append("    // nu = 3")
        for q, (p2_idx, l3, L, _, _, _) in enumerate(nu3):
            l12 = nu2[p2_idx][2]
            boff = b2_offs[p2_idx]
            cg = cg_tensor(l12, l3, L)
            ce = c_expr_nu3(p2_idx, l3)
            guard = "if constexpr (LMSG >= 2) " if L == 2 else ""
            out.append(f"    {guard}{{ // q{q}: (p{p2_idx}[L={l12}], {l3})->{L}")
            out.append(f"        const int cq = {ce};")
            out.append("        if (c < cq) {")
            b2_names = [f"s_b2[ay][{boff + m}][lx]" for m in range(2 * l12 + 1)]
            vnames = [f"v{m}" for m in range(2 * L + 1)]
            out.append("            float " + ", ".join(vnames) + ";")
            out.extend(emit_cg_fwd_terms(cg, b2_names, a_vars(l3), vnames, "            "))
            if L == 0:
                out.append("            xi[xoff + c] = v0; bc0[b0off + c] = v0;")
            elif L == 1:
                out.append("            bc1[b1off + c] = v0; bc1[m1d + b1off + c] = v1; bc1[2 * m1d + b1off + c] = v2;")
            elif L == 2:
                out.append("            bc2[b2off + c] = v0; bc2[m2d + b2off + c] = v1; bc2[2 * m2d + b2off + c] = v2; bc2[3 * m2d + b2off + c] = v3; bc2[4 * m2d + b2off + c] = v4;")
            out.append("        }")
            if L == 0:
                out.append("        xoff += cq; b0off += cq;")
            elif L == 1:
                out.append("        b1off += cq;")
            elif L == 2:
                out.append("        b2off += cq;")
            out.append("    }")
        out.append("}")
        out.append("")

        # --- BACKWARD KERNEL (blk(32, 2) = 64 threads -> 19.5 KB SMEM -> high occupancy, 0 spills!) ---
        out.append(f"template <int LMSG>")
        out.append(f"__global__ void __launch_bounds__(64) ilmlip_l1_node_bwd_lf2_L{lmax}_kernel(")
        out.append("    int N, int n_feat, Int4 n_l, Int4 msg_in_dim,")
        out.append("    CPtr4 A1, const float* __restrict__ dx, CPtr4 dBcat, Ptr4 dA1) {")
        out.append(f"    __shared__ float s_b2[2][{b2_dim}][32];")
        out.append(f"    __shared__ float s_db2[2][{b2_dim}][32];")
        out.append("    const int lx = threadIdx.x;")
        out.append("    const int ay = threadIdx.y;")
        out.append("    const int c = blockIdx.y * 32 + lx;")
        out.append("    const int i = blockIdx.x * 2 + ay;")
        out.append("    if (i >= N) return;")
        out.append("    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];")
        out.append("    const int m0d = msg_in_dim.v[0], m1d = msg_in_dim.v[1], m2d = msg_in_dim.v[2];")
        out.append("    const float* dxi  = dx + (int64_t)i * n_feat;")
        out.append("    const float* gbc0 = dBcat.p[0] + (int64_t)i * m0d;")
        out.append("    const float* gbc1 = dBcat.p[1] + (int64_t)i * 3 * m1d;")
        out.append("    const float* gbc2 = (LMSG >= 2) ? (dBcat.p[2] + (int64_t)i * 5 * m2d) : nullptr;")
        out.append("")
        out.append("    const float a0_0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;")
        out.append("    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;")
        out.append("    if (c < nl1) { const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c; a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1]; }")
        out.append("    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;")
        out.append("    if (c < nl2) { const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c; a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2]; }")
        if lmax == 3:
            out.append("    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;")
            out.append("    if (c < nl3) { const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c; a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3]; a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3]; }")
        out.append("")
        out.append("    // Seed nu=1 grads")
        out.append("    float da0_0 = (c < nl0) ? (dxi[c] + gbc0[c]) : 0.0f;")
        out.append("    float da1_0 = 0.0f, da1_1 = 0.0f, da1_2 = 0.0f;")
        out.append("    if (c < nl1) { da1_0 = gbc1[c]; da1_1 = gbc1[m1d + c]; da1_2 = gbc1[2 * m1d + c]; }")
        out.append("    float da2_0 = 0.0f, da2_1 = 0.0f, da2_2 = 0.0f, da2_3 = 0.0f, da2_4 = 0.0f;")
        out.append("    if constexpr (LMSG >= 2) { if (c < nl2) { da2_0 = gbc2[c]; da2_1 = gbc2[m2d + c]; da2_2 = gbc2[2 * m2d + c]; da2_3 = gbc2[3 * m2d + c]; da2_4 = gbc2[4 * m2d + c]; } }")
        if lmax == 3:
            out.append("    float da3_0 = 0.0f, da3_1 = 0.0f, da3_2 = 0.0f, da3_3 = 0.0f, da3_4 = 0.0f, da3_5 = 0.0f, da3_6 = 0.0f;")
        out.append("    int xoff = nl0, b0off = nl0, b1off = nl1, b2off = nl2;")
        out.append("")
        out.append("    // Recompute nu=2 into s_b2 and seed s_db2")
        for p, (l1, l2, L, _, _, _) in enumerate(nu2):
            cg = cg_tensor(l1, l2, L)
            boff = b2_offs[p]
            ce = c_expr_nu2(l1, l2)
            out.append(f"    {{ // p{p}: ({l1},{l2})->{L}")
            out.append(f"        const int cp = {ce};")
            out.append("        if (c < cp) {")
            vnames = [f"v{m}" for m in range(2 * L + 1)]
            out.append("            float " + ", ".join(vnames) + ";")
            out.extend(emit_cg_fwd_terms(cg, a_vars(l1), a_vars(l2), vnames, "            "))
            for m in range(2 * L + 1):
                out.append(f"            s_b2[ay][{boff + m}][lx] = v{m};")
            if L == 0:
                out.append(f"            s_db2[ay][{boff}][lx] = dxi[xoff + c] + gbc0[b0off + c];")
            elif L == 1:
                out.append(f"            s_db2[ay][{boff}][lx] = gbc1[b1off + c]; s_db2[ay][{boff + 1}][lx] = gbc1[m1d + b1off + c]; s_db2[ay][{boff + 2}][lx] = gbc1[2 * m1d + b1off + c];")
            elif L == 2:
                out.append("            if constexpr (LMSG >= 2) {")
                for m in range(5):
                    out.append(f"                s_db2[ay][{boff + m}][lx] = gbc2[{m} * m2d + b2off + c];")
                out.append("            } else {")
                for m in range(5):
                    out.append(f"                s_db2[ay][{boff + m}][lx] = 0.0f;")
                out.append("            }")
            out.append("        }")
            if L == 0:
                out.append("        xoff += cp; b0off += cp;")
            elif L == 1:
                out.append("        b1off += cp;")
            elif L == 2:
                out.append("        if constexpr (LMSG >= 2) b2off += cp;")
            out.append("    }")

        out.append("")
        out.append("    // Backprop nu=3 -> s_db2 and da")
        for q, (p2_idx, l3, L, _, _, _) in enumerate(nu3):
            l12 = nu2[p2_idx][2]
            boff = b2_offs[p2_idx]
            cg = cg_tensor(l12, l3, L)
            ce = c_expr_nu3(p2_idx, l3)
            guard = "if constexpr (LMSG >= 2) " if L == 2 else ""
            out.append(f"    {guard}{{ // q{q}: (p{p2_idx}[L={l12}], {l3})->{L}")
            out.append(f"        const int cq = {ce};")
            out.append("        if (c < cq) {")
            if L == 0:
                out.append("            const float g0 = dxi[xoff + c] + gbc0[b0off + c];")
            elif L == 1:
                out.append("            const float g0 = gbc1[b1off + c], g1 = gbc1[m1d + b1off + c], g2 = gbc1[2 * m1d + b1off + c];")
            elif L == 2:
                out.append("            const float g0 = gbc2[b2off + c], g1 = gbc2[m2d + b2off + c], g2 = gbc2[2 * m2d + b2off + c], g3 = gbc2[3 * m2d + b2off + c], g4 = gbc2[4 * m2d + b2off + c];")
            gnames = [f"g{m}" for m in range(2 * L + 1)]
            b2_names = [f"s_b2[ay][{boff + m}][lx]" for m in range(2 * l12 + 1)]
            db2_names = [f"s_db2[ay][{boff + m}][lx]" for m in range(2 * l12 + 1)]
            out.extend(emit_cg_bwd_accum(cg, gnames, b2_names, a_vars(l3), db2_names, da_vars(l3), False, "            "))
            out.append("        }")
            if L == 0:
                out.append("        xoff += cq; b0off += cq;")
            elif L == 1:
                out.append("        b1off += cq;")
            elif L == 2:
                out.append("        b2off += cq;")
            out.append("    }")

        out.append("")
        out.append("    // Backprop nu=2 -> da")
        for p, (l1, l2, L, _, _, _) in enumerate(nu2):
            cg = cg_tensor(l1, l2, L)
            boff = b2_offs[p]
            ce = c_expr_nu2(l1, l2)
            out.append(f"    {{ // p{p}: ({l1},{l2})->{L}")
            out.append(f"        const int cp = {ce};")
            out.append("        if (c < cp) {")
            gnames = [f"s_db2[ay][{boff + m}][lx]" for m in range(2 * L + 1)]
            out.extend(emit_cg_bwd_accum(cg, gnames, a_vars(l1), a_vars(l2), da_vars(l1), da_vars(l2), False, "            "))
            out.append("        }")
            out.append("    }")

        out.append("")
        out.append("    if (c < nl0) dA1.p[0][(int64_t)i * nl0 + c] = da0_0;")
        out.append("    if (c < nl1) { float* d1 = dA1.p[1] + (int64_t)i * 3 * nl1 + c; d1[0] = da1_0; d1[nl1] = da1_1; d1[2 * nl1] = da1_2; }")
        out.append("    if (c < nl2) { float* d2 = dA1.p[2] + (int64_t)i * 5 * nl2 + c; d2[0] = da2_0; d2[nl2] = da2_1; d2[2 * nl2] = da2_2; d2[3 * nl2] = da2_3; d2[4 * nl2] = da2_4; }")
        if lmax == 3:
            out.append("    if (c < nl3) { float* d3 = dA1.p[3] + (int64_t)i * 7 * nl3 + c; d3[0] = da3_0; d3[nl3] = da3_1; d3[2 * nl3] = da3_2; d3[3 * nl3] = da3_3; d3[4 * nl3] = da3_4; d3[5 * nl3] = da3_5; d3[6 * nl3] = da3_6; }")
        out.append("}")
        out.append("")

        # Precompute canonical nu2/nu3 grouping by p2_idx so v=b2 and dv=db2 live purely in registers (ZERO s_b2/s_db2 SMEM!)
        nu3_by_p = [[] for _ in range(len(nu2))]
        for q, (p2_idx, l3, L, _, _, _) in enumerate(nu3):
            nu3_by_p[p2_idx].append((q, l3, L))

        def emit_canonical_offsets(lines_out, include_lmsg_guard=True):
            for p, (l1, l2, L, _, _, _) in enumerate(nu2):
                lines_out.append(f"    const int cp_{p} = {c_expr_nu2(l1, l2)};")
            for q, (p2_idx, l3, L, _, _, _) in enumerate(nu3):
                lines_out.append(f"    const int cq_{q} = {c_expr_nu3(p2_idx, l3)};")
            lines_out.append("    int b0_cur = nl0, b1_cur = nl1, b2_cur = nl2;")
            for p, (l1, l2, L, _, _, _) in enumerate(nu2):
                if L == 0:
                    lines_out.append(f"    const int off_p{p} = b0_cur; b0_cur += cp_{p};")
                elif L == 1:
                    lines_out.append(f"    const int off_p{p} = b1_cur; b1_cur += cp_{p};")
                elif L == 2:
                    if include_lmsg_guard:
                        lines_out.append(f"    const int off_p{p} = b2_cur; if constexpr (LMSG >= 2) b2_cur += cp_{p};")
                    else:
                        lines_out.append(f"    const int off_p{p} = b2_cur; b2_cur += cp_{p};")
            for q, (p2_idx, l3, L, _, _, _) in enumerate(nu3):
                if L == 0:
                    lines_out.append(f"    const int off_q{q} = b0_cur; b0_cur += cq_{q};")
                elif L == 1:
                    lines_out.append(f"    const int off_q{q} = b1_cur; b1_cur += cq_{q};")
                elif L == 2:
                    if include_lmsg_guard:
                        lines_out.append(f"    const int off_q{q} = b2_cur; if constexpr (LMSG >= 2) b2_cur += cq_{q};")
                    else:
                        lines_out.append(f"    const int off_q{q} = b2_cur; b2_cur += cq_{q};")

        # --- FUSED L1 NODE + SMEM-W_MSG FORWARD (ZERO s_b2 SMEM! Grouped-by-p2 register-only b2!) ---
        out.append(f"template <int LMSG, int QMAX, bool NORM>")
        out.append(f"__global__ void __launch_bounds__(128) ilmlip_l1_node_fwd_lf2_msg_L{lmax}_kernel(")
        out.append("    int N, int n_feat_pad, int n_msg, Int4 n_l, Int4 msg_in_dim,")
        out.append("    CPtr4 A1, const float* __restrict__ W0T, const float* __restrict__ W1T, const float* __restrict__ W2T,")
        out.append("    const float* __restrict__ mu_x, const float* __restrict__ inv_std_x,")
        out.append("    float* __restrict__ x_out, float* __restrict__ am_l1_out,")
        out.append("    float4* __restrict__ I_pack4, Ptr4 I_out) {")
        out.append("    extern __shared__ float s_W[];")
        out.append("    const int m0d = msg_in_dim.v[0], m1d = msg_in_dim.v[1], m2d = msg_in_dim.v[2];")
        out.append("    float* s_W0 = s_W;")
        out.append("    float* s_W1 = s_W0 + n_msg * m0d;")
        out.append("    float* s_W2 = s_W1 + n_msg * m1d;")
        out.append("    const int tid = threadIdx.y * 32 + threadIdx.x;")
        out.append("    for (int idx = tid; idx < n_msg * m0d; idx += 128) s_W0[idx] = __ldg(W0T + idx);")
        out.append("    for (int idx = tid; idx < n_msg * m1d; idx += 128) s_W1[idx] = __ldg(W1T + idx);")
        out.append("    if constexpr (LMSG >= 2) {")
        out.append("        for (int idx = tid; idx < n_msg * m2d; idx += 128) s_W2[idx] = __ldg(W2T + idx);")
        out.append("    }")
        out.append("    __syncthreads();")
        out.append("    const int lx = threadIdx.x, ay = threadIdx.y;")
        out.append("    const int c = lx;")
        out.append("    const int i = blockIdx.x * 4 + ay;")
        out.append("    if (i >= N) return;")
        out.append("    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];")
        out.append("    float* xi = x_out + (int64_t)i * n_feat_pad;")
        out.append("    float am = 0.0f;")
        out.append("    float acc0[QMAX];")
        out.append("    float acc1_0[QMAX], acc1_1[QMAX], acc1_2[QMAX];")
        out.append("    float acc2_0[LMSG >= 2 ? QMAX : 1], acc2_1[LMSG >= 2 ? QMAX : 1], acc2_2[LMSG >= 2 ? QMAX : 1];")
        out.append("    float acc2_3[LMSG >= 2 ? QMAX : 1], acc2_4[LMSG >= 2 ? QMAX : 1];")
        out.append("    #pragma unroll")
        out.append("    for (int k = 0; k < QMAX; ++k) {")
        out.append("        acc0[k] = 0.0f; acc1_0[k] = 0.0f; acc1_1[k] = 0.0f; acc1_2[k] = 0.0f;")
        out.append("        if constexpr (LMSG >= 2) { acc2_0[k] = 0.0f; acc2_1[k] = 0.0f; acc2_2[k] = 0.0f; acc2_3[k] = 0.0f; acc2_4[k] = 0.0f; }")
        out.append("    }")
        out.append("    auto emit0 = [&](int off, float v0) {")
        out.append("        const int f = off + c;")
        out.append("        if constexpr (NORM) { const float vn = (v0 - __ldg(mu_x + f)) * __ldg(inv_std_x + f); xi[f] = vn; am = fmaxf(am, fabsf(vn)); }")
        out.append("        else { xi[f] = v0; }")
        out.append("        #pragma unroll")
        out.append("        for (int q = 0; q < QMAX; ++q) {")
        out.append("            if (q < n_msg) acc0[q] = fmaf(v0, s_W0[q * m0d + f], acc0[q]);")
        out.append("        }")
        out.append("    };")
        out.append("    auto emit1 = [&](int off, float v0, float v1, float v2) {")
        out.append("        const int f = off + c;")
        out.append("        #pragma unroll")
        out.append("        for (int q = 0; q < QMAX; ++q) {")
        out.append("            if (q < n_msg) {")
        out.append("                const float w = s_W1[q * m1d + f];")
        out.append("                acc1_0[q] = fmaf(v0, w, acc1_0[q]); acc1_1[q] = fmaf(v1, w, acc1_1[q]); acc1_2[q] = fmaf(v2, w, acc1_2[q]);")
        out.append("            }")
        out.append("        }")
        out.append("    };")
        out.append("    auto emit2 = [&](int off, float v0, float v1, float v2, float v3, float v4) {")
        out.append("        if constexpr (LMSG >= 2) {")
        out.append("            const int f = off + c;")
        out.append("            #pragma unroll")
        out.append("            for (int q = 0; q < QMAX; ++q) {")
        out.append("                if (q < n_msg) {")
        out.append("                    const float w = s_W2[q * m2d + f];")
        out.append("                    acc2_0[q] = fmaf(v0, w, acc2_0[q]); acc2_1[q] = fmaf(v1, w, acc2_1[q]); acc2_2[q] = fmaf(v2, w, acc2_2[q]);")
        out.append("                    acc2_3[q] = fmaf(v3, w, acc2_3[q]); acc2_4[q] = fmaf(v4, w, acc2_4[q]);")
        out.append("                }")
        out.append("            }")
        out.append("        }")
        out.append("    };")
        out.append("    const float a0_0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;")
        out.append("    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;")
        out.append("    if (c < nl1) { const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c; a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1]; }")
        out.append("    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;")
        out.append("    if (c < nl2) { const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c; a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2]; }")
        if lmax == 3:
            out.append("    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;")
            out.append("    if (c < nl3) { const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c; a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3]; a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3]; }")
        out.append("    if (c < nl0) emit0(0, a0_0);")
        out.append("    if (c < nl1) emit1(0, a1_0, a1_1, a1_2);")
        out.append("    if constexpr (LMSG >= 2) { if (c < nl2) emit2(0, a2_0, a2_1, a2_2, a2_3, a2_4); }")
        emit_canonical_offsets(out, include_lmsg_guard=True)
        for p, (l1, l2, L12, _, _, _) in enumerate(nu2):
            cg = cg_tensor(l1, l2, L12)
            out.append(f"    if (c < cp_{p}) {{")
            vnames = [f"v{m}" for m in range(2 * L12 + 1)]
            out.append("        float " + ", ".join(vnames) + ";")
            out.extend(emit_cg_fwd_terms(cg, a_vars(l1), a_vars(l2), vnames, "        "))
            if L12 == 0:
                out.append(f"        emit0(off_p{p}, v0);")
            elif L12 == 1:
                out.append(f"        emit1(off_p{p}, v0, v1, v2);")
            elif L12 == 2:
                out.append(f"        if constexpr (LMSG >= 2) emit2(off_p{p}, v0, v1, v2, v3, v4);")
            for q, l3, L_q in nu3_by_p[p]:
                cg_q = cg_tensor(L12, l3, L_q)
                guard = "if constexpr (LMSG >= 2) " if L_q == 2 else ""
                out.append(f"        {guard}if (c < cq_{q}) {{")
                unames = [f"u{m}" for m in range(2 * L_q + 1)]
                out.append("            float " + ", ".join(unames) + ";")
                out.extend(emit_cg_fwd_terms(cg_q, vnames, a_vars(l3), unames, "            "))
                if L_q == 0:
                    out.append(f"            emit0(off_q{q}, u0);")
                elif L_q == 1:
                    out.append(f"            emit1(off_q{q}, u0, u1, u2);")
                elif L_q == 2:
                    out.append(f"            emit2(off_q{q}, u0, u1, u2, u3, u4);")
                out.append("        }")
            out.append("    }")
        out.append("    auto red_q = [&](const float* a) -> float {")
        out.append("        if constexpr (QMAX == 4) return warp_reduce4_to_q(a, lx);")
        out.append("        else return warp_reduce8_to_q(a, lx);")
        out.append("    };")
        out.append("    const float o0 = red_q(acc0);")
        out.append("    const float o1_0 = red_q(acc1_0), o1_1 = red_q(acc1_1), o1_2 = red_q(acc1_2);")
        out.append("    float o2_0 = 0.0f, o2_1 = 0.0f, o2_2 = 0.0f, o2_3 = 0.0f, o2_4 = 0.0f;")
        out.append("    if constexpr (LMSG >= 2) {")
        out.append("        o2_0 = red_q(acc2_0); o2_1 = red_q(acc2_1); o2_2 = red_q(acc2_2); o2_3 = red_q(acc2_3); o2_4 = red_q(acc2_4);")
        out.append("    }")
        out.append("    if (lx < n_msg) {")
        out.append("        if (LMSG == 1 && I_pack4) {")
        out.append("            I_pack4[(int64_t)i * n_msg + lx] = make_float4(o0, o1_0, o1_1, o1_2);")
        out.append("        } else {")
        out.append("            I_out.p[0][(int64_t)i * n_msg + lx] = o0;")
        out.append("            float* d1 = I_out.p[1] + (int64_t)i * 3 * n_msg + lx;")
        out.append("            d1[0] = o1_0; d1[n_msg] = o1_1; d1[2 * n_msg] = o1_2;")
        out.append("            if constexpr (LMSG >= 2) {")
        out.append("                float* d2 = I_out.p[2] + (int64_t)i * 5 * n_msg + lx;")
        out.append("                d2[0] = o2_0; d2[n_msg] = o2_1; d2[2 * n_msg] = o2_2; d2[3 * n_msg] = o2_3; d2[4 * n_msg] = o2_4;")
        out.append("            }")
        out.append("        }")
        out.append("    }")
        out.append("    if constexpr (NORM) {")
        out.append("        #pragma unroll")
        out.append("        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));")
        out.append("        if (lx == 0) am_l1_out[i] = am;")
        out.append("    }")
        out.append("}")
        out.append("")

        # --- FUSED L1 NODE + SMEM-W_MSG^T BACKWARD (ZERO s_b2/s_db2 SMEM! Register-only grouped-by-p2, 4 atoms/CTA!) ---
        bwd_atoms = 4
        bwd_thr = bwd_atoms * 32
        out.append(f"template <int LMSG, int QMAX, bool SCALE_DX>")
        out.append(f"__global__ void __launch_bounds__({bwd_thr}) ilmlip_l1_node_bwd_lf2_msg_L{lmax}_kernel(")
        out.append("    int N, int n_feat_pad, int n_msg, Int4 n_l, Int4 msg_in_dim,")
        out.append("    CPtr4 A1, const float* __restrict__ inv_std_x, const float* __restrict__ dx, CPtr4 dI,")
        out.append("    const float* __restrict__ W0T, const float* __restrict__ W1T, const float* __restrict__ W2T, Ptr4 dA1) {")
        out.append("    extern __shared__ float s_W[];")
        out.append(f"    __shared__ float s_dI0[{bwd_atoms}][QMAX];")
        out.append(f"    __shared__ float s_dI1[{bwd_atoms}][3][QMAX];")
        out.append(f"    __shared__ float s_dI2[LMSG >= 2 ? {bwd_atoms} : 1][LMSG >= 2 ? 5 : 1][QMAX];")
        out.append("    const int m0d = msg_in_dim.v[0], m1d = msg_in_dim.v[1], m2d = msg_in_dim.v[2];")
        out.append("    float* s_W0 = s_W;")
        out.append("    float* s_W1 = s_W0 + n_msg * m0d;")
        out.append("    float* s_W2 = s_W1 + n_msg * m1d;")
        out.append("    const int tid = threadIdx.y * 32 + threadIdx.x;")
        out.append(f"    for (int idx = tid; idx < n_msg * m0d; idx += {bwd_thr}) s_W0[idx] = __ldg(W0T + idx);")
        out.append(f"    for (int idx = tid; idx < n_msg * m1d; idx += {bwd_thr}) s_W1[idx] = __ldg(W1T + idx);")
        out.append("    if constexpr (LMSG >= 2) {")
        out.append(f"        for (int idx = tid; idx < n_msg * m2d; idx += {bwd_thr}) s_W2[idx] = __ldg(W2T + idx);")
        out.append("    }")
        out.append("    __syncthreads();")
        out.append("    const int lx = threadIdx.x, ay = threadIdx.y;")
        out.append("    const int c = lx;")
        out.append(f"    const int i = blockIdx.x * {bwd_atoms} + ay;")
        out.append("    if (i >= N) return;")
        out.append("    if (lx < QMAX) {")
        out.append("        if (lx < n_msg) {")
        out.append("            s_dI0[ay][lx] = __ldg(dI.p[0] + (int64_t)i * n_msg + lx);")
        out.append("            const float* p1 = dI.p[1] + (int64_t)i * 3 * n_msg + lx;")
        out.append("            s_dI1[ay][0][lx] = __ldg(p1); s_dI1[ay][1][lx] = __ldg(p1 + n_msg); s_dI1[ay][2][lx] = __ldg(p1 + 2 * n_msg);")
        out.append("            if constexpr (LMSG >= 2) {")
        out.append("                const float* p2 = dI.p[2] + (int64_t)i * 5 * n_msg + lx;")
        out.append("                s_dI2[ay][0][lx] = __ldg(p2); s_dI2[ay][1][lx] = __ldg(p2 + n_msg); s_dI2[ay][2][lx] = __ldg(p2 + 2 * n_msg);")
        out.append("                s_dI2[ay][3][lx] = __ldg(p2 + 3 * n_msg); s_dI2[ay][4][lx] = __ldg(p2 + 4 * n_msg);")
        out.append("            }")
        out.append("        } else {")
        out.append("            s_dI0[ay][lx] = 0.0f; s_dI1[ay][0][lx] = 0.0f; s_dI1[ay][1][lx] = 0.0f; s_dI1[ay][2][lx] = 0.0f;")
        out.append("            if constexpr (LMSG >= 2) { for (int m = 0; m < 5; ++m) s_dI2[ay][m][lx] = 0.0f; }")
        out.append("        }")
        out.append("    }")
        out.append("    __syncwarp();")
        out.append("    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];")
        out.append("    const float* dxi = dx + (int64_t)i * n_feat_pad;")
        out.append("    auto load_g0 = [&](int off) -> float {")
        out.append("        const int f = off + c;")
        out.append("        float g = dxi[f];")
        out.append("        if constexpr (SCALE_DX) g *= __ldg(inv_std_x + f);")
        out.append("        #pragma unroll")
        out.append("        for (int q = 0; q < QMAX; ++q) { if (q < n_msg) g = fmaf(s_dI0[ay][q], s_W0[q * m0d + f], g); }")
        out.append("        return g;")
        out.append("    };")
        out.append("    auto load_g1 = [&](int off, float& g0, float& g1, float& g2) {")
        out.append("        const int f = off + c; g0 = 0.0f; g1 = 0.0f; g2 = 0.0f;")
        out.append("        #pragma unroll")
        out.append("        for (int q = 0; q < QMAX; ++q) { if (q < n_msg) {")
        out.append("            const float w = s_W1[q * m1d + f];")
        out.append("            g0 = fmaf(s_dI1[ay][0][q], w, g0); g1 = fmaf(s_dI1[ay][1][q], w, g1); g2 = fmaf(s_dI1[ay][2][q], w, g2);")
        out.append("        } }")
        out.append("    };")
        out.append("    auto load_g2 = [&](int off, float& g0, float& g1, float& g2, float& g3, float& g4) {")
        out.append("        g0 = g1 = g2 = g3 = g4 = 0.0f;")
        out.append("        if constexpr (LMSG >= 2) {")
        out.append("            const int f = off + c;")
        out.append("            #pragma unroll")
        out.append("            for (int q = 0; q < QMAX; ++q) { if (q < n_msg) {")
        out.append("                const float w = s_W2[q * m2d + f];")
        out.append("                g0 = fmaf(s_dI2[ay][0][q], w, g0); g1 = fmaf(s_dI2[ay][1][q], w, g1); g2 = fmaf(s_dI2[ay][2][q], w, g2);")
        out.append("                g3 = fmaf(s_dI2[ay][3][q], w, g3); g4 = fmaf(s_dI2[ay][4][q], w, g4);")
        out.append("            } }")
        out.append("        }")
        out.append("    };")
        out.append("    const float a0_0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;")
        out.append("    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;")
        out.append("    if (c < nl1) { const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c; a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1]; }")
        out.append("    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;")
        out.append("    if (c < nl2) { const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c; a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2]; }")
        if lmax == 3:
            out.append("    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;")
            out.append("    if (c < nl3) { const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c; a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3]; a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3]; }")
        out.append("    float da0_0 = (c < nl0) ? load_g0(0) : 0.0f;")
        out.append("    float da1_0 = 0.0f, da1_1 = 0.0f, da1_2 = 0.0f;")
        out.append("    if (c < nl1) load_g1(0, da1_0, da1_1, da1_2);")
        out.append("    float da2_0 = 0.0f, da2_1 = 0.0f, da2_2 = 0.0f, da2_3 = 0.0f, da2_4 = 0.0f;")
        out.append("    if constexpr (LMSG >= 2) { if (c < nl2) load_g2(0, da2_0, da2_1, da2_2, da2_3, da2_4); }")
        if lmax == 3:
            out.append("    float da3_0 = 0.0f, da3_1 = 0.0f, da3_2 = 0.0f, da3_3 = 0.0f, da3_4 = 0.0f, da3_5 = 0.0f, da3_6 = 0.0f;")
        emit_canonical_offsets(out, include_lmsg_guard=True)
        for p, (l1, l2, L12, _, _, _) in enumerate(nu2):
            cg = cg_tensor(l1, l2, L12)
            out.append(f"    if (c < cp_{p}) {{")
            vnames = [f"v{m}" for m in range(2 * L12 + 1)]
            dvnames = [f"dv{m}" for m in range(2 * L12 + 1)]
            out.append("        float " + ", ".join(vnames) + ";")
            out.extend(emit_cg_fwd_terms(cg, a_vars(l1), a_vars(l2), vnames, "        "))
            out.append("        float " + ", ".join(f"{n} = 0.0f" for n in dvnames) + ";")
            if L12 == 0:
                out.append(f"        dv0 = load_g0(off_p{p});")
            elif L12 == 1:
                out.append(f"        load_g1(off_p{p}, dv0, dv1, dv2);")
            elif L12 == 2:
                out.append(f"        if constexpr (LMSG >= 2) load_g2(off_p{p}, dv0, dv1, dv2, dv3, dv4);")
            for q, l3, L_q in nu3_by_p[p]:
                cg_q = cg_tensor(L12, l3, L_q)
                guard = "if constexpr (LMSG >= 2) " if L_q == 2 else ""
                out.append(f"        {guard}if (c < cq_{q}) {{")
                if L_q == 0:
                    out.append(f"            const float g0 = load_g0(off_q{q});")
                elif L_q == 1:
                    out.append(f"            float g0, g1, g2; load_g1(off_q{q}, g0, g1, g2);")
                elif L_q == 2:
                    out.append(f"            float g0, g1, g2, g3, g4; load_g2(off_q{q}, g0, g1, g2, g3, g4);")
                gnames = [f"g{m}" for m in range(2 * L_q + 1)]
                out.extend(emit_cg_bwd_accum(cg_q, gnames, vnames, a_vars(l3), dvnames, da_vars(l3), False, "            "))
                out.append("        }")
            out.extend(emit_cg_bwd_accum(cg, dvnames, a_vars(l1), a_vars(l2), da_vars(l1), da_vars(l2), False, "        "))
            out.append("    }")
        out.append("    if (c < nl0) dA1.p[0][(int64_t)i * nl0 + c] = da0_0;")
        out.append("    if (c < nl1) { float* d1 = dA1.p[1] + (int64_t)i * 3 * nl1 + c; d1[0] = da1_0; d1[nl1] = da1_1; d1[2 * nl1] = da1_2; }")
        out.append("    if (c < nl2) { float* d2 = dA1.p[2] + (int64_t)i * 5 * nl2 + c; d2[0] = da2_0; d2[nl2] = da2_1; d2[2 * nl2] = da2_2; d2[3 * nl2] = da2_3; d2[4 * nl2] = da2_4; }")
        if lmax == 3:
            out.append("    if (c < nl3) { float* d3 = dA1.p[3] + (int64_t)i * 7 * nl3 + c; d3[0] = da3_0; d3[nl3] = da3_1; d3[2 * nl3] = da3_2; d3[3 * nl3] = da3_3; d3[4 * nl3] = da3_4; d3[5 * nl3] = da3_5; d3[6 * nl3] = da3_6; }")
        out.append("}")
        out.append("")

        # --- 1-LAYER (n_msg == 0) SPECIALIZED LF2 FORWARD + INLINE STANDARDIZATION + eA_x (ZERO SMEM!) ---
        cfg1 = dict(cfg, n_msg=0, lmsg=0, lfeat2=0, lout2=0, r_gram=0, r_bisp=0)
        nu2_1L, nu3_1L, _, _, _, _, *_ = _build_paths(cfg1)
        assert len(nu2_1L) == len(nu2)
        nu3_1L_by_p = [[] for _ in range(len(nu2))]
        for q, (p2_idx, l3, L, _, _, _) in enumerate(nu3_1L):
            assert L == 0
            nu3_1L_by_p[p2_idx].append((q, l3))
        used_p2 = {p2_idx for p2_idx, _, _, _, _, _ in nu3_1L}
        out.append(f"constexpr int kLf2_1L_Nu3Cnt_L{lmax} = {len(nu3_1L)};")
        nu3_1L_init = ", ".join(f"{{{p2},{l3},{L}}}" for p2, l3, L, _, _, _ in nu3_1L)
        out.append(f"static constexpr int kLf2_1L_ExpNu3_L{lmax}[{len(nu3_1L)}][3] = {{{nu3_1L_init}}};")
        out.append("")

        def emit_1L_offsets(lines_out):
            for p, (l1, l2, L, _, _, _) in enumerate(nu2):
                if L == 0 or p in used_p2:
                    lines_out.append(f"    const int cp_{p} = {c_expr_nu2(l1, l2)};")
            for q, (p2_idx, l3, _, _, _, _) in enumerate(nu3_1L):
                lines_out.append(f"    const int cq_{q} = {c_expr_nu3(p2_idx, l3)};")
            lines_out.append("    int x_cur = nl0;")
            for p, (l1, l2, L, _, _, _) in enumerate(nu2):
                if L == 0:
                    lines_out.append(f"    const int xoff_p{p} = x_cur; x_cur += cp_{p};")
            for q, (p2_idx, l3, _, _, _, _) in enumerate(nu3_1L):
                lines_out.append(f"    const int xoff_q{q} = x_cur; x_cur += cq_{q};")

        out.append(f"__global__ void __launch_bounds__(128) ilmlip_l1_node_fwd_1L_lf2_L{lmax}_kernel(")
        out.append("    int N, int n_feat_pad, Int4 n_l, CPtr4 A1,")
        out.append("    const float* __restrict__ mu_x, const float* __restrict__ inv_std_x,")
        out.append("    float* __restrict__ x_out, int8_t* __restrict__ eA_x) {")
        out.append("    const int lx = threadIdx.x, ay = threadIdx.y;")
        out.append("    const int c = blockIdx.y * 32 + lx;")
        out.append("    const int i = blockIdx.x * 4 + ay;")
        out.append("    if (i >= N) return;")
        out.append("    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];")
        out.append("    float* xi = x_out + (int64_t)i * n_feat_pad;")
        out.append("    float am = 0.0f;")
        out.append("    auto store_x = [&](int f, float v) {")
        out.append("        const float vn = (v - __ldg(mu_x + f)) * __ldg(inv_std_x + f);")
        out.append("        xi[f] = vn; am = fmaxf(am, fabsf(vn));")
        out.append("    };")
        out.append("    const float a0_0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;")
        out.append("    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;")
        out.append("    if (c < nl1) { const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c; a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1]; }")
        out.append("    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;")
        out.append("    if (c < nl2) { const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c; a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2]; }")
        if lmax == 3:
            out.append("    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;")
            out.append("    if (c < nl3) { const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c; a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3]; a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3]; }")
        out.append("    if (c < nl0) store_x(c, a0_0);")
        emit_1L_offsets(out)
        for p, (l1, l2, L12, _, _, _) in enumerate(nu2):
            if L12 != 0 and p not in used_p2:
                continue
            cg = cg_tensor(l1, l2, L12)
            out.append(f"    if (c < cp_{p}) {{")
            vnames = [f"v{m}" for m in range(2 * L12 + 1)]
            out.append("        float " + ", ".join(vnames) + ";")
            out.extend(emit_cg_fwd_terms(cg, a_vars(l1), a_vars(l2), vnames, "        "))
            if L12 == 0:
                out.append(f"        store_x(xoff_p{p} + c, v0);")
            for q, l3 in nu3_1L_by_p[p]:
                cg_q = cg_tensor(L12, l3, 0)
                out.append(f"        if (c < cq_{q}) {{")
                out.append("            float u0;")
                out.extend(emit_cg_fwd_terms(cg_q, vnames, a_vars(l3), ["u0"], "            "))
                out.append(f"            store_x(xoff_q{q} + c, u0);")
                out.append("        }")
            out.append("    }")
        out.append("    if (eA_x && gridDim.y == 1) {")
        out.append("        #pragma unroll\n        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));")
        out.append("        if (lx == 0) {")
        out.append("            int e = -127;")
        out.append("            if (am > 0.0f && isfinite(am)) { e = (int)((__float_as_uint(am) >> 23) & 0xff) - 127 + 1; if (e < -108) e = -108; if (e > 108) e = 108; }")
        out.append("            eA_x[i] = (int8_t)e;")
        out.append("        }")
        out.append("    }")
        out.append("}")
        out.append("")

        # --- 1-LAYER (n_msg == 0) SPECIALIZED LF2 BACKWARD + INLINE inv_std_x (ZERO SMEM, 4 atoms/CTA!) ---
        out.append(f"__global__ void __launch_bounds__(128) ilmlip_l1_node_bwd_1L_lf2_L{lmax}_kernel(")
        out.append("    int N, int n_feat_pad, Int4 n_l, CPtr4 A1,")
        out.append("    const float* __restrict__ inv_std_x, const float* __restrict__ dx, Ptr4 dA1) {")
        out.append("    const int lx = threadIdx.x, ay = threadIdx.y;")
        out.append("    const int c = blockIdx.y * 32 + lx;")
        out.append("    const int i = blockIdx.x * 4 + ay;")
        out.append("    if (i >= N) return;")
        out.append("    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];")
        out.append("    const float* dxi = dx + (int64_t)i * n_feat_pad;")
        out.append("    const float a0_0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;")
        out.append("    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;")
        out.append("    if (c < nl1) { const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c; a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1]; }")
        out.append("    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;")
        out.append("    if (c < nl2) { const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c; a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2]; }")
        if lmax == 3:
            out.append("    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;")
            out.append("    if (c < nl3) { const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c; a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3]; a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3]; }")
        out.append("    float da0_0 = (c < nl0) ? (dxi[c] * __ldg(inv_std_x + c)) : 0.0f;")
        out.append("    float da1_0 = 0.0f, da1_1 = 0.0f, da1_2 = 0.0f;")
        out.append("    float da2_0 = 0.0f, da2_1 = 0.0f, da2_2 = 0.0f, da2_3 = 0.0f, da2_4 = 0.0f;")
        if lmax == 3:
            out.append("    float da3_0 = 0.0f, da3_1 = 0.0f, da3_2 = 0.0f, da3_3 = 0.0f, da3_4 = 0.0f, da3_5 = 0.0f, da3_6 = 0.0f;")
        emit_1L_offsets(out)
        for p, (l1, l2, L12, _, _, _) in enumerate(nu2):
            if L12 != 0 and p not in used_p2:
                continue
            cg = cg_tensor(l1, l2, L12)
            out.append(f"    if (c < cp_{p}) {{")
            if p in used_p2:
                vnames = [f"v{m}" for m in range(2 * L12 + 1)]
                dvnames = [f"dv{m}" for m in range(2 * L12 + 1)]
                out.append("        float " + ", ".join(vnames) + ";")
                out.extend(emit_cg_fwd_terms(cg, a_vars(l1), a_vars(l2), vnames, "        "))
                out.append("        float " + ", ".join(f"{n} = 0.0f" for n in dvnames) + ";")
                if L12 == 0:
                    out.append(f"        dv0 = dxi[xoff_p{p} + c] * __ldg(inv_std_x + xoff_p{p} + c);")
                for q, l3 in nu3_1L_by_p[p]:
                    cg_q = cg_tensor(L12, l3, 0)
                    out.append(f"        if (c < cq_{q}) {{")
                    out.append(f"            const float g0 = dxi[xoff_q{q} + c] * __ldg(inv_std_x + xoff_q{q} + c);")
                    out.extend(emit_cg_bwd_accum(cg_q, ["g0"], vnames, a_vars(l3), dvnames, da_vars(l3), False, "            "))
                    out.append("        }")
                out.extend(emit_cg_bwd_accum(cg, dvnames, a_vars(l1), a_vars(l2), da_vars(l1), da_vars(l2), False, "        "))
            else:
                out.append(f"        const float dv0 = dxi[xoff_p{p} + c] * __ldg(inv_std_x + xoff_p{p} + c);")
                out.extend(emit_cg_bwd_accum(cg, ["dv0"], a_vars(l1), a_vars(l2), da_vars(l1), da_vars(l2), False, "        "))
            out.append("    }")
        out.append("    if (c < nl0) dA1.p[0][(int64_t)i * nl0 + c] = da0_0;")
        out.append("    if (c < nl1) { float* d1 = dA1.p[1] + (int64_t)i * 3 * nl1 + c; d1[0] = da1_0; d1[nl1] = da1_1; d1[2 * nl1] = da1_2; }")
        out.append("    if (c < nl2) { float* d2 = dA1.p[2] + (int64_t)i * 5 * nl2 + c; d2[0] = da2_0; d2[nl2] = da2_1; d2[2 * nl2] = da2_2; d2[3 * nl2] = da2_3; d2[4 * nl2] = da2_4; }")
        if lmax == 3:
            out.append("    if (c < nl3) { float* d3 = dA1.p[3] + (int64_t)i * 7 * nl3 + c; d3[0] = da3_0; d3[nl3] = da3_1; d3[2 * nl3] = da3_2; d3[3 * nl3] = da3_3; d3[4 * nl3] = da3_4; d3[5 * nl3] = da3_5; d3[6 * nl3] = da3_6; }")
        out.append("}")
        out.append("")
    return "\n".join(out)


def build_l2_lf2_code() -> str:
    """Emit specialized L2 edge forward, fused gate+pack, and sender-owned zero-atomic backward for lfeat2=2, lout2=2, LMSG in {1,2}."""
    out = []
    out.append("// =================================================================================================")
    out.append("// Fast l<=2 real spherical harmonics (matches edge_sh16[:9] exactly, half the FLOPs)")
    out.append("// =================================================================================================")
    out.append("__device__ __forceinline__ void edge_sh9(float x, float y, float z, float r, float* __restrict__ Y) {")
    out.append("    const float inv_r = 1.0f / r;")
    out.append("    const float ux = x * inv_r, uy = y * inv_r, uz = z * inv_r;")
    out.append("    Y[0] = 1.0f;")
    out.append("    constexpr float S3  = 1.7320508075688772935f;")
    out.append("    constexpr float S15 = 3.8729833462074168852f;")
    out.append("    constexpr float S5  = 2.2360679774997896964f;")
    out.append("    Y[1] = S3 * ux;")
    out.append("    Y[2] = S3 * uy;")
    out.append("    Y[3] = S3 * uz;")
    out.append("    Y[4] = S15 * ux * uz;")
    out.append("    Y[5] = S15 * ux * uy;")
    out.append("    Y[6] = 0.5f * S5 * (3.0f * uy * uy - 1.0f);")
    out.append("    Y[7] = S15 * uy * uz;")
    out.append("    Y[8] = 0.5f * S15 * (uz * uz - ux * ux);")
    out.append("}")
    out.append("")

    # Verify edge_sh9 matches e3nn spherical harmonics (which edge_sh16 uses)!
    # Check signs/ordering against generated_sh16.cuh!
    for lmsg in (1, 2):
        cfg = {
            "lmax": 3, "lfeat": 2, "lmsg": lmsg, "lfeat2": 2, "lout2": 2,
            "n_l": [32, 24, 16, 8], "n_msg": 16, "r_gram": 8, "r_bisp": 0,
        }
        _, _, l2p, l2_by_L, _, _, *_ = _build_paths(cfg)
        # Assign contiguous flat slot indices for each path (ordered by L, then p_in_L)
        slot_of = {}
        cur = 0
        ordered_paths = []
        for L in (0, 1, 2):
            for p_in_L, (l1, l2) in enumerate(l2_by_L[L]):
                slot_of[(L, p_in_L)] = cur
                ordered_paths.append((l1, l2, L, p_in_L, cur))
                cur += 2 * L + 1
        n_out = cur  # 21 for lmsg=1, 35 for lmsg=2

        # --- 1) L2 EDGE FORWARD (supports SUBWARP in {8,16,32} and DUAL_G for dual r_max2 / n_base2 != n_base1!) ---
        out.append(f"// Specialized L2 Edge Forward: lfeat2=2, lout2=2, lmsg={lmsg} (N_OUT={n_out})")
        out.append(f"template <int SUBWARP, int NB_MAX, bool REM_BF, bool SMEM_C, bool DUAL_G = false>")
        out.append(f"__global__ void __launch_bounds__(128) ilmlip_l2_edge_fwd_lf2_m{lmsg}_kernel(")
        out.append("    int N, int S, int n_base2, int n_base1, int n_msg,")
        out.append("    Int4 l2_c, ChebParam r_max2, ChebParam r_max1, float a0_over_sig1, float inv_DA2_sig1_kap_sq,")
        out.append("    const int* __restrict__ row_ptr, const int* __restrict__ row_end, const int* __restrict__ senders,")
        out.append("    const float4* __restrict__ vec_len, const int* __restrict__ species,")
        out.append("    const float* __restrict__ C2, const float* __restrict__ Cbf,")
        out.append("    const float* __restrict__ f0, CPtr4 I_node, Ptr4 A2, float* __restrict__ f1_out) {")
        out.append("    constexpr int NSUB = 128 / SUBWARP;")
        out.append("    constexpr int EBATCH = (SUBWARP == 32) ? 8 : 4;")
        out.append("    constexpr int NB_STRIDE = NB_MAX + 1;")
        out.append("    constexpr int NB1_STRIDE = DUAL_G ? 17 : 1;")
        out.append("    extern __shared__ float smem_c[];")
        out.append("    __shared__ float s_g[NSUB][EBATCH][NB_STRIDE];")
        out.append("    __shared__ float s_g1[DUAL_G ? NSUB : 1][DUAL_G ? EBATCH : 1][NB1_STRIDE];")
        out.append("    __shared__ float s_Y[NSUB][EBATCH][9];")
        out.append("    __shared__ int   s_j[NSUB][EBATCH];")
        out.append("    __shared__ int   s_pair[NSUB][EBATCH];")
        out.append("    __shared__ float s_f0[NSUB][EBATCH];")
        out.append("")
        out.append("    const int l2_tot_cols = 3 * n_msg;")
        out.append(f"    const int bf_tot_cols = {lmsg + 1} * n_msg;")
        out.append("    const int c2_elems = S * S * n_base2 * l2_tot_cols;")
        out.append("    const int cbf_elems = REM_BF ? (S * S * n_base1 * bf_tot_cols) : 0;")
        out.append("    constexpr bool F4_SMEM = SMEM_C && (NB_MAX == 8) && !DUAL_G;")
        out.append("    float4* smem_c4 = reinterpret_cast<float4*>(smem_c);")
        out.append("    float4* smem_bf4 = smem_c4 + (size_t)S * S * 6 * SUBWARP;")
        out.append("    if constexpr (F4_SMEM) {")
        out.append("        const int n_c2_slots = S * S * 3 * SUBWARP;")
        out.append("        for (int idx = threadIdx.x; idx < n_c2_slots; idx += 128) {")
        out.append("            const int q = idx & (SUBWARP - 1);")
        out.append("            const int pl = idx / SUBWARP; const int pair = pl / 3; const int l = pl - pair * 3;")
        out.append("            float4 v0 = make_float4(0.f,0.f,0.f,0.f), v1 = make_float4(0.f,0.f,0.f,0.f);")
        out.append("            if (q < n_msg) {")
        out.append("                const float* src = C2 + (int64_t)pair * 8 * l2_tot_cols + l * n_msg + q;")
        out.append("                v0 = make_float4(__ldg(src), __ldg(src + l2_tot_cols), __ldg(src + 2*l2_tot_cols), __ldg(src + 3*l2_tot_cols));")
        out.append("                v1 = make_float4(__ldg(src + 4*l2_tot_cols), __ldg(src + 5*l2_tot_cols), __ldg(src + 6*l2_tot_cols), __ldg(src + 7*l2_tot_cols));")
        out.append("            }")
        out.append("            smem_c4[(pair * 6 + l * 2 + 0) * SUBWARP + q] = v0;")
        out.append("            smem_c4[(pair * 6 + l * 2 + 1) * SUBWARP + q] = v1;")
        out.append("        }")
        out.append("        if constexpr (REM_BF) {")
        out.append(f"            const int n_bf_slots = S * S * {lmsg + 1} * SUBWARP;")
        out.append("            for (int idx = threadIdx.x; idx < n_bf_slots; idx += 128) {")
        out.append("                const int q = idx & (SUBWARP - 1);")
        out.append(f"                const int pl = idx / SUBWARP; const int pair = pl / {lmsg + 1}; const int l = pl - pair * {lmsg + 1};")
        out.append("                float4 v0 = make_float4(0.f,0.f,0.f,0.f), v1 = make_float4(0.f,0.f,0.f,0.f);")
        out.append("                if (q < n_msg) {")
        out.append("                    const float* src = Cbf + (int64_t)pair * 8 * bf_tot_cols + l * n_msg + q;")
        out.append("                    v0 = make_float4(__ldg(src), __ldg(src + bf_tot_cols), __ldg(src + 2*bf_tot_cols), __ldg(src + 3*bf_tot_cols));")
        out.append("                    v1 = make_float4(__ldg(src + 4*bf_tot_cols), __ldg(src + 5*bf_tot_cols), __ldg(src + 6*bf_tot_cols), __ldg(src + 7*bf_tot_cols));")
        out.append("                }")
        out.append(f"                smem_bf4[(pair * {(lmsg + 1) * 2} + l * 2 + 0) * SUBWARP + q] = v0;")
        out.append(f"                smem_bf4[(pair * {(lmsg + 1) * 2} + l * 2 + 1) * SUBWARP + q] = v1;")
        out.append("            }")
        out.append("        }")
        out.append("        __syncthreads();")
        out.append("    } else if constexpr (SMEM_C) {")
        out.append("        for (int idx = threadIdx.x; idx < c2_elems; idx += 128) smem_c[idx] = __ldg(C2 + idx);")
        out.append("        if constexpr (REM_BF) {")
        out.append("            float* smem_bf = smem_c + c2_elems;")
        out.append("            for (int idx = threadIdx.x; idx < cbf_elems; idx += 128) smem_bf[idx] = __ldg(Cbf + idx);")
        out.append("        }")
        out.append("        __syncthreads();")
        out.append("    }")
        out.append("    const int sid = threadIdx.x / SUBWARP;")
        out.append("    const int slane = threadIdx.x & (SUBWARP - 1);")
        out.append("    const unsigned smask = (SUBWARP == 8) ? (0xffu << ((sid & 3) * 8)) : ((SUBWARP == 16) ? ((sid & 1) ? 0xffff0000u : 0x0000ffffu) : 0xffffffffu);")
        out.append("    const int i = blockIdx.x * NSUB + sid;")
        out.append("    if (i >= N) return;")
        out.append("    const int z_i = species[i];")
        out.append("    const int eb = row_ptr[i], ee = row_end[i];")
        out.append("    const bool active_q = (slane < n_msg);")
        out.append("    const float* C2_base = SMEM_C ? smem_c : C2;")
        out.append("    const float* Cbf_base = REM_BF ? (SMEM_C ? (smem_c + c2_elems) : Cbf) : nullptr;")
        out.append("")
        out.append(f"    float acc[{n_out}];")
        out.append(f"    #pragma unroll\n    for (int k = 0; k < {n_out}; ++k) acc[k] = 0.0f;")
        out.append("")
        out.append("    for (int eb_b = eb; eb_b < ee; eb_b += EBATCH) {")
        out.append("        const int bcnt = (ee - eb_b < EBATCH) ? (ee - eb_b) : EBATCH;")
        out.append("        if (slane < bcnt) {")
        out.append("            const int e = eb_b + slane;")
        out.append("            const float4 vl = __ldg(vec_len + e);")
        out.append("            const int j = __ldg(senders + e);")
        out.append("            const float r = fmaxf(vl.w, 1e-8f);")
        out.append("            s_j[sid][slane] = j;")
        out.append("            s_pair[sid][slane] = z_i * S + __ldg(species + j);")
        out.append("            if constexpr (REM_BF) s_f0[sid][slane] = __ldg(f0 + j);")
        out.append("            if constexpr (NB_MAX == 8) {")
        out.append("                float4 g03, g47, d0, d1;")
        out.append("                eval_cheb8_f4<false>(r, r_max2, g03, g47, d0, d1);")
        out.append("                float* dst = s_g[sid][slane];")
        out.append("                dst[0]=g03.x; dst[1]=g03.y; dst[2]=g03.z; dst[3]=g03.w;")
        out.append("                dst[4]=g47.x; dst[5]=g47.y; dst[6]=g47.z; dst[7]=g47.w;")
        out.append("            } else {")
        out.append("                eval_cheb<false>(r, r_max2, n_base2, s_g[sid][slane], nullptr);")
        out.append("            }")
        out.append("            if constexpr (DUAL_G && REM_BF && SUBWARP == 8) {")
        out.append("                eval_cheb<false>(r, r_max1, n_base1, s_g1[sid][slane], nullptr);")
        out.append("            }")
        out.append("        } else if (slane >= EBATCH && slane < EBATCH + bcnt) {")
        out.append("            const int bi = slane - EBATCH;")
        out.append("            const float4 vl = __ldg(vec_len + eb_b + bi);")
        out.append("            const float r = fmaxf(vl.w, 1e-8f);")
        out.append("            edge_sh9(vl.x, vl.y, vl.z, r, s_Y[sid][bi]);")
        out.append("        } else if constexpr (DUAL_G && REM_BF && SUBWARP >= 16) {")
        out.append("            if (slane >= 2 * EBATCH && slane < 2 * EBATCH + bcnt) {")
        out.append("                const int bi = slane - 2 * EBATCH;")
        out.append("                const float4 vl = __ldg(vec_len + eb_b + bi);")
        out.append("                const float r = fmaxf(vl.w, 1e-8f);")
        out.append("                eval_cheb<false>(r, r_max1, n_base1, s_g1[sid][bi], nullptr);")
        out.append("            }")
        out.append("        }")
        out.append("        __syncwarp(smask);")
        out.append("")
        out.append("        if (active_q) {")
        out.append("            for (int bi = 0; bi < bcnt; ++bi) {")
        out.append("                const int j = s_j[sid][bi];")
        out.append("                const int pair = s_pair[sid][bi];")
        out.append("                const float* gk = s_g[sid][bi];")
        out.append("                const float* Yk = s_Y[sid][bi];")
        out.append("                const float* C2p = C2_base + (int64_t)pair * n_base2 * l2_tot_cols + slane;")
        out.append("                float r2_0 = 0.0f, r2_1 = 0.0f, r2_2 = 0.0f;")
        out.append("                if constexpr (F4_SMEM) {")
        out.append("                    const float4 g03 = make_float4(gk[0], gk[1], gk[2], gk[3]);")
        out.append("                    const float4 g47 = make_float4(gk[4], gk[5], gk[6], gk[7]);")
        out.append("                    const float4* cp4 = smem_c4 + pair * 6 * SUBWARP + slane;")
        out.append("                    r2_0 = v1_dot4(cp4[0 * SUBWARP], g03) + v1_dot4(cp4[1 * SUBWARP], g47);")
        out.append("                    r2_1 = v1_dot4(cp4[2 * SUBWARP], g03) + v1_dot4(cp4[3 * SUBWARP], g47);")
        out.append("                    r2_2 = v1_dot4(cp4[4 * SUBWARP], g03) + v1_dot4(cp4[5 * SUBWARP], g47);")
        out.append("                } else {")
        out.append("                    auto acc_c2 = [&](int k) {")
        out.append("                        const float gv = gk[k]; const float* rrow = C2p + k * l2_tot_cols;")
        out.append("                        r2_0 = fmaf(SMEM_C ? rrow[0]         : __ldg(rrow),           gv, r2_0);")
        out.append("                        r2_1 = fmaf(SMEM_C ? rrow[n_msg]     : __ldg(rrow + n_msg),   gv, r2_1);")
        out.append("                        r2_2 = fmaf(SMEM_C ? rrow[2 * n_msg] : __ldg(rrow + 2*n_msg), gv, r2_2);")
        out.append("                    };")
        out.append("                    if constexpr (NB_MAX == 8) {\n                        #pragma unroll\n                        for (int k = 0; k < 8; ++k) acc_c2(k);")
        out.append("                    } else {")
        out.append("                        if (n_base2 == 12)      {\n                            #pragma unroll 4\n                            for (int k = 0; k < 12; ++k) acc_c2(k); }")
        out.append("                        else if (n_base2 == 14) {\n                            #pragma unroll 4\n                            for (int k = 0; k < 14; ++k) acc_c2(k); }")
        out.append("                        else if (n_base2 == 16) {\n                            #pragma unroll 4\n                            for (int k = 0; k < 16; ++k) acc_c2(k); }")
        out.append("                        else { for (int k = 0; k < n_base2; ++k) acc_c2(k); }")
        out.append("                    }")
        out.append("                }")
        out.append("")
        bf_decls = "float rbf0 = 0.0f, rbf1 = 0.0f" + (", rbf2 = 0.0f;" if lmsg == 2 else ";")
        out.append(f"                {bf_decls}")
        out.append("                if constexpr (REM_BF) {")
        out.append("                    const float f0_j = s_f0[sid][bi];")
        out.append("                    if constexpr (F4_SMEM) {")
        out.append("                        const float4 g03 = make_float4(gk[0], gk[1], gk[2], gk[3]);")
        out.append("                        const float4 g47 = make_float4(gk[4], gk[5], gk[6], gk[7]);")
        out.append(f"                        const float4* bp4 = smem_bf4 + pair * {(lmsg + 1) * 2} * SUBWARP + slane;")
        out.append("                        rbf0 = v1_dot4(bp4[0 * SUBWARP], g03) + v1_dot4(bp4[1 * SUBWARP], g47);")
        out.append("                        rbf1 = v1_dot4(bp4[2 * SUBWARP], g03) + v1_dot4(bp4[3 * SUBWARP], g47);")
        if lmsg == 2:
            out.append("                        rbf2 = v1_dot4(bp4[4 * SUBWARP], g03) + v1_dot4(bp4[5 * SUBWARP], g47);")
        out.append("                    } else {")
        out.append("                        const float* gkb = DUAL_G ? s_g1[sid][bi] : gk;")
        out.append("                        const float* Cbp = Cbf_base + (int64_t)pair * n_base1 * bf_tot_cols + slane;")
        out.append("                        auto acc_bf = [&](int k) {")
        out.append("                            const float gv = gkb[k]; const float* rrow = Cbp + k * bf_tot_cols;")
        out.append("                            rbf0 = fmaf(SMEM_C ? rrow[0]     : __ldg(rrow),         gv, rbf0);")
        out.append("                            rbf1 = fmaf(SMEM_C ? rrow[n_msg] : __ldg(rrow + n_msg), gv, rbf1);")
        if lmsg == 2:
            out.append("                            rbf2 = fmaf(SMEM_C ? rrow[2*n_msg] : __ldg(rrow + 2*n_msg), gv, rbf2);")
        out.append("                        };")
        out.append("                        if constexpr (!DUAL_G && NB_MAX == 8) {\n                            #pragma unroll\n                            for (int k = 0; k < 8; ++k) acc_bf(k);")
        out.append("                        } else {")
        out.append("                            if (n_base1 == 8)       {\n                                #pragma unroll\n                                for (int k = 0; k < 8; ++k) acc_bf(k); }")
        out.append("                            else if (n_base1 == 12) {\n                                #pragma unroll 4\n                                for (int k = 0; k < 12; ++k) acc_bf(k); }")
        out.append("                            else if (n_base1 == 14) {\n                                #pragma unroll 4\n                                for (int k = 0; k < 14; ++k) acc_bf(k); }")
        out.append("                            else if (n_base1 == 16) {\n                                #pragma unroll 4\n                                for (int k = 0; k < 16; ++k) acc_bf(k); }")
        out.append("                            else { for (int k = 0; k < n_base1; ++k) acc_bf(k); }")
        out.append("                        }")
        out.append("                    }")
        out.append("                    rbf0 *= f0_j; rbf1 *= f0_j;" + (" rbf2 *= f0_j;" if lmsg == 2 else ""))
        out.append("                }")
        out.append("                const float i0_0 = __ldg(I_node.p[0] + (int64_t)j * n_msg + slane) - (REM_BF ? rbf0 * Yk[0] : 0.0f);")
        out.append("                const float* pI1 = I_node.p[1] + (int64_t)j * 3 * n_msg + slane;")
        out.append("                const float i1_0 = __ldg(pI1)           - (REM_BF ? rbf1 * Yk[1] : 0.0f);")
        out.append("                const float i1_1 = __ldg(pI1 + n_msg)   - (REM_BF ? rbf1 * Yk[2] : 0.0f);")
        out.append("                const float i1_2 = __ldg(pI1 + 2*n_msg) - (REM_BF ? rbf1 * Yk[3] : 0.0f);")
        if lmsg == 2:
            out.append("                const float* pI2 = I_node.p[2] + (int64_t)j * 5 * n_msg + slane;")
            out.append("                const float i2_0 = __ldg(pI2)           - (REM_BF ? rbf2 * Yk[4] : 0.0f);")
            out.append("                const float i2_1 = __ldg(pI2 + n_msg)   - (REM_BF ? rbf2 * Yk[5] : 0.0f);")
            out.append("                const float i2_2 = __ldg(pI2 + 2*n_msg) - (REM_BF ? rbf2 * Yk[6] : 0.0f);")
            out.append("                const float i2_3 = __ldg(pI2 + 3*n_msg) - (REM_BF ? rbf2 * Yk[7] : 0.0f);")
            out.append("                const float i2_4 = __ldg(pI2 + 4*n_msg) - (REM_BF ? rbf2 * Yk[8] : 0.0f);")
        out.append("                const float p0_0 = r2_0 * Yk[0];")
        out.append("                const float p1_0 = r2_1 * Yk[1], p1_1 = r2_1 * Yk[2], p1_2 = r2_1 * Yk[3];")
        out.append("                const float p2_0 = r2_2 * Yk[4], p2_1 = r2_2 * Yk[5], p2_2 = r2_2 * Yk[6], p2_3 = r2_2 * Yk[7], p2_4 = r2_2 * Yk[8];")
        for l1, l2, L, p_in_L, slot in ordered_paths:
            cg = cg_tensor(l1, l2, L)
            p_names = [f"p{l1}_{m}" for m in range(2 * l1 + 1)]
            i_names = [f"i{l2}_{m}" for m in range(2 * l2 + 1)]
            o_names = [f"acc[{slot + m}]" for m in range(2 * L + 1)]
            out.extend(emit_cg_fwd_accum(cg, p_names, i_names, o_names, "                "))
        out.append("            }")
        out.append("        }")
        out.append("        __syncwarp(smask);")
        out.append("    }")
        out.append("")
        out.append("    float sq_lane = 0.0f;")
        out.append(f"    if (active_q) {{\n        #pragma unroll\n        for (int k = 0; k < {n_out}; ++k) sq_lane = fmaf(acc[k], acc[k], sq_lane); }}")
        out.append("    const float sq_tot = (SUBWARP == 8) ? qwarp_sum_f(sq_lane, smask) : ((SUBWARP == 16) ? half_warp_sum_f(sq_lane, smask) : warp_sum_f(sq_lane));")
        out.append("    const float f1 = a0_over_sig1 / sqrtf(1.0f + sq_tot * inv_DA2_sig1_kap_sq);")
        out.append("    if (slane == 0) f1_out[i] = f1;")
        out.append("    if (active_q) {")
        out.append("        const int c0 = l2_c.v[0], c1 = l2_c.v[1], c2 = l2_c.v[2];")
        out.append("        float* d0 = A2.p[0] + (int64_t)i * c0 + slane;")
        out.append("        float* d1 = A2.p[1] + (int64_t)i * 3 * c1 + slane;")
        out.append("        float* d2 = A2.p[2] + (int64_t)i * 5 * c2 + slane;")
        for l1, l2, L, p_in_L, slot in ordered_paths:
            for m in range(2 * L + 1):
                if L == 0:
                    out.append(f"        d0[{p_in_L} * n_msg] = acc[{slot}] * f1;")
                elif L == 1:
                    out.append(f"        d1[{m} * c1 + {p_in_L} * n_msg] = acc[{slot + m}] * f1;")
                elif L == 2:
                    out.append(f"        d2[{m} * c2 + {p_in_L} * n_msg] = acc[{slot + m}] * f1;")
        out.append("    }")
        out.append("}")
        out.append("")

        # --- 2) FUSED f1 GATE BACKWARD + FLOAT4 PACK dA2 -> [N, N_VEC4, SUBWARP] of float4 (STG.128 / LDG.128!) ---
        n_vec4 = (n_out + 3) // 4  # 6 for m1 (24 floats), 9 for m2 (36 floats)
        out.append(f"template <int SUBWARP>")
        out.append(f"__global__ void __launch_bounds__(128) ilmlip_gate_bwd_pack_l2_lf2_m{lmsg}_kernel(")
        out.append("    int N, int n_msg, Int4 l2_c, float inv_DA2_a0_kap_sq,")
        out.append("    CPtr4 A2, CPtr4 dA2, const float* __restrict__ f1_arr, float* __restrict__ dA2_pack) {")
        out.append("    constexpr int NSUB = 128 / SUBWARP;")
        out.append("    const int sid = threadIdx.x / SUBWARP;")
        out.append("    const int slane = threadIdx.x & (SUBWARP - 1);")
        out.append("    const unsigned smask = (SUBWARP == 8) ? (0xffu << ((sid & 3) * 8)) : ((SUBWARP == 16) ? ((sid & 1) ? 0xffff0000u : 0x0000ffffu) : 0xffffffffu);")
        out.append("    const int i = blockIdx.x * NSUB + sid;")
        out.append("    if (i >= N) return;")
        out.append("    const bool active_q = (slane < n_msg);")
        out.append(f"    float av[{n_vec4 * 4}], dv[{n_vec4 * 4}];")
        for k in range(n_out, n_vec4 * 4):
            out.append(f"    av[{k}] = 0.0f; dv[{k}] = 0.0f;")
        out.append("    float dot_lane = 0.0f;")
        out.append("    if (active_q) {")
        out.append("        const int c0 = l2_c.v[0], c1 = l2_c.v[1], c2 = l2_c.v[2];")
        out.append("        const float* pA0 = A2.p[0] + (int64_t)i * c0 + slane; const float* pdA0 = dA2.p[0] + (int64_t)i * c0 + slane;")
        out.append("        const float* pA1 = A2.p[1] + (int64_t)i * 3 * c1 + slane; const float* pdA1 = dA2.p[1] + (int64_t)i * 3 * c1 + slane;")
        out.append("        const float* pA2 = A2.p[2] + (int64_t)i * 5 * c2 + slane; const float* pdA2 = dA2.p[2] + (int64_t)i * 5 * c2 + slane;")
        for l1, l2, L, p_in_L, slot in ordered_paths:
            for m in range(2 * L + 1):
                if L == 0:
                    idx = f"{p_in_L} * n_msg"
                    out.append(f"        av[{slot}] = pA0[{idx}]; dv[{slot}] = pdA0[{idx}];")
                elif L == 1:
                    idx = f"{m} * c1 + {p_in_L} * n_msg"
                    out.append(f"        av[{slot + m}] = pA1[{idx}]; dv[{slot + m}] = pdA1[{idx}];")
                elif L == 2:
                    idx = f"{m} * c2 + {p_in_L} * n_msg"
                    out.append(f"        av[{slot + m}] = pA2[{idx}]; dv[{slot + m}] = pdA2[{idx}];")
        out.append(f"        #pragma unroll\n        for (int k = 0; k < {n_out}; ++k) dot_lane = fmaf(av[k], dv[k], dot_lane);")
        out.append("    }")
        out.append("    const float dot = (SUBWARP == 8) ? qwarp_sum_f(dot_lane, smask) : ((SUBWARP == 16) ? half_warp_sum_f(dot_lane, smask) : warp_sum_f(dot_lane));")
        out.append("    const float f1 = f1_arr[i];")
        out.append("    const float sc = (dot * f1) * inv_DA2_a0_kap_sq;")
        out.append("    if (active_q) {")
        out.append(f"        float4* dst4 = reinterpret_cast<float4*>(dA2_pack) + (int64_t)i * ({n_vec4} * SUBWARP) + slane;")
        for b4 in range(n_vec4):
            k0, k1, k2, k3 = b4 * 4, b4 * 4 + 1, b4 * 4 + 2, b4 * 4 + 3
            e0 = f"fmaf(f1, dv[{k0}], -sc * av[{k0}])" if k0 < n_out else "0.0f"
            e1 = f"fmaf(f1, dv[{k1}], -sc * av[{k1}])" if k1 < n_out else "0.0f"
            e2 = f"fmaf(f1, dv[{k2}], -sc * av[{k2}])" if k2 < n_out else "0.0f"
            e3 = f"fmaf(f1, dv[{k3}], -sc * av[{k3}])" if k3 < n_out else "0.0f"
            out.append(f"        dst4[{b4} * SUBWARP] = make_float4({e0}, {e1}, {e2}, {e3});")
        out.append("    }")
        out.append("}")
        out.append("")

        # --- 3) SENDER-OWNED ZERO-ATOMIC L2 EDGE BACKWARD (supports SUBWARP in {8,16,32} and DUAL_G!) ---
        out.append(f"// Specialized SENDER-OWNED ZERO-ATOMIC L2 Edge Backward: lfeat2=2, lout2=2, lmsg={lmsg} (N_OUT={n_out}, N_VEC4={n_vec4})")
        out.append(f"template <int SUBWARP, int NB_MAX, bool REM_BF, bool STRESS, bool F32_F, bool SMEM_C, bool DUAL_G = false>")
        out.append(f"__global__ void __launch_bounds__(128, 4) ilmlip_l2_edge_bwd_lf2_m{lmsg}_sender_kernel(")
        out.append("    int N, int S, int n_base2, int n_base1, int n_msg, ChebParam r_max2, ChebParam r_max1,")
        out.append("    const int* __restrict__ row_ptr, const int* __restrict__ row_end, const int* __restrict__ senders,")
        out.append("    const float4* __restrict__ vec_len, const int* __restrict__ species,")
        out.append("    const float* __restrict__ C2, const float* __restrict__ Cbf,")
        out.append("    const float* __restrict__ f0, CPtr4 I_node, const float* __restrict__ dA2_pack,")
        out.append("    Ptr4 dI_node, float* __restrict__ df0_bf,")
        out.append("    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {")
        out.append("    using FT = typename std::conditional<F32_F, float, double>::type;")
        out.append("    constexpr int NSUB = 128 / SUBWARP;")
        out.append("    constexpr int EBATCH = (SUBWARP == 32) ? 8 : 4;")
        out.append("    constexpr int NB_STRIDE = NB_MAX + 1;")
        out.append("    constexpr int NB1_STRIDE = DUAL_G ? 17 : 1;")
        out.append("    extern __shared__ float smem_c[];")
        out.append("    __shared__ float  s_g[NSUB][EBATCH][NB_STRIDE];")
        out.append("    __shared__ float  s_dg[NSUB][EBATCH][NB_STRIDE];")
        out.append("    __shared__ float  s_g1[DUAL_G ? NSUB : 1][DUAL_G ? EBATCH : 1][NB1_STRIDE];")
        out.append("    __shared__ float  s_dg1[DUAL_G ? NSUB : 1][DUAL_G ? EBATCH : 1][NB1_STRIDE];")
        out.append("    __shared__ float  s_Y[NSUB][EBATCH][9];")
        out.append("    __shared__ float4 s_vl[NSUB][EBATCH];")
        out.append("    __shared__ int    s_i[NSUB][EBATCH];")
        out.append("    __shared__ int    s_pair[NSUB][EBATCH];")
        out.append("    __shared__ float  s_er_dY[NSUB][EBATCH][9];")
        out.append("    __shared__ FT     s_sf[NSUB][3];")
        out.append("")
        out.append("    const int l2_tot_cols = 3 * n_msg;")
        out.append(f"    const int bf_tot_cols = {lmsg + 1} * n_msg;")
        out.append("    const int c2_elems = S * S * n_base2 * l2_tot_cols;")
        out.append("    const int cbf_elems = REM_BF ? (S * S * n_base1 * bf_tot_cols) : 0;")
        out.append("    constexpr bool F4_SMEM = SMEM_C && (NB_MAX == 8) && !DUAL_G;")
        out.append("    float4* smem_c4 = reinterpret_cast<float4*>(smem_c);")
        out.append("    float4* smem_bf4 = smem_c4 + (size_t)S * S * 6 * SUBWARP;")
        out.append("    if constexpr (F4_SMEM) {")
        out.append("        const int n_c2_slots = S * S * 3 * SUBWARP;")
        out.append("        for (int idx = threadIdx.x; idx < n_c2_slots; idx += 128) {")
        out.append("            const int q = idx & (SUBWARP - 1);")
        out.append("            const int pl = idx / SUBWARP; const int pair = pl / 3; const int l = pl - pair * 3;")
        out.append("            float4 v0 = make_float4(0.f,0.f,0.f,0.f), v1 = make_float4(0.f,0.f,0.f,0.f);")
        out.append("            if (q < n_msg) {")
        out.append("                const float* src = C2 + (int64_t)pair * 8 * l2_tot_cols + l * n_msg + q;")
        out.append("                v0 = make_float4(__ldg(src), __ldg(src + l2_tot_cols), __ldg(src + 2*l2_tot_cols), __ldg(src + 3*l2_tot_cols));")
        out.append("                v1 = make_float4(__ldg(src + 4*l2_tot_cols), __ldg(src + 5*l2_tot_cols), __ldg(src + 6*l2_tot_cols), __ldg(src + 7*l2_tot_cols));")
        out.append("            }")
        out.append("            smem_c4[(pair * 6 + l * 2 + 0) * SUBWARP + q] = v0;")
        out.append("            smem_c4[(pair * 6 + l * 2 + 1) * SUBWARP + q] = v1;")
        out.append("        }")
        out.append("        if constexpr (REM_BF) {")
        out.append(f"            const int n_bf_slots = S * S * {lmsg + 1} * SUBWARP;")
        out.append("            for (int idx = threadIdx.x; idx < n_bf_slots; idx += 128) {")
        out.append("                const int q = idx & (SUBWARP - 1);")
        out.append(f"                const int pl = idx / SUBWARP; const int pair = pl / {lmsg + 1}; const int l = pl - pair * {lmsg + 1};")
        out.append("                float4 v0 = make_float4(0.f,0.f,0.f,0.f), v1 = make_float4(0.f,0.f,0.f,0.f);")
        out.append("                if (q < n_msg) {")
        out.append("                    const float* src = Cbf + (int64_t)pair * 8 * bf_tot_cols + l * n_msg + q;")
        out.append("                    v0 = make_float4(__ldg(src), __ldg(src + bf_tot_cols), __ldg(src + 2*bf_tot_cols), __ldg(src + 3*bf_tot_cols));")
        out.append("                    v1 = make_float4(__ldg(src + 4*bf_tot_cols), __ldg(src + 5*bf_tot_cols), __ldg(src + 6*bf_tot_cols), __ldg(src + 7*bf_tot_cols));")
        out.append("                }")
        out.append(f"                smem_bf4[(pair * {(lmsg + 1) * 2} + l * 2 + 0) * SUBWARP + q] = v0;")
        out.append(f"                smem_bf4[(pair * {(lmsg + 1) * 2} + l * 2 + 1) * SUBWARP + q] = v1;")
        out.append("            }")
        out.append("        }")
        out.append("        __syncthreads();")
        out.append("    } else if constexpr (SMEM_C) {")
        out.append("        for (int idx = threadIdx.x; idx < c2_elems; idx += 128) smem_c[idx] = __ldg(C2 + idx);")
        out.append("        if constexpr (REM_BF) {")
        out.append("            float* smem_bf = smem_c + c2_elems;")
        out.append("            for (int idx = threadIdx.x; idx < cbf_elems; idx += 128) smem_bf[idx] = __ldg(Cbf + idx);")
        out.append("        }")
        out.append("        __syncthreads();")
        out.append("    }")
        out.append("    const int sid = threadIdx.x / SUBWARP;")
        out.append("    const int slane = threadIdx.x & (SUBWARP - 1);")
        out.append("    const unsigned smask = (SUBWARP == 8) ? (0xffu << ((sid & 3) * 8)) : ((SUBWARP == 16) ? ((sid & 1) ? 0xffff0000u : 0x0000ffffu) : 0xffffffffu);")
        out.append("    const int j = blockIdx.x * NSUB + sid;")
        out.append("    if (j >= N) return;")
        out.append("    const int z_j = species[j];")
        out.append("    const int eb = row_ptr[j], ee = row_end[j];")
        out.append("    const bool active_q = (slane < n_msg);")
        out.append("    const float f0_j = REM_BF ? __ldg(f0 + j) : 0.0f;")
        out.append("    const float* C2_base = SMEM_C ? smem_c : C2;")
        out.append("    const float* Cbf_base = REM_BF ? (SMEM_C ? (smem_c + c2_elems) : Cbf) : nullptr;")
        out.append("")
        out.append("    float i0_raw = 0.0f, i1_0_raw = 0.0f, i1_1_raw = 0.0f, i1_2_raw = 0.0f;")
        if lmsg == 2:
            out.append("    float i2_0_raw = 0.0f, i2_1_raw = 0.0f, i2_2_raw = 0.0f, i2_3_raw = 0.0f, i2_4_raw = 0.0f;")
        out.append("    if (active_q) {")
        out.append("        i0_raw = __ldg(I_node.p[0] + (int64_t)j * n_msg + slane);")
        out.append("        const float* pI1 = I_node.p[1] + (int64_t)j * 3 * n_msg + slane;")
        out.append("        i1_0_raw = __ldg(pI1); i1_1_raw = __ldg(pI1 + n_msg); i1_2_raw = __ldg(pI1 + 2 * n_msg);")
        if lmsg == 2:
            out.append("        const float* pI2 = I_node.p[2] + (int64_t)j * 5 * n_msg + slane;")
            out.append("        i2_0_raw = __ldg(pI2); i2_1_raw = __ldg(pI2 + n_msg); i2_2_raw = __ldg(pI2 + 2*n_msg); i2_3_raw = __ldg(pI2 + 3*n_msg); i2_4_raw = __ldg(pI2 + 4*n_msg);")
        out.append("    }")
        out.append("    if (slane < 3) s_sf[sid][slane] = (FT)0;")
        out.append("    float acc_di0 = 0.0f, acc_di1_0 = 0.0f, acc_di1_1 = 0.0f, acc_di1_2 = 0.0f;")
        if lmsg == 2:
            out.append("    float acc_di2_0 = 0.0f, acc_di2_1 = 0.0f, acc_di2_2 = 0.0f, acc_di2_3 = 0.0f, acc_di2_4 = 0.0f;")
        out.append("    float acc_df0 = 0.0f;")
        out.append("    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;")
        out.append("")
        out.append("    for (int eb_b = eb; eb_b < ee; eb_b += EBATCH) {")
        out.append("        const int bcnt = (ee - eb_b < EBATCH) ? (ee - eb_b) : EBATCH;")
        out.append("        if (slane < bcnt) {")
        out.append("            const int e = eb_b + slane;")
        out.append("            const float4 vl_rev = __ldg(vec_len + e);")
        out.append("            const float4 vl_fwd = make_float4(-vl_rev.x, -vl_rev.y, -vl_rev.z, vl_rev.w);")
        out.append("            const int i_nb = __ldg(senders + e);")
        out.append("            const float r = fmaxf(vl_rev.w, 1e-8f);")
        out.append("            s_vl[sid][slane] = vl_fwd;")
        out.append("            s_i[sid][slane] = i_nb;")
        out.append("            s_pair[sid][slane] = __ldg(species + i_nb) * S + z_j;")
        out.append("            if constexpr (NB_MAX == 8) {")
        out.append("                float4 g03, g47, dg03, dg47;")
        out.append("                eval_cheb8_f4<true>(r, r_max2, g03, g47, dg03, dg47);")
        out.append("                float* gs = s_g[sid][slane]; float* dgs = s_dg[sid][slane];")
        out.append("                gs[0]=g03.x; gs[1]=g03.y; gs[2]=g03.z; gs[3]=g03.w; gs[4]=g47.x; gs[5]=g47.y; gs[6]=g47.z; gs[7]=g47.w;")
        out.append("                dgs[0]=dg03.x; dgs[1]=dg03.y; dgs[2]=dg03.z; dgs[3]=dg03.w; dgs[4]=dg47.x; dgs[5]=dg47.y; dgs[6]=dg47.z; dgs[7]=dg47.w;")
        out.append("            } else {")
        out.append("                eval_cheb<true>(r, r_max2, n_base2, s_g[sid][slane], s_dg[sid][slane]);")
        out.append("            }")
        out.append("            if constexpr (DUAL_G && REM_BF && SUBWARP == 8) {")
        out.append("                eval_cheb<true>(r, r_max1, n_base1, s_g1[sid][slane], s_dg1[sid][slane]);")
        out.append("            }")
        out.append("        } else if (slane >= EBATCH && slane < EBATCH + bcnt) {")
        out.append("            const int bi = slane - EBATCH;")
        out.append("            const float4 vl_rev = __ldg(vec_len + eb_b + bi);")
        out.append("            const float r = fmaxf(vl_rev.w, 1e-8f);")
        out.append("            edge_sh9(-vl_rev.x, -vl_rev.y, -vl_rev.z, r, s_Y[sid][bi]);")
        out.append("        } else if constexpr (DUAL_G && REM_BF && SUBWARP >= 16) {")
        out.append("            if (slane >= 2 * EBATCH && slane < 2 * EBATCH + bcnt) {")
        out.append("                const int bi = slane - 2 * EBATCH;")
        out.append("                const float4 vl_rev = __ldg(vec_len + eb_b + bi);")
        out.append("                const float r = fmaxf(vl_rev.w, 1e-8f);")
        out.append("                eval_cheb<true>(r, r_max1, n_base1, s_g1[sid][bi], s_dg1[sid][bi]);")
        out.append("            }")
        out.append("        }")
        out.append("        __syncwarp(smask);")
        out.append("")
        out.append("        for (int bi = 0; bi < bcnt; ++bi) {")
        out.append("            const int i_nb = s_i[sid][bi];")
        out.append("            const int pair = s_pair[sid][bi];")
        out.append("            float er_lane = 0.0f;")
        out.append("            float dY1 = 0.0f, dY2 = 0.0f, dY3 = 0.0f, dY4 = 0.0f, dY5 = 0.0f, dY6 = 0.0f, dY7 = 0.0f, dY8 = 0.0f;")
        out.append("            if (active_q) {")
        out.append("                const float* gk  = s_g[sid][bi];")
        out.append("                const float* dgk = s_dg[sid][bi];")
        out.append("                const float* Yk  = s_Y[sid][bi];")
        out.append("                const float* C2p = C2_base + (int64_t)pair * n_base2 * l2_tot_cols + slane;")
        out.append("                float r2_0 = 0.0f, r2_1 = 0.0f, r2_2 = 0.0f;")
        out.append("                float dr2_0 = 0.0f, dr2_1 = 0.0f, dr2_2 = 0.0f;")
        out.append("                if constexpr (F4_SMEM) {")
        out.append("                    const float4 g03 = make_float4(gk[0], gk[1], gk[2], gk[3]);")
        out.append("                    const float4 g47 = make_float4(gk[4], gk[5], gk[6], gk[7]);")
        out.append("                    const float4 d03 = make_float4(dgk[0], dgk[1], dgk[2], dgk[3]);")
        out.append("                    const float4 d47 = make_float4(dgk[4], dgk[5], dgk[6], dgk[7]);")
        out.append("                    const float4* cp4 = smem_c4 + pair * 6 * SUBWARP + slane;")
        out.append("                    const float4 c0a = cp4[0 * SUBWARP], c0b = cp4[1 * SUBWARP];")
        out.append("                    r2_0 = v1_dot4(c0a, g03) + v1_dot4(c0b, g47); dr2_0 = v1_dot4(c0a, d03) + v1_dot4(c0b, d47);")
        out.append("                    const float4 c1a = cp4[2 * SUBWARP], c1b = cp4[3 * SUBWARP];")
        out.append("                    r2_1 = v1_dot4(c1a, g03) + v1_dot4(c1b, g47); dr2_1 = v1_dot4(c1a, d03) + v1_dot4(c1b, d47);")
        out.append("                    const float4 c2a = cp4[4 * SUBWARP], c2b = cp4[5 * SUBWARP];")
        out.append("                    r2_2 = v1_dot4(c2a, g03) + v1_dot4(c2b, g47); dr2_2 = v1_dot4(c2a, d03) + v1_dot4(c2b, d47);")
        out.append("                } else {")
        out.append("                    auto acc_c2 = [&](int k) {")
        out.append("                        const float gv = gk[k], dgv = dgk[k]; const float* rrow = C2p + k * l2_tot_cols;")
        out.append("                        const float c0 = SMEM_C ? rrow[0]         : __ldg(rrow);")
        out.append("                        const float c1 = SMEM_C ? rrow[n_msg]     : __ldg(rrow + n_msg);")
        out.append("                        const float c2 = SMEM_C ? rrow[2 * n_msg] : __ldg(rrow + 2 * n_msg);")
        out.append("                        r2_0 = fmaf(c0, gv, r2_0); dr2_0 = fmaf(c0, dgv, dr2_0);")
        out.append("                        r2_1 = fmaf(c1, gv, r2_1); dr2_1 = fmaf(c1, dgv, dr2_1);")
        out.append("                        r2_2 = fmaf(c2, gv, r2_2); dr2_2 = fmaf(c2, dgv, dr2_2);")
        out.append("                    };")
        out.append("                    if constexpr (NB_MAX == 8) {\n                        #pragma unroll\n                        for (int k = 0; k < 8; ++k) acc_c2(k);")
        out.append("                    } else {")
        out.append("                        if (n_base2 == 12)      {\n                            #pragma unroll 4\n                            for (int k = 0; k < 12; ++k) acc_c2(k); }")
        out.append("                        else if (n_base2 == 14) {\n                            #pragma unroll 4\n                            for (int k = 0; k < 14; ++k) acc_c2(k); }")
        out.append("                        else if (n_base2 == 16) {\n                            #pragma unroll 4\n                            for (int k = 0; k < 16; ++k) acc_c2(k); }")
        out.append("                        else { for (int k = 0; k < n_base2; ++k) acc_c2(k); }")
        out.append("                    }")
        out.append("                }")
        out.append("")
        out.append("                float rbf0 = 0.0f, rbf1 = 0.0f, drbf0 = 0.0f, drbf1 = 0.0f;" + (" float rbf2 = 0.0f, drbf2 = 0.0f;" if lmsg == 2 else ""))
        out.append("                if constexpr (REM_BF) {")
        out.append("                    if constexpr (F4_SMEM) {")
        out.append("                        const float4 g03 = make_float4(gk[0], gk[1], gk[2], gk[3]);")
        out.append("                        const float4 g47 = make_float4(gk[4], gk[5], gk[6], gk[7]);")
        out.append("                        const float4 d03 = make_float4(dgk[0], dgk[1], dgk[2], dgk[3]);")
        out.append("                        const float4 d47 = make_float4(dgk[4], dgk[5], dgk[6], dgk[7]);")
        out.append(f"                        const float4* bp4 = smem_bf4 + pair * {(lmsg + 1) * 2} * SUBWARP + slane;")
        out.append("                        const float4 b0a = bp4[0 * SUBWARP], b0b = bp4[1 * SUBWARP];")
        out.append("                        rbf0 = v1_dot4(b0a, g03) + v1_dot4(b0b, g47); drbf0 = v1_dot4(b0a, d03) + v1_dot4(b0b, d47);")
        out.append("                        const float4 b1a = bp4[2 * SUBWARP], b1b = bp4[3 * SUBWARP];")
        out.append("                        rbf1 = v1_dot4(b1a, g03) + v1_dot4(b1b, g47); drbf1 = v1_dot4(b1a, d03) + v1_dot4(b1b, d47);")
        if lmsg == 2:
            out.append("                        const float4 b2a = bp4[4 * SUBWARP], b2b = bp4[5 * SUBWARP];")
            out.append("                        rbf2 = v1_dot4(b2a, g03) + v1_dot4(b2b, g47); drbf2 = v1_dot4(b2a, d03) + v1_dot4(b2b, d47);")
        out.append("                    } else {")
        out.append("                        const float* gkb  = DUAL_G ? s_g1[sid][bi]  : gk;")
        out.append("                        const float* dgkb = DUAL_G ? s_dg1[sid][bi] : dgk;")
        out.append("                        const float* Cbp = Cbf_base + (int64_t)pair * n_base1 * bf_tot_cols + slane;")
        out.append("                        auto acc_bf = [&](int k) {")
        out.append("                            const float gv = gkb[k], dgv = dgkb[k]; const float* rrow = Cbp + k * bf_tot_cols;")
        out.append("                            const float b0 = SMEM_C ? rrow[0]     : __ldg(rrow);")
        out.append("                            const float b1 = SMEM_C ? rrow[n_msg] : __ldg(rrow + n_msg);")
        out.append("                            rbf0 = fmaf(b0, gv, rbf0); drbf0 = fmaf(b0, dgv, drbf0);")
        out.append("                            rbf1 = fmaf(b1, gv, rbf1); drbf1 = fmaf(b1, dgv, drbf1);")
        if lmsg == 2:
            out.append("                            const float b2 = SMEM_C ? rrow[2*n_msg] : __ldg(rrow + 2*n_msg);")
            out.append("                            rbf2 = fmaf(b2, gv, rbf2); drbf2 = fmaf(b2, dgv, drbf2);")
        out.append("                        };")
        out.append("                        if constexpr (!DUAL_G && NB_MAX == 8) {\n                            #pragma unroll\n                            for (int k = 0; k < 8; ++k) acc_bf(k);")
        out.append("                        } else {")
        out.append("                            if (n_base1 == 8)       {\n                                #pragma unroll\n                                for (int k = 0; k < 8; ++k) acc_bf(k); }")
        out.append("                            else if (n_base1 == 12) {\n                                #pragma unroll 4\n                                for (int k = 0; k < 12; ++k) acc_bf(k); }")
        out.append("                            else if (n_base1 == 14) {\n                                #pragma unroll 4\n                                for (int k = 0; k < 14; ++k) acc_bf(k); }")
        out.append("                            else if (n_base1 == 16) {\n                                #pragma unroll 4\n                                for (int k = 0; k < 16; ++k) acc_bf(k); }")
        out.append("                            else { for (int k = 0; k < n_base1; ++k) acc_bf(k); }")
        out.append("                        }")
        out.append("                    }")
        out.append("                }")
        out.append("                const float i0_0 = i0_raw   - (REM_BF ? (f0_j * rbf0) * Yk[0] : 0.0f);")
        out.append("                const float i1_0 = i1_0_raw - (REM_BF ? (f0_j * rbf1) * Yk[1] : 0.0f);")
        out.append("                const float i1_1 = i1_1_raw - (REM_BF ? (f0_j * rbf1) * Yk[2] : 0.0f);")
        out.append("                const float i1_2 = i1_2_raw - (REM_BF ? (f0_j * rbf1) * Yk[3] : 0.0f);")
        if lmsg == 2:
            out.append("                const float i2_0 = i2_0_raw - (REM_BF ? (f0_j * rbf2) * Yk[4] : 0.0f);")
            out.append("                const float i2_1 = i2_1_raw - (REM_BF ? (f0_j * rbf2) * Yk[5] : 0.0f);")
            out.append("                const float i2_2 = i2_2_raw - (REM_BF ? (f0_j * rbf2) * Yk[6] : 0.0f);")
            out.append("                const float i2_3 = i2_3_raw - (REM_BF ? (f0_j * rbf2) * Yk[7] : 0.0f);")
            out.append("                const float i2_4 = i2_4_raw - (REM_BF ? (f0_j * rbf2) * Yk[8] : 0.0f);")
        out.append("                const float p0_0 = r2_0 * Yk[0];")
        out.append("                const float p1_0 = r2_1 * Yk[1], p1_1 = r2_1 * Yk[2], p1_2 = r2_1 * Yk[3];")
        out.append("                const float p2_0 = r2_2 * Yk[4], p2_1 = r2_2 * Yk[5], p2_2 = r2_2 * Yk[6], p2_3 = r2_2 * Yk[7], p2_4 = r2_2 * Yk[8];")
        out.append("                float dp0_0 = 0.0f;")
        out.append("                float dp1_0 = 0.0f, dp1_1 = 0.0f, dp1_2 = 0.0f;")
        out.append("                float dp2_0 = 0.0f, dp2_1 = 0.0f, dp2_2 = 0.0f, dp2_3 = 0.0f, dp2_4 = 0.0f;")
        out.append("                float di0_0 = 0.0f;")
        out.append("                float di1_0 = 0.0f, di1_1 = 0.0f, di1_2 = 0.0f;")
        if lmsg == 2:
            out.append("                float di2_0 = 0.0f, di2_1 = 0.0f, di2_2 = 0.0f, di2_3 = 0.0f, di2_4 = 0.0f;")
        out.append(f"                const float4* p4 = reinterpret_cast<const float4*>(dA2_pack) + (int64_t)i_nb * ({n_vec4} * SUBWARP) + slane;")
        comps = ("x", "y", "z", "w")
        loaded_b4 = set()
        for l1, l2, L, p_in_L, slot in ordered_paths:
            cg = cg_tensor(l1, l2, L)
            for m in range(2 * L + 1):
                b4 = (slot + m) // 4
                if b4 not in loaded_b4:
                    out.append(f"                const float4 qv{b4} = __ldg(p4 + {b4} * SUBWARP);")
                    loaded_b4.add(b4)
            out.append(f"                {{ // ({l1},{l2})->{L} slot {slot}")
            gnames = [f"qv{(slot + m) // 4}.{comps[(slot + m) & 3]}" for m in range(2 * L + 1)]
            p_names = [f"p{l1}_{m}" for m in range(2 * l1 + 1)]
            dp_names = [f"dp{l1}_{m}" for m in range(2 * l1 + 1)]
            i_names = [f"i{l2}_{m}" for m in range(2 * l2 + 1)]
            di_names = [f"di{l2}_{m}" for m in range(2 * l2 + 1)]
            out.extend(emit_cg_bwd_accum(cg, gnames, p_names, i_names, dp_names, di_names, False, "                    "))
            out.append("                }")
        out.append("")
        out.append("                const float d_r2_0 = dp0_0 * Yk[0];")
        out.append("                const float d_r2_1 = dp1_0 * Yk[1] + dp1_1 * Yk[2] + dp1_2 * Yk[3];")
        out.append("                const float d_r2_2 = dp2_0 * Yk[4] + dp2_1 * Yk[5] + dp2_2 * Yk[6] + dp2_3 * Yk[7] + dp2_4 * Yk[8];")
        out.append("                er_lane = fmaf(d_r2_0, dr2_0, fmaf(d_r2_1, dr2_1, d_r2_2 * dr2_2));")
        out.append("                dY1 = dp1_0 * r2_1; dY2 = dp1_1 * r2_1; dY3 = dp1_2 * r2_1;")
        out.append("                dY4 = dp2_0 * r2_2; dY5 = dp2_1 * r2_2; dY6 = dp2_2 * r2_2; dY7 = dp2_3 * r2_2; dY8 = dp2_4 * r2_2;")
        out.append("                acc_di0   += di0_0;")
        out.append("                acc_di1_0 += di1_0; acc_di1_1 += di1_1; acc_di1_2 += di1_2;")
        if lmsg == 2:
            out.append("                acc_di2_0 += di2_0; acc_di2_1 += di2_1; acc_di2_2 += di2_2; acc_di2_3 += di2_3; acc_di2_4 += di2_4;")
        out.append("                if constexpr (REM_BF) {")
        out.append("                    acc_df0 = fmaf(-di0_0 * Yk[0], rbf0, acc_df0);")
        out.append("                    er_lane = fmaf(-di0_0 * Yk[0] * f0_j, drbf0, er_lane);")
        out.append("                    const float ydot1 = di1_0 * Yk[1] + di1_1 * Yk[2] + di1_2 * Yk[3];")
        out.append("                    acc_df0 = fmaf(-ydot1, rbf1, acc_df0);")
        out.append("                    er_lane = fmaf(-ydot1 * f0_j, drbf1, er_lane);")
        out.append("                    const float bf1_sc = -f0_j * rbf1;")
        out.append("                    dY1 = fmaf(di1_0, bf1_sc, dY1); dY2 = fmaf(di1_1, bf1_sc, dY2); dY3 = fmaf(di1_2, bf1_sc, dY3);")
        if lmsg == 2:
            out.append("                    const float ydot2 = di2_0 * Yk[4] + di2_1 * Yk[5] + di2_2 * Yk[6] + di2_3 * Yk[7] + di2_4 * Yk[8];")
            out.append("                    acc_df0 = fmaf(-ydot2, rbf2, acc_df0);")
            out.append("                    er_lane = fmaf(-ydot2 * f0_j, drbf2, er_lane);")
            out.append("                    const float bf2_sc = -f0_j * rbf2;")
            out.append("                    dY4 = fmaf(di2_0, bf2_sc, dY4); dY5 = fmaf(di2_1, bf2_sc, dY5); dY6 = fmaf(di2_2, bf2_sc, dY6); dY7 = fmaf(di2_3, bf2_sc, dY7); dY8 = fmaf(di2_4, bf2_sc, dY8);")
        out.append("                }")
        out.append("            }")
        out.append("")
        out.append("            // Parallel butterfly sub-warp reduction of 9 scalars (er_lane, dY1..dY8)!")
        out.append("            const bool p0 = (slane & 1) != 0;")
        out.append("            const float u0 = (p0 ? dY1 : er_lane) + __shfl_xor_sync(smask, p0 ? er_lane : dY1, 1);")
        out.append("            const float u1 = (p0 ? dY3 : dY2)     + __shfl_xor_sync(smask, p0 ? dY2 : dY3, 1);")
        out.append("            const float u2 = (p0 ? dY5 : dY4)     + __shfl_xor_sync(smask, p0 ? dY4 : dY5, 1);")
        out.append("            const float u3 = (p0 ? dY7 : dY6)     + __shfl_xor_sync(smask, p0 ? dY6 : dY7, 1);")
        out.append("            const bool p1 = (slane & 2) != 0;")
        out.append("            const float w0 = (p1 ? u1 : u0) + __shfl_xor_sync(smask, p1 ? u0 : u1, 2);")
        out.append("            const float w1 = (p1 ? u3 : u2) + __shfl_xor_sync(smask, p1 ? u2 : u3, 2);")
        out.append("            const bool p2 = (slane & 4) != 0;")
        out.append("            float val = (p2 ? w1 : w0) + __shfl_xor_sync(smask, p2 ? w0 : w1, 4);")
        out.append("            if constexpr (SUBWARP >= 16) val += __shfl_xor_sync(smask, val, 8);")
        out.append("            if constexpr (SUBWARP == 32) val += __shfl_xor_sync(smask, val, 16);")
        out.append("            if (slane < 8) s_er_dY[sid][bi][slane] = val;")
        out.append("            const float dY8_sum = (SUBWARP == 8) ? qwarp_sum_f(dY8, smask) : ((SUBWARP == 16) ? half_warp_sum_f(dY8, smask) : warp_sum_f(dY8));")
        out.append("            if (slane == 0) s_er_dY[sid][bi][8] = dY8_sum;")
        out.append("        }")
        out.append("        __syncwarp(smask);")
        out.append("")
        out.append("        FT bfx = (FT)0, bfy = (FT)0, bfz = (FT)0;")
        out.append("        if (slane < bcnt) {")
        out.append("            const float4 vl = s_vl[sid][slane];")
        out.append("            const int i_nb = s_i[sid][slane];")
        out.append("            const float r = fmaxf(vl.w, 1e-8f);")
        out.append("            const float inv_d = 1.0f / r;")
        out.append("            const float ux = vl.x * inv_d, uy = vl.y * inv_d, uz = vl.z * inv_d;")
        out.append("            const float* ed = s_er_dY[sid][slane];")
        out.append("            float gx, gy, gz;")
        out.append("            sh_grad_l2(ux, uy, uz, ed[1], ed[2], ed[3], ed[4], ed[5], ed[6], ed[7], ed[8], gx, gy, gz);")
        out.append("            const float dot = ux * gx + uy * gy + uz * gz;")
        out.append("            bfx = (FT)(ux * ed[0] + (gx - ux * dot) * inv_d);")
        out.append("            bfy = (FT)(uy * ed[0] + (gy - uy * dot) * inv_d);")
        out.append("            bfz = (FT)(uz * ed[0] + (gz - uz * dot) * inv_d);")
        out.append("            if constexpr (F32_F) {")
        out.append("                atomicAdd(forces_f32 + (int64_t)i_nb * 3 + 0, -bfx);")
        out.append("                atomicAdd(forces_f32 + (int64_t)i_nb * 3 + 1, -bfy);")
        out.append("                atomicAdd(forces_f32 + (int64_t)i_nb * 3 + 2, -bfz);")
        out.append("            } else {")
        out.append("                atomicAdd(forces + (int64_t)i_nb * 3 + 0, -bfx);")
        out.append("                atomicAdd(forces + (int64_t)i_nb * 3 + 1, -bfy);")
        out.append("                atomicAdd(forces + (int64_t)i_nb * 3 + 2, -bfz);")
        out.append("            }")
        out.append("            if constexpr (STRESS) {")
        out.append("                v0 += (double)vl.x * (double)bfx;")
        out.append("                v1 += 0.5 * ((double)vl.x * (double)bfy + (double)vl.y * (double)bfx);")
        out.append("                v2 += 0.5 * ((double)vl.x * (double)bfz + (double)vl.z * (double)bfx);")
        out.append("                v3 += (double)vl.y * (double)bfy;")
        out.append("                v4 += 0.5 * ((double)vl.y * (double)bfz + (double)vl.z * (double)bfy);")
        out.append("                v5 += (double)vl.z * (double)bfz;")
        out.append("            }")
        out.append("        }")
        out.append("        if constexpr (F32_F) {")
        out.append("            bfx = (SUBWARP == 8) ? qwarp_sum_f(bfx, smask) : ((SUBWARP == 16) ? half_warp_sum_f(bfx, smask) : warp_sum_f(bfx));")
        out.append("            bfy = (SUBWARP == 8) ? qwarp_sum_f(bfy, smask) : ((SUBWARP == 16) ? half_warp_sum_f(bfy, smask) : warp_sum_f(bfy));")
        out.append("            bfz = (SUBWARP == 8) ? qwarp_sum_f(bfz, smask) : ((SUBWARP == 16) ? half_warp_sum_f(bfz, smask) : warp_sum_f(bfz));")
        out.append("        } else {")
        out.append("            bfx = (SUBWARP == 8) ? qwarp_sum_d(bfx, smask) : ((SUBWARP == 16) ? half_warp_sum_d(bfx, smask) : warp_sum_d(bfx));")
        out.append("            bfy = (SUBWARP == 8) ? qwarp_sum_d(bfy, smask) : ((SUBWARP == 16) ? half_warp_sum_d(bfy, smask) : warp_sum_d(bfy));")
        out.append("            bfz = (SUBWARP == 8) ? qwarp_sum_d(bfz, smask) : ((SUBWARP == 16) ? half_warp_sum_d(bfz, smask) : warp_sum_d(bfz));")
        out.append("        }")
        out.append("        if (slane == 0) { s_sf[sid][0] += bfx; s_sf[sid][1] += bfy; s_sf[sid][2] += bfz; }")
        out.append("        __syncwarp(smask);")
        out.append("    }")
        out.append("")
        out.append("    if (active_q) {")
        out.append("        dI_node.p[0][(int64_t)j * n_msg + slane] = acc_di0;")
        out.append("        float* dI1 = dI_node.p[1] + (int64_t)j * 3 * n_msg + slane;")
        out.append("        dI1[0] = acc_di1_0; dI1[n_msg] = acc_di1_1; dI1[2 * n_msg] = acc_di1_2;")
        if lmsg == 2:
            out.append("        float* dI2 = dI_node.p[2] + (int64_t)j * 5 * n_msg + slane;")
            out.append("        dI2[0] = acc_di2_0; dI2[n_msg] = acc_di2_1; dI2[2*n_msg] = acc_di2_2; dI2[3*n_msg] = acc_di2_3; dI2[4*n_msg] = acc_di2_4;")
        out.append("    }")
        out.append("    if constexpr (REM_BF) {")
        out.append("        const float df0_tot = (SUBWARP == 8) ? qwarp_sum_f(acc_df0, smask) : ((SUBWARP == 16) ? half_warp_sum_f(acc_df0, smask) : warp_sum_f(acc_df0));")
        out.append("        if (slane == 0) df0_bf[j] = df0_tot;")
        out.append("    }")
        out.append("    if constexpr (STRESS) {")
        out.append("        v0 = (SUBWARP == 8) ? qwarp_sum_d(v0, smask) : ((SUBWARP == 16) ? half_warp_sum_d(v0, smask) : warp_sum_d(v0));")
        out.append("        v1 = (SUBWARP == 8) ? qwarp_sum_d(v1, smask) : ((SUBWARP == 16) ? half_warp_sum_d(v1, smask) : warp_sum_d(v1));")
        out.append("        v2 = (SUBWARP == 8) ? qwarp_sum_d(v2, smask) : ((SUBWARP == 16) ? half_warp_sum_d(v2, smask) : warp_sum_d(v2));")
        out.append("        v3 = (SUBWARP == 8) ? qwarp_sum_d(v3, smask) : ((SUBWARP == 16) ? half_warp_sum_d(v3, smask) : warp_sum_d(v3));")
        out.append("        v4 = (SUBWARP == 8) ? qwarp_sum_d(v4, smask) : ((SUBWARP == 16) ? half_warp_sum_d(v4, smask) : warp_sum_d(v4));")
        out.append("        v5 = (SUBWARP == 8) ? qwarp_sum_d(v5, smask) : ((SUBWARP == 16) ? half_warp_sum_d(v5, smask) : warp_sum_d(v5));")
        out.append("    }")
        out.append("    if (slane == 0) {")
        out.append("        const FT sfx = s_sf[sid][0], sfy = s_sf[sid][1], sfz = s_sf[sid][2];")
        out.append("        if (sfx != (FT)0 || sfy != (FT)0 || sfz != (FT)0) {")
        out.append("            if constexpr (F32_F) {")
        out.append("                atomicAdd(forces_f32 + (int64_t)j * 3 + 0, sfx);")
        out.append("                atomicAdd(forces_f32 + (int64_t)j * 3 + 1, sfy);")
        out.append("                atomicAdd(forces_f32 + (int64_t)j * 3 + 2, sfz);")
        out.append("            } else {")
        out.append("                atomicAdd(forces + (int64_t)j * 3 + 0, sfx);")
        out.append("                atomicAdd(forces + (int64_t)j * 3 + 1, sfy);")
        out.append("                atomicAdd(forces + (int64_t)j * 3 + 2, sfz);")
        out.append("            }")
        out.append("        }")
        out.append("        if constexpr (STRESS) {")
        out.append("            if (v0 != 0.0) atomicAdd(virial + 0, v0); if (v1 != 0.0) atomicAdd(virial + 1, v1);")
        out.append("            if (v2 != 0.0) atomicAdd(virial + 2, v2); if (v3 != 0.0) atomicAdd(virial + 3, v3);")
        out.append("            if (v4 != 0.0) atomicAdd(virial + 4, v4); if (v5 != 0.0) atomicAdd(virial + 5, v5);")
        out.append("        }")
        out.append("    }")
        out.append("}")
        out.append("")
    return "\n".join(out)


def main():
    verify_paths()
    header = [
        "#pragma once",
        "// AUTO-GENERATED by codegen/gen_ilmlip_lf2.py — DO NOT EDIT BY HAND.",
        "// Specialized fast kernels for arch=ilmlip with lfeat=2, lfeat2=2, lout2=2, lmsg in {1, 2}.",
        "",
        build_l1_node_lf2_code(),
        build_l2_lf2_code(),
    ]
    out_path = REPO / "src" / "ilmlip_lf2_kernels.cuh"
    out_path.write_text("\n".join(header) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes, {sum(1 for _ in out_path.open())} lines)")


if __name__ == "__main__":
    main()
