#!/usr/bin/env python3
"""Generate LoupMD/src/generated_kernels.cuh from mace-omat-0-medium.model (CPU-only).

Generates self-contained CUDA C++ kernels for:
  1. sc0_forward / sc0_backward: prod[0] factored symmetric contraction + 1/(D0+1) + readout[0]
  2. sc1_fused: prod[1] factored symmetric contraction + 1/(D1+1) over the 4 contiguous BlockLinear segment buffers
     (b0, b1, b2, b3), readout[1] (with skip) forward and backward, and the prod[1] backward, one block per atom.
  3. spline_forward_inter1: Interaction 1 PACKED8 + STAGE_EDGES + R128 + fused density_1 spline.
  4. wprime_backward_inter1: Interaction 1 LATE_PULLBACK + STAGE_DOUT + STAGE_EDGES +
     fused dD1*ddens/dr + direct analytical edge_force_1 [E, 3] (no dsh_1 [E, 16]).
"""
from __future__ import annotations

import collections
import math
import os
import re
import struct
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

FASTMACE_ROOT = Path("/mnt/d/Margraf/FastMACE")
sys.path.insert(0, str(FASTMACE_ROOT))

import numpy as np
import torch
from mace.cli.convert_e3nn_cueq import run as convert_e3nn_cueq

from fastmace.kernels.cuda.sc_factored import _expr, _literal, factor
from fastmace.kernels.cuda.sc_generated import merged_terms, shape_of
from fastmace.kernels.oeq_streaming import wprime_pullback as wp
from fastmace.kernels.oeq_streaming.fused_forward import _lit, _node, block_map
from fastmace.kernels.oeq_streaming.runtime import NUM_THREADS, render_kernels
from fastmace.kernels.oeq_streaming.wprime_backward import _stage_layout, factored_paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the repository root: the loupmd package
from loupmd.export.omat0 import check_architecture, kernel_signature, product_signature  # noqa: E402


def _load_model(model_path: str):
    m = torch.load(model_path, map_location="cpu", weights_only=False).float()
    return convert_e3nn_cueq(m)


def _emit_ssa_node(lines, i, op):
    if op[0] == "mul":
        lines.append(f"  const float v{i} = {_expr(op[1])} * {_expr(op[2])};")
        return
    if op[0] == "lin":
        (c0, u0), rest = op[1][0], op[1][1:]
        expr = f"{_literal(c0)} * {_expr(u0)}"
        for c, u in rest:
            expr = f"fmaf({_literal(c)}, {_expr(u)}, {expr})"
    else:
        (a0, b0), rest = op[1][0], op[1][1:]
        expr = f"{_expr(a0)} * {_expr(b0)}"
        for a, b in rest:
            expr = f"fmaf({_expr(a)}, {_expr(b)}, {expr})"
    lines.append(f"  const float v{i} = {expr};")


def _emit_sc0(prog, features, segments_w, mul=128):
    # Input: m0_0 [N, 128], m0_1 [N, 3*128], m0_2 [N, 5*128], m0_3 [N, 7*128]
    # Output: h0_0 [N, 128], h0_1 [N, 3*128], out_energy [double*] (atomicAdd)
    roots = [u for f in features for u in f.values() if u is not None]
    used = prog.used(roots)
    warps = mul // 32

    def load_x(lines):
        lines.append("  const float s = 1.0f / (d0[a] + 1.0f);")
        lines.append(f"  const float x0 = __ldg(m0_0 + (int64_t)a * {mul} + ch) * s;")
        for m in range(3):
            lines.append(f"  const float x{1 + m} = __ldg(m0_1 + ((int64_t)a * 3 + {m}) * {mul} + ch) * s;")
        for m in range(5):
            lines.append(f"  const float x{4 + m} = __ldg(m0_2 + ((int64_t)a * 5 + {m}) * {mul} + ch) * s;")
        for m in range(7):
            lines.append(f"  const float x{9 + m} = __ldg(m0_3 + ((int64_t)a * 7 + {m}) * {mul} + ch) * s;")

    # Forward
    fwd = [
        f'extern "C" __global__ void __launch_bounds__({mul}) sc0_forward(',
        "    int N, const float* __restrict__ m0_0, const float* __restrict__ m0_1,",
        "    const float* __restrict__ m0_2, const float* __restrict__ m0_3,",
        "    const float* __restrict__ d0, const float* __restrict__ w0_ptr,",
        "    const int* __restrict__ sp, const float* __restrict__ w_ro0,",
        "    float* __restrict__ h0_0, float* __restrict__ h0_1, double* __restrict__ out_energy) {",
        "  const int a = blockIdx.x, ch = threadIdx.x;",
        "  if (a >= N) return;",
        f"  const float* wr = w0_ptr + (int64_t)sp[a] * {segments_w * mul} + ch;",
    ]
    load_x(fwd)
    for i in used:
        _emit_ssa_node(fwd, i, prog.nodes[i])
    fwd.append("  float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;")
    for b, f in enumerate(features):
        items = [(c, u) for c, u in sorted(f.items()) if u is not None]
        if items:
            fwd.append(f"  {{ const float wb = __ldg(wr + {b * mul});")
            for c, u in items:
                fwd.append(f"    y{c} = fmaf(wb, {_expr(u)}, y{c});")
            fwd.append("  }")
    fwd.append(f"  h0_0[(int64_t)a * {mul} + ch] = y0;")
    for m in range(3):
        fwd.append(f"  h0_1[((int64_t)a * 3 + {m}) * {mul} + ch] = y{1 + m};")
    fwd += [
        "  double ed = (double)y0 * (double)__ldg(w_ro0 + ch);",
        "  #pragma unroll",
        "  for (int off = 16; off > 0; off >>= 1) ed += __shfl_xor_sync(0xffffffffu, ed, off);",
        f"  __shared__ double ered[{warps}];",
        "  if ((ch & 31) == 0) ered[ch >> 5] = ed;",
        "  __syncthreads();",
        "  if (ch == 0) {",
        "    double sum = 0.0;",
        f"    for (int k = 0; k < {warps}; ++k) sum += ered[k];",
        "    atomicAdd(out_energy, sum);",
        "  }",
        "}",
    ]

    # Backward
    bwd = [
        f'extern "C" __global__ void __launch_bounds__({mul}, 4) sc0_backward(',
        "    int N, const float* __restrict__ m0_0, const float* __restrict__ m0_1,",
        "    const float* __restrict__ m0_2, const float* __restrict__ m0_3,",
        "    const float* __restrict__ d0, const float* __restrict__ w0_ptr,",
        "    const int* __restrict__ sp, const float* __restrict__ w_ro0,",
        "    const float* __restrict__ dh0_0, const float* __restrict__ dh0_1,",
        "    float* __restrict__ dm0_0, float* __restrict__ dm0_1,",
        "    float* __restrict__ dm0_2, float* __restrict__ dm0_3, float* __restrict__ dd0) {",
        "  const int a = blockIdx.x, ch = threadIdx.x;",
        "  if (a >= N) return;",
        f"  const float* wr = w0_ptr + (int64_t)sp[a] * {segments_w * mul} + ch;",
    ]
    load_x(bwd)
    bwd.append(f"  const float g0 = __ldg(dh0_0 + (int64_t)a * {mul} + ch) + __ldg(w_ro0 + ch);")
    for m in range(3):
        bwd.append(f"  const float g{1 + m} = __ldg(dh0_1 + ((int64_t)a * 3 + {m}) * {mul} + ch);")
    for i in used:
        _emit_ssa_node(bwd, i, prog.nodes[i])
    adj = {}

    def add(u, expr_mul):
        name = f"d{_expr(u)}"
        if isinstance(expr_mul, tuple):
            a_, b_ = expr_mul
            if u in adj:
                bwd.append(f"  {name} = fmaf({a_}, {b_}, {name});")
            else:
                bwd.append(f"  float {name} = {a_} * {b_};")
        else:
            if u in adj:
                bwd.append(f"  {name} += {expr_mul};")
            else:
                bwd.append(f"  float {name} = {expr_mul};")
        adj[u] = name

    for b, f in enumerate(features):
        items = [(c, u) for c, u in sorted(f.items()) if u is not None]
        if not items:
            continue
        bwd.append(f"  const float w{b} = __ldg(wr + {b * mul});")
        for c, u in items:
            add(u, (f"w{b}", f"g{c}"))
    for i in reversed(used):
        if i not in adj:
            continue
        op, da = prog.nodes[i], adj[i]
        if op[0] == "mul":
            add(op[1], (da, _expr(op[2])))
            add(op[2], (da, _expr(op[1])))
        elif op[0] == "lin":
            for c, u in op[1]:
                add(u, (_literal(c), da))
        else:
            for a_, b_ in op[1]:
                add(a_, (da, _expr(b_)))
                add(b_, (da, _expr(a_)))
    bwd.append("  float dot = 0.0f;")
    for i in range(16):
        dx = adj.get(-1 - i, "0.0f")
        bwd.append(f"  dot = fmaf({dx}, x{i}, dot);")
        if i == 0:
            bwd.append(f"  dm0_0[(int64_t)a * {mul} + ch] = {dx} * s;")
        elif i < 4:
            bwd.append(f"  dm0_1[((int64_t)a * 3 + {i - 1}) * {mul} + ch] = {dx} * s;")
        elif i < 9:
            bwd.append(f"  dm0_2[((int64_t)a * 5 + {i - 4}) * {mul} + ch] = {dx} * s;")
        else:
            bwd.append(f"  dm0_3[((int64_t)a * 7 + {i - 9}) * {mul} + ch] = {dx} * s;")
    bwd += [
        "  #pragma unroll",
        "  for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);",
        f"  __shared__ float red[{warps}];",
        "  if ((ch & 31) == 0) red[ch >> 5] = dot;",
        "  __syncthreads();",
        "  if (ch == 0) {",
        "    float total = 0.0f;",
        f"    for (int k = 0; k < {warps}; ++k) total += red[k];",
        "    dd0[a] = -s * total;",
        "  }",
        "}",
    ]
    return "\n".join(fwd) + "\n\n" + "\n".join(bwd) + "\n"


def _emit_sc1(prog, features, segments_w, mul=128):
    # One block per atom, one thread per channel: product 1 forward, readout 1 (Linear 128->16 + skip, SiLU, Linear
    # 16->1, energy), readout backward and product 1 backward in one kernel. The forward monomials stay in registers
    # for the backward, and y1 / hidden16 never round-trip through memory except for the outputs below.
    # Input: b0 [N, 128], b1 [N, 3*128], b2 [N, 5*128], b3 [N, 7*128], d1 [N], sc_skip16 [N, 16]
    # Output: y1 [N, 128] (debug readback), d_hidden16 [N, 16] (skip backward), energy, db0..db3, dd1
    assert mul == 128
    roots = [u for f in features for u in f.values() if u is not None]
    used = prog.used(roots)
    warps = mul // 32

    lines = [
        f'extern "C" __global__ void __launch_bounds__({mul}, 4) sc1_fused(',
        "    int N, const float* __restrict__ b0, const float* __restrict__ b1,",
        "    const float* __restrict__ b2, const float* __restrict__ b3,",
        "    const float* __restrict__ d1, const float* __restrict__ w1_ptr, const int* __restrict__ sp,",
        "    const float* __restrict__ sc_skip16, const float* __restrict__ w_ro_a, const float* __restrict__ w_ro_b,",
        "    float* __restrict__ y1, float* __restrict__ d_hidden16, double* __restrict__ out_energy,",
        "    float* __restrict__ db0, float* __restrict__ db1,",
        "    float* __restrict__ db2, float* __restrict__ db3, float* __restrict__ dd1) {",
        "  const int a = blockIdx.x, ch = threadIdx.x;",
        "  if (a >= N) return;",
        f"  const float* wr = w1_ptr + (int64_t)sp[a] * {segments_w * mul} + ch;",
        "  const float s = 1.0f / (d1[a] + 1.0f);",
        f"  const float x0 = __ldg(b0 + (int64_t)a * {mul} + ch) * s;",
    ]
    for m in range(3):
        lines.append(f"  const float x{1 + m} = __ldg(b1 + ((int64_t)a * 3 + {m}) * {mul} + ch) * s;")
    for m in range(5):
        lines.append(f"  const float x{4 + m} = __ldg(b2 + ((int64_t)a * 5 + {m}) * {mul} + ch) * s;")
    for m in range(7):
        lines.append(f"  const float x{9 + m} = __ldg(b3 + ((int64_t)a * 7 + {m}) * {mul} + ch) * s;")
    for i in used:
        _emit_ssa_node(lines, i, prog.nodes[i])

    # Product 1 forward: y0 = sum_b w_b * feature_b
    lines.append("  float y0 = 0.0f;")
    for b, f in enumerate(features):
        items = [(c, u) for c, u in sorted(f.items()) if u is not None]
        if items:
            lines.append(f"  const float w{b} = __ldg(wr + {b * mul});")
            for c, u in items:
                lines.append(f"  y{c} = fmaf(w{b}, {_expr(u)}, y{c});")

    # Readout 1 forward + backward: hidden = y1 @ w_ro_a [128, 16] + skip; E += silu(hidden) . w_ro_b;
    # g0 = dE/dy1 = (silu'(hidden) * w_ro_b) @ w_ro_a^T
    lines += [
        f"  y1[(int64_t)a * {mul} + ch] = y0;",
        f"  __shared__ float s_y[{mul}];",
        f"  __shared__ float s_part[{mul // 16}][16];",
        "  __shared__ float s_dh[16];",
        "  s_y[ch] = y0;",
        "  __syncthreads();",
        "  {",
        "    const int k = ch & 15, c0 = (ch >> 4) * 16;",
        "    float p = 0.0f;",
        "    #pragma unroll",
        "    for (int c = 0; c < 16; ++c) p = fmaf(s_y[c0 + c], __ldg(w_ro_a + (c0 + c) * 16 + k), p);",
        "    s_part[ch >> 4][k] = p;",
        "  }",
        "  __syncthreads();",
        "  if (ch < 16) {",
        "    float hsum = __ldg(sc_skip16 + (int64_t)a * 16 + ch);",
        "    #pragma unroll",
        f"    for (int j = 0; j < {mul // 16}; ++j) hsum += s_part[j][ch];",
        "    const float sig = 1.0f / (1.0f + expf(-hsum));",
        "    const float w2 = __ldg(w_ro_b + ch);",
        "    double e = (double)(hsum * sig) * (double)w2;",
        "    const float dh = sig * (1.0f + hsum * (1.0f - sig)) * w2;",
        "    s_dh[ch] = dh;",
        "    d_hidden16[(int64_t)a * 16 + ch] = dh;",
        "    #pragma unroll",
        "    for (int off = 8; off > 0; off >>= 1) e += __shfl_xor_sync(0x0000ffffu, e, off);",
        "    if (ch == 0) atomicAdd(out_energy, e);",
        "  }",
        "  __syncthreads();",
        "  float g0 = 0.0f;",
        "  #pragma unroll",
        "  for (int k = 0; k < 16; ++k) g0 = fmaf(s_dh[k], __ldg(w_ro_a + ch * 16 + k), g0);",
    ]

    # Product 1 backward
    adj = {}

    def add(u, expr_mul):
        name = f"d{_expr(u)}"
        if isinstance(expr_mul, tuple):
            a_, b_ = expr_mul
            if u in adj:
                lines.append(f"  {name} = fmaf({a_}, {b_}, {name});")
            else:
                lines.append(f"  float {name} = {a_} * {b_};")
        else:
            if u in adj:
                lines.append(f"  {name} += {expr_mul};")
            else:
                lines.append(f"  float {name} = {expr_mul};")
        adj[u] = name

    for b, f in enumerate(features):
        items = [(c, u) for c, u in sorted(f.items()) if u is not None]
        for c, u in items:
            add(u, (f"w{b}", f"g{c}"))
    for i in reversed(used):
        if i not in adj:
            continue
        op, da = prog.nodes[i], adj[i]
        if op[0] == "mul":
            add(op[1], (da, _expr(op[2])))
            add(op[2], (da, _expr(op[1])))
        elif op[0] == "lin":
            for c, u in op[1]:
                add(u, (_literal(c), da))
        else:
            for a_, b_ in op[1]:
                add(a_, (da, _expr(b_)))
                add(b_, (da, _expr(a_)))
    lines.append("  float dot = 0.0f;")
    for i in range(16):
        dx = adj.get(-1 - i, "0.0f")
        lines.append(f"  dot = fmaf({dx}, x{i}, dot);")
        if i == 0:
            lines.append(f"  db0[(int64_t)a * {mul} + ch] = {dx} * s;")
        elif i < 4:
            lines.append(f"  db1[((int64_t)a * 3 + {i - 1}) * {mul} + ch] = {dx} * s;")
        elif i < 9:
            lines.append(f"  db2[((int64_t)a * 5 + {i - 4}) * {mul} + ch] = {dx} * s;")
        else:
            lines.append(f"  db3[((int64_t)a * 7 + {i - 9}) * {mul} + ch] = {dx} * s;")
    lines += [
        "  #pragma unroll",
        "  for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);",
        f"  __shared__ float red[{warps}];",
        "  if ((ch & 31) == 0) red[ch >> 5] = dot;",
        "  __syncthreads();",
        "  if (ch == 0) {",
        "    float total = 0.0f;",
        f"    for (int k = 0; k < {warps}; ++k) total += red[k];",
        "    dd1[a] = -s * total;",
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _emit_spline_forward1(schedule, blocks, C=128):
    # Generates spline_forward_inter1 with PACKED8 + STAGE_EDGES + runtime SplineGrid + fused density_1
    bmap = block_map(blocks, C)
    slots = []
    acc_slot = {}
    for p in schedule.paths:
        for o in p.outputs:
            name = o.accumulator.name
            if name not in acc_slot:
                acc_slot[name] = len(slots)
                slots.append(o.accumulator)
    n_acc = len(slots)
    q_dim = len(schedule.paths) * C

    def _f32(x):
        return struct.unpack("f", struct.pack("f", x))[0]

    # Group (k_acc, ic) terms per path so each per-edge CG linear form of the SH is computed once in shared memory.
    cg_terms = collections.OrderedDict()
    path_pairs = {}
    for p in schedule.paths:
        p_map = collections.OrderedDict()
        for o in p.outputs:
            k_acc = acc_slot[o.accumulator.name]
            for t in o.terms:
                p_map.setdefault((k_acc, t.input_component), []).append(
                    (_f32(float(t.coefficient)), p.edge_start + t.edge_component)
                )
        for key, tlist in p_map.items():
            assert key not in cg_terms, f"duplicate (acc, ic) across paths: {key}"
            cg_terms[key] = tlist
        path_pairs[p.weight_start] = list(p_map.keys())

    acc_scale, cg_use, extras = {}, {}, collections.OrderedDict()
    for k_acc in range(n_acc):
        singles = [abs(t[0][0]) for (i, _), t in cg_terms.items() if i == k_acc and len(t) == 1]
        if singles:
            cnt = collections.Counter(round(s, 6) for s in singles)
            best = max(cnt.items(), key=lambda kv: kv[1])[0]
            sc = next(s for s in singles if round(s, 6) == best)
        else:
            sc = abs(next(t for (i, _), t in cg_terms.items() if i == k_acc)[0][0])
        acc_scale[k_acc] = sc
        for (i, ic), tlist in cg_terms.items():
            if i != k_acc:
                continue
            norm = [(c / sc, ec) for c, ec in tlist]
            if len(norm) == 1 and abs(abs(norm[0][0]) - 1.0) < 1e-6:
                cg_use[(k_acc, ic)] = ("-" if norm[0][0] < 0 else "", norm[0][1])
                continue
            sign = -1.0 if norm[0][0] < 0 else 1.0
            key = tuple(sorted((ec, _lit(_f32(sign * c))) for c, ec in norm))
            if key not in extras:
                extras[key] = 16 + len(extras)
            cg_use[(k_acc, ic)] = ("-" if sign < 0 else "", extras[key])

    ED = 4 * math.ceil((16 + len(extras)) / 4)
    extra_lines = []
    if extras:
        extra_lines.append("            { float* p = s_sh + (ch - 32) * ED;  // per-edge CG linear forms shared by all channels")
        for key, eslot in extras.items():
            expr = " + ".join(f"{c_lit} * p[{ec}]" for ec, c_lit in key)
            extra_lines.append(f"              p[{eslot}] = {expr};")
        extra_lines.append("            }")
    extra_block = ("\n" + "\n".join(extra_lines)) if extra_lines else ""

    # Input map: xi[(p.input_start, comp, dim)] -> expression loading from x0 or x1
    # In OMAT-0 medium inter[1], input irreps are 128x0e (start=0, dim=1) and 128x1o (start=128, dim=3)
    body = []
    body_pair = []  # two edges sharing one slab: edge A (snd, pw, she) and edge B (sndB, pwB, sheB)
    for p in schedule.paths:
        wstart = p.weight_start
        decode = [
            f"                const uint2 c = __ldg(tq + {wstart});",
            "                const unsigned eb = (c.y & 0x0F800000u) + bias_shift;",
            "                const float eb_f = __uint_as_float(eb);",
            "                const float a0 = __uint_as_float((c.x & 0x7FFFF8u) | eb);",
            "                const float a1 = __uint_as_float((c.y & 0x7FFFF0u) | eb);",
            "                const float a2 = __uint_as_float((__funnelshift_l(c.x, c.x, 20) & 0x7FF800u) | eb);",
            "                const float a3 = __uint_as_float((__funnelshift_l(c.y, c.y, 19) & 0x7F8000u) | eb);",
        ]
        body.append(f"            {{  // path weight_start={wstart}")
        body += decode
        body.append("                const float w = fmaf(pw.z, a3, fmaf(pw.y, a2, fmaf(pw.x, a1, fmaf(pw.w, eb_f, a0))));")
        body_pair.append(f"            {{  // path weight_start={wstart} (pair)")
        body_pair += decode
        body_pair.append("                const float w = fmaf(pw.z, a3, fmaf(pw.y, a2, fmaf(pw.x, a1, fmaf(pw.w, eb_f, a0))));")
        body_pair.append("                const float wB = fmaf(pwB.z, a3, fmaf(pwB.y, a2, fmaf(pwB.x, a1, fmaf(pwB.w, eb_f, a0))));")
        x_needed = sorted({t.input_component for o in p.outputs for t in o.terms})
        for ic in x_needed:
            for out, sfx, sv in ((body, "", "snd"), (body_pair, "", "snd"), (body_pair, "B", "sndB")):
                if p.input_start == 0:
                    out.append(f"                const float xi{sfx}_{ic} = __ldg(x0 + (int64_t){sv} * {C} + ch);")
                else:
                    out.append(f"                const float xi{sfx}_{ic} = __ldg(x1 + ((int64_t){sv} * 3 + {ic}) * {C} + ch);")
        for k_acc, ic in path_pairs[wstart]:
            sg, eslot = cg_use[(k_acc, ic)]
            body.append(f"                acc[{k_acc}] = fmaf(w * xi_{ic}, {sg}she[{eslot}], acc[{k_acc}]);")
            body_pair.append(f"                acc[{k_acc}] = fmaf(w * xi_{ic}, {sg}she[{eslot}], acc[{k_acc}]);")
            body_pair.append(f"                acc[{k_acc}] = fmaf(wB * xiB_{ic}, {sg}sheB[{eslot}], acc[{k_acc}]);")
        body.append("            }")
        body_pair.append("            }")

    stores = []
    for k_acc, s_acc in enumerate(slots):
        g, width, base, stride = bmap[s_acc.start]
        off = base + s_acc.component * stride
        sc = acc_scale[k_acc]
        mult = "" if abs(sc - 1.0) < 1e-7 else f" * {_lit(sc)}"
        stores.append(f"    m1_{g}[(int64_t)rcv * {width} + {off} + ch] = acc[{k_acc}]{mult};")

    src = f"""
// R receivers per block (128 threads each, synchronised per receiver by named barrier 1 + index). R > 1 lets the
// receivers' warps share table slabs in L1 (consecutive receivers of a symmetric crystal walk the same slabs) at the
// cost of block-granular register release; the engine times the variants per system (tune_rebuild_variants).
// PAIR: decode a slab once for two consecutive edges that read it (wins when many do; costs registers otherwise).
// The per-receiver named barriers (bar.sync 1 + index, 128) are correct only while (a) each receiver is exactly four
// whole warps, (b) all 128 threads of a receiver run the same number of tile iterations, and (c) the kernel has no
// block-wide barrier (__syncthreads / bar.sync 0): a receiver group that returns early (rcv >= N) must never be
// waited for by another group. Adding a __syncthreads breaks (c) without any compile error. Barrier ids 1..15
// limit R to 15.
template <int R, bool PAIR>
__global__ void __launch_bounds__(128 * R) spline_forward_inter1(
    int N, const int* __restrict__ E_ptr,
    const float* __restrict__ x0, const float* __restrict__ x1,
    const float4* __restrict__ vec_len, const int* __restrict__ senders,
    const int* __restrict__ row_ptr,
    const int* __restrict__ act_slab, const float2* __restrict__ act_t_sx,
    const uint2* __restrict__ packed_table,
    const float4* __restrict__ density_table,
    unsigned bias_shift,
    float* __restrict__ m1_0, float* __restrict__ m1_1,
    float* __restrict__ m1_2, float* __restrict__ m1_3,
    float* __restrict__ out_density1) {{
    constexpr int TILE = 16;
    constexpr int ED = {ED};
    static_assert(R >= 1 && R <= 15, "per-receiver named barriers use ids 1..15");
    constexpr int64_t Q = {q_dim};
    __shared__ int s_snd_all[R][TILE];
    __shared__ __align__(16) float4 s_pw_all[R][TILE];
    __shared__ int64_t s_tq_all[R][TILE];
    __shared__ __align__(16) float s_sh_all[R][TILE * ED];
    const int half = R == 1 ? 0 : threadIdx.x >> 7, ch = R == 1 ? threadIdx.x : threadIdx.x & 127;
    const int rcv = blockIdx.x * R + half;
    int* s_snd = s_snd_all[half];
    float4* s_pw = s_pw_all[half];
    int64_t* s_tq = s_tq_all[half];
    float* s_sh = s_sh_all[half];
    float acc[{n_acc}];
    #pragma unroll
    for (int i = 0; i < {n_acc}; ++i) acc[i] = 0.0f;
    float d1_acc = 0.0f;
    const int E = *E_ptr;
    if (rcv >= N) return;
    int64_t eb = row_ptr[rcv], ee = row_ptr[rcv + 1];
    eb = eb < 0 ? 0 : eb;
    ee = ee > E ? E : ee;
    for (int64_t base = eb; base < ee; base += TILE) {{
        const int count_in = (int)(ee - base < TILE ? ee - base : TILE);
        if (ch >= 32 && ch - 32 < count_in) {{
            // Warp 1 computes the staged edges' SH and per-edge CG linear forms while warp 0 gathers metadata.
            const float4 vl = __ldg(vec_len + base + (ch - 32));
            loup::edge_sh16(vl.x, vl.y, vl.z, vl.w, reinterpret_cast<float4*>(s_sh + (ch - 32) * ED));{extra_block}
        }}
        if (ch < 32) {{
            float d_edge = 0.0f;
            if (ch < count_in) {{
                const int64_t e = base + ch;
                const int64_t slab = (int64_t)act_slab[e];
                const float t_k = act_t_sx[e].x;
                const float sk = t_k - 0.5f, sk2 = sk * sk;
                const float q1 = sk * 0x1.0p-8f, q2 = sk2 * 0x1.0p-14f, q3 = sk * sk2 * 0x1.0p-19f;
                s_snd[ch] = senders[e];
                s_pw[ch] = make_float4(q1, q2, q3, -1.5f * (1.0f + q1 + q2 + q3));
                s_tq[ch] = slab * Q;
                const float4 dc = __ldg(density_table + slab);
                d_edge = fmaf(t_k, fmaf(t_k, fmaf(t_k, dc.w, dc.z), dc.y), dc.x);
            }}
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) d_edge += __shfl_xor_sync(0xffffffffu, d_edge, off);
            if (ch == 0) d1_acc += d_edge;
        }}
        asm volatile("bar.sync %0, 128;" :: "r"(half + 1) : "memory");  // this receiver's 4 warps only
        for (int k = 0; k < count_in;) {{
            const int snd = s_snd[k];
            const float4 pw = s_pw[k];
            const float* she = s_sh + k * ED;
            const uint2* tq = packed_table + s_tq[k] + ch;
            if (PAIR && k + 1 < count_in && s_tq[k + 1] == s_tq[k]) {{
                // The next edge reads the same table slab (same pair and interval; a receiver's edges are sorted by
                // (pair, r)): decode each coefficient once for both.
                const int sndB = s_snd[k + 1];
                const float4 pwB = s_pw[k + 1];
                const float* sheB = she + ED;
{chr(10).join(body_pair)}
                k += 2;
            }} else {{
{chr(10).join(body)}
                k += 1;
            }}
        }}
        asm volatile("bar.sync %0, 128;" :: "r"(half + 1) : "memory");  // this receiver's 4 warps only
    }}
    if (ch == 0) out_density1[rcv] = d1_acc;
{chr(10).join(stores)}
}}
"""
    return src


# w' decodes each cubic coefficient as raw_k - hb, raw_k = 2^e (1 + mantissa field), hb = 1.5 * 2^e, and then evaluates
#   w = d0 + s1 d1 + s2 d2 + s3 d3,  radial_dt = d1 + ds2 d2 + ds3 d3.
# With DECODE_BIAS_FOLD the four subtractions per path fold into the polynomials:
#   w = raw0 + s1 raw1 + s2 raw2 + s3 raw3 - 1.5 (1 + s1 + s2 + s3) 2^e, and likewise radial_dt,
# with the two bias factors computed once per edge: 4 FADD + 1 LOP3 per path become 2 FFMA. The price is the rounding
# of the bias factor (up to about 2^-23 of 2^e, against a quantisation step of 2^-20 for d0); the gate decides.
# Adopted 13 November 2026 (OPTIMISATION_LEDGER.md, wp-decode-bias-fold): RTX 5060 Ti SrTiO3 0.9929 / 0.9948 and
# NbCl5 0.9951 / 0.9938 in two a-b-b-a runs, 24 instructions and the 8-byte stack spill fewer; RTX 4000 Ada a tie
# (1.0015 / 0.9955). The gate passes on both, with ratios inside its own run-to-run spread.
# LOUP_GEN_DECODE_FOLD=0 regenerates the unfolded form for an A/B.
DECODE_BIAS_FOLD = os.environ.get("LOUP_GEN_DECODE_FOLD", "1") != "0"
DECODE_FOLD_CONSTANTS = (
    "        const float nw_hb = -1.5f * (1.0f + s1 + s2 + s3), ndt_hb = -1.5f * (1.0f + ds2 + ds3);" + chr(10)
    if DECODE_BIAS_FOLD else "")


def _cubic_decode_lines():
    raw = [
        "__uint_as_float((c.x & 0x7FFFF8u) | eb0)",
        "__uint_as_float((c.y & 0x7FFFF0u) | eb0)",
        "__uint_as_float((__funnelshift_l(c.x, c.x, 20) & 0x7FF800u) | eb0)",
        "__uint_as_float((__funnelshift_l(c.y, c.y, 19) & 0x7F8000u) | eb0)",
    ]
    pad = "                    "
    if DECODE_BIAS_FOLD:
        return [pad + "const float eb_f = __uint_as_float(eb0);"] + [
            f"{pad}const float r{k} = {raw[k]};" for k in range(4)] + [
            pad + "const float w = fmaf(s3, r3, fmaf(s2, r2, fmaf(s1, r1, fmaf(nw_hb, eb_f, r0))));",
            pad + "const float radial_dt = fmaf(ds3, r3, fmaf(ds2, r2, fmaf(ndt_hb, eb_f, r1)));  // 2^8 x d(w)/dt",
        ]
    return [pad + "const float hb0 = __uint_as_float(eb0 | 0x400000u);"] + [
        f"{pad}const float d{k} = {raw[k]} - hb0;" for k in range(4)] + [
        pad + "const float w = fmaf(s3, d3, fmaf(s2, d2, fmaf(s1, d1, d0)));",
        pad + "const float radial_dt = fmaf(ds3, d3, fmaf(ds2, d2, d1));  // 2^8 x d(w)/dt",
    ]


def _emit_wprime_backward1(schedule, blocks, C=128):
    # Generates wprime_backward_inter1 with:
    #   LATE_PULLBACK (direct Cartesian forms, no dsh[E, 16])
    #   STAGE_DOUT (split dm1_0..dm1_3 inputs, q = M * g staged)
    #   STAGE_EDGES (32-edge tile in warp 0)
    #   fused dD1 * ddens/dr -> direct edge_force_1 [E, 3]
    plist = wp.paths(schedule, blocks, fold_alpha=False)
    _, offsets, stage_total = _stage_layout(schedule, blocks)
    prologue_body = []
    for p in plist:
        prologue_body.append(f"                {{ // path {p['index']} ({p['a']}, {p['l']}, {p['L']})")
        for o, (g, width, col) in enumerate(p["dout"]):
            prologue_body.append(f"                    const float g{o} = __ldcs(dm1_{g} + (int64_t)stage_r * {width} + {col} + ch);")
        for k, col in enumerate(p["stage"]):
            terms = [f"{wp._f(p['M'][k, o])} * g{o}" for o in range(2 * p["L"] + 1) if abs(float(p['M'][k, o])) > 1e-12]
            val = " + ".join(terms) or "0.0f"
            prologue_body.append(f"                    stage_row[{col} + ch] = {val};")
        prologue_body.append("                }")

    # wprime_backward_inter1_rcv stages q channel-major: s_q[channel * QS + slot] with the 40 slots (slot = col / C)
    # padded to QS = 44 floats (11 x 16 B, odd, so a warp's 128-bit loads at stride QS hit distinct banks per phase).
    # A lane then reads its channel's q as 10 LDS.128 instead of 40 scalar LDS.
    n_slots = stage_total // C
    QS = (n_slots + 3) // 4 * 4
    if (QS // 4) % 2 == 0:
        QS += 4
    prologue_rcv = [re.sub(r"stage_row\[(\d+) \+ ch\]", lambda m: f"s_q[ch * {QS} + {int(m.group(1)) // C}]", x)
                    for x in prologue_body]

    # Channel loop body per path
    # In schedule, inputs are 0e (istart=0) and 1o (istart=128)
    path_bodies = []
    path_bodies_rcv = []  # same, with q from the float4 chunks Q<c> of the lane's channel row
    chunk_done = set()
    for p in plist:
        a, l, L, kind = p["a"], p["l"], p["L"], p["form"]
        wstart = p["weight_start"]
        q = [f"stage_row[{c} + channel]" for c in p["stage"]]
        b = [
            f"                {{ // path {p['index']} ({a}, {l}, {L}) {kind}",
            f"                    const uint2 c = __ldg(slab_ptr + {wstart} + channel);",
            "                    const unsigned eb0 = (c.y & 0x0F800000u) + bias_shift;",
            *_cubic_decode_lines(),
        ]
        b += [f"                    const float q{k} = {q[k]};" for k in range(2 * L + 1)]
        q_rcv = []
        loads = []
        for k, col in enumerate(p["stage"]):
            slot = col // C
            c4, comp = divmod(slot, 4)
            if c4 not in chunk_done:
                chunk_done.add(c4)
                loads.append(f"            const float4 Q{c4} = sq4[{c4}];")
            q_rcv.append(f"                    const float q{k} = Q{c4}.{'xyzw'[comp]};")
        rcv_prefix = loads
        if a == 0:
            dx_names = ["dx0_acc"]
            x_prefix = "x0_v"
        else:
            dx_names = ["dx1x_acc", "dx1y_acc", "dx1z_acc"]
        if L == 1:
            b.append("                    const float h = q0 * nx + q1 * ny + q2 * nz;")
        elif L == 2:
            b += [
                "                    const float qzz = -q0 - q3;",
                "                    const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz, vz = q2 * nx + q4 * ny + qzz * nz;",
                "                    const float h = nx * vx + ny * vy + nz * vz;",
            ]
        elif L == 3:
            b += [
                "                    const float q7 = -q0 - q3, q8 = -q1 - q5;",
                "                    const float Qxx = q0 * nx + q1 * ny + q2 * nz, Qxy = q1 * nx + q3 * ny + q4 * nz, Qxz = q2 * nx + q4 * ny + q7 * nz;",
                "                    const float Qyy = q3 * nx + q5 * ny + q6 * nz, Qyz = q4 * nx + q6 * ny + q8 * nz, Qzz = -Qxx - Qyy;",
                "                    const float vx = Qxx * nx + Qxy * ny + Qxz * nz, vy = Qxy * nx + Qyy * ny + Qyz * nz, vz = Qxz * nx + Qyz * ny + Qzz * nz;",
                "                    const float h = nx * vx + ny * vy + nz * vz;",
            ]

        def acc(tgt, val):
            return f"                    {tgt} = fmaf({val}, {tgt});"

        if kind == "scalar":
            d = dx_names[0]
            if L == 0:
                b += [acc(d, "w, q0"), "                    pull_R = fmaf(radial_dt, x0_v * q0, pull_R);"]
            elif L == 1:
                b += [
                    "                    const float wx = w * x0_v;",
                    acc("pull_Gx", "wx, q0"), acc("pull_Gy", "wx, q1"), acc("pull_Gz", "wx, q2"),
                    acc(d, "w, h"), "                    pull_R = fmaf(radial_dt, x0_v * h, pull_R);",
                ]
            else:
                b += [
                    f"                    const float wx = {wp._f(L)} * w * x0_v;",
                    acc("pull_Gx", "wx, vx"), acc("pull_Gy", "wx, vy"), acc("pull_Gz", "wx, vz"),
                    acc(d, "w, h"), "                    pull_R = fmaf(radial_dt, x0_v * h, pull_R);",
                ]
        elif kind == "raise":
            if L == 1:
                b += [
                    "                    pull_R = fmaf(radial_dt, xx * q0 + xy * q1 + xz * q2, pull_R);",
                    acc(dx_names[0], "w, q0"), acc(dx_names[1], "w, q1"), acc(dx_names[2], "w, q2"),
                ]
            elif L == 2:
                b += [
                    "                    pull_R = fmaf(radial_dt, 2.0f * (xx * vx + xy * vy + xz * vz), pull_R);",
                    "                    const float w2 = 2.0f * w;",
                    acc(dx_names[0], "w2, vx"), acc(dx_names[1], "w2, vy"), acc(dx_names[2], "w2, vz"),
                    acc("pull_Gx", "w2, q0 * xx + q1 * xy + q2 * xz"),
                    acc("pull_Gy", "w2, q1 * xx + q3 * xy + q4 * xz"),
                    acc("pull_Gz", "w2, q2 * xx + q4 * xy + qzz * xz"),
                ]
            else:
                b += [
                    "                    pull_R = fmaf(radial_dt, 3.0f * (xx * vx + xy * vy + xz * vz), pull_R);",
                    "                    const float w3 = 3.0f * w, w6 = 6.0f * w;",
                    acc(dx_names[0], "w3, vx"), acc(dx_names[1], "w3, vy"), acc(dx_names[2], "w3, vz"),
                    acc("pull_Gx", "w6, Qxx * xx + Qxy * xy + Qxz * xz"),
                    acc("pull_Gy", "w6, Qxy * xx + Qyy * xy + Qyz * xz"),
                    acc("pull_Gz", "w6, Qxz * xx + Qyz * xy + Qzz * xz"),
                ]
        else:  # lower
            k_c = 1.0 / (2 * L + 1)
            b.append("                    const float A = xx * nx + xy * ny + xz * nz;")
            if L == 0:
                b += [
                    "                    const float wq = w * q0;",
                    acc("pull_Gx", "wq, xx"), acc("pull_Gy", "wq, xy"), acc("pull_Gz", "wq, xz"),
                    acc(dx_names[0], "wq, nx"), acc(dx_names[1], "wq, ny"), acc(dx_names[2], "wq, nz"),
                    "                    pull_R = fmaf(radial_dt, A * q0, pull_R);",
                ]
            elif L == 1:
                b += [
                    "                    const float B = xx * q0 + xy * q1 + xz * q2;",
                    f"                    pull_R = fmaf(radial_dt, A * h - {wp._f(k_c)} * B, pull_R);",
                    f"                    const float wh = w * h, wA = w * A, wk = w * {wp._f(k_c)};",
                    "                    pull_Gx = fmaf(wh, xx, fmaf(wA, q0, pull_Gx));",
                    "                    pull_Gy = fmaf(wh, xy, fmaf(wA, q1, pull_Gy));",
                    "                    pull_Gz = fmaf(wh, xz, fmaf(wA, q2, pull_Gz));",
                    f"                    {dx_names[0]} = fmaf(wh, nx, fmaf(-wk, q0, {dx_names[0]}));",
                    f"                    {dx_names[1]} = fmaf(wh, ny, fmaf(-wk, q1, {dx_names[1]}));",
                    f"                    {dx_names[2]} = fmaf(wh, nz, fmaf(-wk, q2, {dx_names[2]}));",
                ]
            elif L == 2:
                b += [
                    "                    const float B2 = xx * vx + xy * vy + xz * vz;",
                    f"                    pull_R = fmaf(radial_dt, A * h - {wp._f(2 * k_c)} * B2, pull_R);",
                    f"                    const float wh = w * h, wA2 = 2.0f * w * A, wk2 = {wp._f(2 * k_c)} * w;",
                    "                    pull_Gx = fmaf(wh, xx, fmaf(wA2, vx, fmaf(-wk2, q0 * xx + q1 * xy + q2 * xz, pull_Gx)));",
                    "                    pull_Gy = fmaf(wh, xy, fmaf(wA2, vy, fmaf(-wk2, q1 * xx + q3 * xy + q4 * xz, pull_Gy)));",
                    "                    pull_Gz = fmaf(wh, xz, fmaf(wA2, vz, fmaf(-wk2, q2 * xx + q4 * xy + qzz * xz, pull_Gz)));",
                    f"                    {dx_names[0]} = fmaf(wh, nx, fmaf(-wk2, vx, {dx_names[0]}));",
                    f"                    {dx_names[1]} = fmaf(wh, ny, fmaf(-wk2, vy, {dx_names[1]}));",
                    f"                    {dx_names[2]} = fmaf(wh, nz, fmaf(-wk2, vz, {dx_names[2]}));",
                ]
        b.append("                }")
        path_bodies.append("\n".join(b))
        nq = 2 * L + 1
        path_bodies_rcv.append("\n".join(rcv_prefix + b[:10] + q_rcv + b[10 + nq:]))

    q_dim = len(schedule.paths) * C
    src = f"""
extern "C" __global__ void __launch_bounds__(256, 4) wprime_backward_inter1(
    int N, const int* __restrict__ E_ptr,
    const float* __restrict__ x0, const float* __restrict__ x1,
    const float4* __restrict__ vec_len,
    const int* __restrict__ senders, const int* __restrict__ receivers,
    const int* __restrict__ act_slab, const float2* __restrict__ act_t_sx,
    const float* __restrict__ dm1_0, const float* __restrict__ dm1_1,
    const float* __restrict__ dm1_2, const float* __restrict__ dm1_3,
    const float* __restrict__ dD1,
    const uint2* __restrict__ spline_table, const float4* __restrict__ density_table,
    unsigned bias_shift,
    float* __restrict__ dx0, float* __restrict__ dx1,
    float* __restrict__ edge_force_1) {{
    constexpr int C = {C};
    constexpr int64_t WEIGHT_DIM = {q_dim};
    constexpr int EDGE_TILE = 64;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int E = *E_ptr;
    const int64_t block_s = (int64_t)blockIdx.x * EDGE_TILE;
    if (block_s >= E) return;
    const int64_t block_e = (block_s + EDGE_TILE < E) ? (block_s + EDGE_TILE) : (int64_t)E;

    // Stage edge metadata in warps 0..1 (64 edges per block)
    __shared__ int s_rcv[EDGE_TILE], s_snd[EDGE_TILE];
    __shared__ float s_t[EDGE_TILE], s_sx[EDGE_TILE], s_r[EDGE_TILE];
    __shared__ float s_nx[EDGE_TILE], s_ny[EDGE_TILE], s_nz[EDGE_TILE], s_ddens[EDGE_TILE];
    __shared__ int64_t s_slab[EDGE_TILE];
    __shared__ unsigned char s_sloped[EDGE_TILE];

    if (tid < EDGE_TILE) {{
        const int64_t e = block_s + tid;
        if (e < block_e) {{
            const float4 vl = __ldg(vec_len + e);
            const float2 tsx = __ldg(act_t_sx + e);
            const float re = vl.w, t = tsx.x, sx_raw = tsx.y;
            const bool sloped = (sx_raw > 0.0f);
            const float sx_inv = fabsf(sx_raw);
            const int64_t slab = (int64_t)act_slab[e];
            s_sloped[tid] = sloped ? 1 : 0;
            s_rcv[tid] = receivers[e];
            s_snd[tid] = senders[e];
            s_r[tid] = re;
            const float vx = vl.x, vy = vl.y, vz = vl.z;
            const float d = re > 1e-12f ? re : 1e-12f;
            s_nx[tid] = vx / d;
            s_ny[tid] = vy / d;
            s_nz[tid] = vz / d;
            s_t[tid] = t;
            s_sx[tid] = sx_inv;
            s_slab[tid] = slab;
            if (sloped) {{
                const float4 dc = __ldg(density_table + slab);
                s_ddens[tid] = (dc.y + t * (2.0f * dc.z + 3.0f * t * dc.w)) * sx_inv;
            }} else {{
                s_ddens[tid] = 0.0f;
            }}
        }}
    }}
    __syncthreads();

    // Stage 1 receiver row at a time (20,480 B SMEM)
    __shared__ __align__(16) float s_dout[{stage_total}];
    __shared__ float s_dD1;

    int64_t stage_s = block_s;
    while (stage_s < block_e) {{
        const int stage_r = s_rcv[stage_s - block_s];
        int64_t stage_t1 = stage_s + 1;
        while (stage_t1 < block_e && s_rcv[stage_t1 - block_s] == stage_r) {{
            ++stage_t1;
        }}
        if (stage_r >= 0 && stage_r < N) {{
            if (threadIdx.x == 0) s_dD1 = __ldg(dD1 + stage_r);
            float* stage_row = s_dout;
            for (int ch = threadIdx.x; ch < {C}; ch += blockDim.x) {{
{chr(10).join(prologue_body)}
            }}
        }}
        __syncthreads();

        for (int64_t e = stage_s + warp; e < stage_t1; e += 8) {{
            const int k = (int)(e - block_s);
            const int snd = s_snd[k];
            const float* stage_row = s_dout;
            const float dD1_r = s_dD1;
            const float t = s_t[k], s = t - 0.5f, s_sq = s * s, sc = s_sx[k];
            const float s1 = s * 0x1.0p-8f, s2 = s_sq * 0x1.0p-14f, s3 = s * s_sq * 0x1.0p-19f;
            const float ds2 = (2.0f * s) * 0x1.0p-6f, ds3 = (3.0f * s_sq) * 0x1.0p-11f;  // radial_dt scaled by 2^8 (exact)
{DECODE_FOLD_CONSTANTS}            const bool sloped = s_sloped[k] != 0;
            const float nx = s_nx[k], ny = s_ny[k], nz = s_nz[k];
            const uint2* slab_ptr = spline_table + s_slab[k] * WEIGHT_DIM;
            float pull_Gx = 0.0f, pull_Gy = 0.0f, pull_Gz = 0.0f, pull_R = 0.0f;

            for (int channel = lane; channel < C; channel += 32) {{
                float dx0_acc = 0.0f, dx1x_acc = 0.0f, dx1y_acc = 0.0f, dx1z_acc = 0.0f;
                const float x0_v = __ldg(x0 + (int64_t)snd * C + channel);
                const float xx = __ldg(x1 + ((int64_t)snd * 3 + 0) * C + channel);
                const float xy = __ldg(x1 + ((int64_t)snd * 3 + 1) * C + channel);
                const float xz = __ldg(x1 + ((int64_t)snd * 3 + 2) * C + channel);
{chr(10).join(path_bodies)}
                atomicAdd(dx0 + (int64_t)snd * C + channel, dx0_acc);
                atomicAdd(dx1 + ((int64_t)snd * 3 + 0) * C + channel, dx1x_acc);
                atomicAdd(dx1 + ((int64_t)snd * 3 + 1) * C + channel, dx1y_acc);
                atomicAdd(dx1 + ((int64_t)snd * 3 + 2) * C + channel, dx1z_acc);
            }}

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {{
                pull_Gx += __shfl_xor_sync(0xffffffffu, pull_Gx, off, 32);
                pull_Gy += __shfl_xor_sync(0xffffffffu, pull_Gy, off, 32);
                pull_Gz += __shfl_xor_sync(0xffffffffu, pull_Gz, off, 32);
                pull_R  += __shfl_xor_sync(0xffffffffu, pull_R,  off, 32);
            }}
            if (lane == 0) {{
                const float rad = sloped ? (pull_R * 0x1.0p-8f * sc + dD1_r * s_ddens[k]) : 0.0f;
                const float d = s_r[k] > 1e-12f ? s_r[k] : 1e-12f;
                const float inv_r = 1.0f / d;
                const float gdotn = pull_Gx * nx + pull_Gy * ny + pull_Gz * nz;
                const float fx = nx * rad + (pull_Gx - nx * gdotn) * inv_r;
                const float fy = ny * rad + (pull_Gy - ny * gdotn) * inv_r;
                const float fz = nz * rad + (pull_Gz - nz * gdotn) * inv_r;
                edge_force_1[e * 3 + 0] = fx;
                edge_force_1[e * 3 + 1] = fy;
                edge_force_1[e * 3 + 2] = fz;
            }}
        }}
        __syncthreads();
        stage_s = stage_t1;
    }}
}}

extern "C" __global__ void __launch_bounds__(512, 2) wprime_backward_inter1_rcv(
    int N, const int* __restrict__ E_ptr,
    const float* __restrict__ x0, const float* __restrict__ x1,
    const float4* __restrict__ vec_len,
    const int* __restrict__ senders, const int* __restrict__ act_row_ptr,
    const int* __restrict__ act_slab, const float2* __restrict__ act_t_sx,
    const float* __restrict__ dm1_0, const float* __restrict__ dm1_1,
    const float* __restrict__ dm1_2, const float* __restrict__ dm1_3,
    const float* __restrict__ dD1,
    const uint2* __restrict__ spline_table, const float4* __restrict__ density_table,
    unsigned bias_shift,
    float* __restrict__ dx0, float* __restrict__ dx1,
    float* __restrict__ edge_force_1) {{
    constexpr int C = {C};
    constexpr int64_t WEIGHT_DIM = {q_dim};
    const int rcv = blockIdx.x;
    if (rcv >= N) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = blockDim.x >> 5;  // 256 or 512 threads (tune_rebuild_variants)
    const int E = *E_ptr;
    int64_t eb = act_row_ptr[rcv], ee = act_row_ptr[rcv + 1];
    eb = eb < 0 ? 0 : eb;
    ee = ee > E ? E : ee;
    if (eb >= ee) return;

    // Stage the receiver's q once per block, channel-major ({C} x {QS} floats, 1 barrier total)
    __shared__ __align__(16) float s_q[{C * QS}];
    {{
        const int stage_r = rcv;
        for (int ch = tid; ch < {C}; ch += blockDim.x) {{
{chr(10).join(prologue_rcv)}
        }}
    }}
    __syncthreads();

    const float dD1_r = __ldg(dD1 + rcv);
    for (int64_t e = eb + warp; e < ee; e += n_warps) {{
        const int snd = __ldg(senders + e);
        const float4 vl = __ldg(vec_len + e);
        const float2 tsx = __ldg(act_t_sx + e);
        const float re = vl.w, t = tsx.x;
        const int64_t slab = (int64_t)__ldg(act_slab + e);
        const float d = re > 1e-12f ? re : 1e-12f;
        const float inv_r = 1.0f / d;
        const float nx = vl.x * inv_r, ny = vl.y * inv_r, nz = vl.z * inv_r;
        const float s = t - 0.5f, s_sq = s * s;
        const float s1 = s * 0x1.0p-8f, s2 = s_sq * 0x1.0p-14f, s3 = s * s_sq * 0x1.0p-19f;
        const float ds2 = (2.0f * s) * 0x1.0p-6f, ds3 = (3.0f * s_sq) * 0x1.0p-11f;  // radial_dt scaled by 2^8 (exact)
{DECODE_FOLD_CONSTANTS}        const uint2* slab_ptr = spline_table + slab * WEIGHT_DIM;
        float pull_Gx = 0.0f, pull_Gy = 0.0f, pull_Gz = 0.0f, pull_R = 0.0f;

        for (int channel = lane; channel < C; channel += 32) {{
            const float4* sq4 = reinterpret_cast<const float4*>(s_q + channel * {QS});
            float dx0_acc = 0.0f, dx1x_acc = 0.0f, dx1y_acc = 0.0f, dx1z_acc = 0.0f;
            const float x0_v = __ldg(x0 + (int64_t)snd * C + channel);
            const float xx = __ldg(x1 + ((int64_t)snd * 3 + 0) * C + channel);
            const float xy = __ldg(x1 + ((int64_t)snd * 3 + 1) * C + channel);
            const float xz = __ldg(x1 + ((int64_t)snd * 3 + 2) * C + channel);
{chr(10).join(path_bodies_rcv)}
            atomicAdd(dx0 + (int64_t)snd * C + channel, dx0_acc);
            atomicAdd(dx1 + ((int64_t)snd * 3 + 0) * C + channel, dx1x_acc);
            atomicAdd(dx1 + ((int64_t)snd * 3 + 1) * C + channel, dx1y_acc);
            atomicAdd(dx1 + ((int64_t)snd * 3 + 2) * C + channel, dx1z_acc);
        }}

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {{
            pull_Gx += __shfl_xor_sync(0xffffffffu, pull_Gx, off, 32);
            pull_Gy += __shfl_xor_sync(0xffffffffu, pull_Gy, off, 32);
            pull_Gz += __shfl_xor_sync(0xffffffffu, pull_Gz, off, 32);
            pull_R  += __shfl_xor_sync(0xffffffffu, pull_R,  off, 32);
        }}
        if (lane == 0) {{
            const float sx_raw = tsx.y;
            float rad = 0.0f;
            if (sx_raw > 0.0f) {{
                const float4 dc = __ldg(density_table + slab);
                const float ddens = (dc.y + t * (2.0f * dc.z + 3.0f * t * dc.w)) * sx_raw;
                rad = pull_R * 0x1.0p-8f * sx_raw + dD1_r * ddens;
            }}
            const float gdotn = pull_Gx * nx + pull_Gy * ny + pull_Gz * nz;
            edge_force_1[e * 3 + 0] = nx * rad + (pull_Gx - nx * gdotn) * inv_r;
            edge_force_1[e * 3 + 1] = ny * rad + (pull_Gy - ny * gdotn) * inv_r;
            edge_force_1[e * 3 + 2] = nz * rad + (pull_Gz - nz * gdotn) * inv_r;
        }}
    }}
}}
"""
    return src


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/fzwin/models/mace-omat-0-medium.model"
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "/mnt/d/Margraf/LoupMD/src/generated_kernels.cuh")
    check_architecture(torch.load(model_path, map_location="cpu", weights_only=False))
    mc = _load_model(model_path)

    # 1. Symmetric contractions (prod[0] and prod[1])
    sc_parts, product_sigs = [], []
    for idx, emitter in ((0, _emit_sc0), (1, _emit_sc1)):
        sc = mc.products[idx].symmetric_contractions
        sh = shape_of(sc)
        terms = merged_terms(sc.etp.polynomial)
        prog, feats, _ = factor(terms, sh[0], sh[1], sh[2])
        sc_parts.append(emitter(prog, feats, sh[1], sh[3]))
        product_sigs.append(product_signature(prog, feats, sh))

    # 2. Interaction 1 schedule
    class _ProblemWrap:
        pass
    # Use FastMACE render_sources problem (exact OMAT-0 inter[1])
    sys.path.insert(0, str(FASTMACE_ROOT / "results" / "wprime-h200-20261111"))
    import render_sources as rs
    schedule, _ = render_kernels(rs.problem(), np.float32, NUM_THREADS, "ir_mul")
    blocks = rs.blocks_of(schedule)

    fwd1 = _emit_spline_forward1(schedule, blocks, C=128)
    bwd1 = _emit_wprime_backward1(schedule, blocks, C=128)

    header = """#pragma once
// Auto-generated by LoupMD/python/gen_kernels.py from mace-omat-0-medium.model
// Do not edit manually.
#include "kernels_common.cuh"
#include <cuda_runtime.h>
#include <cstdint>

"""
    out_path.write_text(header + "\n".join(sc_parts) + "\n" + fwd1 + "\n" + bwd1)
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
    # The loader refuses bundles whose architecture or factored product programs differ from the ones these kernels
    # were generated for (the product kernels above evaluate exactly prog / feats).
    sig_path = out_path.with_name("generated_kernels_signature.h")
    sig_path.write_text("#pragma once\n// Auto-generated by LoupMD/python/gen_kernels.py; matches export_omat0.kernel_signature().\n"
                        f'#define LOUP_KERNEL_SIGNATURE "{kernel_signature(product_sigs)}"\n')
    print(f"Wrote {sig_path}")


if __name__ == "__main__":
    main()
