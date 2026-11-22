// The kind-4 gate's backward for the atoms of one staged output tile: the share of gate_bwd_kernel (generic.cu) that
// the W_2^T GEMMs take into their epilogue (gen-gate-bwd-epilogue), for the own SIMT sgemm128 (sgemm.cu). The two-half
// FP16 tensor-core GEMM (hgemm2.cu, LOUP_W2T_HALF) runs the same per-element arithmetic in the same order in its own
// gate_bwd_tile (list 26, ledger gen-h2-gate-tile: batched loads, dD summed per atom from shared memory), so the two
// differ only in the order of dD's sum.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "generic.hpp"

namespace loup {
namespace {

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float dsilu_f(float x) {
    const float s = sigmoid_f(x);
    return s * (1.0f + x * (1.0f - s));
}

// dgout_l for the tile's atoms a0 .. a0 + na - 1 (DEG = 2l + 1 rows each) at column c of degree l, staged in shared
// memory: row r (of the tile), local column cl at sC[r * SROW + cl]. Thread group grp of NGRP (warp-aligned, the
// same grp for the 32 lanes of a warp) runs the atoms grp, grp + NGRP, .. with gate_bwd_kernel's arithmetic in its
// order and adds each atom's dD partial over its 32 lanes to s_acc / s_accq (one shared atomic per warp and atom).
template <int DEG, int NGRP, int SROW>
__device__ __forceinline__ void gate_bwd_rows(const float* sC, float* s_acc, float* s_accq, int a0, int na, int c,
                                              int cl, int grp, int tid, const GateArgs& g) {
    const int N = g.C;
    for (int ai = grp; ai < na; ai += NGRP) {
        const int a = a0 + ai;
        const float s = 1.0f / (g.D[a] * g.beta + g.alpha);
        const int64_t r0 = (int64_t)a * 4 * N, q0 = g.res0_sp ? (int64_t)g.res0_sp[a] * 4 * N : r0;
        auto pre0 = [&](int col) { return g.lin0[r0 + col] * s + (g.res0 ? g.res0[q0 + col] : 0.0f); };
        const float* go = sC + ai * DEG * SROW + cl;
        float part = 0.0f, partq = 0.0f;
        if (DEG == 1) {
            const float p = pre0(c);
            const float d = go[0] * g.cs * dsilu_f(p);
            if (g.dp0) g.dp0[r0 + c] = d;
            g.dl0[r0 + c] = d * s;
            part = fmaf(d, g.lin0[r0 + c], part);
        } else {
            constexpr int l = DEG / 2;
            const float gp = pre0(l * N + c);
            const float sg = sigmoid_f(gp);
            const float gt = g.cg * sg;
            float dgp;
            if (g.vform) {  // lin_l holds v = gt (lin * s): d gp = (1 - sg) sum dgout v, dD += -beta s sum dgout v
#pragma unroll
                for (int m = 0; m < DEG; ++m) {
                    const int64_t idx = ((int64_t)a * DEG + m) * N + c;
                    const float o = go[m * SROW];
                    partq = fmaf(o, g.lin_l[idx], partq);
                    g.dl_l[idx] = o * gt * s;
                }
                dgp = (1.0f - sg) * partq;
            } else {
                float dg = 0.0f;
#pragma unroll
                for (int m = 0; m < DEG; ++m) {
                    const int64_t idx = ((int64_t)a * DEG + m) * N + c;
                    const float li = g.lin_l[idx];
                    const float p = li * s + (g.res_l ? g.res_l[idx] : 0.0f);
                    const float o = go[m * SROW];
                    dg = fmaf(o, p, dg);
                    const float d = o * gt;
                    if (g.dp_l) g.dp_l[idx] = d;
                    g.dl_l[idx] = d * s;
                    part = fmaf(d, li, part);
                }
                dgp = dg * g.cg * sg * (1.0f - sg);
            }
            if (g.dp0) g.dp0[r0 + l * N + c] = dgp;
            g.dl0[r0 + l * N + c] = dgp * s;
            part = fmaf(dgp, g.lin0[r0 + l * N + c], part);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            part += __shfl_xor_sync(0xffffffffu, part, off);
            partq += __shfl_xor_sync(0xffffffffu, partq, off);
        }
        if ((tid & 31) == 0) {
            atomicAdd(s_acc + ai, part);
            if (DEG > 1 && g.vform) atomicAdd(s_accq + ai, partq);
        }
    }
}

// The tile's dD: one global atomic per atom (threads 0 .. na - 1), after the block's last gate_bwd_rows.
__device__ __forceinline__ void gate_bwd_dD(const float* s_acc, const float* s_accq, int a0, int na, int tid,
                                            const GateArgs& g) {
    if (tid < na) {
        const int a = a0 + tid;
        const float s = 1.0f / (g.D[a] * g.beta + g.alpha);
        atomicAdd(g.dD + a, -g.beta * s * s * s_acc[tid] - g.beta * s * s_accq[tid]);
    }
}

}  // namespace
}  // namespace loup
