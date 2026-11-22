// The kind-4 gate's backward for the atoms of one staged output tile: the share of gate_bwd_kernel (generic.cu) that
// the W_2^T GEMMs take into their epilogue (gen-gate-bwd-epilogue): gate_bwd_tile, for the own SIMT sgemm128
// (sgemm.cu, since list 27 item 2) and the two-half FP16 tensor-core GEMM (hgemm2.cu, LOUP_W2T_HALF; since list 26,
// ledger gen-h2-gate-tile). Its per-element arithmetic is gate_bwd_kernel's in its order; only dD's sum is reassociated.
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

// dgout_l for the tile's atoms a0 .. a0 + na - 1 (DEG = 2l + 1 rows each) at the tile's columns n0 .. n0 + 63 of
// degree l, staged in shared memory (row r of the tile, local column cl at sC[r * SROW + cl]), with gate_bwd_kernel's
// arithmetic in its order (ledger gen-h2-gate-tile, list 26 item 2; for sgemm128 too since list 27 item 2, replacing
// gate_bwd_rows, one atom at a time per thread group), with (a) the tile's per-atom scale s = 1 / (D beta + alpha)
// and res0 row staged once in shared memory (sS, sQ) instead of per thread and atom; (b) B atoms' operand loads issued
// before their arithmetic (DEG 1:
// 8 atoms, DEG 3: 2; DEG >= 5's lin_l / res_l rows already give a dozen independent loads an atom), since one atom at
// a time left DEG 1's epilogue waiting on its two loads (ncu: long_scoreboard 2.7 per issued instruction, the kernel
// 3x hgemm2f's time on the same grid); (c) each thread's dD partial of an (atom, column) stored in that atom's first
// staged row (and the vform partial in its second), which only this thread reads, and one thread per atom sums the
// 64 columns (gate_bwd_tile_sum), instead of a 32-lane shuffle tree and a shared atomic per warp and atom. Thread
// (grp, cl) = (tid / 64, tid % 64) of NGRP groups takes the atoms grp, grp + NGRP, .. at column n0 + cl.
template <int DEG, int SROW, int NGRP>
__device__ __forceinline__ void gate_bwd_tile(float* sC, const float* sS, const int* sQ, int a0, int na, int n0,
                                              int tid, const GateArgs& g) {
    const int N = g.C;
    const int cl = tid & 63, grp = tid >> 6, c = n0 + cl;
    constexpr int B = DEG == 1 ? 8 : (DEG == 3 ? 2 : 1);
    constexpr int l = DEG / 2;
    for (int ai0 = grp; ai0 < na; ai0 += NGRP * B) {
        float l0[B], r0[B], li[B][DEG], rl[B][DEG];
#pragma unroll
        for (int j = 0; j < B; ++j) {
            const int ai = ai0 + NGRP * j;
            const bool ok = ai < na;
            const int aj = ok ? ai : 0;
            const int64_t ro = (int64_t)(a0 + aj) * 4 * N + l * N + c;
            l0[j] = ok ? g.lin0[ro] : 0.0f;
            r0[j] = ok && g.res0 ? g.res0[(int64_t)sQ[aj] * 4 * N + l * N + c] : 0.0f;
            if constexpr (DEG > 1) {
#pragma unroll
                for (int m = 0; m < DEG; ++m) {
                    const int64_t idx = ((int64_t)(a0 + aj) * DEG + m) * N + c;
                    li[j][m] = ok ? g.lin_l[idx] : 0.0f;
                    rl[j][m] = ok && !g.vform && g.res_l ? g.res_l[idx] : 0.0f;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < B; ++j) {
            const int ai = ai0 + NGRP * j;
            if (ai >= na) break;
            const int a = a0 + ai;
            const float s = sS[ai];
            const int64_t ro = (int64_t)a * 4 * N + l * N + c;
            float* go = sC + ai * DEG * SROW + cl;
            float part = 0.0f, partq = 0.0f;
            if constexpr (DEG == 1) {
                const float p = l0[j] * s + r0[j];
                const float d = go[0] * g.cs * dsilu_f(p);
                if (g.dp0) g.dp0[ro] = d;
                g.dl0[ro] = d * s;
                part = fmaf(d, l0[j], part);
            } else {
                const float gp = l0[j] * s + r0[j];
                const float sg = sigmoid_f(gp);
                const float gt = g.cg * sg;
                float dgp;
                if (g.vform) {
#pragma unroll
                    for (int m = 0; m < DEG; ++m) {
                        const int64_t idx = ((int64_t)a * DEG + m) * N + c;
                        const float o = go[m * SROW];
                        partq = fmaf(o, li[j][m], partq);
                        g.dl_l[idx] = o * gt * s;
                    }
                    dgp = (1.0f - sg) * partq;
                } else {
                    float dg = 0.0f;
#pragma unroll
                    for (int m = 0; m < DEG; ++m) {
                        const int64_t idx = ((int64_t)a * DEG + m) * N + c;
                        const float p = li[j][m] * s + rl[j][m];
                        const float o = go[m * SROW];
                        dg = fmaf(o, p, dg);
                        const float d = o * gt;
                        if (g.dp_l) g.dp_l[idx] = d;
                        g.dl_l[idx] = d * s;
                        part = fmaf(d, li[j][m], part);
                    }
                    dgp = dg * g.cg * sg * (1.0f - sg);
                }
                if (g.dp0) g.dp0[ro] = dgp;
                g.dl0[ro] = dgp * s;
                part = fmaf(dgp, l0[j], part);
                go[SROW] = partq;  // the atom's second staged row, read above by this thread only
            }
            go[0] = part;
        }
    }
}

// After gate_bwd_tile: thread tid (< na) adds atom tid's 64 column partials to p (and its vform partials to q). The
// reads are rotated so a quarter warp's 16-byte loads cover the banks once: by 4 every 4 threads where the atoms' rows
// are 8 banks apart (SROW 72, hgemm2_kernel), by 4 every thread where they start on one bank (SROW 64, sgemm128).
template <int DEG, int SROW>
__device__ __forceinline__ void gate_bwd_tile_sum(const float* sC, int tid, const GateArgs& g, float& p, float& q) {
    const float* row = sC + tid * DEG * SROW;
    const bool vq = DEG > 1 && g.vform;
#pragma unroll
    for (int k = 0; k < 64; k += 4) {
        const int kk = (k + (SROW % 32 == 0 ? 4 * tid : 4 * (tid >> 2))) & 63;
        const float4 v = *reinterpret_cast<const float4*>(row + kk);
        p += (v.x + v.y) + (v.z + v.w);
        if (vq) {
            const float4 w = *reinterpret_cast<const float4*>(row + SROW + kk);
            q += (w.x + w.y) + (w.z + w.w);
        }
    }
}

}  // namespace
}  // namespace loup
