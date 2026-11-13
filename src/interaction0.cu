#include "kernels_common.cuh"
#include "model.hpp"

namespace loup {

// Receiver-owned interaction 0 forward: one warp per receiver (4 per 128-thread block), each lane two adjacent radial
// channels (64-bit coefficient loads; one broadcast read of an edge's staged SH serves both).
// Active edges of a receiver are contiguous and sorted by (pair, r); for a fixed receiver the pair index grows with
// the sender species, so each sender-species slot is one contiguous run in ascending species order. Its 16 sums stay
// in registers and are stored once (no atomics); slots of absent species are written as zeros, and D0 is always
// written, so S0 and D0 need no per-step zero fill. Edges are staged 32 at a time, one per lane: metadata in
// registers (broadcast by shuffles), SH in the warp's shared rows, and the edge's D0 radial term summed right away.
__global__ void __launch_bounds__(128) tabled_outer_sum0_forward_kernel(
    int N, int S, int pieces0, int n_bins0, float r_lo0, float bin_scale0,
    const int* __restrict__ act_row_ptr,
    const float4* __restrict__ vec_len,
    const int* __restrict__ senders,
    const int* __restrict__ species,
    const float* __restrict__ coef0,   // [P_sym, pieces0, 6, 66]
    const float* __restrict__ starts0,
    const float* __restrict__ scales0,
    const int32_t* __restrict__ bins0,
    float* __restrict__ S0_0,
    float* __restrict__ S0_1,
    float* __restrict__ S0_2,
    float* __restrict__ S0_3,
    float* __restrict__ D0
) {
    constexpr int TILE = 32;
    constexpr int J = 66;
    constexpr int TERMS = 6;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int rcv = blockIdx.x * 4 + warp;
    __shared__ __align__(16) float s_sh[4][TILE * 16];
    if (rcv >= N) return;  // warp-level only (no block barriers below)
    const int eb = act_row_ptr[rcv];
    const int ee = act_row_ptr[rcv + 1];
    const int rcv_sp = species[rcv];
    float* sh_w = s_sh[warp];

    float acc0[16], acc1[16];
    #pragma unroll
    for (int d = 0; d < 16; ++d) { acc0[d] = 0.0f; acc1[d] = 0.0f; }
    float d0_acc = 0.0f;
    int cur_sp = -1;
    const int64_t row_stride = (int64_t)S * 64;

    auto flush = [&](int sp) {
        const int64_t col = (int64_t)sp * 64 + 2 * lane;
        auto st = [&](float* base, int64_t row, int d) {
            *reinterpret_cast<float2*>(base + row * row_stride + col) = make_float2(acc0[d], acc1[d]);
        };
        st(S0_0, rcv, 0);
        #pragma unroll
        for (int d = 0; d < 3; ++d) st(S0_1, (int64_t)rcv * 3 + d, 1 + d);
        #pragma unroll
        for (int d = 0; d < 5; ++d) st(S0_2, (int64_t)rcv * 5 + d, 4 + d);
        #pragma unroll
        for (int d = 0; d < 7; ++d) st(S0_3, (int64_t)rcv * 7 + d, 9 + d);
        #pragma unroll
        for (int d = 0; d < 16; ++d) { acc0[d] = 0.0f; acc1[d] = 0.0f; }
    };

    for (int base = eb; base < ee; base += TILE) {
        const int cnt = (ee - base < TILE) ? (ee - base) : TILE;
        float m_x = 0.0f, m_d0 = 0.0f;
        int m_base = 0, m_sp = 0;
        if (lane < cnt) {
            const int e = base + lane;
            const float4 vl = __ldg(vec_len + e);
            const float r = vl.w;
            edge_sh16(vl.x, vl.y, vl.z, r, reinterpret_cast<float4*>(sh_w + lane * 16));
            const int snd_sp = species[senders[e]];
            const float clamped = r > r_lo0 ? r : r_lo0;
            int bin = (int)((clamped - r_lo0) * bin_scale0);
            bin = bin < 0 ? 0 : (bin >= n_bins0 ? n_bins0 - 1 : bin);
            const int piece = bins0[bin];
            m_x = (clamped - starts0[piece]) * scales0[piece] - 1.0f;
            m_base = (sym_pair(snd_sp, rcv_sp, S) * pieces0 + piece) * TERMS * J;
            m_sp = snd_sp;
            // The edge's D0 radial term (Chebyshev column J - 1)
            const float* cb = coef0 + m_base;
            const float two_x = 2.0f * m_x;
            float c1 = 0.0f, c2 = 0.0f;
            #pragma unroll
            for (int t = TERMS - 1; t >= 1; --t) {
                const float cn = fmaf(two_x, c1, __ldg(cb + t * J + (J - 1)) - c2);
                c2 = c1;
                c1 = cn;
            }
            m_d0 = fmaf(m_x, c1, __ldg(cb + (J - 1)) - c2);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) m_d0 += __shfl_xor_sync(0xffffffffu, m_d0, off);
        d0_acc += m_d0;
        __syncwarp();
        for (int k = 0; k < cnt; ++k) {
            const int sp = __shfl_sync(0xffffffffu, m_sp, k);
            if (sp != cur_sp) {
                if (cur_sp >= 0) flush(cur_sp);
                for (int q = cur_sp + 1; q < sp; ++q) flush(q);  // absent species: zeros
                cur_sp = sp;
            }
            const float x = __shfl_sync(0xffffffffu, m_x, k);
            const int cbase = __shfl_sync(0xffffffffu, m_base, k);
            const float2* cb = reinterpret_cast<const float2*>(coef0 + cbase + 2 * lane);
            const float two_x = 2.0f * x;
            float v1a = 0.0f, v2a = 0.0f, v1b = 0.0f, v2b = 0.0f;
            #pragma unroll
            for (int t = TERMS - 1; t >= 1; --t) {
                const float2 c = __ldg(cb + t * J / 2);
                const float va = fmaf(two_x, v1a, c.x - v2a);
                const float vb = fmaf(two_x, v1b, c.y - v2b);
                v2a = v1a; v1a = va;
                v2b = v1b; v1b = vb;
            }
            const float2 c0 = __ldg(cb);
            const float a0 = fmaf(x, v1a, c0.x - v2a);
            const float a1 = fmaf(x, v1b, c0.y - v2b);
            const float4* she = reinterpret_cast<const float4*>(sh_w + k * 16);
            const float4 Y0 = she[0], Y1 = she[1], Y2 = she[2], Y3 = she[3];
            const float y[16] = {Y0.x, Y0.y, Y0.z, Y0.w, Y1.x, Y1.y, Y1.z, Y1.w,
                                 Y2.x, Y2.y, Y2.z, Y2.w, Y3.x, Y3.y, Y3.z, Y3.w};
            #pragma unroll
            for (int d = 0; d < 16; ++d) {
                acc0[d] = fmaf(y[d], a0, acc0[d]);
                acc1[d] = fmaf(y[d], a1, acc1[d]);
            }
        }
        __syncwarp();
    }
    if (cur_sp >= 0) flush(cur_sp);
    for (int q = cur_sp + 1; q < S; ++q) flush(q);
    if (lane == 0) D0[rcv] = d0_acc;
}

// Interaction 0 tabled outer sum backward + fused SH pullback, ZBL, and force/virial assembly:
// Grid: ceil(E / 128), Block: 128 threads (4 warps x 32 edges/warp, 2 channels/lane).
// Stages each warp's 32 edges' SH (computed from vec_len), then their (dr0, dsh0[1..15]) results, in 10 KB SMEM; all
// 32 lanes per warp then assemble 32 edges in parallel (no dsh0/dr0 round-trip, no separate force_virial launch).
__global__ void __launch_bounds__(128) tabled_outer_sum0_backward_kernel(
    const int* __restrict__ E_ptr, int S, int pieces0, int n_bins0, float r_lo0, float bin_scale0, float r_max,
    int compute_stress, float zbl_r_max_max, float zbl_p,
    float zbl_c0, float zbl_c1, float zbl_c2, float zbl_c3,
    const float4* __restrict__ vec_len,
    const int* __restrict__ senders,
    const int* __restrict__ receivers,
    const int* __restrict__ pairs,
    const int* __restrict__ species,
    const float* __restrict__ pcoef0,  // [P_sym, pieces0, 6, 66] power basis in x
    const float* __restrict__ starts0,
    const float* __restrict__ scales0,
    const int32_t* __restrict__ bins0,
    const float* __restrict__ dS0_0,
    const float* __restrict__ dS0_1,
    const float* __restrict__ dS0_2,
    const float* __restrict__ dS0_3,
    const float* __restrict__ dD0,
    const float* __restrict__ zbl_r_cut,
    const float* __restrict__ zbl_inv_a,
    const float* __restrict__ zbl_pref,
    const float* __restrict__ edge_force_1,
    double* __restrict__ energy_f64,
    double* __restrict__ forces_f64,
    double* __restrict__ virial_grad_f64
) {
    constexpr int CHUNK = 32;
    constexpr int J = 66;
    constexpr int TERMS = 6;

    const int E = *E_ptr;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int warp_id = blockIdx.x * 4 + warp;
    const int lane = tid & 31;
    // Lane owns the adjacent channels 2 * lane and 2 * lane + 1 (64-bit loads of coefficients and dS0 rows).
    const int start = warp_id * CHUNK;

    // [warp][edge][0..15]: first the edge's SH (staged), then 0 = dr0, 1..15 = dsh0[1..15]. Stride 20 keeps rows
    // 16-byte aligned for 128-bit access and conflict-free across a quarter warp (5 x 16 B per row).
    __shared__ __align__(16) float s_edge[4][CHUNK][20];
    double v00 = 0.0, v01 = 0.0, v02 = 0.0, v11 = 0.0, v12 = 0.0, v22 = 0.0;
    double e_zbl_thread = 0.0;

    if (start < E) {
        const int stop = (start + CHUNK < E) ? (start + CHUNK) : E;
        const int cnt = stop - start;
        // Lane k < cnt holds edge start + k's metadata.
        float m_x = 0.0f, m_sc = 0.0f;
        int m_base = 0, m_rcv = -1, m_sp = -1, m_flags = 0;  // flags: 1 = r < r_max, 2 = r >= r_lo0
        if (lane < cnt) {
            const int e = start + lane;
            const float4 vl = __ldg(vec_len + e);
            const float r = vl.w;
            // Stage the edge's SH in its s_edge row; edge k's row is read before its results overwrite it (the shuffle
            // fold in between synchronizes the warp).
            edge_sh16(vl.x, vl.y, vl.z, r, reinterpret_cast<float4*>(s_edge[warp][lane]));
            m_rcv = receivers[e];
            m_sp = species[senders[e]];
            const int rcv_sp = species[m_rcv];
            const float clamped = r > r_lo0 ? r : r_lo0;
            int bin = (int)((clamped - r_lo0) * bin_scale0);
            if (bin < 0) bin = 0;
            if (bin >= n_bins0) bin = n_bins0 - 1;
            const int piece = bins0[bin];
            m_sc = scales0[piece];
            m_x = (clamped - starts0[piece]) * m_sc - 1.0f;
            m_base = (sym_pair(m_sp, rcv_sp, S) * pieces0 + piece) * TERMS * J;
            m_flags = (r < r_max ? 1 : 0) | (r >= r_lo0 ? 2 : 0);
        }
        __syncwarp();

        float blk0[16], blk1[16];
        #pragma unroll
        for (int d = 0; d < 16; ++d) { blk0[d] = 0.0f; blk1[d] = 0.0f; }
        float dD0_r = 0.0f;
        int cur_rcv = -1;
        int cur_snd_sp = -1;

        for (int k = 0; k < cnt; ++k) {
            const int e = start + k;
            const int flags = __shfl_sync(0xffffffffu, m_flags, k);
            if (!(flags & 1)) continue;
            const int rcv = __shfl_sync(0xffffffffu, m_rcv, k);
            const int snd_sp = __shfl_sync(0xffffffffu, m_sp, k);

            if (rcv != cur_rcv || snd_sp != cur_snd_sp) {
                cur_rcv = rcv;
                cur_snd_sp = snd_sp;
                const int64_t row_stride = (int64_t)S * 64;
                const int64_t off = (int64_t)snd_sp * 64 + 2 * lane;
                auto ld2 = [&](const float* base_ptr, int64_t row, int d) {
                    const float2 v = __ldg(reinterpret_cast<const float2*>(base_ptr + row * row_stride + off));
                    blk0[d] = v.x;
                    blk1[d] = v.y;
                };
                ld2(dS0_0, rcv, 0);
                #pragma unroll
                for (int d = 0; d < 3; ++d) ld2(dS0_1, (int64_t)rcv * 3 + d, 1 + d);
                #pragma unroll
                for (int d = 0; d < 5; ++d) ld2(dS0_2, (int64_t)rcv * 5 + d, 4 + d);
                #pragma unroll
                for (int d = 0; d < 7; ++d) ld2(dS0_3, (int64_t)rcv * 7 + d, 9 + d);
                dD0_r = __ldg(dD0 + rcv);
            }

            const float sc = __shfl_sync(0xffffffffu, m_sc, k);
            const float x = __shfl_sync(0xffffffffu, m_x, k);
            const int64_t base = __shfl_sync(0xffffffffu, m_base, k);
            const float2* pc = reinterpret_cast<const float2*>(pcoef0 + base + 2 * lane);
            const float2 c5 = __ldg(pc + (TERMS - 1) * J / 2);
            float pa0 = c5.x, pd0 = 0.0f;
            float pa1 = c5.y, pd1 = 0.0f;
            #pragma unroll
            for (int t = TERMS - 2; t >= 0; --t) {
                const float2 ct = __ldg(pc + t * J / 2);
                pd0 = fmaf(pd0, x, pa0);
                pa0 = fmaf(pa0, x, ct.x);
                pd1 = fmaf(pd1, x, pa1);
                pa1 = fmaf(pa1, x, ct.y);
            }

            // The edge's SH from its staged row (4 broadcast 128-bit shared loads)
            const float4* shv = reinterpret_cast<const float4*>(s_edge[warp][k]);
            const float4 Y0 = shv[0], Y1 = shv[1], Y2 = shv[2], Y3 = shv[3];
            const float shk[16] = {Y0.x, Y0.y, Y0.z, Y0.w, Y1.x, Y1.y, Y1.z, Y1.w,
                                   Y2.x, Y2.y, Y2.z, Y2.w, Y3.x, Y3.y, Y3.z, Y3.w};
            float grad_a0 = 0.0f, grad_a1 = 0.0f;
            float gy[16];
            #pragma unroll
            for (int d = 0; d < 16; ++d) {
                const float sh_d = shk[d];
                grad_a0 = fmaf(blk0[d], sh_d, grad_a0);
                grad_a1 = fmaf(blk1[d], sh_d, grad_a1);
                gy[d] = fmaf(blk0[d], pa0, blk1[d] * pa1);
            }

            // 16-shuffle fold of gy[0..15] across 32 lanes -> lane 0..15 holds dsh0[e, lane]
            float g8[8];
            {
                const bool odd = (lane & 1) != 0;
                #pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const float keep = odd ? gy[2 * q + 1] : gy[2 * q];
                    const float send = odd ? gy[2 * q] : gy[2 * q + 1];
                    g8[q] = keep + __shfl_xor_sync(0xffffffffu, send, 1);
                }
            }
            float g4[4];
            {
                const bool hi = (lane & 2) != 0;
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const float keep = hi ? g8[2 * q + 1] : g8[2 * q];
                    const float send = hi ? g8[2 * q] : g8[2 * q + 1];
                    g4[q] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
                }
            }
            float g2[2];
            {
                const bool hi = (lane & 4) != 0;
                #pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const float keep = hi ? g4[2 * q + 1] : g4[2 * q];
                    const float send = hi ? g4[2 * q] : g4[2 * q + 1];
                    g2[q] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
                }
            }
            float g1;
            {
                const bool hi = (lane & 8) != 0;
                const float keep = hi ? g2[1] : g2[0];
                const float send = hi ? g2[0] : g2[1];
                g1 = keep + __shfl_xor_sync(0xffffffffu, send, 8);
            }
            g1 += __shfl_xor_sync(0xffffffffu, g1, 16);
            if (lane >= 1 && lane < 16) s_edge[warp][k][lane] = g1;

            float dr_v = fmaf(grad_a0, pd0, grad_a1 * pd1);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dr_v += __shfl_xor_sync(0xffffffffu, dr_v, off);
            if (lane == 0) {
                float total_dr = dr_v * sc;
                if (flags & 2) {
                    float qd = 0.0f, qa = __ldg(pcoef0 + base + (TERMS - 1) * J + (J - 1));
                    #pragma unroll
                    for (int t = TERMS - 2; t >= 0; --t) {
                        qd = fmaf(qd, x, qa);
                        qa = fmaf(qa, x, __ldg(pcoef0 + base + t * J + (J - 1)));
                    }
                    total_dr += dD0_r * qd * sc;
                    s_edge[warp][k][0] = total_dr;
                } else {
                    s_edge[warp][k][0] = 0.0f;
                }
            }
        }

        __syncwarp();
        // Parallel force & virial assembly across the chunk's `cnt` edges (one edge per lane < cnt)
        if (lane < cnt) {
            const int e = start + lane;
            const float4 vl = __ldg(vec_len + e);
            const float n = vl.w;
            if (n < r_max) {
                const float vx = vl.x, vy = vl.y, vz = vl.z;
                const float d = n > EPS_SH ? n : EPS_SH;
                const float inv_d = 1.0f / d;
                const float x = vx * inv_d, y = vy * inv_d, z = vz * inv_d;

                const float4* g4v = reinterpret_cast<const float4*>(s_edge[warp][lane]);
                const float4 G0 = g4v[0], G1 = g4v[1], G2 = g4v[2], G3 = g4v[3];
                const float g[16] = {G0.x, G0.y, G0.z, G0.w, G1.x, G1.y, G1.z, G1.w,
                                     G2.x, G2.y, G2.z, G2.w, G3.x, G3.y, G3.z, G3.w};
                const float g1x = g[1] * SH_C1, g1y = g[2] * SH_C1, g1z = g[3] * SH_C1;
                const float g20 = g[4] * SH_C2, g21 = g[5] * SH_C2, g22 = g[6] * SH_C2, g23 = g[7] * SH_C2, g24 = g[8] * SH_C2;
                const float g30 = g[9] * SH_C3, g31 = g[10] * SH_C3, g32 = g[11] * SH_C3, g33 = g[12] * SH_C3;
                const float g34 = g[13] * SH_C3, g35 = g[14] * SH_C3, g36 = g[15] * SH_C3;

                const float s20 = SH_S3 * x * z;
                const float s24 = (0.5f * SH_S3) * (z * z - x * x);
                const float y2 = y * y;
                const float x2z2 = x * x + z * z;
                const float a = 4.0f * y2 - x2z2;
                const float b = 2.0f * y2 - 3.0f * x2z2;

                const float gs20 = g20 + SH_S56 * (g30 * z - g36 * x) + SH_S5 * g31 * y;
                const float gs24 = g24 + SH_S56 * (g30 * x + g36 * z) + SH_S5 * g35 * y;
                const float ga = SH_S38 * (g32 * x + g34 * z);
                const float gb = 0.5f * y * g33;
                const float gy2 = g22 + 4.0f * ga + 2.0f * gb;
                const float gx2z2 = -0.5f * g22 - ga - 3.0f * gb;

                const float gx = g1x + SH_S3 * z * gs20 + SH_S3 * y * g21 - SH_S3 * x * gs24 + 2.0f * x * gx2z2
                                 + SH_S56 * (g30 * s24 - g36 * s20) + SH_S38 * a * g32;
                const float gy = g1y + SH_S3 * x * g21 + SH_S3 * z * g23 + 2.0f * y * gy2
                                 + SH_S5 * (g31 * s20 + g35 * s24) + 0.5f * b * g33;
                const float gz = g1z + SH_S3 * x * gs20 + SH_S3 * y * g23 + SH_S3 * z * gs24 + 2.0f * z * gx2z2
                                 + SH_S56 * (g30 * s20 + g36 * s24) + SH_S38 * a * g34;

                const float dot = (n > EPS_SH) ? (x * gx + y * gy + z * gz) : 0.0f;
                float zbl_dr = 0.0f;
                if (zbl_r_max_max > 0.0f && n < zbl_r_max_max) {
                    const int p = pairs[e];
                    const float rc = zbl_r_cut[p];
                    if (n > 0.0f && n < rc) {
                        const double r = (double)n;
                        const double inv_a = (double)zbl_inv_a[p];
                        const double pref = (double)zbl_pref[p];
                        const double u = r * inv_a;
                        const double e0 = exp(-3.2 * u);
                        const double e1 = exp(-0.9423 * u);
                        const double e2 = exp(-0.4028 * u);
                        const double e3 = exp(-0.2016 * u);
                        const double phi = (double)zbl_c0 * e0 + (double)zbl_c1 * e1 + (double)zbl_c2 * e2 + (double)zbl_c3 * e3;
                        const double dphi = -inv_a * (3.2 * (double)zbl_c0 * e0 + 0.9423 * (double)zbl_c1 * e1 + 0.4028 * (double)zbl_c2 * e2 + 0.2016 * (double)zbl_c3 * e3);
                        const double inv_r = 1.0 / r;
                        const double v = pref * inv_r * phi;
                        const double dv = pref * inv_r * (dphi - phi * inv_r);
                        const double zp = (double)zbl_p;
                        const double s = r / (double)rc;
                        const double sp1 = (zbl_p == 5.0f) ? (s * s * s * s) : pow(s, zp - 1.0);
                        const double sp = sp1 * s;
                        const double om = 1.0 - s;
                        const double env = 1.0 - ((zp + 1.0) * (zp + 2.0) * 0.5) * sp
                                               + (zp * (zp + 2.0)) * (sp * s)
                                               - (zp * (zp + 1.0) * 0.5) * (sp * s * s);
                        const double denv = -(zp * (zp + 1.0) * (zp + 2.0) * 0.5) * sp1 * (om * om) / (double)rc;
                        e_zbl_thread += v * env;
                        zbl_dr = (float)(dv * env + v * denv);
                    }
                }
                const float r_grad = ((n > EPS_SH) ? g[0] : 0.0f) + zbl_dr;
                const float fx = x * r_grad + (gx - x * dot) * inv_d + edge_force_1[e * 3 + 0];
                const float fy = y * r_grad + (gy - y * dot) * inv_d + edge_force_1[e * 3 + 1];
                const float fz = z * r_grad + (gz - z * dot) * inv_d + edge_force_1[e * 3 + 2];

                const int snd = senders[e];
                const int rcv = receivers[e];
                atomicAdd(forces_f64 + rcv * 3 + 0, -(double)fx);
                atomicAdd(forces_f64 + rcv * 3 + 1, -(double)fy);
                atomicAdd(forces_f64 + rcv * 3 + 2, -(double)fz);
                atomicAdd(forces_f64 + snd * 3 + 0,  (double)fx);
                atomicAdd(forces_f64 + snd * 3 + 1,  (double)fy);
                atomicAdd(forces_f64 + snd * 3 + 2,  (double)fz);

                if (compute_stress) {
                    v00 = (double)vx * (double)fx;
                    v01 = 0.5 * ((double)vx * (double)fy + (double)vy * (double)fx);
                    v02 = 0.5 * ((double)vx * (double)fz + (double)vz * (double)fx);
                    v11 = (double)vy * (double)fy;
                    v12 = 0.5 * ((double)vy * (double)fz + (double)vz * (double)fy);
                    v22 = (double)vz * (double)fz;
                }
            }
        }
    }  // if (start < E)

    if (zbl_r_max_max > 0.0f && __any_sync(0xffffffffu, e_zbl_thread != 0.0)) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) e_zbl_thread += __shfl_xor_sync(0xffffffffu, e_zbl_thread, off);
        if (lane == 0) atomicAdd(energy_f64, e_zbl_thread);
    }

    if (compute_stress) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v00 += __shfl_xor_sync(0xffffffffu, v00, off);
            v01 += __shfl_xor_sync(0xffffffffu, v01, off);
            v02 += __shfl_xor_sync(0xffffffffu, v02, off);
            v11 += __shfl_xor_sync(0xffffffffu, v11, off);
            v12 += __shfl_xor_sync(0xffffffffu, v12, off);
            v22 += __shfl_xor_sync(0xffffffffu, v22, off);
        }
        __shared__ double s_v[4][6];  // 4 warps per block
        if (lane == 0) {
            s_v[warp][0] = v00; s_v[warp][1] = v01; s_v[warp][2] = v02;
            s_v[warp][3] = v11; s_v[warp][4] = v12; s_v[warp][5] = v22;
        }
        __syncthreads();
        if (tid < 6) {
            double sum = 0.0;
            #pragma unroll
            for (int w = 0; w < 4; ++w) sum += s_v[w][tid];
            atomicAdd(virial_grad_f64 + tid, sum);
        }
    }
}

}  // namespace loup
