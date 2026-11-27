// Optimized H2 (__half2 / __nv_bfloat162) edge kernels (Wave 13B/13C/13D):
//   - LST_EDGE_H2 bitmask (default 5): 1=conv_h2, 2=slot_h2 (opt-in), 4=force_h2 (1 warp = 64 ch!)
//   - LST_SLOT_OPT (default 1): 0=legacy k_slot_xd_sp, 1=2-warp k_slot_xd_opt (packed gn2/an2 + streaming l + rptr0),
//                               2=k_slot_xd_opt + prev_c[4] register cache.
// Wave 13D fixes from fresh bd46c nsys+ncu:
//   - k_conv_pair_h2: restored blockDim=256 + __launch_bounds__(256, 2) + 4x parallel Ck4 loads upfront (fixes +54 us
//     MLP serialization regression from bd46c) while factoring xs/xb out of tx_s/rx_s/bx_s (-24 regs -> <=128 regs,
//     0 spills, 2 blocks/SM on BOTH sm_89 and sm_120!).
//   - k_force_pair_h2: __launch_bounds__(128, 4) (fixes sm_89 161 -> <=128 regs, 2x occ on Ada, 0 spills!) +
//     1-ahead pipelined Ck[l+1] prefetch inside each layer (hides Ck latency with only 4 extra regs).
#pragma once
#include "act_half.cuh"
#include "mlp_spline.cuh"
#include "sh16_pair.cuh"

namespace lst {
namespace eh2 {

__device__ __forceinline__ int lofm_h2(int m) { return (m >= 1) + (m >= 4) + (m >= 9); }

__device__ __forceinline__ void eval_c4_pair(uint4 ck, float u, float& rx, float& ry) {
    const float2 ax = __half22float2(*reinterpret_cast<const __half2*>(&ck.x));
    const float2 bx = __half22float2(*reinterpret_cast<const __half2*>(&ck.y));
    const float2 ay = __half22float2(*reinterpret_cast<const __half2*>(&ck.z));
    const float2 by = __half22float2(*reinterpret_cast<const __half2*>(&ck.w));
    rx = ((bx.y * u + bx.x) * u + ax.y) * u + ax.x;
    ry = ((by.y * u + by.x) * u + ay.y) * u + ay.x;
}

__device__ __forceinline__ void eval_c4_pair_rt(uint4 ck, float u, float invh, float& rx, float& ry, float& tx,
                                                float& ty) {
    const float2 ax = __half22float2(*reinterpret_cast<const __half2*>(&ck.x));
    const float2 bx = __half22float2(*reinterpret_cast<const __half2*>(&ck.y));
    const float2 ay = __half22float2(*reinterpret_cast<const __half2*>(&ck.z));
    const float2 by = __half22float2(*reinterpret_cast<const __half2*>(&ck.w));
    rx = ((bx.y * u + bx.x) * u + ax.y) * u + ax.x;
    tx = ((3.f * bx.y * u + 2.f * bx.x) * u + ax.y) * invh;
    ry = ((by.y * u + by.x) * u + ay.y) * u + ay.x;
    ty = ((3.f * by.y * u + 2.f * by.x) * u + ay.y) * invh;
}

__device__ __forceinline__ float eval_c4_s(uint2 ck, float u) {
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&ck.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&ck.y));
    return ((b.y * u + b.x) * u + a.y) * u + a.x;
}

__device__ __forceinline__ void eval_c4_s_rt(uint2 ck, float u, float invh, float& r, float& t) {
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&ck.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&ck.y));
    r = ((b.y * u + b.x) * u + a.y) * u + a.x;
    t = ((3.f * b.y * u + 2.f * b.x) * u + a.y) * invh;
}

// ============================================================================ 1b. k_conv_c64 (Wave 24, C == 64, POLY == 2)
// 1 ch/thread (64 threads/node, 128 threads/blk = 2 nodes/blk), ZERO smem (full 128 KB L1!), ZERO barriers!
//   - Cuts og[16] + gn2[8] register footprint in HALF vs k_conv_pair_h2 -> <=64-80 regs (50%-67% occ vs 29%!).
//   - Caches prev_ck[4] (8 regs) across unified (n, kidx)-sorted emeta_dir -> ~85% fewer C_kh loads!
//   - T0S_DIR=true (Layer 0, S <= 4): reads X0 from 4 registers T0[0..3] (0 X0 loads!) and accumulates Xd0 into
//     4 registers t0s0..3 -> ZERO Xd0 atomics (5.4M -> 0!), flushing T0s[0..S-1] ONCE per node!
template <typename H, bool DG, bool DX, int NT, bool XB, int FACT = 0, bool SHELL = false, bool T0S_DIR = false>
__global__ void __launch_bounds__(128, 6) k_conv_c64(
    const int* __restrict__ rptr, const int4* __restrict__ emeta_dir,
    const float* __restrict__ X, const H* __restrict__ G, const float* __restrict__ C_kh,
    const float* __restrict__ Y0, const float* __restrict__ Y1, const float* __restrict__ Y2,
    const float* __restrict__ Xb, H* __restrict__ OG, float* __restrict__ OX,
    float sp_invh, int N, const int* __restrict__ norder = nullptr, const float* __restrict__ Y3P = nullptr,
    int kcut2 = 999999, int kcut3 = 999999,
    const int* __restrict__ spc = nullptr, const float* __restrict__ T0 = nullptr,
    float* __restrict__ T0s = nullptr, int S = 0) {
    const int c = threadIdx.x & 63;
    const int nib = threadIdx.x >> 6;
    const int nn = blockIdx.x * 2 + nib;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;

    const int e_beg = rptr[n];
    const int e_end = rptr[n + 1];

    float og[16];
    if constexpr (DG) {
#pragma unroll
        for (int m = 0; m < 16; ++m) og[m] = 0.f;
    }
    uint32_t gn2[8];
    if constexpr (DX) {
        const long nbase = (long)n * 16 * 64 + c;
#pragma unroll
        for (int k = 0; k < 8; ++k)
            gn2[k] = ah::pack2_raw<H>(G[nbase + (long)(2 * k) * 64], G[nbase + (long)(2 * k + 1) * 64]);
    }
    float t0r0 = 0.f, t0r1 = 0.f, t0r2 = 0.f, t0r3 = 0.f;
    float t0s0 = 0.f, t0s1 = 0.f, t0s2 = 0.f, t0s3 = 0.f;
    if constexpr (T0S_DIR) {
        if constexpr (DG) {
            t0r0 = T0[0 * 64 + c];
            if (S > 1) t0r1 = T0[1 * 64 + c];
            if (S > 2) t0r2 = T0[2 * 64 + c];
            if (S > 3) t0r3 = T0[3 * 64 + c];
        }
    }

    const uint2* Ck2 = reinterpret_cast<const uint2*>(C_kh);
    int prev_idx = -1;
    uint2 prev_c[4];

    for (int e = e_beg; e < e_end; ++e) {
        const int4 em = emeta_dir[e];
        const int idx = em.z;
        if (idx < 0) continue;
        const int ps_e = em.x;
        const long p = ps_e >> 1;
        const long s = em.y;
        const float u = __int_as_float(em.w);
        const float sgn = (ps_e & 1) ? -1.f : 1.f;
        const bool do_l2 = !SHELL || (idx < kcut2);
        const bool do_l3 = !SHELL || (idx < kcut3);

        if (idx != prev_idx) {
            const long cbase = (long)idx * 4 * 64 + c;
            prev_c[0] = Ck2[cbase + 0 * 64];
            prev_c[1] = Ck2[cbase + 1 * 64];
            if (do_l2) prev_c[2] = Ck2[cbase + 2 * 64];
            if (do_l3) prev_c[3] = Ck2[cbase + 3 * 64];
            prev_idx = idx;
        }
        int sp_s = 0;
        float xs = 0.f, xb = 0.f;
        if constexpr (T0S_DIR) {
            sp_s = spc[s];
            if constexpr (DG) xs = (sp_s == 0) ? t0r0 : ((sp_s == 1) ? t0r1 : ((sp_s == 2) ? t0r2 : t0r3));
        } else if constexpr (DG) {
            xs = X[s * 64 + c];
            if constexpr (XB) xb = Xb[s * 64 + c];
        }

        if constexpr (NT == 1) {
            float r0 = eval_c4_s(prev_c[0], u);
            float r1 = eval_c4_s(prev_c[1], u);
            float r2 = do_l2 ? eval_c4_s(prev_c[2], u) : 0.f;
            float r3 = do_l3 ? eval_c4_s(prev_c[3], u) : 0.f;
            const float r0x = DG ? (r0 * xs) : 0.f;
            const float r1x = DG ? (sgn * r1 * xs) : 0.f;
            const float r2x = (DG && do_l2) ? (r2 * xs) : 0.f;
            const float r3x = (DG && do_l3) ? (sgn * r3 * xs) : 0.f;
            float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
            {
                const float4 vy0 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 0);
                if constexpr (DG) {
                    og[0] += r0x * vy0.x;
                    og[1] += r1x * vy0.y; og[2] += r1x * vy0.z; og[3] += r1x * vy0.w;
                }
                if constexpr (DX) {
                    const float2 g01 = ah::unpack2<H>(gn2[0]), g23 = ah::unpack2<H>(gn2[1]);
                    q0 += g01.x * vy0.x;
                    q1 += g01.y * vy0.y + g23.x * vy0.z + g23.y * vy0.w;
                }
            }
            if (do_l2 || do_l3) {
                const float4 vy2 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 8);
                if (do_l2) {
                    const float4 vy1 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 4);
                    if constexpr (DG) {
                        og[4] += r2x * vy1.x; og[5] += r2x * vy1.y; og[6] += r2x * vy1.z; og[7] += r2x * vy1.w;
                        og[8] += r2x * vy2.x;
                    }
                    if constexpr (DX) {
                        const float2 g45 = ah::unpack2<H>(gn2[2]), g67 = ah::unpack2<H>(gn2[3]), g89 = ah::unpack2<H>(gn2[4]);
                        q2 += g45.x * vy1.x + g45.y * vy1.y + g67.x * vy1.z + g67.y * vy1.w + g89.x * vy2.x;
                    }
                }
                if (do_l3) {
                    const float4 vy3 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 12);
                    if constexpr (DG) {
                        og[9] += r3x * vy2.y; og[10] += r3x * vy2.z; og[11] += r3x * vy2.w;
                        og[12] += r3x * vy3.x; og[13] += r3x * vy3.y; og[14] += r3x * vy3.z; og[15] += r3x * vy3.w;
                    }
                    if constexpr (DX) {
                        const float2 g89 = ah::unpack2<H>(gn2[4]), gAB = ah::unpack2<H>(gn2[5]);
                        const float2 gCD = ah::unpack2<H>(gn2[6]), gEF = ah::unpack2<H>(gn2[7]);
                        q3 += g89.y * vy2.y + gAB.x * vy2.z + gAB.y * vy2.w
                            + gCD.x * vy3.x + gCD.y * vy3.y + gEF.x * vy3.z + gEF.y * vy3.w;
                    }
                }
            }
            if constexpr (DX) {
                float ox = r0 * q0 + sgn * r1 * q1;
                if (do_l2) ox += r2 * q2;
                if (do_l3) ox += sgn * r3 * q3;
                if constexpr (T0S_DIR) {
                    if (sp_s == 0) t0s0 += ox;
                    else if (sp_s == 1) t0s1 += ox;
                    else if (sp_s == 2) t0s2 += ox;
                    else t0s3 += ox;
                } else {
                    atomicAdd(OX + s * 64 + c, ox);
                }
            }
        } else {
            // NT == 2 (conv_rev2), factored 1-l-at-a-time with prev_c[4] cached!
            auto load_y4 = [&](int q, float4& v0, float4& v1, float4& v2) {
                if constexpr (FACT == 3) {
                    const uint4* yhp = reinterpret_cast<const uint4*>(Y3P) + 6L * p;
                    const uint4 qh = yhp[q];
                    const float2 w01 = __half22float2(*reinterpret_cast<const __half2*>(&qh.x));
                    const float2 w23 = __half22float2(*reinterpret_cast<const __half2*>(&qh.y));
                    const float2 g01 = __half22float2(*reinterpret_cast<const __half2*>(&qh.z));
                    const float2 g23 = __half22float2(*reinterpret_cast<const __half2*>(&qh.w));
                    v0 = make_float4(w01.x, w01.y, w23.x, w23.y);
                    v1 = make_float4(g01.x, g01.y, g23.x, g23.y);
                    if constexpr (DG && XB) {
                        const uint2 yy = *(reinterpret_cast<const uint2*>(yhp + 4) + q);
                        const float2 y01 = __half22float2(*reinterpret_cast<const __half2*>(&yy.x));
                        const float2 y23 = __half22float2(*reinterpret_cast<const __half2*>(&yy.y));
                        v2 = make_float4(y01.x, y01.y, y23.x, y23.y);
                    } else {
                        v2 = make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                } else if constexpr (FACT == 2) {
                    const float* yp = Y3P + 48L * p + 12 * q;
                    v0 = *reinterpret_cast<const float4*>(yp + 0);
                    v1 = *reinterpret_cast<const float4*>(yp + 4);
                    if constexpr (DG && XB) v2 = *reinterpret_cast<const float4*>(yp + 8);
                    else v2 = make_float4(0.f, 0.f, 0.f, 0.f);
                } else {
                    v0 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 4 * q);
                    v1 = *reinterpret_cast<const float4*>(Y1 + 16L * p + 4 * q);
                    if constexpr (DG && XB) v2 = *reinterpret_cast<const float4*>(Y2 + 16L * p + 4 * q);
                    else v2 = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            };
            float ox = 0.f;
            auto do_m = [&](int m, float gval, float y0v, float y1v, float y2v,
                            float txs, float rxs, float rxb, float& q0, float& q1) {
                if constexpr (DG) {
                    og[m] += txs * y0v + rxs * y1v;
                    if constexpr (XB) og[m] += rxb * y2v;
                }
                if constexpr (DX) {
                    q0 += gval * y0v;
                    q1 += gval * y1v;
                }
            };
            {
                float4 v0, v1, v2;
                load_y4(0, v0, v1, v2);
                const float2 g01 = DX ? ah::unpack2<H>(gn2[0]) : make_float2(0.f, 0.f);
                const float2 g23 = DX ? ah::unpack2<H>(gn2[1]) : make_float2(0.f, 0.f);
                float r, t;
                eval_c4_s_rt(prev_c[0], u, sp_invh, r, t);
                {
                    float q0 = 0.f, q1 = 0.f;
                    do_m(0, g01.x, v0.x, v1.x, v2.x, t * xs, r * xs, XB ? r * xb : 0.f, q0, q1);
                    if constexpr (DX) ox += t * q0 + r * q1;
                }
                eval_c4_s_rt(prev_c[1], u, sp_invh, r, t);
                r *= sgn; t *= sgn;
                {
                    const float txs = t * xs, rxs = r * xs, rxb = XB ? r * xb : 0.f;
                    float q0 = 0.f, q1 = 0.f;
                    do_m(1, g01.y, v0.y, v1.y, v2.y, txs, rxs, rxb, q0, q1);
                    do_m(2, g23.x, v0.z, v1.z, v2.z, txs, rxs, rxb, q0, q1);
                    do_m(3, g23.y, v0.w, v1.w, v2.w, txs, rxs, rxb, q0, q1);
                    if constexpr (DX) ox += t * q0 + r * q1;
                }
            }
            if (do_l2 || do_l3) {
                float4 v0b, v1b, v2b;
                load_y4(2, v0b, v1b, v2b);
                const float2 g89 = DX ? ah::unpack2<H>(gn2[4]) : make_float2(0.f, 0.f);
                if (do_l2) {
                    float r, t;
                    eval_c4_s_rt(prev_c[2], u, sp_invh, r, t);
                    const float txs = t * xs, rxs = r * xs, rxb = XB ? r * xb : 0.f;
                    float q0 = 0.f, q1 = 0.f;
                    {
                        float4 v0, v1, v2;
                        load_y4(1, v0, v1, v2);
                        const float2 g45 = DX ? ah::unpack2<H>(gn2[2]) : make_float2(0.f, 0.f);
                        const float2 g67 = DX ? ah::unpack2<H>(gn2[3]) : make_float2(0.f, 0.f);
                        do_m(4, g45.x, v0.x, v1.x, v2.x, txs, rxs, rxb, q0, q1);
                        do_m(5, g45.y, v0.y, v1.y, v2.y, txs, rxs, rxb, q0, q1);
                        do_m(6, g67.x, v0.z, v1.z, v2.z, txs, rxs, rxb, q0, q1);
                        do_m(7, g67.y, v0.w, v1.w, v2.w, txs, rxs, rxb, q0, q1);
                    }
                    do_m(8, g89.x, v0b.x, v1b.x, v2b.x, txs, rxs, rxb, q0, q1);
                    if constexpr (DX) ox += t * q0 + r * q1;
                }
                if (do_l3) {
                    float r, t;
                    eval_c4_s_rt(prev_c[3], u, sp_invh, r, t);
                    r *= sgn; t *= sgn;
                    const float txs = t * xs, rxs = r * xs, rxb = XB ? r * xb : 0.f;
                    float q0 = 0.f, q1 = 0.f;
                    const float2 gAB = DX ? ah::unpack2<H>(gn2[5]) : make_float2(0.f, 0.f);
                    do_m(9, g89.y, v0b.y, v1b.y, v2b.y, txs, rxs, rxb, q0, q1);
                    do_m(10, gAB.x, v0b.z, v1b.z, v2b.z, txs, rxs, rxb, q0, q1);
                    do_m(11, gAB.y, v0b.w, v1b.w, v2b.w, txs, rxs, rxb, q0, q1);
                    {
                        float4 v0c, v1c, v2c;
                        load_y4(3, v0c, v1c, v2c);
                        const float2 gCD = DX ? ah::unpack2<H>(gn2[6]) : make_float2(0.f, 0.f);
                        const float2 gEF = DX ? ah::unpack2<H>(gn2[7]) : make_float2(0.f, 0.f);
                        do_m(12, gCD.x, v0c.x, v1c.x, v2c.x, txs, rxs, rxb, q0, q1);
                        do_m(13, gCD.y, v0c.y, v1c.y, v2c.y, txs, rxs, rxb, q0, q1);
                        do_m(14, gEF.x, v0c.z, v1c.z, v2c.z, txs, rxs, rxb, q0, q1);
                        do_m(15, gEF.y, v0c.w, v1c.w, v2c.w, txs, rxs, rxb, q0, q1);
                    }
                    if constexpr (DX) ox += t * q0 + r * q1;
                }
            }
            if constexpr (DX) {
                if constexpr (T0S_DIR) {
                    if (sp_s == 0) t0s0 += ox;
                    else if (sp_s == 1) t0s1 += ox;
                    else if (sp_s == 2) t0s2 += ox;
                    else t0s3 += ox;
                } else {
                    atomicAdd(OX + s * 64 + c, ox);
                }
            }
        }
    }
    if constexpr (T0S_DIR && DX) {
        if (t0s0 != 0.f) atomicAdd(T0s + 0 * 64 + c, t0s0);
        if (S > 1 && t0s1 != 0.f) atomicAdd(T0s + 1 * 64 + c, t0s1);
        if (S > 2 && t0s2 != 0.f) atomicAdd(T0s + 2 * 64 + c, t0s2);
        if (S > 3 && t0s3 != 0.f) atomicAdd(T0s + 3 * 64 + c, t0s3);
    }
    if constexpr (DG) {
        const long obase = (long)n * 16 * 64 + c;
#pragma unroll
        for (int m = 0; m < 16; ++m) ah::st_a16<H>(OG, obase + (long)m * 64, og[m]);
    }
}

// ============================================================================ 1c. k_conv_pair_h2_opt (Wave 25: LST_CONV_OPT=1!)
// 1 warp/node (2 ch/thread, 4 nodes/block = 128 threads), unified (n, kidx)-sorted emeta_dir:
//   - s_ck[4][4][32] (8 KB smem) caches Ck4 across prev_idx -> ~85% fewer Ck4 L2 reads, 0 extra regs!
//   - When DX=true (conv_rev2): s_gn[4][16][32] (8 KB smem) frees gn[16] -> <=80 regs -> 50% occ (2x bo57's 29%!)
//   - When T0S_DIR=true (Layer 0, S <= 4): s_t0[4][32] (1 KB smem) -> 0 X0 loads; t0s0..3 regs -> 0 Xd0 atomics!
template <typename H, bool DG, bool DX, int NT, bool XB, int FACT = 0, bool SHELL = false, bool T0S_DIR = false>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
__global__ void __launch_bounds__(128, (DX && T0S_DIR) ? 5 : 6) k_conv_pair_h2_opt(
#else
__global__ void __launch_bounds__(128, 5) k_conv_pair_h2_opt(
#endif
    const int* __restrict__ rptr, const int4* __restrict__ emeta_dir,
    const float* __restrict__ X, const H* __restrict__ G, const float* __restrict__ C_kh,
    const float* __restrict__ Y0, const float* __restrict__ Y1, const float* __restrict__ Y2,
    const float* __restrict__ Xb, H* __restrict__ OG, float* __restrict__ OX,
    float sp_invh, int N, const int* __restrict__ norder = nullptr, const float* __restrict__ Y3P = nullptr,
    int kcut2 = 999999, int kcut3 = 999999,
    const int* __restrict__ spc = nullptr, const float* __restrict__ T0 = nullptr,
    float* __restrict__ T0s = nullptr, int S = 0) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nn = blockIdx.x * 4 + warp;

    __shared__ uint4 s_ck[4][4][32];                                  // 8 KB
    __shared__ uint32_t s_gn[DX ? 4 : 1][DX ? 16 : 1][DX ? 32 : 1];   // 8 KB when DX
    __shared__ float2 s_t0[T0S_DIR ? 4 : 1][T0S_DIR ? 32 : 1];        // 1 KB when T0S_DIR

    if constexpr (T0S_DIR) {
        if (warp == 0) {
            const float2* T02 = reinterpret_cast<const float2*>(T0);
            for (int sp = 0; sp < S; ++sp) s_t0[sp][lane] = T02[sp * 32 + lane];
        }
        __syncthreads();
    }
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;

    float ogx[16], ogy[16];
    if constexpr (DG) {
#pragma unroll
        for (int m = 0; m < 16; ++m) { ogx[m] = 0.f; ogy[m] = 0.f; }
    }
    if constexpr (DX) {
        const long nbase = (long)n * 16 * 32 + lane;
#pragma unroll
        for (int m = 0; m < 16; ++m) s_gn[warp][m][lane] = ah::ld_h2<H>(G, nbase + (long)m * 32);
    }
    float2 t0s0 = make_float2(0.f, 0.f), t0s1 = make_float2(0.f, 0.f);
    float2 t0s2 = make_float2(0.f, 0.f), t0s3 = make_float2(0.f, 0.f);

    const float2* X2 = reinterpret_cast<const float2*>(X);
    const float2* Xb2 = reinterpret_cast<const float2*>(Xb);
    const uint4* Ck4 = reinterpret_cast<const uint4*>(C_kh);

    const int e_beg = rptr[n];
    const int e_end = rptr[n + 1];
    int prev_idx = -1;

    for (int e = e_beg; e < e_end; ++e) {
        const int4 em = emeta_dir[e];
        const int idx = em.z;
        if (idx < 0) continue;
        const int ps_e = em.x;
        const long p = ps_e >> 1;
        const long s = em.y;
        const float u = __int_as_float(em.w);
        const float sgn = (ps_e & 1) ? -1.f : 1.f;
        const bool do_l2 = !SHELL || (idx < kcut2);
        const bool do_l3 = !SHELL || (idx < kcut3);

        if (idx != prev_idx) {
            const long cbase = (long)idx * 4 * 32 + lane;
            s_ck[warp][0][lane] = Ck4[cbase + 0 * 32];
            s_ck[warp][1][lane] = Ck4[cbase + 1 * 32];
            if (do_l2) s_ck[warp][2][lane] = Ck4[cbase + 2 * 32];
            if (do_l3) s_ck[warp][3][lane] = Ck4[cbase + 3 * 32];
            prev_idx = idx;
        }
        int sp_s = 0;
        float2 xs = make_float2(0.f, 0.f), xb = make_float2(0.f, 0.f);
        if constexpr (T0S_DIR) {
            sp_s = spc[s];
            if constexpr (DG) xs = s_t0[sp_s][lane];
        } else if constexpr (DG) {
            xs = X2[s * 32 + lane];
            if constexpr (XB) xb = Xb2[s * 32 + lane];
        }

        if constexpr (NT == 1) {
            float rx[4] = {0.f, 0.f, 0.f, 0.f}, ry[4] = {0.f, 0.f, 0.f, 0.f};
            eval_c4_pair(s_ck[warp][0][lane], u, rx[0], ry[0]);
            eval_c4_pair(s_ck[warp][1][lane], u, rx[1], ry[1]);
            if (do_l2) eval_c4_pair(s_ck[warp][2][lane], u, rx[2], ry[2]);
            if (do_l3) eval_c4_pair(s_ck[warp][3][lane], u, rx[3], ry[3]);
            if constexpr (DG) {
                rx[0] *= xs.x; ry[0] *= xs.y;
                rx[1] *= sgn * xs.x; ry[1] *= sgn * xs.y;
                if (do_l2) { rx[2] *= xs.x; ry[2] *= xs.y; }
                if (do_l3) { rx[3] *= sgn * xs.x; ry[3] *= sgn * xs.y; }
            }
            auto do_m = [&](int m, int l, float yv) {
                if constexpr (DG) { ogx[m] += rx[l] * yv; ogy[m] += ry[l] * yv; }
            };
            {
                const float4 vy = *reinterpret_cast<const float4*>(Y0 + 16L * p + 0);
                do_m(0, 0, vy.x); do_m(1, 1, vy.y); do_m(2, 1, vy.z); do_m(3, 1, vy.w);
            }
            if (do_l2 || do_l3) {
                const float4 vy2 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 8);
                if (do_l2) {
                    const float4 vy1 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 4);
                    do_m(4, 2, vy1.x); do_m(5, 2, vy1.y); do_m(6, 2, vy1.z); do_m(7, 2, vy1.w);
                    do_m(8, 2, vy2.x);
                }
                if (do_l3) {
                    do_m(9, 3, vy2.y); do_m(10, 3, vy2.z); do_m(11, 3, vy2.w);
                    const float4 vy3 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 12);
                    do_m(12, 3, vy3.x); do_m(13, 3, vy3.y); do_m(14, 3, vy3.z); do_m(15, 3, vy3.w);
                }
            }
        } else {
            // NT == 2 (conv_rev2): 1-l-at-a-time factored with s_ck & s_gn!
            auto load_y4 = [&](int q, float4& v0, float4& v1, float4& v2) {
                if constexpr (FACT == 3) {
                    const uint4* yhp = reinterpret_cast<const uint4*>(Y3P) + 6L * p;
                    const uint4 qh = yhp[q];
                    const float2 w01 = __half22float2(*reinterpret_cast<const __half2*>(&qh.x));
                    const float2 w23 = __half22float2(*reinterpret_cast<const __half2*>(&qh.y));
                    const float2 g01 = __half22float2(*reinterpret_cast<const __half2*>(&qh.z));
                    const float2 g23 = __half22float2(*reinterpret_cast<const __half2*>(&qh.w));
                    v0 = make_float4(w01.x, w01.y, w23.x, w23.y);
                    v1 = make_float4(g01.x, g01.y, g23.x, g23.y);
                    if constexpr (DG && XB) {
                        const uint2 yy = *(reinterpret_cast<const uint2*>(yhp + 4) + q);
                        const float2 y01 = __half22float2(*reinterpret_cast<const __half2*>(&yy.x));
                        const float2 y23 = __half22float2(*reinterpret_cast<const __half2*>(&yy.y));
                        v2 = make_float4(y01.x, y01.y, y23.x, y23.y);
                    } else {
                        v2 = make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                } else if constexpr (FACT == 2) {
                    const float* yp = Y3P + 48L * p + 12 * q;
                    v0 = *reinterpret_cast<const float4*>(yp + 0);
                    v1 = *reinterpret_cast<const float4*>(yp + 4);
                    if constexpr (DG && XB) v2 = *reinterpret_cast<const float4*>(yp + 8);
                    else v2 = make_float4(0.f, 0.f, 0.f, 0.f);
                } else {
                    v0 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 4 * q);
                    v1 = *reinterpret_cast<const float4*>(Y1 + 16L * p + 4 * q);
                    if constexpr (DG && XB) v2 = *reinterpret_cast<const float4*>(Y2 + 16L * p + 4 * q);
                    else v2 = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            };
            float ox_x = 0.f, ox_y = 0.f;
            auto do_m = [&](int m, float y0v, float y1v, float y2v, float txs_x, float txs_y, float rxs_x, float rxs_y,
                            float rxb_x, float rxb_y, float& q0x, float& q0y, float& q1x, float& q1y) {
                if constexpr (DG) {
                    ogx[m] += txs_x * y0v + rxs_x * y1v;
                    ogy[m] += txs_y * y0v + rxs_y * y1v;
                    if constexpr (XB) {
                        ogx[m] += rxb_x * y2v;
                        ogy[m] += rxb_y * y2v;
                    }
                }
                if constexpr (DX) {
                    const float2 g2 = ah::unpack2<H>(s_gn[warp][m][lane]);
                    q0x += g2.x * y0v; q0y += g2.y * y0v;
                    q1x += g2.x * y1v; q1y += g2.y * y1v;
                }
            };
            {
                float4 v0, v1, v2;
                load_y4(0, v0, v1, v2);
                float rx, ry, tx, ty;
                eval_c4_pair_rt(s_ck[warp][0][lane], u, sp_invh, rx, ry, tx, ty);
                {
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    do_m(0, v0.x, v1.x, v2.x, tx * xs.x, ty * xs.y, rx * xs.x, ry * xs.y,
                         XB ? rx * xb.x : 0.f, XB ? ry * xb.y : 0.f, q0x, q0y, q1x, q1y);
                    if constexpr (DX) { ox_x += tx * q0x + rx * q1x; ox_y += ty * q0y + ry * q1y; }
                }
                eval_c4_pair_rt(s_ck[warp][1][lane], u, sp_invh, rx, ry, tx, ty);
                rx *= sgn; ry *= sgn; tx *= sgn; ty *= sgn;
                {
                    const float txs_x = tx * xs.x, txs_y = ty * xs.y;
                    const float rxs_x = rx * xs.x, rxs_y = ry * xs.y;
                    const float rxb_x = XB ? rx * xb.x : 0.f, rxb_y = XB ? ry * xb.y : 0.f;
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    do_m(1, v0.y, v1.y, v2.y, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                    do_m(2, v0.z, v1.z, v2.z, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                    do_m(3, v0.w, v1.w, v2.w, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                    if constexpr (DX) { ox_x += tx * q0x + rx * q1x; ox_y += ty * q0y + ry * q1y; }
                }
            }
            if (do_l2 || do_l3) {
                float4 v0b, v1b, v2b;
                load_y4(2, v0b, v1b, v2b);
                if (do_l2) {
                    float rx, ry, tx, ty;
                    eval_c4_pair_rt(s_ck[warp][2][lane], u, sp_invh, rx, ry, tx, ty);
                    const float txs_x = tx * xs.x, txs_y = ty * xs.y;
                    const float rxs_x = rx * xs.x, rxs_y = ry * xs.y;
                    const float rxb_x = XB ? rx * xb.x : 0.f, rxb_y = XB ? ry * xb.y : 0.f;
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    {
                        float4 v0, v1, v2;
                        load_y4(1, v0, v1, v2);
                        do_m(4, v0.x, v1.x, v2.x, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                        do_m(5, v0.y, v1.y, v2.y, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                        do_m(6, v0.z, v1.z, v2.z, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                        do_m(7, v0.w, v1.w, v2.w, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                    }
                    do_m(8, v0b.x, v1b.x, v2b.x, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                    if constexpr (DX) { ox_x += tx * q0x + rx * q1x; ox_y += ty * q0y + ry * q1y; }
                }
                if (do_l3) {
                    float rx, ry, tx, ty;
                    eval_c4_pair_rt(s_ck[warp][3][lane], u, sp_invh, rx, ry, tx, ty);
                    rx *= sgn; ry *= sgn; tx *= sgn; ty *= sgn;
                    const float txs3_x = tx * xs.x, txs3_y = ty * xs.y;
                    const float rxs3_x = rx * xs.x, rxs3_y = ry * xs.y;
                    const float rxb3_x = XB ? rx * xb.x : 0.f, rxb3_y = XB ? ry * xb.y : 0.f;
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    do_m(9, v0b.y, v1b.y, v2b.y, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                    do_m(10, v0b.z, v1b.z, v2b.z, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                    do_m(11, v0b.w, v1b.w, v2b.w, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                    {
                        float4 v0c, v1c, v2c;
                        load_y4(3, v0c, v1c, v2c);
                        do_m(12, v0c.x, v1c.x, v2c.x, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                        do_m(13, v0c.y, v1c.y, v2c.y, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                        do_m(14, v0c.z, v1c.z, v2c.z, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                        do_m(15, v0c.w, v1c.w, v2c.w, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                    }
                    if constexpr (DX) { ox_x += tx * q0x + rx * q1x; ox_y += ty * q0y + ry * q1y; }
                }
            }
            if constexpr (DX) {
                if constexpr (T0S_DIR) {
                    if (sp_s == 0) { t0s0.x += ox_x; t0s0.y += ox_y; }
                    else if (sp_s == 1) { t0s1.x += ox_x; t0s1.y += ox_y; }
                    else if (sp_s == 2) { t0s2.x += ox_x; t0s2.y += ox_y; }
                    else { t0s3.x += ox_x; t0s3.y += ox_y; }
                } else {
                    atomicAdd(OX + s * 64 + 2 * lane, ox_x);
                    atomicAdd(OX + s * 64 + 2 * lane + 1, ox_y);
                }
            }
        }
    }
    if constexpr (T0S_DIR && DX) {
        if (t0s0.x != 0.f) atomicAdd(T0s + 0 * 64 + 2 * lane, t0s0.x);
        if (t0s0.y != 0.f) atomicAdd(T0s + 0 * 64 + 2 * lane + 1, t0s0.y);
        if (S > 1) {
            if (t0s1.x != 0.f) atomicAdd(T0s + 1 * 64 + 2 * lane, t0s1.x);
            if (t0s1.y != 0.f) atomicAdd(T0s + 1 * 64 + 2 * lane + 1, t0s1.y);
        }
        if (S > 2) {
            if (t0s2.x != 0.f) atomicAdd(T0s + 2 * 64 + 2 * lane, t0s2.x);
            if (t0s2.y != 0.f) atomicAdd(T0s + 2 * 64 + 2 * lane + 1, t0s2.y);
        }
        if (S > 3) {
            if (t0s3.x != 0.f) atomicAdd(T0s + 3 * 64 + 2 * lane, t0s3.x);
            if (t0s3.y != 0.f) atomicAdd(T0s + 3 * 64 + 2 * lane + 1, t0s3.y);
        }
    }
    if constexpr (DG) {
        const long obase = (long)n * 16 * 32 + lane;
#pragma unroll
        for (int m = 0; m < 16; ++m) ah::st_h2<H>(OG, obase + (long)m * 32, ah::pack2<H>(ogx[m], ogy[m]));
    }
}

// ============================================================================ 1. k_conv_pair_h2 (C == 64, POLY == 2)
// Wave 43: template int NPB = (ES == 1 ? 4 : 8 / ES) supports NPB == 1 (1 atom/CTA: B=32*ES, G=N=357)
// eliminating 50% tail-wave idle on G=90 (36/48 SMs) and 4-atom intra-CTA barrier stragglers!
template <typename H, bool DG, bool DX, int NT, bool XB, int ES, bool SDX, bool EMETA, int FACT = 0,
          bool SHELL = false, bool T0S_DIR = false, bool SCK = false, int NPB = (ES == 1 ? 4 : 8 / ES)>
__global__ void __launch_bounds__(32 * NPB * ES,
                                  (NPB == 1 && ES == 1)
                                      ? 16
                                      : (((((FACT == 5 || FACT == 6) && std::is_same<H, __half>::value) ||
                                           (DG && !DX && ES == 4))
                                              ? (ES == 1 ? 20 : 24)
                                              : 16) /
                                         (NPB * ES))) k_conv_pair_h2(
    const int* __restrict__ rptr, const int* __restrict__ snd, const int* __restrict__ ps,
    const int* __restrict__ sp_idx, const float* __restrict__ sp_u, const float* __restrict__ X,
    const H* __restrict__ G, const float* __restrict__ C_kh, const float* __restrict__ Y0,
    const float* __restrict__ Y1, const float* __restrict__ Y2, const float* __restrict__ Xb,
    H* __restrict__ OG, float* __restrict__ OX, float sp_invh, int N, const int4* __restrict__ emeta,
    const int* __restrict__ norder = nullptr, const float* __restrict__ Y3P = nullptr,
    int kcut2 = 999999, int kcut3 = 999999,
    const int* __restrict__ spc = nullptr, const float* __restrict__ T0 = nullptr,
    float* __restrict__ T0s = nullptr, int S = 0) {
    const int lane = threadIdx.x & 31;
    const int wblk = threadIdx.x >> 5;
    const int grp = (ES > 1) ? (wblk % ES) : 0;
    const int node_in_blk = (NPB > 1) ? (wblk / ES) : 0;
    const int nn = blockIdx.x * NPB + node_in_blk;
    const int n = (norder && nn < N) ? norder[nn] : nn;

    constexpr int NW_BLK = NPB * ES;
    constexpr bool OG_H2 = (FACT == 5 || FACT == 6) && std::is_same<H, __half>::value;
    __shared__ float2 s_t0[T0S_DIR ? 4 : 1][T0S_DIR ? 32 : 1];
    __shared__ float2 s_t0s[(T0S_DIR && DX && SDX) ? NW_BLK : 1][(T0S_DIR && DX && SDX) ? 4 : 1]
                           [(T0S_DIR && DX && SDX) ? 32 : 1];
    __shared__ uint4 s_ck[(SCK && FACT >= 1) ? NW_BLK : 1][(SCK && FACT >= 1) ? 4 : 1][(SCK && FACT >= 1) ? 32 : 1];
    if constexpr (T0S_DIR) {
        if (wblk == 0) {
            const float2* T02 = reinterpret_cast<const float2*>(T0);
            for (int sp = 0; sp < S; ++sp) s_t0[sp][lane] = T02[sp * 32 + lane];
        }
        if constexpr (DX && SDX) {
            for (int sp = 0; sp < 4; ++sp) s_t0s[wblk][sp][lane] = make_float2(0.f, 0.f);
        }
        if constexpr (NW_BLK > 1) __syncthreads();
    }
    if ((ES == 1 || (SDX && !DG)) && n >= N) return;

    float ogx[OG_H2 ? 1 : 16], ogy[OG_H2 ? 1 : 16];
    __half2 og_h[OG_H2 ? 16 : 1];
    if constexpr (DG) {
        if constexpr (OG_H2) {
            const __half2 zh = __float2half2_rn(0.f);
#pragma unroll
            for (int m = 0; m < 16; ++m) og_h[m] = zh;
        } else {
#pragma unroll
            for (int m = 0; m < 16; ++m) { ogx[m] = 0.f; ogy[m] = 0.f; }
        }
    }
    uint32_t gn[16];
    if constexpr (SDX && DX) {
        if (n < N) {
            const long nbase = (long)n * 16 * 32 + lane;
#pragma unroll
            for (int m = 0; m < 16; ++m) gn[m] = ah::ld_h2<H>(G, nbase + (long)m * 32);
        }
    }
    const float2* X2 = reinterpret_cast<const float2*>(X);
    const float2* Xb2 = reinterpret_cast<const float2*>(Xb);
    const uint4* Ck4 = reinterpret_cast<const uint4*>(C_kh);

    const int e0_n = n < N ? rptr[n] : 0;
    const int e1_n = n < N ? rptr[n + 1] : 0;
    const int cnt_n = e1_n - e0_n;
    const int e_beg = EMETA ? (ES > 1 ? e0_n + (cnt_n * grp) / ES : e0_n) : (e0_n + grp);
    const int e_end = EMETA ? (ES > 1 ? e0_n + (cnt_n * (grp + 1)) / ES : e1_n) : e1_n;
    const int e_step = EMETA ? 1 : ES;

    int prev_idx = -1;
    constexpr bool USE_PC = (NT == 1 && EMETA && ES > 1);
    constexpr int PC_N = USE_PC ? 4 : 1;
    uint4 prev_c[PC_N];

    for (int e = e_beg; e < e_end; e += e_step) {
        int ps_e, idx;
        long p, s;
        float u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            p = ps_e >> 1;
            idx = em.z;
            if (idx < 0) continue;
            s = em.y;
            u = __int_as_float(em.w);
        } else {
            ps_e = ps[e];
            p = ps_e >> 1;
            idx = sp_idx[p];
            if (idx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        const float sgn = (ps_e & 1) ? -1.f : 1.f;
        const int sp_s = T0S_DIR ? spc[s] : 0;
        float2 xs = make_float2(0.f, 0.f), xb = make_float2(0.f, 0.f);
        if constexpr (DG) {
            xs = T0S_DIR ? s_t0[sp_s][lane] : X2[s * 32 + lane];
            if constexpr (XB) xb = Xb2[s * 32 + lane];
        }
        const long cbase = (long)idx * 4 * 32 + lane;
        if constexpr (NT == 1) {
            const bool do_l2 = !SHELL || (idx < kcut2);
            const bool do_l3 = !SHELL || (idx < kcut3);
            float rx[4] = {0.f, 0.f, 0.f, 0.f}, ry[4] = {0.f, 0.f, 0.f, 0.f};
            if constexpr (USE_PC) {
                if (idx != prev_idx) {
                    prev_c[0] = Ck4[cbase + 0 * 32];
                    prev_c[1] = Ck4[cbase + 1 * 32];
                    if (do_l2) prev_c[2] = Ck4[cbase + 2 * 32];
                    if (do_l3) prev_c[3] = Ck4[cbase + 3 * 32];
                    prev_idx = idx;
                }
                eval_c4_pair(prev_c[0], u, rx[0], ry[0]);
                eval_c4_pair(prev_c[1], u, rx[1], ry[1]);
                if (do_l2) eval_c4_pair(prev_c[2], u, rx[2], ry[2]);
                if (do_l3) eval_c4_pair(prev_c[3], u, rx[3], ry[3]);
            } else {
                eval_c4_pair(Ck4[cbase + 0 * 32], u, rx[0], ry[0]);
                eval_c4_pair(Ck4[cbase + 1 * 32], u, rx[1], ry[1]);
                if (do_l2) eval_c4_pair(Ck4[cbase + 2 * 32], u, rx[2], ry[2]);
                if (do_l3) eval_c4_pair(Ck4[cbase + 3 * 32], u, rx[3], ry[3]);
            }
            if constexpr (DG) {
                rx[0] *= xs.x; ry[0] *= xs.y;
                rx[1] *= sgn * xs.x; ry[1] *= sgn * xs.y;
                if (do_l2) { rx[2] *= xs.x; ry[2] *= xs.y; }
                if (do_l3) { rx[3] *= sgn * xs.x; ry[3] *= sgn * xs.y; }
            }
            float qx[4] = {0.f, 0.f, 0.f, 0.f}, qy[4] = {0.f, 0.f, 0.f, 0.f};
            auto do_nt1_m = [&](int m, int l, float yv) {
                if constexpr (DG) { ogx[m] += rx[l] * yv; ogy[m] += ry[l] * yv; }
                if constexpr (DX && SDX) {
                    const float2 g2 = ah::unpack2<H>(gn[m]);
                    qx[l] += g2.x * yv; qy[l] += g2.y * yv;
                }
            };
            {
                const float4 vy = *reinterpret_cast<const float4*>(Y0 + 16L * p + 0);
                do_nt1_m(0, 0, vy.x); do_nt1_m(1, 1, vy.y); do_nt1_m(2, 1, vy.z); do_nt1_m(3, 1, vy.w);
            }
            if (do_l2 || do_l3) {
                const float4 vy2 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 8);
                if (do_l2) {
                    const float4 vy1 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 4);
                    do_nt1_m(4, 2, vy1.x); do_nt1_m(5, 2, vy1.y); do_nt1_m(6, 2, vy1.z); do_nt1_m(7, 2, vy1.w);
                    do_nt1_m(8, 2, vy2.x);
                }
                if (do_l3) {
                    do_nt1_m(9, 3, vy2.y); do_nt1_m(10, 3, vy2.z); do_nt1_m(11, 3, vy2.w);
                    const float4 vy3 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 12);
                    do_nt1_m(12, 3, vy3.x); do_nt1_m(13, 3, vy3.y); do_nt1_m(14, 3, vy3.z); do_nt1_m(15, 3, vy3.w);
                }
            }
            if constexpr (DX && SDX) {
                float ox_x = rx[0] * qx[0] + sgn * rx[1] * qx[1];
                float ox_y = ry[0] * qy[0] + sgn * ry[1] * qy[1];
                if (do_l2) { ox_x += rx[2] * qx[2]; ox_y += ry[2] * qy[2]; }
                if (do_l3) { ox_x += sgn * rx[3] * qx[3]; ox_y += sgn * ry[3] * qy[3]; }
                if constexpr (T0S_DIR) {
                    if (sp_s < S) { s_t0s[wblk][sp_s][lane].x += ox_x; s_t0s[wblk][sp_s][lane].y += ox_y; }
                } else {
                    atomicAdd(OX + s * 64 + 2 * lane, ox_x);
                    atomicAdd(OX + s * 64 + 2 * lane + 1, ox_y);
                }
            }
        } else if constexpr (FACT == 6 && std::is_same<H, __half>::value) {
            // Wave 45: Interleaved Y3P_H (yhp[0..3]=half2(Yw[m], Y[m]), yhp[4..5]=half2(Gy[2k], Gy[2k+1]))
            // with native __low2half2/__high2half2 -> hardware HFMA2 .H0_H0/.H1_H1 (0 PRMTs/edge, -2 live regs!).
            const bool do_l2 = !SHELL || (idx < kcut2);
            const bool do_l3 = !SHELL || (idx < kcut3);
            const uint4* yhp = reinterpret_cast<const uint4*>(Y3P) + 6L * p;
            const __half2 z_h = __float2half2_rn(0.f);
            float ox_x = 0.f, ox_y = 0.f;

            auto do_m_h2_i = [&](int m, uint32_t ywy_u, uint32_t gy_pair, bool hi,
                                 __half2 txs_h, __half2 rxs_h, __half2 rxb_h,
                                 __half2& q0_h, __half2& q1_h) {
                const __half2 ywy_h = *reinterpret_cast<const __half2*>(&ywy_u);
                const __half2 gy_h = *reinterpret_cast<const __half2*>(&gy_pair);
                const __half2 y0_h = __low2half2(ywy_h);
                const __half2 y1_h = hi ? __high2half2(gy_h) : __low2half2(gy_h);
                if constexpr (DG) {
                    og_h[m] = __hfma2(txs_h, y0_h, og_h[m]);
                    og_h[m] = __hfma2(rxs_h, y1_h, og_h[m]);
                    if constexpr (XB) {
                        og_h[m] = __hfma2(rxb_h, __high2half2(ywy_h), og_h[m]);
                    }
                }
                if constexpr (DX && SDX) {
                    const __half2 gh = *reinterpret_cast<const __half2*>(&gn[m]);
                    q0_h = __hfma2(gh, y0_h, q0_h);
                    q1_h = __hfma2(gh, y1_h, q1_h);
                }
            };
            auto finish_dx_l = [&](float rx, float ry, float tx, float ty, __half2 q0_h, __half2 q1_h) {
                if constexpr (DX && SDX) {
                    const float2 fq0 = __half22float2(q0_h);
                    const float2 fq1 = __half22float2(q1_h);
                    ox_x += tx * fq0.x + rx * fq1.x;
                    ox_y += ty * fq0.y + ry * fq1.y;
                }
            };
            {
                const uint4 w0 = yhp[0];
                const uint2 g0 = *(reinterpret_cast<const uint2*>(yhp + 4) + 0);
                float rx, ry, tx, ty;
                eval_c4_pair_rt(Ck4[cbase + 0 * 32], u, sp_invh, rx, ry, tx, ty);
                {
                    const __half2 txs_h = __floats2half2_rn(tx * xs.x, ty * xs.y);
                    const __half2 rxs_h = __floats2half2_rn(rx * xs.x, ry * xs.y);
                    const __half2 rxb_h = XB ? __floats2half2_rn(rx * xb.x, ry * xb.y) : z_h;
                    __half2 q0_h = z_h, q1_h = z_h;
                    do_m_h2_i(0, w0.x, g0.x, false, txs_h, rxs_h, rxb_h, q0_h, q1_h);
                    finish_dx_l(rx, ry, tx, ty, q0_h, q1_h);
                }
                eval_c4_pair_rt(Ck4[cbase + 1 * 32], u, sp_invh, rx, ry, tx, ty);
                rx *= sgn; ry *= sgn; tx *= sgn; ty *= sgn;
                {
                    const __half2 txs_h = __floats2half2_rn(tx * xs.x, ty * xs.y);
                    const __half2 rxs_h = __floats2half2_rn(rx * xs.x, ry * xs.y);
                    const __half2 rxb_h = XB ? __floats2half2_rn(rx * xb.x, ry * xb.y) : z_h;
                    __half2 q0_h = z_h, q1_h = z_h;
                    do_m_h2_i(1, w0.y, g0.x, true,  txs_h, rxs_h, rxb_h, q0_h, q1_h);
                    do_m_h2_i(2, w0.z, g0.y, false, txs_h, rxs_h, rxb_h, q0_h, q1_h);
                    do_m_h2_i(3, w0.w, g0.y, true,  txs_h, rxs_h, rxb_h, q0_h, q1_h);
                    finish_dx_l(rx, ry, tx, ty, q0_h, q1_h);
                }
            }
            if (do_l2 || do_l3) {
                if (do_l2) {
                    float rx, ry, tx, ty;
                    eval_c4_pair_rt(Ck4[cbase + 2 * 32], u, sp_invh, rx, ry, tx, ty);
                    const __half2 txs_h = __floats2half2_rn(tx * xs.x, ty * xs.y);
                    const __half2 rxs_h = __floats2half2_rn(rx * xs.x, ry * xs.y);
                    const __half2 rxb_h = XB ? __floats2half2_rn(rx * xb.x, ry * xb.y) : z_h;
                    __half2 q0_h = z_h, q1_h = z_h;
                    {
                        const uint4 w1 = yhp[1];
                        const uint2 g1 = *(reinterpret_cast<const uint2*>(yhp + 4) + 1);
                        do_m_h2_i(4, w1.x, g1.x, false, txs_h, rxs_h, rxb_h, q0_h, q1_h);
                        do_m_h2_i(5, w1.y, g1.x, true,  txs_h, rxs_h, rxb_h, q0_h, q1_h);
                        do_m_h2_i(6, w1.z, g1.y, false, txs_h, rxs_h, rxb_h, q0_h, q1_h);
                        do_m_h2_i(7, w1.w, g1.y, true,  txs_h, rxs_h, rxb_h, q0_h, q1_h);
                    }
                    {
                        const uint32_t w2x = *reinterpret_cast<const uint32_t*>(yhp + 2);
                        const uint32_t g2x = *(reinterpret_cast<const uint32_t*>(yhp + 4) + 4);
                        do_m_h2_i(8, w2x, g2x, false, txs_h, rxs_h, rxb_h, q0_h, q1_h);
                    }
                    finish_dx_l(rx, ry, tx, ty, q0_h, q1_h);
                }
                if (do_l3) {
                    float rx, ry, tx, ty;
                    eval_c4_pair_rt(Ck4[cbase + 3 * 32], u, sp_invh, rx, ry, tx, ty);
                    rx *= sgn; ry *= sgn; tx *= sgn; ty *= sgn;
                    const __half2 txs_h = __floats2half2_rn(tx * xs.x, ty * xs.y);
                    const __half2 rxs_h = __floats2half2_rn(rx * xs.x, ry * xs.y);
                    const __half2 rxb_h = XB ? __floats2half2_rn(rx * xb.x, ry * xb.y) : z_h;
                    __half2 q0_h = z_h, q1_h = z_h;
                    {
                        const uint4 w2 = yhp[2];
                        const uint2 g2 = *(reinterpret_cast<const uint2*>(yhp + 4) + 2);
                        do_m_h2_i(9,  w2.y, g2.x, true,  txs_h, rxs_h, rxb_h, q0_h, q1_h);
                        do_m_h2_i(10, w2.z, g2.y, false, txs_h, rxs_h, rxb_h, q0_h, q1_h);
                        do_m_h2_i(11, w2.w, g2.y, true,  txs_h, rxs_h, rxb_h, q0_h, q1_h);
                    }
                    {
                        const uint4 w3 = yhp[3];
                        const uint2 g3 = *(reinterpret_cast<const uint2*>(yhp + 4) + 3);
                        do_m_h2_i(12, w3.x, g3.x, false, txs_h, rxs_h, rxb_h, q0_h, q1_h);
                        do_m_h2_i(13, w3.y, g3.x, true,  txs_h, rxs_h, rxb_h, q0_h, q1_h);
                        do_m_h2_i(14, w3.z, g3.y, false, txs_h, rxs_h, rxb_h, q0_h, q1_h);
                        do_m_h2_i(15, w3.w, g3.y, true,  txs_h, rxs_h, rxb_h, q0_h, q1_h);
                    }
                    finish_dx_l(rx, ry, tx, ty, q0_h, q1_h);
                }
            }
            if constexpr (DX && SDX) {
                if constexpr (T0S_DIR) {
                    if (sp_s < S) { s_t0s[wblk][sp_s][lane].x += ox_x; s_t0s[wblk][sp_s][lane].y += ox_y; }
                } else {
                    atomicAdd(OX + s * 64 + 2 * lane, ox_x);
                    atomicAdd(OX + s * 64 + 2 * lane + 1, ox_y);
                }
            }
        } else if constexpr (FACT >= 4 && std::is_same<H, __half>::value) {
            // Wave 28: Native __half2 HFMA2 factored conv_rev2!
            //   - FACT == 4 (CONV_HFMA2=1): exact FP32 ogx/ogy + native HFMA2 DX (0 F2F unpacks of gn[0..15]!)
            //   - FACT == 5 (CONV_HFMA2=2): native HFMA2 for BOTH og_h[16] & DX (-16 regs -> 50% occ, 0 F2F unpacks!)
            const bool do_l2 = !SHELL || (idx < kcut2);
            const bool do_l3 = !SHELL || (idx < kcut3);
            const uint4* yhp = reinterpret_cast<const uint4*>(Y3P) + 6L * p;
            const __half2 z_h = __float2half2_rn(0.f);
            float ox_x = 0.f, ox_y = 0.f;

            auto do_m_h2 = [&](int m, uint32_t yw_pair, uint32_t gy_pair, uint32_t y_pair, bool hi,
                               __half2 txs_h, __half2 rxs_h, __half2 rxb_h,
                               float txs_x, float txs_y, float rxs_x, float rxs_y, float rxb_x, float rxb_y,
                               __half2& q0_h, __half2& q1_h) {
                const uint32_t y0_u = __byte_perm(yw_pair, yw_pair, hi ? 0x3232u : 0x1010u);
                const uint32_t y1_u = __byte_perm(gy_pair, gy_pair, hi ? 0x3232u : 0x1010u);
                const __half2 y0_h = *reinterpret_cast<const __half2*>(&y0_u);
                const __half2 y1_h = *reinterpret_cast<const __half2*>(&y1_u);
                if constexpr (DG) {
                    if constexpr (OG_H2) {
                        og_h[m] = __hfma2(txs_h, y0_h, og_h[m]);
                        og_h[m] = __hfma2(rxs_h, y1_h, og_h[m]);
                        if constexpr (XB) {
                            const uint32_t y2_u = __byte_perm(y_pair, y_pair, hi ? 0x3232u : 0x1010u);
                            og_h[m] = __hfma2(rxb_h, *reinterpret_cast<const __half2*>(&y2_u), og_h[m]);
                        }
                    } else {
                        const float y0v = __half2float(__low2half(y0_h));
                        const float y1v = __half2float(__low2half(y1_h));
                        ogx[m] += txs_x * y0v + rxs_x * y1v;
                        ogy[m] += txs_y * y0v + rxs_y * y1v;
                        if constexpr (XB) {
                            const uint32_t y2_u = __byte_perm(y_pair, y_pair, hi ? 0x3232u : 0x1010u);
                            const float y2v = __half2float(__low2half(*reinterpret_cast<const __half2*>(&y2_u)));
                            ogx[m] += rxb_x * y2v;
                            ogy[m] += rxb_y * y2v;
                        }
                    }
                }
                if constexpr (DX && SDX) {
                    const __half2 gh = *reinterpret_cast<const __half2*>(&gn[m]);
                    q0_h = __hfma2(gh, y0_h, q0_h);
                    q1_h = __hfma2(gh, y1_h, q1_h);
                }
            };
            auto finish_dx_l = [&](float rx, float ry, float tx, float ty, __half2 q0_h, __half2 q1_h) {
                if constexpr (DX && SDX) {
                    const float2 fq0 = __half22float2(q0_h);
                    const float2 fq1 = __half22float2(q1_h);
                    ox_x += tx * fq0.x + rx * fq1.x;
                    ox_y += ty * fq0.y + ry * fq1.y;
                }
            };
            {
                const uint4 q0 = yhp[0];
                const uint2 y0 = (DG && XB) ? *(reinterpret_cast<const uint2*>(yhp + 4) + 0) : make_uint2(0u, 0u);
                float rx, ry, tx, ty;
                eval_c4_pair_rt(Ck4[cbase + 0 * 32], u, sp_invh, rx, ry, tx, ty);
                {
                    const float txs_x = tx * xs.x, txs_y = ty * xs.y;
                    const float rxs_x = rx * xs.x, rxs_y = ry * xs.y;
                    const float rxb_x = XB ? rx * xb.x : 0.f, rxb_y = XB ? ry * xb.y : 0.f;
                    const __half2 txs_h = OG_H2 ? __floats2half2_rn(txs_x, txs_y) : z_h;
                    const __half2 rxs_h = OG_H2 ? __floats2half2_rn(rxs_x, rxs_y) : z_h;
                    const __half2 rxb_h = (OG_H2 && XB) ? __floats2half2_rn(rxb_x, rxb_y) : z_h;
                    __half2 q0_h = z_h, q1_h = z_h;
                    do_m_h2(0, q0.x, q0.z, y0.x, false, txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                    finish_dx_l(rx, ry, tx, ty, q0_h, q1_h);
                }
                eval_c4_pair_rt(Ck4[cbase + 1 * 32], u, sp_invh, rx, ry, tx, ty);
                rx *= sgn; ry *= sgn; tx *= sgn; ty *= sgn;
                {
                    const float txs_x = tx * xs.x, txs_y = ty * xs.y;
                    const float rxs_x = rx * xs.x, rxs_y = ry * xs.y;
                    const float rxb_x = XB ? rx * xb.x : 0.f, rxb_y = XB ? ry * xb.y : 0.f;
                    const __half2 txs_h = OG_H2 ? __floats2half2_rn(txs_x, txs_y) : z_h;
                    const __half2 rxs_h = OG_H2 ? __floats2half2_rn(rxs_x, rxs_y) : z_h;
                    const __half2 rxb_h = (OG_H2 && XB) ? __floats2half2_rn(rxb_x, rxb_y) : z_h;
                    __half2 q0_h = z_h, q1_h = z_h;
                    do_m_h2(1, q0.x, q0.z, y0.x, true,  txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                    do_m_h2(2, q0.y, q0.w, y0.y, false, txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                    do_m_h2(3, q0.y, q0.w, y0.y, true,  txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                    finish_dx_l(rx, ry, tx, ty, q0_h, q1_h);
                }
            }
            if (do_l2 || do_l3) {
                if (do_l2) {
                    float rx, ry, tx, ty;
                    eval_c4_pair_rt(Ck4[cbase + 2 * 32], u, sp_invh, rx, ry, tx, ty);
                    const float txs_x = tx * xs.x, txs_y = ty * xs.y;
                    const float rxs_x = rx * xs.x, rxs_y = ry * xs.y;
                    const float rxb_x = XB ? rx * xb.x : 0.f, rxb_y = XB ? ry * xb.y : 0.f;
                    const __half2 txs_h = OG_H2 ? __floats2half2_rn(txs_x, txs_y) : z_h;
                    const __half2 rxs_h = OG_H2 ? __floats2half2_rn(rxs_x, rxs_y) : z_h;
                    const __half2 rxb_h = (OG_H2 && XB) ? __floats2half2_rn(rxb_x, rxb_y) : z_h;
                    __half2 q0_h = z_h, q1_h = z_h;
                    {
                        const uint4 q1 = yhp[1];
                        const uint2 y1 = (DG && XB) ? *(reinterpret_cast<const uint2*>(yhp + 4) + 1) : make_uint2(0u, 0u);
                        do_m_h2(4, q1.x, q1.z, y1.x, false, txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                        do_m_h2(5, q1.x, q1.z, y1.x, true,  txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                        do_m_h2(6, q1.y, q1.w, y1.y, false, txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                        do_m_h2(7, q1.y, q1.w, y1.y, true,  txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                    }
                    {
                        // Load only q2.x/q2.z & y2.x after q1/y1 are dead -> saves 6 peak regs in Layer 1 (STACK:8 -> STACK:0!)
                        const uint32_t* q2p = reinterpret_cast<const uint32_t*>(yhp + 2);
                        const uint32_t y2x = (DG && XB) ? *(reinterpret_cast<const uint32_t*>(yhp + 4) + 4) : 0u;
                        do_m_h2(8, q2p[0], q2p[2], y2x, false, txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                    }
                    finish_dx_l(rx, ry, tx, ty, q0_h, q1_h);
                }
                if (do_l3) {
                    float rx, ry, tx, ty;
                    eval_c4_pair_rt(Ck4[cbase + 3 * 32], u, sp_invh, rx, ry, tx, ty);
                    rx *= sgn; ry *= sgn; tx *= sgn; ty *= sgn;
                    const float txs_x = tx * xs.x, txs_y = ty * xs.y;
                    const float rxs_x = rx * xs.x, rxs_y = ry * xs.y;
                    const float rxb_x = XB ? rx * xb.x : 0.f, rxb_y = XB ? ry * xb.y : 0.f;
                    const __half2 txs_h = OG_H2 ? __floats2half2_rn(txs_x, txs_y) : z_h;
                    const __half2 rxs_h = OG_H2 ? __floats2half2_rn(rxs_x, rxs_y) : z_h;
                    const __half2 rxb_h = (OG_H2 && XB) ? __floats2half2_rn(rxb_x, rxb_y) : z_h;
                    __half2 q0_h = z_h, q1_h = z_h;
                    {
                        const uint4 q2 = yhp[2];
                        const uint2 y2 = (DG && XB) ? *(reinterpret_cast<const uint2*>(yhp + 4) + 2) : make_uint2(0u, 0u);
                        do_m_h2(9,  q2.x, q2.z, y2.x, true,  txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                        do_m_h2(10, q2.y, q2.w, y2.y, false, txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                        do_m_h2(11, q2.y, q2.w, y2.y, true,  txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                    }
                    {
                        const uint4 q3 = yhp[3];
                        const uint2 y3 = (DG && XB) ? *(reinterpret_cast<const uint2*>(yhp + 4) + 3) : make_uint2(0u, 0u);
                        do_m_h2(12, q3.x, q3.z, y3.x, false, txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                        do_m_h2(13, q3.x, q3.z, y3.x, true,  txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                        do_m_h2(14, q3.y, q3.w, y3.y, false, txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                        do_m_h2(15, q3.y, q3.w, y3.y, true,  txs_h, rxs_h, rxb_h, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0_h, q1_h);
                    }
                    finish_dx_l(rx, ry, tx, ty, q0_h, q1_h);
                }
            }
            if constexpr (DX && SDX) {
                if constexpr (T0S_DIR) {
                    if (sp_s < S) { s_t0s[wblk][sp_s][lane].x += ox_x; s_t0s[wblk][sp_s][lane].y += ox_y; }
                } else {
                    atomicAdd(OX + s * 64 + 2 * lane, ox_x);
                    atomicAdd(OX + s * 64 + 2 * lane + 1, ox_y);
                }
            }
        } else if constexpr (FACT >= 1) {
            // Wave 20+22+26: 1-l-at-a-time factored conv_rev2; optional SCK smem Ck cache (ES=1) & T0S_DIR (Layer 0)!
            const bool do_l2 = !SHELL || (idx < kcut2);
            const bool do_l3 = !SHELL || (idx < kcut3);
            if constexpr (SCK) {
                if (idx != prev_idx) {
                    s_ck[wblk][0][lane] = Ck4[cbase + 0 * 32];
                    s_ck[wblk][1][lane] = Ck4[cbase + 1 * 32];
                    if (do_l2) s_ck[wblk][2][lane] = Ck4[cbase + 2 * 32];
                    if (do_l3) s_ck[wblk][3][lane] = Ck4[cbase + 3 * 32];
                    prev_idx = idx;
                }
            }
            auto get_ck = [&](int l) -> uint4 {
                if constexpr (SCK) return s_ck[wblk][l][lane];
                else return Ck4[cbase + l * 32];
            };
            auto load_y4 = [&](int q, float4& v0, float4& v1, float4& v2) {
                if constexpr (FACT == 3) {
                    const uint4* yhp = reinterpret_cast<const uint4*>(Y3P) + 6L * p;
                    const uint4 qh = yhp[q];
                    const float2 w01 = __half22float2(*reinterpret_cast<const __half2*>(&qh.x));
                    const float2 w23 = __half22float2(*reinterpret_cast<const __half2*>(&qh.y));
                    const float2 g01 = __half22float2(*reinterpret_cast<const __half2*>(&qh.z));
                    const float2 g23 = __half22float2(*reinterpret_cast<const __half2*>(&qh.w));
                    v0 = make_float4(w01.x, w01.y, w23.x, w23.y);
                    v1 = make_float4(g01.x, g01.y, g23.x, g23.y);
                    if constexpr (DG && XB) {
                        const uint2 yy = *(reinterpret_cast<const uint2*>(yhp + 4) + q);
                        const float2 y01 = __half22float2(*reinterpret_cast<const __half2*>(&yy.x));
                        const float2 y23 = __half22float2(*reinterpret_cast<const __half2*>(&yy.y));
                        v2 = make_float4(y01.x, y01.y, y23.x, y23.y);
                    } else {
                        v2 = make_float4(0.f, 0.f, 0.f, 0.f);
                    }
                } else if constexpr (FACT == 2) {
                    const float* yp = Y3P + 48L * p + 12 * q;
                    v0 = *reinterpret_cast<const float4*>(yp + 0);
                    v1 = *reinterpret_cast<const float4*>(yp + 4);
                    if constexpr (DG && XB) v2 = *reinterpret_cast<const float4*>(yp + 8);
                    else v2 = make_float4(0.f, 0.f, 0.f, 0.f);
                } else {
                    v0 = *reinterpret_cast<const float4*>(Y0 + 16L * p + 4 * q);
                    v1 = *reinterpret_cast<const float4*>(Y1 + 16L * p + 4 * q);
                    if constexpr (DG && XB) v2 = *reinterpret_cast<const float4*>(Y2 + 16L * p + 4 * q);
                    else v2 = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            };
            float ox_x = 0.f, ox_y = 0.f;
            auto do_m = [&](int m, float y0v, float y1v, float y2v, float txs_x, float txs_y, float rxs_x, float rxs_y,
                            float rxb_x, float rxb_y, float& q0x, float& q0y, float& q1x, float& q1y) {
                if constexpr (DG) {
                    ogx[m] += txs_x * y0v + rxs_x * y1v;
                    ogy[m] += txs_y * y0v + rxs_y * y1v;
                    if constexpr (XB) {
                        ogx[m] += rxb_x * y2v;
                        ogy[m] += rxb_y * y2v;
                    }
                }
                if constexpr (DX && SDX) {
                    const float2 g2 = ah::unpack2<H>(gn[m]);
                    q0x += g2.x * y0v; q0y += g2.y * y0v;
                    q1x += g2.x * y1v; q1y += g2.y * y1v;
                }
            };
            {
                float4 v0, v1, v2;
                load_y4(0, v0, v1, v2);
                float rx, ry, tx, ty;
                eval_c4_pair_rt(get_ck(0), u, sp_invh, rx, ry, tx, ty);
                {
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    do_m(0, v0.x, v1.x, v2.x, tx * xs.x, ty * xs.y, rx * xs.x, ry * xs.y,
                         XB ? rx * xb.x : 0.f, XB ? ry * xb.y : 0.f, q0x, q0y, q1x, q1y);
                    if constexpr (DX && SDX) { ox_x += tx * q0x + rx * q1x; ox_y += ty * q0y + ry * q1y; }
                }
                eval_c4_pair_rt(get_ck(1), u, sp_invh, rx, ry, tx, ty);
                rx *= sgn; ry *= sgn; tx *= sgn; ty *= sgn;
                {
                    const float txs_x = tx * xs.x, txs_y = ty * xs.y;
                    const float rxs_x = rx * xs.x, rxs_y = ry * xs.y;
                    const float rxb_x = XB ? rx * xb.x : 0.f, rxb_y = XB ? ry * xb.y : 0.f;
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    do_m(1, v0.y, v1.y, v2.y, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                    do_m(2, v0.z, v1.z, v2.z, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                    do_m(3, v0.w, v1.w, v2.w, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                    if constexpr (DX && SDX) { ox_x += tx * q0x + rx * q1x; ox_y += ty * q0y + ry * q1y; }
                }
            }
            if (do_l2 || do_l3) {
                float4 v0b, v1b, v2b;
                load_y4(2, v0b, v1b, v2b);
                if (do_l2) {
                    float rx, ry, tx, ty;
                    eval_c4_pair_rt(get_ck(2), u, sp_invh, rx, ry, tx, ty);
                    const float txs_x = tx * xs.x, txs_y = ty * xs.y;
                    const float rxs_x = rx * xs.x, rxs_y = ry * xs.y;
                    const float rxb_x = XB ? rx * xb.x : 0.f, rxb_y = XB ? ry * xb.y : 0.f;
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    {
                        float4 v0, v1, v2;
                        load_y4(1, v0, v1, v2);
                        do_m(4, v0.x, v1.x, v2.x, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                        do_m(5, v0.y, v1.y, v2.y, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                        do_m(6, v0.z, v1.z, v2.z, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                        do_m(7, v0.w, v1.w, v2.w, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                    }
                    do_m(8, v0b.x, v1b.x, v2b.x, txs_x, txs_y, rxs_x, rxs_y, rxb_x, rxb_y, q0x, q0y, q1x, q1y);
                    if constexpr (DX && SDX) { ox_x += tx * q0x + rx * q1x; ox_y += ty * q0y + ry * q1y; }
                }
                if (do_l3) {
                    float rx, ry, tx, ty;
                    eval_c4_pair_rt(get_ck(3), u, sp_invh, rx, ry, tx, ty);
                    rx *= sgn; ry *= sgn; tx *= sgn; ty *= sgn;
                    const float txs3_x = tx * xs.x, txs3_y = ty * xs.y;
                    const float rxs3_x = rx * xs.x, rxs3_y = ry * xs.y;
                    const float rxb3_x = XB ? rx * xb.x : 0.f, rxb3_y = XB ? ry * xb.y : 0.f;
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    do_m(9, v0b.y, v1b.y, v2b.y, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                    do_m(10, v0b.z, v1b.z, v2b.z, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                    do_m(11, v0b.w, v1b.w, v2b.w, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                    {
                        float4 v0c, v1c, v2c;
                        load_y4(3, v0c, v1c, v2c);
                        do_m(12, v0c.x, v1c.x, v2c.x, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                        do_m(13, v0c.y, v1c.y, v2c.y, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                        do_m(14, v0c.z, v1c.z, v2c.z, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                        do_m(15, v0c.w, v1c.w, v2c.w, txs3_x, txs3_y, rxs3_x, rxs3_y, rxb3_x, rxb3_y, q0x, q0y, q1x, q1y);
                    }
                    if constexpr (DX && SDX) { ox_x += tx * q0x + rx * q1x; ox_y += ty * q0y + ry * q1y; }
                }
            }
            if constexpr (DX && SDX) {
                if constexpr (T0S_DIR) {
                    if (sp_s < S) { s_t0s[wblk][sp_s][lane].x += ox_x; s_t0s[wblk][sp_s][lane].y += ox_y; }
                } else {
                    atomicAdd(OX + s * 64 + 2 * lane, ox_x);
                    atomicAdd(OX + s * 64 + 2 * lane + 1, ox_y);
                }
            }
        } else {
            uint4 ck[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) ck[l] = Ck4[cbase + l * 32];
            float rx[4], ry[4], tx[4], ty[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                eval_c4_pair_rt(ck[l], u, sp_invh, rx[l], ry[l], tx[l], ty[l]);
                if (l & 1) { rx[l] *= sgn; ry[l] *= sgn; tx[l] *= sgn; ty[l] *= sgn; }
            }
            float q0x[4] = {0.f, 0.f, 0.f, 0.f}, q0y[4] = {0.f, 0.f, 0.f, 0.f};
            float q1x[4] = {0.f, 0.f, 0.f, 0.f}, q1y[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int m0 = 0; m0 < 16; m0 += 4) {
                const float4 v0 = *reinterpret_cast<const float4*>(Y0 + 16L * p + m0);
                const float4 v1 = *reinterpret_cast<const float4*>(Y1 + 16L * p + m0);
                const float y0a[4] = {v0.x, v0.y, v0.z, v0.w};
                const float y1a[4] = {v1.x, v1.y, v1.z, v1.w};
                float y2a[4];
                if constexpr (DG && XB) {
                    const float4 v2 = *reinterpret_cast<const float4*>(Y2 + 16L * p + m0);
                    y2a[0] = v2.x; y2a[1] = v2.y; y2a[2] = v2.z; y2a[3] = v2.w;
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int m = m0 + j;
                    const int l = lofm_h2(m);
                    const float y0v = y0a[j], y1v = y1a[j];
                    if constexpr (DG) {
                        const float ax = tx[l] * y0v + rx[l] * y1v;
                        const float ay = ty[l] * y0v + ry[l] * y1v;
                        ogx[m] += ax * xs.x;
                        ogy[m] += ay * xs.y;
                        if constexpr (XB) {
                            ogx[m] += rx[l] * y2a[j] * xb.x;
                            ogy[m] += ry[l] * y2a[j] * xb.y;
                        }
                    }
                    if constexpr (DX && SDX) {
                        const float2 g2 = ah::unpack2<H>(gn[m]);
                        q0x[l] += g2.x * y0v; q0y[l] += g2.y * y0v;
                        q1x[l] += g2.x * y1v; q1y[l] += g2.y * y1v;
                    }
                }
            }
            if constexpr (DX && SDX) {
                float ox_x = 0.f, ox_y = 0.f;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    ox_x += tx[l] * q0x[l] + rx[l] * q1x[l];
                    ox_y += ty[l] * q0y[l] + ry[l] * q1y[l];
                }
                if constexpr (T0S_DIR) {
                    if (sp_s < S) { s_t0s[wblk][sp_s][lane].x += ox_x; s_t0s[wblk][sp_s][lane].y += ox_y; }
                } else {
                    atomicAdd(OX + s * 64 + 2 * lane, ox_x);
                    atomicAdd(OX + s * 64 + 2 * lane + 1, ox_y);
                }
            }
        }
    }
    if constexpr (T0S_DIR && DX && SDX && !(ES > 1 && DG)) {
        if (n < N) {
            for (int sp = 0; sp < S; ++sp) {
                const float2 v = s_t0s[wblk][sp][lane];
                if (v.x != 0.f) atomicAdd(T0s + sp * 64 + 2 * lane, v.x);
                if (v.y != 0.f) atomicAdd(T0s + sp * 64 + 2 * lane + 1, v.y);
            }
        }
    }
    if constexpr (SDX && !DG) return;
    if constexpr (ES > 1 && DG) {
        constexpr int NS = NPB * (ES - 1) * 32;
        if constexpr (OG_H2) {
            __shared__ uint32_t sb_h[16][NS];
            for (int q = 1; q < ES; ++q) {
                const int slot = (node_in_blk * 32 + lane) * (ES - 1) + (q - 1);
                if (grp == q) {
#pragma unroll
                    for (int m = 0; m < 16; ++m) sb_h[m][slot] = *reinterpret_cast<const uint32_t*>(&og_h[m]);
                }
            }
            __syncthreads();
            if constexpr (T0S_DIR && DX && SDX) {
                for (int sp = wblk; sp < S; sp += NW_BLK) {
                    float sx = 0.f, sy = 0.f;
#pragma unroll
                    for (int w = 0; w < NW_BLK; ++w) {
                        const float2 v = s_t0s[w][sp][lane];
                        sx += v.x;
                        sy += v.y;
                    }
                    if (sx != 0.f) atomicAdd(T0s + sp * 64 + 2 * lane, sx);
                    if (sy != 0.f) atomicAdd(T0s + sp * 64 + 2 * lane + 1, sy);
                }
            }
            if (grp != 0 || n >= N) return;
            const long obase = (long)n * 16 * 32 + lane;
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                float2 acc = __half22float2(og_h[m]);
                for (int q = 1; q < ES; ++q) {
                    const int slot = (node_in_blk * 32 + lane) * (ES - 1) + (q - 1);
                    const uint32_t u = sb_h[m][slot];
                    const float2 v = __half22float2(*reinterpret_cast<const __half2*>(&u));
                    acc.x += v.x;
                    acc.y += v.y;
                }
                ah::st_h2<H>(OG, obase + (long)m * 32, ah::pack2<H>(acc.x, acc.y));
            }
            return;
        } else {
        __shared__ float2 sb[16][NS];
        for (int q = 1; q < ES; ++q) {
            const int slot = (node_in_blk * 32 + lane) * (ES - 1) + (q - 1);
            if (grp == q) {
#pragma unroll
                for (int m = 0; m < 16; ++m) sb[m][slot] = make_float2(ogx[m], ogy[m]);
            }
        }
        __syncthreads();
        if constexpr (T0S_DIR && DX && SDX) {
            for (int sp = wblk; sp < S; sp += NW_BLK) {
                float sx = 0.f, sy = 0.f;
#pragma unroll
                for (int w = 0; w < NW_BLK; ++w) {
                    const float2 v = s_t0s[w][sp][lane];
                    sx += v.x;
                    sy += v.y;
                }
                if (sx != 0.f) atomicAdd(T0s + sp * 64 + 2 * lane, sx);
                if (sy != 0.f) atomicAdd(T0s + sp * 64 + 2 * lane + 1, sy);
            }
        }
        if (grp != 0 || n >= N) return;
        for (int q = 1; q < ES; ++q) {
            const int slot = (node_in_blk * 32 + lane) * (ES - 1) + (q - 1);
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                const float2 v = sb[m][slot];
                ogx[m] += v.x;
                ogy[m] += v.y;
            }
        }
        }
    }
    if constexpr (DG) {
        if (n >= N) return;
        const long obase = (long)n * 16 * 32 + lane;
        if constexpr (OG_H2) {
#pragma unroll
            for (int m = 0; m < 16; ++m)
                ah::st_h2<H>(OG, obase + (long)m * 32, *reinterpret_cast<const uint32_t*>(&og_h[m]));
        } else {
#pragma unroll
            for (int m = 0; m < 16; ++m) ah::st_h2<H>(OG, obase + (long)m * 32, ah::pack2<H>(ogx[m], ogy[m]));
        }
    }
}

// ============================================================================ 2. k_slot_xd_opt (2-warp 64ch, C == 64)
// High-occupancy (~84-96 regs, 0 spills -> 5 blocks/SM = 20 warps/SM!).
// Wave 15: RUN_ACC=true accumulates gC_k in 8 KB smem across consecutive same-kidx edges (emeta is kidx-sorted!)
// and flushes atomics ONLY on kidx transitions -> 3x-6x fewer L2 atomicAdds with bit-exact sums!
template <typename H, bool XB, bool POLY_BWD, bool EMETA, bool CACHE_C, bool RUN_ACC = false>
__global__ void __launch_bounds__(128, 5) k_slot_xd_opt(
    const int* __restrict__ rptr, const int* __restrict__ rptr0, const int* __restrict__ snd,
    const int* __restrict__ ps, const int* __restrict__ sp_idx, const float* __restrict__ sp_u,
    const H* __restrict__ G, const H* __restrict__ A, const float* __restrict__ X, const float* __restrict__ Xb,
    const float* __restrict__ YwP, const float* __restrict__ GyP, const float* __restrict__ YP,
    const float* __restrict__ C_kh, float* __restrict__ gR_k, float* __restrict__ gT_k, float* __restrict__ Xd,
    float sp_h, float sp_invh, int N, const int4* __restrict__ emeta, const int* __restrict__ norder = nullptr) {
    const int nn = blockIdx.x;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    const int c = threadIdx.x & 63;
    const int grp = threadIdx.x >> 6;  // ES == 2: 0 or 1 (128 threads/block = 1 node/block)

    const int e0_n = rptr[n];
    const int e1_n = EMETA ? rptr0[n] : rptr[n + 1];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = EMETA ? (e0_n + (cnt_n * grp) / 2) : (e0_n + grp);
    const int e_end = EMETA ? (e0_n + (cnt_n * (grp + 1)) / 2) : e1_n;
    const int e_step = EMETA ? 1 : 2;
    if (e_beg >= e_end) return;

    __shared__ float s_gc[RUN_ACC ? 2 : 1][RUN_ACC ? 16 : 1][RUN_ACC ? 64 : 1];

    uint32_t gn2[8], an2[8];
    const long nbase = (long)n * 16 * 64 + c;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        gn2[k] = ah::pack2_raw<H>(G[nbase + (long)(2 * k) * 64], G[nbase + (long)(2 * k + 1) * 64]);
        an2[k] = ah::pack2_raw<H>(A[nbase + (long)(2 * k) * 64], A[nbase + (long)(2 * k + 1) * 64]);
    }
    const float xn = X[(long)n * 64 + c];
    const float xbn = XB ? Xb[(long)n * 64 + c] : 0.f;
    float xd_n = 0.f;
    const uint2* Ck2 = reinterpret_cast<const uint2*>(C_kh);

    int prev_kidx = -1;
    uint2 prev_c[4];

    auto flush_gc = [&](int pk) {
        if constexpr (RUN_ACC && POLY_BWD) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const long base = ((long)pk * 4 + l) * 4 * 64 + c;
#pragma unroll
                for (int q = 0; q < 4; ++q) atomicAdd(gR_k + base + q * 64, s_gc[grp][l * 4 + q][c]);
            }
        }
    };

    for (int e = e_beg; e < e_end; e += e_step) {
        int ps_e, kidx;
        long p, s;
        float u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            if (ps_e & 1) continue;
            kidx = em.z;
            if (kidx < 0) continue;
            p = ps_e >> 1;
            s = em.y;
            u = __int_as_float(em.w);
        } else {
            ps_e = ps[e];
            if (ps_e & 1) continue;
            p = ps_e >> 1;
            kidx = sp_idx[p];
            if (kidx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        const bool same_k = (kidx == prev_kidx);
        if constexpr (RUN_ACC && POLY_BWD) {
            if (!same_k) {
                if (prev_kidx >= 0) flush_gc(prev_kidx);
                prev_kidx = kidx;
            }
        } else if constexpr (CACHE_C) {
            if (!same_k) {
                const long cbase = (long)kidx * 4 * 64 + c;
#pragma unroll
                for (int l = 0; l < 4; ++l) prev_c[l] = Ck2[cbase + l * 64];
                prev_kidx = kidx;
            }
        }
        const float xs = X[s * 64 + c];
        const float xbs = XB ? Xb[s * 64 + c] : 0.f;
        const long sbase = s * 16 * 64 + c;
        const long cbase = (long)kidx * 4 * 64 + c;
        const long k0 = cbase, k1 = k0 + 4 * 64;

        float h00 = 0.f, h10 = 0.f, h11 = 0.f, d00 = 0.f, d10 = 0.f, d11 = 0.f;
        if constexpr (!POLY_BWD) mspl::hermite_rt<float>(u, sp_h, sp_invh, h00, h10, h11, d00, d10, d11);
        const float h01 = 1.f - h00;
        const float u2 = u * u, u3 = u2 * u;
        const float ui = u * sp_invh, u2i = u2 * sp_invh;
        float xd_s = 0.f;

        auto finish_l = [&](int l, float sgn2, float tv, float rv, float ayn, float ays) {
            float rval;
            if constexpr (CACHE_C && !RUN_ACC) rval = eval_c4_s(prev_c[l], u);
            else rval = eval_c4_s(Ck2[cbase + l * 64], u);
            xd_s += rval * ayn;
            xd_n += sgn2 * rval * ays;
            rv += ayn * xs + sgn2 * ays * xn;
            if constexpr (POLY_BWD) {
                const float g0 = rv;
                const float g1 = u * rv + sp_invh * tv;
                const float g2 = u2 * rv + 2.f * ui * tv;
                const float g3 = u3 * rv + 3.f * u2i * tv;
                if constexpr (RUN_ACC) {
                    if (same_k) {
                        s_gc[grp][l * 4 + 0][c] += g0;
                        s_gc[grp][l * 4 + 1][c] += g1;
                        s_gc[grp][l * 4 + 2][c] += g2;
                        s_gc[grp][l * 4 + 3][c] += g3;
                    } else {
                        s_gc[grp][l * 4 + 0][c] = g0;
                        s_gc[grp][l * 4 + 1][c] = g1;
                        s_gc[grp][l * 4 + 2][c] = g2;
                        s_gc[grp][l * 4 + 3][c] = g3;
                    }
                } else {
                    const long base = ((long)kidx * 4 + l) * 4 * 64 + c;
                    atomicAdd(gR_k + base + 0 * 64, g0);
                    atomicAdd(gR_k + base + 1 * 64, g1);
                    atomicAdd(gR_k + base + 2 * 64, g2);
                    atomicAdd(gR_k + base + 3 * 64, g3);
                }
            } else {
                atomicAdd(gR_k + k0 + l * 64, h00 * rv + d00 * tv);
                atomicAdd(gR_k + k1 + l * 64, h01 * rv - d00 * tv);
                atomicAdd(gT_k + k0 + l * 64, h10 * rv + d10 * tv);
                atomicAdd(gT_k + k1 + l * 64, h11 * rv + d11 * tv);
            }
        };
        auto accum_m = [&](int m, float gnm, float anm, float ywv, float gyv, float yv, float xns, float xbns,
                           float& tv, float& rv, float& ayn, float& ays) {
            const float gs = ah::ld_a16<float>(G, sbase + (long)m * 64);
            const float as = ah::ld_a16<float>(A, sbase + (long)m * 64);
            const float gc = gnm * xs + gs * xns;
            tv += gc * ywv;
            rv += gc * gyv;
            if constexpr (XB) rv += (gnm * xbs + gs * xbns) * yv;
            ayn += anm * yv;
            ays += as * yv;
        };

        // Chunk 0 (m = 0..3): l=0 (m=0), l=1 (m=1..3)
        {
            const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 0);
            const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 0);
            const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
            const float2 g01 = ah::unpack2<H>(gn2[0]);
            const float2 a01 = ah::unpack2<H>(an2[0]);
            float tv = 0.f, rv = 0.f, ayn = 0.f, ays = 0.f;
            accum_m(0, g01.x, a01.x, v0.x, v1.x, v2.x, xn, xbn, tv, rv, ayn, ays);
            finish_l(0, 1.f, tv, rv, ayn, ays);

            tv = 0.f; rv = 0.f; ayn = 0.f; ays = 0.f;
            accum_m(1, g01.y, a01.y, v0.y, v1.y, v2.y, -xn, -xbn, tv, rv, ayn, ays);
            const float2 g23 = ah::unpack2<H>(gn2[1]);
            const float2 a23 = ah::unpack2<H>(an2[1]);
            accum_m(2, g23.x, a23.x, v0.z, v1.z, v2.z, -xn, -xbn, tv, rv, ayn, ays);
            accum_m(3, g23.y, a23.y, v0.w, v1.w, v2.w, -xn, -xbn, tv, rv, ayn, ays);
            finish_l(1, -1.f, tv, rv, ayn, ays);
        }
        // Chunk 1 & 2 (m = 4..11): l=2 (m=4..8), start l=3 (m=9..11)
        {
            float tv = 0.f, rv = 0.f, ayn = 0.f, ays = 0.f;
            {
                const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 4);
                const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 4);
                const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                const float2 g45 = ah::unpack2<H>(gn2[2]);
                const float2 a45 = ah::unpack2<H>(an2[2]);
                accum_m(4, g45.x, a45.x, v0.x, v1.x, v2.x, xn, xbn, tv, rv, ayn, ays);
                accum_m(5, g45.y, a45.y, v0.y, v1.y, v2.y, xn, xbn, tv, rv, ayn, ays);
                const float2 g67 = ah::unpack2<H>(gn2[3]);
                const float2 a67 = ah::unpack2<H>(an2[3]);
                accum_m(6, g67.x, a67.x, v0.z, v1.z, v2.z, xn, xbn, tv, rv, ayn, ays);
                accum_m(7, g67.y, a67.y, v0.w, v1.w, v2.w, xn, xbn, tv, rv, ayn, ays);
            }
            const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 8);
            const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 8);
            const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
            const float2 g89 = ah::unpack2<H>(gn2[4]);
            const float2 a89 = ah::unpack2<H>(an2[4]);
            accum_m(8, g89.x, a89.x, v0.x, v1.x, v2.x, xn, xbn, tv, rv, ayn, ays);
            finish_l(2, 1.f, tv, rv, ayn, ays);

            tv = 0.f; rv = 0.f; ayn = 0.f; ays = 0.f;
            accum_m(9, g89.y, a89.y, v0.y, v1.y, v2.y, -xn, -xbn, tv, rv, ayn, ays);
            const float2 gAB = ah::unpack2<H>(gn2[5]);
            const float2 aAB = ah::unpack2<H>(an2[5]);
            accum_m(10, gAB.x, aAB.x, v0.z, v1.z, v2.z, -xn, -xbn, tv, rv, ayn, ays);
            accum_m(11, gAB.y, aAB.y, v0.w, v1.w, v2.w, -xn, -xbn, tv, rv, ayn, ays);
            {
                const float4 v0c = *reinterpret_cast<const float4*>(YwP + 16L * p + 12);
                const float4 v1c = *reinterpret_cast<const float4*>(GyP + 16L * p + 12);
                const float4 v2c = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                const float2 gCD = ah::unpack2<H>(gn2[6]);
                const float2 aCD = ah::unpack2<H>(an2[6]);
                accum_m(12, gCD.x, aCD.x, v0c.x, v1c.x, v2c.x, -xn, -xbn, tv, rv, ayn, ays);
                accum_m(13, gCD.y, aCD.y, v0c.y, v1c.y, v2c.y, -xn, -xbn, tv, rv, ayn, ays);
                const float2 gEF = ah::unpack2<H>(gn2[7]);
                const float2 aEF = ah::unpack2<H>(an2[7]);
                accum_m(14, gEF.x, aEF.x, v0c.z, v1c.z, v2c.z, -xn, -xbn, tv, rv, ayn, ays);
                accum_m(15, gEF.y, aEF.y, v0c.w, v1c.w, v2c.w, -xn, -xbn, tv, rv, ayn, ays);
            }
            finish_l(3, -1.f, tv, rv, ayn, ays);
        }
        atomicAdd(Xd + s * 64 + c, xd_s);
    }
    if constexpr (RUN_ACC && POLY_BWD) {
        if (prev_kidx >= 0) flush_gc(prev_kidx);
    }
    atomicAdd(Xd + (long)n * 64 + c, xd_n);
}

// ============================================================================ 2b. k_slot_xd_h2_run (1-warp 64ch __half2!)
// Wave 15 breakthrough: combines 2x fewer LSU ops (16x LDG.32 __half2 for G[s] & A[s] = 100% 128B cache lines!),
// streaming per-degree l=0..3 (<=104 regs, 0 spills), rptr0 zero-skip, spatial norder, AND 8 KB smem knot-run
// gC_k accumulation (3x-6x fewer L2 atomicAdds!). 64 threads (2 warps = 1 node/block), ZERO __syncthreads()!
template <typename H, bool XB, bool EMETA>
__global__ void __launch_bounds__(64, 6) k_slot_xd_h2_run(
    const int* __restrict__ rptr, const int* __restrict__ rptr0, const int* __restrict__ snd,
    const int* __restrict__ ps, const int* __restrict__ sp_idx, const float* __restrict__ sp_u,
    const H* __restrict__ G, const H* __restrict__ A, const float* __restrict__ X, const float* __restrict__ Xb,
    const float* __restrict__ YwP, const float* __restrict__ GyP, const float* __restrict__ YP,
    const float* __restrict__ C_kh, float* __restrict__ gR_k, float* __restrict__ Xd,
    float sp_invh, int N, const int4* __restrict__ emeta, const int* __restrict__ norder = nullptr) {
    const int nn = blockIdx.x;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    const int lane = threadIdx.x & 31;
    const int grp = threadIdx.x >> 5;  // 0 or 1 (2 warps = 1 node/block)

    const int e0_n = rptr[n];
    const int e1_n = EMETA ? rptr0[n] : rptr[n + 1];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = EMETA ? (e0_n + (cnt_n * grp) / 2) : (e0_n + grp);
    const int e_end = EMETA ? (e0_n + (cnt_n * (grp + 1)) / 2) : e1_n;
    const int e_step = EMETA ? 1 : 2;
    if (e_beg >= e_end) return;

    __shared__ __align__(8) float2 s_gc[2][16][32];  // 8 KB: zero bank conflicts (64-bit LDS/STS across 32 lanes)

    uint32_t gn[16], an[16];
    const long nbase = (long)n * 16 * 32 + lane;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        gn[m] = ah::ld_h2<H>(G, nbase + (long)m * 32);
        an[m] = ah::ld_h2<H>(A, nbase + (long)m * 32);
    }
    const float2* X2 = reinterpret_cast<const float2*>(X);
    const float2* Xb2 = reinterpret_cast<const float2*>(Xb);
    const uint4* Ck4 = reinterpret_cast<const uint4*>(C_kh);
    const float2 xn = X2[(long)n * 32 + lane];
    const float2 xbn = XB ? Xb2[(long)n * 32 + lane] : make_float2(0.f, 0.f);
    float2 xd_n = make_float2(0.f, 0.f);

    int prev_kidx = -1;
    auto flush_gc = [&](int pk) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const long base = ((long)pk * 4 + l) * 4 * 64 + 2 * lane;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float2 v = s_gc[grp][l * 4 + q][lane];
                atomicAdd(gR_k + base + q * 64, v.x);
                atomicAdd(gR_k + base + q * 64 + 1, v.y);
            }
        }
    };

    for (int e = e_beg; e < e_end; e += e_step) {
        int ps_e, kidx;
        long p, s;
        float u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            if (ps_e & 1) continue;
            kidx = em.z;
            if (kidx < 0) continue;
            p = ps_e >> 1;
            s = em.y;
            u = __int_as_float(em.w);
        } else {
            ps_e = ps[e];
            if (ps_e & 1) continue;
            p = ps_e >> 1;
            kidx = sp_idx[p];
            if (kidx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        const bool same_k = (kidx == prev_kidx);
        if (!same_k) {
            if (prev_kidx >= 0) flush_gc(prev_kidx);
            prev_kidx = kidx;
        }
        const float2 xs = X2[s * 32 + lane];
        const float2 xbs = XB ? Xb2[s * 32 + lane] : make_float2(0.f, 0.f);
        const long sbase = s * 16 * 32 + lane;
        const long cbase = (long)kidx * 4 * 32 + lane;
        const float u2 = u * u, u3 = u2 * u;
        const float ui = u * sp_invh, u2i = u2 * sp_invh;
        float2 xd_s = make_float2(0.f, 0.f);

        auto finish_l = [&](int l, float sgn2, float tvx, float tvy, float rvx, float rvy, float aynx, float ayny,
                            float aysx, float aysy) {
            float rx, ry;
            eval_c4_pair(Ck4[cbase + l * 32], u, rx, ry);
            xd_s.x += rx * aynx; xd_s.y += ry * ayny;
            xd_n.x += sgn2 * rx * aysx; xd_n.y += sgn2 * ry * aysy;
            rvx += aynx * xs.x + sgn2 * aysx * xn.x;
            rvy += ayny * xs.y + sgn2 * aysy * xn.y;
            const float2 g0 = make_float2(rvx, rvy);
            const float2 g1 = make_float2(u * rvx + sp_invh * tvx, u * rvy + sp_invh * tvy);
            const float2 g2 = make_float2(u2 * rvx + 2.f * ui * tvx, u2 * rvy + 2.f * ui * tvy);
            const float2 g3 = make_float2(u3 * rvx + 3.f * u2i * tvx, u3 * rvy + 3.f * u2i * tvy);
            if (same_k) {
                float2 p0 = s_gc[grp][l * 4 + 0][lane]; p0.x += g0.x; p0.y += g0.y; s_gc[grp][l * 4 + 0][lane] = p0;
                float2 p1 = s_gc[grp][l * 4 + 1][lane]; p1.x += g1.x; p1.y += g1.y; s_gc[grp][l * 4 + 1][lane] = p1;
                float2 p2 = s_gc[grp][l * 4 + 2][lane]; p2.x += g2.x; p2.y += g2.y; s_gc[grp][l * 4 + 2][lane] = p2;
                float2 p3 = s_gc[grp][l * 4 + 3][lane]; p3.x += g3.x; p3.y += g3.y; s_gc[grp][l * 4 + 3][lane] = p3;
            } else {
                s_gc[grp][l * 4 + 0][lane] = g0;
                s_gc[grp][l * 4 + 1][lane] = g1;
                s_gc[grp][l * 4 + 2][lane] = g2;
                s_gc[grp][l * 4 + 3][lane] = g3;
            }
        };
        auto accum_m = [&](int m, float ywv, float gyv, float yv, float sgn2, float& tvx, float& tvy, float& rvx,
                           float& rvy, float& aynx, float& ayny, float& aysx, float& aysy) {
            const float2 gnm = ah::unpack2<H>(gn[m]);
            const float2 anm = ah::unpack2<H>(an[m]);
            const float2 gs = ah::unpack2<H>(ah::ld_h2<H>(G, sbase + (long)m * 32));
            const float2 as = ah::unpack2<H>(ah::ld_h2<H>(A, sbase + (long)m * 32));
            const float gx = gnm.x * xs.x + sgn2 * gs.x * xn.x;
            const float gy = gnm.y * xs.y + sgn2 * gs.y * xn.y;
            tvx += gx * ywv; tvy += gy * ywv;
            rvx += gx * gyv; rvy += gy * gyv;
            if constexpr (XB) {
                rvx += (gnm.x * xbs.x + sgn2 * gs.x * xbn.x) * yv;
                rvy += (gnm.y * xbs.y + sgn2 * gs.y * xbn.y) * yv;
            }
            aynx += anm.x * yv; ayny += anm.y * yv;
            aysx += as.x * yv; aysy += as.y * yv;
        };

        {
            const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 0);
            const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 0);
            const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
            float tvx = 0.f, tvy = 0.f, rvx = 0.f, rvy = 0.f, aynx = 0.f, ayny = 0.f, aysx = 0.f, aysy = 0.f;
            accum_m(0, v0.x, v1.x, v2.x, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            finish_l(0, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);

            tvx = tvy = rvx = rvy = aynx = ayny = aysx = aysy = 0.f;
            accum_m(1, v0.y, v1.y, v2.y, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            accum_m(2, v0.z, v1.z, v2.z, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            accum_m(3, v0.w, v1.w, v2.w, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            finish_l(1, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
        }
        {
            float tvx = 0.f, tvy = 0.f, rvx = 0.f, rvy = 0.f, aynx = 0.f, ayny = 0.f, aysx = 0.f, aysy = 0.f;
            {
                const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 4);
                const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 4);
                const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                accum_m(4, v0.x, v1.x, v2.x, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_m(5, v0.y, v1.y, v2.y, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_m(6, v0.z, v1.z, v2.z, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_m(7, v0.w, v1.w, v2.w, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            }
            const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 8);
            const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 8);
            const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
            accum_m(8, v0.x, v1.x, v2.x, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            finish_l(2, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);

            tvx = tvy = rvx = rvy = aynx = ayny = aysx = aysy = 0.f;
            accum_m(9, v0.y, v1.y, v2.y, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            accum_m(10, v0.z, v1.z, v2.z, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            accum_m(11, v0.w, v1.w, v2.w, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            {
                const float4 v0c = *reinterpret_cast<const float4*>(YwP + 16L * p + 12);
                const float4 v1c = *reinterpret_cast<const float4*>(GyP + 16L * p + 12);
                const float4 v2c = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                accum_m(12, v0c.x, v1c.x, v2c.x, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_m(13, v0c.y, v1c.y, v2c.y, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_m(14, v0c.z, v1c.z, v2c.z, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_m(15, v0c.w, v1c.w, v2c.w, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            }
            finish_l(3, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
        }
        atomicAdd(Xd + s * 64 + 2 * lane, xd_s.x);
        atomicAdd(Xd + s * 64 + 2 * lane + 1, xd_s.y);
    }
    if (prev_kidx >= 0) flush_gc(prev_kidx);
    atomicAdd(Xd + (long)n * 64 + 2 * lane, xd_n.x);
    atomicAdd(Xd + (long)n * 64 + 2 * lane + 1, xd_n.y);
}

// ============================================================================ 2c. k_slot_xd_h2_smem (sopt=5)
// Wave 16: moves gn[16] & an[16] out of registers into 4 KB shared memory (shared across both warps of the node!)
// and avoids unrolling flush_gc -> cuts regs from 168 down to <=104 (0 spills!), doubling H2 occupancy while
// keeping 2x fewer G[s]/A[s] LSU ops (LDG.32 __half2) + 3x-6x fewer gC_k atomics (8 KB s_gc)!
template <typename H, bool XB, bool EMETA>
__global__ void __launch_bounds__(64, 8) k_slot_xd_h2_smem(
    const int* __restrict__ rptr, const int* __restrict__ rptr0, const int* __restrict__ snd,
    const int* __restrict__ ps, const int* __restrict__ sp_idx, const float* __restrict__ sp_u,
    const H* __restrict__ G, const H* __restrict__ A, const float* __restrict__ X, const float* __restrict__ Xb,
    const float* __restrict__ YwP, const float* __restrict__ GyP, const float* __restrict__ YP,
    const float* __restrict__ C_kh, float* __restrict__ gR_k, float* __restrict__ Xd,
    float sp_invh, int N, const int4* __restrict__ emeta, const int* __restrict__ norder = nullptr) {
    const int nn = blockIdx.x;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    const int lane = threadIdx.x & 31;
    const int grp = threadIdx.x >> 5;

    __shared__ uint32_t s_gn[16][32];
    __shared__ uint32_t s_an[16][32];
    __shared__ __align__(8) float2 s_gc[2][16][32];

    const long nbase = (long)n * 16 * 32 + lane;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int m = grp * 8 + k;
        s_gn[m][lane] = ah::ld_h2<H>(G, nbase + (long)m * 32);
        s_an[m][lane] = ah::ld_h2<H>(A, nbase + (long)m * 32);
    }
    __syncthreads();

    const int e0_n = rptr[n];
    const int e1_n = EMETA ? rptr0[n] : rptr[n + 1];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = EMETA ? (e0_n + (cnt_n * grp) / 2) : (e0_n + grp);
    const int e_end = EMETA ? (e0_n + (cnt_n * (grp + 1)) / 2) : e1_n;
    const int e_step = EMETA ? 1 : 2;
    if (e_beg >= e_end) return;

    const float2* X2 = reinterpret_cast<const float2*>(X);
    const float2* Xb2 = reinterpret_cast<const float2*>(Xb);
    const uint4* Ck4 = reinterpret_cast<const uint4*>(C_kh);
    const float2 xn = X2[(long)n * 32 + lane];
    const float2 xbn = XB ? Xb2[(long)n * 32 + lane] : make_float2(0.f, 0.f);
    float2 xd_n = make_float2(0.f, 0.f);

    int prev_kidx = -1;
    auto flush_gc = [&](int pk) {
#pragma unroll 1
        for (int l = 0; l < 4; ++l) {
            const long base = ((long)pk * 4 + l) * 4 * 64 + 2 * lane;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float2 v = s_gc[grp][l * 4 + q][lane];
                atomicAdd(gR_k + base + q * 64, v.x);
                atomicAdd(gR_k + base + q * 64 + 1, v.y);
            }
        }
    };

    for (int e = e_beg; e < e_end; e += e_step) {
        int ps_e, kidx;
        long p, s;
        float u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            if (ps_e & 1) continue;
            kidx = em.z;
            if (kidx < 0) continue;
            p = ps_e >> 1;
            s = em.y;
            u = __int_as_float(em.w);
        } else {
            ps_e = ps[e];
            if (ps_e & 1) continue;
            p = ps_e >> 1;
            kidx = sp_idx[p];
            if (kidx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        const bool same_k = (kidx == prev_kidx);
        if (!same_k) {
            if (prev_kidx >= 0) flush_gc(prev_kidx);
            prev_kidx = kidx;
        }
        const float2 xs = X2[s * 32 + lane];
        const float2 xbs = XB ? Xb2[s * 32 + lane] : make_float2(0.f, 0.f);
        const long sbase = s * 16 * 32 + lane;
        const long cbase = (long)kidx * 4 * 32 + lane;
        const float u2 = u * u, u3 = u2 * u;
        const float ui = u * sp_invh, u2i = u2 * sp_invh;
        float2 xd_s = make_float2(0.f, 0.f);

        auto finish_l = [&](int l, float sgn2, float tvx, float tvy, float rvx, float rvy, float aynx, float ayny,
                            float aysx, float aysy) {
            float rx, ry;
            eval_c4_pair(Ck4[cbase + l * 32], u, rx, ry);
            xd_s.x += rx * aynx; xd_s.y += ry * ayny;
            xd_n.x += sgn2 * rx * aysx; xd_n.y += sgn2 * ry * aysy;
            rvx += aynx * xs.x + sgn2 * aysx * xn.x;
            rvy += ayny * xs.y + sgn2 * aysy * xn.y;
            const float2 g0 = make_float2(rvx, rvy);
            const float2 g1 = make_float2(u * rvx + sp_invh * tvx, u * rvy + sp_invh * tvy);
            const float2 g2 = make_float2(u2 * rvx + 2.f * ui * tvx, u2 * rvy + 2.f * ui * tvy);
            const float2 g3 = make_float2(u3 * rvx + 3.f * u2i * tvx, u3 * rvy + 3.f * u2i * tvy);
            if (same_k) {
                float2 p0 = s_gc[grp][l * 4 + 0][lane]; p0.x += g0.x; p0.y += g0.y; s_gc[grp][l * 4 + 0][lane] = p0;
                float2 p1 = s_gc[grp][l * 4 + 1][lane]; p1.x += g1.x; p1.y += g1.y; s_gc[grp][l * 4 + 1][lane] = p1;
                float2 p2 = s_gc[grp][l * 4 + 2][lane]; p2.x += g2.x; p2.y += g2.y; s_gc[grp][l * 4 + 2][lane] = p2;
                float2 p3 = s_gc[grp][l * 4 + 3][lane]; p3.x += g3.x; p3.y += g3.y; s_gc[grp][l * 4 + 3][lane] = p3;
            } else {
                s_gc[grp][l * 4 + 0][lane] = g0;
                s_gc[grp][l * 4 + 1][lane] = g1;
                s_gc[grp][l * 4 + 2][lane] = g2;
                s_gc[grp][l * 4 + 3][lane] = g3;
            }
        };
        auto accum_m = [&](int m, float ywv, float gyv, float yv, float sgn2, float& tvx, float& tvy, float& rvx,
                           float& rvy, float& aynx, float& ayny, float& aysx, float& aysy) {
            const float2 gnm = ah::unpack2<H>(s_gn[m][lane]);
            const float2 anm = ah::unpack2<H>(s_an[m][lane]);
            const float2 gs = ah::unpack2<H>(ah::ld_h2<H>(G, sbase + (long)m * 32));
            const float2 as = ah::unpack2<H>(ah::ld_h2<H>(A, sbase + (long)m * 32));
            const float gx = gnm.x * xs.x + sgn2 * gs.x * xn.x;
            const float gy = gnm.y * xs.y + sgn2 * gs.y * xn.y;
            tvx += gx * ywv; tvy += gy * ywv;
            rvx += gx * gyv; rvy += gy * gyv;
            if constexpr (XB) {
                rvx += (gnm.x * xbs.x + sgn2 * gs.x * xbn.x) * yv;
                rvy += (gnm.y * xbs.y + sgn2 * gs.y * xbn.y) * yv;
            }
            aynx += anm.x * yv; ayny += anm.y * yv;
            aysx += as.x * yv; aysy += as.y * yv;
        };

        {
            const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 0);
            const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 0);
            const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
            float tvx = 0.f, tvy = 0.f, rvx = 0.f, rvy = 0.f, aynx = 0.f, ayny = 0.f, aysx = 0.f, aysy = 0.f;
            accum_m(0, v0.x, v1.x, v2.x, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            finish_l(0, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);

            tvx = tvy = rvx = rvy = aynx = ayny = aysx = aysy = 0.f;
            accum_m(1, v0.y, v1.y, v2.y, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            accum_m(2, v0.z, v1.z, v2.z, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            accum_m(3, v0.w, v1.w, v2.w, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            finish_l(1, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
        }
        {
            float tvx = 0.f, tvy = 0.f, rvx = 0.f, rvy = 0.f, aynx = 0.f, ayny = 0.f, aysx = 0.f, aysy = 0.f;
            {
                const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 4);
                const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 4);
                const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                accum_m(4, v0.x, v1.x, v2.x, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_m(5, v0.y, v1.y, v2.y, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_m(6, v0.z, v1.z, v2.z, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_m(7, v0.w, v1.w, v2.w, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            }
            const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 8);
            const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 8);
            const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
            accum_m(8, v0.x, v1.x, v2.x, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            finish_l(2, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);

            tvx = tvy = rvx = rvy = aynx = ayny = aysx = aysy = 0.f;
            accum_m(9, v0.y, v1.y, v2.y, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            accum_m(10, v0.z, v1.z, v2.z, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            accum_m(11, v0.w, v1.w, v2.w, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            {
                const float4 v0c = *reinterpret_cast<const float4*>(YwP + 16L * p + 12);
                const float4 v1c = *reinterpret_cast<const float4*>(GyP + 16L * p + 12);
                const float4 v2c = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                accum_m(12, v0c.x, v1c.x, v2c.x, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_m(13, v0c.y, v1c.y, v2c.y, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_m(14, v0c.z, v1c.z, v2c.z, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_m(15, v0c.w, v1c.w, v2c.w, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            }
            finish_l(3, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
        }
        atomicAdd(Xd + s * 64 + 2 * lane, xd_s.x);
        atomicAdd(Xd + s * 64 + 2 * lane + 1, xd_s.y);
    }
    if (prev_kidx >= 0) flush_gc(prev_kidx);
    atomicAdd(Xd + (long)n * 64 + 2 * lane, xd_n.x);
    atomicAdd(Xd + (long)n * 64 + 2 * lane + 1, xd_n.y);
}

// ============================================================================ 2d. k_slot_xd_opt_v2 (sopt=6, Wave 17!)
// ES=1 (64 threads = 1 node/block):
//   - Half G[n]/A[n]/X[n]/Xb[n] prolog loads & half Xd[n] atomics vs ES=2 (sopt=3)!
//   - 2x longer kidx-sorted edge runs -> fewer gR_k atomic flushes!
//   - 2 KB s_ck[4][64] smem polynomial cache (0 extra regs!) -> eliminates ~75% of Ck2 L1/L2 loads!
//   - Batched 4-at-a-time G[s]/A[s] loads -> higher MLP to hide sb=65% L2 latency!
//   - Only 6 KB smem/block, 0 barriers, <=96 regs -> 10 blk/SM (20 warps/SM, 0 spills on both sm_89 and sm_120!).
template <typename H, bool XB, bool EMETA>
__global__ void __launch_bounds__(64, 10) k_slot_xd_opt_v2(
    const int* __restrict__ rptr, const int* __restrict__ rptr0, const int* __restrict__ snd,
    const int* __restrict__ ps, const int* __restrict__ sp_idx, const float* __restrict__ sp_u,
    const H* __restrict__ G, const H* __restrict__ A, const float* __restrict__ X, const float* __restrict__ Xb,
    const float* __restrict__ YwP, const float* __restrict__ GyP, const float* __restrict__ YP,
    const float* __restrict__ C_kh, float* __restrict__ gR_k, float* __restrict__ Xd,
    float sp_invh, int N, const int4* __restrict__ emeta, const int* __restrict__ norder = nullptr) {
    const int nn = blockIdx.x;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    const int c = threadIdx.x;  // 0..63

    const int e_beg = rptr[n];
    const int e_end = EMETA ? rptr0[n] : rptr[n + 1];
    if (e_beg >= e_end) return;

    __shared__ float s_gc[16][64];   // 4 KB (100% thread-private on c -> 0 barriers, 0 bank conflicts!)
    __shared__ uint2 s_ck[4][64];    // 2 KB cached cubic coefficients for current kidx (0 extra regs!)

    uint32_t gn2[8], an2[8];
    const long nbase = (long)n * 16 * 64 + c;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        gn2[k] = ah::pack2_raw<H>(G[nbase + (long)(2 * k) * 64], G[nbase + (long)(2 * k + 1) * 64]);
        an2[k] = ah::pack2_raw<H>(A[nbase + (long)(2 * k) * 64], A[nbase + (long)(2 * k + 1) * 64]);
    }
    const float xn = X[(long)n * 64 + c];
    const float xbn = XB ? Xb[(long)n * 64 + c] : 0.f;
    float xd_n = 0.f;
    const uint2* Ck2 = reinterpret_cast<const uint2*>(C_kh);

    int prev_kidx = -1;
    auto flush_gc = [&](int pk) {
#pragma unroll 1
        for (int l = 0; l < 4; ++l) {
            const long base = ((long)pk * 4 + l) * 4 * 64 + c;
#pragma unroll
            for (int q = 0; q < 4; ++q) atomicAdd(gR_k + base + q * 64, s_gc[l * 4 + q][c]);
        }
    };

    for (int e = e_beg; e < e_end; ++e) {
        int ps_e, kidx;
        long p, s;
        float u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            if (ps_e & 1) continue;
            kidx = em.z;
            if (kidx < 0) continue;
            p = ps_e >> 1;
            s = em.y;
            u = __int_as_float(em.w);
        } else {
            ps_e = ps[e];
            if (ps_e & 1) continue;
            p = ps_e >> 1;
            kidx = sp_idx[p];
            if (kidx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        const bool same_k = (kidx == prev_kidx);
        if (!same_k) {
            if (prev_kidx >= 0) flush_gc(prev_kidx);
            const long cbase = (long)kidx * 4 * 64 + c;
#pragma unroll
            for (int l = 0; l < 4; ++l) s_ck[l][c] = Ck2[cbase + l * 64];
            prev_kidx = kidx;
        }
        const float xs = X[s * 64 + c];
        const float xbs = XB ? Xb[s * 64 + c] : 0.f;
        const long sbase = s * 16 * 64 + c;
        const float u2 = u * u, u3 = u2 * u;
        const float ui = u * sp_invh, u2i = u2 * sp_invh;
        float xd_s = 0.f;

        auto finish_l = [&](int l, float sgn2, float tv, float rv, float ayn, float ays) {
            const float rval = eval_c4_s(s_ck[l][c], u);
            xd_s += rval * ayn;
            xd_n += sgn2 * rval * ays;
            rv += ayn * xs + sgn2 * ays * xn;
            const float g0 = rv;
            const float g1 = u * rv + sp_invh * tv;
            const float g2 = u2 * rv + 2.f * ui * tv;
            const float g3 = u3 * rv + 3.f * u2i * tv;
            if (same_k) {
                s_gc[l * 4 + 0][c] += g0;
                s_gc[l * 4 + 1][c] += g1;
                s_gc[l * 4 + 2][c] += g2;
                s_gc[l * 4 + 3][c] += g3;
            } else {
                s_gc[l * 4 + 0][c] = g0;
                s_gc[l * 4 + 1][c] = g1;
                s_gc[l * 4 + 2][c] = g2;
                s_gc[l * 4 + 3][c] = g3;
            }
        };
        auto accum_v = [&](float gnm, float anm, float gs, float as, float ywv, float gyv, float yv, float xns,
                           float xbns, float& tv, float& rv, float& ayn, float& ays) {
            const float gc = gnm * xs + gs * xns;
            tv += gc * ywv;
            rv += gc * gyv;
            if constexpr (XB) rv += (gnm * xbs + gs * xbns) * yv;
            ayn += anm * yv;
            ays += as * yv;
        };

        // Chunk 0 (m = 0..3): batch-issue 4x G[s] and 4x A[s] loads upfront for MLP!
        {
            const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 0);
            const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 0);
            const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
            const float gs0 = ah::ld_a16<float>(G, sbase + 0L * 64), as0 = ah::ld_a16<float>(A, sbase + 0L * 64);
            const float gs1 = ah::ld_a16<float>(G, sbase + 1L * 64), as1 = ah::ld_a16<float>(A, sbase + 1L * 64);
            const float gs2 = ah::ld_a16<float>(G, sbase + 2L * 64), as2 = ah::ld_a16<float>(A, sbase + 2L * 64);
            const float gs3 = ah::ld_a16<float>(G, sbase + 3L * 64), as3 = ah::ld_a16<float>(A, sbase + 3L * 64);
            const float2 g01 = ah::unpack2<H>(gn2[0]), a01 = ah::unpack2<H>(an2[0]);
            const float2 g23 = ah::unpack2<H>(gn2[1]), a23 = ah::unpack2<H>(an2[1]);
            float tv = 0.f, rv = 0.f, ayn = 0.f, ays = 0.f;
            accum_v(g01.x, a01.x, gs0, as0, v0.x, v1.x, v2.x, xn, xbn, tv, rv, ayn, ays);
            finish_l(0, 1.f, tv, rv, ayn, ays);
            tv = 0.f; rv = 0.f; ayn = 0.f; ays = 0.f;
            accum_v(g01.y, a01.y, gs1, as1, v0.y, v1.y, v2.y, -xn, -xbn, tv, rv, ayn, ays);
            accum_v(g23.x, a23.x, gs2, as2, v0.z, v1.z, v2.z, -xn, -xbn, tv, rv, ayn, ays);
            accum_v(g23.y, a23.y, gs3, as3, v0.w, v1.w, v2.w, -xn, -xbn, tv, rv, ayn, ays);
            finish_l(1, -1.f, tv, rv, ayn, ays);
        }
        // Chunks 1..3 (m = 4..15): batch-issue 4x G[s] + 4x A[s] per chunk!
        {
            float tv = 0.f, rv = 0.f, ayn = 0.f, ays = 0.f;
            {
                const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 4);
                const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 4);
                const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                const float gs4 = ah::ld_a16<float>(G, sbase + 4L * 64), as4 = ah::ld_a16<float>(A, sbase + 4L * 64);
                const float gs5 = ah::ld_a16<float>(G, sbase + 5L * 64), as5 = ah::ld_a16<float>(A, sbase + 5L * 64);
                const float gs6 = ah::ld_a16<float>(G, sbase + 6L * 64), as6 = ah::ld_a16<float>(A, sbase + 6L * 64);
                const float gs7 = ah::ld_a16<float>(G, sbase + 7L * 64), as7 = ah::ld_a16<float>(A, sbase + 7L * 64);
                const float2 g45 = ah::unpack2<H>(gn2[2]), a45 = ah::unpack2<H>(an2[2]);
                const float2 g67 = ah::unpack2<H>(gn2[3]), a67 = ah::unpack2<H>(an2[3]);
                accum_v(g45.x, a45.x, gs4, as4, v0.x, v1.x, v2.x, xn, xbn, tv, rv, ayn, ays);
                accum_v(g45.y, a45.y, gs5, as5, v0.y, v1.y, v2.y, xn, xbn, tv, rv, ayn, ays);
                accum_v(g67.x, a67.x, gs6, as6, v0.z, v1.z, v2.z, xn, xbn, tv, rv, ayn, ays);
                accum_v(g67.y, a67.y, gs7, as7, v0.w, v1.w, v2.w, xn, xbn, tv, rv, ayn, ays);
            }
            {
                const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 8);
                const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 8);
                const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
                const float gs8 = ah::ld_a16<float>(G, sbase + 8L * 64), as8 = ah::ld_a16<float>(A, sbase + 8L * 64);
                const float gs9 = ah::ld_a16<float>(G, sbase + 9L * 64), as9 = ah::ld_a16<float>(A, sbase + 9L * 64);
                const float gsA = ah::ld_a16<float>(G, sbase + 10L * 64), asA = ah::ld_a16<float>(A, sbase + 10L * 64);
                const float gsB = ah::ld_a16<float>(G, sbase + 11L * 64), asB = ah::ld_a16<float>(A, sbase + 11L * 64);
                const float2 g89 = ah::unpack2<H>(gn2[4]), a89 = ah::unpack2<H>(an2[4]);
                const float2 gAB = ah::unpack2<H>(gn2[5]), aAB = ah::unpack2<H>(an2[5]);
                accum_v(g89.x, a89.x, gs8, as8, v0.x, v1.x, v2.x, xn, xbn, tv, rv, ayn, ays);
                finish_l(2, 1.f, tv, rv, ayn, ays);
                tv = 0.f; rv = 0.f; ayn = 0.f; ays = 0.f;
                accum_v(g89.y, a89.y, gs9, as9, v0.y, v1.y, v2.y, -xn, -xbn, tv, rv, ayn, ays);
                accum_v(gAB.x, aAB.x, gsA, asA, v0.z, v1.z, v2.z, -xn, -xbn, tv, rv, ayn, ays);
                accum_v(gAB.y, aAB.y, gsB, asB, v0.w, v1.w, v2.w, -xn, -xbn, tv, rv, ayn, ays);
            }
            {
                const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 12);
                const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 12);
                const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                const float gsC = ah::ld_a16<float>(G, sbase + 12L * 64), asC = ah::ld_a16<float>(A, sbase + 12L * 64);
                const float gsD = ah::ld_a16<float>(G, sbase + 13L * 64), asD = ah::ld_a16<float>(A, sbase + 13L * 64);
                const float gsE = ah::ld_a16<float>(G, sbase + 14L * 64), asE = ah::ld_a16<float>(A, sbase + 14L * 64);
                const float gsF = ah::ld_a16<float>(G, sbase + 15L * 64), asF = ah::ld_a16<float>(A, sbase + 15L * 64);
                const float2 gCD = ah::unpack2<H>(gn2[6]), aCD = ah::unpack2<H>(an2[6]);
                const float2 gEF = ah::unpack2<H>(gn2[7]), aEF = ah::unpack2<H>(an2[7]);
                accum_v(gCD.x, aCD.x, gsC, asC, v0.x, v1.x, v2.x, -xn, -xbn, tv, rv, ayn, ays);
                accum_v(gCD.y, aCD.y, gsD, asD, v0.y, v1.y, v2.y, -xn, -xbn, tv, rv, ayn, ays);
                accum_v(gEF.x, aEF.x, gsE, asE, v0.z, v1.z, v2.z, -xn, -xbn, tv, rv, ayn, ays);
                accum_v(gEF.y, aEF.y, gsF, asF, v0.w, v1.w, v2.w, -xn, -xbn, tv, rv, ayn, ays);
            }
            finish_l(3, -1.f, tv, rv, ayn, ays);
        }
        atomicAdd(Xd + s * 64 + c, xd_s);
    }
    if (prev_kidx >= 0) flush_gc(prev_kidx);
    atomicAdd(Xd + (long)n * 64 + c, xd_n);
}

// ============================================================================ 2e. k_slot_xd_opt_v3 (sopt=7, Wave 18 + Wave 20 ZIP_GA!)
// Combines sopt=3's ES=2 (128 threads/node -> full b1 & b4 SM saturation!) with sopt=6's 4 KB s_ck[2][4][64] smem
// polynomial cache (0 extra regs -> cuts Ck2 loads ~75%) + batched 4-at-a-time G[s]/A[s] loads (hides sb=65% L2!).
// Wave 20 ZIP_GA=true: reads fused interleaved GA[N,16,64] (uint32_t __half2(G,A)) via 16x LDG.32 (100% 128B
// coalesced, HALF the LSU ops & L1TEX wavefronts vs 32x LDG.16!). 12 KB smem/blk, 0 barriers, <=96 regs (0 spills!).
template <typename H, bool XB, bool EMETA, bool ZIP_GA = false, int Y3_MODE = 0, bool SHELL = false,
          bool T0S_DIR = false, int GA_MODE = 0, bool LDCG = false>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
__global__ void __launch_bounds__(128, 6) k_slot_xd_opt_v3(
#else
__global__ void __launch_bounds__(128, 5) k_slot_xd_opt_v3(
#endif
    const int* __restrict__ rptr, const int* __restrict__ rptr0, const int* __restrict__ snd,
    const int* __restrict__ ps, const int* __restrict__ sp_idx, const float* __restrict__ sp_u,
    const H* __restrict__ G, const H* __restrict__ A, const float* __restrict__ X, const float* __restrict__ Xb,
    const float* __restrict__ YwP, const float* __restrict__ GyP, const float* __restrict__ YP,
    const float* __restrict__ C_kh, float* __restrict__ gR_k, float* __restrict__ Xd,
    float sp_invh, int N, const int4* __restrict__ emeta, const int* __restrict__ norder = nullptr,
    const uint32_t* __restrict__ GA = nullptr, const float* __restrict__ Y3P = nullptr,
    int kcut2 = 999999, int kcut3 = 999999,
    const int* __restrict__ spc = nullptr, const float* __restrict__ T0 = nullptr,
    float* __restrict__ T0s = nullptr, int S = 0) {
    const int nn = blockIdx.x;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    const int e0_n = rptr[n];
    const int e1_n = EMETA ? rptr0[n] : rptr[n + 1];
    const int cnt_n = e1_n - e0_n;
    if (cnt_n <= 0) return;

    const int c = threadIdx.x & 63;
    const int grp = threadIdx.x >> 6;  // 0 or 1

    __shared__ float s_gc[2][16][64];  // 8 KB
    __shared__ uint2 s_ck[2][4][64];   // 4 KB cached cubic coeffs per grp (0 extra regs, 0 barriers!)
    __shared__ float s_t0r[T0S_DIR ? 4 : 1][T0S_DIR ? 64 : 1];       // 1 KB when T0S_DIR (0 extra regs!)
    __shared__ float s_t0s[T0S_DIR ? 2 : 1][T0S_DIR ? 4 : 1][T0S_DIR ? 64 : 1];  // 2 KB when T0S_DIR (0 extra regs!)
    if constexpr (T0S_DIR) {
        if (grp == 0) {
            for (int sp = 0; sp < S; ++sp) s_t0r[sp][c] = T0[sp * 64 + c];
        }
        for (int sp = 0; sp < 4; ++sp) s_t0s[grp][sp][c] = 0.f;
        __syncthreads();
    }

    const int e_beg = EMETA ? (e0_n + (cnt_n * grp) / 2) : (e0_n + grp);
    const int e_end = EMETA ? (e0_n + (cnt_n * (grp + 1)) / 2) : e1_n;
    const int e_step = EMETA ? 1 : 2;
    if (e_beg >= e_end) return;

    uint32_t gan[16];
    const long nbase = (long)n * 16 * 64 + c;
    const uint16_t* GA8 = (GA_MODE == 2) ? reinterpret_cast<const uint16_t*>(GA) : nullptr;
    const float2* GAs = (GA_MODE == 2) ? reinterpret_cast<const float2*>(GA + 8L * N * 64) : nullptr;
    if constexpr (GA_MODE == 2) {
#pragma unroll
        for (int m = 0; m < 16; ++m) gan[m] = ah::pack2_raw<H>(G[nbase + (long)m * 64], A[nbase + (long)m * 64]);
    } else if constexpr (ZIP_GA) {
#pragma unroll
        for (int m = 0; m < 16; ++m) gan[m] = GA[nbase + (long)m * 64];
    } else {
#pragma unroll
        for (int m = 0; m < 16; ++m) gan[m] = ah::pack2_raw<H>(G[nbase + (long)m * 64], A[nbase + (long)m * 64]);
    }
    const int sp_n = T0S_DIR ? spc[n] : 0;
    const float xn = T0S_DIR ? s_t0r[sp_n][c] : X[(long)n * 64 + c];
    const float xbn = XB ? Xb[(long)n * 64 + c] : 0.f;
    float xd_n = 0.f;
    const uint2* Ck2 = reinterpret_cast<const uint2*>(C_kh);

    int prev_kidx = -1;
    auto flush_gc = [&](int pk) {
#pragma unroll 1
        for (int l = 0; l < 4; ++l) {
            if constexpr (SHELL) {
                if ((l == 2 && pk >= kcut2) || (l == 3 && pk >= kcut3)) continue;
            }
            const long base = ((long)pk * 4 + l) * 4 * 64 + c;
#pragma unroll
            for (int q = 0; q < 4; ++q) atomicAdd(gR_k + base + q * 64, s_gc[grp][l * 4 + q][c]);
        }
    };

    for (int e = e_beg; e < e_end; e += e_step) {
        int ps_e, kidx;
        long p, s;
        float u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            if (ps_e & 1) continue;
            kidx = em.z;
            if (kidx < 0) continue;
            p = ps_e >> 1;
            s = em.y;
            u = __int_as_float(em.w);
        } else {
            ps_e = ps[e];
            if (ps_e & 1) continue;
            p = ps_e >> 1;
            kidx = sp_idx[p];
            if (kidx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        const bool do_l2 = !SHELL || (kidx < kcut2);
        const bool do_l3 = !SHELL || (kidx < kcut3);
        const bool same_k = (kidx == prev_kidx);
        if (!same_k) {
            if (prev_kidx >= 0) flush_gc(prev_kidx);
            const long cbase = (long)kidx * 4 * 64 + c;
            s_ck[grp][0][c] = Ck2[cbase + 0 * 64];
            s_ck[grp][1][c] = Ck2[cbase + 1 * 64];
            if (do_l2) s_ck[grp][2][c] = Ck2[cbase + 2 * 64];
            if (do_l3) s_ck[grp][3][c] = Ck2[cbase + 3 * 64];
            prev_kidx = kidx;
        }
        const int sp_s = T0S_DIR ? spc[s] : 0;
        const float xs = T0S_DIR ? s_t0r[sp_s][c] : X[s * 64 + c];
        const float xbs = XB ? Xb[s * 64 + c] : 0.f;
        const long sbase = s * 16 * 64 + c;
        const float u2 = u * u, u3 = u2 * u;
        const float ui = u * sp_invh, u2i = u2 * sp_invh;
        float xd_s = 0.f;

        auto finish_l = [&](int l, float sgn2, float tv, float rv, float ayn, float ays) {
            const float rval = eval_c4_s(s_ck[grp][l][c], u);
            xd_s += rval * ayn;
            xd_n += sgn2 * rval * ays;
            rv += ayn * xs + sgn2 * ays * xn;
            const float g0 = rv;
            const float g1 = u * rv + sp_invh * tv;
            const float g2 = u2 * rv + 2.f * ui * tv;
            const float g3 = u3 * rv + 3.f * u2i * tv;
            if (same_k) {
                s_gc[grp][l * 4 + 0][c] += g0;
                s_gc[grp][l * 4 + 1][c] += g1;
                s_gc[grp][l * 4 + 2][c] += g2;
                s_gc[grp][l * 4 + 3][c] += g3;
            } else {
                s_gc[grp][l * 4 + 0][c] = g0;
                s_gc[grp][l * 4 + 1][c] = g1;
                s_gc[grp][l * 4 + 2][c] = g2;
                s_gc[grp][l * 4 + 3][c] = g3;
            }
        };
        if constexpr (GA_MODE > 0 && Y3_MODE == 3 && std::is_same<H, __half>::value) {
            // Wave 45: Interleaved Y3P_H (yhp[0..3]=half2(Yw[m], Y[m]), yhp[4..5]=half2(Gy[2k], Gy[2k+1]))
            // with native __low2half2/__high2half2 -> deletes 48 PRMTs/edge in L1 & 32 PRMTs/edge in L0!
            const __half2 z_h = __float2half2_rn(0.f);
            auto load_ga4_h = [&](long off, uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3) {
                if constexpr (LDCG) {
                    w0 = __ldcg(GA + off + 0L * 64);
                    w1 = __ldcg(GA + off + 1L * 64);
                    w2 = __ldcg(GA + off + 2L * 64);
                    w3 = __ldcg(GA + off + 3L * 64);
                } else {
                    w0 = GA[off + 0L * 64];
                    w1 = GA[off + 1L * 64];
                    w2 = GA[off + 2L * 64];
                    w3 = GA[off + 3L * 64];
                }
            };
            auto accum_h2_i = [&](uint32_t gn_w, uint32_t gs_u, uint32_t ywy_u, uint32_t gy_pair, bool hi,
                                  __half2& an, __half2& as, __half2& gns_gy, __half2& gns_y) {
                const __half2 gn_h = *reinterpret_cast<const __half2*>(&gn_w);
                const __half2 gs_h = *reinterpret_cast<const __half2*>(&gs_u);
                const __half2 ywy_h = *reinterpret_cast<const __half2*>(&ywy_u);
                const __half2 gy_p = *reinterpret_cast<const __half2*>(&gy_pair);
                const __half2 gy2_h = hi ? __high2half2(gy_p) : __low2half2(gy_p);
                const uint32_t gns_u = __byte_perm(gn_w, gs_u, 0x5410u);
                const __half2 gns_h = *reinterpret_cast<const __half2*>(&gns_u);
                an = __hfma2(gn_h, ywy_h, an);
                as = __hfma2(gs_h, ywy_h, as);
                gns_gy = __hfma2(gns_h, gy2_h, gns_gy);
                if constexpr (XB) {
                    gns_y = __hfma2(gns_h, __high2half2(ywy_h), gns_y);
                }
            };
            auto finish_h2_l = [&](int l, float sgn2, __half2 an, __half2 as, __half2 gns_gy, __half2 gns_y) {
                const float2 fan = __half22float2(an);
                const float2 fas = __half22float2(as);
                const float2 fgy = __half22float2(gns_gy);
                const float xns = sgn2 * xn;
                const float tv = fan.x * xs + fas.x * xns;
                float rv = fgy.x * xs + fgy.y * xns;
                if constexpr (XB) {
                    const float2 fy = __half22float2(gns_y);
                    rv += fy.x * xbs + fy.y * (sgn2 * xbn);
                }
                finish_l(l, sgn2, tv, rv, fan.y, fas.y);
            };
            const uint4* yhp = reinterpret_cast<const uint4*>(Y3P) + 6L * p;
            {
                const uint4 q0 = yhp[0];
                const uint2 g0 = *(reinterpret_cast<const uint2*>(yhp + 4) + 0);
                uint32_t w0, w1, w2, w3;
                load_ga4_h(sbase + 0L * 64, w0, w1, w2, w3);
                __half2 an = z_h, as = z_h, gns_gy = z_h, gns_y = z_h;
                accum_h2_i(gan[0], w0, q0.x, g0.x, false, an, as, gns_gy, gns_y);
                finish_h2_l(0, 1.f, an, as, gns_gy, gns_y);
                an = z_h; as = z_h; gns_gy = z_h; gns_y = z_h;
                accum_h2_i(gan[1], w1, q0.y, g0.x, true,  an, as, gns_gy, gns_y);
                accum_h2_i(gan[2], w2, q0.z, g0.y, false, an, as, gns_gy, gns_y);
                accum_h2_i(gan[3], w3, q0.w, g0.y, true,  an, as, gns_gy, gns_y);
                finish_h2_l(1, -1.f, an, as, gns_gy, gns_y);
            }
            if (do_l2 || do_l3) {
                const uint4 q2 = yhp[2];
                const uint2 g2 = *(reinterpret_cast<const uint2*>(yhp + 4) + 2);
                uint32_t w8 = 0, w9 = 0, wA = 0, wB = 0;
                if (do_l3) {
                    load_ga4_h(sbase + 8L * 64, w8, w9, wA, wB);
                } else {
                    if constexpr (LDCG) w8 = __ldcg(GA + sbase + 8L * 64);
                    else w8 = GA[sbase + 8L * 64];
                }
                if (do_l2) {
                    __half2 an = z_h, as = z_h, gns_gy = z_h, gns_y = z_h;
                    {
                        const uint4 q1 = yhp[1];
                        const uint2 g1 = *(reinterpret_cast<const uint2*>(yhp + 4) + 1);
                        uint32_t w4, w5, w6, w7;
                        load_ga4_h(sbase + 4L * 64, w4, w5, w6, w7);
                        accum_h2_i(gan[4], w4, q1.x, g1.x, false, an, as, gns_gy, gns_y);
                        accum_h2_i(gan[5], w5, q1.y, g1.x, true,  an, as, gns_gy, gns_y);
                        accum_h2_i(gan[6], w6, q1.z, g1.y, false, an, as, gns_gy, gns_y);
                        accum_h2_i(gan[7], w7, q1.w, g1.y, true,  an, as, gns_gy, gns_y);
                    }
                    accum_h2_i(gan[8], w8, q2.x, g2.x, false, an, as, gns_gy, gns_y);
                    finish_h2_l(2, 1.f, an, as, gns_gy, gns_y);
                }
                if (do_l3) {
                    __half2 an = z_h, as = z_h, gns_gy = z_h, gns_y = z_h;
                    accum_h2_i(gan[9],  w9, q2.y, g2.x, true,  an, as, gns_gy, gns_y);
                    accum_h2_i(gan[10], wA, q2.z, g2.y, false, an, as, gns_gy, gns_y);
                    accum_h2_i(gan[11], wB, q2.w, g2.y, true,  an, as, gns_gy, gns_y);
                    {
                        const uint4 q3 = yhp[3];
                        const uint2 g3 = *(reinterpret_cast<const uint2*>(yhp + 4) + 3);
                        uint32_t wC, wD, wE, wF;
                        load_ga4_h(sbase + 12L * 64, wC, wD, wE, wF);
                        accum_h2_i(gan[12], wC, q3.x, g3.x, false, an, as, gns_gy, gns_y);
                        accum_h2_i(gan[13], wD, q3.y, g3.x, true,  an, as, gns_gy, gns_y);
                        accum_h2_i(gan[14], wE, q3.z, g3.y, false, an, as, gns_gy, gns_y);
                        accum_h2_i(gan[15], wF, q3.w, g3.y, true,  an, as, gns_gy, gns_y);
                    }
                    finish_h2_l(3, -1.f, an, as, gns_gy, gns_y);
                }
            }
        } else if constexpr (GA_MODE > 0 && Y3_MODE == 2 && std::is_same<H, __half>::value) {
            // Wave 27: Factored native __half2 HFMA2 inner loop!
            //   - GA_MODE == 1: exact FP16 GA[s] (0 F2F unpacks in m-loop, 3x fewer ALU ops!)
            //   - GA_MODE == 2: per-(node, l) FP8 E4M3 GA8[s] (QMAX=32) -> 2x less L2 traffic + only 4 GAs loads/edge!
            const __half2 z_h = __float2half2_rn(0.f);
            auto load_ga4_h = [&](long off, uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3) {
                if constexpr (GA_MODE == 2) {
                    w0 = GA8[off + 0L * 64];
                    w1 = GA8[off + 1L * 64];
                    w2 = GA8[off + 2L * 64];
                    w3 = GA8[off + 3L * 64];
                } else if constexpr (LDCG) {
                    w0 = __ldcg(GA + off + 0L * 64);
                    w1 = __ldcg(GA + off + 1L * 64);
                    w2 = __ldcg(GA + off + 2L * 64);
                    w3 = __ldcg(GA + off + 3L * 64);
                } else {
                    w0 = GA[off + 0L * 64];
                    w1 = GA[off + 1L * 64];
                    w2 = GA[off + 2L * 64];
                    w3 = GA[off + 3L * 64];
                }
            };
            auto accum_h2 = [&](uint32_t gn_w, uint32_t gs_raw, uint32_t yw_pair, uint32_t gy_pair, uint32_t y_pair,
                                bool hi, __half2& an, __half2& as, __half2& gns_gy, __half2& gns_y) {
                const __half2 gn_h = *reinterpret_cast<const __half2*>(&gn_w);
                __half2 gs_h;
                uint32_t gs_u;
                if constexpr (GA_MODE == 2) {
                    gs_h = ah::unpack_ga_fp8_h2(static_cast<uint16_t>(gs_raw));
                    gs_u = *reinterpret_cast<const uint32_t*>(&gs_h);
                } else {
                    gs_u = gs_raw;
                    gs_h = *reinterpret_cast<const __half2*>(&gs_u);
                }
                const uint32_t ywy_u = __byte_perm(yw_pair, y_pair, hi ? 0x7632u : 0x5410u);
                const uint32_t gy2_u = __byte_perm(gy_pair, gy_pair, hi ? 0x3232u : 0x1010u);
                const uint32_t gns_u = __byte_perm(gn_w, gs_u, 0x5410u);
                const __half2 ywy_h = *reinterpret_cast<const __half2*>(&ywy_u);
                const __half2 gy2_h = *reinterpret_cast<const __half2*>(&gy2_u);
                const __half2 gns_h = *reinterpret_cast<const __half2*>(&gns_u);
                an = __hfma2(gn_h, ywy_h, an);
                as = __hfma2(gs_h, ywy_h, as);
                gns_gy = __hfma2(gns_h, gy2_h, gns_gy);
                if constexpr (XB) {
                    const uint32_t y2_u = __byte_perm(y_pair, y_pair, hi ? 0x3232u : 0x1010u);
                    gns_y = __hfma2(gns_h, *reinterpret_cast<const __half2*>(&y2_u), gns_y);
                }
            };
            auto finish_h2_l = [&](int l, float sgn2, __half2 an, __half2 as, __half2 gns_gy, __half2 gns_y) {
                const float2 fan = __half22float2(an);
                const float2 fas = __half22float2(as);
                const float2 fgy = __half22float2(gns_gy);
                float sc_g = 1.f, sc_a = 1.f;
                if constexpr (GA_MODE == 2) {
                    const float2 sc = GAs[s * 4 + l];
                    sc_g = sc.x; sc_a = sc.y;
                }
                const float xns = sgn2 * xn * sc_g;
                const float tv = fan.x * xs + fas.x * xns;
                float rv = fgy.x * xs + fgy.y * xns;
                if constexpr (XB) {
                    const float2 fy = __half22float2(gns_y);
                    rv += fy.x * xbs + fy.y * (sgn2 * xbn * sc_g);
                }
                finish_l(l, sgn2, tv, rv, fan.y, fas.y * sc_a);
            };
            const uint4* yhp = reinterpret_cast<const uint4*>(Y3P) + 6L * p;
            {
                const uint4 q0 = yhp[0];
                const uint2 y0 = *(reinterpret_cast<const uint2*>(yhp + 4) + 0);
                uint32_t w0, w1, w2, w3;
                load_ga4_h(sbase + 0L * 64, w0, w1, w2, w3);
                __half2 an = z_h, as = z_h, gns_gy = z_h, gns_y = z_h;
                accum_h2(gan[0], w0, q0.x, q0.z, y0.x, false, an, as, gns_gy, gns_y);
                finish_h2_l(0, 1.f, an, as, gns_gy, gns_y);
                an = z_h; as = z_h; gns_gy = z_h; gns_y = z_h;
                accum_h2(gan[1], w1, q0.x, q0.z, y0.x, true, an, as, gns_gy, gns_y);
                accum_h2(gan[2], w2, q0.y, q0.w, y0.y, false, an, as, gns_gy, gns_y);
                accum_h2(gan[3], w3, q0.y, q0.w, y0.y, true, an, as, gns_gy, gns_y);
                finish_h2_l(1, -1.f, an, as, gns_gy, gns_y);
            }
            if (do_l2 || do_l3) {
                const uint4 q2 = yhp[2];
                const uint2 y2 = *(reinterpret_cast<const uint2*>(yhp + 4) + 2);
                uint32_t w8 = 0, w9 = 0, wA = 0, wB = 0;
                if (do_l3) {
                    load_ga4_h(sbase + 8L * 64, w8, w9, wA, wB);
                } else {
                    if constexpr (GA_MODE == 2) w8 = GA8[sbase + 8L * 64];
                    else if constexpr (LDCG) w8 = __ldcg(GA + sbase + 8L * 64);
                    else w8 = GA[sbase + 8L * 64];
                }
                if (do_l2) {
                    __half2 an = z_h, as = z_h, gns_gy = z_h, gns_y = z_h;
                    {
                        const uint4 q1 = yhp[1];
                        const uint2 y1 = *(reinterpret_cast<const uint2*>(yhp + 4) + 1);
                        uint32_t w4, w5, w6, w7;
                        load_ga4_h(sbase + 4L * 64, w4, w5, w6, w7);
                        accum_h2(gan[4], w4, q1.x, q1.z, y1.x, false, an, as, gns_gy, gns_y);
                        accum_h2(gan[5], w5, q1.x, q1.z, y1.x, true, an, as, gns_gy, gns_y);
                        accum_h2(gan[6], w6, q1.y, q1.w, y1.y, false, an, as, gns_gy, gns_y);
                        accum_h2(gan[7], w7, q1.y, q1.w, y1.y, true, an, as, gns_gy, gns_y);
                    }
                    accum_h2(gan[8], w8, q2.x, q2.z, y2.x, false, an, as, gns_gy, gns_y);
                    finish_h2_l(2, 1.f, an, as, gns_gy, gns_y);
                }
                if (do_l3) {
                    __half2 an = z_h, as = z_h, gns_gy = z_h, gns_y = z_h;
                    accum_h2(gan[9], w9, q2.x, q2.z, y2.x, true, an, as, gns_gy, gns_y);
                    accum_h2(gan[10], wA, q2.y, q2.w, y2.y, false, an, as, gns_gy, gns_y);
                    accum_h2(gan[11], wB, q2.y, q2.w, y2.y, true, an, as, gns_gy, gns_y);
                    {
                        const uint4 q3 = yhp[3];
                        const uint2 y3 = *(reinterpret_cast<const uint2*>(yhp + 4) + 3);
                        uint32_t wC, wD, wE, wF;
                        load_ga4_h(sbase + 12L * 64, wC, wD, wE, wF);
                        accum_h2(gan[12], wC, q3.x, q3.z, y3.x, false, an, as, gns_gy, gns_y);
                        accum_h2(gan[13], wD, q3.x, q3.z, y3.x, true, an, as, gns_gy, gns_y);
                        accum_h2(gan[14], wE, q3.y, q3.w, y3.y, false, an, as, gns_gy, gns_y);
                        accum_h2(gan[15], wF, q3.y, q3.w, y3.y, true, an, as, gns_gy, gns_y);
                    }
                    finish_h2_l(3, -1.f, an, as, gns_gy, gns_y);
                }
            }
        } else {
        auto accum_raw = [&](int m, uint32_t gn_w, uint32_t gs_w, float ywv, float gyv, float yv, float xns, float xbns,
                             float& tv, float& rv, float& ayn, float& ays) {
            const float2 gn = ah::unpack2<H>(gn_w);
            const float2 gs = ah::unpack2<H>(gs_w);
            const float gc = gn.x * xs + gs.x * xns;
            tv += gc * ywv;
            rv += gc * gyv;
            if constexpr (XB) rv += (gn.x * xbs + gs.x * xbns) * yv;
            ayn += gn.y * yv;
            ays += gs.y * yv;
        };

        auto load_ga4_raw = [&](long off, uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3) {
            if constexpr (ZIP_GA) {
                w0 = GA[off + 0L * 64];
                w1 = GA[off + 1L * 64];
                w2 = GA[off + 2L * 64];
                w3 = GA[off + 3L * 64];
            } else {
                w0 = ah::pack2_raw<H>(G[off + 0L * 64], A[off + 0L * 64]);
                w1 = ah::pack2_raw<H>(G[off + 1L * 64], A[off + 1L * 64]);
                w2 = ah::pack2_raw<H>(G[off + 2L * 64], A[off + 2L * 64]);
                w3 = ah::pack2_raw<H>(G[off + 3L * 64], A[off + 3L * 64]);
            }
        };

        auto load_y4 = [&](int q, float4& v0, float4& v1, float4& v2) {
            if constexpr (Y3_MODE == 2) {
                const uint4* yhp = reinterpret_cast<const uint4*>(Y3P) + 6L * p;
                const uint4 qh = yhp[q];
                const float2 w01 = __half22float2(*reinterpret_cast<const __half2*>(&qh.x));
                const float2 w23 = __half22float2(*reinterpret_cast<const __half2*>(&qh.y));
                const float2 g01 = __half22float2(*reinterpret_cast<const __half2*>(&qh.z));
                const float2 g23 = __half22float2(*reinterpret_cast<const __half2*>(&qh.w));
                v0 = make_float4(w01.x, w01.y, w23.x, w23.y);
                v1 = make_float4(g01.x, g01.y, g23.x, g23.y);
                const uint2 yy = *(reinterpret_cast<const uint2*>(yhp + 4) + q);
                const float2 y01 = __half22float2(*reinterpret_cast<const __half2*>(&yy.x));
                const float2 y23 = __half22float2(*reinterpret_cast<const __half2*>(&yy.y));
                v2 = make_float4(y01.x, y01.y, y23.x, y23.y);
            } else if constexpr (Y3_MODE == 1) {
                const float* yp = Y3P + 48L * p + 12 * q;
                v0 = *reinterpret_cast<const float4*>(yp + 0);
                v1 = *reinterpret_cast<const float4*>(yp + 4);
                v2 = *reinterpret_cast<const float4*>(yp + 8);
            } else {
                v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 4 * q);
                v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 4 * q);
                v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 4 * q);
            }
        };

        {
            float4 v0, v1, v2;
            load_y4(0, v0, v1, v2);
            uint32_t w0, w1, w2, w3;
            load_ga4_raw(sbase + 0L * 64, w0, w1, w2, w3);
            float tv = 0.f, rv = 0.f, ayn = 0.f, ays = 0.f;
            accum_raw(0, gan[0], w0, v0.x, v1.x, v2.x, xn, xbn, tv, rv, ayn, ays);
            finish_l(0, 1.f, tv, rv, ayn, ays);
            tv = 0.f; rv = 0.f; ayn = 0.f; ays = 0.f;
            accum_raw(1, gan[1], w1, v0.y, v1.y, v2.y, -xn, -xbn, tv, rv, ayn, ays);
            accum_raw(2, gan[2], w2, v0.z, v1.z, v2.z, -xn, -xbn, tv, rv, ayn, ays);
            accum_raw(3, gan[3], w3, v0.w, v1.w, v2.w, -xn, -xbn, tv, rv, ayn, ays);
            finish_l(1, -1.f, tv, rv, ayn, ays);
        }
        if (do_l2 || do_l3) {
            float4 v0b, v1b, v2b;
            load_y4(2, v0b, v1b, v2b);
            uint32_t w8 = 0, w9 = 0, wA = 0, wB = 0;
            if (do_l3) {
                load_ga4_raw(sbase + 8L * 64, w8, w9, wA, wB);
            } else {
                if constexpr (ZIP_GA) w8 = GA[sbase + 8L * 64];
                else w8 = ah::pack2_raw<H>(G[sbase + 8L * 64], A[sbase + 8L * 64]);
            }
            if (do_l2) {
                float tv = 0.f, rv = 0.f, ayn = 0.f, ays = 0.f;
                {
                    float4 v0, v1, v2;
                    load_y4(1, v0, v1, v2);
                    uint32_t w4, w5, w6, w7;
                    load_ga4_raw(sbase + 4L * 64, w4, w5, w6, w7);
                    accum_raw(4, gan[4], w4, v0.x, v1.x, v2.x, xn, xbn, tv, rv, ayn, ays);
                    accum_raw(5, gan[5], w5, v0.y, v1.y, v2.y, xn, xbn, tv, rv, ayn, ays);
                    accum_raw(6, gan[6], w6, v0.z, v1.z, v2.z, xn, xbn, tv, rv, ayn, ays);
                    accum_raw(7, gan[7], w7, v0.w, v1.w, v2.w, xn, xbn, tv, rv, ayn, ays);
                }
                accum_raw(8, gan[8], w8, v0b.x, v1b.x, v2b.x, xn, xbn, tv, rv, ayn, ays);
                finish_l(2, 1.f, tv, rv, ayn, ays);
            }
            if (do_l3) {
                float tv = 0.f, rv = 0.f, ayn = 0.f, ays = 0.f;
                accum_raw(9, gan[9], w9, v0b.y, v1b.y, v2b.y, -xn, -xbn, tv, rv, ayn, ays);
                accum_raw(10, gan[10], wA, v0b.z, v1b.z, v2b.z, -xn, -xbn, tv, rv, ayn, ays);
                accum_raw(11, gan[11], wB, v0b.w, v1b.w, v2b.w, -xn, -xbn, tv, rv, ayn, ays);
                {
                    float4 v0, v1, v2;
                    load_y4(3, v0, v1, v2);
                    uint32_t wC, wD, wE, wF;
                    load_ga4_raw(sbase + 12L * 64, wC, wD, wE, wF);
                    accum_raw(12, gan[12], wC, v0.x, v1.x, v2.x, -xn, -xbn, tv, rv, ayn, ays);
                    accum_raw(13, gan[13], wD, v0.y, v1.y, v2.y, -xn, -xbn, tv, rv, ayn, ays);
                    accum_raw(14, gan[14], wE, v0.z, v1.z, v2.z, -xn, -xbn, tv, rv, ayn, ays);
                    accum_raw(15, gan[15], wF, v0.w, v1.w, v2.w, -xn, -xbn, tv, rv, ayn, ays);
                }
                finish_l(3, -1.f, tv, rv, ayn, ays);
            }
        }
        }
        if constexpr (T0S_DIR) {
            if (sp_s < S) s_t0s[grp][sp_s][c] += xd_s;
        } else {
            atomicAdd(Xd + s * 64 + c, xd_s);
        }
    }
    if (prev_kidx >= 0) flush_gc(prev_kidx);
    if constexpr (T0S_DIR) {
        if (sp_n < S) s_t0s[grp][sp_n][c] += xd_n;
        for (int sp = 0; sp < S; ++sp) {
            const float v = s_t0s[grp][sp][c];
            if (v != 0.f) atomicAdd(T0s + sp * 64 + c, v);
        }
    } else {
        atomicAdd(Xd + (long)n * 64 + c, xd_n);
    }
}

// ============================================================================ 2f. k_slot_xd_dir (Wave 23+24: sopt=11..15!)
// Directed-edge slot_xd over unified (n, kidx)-sorted emeta_dir:
//   - Eliminates 100% of neighbor G[s]/A[s]/GA[s] loads (-164 MB L2/layer!).
//   - Unified emeta_dir (no rptr0 split!) halves kidx transitions -> 2x fewer s_ck reloads & gR_k atomics!
//   - T0S_DIR=true (Layer 0, S <= 4): X0 from 4 regs (0 X0 loads!) & Xd0 into 4 regs t0s0..3 (5.4M Xd0 atomics -> 0!).
template <typename H, bool XB, bool EMETA, int Y3_MODE = 0, bool SHELL = false, int ES = 2, bool T0S_DIR = false>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
__global__ void __launch_bounds__(ES == 2 ? 128 : 64, ES == 2 ? 6 : 12) k_slot_xd_dir(
#else
__global__ void __launch_bounds__(ES == 2 ? 128 : 64, ES == 2 ? 5 : 10) k_slot_xd_dir(
#endif
    const int* __restrict__ rptr, const int* __restrict__ snd, const int* __restrict__ ps,
    const int* __restrict__ sp_idx, const float* __restrict__ sp_u,
    const H* __restrict__ G, const H* __restrict__ A, const float* __restrict__ X, const float* __restrict__ Xb,
    const float* __restrict__ YwP, const float* __restrict__ GyP, const float* __restrict__ YP,
    const float* __restrict__ C_kh, float* __restrict__ gR_k, float* __restrict__ Xd,
    float sp_invh, int N, const int4* __restrict__ emeta, const int* __restrict__ norder = nullptr,
    const float* __restrict__ Y3P = nullptr, int kcut2 = 999999, int kcut3 = 999999,
    const int* __restrict__ spc = nullptr, const float* __restrict__ T0 = nullptr,
    float* __restrict__ T0s = nullptr, int S = 0) {
    const int nn = blockIdx.x;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    const int c = threadIdx.x & 63;
    const int grp = (ES > 1) ? (threadIdx.x >> 6) : 0;

    const int e0_n = rptr[n];
    const int e1_n = rptr[n + 1];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = EMETA ? (ES > 1 ? e0_n + (cnt_n * grp) / ES : e0_n) : (e0_n + grp);
    const int e_end = EMETA ? (ES > 1 ? e0_n + (cnt_n * (grp + 1)) / ES : e1_n) : e1_n;
    const int e_step = EMETA ? 1 : ES;
    if (e_beg >= e_end) return;

    __shared__ float s_gc[ES][16][64];  // 4 KB (ES=1) or 8 KB (ES=2)
    __shared__ uint2 s_ck[ES][4][64];   // 2 KB (ES=1) or 4 KB (ES=2)

    uint32_t gan[16];
    const long nbase = (long)n * 16 * 64 + c;
#pragma unroll
    for (int m = 0; m < 16; ++m)
        gan[m] = ah::pack2_raw<H>(G[nbase + (long)m * 64], A[nbase + (long)m * 64]);
    float t0r0 = 0.f, t0r1 = 0.f, t0r2 = 0.f, t0r3 = 0.f;
    float t0s0 = 0.f, t0s1 = 0.f, t0s2 = 0.f, t0s3 = 0.f;
    if constexpr (T0S_DIR) {
        t0r0 = T0[0 * 64 + c];
        if (S > 1) t0r1 = T0[1 * 64 + c];
        if (S > 2) t0r2 = T0[2 * 64 + c];
        if (S > 3) t0r3 = T0[3 * 64 + c];
    }
    const uint2* Ck2 = reinterpret_cast<const uint2*>(C_kh);

    int prev_kidx = -1;
    auto flush_gc = [&](int pk) {
#pragma unroll 1
        for (int l = 0; l < 4; ++l) {
            if constexpr (SHELL) {
                if ((l == 2 && pk >= kcut2) || (l == 3 && pk >= kcut3)) continue;
            }
            const long base = ((long)pk * 4 + l) * 4 * 64 + c;
#pragma unroll
            for (int q = 0; q < 4; ++q) atomicAdd(gR_k + base + q * 64, s_gc[grp][l * 4 + q][c]);
        }
    };

    for (int e = e_beg; e < e_end; e += e_step) {
        int ps_e, kidx;
        long p, s;
        float u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            kidx = em.z;
            if (kidx < 0) continue;
            p = ps_e >> 1;
            s = em.y;
            u = __int_as_float(em.w);
        } else {
            ps_e = ps[e];
            p = ps_e >> 1;
            kidx = sp_idx[p];
            if (kidx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        const bool do_l2 = !SHELL || (kidx < kcut2);
        const bool do_l3 = !SHELL || (kidx < kcut3);
        const bool same_k = (kidx == prev_kidx);
        if (!same_k) {
            if (prev_kidx >= 0) flush_gc(prev_kidx);
            const long cbase = (long)kidx * 4 * 64 + c;
            s_ck[grp][0][c] = Ck2[cbase + 0 * 64];
            s_ck[grp][1][c] = Ck2[cbase + 1 * 64];
            if (do_l2) s_ck[grp][2][c] = Ck2[cbase + 2 * 64];
            if (do_l3) s_ck[grp][3][c] = Ck2[cbase + 3 * 64];
            prev_kidx = kidx;
        }
        const float sgn = (ps_e & 1) ? -1.f : 1.f;
        int sp_s = 0;
        float xs;
        if constexpr (T0S_DIR) {
            sp_s = spc[s];
            xs = (sp_s == 0) ? t0r0 : ((sp_s == 1) ? t0r1 : ((sp_s == 2) ? t0r2 : t0r3));
        } else {
            xs = X[s * 64 + c];
        }
        const float xbs = XB ? Xb[s * 64 + c] : 0.f;
        const float u2 = u * u, u3 = u2 * u;
        const float ui = u * sp_invh, u2i = u2 * sp_invh;
        float xd_s = 0.f;

        auto finish_l = [&](int l, float sl, float tv, float rv, float rb, float ay) {
            const float rval = eval_c4_s(s_ck[grp][l][c], u);
            const float ays = sl * ay;
            xd_s += rval * ays;
            const float tv_l = sl * xs * tv;
            const float rv_l = sl * (xs * rv + (XB ? xbs * rb : 0.f)) + ays * xs;
            const float g0 = rv_l;
            const float g1 = u * rv_l + sp_invh * tv_l;
            const float g2 = u2 * rv_l + 2.f * ui * tv_l;
            const float g3 = u3 * rv_l + 3.f * u2i * tv_l;
            if (same_k) {
                s_gc[grp][l * 4 + 0][c] += g0;
                s_gc[grp][l * 4 + 1][c] += g1;
                s_gc[grp][l * 4 + 2][c] += g2;
                s_gc[grp][l * 4 + 3][c] += g3;
            } else {
                s_gc[grp][l * 4 + 0][c] = g0;
                s_gc[grp][l * 4 + 1][c] = g1;
                s_gc[grp][l * 4 + 2][c] = g2;
                s_gc[grp][l * 4 + 3][c] = g3;
            }
        };
        auto accum_m = [&](int m, float ywv, float gyv, float yv, float& tv, float& rv, float& rb, float& ay) {
            const float2 ga = ah::unpack2<H>(gan[m]);
            tv += ga.x * ywv;
            rv += ga.x * gyv;
            if constexpr (XB) rb += ga.x * yv;
            ay += ga.y * yv;
        };
        auto load_y4 = [&](int q, float4& v0, float4& v1, float4& v2) {
            if constexpr (Y3_MODE == 2) {
                const uint4* yhp = reinterpret_cast<const uint4*>(Y3P) + 6L * p;
                const uint4 qh = yhp[q];
                const float2 w01 = __half22float2(*reinterpret_cast<const __half2*>(&qh.x));
                const float2 w23 = __half22float2(*reinterpret_cast<const __half2*>(&qh.y));
                const float2 g01 = __half22float2(*reinterpret_cast<const __half2*>(&qh.z));
                const float2 g23 = __half22float2(*reinterpret_cast<const __half2*>(&qh.w));
                v0 = make_float4(w01.x, w01.y, w23.x, w23.y);
                v1 = make_float4(g01.x, g01.y, g23.x, g23.y);
                const uint2 yy = *(reinterpret_cast<const uint2*>(yhp + 4) + q);
                const float2 y01 = __half22float2(*reinterpret_cast<const __half2*>(&yy.x));
                const float2 y23 = __half22float2(*reinterpret_cast<const __half2*>(&yy.y));
                v2 = make_float4(y01.x, y01.y, y23.x, y23.y);
            } else if constexpr (Y3_MODE == 1) {
                const float* yp = Y3P + 48L * p + 12 * q;
                v0 = *reinterpret_cast<const float4*>(yp + 0);
                v1 = *reinterpret_cast<const float4*>(yp + 4);
                v2 = *reinterpret_cast<const float4*>(yp + 8);
            } else {
                v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 4 * q);
                v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 4 * q);
                v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 4 * q);
            }
        };

        {
            float4 v0, v1, v2;
            load_y4(0, v0, v1, v2);
            float tv = 0.f, rv = 0.f, rb = 0.f, ay = 0.f;
            accum_m(0, v0.x, v1.x, v2.x, tv, rv, rb, ay);
            finish_l(0, 1.f, tv, rv, rb, ay);
            tv = 0.f; rv = 0.f; rb = 0.f; ay = 0.f;
            accum_m(1, v0.y, v1.y, v2.y, tv, rv, rb, ay);
            accum_m(2, v0.z, v1.z, v2.z, tv, rv, rb, ay);
            accum_m(3, v0.w, v1.w, v2.w, tv, rv, rb, ay);
            finish_l(1, sgn, tv, rv, rb, ay);
        }
        if (do_l2 || do_l3) {
            float4 v0b, v1b, v2b;
            load_y4(2, v0b, v1b, v2b);
            if (do_l2) {
                float tv = 0.f, rv = 0.f, rb = 0.f, ay = 0.f;
                {
                    float4 v0, v1, v2;
                    load_y4(1, v0, v1, v2);
                    accum_m(4, v0.x, v1.x, v2.x, tv, rv, rb, ay);
                    accum_m(5, v0.y, v1.y, v2.y, tv, rv, rb, ay);
                    accum_m(6, v0.z, v1.z, v2.z, tv, rv, rb, ay);
                    accum_m(7, v0.w, v1.w, v2.w, tv, rv, rb, ay);
                }
                accum_m(8, v0b.x, v1b.x, v2b.x, tv, rv, rb, ay);
                finish_l(2, 1.f, tv, rv, rb, ay);
            }
            if (do_l3) {
                float tv = 0.f, rv = 0.f, rb = 0.f, ay = 0.f;
                accum_m(9, v0b.y, v1b.y, v2b.y, tv, rv, rb, ay);
                accum_m(10, v0b.z, v1b.z, v2b.z, tv, rv, rb, ay);
                accum_m(11, v0b.w, v1b.w, v2b.w, tv, rv, rb, ay);
                {
                    float4 v0, v1, v2;
                    load_y4(3, v0, v1, v2);
                    accum_m(12, v0.x, v1.x, v2.x, tv, rv, rb, ay);
                    accum_m(13, v0.y, v1.y, v2.y, tv, rv, rb, ay);
                    accum_m(14, v0.z, v1.z, v2.z, tv, rv, rb, ay);
                    accum_m(15, v0.w, v1.w, v2.w, tv, rv, rb, ay);
                }
                finish_l(3, sgn, tv, rv, rb, ay);
            }
        }
        if constexpr (T0S_DIR) {
            if (sp_s == 0) t0s0 += xd_s;
            else if (sp_s == 1) t0s1 += xd_s;
            else if (sp_s == 2) t0s2 += xd_s;
            else t0s3 += xd_s;
        } else {
            atomicAdd(Xd + s * 64 + c, xd_s);
        }
    }
    if (prev_kidx >= 0) flush_gc(prev_kidx);
    if constexpr (T0S_DIR) {
        if (t0s0 != 0.f) atomicAdd(T0s + 0 * 64 + c, t0s0);
        if (S > 1 && t0s1 != 0.f) atomicAdd(T0s + 1 * 64 + c, t0s1);
        if (S > 2 && t0s2 != 0.f) atomicAdd(T0s + 2 * 64 + c, t0s2);
        if (S > 3 && t0s3 != 0.f) atomicAdd(T0s + 3 * 64 + c, t0s3);
    }
}

// ============================================================================ 2e2. k_slot_xd_q4 (sopt=9/10, Wave 22!)
// 1-warp 64ch (2 ch/lane) + GA_Q4 [N, 8, 32] uint4 -> ONLY 8x LDG.128 per edge (4x fewer LDGs vs v3's 32x LDG.32!),
// 2x fewer Y3P loads, shared 4 KB s_gn[8][32] across both warps of node n (0 regs for gn/an!), 8 KB warp-private s_gc,
// prev_ck[4] cached in regs (0 extra smem -> 12 KB total, 8 blks/SM, <=128 regs, 0 spills on sm_120 & sm_89!).
// USE_HFMA2=false (sopt=9): exact FP32 m-loop; USE_HFMA2=true (sopt=10): native __half2 HFMA2 m-loop (0 unpacks!).
template <typename H, bool XB, bool EMETA, bool USE_HFMA2 = false, int Y3_MODE = 0>
__global__ void __launch_bounds__(64, 8) k_slot_xd_q4(
    const int* __restrict__ rptr, const int* __restrict__ rptr0, const int* __restrict__ snd,
    const int* __restrict__ ps, const int* __restrict__ sp_idx, const float* __restrict__ sp_u,
    const float* __restrict__ X, const float* __restrict__ Xb,
    const float* __restrict__ YwP, const float* __restrict__ GyP, const float* __restrict__ YP,
    const float* __restrict__ C_kh, float* __restrict__ gR_k, float* __restrict__ Xd,
    float sp_invh, int N, const int4* __restrict__ emeta, const int* __restrict__ norder,
    const uint32_t* __restrict__ GA, const float* __restrict__ Y3P = nullptr) {
    const int nn = blockIdx.x;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    const int lane = threadIdx.x & 31;
    const int grp = threadIdx.x >> 5;  // 0 or 1 (ES=2 warps/node, 64 threads/block)

    __shared__ __align__(16) uint4 s_gn[8][32];        // 4 KB shared by both warps of node n
    __shared__ __align__(8) float2 s_gc[2][16][32];    // 8 KB warp-private knot-run accumulator

    const uint4* GA4 = reinterpret_cast<const uint4*>(GA);
    const long nbase4 = (long)n * 8 * 32 + lane;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int km = grp * 4 + k;
        s_gn[km][lane] = GA4[nbase4 + (long)km * 32];
    }
    __syncthreads();

    const int e0_n = rptr[n];
    const int e1_n = EMETA ? rptr0[n] : rptr[n + 1];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = EMETA ? (e0_n + (cnt_n * grp) / 2) : (e0_n + grp);
    const int e_end = EMETA ? (e0_n + (cnt_n * (grp + 1)) / 2) : e1_n;
    const int e_step = EMETA ? 1 : 2;
    if (e_beg >= e_end) return;

    const float2* X2 = reinterpret_cast<const float2*>(X);
    const float2* Xb2 = reinterpret_cast<const float2*>(Xb);
    const uint4* Ck4 = reinterpret_cast<const uint4*>(C_kh);
    const float2 xn = X2[(long)n * 32 + lane];
    const float2 xbn = XB ? Xb2[(long)n * 32 + lane] : make_float2(0.f, 0.f);
    float2 xd_n = make_float2(0.f, 0.f);

    uint4 prev_ck[4];
    int prev_kidx = -1;
    auto flush_gc = [&](int pk) {
#pragma unroll 1
        for (int l = 0; l < 4; ++l) {
            const long base = ((long)pk * 4 + l) * 4 * 64 + 2 * lane;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float2 v = s_gc[grp][l * 4 + q][lane];
                atomicAdd(gR_k + base + q * 64, v.x);
                atomicAdd(gR_k + base + q * 64 + 1, v.y);
            }
        }
    };

    for (int e = e_beg; e < e_end; e += e_step) {
        int ps_e, kidx;
        long p, s;
        float u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            if (ps_e & 1) continue;
            kidx = em.z;
            if (kidx < 0) continue;
            p = ps_e >> 1;
            s = em.y;
            u = __int_as_float(em.w);
        } else {
            ps_e = ps[e];
            if (ps_e & 1) continue;
            p = ps_e >> 1;
            kidx = sp_idx[p];
            if (kidx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        const bool same_k = (kidx == prev_kidx);
        if (!same_k) {
            if (prev_kidx >= 0) flush_gc(prev_kidx);
            const long cbase = (long)kidx * 4 * 32 + lane;
#pragma unroll
            for (int l = 0; l < 4; ++l) prev_ck[l] = Ck4[cbase + l * 32];
            prev_kidx = kidx;
        }
        const float2 xs = X2[s * 32 + lane];
        const float2 xbs = XB ? Xb2[s * 32 + lane] : make_float2(0.f, 0.f);
        const long sbase4 = s * 8 * 32 + lane;
        const float u2 = u * u, u3 = u2 * u;
        const float ui = u * sp_invh, u2i = u2 * sp_invh;
        float2 xd_s = make_float2(0.f, 0.f);

        auto finish_l = [&](int l, float sgn2, float tvx, float tvy, float rvx, float rvy,
                            float aynx, float ayny, float aysx, float aysy) {
            float rx, ry;
            eval_c4_pair(prev_ck[l], u, rx, ry);
            xd_s.x += rx * aynx; xd_s.y += ry * ayny;
            xd_n.x += sgn2 * rx * aysx; xd_n.y += sgn2 * ry * aysy;
            rvx += aynx * xs.x + sgn2 * aysx * xn.x;
            rvy += ayny * xs.y + sgn2 * aysy * xn.y;
            const float2 g0 = make_float2(rvx, rvy);
            const float2 g1 = make_float2(u * rvx + sp_invh * tvx, u * rvy + sp_invh * tvy);
            const float2 g2 = make_float2(u2 * rvx + 2.f * ui * tvx, u2 * rvy + 2.f * ui * tvy);
            const float2 g3 = make_float2(u3 * rvx + 3.f * u2i * tvx, u3 * rvy + 3.f * u2i * tvy);
            if (same_k) {
                float2 p0 = s_gc[grp][l * 4 + 0][lane]; p0.x += g0.x; p0.y += g0.y; s_gc[grp][l * 4 + 0][lane] = p0;
                float2 p1 = s_gc[grp][l * 4 + 1][lane]; p1.x += g1.x; p1.y += g1.y; s_gc[grp][l * 4 + 1][lane] = p1;
                float2 p2 = s_gc[grp][l * 4 + 2][lane]; p2.x += g2.x; p2.y += g2.y; s_gc[grp][l * 4 + 2][lane] = p2;
                float2 p3 = s_gc[grp][l * 4 + 3][lane]; p3.x += g3.x; p3.y += g3.y; s_gc[grp][l * 4 + 3][lane] = p3;
            } else {
                s_gc[grp][l * 4 + 0][lane] = g0;
                s_gc[grp][l * 4 + 1][lane] = g1;
                s_gc[grp][l * 4 + 2][lane] = g2;
                s_gc[grp][l * 4 + 3][lane] = g3;
            }
        };

        if constexpr (USE_HFMA2 && std::is_same<H, __half>::value) {
            const __half2 xs_h = __float22half2_rn(xs);
            const __half2 xn_p = __float22half2_rn(xn), xn_m = __hneg2(xn_p);
            const __half2 xbs_h = XB ? __float22half2_rn(xbs) : __float2half2_rn(0.f);
            const __half2 xbn_p = XB ? __float22half2_rn(xbn) : __float2half2_rn(0.f);
            const __half2 xbn_m = XB ? __hneg2(xbn_p) : __float2half2_rn(0.f);
            const __half2 z_h = __float2half2_rn(0.f);

            auto load_y4_h = [&](int q, __half2& w01, __half2& w23, __half2& g01, __half2& g23, __half2& y01, __half2& y23) {
                if constexpr (Y3_MODE == 2) {
                    const uint4* yhp = reinterpret_cast<const uint4*>(Y3P) + 6L * p;
                    const uint4 qh = yhp[q];
                    w01 = *reinterpret_cast<const __half2*>(&qh.x);
                    w23 = *reinterpret_cast<const __half2*>(&qh.y);
                    g01 = *reinterpret_cast<const __half2*>(&qh.z);
                    g23 = *reinterpret_cast<const __half2*>(&qh.w);
                    const uint2 yy = *(reinterpret_cast<const uint2*>(yhp + 4) + q);
                    y01 = *reinterpret_cast<const __half2*>(&yy.x);
                    y23 = *reinterpret_cast<const __half2*>(&yy.y);
                } else {
                    float4 v0, v1, v2;
                    if constexpr (Y3_MODE == 1) {
                        const float* yp = Y3P + 48L * p + 12 * q;
                        v0 = *reinterpret_cast<const float4*>(yp + 0);
                        v1 = *reinterpret_cast<const float4*>(yp + 4);
                        v2 = *reinterpret_cast<const float4*>(yp + 8);
                    } else {
                        v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 4 * q);
                        v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 4 * q);
                        v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 4 * q);
                    }
                    w01 = __floats2half2_rn(v0.x, v0.y); w23 = __floats2half2_rn(v0.z, v0.w);
                    g01 = __floats2half2_rn(v1.x, v1.y); g23 = __floats2half2_rn(v1.z, v1.w);
                    y01 = __floats2half2_rn(v2.x, v2.y); y23 = __floats2half2_rn(v2.z, v2.w);
                }
            };
            auto accum_h2 = [&](uint32_t gnw, uint32_t anw, uint32_t gsw, uint32_t asw,
                                __half2 yw_h, __half2 gy_h, __half2 y_h, __half2 xns_h, __half2 xbns_h,
                                __half2& tv, __half2& rv, __half2& ayn, __half2& ays) {
                const __half2 gn_h = *reinterpret_cast<const __half2*>(&gnw);
                const __half2 an_h = *reinterpret_cast<const __half2*>(&anw);
                const __half2 gs_h = *reinterpret_cast<const __half2*>(&gsw);
                const __half2 as_h = *reinterpret_cast<const __half2*>(&asw);
                const __half2 gc = __hfma2(gn_h, xs_h, __hmul2(gs_h, xns_h));
                tv = __hfma2(gc, yw_h, tv);
                rv = __hfma2(gc, gy_h, rv);
                if constexpr (XB) rv = __hfma2(__hfma2(gn_h, xbs_h, __hmul2(gs_h, xbns_h)), y_h, rv);
                ayn = __hfma2(an_h, y_h, ayn);
                ays = __hfma2(as_h, y_h, ays);
            };
            auto fin_h2 = [&](int l, float sgn2, __half2 tv, __half2 rv, __half2 ayn, __half2 ays) {
                const float2 tf = __half22float2(tv), rf = __half22float2(rv);
                const float2 anf = __half22float2(ayn), asf = __half22float2(ays);
                finish_l(l, sgn2, tf.x, tf.y, rf.x, rf.y, anf.x, anf.y, asf.x, asf.y);
            };
            {
                __half2 w01, w23, g01, g23, y01, y23;
                load_y4_h(0, w01, w23, g01, g23, y01, y23);
                const uint4 gsa = GA4[sbase4 + 0 * 32], gsb = GA4[sbase4 + 1 * 32];
                const uint4 gna = s_gn[0][lane], gnb = s_gn[1][lane];
                __half2 tv = z_h, rv = z_h, ayn = z_h, ays = z_h;
                accum_h2(gna.x, gna.y, gsa.x, gsa.y, __low2half2(w01), __low2half2(g01), __low2half2(y01), xn_p, xbn_p, tv, rv, ayn, ays);
                fin_h2(0, 1.f, tv, rv, ayn, ays);
                tv = rv = ayn = ays = z_h;
                accum_h2(gna.z, gna.w, gsa.z, gsa.w, __high2half2(w01), __high2half2(g01), __high2half2(y01), xn_m, xbn_m, tv, rv, ayn, ays);
                accum_h2(gnb.x, gnb.y, gsb.x, gsb.y, __low2half2(w23), __low2half2(g23), __low2half2(y23), xn_m, xbn_m, tv, rv, ayn, ays);
                accum_h2(gnb.z, gnb.w, gsb.z, gsb.w, __high2half2(w23), __high2half2(g23), __high2half2(y23), xn_m, xbn_m, tv, rv, ayn, ays);
                fin_h2(1, -1.f, tv, rv, ayn, ays);
            }
            {
                __half2 tv = z_h, rv = z_h, ayn = z_h, ays = z_h;
                {
                    __half2 w01, w23, g01, g23, y01, y23;
                    load_y4_h(1, w01, w23, g01, g23, y01, y23);
                    const uint4 gsa = GA4[sbase4 + 2 * 32], gsb = GA4[sbase4 + 3 * 32];
                    const uint4 gna = s_gn[2][lane], gnb = s_gn[3][lane];
                    accum_h2(gna.x, gna.y, gsa.x, gsa.y, __low2half2(w01), __low2half2(g01), __low2half2(y01), xn_p, xbn_p, tv, rv, ayn, ays);
                    accum_h2(gna.z, gna.w, gsa.z, gsa.w, __high2half2(w01), __high2half2(g01), __high2half2(y01), xn_p, xbn_p, tv, rv, ayn, ays);
                    accum_h2(gnb.x, gnb.y, gsb.x, gsb.y, __low2half2(w23), __low2half2(g23), __low2half2(y23), xn_p, xbn_p, tv, rv, ayn, ays);
                    accum_h2(gnb.z, gnb.w, gsb.z, gsb.w, __high2half2(w23), __high2half2(g23), __high2half2(y23), xn_p, xbn_p, tv, rv, ayn, ays);
                }
                {
                    __half2 w01, w23, g01, g23, y01, y23;
                    load_y4_h(2, w01, w23, g01, g23, y01, y23);
                    const uint4 gsa = GA4[sbase4 + 4 * 32], gsb = GA4[sbase4 + 5 * 32];
                    const uint4 gna = s_gn[4][lane], gnb = s_gn[5][lane];
                    accum_h2(gna.x, gna.y, gsa.x, gsa.y, __low2half2(w01), __low2half2(g01), __low2half2(y01), xn_p, xbn_p, tv, rv, ayn, ays);
                    fin_h2(2, 1.f, tv, rv, ayn, ays);
                    tv = rv = ayn = ays = z_h;
                    accum_h2(gna.z, gna.w, gsa.z, gsa.w, __high2half2(w01), __high2half2(g01), __high2half2(y01), xn_m, xbn_m, tv, rv, ayn, ays);
                    accum_h2(gnb.x, gnb.y, gsb.x, gsb.y, __low2half2(w23), __low2half2(g23), __low2half2(y23), xn_m, xbn_m, tv, rv, ayn, ays);
                    accum_h2(gnb.z, gnb.w, gsb.z, gsb.w, __high2half2(w23), __high2half2(g23), __high2half2(y23), xn_m, xbn_m, tv, rv, ayn, ays);
                }
                {
                    __half2 w01, w23, g01, g23, y01, y23;
                    load_y4_h(3, w01, w23, g01, g23, y01, y23);
                    const uint4 gsa = GA4[sbase4 + 6 * 32], gsb = GA4[sbase4 + 7 * 32];
                    const uint4 gna = s_gn[6][lane], gnb = s_gn[7][lane];
                    accum_h2(gna.x, gna.y, gsa.x, gsa.y, __low2half2(w01), __low2half2(g01), __low2half2(y01), xn_m, xbn_m, tv, rv, ayn, ays);
                    accum_h2(gna.z, gna.w, gsa.z, gsa.w, __high2half2(w01), __high2half2(g01), __high2half2(y01), xn_m, xbn_m, tv, rv, ayn, ays);
                    accum_h2(gnb.x, gnb.y, gsb.x, gsb.y, __low2half2(w23), __low2half2(g23), __low2half2(y23), xn_m, xbn_m, tv, rv, ayn, ays);
                    accum_h2(gnb.z, gnb.w, gsb.z, gsb.w, __high2half2(w23), __high2half2(g23), __high2half2(y23), xn_m, xbn_m, tv, rv, ayn, ays);
                }
                fin_h2(3, -1.f, tv, rv, ayn, ays);
            }
        } else {
            auto load_y4 = [&](int q, float4& v0, float4& v1, float4& v2) {
                if constexpr (Y3_MODE == 2) {
                    const uint4* yhp = reinterpret_cast<const uint4*>(Y3P) + 6L * p;
                    const uint4 qh = yhp[q];
                    const float2 w01 = __half22float2(*reinterpret_cast<const __half2*>(&qh.x));
                    const float2 w23 = __half22float2(*reinterpret_cast<const __half2*>(&qh.y));
                    const float2 g01 = __half22float2(*reinterpret_cast<const __half2*>(&qh.z));
                    const float2 g23 = __half22float2(*reinterpret_cast<const __half2*>(&qh.w));
                    v0 = make_float4(w01.x, w01.y, w23.x, w23.y);
                    v1 = make_float4(g01.x, g01.y, g23.x, g23.y);
                    const uint2 yy = *(reinterpret_cast<const uint2*>(yhp + 4) + q);
                    const float2 y01 = __half22float2(*reinterpret_cast<const __half2*>(&yy.x));
                    const float2 y23 = __half22float2(*reinterpret_cast<const __half2*>(&yy.y));
                    v2 = make_float4(y01.x, y01.y, y23.x, y23.y);
                } else if constexpr (Y3_MODE == 1) {
                    const float* yp = Y3P + 48L * p + 12 * q;
                    v0 = *reinterpret_cast<const float4*>(yp + 0);
                    v1 = *reinterpret_cast<const float4*>(yp + 4);
                    v2 = *reinterpret_cast<const float4*>(yp + 8);
                } else {
                    v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 4 * q);
                    v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 4 * q);
                    v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 4 * q);
                }
            };
            auto accum_f2 = [&](uint32_t gnw, uint32_t anw, uint32_t gsw, uint32_t asw,
                                float ywv, float gyv, float yv, float sgn2,
                                float& tvx, float& tvy, float& rvx, float& rvy,
                                float& aynx, float& ayny, float& aysx, float& aysy) {
                const float2 gnf = ah::unpack2<H>(gnw), anf = ah::unpack2<H>(anw);
                const float2 gsf = ah::unpack2<H>(gsw), asf = ah::unpack2<H>(asw);
                const float gx = gnf.x * xs.x + sgn2 * gsf.x * xn.x;
                const float gy = gnf.y * xs.y + sgn2 * gsf.y * xn.y;
                tvx += gx * ywv; tvy += gy * ywv;
                rvx += gx * gyv; rvy += gy * gyv;
                if constexpr (XB) {
                    rvx += (gnf.x * xbs.x + sgn2 * gsf.x * xbn.x) * yv;
                    rvy += (gnf.y * xbs.y + sgn2 * gsf.y * xbn.y) * yv;
                }
                aynx += anf.x * yv; ayny += anf.y * yv;
                aysx += asf.x * yv; aysy += asf.y * yv;
            };
            {
                float4 v0, v1, v2;
                load_y4(0, v0, v1, v2);
                const uint4 gsa = GA4[sbase4 + 0 * 32], gsb = GA4[sbase4 + 1 * 32];
                const uint4 gna = s_gn[0][lane], gnb = s_gn[1][lane];
                float tvx = 0.f, tvy = 0.f, rvx = 0.f, rvy = 0.f, aynx = 0.f, ayny = 0.f, aysx = 0.f, aysy = 0.f;
                accum_f2(gna.x, gna.y, gsa.x, gsa.y, v0.x, v1.x, v2.x, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                finish_l(0, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                tvx = tvy = rvx = rvy = aynx = ayny = aysx = aysy = 0.f;
                accum_f2(gna.z, gna.w, gsa.z, gsa.w, v0.y, v1.y, v2.y, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_f2(gnb.x, gnb.y, gsb.x, gsb.y, v0.z, v1.z, v2.z, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_f2(gnb.z, gnb.w, gsb.z, gsb.w, v0.w, v1.w, v2.w, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                finish_l(1, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            }
            {
                float tvx = 0.f, tvy = 0.f, rvx = 0.f, rvy = 0.f, aynx = 0.f, ayny = 0.f, aysx = 0.f, aysy = 0.f;
                {
                    float4 v0, v1, v2;
                    load_y4(1, v0, v1, v2);
                    const uint4 gsa = GA4[sbase4 + 2 * 32], gsb = GA4[sbase4 + 3 * 32];
                    const uint4 gna = s_gn[2][lane], gnb = s_gn[3][lane];
                    accum_f2(gna.x, gna.y, gsa.x, gsa.y, v0.x, v1.x, v2.x, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                    accum_f2(gna.z, gna.w, gsa.z, gsa.w, v0.y, v1.y, v2.y, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                    accum_f2(gnb.x, gnb.y, gsb.x, gsb.y, v0.z, v1.z, v2.z, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                    accum_f2(gnb.z, gnb.w, gsb.z, gsb.w, v0.w, v1.w, v2.w, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                }
                {
                    float4 v0, v1, v2;
                    load_y4(2, v0, v1, v2);
                    const uint4 gsa = GA4[sbase4 + 4 * 32], gsb = GA4[sbase4 + 5 * 32];
                    const uint4 gna = s_gn[4][lane], gnb = s_gn[5][lane];
                    accum_f2(gna.x, gna.y, gsa.x, gsa.y, v0.x, v1.x, v2.x, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                    finish_l(2, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                    tvx = tvy = rvx = rvy = aynx = ayny = aysx = aysy = 0.f;
                    accum_f2(gna.z, gna.w, gsa.z, gsa.w, v0.y, v1.y, v2.y, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                    accum_f2(gnb.x, gnb.y, gsb.x, gsb.y, v0.z, v1.z, v2.z, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                    accum_f2(gnb.z, gnb.w, gsb.z, gsb.w, v0.w, v1.w, v2.w, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                }
                {
                    float4 v0, v1, v2;
                    load_y4(3, v0, v1, v2);
                    const uint4 gsa = GA4[sbase4 + 6 * 32], gsb = GA4[sbase4 + 7 * 32];
                    const uint4 gna = s_gn[6][lane], gnb = s_gn[7][lane];
                    accum_f2(gna.x, gna.y, gsa.x, gsa.y, v0.x, v1.x, v2.x, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                    accum_f2(gna.z, gna.w, gsa.z, gsa.w, v0.y, v1.y, v2.y, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                    accum_f2(gnb.x, gnb.y, gsb.x, gsb.y, v0.z, v1.z, v2.z, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                    accum_f2(gnb.z, gnb.w, gsb.z, gsb.w, v0.w, v1.w, v2.w, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                }
                finish_l(3, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            }
        }
        atomicAdd(Xd + s * 64 + 2 * lane, xd_s.x);
        atomicAdd(Xd + s * 64 + 2 * lane + 1, xd_s.y);
    }
    if (prev_kidx >= 0) flush_gc(prev_kidx);
    atomicAdd(Xd + (long)n * 64 + 2 * lane, xd_n.x);
    atomicAdd(Xd + (long)n * 64 + 2 * lane + 1, xd_n.y);
}

// ============================================================================ 2f. k_slot_xd_h2_fast (sopt=8, Wave 19!)
// 1-warp 64ch __half2 (LDG.32 = half LSU ops, 100% 128B full cache lines!) modeled on fast k_force_pair_h2:
//   - 128 threads/block (4 warps): ES=4 (1 node/blk at b1) or ES=2 (2 nodes/blk at b4) -> full SM saturation!
//   - gn[16]/an[16] in registers (ZERO smem reads in inner loop!), NO s_ck (full L1 cache!), #pragma unroll 1 flush_gc
//   - 4-at-a-time batched LDG.32 on G[s] & A[s] -> hides L2 latency, <=128 regs (0 spills, 16 warps/SM)!
template <typename H, bool XB, bool EMETA, int ES>
__global__ void __launch_bounds__(128, 4) k_slot_xd_h2_fast(
    const int* __restrict__ rptr, const int* __restrict__ rptr0, const int* __restrict__ snd,
    const int* __restrict__ ps, const int* __restrict__ sp_idx, const float* __restrict__ sp_u,
    const H* __restrict__ G, const H* __restrict__ A, const float* __restrict__ X, const float* __restrict__ Xb,
    const float* __restrict__ YwP, const float* __restrict__ GyP, const float* __restrict__ YP,
    const float* __restrict__ C_kh, float* __restrict__ gR_k, float* __restrict__ Xd,
    float sp_invh, int N, const int4* __restrict__ emeta, const int* __restrict__ norder = nullptr) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int grp = (ES == 4) ? warp : (warp & 1);
    const int nib = (ES == 4) ? 0 : (warp >> 1);
    const int nn = blockIdx.x * (4 / ES) + nib;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;

    const int e0_n = rptr[n];
    const int e1_n = EMETA ? rptr0[n] : rptr[n + 1];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = EMETA ? (e0_n + (cnt_n * grp) / ES) : (e0_n + grp);
    const int e_end = EMETA ? (e0_n + (cnt_n * (grp + 1)) / ES) : e1_n;
    const int e_step = EMETA ? 1 : ES;
    if (e_beg >= e_end) return;

    __shared__ __align__(8) float2 s_gc[4][16][32];  // 16 KB: 100% warp-private -> 0 barriers, 0 bank conflicts!

    uint32_t gn[16], an[16];
    const long nbase = (long)n * 16 * 32 + lane;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        gn[m] = ah::ld_h2<H>(G, nbase + (long)m * 32);
        an[m] = ah::ld_h2<H>(A, nbase + (long)m * 32);
    }
    const float2* X2 = reinterpret_cast<const float2*>(X);
    const float2* Xb2 = reinterpret_cast<const float2*>(Xb);
    const uint4* Ck4 = reinterpret_cast<const uint4*>(C_kh);
    const float2 xn = X2[(long)n * 32 + lane];
    const float2 xbn = XB ? Xb2[(long)n * 32 + lane] : make_float2(0.f, 0.f);
    float2 xd_n = make_float2(0.f, 0.f);

    int prev_kidx = -1;
    auto flush_gc = [&](int pk) {
#pragma unroll 1
        for (int l = 0; l < 4; ++l) {
            const long base = ((long)pk * 4 + l) * 4 * 64 + 2 * lane;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float2 v = s_gc[warp][l * 4 + q][lane];
                atomicAdd(gR_k + base + q * 64, v.x);
                atomicAdd(gR_k + base + q * 64 + 1, v.y);
            }
        }
    };

    for (int e = e_beg; e < e_end; e += e_step) {
        int ps_e, kidx;
        long p, s;
        float u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            if (ps_e & 1) continue;
            kidx = em.z;
            if (kidx < 0) continue;
            p = ps_e >> 1;
            s = em.y;
            u = __int_as_float(em.w);
        } else {
            ps_e = ps[e];
            if (ps_e & 1) continue;
            p = ps_e >> 1;
            kidx = sp_idx[p];
            if (kidx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        const bool same_k = (kidx == prev_kidx);
        if (!same_k) {
            if (prev_kidx >= 0) flush_gc(prev_kidx);
            prev_kidx = kidx;
        }
        const float2 xs = X2[s * 32 + lane];
        const float2 xbs = XB ? Xb2[s * 32 + lane] : make_float2(0.f, 0.f);
        const long sbase = s * 16 * 32 + lane;
        const long cbase = (long)kidx * 4 * 32 + lane;
        float2 xd_s = make_float2(0.f, 0.f);

        auto finish_l = [&](int l, float sgn2, float tvx, float tvy, float rvx, float rvy, float aynx, float ayny,
                            float aysx, float aysy) {
            float rx, ry;
            eval_c4_pair(Ck4[cbase + l * 32], u, rx, ry);
            xd_s.x += rx * aynx; xd_s.y += ry * ayny;
            xd_n.x += sgn2 * rx * aysx; xd_n.y += sgn2 * ry * aysy;
            rvx += aynx * xs.x + sgn2 * aysx * xn.x;
            rvy += ayny * xs.y + sgn2 * aysy * xn.y;
            const float u2 = u * u, u3 = u2 * u;
            const float ui = u * sp_invh, u2i = u2 * sp_invh;
            const float2 g0 = make_float2(rvx, rvy);
            const float2 g1 = make_float2(u * rvx + sp_invh * tvx, u * rvy + sp_invh * tvy);
            const float2 g2 = make_float2(u2 * rvx + 2.f * ui * tvx, u2 * rvy + 2.f * ui * tvy);
            const float2 g3 = make_float2(u3 * rvx + 3.f * u2i * tvx, u3 * rvy + 3.f * u2i * tvy);
            if (same_k) {
                float2 p0 = s_gc[warp][l * 4 + 0][lane]; p0.x += g0.x; p0.y += g0.y; s_gc[warp][l * 4 + 0][lane] = p0;
                float2 p1 = s_gc[warp][l * 4 + 1][lane]; p1.x += g1.x; p1.y += g1.y; s_gc[warp][l * 4 + 1][lane] = p1;
                float2 p2 = s_gc[warp][l * 4 + 2][lane]; p2.x += g2.x; p2.y += g2.y; s_gc[warp][l * 4 + 2][lane] = p2;
                float2 p3 = s_gc[warp][l * 4 + 3][lane]; p3.x += g3.x; p3.y += g3.y; s_gc[warp][l * 4 + 3][lane] = p3;
            } else {
                s_gc[warp][l * 4 + 0][lane] = g0;
                s_gc[warp][l * 4 + 1][lane] = g1;
                s_gc[warp][l * 4 + 2][lane] = g2;
                s_gc[warp][l * 4 + 3][lane] = g3;
            }
        };
        auto accum_v = [&](uint32_t gnm_u, uint32_t anm_u, uint32_t gs_u, uint32_t as_u, float ywv, float gyv, float yv,
                           float sgn2, float& tvx, float& tvy, float& rvx, float& rvy, float& aynx, float& ayny,
                           float& aysx, float& aysy) {
            const float2 gnm = ah::unpack2<H>(gnm_u), anm = ah::unpack2<H>(anm_u);
            const float2 gs = ah::unpack2<H>(gs_u), as = ah::unpack2<H>(as_u);
            const float gx = gnm.x * xs.x + sgn2 * gs.x * xn.x;
            const float gy = gnm.y * xs.y + sgn2 * gs.y * xn.y;
            tvx += gx * ywv; tvy += gy * ywv;
            rvx += gx * gyv; rvy += gy * gyv;
            if constexpr (XB) {
                rvx += (gnm.x * xbs.x + sgn2 * gs.x * xbn.x) * yv;
                rvy += (gnm.y * xbs.y + sgn2 * gs.y * xbn.y) * yv;
            }
            aynx += anm.x * yv; ayny += anm.y * yv;
            aysx += as.x * yv; aysy += as.y * yv;
        };

        // Chunk 0 (m=0..3): batch-issue 4x LDG.32 G[s] + A[s]
        {
            const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 0);
            const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 0);
            const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
            const uint32_t gs0 = ah::ld_h2<H>(G, sbase + 0L * 32), as0 = ah::ld_h2<H>(A, sbase + 0L * 32);
            const uint32_t gs1 = ah::ld_h2<H>(G, sbase + 1L * 32), as1 = ah::ld_h2<H>(A, sbase + 1L * 32);
            const uint32_t gs2 = ah::ld_h2<H>(G, sbase + 2L * 32), as2 = ah::ld_h2<H>(A, sbase + 2L * 32);
            const uint32_t gs3 = ah::ld_h2<H>(G, sbase + 3L * 32), as3 = ah::ld_h2<H>(A, sbase + 3L * 32);
            float tvx = 0.f, tvy = 0.f, rvx = 0.f, rvy = 0.f, aynx = 0.f, ayny = 0.f, aysx = 0.f, aysy = 0.f;
            accum_v(gn[0], an[0], gs0, as0, v0.x, v1.x, v2.x, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            finish_l(0, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            tvx = tvy = rvx = rvy = aynx = ayny = aysx = aysy = 0.f;
            accum_v(gn[1], an[1], gs1, as1, v0.y, v1.y, v2.y, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            accum_v(gn[2], an[2], gs2, as2, v0.z, v1.z, v2.z, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            accum_v(gn[3], an[3], gs3, as3, v0.w, v1.w, v2.w, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            finish_l(1, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
        }
        // Chunks 1..3 (m=4..15): batch-issue 4x LDG.32 per chunk
        {
            float tvx = 0.f, tvy = 0.f, rvx = 0.f, rvy = 0.f, aynx = 0.f, ayny = 0.f, aysx = 0.f, aysy = 0.f;
            {
                const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 4);
                const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 4);
                const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                const uint32_t gs4 = ah::ld_h2<H>(G, sbase + 4L * 32), as4 = ah::ld_h2<H>(A, sbase + 4L * 32);
                const uint32_t gs5 = ah::ld_h2<H>(G, sbase + 5L * 32), as5 = ah::ld_h2<H>(A, sbase + 5L * 32);
                const uint32_t gs6 = ah::ld_h2<H>(G, sbase + 6L * 32), as6 = ah::ld_h2<H>(A, sbase + 6L * 32);
                const uint32_t gs7 = ah::ld_h2<H>(G, sbase + 7L * 32), as7 = ah::ld_h2<H>(A, sbase + 7L * 32);
                accum_v(gn[4], an[4], gs4, as4, v0.x, v1.x, v2.x, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_v(gn[5], an[5], gs5, as5, v0.y, v1.y, v2.y, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_v(gn[6], an[6], gs6, as6, v0.z, v1.z, v2.z, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_v(gn[7], an[7], gs7, as7, v0.w, v1.w, v2.w, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            }
            {
                const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 8);
                const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 8);
                const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
                const uint32_t gs8 = ah::ld_h2<H>(G, sbase + 8L * 32), as8 = ah::ld_h2<H>(A, sbase + 8L * 32);
                const uint32_t gs9 = ah::ld_h2<H>(G, sbase + 9L * 32), as9 = ah::ld_h2<H>(A, sbase + 9L * 32);
                const uint32_t gsA = ah::ld_h2<H>(G, sbase + 10L * 32), asA = ah::ld_h2<H>(A, sbase + 10L * 32);
                const uint32_t gsB = ah::ld_h2<H>(G, sbase + 11L * 32), asB = ah::ld_h2<H>(A, sbase + 11L * 32);
                accum_v(gn[8], an[8], gs8, as8, v0.x, v1.x, v2.x, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                finish_l(2, 1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                tvx = tvy = rvx = rvy = aynx = ayny = aysx = aysy = 0.f;
                accum_v(gn[9], an[9], gs9, as9, v0.y, v1.y, v2.y, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_v(gn[10], an[10], gsA, asA, v0.z, v1.z, v2.z, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_v(gn[11], an[11], gsB, asB, v0.w, v1.w, v2.w, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            }
            {
                const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + 12);
                const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + 12);
                const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                const uint32_t gsC = ah::ld_h2<H>(G, sbase + 12L * 32), asC = ah::ld_h2<H>(A, sbase + 12L * 32);
                const uint32_t gsD = ah::ld_h2<H>(G, sbase + 13L * 32), asD = ah::ld_h2<H>(A, sbase + 13L * 32);
                const uint32_t gsE = ah::ld_h2<H>(G, sbase + 14L * 32), asE = ah::ld_h2<H>(A, sbase + 14L * 32);
                const uint32_t gsF = ah::ld_h2<H>(G, sbase + 15L * 32), asF = ah::ld_h2<H>(A, sbase + 15L * 32);
                accum_v(gn[12], an[12], gsC, asC, v0.x, v1.x, v2.x, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_v(gn[13], an[13], gsD, asD, v0.y, v1.y, v2.y, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_v(gn[14], an[14], gsE, asE, v0.z, v1.z, v2.z, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
                accum_v(gn[15], an[15], gsF, asF, v0.w, v1.w, v2.w, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
            }
            finish_l(3, -1.f, tvx, tvy, rvx, rvy, aynx, ayny, aysx, aysy);
        }
        atomicAdd(Xd + s * 64 + 2 * lane, xd_s.x);
        atomicAdd(Xd + s * 64 + 2 * lane + 1, xd_s.y);
    }
    if (prev_kidx >= 0) flush_gc(prev_kidx);
    atomicAdd(Xd + (long)n * 64 + 2 * lane, xd_n.x);
    atomicAdd(Xd + (long)n * 64 + 2 * lane + 1, xd_n.y);
}

// ============================================================================ 3. k_slot_xd_h2 (opt-in bit 1, rejected)
template <typename H, bool XB, int ES, bool POLY_BWD, bool EMETA>
__global__ void __launch_bounds__(256) k_slot_xd_h2(
    const int* __restrict__ rptr, const int* __restrict__ snd, const int* __restrict__ ps,
    const int* __restrict__ sp_idx, const float* __restrict__ sp_u, const H* __restrict__ G,
    const H* __restrict__ A, const float* __restrict__ X, const float* __restrict__ Xb,
    const float* __restrict__ YwP, const float* __restrict__ GyP, const float* __restrict__ YP,
    const float* __restrict__ C_kh, float* __restrict__ gR_k, float* __restrict__ gT_k, float* __restrict__ Xd,
    float sp_h, float sp_invh, int N, const int4* __restrict__ emeta) {
    const int lane = threadIdx.x & 31;
    const int wblk = threadIdx.x >> 5;
    const int grp = (ES > 1) ? (wblk % ES) : 0;
    const int n = blockIdx.x * (8 / ES) + (wblk / ES);
    if (n >= N) return;

    uint32_t gn[16], an[16];
    const long nbase = (long)n * 16 * 32 + lane;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        gn[m] = ah::ld_h2<H>(G, nbase + (long)m * 32);
        an[m] = ah::ld_h2<H>(A, nbase + (long)m * 32);
    }
    const float2* X2 = reinterpret_cast<const float2*>(X);
    const float2* Xb2 = reinterpret_cast<const float2*>(Xb);
    const uint4* Ck4 = reinterpret_cast<const uint4*>(C_kh);
    const float2 xn = X2[(long)n * 32 + lane];
    const float2 xbn = XB ? Xb2[(long)n * 32 + lane] : make_float2(0.f, 0.f);
    float2 xd_n = make_float2(0.f, 0.f);

    const int e0_n = rptr[n], e1_n = rptr[n + 1];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = EMETA ? (ES > 1 ? e0_n + (cnt_n * grp) / ES : e0_n) : (e0_n + grp);
    const int e_end = EMETA ? (ES > 1 ? e0_n + (cnt_n * (grp + 1)) / ES : e1_n) : e1_n;
    const int e_step = EMETA ? 1 : ES;

    for (int e = e_beg; e < e_end; e += e_step) {
        int ps_e, kidx;
        long p, s;
        float u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            if (ps_e & 1) continue;
            kidx = em.z;
            if (kidx < 0) continue;
            p = ps_e >> 1;
            s = em.y;
            u = __int_as_float(em.w);
        } else {
            ps_e = ps[e];
            if (ps_e & 1) continue;
            p = ps_e >> 1;
            kidx = sp_idx[p];
            if (kidx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        const float2 xs = X2[s * 32 + lane];
        const float2 xbs = XB ? Xb2[s * 32 + lane] : make_float2(0.f, 0.f);
        const long sbase = s * 16 * 32 + lane;
        float2 tv[4], rv[4], ayn[4], ays[4];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            tv[l] = make_float2(0.f, 0.f);
            rv[l] = make_float2(0.f, 0.f);
            ayn[l] = make_float2(0.f, 0.f);
            ays[l] = make_float2(0.f, 0.f);
        }
#pragma unroll
        for (int m0 = 0; m0 < 16; m0 += 4) {
            const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + m0);
            const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + m0);
            const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + m0);
            const float yw4[4] = {v0.x, v0.y, v0.z, v0.w};
            const float gy4[4] = {v1.x, v1.y, v1.z, v1.w};
            const float y4[4] = {v2.x, v2.y, v2.z, v2.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int m = m0 + j;
                const int l = lofm_h2(m);
                const float sgn2 = (l & 1) ? -1.f : 1.f;
                const float2 gn2 = ah::unpack2<H>(gn[m]);
                const float2 an2 = ah::unpack2<H>(an[m]);
                const float2 gs2 = ah::unpack2<H>(ah::ld_h2<H>(G, sbase + (long)m * 32));
                const float2 as2 = ah::unpack2<H>(ah::ld_h2<H>(A, sbase + (long)m * 32));
                const float gx = gn2.x * xs.x + sgn2 * gs2.x * xn.x;
                const float gyc = gn2.y * xs.y + sgn2 * gs2.y * xn.y;
                const float ywv = yw4[j], gyv = gy4[j], yv = y4[j];
                tv[l].x += gx * ywv;
                tv[l].y += gyc * ywv;
                if constexpr (XB) {
                    rv[l].x += gx * gyv + (gn2.x * xbs.x + sgn2 * gs2.x * xbn.x) * yv;
                    rv[l].y += gyc * gyv + (gn2.y * xbs.y + sgn2 * gs2.y * xbn.y) * yv;
                } else {
                    rv[l].x += gx * gyv;
                    rv[l].y += gyc * gyv;
                }
                ayn[l].x += an2.x * yv; ayn[l].y += an2.y * yv;
                ays[l].x += as2.x * yv; ays[l].y += as2.y * yv;
            }
        }
        float h00 = 0.f, h10 = 0.f, h11 = 0.f, d00 = 0.f, d10 = 0.f, d11 = 0.f;
        if constexpr (!POLY_BWD) mspl::hermite_rt<float>(u, sp_h, sp_invh, h00, h10, h11, d00, d10, d11);
        const float h01 = 1.f - h00;
        const float u2 = u * u, u3 = u2 * u;
        const float ui = u * sp_invh, u2i = u2 * sp_invh;
        const long cbase = (long)kidx * 4 * 32 + lane;
        const long k0 = (long)kidx * 4 * 64 + 2 * lane, k1 = k0 + 4 * 64;
        float2 xd_s = make_float2(0.f, 0.f);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float sgn2 = (l & 1) ? -1.f : 1.f;
            float rx, ry;
            eval_c4_pair(Ck4[cbase + l * 32], u, rx, ry);
            xd_s.x += rx * ayn[l].x;
            xd_s.y += ry * ayn[l].y;
            xd_n.x += sgn2 * rx * ays[l].x;
            xd_n.y += sgn2 * ry * ays[l].y;
            const float rvx = rv[l].x + ayn[l].x * xs.x + sgn2 * ays[l].x * xn.x;
            const float rvy = rv[l].y + ayn[l].y * xs.y + sgn2 * ays[l].y * xn.y;
            const float tvx = tv[l].x, tvy = tv[l].y;
            if constexpr (POLY_BWD) {
                const long base = ((long)kidx * 4 + l) * 4 * 64 + 2 * lane;
                atomicAdd(gR_k + base + 0 * 64, rvx);
                atomicAdd(gR_k + base + 0 * 64 + 1, rvy);
                atomicAdd(gR_k + base + 1 * 64, u * rvx + sp_invh * tvx);
                atomicAdd(gR_k + base + 1 * 64 + 1, u * rvy + sp_invh * tvy);
                atomicAdd(gR_k + base + 2 * 64, u2 * rvx + 2.f * ui * tvx);
                atomicAdd(gR_k + base + 2 * 64 + 1, u2 * rvy + 2.f * ui * tvy);
                atomicAdd(gR_k + base + 3 * 64, u3 * rvx + 3.f * u2i * tvx);
                atomicAdd(gR_k + base + 3 * 64 + 1, u3 * rvy + 3.f * u2i * tvy);
            } else {
                atomicAdd(gR_k + k0 + l * 64, h00 * rvx + d00 * tvx);
                atomicAdd(gR_k + k0 + l * 64 + 1, h00 * rvy + d00 * tvy);
                atomicAdd(gR_k + k1 + l * 64, h01 * rvx - d00 * tvx);
                atomicAdd(gR_k + k1 + l * 64 + 1, h01 * rvy - d00 * tvy);
                atomicAdd(gT_k + k0 + l * 64, h10 * rvx + d10 * tvx);
                atomicAdd(gT_k + k0 + l * 64 + 1, h10 * rvy + d10 * tvy);
                atomicAdd(gT_k + k1 + l * 64, h11 * rvx + d11 * tvx);
                atomicAdd(gT_k + k1 + l * 64 + 1, h11 * rvy + d11 * tvy);
            }
        }
        atomicAdd(Xd + s * 64 + 2 * lane, xd_s.x);
        atomicAdd(Xd + s * 64 + 2 * lane + 1, xd_s.y);
    }
    if (e_end > e_beg) {
        atomicAdd(Xd + (long)n * 64 + 2 * lane, xd_n.x);
        atomicAdd(Xd + (long)n * 64 + 2 * lane + 1, xd_n.y);
    }
}

// ============================================================================ 4. k_force_pair_h2 (1 warp = 64 ch!)
// Wave 20 ZIP_G01=true: merges t=0 & t=1 into ONE single m=0..15 pass reading fused uint2 G01 (16x batched LDG.64
// instead of 32x LDG.32 -> half LSU ops & L1TEX wavefronts, YP read ONCE!).
template <typename H, bool EMETA, bool ZIP_G01 = false, bool USE_HFMA2 = false, bool SHELL = false>
__global__ void __launch_bounds__(128) k_force_pair_h2(
    const int* __restrict__ rptr, const int* __restrict__ rptr0, const int* __restrict__ snd,
    const int* __restrict__ ps, const int* __restrict__ sp_idx, const float* __restrict__ sp_u,
    const int* __restrict__ ngraph, const float* __restrict__ YP, const float* __restrict__ UP,
    const float* __restrict__ vec, const H* __restrict__ G0, const float* __restrict__ C_kh0,
    const float* __restrict__ X0, const H* __restrict__ G1, const float* __restrict__ C_kh1,
    const float* __restrict__ X1, float* F, float* mv, float sp_invh, int N, int fes,
    const int* __restrict__ norder, const int4* __restrict__ emeta, const uint2* __restrict__ G01 = nullptr,
    int k0_2 = 999999, int k0_3 = 999999, int k1_2 = 999999, int k1_3 = 999999) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wid = blockIdx.x * 4 + warp;
    const int nn = wid / fes, wsub = wid % fes;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;

    const int e0_n = rptr[n];
    const int e1_n = EMETA ? rptr0[n] : rptr[n + 1];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = EMETA ? (fes > 1 ? e0_n + (cnt_n * wsub) / fes : e0_n) : (e0_n + wsub);
    const int e_end = EMETA ? (fes > 1 ? e0_n + (cnt_n * (wsub + 1)) / fes : e1_n) : e1_n;
    const int e_step = EMETA ? 1 : fes;
    if (e_beg >= e_end) return;

    uint32_t gn0[16], gn1[16];
    const long nbase = (long)n * 16 * 32 + lane;
    if constexpr (ZIP_G01) {
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const uint2 v = G01[nbase + (long)m * 32];
            gn0[m] = v.x;
            gn1[m] = v.y;
        }
    } else {
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            gn0[m] = ah::ld_h2<H>(G0, nbase + (long)m * 32);
            gn1[m] = ah::ld_h2<H>(G1, nbase + (long)m * 32);
        }
    }
    const float2* X0_2 = reinterpret_cast<const float2*>(X0);
    const float2* X1_2 = reinterpret_cast<const float2*>(X1);
    const uint4* Ck0_4 = reinterpret_cast<const uint4*>(C_kh0);
    const uint4* Ck1_4 = reinterpret_cast<const uint4*>(C_kh1);
    const float2 xn0 = X0_2[(long)n * 32 + lane];
    const float2 xn1 = X1_2[(long)n * 32 + lane];

    __shared__ float sv[4][12];
    __shared__ float sgp[4][16];
    if (lane < 12) sv[warp][lane] = 0.f;
    bool any_pair = false;

    for (int e = e_beg; e < e_end; e += e_step) {
        int ps_e, kidx;
        long p, s;
        float u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            if (ps_e & 1) continue;
            kidx = em.z;
            if (kidx < 0) continue;
            p = ps_e >> 1;
            s = em.y;
            u = __int_as_float(em.w);
        } else {
            ps_e = ps[e];
            if (ps_e & 1) continue;
            p = ps_e >> 1;
            kidx = sp_idx[p];
            if (kidx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        any_pair = true;
        float gp[16];
        float sp = 0.f;
        const long sbase = s * 16 * 32 + lane;
        const long cbase = (long)kidx * 4 * 32 + lane;

        if constexpr (ZIP_G01) {
            const float2 xs0 = X0_2[s * 32 + lane], xs1 = X1_2[s * 32 + lane];
            if constexpr (USE_HFMA2 && std::is_same<H, __half>::value) {
                const __half2 xs0_h = __float22half2_rn(xs0), xs1_h = __float22half2_rn(xs1);
                const __half2 xn0_p = __float22half2_rn(xn0), xn0_m = __hneg2(xn0_p);
                const __half2 xn1_p = __float22half2_rn(xn1), xn1_m = __hneg2(xn1_p);
                const __half2 z_h = __float2half2_rn(0.f);
                auto do_m2h = [&](int m, float yv, uint2 g01, __half2 r0_h, __half2 r1_h,
                                  __half2 xn0_s, __half2 xn1_s, __half2& q0_h, __half2& q1_h) {
                    const __half2 g0n = *reinterpret_cast<const __half2*>(&gn0[m]);
                    const __half2 g0s = *reinterpret_cast<const __half2*>(&g01.x);
                    const __half2 w0 = __hfma2(g0n, xs0_h, __hmul2(g0s, xn0_s));
                    const __half2 g1n = *reinterpret_cast<const __half2*>(&gn1[m]);
                    const __half2 g1s = *reinterpret_cast<const __half2*>(&g01.y);
                    const __half2 w1 = __hfma2(g1n, xs1_h, __hmul2(g1s, xn1_s));
                    const float2 gpf = __half22float2(__hfma2(w0, r0_h, __hmul2(w1, r1_h)));
                    gp[m] = gpf.x + gpf.y;
                    const __half2 yh = __float2half2_rn(yv);
                    q0_h = __hfma2(yh, w0, q0_h);
                    q1_h = __hfma2(yh, w1, q1_h);
                };
                {
                    const float4 vy0 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
                    const uint2 v0 = G01[sbase + 0L * 32], v1 = G01[sbase + 1L * 32];
                    const uint2 v2 = G01[sbase + 2L * 32], v3 = G01[sbase + 3L * 32];
                    float rx0, ry0, tx0, ty0, rx1, ry1, tx1, ty1;
                    eval_c4_pair_rt(Ck0_4[cbase + 0 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                    eval_c4_pair_rt(Ck1_4[cbase + 0 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                    __half2 q0_h = z_h, q1_h = z_h;
                    do_m2h(0, vy0.x, v0, __floats2half2_rn(rx0, ry0), __floats2half2_rn(rx1, ry1), xn0_p, xn1_p, q0_h, q1_h);
                    { const float2 f0 = __half22float2(q0_h), f1 = __half22float2(q1_h); sp += (f0.x * tx0 + f0.y * ty0) + (f1.x * tx1 + f1.y * ty1); }

                    q0_h = z_h; q1_h = z_h;
                    eval_c4_pair_rt(Ck0_4[cbase + 1 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                    eval_c4_pair_rt(Ck1_4[cbase + 1 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                    const __half2 r0_1 = __floats2half2_rn(rx0, ry0), r1_1 = __floats2half2_rn(rx1, ry1);
                    do_m2h(1, vy0.y, v1, r0_1, r1_1, xn0_m, xn1_m, q0_h, q1_h);
                    do_m2h(2, vy0.z, v2, r0_1, r1_1, xn0_m, xn1_m, q0_h, q1_h);
                    do_m2h(3, vy0.w, v3, r0_1, r1_1, xn0_m, xn1_m, q0_h, q1_h);
                    { const float2 f0 = __half22float2(q0_h), f1 = __half22float2(q1_h); sp += (f0.x * tx0 + f0.y * ty0) + (f1.x * tx1 + f1.y * ty1); }
                }
                {
                    float rx0, ry0, tx0, ty0, rx1, ry1, tx1, ty1;
                    eval_c4_pair_rt(Ck0_4[cbase + 2 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                    eval_c4_pair_rt(Ck1_4[cbase + 2 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                    const __half2 r0_2 = __floats2half2_rn(rx0, ry0), r1_2 = __floats2half2_rn(rx1, ry1);
                    __half2 q0_h = z_h, q1_h = z_h;
                    {
                        const float4 vy1 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                        const uint2 v4 = G01[sbase + 4L * 32], v5 = G01[sbase + 5L * 32];
                        const uint2 v6 = G01[sbase + 6L * 32], v7 = G01[sbase + 7L * 32];
                        do_m2h(4, vy1.x, v4, r0_2, r1_2, xn0_p, xn1_p, q0_h, q1_h);
                        do_m2h(5, vy1.y, v5, r0_2, r1_2, xn0_p, xn1_p, q0_h, q1_h);
                        do_m2h(6, vy1.z, v6, r0_2, r1_2, xn0_p, xn1_p, q0_h, q1_h);
                        do_m2h(7, vy1.w, v7, r0_2, r1_2, xn0_p, xn1_p, q0_h, q1_h);
                    }
                    const float4 vy2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
                    const uint2 v8 = G01[sbase + 8L * 32], v9 = G01[sbase + 9L * 32];
                    const uint2 vA = G01[sbase + 10L * 32], vB = G01[sbase + 11L * 32];
                    do_m2h(8, vy2.x, v8, r0_2, r1_2, xn0_p, xn1_p, q0_h, q1_h);
                    { const float2 f0 = __half22float2(q0_h), f1 = __half22float2(q1_h); sp += (f0.x * tx0 + f0.y * ty0) + (f1.x * tx1 + f1.y * ty1); }

                    q0_h = z_h; q1_h = z_h;
                    eval_c4_pair_rt(Ck0_4[cbase + 3 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                    eval_c4_pair_rt(Ck1_4[cbase + 3 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                    const __half2 r0_3 = __floats2half2_rn(rx0, ry0), r1_3 = __floats2half2_rn(rx1, ry1);
                    do_m2h(9, vy2.y, v9, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                    do_m2h(10, vy2.z, vA, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                    do_m2h(11, vy2.w, vB, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                    {
                        const float4 vy3 = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                        const uint2 vC = G01[sbase + 12L * 32], vD = G01[sbase + 13L * 32];
                        const uint2 vE = G01[sbase + 14L * 32], vF = G01[sbase + 15L * 32];
                        do_m2h(12, vy3.x, vC, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                        do_m2h(13, vy3.y, vD, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                        do_m2h(14, vy3.z, vE, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                        do_m2h(15, vy3.w, vF, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                    }
                    { const float2 f0 = __half22float2(q0_h), f1 = __half22float2(q1_h); sp += (f0.x * tx0 + f0.y * ty0) + (f1.x * tx1 + f1.y * ty1); }
                }
            } else {
                auto do_m2 = [&](int m, float yv, uint2 g01, float rx0, float ry0, float rx1, float ry1, float sgn2,
                                 float& q0x, float& q0y, float& q1x, float& q1y) {
                    const float2 g0n = ah::unpack2<H>(gn0[m]), g0s = ah::unpack2<H>(g01.x);
                    const float w0x = g0n.x * xs0.x + sgn2 * g0s.x * xn0.x;
                    const float w0y = g0n.y * xs0.y + sgn2 * g0s.y * xn0.y;
                    const float2 g1n = ah::unpack2<H>(gn1[m]), g1s = ah::unpack2<H>(g01.y);
                    const float w1x = g1n.x * xs1.x + sgn2 * g1s.x * xn1.x;
                    const float w1y = g1n.y * xs1.y + sgn2 * g1s.y * xn1.y;
                    gp[m] = (w0x * rx0 + w0y * ry0) + (w1x * rx1 + w1y * ry1);
                    q0x += yv * w0x; q0y += yv * w0y;
                    q1x += yv * w1x; q1y += yv * w1y;
                };
                {
                    const float4 vy0 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
                    const uint2 v0 = G01[sbase + 0L * 32], v1 = G01[sbase + 1L * 32];
                    const uint2 v2 = G01[sbase + 2L * 32], v3 = G01[sbase + 3L * 32];
                    float rx0, ry0, tx0, ty0, rx1, ry1, tx1, ty1;
                    eval_c4_pair_rt(Ck0_4[cbase + 0 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                    eval_c4_pair_rt(Ck1_4[cbase + 0 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    do_m2(0, vy0.x, v0, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                    sp += (q0x * tx0 + q0y * ty0) + (q1x * tx1 + q1y * ty1);

                    q0x = q0y = q1x = q1y = 0.f;
                    eval_c4_pair_rt(Ck0_4[cbase + 1 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                    eval_c4_pair_rt(Ck1_4[cbase + 1 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                    do_m2(1, vy0.y, v1, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                    do_m2(2, vy0.z, v2, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                    do_m2(3, vy0.w, v3, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                    sp += (q0x * tx0 + q0y * ty0) + (q1x * tx1 + q1y * ty1);
                }
                {
                    const bool do0_2 = !SHELL || (kidx < k0_2);
                    const bool do1_2 = !SHELL || (kidx < k1_2);
                    const bool do0_3 = !SHELL || (kidx < k0_3);
                    const bool do1_3 = !SHELL || (kidx < k1_3);
                    if (!do0_2 && !do1_2 && !do0_3 && !do1_3) {
#pragma unroll
                        for (int m = 4; m < 16; ++m) gp[m] = 0.f;
                    } else {
                        const float4 vy2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
                        if (do0_2 || do1_2) {
                            float rx0 = 0.f, ry0 = 0.f, tx0 = 0.f, ty0 = 0.f;
                            float rx1 = 0.f, ry1 = 0.f, tx1 = 0.f, ty1 = 0.f;
                            if (do0_2) eval_c4_pair_rt(Ck0_4[cbase + 2 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                            if (do1_2) eval_c4_pair_rt(Ck1_4[cbase + 2 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                            float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                            {
                                const float4 vy1 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                                const uint2 v4 = G01[sbase + 4L * 32], v5 = G01[sbase + 5L * 32];
                                const uint2 v6 = G01[sbase + 6L * 32], v7 = G01[sbase + 7L * 32];
                                do_m2(4, vy1.x, v4, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                                do_m2(5, vy1.y, v5, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                                do_m2(6, vy1.z, v6, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                                do_m2(7, vy1.w, v7, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                            }
                            const uint2 v8 = G01[sbase + 8L * 32];
                            do_m2(8, vy2.x, v8, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                            if (do0_2) sp += q0x * tx0 + q0y * ty0;
                            if (do1_2) sp += q1x * tx1 + q1y * ty1;
                        } else {
                            gp[4] = gp[5] = gp[6] = gp[7] = gp[8] = 0.f;
                        }
                        if (do0_3 || do1_3) {
                            float rx0 = 0.f, ry0 = 0.f, tx0 = 0.f, ty0 = 0.f;
                            float rx1 = 0.f, ry1 = 0.f, tx1 = 0.f, ty1 = 0.f;
                            if (do0_3) eval_c4_pair_rt(Ck0_4[cbase + 3 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                            if (do1_3) eval_c4_pair_rt(Ck1_4[cbase + 3 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                            float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                            const uint2 v9 = G01[sbase + 9L * 32], vA = G01[sbase + 10L * 32], vB = G01[sbase + 11L * 32];
                            do_m2(9, vy2.y, v9, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                            do_m2(10, vy2.z, vA, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                            do_m2(11, vy2.w, vB, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                            {
                                const float4 vy3 = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                                const uint2 vC = G01[sbase + 12L * 32], vD = G01[sbase + 13L * 32];
                                const uint2 vE = G01[sbase + 14L * 32], vF = G01[sbase + 15L * 32];
                                do_m2(12, vy3.x, vC, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                                do_m2(13, vy3.y, vD, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                                do_m2(14, vy3.z, vE, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                                do_m2(15, vy3.w, vF, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                            }
                            if (do0_3) sp += q0x * tx0 + q0y * ty0;
                            if (do1_3) sp += q1x * tx1 + q1y * ty1;
                        } else {
                            gp[9] = gp[10] = gp[11] = gp[12] = gp[13] = gp[14] = gp[15] = 0.f;
                        }
                    }
                }
            }
        } else {
#pragma unroll
        for (int m = 0; m < 16; ++m) gp[m] = 0.f;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const H* Gs = t ? G1 : G0;
            const uint4* Ck = t ? Ck1_4 : Ck0_4;
            const float2 xs = (t ? X1_2 : X0_2)[s * 32 + lane];
            const float2 xn = t ? xn1 : xn0;
            const uint32_t* gn = t ? gn1 : gn0;

            auto do_m = [&](int m, float yv, float rx, float ry, float sgn2, float& qx, float& qy) {
                const float2 gn2 = ah::unpack2<H>(gn[m]);
                const float2 gs2 = ah::unpack2<H>(ah::ld_h2<H>(Gs, sbase + (long)m * 32));
                const float wx = gn2.x * xs.x + sgn2 * gs2.x * xn.x;
                const float wy = gn2.y * xs.y + sgn2 * gs2.y * xn.y;
                gp[m] += wx * rx + wy * ry;
                qx += yv * wx;
                qy += yv * wy;
            };
            {
                const float4 vy0 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
                float rx, ry, tx, ty, qx = 0.f, qy = 0.f;
                eval_c4_pair_rt(Ck[cbase + 0 * 32], u, sp_invh, rx, ry, tx, ty);
                do_m(0, vy0.x, rx, ry, 1.f, qx, qy);
                sp += qx * tx + qy * ty;

                qx = 0.f; qy = 0.f;
                eval_c4_pair_rt(Ck[cbase + 1 * 32], u, sp_invh, rx, ry, tx, ty);
                do_m(1, vy0.y, rx, ry, -1.f, qx, qy);
                do_m(2, vy0.z, rx, ry, -1.f, qx, qy);
                do_m(3, vy0.w, rx, ry, -1.f, qx, qy);
                sp += qx * tx + qy * ty;
            }
            {
                float rx, ry, tx, ty, qx = 0.f, qy = 0.f;
                eval_c4_pair_rt(Ck[cbase + 2 * 32], u, sp_invh, rx, ry, tx, ty);
                {
                    const float4 vy1 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                    do_m(4, vy1.x, rx, ry, 1.f, qx, qy);
                    do_m(5, vy1.y, rx, ry, 1.f, qx, qy);
                    do_m(6, vy1.z, rx, ry, 1.f, qx, qy);
                    do_m(7, vy1.w, rx, ry, 1.f, qx, qy);
                }
                const float4 vy2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
                do_m(8, vy2.x, rx, ry, 1.f, qx, qy);
                sp += qx * tx + qy * ty;

                qx = 0.f; qy = 0.f;
                eval_c4_pair_rt(Ck[cbase + 3 * 32], u, sp_invh, rx, ry, tx, ty);
                do_m(9, vy2.y, rx, ry, -1.f, qx, qy);
                do_m(10, vy2.z, rx, ry, -1.f, qx, qy);
                do_m(11, vy2.w, rx, ry, -1.f, qx, qy);
                {
                    const float4 vy3 = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                    do_m(12, vy3.x, rx, ry, -1.f, qx, qy);
                    do_m(13, vy3.y, rx, ry, -1.f, qx, qy);
                    do_m(14, vy3.z, rx, ry, -1.f, qx, qy);
                    do_m(15, vy3.w, rx, ry, -1.f, qx, qy);
                }
                sp += qx * tx + qy * ty;
            }
        }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        float a8[8];
        const bool up = lane & 16;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            a8[j] = (up ? gp[j + 8] : gp[j]) + __shfl_xor_sync(0xffffffffu, up ? gp[j] : gp[j + 8], 16);
        float a4[4];
        const bool b8 = lane & 8;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a4[j] = (b8 ? a8[j + 4] : a8[j]) + __shfl_xor_sync(0xffffffffu, b8 ? a8[j] : a8[j + 4], 8);
        float a2[2];
        const bool b4 = lane & 4;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            a2[j] = (b4 ? a4[j + 2] : a4[j]) + __shfl_xor_sync(0xffffffffu, b4 ? a4[j] : a4[j + 2], 4);
        const bool b2 = lane & 2;
        float a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 2);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        int src = 0;
        if (lane < 16) {
            const int m = lane;
            src = ((m & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (((m >> 3) & 1) << 4);
        }
        const float gm = __shfl_sync(0xffffffffu, a1, src);
        if (lane < 16) sgp[warp][lane] = gm;
        __syncwarp();
        float g0 = 0.f, g1 = 0.f, g2 = 0.f;
        if (lane == 0) {
            const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
            float jx, jy_, jz;
            sh16_jyt<float>(up4.x, up4.y, up4.z, up4.w, 1.f, YP + 16L * p, sgp[warp], jx, jy_, jz);
            g0 = jx + sp * up4.x; g1 = jy_ + sp * up4.y; g2 = jz + sp * up4.z;
            sv[warp][9] -= g0; sv[warp][10] -= g1; sv[warp][11] -= g2;
            float vx, vy_, vz;
            if (vec) {
                vx = vec[3L * e]; vy_ = vec[3L * e + 1]; vz = vec[3L * e + 2];
            } else {
                const float r_len = 1.f / up4.w;
                vx = up4.x * r_len; vy_ = up4.y * r_len; vz = up4.z * r_len;
            }
            sv[warp][0] += vx * g0; sv[warp][1] += vx * g1; sv[warp][2] += vx * g2;
            sv[warp][3] += vy_ * g0; sv[warp][4] += vy_ * g1; sv[warp][5] += vy_ * g2;
            sv[warp][6] += vz * g0; sv[warp][7] += vz * g1; sv[warp][8] += vz * g2;
        }
        g0 = __shfl_sync(0xffffffffu, g0, 0);
        g1 = __shfl_sync(0xffffffffu, g1, 0);
        g2 = __shfl_sync(0xffffffffu, g2, 0);
        if (lane < 3) atomicAdd(F + 3L * s + lane, lane == 0 ? g0 : (lane == 1 ? g1 : g2));
    }
    if (any_pair) {
        __syncwarp();
        if (lane < 3) atomicAdd(F + 3L * n + lane, sv[warp][9 + lane]);
        if (lane < 9) atomicAdd(mv + 9L * ngraph[n] + lane, sv[warp][lane]);
    }
}

// ============================================================================ 4b. k_force_dir_h2 (Wave 23: LST_FORCE_DIR=1!)
// Directed-edge force+virial kernel: loops over ALL incoming edges of receiver n (both side 0 & side 1).
// Eliminates 100% of neighbor G0[s]/G1[s]/G01[s] loads (212 MB -> ~25 MB L2 traffic, 8.5x reduction!),
// eliminates ZIP_G01 in seg_gemm, halves H2F2 unpacks, factors xs0/xs1/sgn_l out of m-loop, <=80/96 regs!
template <typename H, bool EMETA, bool SHELL = false>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
__global__ void __launch_bounds__(128, 6) k_force_dir_h2(
#else
__global__ void __launch_bounds__(128, 5) k_force_dir_h2(
#endif
    const int* __restrict__ rptr, const int* __restrict__ snd, const int* __restrict__ ps,
    const int* __restrict__ sp_idx, const float* __restrict__ sp_u,
    const int* __restrict__ ngraph, const float* __restrict__ YP, const float* __restrict__ UP,
    const float* __restrict__ vec, const H* __restrict__ G0, const float* __restrict__ C_kh0,
    const float* __restrict__ X0, const H* __restrict__ G1, const float* __restrict__ C_kh1,
    const float* __restrict__ X1, float* F, float* mv, float sp_invh, int N, int fes,
    const int* __restrict__ norder, const int4* __restrict__ emeta,
    int k0_2 = 999999, int k0_3 = 999999, int k1_2 = 999999, int k1_3 = 999999) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wid = blockIdx.x * 4 + warp;
    const int nn = wid / fes, wsub = wid % fes;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;

    const int e0_n = rptr[n];
    const int e1_n = rptr[n + 1];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = EMETA ? (fes > 1 ? e0_n + (cnt_n * wsub) / fes : e0_n) : (e0_n + wsub);
    const int e_end = EMETA ? (fes > 1 ? e0_n + (cnt_n * (wsub + 1)) / fes : e1_n) : e1_n;
    const int e_step = EMETA ? 1 : fes;
    if (e_beg >= e_end) return;

    uint32_t gn0[16], gn1[16];
    const long nbase = (long)n * 16 * 32 + lane;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        gn0[m] = ah::ld_h2<H>(G0, nbase + (long)m * 32);
        gn1[m] = ah::ld_h2<H>(G1, nbase + (long)m * 32);
    }
    const float2* X0_2 = reinterpret_cast<const float2*>(X0);
    const float2* X1_2 = reinterpret_cast<const float2*>(X1);
    const uint4* Ck0_4 = reinterpret_cast<const uint4*>(C_kh0);
    const uint4* Ck1_4 = reinterpret_cast<const uint4*>(C_kh1);

    __shared__ float sv[4][12];
    __shared__ float sgp[4][16];
    if (lane < 12) sv[warp][lane] = 0.f;
    bool any_edge = false;

    for (int e = e_beg; e < e_end; e += e_step) {
        int ps_e, kidx;
        long p, s;
        float u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            kidx = em.z;
            if (kidx < 0) continue;
            p = ps_e >> 1;
            s = em.y;
            u = __int_as_float(em.w);
        } else {
            ps_e = ps[e];
            p = ps_e >> 1;
            kidx = sp_idx[p];
            if (kidx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        any_edge = true;
        const float sgn = (ps_e & 1) ? -1.f : 1.f;
        const float2 xs0 = X0_2[s * 32 + lane], xs1 = X1_2[s * 32 + lane];
        const long cbase = (long)kidx * 4 * 32 + lane;
        float gp[16];
        float sp = 0.f;

        auto do_md = [&](int m, float yv, float r0x, float r0y, float r1x, float r1y,
                         float& q0x, float& q0y, float& q1x, float& q1y) {
            const float2 g0 = ah::unpack2<H>(gn0[m]), g1 = ah::unpack2<H>(gn1[m]);
            gp[m] = (g0.x * r0x + g0.y * r0y) + (g1.x * r1x + g1.y * r1y);
            q0x += yv * g0.x; q0y += yv * g0.y;
            q1x += yv * g1.x; q1y += yv * g1.y;
        };
        {
            const float4 vy0 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
            float rx0, ry0, tx0, ty0, rx1, ry1, tx1, ty1;
            eval_c4_pair_rt(Ck0_4[cbase + 0 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
            eval_c4_pair_rt(Ck1_4[cbase + 0 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
            float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
            do_md(0, vy0.x, xs0.x * rx0, xs0.y * ry0, xs1.x * rx1, xs1.y * ry1, q0x, q0y, q1x, q1y);
            sp += (q0x * xs0.x * tx0 + q0y * xs0.y * ty0) + (q1x * xs1.x * tx1 + q1y * xs1.y * ty1);

            q0x = q0y = q1x = q1y = 0.f;
            eval_c4_pair_rt(Ck0_4[cbase + 1 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
            eval_c4_pair_rt(Ck1_4[cbase + 1 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
            const float sx0x = sgn * xs0.x, sx0y = sgn * xs0.y, sx1x = sgn * xs1.x, sx1y = sgn * xs1.y;
            const float r0x = sx0x * rx0, r0y = sx0y * ry0, r1x = sx1x * rx1, r1y = sx1y * ry1;
            do_md(1, vy0.y, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
            do_md(2, vy0.z, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
            do_md(3, vy0.w, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
            sp += (q0x * sx0x * tx0 + q0y * sx0y * ty0) + (q1x * sx1x * tx1 + q1y * sx1y * ty1);
        }
        {
            const bool do0_2 = !SHELL || (kidx < k0_2);
            const bool do1_2 = !SHELL || (kidx < k1_2);
            const bool do0_3 = !SHELL || (kidx < k0_3);
            const bool do1_3 = !SHELL || (kidx < k1_3);
            if (!do0_2 && !do1_2 && !do0_3 && !do1_3) {
#pragma unroll
                for (int m = 4; m < 16; ++m) gp[m] = 0.f;
            } else {
                const float4 vy2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
                if (do0_2 || do1_2) {
                    float rx0 = 0.f, ry0 = 0.f, tx0 = 0.f, ty0 = 0.f;
                    float rx1 = 0.f, ry1 = 0.f, tx1 = 0.f, ty1 = 0.f;
                    if (do0_2) eval_c4_pair_rt(Ck0_4[cbase + 2 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                    if (do1_2) eval_c4_pair_rt(Ck1_4[cbase + 2 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                    const float r0x = xs0.x * rx0, r0y = xs0.y * ry0, r1x = xs1.x * rx1, r1y = xs1.y * ry1;
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    {
                        const float4 vy1 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                        do_md(4, vy1.x, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
                        do_md(5, vy1.y, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
                        do_md(6, vy1.z, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
                        do_md(7, vy1.w, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
                    }
                    do_md(8, vy2.x, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
                    if (do0_2) sp += q0x * xs0.x * tx0 + q0y * xs0.y * ty0;
                    if (do1_2) sp += q1x * xs1.x * tx1 + q1y * xs1.y * ty1;
                } else {
                    gp[4] = gp[5] = gp[6] = gp[7] = gp[8] = 0.f;
                }
                if (do0_3 || do1_3) {
                    float rx0 = 0.f, ry0 = 0.f, tx0 = 0.f, ty0 = 0.f;
                    float rx1 = 0.f, ry1 = 0.f, tx1 = 0.f, ty1 = 0.f;
                    if (do0_3) eval_c4_pair_rt(Ck0_4[cbase + 3 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                    if (do1_3) eval_c4_pair_rt(Ck1_4[cbase + 3 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                    const float sx0x = sgn * xs0.x, sx0y = sgn * xs0.y, sx1x = sgn * xs1.x, sx1y = sgn * xs1.y;
                    const float r0x = sx0x * rx0, r0y = sx0y * ry0, r1x = sx1x * rx1, r1y = sx1y * ry1;
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    do_md(9, vy2.y, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
                    do_md(10, vy2.z, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
                    do_md(11, vy2.w, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
                    {
                        const float4 vy3 = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                        do_md(12, vy3.x, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
                        do_md(13, vy3.y, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
                        do_md(14, vy3.z, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
                        do_md(15, vy3.w, r0x, r0y, r1x, r1y, q0x, q0y, q1x, q1y);
                    }
                    if (do0_3) sp += q0x * sx0x * tx0 + q0y * sx0y * ty0;
                    if (do1_3) sp += q1x * sx1x * tx1 + q1y * sx1y * ty1;
                } else {
                    gp[9] = gp[10] = gp[11] = gp[12] = gp[13] = gp[14] = gp[15] = 0.f;
                }
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        float a8[8];
        const bool up = lane & 16;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            a8[j] = (up ? gp[j + 8] : gp[j]) + __shfl_xor_sync(0xffffffffu, up ? gp[j] : gp[j + 8], 16);
        float a4[4];
        const bool b8 = lane & 8;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a4[j] = (b8 ? a8[j + 4] : a8[j]) + __shfl_xor_sync(0xffffffffu, b8 ? a8[j] : a8[j + 4], 8);
        float a2[2];
        const bool b4 = lane & 4;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            a2[j] = (b4 ? a4[j + 2] : a4[j]) + __shfl_xor_sync(0xffffffffu, b4 ? a4[j] : a4[j + 2], 4);
        const bool b2 = lane & 2;
        float a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 2);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        int src = 0;
        if (lane < 16) {
            const int m = lane;
            src = ((m & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (((m >> 3) & 1) << 4);
        }
        const float gm = __shfl_sync(0xffffffffu, a1, src);
        if (lane < 16) sgp[warp][lane] = gm;
        __syncwarp();
        float g0 = 0.f, g1 = 0.f, g2 = 0.f;
        if (lane == 0) {
            const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
            float jx, jy_, jz;
            sh16_jyt<float>(up4.x, up4.y, up4.z, up4.w, 1.f, YP + 16L * p, sgp[warp], jx, jy_, jz);
            g0 = sgn * (jx + sp * up4.x); g1 = sgn * (jy_ + sp * up4.y); g2 = sgn * (jz + sp * up4.z);
            sv[warp][9] -= g0; sv[warp][10] -= g1; sv[warp][11] -= g2;
            float vx, vy_, vz;
            if (vec) {
                vx = vec[3L * e]; vy_ = vec[3L * e + 1]; vz = vec[3L * e + 2];
            } else {
                const float r_len = 1.f / up4.w;
                vx = up4.x * r_len; vy_ = up4.y * r_len; vz = up4.z * r_len;
            }
            sv[warp][0] += vx * g0; sv[warp][1] += vx * g1; sv[warp][2] += vx * g2;
            sv[warp][3] += vy_ * g0; sv[warp][4] += vy_ * g1; sv[warp][5] += vy_ * g2;
            sv[warp][6] += vz * g0; sv[warp][7] += vz * g1; sv[warp][8] += vz * g2;
        }
        g0 = __shfl_sync(0xffffffffu, g0, 0);
        g1 = __shfl_sync(0xffffffffu, g1, 0);
        g2 = __shfl_sync(0xffffffffu, g2, 0);
        if (lane < 3) atomicAdd(F + 3L * s + lane, lane == 0 ? g0 : (lane == 1 ? g1 : g2));
    }
    if (any_edge) {
        __syncwarp();
        if (lane < 3) atomicAdd(F + 3L * n + lane, sv[warp][9 + lane]);
        if (lane < 9) atomicAdd(mv + 9L * ngraph[n] + lane, sv[warp][lane]);
    }
}

// ============================================================================ 4c. k_force_dir_1l (Wave 24: LST_FORCE_DIR=2!)
// Split single-layer directed force+virial (+ fused conv_vjp when DO_VJP=true for Layer 1!):
//   - Loops over unified (n, kidx)-sorted emeta_dir -> caches prev_ck[4] in regs (~85% fewer C_kh reads!).
//   - ZERO G[s] loads (-164 MB L2!), ZERO b.vec loads (vec = sgn * (1/rinv) * u from UP[p]!), ZERO ZIP_G01!
//   - DO_VJP=true (Layer 1): computes x1b = sum_l R_l * q_l in the SAME pass (4 FMAs/edge), deleting conv_vjp (47-56 us)!
//   - T0S_DIR=true (Layer 0, S <= 4): reads X0 from 4 float2 regs T0[0..3] (ZERO X0[s] global loads!).
//   - <=72-80 regs -> 6 blocks/SM = 50% occupancy (2x k_force_pair_h2's 24.7%!).
template <typename H, bool DO_VJP, bool SHELL = false, bool T0S_DIR = false>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
__global__ void __launch_bounds__(128, 6) k_force_dir_1l(
#else
__global__ void __launch_bounds__(128, 5) k_force_dir_1l(
#endif
    const int* __restrict__ rptr, const int4* __restrict__ emeta_dir,
    const int* __restrict__ ngraph, const float* __restrict__ YP, const float* __restrict__ UP,
    const H* __restrict__ G, const float* __restrict__ C_kh, const float* __restrict__ X,
    float* __restrict__ F, float* __restrict__ mv, float* __restrict__ OX,
    float sp_invh, int N, int fes, const int* __restrict__ norder,
    int kcut2 = 999999, int kcut3 = 999999,
    const int* __restrict__ spc = nullptr, const float* __restrict__ T0 = nullptr, int S = 0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wid = blockIdx.x * 4 + warp;
    const int nn = wid / fes, wsub = wid % fes;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;

    const int e0_n = rptr[n];
    const int e1_n = rptr[n + 1];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = (fes > 1) ? (e0_n + (cnt_n * wsub) / fes) : e0_n;
    const int e_end = (fes > 1) ? (e0_n + (cnt_n * (wsub + 1)) / fes) : e1_n;
    if (e_beg >= e_end) return;

    uint32_t gn[16];
    const long nbase = (long)n * 16 * 32 + lane;
#pragma unroll
    for (int m = 0; m < 16; ++m) gn[m] = ah::ld_h2<H>(G, nbase + (long)m * 32);

    const float2* X2 = reinterpret_cast<const float2*>(X);
    const uint4* Ck4 = reinterpret_cast<const uint4*>(C_kh);
    float2 t0r0 = make_float2(0.f, 0.f), t0r1 = make_float2(0.f, 0.f);
    float2 t0r2 = make_float2(0.f, 0.f), t0r3 = make_float2(0.f, 0.f);
    if constexpr (T0S_DIR) {
        const float2* T02 = reinterpret_cast<const float2*>(T0);
        t0r0 = T02[0 * 32 + lane];
        if (S > 1) t0r1 = T02[1 * 32 + lane];
        if (S > 2) t0r2 = T02[2 * 32 + lane];
        if (S > 3) t0r3 = T02[3 * 32 + lane];
    }

    __shared__ float sv[4][12];
    __shared__ float sgp[4][16];
    if (lane < 12) sv[warp][lane] = 0.f;
    bool any_edge = false;

    int prev_kidx = -1;
    uint4 prev_ck[4];

    for (int e = e_beg; e < e_end; ++e) {
        const int4 em = emeta_dir[e];
        const int kidx = em.z;
        if (kidx < 0) continue;
        any_edge = true;
        const int ps_e = em.x;
        const long p = ps_e >> 1;
        const long s = em.y;
        const float u = __int_as_float(em.w);
        const float sgn = (ps_e & 1) ? -1.f : 1.f;
        const bool do_l2 = !SHELL || (kidx < kcut2);
        const bool do_l3 = !SHELL || (kidx < kcut3);

        if (kidx != prev_kidx) {
            const long cbase = (long)kidx * 4 * 32 + lane;
            prev_ck[0] = Ck4[cbase + 0 * 32];
            prev_ck[1] = Ck4[cbase + 1 * 32];
            if (do_l2) prev_ck[2] = Ck4[cbase + 2 * 32];
            if (do_l3) prev_ck[3] = Ck4[cbase + 3 * 32];
            prev_kidx = kidx;
        }

        float2 xs;
        if constexpr (T0S_DIR) {
            const int sp_s = spc[s];
            xs = (sp_s == 0) ? t0r0 : ((sp_s == 1) ? t0r1 : ((sp_s == 2) ? t0r2 : t0r3));
        } else {
            xs = X2[s * 32 + lane];
        }

        float gp[16];
        float sp = 0.f;
        float ox_x = 0.f, ox_y = 0.f;

        auto do_m1 = [&](int m, float yv, float rxs, float rys, float& qx, float& qy) {
            const float2 g2 = ah::unpack2<H>(gn[m]);
            gp[m] = g2.x * rxs + g2.y * rys;
            qx += yv * g2.x;
            qy += yv * g2.y;
        };
        {
            const float4 vy0 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
            float rx, ry, tx, ty;
            eval_c4_pair_rt(prev_ck[0], u, sp_invh, rx, ry, tx, ty);
            float qx = 0.f, qy = 0.f;
            do_m1(0, vy0.x, xs.x * rx, xs.y * ry, qx, qy);
            sp += qx * xs.x * tx + qy * xs.y * ty;
            if constexpr (DO_VJP) { ox_x += rx * qx; ox_y += ry * qy; }

            qx = 0.f; qy = 0.f;
            eval_c4_pair_rt(prev_ck[1], u, sp_invh, rx, ry, tx, ty);
            const float sx = sgn * xs.x, sy = sgn * xs.y;
            const float rxs = sx * rx, rys = sy * ry;
            do_m1(1, vy0.y, rxs, rys, qx, qy);
            do_m1(2, vy0.z, rxs, rys, qx, qy);
            do_m1(3, vy0.w, rxs, rys, qx, qy);
            sp += qx * sx * tx + qy * sy * ty;
            if constexpr (DO_VJP) { ox_x += sgn * rx * qx; ox_y += sgn * ry * qy; }
        }
        if (!do_l2 && !do_l3) {
#pragma unroll
            for (int m = 4; m < 16; ++m) gp[m] = 0.f;
        } else {
            const float4 vy2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
            if (do_l2) {
                float rx, ry, tx, ty;
                eval_c4_pair_rt(prev_ck[2], u, sp_invh, rx, ry, tx, ty);
                const float rxs = xs.x * rx, rys = xs.y * ry;
                float qx = 0.f, qy = 0.f;
                {
                    const float4 vy1 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                    do_m1(4, vy1.x, rxs, rys, qx, qy);
                    do_m1(5, vy1.y, rxs, rys, qx, qy);
                    do_m1(6, vy1.z, rxs, rys, qx, qy);
                    do_m1(7, vy1.w, rxs, rys, qx, qy);
                }
                do_m1(8, vy2.x, rxs, rys, qx, qy);
                sp += qx * xs.x * tx + qy * xs.y * ty;
                if constexpr (DO_VJP) { ox_x += rx * qx; ox_y += ry * qy; }
            } else {
                gp[4] = gp[5] = gp[6] = gp[7] = gp[8] = 0.f;
            }
            if (do_l3) {
                float rx, ry, tx, ty;
                eval_c4_pair_rt(prev_ck[3], u, sp_invh, rx, ry, tx, ty);
                const float sx = sgn * xs.x, sy = sgn * xs.y;
                const float rxs = sx * rx, rys = sy * ry;
                float qx = 0.f, qy = 0.f;
                do_m1(9, vy2.y, rxs, rys, qx, qy);
                do_m1(10, vy2.z, rxs, rys, qx, qy);
                do_m1(11, vy2.w, rxs, rys, qx, qy);
                {
                    const float4 vy3 = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                    do_m1(12, vy3.x, rxs, rys, qx, qy);
                    do_m1(13, vy3.y, rxs, rys, qx, qy);
                    do_m1(14, vy3.z, rxs, rys, qx, qy);
                    do_m1(15, vy3.w, rxs, rys, qx, qy);
                }
                sp += qx * sx * tx + qy * sy * ty;
                if constexpr (DO_VJP) { ox_x += sgn * rx * qx; ox_y += sgn * ry * qy; }
            } else {
                gp[9] = gp[10] = gp[11] = gp[12] = gp[13] = gp[14] = gp[15] = 0.f;
            }
        }
        if constexpr (DO_VJP) {
            atomicAdd(OX + s * 64 + 2 * lane, ox_x);
            atomicAdd(OX + s * 64 + 2 * lane + 1, ox_y);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        float a8[8];
        const bool up = lane & 16;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            a8[j] = (up ? gp[j + 8] : gp[j]) + __shfl_xor_sync(0xffffffffu, up ? gp[j] : gp[j + 8], 16);
        float a4[4];
        const bool b8 = lane & 8;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a4[j] = (b8 ? a8[j + 4] : a8[j]) + __shfl_xor_sync(0xffffffffu, b8 ? a8[j] : a8[j + 4], 8);
        float a2[2];
        const bool b4 = lane & 4;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            a2[j] = (b4 ? a4[j + 2] : a4[j]) + __shfl_xor_sync(0xffffffffu, b4 ? a4[j] : a4[j + 2], 4);
        const bool b2 = lane & 2;
        float a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 2);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        int src = 0;
        if (lane < 16) {
            const int m = lane;
            src = ((m & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (((m >> 3) & 1) << 4);
        }
        const float gm = __shfl_sync(0xffffffffu, a1, src);
        if (lane < 16) sgp[warp][lane] = gm;
        __syncwarp();
        float g0 = 0.f, g1 = 0.f, g2 = 0.f;
        if (lane == 0) {
            const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
            float jx, jy_, jz;
            sh16_jyt<float>(up4.x, up4.y, up4.z, up4.w, 1.f, YP + 16L * p, sgp[warp], jx, jy_, jz);
            const float gx0 = jx + sp * up4.x, gy0 = jy_ + sp * up4.y, gz0 = jz + sp * up4.z;
            g0 = sgn * gx0; g1 = sgn * gy0; g2 = sgn * gz0;
            sv[warp][9] -= g0; sv[warp][10] -= g1; sv[warp][11] -= g2;
            const float r_len = __frcp_rn(up4.w);
            const float vx = r_len * up4.x, vy_ = r_len * up4.y, vz = r_len * up4.z;
            sv[warp][0] += vx * gx0; sv[warp][1] += vx * gy0; sv[warp][2] += vx * gz0;
            sv[warp][3] += vy_ * gx0; sv[warp][4] += vy_ * gy0; sv[warp][5] += vy_ * gz0;
            sv[warp][6] += vz * gx0; sv[warp][7] += vz * gy0; sv[warp][8] += vz * gz0;
        }
        g0 = __shfl_sync(0xffffffffu, g0, 0);
        g1 = __shfl_sync(0xffffffffu, g1, 0);
        g2 = __shfl_sync(0xffffffffu, g2, 0);
        if (lane < 3) atomicAdd(F + 3L * s + lane, lane == 0 ? g0 : (lane == 1 ? g1 : g2));
    }
    if (any_edge) {
        __syncwarp();
        if (lane < 3) atomicAdd(F + 3L * n + lane, sv[warp][9 + lane]);
        if (lane < 9) atomicAdd(mv + 9L * ngraph[n] + lane, sv[warp][lane]);
    }
}

// ============================================================================ 4d. k_force_pair_h2_opt (Wave 25: LST_FORCE_OPT=1!)
// Pair-folded fused L0+L1 force+virial with smem-cached s_gn (frees 32 regs -> <=80-96 regs -> ~2x occ!),
// T0 smem cache (ZERO X0 loads!) and zero b.vec loads (vec = (1/rinv)*u from UP[p]!).
template <typename H, bool SHELL = false, bool T0S_DIR = false>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
__global__ void __launch_bounds__(128, 5) k_force_pair_h2_opt(
#else
__global__ void __launch_bounds__(128, 5) k_force_pair_h2_opt(
#endif
    const int* __restrict__ rptr, const int* __restrict__ rptr0,
    const int* __restrict__ ngraph, const float* __restrict__ YP, const float* __restrict__ UP,
    const float* __restrict__ C_kh0, const float* __restrict__ X0,
    const float* __restrict__ C_kh1, const float* __restrict__ X1,
    float* __restrict__ F, float* __restrict__ mv, float sp_invh, int N, int fes,
    const int* __restrict__ norder, const int4* __restrict__ emeta, const uint2* __restrict__ G01,
    int k0_2 = 999999, int k0_3 = 999999, int k1_2 = 999999, int k1_3 = 999999,
    const int* __restrict__ spc = nullptr, const float* __restrict__ T0 = nullptr, int S = 0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wid = blockIdx.x * 4 + warp;
    const int nn = wid / fes, wsub = wid % fes;

    __shared__ uint2 s_gn[4][16][32];                                  // 16 KB (0 bank conflicts!)
    __shared__ float2 s_t0[T0S_DIR ? 4 : 1][T0S_DIR ? 32 : 1];         // 1 KB when T0S_DIR
    __shared__ float sv[4][12];
    __shared__ float sgp[4][16];
    if constexpr (T0S_DIR) {
        if (warp == 0) {
            const float2* T02 = reinterpret_cast<const float2*>(T0);
            for (int sp = 0; sp < S; ++sp) s_t0[sp][lane] = T02[sp * 32 + lane];
        }
        __syncthreads();
    }
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;

    const int e0_n = rptr[n];
    const int e1_n = rptr0[n];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = (fes > 1) ? (e0_n + (cnt_n * wsub) / fes) : e0_n;
    const int e_end = (fes > 1) ? (e0_n + (cnt_n * (wsub + 1)) / fes) : e1_n;
    if (e_beg >= e_end) return;

    const long nbase = (long)n * 16 * 32 + lane;
#pragma unroll
    for (int m = 0; m < 16; ++m) s_gn[warp][m][lane] = G01[nbase + (long)m * 32];

    const float2* X0_2 = reinterpret_cast<const float2*>(X0);
    const float2* X1_2 = reinterpret_cast<const float2*>(X1);
    const uint4* Ck0_4 = reinterpret_cast<const uint4*>(C_kh0);
    const uint4* Ck1_4 = reinterpret_cast<const uint4*>(C_kh1);
    const float2 xn0 = T0S_DIR ? s_t0[spc[n]][lane] : X0_2[(long)n * 32 + lane];
    const float2 xn1 = X1_2[(long)n * 32 + lane];

    if (lane < 12) sv[warp][lane] = 0.f;
    bool any_pair = false;

    for (int e = e_beg; e < e_end; ++e) {
        const int4 em = emeta[e];
        const int ps_e = em.x;
        if (ps_e & 1) continue;
        const int kidx = em.z;
        if (kidx < 0) continue;
        const long p = ps_e >> 1;
        const long s = em.y;
        const float u = __int_as_float(em.w);
        any_pair = true;
        float gp[16];
        float sp = 0.f;
        const long sbase = s * 16 * 32 + lane;
        const long cbase = (long)kidx * 4 * 32 + lane;
        const float2 xs0 = T0S_DIR ? s_t0[spc[s]][lane] : X0_2[s * 32 + lane];
        const float2 xs1 = X1_2[s * 32 + lane];

        auto do_m2 = [&](int m, float yv, uint2 g01, float rx0, float ry0, float rx1, float ry1, float sgn2,
                         float& q0x, float& q0y, float& q1x, float& q1y) {
            const uint2 gn01 = s_gn[warp][m][lane];
            const float2 g0n = ah::unpack2<H>(gn01.x), g0s = ah::unpack2<H>(g01.x);
            const float w0x = g0n.x * xs0.x + sgn2 * g0s.x * xn0.x;
            const float w0y = g0n.y * xs0.y + sgn2 * g0s.y * xn0.y;
            const float2 g1n = ah::unpack2<H>(gn01.y), g1s = ah::unpack2<H>(g01.y);
            const float w1x = g1n.x * xs1.x + sgn2 * g1s.x * xn1.x;
            const float w1y = g1n.y * xs1.y + sgn2 * g1s.y * xn1.y;
            gp[m] = (w0x * rx0 + w0y * ry0) + (w1x * rx1 + w1y * ry1);
            q0x += yv * w0x; q0y += yv * w0y;
            q1x += yv * w1x; q1y += yv * w1y;
        };
        {
            const float4 vy0 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
            const uint2 v0 = G01[sbase + 0L * 32], v1 = G01[sbase + 1L * 32];
            const uint2 v2 = G01[sbase + 2L * 32], v3 = G01[sbase + 3L * 32];
            float rx0, ry0, tx0, ty0, rx1, ry1, tx1, ty1;
            eval_c4_pair_rt(Ck0_4[cbase + 0 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
            eval_c4_pair_rt(Ck1_4[cbase + 0 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
            float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
            do_m2(0, vy0.x, v0, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
            sp += (q0x * tx0 + q0y * ty0) + (q1x * tx1 + q1y * ty1);

            q0x = q0y = q1x = q1y = 0.f;
            eval_c4_pair_rt(Ck0_4[cbase + 1 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
            eval_c4_pair_rt(Ck1_4[cbase + 1 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
            do_m2(1, vy0.y, v1, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
            do_m2(2, vy0.z, v2, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
            do_m2(3, vy0.w, v3, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
            sp += (q0x * tx0 + q0y * ty0) + (q1x * tx1 + q1y * ty1);
        }
        {
            const bool do0_2 = !SHELL || (kidx < k0_2);
            const bool do1_2 = !SHELL || (kidx < k1_2);
            const bool do0_3 = !SHELL || (kidx < k0_3);
            const bool do1_3 = !SHELL || (kidx < k1_3);
            if (!do0_2 && !do1_2 && !do0_3 && !do1_3) {
#pragma unroll
                for (int m = 4; m < 16; ++m) gp[m] = 0.f;
            } else {
                const float4 vy2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
                if (do0_2 || do1_2) {
                    float rx0 = 0.f, ry0 = 0.f, tx0 = 0.f, ty0 = 0.f;
                    float rx1 = 0.f, ry1 = 0.f, tx1 = 0.f, ty1 = 0.f;
                    if (do0_2) eval_c4_pair_rt(Ck0_4[cbase + 2 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                    if (do1_2) eval_c4_pair_rt(Ck1_4[cbase + 2 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    {
                        const float4 vy1 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                        const uint2 v4 = G01[sbase + 4L * 32], v5 = G01[sbase + 5L * 32];
                        const uint2 v6 = G01[sbase + 6L * 32], v7 = G01[sbase + 7L * 32];
                        do_m2(4, vy1.x, v4, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                        do_m2(5, vy1.y, v5, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                        do_m2(6, vy1.z, v6, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                        do_m2(7, vy1.w, v7, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                    }
                    const uint2 v8 = G01[sbase + 8L * 32];
                    do_m2(8, vy2.x, v8, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                    if (do0_2) sp += q0x * tx0 + q0y * ty0;
                    if (do1_2) sp += q1x * tx1 + q1y * ty1;
                } else {
                    gp[4] = gp[5] = gp[6] = gp[7] = gp[8] = 0.f;
                }
                if (do0_3 || do1_3) {
                    float rx0 = 0.f, ry0 = 0.f, tx0 = 0.f, ty0 = 0.f;
                    float rx1 = 0.f, ry1 = 0.f, tx1 = 0.f, ty1 = 0.f;
                    if (do0_3) eval_c4_pair_rt(Ck0_4[cbase + 3 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                    if (do1_3) eval_c4_pair_rt(Ck1_4[cbase + 3 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                    float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                    const uint2 v9 = G01[sbase + 9L * 32], vA = G01[sbase + 10L * 32], vB = G01[sbase + 11L * 32];
                    do_m2(9, vy2.y, v9, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                    do_m2(10, vy2.z, vA, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                    do_m2(11, vy2.w, vB, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                    {
                        const float4 vy3 = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                        const uint2 vC = G01[sbase + 12L * 32], vD = G01[sbase + 13L * 32];
                        const uint2 vE = G01[sbase + 14L * 32], vF = G01[sbase + 15L * 32];
                        do_m2(12, vy3.x, vC, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                        do_m2(13, vy3.y, vD, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                        do_m2(14, vy3.z, vE, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                        do_m2(15, vy3.w, vF, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                    }
                    if (do0_3) sp += q0x * tx0 + q0y * ty0;
                    if (do1_3) sp += q1x * tx1 + q1y * ty1;
                } else {
                    gp[9] = gp[10] = gp[11] = gp[12] = gp[13] = gp[14] = gp[15] = 0.f;
                }
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        float a8[8];
        const bool up = lane & 16;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            a8[j] = (up ? gp[j + 8] : gp[j]) + __shfl_xor_sync(0xffffffffu, up ? gp[j] : gp[j + 8], 16);
        float a4[4];
        const bool b8 = lane & 8;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a4[j] = (b8 ? a8[j + 4] : a8[j]) + __shfl_xor_sync(0xffffffffu, b8 ? a8[j] : a8[j + 4], 8);
        float a2[2];
        const bool b4 = lane & 4;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            a2[j] = (b4 ? a4[j + 2] : a4[j]) + __shfl_xor_sync(0xffffffffu, b4 ? a4[j] : a4[j + 2], 4);
        const bool b2 = lane & 2;
        float a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 2);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        int src = 0;
        if (lane < 16) {
            const int m = lane;
            src = ((m & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (((m >> 3) & 1) << 4);
        }
        const float gm = __shfl_sync(0xffffffffu, a1, src);
        if (lane < 16) sgp[warp][lane] = gm;
        __syncwarp();
        float g0 = 0.f, g1 = 0.f, g2 = 0.f;
        if (lane == 0) {
            const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
            float jx, jy_, jz;
            sh16_jyt<float>(up4.x, up4.y, up4.z, up4.w, 1.f, YP + 16L * p, sgp[warp], jx, jy_, jz);
            g0 = jx + sp * up4.x; g1 = jy_ + sp * up4.y; g2 = jz + sp * up4.z;
            sv[warp][9] -= g0; sv[warp][10] -= g1; sv[warp][11] -= g2;
            const float r_len = __frcp_rn(up4.w);
            const float vx = r_len * up4.x, vy_ = r_len * up4.y, vz = r_len * up4.z;
            sv[warp][0] += vx * g0; sv[warp][1] += vx * g1; sv[warp][2] += vx * g2;
            sv[warp][3] += vy_ * g0; sv[warp][4] += vy_ * g1; sv[warp][5] += vy_ * g2;
            sv[warp][6] += vz * g0; sv[warp][7] += vz * g1; sv[warp][8] += vz * g2;
        }
        g0 = __shfl_sync(0xffffffffu, g0, 0);
        g1 = __shfl_sync(0xffffffffu, g1, 0);
        g2 = __shfl_sync(0xffffffffu, g2, 0);
        if (lane < 3) atomicAdd(F + 3L * s + lane, lane == 0 ? g0 : (lane == 1 ? g1 : g2));
    }
    if (any_pair) {
        __syncwarp();
        if (lane < 3) atomicAdd(F + 3L * n + lane, sv[warp][9 + lane]);
        if (lane < 9) atomicAdd(mv + 9L * ngraph[n] + lane, sv[warp][lane]);
    }
}

// ============================================================================ 4e. k_force_pair_1l (Wave 25: LST_FORCE_OPT=2!)
// PAIR-FOLDED split single-layer force+virial (+ ZERO-COST fused conv_vjp when DO_VJP=true for Layer 1!):
//   - Loops over PAIRS P = E/2 (rptr0 + emeta) -> HALF the edges/sh16_jyt/atomics vs Wave 24 k_force_dir_1l!
//   - 8 KB s_ck[4][4][32] caches Ck across prev_kidx (-75% Ck L2 reads!), <=80 regs -> 6 blocks/SM = 50% occ!
//   - DO_VJP=true (Layer 1): factors qn_l = sum(yv*gn), qs_l = sum(yv*gs) -> DELETES conv_vjp (36-65 us) with ZERO extra loads!
//   - T0S_DIR=true (Layer 0): reads X0 from 1 KB s_t0[4][32] (ZERO X0 loads!), ZERO b.vec loads, ZERO ZIP_G01!
template <typename H, bool DO_VJP, bool SHELL = false, bool T0S_DIR = false>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
__global__ void __launch_bounds__(128, DO_VJP ? 5 : 6) k_force_pair_1l(
#else
__global__ void __launch_bounds__(128, DO_VJP ? 4 : 5) k_force_pair_1l(
#endif
    const int* __restrict__ rptr, const int* __restrict__ rptr0, const int4* __restrict__ emeta,
    const int* __restrict__ ngraph, const float* __restrict__ YP, const float* __restrict__ UP,
    const H* __restrict__ G, const float* __restrict__ C_kh, const float* __restrict__ X,
    float* __restrict__ F, float* __restrict__ mv, float* __restrict__ OX,
    float sp_invh, int N, int fes, const int* __restrict__ norder,
    int kcut2 = 999999, int kcut3 = 999999,
    const int* __restrict__ spc = nullptr, const float* __restrict__ T0 = nullptr, int S = 0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wid = blockIdx.x * 4 + warp;
    const int nn = wid / fes, wsub = wid % fes;

    __shared__ uint4 s_ck[4][4][32];                                   // 8 KB
    __shared__ float2 s_t0[T0S_DIR ? 4 : 1][T0S_DIR ? 32 : 1];         // 1 KB when T0S_DIR
    __shared__ float sv[4][12];
    __shared__ float sgp[4][16];
    if constexpr (T0S_DIR) {
        if (warp == 0) {
            const float2* T02 = reinterpret_cast<const float2*>(T0);
            for (int sp = 0; sp < S; ++sp) s_t0[sp][lane] = T02[sp * 32 + lane];
        }
        __syncthreads();
    }
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;

    const int e0_n = rptr[n];
    const int e1_n = rptr0[n];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = (fes > 1) ? (e0_n + (cnt_n * wsub) / fes) : e0_n;
    const int e_end = (fes > 1) ? (e0_n + (cnt_n * (wsub + 1)) / fes) : e1_n;
    if (e_beg >= e_end) return;

    uint32_t gn[16];
    const long nbase = (long)n * 16 * 32 + lane;
#pragma unroll
    for (int m = 0; m < 16; ++m) gn[m] = ah::ld_h2<H>(G, nbase + (long)m * 32);

    const float2* X2 = reinterpret_cast<const float2*>(X);
    const uint4* Ck4 = reinterpret_cast<const uint4*>(C_kh);
    const float2 xn = T0S_DIR ? s_t0[spc[n]][lane] : X2[(long)n * 32 + lane];

    if (lane < 12) sv[warp][lane] = 0.f;
    bool any_pair = false;
    float ox_nx = 0.f, ox_ny = 0.f;
    int prev_kidx = -1;

    for (int e = e_beg; e < e_end; ++e) {
        const int4 em = emeta[e];
        const int ps_e = em.x;
        if (ps_e & 1) continue;
        const int kidx = em.z;
        if (kidx < 0) continue;
        const long p = ps_e >> 1;
        const long s = em.y;
        const float u = __int_as_float(em.w);
        any_pair = true;
        const bool do_l2 = !SHELL || (kidx < kcut2);
        const bool do_l3 = !SHELL || (kidx < kcut3);

        if (kidx != prev_kidx) {
            const long cbase = (long)kidx * 4 * 32 + lane;
            s_ck[warp][0][lane] = Ck4[cbase + 0 * 32];
            s_ck[warp][1][lane] = Ck4[cbase + 1 * 32];
            if (do_l2) s_ck[warp][2][lane] = Ck4[cbase + 2 * 32];
            if (do_l3) s_ck[warp][3][lane] = Ck4[cbase + 3 * 32];
            prev_kidx = kidx;
        }
        const float2 xs = T0S_DIR ? s_t0[spc[s]][lane] : X2[s * 32 + lane];
        const long sbase = s * 16 * 32 + lane;
        float gp[16];
        float sp = 0.f;
        float ox_sx = 0.f, ox_sy = 0.f;

        auto do_mp = [&](int m, float yv, uint32_t gs_w, float rx, float ry, float sgn2,
                         float& qnx, float& qny, float& qsx, float& qsy) {
            const float2 gn2 = ah::unpack2<H>(gn[m]);
            const float2 gs2 = ah::unpack2<H>(gs_w);
            const float wx = gn2.x * xs.x + sgn2 * gs2.x * xn.x;
            const float wy = gn2.y * xs.y + sgn2 * gs2.y * xn.y;
            gp[m] = wx * rx + wy * ry;
            if constexpr (DO_VJP) {
                qnx += yv * gn2.x; qny += yv * gn2.y;
                qsx += yv * gs2.x; qsy += yv * gs2.y;
            } else {
                qnx += yv * wx; qny += yv * wy;
            }
        };
        auto finish_dl = [&](float rx, float ry, float tx, float ty, float sgn2,
                             float qnx, float qny, float qsx, float qsy) {
            if constexpr (DO_VJP) {
                sp += (qnx * xs.x + sgn2 * qsx * xn.x) * tx + (qny * xs.y + sgn2 * qsy * xn.y) * ty;
                ox_sx += rx * qnx; ox_sy += ry * qny;
                ox_nx += sgn2 * rx * qsx; ox_ny += sgn2 * ry * qsy;
            } else {
                sp += qnx * tx + qny * ty;
            }
        };
        {
            const float4 vy0 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
            const uint32_t g0 = ah::ld_h2<H>(G, sbase + 0L * 32), g1 = ah::ld_h2<H>(G, sbase + 1L * 32);
            const uint32_t g2 = ah::ld_h2<H>(G, sbase + 2L * 32), g3 = ah::ld_h2<H>(G, sbase + 3L * 32);
            float rx, ry, tx, ty;
            eval_c4_pair_rt(s_ck[warp][0][lane], u, sp_invh, rx, ry, tx, ty);
            float qnx = 0.f, qny = 0.f, qsx = 0.f, qsy = 0.f;
            do_mp(0, vy0.x, g0, rx, ry, 1.f, qnx, qny, qsx, qsy);
            finish_dl(rx, ry, tx, ty, 1.f, qnx, qny, qsx, qsy);

            qnx = qny = qsx = qsy = 0.f;
            eval_c4_pair_rt(s_ck[warp][1][lane], u, sp_invh, rx, ry, tx, ty);
            do_mp(1, vy0.y, g1, rx, ry, -1.f, qnx, qny, qsx, qsy);
            do_mp(2, vy0.z, g2, rx, ry, -1.f, qnx, qny, qsx, qsy);
            do_mp(3, vy0.w, g3, rx, ry, -1.f, qnx, qny, qsx, qsy);
            finish_dl(rx, ry, tx, ty, -1.f, qnx, qny, qsx, qsy);
        }
        if (!do_l2 && !do_l3) {
#pragma unroll
            for (int m = 4; m < 16; ++m) gp[m] = 0.f;
        } else {
            const float4 vy2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
            if (do_l2) {
                float rx, ry, tx, ty;
                eval_c4_pair_rt(s_ck[warp][2][lane], u, sp_invh, rx, ry, tx, ty);
                float qnx = 0.f, qny = 0.f, qsx = 0.f, qsy = 0.f;
                {
                    const float4 vy1 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                    const uint32_t g4 = ah::ld_h2<H>(G, sbase + 4L * 32), g5 = ah::ld_h2<H>(G, sbase + 5L * 32);
                    const uint32_t g6 = ah::ld_h2<H>(G, sbase + 6L * 32), g7 = ah::ld_h2<H>(G, sbase + 7L * 32);
                    do_mp(4, vy1.x, g4, rx, ry, 1.f, qnx, qny, qsx, qsy);
                    do_mp(5, vy1.y, g5, rx, ry, 1.f, qnx, qny, qsx, qsy);
                    do_mp(6, vy1.z, g6, rx, ry, 1.f, qnx, qny, qsx, qsy);
                    do_mp(7, vy1.w, g7, rx, ry, 1.f, qnx, qny, qsx, qsy);
                }
                const uint32_t g8 = ah::ld_h2<H>(G, sbase + 8L * 32);
                do_mp(8, vy2.x, g8, rx, ry, 1.f, qnx, qny, qsx, qsy);
                finish_dl(rx, ry, tx, ty, 1.f, qnx, qny, qsx, qsy);
            } else {
                gp[4] = gp[5] = gp[6] = gp[7] = gp[8] = 0.f;
            }
            if (do_l3) {
                float rx, ry, tx, ty;
                eval_c4_pair_rt(s_ck[warp][3][lane], u, sp_invh, rx, ry, tx, ty);
                float qnx = 0.f, qny = 0.f, qsx = 0.f, qsy = 0.f;
                const uint32_t g9 = ah::ld_h2<H>(G, sbase + 9L * 32), gA = ah::ld_h2<H>(G, sbase + 10L * 32);
                const uint32_t gB = ah::ld_h2<H>(G, sbase + 11L * 32);
                do_mp(9, vy2.y, g9, rx, ry, -1.f, qnx, qny, qsx, qsy);
                do_mp(10, vy2.z, gA, rx, ry, -1.f, qnx, qny, qsx, qsy);
                do_mp(11, vy2.w, gB, rx, ry, -1.f, qnx, qny, qsx, qsy);
                {
                    const float4 vy3 = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                    const uint32_t gC = ah::ld_h2<H>(G, sbase + 12L * 32), gD = ah::ld_h2<H>(G, sbase + 13L * 32);
                    const uint32_t gE = ah::ld_h2<H>(G, sbase + 14L * 32), gF = ah::ld_h2<H>(G, sbase + 15L * 32);
                    do_mp(12, vy3.x, gC, rx, ry, -1.f, qnx, qny, qsx, qsy);
                    do_mp(13, vy3.y, gD, rx, ry, -1.f, qnx, qny, qsx, qsy);
                    do_mp(14, vy3.z, gE, rx, ry, -1.f, qnx, qny, qsx, qsy);
                    do_mp(15, vy3.w, gF, rx, ry, -1.f, qnx, qny, qsx, qsy);
                }
                finish_dl(rx, ry, tx, ty, -1.f, qnx, qny, qsx, qsy);
            } else {
                gp[9] = gp[10] = gp[11] = gp[12] = gp[13] = gp[14] = gp[15] = 0.f;
            }
        }
        if constexpr (DO_VJP) {
            atomicAdd(OX + s * 64 + 2 * lane, ox_sx);
            atomicAdd(OX + s * 64 + 2 * lane + 1, ox_sy);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        float a8[8];
        const bool up = lane & 16;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            a8[j] = (up ? gp[j + 8] : gp[j]) + __shfl_xor_sync(0xffffffffu, up ? gp[j] : gp[j + 8], 16);
        float a4[4];
        const bool b8 = lane & 8;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a4[j] = (b8 ? a8[j + 4] : a8[j]) + __shfl_xor_sync(0xffffffffu, b8 ? a8[j] : a8[j + 4], 8);
        float a2[2];
        const bool b4 = lane & 4;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            a2[j] = (b4 ? a4[j + 2] : a4[j]) + __shfl_xor_sync(0xffffffffu, b4 ? a4[j] : a4[j + 2], 4);
        const bool b2 = lane & 2;
        float a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 2);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        int src = 0;
        if (lane < 16) {
            const int m = lane;
            src = ((m & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (((m >> 3) & 1) << 4);
        }
        const float gm = __shfl_sync(0xffffffffu, a1, src);
        if (lane < 16) sgp[warp][lane] = gm;
        __syncwarp();
        float g0 = 0.f, g1 = 0.f, g2 = 0.f;
        if (lane == 0) {
            const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
            float jx, jy_, jz;
            sh16_jyt<float>(up4.x, up4.y, up4.z, up4.w, 1.f, YP + 16L * p, sgp[warp], jx, jy_, jz);
            g0 = jx + sp * up4.x; g1 = jy_ + sp * up4.y; g2 = jz + sp * up4.z;
            sv[warp][9] -= g0; sv[warp][10] -= g1; sv[warp][11] -= g2;
            const float r_len = __frcp_rn(up4.w);
            const float vx = r_len * up4.x, vy_ = r_len * up4.y, vz = r_len * up4.z;
            sv[warp][0] += vx * g0; sv[warp][1] += vx * g1; sv[warp][2] += vx * g2;
            sv[warp][3] += vy_ * g0; sv[warp][4] += vy_ * g1; sv[warp][5] += vy_ * g2;
            sv[warp][6] += vz * g0; sv[warp][7] += vz * g1; sv[warp][8] += vz * g2;
        }
        g0 = __shfl_sync(0xffffffffu, g0, 0);
        g1 = __shfl_sync(0xffffffffu, g1, 0);
        g2 = __shfl_sync(0xffffffffu, g2, 0);
        if (lane < 3) atomicAdd(F + 3L * s + lane, lane == 0 ? g0 : (lane == 1 ? g1 : g2));
    }
    if (any_pair) {
        __syncwarp();
        if (lane < 3) atomicAdd(F + 3L * n + lane, sv[warp][9 + lane]);
        if (lane < 9) atomicAdd(mv + 9L * ngraph[n] + lane, sv[warp][lane]);
        if constexpr (DO_VJP) {
            atomicAdd(OX + (long)n * 64 + 2 * lane, ox_nx);
            atomicAdd(OX + (long)n * 64 + 2 * lane + 1, ox_ny);
        }
    }
}

// ============================================================================ 4f. k_force_pair_h2_w29 (Wave 29: LST_FORCE_OPT=3..6!)
// Zero-s_gn (full 128 KB L1 for Ck0+Ck1!), streaming 4-way gp warp reduction (-18 regs: eliminates gp[16] & a8[8]!),
// T0 smem cache (0 X0 loads), zero b.vec loads (from UP), optional __ldcg on G01[s] (L2-cache/L1-bypass so G01[s]
// cannot evict 64 KB Ck0+Ck1 from L1!), and optional native __half2 HFMA2 (0 H2F2 unpacks!).
template <typename H, bool USE_HFMA2, bool LDCG, bool SHELL = false, bool T0S_DIR = false>
__global__ void __launch_bounds__(128, 5) k_force_pair_h2_w29(
    const int* __restrict__ rptr, const int* __restrict__ rptr0,
    const int* __restrict__ ngraph, const float* __restrict__ YP, const float* __restrict__ UP,
    const float* __restrict__ C_kh0, const float* __restrict__ X0,
    const float* __restrict__ C_kh1, const float* __restrict__ X1,
    float* __restrict__ F, float* __restrict__ mv, float sp_invh, int N, int fes,
    const int* __restrict__ norder, const int4* __restrict__ emeta, const uint2* __restrict__ G01,
    int k0_2 = 999999, int k0_3 = 999999, int k1_2 = 999999, int k1_3 = 999999,
    const int* __restrict__ spc = nullptr, const float* __restrict__ T0 = nullptr, int S = 0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wid = blockIdx.x * 4 + warp;
    const int nn = wid / fes, wsub = wid % fes;

    __shared__ float2 s_t0[T0S_DIR ? 4 : 1][T0S_DIR ? 32 : 1];  // 1 KB when T0S_DIR (full 127 KB L1 left!)
    __shared__ float sv[4][12];
    __shared__ float sgp[4][16];
    if constexpr (T0S_DIR) {
        if (warp == 0) {
            const float2* T02 = reinterpret_cast<const float2*>(T0);
            for (int sp = 0; sp < S; ++sp) s_t0[sp][lane] = T02[sp * 32 + lane];
        }
        __syncthreads();
    }
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;

    const int e0_n = rptr[n];
    const int e1_n = rptr0[n];
    const int cnt_n = e1_n - e0_n;
    const int e_beg = (fes > 1) ? (e0_n + (cnt_n * wsub) / fes) : e0_n;
    const int e_end = (fes > 1) ? (e0_n + (cnt_n * (wsub + 1)) / fes) : e1_n;
    if (e_beg >= e_end) return;

    uint32_t gn0[16], gn1[16];
    const long nbase = (long)n * 16 * 32 + lane;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const uint2 v = G01[nbase + (long)m * 32];
        gn0[m] = v.x;
        gn1[m] = v.y;
    }

    const float2* X0_2 = reinterpret_cast<const float2*>(X0);
    const float2* X1_2 = reinterpret_cast<const float2*>(X1);
    const uint4* Ck0_4 = reinterpret_cast<const uint4*>(C_kh0);
    const uint4* Ck1_4 = reinterpret_cast<const uint4*>(C_kh1);
    const float2 xn0 = T0S_DIR ? s_t0[spc[n]][lane] : X0_2[(long)n * 32 + lane];
    const float2 xn1 = X1_2[(long)n * 32 + lane];

    if (lane < 12) sv[warp][lane] = 0.f;
    bool any_pair = false;

    auto ld_g01 = [&](long idx) -> uint2 {
        if constexpr (LDCG) {
            uint2 v;
            asm("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(G01 + idx));
            return v;
        } else {
            return G01[idx];
        }
    };
    auto reduce4_gp = [&](int base_m, float g0, float g1, float g2, float g3) {
        const bool up = (lane & 16) != 0;
        const float a0 = (up ? g2 : g0) + __shfl_xor_sync(0xffffffffu, up ? g0 : g2, 16);
        const float a1 = (up ? g3 : g1) + __shfl_xor_sync(0xffffffffu, up ? g1 : g3, 16);
        const bool b8 = (lane & 8) != 0;
        float v = (b8 ? a1 : a0) + __shfl_xor_sync(0xffffffffu, b8 ? a0 : a1, 8);
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        if ((lane & 7) == 0) sgp[warp][base_m + (lane >> 3)] = v;
    };

    for (int e = e_beg; e < e_end; ++e) {
        const int4 em = emeta[e];
        const int ps_e = em.x;
        if (ps_e & 1) continue;
        const int kidx = em.z;
        if (kidx < 0) continue;
        const long p = ps_e >> 1;
        const long s = em.y;
        const float u = __int_as_float(em.w);
        any_pair = true;

        float sp = 0.f;
        const long sbase = s * 16 * 32 + lane;
        const long cbase = (long)kidx * 4 * 32 + lane;
        const float2 xs0 = T0S_DIR ? s_t0[spc[s]][lane] : X0_2[s * 32 + lane];
        const float2 xs1 = X1_2[s * 32 + lane];

        if constexpr (USE_HFMA2 && std::is_same<H, __half>::value) {
            const __half2 xs0_h = __float22half2_rn(xs0), xs1_h = __float22half2_rn(xs1);
            const __half2 xn0_p = __float22half2_rn(xn0), xn0_m = __hneg2(xn0_p);
            const __half2 xn1_p = __float22half2_rn(xn1), xn1_m = __hneg2(xn1_p);
            const __half2 z_h = __float2half2_rn(0.f);
            auto do_m2h = [&](int m, float yv, uint2 g01, __half2 r0_h, __half2 r1_h,
                              __half2 xn0_s, __half2 xn1_s, __half2& q0_h, __half2& q1_h) -> float {
                const __half2 g0n = *reinterpret_cast<const __half2*>(&gn0[m]);
                const __half2 g0s = *reinterpret_cast<const __half2*>(&g01.x);
                const __half2 w0 = __hfma2(g0n, xs0_h, __hmul2(g0s, xn0_s));
                const __half2 g1n = *reinterpret_cast<const __half2*>(&gn1[m]);
                const __half2 g1s = *reinterpret_cast<const __half2*>(&g01.y);
                const __half2 w1 = __hfma2(g1n, xs1_h, __hmul2(g1s, xn1_s));
                const float2 gpf = __half22float2(__hfma2(w0, r0_h, __hmul2(w1, r1_h)));
                const __half2 yh = __float2half2_rn(yv);
                q0_h = __hfma2(yh, w0, q0_h);
                q1_h = __hfma2(yh, w1, q1_h);
                return gpf.x + gpf.y;
            };
            {
                const float4 vy0 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
                const uint2 v0 = ld_g01(sbase + 0L * 32), v1 = ld_g01(sbase + 1L * 32);
                const uint2 v2 = ld_g01(sbase + 2L * 32), v3 = ld_g01(sbase + 3L * 32);
                float rx0, ry0, tx0, ty0, rx1, ry1, tx1, ty1;
                eval_c4_pair_rt(Ck0_4[cbase + 0 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                eval_c4_pair_rt(Ck1_4[cbase + 0 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                __half2 q0_h = z_h, q1_h = z_h;
                const float g0 = do_m2h(0, vy0.x, v0, __floats2half2_rn(rx0, ry0), __floats2half2_rn(rx1, ry1), xn0_p, xn1_p, q0_h, q1_h);
                { const float2 f0 = __half22float2(q0_h), f1 = __half22float2(q1_h); sp += (f0.x * tx0 + f0.y * ty0) + (f1.x * tx1 + f1.y * ty1); }

                q0_h = z_h; q1_h = z_h;
                eval_c4_pair_rt(Ck0_4[cbase + 1 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                eval_c4_pair_rt(Ck1_4[cbase + 1 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                const __half2 r0_1 = __floats2half2_rn(rx0, ry0), r1_1 = __floats2half2_rn(rx1, ry1);
                const float g1 = do_m2h(1, vy0.y, v1, r0_1, r1_1, xn0_m, xn1_m, q0_h, q1_h);
                const float g2 = do_m2h(2, vy0.z, v2, r0_1, r1_1, xn0_m, xn1_m, q0_h, q1_h);
                const float g3 = do_m2h(3, vy0.w, v3, r0_1, r1_1, xn0_m, xn1_m, q0_h, q1_h);
                { const float2 f0 = __half22float2(q0_h), f1 = __half22float2(q1_h); sp += (f0.x * tx0 + f0.y * ty0) + (f1.x * tx1 + f1.y * ty1); }
                reduce4_gp(0, g0, g1, g2, g3);
            }
            {
                const bool do0_2 = !SHELL || (kidx < k0_2);
                const bool do1_2 = !SHELL || (kidx < k1_2);
                const bool do0_3 = !SHELL || (kidx < k0_3);
                const bool do1_3 = !SHELL || (kidx < k1_3);
                if (!do0_2 && !do1_2 && !do0_3 && !do1_3) {
                    if (lane >= 4 && lane < 16) sgp[warp][lane] = 0.f;
                } else {
                    float g8 = 0.f;
                    const float4 vy2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
                    if (do0_2 || do1_2) {
                        float rx0 = 0.f, ry0 = 0.f, tx0 = 0.f, ty0 = 0.f;
                        float rx1 = 0.f, ry1 = 0.f, tx1 = 0.f, ty1 = 0.f;
                        if (do0_2) eval_c4_pair_rt(Ck0_4[cbase + 2 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                        if (do1_2) eval_c4_pair_rt(Ck1_4[cbase + 2 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                        const __half2 r0_2 = __floats2half2_rn(rx0, ry0), r1_2 = __floats2half2_rn(rx1, ry1);
                        __half2 q0_h = z_h, q1_h = z_h;
                        {
                            const float4 vy1 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                            const uint2 v4 = ld_g01(sbase + 4L * 32), v5 = ld_g01(sbase + 5L * 32);
                            const uint2 v6 = ld_g01(sbase + 6L * 32), v7 = ld_g01(sbase + 7L * 32);
                            const float g4 = do_m2h(4, vy1.x, v4, r0_2, r1_2, xn0_p, xn1_p, q0_h, q1_h);
                            const float g5 = do_m2h(5, vy1.y, v5, r0_2, r1_2, xn0_p, xn1_p, q0_h, q1_h);
                            const float g6 = do_m2h(6, vy1.z, v6, r0_2, r1_2, xn0_p, xn1_p, q0_h, q1_h);
                            const float g7 = do_m2h(7, vy1.w, v7, r0_2, r1_2, xn0_p, xn1_p, q0_h, q1_h);
                            reduce4_gp(4, g4, g5, g6, g7);
                        }
                        const uint2 v8 = ld_g01(sbase + 8L * 32);
                        g8 = do_m2h(8, vy2.x, v8, r0_2, r1_2, xn0_p, xn1_p, q0_h, q1_h);
                        { const float2 f0 = __half22float2(q0_h), f1 = __half22float2(q1_h); sp += (f0.x * tx0 + f0.y * ty0) + (f1.x * tx1 + f1.y * ty1); }
                    } else {
                        if (lane < 4) sgp[warp][4 + lane] = 0.f;
                    }
                    if (do0_3 || do1_3) {
                        float rx0 = 0.f, ry0 = 0.f, tx0 = 0.f, ty0 = 0.f;
                        float rx1 = 0.f, ry1 = 0.f, tx1 = 0.f, ty1 = 0.f;
                        if (do0_3) eval_c4_pair_rt(Ck0_4[cbase + 3 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                        if (do1_3) eval_c4_pair_rt(Ck1_4[cbase + 3 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                        const __half2 r0_3 = __floats2half2_rn(rx0, ry0), r1_3 = __floats2half2_rn(rx1, ry1);
                        __half2 q0_h = z_h, q1_h = z_h;
                        {
                            const uint2 v9 = ld_g01(sbase + 9L * 32), vA = ld_g01(sbase + 10L * 32), vB = ld_g01(sbase + 11L * 32);
                            const float g9 = do_m2h(9,  vy2.y, v9, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                            const float gA = do_m2h(10, vy2.z, vA, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                            const float gB = do_m2h(11, vy2.w, vB, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                            reduce4_gp(8, g8, g9, gA, gB);
                        }
                        {
                            const float4 vy3 = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                            const uint2 vC = ld_g01(sbase + 12L * 32), vD = ld_g01(sbase + 13L * 32);
                            const uint2 vE = ld_g01(sbase + 14L * 32), vF = ld_g01(sbase + 15L * 32);
                            const float gC = do_m2h(12, vy3.x, vC, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                            const float gD = do_m2h(13, vy3.y, vD, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                            const float gE = do_m2h(14, vy3.z, vE, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                            const float gF = do_m2h(15, vy3.w, vF, r0_3, r1_3, xn0_m, xn1_m, q0_h, q1_h);
                            reduce4_gp(12, gC, gD, gE, gF);
                        }
                        { const float2 f0 = __half22float2(q0_h), f1 = __half22float2(q1_h); sp += (f0.x * tx0 + f0.y * ty0) + (f1.x * tx1 + f1.y * ty1); }
                    } else {
                        reduce4_gp(8, g8, 0.f, 0.f, 0.f);
                        if (lane < 4) sgp[warp][12 + lane] = 0.f;
                    }
                }
            }
        } else {
            auto do_m2 = [&](int m, float yv, uint2 g01, float rx0, float ry0, float rx1, float ry1, float sgn2,
                             float& q0x, float& q0y, float& q1x, float& q1y) -> float {
                const float2 g0n = ah::unpack2<H>(gn0[m]), g0s = ah::unpack2<H>(g01.x);
                const float w0x = g0n.x * xs0.x + sgn2 * g0s.x * xn0.x;
                const float w0y = g0n.y * xs0.y + sgn2 * g0s.y * xn0.y;
                const float2 g1n = ah::unpack2<H>(gn1[m]), g1s = ah::unpack2<H>(g01.y);
                const float w1x = g1n.x * xs1.x + sgn2 * g1s.x * xn1.x;
                const float w1y = g1n.y * xs1.y + sgn2 * g1s.y * xn1.y;
                q0x += yv * w0x; q0y += yv * w0y;
                q1x += yv * w1x; q1y += yv * w1y;
                return (w0x * rx0 + w0y * ry0) + (w1x * rx1 + w1y * ry1);
            };
            {
                const float4 vy0 = *reinterpret_cast<const float4*>(YP + 16L * p + 0);
                const uint2 v0 = ld_g01(sbase + 0L * 32), v1 = ld_g01(sbase + 1L * 32);
                const uint2 v2 = ld_g01(sbase + 2L * 32), v3 = ld_g01(sbase + 3L * 32);
                float rx0, ry0, tx0, ty0, rx1, ry1, tx1, ty1;
                eval_c4_pair_rt(Ck0_4[cbase + 0 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                eval_c4_pair_rt(Ck1_4[cbase + 0 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                const float g0 = do_m2(0, vy0.x, v0, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                sp += (q0x * tx0 + q0y * ty0) + (q1x * tx1 + q1y * ty1);

                q0x = q0y = q1x = q1y = 0.f;
                eval_c4_pair_rt(Ck0_4[cbase + 1 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                eval_c4_pair_rt(Ck1_4[cbase + 1 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                const float g1 = do_m2(1, vy0.y, v1, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                const float g2 = do_m2(2, vy0.z, v2, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                const float g3 = do_m2(3, vy0.w, v3, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                sp += (q0x * tx0 + q0y * ty0) + (q1x * tx1 + q1y * ty1);
                reduce4_gp(0, g0, g1, g2, g3);
            }
            {
                const bool do0_2 = !SHELL || (kidx < k0_2);
                const bool do1_2 = !SHELL || (kidx < k1_2);
                const bool do0_3 = !SHELL || (kidx < k0_3);
                const bool do1_3 = !SHELL || (kidx < k1_3);
                if (!do0_2 && !do1_2 && !do0_3 && !do1_3) {
                    if (lane >= 4 && lane < 16) sgp[warp][lane] = 0.f;
                } else {
                    float g8 = 0.f;
                    const float4 vy2 = *reinterpret_cast<const float4*>(YP + 16L * p + 8);
                    if (do0_2 || do1_2) {
                        float rx0 = 0.f, ry0 = 0.f, tx0 = 0.f, ty0 = 0.f;
                        float rx1 = 0.f, ry1 = 0.f, tx1 = 0.f, ty1 = 0.f;
                        if (do0_2) eval_c4_pair_rt(Ck0_4[cbase + 2 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                        if (do1_2) eval_c4_pair_rt(Ck1_4[cbase + 2 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                        float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                        {
                            const float4 vy1 = *reinterpret_cast<const float4*>(YP + 16L * p + 4);
                            const uint2 v4 = ld_g01(sbase + 4L * 32), v5 = ld_g01(sbase + 5L * 32);
                            const uint2 v6 = ld_g01(sbase + 6L * 32), v7 = ld_g01(sbase + 7L * 32);
                            const float g4 = do_m2(4, vy1.x, v4, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                            const float g5 = do_m2(5, vy1.y, v5, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                            const float g6 = do_m2(6, vy1.z, v6, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                            const float g7 = do_m2(7, vy1.w, v7, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                            reduce4_gp(4, g4, g5, g6, g7);
                        }
                        const uint2 v8 = ld_g01(sbase + 8L * 32);
                        g8 = do_m2(8, vy2.x, v8, rx0, ry0, rx1, ry1, 1.f, q0x, q0y, q1x, q1y);
                        if (do0_2) sp += q0x * tx0 + q0y * ty0;
                        if (do1_2) sp += q1x * tx1 + q1y * ty1;
                    } else {
                        if (lane < 4) sgp[warp][4 + lane] = 0.f;
                    }
                    if (do0_3 || do1_3) {
                        float rx0 = 0.f, ry0 = 0.f, tx0 = 0.f, ty0 = 0.f;
                        float rx1 = 0.f, ry1 = 0.f, tx1 = 0.f, ty1 = 0.f;
                        if (do0_3) eval_c4_pair_rt(Ck0_4[cbase + 3 * 32], u, sp_invh, rx0, ry0, tx0, ty0);
                        if (do1_3) eval_c4_pair_rt(Ck1_4[cbase + 3 * 32], u, sp_invh, rx1, ry1, tx1, ty1);
                        float q0x = 0.f, q0y = 0.f, q1x = 0.f, q1y = 0.f;
                        {
                            const uint2 v9 = ld_g01(sbase + 9L * 32), vA = ld_g01(sbase + 10L * 32), vB = ld_g01(sbase + 11L * 32);
                            const float g9 = do_m2(9,  vy2.y, v9, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                            const float gA = do_m2(10, vy2.z, vA, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                            const float gB = do_m2(11, vy2.w, vB, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                            reduce4_gp(8, g8, g9, gA, gB);
                        }
                        {
                            const float4 vy3 = *reinterpret_cast<const float4*>(YP + 16L * p + 12);
                            const uint2 vC = ld_g01(sbase + 12L * 32), vD = ld_g01(sbase + 13L * 32);
                            const uint2 vE = ld_g01(sbase + 14L * 32), vF = ld_g01(sbase + 15L * 32);
                            const float gC = do_m2(12, vy3.x, vC, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                            const float gD = do_m2(13, vy3.y, vD, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                            const float gE = do_m2(14, vy3.z, vE, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                            const float gF = do_m2(15, vy3.w, vF, rx0, ry0, rx1, ry1, -1.f, q0x, q0y, q1x, q1y);
                            reduce4_gp(12, gC, gD, gE, gF);
                        }
                        if (do0_3) sp += q0x * tx0 + q0y * ty0;
                        if (do1_3) sp += q1x * tx1 + q1y * ty1;
                    } else {
                        reduce4_gp(8, g8, 0.f, 0.f, 0.f);
                        if (lane < 4) sgp[warp][12 + lane] = 0.f;
                    }
                }
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        __syncwarp();
        float g0 = 0.f, g1 = 0.f, g2 = 0.f;
        if (lane == 0) {
            const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
            float jx, jy_, jz;
            sh16_jyt<float>(up4.x, up4.y, up4.z, up4.w, 1.f, YP + 16L * p, sgp[warp], jx, jy_, jz);
            g0 = jx + sp * up4.x; g1 = jy_ + sp * up4.y; g2 = jz + sp * up4.z;
            sv[warp][9] -= g0; sv[warp][10] -= g1; sv[warp][11] -= g2;
            const float r_len = __frcp_rn(up4.w);
            const float vx = r_len * up4.x, vy_ = r_len * up4.y, vz = r_len * up4.z;
            sv[warp][0] += vx * g0; sv[warp][1] += vx * g1; sv[warp][2] += vx * g2;
            sv[warp][3] += vy_ * g0; sv[warp][4] += vy_ * g1; sv[warp][5] += vy_ * g2;
            sv[warp][6] += vz * g0; sv[warp][7] += vz * g1; sv[warp][8] += vz * g2;
        }
        g0 = __shfl_sync(0xffffffffu, g0, 0);
        g1 = __shfl_sync(0xffffffffu, g1, 0);
        g2 = __shfl_sync(0xffffffffu, g2, 0);
        if (lane < 3) atomicAdd(F + 3L * s + lane, lane == 0 ? g0 : (lane == 1 ? g1 : g2));
    }
    if (any_pair) {
        __syncwarp();
        if (lane < 3) atomicAdd(F + 3L * n + lane, sv[warp][9 + lane]);
        if (lane < 9) atomicAdd(mv + 9L * ngraph[n] + lane, sv[warp][lane]);
    }
}

}  // namespace eh2
}  // namespace lst
