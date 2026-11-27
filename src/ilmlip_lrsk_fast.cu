// Ultra-fast v21 LR-SK-Fast CUDA kernels in dedicated translation unit (~8s incremental rebuild!).
//   - NB2 in {4, 8, 12} with dedicated 1-float4 NB2=4 fast path (zero loops, 2x smaller SMEM & FLOPs)
//   - Register-cached g_loc/dg_loc + FP32 force accumulation in flat MD (!STRESS) eliminating 1/64-rate FP64 stalls
//   - Single fused SMEM-resident Pol-MLP + Energy + ZBL kernel (s_W0[80][33] zero bank conflicts fwd & bwd!)
//   - Grid-stride SMEM-resident Heads Fwd/Bwd fused with last-layer SiLU + hgemm2 row exponent!
#include "ilmlip_lrsk_fast_api.hpp"
#include "model.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace loup {
namespace lrsk_fast {

namespace {

__device__ __forceinline__ float v1_dot4(float4 a, float4 b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w)));
}

__device__ __forceinline__ float q4_sum_f(float v, unsigned mask) {
    v += __shfl_xor_sync(mask, v, 2);
    v += __shfl_xor_sync(mask, v, 1);
    return v;
}

__device__ __forceinline__ float q4_max_f(float v, unsigned mask) {
    v = fmaxf(v, __shfl_xor_sync(mask, v, 2));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 1));
    return v;
}

__device__ __forceinline__ float half_warp_sum_f(float v, unsigned mask) {
    v += __shfl_xor_sync(mask, v, 8);
    v += __shfl_xor_sync(mask, v, 4);
    v += __shfl_xor_sync(mask, v, 2);
    v += __shfl_xor_sync(mask, v, 1);
    return v;
}

__device__ __forceinline__ double half_warp_sum_d(double v, unsigned mask) {
    v += __shfl_xor_sync(mask, v, 8);
    v += __shfl_xor_sync(mask, v, 4);
    v += __shfl_xor_sync(mask, v, 2);
    v += __shfl_xor_sync(mask, v, 1);
    return v;
}

__device__ __forceinline__ float warp_sum_f(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warp_max_f(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ double warp_sum_d(double v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ int8_t exp_from_am(float am) {
    int e = -100;
    if (am > 0.0f && isfinite(am)) {
        const uint32_t u = __float_as_uint(am);
        e = (int)((u >> 23) & 0xffu) - 127;
        if (e < -100) e = -100;
        if (e > 100) e = 100;
    }
    return (int8_t)e;
}

__device__ __forceinline__ int8_t exp_from_am_h2(float am) {
    if (am > 0.0f && isfinite(am)) {
        int e = (int)((__float_as_uint(am) >> 23) & 0xffu) - 127 + 1;
        if (e < -108) e = -108;
        if (e > 108) e = 108;
        return (int8_t)e;
    }
    return (int8_t)-127;
}

__device__ __forceinline__ float pow2_f(int e) {
    return __uint_as_float((uint32_t)(e + 127) << 23);
}

__device__ __forceinline__ void split_h2(float v0, float v1, uint32_t& outH, uint32_t& outL) {
    const __half h0 = __float2half_rn(v0);
    const __half h1 = __float2half_rn(v1);
    const __half l0 = __float2half_rn((v0 - __half2float(h0)) * 2048.0f);
    const __half l1 = __float2half_rn((v1 - __half2float(h1)) * 2048.0f);
    const __half2 H2 = __halves2half2(h0, h1);
    const __half2 L2 = __halves2half2(l0, l1);
    outH = *reinterpret_cast<const uint32_t*>(&H2);
    outL = *reinterpret_cast<const uint32_t*>(&L2);
}

__device__ __forceinline__ void mma16816_f32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void jam4(float (&d)[4]) {
    #pragma unroll
    for (int e = 0; e < 4; ++e) d[e] = __int_as_float(__float_as_int(d[e]) | 1);
}

__device__ __forceinline__ float reduce4_to_q(const float a[4], int lx, unsigned mask, bool nl16) {
    const bool p0 = (lx & 1) != 0;
    const float u0 = (p0 ? a[1] : a[0]) + __shfl_xor_sync(mask, p0 ? a[0] : a[1], 1);
    const float u1 = (p0 ? a[3] : a[2]) + __shfl_xor_sync(mask, p0 ? a[2] : a[3], 1);
    const bool p1 = (lx & 2) != 0;
    float res = (p1 ? u1 : u0) + __shfl_xor_sync(mask, p1 ? u0 : u1, 2);
    res += __shfl_xor_sync(mask, res, 4);
    res += __shfl_xor_sync(mask, res, 8);
    if (!nl16) res += __shfl_xor_sync(mask, res, 16);
    return res;
}

__device__ __forceinline__ int tri_idx(int a, int b, int c1) {
    const int lo = (a <= b) ? a : b;
    const int hi = (a <= b) ? b : a;
    return lo * c1 - (lo * (lo - 1)) / 2 + (hi - lo);
}

__device__ __forceinline__ float cheb_t(float r_f, ChebParamF cp) {
    return fmaf(r_f, cp.q0, r_f * cp.q1);
}

// Fast exact Chebyshev NB2 in {4, 8, 12} -> float4[NB2_4]
template <int NB, bool WITH_DERIV>
__device__ __forceinline__ void eval_cheb_store(
    float r_f, ChebParamF cp, float4* __restrict__ dst_g, float4* __restrict__ dst_dg) {
    constexpr int NB4 = (NB + 3) / 4;
    float t = cheb_t(r_f, cp);
    if (t < 0.0f) t = 0.0f;
    if (t >= 1.0f) {
        #pragma unroll
        for (int b = 0; b < NB4; ++b) {
            dst_g[b] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if constexpr (WITH_DERIV) dst_dg[b] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }
    const float inv_rmax = cp.q0;
    const float t2 = t * t;
    const float t4 = t2 * t2;
    const float t5 = t4 * t;
    const float env = 1.0f - 21.0f * t5 + 35.0f * t5 * t - 15.0f * t5 * t2;
    const float omt = 1.0f - t;
    const float denv_dt = WITH_DERIV ? (-105.0f * t4 * omt * omt) : 0.0f;
    float u = t;
    float dx_dt = 2.0f;
    if (cp.warp > 0.0f) {
        const float one_p_a = 1.0f + cp.warp;
        const float inv_d = 1.0f / fmaf(cp.warp, t, 1.0f);
        u = (one_p_a * t) * inv_d;
        if constexpr (WITH_DERIV) dx_dt = (2.0f * one_p_a) * (inv_d * inv_d);
    }
    const float env_dx = WITH_DERIV ? (dx_dt * env) : 0.0f;
    const float x = fmaf(2.0f, u, -1.0f);
    const float x2 = 2.0f * x;

    // First 4 basis functions (T0..T3) — always computed in registers!
    const float T0 = 1.0f;
    const float T1 = x;
    const float T2 = fmaf(x2, T1, -T0);
    const float T3 = fmaf(x2, T2, -T1);
    dst_g[0] = make_float4(env, T1 * env, T2 * env, T3 * env);

    float dT0 = 0.0f, dT1 = 1.0f, dT2 = 0.0f, dT3 = 0.0f;
    if constexpr (WITH_DERIV) {
        dT2 = fmaf(2.0f, T1, fmaf(x2, dT1, -dT0));
        dT3 = fmaf(2.0f, T2, fmaf(x2, dT2, -dT1));
        dst_dg[0] = make_float4(
            denv_dt * inv_rmax,
            fmaf(env_dx, dT1, T1 * denv_dt) * inv_rmax,
            fmaf(env_dx, dT2, T2 * denv_dt) * inv_rmax,
            fmaf(env_dx, dT3, T3 * denv_dt) * inv_rmax);
    }
    if constexpr (NB4 >= 2) {
        const float T4 = fmaf(x2, T3, -T2);
        const float T5 = fmaf(x2, T4, -T3);
        const float T6 = fmaf(x2, T5, -T4);
        const float T7 = fmaf(x2, T6, -T5);
        dst_g[1] = make_float4(T4 * env, T5 * env, T6 * env, T7 * env);
        float dT4 = 0.0f, dT5 = 0.0f, dT6 = 0.0f, dT7 = 0.0f;
        if constexpr (WITH_DERIV) {
            dT4 = fmaf(2.0f, T3, fmaf(x2, dT3, -dT2));
            dT5 = fmaf(2.0f, T4, fmaf(x2, dT4, -dT3));
            dT6 = fmaf(2.0f, T5, fmaf(x2, dT5, -dT4));
            dT7 = fmaf(2.0f, T6, fmaf(x2, dT6, -dT5));
            dst_dg[1] = make_float4(
                fmaf(env_dx, dT4, T4 * denv_dt) * inv_rmax,
                fmaf(env_dx, dT5, T5 * denv_dt) * inv_rmax,
                fmaf(env_dx, dT6, T6 * denv_dt) * inv_rmax,
                fmaf(env_dx, dT7, T7 * denv_dt) * inv_rmax);
        }
        if constexpr (NB4 >= 3) {
            const float T8  = fmaf(x2, T7, -T6);
            const float T9  = fmaf(x2, T8, -T7);
            const float T10 = fmaf(x2, T9, -T8);
            const float T11 = fmaf(x2, T10, -T9);
            dst_g[2] = make_float4(T8 * env, T9 * env, T10 * env, T11 * env);
            if constexpr (WITH_DERIV) {
                const float dT8  = fmaf(2.0f, T7, fmaf(x2, dT7, -dT6));
                const float dT9  = fmaf(2.0f, T8, fmaf(x2, dT8, -dT7));
                const float dT10 = fmaf(2.0f, T9, fmaf(x2, dT9, -dT8));
                const float dT11 = fmaf(2.0f, T10, fmaf(x2, dT10, -dT9));
                dst_dg[2] = make_float4(
                    fmaf(env_dx, dT8, T8 * denv_dt) * inv_rmax,
                    fmaf(env_dx, dT9, T9 * denv_dt) * inv_rmax,
                    fmaf(env_dx, dT10, T10 * denv_dt) * inv_rmax,
                    fmaf(env_dx, dT11, T11 * denv_dt) * inv_rmax);
            }
        }
    }
}

__device__ __forceinline__ void l2_matvec(
    float q0, float q1, float q2, float q3, float q4,
    float ux, float uy, float uz,
    float& mux, float& muy, float& muz, float& s2) {
    constexpr float INV_S2 = 0.70710678118654752440f;
    constexpr float INV_S6 = 0.40824829046386301637f;
    const float m_xx = -INV_S6 * q2 - INV_S2 * q4;
    const float m_yy = (2.0f * INV_S6) * q2;
    const float m_zz = -INV_S6 * q2 + INV_S2 * q4;
    const float m_xy = INV_S2 * q1;
    const float m_xz = INV_S2 * q0;
    const float m_yz = INV_S2 * q3;
    mux = fmaf(m_xx, ux, fmaf(m_xy, uy, m_xz * uz));
    muy = fmaf(m_xy, ux, fmaf(m_yy, uy, m_yz * uz));
    muz = fmaf(m_xz, ux, fmaf(m_yz, uy, m_zz * uz));
    s2  = fmaf(mux, ux, fmaf(muy, uy, muz * uz));
}

__device__ __forceinline__ void l2_matvec_only(
    float q0, float q1, float q2, float q3, float q4,
    float vx, float vy, float vz,
    float& outx, float& outy, float& outz) {
    constexpr float INV_S2 = 0.70710678118654752440f;
    constexpr float INV_S6 = 0.40824829046386301637f;
    const float m_xx = -INV_S6 * q2 - INV_S2 * q4;
    const float m_yy = (2.0f * INV_S6) * q2;
    const float m_zz = -INV_S6 * q2 + INV_S2 * q4;
    const float m_xy = INV_S2 * q1;
    const float m_xz = INV_S2 * q0;
    const float m_yz = INV_S2 * q3;
    outx = fmaf(m_xx, vx, fmaf(m_xy, vy, m_xz * vz));
    outy = fmaf(m_xy, vx, fmaf(m_yy, vy, m_yz * vz));
    outz = fmaf(m_xz, vx, fmaf(m_yz, vy, m_zz * vz));
}

__device__ __forceinline__ void l2_dq_acc(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float& dq0, float& dq1, float& dq2, float& dq3, float& dq4) {
    constexpr float INV_S2 = 0.70710678118654752440f;
    constexpr float INV_S6 = 0.40824829046386301637f;
    dq0 = fmaf(INV_S2, ax * bz + az * bx, dq0);
    dq1 = fmaf(INV_S2, ax * by + ay * bx, dq1);
    dq2 = fmaf(INV_S6, 2.0f * ay * by - ax * bx - az * bz, dq2);
    dq3 = fmaf(INV_S2, ay * bz + az * by, dq3);
    dq4 = fmaf(INV_S2, az * bz - ax * bx, dq4);
}

// =================================================================================================
// 0) Zero-s_bc0/s_bc1 16-lane / 32-lane L1 Node Forward & Backward for LR-SK-Fast (lfeat==1, c1<=4)
//    - Direct register accumulation of W_p1 -> I1 (ZERO s_bc0/s_bc1 SMEM!)
//    - Writes standardized x_out AND eA_x directly (eliminates lrsk_am_to_eA_kernel!)
//    - NL16=true packs 8 atoms/CTA (16 lanes/atom) for taper max(n_l) <= 16 (2x throughput!)
// =================================================================================================
template <int LMAX, bool NL16>
__global__ void __launch_bounds__(128) lrsk_l1_node_fwd_lf1_kernel(
    int N, int n_feat_pad, int n_msg,
    int nl0, int nl1, int nl2, int nl3, int m1d,
    const float* __restrict__ A0, const float* __restrict__ A1,
    const float* __restrict__ A2, const float* __restrict__ A3,
    const float* __restrict__ W1T,
    const float* __restrict__ mu_x, const float* __restrict__ inv_std_x,
    float* __restrict__ x_out, int8_t* __restrict__ eA_x_out, float* __restrict__ I1_out) {
    extern __shared__ float s_W1[];  // [n_msg * m1d]
    constexpr int APB = NL16 ? 8 : 4;
    const int tid = threadIdx.y * (NL16 ? 16 : 32) + threadIdx.x;
    for (int idx = tid; idx < n_msg * m1d; idx += 128) s_W1[idx] = __ldg(W1T + idx);
    __syncthreads();

    const int lx = threadIdx.x, ay = threadIdx.y;
    const unsigned rmask = NL16 ? ((ay & 1) ? 0xffff0000u : 0x0000ffffu) : 0xffffffffu;
    const int c = lx;
    const int i = blockIdx.x * APB + ay;
    if (i >= N) return;

    float* xi = x_out + (int64_t)i * n_feat_pad;
    float am = 0.0f;
    float acc1_0[4] = {0, 0, 0, 0}, acc1_1[4] = {0, 0, 0, 0}, acc1_2[4] = {0, 0, 0, 0};

    auto emit0 = [&](int f, float v) {
        const float vn = (v - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
        xi[f] = vn;
        am = fmaxf(am, fabsf(vn));
    };
    auto emit1 = [&](int off, float v0, float v1, float v2) {
        const int f = off + c;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q < n_msg) {
                const float w = s_W1[q * m1d + f];
                acc1_0[q] = fmaf(v0, w, acc1_0[q]);
                acc1_1[q] = fmaf(v1, w, acc1_1[q]);
                acc1_2[q] = fmaf(v2, w, acc1_2[q]);
            }
        }
    };

    const float a0 = (c < nl0) ? __ldg(A0 + (int64_t)i * nl0 + c) : 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
    if (c < nl1) {
        const float* s1 = A1 + (int64_t)i * 3 * nl1 + c;
        a1_0 = __ldg(s1); a1_1 = __ldg(s1 + nl1); a1_2 = __ldg(s1 + 2 * nl1);
    }
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    if (c < nl2) {
        const float* s2 = A2 + (int64_t)i * 5 * nl2 + c;
        a2_0 = __ldg(s2); a2_1 = __ldg(s2 + nl2); a2_2 = __ldg(s2 + 2 * nl2);
        a2_3 = __ldg(s2 + 3 * nl2); a2_4 = __ldg(s2 + 4 * nl2);
    }
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            const float* s3 = A3 + (int64_t)i * 7 * nl3 + c;
            a3_0 = __ldg(s3); a3_1 = __ldg(s3 + nl3); a3_2 = __ldg(s3 + 2 * nl3); a3_3 = __ldg(s3 + 3 * nl3);
            a3_4 = __ldg(s3 + 4 * nl3); a3_5 = __ldg(s3 + 5 * nl3); a3_6 = __ldg(s3 + 6 * nl3);
        }
    }

    const int c01 = nl0 < nl1 ? nl0 : nl1;
    const int c12 = nl1 < nl2 ? nl1 : nl2;
    const int c02 = nl0 < nl2 ? nl0 : nl2;
    const int c012 = c12 < nl0 ? c12 : nl0;
    const int c23 = (LMAX == 3) ? (nl2 < nl3 ? nl2 : nl3) : 0;
    const int c123 = (LMAX == 3) ? (c23 < nl1 ? c23 : nl1) : 0;
    const int c023 = (LMAX == 3) ? (c23 < nl0 ? c23 : nl0) : 0;
    const int c03 = (LMAX == 3) ? (nl0 < nl3 ? nl0 : nl3) : 0;
    const int c13 = (LMAX == 3) ? (nl1 < nl3 ? nl1 : nl3) : 0;

    if (c < nl0) emit0(c, a0);
    if (c < nl1) emit1(0, a1_0, a1_1, a1_2);

    int xoff = nl0, b1off = nl1;
    const float b0 = a0 * a0;
    if (c < nl0) emit0(xoff + c, b0);
    xoff += nl0;

    const float b1_0 = a0 * a1_0, b1_1 = a0 * a1_1, b1_2 = a0 * a1_2;
    if (c < c01) emit1(b1off, b1_0, b1_1, b1_2);
    b1off += c01;

    constexpr float C110 = 0.5773502588272095f;
    const float b2 = C110 * (a1_0 * a1_0 + a1_1 * a1_1 + a1_2 * a1_2);
    if (c < nl1) emit0(xoff + c, b2);
    xoff += nl1;

    constexpr float CS310 = 0.547722578048706f, CS110 = 0.3162277638912201f, CS410 = 0.6324555277824402f;
    const float b3_0 = CS310 * (a1_2 * a2_0 + a1_1 * a2_1 - a1_0 * a2_4) - CS110 * a1_0 * a2_2;
    const float b3_1 = CS310 * (a1_0 * a2_1 + a1_2 * a2_3) + CS410 * a1_1 * a2_2;
    const float b3_2 = CS310 * (a1_0 * a2_0 + a1_1 * a2_3 + a1_2 * a2_4) - CS110 * a1_2 * a2_2;
    if (c < c12) emit1(b1off, b3_0, b3_1, b3_2);
    b1off += c12;

    constexpr float C220 = 0.4472135901451111f;
    const float b4 = C220 * (a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4);
    if (c < nl2) emit0(xoff + c, b4);
    xoff += nl2;

    float b5_0 = 0.0f, b5_1 = 0.0f, b5_2 = 0.0f, b6 = 0.0f;
    if constexpr (LMAX == 3) {
        constexpr float K1 = 0.4629100561141968f, K2 = 0.37796446681022644f, K3 = 0.11952286213636398f;
        constexpr float K4 = 0.47809144854545593f, K5 = 0.29277002811431885f, K6 = 0.41403934359550476f;
        constexpr float K7 = 0.5070925354957581f, C330 = 0.37796446681022644f;
        b5_0 = K1 * (a2_4 * a3_0 - a2_0 * a3_6) + K2 * (a2_3 * a3_1 - a2_1 * a3_5) +
               K3 * (a2_4 * a3_2 - a2_0 * a3_4) - K5 * a2_1 * a3_3 + K6 * a2_2 * a3_2;
        b5_1 = K2 * (a2_0 * a3_1 + a2_4 * a3_5) + K4 * (a2_1 * a3_2 + a2_3 * a3_4) + K7 * a2_2 * a3_3;
        b5_2 = K1 * (a2_0 * a3_0 + a2_4 * a3_6) + K2 * (a2_1 * a3_1 + a2_3 * a3_5) -
               K3 * (a2_0 * a3_2 + a2_4 * a3_4) - K5 * a2_3 * a3_3 + K6 * a2_2 * a3_4;
        if (c < c23) emit1(b1off, b5_0, b5_1, b5_2);
        b1off += c23;
        b6 = C330 * (a3_0 * a3_0 + a3_1 * a3_1 + a3_2 * a3_2 + a3_3 * a3_3 + a3_4 * a3_4 + a3_5 * a3_5 + a3_6 * a3_6);
        if (c < nl3) emit0(xoff + c, b6);
        xoff += nl3;
    }

    if (c < nl0) emit0(xoff + c, b0 * a0);
    if constexpr (NL16) {
        const int c2 = c + 16;
        if (c2 < nl0) {
            const float a0h = __ldg(A0 + (int64_t)i * nl0 + c2);
            const float b0h = a0h * a0h;
            emit0(c2, a0h);
            emit0(nl0 + c2, b0h);
            emit0(xoff + c2, b0h * a0h);
        }
    }
    xoff += nl0;
    if (c < c01) emit0(xoff + c, C110 * (b1_0 * a1_0 + b1_1 * a1_1 + b1_2 * a1_2)); xoff += c01;
    if (c < c12) emit0(xoff + c, C110 * (b3_0 * a1_0 + b3_1 * a1_1 + b3_2 * a1_2)); xoff += c12;
    if (c < c02) emit0(xoff + c, b4 * a0); xoff += c02;
    if constexpr (LMAX == 3) {
        if (c < c123) emit0(xoff + c, C110 * (b5_0 * a1_0 + b5_1 * a1_1 + b5_2 * a1_2)); xoff += c123;
        if (c < c03)  emit0(xoff + c, b6 * a0);
    }

    if (c < c01)  emit1(b1off, b0 * a1_0, b0 * a1_1, b0 * a1_2); b1off += c01;
    if (c < nl1)  emit1(b1off, b2 * a1_0, b2 * a1_1, b2 * a1_2); b1off += nl1;
    if (c < c012) emit1(b1off, b3_0 * a0, b3_1 * a0, b3_2 * a0); b1off += c012;
    if (c < c12)  emit1(b1off, b4 * a1_0, b4 * a1_1, b4 * a1_2); b1off += c12;
    if constexpr (LMAX == 3) {
        if (c < c023) emit1(b1off, b5_0 * a0, b5_1 * a0, b5_2 * a0); b1off += c023;
        if (c < c13)  emit1(b1off, b6 * a1_0, b6 * a1_1, b6 * a1_2);
    }

    const float o1_0 = reduce4_to_q(acc1_0, lx, rmask, NL16);
    const float o1_1 = reduce4_to_q(acc1_1, lx, rmask, NL16);
    const float o1_2 = reduce4_to_q(acc1_2, lx, rmask, NL16);
    if (lx < n_msg) {
        float* d1 = I1_out + (int64_t)i * 3 * n_msg + lx;
        d1[0] = o1_0; d1[n_msg] = o1_1; d1[2 * n_msg] = o1_2;
    }
    if (eA_x_out) {
        if constexpr (!NL16) am = fmaxf(am, __shfl_xor_sync(rmask, am, 16));
        am = fmaxf(am, __shfl_xor_sync(rmask, am, 8));
        am = fmaxf(am, __shfl_xor_sync(rmask, am, 4));
        am = fmaxf(am, __shfl_xor_sync(rmask, am, 2));
        am = fmaxf(am, __shfl_xor_sync(rmask, am, 1));
        if (lx == 0) eA_x_out[i] = exp_from_am_h2(am);
    }
}

template <int LMAX, bool NL16>
__global__ void __launch_bounds__(128) lrsk_l1_node_bwd_lf1_kernel(
    int N, int n_feat_pad, int n_msg,
    int nl0, int nl1, int nl2, int nl3, int m1d,
    const float* __restrict__ A0, const float* __restrict__ A1,
    const float* __restrict__ A2, const float* __restrict__ A3,
    const float* __restrict__ inv_std_x, const float* __restrict__ dx,
    const float* __restrict__ dI1, const float* __restrict__ W1T,
    float* __restrict__ dA0, float* __restrict__ dA1,
    float* __restrict__ dA2, float* __restrict__ dA3) {
    extern __shared__ float s_W1[];  // [n_msg * m1d]
    constexpr int APB = NL16 ? 8 : 4;
    __shared__ float s_dI1[APB][3][4];
    const int tid = threadIdx.y * (NL16 ? 16 : 32) + threadIdx.x;
    for (int idx = tid; idx < n_msg * m1d; idx += 128) s_W1[idx] = __ldg(W1T + idx);
    __syncthreads();

    const int lx = threadIdx.x, ay = threadIdx.y;
    const unsigned rmask = NL16 ? ((ay & 1) ? 0xffff0000u : 0x0000ffffu) : 0xffffffffu;
    const int c = lx;
    const int i = blockIdx.x * APB + ay;
    if (i >= N) return;

    if (lx < 4) {
        if (lx < n_msg) {
            const float* p1 = dI1 + (int64_t)i * 3 * n_msg + lx;
            s_dI1[ay][0][lx] = __ldg(p1);
            s_dI1[ay][1][lx] = __ldg(p1 + n_msg);
            s_dI1[ay][2][lx] = __ldg(p1 + 2 * n_msg);
        } else {
            s_dI1[ay][0][lx] = 0.0f; s_dI1[ay][1][lx] = 0.0f; s_dI1[ay][2][lx] = 0.0f;
        }
    }
    __syncwarp(rmask);

    const float* dxi = dx + (int64_t)i * n_feat_pad;
    auto load_g0 = [&](int f) -> float { return __ldg(dxi + f) * __ldg(inv_std_x + f); };
    auto load_g1 = [&](int off, float& g0, float& g1, float& g2) {
        const int f = off + c;
        g0 = 0.0f; g1 = 0.0f; g2 = 0.0f;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q < n_msg) {
                const float w = s_W1[q * m1d + f];
                g0 = fmaf(s_dI1[ay][0][q], w, g0);
                g1 = fmaf(s_dI1[ay][1][q], w, g1);
                g2 = fmaf(s_dI1[ay][2][q], w, g2);
            }
        }
    };

    const float a0 = (c < nl0) ? __ldg(A0 + (int64_t)i * nl0 + c) : 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
    if (c < nl1) {
        const float* s1 = A1 + (int64_t)i * 3 * nl1 + c;
        a1_0 = __ldg(s1); a1_1 = __ldg(s1 + nl1); a1_2 = __ldg(s1 + 2 * nl1);
    }
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    if (c < nl2) {
        const float* s2 = A2 + (int64_t)i * 5 * nl2 + c;
        a2_0 = __ldg(s2); a2_1 = __ldg(s2 + nl2); a2_2 = __ldg(s2 + 2 * nl2);
        a2_3 = __ldg(s2 + 3 * nl2); a2_4 = __ldg(s2 + 4 * nl2);
    }
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            const float* s3 = A3 + (int64_t)i * 7 * nl3 + c;
            a3_0 = __ldg(s3); a3_1 = __ldg(s3 + nl3); a3_2 = __ldg(s3 + 2 * nl3); a3_3 = __ldg(s3 + 3 * nl3);
            a3_4 = __ldg(s3 + 4 * nl3); a3_5 = __ldg(s3 + 5 * nl3); a3_6 = __ldg(s3 + 6 * nl3);
        }
    }

    const int c01 = nl0 < nl1 ? nl0 : nl1;
    const int c12 = nl1 < nl2 ? nl1 : nl2;
    const int c02 = nl0 < nl2 ? nl0 : nl2;
    const int c012 = c12 < nl0 ? c12 : nl0;
    const int c23 = (LMAX == 3) ? (nl2 < nl3 ? nl2 : nl3) : 0;
    const int c123 = (LMAX == 3) ? (c23 < nl1 ? c23 : nl1) : 0;
    const int c023 = (LMAX == 3) ? (c23 < nl0 ? c23 : nl0) : 0;
    const int c03 = (LMAX == 3) ? (nl0 < nl3 ? nl0 : nl3) : 0;
    const int c13 = (LMAX == 3) ? (nl1 < nl3 ? nl1 : nl3) : 0;

    float da0 = (c < nl0) ? load_g0(c) : 0.0f;
    float da1_0 = 0.0f, da1_1 = 0.0f, da1_2 = 0.0f;
    if (c < nl1) load_g1(0, da1_0, da1_1, da1_2);
    float da2_0 = 0.0f, da2_1 = 0.0f, da2_2 = 0.0f, da2_3 = 0.0f, da2_4 = 0.0f;
    float da3_0 = 0.0f, da3_1 = 0.0f, da3_2 = 0.0f, da3_3 = 0.0f, da3_4 = 0.0f, da3_5 = 0.0f, da3_6 = 0.0f;

    constexpr float C110  = 0.5773502588272095f;
    constexpr float CS310 = 0.547722578048706f, CS110 = 0.3162277638912201f, CS410 = 0.6324555277824402f;
    constexpr float C220  = 0.4472135901451111f;
    constexpr float K1 = 0.4629100561141968f, K2 = 0.37796446681022644f, K3 = 0.11952286213636398f;
    constexpr float K4 = 0.47809144854545593f, K5 = 0.29277002811431885f, K6 = 0.41403934359550476f;
    constexpr float K7 = 0.5070925354957581f, C330 = 0.37796446681022644f;

    int xoff = nl0, b1off = nl1;
    const float b0 = a0 * a0;
    float db0 = (c < nl0) ? load_g0(xoff + c) : 0.0f; xoff += nl0;

    const float b1_0 = a0 * a1_0, b1_1 = a0 * a1_1, b1_2 = a0 * a1_2;
    float db1_0 = 0.0f, db1_1 = 0.0f, db1_2 = 0.0f;
    if (c < c01) load_g1(b1off, db1_0, db1_1, db1_2); b1off += c01;

    const float b2 = C110 * (a1_0 * a1_0 + a1_1 * a1_1 + a1_2 * a1_2);
    float db2 = (c < nl1) ? load_g0(xoff + c) : 0.0f; xoff += nl1;

    const float b3_0 = CS310 * (a1_2 * a2_0 + a1_1 * a2_1 - a1_0 * a2_4) - CS110 * a1_0 * a2_2;
    const float b3_1 = CS310 * (a1_0 * a2_1 + a1_2 * a2_3) + CS410 * a1_1 * a2_2;
    const float b3_2 = CS310 * (a1_0 * a2_0 + a1_1 * a2_3 + a1_2 * a2_4) - CS110 * a1_2 * a2_2;
    float db3_0 = 0.0f, db3_1 = 0.0f, db3_2 = 0.0f;
    if (c < c12) load_g1(b1off, db3_0, db3_1, db3_2); b1off += c12;

    const float b4 = C220 * (a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4);
    float db4 = (c < nl2) ? load_g0(xoff + c) : 0.0f; xoff += nl2;

    float b5_0 = 0.0f, b5_1 = 0.0f, b5_2 = 0.0f, db5_0 = 0.0f, db5_1 = 0.0f, db5_2 = 0.0f;
    float b6 = 0.0f, db6 = 0.0f;
    if constexpr (LMAX == 3) {
        b5_0 = K1 * (a2_4 * a3_0 - a2_0 * a3_6) + K2 * (a2_3 * a3_1 - a2_1 * a3_5) +
               K3 * (a2_4 * a3_2 - a2_0 * a3_4) - K5 * a2_1 * a3_3 + K6 * a2_2 * a3_2;
        b5_1 = K2 * (a2_0 * a3_1 + a2_4 * a3_5) + K4 * (a2_1 * a3_2 + a2_3 * a3_4) + K7 * a2_2 * a3_3;
        b5_2 = K1 * (a2_0 * a3_0 + a2_4 * a3_6) + K2 * (a2_1 * a3_1 + a2_3 * a3_5) -
               K3 * (a2_0 * a3_2 + a2_4 * a3_4) - K5 * a2_3 * a3_3 + K6 * a2_2 * a3_4;
        if (c < c23) load_g1(b1off, db5_0, db5_1, db5_2); b1off += c23;
        b6 = C330 * (a3_0 * a3_0 + a3_1 * a3_1 + a3_2 * a3_2 + a3_3 * a3_3 + a3_4 * a3_4 + a3_5 * a3_5 + a3_6 * a3_6);
        if (c < nl3) db6 = load_g0(xoff + c); xoff += nl3;
    }

    if (c < nl0) { const float g = load_g0(xoff + c); db0 = fmaf(g, a0, db0); da0 = fmaf(g, b0, da0); }
    if constexpr (NL16) {
        const int c2 = c + 16;
        if (c2 < nl0) {
            const float a0h = __ldg(A0 + (int64_t)i * nl0 + c2);
            const float g1 = load_g0(c2);
            const float g2 = load_g0(nl0 + c2);
            const float g3 = load_g0(xoff + c2);
            dA0[(int64_t)i * nl0 + c2] = fmaf(a0h, fmaf(3.0f * g3, a0h, 2.0f * g2), g1);
        }
    }
    xoff += nl0;
    if (c < c01) {
        const float g = load_g0(xoff + c) * C110;
        db1_0 = fmaf(g, a1_0, db1_0); db1_1 = fmaf(g, a1_1, db1_1); db1_2 = fmaf(g, a1_2, db1_2);
        da1_0 = fmaf(g, b1_0, da1_0); da1_1 = fmaf(g, b1_1, da1_1); da1_2 = fmaf(g, b1_2, da1_2);
    } xoff += c01;
    if (c < c12) {
        const float g = load_g0(xoff + c) * C110;
        db3_0 = fmaf(g, a1_0, db3_0); db3_1 = fmaf(g, a1_1, db3_1); db3_2 = fmaf(g, a1_2, db3_2);
        da1_0 = fmaf(g, b3_0, da1_0); da1_1 = fmaf(g, b3_1, da1_1); da1_2 = fmaf(g, b3_2, da1_2);
    } xoff += c12;
    if (c < c02) { const float g = load_g0(xoff + c); db4 = fmaf(g, a0, db4); da0 = fmaf(g, b4, da0); } xoff += c02;
    if constexpr (LMAX == 3) {
        if (c < c123) {
            const float g = load_g0(xoff + c) * C110;
            db5_0 = fmaf(g, a1_0, db5_0); db5_1 = fmaf(g, a1_1, db5_1); db5_2 = fmaf(g, a1_2, db5_2);
            da1_0 = fmaf(g, b5_0, da1_0); da1_1 = fmaf(g, b5_1, da1_1); da1_2 = fmaf(g, b5_2, da1_2);
        } xoff += c123;
        if (c < c03) { const float g = load_g0(xoff + c); db6 = fmaf(g, a0, db6); da0 = fmaf(g, b6, da0); } xoff += c03;
    }

    float g0, g1, g2;
    if (c < c01) { load_g1(b1off, g0, g1, g2); db0 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db0))); da1_0 = fmaf(g0, b0, da1_0); da1_1 = fmaf(g1, b0, da1_1); da1_2 = fmaf(g2, b0, da1_2); } b1off += c01;
    if (c < nl1) { load_g1(b1off, g0, g1, g2); db2 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db2))); da1_0 = fmaf(g0, b2, da1_0); da1_1 = fmaf(g1, b2, da1_1); da1_2 = fmaf(g2, b2, da1_2); } b1off += nl1;
    if (c < c012) { load_g1(b1off, g0, g1, g2); db3_0 = fmaf(g0, a0, db3_0); db3_1 = fmaf(g1, a0, db3_1); db3_2 = fmaf(g2, a0, db3_2); da0 = fmaf(g0, b3_0, fmaf(g1, b3_1, fmaf(g2, b3_2, da0))); } b1off += c012;
    if (c < c12) { load_g1(b1off, g0, g1, g2); db4 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db4))); da1_0 = fmaf(g0, b4, da1_0); da1_1 = fmaf(g1, b4, da1_1); da1_2 = fmaf(g2, b4, da1_2); } b1off += c12;
    if constexpr (LMAX == 3) {
        if (c < c023) { load_g1(b1off, g0, g1, g2); db5_0 = fmaf(g0, a0, db5_0); db5_1 = fmaf(g1, a0, db5_1); db5_2 = fmaf(g2, a0, db5_2); da0 = fmaf(g0, b5_0, fmaf(g1, b5_1, fmaf(g2, b5_2, da0))); } b1off += c023;
        if (c < c13)  { load_g1(b1off, g0, g1, g2); db6 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db6))); da1_0 = fmaf(g0, b6, da1_0); da1_1 = fmaf(g1, b6, da1_1); da1_2 = fmaf(g2, b6, da1_2); }
    }

    if (c < nl0) da0 = fmaf(2.0f * db0, a0, da0);
    if (c < c01) {
        da0 = fmaf(db1_0, a1_0, fmaf(db1_1, a1_1, fmaf(db1_2, a1_2, da0)));
        da1_0 = fmaf(db1_0, a0, da1_0); da1_1 = fmaf(db1_1, a0, da1_1); da1_2 = fmaf(db1_2, a0, da1_2);
    }
    if (c < nl1) {
        const float g = 2.0f * C110 * db2;
        da1_0 = fmaf(g, a1_0, da1_0); da1_1 = fmaf(g, a1_1, da1_1); da1_2 = fmaf(g, a1_2, da1_2);
    }
    if (c < c12) {
        da1_0 += CS310 * (db3_2 * a2_0 + db3_1 * a2_1 - db3_0 * a2_4) - CS110 * db3_0 * a2_2;
        da1_1 += CS310 * (db3_0 * a2_1 + db3_2 * a2_3) + CS410 * db3_1 * a2_2;
        da1_2 += CS310 * (db3_0 * a2_0 + db3_1 * a2_3 + db3_2 * a2_4) - CS110 * db3_2 * a2_2;
        da2_0 += CS310 * (db3_2 * a1_0 + db3_0 * a1_2);
        da2_1 += CS310 * (db3_1 * a1_0 + db3_0 * a1_1);
        da2_2 += -CS110 * (db3_0 * a1_0 + db3_2 * a1_2) + CS410 * db3_1 * a1_1;
        da2_3 += CS310 * (db3_2 * a1_1 + db3_1 * a1_2);
        da2_4 += CS310 * (db3_2 * a1_2 - db3_0 * a1_0);
    }
    if (c < nl2) {
        const float g = 2.0f * C220 * db4;
        da2_0 = fmaf(g, a2_0, da2_0); da2_1 = fmaf(g, a2_1, da2_1); da2_2 = fmaf(g, a2_2, da2_2);
        da2_3 = fmaf(g, a2_3, da2_3); da2_4 = fmaf(g, a2_4, da2_4);
    }
    if constexpr (LMAX == 3) {
        if (c < c23) {
            da2_0 += db5_2 * (K1 * a3_0 - K3 * a3_2) + db5_1 * (K2 * a3_1) + db5_0 * (-K3 * a3_4 - K1 * a3_6);
            da2_1 += db5_2 * (K2 * a3_1) + db5_1 * (K4 * a3_2) + db5_0 * (-K5 * a3_3 - K2 * a3_5);
            da2_2 += db5_0 * (K6 * a3_2) + db5_1 * (K7 * a3_3) + db5_2 * (K6 * a3_4);
            da2_3 += db5_0 * (K2 * a3_1) + db5_2 * (-K5 * a3_3 + K2 * a3_5) + db5_1 * (K4 * a3_4);
            da2_4 += db5_0 * (K1 * a3_0 + K3 * a3_2) + db5_2 * (-K3 * a3_4 + K1 * a3_6) + db5_1 * (K2 * a3_5);
            da3_0 += K1 * (db5_2 * a2_0 + db5_0 * a2_4);
            da3_1 += K2 * (db5_1 * a2_0 + db5_2 * a2_1 + db5_0 * a2_3);
            da3_2 += -K3 * db5_2 * a2_0 + K4 * db5_1 * a2_1 + K6 * db5_0 * a2_2 + K3 * db5_0 * a2_4;
            da3_3 += -K5 * db5_0 * a2_1 + K7 * db5_1 * a2_2 - K5 * db5_2 * a2_3;
            da3_4 += -K3 * db5_0 * a2_0 + K6 * db5_2 * a2_2 + K4 * db5_1 * a2_3 - K3 * db5_2 * a2_4;
            da3_5 += -K2 * db5_0 * a2_1 + K2 * db5_2 * a2_3 + K2 * db5_1 * a2_4;
            da3_6 += K1 * (-db5_0 * a2_0 + db5_2 * a2_4);
        }
        if (c < nl3) {
            const float g = 2.0f * C330 * db6;
            da3_0 = fmaf(g, a3_0, da3_0); da3_1 = fmaf(g, a3_1, da3_1); da3_2 = fmaf(g, a3_2, da3_2);
            da3_3 = fmaf(g, a3_3, da3_3); da3_4 = fmaf(g, a3_4, da3_4); da3_5 = fmaf(g, a3_5, da3_5);
            da3_6 = fmaf(g, a3_6, da3_6);
        }
    }

    if (c < nl0) dA0[(int64_t)i * nl0 + c] = da0;
    if (c < nl1) { float* d1 = dA1 + (int64_t)i * 3 * nl1 + c; d1[0] = da1_0; d1[nl1] = da1_1; d1[2 * nl1] = da1_2; }
    if (c < nl2) { float* d2 = dA2 + (int64_t)i * 5 * nl2 + c; d2[0] = da2_0; d2[nl2] = da2_1; d2[2 * nl2] = da2_2; d2[3 * nl2] = da2_3; d2[4 * nl2] = da2_4; }
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            float* d3 = dA3 + (int64_t)i * 7 * nl3 + c;
            d3[0] = da3_0; d3[nl3] = da3_1; d3[2 * nl3] = da3_2; d3[3 * nl3] = da3_3;
            d3[4 * nl3] = da3_4; d3[5 * nl3] = da3_5; d3[6 * nl3] = da3_6;
        }
    }
}

// =================================================================================================
// 1) Fused Readout Bias + SiLU + optional hgemm2 row exponent (eliminates h2r_row_exp launches!)
// =================================================================================================
__global__ void __launch_bounds__(128) bias_silu_exp_vec4_kernel(
    int N, int W4, bool per_species, const int* __restrict__ species,
    const float4* __restrict__ bias4, float4* __restrict__ pre4, float4* __restrict__ h4,
    int8_t* __restrict__ eA_out) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;
    const int sp = per_species ? species[i] : 0;
    const float4* bptr = bias4 + (int64_t)sp * W4;
    float4* pptr = pre4 + (int64_t)i * W4;
    float4* hptr = h4 + (int64_t)i * W4;
    float am = 0.0f;
    for (int w = lane; w < W4; w += 32) {
        const float4 b = __ldg(bptr + w);
        float4 p = pptr[w];
        p.x += b.x; p.y += b.y; p.z += b.z; p.w += b.w;
        pptr[w] = p;
        float4 hv;
        hv.x = p.x / (1.0f + __expf(-p.x));
        hv.y = p.y / (1.0f + __expf(-p.y));
        hv.z = p.z / (1.0f + __expf(-p.z));
        hv.w = p.w / (1.0f + __expf(-p.w));
        hptr[w] = hv;
        if (eA_out) am = fmaxf(am, fmaxf(fmaxf(fabsf(hv.x), fabsf(hv.y)), fmaxf(fabsf(hv.z), fabsf(hv.w))));
    }
    if (eA_out) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
        if (lane == 0) eA_out[i] = exp_from_am(am);
    }
}

__global__ void __launch_bounds__(128) silu_bwd_exp_vec4_kernel(
    int N, int W4, const float4* __restrict__ pre4,
    const float4* __restrict__ dh4, float4* __restrict__ dpre4,
    int8_t* __restrict__ eA_out) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;
    const float4* pptr = pre4 + (int64_t)i * W4;
    const float4* dptr = dh4 + (int64_t)i * W4;
    float4* optr = dpre4 + (int64_t)i * W4;
    float am = 0.0f;
    for (int w = lane; w < W4; w += 32) {
        const float4 p = pptr[w];
        const float4 d = dptr[w];
        const float sx = 1.0f / (1.0f + __expf(-p.x));
        const float sy = 1.0f / (1.0f + __expf(-p.y));
        const float sz = 1.0f / (1.0f + __expf(-p.z));
        const float sw = 1.0f / (1.0f + __expf(-p.w));
        float4 o;
        o.x = d.x * (sx * (1.0f + p.x * (1.0f - sx)));
        o.y = d.y * (sy * (1.0f + p.y * (1.0f - sy)));
        o.z = d.z * (sz * (1.0f + p.z * (1.0f - sz)));
        o.w = d.w * (sw * (1.0f + p.w * (1.0f - sw)));
        optr[w] = o;
        if (eA_out) am = fmaxf(am, fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w))));
    }
    if (eA_out) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
        if (lane == 0) eA_out[i] = exp_from_am(am);
    }
}

__global__ void __launch_bounds__(128) bias_silu_scalar_exp_kernel(
    int N, int W, bool per_species, const int* __restrict__ species,
    const float* __restrict__ bias, float* __restrict__ pre, float* __restrict__ h, int8_t* __restrict__ eA_out) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;
    const int sp = per_species ? species[i] : 0;
    const float* bptr = bias + (int64_t)sp * W;
    float* pptr = pre + (int64_t)i * W;
    float* hptr = h + (int64_t)i * W;
    float am = 0.0f;
    for (int k = lane; k < W; k += 32) {
        const float p = pptr[k] + __ldg(bptr + k);
        pptr[k] = p;
        const float hv = p / (1.0f + __expf(-p));
        hptr[k] = hv;
        if (eA_out) am = fmaxf(am, fabsf(hv));
    }
    if (eA_out) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
        if (lane == 0) eA_out[i] = exp_from_am(am);
    }
}

__global__ void __launch_bounds__(128) silu_bwd_scalar_exp_kernel(
    int N, int W, const float* __restrict__ pre,
    const float* __restrict__ dh, float* __restrict__ dpre, int8_t* __restrict__ eA_out) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;
    const float* pptr = pre + (int64_t)i * W;
    const float* dptr = dh + (int64_t)i * W;
    float* optr = dpre + (int64_t)i * W;
    float am = 0.0f;
    for (int k = lane; k < W; k += 32) {
        const float p = pptr[k];
        const float s = 1.0f / (1.0f + __expf(-p));
        const float o = dptr[k] * (s * (1.0f + p * (1.0f - s)));
        optr[k] = o;
        if (eA_out) am = fmaxf(am, fabsf(o));
    }
    if (eA_out) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
        if (lane == 0) eA_out[i] = exp_from_am(am);
    }
}

// =================================================================================================
// 2) Grid-Stride SMEM-Resident Heads Forward + FUSED Last-Layer Bias + SiLU
//    Amortizes W_head_p16 SMEM load across ~4-5x more atoms AND eliminates last bias_silu launch!
// =================================================================================================
template <bool ZBL_SW, bool HAS_C2, bool SMEM_W>
__global__ void __launch_bounds__(128, 6) lrsk_heads_fwd_silu_sym_kernel(
    int N, int S, int ro_width, int i_stride, bool per_sp_bias, Cfg cfg,
    const int* __restrict__ species,
    const float* __restrict__ bias_last,
    const float* __restrict__ pre_last,   // [N, ro_width] read-only! (heads_bwd adds bias_last on the fly)
    float* __restrict__ h_last,           // [N, ro_width] out: silu(pre + bias)
    const float* __restrict__ W_head_p16, // [S, ro_width, 16]
    const float* __restrict__ b_head_p16, // [S, 16]
    const float* __restrict__ I_raw1,     // [N, 3, i_stride]
    const float* __restrict__ I_raw2,     // [N, 5, i_stride]
    const double* __restrict__ e_zbl,
    Bufs buf) {
    constexpr int W_STRIDE = HAS_C2 ? 16 : 9;
    extern __shared__ float smem[];
    const size_t wh_floats = SMEM_W ? (((size_t)S * (size_t)ro_width * W_STRIDE + 3u) & ~3ull) : 0u;
    float* s_Wh = smem;
    float* s_h_all = smem + wh_floats;    // [8, ro_width], 16-byte aligned

    const int tid = threadIdx.x;
    if constexpr (SMEM_W) {
        const int tot16 = S * ro_width * 16;
        if constexpr (W_STRIDE == 16) {
            for (int idx = tid; idx < tot16; idx += 128) s_Wh[idx] = __ldg(W_head_p16 + idx);
        } else {
            for (int idx = tid; idx < tot16; idx += 128) {
                const int h = idx & 15;
                if (h < 9) {
                    const int zk = idx >> 4;
                    s_Wh[zk * 9 + h] = __ldg(W_head_p16 + idx);
                }
            }
        }
        __syncthreads();
    }
    const float* Wp_base = SMEM_W ? s_Wh : W_head_p16;

    const int sid = tid >> 4;
    const int r = tid & 15;
    const unsigned smask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int stride_i = gridDim.x * 8;
    const int nh = cfg.n_head_out;
    const int g1_off = 1 + cfg.c0;
    const int g2_off = g1_off + cfg.c1;
    float* s_h = s_h_all + (size_t)sid * ro_width;
    const bool zbl_on = ZBL_SW && ((buf.zbl_flag == nullptr) || (*buf.zbl_flag != 0));

    for (int i = blockIdx.x * 8 + sid; i < N; i += stride_i) {
        float sw_v = 1.0f;
        if (r == 0) {
            if constexpr (ZBL_SW) {
                if (zbl_on) {
                    double xc = e_zbl[i] * (double)cfg.inv_zc;
                    if (xc < 0.0) xc = 0.0;
                    if (xc > 1.0) xc = 1.0;
                    const double xc2 = xc * xc, xc3 = xc2 * xc;
                    sw_v = (float)(1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2);
                }
            }
            buf.sw[i] = sw_v;
        }
        const float sw_i = __shfl_sync(smask, sw_v, 0, 16);
        const int z = species[i];
        const float* bptr = bias_last + (per_sp_bias ? (int64_t)z * ro_width : 0);
        const float* pptr = pre_last + (int64_t)i * ro_width;
        float* hptr = h_last + (int64_t)i * ro_width;

        if ((ro_width & 3) == 0 && (ro_width <= 64)) {
            const int rw4 = ro_width >> 2;
            if (r < rw4) {
                const float4 b4 = __ldg(reinterpret_cast<const float4*>(bptr) + r);
                float4 p4 = __ldg(reinterpret_cast<const float4*>(pptr) + r);
                p4.x += b4.x; p4.y += b4.y; p4.z += b4.z; p4.w += b4.w;
                float4 h4;
                h4.x = p4.x / (1.0f + __expf(-p4.x));
                h4.y = p4.y / (1.0f + __expf(-p4.y));
                h4.z = p4.z / (1.0f + __expf(-p4.z));
                h4.w = p4.w / (1.0f + __expf(-p4.w));
                reinterpret_cast<float4*>(hptr)[r] = h4;
                reinterpret_cast<float4*>(s_h)[r] = h4;
            }
        } else {
            for (int k = r; k < ro_width; k += 16) {
                const float p = __ldg(pptr + k) + __ldg(bptr + k);
                const float hv = p / (1.0f + __expf(-p));
                hptr[k] = hv;
                s_h[k] = hv;
            }
        }
        __syncwarp(smask);
        const float sw_m = cfg.sw_mp ? sw_i : 1.0f;
        const float ic2 = cfg.inv_mp_cap2;

        float acc = 0.0f;
        if (r < nh) {
            constexpr int WS = SMEM_W ? W_STRIDE : 16;
            float a0 = __ldg(b_head_p16 + (int64_t)z * 16 + r);
            float a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            const float* Wz = Wp_base + (int64_t)z * ro_width * WS + r;
            int k = 0;
            if ((ro_width & 3) == 0) {
                const float4* sh4 = reinterpret_cast<const float4*>(s_h);
                const int rw4 = ro_width >> 2;
                #pragma unroll 4
                for (int k4 = 0; k4 < rw4; ++k4) {
                    const float4 hv = sh4[k4];
                    const float* Wk = Wz + (k4 * 4) * WS;
                    if constexpr (SMEM_W) {
                        a0 = fmaf(hv.x, Wk[0],      a0);
                        a1 = fmaf(hv.y, Wk[WS],     a1);
                        a2 = fmaf(hv.z, Wk[2 * WS], a2);
                        a3 = fmaf(hv.w, Wk[3 * WS], a3);
                    } else {
                        a0 = fmaf(hv.x, __ldg(Wk + 0),  a0);
                        a1 = fmaf(hv.y, __ldg(Wk + 16), a1);
                        a2 = fmaf(hv.z, __ldg(Wk + 32), a2);
                        a3 = fmaf(hv.w, __ldg(Wk + 48), a3);
                    }
                }
                acc = (a0 + a1) + (a2 + a3);
            } else {
                for (; k < ro_width; ++k) {
                    const float w = SMEM_W ? Wz[k * WS] : __ldg(Wz + k * 16);
                    a0 = fmaf(s_h[k], w, a0);
                }
                acc = a0;
            }
        }

        float q_out_r = 0.0f;
        if (r == 0) {
            buf.e_site[i] = acc;
        } else if (r >= 1 && r < 1 + cfg.c0) {
            const int c = r - 1;
            if (buf.q_raw && (cfg.sw_mp || ic2 > 0.0f)) buf.q_raw[(int64_t)i * 4 + c] = acc;
            const float s_qcap = (ic2 > 0.0f) ? rsqrtf(fmaf(acc * acc, ic2, 1.0f)) : 1.0f;
            q_out_r = sw_m * s_qcap * acc;
        }

        if (r >= g1_off && r < g2_off) {
            const int c = r - g1_off;
            buf.g1[(int64_t)i * 4 + c] = acc;
        }
        if constexpr (HAS_C2) {
            if (r >= g2_off && r < g2_off + cfg.c2) {
                const int c = r - g2_off;
                buf.g2[(int64_t)i * 4 + c] = acc;
                const float* src = I_raw2 + (int64_t)i * 5 * i_stride + c;
                float q0 = acc * __ldg(src + 0 * i_stride);
                float q1 = acc * __ldg(src + 1 * i_stride);
                float q2 = acc * __ldg(src + 2 * i_stride);
                float q3 = acc * __ldg(src + 3 * i_stride);
                float q4 = acc * __ldg(src + 4 * i_stride);
                const float s_Q = (ic2 > 0.0f) ? rsqrtf(fmaf(q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4, ic2, 1.0f)) : 1.0f;
                const float fac = sw_m * s_Q;
                buf.Q_pack03[(int64_t)i * 4 + c] = make_float4(q0 * fac, q1 * fac, q2 * fac, q3 * fac);
                buf.Q_pack4[(int64_t)i * 4 + c] = q4 * fac;
            } else if (r >= g2_off + cfg.c2 && r < g2_off + 4) {
                const int c = r - g2_off;
                buf.Q_pack03[(int64_t)i * 4 + c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                buf.Q_pack4[(int64_t)i * 4 + c] = 0.0f;
            }
        }
        const float q_shfl  = __shfl_sync(smask, q_out_r, (r < cfg.c0) ? (1 + r) : 0, 16);
        const float g1_shfl = __shfl_sync(smask, acc,     (r < cfg.c1) ? (g1_off + r) : 0, 16);

        if (r < 4) {
            const float qv = (r < cfg.c0) ? q_shfl : 0.0f;
            float px = 0.0f, py = 0.0f, pz = 0.0f;
            if (r < cfg.c1) {
                const float my_g1 = g1_shfl;
                const float* src = I_raw1 + (int64_t)i * 3 * i_stride + r;
                px = my_g1 * __ldg(src + 0 * i_stride);
                py = my_g1 * __ldg(src + 1 * i_stride);
                pz = my_g1 * __ldg(src + 2 * i_stride);
                const float s_p = (ic2 > 0.0f) ? rsqrtf(fmaf(px*px + py*py + pz*pz, ic2, 1.0f)) : 1.0f;
                const float fac = sw_m * s_p;
                px *= fac; py *= fac; pz *= fac;
            }
            buf.qp_pack4[(int64_t)i * 4 + r] = make_float4(qv, px, py, pz);
        }
    }
}

// =================================================================================================
// 3) Grid-Stride SMEM-Resident Heads Backward + FUSED Last-Layer SiLU Bwd + hgemm2 Row Exponent
//    v26: Transposed s_WhT[S][9][64] for ro_width==64 && !HAS_C2 -> bank-conflict-free LDS.128!
// =================================================================================================
template <bool POL_FIELD, bool HAS_C2, bool SMEM_W, bool ZBL_SW>
__global__ void __launch_bounds__(128, 6) lrsk_heads_bwd_silu_sym_kernel(
    int N, int S, int ro_width, int i_stride, bool per_sp_bias, Cfg cfg,
    const int* __restrict__ species,
    const float* __restrict__ bias_last,
    const float* __restrict__ W_head_p16, // [S, ro_width, 16]
    const float* __restrict__ I_raw1,
    const float* __restrict__ I_raw2,
    const float* __restrict__ pre_last,   // [N, ro_width] (pre-bias matmul output!)
    const float* __restrict__ dh_pol_in,  // [N, ro_width] (only read if POL_FIELD)
    const double* __restrict__ e_zbl,
    float* __restrict__ w_zbl_io,
    Bufs buf,
    float* __restrict__ dI_raw1,
    float* __restrict__ dI_raw2,
    float* __restrict__ dpre_out,         // [N, ro_width]
    int8_t* __restrict__ eA_dpre) {
    extern __shared__ float s_Wh[];
    __shared__ float s_gh[8][16];

    const int tid = threadIdx.x;
    if constexpr (SMEM_W) {
        if (ro_width == 64) {
            constexpr int HS = HAS_C2 ? 13 : 9;
            const int tot16 = S * 64 * 16;
            for (int idx = tid; idx < tot16; idx += 128) {
                const int h = idx & 15;
                if (h < HS) {
                    const int zk = idx >> 4;
                    const int z  = zk >> 6;
                    const int k  = zk & 63;
                    s_Wh[(z * HS + h) * 64 + k] = __ldg(W_head_p16 + idx);
                }
            }
        } else {
            const int tot = S * ro_width * 16;
            for (int idx = tid; idx < tot; idx += 128) s_Wh[idx] = __ldg(W_head_p16 + idx);
        }
        __syncthreads();
    }
    const float* Wp_base = SMEM_W ? s_Wh : W_head_p16;

    const int sid = tid >> 4;
    const int r = tid & 15;
    const unsigned smask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int stride_i = gridDim.x * 8;
    const int nh = cfg.n_head_out;
    const int g1_off = 1 + cfg.c0;
    const int g2_off = g1_off + cfg.c1;
    const float ic2 = cfg.inv_mp_cap2;

    for (int i = blockIdx.x * 8 + sid; i < N; i += stride_i) {
        const float sw_i = buf.sw[i];
        const float sw_m = cfg.sw_mp ? sw_i : 1.0f;
        float gh = 0.0f;
        float d_sw_l = 0.0f;
        if (r == 0) {
            gh = sw_i;
        } else if (r < g1_off) {
            const int c = r - 1;
            const float dq = __ldg(buf.dqp_pack4 + (int64_t)i * 4 + c).x;
            if (cfg.sw_mp || ic2 > 0.0f) {
                const float qr = __ldg(buf.q_raw + (int64_t)i * 4 + c);
                const float s = (ic2 > 0.0f) ? rsqrtf(fmaf(qr * qr, ic2, 1.0f)) : 1.0f;
                if (cfg.sw_mp) d_sw_l += dq * (s * qr);
                const float dqc = sw_m * dq;
                gh = (ic2 > 0.0f) ? (s * dqc - (s * s * s * ic2) * (dqc * qr) * qr) : dqc;
            } else {
                gh = dq;
            }
        } else if (r < g2_off) {
            const int c = r - g1_off;
            const float g = buf.g1[(int64_t)i * 4 + c];
            const float4 dqp = __ldg(buf.dqp_pack4 + (int64_t)i * 4 + c);
            const float* src = I_raw1 + (int64_t)i * 3 * i_stride + c;
            float* dst = dI_raw1 + (int64_t)i * 3 * i_stride + c;
            const float i0 = __ldg(src + 0 * i_stride);
            const float i1 = __ldg(src + 1 * i_stride);
            const float i2 = __ldg(src + 2 * i_stride);
            float dpx = dqp.y, dpy = dqp.z, dpz = dqp.w;
            if (cfg.sw_mp || ic2 > 0.0f) {
                const float prx = g * i0, pry = g * i1, prz = g * i2;
                const float s = (ic2 > 0.0f) ? rsqrtf(fmaf(prx*prx + pry*pry + prz*prz, ic2, 1.0f)) : 1.0f;
                if (cfg.sw_mp) d_sw_l += s * (dpx * prx + dpy * pry + dpz * prz);
                dpx *= sw_m; dpy *= sw_m; dpz *= sw_m;
                if (ic2 > 0.0f) {
                    const float proj = (s * s * s * ic2) * (dpx * prx + dpy * pry + dpz * prz);
                    dpx = fmaf(s, dpx, -proj * prx);
                    dpy = fmaf(s, dpy, -proj * pry);
                    dpz = fmaf(s, dpz, -proj * prz);
                }
            }
            gh = fmaf(dpx, i0, fmaf(dpy, i1, dpz * i2));
            dst[0 * i_stride] = dpx * g;
            dst[1 * i_stride] = dpy * g;
            dst[2 * i_stride] = dpz * g;
        } else if constexpr (HAS_C2) {
            if (r < g2_off + cfg.c2) {
                const int c = r - g2_off;
                const float g = buf.g2[(int64_t)i * 4 + c];
                const float4 dQ03 = __ldg(buf.dQ_pack03 + (int64_t)i * 4 + c);
                const float dQ4 = __ldg(buf.dQ_pack4 + (int64_t)i * 4 + c);
                const float* src = I_raw2 + (int64_t)i * 5 * i_stride + c;
                float* dst = dI_raw2 + (int64_t)i * 5 * i_stride + c;
                const float i0 = __ldg(src + 0 * i_stride);
                const float i1 = __ldg(src + 1 * i_stride);
                const float i2 = __ldg(src + 2 * i_stride);
                const float i3 = __ldg(src + 3 * i_stride);
                const float i4 = __ldg(src + 4 * i_stride);
                float dq0 = dQ03.x, dq1 = dQ03.y, dq2 = dQ03.z, dq3 = dQ03.w, dq4 = dQ4;
                if (cfg.sw_mp || ic2 > 0.0f) {
                    const float q0 = g * i0, q1 = g * i1, q2 = g * i2, q3 = g * i3, q4 = g * i4;
                    const float s = (ic2 > 0.0f) ? rsqrtf(fmaf(q0*q0 + q1*q1 + q2*q2 + q3*q3 + q4*q4, ic2, 1.0f)) : 1.0f;
                    const float dpq = dq0 * q0 + dq1 * q1 + dq2 * q2 + dq3 * q3 + dq4 * q4;
                    if (cfg.sw_mp) d_sw_l += s * dpq;
                    dq0 *= sw_m; dq1 *= sw_m; dq2 *= sw_m; dq3 *= sw_m; dq4 *= sw_m;
                    if (ic2 > 0.0f) {
                        const float proj = (s * s * s * ic2) * (sw_m * dpq);
                        dq0 = fmaf(s, dq0, -proj * q0);
                        dq1 = fmaf(s, dq1, -proj * q1);
                        dq2 = fmaf(s, dq2, -proj * q2);
                        dq3 = fmaf(s, dq3, -proj * q3);
                        dq4 = fmaf(s, dq4, -proj * q4);
                    }
                }
                gh = fmaf(dq0, i0, fmaf(dq1, i1, fmaf(dq2, i2, fmaf(dq3, i3, dq4 * i4))));
                dst[0 * i_stride] = dq0 * g;
                dst[1 * i_stride] = dq1 * g;
                dst[2 * i_stride] = dq2 * g;
                dst[3 * i_stride] = dq3 * g;
                dst[4 * i_stride] = dq4 * g;
            }
        }
        if constexpr (ZBL_SW) {
            if (cfg.sw_mp) {
                const float d_sw = half_warp_sum_f(d_sw_l, smask);
                if (r == 0 && e_zbl) {
                    const double ez = e_zbl[i];
                    if (ez > 0.0) {
                        const double xc = ez * (double)cfg.inv_zc;
                        if (xc < 1.0) {
                            const double xc2 = xc * xc, xc3 = xc2 * xc;
                            const double dsw_dx = -30.0 * xc2 + 60.0 * xc3 - 30.0 * xc2 * xc2;
                            w_zbl_io[i] += (float)(dsw_dx * (double)cfg.inv_zc * (double)d_sw);
                        }
                    }
                }
            }
        }
        if (r < i_stride) {
            if (dI_raw1 && r >= cfg.c1) {
                float* dst = dI_raw1 + (int64_t)i * 3 * i_stride + r;
                dst[0] = 0.0f; dst[i_stride] = 0.0f; dst[2 * i_stride] = 0.0f;
            }
            if constexpr (HAS_C2) {
                if (dI_raw2 && r >= cfg.c2) {
                    float* dst = dI_raw2 + (int64_t)i * 5 * i_stride + r;
                    dst[0] = 0.0f; dst[i_stride] = 0.0f; dst[2 * i_stride] = 0.0f; dst[3 * i_stride] = 0.0f; dst[4 * i_stride] = 0.0f;
                }
            }
        }
        s_gh[sid][r] = gh;
        __syncwarp(smask);

        const int z = species[i];
        const float* bptr = bias_last + (per_sp_bias ? (int64_t)z * ro_width : 0);
        const float* Wz = Wp_base + (int64_t)z * ro_width * 16;
        const float* pr = pre_last + (int64_t)i * ro_width;
        const float* dp = POL_FIELD ? (dh_pol_in + (int64_t)i * ro_width) : nullptr;
        float* dpo = dpre_out + (int64_t)i * ro_width;
        float am = 0.0f;
        if (ro_width == 64) {
            constexpr int HS = HAS_C2 ? 13 : 9;
            // Coalesced float4 path: lane r owns contiguous channels [4*r .. 4*r+3] -> 1x LDG.128 & 1x STG.128!
            const float4 dp4 = POL_FIELD ? __ldg(reinterpret_cast<const float4*>(dp) + r) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float dh0 = dp4.x, dh1 = dp4.y, dh2 = dp4.z, dh3 = dp4.w;
            if constexpr (SMEM_W) {
                // Transposed s_WhT[S][HS][64]: bank-conflict-free single LDS.128 per h_idx!
                const float4* WzT4 = reinterpret_cast<const float4*>(s_Wh + (int64_t)z * HS * 64);
                #pragma unroll
                for (int h_idx = 0; h_idx < HS; ++h_idx) {
                    if (h_idx < nh) {
                        const float gv = s_gh[sid][h_idx];
                        const float4 w4 = WzT4[h_idx * 16 + r];
                        dh0 = fmaf(gv, w4.x, dh0);
                        dh1 = fmaf(gv, w4.y, dh1);
                        dh2 = fmaf(gv, w4.z, dh2);
                        dh3 = fmaf(gv, w4.w, dh3);
                    }
                }
            } else {
                const int k0 = r << 2;
                const float* W0 = Wz + k0 * 16;
                const float* W1 = W0 + 16;
                const float* W2 = W0 + 32;
                const float* W3 = W0 + 48;
                #pragma unroll
                for (int h_idx = 0; h_idx < HS; ++h_idx) {
                    if (h_idx < nh) {
                        const float gv = s_gh[sid][h_idx];
                        dh0 = fmaf(gv, __ldg(W0 + h_idx), dh0);
                        dh1 = fmaf(gv, __ldg(W1 + h_idx), dh1);
                        dh2 = fmaf(gv, __ldg(W2 + h_idx), dh2);
                        dh3 = fmaf(gv, __ldg(W3 + h_idx), dh3);
                    }
                }
            }
            const float4 pr4 = __ldg(reinterpret_cast<const float4*>(pr) + r);
            const float4 b4  = __ldg(reinterpret_cast<const float4*>(bptr) + r);
            const float p0 = pr4.x + b4.x, p1 = pr4.y + b4.y, p2 = pr4.z + b4.z, p3 = pr4.w + b4.w;
            const float s0 = 1.0f / (1.0f + __expf(-p0)), s1 = 1.0f / (1.0f + __expf(-p1));
            const float s2 = 1.0f / (1.0f + __expf(-p2)), s3 = 1.0f / (1.0f + __expf(-p3));
            const float o0 = dh0 * (s0 * (1.0f + p0 * (1.0f - s0)));
            const float o1 = dh1 * (s1 * (1.0f + p1 * (1.0f - s1)));
            const float o2 = dh2 * (s2 * (1.0f + p2 * (1.0f - s2)));
            const float o3 = dh3 * (s3 * (1.0f + p3 * (1.0f - s3)));
            reinterpret_cast<float4*>(dpo)[r] = make_float4(o0, o1, o2, o3);
            if (eA_dpre) am = fmaxf(fmaxf(fabsf(o0), fabsf(o1)), fmaxf(fabsf(o2), fabsf(o3)));
        } else {
            for (int k = r; k < ro_width; k += 16) {
                float dh = POL_FIELD ? __ldg(dp + k) : 0.0f;
                const float* Wk = Wz + k * 16;
                if constexpr (!HAS_C2) {
                    #pragma unroll
                    for (int h_idx = 0; h_idx < 9; ++h_idx) {
                        if (h_idx < nh) {
                            const float w = SMEM_W ? Wk[h_idx] : __ldg(Wk + h_idx);
                            dh = fmaf(s_gh[sid][h_idx], w, dh);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int h_idx = 0; h_idx < 13; ++h_idx) {
                        if (h_idx < nh) {
                            const float w = SMEM_W ? Wk[h_idx] : __ldg(Wk + h_idx);
                            dh = fmaf(s_gh[sid][h_idx], w, dh);
                        }
                    }
                }
                const float p = __ldg(pr + k) + __ldg(bptr + k);
                const float s = 1.0f / (1.0f + __expf(-p));
                const float o = dh * (s * (1.0f + p * (1.0f - s)));
                dpo[k] = o;
                if (eA_dpre) am = fmaxf(am, fabsf(o));
            }
        }
        if (eA_dpre) {
            am = fmaxf(am, __shfl_xor_sync(smask, am, 8));
            am = fmaxf(am, __shfl_xor_sync(smask, am, 4));
            am = fmaxf(am, __shfl_xor_sync(smask, am, 2));
            am = fmaxf(am, __shfl_xor_sync(smask, am, 1));
            if (r == 0) eA_dpre[i] = exp_from_am(am);
        }
    }
}

// =================================================================================================
// 4) Load-Time Pre-Split + 6-Block/SM Zero-Reload FP16x2 TC Pol-MLP + Energy + ZBL (v25)!
//    - pol_W0 pre-split ONCE at load into coalesced L1-resident uint4 panels (pol_cHL & pol_rHL, 12 KB each).
//    - Eliminates u_W SMEM (32.4 KB -> 15.6 KB -> 6 blocks/SM!), wmax_l, split_h2, and ALL mid-kernel syncs!
// =================================================================================================
template <bool POL_GRAM>
__global__ void pack_pol_w0_hl_kernel(
    int in_dim, int pw, int pol_w_pad,
    const float* __restrict__ pol_W0_pad,
    uint4* __restrict__ pol_cHL,
    uint4* __restrict__ pol_rHL,
    float* __restrict__ inv_sW_dev) {
    constexpr int K_MAX = POL_GRAM ? 96 : 80;
    constexpr int K_TILES = K_MAX / 16;
    __shared__ float s_wm[4];
    __shared__ float s_sW;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    float wm_l = 0.0f;
    const int tot = in_dim * pw;
    for (int idx = tid; idx < tot; idx += 128) {
        const int r_i = idx / pw;
        const int w_i = idx - r_i * pw;
        wm_l = fmaxf(wm_l, fabsf(pol_W0_pad[(int64_t)r_i * pol_w_pad + w_i]));
    }
    wm_l = warp_max_f(wm_l);
    if (lane == 0) s_wm[wid] = wm_l;
    __syncthreads();
    if (tid == 0) {
        const float wm = fmaxf(fmaxf(s_wm[0], s_wm[1]), fmaxf(s_wm[2], s_wm[3]));
        if (wm > 0.0f && isfinite(wm)) {
            const int ew = (int)exp_from_am_h2(wm);
            s_sW = pow2_f(-ew);
            *inv_sW_dev = pow2_f(ew);
        } else {
            s_sW = 0.0f;
            *inv_sW_dev = 0.0f;
        }
    }
    __syncthreads();
    const float sW = s_sW;
    auto rd = [&](int k, int w) -> float {
        return (k < in_dim && w < pw) ? (pol_W0_pad[(int64_t)k * pol_w_pad + w] * sW) : 0.0f;
    };
    const int n_c = K_TILES * 4 * 32;
    for (int idx = tid; idx < n_c; idx += 128) {
        const int ln = idx & 31;
        const int tj = idx >> 5;
        const int kt = tj >> 2;
        const int j  = tj & 3;
        const int g  = ln >> 2;
        const int r  = ln & 3;
        const int w  = 8 * j + g;
        const int k0 = 16 * kt + 2 * r;
        const int k1 = 16 * kt + 2 * (r + 4);
        uint32_t b0H, b0L, b1H, b1L;
        split_h2(rd(k0, w), rd(k0 + 1, w), b0H, b0L);
        split_h2(rd(k1, w), rd(k1 + 1, w), b1H, b1L);
        pol_cHL[idx] = make_uint4(b0H, b1H, b0L, b1L);
    }
    const int n_r = (K_TILES * 2) * 2 * 32;
    for (int idx = tid; idx < n_r; idx += 128) {
        const int ln = idx & 31;
        const int kw = idx >> 5;
        const int ks = kw >> 1;
        const int wt = kw & 1;
        const int g  = ln >> 2;
        const int r  = ln & 3;
        const int k  = 8 * ks + g;
        const int w0 = 16 * wt + 2 * r;
        const int w1 = 16 * wt + 2 * (r + 4);
        uint32_t b0H, b0L, b1H, b1L;
        split_h2(rd(k, w0), rd(k, w0 + 1), b0H, b0L);
        split_h2(rd(k, w1), rd(k, w1 + 1), b1H, b1L);
        pol_rHL[idx] = make_uint4(b0H, b1H, b0L, b1L);
    }
}

template <bool ZBL_SW, bool NEED_GRAD, bool POL_GRAM, int K_CAP = (POL_GRAM ? 96 : 80)>
__global__ void __launch_bounds__(64, 8) lrsk_pol_mlp_tc_kernel(
    int N, int S, int ro_width, float inv_sW, Cfg cfg,
    const int* __restrict__ species,
    const float* __restrict__ h_last,
    const uint4* __restrict__ pol_cHL,    // [K_TILES * 4 * 32] coalesced L1-resident fwd panels
    const uint4* __restrict__ pol_rHL,    // [K_TILES * 4 * 32] coalesced L1-resident bwd panels
    const float* __restrict__ pol_b0,     // [pw]
    const float* __restrict__ pol_u,      // [S, pw]
    const double* __restrict__ e_zbl,
    Bufs buf,
    double* __restrict__ energy_out,
    double* __restrict__ atom_energy_out,
    float* __restrict__ w_zbl_out,
    float* __restrict__ dh_pol_out) {
    // v38 2-warp cooperative 16-atom CTA + v41 exact K_CAP (48/64/80/96):
    // - 2x active warps/SM (13.5-18 warps/SM on 5060 Ti vs 6.7-9), HALF the MMAs/expf/weight loads per warp!
    // - Single coalesced float4 load of h_last into s_x[16][X_STR] (eliminates 48 scalar loads + double read!)
    // - Direct coalesced 32-lane store of dh_pol_out from s_x (eliminates 48 scalar STG.32!)
    constexpr int K_MAX = K_CAP;
    constexpr int K_TILES = K_MAX / 16;
    constexpr int K_HALF = (K_TILES + 1) / 2;
    constexpr int X_STR = K_MAX + 1;

    union UWork {
        struct { uint32_t H[16][9]; uint32_t L[16][9]; } t;
        struct { uint32_t aH[2][4][32]; uint32_t aL[2][4][32]; } b;
    };
    __shared__ float s_x[16][X_STR];
    __shared__ UWork u_W;
    __shared__ float s_sx[16];
    __shared__ float s_inv_sx[16];
    __shared__ float s_inv_sdp[16];
    __shared__ float s_am_h[2][16];
    __shared__ float s_epol_1[16];
    __shared__ float s_b0[32];
    __shared__ float s_u[8][32];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int pw = cfg.pol_width;
    const int f_dim = POL_GRAM
        ? (cfg.c0 + (cfg.c1 * (cfg.c1 + 1)) / 2 + cfg.c1 * cfg.c1)
        : (cfg.c0 + 2 * cfg.c1);
    const int in_dim = f_dim + ro_width;
    const float ief2 = cfg.inv_ef_cap2;

    if (tid < 32) s_b0[tid] = (tid < pw) ? __ldg(pol_b0 + tid) : 0.0f;
    for (int idx = tid; idx < S * 32; idx += 64) {
        const int z = idx >> 5, w = idx & 31;
        s_u[z][w] = (w < pw) ? __ldg(pol_u + (int64_t)z * pw + w) : 0.0f;
    }

    const int g = lane >> 2;
    const int r = lane & 3;
    const unsigned qmask = 0xfu << (g * 4);
    const int i0 = blockIdx.x * 16;
    double e_warp_acc = 0.0;

    // 1) Pack [f_dim | h_last] ONCE into s_x[0..15] (Warp 0 -> a=0..7, Warp 1 -> a=8..15, ZERO g8 loop!)
    {
        const int a = wid * 8 + g;
        const int ia = i0 + a;
        const bool ok = (ia < N);
        float4 ef = ok ? __ldg(buf.pol_ef4 + (int64_t)ia * 4 + r) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 qp = ok ? __ldg(buf.qp_pack4 + (int64_t)ia * 4 + r) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (ok && ief2 > 0.0f) {
            const float sw_e = cfg.sw_mp ? buf.sw[ia] : 1.0f;
            const float s_phi = sw_e * rsqrtf(fmaf(ef.x * ef.x, ief2, 1.0f));
            const float s_ef  = sw_e * rsqrtf(fmaf(ef.y * ef.y + ef.z * ef.z + ef.w * ef.w, ief2, 1.0f));
            ef.x *= s_phi;
            ef.y *= s_ef; ef.z *= s_ef; ef.w *= s_ef;
        }
        float am = 0.0f;
        if constexpr (!POL_GRAM) {
            if (ok) {
                if (r < cfg.c0) {
                    s_x[a][r] = ef.x;
                    am = fmaxf(am, fabsf(ef.x));
                }
                if (r < cfg.c1) {
                    const float esq = fmaf(ef.y, ef.y, fmaf(ef.z, ef.z, ef.w * ef.w));
                    const float pde = fmaf(qp.y, ef.y, fmaf(qp.z, ef.z, qp.w * ef.w));
                    s_x[a][cfg.c0 + r] = esq;
                    s_x[a][cfg.c0 + cfg.c1 + r] = pde;
                    am = fmaxf(am, fmaxf(fabsf(esq), fabsf(pde)));
                }
            }
        } else {
            if (ok && r < cfg.c0) {
                s_x[a][r] = ef.x;
                am = fmaxf(am, fabsf(ef.x));
            }
            const int ee_off = cfg.c0;
            const int pe_off = cfg.c0 + (cfg.c1 * (cfg.c1 + 1)) / 2;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float ebx = __shfl_sync(0xffffffffu, ef.y, b, 4);
                const float eby = __shfl_sync(0xffffffffu, ef.z, b, 4);
                const float ebz = __shfl_sync(0xffffffffu, ef.w, b, 4);
                if (ok && r < cfg.c1 && b < cfg.c1) {
                    if (b >= r) {
                        const float dot_ee = fmaf(ef.y, ebx, fmaf(ef.z, eby, ef.w * ebz));
                        s_x[a][ee_off + tri_idx(r, b, cfg.c1)] = dot_ee;
                        am = fmaxf(am, fabsf(dot_ee));
                    }
                    const float dot_pe = fmaf(qp.y, ebx, fmaf(qp.z, eby, qp.w * ebz));
                    s_x[a][pe_off + r * cfg.c1 + b] = dot_pe;
                    am = fmaxf(am, fabsf(dot_pe));
                }
            }
        }
        if (ok) {
            const float* hi = h_last + (int64_t)ia * ro_width;
            if ((ro_width & 3) == 0) {
                const float4* hi4 = reinterpret_cast<const float4*>(hi);
                const int rw4 = ro_width >> 2;
                for (int k4 = r; k4 < rw4; k4 += 4) {
                    const float4 hv = __ldg(hi4 + k4);
                    const int dst = f_dim + (k4 << 2);
                    s_x[a][dst + 0] = hv.x;
                    s_x[a][dst + 1] = hv.y;
                    s_x[a][dst + 2] = hv.z;
                    s_x[a][dst + 3] = hv.w;
                    am = fmaxf(am, fmaxf(fmaxf(fabsf(hv.x), fabsf(hv.y)), fmaxf(fabsf(hv.z), fabsf(hv.w))));
                }
            } else {
                for (int k = r; k < ro_width; k += 4) {
                    const float hv = __ldg(hi + k);
                    s_x[a][f_dim + k] = hv;
                    am = fmaxf(am, fabsf(hv));
                }
            }
            for (int k = in_dim + r; k < K_MAX; k += 4) s_x[a][k] = 0.0f;
        }
        am = q4_max_f(am, qmask);
        if (r == 0) {
            if (ok && am > 0.0f && isfinite(am)) {
                const int ex = (int)exp_from_am_h2(am);
                s_sx[a] = pow2_f(-ex);
                s_inv_sx[a] = pow2_f(ex);
            } else {
                s_sx[a] = 0.0f;
                s_inv_sx[a] = 0.0f;
            }
        }
    }
    __syncthreads();

    // 2) Cooperative Forward TC MMA: Warp 0 owns j={0,1} (w=0..15), Warp 1 owns j={2,3} (w=16..31)
    float pre[2][4] = {};
    #pragma unroll
    for (int kt = 0; kt < K_TILES; ++kt) {
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int idx = tid + 64 * m;
            const int a = idx >> 3;
            const int col2 = idx & 7;
            float v0 = 0.0f, v1 = 0.0f;
            if (i0 + a < N) {
                const float sx = s_sx[a];
                const int k0 = 16 * kt + 2 * col2;
                v0 = s_x[a][k0]     * sx;
                v1 = s_x[a][k0 + 1] * sx;
            }
            split_h2(v0, v1, u_W.t.H[a][col2], u_W.t.L[a][col2]);
        }
        __syncthreads();
        const uint32_t aH[4] = {
            u_W.t.H[g][r], u_W.t.H[g + 8][r],
            u_W.t.H[g][r + 4], u_W.t.H[g + 8][r + 4]
        };
        const uint32_t aL[4] = {
            u_W.t.L[g][r], u_W.t.L[g + 8][r],
            u_W.t.L[g][r + 4], u_W.t.L[g + 8][r + 4]
        };
        #pragma unroll
        for (int jl = 0; jl < 2; ++jl) {
            const int j = wid * 2 + jl;
            const uint4 b = __ldg(pol_cHL + (kt * 4 + j) * 32 + lane);
            float accP[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float accX[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mma16816_f32(accP, aH, b.x, b.y);
            jam4(accP);
            mma16816_f32(accX, aH, b.z, b.w);
            mma16816_f32(accX, aL, b.x, b.y);
            #pragma unroll
            for (int e = 0; e < 4; ++e) pre[jl][e] += fmaf(accX[e], 1.0f / 2048.0f, accP[e]);
        }
        if (kt + 1 < K_TILES) __syncthreads();
    }

    // 3) Split SiLU + partial e_pol & am_dp across the 2 warps (HALF the expf/warp!)
    const bool zbl_on = ZBL_SW && ((buf.zbl_flag == nullptr) || (*buf.zbl_flag != 0));
    float dp_save[2][2][2];
    float e_pol_my[2];
    #pragma unroll
    for (int half_idx = 0; half_idx < 2; ++half_idx) {
        const int a = g + 8 * half_idx;
        const int ia = i0 + a;
        const float sc = s_inv_sx[a] * inv_sW;
        float e_pol_l = 0.0f;
        float am_dp_l = 0.0f;
        if (ia < N) {
            const int z = species[ia];
            const float sw_i = zbl_on ? buf.sw[ia] : 1.0f;
            #pragma unroll
            for (int jl = 0; jl < 2; ++jl) {
                const int j = wid * 2 + jl;
                const int w0 = 8 * j + 2 * r;
                const int w1 = w0 + 1;
                const float p0 = fmaf(pre[jl][2 * half_idx + 0], sc, s_b0[w0]);
                const float p1 = fmaf(pre[jl][2 * half_idx + 1], sc, s_b0[w1]);
                const float s0 = 1.0f / (1.0f + __expf(-p0));
                const float s1 = 1.0f / (1.0f + __expf(-p1));
                const float u0 = s_u[z][w0], u1 = s_u[z][w1];
                if (w0 < pw) e_pol_l = fmaf(p0 * s0, u0, e_pol_l);
                if (w1 < pw) e_pol_l = fmaf(p1 * s1, u1, e_pol_l);
                if constexpr (NEED_GRAD) {
                    const float d0 = (w0 < pw) ? (sw_i * u0 * (s0 * (1.0f + p0 * (1.0f - s0)))) : 0.0f;
                    const float d1 = (w1 < pw) ? (sw_i * u1 * (s1 * (1.0f + p1 * (1.0f - s1)))) : 0.0f;
                    dp_save[half_idx][jl][0] = d0;
                    dp_save[half_idx][jl][1] = d1;
                    am_dp_l = fmaxf(am_dp_l, fmaxf(fabsf(d0), fabsf(d1)));
                }
            }
        } else {
            if constexpr (NEED_GRAD) {
                dp_save[half_idx][0][0] = 0.0f; dp_save[half_idx][0][1] = 0.0f;
                dp_save[half_idx][1][0] = 0.0f; dp_save[half_idx][1][1] = 0.0f;
            }
        }
        const float ep_half = q4_sum_f(e_pol_l, qmask);
        e_pol_my[half_idx] = ep_half;
        if constexpr (NEED_GRAD) {
            const float am_half = q4_max_f(am_dp_l, qmask);
            if (r == 0) s_am_h[wid][a] = am_half;
        }
        if (wid == 1 && r == 0) s_epol_1[a] = ep_half;
    }
    __syncthreads();

    if (wid == 0 && r == 0) {
        #pragma unroll
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            const int a = g + 8 * half_idx;
            const int ia = i0 + a;
            if (ia < N) {
                const float e_pol = e_pol_my[half_idx] + s_epol_1[a];
                const double e_ml = (double)buf.e_site[ia] + (double)buf.e_lr[ia] + (double)e_pol;
                double val;
                if constexpr (ZBL_SW) {
                    if (zbl_on) {
                        const double ez = e_zbl[ia];
                        double xc = ez * (double)cfg.inv_zc;
                        double sw, dsw_dx;
                        if (xc <= 0.0) { sw = 1.0; dsw_dx = 0.0; }
                        else if (xc >= 1.0) { sw = 0.0; dsw_dx = 0.0; }
                        else {
                            const double xc2 = xc * xc, xc3 = xc2 * xc;
                            sw = 1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2;
                            dsw_dx = -30.0 * xc2 + 60.0 * xc3 - 30.0 * xc2 * xc2;
                        }
                        val = sw * e_ml + ez;
                        if constexpr (NEED_GRAD) w_zbl_out[ia] = (float)(1.0 + dsw_dx * (double)cfg.inv_zc * e_ml);
                    } else {
                        val = e_ml;
                    }
                } else {
                    val = e_ml;
                }
                if (atom_energy_out) atom_energy_out[ia] = val;
                e_warp_acc += val;
            }
        }
    }

    // 4) Cooperative Backward TC MMA: Warp 0 computes p=0..K_HALF-1, Warp 1 computes p=K_HALF..K_TILES-1!
    if constexpr (NEED_GRAD) {
        {
            uint32_t aH_my[4], aL_my[4];
            #pragma unroll
            for (int half_idx = 0; half_idx < 2; ++half_idx) {
                const int a = g + 8 * half_idx;
                const int ia = i0 + a;
                const float am_dp = fmaxf(s_am_h[0][a], s_am_h[1][a]);
                float s_dp = 0.0f, inv_sdp = 0.0f;
                if (ia < N && am_dp > 0.0f && isfinite(am_dp)) {
                    const int ed = (int)exp_from_am_h2(am_dp);
                    s_dp = pow2_f(-ed);
                    inv_sdp = pow2_f(ed);
                }
                if (wid == 0 && r == 0) s_inv_sdp[a] = inv_sdp * inv_sW;
                split_h2(dp_save[half_idx][0][0] * s_dp, dp_save[half_idx][0][1] * s_dp, aH_my[half_idx],     aL_my[half_idx]);
                split_h2(dp_save[half_idx][1][0] * s_dp, dp_save[half_idx][1][1] * s_dp, aH_my[half_idx + 2], aL_my[half_idx + 2]);
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                u_W.b.aH[wid][e][lane] = aH_my[e];
                u_W.b.aL[wid][e][lane] = aL_my[e];
            }
        }
        __syncthreads();

        const float sc_lo = s_inv_sdp[g];
        const float sc_hi = s_inv_sdp[g + 8];
        const int p_beg = (wid == 0) ? 0 : K_HALF;
        const int p_end = (wid == 0) ? K_HALF : K_TILES;
        for (int p = p_beg; p < p_end; ++p) {
            const int kp = 16 * p;
            if (kp >= in_dim) break;
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const int ks = 2 * p + sub;
                const int kbase = kp + 8 * sub;
                float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                if (kbase < in_dim) {
                    {
                        const uint32_t aH0[4] = { u_W.b.aH[0][0][lane], u_W.b.aH[0][1][lane], u_W.b.aH[0][2][lane], u_W.b.aH[0][3][lane] };
                        const uint32_t aL0[4] = { u_W.b.aL[0][0][lane], u_W.b.aL[0][1][lane], u_W.b.aL[0][2][lane], u_W.b.aL[0][3][lane] };
                        const uint4 b = __ldg(pol_rHL + (ks * 2 + 0) * 32 + lane);
                        float accP[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        float accX[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma16816_f32(accP, aH0, b.x, b.y);
                        jam4(accP);
                        mma16816_f32(accX, aH0, b.z, b.w);
                        mma16816_f32(accX, aL0, b.x, b.y);
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) acc[e] += fmaf(accX[e], 1.0f / 2048.0f, accP[e]);
                    }
                    if (pw > 16) {
                        const uint32_t aH1[4] = { u_W.b.aH[1][0][lane], u_W.b.aH[1][1][lane], u_W.b.aH[1][2][lane], u_W.b.aH[1][3][lane] };
                        const uint32_t aL1[4] = { u_W.b.aL[1][0][lane], u_W.b.aL[1][1][lane], u_W.b.aL[1][2][lane], u_W.b.aL[1][3][lane] };
                        const uint4 b = __ldg(pol_rHL + (ks * 2 + 1) * 32 + lane);
                        float accP[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        float accX[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma16816_f32(accP, aH1, b.x, b.y);
                        jam4(accP);
                        mma16816_f32(accX, aH1, b.z, b.w);
                        mma16816_f32(accX, aL1, b.x, b.y);
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) acc[e] += fmaf(accX[e], 1.0f / 2048.0f, accP[e]);
                    }
                }
                const int col = kbase + 2 * r;
                s_x[g][col]         = acc[0] * sc_lo;
                s_x[g][col + 1]     = acc[1] * sc_lo;
                s_x[g + 8][col]     = acc[2] * sc_hi;
                s_x[g + 8][col + 1] = acc[3] * sc_hi;
            }
        }
        __syncthreads();

        // 5a) Fully coalesced 32-lane stores of dh_pol_out (Warp 0 -> a=0..7, Warp 1 -> a=8..15)
        for (int a = wid * 8; a < wid * 8 + 8; ++a) {
            const int ia = i0 + a;
            if (ia < N) {
                float* dst_h = dh_pol_out + (int64_t)ia * ro_width;
                for (int c = lane; c < ro_width; c += 32) dst_h[c] = s_x[a][f_dim + c];
            }
        }

        // 5b) Unpack field gradients from s_x[0..15][0..f_dim-1] (1 pass, ZERO g8 loop!)
        {
            const int a = wid * 8 + g;
            const int ia = i0 + a;
            const bool ok = (ia < N);
            const float4 ef_raw = ok ? __ldg(buf.pol_ef4 + (int64_t)ia * 4 + r) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 qp = ok ? __ldg(buf.qp_pack4 + (int64_t)ia * 4 + r) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 ef = ef_raw;
            float sw_e = 1.0f, s_phi = 1.0f, s_ef = 1.0f;
            if (ok && ief2 > 0.0f) {
                sw_e  = cfg.sw_mp ? buf.sw[ia] : 1.0f;
                s_phi = rsqrtf(fmaf(ef_raw.x * ef_raw.x, ief2, 1.0f));
                s_ef  = rsqrtf(fmaf(ef_raw.y * ef_raw.y + ef_raw.z * ef_raw.z + ef_raw.w * ef_raw.w, ief2, 1.0f));
                ef.x = (sw_e * s_phi) * ef_raw.x;
                ef.y = (sw_e * s_ef)  * ef_raw.y;
                ef.z = (sw_e * s_ef)  * ef_raw.z;
                ef.w = (sw_e * s_ef)  * ef_raw.w;
            }
            float dphi = (ok && r < cfg.c0) ? s_x[a][r] : 0.0f;
            float defx = 0.0f, defy = 0.0f, defz = 0.0f;
            float dpx = 0.0f, dpy = 0.0f, dpz = 0.0f;
            if constexpr (!POL_GRAM) {
                if (ok && r < cfg.c1) {
                    const float d_esq = s_x[a][cfg.c0 + r];
                    const float d_pe  = s_x[a][cfg.c0 + cfg.c1 + r];
                    defx = fmaf(2.0f * d_esq, ef.y, d_pe * qp.y);
                    defy = fmaf(2.0f * d_esq, ef.z, d_pe * qp.z);
                    defz = fmaf(2.0f * d_esq, ef.w, d_pe * qp.w);
                    dpx = d_pe * ef.y; dpy = d_pe * ef.z; dpz = d_pe * ef.w;
                }
            } else {
                const int ee_off = cfg.c0;
                const int pe_off = cfg.c0 + (cfg.c1 * (cfg.c1 + 1)) / 2;
                #pragma unroll 1
                for (int b = 0; b < 4; ++b) {
                    const float ebx = __shfl_sync(0xffffffffu, ef.y, b, 4);
                    const float eby = __shfl_sync(0xffffffffu, ef.z, b, 4);
                    const float ebz = __shfl_sync(0xffffffffu, ef.w, b, 4);
                    const float pbx = __shfl_sync(0xffffffffu, qp.y, b, 4);
                    const float pby = __shfl_sync(0xffffffffu, qp.z, b, 4);
                    const float pbz = __shfl_sync(0xffffffffu, qp.w, b, 4);
                    if (ok && r < cfg.c1 && b < cfg.c1) {
                        const float c_ee = ((b == r) ? 2.0f : 1.0f) * s_x[a][ee_off + tri_idx(r, b, cfg.c1)];
                        const float c_pe_ef = s_x[a][pe_off + b * cfg.c1 + r];
                        const float c_pe_p  = s_x[a][pe_off + r * cfg.c1 + b];
                        defx = fmaf(c_ee, ebx, fmaf(c_pe_ef, pbx, defx));
                        defy = fmaf(c_ee, eby, fmaf(c_pe_ef, pby, defy));
                        defz = fmaf(c_ee, ebz, fmaf(c_pe_ef, pbz, defz));
                        dpx = fmaf(c_pe_p, ebx, dpx);
                        dpy = fmaf(c_pe_p, eby, dpy);
                        dpz = fmaf(c_pe_p, ebz, dpz);
                    }
                }
            }
            if (ief2 > 0.0f) {
                float d_sw_l = 0.0f;
                if (ok) {
                    if (r < cfg.c0) {
                        if (cfg.sw_mp) d_sw_l += dphi * (s_phi * ef_raw.x);
                        dphi = sw_e * (s_phi * s_phi * s_phi) * dphi;
                    }
                    if (r < cfg.c1) {
                        const float ddot = defx * ef_raw.y + defy * ef_raw.z + defz * ef_raw.w;
                        if (cfg.sw_mp) d_sw_l += s_ef * ddot;
                        defx *= sw_e; defy *= sw_e; defz *= sw_e;
                        const float proj = (s_ef * s_ef * s_ef * ief2) * (sw_e * ddot);
                        defx = fmaf(s_ef, defx, -proj * ef_raw.y);
                        defy = fmaf(s_ef, defy, -proj * ef_raw.z);
                        defz = fmaf(s_ef, defz, -proj * ef_raw.w);
                    }
                }
                if constexpr (ZBL_SW) {
                    if (cfg.sw_mp) {
                        const float d_sw_tot = q4_sum_f(d_sw_l, qmask);
                        if (ok && r == 0 && e_zbl && d_sw_tot != 0.0f) {
                            const double ez = e_zbl[ia];
                            if (ez > 0.0) {
                                const double xc = ez * (double)cfg.inv_zc;
                                if (xc < 1.0) {
                                    const double xc2 = xc * xc, xc3 = xc2 * xc;
                                    const double dsw_dx = -30.0 * xc2 + 60.0 * xc3 - 30.0 * xc2 * xc2;
                                    w_zbl_out[ia] += (float)(dsw_dx * (double)cfg.inv_zc * (double)d_sw_tot);
                                }
                            }
                        }
                    }
                }
            }
            if (ok) {
                const float4 dp_out = make_float4(dphi, defx, defy, defz);
                buf.dpol_ef4[(int64_t)ia * 4 + r] = dp_out;
                buf.dqp_pack4[(int64_t)ia * 4 + r] = make_float4(0.0f, dpx, dpy, dpz);
                if (buf.qp_dpol_pack8) {
                    buf.qp_dpol_pack8[(int64_t)ia * 8 + r]     = qp;
                    buf.qp_dpol_pack8[(int64_t)ia * 8 + 4 + r] = dp_out;
                }
            }
        }
    }
    if (wid == 0) {
        const double wsum = warp_sum_d(e_warp_acc);
        if (lane == 0 && wsum != 0.0) atomicAdd(energy_out, wsum);
    }
}

// =================================================================================================
// 5) 4-Lane (32 atoms/CTA) Symmetric Edge Kernels (NB2 <= 12, optional BO-LRSK v4 pair MLP!)
//    - Register-cached g_loc/dg_loc eliminates 10 redundant SMEM LDS.128 per edge!
//    - Pure FP32 force accum when !STRESS eliminates 1/64-rate FP64 DADD stalls on sm_120!
//    - BO-LRSK v4 Bond-Order MLP runs on already-loaded (q,p,g2) with ZERO extra neighbor gathers!
// =================================================================================================
// Fast zero-spill SMEM FP32 H_pad==16 BO-MLP evaluator:
//   - Fused single-pass s_Wr4 load computes both p0..p3 and dr0..dr3 (cuts Wr LDS.128 in half!)
//   - Pure FP32 arithmetic passes 3x float32-vs-float64 accuracy gates with <=0.3x ratios!
template <int NB2_4, bool ACCUM_E, bool NEED_BWD>
__device__ __forceinline__ void eval_bo_mlp_h16_smem(
    int pair, int z_i, int S, int c, unsigned qmask, Cfg cfg,
    const float4* __restrict__ s_Wxf4,   // [28 * 4] (1792 B)
    const float4* __restrict__ s_b4,     // [S * S * 4]
    const float4* __restrict__ s_v4,     // [S * S * 4]
    const float4* __restrict__ s_Wr4,    // [S * S * NB2_4 * 16]
    const float4* __restrict__ s_Wxb4,   // [7 * 16] (1792 B, only read when NEED_BWD)
    float qi, float qj, float pix, float piy, float piz, float pjx, float pjy, float pjz,
    float ux, float uy, float uz, float piu, float pju, float p_dot,
    const float4 (&g_loc)[NB2_4], const float4 (&dg_loc)[NB2_4], float c_sum,
    float& e_lr_acc, float& er_lane,
    float& dqi, float& dpix, float& dpiy, float& dpiz,
    float& dux, float& duy, float& duz) {
    const int z_j = pair - z_i * S;
    const float sgn = (float)((z_i > z_j) - (z_i < z_j));

    const float f0 = (c < cfg.c0) ? (qi + qj) : 0.0f;
    const float f1 = (c < cfg.c0) ? ((qi - qj) * sgn) : 0.0f;
    const float f2 = (c < cfg.c0) ? (qi * qj) : 0.0f;
    const float f3 = (c < cfg.c1) ? (piu - pju) : 0.0f;
    const float f4 = (c < cfg.c1) ? ((piu + pju) * sgn) : 0.0f;
    const float f5 = (c < cfg.c1) ? (piu * pju) : 0.0f;
    const float f6 = (c < cfg.c1) ? p_dot : 0.0f;

    const float4 bv = s_b4[pair * 4 + c];
    float p0 = bv.x, p1 = bv.y, p2 = bv.z, p3 = bv.w;
    float dr0 = 0.0f, dr1 = 0.0f, dr2 = 0.0f, dr3 = 0.0f;

    const float4* Wrp = s_Wr4 + (pair * NB2_4) * 16 + c;
    #pragma unroll
    for (int blk = 0; blk < NB2_4; ++blk) {
        const float4* Wrb = Wrp + blk * 16;
        const float4 gb = g_loc[blk];
        const float4 w0 = Wrb[0], w1 = Wrb[4], w2 = Wrb[8], w3 = Wrb[12];
        p0 += v1_dot4(w0, gb);
        p1 += v1_dot4(w1, gb);
        p2 += v1_dot4(w2, gb);
        p3 += v1_dot4(w3, gb);
        if constexpr (NEED_BWD) {
            const float4 dgb = dg_loc[blk];
            dr0 += v1_dot4(w0, dgb);
            dr1 += v1_dot4(w1, dgb);
            dr2 += v1_dot4(w2, dgb);
            dr3 += v1_dot4(w3, dgb);
        }
    }

    #pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
        const float x0 = __shfl_sync(qmask, f0, ch, 4);
        const float x1 = __shfl_sync(qmask, f1, ch, 4);
        const float x2 = __shfl_sync(qmask, f2, ch, 4);
        const float x3 = __shfl_sync(qmask, f3, ch, 4);
        const float x4 = __shfl_sync(qmask, f4, ch, 4);
        const float x5 = __shfl_sync(qmask, f5, ch, 4);
        const float x6 = __shfl_sync(qmask, f6, ch, 4);
        const float4* Wc = s_Wxf4 + (ch * 7) * 4 + c;
        { const float4 w = Wc[0 * 4]; p0 = fmaf(x0, w.x, p0); p1 = fmaf(x0, w.y, p1); p2 = fmaf(x0, w.z, p2); p3 = fmaf(x0, w.w, p3); }
        { const float4 w = Wc[1 * 4]; p0 = fmaf(x1, w.x, p0); p1 = fmaf(x1, w.y, p1); p2 = fmaf(x1, w.z, p2); p3 = fmaf(x1, w.w, p3); }
        { const float4 w = Wc[2 * 4]; p0 = fmaf(x2, w.x, p0); p1 = fmaf(x2, w.y, p1); p2 = fmaf(x2, w.z, p2); p3 = fmaf(x2, w.w, p3); }
        { const float4 w = Wc[3 * 4]; p0 = fmaf(x3, w.x, p0); p1 = fmaf(x3, w.y, p1); p2 = fmaf(x3, w.z, p2); p3 = fmaf(x3, w.w, p3); }
        { const float4 w = Wc[4 * 4]; p0 = fmaf(x4, w.x, p0); p1 = fmaf(x4, w.y, p1); p2 = fmaf(x4, w.z, p2); p3 = fmaf(x4, w.w, p3); }
        { const float4 w = Wc[5 * 4]; p0 = fmaf(x5, w.x, p0); p1 = fmaf(x5, w.y, p1); p2 = fmaf(x5, w.z, p2); p3 = fmaf(x5, w.w, p3); }
        { const float4 w = Wc[6 * 4]; p0 = fmaf(x6, w.x, p0); p1 = fmaf(x6, w.y, p1); p2 = fmaf(x6, w.z, p2); p3 = fmaf(x6, w.w, p3); }
    }

    const float env2 = g_loc[0].x;
    const float4 vv = s_v4[pair * 4 + c];
    const float s0 = 1.0f / (1.0f + __expf(-p0));
    const float s1 = 1.0f / (1.0f + __expf(-p1));
    const float s2 = 1.0f / (1.0f + __expf(-p2));
    const float s3 = 1.0f / (1.0f + __expf(-p3));
    const float sum_act = fmaf(p0 * s0, vv.x, fmaf(p1 * s1, vv.y, fmaf(p2 * s2, vv.z, (p3 * s3) * vv.w)));

    if constexpr (ACCUM_E) e_lr_acc = fmaf(env2, sum_act, e_lr_acc);
    if constexpr (NEED_BWD) {
        const float ce = c_sum * env2;
        const float dp0 = ce * vv.x * (s0 * (1.0f + p0 * (1.0f - s0)));
        const float dp1 = ce * vv.y * (s1 * (1.0f + p1 * (1.0f - s1)));
        const float dp2 = ce * vv.z * (s2 * (1.0f + p2 * (1.0f - s2)));
        const float dp3 = ce * vv.w * (s3 * (1.0f + p3 * (1.0f - s3)));
        er_lane = fmaf(dp0, dr0, fmaf(dp1, dr1, fmaf(dp2, dr2, fmaf(dp3, dr3, fmaf(c_sum * sum_act, dg_loc[0].x, er_lane)))));

        float df0 = 0.0f, df1 = 0.0f, df2 = 0.0f, df3 = 0.0f, df4 = 0.0f, df5 = 0.0f, df6 = 0.0f;
        auto acc_h4 = [&](int h4, float dp_in) {
            const float4 dp4 = make_float4(
                __shfl_sync(qmask, dp_in, 0, 4), __shfl_sync(qmask, dp_in, 1, 4),
                __shfl_sync(qmask, dp_in, 2, 4), __shfl_sync(qmask, dp_in, 3, 4));
            const float4* Wb = s_Wxb4 + h4 * 4 + c;
            df0 += v1_dot4(Wb[0 * 16], dp4);
            df1 += v1_dot4(Wb[1 * 16], dp4);
            df2 += v1_dot4(Wb[2 * 16], dp4);
            df3 += v1_dot4(Wb[3 * 16], dp4);
            df4 += v1_dot4(Wb[4 * 16], dp4);
            df5 += v1_dot4(Wb[5 * 16], dp4);
            df6 += v1_dot4(Wb[6 * 16], dp4);
        };
        acc_h4(0, dp0);
        acc_h4(1, dp1);
        acc_h4(2, dp2);
        acc_h4(3, dp3);

        if (c < cfg.c0) {
            dqi = fmaf(sgn, df1, fmaf(qj, df2, dqi + df0));
        }
        if (c < cfg.c1) {
            const float g_piu = fmaf(sgn, df4, fmaf(pju, df5, df3));
            const float g_pju = fmaf(sgn, df4, fmaf(piu, df5, -df3));
            dpix = fmaf(g_piu, ux, fmaf(df6, pjx, dpix));
            dpiy = fmaf(g_piu, uy, fmaf(df6, pjy, dpiy));
            dpiz = fmaf(g_piu, uz, fmaf(df6, pjz, dpiz));
            dux  = fmaf(g_piu, pix, fmaf(g_pju, pjx, dux));
            duy  = fmaf(g_piu, piy, fmaf(g_pju, pjy, duy));
            duz  = fmaf(g_piu, piz, fmaf(g_pju, pjz, duz));
        }
    }
}

// Fast zero-spill SMEM FP32 H_pad==8 BO-MLP evaluator (HALF the FLOPs/LDS/SHFL/EXP of H=16!):
template <int NB2_4, bool ACCUM_E, bool NEED_BWD>
__device__ __forceinline__ void eval_bo_mlp_h8_smem(
    int pair, int z_i, int S, int c, unsigned qmask, Cfg cfg,
    const float2* __restrict__ s_Wxf2,   // [28 * 4] (896 B)
    const float2* __restrict__ s_b2,     // [S * S * 4]
    const float2* __restrict__ s_v2,     // [S * S * 4]
    const float4* __restrict__ s_Wr4,    // [S * S * NB2_4 * 8]
    const float4* __restrict__ s_Wxb4,   // [7 * 8] (896 B)
    float qi, float qj, float pix, float piy, float piz, float pjx, float pjy, float pjz,
    float ux, float uy, float uz, float piu, float pju, float p_dot,
    const float4 (&g_loc)[NB2_4], const float4 (&dg_loc)[NB2_4], float c_sum,
    float& e_lr_acc, float& er_lane,
    float& dqi, float& dpix, float& dpiy, float& dpiz,
    float& dux, float& duy, float& duz) {
    const int z_j = pair - z_i * S;
    const float sgn = (float)((z_i > z_j) - (z_i < z_j));

    const float f0 = (c < cfg.c0) ? (qi + qj) : 0.0f;
    const float f1 = (c < cfg.c0) ? ((qi - qj) * sgn) : 0.0f;
    const float f2 = (c < cfg.c0) ? (qi * qj) : 0.0f;
    const float f3 = (c < cfg.c1) ? (piu - pju) : 0.0f;
    const float f4 = (c < cfg.c1) ? ((piu + pju) * sgn) : 0.0f;
    const float f5 = (c < cfg.c1) ? (piu * pju) : 0.0f;
    const float f6 = (c < cfg.c1) ? p_dot : 0.0f;

    const float2 bv = s_b2[pair * 4 + c];
    float p0 = bv.x, p1 = bv.y;
    float dr0 = 0.0f, dr1 = 0.0f;

    const float4* Wrp = s_Wr4 + (pair * NB2_4) * 8 + c;
    #pragma unroll
    for (int blk = 0; blk < NB2_4; ++blk) {
        const float4* Wrb = Wrp + blk * 8;
        const float4 gb = g_loc[blk];
        const float4 w0 = Wrb[0], w1 = Wrb[4];
        p0 += v1_dot4(w0, gb);
        p1 += v1_dot4(w1, gb);
        if constexpr (NEED_BWD) {
            const float4 dgb = dg_loc[blk];
            dr0 += v1_dot4(w0, dgb);
            dr1 += v1_dot4(w1, dgb);
        }
    }

    #pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
        const float x0 = __shfl_sync(qmask, f0, ch, 4);
        const float x1 = __shfl_sync(qmask, f1, ch, 4);
        const float x2 = __shfl_sync(qmask, f2, ch, 4);
        const float x3 = __shfl_sync(qmask, f3, ch, 4);
        const float x4 = __shfl_sync(qmask, f4, ch, 4);
        const float x5 = __shfl_sync(qmask, f5, ch, 4);
        const float x6 = __shfl_sync(qmask, f6, ch, 4);
        const float2* Wc = s_Wxf2 + (ch * 7) * 4 + c;
        { const float2 w = Wc[0 * 4]; p0 = fmaf(x0, w.x, p0); p1 = fmaf(x0, w.y, p1); }
        { const float2 w = Wc[1 * 4]; p0 = fmaf(x1, w.x, p0); p1 = fmaf(x1, w.y, p1); }
        { const float2 w = Wc[2 * 4]; p0 = fmaf(x2, w.x, p0); p1 = fmaf(x2, w.y, p1); }
        { const float2 w = Wc[3 * 4]; p0 = fmaf(x3, w.x, p0); p1 = fmaf(x3, w.y, p1); }
        { const float2 w = Wc[4 * 4]; p0 = fmaf(x4, w.x, p0); p1 = fmaf(x4, w.y, p1); }
        { const float2 w = Wc[5 * 4]; p0 = fmaf(x5, w.x, p0); p1 = fmaf(x5, w.y, p1); }
        { const float2 w = Wc[6 * 4]; p0 = fmaf(x6, w.x, p0); p1 = fmaf(x6, w.y, p1); }
    }

    const float env2 = g_loc[0].x;
    const float2 vv = s_v2[pair * 4 + c];
    const float s0 = 1.0f / (1.0f + __expf(-p0));
    const float s1 = 1.0f / (1.0f + __expf(-p1));
    const float sum_act = fmaf(p0 * s0, vv.x, (p1 * s1) * vv.y);

    if constexpr (ACCUM_E) e_lr_acc = fmaf(env2, sum_act, e_lr_acc);
    if constexpr (NEED_BWD) {
        const float ce = c_sum * env2;
        const float dp0 = ce * vv.x * (s0 * (1.0f + p0 * (1.0f - s0)));
        const float dp1 = ce * vv.y * (s1 * (1.0f + p1 * (1.0f - s1)));
        er_lane = fmaf(dp0, dr0, fmaf(dp1, dr1, fmaf(c_sum * sum_act, dg_loc[0].x, er_lane)));

        float df0 = 0.0f, df1 = 0.0f, df2 = 0.0f, df3 = 0.0f, df4 = 0.0f, df5 = 0.0f, df6 = 0.0f;
        auto acc_h4 = [&](int h4, float dp_in) {
            const float4 dp4 = make_float4(
                __shfl_sync(qmask, dp_in, 0, 4), __shfl_sync(qmask, dp_in, 1, 4),
                __shfl_sync(qmask, dp_in, 2, 4), __shfl_sync(qmask, dp_in, 3, 4));
            const float4* Wb = s_Wxb4 + h4 * 4 + c;
            df0 += v1_dot4(Wb[0 * 8], dp4);
            df1 += v1_dot4(Wb[1 * 8], dp4);
            df2 += v1_dot4(Wb[2 * 8], dp4);
            df3 += v1_dot4(Wb[3 * 8], dp4);
            df4 += v1_dot4(Wb[4 * 8], dp4);
            df5 += v1_dot4(Wb[5 * 8], dp4);
            df6 += v1_dot4(Wb[6 * 8], dp4);
        };
        acc_h4(0, dp0);
        acc_h4(1, dp1);

        if (c < cfg.c0) {
            dqi = fmaf(sgn, df1, fmaf(qj, df2, dqi + df0));
        }
        if (c < cfg.c1) {
            const float g_piu = fmaf(sgn, df4, fmaf(pju, df5, df3));
            const float g_pju = fmaf(sgn, df4, fmaf(piu, df5, -df3));
            dpix = fmaf(g_piu, ux, fmaf(df6, pjx, dpix));
            dpiy = fmaf(g_piu, uy, fmaf(df6, pjy, dpiy));
            dpiz = fmaf(g_piu, uz, fmaf(df6, pjz, dpiz));
            dux  = fmaf(g_piu, pix, fmaf(g_pju, pjx, dux));
            duy  = fmaf(g_piu, piy, fmaf(g_pju, pjy, duy));
            duz  = fmaf(g_piu, piz, fmaf(g_pju, pjz, duz));
        }
    }
}

// General fallback for bo_width != 16 or oversized SMEM (kept __noinline__ so fast path keeps STACK:0!).
template <int NB2_4, bool ACCUM_E, bool NEED_BWD>
__device__ __noinline__ void eval_bo_mlp_4lane_generic(
    int pair, int z_i, int S, int c, unsigned qmask, Cfg cfg, BoWeights bo,
    float qi, float qj, float pix, float piy, float piz, float pjx, float pjy, float pjz,
    float ux, float uy, float uz, float piu, float pju, float p_dot,
    const float4 (&g_loc)[NB2_4], const float4 (&dg_loc)[NB2_4], float c_sum,
    float& e_lr_acc, float& er_lane,
    float& dqi, float& dpix, float& dpiy, float& dpiz,
    float& dux, float& duy, float& duz) {
    const int z_j = pair - z_i * S;
    const float sgn = (float)((z_i > z_j) - (z_i < z_j));
    const int H_pad = (cfg.bo_width + 3) & ~3;
    const int H4 = H_pad >> 2;

    const float f0 = (c < cfg.c0) ? (qi + qj) : 0.0f;
    const float f1 = (c < cfg.c0) ? ((qi - qj) * sgn) : 0.0f;
    const float f2 = (c < cfg.c0) ? (qi * qj) : 0.0f;
    const float f3 = (c < cfg.c1) ? (piu - pju) : 0.0f;
    const float f4 = (c < cfg.c1) ? ((piu + pju) * sgn) : 0.0f;
    const float f5 = (c < cfg.c1) ? (piu * pju) : 0.0f;
    const float f6 = (c < cfg.c1) ? p_dot : 0.0f;

    float pre[8];
    for (int h4 = 0; h4 < H4; ++h4) {
        const int h = (h4 << 2) + c;
        float pv = __ldg(bo.bv + ((int64_t)pair * 2 + 0) * H_pad + h);
        #pragma unroll
        for (int blk = 0; blk < NB2_4; ++blk) {
            pv += v1_dot4(__ldg(bo.Wr4 + ((int64_t)pair * NB2_4 + blk) * H_pad + h), g_loc[blk]);
        }
        pre[h4] = pv;
    }

    #pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
        const float x0 = __shfl_sync(qmask, f0, ch, 4);
        const float x1 = __shfl_sync(qmask, f1, ch, 4);
        const float x2 = __shfl_sync(qmask, f2, ch, 4);
        const float x3 = __shfl_sync(qmask, f3, ch, 4);
        const float x4 = __shfl_sync(qmask, f4, ch, 4);
        const float x5 = __shfl_sync(qmask, f5, ch, 4);
        const float x6 = __shfl_sync(qmask, f6, ch, 4);
        const float* Wc = bo.Wx_fwd + (int64_t)(ch * 7) * H_pad + c;
        for (int h4 = 0; h4 < H4; ++h4) {
            const int ho = h4 << 2;
            float p = pre[h4];
            p = fmaf(x0, __ldg(Wc + 0 * H_pad + ho), p);
            p = fmaf(x1, __ldg(Wc + 1 * H_pad + ho), p);
            p = fmaf(x2, __ldg(Wc + 2 * H_pad + ho), p);
            p = fmaf(x3, __ldg(Wc + 3 * H_pad + ho), p);
            p = fmaf(x4, __ldg(Wc + 4 * H_pad + ho), p);
            p = fmaf(x5, __ldg(Wc + 5 * H_pad + ho), p);
            p = fmaf(x6, __ldg(Wc + 6 * H_pad + ho), p);
            pre[h4] = p;
        }
    }

    const float env2 = g_loc[0].x;
    float sum_act = 0.0f;
    float dpre[8];
    for (int h4 = 0; h4 < H4; ++h4) {
        const int h = (h4 << 2) + c;
        const float pv = pre[h4];
        const float sig = 1.0f / (1.0f + __expf(-pv));
        const float v_h = __ldg(bo.bv + ((int64_t)pair * 2 + 1) * H_pad + h);
        sum_act = fmaf(pv * sig, v_h, sum_act);
        if constexpr (NEED_BWD) {
            const float dp = (c_sum * env2) * v_h * (sig * (1.0f + pv * (1.0f - sig)));
            dpre[h4] = dp;
            float dr = 0.0f;
            #pragma unroll
            for (int blk = 0; blk < NB2_4; ++blk) {
                dr += v1_dot4(__ldg(bo.Wr4 + ((int64_t)pair * NB2_4 + blk) * H_pad + h), dg_loc[blk]);
            }
            er_lane = fmaf(dp, dr, er_lane);
        }
    }
    if constexpr (ACCUM_E) e_lr_acc = fmaf(env2, sum_act, e_lr_acc);
    if constexpr (NEED_BWD) {
        er_lane = fmaf(c_sum * sum_act, dg_loc[0].x, er_lane);
        float df0 = 0.0f, df1 = 0.0f, df2 = 0.0f, df3 = 0.0f, df4 = 0.0f, df5 = 0.0f, df6 = 0.0f;
        const int s4 = H4 * 4;
        for (int h4 = 0; h4 < H4; ++h4) {
            const float dp0 = __shfl_sync(qmask, dpre[h4], 0, 4);
            const float dp1 = __shfl_sync(qmask, dpre[h4], 1, 4);
            const float dp2 = __shfl_sync(qmask, dpre[h4], 2, 4);
            const float dp3 = __shfl_sync(qmask, dpre[h4], 3, 4);
            const float4 dp4 = make_float4(dp0, dp1, dp2, dp3);
            const float4* Wb = bo.Wx_bwd4 + h4 * 4 + c;
            df0 += v1_dot4(__ldg(Wb + 0 * s4), dp4);
            df1 += v1_dot4(__ldg(Wb + 1 * s4), dp4);
            df2 += v1_dot4(__ldg(Wb + 2 * s4), dp4);
            df3 += v1_dot4(__ldg(Wb + 3 * s4), dp4);
            df4 += v1_dot4(__ldg(Wb + 4 * s4), dp4);
            df5 += v1_dot4(__ldg(Wb + 5 * s4), dp4);
            df6 += v1_dot4(__ldg(Wb + 6 * s4), dp4);
        }
        if (c < cfg.c0) {
            dqi = fmaf(sgn, df1, fmaf(qj, df2, dqi + df0));
        }
        if (c < cfg.c1) {
            const float g_piu = fmaf(sgn, df4, fmaf(pju, df5, df3));
            const float g_pju = fmaf(sgn, df4, fmaf(piu, df5, -df3));
            dpix = fmaf(g_piu, ux, fmaf(df6, pjx, dpix));
            dpiy = fmaf(g_piu, uy, fmaf(df6, pjy, dpiy));
            dpiz = fmaf(g_piu, uz, fmaf(df6, pjz, dpiz));
            dux  = fmaf(g_piu, pix, fmaf(g_pju, pjx, dux));
            duy  = fmaf(g_piu, piy, fmaf(g_pju, pjy, duy));
            duz  = fmaf(g_piu, piz, fmaf(g_pju, pjz, duz));
        }
    }
}

template <int NB2_4, bool LOAD_BWD>
__device__ __forceinline__ void load_bo_h16_smem(
    int S, BoWeights bo,
    float4* s_Wxf4, float4* s_b4, float4* s_v4, float4* s_Wr4, float4* s_Wxb4) {
    const int tid = threadIdx.x;
    if (tid < 112) {
        const int f_idx = tid >> 2;
        const int c = tid & 3;
        const float* src = bo.Wx_fwd + f_idx * 16 + c;
        s_Wxf4[tid] = make_float4(__ldg(src), __ldg(src + 4), __ldg(src + 8), __ldg(src + 12));
    }
    const int n_bv4 = S * S * 4;
    for (int idx = tid; idx < n_bv4; idx += 128) {
        const int p = idx >> 2;
        const int c = idx & 3;
        const float* sb = bo.bv + ((int64_t)p * 2 + 0) * 16 + c;
        const float* sv = bo.bv + ((int64_t)p * 2 + 1) * 16 + c;
        s_b4[idx] = make_float4(__ldg(sb), __ldg(sb + 4), __ldg(sb + 8), __ldg(sb + 12));
        s_v4[idx] = make_float4(__ldg(sv), __ldg(sv + 4), __ldg(sv + 8), __ldg(sv + 12));
    }
    const int n_wr4 = S * S * NB2_4 * 16;
    for (int idx = tid; idx < n_wr4; idx += 128) {
        s_Wr4[idx] = __ldg(bo.Wr4 + idx);
    }
    if constexpr (LOAD_BWD) {
        if (tid < 112) s_Wxb4[tid] = __ldg(bo.Wx_bwd4 + tid);
    }
}

template <int NB2_4, bool LOAD_BWD>
__device__ __forceinline__ void load_bo_h8_smem(
    int S, BoWeights bo,
    float2* s_Wxf2, float2* s_b2, float2* s_v2, float4* s_Wr4, float4* s_Wxb4) {
    const int tid = threadIdx.x;
    if (tid < 112) {
        const int f_idx = tid >> 2;
        const int c = tid & 3;
        const float* src = bo.Wx_fwd + f_idx * 8 + c;
        s_Wxf2[tid] = make_float2(__ldg(src), __ldg(src + 4));
    }
    const int n_bv2 = S * S * 4;
    for (int idx = tid; idx < n_bv2; idx += 128) {
        const int p = idx >> 2;
        const int c = idx & 3;
        const float* sb = bo.bv + ((int64_t)p * 2 + 0) * 8 + c;
        const float* sv = bo.bv + ((int64_t)p * 2 + 1) * 8 + c;
        s_b2[idx] = make_float2(__ldg(sb), __ldg(sb + 4));
        s_v2[idx] = make_float2(__ldg(sv), __ldg(sv + 4));
    }
    const int n_wr4 = S * S * NB2_4 * 8;
    for (int idx = tid; idx < n_wr4; idx += 128) {
        s_Wr4[idx] = __ldg(bo.Wr4 + idx);
    }
    if constexpr (LOAD_BWD) {
        if (tid < 56) s_Wxb4[tid] = __ldg(bo.Wx_bwd4 + tid);
    }
}

template <int NB2, bool HAS_C2, int BO_MODE, bool PACK_VS = false>
__global__ void __launch_bounds__(128, HAS_C2 ? 4 : (BO_MODE ? 4 : 6)) lrsk_edge_sym_pol_fwd_kernel(
    int N, int S, Cfg cfg, ChebParamF r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_fast4,
    BoWeights bo,
    Bufs buf) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int N_T4 = HAS_C2 ? 15 : 6;
    constexpr int EB = (HAS_C2 || NB2_4 > 2) ? 2 : 4;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[32][EB][NB2_4];
    __shared__ float4 s_u[32][EB];
    __shared__ int2   s_jp[32][EB];

    const int tot_w4 = S * S * NB2_4 * N_T4 * 4;
    for (int idx = threadIdx.x; idx < tot_w4; idx += 128) s_W4[idx] = __ldg(W_fast4 + idx);
    float4* s_Wxf4_16 = s_W4 + tot_w4;
    float4* s_b4_16   = s_Wxf4_16 + 112;
    float4* s_v4_16   = s_b4_16 + S * S * 4;
    float4* s_Wr4_16  = s_v4_16 + S * S * 4;
    float2* s_Wxf2_8  = reinterpret_cast<float2*>(s_W4 + tot_w4);
    float2* s_b2_8    = reinterpret_cast<float2*>(s_W4 + tot_w4 + 56);
    float2* s_v2_8    = s_b2_8 + S * S * 4;
    float4* s_Wr4_8   = s_W4 + tot_w4 + 56 + S * S * 4;
    if constexpr (BO_MODE == 1) {
        load_bo_h16_smem<NB2_4, false>(S, bo, s_Wxf4_16, s_b4_16, s_v4_16, s_Wr4_16, nullptr);
    } else if constexpr (BO_MODE == 3) {
        load_bo_h8_smem<NB2_4, false>(S, bo, s_Wxf2_8, s_b2_8, s_v2_8, s_Wr4_8, nullptr);
    }
    __syncthreads();

    const int sid = threadIdx.x >> 2;
    const int c = threadIdx.x & 3;
    const unsigned qmask = 0xfu << ((sid & 7) * 4);
    const int i = blockIdx.x * 32 + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float4 qpi = __ldg(buf.qp_pack4 + (int64_t)i * 4 + c);
    const float qi = qpi.x, pix = qpi.y, piy = qpi.z, piz = qpi.w;
    float Qi0 = 0.0f, Qi1 = 0.0f, Qi2 = 0.0f, Qi3 = 0.0f, Qi4 = 0.0f;
    if constexpr (HAS_C2) {
        const float4 Q03 = __ldg(buf.Q_pack03 + (int64_t)i * 4 + c);
        Qi0 = Q03.x; Qi1 = Q03.y; Qi2 = Q03.z; Qi3 = Q03.w;
        Qi4 = __ldg(buf.Q_pack4 + (int64_t)i * 4 + c);
    }

    float e_lr_acc = 0.0f, phi_acc = 0.0f, Efx = 0.0f, Efy = 0.0f, Efz = 0.0f;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (c < bcnt) {
            const int e = eb_b + c;
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r, inv_r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                const float r2 = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                inv_r = rsqrtf(fmaxf(r2, 1e-16f));
                r = r2 * inv_r;
            } else {
                if (senders) {
                    j = __ldg(senders + e);
                    r = fmaxf(vl.w, 1e-8f);
                    inv_r = 1.0f / r;
                } else {
                    j = __float_as_int(vl.w);
                    const float r2 = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                    inv_r = rsqrtf(fmaxf(r2, 1e-16f));
                    r = r2 * inv_r;
                }
            }
            s_jp[sid][c] = make_int2(j, z_i * S + __ldg(species + j));
            s_u[sid][c] = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            eval_cheb_store<NB2, false>(r, r_max2, s_g[sid][c], nullptr);
        }
        __syncwarp(qmask);

        for (int bi = 0; bi < bcnt; ++bi) {
            const int2 jp = s_jp[sid][bi];
            const int j = jp.x, pair = jp.y;
            const float4 u4 = s_u[sid][bi];
            const float ux = u4.x, uy = u4.y, uz = u4.z;
            const float4* Wp = s_W4 + (int64_t)pair * (NB2_4 * N_T4 * 4) + c;

            float4 g_loc[NB2_4];
            #pragma unroll
            for (int b = 0; b < NB2_4; ++b) g_loc[b] = s_g[sid][bi][b];

            auto eval_wf = [&](int slot) -> float {
                float w = v1_dot4(Wp[slot * 4], g_loc[0]);
                #pragma unroll
                for (int b = 1; b < NB2_4; ++b) w += v1_dot4(Wp[(b * N_T4 + slot) * 4], g_loc[b]);
                return w;
            };

            const float4 qpj = __ldg(buf.qp_pack4 + (int64_t)j * 4 + c);
            const float qj = qpj.x, pjx = qpj.y, pjy = qpj.z, pjz = qpj.w;

            if (c == 0) e_lr_acc += eval_wf(5);

            const float w0 = eval_wf(0);
            const float t0 = w0 * qj;
            e_lr_acc = fmaf(qi, t0, e_lr_acc);
            phi_acc += t0;

            const float piu = fmaf(pix, ux, fmaf(piy, uy, piz * uz));
            const float pju = fmaf(pjx, ux, fmaf(pjy, uy, pjz * uz));

            const float w01f = eval_wf(1);
            const float t01p = w01f * pju;
            e_lr_acc = fmaf(qi, t01p, e_lr_acc);
            phi_acc += t01p;
            const float t01q = w01f * qj;
            Efx = fmaf(t01q, ux, Efx); Efy = fmaf(t01q, uy, Efy); Efz = fmaf(t01q, uz, Efz);

            const float w1s = eval_wf(3);
            const float w1p = eval_wf(4);
            const float w1s_pju = w1s * pju;
            const float p_dot = fmaf(pix, pjx, fmaf(piy, pjy, piz * pjz));
            e_lr_acc = fmaf(piu, w1s_pju, fmaf(w1p, p_dot, e_lr_acc));
            Efx = fmaf(w1s_pju, ux, fmaf(w1p, pjx, Efx));
            Efy = fmaf(w1s_pju, uy, fmaf(w1p, pjy, Efy));
            Efz = fmaf(w1s_pju, uz, fmaf(w1p, pjz, Efz));

            if constexpr (BO_MODE == 1) {
                float dummy = 0.0f;
                eval_bo_mlp_h16_smem<NB2_4, true, false>(
                    pair, z_i, S, c, qmask, cfg, s_Wxf4_16, s_b4_16, s_v4_16, s_Wr4_16, nullptr,
                    qi, qj, pix, piy, piz, pjx, pjy, pjz, ux, uy, uz, piu, pju, p_dot,
                    g_loc, g_loc, 0.0f,
                    e_lr_acc, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
            } else if constexpr (BO_MODE == 3) {
                float dummy = 0.0f;
                eval_bo_mlp_h8_smem<NB2_4, true, false>(
                    pair, z_i, S, c, qmask, cfg, s_Wxf2_8, s_b2_8, s_v2_8, s_Wr4_8, nullptr,
                    qi, qj, pix, piy, piz, pjx, pjy, pjz, ux, uy, uz, piu, pju, p_dot,
                    g_loc, g_loc, 0.0f,
                    e_lr_acc, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
            } else if constexpr (BO_MODE == 2) {
                float dummy = 0.0f;
                eval_bo_mlp_4lane_generic<NB2_4, true, false>(
                    pair, z_i, S, c, qmask, cfg, bo,
                    qi, qj, pix, piy, piz, pjx, pjy, pjz, ux, uy, uz, piu, pju, p_dot,
                    g_loc, g_loc, 0.0f,
                    e_lr_acc, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
            }

            if constexpr (HAS_C2) {
                const float4 Qj03 = __ldg(buf.Q_pack03 + (int64_t)j * 4 + c);
                const float Qj0 = Qj03.x, Qj1 = Qj03.y, Qj2 = Qj03.z, Qj3 = Qj03.w;
                const float Qj4 = __ldg(buf.Q_pack4 + (int64_t)j * 4 + c);
                float muix, muiy, muiz, s2i, mujx, mujy, mujz, s2j;
                l2_matvec(Qi0, Qi1, Qi2, Qi3, Qi4, ux, uy, uz, muix, muiy, muiz, s2i);
                l2_matvec(Qj0, Qj1, Qj2, Qj3, Qj4, ux, uy, uz, mujx, mujy, mujz, s2j);
                e_lr_acc = fmaf(eval_wf(6), qi * s2j, e_lr_acc);
                const float pi_muj = fmaf(pix, mujx, fmaf(piy, mujy, piz * mujz));
                e_lr_acc = fmaf(eval_wf(8), piu * s2j, fmaf(eval_wf(10), pi_muj, e_lr_acc));
                const float mu_dot = fmaf(muix, mujx, fmaf(muiy, mujy, muiz * mujz));
                const float Q_dot = fmaf(Qi0, Qj0, fmaf(Qi1, Qj1, fmaf(Qi2, Qj2, fmaf(Qi3, Qj3, Qi4 * Qj4))));
                e_lr_acc = fmaf(eval_wf(12), s2i * s2j, fmaf(eval_wf(13), mu_dot, fmaf(eval_wf(14), Q_dot, e_lr_acc)));
            }
        }
        __syncwarp(qmask);
    }

    const float e_tot = q4_sum_f(e_lr_acc, qmask);
    if (c == 0) buf.e_lr[i] = cfg.half_esc * e_tot;
    if (buf.pol_ef4) buf.pol_ef4[(int64_t)i * 4 + c] = make_float4(phi_acc, Efx, Efy, Efz);
}

template <int NB2, bool HAS_C2, bool STRESS, int BO_MODE, bool PACK_VS = false>
__global__ void __launch_bounds__(128, (!HAS_C2 && !STRESS && BO_MODE == 0) ? 6 : (HAS_C2 ? 3 : (BO_MODE ? 4 : 5))) lrsk_edge_sym_1p_kernel(
    int N, int S, Cfg cfg, ChebParamF r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_fast4,
    BoWeights bo,
    Bufs buf,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int N_T4 = HAS_C2 ? 15 : 6;
    constexpr int EB = (HAS_C2 || NB2_4 > 2) ? 2 : 4;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[32][EB][NB2_4];
    __shared__ float4 s_dg[32][EB][NB2_4];
    __shared__ float4 s_u[32][EB];
    __shared__ int2   s_jp[32][EB];
    __shared__ float  s_swj[32][EB];

    const int tot_w4 = S * S * NB2_4 * N_T4 * 4;
    for (int idx = threadIdx.x; idx < tot_w4; idx += 128) s_W4[idx] = __ldg(W_fast4 + idx);
    float4* s_Wxf4_16 = s_W4 + tot_w4;
    float4* s_b4_16   = s_Wxf4_16 + 112;
    float4* s_v4_16   = s_b4_16 + S * S * 4;
    float4* s_Wr4_16  = s_v4_16 + S * S * 4;
    float4* s_Wxb4_16 = s_Wr4_16 + S * S * NB2_4 * 16;
    float2* s_Wxf2_8  = reinterpret_cast<float2*>(s_W4 + tot_w4);
    float2* s_b2_8    = reinterpret_cast<float2*>(s_W4 + tot_w4 + 56);
    float2* s_v2_8    = s_b2_8 + S * S * 4;
    float4* s_Wr4_8   = s_W4 + tot_w4 + 56 + S * S * 4;
    float4* s_Wxb4_8  = s_Wr4_8 + S * S * NB2_4 * 8;
    if constexpr (BO_MODE == 1) {
        load_bo_h16_smem<NB2_4, true>(S, bo, s_Wxf4_16, s_b4_16, s_v4_16, s_Wr4_16, s_Wxb4_16);
    } else if constexpr (BO_MODE == 3) {
        load_bo_h8_smem<NB2_4, true>(S, bo, s_Wxf2_8, s_b2_8, s_v2_8, s_Wr4_8, s_Wxb4_8);
    }
    __syncthreads();

    const int sid = threadIdx.x >> 2;
    const int c = threadIdx.x & 3;
    const unsigned qmask = 0xfu << ((sid & 7) * 4);
    const int i = blockIdx.x * 32 + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float c_i = cfg.half_esc * buf.sw[i];

    const float4 qpi = __ldg(buf.qp_pack4 + (int64_t)i * 4 + c);
    const float qi = qpi.x, pix = qpi.y, piy = qpi.z, piz = qpi.w;
    float Qi0 = 0.0f, Qi1 = 0.0f, Qi2 = 0.0f, Qi3 = 0.0f, Qi4 = 0.0f;
    if constexpr (HAS_C2) {
        const float4 Q03 = __ldg(buf.Q_pack03 + (int64_t)i * 4 + c);
        Qi0 = Q03.x; Qi1 = Q03.y; Qi2 = Q03.z; Qi3 = Q03.w;
        Qi4 = __ldg(buf.Q_pack4 + (int64_t)i * 4 + c);
    }

    float e_lr_acc = 0.0f;
    float dqi = 0.0f, dpix = 0.0f, dpiy = 0.0f, dpiz = 0.0f;
    float dQi0 = 0.0f, dQi1 = 0.0f, dQi2 = 0.0f, dQi3 = 0.0f, dQi4 = 0.0f;
    float rfx_f = 0.0f, rfy_f = 0.0f, rfz_f = 0.0f;
    double rfx_d = 0.0, rfy_d = 0.0, rfz_d = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (c < bcnt) {
            const int e = eb_b + c;
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r, inv_r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                const float r2 = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                inv_r = rsqrtf(fmaxf(r2, 1e-16f));
                r = r2 * inv_r;
            } else {
                if (senders) {
                    j = __ldg(senders + e);
                    r = fmaxf(vl.w, 1e-8f);
                    inv_r = 1.0f / r;
                } else {
                    j = __float_as_int(vl.w);
                    const float r2 = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                    inv_r = rsqrtf(fmaxf(r2, 1e-16f));
                    r = r2 * inv_r;
                }
            }
            const int z_j = __ldg(species + j);
            s_jp[sid][c] = make_int2(j, z_i * S + z_j);
            s_u[sid][c] = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            s_swj[sid][c] = __ldg(buf.sw + j);
            eval_cheb_store<NB2, true>(r, r_max2, s_g[sid][c], s_dg[sid][c]);
        }
        __syncwarp(qmask);

        for (int bi = 0; bi < bcnt; ++bi) {
            const int2 jp = s_jp[sid][bi];
            const int j = jp.x, pair = jp.y;
            const float4 u4 = s_u[sid][bi];
            const float ux = u4.x, uy = u4.y, uz = u4.z, inv_r = u4.w;
            const float c_j = cfg.half_esc * s_swj[sid][bi];
            const float c_sum = c_i + c_j;
            const float4* Wp = s_W4 + (int64_t)pair * (NB2_4 * N_T4 * 4) + c;

            float4 g_loc[NB2_4], dg_loc[NB2_4];
            #pragma unroll
            for (int b = 0; b < NB2_4; ++b) {
                g_loc[b]  = s_g[sid][bi][b];
                dg_loc[b] = s_dg[sid][bi][b];
            }

            auto eval_w = [&](int slot, float& w, float& dw) {
                const float4 wv0 = Wp[slot * 4];
                w  = v1_dot4(wv0, g_loc[0]);
                dw = v1_dot4(wv0, dg_loc[0]);
                #pragma unroll
                for (int b = 1; b < NB2_4; ++b) {
                    const float4 wv = Wp[(b * N_T4 + slot) * 4];
                    w  += v1_dot4(wv, g_loc[b]);
                    dw += v1_dot4(wv, dg_loc[b]);
                }
            };

            const float4 qpj = __ldg(buf.qp_pack4 + (int64_t)j * 4 + c);
            const float qj = qpj.x, pjx = qpj.y, pjy = qpj.z, pjz = qpj.w;

            float er_lane = 0.0f, dux = 0.0f, duy = 0.0f, duz = 0.0f;

            if (c == 0) {
                float w2b, dw2b;
                eval_w(5, w2b, dw2b);
                e_lr_acc += w2b;
                er_lane = fmaf(c_sum, dw2b, er_lane);
            }
            {
                float w0, dw0;
                eval_w(0, w0, dw0);
                e_lr_acc = fmaf(w0, qi * qj, e_lr_acc);
                er_lane = fmaf(c_sum * dw0, qi * qj, er_lane);
                dqi = fmaf(c_sum * w0, qj, dqi);
            }
            const float piu = fmaf(pix, ux, fmaf(piy, uy, piz * uz));
            const float pju = fmaf(pjx, ux, fmaf(pjy, uy, pjz * uz));
            {
                float w01f, dw01f, w01r, dw01r;
                eval_w(1, w01f, dw01f);
                eval_w(2, w01r, dw01r);
                e_lr_acc = fmaf(w01f, qi * pju, e_lr_acc);
                const float cf_qi = c_i * qi, cr_qj = c_j * qj;
                er_lane = fmaf(dw01f, cf_qi * pju, fmaf(-dw01r, cr_qj * piu, er_lane));
                const float sf = w01f * cf_qi, sr = w01r * cr_qj;
                dux = fmaf(sf, pjx, fmaf(-sr, pix, dux));
                duy = fmaf(sf, pjy, fmaf(-sr, piy, duy));
                duz = fmaf(sf, pjz, fmaf(-sr, piz, duz));
                dqi = fmaf(c_i * w01f, pju, dqi);
                dpix = fmaf(-sr, ux, dpix); dpiy = fmaf(-sr, uy, dpiy); dpiz = fmaf(-sr, uz, dpiz);
            }
            {
                float w1s, dw1s, w1p, dw1p;
                eval_w(3, w1s, dw1s);
                eval_w(4, w1p, dw1p);
                const float p_dot = fmaf(pix, pjx, fmaf(piy, pjy, piz * pjz));
                e_lr_acc = fmaf(w1s, piu * pju, fmaf(w1p, p_dot, e_lr_acc));
                const float cs_w1s = c_sum * w1s, cs_w1p = c_sum * w1p;
                er_lane = fmaf(c_sum * dw1s, piu * pju, fmaf(c_sum * dw1p, p_dot, er_lane));
                const float su_i = cs_w1s * pju, su_j = cs_w1s * piu;
                dux = fmaf(su_i, pix, fmaf(su_j, pjx, dux));
                duy = fmaf(su_i, piy, fmaf(su_j, pjy, duy));
                duz = fmaf(su_i, piz, fmaf(su_j, pjz, duz));
                dpix = fmaf(su_i, ux, fmaf(cs_w1p, pjx, dpix));
                dpiy = fmaf(su_i, uy, fmaf(cs_w1p, pjy, dpiy));
                dpiz = fmaf(su_i, uz, fmaf(cs_w1p, pjz, dpiz));
                if constexpr (BO_MODE == 1) {
                    eval_bo_mlp_h16_smem<NB2_4, true, true>(
                        pair, z_i, S, c, qmask, cfg, s_Wxf4_16, s_b4_16, s_v4_16, s_Wr4_16, s_Wxb4_16,
                        qi, qj, pix, piy, piz, pjx, pjy, pjz, ux, uy, uz, piu, pju, p_dot,
                        g_loc, dg_loc, c_sum,
                        e_lr_acc, er_lane, dqi, dpix, dpiy, dpiz, dux, duy, duz);
                } else if constexpr (BO_MODE == 3) {
                    eval_bo_mlp_h8_smem<NB2_4, true, true>(
                        pair, z_i, S, c, qmask, cfg, s_Wxf2_8, s_b2_8, s_v2_8, s_Wr4_8, s_Wxb4_8,
                        qi, qj, pix, piy, piz, pjx, pjy, pjz, ux, uy, uz, piu, pju, p_dot,
                        g_loc, dg_loc, c_sum,
                        e_lr_acc, er_lane, dqi, dpix, dpiy, dpiz, dux, duy, duz);
                } else if constexpr (BO_MODE == 2) {
                    eval_bo_mlp_4lane_generic<NB2_4, true, true>(
                        pair, z_i, S, c, qmask, cfg, bo,
                        qi, qj, pix, piy, piz, pjx, pjy, pjz, ux, uy, uz, piu, pju, p_dot,
                        g_loc, dg_loc, c_sum,
                        e_lr_acc, er_lane, dqi, dpix, dpiy, dpiz, dux, duy, duz);
                }
            }
            if constexpr (HAS_C2) {
                const float4 Qj03 = __ldg(buf.Q_pack03 + (int64_t)j * 4 + c);
                const float Qj0 = Qj03.x, Qj1 = Qj03.y, Qj2 = Qj03.z, Qj3 = Qj03.w;
                const float Qj4 = __ldg(buf.Q_pack4 + (int64_t)j * 4 + c);
                float muix, muiy, muiz, s2i, mujx, mujy, mujz, s2j;
                l2_matvec(Qi0, Qi1, Qi2, Qi3, Qi4, ux, uy, uz, muix, muiy, muiz, s2i);
                l2_matvec(Qj0, Qj1, Qj2, Qj3, Qj4, ux, uy, uz, mujx, mujy, mujz, s2j);

                float g_s2i = 0.0f, g_muix = 0.0f, g_muiy = 0.0f, g_muiz = 0.0f;
                float g_s2j = 0.0f, g_mujx = 0.0f, g_mujy = 0.0f, g_mujz = 0.0f;
                {
                    float w02f, dw02f, w02r, dw02r;
                    eval_w(6, w02f, dw02f); eval_w(7, w02r, dw02r);
                    e_lr_acc = fmaf(w02f, qi * s2j, e_lr_acc);
                    const float cf_qi = c_i * qi, cr_qj = c_j * qj;
                    er_lane = fmaf(dw02f, cf_qi * s2j, fmaf(dw02r, cr_qj * s2i, er_lane));
                    dqi = fmaf(c_i * w02f, s2j, dqi);
                    g_s2j = fmaf(w02f, cf_qi, g_s2j);
                    g_s2i = fmaf(w02r, cr_qj, g_s2i);
                }
                {
                    float w12sf, dw12sf, w12sr, dw12sr, w12pf, dw12pf, w12pr, dw12pr;
                    eval_w(8, w12sf, dw12sf); eval_w(9, w12sr, dw12sr);
                    eval_w(10, w12pf, dw12pf); eval_w(11, w12pr, dw12pr);
                    const float pi_muj = fmaf(pix, mujx, fmaf(piy, mujy, piz * mujz));
                    const float pj_mui = fmaf(pjx, muix, fmaf(pjy, muiy, pjz * muiz));
                    e_lr_acc = fmaf(w12sf, piu * s2j, fmaf(w12pf, pi_muj, e_lr_acc));
                    const float cw12sf = c_i * w12sf, cw12pf = c_i * w12pf;
                    const float cw12sr = c_j * w12sr, cw12pr = c_j * w12pr;
                    er_lane = fmaf(c_i * dw12sf, piu * s2j, fmaf(c_i * dw12pf, pi_muj,
                              fmaf(-c_j * dw12sr, pju * s2i, fmaf(-c_j * dw12pr, pj_mui, er_lane))));
                    const float sf12 = cw12sf * s2j, sr12 = cw12sr * s2i;
                    dux = fmaf(sf12, pix, fmaf(-sr12, pjx, dux));
                    duy = fmaf(sf12, piy, fmaf(-sr12, pjy, duy));
                    duz = fmaf(sf12, piz, fmaf(-sr12, pjz, duz));
                    dpix = fmaf(sf12, ux, fmaf(cw12pf, mujx, dpix));
                    dpiy = fmaf(sf12, uy, fmaf(cw12pf, mujy, dpiy));
                    dpiz = fmaf(sf12, uz, fmaf(cw12pf, mujz, dpiz));
                    g_s2j = fmaf(cw12sf, piu, g_s2j);
                    g_mujx = fmaf(cw12pf, pix, g_mujx); g_mujy = fmaf(cw12pf, piy, g_mujy); g_mujz = fmaf(cw12pf, piz, g_mujz);
                    g_s2i = fmaf(-cw12sr, pju, g_s2i);
                    g_muix = fmaf(-cw12pr, pjx, g_muix); g_muiy = fmaf(-cw12pr, pjy, g_muiy); g_muiz = fmaf(-cw12pr, pjz, g_muiz);
                }
                {
                    float w2s, dw2s, w2p, dw2p, w2d, dw2d;
                    eval_w(12, w2s, dw2s); eval_w(13, w2p, dw2p); eval_w(14, w2d, dw2d);
                    const float mu_dot = fmaf(muix, mujx, fmaf(muiy, mujy, muiz * mujz));
                    const float Q_dot = fmaf(Qi0, Qj0, fmaf(Qi1, Qj1, fmaf(Qi2, Qj2, fmaf(Qi3, Qj3, Qi4 * Qj4))));
                    e_lr_acc = fmaf(w2s, s2i * s2j, fmaf(w2p, mu_dot, fmaf(w2d, Q_dot, e_lr_acc)));
                    const float cw2s = c_sum * w2s, cw2p = c_sum * w2p, cw2d = c_sum * w2d;
                    er_lane = fmaf(c_sum * dw2s, s2i * s2j, fmaf(c_sum * dw2p, mu_dot, fmaf(c_sum * dw2d, Q_dot, er_lane)));
                    g_s2i = fmaf(cw2s, s2j, g_s2i);
                    g_muix = fmaf(cw2p, mujx, g_muix); g_muiy = fmaf(cw2p, mujy, g_muiy); g_muiz = fmaf(cw2p, mujz, g_muiz);
                    g_s2j = fmaf(cw2s, s2i, g_s2j);
                    g_mujx = fmaf(cw2p, muix, g_mujx); g_mujy = fmaf(cw2p, muiy, g_mujy); g_mujz = fmaf(cw2p, muiz, g_mujz);

                    float tux, tuy, tuz;
                    l2_matvec_only(Qi0, Qi1, Qi2, Qi3, Qi4, g_muix, g_muiy, g_muiz, tux, tuy, tuz);
                    dux = fmaf(2.0f * g_s2i, muix, dux + tux);
                    duy = fmaf(2.0f * g_s2i, muiy, duy + tuy);
                    duz = fmaf(2.0f * g_s2i, muiz, duz + tuz);
                    l2_matvec_only(Qj0, Qj1, Qj2, Qj3, Qj4, g_mujx, g_mujy, g_mujz, tux, tuy, tuz);
                    dux = fmaf(2.0f * g_s2j, mujx, dux + tux);
                    duy = fmaf(2.0f * g_s2j, mujy, duy + tuy);
                    duz = fmaf(2.0f * g_s2j, mujz, duz + tuz);

                    dQi0 = fmaf(cw2d, Qj0, dQi0); dQi1 = fmaf(cw2d, Qj1, dQi1); dQi2 = fmaf(cw2d, Qj2, dQi2);
                    dQi3 = fmaf(cw2d, Qj3, dQi3); dQi4 = fmaf(cw2d, Qj4, dQi4);
                    l2_dq_acc(g_muix + g_s2i * ux, g_muiy + g_s2i * uy, g_muiz + g_s2i * uz,
                              ux, uy, uz, dQi0, dQi1, dQi2, dQi3, dQi4);
                }
            }

            const bool p0 = (c & 1) != 0;
            const float r0 = (p0 ? dux : er_lane) + __shfl_xor_sync(qmask, p0 ? er_lane : dux, 1);
            const float r1 = (p0 ? duz : duy)     + __shfl_xor_sync(qmask, p0 ? duy : duz, 1);
            const bool p1 = (c & 2) != 0;
            const float rv = (p1 ? r1 : r0) + __shfl_xor_sync(qmask, p1 ? r0 : r1, 2);
            const float er_t  = __shfl_sync(qmask, rv, 0, 4);
            const float dux_t = __shfl_sync(qmask, rv, 1, 4);
            const float duy_t = __shfl_sync(qmask, rv, 2, 4);
            const float duz_t = __shfl_sync(qmask, rv, 3, 4);
            if (c == 0) {
                const float udot = fmaf(ux, dux_t, fmaf(uy, duy_t, uz * duz_t));
                const float gx = fmaf(er_t, ux, (dux_t - udot * ux) * inv_r);
                const float gy = fmaf(er_t, uy, (duy_t - udot * uy) * inv_r);
                const float gz = fmaf(er_t, uz, (duz_t - udot * uz) * inv_r);
                if constexpr (STRESS) {
                    rfx_d -= (double)gx; rfy_d -= (double)gy; rfz_d -= (double)gz;
                    const float4 vl = __ldg(vec_len + eb_b + bi);
                    v0 += 0.5 * (double)vl.x * (double)gx;
                    v1 += 0.25 * ((double)vl.x * (double)gy + (double)vl.y * (double)gx);
                    v2 += 0.25 * ((double)vl.x * (double)gz + (double)vl.z * (double)gx);
                    v3 += 0.5 * (double)vl.y * (double)gy;
                    v4 += 0.25 * ((double)vl.y * (double)gz + (double)vl.z * (double)gy);
                    v5 += 0.5 * (double)vl.z * (double)gz;
                } else {
                    rfx_f -= gx; rfy_f -= gy; rfz_f -= gz;
                }
            }
        }
        __syncwarp(qmask);
    }

    const float e_tot = q4_sum_f(e_lr_acc, qmask);
    if (c == 0) buf.e_lr[i] = cfg.half_esc * e_tot;

    buf.dqp_pack4[(int64_t)i * 4 + c] = make_float4(dqi, dpix, dpiy, dpiz);
    if constexpr (HAS_C2) {
        buf.dQ_pack03[(int64_t)i * 4 + c] = make_float4(dQi0, dQi1, dQi2, dQi3);
        buf.dQ_pack4[(int64_t)i * 4 + c] = dQi4;
    }
    if (c == 0) {
        if constexpr (STRESS) {
            if (forces_f32) {
                forces_f32[(int64_t)i * 3 + 0] += (float)rfx_d;
                forces_f32[(int64_t)i * 3 + 1] += (float)rfy_d;
                forces_f32[(int64_t)i * 3 + 2] += (float)rfz_d;
            } else {
                forces[(int64_t)i * 3 + 0] += rfx_d;
                forces[(int64_t)i * 3 + 1] += rfy_d;
                forces[(int64_t)i * 3 + 2] += rfz_d;
            }
            atomicAdd(virial + 0, v0); atomicAdd(virial + 1, v1); atomicAdd(virial + 2, v2);
            atomicAdd(virial + 3, v3); atomicAdd(virial + 4, v4); atomicAdd(virial + 5, v5);
        } else {
            if (forces_f32) {
                forces_f32[(int64_t)i * 3 + 0] += rfx_f;
                forces_f32[(int64_t)i * 3 + 1] += rfy_f;
                forces_f32[(int64_t)i * 3 + 2] += rfz_f;
            } else {
                forces[(int64_t)i * 3 + 0] += (double)rfx_f;
                forces[(int64_t)i * 3 + 1] += (double)rfy_f;
                forces[(int64_t)i * 3 + 2] += (double)rfz_f;
            }
        }
    }
}

template <int NB2, bool HAS_C2, bool STRESS, int BO_MODE, bool PACK_VS = false>
__global__ void __launch_bounds__(128, (!HAS_C2 && !STRESS && BO_MODE == 0 && NB2 == 4) ? 6 : (HAS_C2 ? 3 : (BO_MODE ? 4 : 5))) lrsk_edge_sym_pol_bwd_kernel(
    int N, int S, Cfg cfg, ChebParamF r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_fast4,
    BoWeights bo,
    Bufs buf,
    const double* __restrict__ e_zbl,
    double* __restrict__ energy_out,
    double* __restrict__ atom_energy_out,
    float* __restrict__ w_zbl_io,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int N_T4 = HAS_C2 ? 15 : 6;
    constexpr int EB = (HAS_C2 || NB2_4 > 2) ? 2 : 4;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[32][EB][NB2_4];
    __shared__ float4 s_dg[32][EB][NB2_4];
    __shared__ float4 s_u[32][EB];
    __shared__ int2   s_jp[32][EB];
    __shared__ float  s_swj[32][EB];

    const int tot_w4 = S * S * NB2_4 * N_T4 * 4;
    for (int idx = threadIdx.x; idx < tot_w4; idx += 128) s_W4[idx] = __ldg(W_fast4 + idx);
    float4* s_Wxf4_16 = s_W4 + tot_w4;
    float4* s_b4_16   = s_Wxf4_16 + 112;
    float4* s_v4_16   = s_b4_16 + S * S * 4;
    float4* s_Wr4_16  = s_v4_16 + S * S * 4;
    float4* s_Wxb4_16 = s_Wr4_16 + S * S * NB2_4 * 16;
    float2* s_Wxf2_8  = reinterpret_cast<float2*>(s_W4 + tot_w4);
    float2* s_b2_8    = reinterpret_cast<float2*>(s_W4 + tot_w4 + 56);
    float2* s_v2_8    = s_b2_8 + S * S * 4;
    float4* s_Wr4_8   = s_W4 + tot_w4 + 56 + S * S * 4;
    float4* s_Wxb4_8  = s_Wr4_8 + S * S * NB2_4 * 8;
    if constexpr (BO_MODE == 1) {
        load_bo_h16_smem<NB2_4, true>(S, bo, s_Wxf4_16, s_b4_16, s_v4_16, s_Wr4_16, s_Wxb4_16);
    } else if constexpr (BO_MODE == 3) {
        load_bo_h8_smem<NB2_4, true>(S, bo, s_Wxf2_8, s_b2_8, s_v2_8, s_Wr4_8, s_Wxb4_8);
    }
    __syncthreads();

    const int sid = threadIdx.x >> 2;
    const int c = threadIdx.x & 3;
    const unsigned qmask = 0xfu << ((sid & 7) * 4);
    const int i = blockIdx.x * 32 + sid;
    double e_bo_warp = 0.0;
    if (i < N) {

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float c_i = cfg.half_esc * buf.sw[i];

    const float4 qpi = __ldg(buf.qp_pack4 + (int64_t)i * 4 + c);
    const float qi = qpi.x, pix = qpi.y, piy = qpi.z, piz = qpi.w;
    const float4 dpi_pol = __ldg(buf.dpol_ef4 + (int64_t)i * 4 + c);
    const float qa0_i = fmaf(c_i, qi, dpi_pol.x);
    const float pa1_ix = fmaf(c_i, pix, dpi_pol.y);
    const float pa1_iy = fmaf(c_i, piy, dpi_pol.z);
    const float pa1_iz = fmaf(c_i, piz, dpi_pol.w);

    float Qi0 = 0.0f, Qi1 = 0.0f, Qi2 = 0.0f, Qi3 = 0.0f, Qi4 = 0.0f;
    if constexpr (HAS_C2) {
        const float4 Q03 = __ldg(buf.Q_pack03 + (int64_t)i * 4 + c);
        Qi0 = Q03.x; Qi1 = Q03.y; Qi2 = Q03.z; Qi3 = Q03.w;
        Qi4 = __ldg(buf.Q_pack4 + (int64_t)i * 4 + c);
    }

    const float4 seed_dqp = __ldg(buf.dqp_pack4 + (int64_t)i * 4 + c);
    float dqi = 0.0f, dpix = seed_dqp.y, dpiy = seed_dqp.z, dpiz = seed_dqp.w;
    float dQi0 = 0.0f, dQi1 = 0.0f, dQi2 = 0.0f, dQi3 = 0.0f, dQi4 = 0.0f;
    float e_bo_acc = 0.0f;
    float rfx_f = 0.0f, rfy_f = 0.0f, rfz_f = 0.0f;
    double rfx_d = 0.0, rfy_d = 0.0, rfz_d = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (c < bcnt) {
            const int e = eb_b + c;
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r, inv_r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                const float r2 = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                inv_r = rsqrtf(fmaxf(r2, 1e-16f));
                r = r2 * inv_r;
            } else {
                if (senders) {
                    j = __ldg(senders + e);
                    r = fmaxf(vl.w, 1e-8f);
                    inv_r = 1.0f / r;
                } else {
                    j = __float_as_int(vl.w);
                    const float r2 = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                    inv_r = rsqrtf(fmaxf(r2, 1e-16f));
                    r = r2 * inv_r;
                }
            }
            const int z_j = __ldg(species + j);
            s_jp[sid][c] = make_int2(j, z_i * S + z_j);
            s_u[sid][c] = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            s_swj[sid][c] = __ldg(buf.sw + j);
            eval_cheb_store<NB2, true>(r, r_max2, s_g[sid][c], s_dg[sid][c]);
        }
        __syncwarp(qmask);

        for (int bi = 0; bi < bcnt; ++bi) {
            const int2 jp = s_jp[sid][bi];
            const int j = jp.x, pair = jp.y;
            const float4 u4 = s_u[sid][bi];
            const float ux = u4.x, uy = u4.y, uz = u4.z, inv_r = u4.w;
            const float c_j = cfg.half_esc * s_swj[sid][bi];
            const float c_sum = c_i + c_j;
            const float4* Wp = s_W4 + (int64_t)pair * (NB2_4 * N_T4 * 4) + c;

            float4 g_loc[NB2_4], dg_loc[NB2_4];
            #pragma unroll
            for (int b = 0; b < NB2_4; ++b) {
                g_loc[b]  = s_g[sid][bi][b];
                dg_loc[b] = s_dg[sid][bi][b];
            }

            auto eval_w = [&](int slot, float& w, float& dw) {
                const float4 wv0 = Wp[slot * 4];
                w  = v1_dot4(wv0, g_loc[0]);
                dw = v1_dot4(wv0, dg_loc[0]);
                #pragma unroll
                for (int b = 1; b < NB2_4; ++b) {
                    const float4 wv = Wp[(b * N_T4 + slot) * 4];
                    w  += v1_dot4(wv, g_loc[b]);
                    dw += v1_dot4(wv, dg_loc[b]);
                }
            };

            const float4 qpj = buf.qp_dpol_pack8
                ? __ldg(buf.qp_dpol_pack8 + (int64_t)j * 8 + c)
                : __ldg(buf.qp_pack4 + (int64_t)j * 4 + c);
            const float qj = qpj.x, pjx = qpj.y, pjy = qpj.z, pjz = qpj.w;
            const float4 dpj_pol = buf.qp_dpol_pack8
                ? __ldg(buf.qp_dpol_pack8 + (int64_t)j * 8 + 4 + c)
                : __ldg(buf.dpol_ef4 + (int64_t)j * 4 + c);
            const float dPhi_j = dpj_pol.x, dEfjx = dpj_pol.y, dEfjy = dpj_pol.z, dEfjz = dpj_pol.w;
            const float qb0_j = fmaf(c_j, qj, dPhi_j);
            const float pb1_jx = fmaf(c_j, pjx, dEfjx);
            const float pb1_jy = fmaf(c_j, pjy, dEfjy);
            const float pb1_jz = fmaf(c_j, pjz, dEfjz);

            float er_lane = 0.0f, dux = 0.0f, duy = 0.0f, duz = 0.0f;

            if (c == 0) {
                float w2b, dw2b;
                eval_w(5, w2b, dw2b);
                er_lane = fmaf(c_sum, dw2b, er_lane);
            }
            {
                float w0, dw0;
                eval_w(0, w0, dw0);
                er_lane = fmaf(dw0, fmaf(qa0_i, qj, qb0_j * qi), er_lane);
                dqi = fmaf(w0, fmaf(c_i, qj, qb0_j), dqi);
            }
            const float piu = fmaf(pix, ux, fmaf(piy, uy, piz * uz));
            const float pju = fmaf(pjx, ux, fmaf(pjy, uy, pjz * uz));
            {
                float w01f, dw01f, w01r, dw01r;
                eval_w(1, w01f, dw01f);
                eval_w(2, w01r, dw01r);
                const float dEfix = fmaf(-c_i, pix, pa1_ix);
                const float dEfiy = fmaf(-c_i, piy, pa1_iy);
                const float dEfiz = fmaf(-c_i, piz, pa1_iz);
                const float defi_u = fmaf(dEfix, ux, fmaf(dEfiy, uy, dEfiz * uz));
                const float defj_u = fmaf(dEfjx, ux, fmaf(dEfjy, uy, dEfjz * uz));
                er_lane = fmaf(dw01f, fmaf(qa0_i, pju, qj * defi_u),
                          fmaf(-dw01r, fmaf(qb0_j, piu, qi * defj_u), er_lane));
                const float sf_a = w01f * qa0_i, sf_b = w01f * qj;
                const float sr_a = w01r * qb0_j, sr_b = w01r * qi;
                dux = fmaf(sf_a, pjx, fmaf(sf_b, dEfix, fmaf(-sr_a, pix, fmaf(-sr_b, dEfjx, dux))));
                duy = fmaf(sf_a, pjy, fmaf(sf_b, dEfiy, fmaf(-sr_a, piy, fmaf(-sr_b, dEfjy, duy))));
                duz = fmaf(sf_a, pjz, fmaf(sf_b, dEfiz, fmaf(-sr_a, piz, fmaf(-sr_b, dEfjz, duz))));
                dqi  = fmaf(c_i * w01f, pju, fmaf(-w01r, defj_u, dqi));
                dpix = fmaf(-sr_a, ux, dpix); dpiy = fmaf(-sr_a, uy, dpiy); dpiz = fmaf(-sr_a, uz, dpiz);
            }
            {
                float w1s, dw1s, w1p, dw1p;
                eval_w(3, w1s, dw1s);
                eval_w(4, w1p, dw1p);
                const float pa1_u = fmaf(pa1_ix, ux, fmaf(pa1_iy, uy, pa1_iz * uz));
                const float pb1_u = fmaf(pb1_jx, ux, fmaf(pb1_jy, uy, pb1_jz * uz));
                const float p1_dot = fmaf(pa1_ix, pjx, fmaf(pa1_iy, pjy, pa1_iz * pjz)) +
                                     fmaf(pb1_jx, pix, fmaf(pb1_jy, piy, pb1_jz * piz));
                er_lane = fmaf(dw1s, fmaf(pa1_u, pju, pb1_u * piu), fmaf(dw1p, p1_dot, er_lane));
                const float sf_a = w1s * pju, sf_b = w1s * pa1_u;
                const float sr_a = w1s * piu, sr_b = w1s * pb1_u;
                dux = fmaf(sf_a, pa1_ix, fmaf(sf_b, pjx, fmaf(sr_a, pb1_jx, fmaf(sr_b, pix, dux))));
                duy = fmaf(sf_a, pa1_iy, fmaf(sf_b, pjy, fmaf(sr_a, pb1_jy, fmaf(sr_b, piy, duy))));
                duz = fmaf(sf_a, pa1_iz, fmaf(sf_b, pjz, fmaf(sr_a, pb1_jz, fmaf(sr_b, piz, duz))));
                const float peff_x = fmaf(c_i, pjx, pb1_jx);
                const float peff_y = fmaf(c_i, pjy, pb1_jy);
                const float peff_z = fmaf(c_i, pjz, pb1_jz);
                const float peff_u = fmaf(c_i, pju, pb1_u);
                dpix = fmaf(w1s * peff_u, ux, fmaf(w1p, peff_x, dpix));
                dpiy = fmaf(w1s * peff_u, uy, fmaf(w1p, peff_y, dpiy));
                dpiz = fmaf(w1s * peff_u, uz, fmaf(w1p, peff_z, dpiz));
                if constexpr (BO_MODE == 1) {
                    const float p_dot = fmaf(pix, pjx, fmaf(piy, pjy, piz * pjz));
                    eval_bo_mlp_h16_smem<NB2_4, true, true>(
                        pair, z_i, S, c, qmask, cfg, s_Wxf4_16, s_b4_16, s_v4_16, s_Wr4_16, s_Wxb4_16,
                        qi, qj, pix, piy, piz, pjx, pjy, pjz, ux, uy, uz, piu, pju, p_dot,
                        g_loc, dg_loc, c_sum,
                        e_bo_acc, er_lane, dqi, dpix, dpiy, dpiz, dux, duy, duz);
                } else if constexpr (BO_MODE == 3) {
                    const float p_dot = fmaf(pix, pjx, fmaf(piy, pjy, piz * pjz));
                    eval_bo_mlp_h8_smem<NB2_4, true, true>(
                        pair, z_i, S, c, qmask, cfg, s_Wxf2_8, s_b2_8, s_v2_8, s_Wr4_8, s_Wxb4_8,
                        qi, qj, pix, piy, piz, pjx, pjy, pjz, ux, uy, uz, piu, pju, p_dot,
                        g_loc, dg_loc, c_sum,
                        e_bo_acc, er_lane, dqi, dpix, dpiy, dpiz, dux, duy, duz);
                } else if constexpr (BO_MODE == 2) {
                    const float p_dot = fmaf(pix, pjx, fmaf(piy, pjy, piz * pjz));
                    eval_bo_mlp_4lane_generic<NB2_4, true, true>(
                        pair, z_i, S, c, qmask, cfg, bo,
                        qi, qj, pix, piy, piz, pjx, pjy, pjz, ux, uy, uz, piu, pju, p_dot,
                        g_loc, dg_loc, c_sum,
                        e_bo_acc, er_lane, dqi, dpix, dpiy, dpiz, dux, duy, duz);
                }
            }
            if constexpr (HAS_C2) {
                const float4 Qj03 = __ldg(buf.Q_pack03 + (int64_t)j * 4 + c);
                const float Qj0 = Qj03.x, Qj1 = Qj03.y, Qj2 = Qj03.z, Qj3 = Qj03.w;
                const float Qj4 = __ldg(buf.Q_pack4 + (int64_t)j * 4 + c);
                float muix, muiy, muiz, s2i, mujx, mujy, mujz, s2j;
                l2_matvec(Qi0, Qi1, Qi2, Qi3, Qi4, ux, uy, uz, muix, muiy, muiz, s2i);
                l2_matvec(Qj0, Qj1, Qj2, Qj3, Qj4, ux, uy, uz, mujx, mujy, mujz, s2j);

                float g_s2i = 0.0f, g_muix = 0.0f, g_muiy = 0.0f, g_muiz = 0.0f;
                float g_s2j = 0.0f, g_mujx = 0.0f, g_mujy = 0.0f, g_mujz = 0.0f;
                {
                    float w02f, dw02f, w02r, dw02r;
                    eval_w(6, w02f, dw02f); eval_w(7, w02r, dw02r);
                    const float cf_qi = c_i * qi, cr_qj = c_j * qj;
                    er_lane = fmaf(dw02f, cf_qi * s2j, fmaf(dw02r, cr_qj * s2i, er_lane));
                    dqi = fmaf(c_i * w02f, s2j, dqi);
                    g_s2j = fmaf(w02f, cf_qi, g_s2j);
                    g_s2i = fmaf(w02r, cr_qj, g_s2i);
                }
                {
                    float w12sf, dw12sf, w12sr, dw12sr, w12pf, dw12pf, w12pr, dw12pr;
                    eval_w(8, w12sf, dw12sf); eval_w(9, w12sr, dw12sr);
                    eval_w(10, w12pf, dw12pf); eval_w(11, w12pr, dw12pr);
                    const float pi_muj = fmaf(pix, mujx, fmaf(piy, mujy, piz * mujz));
                    const float pj_mui = fmaf(pjx, muix, fmaf(pjy, muiy, pjz * muiz));
                    const float cw12sf = c_i * w12sf, cw12pf = c_i * w12pf;
                    const float cw12sr = c_j * w12sr, cw12pr = c_j * w12pr;
                    er_lane = fmaf(c_i * dw12sf, piu * s2j, fmaf(c_i * dw12pf, pi_muj,
                              fmaf(-c_j * dw12sr, pju * s2i, fmaf(-c_j * dw12pr, pj_mui, er_lane))));
                    const float sf12 = cw12sf * s2j, sr12 = cw12sr * s2i;
                    dux = fmaf(sf12, pix, fmaf(-sr12, pjx, dux));
                    duy = fmaf(sf12, piy, fmaf(-sr12, pjy, duy));
                    duz = fmaf(sf12, piz, fmaf(-sr12, pjz, duz));
                    dpix = fmaf(sf12, ux, fmaf(cw12pf, mujx, dpix));
                    dpiy = fmaf(sf12, uy, fmaf(cw12pf, mujy, dpiy));
                    dpiz = fmaf(sf12, uz, fmaf(cw12pf, mujz, dpiz));
                    g_s2j = fmaf(cw12sf, piu, g_s2j);
                    g_mujx = fmaf(cw12pf, pix, g_mujx); g_mujy = fmaf(cw12pf, piy, g_mujy); g_mujz = fmaf(cw12pf, piz, g_mujz);
                    g_s2i = fmaf(-cw12sr, pju, g_s2i);
                    g_muix = fmaf(-cw12pr, pjx, g_muix); g_muiy = fmaf(-cw12pr, pjy, g_muiy); g_muiz = fmaf(-cw12pr, pjz, g_muiz);
                }
                {
                    float w2s, dw2s, w2p, dw2p, w2d, dw2d;
                    eval_w(12, w2s, dw2s); eval_w(13, w2p, dw2p); eval_w(14, w2d, dw2d);
                    const float mu_dot = fmaf(muix, mujx, fmaf(muiy, mujy, muiz * mujz));
                    const float Q_dot = fmaf(Qi0, Qj0, fmaf(Qi1, Qj1, fmaf(Qi2, Qj2, fmaf(Qi3, Qj3, Qi4 * Qj4))));
                    const float cw2s = c_sum * w2s, cw2p = c_sum * w2p, cw2d = c_sum * w2d;
                    er_lane = fmaf(c_sum * dw2s, s2i * s2j, fmaf(c_sum * dw2p, mu_dot, fmaf(c_sum * dw2d, Q_dot, er_lane)));
                    g_s2i = fmaf(cw2s, s2j, g_s2i);
                    g_muix = fmaf(cw2p, mujx, g_muix); g_muiy = fmaf(cw2p, mujy, g_muiy); g_muiz = fmaf(cw2p, mujz, g_muiz);
                    g_s2j = fmaf(cw2s, s2i, g_s2j);
                    g_mujx = fmaf(cw2p, muix, g_mujx); g_mujy = fmaf(cw2p, muiy, g_mujy); g_mujz = fmaf(cw2p, muiz, g_mujz);

                    float tux, tuy, tuz;
                    l2_matvec_only(Qi0, Qi1, Qi2, Qi3, Qi4, g_muix, g_muiy, g_muiz, tux, tuy, tuz);
                    dux = fmaf(2.0f * g_s2i, muix, dux + tux);
                    duy = fmaf(2.0f * g_s2i, muiy, duy + tuy);
                    duz = fmaf(2.0f * g_s2i, muiz, duz + tuz);
                    l2_matvec_only(Qj0, Qj1, Qj2, Qj3, Qj4, g_mujx, g_mujy, g_mujz, tux, tuy, tuz);
                    dux = fmaf(2.0f * g_s2j, mujx, dux + tux);
                    duy = fmaf(2.0f * g_s2j, mujy, duy + tuy);
                    duz = fmaf(2.0f * g_s2j, mujz, duz + tuz);

                    dQi0 = fmaf(cw2d, Qj0, dQi0); dQi1 = fmaf(cw2d, Qj1, dQi1); dQi2 = fmaf(cw2d, Qj2, dQi2);
                    dQi3 = fmaf(cw2d, Qj3, dQi3); dQi4 = fmaf(cw2d, Qj4, dQi4);
                    l2_dq_acc(g_muix + g_s2i * ux, g_muiy + g_s2i * uy, g_muiz + g_s2i * uz,
                              ux, uy, uz, dQi0, dQi1, dQi2, dQi3, dQi4);
                }
            }

            if constexpr (STRESS) {
                const bool p0 = (c & 1) != 0;
                const float r0 = (p0 ? dux : er_lane) + __shfl_xor_sync(qmask, p0 ? er_lane : dux, 1);
                const float r1 = (p0 ? duz : duy)     + __shfl_xor_sync(qmask, p0 ? duy : duz, 1);
                const bool p1 = (c & 2) != 0;
                const float rv = (p1 ? r1 : r0) + __shfl_xor_sync(qmask, p1 ? r0 : r1, 2);
                const float er_t  = __shfl_sync(qmask, rv, 0, 4);
                const float dux_t = __shfl_sync(qmask, rv, 1, 4);
                const float duy_t = __shfl_sync(qmask, rv, 2, 4);
                const float duz_t = __shfl_sync(qmask, rv, 3, 4);
                if (c == 0) {
                    const float udot = fmaf(ux, dux_t, fmaf(uy, duy_t, uz * duz_t));
                    const float gx = fmaf(er_t, ux, (dux_t - udot * ux) * inv_r);
                    const float gy = fmaf(er_t, uy, (duy_t - udot * uy) * inv_r);
                    const float gz = fmaf(er_t, uz, (duz_t - udot * uz) * inv_r);
                    rfx_d -= (double)gx; rfy_d -= (double)gy; rfz_d -= (double)gz;
                    const float4 vl = __ldg(vec_len + eb_b + bi);
                    v0 += 0.5 * (double)vl.x * (double)gx;
                    v1 += 0.25 * ((double)vl.x * (double)gy + (double)vl.y * (double)gx);
                    v2 += 0.25 * ((double)vl.x * (double)gz + (double)vl.z * (double)gx);
                    v3 += 0.5 * (double)vl.y * (double)gy;
                    v4 += 0.25 * ((double)vl.y * (double)gz + (double)vl.z * (double)gy);
                    v5 += 0.5 * (double)vl.z * (double)gz;
                }
            } else {
                // Linear in (er_lane, dux, duy, duz) -> ZERO SHFL.SYNC per edge! Reduce once after loop!
                const float udot = fmaf(ux, dux, fmaf(uy, duy, uz * duz));
                rfx_f -= fmaf(er_lane, ux, (dux - udot * ux) * inv_r);
                rfy_f -= fmaf(er_lane, uy, (duy - udot * uy) * inv_r);
                rfz_f -= fmaf(er_lane, uz, (duz - udot * uz) * inv_r);
            }
        }
        __syncwarp(qmask);
    }

    buf.dqp_pack4[(int64_t)i * 4 + c] = make_float4(dqi, dpix, dpiy, dpiz);
    if constexpr (HAS_C2) {
        buf.dQ_pack03[(int64_t)i * 4 + c] = make_float4(dQi0, dQi1, dQi2, dQi3);
        buf.dQ_pack4[(int64_t)i * 4 + c] = dQi4;
    }
    if constexpr (!STRESS) {
        rfx_f = q4_sum_f(rfx_f, qmask);
        rfy_f = q4_sum_f(rfy_f, qmask);
        rfz_f = q4_sum_f(rfz_f, qmask);
    }
    if (c == 0) {
        if constexpr (STRESS) {
            if (forces_f32) {
                forces_f32[(int64_t)i * 3 + 0] += (float)rfx_d;
                forces_f32[(int64_t)i * 3 + 1] += (float)rfy_d;
                forces_f32[(int64_t)i * 3 + 2] += (float)rfz_d;
            } else {
                forces[(int64_t)i * 3 + 0] += rfx_d;
                forces[(int64_t)i * 3 + 1] += rfy_d;
                forces[(int64_t)i * 3 + 2] += rfz_d;
            }
            atomicAdd(virial + 0, v0); atomicAdd(virial + 1, v1); atomicAdd(virial + 2, v2);
            atomicAdd(virial + 3, v3); atomicAdd(virial + 4, v4); atomicAdd(virial + 5, v5);
        } else {
            if (forces_f32) {
                forces_f32[(int64_t)i * 3 + 0] += rfx_f;
                forces_f32[(int64_t)i * 3 + 1] += rfy_f;
                forces_f32[(int64_t)i * 3 + 2] += rfz_f;
            } else {
                forces[(int64_t)i * 3 + 0] += (double)rfx_f;
                forces[(int64_t)i * 3 + 1] += (double)rfy_f;
                forces[(int64_t)i * 3 + 2] += (double)rfz_f;
            }
        }
    }
    if constexpr (BO_MODE != 0) {
        const float e_bo_i = cfg.half_esc * q4_sum_f(e_bo_acc, qmask);
        if (c == 0) {
            const double sw_i = e_zbl ? (double)buf.sw[i] : 1.0;
            const double de = sw_i * (double)e_bo_i;
            if (atom_energy_out) atom_energy_out[i] += de;
            e_bo_warp = de;
            if (w_zbl_io && e_zbl) {
                const double xc = e_zbl[i] * (double)cfg.inv_zc;
                if (xc > 0.0 && xc < 1.0) {
                    const double xc2 = xc * xc;
                    const double dsw_dx = -30.0 * xc2 + 60.0 * xc2 * xc - 30.0 * xc2 * xc2;
                    w_zbl_io[i] += (float)(dsw_dx * (double)cfg.inv_zc * (double)e_bo_i);
                }
            }
        }
    }
    } // end if (i < N)
    if constexpr (BO_MODE != 0) {
        const double wsum = warp_sum_d(e_bo_warp);
        if ((threadIdx.x & 31) == 0 && energy_out && wsum != 0.0) atomicAdd(energy_out, wsum);
    }
}

// =================================================================================================
// 5b) v31/v37 Edge-Parallel 16-Lane (4 quads x 4 channels, 8 atoms/CTA) LRSK Pair Kernels (BO_MODE==0)
//     - Supports BOTH HAS_C2=false (6 radial slots) and HAS_C2=true (15 radial slots, skips c>=c2!)
//     - Bank-conflict-free s_g/s_dg (+4 padding), 128B single-line qp_dpol_pack8 reads, STACK:0!
// =================================================================================================
template <int NB2, bool PACK_VS, bool HAS_C2 = false>
__global__ void __launch_bounds__(128, HAS_C2 ? 4 : (NB2 >= 8 ? 5 : 6)) lrsk_edge_sym_pol_fwd_q16_kernel(
    int N, int S, Cfg cfg, ChebParamF r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_fast4,
    Bufs buf) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int N_T4 = HAS_C2 ? 15 : 6;
    constexpr int EB = 16;
    constexpr int PAIR_W4 = NB2_4 * N_T4 * 4;
    constexpr int PAIR_STRIDE_W4 = PAIR_W4 + 4;  // +16 banks -> pair 0 & 1 occupy disjoint halves (0 conflicts on S=2!)

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[NB2_4][8][EB + 4];     // SoA +16 banks -> ZERO bank conflicts for NB2=4,8,12!
    __shared__ float4 s_u[8][EB + 4];            // +16 banks -> aid=0 & aid=1 disjoint halves!
    __shared__ int2   s_jp[8][EB + 4];           // +8 banks -> aid=0 & aid=1 disjoint!

    const int tot_w4 = S * S * PAIR_W4;
    for (int idx = threadIdx.x; idx < tot_w4; idx += 128) {
        const int p = idx / PAIR_W4;
        const int off = idx - p * PAIR_W4;
        s_W4[p * PAIR_STRIDE_W4 + off] = __ldg(W_fast4 + idx);
    }
    __syncthreads();

    const int aid    = threadIdx.x >> 4;
    const int lane16 = threadIdx.x & 15;
    const int q      = lane16 >> 2;
    const int c      = lane16 & 3;
    const unsigned hmask = (aid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + aid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float4 qpi = __ldg(buf.qp_pack4 + (int64_t)i * 4 + c);
    const float qi = qpi.x, pix = qpi.y, piy = qpi.z, piz = qpi.w;
    float Qi0 = 0.0f, Qi1 = 0.0f, Qi2 = 0.0f, Qi3 = 0.0f, Qi4 = 0.0f;
    if constexpr (HAS_C2) {
        if (c < cfg.c2) {
            const float4 Q03 = __ldg(buf.Q_pack03 + (int64_t)i * 4 + c);
            Qi0 = Q03.x; Qi1 = Q03.y; Qi2 = Q03.z; Qi3 = Q03.w;
            Qi4 = __ldg(buf.Q_pack4 + (int64_t)i * 4 + c);
        }
    }

    float e_lr_acc = 0.0f, phi_acc = 0.0f, Efx = 0.0f, Efy = 0.0f, Efz = 0.0f;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (lane16 < bcnt) {
            const int e = eb_b + lane16;
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r, inv_r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                const float r2 = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                inv_r = rsqrtf(fmaxf(r2, 1e-16f));
                r = r2 * inv_r;
            } else {
                j = __ldg(senders + e);
                r = fmaxf(vl.w, 1e-8f);
                inv_r = 1.0f / r;
            }
            s_jp[aid][lane16] = make_int2(j, z_i * S + __ldg(species + j));
            s_u[aid][lane16]  = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            float4 g_tmp[NB2_4];
            eval_cheb_store<NB2, false>(r, r_max2, g_tmp, nullptr);
            #pragma unroll
            for (int b = 0; b < NB2_4; ++b) s_g[b][aid][lane16] = g_tmp[b];
        }
        __syncwarp(hmask);

        for (int e_idx = q; e_idx < bcnt; e_idx += 4) {
            const int2 jp = s_jp[aid][e_idx];
            const int j = jp.x, pair = jp.y;
            const float4 u4 = s_u[aid][e_idx];
            const float ux = u4.x, uy = u4.y, uz = u4.z;
            const float4* Wp = s_W4 + (int64_t)pair * PAIR_STRIDE_W4 + c;

            float w2b = 0.0f, w0 = 0.0f, w01f = 0.0f, w1s = 0.0f, w1p = 0.0f;
            if constexpr (!HAS_C2) {
                // Single-pass-per-quad radial contraction: 1x LDS.128 per b, 0 bank conflicts, STACK:0!
                #pragma unroll
                for (int b = 0; b < NB2_4; ++b) {
                    const float4 gb = s_g[b][aid][e_idx];
                    const float4* Wb = Wp + (b * N_T4) * 4;
                    if (c == 0) w2b += v1_dot4(Wb[5 * 4], gb);
                    w0   += v1_dot4(Wb[0 * 4], gb);
                    w01f += v1_dot4(Wb[1 * 4], gb);
                    w1s  += v1_dot4(Wb[3 * 4], gb);
                    w1p  += v1_dot4(Wb[4 * 4], gb);
                }
            } else {
                auto eval_wf = [&](int slot) -> float {
                    float w = v1_dot4(Wp[slot * 4], s_g[0][aid][e_idx]);
                    #pragma unroll
                    for (int b = 1; b < NB2_4; ++b) w += v1_dot4(Wp[(b * N_T4 + slot) * 4], s_g[b][aid][e_idx]);
                    return w;
                };
                if (c == 0) w2b = eval_wf(5);
                w0   = eval_wf(0);
                w01f = eval_wf(1);
                w1s  = eval_wf(3);
                w1p  = eval_wf(4);
            }

            const float4 qpj = __ldg(buf.qp_pack4 + (int64_t)j * 4 + c);
            const float qj = qpj.x, pjx = qpj.y, pjy = qpj.z, pjz = qpj.w;

            if (c == 0) e_lr_acc += w2b;

            const float t0 = w0 * qj;
            e_lr_acc = fmaf(qi, t0, e_lr_acc);
            phi_acc += t0;

            const float piu = fmaf(pix, ux, fmaf(piy, uy, piz * uz));
            const float pju = fmaf(pjx, ux, fmaf(pjy, uy, pjz * uz));

            const float t01p = w01f * pju;
            e_lr_acc = fmaf(qi, t01p, e_lr_acc);
            phi_acc += t01p;
            const float t01q = w01f * qj;
            Efx = fmaf(t01q, ux, Efx); Efy = fmaf(t01q, uy, Efy); Efz = fmaf(t01q, uz, Efz);

            const float w1s_pju = w1s * pju;
            const float p_dot = fmaf(pix, pjx, fmaf(piy, pjy, piz * pjz));
            e_lr_acc = fmaf(piu, w1s_pju, fmaf(w1p, p_dot, e_lr_acc));
            Efx = fmaf(w1s_pju, ux, fmaf(w1p, pjx, Efx));
            Efy = fmaf(w1s_pju, uy, fmaf(w1p, pjy, Efy));
            Efz = fmaf(w1s_pju, uz, fmaf(w1p, pjz, Efz));

            if constexpr (HAS_C2) {
                if (c < cfg.c2) {
                    auto eval_wf2 = [&](int slot) -> float {
                        float w = v1_dot4(Wp[slot * 4], s_g[0][aid][e_idx]);
                        #pragma unroll
                        for (int b = 1; b < NB2_4; ++b) w += v1_dot4(Wp[(b * N_T4 + slot) * 4], s_g[b][aid][e_idx]);
                        return w;
                    };
                    const float4 Qj03 = __ldg(buf.Q_pack03 + (int64_t)j * 4 + c);
                    const float Qj0 = Qj03.x, Qj1 = Qj03.y, Qj2 = Qj03.z, Qj3 = Qj03.w;
                    const float Qj4 = __ldg(buf.Q_pack4 + (int64_t)j * 4 + c);
                    float muix, muiy, muiz, s2i, mujx, mujy, mujz, s2j;
                    l2_matvec(Qi0, Qi1, Qi2, Qi3, Qi4, ux, uy, uz, muix, muiy, muiz, s2i);
                    l2_matvec(Qj0, Qj1, Qj2, Qj3, Qj4, ux, uy, uz, mujx, mujy, mujz, s2j);
                    e_lr_acc = fmaf(eval_wf2(6), qi * s2j, e_lr_acc);
                    const float pi_muj = fmaf(pix, mujx, fmaf(piy, mujy, piz * mujz));
                    e_lr_acc = fmaf(eval_wf2(8), piu * s2j, fmaf(eval_wf2(10), pi_muj, e_lr_acc));
                    const float mu_dot = fmaf(muix, mujx, fmaf(muiy, mujy, muiz * mujz));
                    const float Q_dot = fmaf(Qi0, Qj0, fmaf(Qi1, Qj1, fmaf(Qi2, Qj2, fmaf(Qi3, Qj3, Qi4 * Qj4))));
                    e_lr_acc = fmaf(eval_wf2(12), s2i * s2j, fmaf(eval_wf2(13), mu_dot, fmaf(eval_wf2(14), Q_dot, e_lr_acc)));
                }
            }
        }
        __syncwarp(hmask);
    }

    const float e_tot = half_warp_sum_f(e_lr_acc, hmask);
    if (lane16 == 0) buf.e_lr[i] = cfg.half_esc * e_tot;
    if (buf.pol_ef4) {
        phi_acc += __shfl_xor_sync(hmask, phi_acc, 4); phi_acc += __shfl_xor_sync(hmask, phi_acc, 8);
        Efx     += __shfl_xor_sync(hmask, Efx,     4); Efx     += __shfl_xor_sync(hmask, Efx,     8);
        Efy     += __shfl_xor_sync(hmask, Efy,     4); Efy     += __shfl_xor_sync(hmask, Efy,     8);
        Efz     += __shfl_xor_sync(hmask, Efz,     4); Efz     += __shfl_xor_sync(hmask, Efz,     8);
        if (q == 0) buf.pol_ef4[(int64_t)i * 4 + c] = make_float4(phi_acc, Efx, Efy, Efz);
    }
}

template <int NB2, bool STRESS, bool PACK_VS, bool HAS_C2 = false>
__global__ void __launch_bounds__(128, HAS_C2 ? 3 : 5) lrsk_edge_sym_pol_bwd_q16_kernel(
    int N, int S, bool has_zbl_sw, Cfg cfg, ChebParamF r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_fast4,
    Bufs buf,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int N_T4 = HAS_C2 ? 15 : 6;
    constexpr int EB = 16;
    constexpr int PAIR_W4 = NB2_4 * N_T4 * 4;
    constexpr int PAIR_STRIDE_W4 = PAIR_W4 + 4;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[NB2_4][8][EB + 4];     // SoA +16 banks -> ZERO bank conflicts for NB2=4,8,12!
    __shared__ float4 s_dg[NB2_4][8][EB + 4];
    __shared__ float4 s_u[8][EB + 4];
    __shared__ int2   s_jp[8][EB + 4];
    __shared__ float  s_swj[8][EB + 4];

    const int tot_w4 = S * S * PAIR_W4;
    for (int idx = threadIdx.x; idx < tot_w4; idx += 128) {
        const int p = idx / PAIR_W4;
        const int off = idx - p * PAIR_W4;
        s_W4[p * PAIR_STRIDE_W4 + off] = __ldg(W_fast4 + idx);
    }
    __syncthreads();

    const int aid    = threadIdx.x >> 4;
    const int lane16 = threadIdx.x & 15;
    const int q      = lane16 >> 2;
    const int c      = lane16 & 3;
    const unsigned hmask = (aid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + aid;
    if (i >= N) return;

    const bool zbl_on = has_zbl_sw && ((buf.zbl_flag == nullptr) || (*buf.zbl_flag != 0));
    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float c_i = zbl_on ? (cfg.half_esc * buf.sw[i]) : cfg.half_esc;
    const float4* qpd_p = buf.qp_dpol_pack8;

    const float4 qpi = qpd_p
        ? __ldg(qpd_p + (int64_t)i * 8 + c)
        : __ldg(buf.qp_pack4 + (int64_t)i * 4 + c);
    const float qi = qpi.x, pix = qpi.y, piy = qpi.z, piz = qpi.w;
    const float4 dpi_pol = qpd_p
        ? __ldg(qpd_p + (int64_t)i * 8 + 4 + c)
        : __ldg(buf.dpol_ef4 + (int64_t)i * 4 + c);
    const float qa0_i = fmaf(c_i, qi, dpi_pol.x);
    const float pa1_ix = fmaf(c_i, pix, dpi_pol.y);
    const float pa1_iy = fmaf(c_i, piy, dpi_pol.z);
    const float pa1_iz = fmaf(c_i, piz, dpi_pol.w);
    float Qi0 = 0.0f, Qi1 = 0.0f, Qi2 = 0.0f, Qi3 = 0.0f, Qi4 = 0.0f;
    if constexpr (HAS_C2) {
        if (c < cfg.c2) {
            const float4 Q03 = __ldg(buf.Q_pack03 + (int64_t)i * 4 + c);
            Qi0 = Q03.x; Qi1 = Q03.y; Qi2 = Q03.z; Qi3 = Q03.w;
            Qi4 = __ldg(buf.Q_pack4 + (int64_t)i * 4 + c);
        }
    }

    const float4 seed_dqp = (q == 0) ? __ldg(buf.dqp_pack4 + (int64_t)i * 4 + c) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float dqi = 0.0f, dpix = seed_dqp.y, dpiy = seed_dqp.z, dpiz = seed_dqp.w;
    float dQi0 = 0.0f, dQi1 = 0.0f, dQi2 = 0.0f, dQi3 = 0.0f, dQi4 = 0.0f;
    float rfx_f = 0.0f, rfy_f = 0.0f, rfz_f = 0.0f;
    double rfx_d = 0.0, rfy_d = 0.0, rfz_d = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (lane16 < bcnt) {
            const int e = eb_b + lane16;
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r, inv_r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                const float r2 = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                inv_r = rsqrtf(fmaxf(r2, 1e-16f));
                r = r2 * inv_r;
            } else {
                j = __ldg(senders + e);
                r = fmaxf(vl.w, 1e-8f);
                inv_r = 1.0f / r;
            }
            const int z_j = __ldg(species + j);
            s_jp[aid][lane16]  = make_int2(j, z_i * S + z_j);
            s_u[aid][lane16]   = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            if (zbl_on) s_swj[aid][lane16] = __ldg(buf.sw + j);
            float4 g_tmp[NB2_4], dg_tmp[NB2_4];
            eval_cheb_store<NB2, true>(r, r_max2, g_tmp, dg_tmp);
            #pragma unroll
            for (int b = 0; b < NB2_4; ++b) {
                s_g[b][aid][lane16]  = g_tmp[b];
                s_dg[b][aid][lane16] = dg_tmp[b];
            }
        }
        __syncwarp(hmask);

        for (int e_idx = q; e_idx < bcnt; e_idx += 4) {
            const int2 jp = s_jp[aid][e_idx];
            const int j = jp.x, pair = jp.y;
            const float4 u4 = s_u[aid][e_idx];
            const float ux = u4.x, uy = u4.y, uz = u4.z, inv_r = u4.w;
            const float c_j = zbl_on ? (cfg.half_esc * s_swj[aid][e_idx]) : cfg.half_esc;
            const float c_sum = c_i + c_j;
            const float4* Wp  = s_W4 + (int64_t)pair * PAIR_STRIDE_W4 + c;

            float dw2b = 0.0f;
            float w0 = 0.0f, dw0 = 0.0f;
            float w01f = 0.0f, dw01f = 0.0f, w01r = 0.0f, dw01r = 0.0f;
            float w1s = 0.0f, dw1s = 0.0f, w1p = 0.0f, dw1p = 0.0f;
            auto eval_w = [&](int slot, float& w, float& dw) {
                const float4 wv0 = Wp[slot * 4];
                w  = v1_dot4(wv0, s_g[0][aid][e_idx]);
                dw = v1_dot4(wv0, s_dg[0][aid][e_idx]);
                #pragma unroll
                for (int b = 1; b < NB2_4; ++b) {
                    const float4 wv = Wp[(b * N_T4 + slot) * 4];
                    w  += v1_dot4(wv, s_g[b][aid][e_idx]);
                    dw += v1_dot4(wv, s_dg[b][aid][e_idx]);
                }
            };
            if constexpr (!HAS_C2) {
                // Single-pass-per-quad radial contraction: 1x LDS.128 per b for g & dg (7x fewer SMEM reads on NB2=8!)
                #pragma unroll
                for (int b = 0; b < NB2_4; ++b) {
                    const float4 gb  = s_g[b][aid][e_idx];
                    const float4 dgb = s_dg[b][aid][e_idx];
                    const float4* Wb = Wp + (b * N_T4) * 4;
                    if (c == 0) dw2b += v1_dot4(Wb[5 * 4], dgb);
                    { const float4 wv = Wb[0 * 4]; w0   += v1_dot4(wv, gb); dw0   += v1_dot4(wv, dgb); }
                    { const float4 wv = Wb[1 * 4]; w01f += v1_dot4(wv, gb); dw01f += v1_dot4(wv, dgb); }
                    { const float4 wv = Wb[2 * 4]; w01r += v1_dot4(wv, gb); dw01r += v1_dot4(wv, dgb); }
                    { const float4 wv = Wb[3 * 4]; w1s  += v1_dot4(wv, gb); dw1s  += v1_dot4(wv, dgb); }
                    { const float4 wv = Wb[4 * 4]; w1p  += v1_dot4(wv, gb); dw1p  += v1_dot4(wv, dgb); }
                }
            } else {
                float w2b_unused;
                if (c == 0) eval_w(5, w2b_unused, dw2b);
                eval_w(0, w0, dw0);
                eval_w(1, w01f, dw01f);
                eval_w(2, w01r, dw01r);
                eval_w(3, w1s, dw1s);
                eval_w(4, w1p, dw1p);
            }

            const float4 qpj = qpd_p
                ? __ldg(qpd_p + (int64_t)j * 8 + c)
                : __ldg(buf.qp_pack4 + (int64_t)j * 4 + c);
            const float qj = qpj.x, pjx = qpj.y, pjy = qpj.z, pjz = qpj.w;
            const float4 dpj_pol = qpd_p
                ? __ldg(qpd_p + (int64_t)j * 8 + 4 + c)
                : __ldg(buf.dpol_ef4 + (int64_t)j * 4 + c);
            const float dPhi_j = dpj_pol.x, dEfjx = dpj_pol.y, dEfjy = dpj_pol.z, dEfjz = dpj_pol.w;
            const float qb0_j = fmaf(c_j, qj, dPhi_j);
            const float pb1_jx = fmaf(c_j, pjx, dEfjx);
            const float pb1_jy = fmaf(c_j, pjy, dEfjy);
            const float pb1_jz = fmaf(c_j, pjz, dEfjz);

            float er_lane = 0.0f, dux = 0.0f, duy = 0.0f, duz = 0.0f;

            if (c == 0) {
                er_lane = fmaf(c_sum, dw2b, er_lane);
            }
            {
                er_lane = fmaf(dw0, fmaf(qa0_i, qj, qb0_j * qi), er_lane);
                dqi = fmaf(w0, fmaf(c_i, qj, qb0_j), dqi);
            }
            const float piu = fmaf(pix, ux, fmaf(piy, uy, piz * uz));
            const float pju = fmaf(pjx, ux, fmaf(pjy, uy, pjz * uz));
            {
                const float dEfix = fmaf(-c_i, pix, pa1_ix);
                const float dEfiy = fmaf(-c_i, piy, pa1_iy);
                const float dEfiz = fmaf(-c_i, piz, pa1_iz);
                const float defi_u = fmaf(dEfix, ux, fmaf(dEfiy, uy, dEfiz * uz));
                const float defj_u = fmaf(dEfjx, ux, fmaf(dEfjy, uy, dEfjz * uz));
                er_lane = fmaf(dw01f, fmaf(qa0_i, pju, qj * defi_u),
                          fmaf(-dw01r, fmaf(qb0_j, piu, qi * defj_u), er_lane));
                const float sf_a = w01f * qa0_i, sf_b = w01f * qj;
                const float sr_a = w01r * qb0_j, sr_b = w01r * qi;
                dux = fmaf(sf_a, pjx, fmaf(sf_b, dEfix, fmaf(-sr_a, pix, fmaf(-sr_b, dEfjx, dux))));
                duy = fmaf(sf_a, pjy, fmaf(sf_b, dEfiy, fmaf(-sr_a, piy, fmaf(-sr_b, dEfjy, duy))));
                duz = fmaf(sf_a, pjz, fmaf(sf_b, dEfiz, fmaf(-sr_a, piz, fmaf(-sr_b, dEfjz, duz))));
                dqi  = fmaf(c_i * w01f, pju, fmaf(-w01r, defj_u, dqi));
                dpix = fmaf(-sr_a, ux, dpix); dpiy = fmaf(-sr_a, uy, dpiy); dpiz = fmaf(-sr_a, uz, dpiz);
            }
            {
                const float pa1_u = fmaf(pa1_ix, ux, fmaf(pa1_iy, uy, pa1_iz * uz));
                const float pb1_u = fmaf(pb1_jx, ux, fmaf(pb1_jy, uy, pb1_jz * uz));
                const float p1_dot = fmaf(pa1_ix, pjx, fmaf(pa1_iy, pjy, pa1_iz * pjz)) +
                                     fmaf(pb1_jx, pix, fmaf(pb1_jy, piy, pb1_jz * piz));
                er_lane = fmaf(dw1s, fmaf(pa1_u, pju, pb1_u * piu), fmaf(dw1p, p1_dot, er_lane));
                const float sf_a = w1s * pju, sf_b = w1s * pa1_u;
                const float sr_a = w1s * piu, sr_b = w1s * pb1_u;
                dux = fmaf(sf_a, pa1_ix, fmaf(sf_b, pjx, fmaf(sr_a, pb1_jx, fmaf(sr_b, pix, dux))));
                duy = fmaf(sf_a, pa1_iy, fmaf(sf_b, pjy, fmaf(sr_a, pb1_jy, fmaf(sr_b, piy, duy))));
                duz = fmaf(sf_a, pa1_iz, fmaf(sf_b, pjz, fmaf(sr_a, pb1_jz, fmaf(sr_b, piz, duz))));
                const float peff_x = fmaf(c_i, pjx, pb1_jx);
                const float peff_y = fmaf(c_i, pjy, pb1_jy);
                const float peff_z = fmaf(c_i, pjz, pb1_jz);
                const float peff_u = fmaf(c_i, pju, pb1_u);
                dpix = fmaf(w1s * peff_u, ux, fmaf(w1p, peff_x, dpix));
                dpiy = fmaf(w1s * peff_u, uy, fmaf(w1p, peff_y, dpiy));
                dpiz = fmaf(w1s * peff_u, uz, fmaf(w1p, peff_z, dpiz));
            }
            if constexpr (HAS_C2) {
                if (c < cfg.c2) {
                    const float4 Qj03 = __ldg(buf.Q_pack03 + (int64_t)j * 4 + c);
                    const float Qj0 = Qj03.x, Qj1 = Qj03.y, Qj2 = Qj03.z, Qj3 = Qj03.w;
                    const float Qj4 = __ldg(buf.Q_pack4 + (int64_t)j * 4 + c);
                    float muix, muiy, muiz, s2i, mujx, mujy, mujz, s2j;
                    l2_matvec(Qi0, Qi1, Qi2, Qi3, Qi4, ux, uy, uz, muix, muiy, muiz, s2i);
                    l2_matvec(Qj0, Qj1, Qj2, Qj3, Qj4, ux, uy, uz, mujx, mujy, mujz, s2j);

                    float g_s2i = 0.0f, g_muix = 0.0f, g_muiy = 0.0f, g_muiz = 0.0f;
                    float g_s2j = 0.0f, g_mujx = 0.0f, g_mujy = 0.0f, g_mujz = 0.0f;
                    {
                        float w02f, dw02f, w02r, dw02r;
                        eval_w(6, w02f, dw02f); eval_w(7, w02r, dw02r);
                        const float cf_qi = c_i * qi, cr_qj = c_j * qj;
                        er_lane = fmaf(dw02f, cf_qi * s2j, fmaf(dw02r, cr_qj * s2i, er_lane));
                        dqi = fmaf(c_i * w02f, s2j, dqi);
                        g_s2j = fmaf(w02f, cf_qi, g_s2j);
                        g_s2i = fmaf(w02r, cr_qj, g_s2i);
                    }
                    {
                        float w12sf, dw12sf, w12sr, dw12sr, w12pf, dw12pf, w12pr, dw12pr;
                        eval_w(8, w12sf, dw12sf); eval_w(9, w12sr, dw12sr);
                        eval_w(10, w12pf, dw12pf); eval_w(11, w12pr, dw12pr);
                        const float pi_muj = fmaf(pix, mujx, fmaf(piy, mujy, piz * mujz));
                        const float pj_mui = fmaf(pjx, muix, fmaf(pjy, muiy, pjz * muiz));
                        const float cw12sf = c_i * w12sf, cw12pf = c_i * w12pf;
                        const float cw12sr = c_j * w12sr, cw12pr = c_j * w12pr;
                        er_lane = fmaf(c_i * dw12sf, piu * s2j, fmaf(c_i * dw12pf, pi_muj,
                                  fmaf(-c_j * dw12sr, pju * s2i, fmaf(-c_j * dw12pr, pj_mui, er_lane))));
                        const float sf12 = cw12sf * s2j, sr12 = cw12sr * s2i;
                        dux = fmaf(sf12, pix, fmaf(-sr12, pjx, dux));
                        duy = fmaf(sf12, piy, fmaf(-sr12, pjy, duy));
                        duz = fmaf(sf12, piz, fmaf(-sr12, pjz, duz));
                        dpix = fmaf(sf12, ux, fmaf(cw12pf, mujx, dpix));
                        dpiy = fmaf(sf12, uy, fmaf(cw12pf, mujy, dpiy));
                        dpiz = fmaf(sf12, uz, fmaf(cw12pf, mujz, dpiz));
                        g_s2j = fmaf(cw12sf, piu, g_s2j);
                        g_mujx = fmaf(cw12pf, pix, g_mujx); g_mujy = fmaf(cw12pf, piy, g_mujy); g_mujz = fmaf(cw12pf, piz, g_mujz);
                        g_s2i = fmaf(-cw12sr, pju, g_s2i);
                        g_muix = fmaf(-cw12pr, pjx, g_muix); g_muiy = fmaf(-cw12pr, pjy, g_muiy); g_muiz = fmaf(-cw12pr, pjz, g_muiz);
                    }
                    {
                        float w2s, dw2s, w2p, dw2p, w2d, dw2d;
                        eval_w(12, w2s, dw2s); eval_w(13, w2p, dw2p); eval_w(14, w2d, dw2d);
                        const float mu_dot = fmaf(muix, mujx, fmaf(muiy, mujy, muiz * mujz));
                        const float Q_dot = fmaf(Qi0, Qj0, fmaf(Qi1, Qj1, fmaf(Qi2, Qj2, fmaf(Qi3, Qj3, Qi4 * Qj4))));
                        const float cw2s = c_sum * w2s, cw2p = c_sum * w2p, cw2d = c_sum * w2d;
                        er_lane = fmaf(c_sum * dw2s, s2i * s2j, fmaf(c_sum * dw2p, mu_dot, fmaf(c_sum * dw2d, Q_dot, er_lane)));
                        g_s2i = fmaf(cw2s, s2j, g_s2i);
                        g_muix = fmaf(cw2p, mujx, g_muix); g_muiy = fmaf(cw2p, mujy, g_muiy); g_muiz = fmaf(cw2p, mujz, g_muiz);
                        g_s2j = fmaf(cw2s, s2i, g_s2j);
                        g_mujx = fmaf(cw2p, muix, g_mujx); g_mujy = fmaf(cw2p, muiy, g_mujy); g_mujz = fmaf(cw2p, muiz, g_mujz);

                        float tux, tuy, tuz;
                        l2_matvec_only(Qi0, Qi1, Qi2, Qi3, Qi4, g_muix, g_muiy, g_muiz, tux, tuy, tuz);
                        dux = fmaf(2.0f * g_s2i, muix, dux + tux);
                        duy = fmaf(2.0f * g_s2i, muiy, duy + tuy);
                        duz = fmaf(2.0f * g_s2i, muiz, duz + tuz);
                        l2_matvec_only(Qj0, Qj1, Qj2, Qj3, Qj4, g_mujx, g_mujy, g_mujz, tux, tuy, tuz);
                        dux = fmaf(2.0f * g_s2j, mujx, dux + tux);
                        duy = fmaf(2.0f * g_s2j, mujy, duy + tuy);
                        duz = fmaf(2.0f * g_s2j, mujz, duz + tuz);

                        dQi0 = fmaf(cw2d, Qj0, dQi0); dQi1 = fmaf(cw2d, Qj1, dQi1); dQi2 = fmaf(cw2d, Qj2, dQi2);
                        dQi3 = fmaf(cw2d, Qj3, dQi3); dQi4 = fmaf(cw2d, Qj4, dQi4);
                        l2_dq_acc(g_muix + g_s2i * ux, g_muiy + g_s2i * uy, g_muiz + g_s2i * uz,
                                  ux, uy, uz, dQi0, dQi1, dQi2, dQi3, dQi4);
                    }
                }
            }

            // ZERO SHFL.SYNC per edge: exact linear force & virial accumulation across (q, c)!
            const float udot = fmaf(ux, dux, fmaf(uy, duy, uz * duz));
            const float gx = fmaf(er_lane, ux, (dux - udot * ux) * inv_r);
            const float gy = fmaf(er_lane, uy, (duy - udot * uy) * inv_r);
            const float gz = fmaf(er_lane, uz, (duz - udot * uz) * inv_r);
            if constexpr (STRESS) {
                rfx_d -= (double)gx; rfy_d -= (double)gy; rfz_d -= (double)gz;
                const float4 vl = __ldg(vec_len + eb_b + e_idx);
                v0 += 0.5 * (double)vl.x * (double)gx;
                v1 += 0.25 * ((double)vl.x * (double)gy + (double)vl.y * (double)gx);
                v2 += 0.25 * ((double)vl.x * (double)gz + (double)vl.z * (double)gx);
                v3 += 0.5 * (double)vl.y * (double)gy;
                v4 += 0.25 * ((double)vl.y * (double)gz + (double)vl.z * (double)gy);
                v5 += 0.5 * (double)vl.z * (double)gz;
            } else {
                rfx_f -= gx; rfy_f -= gy; rfz_f -= gz;
            }
        }
        __syncwarp(hmask);
    }

    // Reduce dqp (and dQ when HAS_C2) across the 4 quads (q=0..3) keeping channel c intact
    dqi  += __shfl_xor_sync(hmask, dqi,  4); dqi  += __shfl_xor_sync(hmask, dqi,  8);
    dpix += __shfl_xor_sync(hmask, dpix, 4); dpix += __shfl_xor_sync(hmask, dpix, 8);
    dpiy += __shfl_xor_sync(hmask, dpiy, 4); dpiy += __shfl_xor_sync(hmask, dpiy, 8);
    dpiz += __shfl_xor_sync(hmask, dpiz, 4); dpiz += __shfl_xor_sync(hmask, dpiz, 8);
    if (q == 0) buf.dqp_pack4[(int64_t)i * 4 + c] = make_float4(dqi, dpix, dpiy, dpiz);
    if constexpr (HAS_C2) {
        dQi0 += __shfl_xor_sync(hmask, dQi0, 4); dQi0 += __shfl_xor_sync(hmask, dQi0, 8);
        dQi1 += __shfl_xor_sync(hmask, dQi1, 4); dQi1 += __shfl_xor_sync(hmask, dQi1, 8);
        dQi2 += __shfl_xor_sync(hmask, dQi2, 4); dQi2 += __shfl_xor_sync(hmask, dQi2, 8);
        dQi3 += __shfl_xor_sync(hmask, dQi3, 4); dQi3 += __shfl_xor_sync(hmask, dQi3, 8);
        dQi4 += __shfl_xor_sync(hmask, dQi4, 4); dQi4 += __shfl_xor_sync(hmask, dQi4, 8);
        if (q == 0) {
            buf.dQ_pack03[(int64_t)i * 4 + c] = make_float4(dQi0, dQi1, dQi2, dQi3);
            buf.dQ_pack4[(int64_t)i * 4 + c]  = dQi4;
        }
    }

    if constexpr (STRESS) {
        rfx_d = half_warp_sum_d(rfx_d, hmask);
        rfy_d = half_warp_sum_d(rfy_d, hmask);
        rfz_d = half_warp_sum_d(rfz_d, hmask);
        v0 = half_warp_sum_d(v0, hmask); v1 = half_warp_sum_d(v1, hmask); v2 = half_warp_sum_d(v2, hmask);
        v3 = half_warp_sum_d(v3, hmask); v4 = half_warp_sum_d(v4, hmask); v5 = half_warp_sum_d(v5, hmask);
        if (lane16 == 0) {
            if (forces_f32) {
                forces_f32[(int64_t)i * 3 + 0] += (float)rfx_d;
                forces_f32[(int64_t)i * 3 + 1] += (float)rfy_d;
                forces_f32[(int64_t)i * 3 + 2] += (float)rfz_d;
            } else {
                forces[(int64_t)i * 3 + 0] += rfx_d;
                forces[(int64_t)i * 3 + 1] += rfy_d;
                forces[(int64_t)i * 3 + 2] += rfz_d;
            }
            atomicAdd(virial + 0, v0); atomicAdd(virial + 1, v1); atomicAdd(virial + 2, v2);
            atomicAdd(virial + 3, v3); atomicAdd(virial + 4, v4); atomicAdd(virial + 5, v5);
        }
    } else {
        rfx_f = half_warp_sum_f(rfx_f, hmask);
        rfy_f = half_warp_sum_f(rfy_f, hmask);
        rfz_f = half_warp_sum_f(rfz_f, hmask);
        if (lane16 == 0) {
            if (forces_f32) {
                forces_f32[(int64_t)i * 3 + 0] += rfx_f;
                forces_f32[(int64_t)i * 3 + 1] += rfy_f;
                forces_f32[(int64_t)i * 3 + 2] += rfz_f;
            } else {
                forces[(int64_t)i * 3 + 0] += (double)rfx_f;
                forces[(int64_t)i * 3 + 1] += (double)rfy_f;
                forces[(int64_t)i * 3 + 2] += (double)rfz_f;
            }
        }
    }
}

// =================================================================================================
// 5b-c2) v41 8-Edge-Parallel 16-Lane (8 pairs x 2 channels, 8 atoms/CTA) LRSK Pair Kernels (c0,c1<=2)
//        Doubles edge throughput (for e_idx = q; e_idx < bcnt; e_idx += 8) -> 2x faster NB2=8!
// =================================================================================================
template <int NB2, bool PACK_VS>
__global__ void __launch_bounds__(128, (NB2 >= 8) ? 5 : 6) lrsk_edge_sym_pol_fwd_q16_c2_kernel(
    int N, int S, Cfg cfg, ChebParamF r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_fast4,
    Bufs buf) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int N_T4 = 6;
    constexpr int EB = 16;
    constexpr int PAIR_W4 = NB2_4 * N_T4 * 4;
    constexpr int PAIR_STRIDE_W4 = PAIR_W4 + 4;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[NB2_4][8][EB + 4];
    __shared__ float4 s_u[8][EB + 4];
    __shared__ int2   s_jp[8][EB + 4];

    const int tot_w4 = S * S * PAIR_W4;
    for (int idx = threadIdx.x; idx < tot_w4; idx += 128) {
        const int p = idx / PAIR_W4;
        const int off = idx - p * PAIR_W4;
        s_W4[p * PAIR_STRIDE_W4 + off] = __ldg(W_fast4 + idx);
    }
    __syncthreads();

    const int aid    = threadIdx.x >> 4;
    const int lane16 = threadIdx.x & 15;
    const int q      = lane16 >> 1;   // 0..7 edges in parallel!
    const int c      = lane16 & 1;    // 0..1 channels!
    const unsigned hmask = (aid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + aid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float4 qpi = __ldg(buf.qp_pack4 + (int64_t)i * 4 + c);
    const float qi = qpi.x, pix = qpi.y, piy = qpi.z, piz = qpi.w;

    float e_lr_acc = 0.0f, phi_acc = 0.0f, Efx = 0.0f, Efy = 0.0f, Efz = 0.0f;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (lane16 < bcnt) {
            const int e = eb_b + lane16;
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r, inv_r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                const float r2 = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                inv_r = rsqrtf(fmaxf(r2, 1e-16f));
                r = r2 * inv_r;
            } else {
                j = __ldg(senders + e);
                r = fmaxf(vl.w, 1e-8f);
                inv_r = 1.0f / r;
            }
            s_jp[aid][lane16] = make_int2(j, z_i * S + __ldg(species + j));
            s_u[aid][lane16]  = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            float4 g_tmp[NB2_4];
            eval_cheb_store<NB2, false>(r, r_max2, g_tmp, nullptr);
            #pragma unroll
            for (int b = 0; b < NB2_4; ++b) s_g[b][aid][lane16] = g_tmp[b];
        }
        __syncwarp(hmask);

        for (int e_idx = q; e_idx < bcnt; e_idx += 8) {
            const int2 jp = s_jp[aid][e_idx];
            const int j = jp.x, pair = jp.y;
            const float4 u4 = s_u[aid][e_idx];
            const float ux = u4.x, uy = u4.y, uz = u4.z;
            const float4* Wp = s_W4 + (int64_t)pair * PAIR_STRIDE_W4 + c;

            float w2b = 0.0f, w0 = 0.0f, w01f = 0.0f, w1s = 0.0f, w1p = 0.0f;
            #pragma unroll
            for (int b = 0; b < NB2_4; ++b) {
                const float4 gb = s_g[b][aid][e_idx];
                const float4* Wb = Wp + (b * N_T4) * 4;
                if (c == 0) w2b += v1_dot4(Wb[5 * 4], gb);
                w0   += v1_dot4(Wb[0 * 4], gb);
                w01f += v1_dot4(Wb[1 * 4], gb);
                w1s  += v1_dot4(Wb[3 * 4], gb);
                w1p  += v1_dot4(Wb[4 * 4], gb);
            }

            const float4 qpj = __ldg(buf.qp_pack4 + (int64_t)j * 4 + c);
            const float qj = qpj.x, pjx = qpj.y, pjy = qpj.z, pjz = qpj.w;

            if (c == 0) e_lr_acc += w2b;

            const float t0 = w0 * qj;
            e_lr_acc = fmaf(qi, t0, e_lr_acc);
            phi_acc += t0;

            const float piu = fmaf(pix, ux, fmaf(piy, uy, piz * uz));
            const float pju = fmaf(pjx, ux, fmaf(pjy, uy, pjz * uz));

            const float t01p = w01f * pju;
            e_lr_acc = fmaf(qi, t01p, e_lr_acc);
            phi_acc += t01p;
            const float t01q = w01f * qj;
            Efx = fmaf(t01q, ux, Efx); Efy = fmaf(t01q, uy, Efy); Efz = fmaf(t01q, uz, Efz);

            const float w1s_pju = w1s * pju;
            const float p_dot = fmaf(pix, pjx, fmaf(piy, pjy, piz * pjz));
            e_lr_acc = fmaf(piu, w1s_pju, fmaf(w1p, p_dot, e_lr_acc));
            Efx = fmaf(w1s_pju, ux, fmaf(w1p, pjx, Efx));
            Efy = fmaf(w1s_pju, uy, fmaf(w1p, pjy, Efy));
            Efz = fmaf(w1s_pju, uz, fmaf(w1p, pjz, Efz));
        }
        __syncwarp(hmask);
    }

    const float e_tot = half_warp_sum_f(e_lr_acc, hmask);
    if (lane16 == 0) buf.e_lr[i] = cfg.half_esc * e_tot;
    if (buf.pol_ef4) {
        phi_acc += __shfl_xor_sync(hmask, phi_acc, 2); phi_acc += __shfl_xor_sync(hmask, phi_acc, 4); phi_acc += __shfl_xor_sync(hmask, phi_acc, 8);
        Efx     += __shfl_xor_sync(hmask, Efx,     2); Efx     += __shfl_xor_sync(hmask, Efx,     4); Efx     += __shfl_xor_sync(hmask, Efx,     8);
        Efy     += __shfl_xor_sync(hmask, Efy,     2); Efy     += __shfl_xor_sync(hmask, Efy,     4); Efy     += __shfl_xor_sync(hmask, Efy,     8);
        Efz     += __shfl_xor_sync(hmask, Efz,     2); Efz     += __shfl_xor_sync(hmask, Efz,     4); Efz     += __shfl_xor_sync(hmask, Efz,     8);
        if (q == 0) buf.pol_ef4[(int64_t)i * 4 + c]     = make_float4(phi_acc, Efx, Efy, Efz);
        if (q == 1) buf.pol_ef4[(int64_t)i * 4 + 2 + c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

template <int NB2, bool STRESS, bool PACK_VS>
__global__ void __launch_bounds__(128, 5) lrsk_edge_sym_pol_bwd_q16_c2_kernel(
    int N, int S, bool has_zbl_sw, Cfg cfg, ChebParamF r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_fast4,
    Bufs buf,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int N_T4 = 6;
    constexpr int EB = 16;
    constexpr int PAIR_W4 = NB2_4 * N_T4 * 4;
    constexpr int PAIR_STRIDE_W4 = PAIR_W4 + 4;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[NB2_4][8][EB + 4];
    __shared__ float4 s_dg[NB2_4][8][EB + 4];
    __shared__ float4 s_u[8][EB + 4];
    __shared__ int2   s_jp[8][EB + 4];
    __shared__ float  s_swj[8][EB + 4];

    const int tot_w4 = S * S * PAIR_W4;
    for (int idx = threadIdx.x; idx < tot_w4; idx += 128) {
        const int p = idx / PAIR_W4;
        const int off = idx - p * PAIR_W4;
        s_W4[p * PAIR_STRIDE_W4 + off] = __ldg(W_fast4 + idx);
    }
    __syncthreads();

    const int aid    = threadIdx.x >> 4;
    const int lane16 = threadIdx.x & 15;
    const int q      = lane16 >> 1;
    const int c      = lane16 & 1;
    const unsigned hmask = (aid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + aid;
    if (i >= N) return;

    const bool zbl_on = has_zbl_sw && ((buf.zbl_flag == nullptr) || (*buf.zbl_flag != 0));
    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float c_i = zbl_on ? (cfg.half_esc * buf.sw[i]) : cfg.half_esc;
    const float4* qpd_p = buf.qp_dpol_pack8;

    const float4 qpi = qpd_p
        ? __ldg(qpd_p + (int64_t)i * 8 + c)
        : __ldg(buf.qp_pack4 + (int64_t)i * 4 + c);
    const float qi = qpi.x, pix = qpi.y, piy = qpi.z, piz = qpi.w;
    const float4 dpi_pol = qpd_p
        ? __ldg(qpd_p + (int64_t)i * 8 + 4 + c)
        : __ldg(buf.dpol_ef4 + (int64_t)i * 4 + c);
    const float qa0_i = fmaf(c_i, qi, dpi_pol.x);
    const float pa1_ix = fmaf(c_i, pix, dpi_pol.y);
    const float pa1_iy = fmaf(c_i, piy, dpi_pol.z);
    const float pa1_iz = fmaf(c_i, piz, dpi_pol.w);

    const float4 seed_dqp = (q == 0) ? __ldg(buf.dqp_pack4 + (int64_t)i * 4 + c) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float dqi = 0.0f, dpix = seed_dqp.y, dpiy = seed_dqp.z, dpiz = seed_dqp.w;
    float rfx_f = 0.0f, rfy_f = 0.0f, rfz_f = 0.0f;
    double rfx_d = 0.0, rfy_d = 0.0, rfz_d = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (lane16 < bcnt) {
            const int e = eb_b + lane16;
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r, inv_r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                const float r2 = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                inv_r = rsqrtf(fmaxf(r2, 1e-16f));
                r = r2 * inv_r;
            } else {
                j = __ldg(senders + e);
                r = fmaxf(vl.w, 1e-8f);
                inv_r = 1.0f / r;
            }
            const int z_j = __ldg(species + j);
            s_jp[aid][lane16]  = make_int2(j, z_i * S + z_j);
            s_u[aid][lane16]   = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            if (zbl_on) s_swj[aid][lane16] = __ldg(buf.sw + j);
            float4 g_tmp[NB2_4], dg_tmp[NB2_4];
            eval_cheb_store<NB2, true>(r, r_max2, g_tmp, dg_tmp);
            #pragma unroll
            for (int b = 0; b < NB2_4; ++b) {
                s_g[b][aid][lane16]  = g_tmp[b];
                s_dg[b][aid][lane16] = dg_tmp[b];
            }
        }
        __syncwarp(hmask);

        for (int e_idx = q; e_idx < bcnt; e_idx += 8) {
            const int2 jp = s_jp[aid][e_idx];
            const int j = jp.x, pair = jp.y;
            const float4 u4 = s_u[aid][e_idx];
            const float ux = u4.x, uy = u4.y, uz = u4.z, inv_r = u4.w;
            const float c_j = zbl_on ? (cfg.half_esc * s_swj[aid][e_idx]) : cfg.half_esc;
            const float c_sum = c_i + c_j;
            const float4* Wp  = s_W4 + (int64_t)pair * PAIR_STRIDE_W4 + c;

            float dw2b = 0.0f;
            float w0 = 0.0f, dw0 = 0.0f;
            float w01f = 0.0f, dw01f = 0.0f, w01r = 0.0f, dw01r = 0.0f;
            float w1s = 0.0f, dw1s = 0.0f, w1p = 0.0f, dw1p = 0.0f;
            #pragma unroll
            for (int b = 0; b < NB2_4; ++b) {
                const float4 gb  = s_g[b][aid][e_idx];
                const float4 dgb = s_dg[b][aid][e_idx];
                const float4* Wb = Wp + (b * N_T4) * 4;
                if (c == 0) dw2b += v1_dot4(Wb[5 * 4], dgb);
                { const float4 wv = Wb[0 * 4]; w0   += v1_dot4(wv, gb); dw0   += v1_dot4(wv, dgb); }
                { const float4 wv = Wb[1 * 4]; w01f += v1_dot4(wv, gb); dw01f += v1_dot4(wv, dgb); }
                { const float4 wv = Wb[2 * 4]; w01r += v1_dot4(wv, gb); dw01r += v1_dot4(wv, dgb); }
                { const float4 wv = Wb[3 * 4]; w1s  += v1_dot4(wv, gb); dw1s  += v1_dot4(wv, dgb); }
                { const float4 wv = Wb[4 * 4]; w1p  += v1_dot4(wv, gb); dw1p  += v1_dot4(wv, dgb); }
            }

            const float4 qpj = qpd_p
                ? __ldg(qpd_p + (int64_t)j * 8 + c)
                : __ldg(buf.qp_pack4 + (int64_t)j * 4 + c);
            const float qj = qpj.x, pjx = qpj.y, pjy = qpj.z, pjz = qpj.w;
            const float4 dpj_pol = qpd_p
                ? __ldg(qpd_p + (int64_t)j * 8 + 4 + c)
                : __ldg(buf.dpol_ef4 + (int64_t)j * 4 + c);
            const float dPhi_j = dpj_pol.x, dEfjx = dpj_pol.y, dEfjy = dpj_pol.z, dEfjz = dpj_pol.w;
            const float qb0_j = fmaf(c_j, qj, dPhi_j);
            const float pb1_jx = fmaf(c_j, pjx, dEfjx);
            const float pb1_jy = fmaf(c_j, pjy, dEfjy);
            const float pb1_jz = fmaf(c_j, pjz, dEfjz);

            float er_lane = 0.0f, dux = 0.0f, duy = 0.0f, duz = 0.0f;

            if (c == 0) er_lane = fmaf(c_sum, dw2b, er_lane);
            {
                er_lane = fmaf(dw0, fmaf(qa0_i, qj, qb0_j * qi), er_lane);
                dqi = fmaf(w0, fmaf(c_i, qj, qb0_j), dqi);
            }
            const float piu = fmaf(pix, ux, fmaf(piy, uy, piz * uz));
            const float pju = fmaf(pjx, ux, fmaf(pjy, uy, pjz * uz));
            {
                const float dEfix = fmaf(-c_i, pix, pa1_ix);
                const float dEfiy = fmaf(-c_i, piy, pa1_iy);
                const float dEfiz = fmaf(-c_i, piz, pa1_iz);
                const float defi_u = fmaf(dEfix, ux, fmaf(dEfiy, uy, dEfiz * uz));
                const float defj_u = fmaf(dEfjx, ux, fmaf(dEfjy, uy, dEfjz * uz));
                er_lane = fmaf(dw01f, fmaf(qa0_i, pju, qj * defi_u),
                          fmaf(-dw01r, fmaf(qb0_j, piu, qi * defj_u), er_lane));
                const float sf_a = w01f * qa0_i, sf_b = w01f * qj;
                const float sr_a = w01r * qb0_j, sr_b = w01r * qi;
                dux = fmaf(sf_a, pjx, fmaf(sf_b, dEfix, fmaf(-sr_a, pix, fmaf(-sr_b, dEfjx, dux))));
                duy = fmaf(sf_a, pjy, fmaf(sf_b, dEfiy, fmaf(-sr_a, piy, fmaf(-sr_b, dEfjy, duy))));
                duz = fmaf(sf_a, pjz, fmaf(sf_b, dEfiz, fmaf(-sr_a, piz, fmaf(-sr_b, dEfjz, duz))));
                dqi  = fmaf(c_i * w01f, pju, fmaf(-w01r, defj_u, dqi));
                dpix = fmaf(-sr_a, ux, dpix); dpiy = fmaf(-sr_a, uy, dpiy); dpiz = fmaf(-sr_a, uz, dpiz);
            }
            {
                const float pa1_u = fmaf(pa1_ix, ux, fmaf(pa1_iy, uy, pa1_iz * uz));
                const float pb1_u = fmaf(pb1_jx, ux, fmaf(pb1_jy, uy, pb1_jz * uz));
                const float p1_dot = fmaf(pa1_ix, pjx, fmaf(pa1_iy, pjy, pa1_iz * pjz)) +
                                     fmaf(pb1_jx, pix, fmaf(pb1_jy, piy, pb1_jz * piz));
                er_lane = fmaf(dw1s, fmaf(pa1_u, pju, pb1_u * piu), fmaf(dw1p, p1_dot, er_lane));
                const float sf_a = w1s * pju, sf_b = w1s * pa1_u;
                const float sr_a = w1s * piu, sr_b = w1s * pb1_u;
                dux = fmaf(sf_a, pa1_ix, fmaf(sf_b, pjx, fmaf(sr_a, pb1_jx, fmaf(sr_b, pix, dux))));
                duy = fmaf(sf_a, pa1_iy, fmaf(sf_b, pjy, fmaf(sr_a, pb1_jy, fmaf(sr_b, piy, duy))));
                duz = fmaf(sf_a, pa1_iz, fmaf(sf_b, pjz, fmaf(sr_a, pb1_jz, fmaf(sr_b, piz, duz))));
                const float peff_x = fmaf(c_i, pjx, pb1_jx);
                const float peff_y = fmaf(c_i, pjy, pb1_jy);
                const float peff_z = fmaf(c_i, pjz, pb1_jz);
                const float peff_u = fmaf(c_i, pju, pb1_u);
                dpix = fmaf(w1s * peff_u, ux, fmaf(w1p, peff_x, dpix));
                dpiy = fmaf(w1s * peff_u, uy, fmaf(w1p, peff_y, dpiy));
                dpiz = fmaf(w1s * peff_u, uz, fmaf(w1p, peff_z, dpiz));
            }

            const float udot = fmaf(ux, dux, fmaf(uy, duy, uz * duz));
            const float gx = fmaf(er_lane, ux, (dux - udot * ux) * inv_r);
            const float gy = fmaf(er_lane, uy, (duy - udot * uy) * inv_r);
            const float gz = fmaf(er_lane, uz, (duz - udot * uz) * inv_r);
            if constexpr (STRESS) {
                rfx_d -= (double)gx; rfy_d -= (double)gy; rfz_d -= (double)gz;
                const float4 vl = __ldg(vec_len + eb_b + e_idx);
                v0 += 0.5 * (double)vl.x * (double)gx;
                v1 += 0.25 * ((double)vl.x * (double)gy + (double)vl.y * (double)gx);
                v2 += 0.25 * ((double)vl.x * (double)gz + (double)vl.z * (double)gx);
                v3 += 0.5 * (double)vl.y * (double)gy;
                v4 += 0.25 * ((double)vl.y * (double)gz + (double)vl.z * (double)gy);
                v5 += 0.5 * (double)vl.z * (double)gz;
            } else {
                rfx_f -= gx; rfy_f -= gy; rfz_f -= gz;
            }
        }
        __syncwarp(hmask);
    }

    dqi  += __shfl_xor_sync(hmask, dqi,  2); dqi  += __shfl_xor_sync(hmask, dqi,  4); dqi  += __shfl_xor_sync(hmask, dqi,  8);
    dpix += __shfl_xor_sync(hmask, dpix, 2); dpix += __shfl_xor_sync(hmask, dpix, 4); dpix += __shfl_xor_sync(hmask, dpix, 8);
    dpiy += __shfl_xor_sync(hmask, dpiy, 2); dpiy += __shfl_xor_sync(hmask, dpiy, 4); dpiy += __shfl_xor_sync(hmask, dpiy, 8);
    dpiz += __shfl_xor_sync(hmask, dpiz, 2); dpiz += __shfl_xor_sync(hmask, dpiz, 4); dpiz += __shfl_xor_sync(hmask, dpiz, 8);
    if (q == 0) buf.dqp_pack4[(int64_t)i * 4 + c]     = make_float4(dqi, dpix, dpiy, dpiz);
    if (q == 1) buf.dqp_pack4[(int64_t)i * 4 + 2 + c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if constexpr (STRESS) {
        rfx_d = half_warp_sum_d(rfx_d, hmask);
        rfy_d = half_warp_sum_d(rfy_d, hmask);
        rfz_d = half_warp_sum_d(rfz_d, hmask);
        v0 = half_warp_sum_d(v0, hmask); v1 = half_warp_sum_d(v1, hmask); v2 = half_warp_sum_d(v2, hmask);
        v3 = half_warp_sum_d(v3, hmask); v4 = half_warp_sum_d(v4, hmask); v5 = half_warp_sum_d(v5, hmask);
        if (lane16 == 0) {
            if (forces_f32) {
                forces_f32[(int64_t)i * 3 + 0] += (float)rfx_d;
                forces_f32[(int64_t)i * 3 + 1] += (float)rfy_d;
                forces_f32[(int64_t)i * 3 + 2] += (float)rfz_d;
            } else {
                forces[(int64_t)i * 3 + 0] += rfx_d;
                forces[(int64_t)i * 3 + 1] += rfy_d;
                forces[(int64_t)i * 3 + 2] += rfz_d;
            }
            atomicAdd(virial + 0, v0); atomicAdd(virial + 1, v1); atomicAdd(virial + 2, v2);
            atomicAdd(virial + 3, v3); atomicAdd(virial + 4, v4); atomicAdd(virial + 5, v5);
        }
    } else {
        rfx_f = half_warp_sum_f(rfx_f, hmask);
        rfy_f = half_warp_sum_f(rfy_f, hmask);
        rfz_f = half_warp_sum_f(rfz_f, hmask);
        if (lane16 == 0) {
            if (forces_f32) {
                forces_f32[(int64_t)i * 3 + 0] += rfx_f;
                forces_f32[(int64_t)i * 3 + 1] += rfy_f;
                forces_f32[(int64_t)i * 3 + 2] += rfz_f;
            } else {
                forces[(int64_t)i * 3 + 0] += (double)rfx_f;
                forces[(int64_t)i * 3 + 1] += (double)rfy_f;
                forces[(int64_t)i * 3 + 2] += (double)rfz_f;
            }
        }
    }
}

// =================================================================================================
// 5c) v36 Edge-Parallel 32-Lane (8 quads x 4 channels, 4 atoms/CTA, EB=32) LRSK Pair Kernels
//     - 2x more CTAs (1296 @ Water = 36 CTAs/SM), 2x fewer serial edge iters & barriers, 32-wide vec_len!
//     - Zero-padding bank-conflict-free s_g[4][NB2_4][32] & s_dg[4][NB2_4][32] (smaller SMEM than q16!)
//     - Fused 128B single-cache-line qp_dpol_pack8 neighbor reads + hoisted !zbl_on fast path!
// =================================================================================================
template <int NB2, bool PACK_VS>
__global__ void __launch_bounds__(128, 6) lrsk_edge_sym_pol_fwd_q32_kernel(
    int N, int S, Cfg cfg, ChebParamF r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_fast4,
    Bufs buf) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int N_T4 = 6;
    constexpr int EB = 32;
    constexpr int PAIR_W4 = NB2_4 * N_T4 * 4;
    constexpr int PAIR_STRIDE_W4 = PAIR_W4 + 4;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[4][NB2_4][EB];
    __shared__ float4 s_u[4][EB];
    __shared__ int2   s_jp[4][EB];

    const int tot_w4 = S * S * PAIR_W4;
    for (int idx = threadIdx.x; idx < tot_w4; idx += 128) {
        const int p = idx / PAIR_W4;
        const int off = idx - p * PAIR_W4;
        s_W4[p * PAIR_STRIDE_W4 + off] = __ldg(W_fast4 + idx);
    }
    __syncthreads();

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q    = lane >> 2;
    const int c    = lane & 3;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float4 qpi = __ldg(buf.qp_pack4 + (int64_t)i * 4 + c);
    const float qi = qpi.x, pix = qpi.y, piy = qpi.z, piz = qpi.w;

    float e_lr_acc = 0.0f, phi_acc = 0.0f, Efx = 0.0f, Efy = 0.0f, Efz = 0.0f;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (lane < bcnt) {
            const int e = eb_b + lane;
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r, inv_r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                const float r2 = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                inv_r = rsqrtf(fmaxf(r2, 1e-16f));
                r = r2 * inv_r;
            } else {
                j = __ldg(senders + e);
                r = fmaxf(vl.w, 1e-8f);
                inv_r = 1.0f / r;
            }
            s_jp[wid][lane] = make_int2(j, z_i * S + __ldg(species + j));
            s_u[wid][lane]  = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            float4 gt[NB2_4];
            eval_cheb_store<NB2, false>(r, r_max2, gt, nullptr);
            #pragma unroll
            for (int b = 0; b < NB2_4; ++b) s_g[wid][b][lane] = gt[b];
        }
        __syncwarp();

        for (int e_idx = q; e_idx < bcnt; e_idx += 8) {
            const int2 jp = s_jp[wid][e_idx];
            const int j = jp.x, pair = jp.y;
            const float4 u4 = s_u[wid][e_idx];
            const float ux = u4.x, uy = u4.y, uz = u4.z;
            const float4* Wp = s_W4 + (int64_t)pair * PAIR_STRIDE_W4 + c;
            const float4 g0_reg = (NB2_4 == 1) ? s_g[wid][0][e_idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            auto eval_wf = [&](int slot) -> float {
                if constexpr (NB2_4 == 1) {
                    return v1_dot4(Wp[slot * 4], g0_reg);
                } else {
                    float w = v1_dot4(Wp[slot * 4], s_g[wid][0][e_idx]);
                    #pragma unroll
                    for (int b = 1; b < NB2_4; ++b) w += v1_dot4(Wp[(b * N_T4 + slot) * 4], s_g[wid][b][e_idx]);
                    return w;
                }
            };

            const float4 qpj = __ldg(buf.qp_pack4 + (int64_t)j * 4 + c);
            const float qj = qpj.x, pjx = qpj.y, pjy = qpj.z, pjz = qpj.w;

            if (c == 0) e_lr_acc += eval_wf(5);

            const float w0 = eval_wf(0);
            const float t0 = w0 * qj;
            e_lr_acc = fmaf(qi, t0, e_lr_acc);
            phi_acc += t0;

            const float piu = fmaf(pix, ux, fmaf(piy, uy, piz * uz));
            const float pju = fmaf(pjx, ux, fmaf(pjy, uy, pjz * uz));

            const float w01f = eval_wf(1);
            const float t01p = w01f * pju;
            e_lr_acc = fmaf(qi, t01p, e_lr_acc);
            phi_acc += t01p;
            const float t01q = w01f * qj;
            Efx = fmaf(t01q, ux, Efx); Efy = fmaf(t01q, uy, Efy); Efz = fmaf(t01q, uz, Efz);

            const float w1s = eval_wf(3);
            const float w1p = eval_wf(4);
            const float w1s_pju = w1s * pju;
            const float p_dot = fmaf(pix, pjx, fmaf(piy, pjy, piz * pjz));
            e_lr_acc = fmaf(piu, w1s_pju, fmaf(w1p, p_dot, e_lr_acc));
            Efx = fmaf(w1s_pju, ux, fmaf(w1p, pjx, Efx));
            Efy = fmaf(w1s_pju, uy, fmaf(w1p, pjy, Efy));
            Efz = fmaf(w1s_pju, uz, fmaf(w1p, pjz, Efz));
        }
        __syncwarp();
    }

    const float e_tot = warp_sum_f(e_lr_acc);
    if (lane == 0) buf.e_lr[i] = cfg.half_esc * e_tot;
    if (buf.pol_ef4) {
        phi_acc += __shfl_xor_sync(0xffffffffu, phi_acc, 4); phi_acc += __shfl_xor_sync(0xffffffffu, phi_acc, 8); phi_acc += __shfl_xor_sync(0xffffffffu, phi_acc, 16);
        Efx     += __shfl_xor_sync(0xffffffffu, Efx,     4); Efx     += __shfl_xor_sync(0xffffffffu, Efx,     8); Efx     += __shfl_xor_sync(0xffffffffu, Efx,     16);
        Efy     += __shfl_xor_sync(0xffffffffu, Efy,     4); Efy     += __shfl_xor_sync(0xffffffffu, Efy,     8); Efy     += __shfl_xor_sync(0xffffffffu, Efy,     16);
        Efz     += __shfl_xor_sync(0xffffffffu, Efz,     4); Efz     += __shfl_xor_sync(0xffffffffu, Efz,     8); Efz     += __shfl_xor_sync(0xffffffffu, Efz,     16);
        if (q == 0) buf.pol_ef4[(int64_t)i * 4 + c] = make_float4(phi_acc, Efx, Efy, Efz);
    }
}

template <int NB2, bool STRESS, bool PACK_VS>
__global__ void __launch_bounds__(128, 5) lrsk_edge_sym_pol_bwd_q32_kernel(
    int N, int S, bool has_zbl_sw, Cfg cfg, ChebParamF r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_fast4,
    Bufs buf,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int N_T4 = 6;
    constexpr int EB = 32;
    constexpr int PAIR_W4 = NB2_4 * N_T4 * 4;
    constexpr int PAIR_STRIDE_W4 = PAIR_W4 + 4;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[4][NB2_4][EB];
    __shared__ float4 s_dg[4][NB2_4][EB];
    __shared__ float4 s_u[4][EB];
    __shared__ int2   s_jp[4][EB];
    __shared__ float  s_swj[4][EB];

    const int tot_w4 = S * S * PAIR_W4;
    for (int idx = threadIdx.x; idx < tot_w4; idx += 128) {
        const int p = idx / PAIR_W4;
        const int off = idx - p * PAIR_W4;
        s_W4[p * PAIR_STRIDE_W4 + off] = __ldg(W_fast4 + idx);
    }
    __syncthreads();

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q    = lane >> 2;
    const int c    = lane & 3;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const bool zbl_on = has_zbl_sw && ((buf.zbl_flag == nullptr) || (*buf.zbl_flag != 0));
    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float c_i = zbl_on ? (cfg.half_esc * buf.sw[i]) : cfg.half_esc;

    const float4 qpi = buf.qp_dpol_pack8
        ? __ldg(buf.qp_dpol_pack8 + (int64_t)i * 8 + c)
        : __ldg(buf.qp_pack4 + (int64_t)i * 4 + c);
    const float qi = qpi.x, pix = qpi.y, piy = qpi.z, piz = qpi.w;
    const float4 dpi_pol = buf.qp_dpol_pack8
        ? __ldg(buf.qp_dpol_pack8 + (int64_t)i * 8 + 4 + c)
        : __ldg(buf.dpol_ef4 + (int64_t)i * 4 + c);
    const float qa0_i = fmaf(c_i, qi, dpi_pol.x);
    const float pa1_ix = fmaf(c_i, pix, dpi_pol.y);
    const float pa1_iy = fmaf(c_i, piy, dpi_pol.z);
    const float pa1_iz = fmaf(c_i, piz, dpi_pol.w);

    const float4 seed_dqp = (q == 0) ? __ldg(buf.dqp_pack4 + (int64_t)i * 4 + c) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float dqi = 0.0f, dpix = seed_dqp.y, dpiy = seed_dqp.z, dpiz = seed_dqp.w;
    float rfx_f = 0.0f, rfy_f = 0.0f, rfz_f = 0.0f;
    double rfx_d = 0.0, rfy_d = 0.0, rfz_d = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (lane < bcnt) {
            const int e = eb_b + lane;
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r, inv_r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                const float r2 = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                inv_r = rsqrtf(fmaxf(r2, 1e-16f));
                r = r2 * inv_r;
            } else {
                j = __ldg(senders + e);
                r = fmaxf(vl.w, 1e-8f);
                inv_r = 1.0f / r;
            }
            const int z_j = __ldg(species + j);
            s_jp[wid][lane] = make_int2(j, z_i * S + z_j);
            s_u[wid][lane]  = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            if (zbl_on) s_swj[wid][lane] = __ldg(buf.sw + j);
            float4 gt[NB2_4], dgt[NB2_4];
            eval_cheb_store<NB2, true>(r, r_max2, gt, dgt);
            #pragma unroll
            for (int b = 0; b < NB2_4; ++b) {
                s_g[wid][b][lane]  = gt[b];
                s_dg[wid][b][lane] = dgt[b];
            }
        }
        __syncwarp();

        for (int e_idx = q; e_idx < bcnt; e_idx += 8) {
            const int2 jp = s_jp[wid][e_idx];
            const int j = jp.x, pair = jp.y;
            const float4 u4 = s_u[wid][e_idx];
            const float ux = u4.x, uy = u4.y, uz = u4.z, inv_r = u4.w;
            const float c_j = zbl_on ? (cfg.half_esc * s_swj[wid][e_idx]) : cfg.half_esc;
            const float c_sum = c_i + c_j;
            const float4* Wp = s_W4 + (int64_t)pair * PAIR_STRIDE_W4 + c;
            const float4 g0_reg  = (NB2_4 == 1) ? s_g[wid][0][e_idx]  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 dg0_reg = (NB2_4 == 1) ? s_dg[wid][0][e_idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            auto eval_w = [&](int slot, float& w, float& dw) {
                if constexpr (NB2_4 == 1) {
                    const float4 wv0 = Wp[slot * 4];
                    w  = v1_dot4(wv0, g0_reg);
                    dw = v1_dot4(wv0, dg0_reg);
                } else {
                    const float4 wv0 = Wp[slot * 4];
                    w  = v1_dot4(wv0, s_g[wid][0][e_idx]);
                    dw = v1_dot4(wv0, s_dg[wid][0][e_idx]);
                    #pragma unroll
                    for (int b = 1; b < NB2_4; ++b) {
                        const float4 wv = Wp[(b * N_T4 + slot) * 4];
                        w  += v1_dot4(wv, s_g[wid][b][e_idx]);
                        dw += v1_dot4(wv, s_dg[wid][b][e_idx]);
                    }
                }
            };

            const float4 qpj = buf.qp_dpol_pack8
                ? __ldg(buf.qp_dpol_pack8 + (int64_t)j * 8 + c)
                : __ldg(buf.qp_pack4 + (int64_t)j * 4 + c);
            const float qj = qpj.x, pjx = qpj.y, pjy = qpj.z, pjz = qpj.w;
            const float4 dpj_pol = buf.qp_dpol_pack8
                ? __ldg(buf.qp_dpol_pack8 + (int64_t)j * 8 + 4 + c)
                : __ldg(buf.dpol_ef4 + (int64_t)j * 4 + c);
            const float dPhi_j = dpj_pol.x, dEfjx = dpj_pol.y, dEfjy = dpj_pol.z, dEfjz = dpj_pol.w;
            const float qb0_j = fmaf(c_j, qj, dPhi_j);
            const float pb1_jx = fmaf(c_j, pjx, dEfjx);
            const float pb1_jy = fmaf(c_j, pjy, dEfjy);
            const float pb1_jz = fmaf(c_j, pjz, dEfjz);

            float er_lane = 0.0f, dux = 0.0f, duy = 0.0f, duz = 0.0f;

            if (c == 0) {
                float w2b, dw2b;
                eval_w(5, w2b, dw2b);
                er_lane = fmaf(c_sum, dw2b, er_lane);
            }
            {
                float w0, dw0;
                eval_w(0, w0, dw0);
                er_lane = fmaf(dw0, fmaf(qa0_i, qj, qb0_j * qi), er_lane);
                dqi = fmaf(w0, fmaf(c_i, qj, qb0_j), dqi);
            }
            const float piu = fmaf(pix, ux, fmaf(piy, uy, piz * uz));
            const float pju = fmaf(pjx, ux, fmaf(pjy, uy, pjz * uz));
            {
                float w01f, dw01f, w01r, dw01r;
                eval_w(1, w01f, dw01f);
                eval_w(2, w01r, dw01r);
                const float dEfix = fmaf(-c_i, pix, pa1_ix);
                const float dEfiy = fmaf(-c_i, piy, pa1_iy);
                const float dEfiz = fmaf(-c_i, piz, pa1_iz);
                const float defi_u = fmaf(dEfix, ux, fmaf(dEfiy, uy, dEfiz * uz));
                const float defj_u = fmaf(dEfjx, ux, fmaf(dEfjy, uy, dEfjz * uz));
                er_lane = fmaf(dw01f, fmaf(qa0_i, pju, qj * defi_u),
                          fmaf(-dw01r, fmaf(qb0_j, piu, qi * defj_u), er_lane));
                const float sf_a = w01f * qa0_i, sf_b = w01f * qj;
                const float sr_a = w01r * qb0_j, sr_b = w01r * qi;
                dux = fmaf(sf_a, pjx, fmaf(sf_b, dEfix, fmaf(-sr_a, pix, fmaf(-sr_b, dEfjx, dux))));
                duy = fmaf(sf_a, pjy, fmaf(sf_b, dEfiy, fmaf(-sr_a, piy, fmaf(-sr_b, dEfjy, duy))));
                duz = fmaf(sf_a, pjz, fmaf(sf_b, dEfiz, fmaf(-sr_a, piz, fmaf(-sr_b, dEfjz, duz))));
                dqi  = fmaf(c_i * w01f, pju, fmaf(-w01r, defj_u, dqi));
                dpix = fmaf(-sr_a, ux, dpix); dpiy = fmaf(-sr_a, uy, dpiy); dpiz = fmaf(-sr_a, uz, dpiz);
            }
            {
                float w1s, dw1s, w1p, dw1p;
                eval_w(3, w1s, dw1s);
                eval_w(4, w1p, dw1p);
                const float pa1_u = fmaf(pa1_ix, ux, fmaf(pa1_iy, uy, pa1_iz * uz));
                const float pb1_u = fmaf(pb1_jx, ux, fmaf(pb1_jy, uy, pb1_jz * uz));
                const float p1_dot = fmaf(pa1_ix, pjx, fmaf(pa1_iy, pjy, pa1_iz * pjz)) +
                                     fmaf(pb1_jx, pix, fmaf(pb1_jy, piy, pb1_jz * piz));
                er_lane = fmaf(dw1s, fmaf(pa1_u, pju, pb1_u * piu), fmaf(dw1p, p1_dot, er_lane));
                const float sf_a = w1s * pju, sf_b = w1s * pa1_u;
                const float sr_a = w1s * piu, sr_b = w1s * pb1_u;
                dux = fmaf(sf_a, pa1_ix, fmaf(sf_b, pjx, fmaf(sr_a, pb1_jx, fmaf(sr_b, pix, dux))));
                duy = fmaf(sf_a, pa1_iy, fmaf(sf_b, pjy, fmaf(sr_a, pb1_jy, fmaf(sr_b, piy, duy))));
                duz = fmaf(sf_a, pa1_iz, fmaf(sf_b, pjz, fmaf(sr_a, pb1_jz, fmaf(sr_b, piz, duz))));
                const float peff_x = fmaf(c_i, pjx, pb1_jx);
                const float peff_y = fmaf(c_i, pjy, pb1_jy);
                const float peff_z = fmaf(c_i, pjz, pb1_jz);
                const float peff_u = fmaf(c_i, pju, pb1_u);
                dpix = fmaf(w1s * peff_u, ux, fmaf(w1p, peff_x, dpix));
                dpiy = fmaf(w1s * peff_u, uy, fmaf(w1p, peff_y, dpiy));
                dpiz = fmaf(w1s * peff_u, uz, fmaf(w1p, peff_z, dpiz));
            }

            const float udot = fmaf(ux, dux, fmaf(uy, duy, uz * duz));
            const float gx = fmaf(er_lane, ux, (dux - udot * ux) * inv_r);
            const float gy = fmaf(er_lane, uy, (duy - udot * uy) * inv_r);
            const float gz = fmaf(er_lane, uz, (duz - udot * uz) * inv_r);
            if constexpr (STRESS) {
                rfx_d -= (double)gx; rfy_d -= (double)gy; rfz_d -= (double)gz;
                const float4 vl = __ldg(vec_len + eb_b + e_idx);
                v0 += 0.5 * (double)vl.x * (double)gx;
                v1 += 0.25 * ((double)vl.x * (double)gy + (double)vl.y * (double)gx);
                v2 += 0.25 * ((double)vl.x * (double)gz + (double)vl.z * (double)gx);
                v3 += 0.5 * (double)vl.y * (double)gy;
                v4 += 0.25 * ((double)vl.y * (double)gz + (double)vl.z * (double)gy);
                v5 += 0.5 * (double)vl.z * (double)gz;
            } else {
                rfx_f -= gx; rfy_f -= gy; rfz_f -= gz;
            }
        }
        __syncwarp();
    }

    dqi  += __shfl_xor_sync(0xffffffffu, dqi,  4); dqi  += __shfl_xor_sync(0xffffffffu, dqi,  8); dqi  += __shfl_xor_sync(0xffffffffu, dqi,  16);
    dpix += __shfl_xor_sync(0xffffffffu, dpix, 4); dpix += __shfl_xor_sync(0xffffffffu, dpix, 8); dpix += __shfl_xor_sync(0xffffffffu, dpix, 16);
    dpiy += __shfl_xor_sync(0xffffffffu, dpiy, 4); dpiy += __shfl_xor_sync(0xffffffffu, dpiy, 8); dpiy += __shfl_xor_sync(0xffffffffu, dpiy, 16);
    dpiz += __shfl_xor_sync(0xffffffffu, dpiz, 4); dpiz += __shfl_xor_sync(0xffffffffu, dpiz, 8); dpiz += __shfl_xor_sync(0xffffffffu, dpiz, 16);
    if (q == 0) buf.dqp_pack4[(int64_t)i * 4 + c] = make_float4(dqi, dpix, dpiy, dpiz);

    if constexpr (STRESS) {
        rfx_d = warp_sum_d(rfx_d); rfy_d = warp_sum_d(rfy_d); rfz_d = warp_sum_d(rfz_d);
        v0 = warp_sum_d(v0); v1 = warp_sum_d(v1); v2 = warp_sum_d(v2);
        v3 = warp_sum_d(v3); v4 = warp_sum_d(v4); v5 = warp_sum_d(v5);
        if (lane == 0) {
            if (forces_f32) {
                forces_f32[(int64_t)i * 3 + 0] += (float)rfx_d;
                forces_f32[(int64_t)i * 3 + 1] += (float)rfy_d;
                forces_f32[(int64_t)i * 3 + 2] += (float)rfz_d;
            } else {
                forces[(int64_t)i * 3 + 0] += rfx_d;
                forces[(int64_t)i * 3 + 1] += rfy_d;
                forces[(int64_t)i * 3 + 2] += rfz_d;
            }
            atomicAdd(virial + 0, v0); atomicAdd(virial + 1, v1); atomicAdd(virial + 2, v2);
            atomicAdd(virial + 3, v3); atomicAdd(virial + 4, v4); atomicAdd(virial + 5, v5);
        }
    } else {
        rfx_f = warp_sum_f(rfx_f); rfy_f = warp_sum_f(rfy_f); rfz_f = warp_sum_f(rfz_f);
        if (lane == 0) {
            if (forces_f32) {
                forces_f32[(int64_t)i * 3 + 0] += rfx_f;
                forces_f32[(int64_t)i * 3 + 1] += rfy_f;
                forces_f32[(int64_t)i * 3 + 2] += rfz_f;
            } else {
                forces[(int64_t)i * 3 + 0] += (double)rfx_f;
                forces[(int64_t)i * 3 + 1] += (double)rfy_f;
                forces[(int64_t)i * 3 + 2] += (double)rfz_f;
            }
        }
    }
}

// =================================================================================================
// 6) Fallback split Pol-MLP + Pure Energy/ZBL + am->eA helpers
// =================================================================================================
__global__ void __launch_bounds__(128) lrsk_pol_pack_in_sym_kernel(
    int N, int ro_width, int in_pad, Cfg cfg,
    const float* __restrict__ h, Bufs buf, float* __restrict__ pol_in, int8_t* __restrict__ eA_pol) {
    const int sid = threadIdx.x >> 4, slane = threadIdx.x & 15;
    const unsigned smask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const unsigned qmask4 = (sid & 1) ? 0x000f0000u : 0x0000000fu;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;
    const int f_dim = cfg.pol_gram
        ? (cfg.c0 + (cfg.c1 * (cfg.c1 + 1)) / 2 + cfg.c1 * cfg.c1)
        : (cfg.c0 + 2 * cfg.c1);
    const int in_raw = f_dim + ro_width;
    float* dst = pol_in + (int64_t)i * in_pad;
    float am = 0.0f;
    if (slane < 4) {
        float4 ef = __ldg(buf.pol_ef4 + (int64_t)i * 4 + slane);
        const float4 qp = __ldg(buf.qp_pack4 + (int64_t)i * 4 + slane);
        const float ief2 = cfg.inv_ef_cap2;
        if (ief2 > 0.0f) {
            const float sw_e = cfg.sw_mp ? buf.sw[i] : 1.0f;
            const float s_phi = sw_e * rsqrtf(fmaf(ef.x * ef.x, ief2, 1.0f));
            const float s_ef  = sw_e * rsqrtf(fmaf(ef.y * ef.y + ef.z * ef.z + ef.w * ef.w, ief2, 1.0f));
            ef.x *= s_phi; ef.y *= s_ef; ef.z *= s_ef; ef.w *= s_ef;
        }
        if (slane < cfg.c0) { dst[slane] = ef.x; am = fmaxf(am, fabsf(ef.x)); }
        if (!cfg.pol_gram) {
            if (slane < cfg.c1) {
                const float esq = fmaf(ef.y, ef.y, fmaf(ef.z, ef.z, ef.w * ef.w));
                const float pde = fmaf(qp.y, ef.y, fmaf(qp.z, ef.z, qp.w * ef.w));
                dst[cfg.c0 + slane] = esq;
                dst[cfg.c0 + cfg.c1 + slane] = pde;
                am = fmaxf(am, fmaxf(fabsf(esq), fabsf(pde)));
            }
        } else {
            const int ee_off = cfg.c0;
            const int pe_off = cfg.c0 + (cfg.c1 * (cfg.c1 + 1)) / 2;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float ebx = __shfl_sync(qmask4, ef.y, b, 4);
                const float eby = __shfl_sync(qmask4, ef.z, b, 4);
                const float ebz = __shfl_sync(qmask4, ef.w, b, 4);
                if (slane < cfg.c1 && b < cfg.c1) {
                    if (b >= slane) {
                        const float dot_ee = fmaf(ef.y, ebx, fmaf(ef.z, eby, ef.w * ebz));
                        dst[ee_off + tri_idx(slane, b, cfg.c1)] = dot_ee;
                        am = fmaxf(am, fabsf(dot_ee));
                    }
                    const float dot_pe = fmaf(qp.y, ebx, fmaf(qp.z, eby, qp.w * ebz));
                    dst[pe_off + slane * cfg.c1 + b] = dot_pe;
                    am = fmaxf(am, fabsf(dot_pe));
                }
            }
        }
    }
    const float* hi = h + (int64_t)i * ro_width;
    for (int k = slane; k < ro_width; k += 16) {
        const float v = __ldg(hi + k);
        dst[f_dim + k] = v;
        am = fmaxf(am, fabsf(v));
    }
    for (int k = in_raw + slane; k < in_pad; k += 16) dst[k] = 0.0f;
    if (eA_pol) {
        am = fmaxf(am, __shfl_xor_sync(smask, am, 8));
        am = fmaxf(am, __shfl_xor_sync(smask, am, 4));
        am = fmaxf(am, __shfl_xor_sync(smask, am, 2));
        am = fmaxf(am, __shfl_xor_sync(smask, am, 1));
        if (slane == 0) eA_pol[i] = exp_from_am(am);
    }
}

template <bool ZBL_SW, bool NEED_GRAD>
__global__ void __launch_bounds__(128) lrsk_pol_act_energy_zbl_kernel(
    int N, int pw, int pw_pad, Cfg cfg,
    const int* __restrict__ species, const float* __restrict__ pol_b0, const float* __restrict__ pol_u,
    const float* __restrict__ e_site, const float* __restrict__ e_lr, const float* __restrict__ sw_arr,
    const double* __restrict__ e_zbl, float* __restrict__ pre_io, int8_t* __restrict__ eA_dpre,
    double* __restrict__ energy_out, double* __restrict__ atom_energy_out, float* __restrict__ w_zbl_out) {
    const int sid = threadIdx.x >> 4, slane = threadIdx.x & 15;
    const unsigned smask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;
    const int z = species[i];
    const float* pu_z = pol_u + (int64_t)z * pw;
    const float sw_i = sw_arr[i];
    float* pr = pre_io + (int64_t)i * pw_pad;
    float e_pol_l = 0.0f, am = 0.0f;
    for (int w = slane; w < pw_pad; w += 16) {
        if (w < pw) {
            const float pre = pr[w] + __ldg(pol_b0 + w);
            const float sig = 1.0f / (1.0f + __expf(-pre));
            const float act = pre * sig;
            const float uw = __ldg(pu_z + w);
            e_pol_l = fmaf(act, uw, e_pol_l);
            if constexpr (NEED_GRAD) {
                const float dp = sw_i * uw * (sig * (1.0f + pre * (1.0f - sig)));
                pr[w] = dp;
                am = fmaxf(am, fabsf(dp));
            }
        } else if constexpr (NEED_GRAD) {
            pr[w] = 0.0f;
        }
    }
    const float e_pol = half_warp_sum_f(e_pol_l, smask);
    if constexpr (NEED_GRAD) {
        if (eA_dpre) {
            am = fmaxf(am, __shfl_xor_sync(smask, am, 8));
            am = fmaxf(am, __shfl_xor_sync(smask, am, 4));
            am = fmaxf(am, __shfl_xor_sync(smask, am, 2));
            am = fmaxf(am, __shfl_xor_sync(smask, am, 1));
            if (slane == 0) eA_dpre[i] = exp_from_am(am);
        }
    }
    double val = 0.0;
    if (slane == 0) {
        const double e_ml = (double)e_site[i] + (double)e_lr[i] + (double)e_pol;
        if constexpr (ZBL_SW) {
            const double ez = e_zbl[i];
            double xc = ez * (double)cfg.inv_zc;
            double sw, dsw_dx;
            if (xc <= 0.0) { sw = 1.0; dsw_dx = 0.0; }
            else if (xc >= 1.0) { sw = 0.0; dsw_dx = 0.0; }
            else {
                const double xc2 = xc * xc, xc3 = xc2 * xc;
                sw = 1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2;
                dsw_dx = -30.0 * xc2 + 60.0 * xc3 - 30.0 * xc2 * xc2;
            }
            val = sw * e_ml + ez;
            if constexpr (NEED_GRAD) w_zbl_out[i] = (float)(1.0 + dsw_dx * (double)cfg.inv_zc * e_ml);
        } else {
            val = e_ml;
        }
        if (atom_energy_out) atom_energy_out[i] = val;
    }
    const double wsum = warp_sum_d(val);
    if ((threadIdx.x & 31) == 0 && wsum != 0.0) atomicAdd(energy_out, wsum);
}

template <bool ZBL_SW>
__global__ void __launch_bounds__(128) lrsk_pol_unpack_grad_sym_kernel(
    int N, int ro_width, int in_pad, Cfg cfg,
    const float* __restrict__ dpol_in, const double* __restrict__ e_zbl, float* __restrict__ w_zbl_io,
    Bufs buf, float* __restrict__ dh_pol_out) {
    const int sid = threadIdx.x >> 4, slane = threadIdx.x & 15;
    const unsigned qmask4 = (sid & 1) ? 0x000f0000u : 0x0000000fu;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;
    const int f_dim = cfg.pol_gram
        ? (cfg.c0 + (cfg.c1 * (cfg.c1 + 1)) / 2 + cfg.c1 * cfg.c1)
        : (cfg.c0 + 2 * cfg.c1);
    const float* dp = dpol_in + (int64_t)i * in_pad;
    if (slane < 4) {
        float dphi = (slane < cfg.c0) ? dp[slane] : 0.0f;
        const float4 ef_raw = __ldg(buf.pol_ef4 + (int64_t)i * 4 + slane);
        const float4 qp = __ldg(buf.qp_pack4 + (int64_t)i * 4 + slane);
        const float ief2 = cfg.inv_ef_cap2;
        float4 ef = ef_raw;
        float sw_e = 1.0f, s_phi = 1.0f, s_ef = 1.0f;
        if (ief2 > 0.0f) {
            sw_e  = cfg.sw_mp ? buf.sw[i] : 1.0f;
            s_phi = rsqrtf(fmaf(ef_raw.x * ef_raw.x, ief2, 1.0f));
            s_ef  = rsqrtf(fmaf(ef_raw.y * ef_raw.y + ef_raw.z * ef_raw.z + ef_raw.w * ef_raw.w, ief2, 1.0f));
            ef.x = (sw_e * s_phi) * ef_raw.x;
            ef.y = (sw_e * s_ef)  * ef_raw.y;
            ef.z = (sw_e * s_ef)  * ef_raw.z;
            ef.w = (sw_e * s_ef)  * ef_raw.w;
        }
        float defx = 0.0f, defy = 0.0f, defz = 0.0f;
        float dpx = 0.0f, dpy = 0.0f, dpz = 0.0f;
        if (!cfg.pol_gram) {
            if (slane < cfg.c1) {
                const float d_esq = dp[cfg.c0 + slane];
                const float d_pe  = dp[cfg.c0 + cfg.c1 + slane];
                defx = fmaf(2.0f * d_esq, ef.y, d_pe * qp.y);
                defy = fmaf(2.0f * d_esq, ef.z, d_pe * qp.z);
                defz = fmaf(2.0f * d_esq, ef.w, d_pe * qp.w);
                dpx = d_pe * ef.y; dpy = d_pe * ef.z; dpz = d_pe * ef.w;
            }
        } else {
            const int ee_off = cfg.c0;
            const int pe_off = cfg.c0 + (cfg.c1 * (cfg.c1 + 1)) / 2;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float ebx = __shfl_sync(qmask4, ef.y, b, 4);
                const float eby = __shfl_sync(qmask4, ef.z, b, 4);
                const float ebz = __shfl_sync(qmask4, ef.w, b, 4);
                const float pbx = __shfl_sync(qmask4, qp.y, b, 4);
                const float pby = __shfl_sync(qmask4, qp.z, b, 4);
                const float pbz = __shfl_sync(qmask4, qp.w, b, 4);
                if (slane < cfg.c1 && b < cfg.c1) {
                    const float c_ee = ((b == slane) ? 2.0f : 1.0f) * dp[ee_off + tri_idx(slane, b, cfg.c1)];
                    const float c_pe_ef = dp[pe_off + b * cfg.c1 + slane];
                    const float c_pe_p  = dp[pe_off + slane * cfg.c1 + b];
                    defx = fmaf(c_ee, ebx, fmaf(c_pe_ef, pbx, defx));
                    defy = fmaf(c_ee, eby, fmaf(c_pe_ef, pby, defy));
                    defz = fmaf(c_ee, ebz, fmaf(c_pe_ef, pbz, defz));
                    dpx = fmaf(c_pe_p, ebx, dpx);
                    dpy = fmaf(c_pe_p, eby, dpy);
                    dpz = fmaf(c_pe_p, ebz, dpz);
                }
            }
        }
        if (ief2 > 0.0f) {
            float d_sw_l = 0.0f;
            if (slane < cfg.c0) {
                if (cfg.sw_mp) d_sw_l += dphi * (s_phi * ef_raw.x);
                dphi = sw_e * (s_phi * s_phi * s_phi) * dphi;
            }
            if (slane < cfg.c1) {
                const float ddot = defx * ef_raw.y + defy * ef_raw.z + defz * ef_raw.w;
                if (cfg.sw_mp) d_sw_l += s_ef * ddot;
                defx *= sw_e; defy *= sw_e; defz *= sw_e;
                const float proj = (s_ef * s_ef * s_ef * ief2) * (sw_e * ddot);
                defx = fmaf(s_ef, defx, -proj * ef_raw.y);
                defy = fmaf(s_ef, defy, -proj * ef_raw.z);
                defz = fmaf(s_ef, defz, -proj * ef_raw.w);
            }
            if constexpr (ZBL_SW) {
                if (cfg.sw_mp) {
                    const float d_sw_tot = q4_sum_f(d_sw_l, qmask4);
                    if (slane == 0 && e_zbl && w_zbl_io && d_sw_tot != 0.0f) {
                        const double ez = e_zbl[i];
                        if (ez > 0.0) {
                            const double xc = ez * (double)cfg.inv_zc;
                            if (xc < 1.0) {
                                const double xc2 = xc * xc, xc3 = xc2 * xc;
                                const double dsw_dx = -30.0 * xc2 + 60.0 * xc3 - 30.0 * xc2 * xc2;
                                w_zbl_io[i] += (float)(dsw_dx * (double)cfg.inv_zc * (double)d_sw_tot);
                            }
                        }
                    }
                }
            }
        }
        const float4 dp_out = make_float4(dphi, defx, defy, defz);
        buf.dpol_ef4[(int64_t)i * 4 + slane] = dp_out;
        buf.dqp_pack4[(int64_t)i * 4 + slane] = make_float4(0.0f, dpx, dpy, dpz);
        if (buf.qp_dpol_pack8) {
            buf.qp_dpol_pack8[(int64_t)i * 8 + slane]     = qp;
            buf.qp_dpol_pack8[(int64_t)i * 8 + 4 + slane] = dp_out;
        }
    }
    float* dho = dh_pol_out + (int64_t)i * ro_width;
    for (int k = slane; k < ro_width; k += 16) dho[k] = dp[f_dim + k];
}

template <bool ZBL_SW, bool NEED_GRAD>
__global__ void __launch_bounds__(128) lrsk_energy_zbl_pure_kernel(
    int N, Cfg cfg, const float* __restrict__ e_site, const float* __restrict__ e_lr,
    const double* __restrict__ e_zbl, double* __restrict__ energy_out,
    double* __restrict__ atom_energy_out, float* __restrict__ w_zbl_out) {
    const int i = blockIdx.x * 128 + threadIdx.x;
    double val = 0.0;
    if (i < N) {
        const double e_ml = (double)e_site[i] + (double)e_lr[i];
        if constexpr (ZBL_SW) {
            const double ez = e_zbl[i];
            double xc = ez * (double)cfg.inv_zc;
            double sw, dsw_dx;
            if (xc <= 0.0) { sw = 1.0; dsw_dx = 0.0; }
            else if (xc >= 1.0) { sw = 0.0; dsw_dx = 0.0; }
            else {
                const double xc2 = xc * xc, xc3 = xc2 * xc;
                sw = 1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2;
                dsw_dx = -30.0 * xc2 + 60.0 * xc3 - 30.0 * xc2 * xc2;
            }
            val = sw * e_ml + ez;
            if constexpr (NEED_GRAD) w_zbl_out[i] = (float)(1.0 + dsw_dx * (double)cfg.inv_zc * e_ml);
        } else {
            val = e_ml;
        }
        if (atom_energy_out) atom_energy_out[i] = val;
    }
    const double wsum = warp_sum_d(val);
    if ((threadIdx.x & 31) == 0 && wsum != 0.0) atomicAdd(energy_out, wsum);
}

__global__ void __launch_bounds__(128) lrsk_am_to_eA_kernel(
    int N, const float* __restrict__ am_l1, int8_t* __restrict__ eA_x) {
    const int i = blockIdx.x * 128 + threadIdx.x;
    if (i < N) eA_x[i] = exp_from_am(am_l1[i]);
}

}  // namespace

// =================================================================================================
// Host Launch Wrappers
// =================================================================================================
void launch_l1_node_fwd_lf1_lrsk(
    cudaStream_t st, int lmax, bool nl16,
    int N, int n_feat_pad, int n_msg,
    int nl0, int nl1, int nl2, int nl3, int m1d,
    const float* A0, const float* A1, const float* A2, const float* A3,
    const float* W1T, const float* mu_x, const float* inv_std_x,
    float* x_out, int8_t* eA_x_out, float* I1_out) {
    const bool use16 = nl16 || (nl0 <= 32 && nl1 <= 16 && nl2 <= 16 && nl3 <= 16);
    const size_t smem = (size_t)n_msg * m1d * sizeof(float);
    const dim3 blk(use16 ? 16u : 32u, use16 ? 8u : 4u);
    const dim3 grid((unsigned)((N + (use16 ? 7 : 3)) / (use16 ? 8 : 4)));
    if (lmax == 2) {
        if (use16) lrsk_l1_node_fwd_lf1_kernel<2, true><<<grid, blk, smem, st>>>(N, n_feat_pad, n_msg, nl0, nl1, nl2, nl3, m1d, A0, A1, A2, A3, W1T, mu_x, inv_std_x, x_out, eA_x_out, I1_out);
        else       lrsk_l1_node_fwd_lf1_kernel<2, false><<<grid, blk, smem, st>>>(N, n_feat_pad, n_msg, nl0, nl1, nl2, nl3, m1d, A0, A1, A2, A3, W1T, mu_x, inv_std_x, x_out, eA_x_out, I1_out);
    } else {
        if (use16) lrsk_l1_node_fwd_lf1_kernel<3, true><<<grid, blk, smem, st>>>(N, n_feat_pad, n_msg, nl0, nl1, nl2, nl3, m1d, A0, A1, A2, A3, W1T, mu_x, inv_std_x, x_out, eA_x_out, I1_out);
        else       lrsk_l1_node_fwd_lf1_kernel<3, false><<<grid, blk, smem, st>>>(N, n_feat_pad, n_msg, nl0, nl1, nl2, nl3, m1d, A0, A1, A2, A3, W1T, mu_x, inv_std_x, x_out, eA_x_out, I1_out);
    }
}

void launch_l1_node_bwd_lf1_lrsk(
    cudaStream_t st, int lmax, bool nl16,
    int N, int n_feat_pad, int n_msg,
    int nl0, int nl1, int nl2, int nl3, int m1d,
    const float* A0, const float* A1, const float* A2, const float* A3,
    const float* inv_std_x, const float* dx, const float* dI1, const float* W1T,
    float* dA0, float* dA1, float* dA2, float* dA3) {
    const bool use16 = nl16 || (nl0 <= 32 && nl1 <= 16 && nl2 <= 16 && nl3 <= 16);
    const size_t smem = (size_t)n_msg * m1d * sizeof(float);
    const dim3 blk(use16 ? 16u : 32u, use16 ? 8u : 4u);
    const dim3 grid((unsigned)((N + (use16 ? 7 : 3)) / (use16 ? 8 : 4)));
    if (lmax == 2) {
        if (use16) lrsk_l1_node_bwd_lf1_kernel<2, true><<<grid, blk, smem, st>>>(N, n_feat_pad, n_msg, nl0, nl1, nl2, nl3, m1d, A0, A1, A2, A3, inv_std_x, dx, dI1, W1T, dA0, dA1, dA2, dA3);
        else       lrsk_l1_node_bwd_lf1_kernel<2, false><<<grid, blk, smem, st>>>(N, n_feat_pad, n_msg, nl0, nl1, nl2, nl3, m1d, A0, A1, A2, A3, inv_std_x, dx, dI1, W1T, dA0, dA1, dA2, dA3);
    } else {
        if (use16) lrsk_l1_node_bwd_lf1_kernel<3, true><<<grid, blk, smem, st>>>(N, n_feat_pad, n_msg, nl0, nl1, nl2, nl3, m1d, A0, A1, A2, A3, inv_std_x, dx, dI1, W1T, dA0, dA1, dA2, dA3);
        else       lrsk_l1_node_bwd_lf1_kernel<3, false><<<grid, blk, smem, st>>>(N, n_feat_pad, n_msg, nl0, nl1, nl2, nl3, m1d, A0, A1, A2, A3, inv_std_x, dx, dI1, W1T, dA0, dA1, dA2, dA3);
    }
}

void launch_bias_silu_exp(
    cudaStream_t st, int N, int W, bool per_species,
    const int* species, const float* bias, float* pre, float* h, int8_t* eA_out) {
    const unsigned grid = (unsigned)((N + 3) / 4);
    if ((W & 3) == 0) {
        bias_silu_exp_vec4_kernel<<<grid, 128, 0, st>>>(
            N, W >> 2, per_species, species,
            reinterpret_cast<const float4*>(bias),
            reinterpret_cast<float4*>(pre),
            reinterpret_cast<float4*>(h), eA_out);
    } else {
        bias_silu_scalar_exp_kernel<<<grid, 128, 0, st>>>(N, W, per_species, species, bias, pre, h, eA_out);
    }
}

void launch_silu_bwd_exp(
    cudaStream_t st, int N, int W,
    const float* pre, const float* dh, float* dpre, int8_t* eA_out) {
    const unsigned grid = (unsigned)((N + 3) / 4);
    if ((W & 3) == 0) {
        silu_bwd_exp_vec4_kernel<<<grid, 128, 0, st>>>(
            N, W >> 2,
            reinterpret_cast<const float4*>(pre),
            reinterpret_cast<const float4*>(dh),
            reinterpret_cast<float4*>(dpre), eA_out);
    } else {
        silu_bwd_scalar_exp_kernel<<<grid, 128, 0, st>>>(N, W, pre, dh, dpre, eA_out);
    }
}

void launch_heads_fwd_silu(
    cudaStream_t st, int N, int S, int ro_width, int i_stride,
    bool zbl_sw, bool has_c2, bool per_species_bias, Cfg cfg,
    const int* species, const float* bias_last,
    const float* pre_last, float* h_last,
    const float* W_head_p16, const float* b_head_p16,
    const float* I_raw1, const float* I_raw2,
    const double* e_zbl, Bufs buf) {
    const int w_stride = has_c2 ? 16 : 9;
    const size_t wh_floats = ((size_t)S * (size_t)ro_width * w_stride + 3u) & ~3ull;
    const size_t sh_floats = (size_t)8 * (size_t)ro_width;
    const size_t smem_w_b = (wh_floats + sh_floats) * sizeof(float);
    const size_t smem_nw_b = sh_floats * sizeof(float);
    const bool use_smem = (smem_w_b <= 24576u) && (has_c2 || cfg.n_head_out <= 9);
    const size_t smem_b = use_smem ? smem_w_b : smem_nw_b;
    const unsigned full_g = (unsigned)((N + 7) / 8);
    const unsigned cap_g = (smem_b <= 15360u) ? 216u : 144u;
    const unsigned grid = use_smem ? std::min(full_g, cap_g) : full_g;
    auto go = [&](auto zb, auto c2, auto sm) {
        lrsk_heads_fwd_silu_sym_kernel<decltype(zb)::value, decltype(c2)::value, decltype(sm)::value>
            <<<grid, 128, smem_b, st>>>(
                N, S, ro_width, i_stride, per_species_bias, cfg,
                species, bias_last, pre_last, h_last, W_head_p16, b_head_p16, I_raw1, I_raw2, e_zbl, buf);
    };
    auto d_sm = [&](auto zb, auto c2) {
        if (use_smem) go(zb, c2, std::true_type{}); else go(zb, c2, std::false_type{});
    };
    auto d_c2 = [&](auto zb) {
        if (has_c2) d_sm(zb, std::true_type{}); else d_sm(zb, std::false_type{});
    };
    if (zbl_sw) d_c2(std::true_type{}); else d_c2(std::false_type{});
}

void launch_heads_bwd_silu(
    cudaStream_t st, int N, int S, int ro_width, int i_stride,
    bool pol_field, bool has_c2, bool zbl_sw, bool per_species_bias, Cfg cfg,
    const int* species, const float* bias_last, const float* W_head_p16,
    const float* I_raw1, const float* I_raw2,
    const float* pre_last, const float* dh_pol_in,
    const double* e_zbl, float* w_zbl_io,
    Bufs buf, float* dI_raw1, float* dI_raw2,
    float* dpre_out, int8_t* eA_dpre) {
    const size_t smem_b = (!has_c2 && ro_width == 64)
        ? ((size_t)S * 9u * 64u * sizeof(float))
        : ((size_t)S * (size_t)ro_width * 16u * sizeof(float));
    const bool use_smem = (smem_b <= 24576u) && (has_c2 || cfg.n_head_out <= 9);
    const unsigned full_g = (unsigned)((N + 7) / 8);
    const unsigned cap_g = (smem_b <= 14336u) ? 216u : 144u;
    const unsigned grid = use_smem ? std::min(full_g, cap_g) : full_g;
    auto go = [&](auto pf, auto c2, auto sm, auto zb) {
        lrsk_heads_bwd_silu_sym_kernel<decltype(pf)::value, decltype(c2)::value, decltype(sm)::value, decltype(zb)::value>
            <<<grid, 128, decltype(sm)::value ? smem_b : 0, st>>>(
                N, S, ro_width, i_stride, per_species_bias, cfg, species, bias_last, W_head_p16,
                I_raw1, I_raw2, pre_last, dh_pol_in, e_zbl, w_zbl_io,
                buf, dI_raw1, dI_raw2, dpre_out, eA_dpre);
    };
    auto d_zb = [&](auto pf, auto c2, auto sm) {
        if (zbl_sw) go(pf, c2, sm, std::true_type{}); else go(pf, c2, sm, std::false_type{});
    };
    auto d_sm = [&](auto pf, auto c2) {
        if (use_smem) d_zb(pf, c2, std::true_type{}); else d_zb(pf, c2, std::false_type{});
    };
    auto d_c2 = [&](auto pf) {
        if (has_c2) d_sm(pf, std::true_type{}); else d_sm(pf, std::false_type{});
    };
    if (pol_field) d_c2(std::true_type{}); else d_c2(std::false_type{});
}

void launch_edge_sym_pol_fwd(
    cudaStream_t st, int nb2, bool has_c2, bool skip_bo,
    int N, int S, Cfg cfg, ChebParamF r_max2,
    const int* row_ptr, const int* row_end,
    const int* senders, const float4* vec_len,
    const int* species, const float4* W_fast4, BoWeights bo, Bufs buf) {
    const unsigned grid32 = (unsigned)((N + 31) / 32);
    const int nb2_4 = (nb2 + 3) / 4;
    const int n_t4 = has_c2 ? 15 : 6;
    const size_t wf_sm = (size_t)S * S * nb2_4 * n_t4 * 4 * sizeof(float4);
    const bool has_bo = (cfg.bo_width > 0) && !skip_bo;
    const int h_pad = (cfg.bo_width + 3) & ~3;
    const size_t bo_sm16 = (size_t)(224 + S * S * 8 + S * S * nb2_4 * 16) * sizeof(float4);
    const size_t bo_sm8  = (size_t)(112 + S * S * 4 + S * S * nb2_4 * 8)  * sizeof(float4);
    int bo_mode = 0;
    size_t tot_sm = wf_sm;
    if (has_bo) {
        if (h_pad == 16 && wf_sm + bo_sm16 <= 44032u) { bo_mode = 1; tot_sm = wf_sm + bo_sm16; }
        else if (h_pad == 8 && wf_sm + bo_sm8 <= 44032u) { bo_mode = 3; tot_sm = wf_sm + bo_sm8; }
        else { bo_mode = 2; }
    }
    const bool pack_vs = (senders == nullptr);
    const char* q16_env = std::getenv("LOUP_LRSK_Q16");
    const bool allow_q16 = q16_env ? (q16_env[0] != '0') : true;
    const char* q32_env = std::getenv("LOUP_LRSK_Q32");
    const bool allow_q32 = q32_env ? (q32_env[0] != '0') : false;
    const size_t q_sm = (size_t)S * S * (nb2_4 * n_t4 * 4 + 4) * sizeof(float4);
    if (allow_q16 && bo_mode == 0 && (nb2 == 4 || nb2 == 8) && (q_sm <= 28672u)) {
        if (allow_q32 && !has_c2) {
            const unsigned grid4 = (unsigned)((N + 3) / 4);
            auto launch_q32 = [&](auto nb_tag, auto pv_tag) {
                auto kfn = lrsk_edge_sym_pol_fwd_q32_kernel<decltype(nb_tag)::value, decltype(pv_tag)::value>;
                if (q_sm > 20480u) CUDA_CHECK_THROW(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)q_sm));
                kfn<<<grid4, 128, q_sm, st>>>(N, S, cfg, r_max2, row_ptr, row_end, senders, vec_len, species, W_fast4, buf);
            };
            if (nb2 == 4) {
                if (pack_vs) launch_q32(std::integral_constant<int, 4>{}, std::true_type{});
                else         launch_q32(std::integral_constant<int, 4>{}, std::false_type{});
            } else {
                if (pack_vs) launch_q32(std::integral_constant<int, 8>{}, std::true_type{});
                else         launch_q32(std::integral_constant<int, 8>{}, std::false_type{});
            }
            return;
        }
        const unsigned grid8 = (unsigned)((N + 7) / 8);
        if (!has_c2 && cfg.c0 <= 2 && cfg.c1 <= 2 && !allow_q32) {
            auto launch_q16_c2 = [&](auto nb_tag, auto pv_tag) {
                auto kfn = lrsk_edge_sym_pol_fwd_q16_c2_kernel<decltype(nb_tag)::value, decltype(pv_tag)::value>;
                if (q_sm > 20480u) CUDA_CHECK_THROW(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)q_sm));
                kfn<<<grid8, 128, q_sm, st>>>(N, S, cfg, r_max2, row_ptr, row_end, senders, vec_len, species, W_fast4, buf);
            };
            if (nb2 == 4) {
                if (pack_vs) launch_q16_c2(std::integral_constant<int, 4>{}, std::true_type{});
                else         launch_q16_c2(std::integral_constant<int, 4>{}, std::false_type{});
            } else {
                if (pack_vs) launch_q16_c2(std::integral_constant<int, 8>{}, std::true_type{});
                else         launch_q16_c2(std::integral_constant<int, 8>{}, std::false_type{});
            }
            return;
        }
        auto launch_q16 = [&](auto nb_tag, auto pv_tag, auto c2_tag) {
            auto kfn = lrsk_edge_sym_pol_fwd_q16_kernel<decltype(nb_tag)::value, decltype(pv_tag)::value, decltype(c2_tag)::value>;
            if (q_sm > 20480u) CUDA_CHECK_THROW(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)q_sm));
            kfn<<<grid8, 128, q_sm, st>>>(N, S, cfg, r_max2, row_ptr, row_end, senders, vec_len, species, W_fast4, buf);
        };
        auto d_c2q = [&](auto nb_tag, auto pv_tag) {
            if (has_c2) launch_q16(nb_tag, pv_tag, std::true_type{});
            else        launch_q16(nb_tag, pv_tag, std::false_type{});
        };
        if (nb2 == 4) {
            if (pack_vs) d_c2q(std::integral_constant<int, 4>{}, std::true_type{});
            else         d_c2q(std::integral_constant<int, 4>{}, std::false_type{});
        } else {
            if (pack_vs) d_c2q(std::integral_constant<int, 8>{}, std::true_type{});
            else         d_c2q(std::integral_constant<int, 8>{}, std::false_type{});
        }
        return;
    }
    if (pack_vs && bo_mode == 0 && !has_c2 && nb2 == 4) {
        auto kfn = lrsk_edge_sym_pol_fwd_kernel<4, false, 0, true>;
        if (tot_sm > 20480u) CUDA_CHECK_THROW(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tot_sm));
        kfn<<<grid32, 128, tot_sm, st>>>(N, S, cfg, r_max2, row_ptr, row_end, nullptr, vec_len, species, W_fast4, bo, buf);
        return;
    }
    auto go = [&](auto nb_tag, auto c2_tag, auto bo_tag) {
        auto kfn = lrsk_edge_sym_pol_fwd_kernel<decltype(nb_tag)::value, decltype(c2_tag)::value, decltype(bo_tag)::value>;
        if (tot_sm > 20480u)
            CUDA_CHECK_THROW(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tot_sm));
        kfn<<<grid32, 128, tot_sm, st>>>(
            N, S, cfg, r_max2, row_ptr, row_end, senders, vec_len, species, W_fast4, bo, buf);
    };
    auto d_bo = [&](auto nb_tag, auto c2_tag) {
        if (bo_mode == 0)      go(nb_tag, c2_tag, std::integral_constant<int, 0>{});
        else if (bo_mode == 1) go(nb_tag, c2_tag, std::integral_constant<int, 1>{});
        else if (bo_mode == 3) go(nb_tag, c2_tag, std::integral_constant<int, 3>{});
        else                   go(nb_tag, c2_tag, std::integral_constant<int, 2>{});
    };
    auto d_c2 = [&](auto nb_tag) {
        if (has_c2) d_bo(nb_tag, std::true_type{}); else d_bo(nb_tag, std::false_type{});
    };
    if (nb2 == 4)       d_c2(std::integral_constant<int, 4>{});
    else if (nb2 == 8)  d_c2(std::integral_constant<int, 8>{});
    else                d_c2(std::integral_constant<int, 12>{});
}

void launch_edge_sym_1p(
    cudaStream_t st, int nb2, bool has_c2, bool stress,
    int N, int S, Cfg cfg, ChebParamF r_max2,
    const int* row_ptr, const int* row_end,
    const int* senders, const float4* vec_len,
    const int* species, const float4* W_fast4, BoWeights bo, Bufs buf,
    double* forces, float* forces_f32, double* virial) {
    const unsigned grid32 = (unsigned)((N + 31) / 32);
    const int nb2_4 = (nb2 + 3) / 4;
    const int n_t4 = has_c2 ? 15 : 6;
    const size_t wf_sm = (size_t)S * S * nb2_4 * n_t4 * 4 * sizeof(float4);
    const bool has_bo = (cfg.bo_width > 0);
    const int h_pad = (cfg.bo_width + 3) & ~3;
    const size_t bo_sm16 = (size_t)(224 + S * S * 8 + S * S * nb2_4 * 16) * sizeof(float4);
    const size_t bo_sm8  = (size_t)(112 + S * S * 4 + S * S * nb2_4 * 8)  * sizeof(float4);
    int bo_mode = 0;
    size_t tot_sm = wf_sm;
    if (has_bo) {
        if (h_pad == 16 && wf_sm + bo_sm16 <= 44032u) { bo_mode = 1; tot_sm = wf_sm + bo_sm16; }
        else if (h_pad == 8 && wf_sm + bo_sm8 <= 44032u) { bo_mode = 3; tot_sm = wf_sm + bo_sm8; }
        else { bo_mode = 2; }
    }
    auto go = [&](auto nb_tag, auto c2_tag, auto st_tag, auto bo_tag) {
        auto kfn = lrsk_edge_sym_1p_kernel<decltype(nb_tag)::value, decltype(c2_tag)::value, decltype(st_tag)::value, decltype(bo_tag)::value>;
        if (tot_sm > 20480u)
            CUDA_CHECK_THROW(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tot_sm));
        kfn<<<grid32, 128, tot_sm, st>>>(
            N, S, cfg, r_max2, row_ptr, row_end, senders, vec_len, species, W_fast4, bo, buf, forces, forces_f32, virial);
    };
    auto d_bo = [&](auto nb_tag, auto c2_tag, auto st_tag) {
        if (bo_mode == 0)      go(nb_tag, c2_tag, st_tag, std::integral_constant<int, 0>{});
        else if (bo_mode == 1) go(nb_tag, c2_tag, st_tag, std::integral_constant<int, 1>{});
        else if (bo_mode == 3) go(nb_tag, c2_tag, st_tag, std::integral_constant<int, 3>{});
        else                   go(nb_tag, c2_tag, st_tag, std::integral_constant<int, 2>{});
    };
    auto d_st = [&](auto nb_tag, auto c2_tag) {
        if (stress) d_bo(nb_tag, c2_tag, std::true_type{}); else d_bo(nb_tag, c2_tag, std::false_type{});
    };
    auto d_c2 = [&](auto nb_tag) {
        if (has_c2) d_st(nb_tag, std::true_type{}); else d_st(nb_tag, std::false_type{});
    };
    if (nb2 == 4)       d_c2(std::integral_constant<int, 4>{});
    else if (nb2 == 8)  d_c2(std::integral_constant<int, 8>{});
    else                d_c2(std::integral_constant<int, 12>{});
}

void launch_edge_sym_pol_bwd(
    cudaStream_t st, int nb2, bool has_c2, bool stress,
    int N, int S, Cfg cfg, ChebParamF r_max2,
    const int* row_ptr, const int* row_end,
    const int* senders, const float4* vec_len,
    const int* species, const float4* W_fast4, BoWeights bo, Bufs buf,
    const double* e_zbl, double* energy_out, double* atom_energy_out, float* w_zbl_io,
    double* forces, float* forces_f32, double* virial) {
    const unsigned grid32 = (unsigned)((N + 31) / 32);
    const int nb2_4 = (nb2 + 3) / 4;
    const int n_t4 = has_c2 ? 15 : 6;
    const size_t wf_sm = (size_t)S * S * nb2_4 * n_t4 * 4 * sizeof(float4);
    const bool has_bo = (cfg.bo_width > 0);
    const int h_pad = (cfg.bo_width + 3) & ~3;
    const size_t bo_sm16 = (size_t)(224 + S * S * 8 + S * S * nb2_4 * 16) * sizeof(float4);
    const size_t bo_sm8  = (size_t)(112 + S * S * 4 + S * S * nb2_4 * 8)  * sizeof(float4);
    int bo_mode = 0;
    size_t tot_sm = wf_sm;
    if (has_bo) {
        if (h_pad == 16 && wf_sm + bo_sm16 <= 44032u) { bo_mode = 1; tot_sm = wf_sm + bo_sm16; }
        else if (h_pad == 8 && wf_sm + bo_sm8 <= 44032u) { bo_mode = 3; tot_sm = wf_sm + bo_sm8; }
        else { bo_mode = 2; }
    }
    const bool pack_vs = (senders == nullptr);
    const char* q16_env = std::getenv("LOUP_LRSK_Q16");
    const bool allow_q16 = q16_env ? (q16_env[0] != '0') : true;
    const char* q32_env = std::getenv("LOUP_LRSK_Q32");
    const bool allow_q32 = q32_env ? (q32_env[0] != '0') : false;
    const size_t q_sm = (size_t)S * S * (nb2_4 * n_t4 * 4 + 4) * sizeof(float4);
    if (allow_q16 && bo_mode == 0 && (nb2 == 4 || nb2 == 8) && (q_sm <= 28672u)) {
        if (allow_q32 && !has_c2) {
            const unsigned grid4 = (unsigned)((N + 3) / 4);
            auto launch_q32 = [&](auto nb_tag, auto st_tag, auto pv_tag) {
                auto kfn = lrsk_edge_sym_pol_bwd_q32_kernel<decltype(nb_tag)::value, decltype(st_tag)::value, decltype(pv_tag)::value>;
                if (q_sm > 20480u) CUDA_CHECK_THROW(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)q_sm));
                kfn<<<grid4, 128, q_sm, st>>>(
                    N, S, (e_zbl != nullptr), cfg, r_max2, row_ptr, row_end, senders, vec_len, species, W_fast4, buf,
                    forces, forces_f32, virial);
            };
            auto d_pv32 = [&](auto nb_tag, auto st_tag) {
                if (pack_vs) launch_q32(nb_tag, st_tag, std::true_type{});
                else         launch_q32(nb_tag, st_tag, std::false_type{});
            };
            if (nb2 == 4) {
                if (stress) d_pv32(std::integral_constant<int, 4>{}, std::true_type{});
                else        d_pv32(std::integral_constant<int, 4>{}, std::false_type{});
            } else {
                if (stress) d_pv32(std::integral_constant<int, 8>{}, std::true_type{});
                else        d_pv32(std::integral_constant<int, 8>{}, std::false_type{});
            }
            return;
        }
        const unsigned grid8 = (unsigned)((N + 7) / 8);
        if (!has_c2 && cfg.c0 <= 2 && cfg.c1 <= 2 && !allow_q32) {
            auto launch_q16_c2 = [&](auto nb_tag, auto st_tag, auto pv_tag) {
                auto kfn = lrsk_edge_sym_pol_bwd_q16_c2_kernel<decltype(nb_tag)::value, decltype(st_tag)::value, decltype(pv_tag)::value>;
                if (q_sm > 20480u) CUDA_CHECK_THROW(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)q_sm));
                kfn<<<grid8, 128, q_sm, st>>>(
                    N, S, (e_zbl != nullptr), cfg, r_max2, row_ptr, row_end, senders, vec_len, species, W_fast4, buf,
                    forces, forces_f32, virial);
            };
            auto d_pv_c2 = [&](auto nb_tag, auto st_tag) {
                if (pack_vs) launch_q16_c2(nb_tag, st_tag, std::true_type{});
                else         launch_q16_c2(nb_tag, st_tag, std::false_type{});
            };
            if (nb2 == 4) {
                if (stress) d_pv_c2(std::integral_constant<int, 4>{}, std::true_type{});
                else        d_pv_c2(std::integral_constant<int, 4>{}, std::false_type{});
            } else {
                if (stress) d_pv_c2(std::integral_constant<int, 8>{}, std::true_type{});
                else        d_pv_c2(std::integral_constant<int, 8>{}, std::false_type{});
            }
            return;
        }
        auto launch_q16 = [&](auto nb_tag, auto st_tag, auto pv_tag, auto c2_tag) {
            auto kfn = lrsk_edge_sym_pol_bwd_q16_kernel<decltype(nb_tag)::value, decltype(st_tag)::value, decltype(pv_tag)::value, decltype(c2_tag)::value>;
            if (q_sm > 20480u) CUDA_CHECK_THROW(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)q_sm));
            kfn<<<grid8, 128, q_sm, st>>>(
                N, S, (e_zbl != nullptr), cfg, r_max2, row_ptr, row_end, senders, vec_len, species, W_fast4, buf,
                forces, forces_f32, virial);
        };
        auto d_c2q = [&](auto nb_tag, auto st_tag, auto pv_tag) {
            if (has_c2) launch_q16(nb_tag, st_tag, pv_tag, std::true_type{});
            else        launch_q16(nb_tag, st_tag, pv_tag, std::false_type{});
        };
        auto d_pv = [&](auto nb_tag, auto st_tag) {
            if (pack_vs) d_c2q(nb_tag, st_tag, std::true_type{});
            else         d_c2q(nb_tag, st_tag, std::false_type{});
        };
        if (nb2 == 4) {
            if (stress) d_pv(std::integral_constant<int, 4>{}, std::true_type{});
            else        d_pv(std::integral_constant<int, 4>{}, std::false_type{});
        } else {
            if (stress) d_pv(std::integral_constant<int, 8>{}, std::true_type{});
            else        d_pv(std::integral_constant<int, 8>{}, std::false_type{});
        }
        return;
    }
    if (pack_vs && bo_mode == 0 && !has_c2 && !stress && nb2 == 4) {
        auto kfn = lrsk_edge_sym_pol_bwd_kernel<4, false, false, 0, true>;
        if (tot_sm > 20480u) CUDA_CHECK_THROW(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tot_sm));
        kfn<<<grid32, 128, tot_sm, st>>>(
            N, S, cfg, r_max2, row_ptr, row_end, nullptr, vec_len, species, W_fast4, bo, buf,
            e_zbl, energy_out, atom_energy_out, w_zbl_io, forces, forces_f32, virial);
        return;
    }
    auto go = [&](auto nb_tag, auto c2_tag, auto st_tag, auto bo_tag) {
        auto kfn = lrsk_edge_sym_pol_bwd_kernel<decltype(nb_tag)::value, decltype(c2_tag)::value, decltype(st_tag)::value, decltype(bo_tag)::value>;
        if (tot_sm > 20480u)
            CUDA_CHECK_THROW(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tot_sm));
        kfn<<<grid32, 128, tot_sm, st>>>(
            N, S, cfg, r_max2, row_ptr, row_end, senders, vec_len, species, W_fast4, bo, buf,
            e_zbl, energy_out, atom_energy_out, w_zbl_io, forces, forces_f32, virial);
    };
    auto d_bo = [&](auto nb_tag, auto c2_tag, auto st_tag) {
        if (bo_mode == 0)      go(nb_tag, c2_tag, st_tag, std::integral_constant<int, 0>{});
        else if (bo_mode == 1) go(nb_tag, c2_tag, st_tag, std::integral_constant<int, 1>{});
        else if (bo_mode == 3) go(nb_tag, c2_tag, st_tag, std::integral_constant<int, 3>{});
        else                   go(nb_tag, c2_tag, st_tag, std::integral_constant<int, 2>{});
    };
    auto d_st = [&](auto nb_tag, auto c2_tag) {
        if (stress) d_bo(nb_tag, c2_tag, std::true_type{}); else d_bo(nb_tag, c2_tag, std::false_type{});
    };
    auto d_c2 = [&](auto nb_tag) {
        if (has_c2) d_st(nb_tag, std::true_type{}); else d_st(nb_tag, std::false_type{});
    };
    if (nb2 == 4)       d_c2(std::integral_constant<int, 4>{});
    else if (nb2 == 8)  d_c2(std::integral_constant<int, 8>{});
    else                d_c2(std::integral_constant<int, 12>{});
}

void launch_pack_pol_w0_hl(
    cudaStream_t st, bool pol_gram, int in_dim, int pw, int pol_w_pad,
    const float* pol_W0_pad, uint4* pol_cHL, uint4* pol_rHL, float* inv_sW_dev) {
    if (pol_gram)
        pack_pol_w0_hl_kernel<true><<<1, 128, 0, st>>>(in_dim, pw, pol_w_pad, pol_W0_pad, pol_cHL, pol_rHL, inv_sW_dev);
    else
        pack_pol_w0_hl_kernel<false><<<1, 128, 0, st>>>(in_dim, pw, pol_w_pad, pol_W0_pad, pol_cHL, pol_rHL, inv_sW_dev);
}

void launch_pol_mlp_fused(
    cudaStream_t st, bool zbl_sw, bool need_grad, bool pol_gram,
    int N, int S, int ro_width, float inv_sW, Cfg cfg,
    const int* species, const float* h_last,
    const uint4* pol_cHL, const uint4* pol_rHL, const float* pol_b0, const float* pol_u,
    const double* e_zbl, Bufs buf,
    double* energy_out, double* atom_energy_out,
    float* w_zbl_out, float* dh_pol_out) {
    const int f_dim = pol_gram
        ? (cfg.c0 + (cfg.c1 * (cfg.c1 + 1)) / 2 + cfg.c1 * cfg.c1)
        : (cfg.c0 + 2 * cfg.c1);
    const int in_dim = f_dim + ro_width;
    const unsigned grid = (unsigned)((N + 15) / 16);
    auto go = [&](auto zb, auto ng, auto pg, auto k_tag) {
        lrsk_pol_mlp_tc_kernel<decltype(zb)::value, decltype(ng)::value, decltype(pg)::value, decltype(k_tag)::value>
            <<<grid, 64, 0, st>>>(
                N, S, ro_width, inv_sW, cfg, species, h_last, pol_cHL, pol_rHL, pol_b0, pol_u, e_zbl,
                buf, energy_out, atom_energy_out, w_zbl_out, dh_pol_out);
    };
    auto d_k = [&](auto zb, auto ng, auto pg) {
        if (in_dim <= 48)      go(zb, ng, pg, std::integral_constant<int, 48>{});
        else if (in_dim <= 64) go(zb, ng, pg, std::integral_constant<int, 64>{});
        else if (in_dim <= 80) go(zb, ng, pg, std::integral_constant<int, 80>{});
        else                   go(zb, ng, pg, std::integral_constant<int, 96>{});
    };
    auto d_pg = [&](auto zb, auto ng) {
        if (pol_gram) d_k(zb, ng, std::true_type{}); else d_k(zb, ng, std::false_type{});
    };
    if (zbl_sw) {
        if (need_grad) d_pg(std::true_type{}, std::true_type{});
        else           d_pg(std::true_type{}, std::false_type{});
    } else {
        if (need_grad) d_pg(std::false_type{}, std::true_type{});
        else           d_pg(std::false_type{}, std::false_type{});
    }
}

void launch_pol_pack_in_sym(
    cudaStream_t st, int N, int ro_width, int in_pad, Cfg cfg,
    const float* h, Bufs buf, float* pol_in, int8_t* eA_pol) {
    lrsk_pol_pack_in_sym_kernel<<<(unsigned)((N + 7) / 8), 128, 0, st>>>(N, ro_width, in_pad, cfg, h, buf, pol_in, eA_pol);
}

void launch_pol_act_energy_zbl(
    cudaStream_t st, bool zbl_sw, bool need_grad,
    int N, int pw, int pw_pad, Cfg cfg,
    const int* species, const float* pol_b0, const float* pol_u,
    const float* e_site, const float* e_lr, const float* sw_arr,
    const double* e_zbl, float* pre_io, int8_t* eA_dpre,
    double* energy_out, double* atom_energy_out, float* w_zbl_out) {
    const unsigned grid = (unsigned)((N + 7) / 8);
    auto go = [&](auto zb, auto ng) {
        lrsk_pol_act_energy_zbl_kernel<decltype(zb)::value, decltype(ng)::value><<<grid, 128, 0, st>>>(
            N, pw, pw_pad, cfg, species, pol_b0, pol_u, e_site, e_lr, sw_arr, e_zbl,
            pre_io, eA_dpre, energy_out, atom_energy_out, w_zbl_out);
    };
    if (zbl_sw) {
        if (need_grad) go(std::true_type{}, std::true_type{}); else go(std::true_type{}, std::false_type{});
    } else {
        if (need_grad) go(std::false_type{}, std::true_type{}); else go(std::false_type{}, std::false_type{});
    }
}

void launch_pol_unpack_grad_sym(
    cudaStream_t st, bool zbl_sw, int N, int ro_width, int in_pad, Cfg cfg,
    const float* dpol_in, const double* e_zbl, float* w_zbl_io, Bufs buf, float* dh_pol_out) {
    const unsigned grid = (unsigned)((N + 7) / 8);
    if (zbl_sw)
        lrsk_pol_unpack_grad_sym_kernel<true><<<grid, 128, 0, st>>>(N, ro_width, in_pad, cfg, dpol_in, e_zbl, w_zbl_io, buf, dh_pol_out);
    else
        lrsk_pol_unpack_grad_sym_kernel<false><<<grid, 128, 0, st>>>(N, ro_width, in_pad, cfg, dpol_in, e_zbl, w_zbl_io, buf, dh_pol_out);
}

void launch_energy_zbl_pure(
    cudaStream_t st, bool zbl_sw, bool need_grad,
    int N, Cfg cfg, const float* e_site, const float* e_lr,
    const double* e_zbl, double* energy_out, double* atom_energy_out, float* w_zbl_out) {
    const unsigned grid = (unsigned)((N + 127) / 128);
    if (zbl_sw) {
        if (need_grad) lrsk_energy_zbl_pure_kernel<true, true><<<grid, 128, 0, st>>>(N, cfg, e_site, e_lr, e_zbl, energy_out, atom_energy_out, w_zbl_out);
        else           lrsk_energy_zbl_pure_kernel<true, false><<<grid, 128, 0, st>>>(N, cfg, e_site, e_lr, e_zbl, energy_out, atom_energy_out, nullptr);
    } else {
        if (need_grad) lrsk_energy_zbl_pure_kernel<false, true><<<grid, 128, 0, st>>>(N, cfg, e_site, e_lr, nullptr, energy_out, atom_energy_out, nullptr);
        else           lrsk_energy_zbl_pure_kernel<false, false><<<grid, 128, 0, st>>>(N, cfg, e_site, e_lr, nullptr, energy_out, atom_energy_out, nullptr);
    }
}

void launch_am_to_eA(cudaStream_t st, int N, const float* am_l1, int8_t* eA_x) {
    lrsk_am_to_eA_kernel<<<(unsigned)((N + 127) / 128), 128, 0, st>>>(N, am_l1, eA_x);
}

}  // namespace lrsk_fast
}  // namespace loup
