#include "ilmlip.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "bundle.hpp"
#include "generated_sh16.cuh"
#include "kernels_common.cuh"
#include "model.hpp"

namespace loup {
namespace {

struct Ptr4 {
    float* p[4];
};
struct CPtr4 {
    const float* p[4];
};
struct Int4 {
    int v[4];
};

template <typename T>
T* upload_vec(std::vector<void*>& owner, const std::vector<T>& h) {
    T* d = nullptr;
    const size_t bytes = std::max<size_t>(h.size(), 1) * sizeof(T);
    CUDA_CHECK_THROW(cudaMalloc(&d, bytes));
    owner.push_back(d);
    if (!h.empty()) CUDA_CHECK_THROW(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warp_sum_f(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ double warp_sum_d(double v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float half_warp_sum_f(float v, unsigned mask) {
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) v += __shfl_xor_sync(mask, v, off);
    return v;
}

__device__ __forceinline__ double half_warp_sum_d(double v, unsigned mask) {
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) v += __shfl_xor_sync(mask, v, off);
    return v;
}

// FP64 fallback for rare close-contact edges (r < 1.2 A, ZBL active): out-of-line to keep hot FP32 registers small.
template <bool WITH_DERIV>
__device__ __noinline__ void eval_cheb_fp64(float r_f, float r_max_f, int n_base, float* g, float* dg) {
    const double inv_rmax = 1.0 / (double)r_max_f;
    double t = (double)r_f * inv_rmax;
    if (t < 0.0) t = 0.0;
    if (t >= 1.0) {
        for (int k = 0; k < n_base; ++k) { g[k] = 0.0f; if (WITH_DERIV) dg[k] = 0.0f; }
        return;
    }
    const double t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
    const double env = 1.0 - 21.0 * t5 + 35.0 * t5 * t - 15.0 * t5 * t2;
    const double omt = 1.0 - t;
    const double denv_dt = WITH_DERIV ? (-105.0 * t4 * omt * omt) : 0.0;
    const double x = 2.0 * t - 1.0;
    double T0 = 1.0, dT0 = 0.0;
    g[0] = (float)env;
    if (WITH_DERIV) dg[0] = (float)(denv_dt * inv_rmax);
    if (n_base > 1) {
        double T1 = x, dT1 = 1.0;
        g[1] = (float)(T1 * env);
        if (WITH_DERIV) dg[1] = (float)((2.0 * dT1 * env + T1 * denv_dt) * inv_rmax);
        for (int k = 2; k < n_base; ++k) {
            const double Tk = 2.0 * x * T1 - T0;
            g[k] = (float)(Tk * env);
            if (WITH_DERIV) {
                const double dTk = 2.0 * T1 + 2.0 * x * dT1 - dT0;
                dg[k] = (float)((2.0 * dTk * env + Tk * denv_dt) * inv_rmax);
                dT0 = dT1; dT1 = dTk;
            }
            T0 = T1; T1 = Tk;
        }
    }
}

// Compute Chebyshev basis g[0..n_base-1] (with C^2 envelope) and optionally dg/dr in pure FP32.
// Uses exact IEEE __fdiv_rn(r, r_max) to avoid systematic 1/r_max stress bias without 1/64-rate FP64
// and with zero __noinline__ ABI spills (0 bytes stack frame in all edge kernels).
template <bool WITH_DERIV>
__device__ __forceinline__ void eval_cheb(float r_f, float r_max_f, int n_base, float* g, float* dg) {
    float t = __fdiv_rn(r_f, r_max_f);
    if (t < 0.0f) t = 0.0f;
    if (t >= 1.0f) {
        for (int k = 0; k < n_base; ++k) {
            g[k] = 0.0f;
            if (WITH_DERIV) dg[k] = 0.0f;
        }
        return;
    }
    const float inv_rmax = 1.0f / r_max_f;
    const float t2 = t * t;
    const float t4 = t2 * t2;
    const float t5 = t4 * t;
    const float env = 1.0f - 21.0f * t5 + 35.0f * t5 * t - 15.0f * t5 * t2;
    const float omt = 1.0f - t;
    const float denv_dt = WITH_DERIV ? (-105.0f * t4 * omt * omt) : 0.0f;
    const float x = 2.0f * t - 1.0f;
    const float x2 = 2.0f * x;
    float T0 = 1.0f;
    float dT0 = 0.0f;
    g[0] = env;
    if (WITH_DERIV) dg[0] = denv_dt * inv_rmax;
    if (n_base > 1) {
        float T1 = x;
        float dT1 = 1.0f;
        g[1] = T1 * env;
        if (WITH_DERIV) dg[1] = (2.0f * dT1 * env + T1 * denv_dt) * inv_rmax;
        for (int k = 2; k < n_base; ++k) {
            const float Tk = fmaf(x2, T1, -T0);
            g[k] = Tk * env;
            if (WITH_DERIV) {
                const float dTk = fmaf(x2, dT1, fmaf(2.0f, T1, -dT0));
                dg[k] = (2.0f * dTk * env + Tk * denv_dt) * inv_rmax;
                dT0 = dT1;
                dT1 = dTk;
            }
            T0 = T1;
            T1 = Tk;
        }
    }
}

// Compute 8 Chebyshev basis values (and optionally derivatives) directly into two float4s (0 stack, 0 bank conflicts).
template <bool WITH_DERIV>
__device__ __forceinline__ void eval_cheb8_f4(
    float r_f, float r_max_f, float4& g03, float4& g47, float4& dg03, float4& dg47) {
    float t = __fdiv_rn(r_f, r_max_f);
    if (t < 0.0f) t = 0.0f;
    if (t >= 1.0f) {
        g03 = g47 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if constexpr (WITH_DERIV) dg03 = dg47 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    const float inv_rmax = 1.0f / r_max_f;
    const float t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
    const float env = 1.0f - 21.0f * t5 + 35.0f * t5 * t - 15.0f * t5 * t2;
    const float omt = 1.0f - t;
    const float denv_dt = WITH_DERIV ? (-105.0f * t4 * omt * omt) : 0.0f;
    const float env2 = 2.0f * env;
    const float x = 2.0f * t - 1.0f;
    const float x2 = 2.0f * x;

    const float T0 = 1.0f;
    const float T1 = x;
    const float T2 = fmaf(x2, T1, -T0);
    const float T3 = fmaf(x2, T2, -T1);
    const float T4 = fmaf(x2, T3, -T2);
    const float T5 = fmaf(x2, T4, -T3);
    const float T6 = fmaf(x2, T5, -T4);
    const float T7 = fmaf(x2, T6, -T5);
    g03 = make_float4(T0 * env, T1 * env, T2 * env, T3 * env);
    g47 = make_float4(T4 * env, T5 * env, T6 * env, T7 * env);

    if constexpr (WITH_DERIV) {
        const float dT0 = 0.0f;
        const float dT1 = 1.0f;
        const float dT2 = fmaf(x2, dT1, fmaf(2.0f, T1, -dT0));
        const float dT3 = fmaf(x2, dT2, fmaf(2.0f, T2, -dT1));
        const float dT4 = fmaf(x2, dT3, fmaf(2.0f, T3, -dT2));
        const float dT5 = fmaf(x2, dT4, fmaf(2.0f, T4, -dT3));
        const float dT6 = fmaf(x2, dT5, fmaf(2.0f, T5, -dT4));
        const float dT7 = fmaf(x2, dT6, fmaf(2.0f, T6, -dT5));
        dg03 = make_float4(
            denv_dt * inv_rmax,
            fmaf(env2, dT1, T1 * denv_dt) * inv_rmax,
            fmaf(env2, dT2, T2 * denv_dt) * inv_rmax,
            fmaf(env2, dT3, T3 * denv_dt) * inv_rmax);
        dg47 = make_float4(
            fmaf(env2, dT4, T4 * denv_dt) * inv_rmax,
            fmaf(env2, dT5, T5 * denv_dt) * inv_rmax,
            fmaf(env2, dT6, T6 * denv_dt) * inv_rmax,
            fmaf(env2, dT7, T7 * denv_dt) * inv_rmax);
    }
}

__device__ __forceinline__ void unpack_sh16(const float4 Yv[4], int lmax, float* dst) {
    dst[0] = Yv[0].x; dst[1] = Yv[0].y; dst[2] = Yv[0].z; dst[3] = Yv[0].w;
    if (lmax >= 2) {
        dst[4] = Yv[1].x; dst[5] = Yv[1].y; dst[6] = Yv[1].z; dst[7] = Yv[1].w;
        dst[8] = Yv[2].x;
    }
    if (lmax >= 3) {
        dst[9]  = Yv[2].y; dst[10] = Yv[2].z; dst[11] = Yv[2].w;
        dst[12] = Yv[3].x; dst[13] = Yv[3].y; dst[14] = Yv[3].z; dst[15] = Yv[3].w;
    }
}

__device__ __forceinline__ double eval_zbl_pair_val_d(double r, double rs, double zz, double au) {
    if (r >= rs) return 0.0;
    const double u = r / rs;
    const double u2 = u * u;
    const double u6 = u2 * u2 * u2;
    const double tau = 1.0 - 28.0 * u6 + 48.0 * u6 * u - 21.0 * u6 * u2;
    const double x = r / au;
    const double phi = 0.18175 * exp(-3.19980 * x) + 0.50986 * exp(-0.94229 * x) +
                       0.28022 * exp(-0.40290 * x) + 0.02817 * exp(-0.20162 * x);
    return 0.5 * (14.39964547842567 * zz / r) * phi * tau;
}

__device__ __forceinline__ double eval_zbl_pair_deriv_d(double r, double rs, double zz, double au) {
    if (r >= rs) return 0.0;
    const double inv_rs = 1.0 / rs;
    const double u = r * inv_rs;
    const double u2 = u * u;
    const double u5 = u2 * u2 * u;
    const double u6 = u5 * u;
    const double omu = 1.0 - u;
    const double tau = 1.0 - 28.0 * u6 + 48.0 * u6 * u - 21.0 * u6 * u2;
    const double dtau_dr = (-168.0 * u5 * omu * omu) * inv_rs;
    const double inv_au = 1.0 / au;
    const double x = r * inv_au;
    const double e0 = 0.18175 * exp(-3.19980 * x);
    const double e1 = 0.50986 * exp(-0.94229 * x);
    const double e2 = 0.28022 * exp(-0.40290 * x);
    const double e3 = 0.02817 * exp(-0.20162 * x);
    const double phi = e0 + e1 + e2 + e3;
    const double dphi_dr = -(3.19980 * e0 + 0.94229 * e1 + 0.40290 * e2 + 0.20162 * e3) * inv_au;
    const double inv_r = 1.0 / r;
    return 0.5 * (14.39964547842567 * zz * inv_r) * ((-phi * inv_r + dphi_dr) * tau + phi * dtau_dr);
}

__device__ __forceinline__ void zbl_exact_vec(
    int i, int j, float4 vl, const double* pos64,
    double& rx, double& ry, double& rz, double& rd) {
    rx = (double)vl.x; ry = (double)vl.y; rz = (double)vl.z; rd = (double)fmaxf(vl.w, 1e-8f);
    if (pos64) {
        const double px = pos64[(int64_t)i * 3 + 0] - pos64[(int64_t)j * 3 + 0];
        const double py = pos64[(int64_t)i * 3 + 1] - pos64[(int64_t)j * 3 + 1];
        const double pz = pos64[(int64_t)i * 3 + 2] - pos64[(int64_t)j * 3 + 2];
        if (fabs(px - rx) < 1e-3 && fabs(py - ry) < 1e-3 && fabs(pz - rz) < 1e-3) {
            rx = px; ry = py; rz = pz;
            rd = sqrt(rx * rx + ry * ry + rz * rz);
        }
    }
}

__device__ __noinline__ double3 eval_zbl_force_cold(
    int i, int j, float4 vl, const double* pos64, float wz_i, double rs, double zz, double au) {
    double rx, ry, rz, rd;
    zbl_exact_vec(i, j, vl, pos64, rx, ry, rz, rd);
    const double zbl_er = (double)wz_i * eval_zbl_pair_deriv_d(rd, rs, zz, au);
    const double inv_rd = 1.0 / rd;
    return make_double3((rx * inv_rd) * zbl_er, (ry * inv_rd) * zbl_er, (rz * inv_rd) * zbl_er);
}

// Returning a pair of float4 by value lets register-based edge kernels invoke FP64 close-contact
// Chebyshev evaluation without taking the address of caller local arrays (0 bytes stack frame!).
struct Float4Pair {
    float4 a;
    float4 b;
};

template <bool WITH_DERIV>
__device__ __noinline__ Float4Pair eval_cheb_fp64_4(float r_f, float r_max_f, int k_start) {
    Float4Pair out{{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    const double inv_rmax = 1.0 / (double)r_max_f;
    double t = (double)r_f * inv_rmax;
    if (t < 0.0) t = 0.0;
    if (t >= 1.0) return out;
    const double t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
    const double env = 1.0 - 21.0 * t5 + 35.0 * t5 * t - 15.0 * t5 * t2;
    const double omt = 1.0 - t;
    const double denv_dt = WITH_DERIV ? (-105.0 * t4 * omt * omt) : 0.0;
    const double x = 2.0 * t - 1.0;
    double T0 = 1.0, dT0 = 0.0, T1 = x, dT1 = 1.0;
    float gv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float dv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int k_end = k_start + 4;
    for (int k = 0; k < k_end; ++k) {
        double Tk, dTk = 0.0;
        if (k == 0) {
            Tk = T0; dTk = dT0;
        } else if (k == 1) {
            Tk = T1; dTk = dT1;
        } else {
            Tk = 2.0 * x * T1 - T0;
            if (WITH_DERIV) {
                dTk = 2.0 * T1 + 2.0 * x * dT1 - dT0;
                dT0 = dT1; dT1 = dTk;
            }
            T0 = T1; T1 = Tk;
        }
        if (k >= k_start) {
            const int j = k - k_start;
            gv[j] = (float)(Tk * env);
            if (WITH_DERIV) dv[j] = (float)((2.0 * dTk * env + Tk * denv_dt) * inv_rmax);
        }
    }
    out.a = make_float4(gv[0], gv[1], gv[2], gv[3]);
    out.b = make_float4(dv[0], dv[1], dv[2], dv[3]);
    return out;
}

// Conflict-free strided compile-time Chebyshev writer: writes g[k * STRIDE] (and dg[k * STRIDE]).
template <bool WITH_DERIV, int NB, int STRIDE>
__device__ __forceinline__ void eval_cheb_nb_strided(float r_f, float r_max_f, float* g, float* dg) {
    if (__builtin_expect(r_f < 1.2f, 0)) {
        #pragma unroll
        for (int kb = 0; kb < NB; kb += 4) {
            const Float4Pair p = eval_cheb_fp64_4<WITH_DERIV>(r_f, r_max_f, kb);
            g[(kb + 0) * STRIDE] = p.a.x; g[(kb + 1) * STRIDE] = p.a.y;
            g[(kb + 2) * STRIDE] = p.a.z; g[(kb + 3) * STRIDE] = p.a.w;
            if constexpr (WITH_DERIV) {
                dg[(kb + 0) * STRIDE] = p.b.x; dg[(kb + 1) * STRIDE] = p.b.y;
                dg[(kb + 2) * STRIDE] = p.b.z; dg[(kb + 3) * STRIDE] = p.b.w;
            }
        }
        return;
    }
    float t = __fdiv_rn(r_f, r_max_f);
    if (t < 0.0f) t = 0.0f;
    if (t >= 1.0f) {
        #pragma unroll
        for (int k = 0; k < NB; ++k) {
            g[k * STRIDE] = 0.0f;
            if constexpr (WITH_DERIV) dg[k * STRIDE] = 0.0f;
        }
        return;
    }
    const float inv_rmax = 1.0f / r_max_f;
    const float t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
    const float env = 1.0f - 21.0f * t5 + 35.0f * t5 * t - 15.0f * t5 * t2;
    const float omt = 1.0f - t;
    const float denv_dt = WITH_DERIV ? (-105.0f * t4 * omt * omt) : 0.0f;
    const float x = 2.0f * t - 1.0f;
    float T0 = 1.0f, dT0 = 0.0f;
    g[0] = env;
    if constexpr (WITH_DERIV) dg[0] = denv_dt * inv_rmax;
    float T1 = x, dT1 = 1.0f;
    g[STRIDE] = T1 * env;
    if constexpr (WITH_DERIV) dg[STRIDE] = (2.0f * dT1 * env + T1 * denv_dt) * inv_rmax;
    #pragma unroll
    for (int k = 2; k < NB; ++k) {
        const float Tk = 2.0f * x * T1 - T0;
        g[k * STRIDE] = Tk * env;
        if constexpr (WITH_DERIV) {
            const float dTk = 2.0f * T1 + 2.0f * x * dT1 - dT0;
            dg[k * STRIDE] = (2.0f * dTk * env + Tk * denv_dt) * inv_rmax;
            dT0 = dT1; dT1 = dTk;
        }
        T0 = T1; T1 = Tk;
    }
}

// Exact inline spherical harmonic gradients (matching generated_sh16.cuh bit-for-bit) in pure scalar regs
__device__ __forceinline__ void sh_grad_l1(float d1, float d2, float d3, float& gx, float& gy, float& gz) {
    constexpr float t0 = 0x1.bb67ae0000000p+0f;
    gx = d1 * t0;
    gy = d2 * t0;
    gz = d3 * t0;
}

__device__ __forceinline__ void sh_grad_l2(
    float x, float y, float z,
    float d1, float d2, float d3, float d4, float d5, float d6, float d7, float d8,
    float& gx, float& gy, float& gz) {
    constexpr float t0 = 0x1.bb67ae0000000p+0f;
    constexpr float t1 = 0x1.efbdec0000000p+1f;
    const float t2 = t1 * x;
    const float t3 = 0x1.1e377a0000000p+1f * d6;
    const float t23 = t1 * z;
    const float t24 = 2.0f * y;
    gx = d1 * t0 + d4 * t1 * z + d5 * t1 * y - d8 * t2 - t3 * x;
    gy = d2 * t0 + d5 * t1 * x + d7 * t23 + t24 * t3;
    gz = d3 * t0 + d4 * t2 + d7 * t1 * y + d8 * t23 - t3 * z;
}

__device__ __forceinline__ void sh_grad_l3(
    float x, float y, float z,
    float d1, float d2, float d3, float d4, float d5, float d6, float d7, float d8,
    float d9, float d10, float d11, float d12, float d13, float d14, float d15,
    float& gx, float& gy, float& gz) {
    constexpr float t0 = 0x1.bb67ae0000000p+0f;
    constexpr float t1 = 0x1.efbdec0000000p+1f;
    const float t2 = t1 * x;
    const float t3 = 0x1.1e377a0000000p+1f * d6;
    constexpr float t4 = 0x1.47e7060000000p+3f;
    const float t5 = t4 * y;
    const float t6 = t5 * x;
    constexpr float t7 = 0x1.9ec4740000000p+2f;
    const float t8 = d13 * t7;
    const float t9 = x * z;
    const float t10 = 0.5f * t9;
    const float t11 = 0x1.52a7fa0000000p+1f * d12;
    const float t12 = 3.0f * t11;
    const float t13 = t12 * y;
    constexpr float t14 = 0x1.0bbb300000000p+3f;
    const float t15 = d15 * t14;
    const float t16 = 1.5f * t9;
    const float t17 = x * x;
    const float t18 = 0.75f * t17;
    const float t19 = d11 * t7;
    const float t20 = y * y;
    const float t21 = z * z;
    const float t22 = d9 * t14;
    const float t23 = t1 * z;
    const float t24 = 2.0f * y;
    const float t25 = 0.5f * d14 * t4;
    const float t26 = 1.5f * t11;
    const float t27 = 0.75f * t21;
    gx = d1 * t0 + d10 * t4 * y * z + d11 * t20 * t7 - d14 * t6 + d4 * t1 * z + d5 * t1 * y - d8 * t2 +
         0.75f * d9 * t14 * t21 - t10 * t8 - t13 * x - t15 * t16 - t18 * t19 - t18 * t22 - 0.25f * t19 * t21 - t3 * x;
    gy = d10 * t4 * x * z + d2 * t0 + d5 * t1 * x + d7 * t23 + t12 * t20 - t17 * t25 - t17 * t26 + t19 * t24 * x +
         t21 * t25 - t21 * t26 + t24 * t3 + t24 * t8 * z;
    gz = d10 * t6 + d13 * t20 * t7 + d14 * t5 * z + d3 * t0 + d4 * t2 + d7 * t1 * y + d8 * t23 - t10 * t19 - t13 * z -
         t15 * t18 + t15 * t27 + t16 * t22 - 0.25f * t17 * t8 - t27 * t8 - t3 * z;
}

// =================================================================================================
// Decoupled ZBL Close-Contact Forward & Backward Kernels (early-exit in ~1.5 us when *zbl_flag == 0!)
// Keeps hot L1 edge fwd/bwd kernels 100% free of FP64/ZBL registers and __noinline__ ABI spills.
// =================================================================================================
__global__ void __launch_bounds__(128) ilmlip_zbl_fwd_kernel(
    int N, int S, const int* __restrict__ zbl_flag,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const double* __restrict__ pos64,
    const double* __restrict__ rs_pair, const double* __restrict__ zz_pair, const double* __restrict__ au_pair,
    double* __restrict__ e_zbl_out) {
    if (!(*zbl_flag)) return;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;
    const int z_i = species[i];
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    double acc = 0.0;
    for (int e = eb + lane; e < ee; e += 32) {
        const float4 vl = __ldg(vec_len + e);
        const float r = fmaxf(vl.w, 1e-8f);
        const int j = __ldg(senders + e);
        const int pair = z_i * S + __ldg(species + j);
        const double rs = __ldg(rs_pair + pair);
        if ((double)r < rs + 1e-3) {
            double rx, ry, rz, rd;
            zbl_exact_vec(i, j, vl, pos64, rx, ry, rz, rd);
            if (rd < rs) acc += eval_zbl_pair_val_d(rd, rs, __ldg(zz_pair + pair), __ldg(au_pair + pair));
        }
    }
    const double tot = warp_sum_d(acc);
    if (lane == 0) e_zbl_out[i] = tot;
}

template <bool STRESS>
__global__ void __launch_bounds__(128) ilmlip_zbl_bwd_kernel(
    int N, int S, const int* __restrict__ zbl_flag,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const double* __restrict__ pos64,
    const double* __restrict__ rs_pair, const double* __restrict__ zz_pair, const double* __restrict__ au_pair,
    const float* __restrict__ w_zbl, double* __restrict__ forces, double* __restrict__ virial) {
    if (!(*zbl_flag)) return;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;
    const int z_i = species[i];
    const float wz_i = w_zbl[i];
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    double rfx = 0.0, rfy = 0.0, rfz = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;
    for (int e = eb + lane; e < ee; e += 32) {
        const float4 vl = __ldg(vec_len + e);
        const float r = fmaxf(vl.w, 1e-8f);
        const int j = __ldg(senders + e);
        const int pair = z_i * S + __ldg(species + j);
        const double rs = __ldg(rs_pair + pair);
        if ((double)r < rs + 1e-3) {
            double rx, ry, rz, rd;
            zbl_exact_vec(i, j, vl, pos64, rx, ry, rz, rd);
            if (rd < rs) {
                const double zbl_er = (double)wz_i * eval_zbl_pair_deriv_d(rd, rs, __ldg(zz_pair + pair), __ldg(au_pair + pair));
                const double inv_rd = 1.0 / rd;
                const double fx = (rx * inv_rd) * zbl_er;
                const double fy = (ry * inv_rd) * zbl_er;
                const double fz = (rz * inv_rd) * zbl_er;
                rfx -= fx; rfy -= fy; rfz -= fz;
                atomicAdd(forces + (int64_t)j * 3 + 0, fx);
                atomicAdd(forces + (int64_t)j * 3 + 1, fy);
                atomicAdd(forces + (int64_t)j * 3 + 2, fz);
                if constexpr (STRESS) {
                    v0 += rx * fx;
                    v1 += 0.5 * (rx * fy + ry * fx);
                    v2 += 0.5 * (rx * fz + rz * fx);
                    v3 += ry * fy;
                    v4 += 0.5 * (ry * fz + rz * fy);
                    v5 += rz * fz;
                }
            }
        }
    }
    rfx = warp_sum_d(rfx); rfy = warp_sum_d(rfy); rfz = warp_sum_d(rfz);
    if constexpr (STRESS) {
        v0 = warp_sum_d(v0); v1 = warp_sum_d(v1); v2 = warp_sum_d(v2);
        v3 = warp_sum_d(v3); v4 = warp_sum_d(v4); v5 = warp_sum_d(v5);
    }
    if (lane == 0) {
        if (rfx != 0.0 || rfy != 0.0 || rfz != 0.0) {
            atomicAdd(forces + (int64_t)i * 3 + 0, rfx);
            atomicAdd(forces + (int64_t)i * 3 + 1, rfy);
            atomicAdd(forces + (int64_t)i * 3 + 2, rfz);
        }
        if constexpr (STRESS) {
            if (v0 != 0.0) atomicAdd(virial + 0, v0);
            if (v1 != 0.0) atomicAdd(virial + 1, v1);
            if (v2 != 0.0) atomicAdd(virial + 2, v2);
            if (v3 != 0.0) atomicAdd(virial + 3, v3);
            if (v4 != 0.0) atomicAdd(virial + 4, v4);
            if (v5 != 0.0) atomicAdd(virial + 5, v5);
        }
    }
}

// =================================================================================================
// Kernel 1s: Sum-Swap Layer-1 Edge Forward (S<=4, NB in {8,12,16}, max_nl<=32, ZBL-decoupled!)
// Swaps sum_j and sum_k: accumulates G[z_j, k, lm] = sum_{j in z_j} g_k(r_ij) Y_lm(u_ij) in
// registers with ZERO C1 reads in the edge loop and bank-conflict-free SMEM, then contracts ONCE!
// =================================================================================================
template <int LMAX, int NB, bool S3>
__global__ void __launch_bounds__(128, 6) ilmlip_l1_edge_fwd_sumswap_kernel(
    int N, int S, int n_tot1, Int4 n_l, Int4 l1_off,
    float r_max, float a0_over_sig0, float inv_DA1_sig0_kap_sq,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C1,  // [S*S, NB, n_tot1]
    const float* __restrict__ rs_pair_f32, int* __restrict__ zbl_flag,
    Ptr4 A1, float* __restrict__ f0_out, double* __restrict__ e_zbl_out) {
    constexpr int N_SH = (LMAX + 1) * (LMAX + 1);
    constexpr int NB_STRIDE = NB + 1;           // odd stride (9, 13, 17) -> ZERO SMEM bank conflicts!
    constexpr int SH_STRIDE = N_SH | 1;         // odd stride (9, 17) -> ZERO SMEM bank conflicts!
    constexpr int K_LANES = (NB <= 8) ? 8 : 16;
    constexpr int M_LANES = 32 / K_LANES;
    constexpr int N_MG = (N_SH + M_LANES - 1) / M_LANES;

    union WarpBuf {
        struct {
            float g[16][NB_STRIDE];
            float Y[16][SH_STRIDE];
            int zj[16];
        } e;
        float G[N_SH][4 * NB + 1];
    };
    __shared__ WarpBuf s_wb[4];

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];

    const int my_k = lane & (K_LANES - 1);
    const int my_ms = lane / K_LANES;

    float g0[N_MG], g1[N_MG], g2[N_MG];
    float g3[S3 ? 1 : N_MG];
    #pragma unroll
    for (int mg = 0; mg < N_MG; ++mg) {
        g0[mg] = 0.0f; g1[mg] = 0.0f; g2[mg] = 0.0f;
        if constexpr (!S3) g3[mg] = 0.0f;
    }

    for (int eb_b = eb; eb_b < ee; eb_b += 16) {
        const int bcnt = (ee - eb_b < 16) ? (ee - eb_b) : 16;
        if (lane < bcnt) {
            const int e = eb_b + lane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const int z_j = __ldg(species + j);
            const float r = fmaxf(vl.w, 1e-8f);
            s_wb[wid].e.zj[lane] = z_j;
            if constexpr (NB == 8) {
                float4 g03, g47, d0, d1;
                eval_cheb8_f4<false>(r, r_max, g03, g47, d0, d1);
                float* dst = s_wb[wid].e.g[lane];
                dst[0] = g03.x; dst[1] = g03.y; dst[2] = g03.z; dst[3] = g03.w;
                dst[4] = g47.x; dst[5] = g47.y; dst[6] = g47.z; dst[7] = g47.w;
            } else {
                eval_cheb<false>(r, r_max, NB, s_wb[wid].e.g[lane], nullptr);
            }
            if (rs_pair_f32 && __builtin_expect(r < __ldg(rs_pair_f32 + z_i * S + z_j), 0)) {
                *zbl_flag = 1;
            }
        } else if (lane >= 16 && lane < 16 + bcnt) {
            const int bi = lane - 16;
            const int e = eb_b + bi;
            const float4 vl = __ldg(vec_len + e);
            const float r = fmaxf(vl.w, 1e-8f);
            float4 Yv[4];
            loup::edge_sh16(vl.x, vl.y, vl.z, r, Yv);
            unpack_sh16(Yv, LMAX, s_wb[wid].e.Y[bi]);
        }
        __syncwarp();

        if (my_k < NB) {
            for (int bi = 0; bi < bcnt; ++bi) {
                const int zj = s_wb[wid].e.zj[bi];
                const float gv = s_wb[wid].e.g[bi][my_k];
                const float* Yk = s_wb[wid].e.Y[bi];
                #pragma unroll
                for (int mg = 0; mg < N_MG; ++mg) {
                    const int m = mg * M_LANES + my_ms;
                    if (m < N_SH) {
                        const float prod = gv * Yk[m];
                        if (zj == 0)      g0[mg] += prod;
                        else if (zj == 1) g1[mg] += prod;
                        else if constexpr (S3) {
                            g2[mg] += prod;
                        } else {
                            if (zj == 2) g2[mg] += prod;
                            else         g3[mg] += prod;
                        }
                    }
                }
            }
        }
        __syncwarp();
    }

    if (my_k < NB) {
        #pragma unroll
        for (int mg = 0; mg < N_MG; ++mg) {
            const int m = mg * M_LANES + my_ms;
            if (m < N_SH) {
                s_wb[wid].G[m][0 * NB + my_k] = g0[mg];
                if (S > 1) s_wb[wid].G[m][1 * NB + my_k] = g1[mg];
                if (S > 2) s_wb[wid].G[m][2 * NB + my_k] = g2[mg];
                if constexpr (!S3) {
                    if (S > 3) s_wb[wid].G[m][3 * NB + my_k] = g3[mg];
                }
            }
        }
    }
    __syncwarp();

    // Contract G[m][zj*NB+k] against C1[z_i, zj, k, l, lane] ONCE per atom (unroll 2 -> 0 stack spills!)
    const float* C1_zi = C1 + (int64_t)z_i * S * NB * n_tot1;
    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];
    const int off1 = l1_off.v[1], off2 = l1_off.v[2], off3 = l1_off.v[3];

    float acc0 = 0.0f;
    if (lane < nl0) {
        for (int zj = 0; zj < S; ++zj) {
            const float* Cz = C1_zi + (int64_t)zj * NB * n_tot1 + lane;
            const float* G0 = &s_wb[wid].G[0][zj * NB];
            #pragma unroll 4
            for (int k = 0; k < NB; ++k) acc0 = fmaf(__ldg(Cz + k * n_tot1), G0[k], acc0);
        }
    }

    float acc1_0 = 0.0f, acc1_1 = 0.0f, acc1_2 = 0.0f;
    if (lane < nl1) {
        for (int zj = 0; zj < S; ++zj) {
            const float* Cz = C1_zi + (int64_t)zj * NB * n_tot1 + off1 + lane;
            const int base = zj * NB;
            #pragma unroll 4
            for (int k = 0; k < NB; ++k) {
                const float cv = __ldg(Cz + k * n_tot1);
                acc1_0 = fmaf(cv, s_wb[wid].G[1][base + k], acc1_0);
                acc1_1 = fmaf(cv, s_wb[wid].G[2][base + k], acc1_1);
                acc1_2 = fmaf(cv, s_wb[wid].G[3][base + k], acc1_2);
            }
        }
    }

    float acc2_0 = 0.0f, acc2_1 = 0.0f, acc2_2 = 0.0f, acc2_3 = 0.0f, acc2_4 = 0.0f;
    if (lane < nl2) {
        for (int zj = 0; zj < S; ++zj) {
            const float* Cz = C1_zi + (int64_t)zj * NB * n_tot1 + off2 + lane;
            const int base = zj * NB;
            #pragma unroll 4
            for (int k = 0; k < NB; ++k) {
                const float cv = __ldg(Cz + k * n_tot1);
                acc2_0 = fmaf(cv, s_wb[wid].G[4][base + k], acc2_0);
                acc2_1 = fmaf(cv, s_wb[wid].G[5][base + k], acc2_1);
                acc2_2 = fmaf(cv, s_wb[wid].G[6][base + k], acc2_2);
                acc2_3 = fmaf(cv, s_wb[wid].G[7][base + k], acc2_3);
                acc2_4 = fmaf(cv, s_wb[wid].G[8][base + k], acc2_4);
            }
        }
    }

    float acc3_0 = 0.0f, acc3_1 = 0.0f, acc3_2 = 0.0f, acc3_3 = 0.0f, acc3_4 = 0.0f, acc3_5 = 0.0f, acc3_6 = 0.0f;
    if constexpr (LMAX == 3) {
        if (lane < nl3) {
            for (int zj = 0; zj < S; ++zj) {
                const float* Cz = C1_zi + (int64_t)zj * NB * n_tot1 + off3 + lane;
                const int base = zj * NB;
                #pragma unroll 4
                for (int k = 0; k < NB; ++k) {
                    const float cv = __ldg(Cz + k * n_tot1);
                    acc3_0 = fmaf(cv, s_wb[wid].G[9][base + k], acc3_0);
                    acc3_1 = fmaf(cv, s_wb[wid].G[10][base + k], acc3_1);
                    acc3_2 = fmaf(cv, s_wb[wid].G[11][base + k], acc3_2);
                    acc3_3 = fmaf(cv, s_wb[wid].G[12][base + k], acc3_3);
                    acc3_4 = fmaf(cv, s_wb[wid].G[13][base + k], acc3_4);
                    acc3_5 = fmaf(cv, s_wb[wid].G[14][base + k], acc3_5);
                    acc3_6 = fmaf(cv, s_wb[wid].G[15][base + k], acc3_6);
                }
            }
        }
    }

    float sq_lane = (lane < nl0) ? (acc0 * acc0) : 0.0f;
    if (lane < nl1) sq_lane = fmaf(acc1_0, acc1_0, fmaf(acc1_1, acc1_1, fmaf(acc1_2, acc1_2, sq_lane)));
    if (lane < nl2) {
        sq_lane = fmaf(acc2_0, acc2_0, fmaf(acc2_1, acc2_1, fmaf(acc2_2, acc2_2,
                  fmaf(acc2_3, acc2_3, fmaf(acc2_4, acc2_4, sq_lane)))));
    }
    if constexpr (LMAX == 3) {
        if (lane < nl3) {
            sq_lane = fmaf(acc3_0, acc3_0, fmaf(acc3_1, acc3_1, fmaf(acc3_2, acc3_2,
                      fmaf(acc3_3, acc3_3, fmaf(acc3_4, acc3_4, fmaf(acc3_5, acc3_5, fmaf(acc3_6, acc3_6, sq_lane)))))));
        }
    }
    const float sq_tot = warp_sum_f(sq_lane);
    const float f0 = a0_over_sig0 / sqrtf(1.0f + sq_tot * inv_DA1_sig0_kap_sq);
    if (lane == 0) {
        f0_out[i] = f0;
        if (e_zbl_out) e_zbl_out[i] = 0.0;
    }

    if (lane < nl0) A1.p[0][(int64_t)i * nl0 + lane] = acc0 * f0;
    if (lane < nl1) {
        float* d1 = A1.p[1] + (int64_t)i * 3 * nl1 + lane;
        d1[0] = acc1_0 * f0; d1[nl1] = acc1_1 * f0; d1[2 * nl1] = acc1_2 * f0;
    }
    if (lane < nl2) {
        float* d2 = A1.p[2] + (int64_t)i * 5 * nl2 + lane;
        d2[0] = acc2_0 * f0; d2[nl2] = acc2_1 * f0; d2[2 * nl2] = acc2_2 * f0;
        d2[3 * nl2] = acc2_3 * f0; d2[4 * nl2] = acc2_4 * f0;
    }
    if constexpr (LMAX == 3) {
        if (lane < nl3) {
            float* d3 = A1.p[3] + (int64_t)i * 7 * nl3 + lane;
            d3[0] = acc3_0 * f0; d3[nl3] = acc3_1 * f0; d3[2 * nl3] = acc3_2 * f0;
            d3[3 * nl3] = acc3_3 * f0; d3[4 * nl3] = acc3_4 * f0; d3[5 * nl3] = acc3_5 * f0;
            d3[6 * nl3] = acc3_6 * f0;
        }
    }
}

// =================================================================================================
// Kernel 1a: Packed Fast Layer-1 Edge Forward
// (n_l[0]<=32, n_l[1]<=16, n_l[2]<=16 (or <=8), n_l[3]<=8: packs l>=1 across 32 lanes,
//  optional shared-memory C1 staging, split eval_cheb / edge_sh16, 8-edge warp batching)
// =================================================================================================
template <int LMAX, int NB_MAX, bool ZBL_SW, bool SMEM_C1, bool L2_16>
__global__ void __launch_bounds__(128) ilmlip_l1_edge_fwd_packed_kernel(
    int N, int S, int n_base, int n_tot1, Int4 n_l, Int4 l1_off,
    float r_max, float a0_over_sig0, float inv_DA1_sig0_kap_sq,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const double* __restrict__ pos64,
    const float* __restrict__ C1,  // [S*S, n_base, n_tot1]
    const double* __restrict__ rs_pair, const double* __restrict__ zz_pair, const double* __restrict__ au_pair,
    Ptr4 A1, float* __restrict__ f0_out, double* __restrict__ e_zbl_out) {
    constexpr int N_SH = (LMAX + 1) * (LMAX + 1);
    constexpr int MAX_DM = (LMAX == 3 && !L2_16) ? 7 : 5;
    extern __shared__ float smem_c1[];
    __shared__ float s_g[4][8][NB_MAX];
    __shared__ float s_Y[4][8][N_SH];
    __shared__ int s_pair[4][8];

    if constexpr (SMEM_C1) {
        const int c1_elems = S * S * n_base * n_tot1;
        for (int idx = threadIdx.x; idx < c1_elems; idx += 128) {
            smem_c1[idx] = __ldg(C1 + idx);
        }
        __syncthreads();
    }

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];

    // Map lane 0..31 to primary (l>=1, c) channel:
    // If !L2_16: 0..15 -> l=1, 16..23 -> l=2 (<=8), 24..31 -> l=3 (<=8 if LMAX==3)
    // If  L2_16: 0..15 -> l=1, 16..31 -> l=2 (<=16); if LMAX==3 also lane<n_l[3] -> acc3[7]
    int my_l = 0, my_c = 0, my_col = -1, my_m0 = 0, my_dm = 0;
    if (lane < 16) {
        my_l = 1; my_c = lane;
        if (my_c < n_l.v[1]) { my_col = l1_off.v[1] + my_c; my_m0 = 1; my_dm = 3; }
    } else if (L2_16) {
        my_l = 2; my_c = lane - 16;
        if (my_c < n_l.v[2]) { my_col = l1_off.v[2] + my_c; my_m0 = 4; my_dm = 5; }
    } else if (lane < 24) {
        my_l = 2; my_c = lane - 16;
        if (my_c < n_l.v[2]) { my_col = l1_off.v[2] + my_c; my_m0 = 4; my_dm = 5; }
    } else if constexpr (LMAX == 3) {
        my_l = 3; my_c = lane - 24;
        if (my_c < n_l.v[3]) { my_col = l1_off.v[3] + my_c; my_m0 = 9; my_dm = 7; }
    }
    const bool has_l0 = (lane < n_l.v[0]);
    const bool has_l3 = (L2_16 && LMAX == 3 && lane < n_l.v[3]);
    const float* C1_base = SMEM_C1 ? smem_c1 : C1;

    float acc0 = 0.0f;
    float acc_loc[MAX_DM];
    #pragma unroll
    for (int m = 0; m < MAX_DM; ++m) acc_loc[m] = 0.0f;
    float acc3[7];
    if constexpr (L2_16 && LMAX == 3) {
        #pragma unroll
        for (int m = 0; m < 7; ++m) acc3[m] = 0.0f;
    }
    double e_zbl_acc = 0.0;

    auto dot_c = [&](const float* col, const float* gk) -> float {
        float rv = 0.0f;
        if (n_base == 8) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) rv = fmaf(SMEM_C1 ? col[k * n_tot1] : __ldg(col + k * n_tot1), gk[k], rv);
        } else if (n_base == 12) {
            #pragma unroll
            for (int k = 0; k < 12; ++k) rv = fmaf(SMEM_C1 ? col[k * n_tot1] : __ldg(col + k * n_tot1), gk[k], rv);
        } else if (n_base == 16) {
            #pragma unroll
            for (int k = 0; k < 16; ++k) rv = fmaf(SMEM_C1 ? col[k * n_tot1] : __ldg(col + k * n_tot1), gk[k], rv);
        } else {
            for (int k = 0; k < n_base; ++k) rv = fmaf(SMEM_C1 ? col[k * n_tot1] : __ldg(col + k * n_tot1), gk[k], rv);
        }
        return rv;
    };

    for (int eb_b = eb; eb_b < ee; eb_b += 8) {
        const int bcnt = (ee - eb_b < 8) ? (ee - eb_b) : 8;
        if (lane < bcnt) {
            const int e = eb_b + lane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const int pair = z_i * S + __ldg(species + j);
            const float r = fmaxf(vl.w, 1e-8f);
            s_pair[wid][lane] = pair;
            eval_cheb<false>(r, r_max, n_base, s_g[wid][lane], nullptr);
            if constexpr (ZBL_SW) {
                const double rs = __ldg(rs_pair + pair);
                if (__builtin_expect((double)r < rs, 0)) {
                    double rx, ry, rz, rd;
                    zbl_exact_vec(i, j, vl, pos64, rx, ry, rz, rd);
                    e_zbl_acc += eval_zbl_pair_val_d(rd, rs, __ldg(zz_pair + pair), __ldg(au_pair + pair));
                }
            }
        } else if (lane >= 8 && lane < 8 + bcnt) {
            const int bi = lane - 8;
            const int e = eb_b + bi;
            const float4 vl = __ldg(vec_len + e);
            const float r = fmaxf(vl.w, 1e-8f);
            float4 Yv[4];
            loup::edge_sh16(vl.x, vl.y, vl.z, r, Yv);
            unpack_sh16(Yv, LMAX, s_Y[wid][bi]);
        }
        __syncwarp();

        for (int bi = 0; bi < bcnt; ++bi) {
            const float* Cp = C1_base + (int64_t)s_pair[wid][bi] * n_base * n_tot1;
            const float* gk = s_g[wid][bi];
            const float* Yk = s_Y[wid][bi];
            if (has_l0) {
                const float rv0 = dot_c(Cp + lane, gk);
                acc0 = fmaf(rv0, Yk[0], acc0);
            }
            if (my_col >= 0) {
                const float rv = dot_c(Cp + my_col, gk);
                acc_loc[0] = fmaf(rv, Yk[my_m0 + 0], acc_loc[0]);
                acc_loc[1] = fmaf(rv, Yk[my_m0 + 1], acc_loc[1]);
                acc_loc[2] = fmaf(rv, Yk[my_m0 + 2], acc_loc[2]);
                if (my_dm > 3) {
                    acc_loc[3] = fmaf(rv, Yk[my_m0 + 3], acc_loc[3]);
                    acc_loc[4] = fmaf(rv, Yk[my_m0 + 4], acc_loc[4]);
                }
                if constexpr (LMAX == 3 && !L2_16) {
                    if (my_dm > 5) {
                        acc_loc[5] = fmaf(rv, Yk[my_m0 + 5], acc_loc[5]);
                        acc_loc[6] = fmaf(rv, Yk[my_m0 + 6], acc_loc[6]);
                    }
                }
            }
            if constexpr (L2_16 && LMAX == 3) {
                if (has_l3) {
                    const float rv3 = dot_c(Cp + l1_off.v[3] + lane, gk);
                    #pragma unroll
                    for (int m = 0; m < 7; ++m) acc3[m] = fmaf(rv3, Yk[9 + m], acc3[m]);
                }
            }
        }
        __syncwarp();
    }

    float sq_lane = has_l0 ? (acc0 * acc0) : 0.0f;
    if (my_col >= 0) {
        sq_lane = fmaf(acc_loc[0], acc_loc[0], sq_lane);
        sq_lane = fmaf(acc_loc[1], acc_loc[1], sq_lane);
        sq_lane = fmaf(acc_loc[2], acc_loc[2], sq_lane);
        if (my_dm > 3) {
            sq_lane = fmaf(acc_loc[3], acc_loc[3], sq_lane);
            sq_lane = fmaf(acc_loc[4], acc_loc[4], sq_lane);
        }
        if constexpr (LMAX == 3 && !L2_16) {
            if (my_dm > 5) {
                sq_lane = fmaf(acc_loc[5], acc_loc[5], sq_lane);
                sq_lane = fmaf(acc_loc[6], acc_loc[6], sq_lane);
            }
        }
    }
    if constexpr (L2_16 && LMAX == 3) {
        if (has_l3) {
            #pragma unroll
            for (int m = 0; m < 7; ++m) sq_lane = fmaf(acc3[m], acc3[m], sq_lane);
        }
    }
    const float sq_tot = warp_sum_f(sq_lane);
    const double zbl_tot = ZBL_SW ? warp_sum_d(e_zbl_acc) : 0.0;
    const float f0 = a0_over_sig0 / sqrtf(1.0f + sq_tot * inv_DA1_sig0_kap_sq);
    if (lane == 0) {
        f0_out[i] = f0;
        if (ZBL_SW) e_zbl_out[i] = zbl_tot;
    }

    if (has_l0) {
        A1.p[0][(int64_t)i * n_l.v[0] + lane] = acc0 * f0;
    }
    if (my_col >= 0) {
        const int nl = n_l.v[my_l];
        float* dst = A1.p[my_l] + (int64_t)i * my_dm * nl + my_c;
        dst[0 * nl] = acc_loc[0] * f0;
        dst[1 * nl] = acc_loc[1] * f0;
        dst[2 * nl] = acc_loc[2] * f0;
        if (my_dm > 3) {
            dst[3 * nl] = acc_loc[3] * f0;
            dst[4 * nl] = acc_loc[4] * f0;
        }
        if constexpr (LMAX == 3 && !L2_16) {
            if (my_dm > 5) {
                dst[5 * nl] = acc_loc[5] * f0;
                dst[6 * nl] = acc_loc[6] * f0;
            }
        }
    }
    if constexpr (L2_16 && LMAX == 3) {
        if (has_l3) {
            const int nl3 = n_l.v[3];
            float* dst3 = A1.p[3] + (int64_t)i * 7 * nl3 + lane;
            #pragma unroll
            for (int m = 0; m < 7; ++m) dst3[m * nl3] = acc3[m] * f0;
        }
    }
}

// =================================================================================================
// Kernel 1: Layer-1 Edge Forward (generic fallback for max_nl > 32 or n_l[1]>16 / n_l[2]>8 / n_l[3]>8)
// =================================================================================================
template <int LMAX, int CH_PASSES, int NB_MAX, bool ZBL_SW>
__global__ void __launch_bounds__(128) ilmlip_l1_edge_fwd_kernel(
    int N, int S, int n_base, int n_tot1, Int4 n_l, Int4 l1_off,
    float r_max, float a0_over_sig0, float inv_DA1_sig0_kap_sq,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const double* __restrict__ pos64,
    const float* __restrict__ C1,  // [S*S, n_base, n_tot1]
    const double* __restrict__ rs_pair, const double* __restrict__ zz_pair, const double* __restrict__ au_pair,
    Ptr4 A1, float* __restrict__ f0_out, double* __restrict__ e_zbl_out) {
    constexpr int N_SH = (LMAX + 1) * (LMAX + 1);
    __shared__ float s_g[4][8][NB_MAX];
    __shared__ float s_Y[4][8][N_SH];
    __shared__ int s_pair[4][8];

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];

    float acc[CH_PASSES][N_SH];
    #pragma unroll
    for (int s = 0; s < CH_PASSES; ++s)
        #pragma unroll
        for (int m = 0; m < N_SH; ++m) acc[s][m] = 0.0f;

    double e_zbl_acc = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += 8) {
        const int bcnt = (ee - eb_b < 8) ? (ee - eb_b) : 8;
        if (lane < bcnt) {
            const int e = eb_b + lane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const int pair = z_i * S + __ldg(species + j);
            const float r = fmaxf(vl.w, 1e-8f);
            s_pair[wid][lane] = pair;
            eval_cheb<false>(r, r_max, n_base, s_g[wid][lane], nullptr);
            float4 Yv[4];
            loup::edge_sh16(vl.x, vl.y, vl.z, r, Yv);
            unpack_sh16(Yv, LMAX, s_Y[wid][lane]);
            if constexpr (ZBL_SW) {
                const double rs = __ldg(rs_pair + pair);
                if (__builtin_expect((double)r < rs, 0)) {
                    double rx, ry, rz, rd;
                    zbl_exact_vec(i, j, vl, pos64, rx, ry, rz, rd);
                    e_zbl_acc += eval_zbl_pair_val_d(rd, rs, __ldg(zz_pair + pair), __ldg(au_pair + pair));
                }
            }
        }
        __syncwarp();

        for (int bi = 0; bi < bcnt; ++bi) {
            const float* Cp = C1 + (int64_t)s_pair[wid][bi] * n_base * n_tot1;
            const float* gk = s_g[wid][bi];
            const float* Yk = s_Y[wid][bi];
            #pragma unroll
            for (int s = 0; s < CH_PASSES; ++s) {
                const int c = lane + s * 32;
                #pragma unroll
                for (int l = 0; l <= LMAX; ++l) {
                    if (c < n_l.v[l]) {
                        float rv = 0.0f;
                        const float* col = Cp + l1_off.v[l] + c;
                        for (int k = 0; k < n_base; ++k) {
                            rv = fmaf(__ldg(col + k * n_tot1), gk[k], rv);
                        }
                        const int m0 = l * l;
                        const int dm = 2 * l + 1;
                        #pragma unroll
                        for (int m = 0; m < dm; ++m) {
                            acc[s][m0 + m] = fmaf(rv, Yk[m0 + m], acc[s][m0 + m]);
                        }
                    }
                }
            }
        }
        __syncwarp();
    }

    float sq_lane = 0.0f;
    #pragma unroll
    for (int s = 0; s < CH_PASSES; ++s) {
        const int c = lane + s * 32;
        #pragma unroll
        for (int l = 0; l <= LMAX; ++l) {
            if (c < n_l.v[l]) {
                const int m0 = l * l;
                const int dm = 2 * l + 1;
                #pragma unroll
                for (int m = 0; m < dm; ++m) {
                    const float v = acc[s][m0 + m];
                    sq_lane = fmaf(v, v, sq_lane);
                }
            }
        }
    }
    const float sq_tot = warp_sum_f(sq_lane);
    const double zbl_tot = ZBL_SW ? warp_sum_d(e_zbl_acc) : 0.0;
    const float f0 = a0_over_sig0 / sqrtf(1.0f + sq_tot * inv_DA1_sig0_kap_sq);
    if (lane == 0) {
        f0_out[i] = f0;
        if (ZBL_SW) e_zbl_out[i] = zbl_tot;
    }

    #pragma unroll
    for (int s = 0; s < CH_PASSES; ++s) {
        const int c = lane + s * 32;
        #pragma unroll
        for (int l = 0; l <= LMAX; ++l) {
            const int nl = n_l.v[l];
            if (c < nl) {
                const int m0 = l * l;
                const int dm = 2 * l + 1;
                float* dst = A1.p[l] + (int64_t)i * dm * nl + c;
                #pragma unroll
                for (int m = 0; m < dm; ++m) {
                    dst[m * nl] = acc[s][m0 + m] * f0;
                }
            }
        }
    }
}

// =================================================================================================
// Kernel 2f: SMEM-Staged 256-Thread FUSED Layer-1 Node Forward + W_msg_0 & W_msg_1 + float4 I_pack4!
// Stages W_msg0 + W_msg1 ONCE per 8 atoms (256 threads) in dynamic SMEM (13.8 KB for l2, 17.9 KB
// for l3) -> zero L1 thrashing, zero Bcat DRAM traffic, writes float4 I_pack4 for 1x LDG.128 L2 fwd!
// =================================================================================================
template <int LMAX>
__global__ void __launch_bounds__(256) ilmlip_l1_node_fwd_lf1_msg_kernel(
    int N, int n_feat, int n_msg, Int4 n_l, Int4 msg_in_dim,
    CPtr4 A1, const float* __restrict__ W_msg0, const float* __restrict__ W_msg1,
    float* __restrict__ x_out, float4* __restrict__ I_pack4, Ptr4 I_out) {
    extern __shared__ float smem_w[];
    const int m0d = msg_in_dim.v[0], m1d = msg_in_dim.v[1];
    const int w0_sz = m0d * n_msg;
    const int w1_sz = m1d * n_msg;
    float* sW0 = smem_w;
    float* sW1 = smem_w + w0_sz;

    const int tid = threadIdx.x;
    for (int idx = tid; idx < w0_sz; idx += 256) sW0[idx] = __ldg(W_msg0 + idx);
    for (int idx = tid; idx < w1_sz; idx += 256) sW1[idx] = __ldg(W_msg1 + idx);
    __syncthreads();

    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int i = blockIdx.x * 8 + wid;
    if (i >= N) return;

    const int c = lane;
    const int q = lane & 15;
    const int half = lane >> 4;
    const bool active_q = (q < n_msg);

    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];
    float* xi = x_out + (int64_t)i * n_feat;

    const float a0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
    if (c < nl1) {
        const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c;
        a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1];
    }
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    if (c < nl2) {
        const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c;
        a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2];
    }
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c;
            a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3];
            a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3];
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

    float acc_i0 = 0.0f, acc_i1_0 = 0.0f, acc_i1_1 = 0.0f, acc_i1_2 = 0.0f;
    auto add_b0 = [&](float val, int b0off, int lim) {
        const float* wbase = sW0 + b0off * n_msg + q;
        for (int k = 0; k < lim; k += 2) {
            const int kk = k + half;
            const float v = __shfl_sync(0xffffffffu, val, (kk < lim) ? kk : 0);
            if (active_q && kk < lim) acc_i0 = fmaf(v, wbase[kk * n_msg], acc_i0);
        }
    };
    auto add_b1 = [&](float v0, float v1, float v2, int b1off, int lim) {
        const float* wbase = sW1 + b1off * n_msg + q;
        for (int k = 0; k < lim; k += 2) {
            const int kk = k + half;
            const int src = (kk < lim) ? kk : 0;
            const float s0 = __shfl_sync(0xffffffffu, v0, src);
            const float s1 = __shfl_sync(0xffffffffu, v1, src);
            const float s2 = __shfl_sync(0xffffffffu, v2, src);
            if (active_q && kk < lim) {
                const float w = wbase[kk * n_msg];
                acc_i1_0 = fmaf(s0, w, acc_i1_0);
                acc_i1_1 = fmaf(s1, w, acc_i1_1);
                acc_i1_2 = fmaf(s2, w, acc_i1_2);
            }
        }
    };

    if (c < nl0) xi[c] = a0;
    add_b0(a0, 0, nl0);
    add_b1(a1_0, a1_1, a1_2, 0, nl1);

    int xoff = nl0, b0off = nl0, b1off = nl1;

    const float b0 = a0 * a0;
    if (c < nl0) xi[xoff + c] = b0;
    add_b0(b0, b0off, nl0);
    xoff += nl0; b0off += nl0;

    const float b1_0 = a0 * a1_0, b1_1 = a0 * a1_1, b1_2 = a0 * a1_2;
    add_b1(b1_0, b1_1, b1_2, b1off, c01);
    b1off += c01;

    constexpr float C110 = 0.5773502588272095f;
    const float b2 = C110 * (a1_0 * a1_0 + a1_1 * a1_1 + a1_2 * a1_2);
    if (c < nl1) xi[xoff + c] = b2;
    add_b0(b2, b0off, nl1);
    xoff += nl1; b0off += nl1;

    constexpr float CS310 = 0.547722578048706f;
    constexpr float CS110 = 0.3162277638912201f;
    constexpr float CS410 = 0.6324555277824402f;
    const float b3_0 = CS310 * (a1_2 * a2_0 + a1_1 * a2_1 - a1_0 * a2_4) - CS110 * a1_0 * a2_2;
    const float b3_1 = CS310 * (a1_0 * a2_1 + a1_2 * a2_3) + CS410 * a1_1 * a2_2;
    const float b3_2 = CS310 * (a1_0 * a2_0 + a1_1 * a2_3 + a1_2 * a2_4) - CS110 * a1_2 * a2_2;
    add_b1(b3_0, b3_1, b3_2, b1off, c12);
    b1off += c12;

    constexpr float C220 = 0.4472135901451111f;
    const float b4 = C220 * (a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4);
    if (c < nl2) xi[xoff + c] = b4;
    add_b0(b4, b0off, nl2);
    xoff += nl2; b0off += nl2;

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
        add_b1(b5_0, b5_1, b5_2, b1off, c23);
        b1off += c23;

        b6 = C330 * (a3_0 * a3_0 + a3_1 * a3_1 + a3_2 * a3_2 + a3_3 * a3_3 + a3_4 * a3_4 + a3_5 * a3_5 + a3_6 * a3_6);
        if (c < nl3) xi[xoff + c] = b6;
        add_b0(b6, b0off, nl3);
        xoff += nl3; b0off += nl3;
    }

    {
        const float v = b0 * a0;
        if (c < nl0) xi[xoff + c] = v;
        add_b0(v, b0off, nl0);
        xoff += nl0; b0off += nl0;
    }
    {
        const float v = C110 * (b1_0 * a1_0 + b1_1 * a1_1 + b1_2 * a1_2);
        if (c < c01) xi[xoff + c] = v;
        add_b0(v, b0off, c01);
        xoff += c01; b0off += c01;
    }
    {
        const float v = C110 * (b3_0 * a1_0 + b3_1 * a1_1 + b3_2 * a1_2);
        if (c < c12) xi[xoff + c] = v;
        add_b0(v, b0off, c12);
        xoff += c12; b0off += c12;
    }
    {
        const float v = b4 * a0;
        if (c < c02) xi[xoff + c] = v;
        add_b0(v, b0off, c02);
        xoff += c02; b0off += c02;
    }
    if constexpr (LMAX == 3) {
        {
            const float v = C110 * (b5_0 * a1_0 + b5_1 * a1_1 + b5_2 * a1_2);
            if (c < c123) xi[xoff + c] = v;
            add_b0(v, b0off, c123);
            xoff += c123; b0off += c123;
        }
        {
            const float v = b6 * a0;
            if (c < c03) xi[xoff + c] = v;
            add_b0(v, b0off, c03);
            xoff += c03; b0off += c03;
        }
    }

    add_b1(b0 * a1_0, b0 * a1_1, b0 * a1_2, b1off, c01); b1off += c01;
    add_b1(b2 * a1_0, b2 * a1_1, b2 * a1_2, b1off, nl1); b1off += nl1;
    add_b1(b3_0 * a0, b3_1 * a0, b3_2 * a0, b1off, c012); b1off += c012;
    add_b1(b4 * a1_0, b4 * a1_1, b4 * a1_2, b1off, c12); b1off += c12;
    if constexpr (LMAX == 3) {
        add_b1(b5_0 * a0, b5_1 * a0, b5_2 * a0, b1off, c023); b1off += c023;
        add_b1(b6 * a1_0, b6 * a1_1, b6 * a1_2, b1off, c13);
    }

    acc_i0   += __shfl_xor_sync(0xffffffffu, acc_i0, 16);
    acc_i1_0 += __shfl_xor_sync(0xffffffffu, acc_i1_0, 16);
    acc_i1_1 += __shfl_xor_sync(0xffffffffu, acc_i1_1, 16);
    acc_i1_2 += __shfl_xor_sync(0xffffffffu, acc_i1_2, 16);
    if (half == 0 && active_q) {
        if (I_pack4) {
            I_pack4[(int64_t)i * n_msg + q] = make_float4(acc_i0, acc_i1_0, acc_i1_1, acc_i1_2);
        } else {
            I_out.p[0][(int64_t)i * n_msg + q] = acc_i0;
            float* d1 = I_out.p[1] + (int64_t)i * 3 * n_msg + q;
            d1[0] = acc_i1_0; d1[n_msg] = acc_i1_1; d1[2 * n_msg] = acc_i1_2;
        }
    }
}

// =================================================================================================
// Kernel 2s: Zero-Spill Register-Specialized Layer-1 Node Forward (lfeat==1, lmsg==1, LMAX in {2,3})
// 0 bytes stack frame, 0 SMEM, 0 CG table lookups — pure scalar register arithmetic!
// =================================================================================================
template <int LMAX>
__global__ void __launch_bounds__(256) ilmlip_l1_node_fwd_lf1_kernel(
    int N, int n_feat, Int4 n_l, Int4 msg_in_dim,
    CPtr4 A1, float* __restrict__ x_out, Ptr4 Bcat) {
    const int c = blockIdx.y * 32 + threadIdx.x;
    const int i = blockIdx.x * 8 + threadIdx.y;
    if (i >= N) return;

    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];
    const int m0d = msg_in_dim.v[0], m1d = msg_in_dim.v[1];
    float* xi = x_out + (int64_t)i * n_feat;
    float* bc0 = Bcat.p[0] + (int64_t)i * m0d;
    float* bc1 = Bcat.p[1] + (int64_t)i * 3 * m1d;

    const float a0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
    if (c < nl1) {
        const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c;
        a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1];
    }
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    if (c < nl2) {
        const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c;
        a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2];
    }
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c;
            a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3];
            a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3];
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

    // nu = 1
    if (c < nl0) { xi[c] = a0; bc0[c] = a0; }
    if (c < nl1) { bc1[c] = a1_0; bc1[m1d + c] = a1_1; bc1[2 * m1d + c] = a1_2; }

    // nu = 2
    int xoff = nl0;
    int b0off = nl0;
    int b1off = nl1;

    // p0: (0,0->0)
    const float b0 = a0 * a0;
    if (c < nl0) { xi[xoff + c] = b0; bc0[b0off + c] = b0; }
    xoff += nl0; b0off += nl0;

    // p1: (0,1->1)
    const float b1_0 = a0 * a1_0, b1_1 = a0 * a1_1, b1_2 = a0 * a1_2;
    if (c < c01) { bc1[b1off + c] = b1_0; bc1[m1d + b1off + c] = b1_1; bc1[2 * m1d + b1off + c] = b1_2; }
    b1off += c01;

    // p2: (1,1->0)
    constexpr float C110 = 0.5773502588272095f;
    const float b2 = C110 * (a1_0 * a1_0 + a1_1 * a1_1 + a1_2 * a1_2);
    if (c < nl1) { xi[xoff + c] = b2; bc0[b0off + c] = b2; }
    xoff += nl1; b0off += nl1;

    // p3: (1,2->1)
    constexpr float CS310 = 0.547722578048706f;
    constexpr float CS110 = 0.3162277638912201f;
    constexpr float CS410 = 0.6324555277824402f;
    const float b3_0 = CS310 * (a1_2 * a2_0 + a1_1 * a2_1 - a1_0 * a2_4) - CS110 * a1_0 * a2_2;
    const float b3_1 = CS310 * (a1_0 * a2_1 + a1_2 * a2_3) + CS410 * a1_1 * a2_2;
    const float b3_2 = CS310 * (a1_0 * a2_0 + a1_1 * a2_3 + a1_2 * a2_4) - CS110 * a1_2 * a2_2;
    if (c < c12) { bc1[b1off + c] = b3_0; bc1[m1d + b1off + c] = b3_1; bc1[2 * m1d + b1off + c] = b3_2; }
    b1off += c12;

    // p4: (2,2->0)
    constexpr float C220 = 0.4472135901451111f;
    const float b4 = C220 * (a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4);
    if (c < nl2) { xi[xoff + c] = b4; bc0[b0off + c] = b4; }
    xoff += nl2; b0off += nl2;

    float b5_0 = 0.0f, b5_1 = 0.0f, b5_2 = 0.0f, b6 = 0.0f;
    if constexpr (LMAX == 3) {
        // p5: (2,3->1)
        constexpr float K1 = 0.4629100561141968f;
        constexpr float K2 = 0.37796446681022644f;
        constexpr float K3 = 0.11952286213636398f;
        constexpr float K4 = 0.47809144854545593f;
        constexpr float K5 = 0.29277002811431885f;
        constexpr float K6 = 0.41403934359550476f;
        constexpr float K7 = 0.5070925354957581f;
        b5_0 = K1 * (a2_4 * a3_0 - a2_0 * a3_6) + K2 * (a2_3 * a3_1 - a2_1 * a3_5) +
               K3 * (a2_4 * a3_2 - a2_0 * a3_4) - K5 * a2_1 * a3_3 + K6 * a2_2 * a3_2;
        b5_1 = K2 * (a2_0 * a3_1 + a2_4 * a3_5) + K4 * (a2_1 * a3_2 + a2_3 * a3_4) + K7 * a2_2 * a3_3;
        b5_2 = K1 * (a2_0 * a3_0 + a2_4 * a3_6) + K2 * (a2_1 * a3_1 + a2_3 * a3_5) -
               K3 * (a2_0 * a3_2 + a2_4 * a3_4) - K5 * a2_3 * a3_3 + K6 * a2_2 * a3_4;
        if (c < c23) { bc1[b1off + c] = b5_0; bc1[m1d + b1off + c] = b5_1; bc1[2 * m1d + b1off + c] = b5_2; }
        b1off += c23;

        // p6: (3,3->0)
        constexpr float C330 = 0.37796446681022644f;
        b6 = C330 * (a3_0 * a3_0 + a3_1 * a3_1 + a3_2 * a3_2 + a3_3 * a3_3 + a3_4 * a3_4 + a3_5 * a3_5 + a3_6 * a3_6);
        if (c < nl3) { xi[xoff + c] = b6; bc0[b0off + c] = b6; }
        xoff += nl3; b0off += nl3;
    }

    // nu = 3, L = 0: (p0,0), (p1,1), (p3,1), (p4,0), [LMAX==3: (p5,1), (p6,0)]
    if (c < nl0) { const float v = b0 * a0; xi[xoff + c] = v; bc0[b0off + c] = v; }
    xoff += nl0; b0off += nl0;

    if (c < c01) {
        const float v = C110 * (b1_0 * a1_0 + b1_1 * a1_1 + b1_2 * a1_2);
        xi[xoff + c] = v; bc0[b0off + c] = v;
    }
    xoff += c01; b0off += c01;

    if (c < c12) {
        const float v = C110 * (b3_0 * a1_0 + b3_1 * a1_1 + b3_2 * a1_2);
        xi[xoff + c] = v; bc0[b0off + c] = v;
    }
    xoff += c12; b0off += c12;

    if (c < c02) { const float v = b4 * a0; xi[xoff + c] = v; bc0[b0off + c] = v; }
    xoff += c02; b0off += c02;

    if constexpr (LMAX == 3) {
        if (c < c123) {
            const float v = C110 * (b5_0 * a1_0 + b5_1 * a1_1 + b5_2 * a1_2);
            xi[xoff + c] = v; bc0[b0off + c] = v;
        }
        xoff += c123; b0off += c123;

        if (c < c03) { const float v = b6 * a0; xi[xoff + c] = v; bc0[b0off + c] = v; }
        xoff += c03; b0off += c03;
    }

    // nu = 3, L = 1: (p0,1), (p2,1), (p3,0), (p4,1), [LMAX==3: (p5,0), (p6,1)]
    if (c < c01) { bc1[b1off + c] = b0 * a1_0; bc1[m1d + b1off + c] = b0 * a1_1; bc1[2 * m1d + b1off + c] = b0 * a1_2; }
    b1off += c01;

    if (c < nl1) { bc1[b1off + c] = b2 * a1_0; bc1[m1d + b1off + c] = b2 * a1_1; bc1[2 * m1d + b1off + c] = b2 * a1_2; }
    b1off += nl1;

    if (c < c012) { bc1[b1off + c] = b3_0 * a0; bc1[m1d + b1off + c] = b3_1 * a0; bc1[2 * m1d + b1off + c] = b3_2 * a0; }
    b1off += c012;

    if (c < c12) { bc1[b1off + c] = b4 * a1_0; bc1[m1d + b1off + c] = b4 * a1_1; bc1[2 * m1d + b1off + c] = b4 * a1_2; }
    b1off += c12;

    if constexpr (LMAX == 3) {
        if (c < c023) { bc1[b1off + c] = b5_0 * a0; bc1[m1d + b1off + c] = b5_1 * a0; bc1[2 * m1d + b1off + c] = b5_2 * a0; }
        b1off += c023;

        if (c < c13) { bc1[b1off + c] = b6 * a1_0; bc1[m1d + b1off + c] = b6 * a1_1; bc1[2 * m1d + b1off + c] = b6 * a1_2; }
    }
}

// =================================================================================================
// Kernel 2: Layer-1 Node Forward (generic fallback: SMEM a1 + b2 -> 0 bytes stack frame!)
// =================================================================================================
template <int LMAX, bool SMEM_TBL>
__global__ void ilmlip_l1_node_fwd_kernel(
    int N, int lmsg, int n_nu2, int n_nu3, int cg_total, int n_feat, int b2_scalars,
    Int4 n_l, Int4 msg_in_dim,
    const int32_t* __restrict__ nu2,  // [n_nu2, 9]: l1, l2, L, c, cg_off, cg_cnt, b2_off, x1_off, bcat_off
    const int32_t* __restrict__ nu3,  // [n_nu3, 8]: b2_off, l3, L, c, cg_off, cg_cnt, x1_off, bcat_off
    const int32_t* __restrict__ cg_idx, const float* __restrict__ cg_val,
    CPtr4 A1, float* __restrict__ x_out, Ptr4 Bcat) {
    constexpr int N_SH = (LMAX + 1) * (LMAX + 1);
    extern __shared__ float smem[];
    const int nu2_ints = n_nu2 * 9;
    const int nu3_ints = n_nu3 * 8;
    int32_t* s_nu2 = reinterpret_cast<int32_t*>(smem);
    int32_t* s_nu3 = s_nu2 + nu2_ints;
    int32_t* s_cg_idx = s_nu3 + nu3_ints;
    float* s_cg_val = reinterpret_cast<float*>(s_cg_idx + cg_total);
    float* work = SMEM_TBL ? (s_cg_val + cg_total) : smem;

    if constexpr (SMEM_TBL) {
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int nthr = blockDim.x * blockDim.y;
        for (int k = tid; k < nu2_ints; k += nthr) s_nu2[k] = __ldg(nu2 + k);
        for (int k = tid; k < nu3_ints; k += nthr) s_nu3[k] = __ldg(nu3 + k);
        for (int k = tid; k < cg_total; k += nthr) {
            s_cg_idx[k] = __ldg(cg_idx + k);
            s_cg_val[k] = __ldg(cg_val + k);
        }
        __syncthreads();
    }

    const int32_t* tbl_nu2 = SMEM_TBL ? s_nu2 : nu2;
    const int32_t* tbl_nu3 = SMEM_TBL ? s_nu3 : nu3;
    const int32_t* tbl_cgi = SMEM_TBL ? s_cg_idx : cg_idx;
    const float*   tbl_cgv = SMEM_TBL ? s_cg_val : cg_val;

    const int c_local = threadIdx.x;
    const int ay = threadIdx.y;
    const int i = blockIdx.x * blockDim.y + ay;
    const int c = blockIdx.y * 32 + c_local;
    if (i >= N) return;

    // Stage a1 and b2 in bank-conflict-free SMEM ([ay][slot][c_local]) -> 0 bytes stack frame!
    float* a1_s = work + ((size_t)ay * (N_SH + b2_scalars)) * 32 + c_local;
    float* b2_s = a1_s + (size_t)N_SH * 32;

    #pragma unroll
    for (int l = 0; l <= LMAX; ++l) {
        const int nl = n_l.v[l];
        const int dm = 2 * l + 1;
        const int m0 = l * l;
        if (c < nl) {
            const float* src = A1.p[l] + (int64_t)i * dm * nl + c;
            #pragma unroll
            for (int m = 0; m < dm; ++m) a1_s[(m0 + m) * 32] = src[m * nl];
            if (l == 0) x_out[(int64_t)i * n_feat + c] = a1_s[0];
            if (l <= lmsg) {
                float* dst = Bcat.p[l] + (int64_t)i * dm * msg_in_dim.v[l] + c;
                #pragma unroll
                for (int m = 0; m < dm; ++m) dst[m * msg_in_dim.v[l]] = a1_s[(m0 + m) * 32];
            }
        } else {
            #pragma unroll
            for (int m = 0; m < dm; ++m) a1_s[(m0 + m) * 32] = 0.0f;
        }
    }

    for (int p = 0; p < n_nu2; ++p) {
        const int32_t* row = tbl_nu2 + p * 9;
        const int c_lim = row[3];
        if (c >= c_lim) continue;
        const int l1 = row[0], l2 = row[1], L = row[2];
        const int cg_off = row[4], cg_cnt = row[5], b2_off = row[6], x1_off = row[7], bcat_off = row[8];
        const int m01 = l1 * l1, m02 = l2 * l2;
        if (L == 0) {
            float out0 = 0.0f;
            for (int k = 0; k < cg_cnt; ++k) {
                const int pk = tbl_cgi[cg_off + k];
                out0 = fmaf(tbl_cgv[cg_off + k], a1_s[(m01 + (pk & 15)) * 32] * a1_s[(m02 + ((pk >> 4) & 15)) * 32], out0);
            }
            b2_s[b2_off * 32] = out0;
            x_out[(int64_t)i * n_feat + x1_off + c] = out0;
            if (0 <= lmsg) Bcat.p[0][(int64_t)i * msg_in_dim.v[0] + bcat_off + c] = out0;
        } else {
            const int dm = 2 * L + 1;
            float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f, out4 = 0.0f;
            for (int k = 0; k < cg_cnt; ++k) {
                const int pk = tbl_cgi[cg_off + k];
                const int m1 = pk & 15, m2 = (pk >> 4) & 15, m3 = (pk >> 8) & 15;
                const float prod = tbl_cgv[cg_off + k] * (a1_s[(m01 + m1) * 32] * a1_s[(m02 + m2) * 32]);
                if (m3 == 0) out0 += prod;
                else if (m3 == 1) out1 += prod;
                else if (m3 == 2) out2 += prod;
                else if (m3 == 3) out3 += prod;
                else out4 += prod;
            }
            b2_s[(b2_off + 0) * 32] = out0;
            b2_s[(b2_off + 1) * 32] = out1;
            b2_s[(b2_off + 2) * 32] = out2;
            if (dm > 3) { b2_s[(b2_off + 3) * 32] = out3; b2_s[(b2_off + 4) * 32] = out4; }
            if (L <= lmsg) {
                float* dst = (L == 1 ? Bcat.p[1] : Bcat.p[2]) + (int64_t)i * dm * (L == 1 ? msg_in_dim.v[1] : msg_in_dim.v[2]) + bcat_off + c;
                const int stL = (L == 1 ? msg_in_dim.v[1] : msg_in_dim.v[2]);
                dst[0] = out0; dst[stL] = out1; dst[2 * stL] = out2;
                if (dm > 3) { dst[3 * stL] = out3; dst[4 * stL] = out4; }
            }
        }
    }

    for (int q = 0; q < n_nu3; ++q) {
        const int32_t* row = tbl_nu3 + q * 8;
        const int c_lim = row[3];
        if (c >= c_lim) continue;
        const int b2_off = row[0], l3 = row[1], L = row[2];
        const int cg_off = row[4], cg_cnt = row[5], x1_off = row[6], bcat_off = row[7];
        const int m03 = l3 * l3;
        if (L == 0) {
            float out0 = 0.0f;
            for (int k = 0; k < cg_cnt; ++k) {
                const int pk = tbl_cgi[cg_off + k];
                out0 = fmaf(tbl_cgv[cg_off + k], b2_s[(b2_off + (pk & 15)) * 32] * a1_s[(m03 + ((pk >> 4) & 15)) * 32], out0);
            }
            x_out[(int64_t)i * n_feat + x1_off + c] = out0;
            if (0 <= lmsg) Bcat.p[0][(int64_t)i * msg_in_dim.v[0] + bcat_off + c] = out0;
        } else {
            const int dm = 2 * L + 1;
            float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f, out4 = 0.0f;
            for (int k = 0; k < cg_cnt; ++k) {
                const int pk = tbl_cgi[cg_off + k];
                const int m12 = pk & 15, m3 = (pk >> 4) & 15, mL = (pk >> 8) & 15;
                const float prod = tbl_cgv[cg_off + k] * (b2_s[(b2_off + m12) * 32] * a1_s[(m03 + m3) * 32]);
                if (mL == 0) out0 += prod;
                else if (mL == 1) out1 += prod;
                else if (mL == 2) out2 += prod;
                else if (mL == 3) out3 += prod;
                else out4 += prod;
            }
            if (L <= lmsg) {
                const int stL = (L == 1 ? msg_in_dim.v[1] : msg_in_dim.v[2]);
                float* dst = (L == 1 ? Bcat.p[1] : Bcat.p[2]) + (int64_t)i * dm * stL + bcat_off + c;
                dst[0] = out0; dst[stL] = out1; dst[2 * stL] = out2;
                if (dm > 3) { dst[3 * stL] = out3; dst[4 * stL] = out4; }
            }
        }
    }
}

// =================================================================================================
// Kernel 3s: 1-Pass Block-SMEM C2+Cbf Layer-2 Edge Forward (256 threads = 16 atoms/CTA, STACK:0!)
// (lfeat2==1, lmsg==1, lout2==2, n_msg<=16, n_base==8, S<=4)
// Stages interleaved float2 s_C2 + s_Cbf ONCE per 16 atoms in dynamic SMEM (18 KB for S=3),
// eliminates the 3x species edge re-scan, frees 32 regs (STACK:32 -> 0, occ 46% -> 67%),
// and software-pipelines 1x LDG.128 float4 I_pack4 loads!
// =================================================================================================
template <bool REM_BF>
__global__ void __launch_bounds__(256, 4) ilmlip_l2_edge_fwd_sp_kernel(
    int N, int S, int n_msg,
    Int4 l2_c, float r_max, float a0_over_sig1, float inv_DA2_sig1_kap_sq,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2,   // [S*S, 8, 2*n_msg]
    const float* __restrict__ Cbf,  // [S*S, 8, 2*n_msg]
    const float* __restrict__ f0, const float4* __restrict__ I_pack4, CPtr4 I_node,
    Ptr4 A2, float* __restrict__ f1_out) {
    extern __shared__ float2 smem_c2[];
    const int c2_f2 = S * S * 8 * 16;
    const int l2_tot_cols = 2 * n_msg;
    for (int idx = threadIdx.x; idx < c2_f2; idx += 256) {
        const int q = idx & 15;
        if (q < n_msg) {
            const int pk = idx >> 4;
            const float* src = C2 + (int64_t)pk * l2_tot_cols + q;
            smem_c2[idx] = make_float2(__ldg(src), __ldg(src + n_msg));
        }
    }
    if constexpr (REM_BF) {
        float2* sCbf = smem_c2 + c2_f2;
        for (int idx = threadIdx.x; idx < c2_f2; idx += 256) {
            const int q = idx & 15;
            if (q < n_msg) {
                const int pk = idx >> 4;
                const float* src = Cbf + (int64_t)pk * l2_tot_cols + q;
                sCbf[idx] = make_float2(__ldg(src), __ldg(src + n_msg));
            }
        }
    }
    __syncthreads();

    __shared__ float4 s_g03[16][5];
    __shared__ float4 s_g47[16][5];
    __shared__ float4 s_Y[16][5];
    __shared__ float  s_f0[16][5];
    __shared__ int    s_j[16][5];
    __shared__ int    s_pair[16][5];

    const int sid = threadIdx.x >> 4;
    const int sub = sid & 1;
    const int slane = threadIdx.x & 15;
    const unsigned smask = sub ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 16 + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    const bool active_q = (slane < n_msg);
    const float4* I4_base = I_pack4 ? (I_pack4 + slane) : nullptr;
    const float* I0_base = I_node.p[0] + slane;
    const float* I1_base = I_node.p[1] + slane;
    const float2* sC2_q  = smem_c2 + slane;
    const float2* sCbf_q = REM_BF ? (smem_c2 + c2_f2 + slane) : nullptr;

    auto load_I = [&](int j_idx) -> float4 {
        if (I4_base) return __ldg(I4_base + (int64_t)j_idx * n_msg);
        const float* pI1 = I1_base + (int64_t)j_idx * 3 * n_msg;
        return make_float4(__ldg(I0_base + (int64_t)j_idx * n_msg),
                           __ldg(pI1), __ldg(pI1 + n_msg), __ldg(pI1 + 2 * n_msg));
    };

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    float acc8 = 0.0f, acc9 = 0.0f, acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f;

    for (int eb_b = eb; eb_b < ee; eb_b += 4) {
        const int bcnt = (ee - eb_b < 4) ? (ee - eb_b) : 4;
        if (slane < bcnt) {
            const int e = eb_b + slane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const float r = fmaxf(vl.w, 1e-8f);
            const float sc = 0x1.bb67ae0000000p+0f / r;
            s_j[sid][slane] = j;
            s_pair[sid][slane] = z_i * S + __ldg(species + j);
            s_Y[sid][slane] = make_float4(1.0f, vl.x * sc, vl.y * sc, vl.z * sc);
            if constexpr (REM_BF) s_f0[sid][slane] = __ldg(f0 + j);
            float4 g03, g47, dummy0, dummy1;
            eval_cheb8_f4<false>(r, r_max, g03, g47, dummy0, dummy1);
            s_g03[sid][slane] = g03;
            s_g47[sid][slane] = g47;
        }
        __syncwarp(smask);

        if (active_q && bcnt > 0) {
            float4 li = load_I(s_j[sid][0]);

            for (int bi = 0; bi < bcnt; ++bi) {
                const int poff = s_pair[sid][bi] << 7;
                const float4 g03 = s_g03[sid][bi];
                const float4 g47 = s_g47[sid][bi];
                const float4 Y   = s_Y[sid][bi];

                const float2* c2p = sC2_q + poff;
                float2 c0 = c2p[0 * 16], c1 = c2p[1 * 16], c2 = c2p[2 * 16], c3 = c2p[3 * 16];
                float2 c4 = c2p[4 * 16], c5 = c2p[5 * 16], c6 = c2p[6 * 16], c7 = c2p[7 * 16];
                const float r2_0 = fmaf(c7.x, g47.w, fmaf(c6.x, g47.z, fmaf(c5.x, g47.y, fmaf(c4.x, g47.x,
                                   fmaf(c3.x, g03.w, fmaf(c2.x, g03.z, fmaf(c1.x, g03.y, c0.x * g03.x)))))));
                const float r2_1 = fmaf(c7.y, g47.w, fmaf(c6.y, g47.z, fmaf(c5.y, g47.y, fmaf(c4.y, g47.x,
                                   fmaf(c3.y, g03.w, fmaf(c2.y, g03.z, fmaf(c1.y, g03.y, c0.y * g03.x)))))));

                float rbf0 = 0.0f, rbf1 = 0.0f;
                if constexpr (REM_BF) {
                    const float f0_j = s_f0[sid][bi];
                    const float2* cbp = sCbf_q + poff;
                    float2 b0 = cbp[0 * 16], b1 = cbp[1 * 16], b2 = cbp[2 * 16], b3 = cbp[3 * 16];
                    float2 b4 = cbp[4 * 16], b5 = cbp[5 * 16], b6 = cbp[6 * 16], b7 = cbp[7 * 16];
                    rbf0 = f0_j * fmaf(b7.x, g47.w, fmaf(b6.x, g47.z, fmaf(b5.x, g47.y, fmaf(b4.x, g47.x,
                                  fmaf(b3.x, g03.w, fmaf(b2.x, g03.z, fmaf(b1.x, g03.y, b0.x * g03.x)))))));
                    rbf1 = f0_j * fmaf(b7.y, g47.w, fmaf(b6.y, g47.z, fmaf(b5.y, g47.y, fmaf(b4.y, g47.x,
                                  fmaf(b3.y, g03.w, fmaf(b2.y, g03.z, fmaf(b1.y, g03.y, b0.y * g03.x)))))));
                }

                const float i0   = li.x - (REM_BF ? rbf0 : 0.0f);
                const float i1_0 = li.y - (REM_BF ? rbf1 * Y.y : 0.0f);
                const float i1_1 = li.z - (REM_BF ? rbf1 * Y.z : 0.0f);
                const float i1_2 = li.w - (REM_BF ? rbf1 * Y.w : 0.0f);

                if (bi + 1 < bcnt) li = load_I(s_j[sid][bi + 1]);

                acc0 = fmaf(r2_0, i0, acc0);
                acc1 = fmaf(r2_0, i1_0, acc1);
                acc2 = fmaf(r2_0, i1_1, acc2);
                acc3 = fmaf(r2_0, i1_2, acc3);

                const float p1_0 = r2_1 * Y.y, p1_1 = r2_1 * Y.z, p1_2 = r2_1 * Y.w;
                acc4 = fmaf(p1_0, i0, acc4);
                acc5 = fmaf(p1_1, i0, acc5);
                acc6 = fmaf(p1_2, i0, acc6);

                constexpr float C110 = 0.57735026918962576451f;
                constexpr float C_SQ2 = 0.70710678118654752440f;
                constexpr float C_SQ6 = 0.40824829046386301637f;
                constexpr float C_2SQ6 = 0.81649658092772603273f;
                acc7  = fmaf(C110, p1_0 * i1_0 + p1_1 * i1_1 + p1_2 * i1_2, acc7);
                acc8  = fmaf(C_SQ2, p1_0 * i1_2 + p1_2 * i1_0, acc8);
                acc9  = fmaf(C_SQ2, p1_0 * i1_1 + p1_1 * i1_0, acc9);
                acc10 += (-C_SQ6) * p1_0 * i1_0 + C_2SQ6 * p1_1 * i1_1 - C_SQ6 * p1_2 * i1_2;
                acc11 = fmaf(C_SQ2, p1_1 * i1_2 + p1_2 * i1_1, acc11);
                acc12 += C_SQ2 * (p1_2 * i1_2 - p1_0 * i1_0);
            }
        }
        __syncwarp(smask);
    }

    float sq_lane = 0.0f;
    if (active_q) {
        sq_lane = fmaf(acc0, acc0, fmaf(acc1, acc1, fmaf(acc2, acc2, fmaf(acc3, acc3,
                  fmaf(acc4, acc4, fmaf(acc5, acc5, fmaf(acc6, acc6, fmaf(acc7, acc7,
                  fmaf(acc8, acc8, fmaf(acc9, acc9, fmaf(acc10, acc10, fmaf(acc11, acc11, acc12 * acc12))))))))))));
    }
    const float sq_tot = half_warp_sum_f(sq_lane, smask);
    const float f1 = a0_over_sig1 / sqrtf(1.0f + sq_tot * inv_DA2_sig1_kap_sq);
    if (slane == 0) f1_out[i] = f1;

    if (active_q) {
        const int c0 = l2_c.v[0], c1 = l2_c.v[1], c2 = l2_c.v[2];
        A2.p[0][(int64_t)i * c0 + slane]         = acc0 * f1;
        A2.p[0][(int64_t)i * c0 + n_msg + slane] = acc7 * f1;
        float* d1 = A2.p[1] + (int64_t)i * 3 * c1 + slane;
        d1[0]          = acc1 * f1; d1[c1]         = acc2 * f1; d1[2 * c1]         = acc3 * f1;
        d1[n_msg]      = acc4 * f1; d1[c1 + n_msg] = acc5 * f1; d1[2 * c1 + n_msg] = acc6 * f1;
        float* d2 = A2.p[2] + (int64_t)i * 5 * c2 + slane;
        d2[0] = acc8 * f1; d2[c2] = acc9 * f1; d2[2 * c2] = acc10 * f1; d2[3 * c2] = acc11 * f1; d2[4 * c2] = acc12 * f1;
    }
}

// =================================================================================================
// Kernel 3a: Fast Layer-2 Edge Forward (half-warp for n_msg<=16, batched edges, smem C2+Cbf, inline CG)
// Supports lfeat2 in {1, 2}, lmsg == 1, lout2 == 2.
// =================================================================================================
template <int LFEAT2, int SUBWARP, int MSG_PASSES, int NB_MAX, bool REM_BF, bool SMEM_C>
__global__ void __launch_bounds__(128) ilmlip_l2_edge_fwd_fast_kernel(
    int N, int S, int n_base, int n_msg,
    Int4 l2_c, float r_max, float a0_over_sig1, float inv_DA2_sig1_kap_sq,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2,   // [S*S, n_base, (LFEAT2+1)*n_msg]
    const float* __restrict__ Cbf,  // [S*S, n_base, 2*n_msg]
    const float* __restrict__ f0, CPtr4 I_node,
    Ptr4 A2, float* __restrict__ f1_out) {
    constexpr int NSUB = 128 / SUBWARP;  // 8 (SUBWARP=16) or 4 (SUBWARP=32)
    constexpr int EBATCH = (SUBWARP == 16) ? 4 : 8;
    constexpr int N_SH = (LFEAT2 == 2) ? 9 : 4;
    constexpr int N_OUT = (LFEAT2 == 2) ? 21 : 13;

    extern __shared__ float smem_c[];
    __shared__ float s_g[NSUB][EBATCH][NB_MAX];
    __shared__ float s_Y[NSUB][EBATCH][N_SH];
    __shared__ int s_j[NSUB][EBATCH];
    __shared__ int s_pair[NSUB][EBATCH];
    __shared__ float s_f0[NSUB][EBATCH];

    const int l2_tot_cols = (LFEAT2 + 1) * n_msg;
    const int bf_tot_cols = 2 * n_msg;
    const int c2_elems = S * S * n_base * l2_tot_cols;
    const int cbf_elems = REM_BF ? (S * S * n_base * bf_tot_cols) : 0;
    if constexpr (SMEM_C) {
        for (int idx = threadIdx.x; idx < c2_elems; idx += 128) smem_c[idx] = __ldg(C2 + idx);
        if constexpr (REM_BF) {
            float* smem_bf = smem_c + c2_elems;
            for (int idx = threadIdx.x; idx < cbf_elems; idx += 128) smem_bf[idx] = __ldg(Cbf + idx);
        }
        __syncthreads();
    }

    const int sid = threadIdx.x / SUBWARP;
    const int slane = threadIdx.x & (SUBWARP - 1);
    const unsigned smask = (SUBWARP == 16) ? ((sid & 1) ? 0xffff0000u : 0x0000ffffu) : 0xffffffffu;
    const int i = blockIdx.x * NSUB + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    const float* C2_base = SMEM_C ? smem_c : C2;
    const float* Cbf_base = REM_BF ? (SMEM_C ? (smem_c + c2_elems) : Cbf) : nullptr;

    float acc[MSG_PASSES][N_OUT];
    #pragma unroll
    for (int s = 0; s < MSG_PASSES; ++s)
        #pragma unroll
        for (int k = 0; k < N_OUT; ++k) acc[s][k] = 0.0f;

    for (int eb_b = eb; eb_b < ee; eb_b += EBATCH) {
        const int bcnt = (ee - eb_b < EBATCH) ? (ee - eb_b) : EBATCH;
        if (slane < bcnt) {
            const int e = eb_b + slane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const int pair = z_i * S + __ldg(species + j);
            const float r = fmaxf(vl.w, 1e-8f);
            s_j[sid][slane] = j;
            s_pair[sid][slane] = pair;
            if (REM_BF) s_f0[sid][slane] = __ldg(f0 + j);
            eval_cheb<false>(r, r_max, n_base, s_g[sid][slane], nullptr);
        } else if (slane >= EBATCH && slane < EBATCH + bcnt) {
            const int bi = slane - EBATCH;
            const int e = eb_b + bi;
            const float4 vl = __ldg(vec_len + e);
            const float r = fmaxf(vl.w, 1e-8f);
            float4 Yv[4];
            loup::edge_sh16(vl.x, vl.y, vl.z, r, Yv);
            unpack_sh16(Yv, LFEAT2, s_Y[sid][bi]);
        }
        __syncwarp(smask);

        for (int bi = 0; bi < bcnt; ++bi) {
            const int j = s_j[sid][bi];
            const int pair = s_pair[sid][bi];
            const float f0_j = REM_BF ? s_f0[sid][bi] : 0.0f;
            const float* gk = s_g[sid][bi];
            const float* Yk = s_Y[sid][bi];
            const float* C2p = C2_base + (int64_t)pair * n_base * l2_tot_cols;
            const float* Cbfp = REM_BF ? (Cbf_base + (int64_t)pair * n_base * bf_tot_cols) : nullptr;

            #pragma unroll
            for (int s = 0; s < MSG_PASSES; ++s) {
                const int q = slane + s * SUBWARP;
                if (q >= n_msg) continue;

                float r2_0 = 0.0f, r2_1 = 0.0f, r2_2 = 0.0f;
                auto accum_c2 = [&](int k) {
                    const float gval = gk[k];
                    const float* row2 = C2p + k * l2_tot_cols + q;
                    r2_0 = fmaf(SMEM_C ? row2[0] : __ldg(row2), gval, r2_0);
                    r2_1 = fmaf(SMEM_C ? row2[n_msg] : __ldg(row2 + n_msg), gval, r2_1);
                    if constexpr (LFEAT2 == 2) r2_2 = fmaf(SMEM_C ? row2[2 * n_msg] : __ldg(row2 + 2 * n_msg), gval, r2_2);
                };
                if (n_base == 8) {
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) accum_c2(k);
                } else if (n_base == 12) {
                    #pragma unroll
                    for (int k = 0; k < 12; ++k) accum_c2(k);
                } else if (n_base == 16) {
                    #pragma unroll
                    for (int k = 0; k < 16; ++k) accum_c2(k);
                } else {
                    for (int k = 0; k < n_base; ++k) accum_c2(k);
                }

                float rbf0 = 0.0f, rbf1 = 0.0f;
                if constexpr (REM_BF) {
                    auto accum_bf = [&](int k) {
                        const float gval = gk[k];
                        const float* rowb = Cbfp + k * bf_tot_cols + q;
                        rbf0 = fmaf(SMEM_C ? rowb[0] : __ldg(rowb), gval, rbf0);
                        rbf1 = fmaf(SMEM_C ? rowb[n_msg] : __ldg(rowb + n_msg), gval, rbf1);
                    };
                    if (n_base == 8) {
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) accum_bf(k);
                    } else if (n_base == 12) {
                        #pragma unroll
                        for (int k = 0; k < 12; ++k) accum_bf(k);
                    } else if (n_base == 16) {
                        #pragma unroll
                        for (int k = 0; k < 16; ++k) accum_bf(k);
                    } else {
                        for (int k = 0; k < n_base; ++k) accum_bf(k);
                    }
                    rbf0 *= f0_j;
                    rbf1 *= f0_j;
                }

                const float i0 = __ldg(I_node.p[0] + (int64_t)j * n_msg + q) - (REM_BF ? rbf0 * Yk[0] : 0.0f);
                const float* pI1 = I_node.p[1] + (int64_t)j * 3 * n_msg + q;
                const float i1_0 = __ldg(pI1)             - (REM_BF ? rbf1 * Yk[1] : 0.0f);
                const float i1_1 = __ldg(pI1 + n_msg)     - (REM_BF ? rbf1 * Yk[2] : 0.0f);
                const float i1_2 = __ldg(pI1 + 2 * n_msg) - (REM_BF ? rbf1 * Yk[3] : 0.0f);

                // (0,0)->0 [reg 0]
                const float p0 = r2_0 * Yk[0];
                acc[s][0] = fmaf(p0, i0, acc[s][0]);
                // (0,1)->1 [reg 1..3]
                acc[s][1] = fmaf(p0, i1_0, acc[s][1]);
                acc[s][2] = fmaf(p0, i1_1, acc[s][2]);
                acc[s][3] = fmaf(p0, i1_2, acc[s][3]);

                // (1,0)->1 [reg 4..6]
                const float p1_0 = r2_1 * Yk[1], p1_1 = r2_1 * Yk[2], p1_2 = r2_1 * Yk[3];
                acc[s][4] = fmaf(p1_0, i0, acc[s][4]);
                acc[s][5] = fmaf(p1_1, i0, acc[s][5]);
                acc[s][6] = fmaf(p1_2, i0, acc[s][6]);

                // (1,1)->0 [reg 7]
                constexpr float C110 = 0.57735026918962576451f;
                acc[s][7] = fmaf(C110, p1_0 * i1_0 + p1_1 * i1_1 + p1_2 * i1_2, acc[s][7]);

                // (1,1)->2 [reg 8..12]
                constexpr float C_SQ2 = 0.70710678118654752440f;
                constexpr float C_SQ6 = 0.40824829046386301637f;
                constexpr float C_2SQ6 = 0.81649658092772603273f;
                acc[s][8]  = fmaf(C_SQ2,  p1_0 * i1_2 + p1_2 * i1_0, acc[s][8]);
                acc[s][9]  = fmaf(C_SQ2,  p1_0 * i1_1 + p1_1 * i1_0, acc[s][9]);
                acc[s][10] += (-C_SQ6) * p1_0 * i1_0 + C_2SQ6 * p1_1 * i1_1 - C_SQ6 * p1_2 * i1_2;
                acc[s][11] = fmaf(C_SQ2,  p1_1 * i1_2 + p1_2 * i1_1, acc[s][11]);
                acc[s][12] += C_SQ2 * (p1_2 * i1_2 - p1_0 * i1_0);

                if constexpr (LFEAT2 == 2) {
                    const float p2_0 = r2_2 * Yk[4], p2_1 = r2_2 * Yk[5], p2_2 = r2_2 * Yk[6];
                    const float p2_3 = r2_2 * Yk[7], p2_4 = r2_2 * Yk[8];
                    // (2,0)->2 [reg 13..17]
                    acc[s][13] = fmaf(p2_0, i0, acc[s][13]);
                    acc[s][14] = fmaf(p2_1, i0, acc[s][14]);
                    acc[s][15] = fmaf(p2_2, i0, acc[s][15]);
                    acc[s][16] = fmaf(p2_3, i0, acc[s][16]);
                    acc[s][17] = fmaf(p2_4, i0, acc[s][17]);
                    // (2,1)->1 [reg 18..20]
                    constexpr float C_S310 = 0.54772255750516611346f;
                    constexpr float C_S110 = 0.31622776601683793320f;
                    constexpr float C_S410 = 0.63245553203367586640f;
                    acc[s][18] += C_S310 * (p2_0 * i1_2 + p2_1 * i1_1 - p2_4 * i1_0) - C_S110 * p2_2 * i1_0;
                    acc[s][19] += C_S310 * (p2_1 * i1_0 + p2_3 * i1_2) + C_S410 * p2_2 * i1_1;
                    acc[s][20] += C_S310 * (p2_0 * i1_0 + p2_3 * i1_1 + p2_4 * i1_2) - C_S110 * p2_2 * i1_2;
                }
            }
        }
        __syncwarp(smask);
    }

    float sq_lane = 0.0f;
    #pragma unroll
    for (int s = 0; s < MSG_PASSES; ++s) {
        const int q = slane + s * SUBWARP;
        if (q < n_msg) {
            #pragma unroll
            for (int k = 0; k < N_OUT; ++k) sq_lane = fmaf(acc[s][k], acc[s][k], sq_lane);
        }
    }
    const float sq_tot = (SUBWARP == 16) ? half_warp_sum_f(sq_lane, smask) : warp_sum_f(sq_lane);
    const float f1 = a0_over_sig1 / sqrtf(1.0f + sq_tot * inv_DA2_sig1_kap_sq);
    if (slane == 0) f1_out[i] = f1;

    #pragma unroll
    for (int s = 0; s < MSG_PASSES; ++s) {
        const int q = slane + s * SUBWARP;
        if (q < n_msg) {
            const int c0 = l2_c.v[0], c1 = l2_c.v[1], c2 = l2_c.v[2];
            // L=0: p=0 (reg 0), p=1 (reg 7)
            A2.p[0][(int64_t)i * c0 + q]         = acc[s][0] * f1;
            A2.p[0][(int64_t)i * c0 + n_msg + q] = acc[s][7] * f1;
            // L=1: p=0 (reg 1..3), p=1 (reg 4..6), p=2 (reg 18..20 if LFEAT2==2)
            float* d1 = A2.p[1] + (int64_t)i * 3 * c1 + q;
            d1[0]          = acc[s][1] * f1; d1[c1]          = acc[s][2] * f1; d1[2 * c1]          = acc[s][3] * f1;
            d1[n_msg]      = acc[s][4] * f1; d1[c1 + n_msg]  = acc[s][5] * f1; d1[2 * c1 + n_msg]  = acc[s][6] * f1;
            if constexpr (LFEAT2 == 2) {
                d1[2 * n_msg] = acc[s][18] * f1; d1[c1 + 2 * n_msg] = acc[s][19] * f1; d1[2 * c1 + 2 * n_msg] = acc[s][20] * f1;
            }
            // L=2: p=0 (reg 8..12), p=1 (reg 13..17 if LFEAT2==2)
            float* d2 = A2.p[2] + (int64_t)i * 5 * c2 + q;
            #pragma unroll
            for (int m = 0; m < 5; ++m) d2[m * c2] = acc[s][8 + m] * f1;
            if constexpr (LFEAT2 == 2) {
                #pragma unroll
                for (int m = 0; m < 5; ++m) d2[m * c2 + n_msg] = acc[s][13 + m] * f1;
            }
        }
    }
}

// =================================================================================================
// Kernel 3: Layer-2 Edge Forward (back-flow removal + fixed-CG coupling + f1 gate, zero edge buf)
// =================================================================================================
template <int MSG_PASSES, int MAX_L2_OUT, bool REM_BF>
__global__ void __launch_bounds__(128) ilmlip_l2_edge_fwd_kernel(
    int N, int S, int lfeat2, int lmsg, int lout2, int n_base, int n_msg, int n_l2_paths,
    Int4 l2_c, float inv_rmax, float a0_over_sig1, float inv_DA2_sig1_kap_sq,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2,   // [S*S, n_base, (lfeat2+1)*n_msg]
    const float* __restrict__ Cbf,  // [S*S, n_base, (lmsg+1)*n_msg]
    const float* __restrict__ f0, CPtr4 I_node,
    const int32_t* __restrict__ l2p,  // [n_l2_paths, 7]: l1, l2, L, p_in_L, cg_off, cg_cnt, reg_off
    const int32_t* __restrict__ cg_idx, const float* __restrict__ cg_val,
    Ptr4 A2, float* __restrict__ f1_out) {
    __shared__ float s_g[4][32];
    __shared__ float s_Y[4][16];

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    const int l2_tot_cols = (lfeat2 + 1) * n_msg;
    const int bf_tot_cols = (lmsg + 1) * n_msg;
    const int sh_lmax = lfeat2 > lmsg ? lfeat2 : lmsg;

    float acc[MSG_PASSES][MAX_L2_OUT];
    #pragma unroll
    for (int s = 0; s < MSG_PASSES; ++s)
        #pragma unroll
        for (int k = 0; k < MAX_L2_OUT; ++k) acc[s][k] = 0.0f;

    for (int e = eb; e < ee; ++e) {
        const float4 vl = __ldg(vec_len + e);
        const int j = __ldg(senders + e);
        const int z_j = __ldg(species + j);
        const int pair = z_i * S + z_j;
        const float r = fmaxf(vl.w, 1e-8f);
        const float f0_j = REM_BF ? __ldg(f0 + j) : 0.0f;

        if (lane == 0) {
            eval_cheb<false>(r, inv_rmax, n_base, s_g[wid], nullptr);
            float4 Yv[4];
            loup::edge_sh16(vl.x, vl.y, vl.z, r, Yv);
            s_Y[wid][0] = Yv[0].x; s_Y[wid][1] = Yv[0].y; s_Y[wid][2] = Yv[0].z; s_Y[wid][3] = Yv[0].w;
            if (sh_lmax >= 2) {
                s_Y[wid][4] = Yv[1].x; s_Y[wid][5] = Yv[1].y; s_Y[wid][6] = Yv[1].z; s_Y[wid][7] = Yv[1].w;
                s_Y[wid][8] = Yv[2].x;
            }
        }
        __syncwarp();

        const float* C2p = C2 + (int64_t)pair * n_base * l2_tot_cols;
        const float* Cbfp = REM_BF ? (Cbf + (int64_t)pair * n_base * bf_tot_cols) : nullptr;

        #pragma unroll
        for (int s = 0; s < MSG_PASSES; ++s) {
            const int q = lane + s * 32;
            if (q >= n_msg) continue;

            float r2[3] = {0.0f, 0.0f, 0.0f};
            for (int l1 = 0; l1 <= lfeat2; ++l1) {
                float rv = 0.0f;
                const float* col = C2p + l1 * n_msg + q;
                for (int k = 0; k < n_base; ++k) rv = fmaf(__ldg(col + k * l2_tot_cols), s_g[wid][k], rv);
                r2[l1] = rv;
            }

            float ie[9];  // lmsg <= 2 -> max 9 components
            for (int l2 = 0; l2 <= lmsg; ++l2) {
                float rbf = 0.0f;
                if (REM_BF) {
                    const float* col = Cbfp + l2 * n_msg + q;
                    for (int k = 0; k < n_base; ++k) rbf = fmaf(__ldg(col + k * bf_tot_cols), s_g[wid][k], rbf);
                    rbf *= f0_j;
                }
                const int m0 = l2 * l2;
                const int dm = 2 * l2 + 1;
                const float* src = I_node.p[l2] + (int64_t)j * dm * n_msg + q;
                for (int m = 0; m < dm; ++m) {
                    float v = __ldg(src + m * n_msg);
                    if (REM_BF) v -= rbf * s_Y[wid][m0 + m];
                    ie[m0 + m] = v;
                }
            }

            for (int p = 0; p < n_l2_paths; ++p) {
                const int32_t* pr = l2p + p * 7;
                const int l1 = pr[0], l2 = pr[1], cg_off = pr[4], cg_cnt = pr[5], reg_off = pr[6];
                const int m01 = l1 * l1, m02 = l2 * l2;
                const float r2_l1 = r2[l1];
                for (int k = 0; k < cg_cnt; ++k) {
                    const int pk = __ldg(cg_idx + cg_off + k);
                    const int m1 = pk & 15, m2 = (pk >> 4) & 15, mL = (pk >> 8) & 15;
                    acc[s][reg_off + mL] =
                        fmaf(__ldg(cg_val + cg_off + k) * (r2_l1 * s_Y[wid][m01 + m1]), ie[m02 + m2], acc[s][reg_off + mL]);
                }
            }
        }
        __syncwarp();
    }

    float sq_lane = 0.0f;
    #pragma unroll
    for (int s = 0; s < MSG_PASSES; ++s) {
        const int q = lane + s * 32;
        if (q < n_msg) {
            for (int p = 0; p < n_l2_paths; ++p) {
                const int L = l2p[p * 7 + 2];
                const int reg_off = l2p[p * 7 + 6];
                const int dm = 2 * L + 1;
                for (int m = 0; m < dm; ++m) {
                    const float v = acc[s][reg_off + m];
                    sq_lane = fmaf(v, v, sq_lane);
                }
            }
        }
    }
    const float sq_tot = warp_sum(sq_lane);
    const float f1 = a0_over_sig1 / sqrtf(1.0f + sq_tot * inv_DA2_sig1_kap_sq);
    if (lane == 0) f1_out[i] = f1;

    #pragma unroll
    for (int s = 0; s < MSG_PASSES; ++s) {
        const int q = lane + s * 32;
        if (q < n_msg) {
            for (int p = 0; p < n_l2_paths; ++p) {
                const int32_t* pr = l2p + p * 7;
                const int L = pr[2], p_in_L = pr[3], reg_off = pr[6];
                const int dm = 2 * L + 1;
                const int cL = l2_c.v[L];
                float* dst = A2.p[L] + (int64_t)i * dm * cL + p_in_L * n_msg + q;
                for (int m = 0; m < dm; ++m) dst[m * cL] = acc[s][reg_off + m] * f1;
            }
        }
    }
}

// =================================================================================================
// 2D Register-Blocked Tall-Skinny GEMM (N <= 16, K*N <= 4096, K % 4 == 0):
//   C[M, N] = A[M, K] @ W[K, N]
// Stages W[K, N] ONCE in SMEM, tiles BM=32 x BN=16 across 128 threads (16x8 -> 2x2 outer-product
// per thread = 4x fewer SMEM/L1 loads than 1x1!), 100% coalesced float4 A loads into sA[32][17]!
// Cuts W_msg_0 fwd (25.1 -> ~7 us) and W_msg_1 fwd (24.5 -> ~12 us)!
// =================================================================================================
__global__ void __launch_bounds__(128) ilmlip_skinny_gemm_n16_kernel(
    int M, int N, int K, const float* __restrict__ A, const float* __restrict__ W, float* __restrict__ C) {
    extern __shared__ float sW[];  // [K * N]
    __shared__ float sA[32][17];   // 32 rows x 16 K-slice (padded to 17 -> 0 bank conflicts!)

    const int tid = threadIdx.x;
    const int kn = K * N;
    for (int idx = tid; idx < kn; idx += 128) sW[idx] = __ldg(W + idx);

    const int m0 = blockIdx.x * 32;
    const int tm = tid >> 3;       // 0..15 (each thread owns rows 2*tm and 2*tm+1)
    const int tn = tid & 7;        // 0..7  (each thread owns cols tn and tn+8)
    const int r0 = m0 + tm;
    const int r1 = m0 + tm + 16;
    const int c0 = tn;
    const int c1 = tn + 8;

    const int la_m = tid >> 2;     // 0..31 (row of sA)
    const int la_k4 = tid & 3;     // 0..3  (float4 chunk in [0..15])
    const int gm = m0 + la_m;
    const bool valid_gm = (gm < M);
    const bool k_mult4 = ((K & 3) == 0);

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    __syncthreads();
    for (int kb = 0; kb < K; kb += 16) {
        const int krem = (K - kb < 16) ? (K - kb) : 16;
        if (valid_gm && k_mult4 && (la_k4 * 4 + 4 <= krem)) {
            const float4 v = *reinterpret_cast<const float4*>(A + (int64_t)gm * K + kb + la_k4 * 4);
            sA[la_m][la_k4 * 4 + 0] = v.x;
            sA[la_m][la_k4 * 4 + 1] = v.y;
            sA[la_m][la_k4 * 4 + 2] = v.z;
            sA[la_m][la_k4 * 4 + 3] = v.w;
        } else {
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                const int dk = la_k4 * 4 + d;
                sA[la_m][dk] = (valid_gm && dk < krem) ? __ldg(A + (int64_t)gm * K + kb + dk) : 0.0f;
            }
        }
        __syncthreads();

        const float* Wb = sW + kb * N;
        if (krem == 16) {
            #pragma unroll
            for (int dk = 0; dk < 16; ++dk) {
                const float a0 = sA[tm][dk];
                const float a1 = sA[tm + 16][dk];
                const float w0 = (c0 < N) ? Wb[dk * N + c0] : 0.0f;
                const float w1 = (c1 < N) ? Wb[dk * N + c1] : 0.0f;
                acc00 = fmaf(a0, w0, acc00);
                acc01 = fmaf(a0, w1, acc01);
                acc10 = fmaf(a1, w0, acc10);
                acc11 = fmaf(a1, w1, acc11);
            }
        } else {
            for (int dk = 0; dk < krem; ++dk) {
                const float a0 = sA[tm][dk];
                const float a1 = sA[tm + 16][dk];
                const float w0 = (c0 < N) ? Wb[dk * N + c0] : 0.0f;
                const float w1 = (c1 < N) ? Wb[dk * N + c1] : 0.0f;
                acc00 = fmaf(a0, w0, acc00);
                acc01 = fmaf(a0, w1, acc01);
                acc10 = fmaf(a1, w0, acc10);
                acc11 = fmaf(a1, w1, acc11);
            }
        }
        __syncthreads();
    }

    if (r0 < M) {
        if (c0 < N) C[(int64_t)r0 * N + c0] = acc00;
        if (c1 < N) C[(int64_t)r0 * N + c1] = acc01;
    }
    if (r1 < M) {
        if (c0 < N) C[(int64_t)r1 * N + c0] = acc10;
        if (c1 < N) C[(int64_t)r1 * N + c1] = acc11;
    }
}

// =================================================================================================
// Kernel 4s: Zero-Spill Fast Layer-2 Invariants + x Standardization (lout2==2, canonical 4 bisp)
// Also caches per-row exponent eA_x[i] for hgemm2_f32a_row (eliminates h2r_row_exp_kernel!)
// =================================================================================================
__global__ void __launch_bounds__(128) ilmlip_l2_inv_fwd_l2_kernel(
    int N, int r_gram, int r_bisp, int n_inv1, int n_feat, int bisp_off,
    Int4 l2_c, Int4 gram_off,
    const int32_t* __restrict__ triu_i, const int32_t* __restrict__ triu_j,
    const float* __restrict__ A2_0, const float* __restrict__ Q0, CPtr4 PQ,
    const float* __restrict__ mu_x, const float* __restrict__ inv_std_x,
    float* __restrict__ x, int8_t* __restrict__ eA_x) {
    extern __shared__ float smem[];
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int r_gb = r_gram + r_bisp;
    float* xi = x + (int64_t)i * n_feat;
    float am = 0.0f;

    for (int f = lane; f < n_inv1; f += 32) {
        const float v = (xi[f] - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
        xi[f] = v;
        am = fmaxf(am, fabsf(v));
    }

    const int c0 = l2_c.v[0];
    for (int c = lane; c < c0; c += 32) {
        const int f = n_inv1 + c;
        const float v = (A2_0[(int64_t)i * c0 + c] - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
        xi[f] = v;
        am = fmaxf(am, fabsf(v));
    }

    if (r_gram > 0) {
        const int n_tri = (r_gram * (r_gram + 1)) >> 1;
        const int p_stride = r_gram + 1;
        float* sP = smem + (size_t)wid * 5 * p_stride;

        if (l2_c.v[1] > 0) {
            constexpr float inv_s1 = 0.57735026918962576451f;
            const float* P1 = PQ.p[1] + (int64_t)i * 3 * r_gb;
            for (int idx = lane; idx < 3 * r_gram; idx += 32) {
                const int m = idx / r_gram, a = idx - m * r_gram;
                sP[m * p_stride + a] = P1[m * r_gb + a];
            }
            __syncwarp();
            const int goff1 = gram_off.v[1];
            for (int idx = lane; idx < n_tri; idx += 32) {
                const int a = __ldg(triu_i + idx);
                const int b = __ldg(triu_j + idx);
                const float s = fmaf(sP[0 * p_stride + a], sP[0 * p_stride + b],
                                fmaf(sP[1 * p_stride + a], sP[1 * p_stride + b],
                                     sP[2 * p_stride + a] * sP[2 * p_stride + b]));
                const int f = goff1 + idx;
                const float v = (s * inv_s1 - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
                xi[f] = v;
                am = fmaxf(am, fabsf(v));
            }
            __syncwarp();
        }
        if (l2_c.v[2] > 0) {
            constexpr float inv_s2 = 0.44721359549995793928f;
            const float* P2 = PQ.p[2] + (int64_t)i * 5 * r_gb;
            for (int idx = lane; idx < 5 * r_gram; idx += 32) {
                const int m = idx / r_gram, a = idx - m * r_gram;
                sP[m * p_stride + a] = P2[m * r_gb + a];
            }
            __syncwarp();
            const int goff2 = gram_off.v[2];
            for (int idx = lane; idx < n_tri; idx += 32) {
                const int a = __ldg(triu_i + idx);
                const int b = __ldg(triu_j + idx);
                const float s = fmaf(sP[0 * p_stride + a], sP[0 * p_stride + b],
                                fmaf(sP[1 * p_stride + a], sP[1 * p_stride + b],
                                fmaf(sP[2 * p_stride + a], sP[2 * p_stride + b],
                                fmaf(sP[3 * p_stride + a], sP[3 * p_stride + b],
                                     sP[4 * p_stride + a] * sP[4 * p_stride + b]))));
                const int f = goff2 + idx;
                const float v = (s * inv_s2 - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
                xi[f] = v;
                am = fmaxf(am, fabsf(v));
            }
        }
    }

    if (r_bisp > 0) {
        for (int b = lane; b < r_bisp; b += 32) {
            const float q0 = Q0[(int64_t)i * r_bisp + b];
            const float* p1 = PQ.p[1] + (int64_t)i * 3 * r_gb + r_gram + b;
            const float q1_0 = p1[0], q1_1 = p1[r_gb], q1_2 = p1[2 * r_gb];
            const float* p2 = PQ.p[2] + (int64_t)i * 5 * r_gb + r_gram + b;
            const float q2_0 = p2[0], q2_1 = p2[r_gb], q2_2 = p2[2 * r_gb], q2_3 = p2[3 * r_gb], q2_4 = p2[4 * r_gb];

            constexpr float C011   =  0.8326832056045532f;
            constexpr float C022   =  0.7647244930267334f;
            constexpr float C112_a = -0.34432995319366455f;
            constexpr float C112_b =  0.5963969826698303f;
            constexpr float C112_c =  0.6886599063873291f;
            constexpr float C222_a =  0.5345224738121033f;
            constexpr float C222_b =  0.26726123690605164f;
            constexpr float C222_c =  0.4629100561141968f;

            const float q1_0s = q1_0 * q1_0, q1_1s = q1_1 * q1_1, q1_2s = q1_2 * q1_2;
            const float q2_0s = q2_0 * q2_0, q2_1s = q2_1 * q2_1, q2_2s = q2_2 * q2_2, q2_3s = q2_3 * q2_3, q2_4s = q2_4 * q2_4;

            const float b0 = C011 * q0 * (q1_0s + q1_1s + q1_2s);
            const float b1 = C022 * q0 * (q2_0s + q2_1s + q2_2s + q2_3s + q2_4s);
            const float b2 = (C112_a * (q1_0s + q1_2s) + C112_c * q1_1s) * q2_2 +
                             C112_b * ((q1_2s - q1_0s) * q2_4 + 2.0f * (q1_0 * q1_2 * q2_0 + q1_0 * q1_1 * q2_1 + q1_1 * q1_2 * q2_3));
            const float b3 = C222_a * q2_2 * (q2_2s - 3.0f * (q2_0s + q2_4s)) +
                             3.0f * C222_b * q2_2 * (q2_1s + q2_3s) +
                             3.0f * C222_c * (2.0f * q2_0 * q2_1 * q2_3 + q2_4 * (q2_3s - q2_1s));

            const int f0 = bisp_off + 0 * r_bisp + b;
            const int f1 = bisp_off + 1 * r_bisp + b;
            const int f2 = bisp_off + 2 * r_bisp + b;
            const int f3 = bisp_off + 3 * r_bisp + b;
            const float v0 = (b0 - __ldg(mu_x + f0)) * __ldg(inv_std_x + f0);
            const float v1 = (b1 - __ldg(mu_x + f1)) * __ldg(inv_std_x + f1);
            const float v2 = (b2 - __ldg(mu_x + f2)) * __ldg(inv_std_x + f2);
            const float v3 = (b3 - __ldg(mu_x + f3)) * __ldg(inv_std_x + f3);
            xi[f0] = v0; xi[f1] = v1; xi[f2] = v2; xi[f3] = v3;
            am = fmaxf(am, fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))));
        }
    }

    if (eA_x) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
        if (lane == 0) {
            int e = -127;
            if (am > 0.0f && isfinite(am)) {
                e = (int)((__float_as_uint(am) >> 23) & 0xff) - 127 + 1;
                if (e < -108) e = -108;
                if (e > 108) e = 108;
            }
            eA_x[i] = (int8_t)e;
        }
    }
}

// =================================================================================================
// Kernel 4a: Layer-2 Invariants (A2[0] + upper-tri Gram + low-rank bispectrum) + x standardization
// Uses merged PQ[L] (L>=1, stride r_gb = r_gram+r_bisp), Q0 (L==0), inline standardization, smem bisp/Q!
// =================================================================================================
__global__ void __launch_bounds__(128) ilmlip_l2_inv_and_norm_fwd_kernel(
    int N, int lout2, int r_gram, int r_bisp, int n_bisp, int bisp_idx_cnt, int n_inv1, int n_feat, int bisp_off,
    Int4 l2_c, Int4 gram_off,
    const int32_t* __restrict__ triu_i, const int32_t* __restrict__ triu_j,
    const int32_t* __restrict__ bisp_tbl, const int32_t* __restrict__ bisp_idx, const float* __restrict__ bisp_val,
    const float* __restrict__ A2_0, const float* __restrict__ Q0, CPtr4 PQ,
    const float* __restrict__ mu_x, const float* __restrict__ inv_std_x,
    float* __restrict__ x) {
    extern __shared__ float smem[];
    const int n_bisp_ints = n_bisp * 5;
    int32_t* s_btbl = reinterpret_cast<int32_t*>(smem);
    int32_t* s_bidx = s_btbl + n_bisp_ints;
    float* s_bval = reinterpret_cast<float*>(s_bidx + bisp_idx_cnt);
    float* s_q_base = s_bval + bisp_idx_cnt;

    if (r_bisp > 0 && n_bisp > 0) {
        for (int k = threadIdx.x; k < n_bisp_ints; k += 128) s_btbl[k] = __ldg(bisp_tbl + k);
        for (int k = threadIdx.x; k < bisp_idx_cnt; k += 128) {
            s_bidx[k] = __ldg(bisp_idx + k);
            s_bval[k] = __ldg(bisp_val + k);
        }
        __syncthreads();
    }

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int r_gb = r_gram + r_bisp;
    float* xi = x + (int64_t)i * n_feat;

    // Standardize Layer-1 invariants [0 .. n_inv1 - 1] in place
    for (int f = lane; f < n_inv1; f += 32) {
        xi[f] = (xi[f] - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
    }

    // A2_0 normalized inline
    const int c0 = l2_c.v[0];
    for (int c = lane; c < c0; c += 32) {
        const int f = n_inv1 + c;
        xi[f] = (A2_0[(int64_t)i * c0 + c] - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
    }

    // Upper-tri Gram normalized inline (P sits at columns 0..r_gram-1 of PQ[L])
    if (r_gram > 0) {
        const int n_tri = (r_gram * (r_gram + 1)) >> 1;
        for (int L = 1; L <= lout2; ++L) {
            if (l2_c.v[L] == 0) continue;
            const int dm = 2 * L + 1;
            const float inv_s = 1.0f / sqrtf((float)dm);
            const float* PL = PQ.p[L] + (int64_t)i * dm * r_gb;
            const int goff = gram_off.v[L];
            for (int idx = lane; idx < n_tri; idx += 32) {
                const int a = __ldg(triu_i + idx);
                const int b = __ldg(triu_j + idx);
                float s = 0.0f;
                for (int m = 0; m < dm; ++m) s = fmaf(PL[m * r_gb + a], PL[m * r_gb + b], s);
                const int f = goff + idx;
                xi[f] = (s * inv_s - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
            }
        }
    }

    // Low-rank bispectrum normalized inline (Q0 at L=0; Q[L] at columns r_gram..r_gb-1 of PQ[L])
    if (r_bisp > 0 && n_bisp > 0) {
        if (lout2 <= 2 && r_bisp <= 32) {
            float* sq = s_q_base + (size_t)wid * 9 * r_bisp;
            if (lane < r_bisp) {
                if (l2_c.v[0] > 0) sq[0 * r_bisp + lane] = Q0[(int64_t)i * r_bisp + lane];
                if (l2_c.v[1] > 0) {
                    const float* q1 = PQ.p[1] + (int64_t)i * 3 * r_gb + r_gram + lane;
                    sq[1 * r_bisp + lane] = q1[0];
                    sq[2 * r_bisp + lane] = q1[r_gb];
                    sq[3 * r_bisp + lane] = q1[2 * r_gb];
                }
                if (lout2 == 2 && l2_c.v[2] > 0) {
                    const float* q2 = PQ.p[2] + (int64_t)i * 5 * r_gb + r_gram + lane;
                    #pragma unroll
                    for (int m = 0; m < 5; ++m) sq[(4 + m) * r_bisp + lane] = q2[m * r_gb];
                }
                for (int t = 0; t < n_bisp; ++t) {
                    const int32_t* br = s_btbl + t * 5;
                    const int o1 = br[0] * br[0], o2 = br[1] * br[1], o3 = br[2] * br[2];
                    const int off = br[3], cnt = br[4];
                    float s = 0.0f;
                    for (int k = 0; k < cnt; ++k) {
                        const int pk = s_bidx[off + k];
                        const int m1 = pk & 15, m2 = (pk >> 4) & 15, m3 = (pk >> 8) & 15;
                        s = fmaf(s_bval[off + k],
                                 sq[(o1 + m1) * r_bisp + lane] * sq[(o2 + m2) * r_bisp + lane] * sq[(o3 + m3) * r_bisp + lane], s);
                    }
                    const int f = bisp_off + t * r_bisp + lane;
                    xi[f] = (s - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
                }
            }
        } else {
            for (int b = lane; b < r_bisp; b += 32) {
                for (int t = 0; t < n_bisp; ++t) {
                    const int32_t* br = s_btbl + t * 5;
                    const int L1 = br[0], L2 = br[1], L3 = br[2], off = br[3], cnt = br[4];
                    const int st1 = (L1 == 0) ? r_bisp : r_gb;
                    const int st2 = (L2 == 0) ? r_bisp : r_gb;
                    const int st3 = (L3 == 0) ? r_bisp : r_gb;
                    const float* Q1 = (L1 == 0) ? (Q0 + (int64_t)i * r_bisp + b) : (PQ.p[L1] + (int64_t)i * (2 * L1 + 1) * r_gb + r_gram + b);
                    const float* Q2 = (L2 == 0) ? (Q0 + (int64_t)i * r_bisp + b) : (PQ.p[L2] + (int64_t)i * (2 * L2 + 1) * r_gb + r_gram + b);
                    const float* Q3 = (L3 == 0) ? (Q0 + (int64_t)i * r_bisp + b) : (PQ.p[L3] + (int64_t)i * (2 * L3 + 1) * r_gb + r_gram + b);
                    float s = 0.0f;
                    for (int k = 0; k < cnt; ++k) {
                        const int pk = s_bidx[off + k];
                        const int m1 = pk & 15, m2 = (pk >> 4) & 15, m3 = (pk >> 8) & 15;
                        s = fmaf(s_bval[off + k], Q1[m1 * st1] * Q2[m2 * st2] * Q3[m3 * st3], s);
                    }
                    const int f = bisp_off + t * r_bisp + b;
                    xi[f] = (s - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
                }
            }
        }
    }
}

// =================================================================================================
// Readout bias + SiLU forward / backward + final energy & ZBL core switch
// =================================================================================================
__global__ void __launch_bounds__(128) ilmlip_bias_silu_fwd_vec4_kernel(
    int N, int W4, bool per_species, const int* __restrict__ species,
    const float4* __restrict__ bias4, float4* __restrict__ pre4, float4* __restrict__ h4) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;
    const int sp = per_species ? species[i] : 0;
    const float4* bptr = bias4 + (int64_t)sp * W4;
    float4* pptr = pre4 + (int64_t)i * W4;
    float4* hptr = h4 + (int64_t)i * W4;
    for (int w = lane; w < W4; w += 32) {
        const float4 b = __ldg(bptr + w);
        float4 p = pptr[w];
        p.x += b.x; p.y += b.y; p.z += b.z; p.w += b.w;
        pptr[w] = p;
        float4 hv;
        hv.x = p.x / (1.0f + expf(-p.x));
        hv.y = p.y / (1.0f + expf(-p.y));
        hv.z = p.z / (1.0f + expf(-p.z));
        hv.w = p.w / (1.0f + expf(-p.w));
        hptr[w] = hv;
    }
}

__global__ void ilmlip_bias_silu_fwd_kernel(int N, int W, bool per_species, const int* __restrict__ species,
                                            const float* __restrict__ bias, float* __restrict__ pre,
                                            float* __restrict__ h) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)N * W) return;
    const int i = (int)(idx / W);
    const int w = (int)(idx - (int64_t)i * W);
    const float b = per_species ? __ldg(bias + species[i] * W + w) : __ldg(bias + w);
    const float p = pre[idx] + b;
    pre[idx] = p;
    h[idx] = p / (1.0f + expf(-p));
}

__global__ void ilmlip_silu_bwd_vec4_kernel(int64_t N4, const float4* __restrict__ pre4,
                                            const float4* __restrict__ dh4, float4* __restrict__ dpre4) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N4) return;
    const float4 p = pre4[idx];
    const float4 d = dh4[idx];
    const float sx = 1.0f / (1.0f + expf(-p.x));
    const float sy = 1.0f / (1.0f + expf(-p.y));
    const float sz = 1.0f / (1.0f + expf(-p.z));
    const float sw = 1.0f / (1.0f + expf(-p.w));
    float4 o;
    o.x = d.x * (sx * (1.0f + p.x * (1.0f - sx)));
    o.y = d.y * (sy * (1.0f + p.y * (1.0f - sy)));
    o.z = d.z * (sz * (1.0f + p.z * (1.0f - sz)));
    o.w = d.w * (sw * (1.0f + p.w * (1.0f - sw)));
    dpre4[idx] = o;
}

__global__ void ilmlip_silu_bwd_kernel(int N, int W, const float* __restrict__ pre, const float* __restrict__ dh,
                                       float* __restrict__ dpre) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)N * W) return;
    const float p = pre[idx];
    const float s = 1.0f / (1.0f + expf(-p));
    dpre[idx] = dh[idx] * (s * (1.0f + p * (1.0f - s)));
}

template <bool ZBL_SW, bool NEED_GRAD>
__global__ void __launch_bounds__(128) ilmlip_energy_ro_kernel(
    int N, int W, float inv_zc, const int* __restrict__ species,
    const float* __restrict__ h_last, const float* __restrict__ u_Z,
    const double* __restrict__ e_zbl, double* __restrict__ energy, double* __restrict__ atom_energy,
    float* __restrict__ dh_last, float* __restrict__ w_zbl_out) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    double e_atom = 0.0;
    if (i < N) {
        const int sp = species[i];
        const float* hi = h_last + (int64_t)i * W;
        const float* us = u_Z + (int64_t)sp * W;
        float dot = 0.0f;
        for (int w = lane; w < W; w += 32) dot = fmaf(hi[w], __ldg(us + w), dot);
        const float e_ml = warp_sum(dot);
        float sw = 1.0f;
        float wz = 0.0f;
        double node_inter = (double)e_ml;
        if constexpr (ZBL_SW) {
            const double ez = e_zbl[i];
            const double inv_zc_d = (double)inv_zc;
            const double xc = ez * inv_zc_d;
            if (xc <= 0.0) {
                sw = 1.0f;
                wz = 1.0f;
                node_inter = (double)e_ml + ez;
            } else if (xc >= 1.0) {
                sw = 0.0f;
                wz = 1.0f;
                node_inter = ez;
            } else {
                const double xc2 = xc * xc;
                const double xc3 = xc2 * xc;
                const double sw_d = 1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2;
                const double omx = 1.0 - xc;
                sw = (float)sw_d;
                wz = (float)(1.0 + (double)e_ml * (-30.0 * xc2 * omx * omx) * inv_zc_d);
                node_inter = sw_d * (double)e_ml + ez;
            }
        }
        if (lane == 0) {
            e_atom = node_inter;
            if (atom_energy) atom_energy[i] += e_atom;
            if (NEED_GRAD && ZBL_SW) w_zbl_out[i] = wz;
        }
        if (NEED_GRAD) {
            float* dhi = dh_last + (int64_t)i * W;
            for (int w = lane; w < W; w += 32) dhi[w] = sw * __ldg(us + w);
        }
    }
    if (lane == 0 && e_atom != 0.0) atomicAdd(energy, e_atom);
}

// =================================================================================================
// Kernel 6s: Zero-Spill Fast Layer-2 Invariants Backward (lout2==2, canonical 4 bisp, r_gram<=32)
// Stages P and symmetric Gmat[r_gram][r_gram+1] in SMEM (0 triangular index math, 0 bank conflicts)
// and computes all 4 bispectrum backward grads in pure scalar registers (0 bytes stack frame!).
// =================================================================================================
__global__ void __launch_bounds__(128) ilmlip_l2_inv_bwd_l2_kernel(
    int N, int r_gram, int r_bisp, int n_inv1, int n_feat, int bisp_off,
    Int4 l2_c, Int4 gram_off,
    const int32_t* __restrict__ triu_i, const int32_t* __restrict__ triu_j,
    const float* __restrict__ inv_std_x, float* __restrict__ dx,
    float* __restrict__ dA2_0, const float* __restrict__ Q0, float* __restrict__ dQ0,
    CPtr4 PQ, Ptr4 dPQ) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    const int n_tri = (r_gram > 0) ? ((r_gram * (r_gram + 1)) >> 1) : 0;
    int32_t* s_ti = reinterpret_cast<int32_t*>(smem);
    int32_t* s_tj = s_ti + n_tri;
    float* work_base = reinterpret_cast<float*>(s_tj + n_tri);
    if (n_tri > 0) {
        for (int k = tid; k < n_tri; k += 128) {
            s_ti[k] = __ldg(triu_i + k);
            s_tj[k] = __ldg(triu_j + k);
        }
        __syncthreads();
    }

    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int r_gb = r_gram + r_bisp;
    float* dxi = dx + (int64_t)i * n_feat;
    if ((n_inv1 & 3) == 0 && (n_feat & 3) == 0) {
        float4* dx4 = reinterpret_cast<float4*>(dxi);
        const float4* is4 = reinterpret_cast<const float4*>(inv_std_x);
        const int n4 = n_inv1 >> 2;
        for (int f = lane; f < n4; f += 32) {
            float4 v = dx4[f];
            const float4 s = __ldg(is4 + f);
            v.x *= s.x; v.y *= s.y; v.z *= s.z; v.w *= s.w;
            dx4[f] = v;
        }
    } else {
        for (int f = lane; f < n_inv1; f += 32) dxi[f] *= __ldg(inv_std_x + f);
    }

    const int c0 = l2_c.v[0];
    if ((c0 & 3) == 0 && (n_inv1 & 3) == 0 && (n_feat & 3) == 0) {
        const float4* dx4 = reinterpret_cast<const float4*>(dxi + n_inv1);
        const float4* is4 = reinterpret_cast<const float4*>(inv_std_x + n_inv1);
        float4* dst4 = reinterpret_cast<float4*>(dA2_0 + (int64_t)i * c0);
        const int c04 = c0 >> 2;
        for (int c = lane; c < c04; c += 32) {
            const float4 v = dx4[c];
            const float4 s = __ldg(is4 + c);
            dst4[c] = make_float4(v.x * s.x, v.y * s.y, v.z * s.z, v.w * s.w);
        }
    } else {
        for (int c = lane; c < c0; c += 32) {
            const int f = n_inv1 + c;
            dA2_0[(int64_t)i * c0 + c] = dxi[f] * __ldg(inv_std_x + f);
        }
    }

    if (r_gram > 0) {
        const int g_stride = r_gram + 1;
        const int per_warp = (5 + r_gram) * g_stride;
        float* sP = work_base + (size_t)wid * per_warp;
        float* sG = sP + (size_t)5 * g_stride;

        auto do_gram_L = [&](int dm, float inv_s, const float* PL, float* dPL, int goff) {
            for (int idx = lane; idx < dm * r_gram; idx += 32) {
                const int m = idx / r_gram, a = idx - m * r_gram;
                sP[m * g_stride + a] = __ldg(PL + m * r_gb + a);
            }
            for (int idx = lane; idx < n_tri; idx += 32) {
                const int a = s_ti[idx];
                const int b = s_tj[idx];
                const int f = goff + idx;
                const float g = __ldg(dxi + f) * __ldg(inv_std_x + f);
                if (a == b) {
                    sG[a * g_stride + a] = 2.0f * g;
                } else {
                    sG[a * g_stride + b] = g;
                    sG[b * g_stride + a] = g;
                }
            }
            __syncwarp();
            const int total = dm * r_gram;
            for (int idx = lane; idx < total; idx += 32) {
                const int m = idx / r_gram, a = idx - m * r_gram;
                const float* pm = sP + m * g_stride;
                const float* ga = sG + a * g_stride;
                float acc = 0.0f;
                if (r_gram == 16) {
                    #pragma unroll
                    for (int b = 0; b < 16; ++b) acc = fmaf(ga[b], pm[b], acc);
                } else {
                    for (int b = 0; b < r_gram; ++b) acc = fmaf(ga[b], pm[b], acc);
                }
                dPL[m * r_gb + a] = acc * inv_s;
            }
            __syncwarp();
        };

        if (l2_c.v[1] > 0) {
            do_gram_L(3, 0.57735026918962576451f,
                      PQ.p[1] + (int64_t)i * 3 * r_gb, dPQ.p[1] + (int64_t)i * 3 * r_gb, gram_off.v[1]);
        }
        if (l2_c.v[2] > 0) {
            do_gram_L(5, 0.44721359549995793928f,
                      PQ.p[2] + (int64_t)i * 5 * r_gb, dPQ.p[2] + (int64_t)i * 5 * r_gb, gram_off.v[2]);
        }
    }

    if (r_bisp > 0) {
        for (int b = lane; b < r_bisp; b += 32) {
            const float q0 = Q0[(int64_t)i * r_bisp + b];
            const float* p1 = PQ.p[1] + (int64_t)i * 3 * r_gb + r_gram + b;
            const float q1_0 = p1[0], q1_1 = p1[r_gb], q1_2 = p1[2 * r_gb];
            const float* p2 = PQ.p[2] + (int64_t)i * 5 * r_gb + r_gram + b;
            const float q2_0 = p2[0], q2_1 = p2[r_gb], q2_2 = p2[2 * r_gb], q2_3 = p2[3 * r_gb], q2_4 = p2[4 * r_gb];

            const int f0 = bisp_off + 0 * r_bisp + b;
            const int f1 = bisp_off + 1 * r_bisp + b;
            const int f2 = bisp_off + 2 * r_bisp + b;
            const int f3 = bisp_off + 3 * r_bisp + b;
            const float gt0 = dxi[f0] * __ldg(inv_std_x + f0);
            const float gt1 = dxi[f1] * __ldg(inv_std_x + f1);
            const float gt2 = dxi[f2] * __ldg(inv_std_x + f2);
            const float gt3 = dxi[f3] * __ldg(inv_std_x + f3);

            constexpr float C011   =  0.8326832056045532f;
            constexpr float C022   =  0.7647244930267334f;
            constexpr float C112_a = -0.34432995319366455f;
            constexpr float C112_b =  0.5963969826698303f;
            constexpr float C112_c =  0.6886599063873291f;
            constexpr float C222_a =  0.5345224738121033f;
            constexpr float C222_b =  0.26726123690605164f;
            constexpr float C222_c =  0.4629100561141968f;

            const float q1_0s = q1_0 * q1_0, q1_1s = q1_1 * q1_1, q1_2s = q1_2 * q1_2;
            const float q2_0s = q2_0 * q2_0, q2_1s = q2_1 * q2_1, q2_2s = q2_2 * q2_2, q2_3s = q2_3 * q2_3, q2_4s = q2_4 * q2_4;

            // t = 0: BISP(0,1,1)
            const float g0 = gt0 * C011;
            float dq0 = g0 * (q1_0s + q1_1s + q1_2s);
            const float g0q = 2.0f * g0 * q0;
            float dq1_0 = g0q * q1_0, dq1_1 = g0q * q1_1, dq1_2 = g0q * q1_2;

            // t = 1: BISP(0,2,2)
            const float g1 = gt1 * C022;
            dq0 = fmaf(g1, q2_0s + q2_1s + q2_2s + q2_3s + q2_4s, dq0);
            const float g1q = 2.0f * g1 * q0;
            float dq2_0 = g1q * q2_0, dq2_1 = g1q * q2_1, dq2_2 = g1q * q2_2, dq2_3 = g1q * q2_3, dq2_4 = g1q * q2_4;

            // t = 2: BISP(1,1,2)
            const float g2a = gt2 * C112_a, g2b = gt2 * C112_b, g2c = gt2 * C112_c;
            dq1_0 += 2.0f * (g2a * q1_0 * q2_2 + g2b * (q1_2 * q2_0 + q1_1 * q2_1 - q1_0 * q2_4));
            dq1_1 += 2.0f * (g2c * q1_1 * q2_2 + g2b * (q1_0 * q2_1 + q1_2 * q2_3));
            dq1_2 += 2.0f * (g2a * q1_2 * q2_2 + g2b * (q1_0 * q2_0 + q1_1 * q2_3 + q1_2 * q2_4));
            dq2_0 = fmaf(2.0f * g2b, q1_0 * q1_2, dq2_0);
            dq2_1 = fmaf(2.0f * g2b, q1_0 * q1_1, dq2_1);
            dq2_2 += g2a * (q1_0s + q1_2s) + g2c * q1_1s;
            dq2_3 = fmaf(2.0f * g2b, q1_1 * q1_2, dq2_3);
            dq2_4 = fmaf(g2b, q1_2s - q1_0s, dq2_4);

            // t = 3: BISP(2,2,2)
            const float g3a = 3.0f * gt3 * C222_a, g3b = 3.0f * gt3 * C222_b, g3c = 3.0f * gt3 * C222_c;
            dq2_0 += -2.0f * g3a * q2_2 * q2_0 + 2.0f * g3c * q2_1 * q2_3;
            dq2_1 +=  2.0f * g3b * q2_2 * q2_1 + 2.0f * g3c * (q2_0 * q2_3 - q2_4 * q2_1);
            dq2_2 +=  g3a * (q2_2s - q2_0s - q2_4s) + g3b * (q2_1s + q2_3s);
            dq2_3 +=  2.0f * g3b * q2_2 * q2_3 + 2.0f * g3c * (q2_0 * q2_1 + q2_4 * q2_3);
            dq2_4 += -2.0f * g3a * q2_2 * q2_4 + g3c * (q2_3s - q2_1s);

            dQ0[(int64_t)i * r_bisp + b] = dq0;
            float* dp1 = dPQ.p[1] + (int64_t)i * 3 * r_gb + r_gram + b;
            dp1[0] = dq1_0; dp1[r_gb] = dq1_1; dp1[2 * r_gb] = dq1_2;
            float* dp2 = dPQ.p[2] + (int64_t)i * 5 * r_gb + r_gram + b;
            dp2[0] = dq2_0; dp2[r_gb] = dq2_1; dp2[2 * r_gb] = dq2_2; dp2[3 * r_gb] = dq2_3; dp2[4 * r_gb] = dq2_4;
        }
    }
}

// =================================================================================================
// Kernel 6a: Layer-2 Invariants Backward (dx_norm -> dx, dA2[0], dQ0, dPQ[1..lout2])
// Inline inv_std_x scaling, smem bisp tables + smem-cached Q/dQ + smem-cached Gram grad!
// =================================================================================================
__global__ void __launch_bounds__(128) ilmlip_l2_inv_bwd_kernel(
    int N, int lout2, int r_gram, int r_bisp, int n_bisp, int bisp_idx_cnt, int n_inv1, int n_feat, int bisp_off,
    Int4 l2_c, Int4 gram_off,
    const int32_t* __restrict__ bisp_tbl, const int32_t* __restrict__ bisp_idx, const float* __restrict__ bisp_val,
    const float* __restrict__ inv_std_x, float* __restrict__ dx,
    float* __restrict__ dA2_0, const float* __restrict__ Q0, float* __restrict__ dQ0,
    CPtr4 PQ, Ptr4 dPQ) {
    extern __shared__ float smem[];
    const int n_bisp_ints = n_bisp * 5;
    int32_t* s_btbl = reinterpret_cast<int32_t*>(smem);
    int32_t* s_bidx = s_btbl + n_bisp_ints;
    float* s_bval = reinterpret_cast<float*>(s_bidx + bisp_idx_cnt);
    float* s_work = s_bval + bisp_idx_cnt;

    if (r_bisp > 0 && n_bisp > 0) {
        for (int k = threadIdx.x; k < n_bisp_ints; k += 128) s_btbl[k] = __ldg(bisp_tbl + k);
        for (int k = threadIdx.x; k < bisp_idx_cnt; k += 128) {
            s_bidx[k] = __ldg(bisp_idx + k);
            s_bval[k] = __ldg(bisp_val + k);
        }
        __syncthreads();
    }

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int r_gb = r_gram + r_bisp;
    float* dxi = dx + (int64_t)i * n_feat;
    // Scale Layer-1 invariant grads [0 .. n_inv1 - 1] in place; L2 parts scale inline on read!
    for (int f = lane; f < n_inv1; f += 32) dxi[f] *= __ldg(inv_std_x + f);

    const int c0 = l2_c.v[0];
    for (int c = lane; c < c0; c += 32) {
        const int f = n_inv1 + c;
        dA2_0[(int64_t)i * c0 + c] = dxi[f] * __ldg(inv_std_x + f);
    }

    if (r_gram > 0) {
        const int n_tri = (r_gram * (r_gram + 1)) >> 1;
        const bool use_smem_g = (r_gram <= 16);
        float* sg = use_smem_g ? (s_work + (size_t)wid * n_tri) : nullptr;
        for (int L = 1; L <= lout2; ++L) {
            if (l2_c.v[L] == 0) continue;
            const int dm = 2 * L + 1;
            const float inv_s = 1.0f / sqrtf((float)dm);
            const float* PL = PQ.p[L] + (int64_t)i * dm * r_gb;
            float* dPL = dPQ.p[L] + (int64_t)i * dm * r_gb;
            const int goff = gram_off.v[L];
            if (use_smem_g) {
                for (int idx = lane; idx < n_tri; idx += 32) {
                    const int f = goff + idx;
                    sg[idx] = dxi[f] * __ldg(inv_std_x + f);
                }
                __syncwarp();
            }
            const int total = dm * r_gram;
            for (int idx = lane; idx < total; idx += 32) {
                const int m = idx / r_gram;
                const int a = idx - m * r_gram;
                float acc = 0.0f;
                const float* Pm = PL + m * r_gb;
                for (int b = 0; b < r_gram; ++b) {
                    const int lo = a < b ? a : b;
                    const int hi = a < b ? b : a;
                    const int tri = lo * r_gram - ((lo * (lo - 1)) >> 1) + (hi - lo);
                    float g = use_smem_g ? sg[tri] : (dxi[goff + tri] * __ldg(inv_std_x + goff + tri));
                    if (a == b) g *= 2.0f;
                    acc = fmaf(g, Pm[b], acc);
                }
                dPL[m * r_gb + a] = acc * inv_s;
            }
            if (use_smem_g) __syncwarp();
        }
    }

    if (r_bisp > 0 && n_bisp > 0) {
        if (lout2 <= 2 && r_bisp <= 32) {
            float* sq = s_work + (size_t)wid * 18 * r_bisp;
            float* sdq = sq + 9 * r_bisp;
            if (lane < r_bisp) {
                if (l2_c.v[0] > 0) sq[0 * r_bisp + lane] = Q0[(int64_t)i * r_bisp + lane];
                if (l2_c.v[1] > 0) {
                    const float* q1 = PQ.p[1] + (int64_t)i * 3 * r_gb + r_gram + lane;
                    sq[1 * r_bisp + lane] = q1[0];
                    sq[2 * r_bisp + lane] = q1[r_gb];
                    sq[3 * r_bisp + lane] = q1[2 * r_gb];
                }
                if (lout2 == 2 && l2_c.v[2] > 0) {
                    const float* q2 = PQ.p[2] + (int64_t)i * 5 * r_gb + r_gram + lane;
                    #pragma unroll
                    for (int m = 0; m < 5; ++m) sq[(4 + m) * r_bisp + lane] = q2[m * r_gb];
                }
                #pragma unroll
                for (int c = 0; c < 9; ++c) sdq[c * r_bisp + lane] = 0.0f;

                for (int t = 0; t < n_bisp; ++t) {
                    const int32_t* br = s_btbl + t * 5;
                    const int o1 = br[0] * br[0], o2 = br[1] * br[1], o3 = br[2] * br[2];
                    const int off = br[3], cnt = br[4];
                    const int f = bisp_off + t * r_bisp + lane;
                    const float gt = dxi[f] * __ldg(inv_std_x + f);
                    for (int k = 0; k < cnt; ++k) {
                        const int pk = s_bidx[off + k];
                        const int i1 = (o1 + (pk & 15)) * r_bisp + lane;
                        const int i2 = (o2 + ((pk >> 4) & 15)) * r_bisp + lane;
                        const int i3 = (o3 + ((pk >> 8) & 15)) * r_bisp + lane;
                        const float g = gt * s_bval[off + k];
                        const float q1 = sq[i1], q2 = sq[i2], q3 = sq[i3];
                        sdq[i1] = fmaf(g, q2 * q3, sdq[i1]);
                        sdq[i2] = fmaf(g, q1 * q3, sdq[i2]);
                        sdq[i3] = fmaf(g, q1 * q2, sdq[i3]);
                    }
                }

                if (l2_c.v[0] > 0) dQ0[(int64_t)i * r_bisp + lane] = sdq[0 * r_bisp + lane];
                if (l2_c.v[1] > 0) {
                    float* dq1 = dPQ.p[1] + (int64_t)i * 3 * r_gb + r_gram + lane;
                    dq1[0]        = sdq[1 * r_bisp + lane];
                    dq1[r_gb]     = sdq[2 * r_bisp + lane];
                    dq1[2 * r_gb] = sdq[3 * r_bisp + lane];
                }
                if (lout2 == 2 && l2_c.v[2] > 0) {
                    float* dq2 = dPQ.p[2] + (int64_t)i * 5 * r_gb + r_gram + lane;
                    #pragma unroll
                    for (int m = 0; m < 5; ++m) dq2[m * r_gb] = sdq[(4 + m) * r_bisp + lane];
                }
            }
        } else {
            if (l2_c.v[0] > 0) {
                float* dq0 = dQ0 + (int64_t)i * r_bisp;
                for (int idx = lane; idx < r_bisp; idx += 32) dq0[idx] = 0.0f;
            }
            for (int L = 1; L <= lout2; ++L) {
                if (l2_c.v[L] == 0) continue;
                const int dm = 2 * L + 1;
                float* dQL = dPQ.p[L] + (int64_t)i * dm * r_gb + r_gram;
                for (int m = 0; m < dm; ++m)
                    for (int idx = lane; idx < r_bisp; idx += 32) dQL[m * r_gb + idx] = 0.0f;
            }
            __syncwarp();
            for (int b = lane; b < r_bisp; b += 32) {
                for (int t = 0; t < n_bisp; ++t) {
                    const int32_t* br = s_btbl + t * 5;
                    const int L1 = br[0], L2 = br[1], L3 = br[2], off = br[3], cnt = br[4];
                    const int f = bisp_off + t * r_bisp + b;
                    const float gt = dxi[f] * __ldg(inv_std_x + f);
                    const int st1 = (L1 == 0) ? r_bisp : r_gb;
                    const int st2 = (L2 == 0) ? r_bisp : r_gb;
                    const int st3 = (L3 == 0) ? r_bisp : r_gb;
                    const float* Q1 = (L1 == 0) ? (Q0 + (int64_t)i * r_bisp + b) : (PQ.p[L1] + (int64_t)i * (2 * L1 + 1) * r_gb + r_gram + b);
                    const float* Q2 = (L2 == 0) ? (Q0 + (int64_t)i * r_bisp + b) : (PQ.p[L2] + (int64_t)i * (2 * L2 + 1) * r_gb + r_gram + b);
                    const float* Q3 = (L3 == 0) ? (Q0 + (int64_t)i * r_bisp + b) : (PQ.p[L3] + (int64_t)i * (2 * L3 + 1) * r_gb + r_gram + b);
                    float* dQ1 = (L1 == 0) ? (dQ0 + (int64_t)i * r_bisp + b) : (dPQ.p[L1] + (int64_t)i * (2 * L1 + 1) * r_gb + r_gram + b);
                    float* dQ2 = (L2 == 0) ? (dQ0 + (int64_t)i * r_bisp + b) : (dPQ.p[L2] + (int64_t)i * (2 * L2 + 1) * r_gb + r_gram + b);
                    float* dQ3 = (L3 == 0) ? (dQ0 + (int64_t)i * r_bisp + b) : (dPQ.p[L3] + (int64_t)i * (2 * L3 + 1) * r_gb + r_gram + b);
                    for (int k = 0; k < cnt; ++k) {
                        const int pk = s_bidx[off + k];
                        const int m1 = pk & 15, m2 = (pk >> 4) & 15, m3 = (pk >> 8) & 15;
                        const float g = gt * s_bval[off + k];
                        const float q1 = Q1[m1 * st1], q2 = Q2[m2 * st2], q3 = Q3[m3 * st3];
                        dQ1[m1 * st1] += g * q2 * q3;
                        dQ2[m2 * st2] += g * q1 * q3;
                        dQ3[m3 * st3] += g * q1 * q2;
                    }
                }
            }
        }
    } else if (r_bisp > 0) {
        // Edge case: r_bisp > 0 but n_bisp == 0 -> zero dQ slots so W_GB^T reads clean zeros
        if (l2_c.v[0] > 0) {
            float* dq0 = dQ0 + (int64_t)i * r_bisp;
            for (int idx = lane; idx < r_bisp; idx += 32) dq0[idx] = 0.0f;
        }
        for (int L = 1; L <= lout2; ++L) {
            if (l2_c.v[L] == 0) continue;
            const int dm = 2 * L + 1;
            float* dQL = dPQ.p[L] + (int64_t)i * dm * r_gb + r_gram;
            for (int m = 0; m < dm; ++m)
                for (int idx = lane; idx < r_bisp; idx += 32) dQL[m * r_gb + idx] = 0.0f;
        }
    }
}

// =================================================================================================
// Shared RMS Bond-Sum Gate Backward (compile-time N_DEG, float4 vectorized -> 0 bytes stack frame!)
// =================================================================================================
template <int N_DEG>
__global__ void __launch_bounds__(128) ilmlip_gate_bwd_kernel(
    int N, Int4 C_deg, float inv_DA_a0_kap_sq,
    CPtr4 A, Ptr4 dA, const float* __restrict__ f_arr, const float* __restrict__ df_ext) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    float dot_lane = 0.0f;
    #pragma unroll
    for (int l = 0; l < N_DEG; ++l) {
        const int sz = (2 * l + 1) * C_deg.v[l];
        if (sz == 0) continue;
        const float* Al = A.p[l] + (int64_t)i * sz;
        const float* dAl = dA.p[l] + (int64_t)i * sz;
        if ((sz & 3) == 0) {
            const int sz4 = sz >> 2;
            const float4* A4 = reinterpret_cast<const float4*>(Al);
            const float4* dA4 = reinterpret_cast<const float4*>(dAl);
            for (int k = lane; k < sz4; k += 32) {
                const float4 a = A4[k];
                const float4 da = dA4[k];
                dot_lane = fmaf(da.x, a.x, fmaf(da.y, a.y, fmaf(da.z, a.z, fmaf(da.w, a.w, dot_lane))));
            }
        } else {
            for (int k = lane; k < sz; k += 32) dot_lane = fmaf(dAl[k], Al[k], dot_lane);
        }
    }
    const float dot = warp_sum_f(dot_lane);
    const float f = f_arr[i];
    const float df_ext_i = df_ext ? df_ext[i] : 0.0f;
    const float scale = (df_ext_i * f * f + dot * f) * inv_DA_a0_kap_sq;

    #pragma unroll
    for (int l = 0; l < N_DEG; ++l) {
        const int sz = (2 * l + 1) * C_deg.v[l];
        if (sz == 0) continue;
        const float* Al = A.p[l] + (int64_t)i * sz;
        float* dAl = dA.p[l] + (int64_t)i * sz;
        if ((sz & 3) == 0) {
            const int sz4 = sz >> 2;
            const float4* A4 = reinterpret_cast<const float4*>(Al);
            float4* dA4 = reinterpret_cast<float4*>(dAl);
            for (int k = lane; k < sz4; k += 32) {
                const float4 a = A4[k];
                const float4 da = dA4[k];
                float4 o;
                o.x = fmaf(f, da.x, -scale * a.x);
                o.y = fmaf(f, da.y, -scale * a.y);
                o.z = fmaf(f, da.z, -scale * a.z);
                o.w = fmaf(f, da.w, -scale * a.w);
                dA4[k] = o;
            }
        } else {
            for (int k = lane; k < sz; k += 32) dAl[k] = fmaf(f, dAl[k], -scale * Al[k]);
        }
    }
}

// =================================================================================================
// Fused f1 Gate Backward + Exact FP32 Pre-Pack of dA2_raw (3x float4 + 1x float = 208 floats/atom)
// =================================================================================================
__global__ void __launch_bounds__(128) ilmlip_gate_bwd_pack_l2_kernel(
    int N, int n_msg, Int4 l2_c, float inv_DA2_a0_kap_sq,
    CPtr4 A2, CPtr4 dA2, const float* __restrict__ f1_arr, float* __restrict__ dA2_pack) {
    const int sid = threadIdx.x >> 4;
    const int slane = threadIdx.x & 15;
    const unsigned smask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;

    const bool active_q = (slane < n_msg);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f;
    float a7 = 0.0f, a8 = 0.0f, a9 = 0.0f, a10 = 0.0f, a11 = 0.0f, a12 = 0.0f;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f, d4 = 0.0f, d5 = 0.0f, d6 = 0.0f;
    float d7 = 0.0f, d8 = 0.0f, d9 = 0.0f, d10 = 0.0f, d11 = 0.0f, d12 = 0.0f;
    float dot_lane = 0.0f;

    if (active_q) {
        const int c0 = l2_c.v[0], c1 = l2_c.v[1], c2 = l2_c.v[2];
        const float* pA0  = A2.p[0]  + (int64_t)i * c0 + slane;
        const float* pdA0 = dA2.p[0] + (int64_t)i * c0 + slane;
        a0 = pA0[0]; a7 = pA0[n_msg]; d0 = pdA0[0]; d7 = pdA0[n_msg];

        const float* pA1  = A2.p[1]  + (int64_t)i * 3 * c1 + slane;
        const float* pdA1 = dA2.p[1] + (int64_t)i * 3 * c1 + slane;
        a1 = pA1[0]; a2 = pA1[c1]; a3 = pA1[2 * c1];
        a4 = pA1[n_msg]; a5 = pA1[c1 + n_msg]; a6 = pA1[2 * c1 + n_msg];
        d1 = pdA1[0]; d2 = pdA1[c1]; d3 = pdA1[2 * c1];
        d4 = pdA1[n_msg]; d5 = pdA1[c1 + n_msg]; d6 = pdA1[2 * c1 + n_msg];

        const float* pA2  = A2.p[2]  + (int64_t)i * 5 * c2 + slane;
        const float* pdA2 = dA2.p[2] + (int64_t)i * 5 * c2 + slane;
        a8 = pA2[0]; a9 = pA2[c2]; a10 = pA2[2 * c2]; a11 = pA2[3 * c2]; a12 = pA2[4 * c2];
        d8 = pdA2[0]; d9 = pdA2[c2]; d10 = pdA2[2 * c2]; d11 = pdA2[3 * c2]; d12 = pdA2[4 * c2];

        dot_lane = fmaf(a0, d0, fmaf(a1, d1, fmaf(a2, d2, fmaf(a3, d3,
                   fmaf(a4, d4, fmaf(a5, d5, fmaf(a6, d6, fmaf(a7, d7,
                   fmaf(a8, d8, fmaf(a9, d9, fmaf(a10, d10, fmaf(a11, d11, a12 * d12))))))))))));
    }
    const float dot = half_warp_sum_f(dot_lane, smask);
    const float f1 = f1_arr[i];
    const float sc = (dot * f1) * inv_DA2_a0_kap_sq;

    if (active_q) {
        const float g0  = fmaf(f1, d0,  -sc * a0);
        const float g1  = fmaf(f1, d1,  -sc * a1);
        const float g2  = fmaf(f1, d2,  -sc * a2);
        const float g3  = fmaf(f1, d3,  -sc * a3);
        const float g4  = fmaf(f1, d4,  -sc * a4);
        const float g5  = fmaf(f1, d5,  -sc * a5);
        const float g6  = fmaf(f1, d6,  -sc * a6);
        const float g7  = fmaf(f1, d7,  -sc * a7);
        const float g8  = fmaf(f1, d8,  -sc * a8);
        const float g9  = fmaf(f1, d9,  -sc * a9);
        const float g10 = fmaf(f1, d10, -sc * a10);
        const float g11 = fmaf(f1, d11, -sc * a11);
        const float g12 = fmaf(f1, d12, -sc * a12);

        constexpr float C110   = 0.57735026918962576451f;
        constexpr float C_SQ2  = 0.70710678118654752440f;
        constexpr float C_SQ6  = 0.40824829046386301637f;
        constexpr float C_2SQ6 = 0.81649658092772603273f;

        const float g7c = g7 * C110;
        const float w00 = -C_SQ6 * g10 - C_SQ2 * g12;
        const float w01 =  C_SQ2 * g9;
        const float w02 =  C_SQ2 * g8;
        const float w11 =  C_2SQ6 * g10;
        const float w12 =  C_SQ2 * g11;
        // Exact FP32 layout: 13*16 = 208 floats/atom (16-byte aligned for all i & n_msg<=16; w22 = -(w00+w11))
        float* dst_f = dA2_pack + (int64_t)i * 208;
        float4* dst4 = reinterpret_cast<float4*>(dst_f);
        dst4[0 * 16 + slane] = make_float4(g0, g1, g2, g3);
        dst4[1 * 16 + slane] = make_float4(g4, g5, g6, g7c);
        dst4[2 * 16 + slane] = make_float4(w00, w01, w02, w11);
        dst_f[12 * 16 + slane] = w12;
    }
}

// =================================================================================================
// Kernel 7z: Register-Cached Species-Grouped SENDER-OWNED ZERO-ATOMIC Layer-2 Edge Backward!
// - __launch_bounds__(128, 5) -> 96 regs/thread (ZERO local-memory stack spills!)
// - Register-cached C2+Cbf (0 SMEM reads for weights), padded [8][9] float4 SMEM, exact FP32 dA2_pack!
// =================================================================================================
template <bool REM_BF, bool STRESS>
__global__ void __launch_bounds__(128, 5) ilmlip_l2_edge_bwd_sender_kernel(
    int N, int S, int n_msg, float r_max,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2, const float* __restrict__ Cbf,
    const float* __restrict__ f0, const float4* __restrict__ I_pack4, CPtr4 I_node, const float* __restrict__ dA2_pack,
    Ptr4 dI_node, float* __restrict__ df0_bf, double* __restrict__ forces, double* __restrict__ virial) {
    __shared__ float4 s_g03[8][9];
    __shared__ float4 s_g47[8][9];
    __shared__ float4 s_dg03[8][9];
    __shared__ float4 s_dg47[8][9];
    __shared__ float4 s_Y[8][9];
    __shared__ float4 s_vl[8][9];
    __shared__ float  s_er_dY[8][9][4];
    __shared__ double s_sf[8][3];
    __shared__ int    s_eq[8][17];
    __shared__ int    s_iq[8][17];

    const int sid = threadIdx.x >> 4;
    const int sub = sid & 1;
    const int slane = threadIdx.x & 15;
    const unsigned smask = sub ? 0xffff0000u : 0x0000ffffu;
    const int shift = sub ? 16 : 0;
    const int j = blockIdx.x * 8 + sid;
    if (j >= N) return;

    const int z_j = species[j];
    const int eb = row_ptr[j];
    const int ee = row_ptr[j + 1];
    const int l2_tot_cols = 2 * n_msg;
    const bool active_q = (slane < n_msg);
    const float f0_j = REM_BF ? __ldg(f0 + j) : 0.0f;

    float i0_raw = 0.0f, i1_0_raw = 0.0f, i1_1_raw = 0.0f, i1_2_raw = 0.0f;
    if (active_q) {
        if (I_pack4) {
            const float4 ir = __ldg(I_pack4 + (int64_t)j * n_msg + slane);
            i0_raw = ir.x; i1_0_raw = ir.y; i1_1_raw = ir.z; i1_2_raw = ir.w;
        } else {
            i0_raw = __ldg(I_node.p[0] + (int64_t)j * n_msg + slane);
            const float* pI1 = I_node.p[1] + (int64_t)j * 3 * n_msg + slane;
            i1_0_raw = __ldg(pI1);
            i1_1_raw = __ldg(pI1 + n_msg);
            i1_2_raw = __ldg(pI1 + 2 * n_msg);
        }
    }
    if (slane < 3) s_sf[sid][slane] = 0.0;

    float acc_di0 = 0.0f, acc_di1_0 = 0.0f, acc_di1_1 = 0.0f, acc_di1_2 = 0.0f;
    float acc_df0 = 0.0f;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int zi = 0; zi < S; ++zi) {
        const int pair = zi * S + z_j;
        const float* C2p = C2 + (int64_t)pair * 8 * l2_tot_cols + slane;
        const float* Cbfp = REM_BF ? (Cbf + (int64_t)pair * 8 * l2_tot_cols + slane) : nullptr;
        float c2_0[8], c2_1[8], cb_0[8], cb_1[8];
        if (active_q) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                c2_0[k] = __ldg(C2p + k * l2_tot_cols);
                c2_1[k] = __ldg(C2p + k * l2_tot_cols + n_msg);
                if constexpr (REM_BF) {
                    cb_0[k] = __ldg(Cbfp + k * l2_tot_cols);
                    cb_1[k] = __ldg(Cbfp + k * l2_tot_cols + n_msg);
                }
            }
        }

        auto flush_q = [&](int nq) {
            for (int qb = 0; qb < nq; qb += 8) {
                const int bcnt = (nq - qb < 8) ? (nq - qb) : 8;
                if (slane < bcnt) {
                    const int e = s_eq[sid][qb + slane];
                    const float4 vl_rev = __ldg(vec_len + e);
                    const float4 vl_fwd = make_float4(-vl_rev.x, -vl_rev.y, -vl_rev.z, vl_rev.w);
                    const float r = fmaxf(vl_rev.w, 1e-8f);
                    const float sc = 0x1.bb67ae0000000p+0f / r;
                    s_vl[sid][slane] = vl_fwd;
                    s_Y[sid][slane] = make_float4(1.0f, vl_fwd.x * sc, vl_fwd.y * sc, vl_fwd.z * sc);
                    float4 g03, g47, dg03, dg47;
                    eval_cheb8_f4<true>(r, r_max, g03, g47, dg03, dg47);
                    s_g03[sid][slane] = g03;   s_g47[sid][slane] = g47;
                    s_dg03[sid][slane] = dg03; s_dg47[sid][slane] = dg47;
                }
                __syncwarp(smask);

                for (int bi = 0; bi < bcnt; ++bi) {
                    const int i_nb = s_iq[sid][qb + bi];
                    float er_lane = 0.0f;
                    float dY1 = 0.0f, dY2 = 0.0f, dY3 = 0.0f;

                    if (active_q) {
                        const float* pbase = dA2_pack + (int64_t)i_nb * 208;
                        const float4* p4 = reinterpret_cast<const float4*>(pbase);
                        const float4 b0 = __ldg(p4 + 0 * 16 + slane);
                        const float4 b1 = __ldg(p4 + 1 * 16 + slane);
                        const float4 b2 = __ldg(p4 + 2 * 16 + slane);
                        const float  w12 = __ldg(pbase + 12 * 16 + slane);

                        const float4 g03 = s_g03[sid][bi];
                        const float4 g47 = s_g47[sid][bi];
                        const float4 Y   = s_Y[sid][bi];

                        const float r2_0 = fmaf(c2_0[7], g47.w, fmaf(c2_0[6], g47.z, fmaf(c2_0[5], g47.y, fmaf(c2_0[4], g47.x,
                                           fmaf(c2_0[3], g03.w, fmaf(c2_0[2], g03.z, fmaf(c2_0[1], g03.y, c2_0[0] * g03.x)))))));
                        const float r2_1 = fmaf(c2_1[7], g47.w, fmaf(c2_1[6], g47.z, fmaf(c2_1[5], g47.y, fmaf(c2_1[4], g47.x,
                                           fmaf(c2_1[3], g03.w, fmaf(c2_1[2], g03.z, fmaf(c2_1[1], g03.y, c2_1[0] * g03.x)))))));

                        float rbf0 = 0.0f, rbf1 = 0.0f;
                        if constexpr (REM_BF) {
                            rbf0 = fmaf(cb_0[7], g47.w, fmaf(cb_0[6], g47.z, fmaf(cb_0[5], g47.y, fmaf(cb_0[4], g47.x,
                                   fmaf(cb_0[3], g03.w, fmaf(cb_0[2], g03.z, fmaf(cb_0[1], g03.y, cb_0[0] * g03.x)))))));
                            rbf1 = fmaf(cb_1[7], g47.w, fmaf(cb_1[6], g47.z, fmaf(cb_1[5], g47.y, fmaf(cb_1[4], g47.x,
                                   fmaf(cb_1[3], g03.w, fmaf(cb_1[2], g03.z, fmaf(cb_1[1], g03.y, cb_1[0] * g03.x)))))));
                        }

                        const float i0   = i0_raw   - (REM_BF ? (f0_j * rbf0) : 0.0f);
                        const float i1_0 = i1_0_raw - (REM_BF ? (f0_j * rbf1) * Y.y : 0.0f);
                        const float i1_1 = i1_1_raw - (REM_BF ? (f0_j * rbf1) * Y.z : 0.0f);
                        const float i1_2 = i1_2_raw - (REM_BF ? (f0_j * rbf1) * Y.w : 0.0f);

                        const float d_r2_0 = fmaf(b0.x, i0, b0.y * i1_0 + b0.z * i1_1 + b0.w * i1_2);
                        float di0   = b0.x * r2_0;
                        float di1_0 = b0.y * r2_0;
                        float di1_1 = b0.z * r2_0;
                        float di1_2 = b0.w * r2_0;

                        const float gy = b1.x * Y.y + b1.y * Y.z + b1.z * Y.w;
                        float d_r2_1 = gy * i0;
                        const float r2i0 = r2_1 * i0;
                        dY1 = b1.x * r2i0; dY2 = b1.y * r2i0; dY3 = b1.z * r2i0;
                        di0 = fmaf(r2_1, gy, di0);
                        d_r2_1 = fmaf(b1.w, Y.y * i1_0 + Y.z * i1_1 + Y.w * i1_2, d_r2_1);
                        const float gr = b1.w * r2_1;
                        dY1 = fmaf(gr, i1_0, dY1); dY2 = fmaf(gr, i1_1, dY2); dY3 = fmaf(gr, i1_2, dY3);
                        di1_0 = fmaf(gr, Y.y, di1_0); di1_1 = fmaf(gr, Y.z, di1_1); di1_2 = fmaf(gr, Y.w, di1_2);

                        const float w00 = b2.x, w01 = b2.y, w02 = b2.z, w11 = b2.w;
                        const float w22 = -(w00 + w11);
                        const float s0 = w00 * i1_0 + w01 * i1_1 + w02 * i1_2;
                        const float s1 = w01 * i1_0 + w11 * i1_1 + w12 * i1_2;
                        const float s2 = w02 * i1_0 + w12 * i1_1 + w22 * i1_2;
                        d_r2_1 += Y.y * s0 + Y.z * s1 + Y.w * s2;
                        dY1 = fmaf(r2_1, s0, dY1); dY2 = fmaf(r2_1, s1, dY2); dY3 = fmaf(r2_1, s2, dY3);
                        di1_0 = fmaf(r2_1, w00 * Y.y + w01 * Y.z + w02 * Y.w, di1_0);
                        di1_1 = fmaf(r2_1, w01 * Y.y + w11 * Y.z + w12 * Y.w, di1_1);
                        di1_2 = fmaf(r2_1, w02 * Y.y + w12 * Y.z + w22 * Y.w, di1_2);

                        const float4 dg03 = s_dg03[sid][bi];
                        const float4 dg47 = s_dg47[sid][bi];
                        const float dr2_0 = fmaf(c2_0[7], dg47.w, fmaf(c2_0[6], dg47.z, fmaf(c2_0[5], dg47.y, fmaf(c2_0[4], dg47.x,
                                            fmaf(c2_0[3], dg03.w, fmaf(c2_0[2], dg03.z, fmaf(c2_0[1], dg03.y, c2_0[0] * dg03.x)))))));
                        const float dr2_1 = fmaf(c2_1[7], dg47.w, fmaf(c2_1[6], dg47.z, fmaf(c2_1[5], dg47.y, fmaf(c2_1[4], dg47.x,
                                            fmaf(c2_1[3], dg03.w, fmaf(c2_1[2], dg03.z, fmaf(c2_1[1], dg03.y, c2_1[0] * dg03.x)))))));
                        er_lane = fmaf(d_r2_0, dr2_0, d_r2_1 * dr2_1);

                        acc_di0   += di0;
                        acc_di1_0 += di1_0;
                        acc_di1_1 += di1_1;
                        acc_di1_2 += di1_2;

                        if constexpr (REM_BF) {
                            const float drbf0 = fmaf(cb_0[7], dg47.w, fmaf(cb_0[6], dg47.z, fmaf(cb_0[5], dg47.y, fmaf(cb_0[4], dg47.x,
                                                fmaf(cb_0[3], dg03.w, fmaf(cb_0[2], dg03.z, fmaf(cb_0[1], dg03.y, cb_0[0] * dg03.x)))))));
                            const float drbf1 = fmaf(cb_1[7], dg47.w, fmaf(cb_1[6], dg47.z, fmaf(cb_1[5], dg47.y, fmaf(cb_1[4], dg47.x,
                                                fmaf(cb_1[3], dg03.w, fmaf(cb_1[2], dg03.z, fmaf(cb_1[1], dg03.y, cb_1[0] * dg03.x)))))));
                            acc_df0 = fmaf(-di0, rbf0, acc_df0);
                            er_lane = fmaf(-di0 * f0_j, drbf0, er_lane);
                            const float ydot = di1_0 * Y.y + di1_1 * Y.z + di1_2 * Y.w;
                            acc_df0 = fmaf(-ydot, rbf1, acc_df0);
                            er_lane = fmaf(-ydot * f0_j, drbf1, er_lane);
                            const float bf1_sc = -f0_j * rbf1;
                            dY1 = fmaf(di1_0, bf1_sc, dY1);
                            dY2 = fmaf(di1_1, bf1_sc, dY2);
                            dY3 = fmaf(di1_2, bf1_sc, dY3);
                        }
                    }

                    const bool p0 = (slane & 1) != 0;
                    const float u0 = (p0 ? dY1 : er_lane) + __shfl_xor_sync(smask, p0 ? er_lane : dY1, 1);
                    const float u1 = (p0 ? dY3 : dY2)     + __shfl_xor_sync(smask, p0 ? dY2 : dY3, 1);
                    const bool p1 = (slane & 2) != 0;
                    float val = (p1 ? u1 : u0) + __shfl_xor_sync(smask, p1 ? u0 : u1, 2);
                    val += __shfl_xor_sync(smask, val, 4);
                    val += __shfl_xor_sync(smask, val, 8);
                    if (slane < 4) s_er_dY[sid][bi][slane] = val;
                }
                __syncwarp(smask);

                double bfx = 0.0, bfy = 0.0, bfz = 0.0;
                if (slane < bcnt) {
                    const float4 vl = s_vl[sid][slane];
                    const int i_nb = s_iq[sid][qb + slane];
                    const float r = fmaxf(vl.w, 1e-8f);
                    const float inv_d = 1.0f / r;
                    const float ux = vl.x * inv_d, uy = vl.y * inv_d, uz = vl.z * inv_d;
                    const float4 ed = *reinterpret_cast<const float4*>(s_er_dY[sid][slane]);
                    float gx, gy, gz;
                    sh_grad_l1(ed.y, ed.z, ed.w, gx, gy, gz);
                    const float dot = ux * gx + uy * gy + uz * gz;
                    bfx = (double)(ux * ed.x + (gx - ux * dot) * inv_d);
                    bfy = (double)(uy * ed.x + (gy - uy * dot) * inv_d);
                    bfz = (double)(uz * ed.x + (gz - uz * dot) * inv_d);
                    atomicAdd(forces + (int64_t)i_nb * 3 + 0, -bfx);
                    atomicAdd(forces + (int64_t)i_nb * 3 + 1, -bfy);
                    atomicAdd(forces + (int64_t)i_nb * 3 + 2, -bfz);
                    if constexpr (STRESS) {
                        v0 += (double)vl.x * bfx;
                        v1 += 0.5 * ((double)vl.x * bfy + (double)vl.y * bfx);
                        v2 += 0.5 * ((double)vl.x * bfz + (double)vl.z * bfx);
                        v3 += (double)vl.y * bfy;
                        v4 += 0.5 * ((double)vl.y * bfz + (double)vl.z * bfy);
                        v5 += (double)vl.z * bfz;
                    }
                }
                bfx = half_warp_sum_d(bfx, smask);
                bfy = half_warp_sum_d(bfy, smask);
                bfz = half_warp_sum_d(bfz, smask);
                if (slane == 0) {
                    s_sf[sid][0] += bfx;
                    s_sf[sid][1] += bfy;
                    s_sf[sid][2] += bfz;
                }
                __syncwarp(smask);
            }
        };

        int qcnt = 0;
        for (int es = eb; es < ee || qcnt > 0; ) {
            bool do_flush = false;
            if (es < ee) {
                const int e_cur = es + slane;
                int i_cur = 0;
                bool match = false;
                if (e_cur < ee) {
                    i_cur = __ldg(senders + e_cur);
                    match = (__ldg(species + i_cur) == zi);
                }
                const unsigned bits = (__ballot_sync(smask, match) >> shift) & 0xffffu;
                const int add = __popc(bits);
                if (add > 0 && qcnt > 0 && qcnt + add > 16) {
                    do_flush = true;
                } else {
                    if (add > 0) {
                        if (match) {
                            const int rank = __popc(bits & ((1u << slane) - 1u));
                            s_eq[sid][qcnt + rank] = e_cur;
                            s_iq[sid][qcnt + rank] = i_cur;
                        }
                        qcnt += add;
                        __syncwarp(smask);
                    }
                    es += 16;
                    if (es >= ee && qcnt > 0) do_flush = true;
                }
            } else {
                do_flush = true;
            }
            if (do_flush) {
                flush_q(qcnt);
                qcnt = 0;
            }
        }
    }

    if (active_q) {
        dI_node.p[0][(int64_t)j * n_msg + slane] = acc_di0;
        float* dI1 = dI_node.p[1] + (int64_t)j * 3 * n_msg + slane;
        dI1[0]         = acc_di1_0;
        dI1[n_msg]     = acc_di1_1;
        dI1[2 * n_msg] = acc_di1_2;
    }
    if constexpr (REM_BF) {
        const float df0_tot = half_warp_sum_f(acc_df0, smask);
        if (slane == 0) df0_bf[j] = df0_tot;
    }

    if constexpr (STRESS) {
        v0 = half_warp_sum_d(v0, smask); v1 = half_warp_sum_d(v1, smask); v2 = half_warp_sum_d(v2, smask);
        v3 = half_warp_sum_d(v3, smask); v4 = half_warp_sum_d(v4, smask); v5 = half_warp_sum_d(v5, smask);
    }
    if (slane == 0) {
        const double sfx = s_sf[sid][0], sfy = s_sf[sid][1], sfz = s_sf[sid][2];
        if (sfx != 0.0 || sfy != 0.0 || sfz != 0.0) {
            atomicAdd(forces + (int64_t)j * 3 + 0, sfx);
            atomicAdd(forces + (int64_t)j * 3 + 1, sfy);
            atomicAdd(forces + (int64_t)j * 3 + 2, sfz);
        }
        if constexpr (STRESS) {
            if (v0 != 0.0) atomicAdd(virial + 0, v0);
            if (v1 != 0.0) atomicAdd(virial + 1, v1);
            if (v2 != 0.0) atomicAdd(virial + 2, v2);
            if (v3 != 0.0) atomicAdd(virial + 3, v3);
            if (v4 != 0.0) atomicAdd(virial + 4, v4);
            if (v5 != 0.0) atomicAdd(virial + 5, v5);
        }
    }
}

// =================================================================================================
// Kernel 7s: Register-Cached Species-Grouped Layer-2 Edge Backward with float4 SMEM (n_ghost > 0)
// =================================================================================================
template <bool REM_BF, bool STRESS>
__global__ void __launch_bounds__(128) ilmlip_l2_edge_bwd_sp_kernel(
    int N, int S, int n_msg,
    Int4 l2_c, float r_max,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2, const float* __restrict__ Cbf,
    const float* __restrict__ f0, const float4* __restrict__ I_pack4, CPtr4 I_node, CPtr4 dA2_raw,
    Ptr4 dI_node, float* __restrict__ df0_bf, double* __restrict__ forces, double* __restrict__ virial) {
    __shared__ float4 s_g4[8][8][2];
    __shared__ float4 s_dg4[8][8][2];
    __shared__ float4 s_Y[8][8];
    __shared__ float4 s_vl[8][8];
    __shared__ float s_f0[8][8];
    __shared__ float s_er_dY[8][8][4];
    __shared__ int s_eq[8][16];
    __shared__ int s_jq[8][16];

    const int sid = threadIdx.x >> 4;
    const int sub = sid & 1;
    const int slane = threadIdx.x & 15;
    const unsigned smask = sub ? 0xffff0000u : 0x0000ffffu;
    const int shift = sub ? 16 : 0;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    const int l2_tot_cols = 2 * n_msg;
    const bool active_q = (slane < n_msg);

    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f, g4 = 0.0f, g5 = 0.0f, g6 = 0.0f;
    float g7c = 0.0f, w00 = 0.0f, w01 = 0.0f, w02 = 0.0f, w11 = 0.0f, w12 = 0.0f, w22 = 0.0f;
    if (active_q) {
        const int c0 = l2_c.v[0], c1 = l2_c.v[1], c2 = l2_c.v[2];
        constexpr float C110   = 0.57735026918962576451f;
        constexpr float C_SQ2  = 0.70710678118654752440f;
        constexpr float C_SQ6  = 0.40824829046386301637f;
        constexpr float C_2SQ6 = 0.81649658092772603273f;
        g0 = dA2_raw.p[0][(int64_t)i * c0 + slane];
        g7c = dA2_raw.p[0][(int64_t)i * c0 + n_msg + slane] * C110;
        const float* src1 = dA2_raw.p[1] + (int64_t)i * 3 * c1 + slane;
        g1 = src1[0];     g2 = src1[c1];         g3 = src1[2 * c1];
        g4 = src1[n_msg]; g5 = src1[c1 + n_msg]; g6 = src1[2 * c1 + n_msg];
        const float* src2 = dA2_raw.p[2] + (int64_t)i * 5 * c2 + slane;
        const float g8 = src2[0], g9 = src2[c2], g10 = src2[2 * c2], g11 = src2[3 * c2], g12 = src2[4 * c2];
        w00 = -C_SQ6 * g10 - C_SQ2 * g12;
        w01 =  C_SQ2 * g9;
        w02 =  C_SQ2 * g8;
        w11 =  C_2SQ6 * g10;
        w12 =  C_SQ2 * g11;
        w22 = -C_SQ6 * g10 + C_SQ2 * g12;
    }

    double rfx = 0.0, rfy = 0.0, rfz = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int zj = 0; zj < S; ++zj) {
        const int pair = z_i * S + zj;
        const float* C2p = C2 + (int64_t)pair * 8 * l2_tot_cols + slane;
        const float* Cbfp = REM_BF ? (Cbf + (int64_t)pair * 8 * l2_tot_cols + slane) : nullptr;
        float c2_0[8], c2_1[8], cb_0[8], cb_1[8];
        if (active_q) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                c2_0[k] = __ldg(C2p + k * l2_tot_cols);
                c2_1[k] = __ldg(C2p + k * l2_tot_cols + n_msg);
                if constexpr (REM_BF) {
                    cb_0[k] = __ldg(Cbfp + k * l2_tot_cols);
                    cb_1[k] = __ldg(Cbfp + k * l2_tot_cols + n_msg);
                }
            }
        }

        auto flush_q = [&](int nq) {
            for (int qb = 0; qb < nq; qb += 8) {
                const int bcnt = (nq - qb < 8) ? (nq - qb) : 8;
                if (slane < bcnt) {
                    const int e = s_eq[sid][qb + slane];
                    const float4 vl = __ldg(vec_len + e);
                    const float r = fmaxf(vl.w, 1e-8f);
                    const float sc = 0x1.bb67ae0000000p+0f / r;
                    s_vl[sid][slane] = vl;
                    s_Y[sid][slane] = make_float4(1.0f, vl.x * sc, vl.y * sc, vl.z * sc);
                    if constexpr (REM_BF) s_f0[sid][slane] = __ldg(f0 + s_jq[sid][qb + slane]);
                    float4 g03, g47, dg03, dg47;
                    eval_cheb8_f4<true>(r, r_max, g03, g47, dg03, dg47);
                    s_g4[sid][slane][0] = g03;   s_g4[sid][slane][1] = g47;
                    s_dg4[sid][slane][0] = dg03; s_dg4[sid][slane][1] = dg47;
                }
                __syncwarp(smask);

                for (int bi = 0; bi < bcnt; ++bi) {
                    const int j = s_jq[sid][qb + bi];
                    float er_lane = 0.0f, df0_lane = 0.0f;
                    float dY1 = 0.0f, dY2 = 0.0f, dY3 = 0.0f;

                    if (active_q) {
                        const float4 g03  = s_g4[sid][bi][0];
                        const float4 g47  = s_g4[sid][bi][1];
                        const float4 dg03 = s_dg4[sid][bi][0];
                        const float4 dg47 = s_dg4[sid][bi][1];
                        const float4 Y    = s_Y[sid][bi];
                        const float f0_j  = REM_BF ? s_f0[sid][bi] : 0.0f;

                        const float r2_0  = fmaf(c2_0[7], g47.w,  fmaf(c2_0[6], g47.z,  fmaf(c2_0[5], g47.y,  fmaf(c2_0[4], g47.x,
                                            fmaf(c2_0[3], g03.w,  fmaf(c2_0[2], g03.z,  fmaf(c2_0[1], g03.y,  c2_0[0] * g03.x)))))));
                        const float dr2_0 = fmaf(c2_0[7], dg47.w, fmaf(c2_0[6], dg47.z, fmaf(c2_0[5], dg47.y, fmaf(c2_0[4], dg47.x,
                                            fmaf(c2_0[3], dg03.w, fmaf(c2_0[2], dg03.z, fmaf(c2_0[1], dg03.y, c2_0[0] * dg03.x)))))));
                        const float r2_1  = fmaf(c2_1[7], g47.w,  fmaf(c2_1[6], g47.z,  fmaf(c2_1[5], g47.y,  fmaf(c2_1[4], g47.x,
                                            fmaf(c2_1[3], g03.w,  fmaf(c2_1[2], g03.z,  fmaf(c2_1[1], g03.y,  c2_1[0] * g03.x)))))));
                        const float dr2_1 = fmaf(c2_1[7], dg47.w, fmaf(c2_1[6], dg47.z, fmaf(c2_1[5], dg47.y, fmaf(c2_1[4], dg47.x,
                                            fmaf(c2_1[3], dg03.w, fmaf(c2_1[2], dg03.z, fmaf(c2_1[1], dg03.y, c2_1[0] * dg03.x)))))));

                        float rbf0 = 0.0f, drbf0 = 0.0f, rbf1 = 0.0f, drbf1 = 0.0f;
                        if constexpr (REM_BF) {
                            rbf0  = fmaf(cb_0[7], g47.w,  fmaf(cb_0[6], g47.z,  fmaf(cb_0[5], g47.y,  fmaf(cb_0[4], g47.x,
                                    fmaf(cb_0[3], g03.w,  fmaf(cb_0[2], g03.z,  fmaf(cb_0[1], g03.y,  cb_0[0] * g03.x)))))));
                            drbf0 = fmaf(cb_0[7], dg47.w, fmaf(cb_0[6], dg47.z, fmaf(cb_0[5], dg47.y, fmaf(cb_0[4], dg47.x,
                                    fmaf(cb_0[3], dg03.w, fmaf(cb_0[2], dg03.z, fmaf(cb_0[1], dg03.y, cb_0[0] * dg03.x)))))));
                            rbf1  = fmaf(cb_1[7], g47.w,  fmaf(cb_1[6], g47.z,  fmaf(cb_1[5], g47.y,  fmaf(cb_1[4], g47.x,
                                    fmaf(cb_1[3], g03.w,  fmaf(cb_1[2], g03.z,  fmaf(cb_1[1], g03.y,  cb_1[0] * g03.x)))))));
                            drbf1 = fmaf(cb_1[7], dg47.w, fmaf(cb_1[6], dg47.z, fmaf(cb_1[5], dg47.y, fmaf(cb_1[4], dg47.x,
                                    fmaf(cb_1[3], dg03.w, fmaf(cb_1[2], dg03.z, fmaf(cb_1[1], dg03.y, cb_1[0] * dg03.x)))))));
                        }

                        float4 ir;
                        if (I_pack4) {
                            ir = __ldg(I_pack4 + (int64_t)j * n_msg + slane);
                        } else {
                            const float* pI1 = I_node.p[1] + (int64_t)j * 3 * n_msg + slane;
                            ir = make_float4(__ldg(I_node.p[0] + (int64_t)j * n_msg + slane),
                                             __ldg(pI1), __ldg(pI1 + n_msg), __ldg(pI1 + 2 * n_msg));
                        }
                        const float i0   = ir.x - (REM_BF ? (f0_j * rbf0) : 0.0f);
                        const float i1_0 = ir.y - (REM_BF ? (f0_j * rbf1) * Y.y : 0.0f);
                        const float i1_1 = ir.z - (REM_BF ? (f0_j * rbf1) * Y.z : 0.0f);
                        const float i1_2 = ir.w - (REM_BF ? (f0_j * rbf1) * Y.w : 0.0f);

                        const float d_r2_0 = fmaf(g0, i0, g1 * i1_0 + g2 * i1_1 + g3 * i1_2);
                        float di0 = g0 * r2_0;
                        float di1_0 = r2_0 * g1, di1_1 = r2_0 * g2, di1_2 = r2_0 * g3;

                        const float gy = g4 * Y.y + g5 * Y.z + g6 * Y.w;
                        float d_r2_1 = gy * i0;
                        const float r2i0 = r2_1 * i0;
                        dY1 = g4 * r2i0; dY2 = g5 * r2i0; dY3 = g6 * r2i0;
                        di0 = fmaf(r2_1, gy, di0);

                        d_r2_1 = fmaf(g7c, Y.y * i1_0 + Y.z * i1_1 + Y.w * i1_2, d_r2_1);
                        const float gr = g7c * r2_1;
                        dY1 = fmaf(gr, i1_0, dY1); dY2 = fmaf(gr, i1_1, dY2); dY3 = fmaf(gr, i1_2, dY3);
                        di1_0 = fmaf(gr, Y.y, di1_0); di1_1 = fmaf(gr, Y.z, di1_1); di1_2 = fmaf(gr, Y.w, di1_2);

                        const float s0 = w00 * i1_0 + w01 * i1_1 + w02 * i1_2;
                        const float s1 = w01 * i1_0 + w11 * i1_1 + w12 * i1_2;
                        const float s2 = w02 * i1_0 + w12 * i1_1 + w22 * i1_2;
                        d_r2_1 += Y.y * s0 + Y.z * s1 + Y.w * s2;
                        dY1 = fmaf(r2_1, s0, dY1); dY2 = fmaf(r2_1, s1, dY2); dY3 = fmaf(r2_1, s2, dY3);
                        di1_0 = fmaf(r2_1, w00 * Y.y + w01 * Y.z + w02 * Y.w, di1_0);
                        di1_1 = fmaf(r2_1, w01 * Y.y + w11 * Y.z + w12 * Y.w, di1_1);
                        di1_2 = fmaf(r2_1, w02 * Y.y + w12 * Y.z + w22 * Y.w, di1_2);

                        er_lane = fmaf(d_r2_0, dr2_0, d_r2_1 * dr2_1);

                        atomicAdd(dI_node.p[0] + (int64_t)j * n_msg + slane, di0);
                        float* dI1 = dI_node.p[1] + (int64_t)j * 3 * n_msg + slane;
                        atomicAdd(dI1, di1_0);
                        atomicAdd(dI1 + n_msg, di1_1);
                        atomicAdd(dI1 + 2 * n_msg, di1_2);

                        if constexpr (REM_BF) {
                            df0_lane = -di0 * rbf0;
                            er_lane = fmaf(-di0 * f0_j, drbf0, er_lane);
                            const float ydot = di1_0 * Y.y + di1_1 * Y.z + di1_2 * Y.w;
                            df0_lane = fmaf(-ydot, rbf1, df0_lane);
                            er_lane = fmaf(-ydot * f0_j, drbf1, er_lane);
                            const float bf1_sc = -f0_j * rbf1;
                            dY1 = fmaf(di1_0, bf1_sc, dY1);
                            dY2 = fmaf(di1_1, bf1_sc, dY2);
                            dY3 = fmaf(di1_2, bf1_sc, dY3);
                        }
                    }

                    const float r4 = REM_BF ? df0_lane : 0.0f;
                    const bool p0 = (slane & 1) != 0;
                    const float u0 = (p0 ? dY1 : er_lane) + __shfl_xor_sync(smask, p0 ? er_lane : dY1, 1);
                    const float u1 = (p0 ? dY3 : dY2)     + __shfl_xor_sync(smask, p0 ? dY2 : dY3, 1);
                    const float u2 = (p0 ? 0.0f : r4)     + __shfl_xor_sync(smask, p0 ? r4 : 0.0f, 1);
                    const bool p1 = (slane & 2) != 0;
                    const float w0 = (p1 ? u1 : u0)   + __shfl_xor_sync(smask, p1 ? u0 : u1, 2);
                    const float w1 = (p1 ? 0.0f : u2) + __shfl_xor_sync(smask, p1 ? u2 : 0.0f, 2);
                    const bool p2 = (slane & 4) != 0;
                    float val = (p2 ? w1 : w0) + __shfl_xor_sync(smask, p2 ? w0 : w1, 4);
                    val += __shfl_xor_sync(smask, val, 8);
                    if (slane < 4) s_er_dY[sid][bi][slane] = val;
                    else if (slane == 4 && REM_BF && val != 0.0f) atomicAdd(df0_bf + j, val);
                }
                __syncwarp(smask);

                if (slane < bcnt) {
                    const float4 vl = s_vl[sid][slane];
                    const int j = s_jq[sid][qb + slane];
                    const float r = fmaxf(vl.w, 1e-8f);
                    const float inv_d = 1.0f / r;
                    const float ux = vl.x * inv_d, uy = vl.y * inv_d, uz = vl.z * inv_d;
                    const float4 ed = *reinterpret_cast<const float4*>(s_er_dY[sid][slane]);
                    float gx, gy, gz;
                    sh_grad_l1(ed.y, ed.z, ed.w, gx, gy, gz);
                    const float dot = ux * gx + uy * gy + uz * gz;
                    const double fx = (double)(ux * ed.x + (gx - ux * dot) * inv_d);
                    const double fy = (double)(uy * ed.x + (gy - uy * dot) * inv_d);
                    const double fz = (double)(uz * ed.x + (gz - uz * dot) * inv_d);
                    rfx -= fx; rfy -= fy; rfz -= fz;
                    atomicAdd(forces + (int64_t)j * 3 + 0, fx);
                    atomicAdd(forces + (int64_t)j * 3 + 1, fy);
                    atomicAdd(forces + (int64_t)j * 3 + 2, fz);
                    if constexpr (STRESS) {
                        v0 += (double)vl.x * fx;
                        v1 += 0.5 * ((double)vl.x * fy + (double)vl.y * fx);
                        v2 += 0.5 * ((double)vl.x * fz + (double)vl.z * fx);
                        v3 += (double)vl.y * fy;
                        v4 += 0.5 * ((double)vl.y * fz + (double)vl.z * fy);
                        v5 += (double)vl.z * fz;
                    }
                }
                __syncwarp(smask);
            }
        };

        int qcnt = 0;
        for (int es = eb; es < ee || qcnt > 0; ) {
            bool do_flush = false;
            if (es < ee) {
                const int e_cur = es + slane;
                int j_cur = 0;
                bool match = false;
                if (e_cur < ee) {
                    j_cur = __ldg(senders + e_cur);
                    match = (__ldg(species + j_cur) == zj);
                }
                const unsigned bits = (__ballot_sync(smask, match) >> shift) & 0xffffu;
                const int add = __popc(bits);
                if (add > 0 && qcnt > 0 && qcnt + add > 16) {
                    do_flush = true;
                } else {
                    if (add > 0) {
                        if (match) {
                            const int rank = __popc(bits & ((1u << slane) - 1u));
                            s_eq[sid][qcnt + rank] = e_cur;
                            s_jq[sid][qcnt + rank] = j_cur;
                        }
                        qcnt += add;
                        __syncwarp(smask);
                    }
                    es += 16;
                    if (es >= ee && qcnt > 0) do_flush = true;
                }
            } else {
                do_flush = true;
            }
            if (do_flush) {
                flush_q(qcnt);
                qcnt = 0;
            }
        }
    }

    rfx = half_warp_sum_d(rfx, smask);
    rfy = half_warp_sum_d(rfy, smask);
    rfz = half_warp_sum_d(rfz, smask);
    if constexpr (STRESS) {
        v0 = half_warp_sum_d(v0, smask); v1 = half_warp_sum_d(v1, smask); v2 = half_warp_sum_d(v2, smask);
        v3 = half_warp_sum_d(v3, smask); v4 = half_warp_sum_d(v4, smask); v5 = half_warp_sum_d(v5, smask);
    }
    if (slane == 0) {
        if (rfx != 0.0 || rfy != 0.0 || rfz != 0.0) {
            atomicAdd(forces + (int64_t)i * 3 + 0, rfx);
            atomicAdd(forces + (int64_t)i * 3 + 1, rfy);
            atomicAdd(forces + (int64_t)i * 3 + 2, rfz);
        }
        if constexpr (STRESS) {
            if (v0 != 0.0) atomicAdd(virial + 0, v0);
            if (v1 != 0.0) atomicAdd(virial + 1, v1);
            if (v2 != 0.0) atomicAdd(virial + 2, v2);
            if (v3 != 0.0) atomicAdd(virial + 3, v3);
            if (v4 != 0.0) atomicAdd(virial + 4, v4);
            if (v5 != 0.0) atomicAdd(virial + 5, v5);
        }
    }
}

// =================================================================================================
// Kernel 7a: Fast Layer-2 Edge Backward (half-warp for n_msg<=16, smem C2+Cbf, butterfly reduce, no dY[0])
// =================================================================================================
template <int LFEAT2, int SUBWARP, int MSG_PASSES, int NB_MAX, bool REM_BF, bool STRESS, bool SMEM_C>
__global__ void __launch_bounds__(128) ilmlip_l2_edge_bwd_fast_kernel(
    int N, int S, int n_base, int n_msg,
    Int4 l2_c, float r_max,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2, const float* __restrict__ Cbf,
    const float* __restrict__ f0, CPtr4 I_node, CPtr4 dA2_raw,
    Ptr4 dI_node, float* __restrict__ df0_bf, double* __restrict__ forces, double* __restrict__ virial) {
    constexpr int NSUB = 128 / SUBWARP;
    constexpr int EBATCH = (SUBWARP == 16) ? 4 : 8;
    constexpr int N_SH = (LFEAT2 == 2) ? 9 : 4;
    constexpr int N_OUT = (LFEAT2 == 2) ? 21 : 13;

    extern __shared__ float smem_c[];
    __shared__ float s_g[NSUB][EBATCH][NB_MAX];
    __shared__ float s_dg[NSUB][EBATCH][NB_MAX];
    __shared__ float s_Y[NSUB][EBATCH][N_SH];
    __shared__ float4 s_vl[NSUB][EBATCH];
    __shared__ int s_j[NSUB][EBATCH];
    __shared__ int s_pair[NSUB][EBATCH];
    __shared__ float s_f0[NSUB][EBATCH];
    __shared__ float s_er[NSUB][EBATCH];
    __shared__ float s_dY[NSUB][EBATCH][N_SH];

    const int l2_tot_cols = (LFEAT2 + 1) * n_msg;
    const int bf_tot_cols = 2 * n_msg;
    const int c2_elems = S * S * n_base * l2_tot_cols;
    const int cbf_elems = REM_BF ? (S * S * n_base * bf_tot_cols) : 0;
    if constexpr (SMEM_C) {
        for (int idx = threadIdx.x; idx < c2_elems; idx += 128) smem_c[idx] = __ldg(C2 + idx);
        if constexpr (REM_BF) {
            float* smem_bf = smem_c + c2_elems;
            for (int idx = threadIdx.x; idx < cbf_elems; idx += 128) smem_bf[idx] = __ldg(Cbf + idx);
        }
        __syncthreads();
    }

    const int sid = threadIdx.x / SUBWARP;
    const int slane = threadIdx.x & (SUBWARP - 1);
    const unsigned smask = (SUBWARP == 16) ? ((sid & 1) ? 0xffff0000u : 0x0000ffffu) : 0xffffffffu;
    const int i = blockIdx.x * NSUB + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    const float* C2_base = SMEM_C ? smem_c : C2;
    const float* Cbf_base = REM_BF ? (SMEM_C ? (smem_c + c2_elems) : Cbf) : nullptr;

    float gA2[MSG_PASSES][N_OUT];
    #pragma unroll
    for (int s = 0; s < MSG_PASSES; ++s) {
        const int q = slane + s * SUBWARP;
        if (q < n_msg) {
            const int c0 = l2_c.v[0], c1 = l2_c.v[1], c2 = l2_c.v[2];
            gA2[s][0] = dA2_raw.p[0][(int64_t)i * c0 + q];
            gA2[s][7] = dA2_raw.p[0][(int64_t)i * c0 + n_msg + q];
            const float* src1 = dA2_raw.p[1] + (int64_t)i * 3 * c1 + q;
            gA2[s][1] = src1[0];     gA2[s][2] = src1[c1];         gA2[s][3] = src1[2 * c1];
            gA2[s][4] = src1[n_msg]; gA2[s][5] = src1[c1 + n_msg]; gA2[s][6] = src1[2 * c1 + n_msg];
            if constexpr (LFEAT2 == 2) {
                gA2[s][18] = src1[2 * n_msg]; gA2[s][19] = src1[c1 + 2 * n_msg]; gA2[s][20] = src1[2 * c1 + 2 * n_msg];
            }
            const float* src2 = dA2_raw.p[2] + (int64_t)i * 5 * c2 + q;
            #pragma unroll
            for (int m = 0; m < 5; ++m) gA2[s][8 + m] = src2[m * c2];
            if constexpr (LFEAT2 == 2) {
                #pragma unroll
                for (int m = 0; m < 5; ++m) gA2[s][13 + m] = src2[m * c2 + n_msg];
            }
        } else {
            #pragma unroll
            for (int k = 0; k < N_OUT; ++k) gA2[s][k] = 0.0f;
        }
    }

    double rfx = 0.0, rfy = 0.0, rfz = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += EBATCH) {
        const int bcnt = (ee - eb_b < EBATCH) ? (ee - eb_b) : EBATCH;
        if (slane < bcnt) {
            const int e = eb_b + slane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const int pair = z_i * S + __ldg(species + j);
            const float r = fmaxf(vl.w, 1e-8f);
            s_vl[sid][slane] = vl;
            s_j[sid][slane] = j;
            s_pair[sid][slane] = pair;
            if (REM_BF) s_f0[sid][slane] = __ldg(f0 + j);
            eval_cheb<true>(r, r_max, n_base, s_g[sid][slane], s_dg[sid][slane]);
        } else if (slane >= EBATCH && slane < EBATCH + bcnt) {
            const int bi = slane - EBATCH;
            const int e = eb_b + bi;
            const float4 vl = __ldg(vec_len + e);
            const float r = fmaxf(vl.w, 1e-8f);
            float4 Yv[4];
            loup::edge_sh16(vl.x, vl.y, vl.z, r, Yv);
            unpack_sh16(Yv, LFEAT2, s_Y[sid][bi]);
        }
        __syncwarp(smask);

        for (int bi = 0; bi < bcnt; ++bi) {
            const int j = s_j[sid][bi];
            const int pair = s_pair[sid][bi];
            const float f0_j = REM_BF ? s_f0[sid][bi] : 0.0f;
            const float* gk = s_g[sid][bi];
            const float* dgk = s_dg[sid][bi];
            const float* Yk = s_Y[sid][bi];
            const float* C2p = C2_base + (int64_t)pair * n_base * l2_tot_cols;
            const float* Cbfp = REM_BF ? (Cbf_base + (int64_t)pair * n_base * bf_tot_cols) : nullptr;

            float er_lane = 0.0f;
            float df0_lane = 0.0f;
            float dY_lane[N_SH];
            #pragma unroll
            for (int m = 0; m < N_SH; ++m) dY_lane[m] = 0.0f;

            #pragma unroll
            for (int s = 0; s < MSG_PASSES; ++s) {
                const int q = slane + s * SUBWARP;
                if (q >= n_msg) continue;

                float r2_0 = 0.0f, r2_1 = 0.0f, r2_2 = 0.0f;
                float dr2_0 = 0.0f, dr2_1 = 0.0f, dr2_2 = 0.0f;
                auto accum_c2 = [&](int k) {
                    const float gval = gk[k], dgval = dgk[k];
                    const float* row2 = C2p + k * l2_tot_cols + q;
                    const float c0v = SMEM_C ? row2[0] : __ldg(row2);
                    const float c1v = SMEM_C ? row2[n_msg] : __ldg(row2 + n_msg);
                    r2_0 = fmaf(c0v, gval, r2_0); dr2_0 = fmaf(c0v, dgval, dr2_0);
                    r2_1 = fmaf(c1v, gval, r2_1); dr2_1 = fmaf(c1v, dgval, dr2_1);
                    if constexpr (LFEAT2 == 2) {
                        const float c2v = SMEM_C ? row2[2 * n_msg] : __ldg(row2 + 2 * n_msg);
                        r2_2 = fmaf(c2v, gval, r2_2); dr2_2 = fmaf(c2v, dgval, dr2_2);
                    }
                };
                if (n_base == 8) {
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) accum_c2(k);
                } else if (n_base == 12) {
                    #pragma unroll
                    for (int k = 0; k < 12; ++k) accum_c2(k);
                } else if (n_base == 16) {
                    #pragma unroll
                    for (int k = 0; k < 16; ++k) accum_c2(k);
                } else {
                    for (int k = 0; k < n_base; ++k) accum_c2(k);
                }

                float rbf0 = 0.0f, rbf1 = 0.0f, drbf0 = 0.0f, drbf1 = 0.0f;
                if constexpr (REM_BF) {
                    auto accum_bf = [&](int k) {
                        const float gval = gk[k], dgval = dgk[k];
                        const float* rowb = Cbfp + k * bf_tot_cols + q;
                        const float cb0 = SMEM_C ? rowb[0] : __ldg(rowb);
                        const float cb1 = SMEM_C ? rowb[n_msg] : __ldg(rowb + n_msg);
                        rbf0 = fmaf(cb0, gval, rbf0); drbf0 = fmaf(cb0, dgval, drbf0);
                        rbf1 = fmaf(cb1, gval, rbf1); drbf1 = fmaf(cb1, dgval, drbf1);
                    };
                    if (n_base == 8) {
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) accum_bf(k);
                    } else if (n_base == 12) {
                        #pragma unroll
                        for (int k = 0; k < 12; ++k) accum_bf(k);
                    } else if (n_base == 16) {
                        #pragma unroll
                        for (int k = 0; k < 16; ++k) accum_bf(k);
                    } else {
                        for (int k = 0; k < n_base; ++k) accum_bf(k);
                    }
                }

                const float i0 = __ldg(I_node.p[0] + (int64_t)j * n_msg + q) - (REM_BF ? (f0_j * rbf0) * Yk[0] : 0.0f);
                const float* pI1 = I_node.p[1] + (int64_t)j * 3 * n_msg + q;
                const float i1_0 = __ldg(pI1)             - (REM_BF ? (f0_j * rbf1) * Yk[1] : 0.0f);
                const float i1_1 = __ldg(pI1 + n_msg)     - (REM_BF ? (f0_j * rbf1) * Yk[2] : 0.0f);
                const float i1_2 = __ldg(pI1 + 2 * n_msg) - (REM_BF ? (f0_j * rbf1) * Yk[3] : 0.0f);

                float d_r2_0 = 0.0f, d_r2_1 = 0.0f, d_r2_2 = 0.0f;
                float di0 = 0.0f, di1_0 = 0.0f, di1_1 = 0.0f, di1_2 = 0.0f;

                // (0,0)->0 [reg 0] (note: Y[0] is constant -> dY[0] is unused by sh16_grad!)
                {
                    const float g = gA2[s][0];
                    d_r2_0 = fmaf(g, Yk[0] * i0, d_r2_0);
                    di0 = fmaf(g, r2_0 * Yk[0], di0);
                }
                // (0,1)->1 [reg 1..3]
                {
                    const float p0 = r2_0 * Yk[0];
                    const float g1 = gA2[s][1], g2 = gA2[s][2], g3 = gA2[s][3];
                    const float gi = g1 * i1_0 + g2 * i1_1 + g3 * i1_2;
                    d_r2_0 = fmaf(Yk[0], gi, d_r2_0);
                    di1_0 = fmaf(p0, g1, di1_0);
                    di1_1 = fmaf(p0, g2, di1_1);
                    di1_2 = fmaf(p0, g3, di1_2);
                }
                // (1,0)->1 [reg 4..6]
                {
                    const float g4 = gA2[s][4], g5 = gA2[s][5], g6 = gA2[s][6];
                    const float gy = g4 * Yk[1] + g5 * Yk[2] + g6 * Yk[3];
                    d_r2_1 = fmaf(gy, i0, d_r2_1);
                    const float r2i0 = r2_1 * i0;
                    dY_lane[1] = fmaf(g4, r2i0, dY_lane[1]);
                    dY_lane[2] = fmaf(g5, r2i0, dY_lane[2]);
                    dY_lane[3] = fmaf(g6, r2i0, dY_lane[3]);
                    di0 = fmaf(r2_1, gy, di0);
                }
                // (1,1)->0 [reg 7]
                {
                    constexpr float C110 = 0.57735026918962576451f;
                    const float g = gA2[s][7] * C110;
                    d_r2_1 += g * (Yk[1] * i1_0 + Yk[2] * i1_1 + Yk[3] * i1_2);
                    const float gr = g * r2_1;
                    dY_lane[1] = fmaf(gr, i1_0, dY_lane[1]);
                    dY_lane[2] = fmaf(gr, i1_1, dY_lane[2]);
                    dY_lane[3] = fmaf(gr, i1_2, dY_lane[3]);
                    di1_0 = fmaf(gr, Yk[1], di1_0);
                    di1_1 = fmaf(gr, Yk[2], di1_1);
                    di1_2 = fmaf(gr, Yk[3], di1_2);
                }
                // (1,1)->2 [reg 8..12]
                {
                    constexpr float C_SQ2 = 0.70710678118654752440f;
                    constexpr float C_SQ6 = 0.40824829046386301637f;
                    constexpr float C_2SQ6 = 0.81649658092772603273f;
                    const float g0 = gA2[s][8], g1 = gA2[s][9], g2 = gA2[s][10], g3 = gA2[s][11], g4 = gA2[s][12];
                    const float w00 = -C_SQ6 * g2 - C_SQ2 * g4;
                    const float w01 = C_SQ2 * g1;
                    const float w02 = C_SQ2 * g0;
                    const float w11 = C_2SQ6 * g2;
                    const float w12 = C_SQ2 * g3;
                    const float w22 = -C_SQ6 * g2 + C_SQ2 * g4;
                    const float s0 = w00 * i1_0 + w01 * i1_1 + w02 * i1_2;
                    const float s1 = w01 * i1_0 + w11 * i1_1 + w12 * i1_2;
                    const float s2 = w02 * i1_0 + w12 * i1_1 + w22 * i1_2;
                    d_r2_1 += Yk[1] * s0 + Yk[2] * s1 + Yk[3] * s2;
                    dY_lane[1] = fmaf(r2_1, s0, dY_lane[1]);
                    dY_lane[2] = fmaf(r2_1, s1, dY_lane[2]);
                    dY_lane[3] = fmaf(r2_1, s2, dY_lane[3]);
                    di1_0 += r2_1 * (w00 * Yk[1] + w01 * Yk[2] + w02 * Yk[3]);
                    di1_1 += r2_1 * (w01 * Yk[1] + w11 * Yk[2] + w12 * Yk[3]);
                    di1_2 += r2_1 * (w02 * Yk[1] + w12 * Yk[2] + w22 * Yk[3]);
                }

                if constexpr (LFEAT2 == 2) {
                    // (2,0)->2 [reg 13..17]
                    {
                        const float gy = gA2[s][13] * Yk[4] + gA2[s][14] * Yk[5] + gA2[s][15] * Yk[6] +
                                         gA2[s][16] * Yk[7] + gA2[s][17] * Yk[8];
                        d_r2_2 = fmaf(gy, i0, d_r2_2);
                        const float r2i0 = r2_2 * i0;
                        dY_lane[4] = fmaf(gA2[s][13], r2i0, dY_lane[4]);
                        dY_lane[5] = fmaf(gA2[s][14], r2i0, dY_lane[5]);
                        dY_lane[6] = fmaf(gA2[s][15], r2i0, dY_lane[6]);
                        dY_lane[7] = fmaf(gA2[s][16], r2i0, dY_lane[7]);
                        dY_lane[8] = fmaf(gA2[s][17], r2i0, dY_lane[8]);
                        di0 = fmaf(r2_2, gy, di0);
                    }
                    // (2,1)->1 [reg 18..20]
                    {
                        constexpr float C_S310 = 0.54772255750516611346f;
                        constexpr float C_S110 = 0.31622776601683793320f;
                        constexpr float C_S410 = 0.63245553203367586640f;
                        const float g0 = gA2[s][18], g1 = gA2[s][19], g2 = gA2[s][20];
                        const float t0 = C_S310 * (g0 * i1_2 + g2 * i1_0);
                        const float t1 = C_S310 * (g0 * i1_1 + g1 * i1_0);
                        const float t2 = -C_S110 * g0 * i1_0 + C_S410 * g1 * i1_1 - C_S110 * g2 * i1_2;
                        const float t3 = C_S310 * (g1 * i1_2 + g2 * i1_1);
                        const float t4 = C_S310 * (-g0 * i1_0 + g2 * i1_2);
                        d_r2_2 += Yk[4] * t0 + Yk[5] * t1 + Yk[6] * t2 + Yk[7] * t3 + Yk[8] * t4;
                        dY_lane[4] = fmaf(r2_2, t0, dY_lane[4]);
                        dY_lane[5] = fmaf(r2_2, t1, dY_lane[5]);
                        dY_lane[6] = fmaf(r2_2, t2, dY_lane[6]);
                        dY_lane[7] = fmaf(r2_2, t3, dY_lane[7]);
                        dY_lane[8] = fmaf(r2_2, t4, dY_lane[8]);
                        di1_0 += r2_2 * (C_S310 * (g2 * Yk[4] + g1 * Yk[5] - g0 * Yk[8]) - C_S110 * g0 * Yk[6]);
                        di1_1 += r2_2 * (C_S310 * (g0 * Yk[5] + g2 * Yk[7]) + C_S410 * g1 * Yk[6]);
                        di1_2 += r2_2 * (C_S310 * (g0 * Yk[4] + g1 * Yk[7] + g2 * Yk[8]) - C_S110 * g2 * Yk[6]);
                    }
                }

                er_lane = fmaf(d_r2_0, dr2_0, er_lane);
                er_lane = fmaf(d_r2_1, dr2_1, er_lane);
                if constexpr (LFEAT2 == 2) er_lane = fmaf(d_r2_2, dr2_2, er_lane);

                atomicAdd(dI_node.p[0] + (int64_t)j * n_msg + q, di0);
                float* dI1 = dI_node.p[1] + (int64_t)j * 3 * n_msg + q;
                atomicAdd(dI1, di1_0);
                atomicAdd(dI1 + n_msg, di1_1);
                atomicAdd(dI1 + 2 * n_msg, di1_2);

                if constexpr (REM_BF) {
                    df0_lane = fmaf(-di0, rbf0 * Yk[0], df0_lane);
                    const float drbf0_v = -di0 * f0_j * Yk[0];
                    er_lane = fmaf(drbf0_v, drbf0, er_lane);

                    const float ydot = di1_0 * Yk[1] + di1_1 * Yk[2] + di1_2 * Yk[3];
                    df0_lane = fmaf(-ydot, rbf1, df0_lane);
                    er_lane = fmaf(-ydot * f0_j, drbf1, er_lane);
                    const float bf1_sc = -f0_j * rbf1;
                    dY_lane[1] = fmaf(di1_0, bf1_sc, dY_lane[1]);
                    dY_lane[2] = fmaf(di1_1, bf1_sc, dY_lane[2]);
                    dY_lane[3] = fmaf(di1_2, bf1_sc, dY_lane[3]);
                }
            }

            // Butterfly transpose-reduce: folds [er, dY[1..N_SH-1], df0] in 7-12 shuffles instead of 24-55!
            if constexpr (LFEAT2 == 1) {
                const float r0 = er_lane, r1 = dY_lane[1], r2 = dY_lane[2], r3 = dY_lane[3];
                const float r4 = REM_BF ? df0_lane : 0.0f;
                const bool p0 = (slane & 1) != 0;
                const float u0 = (p0 ? r1 : r0) + __shfl_xor_sync(smask, p0 ? r0 : r1, 1);
                const float u1 = (p0 ? r3 : r2) + __shfl_xor_sync(smask, p0 ? r2 : r3, 1);
                const float u2 = (p0 ? 0.0f : r4) + __shfl_xor_sync(smask, p0 ? r4 : 0.0f, 1);
                const bool p1 = (slane & 2) != 0;
                const float w0 = (p1 ? u1 : u0) + __shfl_xor_sync(smask, p1 ? u0 : u1, 2);
                const float w1 = (p1 ? 0.0f : u2) + __shfl_xor_sync(smask, p1 ? u2 : 0.0f, 2);
                const bool p2 = (slane & 4) != 0;
                float val = (p2 ? w1 : w0) + __shfl_xor_sync(smask, p2 ? w0 : w1, 4);
                val += __shfl_xor_sync(smask, val, 8);
                if constexpr (SUBWARP == 32) val += __shfl_xor_sync(smask, val, 16);
                if (slane == 0) s_er[sid][bi] = val;
                else if (slane <= 3) s_dY[sid][bi][slane] = val;
                else if (slane == 4 && REM_BF && val != 0.0f) atomicAdd(df0_bf + j, val);
            } else {
                const float r[10] = {er_lane, dY_lane[1], dY_lane[2], dY_lane[3], dY_lane[4],
                                     dY_lane[5], dY_lane[6], dY_lane[7], dY_lane[8], REM_BF ? df0_lane : 0.0f};
                const bool p0 = (slane & 1) != 0;
                float u[5];
                #pragma unroll
                for (int k = 0; k < 5; ++k) u[k] = (p0 ? r[2 * k + 1] : r[2 * k]) + __shfl_xor_sync(smask, p0 ? r[2 * k] : r[2 * k + 1], 1);
                const bool p1 = (slane & 2) != 0;
                float w[3];
                w[0] = (p1 ? u[1] : u[0]) + __shfl_xor_sync(smask, p1 ? u[0] : u[1], 2);
                w[1] = (p1 ? u[3] : u[2]) + __shfl_xor_sync(smask, p1 ? u[2] : u[3], 2);
                w[2] = (p1 ? 0.0f : u[4]) + __shfl_xor_sync(smask, p1 ? u[4] : 0.0f, 2);
                const bool p2 = (slane & 4) != 0;
                float z0 = (p2 ? w[1] : w[0]) + __shfl_xor_sync(smask, p2 ? w[0] : w[1], 4);
                float z1 = (p2 ? 0.0f : w[2]) + __shfl_xor_sync(smask, p2 ? w[2] : 0.0f, 4);
                const bool p3 = (slane & 8) != 0;
                float val = (p3 ? z1 : z0) + __shfl_xor_sync(smask, p3 ? z0 : z1, 8);
                if constexpr (SUBWARP == 32) val += __shfl_xor_sync(smask, val, 16);
                if (slane == 0) s_er[sid][bi] = val;
                else if (slane <= 8) s_dY[sid][bi][slane] = val;
                else if (slane == 9 && REM_BF && val != 0.0f) atomicAdd(df0_bf + j, val);
            }
        }
        __syncwarp(smask);

        if (slane < bcnt) {
            const float4 vl = s_vl[sid][slane];
            const int j = s_j[sid][slane];
            const float r = fmaxf(vl.w, 1e-8f);
            const float inv_d = 1.0f / r;
            const float ux = vl.x * inv_d, uy = vl.y * inv_d, uz = vl.z * inv_d;
            float dY16[16];
            dY16[0] = 0.0f;
            #pragma unroll
            for (int m = 1; m < 16; ++m) dY16[m] = (m < N_SH) ? s_dY[sid][slane][m] : 0.0f;
            float gx, gy, gz;
            ::sh16_grad(ux, uy, uz, dY16, gx, gy, gz);
            const float dot = ux * gx + uy * gy + uz * gz;
            const float er = s_er[sid][slane];
            const double fx = (double)(ux * er + (gx - ux * dot) * inv_d);
            const double fy = (double)(uy * er + (gy - uy * dot) * inv_d);
            const double fz = (double)(uz * er + (gz - uz * dot) * inv_d);
            rfx -= fx; rfy -= fy; rfz -= fz;
            atomicAdd(forces + (int64_t)j * 3 + 0, fx);
            atomicAdd(forces + (int64_t)j * 3 + 1, fy);
            atomicAdd(forces + (int64_t)j * 3 + 2, fz);
            if (STRESS) {
                v0 += (double)vl.x * fx;
                v1 += 0.5 * ((double)vl.x * fy + (double)vl.y * fx);
                v2 += 0.5 * ((double)vl.x * fz + (double)vl.z * fx);
                v3 += (double)vl.y * fy;
                v4 += 0.5 * ((double)vl.y * fz + (double)vl.z * fy);
                v5 += (double)vl.z * fz;
            }
        }
        __syncwarp(smask);
    }

    rfx = (SUBWARP == 16) ? half_warp_sum_d(rfx, smask) : warp_sum_d(rfx);
    rfy = (SUBWARP == 16) ? half_warp_sum_d(rfy, smask) : warp_sum_d(rfy);
    rfz = (SUBWARP == 16) ? half_warp_sum_d(rfz, smask) : warp_sum_d(rfz);
    if (STRESS) {
        v0 = (SUBWARP == 16) ? half_warp_sum_d(v0, smask) : warp_sum_d(v0);
        v1 = (SUBWARP == 16) ? half_warp_sum_d(v1, smask) : warp_sum_d(v1);
        v2 = (SUBWARP == 16) ? half_warp_sum_d(v2, smask) : warp_sum_d(v2);
        v3 = (SUBWARP == 16) ? half_warp_sum_d(v3, smask) : warp_sum_d(v3);
        v4 = (SUBWARP == 16) ? half_warp_sum_d(v4, smask) : warp_sum_d(v4);
        v5 = (SUBWARP == 16) ? half_warp_sum_d(v5, smask) : warp_sum_d(v5);
    }
    if (slane == 0) {
        if (rfx != 0.0 || rfy != 0.0 || rfz != 0.0) {
            atomicAdd(forces + (int64_t)i * 3 + 0, rfx);
            atomicAdd(forces + (int64_t)i * 3 + 1, rfy);
            atomicAdd(forces + (int64_t)i * 3 + 2, rfz);
        }
        if (STRESS) {
            if (v0 != 0.0) atomicAdd(virial + 0, v0);
            if (v1 != 0.0) atomicAdd(virial + 1, v1);
            if (v2 != 0.0) atomicAdd(virial + 2, v2);
            if (v3 != 0.0) atomicAdd(virial + 3, v3);
            if (v4 != 0.0) atomicAdd(virial + 4, v4);
            if (v5 != 0.0) atomicAdd(virial + 5, v5);
        }
    }
}

// =================================================================================================
// Kernel 7: Layer-2 Edge Backward (generic fallback for lmsg!=1 or lout2!=2 or lfeat2>2)
// =================================================================================================
template <int MSG_PASSES, int MAX_L2_OUT, bool REM_BF, bool STRESS>
__global__ void __launch_bounds__(128) ilmlip_l2_edge_bwd_kernel(
    int N, int S, int lfeat2, int lmsg, int lout2, int n_base, int n_msg, int n_l2_paths,
    Int4 l2_c, float inv_rmax,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2, const float* __restrict__ Cbf,
    const float* __restrict__ f0, CPtr4 I_node, CPtr4 dA2_raw,
    const int32_t* __restrict__ l2p, const int32_t* __restrict__ cg_idx, const float* __restrict__ cg_val,
    Ptr4 dI_node, float* __restrict__ df0_bf, double* __restrict__ forces, double* __restrict__ virial) {
    __shared__ float s_g[4][32];
    __shared__ float s_dg[4][32];
    __shared__ float s_Y[4][16];

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    const int l2_tot_cols = (lfeat2 + 1) * n_msg;
    const int bf_tot_cols = (lmsg + 1) * n_msg;
    const int sh_lmax = lfeat2 > lmsg ? lfeat2 : lmsg;
    const int n_sh = (sh_lmax + 1) * (sh_lmax + 1);

    // Stage receiver i's dA2_raw once in registers!
    float gA2[MSG_PASSES][MAX_L2_OUT];
    #pragma unroll
    for (int s = 0; s < MSG_PASSES; ++s) {
        const int q = lane + s * 32;
        if (q < n_msg) {
            for (int p = 0; p < n_l2_paths; ++p) {
                const int32_t* pr = l2p + p * 7;
                const int L = pr[2], p_in_L = pr[3], reg_off = pr[6];
                const int dm = 2 * L + 1;
                const int cL = l2_c.v[L];
                const float* src = dA2_raw.p[L] + (int64_t)i * dm * cL + p_in_L * n_msg + q;
                for (int m = 0; m < dm; ++m) gA2[s][reg_off + m] = src[m * cL];
            }
        }
    }

    double rfx = 0.0, rfy = 0.0, rfz = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int e = eb; e < ee; ++e) {
        const float4 vl = __ldg(vec_len + e);
        const int j = __ldg(senders + e);
        const int z_j = __ldg(species + j);
        const int pair = z_i * S + z_j;
        const float r = fmaxf(vl.w, 1e-8f);
        const float f0_j = REM_BF ? __ldg(f0 + j) : 0.0f;

        if (lane == 0) {
            eval_cheb<true>(r, inv_rmax, n_base, s_g[wid], s_dg[wid]);
            float4 Yv[4];
            loup::edge_sh16(vl.x, vl.y, vl.z, r, Yv);
            s_Y[wid][0] = Yv[0].x; s_Y[wid][1] = Yv[0].y; s_Y[wid][2] = Yv[0].z; s_Y[wid][3] = Yv[0].w;
            if (sh_lmax >= 2) {
                s_Y[wid][4] = Yv[1].x; s_Y[wid][5] = Yv[1].y; s_Y[wid][6] = Yv[1].z; s_Y[wid][7] = Yv[1].w;
                s_Y[wid][8] = Yv[2].x;
            }
        }
        __syncwarp();

        const float* C2p = C2 + (int64_t)pair * n_base * l2_tot_cols;
        const float* Cbfp = REM_BF ? (Cbf + (int64_t)pair * n_base * bf_tot_cols) : nullptr;

        float er_lane = 0.0f;
        float df0_lane = 0.0f;
        float dY_lane[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int s = 0; s < MSG_PASSES; ++s) {
            const int q = lane + s * 32;
            if (q >= n_msg) continue;

            float r2[3] = {0.0f, 0.0f, 0.0f};
            float dr2_dr[3] = {0.0f, 0.0f, 0.0f};
            for (int l1 = 0; l1 <= lfeat2; ++l1) {
                float rv = 0.0f, drv = 0.0f;
                const float* col = C2p + l1 * n_msg + q;
                for (int k = 0; k < n_base; ++k) {
                    const float cval = __ldg(col + k * l2_tot_cols);
                    rv = fmaf(cval, s_g[wid][k], rv);
                    drv = fmaf(cval, s_dg[wid][k], drv);
                }
                r2[l1] = rv;
                dr2_dr[l1] = drv;
            }

            float rbf[3] = {0.0f, 0.0f, 0.0f};
            float drbf_dr[3] = {0.0f, 0.0f, 0.0f};
            float ie[9];
            float die[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            for (int l2 = 0; l2 <= lmsg; ++l2) {
                if (REM_BF) {
                    float rv = 0.0f, drv = 0.0f;
                    const float* col = Cbfp + l2 * n_msg + q;
                    for (int k = 0; k < n_base; ++k) {
                        const float cval = __ldg(col + k * bf_tot_cols);
                        rv = fmaf(cval, s_g[wid][k], rv);
                        drv = fmaf(cval, s_dg[wid][k], drv);
                    }
                    rbf[l2] = rv;
                    drbf_dr[l2] = drv;
                }
                const int m0 = l2 * l2;
                const int dm = 2 * l2 + 1;
                const float* src = I_node.p[l2] + (int64_t)j * dm * n_msg + q;
                for (int m = 0; m < dm; ++m) {
                    float v = __ldg(src + m * n_msg);
                    if (REM_BF) v -= (f0_j * rbf[l2]) * s_Y[wid][m0 + m];
                    ie[m0 + m] = v;
                }
            }

            float d_r2[3] = {0.0f, 0.0f, 0.0f};
            for (int p = 0; p < n_l2_paths; ++p) {
                const int32_t* pr = l2p + p * 7;
                const int l1 = pr[0], l2 = pr[1], cg_off = pr[4], cg_cnt = pr[5], reg_off = pr[6];
                const int m01 = l1 * l1, m02 = l2 * l2;
                const float r2_l1 = r2[l1];
                for (int k = 0; k < cg_cnt; ++k) {
                    const int pk = __ldg(cg_idx + cg_off + k);
                    const int m1 = pk & 15, m2 = (pk >> 4) & 15, mL = (pk >> 8) & 15;
                    const float gv = __ldg(cg_val + cg_off + k) * gA2[s][reg_off + mL];
                    const float y1 = s_Y[wid][m01 + m1];
                    const float iev = ie[m02 + m2];
                    d_r2[l1] = fmaf(gv, y1 * iev, d_r2[l1]);
                    dY_lane[m01 + m1] = fmaf(gv, r2_l1 * iev, dY_lane[m01 + m1]);
                    die[m02 + m2] = fmaf(gv, r2_l1 * y1, die[m02 + m2]);
                }
            }

            for (int l1 = 0; l1 <= lfeat2; ++l1) er_lane = fmaf(d_r2[l1], dr2_dr[l1], er_lane);

            for (int l2 = 0; l2 <= lmsg; ++l2) {
                const int m0 = l2 * l2;
                const int dm = 2 * l2 + 1;
                float* dst = dI_node.p[l2] + (int64_t)j * dm * n_msg + q;
                float d_rbf = 0.0f;
                for (int m = 0; m < dm; ++m) {
                    const float gI = die[m0 + m];
                    atomicAdd(dst + m * n_msg, gI);
                    if (REM_BF) {
                        const float neg_g = -gI;
                        const float y2 = s_Y[wid][m0 + m];
                        df0_lane = fmaf(neg_g, rbf[l2] * y2, df0_lane);
                        d_rbf = fmaf(neg_g, f0_j * y2, d_rbf);
                        dY_lane[m0 + m] = fmaf(neg_g, f0_j * rbf[l2], dY_lane[m0 + m]);
                    }
                }
                if (REM_BF) er_lane = fmaf(d_rbf, drbf_dr[l2], er_lane);
            }
        }

        const float er = warp_sum(er_lane);
        const float df0_e = REM_BF ? warp_sum(df0_lane) : 0.0f;
        float dY16[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int m = 0; m < n_sh; ++m) dY16[m] = warp_sum(dY_lane[m]);

        if (lane == 0) {
            if (REM_BF && df0_e != 0.0f) atomicAdd(df0_bf + j, df0_e);
            const float inv_d = 1.0f / r;
            const float ux = vl.x * inv_d, uy = vl.y * inv_d, uz = vl.z * inv_d;
            float gx, gy, gz;
            ::sh16_grad(ux, uy, uz, dY16, gx, gy, gz);
            const float dot = ux * gx + uy * gy + uz * gz;
            const double fx = (double)(ux * er + (gx - ux * dot) * inv_d);
            const double fy = (double)(uy * er + (gy - uy * dot) * inv_d);
            const double fz = (double)(uz * er + (gz - uz * dot) * inv_d);
            rfx -= fx;
            rfy -= fy;
            rfz -= fz;
            atomicAdd(forces + (int64_t)j * 3 + 0, fx);
            atomicAdd(forces + (int64_t)j * 3 + 1, fy);
            atomicAdd(forces + (int64_t)j * 3 + 2, fz);
            if (STRESS) {
                v0 += (double)vl.x * fx;
                v1 += 0.5 * ((double)vl.x * fy + (double)vl.y * fx);
                v2 += 0.5 * ((double)vl.x * fz + (double)vl.z * fx);
                v3 += (double)vl.y * fy;
                v4 += 0.5 * ((double)vl.y * fz + (double)vl.z * fy);
                v5 += (double)vl.z * fz;
            }
        }
        __syncwarp();
    }

    if (lane == 0) {
        if (rfx != 0.0 || rfy != 0.0 || rfz != 0.0) {
            atomicAdd(forces + (int64_t)i * 3 + 0, rfx);
            atomicAdd(forces + (int64_t)i * 3 + 1, rfy);
            atomicAdd(forces + (int64_t)i * 3 + 2, rfz);
        }
        if (STRESS) {
            if (v0 != 0.0) atomicAdd(virial + 0, v0);
            if (v1 != 0.0) atomicAdd(virial + 1, v1);
            if (v2 != 0.0) atomicAdd(virial + 2, v2);
            if (v3 != 0.0) atomicAdd(virial + 3, v3);
            if (v4 != 0.0) atomicAdd(virial + 4, v4);
            if (v5 != 0.0) atomicAdd(virial + 5, v5);
        }
    }
}

// =================================================================================================
// Kernel 8f: SMEM-Staged 256-Thread FUSED W_msg_0^T & W_msg_1^T + Layer-1 Node Backward
// Stages W_msg0_T + W_msg1_T ONCE per 8 atoms (256 threads) in dynamic SMEM (13.8 KB for l2, 17.9 KB
// for l3) -> zero L1 thrashing, zero bank conflicts across c=0..31, zero dBcat DRAM traffic!
// =================================================================================================
template <int LMAX>
__global__ void __launch_bounds__(256) ilmlip_l1_node_bwd_lf1_msg_kernel(
    int N, int n_feat, int n_msg, Int4 n_l, Int4 msg_in_dim,
    CPtr4 A1, const float* __restrict__ dx, CPtr4 dI,
    const float* __restrict__ W_msg0_T, const float* __restrict__ W_msg1_T, Ptr4 dA1) {
    extern __shared__ float smem_w[];
    __shared__ float s_dI0[8][16];
    __shared__ float s_dI1[8][3][16];

    const int m0d = msg_in_dim.v[0], m1d = msg_in_dim.v[1];
    const int w0_sz = m0d * n_msg;
    const int w1_sz = m1d * n_msg;
    float* sW0T = smem_w;
    float* sW1T = smem_w + w0_sz;

    const int tid = threadIdx.x;
    for (int idx = tid; idx < w0_sz; idx += 256) sW0T[idx] = __ldg(W_msg0_T + idx);
    for (int idx = tid; idx < w1_sz; idx += 256) sW1T[idx] = __ldg(W_msg1_T + idx);
    __syncthreads();

    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int i = blockIdx.x * 8 + wid;
    if (i >= N) return;

    if (lane < n_msg) {
        s_dI0[wid][lane] = __ldg(dI.p[0] + (int64_t)i * n_msg + lane);
        const float* p1 = dI.p[1] + (int64_t)i * 3 * n_msg + lane;
        s_dI1[wid][0][lane] = __ldg(p1);
        s_dI1[wid][1][lane] = __ldg(p1 + n_msg);
        s_dI1[wid][2][lane] = __ldg(p1 + 2 * n_msg);
    }
    __syncwarp();

    const int c = lane;
    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];
    const float* dxi = dx + (int64_t)i * n_feat;

    auto get_gbc0 = [&](int off) -> float {
        const float* wt = sW0T + off + c;
        float acc = 0.0f;
        if (n_msg == 16) {
            #pragma unroll
            for (int q = 0; q < 16; ++q) acc = fmaf(s_dI0[wid][q], wt[q * m0d], acc);
        } else {
            for (int q = 0; q < n_msg; ++q) acc = fmaf(s_dI0[wid][q], wt[q * m0d], acc);
        }
        return acc;
    };
    auto get_gbc1 = [&](int off, float& o0, float& o1, float& o2) {
        const float* wt = sW1T + off + c;
        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;
        if (n_msg == 16) {
            #pragma unroll
            for (int q = 0; q < 16; ++q) {
                const float w = wt[q * m1d];
                r0 = fmaf(s_dI1[wid][0][q], w, r0);
                r1 = fmaf(s_dI1[wid][1][q], w, r1);
                r2 = fmaf(s_dI1[wid][2][q], w, r2);
            }
        } else {
            for (int q = 0; q < n_msg; ++q) {
                const float w = wt[q * m1d];
                r0 = fmaf(s_dI1[wid][0][q], w, r0);
                r1 = fmaf(s_dI1[wid][1][q], w, r1);
                r2 = fmaf(s_dI1[wid][2][q], w, r2);
            }
        }
        o0 = r0; o1 = r1; o2 = r2;
    };

    const float a0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
    if (c < nl1) {
        const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c;
        a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1];
    }
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    if (c < nl2) {
        const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c;
        a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2];
    }
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c;
            a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3];
            a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3];
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

    // Seed nu=1 grads
    float da0 = (c < nl0) ? (dxi[c] + get_gbc0(0)) : 0.0f;
    float da1_0 = 0.0f, da1_1 = 0.0f, da1_2 = 0.0f;
    if (c < nl1) get_gbc1(0, da1_0, da1_1, da1_2);
    float da2_0 = 0.0f, da2_1 = 0.0f, da2_2 = 0.0f, da2_3 = 0.0f, da2_4 = 0.0f;
    float da3_0 = 0.0f, da3_1 = 0.0f, da3_2 = 0.0f, da3_3 = 0.0f, da3_4 = 0.0f, da3_5 = 0.0f, da3_6 = 0.0f;

    constexpr float C110  = 0.5773502588272095f;
    constexpr float CS310 = 0.547722578048706f;
    constexpr float CS110 = 0.3162277638912201f;
    constexpr float CS410 = 0.6324555277824402f;
    constexpr float C220  = 0.4472135901451111f;
    constexpr float K1 = 0.4629100561141968f, K2 = 0.37796446681022644f, K3 = 0.11952286213636398f;
    constexpr float K4 = 0.47809144854545593f, K5 = 0.29277002811431885f, K6 = 0.41403934359550476f;
    constexpr float K7 = 0.5070925354957581f, C330 = 0.37796446681022644f;

    int xoff = nl0, b0off = nl0, b1off = nl1;

    const float b0 = a0 * a0;
    float db0 = (c < nl0) ? (dxi[xoff + c] + get_gbc0(b0off)) : 0.0f;
    xoff += nl0; b0off += nl0;

    const float b1_0 = a0 * a1_0, b1_1 = a0 * a1_1, b1_2 = a0 * a1_2;
    float db1_0 = 0.0f, db1_1 = 0.0f, db1_2 = 0.0f;
    if (c < c01) get_gbc1(b1off, db1_0, db1_1, db1_2);
    b1off += c01;

    const float b2 = C110 * (a1_0 * a1_0 + a1_1 * a1_1 + a1_2 * a1_2);
    float db2 = (c < nl1) ? (dxi[xoff + c] + get_gbc0(b0off)) : 0.0f;
    xoff += nl1; b0off += nl1;

    const float b3_0 = CS310 * (a1_2 * a2_0 + a1_1 * a2_1 - a1_0 * a2_4) - CS110 * a1_0 * a2_2;
    const float b3_1 = CS310 * (a1_0 * a2_1 + a1_2 * a2_3) + CS410 * a1_1 * a2_2;
    const float b3_2 = CS310 * (a1_0 * a2_0 + a1_1 * a2_3 + a1_2 * a2_4) - CS110 * a1_2 * a2_2;
    float db3_0 = 0.0f, db3_1 = 0.0f, db3_2 = 0.0f;
    if (c < c12) get_gbc1(b1off, db3_0, db3_1, db3_2);
    b1off += c12;

    const float b4 = C220 * (a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4);
    float db4 = (c < nl2) ? (dxi[xoff + c] + get_gbc0(b0off)) : 0.0f;
    xoff += nl2; b0off += nl2;

    float b5_0 = 0.0f, b5_1 = 0.0f, b5_2 = 0.0f, db5_0 = 0.0f, db5_1 = 0.0f, db5_2 = 0.0f;
    float b6 = 0.0f, db6 = 0.0f;
    if constexpr (LMAX == 3) {
        b5_0 = K1 * (a2_4 * a3_0 - a2_0 * a3_6) + K2 * (a2_3 * a3_1 - a2_1 * a3_5) +
               K3 * (a2_4 * a3_2 - a2_0 * a3_4) - K5 * a2_1 * a3_3 + K6 * a2_2 * a3_2;
        b5_1 = K2 * (a2_0 * a3_1 + a2_4 * a3_5) + K4 * (a2_1 * a3_2 + a2_3 * a3_4) + K7 * a2_2 * a3_3;
        b5_2 = K1 * (a2_0 * a3_0 + a2_4 * a3_6) + K2 * (a2_1 * a3_1 + a2_3 * a3_5) -
               K3 * (a2_0 * a3_2 + a2_4 * a3_4) - K5 * a2_3 * a3_3 + K6 * a2_2 * a3_4;
        if (c < c23) get_gbc1(b1off, db5_0, db5_1, db5_2);
        b1off += c23;

        b6 = C330 * (a3_0 * a3_0 + a3_1 * a3_1 + a3_2 * a3_2 + a3_3 * a3_3 + a3_4 * a3_4 + a3_5 * a3_5 + a3_6 * a3_6);
        if (c < nl3) db6 = dxi[xoff + c] + get_gbc0(b0off);
        xoff += nl3; b0off += nl3;
    }

    // Backprop nu = 3, L = 0
    if (c < nl0) {
        const float g = dxi[xoff + c] + get_gbc0(b0off);
        db0 = fmaf(g, a0, db0);
        da0 = fmaf(g, b0, da0);
    }
    xoff += nl0; b0off += nl0;

    if (c < c01) {
        const float g = (dxi[xoff + c] + get_gbc0(b0off)) * C110;
        db1_0 = fmaf(g, a1_0, db1_0); db1_1 = fmaf(g, a1_1, db1_1); db1_2 = fmaf(g, a1_2, db1_2);
        da1_0 = fmaf(g, b1_0, da1_0); da1_1 = fmaf(g, b1_1, da1_1); da1_2 = fmaf(g, b1_2, da1_2);
    }
    xoff += c01; b0off += c01;

    if (c < c12) {
        const float g = (dxi[xoff + c] + get_gbc0(b0off)) * C110;
        db3_0 = fmaf(g, a1_0, db3_0); db3_1 = fmaf(g, a1_1, db3_1); db3_2 = fmaf(g, a1_2, db3_2);
        da1_0 = fmaf(g, b3_0, da1_0); da1_1 = fmaf(g, b3_1, da1_1); da1_2 = fmaf(g, b3_2, da1_2);
    }
    xoff += c12; b0off += c12;

    if (c < c02) {
        const float g = dxi[xoff + c] + get_gbc0(b0off);
        db4 = fmaf(g, a0, db4);
        da0 = fmaf(g, b4, da0);
    }
    xoff += c02; b0off += c02;

    if constexpr (LMAX == 3) {
        if (c < c123) {
            const float g = (dxi[xoff + c] + get_gbc0(b0off)) * C110;
            db5_0 = fmaf(g, a1_0, db5_0); db5_1 = fmaf(g, a1_1, db5_1); db5_2 = fmaf(g, a1_2, db5_2);
            da1_0 = fmaf(g, b5_0, da1_0); da1_1 = fmaf(g, b5_1, da1_1); da1_2 = fmaf(g, b5_2, da1_2);
        }
        xoff += c123; b0off += c123;

        if (c < c03) {
            const float g = dxi[xoff + c] + get_gbc0(b0off);
            db6 = fmaf(g, a0, db6);
            da0 = fmaf(g, b6, da0);
        }
        xoff += c03; b0off += c03;
    }

    // Backprop nu = 3, L = 1
    if (c < c01) {
        float g0, g1, g2; get_gbc1(b1off, g0, g1, g2);
        db0 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db0)));
        da1_0 = fmaf(g0, b0, da1_0); da1_1 = fmaf(g1, b0, da1_1); da1_2 = fmaf(g2, b0, da1_2);
    }
    b1off += c01;

    if (c < nl1) {
        float g0, g1, g2; get_gbc1(b1off, g0, g1, g2);
        db2 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db2)));
        da1_0 = fmaf(g0, b2, da1_0); da1_1 = fmaf(g1, b2, da1_1); da1_2 = fmaf(g2, b2, da1_2);
    }
    b1off += nl1;

    if (c < c012) {
        float g0, g1, g2; get_gbc1(b1off, g0, g1, g2);
        db3_0 = fmaf(g0, a0, db3_0); db3_1 = fmaf(g1, a0, db3_1); db3_2 = fmaf(g2, a0, db3_2);
        da0 = fmaf(g0, b3_0, fmaf(g1, b3_1, fmaf(g2, b3_2, da0)));
    }
    b1off += c012;

    if (c < c12) {
        float g0, g1, g2; get_gbc1(b1off, g0, g1, g2);
        db4 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db4)));
        da1_0 = fmaf(g0, b4, da1_0); da1_1 = fmaf(g1, b4, da1_1); da1_2 = fmaf(g2, b4, da1_2);
    }
    b1off += c12;

    if constexpr (LMAX == 3) {
        if (c < c023) {
            float g0, g1, g2; get_gbc1(b1off, g0, g1, g2);
            db5_0 = fmaf(g0, a0, db5_0); db5_1 = fmaf(g1, a0, db5_1); db5_2 = fmaf(g2, a0, db5_2);
            da0 = fmaf(g0, b5_0, fmaf(g1, b5_1, fmaf(g2, b5_2, da0)));
        }
        b1off += c023;

        if (c < c13) {
            float g0, g1, g2; get_gbc1(b1off, g0, g1, g2);
            db6 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db6)));
            da1_0 = fmaf(g0, b6, da1_0); da1_1 = fmaf(g1, b6, da1_1); da1_2 = fmaf(g2, b6, da1_2);
        }
    }

    // Backprop nu = 2 -> da0..da3
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

    if (c < nl0) dA1.p[0][(int64_t)i * nl0 + c] = da0;
    if (c < nl1) {
        float* d1 = dA1.p[1] + (int64_t)i * 3 * nl1 + c;
        d1[0] = da1_0; d1[nl1] = da1_1; d1[2 * nl1] = da1_2;
    }
    if (c < nl2) {
        float* d2 = dA1.p[2] + (int64_t)i * 5 * nl2 + c;
        d2[0] = da2_0; d2[nl2] = da2_1; d2[2 * nl2] = da2_2; d2[3 * nl2] = da2_3; d2[4 * nl2] = da2_4;
    }
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            float* d3 = dA1.p[3] + (int64_t)i * 7 * nl3 + c;
            d3[0] = da3_0; d3[nl3] = da3_1; d3[2 * nl3] = da3_2; d3[3 * nl3] = da3_3;
            d3[4 * nl3] = da3_4; d3[5 * nl3] = da3_5; d3[6 * nl3] = da3_6;
        }
    }
}

// =================================================================================================
// Kernel 8s: Zero-Spill Register-Specialized Layer-1 Node Backward (lfeat==1, lmsg==1, LMAX in {2,3})
// 0 bytes stack frame, 0 SMEM, 0 CG table lookups — exact reverse of ilmlip_l1_node_fwd_lf1_kernel!
// =================================================================================================
template <int LMAX>
__global__ void __launch_bounds__(256) ilmlip_l1_node_bwd_lf1_kernel(
    int N, int n_feat, Int4 n_l, Int4 msg_in_dim,
    CPtr4 A1, const float* __restrict__ dx, CPtr4 dBcat, Ptr4 dA1) {
    const int c = blockIdx.y * 32 + threadIdx.x;
    const int i = blockIdx.x * 8 + threadIdx.y;
    if (i >= N) return;

    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];
    const int m0d = msg_in_dim.v[0], m1d = msg_in_dim.v[1];
    const float* dxi = dx + (int64_t)i * n_feat;
    const float* gbc0 = dBcat.p[0] + (int64_t)i * m0d;
    const float* gbc1 = dBcat.p[1] + (int64_t)i * 3 * m1d;

    const float a0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
    if (c < nl1) {
        const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c;
        a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1];
    }
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    if (c < nl2) {
        const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c;
        a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2];
    }
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c;
            a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3];
            a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3];
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

    // Seed nu=1 grads
    float da0 = (c < nl0) ? (dxi[c] + gbc0[c]) : 0.0f;
    float da1_0 = 0.0f, da1_1 = 0.0f, da1_2 = 0.0f;
    if (c < nl1) { da1_0 = gbc1[c]; da1_1 = gbc1[m1d + c]; da1_2 = gbc1[2 * m1d + c]; }
    float da2_0 = 0.0f, da2_1 = 0.0f, da2_2 = 0.0f, da2_3 = 0.0f, da2_4 = 0.0f;
    float da3_0 = 0.0f, da3_1 = 0.0f, da3_2 = 0.0f, da3_3 = 0.0f, da3_4 = 0.0f, da3_5 = 0.0f, da3_6 = 0.0f;

    // Recompute nu=2 values and seed db2 from direct outputs
    constexpr float C110  = 0.5773502588272095f;
    constexpr float CS310 = 0.547722578048706f;
    constexpr float CS110 = 0.3162277638912201f;
    constexpr float CS410 = 0.6324555277824402f;
    constexpr float C220  = 0.4472135901451111f;
    constexpr float K1 = 0.4629100561141968f, K2 = 0.37796446681022644f, K3 = 0.11952286213636398f;
    constexpr float K4 = 0.47809144854545593f, K5 = 0.29277002811431885f, K6 = 0.41403934359550476f;
    constexpr float K7 = 0.5070925354957581f, C330 = 0.37796446681022644f;

    int xoff = nl0, b0off = nl0, b1off = nl1;

    const float b0 = a0 * a0;
    float db0 = (c < nl0) ? (dxi[xoff + c] + gbc0[b0off + c]) : 0.0f;
    xoff += nl0; b0off += nl0;

    const float b1_0 = a0 * a1_0, b1_1 = a0 * a1_1, b1_2 = a0 * a1_2;
    float db1_0 = 0.0f, db1_1 = 0.0f, db1_2 = 0.0f;
    if (c < c01) { db1_0 = gbc1[b1off + c]; db1_1 = gbc1[m1d + b1off + c]; db1_2 = gbc1[2 * m1d + b1off + c]; }
    b1off += c01;

    const float b2 = C110 * (a1_0 * a1_0 + a1_1 * a1_1 + a1_2 * a1_2);
    float db2 = (c < nl1) ? (dxi[xoff + c] + gbc0[b0off + c]) : 0.0f;
    xoff += nl1; b0off += nl1;

    const float b3_0 = CS310 * (a1_2 * a2_0 + a1_1 * a2_1 - a1_0 * a2_4) - CS110 * a1_0 * a2_2;
    const float b3_1 = CS310 * (a1_0 * a2_1 + a1_2 * a2_3) + CS410 * a1_1 * a2_2;
    const float b3_2 = CS310 * (a1_0 * a2_0 + a1_1 * a2_3 + a1_2 * a2_4) - CS110 * a1_2 * a2_2;
    float db3_0 = 0.0f, db3_1 = 0.0f, db3_2 = 0.0f;
    if (c < c12) { db3_0 = gbc1[b1off + c]; db3_1 = gbc1[m1d + b1off + c]; db3_2 = gbc1[2 * m1d + b1off + c]; }
    b1off += c12;

    const float b4 = C220 * (a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4);
    float db4 = (c < nl2) ? (dxi[xoff + c] + gbc0[b0off + c]) : 0.0f;
    xoff += nl2; b0off += nl2;

    float b5_0 = 0.0f, b5_1 = 0.0f, b5_2 = 0.0f, db5_0 = 0.0f, db5_1 = 0.0f, db5_2 = 0.0f;
    float b6 = 0.0f, db6 = 0.0f;
    if constexpr (LMAX == 3) {
        b5_0 = K1 * (a2_4 * a3_0 - a2_0 * a3_6) + K2 * (a2_3 * a3_1 - a2_1 * a3_5) +
               K3 * (a2_4 * a3_2 - a2_0 * a3_4) - K5 * a2_1 * a3_3 + K6 * a2_2 * a3_2;
        b5_1 = K2 * (a2_0 * a3_1 + a2_4 * a3_5) + K4 * (a2_1 * a3_2 + a2_3 * a3_4) + K7 * a2_2 * a3_3;
        b5_2 = K1 * (a2_0 * a3_0 + a2_4 * a3_6) + K2 * (a2_1 * a3_1 + a2_3 * a3_5) -
               K3 * (a2_0 * a3_2 + a2_4 * a3_4) - K5 * a2_3 * a3_3 + K6 * a2_2 * a3_4;
        if (c < c23) { db5_0 = gbc1[b1off + c]; db5_1 = gbc1[m1d + b1off + c]; db5_2 = gbc1[2 * m1d + b1off + c]; }
        b1off += c23;

        b6 = C330 * (a3_0 * a3_0 + a3_1 * a3_1 + a3_2 * a3_2 + a3_3 * a3_3 + a3_4 * a3_4 + a3_5 * a3_5 + a3_6 * a3_6);
        if (c < nl3) db6 = dxi[xoff + c] + gbc0[b0off + c];
        xoff += nl3; b0off += nl3;
    }

    // Backprop nu = 3, L = 0
    if (c < nl0) {
        const float g = dxi[xoff + c] + gbc0[b0off + c];
        db0 = fmaf(g, a0, db0);
        da0 = fmaf(g, b0, da0);
    }
    xoff += nl0; b0off += nl0;

    if (c < c01) {
        const float g = (dxi[xoff + c] + gbc0[b0off + c]) * C110;
        db1_0 = fmaf(g, a1_0, db1_0); db1_1 = fmaf(g, a1_1, db1_1); db1_2 = fmaf(g, a1_2, db1_2);
        da1_0 = fmaf(g, b1_0, da1_0); da1_1 = fmaf(g, b1_1, da1_1); da1_2 = fmaf(g, b1_2, da1_2);
    }
    xoff += c01; b0off += c01;

    if (c < c12) {
        const float g = (dxi[xoff + c] + gbc0[b0off + c]) * C110;
        db3_0 = fmaf(g, a1_0, db3_0); db3_1 = fmaf(g, a1_1, db3_1); db3_2 = fmaf(g, a1_2, db3_2);
        da1_0 = fmaf(g, b3_0, da1_0); da1_1 = fmaf(g, b3_1, da1_1); da1_2 = fmaf(g, b3_2, da1_2);
    }
    xoff += c12; b0off += c12;

    if (c < c02) {
        const float g = dxi[xoff + c] + gbc0[b0off + c];
        db4 = fmaf(g, a0, db4);
        da0 = fmaf(g, b4, da0);
    }
    xoff += c02; b0off += c02;

    if constexpr (LMAX == 3) {
        if (c < c123) {
            const float g = (dxi[xoff + c] + gbc0[b0off + c]) * C110;
            db5_0 = fmaf(g, a1_0, db5_0); db5_1 = fmaf(g, a1_1, db5_1); db5_2 = fmaf(g, a1_2, db5_2);
            da1_0 = fmaf(g, b5_0, da1_0); da1_1 = fmaf(g, b5_1, da1_1); da1_2 = fmaf(g, b5_2, da1_2);
        }
        xoff += c123; b0off += c123;

        if (c < c03) {
            const float g = dxi[xoff + c] + gbc0[b0off + c];
            db6 = fmaf(g, a0, db6);
            da0 = fmaf(g, b6, da0);
        }
        xoff += c03; b0off += c03;
    }

    // Backprop nu = 3, L = 1
    if (c < c01) {
        const float g0 = gbc1[b1off + c], g1 = gbc1[m1d + b1off + c], g2 = gbc1[2 * m1d + b1off + c];
        db0 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db0)));
        da1_0 = fmaf(g0, b0, da1_0); da1_1 = fmaf(g1, b0, da1_1); da1_2 = fmaf(g2, b0, da1_2);
    }
    b1off += c01;

    if (c < nl1) {
        const float g0 = gbc1[b1off + c], g1 = gbc1[m1d + b1off + c], g2 = gbc1[2 * m1d + b1off + c];
        db2 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db2)));
        da1_0 = fmaf(g0, b2, da1_0); da1_1 = fmaf(g1, b2, da1_1); da1_2 = fmaf(g2, b2, da1_2);
    }
    b1off += nl1;

    if (c < c012) {
        const float g0 = gbc1[b1off + c], g1 = gbc1[m1d + b1off + c], g2 = gbc1[2 * m1d + b1off + c];
        db3_0 = fmaf(g0, a0, db3_0); db3_1 = fmaf(g1, a0, db3_1); db3_2 = fmaf(g2, a0, db3_2);
        da0 = fmaf(g0, b3_0, fmaf(g1, b3_1, fmaf(g2, b3_2, da0)));
    }
    b1off += c012;

    if (c < c12) {
        const float g0 = gbc1[b1off + c], g1 = gbc1[m1d + b1off + c], g2 = gbc1[2 * m1d + b1off + c];
        db4 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db4)));
        da1_0 = fmaf(g0, b4, da1_0); da1_1 = fmaf(g1, b4, da1_1); da1_2 = fmaf(g2, b4, da1_2);
    }
    b1off += c12;

    if constexpr (LMAX == 3) {
        if (c < c023) {
            const float g0 = gbc1[b1off + c], g1 = gbc1[m1d + b1off + c], g2 = gbc1[2 * m1d + b1off + c];
            db5_0 = fmaf(g0, a0, db5_0); db5_1 = fmaf(g1, a0, db5_1); db5_2 = fmaf(g2, a0, db5_2);
            da0 = fmaf(g0, b5_0, fmaf(g1, b5_1, fmaf(g2, b5_2, da0)));
        }
        b1off += c023;

        if (c < c13) {
            const float g0 = gbc1[b1off + c], g1 = gbc1[m1d + b1off + c], g2 = gbc1[2 * m1d + b1off + c];
            db6 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db6)));
            da1_0 = fmaf(g0, b6, da1_0); da1_1 = fmaf(g1, b6, da1_1); da1_2 = fmaf(g2, b6, da1_2);
        }
    }

    // Backprop nu = 2 -> da0..da3
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

    if (c < nl0) dA1.p[0][(int64_t)i * nl0 + c] = da0;
    if (c < nl1) {
        float* d1 = dA1.p[1] + (int64_t)i * 3 * nl1 + c;
        d1[0] = da1_0; d1[nl1] = da1_1; d1[2 * nl1] = da1_2;
    }
    if (c < nl2) {
        float* d2 = dA1.p[2] + (int64_t)i * 5 * nl2 + c;
        d2[0] = da2_0; d2[nl2] = da2_1; d2[2 * nl2] = da2_2; d2[3 * nl2] = da2_3; d2[4 * nl2] = da2_4;
    }
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            float* d3 = dA1.p[3] + (int64_t)i * 7 * nl3 + c;
            d3[0] = da3_0; d3[nl3] = da3_1; d3[2 * nl3] = da3_2; d3[3 * nl3] = da3_3;
            d3[4 * nl3] = da3_4; d3[5 * nl3] = da3_5; d3[6 * nl3] = da3_6;
        }
    }
}

// =================================================================================================
// Kernel 8a: Layer-1 Node Backward (generic fallback: SMEM a1/da1/b2/db2 -> 0 bytes stack frame!)
// =================================================================================================
template <int LMAX, bool SMEM_TBL>
__global__ void ilmlip_l1_node_bwd_kernel(
    int N, int lmsg, int n_nu2, int n_nu3, int cg_total, int n_feat, int b2_scalars,
    Int4 n_l, Int4 msg_in_dim,
    const int32_t* __restrict__ nu2, const int32_t* __restrict__ nu3,
    const int32_t* __restrict__ cg_idx, const float* __restrict__ cg_val,
    CPtr4 A1, const float* __restrict__ dx, CPtr4 dBcat, Ptr4 dA1) {
    constexpr int N_SH = (LMAX + 1) * (LMAX + 1);
    extern __shared__ float smem[];
    const int nu2_ints = n_nu2 * 9;
    const int nu3_ints = n_nu3 * 8;
    int32_t* s_nu2 = reinterpret_cast<int32_t*>(smem);
    int32_t* s_nu3 = s_nu2 + nu2_ints;
    int32_t* s_cg_idx = s_nu3 + nu3_ints;
    float* s_cg_val = reinterpret_cast<float*>(s_cg_idx + cg_total);
    float* work = SMEM_TBL ? (s_cg_val + cg_total) : smem;

    if constexpr (SMEM_TBL) {
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int nthr = blockDim.x * blockDim.y;
        for (int k = tid; k < nu2_ints; k += nthr) s_nu2[k] = __ldg(nu2 + k);
        for (int k = tid; k < nu3_ints; k += nthr) s_nu3[k] = __ldg(nu3 + k);
        for (int k = tid; k < cg_total; k += nthr) {
            s_cg_idx[k] = __ldg(cg_idx + k);
            s_cg_val[k] = __ldg(cg_val + k);
        }
        __syncthreads();
    }

    const int32_t* tbl_nu2 = SMEM_TBL ? s_nu2 : nu2;
    const int32_t* tbl_nu3 = SMEM_TBL ? s_nu3 : nu3;
    const int32_t* tbl_cgi = SMEM_TBL ? s_cg_idx : cg_idx;
    const float*   tbl_cgv = SMEM_TBL ? s_cg_val : cg_val;

    const int c_local = threadIdx.x;
    const int ay = threadIdx.y;
    const int i = blockIdx.x * blockDim.y + ay;
    const int c = blockIdx.y * 32 + c_local;
    if (i >= N) return;

    const int per_atom = 2 * (N_SH + b2_scalars);
    float* a1_s  = work + ((size_t)ay * per_atom) * 32 + c_local;
    float* da1_s = a1_s + (size_t)N_SH * 32;
    float* b2_s  = da1_s + (size_t)N_SH * 32;
    float* db2_s = b2_s + (size_t)b2_scalars * 32;

    #pragma unroll
    for (int m = 0; m < N_SH; ++m) { a1_s[m * 32] = 0.0f; da1_s[m * 32] = 0.0f; }

    #pragma unroll
    for (int l = 0; l <= LMAX; ++l) {
        const int nl = n_l.v[l];
        const int dm = 2 * l + 1;
        const int m0 = l * l;
        if (c < nl) {
            const float* src = A1.p[l] + (int64_t)i * dm * nl + c;
            #pragma unroll
            for (int m = 0; m < dm; ++m) a1_s[(m0 + m) * 32] = src[m * nl];
            if (l == 0) da1_s[0] += dx[(int64_t)i * n_feat + c];
            if (l <= lmsg) {
                const float* gsrc = dBcat.p[l] + (int64_t)i * dm * msg_in_dim.v[l] + c;
                #pragma unroll
                for (int m = 0; m < dm; ++m) da1_s[(m0 + m) * 32] += gsrc[m * msg_in_dim.v[l]];
            }
        }
    }

    for (int p = 0; p < n_nu2; ++p) {
        const int32_t* row = tbl_nu2 + p * 9;
        const int c_lim = row[3];
        if (c >= c_lim) continue;
        const int l1 = row[0], l2 = row[1], L = row[2];
        const int cg_off = row[4], cg_cnt = row[5], b2_off = row[6], x1_off = row[7], bcat_off = row[8];
        const int m01 = l1 * l1, m02 = l2 * l2;
        if (L == 0) {
            float out0 = 0.0f;
            for (int k = 0; k < cg_cnt; ++k) {
                const int pk = tbl_cgi[cg_off + k];
                out0 = fmaf(tbl_cgv[cg_off + k], a1_s[(m01 + (pk & 15)) * 32] * a1_s[(m02 + ((pk >> 4) & 15)) * 32], out0);
            }
            const float g0 = dx[(int64_t)i * n_feat + x1_off + c] +
                             ((0 <= lmsg) ? dBcat.p[0][(int64_t)i * msg_in_dim.v[0] + bcat_off + c] : 0.0f);
            b2_s[b2_off * 32] = out0;
            db2_s[b2_off * 32] = g0;
        } else {
            const int dm = 2 * L + 1;
            float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f, out4 = 0.0f;
            for (int k = 0; k < cg_cnt; ++k) {
                const int pk = tbl_cgi[cg_off + k];
                const int m1 = pk & 15, m2 = (pk >> 4) & 15, m3 = (pk >> 8) & 15;
                const float prod = tbl_cgv[cg_off + k] * (a1_s[(m01 + m1) * 32] * a1_s[(m02 + m2) * 32]);
                if (m3 == 0) out0 += prod;
                else if (m3 == 1) out1 += prod;
                else if (m3 == 2) out2 += prod;
                else if (m3 == 3) out3 += prod;
                else out4 += prod;
            }
            const int stL = (L == 1 ? msg_in_dim.v[1] : msg_in_dim.v[2]);
            const float* gsrc = (L <= lmsg) ? ((L == 1 ? dBcat.p[1] : dBcat.p[2]) + (int64_t)i * dm * stL + bcat_off + c) : nullptr;
            b2_s[(b2_off + 0) * 32] = out0; b2_s[(b2_off + 1) * 32] = out1; b2_s[(b2_off + 2) * 32] = out2;
            db2_s[(b2_off + 0) * 32] = (L <= lmsg) ? gsrc[0] : 0.0f;
            db2_s[(b2_off + 1) * 32] = (L <= lmsg) ? gsrc[stL] : 0.0f;
            db2_s[(b2_off + 2) * 32] = (L <= lmsg) ? gsrc[2 * stL] : 0.0f;
            if (dm > 3) {
                b2_s[(b2_off + 3) * 32] = out3; b2_s[(b2_off + 4) * 32] = out4;
                db2_s[(b2_off + 3) * 32] = (L <= lmsg) ? gsrc[3 * stL] : 0.0f;
                db2_s[(b2_off + 4) * 32] = (L <= lmsg) ? gsrc[4 * stL] : 0.0f;
            }
        }
    }

    for (int q = 0; q < n_nu3; ++q) {
        const int32_t* row = tbl_nu3 + q * 8;
        const int c_lim = row[3];
        if (c >= c_lim) continue;
        const int b2_off = row[0], l3 = row[1], L = row[2];
        const int cg_off = row[4], cg_cnt = row[5], x1_off = row[6], bcat_off = row[7];
        const int m03 = l3 * l3;
        if (L == 0) {
            const float g0 = dx[(int64_t)i * n_feat + x1_off + c] +
                             ((0 <= lmsg) ? dBcat.p[0][(int64_t)i * msg_in_dim.v[0] + bcat_off + c] : 0.0f);
            for (int k = 0; k < cg_cnt; ++k) {
                const int pk = tbl_cgi[cg_off + k];
                const int m12 = pk & 15, m3 = (pk >> 4) & 15;
                const float gv = g0 * tbl_cgv[cg_off + k];
                db2_s[(b2_off + m12) * 32] = fmaf(gv, a1_s[(m03 + m3) * 32], db2_s[(b2_off + m12) * 32]);
                da1_s[(m03 + m3) * 32] = fmaf(gv, b2_s[(b2_off + m12) * 32], da1_s[(m03 + m3) * 32]);
            }
        } else {
            const int dm = 2 * L + 1;
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f, g4 = 0.0f;
            if (L <= lmsg) {
                const int stL = (L == 1 ? msg_in_dim.v[1] : msg_in_dim.v[2]);
                const float* gsrc = (L == 1 ? dBcat.p[1] : dBcat.p[2]) + (int64_t)i * dm * stL + bcat_off + c;
                g0 = gsrc[0]; g1 = gsrc[stL]; g2 = gsrc[2 * stL];
                if (dm > 3) { g3 = gsrc[3 * stL]; g4 = gsrc[4 * stL]; }
            }
            for (int k = 0; k < cg_cnt; ++k) {
                const int pk = tbl_cgi[cg_off + k];
                const int m12 = pk & 15, m3 = (pk >> 4) & 15, mL = (pk >> 8) & 15;
                const float gv = (mL == 0 ? g0 : mL == 1 ? g1 : mL == 2 ? g2 : mL == 3 ? g3 : g4) * tbl_cgv[cg_off + k];
                db2_s[(b2_off + m12) * 32] = fmaf(gv, a1_s[(m03 + m3) * 32], db2_s[(b2_off + m12) * 32]);
                da1_s[(m03 + m3) * 32] = fmaf(gv, b2_s[(b2_off + m12) * 32], da1_s[(m03 + m3) * 32]);
            }
        }
    }

    for (int p = 0; p < n_nu2; ++p) {
        const int32_t* row = tbl_nu2 + p * 9;
        const int c_lim = row[3];
        if (c >= c_lim) continue;
        const int l1 = row[0], l2 = row[1];
        const int cg_off = row[4], cg_cnt = row[5], b2_off = row[6];
        const int m01 = l1 * l1, m02 = l2 * l2;
        for (int k = 0; k < cg_cnt; ++k) {
            const int pk = tbl_cgi[cg_off + k];
            const int m1 = pk & 15, m2 = (pk >> 4) & 15, m3 = (pk >> 8) & 15;
            const float g = db2_s[(b2_off + m3) * 32] * tbl_cgv[cg_off + k];
            da1_s[(m01 + m1) * 32] = fmaf(g, a1_s[(m02 + m2) * 32], da1_s[(m01 + m1) * 32]);
            da1_s[(m02 + m2) * 32] = fmaf(g, a1_s[(m01 + m1) * 32], da1_s[(m02 + m2) * 32]);
        }
    }

    #pragma unroll
    for (int l = 0; l <= LMAX; ++l) {
        const int nl = n_l.v[l];
        if (c < nl) {
            const int dm = 2 * l + 1;
            const int m0 = l * l;
            float* dst = dA1.p[l] + (int64_t)i * dm * nl + c;
            #pragma unroll
            for (int m = 0; m < dm; ++m) dst[m * nl] = da1_s[(m0 + m) * 32];
        }
    }
}

// =================================================================================================
// Kernel 8b-pre: Pre-Contracted Layer-1 Edge Backward (S*NB<=32, NB in {8,12,16}, max_nl<=32)
// Phase 1: ZERO-SHUFFLE pre-contraction! Each lane p = zj*NB + k (< S*NB <= 32) independently
//   contracts s_gA1[wid] against coalesced C1_zick[z_i, c, p] -> 0 shuffles (~12x faster Phase 1!).
// Phase 2: 1 THREAD = 1 FULL EDGE with streaming FP32 Chebyshev + exact pos64 FP64 ZBL close-contact!
// =================================================================================================
template <int LMAX, int NB, bool ZBL_SW, bool STRESS>
__global__ void __launch_bounds__(128) ilmlip_l1_edge_bwd_precontract_kernel(
    int N, int S, int n_tot1, Int4 n_l, Int4 l1_off, float r_max, float inv_DA1_a0_kap_sq,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const double* __restrict__ pos64,
    const float* __restrict__ C1_zick,  // [S, n_tot1, S*NB]
    const double* __restrict__ rs_pair, const double* __restrict__ zz_pair, const double* __restrict__ au_pair,
    const float* __restrict__ w_zbl,
    CPtr4 A1, CPtr4 dA1, const float* __restrict__ f0_arr, const float* __restrict__ df0_bf,
    double* __restrict__ forces, double* __restrict__ virial) {
    constexpr int N_SH = (LMAX + 1) * (LMAX + 1);
    constexpr int GT_STRIDE = ((NB * N_SH) % 16 == 0) ? (NB * N_SH + 8) : (NB * N_SH);
    constexpr int MAX_GA = (LMAX == 2) ? 256 : 384;
    __shared__ float s_Gt[4][4 * GT_STRIDE];
    __shared__ float s_gA1[4][MAX_GA];

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    if (eb >= ee) return;

    const int z_i = species[i];
    const float wz_i = ZBL_SW ? w_zbl[i] : 0.0f;
    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];
    const int off1 = l1_off.v[1], off2 = l1_off.v[2], off3 = l1_off.v[3];
    const int ga1 = nl0;
    const int ga2 = ga1 + 3 * nl1;
    const int ga3 = ga2 + 5 * nl2;

    // Phase 1a: FUSED f0 gate backward directly into s_gA1[wid] (skips ilmlip_gate_bwd_kernel & DRAM round-trip!)
    float* gA = s_gA1[wid];
    float dot_lane = 0.0f;
    const float* pA0 = A1.p[0] + (int64_t)i * nl0;
    const float* pd0 = dA1.p[0] + (int64_t)i * nl0;
    const float* pA1 = A1.p[1] + (int64_t)i * 3 * nl1;
    const float* pd1 = dA1.p[1] + (int64_t)i * 3 * nl1;
    const float* pA2 = A1.p[2] + (int64_t)i * 5 * nl2;
    const float* pd2 = dA1.p[2] + (int64_t)i * 5 * nl2;
    const float* pA3 = (LMAX == 3) ? (A1.p[3] + (int64_t)i * 7 * nl3) : nullptr;
    const float* pd3 = (LMAX == 3) ? (dA1.p[3] + (int64_t)i * 7 * nl3) : nullptr;

    if (lane < nl0) dot_lane = fmaf(pA0[lane], pd0[lane], dot_lane);
    for (int idx = lane; idx < 3 * nl1; idx += 32) dot_lane = fmaf(pA1[idx], pd1[idx], dot_lane);
    for (int idx = lane; idx < 5 * nl2; idx += 32) dot_lane = fmaf(pA2[idx], pd2[idx], dot_lane);
    if constexpr (LMAX == 3) {
        for (int idx = lane; idx < 7 * nl3; idx += 32) dot_lane = fmaf(pA3[idx], pd3[idx], dot_lane);
    }
    const float dot = warp_sum_f(dot_lane);
    const float f0_i = f0_arr[i];
    const float df_ext = df0_bf ? df0_bf[i] : 0.0f;
    const float sc = (df_ext * f0_i * f0_i + dot * f0_i) * inv_DA1_a0_kap_sq;

    if (lane < nl0) gA[lane] = fmaf(f0_i, pd0[lane], -sc * pA0[lane]);
    for (int idx = lane; idx < 3 * nl1; idx += 32) gA[ga1 + idx] = fmaf(f0_i, pd1[idx], -sc * pA1[idx]);
    for (int idx = lane; idx < 5 * nl2; idx += 32) gA[ga2 + idx] = fmaf(f0_i, pd2[idx], -sc * pA2[idx]);
    if constexpr (LMAX == 3) {
        for (int idx = lane; idx < 7 * nl3; idx += 32) gA[ga3 + idx] = fmaf(f0_i, pd3[idx], -sc * pA3[idx]);
    }
    __syncwarp();

    // Phase 1b: ZERO-SHUFFLE contraction! Lane p = zj*NB + k (< S*NB <= 32) computes its full N_SH row!
    const int snb = S * NB;
    if (lane < snb) {
        const float* C_zi = C1_zick + (int64_t)z_i * n_tot1 * snb + lane;
        float s0 = 0.0f;
        for (int c = 0; c < nl0; ++c) s0 = fmaf(gA[c], __ldg(C_zi + c * snb), s0);

        float s1_0 = 0.0f, s1_1 = 0.0f, s1_2 = 0.0f;
        for (int c = 0; c < nl1; ++c) {
            const float cv = __ldg(C_zi + (off1 + c) * snb);
            s1_0 = fmaf(gA[ga1 + 0 * nl1 + c], cv, s1_0);
            s1_1 = fmaf(gA[ga1 + 1 * nl1 + c], cv, s1_1);
            s1_2 = fmaf(gA[ga1 + 2 * nl1 + c], cv, s1_2);
        }

        float s2_0 = 0.0f, s2_1 = 0.0f, s2_2 = 0.0f, s2_3 = 0.0f, s2_4 = 0.0f;
        for (int c = 0; c < nl2; ++c) {
            const float cv = __ldg(C_zi + (off2 + c) * snb);
            s2_0 = fmaf(gA[ga2 + 0 * nl2 + c], cv, s2_0);
            s2_1 = fmaf(gA[ga2 + 1 * nl2 + c], cv, s2_1);
            s2_2 = fmaf(gA[ga2 + 2 * nl2 + c], cv, s2_2);
            s2_3 = fmaf(gA[ga2 + 3 * nl2 + c], cv, s2_3);
            s2_4 = fmaf(gA[ga2 + 4 * nl2 + c], cv, s2_4);
        }

        const int zj = lane / NB;
        const int k  = lane - zj * NB;
        float* row = &s_Gt[wid][zj * GT_STRIDE + k * N_SH];
        row[0] = s0;
        row[1] = s1_0; row[2] = s1_1; row[3] = s1_2;
        row[4] = s2_0; row[5] = s2_1; row[6] = s2_2; row[7] = s2_3; row[8] = s2_4;

        if constexpr (LMAX == 3) {
            float s3_0 = 0.0f, s3_1 = 0.0f, s3_2 = 0.0f, s3_3 = 0.0f, s3_4 = 0.0f, s3_5 = 0.0f, s3_6 = 0.0f;
            for (int c = 0; c < nl3; ++c) {
                const float cv = __ldg(C_zi + (off3 + c) * snb);
                s3_0 = fmaf(gA[ga3 + 0 * nl3 + c], cv, s3_0);
                s3_1 = fmaf(gA[ga3 + 1 * nl3 + c], cv, s3_1);
                s3_2 = fmaf(gA[ga3 + 2 * nl3 + c], cv, s3_2);
                s3_3 = fmaf(gA[ga3 + 3 * nl3 + c], cv, s3_3);
                s3_4 = fmaf(gA[ga3 + 4 * nl3 + c], cv, s3_4);
                s3_5 = fmaf(gA[ga3 + 5 * nl3 + c], cv, s3_5);
                s3_6 = fmaf(gA[ga3 + 6 * nl3 + c], cv, s3_6);
            }
            row[9] = s3_0; row[10] = s3_1; row[11] = s3_2; row[12] = s3_3;
            row[13] = s3_4; row[14] = s3_5; row[15] = s3_6;
        }
    }
    __syncwarp();

    // Phase 2: 1 THREAD = 1 FULL EDGE! Zero C1/dA1 reads, zero shuffles, bank-conflict-free Gt!
    double rfx = 0.0, rfy = 0.0, rfz = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;
    const float inv_rmax = 1.0f / r_max;

    for (int e = eb + lane; e < ee; e += 32) {
        const float4 vl = __ldg(vec_len + e);
        const int j = __ldg(senders + e);
        const int z_j = __ldg(species + j);
        const float r = fmaxf(vl.w, 1e-8f);

        float4 Yv[4];
        loup::edge_sh16(vl.x, vl.y, vl.z, r, Yv);
        const float y0 = Yv[0].x, y1 = Yv[0].y, y2 = Yv[0].z, y3 = Yv[0].w;
        const float y4 = Yv[1].x, y5 = Yv[1].y, y6 = Yv[1].z, y7 = Yv[1].w, y8 = Yv[2].x;
        const float y9  = (LMAX == 3) ? Yv[2].y : 0.0f;
        const float y10 = (LMAX == 3) ? Yv[2].z : 0.0f;
        const float y11 = (LMAX == 3) ? Yv[2].w : 0.0f;
        const float y12 = (LMAX == 3) ? Yv[3].x : 0.0f;
        const float y13 = (LMAX == 3) ? Yv[3].y : 0.0f;
        const float y14 = (LMAX == 3) ? Yv[3].z : 0.0f;
        const float y15 = (LMAX == 3) ? Yv[3].w : 0.0f;

        const float* Gt = &s_Gt[wid][z_j * GT_STRIDE];
        float er = 0.0f;
        float dY1 = 0.0f, dY2 = 0.0f, dY3 = 0.0f, dY4 = 0.0f, dY5 = 0.0f, dY6 = 0.0f, dY7 = 0.0f, dY8 = 0.0f;
        float dY9 = 0.0f, dY10 = 0.0f, dY11 = 0.0f, dY12 = 0.0f, dY13 = 0.0f, dY14 = 0.0f, dY15 = 0.0f;

        auto accum_k = [&](int k, float gk, float dgk) {
            const float* row = Gt + k * N_SH;
            float rdot = row[0] * y0;
            float g;
            g = row[1]; rdot = fmaf(g, y1, rdot); dY1 = fmaf(g, gk, dY1);
            g = row[2]; rdot = fmaf(g, y2, rdot); dY2 = fmaf(g, gk, dY2);
            g = row[3]; rdot = fmaf(g, y3, rdot); dY3 = fmaf(g, gk, dY3);
            g = row[4]; rdot = fmaf(g, y4, rdot); dY4 = fmaf(g, gk, dY4);
            g = row[5]; rdot = fmaf(g, y5, rdot); dY5 = fmaf(g, gk, dY5);
            g = row[6]; rdot = fmaf(g, y6, rdot); dY6 = fmaf(g, gk, dY6);
            g = row[7]; rdot = fmaf(g, y7, rdot); dY7 = fmaf(g, gk, dY7);
            g = row[8]; rdot = fmaf(g, y8, rdot); dY8 = fmaf(g, gk, dY8);
            if constexpr (LMAX == 3) {
                g = row[9];  rdot = fmaf(g, y9,  rdot); dY9  = fmaf(g, gk, dY9);
                g = row[10]; rdot = fmaf(g, y10, rdot); dY10 = fmaf(g, gk, dY10);
                g = row[11]; rdot = fmaf(g, y11, rdot); dY11 = fmaf(g, gk, dY11);
                g = row[12]; rdot = fmaf(g, y12, rdot); dY12 = fmaf(g, gk, dY12);
                g = row[13]; rdot = fmaf(g, y13, rdot); dY13 = fmaf(g, gk, dY13);
                g = row[14]; rdot = fmaf(g, y14, rdot); dY14 = fmaf(g, gk, dY14);
                g = row[15]; rdot = fmaf(g, y15, rdot); dY15 = fmaf(g, gk, dY15);
            }
            er = fmaf(rdot, dgk, er);
        };

        {
            float t = __fdiv_rn(r, r_max);
            if (t < 1.0f) {
                if (t < 0.0f) t = 0.0f;
                const float t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
                const float env = 1.0f - 21.0f * t5 + 35.0f * t5 * t - 15.0f * t5 * t2;
                const float omt = 1.0f - t;
                const float denv_dt = -105.0f * t4 * omt * omt;
                const float x = 2.0f * t - 1.0f;
                const float x2 = 2.0f * x;
                float T0 = 1.0f, dT0 = 0.0f;
                accum_k(0, env, denv_dt * inv_rmax);
                float T1 = x, dT1 = 1.0f;
                accum_k(1, T1 * env, (2.0f * dT1 * env + T1 * denv_dt) * inv_rmax);
                #pragma unroll
                for (int k = 2; k < NB; ++k) {
                    const float Tk = fmaf(x2, T1, -T0);
                    const float dTk = fmaf(x2, dT1, fmaf(2.0f, T1, -dT0));
                    accum_k(k, Tk * env, (2.0f * dTk * env + Tk * denv_dt) * inv_rmax);
                    T0 = T1; T1 = Tk; dT0 = dT1; dT1 = dTk;
                }
            }
        }

        const float inv_d = 1.0f / r;
        const float ux = vl.x * inv_d, uy = vl.y * inv_d, uz = vl.z * inv_d;
        float gx, gy, gz;
        if constexpr (LMAX == 2) {
            sh_grad_l2(ux, uy, uz, dY1, dY2, dY3, dY4, dY5, dY6, dY7, dY8, gx, gy, gz);
        } else {
            sh_grad_l3(ux, uy, uz, dY1, dY2, dY3, dY4, dY5, dY6, dY7, dY8,
                       dY9, dY10, dY11, dY12, dY13, dY14, dY15, gx, gy, gz);
        }
        const float dot = ux * gx + uy * gy + uz * gz;
        double fx = (double)(ux * er + (gx - ux * dot) * inv_d);
        double fy = (double)(uy * er + (gy - uy * dot) * inv_d);
        double fz = (double)(uz * er + (gz - uz * dot) * inv_d);
        if constexpr (ZBL_SW) {
            const int pair = z_i * S + z_j;
            const double rs = __ldg(rs_pair + pair);
            if (__builtin_expect((double)r < rs, 0)) {
                const double3 zf = eval_zbl_force_cold(i, j, vl, pos64, wz_i, rs, __ldg(zz_pair + pair), __ldg(au_pair + pair));
                fx += zf.x;
                fy += zf.y;
                fz += zf.z;
            }
        }
        rfx -= fx; rfy -= fy; rfz -= fz;
        atomicAdd(forces + (int64_t)j * 3 + 0, fx);
        atomicAdd(forces + (int64_t)j * 3 + 1, fy);
        atomicAdd(forces + (int64_t)j * 3 + 2, fz);
        if constexpr (STRESS) {
            v0 += (double)vl.x * fx;
            v1 += 0.5 * ((double)vl.x * fy + (double)vl.y * fx);
            v2 += 0.5 * ((double)vl.x * fz + (double)vl.z * fx);
            v3 += (double)vl.y * fy;
            v4 += 0.5 * ((double)vl.y * fz + (double)vl.z * fy);
            v5 += (double)vl.z * fz;
        }
    }

    rfx = warp_sum_d(rfx); rfy = warp_sum_d(rfy); rfz = warp_sum_d(rfz);
    if constexpr (STRESS) {
        v0 = warp_sum_d(v0); v1 = warp_sum_d(v1); v2 = warp_sum_d(v2);
        v3 = warp_sum_d(v3); v4 = warp_sum_d(v4); v5 = warp_sum_d(v5);
    }
    if (lane == 0) {
        if (rfx != 0.0 || rfy != 0.0 || rfz != 0.0) {
            atomicAdd(forces + (int64_t)i * 3 + 0, rfx);
            atomicAdd(forces + (int64_t)i * 3 + 1, rfy);
            atomicAdd(forces + (int64_t)i * 3 + 2, rfz);
        }
        if constexpr (STRESS) {
            if (v0 != 0.0) atomicAdd(virial + 0, v0);
            if (v1 != 0.0) atomicAdd(virial + 1, v1);
            if (v2 != 0.0) atomicAdd(virial + 2, v2);
            if (v3 != 0.0) atomicAdd(virial + 3, v3);
            if (v4 != 0.0) atomicAdd(virial + 4, v4);
            if (v5 != 0.0) atomicAdd(virial + 5, v5);
        }
    }
}

// =================================================================================================
// Kernel 8b-fast: Edge-Parallel Layer-1 Edge Backward (1 block/atom, 8 half-warps = 8 edges/round)
// Stages atom i's dA1_raw (<1 KB) and C1[z_i, 0..S-1] (~5-7 KB) in SMEM, skips rv for l=0 (Y0 const),
// folds [er, dY[1..15]] in a single 4-step butterfly (11-15 shuffles), 1 block atomic for forces[i]/virial!
// =================================================================================================
template <int LMAX, int NB_MAX, bool ZBL_SW, bool STRESS, bool SMEM_C1>
__global__ void __launch_bounds__(128) ilmlip_l1_edge_bwd_edgepar_kernel(
    int N, int S, int n_base, int n_tot1, Int4 n_l, Int4 l1_off, float r_max,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const double* __restrict__ pos64,
    const float* __restrict__ C1,
    const double* __restrict__ rs_pair, const double* __restrict__ zz_pair, const double* __restrict__ au_pair,
    const float* __restrict__ w_zbl, CPtr4 dA1_raw,
    double* __restrict__ forces, double* __restrict__ virial) {
    const int i = blockIdx.x;
    if (i >= N) return;
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    if (eb >= ee) return;

    __shared__ float s_g[4][2][NB_MAX];
    __shared__ float s_dg[4][2][NB_MAX];
    __shared__ float s_Y[4][2][16];
    __shared__ float s_dy[4][32];
    __shared__ float4 s_vl[4][2];
    __shared__ int s_j[4][2];
    __shared__ int s_z_j[4][2];
    __shared__ double s_zbl[4][2][3];
    __shared__ double s_red[4][9];
    extern __shared__ float dyn_smem[];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int half = lane >> 4;
    const int hl = lane & 15;

    const int z_i = species[i];
    const float wz_i = ZBL_SW ? w_zbl[i] : 0.0f;

    int ga_off[4];
    ga_off[0] = 0;
    ga_off[1] = n_l.v[0];
    ga_off[2] = ga_off[1] + 3 * n_l.v[1];
    ga_off[3] = ga_off[2] + 5 * n_l.v[2];
    const int ga_tot = ga_off[3] + ((LMAX == 3) ? 7 * n_l.v[3] : 0);

    float* s_gA1 = dyn_smem;
    float* s_C1_i = dyn_smem + ga_tot;

    #pragma unroll
    for (int l = 0; l <= LMAX; ++l) {
        const int cnt = (2 * l + 1) * n_l.v[l];
        const float* src = dA1_raw.p[l] + (int64_t)i * cnt;
        float* dst = s_gA1 + ga_off[l];
        for (int idx = tid; idx < cnt; idx += 128) dst[idx] = __ldg(src + idx);
    }
    const float* C1_zi = C1 + (int64_t)z_i * S * n_base * n_tot1;
    if constexpr (SMEM_C1) {
        const int c1_zi_elems = S * n_base * n_tot1;
        for (int idx = tid; idx < c1_zi_elems; idx += 128) s_C1_i[idx] = __ldg(C1_zi + idx);
    }
    __syncthreads();

    const float* C1_base = SMEM_C1 ? s_C1_i : C1_zi;
    double rfx = 0.0, rfy = 0.0, rfz = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int base = eb + 2 * wid; base < ee; base += 8) {
        const bool valid = (base + half < ee);
        const int e = valid ? (base + half) : base;
        if (hl < 2) {
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const int z_j = __ldg(species + j);
            const float r = fmaxf(vl.w, 1e-8f);
            if (hl == 0) {
                s_vl[wid][half] = vl;
                s_j[wid][half] = j;
                s_z_j[wid][half] = z_j;
                eval_cheb<true>(r, r_max, n_base, s_g[wid][half], s_dg[wid][half]);
            } else {
                float4 Yv[4];
                loup::edge_sh16(vl.x, vl.y, vl.z, r, Yv);
                unpack_sh16(Yv, LMAX, s_Y[wid][half]);
                if constexpr (ZBL_SW) {
                    const int pair = z_i * S + z_j;
                    const double rs = __ldg(rs_pair + pair);
                    if (__builtin_expect((double)r < rs, 0)) {
                        double rx, ry, rz, rd;
                        zbl_exact_vec(i, j, vl, pos64, rx, ry, rz, rd);
                        const double zbl_er = (double)wz_i * eval_zbl_pair_deriv_d(rd, rs, __ldg(zz_pair + pair), __ldg(au_pair + pair));
                        const double inv_rd = 1.0 / rd;
                        s_zbl[wid][half][0] = (rx * inv_rd) * zbl_er;
                        s_zbl[wid][half][1] = (ry * inv_rd) * zbl_er;
                        s_zbl[wid][half][2] = (rz * inv_rd) * zbl_er;
                    } else {
                        s_zbl[wid][half][0] = 0.0; s_zbl[wid][half][1] = 0.0; s_zbl[wid][half][2] = 0.0;
                    }
                }
            }
        }
        __syncwarp();

        const int z_j = s_z_j[wid][half];
        const float* gk = s_g[wid][half];
        const float* dgk = s_dg[wid][half];
        const float* Yk = s_Y[wid][half];
        const float* Cp = C1_base + (int64_t)z_j * n_base * n_tot1;

        // dY[0] accumulates radial force derivative `er` (Y0 is constant so sh16_grad ignores d[0]!)
        float dY[16];
        #pragma unroll
        for (int m = 0; m < 16; ++m) dY[m] = 0.0f;

        // l = 0: ONLY drv needed (skip rv completely -> 50% fewer FMAs for l=0!)
        {
            const int nl0 = n_l.v[0];
            const float y0 = Yk[0];
            for (int c = hl; c < nl0; c += 16) {
                float drv = 0.0f;
                const float* col = Cp + c;
                if (n_base == 8) {
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) drv = fmaf(SMEM_C1 ? col[k * n_tot1] : __ldg(col + k * n_tot1), dgk[k], drv);
                } else if (n_base == 12) {
                    #pragma unroll
                    for (int k = 0; k < 12; ++k) drv = fmaf(SMEM_C1 ? col[k * n_tot1] : __ldg(col + k * n_tot1), dgk[k], drv);
                } else if (n_base == 16) {
                    #pragma unroll
                    for (int k = 0; k < 16; ++k) drv = fmaf(SMEM_C1 ? col[k * n_tot1] : __ldg(col + k * n_tot1), dgk[k], drv);
                } else {
                    for (int k = 0; k < n_base; ++k) drv = fmaf(SMEM_C1 ? col[k * n_tot1] : __ldg(col + k * n_tot1), dgk[k], drv);
                }
                dY[0] = fmaf(s_gA1[c] * y0, drv, dY[0]);
            }
        }

        // l = 1..LMAX
        #pragma unroll
        for (int l = 1; l <= LMAX; ++l) {
            const int nl = n_l.v[l];
            const int dm = 2 * l + 1;
            const int m0 = l * l;
            const float* gal = s_gA1 + ga_off[l];
            for (int c = hl; c < nl; c += 16) {
                float rv = 0.0f, drv = 0.0f;
                const float* col = Cp + l1_off.v[l] + c;
                auto accum_c = [&](int k) {
                    const float cval = SMEM_C1 ? col[k * n_tot1] : __ldg(col + k * n_tot1);
                    rv = fmaf(cval, gk[k], rv);
                    drv = fmaf(cval, dgk[k], drv);
                };
                if (n_base == 8) {
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) accum_c(k);
                } else if (n_base == 12) {
                    #pragma unroll
                    for (int k = 0; k < 12; ++k) accum_c(k);
                } else if (n_base == 16) {
                    #pragma unroll
                    for (int k = 0; k < 16; ++k) accum_c(k);
                } else {
                    for (int k = 0; k < n_base; ++k) accum_c(k);
                }
                float dr1 = 0.0f;
                #pragma unroll
                for (int m = 0; m < dm; ++m) {
                    const float gval = gal[m * nl + c];
                    dr1 = fmaf(gval, Yk[m0 + m], dr1);
                    dY[m0 + m] = fmaf(gval, rv, dY[m0 + m]);
                }
                dY[0] = fmaf(dr1, drv, dY[0]);
            }
        }

        // 4-step half-warp butterfly fold of dY[0..15]: lane hl ends up with sum_{half} dY[hl]
        // (hl == 0 holds er, hl == 1..N_SH-1 hold dY[hl]). All 32 lanes execute convergently.
        float v_fold;
        if constexpr (LMAX <= 2) {
            const bool p0 = (lane & 1) != 0;
            float u[5];
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                u[k] = (p0 ? dY[2 * k + 1] : dY[2 * k]) + __shfl_xor_sync(0xffffffffu, p0 ? dY[2 * k] : dY[2 * k + 1], 1);
            u[4] = (p0 ? 0.0f : dY[8]) + __shfl_xor_sync(0xffffffffu, p0 ? dY[8] : 0.0f, 1);
            const bool p1 = (lane & 2) != 0;
            float w[3];
            w[0] = (p1 ? u[1] : u[0]) + __shfl_xor_sync(0xffffffffu, p1 ? u[0] : u[1], 2);
            w[1] = (p1 ? u[3] : u[2]) + __shfl_xor_sync(0xffffffffu, p1 ? u[2] : u[3], 2);
            w[2] = (p1 ? 0.0f : u[4]) + __shfl_xor_sync(0xffffffffu, p1 ? u[4] : 0.0f, 2);
            const bool p2 = (lane & 4) != 0;
            const float z0 = (p2 ? w[1] : w[0]) + __shfl_xor_sync(0xffffffffu, p2 ? w[0] : w[1], 4);
            const float z1 = (p2 ? 0.0f : w[2]) + __shfl_xor_sync(0xffffffffu, p2 ? w[2] : 0.0f, 4);
            const bool p3 = (lane & 8) != 0;
            v_fold = (p3 ? z1 : z0) + __shfl_xor_sync(0xffffffffu, p3 ? z0 : z1, 8);
        } else {
            const bool p0 = (lane & 1) != 0;
            float v8[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                v8[k] = (p0 ? dY[2 * k + 1] : dY[2 * k]) + __shfl_xor_sync(0xffffffffu, p0 ? dY[2 * k] : dY[2 * k + 1], 1);
            const bool p1 = (lane & 2) != 0;
            float v4[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                v4[k] = (p1 ? v8[2 * k + 1] : v8[2 * k]) + __shfl_xor_sync(0xffffffffu, p1 ? v8[2 * k] : v8[2 * k + 1], 2);
            const bool p2 = (lane & 4) != 0;
            float v2[2];
            #pragma unroll
            for (int k = 0; k < 2; ++k)
                v2[k] = (p2 ? v4[2 * k + 1] : v4[2 * k]) + __shfl_xor_sync(0xffffffffu, p2 ? v4[2 * k] : v4[2 * k + 1], 4);
            const bool p3 = (lane & 8) != 0;
            v_fold = (p3 ? v2[1] : v2[0]) + __shfl_xor_sync(0xffffffffu, p3 ? v2[0] : v2[1], 8);
        }
        s_dy[wid][lane] = v_fold;
        __syncwarp();

        if (hl == 0 && valid) {
            const float4 vl = s_vl[wid][half];
            const int j = s_j[wid][half];
            const float r = fmaxf(vl.w, 1e-8f);
            float er = v_fold;
            s_dy[wid][half * 16] = 0.0f;
            const float inv_d = 1.0f / r;
            const float ux = vl.x * inv_d, uy = vl.y * inv_d, uz = vl.z * inv_d;
            float gx, gy, gz;
            ::sh16_grad(ux, uy, uz, &s_dy[wid][half * 16], gx, gy, gz);
            const float dot = ux * gx + uy * gy + uz * gz;
            double fx = (double)(ux * er + (gx - ux * dot) * inv_d);
            double fy = (double)(uy * er + (gy - uy * dot) * inv_d);
            double fz = (double)(uz * er + (gz - uz * dot) * inv_d);
            if constexpr (ZBL_SW) {
                fx += s_zbl[wid][half][0];
                fy += s_zbl[wid][half][1];
                fz += s_zbl[wid][half][2];
            }
            rfx -= fx; rfy -= fy; rfz -= fz;
            atomicAdd(forces + (int64_t)j * 3 + 0, fx);
            atomicAdd(forces + (int64_t)j * 3 + 1, fy);
            atomicAdd(forces + (int64_t)j * 3 + 2, fz);
            if constexpr (STRESS) {
                v0 += (double)vl.x * fx;
                v1 += 0.5 * ((double)vl.x * fy + (double)vl.y * fx);
                v2 += 0.5 * ((double)vl.x * fz + (double)vl.z * fx);
                v3 += (double)vl.y * fy;
                v4 += 0.5 * ((double)vl.y * fz + (double)vl.z * fy);
                v5 += (double)vl.z * fz;
            }
        }
        __syncwarp();
    }

    // Reduce receiver i forces + virial across the 4 warps of the block -> single atomicAdd per atom!
    rfx = warp_sum_d(rfx); rfy = warp_sum_d(rfy); rfz = warp_sum_d(rfz);
    if constexpr (STRESS) {
        v0 = warp_sum_d(v0); v1 = warp_sum_d(v1); v2 = warp_sum_d(v2);
        v3 = warp_sum_d(v3); v4 = warp_sum_d(v4); v5 = warp_sum_d(v5);
    }
    if (lane == 0) {
        s_red[wid][0] = rfx; s_red[wid][1] = rfy; s_red[wid][2] = rfz;
        if constexpr (STRESS) {
            s_red[wid][3] = v0; s_red[wid][4] = v1; s_red[wid][5] = v2;
            s_red[wid][6] = v3; s_red[wid][7] = v4; s_red[wid][8] = v5;
        }
    }
    __syncthreads();
    if (tid == 0) {
        const double fx_i = s_red[0][0] + s_red[1][0] + s_red[2][0] + s_red[3][0];
        const double fy_i = s_red[0][1] + s_red[1][1] + s_red[2][1] + s_red[3][1];
        const double fz_i = s_red[0][2] + s_red[1][2] + s_red[2][2] + s_red[3][2];
        if (fx_i != 0.0 || fy_i != 0.0 || fz_i != 0.0) {
            atomicAdd(forces + (int64_t)i * 3 + 0, fx_i);
            atomicAdd(forces + (int64_t)i * 3 + 1, fy_i);
            atomicAdd(forces + (int64_t)i * 3 + 2, fz_i);
        }
        if constexpr (STRESS) {
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                const double vc = s_red[0][3 + c] + s_red[1][3 + c] + s_red[2][3 + c] + s_red[3][3 + c];
                if (vc != 0.0) atomicAdd(virial + c, vc);
            }
        }
    }
}

// =================================================================================================
// Kernel 8b: Layer-1 Edge Backward (8-edge warp batching, pure FP32, parallel sh16_grad, STACK:0)
// =================================================================================================
template <int LMAX, int CH_PASSES, int NB_MAX, bool ZBL_SW, bool STRESS>
__global__ void __launch_bounds__(128) ilmlip_l1_edge_bwd_kernel(
    int N, int S, int n_base, int n_tot1, Int4 n_l, Int4 l1_off, float r_max,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C1,
    const float* __restrict__ rs_pair, const float* __restrict__ zz_pair, const float* __restrict__ au_pair,
    const float* __restrict__ w_zbl, CPtr4 dA1_raw,
    double* __restrict__ forces, double* __restrict__ virial) {
    constexpr int N_SH = (LMAX + 1) * (LMAX + 1);
    __shared__ float s_g[4][8][NB_MAX];
    __shared__ float s_dg[4][8][NB_MAX];
    __shared__ float s_Y[4][8][N_SH];
    __shared__ float4 s_vl[4][8];
    __shared__ int s_j[4][8];
    __shared__ int s_pair[4][8];
    __shared__ float s_er[4][8];
    __shared__ float s_dY[4][8][N_SH];

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    const float wz_i = ZBL_SW ? w_zbl[i] : 0.0f;

    float gA1[CH_PASSES][N_SH];
    #pragma unroll
    for (int s = 0; s < CH_PASSES; ++s) {
        const int c = lane + s * 32;
        #pragma unroll
        for (int l = 0; l <= LMAX; ++l) {
            const int nl = n_l.v[l];
            const int m0 = l * l;
            const int dm = 2 * l + 1;
            if (c < nl) {
                const float* src = dA1_raw.p[l] + (int64_t)i * dm * nl + c;
                #pragma unroll
                for (int m = 0; m < dm; ++m) gA1[s][m0 + m] = src[m * nl];
            } else {
                #pragma unroll
                for (int m = 0; m < dm; ++m) gA1[s][m0 + m] = 0.0f;
            }
        }
    }

    double rfx = 0.0, rfy = 0.0, rfz = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += 8) {
        const int bcnt = (ee - eb_b < 8) ? (ee - eb_b) : 8;
        if (lane < bcnt) {
            const int e = eb_b + lane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const int pair = z_i * S + __ldg(species + j);
            const float r = fmaxf(vl.w, 1e-8f);
            s_vl[wid][lane] = vl;
            s_j[wid][lane] = j;
            s_pair[wid][lane] = pair;
            eval_cheb<true>(r, r_max, n_base, s_g[wid][lane], s_dg[wid][lane]);
            float4 Yv[4];
            loup::edge_sh16(vl.x, vl.y, vl.z, r, Yv);
            unpack_sh16(Yv, LMAX, s_Y[wid][lane]);
        }
        __syncwarp();

        for (int bi = 0; bi < bcnt; ++bi) {
            const float* Cp = C1 + (int64_t)s_pair[wid][bi] * n_base * n_tot1;
            const float* gk = s_g[wid][bi];
            const float* dgk = s_dg[wid][bi];
            const float* Yk = s_Y[wid][bi];
            float er_lane = 0.0f;
            float dY_lane[N_SH];
            #pragma unroll
            for (int m = 0; m < N_SH; ++m) dY_lane[m] = 0.0f;

            #pragma unroll
            for (int s = 0; s < CH_PASSES; ++s) {
                const int c = lane + s * 32;
                #pragma unroll
                for (int l = 0; l <= LMAX; ++l) {
                    if (c < n_l.v[l]) {
                        float rv = 0.0f, drv = 0.0f;
                        const float* col = Cp + l1_off.v[l] + c;
                        for (int k = 0; k < n_base; ++k) {
                            const float cval = __ldg(col + k * n_tot1);
                            rv = fmaf(cval, gk[k], rv);
                            drv = fmaf(cval, dgk[k], drv);
                        }
                        const int m0 = l * l;
                        const int dm = 2 * l + 1;
                        float dr1 = 0.0f;
                        #pragma unroll
                        for (int m = 0; m < dm; ++m) {
                            const float ga = gA1[s][m0 + m];
                            dr1 = fmaf(ga, Yk[m0 + m], dr1);
                            dY_lane[m0 + m] = fmaf(ga, rv, dY_lane[m0 + m]);
                        }
                        er_lane = fmaf(dr1, drv, er_lane);
                    }
                }
            }

            const float er = warp_sum_f(er_lane);
            if (lane == 0) s_er[wid][bi] = er;
            #pragma unroll
            for (int m = 0; m < N_SH; ++m) {
                const float dy = warp_sum_f(dY_lane[m]);
                if (lane == 0) s_dY[wid][bi][m] = dy;
            }
        }
        __syncwarp();

        if (lane < bcnt) {
            const float4 vl = s_vl[wid][lane];
            const int j = s_j[wid][lane];
            const int pair = s_pair[wid][lane];
            const float r = fmaxf(vl.w, 1e-8f);
            float er = s_er[wid][lane];
            if (ZBL_SW) {
                const double rs = (double)__ldg(rs_pair + pair);
                if ((double)r < rs) {
                    er += wz_i * (float)eval_zbl_pair_deriv_d((double)r, rs, (double)__ldg(zz_pair + pair), (double)__ldg(au_pair + pair));
                }
            }
            float dY16[16];
            #pragma unroll
            for (int m = 0; m < 16; ++m) dY16[m] = (m < N_SH) ? s_dY[wid][lane][m] : 0.0f;
            const float inv_d = 1.0f / r;
            const float ux = vl.x * inv_d, uy = vl.y * inv_d, uz = vl.z * inv_d;
            float gx, gy, gz;
            ::sh16_grad(ux, uy, uz, dY16, gx, gy, gz);
            const float dot = ux * gx + uy * gy + uz * gz;
            const double fx = (double)(ux * er + (gx - ux * dot) * inv_d);
            const double fy = (double)(uy * er + (gy - uy * dot) * inv_d);
            const double fz = (double)(uz * er + (gz - uz * dot) * inv_d);
            rfx -= fx; rfy -= fy; rfz -= fz;
            atomicAdd(forces + (int64_t)j * 3 + 0, fx);
            atomicAdd(forces + (int64_t)j * 3 + 1, fy);
            atomicAdd(forces + (int64_t)j * 3 + 2, fz);
            if (STRESS) {
                v0 += (double)vl.x * fx;
                v1 += 0.5 * ((double)vl.x * fy + (double)vl.y * fx);
                v2 += 0.5 * ((double)vl.x * fz + (double)vl.z * fx);
                v3 += (double)vl.y * fy;
                v4 += 0.5 * ((double)vl.y * fz + (double)vl.z * fy);
                v5 += (double)vl.z * fz;
            }
        }
        __syncwarp();
    }

    rfx = warp_sum_d(rfx);
    rfy = warp_sum_d(rfy);
    rfz = warp_sum_d(rfz);
    if (STRESS) {
        v0 = warp_sum_d(v0); v1 = warp_sum_d(v1); v2 = warp_sum_d(v2);
        v3 = warp_sum_d(v3); v4 = warp_sum_d(v4); v5 = warp_sum_d(v5);
    }
    if (lane == 0) {
        if (rfx != 0.0 || rfy != 0.0 || rfz != 0.0) {
            atomicAdd(forces + (int64_t)i * 3 + 0, rfx);
            atomicAdd(forces + (int64_t)i * 3 + 1, rfy);
            atomicAdd(forces + (int64_t)i * 3 + 2, rfz);
        }
        if (STRESS) {
            if (v0 != 0.0) atomicAdd(virial + 0, v0);
            if (v1 != 0.0) atomicAdd(virial + 1, v1);
            if (v2 != 0.0) atomicAdd(virial + 2, v2);
            if (v3 != 0.0) atomicAdd(virial + 3, v3);
            if (v4 != 0.0) atomicAdd(virial + 4, v4);
            if (v5 != 0.0) atomicAdd(virial + 5, v5);
        }
    }
}

}  // namespace

// =================================================================================================
// IlmlipModel lifecycle
// =================================================================================================
IlmlipModel::~IlmlipModel() {
    free_system();
    free_model();
}

void IlmlipModel::free_model() {
    for (void* p : model_allocs_) cudaFree(p);
    model_allocs_.clear();
    d_C1_ = d_C1_zick_ = d_C2_ = d_Cbf_ = nullptr;
    d_rs_pair_ = d_zz_pair_ = d_au_pair_ = nullptr;
    d_rs_pair_f32_ = nullptr;
    max_rs_ = 0.0f;
    d_nu2_ = d_nu3_ = d_l2p_ = d_cg_idx_ = d_bisp_ = d_bisp_idx_ = d_triu_i_ = d_triu_j_ = nullptr;
    d_cg_val_ = d_bisp_val_ = d_mu_x_ = d_inv_std_x_ = d_alpha_Z_ = d_u_Z_ = d_W_B0_ = nullptr;
    for (int l = 0; l < kMaxL; ++l) d_W_msg_[l] = d_W_msg_T_[l] = d_W_GB_[l] = nullptr;
    for (int d = 0; d < kMaxDepth; ++d) {
        d_ro_W_[d] = d_ro_b_[d] = nullptr;
        d_ro_h2_B_[d] = d_ro_h2_BT_[d] = nullptr;
        d_ro_h2_pB_[d] = d_ro_h2_pBT_[d] = nullptr;
    }
    use_ro_h2_ = false;
}

void IlmlipModel::free_system() {
    for (void* p : sys_allocs_) cudaFree(p);
    sys_allocs_.clear();
    if (d_zero_slab_) cudaFree(d_zero_slab_);
    d_zero_slab_ = nullptr;
    zero_slab_bytes_ = 0;
    sys_bytes_ = 0;
    sys_max_alloc_ = 0;
    N_ = 0;
    for (int l = 0; l < kMaxL; ++l) {
        d_A1_[l] = d_dA1_[l] = d_Bcat_[l] = d_dBcat_[l] = d_I_[l] = d_dI_[l] = nullptr;
        d_A2_[l] = d_dA2_[l] = d_PQ_[l] = d_dPQ_[l] = nullptr;
    }
    d_dA2_pack_ = nullptr;
    d_I_pack4_ = nullptr;
    d_Q0_ = d_dQ0_ = nullptr;
    d_e_zbl_ = nullptr;
    d_zbl_flag_ = nullptr;
    d_eA_x_ = nullptr;
    d_f0_ = d_w_zbl_ = d_df0_bf_ = d_f1_ = d_x_ = d_dx_ = nullptr;
    for (int d = 0; d < kMaxDepth; ++d) d_pre_[d] = d_h_[d] = nullptr;
    d_dh_[0] = d_dh_[1] = nullptr;
}

void IlmlipModel::load(const Bundle& b) {
    free_system();
    free_model();
    if (b.meta("arch") != "ilmlip") throw std::runtime_error("not an ilmlip bundle");
    if (b.meta_int("exporter_version") != 1)
        throw std::runtime_error("ilmlip bundle exporter_version != 1");

    S_ = b.meta_int("S");
    Z_ = b.meta_ints("Z");
    if (S_ < 1 || S_ > 64 || (int)Z_.size() != S_) throw std::runtime_error("ilmlip bundle: bad S / Z");
    r_max_ = (float)b.f64("r_max", {1})[0];
    e0_ = b.f64("e0", {S_});

    lmax_ = b.meta_int("lmax");
    lfeat_ = b.meta_int("lfeat");
    lmsg_ = b.meta_int("lmsg");
    lfeat2_ = b.meta_int("lfeat2");
    lout2_ = b.meta_int("lout2");
    n_base_ = b.meta_int("n_base");
    n_msg_ = b.meta_int("n_msg");
    r_gram_ = b.meta_int("r_gram");
    r_bisp_ = b.meta_int("r_bisp");
    ro_depth_ = b.meta_int("readout_depth");
    ro_width_ = b.meta_int("readout_width");
    remove_backflow_ = b.meta_int("remove_backflow");
    use_zbl_switch_ = b.meta_int("use_zbl_switch");
    n_nu2_ = b.meta_int("n_nu2");
    n_nu3_ = b.meta_int("n_nu3");
    n_l2_paths_ = b.meta_int("n_l2_paths");
    n_bisp_ = b.meta_int("n_bisp");
    n_inv1_ = b.meta_int("n_inv1");
    n_inv2_ = b.meta_int("n_inv2");
    n_feat_ = b.meta_int("n_feat");
    D_A1_ = b.meta_int("D_A1");
    D_A2_ = b.meta_int("D_A2");

    if (lmax_ < 1 || lmax_ > 3 || lfeat_ > 2 || lmsg_ > 2 || lfeat2_ > 2 || lout2_ > 3 ||
        n_base_ < 1 || n_base_ > 32 || n_msg_ < 1 || n_msg_ > 64 || ro_depth_ < 1 || ro_depth_ > kMaxDepth) {
        throw std::runtime_error("ilmlip bundle: unsupported dimensions");
    }

    auto nl_v = b.meta_ints("n_l");
    if ((int)nl_v.size() != lmax_ + 1) throw std::runtime_error("ilmlip bundle: n_l mismatch");
    n_tot1_ = 0;
    max_nl_ = 0;
    for (int l = 0; l < kMaxL; ++l) {
        n_l_[l] = (l <= lmax_) ? nl_v[l] : 0;
        l1_off_[l] = n_tot1_;
        n_tot1_ += n_l_[l];
        max_nl_ = std::max(max_nl_, n_l_[l]);
    }
    if (max_nl_ > 64) throw std::runtime_error("ilmlip bundle: max(n_l) > 64 not supported");

    auto mid_v = b.meta_ints("msg_in_dim");
    if ((int)mid_v.size() != lmsg_ + 1) throw std::runtime_error("ilmlip bundle: msg_in_dim mismatch");
    for (int l = 0; l < kMaxL; ++l) msg_in_dim_[l] = (l <= lmsg_) ? mid_v[l] : 0;

    auto sc = b.f64("scalars", {6});
    sigma0_ = (float)sc[0];
    sigma1_ = (float)sc[1];
    a0_ = (float)sc[2];
    kappa_ = (float)sc[3];
    // energy_scale = sc[4] (already folded into u_Z at export)
    z_c_ = (float)sc[5];

    const int P2 = S_ * S_;
    auto read_pair_f64 = [&](const std::string& name) -> std::vector<double> {
        if (b.entry(name).dtype == DType::F64) return b.f64(name, {P2});
        auto f = b.f32(name, {P2});
        return std::vector<double>(f.begin(), f.end());
    };
    auto rs_h = read_pair_f64("rs_pair");
    std::vector<float> rs_f32(P2);
    max_rs_ = 0.0f;
    for (int p = 0; p < P2; ++p) {
        rs_f32[p] = (float)rs_h[p] + 1e-3f;
        max_rs_ = std::max(max_rs_, rs_f32[p]);
    }
    d_rs_pair_ = upload_vec(model_allocs_, rs_h);
    d_rs_pair_f32_ = upload_vec(model_allocs_, rs_f32);
    d_zz_pair_ = upload_vec(model_allocs_, read_pair_f64("zz_pair"));
    d_au_pair_ = upload_vec(model_allocs_, read_pair_f64("au_pair"));

    // Transpose radial tables [P2, C, n_base] -> [P2, n_base, C] so channel c is coalesced across warp lanes!
    auto upload_radial_T = [&](const std::string& name, int C) -> float* {
        auto src = b.f32(name, {P2, C, n_base_});
        std::vector<float> dst((size_t)P2 * n_base_ * C);
        for (int p = 0; p < P2; ++p)
            for (int c = 0; c < C; ++c)
                for (int k = 0; k < n_base_; ++k)
                    dst[((size_t)p * n_base_ + k) * C + c] = src[((size_t)p * C + c) * n_base_ + k];
        return upload_vec(model_allocs_, dst);
    };
    d_C1_ = upload_radial_T("C1_eff", n_tot1_);
    {
        // Also build [z_i, c, z_j * n_base + k] layout for zero-shuffle coalesced Phase 1 in L1 edge bwd!
        auto c1_src = b.f32("C1_eff", {P2, n_tot1_, n_base_});
        const int snb = S_ * n_base_;
        std::vector<float> c1_zick((size_t)S_ * n_tot1_ * snb);
        for (int zi = 0; zi < S_; ++zi)
            for (int zj = 0; zj < S_; ++zj)
                for (int c = 0; c < n_tot1_; ++c)
                    for (int k = 0; k < n_base_; ++k)
                        c1_zick[((size_t)zi * n_tot1_ + c) * snb + zj * n_base_ + k] =
                            c1_src[((size_t)(zi * S_ + zj) * n_tot1_ + c) * n_base_ + k];
        d_C1_zick_ = upload_vec(model_allocs_, c1_zick);
    }
    d_C2_ = upload_radial_T("C2_eff", (lfeat2_ + 1) * n_msg_);
    if (remove_backflow_) d_Cbf_ = upload_radial_T("C_bf", (lmsg_ + 1) * n_msg_);

    // Build enriched nu2 / nu3 tables with precomputed b2_off, x1_off, bcat_off
    auto nu2_raw = b.i32("nu2_table", {n_nu2_, 6});
    std::vector<int32_t> nu2_ext((size_t)n_nu2_ * 9, 0);
    int b2_off_cur = 0;
    int x1_off_cur = n_l_[0];
    int bcat_cur[kMaxL] = {n_l_[0], n_l_[1], n_l_[2], n_l_[3]};
    for (int p = 0; p < n_nu2_; ++p) {
        const int l1 = nu2_raw[p * 6 + 0];
        const int l2 = nu2_raw[p * 6 + 1];
        const int L = nu2_raw[p * 6 + 2];
        const int c = nu2_raw[p * 6 + 3];
        nu2_ext[p * 9 + 0] = l1;
        nu2_ext[p * 9 + 1] = l2;
        nu2_ext[p * 9 + 2] = L;
        nu2_ext[p * 9 + 3] = c;
        nu2_ext[p * 9 + 4] = nu2_raw[p * 6 + 4];
        nu2_ext[p * 9 + 5] = nu2_raw[p * 6 + 5];
        nu2_ext[p * 9 + 6] = b2_off_cur;
        b2_off_cur += 2 * L + 1;
        if (L == 0) {
            nu2_ext[p * 9 + 7] = x1_off_cur;
            x1_off_cur += c;
        }
        if (L <= lmsg_) {
            nu2_ext[p * 9 + 8] = bcat_cur[L];
            bcat_cur[L] += c;
        }
    }
    b2_scalars_ = b2_off_cur;

    auto nu3_raw = n_nu3_ > 0 ? b.i32("nu3_table", {n_nu3_, 6}) : std::vector<int32_t>();
    std::vector<int32_t> nu3_ext((size_t)n_nu3_ * 8, 0);
    for (int q = 0; q < n_nu3_; ++q) {
        const int p2_idx = nu3_raw[q * 6 + 0];
        const int l3 = nu3_raw[q * 6 + 1];
        const int L = nu3_raw[q * 6 + 2];
        const int c = nu3_raw[q * 6 + 3];
        nu3_ext[q * 8 + 0] = nu2_ext[p2_idx * 9 + 6];  // pre-resolved b2_off!
        nu3_ext[q * 8 + 1] = l3;
        nu3_ext[q * 8 + 2] = L;
        nu3_ext[q * 8 + 3] = c;
        nu3_ext[q * 8 + 4] = nu3_raw[q * 6 + 4];
        nu3_ext[q * 8 + 5] = nu3_raw[q * 6 + 5];
        if (L == 0) {
            nu3_ext[q * 8 + 6] = x1_off_cur;
            x1_off_cur += c;
        }
        if (L <= lmsg_) {
            nu3_ext[q * 8 + 7] = bcat_cur[L];
            bcat_cur[L] += c;
        }
    }
    if (x1_off_cur != n_inv1_) throw std::runtime_error("ilmlip bundle: n_inv1 mismatch");
    for (int l = 0; l <= lmsg_; ++l)
        if (bcat_cur[l] != msg_in_dim_[l]) throw std::runtime_error("ilmlip bundle: msg_in_dim mismatch");

    d_nu2_ = upload_vec(model_allocs_, nu2_ext);
    d_nu3_ = upload_vec(model_allocs_, nu3_ext);

    // Verify canonical (lfeat==1, lmsg==1, lmax in {2,3}) nu2/nu3 layout for zero-spill specialized kernels
    fast_l1_node_lf1_ = false;
    if (lfeat_ == 1 && lmsg_ == 1 && (lmax_ == 2 || lmax_ == 3)) {
        const int exp_nu2_cnt = (lmax_ == 2) ? 5 : 7;
        const int exp_nu3_cnt = (lmax_ == 2) ? 8 : 12;
        static const int exp_nu2[7][3] = {{0,0,0}, {0,1,1}, {1,1,0}, {1,2,1}, {2,2,0}, {2,3,1}, {3,3,0}};
        static const int exp_nu3_l2[8][3]  = {{0,0,0}, {1,1,0}, {3,1,0}, {4,0,0}, {0,1,1}, {2,1,1}, {3,0,1}, {4,1,1}};
        static const int exp_nu3_l3[12][3] = {{0,0,0}, {1,1,0}, {3,1,0}, {4,0,0}, {5,1,0}, {6,0,0},
                                              {0,1,1}, {2,1,1}, {3,0,1}, {4,1,1}, {5,0,1}, {6,1,1}};
        if (n_nu2_ == exp_nu2_cnt && n_nu3_ == exp_nu3_cnt) {
            bool ok = true;
            for (int p = 0; p < exp_nu2_cnt; ++p) {
                if (nu2_raw[p * 6 + 0] != exp_nu2[p][0] || nu2_raw[p * 6 + 1] != exp_nu2[p][1] ||
                    nu2_raw[p * 6 + 2] != exp_nu2[p][2]) ok = false;
            }
            const auto* e3 = (lmax_ == 2) ? exp_nu3_l2 : exp_nu3_l3;
            for (int q = 0; q < exp_nu3_cnt; ++q) {
                if (nu3_raw[q * 6 + 0] != e3[q][0] || nu3_raw[q * 6 + 1] != e3[q][1] ||
                    nu3_raw[q * 6 + 2] != e3[q][2]) ok = false;
            }
            fast_l1_node_lf1_ = ok;
        }
    }

    auto l2p_raw = b.i32("l2_paths", {n_l2_paths_, 6});
    std::vector<int32_t> l2p_ext((size_t)n_l2_paths_ * 7, 0);
    for (int L = 0; L < kMaxL; ++L) l2_npaths_[L] = 0;
    int reg_off_cur = 0;
    for (int p = 0; p < n_l2_paths_; ++p) {
        const int L = l2p_raw[p * 6 + 2];
        l2p_ext[p * 7 + 0] = l2p_raw[p * 6 + 0];
        l2p_ext[p * 7 + 1] = l2p_raw[p * 6 + 1];
        l2p_ext[p * 7 + 2] = L;
        l2p_ext[p * 7 + 3] = l2p_raw[p * 6 + 3];
        l2p_ext[p * 7 + 4] = l2p_raw[p * 6 + 4];
        l2p_ext[p * 7 + 5] = l2p_raw[p * 6 + 5];
        l2p_ext[p * 7 + 6] = reg_off_cur;
        reg_off_cur += 2 * L + 1;
        l2_npaths_[L] = std::max(l2_npaths_[L], l2p_raw[p * 6 + 3] + 1);
    }
    total_l2_out_ = reg_off_cur;
    if (total_l2_out_ > 48) throw std::runtime_error("ilmlip bundle: total_l2_out > 48");
    for (int L = 0; L < kMaxL; ++L) l2_c_[L] = l2_npaths_[L] * n_msg_;
    d_l2p_ = upload_vec(model_allocs_, l2p_ext);

    auto cg_i = b.i32("cg_idx");
    cg_count_ = (int)cg_i.size();
    d_cg_idx_ = upload_vec(model_allocs_, cg_i);
    d_cg_val_ = upload_vec(model_allocs_, b.f32("cg_val"));

    // Layer-2 invariant offsets in x[N, n_feat]
    int x_cur = n_inv1_ + l2_c_[0];
    const int n_tri = (r_gram_ * (r_gram_ + 1)) >> 1;
    for (int L = 0; L < kMaxL; ++L) gram_off_[L] = 0;
    if (r_gram_ > 0) {
        std::vector<int32_t> ti, tj;
        ti.reserve(n_tri);
        tj.reserve(n_tri);
        for (int a = 0; a < r_gram_; ++a)
            for (int bb = a; bb < r_gram_; ++bb) { ti.push_back(a); tj.push_back(bb); }
        d_triu_i_ = upload_vec(model_allocs_, ti);
        d_triu_j_ = upload_vec(model_allocs_, tj);
        for (int L = 1; L <= lout2_; ++L) {
            if (l2_c_[L] == 0) continue;
            gram_off_[L] = x_cur;
            x_cur += n_tri;
        }
    }
    bisp_off_ = x_cur;
    bisp_idx_count_ = 0;
    fast_bisp_l2_ = false;
    if (r_bisp_ > 0 && n_bisp_ > 0) {
        auto bt = b.i32("bisp_table", {n_bisp_, 5});
        d_bisp_ = upload_vec(model_allocs_, bt);
        auto bi = b.i32("bisp_idx");
        bisp_idx_count_ = (int)bi.size();
        d_bisp_idx_ = upload_vec(model_allocs_, bi);
        d_bisp_val_ = upload_vec(model_allocs_, b.f32("bisp_val"));
        x_cur += n_bisp_ * r_bisp_;
        if (lout2_ == 2 && n_bisp_ == 4 && l2_c_[0] > 0 && l2_c_[1] > 0 && l2_c_[2] > 0) {
            static const int exp_b[4][3] = {{0,1,1}, {0,2,2}, {1,1,2}, {2,2,2}};
            bool ok = true;
            for (int t = 0; t < 4; ++t) {
                if (bt[t * 5 + 0] != exp_b[t][0] || bt[t * 5 + 1] != exp_b[t][1] || bt[t * 5 + 2] != exp_b[t][2])
                    ok = false;
            }
            fast_bisp_l2_ = ok;
        }
    } else if (r_bisp_ == 0 && lout2_ == 2) {
        fast_bisp_l2_ = true;
    }
    if (x_cur != n_feat_) throw std::runtime_error("ilmlip bundle: n_feat layout mismatch");

    // Node weights (merge W_G[L] and W_B[L] along columns into W_GB[L] for L>=1!)
    for (int l = 0; l <= lmsg_; ++l) {
        auto wm = b.f32("W_msg_" + std::to_string(l), {msg_in_dim_[l], n_msg_});
        d_W_msg_[l] = upload_vec(model_allocs_, wm);
        std::vector<float> wm_T((size_t)n_msg_ * msg_in_dim_[l]);
        for (int k = 0; k < msg_in_dim_[l]; ++k)
            for (int q = 0; q < n_msg_; ++q)
                wm_T[(size_t)q * msg_in_dim_[l] + k] = wm[(size_t)k * n_msg_ + q];
        d_W_msg_T_[l] = upload_vec(model_allocs_, wm_T);
    }
    r_gb_ = r_gram_ + r_bisp_;
    if (l2_c_[0] > 0 && r_bisp_ > 0) {
        d_W_B0_ = upload_vec(model_allocs_, b.f32("W_B_0", {l2_c_[0], r_bisp_}));
    }
    for (int L = 1; L <= lout2_; ++L) {
        if (l2_c_[L] == 0 || r_gb_ == 0) continue;
        const int K = l2_c_[L];
        std::vector<float> wg = (r_gram_ > 0) ? b.f32("W_G_" + std::to_string(L), {K, r_gram_}) : std::vector<float>();
        std::vector<float> wb = (r_bisp_ > 0) ? b.f32("W_B_" + std::to_string(L), {K, r_bisp_}) : std::vector<float>();
        std::vector<float> wgb((size_t)K * r_gb_);
        for (int k = 0; k < K; ++k) {
            for (int c = 0; c < r_gram_; ++c) wgb[(size_t)k * r_gb_ + c] = wg[(size_t)k * r_gram_ + c];
            for (int c = 0; c < r_bisp_; ++c) wgb[(size_t)k * r_gb_ + r_gram_ + c] = wb[(size_t)k * r_bisp_ + c];
        }
        d_W_GB_[L] = upload_vec(model_allocs_, wgb);
    }
    d_mu_x_ = upload_vec(model_allocs_, b.f32("mu_x", {n_feat_}));
    d_inv_std_x_ = upload_vec(model_allocs_, b.f32("inv_std_x", {n_feat_}));

    // Pad n_feat_ to multiple of 128 (or 32) when readout width qualifies for hgemm2_f32a_row FP16x2 TC GEMMs!
    // (For ro_width == 128, n_feat_pad_ % 128 == 0 lets BOTH ro_W_0 bwd and ro_W_1 fwd/bwd hit h2f_64x128_k128_fused!)
    const char* h2_env = std::getenv("LOUP_ILMLIP_H2");
    const bool h2_enabled = !(h2_env && h2_env[0] == '0');
    const bool h2_shape_ok = h2_enabled && (ro_width_ >= 64) && ((ro_width_ & 15) == 0);
    if (h2_shape_ok) {
        const int align_k = (ro_width_ == 128) ? 128 : 32;
        n_feat_pad_ = ((n_feat_ + align_k - 1) / align_k) * align_k;
    } else {
        n_feat_pad_ = n_feat_;
    }

    for (int d = 0; d < ro_depth_; ++d) {
        const int in_d = (d == 0) ? n_feat_ : ro_width_;
        auto w = b.f32("ro_W_" + std::to_string(d), {in_d, ro_width_});
        if (d == 0 && n_feat_pad_ > n_feat_) {
            w.resize((size_t)n_feat_pad_ * ro_width_, 0.0f);
        }
        d_ro_W_[d] = upload_vec(model_allocs_, w);
        if (d > 0) d_ro_b_[d] = upload_vec(model_allocs_, b.f32("ro_b_" + std::to_string(d), {ro_width_}));
    }

    use_ro_h2_ = false;
    if (h2_shape_ok) {
        int dev = 0, maj = 0, min = 0;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
            ((maj == 12 && min == 0) || (maj == 8 && (min == 0 || min == 6 || min == 9)))) {
            for (int d = 0; d < ro_depth_; ++d) {
                const int rows_w = (d == 0) ? n_feat_pad_ : ro_width_;
                const int cols_w = ro_width_;
                CUDA_CHECK_THROW(cudaMalloc(&d_ro_h2_B_[d], 2 * (size_t)hgemm2_pad_elems(cols_w, rows_w) * 2));
                CUDA_CHECK_THROW(cudaMalloc(&d_ro_h2_pB_[d], (size_t)cols_w * sizeof(float)));
                model_allocs_.push_back(d_ro_h2_B_[d]);
                model_allocs_.push_back(d_ro_h2_pB_[d]);
                hgemm2_pack_b_col(cols_w, rows_w, d_ro_W_[d], true, d_ro_h2_B_[d], d_ro_h2_pB_[d], nullptr);

                CUDA_CHECK_THROW(cudaMalloc(&d_ro_h2_BT_[d], 2 * (size_t)hgemm2_pad_elems(rows_w, cols_w) * 2));
                CUDA_CHECK_THROW(cudaMalloc(&d_ro_h2_pBT_[d], (size_t)rows_w * sizeof(float)));
                model_allocs_.push_back(d_ro_h2_BT_[d]);
                model_allocs_.push_back(d_ro_h2_pBT_[d]);
                hgemm2_pack_b_col(rows_w, cols_w, d_ro_W_[d], false, d_ro_h2_BT_[d], d_ro_h2_pBT_[d], nullptr);
            }
            CUDA_CHECK_THROW(cudaStreamSynchronize(nullptr));
            hgemm2_row_init();
            use_ro_h2_ = true;
        }
    }

    d_alpha_Z_ = upload_vec(model_allocs_, b.f32("alpha_Z", {S_, ro_width_}));
    d_u_Z_ = upload_vec(model_allocs_, b.f32("u_Z", {S_, ro_width_}));
}

std::string IlmlipModel::kernel_signature() const {
    return "ilmlip_l" + std::to_string(lmax_) + "_f" + std::to_string(lfeat_) + "_m" + std::to_string(n_msg_) +
           "_b" + std::to_string(n_base_);
}

void IlmlipModel::alloc_system(int N) {
    free_system();
    const size_t n = (size_t)N;
    auto alloc = [&](size_t floats) -> float* {
        if (floats == 0) return nullptr;
        float* p = nullptr;
        const size_t bytes = floats * sizeof(float);
        sys_max_alloc_ = std::max(sys_max_alloc_, bytes);
        CUDA_CHECK_THROW(device_malloc(&p, bytes));
        sys_bytes_ += bytes;
        sys_allocs_.push_back(p);
        return p;
    };
    auto align256 = [](size_t b) { return (b + 255) & ~size_t(255); };

    // Contiguous zero-initialized slab for atomic backward accumulators (dI_node + df0_bf)
    size_t zbytes = align256(n * sizeof(float));
    for (int l = 0; l <= lmsg_; ++l) zbytes += align256(n * (2 * l + 1) * n_msg_ * sizeof(float));
    CUDA_CHECK_THROW(device_malloc(&d_zero_slab_, zbytes));
    zero_slab_bytes_ = zbytes;
    sys_bytes_ += zbytes;
    sys_max_alloc_ = std::max(sys_max_alloc_, zbytes);
    char* zb = static_cast<char*>(d_zero_slab_);
    size_t zoff = 0;
    auto zalloc = [&](size_t floats) -> float* {
        float* p = reinterpret_cast<float*>(zb + zoff);
        zoff += align256(floats * sizeof(float));
        return p;
    };
    d_df0_bf_ = zalloc(n);
    for (int l = 0; l <= lmsg_; ++l) d_dI_[l] = zalloc(n * (2 * l + 1) * n_msg_);

    for (int l = 0; l <= lmax_; ++l) {
        d_A1_[l] = alloc(n * (2 * l + 1) * n_l_[l]);
        d_dA1_[l] = alloc(n * (2 * l + 1) * n_l_[l]);
    }
    d_f0_ = alloc(n);
    if (use_zbl_switch_) {
        double* ez = nullptr;
        const size_t ez_bytes = n * sizeof(double);
        sys_max_alloc_ = std::max(sys_max_alloc_, ez_bytes);
        CUDA_CHECK_THROW(device_malloc(&ez, ez_bytes));
        sys_bytes_ += ez_bytes;
        sys_allocs_.push_back(ez);
        d_e_zbl_ = ez;
        d_w_zbl_ = alloc(n);
        int* zf = nullptr;
        CUDA_CHECK_THROW(device_malloc(&zf, 256));
        sys_bytes_ += 256;
        sys_allocs_.push_back(zf);
        d_zbl_flag_ = zf;
    }

    for (int l = 0; l <= lmsg_; ++l) {
        d_Bcat_[l] = alloc(n * (2 * l + 1) * msg_in_dim_[l]);
        d_dBcat_[l] = alloc(n * (2 * l + 1) * msg_in_dim_[l]);
        d_I_[l] = alloc(n * (2 * l + 1) * n_msg_);
    }

    for (int L = 0; L <= lout2_; ++L) {
        if (l2_c_[L] == 0) continue;
        d_A2_[L] = alloc(n * (2 * L + 1) * l2_c_[L]);
        d_dA2_[L] = alloc(n * (2 * L + 1) * l2_c_[L]);
        if (L == 0 && r_bisp_ > 0) {
            d_Q0_ = alloc(n * r_bisp_);
            d_dQ0_ = alloc(n * r_bisp_);
        }
        if (L >= 1 && r_gb_ > 0) {
            d_PQ_[L] = alloc(n * (2 * L + 1) * r_gb_);
            d_dPQ_[L] = alloc(n * (2 * L + 1) * r_gb_);
        }
    }
    if (lmsg_ == 1 && lout2_ == 2 && lfeat2_ == 1 && n_msg_ <= 16 && n_base_ == 8) {
        d_dA2_pack_ = alloc(n * 208);  // 13*16 floats/atom (3x float4 + 1x float, exact FP32, 16-byte aligned!)
        float4* ip4 = nullptr;
        const size_t ip4_bytes = n * (size_t)n_msg_ * sizeof(float4);
        sys_max_alloc_ = std::max(sys_max_alloc_, ip4_bytes);
        CUDA_CHECK_THROW(device_malloc(&ip4, ip4_bytes));
        sys_bytes_ += ip4_bytes;
        sys_allocs_.push_back(ip4);
        d_I_pack4_ = ip4;
    }
    d_f1_ = alloc(n);

    d_x_ = alloc(n * n_feat_pad_);
    d_dx_ = alloc(n * n_feat_pad_);
    {
        int8_t* ea = nullptr;
        const size_t ea_bytes = align256(n * sizeof(int8_t));
        CUDA_CHECK_THROW(device_malloc(&ea, ea_bytes));
        sys_bytes_ += ea_bytes;
        sys_allocs_.push_back(ea);
        d_eA_x_ = ea;
    }
    if (n_feat_pad_ > n_feat_) {
        CUDA_CHECK_THROW(cudaMemset(d_x_, 0, n * n_feat_pad_ * sizeof(float)));
        CUDA_CHECK_THROW(cudaMemset(d_dx_, 0, n * n_feat_pad_ * sizeof(float)));
    }
    for (int d = 0; d < ro_depth_; ++d) {
        d_pre_[d] = alloc(n * ro_width_);
        d_h_[d] = alloc(n * ro_width_);
    }
    d_dh_[0] = alloc(n * ro_width_);
    d_dh_[1] = alloc(n * ro_width_);
    N_ = N;
}

void IlmlipModel::record(const GenericPassInputs& in) {
    const int N = in.N;
    cudaStream_t st = in.stream;
    const unsigned warp_grid = (unsigned)((N + 3) / 4);
    const unsigned half_grid = (unsigned)((N + 7) / 8);
    const float inv_rmax = r_max_;  // pass exact r_max into eval_cheb

    Int4 nl4{{n_l_[0], n_l_[1], n_l_[2], n_l_[3]}};
    Int4 l1off4{{l1_off_[0], l1_off_[1], l1_off_[2], l1_off_[3]}};
    Int4 mid4{{msg_in_dim_[0], msg_in_dim_[1], msg_in_dim_[2], msg_in_dim_[3]}};
    Int4 l2c4{{l2_c_[0], l2_c_[1], l2_c_[2], l2_c_[3]}};
    Int4 goff4{{gram_off_[0], gram_off_[1], gram_off_[2], gram_off_[3]}};

    Ptr4 pA1{{d_A1_[0], d_A1_[1], d_A1_[2], d_A1_[3]}};
    CPtr4 cA1{{d_A1_[0], d_A1_[1], d_A1_[2], d_A1_[3]}};
    Ptr4 pdA1{{d_dA1_[0], d_dA1_[1], d_dA1_[2], d_dA1_[3]}};
    CPtr4 cdA1{{d_dA1_[0], d_dA1_[1], d_dA1_[2], d_dA1_[3]}};
    Ptr4 pBcat{{d_Bcat_[0], d_Bcat_[1], d_Bcat_[2], d_Bcat_[3]}};
    CPtr4 cdBcat{{d_dBcat_[0], d_dBcat_[1], d_dBcat_[2], d_dBcat_[3]}};
    CPtr4 cI{{d_I_[0], d_I_[1], d_I_[2], d_I_[3]}};
    Ptr4 pdI{{d_dI_[0], d_dI_[1], d_dI_[2], d_dI_[3]}};
    Ptr4 pA2{{d_A2_[0], d_A2_[1], d_A2_[2], d_A2_[3]}};
    CPtr4 cA2{{d_A2_[0], d_A2_[1], d_A2_[2], d_A2_[3]}};
    Ptr4 pdA2{{d_dA2_[0], d_dA2_[1], d_dA2_[2], d_dA2_[3]}};
    CPtr4 cdA2{{d_dA2_[0], d_dA2_[1], d_dA2_[2], d_dA2_[3]}};
    CPtr4 cPQ{{nullptr, d_PQ_[1], d_PQ_[2], d_PQ_[3]}};
    Ptr4 pdPQ{{nullptr, d_dPQ_[1], d_dPQ_[2], d_dPQ_[3]}};

    const bool use_nb16 = (n_base_ <= 16);
    const bool use_fast_l2 = (lmsg_ == 1 && lout2_ == 2 && (lfeat2_ == 1 || lfeat2_ == 2));
    const bool use_l2_sp = (lmsg_ == 1 && lout2_ == 2 && lfeat2_ == 1 && n_msg_ <= 16 && n_base_ == 8 && S_ <= 4);
    const unsigned l2_sp_grid = (unsigned)((N + 15) / 16);
    const size_t l2_sp_smem = (size_t)S_ * S_ * 8 * 16 * sizeof(float2) * (remove_backflow_ ? 2 : 1);
    const bool use_l1_sumswap = (S_ <= 4 && (n_base_ == 8 || n_base_ == 12 || n_base_ == 16) && max_nl_ <= 32 &&
                                 (lmax_ == 2 || lmax_ == 3));
    const bool use_l1_packed = (n_l_[0] <= 32 && n_l_[1] <= 16 && n_l_[2] <= 16 && n_l_[3] <= 8);
    const bool l2_is_16 = (n_l_[2] > 8);
    const size_t c1_bytes = (size_t)S_ * S_ * n_base_ * n_tot1_ * sizeof(float);
    const bool use_smem_c1_fwd = (c1_bytes <= 32768);
    const size_t l2_c_bytes = (size_t)S_ * S_ * n_base_ * ((lfeat2_ + 1) + (remove_backflow_ ? 2 : 0)) * n_msg_ * sizeof(float);
    const bool use_smem_c2 = (l2_c_bytes <= 32768);

    // 1. Layer-1 Edge Forward (ZBL decoupled via d_zbl_flag_ on sumswap path -> <=80 regs!)
    const float a0_s0 = a0_ / sigma0_;
    const float inv_d1_s0k = 1.0f / ((float)D_A1_ * sigma0_ * sigma0_ * kappa_ * kappa_);
    if (use_l1_sumswap) {
        if (use_zbl_switch_) CUDA_CHECK_THROW(cudaMemsetAsync(d_zbl_flag_, 0, sizeof(int), st));
        auto launch_l1_ss = [&](auto lm_tag, auto nb_tag, auto s3_tag) {
            constexpr int LM = decltype(lm_tag)::value;
            constexpr int NB = decltype(nb_tag)::value;
            constexpr bool S3 = decltype(s3_tag)::value;
            ilmlip_l1_edge_fwd_sumswap_kernel<LM, NB, S3><<<warp_grid, 128, 0, st>>>(
                N, S_, n_tot1_, nl4, l1off4, inv_rmax, a0_s0, inv_d1_s0k,
                in.act_row_ptr, in.act_senders, in.act_vec_len, in.species, d_C1_,
                use_zbl_switch_ ? d_rs_pair_f32_ : nullptr, use_zbl_switch_ ? d_zbl_flag_ : nullptr,
                pA1, d_f0_, use_zbl_switch_ ? d_e_zbl_ : nullptr);
        };
        auto disp_s3 = [&](auto lm_tag, auto nb_tag) {
            if (S_ <= 3) launch_l1_ss(lm_tag, nb_tag, std::true_type{});
            else         launch_l1_ss(lm_tag, nb_tag, std::false_type{});
        };
        auto disp_nb = [&](auto lm_tag) {
            if (n_base_ == 8)       disp_s3(lm_tag, std::integral_constant<int, 8>{});
            else if (n_base_ == 12) disp_s3(lm_tag, std::integral_constant<int, 12>{});
            else                    disp_s3(lm_tag, std::integral_constant<int, 16>{});
        };
        if (lmax_ == 2) disp_nb(std::integral_constant<int, 2>{});
        else            disp_nb(std::integral_constant<int, 3>{});
        if (use_zbl_switch_) {
            ilmlip_zbl_fwd_kernel<<<warp_grid, 128, 0, st>>>(
                N, S_, d_zbl_flag_, in.act_row_ptr, in.act_senders, in.act_vec_len, in.species, in.pos64,
                d_rs_pair_, d_zz_pair_, d_au_pair_, d_e_zbl_);
        }
    } else if (use_l1_packed) {
        auto launch_l1_pk = [&](auto lm_tag, auto nb_tag, auto zb_tag, auto sm_tag, auto l216_tag) {
            constexpr int LM = decltype(lm_tag)::value;
            constexpr int NB = decltype(nb_tag)::value;
            constexpr bool ZB = decltype(zb_tag)::value;
            constexpr bool SM = decltype(sm_tag)::value;
            constexpr bool L216 = decltype(l216_tag)::value;
            ilmlip_l1_edge_fwd_packed_kernel<LM, NB, ZB, SM, L216><<<warp_grid, 128, SM ? c1_bytes : 0, st>>>(
                N, S_, n_base_, n_tot1_, nl4, l1off4, inv_rmax, a0_s0, inv_d1_s0k,
                in.act_row_ptr, in.act_senders, in.act_vec_len, in.species, in.pos64, d_C1_,
                d_rs_pair_, d_zz_pair_, d_au_pair_, pA1, d_f0_, d_e_zbl_);
        };
        auto disp_l216 = [&](auto lm_tag, auto nb_tag, auto zb_tag, auto sm_tag) {
            if (l2_is_16) launch_l1_pk(lm_tag, nb_tag, zb_tag, sm_tag, std::true_type{});
            else          launch_l1_pk(lm_tag, nb_tag, zb_tag, sm_tag, std::false_type{});
        };
        auto disp_sm = [&](auto lm_tag, auto nb_tag, auto zb_tag) {
            if (use_smem_c1_fwd) disp_l216(lm_tag, nb_tag, zb_tag, std::true_type{});
            else                 disp_l216(lm_tag, nb_tag, zb_tag, std::false_type{});
        };
        auto disp_zb = [&](auto lm_tag, auto nb_tag) {
            if (use_zbl_switch_) disp_sm(lm_tag, nb_tag, std::true_type{});
            else                 disp_sm(lm_tag, nb_tag, std::false_type{});
        };
        auto disp_nb = [&](auto lm_tag) {
            if (use_nb16) disp_zb(lm_tag, std::integral_constant<int, 16>{});
            else          disp_zb(lm_tag, std::integral_constant<int, 32>{});
        };
        if (lmax_ <= 2) disp_nb(std::integral_constant<int, 2>{});
        else            disp_nb(std::integral_constant<int, 3>{});
    } else {
        auto launch_l1_fwd = [&](auto lm_tag, auto ch_tag, auto nb_tag, auto zbl_tag) {
            constexpr int LM = decltype(lm_tag)::value;
            constexpr int CP = decltype(ch_tag)::value;
            constexpr int NB = decltype(nb_tag)::value;
            constexpr bool ZB = decltype(zbl_tag)::value;
            ilmlip_l1_edge_fwd_kernel<LM, CP, NB, ZB><<<warp_grid, 128, 0, st>>>(
                N, S_, n_base_, n_tot1_, nl4, l1off4, inv_rmax, a0_s0, inv_d1_s0k,
                in.act_row_ptr, in.act_senders, in.act_vec_len, in.species, in.pos64, d_C1_,
                d_rs_pair_, d_zz_pair_, d_au_pair_, pA1, d_f0_, d_e_zbl_);
        };
        auto disp_l1_fwd_zb = [&](auto lm_tag, auto ch_tag, auto nb_tag) {
            if (use_zbl_switch_) launch_l1_fwd(lm_tag, ch_tag, nb_tag, std::true_type{});
            else                 launch_l1_fwd(lm_tag, ch_tag, nb_tag, std::false_type{});
        };
        auto disp_l1_fwd_nb = [&](auto lm_tag, auto ch_tag) {
            if (use_nb16) disp_l1_fwd_zb(lm_tag, ch_tag, std::integral_constant<int, 16>{});
            else          disp_l1_fwd_zb(lm_tag, ch_tag, std::integral_constant<int, 32>{});
        };
        auto disp_l1_fwd_ch = [&](auto lm_tag) {
            if (max_nl_ <= 32) disp_l1_fwd_nb(lm_tag, std::integral_constant<int, 1>{});
            else               disp_l1_fwd_nb(lm_tag, std::integral_constant<int, 2>{});
        };
        if (lmax_ <= 2) disp_l1_fwd_ch(std::integral_constant<int, 2>{});
        else            disp_l1_fwd_ch(std::integral_constant<int, 3>{});
    }

    // 2. Layer-1 Node Forward (ACE nu=2,3) + W_msg GEMMs (SMEM-staged 256-thr fused + float4 I_pack4!)
    const size_t l1n_tbl_bytes = (size_t)(n_nu2_ * 9 + n_nu3_ * 8 + cg_count_ * 2) * sizeof(int32_t);
    const bool l1n_smem_tbl = (l1n_tbl_bytes <= 16384);
    const int n_sh_l1 = (lmax_ <= 2) ? 9 : 16;
    const size_t w_msg_smem = (lmsg_ == 1) ? (size_t)(msg_in_dim_[0] + msg_in_dim_[1]) * n_msg_ * sizeof(float) : 0;
    const bool use_fused_l1_msg = fast_l1_node_lf1_ && (max_nl_ <= 32) && (n_msg_ <= 16) && (w_msg_smem <= 32768);
    const bool use_I_pack4 = use_fused_l1_msg && use_l2_sp && d_I_pack4_;
    if (use_fused_l1_msg) {
        Ptr4 pI{{d_I_[0], d_I_[1], d_I_[2], d_I_[3]}};
        const unsigned g256 = (unsigned)((N + 7) / 8);
        if (lmax_ == 2) {
            ilmlip_l1_node_fwd_lf1_msg_kernel<2><<<g256, 256, w_msg_smem, st>>>(
                N, n_feat_pad_, n_msg_, nl4, mid4, cA1, d_W_msg_[0], d_W_msg_[1],
                d_x_, use_I_pack4 ? d_I_pack4_ : nullptr, pI);
        } else {
            ilmlip_l1_node_fwd_lf1_msg_kernel<3><<<g256, 256, w_msg_smem, st>>>(
                N, n_feat_pad_, n_msg_, nl4, mid4, cA1, d_W_msg_[0], d_W_msg_[1],
                d_x_, use_I_pack4 ? d_I_pack4_ : nullptr, pI);
        }
    } else {
        if (fast_l1_node_lf1_) {
            const dim3 grid((unsigned)((N + 7) / 8), (unsigned)((max_nl_ + 31) / 32));
            const dim3 blk(32, 8);
            if (lmax_ == 2)
                ilmlip_l1_node_fwd_lf1_kernel<2><<<grid, blk, 0, st>>>(N, n_feat_pad_, nl4, mid4, cA1, d_x_, pBcat);
            else
                ilmlip_l1_node_fwd_lf1_kernel<3><<<grid, blk, 0, st>>>(N, n_feat_pad_, nl4, mid4, cA1, d_x_, pBcat);
        } else {
            const int per_atom_f = n_sh_l1 + std::max(b2_scalars_, 1);
            const size_t tbl_b = l1n_smem_tbl ? l1n_tbl_bytes : 0;
            const int atoms_per_blk_fwd = ((size_t)4 * per_atom_f * 32 * sizeof(float) + tbl_b <= 48000) ? 4 : 2;
            const dim3 l1n_grid((unsigned)((N + atoms_per_blk_fwd - 1) / atoms_per_blk_fwd),
                                (unsigned)((max_nl_ + 31) / 32));
            const dim3 l1n_blk(32, (unsigned)atoms_per_blk_fwd);
            const size_t l1n_smem_fwd = tbl_b + (size_t)atoms_per_blk_fwd * per_atom_f * 32 * sizeof(float);
            auto launch_l1_node_fwd = [&](auto lm_tag, auto st_tag) {
                constexpr int LM = decltype(lm_tag)::value;
                constexpr bool ST = decltype(st_tag)::value;
                ilmlip_l1_node_fwd_kernel<LM, ST><<<l1n_grid, l1n_blk, l1n_smem_fwd, st>>>(
                    N, lmsg_, n_nu2_, n_nu3_, cg_count_, n_feat_pad_, b2_scalars_, nl4, mid4,
                    d_nu2_, d_nu3_, d_cg_idx_, d_cg_val_, cA1, d_x_, pBcat);
            };
            auto disp_l1nf_st = [&](auto lm_tag) {
                if (l1n_smem_tbl) launch_l1_node_fwd(lm_tag, std::true_type{});
                else              launch_l1_node_fwd(lm_tag, std::false_type{});
            };
            if (lmax_ <= 2) disp_l1nf_st(std::integral_constant<int, 2>{});
            else            disp_l1nf_st(std::integral_constant<int, 3>{});
        }

        for (int l = 0; l <= lmsg_; ++l) {
            const int M = N * (2 * l + 1);
            const int K = msg_in_dim_[l];
            if (n_msg_ <= 16 && K >= 64 && (int64_t)K * n_msg_ <= 4096) {
                const unsigned g = (unsigned)((M + 31) / 32);
                const size_t sm = (size_t)K * n_msg_ * sizeof(float);
                ilmlip_skinny_gemm_n16_kernel<<<g, 128, sm, st>>>(M, n_msg_, K, d_Bcat_[l], d_W_msg_[l], d_I_[l]);
            } else {
                in.gemm(false, M, n_msg_, K, 1.0f, d_Bcat_[l], K, d_W_msg_[l], n_msg_, 0.0f, d_I_[l], n_msg_);
            }
        }
    }

    // 3. Layer-2 Edge Forward
    const float a0_s1 = a0_ / sigma1_;
    const float inv_d2_s1k = 1.0f / ((float)D_A2_ * sigma1_ * sigma1_ * kappa_ * kappa_);
    const float4* I_pack4_ptr = use_I_pack4 ? d_I_pack4_ : nullptr;
    if (use_l2_sp) {
        if (remove_backflow_) {
            ilmlip_l2_edge_fwd_sp_kernel<true><<<l2_sp_grid, 256, l2_sp_smem, st>>>(
                N, S_, n_msg_, l2c4, inv_rmax, a0_s1, inv_d2_s1k,
                in.act_row_ptr, in.act_senders, in.act_vec_len, in.species,
                d_C2_, d_Cbf_, d_f0_, I_pack4_ptr, cI, pA2, d_f1_);
        } else {
            ilmlip_l2_edge_fwd_sp_kernel<false><<<l2_sp_grid, 256, l2_sp_smem, st>>>(
                N, S_, n_msg_, l2c4, inv_rmax, a0_s1, inv_d2_s1k,
                in.act_row_ptr, in.act_senders, in.act_vec_len, in.species,
                d_C2_, d_Cbf_, d_f0_, I_pack4_ptr, cI, pA2, d_f1_);
        }
    } else if (use_fast_l2) {
        auto launch_l2_fast = [&](auto lf_tag, auto sw_tag, auto mp_tag, auto nb_tag, auto bf_tag, auto sm_tag) {
            constexpr int LF = decltype(lf_tag)::value;
            constexpr int SW = decltype(sw_tag)::value;
            constexpr int MP = decltype(mp_tag)::value;
            constexpr int NB = decltype(nb_tag)::value;
            constexpr bool BF = decltype(bf_tag)::value;
            constexpr bool SM = decltype(sm_tag)::value;
            const unsigned grid = (SW == 16) ? half_grid : warp_grid;
            ilmlip_l2_edge_fwd_fast_kernel<LF, SW, MP, NB, BF, SM><<<grid, 128, SM ? l2_c_bytes : 0, st>>>(
                N, S_, n_base_, n_msg_, l2c4, inv_rmax, a0_s1, inv_d2_s1k,
                in.act_row_ptr, in.act_senders, in.act_vec_len, in.species,
                d_C2_, d_Cbf_, d_f0_, cI, pA2, d_f1_);
        };
        auto disp_sm = [&](auto lf_tag, auto sw_tag, auto mp_tag, auto nb_tag, auto bf_tag) {
            if (use_smem_c2) launch_l2_fast(lf_tag, sw_tag, mp_tag, nb_tag, bf_tag, std::true_type{});
            else             launch_l2_fast(lf_tag, sw_tag, mp_tag, nb_tag, bf_tag, std::false_type{});
        };
        auto disp_bf = [&](auto lf_tag, auto sw_tag, auto mp_tag, auto nb_tag) {
            if (remove_backflow_) disp_sm(lf_tag, sw_tag, mp_tag, nb_tag, std::true_type{});
            else                  disp_sm(lf_tag, sw_tag, mp_tag, nb_tag, std::false_type{});
        };
        auto disp_nb = [&](auto lf_tag, auto sw_tag, auto mp_tag) {
            if (use_nb16) disp_bf(lf_tag, sw_tag, mp_tag, std::integral_constant<int, 16>{});
            else          disp_bf(lf_tag, sw_tag, mp_tag, std::integral_constant<int, 32>{});
        };
        auto disp_sw = [&](auto lf_tag) {
            if (n_msg_ <= 16)      disp_nb(lf_tag, std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
            else if (n_msg_ <= 32) disp_nb(lf_tag, std::integral_constant<int, 32>{}, std::integral_constant<int, 1>{});
            else                   disp_nb(lf_tag, std::integral_constant<int, 32>{}, std::integral_constant<int, 2>{});
        };
        if (lfeat2_ == 1) disp_sw(std::integral_constant<int, 1>{});
        else              disp_sw(std::integral_constant<int, 2>{});
    } else {
        auto launch_l2_fwd = [&](auto mp_tag, auto out_tag, auto bf_tag) {
            constexpr int MP = decltype(mp_tag)::value;
            constexpr int MO = decltype(out_tag)::value;
            constexpr bool BF = decltype(bf_tag)::value;
            ilmlip_l2_edge_fwd_kernel<MP, MO, BF><<<warp_grid, 128, 0, st>>>(
                N, S_, lfeat2_, lmsg_, lout2_, n_base_, n_msg_, n_l2_paths_, l2c4,
                inv_rmax, a0_s1, inv_d2_s1k, in.act_row_ptr, in.act_senders, in.act_vec_len, in.species,
                d_C2_, d_Cbf_, d_f0_, cI, d_l2p_, d_cg_idx_, d_cg_val_, pA2, d_f1_);
        };
        auto disp_l2_fwd_bf = [&](auto mp_tag, auto out_tag) {
            if (remove_backflow_) launch_l2_fwd(mp_tag, out_tag, std::true_type{});
            else                  launch_l2_fwd(mp_tag, out_tag, std::false_type{});
        };
        auto disp_l2_fwd_out = [&](auto mp_tag) {
            if (total_l2_out_ <= 24) disp_l2_fwd_bf(mp_tag, std::integral_constant<int, 24>{});
            else                     disp_l2_fwd_bf(mp_tag, std::integral_constant<int, 48>{});
        };
        if (n_msg_ <= 32) disp_l2_fwd_out(std::integral_constant<int, 1>{});
        else              disp_l2_fwd_out(std::integral_constant<int, 2>{});
    }

    // 4. Layer-2 Projections + Invariants + Standardization + Readout MLP
    const size_t bisp_tbl_bytes = (r_bisp_ > 0 && n_bisp_ > 0)
        ? (size_t)(n_bisp_ * 5 + bisp_idx_count_ * 2) * sizeof(int32_t) : 0;

    if (l2_c_[0] > 0 && r_bisp_ > 0) {
        in.gemm(false, N, r_bisp_, l2_c_[0], 1.0f, d_A2_[0], l2_c_[0], d_W_B0_, r_bisp_, 0.0f, d_Q0_, r_bisp_);
    }
    for (int L = 1; L <= lout2_; ++L) {
        if (l2_c_[L] == 0 || r_gb_ == 0) continue;
        const int M = N * (2 * L + 1);
        const int K = l2_c_[L];
        in.gemm(false, M, r_gb_, K, 1.0f, d_A2_[L], K, d_W_GB_[L], r_gb_, 0.0f, d_PQ_[L], r_gb_);
    }
    if (fast_bisp_l2_ && r_gram_ <= 32) {
        const size_t sm = (r_gram_ > 0) ? (size_t)4 * 5 * (r_gram_ + 1) * sizeof(float) : 0;
        ilmlip_l2_inv_fwd_l2_kernel<<<warp_grid, 128, sm, st>>>(
            N, r_gram_, r_bisp_, n_inv1_, n_feat_pad_, bisp_off_, l2c4, goff4,
            d_triu_i_, d_triu_j_, d_A2_[0], d_Q0_, cPQ, d_mu_x_, d_inv_std_x_, d_x_, d_eA_x_);
    } else {
        const size_t inv_fwd_smem = bisp_tbl_bytes +
            ((r_bisp_ > 0 && n_bisp_ > 0 && lout2_ <= 2 && r_bisp_ <= 32) ? (size_t)4 * 9 * r_bisp_ * sizeof(float) : 0);
        ilmlip_l2_inv_and_norm_fwd_kernel<<<warp_grid, 128, inv_fwd_smem, st>>>(
            N, lout2_, r_gram_, r_bisp_, n_bisp_, bisp_idx_count_, n_inv1_, n_feat_pad_, bisp_off_, l2c4, goff4,
            d_triu_i_, d_triu_j_, d_bisp_, d_bisp_idx_, d_bisp_val_, d_A2_[0], d_Q0_, cPQ,
            d_mu_x_, d_inv_std_x_, d_x_);
    }

    const bool do_ro_h2 = use_ro_h2_ && (bool)in.gemm_h2_row && (N >= 128);
    const int8_t* eA_x_ptr = (fast_bisp_l2_ && r_gram_ <= 32) ? d_eA_x_ : nullptr;
    const float inv_zc = 1.0f / z_c_;
    const bool need_grad = (in.energy_only == 0);
    const char* ro_fuse_env = std::getenv("LOUP_ILMLIP_RO_FUSE");
    const bool use_ro_tail_fused = do_ro_h2 && need_grad && (ro_depth_ == 2) && (ro_width_ == 128) &&
                                   d_ro_h2_B_[0] && d_ro_h2_B_[1] && d_ro_h2_BT_[1] && d_ro_h2_BT_[0] &&
                                   !(ro_fuse_env && ro_fuse_env[0] == '0');

    if (use_ro_tail_fused) {
        // 1) ro_W_0 fwd using cached eA_x_ptr (ZERO h2r_row_exp_kernel!)
        in.gemm_h2_row(N, ro_width_, n_feat_pad_, d_x_, d_ro_h2_B_[0], d_ro_h2_pB_[0], d_pre_[0], 0.0f, eA_x_ptr);
        // 2) Fused Readout Tail: collapses 7 kernels (bias_silu x2, W1 fwd, energy_ro+ZBL, silu_bwd x2, W1 bwd) into ONE CTA!
        hgemm2_ilmlip_ro_tail_w128(
            N, use_zbl_switch_ != 0, inv_zc, in.species,
            d_pre_[0], d_alpha_Z_, d_ro_b_[1], d_u_Z_,
            d_ro_h2_B_[1], d_ro_h2_pB_[1], d_ro_h2_BT_[1], d_ro_h2_pBT_[1],
            d_e_zbl_, in.energy, in.atom_energy, d_w_zbl_, d_dh_[0], st);
        // 3) ro_W_0 bwd: d_dx = dpre0 @ ro_W_0^T
        in.gemm_h2_row(N, n_feat_pad_, ro_width_, d_dh_[0], d_ro_h2_BT_[0], d_ro_h2_pBT_[0], d_dx_, 0.0f, nullptr);
    } else {
        for (int d = 0; d < ro_depth_; ++d) {
            const float* inp = (d == 0) ? d_x_ : d_h_[d - 1];
            const int K = (d == 0) ? n_feat_pad_ : ro_width_;
            if (do_ro_h2 && d_ro_h2_B_[d]) {
                in.gemm_h2_row(N, ro_width_, K, inp, d_ro_h2_B_[d], d_ro_h2_pB_[d], d_pre_[d], 0.0f,
                               (d == 0) ? eA_x_ptr : nullptr);
            } else {
                in.gemm(false, N, ro_width_, K, 1.0f, inp, K, d_ro_W_[d], ro_width_, 0.0f, d_pre_[d], ro_width_);
            }
            if ((ro_width_ & 3) == 0) {
                ilmlip_bias_silu_fwd_vec4_kernel<<<warp_grid, 128, 0, st>>>(
                    N, ro_width_ >> 2, d == 0, in.species,
                    reinterpret_cast<const float4*>((d == 0) ? d_alpha_Z_ : d_ro_b_[d]),
                    reinterpret_cast<float4*>(d_pre_[d]), reinterpret_cast<float4*>(d_h_[d]));
            } else {
                const int64_t tot = (int64_t)N * ro_width_;
                ilmlip_bias_silu_fwd_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(
                    N, ro_width_, d == 0, in.species, (d == 0) ? d_alpha_Z_ : d_ro_b_[d], d_pre_[d], d_h_[d]);
            }
        }

        if (use_zbl_switch_) {
            if (need_grad)
                ilmlip_energy_ro_kernel<true, true><<<warp_grid, 128, 0, st>>>(
                    N, ro_width_, inv_zc, in.species, d_h_[ro_depth_ - 1], d_u_Z_, d_e_zbl_,
                    in.energy, in.atom_energy, d_dh_[0], d_w_zbl_);
            else
                ilmlip_energy_ro_kernel<true, false><<<warp_grid, 128, 0, st>>>(
                    N, ro_width_, inv_zc, in.species, d_h_[ro_depth_ - 1], d_u_Z_, d_e_zbl_,
                    in.energy, in.atom_energy, nullptr, nullptr);
        } else {
            if (need_grad)
                ilmlip_energy_ro_kernel<false, true><<<warp_grid, 128, 0, st>>>(
                    N, ro_width_, inv_zc, in.species, d_h_[ro_depth_ - 1], d_u_Z_, nullptr,
                    in.energy, in.atom_energy, d_dh_[0], nullptr);
            else
                ilmlip_energy_ro_kernel<false, false><<<warp_grid, 128, 0, st>>>(
                    N, ro_width_, inv_zc, in.species, d_h_[ro_depth_ - 1], d_u_Z_, nullptr,
                    in.energy, in.atom_energy, nullptr, nullptr);
        }

        if (!need_grad) return;

        // 5. Readout MLP Backward
        int cur_dh = 0;
        for (int d = ro_depth_ - 1; d >= 0; --d) {
            const int64_t tot = (int64_t)N * ro_width_;
            float* dpre = d_dh_[1 - cur_dh];
            if ((tot & 3) == 0) {
                const int64_t tot4 = tot >> 2;
                ilmlip_silu_bwd_vec4_kernel<<<(unsigned)((tot4 + 255) / 256), 256, 0, st>>>(
                    tot4, reinterpret_cast<const float4*>(d_pre_[d]),
                    reinterpret_cast<const float4*>(d_dh_[cur_dh]), reinterpret_cast<float4*>(dpre));
            } else {
                ilmlip_silu_bwd_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(
                    N, ro_width_, d_pre_[d], d_dh_[cur_dh], dpre);
            }
            if (d > 0) {
                if (do_ro_h2 && d_ro_h2_BT_[d]) {
                    in.gemm_h2_row(N, ro_width_, ro_width_, dpre, d_ro_h2_BT_[d], d_ro_h2_pBT_[d], d_dh_[cur_dh], 0.0f, nullptr);
                } else {
                    in.gemm(true, N, ro_width_, ro_width_, 1.0f, dpre, ro_width_, d_ro_W_[d], ro_width_,
                            0.0f, d_dh_[cur_dh], ro_width_);
                }
            } else {
                if (do_ro_h2 && d_ro_h2_BT_[0]) {
                    in.gemm_h2_row(N, n_feat_pad_, ro_width_, dpre, d_ro_h2_BT_[0], d_ro_h2_pBT_[0], d_dx_, 0.0f, nullptr);
                } else {
                    in.gemm(true, N, n_feat_pad_, ro_width_, 1.0f, dpre, ro_width_, d_ro_W_[0], ro_width_,
                            0.0f, d_dx_, n_feat_pad_);
                }
            }
        }
    }

    // 6. Layer-2 Invariants Backward + merged W_GB^T GEMMs + f1 Gate Backward
    auto launch_gate_bwd = [&](int n_deg, const Int4& C_deg, float inv_s, CPtr4 A, Ptr4 dA,
                               const float* f_arr, const float* df_ext) {
        if (n_deg == 2)
            ilmlip_gate_bwd_kernel<2><<<warp_grid, 128, 0, st>>>(N, C_deg, inv_s, A, dA, f_arr, df_ext);
        else if (n_deg == 3)
            ilmlip_gate_bwd_kernel<3><<<warp_grid, 128, 0, st>>>(N, C_deg, inv_s, A, dA, f_arr, df_ext);
        else
            ilmlip_gate_bwd_kernel<4><<<warp_grid, 128, 0, st>>>(N, C_deg, inv_s, A, dA, f_arr, df_ext);
    };
    const float inv_d2_ak = 1.0f / ((float)D_A2_ * a0_ * a0_ * kappa_ * kappa_);

    if (fast_bisp_l2_ && r_gram_ <= 32) {
        const int n_tri = (r_gram_ > 0) ? ((r_gram_ * (r_gram_ + 1)) >> 1) : 0;
        const size_t sm = (r_gram_ > 0)
            ? ((size_t)2 * n_tri * sizeof(int32_t) + (size_t)4 * (5 + r_gram_) * (r_gram_ + 1) * sizeof(float))
            : 0;
        ilmlip_l2_inv_bwd_l2_kernel<<<warp_grid, 128, sm, st>>>(
            N, r_gram_, r_bisp_, n_inv1_, n_feat_pad_, bisp_off_, l2c4, goff4,
            d_triu_i_, d_triu_j_, d_inv_std_x_, d_dx_, d_dA2_[0], d_Q0_, d_dQ0_, cPQ, pdPQ);
    } else {
        const int n_tri = (r_gram_ * (r_gram_ + 1)) >> 1;
        const size_t g_work = (r_gram_ > 0 && r_gram_ <= 16) ? (size_t)4 * n_tri * sizeof(float) : 0;
        const size_t q_work = (r_bisp_ > 0 && n_bisp_ > 0 && lout2_ <= 2 && r_bisp_ <= 32)
            ? (size_t)4 * 18 * r_bisp_ * sizeof(float) : 0;
        const size_t inv_bwd_smem = bisp_tbl_bytes + std::max(g_work, q_work);
        ilmlip_l2_inv_bwd_kernel<<<warp_grid, 128, inv_bwd_smem, st>>>(
            N, lout2_, r_gram_, r_bisp_, n_bisp_, bisp_idx_count_, n_inv1_, n_feat_pad_, bisp_off_, l2c4, goff4,
            d_bisp_, d_bisp_idx_, d_bisp_val_, d_inv_std_x_, d_dx_, d_dA2_[0], d_Q0_, d_dQ0_, cPQ, pdPQ);
    }

    if (l2_c_[0] > 0 && r_bisp_ > 0) {
        in.gemm(true, N, l2_c_[0], r_bisp_, 1.0f, d_dQ0_, r_bisp_, d_W_B0_, r_bisp_, 1.0f, d_dA2_[0], l2_c_[0]);
    }
    for (int L = 1; L <= lout2_; ++L) {
        if (l2_c_[L] == 0) continue;
        const int M = N * (2 * L + 1);
        const int cL = l2_c_[L];
        if (r_gb_ > 0) {
            in.gemm(true, M, cL, r_gb_, 1.0f, d_dPQ_[L], r_gb_, d_W_GB_[L], r_gb_, 0.0f, d_dA2_[L], cL);
        } else {
            CUDA_CHECK_THROW(cudaMemsetAsync(d_dA2_[L], 0, (size_t)M * cL * sizeof(float), st));
        }
    }

    // 7. Layer-2 Gate Backward + Layer-2 Edge Backward
    const char* l2_snd_env = std::getenv("LOUP_ILMLIP_L2_SENDER");
    const bool use_l2_sender = use_l2_sp && d_dA2_pack_ && (in.n_ghost == 0) &&
                               !(l2_snd_env && l2_snd_env[0] == '0');
    if (use_l2_sender) {
        // Fused f1 gate backward + pack [g0..g6, g7c, w00..w22] -> ZERO atomicAdd(dI_node) & ZERO zero-slab memset!
        ilmlip_gate_bwd_pack_l2_kernel<<<half_grid, 128, 0, st>>>(
            N, n_msg_, l2c4, inv_d2_ak, cA2, cdA2, d_f1_, d_dA2_pack_);
        auto launch_l2_snd = [&](auto bf_tag, auto st_tag) {
            ilmlip_l2_edge_bwd_sender_kernel<decltype(bf_tag)::value, decltype(st_tag)::value><<<half_grid, 128, 0, st>>>(
                N, S_, n_msg_, inv_rmax,
                in.act_row_ptr, in.act_senders, in.act_vec_len, in.species,
                d_C2_, d_Cbf_, d_f0_, I_pack4_ptr, cI, d_dA2_pack_,
                pdI, d_df0_bf_, in.forces, in.virial);
        };
        if (remove_backflow_) {
            if (in.compute_stress) launch_l2_snd(std::true_type{}, std::true_type{});
            else                   launch_l2_snd(std::true_type{}, std::false_type{});
        } else {
            if (in.compute_stress) launch_l2_snd(std::false_type{}, std::true_type{});
            else                   launch_l2_snd(std::false_type{}, std::false_type{});
        }
    } else {
        launch_gate_bwd(lout2_ + 1, l2c4, inv_d2_ak, cA2, pdA2, d_f1_, nullptr);
        CUDA_CHECK_THROW(cudaMemsetAsync(d_zero_slab_, 0, zero_slab_bytes_, st));
        if (use_l2_sp) {
            auto launch_l2_sp = [&](auto bf_tag, auto st_tag) {
                ilmlip_l2_edge_bwd_sp_kernel<decltype(bf_tag)::value, decltype(st_tag)::value><<<half_grid, 128, 0, st>>>(
                    N, S_, n_msg_, l2c4, inv_rmax,
                    in.act_row_ptr, in.act_senders, in.act_vec_len, in.species,
                    d_C2_, d_Cbf_, d_f0_, I_pack4_ptr, cI, cdA2,
                    pdI, d_df0_bf_, in.forces, in.virial);
            };
            if (remove_backflow_) {
                if (in.compute_stress) launch_l2_sp(std::true_type{}, std::true_type{});
                else                   launch_l2_sp(std::true_type{}, std::false_type{});
            } else {
                if (in.compute_stress) launch_l2_sp(std::false_type{}, std::true_type{});
                else                   launch_l2_sp(std::false_type{}, std::false_type{});
            }
        } else if (use_fast_l2) {
            auto launch_l2_bwd_fast = [&](auto lf_tag, auto sw_tag, auto mp_tag, auto nb_tag, auto bf_tag, auto st_tag, auto sm_tag) {
                constexpr int LF = decltype(lf_tag)::value;
                constexpr int SW = decltype(sw_tag)::value;
                constexpr int MP = decltype(mp_tag)::value;
                constexpr int NB = decltype(nb_tag)::value;
                constexpr bool BF = decltype(bf_tag)::value;
                constexpr bool ST = decltype(st_tag)::value;
                constexpr bool SM = decltype(sm_tag)::value;
                const unsigned grid = (SW == 16) ? half_grid : warp_grid;
                ilmlip_l2_edge_bwd_fast_kernel<LF, SW, MP, NB, BF, ST, SM><<<grid, 128, SM ? l2_c_bytes : 0, st>>>(
                    N, S_, n_base_, n_msg_, l2c4, inv_rmax,
                    in.act_row_ptr, in.act_senders, in.act_vec_len, in.species,
                    d_C2_, d_Cbf_, d_f0_, cI, cdA2,
                    pdI, d_df0_bf_, in.forces, in.virial);
            };
            auto disp_sm = [&](auto lf_tag, auto sw_tag, auto mp_tag, auto nb_tag, auto bf_tag, auto st_tag) {
                if (use_smem_c2) launch_l2_bwd_fast(lf_tag, sw_tag, mp_tag, nb_tag, bf_tag, st_tag, std::true_type{});
                else             launch_l2_bwd_fast(lf_tag, sw_tag, mp_tag, nb_tag, bf_tag, st_tag, std::false_type{});
            };
            auto disp_st = [&](auto lf_tag, auto sw_tag, auto mp_tag, auto nb_tag, auto bf_tag) {
                if (in.compute_stress) disp_sm(lf_tag, sw_tag, mp_tag, nb_tag, bf_tag, std::true_type{});
                else                   disp_sm(lf_tag, sw_tag, mp_tag, nb_tag, bf_tag, std::false_type{});
            };
            auto disp_bf = [&](auto lf_tag, auto sw_tag, auto mp_tag, auto nb_tag) {
                if (remove_backflow_) disp_st(lf_tag, sw_tag, mp_tag, nb_tag, std::true_type{});
                else                  disp_st(lf_tag, sw_tag, mp_tag, nb_tag, std::false_type{});
            };
            auto disp_nb = [&](auto lf_tag, auto sw_tag, auto mp_tag) {
                if (use_nb16) disp_bf(lf_tag, sw_tag, mp_tag, std::integral_constant<int, 16>{});
                else          disp_bf(lf_tag, sw_tag, mp_tag, std::integral_constant<int, 32>{});
            };
            auto disp_sw = [&](auto lf_tag) {
                if (n_msg_ <= 16)      disp_nb(lf_tag, std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
                else if (n_msg_ <= 32) disp_nb(lf_tag, std::integral_constant<int, 32>{}, std::integral_constant<int, 1>{});
                else                   disp_nb(lf_tag, std::integral_constant<int, 32>{}, std::integral_constant<int, 2>{});
            };
            if (lfeat2_ == 1) disp_sw(std::integral_constant<int, 1>{});
            else              disp_sw(std::integral_constant<int, 2>{});
        } else {
            auto launch_l2_bwd = [&](auto mp_tag, auto out_tag, auto bf_tag, auto st_tag) {
                constexpr int MP = decltype(mp_tag)::value;
                constexpr int MO = decltype(out_tag)::value;
                constexpr bool BF = decltype(bf_tag)::value;
                constexpr bool ST = decltype(st_tag)::value;
                ilmlip_l2_edge_bwd_kernel<MP, MO, BF, ST><<<warp_grid, 128, 0, st>>>(
                    N, S_, lfeat2_, lmsg_, lout2_, n_base_, n_msg_, n_l2_paths_, l2c4, inv_rmax,
                    in.act_row_ptr, in.act_senders, in.act_vec_len, in.species,
                    d_C2_, d_Cbf_, d_f0_, cI, cdA2, d_l2p_, d_cg_idx_, d_cg_val_,
                    pdI, d_df0_bf_, in.forces, in.virial);
            };
            auto disp_l2_bwd_st = [&](auto mp_tag, auto out_tag, auto bf_tag) {
                if (in.compute_stress) launch_l2_bwd(mp_tag, out_tag, bf_tag, std::true_type{});
                else                   launch_l2_bwd(mp_tag, out_tag, bf_tag, std::false_type{});
            };
            auto disp_l2_bwd_bf = [&](auto mp_tag, auto out_tag) {
                if (remove_backflow_) disp_l2_bwd_st(mp_tag, out_tag, std::true_type{});
                else                  disp_l2_bwd_st(mp_tag, out_tag, std::false_type{});
            };
            auto disp_l2_bwd_out = [&](auto mp_tag) {
                if (total_l2_out_ <= 24) disp_l2_bwd_bf(mp_tag, std::integral_constant<int, 24>{});
                else                     disp_l2_bwd_bf(mp_tag, std::integral_constant<int, 48>{});
            };
            if (n_msg_ <= 32) disp_l2_bwd_out(std::integral_constant<int, 1>{});
            else              disp_l2_bwd_out(std::integral_constant<int, 2>{});
        }
    }

    // 8. Layer-1 W_msg^T GEMMs + Node Backward + f0 Gate Backward + Edge-Parallel Layer-1 Edge Backward (+ ZBL)
    if (use_fused_l1_msg) {
        CPtr4 cdI{{d_dI_[0], d_dI_[1], d_dI_[2], d_dI_[3]}};
        const unsigned g256 = (unsigned)((N + 7) / 8);
        if (lmax_ == 2) {
            ilmlip_l1_node_bwd_lf1_msg_kernel<2><<<g256, 256, w_msg_smem, st>>>(
                N, n_feat_pad_, n_msg_, nl4, mid4, cA1, d_dx_, cdI, d_W_msg_T_[0], d_W_msg_T_[1], pdA1);
        } else {
            ilmlip_l1_node_bwd_lf1_msg_kernel<3><<<g256, 256, w_msg_smem, st>>>(
                N, n_feat_pad_, n_msg_, nl4, mid4, cA1, d_dx_, cdI, d_W_msg_T_[0], d_W_msg_T_[1], pdA1);
        }
    } else {
        for (int l = 0; l <= lmsg_; ++l) {
            const int M = N * (2 * l + 1);
            const int K = msg_in_dim_[l];
            in.gemm(true, M, K, n_msg_, 1.0f, d_dI_[l], n_msg_, d_W_msg_[l], n_msg_, 0.0f, d_dBcat_[l], K);
        }

        if (fast_l1_node_lf1_) {
            const dim3 grid((unsigned)((N + 7) / 8), (unsigned)((max_nl_ + 31) / 32));
            const dim3 blk(32, 8);
            if (lmax_ == 2)
                ilmlip_l1_node_bwd_lf1_kernel<2><<<grid, blk, 0, st>>>(N, n_feat_pad_, nl4, mid4, cA1, d_dx_, cdBcat, pdA1);
            else
                ilmlip_l1_node_bwd_lf1_kernel<3><<<grid, blk, 0, st>>>(N, n_feat_pad_, nl4, mid4, cA1, d_dx_, cdBcat, pdA1);
        } else {
            const size_t tbl_b = l1n_smem_tbl ? l1n_tbl_bytes : 0;
            const int per_atom_b = 2 * (n_sh_l1 + std::max(b2_scalars_, 1));
            const int atoms_per_blk_bwd = ((size_t)4 * per_atom_b * 32 * sizeof(float) + tbl_b <= 48000) ? 4 : 2;
            const dim3 l1nb_grid((unsigned)((N + atoms_per_blk_bwd - 1) / atoms_per_blk_bwd),
                                 (unsigned)((max_nl_ + 31) / 32));
            const dim3 l1nb_blk(32, (unsigned)atoms_per_blk_bwd);
            const size_t l1nb_smem = tbl_b + (size_t)atoms_per_blk_bwd * per_atom_b * 32 * sizeof(float);
            auto launch_l1_node_bwd = [&](auto lm_tag, auto st_tag) {
                constexpr int LM = decltype(lm_tag)::value;
                constexpr bool ST = decltype(st_tag)::value;
                ilmlip_l1_node_bwd_kernel<LM, ST><<<l1nb_grid, l1nb_blk, l1nb_smem, st>>>(
                    N, lmsg_, n_nu2_, n_nu3_, cg_count_, n_feat_pad_, b2_scalars_, nl4, mid4,
                    d_nu2_, d_nu3_, d_cg_idx_, d_cg_val_, cA1, d_dx_, cdBcat, pdA1);
            };
            auto disp_l1nb_st = [&](auto lm_tag) {
                if (l1n_smem_tbl) launch_l1_node_bwd(lm_tag, std::true_type{});
                else              launch_l1_node_bwd(lm_tag, std::false_type{});
            };
            if (lmax_ <= 2) disp_l1nb_st(std::integral_constant<int, 2>{});
            else            disp_l1nb_st(std::integral_constant<int, 3>{});
        }
    }

    const float inv_d1_ak = 1.0f / ((float)D_A1_ * a0_ * a0_ * kappa_ * kappa_);
    const bool use_l1_pre = use_l1_sumswap && (S_ * n_base_ <= 32) && (D_A1_ <= ((lmax_ == 2) ? 256 : 384));
    if (use_l1_pre) {
        // f0 gate backward is FUSED into Phase 1a; ZBL is decoupled via d_zbl_flag_ -> 72 regs (67% occ)!
        auto launch_l1_pre = [&](auto lm_tag, auto nb_tag, auto st_tag) {
            constexpr int LM = decltype(lm_tag)::value;
            constexpr int NB = decltype(nb_tag)::value;
            constexpr bool ST = decltype(st_tag)::value;
            ilmlip_l1_edge_bwd_precontract_kernel<LM, NB, false, ST><<<warp_grid, 128, 0, st>>>(
                N, S_, n_tot1_, nl4, l1off4, inv_rmax, inv_d1_ak,
                in.act_row_ptr, in.act_senders, in.act_vec_len, in.species, nullptr, d_C1_zick_,
                nullptr, nullptr, nullptr, nullptr,
                cA1, cdA1, d_f0_, remove_backflow_ ? d_df0_bf_ : nullptr, in.forces, in.virial);
        };
        auto disp_st = [&](auto lm_tag, auto nb_tag) {
            if (in.compute_stress) launch_l1_pre(lm_tag, nb_tag, std::true_type{});
            else                   launch_l1_pre(lm_tag, nb_tag, std::false_type{});
        };
        auto disp_nb = [&](auto lm_tag) {
            if (n_base_ == 8)       disp_st(lm_tag, std::integral_constant<int, 8>{});
            else if (n_base_ == 12) disp_st(lm_tag, std::integral_constant<int, 12>{});
            else                    disp_st(lm_tag, std::integral_constant<int, 16>{});
        };
        if (lmax_ == 2) disp_nb(std::integral_constant<int, 2>{});
        else            disp_nb(std::integral_constant<int, 3>{});
        if (use_zbl_switch_) {
            if (in.compute_stress) {
                ilmlip_zbl_bwd_kernel<true><<<warp_grid, 128, 0, st>>>(
                    N, S_, d_zbl_flag_, in.act_row_ptr, in.act_senders, in.act_vec_len, in.species, in.pos64,
                    d_rs_pair_, d_zz_pair_, d_au_pair_, d_w_zbl_, in.forces, in.virial);
            } else {
                ilmlip_zbl_bwd_kernel<false><<<warp_grid, 128, 0, st>>>(
                    N, S_, d_zbl_flag_, in.act_row_ptr, in.act_senders, in.act_vec_len, in.species, in.pos64,
                    d_rs_pair_, d_zz_pair_, d_au_pair_, d_w_zbl_, in.forces, in.virial);
            }
        }
    } else {
        launch_gate_bwd(lmax_ + 1, nl4, inv_d1_ak, cA1, pdA1, d_f0_, remove_backflow_ ? d_df0_bf_ : nullptr);
        const size_t ga_bytes = (size_t)D_A1_ * sizeof(float);
        const size_t c1_zi_bytes = (size_t)S_ * n_base_ * n_tot1_ * sizeof(float);
        const bool use_smem_c1_bwd = (ga_bytes + c1_zi_bytes <= 36864);
        const size_t l1b_smem = ga_bytes + (use_smem_c1_bwd ? c1_zi_bytes : 0);
        auto launch_l1_ep = [&](auto lm_tag, auto nb_tag, auto zb_tag, auto st_tag, auto sm_tag) {
            constexpr int LM = decltype(lm_tag)::value;
            constexpr int NB = decltype(nb_tag)::value;
            constexpr bool ZB = decltype(zb_tag)::value;
            constexpr bool ST = decltype(st_tag)::value;
            constexpr bool SM = decltype(sm_tag)::value;
            ilmlip_l1_edge_bwd_edgepar_kernel<LM, NB, ZB, ST, SM><<<(unsigned)N, 128, l1b_smem, st>>>(
                N, S_, n_base_, n_tot1_, nl4, l1off4, inv_rmax,
                in.act_row_ptr, in.act_senders, in.act_vec_len, in.species, in.pos64, d_C1_,
                d_rs_pair_, d_zz_pair_, d_au_pair_, d_w_zbl_, cdA1, in.forces, in.virial);
        };
        auto disp_l1ep_sm = [&](auto lm_tag, auto nb_tag, auto zb_tag, auto st_tag) {
            if (use_smem_c1_bwd) launch_l1_ep(lm_tag, nb_tag, zb_tag, st_tag, std::true_type{});
            else                 launch_l1_ep(lm_tag, nb_tag, zb_tag, st_tag, std::false_type{});
        };
        auto disp_l1ep_st = [&](auto lm_tag, auto nb_tag, auto zb_tag) {
            if (in.compute_stress) disp_l1ep_sm(lm_tag, nb_tag, zb_tag, std::true_type{});
            else                   disp_l1ep_sm(lm_tag, nb_tag, zb_tag, std::false_type{});
        };
        auto disp_l1ep_zb = [&](auto lm_tag, auto nb_tag) {
            if (use_zbl_switch_) disp_l1ep_st(lm_tag, nb_tag, std::true_type{});
            else                 disp_l1ep_st(lm_tag, nb_tag, std::false_type{});
        };
        auto disp_l1ep_nb = [&](auto lm_tag) {
            if (use_nb16) disp_l1ep_zb(lm_tag, std::integral_constant<int, 16>{});
            else          disp_l1ep_zb(lm_tag, std::integral_constant<int, 32>{});
        };
        if (lmax_ <= 2) disp_l1ep_nb(std::integral_constant<int, 2>{});
        else            disp_l1ep_nb(std::integral_constant<int, 3>{});
    }
}

}  // namespace loup
