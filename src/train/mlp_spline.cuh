// 1D Cubic-Hermite Knot-Grid Radial MLP (LST_MLP_SPLINE=K, e.g. 256 or 512):
// Since the radial MLP is a smooth 1D function of scalar distance r in [0, r_max), we evaluate
// the 4-layer radial MLP (forward, backward, dW4) on only M_k = K + 1 uniform radial knots
// instead of all P = 7,890 (b1) or 31,560 (b4) edge pairs (~15x-120x less radial MLP work).
// Cubic Hermite interpolation of R(r) and its exact derivative T(r) = dR_hermite/dr are linear
// in (R_knot, T_knot), so:
//   - Forces are exactly conservative (F = -dE/dr);
//   - Accuracy at K=512 (h ~ 0.01 A): relR ~ 1e-9, relT ~ 2e-7 (below FP32 rounding);
//   - Backward weight gradients are the exact adjoint (transpose scatter onto gR_knot, gT_knot).
// Phase B (LST_MLP_SPLINE_FUSED=1, default): fuses Hermite lookup directly into k_conv_pair_sp
// and k_force_node_pair_sp, and adjoint scatter into k_slot_pair_sp, deleting k_spline_interp,
// k_spline_scatter, and all R/T/Rb/Tb DRAM traffic (~526 KB-1 MB L2-resident knot tables).
// Small-tile MLP (LST_MLP_SPLINE_SMALL=1, default): TP=8 fwd / TP=16 bwd fills all SMs on 257-513 knots.
#pragma once
#include <cuda_fp16.h>
#include <type_traits>
#include "act_half.cuh"

namespace lst {
namespace mspl {

// Find minimum nonzero edge distance across all frames' edge vectors (called once at set_frames for r_lo shift).
template <typename T>
__global__ void k_min_edge_r(const T* __restrict__ V, long E_tot, float* __restrict__ out_min) {
    __shared__ float sm[256];
    float best = 1e30f;
    for (long e = (long)blockIdx.x * blockDim.x + threadIdx.x; e < E_tot; e += (long)gridDim.x * blockDim.x) {
        const float vx = (float)V[3L * e], vy = (float)V[3L * e + 1], vz = (float)V[3L * e + 2];
        const float r2 = vx * vx + vy * vy + vz * vz;
        if (r2 > 1e-8f) {
            const float r = sqrtf(r2);
            if (r < best) best = r;
        }
    }
    sm[threadIdx.x] = best;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s && sm[threadIdx.x + s] < sm[threadIdx.x]) sm[threadIdx.x] = sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // Atomic min on positive float (IEEE 754 positive floats preserve order as int)
        atomicMin(reinterpret_cast<int*>(out_min), __float_as_int(sm[0]));
    }
}

// Precompute Bessel basis EF_k [M_k, NB], DEF_k [M_k, NB], and identity pe0_k [M_k] = 0..M_k-1 on uniform knots
// r_k = r_lo + (k / K) * (r_max - r_lo) for k = 0..K (M_k = K + 1).
template <typename T>
__global__ void k_knot_basis(T* __restrict__ EF_k, T* __restrict__ DEF_k, int* __restrict__ pe0_k,
                             const T* __restrict__ BW, T pref, T rmax, int p_cut, int NB, int K, T r_lo = T(0)) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > K) return;
    pe0_k[k] = k;
    const T r = r_lo + (T(k) / T(K)) * (rmax - r_lo);
    const T t = r / rmax;
    const T inside = k < K ? T(1) : T(0);
    const T Pc = T(p_cut);
    const T a = (Pc + 1) * (Pc + 2) / 2, b = Pc * (Pc + 2), c = Pc * (Pc + 1) / 2;
    T tp1 = t;
    for (int j = 0; j < p_cut - 2; ++j) tp1 *= t;
    const T tp = tp1 * t, tq = tp * t;
    const T env = (T(1) - a * tp + b * tq - c * tq * t) * inside;
    const T denv = (-a * Pc * tp1 + b * (Pc + 1) * tp - c * (Pc + 2) * tq) / rmax * inside;
    for (int n = 0; n < NB; ++n) {
        const T w = BW[n];
        T f, df;
        if (r <= T(0)) {
            f = pref * w;
            df = T(0);
        } else {
            const T rinv = T(1) / r;
            const T s = sin(w * r), co = cos(w * r);
            f = pref * s * rinv;
            df = pref * (w * co * rinv - s * rinv * rinv);
        }
        EF_k[(long)k * NB + n] = f * env;
        DEF_k[(long)k * NB + n] = df * env + f * denv;
    }
}

// Cubic Hermite basis weights (factored form: R = r1 + h00*(r0 - r1) + h10*t0 + h11*t1,
// T = d00*(r0 - r1) + d10*t0 + d11*t1; exact derivative and reduced FP32 cancellation).
template <typename T>
__device__ __forceinline__ void hermite_r(T u, T h, T& h00, T& h10, T& h11) {
    const T u2 = u * u, u3 = u2 * u;
    h00 = T(2) * u3 - T(3) * u2 + T(1);
    h10 = (u3 - T(2) * u2 + u) * h;
    h11 = (u3 - u2) * h;
}

template <typename T>
__device__ __forceinline__ void hermite_rt(T u, T h, T inv_h, T& h00, T& h10, T& h11, T& d00, T& d10, T& d11) {
    const T u2 = u * u, u3 = u2 * u;
    h00 = T(2) * u3 - T(3) * u2 + T(1);
    h10 = (u3 - T(2) * u2 + u) * h;
    h11 = (u3 - u2) * h;
    d00 = (T(6) * u2 - T(6) * u) * inv_h;
    d10 = T(3) * u2 - T(4) * u + T(1);
    d11 = T(3) * u2 - T(2) * u;
}

// Compute pair cubic-Hermite interval index and normalized u in [0, 1] from UP[4*p + 3] = 1/r.
// Returns false if r >= r_max (outside cutoff -> R = T = 0).
template <typename T>
__device__ __forceinline__ bool pair_knot(const T* __restrict__ UP, long p, T rmax, int K, int& idx, T& u, T& h,
                                          T& inv_h) {
    const T rinv = UP[4L * p + 3];
    const T r = T(1) / rinv;
    if (r >= rmax) return false;
    h = rmax / T(K);
    inv_h = T(K) / rmax;
    const T pos = r * inv_h;
    int i = (int)pos;
    if (i < 0) i = 0;
    if (i >= K) i = K - 1;
    idx = i;
    u = pos - T(i);
    return true;
}

// Phase A forward interpolation: R_out[p, o], T_out[p, o] from R_knot[M_k, O], T_knot[M_k, O].
template <typename T>
__global__ void __launch_bounds__(256) k_spline_interp(const T* __restrict__ UP, const T* __restrict__ R_k,
                                                       const T* __restrict__ T_k, T* __restrict__ R_out,
                                                       T* __restrict__ T_out, T rmax, int K, int P, int O) {
    const long p = blockIdx.x;
    const int o = threadIdx.x;
    if (p >= P || o >= O) return;
    int idx;
    T u, h, inv_h;
    if (!pair_knot<T>(UP, p, rmax, K, idx, u, h, inv_h)) {
        R_out[p * O + o] = T(0);
        T_out[p * O + o] = T(0);
        return;
    }
    T h00, h10, h11, d00, d10, d11;
    hermite_rt<T>(u, h, inv_h, h00, h10, h11, d00, d10, d11);
    const T r0 = R_k[(long)idx * O + o], r1 = R_k[(long)(idx + 1) * O + o];
    const T t0 = T_k[(long)idx * O + o], t1 = T_k[(long)(idx + 1) * O + o];
    const T dr = r0 - r1;
    R_out[p * O + o] = r1 + h00 * dr + h10 * t0 + h11 * t1;
    T_out[p * O + o] = d00 * dr + d10 * t0 + d11 * t1;
}

// Phase A backward adjoint scatter: (Rb_in[p, o], Tb_in[p, o]) -> atomicAdd into zeroed (gR_k, gT_k).
template <typename T>
__global__ void __launch_bounds__(256) k_spline_scatter(const T* __restrict__ UP, const T* __restrict__ Rb_in,
                                                        const T* __restrict__ Tb_in, T* __restrict__ gR_k,
                                                        T* __restrict__ gT_k, T rmax, int K, int P, int O) {
    const long p = blockIdx.x;
    const int o = threadIdx.x;
    if (p >= P || o >= O) return;
    int idx;
    T u, h, inv_h;
    if (!pair_knot<T>(UP, p, rmax, K, idx, u, h, inv_h)) return;
    T h00, h10, h11, d00, d10, d11;
    hermite_rt<T>(u, h, inv_h, h00, h10, h11, d00, d10, d11);
    const T rb = Rb_in[p * O + o];
    const T tb = Tb_in[p * O + o];
    atomicAdd(gR_k + (long)idx * O + o, h00 * rb + d00 * tb);
    atomicAdd(gR_k + (long)(idx + 1) * O + o, (T(1) - h00) * rb - d00 * tb);
    atomicAdd(gT_k + (long)idx * O + o, h10 * rb + d10 * tb);
    atomicAdd(gT_k + (long)(idx + 1) * O + o, h11 * rb + d11 * tb);
}

// Smooth C^2 per-(layer, l) radial shell cutoff envelope (Arch 1: Multi-Shell / Asymmetric Lmax MACE).
// rc < 0: disabled (f=1, df=0); rc == 0: kill degree l (f=0, df=0);
// sw <= 0: MACE p=5 polynomial cutoff over [0, rc];
// sw > 0: flat-top C^2 quintic switch (f=1 on [0, r_on], smooth 1->0 over [r_on, rc] where r_on = max(0, rc - sw))!
template <typename T>
__device__ __forceinline__ void shell_env(T r, T rc, T& f, T& df, T sw = T(0)) {
    if (rc < T(0)) { f = T(1); df = T(0); return; }
    if (rc <= T(0) || r >= rc) { f = T(0); df = T(0); return; }
    if (sw > T(0)) {
        const T r_on = rc > sw ? (rc - sw) : T(0);
        if (r <= r_on) { f = T(1); df = T(0); return; }
        const T inv_w = T(1) / (rc - r_on);
        const T u = (r - r_on) * inv_w;
        const T u2 = u * u, u3 = u2 * u, om = T(1) - u;
        f = T(1) - u3 * (T(10) - T(15) * u + T(6) * u2);
        df = T(-30) * inv_w * u2 * om * om;
        return;
    }
    const T inv_rc = T(1) / rc;
    const T x = r * inv_rc;
    const T x2 = x * x, x4 = x2 * x2, x5 = x4 * x, x6 = x5 * x, x7 = x6 * x;
    f = T(1) - T(21) * x5 + T(35) * x6 - T(15) * x7;
    const T omx = T(1) - x;
    df = T(-105) * inv_rc * x4 * omx * omx;
}

// Convert knot values/derivatives (R_k, T_k) [K+1, O] -> packed cubic polynomial coefficients
// C_k [K, O, 4] where R(u) = c0 + c1*u + c2*u^2 + c3*u^3 and T(u) = (c1 + 2*c2*u + 3*c3*u^2) * inv_h.
// Each (k, l, c) is a single 16-byte float4 (LDG.E.128 in FP32) -> 4x fewer radial loads in conv & force!
// Grid: K blocks of O (=256) threads.
template <typename T>
__global__ void __launch_bounds__(256) k_spline_rt_to_c(const T* __restrict__ R_k, const T* __restrict__ T_k,
                                                        T* __restrict__ C_k, T h, int K, int O,
                                                        float4 rcut4 = make_float4(-1.f, -1.f, -1.f, -1.f),
                                                        T r_lo = T(0), T sw = T(0)) {
    const int k = blockIdx.x;
    const int o = threadIdx.x;
    if (k >= K || o >= O) return;
    T r0 = R_k[(long)k * O + o], r1 = R_k[(long)(k + 1) * O + o];
    T t0 = T_k[(long)k * O + o], t1 = T_k[(long)(k + 1) * O + o];
    const int l = o / (O >> 2);
    const T rc = (l == 0) ? T(rcut4.x) : (l == 1 ? T(rcut4.y) : (l == 2 ? T(rcut4.z) : T(rcut4.w)));
    if (rc >= T(0)) {
        T f0, df0, f1, df1;
        shell_env<T>(r_lo + T(k) * h, rc, f0, df0, sw);
        shell_env<T>(r_lo + T(k + 1) * h, rc, f1, df1, sw);
        t0 = f0 * t0 + df0 * r0; r0 = f0 * r0;
        t1 = f1 * t1 + df1 * r1; r1 = f1 * r1;
    }
    const T dr = r1 - r0;
    const T c0 = r0;
    const T c1 = h * t0;
    const T c2 = T(3) * dr - h * (T(2) * t0 + t1);
    const T c3 = T(-2) * dr + h * (t0 + t1);
    const long idx = ((long)k * O + o) * 4;
    if constexpr (std::is_same<T, float>::value) {
        *reinterpret_cast<float4*>(C_k + idx) = make_float4(c0, c1, c2, c3);
    } else {
        C_k[idx + 0] = c0;
        C_k[idx + 1] = c1;
        C_k[idx + 2] = c2;
        C_k[idx + 3] = c3;
    }
}

// Exact transpose of k_spline_rt_to_c: converts accumulated polynomial gradients gC_k [K, 4, 4, C]
// (layout: ((k * 4 + l) * 4 + q) * C + c for warp-coalesced atomics) -> overwrites (gR_k, gT_k) [K+1, O].
// Grid: (K + 1) blocks of O (=256) threads.
template <typename T>
__global__ void __launch_bounds__(256) k_spline_c_to_rt_bwd(const T* __restrict__ gC_k, T* __restrict__ gR_k,
                                                            T* __restrict__ gT_k, T h, int K, int C,
                                                            float4 rcut4 = make_float4(-1.f, -1.f, -1.f, -1.f),
                                                            T r_lo = T(0), T sw = T(0)) {
    const int k = blockIdx.x;
    const int o = threadIdx.x;
    const int O = 4 * C;
    if (k > K || o >= O) return;
    const int l = o / C, c = o % C;
    T gr = T(0), gt = T(0);
    if (k < K) {
        const long base = ((long)k * 4 + l) * 4 * C + c;
        const T g0 = gC_k[base + 0 * C], g1 = gC_k[base + 1 * C];
        const T g2 = gC_k[base + 2 * C], g3 = gC_k[base + 3 * C];
        gr += g0 - T(3) * g2 + T(2) * g3;
        gt += h * (g1 - T(2) * g2 + g3);
    }
    if (k > 0) {
        const long base = ((long)(k - 1) * 4 + l) * 4 * C + c;
        const T g2 = gC_k[base + 2 * C], g3 = gC_k[base + 3 * C];
        gr += T(3) * g2 - T(2) * g3;
        gt += h * (-g2 + g3);
    }
    const T rc = (l == 0) ? T(rcut4.x) : (l == 1 ? T(rcut4.y) : (l == 2 ? T(rcut4.z) : T(rcut4.w)));
    if (rc >= T(0)) {
        T fk, dfk;
        shell_env<T>(r_lo + T(k) * h, rc, fk, dfk, sw);
        const T gr_raw = gr, gt_raw = gt;
        gr = fk * gr_raw + dfk * gt_raw;
        gt = fk * gt_raw;
    }
    gR_k[(long)k * O + o] = gr;
    gT_k[(long)k * O + o] = gt;
}

// Wave 34: Dual-layer spline c_to_rt backward (grid dim3(K+1, 2), block O): both layers in 1 launch!
template <typename T>
__global__ void __launch_bounds__(256) k_spline_c_to_rt_bwd_dual(
    const T* __restrict__ gC0, T* __restrict__ gR0, T* __restrict__ gT0, float4 rcut0,
    const T* __restrict__ gC1, T* __restrict__ gR1, T* __restrict__ gT1, float4 rcut1,
    T h, int K, int C, T r_lo = T(0), T sw = T(0)) {
    const int lyr = blockIdx.y;
    const T* __restrict__ gC_k = lyr ? gC1 : gC0;
    T* __restrict__ gR_k = lyr ? gR1 : gR0;
    T* __restrict__ gT_k = lyr ? gT1 : gT0;
    const float4 rcut4 = lyr ? rcut1 : rcut0;
    const int k = blockIdx.x;
    const int o = threadIdx.x;
    const int O = 4 * C;
    if (k > K || o >= O) return;
    const int l = o / C, c = o % C;
    T gr = T(0), gt = T(0);
    if (k < K) {
        const long base = ((long)k * 4 + l) * 4 * C + c;
        const T g0 = gC_k[base + 0 * C], g1 = gC_k[base + 1 * C];
        const T g2 = gC_k[base + 2 * C], g3 = gC_k[base + 3 * C];
        gr += g0 - T(3) * g2 + T(2) * g3;
        gt += h * (g1 - T(2) * g2 + g3);
    }
    if (k > 0) {
        const long base = ((long)(k - 1) * 4 + l) * 4 * C + c;
        const T g2 = gC_k[base + 2 * C], g3 = gC_k[base + 3 * C];
        gr += T(3) * g2 - T(2) * g3;
        gt += h * (-g2 + g3);
    }
    const T rc = (l == 0) ? T(rcut4.x) : (l == 1 ? T(rcut4.y) : (l == 2 ? T(rcut4.z) : T(rcut4.w)));
    if (rc >= T(0)) {
        T fk, dfk;
        shell_env<T>(r_lo + T(k) * h, rc, fk, dfk, sw);
        const T gr_raw = gr, gt_raw = gt;
        gr = fk * gr_raw + dfk * gt_raw;
        gt = fk * gt_raw;
    }
    gR_k[(long)k * O + o] = gr;
    gT_k[(long)k * O + o] = gt;
}

// Convert (R_k, T_k) [K+1, O] -> packed half4 (uint2, 8 B) cubic coefficients C_kh [K, O] (LST_SPLINE_HALF=1):
// halves L2 bandwidth (LDG.E.64 vs LDG.E.128) and fits K=64 entirely inside 128 KB L1 cache!
__global__ void __launch_bounds__(256) k_spline_rt_to_ch(const float* __restrict__ R_k, const float* __restrict__ T_k,
                                                         uint2* __restrict__ C_kh, float h, int K, int O,
                                                         float4 rcut4 = make_float4(-1.f, -1.f, -1.f, -1.f),
                                                         float r_lo = 0.f, float sw = 0.f) {
    const int k = blockIdx.x;
    const int o = threadIdx.x;
    if (k >= K || o >= O) return;
    float r0 = R_k[(long)k * O + o], r1 = R_k[(long)(k + 1) * O + o];
    float t0 = T_k[(long)k * O + o], t1 = T_k[(long)(k + 1) * O + o];
    const int l = o / (O >> 2);
    const float rc = (l == 0) ? rcut4.x : (l == 1 ? rcut4.y : (l == 2 ? rcut4.z : rcut4.w));
    if (rc >= 0.f) {
        float f0, df0, f1, df1;
        shell_env<float>(r_lo + float(k) * h, rc, f0, df0, sw);
        shell_env<float>(r_lo + float(k + 1) * h, rc, f1, df1, sw);
        t0 = f0 * t0 + df0 * r0; r0 = f0 * r0;
        t1 = f1 * t1 + df1 * r1; r1 = f1 * r1;
    }
    const float dr = r1 - r0;
    const float c0 = r0;
    const float c1 = h * t0;
    const float c2 = 3.f * dr - h * (2.f * t0 + t1);
    const float c3 = -2.f * dr + h * (t0 + t1);
    const __half2 h01 = __floats2half2_rn(c0, c1);
    const __half2 h23 = __floats2half2_rn(c2, c3);
    uint2 out;
    out.x = *reinterpret_cast<const unsigned int*>(&h01);
    out.y = *reinterpret_cast<const unsigned int*>(&h23);
    C_kh[(long)k * O + o] = out;
}

template <typename T, int POLY_MODE = 1>
__device__ __forceinline__ void ld_c4(const T* __restrict__ C_k, long elem_idx, T& c0, T& c1, T& c2, T& c3) {
    if constexpr (POLY_MODE == 2 && std::is_same<T, float>::value) {
        const uint2 v = reinterpret_cast<const uint2*>(C_k)[elem_idx];
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
        c0 = a.x; c1 = a.y; c2 = b.x; c3 = b.y;
    } else if constexpr (std::is_same<T, float>::value) {
        const float4 v = reinterpret_cast<const float4*>(C_k)[elem_idx];
        c0 = v.x; c1 = v.y; c2 = v.z; c3 = v.w;
    } else {
        const long off4 = elem_idx * 4;
        c0 = C_k[off4 + 0]; c1 = C_k[off4 + 1]; c2 = C_k[off4 + 2]; c3 = C_k[off4 + 3];
    }
}

// ============================================================================ Small-tile FP32 knot-grid MLP kernels
// Fills all SMs when P = M_k is small (257 or 513 knots):
//   - Forward:  TP_F = 8  (ROWS_F = 16, NTH_F = 128) -> 33 blocks at K=256, 65 at K=512; 20 KB smem, ~32 regs!
//   - Backward: TP_B = 16 (ROWS_B = 32, NTH_B = 256) -> 17 blocks at K=256, 33 at K=512; 34.8 KB smem, ~72 regs!
constexpr int SP_NB = 8, SP_H = 64, SP_O = 256;
constexpr int TP_F = 8, ROWS_F = 2 * TP_F, NTH_F = 128, LDF_S = 64;
constexpr int SMEM_FWD_SP = (ROWS_F * LDF_S + SP_H * SP_H) * 4;

constexpr int TP_B = 16, ROWS_B = 2 * TP_B, NTH_B = 256, LDB_S = 68;
constexpr int SMEM_BWD_SP = (2 * ROWS_B * LDB_S + SP_H * SP_H + ROWS_B * SP_NB) * 4;

__device__ __forceinline__ float sp_sig(float z) { return 1.f / (1.f + __expf(-z)); }

template <int TP_V, int K, int LDA>
__device__ __forceinline__ void gemm2x4(const float* __restrict__ A, const float* __restrict__ B, int rg, int cg,
                                        float acc[2][4]) {
#pragma unroll 4
    for (int k = 0; k < K; k += 4) {
        float4 b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) b[j] = *reinterpret_cast<const float4*>(B + (k + j) * 64 + 4 * cg);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int row = i == 0 ? rg : (TP_V + rg);
            const float4 a = *reinterpret_cast<const float4*>(A + row * LDA + k);
            acc[i][0] += a.x * b[0].x + a.y * b[1].x + a.z * b[2].x + a.w * b[3].x;
            acc[i][1] += a.x * b[0].y + a.y * b[1].y + a.z * b[2].y + a.w * b[3].y;
            acc[i][2] += a.x * b[0].z + a.y * b[1].z + a.z * b[2].z + a.w * b[3].z;
            acc[i][3] += a.x * b[0].w + a.y * b[1].w + a.z * b[2].w + a.w * b[3].w;
        }
    }
}

__device__ __forceinline__ void zero2x4(float acc[2][4]) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
}

__device__ __forceinline__ void act_fwd2x4(float z[2][4], float a) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float v = z[0][j], s = sp_sig(v);
        z[0][j] = a * v * s;
        z[1][j] = a * s * (1.f + v * (1.f - s)) * z[1][j];
    }
}

__device__ __forceinline__ void act_bwd2x4(const float z[2][4], float g[2][4], float a) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float v = z[0][j], dv = z[1][j], s = sp_sig(v);
        const float d1 = s * (1.f + v * (1.f - s));
        const float d2 = s * (1.f - s) * (2.f + v * (1.f - 2.f * s));
        const float gv = g[0][j], gt = g[1][j];
        g[0][j] = a * (d1 * gv + d2 * dv * gt);
        g[1][j] = a * d1 * gt;
    }
}

template <int TP_V, int LDA>
__device__ __forceinline__ void store_act2x4(float* S, const float z[2][4], float a, int rg, int cg) {
    float hv[4], ht[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float v = z[0][j], s = sp_sig(v);
        hv[j] = a * v * s;
        ht[j] = a * s * (1.f + v * (1.f - s)) * z[1][j];
    }
    *reinterpret_cast<float4*>(S + rg * LDA + 4 * cg) = make_float4(hv[0], hv[1], hv[2], hv[3]);
    *reinterpret_cast<float4*>(S + (TP_V + rg) * LDA + 4 * cg) = make_float4(ht[0], ht[1], ht[2], ht[3]);
}

template <int TP_V, int LDA>
__device__ __forceinline__ void store_tile2x4(float* S, const float v[2][4], int rg, int cg) {
    *reinterpret_cast<float4*>(S + rg * LDA + 4 * cg) = make_float4(v[0][0], v[0][1], v[0][2], v[0][3]);
    *reinterpret_cast<float4*>(S + (TP_V + rg) * LDA + 4 * cg) = make_float4(v[1][0], v[1][1], v[1][2], v[1][3]);
}

template <int NTH_V, int K>
__device__ __forceinline__ void load_w_sp(float* Ws, const float* __restrict__ W, int ldw, int c0) {
    for (int idx = threadIdx.x; idx < K * 16; idx += NTH_V) {
        const int k = idx >> 4, q = idx & 15;
        *reinterpret_cast<float4*>(Ws + k * 64 + 4 * q) =
            *reinterpret_cast<const float4*>(W + (long)k * ldw + c0 + 4 * q);
    }
}

// Small-tile forward (TP=8, NTH=128): direct knot indexing (no pe0 indirection).
__global__ void __launch_bounds__(NTH_F, 4) k_mlp_fwd_sp(const float* __restrict__ EF_k,
                                                         const float* __restrict__ DEF_k,
                                                         const float* __restrict__ W1, const float* __restrict__ W2,
                                                         const float* __restrict__ W3, const float* __restrict__ W4,
                                                         float* __restrict__ out_k, float* __restrict__ h3_k,
                                                         float act, int Pm) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Ws = As + ROWS_F * LDF_S;
    const int t = threadIdx.x;
    const int rg = t >> 4, cg = t & 15;
    const long p0 = (long)blockIdx.x * TP_F;
    if (t < ROWS_F * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_F - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_F ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(As + row * LDF_S + 4 * h4) = v;
    }
    load_w_sp<NTH_F, SP_NB>(Ws, W1, 64, 0);
    __syncthreads();
    float z[2][4];
    zero2x4(z);
    gemm2x4<TP_F, SP_NB, LDF_S>(As, Ws, rg, cg, z);
    act_fwd2x4(z, act);
    __syncthreads();
    store_tile2x4<TP_F, LDF_S>(As, z, rg, cg);
    load_w_sp<NTH_F, 64>(Ws, W2, 64, 0);
    __syncthreads();
    zero2x4(z);
    gemm2x4<TP_F, 64, LDF_S>(As, Ws, rg, cg, z);
    act_fwd2x4(z, act);
    __syncthreads();
    store_tile2x4<TP_F, LDF_S>(As, z, rg, cg);
    load_w_sp<NTH_F, 64>(Ws, W3, 64, 0);
    __syncthreads();
    zero2x4(z);
    gemm2x4<TP_F, 64, LDF_S>(As, Ws, rg, cg, z);
    act_fwd2x4(z, act);
    __syncthreads();
    store_tile2x4<TP_F, LDF_S>(As, z, rg, cg);
    const long p = p0 + rg;
    if (h3_k && p < Pm) {
        *reinterpret_cast<float4*>(h3_k + p * SP_H + 4 * cg) = make_float4(z[0][0], z[0][1], z[0][2], z[0][3]);
        *reinterpret_cast<float4*>(h3_k + (Pm + p) * SP_H + 4 * cg) = make_float4(z[1][0], z[1][1], z[1][2], z[1][3]);
    }
    for (int q = 0; q < SP_O / 64; ++q) {
        if (q) __syncthreads();
        load_w_sp<NTH_F, 64>(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        zero2x4(z);
        gemm2x4<TP_F, 64, LDF_S>(As, Ws, rg, cg, z);
        if (p < Pm) {
            *reinterpret_cast<float4*>(out_k + p * SP_O + 64 * q + 4 * cg) =
                make_float4(z[0][0], z[0][1], z[0][2], z[0][3]);
            *reinterpret_cast<float4*>(out_k + (Pm + p) * SP_O + 64 * q + 4 * cg) =
                make_float4(z[1][0], z[1][1], z[1][2], z[1][3]);
        }
    }
}

// Wave 25 (LST_MLP_DUAL_SP=1): batched 2-layer exact FP32 spline MLP forward (blockIdx.y = 0 or 1).
// Doubles SM utilization (e.g. 9 -> 18 blocks at K=64) in 1 launch with bitwise-identical FP32 math!
__global__ void __launch_bounds__(NTH_F, 4) k_mlp_fwd_sp2(
    const float* __restrict__ EF_k, const float* __restrict__ DEF_k,
    const float* __restrict__ W1_0, const float* __restrict__ W2_0, const float* __restrict__ W3_0,
    const float* __restrict__ W4_0, float* __restrict__ out_k0, float* __restrict__ h3_k0,
    const float* __restrict__ W1_1, const float* __restrict__ W2_1, const float* __restrict__ W3_1,
    const float* __restrict__ W4_1, float* __restrict__ out_k1, float* __restrict__ h3_k1,
    float act, int Pm) {
    const bool l1 = (blockIdx.y != 0);
    const float* W1 = l1 ? W1_1 : W1_0;
    const float* W2 = l1 ? W2_1 : W2_0;
    const float* W3 = l1 ? W3_1 : W3_0;
    const float* W4 = l1 ? W4_1 : W4_0;
    float* out_k = l1 ? out_k1 : out_k0;
    float* h3_k = l1 ? h3_k1 : h3_k0;
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Ws = As + ROWS_F * LDF_S;
    const int t = threadIdx.x;
    const int rg = t >> 4, cg = t & 15;
    const long p0 = (long)blockIdx.x * TP_F;
    if (t < ROWS_F * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_F - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_F ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(As + row * LDF_S + 4 * h4) = v;
    }
    load_w_sp<NTH_F, SP_NB>(Ws, W1, 64, 0);
    __syncthreads();
    float z[2][4];
    zero2x4(z);
    gemm2x4<TP_F, SP_NB, LDF_S>(As, Ws, rg, cg, z);
    act_fwd2x4(z, act);
    __syncthreads();
    store_tile2x4<TP_F, LDF_S>(As, z, rg, cg);
    load_w_sp<NTH_F, 64>(Ws, W2, 64, 0);
    __syncthreads();
    zero2x4(z);
    gemm2x4<TP_F, 64, LDF_S>(As, Ws, rg, cg, z);
    act_fwd2x4(z, act);
    __syncthreads();
    store_tile2x4<TP_F, LDF_S>(As, z, rg, cg);
    load_w_sp<NTH_F, 64>(Ws, W3, 64, 0);
    __syncthreads();
    zero2x4(z);
    gemm2x4<TP_F, 64, LDF_S>(As, Ws, rg, cg, z);
    act_fwd2x4(z, act);
    __syncthreads();
    store_tile2x4<TP_F, LDF_S>(As, z, rg, cg);
    const long p = p0 + rg;
    if (h3_k && p < Pm) {
        *reinterpret_cast<float4*>(h3_k + p * SP_H + 4 * cg) = make_float4(z[0][0], z[0][1], z[0][2], z[0][3]);
        *reinterpret_cast<float4*>(h3_k + (Pm + p) * SP_H + 4 * cg) = make_float4(z[1][0], z[1][1], z[1][2], z[1][3]);
    }
    for (int q = 0; q < SP_O / 64; ++q) {
        if (q) __syncthreads();
        load_w_sp<NTH_F, 64>(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        zero2x4(z);
        gemm2x4<TP_F, 64, LDF_S>(As, Ws, rg, cg, z);
        if (p < Pm) {
            *reinterpret_cast<float4*>(out_k + p * SP_O + 64 * q + 4 * cg) =
                make_float4(z[0][0], z[0][1], z[0][2], z[0][3]);
            *reinterpret_cast<float4*>(out_k + (Pm + p) * SP_O + 64 * q + 4 * cg) =
                make_float4(z[1][0], z[1][1], z[1][2], z[1][3]);
        }
    }
}

// Small-tile backward (TP=16, NTH=256): one tile per block (17 blocks at K=256, 33 at K=512).
// Low register count (~72 regs) + 34.8 KB smem -> 2 blocks/SM on sm_120 and sm_89!
__global__ void __launch_bounds__(NTH_B, 2) k_mlp_bwd_sp(const float* __restrict__ EF_k,
                                                         const float* __restrict__ DEF_k,
                                                         const float* __restrict__ W1, const float* __restrict__ W2,
                                                         const float* __restrict__ W3, const float* __restrict__ WT,
                                                         const float* __restrict__ gout_k, float* dW1, float* dW2,
                                                         float* dW3, float act, int Pm) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS_B * LDB_S;
    float* Ws = Bs + ROWS_B * LDB_S;
    float* Xs = Ws + SP_H * SP_H;
    const int t = threadIdx.x;
    const int rg = t >> 4, cg = t & 15;
    const long p0 = (long)blockIdx.x * TP_B;
    const float* WT4 = WT + 2 * SP_H * SP_H;
    // Load Xs [32][8] and W1 [8][64]
    if (t < ROWS_B * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_B - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_B ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(Xs + row * SP_NB + 4 * h4) = v;
    }
    load_w_sp<NTH_B, SP_NB>(Ws, W1, 64, 0);
    __syncthreads();
    // Forward recompute keeping z1, z2, z3 (only 24 floats in registers!)
    float z1[2][4], z2[2][4], z3[2][4], g[2][4];
    zero2x4(z1);
    gemm2x4<TP_B, SP_NB, SP_NB>(Xs, Ws, rg, cg, z1);
    store_act2x4<TP_B, LDB_S>(As, z1, act, rg, cg);  // h1
    __syncthreads();
    load_w_sp<NTH_B, 64>(Ws, W2, 64, 0);
    __syncthreads();
    zero2x4(z2);
    gemm2x4<TP_B, 64, LDB_S>(As, Ws, rg, cg, z2);
    __syncthreads();
    store_act2x4<TP_B, LDB_S>(As, z2, act, rg, cg);  // h2
    load_w_sp<NTH_B, 64>(Ws, W3, 64, 0);
    __syncthreads();
    zero2x4(z3);
    gemm2x4<TP_B, 64, LDB_S>(As, Ws, rg, cg, z3);
    // gh3 = gout_k @ WT4 (4 slices of 64); As still holds h2
    zero2x4(g);
#pragma unroll
    for (int q = 0; q < SP_O / 64; ++q) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < ROWS_B * 16 / NTH_B; ++i) {
            const int idx = t + i * NTH_B;
            const int row = idx >> 4, k4 = idx & 15;
            const long p = p0 + (row & (TP_B - 1));
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (p < Pm) {
                const long grow = row < TP_B ? p : (Pm + p);
                v = *reinterpret_cast<const float4*>(gout_k + grow * SP_O + 64 * q + 4 * k4);
            }
            *reinterpret_cast<float4*>(Bs + row * LDB_S + 4 * k4) = v;
        }
        load_w_sp<NTH_B, 64>(Ws, WT4 + 64 * q * SP_H, 64, 0);
        __syncthreads();
        gemm2x4<TP_B, 64, LDB_S>(Bs, Ws, rg, cg, g);
    }
    act_bwd2x4(z3, g, act);  // gz3
    __syncthreads();
    store_tile2x4<TP_B, LDB_S>(Bs, g, rg, cg);
    load_w_sp<NTH_B, 64>(Ws, WT + SP_H * SP_H, 64, 0);  // W3^T
    __syncthreads();
    float d3[4][4], d2[4][4];
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) d3[u][v] = d2[u][v] = 0.f;
#pragma unroll 4
    for (int r = 0; r < ROWS_B; ++r) {
        const float4 a = *reinterpret_cast<const float4*>(As + r * LDB_S + 4 * rg);
        const float4 b = *reinterpret_cast<const float4*>(Bs + r * LDB_S + 4 * cg);
        const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) d3[u][v] += av[u] * bv[v];
    }
    zero2x4(g);
    gemm2x4<TP_B, 64, LDB_S>(Bs, Ws, rg, cg, g);  // gh2
    act_bwd2x4(z2, g, act);                       // gz2
    __syncthreads();
    store_tile2x4<TP_B, LDB_S>(Bs, g, rg, cg);
    store_act2x4<TP_B, LDB_S>(As, z1, act, rg, cg);  // h1
    load_w_sp<NTH_B, 64>(Ws, WT, 64, 0);             // W2^T
    __syncthreads();
#pragma unroll 4
    for (int r = 0; r < ROWS_B; ++r) {
        const float4 a = *reinterpret_cast<const float4*>(As + r * LDB_S + 4 * rg);
        const float4 b = *reinterpret_cast<const float4*>(Bs + r * LDB_S + 4 * cg);
        const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) d2[u][v] += av[u] * bv[v];
    }
    zero2x4(g);
    gemm2x4<TP_B, 64, LDB_S>(Bs, Ws, rg, cg, g);  // gh1
    act_bwd2x4(z1, g, act);                       // gz1
    __syncthreads();
    store_tile2x4<TP_B, LDB_S>(Bs, g, rg, cg);
    __syncthreads();
    const int a = t >> 5, b0 = 2 * (t & 31);
    float d10 = 0.f, d11 = 0.f;
#pragma unroll 4
    for (int r = 0; r < ROWS_B; ++r) {
        const float xa = Xs[r * SP_NB + a];
        d10 += xa * Bs[r * LDB_S + b0];
        d11 += xa * Bs[r * LDB_S + b0 + 1];
    }
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            atomicAdd(dW3 + (4 * rg + u) * 64 + 4 * cg + v, d3[u][v]);
            atomicAdd(dW2 + (4 * rg + u) * 64 + 4 * cg + v, d2[u][v]);
        }
    atomicAdd(dW1 + a * 64 + b0, d10);
    atomicAdd(dW1 + a * 64 + b0 + 1, d11);
}

// ============================================================================ Tiny-tile FP32 knot-grid MLP (LST_MLP_TINY=1)
// Parameterised by TP_V in {2, 4, 8} with NTH_T = 128 threads so K=64 (TP=2), K=128 (TP=4), K=256 (TP=8) all launch
// 33 blocks (full SM utilisation vs 5-9 blocks before). Backward also fuses dW4 = H3^T @ gout inside the q-loop and
// loads W2/W3/W4 transposed directly in shared memory, deleting 2x cutlass, 2x splitKreduce, and 2x k_mlp_wt per step!
constexpr int NTH_T = 128, LDT_S = 68;
template <int TP_V>
constexpr int smem_fwd_sp_t() { return (2 * TP_V * LDT_S + SP_H * LDT_S) * 4; }
template <int TP_V>
constexpr int smem_bwd_sp_t() { return (3 * 2 * TP_V * LDT_S + SP_H * LDT_S + 2 * TP_V * SP_NB) * 4; }

template <int K>
__device__ __forceinline__ void load_w_t(float* Ws, const float* __restrict__ W, int ldw, int c0) {
    for (int idx = threadIdx.x; idx < K * 16; idx += NTH_T) {
        const int k = idx >> 4, q = idx & 15;
        *reinterpret_cast<float4*>(Ws + k * LDT_S + 4 * q) =
            *reinterpret_cast<const float4*>(W + (long)k * ldw + c0 + 4 * q);
    }
}

// Load W[0..63, c0..c0+63] transposed into Ws[k_in * LDT_S + n_out] = W[n_out * ldw + c0 + k_in] (coalesced float4!).
__device__ __forceinline__ void load_w_trans_t(float* Ws, const float* __restrict__ W, int ldw, int c0) {
    for (int idx = threadIdx.x; idx < 64 * 16; idx += NTH_T) {
        const int n_out = idx >> 4, k4 = (idx & 15) * 4;
        const float4 v = *reinterpret_cast<const float4*>(W + (long)n_out * ldw + c0 + k4);
        Ws[(k4 + 0) * LDT_S + n_out] = v.x;
        Ws[(k4 + 1) * LDT_S + n_out] = v.y;
        Ws[(k4 + 2) * LDT_S + n_out] = v.z;
        Ws[(k4 + 3) * LDT_S + n_out] = v.w;
    }
}

template <int TP_V, int NCOL, int K, int LDA>
__device__ __forceinline__ void gemm_t(const float* __restrict__ A, const float* __restrict__ Ws, int rg, int col0,
                                       float acc[2][NCOL]) {
#pragma unroll 4
    for (int k = 0; k < K; k += 4) {
        const float4 a0 = *reinterpret_cast<const float4*>(A + rg * LDA + k);
        const float4 a1 = *reinterpret_cast<const float4*>(A + (TP_V + rg) * LDA + k);
#pragma unroll
        for (int j = 0; j < NCOL; ++j) {
            const int col = col0 + j;
            const float b0 = Ws[(k + 0) * LDT_S + col];
            const float b1 = Ws[(k + 1) * LDT_S + col];
            const float b2 = Ws[(k + 2) * LDT_S + col];
            const float b3 = Ws[(k + 3) * LDT_S + col];
            acc[0][j] += a0.x * b0 + a0.y * b1 + a0.z * b2 + a0.w * b3;
            acc[1][j] += a1.x * b0 + a1.y * b1 + a1.z * b2 + a1.w * b3;
        }
    }
}

template <int NCOL>
__device__ __forceinline__ void zero_t(float acc[2][NCOL]) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NCOL; ++j) acc[i][j] = 0.f;
}

template <int NCOL>
__device__ __forceinline__ void act_fwd_t(float z[2][NCOL], float a) {
#pragma unroll
    for (int j = 0; j < NCOL; ++j) {
        const float v = z[0][j], s = sp_sig(v);
        z[0][j] = a * v * s;
        z[1][j] = a * s * (1.f + v * (1.f - s)) * z[1][j];
    }
}

template <int NCOL>
__device__ __forceinline__ void act_bwd_t(const float z[2][NCOL], float g[2][NCOL], float a) {
#pragma unroll
    for (int j = 0; j < NCOL; ++j) {
        const float v = z[0][j], dv = z[1][j], s = sp_sig(v);
        const float d1 = s * (1.f + v * (1.f - s));
        const float d2 = s * (1.f - s) * (2.f + v * (1.f - 2.f * s));
        const float gv = g[0][j], gt = g[1][j];
        g[0][j] = a * (d1 * gv + d2 * dv * gt);
        g[1][j] = a * d1 * gt;
    }
}

template <int TP_V, int NCOL>
__device__ __forceinline__ void store_tile_t(float* S, const float v[2][NCOL], int rg, int col0) {
#pragma unroll
    for (int j = 0; j < NCOL; ++j) {
        S[rg * LDT_S + col0 + j] = v[0][j];
        S[(TP_V + rg) * LDT_S + col0 + j] = v[1][j];
    }
}

template <int TP_V, int NCOL>
__device__ __forceinline__ void store_act_t(float* S, const float z[2][NCOL], float a, int rg, int col0) {
#pragma unroll
    for (int j = 0; j < NCOL; ++j) {
        const float v = z[0][j], s = sp_sig(v);
        S[rg * LDT_S + col0 + j] = a * v * s;
        S[(TP_V + rg) * LDT_S + col0 + j] = a * s * (1.f + v * (1.f - s)) * z[1][j];
    }
}

// Compute dW[64, 64] += A^T @ B (A, B are [ROWS, LDT_S]) across 128 threads (8x4 outputs/thread) and atomicAdd into dW.
template <int ROWS>
__device__ __forceinline__ void accum_dw64_t(const float* __restrict__ A, const float* __restrict__ B, float* dW,
                                             int ldw_out, int c0_out) {
    const int t = threadIdx.x;
    const int ty = t >> 4, tx = t & 15;   // 8 rows (8*ty..+7), 4 cols (4*tx..+3)
    float d[8][4];
#pragma unroll
    for (int u = 0; u < 8; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) d[u][v] = 0.f;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float4 a0 = *reinterpret_cast<const float4*>(A + r * LDT_S + 8 * ty);
        const float4 a1 = *reinterpret_cast<const float4*>(A + r * LDT_S + 8 * ty + 4);
        const float4 b = *reinterpret_cast<const float4*>(B + r * LDT_S + 4 * tx);
        const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (int u = 0; u < 8; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) d[u][v] += av[u] * bv[v];
    }
#pragma unroll
    for (int u = 0; u < 8; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) atomicAdd(dW + (long)(8 * ty + u) * ldw_out + c0_out + 4 * tx + v, d[u][v]);
}

template <int TP_V>
__global__ void __launch_bounds__(NTH_T, 4) k_mlp_fwd_sp_t(const float* __restrict__ EF_k,
                                                           const float* __restrict__ DEF_k,
                                                           const float* __restrict__ W1, const float* __restrict__ W2,
                                                           const float* __restrict__ W3, const float* __restrict__ W4,
                                                           float* __restrict__ out_k, float* __restrict__ h3_k,
                                                           float act, int Pm) {
    constexpr int ROWS = 2 * TP_V;
    constexpr int NCOL = TP_V / 2;
    constexpr int NG = SP_H / NCOL;
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Ws = As + ROWS * LDT_S;
    const int t = threadIdx.x;
    const int rg = t / NG, col0 = (t % NG) * NCOL;
    const long p0 = (long)blockIdx.x * TP_V;
    if (t < ROWS * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_V - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_V ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(As + row * LDT_S + 4 * h4) = v;
    }
    load_w_t<SP_NB>(Ws, W1, 64, 0);
    __syncthreads();
    float z[2][NCOL];
    zero_t<NCOL>(z);
    gemm_t<TP_V, NCOL, SP_NB, LDT_S>(As, Ws, rg, col0, z);
    act_fwd_t<NCOL>(z, act);
    __syncthreads();
    store_tile_t<TP_V, NCOL>(As, z, rg, col0);
    load_w_t<64>(Ws, W2, 64, 0);
    __syncthreads();
    zero_t<NCOL>(z);
    gemm_t<TP_V, NCOL, 64, LDT_S>(As, Ws, rg, col0, z);
    act_fwd_t<NCOL>(z, act);
    __syncthreads();
    store_tile_t<TP_V, NCOL>(As, z, rg, col0);
    load_w_t<64>(Ws, W3, 64, 0);
    __syncthreads();
    zero_t<NCOL>(z);
    gemm_t<TP_V, NCOL, 64, LDT_S>(As, Ws, rg, col0, z);
    act_fwd_t<NCOL>(z, act);
    __syncthreads();
    store_tile_t<TP_V, NCOL>(As, z, rg, col0);
    const long p = p0 + rg;
    if (h3_k && p < Pm) {
#pragma unroll
        for (int j = 0; j < NCOL; ++j) {
            h3_k[p * SP_H + col0 + j] = z[0][j];
            h3_k[(Pm + p) * SP_H + col0 + j] = z[1][j];
        }
    }
    for (int q = 0; q < SP_O / 64; ++q) {
        if (q) __syncthreads();
        load_w_t<64>(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        zero_t<NCOL>(z);
        gemm_t<TP_V, NCOL, 64, LDT_S>(As, Ws, rg, col0, z);
        if (p < Pm) {
#pragma unroll
            for (int j = 0; j < NCOL; ++j) {
                out_k[p * SP_O + 64 * q + col0 + j] = z[0][j];
                out_k[(Pm + p) * SP_O + 64 * q + col0 + j] = z[1][j];
            }
        }
    }
}

template <int TP_V, bool FUSE_CRT = false>
__global__ void __launch_bounds__(NTH_T, 3) k_mlp_bwd_sp_t(const float* __restrict__ EF_k,
                                                           const float* __restrict__ DEF_k,
                                                           const float* __restrict__ W1, const float* __restrict__ W2,
                                                           const float* __restrict__ W3, const float* __restrict__ W4,
                                                           const float* __restrict__ gout_k, float* dW1, float* dW2,
                                                           float* dW3, float* dW4, float act, int Pm,
                                                           float sp_h = 0.f) {
    constexpr int ROWS = 2 * TP_V;
    constexpr int NCOL = TP_V / 2;
    constexpr int NG = SP_H / NCOL;
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS * LDT_S;
    float* H3s = Bs + ROWS * LDT_S;
    float* Ws = H3s + ROWS * LDT_S;
    float* Xs = Ws + SP_H * LDT_S;
    const int t = threadIdx.x;
    const int rg = t / NG, col0 = (t % NG) * NCOL;
    const long p0 = (long)blockIdx.x * TP_V;
    if (t < ROWS * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_V - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_V ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(Xs + row * SP_NB + 4 * h4) = v;
    }
    load_w_t<SP_NB>(Ws, W1, 64, 0);
    __syncthreads();
    float z1[2][NCOL], z2[2][NCOL], z3[2][NCOL], g[2][NCOL];
    zero_t<NCOL>(z1);
    gemm_t<TP_V, NCOL, SP_NB, SP_NB>(Xs, Ws, rg, col0, z1);
    store_act_t<TP_V, NCOL>(As, z1, act, rg, col0);  // h1
    __syncthreads();
    load_w_t<64>(Ws, W2, 64, 0);
    __syncthreads();
    zero_t<NCOL>(z2);
    gemm_t<TP_V, NCOL, 64, LDT_S>(As, Ws, rg, col0, z2);
    __syncthreads();
    store_act_t<TP_V, NCOL>(As, z2, act, rg, col0);  // h2
    load_w_t<64>(Ws, W3, 64, 0);
    __syncthreads();
    zero_t<NCOL>(z3);
    gemm_t<TP_V, NCOL, 64, LDT_S>(As, Ws, rg, col0, z3);
    store_act_t<TP_V, NCOL>(H3s, z3, act, rg, col0); // h3 (zero for padded rows since z3=0)
    // gh3 = gout_k @ W4^T AND fused dW4 += H3s^T @ gout_k (As still holds h2)
    zero_t<NCOL>(g);
    const int Ksp = Pm - 1;
#pragma unroll
    for (int q = 0; q < SP_O / 64; ++q) {
        __syncthreads();
        if constexpr (FUSE_CRT) {
            const int c = t & 63, ph = t >> 6;
#pragma unroll
            for (int r = ph; r < TP_V; r += 2) {
                const long p = p0 + r;
                float gr = 0.f, gt = 0.f;
                if (p < Pm) {
                    if (p < Ksp) {
                        const long base = (p * 4 + q) * 4 * 64 + c;
                        const float g0 = gout_k[base + 0 * 64], g1 = gout_k[base + 1 * 64];
                        const float g2 = gout_k[base + 2 * 64], g3 = gout_k[base + 3 * 64];
                        gr += g0 - 3.f * g2 + 2.f * g3;
                        gt += sp_h * (g1 - 2.f * g2 + g3);
                    }
                    if (p > 0) {
                        const long base = ((p - 1) * 4 + q) * 4 * 64 + c;
                        const float g2 = gout_k[base + 2 * 64], g3 = gout_k[base + 3 * 64];
                        gr += 3.f * g2 - 2.f * g3;
                        gt += sp_h * (-g2 + g3);
                    }
                }
                Bs[r * LDT_S + c] = gr;
                Bs[(TP_V + r) * LDT_S + c] = gt;
            }
        } else {
            for (int idx = t; idx < ROWS * 16; idx += NTH_T) {
                const int row = idx >> 4, k4 = idx & 15;
                const long p = p0 + (row & (TP_V - 1));
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (p < Pm) {
                    const long grow = row < TP_V ? p : (Pm + p);
                    v = *reinterpret_cast<const float4*>(gout_k + grow * SP_O + 64 * q + 4 * k4);
                }
                *reinterpret_cast<float4*>(Bs + row * LDT_S + 4 * k4) = v;
            }
        }
        load_w_trans_t(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        gemm_t<TP_V, NCOL, 64, LDT_S>(Bs, Ws, rg, col0, g);
        accum_dw64_t<ROWS>(H3s, Bs, dW4, SP_O, 64 * q);
    }
    act_bwd_t<NCOL>(z3, g, act);  // gz3
    __syncthreads();
    store_tile_t<TP_V, NCOL>(Bs, g, rg, col0);
    load_w_trans_t(Ws, W3, 64, 0);
    __syncthreads();
    accum_dw64_t<ROWS>(As, Bs, dW3, 64, 0);  // dW3 += h2^T gz3
    zero_t<NCOL>(g);
    gemm_t<TP_V, NCOL, 64, LDT_S>(Bs, Ws, rg, col0, g);  // gh2
    act_bwd_t<NCOL>(z2, g, act);                         // gz2
    __syncthreads();
    store_tile_t<TP_V, NCOL>(Bs, g, rg, col0);
    store_act_t<TP_V, NCOL>(As, z1, act, rg, col0);      // h1
    load_w_trans_t(Ws, W2, 64, 0);
    __syncthreads();
    accum_dw64_t<ROWS>(As, Bs, dW2, 64, 0);  // dW2 += h1^T gz2
    zero_t<NCOL>(g);
    gemm_t<TP_V, NCOL, 64, LDT_S>(Bs, Ws, rg, col0, g);  // gh1
    act_bwd_t<NCOL>(z1, g, act);                         // gz1
    __syncthreads();
    store_tile_t<TP_V, NCOL>(Bs, g, rg, col0);
    __syncthreads();
    const int a = t >> 4, b0 = 4 * (t & 15);
    float d1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float xa = Xs[r * SP_NB + a];
        const float4 bv = *reinterpret_cast<const float4*>(Bs + r * LDT_S + b0);
        d1[0] += xa * bv.x; d1[1] += xa * bv.y; d1[2] += xa * bv.z; d1[3] += xa * bv.w;
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) atomicAdd(dW1 + a * 64 + b0 + u, d1[u]);
}

// ============================================================================ Wave 14.1: High-occupancy TP=4 spline MLP
// (LST_MLP_SP4=1): fills 33 blocks at K=128 / 17 at K=64 (vs 9 / 5 for k_mlp_bwd_sp!), uses float4 LDS.128 on BOTH A
// and Ws (stride 68, zero bank conflicts), loads W2/W3/W4 transposed directly in smem (deletes k_mlp_wt!), and pairs
// with dedicated non-atomic k_mlp_dw4 (32 blocks, 0 cutlass, 0 splitKreduce, 0 dW4 atomics!).
constexpr int TP_4 = 4, ROWS_4 = 2 * TP_4, NTH_4 = 64, LD4_S = 68;
constexpr int SMEM_FWD_SP4 = (ROWS_4 * LD4_S + SP_H * LD4_S) * 4;
constexpr int SMEM_BWD_SP4 = (2 * ROWS_4 * LD4_S + SP_H * LD4_S + ROWS_4 * SP_NB) * 4;

template <int K>
__device__ __forceinline__ void load_w4(float* Ws, const float* __restrict__ W, int ldw, int c0) {
    for (int idx = threadIdx.x; idx < K * 16; idx += NTH_4) {
        const int k = idx >> 4, q = idx & 15;
        *reinterpret_cast<float4*>(Ws + k * LD4_S + 4 * q) =
            *reinterpret_cast<const float4*>(W + (long)k * ldw + c0 + 4 * q);
    }
}

__device__ __forceinline__ void load_w4_trans(float* Ws, const float* __restrict__ W, int ldw, int c0) {
    for (int idx = threadIdx.x; idx < 64 * 16; idx += NTH_4) {
        const int n_out = idx >> 4, k4 = (idx & 15) * 4;
        const float4 v = *reinterpret_cast<const float4*>(W + (long)n_out * ldw + c0 + k4);
        Ws[(k4 + 0) * LD4_S + n_out] = v.x;
        Ws[(k4 + 1) * LD4_S + n_out] = v.y;
        Ws[(k4 + 2) * LD4_S + n_out] = v.z;
        Ws[(k4 + 3) * LD4_S + n_out] = v.w;
    }
}

template <int K, int LDA>
__device__ __forceinline__ void gemm2x4_68(const float* __restrict__ A, const float* __restrict__ B, int rg, int cg,
                                           float acc[2][4]) {
#pragma unroll 4
    for (int k = 0; k < K; k += 4) {
        float4 b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) b[j] = *reinterpret_cast<const float4*>(B + (k + j) * LD4_S + 4 * cg);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int row = i == 0 ? rg : (TP_4 + rg);
            const float4 a = *reinterpret_cast<const float4*>(A + row * LDA + k);
            acc[i][0] += a.x * b[0].x + a.y * b[1].x + a.z * b[2].x + a.w * b[3].x;
            acc[i][1] += a.x * b[0].y + a.y * b[1].y + a.z * b[2].y + a.w * b[3].y;
            acc[i][2] += a.x * b[0].z + a.y * b[1].z + a.z * b[2].z + a.w * b[3].z;
            acc[i][3] += a.x * b[0].w + a.y * b[1].w + a.z * b[2].w + a.w * b[3].w;
        }
    }
}

// Forward TP=4, NTH=64: 33 blocks at K=128, 17 at K=64; 19.6 KB smem, float4 LDS.128!
__global__ void __launch_bounds__(NTH_4, 4) k_mlp_fwd_sp4(const float* __restrict__ EF_k,
                                                          const float* __restrict__ DEF_k,
                                                          const float* __restrict__ W1, const float* __restrict__ W2,
                                                          const float* __restrict__ W3, const float* __restrict__ W4,
                                                          float* __restrict__ out_k, float* __restrict__ h3_k,
                                                          float act, int Pm) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Ws = As + ROWS_4 * LD4_S;
    const int t = threadIdx.x;
    const int rg = t >> 4, cg = t & 15;
    const long p0 = (long)blockIdx.x * TP_4;
    if (t < ROWS_4 * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_4 - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_4 ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(As + row * LD4_S + 4 * h4) = v;
    }
    load_w4<SP_NB>(Ws, W1, 64, 0);
    __syncthreads();
    float z[2][4];
    zero2x4(z);
    gemm2x4_68<SP_NB, LD4_S>(As, Ws, rg, cg, z);
    act_fwd2x4(z, act);
    __syncthreads();
    store_tile2x4<TP_4, LD4_S>(As, z, rg, cg);
    load_w4<64>(Ws, W2, 64, 0);
    __syncthreads();
    zero2x4(z);
    gemm2x4_68<64, LD4_S>(As, Ws, rg, cg, z);
    act_fwd2x4(z, act);
    __syncthreads();
    store_tile2x4<TP_4, LD4_S>(As, z, rg, cg);
    load_w4<64>(Ws, W3, 64, 0);
    __syncthreads();
    zero2x4(z);
    gemm2x4_68<64, LD4_S>(As, Ws, rg, cg, z);
    act_fwd2x4(z, act);
    __syncthreads();
    store_tile2x4<TP_4, LD4_S>(As, z, rg, cg);
    const long p = p0 + rg;
    if (h3_k && p < Pm) {
        *reinterpret_cast<float4*>(h3_k + p * SP_H + 4 * cg) = make_float4(z[0][0], z[0][1], z[0][2], z[0][3]);
        *reinterpret_cast<float4*>(h3_k + (Pm + p) * SP_H + 4 * cg) = make_float4(z[1][0], z[1][1], z[1][2], z[1][3]);
    }
    for (int q = 0; q < SP_O / 64; ++q) {
        if (q) __syncthreads();
        load_w4<64>(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        zero2x4(z);
        gemm2x4_68<64, LD4_S>(As, Ws, rg, cg, z);
        if (p < Pm) {
            *reinterpret_cast<float4*>(out_k + p * SP_O + 64 * q + 4 * cg) =
                make_float4(z[0][0], z[0][1], z[0][2], z[0][3]);
            *reinterpret_cast<float4*>(out_k + (Pm + p) * SP_O + 64 * q + 4 * cg) =
                make_float4(z[1][0], z[1][1], z[1][2], z[1][3]);
        }
    }
}

// Accumulate dW[64, 64] += A^T @ B (A, B are [8, 68]) with 64 threads in 4 row-passes (16 regs only!).
__device__ __forceinline__ void accum_dw64_sp4(const float* __restrict__ As, const float* __restrict__ Bs,
                                               float* __restrict__ dW, int ldw = 64, int c0 = 0) {
    const int t = threadIdx.x;
    const int ty = t >> 4, tx = t & 15;
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int rbase = 16 * pass + 4 * ty;
        float d[4][4];
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) d[u][v] = 0.f;
#pragma unroll
        for (int r = 0; r < ROWS_4; ++r) {
            const float4 a = *reinterpret_cast<const float4*>(As + r * LD4_S + rbase);
            const float4 b = *reinterpret_cast<const float4*>(Bs + r * LD4_S + 4 * tx);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) d[u][v] += av[u] * bv[v];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) atomicAdd(dW + (long)(rbase + u) * ldw + c0 + 4 * tx + v, d[u][v]);
    }
}

constexpr int SMEM_BWD_SP4_F = (3 * ROWS_4 * LD4_S + SP_H * LD4_S + ROWS_4 * SP_NB) * 4;

// Wave 21: High-efficiency TP=4, NTH=64 (float4 LDS.128, 17 blk at K=64) fused dW4 + fused c_to_rt_bwd (LST_MLP_TINY=2)!
template <bool FUSE_CRT = true>
__global__ void __launch_bounds__(NTH_4, 4) k_mlp_bwd_sp4_fused(const float* __restrict__ EF_k,
                                                                const float* __restrict__ DEF_k,
                                                                const float* __restrict__ W1,
                                                                const float* __restrict__ W2,
                                                                const float* __restrict__ W3,
                                                                const float* __restrict__ W4,
                                                                const float* __restrict__ gout_k, float* dW1,
                                                                float* dW2, float* dW3, float* dW4, float act, int Pm,
                                                                float sp_h = 0.f) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS_4 * LD4_S;
    float* H3s = Bs + ROWS_4 * LD4_S;
    float* Ws = H3s + ROWS_4 * LD4_S;
    float* Xs = Ws + SP_H * LD4_S;
    const int t = threadIdx.x;
    const int rg = t >> 4, cg = t & 15;
    const long p0 = (long)blockIdx.x * TP_4;
    if (t < ROWS_4 * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_4 - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_4 ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(Xs + row * SP_NB + 4 * h4) = v;
    }
    load_w4<SP_NB>(Ws, W1, 64, 0);
    __syncthreads();
    float z1[2][4], z2[2][4], z3[2][4], g[2][4];
    zero2x4(z1);
    gemm2x4_68<SP_NB, SP_NB>(Xs, Ws, rg, cg, z1);
    store_act2x4<TP_4, LD4_S>(As, z1, act, rg, cg);  // h1
    __syncthreads();
    load_w4<64>(Ws, W2, 64, 0);
    __syncthreads();
    zero2x4(z2);
    gemm2x4_68<64, LD4_S>(As, Ws, rg, cg, z2);
    __syncthreads();
    store_act2x4<TP_4, LD4_S>(As, z2, act, rg, cg);  // h2
    load_w4<64>(Ws, W3, 64, 0);
    __syncthreads();
    zero2x4(z3);
    gemm2x4_68<64, LD4_S>(As, Ws, rg, cg, z3);
    store_act2x4<TP_4, LD4_S>(H3s, z3, act, rg, cg);  // h3
    zero2x4(g);
    const int Ksp = Pm - 1;
#pragma unroll
    for (int q = 0; q < SP_O / 64; ++q) {
        __syncthreads();
        if constexpr (FUSE_CRT) {
            // Each thread t=0..63 owns channel c=t; stream intervals p0-1 .. p0+3 reusing (g2,g3) across adjacent knots!
            const int c = t;
            float prev_g2 = 0.f, prev_g3 = 0.f;
            if (p0 > 0) {
                const long b0 = ((p0 - 1) * 4 + q) * 4 * 64 + c;
                prev_g2 = gout_k[b0 + 2 * 64];
                prev_g3 = gout_k[b0 + 3 * 64];
            }
#pragma unroll
            for (int r = 0; r < TP_4; ++r) {
                const long p = p0 + r;
                float gr = 0.f, gt = 0.f;
                if (p < Pm) {
                    if (p > 0) {
                        gr += 3.f * prev_g2 - 2.f * prev_g3;
                        gt += sp_h * (-prev_g2 + prev_g3);
                    }
                    if (p < Ksp) {
                        const long bp = (p * 4 + q) * 4 * 64 + c;
                        const float g0 = gout_k[bp + 0 * 64], g1 = gout_k[bp + 1 * 64];
                        const float g2 = gout_k[bp + 2 * 64], g3 = gout_k[bp + 3 * 64];
                        gr += g0 - 3.f * g2 + 2.f * g3;
                        gt += sp_h * (g1 - 2.f * g2 + g3);
                        prev_g2 = g2;
                        prev_g3 = g3;
                    }
                }
                Bs[r * LD4_S + c] = gr;
                Bs[(TP_4 + r) * LD4_S + c] = gt;
            }
        } else {
#pragma unroll
            for (int i = 0; i < ROWS_4 * 16 / NTH_4; ++i) {
                const int idx = t + i * NTH_4;
                const int row = idx >> 4, k4 = idx & 15;
                const long p = p0 + (row & (TP_4 - 1));
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (p < Pm) {
                    const long grow = row < TP_4 ? p : (Pm + p);
                    v = *reinterpret_cast<const float4*>(gout_k + grow * SP_O + 64 * q + 4 * k4);
                }
                *reinterpret_cast<float4*>(Bs + row * LD4_S + 4 * k4) = v;
            }
        }
        load_w4_trans(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        gemm2x4_68<64, LD4_S>(Bs, Ws, rg, cg, g);
        accum_dw64_sp4(H3s, Bs, dW4, SP_O, 64 * q);
    }
    act_bwd2x4(z3, g, act);
    __syncthreads();
    store_tile2x4<TP_4, LD4_S>(Bs, g, rg, cg);
    load_w4_trans(Ws, W3, 64, 0);
    __syncthreads();
    accum_dw64_sp4(As, Bs, dW3);
    zero2x4(g);
    gemm2x4_68<64, LD4_S>(Bs, Ws, rg, cg, g);
    act_bwd2x4(z2, g, act);
    __syncthreads();
    store_tile2x4<TP_4, LD4_S>(Bs, g, rg, cg);
    store_act2x4<TP_4, LD4_S>(As, z1, act, rg, cg);
    load_w4_trans(Ws, W2, 64, 0);
    __syncthreads();
    accum_dw64_sp4(As, Bs, dW2);
    zero2x4(g);
    gemm2x4_68<64, LD4_S>(Bs, Ws, rg, cg, g);
    act_bwd2x4(z1, g, act);
    __syncthreads();
    store_tile2x4<TP_4, LD4_S>(Bs, g, rg, cg);
    __syncthreads();
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int a = rg + 4 * p;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#pragma unroll
        for (int r = 0; r < ROWS_4; ++r) {
            const float xa = Xs[r * SP_NB + a];
            const float4 b = *reinterpret_cast<const float4*>(Bs + r * LD4_S + 4 * cg);
            d0 += xa * b.x; d1 += xa * b.y; d2 += xa * b.z; d3 += xa * b.w;
        }
        atomicAdd(dW1 + a * 64 + 4 * cg + 0, d0);
        atomicAdd(dW1 + a * 64 + 4 * cg + 1, d1);
        atomicAdd(dW1 + a * 64 + 4 * cg + 2, d2);
        atomicAdd(dW1 + a * 64 + 4 * cg + 3, d3);
    }
}

// Backward TP=4, NTH=64: 33 blocks at K=128, 17 at K=64; 22 KB smem, ~64 regs, ZERO k_mlp_wt, ZERO dW4 atomics!
__global__ void __launch_bounds__(NTH_4, 4) k_mlp_bwd_sp4(const float* __restrict__ EF_k,
                                                          const float* __restrict__ DEF_k,
                                                          const float* __restrict__ W1, const float* __restrict__ W2,
                                                          const float* __restrict__ W3, const float* __restrict__ W4,
                                                          const float* __restrict__ gout_k, float* dW1, float* dW2,
                                                          float* dW3, float act, int Pm) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS_4 * LD4_S;
    float* Ws = Bs + ROWS_4 * LD4_S;
    float* Xs = Ws + SP_H * LD4_S;
    const int t = threadIdx.x;
    const int rg = t >> 4, cg = t & 15;
    const long p0 = (long)blockIdx.x * TP_4;
    if (t < ROWS_4 * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_4 - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_4 ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(Xs + row * SP_NB + 4 * h4) = v;
    }
    load_w4<SP_NB>(Ws, W1, 64, 0);
    __syncthreads();
    float z1[2][4], z2[2][4], z3[2][4], g[2][4];
    zero2x4(z1);
    gemm2x4_68<SP_NB, SP_NB>(Xs, Ws, rg, cg, z1);
    store_act2x4<TP_4, LD4_S>(As, z1, act, rg, cg);  // h1
    __syncthreads();
    load_w4<64>(Ws, W2, 64, 0);
    __syncthreads();
    zero2x4(z2);
    gemm2x4_68<64, LD4_S>(As, Ws, rg, cg, z2);
    __syncthreads();
    store_act2x4<TP_4, LD4_S>(As, z2, act, rg, cg);  // h2
    load_w4<64>(Ws, W3, 64, 0);
    __syncthreads();
    zero2x4(z3);
    gemm2x4_68<64, LD4_S>(As, Ws, rg, cg, z3);
    // gh3 = gout_k @ W4^T (As still holds h2)
    zero2x4(g);
#pragma unroll
    for (int q = 0; q < SP_O / 64; ++q) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < ROWS_4 * 16 / NTH_4; ++i) {
            const int idx = t + i * NTH_4;
            const int row = idx >> 4, k4 = idx & 15;
            const long p = p0 + (row & (TP_4 - 1));
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (p < Pm) {
                const long grow = row < TP_4 ? p : (Pm + p);
                v = *reinterpret_cast<const float4*>(gout_k + grow * SP_O + 64 * q + 4 * k4);
            }
            *reinterpret_cast<float4*>(Bs + row * LD4_S + 4 * k4) = v;
        }
        load_w4_trans(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        gemm2x4_68<64, LD4_S>(Bs, Ws, rg, cg, g);
    }
    act_bwd2x4(z3, g, act);  // gz3
    __syncthreads();
    store_tile2x4<TP_4, LD4_S>(Bs, g, rg, cg);
    load_w4_trans(Ws, W3, 64, 0);
    __syncthreads();
    accum_dw64_sp4(As, Bs, dW3);
    zero2x4(g);
    gemm2x4_68<64, LD4_S>(Bs, Ws, rg, cg, g);  // gh2
    act_bwd2x4(z2, g, act);                    // gz2
    __syncthreads();
    store_tile2x4<TP_4, LD4_S>(Bs, g, rg, cg);
    store_act2x4<TP_4, LD4_S>(As, z1, act, rg, cg);  // h1
    load_w4_trans(Ws, W2, 64, 0);
    __syncthreads();
    accum_dw64_sp4(As, Bs, dW2);
    zero2x4(g);
    gemm2x4_68<64, LD4_S>(Bs, Ws, rg, cg, g);  // gh1
    act_bwd2x4(z1, g, act);                    // gz1
    __syncthreads();
    store_tile2x4<TP_4, LD4_S>(Bs, g, rg, cg);
    __syncthreads();
    // dW1[8, 64] += Xs^T @ Bs (2 passes over a = rg and rg+4, float4 on Bs)
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int a = rg + 4 * p;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#pragma unroll
        for (int r = 0; r < ROWS_4; ++r) {
            const float xa = Xs[r * SP_NB + a];
            const float4 b = *reinterpret_cast<const float4*>(Bs + r * LD4_S + 4 * cg);
            d0 += xa * b.x; d1 += xa * b.y; d2 += xa * b.z; d3 += xa * b.w;
        }
        atomicAdd(dW1 + a * 64 + 4 * cg + 0, d0);
        atomicAdd(dW1 + a * 64 + 4 * cg + 1, d1);
        atomicAdd(dW1 + a * 64 + 4 * cg + 2, d2);
        atomicAdd(dW1 + a * 64 + 4 * cg + 3, d3);
    }
}

// Dedicated non-atomic dW4 = h3_k^T @ gout_k ([64, 256], K = 2*Pm): 32 blocks (dim3(8, 4)) of 128 threads,
// 7.2 KB smem, 100% coalesced float4 LDG.128/LDS.128/STG.128, ZERO cutlass, ZERO splitKreduce, ZERO atomics!
__global__ void __launch_bounds__(128) k_mlp_dw4(const float* __restrict__ h3_k, const float* __restrict__ gout_k,
                                                 float* __restrict__ dW4, int Ktot) {
    const int bx = blockIdx.x;  // 0..7 -> 32 cols of O=256
    const int by = blockIdx.y;  // 0..3 -> 16 rows of H=64
    const int tid = threadIdx.x;
    const int ty = tid >> 3, tx = tid & 7;  // ty in 0..15 (row), tx in 0..7 (4 cols)
    __shared__ __align__(16) float Hs[32][20];
    __shared__ __align__(16) float Gs[32][36];
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const int h0 = 16 * by, o0 = 32 * bx;
    for (int k0 = 0; k0 < Ktot; k0 += 32) {
        {
            const int rk = tid >> 2, c4 = (tid & 3) * 4;
            const int k = k0 + rk;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (k < Ktot) v = *reinterpret_cast<const float4*>(h3_k + (long)k * SP_H + h0 + c4);
            *reinterpret_cast<float4*>(&Hs[rk][c4]) = v;
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 128 * q, rk = idx >> 3, c4 = (idx & 7) * 4;
            const int k = k0 + rk;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (k < Ktot) v = *reinterpret_cast<const float4*>(gout_k + (long)k * SP_O + o0 + c4);
            *reinterpret_cast<float4*>(&Gs[rk][c4]) = v;
        }
        __syncthreads();
#pragma unroll 8
        for (int rk = 0; rk < 32; ++rk) {
            const float a = Hs[rk][ty];
            const float4 b = *reinterpret_cast<const float4*>(&Gs[rk][4 * tx]);
            a0 += a * b.x; a1 += a * b.y; a2 += a * b.z; a3 += a * b.w;
        }
        __syncthreads();
    }
    *reinterpret_cast<float4*>(dW4 + (long)(h0 + ty) * SP_O + o0 + 4 * tx) = make_float4(a0, a1, a2, a3);
}

// Wave 34: Dual-layer k_mlp_dw4 (grid dim3(8, 4, 2), block 128): computes both L0 & L1 dW4 in 1 launch!
__global__ void __launch_bounds__(128) k_mlp_dw4_dual(
    const float* __restrict__ h3_0, const float* __restrict__ gout_0, float* __restrict__ dW4_0,
    const float* __restrict__ h3_1, const float* __restrict__ gout_1, float* __restrict__ dW4_1,
    int Ktot) {
    const int lyr = blockIdx.z;
    const float* __restrict__ h3_k = lyr ? h3_1 : h3_0;
    const float* __restrict__ gout_k = lyr ? gout_1 : gout_0;
    float* __restrict__ dW4 = lyr ? dW4_1 : dW4_0;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int ty = tid >> 3, tx = tid & 7;
    __shared__ __align__(16) float Hs[32][20];
    __shared__ __align__(16) float Gs[32][36];
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const int h0 = 16 * by, o0 = 32 * bx;
    for (int k0 = 0; k0 < Ktot; k0 += 32) {
        {
            const int rk = tid >> 2, c4 = (tid & 3) * 4;
            const int k = k0 + rk;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (k < Ktot) v = *reinterpret_cast<const float4*>(h3_k + (long)k * SP_H + h0 + c4);
            *reinterpret_cast<float4*>(&Hs[rk][c4]) = v;
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 128 * q, rk = idx >> 3, c4 = (idx & 7) * 4;
            const int k = k0 + rk;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (k < Ktot) v = *reinterpret_cast<const float4*>(gout_k + (long)k * SP_O + o0 + c4);
            *reinterpret_cast<float4*>(&Gs[rk][c4]) = v;
        }
        __syncthreads();
#pragma unroll 8
        for (int rk = 0; rk < 32; ++rk) {
            const float a = Hs[rk][ty];
            const float4 b = *reinterpret_cast<const float4*>(&Gs[rk][4 * tx]);
            a0 += a * b.x; a1 += a * b.y; a2 += a * b.z; a3 += a * b.w;
        }
        __syncthreads();
    }
    *reinterpret_cast<float4*>(dW4 + (long)(h0 + ty) * SP_O + o0 + 4 * tx) = make_float4(a0, a1, a2, a3);
}

// ============================================================================ Wave 15: 8-warp TP=8 MLP backward (LST_MLP_SP8=1)
// Fixes sp4's 2-warp/SM scheduler under-utilization (4.18% occ) and 33-way atomic contention while doubling
// SM fill over k_mlp_bwd_sp (17 blocks of 256 threads = 8 warps/block at K=128!), halving regs (2x2/thread),
// and deleting k_mlp_wt (direct smem transpose), while keeping fast cuBLAS/cutlass for dW4!
constexpr int TP_8 = 8, ROWS_8 = 2 * TP_8, NTH_8 = 256, LD8_S = 68;
constexpr int SMEM_BWD_SP8 = (2 * ROWS_8 * LD8_S + SP_H * LD8_S + ROWS_8 * SP_NB) * 4;  // 26.0 KB

template <int K>
__device__ __forceinline__ void load_w8(float* Ws, const float* __restrict__ W, int ldw, int c0) {
    for (int idx = threadIdx.x; idx < K * 16; idx += NTH_8) {
        const int k = idx >> 4, q = idx & 15;
        *reinterpret_cast<float4*>(Ws + k * LD8_S + 4 * q) =
            *reinterpret_cast<const float4*>(W + (long)k * ldw + c0 + 4 * q);
    }
}

__device__ __forceinline__ void load_w8_trans(float* Ws, const float* __restrict__ W, int ldw, int c0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = threadIdx.x + i * NTH_8;
        const int n_out = idx >> 4, k4 = (idx & 15) * 4;
        const float4 v = *reinterpret_cast<const float4*>(W + (long)n_out * ldw + c0 + k4);
        Ws[(k4 + 0) * LD8_S + n_out] = v.x;
        Ws[(k4 + 1) * LD8_S + n_out] = v.y;
        Ws[(k4 + 2) * LD8_S + n_out] = v.z;
        Ws[(k4 + 3) * LD8_S + n_out] = v.w;
    }
}

template <int K, int LDA>
__device__ __forceinline__ void gemm2x2_8(const float* __restrict__ A, const float* __restrict__ B, int rg, int cg,
                                          float acc[2][2]) {
#pragma unroll 4
    for (int k = 0; k < K; k += 4) {
        float2 b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) b[j] = *reinterpret_cast<const float2*>(B + (k + j) * LD8_S + 2 * cg);
        const float4 a0 = *reinterpret_cast<const float4*>(A + rg * LDA + k);
        const float4 a1 = *reinterpret_cast<const float4*>(A + (TP_8 + rg) * LDA + k);
        acc[0][0] += a0.x * b[0].x + a0.y * b[1].x + a0.z * b[2].x + a0.w * b[3].x;
        acc[0][1] += a0.x * b[0].y + a0.y * b[1].y + a0.z * b[2].y + a0.w * b[3].y;
        acc[1][0] += a1.x * b[0].x + a1.y * b[1].x + a1.z * b[2].x + a1.w * b[3].x;
        acc[1][1] += a1.x * b[0].y + a1.y * b[1].y + a1.z * b[2].y + a1.w * b[3].y;
    }
}

__device__ __forceinline__ void store_tile2x2_8(float* S, const float v[2][2], int rg, int cg) {
    *reinterpret_cast<float2*>(S + rg * LD8_S + 2 * cg) = make_float2(v[0][0], v[0][1]);
    *reinterpret_cast<float2*>(S + (TP_8 + rg) * LD8_S + 2 * cg) = make_float2(v[1][0], v[1][1]);
}

__device__ __forceinline__ void store_act2x2_8(float* S, const float z[2][2], float a, int rg, int cg) {
    float h0[2], h1[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float v = z[0][j], s = sp_sig(v);
        h0[j] = a * v * s;
        h1[j] = a * s * (1.f + v * (1.f - s)) * z[1][j];
    }
    *reinterpret_cast<float2*>(S + rg * LD8_S + 2 * cg) = make_float2(h0[0], h0[1]);
    *reinterpret_cast<float2*>(S + (TP_8 + rg) * LD8_S + 2 * cg) = make_float2(h1[0], h1[1]);
}

__device__ __forceinline__ void act_bwd2x2_8(const float z[2][2], float g[2][2], float a) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float v = z[0][j], dv = z[1][j], s = sp_sig(v);
        const float d1 = s * (1.f + v * (1.f - s));
        const float d2 = s * (1.f - s) * (2.f + v * (1.f - 2.f * s));
        const float gv = g[0][j], gt = g[1][j];
        g[0][j] = a * (d1 * gv + d2 * dv * gt);
        g[1][j] = a * d1 * gt;
    }
}

__global__ void __launch_bounds__(NTH_8, 3) k_mlp_bwd_sp8(const float* __restrict__ EF_k,
                                                          const float* __restrict__ DEF_k,
                                                          const float* __restrict__ W1, const float* __restrict__ W2,
                                                          const float* __restrict__ W3, const float* __restrict__ W4,
                                                          const float* __restrict__ gout_k, float* dW1, float* dW2,
                                                          float* dW3, float act, int Pm) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS_8 * LD8_S;
    float* Ws = Bs + ROWS_8 * LD8_S;
    float* Xs = Ws + SP_H * LD8_S;
    const int t = threadIdx.x;
    const int rg = t >> 5, cg = t & 31;  // 8 pairs, 32 col-pairs (2 cols/thread)
    const long p0 = (long)blockIdx.x * TP_8;
    if (t < ROWS_8 * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_8 - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_8 ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(Xs + row * SP_NB + 4 * h4) = v;
    }
    load_w8<SP_NB>(Ws, W1, 64, 0);
    __syncthreads();
    float z1[2][2] = {}, z2[2][2] = {}, z3[2][2] = {}, g[2][2] = {};
    gemm2x2_8<SP_NB, SP_NB>(Xs, Ws, rg, cg, z1);
    store_act2x2_8(As, z1, act, rg, cg);  // h1
    __syncthreads();
    load_w8<64>(Ws, W2, 64, 0);
    __syncthreads();
    gemm2x2_8<64, LD8_S>(As, Ws, rg, cg, z2);
    __syncthreads();
    store_act2x2_8(As, z2, act, rg, cg);  // h2
    load_w8<64>(Ws, W3, 64, 0);
    __syncthreads();
    gemm2x2_8<64, LD8_S>(As, Ws, rg, cg, z3);
#pragma unroll
    for (int q = 0; q < SP_O / 64; ++q) {
        __syncthreads();
        {
            const int row = t >> 4, k4 = t & 15;
            const long p = p0 + (row & (TP_8 - 1));
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (p < Pm) {
                const long grow = row < TP_8 ? p : (Pm + p);
                v = *reinterpret_cast<const float4*>(gout_k + grow * SP_O + 64 * q + 4 * k4);
            }
            *reinterpret_cast<float4*>(Bs + row * LD8_S + 4 * k4) = v;
        }
        load_w8_trans(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        gemm2x2_8<64, LD8_S>(Bs, Ws, rg, cg, g);
    }
    act_bwd2x2_8(z3, g, act);  // gz3
    __syncthreads();
    store_tile2x2_8(Bs, g, rg, cg);
    load_w8_trans(Ws, W3, 64, 0);
    __syncthreads();
    const int dty = t >> 4, dtx = t & 15;
    float d3[4][4] = {}, d2[4][4] = {};
#pragma unroll 4
    for (int r = 0; r < ROWS_8; ++r) {
        const float4 a = *reinterpret_cast<const float4*>(As + r * LD8_S + 4 * dty);
        const float4 b = *reinterpret_cast<const float4*>(Bs + r * LD8_S + 4 * dtx);
        const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) d3[u][v] += av[u] * bv[v];
    }
    g[0][0] = g[0][1] = g[1][0] = g[1][1] = 0.f;
    gemm2x2_8<64, LD8_S>(Bs, Ws, rg, cg, g);  // gh2
    act_bwd2x2_8(z2, g, act);                 // gz2
    __syncthreads();
    store_tile2x2_8(Bs, g, rg, cg);
    store_act2x2_8(As, z1, act, rg, cg);      // h1
    load_w8_trans(Ws, W2, 64, 0);
    __syncthreads();
#pragma unroll 4
    for (int r = 0; r < ROWS_8; ++r) {
        const float4 a = *reinterpret_cast<const float4*>(As + r * LD8_S + 4 * dty);
        const float4 b = *reinterpret_cast<const float4*>(Bs + r * LD8_S + 4 * dtx);
        const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) d2[u][v] += av[u] * bv[v];
    }
    g[0][0] = g[0][1] = g[1][0] = g[1][1] = 0.f;
    gemm2x2_8<64, LD8_S>(Bs, Ws, rg, cg, g);  // gh1
    act_bwd2x2_8(z1, g, act);                 // gz1
    __syncthreads();
    store_tile2x2_8(Bs, g, rg, cg);
    __syncthreads();
    const int a = t >> 5, b0 = 2 * (t & 31);
    float d10 = 0.f, d11 = 0.f;
#pragma unroll 4
    for (int r = 0; r < ROWS_8; ++r) {
        const float xa = Xs[r * SP_NB + a];
        const float2 bv = *reinterpret_cast<const float2*>(Bs + r * LD8_S + b0);
        d10 += xa * bv.x;
        d11 += xa * bv.y;
    }
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            atomicAdd(dW3 + (4 * dty + u) * 64 + 4 * dtx + v, d3[u][v]);
            atomicAdd(dW2 + (4 * dty + u) * 64 + 4 * dtx + v, d2[u][v]);
        }
    atomicAdd(dW1 + a * 64 + b0, d10);
    atomicAdd(dW1 + a * 64 + b0 + 1, d11);
}

// ============================================================================ Wave 17: Tensor-Core TF32 Radial MLP
// (LST_MLP_TC2=1):
//   - k_mlp_fwd_sp_tc: fuses BOTH layers (blockIdx.y = 0, 1) into ONE launch (34 blocks at K=128 -> fills all SMs!)
//     and runs 8 warps x 8 mma_tf32_16x8x8 per layer! ROWS=16 (8 val + 8 deriv) matches m16n8k8 natively so
//     (c[0], c[2]) = (v, dv) and (c[1], c[3]) = (v, dv) stay in registers for in-place SiLU + derivative!
//   - k_mlp_bwd_sp8_tc: drop-in 256-thread TP=8 Tensor-Core TF32 backward replacing all 8 64x64 SIMT GEMMs with
//     mma_tf32_16x8x8 (8 MMAs/warp instead of 256 FMAs!) and zero k_mlp_wt!

__device__ __forceinline__ void act_fwd_tc(float c[4], float a) {
    const float v0 = c[0], s0 = sp_sig(v0);
    c[0] = a * v0 * s0;
    c[2] = a * s0 * (1.f + v0 * (1.f - s0)) * c[2];
    const float v1 = c[1], s1 = sp_sig(v1);
    c[1] = a * v1 * s1;
    c[3] = a * s1 * (1.f + v1 * (1.f - s1)) * c[3];
}

__device__ __forceinline__ void act_bwd_tc(const float z[4], float g[4], float a) {
    {
        const float v = z[0], dv = z[2], s = sp_sig(v);
        const float d1 = s * (1.f + v * (1.f - s));
        const float d2 = s * (1.f - s) * (2.f + v * (1.f - 2.f * s));
        const float gv = g[0], gt = g[2];
        g[0] = a * (d1 * gv + d2 * dv * gt);
        g[2] = a * d1 * gt;
    }
    {
        const float v = z[1], dv = z[3], s = sp_sig(v);
        const float d1 = s * (1.f + v * (1.f - s));
        const float d2 = s * (1.f - s) * (2.f + v * (1.f - 2.f * s));
        const float gv = g[1], gt = g[3];
        g[1] = a * (d1 * gv + d2 * dv * gt);
        g[3] = a * d1 * gt;
    }
}

// 16xK @ Kx64 -> 16x64 (K = 8 or 64), 8 warps, warp w computes 16x8 cols [8*w .. 8*w+7] into c[4].
template <int K, int LDA, int LDW>
__device__ __forceinline__ void gemm_tc_16x64(const float* __restrict__ A, const float* __restrict__ W, int warp,
                                              int g, int tl, float c[4]) {
    const int col = warp * 8 + g;
#pragma unroll
    for (int k0 = 0; k0 < K; k0 += 8) {
        const uint32_t a0 = ah::cvt_tf32(A[g * LDA + k0 + tl]);
        const uint32_t a1 = ah::cvt_tf32(A[(g + 8) * LDA + k0 + tl]);
        const uint32_t a2 = ah::cvt_tf32(A[g * LDA + k0 + tl + 4]);
        const uint32_t a3 = ah::cvt_tf32(A[(g + 8) * LDA + k0 + tl + 4]);
        const uint32_t b0 = ah::cvt_tf32(W[(k0 + tl) * LDW + col]);
        const uint32_t b1 = ah::cvt_tf32(W[(k0 + tl + 4) * LDW + col]);
        ah::mma_tf32_16x8x8(c, a0, a1, a2, a3, b0, b1);
    }
}

__device__ __forceinline__ void store_tc_16x64(float* S, const float c[4], int warp, int g, int tl, int lds) {
    const int col = warp * 8 + 2 * tl;
    *reinterpret_cast<float2*>(S + g * lds + col) = make_float2(c[0], c[1]);
    *reinterpret_cast<float2*>(S + (g + 8) * lds + col) = make_float2(c[2], c[3]);
}

__device__ __forceinline__ void store_act_tc_16x64(float* S, const float z[4], float a, int warp, int g, int tl,
                                                   int lds) {
    float c[4] = {z[0], z[1], z[2], z[3]};
    act_fwd_tc(c, a);
    store_tc_16x64(S, c, warp, g, tl, lds);
}

// Accumulate dW[64, 64] += A^T @ B (A, B are [16, LD8_S]) using 8 warps (4x2 of 16x32) x 8 mma_tf32_16x8x8!
__device__ __forceinline__ void accum_dw64_tc(const float* __restrict__ A, const float* __restrict__ B, float d[4][4],
                                              int warp, int g, int tl) {
    const int wm = warp >> 1, wn = warp & 1;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
#pragma unroll
    for (int k0 = 0; k0 < 16; k0 += 8) {
        const uint32_t a0 = ah::cvt_tf32(A[(k0 + tl) * LD8_S + r0]);
        const uint32_t a1 = ah::cvt_tf32(A[(k0 + tl) * LD8_S + r1]);
        const uint32_t a2 = ah::cvt_tf32(A[(k0 + tl + 4) * LD8_S + r0]);
        const uint32_t a3 = ah::cvt_tf32(A[(k0 + tl + 4) * LD8_S + r1]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = 32 * wn + 8 * j + g;
            const uint32_t b0 = ah::cvt_tf32(B[(k0 + tl) * LD8_S + col]);
            const uint32_t b1 = ah::cvt_tf32(B[(k0 + tl + 4) * LD8_S + col]);
            ah::mma_tf32_16x8x8(d[j], a0, a1, a2, a3, b0, b1);
        }
    }
}

constexpr int SMEM_FWD_TC = (ROWS_8 * LD8_S + SP_H * LD8_S) * 4;  // 21.8 KB

// Dual-layer fused TF32 Tensor-Core spline MLP forward: grid dim3((Pm+7)/8, nlayers), 256 threads.
__global__ void __launch_bounds__(256, 4) k_mlp_fwd_sp_tc(
    const float* __restrict__ EF_k, const float* __restrict__ DEF_k,
    const float* __restrict__ W1_0, const float* __restrict__ W2_0, const float* __restrict__ W3_0,
    const float* __restrict__ W4_0, float* __restrict__ out_0, float* __restrict__ h3_0,
    const float* __restrict__ W1_1, const float* __restrict__ W2_1, const float* __restrict__ W3_1,
    const float* __restrict__ W4_1, float* __restrict__ out_1, float* __restrict__ h3_1,
    float act, int Pm) {
    const int layer = blockIdx.y;
    const float* W1 = layer ? W1_1 : W1_0;
    const float* W2 = layer ? W2_1 : W2_0;
    const float* W3 = layer ? W3_1 : W3_0;
    const float* W4 = layer ? W4_1 : W4_0;
    float* out_k = layer ? out_1 : out_0;
    float* h3_k = layer ? h3_1 : h3_0;

    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Ws = As + ROWS_8 * LD8_S;
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31, g = lane >> 2, tl = lane & 3;
    const long p0 = (long)blockIdx.x * TP_8;

    if (t < ROWS_8 * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_8 - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_8 ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(As + row * LD8_S + 4 * h4) = v;
    }
    load_w8<SP_NB>(Ws, W1, 64, 0);
    __syncthreads();
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    gemm_tc_16x64<SP_NB, LD8_S, LD8_S>(As, Ws, warp, g, tl, c);
    act_fwd_tc(c, act);
    __syncthreads();
    store_tc_16x64(As, c, warp, g, tl, LD8_S);
    load_w8<64>(Ws, W2, 64, 0);
    __syncthreads();
    c[0] = c[1] = c[2] = c[3] = 0.f;
    gemm_tc_16x64<64, LD8_S, LD8_S>(As, Ws, warp, g, tl, c);
    act_fwd_tc(c, act);
    __syncthreads();
    store_tc_16x64(As, c, warp, g, tl, LD8_S);
    load_w8<64>(Ws, W3, 64, 0);
    __syncthreads();
    c[0] = c[1] = c[2] = c[3] = 0.f;
    gemm_tc_16x64<64, LD8_S, LD8_S>(As, Ws, warp, g, tl, c);
    act_fwd_tc(c, act);
    __syncthreads();
    store_tc_16x64(As, c, warp, g, tl, LD8_S);
    const long p = p0 + g;
    const int col = warp * 8 + 2 * tl;
    if (p < Pm) {
        *reinterpret_cast<float2*>(h3_k + p * SP_H + col) = make_float2(c[0], c[1]);
        *reinterpret_cast<float2*>(h3_k + (Pm + p) * SP_H + col) = make_float2(c[2], c[3]);
    }
    for (int q = 0; q < SP_O / 64; ++q) {
        if (q) __syncthreads();
        load_w8<64>(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        c[0] = c[1] = c[2] = c[3] = 0.f;
        gemm_tc_16x64<64, LD8_S, LD8_S>(As, Ws, warp, g, tl, c);
        if (p < Pm) {
            *reinterpret_cast<float2*>(out_k + p * SP_O + 64 * q + col) = make_float2(c[0], c[1]);
            *reinterpret_cast<float2*>(out_k + (Pm + p) * SP_O + 64 * q + col) = make_float2(c[2], c[3]);
        }
    }
}

__global__ void __launch_bounds__(256, 3) k_mlp_bwd_sp8_tc(
    const float* __restrict__ EF_k, const float* __restrict__ DEF_k, const float* __restrict__ W1,
    const float* __restrict__ W2, const float* __restrict__ W3, const float* __restrict__ W4,
    const float* __restrict__ gout_k, float* dW1, float* dW2, float* dW3, float act, int Pm) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS_8 * LD8_S;
    float* Ws = Bs + ROWS_8 * LD8_S;
    float* Xs = Ws + SP_H * LD8_S;
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31, g = lane >> 2, tl = lane & 3;
    const long p0 = (long)blockIdx.x * TP_8;
    if (t < ROWS_8 * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_8 - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_8 ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(Xs + row * SP_NB + 4 * h4) = v;
    }
    load_w8<SP_NB>(Ws, W1, 64, 0);
    __syncthreads();
    float z1[4] = {}, z2[4] = {}, z3[4] = {}, gr[4] = {};
    gemm_tc_16x64<SP_NB, SP_NB, LD8_S>(Xs, Ws, warp, g, tl, z1);
    store_act_tc_16x64(As, z1, act, warp, g, tl, LD8_S);
    __syncthreads();
    load_w8<64>(Ws, W2, 64, 0);
    __syncthreads();
    gemm_tc_16x64<64, LD8_S, LD8_S>(As, Ws, warp, g, tl, z2);
    __syncthreads();
    store_act_tc_16x64(As, z2, act, warp, g, tl, LD8_S);
    load_w8<64>(Ws, W3, 64, 0);
    __syncthreads();
    gemm_tc_16x64<64, LD8_S, LD8_S>(As, Ws, warp, g, tl, z3);
#pragma unroll
    for (int q = 0; q < SP_O / 64; ++q) {
        __syncthreads();
        {
            const int row = t >> 4, k4 = t & 15;
            const long p = p0 + (row & (TP_8 - 1));
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (p < Pm) {
                const long grow = row < TP_8 ? p : (Pm + p);
                v = *reinterpret_cast<const float4*>(gout_k + grow * SP_O + 64 * q + 4 * k4);
            }
            *reinterpret_cast<float4*>(Bs + row * LD8_S + 4 * k4) = v;
        }
        load_w8_trans(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        gemm_tc_16x64<64, LD8_S, LD8_S>(Bs, Ws, warp, g, tl, gr);
    }
    act_bwd_tc(z3, gr, act);
    __syncthreads();
    store_tc_16x64(Bs, gr, warp, g, tl, LD8_S);
    load_w8_trans(Ws, W3, 64, 0);
    __syncthreads();
    float d3[4][4] = {}, d2[4][4] = {};
    accum_dw64_tc(As, Bs, d3, warp, g, tl);
    gr[0] = gr[1] = gr[2] = gr[3] = 0.f;
    gemm_tc_16x64<64, LD8_S, LD8_S>(Bs, Ws, warp, g, tl, gr);
    act_bwd_tc(z2, gr, act);
    __syncthreads();
    store_tc_16x64(Bs, gr, warp, g, tl, LD8_S);
    store_act_tc_16x64(As, z1, act, warp, g, tl, LD8_S);
    load_w8_trans(Ws, W2, 64, 0);
    __syncthreads();
    accum_dw64_tc(As, Bs, d2, warp, g, tl);
    gr[0] = gr[1] = gr[2] = gr[3] = 0.f;
    gemm_tc_16x64<64, LD8_S, LD8_S>(Bs, Ws, warp, g, tl, gr);
    act_bwd_tc(z1, gr, act);
    __syncthreads();
    store_tc_16x64(Bs, gr, warp, g, tl, LD8_S);
    __syncthreads();
    const int a = t >> 5, b0 = 2 * (t & 31);
    float d10 = 0.f, d11 = 0.f;
#pragma unroll 4
    for (int r = 0; r < ROWS_8; ++r) {
        const float xa = Xs[r * SP_NB + a];
        const float2 bv = *reinterpret_cast<const float2*>(Bs + r * LD8_S + b0);
        d10 += xa * bv.x;
        d11 += xa * bv.y;
    }
    const int wm = warp >> 1, wn = warp & 1;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = 32 * wn + 8 * j + 2 * tl;
        atomicAdd(dW3 + r0 * 64 + c0, d3[j][0]);
        atomicAdd(dW3 + r0 * 64 + c0 + 1, d3[j][1]);
        atomicAdd(dW3 + r1 * 64 + c0, d3[j][2]);
        atomicAdd(dW3 + r1 * 64 + c0 + 1, d3[j][3]);
        atomicAdd(dW2 + r0 * 64 + c0, d2[j][0]);
        atomicAdd(dW2 + r0 * 64 + c0 + 1, d2[j][1]);
        atomicAdd(dW2 + r1 * 64 + c0, d2[j][2]);
        atomicAdd(dW2 + r1 * 64 + c0 + 1, d2[j][3]);
    }
    atomicAdd(dW1 + a * 64 + b0, d10);
    atomicAdd(dW1 + a * 64 + b0 + 1, d11);
}

// ============================================================================ Wave 31: Exact-FP32 4-way K-split TP=4 MLP
// (LST_MLP_SPV2=1): Targets K=32 (Pm=33 -> 9 blocks vs 5) and K=64 (Pm=65 -> 17 blocks vs 9) with 256 threads/block:
//   - Warp layout: kh = t & 3 (4-way K-split), rg = (t >> 2) & 3 (4 pairs -> 4-way broadcast on Ws!), cg = t >> 4 (0..15)
//   - LDW_V2 = 72 (72 % 32 = 8): 8 distinct float4s in Ws hit ALL 32 banks with ZERO bank conflicts (1 LDS.128 cycle/warp)!
//   - LDA_V2 = 68 (68 % 32 = 4): 16 distinct floats in As hit banks 0..15 with ZERO bank conflicts (1 LDS.32 cycle/warp)!
//   - 3-shuffle butterfly reduce-scatter -> each thread owns 1 exact column col = 4*cg + kh (4x fewer expf, 6 regs for z1..z3)!
//   - 20.6 KB fwd / 22.5 KB bwd smem -> 4 blocks/SM on Ada & 5060 Ti; 100% exact FP32 arithmetic!
constexpr int TP_V2 = 4, ROWS_V2 = 2 * TP_V2, NTH_V2 = 256, LDA_V2 = 68, LDW_V2 = 72;
constexpr int SMEM_FWD_SPV2 = (ROWS_V2 * LDA_V2 + SP_H * LDW_V2) * 4;
constexpr int SMEM_BWD_SPV2 = (2 * ROWS_V2 * LDA_V2 + SP_H * LDW_V2 + ROWS_V2 * SP_NB) * 4;

template <int K>
__device__ __forceinline__ void load_w_v2(float* Ws, const float* __restrict__ W, int ldw, int c0) {
    for (int idx = threadIdx.x; idx < K * 16; idx += NTH_V2) {
        const int k = idx >> 4, q = idx & 15;
        *reinterpret_cast<float4*>(Ws + k * LDW_V2 + 4 * q) =
            *reinterpret_cast<const float4*>(W + (long)k * ldw + c0 + 4 * q);
    }
}

template <int K>
__device__ __forceinline__ void load_w_v2_sc(float* Ws, const float* __restrict__ W, int ldw, int c0, float sc) {
    for (int idx = threadIdx.x; idx < K * 16; idx += NTH_V2) {
        const int k = idx >> 4, q = idx & 15;
        const float4 v = *reinterpret_cast<const float4*>(W + (long)k * ldw + c0 + 4 * q);
        *reinterpret_cast<float4*>(Ws + k * LDW_V2 + 4 * q) = make_float4(v.x * sc, v.y * sc, v.z * sc, v.w * sc);
    }
}

// Coalesced 64B (4 x float4 per row) transpose load into Ws[k_in * LDW_V2 + n_out] = W[n_out * ldw + c0 + k_in].
// Each warp loads 8 distinct n_out rows -> 4-way max STS conflict (down from 16-way) and 100% L2 sector utilization.
__device__ __forceinline__ void load_w_v2_trans(float* Ws, const float* __restrict__ W, int ldw, int c0) {
    const int n_out = threadIdx.x >> 2;
    const int q4 = threadIdx.x & 3;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int k4 = 16 * s + 4 * q4;
        const float4 v = *reinterpret_cast<const float4*>(W + (long)n_out * ldw + c0 + k4);
        Ws[(k4 + 0) * LDW_V2 + n_out] = v.x;
        Ws[(k4 + 1) * LDW_V2 + n_out] = v.y;
        Ws[(k4 + 2) * LDW_V2 + n_out] = v.z;
        Ws[(k4 + 3) * LDW_V2 + n_out] = v.w;
    }
}

template <int K, int LDA>
__device__ __forceinline__ void gemm_v2(const float* __restrict__ A, const float* __restrict__ Ws,
                                        int rg, int cg, int kh, float& out0, float& out1, bool accum = false) {
    float acc0[4] = {0.f, 0.f, 0.f, 0.f};
    float acc1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k0 = 0; k0 < K; k0 += 4) {
        const int k = k0 + kh;
        const float a0 = A[rg * LDA + k];
        const float a1 = A[(TP_V2 + rg) * LDA + k];
        const float4 b = *reinterpret_cast<const float4*>(Ws + k * LDW_V2 + 4 * cg);
        acc0[0] += a0 * b.x; acc0[1] += a0 * b.y; acc0[2] += a0 * b.z; acc0[3] += a0 * b.w;
        acc1[0] += a1 * b.x; acc1[1] += a1 * b.y; acc1[2] += a1 * b.z; acc1[3] += a1 * b.w;
    }
    auto red4 = [&](const float v[4]) -> float {
        const float a0 = (kh & 2) ? v[0] : v[2];
        const float a1 = (kh & 2) ? v[1] : v[3];
        const float r0 = __shfl_xor_sync(0xffffffffu, a0, 2);
        const float r1 = __shfl_xor_sync(0xffffffffu, a1, 2);
        const float s0 = ((kh & 2) ? v[2] : v[0]) + r0;
        const float s1 = ((kh & 2) ? v[3] : v[1]) + r1;
        const float r2 = __shfl_xor_sync(0xffffffffu, (kh & 1) ? s0 : s1, 1);
        return ((kh & 1) ? s1 : s0) + r2;
    };
    if (accum) {
        out0 += red4(acc0);
        out1 += red4(acc1);
    } else {
        out0 = red4(acc0);
        out1 = red4(acc1);
    }
}

__device__ __forceinline__ void act_fwd_v2(float& z0, float& z1, float a) {
    const float v = z0, s = sp_sig(v);
    z0 = a * v * s;
    z1 = a * s * (1.f + v * (1.f - s)) * z1;
}

__device__ __forceinline__ void act_bwd_v2(float v, float dv, float& gv, float& gt, float a) {
    const float s = sp_sig(v);
    const float d1 = s * (1.f + v * (1.f - s));
    const float d2 = s * (1.f - s) * (2.f + v * (1.f - 2.f * s));
    const float ogv = gv, ogt = gt;
    gv = a * (d1 * ogv + d2 * dv * ogt);
    gt = a * d1 * ogt;
}

__global__ void __launch_bounds__(NTH_V2, 4) k_mlp_fwd_sp_v2(
    const float* __restrict__ EF_k, const float* __restrict__ DEF_k,
    const float* __restrict__ W1, const float* __restrict__ W2,
    const float* __restrict__ W3, const float* __restrict__ W4,
    float* __restrict__ out_k, float* __restrict__ h3_k,
    float act, int Pm,
    float* __restrict__ z1_k = nullptr, float* __restrict__ z2_k = nullptr,
    float* __restrict__ z3_k = nullptr,
    float sw1 = 1.f, float sw2 = 1.f, float sw3 = 1.f, float sw4 = 1.f) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Ws = As + ROWS_V2 * LDA_V2;
    const int t = threadIdx.x;
    const int kh = t & 3, rg = (t >> 2) & 3, cg = t >> 4, col = 4 * cg + kh;
    const long p0 = (long)blockIdx.x * TP_V2;
    const long p = p0 + rg;

    if (t < ROWS_V2 * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long pp = p0 + (row & (TP_V2 - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pp < Pm) v = *reinterpret_cast<const float4*>((row < TP_V2 ? EF_k : DEF_k) + pp * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(As + row * LDA_V2 + 4 * h4) = v;
    }
    load_w_v2_sc<SP_NB>(Ws, W1, 64, 0, sw1);
    __syncthreads();

    float z0, z1;
    gemm_v2<SP_NB, LDA_V2>(As, Ws, rg, cg, kh, z0, z1);
    if (z1_k && p < Pm) {
        z1_k[p * SP_H + col] = z0;
        z1_k[(Pm + p) * SP_H + col] = z1;
    }
    act_fwd_v2(z0, z1, act);
    __syncthreads();
    As[rg * LDA_V2 + col] = z0;
    As[(TP_V2 + rg) * LDA_V2 + col] = z1;
    load_w_v2_sc<64>(Ws, W2, 64, 0, sw2);
    __syncthreads();

    gemm_v2<64, LDA_V2>(As, Ws, rg, cg, kh, z0, z1);
    if (z2_k && p < Pm) {
        z2_k[p * SP_H + col] = z0;
        z2_k[(Pm + p) * SP_H + col] = z1;
    }
    act_fwd_v2(z0, z1, act);
    __syncthreads();
    As[rg * LDA_V2 + col] = z0;
    As[(TP_V2 + rg) * LDA_V2 + col] = z1;
    load_w_v2_sc<64>(Ws, W3, 64, 0, sw3);
    __syncthreads();

    gemm_v2<64, LDA_V2>(As, Ws, rg, cg, kh, z0, z1);
    if (z3_k && p < Pm) {
        z3_k[p * SP_H + col] = z0;
        z3_k[(Pm + p) * SP_H + col] = z1;
    }
    act_fwd_v2(z0, z1, act);
    __syncthreads();
    As[rg * LDA_V2 + col] = z0;
    As[(TP_V2 + rg) * LDA_V2 + col] = z1;
    load_w_v2_sc<64>(Ws, W4, SP_O, 0, sw4);
    __syncthreads();

    if (h3_k && t < ROWS_V2 * 16) {
        const int row = t >> 4, q4 = t & 15;
        const long pp = p0 + (row & (TP_V2 - 1));
        if (pp < Pm) {
            const long dst_row = (row < TP_V2) ? pp : (Pm + pp);
            *reinterpret_cast<float4*>(h3_k + dst_row * SP_H + 4 * q4) =
                *reinterpret_cast<const float4*>(As + row * LDA_V2 + 4 * q4);
        }
    }

#pragma unroll
    for (int q = 0; q < SP_O / 64; ++q) {
        if (q) {
            __syncthreads();
            load_w_v2_sc<64>(Ws, W4, SP_O, 64 * q, sw4);
            __syncthreads();
        }
        gemm_v2<64, LDA_V2>(As, Ws, rg, cg, kh, z0, z1);
        if (p < Pm) {
            out_k[p * SP_O + 64 * q + col] = z0;
            out_k[(Pm + p) * SP_O + 64 * q + col] = z1;
        }
    }
}

__global__ void __launch_bounds__(NTH_V2, 4) k_mlp_bwd_sp4_v2(
    const float* __restrict__ EF_k, const float* __restrict__ DEF_k,
    const float* __restrict__ W1, const float* __restrict__ W2,
    const float* __restrict__ W3, const float* __restrict__ W4,
    const float* __restrict__ gout_k, float* dW1, float* dW2,
    float* dW3, float act, int Pm) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS_V2 * LDA_V2;
    float* Ws = Bs + ROWS_V2 * LDA_V2;
    float* Xs = Ws + SP_H * LDW_V2;
    const int t = threadIdx.x;
    const int kh = t & 3, rg = (t >> 2) & 3, cg = t >> 4, col = 4 * cg + kh;
    const long p0 = (long)blockIdx.x * TP_V2;

    if (t < ROWS_V2 * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_V2 - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_V2 ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(Xs + row * SP_NB + 4 * h4) = v;
    }
    load_w_v2<SP_NB>(Ws, W1, 64, 0);
    __syncthreads();

    float z1_0, z1_1, z2_0, z2_1, z3_0, z3_1;
    gemm_v2<SP_NB, SP_NB>(Xs, Ws, rg, cg, kh, z1_0, z1_1);
    {
        float h0 = z1_0, h1 = z1_1;
        act_fwd_v2(h0, h1, act);
        As[rg * LDA_V2 + col] = h0;
        As[(TP_V2 + rg) * LDA_V2 + col] = h1;
    }
    __syncthreads();
    load_w_v2<64>(Ws, W2, 64, 0);
    __syncthreads();

    gemm_v2<64, LDA_V2>(As, Ws, rg, cg, kh, z2_0, z2_1);
    __syncthreads();
    {
        float h0 = z2_0, h1 = z2_1;
        act_fwd_v2(h0, h1, act);
        As[rg * LDA_V2 + col] = h0;
        As[(TP_V2 + rg) * LDA_V2 + col] = h1;
    }
    load_w_v2<64>(Ws, W3, 64, 0);
    __syncthreads();

    gemm_v2<64, LDA_V2>(As, Ws, rg, cg, kh, z3_0, z3_1);

    // gh3 = gout_k @ W4^T (As still holds h2)
    float g0 = 0.f, g1 = 0.f;
#pragma unroll
    for (int q = 0; q < SP_O / 64; ++q) {
        __syncthreads();
        if (t < ROWS_V2 * 16) {
            const int row = t >> 4, k4 = t & 15;
            const long p = p0 + (row & (TP_V2 - 1));
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (p < Pm) {
                const long grow = (row < TP_V2) ? p : (Pm + p);
                v = *reinterpret_cast<const float4*>(gout_k + grow * SP_O + 64 * q + 4 * k4);
            }
            *reinterpret_cast<float4*>(Bs + row * LDA_V2 + 4 * k4) = v;
        }
        load_w_v2_trans(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        gemm_v2<64, LDA_V2>(Bs, Ws, rg, cg, kh, g0, g1, true);
    }

    act_bwd_v2(z3_0, z3_1, g0, g1, act);  // gz3
    __syncthreads();
    Bs[rg * LDA_V2 + col] = g0;
    Bs[(TP_V2 + rg) * LDA_V2 + col] = g1;
    load_w_v2_trans(Ws, W3, 64, 0);
    __syncthreads();

    const int dty = t >> 4, dtx = t & 15;
    {
        float d3[4][4] = {};
#pragma unroll
        for (int r = 0; r < ROWS_V2; ++r) {
            const float4 a = *reinterpret_cast<const float4*>(As + r * LDA_V2 + 4 * dty);
            const float4 b = *reinterpret_cast<const float4*>(Bs + r * LDA_V2 + 4 * dtx);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) d3[u][v] += av[u] * bv[v];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) atomicAdd(dW3 + (4 * dty + u) * 64 + 4 * dtx + v, d3[u][v]);
    }

    gemm_v2<64, LDA_V2>(Bs, Ws, rg, cg, kh, g0, g1);  // gh2
    act_bwd_v2(z2_0, z2_1, g0, g1, act);              // gz2
    __syncthreads();
    Bs[rg * LDA_V2 + col] = g0;
    Bs[(TP_V2 + rg) * LDA_V2 + col] = g1;
    {
        float h0 = z1_0, h1 = z1_1;
        act_fwd_v2(h0, h1, act);
        As[rg * LDA_V2 + col] = h0;
        As[(TP_V2 + rg) * LDA_V2 + col] = h1;
    }
    load_w_v2_trans(Ws, W2, 64, 0);
    __syncthreads();

    {
        float d2[4][4] = {};
#pragma unroll
        for (int r = 0; r < ROWS_V2; ++r) {
            const float4 a = *reinterpret_cast<const float4*>(As + r * LDA_V2 + 4 * dty);
            const float4 b = *reinterpret_cast<const float4*>(Bs + r * LDA_V2 + 4 * dtx);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) d2[u][v] += av[u] * bv[v];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) atomicAdd(dW2 + (4 * dty + u) * 64 + 4 * dtx + v, d2[u][v]);
    }

    gemm_v2<64, LDA_V2>(Bs, Ws, rg, cg, kh, g0, g1);  // gh1
    act_bwd_v2(z1_0, z1_1, g0, g1, act);              // gz1
    __syncthreads();
    Bs[rg * LDA_V2 + col] = g0;
    Bs[(TP_V2 + rg) * LDA_V2 + col] = g1;
    __syncthreads();

    const int a = t >> 5, b0 = 2 * (t & 31);
    float d10 = 0.f, d11 = 0.f;
#pragma unroll
    for (int r = 0; r < ROWS_V2; ++r) {
        const float xa = Xs[r * SP_NB + a];
        const float2 bv = *reinterpret_cast<const float2*>(Bs + r * LDA_V2 + b0);
        d10 += xa * bv.x;
        d11 += xa * bv.y;
    }
    atomicAdd(dW1 + a * 64 + b0, d10);
    atomicAdd(dW1 + a * 64 + b0 + 1, d11);
}

// Wave 34: Dual-layer exact-FP32 TP=4 MLP backward (grid dim3(ntiles, 2), block 256):
// At K=32 (18 CTAs) & K=64 (34 CTAs), BOTH layers fit in ONE single wave on 36-SM 5060 Ti & 48-SM Ada,
// eliminating mid-step contention during k_q_bwd_c32 / k_prod_rev1_h2 / m0_bwd at ZERO tail cost!
__global__ void __launch_bounds__(NTH_V2, 4) k_mlp_bwd_sp4_v2_dual(
    const float* __restrict__ EF_k, const float* __restrict__ DEF_k,
    const float* __restrict__ W1_0, const float* __restrict__ W2_0,
    const float* __restrict__ W3_0, const float* __restrict__ W4_0,
    const float* __restrict__ gout_0, float* dW1_0, float* dW2_0, float* dW3_0,
    const float* __restrict__ W1_1, const float* __restrict__ W2_1,
    const float* __restrict__ W3_1, const float* __restrict__ W4_1,
    const float* __restrict__ gout_1, float* dW1_1, float* dW2_1, float* dW3_1,
    float act, int Pm) {
    const int lyr = blockIdx.y;
    const float* __restrict__ W1 = lyr ? W1_1 : W1_0;
    const float* __restrict__ W2 = lyr ? W2_1 : W2_0;
    const float* __restrict__ W3 = lyr ? W3_1 : W3_0;
    const float* __restrict__ W4 = lyr ? W4_1 : W4_0;
    const float* __restrict__ gout_k = lyr ? gout_1 : gout_0;
    float* dW1 = lyr ? dW1_1 : dW1_0;
    float* dW2 = lyr ? dW2_1 : dW2_0;
    float* dW3 = lyr ? dW3_1 : dW3_0;

    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS_V2 * LDA_V2;
    float* Ws = Bs + ROWS_V2 * LDA_V2;
    float* Xs = Ws + SP_H * LDW_V2;
    const int t = threadIdx.x;
    const int kh = t & 3, rg = (t >> 2) & 3, cg = t >> 4, col = 4 * cg + kh;
    const long p0 = (long)blockIdx.x * TP_V2;

    if (t < ROWS_V2 * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_V2 - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_V2 ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(Xs + row * SP_NB + 4 * h4) = v;
    }
    load_w_v2<SP_NB>(Ws, W1, 64, 0);
    __syncthreads();

    float z1_0, z1_1, z2_0, z2_1, z3_0, z3_1;
    gemm_v2<SP_NB, SP_NB>(Xs, Ws, rg, cg, kh, z1_0, z1_1);
    {
        float h0 = z1_0, h1 = z1_1;
        act_fwd_v2(h0, h1, act);
        As[rg * LDA_V2 + col] = h0;
        As[(TP_V2 + rg) * LDA_V2 + col] = h1;
    }
    __syncthreads();
    load_w_v2<64>(Ws, W2, 64, 0);
    __syncthreads();

    gemm_v2<64, LDA_V2>(As, Ws, rg, cg, kh, z2_0, z2_1);
    __syncthreads();
    {
        float h0 = z2_0, h1 = z2_1;
        act_fwd_v2(h0, h1, act);
        As[rg * LDA_V2 + col] = h0;
        As[(TP_V2 + rg) * LDA_V2 + col] = h1;
    }
    load_w_v2<64>(Ws, W3, 64, 0);
    __syncthreads();

    gemm_v2<64, LDA_V2>(As, Ws, rg, cg, kh, z3_0, z3_1);

    float g0 = 0.f, g1 = 0.f;
#pragma unroll
    for (int q = 0; q < SP_O / 64; ++q) {
        __syncthreads();
        if (t < ROWS_V2 * 16) {
            const int row = t >> 4, k4 = t & 15;
            const long p = p0 + (row & (TP_V2 - 1));
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (p < Pm) {
                const long grow = (row < TP_V2) ? p : (Pm + p);
                v = *reinterpret_cast<const float4*>(gout_k + grow * SP_O + 64 * q + 4 * k4);
            }
            *reinterpret_cast<float4*>(Bs + row * LDA_V2 + 4 * k4) = v;
        }
        load_w_v2_trans(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        gemm_v2<64, LDA_V2>(Bs, Ws, rg, cg, kh, g0, g1, true);
    }

    act_bwd_v2(z3_0, z3_1, g0, g1, act);
    __syncthreads();
    Bs[rg * LDA_V2 + col] = g0;
    Bs[(TP_V2 + rg) * LDA_V2 + col] = g1;
    load_w_v2_trans(Ws, W3, 64, 0);
    __syncthreads();

    const int dty = t >> 4, dtx = t & 15;
    {
        float d3[4][4] = {};
#pragma unroll
        for (int r = 0; r < ROWS_V2; ++r) {
            const float4 a = *reinterpret_cast<const float4*>(As + r * LDA_V2 + 4 * dty);
            const float4 b = *reinterpret_cast<const float4*>(Bs + r * LDA_V2 + 4 * dtx);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) d3[u][v] += av[u] * bv[v];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) atomicAdd(dW3 + (4 * dty + u) * 64 + 4 * dtx + v, d3[u][v]);
    }

    gemm_v2<64, LDA_V2>(Bs, Ws, rg, cg, kh, g0, g1);
    act_bwd_v2(z2_0, z2_1, g0, g1, act);
    __syncthreads();
    Bs[rg * LDA_V2 + col] = g0;
    Bs[(TP_V2 + rg) * LDA_V2 + col] = g1;
    {
        float h0 = z1_0, h1 = z1_1;
        act_fwd_v2(h0, h1, act);
        As[rg * LDA_V2 + col] = h0;
        As[(TP_V2 + rg) * LDA_V2 + col] = h1;
    }
    load_w_v2_trans(Ws, W2, 64, 0);
    __syncthreads();

    {
        float d2[4][4] = {};
#pragma unroll
        for (int r = 0; r < ROWS_V2; ++r) {
            const float4 a = *reinterpret_cast<const float4*>(As + r * LDA_V2 + 4 * dty);
            const float4 b = *reinterpret_cast<const float4*>(Bs + r * LDA_V2 + 4 * dtx);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) d2[u][v] += av[u] * bv[v];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) atomicAdd(dW2 + (4 * dty + u) * 64 + 4 * dtx + v, d2[u][v]);
    }

    gemm_v2<64, LDA_V2>(Bs, Ws, rg, cg, kh, g0, g1);
    act_bwd_v2(z1_0, z1_1, g0, g1, act);
    __syncthreads();
    Bs[rg * LDA_V2 + col] = g0;
    Bs[(TP_V2 + rg) * LDA_V2 + col] = g1;
    __syncthreads();

    const int a = t >> 5, b0 = 2 * (t & 31);
    float d10 = 0.f, d11 = 0.f;
#pragma unroll
    for (int r = 0; r < ROWS_V2; ++r) {
        const float xa = Xs[r * SP_NB + a];
        const float2 bv = *reinterpret_cast<const float2*>(Bs + r * LDA_V2 + b0);
        d10 += xa * bv.x;
        d11 += xa * bv.y;
    }
    atomicAdd(dW1 + a * 64 + b0, d10);
    atomicAdd(dW1 + a * 64 + b0 + 1, d11);
}

// Wave 37 (LST_MLP_SAVEZ=1): Zero-recompute dual-layer exact-FP32 TP=4 MLP backward.
// Reads saved L2-resident z1_k, z2_k, z3_k (101 KB total across both layers) saved during hidden k_mlp_fwd_sp_v2,
// skipping W1/W2/W3 forward loads, all 3 forward GEMMs, and 7 barriers on the critical tail!
__global__ void __launch_bounds__(NTH_V2, 4) k_mlp_bwd_sp4_v2_dual_noz(
    const float* __restrict__ EF_k, const float* __restrict__ DEF_k,
    const float* __restrict__ W2_0, const float* __restrict__ W3_0, const float* __restrict__ W4_0,
    const float* __restrict__ z1_0, const float* __restrict__ z2_0, const float* __restrict__ z3_0,
    const float* __restrict__ gout_0, float* dW1_0, float* dW2_0, float* dW3_0,
    const float* __restrict__ W2_1, const float* __restrict__ W3_1, const float* __restrict__ W4_1,
    const float* __restrict__ z1_1, const float* __restrict__ z2_1, const float* __restrict__ z3_1,
    const float* __restrict__ gout_1, float* dW1_1, float* dW2_1, float* dW3_1,
    float act, int Pm) {
    const int lyr = blockIdx.y;
    const float* __restrict__ W2 = lyr ? W2_1 : W2_0;
    const float* __restrict__ W3 = lyr ? W3_1 : W3_0;
    const float* __restrict__ W4 = lyr ? W4_1 : W4_0;
    const float* __restrict__ z1_k = lyr ? z1_1 : z1_0;
    const float* __restrict__ z2_k = lyr ? z2_1 : z2_0;
    const float* __restrict__ z3_k = lyr ? z3_1 : z3_0;
    const float* __restrict__ gout_k = lyr ? gout_1 : gout_0;
    float* dW1 = lyr ? dW1_1 : dW1_0;
    float* dW2 = lyr ? dW2_1 : dW2_0;
    float* dW3 = lyr ? dW3_1 : dW3_0;

    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS_V2 * LDA_V2;
    float* Ws = Bs + ROWS_V2 * LDA_V2;
    float* Xs = Ws + SP_H * LDW_V2;
    const int t = threadIdx.x;
    const int kh = t & 3, rg = (t >> 2) & 3, cg = t >> 4, col = 4 * cg + kh;
    const long p0 = (long)blockIdx.x * TP_V2;
    const long p_rg = p0 + rg;

    if (t < ROWS_V2 * (SP_NB / 4)) {
        const int row = t >> 1, h4 = t & 1;
        const long p = p0 + (row & (TP_V2 - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (p < Pm) v = *reinterpret_cast<const float4*>((row < TP_V2 ? EF_k : DEF_k) + p * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(Xs + row * SP_NB + 4 * h4) = v;
    }

    float z1_0v = 0.f, z1_1v = 0.f, z2_0v = 0.f, z2_1v = 0.f, z3_0v = 0.f, z3_1v = 0.f;
    if (p_rg < Pm) {
        const long r0 = p_rg * SP_H + col;
        const long r1 = (Pm + p_rg) * SP_H + col;
        z1_0v = z1_k[r0]; z1_1v = z1_k[r1];
        z2_0v = z2_k[r0]; z2_1v = z2_k[r1];
        z3_0v = z3_k[r0]; z3_1v = z3_k[r1];
    }
    {
        float h0 = z2_0v, h1 = z2_1v;
        if (p_rg < Pm) act_fwd_v2(h0, h1, act);
        As[rg * LDA_V2 + col] = h0;
        As[(TP_V2 + rg) * LDA_V2 + col] = h1;
    }

    float g0 = 0.f, g1 = 0.f;
#pragma unroll
    for (int q = 0; q < SP_O / 64; ++q) {
        if (q) __syncthreads();
        if (t < ROWS_V2 * 16) {
            const int row = t >> 4, k4 = t & 15;
            const long p = p0 + (row & (TP_V2 - 1));
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (p < Pm) {
                const long grow = (row < TP_V2) ? p : (Pm + p);
                v = *reinterpret_cast<const float4*>(gout_k + grow * SP_O + 64 * q + 4 * k4);
            }
            *reinterpret_cast<float4*>(Bs + row * LDA_V2 + 4 * k4) = v;
        }
        load_w_v2_trans(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        gemm_v2<64, LDA_V2>(Bs, Ws, rg, cg, kh, g0, g1, true);
    }

    act_bwd_v2(z3_0v, z3_1v, g0, g1, act);
    __syncthreads();
    Bs[rg * LDA_V2 + col] = g0;
    Bs[(TP_V2 + rg) * LDA_V2 + col] = g1;
    load_w_v2_trans(Ws, W3, 64, 0);
    __syncthreads();

    const int dty = t >> 4, dtx = t & 15;
    {
        float d3[4][4] = {};
#pragma unroll
        for (int r = 0; r < ROWS_V2; ++r) {
            const float4 a = *reinterpret_cast<const float4*>(As + r * LDA_V2 + 4 * dty);
            const float4 b = *reinterpret_cast<const float4*>(Bs + r * LDA_V2 + 4 * dtx);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) d3[u][v] += av[u] * bv[v];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) atomicAdd(dW3 + (4 * dty + u) * 64 + 4 * dtx + v, d3[u][v]);
    }

    gemm_v2<64, LDA_V2>(Bs, Ws, rg, cg, kh, g0, g1);
    act_bwd_v2(z2_0v, z2_1v, g0, g1, act);
    __syncthreads();
    Bs[rg * LDA_V2 + col] = g0;
    Bs[(TP_V2 + rg) * LDA_V2 + col] = g1;
    {
        float h0 = z1_0v, h1 = z1_1v;
        if (p_rg < Pm) act_fwd_v2(h0, h1, act);
        As[rg * LDA_V2 + col] = h0;
        As[(TP_V2 + rg) * LDA_V2 + col] = h1;
    }
    load_w_v2_trans(Ws, W2, 64, 0);
    __syncthreads();

    {
        float d2[4][4] = {};
#pragma unroll
        for (int r = 0; r < ROWS_V2; ++r) {
            const float4 a = *reinterpret_cast<const float4*>(As + r * LDA_V2 + 4 * dty);
            const float4 b = *reinterpret_cast<const float4*>(Bs + r * LDA_V2 + 4 * dtx);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) d2[u][v] += av[u] * bv[v];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) atomicAdd(dW2 + (4 * dty + u) * 64 + 4 * dtx + v, d2[u][v]);
    }

    gemm_v2<64, LDA_V2>(Bs, Ws, rg, cg, kh, g0, g1);
    act_bwd_v2(z1_0v, z1_1v, g0, g1, act);
    __syncthreads();
    Bs[rg * LDA_V2 + col] = g0;
    Bs[(TP_V2 + rg) * LDA_V2 + col] = g1;
    __syncthreads();

    const int a = t >> 5, b0 = 2 * (t & 31);
    float d10 = 0.f, d11 = 0.f;
#pragma unroll
    for (int r = 0; r < ROWS_V2; ++r) {
        const float xa = Xs[r * SP_NB + a];
        const float2 bv = *reinterpret_cast<const float2*>(Bs + r * LDA_V2 + b0);
        d10 += xa * bv.x;
        d11 += xa * bv.y;
    }
    atomicAdd(dW1 + a * 64 + b0, d10);
    atomicAdd(dW1 + a * 64 + b0 + 1, d11);
}

// ============================================================================ Wave 42: TP=2 (ROWS=4, 1 row/thread, 256 threads)
// Exact-FP32 4-way K-split radial MLP fwd & zero-recompute dual bwd (LST_MLP_T2):
//   - Doubles CTA count at K=32 (Pm=33): 17 CTAs (fwd) and 34 CTAs (dual bwd, 94% of 36 SMs on 5060 Ti, 71% of 48 SMs on Ada!)
//   - 1 row/thread (row = (t>>2)&3: 0..1=val, 2..3=der) -> 2x fewer FMAs/thread, 2x fewer LDS on A/B, 2x fewer outer-product rows!
//   - Value/derivative partner sits at lane ^ 8 inside the same warp -> 1-cycle __shfl_xor_sync for SiLU tangent fwd/bwd!
constexpr int TP_T2 = 2, ROWS_T2 = 2 * TP_T2;
constexpr int SMEM_FWD_SPT2 = (ROWS_T2 * LDA_V2 + SP_H * LDW_V2) * 4;
constexpr int SMEM_BWD_SPT2 = (2 * ROWS_T2 * LDA_V2 + SP_H * LDW_V2 + ROWS_T2 * SP_NB) * 4;

template <int K, int LDA>
__device__ __forceinline__ void gemm_v2_1r(const float* __restrict__ A, const float* __restrict__ Ws,
                                           int row, int cg, int kh, float& out, bool accum = false) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k0 = 0; k0 < K; k0 += 4) {
        const int k = k0 + kh;
        const float a = A[row * LDA + k];
        const float4 b = *reinterpret_cast<const float4*>(Ws + k * LDW_V2 + 4 * cg);
        acc[0] += a * b.x; acc[1] += a * b.y; acc[2] += a * b.z; acc[3] += a * b.w;
    }
    const float a0 = (kh & 2) ? acc[0] : acc[2];
    const float a1 = (kh & 2) ? acc[1] : acc[3];
    const float r0 = __shfl_xor_sync(0xffffffffu, a0, 2);
    const float r1 = __shfl_xor_sync(0xffffffffu, a1, 2);
    const float s0 = ((kh & 2) ? acc[2] : acc[0]) + r0;
    const float s1 = ((kh & 2) ? acc[3] : acc[1]) + r1;
    const float r2 = __shfl_xor_sync(0xffffffffu, (kh & 1) ? s0 : s1, 1);
    const float res = ((kh & 1) ? s1 : s0) + r2;
    if (accum) out += res;
    else out = res;
}

__device__ __forceinline__ float act_fwd_v2_1r(float z, int is_der, float a) {
    const float zp = __shfl_xor_sync(0xffffffffu, z, 8);
    const float v = is_der ? zp : z;
    const float s = sp_sig(v);
    return is_der ? (a * s * (1.f + v * (1.f - s)) * z) : (a * v * s);
}

__device__ __forceinline__ float act_bwd_v2_1r(float z, float g, int is_der, float a) {
    const float zp = __shfl_xor_sync(0xffffffffu, z, 8);
    const float gp = __shfl_xor_sync(0xffffffffu, g, 8);
    const float v = is_der ? zp : z;
    const float s = sp_sig(v);
    const float d1 = s * (1.f + v * (1.f - s));
    if (is_der) return a * d1 * g;
    const float d2 = s * (1.f - s) * (2.f + v * (1.f - 2.f * s));
    return a * (d1 * g + d2 * zp * gp);
}

__global__ void __launch_bounds__(NTH_V2, 4) k_mlp_fwd_sp_v2_t2(
    const float* __restrict__ EF_k, const float* __restrict__ DEF_k,
    const float* __restrict__ W1, const float* __restrict__ W2,
    const float* __restrict__ W3, const float* __restrict__ W4,
    float* __restrict__ out_k, float* __restrict__ h3_k,
    float act, int Pm,
    float* __restrict__ z1_k = nullptr, float* __restrict__ z2_k = nullptr,
    float* __restrict__ z3_k = nullptr,
    float sw1 = 1.f, float sw2 = 1.f, float sw3 = 1.f, float sw4 = 1.f) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Ws = As + ROWS_T2 * LDA_V2;
    const int t = threadIdx.x;
    const int kh = t & 3, row = (t >> 2) & 3, cg = t >> 4, col = 4 * cg + kh;
    const int is_der = row >> 1;
    const long p0 = (long)blockIdx.x * TP_T2;
    const long p = p0 + (row & 1);
    const long dst_row = is_der ? (Pm + p) : p;

    if (t < ROWS_T2 * (SP_NB / 4)) {
        const int r = t >> 1, h4 = t & 1;
        const long pp = p0 + (r & (TP_T2 - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pp < Pm) v = *reinterpret_cast<const float4*>((r < TP_T2 ? EF_k : DEF_k) + pp * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(As + r * LDA_V2 + 4 * h4) = v;
    }
    load_w_v2_sc<SP_NB>(Ws, W1, 64, 0, sw1);
    __syncthreads();

    float z;
    gemm_v2_1r<SP_NB, LDA_V2>(As, Ws, row, cg, kh, z);
    if (z1_k && p < Pm) z1_k[dst_row * SP_H + col] = z;
    z = act_fwd_v2_1r(z, is_der, act);
    __syncthreads();
    As[row * LDA_V2 + col] = z;
    load_w_v2_sc<64>(Ws, W2, 64, 0, sw2);
    __syncthreads();

    gemm_v2_1r<64, LDA_V2>(As, Ws, row, cg, kh, z);
    if (z2_k && p < Pm) z2_k[dst_row * SP_H + col] = z;
    z = act_fwd_v2_1r(z, is_der, act);
    __syncthreads();
    As[row * LDA_V2 + col] = z;
    load_w_v2_sc<64>(Ws, W3, 64, 0, sw3);
    __syncthreads();

    gemm_v2_1r<64, LDA_V2>(As, Ws, row, cg, kh, z);
    if (z3_k && p < Pm) z3_k[dst_row * SP_H + col] = z;
    z = act_fwd_v2_1r(z, is_der, act);
    __syncthreads();
    As[row * LDA_V2 + col] = z;
    load_w_v2_sc<64>(Ws, W4, SP_O, 0, sw4);
    __syncthreads();

    if (h3_k && t < ROWS_T2 * 16) {
        const int r = t >> 4, q4 = t & 15;
        const long pp = p0 + (r & (TP_T2 - 1));
        if (pp < Pm) {
            const long hr = (r < TP_T2) ? pp : (Pm + pp);
            *reinterpret_cast<float4*>(h3_k + hr * SP_H + 4 * q4) =
                *reinterpret_cast<const float4*>(As + r * LDA_V2 + 4 * q4);
        }
    }

#pragma unroll
    for (int q = 0; q < SP_O / 64; ++q) {
        if (q) {
            __syncthreads();
            load_w_v2_sc<64>(Ws, W4, SP_O, 64 * q, sw4);
            __syncthreads();
        }
        gemm_v2_1r<64, LDA_V2>(As, Ws, row, cg, kh, z);
        if (p < Pm) out_k[dst_row * SP_O + 64 * q + col] = z;
    }
}

__global__ void __launch_bounds__(NTH_V2, 4) k_mlp_bwd_sp4_v2_dual_noz_t2(
    const float* __restrict__ EF_k, const float* __restrict__ DEF_k,
    const float* __restrict__ W2_0, const float* __restrict__ W3_0, const float* __restrict__ W4_0,
    const float* __restrict__ z1_0, const float* __restrict__ z2_0, const float* __restrict__ z3_0,
    const float* __restrict__ gout_0, float* dW1_0, float* dW2_0, float* dW3_0,
    const float* __restrict__ W2_1, const float* __restrict__ W3_1, const float* __restrict__ W4_1,
    const float* __restrict__ z1_1, const float* __restrict__ z2_1, const float* __restrict__ z3_1,
    const float* __restrict__ gout_1, float* dW1_1, float* dW2_1, float* dW3_1,
    float act, int Pm) {
    const int lyr = blockIdx.y;
    const float* __restrict__ W2 = lyr ? W2_1 : W2_0;
    const float* __restrict__ W3 = lyr ? W3_1 : W3_0;
    const float* __restrict__ W4 = lyr ? W4_1 : W4_0;
    const float* __restrict__ z1_k = lyr ? z1_1 : z1_0;
    const float* __restrict__ z2_k = lyr ? z2_1 : z2_0;
    const float* __restrict__ z3_k = lyr ? z3_1 : z3_0;
    const float* __restrict__ gout_k = lyr ? gout_1 : gout_0;
    float* dW1 = lyr ? dW1_1 : dW1_0;
    float* dW2 = lyr ? dW2_1 : dW2_0;
    float* dW3 = lyr ? dW3_1 : dW3_0;

    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS_T2 * LDA_V2;
    float* Ws = Bs + ROWS_T2 * LDA_V2;
    float* Xs = Ws + SP_H * LDW_V2;
    const int t = threadIdx.x;
    const int kh = t & 3, row = (t >> 2) & 3, cg = t >> 4, col = 4 * cg + kh;
    const int is_der = row >> 1;
    const long p0 = (long)blockIdx.x * TP_T2;
    const long p = p0 + (row & 1);
    const long src_row = is_der ? (Pm + p) : p;

    if (t < ROWS_T2 * (SP_NB / 4)) {
        const int r = t >> 1, h4 = t & 1;
        const long pp = p0 + (r & (TP_T2 - 1));
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pp < Pm) v = *reinterpret_cast<const float4*>((r < TP_T2 ? EF_k : DEF_k) + pp * SP_NB + 4 * h4);
        *reinterpret_cast<float4*>(Xs + r * SP_NB + 4 * h4) = v;
    }

    float z1v = 0.f, z2v = 0.f, z3v = 0.f;
    if (p < Pm) {
        const long idx = src_row * SP_H + col;
        z1v = z1_k[idx];
        z2v = z2_k[idx];
        z3v = z3_k[idx];
    }
    {
        const float h2 = act_fwd_v2_1r(z2v, is_der, act);
        As[row * LDA_V2 + col] = (p < Pm) ? h2 : 0.f;
    }

    float g = 0.f;
#pragma unroll
    for (int q = 0; q < SP_O / 64; ++q) {
        if (q) __syncthreads();
        if (t < ROWS_T2 * 16) {
            const int r = t >> 4, k4 = t & 15;
            const long pp = p0 + (r & (TP_T2 - 1));
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (pp < Pm) {
                const long grow = (r < TP_T2) ? pp : (Pm + pp);
                v = *reinterpret_cast<const float4*>(gout_k + grow * SP_O + 64 * q + 4 * k4);
            }
            *reinterpret_cast<float4*>(Bs + r * LDA_V2 + 4 * k4) = v;
        }
        load_w_v2_trans(Ws, W4, SP_O, 64 * q);
        __syncthreads();
        gemm_v2_1r<64, LDA_V2>(Bs, Ws, row, cg, kh, g, true);
    }

    g = act_bwd_v2_1r(z3v, g, is_der, act);
    __syncthreads();
    Bs[row * LDA_V2 + col] = (p < Pm) ? g : 0.f;
    load_w_v2_trans(Ws, W3, 64, 0);
    __syncthreads();

    const int dty = t >> 4, dtx = t & 15;
    {
        float d3[4][4] = {};
#pragma unroll
        for (int r = 0; r < ROWS_T2; ++r) {
            const float4 a = *reinterpret_cast<const float4*>(As + r * LDA_V2 + 4 * dty);
            const float4 b = *reinterpret_cast<const float4*>(Bs + r * LDA_V2 + 4 * dtx);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) d3[u][v] += av[u] * bv[v];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) atomicAdd(dW3 + (4 * dty + u) * 64 + 4 * dtx + v, d3[u][v]);
    }

    gemm_v2_1r<64, LDA_V2>(Bs, Ws, row, cg, kh, g);
    g = act_bwd_v2_1r(z2v, g, is_der, act);
    __syncthreads();
    Bs[row * LDA_V2 + col] = (p < Pm) ? g : 0.f;
    {
        const float h1 = act_fwd_v2_1r(z1v, is_der, act);
        As[row * LDA_V2 + col] = (p < Pm) ? h1 : 0.f;
    }
    load_w_v2_trans(Ws, W2, 64, 0);
    __syncthreads();

    {
        float d2[4][4] = {};
#pragma unroll
        for (int r = 0; r < ROWS_T2; ++r) {
            const float4 a = *reinterpret_cast<const float4*>(As + r * LDA_V2 + 4 * dty);
            const float4 b = *reinterpret_cast<const float4*>(Bs + r * LDA_V2 + 4 * dtx);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) d2[u][v] += av[u] * bv[v];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) atomicAdd(dW2 + (4 * dty + u) * 64 + 4 * dtx + v, d2[u][v]);
    }

    gemm_v2_1r<64, LDA_V2>(Bs, Ws, row, cg, kh, g);
    g = act_bwd_v2_1r(z1v, g, is_der, act);
    __syncthreads();
    Bs[row * LDA_V2 + col] = (p < Pm) ? g : 0.f;
    __syncthreads();

    const int a = t >> 5, b0 = 2 * (t & 31);
    float d10 = 0.f, d11 = 0.f;
#pragma unroll
    for (int r = 0; r < ROWS_T2; ++r) {
        const float xa = Xs[r * SP_NB + a];
        const float2 bv = *reinterpret_cast<const float2*>(Bs + r * LDA_V2 + b0);
        d10 += xa * bv.x;
        d11 += xa * bv.y;
    }
    atomicAdd(dW1 + a * 64 + b0, d10);
    atomicAdd(dW1 + a * 64 + b0 + 1, d11);
}

}  // namespace mspl
}  // namespace lst
