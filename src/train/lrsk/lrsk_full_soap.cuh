// Self-contained GPU Block-Jacobi SOAP + Compact Cautious AMSGrad-Adam + Grad-Clip + EMA
// Adapted from LoupMD's battle-tested src/train/strain.cu (Wave 14.3 + Wave 39) for the general
// pure-CUDA LRSK trainer (`src/train/lrsk_full_train.cu`).
//
// Features:
//  - Zero CPU-GPU syncs during an entire epoch (all step counts, norms, clip scales, cautious counts live on GPU).
//  - Supports arbitrary parameter counts and arbitrary 2D/3D matrix block shapes:
//      * Fast 2-CTA/SM shared-memory path for m <= 64, (pn == 0 || n <= 64) with 64-col chunking for wide matrices.
//      * General global-backed Block-Jacobi + tiled bmm path for any 64 < d <= soap_max_dim (e.g. 128 or 256).
//  - Exact cautious AMSGrad-Adam with warp-ballot popc count (ZERO global atomics).
//  - Fused EMA update inside parameter stores.
//  - Decoupled SOAP weight decay + L2/radial-smooth/wd_ro regularizer gradient additions.
#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <type_traits>

namespace lrsk_opt {

constexpr int TS = 64;
constexpr int LD = 65;
constexpr int LDV = 68; // 16-byte aligned + bank-conflict-free stride

template <typename T>
struct PDesc {
    T* p;
    T* g;
    T* ema;
    T* m;      // 1st moment (both Adam and SOAP)
    T* v;      // 2nd moment (Adam only; SOAP uses per-block ws V)
    T* vmax;   // AMSGrad max(v) (Adam only)
    T* dir;    // clipped gradient buffer (SOAP only)
    long n;
    int is_soap;
    int is_ro; // participates in wd_ro L2 regularization
};

struct SoapMat {
    int param;
    int m, n;       // oriented so m <= n (via xi, xj strides)
    int pm, pn;     // 1 = precondition this side; 0 = identity
    int nch;        // ceil(n / TS) when !pn && n > TS && m <= TS, else 1
    int use_smem;   // 1 if m <= TS && (!pn || n <= TS)
    long off, xi, xj;
    long Lm, Qm, Bm;
    long Rn, Qn, Bn;
    long V;
    long Lp;        // partial m*m chunks when nch > 1 && pm
    long W1, W2, W3, W4; // scratch for general (!use_smem) path
};

// =============================================================================
// Math helpers
// =============================================================================
template <typename T>
__device__ __forceinline__ T o_sqrt(T x);
template <> __device__ __forceinline__ float o_sqrt<float>(float x) { return sqrtf(x); }
template <> __device__ __forceinline__ double o_sqrt<double>(double x) { return sqrt(x); }

template <typename T>
__device__ __forceinline__ T o_pow(T a, T b);
template <> __device__ __forceinline__ float o_pow<float>(float a, float b) { return powf(a, b); }
template <> __device__ __forceinline__ double o_pow<double>(double a, double b) { return pow(a, b); }

template <typename T>
__device__ __forceinline__ void ld4(T (&v)[4], const T* p) {
    if constexpr (sizeof(T) == 4) {
        const float4 x = *reinterpret_cast<const float4*>(p);
        v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w;
    } else {
        const double2 x = reinterpret_cast<const double2*>(p)[0];
        const double2 y = reinterpret_cast<const double2*>(p)[1];
        v[0] = x.x; v[1] = x.y; v[2] = y.x; v[3] = y.y;
    }
}

// =============================================================================
// 1. Global Gradient Norm Reduction + Clip Coefficient + Step Prep
//    sc_out[0] = clip_coef, sc_out[1] = adam_step_size, sc_out[2] = adam_bc2_sqrt,
//    sc_out[3] = ema_omd (1 - d_ema), sc_out[4] = inv_soap_bc1, sc_out[5] = inv_soap_bc2
// =============================================================================
template <typename T>
__global__ void k_grad_norm2(const PDesc<T>* __restrict__ d, int np, double* __restrict__ out_norm2) {
    __shared__ double sm[8];
    const int tid = threadIdx.x;
    const int pid = blockIdx.y;
    const PDesc<T> q = d[pid];
    double acc = 0.0;
    for (long i = (long)blockIdx.x * blockDim.x + tid; i < q.n; i += (long)gridDim.x * blockDim.x) {
        const double g = (double)q.g[i];
        acc += g * g;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((tid & 31) == 0) sm[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        double tot = 0.0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += sm[w];
        if (tot != 0.0) atomicAdd(out_norm2, tot);
    }
}

template <typename T>
__global__ void k_opt_prep(
    const double* __restrict__ norm2_ptr, double max_norm, double lr,
    double adam_b1, double adam_b2, double soap_b1, double soap_b2,
    double ema_dmax, int step_t, T* __restrict__ sc_out
) {
    if (threadIdx.x != 0) return;
    const double gn = sqrt(*norm2_ptr);
    double clip = 1.0;
    if (max_norm > 0.0 && gn > max_norm) clip = max_norm / (gn + 1e-12);
    const double t = (double)step_t;
    const double abc1 = 1.0 - pow(adam_b1, t);
    const double abc2s = sqrt(1.0 - pow(adam_b2, t));
    const double d_ema = fmin(ema_dmax, (1.0 + t) / (10.0 + t));
    const double sbc1 = 1.0 - pow(soap_b1, t);
    const double sbc2 = 1.0 - pow(soap_b2, t);
    sc_out[0] = (T)clip;
    sc_out[1] = (T)(lr / abc1);
    sc_out[2] = (T)abc2s;
    sc_out[3] = (T)(1.0 - d_ema);
    sc_out[4] = (T)(1.0 / sbc1);
    sc_out[5] = (T)(1.0 / sbc2);
}

// =============================================================================
// 2. Compact Cautious AMSGrad-Adam (1 CTA per non-SOAP param) + SOAP Prep (decoupled WD + m + dir)
// =============================================================================
template <typename T>
__global__ void __launch_bounds__(256) k_soap_prep_params(
    const PDesc<T>* __restrict__ d, const int* __restrict__ soap_pids,
    const T* __restrict__ sc, double lr, double mult, double wd, double soap_b1
) {
    const int pid = soap_pids[blockIdx.y];
    const PDesc<T> q = d[pid];
    const T coef = sc[0];
    const T dec = wd > 0.0 ? (T)(1.0 - lr * mult * wd) : T(1);
    const T b1 = (T)soap_b1;
    const T omb1 = (T)(1.0 - soap_b1);
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < q.n; i += (long)gridDim.x * blockDim.x) {
        if (wd > 0.0) q.p[i] *= dec;
        const T g = q.g[i] * coef;
        q.m[i] = b1 * q.m[i] + omb1 * g;
        q.dir[i] = g;
    }
}

template <typename T>
__global__ void __launch_bounds__(256) k_caut_adam_compact(
    const PDesc<T>* __restrict__ d, const int* __restrict__ adam_pids,
    const T* __restrict__ sc, double b1_d, double b2_d, double eps_d, int do_caut
) {
    __shared__ int sm_c[8];
    const int pid = adam_pids[blockIdx.x];
    const PDesc<T> q = d[pid];
    const int tid = threadIdx.x;
    const T coef = sc[0];
    const T step_size = sc[1];
    const T bc2s = sc[2];
    const T omd = sc[3];
    const T b1 = (T)b1_d, omb1 = (T)(1.0 - b1_d);
    const T b2 = (T)b2_d, omb2 = (T)(1.0 - b2_d);
    const T eps = (T)eps_d;
    const bool do_ema = (q.ema != nullptr);

    if (!do_caut) {
        for (long i = tid; i < q.n; i += 256) {
            T p = q.p[i];
            const T g = q.g[i] * coef;
            const T m = b1 * q.m[i] + omb1 * g;
            T v = b2 * q.v[i] + omb2 * g * g;
            q.m[i] = m; q.v[i] = v;
            if (q.vmax) { v = v > q.vmax[i] ? v : q.vmax[i]; q.vmax[i] = v; }
            p -= step_size * m / (o_sqrt(v) / bc2s + eps);
            q.p[i] = p;
            if (do_ema) q.ema[i] -= omd * (q.ema[i] - p);
        }
        return;
    }

    if (q.n <= 256) {
        T p = T(0), m = T(0), v = T(0);
        int pos = 0;
        if (tid < q.n) {
            p = q.p[tid];
            const T g = q.g[tid] * coef;
            m = b1 * q.m[tid] + omb1 * g;
            v = b2 * q.v[tid] + omb2 * g * g;
            q.m[tid] = m; q.v[tid] = v;
            if (q.vmax) { v = v > q.vmax[tid] ? v : q.vmax[tid]; q.vmax[tid] = v; }
            if (m * g > T(0)) pos = 1;
        }
        const int wcnt = __popc(__ballot_sync(0xffffffffu, pos != 0));
        if ((tid & 31) == 0) sm_c[tid >> 5] = wcnt;
        __syncthreads();
        int tot = 0;
        #pragma unroll
        for (int k = 0; k < 8; ++k) tot += sm_c[k];
        if (tid < q.n) {
            if (pos) {
                const T cs = (T)((double)q.n / ((double)tot + 1.0));
                const T denom = o_sqrt(v) / bc2s + eps;
                p -= cs * (step_size * m / denom);
            }
            q.p[tid] = p;
            if (do_ema) q.ema[tid] -= omd * (q.ema[tid] - p);
        }
        return;
    }

    // Two-pass for q.n > 256: pass 1 updates m, v, vmax and counts m*g > 0; pass 2 applies cautious update + EMA
    int cnt = 0;
    for (long i = tid; i < q.n; i += 256) {
        const T g = q.g[i] * coef;
        const T m = b1 * q.m[i] + omb1 * g;
        T v = b2 * q.v[i] + omb2 * g * g;
        q.m[i] = m; q.v[i] = v;
        if (q.vmax) { v = v > q.vmax[i] ? v : q.vmax[i]; q.vmax[i] = v; }
        if (m * g > T(0)) ++cnt;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
    if ((tid & 31) == 0) sm_c[tid >> 5] = cnt;
    __syncthreads();
    int tot = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) tot += sm_c[k];
    const T cs = (T)((double)q.n / ((double)tot + 1.0));
    for (long i = tid; i < q.n; i += 256) {
        T p = q.p[i];
        const T g = q.g[i] * coef;
        const T m = q.m[i];
        const T v = q.vmax ? q.vmax[i] : q.v[i];
        if (m * g > T(0)) {
            const T denom = o_sqrt(v) / bc2s + eps;
            p -= cs * (step_size * m / denom);
            q.p[i] = p;
        }
        if (do_ema) q.ema[i] -= omd * (q.ema[i] - p);
    }
}

// =============================================================================
// 3. Block-Jacobi + Shared-Memory GEMM Helpers (from strain.cu, extended to d <= 256)
// =============================================================================
template <typename T>
struct alignas(2 * sizeof(T)) JRot { T c, s; };

// Fast SMEM cyclic Jacobi for d <= 64 (Bs[d*LD], Qs[d*LD] preloaded or loaded from Bg/Qg)
template <typename T>
__device__ void jacobi_smem(const T* Bg, T* Qg, int d, int maxsw, double tol, T* Bs, T* Qs, bool preloaded = false) {
    __shared__ JRot<T> rot_[TS / 2 + 1];
    __shared__ int2 pq_[TS / 2 + 1];
    __shared__ double jred[2][32];
    const int nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (!preloaded) {
        for (int e = threadIdx.x; e < d * d; e += blockDim.x) {
            const int i = e / d, j = e % d;
            Bs[i * LD + j] = Bg[e];
            Qs[i * LD + j] = Qg[e];
        }
    }
    const int n = d + (d & 1), np = n / 2, n1 = n - 1;
    const int bstep = blockDim.x / np, kb0 = threadIdx.x % np, ka0 = threadIdx.x / np;
    const int qstep = blockDim.x / d, qk0 = threadIdx.x / d;
    T* const qrow = Qs + (threadIdx.x % d) * LD;
    const bool bact = ka0 < bstep, qact = qk0 < qstep;
    __syncthreads();
    if (d < 2) {
        for (int e = threadIdx.x; e < d * d; e += blockDim.x) Qg[e] = Qs[(e / d) * LD + (e % d)];
        __syncthreads();
        return;
    }
    const double tol2 = tol * tol;
    for (int sw = 0; sw < maxsw; ++sw) {
        double off = 0.0, tot = 0.0;
        for (int e = threadIdx.x; e < d * d; e += blockDim.x) {
            const int i = e / d, j = e % d;
            const double x = (double)Bs[i * LD + j];
            const double xx = x * x;
            tot += xx;
            if (i != j) off += xx;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            off += __shfl_xor_sync(0xffffffffu, off, o);
            tot += __shfl_xor_sync(0xffffffffu, tot, o);
        }
        if (lane == 0) { jred[0][warp] = off; jred[1][warp] = tot; }
        __syncthreads();
        off = 0.0; tot = 0.0;
        for (int w = 0; w < nw; ++w) { off += jred[0][w]; tot += jred[1][w]; }
        __syncthreads();
        if (off <= tol2 * tot) break;

        for (int rd = 0; rd < n1; ++rd) {
            if (threadIdx.x < np) {
                const int k = threadIdx.x;
                int p = k == 0 ? 0 : 1 + (k - 1 - rd + n1) % n1;
                int q = 1 + (n1 - 1 - k - rd + n1) % n1;
                if (p > q) { int tmp = p; p = q; q = tmp; }
                pq_[k] = make_int2(p, q);
                if (q < d) {
                    const T app = Bs[p * LD + p], aqq = Bs[q * LD + q], apq = Bs[p * LD + q];
                    JRot<T> r;
                    if constexpr (std::is_same<T, float>::value) {
                        if (fabsf(apq) <= 1e-12f * (fabsf(app) + fabsf(aqq))) {
                            r.c = 1.f; r.s = 0.f;
                        } else {
                            const float tau = (aqq - app) / (2.f * apq);
                            const float tv = fabsf(tau) > 1e7f
                                ? (0.5f / tau)
                                : (copysignf(1.f, tau) / (fabsf(tau) + sqrtf(1.f + tau * tau)));
                            r.c = rsqrtf(1.f + tv * tv);
                            r.s = tv * r.c;
                        }
                    } else {
                        const double th = 0.5 * atan2(2.0 * apq, aqq - app);
                        r.c = cos(th); r.s = sin(th);
                    }
                    rot_[k] = r;
                }
            }
            __syncthreads();
            if (bact) {
                const int2 pqc = pq_[kb0];
                const bool cb = pqc.y < d;
                const JRot<T> rc = rot_[kb0];
                for (int ka = ka0; ka < np; ka += bstep) {
                    const int2 pq = pq_[ka];
                    const bool ra = pq.y < d;
                    T* const rp = Bs + pq.x * LD;
                    T* const rq = Bs + pq.y * LD;
                    T x00 = rp[pqc.x], x01 = cb ? rp[pqc.y] : T(0);
                    T x10 = ra ? rq[pqc.x] : T(0), x11 = ra && cb ? rq[pqc.y] : T(0);
                    if (ra) {
                        const JRot<T> r = rot_[ka];
                        const T y00 = r.c * x00 - r.s * x10, y10 = r.s * x00 + r.c * x10;
                        const T y01 = r.c * x01 - r.s * x11, y11 = r.s * x01 + r.c * x11;
                        x00 = y00; x10 = y10; x01 = y01; x11 = y11;
                    }
                    if (cb) {
                        const T y00 = rc.c * x00 - rc.s * x01, y01 = rc.s * x00 + rc.c * x01;
                        const T y10 = rc.c * x10 - rc.s * x11, y11 = rc.s * x10 + rc.c * x11;
                        x00 = y00; x01 = y01; x10 = y10; x11 = y11;
                    }
                    rp[pqc.x] = x00;
                    if (cb) rp[pqc.y] = x01;
                    if (ra) { rq[pqc.x] = x10; if (cb) rq[pqc.y] = x11; }
                }
            }
            if (qact) {
                for (int k = qk0; k < np; k += qstep) {
                    const int2 pq = pq_[k];
                    if (pq.y >= d) continue;
                    const JRot<T> r = rot_[k];
                    const T qp = qrow[pq.x], qv = qrow[pq.y];
                    qrow[pq.x] = r.c * qp - r.s * qv;
                    qrow[pq.y] = r.s * qp + r.c * qv;
                }
            }
            __syncthreads();
        }
    }
    for (int e = threadIdx.x; e < d * d; e += blockDim.x) Qg[e] = Qs[(e / d) * LD + (e % d)];
    __syncthreads();
}

// Fast SMEM cyclic Jacobi for FP32 with 64 < d <= 128:
// Keeps Bs[d * 129] in dynamic SMEM (<= 66,048 B) with 1-pass 2x2 rotations,
// and transposes Q -> Qt (row-major columns!) in global scratch for 100% coalesced L2 updates!
template <typename T>
__device__ void jacobi_smem_128(const T* Bg, T* Qg, T* Qt, int d, int maxsw, double tol, T* Bs) {
    constexpr int LD128 = 129;
    __shared__ JRot<T> rot_[64];
    __shared__ int2 pq_[64];
    __shared__ double jred[2][32];
    const int nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int e = threadIdx.x; e < d * d; e += blockDim.x) {
        const int i = e / d, j = e % d;
        Bs[i * LD128 + j] = Bg[e];
        Qt[j * d + i] = Qg[e];
    }
    const int n = d + (d & 1), np = n / 2, n1 = n - 1;
    const int bstep = blockDim.x / np, kb0 = threadIdx.x % np, ka0 = threadIdx.x / np;
    const bool bact = ka0 < bstep;
    __syncthreads();
    const double tol2 = tol * tol;
    for (int sw = 0; sw < maxsw; ++sw) {
        double off = 0.0, tot = 0.0;
        for (int e = threadIdx.x; e < d * d; e += blockDim.x) {
            const int i = e / d, j = e % d;
            const double x = (double)Bs[i * LD128 + j];
            const double xx = x * x;
            tot += xx;
            if (i != j) off += xx;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            off += __shfl_xor_sync(0xffffffffu, off, o);
            tot += __shfl_xor_sync(0xffffffffu, tot, o);
        }
        if (lane == 0) { jred[0][warp] = off; jred[1][warp] = tot; }
        __syncthreads();
        off = 0.0; tot = 0.0;
        for (int w = 0; w < nw; ++w) { off += jred[0][w]; tot += jred[1][w]; }
        __syncthreads();
        if (off <= tol2 * tot) break;

        for (int rd = 0; rd < n1; ++rd) {
            if (threadIdx.x < np) {
                const int k = threadIdx.x;
                int p = k == 0 ? 0 : 1 + (k - 1 - rd + n1) % n1;
                int q = 1 + (n1 - 1 - k - rd + n1) % n1;
                if (p > q) { int tmp = p; p = q; q = tmp; }
                pq_[k] = make_int2(p, q);
                if (q < d) {
                    const T app = Bs[p * LD128 + p], aqq = Bs[q * LD128 + q], apq = Bs[p * LD128 + q];
                    JRot<T> r;
                    if constexpr (std::is_same<T, float>::value) {
                        if (fabsf(apq) <= 1e-12f * (fabsf(app) + fabsf(aqq))) {
                            r.c = 1.f; r.s = 0.f;
                        } else {
                            const float tau = (aqq - app) / (2.f * apq);
                            const float tv = fabsf(tau) > 1e7f
                                ? (0.5f / tau)
                                : (copysignf(1.f, tau) / (fabsf(tau) + sqrtf(1.f + tau * tau)));
                            r.c = rsqrtf(1.f + tv * tv);
                            r.s = tv * r.c;
                        }
                    } else {
                        const double th = 0.5 * atan2(2.0 * apq, aqq - app);
                        r.c = cos(th); r.s = sin(th);
                    }
                    rot_[k] = r;
                }
            }
            __syncthreads();
            if (bact) {
                const int2 pqc = pq_[kb0];
                const bool cb = pqc.y < d;
                const JRot<T> rc = rot_[kb0];
                for (int ka = ka0; ka < np; ka += bstep) {
                    const int2 pq = pq_[ka];
                    const bool ra = pq.y < d;
                    T* const rp = Bs + pq.x * LD128;
                    T* const rq = Bs + pq.y * LD128;
                    T x00 = rp[pqc.x], x01 = cb ? rp[pqc.y] : T(0);
                    T x10 = ra ? rq[pqc.x] : T(0), x11 = ra && cb ? rq[pqc.y] : T(0);
                    if (ra) {
                        const JRot<T> r = rot_[ka];
                        const T y00 = r.c * x00 - r.s * x10, y10 = r.s * x00 + r.c * x10;
                        const T y01 = r.c * x01 - r.s * x11, y11 = r.s * x01 + r.c * x11;
                        x00 = y00; x10 = y10; x01 = y01; x11 = y11;
                    }
                    if (cb) {
                        const T y00 = rc.c * x00 - rc.s * x01, y01 = rc.s * x00 + rc.c * x01;
                        const T y10 = rc.c * x10 - rc.s * x11, y11 = rc.s * x10 + rc.c * x11;
                        x00 = y00; x01 = y01; x10 = y10; x11 = y11;
                    }
                    rp[pqc.x] = x00;
                    if (cb) rp[pqc.y] = x01;
                    if (ra) { rq[pqc.x] = x10; if (cb) rq[pqc.y] = x11; }
                }
            }
            for (int e = threadIdx.x; e < np * d; e += blockDim.x) {
                const int k = e / d, i = e % d;
                const int2 pq = pq_[k];
                if (pq.y >= d) continue;
                const JRot<T> r = rot_[k];
                if (r.s == T(0)) continue;
                T* const qp = Qt + pq.x * d;
                T* const qq = Qt + pq.y * d;
                const T vp = qp[i], vq = qq[i];
                qp[i] = r.c * vp - r.s * vq;
                qq[i] = r.s * vp + r.c * vq;
            }
            __syncthreads();
        }
    }
    for (int e = threadIdx.x; e < d * d; e += blockDim.x) {
        const int i = e / d, j = e % d;
        Qg[i * d + j] = Qt[j * d + i];
    }
    __syncthreads();
}

// General global-memory cyclic Jacobi for any 2 <= d <= 256 (works for arbitrary architectures!)
template <typename T>
__device__ void jacobi_global(T* Bg, T* Qg, int d, int maxsw, double tol) {
    __shared__ JRot<T> rot_[128];
    __shared__ int2 pq_[128];
    __shared__ double jred[2][32];
    const int nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (d < 2) return;
    const int n = d + (d & 1), np = n / 2, n1 = n - 1;
    const double tol2 = tol * tol;
    for (int sw = 0; sw < maxsw; ++sw) {
        double off = 0.0, tot = 0.0;
        for (int e = threadIdx.x; e < d * d; e += blockDim.x) {
            const int i = e / d, j = e % d;
            const double x = (double)Bg[e];
            tot += x * x;
            if (i != j) off += x * x;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            off += __shfl_xor_sync(0xffffffffu, off, o);
            tot += __shfl_xor_sync(0xffffffffu, tot, o);
        }
        if (lane == 0) { jred[0][warp] = off; jred[1][warp] = tot; }
        __syncthreads();
        off = 0.0; tot = 0.0;
        for (int w = 0; w < nw; ++w) { off += jred[0][w]; tot += jred[1][w]; }
        __syncthreads();
        if (off <= tol2 * tot) break;

        for (int rd = 0; rd < n1; ++rd) {
            for (int k = threadIdx.x; k < np; k += blockDim.x) {
                int p = k == 0 ? 0 : 1 + (k - 1 - rd + n1) % n1;
                int q = 1 + (n1 - 1 - k - rd + n1) % n1;
                if (p > q) { int tmp = p; p = q; q = tmp; }
                pq_[k] = make_int2(p, q);
                if (q < d) {
                    const T app = Bg[p * d + p], aqq = Bg[q * d + q], apq = Bg[p * d + q];
                    JRot<T> r;
                    if constexpr (std::is_same<T, float>::value) {
                        if (fabsf(apq) <= 1e-12f * (fabsf(app) + fabsf(aqq))) {
                            r.c = 1.f; r.s = 0.f;
                        } else {
                            const float tau = (aqq - app) / (2.f * apq);
                            const float tv = fabsf(tau) > 1e7f
                                ? (0.5f / tau)
                                : (copysignf(1.f, tau) / (fabsf(tau) + sqrtf(1.f + tau * tau)));
                            r.c = rsqrtf(1.f + tv * tv);
                            r.s = tv * r.c;
                        }
                    } else {
                        const double th = 0.5 * atan2(2.0 * apq, aqq - app);
                        r.c = cos(th); r.s = sin(th);
                    }
                    rot_[k] = r;
                }
            }
            __syncthreads();
            // Rows: B <- J^T B
            for (int e = threadIdx.x; e < np * d; e += blockDim.x) {
                const int k = e / d, j = e % d;
                const int2 pq = pq_[k];
                if (pq.y >= d) continue;
                const JRot<T> r = rot_[k];
                const T bp = Bg[pq.x * d + j], bq = Bg[pq.y * d + j];
                Bg[pq.x * d + j] = r.c * bp - r.s * bq;
                Bg[pq.y * d + j] = r.s * bp + r.c * bq;
            }
            __syncthreads();
            // Cols: B <- B J, Q <- Q J
            for (int e = threadIdx.x; e < np * d; e += blockDim.x) {
                const int k = e / d, i = e % d;
                const int2 pq = pq_[k];
                if (pq.y >= d) continue;
                const JRot<T> r = rot_[k];
                const T bp = Bg[i * d + pq.x], bq = Bg[i * d + pq.y];
                Bg[i * d + pq.x] = r.c * bp - r.s * bq;
                Bg[i * d + pq.y] = r.s * bp + r.c * bq;
                const T qp = Qg[i * d + pq.x], qq = Qg[i * d + pq.y];
                Qg[i * d + pq.x] = r.c * qp - r.s * qq;
                Qg[i * d + pq.y] = r.s * qp + r.c * qq;
            }
            __syncthreads();
        }
    }
}

// Tiled general bmm for any M, N, K (supports M > 64 too by tiling M in chunks of TS!)
template <typename T>
__device__ void bmm_any(
    T* C, long ldc, const T* A, long ai, long ak, const T* B, long bi, long bj,
    int M, int N, int K, T alpha, T beta, T* As, T* Bs
) {
    const int r0 = (threadIdx.x >> 4) * 4, c0 = (threadIdx.x & 15) * 4;
    const bool arow = (ai == 1);
    const bool bk = (bi == 1 && bj != 1);
    for (int m0 = 0; m0 < M; m0 += TS) {
        for (int n0 = 0; n0 < N; n0 += TS) {
            T acc[4][4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
            for (int k0 = 0; k0 < K; k0 += TS) {
                constexpr int PER = TS * TS / 256;
                T va[PER], vb[PER];
                #pragma unroll
                for (int u = 0; u < PER; ++u) {
                    const int e = threadIdx.x + u * 256, hi = e >> 6, lo = e & 63;
                    const int r = arow ? lo : hi, ka = arow ? hi : lo;
                    va[u] = (m0 + r < M && k0 + ka < K) ? A[(long)(m0 + r) * ai + (long)(k0 + ka) * ak] : T(0);
                    const int kb = bk ? lo : hi, c = bk ? hi : lo;
                    vb[u] = (k0 + kb < K && n0 + c < N) ? B[(long)(k0 + kb) * bi + (long)(n0 + c) * bj] : T(0);
                }
                #pragma unroll
                for (int u = 0; u < PER; ++u) {
                    const int e = threadIdx.x + u * 256, hi = e >> 6, lo = e & 63;
                    As[(arow ? hi : lo) * LDV + (arow ? lo : hi)] = va[u];
                    Bs[(bk ? lo : hi) * LDV + (bk ? hi : lo)] = vb[u];
                }
                __syncthreads();
                #pragma unroll 8
                for (int kk = 0; kk < TS; ++kk) {
                    T a[4], b[4];
                    ld4<T>(a, As + kk * LDV + r0);
                    ld4<T>(b, Bs + kk * LDV + c0);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
                }
                __syncthreads();
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = m0 + r0 + i, c = n0 + c0 + j;
                    if (r < M && c < N) {
                        T* o = C + (long)r * ldc + c;
                        *o = (beta != T(0)) ? (alpha * acc[i][j] + beta * (*o)) : (alpha * acc[i][j]);
                    }
                }
        }
    }
    __syncthreads();
}

template <typename T>
__device__ __forceinline__ void smem_gemm(T (&acc)[4][4], const T* A_cm, const T* B_rm, int K) {
    const int r0 = (threadIdx.x >> 4) * 4, c0 = (threadIdx.x & 15) * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    #pragma unroll 8
    for (int k = 0; k < K; ++k) {
        T a[4], b[4];
        ld4<T>(a, A_cm + k * LDV + r0);
        ld4<T>(b, B_rm + k * LDV + c0);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
    }
}

template <typename T>
__device__ __forceinline__ void load_mat_smem(T* S, const T* X, long xi, long xj, int m, int nc) {
    const bool col_fast = (xi == 1 && xj != 1);
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        T v[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int e = threadIdx.x + (g * 4 + u) * 256, hi = e >> 6, lo = e & 63;
            const int r = col_fast ? lo : hi, c = col_fast ? hi : lo;
            v[u] = (r < m && c < nc) ? X[(long)r * xi + (long)c * xj] : T(0);
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int e = threadIdx.x + (g * 4 + u) * 256, hi = e >> 6, lo = e & 63;
            const int r = col_fast ? lo : hi, c = col_fast ? hi : lo;
            S[r * LDV + c] = v[u];
        }
    }
}

template <typename T>
__device__ __forceinline__ void load_sq_smem(T* S, const T* Q, int d) {
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        T v[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int e = threadIdx.x + (g * 4 + u) * 256, r = e >> 6, c = e & 63;
            v[u] = (r < d && c < d) ? Q[(long)r * d + c] : T(0);
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int e = threadIdx.x + (g * 4 + u) * 256, r = e >> 6, c = e & 63;
            S[r * LDV + c] = v[u];
        }
    }
}

template <typename T>
__device__ __forceinline__ void load_sq_smem_T(T* S, const T* Q, int d) {
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        T v[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int e = threadIdx.x + (g * 4 + u) * 256, r = e >> 6, c = e & 63;
            v[u] = (r < d && c < d) ? Q[(long)r * d + c] : T(0);
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int e = threadIdx.x + (g * 4 + u) * 256, r = e >> 6, c = e & 63;
            S[c * LDV + r] = v[u];
        }
    }
}

template <typename T>
__device__ void smem_qtlq_jacobi(const T* L, T* Q, int d, int maxsw, double tol, T* S0, T* S1) {
    load_sq_smem_T<T>(S0, L, d); // S0[k*LDV + r] = L[r, k]
    load_sq_smem<T>(S1, Q, d);   // S1[k*LDV + c] = Q[k, c]
    __syncthreads();
    const int r0 = (threadIdx.x >> 4) * 4, c0 = (threadIdx.x & 15) * 4;
    T acc[4][4];
    smem_gemm<T>(acc, S0, S1, d); // L Q
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            S0[(r0 + i) * LDV + (c0 + j)] = (r0 + i < d && c0 + j < d) ? acc[i][j] : T(0);
    __syncthreads();
    smem_gemm<T>(acc, S1, S0, d); // Q^T (L Q)
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            if (r0 + i < d && c0 + j < d) S0[(r0 + i) * LD + (c0 + j)] = acc[i][j];
    for (int e = threadIdx.x; e < d * d; e += blockDim.x) S1[(e / d) * LD + (e % d)] = Q[e];
    jacobi_smem<T>(nullptr, Q, d, maxsw, tol, S0, S1, true);
}

// =============================================================================
// 4. SOAP Apply (SMEM path + General path) and Kernels (k_soap_a, k_soap_eig, k_soap_ap)
// =============================================================================
template <typename T>
__device__ void soap_apply_smem(
    const SoapMat& mt, const PDesc<T>& q, T* ws, int c0, int nc,
    const T* sc, double lr_mult, double b2_d, double eps_d, T* S0, T* S1, bool x_in_s0
) {
    const int m = mt.m, n = mt.n;
    const long xi = mt.xi, xj = mt.xj;
    const T* X = q.dir + mt.off + (long)c0 * xj;
    const T* Mx = q.m + mt.off + (long)c0 * xj;
    const T *Qm = ws + mt.Qm, *Qn = ws + mt.Qn;
    T* V = ws + mt.V + c0;
    const T b2 = (T)b2_d, a2 = (T)(1.0 - b2_d), eps = (T)eps_d;
    const T inv_bc1 = sc[4], inv_bc2 = sc[5], omd = sc[3];
    const int r0 = (threadIdx.x >> 4) * 4, c0t = (threadIdx.x & 15) * 4;

    if (!x_in_s0) load_mat_smem<T>(S0, X, xi, xj, m, nc);
    if (mt.pm) {
        load_sq_smem<T>(S1, Qm, m);
        __syncthreads();
    } else {
        __syncthreads();
    }
    T acc[4][4];
    if (mt.pm) {
        smem_gemm<T>(acc, S1, S0, m); // Qm^T X
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = S0[(r0 + i) * LDV + (c0t + j)];
    }
    if (mt.pn) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                S0[(c0t + j) * LDV + (r0 + i)] = (r0 + i < m && c0t + j < nc) ? acc[i][j] : T(0);
        load_sq_smem<T>(S1, Qn, n);
        __syncthreads();
        smem_gemm<T>(acc, S0, S1, n); // (Qm^T X) Qn
    }
    T denom[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = r0 + i, c = c0t + j;
            if (r < m && c < nc) {
                const long k = (long)r * n + c;
                const T g = acc[i][j];
                const T v = b2 * V[k] + a2 * g * g;
                V[k] = v;
                denom[i][j] = o_sqrt(v * inv_bc2) + eps;
            }
        }
    __syncthreads();
    load_mat_smem<T>(S0, Mx, xi, xj, m, nc);
    if (mt.pm) load_sq_smem<T>(S1, Qm, m);
    __syncthreads();
    if (mt.pm) {
        smem_gemm<T>(acc, S1, S0, m); // Qm^T M
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = S0[(r0 + i) * LDV + (c0t + j)];
    }
    if (mt.pn) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                S0[(c0t + j) * LDV + (r0 + i)] = (r0 + i < m && c0t + j < nc) ? acc[i][j] : T(0);
        load_sq_smem<T>(S1, Qn, n);
        __syncthreads();
        smem_gemm<T>(acc, S0, S1, n); // Mr = (Qm^T M) Qn
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = r0 + i, c = c0t + j;
            S0[r * LDV + c] = (r < m && c < nc) ? ((acc[i][j] * inv_bc1) / denom[i][j]) : T(0);
        }
    if (mt.pm) load_sq_smem_T<T>(S1, Qm, m);
    __syncthreads();
    if (mt.pm) {
        smem_gemm<T>(acc, S1, S0, m); // Qm Ur
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = S0[(r0 + i) * LDV + (c0t + j)];
    }
    if (mt.pn) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                S0[(c0t + j) * LDV + (r0 + i)] = (r0 + i < m && c0t + j < nc) ? acc[i][j] : T(0);
        load_sq_smem_T<T>(S1, Qn, n);
        __syncthreads();
        smem_gemm<T>(acc, S0, S1, n); // (Qm Ur) Qn^T
    }
    const T f = (T)lr_mult;
    const bool do_ema = (q.ema != nullptr);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = r0 + i, c = c0t + j;
            if (r < m && c < nc) {
                const long idx = mt.off + (long)r * xi + (long)(c0 + c) * xj;
                const T pnew = q.p[idx] - f * acc[i][j];
                q.p[idx] = pnew;
                if (do_ema) q.ema[idx] -= omd * (q.ema[idx] - pnew);
            }
        }
}

template <typename T>
__device__ void soap_apply_general(
    const SoapMat& mt, const PDesc<T>& q, T* ws,
    const T* sc, double lr_mult, double b2_d, double eps_d, T* As, T* Bs
) {
    const int m = mt.m, n = mt.n;
    const long xi = mt.xi, xj = mt.xj;
    const T* X = q.dir + mt.off;
    const T* Mx = q.m + mt.off;
    const T *Qm = ws + mt.Qm, *Qn = ws + mt.Qn;
    T *V = ws + mt.V;
    T *W1 = ws + mt.W1, *W2 = ws + mt.W2, *W3 = ws + mt.W3, *W4 = ws + mt.W4;
    const T b2 = (T)b2_d, a2 = (T)(1.0 - b2_d), eps = (T)eps_d;
    const T inv_bc1 = sc[4], inv_bc2 = sc[5], omd = sc[3];

    if (mt.pm) {
        bmm_any<T>(W1, n, Qm, 1, m, X, xi, xj, m, n, m, T(1), T(0), As, Bs);
        bmm_any<T>(W3, n, Qm, 1, m, Mx, xi, xj, m, n, m, T(1), T(0), As, Bs);
    } else {
        for (long e = threadIdx.x; e < (long)m * n; e += blockDim.x) {
            const int r = (int)(e / n), c = (int)(e % n);
            W1[e] = X[(long)r * xi + (long)c * xj];
            W3[e] = Mx[(long)r * xi + (long)c * xj];
        }
        __syncthreads();
    }
    T *Gr = W1, *Mr = W3;
    if (mt.pn) {
        bmm_any<T>(W2, n, W1, n, 1, Qn, n, 1, m, n, n, T(1), T(0), As, Bs);
        bmm_any<T>(W4, n, W3, n, 1, Qn, n, 1, m, n, n, T(1), T(0), As, Bs);
        Gr = W2; Mr = W4;
    }
    for (long k = threadIdx.x; k < (long)m * n; k += blockDim.x) {
        const T g = Gr[k];
        const T v = b2 * V[k] + a2 * g * g;
        V[k] = v;
        Gr[k] = (Mr[k] * inv_bc1) / (o_sqrt(v * inv_bc2) + eps);
    }
    __syncthreads();
    T* U = Gr;
    if (mt.pm && mt.pn) {
        bmm_any<T>(W3, n, Qm, m, 1, Gr, n, 1, m, n, m, T(1), T(0), As, Bs);
        bmm_any<T>(W1, n, W3, n, 1, Qn, 1, n, m, n, n, T(1), T(0), As, Bs);
        U = W1;
    } else if (mt.pm) {
        bmm_any<T>(W3, n, Qm, m, 1, Gr, n, 1, m, n, m, T(1), T(0), As, Bs);
        U = W3;
    } else if (mt.pn) {
        bmm_any<T>(W3, n, Gr, n, 1, Qn, 1, n, m, n, n, T(1), T(0), As, Bs);
        U = W3;
    }
    const T f = (T)lr_mult;
    const bool do_ema = (q.ema != nullptr);
    for (long k = threadIdx.x; k < (long)m * n; k += blockDim.x) {
        const int r = (int)(k / n), c = (int)(k % n);
        const long idx = mt.off + (long)r * xi + (long)c * xj;
        const T pnew = q.p[idx] - f * U[k];
        q.p[idx] = pnew;
        if (do_ema) q.ema[idx] -= omd * (q.ema[idx] - pnew);
    }
}

template <typename T>
__global__ void __launch_bounds__(256) k_soap_a(
    const SoapMat* __restrict__ mats, const int2* __restrict__ units,
    const PDesc<T>* __restrict__ d, const T* __restrict__ sc, T* __restrict__ ws,
    double lr_mult, double b2_d, double eps_d, int do_refresh
) {
    extern __shared__ unsigned char smem_raw[];
    T* S0 = reinterpret_cast<T*>(smem_raw);
    T* S1 = S0 + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<T> q = d[mt.param];
    const int m = mt.m, n = mt.n;
    const T b2 = (T)b2_d, a2 = (T)(1.0 - b2_d);

    if (mt.use_smem) {
        const int c0 = (mt.nch > 1) ? (u.y * TS) : 0;
        const int nc = (mt.nch > 1) ? min(TS, n - c0) : n;
        const T* X = q.dir + mt.off + (long)c0 * mt.xj;
        const long xi = mt.xi, xj = mt.xj;
        const int r0 = (threadIdx.x >> 4) * 4, c0t = (threadIdx.x & 15) * 4;
        const bool col_fast = (xi == 1 && xj != 1);
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            T v[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e = threadIdx.x + (g * 4 + k) * 256, hi = e >> 6, lo = e & 63;
                const int r = col_fast ? lo : hi, c = col_fast ? hi : lo;
                v[k] = (r < m && c < nc) ? X[(long)r * xi + (long)c * xj] : T(0);
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e = threadIdx.x + (g * 4 + k) * 256, hi = e >> 6, lo = e & 63;
                const int r = col_fast ? lo : hi, c = col_fast ? hi : lo;
                S0[r * LDV + c] = v[k];
                if (mt.pm) S1[c * LDV + r] = v[k];
            }
        }
        __syncthreads();
        T acc[4][4];
        if (mt.pm) {
            smem_gemm<T>(acc, S1, S1, nc);
            T* L = (mt.nch > 1) ? (ws + mt.Lp + (long)u.y * m * m) : (ws + mt.Lm);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = r0 + i, c = c0t + j;
                    if (r < m && c < m) L[(long)r * m + c] = a2 * acc[i][j] + b2 * L[(long)r * m + c];
                }
        }
        if (mt.pn) {
            smem_gemm<T>(acc, S0, S0, m);
            T* R = ws + mt.Rn;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = r0 + i, c = c0t + j;
                    if (r < n && c < n) R[(long)r * n + c] = a2 * acc[i][j] + b2 * R[(long)r * n + c];
                }
        }
        if (do_refresh) return;
        __syncthreads();
        soap_apply_smem<T>(mt, q, ws, c0, nc, sc, lr_mult, b2_d, eps_d, S0, S1, true);
    } else {
        const T* X = q.dir + mt.off;
        if (mt.pm) bmm_any<T>(ws + mt.Lm, m, X, mt.xi, mt.xj, X, mt.xj, mt.xi, m, m, n, a2, b2, S0, S1);
        if (mt.pn) bmm_any<T>(ws + mt.Rn, n, X, mt.xj, mt.xi, X, mt.xi, mt.xj, n, n, m, a2, b2, S0, S1);
        if (do_refresh) return;
        soap_apply_general<T>(mt, q, ws, sc, lr_mult, b2_d, eps_d, S0, S1);
    }
}

template <typename T>
__global__ void __launch_bounds__(256) k_soap_eig(
    const SoapMat* __restrict__ mats, const int2* __restrict__ eig_units,
    T* __restrict__ ws, int maxsw, double tol
) {
    extern __shared__ unsigned char smem_raw[];
    T* S0 = reinterpret_cast<T*>(smem_raw);
    T* S1 = S0 + TS * LDV;
    const int2 u = eig_units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const bool is_left = (u.y == 0);
    const int d = is_left ? mt.m : mt.n;
    T* L = is_left ? (ws + mt.Lm) : (ws + mt.Rn);
    T* Q = is_left ? (ws + mt.Qm) : (ws + mt.Qn);
    T* B = is_left ? (ws + mt.Bm) : (ws + mt.Bn);
    if (is_left && mt.nch > 1) {
        const T* Lp = ws + mt.Lp;
        for (int e = threadIdx.x; e < d * d; e += blockDim.x) {
            T acc = Lp[e];
            for (int c = 1; c < mt.nch; ++c) acc += Lp[(long)c * d * d + e];
            L[e] = acc;
        }
        __syncthreads();
    }
    if (d <= TS) {
        smem_qtlq_jacobi<T>(L, Q, d, maxsw, tol, S0, S1);
    } else {
        // General d > 64: B = Q^T (L Q) via bmm_any using W1/W2 as scratch
        T* tmp = is_left ? (ws + mt.W1) : (ws + mt.W2);
        bmm_any<T>(tmp, d, L, d, 1, Q, d, 1, d, d, d, T(1), T(0), S0, S1);
        bmm_any<T>(B, d, Q, 1, d, tmp, d, 1, d, d, d, T(1), T(0), S0, S1);
        if constexpr (std::is_same<T, float>::value) {
            if (d <= 128) {
                T* Qt = is_left ? (ws + mt.W3) : (ws + mt.W4);
                jacobi_smem_128<T>(B, Q, Qt, d, maxsw, tol, S0);
                return;
            }
        }
        jacobi_global<T>(B, Q, d, maxsw, tol);
    }
}

template <typename T>
__global__ void __launch_bounds__(256) k_soap_ap(
    const SoapMat* __restrict__ mats, const int2* __restrict__ units,
    const PDesc<T>* __restrict__ d, const T* __restrict__ sc, T* __restrict__ ws,
    double lr_mult, double b2_d, double eps_d
) {
    extern __shared__ unsigned char smem_raw[];
    T* S0 = reinterpret_cast<T*>(smem_raw);
    T* S1 = S0 + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<T> q = d[mt.param];
    if (mt.use_smem) {
        const int n = mt.n;
        const int c0 = (mt.nch > 1) ? (u.y * TS) : 0;
        const int nc = (mt.nch > 1) ? min(TS, n - c0) : n;
        soap_apply_smem<T>(mt, q, ws, c0, nc, sc, lr_mult, b2_d, eps_d, S0, S1, false);
    } else {
        soap_apply_general<T>(mt, q, ws, sc, lr_mult, b2_d, eps_d, S0, S1);
    }
}

// =============================================================================
// 5. Regularizer Gradient Kernels (exact match to FastLRSKStudent training loop!)
// =============================================================================
template <typename T>
__global__ void k_add_reg_kpow(T* __restrict__ g, const T* __restrict__ p, int rows, int K, int pow_k, T coeff) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = rows * K;
    if (idx >= tot) return;
    const int k = idx % K;
    T kw = T(1);
    for (int t = 0; t < pow_k; ++t) kw *= (T)k;
    g[idx] += coeff * kw * p[idx];
}

template <typename T>
__global__ void k_add_reg_l2(T* __restrict__ g, const T* __restrict__ p, long n, T coeff) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) g[idx] += coeff * p[idx];
}

template <typename T>
__global__ void k_init_eye(T* __restrict__ Q, int d) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d * d) Q[idx] = ((idx / d) == (idx % d)) ? T(1) : T(0);
}

} // namespace lrsk_opt
