// Loup-S training library: one optimizer step of the Loup-S student per CUDA-graph replay.
// Specification: tools/dev/strain_reference.py (stage names below follow it). Layouts: node tensors of the
// 16 components [16, N, C] (component-major), node features [N, C], pair tensors [P, 4, C], edges [E, 16].
#include "loupstrain.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "generated_train.cuh"
#include "poly_dedup.cuh"
#include "sh16_pair.cuh"
#include "mlp_fused.cuh"
#include "mlp_bwd2.cuh"
#include "mlp_bwdp.cuh"
#include "seg_gemm.cuh"
#include "multi_gemm.cuh"
#include "node_ops.cuh"
#include "mlp_spline.cuh"
#include "edge_h2.cuh"
#include "mlp_tc.cuh"
#include "gemm_tn.cuh"

#include <cstdlib>
#include <type_traits>

namespace lst {

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

#define CK(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) throw Error(std::string(#x) + ": " + cudaGetErrorString(e_));               \
    } while (0)
#define CB(x)                                                                                              \
    do {                                                                                                   \
        cublasStatus_t s_ = (x);                                                                           \
        if (s_ != CUBLAS_STATUS_SUCCESS) throw Error(std::string(#x) + ": cublas status " + std::to_string((int)s_)); \
    } while (0)
#define KCHECK() CK(cudaGetLastError())

constexpr int TPB = 256;
inline int nblk(long n, int t = TPB) { return (int)((n + t - 1) / t); }

__device__ __forceinline__ int lofm(int m) { return (m >= 1) + (m >= 4) + (m >= 9); }

template <typename T>
__device__ __forceinline__ T sigm(T z) { return T(1) / (T(1) + exp(-z)); }

// 16 consecutive values (64 / 128 B aligned rows) with vector loads
__device__ __forceinline__ void ld16(const float* __restrict__ p, float* y) {
    const float4* q = reinterpret_cast<const float4*>(p);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float4 v = q[k];
        y[4 * k] = v.x; y[4 * k + 1] = v.y; y[4 * k + 2] = v.z; y[4 * k + 3] = v.w;
    }
}
__device__ __forceinline__ void ld16(const double* __restrict__ p, double* y) {
    const double2* q = reinterpret_cast<const double2*>(p);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        double2 v = q[k];
        y[2 * k] = v.x; y[2 * k + 1] = v.y;
    }
}

__device__ __forceinline__ void st16(float* __restrict__ p, const float* y) {
    float4* q = reinterpret_cast<float4*>(p);
#pragma unroll
    for (int k = 0; k < 4; ++k) q[k] = make_float4(y[4 * k], y[4 * k + 1], y[4 * k + 2], y[4 * k + 3]);
}
__device__ __forceinline__ void st16(double* __restrict__ p, const double* y) {
    double2* q = reinterpret_cast<double2*>(p);
#pragma unroll
    for (int k = 0; k < 8; ++k) q[k] = make_double2(y[2 * k], y[2 * k + 1]);
}

// ============================================================================ GEMM (row-major helper)
// FP32 GEMMs' cuBLAS compute type: CUBLAS_COMPUTE_32F, or with LST_MATH=tf32 / bf16 (read at engine creation)
// the TF32 / bf16 tensor-core types (operands rounded inside cuBLAS, FP32 accumulation and storage).
inline cublasComputeType_t g_f32_compute = CUBLAS_COMPUTE_32F;
inline cublasStatus_t xgemm(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, int m, int n, int k,
                            const float* al, const float* A, int lda, const float* B, int ldb, const float* be,
                            float* C, int ldc) {
    if (g_f32_compute == CUBLAS_COMPUTE_32F) return cublasSgemm(h, a, b, m, n, k, al, A, lda, B, ldb, be, C, ldc);
    return cublasGemmEx(h, a, b, m, n, k, al, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, be, C, CUDA_R_32F, ldc,
                        g_f32_compute, CUBLAS_GEMM_DEFAULT);
}
inline cublasStatus_t xgemm(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, int m, int n, int k,
                            const double* al, const double* A, int lda, const double* B, int ldb, const double* be,
                            double* C, int ldc) {
    return cublasDgemm(h, a, b, m, n, k, al, A, lda, B, ldb, be, C, ldc);
}

// C[M, N] = alpha op(A) op(B) + beta C, all row-major; op(A) is M x K (A stored K x M when ta).
template <typename T>
void gemm(cublasHandle_t h, bool ta, bool tb, int M, int N, int K, T alpha, const T* A, int lda, const T* B,
          int ldb, T beta, T* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    CB(xgemm(h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb, A, lda,
             &beta, C, ldc));
}

// ============================================================================ batch gather
struct Topo {
    const int32_t *node_off, *edge_off, *pair_off, *snd, *rcv, *rptr, *pidx, *pe0, *pe1;
    int nf, D;
    const int32_t* norder = nullptr;   // [nodes] frame-local processing order of the force pass (lst_set_node_order)
    const int32_t* eps = nullptr;      // [E_tot] frame-local (pair_idx << 1) | side for k_conv_pair_sp emeta (LST_EDGE_KSORT)
    const int32_t* ek = nullptr;       // [E_tot] precomputed spline knot index per edge
    const float* eu = nullptr;         // [E_tot] precomputed spline u per edge (FP32)
    const int32_t* ed_eps = nullptr;   // [E_tot] unified (n, kidx)-sorted directed (pair_idx << 1) | side (Wave 24)
    const int32_t* ed_snd = nullptr;   // [E_tot] unified (n, kidx)-sorted directed sender (Wave 24)
    const int32_t* ed_ek = nullptr;    // [E_tot] unified (n, kidx)-sorted directed kidx (Wave 24)
    const float* ed_eu = nullptr;      // [E_tot] unified (n, kidx)-sorted directed u (Wave 24)
    const int2* puv = nullptr;         // [P_tot] frame-local {rcv[pe0], snd[pe0]} per pair
    const int32_t* rptr0 = nullptr;    // [nodes] frame-local end of side==0 & kidx>=0 edges per node (LST_EDGE_KSORT)
    const int4* emeta_pre = nullptr;   // [E_tot] Wave 43: pre-packed {eps, snd, ek, float_as_int(eu)} for 1-load gather
    const float4* p_vec4 = nullptr;    // [P_tot+1] Wave 43: pre-packed canonical pair {vx, vy, vz, 0} (+ dummy at tot_p)
    const int2* p_sr = nullptr;        // [P_tot] Wave 43: pre-packed canonical pair {snd[pe0], rcv[pe0]}
    int dummy_kidx = -1;
    float dummy_u = 0.f;
};

template <typename T>
struct FramesT {
    const T* vec;
    const int64_t* species;
    const T *f_ref, *core_f, *e_ref, *core_e, *volume, *n_atoms, *w_e, *w_f, *w_s, *s_ref, *core_s;
    const T* ei_ref = nullptr;  // [nodes] optional per-atom target MLIP local energy (Wave 30)
};

template <typename T>
struct BatchT {
    T* vec;
    int *snd, *rcv, *pidx, *pe0, *pe1, *ngraph, *egraph, *rptr, *rptr0, *spc;
    int* ps;     // [E] (pair_idx << 1) | (e == pe1[pair_idx]) for pair-parity kernels (LST_PAIR_BASIS=1)
    int4* emeta; // [E] packed {ps_e, snd_e, kidx, float_as_int(u)} for 1-load k_conv_pair_sp (LST_EDGE_KSORT=1)
    int4* emeta_dir; // [E] unified (n, kidx)-sorted directed {ps_e, snd_e, kidx, float_as_int(u)} (Wave 24)
    float4* p_vec4 = nullptr; // [P] Wave 43: batch canonical pair {vx, vy, vz, 0}
    int4* p_srg = nullptr;    // [P] Wave 43: batch canonical pair {snd, rcv, egraph, 0}
    int2* puv;   // [P] packed {rcv[pe0[p]], snd[pe0[p]]} in batch node ids (LST_SLOT_KB / LST_FORCE_KB)
    int4* f_meta;// [B] per-batch-slot {fid, n0, p0, e0} (fid = -1 for padding slots)
    int *spc_off, *spc_idx, *spc_tiles, *spc_tiles32, *spc_wsk;  // species-grouped atom indices & tile/split-K tables
    int* pptr;   // [N + 1] each node's own pairs [pptr[n], pptr[n + 1]) (k_pair_ptr; LST_SLOT_PP)
    int* norder; // [N] the force pass's node of each warp slot (the frames' order; identity for dummy nodes)
    int* sp_idx; // [P] cubic-Hermite knot interval index (-1 if r >= r_max) for LST_MLP_SPLINE
    T* sp_u;     // [P] normalized knot coordinate u in [0, 1] for LST_MLP_SPLINE
    T *onehot, *f_ref, *core_f, *w_f, *e_ref, *core_e, *volume, *n_atoms, *w_e, *w_s, *s_ref, *core_s, *counts;
    T* ei_ref = nullptr;  // [N] per-atom target local energy for batch (Wave 30)
};

// The padded batch of step *step from its frames (fids[step * B ...], -1 after the last): the frames' nodes,
// edges and pairs in order, then dummy nodes, slack self-loops (D per dummy node, paired consecutively) and
// the dummy graph slot B; graph slots past the batch's frames read the dummy frame (zero weights).
template <typename T, bool SLIM = false>
__global__ void k_gather(const int* step, const int* __restrict__ fids, int B, Topo tp, FramesT<T> fr, BatchT<T> b,
                         int N, int E, int P, int G1, int S, T* F_init = nullptr) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int* f = fids + (long)*step * B;
    // locate i among the frames' nodes, edges and pairs
    int n0 = 0, e0 = 0, p0 = 0, G = 0;
    int eg = -1, ef = 0, ee0 = 0, en0 = 0, ep0 = 0;
    int ng = -1, nfm = 0, nn0 = 0, nE0 = 0;
    int pg = -1, pf = 0, pp0 = 0, pE0 = 0, pN0 = 0;
    for (int k = 0; k < B; ++k) {
        const int fid = f[k];
        if (fid < 0) break;
        const int nn = tp.node_off[fid + 1] - tp.node_off[fid];
        const int ne = tp.edge_off[fid + 1] - tp.edge_off[fid];
        const int np = tp.pair_off[fid + 1] - tp.pair_off[fid];
        if (i >= e0 && i < e0 + ne) { eg = k; ef = fid; ee0 = e0; en0 = n0; ep0 = p0; }
        if (i >= n0 && i < n0 + nn) { ng = k; nfm = fid; nn0 = n0; nE0 = e0; }
        if (i >= p0 && i < p0 + np) { pg = k; pf = fid; pp0 = p0; pE0 = e0; pN0 = n0; }
        n0 += nn;
        e0 += ne;
        p0 += np;
        ++G;
    }
    const int n_real = n0, e_real = e0, p_real = p0;
    if (i < E) {
        long r;
        int snd_i;
        if (eg >= 0) {
            const long j = tp.edge_off[ef] + (i - ee0);
            r = j;
            snd_i = tp.snd[j] + en0;
            b.snd[i] = snd_i;
            b.rcv[i] = tp.rcv[j] + en0;
            b.egraph[i] = eg;
            if constexpr (!SLIM) b.pidx[i] = tp.pidx[j] + ep0;
            if (tp.eps && b.emeta)
                b.emeta[i] = make_int4(tp.eps[j] + (ep0 << 1), snd_i, tp.ek[j], __float_as_int(tp.eu[j]));
            if (tp.ed_eps && b.emeta_dir)
                b.emeta_dir[i] = make_int4(tp.ed_eps[j] + (ep0 << 1), tp.ed_snd[j] + en0, tp.ed_ek[j],
                                           __float_as_int(tp.ed_eu[j]));
        } else {
            const int k = (int)i - e_real;
            const int dn = n_real + k / tp.D;
            r = tp.edge_off[tp.nf];
            snd_i = dn;
            b.snd[i] = dn;
            b.rcv[i] = dn;
            b.egraph[i] = B;
            if constexpr (!SLIM) b.pidx[i] = p_real + k / 2;
            if (tp.eps && b.emeta)
                b.emeta[i] = make_int4(((p_real + (k >> 1)) << 1) | (k & 1), dn, tp.dummy_kidx,
                                       __float_as_int(tp.dummy_u));
            if (tp.ed_eps && b.emeta_dir)
                b.emeta_dir[i] = make_int4(((p_real + (k >> 1)) << 1) | (k & 1), dn, tp.dummy_kidx,
                                           __float_as_int(tp.dummy_u));
        }
        for (int k = 0; k < 3; ++k) b.vec[3 * i + k] = fr.vec[3 * r + k];
    }
    if (i < N) {
        long nr;
        int g, frame;
        if (ng >= 0) {
            nr = tp.node_off[nfm] + (i - nn0);
            g = ng;
            frame = nfm;
            b.rptr[i] = tp.rptr[tp.node_off[nfm] + nfm + (i - nn0)] + nE0;
            if (b.rptr0)
                b.rptr0[i] = tp.rptr0 ? (tp.rptr0[tp.node_off[nfm] + (i - nn0)] + nE0)
                                      : (tp.rptr[tp.node_off[nfm] + nfm + (i - nn0) + 1] + nE0);
        } else {
            nr = tp.node_off[tp.nf];
            g = B;
            frame = tp.nf;
            const long k = i - n_real;
            const long ed = E - e_real;
            b.rptr[i] = e_real + (int)(k * tp.D < ed ? k * tp.D : ed);
            if (b.rptr0)
                b.rptr0[i] = (tp.rptr0 && tp.dummy_kidx < 0)
                                 ? b.rptr[i]
                                 : (e_real + (int)((k + 1) * tp.D < ed ? (k + 1) * tp.D : ed));
        }
        int sp = (int)fr.species[nr];
        b.spc[i] = sp;
        if constexpr (!SLIM) {
            for (int k = 0; k < S; ++k) b.onehot[i * S + k] = T(k == sp ? 1 : 0);
        }
        for (int k = 0; k < 3; ++k) {
            b.f_ref[3 * i + k] = fr.f_ref[3 * nr + k];
            const T cf = fr.core_f[3 * nr + k];
            b.core_f[3 * i + k] = cf;
            if (F_init) F_init[3 * i + k] = cf;
        }
        b.ngraph[i] = g;
        b.w_f[i] = fr.w_f[frame];
        if (fr.ei_ref && b.ei_ref) b.ei_ref[i] = (ng >= 0) ? fr.ei_ref[nr] : T(0);
        b.norder[i] = (ng >= 0 && tp.norder) ? tp.norder[tp.node_off[nfm] + (i - nn0)] + nn0 : (int)i;
    }
    if (i == N) b.rptr[N] = E;
    if (i < P) {
        int e0_, e1_;
        if (pg >= 0) {
            const long j = tp.pair_off[pf] + (i - pp0);
            e0_ = tp.pe0[j] + pE0;
            if constexpr (!SLIM) e1_ = tp.pe1[j] + pE0;
        } else {
            const int k = (int)i - p_real;
            e0_ = e_real + 2 * k;
            if constexpr (!SLIM) e1_ = e_real + 2 * k + 1;
        }
        b.pe0[i] = e0_;
        if constexpr (!SLIM) {
            b.pe1[i] = e1_;
            b.ps[e0_] = (int)(i << 1);
            b.ps[e1_] = (int)((i << 1) | 1);
            if (tp.puv && b.puv) {
                if (pg >= 0) {
                    const long j = tp.pair_off[pf] + (i - pp0);
                    const int2 uv = tp.puv[j];
                    b.puv[i] = make_int2(uv.x + pN0, uv.y + pN0);
                } else {
                    const int k = (int)i - p_real;
                    const int dn = n_real + (2 * k) / tp.D;
                    b.puv[i] = make_int2(dn, dn);
                }
            }
        }
    }
    if (i < B && b.f_meta) {
        int fn0 = 0, fp0 = 0, fe0 = 0, ffid = -1;
        for (int k = 0; k <= (int)i; ++k) {
            const int fk = f[k];
            if (fk < 0) break;
            if (k == (int)i) { ffid = fk; break; }
            fn0 += tp.node_off[fk + 1] - tp.node_off[fk];
            fe0 += tp.edge_off[fk + 1] - tp.edge_off[fk];
            fp0 += tp.pair_off[fk + 1] - tp.pair_off[fk];
        }
        b.f_meta[i] = make_int4(ffid, fn0, fp0, fe0);
    }
    if (i < G1) {
        const long f_ = (i < B && i < G) ? f[i] : tp.nf;
        b.e_ref[i] = fr.e_ref[f_];
        b.core_e[i] = fr.core_e[f_];
        b.volume[i] = fr.volume[f_];
        b.n_atoms[i] = fr.n_atoms[f_];
        b.w_e[i] = fr.w_e[f_];
        b.w_s[i] = fr.w_s[f_];
        for (int k = 0; k < 9; ++k) {
            b.s_ref[9 * i + k] = fr.s_ref[9 * f_ + k];
            b.core_s[9 * i + k] = fr.core_s[9 * f_ + k];
        }
    }
    if (i == 0) {
        b.counts[0] = T(n_real);
        b.counts[1] = T(G);
    }
}

// pptr[n] = the first pair whose owner (the receiver of its canonical edge pe0) is >= n: owners are
// non-decreasing in p (frames, then slack pairs; edges receiver-sorted, canonical edges ascending)
__global__ void k_pair_ptr(const int* __restrict__ pe0, const int* __restrict__ rcv, int* __restrict__ pptr, int P,
                           int N) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p > P) return;
    const int o = p < P ? rcv[pe0[p]] : N;
    const int op = p > 0 ? rcv[pe0[p - 1]] : -1;
    for (int n = op + 1; n <= o; ++n) pptr[n] = p;
}

// ============================================================================ edge basis
template <typename T>
__global__ void k_basis(const T* __restrict__ V, T* Y, T* JY, T* EF, T* DEF, T* U, const T* __restrict__ BW, T pref,
                        T rmax, int p, int NB, int E) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    T vx = V[3 * e], vy = V[3 * e + 1], vz = V[3 * e + 2];
    T r = sqrt(vx * vx + vy * vy + vz * vz);
    T rinv = T(1) / r;
    T ux = vx * rinv, uy = vy * rinv, uz = vz * rinv;
    U[3 * e] = ux;
    U[3 * e + 1] = uy;
    U[3 * e + 2] = uz;
    {
        T y[16], jy[48];
        sh16<T>(ux, uy, uz, rinv, y, jy);
        st16(Y + 16L * e, y);
        st16(JY + 48L * e, jy);
        st16(JY + 48L * e + 16, jy + 16);
        st16(JY + 48L * e + 32, jy + 32);
    }
    T t = r / rmax;
    T inside = r < rmax ? T(1) : T(0);
    T P = T(p);
    T a = (P + 1) * (P + 2) / 2, b = P * (P + 2), c = P * (P + 1) / 2;
    T tp1 = t;
    for (int k = 0; k < p - 2; ++k) tp1 *= t;  // t^(p-1)
    T tp = tp1 * t, tq = tp * t;
    T env = (T(1) - a * tp + b * tq - c * tq * t) * inside;
    T denv = (-a * P * tp1 + b * (P + 1) * tp - c * (P + 2) * tq) / rmax * inside;
    for (int n = 0; n < NB; ++n) {
        T w = BW[n];
        T s = sin(w * r), co = cos(w * r);
        T f = pref * s * rinv;
        T df = pref * (w * co * rinv - s * rinv * rinv);
        EF[(long)e * NB + n] = f * env;
        DEF[(long)e * NB + n] = df * env + f * denv;
    }
}

// Pair-indexed basis (LST_PAIR_BASIS=1): one thread per pair p < P, canonical edge e0 = pe0[p];
// writes UP[p] = (ux, uy, uz, rinv), YP[p, 0..15] = sh16_y(u), and either (EF[e0], DEF[e0]) or (sp_idx[p], sp_u[p]).
template <typename T, bool SKIP_BESSEL = false>
__global__ void k_basis_pair(const T* __restrict__ V, const int* __restrict__ pe0, T* __restrict__ YP,
                             T* __restrict__ EF, T* __restrict__ DEF, T* __restrict__ UP, const T* __restrict__ BW,
                             T pref, T rmax, int p_cut, int NB, int P_cnt, int* __restrict__ sp_idx = nullptr,
                             T* __restrict__ sp_u = nullptr, T inv_h = T(0), int sp_K = 0, T r_lo = T(0)) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P_cnt) return;
    const long e0 = pe0[p];
    const T vx = V[3L * e0], vy = V[3L * e0 + 1], vz = V[3L * e0 + 2];
    const T r = sqrt(vx * vx + vy * vy + vz * vz);
    const T rinv = T(1) / r;
    const T ux = vx * rinv, uy = vy * rinv, uz = vz * rinv;
    if constexpr (std::is_same<T, float>::value) {
        *reinterpret_cast<float4*>(UP + 4L * p) = make_float4(ux, uy, uz, rinv);
    } else {
        UP[4L * p] = ux;
        UP[4L * p + 1] = uy;
        UP[4L * p + 2] = uz;
        UP[4L * p + 3] = rinv;
    }
    {
        T y[16];
        sh16_y<T>(ux, uy, uz, y);
        st16(YP + 16L * p, y);
    }
    if constexpr (SKIP_BESSEL) {
        if (r >= rmax) {
            sp_idx[p] = -1;
            sp_u[p] = T(0);
        } else {
            const T pos = (r - r_lo) * inv_h;
            int idx = (int)pos;
            if (idx < 0) idx = 0;
            if (idx >= sp_K) idx = sp_K - 1;
            sp_idx[p] = idx;
            sp_u[p] = pos - T(idx);
        }
        return;
    }
    const T t = r / rmax;
    const T inside = r < rmax ? T(1) : T(0);
    const T Pc = T(p_cut);
    const T a = (Pc + 1) * (Pc + 2) / 2, b = Pc * (Pc + 2), c = Pc * (Pc + 1) / 2;
    T tp1 = t;
    for (int k = 0; k < p_cut - 2; ++k) tp1 *= t;
    const T tp = tp1 * t, tq = tp * t;
    const T env = (T(1) - a * tp + b * tq - c * tq * t) * inside;
    const T denv = (-a * Pc * tp1 + b * (Pc + 1) * tp - c * (Pc + 2) * tq) / rmax * inside;
    for (int n = 0; n < NB; ++n) {
        const T w = BW[n];
        const T s = sin(w * r), co = cos(w * r);
        const T f = pref * s * rinv;
        const T df = pref * (w * co * rinv - s * rinv * rinv);
        EF[e0 * NB + n] = f * env;
        DEF[e0 * NB + n] = df * env + f * denv;
    }
}

// Wave 43+44 (LST_GATHER_SLIM=2): Disjoint-block fused coalesced int4/float4/int2 gather + UP/YP spherical harmonics!
// Disjoint blockIdx.x regions (nb_p | nb_e | nb_n | 1 graph block) eliminate the Block-0 serial E+P+N+G1 bottleneck:
//   - [0 .. nb_p): canonical pairs (p_vec4, p_srg, register-resident UP/YP sh16_y)
//   - [nb_p .. nb_p+nb_e): pure streaming int4 emeta copy
//   - [nb_p+nb_e .. nb_p+nb_e+nb_n): node metadata + rptr[N]
//   - last block: parallel graph metadata (s_ref/core_s across 9*G1 threads) + counts
__global__ void k_gather_basis_fast(const int* step, const int* __restrict__ fids, int B, Topo tp, FramesT<float> fr,
                                    BatchT<float> b, int N, int E, int P, int G1, int S, float* F_init,
                                    float* __restrict__ YP, float* __restrict__ UP, int nb_p, int nb_e, int nb_n) {
    const int bx = (int)blockIdx.x;
    const int* f = fids + (long)*step * B;
    if (B == 1 && f[0] >= 0) {
        const int fid = f[0];
        if (bx < nb_p) {
            const long i = (long)bx * blockDim.x + threadIdx.x;
            if (i >= P) return;
            const int p0_f = tp.pair_off[fid], p_real = tp.pair_off[fid + 1] - p0_f;
            float4 v4;
            if (i < p_real) {
                v4 = tp.p_vec4[p0_f + i];
                const int2 sr = tp.p_sr[p0_f + i];
                b.p_srg[i] = make_int4(sr.x, sr.y, 0, 0);
            } else {
                const int n_real = tp.node_off[fid + 1] - tp.node_off[fid];
                v4 = tp.p_vec4[tp.pair_off[tp.nf]];
                const int dn = n_real + (2 * ((int)i - p_real)) / tp.D;
                b.p_srg[i] = make_int4(dn, dn, 1, 0);
            }
            b.p_vec4[i] = v4;
            const float vx = v4.x, vy = v4.y, vz = v4.z;
            const float r = sqrtf(vx * vx + vy * vy + vz * vz);
            const float rinv = 1.f / r;
            const float ux = vx * rinv, uy = vy * rinv, uz = vz * rinv;
            *reinterpret_cast<float4*>(UP + 4L * i) = make_float4(ux, uy, uz, rinv);
            float y[16];
            sh16_y<float>(ux, uy, uz, y);
            st16(YP + 16L * i, y);
            return;
        }
        if (bx < nb_p + nb_e) {
            const long i = (long)(bx - nb_p) * blockDim.x + threadIdx.x;
            if (i >= E) return;
            const int e0_f = tp.edge_off[fid], e_real = tp.edge_off[fid + 1] - e0_f;
            if (i < e_real) {
                b.emeta[i] = tp.emeta_pre[e0_f + i];
            } else {
                const int n_real = tp.node_off[fid + 1] - tp.node_off[fid];
                const int p_real = tp.pair_off[fid + 1] - tp.pair_off[fid];
                const int k = (int)i - e_real;
                const int dn = n_real + k / tp.D;
                b.emeta[i] = make_int4(((p_real + (k >> 1)) << 1) | (k & 1), dn, tp.dummy_kidx,
                                       __float_as_int(tp.dummy_u));
            }
            return;
        }
        if (bx < nb_p + nb_e + nb_n) {
            const long i = (long)(bx - nb_p - nb_e) * blockDim.x + threadIdx.x;
            if (i == N) { b.rptr[N] = E; return; }
            if (i > N) return;
            const int n0_f = tp.node_off[fid], n_real = tp.node_off[fid + 1] - n0_f;
            const bool real_n = (i < n_real);
            const long nr = real_n ? ((long)n0_f + i) : (long)tp.node_off[tp.nf];
            const int g = real_n ? 0 : 1;
            const int frame = real_n ? fid : tp.nf;
            if (real_n) {
                b.rptr[i] = tp.rptr[n0_f + fid + i];
                if (b.rptr0) b.rptr0[i] = tp.rptr0 ? tp.rptr0[n0_f + i] : tp.rptr[n0_f + fid + i + 1];
            } else {
                const int e_real = tp.edge_off[fid + 1] - tp.edge_off[fid];
                const long k = i - n_real;
                const long ed = E - e_real;
                b.rptr[i] = e_real + (int)(k * tp.D < ed ? k * tp.D : ed);
                if (b.rptr0)
                    b.rptr0[i] = (tp.rptr0 && tp.dummy_kidx < 0)
                                     ? b.rptr[i]
                                     : (e_real + (int)((k + 1) * tp.D < ed ? (k + 1) * tp.D : ed));
            }
            b.spc[i] = (int)fr.species[nr];
            for (int k = 0; k < 3; ++k) {
                b.f_ref[3 * i + k] = fr.f_ref[3 * nr + k];
                const float cf = fr.core_f[3 * nr + k];
                b.core_f[3 * i + k] = cf;
                if (F_init) F_init[3 * i + k] = cf;
            }
            b.ngraph[i] = g;
            b.w_f[i] = fr.w_f[frame];
            if (fr.ei_ref && b.ei_ref) b.ei_ref[i] = real_n ? fr.ei_ref[nr] : 0.f;
            b.norder[i] = (real_n && tp.norder) ? tp.norder[n0_f + i] : (int)i;
            return;
        }
        const int t = threadIdx.x;
        if (t < G1) {
            const long f_ = (t == 0) ? (long)fid : (long)tp.nf;
            b.e_ref[t] = fr.e_ref[f_];
            b.core_e[t] = fr.core_e[f_];
            b.volume[t] = fr.volume[f_];
            b.n_atoms[t] = fr.n_atoms[f_];
            b.w_e[t] = fr.w_e[f_];
            b.w_s[t] = fr.w_s[f_];
        }
        if (t < 9 * G1) {
            const int gi = t / 9, kk = t % 9;
            const long f_ = (gi == 0) ? (long)fid : (long)tp.nf;
            b.s_ref[t] = fr.s_ref[9 * f_ + kk];
            b.core_s[t] = fr.core_s[9 * f_ + kk];
        }
        if (t == 0) {
            b.counts[0] = float(tp.node_off[fid + 1] - tp.node_off[fid]);
            b.counts[1] = 1.f;
        }
        return;
    }

    if (bx < nb_p) {
        const long i = (long)bx * blockDim.x + threadIdx.x;
        if (i >= P) return;
        int n0 = 0, p0 = 0, pg = -1, pf = 0, pp0 = 0, pN0 = 0;
        for (int k = 0; k < B; ++k) {
            const int fid = f[k];
            if (fid < 0) break;
            const int nn = tp.node_off[fid + 1] - tp.node_off[fid];
            const int np = tp.pair_off[fid + 1] - tp.pair_off[fid];
            if (i >= p0 && i < p0 + np) { pg = k; pf = fid; pp0 = p0; pN0 = n0; }
            n0 += nn; p0 += np;
        }
        float4 v4;
        if (pg >= 0) {
            const long j = tp.pair_off[pf] + (i - pp0);
            v4 = tp.p_vec4[j];
            const int2 sr = tp.p_sr[j];
            b.p_srg[i] = make_int4(sr.x + pN0, sr.y + pN0, pg, 0);
        } else {
            v4 = tp.p_vec4[tp.pair_off[tp.nf]];
            const int dn = n0 + (2 * ((int)i - p0)) / tp.D;
            b.p_srg[i] = make_int4(dn, dn, B, 0);
        }
        b.p_vec4[i] = v4;
        const float vx = v4.x, vy = v4.y, vz = v4.z;
        const float r = sqrtf(vx * vx + vy * vy + vz * vz);
        const float rinv = 1.f / r;
        const float ux = vx * rinv, uy = vy * rinv, uz = vz * rinv;
        *reinterpret_cast<float4*>(UP + 4L * i) = make_float4(ux, uy, uz, rinv);
        float y[16];
        sh16_y<float>(ux, uy, uz, y);
        st16(YP + 16L * i, y);
        return;
    }
    if (bx < nb_p + nb_e) {
        const long i = (long)(bx - nb_p) * blockDim.x + threadIdx.x;
        if (i >= E) return;
        int n0 = 0, e0 = 0, p0 = 0, eg = -1, ef = 0, ee0 = 0, en0 = 0, ep0 = 0;
        for (int k = 0; k < B; ++k) {
            const int fid = f[k];
            if (fid < 0) break;
            const int nn = tp.node_off[fid + 1] - tp.node_off[fid];
            const int ne = tp.edge_off[fid + 1] - tp.edge_off[fid];
            const int np = tp.pair_off[fid + 1] - tp.pair_off[fid];
            if (i >= e0 && i < e0 + ne) { eg = k; ef = fid; ee0 = e0; en0 = n0; ep0 = p0; }
            n0 += nn; e0 += ne; p0 += np;
        }
        if (eg >= 0) {
            const long j = tp.edge_off[ef] + (i - ee0);
            int4 em = tp.emeta_pre[j];
            em.x += (ep0 << 1);
            em.y += en0;
            b.emeta[i] = em;
        } else {
            const int k = (int)i - e0;
            const int dn = n0 + k / tp.D;
            b.emeta[i] = make_int4(((p0 + (k >> 1)) << 1) | (k & 1), dn, tp.dummy_kidx,
                                   __float_as_int(tp.dummy_u));
        }
        return;
    }
    if (bx < nb_p + nb_e + nb_n) {
        const long i = (long)(bx - nb_p - nb_e) * blockDim.x + threadIdx.x;
        if (i == N) { b.rptr[N] = E; return; }
        if (i > N) return;
        int n0 = 0, e0 = 0, ng = -1, nfm = 0, nn0 = 0, nE0 = 0;
        for (int k = 0; k < B; ++k) {
            const int fid = f[k];
            if (fid < 0) break;
            const int nn = tp.node_off[fid + 1] - tp.node_off[fid];
            const int ne = tp.edge_off[fid + 1] - tp.edge_off[fid];
            if (i >= n0 && i < n0 + nn) { ng = k; nfm = fid; nn0 = n0; nE0 = e0; }
            n0 += nn; e0 += ne;
        }
        long nr;
        int g, frame;
        if (ng >= 0) {
            nr = tp.node_off[nfm] + (i - nn0);
            g = ng;
            frame = nfm;
            b.rptr[i] = tp.rptr[tp.node_off[nfm] + nfm + (i - nn0)] + nE0;
            if (b.rptr0)
                b.rptr0[i] = tp.rptr0 ? (tp.rptr0[tp.node_off[nfm] + (i - nn0)] + nE0)
                                      : (tp.rptr[tp.node_off[nfm] + nfm + (i - nn0) + 1] + nE0);
        } else {
            nr = tp.node_off[tp.nf];
            g = B;
            frame = tp.nf;
            const long k = i - n0;
            const long ed = E - e0;
            b.rptr[i] = e0 + (int)(k * tp.D < ed ? k * tp.D : ed);
            if (b.rptr0)
                b.rptr0[i] = (tp.rptr0 && tp.dummy_kidx < 0)
                                 ? b.rptr[i]
                                 : (e0 + (int)((k + 1) * tp.D < ed ? (k + 1) * tp.D : ed));
        }
        b.spc[i] = (int)fr.species[nr];
        for (int k = 0; k < 3; ++k) {
            b.f_ref[3 * i + k] = fr.f_ref[3 * nr + k];
            const float cf = fr.core_f[3 * nr + k];
            b.core_f[3 * i + k] = cf;
            if (F_init) F_init[3 * i + k] = cf;
        }
        b.ngraph[i] = g;
        b.w_f[i] = fr.w_f[frame];
        if (fr.ei_ref && b.ei_ref) b.ei_ref[i] = (ng >= 0) ? fr.ei_ref[nr] : 0.f;
        b.norder[i] = (ng >= 0 && tp.norder) ? tp.norder[tp.node_off[nfm] + (i - nn0)] + nn0 : (int)i;
        return;
    }
    const int t = threadIdx.x;
    int n_real = 0, G = 0;
    for (int k = 0; k < B; ++k) {
        const int fid = f[k];
        if (fid < 0) break;
        n_real += tp.node_off[fid + 1] - tp.node_off[fid];
        ++G;
    }
    if (t < G1) {
        const long f_ = (t < B && t < G) ? f[t] : tp.nf;
        b.e_ref[t] = fr.e_ref[f_];
        b.core_e[t] = fr.core_e[f_];
        b.volume[t] = fr.volume[f_];
        b.n_atoms[t] = fr.n_atoms[f_];
        b.w_e[t] = fr.w_e[f_];
        b.w_s[t] = fr.w_s[f_];
    }
    if (t < 9 * G1) {
        const int gi = t / 9, kk = t % 9;
        const long f_ = (gi < B && gi < G) ? f[gi] : tp.nf;
        b.s_ref[t] = fr.s_ref[9 * f_ + kk];
        b.core_s[t] = fr.core_s[9 * f_ + kk];
    }
    if (t == 0) {
        b.counts[0] = float(n_real);
        b.counts[1] = float(G);
    }
}

template <typename T>
__global__ void k_pair_feats(const int* pe0, const T* EF, const T* DEF, T* xe, int P, int NB) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)P * NB) return;
    long p = i / NB, j = i % NB;
    long e = pe0[p];
    xe[i] = EF[e * NB + j];
    xe[(long)P * NB + i] = DEF[e * NB + j];
}

// ============================================================================ radial MLP activations
// z, h [2P, H]: value rows [0, P), tangent rows [P, 2P). h = act silu(z), dh = act silu'(z) dz.
template <typename T>
__global__ void k_silu_t_fwd(const T* z, T* h, long PH, T act) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= PH) return;
    T v = z[i], dv = z[i + PH];
    T s = sigm(v);
    h[i] = act * v * s;
    h[i + PH] = act * s * (T(1) + v * (T(1) - s)) * dv;
}

template <typename T>
__global__ void k_silu_t_bwd(const T* z, const T* gh, T* gz, long PH, T act) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= PH) return;
    T v = z[i], dv = z[i + PH];
    T s = sigm(v);
    T d1 = s * (T(1) + v * (T(1) - s));
    T d2 = s * (T(1) - s) * (T(2) + v * (T(1) - T(2) * s));
    T g = gh[i], gd = gh[i + PH];
    gz[i] = act * (d1 * g + d2 * dv * gd);
    gz[i + PH] = act * d1 * gd;
}

// ============================================================================ convolution slots
// out[m, n, c] (+)= sum_{e -> n} sum_t R_t[p(e), l(m), c] Y_t[e, m] X[s(e), c]
template <typename T, int NT, bool ACC>
__global__ void k_conv_g(T* __restrict__ out, const int* __restrict__ rptr, const int* __restrict__ snd,
                         const int* __restrict__ pidx, const T* __restrict__ X, const T* __restrict__ R0,
                         const T* __restrict__ Y0, const T* __restrict__ R1, const T* __restrict__ Y1, int N, int C) {
    int n = blockIdx.x * (blockDim.x / C) + threadIdx.x / C;
    int c = threadIdx.x % C;
    if (n >= N) return;
    T acc[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) acc[m] = T(0);
    int e1 = rptr[n + 1];
    for (int e = rptr[n]; e < e1; ++e) {
        long p = pidx[e];
        T x = X[(long)snd[e] * C + c];
        const T* r = R0 + p * 4 * C + c;
        T y[16];
        ld16(Y0 + 16L * e, y);
        T r0 = r[0] * x, r1 = r[C] * x, r2 = r[2 * C] * x, r3 = r[3 * C] * x;
        acc[0] += r0 * y[0];
#pragma unroll
        for (int m = 1; m < 4; ++m) acc[m] += r1 * y[m];
#pragma unroll
        for (int m = 4; m < 9; ++m) acc[m] += r2 * y[m];
#pragma unroll
        for (int m = 9; m < 16; ++m) acc[m] += r3 * y[m];
        if (NT == 2) {
            const T* q = R1 + p * 4 * C + c;
            T w[16];
            ld16(Y1 + 16L * e, w);
            T q0 = q[0] * x, q1 = q[C] * x, q2 = q[2 * C] * x, q3 = q[3 * C] * x;
            acc[0] += q0 * w[0];
#pragma unroll
            for (int m = 1; m < 4; ++m) acc[m] += q1 * w[m];
#pragma unroll
            for (int m = 4; m < 9; ++m) acc[m] += q2 * w[m];
#pragma unroll
            for (int m = 9; m < 16; ++m) acc[m] += q3 * w[m];
        }
    }
    long NC = (long)N * C;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        long idx = m * NC + (long)n * C + c;
        out[idx] = ACC ? out[idx] + acc[m] : acc[m];
    }
}

// out[j, c] (+)= sum_{e: s(e) = j} sum_m G[m, r(e), c] sum_t R_t[p(e), l(m), c] Y_t[e, m]
template <typename T, int NT, bool ACC>
__global__ void k_conv_x(T* __restrict__ out, const int* __restrict__ sptr, const int* __restrict__ sperm,
                         const int* __restrict__ rcv, const int* __restrict__ pidx, const T* __restrict__ G,
                         const T* __restrict__ R0, const T* __restrict__ Y0, const T* __restrict__ R1,
                         const T* __restrict__ Y1, int N, int C) {
    int j = blockIdx.x * (blockDim.x / C) + threadIdx.x / C;
    int c = threadIdx.x % C;
    if (j >= N) return;
    long NC = (long)N * C;
    T acc = T(0);
    int k1 = sptr[j + 1];
    for (int k = sptr[j]; k < k1; ++k) {
        int e = sperm[k];
        long p = pidx[e];
        const T* g = G + (long)rcv[e] * C + c;
        T gm[16];
#pragma unroll
        for (int m = 0; m < 16; ++m) gm[m] = g[m * NC];
        const T* r = R0 + p * 4 * C + c;
        T y[16];
        ld16(Y0 + 16L * e, y);
        T s0 = gm[0] * y[0], s1 = T(0), s2 = T(0), s3 = T(0);
#pragma unroll
        for (int m = 1; m < 4; ++m) s1 += gm[m] * y[m];
#pragma unroll
        for (int m = 4; m < 9; ++m) s2 += gm[m] * y[m];
#pragma unroll
        for (int m = 9; m < 16; ++m) s3 += gm[m] * y[m];
        acc += r[0] * s0 + r[C] * s1 + r[2 * C] * s2 + r[3 * C] * s3;
        if (NT == 2) {
            const T* q = R1 + p * 4 * C + c;
            T w[16];
            ld16(Y1 + 16L * e, w);
            T t0 = gm[0] * w[0], t1 = T(0), t2 = T(0), t3 = T(0);
#pragma unroll
            for (int m = 1; m < 4; ++m) t1 += gm[m] * w[m];
#pragma unroll
            for (int m = 4; m < 9; ++m) t2 += gm[m] * w[m];
#pragma unroll
            for (int m = 9; m < 16; ++m) t3 += gm[m] * w[m];
            acc += q[0] * t0 + q[C] * t1 + q[2 * C] * t2 + q[3 * C] * t3;
        }
    }
    long idx = (long)j * C + c;
    out[idx] = ACC ? out[idx] + acc : acc;
}

// outk[p, l, c] (+)= sum_{e in p} sum_{m in l} G[m, r(e), c] Yk[e, m] X[s(e), c]   (k < NY)
template <typename T, int NY, bool ACC>
__global__ void k_slot_r(T* __restrict__ out0, T* __restrict__ out1, const int* __restrict__ pe0,
                         const int* __restrict__ pe1, const int* __restrict__ rcv, const int* __restrict__ snd,
                         const T* __restrict__ G, const T* __restrict__ X, const T* __restrict__ Y0,
                         const T* __restrict__ Y1, int P, int N, int C) {
    int p = blockIdx.x * (blockDim.x / C) + threadIdx.x / C;
    int c = threadIdx.x % C;
    if (p >= P) return;
    long NC = (long)N * C;
    T a0[4] = {T(0), T(0), T(0), T(0)}, a1[4] = {T(0), T(0), T(0), T(0)};
#pragma unroll
    for (int side = 0; side < 2; ++side) {
        int e = side ? pe1[p] : pe0[p];
        T x = X[(long)snd[e] * C + c];
        const T* g = G + (long)rcv[e] * C + c;
        T gm[16];
#pragma unroll
        for (int m = 0; m < 16; ++m) gm[m] = g[m * NC] * x;
        T y[16];
        ld16(Y0 + 16L * e, y);
        a0[0] += gm[0] * y[0];
#pragma unroll
        for (int m = 1; m < 4; ++m) a0[1] += gm[m] * y[m];
#pragma unroll
        for (int m = 4; m < 9; ++m) a0[2] += gm[m] * y[m];
#pragma unroll
        for (int m = 9; m < 16; ++m) a0[3] += gm[m] * y[m];
        if (NY == 2) {
            T w[16];
            ld16(Y1 + 16L * e, w);
            a1[0] += gm[0] * w[0];
#pragma unroll
            for (int m = 1; m < 4; ++m) a1[1] += gm[m] * w[m];
#pragma unroll
            for (int m = 4; m < 9; ++m) a1[2] += gm[m] * w[m];
#pragma unroll
            for (int m = 9; m < 16; ++m) a1[3] += gm[m] * w[m];
        }
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        long idx = (long)p * 4 * C + l * C + c;
        out0[idx] = ACC ? out0[idx] + a0[l] : a0[l];
        if (NY == 2) out1[idx] = ACC ? out1[idx] + a1[l] : a1[l];
    }
}

// Force pass, both layers: s[e] = sum_t sum_m Y[e,m] sum_c G_t[m,r,c] T_t[p,l(m),c] X_t[s,c];
// gy[e, m] = sum_t sum_c G_t[m,r,c] R_t[p,l(m),c] X_t[s,c]. One warp per edge.
template <typename T>
__global__ void k_force_edge(T* __restrict__ se, T* __restrict__ gy, const int* __restrict__ rcv,
                             const int* __restrict__ snd, const int* __restrict__ pidx, const T* __restrict__ Y,
                             const T* G0, const T* R0, const T* T0, const T* X0, const T* G1, const T* R1,
                             const T* T1, const T* X1, int E, int N, int C) {
    int e = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (e >= E) return;
    long NC = (long)N * C;
    long r = rcv[e], s = snd[e], p = pidx[e];
    T y[16];
    ld16(Y + 16L * e, y);
    T gp[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) gp[m] = T(0);
    T sp = T(0);
    for (int t = 0; t < 2; ++t) {
        const T* G = t ? G1 : G0;
        const T* R = t ? R1 : R0;
        const T* Tt = t ? T1 : T0;
        const T* X = t ? X1 : X0;
        for (int c = lane; c < C; c += 32) {
            T x = X[s * C + c];
            T rr[4], tt[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                rr[l] = R[p * 4 * C + l * C + c] * x;
                tt[l] = Tt[p * 4 * C + l * C + c] * x;
            }
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                T g = G[m * NC + r * C + c];
                int l = lofm(m);
                gp[m] += g * rr[l];
                sp += y[m] * g * tt[l];
            }
        }
    }
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        T v = gp[m];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        gp[m] = v;
    }
    for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
    if (lane < 16) gy[16L * e + lane] = gp[lane];  // every lane holds the sums
    if (lane == 0) se[e] = sp;
}

// ============================================================================ convolution passes
// Receiver-ordered (thread (n, c), edges e -> n, partner e' = the reverse edge of the same pair p).
// Terms t < NT: radial weights R_t [P, 4, C] with edge factors Y_t [E, 16]; XB adds the term (R_1, Y_2, XB).
//   DG: OG[m, n, c] (+)= sum_{e -> n} [sum_t R_t[p, l(m), c] Y_t[e, m] X[s(e), c] + R_1 Y_2[e, m] XB[s(e), c]]
//   DX: OX[n, c]    (+)= sum_{e -> n} sum_t sum_m G[m, s(e), c] R_t[p, l(m), c] Y_t[e', m]
//       (the edges with sender n are the partners of those with receiver n)
// ES > 1: ES thread groups per node take alternate edges; group 0 adds the others' sums (shared memory).
template <typename T, bool DG, bool DX, int NT, bool XB, bool ACC, int ES = 1>
__global__ void __launch_bounds__(256) k_conv(const int* __restrict__ rptr, const int* __restrict__ snd,
                                              const int* __restrict__ pidx, const int* __restrict__ pe0,
                                              const int* __restrict__ pe1, const T* __restrict__ X,
                                              const T* __restrict__ G, const T* __restrict__ R0,
                                              const T* __restrict__ Y0, const T* __restrict__ R1,
                                              const T* __restrict__ Y1, const T* __restrict__ Y2,
                                              const T* __restrict__ Xb, T* __restrict__ OG, T* __restrict__ OX,
                                              int N, int C) {
    const int n = blockIdx.x * (blockDim.x / (ES * C)) + threadIdx.x / (ES * C);
    const int c = threadIdx.x % C;
    const int grp = (threadIdx.x / C) % ES;
    if (ES == 1 && n >= N) return;
    const long NC = (long)N * C;
    T og[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) og[m] = T(0);
    T ox = T(0);
    const int e1 = n < N ? rptr[n + 1] : 0;
    for (int e = (n < N ? rptr[n] : 0) + grp; e < e1; e += ES) {
        const long s = snd[e];
        const long p = pidx[e];
        int ep = 0;
        if (DX) {
            const int a = pe0[p];
            ep = a == e ? pe1[p] : a;
        }
        T xs = T(0), xb = T(0);
        if (DG) {
            xs = X[s * C + c];
            if (XB) xb = Xb[s * C + c];
        }
        T gs[16];
        if (DX) {
#pragma unroll
            for (int m = 0; m < 16; ++m) gs[m] = G[m * NC + s * C + c];
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const T* R = t ? R1 : R0;
            const T* Yt = t ? Y1 : Y0;
            T r[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) r[l] = R[p * 4 * C + l * C + c];
            if (DG) {
                T y[16];
                ld16(Yt + 16L * e, y);
                const T r0 = r[0] * xs, r1 = r[1] * xs, r2 = r[2] * xs, r3 = r[3] * xs;
                og[0] += r0 * y[0];
#pragma unroll
                for (int m = 1; m < 4; ++m) og[m] += r1 * y[m];
#pragma unroll
                for (int m = 4; m < 9; ++m) og[m] += r2 * y[m];
#pragma unroll
                for (int m = 9; m < 16; ++m) og[m] += r3 * y[m];
                if (XB && t == NT - 1) {
                    ld16(Y2 + 16L * e, y);
                    const T b0 = r[0] * xb, b1 = r[1] * xb, b2 = r[2] * xb, b3 = r[3] * xb;
                    og[0] += b0 * y[0];
#pragma unroll
                    for (int m = 1; m < 4; ++m) og[m] += b1 * y[m];
#pragma unroll
                    for (int m = 4; m < 9; ++m) og[m] += b2 * y[m];
#pragma unroll
                    for (int m = 9; m < 16; ++m) og[m] += b3 * y[m];
                }
            }
            if (DX) {
                T yp[16];
                ld16(Yt + 16L * ep, yp);
                T q1 = gs[1] * yp[1] + gs[2] * yp[2] + gs[3] * yp[3], q2 = T(0), q3 = T(0);
#pragma unroll
                for (int m = 4; m < 9; ++m) q2 += gs[m] * yp[m];
#pragma unroll
                for (int m = 9; m < 16; ++m) q3 += gs[m] * yp[m];
                ox += r[0] * gs[0] * yp[0] + r[1] * q1 + r[2] * q2 + r[3] * q3;
            }
        }
    }
    if constexpr (ES > 1) {
        constexpr int NS = 256 / ES * (ES - 1);           // threads of groups 1..ES-1 in a 256-thread block
        __shared__ __align__(16) unsigned char sraw[17 * NS * sizeof(T)];
        T* sb = reinterpret_cast<T*>(sraw);
        const int node_in_blk = threadIdx.x / (ES * C);
        for (int q = 1; q < ES; ++q) {
            const int slot = (node_in_blk * C + c) * (ES - 1) + (q - 1);
            if (grp == q) {
#pragma unroll
                for (int m = 0; m < 16; ++m) sb[m * NS + slot] = og[m];
                sb[16 * NS + slot] = ox;
            }
        }
        __syncthreads();
        if (grp != 0 || n >= N) return;
        for (int q = 1; q < ES; ++q) {
            const int slot = (node_in_blk * C + c) * (ES - 1) + (q - 1);
#pragma unroll
            for (int m = 0; m < 16; ++m) og[m] += sb[m * NS + slot];
            ox += sb[16 * NS + slot];
        }
    }
    if (DG) {
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const long idx = m * NC + (long)n * C + c;
            OG[idx] = ACC ? OG[idx] + og[m] : og[m];
        }
    }
    if (DX) {
        const long idx = (long)n * C + c;
        OX[idx] = ACC ? OX[idx] + ox : ox;
    }
}

// Pair-parity k_conv (LST_PAIR_BASIS=1): reads Yt[p] once (no pe0/pe1/ep loads) and applies (-1)^l parity via sgn.
// NM=true: node-major [N, 16, C] layout for G and OG (LST_NODE_MAJOR=1).
template <typename T, bool DG, bool DX, int NT, bool XB, bool ACC, int ES = 1, bool NM = false>
__global__ void __launch_bounds__(256) k_conv_pair(const int* __restrict__ rptr, const int* __restrict__ snd,
                                                    const int* __restrict__ ps, const T* __restrict__ X,
                                                    const T* __restrict__ G, const T* __restrict__ R0,
                                                    const T* __restrict__ Y0, const T* __restrict__ R1,
                                                    const T* __restrict__ Y1, const T* __restrict__ Y2,
                                                    const T* __restrict__ Xb, T* __restrict__ OG, T* __restrict__ OX,
                                                    int N, int C, const int* __restrict__ norder = nullptr) {
    const int nn = blockIdx.x * (blockDim.x / (ES * C)) + threadIdx.x / (ES * C);
    const int c = threadIdx.x % C;
    const int grp = (threadIdx.x / C) % ES;
    if (ES == 1 && nn >= N) return;
    const int n = nn < N ? (norder ? norder[nn] : nn) : N;
    const long NC = (long)N * C;
    T og[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) og[m] = T(0);
    T ox = T(0);
    const int e1 = nn < N ? rptr[n + 1] : 0;
    for (int e = (nn < N ? rptr[n] : 0) + grp; e < e1; e += ES) {
        const long s = snd[e];
        const int ps_e = ps[e];
        const long p = ps_e >> 1;
        const T sgn = (ps_e & 1) ? T(-1) : T(1);
        T xs = T(0), xb = T(0);
        if (DG) {
            xs = X[s * C + c];
            if (XB) xb = Xb[s * C + c];
        }
        T gs[16];
        if (DX) {
#pragma unroll
            for (int m = 0; m < 16; ++m)
                gs[m] = NM ? G[(s * 16 + m) * C + c] : G[m * NC + s * C + c];
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const T* R = t ? R1 : R0;
            const T* Yt = t ? Y1 : Y0;
            T r[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) r[l] = R[p * 4 * C + l * C + c];
            T y[16];
            ld16(Yt + 16L * p, y);
            if (DG) {
                const T r0 = r[0] * xs, r1 = sgn * r[1] * xs, r2 = r[2] * xs, r3 = sgn * r[3] * xs;
                og[0] += r0 * y[0];
#pragma unroll
                for (int m = 1; m < 4; ++m) og[m] += r1 * y[m];
#pragma unroll
                for (int m = 4; m < 9; ++m) og[m] += r2 * y[m];
#pragma unroll
                for (int m = 9; m < 16; ++m) og[m] += r3 * y[m];
            }
            if (DX) {
                T q1 = gs[1] * y[1] + gs[2] * y[2] + gs[3] * y[3], q2 = T(0), q3 = T(0);
#pragma unroll
                for (int m = 4; m < 9; ++m) q2 += gs[m] * y[m];
#pragma unroll
                for (int m = 9; m < 16; ++m) q3 += gs[m] * y[m];
                ox += r[0] * gs[0] * y[0] - sgn * r[1] * q1 + r[2] * q2 - sgn * r[3] * q3;
            }
            if (DG && XB && t == NT - 1) {
                ld16(Y2 + 16L * p, y);
                const T b0 = r[0] * xb, b1 = sgn * r[1] * xb, b2 = r[2] * xb, b3 = sgn * r[3] * xb;
                og[0] += b0 * y[0];
#pragma unroll
                for (int m = 1; m < 4; ++m) og[m] += b1 * y[m];
#pragma unroll
                for (int m = 4; m < 9; ++m) og[m] += b2 * y[m];
#pragma unroll
                for (int m = 9; m < 16; ++m) og[m] += b3 * y[m];
            }
        }
    }
    if constexpr (ES > 1) {
        constexpr int NS = 256 / ES * (ES - 1);
        __shared__ __align__(16) unsigned char sraw[17 * NS * sizeof(T)];
        T* sb = reinterpret_cast<T*>(sraw);
        const int node_in_blk = threadIdx.x / (ES * C);
        for (int q = 1; q < ES; ++q) {
            const int slot = (node_in_blk * C + c) * (ES - 1) + (q - 1);
            if (grp == q) {
#pragma unroll
                for (int m = 0; m < 16; ++m) sb[m * NS + slot] = og[m];
                sb[16 * NS + slot] = ox;
            }
        }
        __syncthreads();
        if (grp != 0 || nn >= N) return;
        for (int q = 1; q < ES; ++q) {
            const int slot = (node_in_blk * C + c) * (ES - 1) + (q - 1);
#pragma unroll
            for (int m = 0; m < 16; ++m) og[m] += sb[m * NS + slot];
            ox += sb[16 * NS + slot];
        }
    }
    if (DG) {
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const long idx = NM ? ((long)n * 16 + m) * C + c : m * NC + (long)n * C + c;
            OG[idx] = ACC ? OG[idx] + og[m] : og[m];
        }
    }
    if (DX) {
        const long idx = (long)n * C + c;
        OX[idx] = ACC ? OX[idx] + ox : ox;
    }
}

// Fused cubic-Hermite spline k_conv_pair (LST_MLP_SPLINE_FUSED=1):
//   POLY=true:  reads packed float4 cubic coefficients C_k[k, l, c] (4 LDG.E.128 vs 16 LDG.E.32) + Horner eval.
//   SDX=true:   sender-owned DX — loads G[n, 0..15, c] ONCE into registers (zero G[s] gathers!) and atomicAdds
//               1 scalar per edge into zeroed L2-resident OX[s, c]. When !DG && DX && SDX: zero smem, zero barriers!
template <typename T, bool DG, bool DX, int NT, bool XB, bool ACC, int ES = 1, bool NM = false, int POLY = 0,
          bool SDX = false, typename A16 = T, bool EMETA = false>
__global__ void __launch_bounds__(256) k_conv_pair_sp(const int* __restrict__ rptr, const int* __restrict__ snd,
                                                      const int* __restrict__ ps, const int* __restrict__ sp_idx,
                                                      const T* __restrict__ sp_u, const T* __restrict__ X,
                                                      const A16* __restrict__ G, const T* __restrict__ Rk,
                                                      const T* __restrict__ Tk, const T* __restrict__ Y0,
                                                      const T* __restrict__ Y1, const T* __restrict__ Y2,
                                                      const T* __restrict__ Xb, A16* __restrict__ OG, T* __restrict__ OX,
                                                      T sp_h, T sp_invh, int N, int C,
                                                      const int4* __restrict__ emeta = nullptr) {
    const int n = blockIdx.x * (blockDim.x / (ES * C)) + threadIdx.x / (ES * C);
    const int c = threadIdx.x % C;
    const int grp = (threadIdx.x / C) % ES;
    if ((ES == 1 || (SDX && !DG)) && n >= N) return;
    const long NC = (long)N * C;
    T og[16];
    if constexpr (DG) {
#pragma unroll
        for (int m = 0; m < 16; ++m) og[m] = T(0);
    }
    T ox = T(0);
    T gn[16];
    if constexpr (SDX && DX) {
        if (n < N) {
#pragma unroll
            for (int m = 0; m < 16; ++m)
                gn[m] = ah::ld_a16<T>(G, NM ? ((long)n * 16 + m) * C + c : m * NC + (long)n * C + c);
        }
    }
    const int e0_n = n < N ? rptr[n] : 0;
    const int e1_n = n < N ? rptr[n + 1] : 0;
    const int cnt_n = e1_n - e0_n;
    const int e_beg = EMETA ? (ES > 1 ? e0_n + (cnt_n * grp) / ES : e0_n) : (e0_n + grp);
    const int e_end = EMETA ? (ES > 1 ? e0_n + (cnt_n * (grp + 1)) / ES : e1_n) : e1_n;
    const int e_step = EMETA ? 1 : ES;
    int prev_idx = -1;
    uint2 prev_c[4];
    for (int e = e_beg; e < e_end; e += e_step) {
        int ps_e, idx;
        long p, s;
        T u;
        if constexpr (EMETA) {
            const int4 em = emeta[e];
            ps_e = em.x;
            p = ps_e >> 1;
            idx = em.z;
            if (idx < 0) continue;
            s = em.y;
            if constexpr (std::is_same<T, float>::value)
                u = __int_as_float(em.w);
            else
                u = sp_u[p];
        } else {
            ps_e = ps[e];
            p = ps_e >> 1;
            idx = sp_idx[p];
            if (idx < 0) continue;
            u = sp_u[p];
            s = snd[e];
        }
        const T sgn = (ps_e & 1) ? T(-1) : T(1);
        T xs = T(0), xb = T(0);
        if constexpr (DG) {
            xs = X[s * C + c];
            if constexpr (XB) xb = Xb[s * C + c];
        }
        T gs[16];
        if constexpr (DX && !SDX) {
#pragma unroll
            for (int m = 0; m < 16; ++m)
                gs[m] = ah::ld_a16<T>(G, NM ? (s * 16 + m) * C + c : m * NC + s * C + c);
        }
        if constexpr (NT == 1) {
            T r[4];
            if constexpr (EMETA && POLY == 2 && std::is_same<T, float>::value) {
                if (idx != prev_idx) {
                    const long cbase = (long)idx * 4 * C + c;
#pragma unroll
                    for (int l = 0; l < 4; ++l) prev_c[l] = reinterpret_cast<const uint2*>(Rk)[cbase + l * C];
                    prev_idx = idx;
                }
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&prev_c[l].x));
                    const float2 b2 = __half22float2(*reinterpret_cast<const __half2*>(&prev_c[l].y));
                    r[l] = ((b2.y * u + b2.x) * u + a.y) * u + a.x;
                }
            } else if constexpr (POLY > 0) {
                const long cbase = (long)idx * 4 * C + c;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    T c0, c1, c2, c3;
                    mspl::ld_c4<T, POLY>(Rk, cbase + l * C, c0, c1, c2, c3);
                    r[l] = ((c3 * u + c2) * u + c1) * u + c0;
                }
            } else {
                T h00, h10, h11;
                mspl::hermite_r<T>(u, sp_h, h00, h10, h11);
                const long k0 = (long)idx * 4 * C + c, k1 = k0 + 4 * C;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const T r0 = Rk[k0 + l * C], r1 = Rk[k1 + l * C];
                    const T t0 = Tk[k0 + l * C], t1 = Tk[k1 + l * C];
                    r[l] = r1 + h00 * (r0 - r1) + h10 * t0 + h11 * t1;
                }
            }
            T y[16];
            ld16(Y0 + 16L * p, y);
            if constexpr (DG) {
                const T r0 = r[0] * xs, r1 = sgn * r[1] * xs, r2 = r[2] * xs, r3 = sgn * r[3] * xs;
                og[0] += r0 * y[0];
#pragma unroll
                for (int m = 1; m < 4; ++m) og[m] += r1 * y[m];
#pragma unroll
                for (int m = 4; m < 9; ++m) og[m] += r2 * y[m];
#pragma unroll
                for (int m = 9; m < 16; ++m) og[m] += r3 * y[m];
            }
            if constexpr (DX) {
                if constexpr (SDX) {
                    T q1 = gn[1] * y[1] + gn[2] * y[2] + gn[3] * y[3], q2 = T(0), q3 = T(0);
#pragma unroll
                    for (int m = 4; m < 9; ++m) q2 += gn[m] * y[m];
#pragma unroll
                    for (int m = 9; m < 16; ++m) q3 += gn[m] * y[m];
                    const T ox_e = r[0] * gn[0] * y[0] + sgn * r[1] * q1 + r[2] * q2 + sgn * r[3] * q3;
                    atomicAdd(OX + s * C + c, ox_e);
                } else {
                    T q1 = gs[1] * y[1] + gs[2] * y[2] + gs[3] * y[3], q2 = T(0), q3 = T(0);
#pragma unroll
                    for (int m = 4; m < 9; ++m) q2 += gs[m] * y[m];
#pragma unroll
                    for (int m = 9; m < 16; ++m) q3 += gs[m] * y[m];
                    ox += r[0] * gs[0] * y[0] - sgn * r[1] * q1 + r[2] * q2 - sgn * r[3] * q3;
                }
            }
        } else {
            // NT == 2: term 0 = T(r) with Y0 (Yw), term 1 = R(r) with Y1 (gyb) and optional Y2 (Y, Xb).
            T tt[4], rr[4];
            if constexpr (POLY > 0) {
                const long cbase = (long)idx * 4 * C + c;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    T c0, c1, c2, c3;
                    mspl::ld_c4<T, POLY>(Rk, cbase + l * C, c0, c1, c2, c3);
                    rr[l] = ((c3 * u + c2) * u + c1) * u + c0;
                    tt[l] = ((T(3) * c3 * u + T(2) * c2) * u + c1) * sp_invh;
                }
            } else {
                T h00, h10, h11, d00, d10, d11;
                mspl::hermite_rt<T>(u, sp_h, sp_invh, h00, h10, h11, d00, d10, d11);
                const long k0 = (long)idx * 4 * C + c, k1 = k0 + 4 * C;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const T r0 = Rk[k0 + l * C], r1 = Rk[k1 + l * C];
                    const T t0 = Tk[k0 + l * C], t1 = Tk[k1 + l * C];
                    const T dr = r0 - r1;
                    tt[l] = d00 * dr + d10 * t0 + d11 * t1;
                    rr[l] = r1 + h00 * dr + h10 * t0 + h11 * t1;
                }
            }
            T ox_e = T(0);
            // Term 0: tt with Y0
            {
                T y[16];
                ld16(Y0 + 16L * p, y);
                if constexpr (DG) {
                    const T r0 = tt[0] * xs, r1 = sgn * tt[1] * xs, r2 = tt[2] * xs, r3 = sgn * tt[3] * xs;
                    og[0] += r0 * y[0];
#pragma unroll
                    for (int m = 1; m < 4; ++m) og[m] += r1 * y[m];
#pragma unroll
                    for (int m = 4; m < 9; ++m) og[m] += r2 * y[m];
#pragma unroll
                    for (int m = 9; m < 16; ++m) og[m] += r3 * y[m];
                }
                if constexpr (DX) {
                    if constexpr (SDX) {
                        T q1 = gn[1] * y[1] + gn[2] * y[2] + gn[3] * y[3], q2 = T(0), q3 = T(0);
#pragma unroll
                        for (int m = 4; m < 9; ++m) q2 += gn[m] * y[m];
#pragma unroll
                        for (int m = 9; m < 16; ++m) q3 += gn[m] * y[m];
                        ox_e += tt[0] * gn[0] * y[0] + sgn * tt[1] * q1 + tt[2] * q2 + sgn * tt[3] * q3;
                    } else {
                        T q1 = gs[1] * y[1] + gs[2] * y[2] + gs[3] * y[3], q2 = T(0), q3 = T(0);
#pragma unroll
                        for (int m = 4; m < 9; ++m) q2 += gs[m] * y[m];
#pragma unroll
                        for (int m = 9; m < 16; ++m) q3 += gs[m] * y[m];
                        ox += tt[0] * gs[0] * y[0] - sgn * tt[1] * q1 + tt[2] * q2 - sgn * tt[3] * q3;
                    }
                }
            }
            // Term 1: rr with Y1 (and Y2, xb when XB)
            {
                T y[16];
                ld16(Y1 + 16L * p, y);
                if constexpr (DG) {
                    const T r0 = rr[0] * xs, r1 = sgn * rr[1] * xs, r2 = rr[2] * xs, r3 = sgn * rr[3] * xs;
                    og[0] += r0 * y[0];
#pragma unroll
                    for (int m = 1; m < 4; ++m) og[m] += r1 * y[m];
#pragma unroll
                    for (int m = 4; m < 9; ++m) og[m] += r2 * y[m];
#pragma unroll
                    for (int m = 9; m < 16; ++m) og[m] += r3 * y[m];
                }
                if constexpr (DX) {
                    if constexpr (SDX) {
                        T q1 = gn[1] * y[1] + gn[2] * y[2] + gn[3] * y[3], q2 = T(0), q3 = T(0);
#pragma unroll
                        for (int m = 4; m < 9; ++m) q2 += gn[m] * y[m];
#pragma unroll
                        for (int m = 9; m < 16; ++m) q3 += gn[m] * y[m];
                        ox_e += rr[0] * gn[0] * y[0] + sgn * rr[1] * q1 + rr[2] * q2 + sgn * rr[3] * q3;
                    } else {
                        T q1 = gs[1] * y[1] + gs[2] * y[2] + gs[3] * y[3], q2 = T(0), q3 = T(0);
#pragma unroll
                        for (int m = 4; m < 9; ++m) q2 += gs[m] * y[m];
#pragma unroll
                        for (int m = 9; m < 16; ++m) q3 += gs[m] * y[m];
                        ox += rr[0] * gs[0] * y[0] - sgn * rr[1] * q1 + rr[2] * q2 - sgn * rr[3] * q3;
                    }
                }
                if constexpr (DG && XB) {
                    ld16(Y2 + 16L * p, y);
                    const T b0 = rr[0] * xb, b1 = sgn * rr[1] * xb, b2 = rr[2] * xb, b3 = sgn * rr[3] * xb;
                    og[0] += b0 * y[0];
#pragma unroll
                    for (int m = 1; m < 4; ++m) og[m] += b1 * y[m];
#pragma unroll
                    for (int m = 4; m < 9; ++m) og[m] += b2 * y[m];
#pragma unroll
                    for (int m = 9; m < 16; ++m) og[m] += b3 * y[m];
                }
            }
            if constexpr (DX && SDX) atomicAdd(OX + s * C + c, ox_e);
        }
    }
    if constexpr (SDX && !DG) return;
    if constexpr (ES > 1) {
        constexpr int NS = 256 / ES * (ES - 1);
        constexpr int NV = (DX && !SDX) ? 17 : 16;
        __shared__ __align__(16) unsigned char sraw[NV * NS * sizeof(T)];
        T* sb = reinterpret_cast<T*>(sraw);
        const int node_in_blk = threadIdx.x / (ES * C);
        for (int q = 1; q < ES; ++q) {
            const int slot = (node_in_blk * C + c) * (ES - 1) + (q - 1);
            if (grp == q) {
                if constexpr (DG) {
#pragma unroll
                    for (int m = 0; m < 16; ++m) sb[m * NS + slot] = og[m];
                }
                if constexpr (DX && !SDX) sb[16 * NS + slot] = ox;
            }
        }
        __syncthreads();
        if (grp != 0 || n >= N) return;
        for (int q = 1; q < ES; ++q) {
            const int slot = (node_in_blk * C + c) * (ES - 1) + (q - 1);
            if constexpr (DG) {
#pragma unroll
                for (int m = 0; m < 16; ++m) og[m] += sb[m * NS + slot];
            }
            if constexpr (DX && !SDX) ox += sb[16 * NS + slot];
        }
    }
    if constexpr (DG) {
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const long idx = NM ? ((long)n * 16 + m) * C + c : m * NC + (long)n * C + c;
            ah::st_a16<A16>(OG, idx, ACC ? ah::ld_a16<T>(OG, idx) + og[m] : og[m]);
        }
    }
    if constexpr (DX && !SDX) {
        const long idx = (long)n * C + c;
        OX[idx] = ACC ? OX[idx] + ox : ox;
    }
}
//   Tb[p, l, c] = sum_{eps in p} sum_{m in l} G[m, r, c] Yw[eps, m] x[s, c]
//   Rb[p, l, c] = sum_{eps in p} sum_{m in l} (G[m, r, c] (gyb[eps, m] x[s, c] + Y[eps, m] xb[s, c])
//                                             + A[m, r, c] Y[eps, m] x[s, c])          (r = r(eps), s = s(eps))
template <typename T, bool XB, int ES = 1>
__global__ void __launch_bounds__(256) k_slot(const int* __restrict__ rptr, const int* __restrict__ snd,
                                              const int* __restrict__ pidx, const int* __restrict__ pe0,
                                              const int* __restrict__ pe1, const T* __restrict__ G,
                                              const T* __restrict__ A, const T* __restrict__ X,
                                              const T* __restrict__ Xb, const T* __restrict__ Yw,
                                              const T* __restrict__ Gy, const T* __restrict__ Y, T* __restrict__ Tb,
                                              T* __restrict__ Rb, int N, int C) {
    const int n = blockIdx.x * (blockDim.x / (ES * C)) + threadIdx.x / (ES * C);
    const int c = threadIdx.x % C;
    const int grp = ES > 1 ? (threadIdx.x / C) % ES : 0;
    if (n >= N) return;
    const long NC = (long)N * C;
    T gn[16], an[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        gn[m] = G[m * NC + (long)n * C + c];
        an[m] = A[m * NC + (long)n * C + c];
    }
    const T xn = X[(long)n * C + c];
    const T xbn = XB ? Xb[(long)n * C + c] : T(0);
    const int e1 = rptr[n + 1];
    for (int e = rptr[n] + grp; e < e1; e += ES) {
        const long p = pidx[e];
        if (pe0[p] != e) continue;
        const int ep = pe1[p];
        const long s = snd[e];
        const T xs = X[s * C + c];
        const T xbs = XB ? Xb[s * C + c] : T(0);
        T tb[4] = {T(0), T(0), T(0), T(0)}, rb[4] = {T(0), T(0), T(0), T(0)};
        // eps = e: receiver n (own G, A), sender s
        {
            T yw[16], gy[16], y[16];
            ld16(Yw + 16L * e, yw);
            ld16(Gy + 16L * e, gy);
            ld16(Y + 16L * e, y);
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                const int l = lofm(m);
                tb[l] += gn[m] * yw[m];
                rb[l] += gn[m] * (gy[m] * xs + y[m] * xbs) + an[m] * y[m] * xs;
            }
#pragma unroll
            for (int l = 0; l < 4; ++l) tb[l] *= xs;
        }
        // eps = e': receiver s (gathered G, A), sender n
        {
            T yw[16], gy[16], y[16];
            ld16(Yw + 16L * ep, yw);
            ld16(Gy + 16L * ep, gy);
            ld16(Y + 16L * ep, y);
            T t2[4] = {T(0), T(0), T(0), T(0)};
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                const int l = lofm(m);
                const T gs = G[m * NC + s * C + c];
                const T as = A[m * NC + s * C + c];
                t2[l] += gs * yw[m];
                rb[l] += gs * (gy[m] * xn + y[m] * xbn) + as * y[m] * xn;
            }
#pragma unroll
            for (int l = 0; l < 4; ++l) tb[l] += t2[l] * xn;
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            Tb[p * 4 * C + l * C + c] = tb[l];
            Rb[p * 4 * C + l * C + c] = rb[l];
        }
    }
}

// Pair-parity k_slot (LST_PAIR_BASIS=1): reads YwP/GyP/YP once per pair (no ep/pe0/pe1 loads) and negates odd l for ep.
// NM=true: node-major [N, 16, C] layout for G and A (LST_NODE_MAJOR=1).
template <typename T, bool XB, int ES = 1, bool NM = false>
__global__ void __launch_bounds__(256) k_slot_pair(const int* __restrict__ rptr, const int* __restrict__ snd,
                                                    const int* __restrict__ ps, const T* __restrict__ G,
                                                    const T* __restrict__ A, const T* __restrict__ X,
                                                    const T* __restrict__ Xb, const T* __restrict__ YwP,
                                                    const T* __restrict__ GyP, const T* __restrict__ YP,
                                                    T* __restrict__ Tb, T* __restrict__ Rb, int N, int C,
                                                    const int* __restrict__ norder = nullptr) {
    const int nn = blockIdx.x * (blockDim.x / (ES * C)) + threadIdx.x / (ES * C);
    const int c = threadIdx.x % C;
    const int grp = ES > 1 ? (threadIdx.x / C) % ES : 0;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    const long NC = (long)N * C;
    T gn[16], an[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const long idx = NM ? ((long)n * 16 + m) * C + c : m * NC + (long)n * C + c;
        gn[m] = G[idx];
        an[m] = A[idx];
    }
    const T xn = X[(long)n * C + c];
    const T xbn = XB ? Xb[(long)n * C + c] : T(0);
    const int e1 = rptr[n + 1];
    for (int e = rptr[n] + grp; e < e1; e += ES) {
        const int ps_e = ps[e];
        if (ps_e & 1) continue;
        const long p = ps_e >> 1;
        const long s = snd[e];
        const T xs = X[s * C + c];
        const T xbs = XB ? Xb[s * C + c] : T(0);
        T yw[16], gy[16], y[16];
        ld16(YwP + 16L * p, yw);
        ld16(GyP + 16L * p, gy);
        ld16(YP + 16L * p, y);
        T tb[4] = {T(0), T(0), T(0), T(0)}, rb[4] = {T(0), T(0), T(0), T(0)};
        T rxb[4] = {T(0), T(0), T(0), T(0)};
        // eps = e (canonical, +1)
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int l = lofm(m);
            tb[l] += gn[m] * yw[m];
            rb[l] += gn[m] * gy[m] + an[m] * y[m];
            if constexpr (XB) rxb[l] += gn[m] * y[m];
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            tb[l] *= xs;
            rb[l] = rb[l] * xs + (XB ? rxb[l] * xbs : T(0));
        }
        // eps = e' (reverse, (-1)^l parity on Yw, Gy, Y)
        T t2[4] = {T(0), T(0), T(0), T(0)}, r2[4] = {T(0), T(0), T(0), T(0)}, r2x[4] = {T(0), T(0), T(0), T(0)};
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int l = lofm(m);
            const long sidx = NM ? (s * 16 + m) * C + c : m * NC + s * C + c;
            const T gs = G[sidx];
            const T as = A[sidx];
            t2[l] += gs * yw[m];
            r2[l] += gs * gy[m] + as * y[m];
            if constexpr (XB) r2x[l] += gs * y[m];
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const T sgn = (l & 1) ? T(-1) : T(1);
            tb[l] += sgn * t2[l] * xn;
            rb[l] += sgn * (r2[l] * xn + (XB ? r2x[l] * xbn : T(0)));
            Tb[p * 4 * C + l * C + c] = tb[l];
            Rb[p * 4 * C + l * C + c] = rb[l];
        }
    }
}

// Fused cubic-Hermite spline k_slot_pair (LST_MLP_SPLINE_FUSED=1):
//   BAL=true:       balanced tournament pair ownership so every atom owns ~50% of its edges (eliminates 50->0 tail imbalance).
//   NO_GATHER=true: visits all incoming edges using ONLY gn/an in registers (zero G[s]/A[s] gathers; rejected: 2x atomics).
//   POLY=true:      scatters into gC_k[K, 4, 4, C] (4 coalesced L2 atomics per l), converted by k_spline_c_to_rt_bwd.
template <typename T, bool XB, int ES = 1, bool NM = false, bool POLY = false, bool NO_GATHER = false, bool BAL = false>
__global__ void __launch_bounds__(256) k_slot_pair_sp(const int* __restrict__ rptr, const int* __restrict__ snd,
                                                      const int* __restrict__ ps, const int* __restrict__ sp_idx,
                                                      const T* __restrict__ sp_u, const T* __restrict__ G,
                                                      const T* __restrict__ A, const T* __restrict__ X,
                                                      const T* __restrict__ Xb, const T* __restrict__ YwP,
                                                      const T* __restrict__ GyP, const T* __restrict__ YP,
                                                      T* __restrict__ gR_k, T* __restrict__ gT_k, T sp_h, T sp_invh,
                                                      int N, int C) {
    const int n = blockIdx.x * (blockDim.x / (ES * C)) + threadIdx.x / (ES * C);
    const int c = threadIdx.x % C;
    const int grp = ES > 1 ? (threadIdx.x / C) % ES : 0;
    if (n >= N) return;
    const long NC = (long)N * C;
    T gn[16], an[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const long idx = NM ? ((long)n * 16 + m) * C + c : m * NC + (long)n * C + c;
        gn[m] = G[idx];
        an[m] = A[idx];
    }
    const T xn = NO_GATHER ? T(0) : X[(long)n * C + c];
    const T xbn = (!NO_GATHER && XB) ? Xb[(long)n * C + c] : T(0);
    const int e1 = rptr[n + 1];
    for (int e = rptr[n] + grp; e < e1; e += ES) {
        const int ps_e = ps[e];
        const long s = snd[e];
        if constexpr (!NO_GATHER) {
            if constexpr (BAL) {
                const bool own = (n == s) ? ((ps_e & 1) == 0)
                                          : ((n < s) ? (((n + (int)s) & 1) == 0) : (((n + (int)s) & 1) == 1));
                if (!own) continue;
            } else {
                if (ps_e & 1) continue;
            }
        }
        const long p = ps_e >> 1;
        const int kidx = sp_idx[p];
        if (kidx < 0) continue;
        const T u = sp_u[p];
        const T xs = X[s * C + c];
        const T xbs = XB ? Xb[s * C + c] : T(0);
        T yw[16], gy[16], y[16];
        ld16(YwP + 16L * p, yw);
        ld16(GyP + 16L * p, gy);
        ld16(YP + 16L * p, y);
        T tb[4] = {T(0), T(0), T(0), T(0)}, rb[4] = {T(0), T(0), T(0), T(0)};
        T rxb[4] = {T(0), T(0), T(0), T(0)};
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int l = lofm(m);
            tb[l] += gn[m] * yw[m];
            rb[l] += gn[m] * gy[m] + an[m] * y[m];
            if constexpr (XB) rxb[l] += gn[m] * y[m];
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            tb[l] *= xs;
            rb[l] = rb[l] * xs + (XB ? rxb[l] * xbs : T(0));
        }
        if constexpr (NO_GATHER) {
            const T esgn = (ps_e & 1) ? T(-1) : T(1);
            if constexpr (POLY) {
                const T u2 = u * u, u3 = u2 * u;
                const T ui = u * sp_invh, u2i = u2 * sp_invh;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const T sg = (l & 1) ? esgn : T(1);
                    const T tv = sg * tb[l], rv = sg * rb[l];
                    const long base = ((long)kidx * 4 + l) * 4 * C + c;
                    atomicAdd(gR_k + base + 0 * C, rv);
                    atomicAdd(gR_k + base + 1 * C, u * rv + sp_invh * tv);
                    atomicAdd(gR_k + base + 2 * C, u2 * rv + T(2) * ui * tv);
                    atomicAdd(gR_k + base + 3 * C, u3 * rv + T(3) * u2i * tv);
                }
            } else {
                T h00, h10, h11, d00, d10, d11;
                mspl::hermite_rt<T>(u, sp_h, sp_invh, h00, h10, h11, d00, d10, d11);
                const T h01 = T(1) - h00;
                const long k0 = (long)kidx * 4 * C + c, k1 = k0 + 4 * C;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const T sg = (l & 1) ? esgn : T(1);
                    const T tv = sg * tb[l], rv = sg * rb[l];
                    atomicAdd(gR_k + k0 + l * C, h00 * rv + d00 * tv);
                    atomicAdd(gR_k + k1 + l * C, h01 * rv - d00 * tv);
                    atomicAdd(gT_k + k0 + l * C, h10 * rv + d10 * tv);
                    atomicAdd(gT_k + k1 + l * C, h11 * rv + d11 * tv);
                }
            }
        } else {
            T t2[4] = {T(0), T(0), T(0), T(0)}, r2[4] = {T(0), T(0), T(0), T(0)}, r2x[4] = {T(0), T(0), T(0), T(0)};
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                const int l = lofm(m);
                const long sidx = NM ? (s * 16 + m) * C + c : m * NC + s * C + c;
                const T gs = G[sidx];
                const T as = A[sidx];
                t2[l] += gs * yw[m];
                r2[l] += gs * gy[m] + as * y[m];
                if constexpr (XB) r2x[l] += gs * y[m];
            }
            const T esgn = (BAL && (ps_e & 1)) ? T(-1) : T(1);
            if constexpr (POLY) {
                const T u2 = u * u, u3 = u2 * u;
                const T ui = u * sp_invh, u2i = u2 * sp_invh;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const T odd = (l & 1) ? esgn : T(1);
                    const T sgn2 = (l & 1) ? T(-1) : T(1);
                    const T tv = odd * (tb[l] + sgn2 * t2[l] * xn);
                    const T rv = odd * (rb[l] + sgn2 * (r2[l] * xn + (XB ? r2x[l] * xbn : T(0))));
                    const long base = ((long)kidx * 4 + l) * 4 * C + c;
                    atomicAdd(gR_k + base + 0 * C, rv);
                    atomicAdd(gR_k + base + 1 * C, u * rv + sp_invh * tv);
                    atomicAdd(gR_k + base + 2 * C, u2 * rv + T(2) * ui * tv);
                    atomicAdd(gR_k + base + 3 * C, u3 * rv + T(3) * u2i * tv);
                }
            } else {
                T h00, h10, h11, d00, d10, d11;
                mspl::hermite_rt<T>(u, sp_h, sp_invh, h00, h10, h11, d00, d10, d11);
                const T h01 = T(1) - h00;
                const long k0 = (long)kidx * 4 * C + c, k1 = k0 + 4 * C;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const T odd = (l & 1) ? esgn : T(1);
                    const T sgn2 = (l & 1) ? T(-1) : T(1);
                    const T tv = odd * (tb[l] + sgn2 * t2[l] * xn);
                    const T rv = odd * (rb[l] + sgn2 * (r2[l] * xn + (XB ? r2x[l] * xbn : T(0))));
                    atomicAdd(gR_k + k0 + l * C, h00 * rv + d00 * tv);
                    atomicAdd(gR_k + k1 + l * C, h01 * rv - d00 * tv);
                    atomicAdd(gT_k + k0 + l * C, h10 * rv + d10 * tv);
                    atomicAdd(gT_k + k1 + l * C, h11 * rv + d11 * tv);
                }
            }
        }
    }
}

// Fused slot + Xd convolution (LST_SLOT_FUSE_XD=1): computes BOTH radial spline cotangents (gR_k/gT_k or gC_k)
// AND the backward Xd += sum R * Y * A convolution in ONE pass over canonical pairs, reusing an[16], as[16], y[16],
// sp_idx, sp_u, and Hermite weights already in registers — completely deleting both conv_sp(Xd1) and conv_sp(Xd0)!
template <typename T, bool XB, int ES = 2, bool NM = true, int POLY_FWD = 0, bool POLY_BWD = false, typename A16 = T>
__global__ void __launch_bounds__(256) k_slot_xd_sp(const int* __restrict__ rptr, const int* __restrict__ snd,
                                                    const int* __restrict__ ps, const int* __restrict__ sp_idx,
                                                    const T* __restrict__ sp_u, const A16* __restrict__ G,
                                                    const A16* __restrict__ A, const T* __restrict__ X,
                                                    const T* __restrict__ Xb, const T* __restrict__ YwP,
                                                    const T* __restrict__ GyP, const T* __restrict__ YP,
                                                    const T* __restrict__ Rk, const T* __restrict__ Tk,
                                                    T* __restrict__ gR_k, T* __restrict__ gT_k, T* __restrict__ Xd,
                                                    T sp_h, T sp_invh, int N, int C,
                                                    const int* __restrict__ rptr0 = nullptr) {
    const int n = blockIdx.x * (blockDim.x / (ES * C)) + threadIdx.x / (ES * C);
    const int c = threadIdx.x % C;
    const int grp = ES > 1 ? (threadIdx.x / C) % ES : 0;
    if (n >= N) return;
    const long NC = (long)N * C;
    T gn[16], an[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const long idx = NM ? ((long)n * 16 + m) * C + c : m * NC + (long)n * C + c;
        gn[m] = ah::ld_a16<T>(G, idx);
        an[m] = ah::ld_a16<T>(A, idx);
    }
    const T xn = X[(long)n * C + c];
    const T xbn = XB ? Xb[(long)n * C + c] : T(0);
    T xd_n = T(0);
    const int e1 = rptr0 ? rptr0[n] : rptr[n + 1];
    for (int e = rptr[n] + grp; e < e1; e += ES) {
        const int ps_e = ps[e];
        if (ps_e & 1) continue;
        const long p = ps_e >> 1;
        const int kidx = sp_idx[p];
        if (kidx < 0) continue;
        const T u = sp_u[p];
        const long s = snd[e];
        const T xs = X[s * C + c];
        const T xbs = XB ? Xb[s * C + c] : T(0);
        // Stream Yw, Gy, Y in 4-element float4 chunks (12 live regs instead of 48 -> doubles/triples occupancy!)
        T tb[4] = {T(0), T(0), T(0), T(0)}, rbg[4] = {T(0), T(0), T(0), T(0)}, ayn[4] = {T(0), T(0), T(0), T(0)};
        T rxb[4] = {T(0), T(0), T(0), T(0)};
        T t2[4] = {T(0), T(0), T(0), T(0)}, r2g[4] = {T(0), T(0), T(0), T(0)}, ays[4] = {T(0), T(0), T(0), T(0)};
        T r2x[4] = {T(0), T(0), T(0), T(0)};
#pragma unroll
        for (int m0 = 0; m0 < 16; m0 += 4) {
            T yw4[4], gy4[4], y4[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + m0);
                const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + m0);
                const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + m0);
                yw4[0] = v0.x; yw4[1] = v0.y; yw4[2] = v0.z; yw4[3] = v0.w;
                gy4[0] = v1.x; gy4[1] = v1.y; gy4[2] = v1.z; gy4[3] = v1.w;
                y4[0] = v2.x; y4[1] = v2.y; y4[2] = v2.z; y4[3] = v2.w;
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    yw4[j] = YwP[16L * p + m0 + j];
                    gy4[j] = GyP[16L * p + m0 + j];
                    y4[j] = YP[16L * p + m0 + j];
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int m = m0 + j;
                const int l = lofm(m);
                const T ywv = yw4[j], gyv = gy4[j], yv = y4[j];
                tb[l] += gn[m] * ywv;
                rbg[l] += gn[m] * gyv;
                ayn[l] += an[m] * yv;
                if constexpr (XB) rxb[l] += gn[m] * yv;
                const long sidx = NM ? (s * 16 + m) * C + c : m * NC + s * C + c;
                const T gs = ah::ld_a16<T>(G, sidx);
                const T as = ah::ld_a16<T>(A, sidx);
                t2[l] += gs * ywv;
                r2g[l] += gs * gyv;
                ays[l] += as * yv;
                if constexpr (XB) r2x[l] += gs * yv;
            }
        }
        T h00 = T(0), h10 = T(0), h11 = T(0), d00 = T(0), d10 = T(0), d11 = T(0);
        if constexpr (POLY_FWD == 0 || !POLY_BWD)
            mspl::hermite_rt<T>(u, sp_h, sp_invh, h00, h10, h11, d00, d10, d11);
        const T h01 = T(1) - h00;
        const T u2 = u * u, u3 = u2 * u;
        const T ui = u * sp_invh, u2i = u2 * sp_invh;
        const long k0 = (long)kidx * 4 * C + c, k1 = k0 + 4 * C;
        T xd_s = T(0);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const T sgn2 = (l & 1) ? T(-1) : T(1);
            // 1) Interpolate R[p, l, c] ONCE per pair -> accumulate both Xd[s] and Xd[n]
            T rval;
            if constexpr (POLY_FWD > 0) {
                T c0, c1, c2, c3;
                mspl::ld_c4<T, POLY_FWD>(Rk, k0 + l * C, c0, c1, c2, c3);
                rval = ((c3 * u + c2) * u + c1) * u + c0;
            } else {
                const T r0 = Rk[k0 + l * C], r1 = Rk[k1 + l * C];
                const T t0 = Tk[k0 + l * C], t1 = Tk[k1 + l * C];
                rval = r1 + h00 * (r0 - r1) + h10 * t0 + h11 * t1;
            }
            xd_s += rval * ayn[l];
            xd_n += sgn2 * rval * ays[l];
            // 2) Scatter radial cotangent (rv, tv) into L2-resident knot gradients
            const T tv = tb[l] * xs + sgn2 * t2[l] * xn;
            const T rv = (rbg[l] + ayn[l]) * xs + (XB ? rxb[l] * xbs : T(0)) +
                         sgn2 * ((r2g[l] + ays[l]) * xn + (XB ? r2x[l] * xbn : T(0)));
            if constexpr (POLY_BWD) {
                const long base = ((long)kidx * 4 + l) * 4 * C + c;
                atomicAdd(gR_k + base + 0 * C, rv);
                atomicAdd(gR_k + base + 1 * C, u * rv + sp_invh * tv);
                atomicAdd(gR_k + base + 2 * C, u2 * rv + T(2) * ui * tv);
                atomicAdd(gR_k + base + 3 * C, u3 * rv + T(3) * u2i * tv);
            } else {
                atomicAdd(gR_k + k0 + l * C, h00 * rv + d00 * tv);
                atomicAdd(gR_k + k1 + l * C, h01 * rv - d00 * tv);
                atomicAdd(gT_k + k0 + l * C, h10 * rv + d10 * tv);
                atomicAdd(gT_k + k1 + l * C, h11 * rv + d11 * tv);
            }
        }
        atomicAdd(Xd + s * C + c, xd_s);
    }
    if (e1 > rptr[n]) atomicAdd(Xd + (long)n * C + c, xd_n);
}

// Knot-bucketed fused slot + Xd kernel (LST_SLOT_KB=1):
// Grid: dim3(max_tiles_per_frame, B), 64 threads (c = threadIdx.x in 0..63, 2 warps/block).
// Each block processes one balanced tile {kidx, p_beg, p_end} (<= 32 or 64 pairs) of ONE fixed knot kidx in ONE frame:
//   - Loads C_kh[kidx, 0..3, c] ONCE before the loop (26x-51x fewer C_kh loads)
//   - Accumulates gC[4][4] in registers across the entire tile -> 16 atomicAdds ONCE per tile (26x-51x fewer atomics!)
//   - Streams G[u], A[u], G[v], A[v] with combined g_comb = gu*xv + sgn*gv*xu (~60 regs -> 2x+ occupancy!)
//   - Coalesces Xd[u] in registers across consecutive same-u pairs (23%-37% hits)
template <typename T, bool XB, int POLY_FWD = 2, typename A16 = T>
__global__ void __launch_bounds__(64) k_slot_xd_kb(
    const int4* __restrict__ f_meta, const int* __restrict__ toff, const int4* __restrict__ tiles,
    const int2* __restrict__ puv, const T* __restrict__ sp_u, const A16* __restrict__ G,
    const A16* __restrict__ A, const T* __restrict__ X, const T* __restrict__ Xb,
    const T* __restrict__ YwP, const T* __restrict__ GyP, const T* __restrict__ YP,
    const T* __restrict__ Rk, const T* __restrict__ Tk, T* __restrict__ gC_k, T* __restrict__ Xd,
    T sp_h, T sp_invh, int C) {
    const int bi = blockIdx.y;
    const int4 fm = f_meta[bi];
    const int fid = fm.x;
    if (fid < 0) return;
    const int t0 = toff[fid], t1 = toff[fid + 1];
    const int ti = t0 + (int)blockIdx.x;
    if (ti >= t1) return;
    const int4 tl = tiles[ti];
    const int kidx = tl.x;
    const long p_beg = (long)fm.z + tl.y, p_end = (long)fm.z + tl.z;
    const int c = threadIdx.x;

    uint2 ckh[4];
    T c0[4], c1[4], c2[4], c3[4];
    if constexpr (POLY_FWD == 2 && std::is_same<T, float>::value) {
        const long k0 = (long)kidx * 4 * C + c;
#pragma unroll
        for (int l = 0; l < 4; ++l) ckh[l] = reinterpret_cast<const uint2*>(Rk)[k0 + l * C];
    } else if constexpr (POLY_FWD > 0) {
        const long k0 = (long)kidx * 4 * C + c;
#pragma unroll
        for (int l = 0; l < 4; ++l) mspl::ld_c4<T, POLY_FWD>(Rk, k0 + l * C, c0[l], c1[l], c2[l], c3[l]);
    } else {
        const long k0 = (long)kidx * 4 * C + c, k1 = k0 + 4 * C;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const T r0 = Rk[k0 + l * C], r1 = Rk[k1 + l * C];
            const T t0v = Tk[k0 + l * C], t1v = Tk[k1 + l * C];
            const T dr = r1 - r0;
            c0[l] = r0;
            c1[l] = sp_h * t0v;
            c2[l] = T(3) * dr - sp_h * (T(2) * t0v + t1v);
            c3[l] = T(-2) * dr + sp_h * (t0v + t1v);
        }
    }
    T gc[4][4];
#pragma unroll
    for (int l = 0; l < 4; ++l)
#pragma unroll
        for (int q = 0; q < 4; ++q) gc[l][q] = T(0);

    int cur_u = -1;
    T xd_u = T(0);
    for (long p = p_beg; p < p_end; ++p) {
        const int2 uv = puv[p];
        const int u = uv.x, v = uv.y;
        if (u != cur_u) {
            if (cur_u >= 0) atomicAdd(Xd + (long)cur_u * C + c, xd_u);
            cur_u = u;
            xd_u = T(0);
        }
        const T xu = X[(long)u * C + c], xv = X[(long)v * C + c];
        const T xbu = XB ? Xb[(long)u * C + c] : T(0);
        const T xbv = XB ? Xb[(long)v * C + c] : T(0);
        const long ubase = (long)u * 16 * C + c;
        const long vbase = (long)v * 16 * C + c;
        T tv[4] = {T(0), T(0), T(0), T(0)}, rv[4] = {T(0), T(0), T(0), T(0)};
        T ayu[4] = {T(0), T(0), T(0), T(0)}, ayv[4] = {T(0), T(0), T(0), T(0)};
#pragma unroll
        for (int m0 = 0; m0 < 16; m0 += 4) {
            T yw4[4], gy4[4], y4[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 v0 = *reinterpret_cast<const float4*>(YwP + 16L * p + m0);
                const float4 v1 = *reinterpret_cast<const float4*>(GyP + 16L * p + m0);
                const float4 v2 = *reinterpret_cast<const float4*>(YP + 16L * p + m0);
                yw4[0] = v0.x; yw4[1] = v0.y; yw4[2] = v0.z; yw4[3] = v0.w;
                gy4[0] = v1.x; gy4[1] = v1.y; gy4[2] = v1.z; gy4[3] = v1.w;
                y4[0] = v2.x; y4[1] = v2.y; y4[2] = v2.z; y4[3] = v2.w;
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    yw4[j] = YwP[16L * p + m0 + j];
                    gy4[j] = GyP[16L * p + m0 + j];
                    y4[j] = YP[16L * p + m0 + j];
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int m = m0 + j;
                const int l = lofm(m);
                const T sgn2 = (l & 1) ? T(-1) : T(1);
                const T gu = ah::ld_a16<T>(G, ubase + (long)m * C);
                const T gv = ah::ld_a16<T>(G, vbase + (long)m * C);
                const T au = ah::ld_a16<T>(A, ubase + (long)m * C);
                const T av = ah::ld_a16<T>(A, vbase + (long)m * C);
                const T g_comb = gu * xv + sgn2 * gv * xu;
                const T yv = y4[j];
                tv[l] += g_comb * yw4[j];
                if constexpr (XB)
                    rv[l] += g_comb * gy4[j] + (gu * xbv + sgn2 * gv * xbu) * yv;
                else
                    rv[l] += g_comb * gy4[j];
                ayu[l] += au * yv;
                ayv[l] += av * yv;
            }
        }
        const T us = sp_u[p];
        const T u2 = us * us, u3 = u2 * us;
        const T ui = us * sp_invh, u2i = u2 * sp_invh;
        T xd_v = T(0);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const T sgn2 = (l & 1) ? T(-1) : T(1);
            T rval;
            if constexpr (POLY_FWD == 2 && std::is_same<T, float>::value) {
                const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&ckh[l].x));
                const float2 b2 = __half22float2(*reinterpret_cast<const __half2*>(&ckh[l].y));
                rval = ((b2.y * us + b2.x) * us + a.y) * us + a.x;
            } else {
                rval = ((c3[l] * us + c2[l]) * us + c1[l]) * us + c0[l];
            }
            xd_v += rval * ayu[l];
            xd_u += sgn2 * rval * ayv[l];
            const T rvl = rv[l] + ayu[l] * xv + sgn2 * ayv[l] * xu;
            const T tvl = tv[l];
            gc[l][0] += rvl;
            gc[l][1] += us * rvl + sp_invh * tvl;
            gc[l][2] += u2 * rvl + T(2) * ui * tvl;
            gc[l][3] += u3 * rvl + T(3) * u2i * tvl;
        }
        atomicAdd(Xd + (long)v * C + c, xd_v);
    }
    if (cur_u >= 0) atomicAdd(Xd + (long)cur_u * C + c, xd_u);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const long base = ((long)kidx * 4 + l) * 4 * C + c;
        atomicAdd(gC_k + base + 0 * C, gc[l][0]);
        atomicAdd(gC_k + base + 1 * C, gc[l][1]);
        atomicAdd(gC_k + base + 2 * C, gc[l][2]);
        atomicAdd(gC_k + base + 3 * C, gc[l][3]);
    }
}

// Low-register per-degree scalar streaming k_slot_xd_sp (LST_SLOT_SCALAR=1): processes l = 0, 1, 2, 3 sequentially so
// only 4 scalar accumulators (tv, rv, ayn, ays) are live instead of 32 array floats -> ~64-80 regs (3-4 blocks/SM)!
template <typename T, bool XB, int ES = 2, bool NM = true, int POLY_FWD = 0, bool POLY_BWD = false>
__global__ void __launch_bounds__(256, 3) k_slot_xd_sp_sc(const int* __restrict__ rptr, const int* __restrict__ snd,
                                                          const int* __restrict__ ps, const int* __restrict__ sp_idx,
                                                          const T* __restrict__ sp_u, const T* __restrict__ G,
                                                          const T* __restrict__ A, const T* __restrict__ X,
                                                          const T* __restrict__ Xb, const T* __restrict__ YwP,
                                                          const T* __restrict__ GyP, const T* __restrict__ YP,
                                                          const T* __restrict__ Rk, const T* __restrict__ Tk,
                                                          T* __restrict__ gR_k, T* __restrict__ gT_k,
                                                          T* __restrict__ Xd, T sp_h, T sp_invh, int N, int C) {
    const int n = blockIdx.x * (blockDim.x / (ES * C)) + threadIdx.x / (ES * C);
    const int c = threadIdx.x % C;
    const int grp = ES > 1 ? (threadIdx.x / C) % ES : 0;
    if (n >= N) return;
    const long NC = (long)N * C;
    T gn[16], an[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const long idx = NM ? ((long)n * 16 + m) * C + c : m * NC + (long)n * C + c;
        gn[m] = G[idx];
        an[m] = A[idx];
    }
    const T xn = X[(long)n * C + c];
    const T xbn = XB ? Xb[(long)n * C + c] : T(0);
    T xd_n = T(0);
    const int e1 = rptr[n + 1];
    for (int e = rptr[n] + grp; e < e1; e += ES) {
        const int ps_e = ps[e];
        if (ps_e & 1) continue;
        const long p = ps_e >> 1;
        const int kidx = sp_idx[p];
        if (kidx < 0) continue;
        const T u = sp_u[p];
        const long s = snd[e];
        const T xs = X[s * C + c];
        const T xbs = XB ? Xb[s * C + c] : T(0);
        T h00 = T(0), h10 = T(0), h11 = T(0), d00 = T(0), d10 = T(0), d11 = T(0);
        if constexpr (POLY_FWD == 0 || !POLY_BWD)
            mspl::hermite_rt<T>(u, sp_h, sp_invh, h00, h10, h11, d00, d10, d11);
        const T h01 = T(1) - h00;
        const T u2 = u * u, u3 = u2 * u;
        const T ui = u * sp_invh, u2i = u2 * sp_invh;
        const long k0 = (long)kidx * 4 * C + c, k1 = k0 + 4 * C;
        const long p16 = 16L * p;
        const long sbase = NM ? (s * 16) * C + c : s * C + c;
        T xd_s = T(0);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const T sgn2 = (l & 1) ? T(-1) : T(1);
            const T xns = sgn2 * xn;
            const T xbns = XB ? (sgn2 * xbn) : T(0);
            T tv = T(0), rv = T(0), ayn = T(0), ays = T(0);
            const int mbeg = l * l, mend = (l + 1) * (l + 1);
#pragma unroll
            for (int m = mbeg; m < mend; ++m) {
                const T ywv = YwP[p16 + m];
                const T gyv = GyP[p16 + m];
                const T yv = YP[p16 + m];
                const long sidx = NM ? sbase + (long)m * C : (long)m * NC + sbase;
                const T gs = G[sidx];
                const T as = A[sidx];
                const T wg = gn[m] * xs + gs * xns;
                tv += wg * ywv;
                rv += wg * gyv;
                if constexpr (XB) rv += (gn[m] * xbs + gs * xbns) * yv;
                ayn += an[m] * yv;
                ays += as * yv;
            }
            rv += ayn * xs + ays * xns;
            T rval;
            if constexpr (POLY_FWD > 0) {
                T c0, c1, c2, c3;
                mspl::ld_c4<T, POLY_FWD>(Rk, k0 + l * C, c0, c1, c2, c3);
                rval = ((c3 * u + c2) * u + c1) * u + c0;
            } else {
                const T r0 = Rk[k0 + l * C], r1 = Rk[k1 + l * C];
                const T t0 = Tk[k0 + l * C], t1 = Tk[k1 + l * C];
                rval = r1 + h00 * (r0 - r1) + h10 * t0 + h11 * t1;
            }
            xd_s += rval * ayn;
            xd_n += sgn2 * rval * ays;
            if constexpr (POLY_BWD) {
                const long base = ((long)kidx * 4 + l) * 4 * C + c;
                atomicAdd(gR_k + base + 0 * C, rv);
                atomicAdd(gR_k + base + 1 * C, u * rv + sp_invh * tv);
                atomicAdd(gR_k + base + 2 * C, u2 * rv + T(2) * ui * tv);
                atomicAdd(gR_k + base + 3 * C, u3 * rv + T(3) * u2i * tv);
            } else {
                atomicAdd(gR_k + k0 + l * C, h00 * rv + d00 * tv);
                atomicAdd(gR_k + k1 + l * C, h01 * rv - d00 * tv);
                atomicAdd(gT_k + k0 + l * C, h10 * rv + d10 * tv);
                atomicAdd(gT_k + k1 + l * C, h11 * rv + d11 * tv);
            }
        }
        atomicAdd(Xd + s * C + c, xd_s);
    }
    if (e1 > rptr[n]) atomicAdd(Xd + (long)n * C + c, xd_n);
}

// k_slot over the node's own pairs [pptr[n], pptr[n + 1]) (no skipped edges, no pidx loads)
template <typename T, bool XB>
__global__ void __launch_bounds__(256) k_slot_pp(const int* __restrict__ pptr, const int* __restrict__ snd,
                                                 const int* __restrict__ pe0, const int* __restrict__ pe1,
                                                 const T* __restrict__ G, const T* __restrict__ A,
                                                 const T* __restrict__ X, const T* __restrict__ Xb,
                                                 const T* __restrict__ Yw, const T* __restrict__ Gy,
                                                 const T* __restrict__ Y, T* __restrict__ Tb, T* __restrict__ Rb, int N,
                                                 int C) {
    const int n = blockIdx.x * (blockDim.x / C) + threadIdx.x / C;
    const int c = threadIdx.x % C;
    if (n >= N) return;
    const long NC = (long)N * C;
    T gn[16], an[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        gn[m] = G[m * NC + (long)n * C + c];
        an[m] = A[m * NC + (long)n * C + c];
    }
    const T xn = X[(long)n * C + c];
    const T xbn = XB ? Xb[(long)n * C + c] : T(0);
    const int p1 = pptr[n + 1];
    for (int p = pptr[n]; p < p1; ++p) {
        const int e = pe0[p];
        const int ep = pe1[p];
        const long s = snd[e];
        const T xs = X[s * C + c];
        const T xbs = XB ? Xb[s * C + c] : T(0);
        T tb[4] = {T(0), T(0), T(0), T(0)}, rb[4] = {T(0), T(0), T(0), T(0)};
        {
            T yw[16], gy[16], y[16];
            ld16(Yw + 16L * e, yw);
            ld16(Gy + 16L * e, gy);
            ld16(Y + 16L * e, y);
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                const int l = lofm(m);
                tb[l] += gn[m] * yw[m];
                rb[l] += gn[m] * (gy[m] * xs + y[m] * xbs) + an[m] * y[m] * xs;
            }
#pragma unroll
            for (int l = 0; l < 4; ++l) tb[l] *= xs;
        }
        {
            T yw[16], gy[16], y[16];
            ld16(Yw + 16L * ep, yw);
            ld16(Gy + 16L * ep, gy);
            ld16(Y + 16L * ep, y);
            T t2[4] = {T(0), T(0), T(0), T(0)};
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                const int l = lofm(m);
                const T gs = G[m * NC + s * C + c];
                const T as = A[m * NC + s * C + c];
                t2[l] += gs * yw[m];
                rb[l] += gs * (gy[m] * xn + y[m] * xbn) + as * y[m] * xn;
            }
#pragma unroll
            for (int l = 0; l < 4; ++l) tb[l] += t2[l] * xn;
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            Tb[(long)p * 4 * C + l * C + c] = tb[l];
            Rb[(long)p * 4 * C + l * C + c] = rb[l];
        }
    }
}

// Force pass, both layers, one warp per receiver node n (lanes over channels, G[:, n, :] in registers):
// s[e] = sum_t sum_m Y[e, m] sum_c G_t[m, n, c] T_t[p, l(m), c] X_t[s, c];
// gy[e, m] = sum_t sum_c G_t[m, n, c] R_t[p, l(m), c] X_t[s, c]; g[e] = s U + JY^T gy = dE/dvec[e];
// the virial sum_e vec (x) g of the node's edges is added to its graph.
// PASS 0: both layers. PASS 1: layer 0 only, its gy and s partials into gy, se. PASS 2: layer 1 plus those
// partials, then g and the virial (two passes stream half the radial weights each: L2 reuse of a pair's rows).
template <typename T, int CPL, int PASS = 0>
__global__ void __launch_bounds__(256) k_force_node(const int* __restrict__ rptr, const int* __restrict__ snd,
                                                    const int* __restrict__ pidx, const int* __restrict__ ngraph,
                                                    const T* __restrict__ Y, const T* __restrict__ U,
                                                    const T* __restrict__ JY, const T* __restrict__ vec,
                                                    const T* __restrict__ G0, const T* __restrict__ R0,
                                                    const T* __restrict__ T0, const T* __restrict__ X0,
                                                    const T* __restrict__ G1, const T* __restrict__ R1,
                                                    const T* __restrict__ T1, const T* __restrict__ X1, T* se, T* gy,
                                                    T* g, T* mv, int N, int C, int jyw = 0, int fes = 1,
                                                    const int* __restrict__ norder = nullptr) {
    const int wid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int nn = wid / fes, wsub = wid % fes;
    const int lane = threadIdx.x % 32;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    const long NC = (long)N * C;
    constexpr int t0 = PASS == 2 ? 1 : 0, t1 = PASS == 1 ? 1 : 2;
    T gn[t1 - t0][CPL][16];
#pragma unroll
    for (int k = 0; k < CPL; ++k)
#pragma unroll
        for (int m = 0; m < 16; ++m) {
#pragma unroll
            for (int t = t0; t < t1; ++t) gn[t - t0][k][m] = (t ? G1 : G0)[m * NC + (long)n * C + lane + 32 * k];
        }
    T v[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) v[k] = T(0);
    const int e1 = rptr[n + 1];
    for (int e = rptr[n] + wsub; e < e1; e += fes) {
        const long s = snd[e];
        const long p = pidx[e];
        T y[16];
        ld16(Y + 16L * e, y);
        T gp[16];
#pragma unroll
        for (int m = 0; m < 16; ++m) gp[m] = T(0);
        T sp = T(0);
#pragma unroll
        for (int t = t0; t < t1; ++t) {
            const T* R = t ? R1 : R0;
            const T* Tt = t ? T1 : T0;
            const T* X = t ? X1 : X0;
#pragma unroll
            for (int k = 0; k < CPL; ++k) {
                const int c = lane + 32 * k;
                const T x = X[s * C + c];
                T rr[4], tt[4];
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    rr[l] = R[p * 4 * C + l * C + c] * x;
                    tt[l] = Tt[p * 4 * C + l * C + c] * x;
                }
                const T* gg = gn[t - t0][k];
                gp[0] += gg[0] * rr[0];
                sp += y[0] * gg[0] * tt[0];
                T q1 = T(0), q2 = T(0), q3 = T(0);
#pragma unroll
                for (int m = 1; m < 4; ++m) {
                    gp[m] += gg[m] * rr[1];
                    q1 += y[m] * gg[m];
                }
#pragma unroll
                for (int m = 4; m < 9; ++m) {
                    gp[m] += gg[m] * rr[2];
                    q2 += y[m] * gg[m];
                }
#pragma unroll
                for (int m = 9; m < 16; ++m) {
                    gp[m] += gg[m] * rr[3];
                    q3 += y[m] * gg[m];
                }
                sp += q1 * tt[1] + q2 * tt[2] + q3 * tt[3];
            }
        }
        // warp sums: 16 + 1 values; a transposing butterfly leaves gp[m] summed on lane m (m < 16)
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        {
            // round 1 (offset 16): lanes < 16 keep m < 8 ... generic form: exchange halves
            T a8[8];
            const bool up = lane & 16;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                T send = up ? gp[j] : gp[j + 8];
                T keep = up ? gp[j + 8] : gp[j];
                a8[j] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
            }   // lane holds m = j + 8 * up
            T a4[4];
            const bool b8 = lane & 8;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                T send = b8 ? a8[j] : a8[j + 4];
                T keep = b8 ? a8[j + 4] : a8[j];
                a4[j] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
            }   // m = j + 4 b8 + 8 up
            T a2[2];
            const bool b4 = lane & 4;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                T send = b4 ? a4[j] : a4[j + 2];
                T keep = b4 ? a4[j + 2] : a4[j];
                a2[j] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
            }   // m = j + 2 b4 + 4 b8 + 8 up
            const bool b2 = lane & 2;
            T a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 2);
            // m = b2 + 2 b4 + 4 b8 + 8 up; sum the pair of lanes differing in bit 0
            a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
            const int mine = ((lane >> 1) & 1) + 2 * ((lane >> 2) & 1) + 4 * ((lane >> 3) & 1) + 8 * ((lane >> 4) & 1);
            // lanes 2k and 2k+1 both hold component mine; gather component m from lane 2 * idx(m)
            T gm = T(0);
            int src = 0;
            // lane index holding m: bit pattern (m & 1) -> bit 1, (m >> 1 & 1) -> bit 2, ...
            if (lane < 16) {
                const int m = lane;
                src = ((m & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (((m >> 3) & 1) << 4);
            }
            gm = __shfl_sync(0xffffffffu, a1, src);
            (void)mine;
            if constexpr (PASS == 1) {
                if (lane < 16) gy[16L * e + lane] = gm;
                if (lane == 0) se[e] = sp;
                continue;
            }
            if constexpr (PASS == 2) {
                gm += lane < 16 ? gy[16L * e + lane] : T(0);
                sp += se[e];
            }
            if (gy && lane < 16) gy[16L * e + lane] = gm;
            T g0, g1, g2;
            if (jyw) {
                // lane m < 16: gy_m times its 3 contiguous JY entries, summed over the 16 lanes
                T a0 = T(0), a1 = T(0), a2 = T(0);
                if (lane < 16) {
                    const T* j = JY + 48L * e + 3 * lane;
                    a0 = gm * j[0]; a1 = gm * j[1]; a2 = gm * j[2];
                }
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) {
                    a0 += __shfl_xor_sync(0xffffffffu, a0, o);
                    a1 += __shfl_xor_sync(0xffffffffu, a1, o);
                    a2 += __shfl_xor_sync(0xffffffffu, a2, o);
                }
                const T* u = U + 3L * e;
                g0 = __shfl_sync(0xffffffffu, a0, 0) + sp * u[0];
                g1 = __shfl_sync(0xffffffffu, a1, 0) + sp * u[1];
                g2 = __shfl_sync(0xffffffffu, a2, 0) + sp * u[2];
                if (lane < 3) g[3L * e + lane] = lane == 0 ? g0 : lane == 1 ? g1 : g2;
            } else {
                // g = s U + JY^T gy (lanes 0..2 one Cartesian component each, gy from lanes 0..15)
                T gk = T(0);
#pragma unroll
                for (int m = 0; m < 16; ++m) {
                    const T gym = __shfl_sync(0xffffffffu, gm, m);
                    if (lane < 3) gk += gym * JY[48L * e + 3 * m + lane];
                }
                if (lane < 3) {
                    gk += sp * U[3 * e + lane];
                    g[3L * e + lane] = gk;
                }
                g0 = __shfl_sync(0xffffffffu, gk, 0);
                g1 = __shfl_sync(0xffffffffu, gk, 1);
                g2 = __shfl_sync(0xffffffffu, gk, 2);
            }
            if (se && lane == 0) se[e] = sp;
            if (lane == 0) {
                const T vx = vec[3L * e], vy = vec[3L * e + 1], vz = vec[3L * e + 2];
                v[0] += vx * g0; v[1] += vx * g1; v[2] += vx * g2;
                v[3] += vy * g0; v[4] += vy * g1; v[5] += vy * g2;
                v[6] += vz * g0; v[7] += vz * g1; v[8] += vz * g2;
            }
        }
    }
    if (PASS != 1 && lane == 0 && e1 > rptr[n]) {
        T* o = mv + 9L * ngraph[n];
#pragma unroll
        for (int k = 0; k < 9; ++k) atomicAdd(o + k, v[k]);
    }
}

// Pair-parity, JY-free k_force_node (LST_PAIR_BASIS=1): reads YP[p] and UP[p] (16 B/pair instead of 204 B/edge of U+JY)
// and computes JY^T gy in registers on lane 0 via sh16_jyt. NM=true: node-major [N, 16, C] G0/G1 (LST_NODE_MAJOR=1).
template <typename T, int CPL, bool NM = false>
__global__ void __launch_bounds__(256) k_force_node_pair(const int* __restrict__ rptr, const int* __restrict__ snd,
                                                          const int* __restrict__ ps, const int* __restrict__ ngraph,
                                                          const T* __restrict__ YP, const T* __restrict__ UP,
                                                          const T* __restrict__ vec, const T* __restrict__ G0,
                                                          const T* __restrict__ R0, const T* __restrict__ T0,
                                                          const T* __restrict__ X0, const T* __restrict__ G1,
                                                          const T* __restrict__ R1, const T* __restrict__ T1,
                                                          const T* __restrict__ X1, T* g, T* mv, int N, int C,
                                                          int fes = 1, const int* __restrict__ norder = nullptr) {
    const int wid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int nn = wid / fes, wsub = wid % fes;
    const int lane = threadIdx.x % 32;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    const long NC = (long)N * C;
    T gn[2][CPL][16];
#pragma unroll
    for (int k = 0; k < CPL; ++k)
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const long idx = NM ? ((long)n * 16 + m) * C + lane + 32 * k : m * NC + (long)n * C + lane + 32 * k;
            gn[0][k][m] = G0[idx];
            gn[1][k][m] = G1[idx];
        }
    T v[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) v[k] = T(0);
    const int e1 = rptr[n + 1];
    for (int e = rptr[n] + wsub; e < e1; e += fes) {
        const long s = snd[e];
        const int ps_e = ps[e];
        const long p = ps_e >> 1;
        const T sgn = (ps_e & 1) ? T(-1) : T(1);
        T y[16];
        ld16(YP + 16L * p, y);
        T gp[16];
#pragma unroll
        for (int m = 0; m < 16; ++m) gp[m] = T(0);
        T sp = T(0);
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const T* R = t ? R1 : R0;
            const T* Tt = t ? T1 : T0;
            const T* X = t ? X1 : X0;
#pragma unroll
            for (int k = 0; k < CPL; ++k) {
                const int c = lane + 32 * k;
                const T x = X[s * C + c];
                T rr[4], tt[4];
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    rr[l] = R[p * 4 * C + l * C + c] * x;
                    tt[l] = Tt[p * 4 * C + l * C + c] * x;
                }
                const T* gg = gn[t][k];
                gp[0] += gg[0] * rr[0];
                sp += y[0] * gg[0] * tt[0];
                T q1 = T(0), q2 = T(0), q3 = T(0);
#pragma unroll
                for (int m = 1; m < 4; ++m) {
                    gp[m] += gg[m] * rr[1];
                    q1 += y[m] * gg[m];
                }
#pragma unroll
                for (int m = 4; m < 9; ++m) {
                    gp[m] += gg[m] * rr[2];
                    q2 += y[m] * gg[m];
                }
#pragma unroll
                for (int m = 9; m < 16; ++m) {
                    gp[m] += gg[m] * rr[3];
                    q3 += y[m] * gg[m];
                }
                sp += sgn * q1 * tt[1] + q2 * tt[2] + sgn * q3 * tt[3];
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        {
            T a8[8];
            const bool up = lane & 16;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                T send = up ? gp[j] : gp[j + 8];
                T keep = up ? gp[j + 8] : gp[j];
                a8[j] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
            }
            T a4[4];
            const bool b8 = lane & 8;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                T send = b8 ? a8[j] : a8[j + 4];
                T keep = b8 ? a8[j + 4] : a8[j];
                a4[j] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
            }
            T a2[2];
            const bool b4 = lane & 4;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                T send = b4 ? a4[j] : a4[j + 2];
                T keep = b4 ? a4[j + 2] : a4[j];
                a2[j] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
            }
            const bool b2 = lane & 2;
            T a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 2);
            a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
            int src = 0;
            if (lane < 16) {
                const int m = lane;
                src = ((m & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (((m >> 3) & 1) << 4);
            }
            const T gm = __shfl_sync(0xffffffffu, a1, src);
#pragma unroll
            for (int m = 1; m < 16; ++m) gp[m] = __shfl_sync(0xffffffffu, gm, m);
            if (lane == 0) {
                T ux, uy, uz, rinv;
                if constexpr (std::is_same<T, float>::value) {
                    const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
                    ux = up4.x; uy = up4.y; uz = up4.z; rinv = up4.w;
                } else {
                    ux = UP[4L * p]; uy = UP[4L * p + 1]; uz = UP[4L * p + 2]; rinv = UP[4L * p + 3];
                }
                const T eux = sgn * ux, euy = sgn * uy, euz = sgn * uz;
                T jx, jy_, jz;
                sh16_jyt<T>(eux, euy, euz, rinv, sgn, y, gp, jx, jy_, jz);
                const T g0 = jx + sp * eux, g1 = jy_ + sp * euy, g2 = jz + sp * euz;
                g[3L * e] = g0;
                g[3L * e + 1] = g1;
                g[3L * e + 2] = g2;
                const T vx = vec[3L * e], vy = vec[3L * e + 1], vz = vec[3L * e + 2];
                v[0] += vx * g0; v[1] += vx * g1; v[2] += vx * g2;
                v[3] += vy * g0; v[4] += vy * g1; v[5] += vy * g2;
                v[6] += vz * g0; v[7] += vz * g1; v[8] += vz * g2;
            }
        }
    }
    if (lane == 0 && e1 > rptr[n]) {
        T* o = mv + 9L * ngraph[n];
#pragma unroll
        for (int k = 0; k < 9; ++k) atomicAdd(o + k, v[k]);
    }
}

// Fused cubic-Hermite spline k_force_node_pair (LST_MLP_SPLINE_FUSED=1):
//   POLY=1: reads packed float4 cubic coefficients C_k[k, l, c]; POLY=2: reads packed half4 (uint2) C_kh!
template <typename T, int CPL, bool NM = false, int POLY = 0>
__global__ void __launch_bounds__(256) k_force_node_pair_sp(
    const int* __restrict__ rptr, const int* __restrict__ snd, const int* __restrict__ ps,
    const int* __restrict__ sp_idx, const T* __restrict__ sp_u, const int* __restrict__ ngraph,
    const T* __restrict__ YP, const T* __restrict__ UP, const T* __restrict__ vec, const T* __restrict__ G0,
    const T* __restrict__ Rk0, const T* __restrict__ Tk0, const T* __restrict__ X0, const T* __restrict__ G1,
    const T* __restrict__ Rk1, const T* __restrict__ Tk1, const T* __restrict__ X1, T* g, T* mv, T sp_h, T sp_invh,
    int N, int C, int fes = 1, const int* __restrict__ norder = nullptr) {
    const int wid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int nn = wid / fes, wsub = wid % fes;
    const int lane = threadIdx.x % 32;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    const long NC = (long)N * C;
    T gn[2][CPL][16];
#pragma unroll
    for (int k = 0; k < CPL; ++k)
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const long idx = NM ? ((long)n * 16 + m) * C + lane + 32 * k : m * NC + (long)n * C + lane + 32 * k;
            gn[0][k][m] = G0[idx];
            gn[1][k][m] = G1[idx];
        }
    T v[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) v[k] = T(0);
    const int e1 = rptr[n + 1];
    for (int e = rptr[n] + wsub; e < e1; e += fes) {
        const int ps_e = ps[e];
        const long p = ps_e >> 1;
        const int kidx = sp_idx[p];
        if (kidx < 0) {
            if (lane == 0) {
                g[3L * e] = T(0);
                g[3L * e + 1] = T(0);
                g[3L * e + 2] = T(0);
            }
            continue;
        }
        const long s = snd[e];
        const T sgn = (ps_e & 1) ? T(-1) : T(1);
        const T u = sp_u[p];
        T h00 = T(0), h10 = T(0), h11 = T(0), d00 = T(0), d10 = T(0), d11 = T(0);
        if constexpr (POLY == 0) mspl::hermite_rt<T>(u, sp_h, sp_invh, h00, h10, h11, d00, d10, d11);
        T y[16];
        ld16(YP + 16L * p, y);
        T gp[16];
#pragma unroll
        for (int m = 0; m < 16; ++m) gp[m] = T(0);
        T sp = T(0);
        const long kbase0 = (long)kidx * 4 * C;
        const long kbase1 = kbase0 + 4 * C;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const T* Rk = t ? Rk1 : Rk0;
            const T* Tk = t ? Tk1 : Tk0;
            const T* X = t ? X1 : X0;
#pragma unroll
            for (int k = 0; k < CPL; ++k) {
                const int c = lane + 32 * k;
                const T x = X[s * C + c];
                T rr[4], tt[4];
                if constexpr (POLY > 0) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        T c0, c1, c2, c3;
                        mspl::ld_c4<T, POLY>(Rk, kbase0 + l * C + c, c0, c1, c2, c3);
                        rr[l] = (((c3 * u + c2) * u + c1) * u + c0) * x;
                        tt[l] = (((T(3) * c3 * u + T(2) * c2) * u + c1) * sp_invh) * x;
                    }
                } else {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const long off0 = kbase0 + l * C + c, off1 = kbase1 + l * C + c;
                        const T r0 = Rk[off0], r1 = Rk[off1];
                        const T t0 = Tk[off0], t1 = Tk[off1];
                        const T dr = r0 - r1;
                        rr[l] = (r1 + h00 * dr + h10 * t0 + h11 * t1) * x;
                        tt[l] = (d00 * dr + d10 * t0 + d11 * t1) * x;
                    }
                }
                const T* gg = gn[t][k];
                gp[0] += gg[0] * rr[0];
                sp += y[0] * gg[0] * tt[0];
                T q1 = T(0), q2 = T(0), q3 = T(0);
#pragma unroll
                for (int m = 1; m < 4; ++m) {
                    gp[m] += gg[m] * rr[1];
                    q1 += y[m] * gg[m];
                }
#pragma unroll
                for (int m = 4; m < 9; ++m) {
                    gp[m] += gg[m] * rr[2];
                    q2 += y[m] * gg[m];
                }
#pragma unroll
                for (int m = 9; m < 16; ++m) {
                    gp[m] += gg[m] * rr[3];
                    q3 += y[m] * gg[m];
                }
                sp += sgn * q1 * tt[1] + q2 * tt[2] + sgn * q3 * tt[3];
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        {
            T a8[8];
            const bool up = lane & 16;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                T send = up ? gp[j] : gp[j + 8];
                T keep = up ? gp[j + 8] : gp[j];
                a8[j] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
            }
            T a4[4];
            const bool b8 = lane & 8;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                T send = b8 ? a8[j] : a8[j + 4];
                T keep = b8 ? a8[j + 4] : a8[j];
                a4[j] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
            }
            T a2[2];
            const bool b4 = lane & 4;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                T send = b4 ? a4[j] : a4[j + 2];
                T keep = b4 ? a4[j + 2] : a4[j];
                a2[j] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
            }
            const bool b2 = lane & 2;
            T a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 2);
            a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
            int src = 0;
            if (lane < 16) {
                const int m = lane;
                src = ((m & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (((m >> 3) & 1) << 4);
            }
            const T gm = __shfl_sync(0xffffffffu, a1, src);
#pragma unroll
            for (int m = 1; m < 16; ++m) gp[m] = __shfl_sync(0xffffffffu, gm, m);
            if (lane == 0) {
                T ux, uy, uz, rinv;
                if constexpr (std::is_same<T, float>::value) {
                    const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
                    ux = up4.x; uy = up4.y; uz = up4.z; rinv = up4.w;
                } else {
                    ux = UP[4L * p]; uy = UP[4L * p + 1]; uz = UP[4L * p + 2]; rinv = UP[4L * p + 3];
                }
                const T eux = sgn * ux, euy = sgn * uy, euz = sgn * uz;
                T jx, jy_, jz;
                sh16_jyt<T>(eux, euy, euz, rinv, sgn, y, gp, jx, jy_, jz);
                const T g0 = jx + sp * eux, g1 = jy_ + sp * euy, g2 = jz + sp * euz;
                g[3L * e] = g0;
                g[3L * e + 1] = g1;
                g[3L * e + 2] = g2;
                const T vx = vec[3L * e], vy = vec[3L * e + 1], vz = vec[3L * e + 2];
                v[0] += vx * g0; v[1] += vx * g1; v[2] += vx * g2;
                v[3] += vy * g0; v[4] += vy * g1; v[5] += vy * g2;
                v[6] += vz * g0; v[7] += vz * g1; v[8] += vz * g2;
            }
        }
    }
    if (lane == 0 && e1 > rptr[n]) {
        T* o = mv + 9L * ngraph[n];
#pragma unroll
        for (int k = 0; k < 9; ++k) atomicAdd(o + k, v[k]);
    }
}

// 64-thread (2-warp, CPL=1) k_force_node_pair_sp (LST_FORCE_64T=1): halves register pressure (32 regs for gn
// instead of 64 -> higher occupancy, zero spills!). Grid: N * fes blocks of 64 threads (c = threadIdx.x in 0..63).
template <typename T, bool NM = true, int POLY = 0>
__global__ void __launch_bounds__(64) k_force_node_pair_sp64(
    const int* __restrict__ rptr, const int* __restrict__ snd, const int* __restrict__ ps,
    const int* __restrict__ sp_idx, const T* __restrict__ sp_u, const int* __restrict__ ngraph,
    const T* __restrict__ YP, const T* __restrict__ UP, const T* __restrict__ vec, const T* __restrict__ G0,
    const T* __restrict__ Rk0, const T* __restrict__ Tk0, const T* __restrict__ X0, const T* __restrict__ G1,
    const T* __restrict__ Rk1, const T* __restrict__ Tk1, const T* __restrict__ X1, T* g, T* mv, T sp_h, T sp_invh,
    int N, int C, int fes = 2, const int* __restrict__ norder = nullptr) {
    const int nn = blockIdx.x / fes, wsub = blockIdx.x % fes;
    const int c = threadIdx.x;
    const int warp = c >> 5, lane = c & 31;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    const long NC = (long)N * C;
    T gn0[16], gn1[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const long idx = NM ? ((long)n * 16 + m) * C + c : m * NC + (long)n * C + c;
        gn0[m] = G0[idx];
        gn1[m] = G1[idx];
    }
    __shared__ T sw[17];
    T v[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) v[k] = T(0);
    const int e1 = rptr[n + 1];
    for (int e = rptr[n] + wsub; e < e1; e += fes) {
        const int ps_e = ps[e];
        const long p = ps_e >> 1;
        const int kidx = sp_idx[p];
        if (kidx < 0) {
            if (c == 0) {
                g[3L * e] = T(0);
                g[3L * e + 1] = T(0);
                g[3L * e + 2] = T(0);
            }
            continue;
        }
        const long s = snd[e];
        const T sgn = (ps_e & 1) ? T(-1) : T(1);
        const T u = sp_u[p];
        T h00 = T(0), h10 = T(0), h11 = T(0), d00 = T(0), d10 = T(0), d11 = T(0);
        if constexpr (POLY == 0) mspl::hermite_rt<T>(u, sp_h, sp_invh, h00, h10, h11, d00, d10, d11);
        T y[16];
        ld16(YP + 16L * p, y);
        T gp[16];
#pragma unroll
        for (int m = 0; m < 16; ++m) gp[m] = T(0);
        T sp = T(0);
        const long kbase0 = (long)kidx * 4 * C + c;
        const long kbase1 = kbase0 + 4 * C;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const T* Rk = t ? Rk1 : Rk0;
            const T* Tk = t ? Tk1 : Tk0;
            const T x = (t ? X1 : X0)[s * C + c];
            const T* gg = t ? gn1 : gn0;
            T rr[4], tt[4];
            if constexpr (POLY > 0) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    T c0, c1, c2, c3;
                    mspl::ld_c4<T, POLY>(Rk, kbase0 + l * C, c0, c1, c2, c3);
                    rr[l] = (((c3 * u + c2) * u + c1) * u + c0) * x;
                    tt[l] = (((T(3) * c3 * u + T(2) * c2) * u + c1) * sp_invh) * x;
                }
            } else {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const T r0 = Rk[kbase0 + l * C], r1 = Rk[kbase1 + l * C];
                    const T t0 = Tk[kbase0 + l * C], t1 = Tk[kbase1 + l * C];
                    const T dr = r0 - r1;
                    rr[l] = (r1 + h00 * dr + h10 * t0 + h11 * t1) * x;
                    tt[l] = (d00 * dr + d10 * t0 + d11 * t1) * x;
                }
            }
            gp[0] += gg[0] * rr[0];
            sp += y[0] * gg[0] * tt[0];
            T q1 = T(0), q2 = T(0), q3 = T(0);
#pragma unroll
            for (int m = 1; m < 4; ++m) { gp[m] += gg[m] * rr[1]; q1 += y[m] * gg[m]; }
#pragma unroll
            for (int m = 4; m < 9; ++m) { gp[m] += gg[m] * rr[2]; q2 += y[m] * gg[m]; }
#pragma unroll
            for (int m = 9; m < 16; ++m) { gp[m] += gg[m] * rr[3]; q3 += y[m] * gg[m]; }
            sp += sgn * q1 * tt[1] + q2 * tt[2] + sgn * q3 * tt[3];
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        T a8[8];
        const bool up = lane & 16;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            a8[j] = (up ? gp[j + 8] : gp[j]) + __shfl_xor_sync(0xffffffffu, up ? gp[j] : gp[j + 8], 16);
        T a4[4];
        const bool b8 = lane & 8;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a4[j] = (b8 ? a8[j + 4] : a8[j]) + __shfl_xor_sync(0xffffffffu, b8 ? a8[j] : a8[j + 4], 8);
        T a2[2];
        const bool b4 = lane & 4;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            a2[j] = (b4 ? a4[j + 2] : a4[j]) + __shfl_xor_sync(0xffffffffu, b4 ? a4[j] : a4[j + 2], 4);
        const bool b2 = lane & 2;
        T a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 2);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        int src = 0;
        if (lane < 16) {
            const int m = lane;
            src = ((m & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (((m >> 3) & 1) << 4);
        }
        T gm = __shfl_sync(0xffffffffu, a1, src);
        if (warp == 1) {
            if (lane < 16) sw[lane] = gm;
            if (lane == 0) sw[16] = sp;
        }
        __syncthreads();
        if (warp == 0) {
            if (lane < 16) gm += sw[lane];
            if (lane == 0) sp += sw[16];
#pragma unroll
            for (int m = 1; m < 16; ++m) gp[m] = __shfl_sync(0xffffffffu, gm, m);
            if (lane == 0) {
                T ux, uy, uz, rinv;
                if constexpr (std::is_same<T, float>::value) {
                    const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
                    ux = up4.x; uy = up4.y; uz = up4.z; rinv = up4.w;
                } else {
                    ux = UP[4L * p]; uy = UP[4L * p + 1]; uz = UP[4L * p + 2]; rinv = UP[4L * p + 3];
                }
                const T eux = sgn * ux, euy = sgn * uy, euz = sgn * uz;
                T jx, jy_, jz;
                sh16_jyt<T>(eux, euy, euz, rinv, sgn, y, gp, jx, jy_, jz);
                const T g0 = jx + sp * eux, g1 = jy_ + sp * euy, g2 = jz + sp * euz;
                g[3L * e] = g0;
                g[3L * e + 1] = g1;
                g[3L * e + 2] = g2;
                const T vx = vec[3L * e], vy = vec[3L * e + 1], vz = vec[3L * e + 2];
                v[0] += vx * g0; v[1] += vx * g1; v[2] += vx * g2;
                v[3] += vy * g0; v[4] += vy * g1; v[5] += vy * g2;
                v[6] += vz * g0; v[7] += vz * g1; v[8] += vz * g2;
            }
        }
    }
    if (c == 0 && e1 > rptr[n]) {
        T* o = mv + 9L * ngraph[n];
#pragma unroll
        for (int k = 0; k < 9; ++k) atomicAdd(o + k, v[k]);
    }
}

// Newton's-3rd-law pair-owned force kernel (LST_FORCE_PAIR=1): visits ONLY canonical half of edges (ps_e & 1 == 0),
// computes g_pair = g[e0] - g[e1] ONCE per pair using exact spherical-harmonic parity (-1)^l, accumulates -g_pair into
// F[n], atomic-adds +g_pair into F[s] (F pre-initialized to core_f in k_gather) and vec[e0] (x) g_pair into mv.
// Halves spline C_kh loads, YP/UP/vec loads, warp reductions, and sh16_jyt calls, and deletes g[E,3] + k_forces_pair_w!
template <typename T, int POLY = 0, typename A16 = T>
__global__ void __launch_bounds__(64) k_force_pair_sp64(
    const int* __restrict__ rptr, const int* __restrict__ snd, const int* __restrict__ ps,
    const int* __restrict__ sp_idx, const T* __restrict__ sp_u, const int* __restrict__ ngraph,
    const T* __restrict__ YP, const T* __restrict__ UP, const T* __restrict__ vec, const A16* __restrict__ G0,
    const T* __restrict__ Rk0, const T* __restrict__ Tk0, const T* __restrict__ X0, const A16* __restrict__ G1,
    const T* __restrict__ Rk1, const T* __restrict__ Tk1, const T* __restrict__ X1, T* F, T* mv, T sp_h, T sp_invh,
    int N, int C, int fes = 2, const int* __restrict__ norder = nullptr) {
    const int nn = blockIdx.x / fes, wsub = blockIdx.x % fes;
    const int c = threadIdx.x;
    const int warp = c >> 5, lane = c & 31;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    T gn0[16], gn1[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const long idx = ((long)n * 16 + m) * C + c;
        gn0[m] = ah::ld_a16<T>(G0, idx);
        gn1[m] = ah::ld_a16<T>(G1, idx);
    }
    const T xn0 = X0[(long)n * C + c], xn1 = X1[(long)n * C + c];
    __shared__ T sw[17];
    T v[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) v[k] = T(0);
    T fn0 = T(0), fn1 = T(0), fn2 = T(0);
    bool any_pair = false;
    const int e1 = rptr[n + 1];
    for (int e = rptr[n] + wsub; e < e1; e += fes) {
        const int ps_e = ps[e];
        if (ps_e & 1) continue;
        const long p = ps_e >> 1;
        const int kidx = sp_idx[p];
        if (kidx < 0) continue;
        any_pair = true;
        const long s = snd[e];
        const T u = sp_u[p];
        T h00 = T(0), h10 = T(0), h11 = T(0), d00 = T(0), d10 = T(0), d11 = T(0);
        if constexpr (POLY == 0) mspl::hermite_rt<T>(u, sp_h, sp_invh, h00, h10, h11, d00, d10, d11);
        T y[16];
        ld16(YP + 16L * p, y);
        T gp[16];
#pragma unroll
        for (int m = 0; m < 16; ++m) gp[m] = T(0);
        T sp = T(0);
        const long kbase0 = (long)kidx * 4 * C + c;
        const long kbase1 = kbase0 + 4 * C;
        const long sbase = (s * 16) * C + c;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const T* Rk = t ? Rk1 : Rk0;
            const T* Tk = t ? Tk1 : Tk0;
            const A16* Gs = t ? G1 : G0;
            const T xs = (t ? X1 : X0)[s * C + c];
            const T xn = t ? xn1 : xn0;
            const T* gg = t ? gn1 : gn0;
            T rr[4], tt[4];
            if constexpr (POLY > 0) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    T c0, c1, c2, c3;
                    mspl::ld_c4<T, POLY>(Rk, kbase0 + l * C, c0, c1, c2, c3);
                    rr[l] = ((c3 * u + c2) * u + c1) * u + c0;
                    tt[l] = ((T(3) * c3 * u + T(2) * c2) * u + c1) * sp_invh;
                }
            } else {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const T r0 = Rk[kbase0 + l * C], r1 = Rk[kbase1 + l * C];
                    const T t0 = Tk[kbase0 + l * C], t1 = Tk[kbase1 + l * C];
                    const T dr = r0 - r1;
                    rr[l] = r1 + h00 * dr + h10 * t0 + h11 * t1;
                    tt[l] = d00 * dr + d10 * t0 + d11 * t1;
                }
            }
            {
                const T w0 = gg[0] * xs + ah::ld_a16<T>(Gs, sbase) * xn;
                gp[0] += w0 * rr[0];
                sp += y[0] * w0 * tt[0];
            }
            T q1 = T(0), q2 = T(0), q3 = T(0);
#pragma unroll
            for (int m = 1; m < 4; ++m) {
                const T w = gg[m] * xs - ah::ld_a16<T>(Gs, sbase + (long)m * C) * xn;
                gp[m] += w * rr[1];
                q1 += y[m] * w;
            }
#pragma unroll
            for (int m = 4; m < 9; ++m) {
                const T w = gg[m] * xs + ah::ld_a16<T>(Gs, sbase + (long)m * C) * xn;
                gp[m] += w * rr[2];
                q2 += y[m] * w;
            }
#pragma unroll
            for (int m = 9; m < 16; ++m) {
                const T w = gg[m] * xs - ah::ld_a16<T>(Gs, sbase + (long)m * C) * xn;
                gp[m] += w * rr[3];
                q3 += y[m] * w;
            }
            sp += q1 * tt[1] + q2 * tt[2] + q3 * tt[3];
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        T a8[8];
        const bool up = lane & 16;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            a8[j] = (up ? gp[j + 8] : gp[j]) + __shfl_xor_sync(0xffffffffu, up ? gp[j] : gp[j + 8], 16);
        T a4[4];
        const bool b8 = lane & 8;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a4[j] = (b8 ? a8[j + 4] : a8[j]) + __shfl_xor_sync(0xffffffffu, b8 ? a8[j] : a8[j + 4], 8);
        T a2[2];
        const bool b4 = lane & 4;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            a2[j] = (b4 ? a4[j + 2] : a4[j]) + __shfl_xor_sync(0xffffffffu, b4 ? a4[j] : a4[j + 2], 4);
        const bool b2 = lane & 2;
        T a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 2);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        int src = 0;
        if (lane < 16) {
            const int m = lane;
            src = ((m & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (((m >> 3) & 1) << 4);
        }
        T gm = __shfl_sync(0xffffffffu, a1, src);
        if (warp == 1) {
            if (lane < 16) sw[lane] = gm;
            if (lane == 0) sw[16] = sp;
        }
        __syncthreads();
        if (warp == 0) {
            if (lane < 16) gm += sw[lane];
            if (lane == 0) sp += sw[16];
#pragma unroll
            for (int m = 1; m < 16; ++m) gp[m] = __shfl_sync(0xffffffffu, gm, m);
            if (lane == 0) {
                T ux, uy, uz, rinv;
                if constexpr (std::is_same<T, float>::value) {
                    const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
                    ux = up4.x; uy = up4.y; uz = up4.z; rinv = up4.w;
                } else {
                    ux = UP[4L * p]; uy = UP[4L * p + 1]; uz = UP[4L * p + 2]; rinv = UP[4L * p + 3];
                }
                T jx, jy_, jz;
                sh16_jyt<T>(ux, uy, uz, rinv, T(1), y, gp, jx, jy_, jz);
                const T g0 = jx + sp * ux, g1 = jy_ + sp * uy, g2 = jz + sp * uz;
                fn0 -= g0; fn1 -= g1; fn2 -= g2;
                atomicAdd(F + 3L * s + 0, g0);
                atomicAdd(F + 3L * s + 1, g1);
                atomicAdd(F + 3L * s + 2, g2);
                const T vx = vec[3L * e], vy = vec[3L * e + 1], vz = vec[3L * e + 2];
                v[0] += vx * g0; v[1] += vx * g1; v[2] += vx * g2;
                v[3] += vy * g0; v[4] += vy * g1; v[5] += vy * g2;
                v[6] += vz * g0; v[7] += vz * g1; v[8] += vz * g2;
            }
        }
    }
    if (c == 0 && any_pair) {
        atomicAdd(F + 3L * n + 0, fn0);
        atomicAdd(F + 3L * n + 1, fn1);
        atomicAdd(F + 3L * n + 2, fn2);
        T* o = mv + 9L * ngraph[n];
#pragma unroll
        for (int k = 0; k < 9; ++k) atomicAdd(o + k, v[k]);
    }
}

// Knot-bucketed Newton's-3rd-law force kernel (LST_FORCE_KB=1):
// Grid: dim3(max_tiles_per_frame, B), 64 threads (c = threadIdx.x in 0..63, 2 warps/block).
// Loads C_kh0[kidx] and C_kh1[kidx] ONCE per tile (26x-51x fewer C_kh loads!), drops gn0[16]/gn1[16]/y[16]/v[9]/fn[3]
// from registers (higher occupancy!), skips 0% wasted ps_e&1 branches, and uses double-buffered sw[2][17] for 1 sync/pair.
template <typename T, int POLY = 2, typename A16 = T>
__global__ void __launch_bounds__(64) k_force_kb(
    const int4* __restrict__ f_meta, const int* __restrict__ toff, const int4* __restrict__ tiles,
    const int2* __restrict__ puv, const T* __restrict__ sp_u, const T* __restrict__ YP, const T* __restrict__ UP,
    const A16* __restrict__ G0, const T* __restrict__ Rk0, const T* __restrict__ Tk0, const T* __restrict__ X0,
    const A16* __restrict__ G1, const T* __restrict__ Rk1, const T* __restrict__ Tk1, const T* __restrict__ X1,
    T* F, T* mv, T sp_h, T sp_invh, int C) {
    const int bi = blockIdx.y;
    const int4 fm = f_meta[bi];
    const int fid = fm.x;
    if (fid < 0) return;
    const int t0 = toff[fid], t1 = toff[fid + 1];
    const int ti = t0 + (int)blockIdx.x;
    if (ti >= t1) return;
    const int4 tl = tiles[ti];
    const int kidx = tl.x;
    const long p_beg = (long)fm.z + tl.y, p_end = (long)fm.z + tl.z;
    const int c = threadIdx.x;
    const int warp = c >> 5, lane = c & 31;

    uint2 ckh0[4], ckh1[4];
    T c00[4], c01[4], c02[4], c03[4], c10[4], c11[4], c12[4], c13[4];
    const long kbase0 = (long)kidx * 4 * C + c;
    const long kbase1 = kbase0 + 4 * C;
    if constexpr (POLY == 2 && std::is_same<T, float>::value) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            ckh0[l] = reinterpret_cast<const uint2*>(Rk0)[kbase0 + l * C];
            ckh1[l] = reinterpret_cast<const uint2*>(Rk1)[kbase0 + l * C];
        }
    } else if constexpr (POLY > 0) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            mspl::ld_c4<T, POLY>(Rk0, kbase0 + l * C, c00[l], c01[l], c02[l], c03[l]);
            mspl::ld_c4<T, POLY>(Rk1, kbase0 + l * C, c10[l], c11[l], c12[l], c13[l]);
        }
    } else {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const T r0 = Rk0[kbase0 + l * C], r1 = Rk0[kbase1 + l * C], t0v = Tk0[kbase0 + l * C], t1v = Tk0[kbase1 + l * C];
            const T dr = r1 - r0;
            c00[l] = r0; c01[l] = sp_h * t0v; c02[l] = T(3) * dr - sp_h * (T(2) * t0v + t1v); c03[l] = T(-2) * dr + sp_h * (t0v + t1v);
            const T s0 = Rk1[kbase0 + l * C], s1 = Rk1[kbase1 + l * C], u0v = Tk1[kbase0 + l * C], u1v = Tk1[kbase1 + l * C];
            const T ds = s1 - s0;
            c10[l] = s0; c11[l] = sp_h * u0v; c12[l] = T(3) * ds - sp_h * (T(2) * u0v + u1v); c13[l] = T(-2) * ds + sp_h * (u0v + u1v);
        }
    }

    __shared__ T sw[2][17];
    __shared__ T sv[12];  // sv[0..8] = virial, sv[9..11] = fu for cur_u (c == 0 only -> saves 12 regs/thread!)
    if (c < 12) sv[c] = T(0);
    int cur_u = -1;
    int sb = 0;

    for (long p = p_beg; p < p_end; ++p, sb ^= 1) {
        const int2 uv = puv[p];
        const int u = uv.x, v = uv.y;
        const T us = sp_u[p];
        T rr0[4], tt0[4], rr1[4], tt1[4];
        if constexpr (POLY == 2 && std::is_same<T, float>::value) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&ckh0[l].x));
                const float2 b0 = __half22float2(*reinterpret_cast<const __half2*>(&ckh0[l].y));
                rr0[l] = ((b0.y * us + b0.x) * us + a0.y) * us + a0.x;
                tt0[l] = ((3.f * b0.y * us + 2.f * b0.x) * us + a0.y) * sp_invh;
                const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(&ckh1[l].x));
                const float2 b1 = __half22float2(*reinterpret_cast<const __half2*>(&ckh1[l].y));
                rr1[l] = ((b1.y * us + b1.x) * us + a1.y) * us + a1.x;
                tt1[l] = ((3.f * b1.y * us + 2.f * b1.x) * us + a1.y) * sp_invh;
            }
        } else {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                rr0[l] = ((c03[l] * us + c02[l]) * us + c01[l]) * us + c00[l];
                tt0[l] = ((T(3) * c03[l] * us + T(2) * c02[l]) * us + c01[l]) * sp_invh;
                rr1[l] = ((c13[l] * us + c12[l]) * us + c11[l]) * us + c10[l];
                tt1[l] = ((T(3) * c13[l] * us + T(2) * c12[l]) * us + c11[l]) * sp_invh;
            }
        }
        const T xu0 = X0[(long)u * C + c], xv0 = X0[(long)v * C + c];
        const T xu1 = X1[(long)u * C + c], xv1 = X1[(long)v * C + c];
        const long ubase = (long)u * 16 * C + c;
        const long vbase = (long)v * 16 * C + c;
        T gp[16];
        T sp = T(0);
#pragma unroll
        for (int m0 = 0; m0 < 16; m0 += 4) {
            T y4[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 vy = *reinterpret_cast<const float4*>(YP + 16L * p + m0);
                y4[0] = vy.x; y4[1] = vy.y; y4[2] = vy.z; y4[3] = vy.w;
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) y4[j] = YP[16L * p + m0 + j];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int m = m0 + j;
                const int l = lofm(m);
                const T sgn2 = (l & 1) ? T(-1) : T(1);
                const T w0 = ah::ld_a16<T>(G0, ubase + (long)m * C) * xv0 + sgn2 * ah::ld_a16<T>(G0, vbase + (long)m * C) * xu0;
                const T w1 = ah::ld_a16<T>(G1, ubase + (long)m * C) * xv1 + sgn2 * ah::ld_a16<T>(G1, vbase + (long)m * C) * xu1;
                gp[m] = w0 * rr0[l] + w1 * rr1[l];
                sp += y4[j] * (w0 * tt0[l] + w1 * tt1[l]);
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        T a8[8];
        const bool up = lane & 16;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            a8[j] = (up ? gp[j + 8] : gp[j]) + __shfl_xor_sync(0xffffffffu, up ? gp[j] : gp[j + 8], 16);
        T a4[4];
        const bool b8 = lane & 8;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a4[j] = (b8 ? a8[j + 4] : a8[j]) + __shfl_xor_sync(0xffffffffu, b8 ? a8[j] : a8[j + 4], 8);
        T a2[2];
        const bool b4 = lane & 4;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            a2[j] = (b4 ? a4[j + 2] : a4[j]) + __shfl_xor_sync(0xffffffffu, b4 ? a4[j] : a4[j + 2], 4);
        const bool b2 = lane & 2;
        T a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 2);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        int src = 0;
        if (lane < 16) {
            const int m = lane;
            src = ((m & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (((m >> 3) & 1) << 4);
        }
        T gm = __shfl_sync(0xffffffffu, a1, src);
        if (warp == 1) {
            if (lane < 16) sw[sb][lane] = gm;
            if (lane == 0) sw[sb][16] = sp;
        }
        __syncthreads();
        if (warp == 0) {
            if (lane < 16) gm += sw[sb][lane];
            if (lane == 0) sp += sw[sb][16];
#pragma unroll
            for (int m = 1; m < 16; ++m) gp[m] = __shfl_sync(0xffffffffu, gm, m);
            if (lane == 0) {
                T y[16];
                ld16(YP + 16L * p, y);
                T ux, uy, uz, rinv;
                if constexpr (std::is_same<T, float>::value) {
                    const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
                    ux = up4.x; uy = up4.y; uz = up4.z; rinv = up4.w;
                } else {
                    ux = UP[4L * p]; uy = UP[4L * p + 1]; uz = UP[4L * p + 2]; rinv = UP[4L * p + 3];
                }
                T jx, jy_, jz;
                sh16_jyt<T>(ux, uy, uz, rinv, T(1), y, gp, jx, jy_, jz);
                const T g0 = jx + sp * ux, g1 = jy_ + sp * uy, g2 = jz + sp * uz;
                if (u != cur_u) {
                    if (cur_u >= 0) {
                        atomicAdd(F + 3L * cur_u + 0, sv[9]);
                        atomicAdd(F + 3L * cur_u + 1, sv[10]);
                        atomicAdd(F + 3L * cur_u + 2, sv[11]);
                    }
                    cur_u = u;
                    sv[9] = -g0; sv[10] = -g1; sv[11] = -g2;
                } else {
                    sv[9] -= g0; sv[10] -= g1; sv[11] -= g2;
                }
                atomicAdd(F + 3L * v + 0, g0);
                atomicAdd(F + 3L * v + 1, g1);
                atomicAdd(F + 3L * v + 2, g2);
                const T rr = T(1) / rinv;
                const T vx = ux * rr, vy = uy * rr, vz = uz * rr;
                sv[0] += vx * g0; sv[1] += vx * g1; sv[2] += vx * g2;
                sv[3] += vy * g0; sv[4] += vy * g1; sv[5] += vy * g2;
                sv[6] += vz * g0; sv[7] += vz * g1; sv[8] += vz * g2;
            }
        }
    }
    if (c == 0 && cur_u >= 0) {
        atomicAdd(F + 3L * cur_u + 0, sv[9]);
        atomicAdd(F + 3L * cur_u + 1, sv[10]);
        atomicAdd(F + 3L * cur_u + 2, sv[11]);
        T* o = mv + 9L * bi;
#pragma unroll
        for (int k = 0; k < 9; ++k) atomicAdd(o + k, sv[k]);
    }
}

// 32-thread (CPL=2) pair-owned force kernel for sm_89 when LST_FORCE_64T=0.
template <typename T, int CPL, int POLY = 0>
__global__ void __launch_bounds__(256) k_force_pair_sp(
    const int* __restrict__ rptr, const int* __restrict__ snd, const int* __restrict__ ps,
    const int* __restrict__ sp_idx, const T* __restrict__ sp_u, const int* __restrict__ ngraph,
    const T* __restrict__ YP, const T* __restrict__ UP, const T* __restrict__ vec, const T* __restrict__ G0,
    const T* __restrict__ Rk0, const T* __restrict__ Tk0, const T* __restrict__ X0, const T* __restrict__ G1,
    const T* __restrict__ Rk1, const T* __restrict__ Tk1, const T* __restrict__ X1, T* F, T* mv, T sp_h, T sp_invh,
    int N, int C, int fes = 2, const int* __restrict__ norder = nullptr) {
    const int wid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int nn = wid / fes, wsub = wid % fes;
    const int lane = threadIdx.x % 32;
    if (nn >= N) return;
    const int n = norder ? norder[nn] : nn;
    T gn[2][CPL][16];
    T xn[2][CPL];
#pragma unroll
    for (int k = 0; k < CPL; ++k) {
        const int c = lane + 32 * k;
        xn[0][k] = X0[(long)n * C + c];
        xn[1][k] = X1[(long)n * C + c];
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const long idx = ((long)n * 16 + m) * C + c;
            gn[0][k][m] = G0[idx];
            gn[1][k][m] = G1[idx];
        }
    }
    T v[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) v[k] = T(0);
    T fn0 = T(0), fn1 = T(0), fn2 = T(0);
    bool any_pair = false;
    const int e1 = rptr[n + 1];
    for (int e = rptr[n] + wsub; e < e1; e += fes) {
        const int ps_e = ps[e];
        if (ps_e & 1) continue;
        const long p = ps_e >> 1;
        const int kidx = sp_idx[p];
        if (kidx < 0) continue;
        any_pair = true;
        const long s = snd[e];
        const T u = sp_u[p];
        T h00 = T(0), h10 = T(0), h11 = T(0), d00 = T(0), d10 = T(0), d11 = T(0);
        if constexpr (POLY == 0) mspl::hermite_rt<T>(u, sp_h, sp_invh, h00, h10, h11, d00, d10, d11);
        T y[16];
        ld16(YP + 16L * p, y);
        T gp[16];
#pragma unroll
        for (int m = 0; m < 16; ++m) gp[m] = T(0);
        T sp = T(0);
        const long kbase0 = (long)kidx * 4 * C;
        const long kbase1 = kbase0 + 4 * C;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const T* Rk = t ? Rk1 : Rk0;
            const T* Tk = t ? Tk1 : Tk0;
            const T* Gs = t ? G1 : G0;
            const T* X = t ? X1 : X0;
#pragma unroll
            for (int k = 0; k < CPL; ++k) {
                const int c = lane + 32 * k;
                const T xs = X[s * C + c];
                const T xnv = xn[t][k];
                const long sbase = (s * 16) * C + c;
                T rr[4], tt[4];
                if constexpr (POLY > 0) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        T c0, c1, c2, c3;
                        mspl::ld_c4<T, POLY>(Rk, kbase0 + l * C + c, c0, c1, c2, c3);
                        rr[l] = ((c3 * u + c2) * u + c1) * u + c0;
                        tt[l] = ((T(3) * c3 * u + T(2) * c2) * u + c1) * sp_invh;
                    }
                } else {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const long off0 = kbase0 + l * C + c, off1 = kbase1 + l * C + c;
                        const T r0 = Rk[off0], r1 = Rk[off1];
                        const T t0 = Tk[off0], t1 = Tk[off1];
                        const T dr = r0 - r1;
                        rr[l] = r1 + h00 * dr + h10 * t0 + h11 * t1;
                        tt[l] = d00 * dr + d10 * t0 + d11 * t1;
                    }
                }
                const T* gg = gn[t][k];
                {
                    const T w0 = gg[0] * xs + Gs[sbase] * xnv;
                    gp[0] += w0 * rr[0];
                    sp += y[0] * w0 * tt[0];
                }
                T q1 = T(0), q2 = T(0), q3 = T(0);
#pragma unroll
                for (int m = 1; m < 4; ++m) {
                    const T w = gg[m] * xs - Gs[sbase + (long)m * C] * xnv;
                    gp[m] += w * rr[1];
                    q1 += y[m] * w;
                }
#pragma unroll
                for (int m = 4; m < 9; ++m) {
                    const T w = gg[m] * xs + Gs[sbase + (long)m * C] * xnv;
                    gp[m] += w * rr[2];
                    q2 += y[m] * w;
                }
#pragma unroll
                for (int m = 9; m < 16; ++m) {
                    const T w = gg[m] * xs - Gs[sbase + (long)m * C] * xnv;
                    gp[m] += w * rr[3];
                    q3 += y[m] * w;
                }
                sp += q1 * tt[1] + q2 * tt[2] + q3 * tt[3];
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
        T a8[8];
        const bool up = lane & 16;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            a8[j] = (up ? gp[j + 8] : gp[j]) + __shfl_xor_sync(0xffffffffu, up ? gp[j] : gp[j + 8], 16);
        T a4[4];
        const bool b8 = lane & 8;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a4[j] = (b8 ? a8[j + 4] : a8[j]) + __shfl_xor_sync(0xffffffffu, b8 ? a8[j] : a8[j + 4], 8);
        T a2[2];
        const bool b4 = lane & 4;
#pragma unroll
        for (int j = 0; j < 2; ++j)
            a2[j] = (b4 ? a4[j + 2] : a4[j]) + __shfl_xor_sync(0xffffffffu, b4 ? a4[j] : a4[j + 2], 4);
        const bool b2 = lane & 2;
        T a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 2);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        int src = 0;
        if (lane < 16) {
            const int m = lane;
            src = ((m & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (((m >> 3) & 1) << 4);
        }
        const T gm = __shfl_sync(0xffffffffu, a1, src);
#pragma unroll
        for (int m = 1; m < 16; ++m) gp[m] = __shfl_sync(0xffffffffu, gm, m);
        if (lane == 0) {
            T ux, uy, uz, rinv;
            if constexpr (std::is_same<T, float>::value) {
                const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
                ux = up4.x; uy = up4.y; uz = up4.z; rinv = up4.w;
            } else {
                ux = UP[4L * p]; uy = UP[4L * p + 1]; uz = UP[4L * p + 2]; rinv = UP[4L * p + 3];
            }
            T jx, jy_, jz;
            sh16_jyt<T>(ux, uy, uz, rinv, T(1), y, gp, jx, jy_, jz);
            const T g0 = jx + sp * ux, g1 = jy_ + sp * uy, g2 = jz + sp * uz;
            fn0 -= g0; fn1 -= g1; fn2 -= g2;
            atomicAdd(F + 3L * s + 0, g0);
            atomicAdd(F + 3L * s + 1, g1);
            atomicAdd(F + 3L * s + 2, g2);
            const T vx = vec[3L * e], vy = vec[3L * e + 1], vz = vec[3L * e + 2];
            v[0] += vx * g0; v[1] += vx * g1; v[2] += vx * g2;
            v[3] += vy * g0; v[4] += vy * g1; v[5] += vy * g2;
            v[6] += vz * g0; v[7] += vz * g1; v[8] += vz * g2;
        }
    }
    if (lane == 0 && any_pair) {
        atomicAdd(F + 3L * n + 0, fn0);
        atomicAdd(F + 3L * n + 1, fn1);
        atomicAdd(F + 3L * n + 2, fn2);
        T* o = mv + 9L * ngraph[n];
#pragma unroll
        for (int k = 0; k < 9; ++k) atomicAdd(o + k, v[k]);
    }
}
template <typename T>
__global__ void k_forces_pair(const T* g, const int* rptr, const int* pidx, const int* pe0, const int* pe1,
                              const T* core_f, T* F, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    T d[3] = {T(0), T(0), T(0)};
    for (int e = rptr[n]; e < rptr[n + 1]; ++e) {
        const long p = pidx[e];
        const int a = pe0[p];
        const long ep = a == e ? pe1[p] : a;
        for (int k = 0; k < 3; ++k) d[k] += g[3L * e + k] - g[3 * ep + k];
    }
    for (int k = 0; k < 3; ++k) F[3L * n + k] = core_f[3L * n + k] - d[k];
}

// k_forces_pair with a warp per atom: lanes take every 32nd incoming edge, then a shuffle sum
template <typename T>
__global__ void k_forces_pair_w(const T* g, const int* rptr, const int* pidx, const int* pe0, const int* pe1,
                                const T* core_f, T* F, int N) {
    const int n = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (n >= N) return;
    T d0 = T(0), d1 = T(0), d2 = T(0);
    const int e1 = rptr[n + 1];
    for (int e = rptr[n] + lane; e < e1; e += 32) {
        const long p = pidx[e];
        const int a = pe0[p];
        const long ep = a == e ? pe1[p] : a;
        d0 += g[3L * e] - g[3 * ep];
        d1 += g[3L * e + 1] - g[3 * ep + 1];
        d2 += g[3L * e + 2] - g[3 * ep + 2];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        d0 += __shfl_xor_sync(0xffffffffu, d0, o);
        d1 += __shfl_xor_sync(0xffffffffu, d1, o);
        d2 += __shfl_xor_sync(0xffffffffu, d2, o);
    }
    if (lane < 3) F[3L * n + lane] = core_f[3L * n + lane] - (lane == 0 ? d0 : lane == 1 ? d1 : d2);
}

// ============================================================================ edges -> forces, stress
// g = dE/dvec per edge, and the virial sum_e vec (x) g per graph (warp-aggregated: edges are graph-sorted)
template <typename T>
__global__ void k_edge_vec(const T* se, const T* gy, const T* U, const T* JY, T* g, const T* vec, const int* egraph,
                           T* mv, int E) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    bool ok = e < E;
    T a[3] = {T(0), T(0), T(0)};
    int gr = -1;
    T v[9];
    if (ok) {
        T s = se[e];
        a[0] = s * U[3 * e]; a[1] = s * U[3 * e + 1]; a[2] = s * U[3 * e + 2];
        T w[16];
        ld16(gy + 16L * e, w);
        for (int m = 0; m < 16; ++m)
            for (int k = 0; k < 3; ++k) a[k] += w[m] * JY[48L * e + 3 * m + k];
        for (int k = 0; k < 3; ++k) g[3L * e + k] = a[k];
        gr = egraph[e];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) v[3 * i + j] = vec[3L * e + i] * a[j];
    } else {
        for (int k = 0; k < 9; ++k) v[k] = T(0);
    }
    int g0 = __shfl_sync(0xffffffffu, gr, 0);
    int glast = __shfl_sync(0xffffffffu, gr, 31);
    bool uni = __all_sync(0xffffffffu, gr == g0 || gr == -1) && g0 >= 0;
    (void)glast;
    if (uni) {
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            T x = v[k];
            for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
            v[k] = x;
        }
        if ((threadIdx.x & 31) == 0)
            for (int k = 0; k < 9; ++k) atomicAdd(mv + 9L * g0 + k, v[k]);
    } else if (ok) {
        for (int k = 0; k < 9; ++k) atomicAdd(mv + 9L * gr + k, v[k]);
    }
}

template <typename T>
__global__ void k_forces(const T* g, const int* rptr, const int* sptr, const int* sperm, const T* core_f, T* F, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    T d[3] = {T(0), T(0), T(0)};
    for (int e = rptr[n]; e < rptr[n + 1]; ++e)
        for (int k = 0; k < 3; ++k) d[k] += g[3L * e + k];
    for (int i = sptr[n]; i < sptr[n + 1]; ++i) {
        long e = sperm[i];
        for (int k = 0; k < 3; ++k) d[k] -= g[3 * e + k];
    }
    for (int k = 0; k < 3; ++k) F[3L * n + k] = core_f[3L * n + k] - d[k];
}


// ============================================================================ node elementwise
template <typename T>
__global__ void k_x0(const T* T0, const int* spc, T* x0, int N, int C) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * C) return;
    long n = i / C, c = i % C;
    x0[i] = T0[(long)spc[n] * C + c];
}

// out[m, n, d] = tmp[(m N + n) SC + s(n) C + d]
template <typename T>
__global__ void k_pick(const T* tmp, const int* spc, T* out, int N, int C, int S) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long NC = (long)N * C;
    if (i >= 16 * NC) return;
    long mn = i / C, d = i % C;
    long n = mn % N;
    out[i] = tmp[mn * S * C + (long)spc[n] * C + d];
}

// tmp[(m N + n) SC + s C + d] = s == s(n) ? in[m, n, d] : 0
template <typename T>
__global__ void k_expand(const T* in, const int* spc, T* tmp, int N, int C, int S) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long SC = (long)S * C;
    if (i >= 16L * N * SC) return;
    long mn = i / SC, j = i % SC;
    long s = j / C, d = j % C;
    long n = mn % N;
    tmp[i] = (s == spc[n]) ? in[mn * C + d] : T(0);
}

// z [N, ZW] -> x1 [N, C], sc [N, C]
template <typename T>
__global__ void k_split_z(const T* z, const int* spc, T* x1, T* sc, int N, int C, int ZW) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * C) return;
    long n = i / C, c = i % C;
    x1[i] = z[n * ZW + 1 + c];
    sc[i] = z[n * ZW + 1 + C + (long)spc[n] * C + c];
}

// zb[n] = [v0[n] or c0, x[n], block s(n) = b[n], 0 elsewhere]
template <typename T>
__global__ void k_build_z(T* zb, const T* v0, T c0, const T* x, const T* b, const int* spc, int N, int C, int ZW) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * ZW) return;
    long n = i / ZW, j = i % ZW;
    T v;
    if (j == 0) v = v0 ? v0[n] : c0;
    else if (j <= C) v = x[n * C + j - 1];
    else {
        long k = j - 1 - C;
        v = (k / C == spc[n]) ? b[n * C + k % C] : T(0);
    }
    zb[i] = v;
}

template <typename T>
__global__ void k_readout(const T* u, const T* z, const T* wro2, const int* ngraph, T* tt, T* ne, T* Eg, T act,
                          T scale, int N, int hid, int ZW) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    T e1 = T(0);
    for (int h = 0; h < hid; ++h) {
        T v = u[(long)n * hid + h];
        T t = act * v * sigm(v);
        tt[(long)n * hid + h] = t;
        e1 += t * wro2[h];
    }
    T e = scale * (z[(long)n * ZW] + e1);
    ne[n] = e;
    atomicAdd(Eg + ngraph[n], e);
}

// ub = scale wro2 act silu'(u)
template <typename T>
__global__ void k_ub(const T* u, const T* wro2, T* ub, T act, T scale, int N, int hid) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * hid) return;
    T v = u[i], s = sigm(v);
    ub[i] = scale * wro2[i % hid] * act * s * (T(1) + v * (T(1) - s));
}

// tbs = ubs act silu'(u); us = ubs scale wro2 act silu''(u) + qn wro2 act silu'(u) ; qn = scale (Eb[g(n)] + Enb[n])
template <typename T>
__global__ void k_readout_rev(const T* u, const T* ubs, const T* wro2, const T* Eb, const int* ngraph, T* tbs, T* us,
                              T* qn, T act, T scale, int N, int hid, const float* __restrict__ d_scot = nullptr,
                              const T* __restrict__ Enb = nullptr) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * hid) return;
    long n = i / hid;
    int h = i % hid;
    T v = u[i], s = sigm(v);
    T d1 = s * (T(1) + v * (T(1) - s));
    T d2 = s * (T(1) - s) * (T(2) + v * (T(1) - T(2) * s));
    const T sc = d_scot ? T(d_scot[0]) : T(1);
    const T eb = Eb[ngraph[n]] + (Enb ? Enb[n] : T(0));
    T q = scale * eb * sc;
    tbs[i] = ubs[i] * act * d1;
    us[i] = ubs[i] * scale * wro2[h] * act * d2 + q * wro2[h] * act * d1;
    if (h == 0) qn[n] = q;
}

template <typename T>
__global__ void k_copy(T* dst, const T* src, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// zero up to 24 segments of 4-byte words (blockIdx.y = segment)
struct ZeroSegs {
    unsigned* p[32];
    long n[32];
};
__global__ void k_zero_words(ZeroSegs z) {
    unsigned* p = z.p[blockIdx.y];
    const long n = z.n[blockIdx.y];
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) p[i] = 0u;
}

template <typename T>
__global__ void k_fill(T* dst, T v, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = v;
}

template <typename T>
__global__ void k_mul(T* dst, const T* a, const T* b, long n) {   // dst[e, m] = a[e, m] * b[e]
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] * b[i / 16];
}

// ============================================================================ product polynomials
// thread (n, d): x = B[:, n, d]; Wl[k] = W[s(n), k, d]
template <typename T, bool NM = false>
__device__ __forceinline__ void load_x_nm(const T* B, long NC, long n, int d, int C, T* x) {
#pragma unroll
    for (int m = 0; m < 16; ++m) x[m] = NM ? B[(n * 16 + m) * C + d] : B[m * NC + n * C + d];
}
template <typename T>
__device__ __forceinline__ void load_x(const T* B, long NC, long nd, T* x) {
#pragma unroll
    for (int m = 0; m < 16; ++m) x[m] = B[m * NC + nd];
}
template <typename T>
__device__ __forceinline__ void load_w(const T* W, int s, int d, int C, T* w) {
#pragma unroll
    for (int k = 0; k < POLY_K; ++k) w[k] = W[((long)s * POLY_K + k) * C + d];
}

template <typename T, bool NM = false>
__global__ void k_prod_fwd(const T* B, const T* W, const int* spc, T* f, int N, int C) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long NC = (long)N * C;
    if (i >= NC) return;
    int n = i / C, d = i % C;
    T x[16], w[POLY_K];
    load_x_nm<T, NM>(B, NC, n, d, C, x);
    load_w(W, spc[n], d, C, w);
    f[i] = poly_dot(x, w);
}

template <typename T, bool NM = false>
__global__ void k_prod_vjp(const T* B, const T* W, const int* spc, const T* fb, T* Bb, int N, int C) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long NC = (long)N * C;
    if (i >= NC) return;
    int n = i / C, d = i % C;
    T x[16], w[POLY_K], G[16];
    load_x_nm<T, NM>(B, NC, n, d, C, x);
    load_w(W, spc[n], d, C, w);
    T f = fb[i];
#pragma unroll
    for (int k = 0; k < POLY_K; ++k) w[k] *= f;
    poly_vjp(x, w, G);
#pragma unroll
    for (int m = 0; m < 16; ++m) Bb[NM ? ((long)n * 16 + m) * C + d : m * NC + i] = G[m];
}

// reverse of Bb = vjp(B, W fb): Bs = hvp(B, W fb, Bbs); fbs = sum_k W_k jvp_k; Wp[n, k, d] = fb jvp_k
template <typename T>
__global__ void k_prod_rev2(const T* B, const T* W, const int* spc, const T* fb, const T* Bbs, T* Bs, T* fbs, T* Wp,
                            int N, int C) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long NC = (long)N * C;
    if (i >= NC) return;
    int n = i / C, d = i % C;
    T x[16], w[POLY_K], t[16], H[16], J[POLY_K], V[POLY_K];
    load_x(B, NC, i, x);
    load_x(Bbs, NC, i, t);
    load_w(W, spc[n], d, C, w);
    T f = fb[i];
#pragma unroll
    for (int k = 0; k < POLY_K; ++k) V[k] = w[k] * f;
    poly_hvp(x, V, t, H);
#pragma unroll
    for (int m = 0; m < 16; ++m) Bs[m * NC + i] = H[m];
    poly_jvp(x, t, J);
    T acc = T(0);
#pragma unroll
    for (int k = 0; k < POLY_K; ++k) {
        acc += w[k] * J[k];
        Wp[((long)n * POLY_K + k) * C + d] = f * J[k];
    }
    fbs[i] = acc;
}

// reverse of f = sum_k W_k P_k(B): Bs += vjp(B, W fs); Wp[n, k, d] += P_k fs
template <typename T>
__global__ void k_prod_rev1(const T* B, const T* W, const int* spc, const T* fs, T* Bs, T* Wp, int N, int C) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long NC = (long)N * C;
    if (i >= NC) return;
    int n = i / C, d = i % C;
    T x[16], w[POLY_K], G[16], Pk[POLY_K];
    load_x(B, NC, i, x);
    load_w(W, spc[n], d, C, w);
    T f = fs[i];
#pragma unroll
    for (int k = 0; k < POLY_K; ++k) w[k] *= f;
    poly_vjp(x, w, G);
#pragma unroll
    for (int m = 0; m < 16; ++m) Bs[m * NC + i] += G[m];
    poly_p(x, Pk);
#pragma unroll
    for (int k = 0; k < POLY_K; ++k) Wp[((long)n * POLY_K + k) * C + d] += f * Pk[k];
}

// LST_PROD_DIRECT=1: reduce f * J[k] / f * Pk[k] across the block's M = 256 / C nodes directly into gW[s, k, d],
// dropping Wp0/Wp1 DRAM writes/reads and both onehot^T @ Wp GEMMs, and computing poly_jvp before poly_hvp to cut regs.
template <typename T>
__device__ __forceinline__ void prod_reduce_gw(const T* vals, T scale, T* gW, int sp, int ln, int d, int M, int C,
                                               bool active, T* sm, int* sm_spc) {
    if (d == 0) sm_spc[ln] = active ? sp : -1;
    __syncthreads();
    const int s0 = sm_spc[0];
    bool same = s0 >= 0;
    for (int m = 1; m < M; ++m)
        if (sm_spc[m] != s0) same = false;
#pragma unroll
    for (int k0 = 0; k0 < POLY_K; k0 += 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) sm[j * 256 + ln * C + d] = active ? scale * vals[k0 + j] : T(0);
        __syncthreads();
        if (M >= 4) {
            if (ln < 4) {
                const int k = k0 + ln;
                if (same) {
                    T acc = T(0);
                    for (int m = 0; m < M; ++m) acc += sm[ln * 256 + m * C + d];
                    atomicAdd(gW + ((long)s0 * POLY_K + k) * C + d, acc);
                } else {
                    for (int m = 0; m < M; ++m) {
                        const int sm_s = sm_spc[m];
                        if (sm_s >= 0) atomicAdd(gW + ((long)sm_s * POLY_K + k) * C + d, sm[ln * 256 + m * C + d]);
                    }
                }
            }
        } else {
            for (int j = ln; j < 4; j += M) {
                const int k = k0 + j;
                if (same) {
                    T acc = T(0);
                    for (int m = 0; m < M; ++m) acc += sm[j * 256 + m * C + d];
                    atomicAdd(gW + ((long)s0 * POLY_K + k) * C + d, acc);
                } else {
                    for (int m = 0; m < M; ++m) {
                        const int sm_s = sm_spc[m];
                        if (sm_s >= 0) atomicAdd(gW + ((long)sm_s * POLY_K + k) * C + d, sm[j * 256 + m * C + d]);
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <typename T, bool NM = false>
__global__ void k_prod_rev2_dir(const T* B, const T* W, const int* spc, const T* fb, const T* Bbs, T* Bs, T* fbs,
                                T* gW, int N, int C) {
    __shared__ T sm[4 * 256];
    __shared__ int sm_spc[8];
    const int tid = threadIdx.x;
    const int M = 256 / C, ln = tid / C, d = tid % C;
    const long i = (long)blockIdx.x * 256 + tid;
    const long NC = (long)N * C;
    const bool active = i < NC;
    const int n = active ? (int)(i / C) : 0;
    const int sp = active ? spc[n] : -1;
    T x[16], t[16], J[POLY_K];
    T f = T(0);
    if (active) {
        load_x_nm<T, NM>(B, NC, n, d, C, x);
        load_x_nm<T, NM>(Bbs, NC, n, d, C, t);
        f = fb[i];
        poly_jvp(x, t, J);
        T acc = T(0);
#pragma unroll
        for (int k = 0; k < POLY_K; ++k) acc += W[((long)sp * POLY_K + k) * C + d] * J[k];
        fbs[i] = acc;
    }
    prod_reduce_gw(J, f, gW, sp, ln, d, M, C, active, sm, sm_spc);
    if (active) {
        T V[POLY_K], H[16];
#pragma unroll
        for (int k = 0; k < POLY_K; ++k) V[k] = W[((long)sp * POLY_K + k) * C + d] * f;
        poly_hvp(x, V, t, H);
#pragma unroll
        for (int m = 0; m < 16; ++m) Bs[NM ? ((long)n * 16 + m) * C + d : m * NC + i] = H[m];
    }
}

template <typename T, bool NM = false>
__global__ void k_prod_rev1_dir(const T* B, const T* W, const int* spc, const T* fs, T* Bs, T* gW, int N, int C) {
    __shared__ T sm[4 * 256];
    __shared__ int sm_spc[8];
    const int tid = threadIdx.x;
    const int M = 256 / C, ln = tid / C, d = tid % C;
    const long i = (long)blockIdx.x * 256 + tid;
    const long NC = (long)N * C;
    const bool active = i < NC;
    const int n = active ? (int)(i / C) : 0;
    const int sp = active ? spc[n] : -1;
    T x[16], Pk[POLY_K];
    T f = T(0);
    if (active) {
        load_x_nm<T, NM>(B, NC, n, d, C, x);
        f = fs[i];
        poly_p(x, Pk);
    }
    prod_reduce_gw(Pk, f, gW, sp, ln, d, M, C, active, sm, sm_spc);
    if (active) {
        T w[POLY_K], G[16];
#pragma unroll
        for (int k = 0; k < POLY_K; ++k) w[k] = W[((long)sp * POLY_K + k) * C + d] * f;
        poly_vjp(x, w, G);
#pragma unroll
        for (int m = 0; m < 16; ++m) Bs[NM ? ((long)n * 16 + m) * C + d : m * NC + i] += G[m];
    }
}

// Deduplicated 13-basis symmetric contraction kernels (LST_PROD_DEDUP=1): collapses the 28 redundant polynomials
// (293 terms) to 13 unique polynomials (94 terms -> ~3.3x fewer FLOPs, 2.15x fewer W loads, barriers & atomics!).
template <typename T>
__device__ __forceinline__ void prod_reduce_gw_u(const T* vals, T scale, T* gWu, int sp, int ln, int d, int M, int C,
                                                 bool active, T* sm, int* sm_spc) {
    if (d == 0) sm_spc[ln] = active ? sp : -1;
    __syncthreads();
    const int s0 = sm_spc[0];
    bool same = s0 >= 0;
    for (int m = 1; m < M; ++m)
        if (sm_spc[m] != s0) same = false;
#pragma unroll
    for (int u0 = 0; u0 < (POLY_U & ~3); u0 += 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) sm[j * 256 + ln * C + d] = active ? scale * vals[u0 + j] : T(0);
        __syncthreads();
        if (M >= 4) {
            if (ln < 4) {
                const int u = u0 + ln;
                if (same) {
                    T acc = T(0);
                    for (int m = 0; m < M; ++m) acc += sm[ln * 256 + m * C + d];
                    atomicAdd(gWu + ((long)s0 * POLY_U + u) * C + d, acc);
                } else {
                    for (int m = 0; m < M; ++m) {
                        const int sm_s = sm_spc[m];
                        if (sm_s >= 0) atomicAdd(gWu + ((long)sm_s * POLY_U + u) * C + d, sm[ln * 256 + m * C + d]);
                    }
                }
            }
        } else {
            for (int j = ln; j < 4; j += M) {
                const int u = u0 + j;
                if (same) {
                    T acc = T(0);
                    for (int m = 0; m < M; ++m) acc += sm[j * 256 + m * C + d];
                    atomicAdd(gWu + ((long)s0 * POLY_U + u) * C + d, acc);
                } else {
                    for (int m = 0; m < M; ++m) {
                        const int sm_s = sm_spc[m];
                        if (sm_s >= 0) atomicAdd(gWu + ((long)sm_s * POLY_U + u) * C + d, sm[j * 256 + m * C + d]);
                    }
                }
            }
        }
        __syncthreads();
    }
    constexpr int REM = POLY_U & 3;
    if constexpr (REM > 0) {
        constexpr int u0 = POLY_U - REM;
#pragma unroll
        for (int j = 0; j < REM; ++j) sm[j * 256 + ln * C + d] = active ? scale * vals[u0 + j] : T(0);
        __syncthreads();
        for (int j = ln; j < REM; j += M) {
            const int u = u0 + j;
            if (same) {
                T acc = T(0);
                for (int m = 0; m < M; ++m) acc += sm[j * 256 + m * C + d];
                atomicAdd(gWu + ((long)s0 * POLY_U + u) * C + d, acc);
            } else {
                for (int m = 0; m < M; ++m) {
                    const int sm_s = sm_spc[m];
                    if (sm_s >= 0) atomicAdd(gWu + ((long)sm_s * POLY_U + u) * C + d, sm[j * 256 + m * C + d]);
                }
            }
        }
    }
}

template <typename T, bool NM = false, typename A16 = T>
__device__ __forceinline__ void load_x_nm_h(const A16* B, long NC, int n, int d, int C, T* x) {
    const long base = NM ? (long)n * 16 * C + d : (long)n * C + d;
    const long step = NM ? (long)C : NC;
#pragma unroll
    for (int m = 0; m < 16; ++m) x[m] = ah::ld_a16<T>(B, base + m * step);
}

template <typename T, bool NM = false, typename A16 = T>
__global__ void k_prod_fwd_u(const A16* B, const T* Wu, const int* spc, T* f, int N, int C) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long NC = (long)N * C;
    if (i >= NC) return;
    const int n = (int)(i / C), d = (int)(i % C), sp = spc[n];
    T x[16], w[POLY_U];
    load_x_nm_h<T, NM>(B, NC, n, d, C, x);
#pragma unroll
    for (int u = 0; u < POLY_U; ++u) w[u] = Wu[((long)sp * POLY_U + u) * C + d];
    f[i] = poly_dot_u(x, w);
}

template <typename T, bool NM = false, typename A16 = T>
__global__ void k_prod_vjp_u(const A16* B, const T* Wu, const int* spc, const T* fb, A16* Bb, int N, int C) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long NC = (long)N * C;
    if (i >= NC) return;
    const int n = (int)(i / C), d = (int)(i % C), sp = spc[n];
    T x[16], w[POLY_U], G[16];
    load_x_nm_h<T, NM>(B, NC, n, d, C, x);
    const T f = fb[i];
#pragma unroll
    for (int u = 0; u < POLY_U; ++u) w[u] = Wu[((long)sp * POLY_U + u) * C + d] * f;
    poly_vjp_u(x, w, G);
#pragma unroll
    for (int m = 0; m < 16; ++m) ah::st_a16<A16>(Bb, NM ? ((long)n * 16 + m) * C + d : m * NC + i, G[m]);
}

// 1-barrier version of prod_reduce_gw_u (LST_PROD_1BAR=1): writes all POLY_U=13 values into sm[13][256] (13 KB in FP32)
// so only ONE __syncthreads() is needed instead of 8! Bit-exact sums.
template <typename T>
__device__ __forceinline__ void prod_reduce_gw_u_1bar(const T* vals, T scale, T* gWu, int sp, int ln, int d, int M,
                                                      int C, bool active, T* sm, int* sm_spc) {
    if (d == 0) sm_spc[ln] = active ? sp : -1;
#pragma unroll
    for (int u = 0; u < POLY_U; ++u) sm[u * 256 + ln * C + d] = active ? scale * vals[u] : T(0);
    __syncthreads();
    const int s0 = sm_spc[0];
    bool same = s0 >= 0;
    for (int m = 1; m < M; ++m)
        if (sm_spc[m] != s0) same = false;
    for (int u = ln; u < POLY_U; u += M) {
        if (same) {
            T acc = T(0);
            for (int m = 0; m < M; ++m) acc += sm[u * 256 + m * C + d];
            atomicAdd(gWu + ((long)s0 * POLY_U + u) * C + d, acc);
        } else {
            for (int m = 0; m < M; ++m) {
                const int sm_s = sm_spc[m];
                if (sm_s >= 0) atomicAdd(gWu + ((long)sm_s * POLY_U + u) * C + d, sm[u * 256 + m * C + d]);
            }
        }
    }
}

template <typename T, bool NM = false, typename A16 = T, bool ONEBAR = false>
__global__ void k_prod_rev2_dir_u(const A16* B, const T* Wu, const int* spc, const T* fb, const A16* Bbs, A16* Bs,
                                  T* fbs, T* gWu, int N, int C) {
    __shared__ T sm[(ONEBAR ? POLY_U : 4) * 256];
    __shared__ int sm_spc[8];
    const int tid = threadIdx.x;
    const int M = 256 / C, ln = tid / C, d = tid % C;
    const long i = (long)blockIdx.x * 256 + tid;
    const long NC = (long)N * C;
    const bool active = i < NC;
    const int n = active ? (int)(i / C) : 0;
    const int sp = active ? spc[n] : -1;
    T x[16], t[16], J[POLY_U];
    T f = T(0);
    if (active) {
        load_x_nm_h<T, NM>(B, NC, n, d, C, x);
        load_x_nm_h<T, NM>(Bbs, NC, n, d, C, t);
        f = fb[i];
        poly_jvp_u(x, t, J);
        T acc = T(0);
#pragma unroll
        for (int u = 0; u < POLY_U; ++u) acc += Wu[((long)sp * POLY_U + u) * C + d] * J[u];
        fbs[i] = acc;
    }
    if constexpr (ONEBAR)
        prod_reduce_gw_u_1bar(J, f, gWu, sp, ln, d, M, C, active, sm, sm_spc);
    else
        prod_reduce_gw_u(J, f, gWu, sp, ln, d, M, C, active, sm, sm_spc);
    if (active) {
        T V[POLY_U], H[16];
#pragma unroll
        for (int u = 0; u < POLY_U; ++u) V[u] = Wu[((long)sp * POLY_U + u) * C + d] * f;
        poly_hvp_u(x, V, t, H);
#pragma unroll
        for (int m = 0; m < 16; ++m) ah::st_a16<A16>(Bs, NM ? ((long)n * 16 + m) * C + d : m * NC + i, H[m]);
    }
}

template <typename T, bool NM = false, typename A16 = T, bool ONEBAR = false>
__global__ void k_prod_rev1_dir_u(const A16* B, const T* Wu, const int* spc, const T* fs, A16* Bs, T* gWu, int N, int C) {
    __shared__ T sm[(ONEBAR ? POLY_U : 4) * 256];
    __shared__ int sm_spc[8];
    const int tid = threadIdx.x;
    const int M = 256 / C, ln = tid / C, d = tid % C;
    const long i = (long)blockIdx.x * 256 + tid;
    const long NC = (long)N * C;
    const bool active = i < NC;
    const int n = active ? (int)(i / C) : 0;
    const int sp = active ? spc[n] : -1;
    T x[16], Pu[POLY_U];
    T f = T(0);
    if (active) {
        load_x_nm_h<T, NM>(B, NC, n, d, C, x);
        f = fs[i];
        poly_p_u(x, Pu);
    }
    if constexpr (ONEBAR)
        prod_reduce_gw_u_1bar(Pu, f, gWu, sp, ln, d, M, C, active, sm, sm_spc);
    else
        prod_reduce_gw_u(Pu, f, gWu, sp, ln, d, M, C, active, sm, sm_spc);
    if (active) {
        T w[POLY_U], G[16];
#pragma unroll
        for (int u = 0; u < POLY_U; ++u) w[u] = Wu[((long)sp * POLY_U + u) * C + d] * f;
        poly_vjp_u(x, w, G);
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const long idx = NM ? ((long)n * 16 + m) * C + d : m * NC + i;
            ah::st_a16<A16>(Bs, idx, ah::ld_a16<T>(Bs, idx) + G[m]);
        }
    }
}

// Wave 19: Species-grouped 1-channel/thread k_prod_rev2_spc64 (LST_PROD_REV2_SPC=1, C == 64).
// 128 threads = 2 atom-lanes (64 threads/atom, 1 ch/thread -> ZERO double-channel spill, <=96 regs, 2x occupancy!):
// loads Wu[s] ONCE into 3.3 KB smem and accumulates gWu in 6.6 KB smem across 8-16 atoms -> 16x fewer Wu loads & atomics!
template <typename H>
__global__ void __launch_bounds__(128, 4) k_prod_rev2_spc64(const H* __restrict__ B, const float* __restrict__ Wu,
                                                            const int* __restrict__ spc_off,
                                                            const int* __restrict__ spc_idx,
                                                            const int* __restrict__ spc_tiles32,
                                                            const float* __restrict__ fb, const H* __restrict__ Bbs,
                                                            H* __restrict__ Bs, float* __restrict__ fbs,
                                                            float* __restrict__ gWu, int S) {
    const int tot = spc_tiles32[S];
    const int by = blockIdx.y;
    if (by >= tot) return;
    int s = 0;
    while (s + 1 < S && by >= spc_tiles32[s + 1]) ++s;
    const int t = by - spc_tiles32[s];
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int chunk = 32 / gridDim.x;
    const int r_base = t * 32 + blockIdx.x * chunk;
    if (r_base >= cnt) return;

    const int tid = threadIdx.x, ln = tid >> 6, c = tid & 63;
    __shared__ float sm_wu[POLY_U][64];
    __shared__ float sm_gw[2][POLY_U][64];

    const float* wsrc = Wu + (long)s * POLY_U * 64;
    for (int idx = tid; idx < POLY_U * 64; idx += 128) sm_wu[idx >> 6][idx & 63] = wsrc[idx];
#pragma unroll
    for (int u = 0; u < POLY_U; ++u) sm_gw[ln][u][c] = 0.f;
    __syncthreads();

    const int npw = chunk >> 1;
    const int w_beg = r_base + ln * npw;
    const int w_end = min(cnt, w_beg + npw);
    for (int r = w_beg; r < w_end; ++r) {
        const int n = spc_idx[off + r];
        const long bbase = (long)n * 16 * 64 + c;
        float x[16], tv[16];
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            x[m] = ah::ld_a16<float>(B, bbase + (long)m * 64);
            tv[m] = ah::ld_a16<float>(Bbs, bbase + (long)m * 64);
        }
        const float f = fb[(long)n * 64 + c];
        {
            float J[POLY_U];
            poly_jvp_u(x, tv, J);
            float acc = 0.f;
#pragma unroll
            for (int u = 0; u < POLY_U; ++u) {
                acc += sm_wu[u][c] * J[u];
                sm_gw[ln][u][c] += f * J[u];
            }
            fbs[(long)n * 64 + c] = acc;
        }
        {
            float V[POLY_U], Hout[16];
#pragma unroll
            for (int u = 0; u < POLY_U; ++u) V[u] = sm_wu[u][c] * f;
            poly_hvp_u(x, V, tv, Hout);
#pragma unroll
            for (int m = 0; m < 16; ++m) ah::st_a16<H>(Bs, bbase + (long)m * 64, Hout[m]);
        }
    }
    __syncthreads();
    for (int u = ln; u < POLY_U; u += 2) {
        atomicAdd(gWu + ((long)s * POLY_U + u) * 64 + c, sm_gw[0][u][c] + sm_gw[1][u][c]);
    }
}

// Wave 15: Species-grouped __half2 / __nv_bfloat162 vectorized product reverse kernels (LST_PROD_H2=1, C == 64).
// 1 warp = 32 threads = all 64 channels (2 ch/thread -> 16x LDG.32 / STG.32 128B coalesced cache lines!).
// Each block of 4 warps processes 8 (b1, nsub=4) or 16 (b4, nsub=2) atoms of ONE species s:
//   - loads Wu[s] ONCE into smem (3.3 KB -> 8x-16x fewer Wu global loads!),
//   - accumulates gWu in smem (13.3 KB -> 0 extra regs, 1 barrier total, 8x-16x fewer gWu L2 atomics!).
template <typename H>
__global__ void __launch_bounds__(128, 4) k_prod_rev2_h2(const H* __restrict__ B, const float* __restrict__ Wu,
                                                         const int* __restrict__ spc_off,
                                                         const int* __restrict__ spc_idx,
                                                         const int* __restrict__ spc_tiles32,
                                                         const float* __restrict__ fb, const H* __restrict__ Bbs,
                                                         H* __restrict__ Bs, float* __restrict__ fbs,
                                                         float* __restrict__ gWu, int S) {
    const int tot = spc_tiles32[S];
    const int by = blockIdx.y;
    if (by >= tot) return;
    int s = 0;
    while (s + 1 < S && by >= spc_tiles32[s + 1]) ++s;
    const int t = by - spc_tiles32[s];
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int chunk = 32 / gridDim.x;
    const int r_base = t * 32 + blockIdx.x * chunk;
    if (r_base >= cnt) return;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ __align__(8) float2 sm_wu[POLY_U][32];
    __shared__ __align__(8) float2 sm_gw[4][POLY_U][32];

    const float2* Wu2 = reinterpret_cast<const float2*>(Wu + (long)s * POLY_U * 64);
    for (int idx = tid; idx < POLY_U * 32; idx += 128) sm_wu[idx >> 5][idx & 31] = Wu2[idx];
#pragma unroll
    for (int u = 0; u < POLY_U; ++u) sm_gw[warp][u][lane] = make_float2(0.f, 0.f);
    __syncthreads();

    const int npw = chunk >> 2;
    const int w_beg = r_base + warp * npw;
    const int w_end = min(cnt, w_beg + npw);
    for (int r = w_beg; r < w_end; ++r) {
        const int n = spc_idx[off + r];
        uint32_t bx[16], bt[16];
        const long bbase = (long)n * 16 * 32 + lane;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            bx[m] = ah::ld_h2<H>(B, bbase + (long)m * 32);
            bt[m] = ah::ld_h2<H>(Bbs, bbase + (long)m * 32);
        }
        const float2 f2 = *reinterpret_cast<const float2*>(fb + (long)n * 64 + 2 * lane);
        float fbs0 = 0.f, fbs1 = 0.f;
        // Channel 0: pack converted H0 into dead low-16 bits of bx[m] (-16 regs) & separate J[13] from V[13] (-13 regs)!
        {
            float x[16], tv[16];
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                x[m] = ah::unpack2<H>(bx[m]).x;
                tv[m] = ah::unpack2<H>(bt[m]).x;
            }
            {
                float J[POLY_U];
                poly_jvp_u(x, tv, J);
#pragma unroll
                for (int u = 0; u < POLY_U; ++u) {
                    fbs0 += sm_wu[u][lane].x * J[u];
                    sm_gw[warp][u][lane].x += f2.x * J[u];
                }
            }
            float V[POLY_U], H0[16];
#pragma unroll
            for (int u = 0; u < POLY_U; ++u) V[u] = sm_wu[u][lane].x * f2.x;
            poly_hvp_u(x, V, tv, H0);
#pragma unroll
            for (int m = 0; m < 16; ++m)
                bx[m] = (bx[m] & 0xffff0000u) | (ah::pack2<H>(H0[m], 0.f) & 0xffffu);
        }
        // Channel 1: unpack hi-16 bits of bx/bt, write directly to Bs combining low-16 H0 with hi-16 H1!
        {
            float x[16], tv[16];
#pragma unroll
            for (int m = 0; m < 16; ++m) {
                x[m] = ah::unpack2<H>(bx[m]).y;
                tv[m] = ah::unpack2<H>(bt[m]).y;
            }
            {
                float J[POLY_U];
                poly_jvp_u(x, tv, J);
#pragma unroll
                for (int u = 0; u < POLY_U; ++u) {
                    fbs1 += sm_wu[u][lane].y * J[u];
                    sm_gw[warp][u][lane].y += f2.y * J[u];
                }
            }
            float V[POLY_U], H1[16];
#pragma unroll
            for (int u = 0; u < POLY_U; ++u) V[u] = sm_wu[u][lane].y * f2.y;
            poly_hvp_u(x, V, tv, H1);
#pragma unroll
            for (int m = 0; m < 16; ++m)
                ah::st_h2<H>(Bs, bbase + (long)m * 32, (bx[m] & 0xffffu) | (ah::pack2<H>(0.f, H1[m]) & 0xffff0000u));
        }
        *reinterpret_cast<float2*>(fbs + (long)n * 64 + 2 * lane) = make_float2(fbs0, fbs1);
    }
    __syncthreads();
    for (int u = warp; u < POLY_U; u += 4) {
        const float2 g0 = sm_gw[0][u][lane], g1 = sm_gw[1][u][lane];
        const float2 g2 = sm_gw[2][u][lane], g3 = sm_gw[3][u][lane];
        const long gidx = ((long)s * POLY_U + u) * 64 + 2 * lane;
        atomicAdd(gWu + gidx, g0.x + g1.x + g2.x + g3.x);
        atomicAdd(gWu + gidx + 1, g0.y + g1.y + g2.y + g3.y);
    }
}

template <typename H, int NW = 4>
__global__ void __launch_bounds__(32 * NW) k_prod_rev1_h2(
    const H* __restrict__ B, const float* __restrict__ Wu,
    const int* __restrict__ spc_off, const int* __restrict__ spc_idx,
    const int* __restrict__ spc_tiles32, const float* __restrict__ fs,
    H* __restrict__ Bs, float* __restrict__ gWu, int S) {
    const int tot = spc_tiles32[S];
    const int by = blockIdx.y;
    if (by >= tot) return;
    int s = 0;
    while (s + 1 < S && by >= spc_tiles32[s + 1]) ++s;
    const int t = by - spc_tiles32[s];
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int chunk = 32 / gridDim.x;
    const int r_base = t * 32 + blockIdx.x * chunk;
    if (r_base >= cnt) return;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ __align__(8) float2 sm_wu[POLY_U][32];
    __shared__ __align__(8) float2 sm_gw[NW][POLY_U][32];

    const float2* Wu2 = reinterpret_cast<const float2*>(Wu + (long)s * POLY_U * 64);
    for (int idx = tid; idx < POLY_U * 32; idx += 32 * NW) sm_wu[idx >> 5][idx & 31] = Wu2[idx];
    const int npw = chunk / NW;
    const int w_beg = r_base + warp * npw;
    const int w_end = min(cnt, w_beg + npw);
    if (npw != 1 || w_beg >= cnt) {
#pragma unroll
        for (int u = 0; u < POLY_U; ++u) sm_gw[warp][u][lane] = make_float2(0.f, 0.f);
    }
    __syncthreads();

    for (int r = w_beg; r < w_end; ++r) {
        const int n = spc_idx[off + r];
        uint32_t bx[16];
        const long bbase = (long)n * 16 * 32 + lane;
#pragma unroll
        for (int m = 0; m < 16; ++m) bx[m] = ah::ld_h2<H>(B, bbase + (long)m * 32);
        const float2 f2 = *reinterpret_cast<const float2*>(fs + (long)n * 64 + 2 * lane);
        float G0[16];
        {
            float x[16], Pu[POLY_U];
#pragma unroll
            for (int m = 0; m < 16; ++m) x[m] = ah::unpack2<H>(bx[m]).x;
            poly_p_u(x, Pu);
#pragma unroll
            for (int u = 0; u < POLY_U; ++u) {
                const float gxv = f2.x * Pu[u];
                if (npw == 1) sm_gw[warp][u][lane].x = gxv;
                else sm_gw[warp][u][lane].x += gxv;
                Pu[u] = sm_wu[u][lane].x * f2.x;
            }
            poly_vjp_u(x, Pu, G0);
        }
        float G1[16];
        {
            float x[16], Pu[POLY_U];
#pragma unroll
            for (int m = 0; m < 16; ++m) x[m] = ah::unpack2<H>(bx[m]).y;
            poly_p_u(x, Pu);
#pragma unroll
            for (int u = 0; u < POLY_U; ++u) {
                const float gyv = f2.y * Pu[u];
                if (npw == 1) sm_gw[warp][u][lane].y = gyv;
                else sm_gw[warp][u][lane].y += gyv;
                Pu[u] = sm_wu[u][lane].y * f2.y;
            }
            poly_vjp_u(x, Pu, G1);
        }
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const long idx = bbase + (long)m * 32;
            const float2 old = ah::unpack2<H>(ah::ld_h2<H>(Bs, idx));
            ah::st_h2<H>(Bs, idx, ah::pack2<H>(old.x + G0[m], old.y + G1[m]));
        }
    }
    __syncthreads();
    for (int u = warp; u < POLY_U; u += NW) {
        float sx = 0.f, sy = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            const float2 gv = sm_gw[w][u][lane];
            sx += gv.x;
            sy += gv.y;
        }
        const long gidx = ((long)s * POLY_U + u) * 64 + 2 * lane;
        atomicAdd(gWu + gidx, sx);
        atomicAdd(gWu + gidx + 1, sy);
    }
}

// Species-grouped k_prod_rev2_dir_spc & k_prod_rev1_dir_spc (LST_PROD_SPC=1):
// Uses spc_off/spc_idx/spc_tiles32 so every block handles 32/NSUB atoms of ONE species s (2 atoms/thread at b1,
// 4 at b4): loads W[s] ONCE, accumulates gw[POLY_K] in registers across the loop, and reduces across 4 threads
// ONCE at the end -> 2x-4x fewer atomics, 4x fewer W loads, and 1/4 the barriers!
template <typename T, bool NM = true>
__global__ void __launch_bounds__(256) k_prod_rev2_dir_spc(const T* __restrict__ B, const T* __restrict__ W,
                                                           const int* __restrict__ spc_off,
                                                           const int* __restrict__ spc_idx,
                                                           const int* __restrict__ spc_tiles32,
                                                           const T* __restrict__ fb, const T* __restrict__ Bbs,
                                                           T* __restrict__ Bs, T* __restrict__ fbs, T* __restrict__ gW,
                                                           int N, int C, int S) {
    const int tot = spc_tiles32[S];
    const int by = blockIdx.y;
    if (by >= tot) return;
    int s = 0;
    while (s + 1 < S && by >= spc_tiles32[s + 1]) ++s;
    const int t = by - spc_tiles32[s];
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int nsub = gridDim.x, sub = blockIdx.x;
    const int chunk = 32 / nsub;
    const int r0 = t * 32 + sub * chunk;
    const int r1 = min(r0 + chunk, cnt);
    if (r0 >= r1) return;
    const int tid = threadIdx.x, ln = tid / C, d = tid % C, M = 256 / C;
    const long NC = (long)N * C;
    T w[POLY_K], gwa[POLY_K];
#pragma unroll
    for (int k = 0; k < POLY_K; ++k) {
        w[k] = W[((long)s * POLY_K + k) * C + d];
        gwa[k] = T(0);
    }
    for (int r = r0 + ln; r < r1; r += M) {
        const long n = spc_idx[off + r];
        const long i = n * C + d;
        T x[16], tv[16], J[POLY_K], V[POLY_K], H[16];
        load_x_nm<T, NM>(B, NC, n, d, C, x);
        load_x_nm<T, NM>(Bbs, NC, n, d, C, tv);
        const T f = fb[i];
        poly_jvp(x, tv, J);
        T acc = T(0);
#pragma unroll
        for (int k = 0; k < POLY_K; ++k) {
            acc += w[k] * J[k];
            gwa[k] += f * J[k];
            V[k] = w[k] * f;
        }
        fbs[i] = acc;
        poly_hvp(x, V, tv, H);
#pragma unroll
        for (int m = 0; m < 16; ++m) Bs[NM ? (n * 16 + m) * C + d : m * NC + i] = H[m];
    }
    __shared__ T sm[4 * 256];
#pragma unroll
    for (int k0 = 0; k0 < POLY_K; k0 += 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            sm[j * 256 + ln * C + d] = (k0 + j < POLY_K) ? gwa[k0 + j] : T(0);
        __syncthreads();
        if (ln < 4 && k0 + ln < POLY_K) {
            T acc = T(0);
#pragma unroll
            for (int m = 0; m < 4; ++m) acc += sm[ln * 256 + m * C + d];
            atomicAdd(gW + ((long)s * POLY_K + k0 + ln) * C + d, acc);
        }
        __syncthreads();
    }
}

template <typename T, bool NM = true>
__global__ void __launch_bounds__(256) k_prod_rev1_dir_spc(const T* __restrict__ B, const T* __restrict__ W,
                                                           const int* __restrict__ spc_off,
                                                           const int* __restrict__ spc_idx,
                                                           const int* __restrict__ spc_tiles32,
                                                           const T* __restrict__ fs, T* __restrict__ Bs,
                                                           T* __restrict__ gW, int N, int C, int S) {
    const int tot = spc_tiles32[S];
    const int by = blockIdx.y;
    if (by >= tot) return;
    int s = 0;
    while (s + 1 < S && by >= spc_tiles32[s + 1]) ++s;
    const int t = by - spc_tiles32[s];
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int nsub = gridDim.x, sub = blockIdx.x;
    const int chunk = 32 / nsub;
    const int r0 = t * 32 + sub * chunk;
    const int r1 = min(r0 + chunk, cnt);
    if (r0 >= r1) return;
    const int tid = threadIdx.x, ln = tid / C, d = tid % C, M = 256 / C;
    const long NC = (long)N * C;
    T w[POLY_K], gwa[POLY_K];
#pragma unroll
    for (int k = 0; k < POLY_K; ++k) {
        w[k] = W[((long)s * POLY_K + k) * C + d];
        gwa[k] = T(0);
    }
    for (int r = r0 + ln; r < r1; r += M) {
        const long n = spc_idx[off + r];
        const long i = n * C + d;
        T x[16], Pk[POLY_K], wf[POLY_K], G[16];
        load_x_nm<T, NM>(B, NC, n, d, C, x);
        const T f = fs[i];
        poly_p(x, Pk);
#pragma unroll
        for (int k = 0; k < POLY_K; ++k) {
            gwa[k] += f * Pk[k];
            wf[k] = w[k] * f;
        }
        poly_vjp(x, wf, G);
#pragma unroll
        for (int m = 0; m < 16; ++m) Bs[NM ? (n * 16 + m) * C + d : m * NC + i] += G[m];
    }
    __shared__ T sm[4 * 256];
#pragma unroll
    for (int k0 = 0; k0 < POLY_K; k0 += 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            sm[j * 256 + ln * C + d] = (k0 + j < POLY_K) ? gwa[k0 + j] : T(0);
        __syncthreads();
        if (ln < 4 && k0 + ln < POLY_K) {
            T acc = T(0);
#pragma unroll
            for (int m = 0; m < 4; ++m) acc += sm[ln * 256 + m * C + d];
            atomicAdd(gW + ((long)s * POLY_K + k0 + ln) * C + d, acc);
        }
        __syncthreads();
    }
}

// ============================================================================ loss (one block)
template <typename T>
__global__ void k_loss(const T* Eg, const T* core_e, const T* e_ref, const T* n_atoms, const T* w_e, const T* F,
                       const T* f_ref, const T* w_f, const T* mv, const T* volume, const T* core_s, const T* s_ref,
                       const T* w_s, const T* counts, const T* lw, T* Sg, T* Eb, T* Fb, T* Ms, T* losses,
                       const int* step, int N, int G1) {
    __shared__ double red[3][32];
    double le = 0, lf = 0, ls = 0;
    lw += 3L * step[0];
    T nr = counts[0], gr = counts[1];
    for (int g = threadIdx.x; g < G1; g += blockDim.x) {
        T E = Eg[g] + core_e[g];
        T d = (e_ref[g] - E) / n_atoms[g];
        le += (double)(w_e[g] * d * d);
        Eb[g] = lw[0] * w_e[g] * T(2) * (E - e_ref[g]) / (n_atoms[g] * n_atoms[g]) / gr;
        T S[9], Sb[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                S[3 * i + j] = T(0.5) * (mv[9 * g + 3 * i + j] + mv[9 * g + 3 * j + i]) / volume[g] + core_s[9 * g + 3 * i + j];
        for (int k = 0; k < 9; ++k) {
            T dd = s_ref[9 * g + k] - S[k];
            ls += (double)(w_s[g] * dd * dd);
            Sg[9 * g + k] = S[k];
            Sb[k] = lw[2] * w_s[g] * T(2) * (S[k] - s_ref[9 * g + k]) / (T(9) * gr);
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) Ms[9 * g + 3 * i + j] = T(0.5) * (Sb[3 * i + j] + Sb[3 * j + i]) / volume[g];
    }
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        for (int k = 0; k < 3; ++k) {
            T dd = f_ref[3 * n + k] - F[3 * n + k];
            lf += (double)(w_f[n] * dd * dd);
            Fb[3 * n + k] = lw[1] * w_f[n] * T(2) * (F[3 * n + k] - f_ref[3 * n + k]) / (T(3) * nr);
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        le += __shfl_xor_sync(0xffffffffu, le, o);
        lf += __shfl_xor_sync(0xffffffffu, lf, o);
        ls += __shfl_xor_sync(0xffffffffu, ls, o);
    }
    int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) {
        red[0][w] = le;
        red[1][w] = lf;
        red[2][w] = ls;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double a = 0, b = 0, c = 0;
        for (int k = 0; k < (int)(blockDim.x / 32); ++k) {
            a += red[0][k];
            b += red[1][k];
            c += red[2][k];
        }
        double loss = (double)lw[0] * a / (double)gr + (double)lw[1] * b / (3.0 * (double)nr) +
                      (double)lw[2] * c / (9.0 * (double)gr);
        losses[*step] = T(loss);
    }
}

// Wave 20+24+31: Multi-block coalesced k_loss_par + dynamic S_cot (LST_SCOT=1) + fused gauge-centered L_ei (HAS_EI)
template <typename T, bool HAS_EI = false>
__global__ void __launch_bounds__(256) k_loss_par(const T* __restrict__ Eg, const T* __restrict__ core_e,
                                                  const T* __restrict__ e_ref, const T* __restrict__ n_atoms,
                                                  const T* __restrict__ w_e, const T* __restrict__ F,
                                                  const T* __restrict__ f_ref, const T* __restrict__ w_f,
                                                  const T* __restrict__ mv, const T* __restrict__ volume,
                                                  const T* __restrict__ core_s, const T* __restrict__ s_ref,
                                                  const T* __restrict__ w_s, const T* __restrict__ counts,
                                                  const T* __restrict__ lw, T* __restrict__ Sg, T* __restrict__ Eb,
                                                  T* __restrict__ Fb, T* __restrict__ Ms, T* __restrict__ losses,
                                                  const int* __restrict__ step, double* __restrict__ acc4,
                                                  int* __restrict__ done_cnt, int N, int G1,
                                                  unsigned int* __restrict__ d_fmax = nullptr,
                                                  float* __restrict__ d_scot = nullptr,
                                                  const T* __restrict__ ne = nullptr,
                                                  const T* __restrict__ ei_ref = nullptr,
                                                  const int* __restrict__ ngraph = nullptr,
                                                  const int* __restrict__ spc = nullptr,
                                                  const double* __restrict__ d_wei = nullptr, double const_wei = 0.0,
                                                  T* __restrict__ Enb = nullptr, int ei_B = 0, int ei_S = 0,
                                                  int ei_fast = 0) {
    __shared__ double red[4][8];
    __shared__ float red_m[8];
    const int tid = threadIdx.x;
    const T* lwp = lw + 3L * step[0];
    const T nr = counts[0], gr = counts[1];
    double le = 0.0, ls = 0.0, lf = 0.0, lei = 0.0;
    float fmax_t = 0.f;
    if (blockIdx.x == 0) {
        for (int g = tid; g < G1; g += 256) {
            const T E = Eg[g] + core_e[g];
            const T d = (e_ref[g] - E) / n_atoms[g];
            le += (double)(w_e[g] * d * d);
            const T ebv = lwp[0] * w_e[g] * T(2) * (E - e_ref[g]) / (n_atoms[g] * n_atoms[g]) / gr;
            Eb[g] = ebv;
            if (d_fmax) fmax_t = fmaxf(fmax_t, 10.f * fabsf((float)ebv));
            T S[9], Sb[9];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    S[3 * i + j] = T(0.5) * (mv[9 * g + 3 * i + j] + mv[9 * g + 3 * j + i]) / volume[g] +
                                   core_s[9 * g + 3 * i + j];
            for (int k = 0; k < 9; ++k) {
                const T dd = s_ref[9 * g + k] - S[k];
                ls += (double)(w_s[g] * dd * dd);
                Sg[9 * g + k] = S[k];
                Sb[k] = lwp[2] * w_s[g] * T(2) * (S[k] - s_ref[9 * g + k]) / (T(9) * gr);
            }
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    Ms[9 * g + 3 * i + j] = T(0.5) * (Sb[3 * i + j] + Sb[3 * j + i]) / volume[g];
        }
    }
    const int ei_blk = (HAS_EI && ei_fast && gridDim.x > 1) ? ((int)gridDim.x - 1) : ((gridDim.x > 1) ? 1 : 0);
    if constexpr (HAS_EI) {
        if ((int)blockIdx.x == ei_blk) {
            const double w_ei = d_wei ? d_wei[step[0]] : const_wei;
            if (w_ei <= 0.0) {
                for (int n = tid; n < N; n += 256) Enb[n] = T(0);
            } else {
                const int G = (int)gr;
                const int BS = ei_B * ei_S;
                const int w_ei_idx = tid >> 5, lane_ei = tid & 31;
                const double inv_nr = ((double)nr > 0.0) ? (1.0 / (double)nr) : 0.0;
                const double gcoef = w_ei * 2.0 * inv_nr;
                if (G == 1 && ei_S <= 4) {
                    // Wave 35+37 fast path (b=1): zero smem double-atomics, 1-pass reg cache for N<=512, FP32 warp shuffles!
                    __shared__ double sw_sum[8][4];
                    __shared__ int sw_cnt[8][4];
                    __shared__ double s_mean[4];
                    if constexpr (std::is_same<T, float>::value) {
                        if (ei_fast) {
                            float rsum[4] = {0.f, 0.f, 0.f, 0.f};
                            int rcnt[4] = {0, 0, 0, 0};
                            float r_c[2] = {0.f, 0.f};
                            int s_c[2] = {-1, -1};
                            if (N <= 512) {
#pragma unroll
                                for (int kk = 0; kk < 2; ++kk) {
                                    const int n = tid + kk * 256;
                                    if (n < N && w_f[n] > 0.f) {
                                        const int g = ngraph[n], s = spc[n];
                                        if (g == 0 && s >= 0 && s < ei_S) {
                                            const float rv = ne[n] - ei_ref[n];
                                            rsum[s] += rv;
                                            rcnt[s] += 1;
                                            r_c[kk] = rv;
                                            s_c[kk] = s;
                                        }
                                    }
                                }
                            } else {
                                for (int n = tid; n < N; n += 256) {
                                    if (w_f[n] > 0.f) {
                                        const int g = ngraph[n], s = spc[n];
                                        if (g == 0 && s >= 0 && s < ei_S) {
                                            rsum[s] += ne[n] - ei_ref[n];
                                            rcnt[s] += 1;
                                        }
                                    }
                                }
                            }
                            for (int s = 0; s < ei_S; ++s) {
                                float sv = rsum[s];
                                int cv = rcnt[s];
#pragma unroll
                                for (int o = 16; o > 0; o >>= 1) {
                                    sv += __shfl_xor_sync(0xffffffffu, sv, o);
                                    cv += __shfl_xor_sync(0xffffffffu, cv, o);
                                }
                                if (lane_ei == 0) {
                                    sw_sum[w_ei_idx][s] = (double)sv;
                                    sw_cnt[w_ei_idx][s] = cv;
                                }
                            }
                            __syncthreads();
                            if (tid < ei_S) {
                                double ts = 0.0;
                                int tc = 0;
#pragma unroll
                                for (int ww = 0; ww < 8; ++ww) {
                                    ts += sw_sum[ww][tid];
                                    tc += sw_cnt[ww][tid];
                                }
                                s_mean[tid] = (tc > 0) ? (ts / (double)tc) : 0.0;
                            }
                            __syncthreads();
                            const float fgcoef = (float)gcoef;
                            float loc_lf = 0.f;
                            if (N <= 512) {
#pragma unroll
                                for (int kk = 0; kk < 2; ++kk) {
                                    const int n = tid + kk * 256;
                                    if (n < N) {
                                        if (s_c[kk] >= 0) {
                                            const float d = r_c[kk] - (float)s_mean[s_c[kk]];
                                            loc_lf += d * d;
                                            const float gv = fgcoef * d;
                                            Enb[n] = gv;
                                            if (d_fmax) fmax_t = fmaxf(fmax_t, fabsf(gv));
                                        } else {
                                            Enb[n] = 0.f;
                                        }
                                    }
                                }
                            } else {
                                for (int n = tid; n < N; n += 256) {
                                    const int g = ngraph[n], s = spc[n];
                                    if (g == 0 && s >= 0 && s < ei_S && w_f[n] > 0.f) {
                                        const float d = (ne[n] - ei_ref[n]) - (float)s_mean[s];
                                        loc_lf += d * d;
                                        const float gv = fgcoef * d;
                                        Enb[n] = gv;
                                        if (d_fmax) fmax_t = fmaxf(fmax_t, fabsf(gv));
                                    } else {
                                        Enb[n] = 0.f;
                                    }
                                }
                            }
                            lei = w_ei * inv_nr * (double)loc_lf;
                        } else {
                    double rsum[4] = {0.0, 0.0, 0.0, 0.0};
                    int rcnt[4] = {0, 0, 0, 0};
                    double r_c[2] = {0.0, 0.0};
                    int s_c[2] = {-1, -1};
                    if (N <= 512) {
#pragma unroll
                        for (int kk = 0; kk < 2; ++kk) {
                            const int n = tid + kk * 256;
                            if (n < N && w_f[n] > T(0)) {
                                const int g = ngraph[n], s = spc[n];
                                if (g == 0 && s >= 0 && s < ei_S) {
                                    const double rv = (double)ne[n] - (double)ei_ref[n];
                                    rsum[s] += rv;
                                    rcnt[s] += 1;
                                    r_c[kk] = rv;
                                    s_c[kk] = s;
                                }
                            }
                        }
                    } else {
                        for (int n = tid; n < N; n += 256) {
                            if (w_f[n] > T(0)) {
                                const int g = ngraph[n], s = spc[n];
                                if (g == 0 && s >= 0 && s < ei_S) {
                                    rsum[s] += (double)ne[n] - (double)ei_ref[n];
                                    rcnt[s] += 1;
                                }
                            }
                        }
                    }
                    for (int s = 0; s < ei_S; ++s) {
                        double sv = rsum[s];
                        int cv = rcnt[s];
#pragma unroll
                        for (int o = 16; o > 0; o >>= 1) {
                            sv += __shfl_xor_sync(0xffffffffu, sv, o);
                            cv += __shfl_xor_sync(0xffffffffu, cv, o);
                        }
                        if (lane_ei == 0) {
                            sw_sum[w_ei_idx][s] = sv;
                            sw_cnt[w_ei_idx][s] = cv;
                        }
                    }
                    __syncthreads();
                    if (tid < ei_S) {
                        double ts = 0.0;
                        int tc = 0;
#pragma unroll
                        for (int ww = 0; ww < 8; ++ww) {
                            ts += sw_sum[ww][tid];
                            tc += sw_cnt[ww][tid];
                        }
                        s_mean[tid] = (tc > 0) ? (ts / (double)tc) : 0.0;
                    }
                    __syncthreads();
                    double loc_l = 0.0;
                    if (N <= 512) {
#pragma unroll
                        for (int kk = 0; kk < 2; ++kk) {
                            const int n = tid + kk * 256;
                            if (n < N) {
                                if (s_c[kk] >= 0) {
                                    const double d = r_c[kk] - s_mean[s_c[kk]];
                                    loc_l += d * d;
                                    const T gv = T(gcoef * d);
                                    Enb[n] = gv;
                                    if (d_fmax) fmax_t = fmaxf(fmax_t, fabsf((float)gv));
                                } else {
                                    Enb[n] = T(0);
                                }
                            }
                        }
                    } else {
                        for (int n = tid; n < N; n += 256) {
                            const int g = ngraph[n], s = spc[n];
                            if (g == 0 && s >= 0 && s < ei_S && w_f[n] > T(0)) {
                                const double d = ((double)ne[n] - (double)ei_ref[n]) - s_mean[s];
                                loc_l += d * d;
                                const T gv = T(gcoef * d);
                                Enb[n] = gv;
                                if (d_fmax) fmax_t = fmaxf(fmax_t, fabsf((float)gv));
                            } else {
                                Enb[n] = T(0);
                            }
                        }
                    }
                    lei = w_ei * inv_nr * loc_l;
                        }
                    } else {
                    double rsum[4] = {0.0, 0.0, 0.0, 0.0};
                    int rcnt[4] = {0, 0, 0, 0};
                    for (int n = tid; n < N; n += 256) {
                        if (w_f[n] > T(0)) {
                            const int g = ngraph[n], s = spc[n];
                            if (g == 0 && s >= 0 && s < ei_S) {
                                rsum[s] += (double)ne[n] - (double)ei_ref[n];
                                rcnt[s] += 1;
                            }
                        }
                    }
                    for (int s = 0; s < ei_S; ++s) {
                        double sv = rsum[s];
                        int cv = rcnt[s];
#pragma unroll
                        for (int o = 16; o > 0; o >>= 1) {
                            sv += __shfl_xor_sync(0xffffffffu, sv, o);
                            cv += __shfl_xor_sync(0xffffffffu, cv, o);
                        }
                        if (lane_ei == 0) {
                            sw_sum[w_ei_idx][s] = sv;
                            sw_cnt[w_ei_idx][s] = cv;
                        }
                    }
                    __syncthreads();
                    if (tid < ei_S) {
                        double ts = 0.0;
                        int tc = 0;
#pragma unroll
                        for (int ww = 0; ww < 8; ++ww) {
                            ts += sw_sum[ww][tid];
                            tc += sw_cnt[ww][tid];
                        }
                        s_mean[tid] = (tc > 0) ? (ts / (double)tc) : 0.0;
                    }
                    __syncthreads();
                    double loc_l = 0.0;
                    for (int n = tid; n < N; n += 256) {
                        const int g = ngraph[n], s = spc[n];
                        if (g == 0 && s >= 0 && s < ei_S && w_f[n] > T(0)) {
                            const double d = ((double)ne[n] - (double)ei_ref[n]) - s_mean[s];
                            loc_l += d * d;
                            const T gv = T(gcoef * d);
                            Enb[n] = gv;
                            if (d_fmax) fmax_t = fmaxf(fmax_t, fabsf((float)gv));
                        } else {
                            Enb[n] = T(0);
                        }
                    }
                    lei = w_ei * inv_nr * loc_l;
                    }
                } else if (BS <= 16) {
                    // Wave 35 fast path (b=4, BS<=16): warp-private smem accumulators -> ZERO smem double atomics!
                    __shared__ double sw_sum[8][16];
                    __shared__ int sw_cnt[8][16];
                    __shared__ double s_mean[16];
                    if (lane_ei < BS) {
                        sw_sum[w_ei_idx][lane_ei] = 0.0;
                        sw_cnt[w_ei_idx][lane_ei] = 0;
                    }
                    __syncwarp();
                    for (int base = 0; base < N; base += 256) {
                        const int n = base + tid;
                        int key = -1;
                        double r = 0.0;
                        if (n < N && w_f[n] > T(0)) {
                            const int g = ngraph[n], s = spc[n];
                            if (g >= 0 && g < G && s >= 0 && s < ei_S) {
                                key = g * ei_S + s;
                                r = (double)ne[n] - (double)ei_ref[n];
                            }
                        }
                        unsigned todo = __ballot_sync(0xffffffffu, key >= 0);
                        while (todo) {
                            const int lead = __ffs(todo) - 1;
                            const int k_cur = __shfl_sync(0xffffffffu, key, lead);
                            const unsigned grp = todo & __ballot_sync(0xffffffffu, key == k_cur);
                            double v = (key == k_cur) ? r : 0.0;
#pragma unroll
                            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                            if (lane_ei == lead) {
                                sw_sum[w_ei_idx][k_cur] += v;
                                sw_cnt[w_ei_idx][k_cur] += __popc(grp);
                            }
                            todo &= ~grp;
                        }
                    }
                    __syncthreads();
                    if (tid < BS) {
                        double ts = 0.0;
                        int tc = 0;
#pragma unroll
                        for (int ww = 0; ww < 8; ++ww) {
                            ts += sw_sum[ww][tid];
                            tc += sw_cnt[ww][tid];
                        }
                        s_mean[tid] = (tc > 0) ? (ts / (double)tc) : 0.0;
                    }
                    __syncthreads();
                    double loc_l = 0.0;
                    for (int n = tid; n < N; n += 256) {
                        const int g = ngraph[n], s = spc[n];
                        if (g >= 0 && g < G && s >= 0 && s < ei_S && w_f[n] > T(0)) {
                            const double d = ((double)ne[n] - (double)ei_ref[n]) - s_mean[g * ei_S + s];
                            loc_l += d * d;
                            const T gv = T(gcoef * d);
                            Enb[n] = gv;
                            if (d_fmax) fmax_t = fmaxf(fmax_t, fabsf((float)gv));
                        } else {
                            Enb[n] = T(0);
                        }
                    }
                    lei = w_ei * inv_nr * loc_l;
                } else {
                    extern __shared__ unsigned char smem_raw[];
                    double* s_sum = reinterpret_cast<double*>(smem_raw);
                    int* s_cnt = reinterpret_cast<int*>(s_sum + BS);
                    for (int k = tid; k < BS; k += 256) {
                        s_sum[k] = 0.0;
                        s_cnt[k] = 0;
                    }
                    __syncthreads();
                    for (int base = 0; base < N; base += 256) {
                        const int n = base + tid;
                        int key = -1;
                        double r = 0.0;
                        if (n < N) {
                            const int g = ngraph[n];
                            const int s = spc[n];
                            if (g >= 0 && g < G && s >= 0 && s < ei_S && w_f[n] > T(0)) {
                                key = g * ei_S + s;
                                r = (double)ne[n] - (double)ei_ref[n];
                            }
                        }
                        unsigned todo = __ballot_sync(0xffffffffu, key >= 0);
                        while (todo) {
                            const int lead = __ffs(todo) - 1;
                            const int k_cur = __shfl_sync(0xffffffffu, key, lead);
                            const unsigned grp = todo & __ballot_sync(0xffffffffu, key == k_cur);
                            double v = (key == k_cur) ? r : 0.0;
                            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                            if (lane_ei == lead) {
                                atomicAdd(s_sum + k_cur, v);
                                atomicAdd(s_cnt + k_cur, __popc(grp));
                            }
                            todo &= ~grp;
                        }
                    }
                    __syncthreads();
                    for (int k = tid; k < BS; k += 256) {
                        s_sum[k] = (s_cnt[k] > 0) ? (s_sum[k] / (double)s_cnt[k]) : 0.0;
                    }
                    __syncthreads();
                    double loc_l = 0.0;
                    for (int n = tid; n < N; n += 256) {
                        const int g = ngraph[n];
                        const int s = spc[n];
                        if (g >= 0 && g < G && s >= 0 && s < ei_S && w_f[n] > T(0)) {
                            const double d = ((double)ne[n] - (double)ei_ref[n]) - s_sum[g * ei_S + s];
                            loc_l += d * d;
                            const T gv = T(gcoef * d);
                            Enb[n] = gv;
                            if (d_fmax) fmax_t = fmaxf(fmax_t, fabsf((float)gv));
                        } else {
                            Enb[n] = T(0);
                        }
                    }
                    lei = w_ei * inv_nr * loc_l;
                }
            }
        }
    }
    const T fcoef = lwp[1] * T(2) / (T(3) * nr);
    const int N3 = 3 * N;
    const int f_nblk = (HAS_EI && ei_fast && gridDim.x > 1) ? ((int)gridDim.x - 1) : (int)gridDim.x;
    if ((int)blockIdx.x < f_nblk) {
        for (int idx = blockIdx.x * 256 + tid; idx < N3; idx += f_nblk * 256) {
            const int n = idx / 3;
            const T wf = w_f[n];
            const T dd = f_ref[idx] - F[idx];
            lf += (double)(wf * dd * dd);
            const T fbv = fcoef * wf * (-dd);
            Fb[idx] = fbv;
            if (d_fmax) fmax_t = fmaxf(fmax_t, fabsf((float)fbv));
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        if ((int)blockIdx.x < f_nblk) lf += __shfl_xor_sync(0xffffffffu, lf, o);
        if (d_fmax) fmax_t = fmaxf(fmax_t, __shfl_xor_sync(0xffffffffu, fmax_t, o));
        if (blockIdx.x == 0) {
            le += __shfl_xor_sync(0xffffffffu, le, o);
            ls += __shfl_xor_sync(0xffffffffu, ls, o);
        }
        if constexpr (HAS_EI) {
            if ((int)blockIdx.x == ei_blk) lei += __shfl_xor_sync(0xffffffffu, lei, o);
        }
    }
    const int w = tid >> 5, lane = tid & 31;
    if (lane == 0) {
        red[1][w] = lf;
        if (d_fmax) red_m[w] = fmax_t;
        if (blockIdx.x == 0) {
            red[0][w] = le;
            red[2][w] = ls;
        }
        if constexpr (HAS_EI) {
            if ((int)blockIdx.x == ei_blk) red[3][w] = lei;
        }
    }
    __syncthreads();
    if (tid == 0) {
        if ((int)blockIdx.x < f_nblk) {
            double bf = 0.0;
            for (int k = 0; k < 8; ++k) bf += red[1][k];
            atomicAdd(acc4 + 1, bf);
        }
        if (d_fmax) {
            float bm = 0.f;
            for (int k = 0; k < 8; ++k) bm = fmaxf(bm, red_m[k]);
            atomicMax(d_fmax, __float_as_uint(bm));
        }
        if (blockIdx.x == 0) {
            double be = 0.0, bs = 0.0;
            for (int k = 0; k < 8; ++k) {
                be += red[0][k];
                bs += red[2][k];
            }
            atomicAdd(acc4 + 0, be);
            atomicAdd(acc4 + 2, bs);
        }
        if constexpr (HAS_EI) {
            if ((int)blockIdx.x == ei_blk) {
                double bei = 0.0;
                for (int k = 0; k < 8; ++k) bei += red[3][k];
                atomicAdd(acc4 + 3, bei);
            }
        }
        __threadfence();
        const int prev = atomicAdd(done_cnt, 1);
        if (prev == (int)gridDim.x - 1) {
            double loss = (double)lwp[0] * acc4[0] / (double)gr +
                          (double)lwp[1] * acc4[1] / (3.0 * (double)nr) +
                          (double)lwp[2] * acc4[2] / (9.0 * (double)gr);
            if constexpr (HAS_EI) {
                loss += acc4[3];
                acc4[3] = 0.0;
            }
            losses[*step] = T(loss);
            acc4[0] = 0.0; acc4[1] = 0.0; acc4[2] = 0.0;
            if (d_fmax && d_scot) {
                const float mfb = __uint_as_float(*d_fmax);
                const float sc = (mfb > 1e-12f) ? fminf(fmaxf(0.25f / mfb, 1.f), 32.f) : 1.f;
                d_scot[0] = sc;
                d_scot[1] = 1.f / sc;
                *d_fmax = 0u;
            }
            *done_cnt = 0;
        }
    }
}

// Wave 36: Single-CTA fast loss kernel for N <= 512, B == 1, S <= 4 (LST_LOSS_1BLK=1).
// Eliminates global acc4 FP64 atomics, atomicMax(d_fmax), __threadfence(), and done_cnt grid barrier (~8.5 -> ~2.5 us!).
template <typename T, bool HAS_EI = false>
__global__ void __launch_bounds__(256) k_loss_1blk(
    const T* __restrict__ Eg, const T* __restrict__ core_e, const T* __restrict__ e_ref,
    const T* __restrict__ n_atoms, const T* __restrict__ w_e, const T* __restrict__ F,
    const T* __restrict__ f_ref, const T* __restrict__ w_f, const T* __restrict__ mv,
    const T* __restrict__ volume, const T* __restrict__ core_s, const T* __restrict__ s_ref,
    const T* __restrict__ w_s, const T* __restrict__ counts, const T* __restrict__ lw,
    T* __restrict__ Sg, T* __restrict__ Eb, T* __restrict__ Fb, T* __restrict__ Ms,
    T* __restrict__ losses, const int* __restrict__ step, int N, int G1,
    float* __restrict__ d_scot = nullptr,
    const T* __restrict__ ne = nullptr, const T* __restrict__ ei_ref = nullptr,
    const int* __restrict__ ngraph = nullptr, const int* __restrict__ spc = nullptr,
    const double* __restrict__ d_wei = nullptr, double const_wei = 0.0,
    T* __restrict__ Enb = nullptr, int ei_S = 0) {
    __shared__ double red_f[8];
    __shared__ double red_ei[8];
    __shared__ float red_m[8];
    __shared__ double s_le_ls[2];
    __shared__ double sw_sum[8][4];
    __shared__ int sw_cnt[8][4];
    __shared__ double s_mean[4];

    const int tid = threadIdx.x;
    const int w = tid >> 5, lane = tid & 31;
    const int s_cur = step[0];
    const T* lwp = lw + 3L * s_cur;
    const T nr = counts[0], gr = counts[1];
    double le = 0.0, ls = 0.0, lf = 0.0, lei = 0.0;
    float fmax_t = 0.f;

    if (tid < G1) {
        const int g = tid;
        const T E = Eg[g] + core_e[g];
        const T d = (e_ref[g] - E) / n_atoms[g];
        le = (double)(w_e[g] * d * d);
        const T ebv = lwp[0] * w_e[g] * T(2) * (E - e_ref[g]) / (n_atoms[g] * n_atoms[g]) / gr;
        Eb[g] = ebv;
        if (d_scot) fmax_t = fmaxf(fmax_t, 10.f * fabsf((float)ebv));
        T S[9], Sb[9];
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int j = 0; j < 3; ++j)
                S[3 * i + j] = T(0.5) * (mv[9 * g + 3 * i + j] + mv[9 * g + 3 * j + i]) / volume[g] +
                               core_s[9 * g + 3 * i + j];
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            const T dd = s_ref[9 * g + k] - S[k];
            ls += (double)(w_s[g] * dd * dd);
            Sg[9 * g + k] = S[k];
            Sb[k] = lwp[2] * w_s[g] * T(2) * (S[k] - s_ref[9 * g + k]) / (T(9) * gr);
        }
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int j = 0; j < 3; ++j)
                Ms[9 * g + 3 * i + j] = T(0.5) * (Sb[3 * i + j] + Sb[3 * j + i]) / volume[g];
    }
    if (w == 0) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            le += __shfl_xor_sync(0xffffffffu, le, o);
            ls += __shfl_xor_sync(0xffffffffu, ls, o);
        }
        if (lane == 0) {
            s_le_ls[0] = le;
            s_le_ls[1] = ls;
        }
    }

    const T fcoef = lwp[1] * T(2) / (T(3) * nr);
    const int N3 = 3 * N;
    for (int idx = tid; idx < N3; idx += 256) {
        const int n = idx / 3;
        const T wf = w_f[n];
        const T dd = f_ref[idx] - F[idx];
        lf += (double)(wf * dd * dd);
        const T fbv = fcoef * wf * (-dd);
        Fb[idx] = fbv;
        if (d_scot) fmax_t = fmaxf(fmax_t, fabsf((float)fbv));
    }

    if constexpr (HAS_EI) {
        const double w_ei = d_wei ? d_wei[s_cur] : const_wei;
        if (w_ei <= 0.0) {
            for (int n = tid; n < N; n += 256) Enb[n] = T(0);
        } else {
            const double inv_nr = ((double)nr > 0.0) ? (1.0 / (double)nr) : 0.0;
            const double gcoef = w_ei * 2.0 * inv_nr;
            double rsum[4] = {0.0, 0.0, 0.0, 0.0};
            int rcnt[4] = {0, 0, 0, 0};
            double r_c[2] = {0.0, 0.0};
            int s_c[2] = {-1, -1};
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int n = tid + kk * 256;
                if (n < N && w_f[n] > T(0)) {
                    const int g = ngraph[n], s = spc[n];
                    if (g == 0 && s >= 0 && s < ei_S) {
                        const double rv = (double)ne[n] - (double)ei_ref[n];
                        rsum[s] += rv;
                        rcnt[s] += 1;
                        r_c[kk] = rv;
                        s_c[kk] = s;
                    }
                }
            }
            for (int s = 0; s < ei_S; ++s) {
                double sv = rsum[s];
                int cv = rcnt[s];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    sv += __shfl_xor_sync(0xffffffffu, sv, o);
                    cv += __shfl_xor_sync(0xffffffffu, cv, o);
                }
                if (lane == 0) {
                    sw_sum[w][s] = sv;
                    sw_cnt[w][s] = cv;
                }
            }
            __syncthreads();
            if (tid < ei_S) {
                double ts = 0.0;
                int tc = 0;
#pragma unroll
                for (int ww = 0; ww < 8; ++ww) {
                    ts += sw_sum[ww][tid];
                    tc += sw_cnt[ww][tid];
                }
                s_mean[tid] = (tc > 0) ? (ts / (double)tc) : 0.0;
            }
            __syncthreads();
            double loc_l = 0.0;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int n = tid + kk * 256;
                if (n < N) {
                    if (s_c[kk] >= 0) {
                        const double d = r_c[kk] - s_mean[s_c[kk]];
                        loc_l += d * d;
                        const T gv = T(gcoef * d);
                        Enb[n] = gv;
                        if (d_scot) fmax_t = fmaxf(fmax_t, fabsf((float)gv));
                    } else {
                        Enb[n] = T(0);
                    }
                }
            }
            lei = w_ei * inv_nr * loc_l;
        }
    }

#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        lf += __shfl_xor_sync(0xffffffffu, lf, o);
        if constexpr (HAS_EI) lei += __shfl_xor_sync(0xffffffffu, lei, o);
        if (d_scot) fmax_t = fmaxf(fmax_t, __shfl_xor_sync(0xffffffffu, fmax_t, o));
    }
    if (lane == 0) {
        red_f[w] = lf;
        if constexpr (HAS_EI) red_ei[w] = lei;
        if (d_scot) red_m[w] = fmax_t;
    }
    __syncthreads();
    if (tid == 0) {
        double tot_f = 0.0, tot_ei = 0.0;
        float bm = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            tot_f += red_f[k];
            if constexpr (HAS_EI) tot_ei += red_ei[k];
            if (d_scot) bm = fmaxf(bm, red_m[k]);
        }
        double loss = (double)lwp[0] * s_le_ls[0] / (double)gr +
                      (double)lwp[1] * tot_f / (3.0 * (double)nr) +
                      (double)lwp[2] * s_le_ls[1] / (9.0 * (double)gr);
        if constexpr (HAS_EI) loss += tot_ei;
        losses[s_cur] = T(loss);
        if (d_scot) {
            const float sc = (bm > 1e-12f) ? fminf(fmaxf(0.25f / bm, 1.f), 32.f) : 1.f;
            d_scot[0] = sc;
            d_scot[1] = 1.f / sc;
        }
    }
}

// Wave 38: Zero-atomic 2-CTA fast loss kernel for N <= 512, B == 1, G1 <= 32, S <= 4 (LST_LOSS_2BLK=1).
// CTA 0 runs E+S+F (plain stores to acc4[0..2] & fmax[0]); CTA 1 runs HAS_EI in parallel (plain stores to acc4[3] & fmax[1]).
// Eliminates ALL FP64 atomicAdd(acc4) and atomicMax(d_fmax) contention (only 1 done_cnt atomic across 2 CTAs; 0 atomics when !HAS_EI!).
template <typename T, bool HAS_EI = false>
__global__ void __launch_bounds__(256) k_loss_2blk(
    const T* __restrict__ Eg, const T* __restrict__ core_e, const T* __restrict__ e_ref,
    const T* __restrict__ n_atoms, const T* __restrict__ w_e, const T* __restrict__ F,
    const T* __restrict__ f_ref, const T* __restrict__ w_f, const T* __restrict__ mv,
    const T* __restrict__ volume, const T* __restrict__ core_s, const T* __restrict__ s_ref,
    const T* __restrict__ w_s, const T* __restrict__ counts, const T* __restrict__ lw,
    T* __restrict__ Sg, T* __restrict__ Eb, T* __restrict__ Fb, T* __restrict__ Ms,
    T* __restrict__ losses, const int* __restrict__ step, double* __restrict__ acc4,
    int* __restrict__ done_cnt, int N, int G1,
    unsigned int* __restrict__ d_fmax = nullptr, float* __restrict__ d_scot = nullptr,
    const T* __restrict__ ne = nullptr, const T* __restrict__ ei_ref = nullptr,
    const int* __restrict__ ngraph = nullptr, const int* __restrict__ spc = nullptr,
    const double* __restrict__ d_wei = nullptr, double const_wei = 0.0,
    T* __restrict__ Enb = nullptr, int ei_S = 0) {
    __shared__ double red[8];
    __shared__ float red_m[8];
    const int tid = threadIdx.x;
    const int w = tid >> 5, lane = tid & 31;
    const int s_cur = step[0];
    const T* lwp = lw + 3L * s_cur;
    const T nr = counts[0], gr = counts[1];

    if (blockIdx.x == 0) {
        double le = 0.0, ls = 0.0, lf = 0.0;
        float fmax_t = 0.f;
        if (tid < G1) {
            const int g = tid;
            const T E = Eg[g] + core_e[g];
            const T d = (e_ref[g] - E) / n_atoms[g];
            le = (double)(w_e[g] * d * d);
            const T ebv = lwp[0] * w_e[g] * T(2) * (E - e_ref[g]) / (n_atoms[g] * n_atoms[g]) / gr;
            Eb[g] = ebv;
            if (d_scot) fmax_t = fmaxf(fmax_t, 10.f * fabsf((float)ebv));
            T S[9], Sb[9];
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j)
                    S[3 * i + j] = T(0.5) * (mv[9 * g + 3 * i + j] + mv[9 * g + 3 * j + i]) / volume[g] +
                                   core_s[9 * g + 3 * i + j];
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                const T dd = s_ref[9 * g + k] - S[k];
                ls += (double)(w_s[g] * dd * dd);
                Sg[9 * g + k] = S[k];
                Sb[k] = lwp[2] * w_s[g] * T(2) * (S[k] - s_ref[9 * g + k]) / (T(9) * gr);
            }
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j)
                    Ms[9 * g + 3 * i + j] = T(0.5) * (Sb[3 * i + j] + Sb[3 * j + i]) / volume[g];
        }
        if (w == 0 && G1 > 1) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                le += __shfl_xor_sync(0xffffffffu, le, o);
                ls += __shfl_xor_sync(0xffffffffu, ls, o);
            }
        }
        const T fcoef = lwp[1] * T(2) / (T(3) * nr);
        const int N3 = 3 * N;
        for (int idx = tid; idx < N3; idx += 256) {
            const int n = idx / 3;
            const T wf = w_f[n];
            const T dd = f_ref[idx] - F[idx];
            lf += (double)(wf * dd * dd);
            const T fbv = fcoef * wf * (-dd);
            Fb[idx] = fbv;
            if (d_scot) fmax_t = fmaxf(fmax_t, fabsf((float)fbv));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            lf += __shfl_xor_sync(0xffffffffu, lf, o);
            if (d_scot) fmax_t = fmaxf(fmax_t, __shfl_xor_sync(0xffffffffu, fmax_t, o));
        }
        if (lane == 0) {
            red[w] = lf;
            if (d_scot) red_m[w] = fmax_t;
        }
        __syncthreads();
        if (tid == 0) {
            double bf = 0.0;
            float bm = 0.f;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                bf += red[k];
                if (d_scot) bm = fmaxf(bm, red_m[k]);
            }
            if constexpr (!HAS_EI) {
                const double loss = (double)lwp[0] * le / (double)gr +
                                    (double)lwp[1] * bf / (3.0 * (double)nr) +
                                    (double)lwp[2] * ls / (9.0 * (double)gr);
                losses[s_cur] = T(loss);
                if (d_scot) {
                    const float sc = (bm > 1e-12f) ? fminf(fmaxf(0.25f / bm, 1.f), 32.f) : 1.f;
                    d_scot[0] = sc;
                    d_scot[1] = 1.f / sc;
                }
            } else {
                acc4[0] = le;
                acc4[1] = bf;
                acc4[2] = ls;
                if (d_fmax) d_fmax[0] = __float_as_uint(bm);
                __threadfence();
                if (atomicAdd(done_cnt, 1) == 1) {
                    const double loss = (double)lwp[0] * le / (double)gr +
                                        (double)lwp[1] * bf / (3.0 * (double)nr) +
                                        (double)lwp[2] * ls / (9.0 * (double)gr) + acc4[3];
                    losses[s_cur] = T(loss);
                    acc4[0] = 0.0; acc4[1] = 0.0; acc4[2] = 0.0; acc4[3] = 0.0;
                    if (d_fmax && d_scot) {
                        const float mfb = fmaxf(bm, __uint_as_float(d_fmax[1]));
                        const float sc = (mfb > 1e-12f) ? fminf(fmaxf(0.25f / mfb, 1.f), 32.f) : 1.f;
                        d_scot[0] = sc;
                        d_scot[1] = 1.f / sc;
                        d_fmax[0] = 0u;
                        d_fmax[1] = 0u;
                    }
                    *done_cnt = 0;
                }
            }
        }
        return;
    }

    // CTA 1 (only launched when HAS_EI == true): 100% parallel gauge-centered L_ei with FP32 warp shuffles!
    if constexpr (HAS_EI) {
        const double w_ei = d_wei ? d_wei[s_cur] : const_wei;
        double bei = 0.0;
        float bm = 0.f;
        if (w_ei <= 0.0) {
            for (int n = tid; n < N; n += 256) Enb[n] = T(0);
        } else {
            __shared__ double sw_sum[8][4];
            __shared__ int sw_cnt[8][4];
            __shared__ double s_mean[4];
            const double inv_nr = ((double)nr > 0.0) ? (1.0 / (double)nr) : 0.0;
            const double gcoef = w_ei * 2.0 * inv_nr;
            double lei = 0.0;
            float fmax_t = 0.f;
            if constexpr (std::is_same<T, float>::value) {
                float rsum[4] = {0.f, 0.f, 0.f, 0.f};
                int rcnt[4] = {0, 0, 0, 0};
                float r_c[2] = {0.f, 0.f};
                int s_c[2] = {-1, -1};
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int n = tid + kk * 256;
                    if (n < N && w_f[n] > 0.f) {
                        const int g = ngraph[n], s = spc[n];
                        if (g == 0 && s >= 0 && s < ei_S) {
                            const float rv = ne[n] - ei_ref[n];
                            rsum[s] += rv;
                            rcnt[s] += 1;
                            r_c[kk] = rv;
                            s_c[kk] = s;
                        }
                    }
                }
                for (int s = 0; s < ei_S; ++s) {
                    float sv = rsum[s];
                    int cv = rcnt[s];
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) {
                        sv += __shfl_xor_sync(0xffffffffu, sv, o);
                        cv += __shfl_xor_sync(0xffffffffu, cv, o);
                    }
                    if (lane == 0) {
                        sw_sum[w][s] = (double)sv;
                        sw_cnt[w][s] = cv;
                    }
                }
                __syncthreads();
                if (tid < ei_S) {
                    double ts = 0.0;
                    int tc = 0;
#pragma unroll
                    for (int ww = 0; ww < 8; ++ww) {
                        ts += sw_sum[ww][tid];
                        tc += sw_cnt[ww][tid];
                    }
                    s_mean[tid] = (tc > 0) ? (ts / (double)tc) : 0.0;
                }
                __syncthreads();
                const float fgcoef = (float)gcoef;
                float loc_lf = 0.f;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int n = tid + kk * 256;
                    if (n < N) {
                        if (s_c[kk] >= 0) {
                            const float d = r_c[kk] - (float)s_mean[s_c[kk]];
                            loc_lf += d * d;
                            const float gv = fgcoef * d;
                            Enb[n] = gv;
                            if (d_scot) fmax_t = fmaxf(fmax_t, fabsf(gv));
                        } else {
                            Enb[n] = 0.f;
                        }
                    }
                }
                lei = w_ei * inv_nr * (double)loc_lf;
            } else {
                double rsum[4] = {0.0, 0.0, 0.0, 0.0};
                int rcnt[4] = {0, 0, 0, 0};
                double r_c[2] = {0.0, 0.0};
                int s_c[2] = {-1, -1};
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int n = tid + kk * 256;
                    if (n < N && w_f[n] > T(0)) {
                        const int g = ngraph[n], s = spc[n];
                        if (g == 0 && s >= 0 && s < ei_S) {
                            const double rv = (double)ne[n] - (double)ei_ref[n];
                            rsum[s] += rv;
                            rcnt[s] += 1;
                            r_c[kk] = rv;
                            s_c[kk] = s;
                        }
                    }
                }
                for (int s = 0; s < ei_S; ++s) {
                    double sv = rsum[s];
                    int cv = rcnt[s];
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) {
                        sv += __shfl_xor_sync(0xffffffffu, sv, o);
                        cv += __shfl_xor_sync(0xffffffffu, cv, o);
                    }
                    if (lane == 0) {
                        sw_sum[w][s] = sv;
                        sw_cnt[w][s] = cv;
                    }
                }
                __syncthreads();
                if (tid < ei_S) {
                    double ts = 0.0;
                    int tc = 0;
#pragma unroll
                    for (int ww = 0; ww < 8; ++ww) {
                        ts += sw_sum[ww][tid];
                        tc += sw_cnt[ww][tid];
                    }
                    s_mean[tid] = (tc > 0) ? (ts / (double)tc) : 0.0;
                }
                __syncthreads();
                double loc_l = 0.0;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int n = tid + kk * 256;
                    if (n < N) {
                        if (s_c[kk] >= 0) {
                            const double d = r_c[kk] - s_mean[s_c[kk]];
                            loc_l += d * d;
                            const T gv = T(gcoef * d);
                            Enb[n] = gv;
                            if (d_scot) fmax_t = fmaxf(fmax_t, fabsf((float)gv));
                        } else {
                            Enb[n] = T(0);
                        }
                    }
                }
                lei = w_ei * inv_nr * loc_l;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                lei += __shfl_xor_sync(0xffffffffu, lei, o);
                if (d_scot) fmax_t = fmaxf(fmax_t, __shfl_xor_sync(0xffffffffu, fmax_t, o));
            }
            if (lane == 0) {
                red[w] = lei;
                if (d_scot) red_m[w] = fmax_t;
            }
            __syncthreads();
            if (tid == 0) {
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    bei += red[k];
                    if (d_scot) bm = fmaxf(bm, red_m[k]);
                }
            }
        }
        if (tid == 0) {
            acc4[3] = bei;
            if (d_fmax) d_fmax[1] = __float_as_uint(bm);
            __threadfence();
            if (atomicAdd(done_cnt, 1) == 1) {
                const double loss = (double)lwp[0] * acc4[0] / (double)gr +
                                    (double)lwp[1] * acc4[1] / (3.0 * (double)nr) +
                                    (double)lwp[2] * acc4[2] / (9.0 * (double)gr) + bei;
                losses[s_cur] = T(loss);
                acc4[0] = 0.0; acc4[1] = 0.0; acc4[2] = 0.0; acc4[3] = 0.0;
                if (d_fmax && d_scot) {
                    const float mfb = fmaxf(__uint_as_float(d_fmax[0]), bm);
                    const float sc = (mfb > 1e-12f) ? fminf(fmaxf(0.25f / mfb, 1.f), 32.f) : 1.f;
                    d_scot[0] = sc;
                    d_scot[1] = 1.f / sc;
                    d_fmax[0] = 0u;
                    d_fmax[1] = 0u;
                }
                *done_cnt = 0;
            }
        }
    }
}

// Wave 39: Pure-FP32 Single-CTA Split-Warp fast loss kernel for float, N <= 512, B == 1, G1 <= 32, S <= 4 (LST_LOSS_F32=1).
// Eliminates ALL 1/64-rate FP64 hot-loop ops, __threadfence(), done_cnt atomics, and global acc4/d_fmax traffic!
// When HAS_EI=true: warps 0..3 compute E+S+F while warps 4..7 compute HAS_EI in parallel (named bar.sync 1, 128)!
template <bool HAS_EI = false>
__global__ void __launch_bounds__(256) k_loss_1blk_f32(
    const float* __restrict__ Eg, const float* __restrict__ core_e, const float* __restrict__ e_ref,
    const float* __restrict__ n_atoms, const float* __restrict__ w_e, const float* __restrict__ F,
    const float* __restrict__ f_ref, const float* __restrict__ w_f, const float* __restrict__ mv,
    const float* __restrict__ volume, const float* __restrict__ core_s, const float* __restrict__ s_ref,
    const float* __restrict__ w_s, const float* __restrict__ counts, const float* __restrict__ lw,
    float* __restrict__ Sg, float* __restrict__ Eb, float* __restrict__ Fb, float* __restrict__ Ms,
    float* __restrict__ losses, const int* __restrict__ step, int N, int G1,
    float* __restrict__ d_scot = nullptr,
    const float* __restrict__ ne = nullptr, const float* __restrict__ ei_ref = nullptr,
    const int* __restrict__ ngraph = nullptr, const int* __restrict__ spc = nullptr,
    const double* __restrict__ d_wei = nullptr, double const_wei = 0.0,
    float* __restrict__ Enb = nullptr, int ei_S = 0) {
    __shared__ float red_f[8];
    __shared__ float red_ei[4];
    __shared__ float red_m[8];
    __shared__ float sw_sum[4][4];
    __shared__ int sw_cnt[4][4];
    __shared__ float s_mean[4];

    const int tid = threadIdx.x;
    const int w = tid >> 5, lane = tid & 31;
    const int s_cur = step[0];
    const float* lwp = lw + 3L * s_cur;
    const float nr = counts[0], gr = counts[1];

    float le = 0.f, ls = 0.f;
    if constexpr (HAS_EI) {
        if (tid < 128) {
            float fmax_t = 0.f;
            if (tid < G1) {
                const int g = tid;
                const float E = Eg[g] + core_e[g];
                const float na = n_atoms[g];
                const float d = (e_ref[g] - E) / na;
                le = w_e[g] * d * d;
                const float ebv = lwp[0] * w_e[g] * 2.f * (E - e_ref[g]) / (na * na) / gr;
                Eb[g] = ebv;
                if (d_scot) fmax_t = fmaxf(fmax_t, 10.f * fabsf(ebv));
                const float inv_v = 1.f / volume[g];
                float S[9], Sb[9];
#pragma unroll
                for (int i = 0; i < 3; ++i)
#pragma unroll
                    for (int j = 0; j < 3; ++j)
                        S[3 * i + j] = 0.5f * (mv[9 * g + 3 * i + j] + mv[9 * g + 3 * j + i]) * inv_v +
                                       core_s[9 * g + 3 * i + j];
                const float scoef = lwp[2] * w_s[g] * 2.f / (9.f * gr);
#pragma unroll
                for (int k = 0; k < 9; ++k) {
                    const float dd = s_ref[9 * g + k] - S[k];
                    ls += w_s[g] * dd * dd;
                    Sg[9 * g + k] = S[k];
                    Sb[k] = scoef * (-dd);
                }
#pragma unroll
                for (int i = 0; i < 3; ++i)
#pragma unroll
                    for (int j = 0; j < 3; ++j)
                        Ms[9 * g + 3 * i + j] = 0.5f * (Sb[3 * i + j] + Sb[3 * j + i]) * inv_v;
            }
            if (w == 0 && G1 > 1) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    le += __shfl_xor_sync(0xffffffffu, le, o);
                    ls += __shfl_xor_sync(0xffffffffu, ls, o);
                }
            }
            const float fcoef = lwp[1] * 2.f / (3.f * nr);
            const int N3 = 3 * N;
            float lf = 0.f;
            for (int idx = tid; idx < N3; idx += 128) {
                const int n = idx / 3;
                const float wf = w_f[n];
                const float dd = f_ref[idx] - F[idx];
                lf += wf * dd * dd;
                const float fbv = fcoef * wf * (-dd);
                Fb[idx] = fbv;
                if (d_scot) fmax_t = fmaxf(fmax_t, fabsf(fbv));
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                lf += __shfl_xor_sync(0xffffffffu, lf, o);
                if (d_scot) fmax_t = fmaxf(fmax_t, __shfl_xor_sync(0xffffffffu, fmax_t, o));
            }
            if (lane == 0) {
                red_f[w] = lf;
                if (d_scot) red_m[w] = fmax_t;
            }
        } else {
            const int ltid = tid - 128;
            const int lw_idx = w - 4;
            const double w_ei = d_wei ? d_wei[s_cur] : const_wei;
            float loc_lf = 0.f;
            float fmax_t = 0.f;
            if (w_ei <= 0.0) {
                for (int n = ltid; n < N; n += 128) Enb[n] = 0.f;
            } else {
                const float inv_nr = (nr > 0.f) ? (1.f / nr) : 0.f;
                const float fgcoef = (float)(w_ei * 2.0 * (double)inv_nr);
                float rsum[4] = {0.f, 0.f, 0.f, 0.f};
                int rcnt[4] = {0, 0, 0, 0};
                float r_c[4] = {0.f, 0.f, 0.f, 0.f};
                int s_c[4] = {-1, -1, -1, -1};
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int n = ltid + kk * 128;
                    if (n < N && w_f[n] > 0.f) {
                        const int g = ngraph[n], s = spc[n];
                        if (g == 0 && s >= 0 && s < ei_S) {
                            const float rv = ne[n] - ei_ref[n];
                            rsum[s] += rv;
                            rcnt[s] += 1;
                            r_c[kk] = rv;
                            s_c[kk] = s;
                        }
                    }
                }
                for (int s = 0; s < ei_S; ++s) {
                    float sv = rsum[s];
                    int cv = rcnt[s];
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) {
                        sv += __shfl_xor_sync(0xffffffffu, sv, o);
                        cv += __shfl_xor_sync(0xffffffffu, cv, o);
                    }
                    if (lane == 0) {
                        sw_sum[lw_idx][s] = sv;
                        sw_cnt[lw_idx][s] = cv;
                    }
                }
                asm volatile("bar.sync 1, 128;");
                if (ltid < ei_S) {
                    const float ts = sw_sum[0][ltid] + sw_sum[1][ltid] + sw_sum[2][ltid] + sw_sum[3][ltid];
                    const int tc = sw_cnt[0][ltid] + sw_cnt[1][ltid] + sw_cnt[2][ltid] + sw_cnt[3][ltid];
                    s_mean[ltid] = (tc > 0) ? (ts / (float)tc) : 0.f;
                }
                asm volatile("bar.sync 1, 128;");
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int n = ltid + kk * 128;
                    if (n < N) {
                        if (s_c[kk] >= 0) {
                            const float d = r_c[kk] - s_mean[s_c[kk]];
                            loc_lf += d * d;
                            const float gv = fgcoef * d;
                            Enb[n] = gv;
                            if (d_scot) fmax_t = fmaxf(fmax_t, fabsf(gv));
                        } else {
                            Enb[n] = 0.f;
                        }
                    }
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    loc_lf += __shfl_xor_sync(0xffffffffu, loc_lf, o);
                    if (d_scot) fmax_t = fmaxf(fmax_t, __shfl_xor_sync(0xffffffffu, fmax_t, o));
                }
            }
            if (lane == 0) {
                red_ei[lw_idx] = loc_lf;
                if (d_scot) red_m[w] = fmax_t;
            }
        }
        __syncthreads();
        if (tid == 0) {
            const double tot_f = (double)red_f[0] + (double)red_f[1] + (double)red_f[2] + (double)red_f[3];
            const double w_ei = d_wei ? d_wei[s_cur] : const_wei;
            const double inv_nr = ((double)nr > 0.0) ? (1.0 / (double)nr) : 0.0;
            const double tot_ei = (w_ei > 0.0)
                ? (w_ei * inv_nr * ((double)red_ei[0] + (double)red_ei[1] + (double)red_ei[2] + (double)red_ei[3]))
                : 0.0;
            const double loss = (double)lwp[0] * (double)le / (double)gr +
                                (double)lwp[1] * tot_f / (3.0 * (double)nr) +
                                (double)lwp[2] * (double)ls / (9.0 * (double)gr) + tot_ei;
            losses[s_cur] = (float)loss;
            if (d_scot) {
                float bm = 0.f;
#pragma unroll
                for (int k = 0; k < 8; ++k) bm = fmaxf(bm, red_m[k]);
                const float sc = (bm > 1e-12f) ? fminf(fmaxf(0.25f / bm, 1.f), 32.f) : 1.f;
                d_scot[0] = sc;
                d_scot[1] = 1.f / sc;
            }
        }
    } else {
        float fmax_t = 0.f;
        if (tid < G1) {
            const int g = tid;
            const float E = Eg[g] + core_e[g];
            const float na = n_atoms[g];
            const float d = (e_ref[g] - E) / na;
            le = w_e[g] * d * d;
            const float ebv = lwp[0] * w_e[g] * 2.f * (E - e_ref[g]) / (na * na) / gr;
            Eb[g] = ebv;
            if (d_scot) fmax_t = fmaxf(fmax_t, 10.f * fabsf(ebv));
            const float inv_v = 1.f / volume[g];
            float S[9], Sb[9];
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j)
                    S[3 * i + j] = 0.5f * (mv[9 * g + 3 * i + j] + mv[9 * g + 3 * j + i]) * inv_v +
                                   core_s[9 * g + 3 * i + j];
            const float scoef = lwp[2] * w_s[g] * 2.f / (9.f * gr);
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                const float dd = s_ref[9 * g + k] - S[k];
                ls += w_s[g] * dd * dd;
                Sg[9 * g + k] = S[k];
                Sb[k] = scoef * (-dd);
            }
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j)
                    Ms[9 * g + 3 * i + j] = 0.5f * (Sb[3 * i + j] + Sb[3 * j + i]) * inv_v;
        }
        if (w == 0 && G1 > 1) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                le += __shfl_xor_sync(0xffffffffu, le, o);
                ls += __shfl_xor_sync(0xffffffffu, ls, o);
            }
        }
        const float fcoef = lwp[1] * 2.f / (3.f * nr);
        const int N3 = 3 * N;
        float lf = 0.f;
        for (int idx = tid; idx < N3; idx += 256) {
            const int n = idx / 3;
            const float wf = w_f[n];
            const float dd = f_ref[idx] - F[idx];
            lf += wf * dd * dd;
            const float fbv = fcoef * wf * (-dd);
            Fb[idx] = fbv;
            if (d_scot) fmax_t = fmaxf(fmax_t, fabsf(fbv));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            lf += __shfl_xor_sync(0xffffffffu, lf, o);
            if (d_scot) fmax_t = fmaxf(fmax_t, __shfl_xor_sync(0xffffffffu, fmax_t, o));
        }
        if (lane == 0) {
            red_f[w] = lf;
            if (d_scot) red_m[w] = fmax_t;
        }
        __syncthreads();
        if (tid == 0) {
            double tot_f = 0.0;
            float bm = 0.f;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                tot_f += (double)red_f[k];
                if (d_scot) bm = fmaxf(bm, red_m[k]);
            }
            const double loss = (double)lwp[0] * (double)le / (double)gr +
                                (double)lwp[1] * tot_f / (3.0 * (double)nr) +
                                (double)lwp[2] * (double)ls / (9.0 * (double)gr);
            losses[s_cur] = (float)loss;
            if (d_scot) {
                const float sc = (bm > 1e-12f) ? fminf(fmaxf(0.25f / bm, 1.f), 32.f) : 1.f;
                d_scot[0] = sc;
                d_scot[1] = 1.f / sc;
            }
        }
    }
}

// Wave 30: Gauge-centered per-atom teacher local energy distillation loss & gradient (<2 us, 1 block).
// L_ei = (w_ei / n_real) * sum_{n real} ((ne[n] - ei_ref[n]) - mean_{g,s}(ne - ei_ref))^2
// dL_ei / d(ne[n]) = (2 * w_ei / n_real) * ((ne[n] - ei_ref[n]) - mean_{g,s}(ne - ei_ref))
template <typename T>
__global__ void __launch_bounds__(256) k_node_energy_grad(
    const T* __restrict__ ne, const T* __restrict__ ei_ref,
    const int* __restrict__ ngraph, const int* __restrict__ spc,
    const T* __restrict__ w_f, const T* __restrict__ counts,
    const double* __restrict__ d_wei, double const_wei,
    const int* __restrict__ step,
    T* __restrict__ Enb, T* __restrict__ losses,
    float* __restrict__ d_scot,
    int N, int B, int S) {
    extern __shared__ unsigned char smem_raw[];
    double* s_sum = reinterpret_cast<double*>(smem_raw);
    int* s_cnt = reinterpret_cast<int*>(s_sum + B * S);
    __shared__ double s_loss[8];
    __shared__ float s_emax[8];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int w = tid >> 5;
    const double w_ei = d_wei ? d_wei[step[0]] : const_wei;
    if (w_ei <= 0.0) {
        for (int n = tid; n < N; n += 256) Enb[n] = T(0);
        return;
    }
    const int BS = B * S;
    for (int k = tid; k < BS; k += 256) {
        s_sum[k] = 0.0;
        s_cnt[k] = 0;
    }
    __syncthreads();

    const int G = (int)counts[1];
    const double nr = (double)counts[0];
    // Warp-coalesced per-(graph, species) mean accumulation (1 iter/warp for species-sorted atoms)
    for (int base = 0; base < N; base += 256) {
        const int n = base + tid;
        int key = -1;
        double r = 0.0;
        if (n < N) {
            const int g = ngraph[n];
            const int s = spc[n];
            if (g >= 0 && g < G && s >= 0 && s < S && w_f[n] > T(0)) {
                key = g * S + s;
                r = (double)ne[n] - (double)ei_ref[n];
            }
        }
        unsigned todo = __ballot_sync(0xffffffffu, key >= 0);
        while (todo) {
            const int lead = __ffs(todo) - 1;
            const int k_cur = __shfl_sync(0xffffffffu, key, lead);
            const unsigned grp = todo & __ballot_sync(0xffffffffu, key == k_cur);
            double v = (key == k_cur) ? r : 0.0;
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == lead) {
                atomicAdd(s_sum + k_cur, v);
                atomicAdd(s_cnt + k_cur, __popc(grp));
            }
            todo &= ~grp;
        }
    }
    __syncthreads();

    for (int k = tid; k < BS; k += 256) {
        s_sum[k] = (s_cnt[k] > 0) ? (s_sum[k] / (double)s_cnt[k]) : 0.0;
    }
    __syncthreads();

    const double inv_nr = (nr > 0.0) ? (1.0 / nr) : 0.0;
    const double gcoef = w_ei * 2.0 * inv_nr;
    double loc_l = 0.0;
    float loc_m = 0.f;
    for (int n = tid; n < N; n += 256) {
        const int g = ngraph[n];
        const int s = spc[n];
        if (g >= 0 && g < G && s >= 0 && s < S && w_f[n] > T(0)) {
            const double d = ((double)ne[n] - (double)ei_ref[n]) - s_sum[g * S + s];
            loc_l += d * d;
            const T gv = T(gcoef * d);
            Enb[n] = gv;
            if (d_scot) loc_m = fmaxf(loc_m, fabsf((float)gv));
        } else {
            Enb[n] = T(0);
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        loc_l += __shfl_xor_sync(0xffffffffu, loc_l, o);
        if (d_scot) loc_m = fmaxf(loc_m, __shfl_xor_sync(0xffffffffu, loc_m, o));
    }
    if (lane == 0) {
        s_loss[w] = loc_l;
        if (d_scot) s_emax[w] = loc_m;
    }
    __syncthreads();
    if (tid == 0) {
        double tot_l = 0.0;
        for (int k = 0; k < 8; ++k) tot_l += s_loss[k];
        losses[*step] += T(w_ei * inv_nr * tot_l);
        if (d_scot) {
            float em = 0.f;
            for (int k = 0; k < 8; ++k) em = fmaxf(em, s_emax[k]);
            if (em > 1e-12f) {
                const float sc_cur = d_scot[0];
                const float sc_ei = fminf(fmaxf(0.25f / em, 1.f), 32.f);
                if (sc_ei < sc_cur) {
                    d_scot[0] = sc_ei;
                    d_scot[1] = 1.f / sc_ei;
                }
            }
        }
    }
}

// lst_evaluate's outputs: energy [G1] (k_loss's E), optional forces [N, 3], optional stress [G1, 9] (k_loss's S)
template <typename T>
__global__ void k_eval_out(const T* Eg, const T* core_e, const T* mv, const T* volume, const T* core_s, const T* F,
                           T* e, T* f, T* s, int N, int G1) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f && i < 3L * N) f[i] = F[i];
    if (i < G1) {
        e[i] = Eg[i] + core_e[i];
        if (s) {
            for (int a = 0; a < 3; ++a)
                for (int c = 0; c < 3; ++c)
                    s[9 * i + 3 * a + c] = T(0.5) * (mv[9 * i + 3 * a + c] + mv[9 * i + 3 * c + a]) / volume[i] +
                                           core_s[9 * i + 3 * a + c];
        }
    }
}

// w = Fb[s] - Fb[r] + vec . Ms[g]; sb = w . U; gyb = JY^T w; Yw = Y sb
template <typename T>
__global__ void k_edge_cot(const T* Fb, const T* Ms, const T* vec, const T* U, const T* JY, const T* Y, const int* snd,
                           const int* rcv, const int* egraph, T* sb, T* gyb, T* Yw, int E) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    long s = snd[e], r = rcv[e];
    const T* M = Ms + 9L * egraph[e];
    T w[3];
    for (int j = 0; j < 3; ++j) {
        T v = Fb[3 * s + j] - Fb[3 * r + j];
        for (int i = 0; i < 3; ++i) v += vec[3L * e + i] * M[3 * i + j];
        w[j] = v;
    }
    T b = w[0] * U[3 * e] + w[1] * U[3 * e + 1] + w[2] * U[3 * e + 2];
    sb[e] = b;
    T jy[48], y[16], o1[16], o2[16];
    ld16(JY + 48L * e, jy);
    ld16(JY + 48L * e + 16, jy + 16);
    ld16(JY + 48L * e + 32, jy + 32);
    ld16(Y + 16L * e, y);
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        o1[m] = jy[3 * m] * w[0] + jy[3 * m + 1] * w[1] + jy[3 * m + 2] * w[2];
        o2[m] = y[m] * b;
    }
    st16(gyb + 16L * e, o1);
    st16(Yw + 16L * e, o2);
}

// Pair-indexed, JY-free k_edge_cot (LST_PAIR_BASIS=1): one thread per pair p < P, canonical edge e0 = pe0[p];
// writes gybP[p, 0..15] = JY(e0) w and YwP[p, 0..15] = YP[p] * (w . u) without reading JY.
template <typename T>
__global__ void k_edge_cot_pair(const T* __restrict__ Fb, const T* __restrict__ Ms, const T* __restrict__ vec,
                                const T* __restrict__ UP, const T* __restrict__ YP, const int* __restrict__ pe0,
                                const int* __restrict__ snd, const int* __restrict__ rcv,
                                const int* __restrict__ egraph, T* __restrict__ gybP, T* __restrict__ YwP, int P_cnt,
                                T* __restrict__ Y3P = nullptr, int y3_mode = 1,
                                const float* __restrict__ d_scot = nullptr,
                                const float4* __restrict__ p_vec4 = nullptr, const int4* __restrict__ p_srg = nullptr) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P_cnt) return;
    long s, r;
    const T* M;
    T vx, vy, vz;
    if (p_vec4 && p_srg) {
        const int4 srg = p_srg[p];
        s = srg.x;
        r = srg.y;
        M = Ms + 9L * srg.z;
        const float4 v4 = p_vec4[p];
        vx = T(v4.x); vy = T(v4.y); vz = T(v4.z);
    } else {
        const long e0 = pe0[p];
        s = snd[e0];
        r = rcv[e0];
        M = Ms + 9L * egraph[e0];
        vx = vec[3L * e0]; vy = vec[3L * e0 + 1]; vz = vec[3L * e0 + 2];
    }
    const T sc = d_scot ? T(d_scot[0]) : T(1);
    const T wx = sc * (Fb[3 * s] - Fb[3 * r] + vx * M[0] + vy * M[3] + vz * M[6]);
    const T wy = sc * (Fb[3 * s + 1] - Fb[3 * r + 1] + vx * M[1] + vy * M[4] + vz * M[7]);
    const T wz = sc * (Fb[3 * s + 2] - Fb[3 * r + 2] + vx * M[2] + vy * M[5] + vz * M[8]);
    T ux, uy, uz, rinv;
    if constexpr (std::is_same<T, float>::value) {
        const float4 up4 = *reinterpret_cast<const float4*>(UP + 4L * p);
        ux = up4.x; uy = up4.y; uz = up4.z; rinv = up4.w;
    } else {
        ux = UP[4L * p]; uy = UP[4L * p + 1]; uz = UP[4L * p + 2]; rinv = UP[4L * p + 3];
    }
    const T b = wx * ux + wy * uy + wz * uz;
    T y[16], o1[16];
    ld16(YP + 16L * p, y);
    sh16_jyv<T>(ux, uy, uz, rinv, y, wx, wy, wz, b, o1);
    if (Y3P) {
        if constexpr (std::is_same<T, float>::value) {
            if (y3_mode == 3) {
                uint4* outh = reinterpret_cast<uint4*>(Y3P) + 6L * p;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    outh[q] = make_uint4(ah::pack2<__half>(y[4 * q + 0] * b, y[4 * q + 0]),
                                         ah::pack2<__half>(y[4 * q + 1] * b, y[4 * q + 1]),
                                         ah::pack2<__half>(y[4 * q + 2] * b, y[4 * q + 2]),
                                         ah::pack2<__half>(y[4 * q + 3] * b, y[4 * q + 3]));
                }
                outh[4] = make_uint4(ah::pack2<__half>(o1[0], o1[1]), ah::pack2<__half>(o1[2], o1[3]),
                                     ah::pack2<__half>(o1[4], o1[5]), ah::pack2<__half>(o1[6], o1[7]));
                outh[5] = make_uint4(ah::pack2<__half>(o1[8], o1[9]), ah::pack2<__half>(o1[10], o1[11]),
                                     ah::pack2<__half>(o1[12], o1[13]), ah::pack2<__half>(o1[14], o1[15]));
            } else if (y3_mode == 2) {
                uint4* outh = reinterpret_cast<uint4*>(Y3P) + 6L * p;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    outh[q] = make_uint4(ah::pack2<__half>(y[4 * q + 0] * b, y[4 * q + 1] * b),
                                         ah::pack2<__half>(y[4 * q + 2] * b, y[4 * q + 3] * b),
                                         ah::pack2<__half>(o1[4 * q + 0], o1[4 * q + 1]),
                                         ah::pack2<__half>(o1[4 * q + 2], o1[4 * q + 3]));
                }
                outh[4] = make_uint4(ah::pack2<__half>(y[0], y[1]), ah::pack2<__half>(y[2], y[3]),
                                     ah::pack2<__half>(y[4], y[5]), ah::pack2<__half>(y[6], y[7]));
                outh[5] = make_uint4(ah::pack2<__half>(y[8], y[9]), ah::pack2<__half>(y[10], y[11]),
                                     ah::pack2<__half>(y[12], y[13]), ah::pack2<__half>(y[14], y[15]));
            } else {
                float4* out4 = reinterpret_cast<float4*>(Y3P + 48L * p);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    out4[3 * q + 0] = make_float4(y[4 * q + 0] * b, y[4 * q + 1] * b, y[4 * q + 2] * b, y[4 * q + 3] * b);
                    out4[3 * q + 1] = make_float4(o1[4 * q + 0], o1[4 * q + 1], o1[4 * q + 2], o1[4 * q + 3]);
                    out4[3 * q + 2] = make_float4(y[4 * q + 0], y[4 * q + 1], y[4 * q + 2], y[4 * q + 3]);
                }
            }
        }
    }
    if (gybP) st16(gybP + 16L * p, o1);
    if (YwP) {
#pragma unroll
        for (int m = 0; m < 16; ++m) y[m] *= b;
        st16(YwP + 16L * p, y);
    }
}

// ============================================================================ parameters
template <typename T>
__global__ void k_scale(T* dst, const T* src, T f, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * f;
}

template <typename T>
struct Seg {
    T* dst;
    const T* src;
    T f;
    long n;
};

template <typename T>
__global__ void k_scale_multi(const Seg<T>* segs) {
    const Seg<T> q = segs[blockIdx.y];
    if constexpr (std::is_same<T, float>::value) {
        if (((q.n & 3L) == 0L) && (((uintptr_t)q.src & 15) == 0) && (((uintptr_t)q.dst & 15) == 0)) {
            const long n4 = q.n >> 2;
            const float4* src4 = reinterpret_cast<const float4*>(q.src);
            float4* dst4 = reinterpret_cast<float4*>(q.dst);
            const float f = q.f;
            for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x) {
                const float4 v = src4[i];
                dst4[i] = make_float4(v.x * f, v.y * f, v.z * f, v.w * f);
            }
            return;
        }
    }
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < q.n; i += (long)gridDim.x * blockDim.x)
        q.dst[i] = q.src[i] * q.f;
}

// W [S, K, C] <-> (wmax [S, K3, C], w2 [S, K2, C], w1 [S, K1, C])
template <typename T>
__global__ void k_cat_w(T* W, const T* a, const T* b, const T* c, int S, int K, int K3, int K2, int C, int dir) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)S * K * C) return;
    long s = i / ((long)K * C), k = (i / C) % K, d = i % C;
    int K1 = K - K3 - K2;
    T* src;
    long j;
    if (k < K3) { src = const_cast<T*>(a); j = (s * K3 + k) * C + d; }
    else if (k < K3 + K2) { src = const_cast<T*>(b); j = (s * K2 + k - K3) * C + d; }
    else { src = const_cast<T*>(c); j = (s * K1 + k - K3 - K2) * C + d; }
    if (dir == 0) W[i] = src[j];
    else src[j] = W[i];
}

// Wcat [C, ZW] <-> (wr0 [C], wu1 [C, C], wsk1 [C, SC]) with factors (forward: raw * f, backward: grad * f)
template <typename T>
__global__ void k_cat_wcat(T* Wc, T* wr0, T* wu1, T* wsk1, T f0, T f1, T f2, int C, int ZW, int dir) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)C * ZW) return;
    long c = i / ZW, j = i % ZW;
    T* p;
    T f;
    if (j == 0) { p = wr0 + c; f = f0; }
    else if (j <= C) { p = wu1 + c * C + j - 1; f = f1; }
    else { p = wsk1 + c * (ZW - 1 - C) + j - 1 - C; f = f2; }
    if (dir == 0) Wc[i] = *p * f;
    else *p = Wc[i] * f;
}

template <typename T>
__global__ void k_sumsq(const T* g, long n, double* out) {
    double s = 0;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        double v = (double)g[i];
        s += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (threadIdx.x % 32 == 0) atomicAdd(out, s);
}

// (k_grad_finalize_sumsq defined after k_adam_prep so it can share adam_prep_warp in Wave 35)

template <typename T>
struct PDesc {
    T* p;
    const T* g;
    T *m, *v, *vmax, *ema;
    float* step;
    long n;
    int group;
    int muon;   // Muon tensor: k_adam updates its momentum and direction only
    T* dir;     // Muon direction (scratch of the tensor's size)
};

struct MuonMat {
    int param, rows, cols, tr;   // tr: rows > cols, iterate on the transpose (m = min, n = max)
    long off, rs, cs, ws;        // first element, row / column strides, workspace offset
    long nv;                     // NorMuon: offset of the rows' second-moment EMA
};

// Muon's last step: p -= ft * X back in the parameter's layout (X: m x n, the transpose when mt.tr). NorMuon
// (nbeta >= 0): row r of the block gets 1 / (sqrt(v_r) + 1e-12), v_r = EMA (nbeta) of its mean square (the first
// step's mean square), and the whole matrix the factor that restores X's Frobenius norm. Y: scratch (>= rows).
template <typename T>
__device__ void muon_apply(const MuonMat& mt, const PDesc<T>& q, const T* X, T* Y, T ft, T* nvb, double nbeta,
                           int m, int n) {
    if (nbeta < 0.0 || !nvb) {
        for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
            const int i = idx / n, j = idx % n;
            const int r = mt.tr ? j : i, cc = mt.tr ? i : j;
            q.p[mt.off + r * mt.rs + cc * mt.cs] -= ft * X[idx];
        }
        return;
    }
    __shared__ double nred[2][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    double so = 0.0, sh = 0.0;
    for (int r = warp; r < mt.rows; r += nw) {
        double s = 0.0;
        for (int c = lane; c < mt.cols; c += 32) {
            const double x = (double)X[mt.tr ? (long)c * n + r : (long)r * n + c];
            s += x * x;
        }
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        const double r2 = s / mt.cols;
        T* v = nvb + mt.nv + r;
        double vr = (double)*v;
        __syncwarp();
        vr = vr == 0.0 ? r2 : nbeta * vr + (1.0 - nbeta) * r2;
        const double sc = 1.0 / (sqrt(vr) + 1e-12);
        if (lane == 0) {
            *v = T(vr);
            Y[r] = T(sc);
        }
        so += s;
        sh += s * sc * sc;
    }
    if (lane == 0) {
        nred[0][warp] = so;
        nred[1][warp] = sh;
    }
    __syncthreads();
    double to = 0.0, th = 0.0;
    for (int w = 0; w < nw; ++w) {
        to += nred[0][w];
        th += nred[1][w];
    }
    const T f = ft * T(sqrt(to) / (sqrt(th) + 1e-12));
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        const int i = idx / n, j = idx % n;
        const int r = mt.tr ? j : i, cc = mt.tr ? i : j;
        q.p[mt.off + r * mt.rs + cc * mt.cs] -= f * Y[r] * X[idx];
    }
}

// Fast exact integer exponentiation in FP64 (~10 FMULs vs slow libdevice pow(double,double), exact to <=1 ULP)
__device__ __forceinline__ double ipow_d(double base, int exp) {
    double r = 1.0, b = base;
    int e = exp;
    while (e > 0) {
        if (e & 1) r *= b;
        b *= b;
        e >>= 1;
    }
    return r;
}

// Shared 1-warp adam_prep logic (called by standalone k_adam_prep AND fused tail of k_grad_finalize_sumsq!)
template <typename T>
__device__ __forceinline__ void adam_prep_warp(const PDesc<T>* d, int np, double norm2_val, const double* lr,
                                               double max_norm, double b1, double b2, const double* ema_decay,
                                               int* step_idx, T* sc, double* urms, double* caut
#if CUDART_VERSION >= 12040
                                               , cudaGraphConditionalHandle soap_cond = 0, int soap_every = 0
#endif
) {
    const int i = threadIdx.x;
    const int s_cur = step_idx[0];
    float s = 0.f;
    if (i < np) {
        float* stp = d[i].step;
        s = *stp + 1.0f;
        *stp = s;
    }
    const float s0 = __shfl_sync(0xffffffffu, s, 0);
    const bool same_s = (__ballot_sync(0xffffffffu, i >= np || s == s0) == 0xffffffffu);
    if (same_s) {
        double val = 0.0;
        const bool int_s = (s0 >= 1.0f && s0 <= 16777215.0f && floorf(s0) == s0);
        if (i == 0) val = 1.0 - (int_s ? ipow_d(b1, (int)s0) : pow(b1, (double)s0));
        else if (i == 1) val = sqrt(1.0 - (int_s ? ipow_d(b2, (int)s0) : pow(b2, (double)s0)));
        const double bc1 = __shfl_sync(0xffffffffu, val, 0);
        const double bc2s = __shfl_sync(0xffffffffu, val, 1);
        if (i < np) {
            sc[3 * i] = T(lr[16L * s_cur + d[i].group] / bc1);
            sc[3 * i + 1] = T(bc2s);
            if (urms) urms[i] = 0.0;
            if (caut) caut[i] = 0.0;
        }
    } else if (i < np) {
        sc[3 * i] = T(lr[16L * s_cur + d[i].group] / (1.0 - pow(b1, (double)s)));
        sc[3 * i + 1] = T(sqrt(1.0 - pow(b2, (double)s)));
        if (urms) urms[i] = 0.0;
        if (caut) caut[i] = 0.0;
    }
    __syncwarp();
    if (i == 0) {
        double coef = 1.0;
        if (max_norm > 0) coef = fmin(1.0, max_norm / (sqrt(norm2_val) + 1e-6));
        sc[3 * np] = T(coef);
        sc[3 * np + 1] = ema_decay ? T(1.0 - ema_decay[s_cur]) : T(0);
        // Wave 34: save current step in step_idx[1] for post-prep optimizer kernels & advance step_idx[0] (deletes k_step_end!)
        step_idx[1] = s_cur;
        step_idx[0] = s_cur + 1;
#if CUDART_VERSION >= 12040
        if (soap_cond && soap_every > 0) {
            const unsigned int fire = (!same_s || (((long)s0 - 1) % soap_every == 0)) ? 1u : 0u;
            cudaGraphSetConditional(soap_cond, fire);
        }
#endif
    }
}

// per parameter: step += 1, step size lr / (1 - b1^t), sqrt(1 - b2^t); the clip coefficient; 1 - EMA decay
template <typename T>
__global__ void k_adam_prep(const PDesc<T>* d, int np, const double* norm2, const double* lr, double max_norm,
                            double b1, double b2, const double* ema_decay, int* step_idx, T* sc,
                            double* urms, double* caut
#if CUDART_VERSION >= 12040
                            , cudaGraphConditionalHandle soap_cond = 0, int soap_every = 0
#endif
) {
    adam_prep_warp<T>(d, np, *norm2, lr, max_norm, b1, b2, ema_decay, step_idx, sc, urms, caut
#if CUDART_VERSION >= 12040
                      , soap_cond, soap_every
#endif
    );
}

// Wave 21+24+35: Fused k_scale_multi + k_uncat_gw_u_pair + k_cat_wcat + k_sumsq + 1/S_cot unscale + optional adam_prep!
// Wave 35: float4 vectorized loops, block-level smem reduction (1 atomicAdd(double) per CTA instead of 8 -> 32x fewer
// atomics at gx=4!), and last-CTA fused adam_prep_warp (eliminates k_adam_prep launch completely!).
template <typename T>
__global__ void __launch_bounds__(256) k_grad_finalize_sumsq(
    const Seg<T>* __restrict__ segs, int n_gseg, const T* __restrict__ gWu0, T* __restrict__ a0, T* __restrict__ b0,
    T* __restrict__ c0, const T* __restrict__ gWu1, T* __restrict__ a1, T* __restrict__ b1, T* __restrict__ c1, int S,
    int K, int K3, int K2, int C, const T* __restrict__ gWcat, T* __restrict__ wr0, T* __restrict__ wu1,
    T* __restrict__ wsk1, T f0, T f1, T f2, int ZW, double* __restrict__ out_norm2,
    const float* __restrict__ d_scot = nullptr,
    int* __restrict__ gf_done = nullptr, const PDesc<T>* __restrict__ pdesc = nullptr, int np = 0,
    const double* __restrict__ lr = nullptr, double max_norm = 0.0, double adam_b1 = 0.0, double adam_b2 = 0.0,
    const double* __restrict__ ema_decay = nullptr, int* __restrict__ step_idx = nullptr, T* __restrict__ sc = nullptr,
    double* __restrict__ urms = nullptr, double* __restrict__ caut = nullptr) {
    __shared__ double sm_red[8];
    __shared__ int is_last;
    const int tid = threadIdx.x;
    const int by = blockIdx.y;
    const long stride = (long)gridDim.x * blockDim.x;
    const long tid0 = (long)blockIdx.x * blockDim.x + tid;
    const T isc = d_scot ? T(d_scot[1]) : T(1);
    double acc = 0.0;
    if (by < n_gseg) {
        const Seg<T> q = segs[by];
        const T qf = q.f * isc;
        if constexpr (std::is_same<T, float>::value) {
            if (((q.n & 3L) == 0L) && (((uintptr_t)q.src & 15) == 0) && (((uintptr_t)q.dst & 15) == 0)) {
                const long n4 = q.n >> 2;
                const float4* src4 = reinterpret_cast<const float4*>(q.src);
                float4* dst4 = reinterpret_cast<float4*>(q.dst);
                for (long i = tid0; i < n4; i += stride) {
                    const float4 s4 = src4[i];
                    const float4 v4 = make_float4(s4.x * qf, s4.y * qf, s4.z * qf, s4.w * qf);
                    dst4[i] = v4;
                    acc += (double)v4.x * (double)v4.x + (double)v4.y * (double)v4.y +
                           (double)v4.z * (double)v4.z + (double)v4.w * (double)v4.w;
                }
            } else {
                for (long i = tid0; i < q.n; i += stride) {
                    const T v = q.src[i] * qf;
                    q.dst[i] = v;
                    acc += (double)v * (double)v;
                }
            }
        } else {
            for (long i = tid0; i < q.n; i += stride) {
                const T v = q.src[i] * qf;
                q.dst[i] = v;
                acc += (double)v * (double)v;
            }
        }
    } else if (by < n_gseg + 2) {
        const bool l1 = (by == n_gseg + 1);
        const T* gWu = l1 ? gWu1 : gWu0;
        T* a = l1 ? a1 : a0;
        T* b = l1 ? b1 : b0;
        T* c = l1 ? c1 : c0;
        const int K1 = K - K3 - K2;
        const long tot = (long)S * K * C;
        for (long i = tid0; i < tot; i += stride) {
            const long s_i = i / ((long)K * C), k = (i / C) % K, d = i % C;
            const T v = gWu[(s_i * POLY_U + POLY_K_TO_U[k]) * C + d] * isc;
            if (k < K3) a[(s_i * K3 + k) * C + d] = v;
            else if (k < K3 + K2) b[(s_i * K2 + (k - K3)) * C + d] = v;
            else c[(s_i * K1 + (k - K3 - K2)) * C + d] = v;
            acc += (double)v * (double)v;
        }
    } else {
        const long tot = (long)C * ZW;
        for (long i = tid0; i < tot; i += stride) {
            const long c = i / ZW, j = i % ZW;
            T* p;
            T f;
            if (j == 0) { p = wr0 + c; f = f0 * isc; }
            else if (j <= C) { p = wu1 + c * C + j - 1; f = f1 * isc; }
            else { p = wsk1 + c * (ZW - 1 - C) + j - 1 - C; f = f2 * isc; }
            const T v = gWcat[i] * f;
            *p = v;
            acc += (double)v * (double)v;
        }
    }
    if (out_norm2) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if ((tid & 31) == 0) sm_red[tid >> 5] = acc;
        __syncthreads();
        if (tid == 0) {
            double bacc = 0.0;
#pragma unroll
            for (int w = 0; w < 8; ++w) bacc += sm_red[w];
            if (bacc != 0.0) atomicAdd(out_norm2, bacc);
            if (gf_done && pdesc) {
                __threadfence();
                is_last = (atomicAdd(gf_done, 1) == (int)(gridDim.x * gridDim.y) - 1) ? 1 : 0;
            }
        }
        if (gf_done && pdesc) {
            __syncthreads();
            if (is_last && tid < 32) {
                if (tid == 0) *gf_done = 0;
                const double n2 = (tid == 0) ? atomicAdd(out_norm2, 0.0) : 0.0;
                adam_prep_warp<T>(pdesc, np, n2, lr, max_norm, adam_b1, adam_b2, ema_decay, step_idx, sc, urms, caut);
            }
        }
    }
}

// Wave 36: Flattened uniform-tile grad finalize + sumsq + fused adam_prep (LST_GF_FLAT=1).
// Eliminates the 4-CTA ws0 (49,152-float) straggler and 64-bit integer divs in gWu0/gWu1 -> ~185 uniform CTAs (~12 -> ~3.5 us!).
struct GfTile {
    uint8_t kind;   // 0=segs[sub], 1=gWu[sub] (rows of C), 2=gWcat (elems)
    uint8_t sub;
    uint16_t pad;
    int off;
    int cnt;
};

template <typename T, bool ACC_F32 = false>
__global__ void __launch_bounds__(256) k_grad_finalize_sumsq_flat(
    const GfTile* __restrict__ tiles, int n_tiles, const Seg<T>* __restrict__ segs,
    const T* __restrict__ gWu0, T* __restrict__ a0, T* __restrict__ b0, T* __restrict__ c0,
    const T* __restrict__ gWu1, T* __restrict__ a1, T* __restrict__ b1, T* __restrict__ c1,
    int K, int K3, int K2, int C, const T* __restrict__ gWcat, T* __restrict__ wr0, T* __restrict__ wu1,
    T* __restrict__ wsk1, T f0, T f1, T f2, int ZW, double* __restrict__ out_norm2,
    const float* __restrict__ d_scot = nullptr,
    int* __restrict__ gf_done = nullptr, const PDesc<T>* __restrict__ pdesc = nullptr, int np = 0,
    const double* __restrict__ lr = nullptr, double max_norm = 0.0, double adam_b1 = 0.0, double adam_b2 = 0.0,
    const double* __restrict__ ema_decay = nullptr, int* __restrict__ step_idx = nullptr, T* __restrict__ sc = nullptr,
    double* __restrict__ urms = nullptr, double* __restrict__ caut = nullptr) {
    using AccT = typename std::conditional<ACC_F32 && std::is_same<T, float>::value, float, double>::type;
    __shared__ AccT sm_red[8];
    __shared__ int is_last;
    const int tid = threadIdx.x;
    const T isc = d_scot ? T(d_scot[1]) : T(1);
    AccT acc = AccT(0);

    for (int ti = blockIdx.x; ti < n_tiles; ti += gridDim.x) {
        const GfTile tl = tiles[ti];
        if (tl.kind == 0) {
            const Seg<T> q = segs[tl.sub];
            const T qf = q.f * isc;
            const T* src = q.src + tl.off;
            T* dst = q.dst + tl.off;
            if constexpr (std::is_same<T, float>::value) {
                if (((tl.cnt & 3) == 0) && (((uintptr_t)src & 15) == 0) && (((uintptr_t)dst & 15) == 0)) {
                    const int n4 = tl.cnt >> 2;
                    if (tid < n4) {
                        const float4 s4 = reinterpret_cast<const float4*>(src)[tid];
                        const float4 v4 = make_float4(s4.x * qf, s4.y * qf, s4.z * qf, s4.w * qf);
                        reinterpret_cast<float4*>(dst)[tid] = v4;
                        acc += (AccT)v4.x * (AccT)v4.x + (AccT)v4.y * (AccT)v4.y +
                               (AccT)v4.z * (AccT)v4.z + (AccT)v4.w * (AccT)v4.w;
                    }
                } else {
                    for (int j = tid; j < tl.cnt; j += 256) {
                        const T v = src[j] * qf;
                        dst[j] = v;
                        acc += (AccT)v * (AccT)v;
                    }
                }
            } else {
                for (int j = tid; j < tl.cnt; j += 256) {
                    const T v = src[j] * qf;
                    dst[j] = v;
                    acc += (AccT)v * (AccT)v;
                }
            }
        } else if (tl.kind == 1) {
            const bool l1 = (tl.sub != 0);
            const T* gWu = l1 ? gWu1 : gWu0;
            T* a = l1 ? a1 : a0;
            T* b = l1 ? b1 : b0;
            T* c = l1 ? c1 : c0;
            const int K1 = K - K3 - K2;
            if constexpr (std::is_same<T, float>::value) {
                if (C == 64) {
                    const int r_loc = tid >> 4;
                    const int c4 = (tid & 15) * 4;
                    if (r_loc < tl.cnt) {
                        const int row = tl.off + r_loc;
                        const int s_i = row / K;
                        const int k = row - s_i * K;
                        const float4 s4 = *reinterpret_cast<const float4*>(gWu + (long)(s_i * POLY_U + POLY_K_TO_U[k]) * 64 + c4);
                        const float4 v4 = make_float4(s4.x * isc, s4.y * isc, s4.z * isc, s4.w * isc);
                        float* dp;
                        if (k < K3) dp = a + (long)(s_i * K3 + k) * 64 + c4;
                        else if (k < K3 + K2) dp = b + (long)(s_i * K2 + (k - K3)) * 64 + c4;
                        else dp = c + (long)(s_i * K1 + (k - K3 - K2)) * 64 + c4;
                        *reinterpret_cast<float4*>(dp) = v4;
                        acc += (AccT)v4.x * (AccT)v4.x + (AccT)v4.y * (AccT)v4.y +
                               (AccT)v4.z * (AccT)v4.z + (AccT)v4.w * (AccT)v4.w;
                    }
                } else {
                    const int tot = tl.cnt * C;
                    for (int j = tid; j < tot; j += 256) {
                        const int r_loc = j / C, d = j - r_loc * C;
                        const int row = tl.off + r_loc, s_i = row / K, k = row - s_i * K;
                        const T v = gWu[(long)(s_i * POLY_U + POLY_K_TO_U[k]) * C + d] * isc;
                        if (k < K3) a[(long)(s_i * K3 + k) * C + d] = v;
                        else if (k < K3 + K2) b[(long)(s_i * K2 + (k - K3)) * C + d] = v;
                        else c[(long)(s_i * K1 + (k - K3 - K2)) * C + d] = v;
                        acc += (AccT)v * (AccT)v;
                    }
                }
            } else {
                const int tot = tl.cnt * C;
                for (int j = tid; j < tot; j += 256) {
                    const int r_loc = j / C, d = j - r_loc * C;
                    const int row = tl.off + r_loc, s_i = row / K, k = row - s_i * K;
                    const T v = gWu[(long)(s_i * POLY_U + POLY_K_TO_U[k]) * C + d] * isc;
                    if (k < K3) a[(long)(s_i * K3 + k) * C + d] = v;
                    else if (k < K3 + K2) b[(long)(s_i * K2 + (k - K3)) * C + d] = v;
                    else c[(long)(s_i * K1 + (k - K3 - K2)) * C + d] = v;
                    acc += (AccT)v * (AccT)v;
                }
            }
        } else {
            for (int j = tid; j < tl.cnt; j += 256) {
                const int idx = tl.off + j;
                const int c = idx / ZW;
                const int col = idx - c * ZW;
                T* p;
                T f;
                if (col == 0) { p = wr0 + c; f = f0 * isc; }
                else if (col <= C) { p = wu1 + c * C + col - 1; f = f1 * isc; }
                else { p = wsk1 + c * (ZW - 1 - C) + col - 1 - C; f = f2 * isc; }
                const T v = gWcat[idx] * f;
                *p = v;
                acc += (AccT)v * (AccT)v;
            }
        }
    }

    if (out_norm2) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if ((tid & 31) == 0) sm_red[tid >> 5] = acc;
        __syncthreads();
        if (tid == 0) {
            double bacc = 0.0;
#pragma unroll
            for (int w = 0; w < 8; ++w) bacc += (double)sm_red[w];
            if (bacc != 0.0) atomicAdd(out_norm2, bacc);
            if (gf_done && pdesc) {
                __threadfence();
                is_last = (atomicAdd(gf_done, 1) == (int)gridDim.x - 1) ? 1 : 0;
            }
        }
        if (gf_done && pdesc) {
            __syncthreads();
            if (is_last && tid < 32) {
                if (tid == 0) *gf_done = 0;
                const double n2 = (tid == 0) ? atomicAdd(out_norm2, 0.0) : 0.0;
                adam_prep_warp<T>(pdesc, np, n2, lr, max_norm, adam_b1, adam_b2, ema_decay, step_idx, sc, urms, caut);
            }
        }
    }
}

// torch.optim.Adam (amsgrad, L2 weight decay; the fused kernel's formula in the parameters' precision),
// gradient clipping coefficient, EMA of the updated parameters
template <typename T>
__global__ void k_adam(const PDesc<T>* d, int np, const T* sc, const double* wd, T b1, T b2, T eps,
                       const double* urms, double uclip, T mu, int nesterov, int soap, const double* caut,
                       const double* lr, const int* step_idx, double mwd, int skip_soap = 0) {
    const PDesc<T> q = d[blockIdx.y];
    if (q.muon) {
        if (skip_soap && soap) return;
        const T coef = sc[3 * np];
        // decoupled weight decay (muon.py muon_wd, DPA4): p *= 1 - lr mult wd before the Muon / SOAP step
        const T dec = mwd > 0 ? T(1.0 - lr[16L * step_idx[0] + q.group] * mwd) : T(1);
        for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < q.n; i += (long)gridDim.x * blockDim.x) {
            if (mwd > 0) q.p[i] *= dec;
            const T g = q.g[i] * coef;   // no L2 weight decay: Muon's step size does not shrink with g
            if (soap) {                  // SOAP: Adam's first moment; the direction slot holds the clipped gradient
                q.m[i] = mu * q.m[i] + (T(1) - mu) * g;
                q.dir[i] = g;
                continue;
            }
            const T m = mu * q.m[i] + g;
            q.m[i] = m;
            q.dir[i] = nesterov ? g + mu * m : m;
        }
        return;
    }
    T step_size = sc[3 * blockIdx.y];
    const T bc2s = sc[3 * blockIdx.y + 1];
    if (urms) {
        const double r = sqrt(urms[blockIdx.y] / (double)(q.n > 0 ? q.n : 1)) / uclip;
        if (r > 1.0) step_size = T((double)step_size / r);
    }
    const T coef = sc[3 * np], omd = sc[3 * np + 1];
    const T w = T(wd[q.group]);
    // cautious (muon.py cautious='adam'): the step -s m / denom opposes the gradient iff m g > 0
    const T cs = caut ? T((double)q.n / (caut[blockIdx.y] + 1.0)) : T(1);
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < q.n; i += (long)gridDim.x * blockDim.x) {
        T p = q.p[i];
        const T g0 = q.g[i] * coef;
        T g = g0;
        if (w != T(0)) g += w * p;
        T m = b1 * q.m[i] + (T(1) - b1) * g;
        T v = b2 * q.v[i] + (T(1) - b2) * g * g;
        q.m[i] = m;
        q.v[i] = v;
        if (q.vmax) {
            v = v > q.vmax[i] ? v : q.vmax[i];
            q.vmax[i] = v;
        }
        T denom = sqrt(v) / bc2s + eps;
        if (caut) {
            if (m * g0 > T(0)) p -= cs * (step_size * m / denom);
        } else {
            p -= step_size * m / denom;
        }
        q.p[i] = p;
        if (q.ema && omd != T(0)) q.ema[i] = (omd == T(1)) ? p : (q.ema[i] - omd * (q.ema[i] - p));
    }
}

// Wave 39: Compact fused Cautious-Adam kernel for non-SOAP parameters ONLY (LST_ADAM_COMPACT=1).
// Replaces 416-CTA k_caut_count + 416-CTA k_adam (832 mostly-dead CTAs, 11.5 us!) with 1 CTA per non-SOAP tensor
// (~2-6 CTAs total, ~1.5 us), using exact integer warp-ballot popc (ZERO global d_caut atomics!).
template <typename T>
__global__ void __launch_bounds__(256) k_caut_adam_compact(
    const PDesc<T>* __restrict__ d, const int* __restrict__ adam_idx, int np, const T* __restrict__ sc,
    const double* __restrict__ wd, T b1, T b2, T eps, int do_caut) {
    __shared__ int sm_c[8];
    const int pid = adam_idx[blockIdx.x];
    const PDesc<T> q = d[pid];
    const int tid = threadIdx.x;
    const T step_size = sc[3 * pid];
    const T bc2s = sc[3 * pid + 1];
    const T coef = sc[3 * np];
    const T omd = sc[3 * np + 1];
    const T w = T(wd[q.group]);
    const bool do_ema = (q.ema != nullptr && omd != T(0));
    const bool copy_ema = (omd == T(1));
    if (!do_caut) {
        for (long i = tid; i < q.n; i += 256) {
            T p = q.p[i];
            const T g0 = q.g[i] * coef;
            T g = (w != T(0)) ? (g0 + w * p) : g0;
            const T m = b1 * q.m[i] + (T(1) - b1) * g;
            T v = b2 * q.v[i] + (T(1) - b2) * g * g;
            q.m[i] = m;
            q.v[i] = v;
            if (q.vmax) { v = v > q.vmax[i] ? v : q.vmax[i]; q.vmax[i] = v; }
            p -= step_size * m / (sqrt(v) / bc2s + eps);
            q.p[i] = p;
            if (do_ema) q.ema[i] = copy_ema ? p : (q.ema[i] - omd * (q.ema[i] - p));
        }
        return;
    }
    if (q.n <= 256) {
        T p = T(0), m = T(0), v = T(0);
        int pos = 0;
        if (tid < q.n) {
            p = q.p[tid];
            const T g0 = q.g[tid] * coef;
            const T g = (w != T(0)) ? (g0 + w * p) : g0;
            m = b1 * q.m[tid] + (T(1) - b1) * g;
            v = b2 * q.v[tid] + (T(1) - b2) * g * g;
            q.m[tid] = m;
            q.v[tid] = v;
            if (q.vmax) { v = v > q.vmax[tid] ? v : q.vmax[tid]; q.vmax[tid] = v; }
            if (m * g0 > T(0)) pos = 1;
        }
        const int wcnt = __popc(__ballot_sync(0xffffffffu, pos != 0));
        if ((tid & 31) == 0) sm_c[tid >> 5] = wcnt;
        __syncthreads();
        int tot = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) tot += sm_c[k];
        if (tid < q.n) {
            if (pos) {
                const T cs = T((double)q.n / ((double)tot + 1.0));
                p -= cs * (step_size * m / (sqrt(v) / bc2s + eps));
            }
            q.p[tid] = p;
            if (do_ema) q.ema[tid] = copy_ema ? p : (q.ema[tid] - omd * (q.ema[tid] - p));
        }
        return;
    }
    int lcnt = 0;
    for (long i = tid; i < q.n; i += 256) {
        const T g0 = q.g[i] * coef;
        const T g = (w != T(0)) ? (g0 + w * q.p[i]) : g0;
        const T m = b1 * q.m[i] + (T(1) - b1) * g;
        if (m * g0 > T(0)) ++lcnt;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) lcnt += __shfl_xor_sync(0xffffffffu, lcnt, o);
    if ((tid & 31) == 0) sm_c[tid >> 5] = lcnt;
    __syncthreads();
    int tot = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) tot += sm_c[k];
    const T cs = T((double)q.n / ((double)tot + 1.0));
    for (long i = tid; i < q.n; i += 256) {
        T p = q.p[i];
        const T g0 = q.g[i] * coef;
        const T g = (w != T(0)) ? (g0 + w * p) : g0;
        const T m = b1 * q.m[i] + (T(1) - b1) * g;
        T v = b2 * q.v[i] + (T(1) - b2) * g * g;
        q.m[i] = m;
        q.v[i] = v;
        if (q.vmax) { v = v > q.vmax[i] ? v : q.vmax[i]; q.vmax[i] = v; }
        if (m * g0 > T(0)) p -= cs * (step_size * m / (sqrt(v) / bc2s + eps));
        q.p[i] = p;
        if (do_ema) q.ema[i] = copy_ema ? p : (q.ema[i] - omd * (q.ema[i] - p));
    }
}

// cautious Adam: per Adam tensor, the number of components with m_new * g > 0 (k_adam's m_new, clipped g)
template <typename T>
__global__ void k_caut_count(const PDesc<T>* d, int np, const T* sc, const double* wd, T b1, double* caut) {
    const PDesc<T> q = d[blockIdx.y];
    if (q.muon) return;
    const T coef = sc[3 * np];
    const T w = T(wd[q.group]);
    double acc = 0.0;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < q.n; i += (long)gridDim.x * blockDim.x) {
        const T g0 = q.g[i] * coef;
        T g = g0;
        if (w != T(0)) g += w * q.p[i];
        const T m = b1 * q.m[i] + (T(1) - b1) * g;
        if (m * g0 > T(0)) acc += 1.0;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(caut + blockIdx.y, acc);
}

// StableAdamW (Wortsman et al. 2023): per tensor, sum over elements of g^2 / max(v_hat, eps^2) with this step's
// second moment (the same g, v and vmax as k_adam forms); k_adam divides the step by max(1, RMS / clip)
template <typename T>
__global__ void k_urms(const PDesc<T>* d, int np, const T* sc, const double* wd, T b2, T eps, double* urms) {
    const PDesc<T> q = d[blockIdx.y];
    if (q.muon) return;
    const T bc2s = sc[3 * blockIdx.y + 1];
    const T coef = sc[3 * np];
    const T w = T(wd[q.group]);
    const T e2 = eps * eps;
    double acc = 0.0;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < q.n; i += (long)gridDim.x * blockDim.x) {
        T g = q.g[i] * coef;
        if (w != T(0)) g += w * q.p[i];
        T v = b2 * q.v[i] + (T(1) - b2) * g * g;
        if (q.vmax) v = v > q.vmax[i] ? v : q.vmax[i];
        T vh = v / (bc2s * bc2s);
        acc += (double)(g * g) / (double)(vh > e2 ? vh : e2);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(urms + blockIdx.y, acc);
}

// Muon: one block per matrix. X = D / |D|_F (m x n, m <= n), five quintic Newton-Schulz iterations
// A = X X^T, B = b A + c A A, X = a X + B X (workspace X, Y, A, B in global memory), then
// p -= lr * scale * sqrt(max(rows, cols)) * X back in the parameter's layout.
template <typename T>
__global__ void __launch_bounds__(256) k_muon_ns(const MuonMat* mats, const PDesc<T>* d, const double* lr,
                                                 const int* step_idx, T* ws, double scale, int steps,
                                                 int polish, T* nvb, double nbeta) {
    const MuonMat mt = mats[blockIdx.x];
    const PDesc<T> q = d[mt.param];
    const int m = mt.tr ? mt.cols : mt.rows, n = mt.tr ? mt.rows : mt.cols;
    T* X = ws + mt.ws;
    T* Y = X + (long)m * n;
    T* A = Y + (long)m * n;
    T* B = A + (long)m * m;
    __shared__ double red[32];
    double ss = 0.0;
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        const int i = idx / n, j = idx % n;
        const int r = mt.tr ? j : i, cc = mt.tr ? i : j;
        const T v = q.dir[mt.off + r * mt.rs + cc * mt.cs];
        X[idx] = v;
        ss += (double)v * (double)v;
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) t += red[w];
        red[0] = t;
    }
    __syncthreads();
    const T inv = T(1) / (T(sqrt(red[0])) + T(1e-7));
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) X[idx] *= inv;
    __syncthreads();
    for (int it = 0; it < steps + polish; ++it) {
        const bool pol = it >= steps;
        const T ca = pol ? T(2) : T(3.4445), cb = pol ? T(-1.5) : T(-4.7750), cc2 = pol ? T(0.5) : T(2.0315);
        for (int idx = threadIdx.x; idx < m * m; idx += blockDim.x) {
            const int i = idx / m, j = idx % m;
            const T* xi = X + (long)i * n;
            const T* xj = X + (long)j * n;
            T s = T(0);
            for (int k = 0; k < n; ++k) s += xi[k] * xj[k];
            A[idx] = s;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < m * m; idx += blockDim.x) {
            const int i = idx / m, j = idx % m;
            T s = T(0);
            for (int k = 0; k < m; ++k) s += A[i * m + k] * A[k * m + j];
            B[idx] = cb * A[idx] + cc2 * s;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
            const int i = idx / n, j = idx % n;
            T s = T(0);
            for (int k = 0; k < m; ++k) s += B[i * m + k] * X[(long)k * n + j];
            Y[idx] = ca * X[idx] + s;
        }
        __syncthreads();
        T* t = X; X = Y; Y = t;
    }
    const double f = lr[16L * step_idx[0] + q.group] * scale * sqrt((double)(mt.rows > mt.cols ? mt.rows : mt.cols));
    muon_apply<T>(mt, q, X, Y, T(f), nvb, nbeta, m, n);
}

// FP32 Muon with min(rows, cols) <= 64: the same iteration with 64 x 64 tiles in shared memory (X in column
// chunks of 64, A and B whole; rows beyond m are zero) and 4 x 4 outputs per thread.
namespace mu {
constexpr int TS = 64, LD = 65;
constexpr size_t SMEM = sizeof(float) * 3 * TS * LD;
__device__ inline void load_chunk(float* Xc, const float* X, int m, int n, int k0) {
    for (int idx = threadIdx.x; idx < TS * TS; idx += blockDim.x) {
        const int r = idx / TS, c = idx % TS;
        Xc[r * LD + c] = (r < m && k0 + c < n) ? X[(long)r * n + k0 + c] : 0.f;
    }
}
}  // namespace mu

__global__ void __launch_bounds__(256) k_muon_ns_f(const MuonMat* mats, const PDesc<float>* d, const double* lr,
                                                   const int* step_idx, float* ws, double scale, int steps,
                                                   int polish, float* nvb, double nbeta) {
    using namespace mu;
    extern __shared__ float smem[];
    float* Xc = smem;
    float* As = smem + TS * LD;
    float* Bs = As + TS * LD;
    const MuonMat mt = mats[blockIdx.x];
    const PDesc<float> q = d[mt.param];
    const int m = mt.tr ? mt.cols : mt.rows, n = mt.tr ? mt.rows : mt.cols;
    float* X = ws + mt.ws;
    float* Y = X + (long)m * n;
    __shared__ double red[8];
    double ss = 0.0;
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        const int i = idx / n, j = idx % n;
        const int r = mt.tr ? j : i, cc = mt.tr ? i : j;
        const float v = q.dir[mt.off + r * mt.rs + cc * mt.cs];
        X[idx] = v;
        ss += (double)v * (double)v;
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    double tot = 0.0;
    for (int w = 0; w < 8; ++w) tot += red[w];
    const float inv = 1.f / ((float)sqrt(tot) + 1e-7f);
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) X[idx] *= inv;
    __syncthreads();
    const int r0 = (threadIdx.x / 16) * 4, c0 = (threadIdx.x % 16) * 4;
    for (int it = 0; it < steps + polish; ++it) {
        const bool pol = it >= steps;
        const float ca = pol ? 2.f : 3.4445f, cb = pol ? -1.5f : -4.7750f, cc2 = pol ? 0.5f : 2.0315f;
        float acc[4][4];
        // A = X X^T
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
        for (int k0 = 0; k0 < n; k0 += TS) {
            load_chunk(Xc, X, m, n, k0);
            __syncthreads();
#pragma unroll 8
            for (int kk = 0; kk < TS; ++kk) {
                float a[4], b[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) a[i] = Xc[(r0 + i) * LD + kk];
#pragma unroll
                for (int j = 0; j < 4; ++j) b[j] = Xc[(c0 + j) * LD + kk];
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
            for (int j = 0; j < 4; ++j) As[(r0 + i) * LD + c0 + j] = acc[i][j];
        __syncthreads();
        // B = b A + c A A
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
#pragma unroll 8
        for (int kk = 0; kk < TS; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[(r0 + i) * LD + kk];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = As[kk * LD + c0 + j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                Bs[(r0 + i) * LD + c0 + j] = cb * As[(r0 + i) * LD + c0 + j] + cc2 * acc[i][j];
        __syncthreads();
        // Y = a X + B X, by column chunks
        for (int k0 = 0; k0 < n; k0 += TS) {
            load_chunk(Xc, X, m, n, k0);
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
#pragma unroll 8
            for (int kk = 0; kk < TS; ++kk) {
                float a[4], b[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) a[i] = Bs[(r0 + i) * LD + kk];
#pragma unroll
                for (int j = 0; j < 4; ++j) b[j] = Xc[kk * LD + c0 + j];
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = r0 + i, cg = k0 + c0 + j;
                    if (r < m && cg < n) Y[(long)r * n + cg] = ca * Xc[r * LD + c0 + j] + acc[i][j];
                }
            __syncthreads();
        }
        float* t = X;
        X = Y;
        Y = t;
    }
    const float ft = (float)(lr[16L * step_idx[0] + q.group] * scale *
                             sqrt((double)(mt.rows > mt.cols ? mt.rows : mt.cols)));
    __syncthreads();
    muon_apply<float>(mt, q, X, Y, ft, nvb, nbeta, m, n);
}


// ============================================================================ SOAP (lst_set_soap)
// Per Muon matrix, in the orientation X (m x n, m = min side <= 64): Lm = EMA X X^T, Rn = EMA X^T X (pn: n <= max_dim,
// else Qn = I), every `every` steps Qm / Qn from cyclic Jacobi on Q^T L Q (warm start), V = EMA (Qm^T X Qn)^2,
// U = Qm [(Qm^T M Qn) / bc1 / (sqrt(V / bc2) + eps)] Qn^T, p -= lr * mult * U (muon.py MuonAdam._soap, jacobi).
struct SoapMat {
    int param, m, n, pm, pn;      // pm / pn: precondition the m / n side (else its basis stays the identity)
    int nch;                      // column chunks of 64 (wide: !pn, n > 64, split on; else 1)
    long off, xi, xj;             // element (i, j) of X at off + i * xi + j * xj in the parameter
    long Lm, Qm, Bm, Rn, Qn, Bn, V, Mr, W1, W2, W3, W4;    // workspace offsets (Wave 36: Mr = cached rotated M_rot)
    long Lp;                      // nch > 1 and pm: nch partial m x m factors (chunk c's EMA of Xc Xc^T); Lm = sum
    long Wn;                      // pn: n x n scratch of k_soap_eig's n side
    long Qm_bf, Qn_bf;            // Wave 21: BF16 shadow copies of Qm / Qn (2x less DRAM read every step!)
};

namespace sp {
constexpr int TS = 64, LD = 65;
constexpr int LDV = TS + 4;      // bmm's k-major tile rows (16-byte aligned); the second tile starts at As + TS * LDV

template <typename T>
__device__ __forceinline__ void ld4(T (&v)[4], const T* p) {
    if constexpr (sizeof(T) == 4) {
        const float4 x = *reinterpret_cast<const float4*>(p);
        v[0] = x.x;
        v[1] = x.y;
        v[2] = x.z;
        v[3] = x.w;
    } else {
        const double2 x = reinterpret_cast<const double2*>(p)[0], y = reinterpret_cast<const double2*>(p)[1];
        v[0] = x.x;
        v[1] = x.y;
        v[2] = y.x;
        v[3] = y.y;
    }
}

// C[M x N] = alpha op(A) op(B) + beta C (beta 0: C not read), M <= 64; N, K in chunks of 64.
// op(A)(i, k) = A[i * ai + k * ak], op(B)(k, j) = B[k * bi + j * bj], C(i, j) = C[i * ldc + j]. C must not alias.
// Tiles k-major (As[kk * LDV + i], Bs[kk * LDV + j]): a thread's 4 rows of A and 4 columns of B are two 16-byte
// loads per k step; each loader walks its operand's unit-stride side (A: i when ai == 1; B: k when bi == 1).
template <typename T>
__device__ void bmm(T* C, long ldc, const T* A, long ai, long ak, const T* B, long bi, long bj, int M, int N, int K,
                    T alpha, T beta, T* As, T* Bs) {
    const int r0 = (threadIdx.x / 16) * 4, c0 = (threadIdx.x % 16) * 4;
    const bool arow = ai == 1, bk = bi == 1 && bj != 1;
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
            for (int u = 0; u < PER; ++u) {          // all global loads first, then the shared stores
                const int e = threadIdx.x + u * 256, hi = e / TS, lo = e % TS;
                const int r = arow ? lo : hi, ka = arow ? hi : lo;
                va[u] = (r < M && k0 + ka < K) ? A[r * ai + (long)(k0 + ka) * ak] : T(0);
                const int kb = bk ? lo : hi, c = bk ? hi : lo;
                vb[u] = (k0 + kb < K && n0 + c < N) ? B[(long)(k0 + kb) * bi + (long)(n0 + c) * bj] : T(0);
            }
#pragma unroll
            for (int u = 0; u < PER; ++u) {
                const int e = threadIdx.x + u * 256, hi = e / TS, lo = e % TS;
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
                const int r = r0 + i, c = n0 + c0 + j;
                if (r < M && c < N) {
                    T* o = C + (long)r * ldc + c;
                    *o = beta != T(0) ? alpha * acc[i][j] + beta * *o : alpha * acc[i][j];
                }
            }
    }
    __syncthreads();
}

// Eigenvectors of the symmetric d x d matrix Bg (= Q^T L Q) by cyclic Jacobi, round-robin pairs (muon.py _jacobi):
// Qg <- Qg J per round until the off-diagonal Frobenius norm <= tol times the whole (at most maxsw sweeps).
template <typename T>
__device__ void jacobi(const T* Bg, T* Qg, int d, int maxsw, double tol, T* Bs, T* Qs) {
    __shared__ int idx[TS + 1];
    __shared__ T cs_[TS / 2 + 1], sn_[TS / 2 + 1];
    __shared__ int pp_[TS / 2 + 1], qq_[TS / 2 + 1];
    __shared__ double jred[2][32];
    const int nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int e = threadIdx.x; e < d * d; e += blockDim.x) {
        const int i = e / d, j = e % d;
        Bs[i * LD + j] = Bg[e];
        Qs[i * LD + j] = Qg[e];
    }
    const int n = d + (d & 1), np = n / 2;
    for (int i = threadIdx.x; i < n; i += blockDim.x) idx[i] = i;
    __syncthreads();
    if (d < 2) return;
    for (int sw = 0; sw < maxsw; ++sw) {
        double off = 0.0, tot = 0.0;
        for (int e = threadIdx.x; e < d * d; e += blockDim.x) {
            const int i = e / d, j = e % d;
            const double x = (double)Bs[i * LD + j];
            tot += x * x;
            if (i != j) off += x * x;
        }
        for (int o = 16; o > 0; o >>= 1) {
            off += __shfl_xor_sync(0xffffffffu, off, o);
            tot += __shfl_xor_sync(0xffffffffu, tot, o);
        }
        if (lane == 0) {
            jred[0][warp] = off;
            jred[1][warp] = tot;
        }
        __syncthreads();
        off = 0.0;
        tot = 0.0;
        for (int w = 0; w < nw; ++w) {
            off += jred[0][w];
            tot += jred[1][w];
        }
        __syncthreads();
        if (off <= tol * tol * tot) break;
        for (int rd = 0; rd < n - 1; ++rd) {
            if (threadIdx.x < np) {
                int p = idx[threadIdx.x], q = idx[n - 1 - threadIdx.x];
                if (p > q) {
                    const int t = p;
                    p = q;
                    q = t;
                }
                if (q < d) {
                    const T app = Bs[p * LD + p], aqq = Bs[q * LD + q], apq = Bs[p * LD + q];
                    const T th = T(0.5) * atan2(T(2) * apq, aqq - app);
                    cs_[threadIdx.x] = cos(th);
                    sn_[threadIdx.x] = sin(th);
                    pp_[threadIdx.x] = p;
                    qq_[threadIdx.x] = q;
                } else {
                    pp_[threadIdx.x] = -1;
                }
            }
            __syncthreads();
            for (int e = threadIdx.x; e < np * d; e += blockDim.x) {      // rows: B <- J^T B
                const int k = e / d, j = e % d, p = pp_[k];
                if (p < 0) continue;
                const int q = qq_[k];
                const T c = cs_[k], s = sn_[k], bp = Bs[p * LD + j], bq = Bs[q * LD + j];
                Bs[p * LD + j] = c * bp - s * bq;
                Bs[q * LD + j] = s * bp + c * bq;
            }
            __syncthreads();
            for (int e = threadIdx.x; e < np * d; e += blockDim.x) {      // columns: B <- B J, Q <- Q J
                const int k = e / d, i = e % d, p = pp_[k];
                if (p < 0) continue;
                const int q = qq_[k];
                const T c = cs_[k], s = sn_[k];
                const T bp = Bs[i * LD + p], bq = Bs[i * LD + q];
                Bs[i * LD + p] = c * bp - s * bq;
                Bs[i * LD + q] = s * bp + c * bq;
                const T qp = Qs[i * LD + p], qq = Qs[i * LD + q];
                Qs[i * LD + p] = c * qp - s * qq;
                Qs[i * LD + q] = s * qp + c * qq;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                const int last = idx[n - 1];
                for (int i = n - 1; i >= 2; --i) idx[i] = idx[i - 1];
                idx[1] = last;
            }
            __syncthreads();
        }
    }
    for (int e = threadIdx.x; e < d * d; e += blockDim.x) Qg[e] = Qs[(e / d) * LD + e % d];
    __syncthreads();
}
// jacobi with one pass per round: the round-robin pairs from the round index (idx[0] = 0, idx[i] = 1 + (i - 1 - rd)
// mod (n - 1)), then J^T B J on each 2 x 2 block (pair a's rows, pair b's columns: jacobi's row rotation, then its
// column rotation) and Q J in the same pass. 2 barriers per round instead of 4; each thread keeps one column pair
// of the blocks and one row of Q for the whole call (no divisions or address arithmetic in the rounds).
template <typename T>
struct alignas(2 * sizeof(T)) JRot {
    T c, s;
};

template <typename T, bool FAST_CHK = false, bool FAST_ALG = false, bool SORT_DESC = false>
__device__ void jacobi_f(const T* Bg, T* Qg, int d, int maxsw, double tol, T* Bs, T* Qs, bool preloaded = false) {
    __shared__ JRot<T> rot_[TS / 2 + 1];
    __shared__ int2 pq_[TS / 2 + 1];
    using RedT = typename std::conditional<FAST_CHK && std::is_same<T, float>::value, float, double>::type;
    __shared__ RedT jred[2][32];
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
    if (d < 2) return;
    const RedT tol2 = (RedT)(tol * tol);
    for (int sw = 0; sw < maxsw; ++sw) {
        RedT off = RedT(0), tot = RedT(0);
        for (int e = threadIdx.x; e < d * d; e += blockDim.x) {
            const int i = e / d, j = e % d;
            const RedT x = (RedT)Bs[i * LD + j];
            const RedT xx = x * x;
            tot += xx;
            if (i != j) off += xx;
        }
        for (int o = 16; o > 0; o >>= 1) {
            off += __shfl_xor_sync(0xffffffffu, off, o);
            tot += __shfl_xor_sync(0xffffffffu, tot, o);
        }
        if (lane == 0) {
            jred[0][warp] = off;
            jred[1][warp] = tot;
        }
        __syncthreads();
        off = RedT(0);
        tot = RedT(0);
        for (int w = 0; w < nw; ++w) {
            off += jred[0][w];
            tot += jred[1][w];
        }
        __syncthreads();
        if (off <= tol2 * tot) break;
        for (int rd = 0; rd < n1; ++rd) {
            if (threadIdx.x < np) {
                const int k = threadIdx.x;
                int p = k == 0 ? 0 : 1 + (k - 1 - rd + n1) % n1;
                int q = 1 + (n1 - 1 - k - rd + n1) % n1;          // idx[n - 1 - k]
                if (p > q) {
                    const int t = p;
                    p = q;
                    q = t;
                }
                pq_[k] = make_int2(p, q);
                if (q < d) {
                    const T app = Bs[p * LD + p], aqq = Bs[q * LD + q], apq = Bs[p * LD + q];
                    JRot<T> r;
                    if constexpr (FAST_ALG && std::is_same<T, float>::value) {
                        if (fabsf(apq) <= 1e-12f * (fabsf(app) + fabsf(aqq))) {
                            r.c = 1.f;
                            r.s = 0.f;
                        } else {
                            const float tau = (aqq - app) / (2.f * apq);
                            const float t = fabsf(tau) > 1e7f
                                                ? (0.5f / tau)
                                                : (copysignf(1.f, tau) / (fabsf(tau) + sqrtf(1.f + tau * tau)));
                            r.c = rsqrtf(1.f + t * t);
                            r.s = t * r.c;
                        }
                    } else {
                        const T th = T(0.5) * atan2(T(2) * apq, aqq - app);
                        r.c = cos(th);
                        r.s = sin(th);
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
                        x00 = y00;
                        x10 = y10;
                        x01 = y01;
                        x11 = y11;
                    }
                    if (cb) {
                        const T y00 = rc.c * x00 - rc.s * x01, y01 = rc.s * x00 + rc.c * x01;
                        const T y10 = rc.c * x10 - rc.s * x11, y11 = rc.s * x10 + rc.c * x11;
                        x00 = y00;
                        x01 = y01;
                        x10 = y10;
                        x11 = y11;
                    }
                    rp[pqc.x] = x00;
                    if (cb) rp[pqc.y] = x01;
                    if (ra) {
                        rq[pqc.x] = x10;
                        if (cb) rq[pqc.y] = x11;
                    }
                }
            }
            if (qact)
                for (int k = qk0; k < np; k += qstep) {      // Q <- Q J
                    const int2 pq = pq_[k];
                    if (pq.y >= d) continue;
                    const JRot<T> r = rot_[k];
                    const T qp = qrow[pq.x], qv = qrow[pq.y];
                    qrow[pq.x] = r.c * qp - r.s * qv;
                    qrow[pq.y] = r.s * qp + r.c * qv;
                }
            __syncthreads();
        }
    }
    if constexpr (SORT_DESC && std::is_same<T, float>::value) {
        __shared__ float s_ev[64];
        __shared__ int s_perm[64];
        if (threadIdx.x < d) s_ev[threadIdx.x] = Bs[threadIdx.x * LD + threadIdx.x];
        __syncthreads();
        if (threadIdx.x < d) {
            const int old_c = threadIdx.x;
            const float ev = s_ev[old_c];
            int rank = 0;
            for (int k = 0; k < d; ++k) {
                const float ek = s_ev[k];
                if (ek > ev || (ek == ev && k < old_c)) ++rank;
            }
            s_perm[rank] = old_c;
        }
        __syncthreads();
        for (int e = threadIdx.x; e < d * d; e += blockDim.x) {
            const int r = e / d, c = e % d;
            Qg[e] = Qs[r * LD + s_perm[c]];
        }
        __syncthreads();
        return;
    }
    if constexpr (FAST_CHK && std::is_same<T, float>::value) {
        if (d == 64 && (((uintptr_t)Qg & 15) == 0)) {
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int idx = threadIdx.x + g * 256;
                const int r = idx >> 4, c4 = (idx & 15) * 4;
                *reinterpret_cast<float4*>(Qg + (long)r * 64 + c4) =
                    make_float4(Qs[r * LD + c4], Qs[r * LD + c4 + 1], Qs[r * LD + c4 + 2], Qs[r * LD + c4 + 3]);
            }
            __syncthreads();
            return;
        }
    }
    for (int e = threadIdx.x; e < d * d; e += blockDim.x) Qg[e] = Qs[(e / d) * LD + e % d];
    __syncthreads();
}

template <typename T>
__device__ void jacobi_any(const T* Bg, T* Qg, int d, int maxsw, double tol, T* Bs, T* Qs, int fused) {
    if (fused)
        jacobi_f<T>(Bg, Qg, d, maxsw, tol, Bs, Qs);
    else
        jacobi<T>(Bg, Qg, d, maxsw, tol, Bs, Qs);
}

// Shared-memory 64x64 GEMM (LST_SOAP_SMEM=1): acc[i][j] = sum_{k<K} A_cm[k*LDV + r0+i] * B_rm[k*LDV + c0+j].
// Uses 2 tiles of 64x68 (34.8 KB in FP32 -> 2 blocks/SM on sm_120 and conflict-free col-major stores!).
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

// Tensor-Core TF32 64x64 GEMM (LST_SOAP_TF32=1): 8 warps arranged 4x2 (wm=0..3 -> rows 16*wm..+15, wn=0..1 -> cols
// 32*wn..+31), 4 m16n8k8 tiles per warp (32 MMAs vs 1024 FMAs). A_cm & B_rm must be zero-padded for k >= K up to 64.
__device__ __forceinline__ void smem_gemm_tf32(float (&acc)[4][4], const float* A_cm, const float* B_rm, int K) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1;
    const int g = lane >> 2, t = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    for (int k0 = 0; k0 < K; k0 += 8) {
        const int ka = k0 + t, kb = ka + 4;
        const uint32_t a0 = mlpt::tf32(A_cm[ka * LDV + r0]);
        const uint32_t a1 = mlpt::tf32(A_cm[ka * LDV + r1]);
        const uint32_t a2 = mlpt::tf32(A_cm[kb * LDV + r0]);
        const uint32_t a3 = mlpt::tf32(A_cm[kb * LDV + r1]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = 32 * wn + 8 * j + g;
            const uint32_t b0 = mlpt::tf32(B_rm[ka * LDV + col]);
            const uint32_t b1 = mlpt::tf32(B_rm[kb * LDV + col]);
            mlpt::mma_tf32(acc[j], a0, a1, a2, a3, b0, b1);
        }
    }
}

// Helper: store TF32 accumulator acc[4][4] into smem S (row-major if !TRANS, col-major if TRANS) with zero-padding.
template <bool TRANS>
__device__ __forceinline__ void store_tf32_smem(float* S, const float (&acc)[4][4], int m, int nc, int ld = LDV) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1;
    const int g = lane >> 2, t = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = 32 * wn + 8 * j + 2 * t, c1 = c0 + 1;
        const float v00 = (r0 < m && c0 < nc) ? acc[j][0] : 0.f;
        const float v01 = (r0 < m && c1 < nc) ? acc[j][1] : 0.f;
        const float v10 = (r1 < m && c0 < nc) ? acc[j][2] : 0.f;
        const float v11 = (r1 < m && c1 < nc) ? acc[j][3] : 0.f;
        if constexpr (!TRANS) {
            S[r0 * ld + c0] = v00; S[r0 * ld + c1] = v01;
            S[r1 * ld + c0] = v10; S[r1 * ld + c1] = v11;
        } else {
            S[c0 * ld + r0] = v00; S[c1 * ld + r0] = v01;
            S[c0 * ld + r1] = v10; S[c1 * ld + r1] = v11;
        }
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

// Compute B = Q^T (L Q) completely in shared memory and run jacobi_f directly on (Bs, Qs) with ZERO global W1/Bm traffic!
template <bool TF32>
__device__ void smem_qtlq_jacobi(const float* L, float* Q, int d, int maxsw, double tol, float* S0, float* S1) {
    load_sq_smem_T<float>(S0, L, d);    // S0[k * LDV + r] = L[r, k]
    load_sq_smem<float>(S1, Q, d);      // S1[k * LDV + c] = Q[k, c]
    __syncthreads();
    float acc[4][4];
    if constexpr (TF32) {
        smem_gemm_tf32(acc, S0, S1, d);
        __syncthreads();
        store_tf32_smem<false>(S0, acc, d, d, LDV);  // S0 = W1 = L Q (row-major)
        __syncthreads();
        smem_gemm_tf32(acc, S1, S0, d);              // acc = Q^T W1
        __syncthreads();
        store_tf32_smem<false>(S0, acc, d, d, LD);   // Bs[r * LD + c] = Q^T L Q
    } else {
        const int r0 = (threadIdx.x >> 4) * 4, c0 = (threadIdx.x & 15) * 4;
        smem_gemm<float>(acc, S0, S1, d);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                S0[(r0 + i) * LDV + (c0 + j)] = (r0 + i < d && c0 + j < d) ? acc[i][j] : 0.f;
        __syncthreads();
        smem_gemm<float>(acc, S1, S0, d);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (r0 + i < d && c0 + j < d) S0[(r0 + i) * LD + (c0 + j)] = acc[i][j];
    }
    for (int e = threadIdx.x; e < d * d; e += blockDim.x) S1[(e / d) * LD + (e % d)] = Q[e];
    jacobi_f<float>(nullptr, Q, d, maxsw, tol, S0, S1, true);
}

// Wave 14.3: Vectorized float4 smem loaders & fused symmetric Lm+Rn / fast Jacobi SOAP kernels (LST_SOAP_OPT=1|2).
__device__ __forceinline__ void load_mat_smem_f4(float* S, const float* X, long xi, long xj, int m, int nc) {
    if (xj == 1 && m == 64 && nc == 64 && (xi & 3) == 0 && (((uintptr_t)X & 15) == 0)) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int idx = threadIdx.x + g * 256;
            const int r = idx >> 4, c4 = (idx & 15) * 4;
            *reinterpret_cast<float4*>(S + r * LDV + c4) = *reinterpret_cast<const float4*>(X + (long)r * xi + c4);
        }
        return;
    }
    load_mat_smem<float>(S, X, xi, xj, m, nc);
}

__device__ __forceinline__ void load_sq_smem_f4(float* S, const float* Q, int d) {
    if (d == 64 && (((uintptr_t)Q & 15) == 0)) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int idx = threadIdx.x + g * 256;
            const int r = idx >> 4, c4 = (idx & 15) * 4;
            *reinterpret_cast<float4*>(S + r * LDV + c4) = *reinterpret_cast<const float4*>(Q + (long)r * 64 + c4);
        }
        return;
    }
    load_sq_smem<float>(S, Q, d);
}

__device__ __forceinline__ void load_sq_smem_T_f4(float* S, const float* Q, int d) {
    if (d == 64 && (((uintptr_t)Q & 15) == 0)) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int idx = threadIdx.x + g * 256;
            const int r = idx >> 4, c4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(Q + (long)r * 64 + c4);
            S[(c4 + 0) * LDV + r] = v.x;
            S[(c4 + 1) * LDV + r] = v.y;
            S[(c4 + 2) * LDV + r] = v.z;
            S[(c4 + 3) * LDV + r] = v.w;
        }
        return;
    }
    load_sq_smem_T<float>(S, Q, d);
}

template <bool FAST_ALG, bool SORT_DESC = false>
__device__ void smem_qtlq_jacobi_opt(const float* L, float* Q, int d, int maxsw, double tol, float* S0, float* S1) {
    load_sq_smem_T_f4(S0, L, d);
    load_sq_smem_f4(S1, Q, d);
    __syncthreads();
    const int r0 = (threadIdx.x >> 4) * 4, c0 = (threadIdx.x & 15) * 4;
    float acc[4][4];
    smem_gemm<float>(acc, S0, S1, d);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            S0[(r0 + i) * LDV + (c0 + j)] = (r0 + i < d && c0 + j < d) ? acc[i][j] : 0.f;
    __syncthreads();
    smem_gemm<float>(acc, S1, S0, d);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (r0 + i < d && c0 + j < d) S0[(r0 + i) * LD + (c0 + j)] = acc[i][j];
    if (d == 64 && (((uintptr_t)Q & 15) == 0)) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int idx = threadIdx.x + g * 256;
            const int r = idx >> 4, c4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(Q + (long)r * 64 + c4);
            S1[r * LD + c4 + 0] = v.x;
            S1[r * LD + c4 + 1] = v.y;
            S1[r * LD + c4 + 2] = v.z;
            S1[r * LD + c4 + 3] = v.w;
        }
    } else {
        for (int e = threadIdx.x; e < d * d; e += blockDim.x) S1[(e / d) * LD + (e % d)] = Q[e];
    }
    jacobi_f<float, true, FAST_ALG, SORT_DESC>(nullptr, Q, d, maxsw, tol, S0, S1, true);
}
}  // namespace sp

// Rotate columns [c0, c0 + nc) of X and M into the eigenbases, Adam there, rotate back, update p (nc = n when pn).
// When osc != nullptr and q.ema != nullptr, fuses k_muon_ema into the parameter store.
template <typename T>
__device__ void soap_apply(const SoapMat& mt, const PDesc<T>& q, T* ws, int c0, int nc, double t, const double* lr,
                           const int* step_idx, double mult, double b1, double b2, double eps, T* As, T* Bs,
                           const T* osc = nullptr) {
    using namespace sp;
    const int m = mt.m, n = mt.n;
    const long xi = mt.xi, xj = mt.xj;
    const T* X = q.dir + mt.off + (long)c0 * xj;
    const T* Mx = q.m + mt.off + (long)c0 * xj;
    T *Qm = ws + mt.Qm, *Qn = ws + mt.Qn, *V = ws + mt.V + c0;
    T *W1 = ws + mt.W1 + c0, *W2 = ws + mt.W2 + c0, *W3 = ws + mt.W3 + c0, *W4 = ws + mt.W4 + c0;
    const T a2 = T(1.0 - b2);
    bmm<T>(W1, n, Qm, 1, m, X, xi, xj, m, nc, m, T(1), T(0), As, Bs);           // Qm^T X
    bmm<T>(W3, n, Qm, 1, m, Mx, xi, xj, m, nc, m, T(1), T(0), As, Bs);          // Qm^T M
    T* Gr = W1;
    T* Mr = W3;
    if (mt.pn) {
        bmm<T>(W2, n, W1, n, 1, Qn, n, 1, m, n, n, T(1), T(0), As, Bs);
        bmm<T>(W4, n, W3, n, 1, Qn, n, 1, m, n, n, T(1), T(0), As, Bs);
        Gr = W2;
        Mr = W4;
    }
    T bc1, bc2;
    if constexpr (std::is_same<T, float>::value) {       // FP64 pow in every warp cost more than the rotations
        bc1 = 1.f - powf((float)b1, (float)t);
        bc2 = 1.f - powf((float)b2, (float)t);
    } else {
        bc1 = T(1.0 - pow(b1, t));
        bc2 = T(1.0 - pow(b2, t));
    }
    const T te = T(eps);
    // element (i, j) of the m x nc block: thread's column j0 fixed, rows i0, i0 + istep, ...
    const int j0 = threadIdx.x % nc, i0 = threadIdx.x / nc, istep = blockDim.x / nc;
    if (i0 < istep)
        for (int i = i0; i < m; i += istep) {
            const long k = (long)i * n + j0;
            const T g = Gr[k];
            const T v = T(b2) * V[k] + a2 * g * g;
            V[k] = v;
            Gr[k] = (Mr[k] / bc1) / (sqrt(v / bc2) + te);
        }
    __syncthreads();
    T* U;
    if (mt.pn) {
        bmm<T>(W3, n, Qm, m, 1, Gr, n, 1, m, n, m, T(1), T(0), As, Bs);       // Qm Ur
        bmm<T>(W1, n, W3, n, 1, Qn, 1, n, m, n, n, T(1), T(0), As, Bs);       // (Qm Ur) Qn^T
        U = W1;
    } else {
        bmm<T>(W3, n, Qm, m, 1, Gr, n, 1, m, nc, m, T(1), T(0), As, Bs);
        U = W3;
    }
    const T f = T(lr[16L * step_idx[0] + q.group] * mult);
    const T omd = osc ? osc[3 * LST_N_PARAMS + 1] : T(0);
    const bool do_ema = osc && q.ema != nullptr;
    if (i0 < istep)
        for (int i = i0; i < m; i += istep) {
            const long idx = mt.off + i * xi + (long)(c0 + j0) * xj;
            const T pnew = q.p[idx] - f * U[(long)i * n + j0];
            q.p[idx] = pnew;
            if (do_ema) q.ema[idx] -= omd * (q.ema[idx] - pnew);
        }
}

// Shared-memory SOAP apply for m <= 64, nc <= 64 (LST_SOAP_SMEM=1): 2 tiles S0, S1 [64][68] (34.8 KB FP32 -> 2 blk/SM
// on sm_120 and conflict-free stores); zero W1..W4 global round-trips; EMA fused into final epilogue.
template <typename T>
__device__ void soap_apply_smem(const SoapMat& mt, const PDesc<T>& q, T* ws, int c0, int nc, double t, const double* lr,
                                const int* step_idx, double mult, double b1, double b2, double eps, T* S0, T* S1,
                                bool x_in_s0, const T* osc = nullptr) {
    using namespace sp;
    const int m = mt.m, n = mt.n;
    const long xi = mt.xi, xj = mt.xj;
    const T* X = q.dir + mt.off + (long)c0 * xj;
    const T* Mx = q.m + mt.off + (long)c0 * xj;
    const T *Qm = ws + mt.Qm, *Qn = ws + mt.Qn;
    T* V = ws + mt.V + c0;
    const T a2 = T(1.0 - b2);
    const int r0 = (threadIdx.x >> 4) * 4, c0t = (threadIdx.x & 15) * 4;
    if (!x_in_s0) load_mat_smem<T>(S0, X, xi, xj, m, nc);
    load_sq_smem<T>(S1, Qm, m);
    __syncthreads();
    T acc[4][4];
    smem_gemm<T>(acc, S1, S0, m);                     // W1 = Qm^T X
    if (mt.pn) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                S0[(c0t + j) * LDV + (r0 + i)] = (r0 + i < m && c0t + j < nc) ? acc[i][j] : T(0);
        load_sq_smem<T>(S1, Qn, n);
        __syncthreads();
        smem_gemm<T>(acc, S0, S1, n);                 // Gr = W1 Qn
    }
    T bc1, bc2;
    if constexpr (std::is_same<T, float>::value) {
        bc1 = 1.f - powf((float)b1, (float)t);
        bc2 = 1.f - powf((float)b2, (float)t);
    } else {
        bc1 = T(1.0 - pow(b1, t));
        bc2 = T(1.0 - pow(b2, t));
    }
    const T te = T(eps);
    T denom[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = r0 + i, c = c0t + j;
            if (r < m && c < nc) {
                const long k = (long)r * n + c;
                const T g = acc[i][j];
                const T v = T(b2) * V[k] + a2 * g * g;
                V[k] = v;
                denom[i][j] = sqrt(v / bc2) + te;
            }
        }
    __syncthreads();
    load_mat_smem<T>(S0, Mx, xi, xj, m, nc);
    if (mt.pn) load_sq_smem<T>(S1, Qm, m);
    __syncthreads();
    smem_gemm<T>(acc, S1, S0, m);                     // W3 = Qm^T Mx
    if (mt.pn) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                S0[(c0t + j) * LDV + (r0 + i)] = (r0 + i < m && c0t + j < nc) ? acc[i][j] : T(0);
        load_sq_smem<T>(S1, Qn, n);
        __syncthreads();
        smem_gemm<T>(acc, S0, S1, n);                 // Mr = W3 Qn
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = r0 + i, c = c0t + j;
            S0[r * LDV + c] = (r < m && c < nc) ? (acc[i][j] / bc1) / denom[i][j] : T(0);
        }
    load_sq_smem_T<T>(S1, Qm, m);                     // Qm^T col-major = Qm row-major
    __syncthreads();
    smem_gemm<T>(acc, S1, S0, m);                     // W3 = Qm Ur
    if (mt.pn) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                S0[(c0t + j) * LDV + (r0 + i)] = (r0 + i < m && c0t + j < nc) ? acc[i][j] : T(0);
        load_sq_smem_T<T>(S1, Qn, n);                 // Qn^T row-major
        __syncthreads();
        smem_gemm<T>(acc, S0, S1, n);                 // U = W3 Qn^T
    }
    const T f = T(lr[16L * step_idx[0] + q.group] * mult);
    const T omd = osc ? osc[3 * LST_N_PARAMS + 1] : T(0);
    const bool do_ema = osc && q.ema != nullptr;
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

// Shared-memory k_soap_a body (LST_SOAP_SMEM=1): 2 x 64x68 smem (34.8 KB in FP32 -> 2 blocks/SM on sm_120).
template <typename T>
__device__ __forceinline__ void soap_a_smem_body(const SoapMat* mats, const int2* units, const PDesc<T>* d,
                                                 const double* lr, const int* step_idx, T* ws, double mult, double b1,
                                                 double b2, double eps, int every, const T* osc, unsigned char* smem_raw) {
    using namespace sp;
    T* S0 = reinterpret_cast<T*>(smem_raw);
    T* S1 = S0 + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<T> q = d[mt.param];
    const int m = mt.m, n = mt.n;
    const int c0 = mt.nch > 1 ? u.y * TS : 0, nc = mt.nch > 1 ? min(TS, n - c0) : n;
    const T* X = q.dir + mt.off + (long)c0 * mt.xj;
    const long xi = mt.xi, xj = mt.xj;
    const double t = (double)*q.step;
    const T a2 = T(1.0 - b2);
    const int r0 = (threadIdx.x >> 4) * 4, c0t = (threadIdx.x & 15) * 4;
    // Load X once in 4 batches of 4 -> S0 (row-major) and S1 (col-major, for Lm = X X^T)
    {
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
    }
    __syncthreads();
    T acc[4][4];
    if (mt.pm) {
        smem_gemm<T>(acc, S1, S1, nc);                // Lm = X X^T
        T* L = mt.nch > 1 ? ws + mt.Lp + (long)u.y * m * m : ws + mt.Lm;
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = r0 + i, c = c0t + j;
                if (r < m && c < m) L[(long)r * m + c] = a2 * acc[i][j] + T(b2) * L[(long)r * m + c];
            }
    }
    if (mt.pn) {
        smem_gemm<T>(acc, S0, S0, m);                 // Rn = X^T X
        T* R = ws + mt.Rn;
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = r0 + i, c = c0t + j;
                if (r < n && c < n) R[(long)r * n + c] = a2 * acc[i][j] + T(b2) * R[(long)r * n + c];
            }
    }
    if (((long)t - 1) % every == 0) return;           // refresh step: k_soap_b
    __syncthreads();
    soap_apply_smem<T>(mt, q, ws, c0, nc, t, lr, step_idx, mult, b1, b2, eps, S0, S1, true, osc);
}

__global__ void __launch_bounds__(256, 2) k_soap_a_smem_f(const SoapMat* mats, const int2* units,
                                                          const PDesc<float>* d, const double* lr, const int* step_idx,
                                                          float* ws, double mult, double b1, double b2, double eps,
                                                          int every, const float* osc) {
    extern __shared__ unsigned char smem_raw[];
    soap_a_smem_body<float>(mats, units, d, lr, step_idx, ws, mult, b1, b2, eps, every, osc, smem_raw);
}

__global__ void __launch_bounds__(256, 1) k_soap_a_smem_d(const SoapMat* mats, const int2* units,
                                                          const PDesc<double>* d, const double* lr, const int* step_idx,
                                                          double* ws, double mult, double b1, double b2, double eps,
                                                          int every, const double* osc) {
    extern __shared__ unsigned char smem_raw[];
    soap_a_smem_body<double>(mats, units, d, lr, step_idx, ws, mult, b1, b2, eps, every, osc, smem_raw);
}

// Tensor-Core TF32 SOAP apply & k_soap_a (LST_SOAP_TF32=1): replaces all 6-8 64x64 SIMT smem_gemms with mma.sync.m16n8k8!
__device__ void soap_apply_smem_tf32(const SoapMat& mt, const PDesc<float>& q, float* ws, int c0, int nc, double t,
                                     const double* lr, const int* step_idx, double mult, double b1, double b2,
                                     double eps, float* S0, float* S1, bool x_in_s0, const float* osc = nullptr) {
    using namespace sp;
    const int m = mt.m, n = mt.n;
    const long xi = mt.xi, xj = mt.xj;
    const float* X = q.dir + mt.off + (long)c0 * xj;
    const float* Mx = q.m + mt.off + (long)c0 * xj;
    const float *Qm = ws + mt.Qm, *Qn = ws + mt.Qn;
    float* V = ws + mt.V + c0;
    const float a2 = float(1.0 - b2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
    if (!x_in_s0) load_mat_smem<float>(S0, X, xi, xj, m, nc);
    load_sq_smem<float>(S1, Qm, m);
    __syncthreads();
    float acc[4][4];
    smem_gemm_tf32(acc, S1, S0, m);                     // W1 = Qm^T X
    if (mt.pn) {
        __syncthreads();
        store_tf32_smem<true>(S0, acc, m, nc);          // S0 = W1^T
        load_sq_smem<float>(S1, Qn, n);
        __syncthreads();
        smem_gemm_tf32(acc, S0, S1, n);                 // Gr = W1 Qn
    }
    const float bc1 = 1.f - powf((float)b1, (float)t);
    const float bc2 = 1.f - powf((float)b2, (float)t);
    const float te = float(eps);
    float denom[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int cl = 32 * wn + 8 * j + 2 * tl;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int r = (u < 2) ? r0 : r1, c = cl + (u & 1);
            if (r < m && c < nc) {
                const long k = (long)r * n + c;
                const float gv = acc[j][u];
                const float v = float(b2) * V[k] + a2 * gv * gv;
                V[k] = v;
                denom[j][u] = sqrtf(v / bc2) + te;
            }
        }
    }
    __syncthreads();
    load_mat_smem<float>(S0, Mx, xi, xj, m, nc);
    if (mt.pn) load_sq_smem<float>(S1, Qm, m);
    __syncthreads();
    smem_gemm_tf32(acc, S1, S0, m);                     // W3 = Qm^T Mx
    if (mt.pn) {
        __syncthreads();
        store_tf32_smem<true>(S0, acc, m, nc);
        load_sq_smem<float>(S1, Qn, n);
        __syncthreads();
        smem_gemm_tf32(acc, S0, S1, n);                 // Mr = W3 Qn
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int cl = 32 * wn + 8 * j + 2 * tl;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int r = (u < 2) ? r0 : r1, c = cl + (u & 1);
            S0[r * LDV + c] = (r < m && c < nc) ? (acc[j][u] / bc1) / denom[j][u] : 0.f;
        }
    }
    load_sq_smem_T<float>(S1, Qm, m);                   // Qm^T col-major = Qm row-major
    __syncthreads();
    smem_gemm_tf32(acc, S1, S0, m);                     // W3 = Qm Ur
    if (mt.pn) {
        __syncthreads();
        store_tf32_smem<true>(S0, acc, m, nc);
        load_sq_smem_T<float>(S1, Qn, n);               // Qn^T row-major
        __syncthreads();
        smem_gemm_tf32(acc, S0, S1, n);                 // U = W3 Qn^T
    }
    const float f = float(lr[16L * step_idx[0] + q.group] * mult);
    const float omd = osc ? osc[3 * LST_N_PARAMS + 1] : 0.f;
    const bool do_ema = osc && q.ema != nullptr;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int cl = 32 * wn + 8 * j + 2 * tl;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int r = (u < 2) ? r0 : r1, c = cl + (u & 1);
            if (r < m && c < nc) {
                const long idx = mt.off + (long)r * xi + (long)(c0 + c) * xj;
                const float pnew = q.p[idx] - f * acc[j][u];
                q.p[idx] = pnew;
                if (do_ema) q.ema[idx] -= omd * (q.ema[idx] - pnew);
            }
        }
    }
}

__global__ void __launch_bounds__(256, 2) k_soap_a_smem_tf32(const SoapMat* mats, const int2* units,
                                                             const PDesc<float>* d, const double* lr,
                                                             const int* step_idx, float* ws, double mult, double b1,
                                                             double b2, double eps, int every, const float* osc) {
    using namespace sp;
    extern __shared__ unsigned char smem_raw[];
    float* S0 = reinterpret_cast<float*>(smem_raw);
    float* S1 = S0 + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<float> q = d[mt.param];
    const int m = mt.m, n = mt.n;
    const int c0 = mt.nch > 1 ? u.y * TS : 0, nc = mt.nch > 1 ? min(TS, n - c0) : n;
    const float* X = q.dir + mt.off + (long)c0 * mt.xj;
    const long xi = mt.xi, xj = mt.xj;
    const double t = (double)*q.step;
    const float a2 = float(1.0 - b2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
    {
        const bool col_fast = (xi == 1 && xj != 1);
#pragma unroll
        for (int gg = 0; gg < 4; ++gg) {
            float v[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e = threadIdx.x + (gg * 4 + k) * 256, hi = e >> 6, lo = e & 63;
                const int r = col_fast ? lo : hi, c = col_fast ? hi : lo;
                v[k] = (r < m && c < nc) ? X[(long)r * xi + (long)c * xj] : 0.f;
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e = threadIdx.x + (gg * 4 + k) * 256, hi = e >> 6, lo = e & 63;
                const int r = col_fast ? lo : hi, c = col_fast ? hi : lo;
                S0[r * LDV + c] = v[k];
                if (mt.pm) S1[c * LDV + r] = v[k];
            }
        }
    }
    __syncthreads();
    float acc[4][4];
    if (mt.pm) {
        smem_gemm_tf32(acc, S1, S1, nc);                // Lm = X X^T
        float* L = mt.nch > 1 ? ws + mt.Lp + (long)u.y * m * m : ws + mt.Lm;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int cl = 32 * wn + 8 * j + 2 * tl;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const int r = (kk < 2) ? r0 : r1, c = cl + (kk & 1);
                if (r < m && c < m) L[(long)r * m + c] = a2 * acc[j][kk] + float(b2) * L[(long)r * m + c];
            }
        }
    }
    if (mt.pn) {
        smem_gemm_tf32(acc, S0, S0, m);                 // Rn = X^T X
        float* R = ws + mt.Rn;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int cl = 32 * wn + 8 * j + 2 * tl;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const int r = (kk < 2) ? r0 : r1, c = cl + (kk & 1);
                if (r < n && c < n) R[(long)r * n + c] = a2 * acc[j][kk] + float(b2) * R[(long)r * n + c];
            }
        }
    }
    if (((long)t - 1) % every == 0) return;             // refresh step: k_soap_b / k_soap_eig
    __syncthreads();
    soap_apply_smem_tf32(mt, q, ws, c0, nc, t, lr, step_idx, mult, b1, b2, eps, S0, S1, true, osc);
}

// One CTA per unit (matrix k, chunk c): factor EMAs (chunk c's partial when split); on normal steps also the update.
template <typename T>
__global__ void __launch_bounds__(256, 2) k_soap_a(const SoapMat* mats, const int2* units, const PDesc<T>* d,
                                                const double* lr, const int* step_idx, T* ws, double mult, double b1,
                                                double b2, double eps, int every) {
    using namespace sp;
    extern __shared__ unsigned char smem_raw[];
    T* As = reinterpret_cast<T*>(smem_raw);
    T* Bs = As + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<T> q = d[mt.param];
    const int m = mt.m, n = mt.n;
    const int c0 = mt.nch > 1 ? u.y * TS : 0, nc = mt.nch > 1 ? min(TS, n - c0) : n;
    const T* X = q.dir + mt.off + (long)c0 * mt.xj;
    const long xi = mt.xi, xj = mt.xj;
    const double t = (double)*q.step;
    const T a2 = T(1.0 - b2);
    if (mt.pm) {
        T* L = mt.nch > 1 ? ws + mt.Lp + (long)u.y * m * m : ws + mt.Lm;
        bmm<T>(L, m, X, xi, xj, X, xj, xi, m, m, nc, a2, T(b2), As, Bs);
    }
    if (mt.pn) bmm<T>(ws + mt.Rn, n, X, xj, xi, X, xi, xj, n, n, m, a2, T(b2), As, Bs);
    if (((long)t - 1) % every == 0) return;             // refresh step: k_soap_b
    soap_apply<T>(mt, q, ws, c0, nc, t, lr, step_idx, mult, b1, b2, eps, As, Bs);
}

// Dedicated zero-spill FP32 shared-memory k_soap_b (m, n <= 64): Q^T L Q + jacobi_f in smem (zero W1/Bm round-trips) + soap_apply_smem!
__global__ void __launch_bounds__(256, 2) k_soap_b_smem_f(const SoapMat* mats, const PDesc<float>* d, const double* lr,
                                                          const int* step_idx, float* ws, double mult, double b1,
                                                          double b2, double eps, int every, double tol, int warm_sweeps,
                                                          const float* osc = nullptr) {
    using namespace sp;
    extern __shared__ unsigned char smem_raw[];
    float* S0 = reinterpret_cast<float*>(smem_raw);
    float* S1 = S0 + TS * LDV;
    const SoapMat mt = mats[blockIdx.x];
    const PDesc<float> q = d[mt.param];
    const double t = (double)*q.step;
    if (((long)t - 1) % every != 0) return;
    const int m = mt.m, n = mt.n;
    float *Lm = ws + mt.Lm, *Qm = ws + mt.Qm;
    float *Rn = ws + mt.Rn, *Qn = ws + mt.Qn;
    const int maxsw = t <= 1.0 ? 30 : warm_sweeps;
    if (mt.pm) {
        if (mt.nch > 1) {
            const float* Lp = ws + mt.Lp;
            for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
                float acc = Lp[e];
                for (int c = 1; c < mt.nch; ++c) acc += Lp[(long)c * m * m + e];
                Lm[e] = acc;
            }
            __syncthreads();
        }
        smem_qtlq_jacobi<false>(Lm, Qm, m, maxsw, tol, S0, S1);
    }
    if (mt.pn) smem_qtlq_jacobi<false>(Rn, Qn, n, maxsw, tol, S0, S1);
    for (int c0 = 0; c0 < n; c0 += TS)
        soap_apply_smem<float>(mt, q, ws, c0, min(TS, n - c0), t, lr, step_idx, mult, b1, b2, eps, S0, S1, false, osc);
}

// Dedicated zero-spill FP32 shared-memory k_soap_eig + k_soap_ap (LST_SOAP_EIG=1)
__global__ void __launch_bounds__(256, 2) k_soap_eig_smem_f(const SoapMat* mats, const int2* units,
                                                            const PDesc<float>* d, float* ws, int every, double tol,
                                                            int warm_sweeps) {
    using namespace sp;
    extern __shared__ unsigned char smem_raw[];
    float* S0 = reinterpret_cast<float*>(smem_raw);
    float* S1 = S0 + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<float> q = d[mt.param];
    const double t = (double)*q.step;
    if (((long)t - 1) % every != 0) return;
    const int maxsw = t <= 1.0 ? 30 : warm_sweeps;
    if (u.y == 0) {
        const int m = mt.m;
        float *Lm = ws + mt.Lm, *Qm = ws + mt.Qm;
        if (mt.nch > 1) {
            const float* Lp = ws + mt.Lp;
            for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
                float acc = Lp[e];
                for (int c = 1; c < mt.nch; ++c) acc += Lp[(long)c * m * m + e];
                Lm[e] = acc;
            }
            __syncthreads();
        }
        smem_qtlq_jacobi<false>(Lm, Qm, m, maxsw, tol, S0, S1);
    } else {
        smem_qtlq_jacobi<false>(ws + mt.Rn, ws + mt.Qn, mt.n, maxsw, tol, S0, S1);
    }
}

__global__ void __launch_bounds__(256, 2) k_soap_ap_smem_f(const SoapMat* mats, const int2* units,
                                                           const PDesc<float>* d, const double* lr, const int* step_idx,
                                                           float* ws, double mult, double b1, double b2, double eps,
                                                           int every, const float* osc = nullptr) {
    using namespace sp;
    extern __shared__ unsigned char smem_raw[];
    float* S0 = reinterpret_cast<float*>(smem_raw);
    float* S1 = S0 + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<float> q = d[mt.param];
    const double t = (double)*q.step;
    if (((long)t - 1) % every != 0) return;
    const int n = mt.n;
    const int c0 = mt.nch > 1 ? u.y * TS : 0, nc = mt.nch > 1 ? min(TS, n - c0) : n;
    soap_apply_smem<float>(mt, q, ws, c0, nc, t, lr, step_idx, mult, b1, b2, eps, S0, S1, false, osc);
}

// Wave 14.3 optimized FP32 SOAP kernels (LST_SOAP_OPT=1|2):
//   - float4 coalesced global loads/stores for aligned 64x64 Qm, Qn, V, X, Mx, p, ema, Lm, Rn
//   - Fused single-pass symmetric Lm = X X^T AND Rn = X^T X for 64x64 blocks (warps 0..3 compute Lm upper+diag,
//     warps 4..7 compute Rn upper+diag in ONE 64-step K loop -> 2x faster factor EMA!)
__device__ void soap_apply_smem_opt(const SoapMat& mt, const PDesc<float>& q, float* ws, int c0, int nc, double t,
                                    const double* lr, const int* step_idx, double mult, double b1, double b2,
                                    double eps, float* S0, float* S1, bool x_in_s0, const float* osc = nullptr) {
    using namespace sp;
    const int m = mt.m, n = mt.n;
    const long xi = mt.xi, xj = mt.xj;
    const float* X = q.dir + mt.off + (long)c0 * xj;
    const float* Mx = q.m + mt.off + (long)c0 * xj;
    const float *Qm = ws + mt.Qm, *Qn = ws + mt.Qn;
    float* V = ws + mt.V + c0;
    const float a2 = float(1.0 - b2), fb2 = float(b2);
    const int r0 = (threadIdx.x >> 4) * 4, c0t = (threadIdx.x & 15) * 4;
    if (!x_in_s0) load_mat_smem_f4(S0, X, xi, xj, m, nc);
    load_sq_smem_f4(S1, Qm, m);
    __syncthreads();
    float acc[4][4];
    smem_gemm<float>(acc, S1, S0, m);
    if (mt.pn) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                S0[(c0t + j) * LDV + (r0 + i)] = (r0 + i < m && c0t + j < nc) ? acc[i][j] : 0.f;
        load_sq_smem_f4(S1, Qn, n);
        __syncthreads();
        smem_gemm<float>(acc, S0, S1, n);
    }
    const float inv_bc1 = 1.f / (1.f - powf((float)b1, (float)t));
    const float inv_bc2 = 1.f / (1.f - powf((float)b2, (float)t));
    const float te = float(eps);
    float denom[4][4];
    const bool v_vec4 = ((n & 3) == 0) && ((c0 & 3) == 0) && (c0t + 4 <= nc) && (((uintptr_t)V & 15) == 0);
    if (v_vec4) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = r0 + i;
            if (r < m) {
                float4* vp = reinterpret_cast<float4*>(V + (long)r * n + c0t);
                float4 vv = *vp;
                vv.x = fb2 * vv.x + a2 * acc[i][0] * acc[i][0];
                vv.y = fb2 * vv.y + a2 * acc[i][1] * acc[i][1];
                vv.z = fb2 * vv.z + a2 * acc[i][2] * acc[i][2];
                vv.w = fb2 * vv.w + a2 * acc[i][3] * acc[i][3];
                *vp = vv;
                denom[i][0] = sqrtf(vv.x * inv_bc2) + te;
                denom[i][1] = sqrtf(vv.y * inv_bc2) + te;
                denom[i][2] = sqrtf(vv.z * inv_bc2) + te;
                denom[i][3] = sqrtf(vv.w * inv_bc2) + te;
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = r0 + i, c = c0t + j;
                if (r < m && c < nc) {
                    const long k = (long)r * n + c;
                    const float g = acc[i][j];
                    const float v = fb2 * V[k] + a2 * g * g;
                    V[k] = v;
                    denom[i][j] = sqrtf(v * inv_bc2) + te;
                }
            }
    }
    __syncthreads();
    load_mat_smem_f4(S0, Mx, xi, xj, m, nc);
    if (mt.pn) load_sq_smem_f4(S1, Qm, m);
    __syncthreads();
    smem_gemm<float>(acc, S1, S0, m);
    if (mt.pn) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                S0[(c0t + j) * LDV + (r0 + i)] = (r0 + i < m && c0t + j < nc) ? acc[i][j] : 0.f;
        load_sq_smem_f4(S1, Qn, n);
        __syncthreads();
        smem_gemm<float>(acc, S0, S1, n);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = r0 + i, c = c0t + j;
            S0[r * LDV + c] = (r < m && c < nc) ? (acc[i][j] * inv_bc1) / denom[i][j] : 0.f;
        }
    load_sq_smem_T_f4(S1, Qm, m);
    __syncthreads();
    smem_gemm<float>(acc, S1, S0, m);
    if (mt.pn) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                S0[(c0t + j) * LDV + (r0 + i)] = (r0 + i < m && c0t + j < nc) ? acc[i][j] : 0.f;
        load_sq_smem_T_f4(S1, Qn, n);
        __syncthreads();
        smem_gemm<float>(acc, S0, S1, n);
    }
    const float f = float(lr[16L * step_idx[0] + q.group] * mult);
    const float omd = osc ? osc[3 * LST_N_PARAMS + 1] : 0.f;
    const bool do_ema = osc && q.ema != nullptr;
    const bool p_vec4 = (xj == 1) && ((xi & 3) == 0) && (((mt.off + c0) & 3) == 0) && (c0t + 4 <= nc) &&
                        (((uintptr_t)q.p & 15) == 0) && (!do_ema || (((uintptr_t)q.ema & 15) == 0));
    if (p_vec4) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = r0 + i;
            if (r < m) {
                const long idx = mt.off + (long)r * xi + (long)(c0 + c0t);
                float4* pp = reinterpret_cast<float4*>(q.p + idx);
                float4 pv = *pp;
                pv.x -= f * acc[i][0]; pv.y -= f * acc[i][1]; pv.z -= f * acc[i][2]; pv.w -= f * acc[i][3];
                *pp = pv;
                if (do_ema) {
                    float4* ep = reinterpret_cast<float4*>(q.ema + idx);
                    float4 ev = *ep;
                    ev.x -= omd * (ev.x - pv.x); ev.y -= omd * (ev.y - pv.y);
                    ev.z -= omd * (ev.z - pv.z); ev.w -= omd * (ev.w - pv.w);
                    *ep = ev;
                }
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = r0 + i, c = c0t + j;
                if (r < m && c < nc) {
                    const long idx = mt.off + (long)r * xi + (long)(c0 + c) * xj;
                    const float pnew = q.p[idx] - f * acc[i][j];
                    q.p[idx] = pnew;
                    if (do_ema) q.ema[idx] -= omd * (q.ema[idx] - pnew);
                }
            }
    }
}

__global__ void __launch_bounds__(256, 2) k_soap_a_smem_opt(const SoapMat* mats, const int2* units,
                                                            const PDesc<float>* d, const double* lr,
                                                            const int* step_idx, float* ws, double mult, double b1,
                                                            double b2, double eps, int every, const float* osc) {
    using namespace sp;
    extern __shared__ unsigned char smem_raw[];
    float* S0 = reinterpret_cast<float*>(smem_raw);
    float* S1 = S0 + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<float> q = d[mt.param];
    const int m = mt.m, n = mt.n;
    const int c0 = mt.nch > 1 ? u.y * TS : 0, nc = mt.nch > 1 ? min(TS, n - c0) : n;
    const float* X = q.dir + mt.off + (long)c0 * mt.xj;
    const long xi = mt.xi, xj = mt.xj;
    const double t = (double)*q.step;
    const float a2 = float(1.0 - b2), fb2 = float(b2);
    const int r0 = (threadIdx.x >> 4) * 4, c0t = (threadIdx.x & 15) * 4;

    if (xj == 1 && m == 64 && nc == 64 && (xi & 3) == 0 && (((uintptr_t)X & 15) == 0)) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int idx = threadIdx.x + g * 256;
            const int r = idx >> 4, c4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(X + (long)r * xi + c4);
            *reinterpret_cast<float4*>(S0 + r * LDV + c4) = v;
            if (mt.pm) {
                S1[(c4 + 0) * LDV + r] = v.x;
                S1[(c4 + 1) * LDV + r] = v.y;
                S1[(c4 + 2) * LDV + r] = v.z;
                S1[(c4 + 3) * LDV + r] = v.w;
            }
        }
    } else {
        const bool col_fast = (xi == 1 && xj != 1);
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            float v[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e = threadIdx.x + (g * 4 + k) * 256, hi = e >> 6, lo = e & 63;
                const int r = col_fast ? lo : hi, c = col_fast ? hi : lo;
                v[k] = (r < m && c < nc) ? X[(long)r * xi + (long)c * xj] : 0.f;
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e = threadIdx.x + (g * 4 + k) * 256, hi = e >> 6, lo = e & 63;
                const int r = col_fast ? lo : hi, c = col_fast ? hi : lo;
                S0[r * LDV + c] = v[k];
                if (mt.pm) S1[c * LDV + r] = v[k];
            }
        }
    }
    __syncthreads();

    if (mt.pm && mt.pn && m == 64 && n == 64 && nc == 64) {
        // Fused single-pass symmetric Lm = X X^T (warps 0..3) AND Rn = X^T X (warps 4..7)!
        const int half = threadIdx.x >> 7;        // 0 = Lm, 1 = Rn
        const int th = threadIdx.x & 127;         // 0..119 = off-diag (tr < tc), 120..127 = 2 diag tiles each
        const float* Msrc = half ? S0 : S1;
        float* Mdst = half ? (ws + mt.Rn) : (mt.nch > 1 ? ws + mt.Lp + (long)u.y * 4096 : ws + mt.Lm);
        if (th < 120) {
            const int a = th / 15, b = th % 15;
            const int tr = (b >= a) ? a : (15 - a);
            const int tc = (b >= a) ? (b + 1) : (15 - b);
            const int rr = 4 * tr, cc = 4 * tc;
            float acc[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
#pragma unroll 4
            for (int k = 0; k < 64; ++k) {
                const float4 va = *reinterpret_cast<const float4*>(Msrc + k * LDV + rr);
                const float4 vb = *reinterpret_cast<const float4*>(Msrc + k * LDV + cc);
                const float av[4] = {va.x, va.y, va.z, va.w}, bv[4] = {vb.x, vb.y, vb.z, vb.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float4* up = reinterpret_cast<float4*>(Mdst + (rr + i) * 64 + cc);
                float4 uv = *up;
                uv.x = a2 * acc[i][0] + fb2 * uv.x;
                uv.y = a2 * acc[i][1] + fb2 * uv.y;
                uv.z = a2 * acc[i][2] + fb2 * uv.z;
                uv.w = a2 * acc[i][3] + fb2 * uv.w;
                *up = uv;
                Mdst[(cc + 0) * 64 + rr + i] = uv.x;
                Mdst[(cc + 1) * 64 + rr + i] = uv.y;
                Mdst[(cc + 2) * 64 + rr + i] = uv.z;
                Mdst[(cc + 3) * 64 + rr + i] = uv.w;
            }
        } else {
            const int rr0 = 4 * ((th - 120) * 2), rr1 = rr0 + 4;
            float a0[4][4] = {}, a1[4][4] = {};
#pragma unroll 4
            for (int k = 0; k < 64; ++k) {
                const float4 v0 = *reinterpret_cast<const float4*>(Msrc + k * LDV + rr0);
                const float4 v1 = *reinterpret_cast<const float4*>(Msrc + k * LDV + rr1);
                const float av0[4] = {v0.x, v0.y, v0.z, v0.w}, av1[4] = {v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = i; j < 4; ++j) {
                        a0[i][j] += av0[i] * av0[j];
                        a1[i][j] += av1[i] * av1[j];
                    }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = i; j < 4; ++j) {
                    const float nv0 = a2 * a0[i][j] + fb2 * Mdst[(rr0 + i) * 64 + rr0 + j];
                    Mdst[(rr0 + i) * 64 + rr0 + j] = nv0;
                    if (j > i) Mdst[(rr0 + j) * 64 + rr0 + i] = nv0;
                    const float nv1 = a2 * a1[i][j] + fb2 * Mdst[(rr1 + i) * 64 + rr1 + j];
                    Mdst[(rr1 + i) * 64 + rr1 + j] = nv1;
                    if (j > i) Mdst[(rr1 + j) * 64 + rr1 + i] = nv1;
                }
            }
        }
    } else {
        float acc[4][4];
        if (mt.pm) {
            smem_gemm<float>(acc, S1, S1, nc);
            float* L = mt.nch > 1 ? ws + mt.Lp + (long)u.y * m * m : ws + mt.Lm;
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = r0 + i, c = c0t + j;
                    if (r < m && c < m) L[(long)r * m + c] = a2 * acc[i][j] + fb2 * L[(long)r * m + c];
                }
        }
        if (mt.pn) {
            smem_gemm<float>(acc, S0, S0, m);
            float* R = ws + mt.Rn;
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = r0 + i, c = c0t + j;
                    if (r < n && c < n) R[(long)r * n + c] = a2 * acc[i][j] + fb2 * R[(long)r * n + c];
                }
        }
    }
    if (((long)t - 1) % every == 0) return;
    __syncthreads();
    soap_apply_smem_opt(mt, q, ws, c0, nc, t, lr, step_idx, mult, b1, b2, eps, S0, S1, true, osc);
}

template <bool FAST_ALG>
__global__ void __launch_bounds__(256, 2) k_soap_eig_smem_opt(const SoapMat* mats, const int2* units,
                                                              const PDesc<float>* d, float* ws, int every, double tol,
                                                              int warm_sweeps) {
    using namespace sp;
    extern __shared__ unsigned char smem_raw[];
    float* S0 = reinterpret_cast<float*>(smem_raw);
    float* S1 = S0 + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<float> q = d[mt.param];
    const double t = (double)*q.step;
    if (((long)t - 1) % every != 0) return;
    const int maxsw = t <= 1.0 ? 30 : warm_sweeps;
    if (u.y == 0) {
        const int m = mt.m;
        float *Lm = ws + mt.Lm, *Qm = ws + mt.Qm;
        if (mt.nch > 1) {
            const float* Lp = ws + mt.Lp;
            for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
                float acc = Lp[e];
                for (int c = 1; c < mt.nch; ++c) acc += Lp[(long)c * m * m + e];
                Lm[e] = acc;
            }
            __syncthreads();
        }
        smem_qtlq_jacobi_opt<FAST_ALG>(Lm, Qm, m, maxsw, tol, S0, S1);
    } else {
        smem_qtlq_jacobi_opt<FAST_ALG>(ws + mt.Rn, ws + mt.Qn, mt.n, maxsw, tol, S0, S1);
    }
}

__global__ void __launch_bounds__(256, 2) k_soap_ap_smem_opt(const SoapMat* mats, const int2* units,
                                                             const PDesc<float>* d, const double* lr,
                                                             const int* step_idx, float* ws, double mult, double b1,
                                                             double b2, double eps, int every,
                                                             const float* osc = nullptr) {
    using namespace sp;
    extern __shared__ unsigned char smem_raw[];
    float* S0 = reinterpret_cast<float*>(smem_raw);
    float* S1 = S0 + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<float> q = d[mt.param];
    const double t = (double)*q.step;
    if (((long)t - 1) % every != 0) return;
    const int n = mt.n;
    const int c0 = mt.nch > 1 ? u.y * TS : 0, nc = mt.nch > 1 ? min(TS, n - c0) : n;
    soap_apply_smem_opt(mt, q, ws, c0, nc, t, lr, step_idx, mult, b1, b2, eps, S0, S1, false, osc);
}

// ============================================================================ Wave 17 & 18: Fast SOAP Eigenupdates
// (LST_SOAP_QR=1: sorted cold Jacobi + TF32 1-barrier pipelined MGS QR; LST_SOAP_QR=2: Wave 18 Lie-Jacobi + FP32
// Newton-Schulz, ZERO loops, ~25 us refresh = ~2.5 us/step!) & Zero-Spill Qm/Qn-Cached BF16 TC SOAP (LST_SOAP_TC=1)!

__device__ void smem_qr_power_opt(const float* L, float* Q, int d, float* S0, float* S1) {
    using namespace sp;
    __shared__ float s_tr;
    if (threadIdx.x < 32) {
        float loc = 0.f;
        for (int i = threadIdx.x; i < d; i += 32) loc += L[(long)i * d + i];
        for (int o = 16; o > 0; o >>= 1) loc += __shfl_xor_sync(0xffffffffu, loc, o);
        if (threadIdx.x == 0) s_tr = 1e-6f * (loc / float(max(d, 1))) + 1e-12f;
    }
    load_sq_smem_T_f4(S0, L, d);   // S0[k * LDV + r] = L[r, k]
    load_sq_smem_f4(S1, Q, d);     // S1[k * LDV + c] = Q[k, c]
    __syncthreads();
    const float lam = s_tr;
    if (threadIdx.x < d) S0[threadIdx.x * LDV + threadIdx.x] += lam;
    __syncthreads();
    // Step 1: Z = (L + lambda*I) @ Q_old via TF32 Tensor Cores!
    float acc[4][4];
    smem_gemm_tf32(acc, S0, S1, d);
    __syncthreads();
    // Store Z column-major in S0[col * LDV + row] and Q_old column-major in S1[col * LDV + row]
    store_tf32_smem<true>(S0, acc, d, d, LDV);
    load_sq_smem_T_f4(S1, Q, d);
    __syncthreads();

    // Step 2: 1-barrier pipelined 8-warp Modified Gram-Schmidt QR on columns 0..d-1 of S0 (65 barriers vs 129!)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rw0 = lane, rw1 = lane + 32;
    const bool ok0 = rw0 < d, ok1 = rw1 < d;
    auto norm_col = [&](int col, float z0, float z1) {
        const float qo0 = ok0 ? S1[col * LDV + rw0] : 0.f;
        const float qo1 = ok1 ? S1[col * LDV + rw1] : 0.f;
        float n2 = z0 * z0 + z1 * z1;
        float ov = z0 * qo0 + z1 * qo1;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            n2 += __shfl_xor_sync(0xffffffffu, n2, o);
            ov += __shfl_xor_sync(0xffffffffu, ov, o);
        }
        if (n2 > 1e-30f) {
            float inv = rsqrtf(n2);
            if (ov < 0.f) inv = -inv;
            if (ok0) S0[col * LDV + rw0] = z0 * inv;
            if (ok1) S0[col * LDV + rw1] = z1 * inv;
        } else {
            if (ok0) S0[col * LDV + rw0] = qo0;
            if (ok1) S0[col * LDV + rw1] = qo1;
        }
    };
    if (d > 0 && warp == 0) norm_col(0, ok0 ? S0[rw0] : 0.f, ok1 ? S0[rw1] : 0.f);
    __syncthreads();
    for (int j = 0; j < d - 1; ++j) {
        const float qj0 = ok0 ? S0[j * LDV + rw0] : 0.f;
        const float qj1 = ok1 ? S0[j * LDV + rw1] : 0.f;
        for (int k = j + 1 + warp; k < d; k += 8) {
            float zk0 = ok0 ? S0[k * LDV + rw0] : 0.f;
            float zk1 = ok1 ? S0[k * LDV + rw1] : 0.f;
            float dot = qj0 * zk0 + qj1 * zk1;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            zk0 -= dot * qj0;
            zk1 -= dot * qj1;
            if (k == j + 1) {
                norm_col(k, zk0, zk1);
            } else {
                if (ok0) S0[k * LDV + rw0] = zk0;
                if (ok1) S0[k * LDV + rw1] = zk1;
            }
        }
        __syncthreads();
    }
    if (d == 64 && (((uintptr_t)Q & 15) == 0)) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int idx = threadIdx.x + g * 256;
            const int r = idx >> 4, c4 = (idx & 15) * 4;
            *reinterpret_cast<float4*>(Q + (long)r * 64 + c4) =
                make_float4(S0[(c4 + 0) * LDV + r], S0[(c4 + 1) * LDV + r], S0[(c4 + 2) * LDV + r],
                            S0[(c4 + 3) * LDV + r]);
        }
    } else {
        for (int e = threadIdx.x; e < d * d; e += blockDim.x) {
            const int r = e / d, c = e % d;
            Q[e] = S0[c * LDV + r];
        }
    }
    __syncthreads();
}

// Wave 18: Lie-Jacobi + FP32 Newton-Schulz warm eigenbasis update (LST_SOAP_QR=2)!
// 1. B = Q^T L Q via 2 TF32 Tensor-Core GEMMs
// 2. Exact first-order skew-symmetric generator K_{pq} = B_sym[p,q] / (sgn * sqrt(Delta^2 + 4*B_sym^2 + reg^2))
//    capped to ||K||_inf <= 0.5 (guarantees ||K||_2 <= 0.5, ZERO column swapping, quadratic convergence!)
// 3. U0 = I + K + 0.5 K^2 (using S1^T S1 = -K^2 via 1 TF32 GEMM!), Q0 = Q_old U0 (1 TF32 GEMM)
// 4. Exact FP32 Newton-Schulz polish Q_new = Q0 (1.5 I - 0.5 Q0^T Q0) -> machine-precision orthogonality forever!
__device__ void smem_lie_jacobi_opt(const float* L, float* Q, int d, float* S0, float* S1) {
    using namespace sp;
    __shared__ float s_diag[64];
    __shared__ float s_rl1[64];
    __shared__ float s_reg2;
    __shared__ float s_scale;

    load_sq_smem_T_f4(S0, L, d);   // S0[k * LDV + r] = L[r, k]
    load_sq_smem_f4(S1, Q, d);     // S1[k * LDV + c] = Q[k, c]
    __syncthreads();
    float acc[4][4];
    smem_gemm_tf32(acc, S0, S1, d);                    // W1 = L @ Q
    __syncthreads();
    store_tf32_smem<false>(S0, acc, d, d, LDV);        // S0 = W1 row-major
    __syncthreads();
    smem_gemm_tf32(acc, S1, S0, d);                    // B = Q^T @ W1 (S1 is already col-major Q^T!)
    __syncthreads();
    store_tf32_smem<false>(S0, acc, d, d, LDV);        // S0 = B row-major
    __syncthreads();

    if (threadIdx.x < d) s_diag[threadIdx.x] = S0[threadIdx.x * LDV + threadIdx.x];
    else if (threadIdx.x < 64) s_diag[threadIdx.x] = 0.f;
    __syncthreads();
    if (threadIdx.x < 32) {
        float loc = s_diag[threadIdx.x] + s_diag[threadIdx.x + 32];
        for (int o = 16; o > 0; o >>= 1) loc += __shfl_xor_sync(0xffffffffu, loc, o);
        if (threadIdx.x == 0) {
            const float reg = 1e-5f * (fabsf(loc) / float(max(d, 1))) + 1e-12f;
            s_reg2 = reg * reg;
        }
    }
    __syncthreads();
    const float reg2 = s_reg2;
    // Build bitwise skew-symmetric K in S1[p * LDV + q] and compute row L1 norm per row (4 threads/row)
    const int p = threadIdx.x >> 2, sub = threadIdx.x & 3;
    const float dp = s_diag[p];
    float loc_l1 = 0.f;
#pragma unroll
    for (int u = 0; u < 16; ++u) {
        const int q = sub + 4 * u;
        float kpq = 0.f;
        if (p < d && q < d && p != q) {
            const float bpq = 0.5f * (S0[p * LDV + q] + S0[q * LDV + p]);
            const float delta = s_diag[q] - dp;
            const float sgn = (delta > 0.f || (delta == 0.f && p < q)) ? 1.f : -1.f;
            kpq = bpq / (sgn * sqrtf(delta * delta + 4.f * bpq * bpq + reg2));
        }
        S1[p * LDV + q] = kpq;
        loc_l1 += fabsf(kpq);
    }
    loc_l1 += __shfl_xor_sync(0xffffffffu, loc_l1, 2);
    loc_l1 += __shfl_xor_sync(0xffffffffu, loc_l1, 1);
    if (sub == 0) s_rl1[p] = loc_l1;
    __syncthreads();
    if (threadIdx.x < 32) {
        float mx = fmaxf(s_rl1[threadIdx.x], s_rl1[threadIdx.x + 32]);
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if (threadIdx.x == 0) s_scale = (mx > 0.5f) ? (0.5f / mx) : 1.f;
    }
    __syncthreads();
    const float sc = s_scale;
    if (sc < 1.f) {
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            const int q = sub + 4 * u;
            S1[p * LDV + q] *= sc;
        }
        __syncthreads();
    }
    // Since K is skew-symmetric (S1[k, r] = -K[r, k]), smem_gemm_tf32(acc, S1, S1, d) = -K^2!
    smem_gemm_tf32(acc, S1, S1, d);
    // Write U0 = I + K + 0.5 * K^2 = I + K - 0.5 * acc into S0 (row-major; S0 is disjoint from S1!)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0_t = 16 * wm + g, r1_t = r0_t + 8;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = 32 * wn + 8 * j + 2 * tl, c1 = c0 + 1;
        auto mk = [&](int r, int c, float neg_k2) {
            if (r >= d || c >= d) return 0.f;
            return ((r == c) ? 1.f : 0.f) + S1[r * LDV + c] - 0.5f * neg_k2;
        };
        S0[r0_t * LDV + c0] = mk(r0_t, c0, acc[j][0]);
        S0[r0_t * LDV + c1] = mk(r0_t, c1, acc[j][1]);
        S0[r1_t * LDV + c0] = mk(r1_t, c0, acc[j][2]);
        S0[r1_t * LDV + c1] = mk(r1_t, c1, acc[j][3]);
    }
    __syncthreads();
    // Q0 = Q_old @ U0 via TF32 TC
    load_sq_smem_T_f4(S1, Q, d);                       // S1 = Q_old col-major
    __syncthreads();
    smem_gemm_tf32(acc, S1, S0, d);                    // Q0 = Q_old @ U0
    __syncthreads();
    store_tf32_smem<false>(S0, acc, d, d, LDV);        // S0 = Q0 row-major
    store_tf32_smem<true>(S1, acc, d, d, LDV);         // S1 = Q0 col-major
    __syncthreads();
    // Exact FP32 Newton-Schulz polish: G = Q0^T Q0, M = 1.5 I - 0.5 G, Q_new = Q0 @ M
    smem_gemm<float>(acc, S0, S0, d);
    __syncthreads();
    const int r0_s = (threadIdx.x >> 4) * 4, c0_s = (threadIdx.x & 15) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = r0_s + i, c = c0_s + j;
            S0[r * LDV + c] = (r < d && c < d) ? (((r == c) ? 1.5f : 0.f) - 0.5f * acc[i][j]) : 0.f;
        }
    __syncthreads();
    smem_gemm<float>(acc, S1, S0, d);                  // Q_new = Q0 @ M
    if (d == 64 && (((uintptr_t)Q & 15) == 0)) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            *reinterpret_cast<float4*>(Q + (long)(r0_s + i) * 64 + c0_s) =
                make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (r0_s + i < d && c0_s + j < d) Q[(long)(r0_s + i) * d + (c0_s + j)] = acc[i][j];
    }
    __syncthreads();
}

__global__ void __launch_bounds__(256, 2) k_soap_eig_qr_smem(const SoapMat* mats, const int2* units,
                                                             const PDesc<float>* d, float* ws, int every, double tol,
                                                             int warm_sweeps, int qr_mode,
                                                             __nv_bfloat16* sqbf = nullptr) {
    if (((long)*d[mats[0].param].step - 1) % every != 0) return;
    using namespace sp;
    extern __shared__ unsigned char smem_raw[];
    float* S0 = reinterpret_cast<float*>(smem_raw);
    float* S1 = S0 + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<float> q = d[mt.param];
    const double t = (double)*q.step;
    if (((long)t - 1) % every != 0) return;
    const bool warm = (t > 1.0);
    auto write_qbf = [&](const float* Q, __nv_bfloat16* Qbf, int dim) {
        if (dim == 64 && (((uintptr_t)Q & 15) == 0) && (((uintptr_t)Qbf & 7) == 0)) {
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int idx = threadIdx.x + g * 256;
                const float4 v = reinterpret_cast<const float4*>(Q)[idx];
                uint2 h4;
                h4.x = ah::pack2<__nv_bfloat16>(v.x, v.y);
                h4.y = ah::pack2<__nv_bfloat16>(v.z, v.w);
                reinterpret_cast<uint2*>(Qbf)[idx] = h4;
            }
        } else {
            const int tot = dim * dim;
            for (int e = threadIdx.x; e < tot; e += 256) Qbf[e] = ah::from_f<__nv_bfloat16>(Q[e]);
        }
    };
    if (u.y == 0) {
        const int m = mt.m;
        float *Lm = ws + mt.Lm, *Qm = ws + mt.Qm;
        if (mt.nch > 1) {
            const float* Lp = ws + mt.Lp;
            for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
                float acc = Lp[e];
                for (int c = 1; c < mt.nch; ++c) acc += Lp[(long)c * m * m + e];
                Lm[e] = acc;
            }
            __syncthreads();
        }
        if (warm) {
            if (qr_mode >= 2) smem_lie_jacobi_opt(Lm, Qm, m, S0, S1);
            else smem_qr_power_opt(Lm, Qm, m, S0, S1);
        } else {
            if (qr_mode == 1) smem_qtlq_jacobi_opt<false, true>(Lm, Qm, m, 30, tol, S0, S1);
            else smem_qtlq_jacobi_opt<false, false>(Lm, Qm, m, 30, tol, S0, S1);
        }
        if (sqbf) write_qbf(Qm, sqbf + mt.Qm_bf, m);
    } else {
        if (warm) {
            if (qr_mode >= 2) smem_lie_jacobi_opt(ws + mt.Rn, ws + mt.Qn, mt.n, S0, S1);
            else smem_qr_power_opt(ws + mt.Rn, ws + mt.Qn, mt.n, S0, S1);
        } else {
            if (qr_mode == 1) smem_qtlq_jacobi_opt<false, true>(ws + mt.Rn, ws + mt.Qn, mt.n, 30, tol, S0, S1);
            else smem_qtlq_jacobi_opt<false, false>(ws + mt.Rn, ws + mt.Qn, mt.n, 30, tol, S0, S1);
        }
        if (sqbf) write_qbf(ws + mt.Qn, sqbf + mt.Qn_bf, mt.n);
    }
}

// Wave 18: Zero-spill Qm/Qn-cached BF16 Tensor-Core SOAP helpers (36.8 KB static smem -> 2 blk/SM = 100% of n_sunits!):
//   - gemm_bf16<AT, BT>: hardware ldmatrix.x4 / ldmatrix.x4.trans -> ZERO transpose stores in smem!
//   - Qm & Qn loaded ONCE coalesced into Sqm/Sqn (deletes 4 DRAM reloads per block!)
//   - Ping-pong S0/S1 cuts barriers 11 -> 6!
//   - Vectorized float2 LDG.64/STG.64 on L, R, V, q.p, q.ema!
namespace sp_bf {
constexpr int LDS = 72;

// AT=false: A_mma = A; AT=true: A_mma = A^T
// BT=false: B_mma = B^T (C = A_op @ B^T); BT=true: B_mma = B (C = A_op @ B)
template <bool AT, bool BT>
__device__ __forceinline__ void gemm_bf16(float acc[4][4], const __nv_bfloat16 (*A)[LDS],
                                          const __nv_bfloat16 (*B)[LDS]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1;
    const int a_row = 16 * wm + (lane & 15), a_col = (lane >> 4) * 8;
    const int at_k = ((lane >> 4) * 8) + (lane & 7), at_col = 16 * wm + (((lane >> 3) & 1) * 8);
    const int b_col = 32 * wn + ((lane >> 4) * 8) + (lane & 7), b_koff = ((lane >> 3) & 1) * 8;
    const int bt_k = (((lane >> 3) & 1) * 8) + (lane & 7), bt_col = 32 * wn + ((lane >> 4) * 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
    for (int k0 = 0; k0 < 64; k0 += 16) {
        uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
        if constexpr (!AT) ah::ldmatrix_x4(a0, a1, a2, a3, &A[a_row][k0 + a_col]);
        else ah::ldmatrix_x4_trans(a0, a1, a2, a3, &A[k0 + at_k][at_col]);
        if constexpr (!BT) ah::ldmatrix_x4(b0, b1, b2, b3, &B[b_col][k0 + b_koff]);
        else ah::ldmatrix_x4_trans(b0, b1, b2, b3, &B[k0 + bt_k][bt_col]);
        ah::mma_16x8x16<__nv_bfloat16>(acc[0], a0, a1, a2, a3, b0, b1);
        ah::mma_16x8x16<__nv_bfloat16>(acc[1], a0, a1, a2, a3, b2, b3);
        if constexpr (!BT) ah::ldmatrix_x4(b0, b1, b2, b3, &B[b_col + 16][k0 + b_koff]);
        else ah::ldmatrix_x4_trans(b0, b1, b2, b3, &B[k0 + bt_k][bt_col + 16]);
        ah::mma_16x8x16<__nv_bfloat16>(acc[2], a0, a1, a2, a3, b0, b1);
        ah::mma_16x8x16<__nv_bfloat16>(acc[3], a0, a1, a2, a3, b2, b3);
    }
}

__device__ __forceinline__ void load_sq_bf16_bf(__nv_bfloat16 (*S)[LDS], const __nv_bfloat16* Qbf, int d) {
    if (d == 64 && (((uintptr_t)Qbf & 15) == 0)) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const int idx = threadIdx.x + g * 256, r = idx >> 3, c8 = (idx & 7) * 8;
            *reinterpret_cast<uint4*>(&S[r][c8]) = *reinterpret_cast<const uint4*>(Qbf + (long)r * 64 + c8);
        }
        return;
    }
#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const int e = threadIdx.x + g * 256, r = e >> 6, c = e & 63;
        S[r][c] = (r < d && c < d) ? Qbf[(long)r * d + c] : __float2bfloat16_rn(0.f);
    }
}

__device__ __forceinline__ void load_sq_bf16_rm(__nv_bfloat16 (*S)[LDS], const float* Q, int d) {
    if (d == 64 && (((uintptr_t)Q & 15) == 0)) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int idx = threadIdx.x + g * 256, r = idx >> 4, c4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(Q + (long)r * 64 + c4);
            uint2 h4;
            h4.x = ah::pack2<__nv_bfloat16>(v.x, v.y);
            h4.y = ah::pack2<__nv_bfloat16>(v.z, v.w);
            *reinterpret_cast<uint2*>(&S[r][c4]) = h4;
        }
        return;
    }
#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const int e = threadIdx.x + g * 256, r = e >> 6, c = e & 63;
        S[r][c] = (r < d && c < d) ? ah::from_f<__nv_bfloat16>(Q[(long)r * d + c]) : __float2bfloat16_rn(0.f);
    }
}

__device__ __forceinline__ void load_mat_bf16_rm(__nv_bfloat16 (*S)[LDS], const float* X, long xi, long xj, int m,
                                                 int nc) {
    if (xj == 1 && m == 64 && nc == 64 && (xi & 3) == 0 && (((uintptr_t)X & 15) == 0)) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int idx = threadIdx.x + g * 256, r = idx >> 4, c4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(X + (long)r * xi + c4);
            uint2 h4;
            h4.x = ah::pack2<__nv_bfloat16>(v.x, v.y);
            h4.y = ah::pack2<__nv_bfloat16>(v.z, v.w);
            *reinterpret_cast<uint2*>(&S[r][c4]) = h4;
        }
        return;
    }
    const bool col_fast = (xi == 1 && xj != 1);
#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const int e = threadIdx.x + g * 256, hi = e >> 6, lo = e & 63;
        const int r = col_fast ? lo : hi, c = col_fast ? hi : lo;
        S[r][c] = (r < m && c < nc) ? ah::from_f<__nv_bfloat16>(X[(long)r * xi + (long)c * xj]) : __float2bfloat16_rn(0.f);
    }
}

__device__ __forceinline__ void load_g_upd_m_bf16_rm(__nv_bfloat16 (*S)[LDS], const float* Gp, float* Mp, long xi,
                                                     long xj, int m, int nc, float coef, float fb1, float a1) {
    if (xj == 1 && m == 64 && nc == 64 && (xi & 3) == 0 && (((uintptr_t)Gp & 15) == 0) && (((uintptr_t)Mp & 15) == 0)) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int idx = threadIdx.x + g * 256, r = idx >> 4, c4 = (idx & 15) * 4;
            const long off = (long)r * xi + c4;
            const float4 vg = *reinterpret_cast<const float4*>(Gp + off);
            float4* mp = reinterpret_cast<float4*>(Mp + off);
            const float4 vm = *mp;
            const float g0 = vg.x * coef, g1 = vg.y * coef, g2 = vg.z * coef, g3 = vg.w * coef;
            *mp = make_float4(fb1 * vm.x + a1 * g0, fb1 * vm.y + a1 * g1, fb1 * vm.z + a1 * g2, fb1 * vm.w + a1 * g3);
            uint2 h4;
            h4.x = ah::pack2<__nv_bfloat16>(g0, g1);
            h4.y = ah::pack2<__nv_bfloat16>(g2, g3);
            *reinterpret_cast<uint2*>(&S[r][c4]) = h4;
        }
        return;
    }
    const bool col_fast = (xi == 1 && xj != 1);
#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const int e = threadIdx.x + g * 256, hi = e >> 6, lo = e & 63;
        const int r = col_fast ? lo : hi, c = col_fast ? hi : lo;
        if (r < m && c < nc) {
            const long off = (long)r * xi + (long)c * xj;
            const float gv = Gp[off] * coef;
            Mp[off] = fb1 * Mp[off] + a1 * gv;
            S[r][c] = ah::from_f<__nv_bfloat16>(gv);
        } else {
            S[r][c] = __float2bfloat16_rn(0.f);
        }
    }
}

__device__ __forceinline__ void load_g_bf16_rm(__nv_bfloat16 (*S)[LDS], const float* Gp, long xi, long xj, int m,
                                               int nc, float coef) {
    if (xj == 1 && m == 64 && nc == 64 && (xi & 3) == 0 && (((uintptr_t)Gp & 15) == 0)) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int idx = threadIdx.x + g * 256, r = idx >> 4, c4 = (idx & 15) * 4;
            const float4 vg = *reinterpret_cast<const float4*>(Gp + (long)r * xi + c4);
            uint2 h4;
            h4.x = ah::pack2<__nv_bfloat16>(vg.x * coef, vg.y * coef);
            h4.y = ah::pack2<__nv_bfloat16>(vg.z * coef, vg.w * coef);
            *reinterpret_cast<uint2*>(&S[r][c4]) = h4;
        }
        return;
    }
    const bool col_fast = (xi == 1 && xj != 1);
#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const int e = threadIdx.x + g * 256, hi = e >> 6, lo = e & 63;
        const int r = col_fast ? lo : hi, c = col_fast ? hi : lo;
        S[r][c] = (r < m && c < nc) ? ah::from_f<__nv_bfloat16>(Gp[(long)r * xi + (long)c * xj] * coef)
                                    : __float2bfloat16_rn(0.f);
    }
}

__device__ __forceinline__ void store_acc_f_rm(float (*Sf)[65], const float acc[4][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = 32 * wn + 8 * j + 2 * tl;
        Sf[r0][c0] = acc[j][0];
        Sf[r0][c0 + 1] = acc[j][1];
        Sf[r1][c0] = acc[j][2];
        Sf[r1][c0 + 1] = acc[j][3];
    }
}

__device__ __forceinline__ void load_acc_f_rm(const float (*Sf)[65], float out[4][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = 32 * wn + 8 * j + 2 * tl;
        out[j][0] = Sf[r0][c0];
        out[j][1] = Sf[r0][c0 + 1];
        out[j][2] = Sf[r1][c0];
        out[j][3] = Sf[r1][c0 + 1];
    }
}

__device__ __forceinline__ void store_acc_bf16_rm(__nv_bfloat16 (*S)[LDS], const float acc[4][4], int m, int nc) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = 32 * wn + 8 * j + 2 * tl, c1 = c0 + 1;
        const float v00 = (r0 < m && c0 < nc) ? acc[j][0] : 0.f;
        const float v01 = (r0 < m && c1 < nc) ? acc[j][1] : 0.f;
        const float v10 = (r1 < m && c0 < nc) ? acc[j][2] : 0.f;
        const float v11 = (r1 < m && c1 < nc) ? acc[j][3] : 0.f;
        *reinterpret_cast<uint32_t*>(&S[r0][c0]) = ah::pack2<__nv_bfloat16>(v00, v01);
        *reinterpret_cast<uint32_t*>(&S[r1][c0]) = ah::pack2<__nv_bfloat16>(v10, v11);
    }
}

__device__ void soap_apply_bf16(const SoapMat& mt, const PDesc<float>& q, float* ws, int c0, int nc, double t,
                                const double* lr, const int* step_idx, double mult, double b1, double b2, double eps,
                                __nv_bfloat16 (*S0)[LDS], __nv_bfloat16 (*S1)[LDS], __nv_bfloat16 (*Sqm)[LDS],
                                __nv_bfloat16 (*Sqn)[LDS], float (*Sf)[65], bool x_in_s0, const float* osc = nullptr,
                                int coal = 0, int fuse_g = 0, const float* g_osc = nullptr,
                                const __nv_bfloat16* sqbf = nullptr, int use_mrot = 0, bool is_qr = false) {
    const int m = mt.m, n = mt.n;
    const long xi = mt.xi, xj = mt.xj;
    const float* X = q.dir + mt.off + (long)c0 * xj;
    const float* Mx = q.m + mt.off + (long)c0 * xj;
    const float *Qm = ws + mt.Qm, *Qn = ws + mt.Qn;
    float* V = ws + mt.V + c0;
    float* Mrot = ws + mt.Mr + c0;
    const float a1 = float(1.0 - b1), fb1 = float(b1);
    const float a2 = float(1.0 - b2), fb2 = float(b2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;

    if (!x_in_s0) {
        if (fuse_g && g_osc)
            load_g_bf16_rm(S0, q.g + mt.off + (long)c0 * xj, xi, xj, m, nc, g_osc[3 * LST_N_PARAMS]);
        else
            load_mat_bf16_rm(S0, X, xi, xj, m, nc);
    }
    if (sqbf) {
        load_sq_bf16_bf(Sqm, sqbf + mt.Qm_bf, m);
        if (mt.pn) load_sq_bf16_bf(Sqn, sqbf + mt.Qn_bf, n);
    } else {
        load_sq_bf16_rm(Sqm, Qm, m);
        if (mt.pn) load_sq_bf16_rm(Sqn, Qn, n);
    }
    __syncthreads();
    float acc[4][4];
    gemm_bf16<true, true>(acc, Sqm, S0);          // W1 = Qm^T @ X
    if (mt.pn) {
        store_acc_bf16_rm(S1, acc, m, nc);
        __syncthreads();
        gemm_bf16<false, true>(acc, S1, Sqn);     // Gr = W1 @ Qn
    }
    const float inv_bc1 = 1.f / (1.f - powf((float)b1, (float)t));
    const float inv_bc2 = 1.f / (1.f - powf((float)b2, (float)t));
    const float te = float(eps);
    const bool step1 = (t <= 1.5);
    // Wave 36 fast path: on non-QR steps Qm/Qn are constant, so Qm^T M_t Qn = b1 * Mrot_{t-1} + (1-b1) * Gr_t (exact linearity!).
    // Deletes 2 BF16 TC GEMMs (Qm^T @ M @ Qn), global M re-read, denom[4][4] staging, and 4 barriers on 90% of steps!
    if (use_mrot && coal && !is_qr && !step1) {
        __syncthreads();
        store_acc_f_rm(Sf, acc);
        __syncthreads();
        if (m == 64 && nc == 64 && (n & 3) == 0 && (((uintptr_t)V & 15) == 0) && (((uintptr_t)Mrot & 15) == 0)) {
            float4 ureg[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int idx = threadIdx.x + k * 256, r = idx >> 4, c4 = (idx & 15) * 4;
                float4* vp = reinterpret_cast<float4*>(V + (long)r * n + c4);
                float4* mp = reinterpret_cast<float4*>(Mrot + (long)r * n + c4);
                const float4 vo = *vp, mo = *mp;
                const float g0 = Sf[r][c4 + 0], g1 = Sf[r][c4 + 1], g2 = Sf[r][c4 + 2], g3 = Sf[r][c4 + 3];
                const float4 vn = make_float4(fb2 * vo.x + a2 * g0 * g0, fb2 * vo.y + a2 * g1 * g1,
                                              fb2 * vo.z + a2 * g2 * g2, fb2 * vo.w + a2 * g3 * g3);
                const float4 mn = make_float4(fb1 * mo.x + a1 * g0, fb1 * mo.y + a1 * g1,
                                              fb1 * mo.z + a1 * g2, fb1 * mo.w + a1 * g3);
                *vp = vn;
                *mp = mn;
                ureg[k] = make_float4((mn.x * inv_bc1) / (sqrtf(vn.x * inv_bc2) + te),
                                      (mn.y * inv_bc1) / (sqrtf(vn.y * inv_bc2) + te),
                                      (mn.z * inv_bc1) / (sqrtf(vn.z * inv_bc2) + te),
                                      (mn.w * inv_bc1) / (sqrtf(vn.w * inv_bc2) + te));
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int idx = threadIdx.x + k * 256, r = idx >> 4, c4 = (idx & 15) * 4;
                *reinterpret_cast<uint32_t*>(&S0[r][c4 + 0]) = ah::pack2<__nv_bfloat16>(ureg[k].x, ureg[k].y);
                *reinterpret_cast<uint32_t*>(&S0[r][c4 + 2]) = ah::pack2<__nv_bfloat16>(ureg[k].z, ureg[k].w);
            }
            __syncthreads();
        } else {
            float ureg[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const int e = threadIdx.x + k * 256, r = e >> 6, c = e & 63;
                if (r < m && c < nc) {
                    const long kk = (long)r * n + c;
                    const float gv = Sf[r][c];
                    const float vn = fb2 * V[kk] + a2 * gv * gv;
                    const float mn = fb1 * Mrot[kk] + a1 * gv;
                    V[kk] = vn;
                    Mrot[kk] = mn;
                    ureg[k] = (mn * inv_bc1) / (sqrtf(vn * inv_bc2) + te);
                } else {
                    ureg[k] = 0.f;
                }
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const int e = threadIdx.x + k * 256, r = e >> 6, c = e & 63;
                S0[r][c] = __float2bfloat16(ureg[k]);
            }
            __syncthreads();
        }
    } else {
    float denom[4][4];
    if (coal) {
        __syncthreads();
        store_acc_f_rm(Sf, acc);
        __syncthreads();
        if (m == 64 && nc == 64 && (n & 3) == 0 && (((uintptr_t)V & 15) == 0)) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int idx = threadIdx.x + k * 256, r = idx >> 4, c4 = (idx & 15) * 4;
                float4* vp = reinterpret_cast<float4*>(V + (long)r * n + c4);
                const float4 vo = *vp;
                const float g0 = Sf[r][c4 + 0], g1 = Sf[r][c4 + 1], g2 = Sf[r][c4 + 2], g3 = Sf[r][c4 + 3];
                const float4 vn = make_float4(fb2 * vo.x + a2 * g0 * g0, fb2 * vo.y + a2 * g1 * g1,
                                              fb2 * vo.z + a2 * g2 * g2, fb2 * vo.w + a2 * g3 * g3);
                *vp = vn;
                if (use_mrot && step1 && (((uintptr_t)Mrot & 15) == 0)) {
                    *reinterpret_cast<float4*>(Mrot + (long)r * n + c4) =
                        make_float4(a1 * g0, a1 * g1, a1 * g2, a1 * g3);
                }
                const float d0 = sqrtf(vn.x * inv_bc2) + te, d1 = sqrtf(vn.y * inv_bc2) + te;
                const float d2 = sqrtf(vn.z * inv_bc2) + te, d3 = sqrtf(vn.w * inv_bc2) + te;
                Sf[r][c4 + 0] = step1 ? (g0 / d0) : d0;
                Sf[r][c4 + 1] = step1 ? (g1 / d1) : d1;
                Sf[r][c4 + 2] = step1 ? (g2 / d2) : d2;
                Sf[r][c4 + 3] = step1 ? (g3 / d3) : d3;
            }
        } else {
            const int tot = m * nc;
            for (int e = threadIdx.x; e < tot; e += 256) {
                const int r = e / nc, c = e - r * nc;
                const long kk = (long)r * n + c;
                const float gv = Sf[r][c];
                const float vn = fb2 * V[kk] + a2 * gv * gv;
                V[kk] = vn;
                if (use_mrot && step1) Mrot[kk] = a1 * gv;
                const float dv = sqrtf(vn * inv_bc2) + te;
                Sf[r][c] = step1 ? (gv / dv) : dv;
            }
        }
        __syncthreads();
        if (step1) load_acc_f_rm(Sf, acc);
        else load_acc_f_rm(Sf, denom);
        __syncthreads();
    } else {
    const bool v_f2 = ((n & 1) == 0) && ((nc & 1) == 0) && ((((uintptr_t)V) & 7) == 0);
    if (v_f2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col0 = 32 * wn + 8 * j + 2 * tl;
            if (col0 < nc) {
                if (r0 < m) {
                    float2* vp = reinterpret_cast<float2*>(V + (long)r0 * n + col0);
                    const float2 vo = *vp;
                    const float2 vn = make_float2(fb2 * vo.x + a2 * acc[j][0] * acc[j][0],
                                                  fb2 * vo.y + a2 * acc[j][1] * acc[j][1]);
                    *vp = vn;
                    denom[j][0] = sqrtf(vn.x * inv_bc2) + te;
                    denom[j][1] = sqrtf(vn.y * inv_bc2) + te;
                }
                if (r1 < m) {
                    float2* vp = reinterpret_cast<float2*>(V + (long)r1 * n + col0);
                    const float2 vo = *vp;
                    const float2 vn = make_float2(fb2 * vo.x + a2 * acc[j][2] * acc[j][2],
                                                  fb2 * vo.y + a2 * acc[j][3] * acc[j][3]);
                    *vp = vn;
                    denom[j][2] = sqrtf(vn.x * inv_bc2) + te;
                    denom[j][3] = sqrtf(vn.y * inv_bc2) + te;
                }
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col0 = 32 * wn + 8 * j + 2 * tl;
            if (r0 < m) {
                if (col0 < nc) {
                    const long k = (long)r0 * n + col0;
                    const float v = fb2 * V[k] + a2 * acc[j][0] * acc[j][0];
                    V[k] = v; denom[j][0] = sqrtf(v * inv_bc2) + te;
                }
                if (col0 + 1 < nc) {
                    const long k = (long)r0 * n + col0 + 1;
                    const float v = fb2 * V[k] + a2 * acc[j][1] * acc[j][1];
                    V[k] = v; denom[j][1] = sqrtf(v * inv_bc2) + te;
                }
            }
            if (r1 < m) {
                if (col0 < nc) {
                    const long k = (long)r1 * n + col0;
                    const float v = fb2 * V[k] + a2 * acc[j][2] * acc[j][2];
                    V[k] = v; denom[j][2] = sqrtf(v * inv_bc2) + te;
                }
                if (col0 + 1 < nc) {
                    const long k = (long)r1 * n + col0 + 1;
                    const float v = fb2 * V[k] + a2 * acc[j][3] * acc[j][3];
                    V[k] = v; denom[j][3] = sqrtf(v * inv_bc2) + te;
                }
            }
        }
    }
    if (step1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col0 = 32 * wn + 8 * j + 2 * tl;
            acc[j][0] = (r0 < m && col0 < nc) ? (acc[j][0] / denom[j][0]) : 0.f;
            acc[j][1] = (r0 < m && col0 + 1 < nc) ? (acc[j][1] / denom[j][1]) : 0.f;
            acc[j][2] = (r1 < m && col0 < nc) ? (acc[j][2] / denom[j][2]) : 0.f;
            acc[j][3] = (r1 < m && col0 + 1 < nc) ? (acc[j][3] / denom[j][3]) : 0.f;
        }
    }
    if (!mt.pn) __syncthreads();
    }
    if (!step1) {
        load_mat_bf16_rm(S0, Mx, xi, xj, m, nc);
        __syncthreads();
        gemm_bf16<true, true>(acc, Sqm, S0);          // W3 = Qm^T @ Mx
        if (mt.pn) {
            store_acc_bf16_rm(S1, acc, m, nc);
            __syncthreads();
            gemm_bf16<false, true>(acc, S1, Sqn);     // Mr = W3 @ Qn
        }
        if (use_mrot && coal) {
            __syncthreads();
            store_acc_f_rm(Sf, acc);
            __syncthreads();
            if (m == 64 && nc == 64 && (n & 3) == 0 && (((uintptr_t)Mrot & 15) == 0)) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int idx = threadIdx.x + k * 256, r = idx >> 4, c4 = (idx & 15) * 4;
                    *reinterpret_cast<float4*>(Mrot + (long)r * n + c4) =
                        make_float4(Sf[r][c4 + 0], Sf[r][c4 + 1], Sf[r][c4 + 2], Sf[r][c4 + 3]);
                }
            } else {
                const int tot = m * nc;
                for (int e = threadIdx.x; e < tot; e += 256) {
                    const int r = e / nc, c = e - r * nc;
                    Mrot[(long)r * n + c] = Sf[r][c];
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col0 = 32 * wn + 8 * j + 2 * tl;
            acc[j][0] = (r0 < m && col0 < nc) ? (acc[j][0] * inv_bc1) / denom[j][0] : 0.f;
            acc[j][1] = (r0 < m && col0 + 1 < nc) ? (acc[j][1] * inv_bc1) / denom[j][1] : 0.f;
            acc[j][2] = (r1 < m && col0 < nc) ? (acc[j][2] * inv_bc1) / denom[j][2] : 0.f;
            acc[j][3] = (r1 < m && col0 + 1 < nc) ? (acc[j][3] * inv_bc1) / denom[j][3] : 0.f;
        }
        if (!mt.pn) __syncthreads();
    }
    store_acc_bf16_rm(S0, acc, m, nc);
    __syncthreads();
    }
    gemm_bf16<false, true>(acc, Sqm, S0);         // W3' = Qm @ Ur
    if (mt.pn) {
        store_acc_bf16_rm(S1, acc, m, nc);
        __syncthreads();
        gemm_bf16<false, false>(acc, S1, Sqn);    // U = W3' @ Qn^T
    }
    const float f = float(lr[16L * step_idx[0] + q.group] * mult);
    const float omd = osc ? osc[3 * LST_N_PARAMS + 1] : 0.f;
    const bool do_ema = osc && q.ema != nullptr && omd != 0.f;
    const bool copy_ema = (omd == 1.f);
    if (coal) {
        __syncthreads();
        store_acc_f_rm(Sf, acc);
        __syncthreads();
        if (xj == 1 && m == 64 && nc == 64 && (xi & 3) == 0 && ((((uintptr_t)(q.p + mt.off + c0)) & 15) == 0) &&
            (!do_ema || ((((uintptr_t)(q.ema + mt.off + c0)) & 15) == 0))) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int idx = threadIdx.x + k * 256, r = idx >> 4, c4 = (idx & 15) * 4;
                const long off = mt.off + (long)r * xi + (long)(c0 + c4);
                float4* pp = reinterpret_cast<float4*>(q.p + off);
                const float4 po = *pp;
                const float4 pn = make_float4(po.x - f * Sf[r][c4 + 0], po.y - f * Sf[r][c4 + 1],
                                              po.z - f * Sf[r][c4 + 2], po.w - f * Sf[r][c4 + 3]);
                *pp = pn;
                if (do_ema) {
                    float4* ep = reinterpret_cast<float4*>(q.ema + off);
                    if (copy_ema) {
                        *ep = pn;
                    } else {
                        const float4 eo = *ep;
                        *ep = make_float4(eo.x - omd * (eo.x - pn.x), eo.y - omd * (eo.y - pn.y),
                                          eo.z - omd * (eo.z - pn.z), eo.w - omd * (eo.w - pn.w));
                    }
                }
            }
        } else {
            const bool col_fast = (xi == 1 && xj != 1);
            const int tot = m * nc;
            for (int e = threadIdx.x; e < tot; e += 256) {
                const int r = col_fast ? (e % m) : (e / nc);
                const int c = col_fast ? (e / m) : (e % nc);
                const long idx = mt.off + (long)r * xi + (long)(c0 + c) * xj;
                const float pnew = q.p[idx] - f * Sf[r][c];
                q.p[idx] = pnew;
                if (do_ema) q.ema[idx] = copy_ema ? pnew : (q.ema[idx] - omd * (q.ema[idx] - pnew));
            }
        }
        return;
    }
    const bool p_f2 = (xj == 1) && ((xi & 1) == 0) && ((nc & 1) == 0) &&
                      ((((uintptr_t)(q.p + mt.off + c0)) & 7) == 0) &&
                      (!do_ema || ((((uintptr_t)(q.ema + mt.off + c0)) & 7) == 0));
    if (p_f2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col0 = 32 * wn + 8 * j + 2 * tl;
            if (col0 < nc) {
                if (r0 < m) {
                    const long idx = mt.off + (long)r0 * xi + (long)(c0 + col0);
                    float2* pp = reinterpret_cast<float2*>(q.p + idx);
                    const float2 po = *pp;
                    const float2 pn = make_float2(po.x - f * acc[j][0], po.y - f * acc[j][1]);
                    *pp = pn;
                    if (do_ema) {
                        float2* ep = reinterpret_cast<float2*>(q.ema + idx);
                        if (copy_ema) {
                            *ep = pn;
                        } else {
                            const float2 eo = *ep;
                            *ep = make_float2(eo.x - omd * (eo.x - pn.x), eo.y - omd * (eo.y - pn.y));
                        }
                    }
                }
                if (r1 < m) {
                    const long idx = mt.off + (long)r1 * xi + (long)(c0 + col0);
                    float2* pp = reinterpret_cast<float2*>(q.p + idx);
                    const float2 po = *pp;
                    const float2 pn = make_float2(po.x - f * acc[j][2], po.y - f * acc[j][3]);
                    *pp = pn;
                    if (do_ema) {
                        float2* ep = reinterpret_cast<float2*>(q.ema + idx);
                        if (copy_ema) {
                            *ep = pn;
                        } else {
                            const float2 eo = *ep;
                            *ep = make_float2(eo.x - omd * (eo.x - pn.x), eo.y - omd * (eo.y - pn.y));
                        }
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col0 = 32 * wn + 8 * j + 2 * tl;
            auto upd = [&](int rr, int cc, float uv) {
                if (rr < m && cc < nc) {
                    const long idx = mt.off + (long)rr * xi + (long)(c0 + cc) * xj;
                    const float pnew = q.p[idx] - f * uv;
                    q.p[idx] = pnew;
                    if (do_ema) q.ema[idx] = copy_ema ? pnew : (q.ema[idx] - omd * (q.ema[idx] - pnew));
                }
            };
            upd(r0, col0, acc[j][0]);
            upd(r0, col0 + 1, acc[j][1]);
            upd(r1, col0, acc[j][2]);
            upd(r1, col0 + 1, acc[j][3]);
        }
    }
}
}  // namespace sp_bf

__global__ void __launch_bounds__(256, 2) k_soap_a_smem_bf16(const SoapMat* mats, const int2* units,
                                                             const PDesc<float>* d, const double* lr,
                                                             const int* step_idx, float* ws, double mult, double b1,
                                                             double b2, double eps, int every,
                                                             const float* osc = nullptr, int coal = 0, int fuse_g = 0,
                                                             const float* g_osc = nullptr,
                                                             const __nv_bfloat16* sqbf = nullptr, int use_mrot = 0,
                                                             int stat_every = 1, float fb2_stat = 0.f) {
    union SmPair {
        __nv_bfloat16 bf[2][64][sp_bf::LDS];
        float f[64][65];
    };
    __shared__ __align__(16) SmPair Us;
    __shared__ __align__(16) SmPair Uq;
    auto S0 = Us.bf[0], S1 = Us.bf[1], Sqm = Uq.bf[0], Sqn = Uq.bf[1];
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<float> q = d[mt.param];
    const int m = mt.m, n = mt.n;
    const int c0 = mt.nch > 1 ? u.y * sp::TS : 0, nc = mt.nch > 1 ? min(sp::TS, n - c0) : n;
    const float* X = q.dir + mt.off + (long)c0 * mt.xj;
    const double t = (double)*q.step;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;

    const bool is_qr = (((long)t - 1) % every == 0);
    const bool step1 = (t <= 1.5);
    const bool skip_qm = (use_mrot >= 2) && coal && fuse_g && g_osc && !step1;
    const bool do_stat = (stat_every <= 1) || (((long)t - 1) % stat_every == 0);
    const float fb2 = (stat_every > 1) ? fb2_stat : float(b2);
    const float a2 = (stat_every > 1) ? (1.f - fb2_stat) : float(1.0 - b2);
    if (fuse_g && g_osc) {
        if (skip_qm) {
            if (do_stat || !is_qr)
                sp_bf::load_g_bf16_rm(S0, q.g + mt.off + (long)c0 * mt.xj, mt.xi, mt.xj, m, nc, g_osc[3 * LST_N_PARAMS]);
        } else {
            sp_bf::load_g_upd_m_bf16_rm(S0, q.g + mt.off + (long)c0 * mt.xj, q.m + mt.off + (long)c0 * mt.xj, mt.xi, mt.xj,
                                        m, nc, g_osc[3 * LST_N_PARAMS], float(b1), float(1.0 - b1));
        }
    } else {
        sp_bf::load_mat_bf16_rm(S0, X, mt.xi, mt.xj, m, nc);
    }
    float acc[4][4];
    if (do_stat) {
        __syncthreads();
        auto ema_sq = [&](float* M, int dim) {
            if (coal) {
                sp_bf::store_acc_f_rm(Uq.f, acc);
                __syncthreads();
                if (dim == 64 && (((uintptr_t)M & 15) == 0)) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int idx = threadIdx.x + k * 256, r = idx >> 4, c4 = (idx & 15) * 4;
                        float4* mp = reinterpret_cast<float4*>(M + (long)r * 64 + c4);
                        const float4 mo = *mp;
                        *mp = make_float4(a2 * Uq.f[r][c4 + 0] + fb2 * mo.x, a2 * Uq.f[r][c4 + 1] + fb2 * mo.y,
                                          a2 * Uq.f[r][c4 + 2] + fb2 * mo.z, a2 * Uq.f[r][c4 + 3] + fb2 * mo.w);
                    }
                } else {
                    const int tot = dim * dim;
                    for (int e = threadIdx.x; e < tot; e += 256) {
                        const int r = e / dim, c = e - r * dim;
                        M[e] = a2 * Uq.f[r][c] + fb2 * M[e];
                    }
                }
                __syncthreads();
                return;
            }
            const bool m_f2 = ((dim & 1) == 0) && ((((uintptr_t)M) & 7) == 0);
            if (m_f2) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col0 = 32 * wn + 8 * j + 2 * tl;
                    if (col0 < dim) {
                        if (r0 < dim) {
                            float2* mp = reinterpret_cast<float2*>(M + (long)r0 * dim + col0);
                            const float2 mo = *mp;
                            *mp = make_float2(a2 * acc[j][0] + fb2 * mo.x, a2 * acc[j][1] + fb2 * mo.y);
                        }
                        if (r1 < dim) {
                            float2* mp = reinterpret_cast<float2*>(M + (long)r1 * dim + col0);
                            const float2 mo = *mp;
                            *mp = make_float2(a2 * acc[j][2] + fb2 * mo.x, a2 * acc[j][3] + fb2 * mo.y);
                        }
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col0 = 32 * wn + 8 * j + 2 * tl;
                    if (r0 < dim) {
                        if (col0 < dim) M[(long)r0 * dim + col0] = a2 * acc[j][0] + fb2 * M[(long)r0 * dim + col0];
                        if (col0 + 1 < dim) M[(long)r0 * dim + col0 + 1] = a2 * acc[j][1] + fb2 * M[(long)r0 * dim + col0 + 1];
                    }
                    if (r1 < dim) {
                        if (col0 < dim) M[(long)r1 * dim + col0] = a2 * acc[j][2] + fb2 * M[(long)r1 * dim + col0];
                        if (col0 + 1 < dim) M[(long)r1 * dim + col0 + 1] = a2 * acc[j][3] + fb2 * M[(long)r1 * dim + col0 + 1];
                    }
                }
            }
        };
        if (mt.pm) {
            sp_bf::gemm_bf16<false, false>(acc, S0, S0);   // Lm = X @ X^T
            float* L = mt.nch > 1 ? ws + mt.Lp + (long)u.y * m * m : ws + mt.Lm;
            ema_sq(L, m);
        }
        if (mt.pn) {
            sp_bf::gemm_bf16<true, true>(acc, S0, S0);     // Rn = X^T @ X
            ema_sq(ws + mt.Rn, n);
        }
    }
    if (is_qr) {
        if (skip_qm) {
            // Wave 37 (LST_SOAP_MROT=2): unrotate Mrot under OLD (Qm, Qn) into q.m BEFORE k_soap_eig_qr_smem refreshes Qm/Qn!
            if (sqbf) {
                sp_bf::load_sq_bf16_bf(Sqm, sqbf + mt.Qm_bf, m);
                if (mt.pn) sp_bf::load_sq_bf16_bf(Sqn, sqbf + mt.Qn_bf, n);
            } else {
                sp_bf::load_sq_bf16_rm(Sqm, ws + mt.Qm, m);
                if (mt.pn) sp_bf::load_sq_bf16_rm(Sqn, ws + mt.Qn, n);
            }
            sp_bf::load_mat_bf16_rm(S0, ws + mt.Mr + c0, n, 1, m, nc);
            __syncthreads();
            sp_bf::gemm_bf16<false, true>(acc, Sqm, S0);
            if (mt.pn) {
                sp_bf::store_acc_bf16_rm(S1, acc, m, nc);
                __syncthreads();
                sp_bf::gemm_bf16<false, false>(acc, S1, Sqn);
            }
            __syncthreads();
            sp_bf::store_acc_f_rm(Us.f, acc);
            __syncthreads();
            const float coef = g_osc[3 * LST_N_PARAMS];
            const float fb1 = float(b1), a1 = float(1.0 - b1);
            const float* Gp = q.g + mt.off + (long)c0 * mt.xj;
            float* Mp = q.m + mt.off + (long)c0 * mt.xj;
            if (mt.xj == 1 && m == 64 && nc == 64 && (mt.xi & 3) == 0 && (((uintptr_t)Gp & 15) == 0) &&
                (((uintptr_t)Mp & 15) == 0)) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int idx = threadIdx.x + k * 256, r = idx >> 4, c4 = (idx & 15) * 4;
                    const long off = (long)r * mt.xi + c4;
                    const float4 vg = *reinterpret_cast<const float4*>(Gp + off);
                    *reinterpret_cast<float4*>(Mp + off) = make_float4(
                        fb1 * Us.f[r][c4 + 0] + a1 * (vg.x * coef),
                        fb1 * Us.f[r][c4 + 1] + a1 * (vg.y * coef),
                        fb1 * Us.f[r][c4 + 2] + a1 * (vg.z * coef),
                        fb1 * Us.f[r][c4 + 3] + a1 * (vg.w * coef));
                }
            } else {
                const bool col_fast = (mt.xi == 1 && mt.xj != 1);
                const int tot = m * nc;
                for (int e = threadIdx.x; e < tot; e += 256) {
                    const int r = col_fast ? (e % m) : (e / nc);
                    const int c = col_fast ? (e / m) : (e % nc);
                    const long off = (long)r * mt.xi + (long)c * mt.xj;
                    Mp[off] = fb1 * Us.f[r][c] + a1 * (Gp[off] * coef);
                }
            }
        }
        return;
    }
    sp_bf::soap_apply_bf16(mt, q, ws, c0, nc, t, lr, step_idx, mult, b1, b2, eps, S0, S1, Sqm, Sqn, Us.f, true, osc,
                           coal, fuse_g, g_osc, sqbf, use_mrot, false);
}

// Wave 45 (LST_SOAP_FAST=1): Dedicated lean non-QR BF16 TC SOAP kernel (soap_mode == 2, use_mrot >= 2, coal=1,
// fuse_g=1, sqbf != nullptr). Strips all QR / step-1 / legacy branches & denom[4][4] (~10x fewer SASS insns, lower regs).
__global__ void __launch_bounds__(256, 2) k_soap_a_fast_bf16(const SoapMat* mats, const int2* units,
                                                             const PDesc<float>* d, const double* lr,
                                                             const int* step_idx, float* ws, double mult, double b1,
                                                             double b2, double eps, const float* osc,
                                                             const float* g_osc, const __nv_bfloat16* sqbf,
                                                             int stat_every, float fb2_stat) {
    union SmPair {
        __nv_bfloat16 bf[2][64][sp_bf::LDS];
        float f[64][65];
    };
    __shared__ __align__(16) SmPair Us;
    __shared__ __align__(16) SmPair Uq;
    auto S0 = Us.bf[0], S1 = Us.bf[1], Sqm = Uq.bf[0], Sqn = Uq.bf[1];
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<float> q = d[mt.param];
    const int m = mt.m, n = mt.n;
    const long xi = mt.xi, xj = mt.xj;
    const int c0 = mt.nch > 1 ? u.y * sp::TS : 0, nc = mt.nch > 1 ? min(sp::TS, n - c0) : n;
    const double t = (double)*q.step;
    const bool do_stat = (stat_every <= 1) || (((long)t - 1) % stat_every == 0);
    const float fb2_st = (stat_every > 1) ? fb2_stat : float(b2);
    const float a2_st = (stat_every > 1) ? (1.f - fb2_stat) : float(1.0 - b2);

    sp_bf::load_g_bf16_rm(S0, q.g + mt.off + (long)c0 * xj, xi, xj, m, nc, g_osc[3 * LST_N_PARAMS]);
    float acc[4][4];
    if (do_stat) {
        __syncthreads();
        auto ema_sq = [&](float* M, int dim) {
            sp_bf::store_acc_f_rm(Uq.f, acc);
            __syncthreads();
            if (dim == 64 && (((uintptr_t)M & 15) == 0)) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int idx = threadIdx.x + k * 256, r = idx >> 4, c4 = (idx & 15) * 4;
                    float4* mp = reinterpret_cast<float4*>(M + (long)r * 64 + c4);
                    const float4 mo = *mp;
                    *mp = make_float4(a2_st * Uq.f[r][c4 + 0] + fb2_st * mo.x, a2_st * Uq.f[r][c4 + 1] + fb2_st * mo.y,
                                      a2_st * Uq.f[r][c4 + 2] + fb2_st * mo.z, a2_st * Uq.f[r][c4 + 3] + fb2_st * mo.w);
                }
            } else {
                const int tot = dim * dim;
                for (int e = threadIdx.x; e < tot; e += 256) {
                    const int r = e / dim, c = e - r * dim;
                    M[e] = a2_st * Uq.f[r][c] + fb2_st * M[e];
                }
            }
            __syncthreads();
        };
        if (mt.pm) {
            sp_bf::gemm_bf16<false, false>(acc, S0, S0);
            float* L = mt.nch > 1 ? ws + mt.Lp + (long)u.y * m * m : ws + mt.Lm;
            ema_sq(L, m);
        }
        if (mt.pn) {
            sp_bf::gemm_bf16<true, true>(acc, S0, S0);
            ema_sq(ws + mt.Rn, n);
        }
    }
    sp_bf::load_sq_bf16_bf(Sqm, sqbf + mt.Qm_bf, m);
    if (mt.pn) sp_bf::load_sq_bf16_bf(Sqn, sqbf + mt.Qn_bf, n);
    __syncthreads();
    sp_bf::gemm_bf16<true, true>(acc, Sqm, S0);
    if (mt.pn) {
        sp_bf::store_acc_bf16_rm(S1, acc, m, nc);
        __syncthreads();
        sp_bf::gemm_bf16<false, true>(acc, S1, Sqn);
    }
    const float a1 = float(1.0 - b1), fb1 = float(b1);
    const float a2 = float(1.0 - b2), fb2 = float(b2);
    const float inv_bc1 = 1.f / (1.f - powf((float)b1, (float)t));
    const float inv_bc2 = 1.f / (1.f - powf((float)b2, (float)t));
    const float te = float(eps);
    float* V = ws + mt.V + c0;
    float* Mrot = ws + mt.Mr + c0;
    __syncthreads();
    sp_bf::store_acc_f_rm(Us.f, acc);
    __syncthreads();
    if (m == 64 && nc == 64 && (n & 3) == 0 && (((uintptr_t)V & 15) == 0) && (((uintptr_t)Mrot & 15) == 0)) {
        float4 ureg[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int idx = threadIdx.x + k * 256, r = idx >> 4, c4 = (idx & 15) * 4;
            float4* vp = reinterpret_cast<float4*>(V + (long)r * n + c4);
            float4* mp = reinterpret_cast<float4*>(Mrot + (long)r * n + c4);
            const float4 vo = *vp, mo = *mp;
            const float g0 = Us.f[r][c4 + 0], g1 = Us.f[r][c4 + 1], g2 = Us.f[r][c4 + 2], g3 = Us.f[r][c4 + 3];
            const float4 vn = make_float4(fb2 * vo.x + a2 * g0 * g0, fb2 * vo.y + a2 * g1 * g1,
                                          fb2 * vo.z + a2 * g2 * g2, fb2 * vo.w + a2 * g3 * g3);
            const float4 mn = make_float4(fb1 * mo.x + a1 * g0, fb1 * mo.y + a1 * g1,
                                          fb1 * mo.z + a1 * g2, fb1 * mo.w + a1 * g3);
            *vp = vn;
            *mp = mn;
            ureg[k] = make_float4((mn.x * inv_bc1) / (sqrtf(vn.x * inv_bc2) + te),
                                  (mn.y * inv_bc1) / (sqrtf(vn.y * inv_bc2) + te),
                                  (mn.z * inv_bc1) / (sqrtf(vn.z * inv_bc2) + te),
                                  (mn.w * inv_bc1) / (sqrtf(vn.w * inv_bc2) + te));
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int idx = threadIdx.x + k * 256, r = idx >> 4, c4 = (idx & 15) * 4;
            *reinterpret_cast<uint32_t*>(&S0[r][c4 + 0]) = ah::pack2<__nv_bfloat16>(ureg[k].x, ureg[k].y);
            *reinterpret_cast<uint32_t*>(&S0[r][c4 + 2]) = ah::pack2<__nv_bfloat16>(ureg[k].z, ureg[k].w);
        }
        __syncthreads();
    } else {
        float ureg[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const int e = threadIdx.x + k * 256, r = e >> 6, c = e & 63;
            if (r < m && c < nc) {
                const long kk = (long)r * n + c;
                const float gv = Us.f[r][c];
                const float vn = fb2 * V[kk] + a2 * gv * gv;
                const float mn = fb1 * Mrot[kk] + a1 * gv;
                V[kk] = vn;
                Mrot[kk] = mn;
                ureg[k] = (mn * inv_bc1) / (sqrtf(vn * inv_bc2) + te);
            } else {
                ureg[k] = 0.f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const int e = threadIdx.x + k * 256, r = e >> 6, c = e & 63;
            S0[r][c] = __float2bfloat16(ureg[k]);
        }
        __syncthreads();
    }
    sp_bf::gemm_bf16<false, true>(acc, Sqm, S0);
    if (mt.pn) {
        sp_bf::store_acc_bf16_rm(S1, acc, m, nc);
        __syncthreads();
        sp_bf::gemm_bf16<false, false>(acc, S1, Sqn);
    }
    const float f = float(lr[16L * step_idx[0] + q.group] * mult);
    const float omd = osc ? osc[3 * LST_N_PARAMS + 1] : 0.f;
    const bool do_ema = osc && q.ema != nullptr && omd != 0.f;
    const bool copy_ema = (omd == 1.f);
    __syncthreads();
    sp_bf::store_acc_f_rm(Us.f, acc);
    __syncthreads();
    if (xj == 1 && m == 64 && nc == 64 && (xi & 3) == 0 && ((((uintptr_t)(q.p + mt.off + c0)) & 15) == 0) &&
        (!do_ema || ((((uintptr_t)(q.ema + mt.off + c0)) & 15) == 0))) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int idx = threadIdx.x + k * 256, r = idx >> 4, c4 = (idx & 15) * 4;
            const long off = mt.off + (long)r * xi + (long)(c0 + c4);
            float4* pp = reinterpret_cast<float4*>(q.p + off);
            const float4 po = *pp;
            const float4 pn = make_float4(po.x - f * Us.f[r][c4 + 0], po.y - f * Us.f[r][c4 + 1],
                                          po.z - f * Us.f[r][c4 + 2], po.w - f * Us.f[r][c4 + 3]);
            *pp = pn;
            if (do_ema) {
                float4* ep = reinterpret_cast<float4*>(q.ema + off);
                if (copy_ema) {
                    *ep = pn;
                } else {
                    const float4 eo = *ep;
                    *ep = make_float4(eo.x - omd * (eo.x - pn.x), eo.y - omd * (eo.y - pn.y),
                                      eo.z - omd * (eo.z - pn.z), eo.w - omd * (eo.w - pn.w));
                }
            }
        }
    } else {
        const bool col_fast = (xi == 1 && xj != 1);
        const int tot = m * nc;
        for (int e = threadIdx.x; e < tot; e += 256) {
            const int r = col_fast ? (e % m) : (e / nc);
            const int c = col_fast ? (e / m) : (e % nc);
            const long idx = mt.off + (long)r * xi + (long)(c0 + c) * xj;
            const float pnew = q.p[idx] - f * Us.f[r][c];
            q.p[idx] = pnew;
            if (do_ema) q.ema[idx] = copy_ema ? pnew : (q.ema[idx] - omd * (q.ema[idx] - pnew));
        }
    }
}

__global__ void __launch_bounds__(256, 2) k_soap_ap_smem_bf16(const SoapMat* mats, const int2* units,
                                                              const PDesc<float>* d, const double* lr,
                                                              const int* step_idx, float* ws, double mult, double b1,
                                                              double b2, double eps, int every,
                                                              const float* osc = nullptr, int coal = 0, int fuse_g = 0,
                                                              const float* g_osc = nullptr,
                                                              const __nv_bfloat16* sqbf = nullptr, int use_mrot = 0) {
    if (((long)*d[mats[0].param].step - 1) % every != 0) return;
    union SmPair {
        __nv_bfloat16 bf[2][64][sp_bf::LDS];
        float f[64][65];
    };
    __shared__ __align__(16) SmPair Us;
    __shared__ __align__(16) SmPair Uq;
    auto S0 = Us.bf[0], S1 = Us.bf[1], Sqm = Uq.bf[0], Sqn = Uq.bf[1];
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<float> q = d[mt.param];
    const double t = (double)*q.step;
    if (((long)t - 1) % every != 0) return;
    const int n = mt.n;
    const int c0 = mt.nch > 1 ? u.y * sp::TS : 0, nc = mt.nch > 1 ? min(sp::TS, n - c0) : n;
    sp_bf::soap_apply_bf16(mt, q, ws, c0, nc, t, lr, step_idx, mult, b1, b2, eps, S0, S1, Sqm, Sqn, Us.f, false, osc,
                           coal, fuse_g, g_osc, sqbf, use_mrot, true);
}

// One CTA per matrix, refresh steps only: Lm = sum of the partials, Jacobi on Q^T L Q (warm), the update.
template <typename T>
__global__ void __launch_bounds__(256, 2) k_soap_b(const SoapMat* mats, const PDesc<T>* d, const double* lr,
                                                const int* step_idx, T* ws, double mult, double b1, double b2,
                                                double eps, int every, double tol, int warm_sweeps, int jfuse,
                                                int use_smem = 0, const T* osc = nullptr) {
    using namespace sp;
    extern __shared__ unsigned char smem_raw[];
    T* As = reinterpret_cast<T*>(smem_raw);
    T* Bs = As + TS * LDV;
    const SoapMat mt = mats[blockIdx.x];
    const PDesc<T> q = d[mt.param];
    const double t = (double)*q.step;
    if (((long)t - 1) % every != 0) return;
    const int m = mt.m, n = mt.n;
    T *Lm = ws + mt.Lm, *Qm = ws + mt.Qm, *Bm = ws + mt.Bm, *W1 = ws + mt.W1;
    T *Rn = ws + mt.Rn, *Qn = ws + mt.Qn, *Bn = ws + mt.Bn;
    const int maxsw = t <= 1.0 ? 30 : warm_sweeps;   // cold start as muon.py _jacobi; warm: LST_SOAP_SWEEPS
    if (mt.pm) {
        if (mt.nch > 1) {
            const T* Lp = ws + mt.Lp;
            for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
                T acc = Lp[e];
                for (int c = 1; c < mt.nch; ++c) acc += Lp[(long)c * m * m + e];
                Lm[e] = acc;
            }
            __syncthreads();
        }
        bmm<T>(W1, m, Lm, m, 1, Qm, m, 1, m, m, m, T(1), T(0), As, Bs);   // L Q
        bmm<T>(Bm, m, Qm, 1, m, W1, m, 1, m, m, m, T(1), T(0), As, Bs);   // Q^T L Q
        jacobi_any<T>(Bm, Qm, m, maxsw, tol, As, Bs, jfuse);
    }
    if (mt.pn) {
        bmm<T>(W1, n, Rn, n, 1, Qn, n, 1, n, n, n, T(1), T(0), As, Bs);
        bmm<T>(Bn, n, Qn, 1, n, W1, n, 1, n, n, n, T(1), T(0), As, Bs);
        jacobi_any<T>(Bn, Qn, n, maxsw, tol, As, Bs, jfuse);
    }
    if constexpr (std::is_same<T, float>::value) {
        if (use_smem && m <= TS && n <= TS) {
            T* S0 = reinterpret_cast<T*>(smem_raw);
            soap_apply_smem<T>(mt, q, ws, 0, n, t, lr, step_idx, mult, b1, b2, eps, S0, S0 + TS * LDV, false, osc);
            return;
        }
    }
    soap_apply<T>(mt, q, ws, 0, n, t, lr, step_idx, mult, b1, b2, eps, As, Bs, osc);
}

// Refresh steps only, one CTA per (matrix, side): side 0 Lm (sum of the partials), side 1 Rn; Jacobi on Q^T F Q.
template <typename T>
__global__ void __launch_bounds__(256, 2) k_soap_eig(const SoapMat* mats, const int2* units, const PDesc<T>* d, T* ws,
                                                  int every, double tol, int warm_sweeps, int jfuse) {
    using namespace sp;
    extern __shared__ unsigned char smem_raw[];
    T* As = reinterpret_cast<T*>(smem_raw);
    T* Bs = As + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<T> q = d[mt.param];
    const double t = (double)*q.step;
    if (((long)t - 1) % every != 0) return;
    const int maxsw = t <= 1.0 ? 30 : warm_sweeps;
    if (u.y == 0) {
        const int m = mt.m;
        T *Lm = ws + mt.Lm, *Qm = ws + mt.Qm, *Bm = ws + mt.Bm, *W1 = ws + mt.W1;
        if (mt.nch > 1) {
            const T* Lp = ws + mt.Lp;
            for (int e = threadIdx.x; e < m * m; e += blockDim.x) {
                T acc = Lp[e];
                for (int c = 1; c < mt.nch; ++c) acc += Lp[(long)c * m * m + e];
                Lm[e] = acc;
            }
            __syncthreads();
        }
        bmm<T>(W1, m, Lm, m, 1, Qm, m, 1, m, m, m, T(1), T(0), As, Bs);
        bmm<T>(Bm, m, Qm, 1, m, W1, m, 1, m, m, m, T(1), T(0), As, Bs);
        jacobi_any<T>(Bm, Qm, m, maxsw, tol, As, Bs, jfuse);
    } else {
        const int n = mt.n;
        T *Rn = ws + mt.Rn, *Qn = ws + mt.Qn, *Bn = ws + mt.Bn, *Wn = ws + mt.Wn;
        bmm<T>(Wn, n, Rn, n, 1, Qn, n, 1, n, n, n, T(1), T(0), As, Bs);
        bmm<T>(Bn, n, Qn, 1, n, Wn, n, 1, n, n, n, T(1), T(0), As, Bs);
        jacobi_any<T>(Bn, Qn, n, maxsw, tol, As, Bs, jfuse);
    }
}

// Refresh steps only, after k_soap_eig: the update over k_soap_a's units (matrix, column chunk)
template <typename T>
__global__ void __launch_bounds__(256, 2) k_soap_ap(const SoapMat* mats, const int2* units, const PDesc<T>* d,
                                                 const double* lr, const int* step_idx, T* ws, double mult, double b1,
                                                 double b2, double eps, int every, const T* osc = nullptr) {
    using namespace sp;
    extern __shared__ unsigned char smem_raw[];
    T* As = reinterpret_cast<T*>(smem_raw);
    T* Bs = As + TS * LDV;
    const int2 u = units[blockIdx.x];
    const SoapMat mt = mats[u.x];
    const PDesc<T> q = d[mt.param];
    const double t = (double)*q.step;
    if (((long)t - 1) % every != 0) return;
    const int n = mt.n;
    const int c0 = mt.nch > 1 ? u.y * TS : 0, nc = mt.nch > 1 ? min(TS, n - c0) : n;
    soap_apply<T>(mt, q, ws, c0, nc, t, lr, step_idx, mult, b1, b2, eps, As, Bs, osc);
}

// EMA of the Muon tensors after their update (k_adam updates the others' EMA)
template <typename T>
__global__ void k_muon_ema(const PDesc<T>* d, int np, const T* sc) {
    const PDesc<T> q = d[blockIdx.y];
    if (!q.muon || !q.ema) return;
    const T omd = sc[3 * np + 1];
    if (omd == T(0)) return;
    if (omd == T(1)) {
        for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < q.n; i += (long)gridDim.x * blockDim.x)
            q.ema[i] = q.p[i];
        return;
    }
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < q.n; i += (long)gridDim.x * blockDim.x)
        q.ema[i] -= omd * (q.ema[i] - q.p[i]);
}

__global__ void k_step_end(int* step) { *step += 1; }

// ============================================================================ Wave 29: GPU-native ksort_build
// Replaces single-threaded CPU ksort_build (~0.8-1.1s D2H + 3x std::sort + H2D) with 3 per-frame GPU kernels (~3ms)!
template <typename T>
__global__ void k_ksort_pairs_gpu(
    int nf, const int* __restrict__ node_off, const int* __restrict__ edge_off, const int* __restrict__ pair_off,
    const int* __restrict__ rcv, const int* __restrict__ pe0, const int* __restrict__ pe1,
    const T* __restrict__ vec, T rmax, T rlo, T invh, int spK, bool pair_bal,
    int* __restrict__ p_kidx, float* __restrict__ p_us, int* __restrict__ p_e0, int* __restrict__ p_e1,
    int* __restrict__ new_p_of_old, int* __restrict__ old_p_of_new) {
    const int fi = blockIdx.x;
    if (fi >= nf) return;
    const int n_cnt = node_off[fi + 1] - node_off[fi];
    const int e_off = edge_off[fi];
    const int p_off = pair_off[fi], p_cnt = pair_off[fi + 1] - p_off;
    __shared__ int s_deg0[2048];
    __shared__ int s_bcnt[68];
    __shared__ int s_bpos[68];
    for (int i = threadIdx.x; i < n_cnt; i += blockDim.x) s_deg0[i] = 0;
    for (int k = threadIdx.x; k <= spK + 1; k += blockDim.x) s_bcnt[k] = 0;
    __syncthreads();

    for (int lp = threadIdx.x; lp < p_cnt; lp += blockDim.x) {
        const int le0 = pe0[p_off + lp];
        const int le1 = pe1[p_off + lp];
        const T vx = vec[3L * (e_off + le0) + 0];
        const T vy = vec[3L * (e_off + le0) + 1];
        const T vz = vec[3L * (e_off + le0) + 2];
        const T r = sqrt(vx * vx + vy * vy + vz * vz);
        int ki;
        float us;
        if (r >= rmax) {
            ki = -1;
            us = 0.f;
        } else {
            const T pos = (r - rlo) * invh;
            int idx = (int)pos;
            if (idx < 0) idx = 0;
            if (idx >= spK) idx = spK - 1;
            ki = idx;
            us = (float)(pos - T(idx));
        }
        p_kidx[p_off + lp] = ki;
        p_us[p_off + lp] = us;
        atomicAdd(&s_bcnt[(ki < 0 ? spK : ki) + 1], 1);
        if (!pair_bal) {
            p_e0[p_off + lp] = le0;
            p_e1[p_off + lp] = le1;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int acc = 0;
        for (int k = 0; k <= spK; ++k) {
            s_bpos[k] = acc;
            acc += s_bcnt[k + 1];
        }
        if (pair_bal) {
            for (int lp = 0; lp < p_cnt; ++lp) {
                int le0 = pe0[p_off + lp];
                int le1 = pe1[p_off + lp];
                int u = rcv[e_off + le0];
                int v = rcv[e_off + le1];
                if (s_deg0[v] < s_deg0[u]) {
                    int t = le0; le0 = le1; le1 = t;
                    u = v;
                }
                s_deg0[u]++;
                p_e0[p_off + lp] = le0;
                p_e1[p_off + lp] = le1;
            }
        }
    }
    __syncthreads();

    for (int lp = threadIdx.x; lp < p_cnt; lp += blockDim.x) {
        const int ki = p_kidx[p_off + lp];
        const int bk = (ki < 0) ? spK : ki;
        const int np = atomicAdd(&s_bpos[bk], 1);
        new_p_of_old[p_off + lp] = np;
        old_p_of_new[p_off + np] = lp;
    }
}

// Wave 31: 1-warp-per-atom shared-memory (5 KB, zero local-memory spill) stable rank-count sort (~10x faster!).
template <typename T>
__global__ void __launch_bounds__(256) k_ksort_edges_gpu(
    int nf, const int* __restrict__ node_off, const int* __restrict__ edge_off, const int* __restrict__ pair_off,
    const int* __restrict__ rptr, const int* __restrict__ snd, const int* __restrict__ rcv,
    const int* __restrict__ pidx, const T* __restrict__ vec,
    const int* __restrict__ p_kidx, const float* __restrict__ p_us, const int* __restrict__ p_e1,
    const int* __restrict__ new_p_of_old,
    int* __restrict__ tmp_oe, uint32_t* __restrict__ tmp_key,
    int* __restrict__ ks_rptr0, int* __restrict__ new_e_of_old,
    int* __restrict__ ks_snd, int* __restrict__ ks_rcv, int* __restrict__ ks_pidx,
    int* __restrict__ ks_eps, int* __restrict__ ks_ek, float* __restrict__ ks_eu, T* __restrict__ ks_vec,
    int4* __restrict__ ks_emeta = nullptr) {
    (void)tmp_oe;
    constexpr int MAX_DEG_SM = 160;
    constexpr int NW = 8;
    __shared__ uint32_t s_key[NW][MAX_DEG_SM];
    const int fi = blockIdx.x;
    if (fi >= nf) return;
    const int n_off = node_off[fi], n_cnt = node_off[fi + 1] - n_off;
    const int e_off = edge_off[fi];
    const int p_off = pair_off[fi];
    const int* rptr_f = rptr + n_off + fi;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    for (int ln = warp; ln < n_cnt; ln += NW) {
        const int eb = rptr_f[ln], ee = rptr_f[ln + 1], deg = ee - eb;
        uint32_t* keys = (deg <= MAX_DEG_SM) ? s_key[warp] : (tmp_key + e_off + eb);
        int loc_s0 = 0;
        for (int i = lane; i < deg; i += 32) {
            const int le = eb + i;
            const int op = pidx[e_off + le];
            const int ki = p_kidx[p_off + op];
            const uint32_t grp = (ki < 0) ? 2u : ((le == p_e1[p_off + op]) ? 1u : 0u);
            if (grp == 0u) ++loc_s0;
            keys[i] = (grp << 28) | ((uint32_t)new_p_of_old[p_off + op] & 0x0FFFFFFFu);
        }
        for (int o = 16; o > 0; o >>= 1) loc_s0 += __shfl_xor_sync(0xffffffffu, loc_s0, o);
        if (lane == 0) ks_rptr0[n_off + ln] = eb + loc_s0;
        __syncwarp();

        for (int i = lane; i < deg; i += 32) {
            const uint32_t ki = keys[i];
            int rank = 0;
            for (int j = 0; j < deg; ++j) {
                const uint32_t kj = keys[j];
                if (kj < ki || (kj == ki && j < i)) ++rank;
            }
            const int ne = eb + rank;
            const int le = eb + i;
            new_e_of_old[e_off + le] = ne;
            const int op = pidx[e_off + le];
            const int np = new_p_of_old[p_off + op];
            const int side = (le == p_e1[p_off + op]) ? 1 : 0;
            const int snd_v = snd[e_off + le];
            const int ki_v = p_kidx[p_off + op];
            const float eu_v = p_us[p_off + op];
            ks_snd[e_off + ne] = snd_v;
            ks_rcv[e_off + ne] = rcv[e_off + le];
            ks_pidx[e_off + ne] = np;
            ks_eps[e_off + ne] = (np << 1) | side;
            ks_ek[e_off + ne] = ki_v;
            ks_eu[e_off + ne] = eu_v;
            if (ks_emeta) ks_emeta[e_off + ne] = make_int4((np << 1) | side, snd_v, ki_v, __float_as_int(eu_v));
            ks_vec[3L * (e_off + ne) + 0] = vec[3L * (e_off + le) + 0];
            ks_vec[3L * (e_off + ne) + 1] = vec[3L * (e_off + le) + 1];
            ks_vec[3L * (e_off + ne) + 2] = vec[3L * (e_off + le) + 2];
        }
        __syncwarp();
    }
}

template <typename T>
__global__ void k_ksort_scatter_pairs_gpu(
    int nf, const int* __restrict__ edge_off, const int* __restrict__ pair_off,
    const int* __restrict__ old_p_of_new, const int* __restrict__ p_e0, const int* __restrict__ p_e1,
    const int* __restrict__ new_e_of_old, int* __restrict__ ks_pe0, int* __restrict__ ks_pe1,
    const T* __restrict__ vec = nullptr, const int* __restrict__ snd = nullptr, const int* __restrict__ rcv = nullptr,
    float4* __restrict__ ks_p_vec4 = nullptr, int2* __restrict__ ks_p_sr = nullptr) {
    const int fi = blockIdx.x;
    if (fi >= nf) return;
    const int e_off = edge_off[fi];
    const int p_off = pair_off[fi], p_cnt = pair_off[fi + 1] - p_off;
    for (int np = threadIdx.x; np < p_cnt; np += blockDim.x) {
        const int old_p = old_p_of_new[p_off + np];
        const int le0 = p_e0[p_off + old_p];
        ks_pe0[p_off + np] = new_e_of_old[e_off + le0];
        ks_pe1[p_off + np] = new_e_of_old[e_off + p_e1[p_off + old_p]];
        if (ks_p_vec4 && vec) {
            const long eo = (long)e_off + le0;
            ks_p_vec4[p_off + np] = make_float4((float)vec[3L * eo + 0], (float)vec[3L * eo + 1],
                                                (float)vec[3L * eo + 2], 0.f);
        }
        if (ks_p_sr && snd && rcv) {
            const long eo = (long)e_off + le0;
            ks_p_sr[p_off + np] = make_int2(snd[eo], rcv[eo]);
        }
    }
}

// ============================================================================ engine
struct Base {
    std::string err;
    virtual ~Base() = default;
    virtual void set_params(void* const* p) = 0;
    virtual void set_frames(const LstFrames* f) = 0;
    virtual void set_node_order(const int32_t* order) = 0;
    virtual void set_node_energy(const void* ei_ref, double weight) = 0;
    virtual void set_node_energy_steps(const double* w_ei_steps, int n_steps, cudaStream_t s) = 0;
    virtual void set_optim(const LstOptim* o) = 0;
    virtual void set_muon(const LstMuonBlock* blocks, int n, double momentum, int nesterov, double scale) = 0;
    virtual void set_muon_ns(int steps, int polish) = 0;
    virtual void set_muon_norm(double beta) = 0;
    virtual void set_cautious(int on) = 0;
    virtual void set_muon_wd(double wd_mult) = 0;
    virtual void set_soap(double b1, double b2, double eps, int every, int max_dim, double mult) = 0;
    virtual void set_math(int gemm, int mlp) = 0;
    virtual void begin_epoch(const LstBatches* b, const double* lr, const double* lw, const double* ema, bool per_step,
                             cudaStream_t s) = 0;
    virtual void run(int n, cudaStream_t s) = 0;
    virtual void losses(double* out, cudaStream_t s) = 0;
    virtual void debug_step(int batch, int update, cudaStream_t s) = 0;
    virtual void evaluate(const int* ids, int n, void* e, void* f, void* s, cudaStream_t st) = 0;
    virtual void set_eval_cache(int on) = 0;
    virtual long stage_size(const std::string& n) = 0;
    virtual void stage_copy(const std::string& n, void* dst, cudaStream_t s) = 0;
    virtual const void* stage_ptr(const std::string& n) = 0;
};

template <typename T>
struct Engine : Base {
    LstModelDesc md;
    LstShape sh;
    int dev;
    int S, C, H, NB, O, hid, K, ZW, SC;
    int N, E, P, G1;
    cudaStream_t st = nullptr;
    cudaStream_t st_main = nullptr;
    cudaEvent_t ev_in = nullptr, ev_out = nullptr;
    cudaEvent_t ev_geo_go = nullptr, ev_geo_ready = nullptr;
    cublasHandle_t cb = nullptr;
    void* cb_ws = nullptr;
    size_t cb_ws_bytes = 32u << 20;
    int gemm_log = -1;
    // concurrent branches: side stream s2 with its own cuBLAS handle; st names the current stream
    cudaStream_t s2 = nullptr;
    cublasHandle_t cb2 = nullptr;
    void* cb2_ws = nullptr;
    cudaEvent_t evf[6] = {};
    bool wg_side = true;            // LST_WG_SIDE (default on, =0 off): weight-gradient GEMMs on s3
    cudaStream_t s3 = nullptr;
    cudaStream_t s_cond = nullptr;
    cublasHandle_t cb3 = nullptr;
    void* cb3_ws = nullptr;
    cudaEvent_t evw[32] = {};
    int nevw = 0;
    cudaStream_t wg_ms = nullptr;
    int par_env = 3;   // LST_PAR bits: 1 forward MLP 1 on the side, 2 backward slots + MLP 1 (default 3, =0 off)
    int par_mask = 3;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t gexec = nullptr;
    cudaGraph_t graph_fast = nullptr;
    cudaGraphExec_t gexec_fast = nullptr;
    cudaGraph_t graph_cycle = nullptr;
    cudaGraphExec_t gexec_cycle = nullptr;
    int graph_cycle_len = 0;
    long opt_step_host = -1;
    bool opt_steps_aligned = false;
    std::vector<void*> allocs;
    std::map<std::string, std::pair<T*, long>> stages;

    // inputs
    T* raw[LST_N_PARAMS] = {};
    long psize[LST_N_PARAMS];
    long poff[LST_N_PARAMS];
    long ptotal = 0;
    FramesT<T> frames{};
    bool have_frames = false, have_params = false, have_optim = false;
    LstOptim opt{};
    // device scalars / small arrays
    int* d_step = nullptr;
    Topo topo{};
    int* d_fids = nullptr;
    int* d_estep = nullptr;     // lst_evaluate: its own step index (always 0) and frame ids [B]
    bool force_order = true;    // LST_FORCE_ORDER=0: k_force_node in index order, not the frames' (lst_set_node_order)
    int* d_efids = nullptr;
    double* d_lr = nullptr;           // [cap_steps, 16] learning rate of each step and group
    std::vector<double> h_lr;
    std::vector<T> h_lw;
    double* d_wd = nullptr;
    T* d_lw = nullptr;
    double* d_ema = nullptr;
    T* d_losses = nullptr;
    int cap_steps = 0;
    int n_epoch = 0;
    double* d_norm2 = nullptr;
    double* d_urms = nullptr;          // [LST_N_PARAMS] StableAdamW sums
    double* d_caut = nullptr;          // [LST_N_PARAMS] cautious Adam: kept components per tensor
    bool adam_cautious = false;
    PDesc<T>* d_pdesc = nullptr;
    float** d_steps = nullptr;
    T* d_osc = nullptr;
    bool have_ema = false;
    // Muon
    std::vector<PDesc<T>> h_pdesc;
    int muon_flag[LST_N_PARAMS] = {};
    MuonMat* d_mmats = nullptr;
    int n_mmats = 0;
    T* d_mdir = nullptr;
    T* d_mws = nullptr;
    T* d_mnv = nullptr;          // NorMuon row state
    long mnv_size = 0;
    double muon_nbeta = -1.0;
    double muon_mu = 0.95, muon_scale = 0.2;
    int muon_nesterov = 1;
    int muon_steps = 5, muon_polish = 0;
    bool muon_fast = false;   // FP32 and every matrix's smaller side <= 64: k_muon_ns_f (LST_MUON_TILED=0: off)
    std::vector<MuonMat> h_mmats;
    // SOAP on the Muon blocks (lst_set_soap)
    bool soap_on = false;
    SoapMat* d_smats = nullptr;
    int2* d_sunits = nullptr;
    int2* d_seunits = nullptr;
    int n_seunits = 0;
    bool soap_eig = false;     // LST_SOAP_EIG=1: refresh as k_soap_eig (per factor) + k_soap_ap (chunked update)
    int n_sunits = 0, n_soap = 0;     // units of k_soap_a; SOAP matrices (k_soap_b; > n_mmats when blocked)
    T* d_sws = nullptr;
    __nv_bfloat16* d_sqbf = nullptr;  // Wave 21: BF16 shadow copies of Qm / Qn (2x less DRAM read every step!)
    double soap_b1 = 0.95, soap_b2 = 0.95, soap_eps = 1e-8, soap_mult = 1.0, soap_tol = 1e-7;
    double muon_wd = 0.0;   // lst_set_muon_wd: muon_lr_mult * decoupled weight decay of the Muon / SOAP blocks
    int soap_every = 10;
    int soap_sweeps = 30;     // warm refreshes' Jacobi sweep cap (LST_SOAP_SWEEPS)
    int soap_jfuse = 1;       // LST_SOAP_JFUSE (default on, =0 off): one pass per Jacobi round (sp::jacobi_f)
    bool soap_smem_on = false;   // LST_SOAP_SMEM=1: shared-memory k_soap_a_smem + fused EMA
    bool soap_smem_ok = false;   // every unit has m <= 64 and nc <= 64
    bool soap_ema_fuse = false;  // SOAP blocks cover 100% of muon_flag params -> skip k_muon_ema
    size_t soap_smem = 0;
    size_t soap_smem_a = 0;
    size_t soap_smem_b = 0;
    T* d_bw = nullptr;

    BatchT<T> b{};
    // views and derived weights
    T *emb, *wup0, *wl0, *ws0, *wl1, *W0, *W1, *wp0, *Wcat, *wp1, *wro1, *wro2, *T0, *M0, *Q, *ones;
    T* mlpW[2][4];
    // edges
    T *Y, *JY, *EF, *DEF, *U, *UP, *xe;
    T *z1[2], *z2[2], *z3[2], *h1[2], *h2[2], *h3[2], *out[2], *gout[2];
    T *gh, *gz;
    // nodes
    T *x0, *A0, *tmp, *B0, *f0, *z, *z0, *x1, *sc, *A1, *B1, *f1, *f2, *u, *tt, *ne, *Eg;
    T *ub, *f2b, *f1b, *B1b, *Gn1, *x1b, *zb, *f0b, *B0b, *Gn0, *se, *gy, *g, *F, *mv, *Sg;
    T *Eb, *Enb = nullptr, *Fb, *Ms;
    bool ei_on = false;
    bool ei_steps_set = false;
    double ei_weight = 0.0;
    double* d_wei = nullptr;
    int cap_wei = 0;
    T *sb, *gyb, *Yw, *Y3P = nullptr, *Gs0, *Gs1, *Xd0, *Xd1, *B0bs, *M0s, *B0s, *f0bs, *Wp0, *Wp1, *zbs, *Qs, *x1bs, *scbs;
    T *B1bs, *wl1s, *B1s, *f1bs, *f2bs, *wp1s, *ubs, *wro1s, *tbs, *us, *wro2s, *qn, *f2s, *f1s, *A1s, *zs, *f0s;
    T *A0s, *T0s;
    // view gradients and the raw-gradient buffer
    T *gemb, *gwup0, *gwl0, *gws0, *gwl1, *gW0, *gW1, *gwp0, *gWcat, *gwp1, *gwro1;
    T* gmlp[2][4];
    T* graw;
    // fused radial MLP kernels (FP32, NB 8, H 64, O 256); LST_MLP_FUSED=0 keeps the cuBLAS path
    bool fused_mlp = false;
    bool force_2pass = false;   // LST_FORCE_2PASS=1: k_force_node per layer (two passes; REJECTED: slower)
    int fwpb = 8;               // LST_FORCE_WPB: k_force_node warps (nodes) per block
    int slot_npb = 0;           // LST_SLOT_NPB: k_slot nodes per block (0: 256 / C)
    bool seg_gemm = true;       // LST_SEG_GEMM (default on, =0 off): per_l in one launch (segg::k_seg_gemm)
    int force_es = 4;           // LST_FORCE_ES=1|2|4 (default 4): k_force_node warps per atom (edges split)
    bool mlp_dual = false;      // LST_MLP_DUAL=1: both MLP backwards in one k_mlp_bwd launch
    int slot_es = 2;            // LST_SLOT_ES=1|2|4 (default 2): k_slot edges split over 2 or 4 thread groups per node
    bool slot_pp = false;       // LST_SLOT_PP=1: k_slot_pp over each node's own pairs (pptr from k_pair_ptr)
    bool fpair_warp = true;     // LST_FPAIR_WARP (default on, =0 off): k_forces_pair one warp per atom
    bool multi_gemm = true;     // LST_MULTI_GEMM (default on, =0 off): weight-only products in one launch each (mgemm)
    bool no_deadw = true;       // LST_NO_DEADW (default on, =0 off): no gy / se writes in k_force_node outside debug_step
    bool zero_fused = true;     // LST_ZERO_FUSED (default on, =0 off): one zeroing launch per step instead of nine memset nodes
    int force_jyw = 0;          // LST_FORCE_JYW=1: JY^T gy over 16 lanes in k_force_node
    bool in_debug = false;
    int conv_es = 2;            // LST_CONV_ES=1|2|4 (default 2): k_conv edges split over 2 or 4 thread groups per node
    bool mlp_bwd2 = false;   // LST_MLP_BWD2=1: the 512-thread k_mlp_bwd2 (REJECTED: slower than k_mlp_bwd)
    int mlp_bwd2_grid = 1;
    bool mlp_split = false;  // LST_MLP_SPLIT=1: gh3 and dW4 in k_mlp_top (no gain on the 5060 Ti)
    bool mlp_pipe = true;    // LST_MLP_PIPE=0 turns off k_mlp_bwdp (gh3 first, cp.async stages)
    bool mlp_noz1 = false;   // LST_MLP_NOZ1=1: k_mlp_bwdp_noz1 (W1 in smem, 3 reg arrays, swizzled gh3 stages)
    bool mlp_fwd_swz = false;   // LST_MLP_FWD_SWZ=1: k_mlp_fwd_swz (XOR-swizzled As/Ws, float4 load_x)
    bool vg_early = false;      // LST_VG_EARLY=1: queue T0s + view_grads on s3 before slot<false>
    bool prod_direct = true;    // LST_PROD_DIRECT=0 off (default on): k_prod_rev{1,2}_dir accumulate directly into gW0/gW1
    bool pair_basis = true;     // LST_PAIR_BASIS=0 off (default on): pair-indexed Y/Yw/gyb + matrix-free JY (sh16_pair.cuh)
    bool mlp0_first = true;     // LST_MLP0_FIRST=0 off (default on): launch mlp_fwd(0) before side-stream mlp_fwd(1)
    bool m0_species = true;     // LST_M0_SPECIES=0 off (default on): species-sliced M0 GEMMs + wgrads
    bool wgrad_bal = true;      // LST_WGRAD_BAL=0 off (default on): balanced (m,s)/(m) wgrads for M0s & wl1s
    bool node_major = true;     // LST_NODE_MAJOR=0 off (default on): [N, 16, C] node-major layout for 16-comp tensors
    bool conv_order = false;    // LST_CONV_ORDER=1: spatial atom order (b.norder) in k_conv_pair & k_slot_pair
    bool fast_mgemm = false;    // LST_FAST_MGEMM=1: 64-thread 4x4 register-blocked k_multi_gemm_f
    bool compact_z = true;      // LST_COMPACT_Z=0 off (default on): species-aware Q/z + single-launch small node GEMMs/wgrads
    int mlp_spline = 0;         // LST_MLP_SPLINE=K (e.g. 256|512): 1D cubic-Hermite knot-grid radial MLP
    bool mlp_spline_fused = true;  // LST_MLP_SPLINE_FUSED=0 off (default on when spline): fuse spline into conv/force/slot
    bool mlp_spline_small = true;  // LST_MLP_SPLINE_SMALL=0 off (default on when spline): TP=8 fwd / TP=16 bwd knot MLP
    int mlp_spline_poly = 0;       // LST_SPLINE_POLY=0|1|2 (0: Hermite; 1: fwd C_k only; 2: fwd C_k + bwd gC_k; default 2 on cc>=10, 0 else)
    bool conv_sender_dx = true;    // LST_CONV_SENDER_DX=0 off (default on when fused): sender-owned DX (no G[s] gather)
    bool sdx_es4 = true;           // LST_SDX_ES4=0 off (default on): ES=4 for zero-smem DG=0,DX=1,SDX=1 conv calls
    bool slot_no_gather = false;   // LST_SLOT_NO_GATHER=1: receiver-only slot (rejected: 2x atomics)
    bool slot_bal = false;         // LST_SLOT_BAL=1: balanced tournament pair ownership (rejected: breaks L1 locality)
    bool slot_fuse_xd = true;      // LST_SLOT_FUSE_XD=0 off (default on when fused): fuse Xd1 & Xd0 convs into slot_sp
    bool force_64t = false;        // LST_FORCE_64T=0|1 (default 1 on cc>=10, 0 else): 64-thread (CPL=1) k_force_node_pair_sp64
    bool prod_spc = false;         // LST_PROD_SPC=1: species-grouped k_prod_rev{1,2}_dir_spc (rejected at b4: reg spill)
    bool prod_dedup = false;       // LST_PROD_DEDUP=0|1 (default 1 for float, 0 for double): 28->13 deduplicated product basis
    bool spline_half = false;      // LST_SPLINE_HALF=0|1: half4 (uint2, 8B) packed cubic C_k (2x less L2 traffic)
    bool spline_rlo = true;        // LST_SPLINE_RLO=0|1 (default 1): auto-shift spline knots to [r_lo, r_max]
    int mlp_tiny = 0;              // LST_MLP_TINY=0|1|2 (1: sp_t<2/4/8>+fused c_to_rt; 2: sp4_fused+fused c_to_rt)
    bool wgrad_opt = true;         // LST_WGRAD_OPT=0|1 (default 1): dynamic per-species split-K M0s + wave-aligned wl1s
    bool soap_tf32 = false;        // LST_SOAP_TF32=0|1 (default 1 for float, 0 double): TF32 MMA in SOAP smem GEMMs
    bool force_pair = true;        // LST_FORCE_PAIR=0|1 (default 1): Newton's-3rd-law pair-owned force kernel (deletes g + k_forces_pair_w)
    int force_pair_es = 2;         // LST_FORCE_PAIR_ES=1|2|4 (default 2): edge splits per node for k_force_pair_sp*
    bool alias_16 = true;          // LST_ALIAS_16=0|1 (default 1): alias {B0bs, B1bs, A1s, A0s} -> 1 buffer (-15.9 MB at b4)
    bool slot_scalar = true;       // LST_SLOT_SCALAR=0|1 (default 0): per-degree scalar streaming k_slot_xd_sp_sc
    int act_half = 0;              // LST_ACT_HALF=0|1|2 (0: FP32, 1: BF16, 2: FP16) for [N,16,64] tensors + TC seg_gemm/wgrad
    bool edge_ksort = true;        // LST_EDGE_KSORT=0|1 (default 1): pre-sort frame pairs/edges by knot + emeta + prev_c cache
    bool slot_kb = false;          // LST_SLOT_KB=0|1 (default 0, rejected: 2x G/A L2 traffic vs node-owned)
    bool force_kb = false;         // LST_FORCE_KB=0|1 (default 0, rejected: 2x G0/G1 L2 traffic vs node-owned)
    int kb_tile = 64;              // LST_KB_TILE: max pairs per knot tile (default 32 at N<=512, 64 else)
    int edge_h2 = 5;               // LST_EDGE_H2 bitmask (default 5): 1=conv_h2, 2=slot_h2 (opt-in), 4=force_h2 (1 warp=64 ch!)
    int slot_opt = 7;              // LST_SLOT_OPT=0..8 (default 7): 7=ES2+s_ck+batch, 8=1-warp H2 fast
    bool pair_bal = false;         // LST_PAIR_BAL=0|1 (default 0): greedy 50/50 canonical pair orientation (0 beats 1 by 30-50 us!)
    bool mlp_sp4 = false;          // LST_MLP_SP4=0|1 (default 0, rejected: 2-warp occ 4.18% + 33-way atomics regressed +54 us b4)
    bool mlp_sp8 = true;           // LST_MLP_SP8=0|1 (default 1): 8-warp TP=8 k_mlp_bwd_sp8 (no k_mlp_wt, keeps fast cutlass dW4)
    bool nops_opt = true;          // LST_NOPS_OPT=0|1 (default 1): k_q_bwd_c32, k_q_wgrad_bal, k_wgrad_pair_64x16, k_wro2s_par, k_multi_gemm_v2
    int soap_opt = 1;              // LST_SOAP_OPT=0|1|2 (default 1): 1=vec4+fused sym Lm/Rn+fp32-jacobi-chk, 2=+algebraic-jacobi
    bool prod_1bar = true;         // LST_PROD_1BAR=0|1 (default 1): 1-barrier 13KB-smem prod_reduce_gw_u (rev2 only)
    bool edge_norder = false;      // LST_EDGE_NORDER=0|1 (default 1 on sm_8x Ada, 0 on sm_120 5060 Ti)
    int prod_h2 = 1;               // LST_PROD_H2=0|1|2 (default 1: rev1_h2 only; 2: both rev1_h2+rev2_h2)
    int slot_h2_es = 2;            // LST_SLOT_H2_ES=2|4 (default 2)
    int conv_h2_es_fwd = 2;        // LST_CONV_H2_ES=1|2|4 (default arch+N aware)
    int conv_h2_es_rev0 = 4;       // Wave 40c: REV2 L0 (T0S_DIR, 0 global atomics) default ES=4
    int conv_h2_es_rev1 = 2;       // Wave 40c: REV2 L1 (XB, OX atomics) default ES=2
    bool segg_v2 = true;           // LST_SEGG_V2=0|1 (default 1): multi-tile ldmatrix seg_gemm + degree-grouped TK=64 wgrad
    int segg_mt = 1;               // LST_SEGG_MT=1|2|4 (default 2 at N>=640, 1 else)
    bool mlp_tc2 = false;          // LST_MLP_TC2=0|1 (default 0: FP32 radial MLP wins +0.45 meV/A on soap40!)
    int soap_qr = 2;               // LST_SOAP_QR=0|1|2 (default 2 for float): 0=Jacobi, 1=sorted MGS QR, 2=Lie-Jacobi+NS
    bool soap_tc = true;           // LST_SOAP_TC=0|1 (default 1 for float): 36.8KB Qm/Qn-cached BF16 TC k_soap_a/ap (0 spills!)
    bool qwgrad_dyn = true;        // LST_QWGRAD_DYN=0|1 (default 1): stoichiometry-balanced k_q_wgrad_dyn using spc_wsk
    bool prod_rev2_spc = false;    // LST_PROD_REV2_SPC=1 (default 0: k_prod_rev2_dir_u beats it on 5060 Ti!)
    int start_par = 1;             // LST_START_PAR=0|1|2 (Wave 35 default 1: SP=2 caused 66 us gap on 5060 Ti & +41 ms on Ada)
    bool vg_split = true;          // LST_VG_SPLIT=0|1 (default 1): launch M0s/Qs view_grads on s3 during k_slot layer 0
    int mid_defer = 2;             // Wave 34/35 (LST_MID_DEFER=0|1|2, default 2): 2 launches k_q_wgrad_dyn right after k_q_bwd_c32!
    bool mgemm_sk = true;          // Wave 34 (LST_MGEMM_SK=0|1): Split-K + LPT ordering in d_mg_vg_early (kills K=257 straggler)
    bool start_defer = false;      // Wave 35/36 (LST_START_DEFER=0|1, default 0: regressed +34ms Ada / +35ms 5060 Ti)
    bool early_t0 = true;          // Wave 36 (LST_EARLY_T0=0|1): signal e_t0_ready after 1-tile T0, hide k_x0 on s3
    int ro_fuse = 2;               // Wave 35/36 (LST_RO_FUSE=0|1|2, default 2): 1=readout, 2=super-fused wp1+readout+wp1^T
    bool gf_prep = true;           // Wave 35 (LST_GF_PREP=0|1): fuse k_adam_prep into last CTA of k_grad_finalize_sumsq
    bool gf_flat = true;           // Wave 36 (LST_GF_FLAT=0|1): flattened uniform-tile k_grad_finalize_sumsq_flat
    int gf_persist = 2;            // Wave 37/38 (LST_GF_PERSIST=0|1|2, default 2): persistent SM-sized grid (2*nsm CTAs)
    bool gf_f32 = true;            // Wave 39 (LST_GF_F32=0|1, default 1): FP32 per-thread/warp sumsq in grad_finalize (kills 1/64 FP64!)
    bool loss_1blk = false;        // Wave 36/37 (LST_LOSS_1BLK=0|1, default 0: regressed +2.5-2.9 us on both GPUs)
    bool loss_2blk = true;         // Wave 38 (LST_LOSS_2BLK=0|1, default 1): zero-atomic 2-CTA loss (E+S+F || HAS_EI)
    bool loss_f32 = true;          // Wave 39 (LST_LOSS_F32=0|1, default 1): pure-FP32 1-CTA split-warp loss (zero FP64/fence/atomics!)
    bool loss_ei_fast = true;      // Wave 37 (LST_LOSS_EI_FAST=0|1): dedicated HAS_EI CTA + FP32 warp shuffles in k_loss_par
    int soap_mrot = 2;             // Wave 36/37 (LST_SOAP_MROT=0|1|2, default 2): 1=Mrot cache, 2=zero-q.m on 9/10 non-QR steps!
    int soap_stat_every = 1;       // Wave 38 (LST_SOAP_STAT_EVERY=1|2|5): subsample SOAP Lm/Rn covariance updates with b2^K!
    bool soap_lpt = true;          // Wave 40 (LST_SOAP_LPT=0|1, default 1): LPT-sort SOAP units (heavy 64x64 first, tiny in tail!)
    bool adam_compact = true;      // Wave 39 (LST_ADAM_COMPACT=0|1, default 1): fused compact non-SOAP Adam (~4 CTAs vs 832!)
    bool start_fuse = false;       // Wave 39/40 (LST_START_FUSE=0|1, default 0: regressed +36ms Ada / +876ms 5060 Ti!)
    bool mlp_savez = true;         // Wave 37 (LST_MLP_SAVEZ=0|1): save z1..z3 (101 KB L2) in hidden fwd -> zero-recompute dual bwd!
    bool vg_merge = false;         // Wave 37/38 (LST_VG_MERGE=0|1, superseded by tail_prio)
    bool tail_prio = true;         // Wave 38/39 (LST_TAIL_PRIO=0|1, default arch-aware: 1 on >40 SM Ada, 0 on <=40 SM 5060 Ti!)
    int ro_na = 16;                // Wave 37/38 (LST_RO_NA=8|12|16, default arch-aware: 16 on <=40 SM, 8 on >40 SM)
    int prod_h2_w = 3;             // Wave 40c (LST_PROD_H2_W=0..3, default 3: gx=8 NW=4 won -34ms 5060 Ti / -17ms Ada!)
    int graph_pipe = 0;            // Wave 41 (LST_GRAPH_PIPE=0..4, default arch-aware: 2 on Ada sm_89, 0 on 5060 Ti sm_120)
    int graph_cyc_mult = 1;        // Wave 41 (LST_GRAPH_CYC_MULT=1..10, default 1): SOAP cycles per captured graph
    bool vg_qsplit = false;        // Wave 41 (LST_VG_QSPLIT=0|1, default arch-aware: 1 on sm_120, 0 on sm_89)
    int mlp_t2 = 0;                // Wave 42/43 (LST_MLP_T2=0..3, default arch-aware: 2 on Ada sm_89, 0 on 5060 Ti sm_120)
    int conv_npb1 = 15;            // Wave 43/44 (LST_CONV_NPB1=0..15): 1 atom/CTA bit0=FWD, bit1=REV1, bit2=REV2_L0, bit3=REV2_L1
    bool mq_after_mlp0 = false;    // Wave 43 (LST_MQ_AFTER_MLP0=0|1): delay d_mg_pv_mq until mlp_fwd(0) finishes
    bool fwd_t0s = false;          // Wave 44 (LST_FWD_T0S=0|1, default 0): 0 runs k_x0 in e_t0_ready..evf[4] gap -> faster T0S_DIR=0 FWD L0!
    int ecot_tpb = 64;             // Wave 38 (LST_ECOT_TPB=64|128|256, default 64: 143 CTAs vs 36 -> fills 48 SMs + hides LgSB)
    int* d_adam_idx = nullptr;     // Wave 39: compact device list of non-SOAP parameter indices
    int n_adam_idx = 0;
    bool q_split = true;           // Wave 36 (LST_Q_SPLIT=0|1): higher-occupancy k_q_fwd_c32 & k_q_bwd_r16 (60 CTAs vs 30)
    bool soap_2graph = true;       // Wave 35 (LST_SOAP_2GRAPH=0|1): host 2-graph SOAP dispatch (skips empty eig/ap on 9/10 steps)
    int gather_slim = 2;           // Wave 43 (LST_GATHER_SLIM=0|1|2, default 2): 1=slim, 2=fused coalesced k_gather_basis_fast!
    bool zip_ga = true;            // LST_ZIP_GA=0|1 (default 1): fused seg_gemm GA/G01 zip -> 2x fewer LDGs in slot_xd & force!
    bool conv_fact = true;         // LST_CONV_FACT=0|1 (default 1): 1-l-at-a-time factored conv_rev2 (-33..-40% DG FMAs, -36 regs!)
    bool wg_sched = false;         // LST_WG_SCHED=0|1 (default 0: caused Ada contention)
    bool wg_defer = false;         // LST_WG_DEFER=0|1 (default 0: won -30 us on Ada, regressed +40 us on 5060 Ti b4)
    int y3p = 3;                   // LST_Y3P=0|1|2|3 (default 3 on sm_8x+: zero-PRMT interleaved FP16 Y3P_H)
    bool force_hfma2 = false;      // LST_FORCE_HFMA2=0|1 (default 0: native __half2 HFMA2 in k_force_pair_h2)
    int force_dir = 0;             // LST_FORCE_DIR=0|1|2 (0=pair, 1=Wave23 fused dir, 2=Wave24 split L1+vjp / L0!)
    int force_dir_es = 0;          // LST_FORCE_DIR_ES=1|2|4 (0=auto: N<=512?2:1)
    bool conv_c64 = false;         // LST_CONV_C64=0|1 (Wave 24: 1 ch/thread k_conv_c64 + emeta_dir + prev_ck)
    bool t0s_dir = false;          // LST_T0S_DIR=0|1 (Wave 24: Layer 0 reg T0s accum -> 0 Xd0 atomics, skips k_t0s_spc!)
    bool scot = false;             // LST_SCOT=0|1 (Wave 24: dynamic per-step cotangent scale S_cot for FP16 Y3P_H)
    bool conv_opt = false;         // LST_CONV_OPT=0|1 (Wave 25: 1-warp k_conv_pair_h2_opt + s_ck + s_gn + emeta_dir)
    bool t0s_pair = false;         // LST_T0S_PAIR=0|1 (Wave 25: Layer 0 T0s smem/reg accum inside PAIR-folded conv+slot!)
    int force_opt = 0;             // LST_FORCE_OPT=0|1|2 (Wave 25: 1=s_gn fused pair, 2=PAIR-folded split L1+vjp / L0!)
    bool mlp_dual_sp = false;      // LST_MLP_DUAL_SP=0|1 (Wave 25: batched 2-layer FP32 k_mlp_fwd_sp2)
    int slot_hfma2 = 0;            // LST_SLOT_HFMA2=0|1 (Wave 27: native __half2 HFMA2 exact FP16 GA in k_slot_xd_opt_v3)
    bool slot_ldcg = false;        // LST_SLOT_LDCG=0|1 (Wave 29: L1-bypass __ldcg on GA[s] in k_slot_xd_opt_v3)
    int conv_hfma2 = 0;            // LST_CONV_HFMA2=0|1|2 (Wave 28: 1=HFMA2 DX + FP32 og, 2=HFMA2 both og & DX in conv_rev2)
    float shell_rcut[2][4] = {{-1.f, -1.f, -1.f, -1.f}, {-1.f, -1.f, -1.f, -1.f}};  // LST_SHELL_RCUT (Arch 1)
    float shell_width = 0.f;       // LST_SHELL_WIDTH (0 = p=5 from 0; >0 = flat-top C^2 quintic switch on [rc-sw, rc])
    bool shell_active = false;
    bool grad_fuse = true;         // LST_GRAD_FUSE=0|1 (default 1): fuse scale+uncat+cat+sumsq into 1 launch!
    bool soap_qbf = true;          // LST_SOAP_QBF=0|1 (default 1): BF16 shadow Qm/Qn (2x less DRAM read every step!)
    bool loss_par = true;          // LST_LOSS_PAR=0|1 (default 1): multi-block coalesced k_loss_par (~3 us vs 11 us)
    bool soap_coal = true;         // LST_SOAP_COAL=0|1 (default 1): coalesced float4 RMW + fused q.g->q.m in k_soap_a_smem_bf16
    bool soap_fast = true;         // Wave 45 (LST_SOAP_FAST=0|1, default 1): dedicated lean non-QR k_soap_a_fast_bf16
    bool segg_h16 = true;          // Wave 45 (LST_SEGG_H16=0|1, default 1): pre-pack FP16/BF16 M0_h/Wl1_h once per step
    int mgemm_v3 = 0;              // LST_MGEMM_V3=0|1|2 (default 0: k_multi_gemm_v2 beats v3/f4 on ZW=257)
    int mlp_spv2 = 1;              // LST_MLP_SPV2=0|1|2 (Wave 31: 1=TP4 Pm<=34, 2=TP4 Pm<=260)
    bool stream_prio = false;      // Wave 33/34 (LST_STREAM_PRIO=0|1, default 0: CUDA graphs ignore stream prio)
    bool soap_cond_en = false;     // Wave 33/34 (LST_SOAP_COND=0|1, default 0: regressed +7/+140 us on 5060 Ti)
    bool mlpf_attrs_ready = false; // Wave 33: lazy cudaFuncSetAttribute for non-spline mlpf/mlpt fallback
    bool eval_cache = false;       // Wave 32: frozen-weights eval view/MLP cache + CUDA graph replay
    bool eval_views_valid = false;
    cudaGraphExec_t eval_e_exec = nullptr;
    cudaGraphExec_t eval_fs_exec = nullptr;
    void* eval_e_dst = nullptr;
    void* eval_f_dst = nullptr;
    void* eval_s_dst = nullptr;
    uint32_t* GA_buf = nullptr;    // reusable [N, 16, 64] uint32_t interleaved (G,A)/(G0,G1) buffer
    uint16_t* M0_h = nullptr;      // Wave 45: pre-packed FP16/BF16 M0 [2, 4*S, 4096] (plane 0=!TW, plane 1=TW)
    uint16_t* Wl1_h = nullptr;     // Wave 45: pre-packed FP16/BF16 wl1 [2, 4, 4096] (plane 0=!TW, plane 1=TW)
    double* d_loss_acc = nullptr;  // [4] double accumulator for k_loss_par (E, F, S, Ei)
    int* d_loss_done = nullptr;    // [1] block completion counter for k_loss_par
    int* d_gf_done = nullptr;      // [1] block completion counter for fused k_grad_finalize_sumsq + adam_prep (Wave 35)
    unsigned int* d_fmax = nullptr;// [1] max|Fb| uint bits for dynamic S_cot (Wave 24)
    float* d_scot = nullptr;       // [2] {S_cot, 1/S_cot} for dynamic FP16 cotangent scaling (Wave 24)
    bool ks_ready = false;
    int ks_max_tiles = 0;
    Topo ks_topo{};
    FramesT<T> ks_frames{};
    std::vector<void*> ks_allocs;
    int* d_ks_toff = nullptr;
    int4* d_ks_tiles = nullptr;
    double sp_r_lo = 0.0;
    float* d_rmin = nullptr;
    int spline_M = 0;           // mlp_spline + 1 knots
    T *EF_k = nullptr, *DEF_k = nullptr, *xe_k = nullptr;
    int* pe0_k = nullptr;
    T *out_k[2] = {}, *gout_k[2] = {}, *h3_k[2] = {}, *C_k[2] = {}, *gC_k[2] = {};
    T *z1_k[2] = {}, *z2_k[2] = {}, *z3_k[2] = {}, *h1_k[2] = {}, *h2_k[2] = {}, *gh_k = nullptr, *gz_k = nullptr;
    int mlp_l2pf = 0;        // LST_MLP_L2PF=1: k_mlp_bwdp prefetches the next tile's gout into L2
    // split-K A^T B for small outputs over long K (weight gradients); LST_GEMM_TN=1 (ties cuBLAS: memory-bound)
    bool own_tn = false;
    int mlp_tc = -1, mlp_tc_grid = 1;   // tensor-core radial MLPs: mlpt::TF32 / BF16 (LST_MLP_TC, LST_MATH)
    bool mlp_tc_ready = false;          // their kernels set up at creation (lst_set_math may switch them back on)
    int nsm = 1;
    T* tn_ws = nullptr;
    long tn_cap = 0;
    T* mlpT[2] = {};
    int mlp_bwd_grid = 0;

    T* alloc(const char* name, long n) {
        void* p = nullptr;
        CK(cudaMalloc(&p, sizeof(T) * (size_t)std::max(1L, n)));
        CK(cudaMemset(p, 0, sizeof(T) * (size_t)std::max(1L, n)));
        allocs.push_back(p);
        if (name) stages[name] = {reinterpret_cast<T*>(p), n};
        return reinterpret_cast<T*>(p);
    }
    template <typename I>
    I* alloci(long n) {
        void* p = nullptr;
        CK(cudaMalloc(&p, sizeof(I) * (size_t)std::max(1L, n)));
        CK(cudaMemset(p, 0, sizeof(I) * (size_t)std::max(1L, n)));
        allocs.push_back(p);
        return reinterpret_cast<I*>(p);
    }

    Engine(const LstModelDesc& m, const LstShape& s, int device) : md(m), sh(s), dev(device) {
        CK(cudaSetDevice(dev));
        S = m.S; C = m.C; H = m.H; NB = m.NB; O = 4 * m.C; hid = m.hid; K = m.K;
        SC = S * C;
        ZW = 1 + C + SC;
        N = s.N; E = s.E; P = s.P; G1 = s.G1;
        if (K != POLY_K) throw Error("model has K = " + std::to_string(K) + " product polynomials, library " + std::to_string(POLY_K));
        if (C % 32 != 0 || C > 256 || 256 % C != 0) throw Error("channels must divide 256 and be a multiple of 32");
        if (P * 2 != E) throw Error("edges must be paired (E = 2 P)");
        {
            const char* esp = std::getenv("LST_STREAM_PRIO");
            stream_prio = (esp && esp[0] == '1');
            const char* esc = std::getenv("LST_SOAP_COND");
            soap_cond_en = (esc && esc[0] == '1');
            int prio_lo = 0, prio_hi = 0;
            if (stream_prio) CK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
            if (stream_prio && prio_hi != prio_lo)
                CK(cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, prio_hi));
            else
                CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            st_main = st;
            CK(cudaEventCreateWithFlags(&ev_in, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_out, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_geo_go, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_geo_ready, cudaEventDisableTiming));
            const char* ew = std::getenv("LST_CB_WS");
            cb_ws_bytes = ew ? (size_t)std::atoll(ew) : (32u << 20);
            const char* ep = std::getenv("LST_PAR");
            par_env = ep ? std::atoi(ep) : 3;
            if (stream_prio && prio_hi != prio_lo) {
                CK(cudaStreamCreateWithPriority(&s2, cudaStreamNonBlocking, prio_lo));
                CK(cudaStreamCreateWithPriority(&s_cond, cudaStreamNonBlocking, prio_hi));
            } else {
                CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
                CK(cudaStreamCreateWithFlags(&s_cond, cudaStreamNonBlocking));
            }
            for (auto& e : evf) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            const char* ew3 = std::getenv("LST_WG_SIDE");
            wg_side = !ew3 || ew3[0] != '0';
            if (wg_side) {
                if (stream_prio && prio_hi != prio_lo)
                    CK(cudaStreamCreateWithPriority(&s3, cudaStreamNonBlocking, prio_lo));
                else
                    CK(cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking));
                for (auto& e : evw) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            }
        }
        {
            const char* ev = std::getenv("LST_MATH");
            const std::string m = ev ? ev : "";
            g_f32_compute = m == "tf32" ? CUBLAS_COMPUTE_32F_FAST_TF32
                          : m == "bf16" ? CUBLAS_COMPUTE_32F_FAST_16BF : CUBLAS_COMPUTE_32F;
        }
        CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
        {
            const char* ev = std::getenv("LST_GEMM_TN");
            own_tn = ev && ev[0] == '1';
            const char* ef = std::getenv("LST_FORCE_2PASS");
            force_2pass = ef && ef[0] == '1';
            const char* ew = std::getenv("LST_FORCE_WPB");
            if (ew) fwpb = std::max(1, std::min(8, std::atoi(ew)));
            const char* es = std::getenv("LST_SLOT_NPB");
            if (es) slot_npb = std::max(0, std::atoi(es));
            const char* eq = std::getenv("LST_SEG_GEMM");
            seg_gemm = !eq || eq[0] != '0';
            const char* ec = std::getenv("LST_CONV_ES");
            conv_es = ec ? std::atoi(ec) : 2;
            const char* efe = std::getenv("LST_FORCE_ES");
            force_es = efe ? std::max(1, std::atoi(efe)) : 4;
            const char* emd = std::getenv("LST_MLP_DUAL");
            mlp_dual = emd && emd[0] == '1';
            const char* ese = std::getenv("LST_SLOT_ES");
            slot_es = ese ? std::atoi(ese) : 2;
            const char* epp = std::getenv("LST_SLOT_PP");
            slot_pp = epp && epp[0] == '1';
            const char* efp = std::getenv("LST_FPAIR_WARP");
            fpair_warp = !efp || efp[0] != '0';
            const char* em = std::getenv("LST_MULTI_GEMM");
            multi_gemm = !em || em[0] != '0';
            const char* e1 = std::getenv("LST_NO_DEADW");
            no_deadw = !e1 || e1[0] != '0';
            const char* e2 = std::getenv("LST_ZERO_FUSED");
            zero_fused = !e2 || e2[0] != '0';
            const char* e3 = std::getenv("LST_FORCE_JYW");
            force_jyw = e3 && e3[0] == '1' ? 1 : 0;
            const char* evg = std::getenv("LST_VG_EARLY");
            vg_early = evg && evg[0] == '1';
            const char* epd = std::getenv("LST_PROD_DIRECT");
            prod_direct = !(epd && epd[0] == '0');
            const char* epb = std::getenv("LST_PAIR_BASIS");
            pair_basis = !(epb && epb[0] == '0');
            const char* em0 = std::getenv("LST_MLP0_FIRST");
            mlp0_first = !(em0 && em0[0] == '0');
            const char* ems = std::getenv("LST_M0_SPECIES");
            m0_species = !(ems && ems[0] == '0') && C % segg::TM == 0 && S <= 16;
            const char* ewb = std::getenv("LST_WGRAD_BAL");
            wgrad_bal = !(ewb && ewb[0] == '0') && C % segg::TM == 0;
            const char* enm = std::getenv("LST_NODE_MAJOR");
            node_major = !(enm && enm[0] == '0') && m0_species && pair_basis && prod_direct && seg_gemm && C % segg::TM == 0 && S <= 16;
            const char* eco = std::getenv("LST_CONV_ORDER");
            conv_order = eco && eco[0] == '1';
            const char* efm = std::getenv("LST_FAST_MGEMM");
            fast_mgemm = efm && efm[0] == '1';
            const char* ecz = std::getenv("LST_COMPACT_Z");
            compact_z = !(ecz && ecz[0] == '0') && m0_species && C % 64 == 0 && hid % 4 == 0 && hid <= 64;
            const char* emsp = std::getenv("LST_MLP_SPLINE");
            if (emsp) {
                const int kv = std::atoi(emsp);
                if (kv >= 16 && kv <= 4096) mlp_spline = kv;
            }
            const char* emsf = std::getenv("LST_MLP_SPLINE_FUSED");
            mlp_spline_fused = !(emsf && emsf[0] == '0');
            const char* emss = std::getenv("LST_MLP_SPLINE_SMALL");
            mlp_spline_small = !(emss && emss[0] == '0');
            int cc_maj = 0;
            CK(cudaDeviceGetAttribute(&cc_maj, cudaDevAttrComputeCapabilityMajor, dev));
            const char* emspoly = std::getenv("LST_SPLINE_POLY");
            mlp_spline_poly = emspoly ? std::atoi(emspoly) : (cc_maj >= 8 ? 2 : 0);
            const char* esdx = std::getenv("LST_CONV_SENDER_DX");
            conv_sender_dx = !(esdx && esdx[0] == '0');
            const char* esdx4 = std::getenv("LST_SDX_ES4");
            sdx_es4 = !(esdx4 && esdx4[0] == '0');
            const char* esng = std::getenv("LST_SLOT_NO_GATHER");
            slot_no_gather = esng && esng[0] == '1';
            const char* esbal = std::getenv("LST_SLOT_BAL");
            slot_bal = esbal && esbal[0] == '1';
            const char* esfx = std::getenv("LST_SLOT_FUSE_XD");
            slot_fuse_xd = !(esfx && esfx[0] == '0');
            const char* ef64 = std::getenv("LST_FORCE_64T");
            force_64t = ef64 ? (ef64[0] == '1') : (cc_maj >= 10);
            const char* epsp = std::getenv("LST_PROD_SPC");
            prod_spc = epsp && epsp[0] == '1' && m0_species && C == 64;
            const char* epdd = std::getenv("LST_PROD_DEDUP");
            prod_dedup = epdd ? (epdd[0] == '1') : std::is_same<T, float>::value;
            const char* esph = std::getenv("LST_SPLINE_HALF");
            spline_half = esph && esph[0] == '1' && std::is_same<T, float>::value;
            const char* esprl = std::getenv("LST_SPLINE_RLO");
            spline_rlo = !(esprl && esprl[0] == '0');
            const char* emt = std::getenv("LST_MLP_TINY");
            mlp_tiny = emt ? std::atoi(emt) : 0;
            const char* ewgo = std::getenv("LST_WGRAD_OPT");
            wgrad_opt = !(ewgo && ewgo[0] == '0');
            const char* estf = std::getenv("LST_SOAP_TF32");
            soap_tf32 = estf && estf[0] == '1' && std::is_same<T, float>::value;
            const char* efp2 = std::getenv("LST_FORCE_PAIR");
            force_pair = !(efp2 && efp2[0] == '0');
            if (!ef64 && force_pair) force_64t = true;
            const char* efpes = std::getenv("LST_FORCE_PAIR_ES");
            force_pair_es = efpes ? std::max(1, std::atoi(efpes)) : (N <= 512 ? 4 : 2);
            const char* ea16 = std::getenv("LST_ALIAS_16");
            alias_16 = !(ea16 && ea16[0] == '0');
            const char* escs = std::getenv("LST_SLOT_SCALAR");
            slot_scalar = escs && escs[0] == '1';
            const char* eah = std::getenv("LST_ACT_HALF");
            if (eah && std::is_same<T, float>::value) {
                const int v = std::atoi(eah);
                if (v == 1 || v == 2) act_half = v;
            }
            const char* eeks = std::getenv("LST_EDGE_KSORT");
            edge_ksort = !(eeks && eeks[0] == '0');
            const char* eskb = std::getenv("LST_SLOT_KB");
            slot_kb = eskb && eskb[0] == '1';
            const char* efkb = std::getenv("LST_FORCE_KB");
            force_kb = efkb && efkb[0] == '1';
            const char* ekbt = std::getenv("LST_KB_TILE");
            kb_tile = ekbt ? std::max(8, std::atoi(ekbt)) : (N <= 512 ? 32 : 64);
            const char* eeh2 = std::getenv("LST_EDGE_H2");
            edge_h2 = eeh2 ? std::atoi(eeh2) : 5;
            const char* esopt = std::getenv("LST_SLOT_OPT");
            slot_opt = esopt ? std::atoi(esopt) : 7;
            const char* epbal = std::getenv("LST_PAIR_BAL");
            pair_bal = epbal && epbal[0] == '1';
            const char* emsp4 = std::getenv("LST_MLP_SP4");
            mlp_sp4 = emsp4 && emsp4[0] == '1';
            const char* emsp8 = std::getenv("LST_MLP_SP8");
            mlp_sp8 = !(emsp8 && emsp8[0] == '0');
            const char* eno = std::getenv("LST_NOPS_OPT");
            nops_opt = !(eno && eno[0] == '0');
            const char* espo = std::getenv("LST_SOAP_OPT");
            soap_opt = espo ? std::atoi(espo) : 1;
            const char* ep1b = std::getenv("LST_PROD_1BAR");
            prod_1bar = !(ep1b && ep1b[0] == '0');
            const char* eend = std::getenv("LST_EDGE_NORDER");
            edge_norder = eend ? (eend[0] == '1') : (cc_maj == 8);
            const char* eph2 = std::getenv("LST_PROD_H2");
            prod_h2 = eph2 ? std::atoi(eph2) : 1;
            const char* esh2 = std::getenv("LST_SLOT_H2_ES");
            slot_h2_es = (esh2 && std::atoi(esh2) == 4) ? 4 : 2;
            const char* ech2 = std::getenv("LST_CONV_H2_ES");
            if (ech2) {
                const int v = std::atoi(ech2);
                const int es = (v == 1 || v == 4) ? v : 2;
                conv_h2_es_fwd = es;
                conv_h2_es_rev0 = es;
                conv_h2_es_rev1 = es;
            } else {
                conv_h2_es_fwd = (N >= 640) ? 1 : 2;
                conv_h2_es_rev0 = (cc_maj >= 10 && N < 640) ? 4 : ((N >= 640 && cc_maj >= 10) ? 1 : 2);
                conv_h2_es_rev1 = (N >= 640 && cc_maj >= 10) ? 1 : 2;
            }
            const char* ech2f = std::getenv("LST_CONV_H2_ES_FWD");
            if (ech2f) {
                const int v = std::atoi(ech2f);
                conv_h2_es_fwd = (v == 1 || v == 4) ? v : 2;
            }
            const char* ech2r = std::getenv("LST_CONV_H2_ES_REV");
            if (ech2r) {
                const int v = std::atoi(ech2r);
                const int es = (v == 1 || v == 4) ? v : 2;
                conv_h2_es_rev0 = es;
                conv_h2_es_rev1 = es;
            }
            const char* ech2r0 = std::getenv("LST_CONV_H2_ES_REV0");
            if (ech2r0) {
                const int v = std::atoi(ech2r0);
                conv_h2_es_rev0 = (v == 1 || v == 4) ? v : 2;
            }
            const char* ech2r1 = std::getenv("LST_CONV_H2_ES_REV1");
            if (ech2r1) {
                const int v = std::atoi(ech2r1);
                conv_h2_es_rev1 = (v == 1 || v == 4) ? v : 2;
            }
            const char* esgv2 = std::getenv("LST_SEGG_V2");
            segg_v2 = !(esgv2 && esgv2[0] == '0');
            const char* esgmt = std::getenv("LST_SEGG_MT");
            segg_mt = esgmt ? std::max(1, std::min(8, std::atoi(esgmt))) : (N >= 640 ? 2 : 1);
            const char* emtc2 = std::getenv("LST_MLP_TC2");
            mlp_tc2 = (emtc2 && emtc2[0] == '1') && std::is_same<T, float>::value;
            const char* espqr = std::getenv("LST_SOAP_QR");
            soap_qr = std::is_same<T, float>::value ? (espqr ? std::atoi(espqr) : 2) : 0;
            const char* esptc = std::getenv("LST_SOAP_TC");
            soap_tc = !(esptc && esptc[0] == '0') && std::is_same<T, float>::value;
            const char* eqwd = std::getenv("LST_QWGRAD_DYN");
            qwgrad_dyn = !(eqwd && eqwd[0] == '0');
            const char* epr2 = std::getenv("LST_PROD_REV2_SPC");
            prod_rev2_spc = (epr2 && epr2[0] == '1');
            const char* esp = std::getenv("LST_START_PAR");
            start_par = esp ? std::max(0, std::min(2, std::atoi(esp))) : 1;
            const char* evgs = std::getenv("LST_VG_SPLIT");
            vg_split = !(evgs && evgs[0] == '0');
            const char* emdf = std::getenv("LST_MID_DEFER");
            mid_defer = emdf ? std::max(0, std::min(2, std::atoi(emdf))) : 2;
            const char* emsk = std::getenv("LST_MGEMM_SK");
            mgemm_sk = !(emsk && emsk[0] == '0');
            const char* esdf = std::getenv("LST_START_DEFER");
            start_defer = (esdf && esdf[0] == '1');
            const char* eet0 = std::getenv("LST_EARLY_T0");
            early_t0 = !(eet0 && eet0[0] == '0');
            const char* erof = std::getenv("LST_RO_FUSE");
            ro_fuse = erof ? std::max(0, std::min(2, std::atoi(erof))) : 2;
            const char* egfp = std::getenv("LST_GF_PREP");
            gf_prep = !(egfp && egfp[0] == '0');
            const char* egff = std::getenv("LST_GF_FLAT");
            gf_flat = !(egff && egff[0] == '0');
            const char* egfper = std::getenv("LST_GF_PERSIST");
            gf_persist = egfper ? std::max(0, std::min(2, std::atoi(egfper))) : 2;
            const char* egff32 = std::getenv("LST_GF_F32");
            gf_f32 = !(egff32 && egff32[0] == '0');
            const char* el1b = std::getenv("LST_LOSS_1BLK");
            loss_1blk = (el1b && el1b[0] == '1');
            const char* el2b = std::getenv("LST_LOSS_2BLK");
            loss_2blk = !(el2b && el2b[0] == '0');
            const char* elf32 = std::getenv("LST_LOSS_F32");
            loss_f32 = !(elf32 && elf32[0] == '0');
            const char* elef = std::getenv("LST_LOSS_EI_FAST");
            loss_ei_fast = !(elef && elef[0] == '0');
            const char* esmr = std::getenv("LST_SOAP_MROT");
            soap_mrot = esmr ? std::max(0, std::min(2, std::atoi(esmr))) : 2;
            const char* esse = std::getenv("LST_SOAP_STAT_EVERY");
            soap_stat_every = esse ? std::max(1, std::atoi(esse)) : 1;
            const char* eslpt = std::getenv("LST_SOAP_LPT");
            soap_lpt = !(eslpt && eslpt[0] == '0');
            const char* eacmp = std::getenv("LST_ADAM_COMPACT");
            adam_compact = !(eacmp && eacmp[0] == '0');
            const char* esfus = std::getenv("LST_START_FUSE");
            start_fuse = (esfus && esfus[0] == '1');
            const char* emsz = std::getenv("LST_MLP_SAVEZ");
            mlp_savez = !(emsz && emsz[0] == '0');
            const char* evgm = std::getenv("LST_VG_MERGE");
            vg_merge = (evgm && evgm[0] == '1');
            const char* etpr = std::getenv("LST_TAIL_PRIO");
            tail_prio = etpr ? (etpr[0] == '1') : (nsm > 40);
            const char* erona = std::getenv("LST_RO_NA");
            const int ro_na_def = (nsm <= 40) ? 16 : 8;
            ro_na = erona ? std::atoi(erona) : ro_na_def;
            if (ro_na != 8 && ro_na != 12 && ro_na != 16) ro_na = ro_na_def;
            const char* eph2w = std::getenv("LST_PROD_H2_W");
            prod_h2_w = eph2w ? std::max(0, std::min(3, std::atoi(eph2w))) : 3;
            const char* egpi = std::getenv("LST_GRAPH_PIPE");
            graph_pipe = egpi ? std::max(0, std::min(4, std::atoi(egpi))) : ((cc_maj < 10) ? 2 : 0);
            const char* egcm = std::getenv("LST_GRAPH_CYC_MULT");
            graph_cyc_mult = egcm ? std::max(1, std::min(10, std::atoi(egcm))) : 1;
            const char* evgq = std::getenv("LST_VG_QSPLIT");
            vg_qsplit = evgq ? (evgq[0] == '1') : (cc_maj >= 10);
            const char* emt2 = std::getenv("LST_MLP_T2");
            mlp_t2 = emt2 ? std::max(0, std::min(3, std::atoi(emt2))) : ((cc_maj < 10) ? 2 : 0);
            const char* ecnpb = std::getenv("LST_CONV_NPB1");
            conv_npb1 = ecnpb ? std::max(0, std::min(15, std::atoi(ecnpb))) : ((cc_maj >= 10) ? 15 : 1);
            const char* emqa = std::getenv("LST_MQ_AFTER_MLP0");
            mq_after_mlp0 = emqa && emqa[0] == '1';
            const char* eft0 = std::getenv("LST_FWD_T0S");
            fwd_t0s = eft0 && eft0[0] == '1';
            const char* eect = std::getenv("LST_ECOT_TPB");
            ecot_tpb = eect ? std::atoi(eect) : 64;
            if (ecot_tpb != 64 && ecot_tpb != 128 && ecot_tpb != 256) ecot_tpb = 64;
            const char* eqsp = std::getenv("LST_Q_SPLIT");
            q_split = !(eqsp && eqsp[0] == '0');
            const char* es2g = std::getenv("LST_SOAP_2GRAPH");
            soap_2graph = !(es2g && es2g[0] == '0');
            const char* egsl = std::getenv("LST_GATHER_SLIM");
            gather_slim = egsl ? std::max(0, std::min(2, std::atoi(egsl))) : 2;
            const char* ezga = std::getenv("LST_ZIP_GA");
            zip_ga = !(ezga && ezga[0] == '0');
            const char* ecf = std::getenv("LST_CONV_FACT");
            conv_fact = !(ecf && ecf[0] == '0');
            const char* ewgs = std::getenv("LST_WG_SCHED");
            wg_sched = ewgs && ewgs[0] == '1';
            const char* ewgd = std::getenv("LST_WG_DEFER");
            wg_defer = ewgd ? (ewgd[0] == '1') : (cc_maj < 10 || N <= 512);
            const char* ey3 = std::getenv("LST_Y3P");
            y3p = ey3 ? std::atoi(ey3) : ((cc_maj >= 8) ? 3 : 0);
            const char* efh = std::getenv("LST_FORCE_HFMA2");
            force_hfma2 = efh && efh[0] == '1';
            const char* efd = std::getenv("LST_FORCE_DIR");
            force_dir = efd ? std::atoi(efd) : 0;
            const char* efdes = std::getenv("LST_FORCE_DIR_ES");
            force_dir_es = efdes ? std::max(0, std::atoi(efdes)) : 0;
            const char* ecc64 = std::getenv("LST_CONV_C64");
            conv_c64 = ecc64 && ecc64[0] == '1';
            const char* et0d = std::getenv("LST_T0S_DIR");
            t0s_dir = et0d && et0d[0] == '1';
            const char* esc = std::getenv("LST_SCOT");
            scot = esc ? (esc[0] == '1') : (y3p >= 2);
            const char* ecopt = std::getenv("LST_CONV_OPT");
            conv_opt = ecopt && ecopt[0] == '1';
            const char* et0p = std::getenv("LST_T0S_PAIR");
            t0s_pair = et0p ? (et0p[0] == '1') : true;
            const char* efopt = std::getenv("LST_FORCE_OPT");
            force_opt = efopt ? std::atoi(efopt) : 0;
            const char* emdsp = std::getenv("LST_MLP_DUAL_SP");
            mlp_dual_sp = emdsp && emdsp[0] == '1';
            const char* eshf = std::getenv("LST_SLOT_HFMA2");
            slot_hfma2 = eshf ? std::max(0, std::min(1, std::atoi(eshf))) : 1;
            const char* esld = std::getenv("LST_SLOT_LDCG");
            slot_ldcg = !(esld && esld[0] == '0');
            const char* echf = std::getenv("LST_CONV_HFMA2");
            conv_hfma2 = echf ? std::max(0, std::min(2, std::atoi(echf))) : 2;
            const char* eshw = std::getenv("LST_SHELL_WIDTH");
            shell_width = (eshw && eshw[0]) ? (float)std::atof(eshw) : 0.f;
            const char* esh = std::getenv("LST_SHELL_RCUT");
            if (esh && esh[0]) {
                float vals[8] = {-1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f};
                int nv = std::sscanf(esh, "%f,%f,%f,%f,%f,%f,%f,%f",
                                     &vals[0], &vals[1], &vals[2], &vals[3],
                                     &vals[4], &vals[5], &vals[6], &vals[7]);
                for (int i = 0; i < nv; ++i) {
                    const float v = vals[i];
                    if (v >= 0.f && v < float(md.r_max) - 1e-6f) {
                        shell_rcut[i / 4][i % 4] = v;
                        shell_active = true;
                    }
                }
            }
            const char* egf = std::getenv("LST_GRAD_FUSE");
            grad_fuse = !(egf && egf[0] == '0');
            const char* eqbf = std::getenv("LST_SOAP_QBF");
            soap_qbf = !(eqbf && eqbf[0] == '0');
            const char* elp = std::getenv("LST_LOSS_PAR");
            loss_par = !(elp && elp[0] == '0');
            const char* espc = std::getenv("LST_SOAP_COAL");
            soap_coal = !(espc && espc[0] == '0');
            const char* espf = std::getenv("LST_SOAP_FAST");
            soap_fast = !(espf && espf[0] == '0');
            const char* esgh = std::getenv("LST_SEGG_H16");
            segg_h16 = !(esgh && esgh[0] == '0');
            const char* emgv3 = std::getenv("LST_MGEMM_V3");
            mgemm_v3 = emgv3 ? std::max(0, std::min(2, std::atoi(emgv3))) : 0;
            const char* emspv2 = std::getenv("LST_MLP_SPV2");
            mlp_spv2 = emspv2 ? std::max(0, std::min(2, std::atoi(emspv2))) : 1;
        }
        if constexpr (std::is_same<T, float>::value) {
            const char* ev = std::getenv("LST_MLP_FUSED");
            fused_mlp = NB == mlpf::NB && H == mlpf::H && O == mlpf::O && !(ev && ev[0] == '0');
            if (fused_mlp) {
                const char* es = std::getenv("LST_MLP_SPLIT");
                mlp_split = es && es[0] == '1';
                const char* ep = std::getenv("LST_MLP_PIPE");
                mlp_pipe = !(ep && ep[0] == '0');
                const char* enz = std::getenv("LST_MLP_NOZ1");
                mlp_noz1 = enz && enz[0] == '1';
                const char* efw = std::getenv("LST_MLP_FWD_SWZ");
                mlp_fwd_swz = efw && efw[0] == '1';
                const char* epf = std::getenv("LST_MLP_L2PF");
                mlp_l2pf = epf && epf[0] == '1' ? 1 : 0;
                const char* e2 = std::getenv("LST_MLP_BWD2");
                mlp_bwd2 = e2 && e2[0] == '1';
                const char* et = std::getenv("LST_MLP_TC");
                const char* em = std::getenv("LST_MATH");
                const std::string mt = et ? et : (em ? em : "");
                mlp_tc = mt == "tf32" ? mlpt::TF32 : mt == "bf16" ? mlpt::BF16 : -1;
                if (!mlp_spline || !mlp_spline_small || mlp_tc >= 0) ensure_mlpf_attrs();
            }
        }
        // parameter sizes (named_parameters order)
        long sz[LST_N_PARAMS] = {(long)S * C, (long)C * C, (long)NB * H, (long)H * H, (long)H * H, (long)H * O,
                                 4L * C * C, 4L * C * SC, (long)C * C, (long)NB * H, (long)H * H, (long)H * H,
                                 (long)H * O, 4L * C * C, (long)C * SC, (long)S * m.K3 * C, (long)S * m.K2 * C,
                                 (long)S * (K - m.K3 - m.K2) * C, (long)C * C, (long)S * m.K3 * C,
                                 (long)S * m.K2 * C, (long)S * (K - m.K3 - m.K2) * C, (long)C * C, (long)C,
                                 (long)C * hid, (long)hid};
        for (int i = 0; i < LST_N_PARAMS; ++i) {
            psize[i] = sz[i];
            poff[i] = ptotal;
            ptotal += sz[i];
        }
        // small device state
        d_step = alloci<int>(2);
        d_estep = alloci<int>(1);
        d_efids = alloci<int>(sh.B);
        {
            const char* fo = std::getenv("LST_FORCE_ORDER");
            force_order = !(fo && fo[0] == '0');
        }
        d_wd = alloci<double>(16);
        d_norm2 = alloci<double>(1);
        d_loss_acc = alloci<double>(4);
        d_loss_done = alloci<int>(1);
        d_gf_done = alloci<int>(1);
        d_fmax = alloci<unsigned int>(2);
        d_scot = alloci<float>(2);
        {
            const float ones2[2] = {1.f, 1.f};
            CK(cudaMemcpy(d_scot, ones2, sizeof(ones2), cudaMemcpyHostToDevice));
        }
        if (C == 64 && std::is_same<T, float>::value) GA_buf = alloci<uint32_t>(16L * N * C);
        d_urms = alloci<double>(LST_N_PARAMS);
        d_caut = alloci<double>(LST_N_PARAMS);
        d_pdesc = alloci<PDesc<T>>(LST_N_PARAMS);
        d_steps = alloci<float*>(LST_N_PARAMS);
        d_osc = alloc(nullptr, 3 * LST_N_PARAMS + 2);
        d_bw = alloc(nullptr, NB);
        std::vector<T> bw(NB);
        for (int i = 0; i < NB; ++i) bw[i] = T(m.bessel_w[i]);
        CK(cudaMemcpy(d_bw, bw.data(), sizeof(T) * NB, cudaMemcpyHostToDevice));
        // batch
        long NC = (long)N * C, N16 = 16 * NC;
        b.vec = alloc("vec", 3L * E);
        b.snd = alloci<int>(E); b.rcv = alloci<int>(E); b.pidx = alloci<int>(E); b.pe0 = alloci<int>(P);
        b.pe1 = alloci<int>(P); b.ngraph = alloci<int>(N); b.egraph = alloci<int>(E); b.rptr = alloci<int>(N + 1);
        b.rptr0 = alloci<int>(N);
        b.pptr = alloci<int>(N + 1);
        b.ps = alloci<int>(E);
        b.emeta = alloci<int4>(E);
        b.emeta_dir = alloci<int4>(E);
        b.p_vec4 = alloci<float4>(P);
        b.p_srg = alloci<int4>(P);
        b.puv = alloci<int2>(P);
        b.f_meta = alloci<int4>(sh.B);
        b.spc_off = alloci<int>(S + 1);
        b.spc_idx = alloci<int>(N);
        b.spc_tiles = alloci<int>(4 * S + 1);
        b.spc_tiles32 = alloci<int>(S + 1);
        b.spc_wsk = alloci<int>(S + 1);
        b.norder = alloci<int>(N);
        b.sp_idx = alloci<int>(P);
        b.sp_u = alloc(nullptr, P);
        b.spc = alloci<int>(N);
        b.onehot = alloc("onehot", (long)N * S); b.f_ref = alloc("f_ref", 3L * N); b.core_f = alloc("core_f", 3L * N);
        b.w_f = alloc("w_f", N); b.e_ref = alloc("e_ref", G1); b.core_e = alloc("core_e", G1);
        b.volume = alloc("volume", G1); b.n_atoms = alloc("n_atoms", G1); b.w_e = alloc("w_e", G1);
        b.w_s = alloc("w_s", G1); b.s_ref = alloc("s_ref", 9L * G1); b.core_s = alloc("core_s", 9L * G1);
        b.counts = alloc("counts", 2);
        b.ei_ref = alloc("ei_ref", N);
        // views
        emb = alloc("emb", (long)S * C); wup0 = alloc("wup0", (long)C * C); wl0 = alloc("wl0", 4L * C * C);
        ws0 = alloc("ws0", 4L * C * SC); wl1 = alloc("wl1", 4L * C * C); W0 = alloc("W0", (long)S * K * C);
        W1 = alloc("W1", (long)S * K * C); wp0 = alloc("wp0", (long)C * C); Wcat = alloc("Wcat", (long)C * ZW);
        wp1 = alloc("wp1", (long)C * C); wro1 = alloc("wro1", (long)C * hid); wro2 = alloc("wro2", hid);
        T0 = alloc("T0", (long)S * C); M0 = alloc("M0", 4L * C * SC); Q = alloc("Q", (long)C * ZW);
        ones = alloc(nullptr, N);
        k_fill<T><<<nblk(N), TPB>>>(ones, T(1), N);
        KCHECK();
        long wsz[4] = {(long)NB * H, (long)H * H, (long)H * H, (long)H * O};
        for (int t = 0; t < 2; ++t)
            for (int k = 0; k < 4; ++k) {
                std::string nm = "mlp" + std::to_string(t) + "_" + std::to_string(k);
                mlpW[t][k] = alloc(nm.c_str(), wsz[k]);
                gmlp[t][k] = alloc(("g" + nm).c_str(), wsz[k]);
            }
        if (fused_mlp)
            for (int t = 0; t < 2; ++t) mlpT[t] = alloc(nullptr, mlpf::WT_SIZE);
        if (own_tn) {
            tn_cap = 2L << 20;
            tn_ws = alloc(nullptr, tn_cap);
        }
        // edges
        Y = alloc("Y", 16L * E); JY = alloc("JY", 48L * E); EF = alloc("EF", (long)E * NB);
        DEF = alloc("DEF", (long)E * NB); U = alloc("U", 3L * E); UP = alloc(nullptr, 4L * P); xe = alloc("xe", 2L * P * NB);
        for (int t = 0; t < 2; ++t) {
            std::string s_ = std::to_string(t);
            z1[t] = alloc(("z1_" + s_).c_str(), 2L * P * H); z2[t] = alloc(("z2_" + s_).c_str(), 2L * P * H);
            z3[t] = alloc(("z3_" + s_).c_str(), 2L * P * H); h1[t] = alloc(("h1_" + s_).c_str(), 2L * P * H);
            h2[t] = alloc(("h2_" + s_).c_str(), 2L * P * H); h3[t] = alloc(("h3_" + s_).c_str(), 2L * P * H);
            out[t] = alloc(("RT" + s_).c_str(), 2L * P * O);     // R_t rows [0, P), T_t rows [P, 2P)
            gout[t] = alloc(("RTb" + s_).c_str(), 2L * P * O);   // their cotangents
        }
        gh = alloc(nullptr, 2L * P * H); gz = alloc(nullptr, 2L * P * H);
        if (mlp_spline > 0) {
            spline_M = mlp_spline + 1;
            d_rmin = alloci<float>(1);
            EF_k = alloc(nullptr, (long)spline_M * NB);
            DEF_k = alloc(nullptr, (long)spline_M * NB);
            pe0_k = alloci<int>(spline_M);
            mspl::k_knot_basis<T><<<nblk(spline_M), TPB>>>(EF_k, DEF_k, pe0_k, d_bw, T(md.prefactor), T(md.r_max),
                                                           md.p_cut, NB, mlp_spline);
            KCHECK();
            for (int t = 0; t < 2; ++t) {
                h3_k[t] = alloc(nullptr, 2L * spline_M * H);
                out_k[t] = alloc(nullptr, 2L * spline_M * O);
                gout_k[t] = alloc(nullptr, 2L * spline_M * O);
                C_k[t] = alloc(nullptr, 4L * mlp_spline * O);
                gC_k[t] = alloc(nullptr, 4L * mlp_spline * O);
                if (!fused_mlp || (mlp_savez && spline_M <= 260)) {
                    z1_k[t] = alloc(nullptr, 2L * spline_M * H);
                    z2_k[t] = alloc(nullptr, 2L * spline_M * H);
                    z3_k[t] = alloc(nullptr, 2L * spline_M * H);
                }
                if (!fused_mlp) {
                    h1_k[t] = alloc(nullptr, 2L * spline_M * H);
                    h2_k[t] = alloc(nullptr, 2L * spline_M * H);
                }
            }
            if (!fused_mlp) {
                xe_k = alloc(nullptr, 2L * spline_M * NB);
                gh_k = alloc(nullptr, 2L * spline_M * H);
                gz_k = alloc(nullptr, 2L * spline_M * H);
                k_pair_feats<T><<<nblk((long)spline_M * NB), TPB>>>(pe0_k, EF_k, DEF_k, xe_k, spline_M, NB);
                KCHECK();
            }
        }
        // nodes
        x0 = alloc("x0", NC); A0 = alloc("A0", N16); tmp = alloc(nullptr, 16L * N * SC); B0 = alloc("B0", N16);
        f0 = alloc("f0", NC); z = alloc("z", (long)N * ZW); z0 = alloc(nullptr, N); x1 = alloc("x1", NC); sc = alloc("sc", NC);
        A1 = alloc("A1", N16); B1 = alloc("B1", N16); f1 = alloc("f1", NC); f2 = alloc("f2", NC);
        u = alloc("u", (long)N * hid); tt = alloc("tt", (long)N * hid); ne = alloc("ne", N); Eg = alloc("E", G1);
        ub = alloc("ub", (long)N * hid); f2b = alloc("f2b", NC); f1b = alloc("f1b", NC); B1b = alloc("B1b", N16);
        Gn1 = alloc("G1", N16); x1b = alloc("x1b", NC); zb = alloc("zb", (long)N * ZW); f0b = alloc("f0b", NC);
        B0b = alloc("B0b", N16); Gn0 = alloc("G0", N16); se = alloc("s_e", E); gy = alloc("gy", 16L * E);
        g = alloc("g", 3L * E); F = alloc("F", 3L * N); mv = alloc("mv", 9L * G1); Sg = alloc("S", 9L * G1);
        Eb = alloc("Eb", G1); Enb = alloc("Enb", N); Fb = alloc("Fb", 3L * N); Ms = alloc("Ms", 9L * G1);
        sb = alloc("sb", E); gyb = alloc("gyb", 16L * E); Yw = alloc("Yw", 16L * E); Y3P = alloc(nullptr, 48L * P);
        Gs0 = alloc("G0s", N16); Gs1 = alloc("G1s", N16); Xd0 = alloc("x0s", NC); Xd1 = alloc("x1s", NC);
        B0bs = alloc("B0bs", N16); M0s = alloc("M0s", 4L * C * SC); B0s = alloc("B0s", N16);
        f0bs = alloc("f0bs", NC); Wp0 = alloc(nullptr, (long)N * K * C); Wp1 = alloc(nullptr, (long)N * K * C);
        zbs = alloc("zbs", (long)N * ZW); Qs = alloc("Qs", (long)C * ZW); x1bs = alloc("x1bs", NC);
        scbs = alloc("scbs", NC); B1bs = alloc("B1bs", N16); wl1s = alloc("gwl1", 4L * C * C);
        B1s = alloc("B1s", N16); f1bs = alloc("f1bs", NC); f2bs = alloc("f2bs", NC); wp1s = alloc("gwp1", (long)C * C);
        ubs = alloc("ubs", (long)N * hid); wro1s = alloc("gwro1", (long)C * hid); tbs = alloc("tbs", (long)N * hid);
        us = alloc("us", (long)N * hid); wro2s = alloc("gwro2", hid); qn = alloc("qn", N); f2s = alloc("f2s", NC);
        f1s = alloc("f1s", NC); A1s = alloc("A1s", N16); zs = alloc("zs", (long)N * ZW); f0s = alloc("f0s", NC);
        A0s = alloc("A0s", N16); T0s = alloc("T0s", (long)S * C);
        gemb = alloc("gemb", (long)S * C); gwup0 = alloc("gwup0", (long)C * C); gwl0 = alloc("gwl0", 4L * C * C);
        gws0 = alloc("gws0", 4L * C * SC); gwl1 = wl1s; gW0 = alloc("gW0", (long)S * K * C);
        gW1 = alloc("gW1", (long)S * K * C); gwp0 = alloc("gwp0", (long)C * C); gWcat = alloc("gWcat", (long)C * ZW);
        build_mg();
        gwp1 = wp1s; gwro1 = wro1s;
        graw = alloc("graw", ptotal);
        if constexpr (std::is_same<T, float>::value) {
            if (C == 64) {
                M0_h = alloci<uint16_t>(2L * 4L * S * 4096L);
                Wl1_h = alloci<uint16_t>(2L * 4L * 4096L);
            }
        }
        // Wave 29 lazy cuBLAS: skip creating 3 cuBLAS handles (~0.8-1.0s + 96 MB VRAM!) when fast custom kernels
        // handle 100% of GEMMs; fallback/non-fast paths initialize eagerly before graph capture.
        const bool fast_no_cublas =
            std::is_same<T, float>::value && C == 64 && H == 64 && hid == 16 &&
            seg_gemm && multi_gemm && fused_mlp && nops_opt && m0_species &&
            prod_direct && (mlp_spline > 0) && !mlp_split && wbal_active() && cz_active();
        if (!fast_no_cublas) {
            ensure_cb_for(st_main);
            ensure_cb_for(s2);
            if (wg_side) ensure_cb_for(s3);
        }
        CK(cudaDeviceSynchronize());
    }

    ~Engine() override {
        if (d_mg_pv) cudaFree(d_mg_pv);
        if (d_mg_pv_t0) cudaFree(d_mg_pv_t0);
        if (d_mg_pv_mq) cudaFree(d_mg_pv_mq);
        if (d_gftiles) cudaFree(d_gftiles);
        if (d_mg_vg) cudaFree(d_mg_vg);
        if (d_mg_vg_early) cudaFree(d_mg_vg_early);
        if (d_mg_vg_t0) cudaFree(d_mg_vg_t0);
        if (d_mg_vg_all) cudaFree(d_mg_vg_all);
        if (d_mg_vg_q) cudaFree(d_mg_vg_q);
        if (d_mg_vg_m0) cudaFree(d_mg_vg_m0);
        if (d_fids) cudaFree(d_fids);
        if (d_ema) cudaFree(d_ema);
        if (d_lr) cudaFree(d_lr);
        if (d_lw) cudaFree(d_lw);
        if (d_wei) cudaFree(d_wei);
        if (d_losses) cudaFree(d_losses);
        if (gexec) cudaGraphExecDestroy(gexec);
        if (graph) cudaGraphDestroy(graph);
        if (gexec_fast) cudaGraphExecDestroy(gexec_fast);
        if (graph_fast) cudaGraphDestroy(graph_fast);
        if (gexec_cycle) cudaGraphExecDestroy(gexec_cycle);
        if (graph_cycle) cudaGraphDestroy(graph_cycle);
        if (eval_e_exec) cudaGraphExecDestroy(eval_e_exec);
        if (eval_fs_exec) cudaGraphExecDestroy(eval_fs_exec);
        if (cb) cublasDestroy(cb);
        if (cb_ws) cudaFree(cb_ws);
        if (cb2) cublasDestroy(cb2);
        if (cb2_ws) cudaFree(cb2_ws);
        for (auto& e : evf)
            if (e) cudaEventDestroy(e);
        if (s2) cudaStreamDestroy(s2);
        if (cb3) cublasDestroy(cb3);
        if (cb3_ws) cudaFree(cb3_ws);
        for (auto& e : evw)
            if (e) cudaEventDestroy(e);
        if (s3) cudaStreamDestroy(s3);
        if (s_cond) cudaStreamDestroy(s_cond);
        for (void* p : ks_allocs) cudaFree(p);
        for (void* p : allocs) cudaFree(p);
        if (ev_in) cudaEventDestroy(ev_in);
        if (ev_out) cudaEventDestroy(ev_out);
        if (ev_geo_go) cudaEventDestroy(ev_geo_go);
        if (ev_geo_ready) cudaEventDestroy(ev_geo_ready);
        if (st_main) cudaStreamDestroy(st_main);
    }

    void ensure_mlpf_attrs() {
        if constexpr (std::is_same<T, float>::value) {
            if (mlpf_attrs_ready || !fused_mlp) return;
            mlpf_attrs_ready = true;
            CK(cudaFuncSetAttribute(mlpf::k_mlp_bwd<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, mlpf::SMEM_BWD));
            CK(cudaFuncSetAttribute(mlpf::k_mlp_bwd<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, mlpf::SMEM_BWD));
            CK(cudaFuncSetAttribute(mlpf::k_mlp_top, cudaFuncAttributeMaxDynamicSharedMemorySize, mlpf::SMEM_BWD));
            CK(cudaFuncSetAttribute(mlpf::k_mlp_bwdp, cudaFuncAttributeMaxDynamicSharedMemorySize, mlpf::SMEM_BWDP));
            CK(cudaFuncSetAttribute(mlpf::k_mlp_bwdp_noz1, cudaFuncAttributeMaxDynamicSharedMemorySize, mlpf::SMEM_BWDP_NOZ1));
            CK(cudaFuncSetAttribute(mlpf::k_mlp_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize, mlpf::SMEM_FWD));
            CK(cudaFuncSetAttribute(mlpf::k_mlp_fwd_swz, cudaFuncAttributeMaxDynamicSharedMemorySize, mlpf::SMEM_FWD));
            int nsm_l = 0, per = 0;
            CK(cudaDeviceGetAttribute(&nsm_l, cudaDevAttrMultiProcessorCount, dev));
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, mlpf::k_mlp_bwd<true>, mlpf::NTH, mlpf::SMEM_BWD));
            const int ntiles = (P + mlpf::TP - 1) / mlpf::TP;
            mlp_bwd_grid = std::max(1, std::min(ntiles, nsm_l * std::max(1, per)));
            const char* eg = std::getenv("LST_MLP_GRIDBAL");
            if (eg && eg[0] == '1') {
                const int rounds = (ntiles + mlp_bwd_grid - 1) / mlp_bwd_grid;
                mlp_bwd_grid = (ntiles + rounds - 1) / rounds;
            }
#ifndef LST_LEAN
            if (mlp_bwd2) {
                CK(cudaFuncSetAttribute(mlpf2::k_mlp_bwd2, cudaFuncAttributeMaxDynamicSharedMemorySize, mlpf2::SMEM_BWD));
                int per2 = 0;
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per2, mlpf2::k_mlp_bwd2, mlpf2::NTH, mlpf2::SMEM_BWD));
                mlp_bwd2_grid = std::max(1, std::min(ntiles, nsm_l * std::max(1, per2)));
            }
#endif
            if (mlp_tc >= 0) {
                mlp_tc_ready = true;
                CK(cudaFuncSetAttribute(mlpt::k_fwd<mlpt::TF32>, cudaFuncAttributeMaxDynamicSharedMemorySize, mlpt::SMEM_FWD));
                CK(cudaFuncSetAttribute(mlpt::k_fwd<mlpt::BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, mlpt::SMEM_FWD));
                CK(cudaFuncSetAttribute(mlpt::k_bwd<mlpt::TF32>, cudaFuncAttributeMaxDynamicSharedMemorySize, mlpt::SMEM_BWD));
                CK(cudaFuncSetAttribute(mlpt::k_bwd<mlpt::BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize, mlpt::SMEM_BWD));
                int pt = 0;
                if (mlp_tc == mlpt::TF32)
                    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&pt, mlpt::k_bwd<mlpt::TF32>, mlpt::NTH, mlpt::SMEM_BWD));
                else
                    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&pt, mlpt::k_bwd<mlpt::BF16>, mlpt::NTH, mlpt::SMEM_BWD));
                mlp_tc_grid = std::max(1, std::min(ntiles, nsm_l * std::max(1, pt)));
            }
        }
    }

    void drop_eval_graphs() {
        if (eval_e_exec) { cudaGraphExecDestroy(eval_e_exec); eval_e_exec = nullptr; }
        if (eval_fs_exec) { cudaGraphExecDestroy(eval_fs_exec); eval_fs_exec = nullptr; }
        eval_e_dst = eval_f_dst = eval_s_dst = nullptr;
        eval_views_valid = false;
    }

    void drop_graph() {
        if (gexec) cudaGraphExecDestroy(gexec);
        if (graph) cudaGraphDestroy(graph);
        if (gexec_fast) cudaGraphExecDestroy(gexec_fast);
        if (graph_fast) cudaGraphDestroy(graph_fast);
        if (gexec_cycle) cudaGraphExecDestroy(gexec_cycle);
        if (graph_cycle) cudaGraphDestroy(graph_cycle);
        gexec = nullptr;
        graph = nullptr;
        gexec_fast = nullptr;
        graph_fast = nullptr;
        gexec_cycle = nullptr;
        graph_cycle = nullptr;
        graph_cycle_len = 0;
        opt_step_host = -1;
        opt_steps_aligned = false;
        drop_eval_graphs();
    }

    void set_eval_cache(int on) override {
        eval_cache = (on != 0);
        drop_eval_graphs();
    }

    void set_params(void* const* p) override {
        for (int i = 0; i < LST_N_PARAMS; ++i) raw[i] = reinterpret_cast<T*>(p[i]);
        build_tables();
        have_params = true;
        drop_graph();
    }

    void ksort_clear() {
        for (void* p : ks_allocs) cudaFree(p);
        ks_allocs.clear();
        d_ks_toff = nullptr;
        d_ks_tiles = nullptr;
        ks_topo = Topo{};
        ks_frames = FramesT<T>{};
        ks_max_tiles = 0;
        ks_ready = false;
    }

    template <typename U>
    U* ks_upload(const std::vector<U>& v) {
        void* p = nullptr;
        const size_t bytes = sizeof(U) * std::max<size_t>(1, v.size());
        CK(cudaMalloc(&p, bytes));
        if (!v.empty()) CK(cudaMemcpy(p, v.data(), sizeof(U) * v.size(), cudaMemcpyHostToDevice));
        ks_allocs.push_back(p);
        return reinterpret_cast<U*>(p);
    }

    void ksort_build(const LstFrames* f) {
        ksort_clear();
        if (!edge_ksort || mlp_spline <= 0 || !mlp_spline_fused || !pair_basis || f->n_frames <= 0) return;
        const int nf = f->n_frames;
        std::vector<int> h_noff(nf + 1), h_eoff(nf + 1), h_poff(nf + 1);
        CK(cudaMemcpy(h_noff.data(), f->node_off, sizeof(int) * (nf + 1), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_eoff.data(), f->edge_off, sizeof(int) * (nf + 1), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_poff.data(), f->pair_off, sizeof(int) * (nf + 1), cudaMemcpyDeviceToHost));
        const int tot_n = h_noff[nf], tot_e = h_eoff[nf], tot_p = h_poff[nf];
        if (tot_e <= 0 || tot_p <= 0) return;

        // Wave 29 fast GPU-native ksort_build (~3 ms vs ~800-1100 ms single-threaded CPU!)
        const bool need_ed = (force_dir == 2 || conv_opt || conv_c64 || slot_opt >= 13);
        int max_n_f = 0;
        for (int fi = 0; fi < nf; ++fi) max_n_f = std::max(max_n_f, h_noff[fi + 1] - h_noff[fi]);
        if (!slot_kb && !force_kb && !need_ed && mlp_spline <= 64 && max_n_f <= 2048) {
            auto align256 = [](size_t b) -> size_t { return (b + 255u) & ~size_t(255u); };
            const size_t sz_e_i = align256(sizeof(int) * (size_t)tot_e);
            const size_t sz_e_f = align256(sizeof(float) * (size_t)tot_e);
            const size_t sz_p_i = align256(sizeof(int) * (size_t)tot_p);
            const size_t sz_p_f = align256(sizeof(float) * (size_t)tot_p);
            const size_t sz_n_i = align256(sizeof(int) * (size_t)tot_n);
            const size_t sz_vec = align256(sizeof(T) * 3L * (size_t)(tot_e + 1));
            const size_t sz_e_i4 = align256(sizeof(int4) * (size_t)tot_e);
            const size_t sz_p_f4 = align256(sizeof(float4) * (size_t)(tot_p + 1));
            const size_t sz_p_i2 = align256(sizeof(int2) * (size_t)tot_p);

            // Single persistent GPU arena for all ks_topo + ks_frames.vec arrays (1 cudaMalloc!)
            const size_t arena_bytes = 5 * sz_e_i + sz_e_f + 2 * sz_p_i + sz_n_i + sz_vec + sz_e_i4 + sz_p_f4 + sz_p_i2;
            char* arena = nullptr;
            CK(cudaMalloc(&arena, arena_bytes));
            ks_allocs.push_back(arena);
            size_t off = 0;
            auto slice_i = [&](size_t sz) -> int* { int* p = reinterpret_cast<int*>(arena + off); off += sz; return p; };
            auto slice_f = [&](size_t sz) -> float* { float* p = reinterpret_cast<float*>(arena + off); off += sz; return p; };
            int* d_ks_snd = slice_i(sz_e_i);
            int* d_ks_rcv = slice_i(sz_e_i);
            int* d_ks_pidx = slice_i(sz_e_i);
            int* d_ks_eps = slice_i(sz_e_i);
            int* d_ks_ek = slice_i(sz_e_i);
            float* d_ks_eu = slice_f(sz_e_f);
            int* d_ks_pe0 = slice_i(sz_p_i);
            int* d_ks_pe1 = slice_i(sz_p_i);
            int* d_ks_rptr0 = slice_i(sz_n_i);
            T* d_ks_vec = reinterpret_cast<T*>(arena + off); off += sz_vec;
            int4* d_ks_emeta = reinterpret_cast<int4*>(arena + off); off += sz_e_i4;
            float4* d_ks_p_vec4 = reinterpret_cast<float4*>(arena + off); off += sz_p_f4;
            int2* d_ks_p_sr = reinterpret_cast<int2*>(arena + off); off += sz_p_i2;

            // Single temporary GPU arena (1 cudaMalloc + 1 cudaFree!)
            const size_t tmp_bytes = 5 * sz_p_i + sz_p_f + 3 * sz_e_i;
            char* tmp_a = nullptr;
            CK(cudaMalloc(&tmp_a, tmp_bytes));
            size_t toff = 0;
            auto tslice_i = [&](size_t sz) -> int* { int* p = reinterpret_cast<int*>(tmp_a + toff); toff += sz; return p; };
            int* d_p_kidx = tslice_i(sz_p_i);
            int* d_p_e0 = tslice_i(sz_p_i);
            int* d_p_e1 = tslice_i(sz_p_i);
            int* d_new_p = tslice_i(sz_p_i);
            int* d_old_p = tslice_i(sz_p_i);
            float* d_p_us = reinterpret_cast<float*>(tmp_a + toff); toff += sz_p_f;
            int* d_new_e = tslice_i(sz_e_i);
            int* d_tmp_oe = tslice_i(sz_e_i);
            uint32_t* d_tmp_key = reinterpret_cast<uint32_t*>(tmp_a + toff); toff += sz_e_i;

            const T rmax = T(md.r_max), rlo = T(sp_r_lo), invh = sp_invh();
            const int spK = mlp_spline;
            k_ksort_pairs_gpu<T><<<nf, 256, 0, st_main>>>(
                nf, f->node_off, f->edge_off, f->pair_off, f->receiver, f->pe0, f->pe1, (const T*)f->vectors,
                rmax, rlo, invh, spK, pair_bal, d_p_kidx, d_p_us, d_p_e0, d_p_e1, d_new_p, d_old_p);
            k_ksort_edges_gpu<T><<<nf, 256, 0, st_main>>>(
                nf, f->node_off, f->edge_off, f->pair_off, f->receiver_ptr, f->sender, f->receiver, f->pidx,
                (const T*)f->vectors, d_p_kidx, d_p_us, d_p_e1, d_new_p, d_tmp_oe, d_tmp_key,
                d_ks_rptr0, d_new_e, d_ks_snd, d_ks_rcv, d_ks_pidx, d_ks_eps, d_ks_ek, d_ks_eu, d_ks_vec, d_ks_emeta);
            k_ksort_scatter_pairs_gpu<T><<<nf, 256, 0, st_main>>>(
                nf, f->edge_off, f->pair_off, d_old_p, d_p_e0, d_p_e1, d_new_e, d_ks_pe0, d_ks_pe1,
                (const T*)f->vectors, f->sender, f->receiver, d_ks_p_vec4, d_ks_p_sr);
            CK(cudaMemcpyAsync(d_ks_vec + 3L * tot_e, (const T*)f->vectors + 3L * tot_e, 3 * sizeof(T),
                               cudaMemcpyDeviceToDevice, st_main));
            T dvec[3] = {T(0), T(0), T(0)};
            CK(cudaMemcpyAsync(dvec, (const T*)f->vectors + 3L * tot_e, 3 * sizeof(T), cudaMemcpyDeviceToHost, st_main));
            CK(cudaStreamSynchronize(st_main));
            const float4 dv4 = make_float4((float)dvec[0], (float)dvec[1], (float)dvec[2], 0.f);
            CK(cudaMemcpy(d_ks_p_vec4 + tot_p, &dv4, sizeof(float4), cudaMemcpyHostToDevice));
            cudaFree(tmp_a);

            const T dr = std::sqrt(dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]);
            int d_ki = -1;
            T d_us = T(0);
            if (dr < rmax) {
                const T pos = (dr - rlo) * invh;
                int idx = (int)pos;
                if (idx < 0) idx = 0;
                if (idx >= spK) idx = spK - 1;
                d_ki = idx;
                d_us = pos - T(idx);
            }
            ks_topo = topo;
            ks_topo.snd = d_ks_snd;
            ks_topo.rcv = d_ks_rcv;
            ks_topo.pidx = d_ks_pidx;
            ks_topo.pe0 = d_ks_pe0;
            ks_topo.pe1 = d_ks_pe1;
            ks_topo.eps = d_ks_eps;
            ks_topo.ek = d_ks_ek;
            ks_topo.eu = d_ks_eu;
            ks_topo.ed_eps = nullptr;
            ks_topo.ed_snd = nullptr;
            ks_topo.ed_ek = nullptr;
            ks_topo.ed_eu = nullptr;
            ks_topo.puv = nullptr;
            ks_topo.rptr0 = d_ks_rptr0;
            ks_topo.emeta_pre = d_ks_emeta;
            ks_topo.p_vec4 = d_ks_p_vec4;
            ks_topo.p_sr = d_ks_p_sr;
            ks_topo.dummy_kidx = d_ki;
            ks_topo.dummy_u = (float)d_us;
            ks_frames = frames;
            ks_frames.vec = d_ks_vec;
            d_ks_toff = nullptr;
            d_ks_tiles = nullptr;
            ks_max_tiles = 0;
            ks_ready = true;
            return;
        }
        std::vector<int> h_snd(tot_e), h_rcv(tot_e), h_pidx(tot_e), h_pe0(tot_p), h_pe1(tot_p);
        std::vector<int> h_rptr(tot_n + nf);
        std::vector<T> h_vec(3L * (tot_e + 1));
        CK(cudaMemcpy(h_snd.data(), f->sender, sizeof(int) * tot_e, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_rcv.data(), f->receiver, sizeof(int) * tot_e, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_pidx.data(), f->pidx, sizeof(int) * tot_e, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_pe0.data(), f->pe0, sizeof(int) * tot_p, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_pe1.data(), f->pe1, sizeof(int) * tot_p, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_rptr.data(), f->receiver_ptr, sizeof(int) * (tot_n + nf), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_vec.data(), f->vectors, sizeof(T) * 3L * (tot_e + 1), cudaMemcpyDeviceToHost));

        std::vector<int> ks_snd(tot_e), ks_rcv(tot_e), ks_pidx(tot_e), ks_pe0(tot_p), ks_pe1(tot_p);
        std::vector<int> ks_eps(tot_e), ks_ek(tot_e), ks_rptr0(tot_n, 0);
        std::vector<float> ks_eu(tot_e);
        std::vector<int> ks_ed_eps(tot_e), ks_ed_snd(tot_e), ks_ed_ek(tot_e);
        std::vector<float> ks_ed_eu(tot_e);
        std::vector<int2> ks_puv(tot_p);
        std::vector<T> ks_vec(3L * (tot_e + 1));
        std::vector<int> h_toff(nf + 1, 0);
        std::vector<int4> h_tiles;

        const T rmax = T(md.r_max), rlo = T(sp_r_lo), invh = sp_invh();
        const int spK = mlp_spline;
        auto calc_knot = [&](T vx, T vy, T vz, int& kidx, T& us) {
            const T r = std::sqrt(vx * vx + vy * vy + vz * vz);
            if (r >= rmax) {
                kidx = -1;
                us = T(0);
            } else {
                const T pos = (r - rlo) * invh;
                int idx = (int)pos;
                if (idx < 0) idx = 0;
                if (idx >= spK) idx = spK - 1;
                kidx = idx;
                us = pos - T(idx);
            }
        };

        int max_tf = 0;
        std::vector<int> p_kidx, p_unode, p_vnode, p_e0, p_e1, old_p_of_new, new_p_of_old, old_e_of_new, new_e_of_old;
        std::vector<int> deg0;
        std::vector<T> p_us;
        std::vector<int> bcnt(spK + 2), bpos(spK + 2);
        for (int fi = 0; fi < nf; ++fi) {
            const int n_off = h_noff[fi], n_cnt = h_noff[fi + 1] - n_off;
            const int e_off = h_eoff[fi], e_cnt = h_eoff[fi + 1] - e_off;
            const int p_off = h_poff[fi], p_cnt = h_poff[fi + 1] - p_off;
            p_kidx.resize(p_cnt);
            p_unode.resize(p_cnt);
            p_vnode.resize(p_cnt);
            p_e0.resize(p_cnt);
            p_e1.resize(p_cnt);
            p_us.resize(p_cnt);
            deg0.assign(n_cnt, 0);
            std::fill(bcnt.begin(), bcnt.end(), 0);
            for (int lp = 0; lp < p_cnt; ++lp) {
                int le0 = h_pe0[p_off + lp];
                int le1 = h_pe1[p_off + lp];
                int u = h_rcv[e_off + le0];
                int v = h_rcv[e_off + le1];
                if (pair_bal && deg0[v] < deg0[u]) {
                    std::swap(le0, le1);
                    std::swap(u, v);
                }
                deg0[u]++;
                p_e0[lp] = le0;
                p_e1[lp] = le1;
                p_unode[lp] = u;
                p_vnode[lp] = v;
                int ki;
                T us;
                calc_knot(h_vec[3L * (e_off + le0) + 0], h_vec[3L * (e_off + le0) + 1], h_vec[3L * (e_off + le0) + 2],
                          ki, us);
                p_kidx[lp] = ki;
                p_us[lp] = us;
                bcnt[(ki < 0 ? spK : ki) + 1]++;
            }
            for (int k = 1; k <= spK + 1; ++k) bcnt[k] += bcnt[k - 1];
            bpos = bcnt;
            old_p_of_new.resize(p_cnt);
            for (int lp = 0; lp < p_cnt; ++lp) {
                const int bk = p_kidx[lp] < 0 ? spK : p_kidx[lp];
                old_p_of_new[bpos[bk]++] = lp;
            }
            for (int k = 0; k <= spK; ++k) {
                const int beg = bcnt[k], end = bcnt[k + 1];
                if (end - beg > 1) {
                    std::sort(old_p_of_new.begin() + beg, old_p_of_new.begin() + end, [&](int a, int b_) {
                        if (p_unode[a] != p_unode[b_]) return p_unode[a] < p_unode[b_];
                        return p_vnode[a] < p_vnode[b_];
                    });
                }
                if (k < spK) {
                    for (int pos = beg; pos < end; pos += kb_tile)
                        h_tiles.push_back(make_int4(k, pos, std::min(pos + kb_tile, end), 0));
                }
            }
            h_toff[fi + 1] = (int)h_tiles.size();
            max_tf = std::max(max_tf, h_toff[fi + 1] - h_toff[fi]);
            new_p_of_old.resize(p_cnt);
            for (int np = 0; np < p_cnt; ++np) new_p_of_old[old_p_of_new[np]] = np;

            // Partition each node's incoming edges: [eb, rptr0) = side==0 & kidx>=0, then side==1 & kidx>=0, then kidx<0,
            // each sorted by ascending new_p (monotonic kidx!) -> ZERO skipped side==1 edges in slot & force!
            old_e_of_new.resize(e_cnt);
            new_e_of_old.resize(e_cnt);
            const int* rptr_f = h_rptr.data() + n_off + fi;
            auto edge_grp = [&](int le) -> int {
                const int op = h_pidx[e_off + le];
                if (p_kidx[op] < 0) return 2;
                return (le == p_e1[op]) ? 1 : 0;
            };
            for (int ln = 0; ln < n_cnt; ++ln) {
                const int eb = rptr_f[ln], ee = rptr_f[ln + 1];
                int n_s0 = 0;
                for (int le = eb; le < ee; ++le) {
                    old_e_of_new[le] = le;
                    if (edge_grp(le) == 0) ++n_s0;
                }
                if (ee - eb > 1) {
                    std::sort(old_e_of_new.begin() + eb, old_e_of_new.begin() + ee, [&](int ea, int eb_) {
                        const int ga = edge_grp(ea), gb = edge_grp(eb_);
                        if (ga != gb) return ga < gb;
                        return new_p_of_old[h_pidx[e_off + ea]] < new_p_of_old[h_pidx[e_off + eb_]];
                    });
                }
                ks_rptr0[n_off + ln] = eb + n_s0;
                for (int ne = eb; ne < ee; ++ne) new_e_of_old[old_e_of_new[ne]] = ne;
            }
            // Unified (n, kidx)-sorted directed order (no side0/side1 split -> 2x longer knot runs in directed kernels!)
            std::vector<int> ed_old_e(e_cnt);
            for (int ln = 0; ln < n_cnt; ++ln) {
                const int eb = rptr_f[ln], ee = rptr_f[ln + 1];
                for (int le = eb; le < ee; ++le) ed_old_e[le] = le;
                if (ee - eb > 1) {
                    std::sort(ed_old_e.begin() + eb, ed_old_e.begin() + ee, [&](int ea, int eb_) {
                        const int ka = p_kidx[h_pidx[e_off + ea]] < 0 ? 1 : 0;
                        const int kb = p_kidx[h_pidx[e_off + eb_]] < 0 ? 1 : 0;
                        if (ka != kb) return ka < kb;
                        return new_p_of_old[h_pidx[e_off + ea]] < new_p_of_old[h_pidx[e_off + eb_]];
                    });
                }
            }
            for (int ne = 0; ne < e_cnt; ++ne) {
                const int le = old_e_of_new[ne];
                const int old_p = h_pidx[e_off + le];
                const int new_p = new_p_of_old[old_p];
                const int side = (le == p_e1[old_p]) ? 1 : 0;
                ks_snd[e_off + ne] = h_snd[e_off + le];
                ks_rcv[e_off + ne] = h_rcv[e_off + le];
                ks_pidx[e_off + ne] = new_p;
                ks_eps[e_off + ne] = (new_p << 1) | side;
                ks_ek[e_off + ne] = p_kidx[old_p];
                ks_eu[e_off + ne] = (float)p_us[old_p];
                ks_vec[3L * (e_off + ne) + 0] = h_vec[3L * (e_off + le) + 0];
                ks_vec[3L * (e_off + ne) + 1] = h_vec[3L * (e_off + le) + 1];
                ks_vec[3L * (e_off + ne) + 2] = h_vec[3L * (e_off + le) + 2];

                const int dle = ed_old_e[ne];
                const int dold_p = h_pidx[e_off + dle];
                const int dnew_p = new_p_of_old[dold_p];
                const int dside = (dle == p_e1[dold_p]) ? 1 : 0;
                ks_ed_eps[e_off + ne] = (dnew_p << 1) | dside;
                ks_ed_snd[e_off + ne] = h_snd[e_off + dle];
                ks_ed_ek[e_off + ne] = p_kidx[dold_p];
                ks_ed_eu[e_off + ne] = (float)p_us[dold_p];
            }
            for (int np = 0; np < p_cnt; ++np) {
                const int old_p = old_p_of_new[np];
                ks_pe0[p_off + np] = new_e_of_old[p_e0[old_p]];
                ks_pe1[p_off + np] = new_e_of_old[p_e1[old_p]];
                ks_puv[p_off + np] = make_int2(p_unode[old_p], p_vnode[old_p]);
            }
        }
        // Dummy edge at tot_e
        for (int k = 0; k < 3; ++k) ks_vec[3L * tot_e + k] = h_vec[3L * tot_e + k];
        int d_ki = -1;
        T d_us = T(0);
        calc_knot(ks_vec[3L * tot_e + 0], ks_vec[3L * tot_e + 1], ks_vec[3L * tot_e + 2], d_ki, d_us);

        ks_topo = topo;
        ks_topo.snd = ks_upload(ks_snd);
        ks_topo.rcv = ks_upload(ks_rcv);
        ks_topo.pidx = ks_upload(ks_pidx);
        ks_topo.pe0 = ks_upload(ks_pe0);
        ks_topo.pe1 = ks_upload(ks_pe1);
        ks_topo.eps = ks_upload(ks_eps);
        ks_topo.ek = ks_upload(ks_ek);
        ks_topo.eu = ks_upload(ks_eu);
        ks_topo.ed_eps = ks_upload(ks_ed_eps);
        ks_topo.ed_snd = ks_upload(ks_ed_snd);
        ks_topo.ed_ek = ks_upload(ks_ed_ek);
        ks_topo.ed_eu = ks_upload(ks_ed_eu);
        ks_topo.puv = (slot_kb || force_kb) ? ks_upload(ks_puv) : nullptr;
        ks_topo.rptr0 = ks_upload(ks_rptr0);
        ks_topo.dummy_kidx = d_ki;
        ks_topo.dummy_u = (float)d_us;
        ks_frames = frames;
        ks_frames.vec = ks_upload(ks_vec);
        d_ks_toff = ks_upload(h_toff);
        d_ks_tiles = ks_upload(h_tiles);
        ks_max_tiles = max_tf;
        ks_ready = true;
    }

    void set_frames(const LstFrames* f) override {
        frames.vec = (const T*)f->vectors; frames.species = f->species; frames.f_ref = (const T*)f->f_ref;
        frames.core_f = (const T*)f->core_f; frames.e_ref = (const T*)f->e_ref; frames.core_e = (const T*)f->core_e;
        frames.volume = (const T*)f->volume; frames.n_atoms = (const T*)f->n_atoms; frames.w_e = (const T*)f->w_e;
        frames.w_f = (const T*)f->w_f; frames.w_s = (const T*)f->w_s; frames.s_ref = (const T*)f->s_ref;
        frames.core_s = (const T*)f->core_s;
        frames.ei_ref = nullptr;
        ei_on = false;
        ei_steps_set = false;
        ei_weight = 0.0;
        topo = Topo{f->node_off, f->edge_off, f->pair_off, f->sender, f->receiver, f->receiver_ptr, f->pidx, f->pe0,
                    f->pe1, f->n_frames, f->dummy_row};
        if (f->dummy_row < 2 || f->dummy_row % 2) throw Error("dummy_row must be even and >= 2");
        if (mlp_spline > 0 && spline_rlo && mlp_spline_fused && f->n_frames > 0) {
            int tot_e = 0;
            CK(cudaMemcpy(&tot_e, f->edge_off + f->n_frames, sizeof(int), cudaMemcpyDeviceToHost));
            double new_rlo = 0.0;
            if (tot_e > 0) {
                float init_min = 1e30f, rmin = 1e30f;
                CK(cudaMemcpy(d_rmin, &init_min, sizeof(float), cudaMemcpyHostToDevice));
                mspl::k_min_edge_r<T><<<std::min(256, nblk(tot_e)), 256>>>(frames.vec, tot_e, d_rmin);
                KCHECK();
                CK(cudaMemcpy(&rmin, d_rmin, sizeof(float), cudaMemcpyDeviceToHost));
                if (rmin < (float)md.r_max && rmin > 0.25f)
                    new_rlo = std::max(0.0, std::floor(((double)rmin - 0.15) * 20.0) / 20.0);
            }
            if (new_rlo != sp_r_lo) {
                sp_r_lo = new_rlo;
                mspl::k_knot_basis<T><<<nblk(spline_M), TPB>>>(EF_k, DEF_k, pe0_k, d_bw, T(md.prefactor), T(md.r_max),
                                                               md.p_cut, NB, mlp_spline, T(sp_r_lo));
                if (!fused_mlp) k_pair_feats<T><<<nblk((long)spline_M * NB), TPB>>>(pe0_k, EF_k, DEF_k, xe_k, spline_M, NB);
                KCHECK();
                CK(cudaDeviceSynchronize());
            }
        }
        ksort_build(f);
        have_frames = true;
        drop_graph();
    }

    void set_node_order(const int32_t* order) override {
        if (!have_frames) throw Error("lst_set_node_order: lst_set_frames first");
        topo.norder = order;
        ks_topo.norder = order;
        drop_graph();
    }

    void set_node_energy(const void* ei_ref, double weight) override {
        if (!have_frames) throw Error("lst_set_node_energy: lst_set_frames first");
        frames.ei_ref = reinterpret_cast<const T*>(ei_ref);
        ks_frames.ei_ref = frames.ei_ref;
        ei_weight = weight;
        ei_steps_set = false;
        ei_on = (frames.ei_ref != nullptr && weight > 0.0);
        drop_graph();
    }

    void set_node_energy_steps(const double* w_ei_steps, int n_steps, cudaStream_t user) override {
        if (!have_frames || !frames.ei_ref) throw Error("lst_set_node_energy_steps: lst_set_node_energy first");
        if (n_steps <= 0) throw Error("lst_set_node_energy_steps: n_steps must be > 0");
        bool need_drop = !ei_on || !ei_steps_set;
        if (n_steps > cap_wei) {
            CK(cudaDeviceSynchronize());
            if (d_wei) cudaFree(d_wei);
            cap_wei = std::max(n_steps, 2 * cap_wei);
            CK(cudaMalloc(&d_wei, sizeof(double) * cap_wei));
            need_drop = true;
        }
        CK(cudaEventRecord(ev_in, user));
        CK(cudaStreamWaitEvent(st, ev_in, 0));
        CK(cudaMemcpyAsync(d_wei, w_ei_steps, sizeof(double) * n_steps, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));
        ei_steps_set = true;
        ei_on = true;
        if (need_drop) drop_graph();
    }

    void set_muon_ns(int steps, int polish) override {
        if (steps < 0 || polish < 0 || steps + polish > 32) throw Error("Muon Newton-Schulz steps out of range");
        muon_steps = steps;
        muon_polish = polish;
        drop_graph();
    }

    void set_soap(double b1, double b2, double eps, int every, int max_dim, double mult) override {
        if (every <= 0) {
            soap_on = false;
            drop_graph();
            return;
        }
        if (h_mmats.empty()) throw Error("SOAP needs the Muon blocks (lst_set_muon) first");
        if (!(b1 >= 0 && b1 < 1 && b2 >= 0 && b2 < 1 && eps > 0)) throw Error("SOAP betas / eps out of range");
        if (max_dim > sp::TS) throw Error("SOAP max_dim above 64");
        std::vector<SoapMat> sm;
        std::vector<int2> units, eunits;
        std::vector<long> eyes;          // offsets of identity matrices: (offset, dim) pairs
        long w = 0, wbf = 0;
        const char* esp = std::getenv("LST_SOAP_SPLIT");
        const bool split = !esp || std::atoi(esp) != 0;
        const char* ebl = std::getenv("LST_SOAP_BLOCK");
        const bool blocked = ebl && std::atoi(ebl) != 0;
        std::vector<long> eyes_bf;
        for (const MuonMat& mt : h_mmats) {
            SoapMat s{};
            s.param = mt.param;
            s.m = std::min(mt.rows, mt.cols);
            s.n = std::max(mt.rows, mt.cols);
            if (s.m > sp::TS) throw Error("SOAP: a matrix's smaller side is above 64");
            const int md = max_dim > 0 ? max_dim : sp::TS;
            s.pm = s.m <= md ? 1 : 0;
            s.pn = s.n <= md ? 1 : 0;
            s.off = mt.off;
            s.xi = mt.tr ? mt.cs : mt.rs;
            s.xj = mt.tr ? mt.rs : mt.cs;
            const int nfull = s.n, nblk = (blocked && s.n > sp::TS) ? (s.n + sp::TS - 1) / sp::TS : 1;
            const long off0 = s.off;
            for (int bk = 0; bk < nblk; ++bk) {
            if (nblk > 1) {          // column block bk: its own m- and n-side factors
                s.off = off0 + (long)bk * sp::TS * s.xj;
                s.n = std::min(sp::TS, nfull - bk * sp::TS);
                s.pn = s.n <= md ? 1 : 0;
            }
            const long mm = (long)s.m * s.m, nn = (long)s.n * s.n, mn = (long)s.m * s.n;
            auto bump = [&](long sz) -> long { const long r = w; w += (sz + 3L) & ~3L; return r; };
            auto bump_bf = [&](long sz) -> long { const long r = wbf; wbf += (sz + 7L) & ~7L; return r; };
            s.Lm = bump(mm);
            s.Qm = bump(mm);
            eyes.push_back(s.Qm); eyes.push_back(s.m);
            s.Qm_bf = bump_bf(mm);
            eyes_bf.push_back(s.Qm_bf); eyes_bf.push_back(s.m);
            s.Bm = bump(mm);
            if (s.pn) {
                s.Rn = bump(nn);
                s.Qn = bump(nn);
                eyes.push_back(s.Qn); eyes.push_back(s.n);
                s.Qn_bf = bump_bf(nn);
                eyes_bf.push_back(s.Qn_bf); eyes_bf.push_back(s.n);
                s.Bn = bump(nn);
                s.Wn = bump(nn);
            } else {
                s.Qn_bf = 0;
            }
            s.nch = (split && !s.pn && s.n > sp::TS) ? (s.n + sp::TS - 1) / sp::TS : 1;
            if (s.nch > 1 && s.pm) {
                s.Lp = bump((long)s.nch * mm);
            }
            for (int c = 0; c < s.nch; ++c) units.push_back(make_int2((int)sm.size(), c));
            if (s.pm) eunits.push_back(make_int2((int)sm.size(), 0));
            if (s.pn) eunits.push_back(make_int2((int)sm.size(), 1));
            s.V = bump(mn);
            s.Mr = bump(mn);
            s.W1 = bump(std::max(mn, std::max(mm, s.pn ? nn : 0L)));
            s.W2 = bump(mn);
            s.W3 = bump(mn);
            s.W4 = bump(mn);
            sm.push_back(s);
            }
        }
        std::vector<T> h((size_t)w, T(0));
        for (size_t k = 0; k < eyes.size(); k += 2)
            for (long i = 0; i < eyes[k + 1]; ++i) h[(size_t)(eyes[k] + i * eyes[k + 1] + i)] = T(1);
        d_sws = alloc(nullptr, w);
        CK(cudaMemcpy(d_sws, h.data(), sizeof(T) * (size_t)w, cudaMemcpyHostToDevice));
        if constexpr (std::is_same<T, float>::value) {
            std::vector<uint16_t> hbf((size_t)std::max(1L, wbf), 0);
            for (size_t k = 0; k < eyes_bf.size(); k += 2)
                for (long i = 0; i < eyes_bf[k + 1]; ++i)
                    hbf[(size_t)(eyes_bf[k] + i * eyes_bf[k + 1] + i)] = 0x3f80u;  // BF16 1.0f
            d_sqbf = alloci<__nv_bfloat16>(std::max(1L, wbf));
            CK(cudaMemcpy(d_sqbf, hbf.data(), sizeof(uint16_t) * hbf.size(), cudaMemcpyHostToDevice));
        }
        if (soap_lpt && units.size() > 1) {
            std::stable_sort(units.begin(), units.end(), [&](const int2& a, const int2& b) {
                const SoapMat& sa = sm[a.x];
                const SoapMat& sb = sm[b.x];
                const int na = sa.nch > 1 ? std::min(sp::TS, sa.n - a.y * sp::TS) : sa.n;
                const int nb = sb.nch > 1 ? std::min(sp::TS, sb.n - b.y * sp::TS) : sb.n;
                const long wa = (long)sa.m * na * (1 + sa.pm + sa.pn);
                const long wb = (long)sb.m * nb * (1 + sb.pm + sb.pn);
                return wa > wb;
            });
        }
        d_smats = alloci<SoapMat>((long)sm.size());
        CK(cudaMemcpy(d_smats, sm.data(), sizeof(SoapMat) * sm.size(), cudaMemcpyHostToDevice));
        n_soap = (int)sm.size();
        n_sunits = (int)units.size();
        d_sunits = alloci<int2>((long)units.size());
        CK(cudaMemcpy(d_sunits, units.data(), sizeof(int2) * units.size(), cudaMemcpyHostToDevice));
        n_seunits = (int)eunits.size();
        d_seunits = alloci<int2>(std::max(1L, (long)eunits.size()));
        if (n_seunits)
            CK(cudaMemcpy(d_seunits, eunits.data(), sizeof(int2) * eunits.size(), cudaMemcpyHostToDevice));
        soap_smem = sizeof(T) * 2 * sp::TS * sp::LDV;
        soap_smem_a = sizeof(T) * 2 * sp::TS * sp::LDV;
        soap_smem_b = soap_smem;
        soap_smem_ok = true;
        long covered[LST_N_PARAMS] = {};
        for (const SoapMat& s : sm) {
            covered[s.param] += (long)s.m * s.n;
            if (s.m > sp::TS || (s.nch == 1 && s.n > sp::TS)) soap_smem_ok = false;
        }
        soap_ema_fuse = true;
        for (int i = 0; i < LST_N_PARAMS; ++i)
            if (muon_flag[i] && covered[i] != psize[i]) soap_ema_fuse = false;
        if (soap_smem > 48 * 1024) {
            CK(cudaFuncSetAttribute(k_soap_a<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem));
            CK(cudaFuncSetAttribute(k_soap_eig<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem));
            CK(cudaFuncSetAttribute(k_soap_ap<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem));
        }
        if (soap_smem_a > 32 * 1024) {
            if constexpr (std::is_same<T, float>::value) {
#ifndef LST_LEAN
                CK(cudaFuncSetAttribute(k_soap_a_smem_f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_a));
                CK(cudaFuncSetAttribute(k_soap_a_smem_tf32, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_a));
                CK(cudaFuncSetAttribute(k_soap_b_smem_f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_a));
                CK(cudaFuncSetAttribute(k_soap_eig_smem_f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_a));
                CK(cudaFuncSetAttribute(k_soap_ap_smem_f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_a));
                CK(cudaFuncSetAttribute(k_soap_a_smem_opt, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_a));
                CK(cudaFuncSetAttribute(k_soap_eig_smem_opt<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_a));
                CK(cudaFuncSetAttribute(k_soap_eig_smem_opt<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_a));
                CK(cudaFuncSetAttribute(k_soap_ap_smem_opt, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_a));
#endif
                CK(cudaFuncSetAttribute(k_soap_eig_qr_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_a));
            } else
                CK(cudaFuncSetAttribute(k_soap_a_smem_d, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_a));
        }
        if (soap_smem_b > 32 * 1024) {
#ifndef LST_LEAN
            CK(cudaFuncSetAttribute(k_soap_b<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_b));
#endif
            CK(cudaFuncSetAttribute(k_soap_eig<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_b));
            CK(cudaFuncSetAttribute(k_soap_ap<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)soap_smem_b));
        }
        soap_b1 = b1;
        soap_b2 = b2;
        soap_eps = eps;
        soap_every = every;
        soap_mult = mult;
        soap_tol = std::is_same<T, float>::value ? 1e-6 : 1e-7;
        {
            const char* ev = std::getenv("LST_SOAP_TOL");
            if (ev) soap_tol = std::atof(ev);
            const char* es = std::getenv("LST_SOAP_SWEEPS");
            soap_sweeps = es ? std::max(1, std::atoi(es)) : 30;
            const char* ej = std::getenv("LST_SOAP_JFUSE");
            soap_jfuse = ej && std::atoi(ej) == 0 ? 0 : 1;
            const char* esm = std::getenv("LST_SOAP_SMEM");
            soap_smem_on = !(esm && esm[0] == '0') && soap_smem_ok;
            const char* ee = std::getenv("LST_SOAP_EIG");
            soap_eig = ee ? (ee[0] == '1') : (std::is_same<T, float>::value && soap_smem_on && soap_jfuse);
        }
        soap_on = true;
        drop_graph();
    }

    void set_cautious(int on) override {
        adam_cautious = on != 0;
        drop_graph();
    }

    void set_muon_wd(double wd_mult) override {
        if (!(wd_mult >= 0.0)) throw std::runtime_error("lst_set_muon_wd: wd_mult must be >= 0");
        muon_wd = wd_mult;
        drop_graph();
    }

    // 0 FP32, 1 TF32, 2 bf16 for the GEMMs' cuBLAS compute type and the radial MLPs; -1 keeps a part as it is.
    // Takes effect at the next step (the step graph is captured again). The MLPs' tensor-core kernels must have
    // been set up at creation (LST_MATH / LST_MLP_TC); FP64 engines ignore the modes.
    void set_math(int gemm, int mlp) override {
        if (gemm < -1 || gemm > 2 || mlp < -1 || mlp > 2) throw Error("math modes: -1 keep, 0 fp32, 1 tf32, 2 bf16");
        if (!std::is_same<T, float>::value) return;
        if (gemm >= 0)
            g_f32_compute = gemm == 1 ? CUBLAS_COMPUTE_32F_FAST_TF32
                          : gemm == 2 ? CUBLAS_COMPUTE_32F_FAST_16BF : CUBLAS_COMPUTE_32F;
        if (mlp >= 0) {
            if (mlp > 0 && !mlp_tc_ready)
                throw Error("tensor-core radial MLPs were not set up at creation (LST_MATH / LST_MLP_TC)");
            mlp_tc = mlp == 1 ? mlpt::TF32 : mlp == 2 ? mlpt::BF16 : -1;
        }
        drop_graph();
    }

    void set_muon_norm(double beta) override {
        if (beta >= 1.0) throw Error("NorMuon beta must be < 1 (negative: off)");
        muon_nbeta = beta;
        if (d_mnv) CK(cudaMemset(d_mnv, 0, sizeof(T) * (size_t)std::max(1L, mnv_size)));
        drop_graph();
    }

    void set_muon(const LstMuonBlock* blocks, int n, double momentum, int nesterov, double scale) override {
        if (!have_params) throw Error("set the parameters before Muon");
        std::vector<MuonMat> mats;
        int flags[LST_N_PARAMS] = {};
        long wsz = 0, nvsz = 0;
        for (int k = 0; k < n; ++k) {
            const LstMuonBlock& bl = blocks[k];
            if (bl.param < 0 || bl.param >= LST_N_PARAMS || bl.batch < 1 || bl.rows < 1 || bl.cols < 1)
                throw Error("Muon block " + std::to_string(k) + ": bad parameter or shape");
            for (int bi = 0; bi < bl.batch; ++bi) {
                MuonMat mt{bl.param, bl.rows, bl.cols, bl.rows > bl.cols ? 1 : 0,
                           bl.offset + (long)bi * bl.batch_stride, bl.row_stride, bl.col_stride, wsz, nvsz};
                const long last = mt.off + (long)(bl.rows - 1) * bl.row_stride + (long)(bl.cols - 1) * bl.col_stride;
                if (mt.off < 0 || last >= psize[bl.param] || bl.row_stride < 0 || bl.col_stride < 0)
                    throw Error("Muon block " + std::to_string(k) + " outside its tensor");
                const long mm = std::min(bl.rows, bl.cols), nn = std::max(bl.rows, bl.cols);
                wsz += 2 * mm * nn + 2 * mm * mm;
                nvsz += bl.rows;
                mats.push_back(mt);
            }
            flags[bl.param] = 1;
        }
        if (!mats.empty()) {
            if (!d_mdir) d_mdir = alloc(nullptr, ptotal);
            d_mws = alloc(nullptr, wsz);
            d_mnv = alloc(nullptr, nvsz);
            mnv_size = nvsz;
            d_mmats = alloci<MuonMat>((long)mats.size());
            CK(cudaMemcpy(d_mmats, mats.data(), sizeof(MuonMat) * mats.size(), cudaMemcpyHostToDevice));
        }
        n_mmats = (int)mats.size();
        h_mmats = mats;
        soap_on = false;
        muon_fast = false;
        if constexpr (std::is_same<T, float>::value) {
            const char* ev = std::getenv("LST_MUON_TILED");
            muon_fast = !(ev && ev[0] == '0');
            for (const MuonMat& mt : mats) muon_fast = muon_fast && std::min(mt.rows, mt.cols) <= mu::TS;
            if (muon_fast)
                CK(cudaFuncSetAttribute(k_muon_ns_f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)mu::SMEM));
        }
        for (int i = 0; i < LST_N_PARAMS; ++i) muon_flag[i] = flags[i];
        muon_mu = momentum;
        muon_nesterov = nesterov ? 1 : 0;
        muon_scale = scale;
        rebuild_adam_idx();
        if (have_optim) {
            for (int i = 0; i < LST_N_PARAMS; ++i) {
                h_pdesc[i].muon = muon_flag[i];
                h_pdesc[i].dir = d_mdir ? d_mdir + poff[i] : nullptr;
            }
            CK(cudaMemcpy(d_pdesc, h_pdesc.data(), sizeof(PDesc<T>) * LST_N_PARAMS, cudaMemcpyHostToDevice));
        }
        drop_graph();
    }

    void rebuild_adam_idx() {
        std::vector<int> idx;
        idx.reserve(LST_N_PARAMS);
        for (int i = 0; i < LST_N_PARAMS; ++i)
            if (!muon_flag[i]) idx.push_back(i);
        n_adam_idx = (int)idx.size();
        if (!d_adam_idx) d_adam_idx = alloci<int>(LST_N_PARAMS);
        if (n_adam_idx > 0)
            CK(cudaMemcpy(d_adam_idx, idx.data(), sizeof(int) * n_adam_idx, cudaMemcpyHostToDevice));
    }

    void set_optim(const LstOptim* o) override {
        if (!have_params) throw Error("set the parameters before the optimizer");
        opt = *o;
        std::vector<PDesc<T>> d(LST_N_PARAMS);
        std::vector<float*> steps(LST_N_PARAMS);
        for (int i = 0; i < LST_N_PARAMS; ++i) {
            d[i] = PDesc<T>{raw[i], graw + poff[i], (T*)o->exp_avg[i], (T*)o->exp_avg_sq[i], (T*)o->max_exp_avg_sq[i],
                            (T*)o->ema[i], o->step[i], psize[i], o->group[i], muon_flag[i],
                            d_mdir ? d_mdir + poff[i] : nullptr};
            steps[i] = o->step[i];
        }
        h_pdesc = d;
        CK(cudaMemcpy(d_pdesc, d.data(), sizeof(PDesc<T>) * LST_N_PARAMS, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_steps, steps.data(), sizeof(float*) * LST_N_PARAMS, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_wd, o->weight_decay, sizeof(double) * 16, cudaMemcpyHostToDevice));
        rebuild_adam_idx();
        have_ema = o->ema[0] != nullptr;
        have_optim = true;
        drop_graph();
    }

    void begin_epoch(const LstBatches* bt, const double* lr, const double* lw, const double* ema, bool per_step,
                     cudaStream_t user) override {
        if (bt->n > cap_steps) {
            CK(cudaDeviceSynchronize());
            if (d_ema) cudaFree(d_ema);
            if (d_losses) cudaFree(d_losses);
            if (d_fids) cudaFree(d_fids);
            if (d_lr) cudaFree(d_lr);
            if (d_lw) cudaFree(d_lw);
            cap_steps = std::max(bt->n, 2 * cap_steps);
            CK(cudaMalloc(&d_lr, sizeof(double) * 16 * cap_steps));
            CK(cudaMalloc(&d_lw, sizeof(T) * 3 * cap_steps));
            CK(cudaMalloc(&d_ema, sizeof(double) * cap_steps));
            CK(cudaMalloc(&d_losses, sizeof(T) * cap_steps));
            CK(cudaMalloc(&d_fids, sizeof(int) * (size_t)cap_steps * sh.B));
            drop_graph();
        }
        n_epoch = bt->n;
        eval_views_valid = false;
        const int G = opt.n_groups;
        h_lr.assign(16L * bt->n, 0.0);
        h_lw.assign(3L * bt->n, T(0));
        for (int i = 0; i < bt->n; ++i) {
            for (int g = 0; g < G; ++g) h_lr[16L * i + g] = per_step ? lr[(long)i * G + g] : lr[g];
            for (int k = 0; k < 3; ++k) h_lw[3L * i + k] = T(per_step ? lw[3L * i + k] : lw[k]);
        }
        // stage on the library stream after the caller's work
        CK(cudaEventRecord(ev_in, user));
        CK(cudaStreamWaitEvent(st, ev_in, 0));
        CK(cudaMemcpyAsync(d_fids, bt->frame_ids, sizeof(int) * (size_t)bt->n * sh.B, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_lr, h_lr.data(), sizeof(double) * h_lr.size(), cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_lw, h_lw.data(), sizeof(T) * h_lw.size(), cudaMemcpyHostToDevice, st));
        if (ema) CK(cudaMemcpyAsync(d_ema, ema, sizeof(double) * bt->n, cudaMemcpyHostToDevice, st));
        CK(cudaMemsetAsync(d_step, 0, 2 * sizeof(int), st));
        CK(cudaStreamSynchronize(st));   // the host arrays above are the caller's
    }

    // ------------------------------------------------------------------ step pieces
    bool par() const { return (par_env & par_mask) && fused_mlp && !mlp_split && std::is_same<T, float>::value; }
    // s2 waits for the work queued on st so far / st waits for the work queued on s2 so far
    void fork(int k) {
        if (!par()) return;
        CK(cudaEventRecord(evf[k], st));
        CK(cudaStreamWaitEvent(s2, evf[k], 0));
    }
    void join(int k) {
        if (!par()) return;
        CK(cudaEventRecord(evf[k], s2));
        CK(cudaStreamWaitEvent(st, evf[k], 0));
    }
    // weight-gradient lane: s3 waits for the main stream so far, then launches go to s3 until wg_end()
    cudaEvent_t wg_record() {
        if (!wg_side) return nullptr;
        cudaEvent_t e = evw[nevw++ % 32];
        CK(cudaEventRecord(e, st));
        return e;
    }
    void wg_begin_ev(cudaEvent_t e) {
        if (!wg_side) return;
        if (e) CK(cudaStreamWaitEvent(s3, e, 0));
        wg_ms = st;
        st = s3;
    }
    void wg_begin() {
        if (!wg_side) return;
        cudaEvent_t e = evw[nevw++ % 32];
        CK(cudaEventRecord(e, st));
        CK(cudaStreamWaitEvent(s3, e, 0));
        wg_ms = st;
        st = s3;
    }
    void wg_end() {
        if (!wg_side) return;
        st = wg_ms;
    }
    // the main stream waits for the weight-gradient lane so far
    void wg_join() {
        if (!wg_side) return;
        cudaEvent_t e = evw[nevw++ % 32];
        CK(cudaEventRecord(e, s3));
        CK(cudaStreamWaitEvent(st, e, 0));
    }
    // queue on the side stream within this scope (no-op when the branches are off)
    struct OnSide {
        Engine* e;
        cudaStream_t ms;
        explicit OnSide(Engine* e_) : e(e_), ms(e_->st) {
            if (e->par()) {
                e->st = e->s2;
            }
        }
        ~OnSide() {
            e->st = ms;
        }
    };

    cublasHandle_t ensure_cb_for(cudaStream_t s) {
        auto init_one = [&](cublasHandle_t& h, void*& ws, cudaStream_t ss) -> cublasHandle_t {
            if (!h) {
                CB(cublasCreate(&h));
                CB(cublasSetStream(h, ss));
                if (cb_ws_bytes) CK(cudaMalloc(&ws, cb_ws_bytes));
                CB(cublasSetWorkspace(h, ws, cb_ws_bytes));
                CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
            }
            return h;
        };
        if (s == s2) return init_one(cb2, cb2_ws, s2);
        if (s3 && s == s3) return init_one(cb3, cb3_ws, s3);
        return init_one(cb, cb_ws, st_main);
    }

    void lgemm(bool ta, bool tb, int M, int Nn, int Kk, T al, const T* A, int lda, const T* B, int ldb, T be, T* Cc,
               int ldc) {
        if (gemm_log < 0) {
            const char* eg = std::getenv("LST_GEMM_LOG");
            gemm_log = eg && eg[0] == '1' ? 400 : 0;
        }
        if (gemm_log > 0) {
            --gemm_log;
            std::fprintf(stderr, "GEMM %d %d %d %d %d\n", (int)ta, (int)tb, M, Nn, Kk);
        }
        if (ta && !tb && Kk >= 256 && (long)M * Nn <= 64L * 320) {
            const T* As[1] = {A};
            const T* Bs[1] = {B};
            T* Cs[1] = {Cc};
            int Ks[1] = {Kk};
            if (gemm_tn(1, As, Bs, Cs, Ks, M, Nn, lda, ldb, ldc, al, be)) return;
        }
        gemm<T>(ensure_cb_for(st), ta, tb, M, Nn, Kk, al, A, lda, B, ldb, be, Cc, ldc);
    }
    // C_s (+)= al A_s^T B_s for up to four segments of one shape; false if not taken (cuBLAS then)
    bool gemm_tn(int nseg, const T* const* A, const T* const* B, T* const* Cs, const int* Ks, int M, int Nn, int lda,
                 int ldb, int ldc, T al, T be) {
        if (!own_tn || st == s2 || (s3 && st == s3)) return false;    // one split-K workspace: main stream only
        const int mt = (M + 63) / 64, nt = (Nn + 63) / 64;
        long ktot = 0;
        for (int i = 0; i < nseg; ++i) ktot += Ks[i];
        int budget = std::max(nseg, 4 * nsm / (mt * nt));
        TnArgs<T> a{};
        for (;;) {
            int z = 0;
            for (int i = 0; i < nseg; ++i) {
                int nc = (int)std::lround((double)budget * Ks[i] / std::max(1L, ktot));
                nc = std::max(1, std::min(nc, (Ks[i] + 63) / 64));
                int kc = (Ks[i] + nc - 1) / nc;
                kc = std::max(32, (kc + 31) / 32 * 32);
                nc = std::max(1, (Ks[i] + kc - 1) / kc);
                a.s[i] = TnSeg<T>{A[i], B[i], Cs[i], Ks[i], kc, nc, z};
                z += nc;
            }
            if ((long)z * M * Nn <= tn_cap) break;
            if (budget <= nseg) return false;
            budget = std::max(nseg, budget / 2);
        }
        a.nseg = nseg; a.M = M; a.N = Nn; a.lda = lda; a.ldb = ldb; a.ldc = ldc;
        a.alpha = al; a.beta = be; a.ws = tn_ws;
        const int zt = a.s[nseg - 1].chunk0 + a.s[nseg - 1].nchunk;
        k_gemm_tn<T><<<dim3(nt, mt, zt), 256, 0, st>>>(a);
        k_gemm_tn_reduce<T><<<nblk((long)M * Nn * nseg), TPB, 0, st>>>(a);
        KCHECK();
        return true;
    }

    // the scaled copies of prepare_views and of the raw gradients, as segment tables for k_scale_multi
    Seg<T>* d_vseg = nullptr;
    Seg<T>* d_gseg = nullptr;
    GfTile* d_gftiles = nullptr;
    int n_vseg = 0, n_gseg = 0, n_gftiles = 0, cap_gftiles = 0;
    void build_tables() {
        const T rC = T(1) / std::sqrt(T(C));
        T fin[4] = {T(1) / std::sqrt(T(NB)), T(1) / std::sqrt(T(H)), T(1) / std::sqrt(T(H)), T(1) / std::sqrt(T(H))};
        std::vector<std::pair<int, std::pair<T*, T*>>> a;   // param index, (view, view grad)
        std::vector<T> f;
        auto add = [&](int i, T* v, T* gv, T fac) {
            a.push_back({i, {v, gv}});
            f.push_back(fac);
        };
        add(0, emb, gemb, T(1) / std::sqrt(T(S)));
        add(1, wup0, gwup0, rC);
        for (int k = 0; k < 4; ++k) {
            add(2 + k, mlpW[0][k], gmlp[0][k], fin[k]);
            add(9 + k, mlpW[1][k], gmlp[1][k], fin[k]);
        }
        add(6, wl0, gwl0, rC / T(md.avg0));
        add(7, ws0, gws0, T(1) / std::sqrt(T(C) * T(S)));
        add(13, wl1, gwl1, rC / T(md.avg1));
        add(18, wp0, gwp0, rC);
        add(22, wp1, gwp1, rC);
        add(24, wro1, gwro1, rC);
        add(25, wro2, wro2s, T(1) / std::sqrt(T(hid)));
        std::vector<Seg<T>> vs, gs;
        for (size_t k = 0; k < a.size(); ++k) {
            int i = a[k].first;
            vs.push_back(Seg<T>{a[k].second.first, raw[i], f[k], psize[i]});
            gs.push_back(Seg<T>{graw + poff[i], a[k].second.second, f[k], psize[i]});
        }
        if (!d_vseg) {
            d_vseg = reinterpret_cast<Seg<T>*>(alloci<char>(sizeof(Seg<T>) * 32));
            d_gseg = reinterpret_cast<Seg<T>*>(alloci<char>(sizeof(Seg<T>) * 32));
        }
        n_vseg = (int)vs.size();
        n_gseg = (int)gs.size();
        CK(cudaMemcpy(d_vseg, vs.data(), sizeof(Seg<T>) * vs.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_gseg, gs.data(), sizeof(Seg<T>) * gs.size(), cudaMemcpyHostToDevice));
        // Wave 36: build uniform 1024-float tile table for k_grad_finalize_sumsq_flat (~185 CTAs, zero stragglers)
        std::vector<GfTile> gft;
        gft.reserve(256);
        for (int s_idx = 0; s_idx < n_gseg; ++s_idx) {
            const long nn = gs[s_idx].n;
            for (long off = 0; off < nn; off += 1024) {
                const int cnt = (int)std::min<long>(1024, nn - off);
                gft.push_back(GfTile{0, (uint8_t)s_idx, 0, (int)off, cnt});
            }
        }
        const int sk_rows = S * K;
        const int rows_per_tile = (C == 64) ? 16 : std::max(1, 1024 / C);
        for (int sub = 0; sub < 2; ++sub) {
            for (int off = 0; off < sk_rows; off += rows_per_tile) {
                const int cnt = std::min(rows_per_tile, sk_rows - off);
                gft.push_back(GfTile{1, (uint8_t)sub, 0, off, cnt});
            }
        }
        const long ztot = (long)C * ZW;
        for (long off = 0; off < ztot; off += 1024) {
            const int cnt = (int)std::min<long>(1024, ztot - off);
            gft.push_back(GfTile{2, 0, 0, (int)off, cnt});
        }
        n_gftiles = (int)gft.size();
        if (n_gftiles > cap_gftiles) {
            if (d_gftiles) cudaFree(d_gftiles);
            CK(cudaMalloc(&d_gftiles, sizeof(GfTile) * n_gftiles));
            cap_gftiles = n_gftiles;
        }
        CK(cudaMemcpy(d_gftiles, gft.data(), sizeof(GfTile) * n_gftiles, cudaMemcpyHostToDevice));
    }

    // ---- one-launch tables of the weight-only products (LST_MULTI_GEMM=1)
    mgemm::Table<T>* d_mg_pv = nullptr;       // prepare_views (full)
    mgemm::Table<T>* d_mg_pv_t0 = nullptr;    // Wave 35: T0 = emb @ wup0 only (1 tile, zero contention with mlp_fwd(0)!)
    mgemm::Table<T>* d_mg_pv_mq = nullptr;    // Wave 35: M0[0..3] + Q deferred into conv_fwd(0) shadow
    mgemm::Table<T>* d_mg_vg = nullptr;       // view gradients
    mgemm::Table<T>* d_mg_vg_early = nullptr; // M0s + Qs view gradients (overlap with k_slot layer 0!)
    mgemm::Table<T>* d_mg_vg_t0 = nullptr;    // T0s-only view gradients (gemb, gwup0)
    mgemm::Table<T>* d_mg_vg_all = nullptr;   // Wave 37: vg_early + vg_t0 merged (1 launch on s3 -> frees s2 for k_mlp_dw4_dual!)
    mgemm::Table<T>* d_mg_vg_q = nullptr;     // Wave 41: Qs-only view gradients (gwp0, gWcat) for early s2 dispatch
    mgemm::Table<T>* d_mg_vg_m0 = nullptr;    // Wave 41: M0s-only view gradients (gwl0[0..3], gws0[0..3])
    int mg_pv_tiles = 0, mg_pv_t0_tiles = 0, mg_pv_mq_tiles = 0;
    int mg_vg_tiles = 0, mg_vg_early_tiles = 0, mg_vg_t0_tiles = 0, mg_vg_all_tiles = 0;
    int mg_vg_q_tiles = 0, mg_vg_m0_tiles = 0;
    struct MgItem {
        bool ta, tb;
        int M, N, K;
        const T* A;
        int lda;
        const T* B;
        int ldb;
        T* C;
        int ldc;
        int sk = 1;
    };
    mgemm::Table<T>* mg_table(const std::vector<MgItem>& v, int& tiles) {
        mgemm::Table<T> h{};
        if ((int)v.size() > mgemm::MAXG) throw Error("multi GEMM table too long");
        tiles = 0;
        const bool use_32 = fast_mgemm || !nops_opt || (mgemm_v3 == 2);
        const int tm_v = use_32 ? mgemm::TS : mgemm::TM2;
        const int tn_v = use_32 ? mgemm::TS : mgemm::TN2;
        for (size_t i = 0; i < v.size(); ++i) {
            const MgItem& q = v[i];
            const int sk = (!use_32 && mgemm_v3 == 0 && q.sk > 1) ? q.sk : 1;
            h.d[i] = mgemm::Desc<T>{q.A, q.B, q.C, q.M, q.N, q.K, q.lda, q.ldb, q.ldc, q.ta ? 1 : 0, q.tb ? 1 : 0,
                                    tiles, sk};
            tiles += ((q.M + tm_v - 1) / tm_v) * ((q.N + tn_v - 1) / tn_v) * sk;
        }
        h.n = (int)v.size();
        mgemm::Table<T>* d = nullptr;
        CK(cudaMalloc(&d, sizeof(h)));
        CK(cudaMemcpy(d, &h, sizeof(h), cudaMemcpyHostToDevice));
        return d;
    }
    void launch_mgemm(const mgemm::Table<T>* tbl, int tiles) {
#ifndef LST_LEAN
        if (fast_mgemm)
            mgemm::k_multi_gemm_f<T><<<tiles, 64, 0, st>>>(tbl);
        else if (nops_opt && mgemm_v3 == 2)
            mgemm::k_multi_gemm_f4<T><<<tiles, 64, 0, st>>>(tbl);
        else if (nops_opt && mgemm_v3 == 1)
            mgemm::k_multi_gemm_v3<T><<<tiles, 64, 0, st>>>(tbl);
        else
#endif
        if (nops_opt)
            mgemm::k_multi_gemm_v2<T><<<tiles, 64, 0, st>>>(tbl);
        else
            mgemm::k_multi_gemm<T><<<tiles, 256, 0, st>>>(tbl);
    }
    void build_mg() {
        if (!multi_gemm) return;
        std::vector<MgItem> pv, vg;
        pv.push_back({false, false, S, C, C, emb, C, wup0, C, T0, C, 1});
        for (int l = 0; l < 4; ++l)
            pv.push_back({false, false, C, SC, C, wl0 + (long)l * C * C, C, ws0 + (long)l * C * SC, SC,
                          M0 + (long)l * C * SC, SC, 1});
        pv.push_back({false, false, C, ZW, C, wp0, C, Wcat, ZW, Q, ZW, 1});
        vg.push_back({false, true, S, C, C, T0s, C, wup0, C, gemb, C, 1});
        vg.push_back({true, false, C, C, S, emb, C, T0s, C, gwup0, C, 1});
        for (int l = 0; l < 4; ++l) {
            vg.push_back({false, true, C, C, SC, M0s + (long)l * C * SC, SC, ws0 + (long)l * C * SC, SC,
                          gwl0 + (long)l * C * C, C, 1});
            vg.push_back({true, false, C, SC, C, wl0 + (long)l * C * C, C, M0s + (long)l * C * SC, SC,
                          gws0 + (long)l * C * SC, SC, 1});
        }
        vg.push_back({false, true, C, C, ZW, Qs, ZW, Wcat, ZW, gwp0, C, 1});
        vg.push_back({true, false, C, ZW, C, wp0, C, Qs, ZW, gWcat, ZW, 1});
        d_mg_pv = mg_table(pv, mg_pv_tiles);
        std::vector<MgItem> pv_t0(pv.begin(), pv.begin() + 1);
        std::vector<MgItem> pv_mq(pv.begin() + 1, pv.end());
        d_mg_pv_t0 = mg_table(pv_t0, mg_pv_t0_tiles);
        d_mg_pv_mq = mg_table(pv_mq, mg_pv_mq_tiles);
        d_mg_vg = mg_table(vg, mg_vg_tiles);
        std::vector<MgItem> vg_t0(vg.begin(), vg.begin() + 2);
        // Wave 34: LPT-first ordering + Split-K on K=257 (gwp0, sk=4) and K=192 (gwl0[0..3], sk=3) in vg_early!
        const bool do_sk = std::is_same<T, float>::value && zero_fused && mgemm_sk && nops_opt && !fast_mgemm &&
                           mgemm_v3 == 0 && C == 64;
        const int sk_wp0 = (do_sk && ZW > 64) ? std::min(4, (ZW + 63) / 64) : 1;
        const int sk_wl0 = (do_sk && SC > 64) ? std::min(4, (SC + 63) / 64) : 1;
        std::vector<MgItem> vg_early;
        vg_early.push_back({false, true, C, C, ZW, Qs, ZW, Wcat, ZW, gwp0, C, sk_wp0});
        for (int l = 0; l < 4; ++l)
            vg_early.push_back({false, true, C, C, SC, M0s + (long)l * C * SC, SC, ws0 + (long)l * C * SC, SC,
                                gwl0 + (long)l * C * C, C, sk_wl0});
        vg_early.push_back({true, false, C, ZW, C, wp0, C, Qs, ZW, gWcat, ZW, 1});
        for (int l = 0; l < 4; ++l)
            vg_early.push_back({true, false, C, SC, C, wl0 + (long)l * C * C, C, M0s + (long)l * C * SC, SC,
                                gws0 + (long)l * C * SC, SC, 1});
        d_mg_vg_t0 = mg_table(vg_t0, mg_vg_t0_tiles);
        d_mg_vg_early = mg_table(vg_early, mg_vg_early_tiles);
        std::vector<MgItem> vg_q, vg_m0;
        vg_q.push_back({false, true, C, C, ZW, Qs, ZW, Wcat, ZW, gwp0, C, sk_wp0});
        vg_q.push_back({true, false, C, ZW, C, wp0, C, Qs, ZW, gWcat, ZW, 1});
        for (int l = 0; l < 4; ++l)
            vg_m0.push_back({false, true, C, C, SC, M0s + (long)l * C * SC, SC, ws0 + (long)l * C * SC, SC,
                             gwl0 + (long)l * C * C, C, sk_wl0});
        for (int l = 0; l < 4; ++l)
            vg_m0.push_back({true, false, C, SC, C, wl0 + (long)l * C * C, C, M0s + (long)l * C * SC, SC,
                             gws0 + (long)l * C * SC, SC, 1});
        d_mg_vg_q = mg_table(vg_q, mg_vg_q_tiles);
        d_mg_vg_m0 = mg_table(vg_m0, mg_vg_m0_tiles);
        std::vector<MgItem> vg_all = vg_early;
        vg_all.insert(vg_all.end(), vg_t0.begin(), vg_t0.end());
        d_mg_vg_all = mg_table(vg_all, mg_vg_all_tiles);
    }

    void prepare_views_scale() {
        const int gx = std::is_same<T, float>::value ? 4 : 16;
        k_scale_multi<T><<<dim3(gx, n_vseg), TPB, 0, st>>>(d_vseg);
    }
    void prepare_views_w() {
        if (prod_dedup_active()) {
            k_cat_w_u_pair<T><<<dim3(nblk((long)S * C), 2), TPB, 0, st>>>(W0, raw[15], raw[16], raw[17], W1, raw[19],
                                                                          raw[20], raw[21], S, K, md.K3, md.K2, C);
        } else {
            k_cat_w<T><<<nblk((long)S * K * C), TPB, 0, st>>>(W0, raw[15], raw[16], raw[17], S, K, md.K3, md.K2, C, 0);
            k_cat_w<T><<<nblk((long)S * K * C), TPB, 0, st>>>(W1, raw[19], raw[20], raw[21], S, K, md.K3, md.K2, C, 0);
        }
        KCHECK();
    }
    void prepare_views_wcat() {
        const T rC = T(1) / std::sqrt(T(C));
        k_cat_wcat<T><<<nblk((long)C * ZW), TPB, 0, st>>>(Wcat, raw[23], raw[8], raw[14], rC, rC,
                                                         T(1) / std::sqrt(T(C) * T(S)), C, ZW, 0);
        KCHECK();
    }
    void prepare_views_early() {
        prepare_views_wcat();
        if (d_mg_pv) {
            launch_mgemm(d_mg_pv, mg_pv_tiles);
            KCHECK();
            return;
        }
        lgemm(false, false, S, C, C, T(1), emb, C, wup0, C, T(0), T0, C);
        for (int l = 0; l < 4; ++l)
            lgemm(false, false, C, SC, C, T(1), wl0 + (long)l * C * C, C, ws0 + (long)l * C * SC, SC, T(0),
                  M0 + (long)l * C * SC, SC);
        lgemm(false, false, C, ZW, C, T(1), wp0, C, Wcat, ZW, T(0), Q, ZW);
    }
    void prepare_views() {
        prepare_views_scale();
        prepare_views_w();
        prepare_views_early();
    }
    void pack_segg_w() {
        if constexpr (std::is_same<T, float>::value) {
            const int ah = act_half_mode();
            if (segg_h16 && ah > 0 && segg_v2 && m0_species && C == 64 && M0_h && Wl1_h) {
                if (ah == 1)
                    segg::k_pack_segg_w_h<__nv_bfloat16><<<4 * S + 4, 256, 0, st>>>(
                        M0, wl1, reinterpret_cast<__nv_bfloat16*>(M0_h), reinterpret_cast<__nv_bfloat16*>(Wl1_h), S);
                else
                    segg::k_pack_segg_w_h<__half><<<4 * S + 4, 256, 0, st>>>(
                        M0, wl1, reinterpret_cast<__half*>(M0_h), reinterpret_cast<__half*>(Wl1_h), S);
                KCHECK();
            }
        }
    }
    bool zip_ga_slot_ok(int layer = -1) const {
        const bool sopt_ok = (slot_opt == 7 || slot_opt == 9 || slot_opt == 10 || (slot_opt == 15 && layer != 0));
        return std::is_same<T, float>::value && zip_ga && GA_buf && act_half_mode() > 0 && segg_v2 &&
               mspl_poly_mode() == 2 && mspl_poly_bwd() && sopt_ok &&
               slot_fuse_xd_active() && m0_species && !(edge_h2 & 2) && !slot_kb_active() && !slot_scalar && C == 64 &&
               !in_debug;
    }
    int slot_hfma2_mode(int layer = -1) const {
        return (slot_hfma2 > 0 && slot_opt == 7 && zip_ga_slot_ok(layer) && edge_ksort_active() && y3p_mode() >= 2 &&
                act_half_mode() == 2)
                   ? slot_hfma2
                   : 0;
    }
    int slot_zip_mode(int layer = -1) const {
        if (!zip_ga_slot_ok(layer)) return 0;
        if (slot_opt == 9 || slot_opt == 10) return 4;
        if (slot_hfma2_mode(layer) == 2) return 0;  // seg_gemm writes r_A; k_ga_pack_fp8_l packs per-(n,l) FP8 GA_buf
        return 1;
    }
    bool zip_ga_force_ok() const {
        return std::is_same<T, float>::value && zip_ga && !force_dir && force_opt != 2 && GA_buf && act_half_mode() > 0 && segg_v2 &&
               mspl_poly_mode() == 2 && force_pair_active() && !force_kb_active() && (edge_h2 & 4) && m0_species &&
               C == 64 && !in_debug;
    }
    bool t0s_pair_ok() const {
        return std::is_same<T, float>::value && t0s_pair && S <= 4 && C == 64 && cz_active() && act_half_mode() > 0 &&
               mspl_poly_mode() == 2 && mspl_poly_bwd() && edge_ksort_active() && !in_debug && !conv_c64 &&
               (edge_h2 & 1) && slot_opt == 7 && zip_ga_slot_ok(0) && slot_fuse_xd_active() && !(edge_h2 & 2) &&
               !slot_kb_active() && !slot_scalar;
    }
    bool t0s_dir_ok() const {
        if (t0s_pair_ok()) return true;
        const bool slot_l0_dir = (slot_opt == 11 || slot_opt == 12 || slot_opt == 13 || slot_opt == 14 || slot_opt == 15);
        return std::is_same<T, float>::value && t0s_dir && S <= 4 && C == 64 && cz_active() && act_half_mode() > 0 &&
               mspl_poly_mode() == 2 && mspl_poly_bwd() && edge_ksort_active() && !in_debug && conv_c64 &&
               (edge_h2 & 1) && slot_l0_dir && slot_fuse_xd_active() && !(edge_h2 & 2) && !slot_kb_active() && !slot_scalar;
    }
    bool force_pair2_ok() const {
        return std::is_same<T, float>::value && force_opt == 2 && !force_dir && C == 64 && act_half_mode() > 0 &&
               mspl_poly_mode() == 2 && edge_ksort_active() && (edge_h2 & 4) && force_pair_active() &&
               !force_kb_active() && !in_debug;
    }
    bool force_dir2_ok() const {
        return std::is_same<T, float>::value && force_dir == 2 && C == 64 && act_half_mode() > 0 &&
               mspl_poly_mode() == 2 && edge_ksort_active() && (edge_h2 & 4) && force_pair_active() &&
               !force_kb_active() && !in_debug;
    }
    bool scot_active() const {
        return std::is_same<T, float>::value && scot && loss_par && grad_fuse && prod_dedup_active() && pair_basis &&
               !in_debug;
    }

    bool prod_dedup_active() const { return prod_dedup && prod_direct && !in_debug && K == POLY_K; }
    bool nm_active() const { return node_major && m0_species && pair_basis && prod_direct && !in_debug; }
    bool wbal_active() const { return (wgrad_bal || nm_active()) && C % segg::TM == 0; }
    bool cz_active() const { return compact_z && m0_species && !in_debug && C % 64 == 0 && hid % 4 == 0 && hid <= 64; }
    bool mspl_active() const { return mlp_spline > 0 && pair_basis && !in_debug; }
    bool mspl_fused() const { return mspl_active() && mlp_spline_fused; }
    int mspl_poly_mode() const {
        if (!mspl_fused() || !nm_active() || conv_es != 2 || slot_es != 2 || 2 * C > 256) return 0;
        if (std::is_same<T, float>::value && spline_half) return 2;
        return mlp_spline_poly >= 1 ? 1 : 0;
    }
    bool mspl_poly_fwd() const { return mspl_poly_mode() > 0; }
    bool edge_ksort_active() const {
        return edge_ksort && mspl_fused() && nm_active() && !in_debug && ks_ready && C == 64;
    }
    bool slot_kb_active() const {
        return slot_kb && edge_ksort_active() && slot_fuse_xd_active() && ks_max_tiles > 0;
    }
    bool force_kb_active() const {
        return force_kb && edge_ksort_active() && force_pair_active() && ks_max_tiles > 0;
    }
    bool mspl_poly_bwd() const {
        return mspl_fused() && (mlp_spline_poly >= 2 || slot_kb_active()) && nm_active() && conv_es == 2 &&
               slot_es == 2 && 2 * C <= 256;
    }
    bool sdx_active() const { return mspl_fused() && conv_sender_dx && nm_active() && conv_es == 2 && 2 * C <= 256; }
    bool sng_active() const { return mspl_fused() && slot_no_gather && nm_active() && slot_es == 2 && 2 * C <= 256; }
    bool slot_fuse_xd_active() const {
        return mspl_fused() && slot_fuse_xd && nm_active() && !slot_no_gather && !slot_bal &&
               (slot_es == 2 || slot_es == 4) && slot_es * C <= 256;
    }
    bool mlp_tiny_active() const {
        return std::is_same<T, float>::value && fused_mlp && mspl_active() && mlp_spline_small && mlp_tiny > 0 &&
               mlp_tc < 0 && mlp_spline <= 256;
    }
    int y3p_mode() const {
        const bool slot_y3_ok = (slot_opt >= 11 && slot_opt <= 15) ||
                                ((slot_opt == 7 || slot_opt == 9 || slot_opt == 10) && zip_ga);
        if (!(std::is_same<T, float>::value && y3p > 0 && pair_basis && !in_debug && act_half_mode() > 0 &&
              mspl_poly_mode() == 2 && mspl_poly_bwd() && (edge_h2 & 1) && slot_y3_ok && segg_v2 && conv_fact &&
              edge_ksort_active() && slot_fuse_xd_active()))
            return 0;
        if (y3p >= 3 && act_half_mode() == 2 && conv_hfma2 == 2 && slot_hfma2 > 0 && slot_opt == 7 && zip_ga &&
            !conv_opt && !conv_c64)
            return 3;
        return y3p >= 2 ? 2 : 1;
    }
    bool y3p_active() const { return y3p_mode() > 0; }
    bool force_pair_active() const {
        return mspl_fused() && force_pair && nm_active() && !force_2pass && !in_debug && (C == 64 || C == 32);
    }
    int act_half_mode() const {
        if (!std::is_same<T, float>::value || act_half <= 0 || C != 64 || in_debug) return 0;
        if (!nm_active() || !mspl_fused() || mspl_poly_mode() < 1 || !sdx_active() || !slot_fuse_xd_active()) return 0;
        if (!prod_dedup_active() || !force_pair_active() || !force_64t || !wbal_active() || !wgrad_opt) return 0;
        if (conv_es != 2 || slot_es != 2 || slot_scalar) return 0;
        return act_half;
    }
    T sp_h() const { return T(md.r_max - sp_r_lo) / T(mlp_spline); }
    T sp_invh() const { return T(mlp_spline) / T(md.r_max - sp_r_lo); }
    float4 shell_rcut4(int t) const {
        return make_float4(shell_rcut[t][0], shell_rcut[t][1], shell_rcut[t][2], shell_rcut[t][3]);
    }
    int shell_kcut(int t, int l) const {
        const float rc = shell_rcut[t][l];
        if (rc < 0.f) return mlp_spline;
        if (rc <= 0.f) return 0;
        const double h = double(md.r_max - sp_r_lo) / double(mlp_spline);
        int kc = std::max(0, (int)std::ceil((double(rc) - sp_r_lo) / h));
        while (kc < mlp_spline && (sp_r_lo + kc * h) < double(rc)) ++kc;
        return std::min(mlp_spline, kc);
    }

    template <bool XB>
    void slot_xd_sp(int layer, const T* G, const T* A, const T* X, const T* Xb, const T* Yw_, const T* Gy, T* Xd) {
        const int pm = mspl_poly_mode();
        const bool pb = mspl_poly_bwd();
        const T* Rk = pm > 0 ? C_k[layer] : out_k[layer];
        const T* Tk = out_k[layer] + (long)spline_M * O;
        T* gRk = pb ? gC_k[layer] : gout_k[layer];
        T* gTk = gout_k[layer] + (long)spline_M * O;
        const T h = sp_h(), ih = sp_invh();
#ifndef LST_LEAN
        if (slot_kb_active() && !slot_scalar) {
            const dim3 grid(ks_max_tiles, sh.B);
            if constexpr (std::is_same<T, float>::value) {
                const int ah = act_half_mode();
                if (ah > 0) {
                    auto run_kbh = [&](auto tag) {
                        using H = decltype(tag);
                        const H* Gh = reinterpret_cast<const H*>(G);
                        const H* Ah = reinterpret_cast<const H*>(A);
                        if (pm == 2) k_slot_xd_kb<float, XB, 2, H><<<grid, 64, 0, st>>>(b.f_meta, d_ks_toff, d_ks_tiles, b.puv, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, Tk, gC_k[layer], Xd, h, ih, C);
                        else k_slot_xd_kb<float, XB, 1, H><<<grid, 64, 0, st>>>(b.f_meta, d_ks_toff, d_ks_tiles, b.puv, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, Tk, gC_k[layer], Xd, h, ih, C);
                    };
                    if (ah == 1) run_kbh(__nv_bfloat16{});
                    else run_kbh(__half{});
                    KCHECK();
                    return;
                }
            }
            if (pm == 2)
                k_slot_xd_kb<T, XB, 2><<<grid, 64, 0, st>>>(b.f_meta, d_ks_toff, d_ks_tiles, b.puv, b.sp_u, G, A, X, Xb,
                                                            Yw_, Gy, Y, Rk, Tk, gC_k[layer], Xd, h, ih, C);
            else if (pm == 1)
                k_slot_xd_kb<T, XB, 1><<<grid, 64, 0, st>>>(b.f_meta, d_ks_toff, d_ks_tiles, b.puv, b.sp_u, G, A, X, Xb,
                                                            Yw_, Gy, Y, Rk, Tk, gC_k[layer], Xd, h, ih, C);
            else
                k_slot_xd_kb<T, XB, 0><<<grid, 64, 0, st>>>(b.f_meta, d_ks_toff, d_ks_tiles, b.puv, b.sp_u, G, A, X, Xb,
                                                            Yw_, Gy, Y, Rk, Tk, gC_k[layer], Xd, h, ih, C);
            KCHECK();
            return;
        }
        auto launch_sc = [&](int es) {
            const int npb = 256 / (es * C);
            const int grid = (N + npb - 1) / npb;
            if (es == 4) {
                if (pm == 2 && pb) k_slot_xd_sp_sc<T, XB, 4, true, 2, true><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C);
                else if (pm == 2) k_slot_xd_sp_sc<T, XB, 4, true, 2, false><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C);
                else if (pm == 1 && pb) k_slot_xd_sp_sc<T, XB, 4, true, 1, true><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C);
                else if (pm == 1) k_slot_xd_sp_sc<T, XB, 4, true, 1, false><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C);
                else k_slot_xd_sp_sc<T, XB, 4, true, 0, false><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C);
            } else {
                if (pm == 2 && pb) k_slot_xd_sp_sc<T, XB, 2, true, 2, true><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C);
                else if (pm == 2) k_slot_xd_sp_sc<T, XB, 2, true, 2, false><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C);
                else if (pm == 1 && pb) k_slot_xd_sp_sc<T, XB, 2, true, 1, true><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C);
                else if (pm == 1) k_slot_xd_sp_sc<T, XB, 2, true, 1, false><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C);
                else k_slot_xd_sp_sc<T, XB, 2, true, 0, false><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C);
            }
        };
        if (slot_scalar) {
            launch_sc((slot_es == 4 && 4 * C <= 256) ? 4 : 2);
            KCHECK();
            return;
        }
#endif
        if constexpr (std::is_same<T, float>::value) {
            const int ah = act_half_mode();
            const bool eks = edge_ksort_active();
#ifndef LST_LEAN
            if (ah > 0 && pm == 2 && (edge_h2 & 2)) {
                const int es = slot_h2_es;
                const int npb = 8 / es;
                const int grid = (N + npb - 1) / npb;
                auto run_h2 = [&](auto tag) {
                    using H = decltype(tag);
                    const H* Gh = reinterpret_cast<const H*>(G);
                    const H* Ah = reinterpret_cast<const H*>(A);
                    if (es == 4) {
                        if (pb && eks) eh2::k_slot_xd_h2<H, XB, 4, true, true><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, b.emeta);
                        else if (pb) eh2::k_slot_xd_h2<H, XB, 4, true, false><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, nullptr);
                        else if (eks) eh2::k_slot_xd_h2<H, XB, 4, false, true><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, b.emeta);
                        else eh2::k_slot_xd_h2<H, XB, 4, false, false><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, nullptr);
                    } else {
                        if (pb && eks) eh2::k_slot_xd_h2<H, XB, 2, true, true><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, b.emeta);
                        else if (pb) eh2::k_slot_xd_h2<H, XB, 2, true, false><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, nullptr);
                        else if (eks) eh2::k_slot_xd_h2<H, XB, 2, false, true><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, b.emeta);
                        else eh2::k_slot_xd_h2<H, XB, 2, false, false><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, nullptr);
                    }
                };
                if (ah == 1) run_h2(__nv_bfloat16{});
                else run_h2(__half{});
                KCHECK();
                return;
            }
#endif
            if (ah > 0 && pm == 2 && slot_opt > 0) {
                const int* nord = (edge_norder && topo.norder) ? b.norder : nullptr;
#ifndef LST_LEAN
                const bool cc = (slot_opt == 2) && eks;
#endif
                auto run_opt = [&](auto tag) {
                    using H = decltype(tag);
                    const H* Gh = reinterpret_cast<const H*>(G);
                    const H* Ah = reinterpret_cast<const H*>(A);
#ifndef LST_LEAN
                    if (slot_opt == 8 && pb) {
                        if (N <= 512) {
                            if (eks) eh2::k_slot_xd_h2_fast<H, XB, true, 4><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord);
                            else eh2::k_slot_xd_h2_fast<H, XB, false, 4><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, nullptr, nord);
                        } else {
                            if (eks) eh2::k_slot_xd_h2_fast<H, XB, true, 2><<<(N + 1) / 2, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord);
                            else eh2::k_slot_xd_h2_fast<H, XB, false, 2><<<(N + 1) / 2, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, nullptr, nord);
                        }
                        return;
                    }
                    if ((slot_opt >= 11 && slot_opt <= 14 || (slot_opt == 15 && !XB)) && pb) {
                        const int kc2 = shell_kcut(layer, 2), kc3 = shell_kcut(layer, 3);
                        const int ym = y3p_mode();
                        const int4* em_ptr = (slot_opt >= 13) ? b.emeta_dir : b.emeta;
                        const bool use_t0d = (!XB) && t0s_dir_ok();
                        auto run_dir = [&](auto es_tag, auto sh_tag, auto t0_tag) {
                            constexpr int ES_V = decltype(es_tag)::value;
                            constexpr bool SH = decltype(sh_tag)::value;
                            constexpr bool T0D = decltype(t0_tag)::value;
                            constexpr int NTH_V = (ES_V == 2) ? 128 : 64;
                            if (eks && ym == 2) eh2::k_slot_xd_dir<H, XB, true, 2, SH, ES_V, T0D><<<N, NTH_V, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, em_ptr, nord, Y3P, kc2, kc3, b.spc, T0, T0s, S);
                            else if (eks && ym == 1) eh2::k_slot_xd_dir<H, XB, true, 1, SH, ES_V, T0D><<<N, NTH_V, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, em_ptr, nord, Y3P, kc2, kc3, b.spc, T0, T0s, S);
                            else if (eks) eh2::k_slot_xd_dir<H, XB, true, 0, SH, ES_V, T0D><<<N, NTH_V, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, em_ptr, nord, nullptr, kc2, kc3, b.spc, T0, T0s, S);
                            else eh2::k_slot_xd_dir<H, XB, false, 0, false, ES_V, false><<<N, NTH_V, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, nullptr, nord, nullptr);
                        };
                        auto run_dir_t0 = [&](auto es_tag, auto sh_tag) {
                            if (use_t0d && eks) run_dir(es_tag, sh_tag, std::true_type{});
                            else run_dir(es_tag, sh_tag, std::false_type{});
                        };
                        auto run_dir_es = [&](auto es_tag) {
                            if (shell_active && eks) run_dir_t0(es_tag, std::true_type{});
                            else run_dir_t0(es_tag, std::false_type{});
                        };
                        if (slot_opt == 12 || slot_opt == 13 || slot_opt == 15) run_dir_es(std::integral_constant<int, 1>{});
                        else run_dir_es(std::integral_constant<int, 2>{});
                        return;
                    }
                    if ((slot_opt == 9 || slot_opt == 10) && pb && zip_ga_slot_ok(layer)) {
                        const int ym = y3p_mode();
                        auto launch_q4 = [&](auto hf_tag, auto ym_tag) {
                            constexpr bool HF = decltype(hf_tag)::value;
                            constexpr int YM = decltype(ym_tag)::value;
                            const float* y3p_in = (YM > 0) ? Y3P : nullptr;
                            if (eks) eh2::k_slot_xd_q4<H, XB, true, HF, YM><<<N, 64, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord, GA_buf, y3p_in);
                            else eh2::k_slot_xd_q4<H, XB, false, HF, YM><<<N, 64, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, nullptr, nord, GA_buf, y3p_in);
                        };
                        auto launch_q4_hf = [&](auto hf_tag) {
                            if (eks && ym == 2) launch_q4(hf_tag, std::integral_constant<int, 2>{});
                            else if (eks && ym == 1) launch_q4(hf_tag, std::integral_constant<int, 1>{});
                            else launch_q4(hf_tag, std::integral_constant<int, 0>{});
                        };
                        if (slot_opt == 10 && std::is_same<H, __half>::value) launch_q4_hf(std::true_type{});
                        else launch_q4_hf(std::false_type{});
                        return;
                    }
#endif
                    if ((slot_opt == 7 || (slot_opt == 15 && XB)) && pb) {
                        const int kc2 = shell_kcut(layer, 2), kc3 = shell_kcut(layer, 3);
                        if (zip_ga_slot_ok(layer)) {
                            const int ym = y3p_mode();
                            const int gm = (slot_opt == 7) ? slot_hfma2_mode(layer) : 0;
                            auto run_v3 = [&](auto sh_tag, auto t0_tag) {
                                constexpr bool SH = decltype(sh_tag)::value;
                                constexpr bool T0D = decltype(t0_tag)::value;
                                if constexpr (std::is_same<H, __half>::value) {
                                    if (eks && ym == 3 && gm == 1) {
#ifndef LST_LEAN
                                        if (slot_ldcg) eh2::k_slot_xd_opt_v3<H, XB, true, true, 3, SH, T0D, 1, true><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord, GA_buf, Y3P, kc2, kc3, b.spc, T0, T0s, S);
                                        else
#endif
                                        eh2::k_slot_xd_opt_v3<H, XB, true, true, 3, SH, T0D, 1, false><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord, GA_buf, Y3P, kc2, kc3, b.spc, T0, T0s, S);
                                        return;
                                    }
                                    if (eks && ym == 2 && gm == 1) {
#ifndef LST_LEAN
                                        if (slot_ldcg) eh2::k_slot_xd_opt_v3<H, XB, true, true, 2, SH, T0D, 1, true><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord, GA_buf, Y3P, kc2, kc3, b.spc, T0, T0s, S);
                                        else
#endif
                                        eh2::k_slot_xd_opt_v3<H, XB, true, true, 2, SH, T0D, 1, false><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord, GA_buf, Y3P, kc2, kc3, b.spc, T0, T0s, S);
                                        return;
                                    }
                                }
                                if (eks && ym == 2) eh2::k_slot_xd_opt_v3<H, XB, true, true, 2, SH, T0D, 0><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord, GA_buf, Y3P, kc2, kc3, b.spc, T0, T0s, S);
#ifndef LST_LEAN
                                else if (eks && ym == 1) eh2::k_slot_xd_opt_v3<H, XB, true, true, 1, SH, T0D, 0><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord, GA_buf, Y3P, kc2, kc3, b.spc, T0, T0s, S);
#endif
                                else if (eks) eh2::k_slot_xd_opt_v3<H, XB, true, true, 0, SH, T0D, 0><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord, GA_buf, nullptr, kc2, kc3, b.spc, T0, T0s, S);
                                else eh2::k_slot_xd_opt_v3<H, XB, false, true, 0, false, false, 0><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, nullptr, nord, GA_buf, nullptr, 999999, 999999, nullptr, nullptr, nullptr, 0);
                            };
                            auto run_v3_t0 = [&](auto sh_tag) {
                                if constexpr (!XB) {
                                    if (t0s_pair_ok() && eks) run_v3(sh_tag, std::true_type{});
                                    else run_v3(sh_tag, std::false_type{});
                                } else {
                                    run_v3(sh_tag, std::false_type{});
                                }
                            };
#ifndef LST_LEAN
                            if (shell_active && eks) run_v3_t0(std::true_type{});
                            else
#endif
                            run_v3_t0(std::false_type{});
                        } else {
                            if (eks) eh2::k_slot_xd_opt_v3<H, XB, true, false, 0, false><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord);
                            else eh2::k_slot_xd_opt_v3<H, XB, false, false, 0, false><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, nullptr, nord);
                        }
                        return;
                    }
#ifndef LST_LEAN
                    if (slot_opt == 6 && pb) {
                        if (eks) eh2::k_slot_xd_opt_v2<H, XB, true><<<N, 64, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord);
                        else eh2::k_slot_xd_opt_v2<H, XB, false><<<N, 64, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, nullptr, nord);
                        return;
                    }
                    if (slot_opt == 5 && pb) {
                        if (eks) eh2::k_slot_xd_h2_smem<H, XB, true><<<N, 64, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord);
                        else eh2::k_slot_xd_h2_smem<H, XB, false><<<N, 64, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, nullptr, nord);
                        return;
                    }
                    if (slot_opt == 4 && pb) {
                        if (eks) eh2::k_slot_xd_h2_run<H, XB, true><<<N, 64, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, b.emeta, nord);
                        else eh2::k_slot_xd_h2_run<H, XB, false><<<N, 64, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, Xd, ih, N, nullptr, nord);
                        return;
                    }
                    if (pb) {
                        if (slot_opt >= 3 && eks) eh2::k_slot_xd_opt<H, XB, true, true, false, true><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, b.emeta, nord);
                        else if (eks && cc) eh2::k_slot_xd_opt<H, XB, true, true, true><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, b.emeta, nord);
                        else if (eks) eh2::k_slot_xd_opt<H, XB, true, true, false><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, b.emeta, nord);
                        else eh2::k_slot_xd_opt<H, XB, true, false, false><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, nullptr, nord);
                    } else {
                        if (eks && cc) eh2::k_slot_xd_opt<H, XB, false, true, true><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, b.emeta, nord);
                        else if (eks) eh2::k_slot_xd_opt<H, XB, false, true, false><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, b.emeta, nord);
                        else eh2::k_slot_xd_opt<H, XB, false, false, false><<<N, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, gRk, gTk, Xd, h, ih, N, nullptr, nord);
                    }
#endif
                };
                if (ah == 1) run_opt(__nv_bfloat16{});
                else run_opt(__half{});
                KCHECK();
                return;
            }
            if (ah > 0) {
#ifndef LST_LEAN
                const int npb = 256 / (2 * C);
                const int grid = (N + npb - 1) / npb;
                const int* r0 = eks ? b.rptr0 : nullptr;
                auto run_h = [&](auto tag) {
                    using H = decltype(tag);
                    const H* Gh = reinterpret_cast<const H*>(G);
                    const H* Ah = reinterpret_cast<const H*>(A);
                    if (pm == 2 && pb) k_slot_xd_sp<float, XB, 2, true, 2, true, H><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
                    else if (pm == 2) k_slot_xd_sp<float, XB, 2, true, 2, false, H><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
                    else if (pb) k_slot_xd_sp<float, XB, 2, true, 1, true, H><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
                    else k_slot_xd_sp<float, XB, 2, true, 1, false, H><<<grid, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, Gh, Ah, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
                };
                if (ah == 1) run_h(__nv_bfloat16{});
                else run_h(__half{});
                KCHECK();
                return;
#endif
            }
        }
        const int* r0 = edge_ksort_active() ? b.rptr0 : nullptr;
#ifndef LST_LEAN
        if (slot_es == 4 && 4 * C <= 256) {
            const int npb = 256 / (4 * C);
            if (pm == 2 && pb)
                k_slot_xd_sp<T, XB, 4, true, 2, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
            else if (pm == 2)
                k_slot_xd_sp<T, XB, 4, true, 2, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
            else if (pm == 1 && pb)
                k_slot_xd_sp<T, XB, 4, true, 1, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
            else if (pm == 1)
                k_slot_xd_sp<T, XB, 4, true, 1, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
            else
                k_slot_xd_sp<T, XB, 4, true, 0, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
            KCHECK();
            return;
        }
#endif
        const int npb = 256 / (2 * C);
#ifndef LST_LEAN
        if (pm == 2 && pb)
            k_slot_xd_sp<T, XB, 2, true, 2, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
        else if (pm == 2)
            k_slot_xd_sp<T, XB, 2, true, 2, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
        else if (pm == 1 && pb)
            k_slot_xd_sp<T, XB, 2, true, 1, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
        else if (pm == 1)
            k_slot_xd_sp<T, XB, 2, true, 1, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
        else
#endif
            k_slot_xd_sp<T, XB, 2, true, 0, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, Rk, Tk, gRk, gTk, Xd, h, ih, N, C, r0);
        KCHECK();
    }

    // Wave 40c+43: Flat member template helper avoids nvcc 13.0 + g++ 11.4 nested-generic-lambda if-constexpr miscompile;
    // npb1=true launches 1 atom/CTA (B=32*ES, G=N=357) eliminating 50% tail-wave idle & 4-atom barrier stragglers!
    template <typename H, bool DG, bool DX, int NT, bool XB, int F, bool SH, bool T0D>
    void launch_conv_h2_es(int es, bool eks, bool npb1, const float* X, const H* Gh, const float* Rk, const float* Y0,
                           const float* Y1, const float* Y2, const float* Xb, H* OGh, float* OX, float ih,
                           const int* nord, const float* y3p_in, int kc2, int kc3, const float* t0p, float* t0sp) {
        if constexpr (std::is_same<H, __half>::value && !SH && ((DX && F == 5) || (!DX && NT == 1))) {
            if (eks && npb1) {
                if (es == 1) {
                    eh2::k_conv_pair_h2<H, DG, DX, NT, XB, 1, DX, true, F, SH, T0D, false, 1><<<N, 32, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, b.emeta, nord,
                        y3p_in, kc2, kc3, b.spc, t0p, t0sp, S);
                    return;
                } else if (es == 4) {
                    eh2::k_conv_pair_h2<H, DG, DX, NT, XB, 4, DX, true, F, SH, T0D, false, 1><<<N, 128, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, b.emeta, nord,
                        y3p_in, kc2, kc3, b.spc, t0p, t0sp, S);
                    return;
                } else {
                    eh2::k_conv_pair_h2<H, DG, DX, NT, XB, 2, DX, true, F, SH, T0D, false, 1><<<N, 64, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, b.emeta, nord,
                        y3p_in, kc2, kc3, b.spc, t0p, t0sp, S);
                    return;
                }
            }
        }
        if (es == 1) {
            if (eks) eh2::k_conv_pair_h2<H, DG, DX, NT, XB, 1, DX, true, F, SH, T0D, false><<<(N + 3) / 4, 128, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, b.emeta, nord, y3p_in, kc2, kc3, b.spc, t0p, t0sp, S);
            else eh2::k_conv_pair_h2<H, DG, DX, NT, XB, 1, DX, false, F, false, false, false><<<(N + 3) / 4, 128, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, nullptr, nord, y3p_in);
        } else if (es == 4) {
#ifdef LST_LEAN
            constexpr int ES4_E = (std::is_same<H, __half>::value && !SH && ((DX && F == 5) || (!DX && NT == 1))) ? 4 : 2;
            constexpr int NPB4 = 8 / ES4_E;
            if (eks) eh2::k_conv_pair_h2<H, DG, DX, NT, XB, ES4_E, DX, true, F, SH, T0D, false><<<(N + NPB4 - 1) / NPB4, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, b.emeta, nord, y3p_in, kc2, kc3, b.spc, t0p, t0sp, S);
            else eh2::k_conv_pair_h2<H, DG, DX, NT, XB, 2, DX, false, F, false, false, false><<<(N + 3) / 4, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, nullptr, nord, y3p_in);
#else
            if (eks) eh2::k_conv_pair_h2<H, DG, DX, NT, XB, 4, DX, true, F, SH, T0D, false><<<(N + 1) / 2, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, b.emeta, nord, y3p_in, kc2, kc3, b.spc, t0p, t0sp, S);
            else eh2::k_conv_pair_h2<H, DG, DX, NT, XB, 4, DX, false, F, false, false, false><<<(N + 1) / 2, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, nullptr, nord, y3p_in);
#endif
        } else {
            if (eks) eh2::k_conv_pair_h2<H, DG, DX, NT, XB, 2, DX, true, F, SH, T0D, false><<<(N + 3) / 4, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, b.emeta, nord, y3p_in, kc2, kc3, b.spc, t0p, t0sp, S);
            else eh2::k_conv_pair_h2<H, DG, DX, NT, XB, 2, DX, false, F, false, false, false><<<(N + 3) / 4, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, nullptr, nord, y3p_in);
        }
    }

    template <bool DG, bool DX, int NT, bool XB, bool ACC>
    void conv_sp(int layer, const T* X, const T* G, const T* Y0, const T* Y1, const T* Y2, const T* Xb, T* OG, T* OX) {
        const int pm = mspl_poly_mode();
        const bool sdx = DX && sdx_active();
        const bool eks = edge_ksort_active();
        const T* Rk = pm > 0 ? C_k[layer] : out_k[layer];
        const T* Tk = out_k[layer] + (long)spline_M * O;
        const bool nm = nm_active();
        const T h = sp_h(), ih = sp_invh();
        if constexpr (std::is_same<T, float>::value && !ACC) {
            const int ah = act_half_mode();
            if (ah > 0 && pm == 2 && (edge_h2 & 1)) {
                const int* nord = (edge_norder && topo.norder) ? b.norder : nullptr;
                auto run_ch2 = [&](auto tag) {
                    using H = decltype(tag);
                    const H* Gh = reinterpret_cast<const H*>(G);
                    H* OGh = reinterpret_cast<H*>(OG);
                    const int kc2 = shell_kcut(layer, 2), kc3 = shell_kcut(layer, 3);
#ifndef LST_LEAN
                    if (conv_opt && !conv_c64 && eks && DG) {
                        const int grid_co = (N + 3) / 4;
                        const bool use_t0p = (layer == 0) && (DX ? t0s_pair_ok() : (S <= 4 && !in_debug));
                        auto launch_co = [&](auto ft, auto sh_tag, auto t0_tag) {
                            constexpr int F = decltype(ft)::value;
                            constexpr bool SH = decltype(sh_tag)::value;
                            constexpr bool T0D = decltype(t0_tag)::value;
                            const float* y3p_in = (F >= 2) ? Y3P : nullptr;
                            eh2::k_conv_pair_h2_opt<H, DG, DX, NT, XB, F, SH, T0D><<<grid_co, 128, 0, st>>>(
                                b.rptr, b.emeta_dir, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, nord, y3p_in,
                                kc2, kc3, b.spc, T0, T0s, S);
                        };
                        auto launch_co_t0 = [&](auto ft, auto sh_tag) {
                            if (use_t0p) launch_co(ft, sh_tag, std::true_type{});
                            else launch_co(ft, sh_tag, std::false_type{});
                        };
                        auto launch_co_sh = [&](auto ft) {
                            if (shell_active) launch_co_t0(ft, std::true_type{});
                            else launch_co_t0(ft, std::false_type{});
                        };
                        if constexpr (DX && NT == 2) {
                            const int ym = y3p_mode();
                            if (ym == 2) launch_co_sh(std::integral_constant<int, 3>{});
                            else if (ym == 1) launch_co_sh(std::integral_constant<int, 2>{});
                            else launch_co_sh(std::integral_constant<int, 1>{});
                        } else {
                            launch_co_sh(std::integral_constant<int, 0>{});
                        }
                        return;
                    }
                    if (conv_c64 && eks && DG) {
                        const int grid_c64 = (N + 1) / 2;
                        const bool use_t0d = (layer == 0) && (DX ? t0s_dir_ok() : (S <= 4 && !in_debug));
                        auto launch_c64 = [&](auto ft, auto sh_tag, auto t0_tag) {
                            constexpr int F = decltype(ft)::value;
                            constexpr bool SH = decltype(sh_tag)::value;
                            constexpr bool T0D = decltype(t0_tag)::value;
                            const float* y3p_in = (F >= 2) ? Y3P : nullptr;
                            eh2::k_conv_c64<H, DG, DX, NT, XB, F, SH, T0D><<<grid_c64, 128, 0, st>>>(
                                b.rptr, b.emeta_dir, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, nord, y3p_in,
                                kc2, kc3, b.spc, T0, T0s, S);
                        };
                        auto launch_c64_t0 = [&](auto ft, auto sh_tag) {
                            if (use_t0d) launch_c64(ft, sh_tag, std::true_type{});
                            else launch_c64(ft, sh_tag, std::false_type{});
                        };
                        auto launch_c64_sh = [&](auto ft) {
                            if (shell_active) launch_c64_t0(ft, std::true_type{});
                            else launch_c64_t0(ft, std::false_type{});
                        };
                        if constexpr (DX && NT == 2) {
                            const int ym = y3p_mode();
                            if (ym == 2) launch_c64_sh(std::integral_constant<int, 3>{});
                            else if (ym == 1) launch_c64_sh(std::integral_constant<int, 2>{});
                            else launch_c64_sh(std::integral_constant<int, 1>{});
                        } else {
                            launch_c64_sh(std::integral_constant<int, 0>{});
                        }
                        return;
                    }
#endif
                    if constexpr (!DG && DX && NT == 1 && !XB) {
                        if constexpr (std::is_same<H, __half>::value) {
                            if ((conv_npb1 & 2) && eks && !shell_active) {
                                eh2::k_conv_pair_h2<H, false, true, 1, false, 4, true, true, 0, false, false, false, 1><<<N, 128, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, b.emeta, nord);
                                return;
                            }
                        }
                        const int npb = 8 / 4;
                        if (eks && shell_active) eh2::k_conv_pair_h2<H, false, true, 1, false, 4, true, true, 0, true><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, b.emeta, nord, nullptr, kc2, kc3);
                        else if (eks) eh2::k_conv_pair_h2<H, false, true, 1, false, 4, true, true, 0, false><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, b.emeta, nord);
                        else eh2::k_conv_pair_h2<H, false, true, 1, false, 4, true, false, 0, false><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, N, nullptr, nord);
                        return;
                    }
                    int es = DX ? (XB ? conv_h2_es_rev1 : conv_h2_es_rev0) : conv_h2_es_fwd;
                    const bool npb1 = DX ? (XB ? ((conv_npb1 & 8) != 0) : ((conv_npb1 & 4) != 0)) : ((conv_npb1 & 1) != 0);
                    if constexpr (DX) {
                        auto launch_dx = [&](auto ft) {
                            constexpr int F = decltype(ft)::value;
                            const float* y3p_in = (F >= 2) ? Y3P : nullptr;
                            auto run_sh_t0 = [&](auto sh_tag, auto t0_tag) {
                                constexpr bool SH = decltype(sh_tag)::value;
                                constexpr bool T0D = decltype(t0_tag)::value;
                                launch_conv_h2_es<H, DG, DX, NT, XB, F, SH, T0D>(
                                    es, eks, npb1, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, nord, y3p_in, kc2, kc3, T0, T0s);
                            };
                            auto run_sh = [&](auto sh_tag) {
                                if constexpr (DG && !XB) {
                                    if (layer == 0 && t0s_pair_ok() && eks) run_sh_t0(sh_tag, std::true_type{});
                                    else run_sh_t0(sh_tag, std::false_type{});
                                } else {
                                    run_sh_t0(sh_tag, std::false_type{});
                                }
                            };
#ifndef LST_LEAN
                            if (shell_active && eks) run_sh(std::true_type{});
                            else
#endif
                            run_sh(std::false_type{});
                        };
                        const int ym = y3p_mode();
                        if constexpr (std::is_same<H, __half>::value) {
                            if (NT == 2 && conv_fact && ym == 3 && conv_hfma2 == 2) { launch_dx(std::integral_constant<int, 6>{}); return; }
                            if (NT == 2 && conv_fact && ym == 2 && conv_hfma2 == 2) { launch_dx(std::integral_constant<int, 5>{}); return; }
#ifndef LST_LEAN
                            if (NT == 2 && conv_fact && ym == 2 && conv_hfma2 == 1) { launch_dx(std::integral_constant<int, 4>{}); return; }
#endif
                        }
                        if (NT == 2 && conv_fact && ym == 2) launch_dx(std::integral_constant<int, 3>{});
#ifndef LST_LEAN
                        else if (NT == 2 && conv_fact && ym == 1) launch_dx(std::integral_constant<int, 2>{});
                        else if (NT == 2 && conv_fact) launch_dx(std::integral_constant<int, 1>{});
#endif
                        else launch_dx(std::integral_constant<int, 0>{});
                    } else {
                        auto run_fwd_sh_t0 = [&](auto sh_tag, auto t0_tag) {
                            constexpr bool SH = decltype(sh_tag)::value;
                            constexpr bool T0D = decltype(t0_tag)::value;
                            launch_conv_h2_es<H, DG, DX, NT, XB, 0, SH, T0D>(
                                es, eks, npb1, X, Gh, Rk, Y0, Y1, Y2, Xb, OGh, OX, ih, nord, nullptr, kc2, kc3, T0, T0s);
                        };
                        auto run_fwd_sh = [&](auto sh_tag) {
                            if constexpr (DG && !XB) {
                                if (layer == 0 && fwd_t0s && t0s_pair && S <= 4 && !in_debug && eks) run_fwd_sh_t0(sh_tag, std::true_type{});
                                else run_fwd_sh_t0(sh_tag, std::false_type{});
                            } else {
                                run_fwd_sh_t0(sh_tag, std::false_type{});
                            }
                        };
#ifndef LST_LEAN
                        if (shell_active && eks) run_fwd_sh(std::true_type{});
                        else
#endif
                        run_fwd_sh(std::false_type{});
                    }
                };
                if (ah == 1) run_ch2(__nv_bfloat16{});
                else run_ch2(__half{});
                KCHECK();
                return;
            }
#ifndef LST_LEAN
            if (ah > 0) {
                auto run_h = [&](auto tag) {
                    using H = decltype(tag);
                    const H* Gh = reinterpret_cast<const H*>(G);
                    H* OGh = reinterpret_cast<H*>(OG);
                    if constexpr (!DG && DX && NT == 1 && !XB) {
                        if (sdx_es4) {
                            const int npb = 256 / (4 * C);
                            if (eks && pm == 2) k_conv_pair_sp<float, false, true, 1, false, false, 4, true, 2, true, H, true><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Tk, Y0, Y1, Y2, Xb, OGh, OX, h, ih, N, C, b.emeta);
                            else if (eks) k_conv_pair_sp<float, false, true, 1, false, false, 4, true, 1, true, H, true><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Tk, Y0, Y1, Y2, Xb, OGh, OX, h, ih, N, C, b.emeta);
                            else if (pm == 2) k_conv_pair_sp<float, false, true, 1, false, false, 4, true, 2, true, H><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Tk, Y0, Y1, Y2, Xb, OGh, OX, h, ih, N, C);
                            else k_conv_pair_sp<float, false, true, 1, false, false, 4, true, 1, true, H><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Tk, Y0, Y1, Y2, Xb, OGh, OX, h, ih, N, C);
                            return;
                        }
                    }
                    const int npb = 256 / (2 * C);
                    if constexpr (DX) {
                        if (eks && pm == 2) k_conv_pair_sp<float, DG, DX, NT, XB, false, 2, true, 2, true, H, true><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Tk, Y0, Y1, Y2, Xb, OGh, OX, h, ih, N, C, b.emeta);
                        else if (eks) k_conv_pair_sp<float, DG, DX, NT, XB, false, 2, true, 1, true, H, true><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Tk, Y0, Y1, Y2, Xb, OGh, OX, h, ih, N, C, b.emeta);
                        else if (pm == 2) k_conv_pair_sp<float, DG, DX, NT, XB, false, 2, true, 2, true, H><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Tk, Y0, Y1, Y2, Xb, OGh, OX, h, ih, N, C);
                        else k_conv_pair_sp<float, DG, DX, NT, XB, false, 2, true, 1, true, H><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Tk, Y0, Y1, Y2, Xb, OGh, OX, h, ih, N, C);
                    } else {
                        if (eks && pm == 2) k_conv_pair_sp<float, DG, DX, NT, XB, false, 2, true, 2, false, H, true><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Tk, Y0, Y1, Y2, Xb, OGh, OX, h, ih, N, C, b.emeta);
                        else if (eks) k_conv_pair_sp<float, DG, DX, NT, XB, false, 2, true, 1, false, H, true><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Tk, Y0, Y1, Y2, Xb, OGh, OX, h, ih, N, C, b.emeta);
                        else if (pm == 2) k_conv_pair_sp<float, DG, DX, NT, XB, false, 2, true, 2, false, H><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Tk, Y0, Y1, Y2, Xb, OGh, OX, h, ih, N, C);
                        else k_conv_pair_sp<float, DG, DX, NT, XB, false, 2, true, 1, false, H><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, Gh, Rk, Tk, Y0, Y1, Y2, Xb, OGh, OX, h, ih, N, C);
                    }
                };
                if (ah == 1) run_h(__nv_bfloat16{});
                else run_h(__half{});
                KCHECK();
                return;
            }
#endif
        }
#ifndef LST_LEAN
        if constexpr (!DG && DX && NT == 1 && !XB) {
            if (sdx && sdx_es4 && 4 * C <= 256) {
                const int npb = 256 / (4 * C);
                if (eks && pm == 2)
                    k_conv_pair_sp<T, false, true, 1, false, false, 4, true, 2, true, T, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C, b.emeta);
                else if (eks && pm == 1)
                    k_conv_pair_sp<T, false, true, 1, false, false, 4, true, 1, true, T, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C, b.emeta);
                else if (pm == 2)
                    k_conv_pair_sp<T, false, true, 1, false, false, 4, true, 2, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C);
                else if (pm == 1)
                    k_conv_pair_sp<T, false, true, 1, false, false, 4, true, 1, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C);
                else
                    k_conv_pair_sp<T, false, true, 1, false, false, 4, true, 0, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C);
                KCHECK();
                return;
            }
        }
        if (conv_es == 2 && 2 * C <= 256) {
            const int npb = 256 / (2 * C);
            if (nm) {
                if (eks && pm == 2 && sdx)
                    k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 2, true, 2, true, T, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C, b.emeta);
                else if (eks && pm == 2)
                    k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 2, true, 2, false, T, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C, b.emeta);
                else if (eks && pm == 1 && sdx)
                    k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 2, true, 1, true, T, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C, b.emeta);
                else if (eks && pm == 1)
                    k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 2, true, 1, false, T, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C, b.emeta);
                else if (pm == 2 && sdx)
                    k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 2, true, 2, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C);
                else if (pm == 2)
                    k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 2, true, 2, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C);
                else if (pm == 1 && sdx)
                    k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 2, true, 1, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C);
                else if (pm == 1)
                    k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 2, true, 1, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C);
                else if (sdx)
                    k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 2, true, 0, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C);
                else
                    k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 2, true, 0, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C);
            } else {
                k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 2, false, 0, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C);
            }
            KCHECK();
            return;
        }
#endif
        const int npb = 256 / C;
        if (nm)
            k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 1, true, 0, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C);
        else
            k_conv_pair_sp<T, DG, DX, NT, XB, ACC, 1, false, 0, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, X, G, Rk, Tk, Y0, Y1, Y2, Xb, OG, OX, h, ih, N, C);
        KCHECK();
    }

    template <bool XB>
    void slot_sp(int layer, const T* G, const T* A, const T* X, const T* Xb, const T* Yw_, const T* Gy) {
        const bool poly = mspl_poly_bwd();
        const bool sng = sng_active();
        const bool bal = slot_bal && !sng;
        T* gRk = poly ? gC_k[layer] : gout_k[layer];
        T* gTk = gout_k[layer] + (long)spline_M * O;
        const bool nm = nm_active();
        const T h = sp_h(), ih = sp_invh();
        if (slot_es == 2 && 2 * C <= 256) {
            const int npb = 256 / (2 * C);
            if (nm) {
                if (poly && sng)
                    k_slot_pair_sp<T, XB, 2, true, true, true, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, gRk, gTk, h, ih, N, C);
                else if (poly && bal)
                    k_slot_pair_sp<T, XB, 2, true, true, false, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, gRk, gTk, h, ih, N, C);
                else if (poly)
                    k_slot_pair_sp<T, XB, 2, true, true, false, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, gRk, gTk, h, ih, N, C);
                else if (sng)
                    k_slot_pair_sp<T, XB, 2, true, false, true, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, gRk, gTk, h, ih, N, C);
                else if (bal)
                    k_slot_pair_sp<T, XB, 2, true, false, false, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, gRk, gTk, h, ih, N, C);
                else
                    k_slot_pair_sp<T, XB, 2, true, false, false, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, gRk, gTk, h, ih, N, C);
            } else {
                k_slot_pair_sp<T, XB, 2, false, false, false, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, gRk, gTk, h, ih, N, C);
            }
            KCHECK();
            return;
        }
        const int npb = slot_npb > 0 ? slot_npb : 256 / C;
        if (nm)
            k_slot_pair_sp<T, XB, 1, true, false, false, false><<<(N + npb - 1) / npb, npb * C, 0, st>>>(
                b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, gRk, gTk, h, ih, N, C);
        else
            k_slot_pair_sp<T, XB, 1, false, false, false, false><<<(N + npb - 1) / npb, npb * C, 0, st>>>(
                b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, G, A, X, Xb, Yw_, Gy, Y, gRk, gTk, h, ih, N, C);
        KCHECK();
    }

    template <bool DG, bool DX, int NT, bool XB, bool ACC>
    void conv(const T* X, const T* G, const T* R0, const T* Y0, const T* R1, const T* Y1, const T* Y2, const T* Xb,
              T* OG, T* OX) {
        if (pair_basis && !in_debug) {
            const bool nm = nm_active();
            const int* nord = (conv_order && topo.norder) ? b.norder : nullptr;
            if (conv_es == 2 && 2 * C <= 256) {
                const int npb = 256 / (2 * C);
                if (nm)
                    k_conv_pair<T, DG, DX, NT, XB, ACC, 2, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, X, G, R0, Y0, R1, Y1, Y2, Xb, OG, OX, N, C, nord);
                else
                    k_conv_pair<T, DG, DX, NT, XB, ACC, 2, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, X, G, R0, Y0, R1, Y1, Y2, Xb, OG, OX, N, C, nord);
                KCHECK();
                return;
            }
            if (conv_es == 4 && 4 * C <= 256) {
                const int npb = 256 / (4 * C);
                if (nm)
                    k_conv_pair<T, DG, DX, NT, XB, ACC, 4, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, X, G, R0, Y0, R1, Y1, Y2, Xb, OG, OX, N, C, nord);
                else
                    k_conv_pair<T, DG, DX, NT, XB, ACC, 4, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                        b.rptr, b.snd, b.ps, X, G, R0, Y0, R1, Y1, Y2, Xb, OG, OX, N, C, nord);
                KCHECK();
                return;
            }
            const int npb = 256 / C;
            if (nm)
                k_conv_pair<T, DG, DX, NT, XB, ACC, 1, true><<<(N + npb - 1) / npb, 256, 0, st>>>(
                    b.rptr, b.snd, b.ps, X, G, R0, Y0, R1, Y1, Y2, Xb, OG, OX, N, C, nord);
            else
                k_conv_pair<T, DG, DX, NT, XB, ACC, 1, false><<<(N + npb - 1) / npb, 256, 0, st>>>(
                    b.rptr, b.snd, b.ps, X, G, R0, Y0, R1, Y1, Y2, Xb, OG, OX, N, C, nord);
            KCHECK();
            return;
        }
        if (conv_es == 2 && 2 * C <= 256) {
            const int npb = 256 / (2 * C);
            k_conv<T, DG, DX, NT, XB, ACC, 2><<<(N + npb - 1) / npb, 256, 0, st>>>(
                b.rptr, b.snd, b.pidx, b.pe0, b.pe1, X, G, R0, Y0, R1, Y1, Y2, Xb, OG, OX, N, C);
            KCHECK();
            return;
        }
        if (conv_es == 4 && 4 * C <= 256) {
            const int npb = 256 / (4 * C);
            k_conv<T, DG, DX, NT, XB, ACC, 4><<<(N + npb - 1) / npb, 256, 0, st>>>(
                b.rptr, b.snd, b.pidx, b.pe0, b.pe1, X, G, R0, Y0, R1, Y1, Y2, Xb, OG, OX, N, C);
            KCHECK();
            return;
        }
        const int npb = 256 / C;
        k_conv<T, DG, DX, NT, XB, ACC><<<(N + npb - 1) / npb, 256, 0, st>>>(
            b.rptr, b.snd, b.pidx, b.pe0, b.pe1, X, G, R0, Y0, R1, Y1, Y2, Xb, OG, OX, N, C);
        KCHECK();
    }
    template <bool XB>
    void slot(const T* G, const T* A, const T* X, const T* Xb, const T* Yw_, const T* Gy, T* Tb, T* Rb) {
        if (pair_basis && !in_debug) {
            const bool nm = nm_active();
            const int* nord = (conv_order && topo.norder) ? b.norder : nullptr;
            if (slot_es == 2 && 2 * C <= 256) {
                const int npb = 256 / (2 * C);
                if (nm)
                    k_slot_pair<T, XB, 2, true><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, G, A, X, Xb,
                                                                                     Yw_, Gy, Y, Tb, Rb, N, C, nord);
                else
                    k_slot_pair<T, XB, 2, false><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, G, A, X, Xb,
                                                                                      Yw_, Gy, Y, Tb, Rb, N, C, nord);
                KCHECK();
                return;
            }
            if (slot_es == 4 && 4 * C <= 256) {
                const int npb = 256 / (4 * C);
                if (nm)
                    k_slot_pair<T, XB, 4, true><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, G, A, X, Xb,
                                                                                     Yw_, Gy, Y, Tb, Rb, N, C, nord);
                else
                    k_slot_pair<T, XB, 4, false><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.ps, G, A, X, Xb,
                                                                                      Yw_, Gy, Y, Tb, Rb, N, C, nord);
                KCHECK();
                return;
            }
            const int npb = slot_npb > 0 ? slot_npb : 256 / C;
            if (nm)
                k_slot_pair<T, XB, 1, true><<<(N + npb - 1) / npb, npb * C, 0, st>>>(b.rptr, b.snd, b.ps, G, A, X, Xb,
                                                                                     Yw_, Gy, Y, Tb, Rb, N, C, nord);
            else
                k_slot_pair<T, XB, 1, false><<<(N + npb - 1) / npb, npb * C, 0, st>>>(b.rptr, b.snd, b.ps, G, A, X, Xb,
                                                                                      Yw_, Gy, Y, Tb, Rb, N, C, nord);
            KCHECK();
            return;
        }
        if (slot_pp && C <= 256) {
            const int npb = 256 / C;
            k_slot_pp<T, XB><<<(N + npb - 1) / npb, npb * C, 0, st>>>(b.pptr, b.snd, b.pe0, b.pe1, G, A, X, Xb, Yw_,
                                                                       Gy, Y, Tb, Rb, N, C);
            KCHECK();
            return;
        }
        if (slot_es == 2 && 2 * C <= 256) {
            const int npb = 256 / (2 * C);
            k_slot<T, XB, 2><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.pidx, b.pe0, b.pe1, G, A, X, Xb,
                                                                 Yw_, Gy, Y, Tb, Rb, N, C);
            KCHECK();
            return;
        }
        if (slot_es == 4 && 4 * C <= 256) {
            const int npb = 256 / (4 * C);
            k_slot<T, XB, 4><<<(N + npb - 1) / npb, 256, 0, st>>>(b.rptr, b.snd, b.pidx, b.pe0, b.pe1, G, A, X, Xb,
                                                                 Yw_, Gy, Y, Tb, Rb, N, C);
            KCHECK();
            return;
        }
        const int npb = slot_npb > 0 ? slot_npb : 256 / C;
        k_slot<T, XB><<<(N + npb - 1) / npb, npb * C, 0, st>>>(b.rptr, b.snd, b.pidx, b.pe0, b.pe1, G, A, X, Xb, Yw_, Gy,
                                                           Y, Tb, Rb, N, C);
        KCHECK();
    }

    // per-degree GEMMs over [16, N, C] (or [N, 16, C] when nm_active()) blocks: rows of degree l are [l^2 N, (l+1)^2 N)
    // out_l = in_l @ (W_l or W_l^T), W_l = W + l * wstride (row-major [C, ncols] or its transpose)
    void per_l(const T* in, int in_cols, const T* W, long wstride, int ldw, bool tw, int out_cols, T* o, T beta,
               const T* G_zip = nullptr, uint32_t* GA_zip = nullptr, int zip_mode = 0) {
        if constexpr (std::is_same<T, float>::value) {
            const int ah = act_half_mode();
            if (ah > 0 && in_cols == 64 && out_cols == 64 && ldw % 4 == 0 && wstride % 4 == 0) {
                if (segg_v2) {
                    const int mt = segg_mt;
                    int tot_b = 0;
                    for (int l = 0; l < 4; ++l) {
                        const int tl = ((2 * l + 1) * N + 63) / 64;
                        tot_b += (tl + mt - 1) / mt;
                    }
                    const dim3 grid(1, tot_b);
                    const int zm = (GA_zip && tw) ? (zip_mode > 0 ? zip_mode : 1) : 0;
                    const bool use_wh = segg_h16 && (W == wl1) && Wl1_h != nullptr;
                    if (ah == 1) {
                        const auto* ih = reinterpret_cast<const __nv_bfloat16*>(in);
                        auto* oh = reinterpret_cast<__nv_bfloat16*>(o);
                        const auto* wh = use_wh ? reinterpret_cast<const __nv_bfloat16*>(Wl1_h) : nullptr;
                        if (zm == 4) segg::k_seg_gemm_nm_h_v2<__nv_bfloat16, true, 4><<<grid, 256, 0, st>>>(ih, W, wstride, ldw, oh, beta, N, mt, reinterpret_cast<const __nv_bfloat16*>(G_zip), GA_zip, wh);
                        else if (zm == 1) segg::k_seg_gemm_nm_h_v2<__nv_bfloat16, true, 1><<<grid, 256, 0, st>>>(ih, W, wstride, ldw, oh, beta, N, mt, reinterpret_cast<const __nv_bfloat16*>(G_zip), GA_zip, wh);
                        else if (tw) segg::k_seg_gemm_nm_h_v2<__nv_bfloat16, true, 0><<<grid, 256, 0, st>>>(ih, W, wstride, ldw, oh, beta, N, mt, nullptr, nullptr, wh);
                        else segg::k_seg_gemm_nm_h_v2<__nv_bfloat16, false, 0><<<grid, 256, 0, st>>>(ih, W, wstride, ldw, oh, beta, N, mt, nullptr, nullptr, wh);
                    } else {
                        const auto* ih = reinterpret_cast<const __half*>(in);
                        auto* oh = reinterpret_cast<__half*>(o);
                        const auto* wh = use_wh ? reinterpret_cast<const __half*>(Wl1_h) : nullptr;
                        if (zm == 4) segg::k_seg_gemm_nm_h_v2<__half, true, 4><<<grid, 256, 0, st>>>(ih, W, wstride, ldw, oh, beta, N, mt, reinterpret_cast<const __half*>(G_zip), GA_zip, wh);
                        else if (zm == 1) segg::k_seg_gemm_nm_h_v2<__half, true, 1><<<grid, 256, 0, st>>>(ih, W, wstride, ldw, oh, beta, N, mt, reinterpret_cast<const __half*>(G_zip), GA_zip, wh);
                        else if (tw) segg::k_seg_gemm_nm_h_v2<__half, true, 0><<<grid, 256, 0, st>>>(ih, W, wstride, ldw, oh, beta, N, mt, nullptr, nullptr, wh);
                        else segg::k_seg_gemm_nm_h_v2<__half, false, 0><<<grid, 256, 0, st>>>(ih, W, wstride, ldw, oh, beta, N, mt, nullptr, nullptr, wh);
                    }
                    KCHECK();
                    return;
                }
                const dim3 grid(1, segg::row_tiles(N));
                if (ah == 1) {
                    const auto* ih = reinterpret_cast<const __nv_bfloat16*>(in);
                    auto* oh = reinterpret_cast<__nv_bfloat16*>(o);
                    if (tw) segg::k_seg_gemm_nm_h<__nv_bfloat16, true><<<grid, 256, 0, st>>>(ih, W, wstride, ldw, oh, beta, N);
                    else segg::k_seg_gemm_nm_h<__nv_bfloat16, false><<<grid, 256, 0, st>>>(ih, W, wstride, ldw, oh, beta, N);
                } else {
                    const auto* ih = reinterpret_cast<const __half*>(in);
                    auto* oh = reinterpret_cast<__half*>(o);
                    if (tw) segg::k_seg_gemm_nm_h<__half, true><<<grid, 256, 0, st>>>(ih, W, wstride, ldw, oh, beta, N);
                    else segg::k_seg_gemm_nm_h<__half, false><<<grid, 256, 0, st>>>(ih, W, wstride, ldw, oh, beta, N);
                }
                KCHECK();
                return;
            }
        }
        if (nm_active() && in_cols % segg::TK == 0 && out_cols % segg::TN == 0 && ldw % 4 == 0 && wstride % 4 == 0) {
            const dim3 grid(out_cols / segg::TN, segg::row_tiles(N));
            if (tw)
                segg::k_seg_gemm_nm<T, true><<<grid, 256, 0, st>>>(in, in_cols, W, wstride, ldw, out_cols, o, beta, N);
            else
                segg::k_seg_gemm_nm<T, false><<<grid, 256, 0, st>>>(in, in_cols, W, wstride, ldw, out_cols, o, beta, N);
            KCHECK();
            return;
        }
        if constexpr (std::is_same<T, float>::value) {
            if (seg_gemm && in_cols % segg::TK == 0 && out_cols % segg::TN == 0 && ldw % 4 == 0 && wstride % 4 == 0) {
                const dim3 grid(out_cols / segg::TN, segg::row_tiles(N));
                if (tw)
                    segg::k_seg_gemm<true><<<grid, 256, 0, st>>>(in, in_cols, W, wstride, ldw, out_cols, o, beta, N);
                else
                    segg::k_seg_gemm<false><<<grid, 256, 0, st>>>(in, in_cols, W, wstride, ldw, out_cols, o, beta, N);
                KCHECK();
                return;
            }
        }
        for (int l = 0; l < 4; ++l) {
            long r0 = (long)l * l * N;
            int nr = (2 * l + 1) * N;
            lgemm(false, tw, nr, out_cols, in_cols, T(1), in + r0 * in_cols, in_cols, W + l * wstride, ldw, beta,
                  o + r0 * out_cols, out_cols);
        }
    }
    // Wg_l (+)= in_l^T @ other_l   ([cols_a, cols_b] per degree)
    void per_l_wgrad(const T* a, int ca, const T* bb, int cbb, T* Wg, long wstride, T beta) {
        {
            const T* As[4];
            const T* Bs[4];
            T* Cs[4];
            int Ks[4];
            for (int l = 0; l < 4; ++l) {
                long r0 = (long)l * l * N;
                As[l] = a + r0 * ca;
                Bs[l] = bb + r0 * cbb;
                Cs[l] = Wg + l * wstride;
                Ks[l] = (2 * l + 1) * N;
            }
            if (gemm_tn(4, As, Bs, Cs, Ks, ca, cbb, ca, cbb, cbb, T(1), beta)) return;
        }
        for (int l = 0; l < 4; ++l) {
            long r0 = (long)l * l * N;
            int nr = (2 * l + 1) * N;
            lgemm(true, false, ca, cbb, nr, T(1), a + r0 * ca, ca, bb + r0 * cbb, cbb, beta, Wg + l * wstride, cbb);
        }
    }

    void mlp_fwd(int t, bool use_raw = false) {
        const bool ms = mspl_active();
        const bool mf = mspl_fused();
        const int mp_mode = mspl_poly_mode();
        const int Pm = ms ? spline_M : P;
        const T* ef_in = ms ? EF_k : EF;
        const T* def_in = ms ? DEF_k : DEF;
        const int* pe0_in = ms ? pe0_k : b.pe0;
        T* out_dst = ms ? out_k[t] : out[t];
        T* h3_dst = ms ? h3_k[t] : h3[t];
        auto post_spline = [&]() {
            if (mp_mode == 2) {
                if constexpr (std::is_same<T, float>::value)
                    mspl::k_spline_rt_to_ch<<<mlp_spline, O, 0, st>>>(out_k[t], out_k[t] + (long)spline_M * O,
                                                                      reinterpret_cast<uint2*>(C_k[t]), float(sp_h()),
                                                                      mlp_spline, O, shell_rcut4(t), float(sp_r_lo),
                                                                      shell_width);
            } else if (mp_mode == 1)
                mspl::k_spline_rt_to_c<T><<<mlp_spline, O, 0, st>>>(out_k[t], out_k[t] + (long)spline_M * O, C_k[t],
                                                                    sp_h(), mlp_spline, O, shell_rcut4(t), T(sp_r_lo),
                                                                    T(shell_width));
            else if (ms && !mf)
                mspl::k_spline_interp<T><<<P, 256, 0, st>>>(UP, out_k[t], out_k[t] + (long)spline_M * O, out[t],
                                                            out[t] + (long)P * O, T(md.r_max), mlp_spline, P, O);
        };
        if constexpr (std::is_same<T, float>::value) {
            if (fused_mlp && mlp_tc >= 0) {
                ensure_mlpf_attrs();
                const int ntiles = (Pm + mlpt::TP - 1) / mlpt::TP;
                if (mlp_tc == mlpt::TF32)
                    mlpt::k_fwd<mlpt::TF32><<<ntiles, mlpt::NTH, mlpt::SMEM_FWD, st>>>(
                        ef_in, def_in, pe0_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], out_dst, h3_dst,
                        float(md.act), Pm);
                else
                    mlpt::k_fwd<mlpt::BF16><<<ntiles, mlpt::NTH, mlpt::SMEM_FWD, st>>>(
                        ef_in, def_in, pe0_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], out_dst, h3_dst,
                        float(md.act), Pm);
                post_spline();
                KCHECK();
                return;
            }
            if (fused_mlp) {
#ifndef LST_LEAN
                if (mlp_tiny_active()) {
                    if (mlp_tiny >= 2) {
                        const int ntiles = (Pm + mspl::TP_4 - 1) / mspl::TP_4;
                        mspl::k_mlp_fwd_sp4<<<ntiles, mspl::NTH_4, mspl::SMEM_FWD_SP4, st>>>(
                            ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], out_dst, nullptr,
                            float(md.act), Pm);
                    } else if (Pm <= 80) {
                        mspl::k_mlp_fwd_sp_t<2><<<(Pm + 1) / 2, mspl::NTH_T, mspl::smem_fwd_sp_t<2>(), st>>>(
                            ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], out_dst, nullptr,
                            float(md.act), Pm);
                    } else if (Pm <= 160) {
                        mspl::k_mlp_fwd_sp_t<4><<<(Pm + 3) / 4, mspl::NTH_T, mspl::smem_fwd_sp_t<4>(), st>>>(
                            ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], out_dst, nullptr,
                            float(md.act), Pm);
                    } else {
                        mspl::k_mlp_fwd_sp_t<8><<<(Pm + 7) / 8, mspl::NTH_T, mspl::smem_fwd_sp_t<8>(), st>>>(
                            ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], out_dst, nullptr,
                            float(md.act), Pm);
                    }
                } else if (ms && mlp_spline_small && mlp_sp4 && Pm <= 260) {
                    const int ntiles = (Pm + mspl::TP_4 - 1) / mspl::TP_4;
                    mspl::k_mlp_fwd_sp4<<<ntiles, mspl::NTH_4, mspl::SMEM_FWD_SP4, st>>>(
                        ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], out_dst, h3_dst, float(md.act),
                        Pm);
                } else if (ms && mlp_spline_small && mlp_tc2) {
                    const int ntiles = (Pm + mspl::TP_8 - 1) / mspl::TP_8;
                    mspl::k_mlp_fwd_sp_tc<<<dim3(ntiles, 1), 256, mspl::SMEM_FWD_TC, st>>>(
                        ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], out_dst, h3_dst,
                        mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], out_dst, h3_dst, float(md.act), Pm);
                } else
#endif
                if (ms && mlp_spline_small && ((mlp_spv2 == 1 && Pm <= 34) || (mlp_spv2 == 2 && Pm <= 260))) {
                    const bool svz = mlp_savez && z1_k[t] && z2_k[t] && z3_k[t];
                    const int r0 = t ? 9 : 2;
                    const float* w1p = use_raw ? raw[r0 + 0] : mlpW[t][0];
                    const float* w2p = use_raw ? raw[r0 + 1] : mlpW[t][1];
                    const float* w3p = use_raw ? raw[r0 + 2] : mlpW[t][2];
                    const float* w4p = use_raw ? raw[r0 + 3] : mlpW[t][3];
                    const float sw1 = use_raw ? (1.f / std::sqrt((float)NB)) : 1.f;
                    const float swH = use_raw ? (1.f / std::sqrt((float)H)) : 1.f;
                    if ((mlp_t2 & 2) && Pm <= 34) {
                        const int ntiles = (Pm + mspl::TP_T2 - 1) / mspl::TP_T2;
                        mspl::k_mlp_fwd_sp_v2_t2<<<ntiles, mspl::NTH_V2, mspl::SMEM_FWD_SPT2, st>>>(
                            ef_in, def_in, w1p, w2p, w3p, w4p, out_dst, h3_dst, float(md.act),
                            Pm, svz ? z1_k[t] : nullptr, svz ? z2_k[t] : nullptr, svz ? z3_k[t] : nullptr,
                            sw1, swH, swH, swH);
                    } else {
                        const int ntiles = (Pm + mspl::TP_V2 - 1) / mspl::TP_V2;
                        mspl::k_mlp_fwd_sp_v2<<<ntiles, mspl::NTH_V2, mspl::SMEM_FWD_SPV2, st>>>(
                            ef_in, def_in, w1p, w2p, w3p, w4p, out_dst, h3_dst, float(md.act),
                            Pm, svz ? z1_k[t] : nullptr, svz ? z2_k[t] : nullptr, svz ? z3_k[t] : nullptr,
                            sw1, swH, swH, swH);
                    }
                } else if (ms && mlp_spline_small) {
                    const int ntiles = (Pm + mspl::TP_F - 1) / mspl::TP_F;
                    mspl::k_mlp_fwd_sp<<<ntiles, mspl::NTH_F, mspl::SMEM_FWD_SP, st>>>(
                        ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], out_dst, h3_dst, float(md.act),
                        Pm);
                } else {
                    ensure_mlpf_attrs();
                    const int ntiles = (Pm + mlpf::TP - 1) / mlpf::TP;
                    if (mlp_fwd_swz)
                        mlpf::k_mlp_fwd_swz<<<ntiles, mlpf::NTH, mlpf::SMEM_FWD, st>>>(
                            ef_in, def_in, pe0_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], out_dst, h3_dst,
                            float(md.act), Pm);
                    else
                        mlpf::k_mlp_fwd<<<ntiles, mlpf::NTH, mlpf::SMEM_FWD, st>>>(
                            ef_in, def_in, pe0_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], out_dst, h3_dst,
                            float(md.act), Pm);
                }
                post_spline();
                KCHECK();
                return;
            }
        }
        const T* xe_in = ms ? xe_k : xe;
        T* z1d = ms ? z1_k[t] : z1[t];
        T* z2d = ms ? z2_k[t] : z2[t];
        T* z3d = ms ? z3_k[t] : z3[t];
        T* h1d = ms ? h1_k[t] : h1[t];
        T* h2d = ms ? h2_k[t] : h2[t];
        long PH = (long)Pm * H;
        lgemm(false, false, 2 * Pm, H, NB, T(1), xe_in, NB, mlpW[t][0], H, T(0), z1d, H);
        k_silu_t_fwd<T><<<nblk(PH), TPB, 0, st>>>(z1d, h1d, PH, T(md.act));
        lgemm(false, false, 2 * Pm, H, H, T(1), h1d, H, mlpW[t][1], H, T(0), z2d, H);
        k_silu_t_fwd<T><<<nblk(PH), TPB, 0, st>>>(z2d, h2d, PH, T(md.act));
        lgemm(false, false, 2 * Pm, H, H, T(1), h2d, H, mlpW[t][2], H, T(0), z3d, H);
        k_silu_t_fwd<T><<<nblk(PH), TPB, 0, st>>>(z3d, h3_dst, PH, T(md.act));
        lgemm(false, false, 2 * Pm, O, H, T(1), h3_dst, H, mlpW[t][3], O, T(0), out_dst, O);
        post_spline();
        KCHECK();
    }

    // both MLP backwards in one launch (LST_MLP_DUAL=1; FP32 fused path only, else two mlp_bwd calls)
    bool mlp_dual_ok() const {
        if constexpr (std::is_same<T, float>::value)
            return mlp_dual && fused_mlp && mlp_tc < 0 && !mlp_split && !mlp_bwd2;
        return false;
    }
    void mlp_bwd_dual() {
        if constexpr (std::is_same<T, float>::value) {
            ensure_mlpf_attrs();
            for (int t = 0; t < 2; ++t) {
                wg_begin();
                lgemm(true, false, H, O, 2 * P, T(1), h3[t], H, gout[t], O, T(0), gmlp[t][3], O);
                wg_end();
                if (!zero_fused) {
                    CK(cudaMemsetAsync(gmlp[t][0], 0, sizeof(float) * NB * H, st));
                    CK(cudaMemsetAsync(gmlp[t][1], 0, sizeof(float) * H * H, st));
                    CK(cudaMemsetAsync(gmlp[t][2], 0, sizeof(float) * H * H, st));
                }
                mlpf::k_mlp_wt<<<(mlpf::WT_SIZE + 255) / 256, 256, 0, st>>>(mlpW[t][1], mlpW[t][2], mlpW[t][3],
                                                                              mlpT[t]);
            }
            mlpf::k_mlp_bwd<false><<<mlp_bwd_grid, mlpf::NTH, mlpf::SMEM_BWD, st>>>(
                EF, DEF, b.pe0, mlpW[0][0], mlpW[0][1], mlpW[0][2], mlpT[0], gout[0], nullptr, gmlp[0][0],
                gmlp[0][1], gmlp[0][2], float(md.act), P, mlpW[1][0], mlpW[1][1], mlpW[1][2], mlpT[1], gout[1],
                gmlp[1][0], gmlp[1][1], gmlp[1][2]);
            KCHECK();
        }
    }

    void mlp_bwd(int t, bool dw4_on_s2 = false) {
        const bool ms = mspl_active();
        const bool mf = mspl_fused();
        const bool mp = mspl_poly_bwd();
        const bool mtiny = mlp_tiny_active();
        const bool fuse_crt = mtiny && mp && std::is_same<T, float>::value && C == 64;
        if (mp && !fuse_crt) {
            mspl::k_spline_c_to_rt_bwd<T><<<spline_M, O, 0, st>>>(gC_k[t], gout_k[t], gout_k[t] + (long)spline_M * O,
                                                                  sp_h(), mlp_spline, C, shell_rcut4(t), T(sp_r_lo),
                                                                  T(shell_width));
            KCHECK();
        } else if (ms && !mf && !mp) {
            mspl::k_spline_scatter<T><<<P, 256, 0, st>>>(UP, gout[t], gout[t] + (long)P * O, gout_k[t],
                                                         gout_k[t] + (long)spline_M * O, T(md.r_max), mlp_spline, P, O);
            KCHECK();
        }
        const int Pm = ms ? spline_M : P;
        const T* ef_in = ms ? EF_k : EF;
        const T* def_in = ms ? DEF_k : DEF;
        const int* pe0_in = ms ? pe0_k : b.pe0;
        const T* h3_in = ms ? h3_k[t] : h3[t];
        const T* gout_in = ms ? gout_k[t] : gout[t];
        const bool msp4 = std::is_same<T, float>::value && fused_mlp && ms && mlp_spline_small && mlp_tc < 0 &&
                          mlp_sp4 && !mtiny && Pm <= 260;
        if (!(fused_mlp && mlp_split) && !mtiny) {
            auto run_dw4 = [&]() {
                if constexpr (std::is_same<T, float>::value) {
                    if (msp4 || (ms && H == mspl::SP_H && O == mspl::SP_O)) {
                        mspl::k_mlp_dw4<<<dim3(8, 4), 128, 0, st>>>(h3_in, gout_in, gmlp[t][3], 2 * Pm);
                        KCHECK();
                    } else {
                        lgemm(true, false, H, O, 2 * Pm, T(1), h3_in, H, gout_in, O, T(0), gmlp[t][3], O);
                    }
                } else {
                    lgemm(true, false, H, O, 2 * Pm, T(1), h3_in, H, gout_in, O, T(0), gmlp[t][3], O);
                }
            };
            if (dw4_on_s2 && par()) {
                fork(1);
                OnSide side(this);
                run_dw4();
            } else {
                wg_begin();
                run_dw4();
                wg_end();
            }
        }
        if constexpr (std::is_same<T, float>::value) {
            if (fused_mlp) {
                if (!zero_fused) {
                    CK(cudaMemsetAsync(gmlp[t][0], 0, sizeof(float) * NB * H, st));
                    CK(cudaMemsetAsync(gmlp[t][1], 0, sizeof(float) * H * H, st));
                    CK(cudaMemsetAsync(gmlp[t][2], 0, sizeof(float) * H * H, st));
                    if (mtiny) CK(cudaMemsetAsync(gmlp[t][3], 0, sizeof(float) * H * O, st));
                }
#ifndef LST_LEAN
                if (mtiny) {
                    const float* gin = fuse_crt ? gC_k[t] : gout_in;
                    const float sph = float(sp_h());
                    if (mlp_tiny >= 2) {
                        const int ntiles = (Pm + mspl::TP_4 - 1) / mspl::TP_4;
                        if (fuse_crt)
                            mspl::k_mlp_bwd_sp4_fused<true><<<ntiles, mspl::NTH_4, mspl::SMEM_BWD_SP4_F, st>>>(
                                ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], gin, gmlp[t][0],
                                gmlp[t][1], gmlp[t][2], gmlp[t][3], float(md.act), Pm, sph);
                        else
                            mspl::k_mlp_bwd_sp4_fused<false><<<ntiles, mspl::NTH_4, mspl::SMEM_BWD_SP4_F, st>>>(
                                ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], gin, gmlp[t][0],
                                gmlp[t][1], gmlp[t][2], gmlp[t][3], float(md.act), Pm, sph);
                    } else if (Pm <= 80) {
                        if (fuse_crt)
                            mspl::k_mlp_bwd_sp_t<2, true><<<(Pm + 1) / 2, mspl::NTH_T, mspl::smem_bwd_sp_t<2>(), st>>>(
                                ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], gin, gmlp[t][0],
                                gmlp[t][1], gmlp[t][2], gmlp[t][3], float(md.act), Pm, sph);
                        else
                            mspl::k_mlp_bwd_sp_t<2, false><<<(Pm + 1) / 2, mspl::NTH_T, mspl::smem_bwd_sp_t<2>(), st>>>(
                                ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], gin, gmlp[t][0],
                                gmlp[t][1], gmlp[t][2], gmlp[t][3], float(md.act), Pm, sph);
                    } else if (Pm <= 160) {
                        if (fuse_crt)
                            mspl::k_mlp_bwd_sp_t<4, true><<<(Pm + 3) / 4, mspl::NTH_T, mspl::smem_bwd_sp_t<4>(), st>>>(
                                ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], gin, gmlp[t][0],
                                gmlp[t][1], gmlp[t][2], gmlp[t][3], float(md.act), Pm, sph);
                        else
                            mspl::k_mlp_bwd_sp_t<4, false><<<(Pm + 3) / 4, mspl::NTH_T, mspl::smem_bwd_sp_t<4>(), st>>>(
                                ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], gin, gmlp[t][0],
                                gmlp[t][1], gmlp[t][2], gmlp[t][3], float(md.act), Pm, sph);
                    } else {
                        if (fuse_crt)
                            mspl::k_mlp_bwd_sp_t<8, true><<<(Pm + 7) / 8, mspl::NTH_T, mspl::smem_bwd_sp_t<8>(), st>>>(
                                ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], gin, gmlp[t][0],
                                gmlp[t][1], gmlp[t][2], gmlp[t][3], float(md.act), Pm, sph);
                        else
                            mspl::k_mlp_bwd_sp_t<8, false><<<(Pm + 7) / 8, mspl::NTH_T, mspl::smem_bwd_sp_t<8>(), st>>>(
                                ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], gin, gmlp[t][0],
                                gmlp[t][1], gmlp[t][2], gmlp[t][3], float(md.act), Pm, sph);
                    }
                    KCHECK();
                    return;
                }
                if (msp4) {
                    const int ntiles = (Pm + mspl::TP_4 - 1) / mspl::TP_4;
                    mspl::k_mlp_bwd_sp4<<<ntiles, mspl::NTH_4, mspl::SMEM_BWD_SP4, st>>>(
                        ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], gout_in, gmlp[t][0], gmlp[t][1],
                        gmlp[t][2], float(md.act), Pm);
                    KCHECK();
                    return;
                }
#endif
                if (ms && mlp_spline_small && mlp_sp8 && mlp_tc < 0 && Pm <= 260) {
#ifndef LST_LEAN
                    if (mlp_tc2) {
                        const int ntiles = (Pm + mspl::TP_8 - 1) / mspl::TP_8;
                        mspl::k_mlp_bwd_sp8_tc<<<ntiles, mspl::NTH_8, mspl::SMEM_BWD_SP8, st>>>(
                            ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], gout_in, gmlp[t][0],
                            gmlp[t][1], gmlp[t][2], float(md.act), Pm);
                    } else
#endif
                    if ((mlp_spv2 == 1 && Pm <= 34) || (mlp_spv2 == 2 && Pm <= 260)) {
                        const int ntiles = (Pm + mspl::TP_V2 - 1) / mspl::TP_V2;
                        mspl::k_mlp_bwd_sp4_v2<<<ntiles, mspl::NTH_V2, mspl::SMEM_BWD_SPV2, st>>>(
                            ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], gout_in, gmlp[t][0],
                            gmlp[t][1], gmlp[t][2], float(md.act), Pm);
                    } else {
                        const int ntiles = (Pm + mspl::TP_8 - 1) / mspl::TP_8;
                        mspl::k_mlp_bwd_sp8<<<ntiles, mspl::NTH_8, mspl::SMEM_BWD_SP8, st>>>(
                            ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpW[t][3], gout_in, gmlp[t][0],
                            gmlp[t][1], gmlp[t][2], float(md.act), Pm);
                    }
                    KCHECK();
                    return;
                }
                ensure_mlpf_attrs();
                const int bgrid = ms ? std::max(1, std::min((Pm + mlpf::TP - 1) / mlpf::TP, mlp_bwd_grid)) : mlp_bwd_grid;
                const int tgrid = ms ? std::max(1, std::min((Pm + mlpt::TP - 1) / mlpt::TP, mlp_tc_grid)) : mlp_tc_grid;
                mlpf::k_mlp_wt<<<(mlpf::WT_SIZE + 255) / 256, 256, 0, st>>>(mlpW[t][1], mlpW[t][2], mlpW[t][3],
                                                                              mlpT[t]);
                if (ms && mlp_spline_small && mlp_tc < 0) {
                    const int ntiles = (Pm + mspl::TP_B - 1) / mspl::TP_B;
                    mspl::k_mlp_bwd_sp<<<ntiles, mspl::NTH_B, mspl::SMEM_BWD_SP, st>>>(
                        ef_in, def_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpT[t], gout_in, gmlp[t][0], gmlp[t][1],
                        gmlp[t][2], float(md.act), Pm);
                    KCHECK();
                    return;
                }
                if (mlp_tc >= 0 && !mlp_split) {
                    if (mlp_tc == mlpt::TF32)
                        mlpt::k_bwd<mlpt::TF32><<<tgrid, mlpt::NTH, mlpt::SMEM_BWD, st>>>(
                            ef_in, def_in, pe0_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpT[t], gout_in, gmlp[t][0],
                            gmlp[t][1], gmlp[t][2], float(md.act), Pm);
                    else
                        mlpt::k_bwd<mlpt::BF16><<<tgrid, mlpt::NTH, mlpt::SMEM_BWD, st>>>(
                            ef_in, def_in, pe0_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpT[t], gout_in, gmlp[t][0],
                            gmlp[t][1], gmlp[t][2], float(md.act), Pm);
                } else if (mlp_split && !ms) {
                    CK(cudaMemsetAsync(gmlp[t][3], 0, sizeof(float) * H * O, st));
                    mlpf::k_mlp_top<<<bgrid, mlpf::NTH, mlpf::SMEM_BWD, st>>>(h3_in, mlpT[t], gout_in, gh,
                                                                              gmlp[t][3], Pm);
                    mlpf::k_mlp_bwd<true><<<bgrid, mlpf::NTH, mlpf::SMEM_BWD, st>>>(
                        ef_in, def_in, pe0_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpT[t], gout_in, gh, gmlp[t][0],
                        gmlp[t][1], gmlp[t][2], float(md.act), Pm);
#ifndef LST_LEAN
                } else if (mlp_bwd2 && !ms) {
                    mlpf2::k_mlp_bwd2<<<mlp_bwd2_grid, mlpf2::NTH, mlpf2::SMEM_BWD, st>>>(
                        ef_in, def_in, pe0_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpT[t], gout_in, gmlp[t][0],
                        gmlp[t][1], gmlp[t][2], float(md.act), Pm);
#endif
                } else if (mlp_pipe && mlp_noz1) {
                    mlpf::k_mlp_bwdp_noz1<<<bgrid, mlpf::NTH, mlpf::SMEM_BWDP_NOZ1, st>>>(
                        ef_in, def_in, pe0_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpT[t], gout_in, gmlp[t][0],
                        gmlp[t][1], gmlp[t][2], float(md.act), Pm, mlp_l2pf);
                } else if (mlp_pipe) {
                    mlpf::k_mlp_bwdp<<<bgrid, mlpf::NTH, mlpf::SMEM_BWDP, st>>>(
                        ef_in, def_in, pe0_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpT[t], gout_in, gmlp[t][0],
                        gmlp[t][1], gmlp[t][2], float(md.act), Pm, mlp_l2pf);
                } else {
                    mlpf::k_mlp_bwd<false><<<bgrid, mlpf::NTH, mlpf::SMEM_BWD, st>>>(
                        ef_in, def_in, pe0_in, mlpW[t][0], mlpW[t][1], mlpW[t][2], mlpT[t], gout_in, nullptr,
                        gmlp[t][0], gmlp[t][1], gmlp[t][2], float(md.act), Pm);
                }
                KCHECK();
                return;
            }
        }
        const T* xe_in = ms ? xe_k : xe;
        const T* z1d = ms ? z1_k[t] : z1[t];
        const T* z2d = ms ? z2_k[t] : z2[t];
        const T* z3d = ms ? z3_k[t] : z3[t];
        const T* h1d = ms ? h1_k[t] : h1[t];
        const T* h2d = ms ? h2_k[t] : h2[t];
        T* ghd = ms ? gh_k : gh;
        T* gzd = ms ? gz_k : gz;
        long PH = (long)Pm * H;
        T act = T(md.act);
        lgemm(false, true, 2 * Pm, H, O, T(1), gout_in, O, mlpW[t][3], O, T(0), ghd, H);
        k_silu_t_bwd<T><<<nblk(PH), TPB, 0, st>>>(z3d, ghd, gzd, PH, act);
        lgemm(true, false, H, H, 2 * Pm, T(1), h2d, H, gzd, H, T(0), gmlp[t][2], H);
        lgemm(false, true, 2 * Pm, H, H, T(1), gzd, H, mlpW[t][2], H, T(0), ghd, H);
        k_silu_t_bwd<T><<<nblk(PH), TPB, 0, st>>>(z2d, ghd, gzd, PH, act);
        lgemm(true, false, H, H, 2 * Pm, T(1), h1d, H, gzd, H, T(0), gmlp[t][1], H);
        lgemm(false, true, 2 * Pm, H, H, T(1), gzd, H, mlpW[t][1], H, T(0), ghd, H);
        k_silu_t_bwd<T><<<nblk(PH), TPB, 0, st>>>(z1d, ghd, gzd, PH, act);
        lgemm(true, false, NB, H, 2 * Pm, T(1), xe_in, NB, gzd, H, T(0), gmlp[t][0], H);
        KCHECK();
    }

    void body(bool update, bool eval_only = false, bool energy_only = false, int soap_mode = 0,
              bool geo_in = false, bool geo_out = false) {
        const long NC = (long)N * C, N16 = 16 * NC;
        const T act = T(md.act), scale = T(md.scale);
        T* R0 = out[0];
        T* T0t = out[0] + (long)P * O;
        T* R1 = out[1];
        T* T1t = out[1] + (long)P * O;
        const bool fp_pair = force_pair_active();
        const int wtarget = N >= 640 ? 9 : 6;
        // ---------------- gather and weights
        const bool use_ks = edge_ksort_active();
        const bool pb = pair_basis && !in_debug;
        const bool cz = cz_active();
        const bool ms = mspl_active();
        const bool mf = mspl_fused();
        const int pm = mspl_poly_mode();
        const bool mpb = mspl_poly_bwd();
        const bool sdx = sdx_active();
        const bool pdu = prod_dedup_active();
        const bool mtiny = mlp_tiny_active();
        const long gwk_n = (long)S * (pdu ? POLY_U : K) * C;
        const bool slim_g = (gather_slim > 0) && use_ks && pb && cz && !pair_bal && !slot_kb_active() &&
                            !force_kb_active() && fp_pair && !in_debug;
        const bool fast_g = std::is_same<T, float>::value && (gather_slim >= 2) && slim_g && ms && mf &&
                            ks_topo.emeta_pre && ks_topo.p_vec4 && ks_topo.p_sr &&
                            ((edge_h2 & 5) == 5) && !(edge_h2 & 2) && (slot_opt == 7) && (act_half_mode() > 0) &&
                            (pm == 2) && (C == 64) && !slot_pp && !conv_opt && !conv_c64 && !force_dir &&
                            (force_opt == 0) && !eval_only;
        par_mask = 1;
        const bool skip_views = eval_only && eval_cache && eval_views_valid && ms && !in_debug;
        const bool do_sp = (start_par > 0) && par() && wg_side && ms && !in_debug && !skip_views;
        const bool do_sp2 = do_sp && (start_par >= 2);
        const bool do_sdef = do_sp && !do_sp2 && start_defer && d_mg_pv_t0 && d_mg_pv_mq && m0_species && pdu && cz;
        const bool do_early_t0 = do_sp && !do_sp2 && !do_sdef && early_t0 && d_mg_pv_t0 && d_mg_pv_mq && mf &&
                                 (act_half_mode() > 0) && t0s_pair && S <= 4 && use_ks && m0_species && cz && !in_debug;
        const int ro_fuse_mode = (std::is_same<T, float>::value && cz && C == 64 && hid == 16 && !in_debug) ? ro_fuse : 0;
        const bool do_q_split = q_split && nops_opt && (C % 32 == 0);
        const bool can_geo_pipe = (graph_pipe >= 2) && update && !eval_only && !in_debug &&
                                  do_sp && do_early_t0 && !do_sp2 && !do_sdef && !slot_pp &&
                                  slim_g && pb && ms && m0_species && wg_side &&
                                  grad_fuse && pdu && have_optim && gf_prep && !soap_cond_en &&
                                  soap_coal && n_mmats && soap_on && soap_smem_on && soap_smem_ok &&
                                  soap_tc && soap_ema_fuse && muon_wd == 0.0 &&
                                  std::is_same<T, float>::value && par() && (opt.update_clip <= 0);
        const bool use_geo_in = geo_in && can_geo_pipe;
        const bool use_geo_out = geo_out && can_geo_pipe;
        cudaEvent_t e_t0_ready = nullptr;

        auto launch_zero_only = [&]() {
            if (zero_fused) {
                ZeroSegs z{};
                int k = 0;
                auto add = [&](void* p, size_t bytes) {
                    z.p[k] = reinterpret_cast<unsigned*>(p);
                    z.n[k] = (long)(bytes / 4);
                    ++k;
                };
                add(Eg, sizeof(T) * G1);
                add(mv, sizeof(T) * 9 * G1);
                add(d_norm2, sizeof(double));
                if (std::is_same<T, float>::value && fused_mlp)
                    for (int t = 0; t < 2; ++t) {
                        add(gmlp[t][0], sizeof(float) * NB * H);
                        add(gmlp[t][1], sizeof(float) * H * H);
                        add(gmlp[t][2], sizeof(float) * H * H);
                        if (mtiny) add(gmlp[t][3], sizeof(float) * H * O);
                    }
                if (prod_direct) {
                    add(gW0, sizeof(T) * gwk_n);
                    add(gW1, sizeof(T) * gwk_n);
                }
                if (m0_species) add(M0s, sizeof(T) * 4L * C * SC);
                if (wbal_active()) add(wl1s, sizeof(T) * 4L * C * C);
                if (cz) {
                    add(Qs, sizeof(T) * C * ZW);
                    add(wp1s, sizeof(T) * C * C);
                    add(wro1s, sizeof(T) * C * hid);
                    if (nops_opt) add(wro2s, sizeof(T) * hid);
                    add(T0s, sizeof(T) * S * C);
                }
                if (std::is_same<T, float>::value && mgemm_sk && nops_opt && !fast_mgemm && mgemm_v3 == 0 && C == 64) {
                    add(gwl0, sizeof(T) * 4L * C * C);
                    add(gwp0, sizeof(T) * C * C);
                }
                if (mpb) {
                    add(gC_k[0], sizeof(T) * 4L * mlp_spline * O);
                    add(gC_k[1], sizeof(T) * 4L * mlp_spline * O);
                } else if (ms) {
                    add(gout_k[0], sizeof(T) * 2L * spline_M * O);
                    add(gout_k[1], sizeof(T) * 2L * spline_M * O);
                }
                if (sdx) {
                    add(x1b, sizeof(T) * NC);
                    if (!t0s_dir_ok()) add(Xd0, sizeof(T) * NC);
                    add(Xd1, sizeof(T) * NC);
                }
                k_zero_words<<<dim3(8, k), 256, 0, st>>>(z);
                KCHECK();
            } else {
                if (prod_direct) {
                    CK(cudaMemsetAsync(gW0, 0, sizeof(T) * gwk_n, st));
                    CK(cudaMemsetAsync(gW1, 0, sizeof(T) * gwk_n, st));
                }
                if (m0_species) CK(cudaMemsetAsync(M0s, 0, sizeof(T) * 4L * C * SC, st));
                if (wbal_active()) CK(cudaMemsetAsync(wl1s, 0, sizeof(T) * 4L * C * C, st));
                if (cz) {
                    CK(cudaMemsetAsync(Qs, 0, sizeof(T) * C * ZW, st));
                    CK(cudaMemsetAsync(wp1s, 0, sizeof(T) * C * C, st));
                    CK(cudaMemsetAsync(wro1s, 0, sizeof(T) * C * hid, st));
                    if (nops_opt) CK(cudaMemsetAsync(wro2s, 0, sizeof(T) * hid, st));
                    CK(cudaMemsetAsync(T0s, 0, sizeof(T) * S * C, st));
                }
                if (mpb) {
                    CK(cudaMemsetAsync(gC_k[0], 0, sizeof(T) * 4L * mlp_spline * O, st));
                    CK(cudaMemsetAsync(gC_k[1], 0, sizeof(T) * 4L * mlp_spline * O, st));
                } else if (ms) {
                    CK(cudaMemsetAsync(gout_k[0], 0, sizeof(T) * 2L * spline_M * O, st));
                    CK(cudaMemsetAsync(gout_k[1], 0, sizeof(T) * 2L * spline_M * O, st));
                }
                if (sdx) {
                    CK(cudaMemsetAsync(x1b, 0, sizeof(T) * NC, st));
                    CK(cudaMemsetAsync(Xd0, 0, sizeof(T) * NC, st));
                    CK(cudaMemsetAsync(Xd1, 0, sizeof(T) * NC, st));
                }
            }
        };
        auto launch_zero_and_views = [&]() {
            launch_zero_only();
            prepare_views();
        };

        if (do_sp) {
            // Wave 39 (do_sf): overlap k_scale_multi on s3 BEFORE T0/M0/Q while st starts k_gather->k_basis_pair
            // AND s2 starts mlp_fwd(0, use_raw=true) immediately at t=0.00 us (saves 2.6-3.5 us/step)!
            const bool do_sf = do_early_t0 && start_fuse && std::is_same<T, float>::value && fused_mlp && ms &&
                               mlp_spline_small && mlp_tc < 0 && !mlp_tiny_active() && !mlp_sp4 && !mlp_tc2 &&
                               !mlp_dual_sp && ((mlp_spv2 == 1 && spline_M <= 34) || (mlp_spv2 == 2 && spline_M <= 260)) &&
                               H == mspl::SP_H && O == mspl::SP_O;
            if (do_sp2) {
                wg_begin();
                prepare_views_scale();
                CK(cudaEventRecord(evf[5], st));
                prepare_views_early();
                wg_end();
                CK(cudaStreamWaitEvent(s2, evf[5], 0));
            } else if (do_sf) {
                fork(0);
            } else {
                prepare_views_scale();
                fork(0);
            }
            {
                OnSide side(this);
                bool dual_tc = false;
                if constexpr (std::is_same<T, float>::value) {
#ifndef LST_LEAN
                    if (mlp_tc2 && fused_mlp && ms && mlp_spline_small && mlp_tc < 0 && !mlp_tiny_active() && !mlp_sp4 &&
                        mspl_poly_mode() == 2) {
                        const int Pm = spline_M;
                        const int ntiles = (Pm + mspl::TP_8 - 1) / mspl::TP_8;
                        mspl::k_mlp_fwd_sp_tc<<<dim3(ntiles, 2), 256, mspl::SMEM_FWD_TC, st>>>(
                            EF_k, DEF_k, mlpW[0][0], mlpW[0][1], mlpW[0][2], mlpW[0][3], out_k[0], h3_k[0],
                            mlpW[1][0], mlpW[1][1], mlpW[1][2], mlpW[1][3], out_k[1], h3_k[1], float(md.act), Pm);
                        mspl::k_spline_rt_to_ch<<<mlp_spline, O, 0, st>>>(out_k[0], out_k[0] + (long)spline_M * O,
                                                                          reinterpret_cast<uint2*>(C_k[0]), float(sp_h()),
                                                                          mlp_spline, O, shell_rcut4(0), float(sp_r_lo),
                                                                          shell_width);
                        CK(cudaEventRecord(evf[4], st));
                        mspl::k_spline_rt_to_ch<<<mlp_spline, O, 0, st>>>(out_k[1], out_k[1] + (long)spline_M * O,
                                                                          reinterpret_cast<uint2*>(C_k[1]), float(sp_h()),
                                                                          mlp_spline, O, shell_rcut4(1), float(sp_r_lo),
                                                                          shell_width);
                        dual_tc = true;
                    } else if (mlp_dual_sp && fused_mlp && ms && mlp_spline_small && mlp_tc < 0 && !mlp_tiny_active() &&
                               !mlp_sp4 && mspl_poly_mode() == 2) {
                        const int Pm = spline_M;
                        const int ntiles = (Pm + mspl::TP_F - 1) / mspl::TP_F;
                        mspl::k_mlp_fwd_sp2<<<dim3(ntiles, 2), mspl::NTH_F, mspl::SMEM_FWD_SP, st>>>(
                            EF_k, DEF_k, mlpW[0][0], mlpW[0][1], mlpW[0][2], mlpW[0][3], out_k[0], h3_k[0],
                            mlpW[1][0], mlpW[1][1], mlpW[1][2], mlpW[1][3], out_k[1], h3_k[1], float(md.act), Pm);
                        mspl::k_spline_rt_to_ch<<<mlp_spline, O, 0, st>>>(out_k[0], out_k[0] + (long)spline_M * O,
                                                                          reinterpret_cast<uint2*>(C_k[0]), float(sp_h()),
                                                                          mlp_spline, O, shell_rcut4(0), float(sp_r_lo),
                                                                          shell_width);
                        CK(cudaEventRecord(evf[4], st));
                        mspl::k_spline_rt_to_ch<<<mlp_spline, O, 0, st>>>(out_k[1], out_k[1] + (long)spline_M * O,
                                                                          reinterpret_cast<uint2*>(C_k[1]), float(sp_h()),
                                                                          mlp_spline, O, shell_rcut4(1), float(sp_r_lo),
                                                                          shell_width);
                        dual_tc = true;
                    }
#endif
                }
                if (!dual_tc) {
                    mlp_fwd(0, do_sf);
                    CK(cudaEventRecord(evf[4], st));
                    mlp_fwd(1, do_sf);
                }
            }
            if (!do_sp2) {
                wg_begin();
                if (do_sf) prepare_views_scale();
                if (do_sdef) {
                    // Wave 35: only compute 1-tile T0 here (zero contention with mlp_fwd(0)!); defer M0/Q to conv_fwd(0).
                    launch_mgemm(d_mg_pv_t0, mg_pv_t0_tiles);
                    KCHECK();
                } else if (do_early_t0) {
                    // Wave 36: compute 1-tile T0 first -> signal e_t0_ready (~4 us), then immediately run M0/Q on s3!
                    launch_mgemm(d_mg_pv_t0, mg_pv_t0_tiles);
                    KCHECK();
                    e_t0_ready = evw[nevw++ % 32];
                    CK(cudaEventRecord(e_t0_ready, st));
                    if (mq_after_mlp0) CK(cudaStreamWaitEvent(st, evf[4], 0));
                    prepare_views_wcat();
                    launch_mgemm(d_mg_pv_mq, mg_pv_mq_tiles);
                    KCHECK();
                } else {
                    prepare_views_early();
                }
                wg_end();
            }
        }
        if (!use_geo_in) {
            long gmax = std::max<long>(std::max<long>(E, N + 1), std::max<long>(P, G1));
            if (fast_g) {
                if constexpr (std::is_same<T, float>::value) {
                    const int nb_p = nblk(P), nb_e = nblk(E), nb_n = nblk(N + 1);
                    k_gather_basis_fast<<<nb_p + nb_e + nb_n + 1, TPB, 0, st>>>(
                        d_step, d_fids, sh.B, ks_topo, ks_frames, b, N, E, P, G1, S, F, Y, UP, nb_p, nb_e, nb_n);
                }
            } else if (slim_g)
                k_gather<T, true><<<nblk(gmax), TPB, 0, st>>>(eval_only ? d_estep : d_step, eval_only ? d_efids : d_fids,
                                                              sh.B, use_ks ? ks_topo : topo, use_ks ? ks_frames : frames, b,
                                                              N, E, P, G1, S, fp_pair ? F : nullptr);
            else
                k_gather<T, false><<<nblk(gmax), TPB, 0, st>>>(eval_only ? d_estep : d_step, eval_only ? d_efids : d_fids,
                                                               sh.B, use_ks ? ks_topo : topo, use_ks ? ks_frames : frames, b,
                                                               N, E, P, G1, S, fp_pair ? F : nullptr);
        }
        if (slot_pp && !eval_only) {
            KCHECK();
            k_pair_ptr<<<nblk((long)P + 1), TPB, 0, st>>>(b.pe0, b.rcv, b.pptr, P, N);
        }
        if (m0_species && !do_sp) {
            KCHECK();
            segg::k_spc_group<<<1, 256, 0, st>>>(b.spc, b.spc_off, b.spc_idx, b.spc_tiles, N, S, b.spc_tiles32,
                                                 b.spc_wsk, wtarget);
        }
        KCHECK();
        if (skip_views) {
            CK(cudaMemsetAsync(Eg, 0, sizeof(T) * G1, st));
            if (!energy_only) {
                CK(cudaMemsetAsync(mv, 0, sizeof(T) * 9 * G1, st));
                if (sdx) CK(cudaMemsetAsync(x1b, 0, sizeof(T) * NC, st));
            }
        } else if (!do_sp) {
            launch_zero_and_views();
            pack_segg_w();
        }
        // ---------------- edges and radial MLPs
        const bool nm = nm_active();
        const bool wbal = wbal_active();
        const int ahm = act_half_mode();
        const bool use_m0h = segg_h16 && M0_h != nullptr;
        auto launch_m0_fwd = [&](const T* in_t, T* out_t) {
            if constexpr (std::is_same<T, float>::value) {
                if (ahm > 0 && segg_v2) {
                    const int mt = segg_mt;
                    const int gy = (segg::spc_max_tiles(N, S) + mt - 1) / mt + 4 * S;
                    if (ahm == 1)
                        segg::k_seg_gemm_spc_h_v2<__nv_bfloat16, false><<<dim3(1, gy), 256, 0, st>>>(
                            reinterpret_cast<const __nv_bfloat16*>(in_t), M0, (long)C * SC, SC,
                            reinterpret_cast<__nv_bfloat16*>(out_t), b.spc_off, b.spc_idx, b.spc_tiles, N, S, mt,
                            nullptr, nullptr, use_m0h ? reinterpret_cast<const __nv_bfloat16*>(M0_h) : nullptr);
                    else
                        segg::k_seg_gemm_spc_h_v2<__half, false><<<dim3(1, gy), 256, 0, st>>>(
                            reinterpret_cast<const __half*>(in_t), M0, (long)C * SC, SC,
                            reinterpret_cast<__half*>(out_t), b.spc_off, b.spc_idx, b.spc_tiles, N, S, mt,
                            nullptr, nullptr, use_m0h ? reinterpret_cast<const __half*>(M0_h) : nullptr);
                    return;
                }
                if (ahm == 1) {
                    segg::k_seg_gemm_spc_h<__nv_bfloat16, false><<<dim3(1, segg::spc_max_tiles(N, S)), 256, 0, st>>>(
                        reinterpret_cast<const __nv_bfloat16*>(in_t), M0, (long)C * SC, SC,
                        reinterpret_cast<__nv_bfloat16*>(out_t), b.spc_off, b.spc_idx, b.spc_tiles, N, S);
                    return;
                } else if (ahm == 2) {
                    segg::k_seg_gemm_spc_h<__half, false><<<dim3(1, segg::spc_max_tiles(N, S)), 256, 0, st>>>(
                        reinterpret_cast<const __half*>(in_t), M0, (long)C * SC, SC,
                        reinterpret_cast<__half*>(out_t), b.spc_off, b.spc_idx, b.spc_tiles, N, S);
                    return;
                }
            }
            if (nm)
                segg::k_seg_gemm_spc<T, false, true><<<dim3(C / segg::TN, segg::spc_max_tiles(N, S)), 256, 0, st>>>(
                    in_t, M0, (long)C * SC, SC, C, out_t, b.spc_off, b.spc_idx, b.spc_tiles, N, S);
            else
                segg::k_seg_gemm_spc<T, false, false><<<dim3(C / segg::TN, segg::spc_max_tiles(N, S)), 256, 0, st>>>(
                    in_t, M0, (long)C * SC, SC, C, out_t, b.spc_off, b.spc_idx, b.spc_tiles, N, S);
        };
        auto launch_m0_bwd = [&](const T* in_t, T* out_t, int zip_mode = 0, const T* G_zip = nullptr,
                                 uint32_t* GA_zip = nullptr) {
            if constexpr (std::is_same<T, float>::value) {
                if (ahm > 0 && segg_v2) {
                    const int mt = segg_mt;
                    const int gy = (segg::spc_max_tiles(N, S) + mt - 1) / mt + 4 * S;
                    if (ahm == 1) {
                        const auto* ih = reinterpret_cast<const __nv_bfloat16*>(in_t);
                        auto* oh = reinterpret_cast<__nv_bfloat16*>(out_t);
                        const auto* gh = reinterpret_cast<const __nv_bfloat16*>(G_zip);
                        const auto* m0h = use_m0h ? reinterpret_cast<const __nv_bfloat16*>(M0_h) : nullptr;
                        if (zip_mode == 4) segg::k_seg_gemm_spc_h_v2<__nv_bfloat16, true, 4><<<dim3(1, gy), 256, 0, st>>>(ih, M0, (long)C * SC, SC, oh, b.spc_off, b.spc_idx, b.spc_tiles, N, S, mt, gh, GA_zip, m0h);
                        else if (zip_mode == 1) segg::k_seg_gemm_spc_h_v2<__nv_bfloat16, true, 1><<<dim3(1, gy), 256, 0, st>>>(ih, M0, (long)C * SC, SC, oh, b.spc_off, b.spc_idx, b.spc_tiles, N, S, mt, gh, GA_zip, m0h);
                        else if (zip_mode == 2) segg::k_seg_gemm_spc_h_v2<__nv_bfloat16, true, 2><<<dim3(1, gy), 256, 0, st>>>(ih, M0, (long)C * SC, SC, oh, b.spc_off, b.spc_idx, b.spc_tiles, N, S, mt, gh, GA_zip, m0h);
                        else segg::k_seg_gemm_spc_h_v2<__nv_bfloat16, true, 0><<<dim3(1, gy), 256, 0, st>>>(ih, M0, (long)C * SC, SC, oh, b.spc_off, b.spc_idx, b.spc_tiles, N, S, mt, nullptr, nullptr, m0h);
                    } else {
                        const auto* ih = reinterpret_cast<const __half*>(in_t);
                        auto* oh = reinterpret_cast<__half*>(out_t);
                        const auto* gh = reinterpret_cast<const __half*>(G_zip);
                        const auto* m0h = use_m0h ? reinterpret_cast<const __half*>(M0_h) : nullptr;
                        if (zip_mode == 4) segg::k_seg_gemm_spc_h_v2<__half, true, 4><<<dim3(1, gy), 256, 0, st>>>(ih, M0, (long)C * SC, SC, oh, b.spc_off, b.spc_idx, b.spc_tiles, N, S, mt, gh, GA_zip, m0h);
                        else if (zip_mode == 1) segg::k_seg_gemm_spc_h_v2<__half, true, 1><<<dim3(1, gy), 256, 0, st>>>(ih, M0, (long)C * SC, SC, oh, b.spc_off, b.spc_idx, b.spc_tiles, N, S, mt, gh, GA_zip, m0h);
                        else if (zip_mode == 2) segg::k_seg_gemm_spc_h_v2<__half, true, 2><<<dim3(1, gy), 256, 0, st>>>(ih, M0, (long)C * SC, SC, oh, b.spc_off, b.spc_idx, b.spc_tiles, N, S, mt, gh, GA_zip, m0h);
                        else segg::k_seg_gemm_spc_h_v2<__half, true, 0><<<dim3(1, gy), 256, 0, st>>>(ih, M0, (long)C * SC, SC, oh, b.spc_off, b.spc_idx, b.spc_tiles, N, S, mt, nullptr, nullptr, m0h);
                    }
                    return;
                }
                if (ahm == 1) {
                    segg::k_seg_gemm_spc_h<__nv_bfloat16, true><<<dim3(1, segg::spc_max_tiles(N, S)), 256, 0, st>>>(
                        reinterpret_cast<const __nv_bfloat16*>(in_t), M0, (long)C * SC, SC,
                        reinterpret_cast<__nv_bfloat16*>(out_t), b.spc_off, b.spc_idx, b.spc_tiles, N, S);
                    return;
                } else if (ahm == 2) {
                    segg::k_seg_gemm_spc_h<__half, true><<<dim3(1, segg::spc_max_tiles(N, S)), 256, 0, st>>>(
                        reinterpret_cast<const __half*>(in_t), M0, (long)C * SC, SC,
                        reinterpret_cast<__half*>(out_t), b.spc_off, b.spc_idx, b.spc_tiles, N, S);
                    return;
                }
            }
            if (nm)
                segg::k_seg_gemm_spc<T, true, true><<<dim3(C / segg::TN, segg::spc_max_tiles(N, S)), 256, 0, st>>>(
                    in_t, M0, (long)C * SC, SC, C, out_t, b.spc_off, b.spc_idx, b.spc_tiles, N, S);
            else
                segg::k_seg_gemm_spc<T, true, false><<<dim3(C / segg::TN, segg::spc_max_tiles(N, S)), 256, 0, st>>>(
                    in_t, M0, (long)C * SC, SC, C, out_t, b.spc_off, b.spc_idx, b.spc_tiles, N, S);
        };
        auto launch_m0_wgrad = [&](const T* a_t, const T* b_t) {
            const int ntc = C / segg::TM;
            if constexpr (std::is_same<T, float>::value) {
                if (ahm > 0 && segg_v2) {
                    if (ahm == 1)
                        segg::k_seg_wgrad_spc_dyn_h_v2<__nv_bfloat16><<<dim3(wtarget + S, 16), 256, 0, st>>>(
                            reinterpret_cast<const __nv_bfloat16*>(a_t), reinterpret_cast<const __nv_bfloat16*>(b_t),
                            M0s, (long)C * SC, SC, b.spc_off, b.spc_idx, b.spc_wsk, S);
                    else
                        segg::k_seg_wgrad_spc_dyn_h_v2<__half><<<dim3(wtarget + S, 16), 256, 0, st>>>(
                            reinterpret_cast<const __half*>(a_t), reinterpret_cast<const __half*>(b_t), M0s,
                            (long)C * SC, SC, b.spc_off, b.spc_idx, b.spc_wsk, S);
                    return;
                }
                if (ahm == 1) {
                    segg::k_seg_wgrad_spc_dyn_h<__nv_bfloat16><<<dim3(wtarget + S, 16), 256, 0, st>>>(
                        reinterpret_cast<const __nv_bfloat16*>(a_t), reinterpret_cast<const __nv_bfloat16*>(b_t), M0s,
                        (long)C * SC, SC, b.spc_off, b.spc_idx, b.spc_wsk, S);
                    return;
                } else if (ahm == 2) {
                    segg::k_seg_wgrad_spc_dyn_h<__half><<<dim3(wtarget + S, 16), 256, 0, st>>>(
                        reinterpret_cast<const __half*>(a_t), reinterpret_cast<const __half*>(b_t), M0s, (long)C * SC,
                        SC, b.spc_off, b.spc_idx, b.spc_wsk, S);
                    return;
                }
            }
            if (wbal && wgrad_opt) {
                const int max_wsk = wtarget + S;
                if (nm)
                    segg::k_seg_wgrad_spc_dyn<T, true><<<dim3(max_wsk, 16 * ntc * ntc), 256, 0, st>>>(
                        a_t, b_t, M0s, (long)C * SC, SC, C, b.spc_off, b.spc_idx, b.spc_wsk, N, S);
                else
                    segg::k_seg_wgrad_spc_dyn<T, false><<<dim3(max_wsk, 16 * ntc * ntc), 256, 0, st>>>(
                        a_t, b_t, M0s, (long)C * SC, SC, C, b.spc_off, b.spc_idx, b.spc_wsk, N, S);
            } else if (wbal) {
                if (nm)
                    segg::k_seg_wgrad_spc_bal<T, true><<<dim3(2, 16 * S * ntc * ntc), 256, 0, st>>>(
                        a_t, b_t, M0s, (long)C * SC, SC, C, b.spc_off, b.spc_idx, N, S);
                else
                    segg::k_seg_wgrad_spc_bal<T, false><<<dim3(2, 16 * S * ntc * ntc), 256, 0, st>>>(
                        a_t, b_t, M0s, (long)C * SC, SC, C, b.spc_off, b.spc_idx, N, S);
            } else {
                segg::k_seg_wgrad_spc<T><<<dim3(4, 4 * S * ntc * ntc), 256, 0, st>>>(
                    a_t, b_t, M0s, (long)C * SC, SC, C, b.spc_off, b.spc_idx, N, S);
            }
        };
        auto launch_wl1_wgrad = [&](const T* a_t, const T* b_t, T beta) {
            if constexpr (std::is_same<T, float>::value) {
                if (ahm > 0 && segg_v2) {
                    if (ahm == 1)
                        segg::k_seg_wgrad_wl1_h_v2<__nv_bfloat16><<<dim3(wtarget, 16), 256, 0, st>>>(
                            reinterpret_cast<const __nv_bfloat16*>(a_t), reinterpret_cast<const __nv_bfloat16*>(b_t),
                            wl1s, (long)C * C, N);
                    else
                        segg::k_seg_wgrad_wl1_h_v2<__half><<<dim3(wtarget, 16), 256, 0, st>>>(
                            reinterpret_cast<const __half*>(a_t), reinterpret_cast<const __half*>(b_t), wl1s,
                            (long)C * C, N);
                    KCHECK();
                    return;
                }
                if (ahm == 1) {
                    segg::k_seg_wgrad_wl1_h<__nv_bfloat16><<<dim3(wtarget, 16), 256, 0, st>>>(
                        reinterpret_cast<const __nv_bfloat16*>(a_t), reinterpret_cast<const __nv_bfloat16*>(b_t), wl1s,
                        (long)C * C, N);
                    KCHECK();
                    return;
                } else if (ahm == 2) {
                    segg::k_seg_wgrad_wl1_h<__half><<<dim3(wtarget, 16), 256, 0, st>>>(
                        reinterpret_cast<const __half*>(a_t), reinterpret_cast<const __half*>(b_t), wl1s, (long)C * C,
                        N);
                    KCHECK();
                    return;
                }
            }
            if (wbal) {
                const int ntc = C / segg::TM;
                const int sk_wl = wgrad_opt ? wtarget : 4;
                if (nm)
                    segg::k_seg_wgrad_wl1<T, true><<<dim3(sk_wl, 16 * ntc * ntc), 256, 0, st>>>(
                        a_t, b_t, wl1s, (long)C * C, C, N);
                else
                    segg::k_seg_wgrad_wl1<T, false><<<dim3(sk_wl, 16 * ntc * ntc), 256, 0, st>>>(
                        a_t, b_t, wl1s, (long)C * C, C, N);
                KCHECK();
            } else {
                per_l_wgrad(a_t, C, b_t, C, wl1s, (long)C * C, beta);
            }
        };
        auto launch_prod_fwd_u = [&](const T* B_in, const T* W_in, T* f_out) {
            if constexpr (std::is_same<T, float>::value) {
                if (ahm == 1) { k_prod_fwd_u<float, true, __nv_bfloat16><<<nblk(NC), TPB, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(B_in), W_in, b.spc, f_out, N, C); return; }
                if (ahm == 2) { k_prod_fwd_u<float, true, __half><<<nblk(NC), TPB, 0, st>>>(reinterpret_cast<const __half*>(B_in), W_in, b.spc, f_out, N, C); return; }
            }
            if (nm) k_prod_fwd_u<T, true><<<nblk(NC), TPB, 0, st>>>(B_in, W_in, b.spc, f_out, N, C);
            else k_prod_fwd_u<T, false><<<nblk(NC), TPB, 0, st>>>(B_in, W_in, b.spc, f_out, N, C);
        };
        auto launch_prod_vjp_u = [&](const T* B_in, const T* W_in, const T* fb_in, T* Bb_out) {
            if constexpr (std::is_same<T, float>::value) {
                if (ahm == 1) { k_prod_vjp_u<float, true, __nv_bfloat16><<<nblk(NC), TPB, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(B_in), W_in, b.spc, fb_in, reinterpret_cast<__nv_bfloat16*>(Bb_out), N, C); return; }
                if (ahm == 2) { k_prod_vjp_u<float, true, __half><<<nblk(NC), TPB, 0, st>>>(reinterpret_cast<const __half*>(B_in), W_in, b.spc, fb_in, reinterpret_cast<__half*>(Bb_out), N, C); return; }
            }
            if (nm) k_prod_vjp_u<T, true><<<nblk(NC), TPB, 0, st>>>(B_in, W_in, b.spc, fb_in, Bb_out, N, C);
            else k_prod_vjp_u<T, false><<<nblk(NC), TPB, 0, st>>>(B_in, W_in, b.spc, fb_in, Bb_out, N, C);
        };
        auto launch_prod_rev2_u = [&](const T* B_in, const T* W_in, const T* fb_in, const T* Bbs_in, T* Bs_out, T* fbs_out, T* gW_out) {
            if constexpr (std::is_same<T, float>::value) {
#ifndef LST_LEAN
                if (prod_rev2_spc && ahm > 0 && nm && m0_species && C == 64) {
                    const dim3 gr(N <= 512 ? 4 : 2, nops::spc_max_tiles32(N, S));
                    if (ahm == 1) k_prod_rev2_spc64<__nv_bfloat16><<<gr, 128, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(B_in), W_in, b.spc_off, b.spc_idx, b.spc_tiles32, fb_in, reinterpret_cast<const __nv_bfloat16*>(Bbs_in), reinterpret_cast<__nv_bfloat16*>(Bs_out), fbs_out, gW_out, S);
                    else k_prod_rev2_spc64<__half><<<gr, 128, 0, st>>>(reinterpret_cast<const __half*>(B_in), W_in, b.spc_off, b.spc_idx, b.spc_tiles32, fb_in, reinterpret_cast<const __half*>(Bbs_in), reinterpret_cast<__half*>(Bs_out), fbs_out, gW_out, S);
                    return;
                }
                if (prod_h2 >= 2 && ahm > 0 && nm && m0_species && C == 64) {
                    const dim3 gr(N <= 512 ? 4 : 2, nops::spc_max_tiles32(N, S));
                    if (ahm == 1) k_prod_rev2_h2<__nv_bfloat16><<<gr, 128, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(B_in), W_in, b.spc_off, b.spc_idx, b.spc_tiles32, fb_in, reinterpret_cast<const __nv_bfloat16*>(Bbs_in), reinterpret_cast<__nv_bfloat16*>(Bs_out), fbs_out, gW_out, S);
                    else k_prod_rev2_h2<__half><<<gr, 128, 0, st>>>(reinterpret_cast<const __half*>(B_in), W_in, b.spc_off, b.spc_idx, b.spc_tiles32, fb_in, reinterpret_cast<const __half*>(Bbs_in), reinterpret_cast<__half*>(Bs_out), fbs_out, gW_out, S);
                    return;
                }
#endif
            }
            auto run = [&](auto ob_tag) {
                constexpr bool OB = decltype(ob_tag)::value;
                if constexpr (std::is_same<T, float>::value) {
                    if (ahm == 1) { k_prod_rev2_dir_u<float, true, __nv_bfloat16, OB><<<nblk(NC), TPB, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(B_in), W_in, b.spc, fb_in, reinterpret_cast<const __nv_bfloat16*>(Bbs_in), reinterpret_cast<__nv_bfloat16*>(Bs_out), fbs_out, gW_out, N, C); return; }
                    if (ahm == 2) { k_prod_rev2_dir_u<float, true, __half, OB><<<nblk(NC), TPB, 0, st>>>(reinterpret_cast<const __half*>(B_in), W_in, b.spc, fb_in, reinterpret_cast<const __half*>(Bbs_in), reinterpret_cast<__half*>(Bs_out), fbs_out, gW_out, N, C); return; }
                }
                if (nm) k_prod_rev2_dir_u<T, true, T, OB><<<nblk(NC), TPB, 0, st>>>(B_in, W_in, b.spc, fb_in, Bbs_in, Bs_out, fbs_out, gW_out, N, C);
                else k_prod_rev2_dir_u<T, false, T, OB><<<nblk(NC), TPB, 0, st>>>(B_in, W_in, b.spc, fb_in, Bbs_in, Bs_out, fbs_out, gW_out, N, C);
            };
            if (prod_1bar) run(std::true_type{});
            else run(std::false_type{});
        };
        auto launch_prod_rev1_u = [&](const T* B_in, const T* W_in, const T* fs_in, T* Bs_io, T* gW_out) {
            if constexpr (std::is_same<T, float>::value) {
                if (prod_h2 && ahm > 0 && nm && m0_species && C == 64) {
                    const int ny = nops::spc_max_tiles32(N, S);
                    if (prod_h2_w == 1 || prod_h2_w == 2) {
                        const int gx = (prod_h2_w == 1) ? (N <= 512 ? 16 : 4) : (N <= 512 ? 8 : 4);
                        const dim3 gr(gx, ny);
                        if (ahm == 1) k_prod_rev1_h2<__nv_bfloat16, 2><<<gr, 64, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(B_in), W_in, b.spc_off, b.spc_idx, b.spc_tiles32, fs_in, reinterpret_cast<__nv_bfloat16*>(Bs_io), gW_out, S);
                        else k_prod_rev1_h2<__half, 2><<<gr, 64, 0, st>>>(reinterpret_cast<const __half*>(B_in), W_in, b.spc_off, b.spc_idx, b.spc_tiles32, fs_in, reinterpret_cast<__half*>(Bs_io), gW_out, S);
                    } else {
                        const int gx = (prod_h2_w == 3) ? (N <= 512 ? 8 : 2) : (N <= 512 ? 4 : 2);
                        const dim3 gr(gx, ny);
                        if (ahm == 1) k_prod_rev1_h2<__nv_bfloat16, 4><<<gr, 128, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(B_in), W_in, b.spc_off, b.spc_idx, b.spc_tiles32, fs_in, reinterpret_cast<__nv_bfloat16*>(Bs_io), gW_out, S);
                        else k_prod_rev1_h2<__half, 4><<<gr, 128, 0, st>>>(reinterpret_cast<const __half*>(B_in), W_in, b.spc_off, b.spc_idx, b.spc_tiles32, fs_in, reinterpret_cast<__half*>(Bs_io), gW_out, S);
                    }
                    return;
                }
                if (ahm == 1) { k_prod_rev1_dir_u<float, true, __nv_bfloat16, false><<<nblk(NC), TPB, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(B_in), W_in, b.spc, fs_in, reinterpret_cast<__nv_bfloat16*>(Bs_io), gW_out, N, C); return; }
                if (ahm == 2) { k_prod_rev1_dir_u<float, true, __half, false><<<nblk(NC), TPB, 0, st>>>(reinterpret_cast<const __half*>(B_in), W_in, b.spc, fs_in, reinterpret_cast<__half*>(Bs_io), gW_out, N, C); return; }
            }
            if (nm) k_prod_rev1_dir_u<T, true, T, false><<<nblk(NC), TPB, 0, st>>>(B_in, W_in, b.spc, fs_in, Bs_io, gW_out, N, C);
            else k_prod_rev1_dir_u<T, false, T, false><<<nblk(NC), TPB, 0, st>>>(B_in, W_in, b.spc, fs_in, Bs_io, gW_out, N, C);
        };
        if (!use_geo_in && !fast_g) {
            if (pb) {
                if (ms)
                    k_basis_pair<T, true><<<nblk(P), TPB, 0, st>>>(b.vec, b.pe0, Y, EF, DEF, UP, d_bw, T(md.prefactor),
                                                                   T(md.r_max), md.p_cut, NB, P, b.sp_idx, b.sp_u,
                                                                   sp_invh(), mlp_spline, T(sp_r_lo));
                else
                    k_basis_pair<T, false><<<nblk(P), TPB, 0, st>>>(b.vec, b.pe0, Y, EF, DEF, UP, d_bw, T(md.prefactor),
                                                                    T(md.r_max), md.p_cut, NB, P);
            } else
                k_basis<T><<<nblk(E), TPB, 0, st>>>(b.vec, Y, JY, EF, DEF, U, d_bw, T(md.prefactor), T(md.r_max), md.p_cut,
                                                    NB, E);
            if (!fused_mlp && !ms) k_pair_feats<T><<<nblk((long)P * NB), TPB, 0, st>>>(b.pe0, EF, DEF, xe, P, NB);
        }
        KCHECK();
        par_mask = 1;
        cudaEvent_t e_m0_ready = nullptr, e_w_ready = nullptr;
        if (do_sp) {
            if (do_early_t0) {
                CK(cudaStreamWaitEvent(st, e_t0_ready, 0));
                if (use_geo_in) CK(cudaStreamWaitEvent(st, ev_geo_ready, 0));
                if (!fwd_t0s) k_x0<T><<<nblk(NC), TPB, 0, st>>>(T0, b.spc, x0, N, C);
                CK(cudaStreamWaitEvent(st, evf[4], 0));
                wg_begin();
                if (m0_species && !use_geo_in)
                    segg::k_spc_group<<<1, 256, 0, st>>>(b.spc, b.spc_off, b.spc_idx, b.spc_tiles, N, S, b.spc_tiles32,
                                                         b.spc_wsk, wtarget);
                // conv_fwd(0) uses T0S_DIR=true when fwd_t0s; hide k_x0 in conv_fwd(0)'s shadow on s3!
                if (fwd_t0s) k_x0<T><<<nblk(NC), TPB, 0, st>>>(T0, b.spc, x0, N, C);
                prepare_views_w();
                launch_zero_only();
                pack_segg_w();
                wg_end();
            } else {
                wg_join();
                k_x0<T><<<nblk(NC), TPB, 0, st>>>(T0, b.spc, x0, N, C);
                CK(cudaStreamWaitEvent(st, evf[4], 0));
                wg_begin();
                if (m0_species)
                    segg::k_spc_group<<<1, 256, 0, st>>>(b.spc, b.spc_off, b.spc_idx, b.spc_tiles, N, S, b.spc_tiles32,
                                                         b.spc_wsk, wtarget);
                if (do_sdef) {
                    prepare_views_wcat();
                    launch_mgemm(d_mg_pv_mq, mg_pv_mq_tiles);
                    KCHECK();
                    pack_segg_w();
                    e_m0_ready = evw[nevw++ % 32];
                    CK(cudaEventRecord(e_m0_ready, st));
                    prepare_views_w();
                    e_w_ready = evw[nevw++ % 32];
                    CK(cudaEventRecord(e_w_ready, st));
                    launch_zero_only();
                } else {
                    prepare_views_w();
                    launch_zero_only();
                    pack_segg_w();
                }
                wg_end();
            }
        } else if (skip_views) {
            k_x0<T><<<nblk(NC), TPB, 0, st>>>(T0, b.spc, x0, N, C);
        } else {
            bool dual_mlp_tc_done = false;
            if constexpr (std::is_same<T, float>::value) {
                if (mlp_tc2 && fused_mlp && ms && mlp_spline_small && mlp_tc < 0 && !mlp_tiny_active() && !mlp_sp4 &&
                    mspl_poly_mode() == 2) {
                    const int Pm = spline_M;
                    const int ntiles = (Pm + mspl::TP_8 - 1) / mspl::TP_8;
                    mspl::k_mlp_fwd_sp_tc<<<dim3(ntiles, 2), 256, mspl::SMEM_FWD_TC, st>>>(
                        EF_k, DEF_k, mlpW[0][0], mlpW[0][1], mlpW[0][2], mlpW[0][3], out_k[0], h3_k[0],
                        mlpW[1][0], mlpW[1][1], mlpW[1][2], mlpW[1][3], out_k[1], h3_k[1], float(md.act), Pm);
                    fork(0);
                    mspl::k_spline_rt_to_ch<<<mlp_spline, O, 0, st>>>(out_k[0], out_k[0] + (long)spline_M * O,
                                                                      reinterpret_cast<uint2*>(C_k[0]), float(sp_h()),
                                                                      mlp_spline, O, shell_rcut4(0), float(sp_r_lo),
                                                                      shell_width);
                    {
                        OnSide side(this);
                        mspl::k_spline_rt_to_ch<<<mlp_spline, O, 0, st>>>(out_k[1], out_k[1] + (long)spline_M * O,
                                                                          reinterpret_cast<uint2*>(C_k[1]), float(sp_h()),
                                                                          mlp_spline, O, shell_rcut4(1), float(sp_r_lo),
                                                                          shell_width);
                    }
                    KCHECK();
                    dual_mlp_tc_done = true;
                }
            }
            if (!dual_mlp_tc_done) {
                fork(0);
                if (mlp0_first) {
                    mlp_fwd(0);
                    {
                        OnSide side(this);
                        mlp_fwd(1);
                    }
                } else {
                    {
                        OnSide side(this);
                        mlp_fwd(1);
                    }
                    mlp_fwd(0);
                }
            }
            // ---------------- forward
            k_x0<T><<<nblk(NC), TPB, 0, st>>>(T0, b.spc, x0, N, C);
        }
        if (mf)
            conv_sp<true, false, 1, false, false>(0, x0, nullptr, Y, nullptr, nullptr, nullptr, A0, nullptr);
        else
            conv<true, false, 1, false, false>(x0, nullptr, R0, Y, nullptr, nullptr, nullptr, nullptr, A0, nullptr);
        if (do_sp) {
            if (do_sdef) CK(cudaStreamWaitEvent(st, e_m0_ready, 0));
            else wg_join();
        }
        if (m0_species) {
            launch_m0_fwd(A0, B0);
        } else {
            per_l(A0, C, M0, (long)C * SC, SC, false, SC, tmp, T(0));
            k_pick<T><<<nblk(N16), TPB, 0, st>>>(tmp, b.spc, B0, N, C, S);
        }
        if (do_sdef) CK(cudaStreamWaitEvent(st, e_w_ready, 0));
        if (pdu)
            launch_prod_fwd_u(B0, W0, f0);
        else if (nm)
            k_prod_fwd<T, true><<<nblk(NC), TPB, 0, st>>>(B0, W0, b.spc, f0, N, C);
        else
            k_prod_fwd<T, false><<<nblk(NC), TPB, 0, st>>>(B0, W0, b.spc, f0, N, C);
        KCHECK();
        if (cz) {
            if (do_q_split)
                nops::k_q_fwd_c32<T><<<dim3(2 * (C / 32), nops::spc_max_tiles32(N, S)), 128, 0, st>>>(
                    f0, Q, b.spc_off, b.spc_idx, b.spc_tiles32, z0, x1, sc, C, ZW, S);
            else
                nops::k_q_fwd<T><<<dim3(2 * (C / nops::CN), nops::spc_max_tiles32(N, S)), 128, 0, st>>>(
                    f0, Q, b.spc_off, b.spc_idx, b.spc_tiles32, z0, x1, sc, C, ZW, S);
        } else {
            lgemm(false, false, N, ZW, C, T(1), f0, C, Q, ZW, T(0), z, ZW);
            k_split_z<T><<<nblk(NC), TPB, 0, st>>>(z, b.spc, x1, sc, N, C, ZW);
        }
        if (do_sdef) wg_join();                    // ensure launch_zero_only() finished well before Eg/mv/bwd!
        if (!skip_views) join(1);                  // R1
        par_mask = 2;
        if (mf)
            conv_sp<true, false, 1, false, false>(1, x1, nullptr, Y, nullptr, nullptr, nullptr, A1, nullptr);
        else
            conv<true, false, 1, false, false>(x1, nullptr, R1, Y, nullptr, nullptr, nullptr, nullptr, A1, nullptr);
        per_l(A1, C, wl1, (long)C * C, C, false, C, B1, T(0));
        if (pdu)
            launch_prod_fwd_u(B1, W1, f1);
        else if (nm)
            k_prod_fwd<T, true><<<nblk(NC), TPB, 0, st>>>(B1, W1, b.spc, f1, N, C);
        else
            k_prod_fwd<T, false><<<nblk(NC), TPB, 0, st>>>(B1, W1, b.spc, f1, N, C);
        if (cz) {
            if (!zero_fused) CK(cudaMemsetAsync(Eg, 0, sizeof(T) * G1, st));
            if (ro_fuse_mode >= 2) {
                if constexpr (std::is_same<T, float>::value) {
                    auto launch_wp1_ro = [&](auto do_bwd_tag) {
                        constexpr bool DB = decltype(do_bwd_tag)::value;
                        float* ub_p = DB ? ub : nullptr;
                        float* f2b_p = DB ? f2b : nullptr;
                        float* f1b_p = DB ? f1b : nullptr;
                        if (ro_na == 16)
                            nops::k_wp1_readout_fwd_bwd<DB, 16><<<(N + 15) / 16, 256, 0, st>>>(
                                f1, wp1, sc, f2, wro1, z0, wro2, b.ngraph, u, tt, ne, Eg, ub_p, f2b_p, f1b_p,
                                act, scale, N);
                        else if (ro_na == 12)
                            nops::k_wp1_readout_fwd_bwd<DB, 12><<<(N + 11) / 12, 192, 0, st>>>(
                                f1, wp1, sc, f2, wro1, z0, wro2, b.ngraph, u, tt, ne, Eg, ub_p, f2b_p, f1b_p,
                                act, scale, N);
                        else
                            nops::k_wp1_readout_fwd_bwd<DB, 8><<<(N + 7) / 8, 128, 0, st>>>(
                                f1, wp1, sc, f2, wro1, z0, wro2, b.ngraph, u, tt, ne, Eg, ub_p, f2b_p, f1b_p,
                                act, scale, N);
                    };
                    if (eval_only && energy_only) {
                        launch_wp1_ro(std::false_type{});
                        KCHECK();
                        return;
                    }
                    launch_wp1_ro(std::true_type{});
                    KCHECK();
                }
            } else if (ro_fuse_mode == 1) {
                if constexpr (std::is_same<T, float>::value) {
                    nops::k_node_gemm<T, false, true><<<dim3(C / nops::CN, (N + nops::RM - 1) / nops::RM), 128, 0, st>>>(
                        f1, wp1, sc, f2, N, C, C, C);
                    if (eval_only && energy_only) {
                        nops::k_readout_fwd_bwd<false><<<(N + 15) / 16, 128, 0, st>>>(
                            f2, wro1, z0, wro2, b.ngraph, u, tt, ne, Eg, nullptr, nullptr, act, scale, N);
                        KCHECK();
                        return;
                    }
                    nops::k_readout_fwd_bwd<true><<<(N + 15) / 16, 128, 0, st>>>(
                        f2, wro1, z0, wro2, b.ngraph, u, tt, ne, Eg, ub, f2b, act, scale, N);
                    nops::k_node_gemm<T, true, false><<<dim3(C / nops::CN, (N + nops::RM - 1) / nops::RM), 128, 0, st>>>(
                        f2b, wp1, nullptr, f1b, N, C, C, C);
                    KCHECK();
                }
            } else {
                nops::k_node_gemm<T, false, true><<<dim3(C / nops::CN, (N + nops::RM - 1) / nops::RM), 128, 0, st>>>(
                    f1, wp1, sc, f2, N, C, C, C);
                nops::k_node_gemm<T, false, false><<<dim3((hid + nops::CN - 1) / nops::CN, (N + nops::RM - 1) / nops::RM),
                                                     128, 0, st>>>(f2, wro1, nullptr, u, N, C, hid, hid);
                nops::k_readout_z0<T><<<nblk(N), TPB, 0, st>>>(u, z0, wro2, b.ngraph, tt, ne, Eg, act, scale, N, hid);
                KCHECK();
                if (eval_only && energy_only) return;
                // ---------------- force pass
                k_ub<T><<<nblk((long)N * hid), TPB, 0, st>>>(u, wro2, ub, act, scale, N, hid);
                KCHECK();
                nops::k_node_gemm<T, true, false><<<dim3(C / nops::CN, (N + nops::RM - 1) / nops::RM), 128, 0, st>>>(
                    ub, wro1, nullptr, f2b, N, hid, C, hid);
                nops::k_node_gemm<T, true, false><<<dim3(C / nops::CN, (N + nops::RM - 1) / nops::RM), 128, 0, st>>>(
                    f2b, wp1, nullptr, f1b, N, C, C, C);
                KCHECK();
            }
        } else {
            k_copy<T><<<nblk(NC), TPB, 0, st>>>(f2, sc, NC);
            KCHECK();
            lgemm(false, false, N, C, C, T(1), f1, C, wp1, C, T(1), f2, C);
            lgemm(false, false, N, hid, C, T(1), f2, C, wro1, hid, T(0), u, hid);
            if (!zero_fused) CK(cudaMemsetAsync(Eg, 0, sizeof(T) * G1, st));
            k_readout<T><<<nblk(N), TPB, 0, st>>>(u, z, wro2, b.ngraph, tt, ne, Eg, act, scale, N, hid, ZW);
            if (eval_only && energy_only) return;      // energy-only lst_evaluate (VarPro): Eg, f0, tt ready!
            // ---------------- force pass
            k_ub<T><<<nblk((long)N * hid), TPB, 0, st>>>(u, wro2, ub, act, scale, N, hid);
            KCHECK();
            lgemm(false, true, N, C, hid, T(1), ub, hid, wro1, hid, T(0), f2b, C);
            lgemm(false, true, N, C, C, T(1), f2b, C, wp1, C, T(0), f1b, C);
        }
        if (pdu)
            launch_prod_vjp_u(B1, W1, f1b, B1b);
        else if (nm)
            k_prod_vjp<T, true><<<nblk(NC), TPB, 0, st>>>(B1, W1, b.spc, f1b, B1b, N, C);
        else
            k_prod_vjp<T, false><<<nblk(NC), TPB, 0, st>>>(B1, W1, b.spc, f1b, B1b, N, C);
        KCHECK();
        per_l(B1b, C, wl1, (long)C * C, C, true, C, Gn1, T(0));
        const bool fdir2 = force_dir2_ok();
        const bool fp2 = force_pair2_ok();
#ifndef LST_LEAN
        if (fp2) {
            if (!zero_fused) CK(cudaMemsetAsync(mv, 0, sizeof(T) * 9 * G1, st));
            if constexpr (std::is_same<T, float>::value) {
                const int fpes = force_pair_es;
                const int* ford = (force_order && topo.norder) ? b.norder : nullptr;
                const int grid = (N * fpes + 3) / 4;
                auto run_fp1 = [&](auto tag) {
                    using H = decltype(tag);
                    const H* G1h = reinterpret_cast<const H*>(Gn1);
                    if (shell_active) eh2::k_force_pair_1l<H, true, true, false><<<grid, 128, 0, st>>>(b.rptr, b.rptr0, b.emeta, b.ngraph, Y, UP, G1h, C_k[1], x1, F, mv, x1b, sp_invh(), N, fpes, ford, shell_kcut(1, 2), shell_kcut(1, 3));
                    else eh2::k_force_pair_1l<H, true, false, false><<<grid, 128, 0, st>>>(b.rptr, b.rptr0, b.emeta, b.ngraph, Y, UP, G1h, C_k[1], x1, F, mv, x1b, sp_invh(), N, fpes, ford);
                };
                if (ahm == 1) run_fp1(__nv_bfloat16{});
                else run_fp1(__half{});
                KCHECK();
            }
        } else if (fdir2) {
            if (!zero_fused) CK(cudaMemsetAsync(mv, 0, sizeof(T) * 9 * G1, st));
            if constexpr (std::is_same<T, float>::value) {
                const int fdes = force_dir_es > 0 ? force_dir_es : (N <= 512 ? 2 : 1);
                const int* ford = (force_order && topo.norder) ? b.norder : nullptr;
                const int grid = (N * fdes + 3) / 4;
                auto run_f1 = [&](auto tag) {
                    using H = decltype(tag);
                    const H* G1h = reinterpret_cast<const H*>(Gn1);
                    if (shell_active) eh2::k_force_dir_1l<H, true, true, false><<<grid, 128, 0, st>>>(b.rptr, b.emeta_dir, b.ngraph, Y, UP, G1h, C_k[1], x1, F, mv, x1b, sp_invh(), N, fdes, ford, shell_kcut(1, 2), shell_kcut(1, 3));
                    else eh2::k_force_dir_1l<H, true, false, false><<<grid, 128, 0, st>>>(b.rptr, b.emeta_dir, b.ngraph, Y, UP, G1h, C_k[1], x1, F, mv, x1b, sp_invh(), N, fdes, ford);
                };
                if (ahm == 1) run_f1(__nv_bfloat16{});
                else run_f1(__half{});
                KCHECK();
            }
        } else
#endif
        if (mf)
            conv_sp<false, true, 1, false, false>(1, nullptr, Gn1, Y, nullptr, nullptr, nullptr, nullptr, x1b);
        else
            conv<false, true, 1, false, false>(nullptr, Gn1, R1, Y, nullptr, nullptr, nullptr, nullptr, nullptr, x1b);
        if (cz) {
            if (do_q_split)
                nops::k_q_bwd_r16<T><<<dim3(C / 32, 2 * nops::spc_max_tiles32(N, S)), 128, 0, st>>>(
                    nullptr, scale, x1b, f2b, Q, b.spc_off, b.spc_idx, b.spc_tiles32, f0b, C, ZW, S);
            else if (nops_opt)
                nops::k_q_bwd_c32<T><<<dim3(C / 32, nops::spc_max_tiles32(N, S)), 128, 0, st>>>(
                    nullptr, scale, x1b, f2b, Q, b.spc_off, b.spc_idx, b.spc_tiles32, f0b, C, ZW, S);
            else
                nops::k_q_bwd<T><<<dim3(C / nops::CN, nops::spc_max_tiles32(N, S)), 128, 0, st>>>(
                    nullptr, scale, x1b, f2b, Q, b.spc_off, b.spc_idx, b.spc_tiles32, f0b, C, ZW, S);
            KCHECK();
        } else {
            k_build_z<T><<<nblk((long)N * ZW), TPB, 0, st>>>(zb, nullptr, scale, x1b, f2b, b.spc, N, C, ZW);
            KCHECK();
            lgemm(false, true, N, C, ZW, T(1), zb, ZW, Q, ZW, T(0), f0b, C);
        }
        if (pdu)
            launch_prod_vjp_u(B0, W0, f0b, B0b);
        else if (nm)
            k_prod_vjp<T, true><<<nblk(NC), TPB, 0, st>>>(B0, W0, b.spc, f0b, B0b, N, C);
        else
            k_prod_vjp<T, false><<<nblk(NC), TPB, 0, st>>>(B0, W0, b.spc, f0b, B0b, N, C);
        const bool zip_f = zip_ga_force_ok();
        if (m0_species) {
            launch_m0_bwd(B0b, Gn0, zip_f ? 2 : 0, zip_f ? Gn1 : nullptr, zip_f ? GA_buf : nullptr);
            KCHECK();
        } else {
            k_expand<T><<<nblk(16L * N * SC), TPB, 0, st>>>(B0b, b.spc, tmp, N, C, S);
            KCHECK();
            per_l(tmp, SC, M0, (long)C * SC, SC, true, C, Gn0, T(0));
        }
        if (!zero_fused && !fdir2 && !fp2) CK(cudaMemsetAsync(mv, 0, sizeof(T) * 9 * G1, st));
        T* fse = (no_deadw && !in_debug) ? nullptr : se;
        T* fgy = (no_deadw && !in_debug) ? nullptr : gy;
#ifndef LST_LEAN
        if (fp2) {
            if constexpr (std::is_same<T, float>::value) {
                const int fpes = force_pair_es;
                const int* ford = (force_order && topo.norder) ? b.norder : nullptr;
                const int grid = (N * fpes + 3) / 4;
                const bool use_t0 = (S <= 4);
                auto run_fp0 = [&](auto tag) {
                    using H = decltype(tag);
                    const H* G0h = reinterpret_cast<const H*>(Gn0);
                    auto launch = [&](auto sh_tag, auto t0_tag) {
                        constexpr bool SH = decltype(sh_tag)::value;
                        constexpr bool T0D = decltype(t0_tag)::value;
                        eh2::k_force_pair_1l<H, false, SH, T0D><<<grid, 128, 0, st>>>(
                            b.rptr, b.rptr0, b.emeta, b.ngraph, Y, UP, G0h, C_k[0], x0, F, mv, nullptr, sp_invh(), N, fpes,
                            ford, shell_kcut(0, 2), shell_kcut(0, 3), b.spc, T0, S);
                    };
                    if (shell_active && use_t0) launch(std::true_type{}, std::true_type{});
                    else if (shell_active) launch(std::true_type{}, std::false_type{});
                    else if (use_t0) launch(std::false_type{}, std::true_type{});
                    else launch(std::false_type{}, std::false_type{});
                };
                if (ahm == 1) run_fp0(__nv_bfloat16{});
                else run_fp0(__half{});
            }
        } else if (fdir2) {
            if constexpr (std::is_same<T, float>::value) {
                const int fdes = force_dir_es > 0 ? force_dir_es : (N <= 512 ? 2 : 1);
                const int* ford = (force_order && topo.norder) ? b.norder : nullptr;
                const int grid = (N * fdes + 3) / 4;
                const bool use_t0 = (S <= 4);
                auto run_f0 = [&](auto tag) {
                    using H = decltype(tag);
                    const H* G0h = reinterpret_cast<const H*>(Gn0);
                    auto launch = [&](auto sh_tag, auto t0_tag) {
                        constexpr bool SH = decltype(sh_tag)::value;
                        constexpr bool T0D = decltype(t0_tag)::value;
                        eh2::k_force_dir_1l<H, false, SH, T0D><<<grid, 128, 0, st>>>(
                            b.rptr, b.emeta_dir, b.ngraph, Y, UP, G0h, C_k[0], x0, F, mv, nullptr, sp_invh(), N, fdes,
                            ford, shell_kcut(0, 2), shell_kcut(0, 3), b.spc, T0, S);
                    };
                    if (shell_active && use_t0) launch(std::true_type{}, std::true_type{});
                    else if (shell_active) launch(std::true_type{}, std::false_type{});
                    else if (use_t0) launch(std::false_type{}, std::true_type{});
                    else launch(std::false_type{}, std::false_type{});
                };
                if (ahm == 1) run_f0(__nv_bfloat16{});
                else run_f0(__half{});
            }
        } else if (force_kb_active() && C == 64) {
            const dim3 grid(ks_max_tiles, sh.B);
            if constexpr (std::is_same<T, float>::value) {
                if (ahm > 0) {
                    auto run_fkb = [&](auto tag) {
                        using H = decltype(tag);
                        const H* G0h = reinterpret_cast<const H*>(Gn0);
                        const H* G1h = reinterpret_cast<const H*>(Gn1);
                        if (pm == 2) k_force_kb<float, 2, H><<<grid, 64, 0, st>>>(b.f_meta, d_ks_toff, d_ks_tiles, b.puv, b.sp_u, Y, UP, G0h, C_k[0], nullptr, x0, G1h, C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), C);
                        else k_force_kb<float, 1, H><<<grid, 64, 0, st>>>(b.f_meta, d_ks_toff, d_ks_tiles, b.puv, b.sp_u, Y, UP, G0h, C_k[0], nullptr, x0, G1h, C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), C);
                    };
                    if (ahm == 1) run_fkb(__nv_bfloat16{});
                    else run_fkb(__half{});
                } else if (pm == 2) {
                    k_force_kb<float, 2><<<grid, 64, 0, st>>>(b.f_meta, d_ks_toff, d_ks_tiles, b.puv, b.sp_u, Y, UP, Gn0, C_k[0], nullptr, x0, Gn1, C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), C);
                } else if (pm == 1) {
                    k_force_kb<float, 1><<<grid, 64, 0, st>>>(b.f_meta, d_ks_toff, d_ks_tiles, b.puv, b.sp_u, Y, UP, Gn0, C_k[0], nullptr, x0, Gn1, C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), C);
                } else {
                    k_force_kb<float, 0><<<grid, 64, 0, st>>>(b.f_meta, d_ks_toff, d_ks_tiles, b.puv, b.sp_u, Y, UP, Gn0, out_k[0], out_k[0] + (long)spline_M * O, x0, Gn1, out_k[1], out_k[1] + (long)spline_M * O, x1, F, mv, sp_h(), sp_invh(), C);
                }
            } else {
                if (pm == 1)
                    k_force_kb<T, 1><<<grid, 64, 0, st>>>(b.f_meta, d_ks_toff, d_ks_tiles, b.puv, b.sp_u, Y, UP, Gn0, C_k[0], nullptr, x0, Gn1, C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), C);
                else
                    k_force_kb<T, 0><<<grid, 64, 0, st>>>(b.f_meta, d_ks_toff, d_ks_tiles, b.puv, b.sp_u, Y, UP, Gn0, out_k[0], out_k[0] + (long)spline_M * O, x0, Gn1, out_k[1], out_k[1] + (long)spline_M * O, x1, F, mv, sp_h(), sp_invh(), C);
            }
        } else
#endif
        if (fp_pair && C == 64) {
            const int fpes = force_pair_es;
            const int* ford = (force_order && topo.norder) ? b.norder : nullptr;
            if constexpr (std::is_same<T, float>::value) {
                if (ahm > 0 && pm == 2 && (edge_h2 & 4)) {
                    auto run_fh2 = [&](auto tag) {
                        using H = decltype(tag);
                        const H* G0h = reinterpret_cast<const H*>(Gn0);
                        const H* G1h = reinterpret_cast<const H*>(Gn1);
                        const int grid = (N * fpes + 3) / 4;
#ifndef LST_LEAN
                        if (force_dir) {
                            if (use_ks && shell_active) eh2::k_force_dir_h2<H, true, true><<<grid, 128, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, G0h, C_k[0], x0, G1h, C_k[1], x1, F, mv, sp_invh(), N, fpes, ford, b.emeta, shell_kcut(0, 2), shell_kcut(0, 3), shell_kcut(1, 2), shell_kcut(1, 3));
                            else if (use_ks) eh2::k_force_dir_h2<H, true, false><<<grid, 128, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, G0h, C_k[0], x0, G1h, C_k[1], x1, F, mv, sp_invh(), N, fpes, ford, b.emeta);
                            else eh2::k_force_dir_h2<H, false, false><<<grid, 128, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, G0h, C_k[0], x0, G1h, C_k[1], x1, F, mv, sp_invh(), N, fpes, ford, nullptr);
                        } else
#endif
                        if (zip_f) {
                            const uint2* G01 = reinterpret_cast<const uint2*>(GA_buf);
#ifndef LST_LEAN
                            if (force_opt >= 3 && force_opt <= 6 && use_ks) {
                                const bool use_t0 = (S <= 4);
                                const bool use_hf = (force_opt >= 5) && std::is_same<H, __half>::value;
                                const bool use_cg = (force_opt == 4 || force_opt == 6);
                                auto launch_w29 = [&](auto hf_tag, auto cg_tag, auto sh_tag, auto t0_tag) {
                                    constexpr bool HF = decltype(hf_tag)::value;
                                    constexpr bool CG = decltype(cg_tag)::value;
                                    constexpr bool SH = decltype(sh_tag)::value;
                                    constexpr bool T0D = decltype(t0_tag)::value;
                                    eh2::k_force_pair_h2_w29<H, HF, CG, SH, T0D><<<grid, 128, 0, st>>>(
                                        b.rptr, b.rptr0, b.ngraph, Y, UP, C_k[0], x0, C_k[1], x1, F, mv, sp_invh(), N, fpes,
                                        ford, b.emeta, G01, shell_kcut(0, 2), shell_kcut(0, 3), shell_kcut(1, 2), shell_kcut(1, 3),
                                        b.spc, T0, S);
                                };
                                auto disp_sh = [&](auto hf_tag, auto cg_tag) {
                                    if (shell_active && use_t0) launch_w29(hf_tag, cg_tag, std::true_type{}, std::true_type{});
                                    else if (shell_active) launch_w29(hf_tag, cg_tag, std::true_type{}, std::false_type{});
                                    else if (use_t0) launch_w29(hf_tag, cg_tag, std::false_type{}, std::true_type{});
                                    else launch_w29(hf_tag, cg_tag, std::false_type{}, std::false_type{});
                                };
                                if (use_hf && use_cg) disp_sh(std::true_type{}, std::true_type{});
                                else if (use_hf) disp_sh(std::true_type{}, std::false_type{});
                                else if (use_cg) disp_sh(std::false_type{}, std::true_type{});
                                else disp_sh(std::false_type{}, std::false_type{});
                            } else if (force_opt == 1 && use_ks) {
                                const bool use_t0 = (S <= 4);
                                auto launch_fo1 = [&](auto sh_tag, auto t0_tag) {
                                    constexpr bool SH = decltype(sh_tag)::value;
                                    constexpr bool T0D = decltype(t0_tag)::value;
                                    eh2::k_force_pair_h2_opt<H, SH, T0D><<<grid, 128, 0, st>>>(
                                        b.rptr, b.rptr0, b.ngraph, Y, UP, C_k[0], x0, C_k[1], x1, F, mv, sp_invh(), N, fpes,
                                        ford, b.emeta, G01, shell_kcut(0, 2), shell_kcut(0, 3), shell_kcut(1, 2), shell_kcut(1, 3),
                                        b.spc, T0, S);
                                };
                                if (shell_active && use_t0) launch_fo1(std::true_type{}, std::true_type{});
                                else if (shell_active) launch_fo1(std::true_type{}, std::false_type{});
                                else if (use_t0) launch_fo1(std::false_type{}, std::true_type{});
                                else launch_fo1(std::false_type{}, std::false_type{});
                            } else if (force_hfma2 && std::is_same<H, __half>::value) {
                                if (use_ks) eh2::k_force_pair_h2<H, true, true, true><<<grid, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, G0h, C_k[0], x0, G1h, C_k[1], x1, F, mv, sp_invh(), N, fpes, ford, b.emeta, G01);
                                else eh2::k_force_pair_h2<H, false, true, true><<<grid, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, G0h, C_k[0], x0, G1h, C_k[1], x1, F, mv, sp_invh(), N, fpes, ford, nullptr, G01);
                            } else
#endif
                            {
#ifndef LST_LEAN
                                if (use_ks && shell_active) eh2::k_force_pair_h2<H, true, true, false, true><<<grid, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, G0h, C_k[0], x0, G1h, C_k[1], x1, F, mv, sp_invh(), N, fpes, ford, b.emeta, G01, shell_kcut(0, 2), shell_kcut(0, 3), shell_kcut(1, 2), shell_kcut(1, 3));
                                else
#endif
                                if (use_ks) eh2::k_force_pair_h2<H, true, true, false, false><<<grid, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, fast_g ? nullptr : b.vec, G0h, C_k[0], x0, G1h, C_k[1], x1, F, mv, sp_invh(), N, fpes, ford, b.emeta, G01);
                                else eh2::k_force_pair_h2<H, false, true, false, false><<<grid, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, G0h, C_k[0], x0, G1h, C_k[1], x1, F, mv, sp_invh(), N, fpes, ford, nullptr, G01);
                            }
                        } else {
                            if (use_ks) eh2::k_force_pair_h2<H, true, false><<<grid, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, G0h, C_k[0], x0, G1h, C_k[1], x1, F, mv, sp_invh(), N, fpes, ford, b.emeta);
                            else eh2::k_force_pair_h2<H, false, false><<<grid, 128, 0, st>>>(b.rptr, b.rptr0, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, G0h, C_k[0], x0, G1h, C_k[1], x1, F, mv, sp_invh(), N, fpes, ford, nullptr);
                        }
                    };
                    if (ahm == 1) run_fh2(__nv_bfloat16{});
                    else run_fh2(__half{});
                }
#ifndef LST_LEAN
                else if (ahm > 0) {
                    auto run_fp = [&](auto tag) {
                        using H = decltype(tag);
                        const H* G0h = reinterpret_cast<const H*>(Gn0);
                        const H* G1h = reinterpret_cast<const H*>(Gn1);
                        if (pm == 2) k_force_pair_sp64<float, 2, H><<<N * fpes, 64, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, G0h, C_k[0], nullptr, x0, G1h, C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), N, C, fpes, ford);
                        else k_force_pair_sp64<float, 1, H><<<N * fpes, 64, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, G0h, C_k[0], nullptr, x0, G1h, C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), N, C, fpes, ford);
                    };
                    if (ahm == 1) run_fp(__nv_bfloat16{});
                    else run_fp(__half{});
                } else if (force_64t) {
                    if (pm == 2)
                        k_force_pair_sp64<T, 2><<<N * fpes, 64, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph,
                                                                         Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                                                                         C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), N,
                                                                         C, fpes, ford);
                    else if (pm == 1)
                        k_force_pair_sp64<T, 1><<<N * fpes, 64, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph,
                                                                         Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                                                                         C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), N,
                                                                         C, fpes, ford);
                    else
                        k_force_pair_sp64<T, 0><<<N * fpes, 64, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph,
                                                                         Y, UP, b.vec, Gn0, out_k[0],
                                                                         out_k[0] + (long)spline_M * O, x0, Gn1, out_k[1],
                                                                         out_k[1] + (long)spline_M * O, x1, F, mv, sp_h(),
                                                                         sp_invh(), N, C, fpes, ford);
                } else if (pm == 2)
                    k_force_pair_sp<T, 2, 2><<<(N * fpes + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                        C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), N, C, fpes, ford);
                else if (pm == 1)
                    k_force_pair_sp<T, 2, 1><<<(N * fpes + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                        C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), N, C, fpes, ford);
                else
#else
                else
#endif
                    k_force_pair_sp<T, 2, 0><<<(N * fpes + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, out_k[0],
                        out_k[0] + (long)spline_M * O, x0, Gn1, out_k[1], out_k[1] + (long)spline_M * O, x1, F, mv, sp_h(),
                        sp_invh(), N, C, fpes, ford);
            } else {
#ifndef LST_LEAN
                if (force_64t) {
                    if (pm == 1)
                        k_force_pair_sp64<T, 1><<<N * fpes, 64, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph,
                                                                         Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                                                                         C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), N,
                                                                         C, fpes, ford);
                    else
                        k_force_pair_sp64<T, 0><<<N * fpes, 64, 0, st>>>(b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph,
                                                                         Y, UP, b.vec, Gn0, out_k[0],
                                                                         out_k[0] + (long)spline_M * O, x0, Gn1, out_k[1],
                                                                         out_k[1] + (long)spline_M * O, x1, F, mv, sp_h(),
                                                                         sp_invh(), N, C, fpes, ford);
                } else if (pm == 1)
                    k_force_pair_sp<T, 2, 1><<<(N * fpes + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                        C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), N, C, fpes, ford);
                else
#endif
                    k_force_pair_sp<T, 2, 0><<<(N * fpes + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, out_k[0],
                        out_k[0] + (long)spline_M * O, x0, Gn1, out_k[1], out_k[1] + (long)spline_M * O, x1, F, mv, sp_h(),
                        sp_invh(), N, C, fpes, ford);
            }
        }
#ifndef LST_LEAN
        else if (fp_pair && C == 32) {
            const int fpes = force_pair_es;
            if (pm == 2)
                k_force_pair_sp<T, 1, 2><<<(N * fpes + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                    C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), N, C, fpes);
            else if (pm == 1)
                k_force_pair_sp<T, 1, 1><<<(N * fpes + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                    C_k[1], nullptr, x1, F, mv, sp_h(), sp_invh(), N, C, fpes);
            else
                k_force_pair_sp<T, 1, 0><<<(N * fpes + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, out_k[0],
                    out_k[0] + (long)spline_M * O, x0, Gn1, out_k[1], out_k[1] + (long)spline_M * O, x1, F, mv, sp_h(),
                    sp_invh(), N, C, fpes);
        } else if (mf && !force_2pass && C == 64) {
            if (nm && force_64t) {
                if (pm == 2)
                    k_force_node_pair_sp64<T, true, 2><<<N * force_es, 64, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                        C_k[1], nullptr, x1, g, mv, sp_h(), sp_invh(), N, C, force_es,
                        (force_order && topo.norder) ? b.norder : nullptr);
                else if (pm == 1)
                    k_force_node_pair_sp64<T, true, 1><<<N * force_es, 64, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                        C_k[1], nullptr, x1, g, mv, sp_h(), sp_invh(), N, C, force_es,
                        (force_order && topo.norder) ? b.norder : nullptr);
                else
                    k_force_node_pair_sp64<T, true, 0><<<N * force_es, 64, 0, st>>>(
                        b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, out_k[0],
                        out_k[0] + (long)spline_M * O, x0, Gn1, out_k[1], out_k[1] + (long)spline_M * O, x1, g, mv,
                        sp_h(), sp_invh(), N, C, force_es, (force_order && topo.norder) ? b.norder : nullptr);
            } else if (nm && pm == 2)
                k_force_node_pair_sp<T, 2, true, 2><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                    C_k[1], nullptr, x1, g, mv, sp_h(), sp_invh(), N, C, force_es,
                    (force_order && topo.norder) ? b.norder : nullptr);
            else if (nm && pm == 1)
                k_force_node_pair_sp<T, 2, true, 1><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                    C_k[1], nullptr, x1, g, mv, sp_h(), sp_invh(), N, C, force_es,
                    (force_order && topo.norder) ? b.norder : nullptr);
            else if (nm)
                k_force_node_pair_sp<T, 2, true, 0><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, out_k[0],
                    out_k[0] + (long)spline_M * O, x0, Gn1, out_k[1], out_k[1] + (long)spline_M * O, x1, g, mv, sp_h(),
                    sp_invh(), N, C, force_es, (force_order && topo.norder) ? b.norder : nullptr);
            else
                k_force_node_pair_sp<T, 2, false, 0><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, out_k[0],
                    out_k[0] + (long)spline_M * O, x0, Gn1, out_k[1], out_k[1] + (long)spline_M * O, x1, g, mv, sp_h(),
                    sp_invh(), N, C, force_es, (force_order && topo.norder) ? b.norder : nullptr);
        } else if (mf && !force_2pass && C == 32) {
            if (nm && pm == 2)
                k_force_node_pair_sp<T, 1, true, 2><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                    C_k[1], nullptr, x1, g, mv, sp_h(), sp_invh(), N, C, force_es);
            else if (nm && pm == 1)
                k_force_node_pair_sp<T, 1, true, 1><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, C_k[0], nullptr, x0, Gn1,
                    C_k[1], nullptr, x1, g, mv, sp_h(), sp_invh(), N, C, force_es);
            else if (nm)
                k_force_node_pair_sp<T, 1, true, 0><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, out_k[0],
                    out_k[0] + (long)spline_M * O, x0, Gn1, out_k[1], out_k[1] + (long)spline_M * O, x1, g, mv, sp_h(),
                    sp_invh(), N, C, force_es);
            else
                k_force_node_pair_sp<T, 1, false, 0><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.sp_idx, b.sp_u, b.ngraph, Y, UP, b.vec, Gn0, out_k[0],
                    out_k[0] + (long)spline_M * O, x0, Gn1, out_k[1], out_k[1] + (long)spline_M * O, x1, g, mv, sp_h(),
                    sp_invh(), N, C, force_es);
        }
#endif
        else if (pb && !force_2pass && C == 64) {
            if (nm)
                k_force_node_pair<T, 2, true><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.ngraph, Y, UP, b.vec, Gn0, R0, T0t, x0, Gn1, R1, T1t, x1, g, mv, N, C,
                    force_es, (force_order && topo.norder) ? b.norder : nullptr);
            else
                k_force_node_pair<T, 2, false><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.ngraph, Y, UP, b.vec, Gn0, R0, T0t, x0, Gn1, R1, T1t, x1, g, mv, N, C,
                    force_es, (force_order && topo.norder) ? b.norder : nullptr);
        } else if (pb && !force_2pass && C == 32) {
            if (nm)
                k_force_node_pair<T, 1, true><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.ngraph, Y, UP, b.vec, Gn0, R0, T0t, x0, Gn1, R1, T1t, x1, g, mv, N, C,
                    force_es);
            else
                k_force_node_pair<T, 1, false><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(
                    b.rptr, b.snd, b.ps, b.ngraph, Y, UP, b.vec, Gn0, R0, T0t, x0, Gn1, R1, T1t, x1, g, mv, N, C,
                    force_es);
        } else if (C == 64 && force_2pass) {
            k_force_node<T, 2, 1><<<(N + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(b.rptr, b.snd, b.pidx, b.ngraph, Y, U, JY, b.vec, Gn0,
                                                              R0, T0t, x0, Gn1, R1, T1t, x1, se, gy, g, mv, N, C);
            k_force_node<T, 2, 2><<<(N + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(b.rptr, b.snd, b.pidx, b.ngraph, Y, U, JY, b.vec, Gn0,
                                                              R0, T0t, x0, Gn1, R1, T1t, x1, se, gy, g, mv, N, C);
        } else if (C == 64)
            k_force_node<T, 2><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(b.rptr, b.snd, b.pidx, b.ngraph, Y, U, JY, b.vec, Gn0, R0,
                                                           T0t, x0, Gn1, R1, T1t, x1, fse, fgy, g, mv, N, C, force_jyw, force_es,
                                                           (force_order && topo.norder) ? b.norder : nullptr);
        else if (C == 32)
            k_force_node<T, 1><<<(N * force_es + fwpb - 1) / fwpb, 32 * fwpb, 0, st>>>(b.rptr, b.snd, b.pidx, b.ngraph, Y, U, JY, b.vec, Gn0, R0,
                                                           T0t, x0, Gn1, R1, T1t, x1, fse, fgy, g, mv, N, C, force_jyw, force_es);
        else
            throw Error("force pass: C must be 32 or 64");
        if (!fp_pair) {
            if (fpair_warp)
                k_forces_pair_w<T><<<(N + 7) / 8, 256, 0, st>>>(g, b.rptr, b.pidx, b.pe0, b.pe1, b.core_f, F, N);
            else
                k_forces_pair<T><<<nblk(N), TPB, 0, st>>>(g, b.rptr, b.pidx, b.pe0, b.pe1, b.core_f, F, N);
        }
        KCHECK();
        if (eval_only) return;                     // lst_evaluate: energies, forces and virials are complete here
        // ---------------- loss
        const bool do_scot = scot_active();
        const bool do_l1b = loss_par && loss_1blk && N <= 512 && sh.B == 1 && S <= 4 && !in_debug;
        const bool do_lf32 = std::is_same<T, float>::value && loss_par && loss_f32 && !do_l1b && N <= 512 && sh.B == 1 &&
                             G1 <= 32 && S <= 4;
        const bool do_l2b = loss_par && loss_2blk && !do_l1b && !do_lf32 && N <= 512 && sh.B == 1 && G1 <= 32 && S <= 4;
        if (do_l1b) {
            if (ei_on && frames.ei_ref) {
                k_loss_1blk<T, true><<<1, 256, 0, st>>>(
                    Eg, b.core_e, b.e_ref, b.n_atoms, b.w_e, F, b.f_ref, b.w_f, mv, b.volume, b.core_s, b.s_ref,
                    b.w_s, b.counts, d_lw, Sg, Eb, Fb, Ms, d_losses, d_step, N, G1,
                    do_scot ? d_scot : nullptr, ne, b.ei_ref, b.ngraph, b.spc,
                    ei_steps_set ? d_wei : nullptr, ei_weight, Enb, S);
            } else {
                k_loss_1blk<T, false><<<1, 256, 0, st>>>(
                    Eg, b.core_e, b.e_ref, b.n_atoms, b.w_e, F, b.f_ref, b.w_f, mv, b.volume, b.core_s, b.s_ref,
                    b.w_s, b.counts, d_lw, Sg, Eb, Fb, Ms, d_losses, d_step, N, G1,
                    do_scot ? d_scot : nullptr);
            }
            KCHECK();
        } else if (do_lf32) {
            if constexpr (std::is_same<T, float>::value) {
                if (ei_on && frames.ei_ref) {
                    k_loss_1blk_f32<true><<<1, 256, 0, st>>>(
                        Eg, b.core_e, b.e_ref, b.n_atoms, b.w_e, F, b.f_ref, b.w_f, mv, b.volume, b.core_s, b.s_ref,
                        b.w_s, b.counts, d_lw, Sg, Eb, Fb, Ms, d_losses, d_step, N, G1,
                        do_scot ? d_scot : nullptr, ne, b.ei_ref, b.ngraph, b.spc,
                        ei_steps_set ? d_wei : nullptr, ei_weight, Enb, S);
                } else {
                    k_loss_1blk_f32<false><<<1, 256, 0, st>>>(
                        Eg, b.core_e, b.e_ref, b.n_atoms, b.w_e, F, b.f_ref, b.w_f, mv, b.volume, b.core_s, b.s_ref,
                        b.w_s, b.counts, d_lw, Sg, Eb, Fb, Ms, d_losses, d_step, N, G1,
                        do_scot ? d_scot : nullptr);
                }
            }
            KCHECK();
        } else if (do_l2b) {
            if (ei_on && frames.ei_ref) {
                k_loss_2blk<T, true><<<2, 256, 0, st>>>(
                    Eg, b.core_e, b.e_ref, b.n_atoms, b.w_e, F, b.f_ref, b.w_f, mv, b.volume, b.core_s, b.s_ref,
                    b.w_s, b.counts, d_lw, Sg, Eb, Fb, Ms, d_losses, d_step, d_loss_acc, d_loss_done, N, G1,
                    do_scot ? d_fmax : nullptr, do_scot ? d_scot : nullptr,
                    ne, b.ei_ref, b.ngraph, b.spc, ei_steps_set ? d_wei : nullptr, ei_weight, Enb, S);
            } else {
                k_loss_2blk<T, false><<<1, 256, 0, st>>>(
                    Eg, b.core_e, b.e_ref, b.n_atoms, b.w_e, F, b.f_ref, b.w_f, mv, b.volume, b.core_s, b.s_ref,
                    b.w_s, b.counts, d_lw, Sg, Eb, Fb, Ms, d_losses, d_step, d_loss_acc, d_loss_done, N, G1,
                    nullptr, do_scot ? d_scot : nullptr);
            }
            KCHECK();
        } else if (loss_par) {
            if (ei_on && frames.ei_ref) {
                const size_t smem_ei = (size_t)sh.B * S * (sizeof(double) + sizeof(int));
                const int eif = loss_ei_fast ? 1 : 0;
                k_loss_par<T, true><<<nblk(3L * N) + eif, 256, smem_ei, st>>>(
                    Eg, b.core_e, b.e_ref, b.n_atoms, b.w_e, F, b.f_ref, b.w_f, mv, b.volume, b.core_s, b.s_ref,
                    b.w_s, b.counts, d_lw, Sg, Eb, Fb, Ms, d_losses, d_step, d_loss_acc, d_loss_done, N, G1,
                    do_scot ? d_fmax : nullptr, do_scot ? d_scot : nullptr,
                    ne, b.ei_ref, b.ngraph, b.spc, ei_steps_set ? d_wei : nullptr, ei_weight, Enb, sh.B, S, eif);
            } else {
                k_loss_par<T, false><<<nblk(3L * N), 256, 0, st>>>(
                    Eg, b.core_e, b.e_ref, b.n_atoms, b.w_e, F, b.f_ref, b.w_f, mv, b.volume, b.core_s, b.s_ref,
                    b.w_s, b.counts, d_lw, Sg, Eb, Fb, Ms, d_losses, d_step, d_loss_acc, d_loss_done, N, G1,
                    do_scot ? d_fmax : nullptr, do_scot ? d_scot : nullptr);
            }
            KCHECK();
        } else {
            k_loss<T><<<1, 512, 0, st>>>(Eg, b.core_e, b.e_ref, b.n_atoms, b.w_e, F, b.f_ref, b.w_f, mv, b.volume,
                                         b.core_s, b.s_ref, b.w_s, b.counts, d_lw, Sg, Eb, Fb, Ms, d_losses, d_step, N,
                                         G1);
            KCHECK();
            if (ei_on && frames.ei_ref) {
                const size_t smem_ei = (size_t)sh.B * S * (sizeof(double) + sizeof(int));
                k_node_energy_grad<T><<<1, 256, smem_ei, st>>>(
                    ne, b.ei_ref, b.ngraph, b.spc, b.w_f, b.counts,
                    ei_steps_set ? d_wei : nullptr, ei_weight, d_step,
                    Enb, d_losses, do_scot ? d_scot : nullptr, N, sh.B, S);
                KCHECK();
            }
        }
        // ---------------- second reverse: force-edge slots
        if (pb) {
            const int ym = y3p_mode();
            const int ec_grid = (int)((P + ecot_tpb - 1) / ecot_tpb);
            const float4* p_v4 = fast_g ? b.p_vec4 : nullptr;
            const int4* p_sg = fast_g ? b.p_srg : nullptr;
            if (ym > 0)
                k_edge_cot_pair<T><<<ec_grid, ecot_tpb, 0, st>>>(Fb, Ms, b.vec, UP, Y, b.pe0, b.snd, b.rcv, b.egraph, nullptr,
                                                                 nullptr, P, Y3P, ym, do_scot ? d_scot : nullptr, p_v4, p_sg);
            else
                k_edge_cot_pair<T><<<ec_grid, ecot_tpb, 0, st>>>(Fb, Ms, b.vec, UP, Y, b.pe0, b.snd, b.rcv, b.egraph, gyb,
                                                                 Yw, P, nullptr, 1, do_scot ? d_scot : nullptr, p_v4, p_sg);
        } else
            k_edge_cot<T><<<nblk(E), TPB, 0, st>>>(Fb, Ms, b.vec, U, JY, Y, b.snd, b.rcv, b.egraph, sb, gyb, Yw, E);
        KCHECK();
        T* Rb0 = gout[0];
        T* Tb0 = gout[0] + (long)P * O;
        T* Rb1 = gout[1];
        T* Tb1 = gout[1] + (long)P * O;
        const bool fxd_chk = slot_fuse_xd_active();
        const bool do_wgd = wg_defer && wg_side && m0_species && fxd_chk;
        if (mf)
            conv_sp<true, true, 2, false, false>(0, x0, Gn0, Yw, gyb, nullptr, nullptr, Gs0, Xd0);
        else
            conv<true, true, 2, false, false>(x0, Gn0, T0t, Yw, R0, gyb, nullptr, nullptr, Gs0, Xd0);
        // (j)* Gn0 = expand(B0b) M0^T
        if (m0_species) {
            launch_m0_fwd(Gs0, B0bs);
            KCHECK();
            if (!wg_sched && !do_wgd) {
                wg_begin();
                launch_m0_wgrad(Gs0, B0b);
                KCHECK();
                wg_end();
            }
        } else {
            per_l(Gs0, C, M0, (long)C * SC, SC, false, SC, tmp, T(0));
            k_pick<T><<<nblk(N16), TPB, 0, st>>>(tmp, b.spc, B0bs, N, C, S);
            k_expand<T><<<nblk(16L * N * SC), TPB, 0, st>>>(B0b, b.spc, tmp, N, C, S);
            KCHECK();
            wg_begin();
            per_l_wgrad(Gs0, C, tmp, SC, M0s, (long)C * SC, T(0));
            wg_end();
        }
        // (i)*
        const int prod_nsub = N <= 512 ? 4 : 2;
        if (prod_direct) {
            if (pdu)
                launch_prod_rev2_u(B0, W0, f0b, B0bs, B0s, f0bs, gW0);
            else if (nm && prod_spc)
                k_prod_rev2_dir_spc<T, true><<<dim3(prod_nsub, nops::spc_max_tiles32(N, S)), TPB, 0, st>>>(
                    B0, W0, b.spc_off, b.spc_idx, b.spc_tiles32, f0b, B0bs, B0s, f0bs, gW0, N, C, S);
            else if (nm)
                k_prod_rev2_dir<T, true><<<nblk(NC), TPB, 0, st>>>(B0, W0, b.spc, f0b, B0bs, B0s, f0bs, gW0, N, C);
            else
                k_prod_rev2_dir<T, false><<<nblk(NC), TPB, 0, st>>>(B0, W0, b.spc, f0b, B0bs, B0s, f0bs, gW0, N, C);
        } else
            k_prod_rev2<T><<<nblk(NC), TPB, 0, st>>>(B0, W0, b.spc, f0b, B0bs, B0s, f0bs, Wp0, N, C);
        KCHECK();
        // (h)*
        if (cz) {
            if (do_q_split)
                nops::k_q_fwd_c32<T><<<dim3(2 * (C / 32), nops::spc_max_tiles32(N, S)), 128, 0, st>>>(
                    f0bs, Q, b.spc_off, b.spc_idx, b.spc_tiles32, nullptr, x1bs, scbs, C, ZW, S);
            else
                nops::k_q_fwd<T><<<dim3(2 * (C / nops::CN), nops::spc_max_tiles32(N, S)), 128, 0, st>>>(
                    f0bs, Q, b.spc_off, b.spc_idx, b.spc_tiles32, nullptr, x1bs, scbs, C, ZW, S);
            KCHECK();
        } else {
            lgemm(false, false, N, ZW, C, T(1), f0bs, C, Q, ZW, T(0), zbs, ZW);
            wg_begin();
            lgemm(true, false, C, ZW, N, T(1), f0bs, C, zb, ZW, T(0), Qs, ZW);
            wg_end();
            k_split_z<T><<<nblk(NC), TPB, 0, st>>>(zbs, b.spc, x1bs, scbs, N, C, ZW);
            KCHECK();
        }
        if (wg_sched && !do_wgd && m0_species) {
            // Overlap M0 wgrad with 117 us conv_rev2(1) instead of starving k_prod_rev2(0) + k_q_fwd!
            wg_begin();
            launch_m0_wgrad(Gs0, B0b);
            KCHECK();
            wg_end();
        }
        // (g)*
        // layer 1's force-edge slots and (g)* G1 cotangent from x1b* in one pass
        if (mf)
            conv_sp<true, true, 2, true, false>(1, x1, Gn1, Yw, gyb, Y, x1bs, Gs1, Xd1);
        else
            conv<true, true, 2, true, false>(x1, Gn1, T1t, Yw, R1, gyb, Y, x1bs, Gs1, Xd1);
        // (f)*
        const bool fxd = fxd_chk;
        T* r_B1bs = (alias_16 && !in_debug) ? B0bs : B1bs;
        T* r_A1s = (alias_16 && fxd && !in_debug) ? B0bs : A1s;
        T* r_A0s = (alias_16 && !in_debug) ? B0bs : A0s;
        per_l(Gs1, C, wl1, (long)C * C, C, false, C, r_B1bs, T(0));
        if (!wg_sched && !do_wgd) {
            wg_begin();
            launch_wl1_wgrad(Gs1, B1b, T(0));
            wg_end();
        }
        // (e)*
        if (prod_direct) {
            if (pdu)
                launch_prod_rev2_u(B1, W1, f1b, r_B1bs, B1s, f1bs, gW1);
            else if (nm && prod_spc)
                k_prod_rev2_dir_spc<T, true><<<dim3(prod_nsub, nops::spc_max_tiles32(N, S)), TPB, 0, st>>>(
                    B1, W1, b.spc_off, b.spc_idx, b.spc_tiles32, f1b, r_B1bs, B1s, f1bs, gW1, N, C, S);
            else if (nm)
                k_prod_rev2_dir<T, true><<<nblk(NC), TPB, 0, st>>>(B1, W1, b.spc, f1b, r_B1bs, B1s, f1bs, gW1, N, C);
            else
                k_prod_rev2_dir<T, false><<<nblk(NC), TPB, 0, st>>>(B1, W1, b.spc, f1b, r_B1bs, B1s, f1bs, gW1, N, C);
        } else
            k_prod_rev2<T><<<nblk(NC), TPB, 0, st>>>(B1, W1, b.spc, f1b, r_B1bs, B1s, f1bs, Wp1, N, C);
        if (wg_sched && !do_wgd) {
            // Queue wl1_wgrad AFTER prod_rev2(1) so prod_rev2(1) enters SMs first without stalling!
            wg_begin();
            launch_wl1_wgrad(Gs1, B1b, T(0));
            wg_end();
        }
        if (cz) {
            if (ro_fuse_mode >= 2) {
                if constexpr (std::is_same<T, float>::value) {
                    const int sk_wp = nops_opt ? (N >= 640 ? 36 : 12) : std::min(32, std::max(4, (N + 63) / 64));
                    if (ro_na == 16)
                        nops::k_wp1_readout_rev_gemm<16><<<(N + 15) / 16, 256, 0, st>>>(
                            f1bs, wp1, scbs, f2bs, wro1, u, wro2, Eb, b.ngraph, tbs, us, qn, f2s, f1s, act, scale, N,
                            do_scot ? d_scot : nullptr, (ei_on && frames.ei_ref) ? Enb : nullptr);
                    else if (ro_na == 12)
                        nops::k_wp1_readout_rev_gemm<12><<<(N + 11) / 12, 192, 0, st>>>(
                            f1bs, wp1, scbs, f2bs, wro1, u, wro2, Eb, b.ngraph, tbs, us, qn, f2s, f1s, act, scale, N,
                            do_scot ? d_scot : nullptr, (ei_on && frames.ei_ref) ? Enb : nullptr);
                    else
                        nops::k_wp1_readout_rev_gemm<8><<<(N + 7) / 8, 128, 0, st>>>(
                            f1bs, wp1, scbs, f2bs, wro1, u, wro2, Eb, b.ngraph, tbs, us, qn, f2s, f1s, act, scale, N,
                            do_scot ? d_scot : nullptr, (ei_on && frames.ei_ref) ? Enb : nullptr);
                    KCHECK();
                    wg_begin();
                    if (nops_opt)
                        nops::k_wro2s_par<T><<<N >= 640 ? 8 : 4, 256, 0, st>>>(tbs, tt, qn, scale, wro2s, N, hid);
                    else
                        nops::k_wro2s<T><<<1, 256, 0, st>>>(tbs, tt, qn, scale, wro2s, N, hid);
                    if (nops_opt && C == 64 && hid == 16)
                        nops::k_wgrad_pair_64x16<T><<<sk_wp, 64, 0, st>>>(f2bs, ub, f2, us, wro1s, N);
                    else
                        nops::k_wgrad_pair<T><<<dim3(sk_wp, ((C + 63) / 64) * ((hid + 63) / 64)), 256, 0, st>>>(
                            f2bs, ub, f2, us, wro1s, N, C, hid);
                    nops::k_wgrad_pair<T><<<dim3(sk_wp, ((C + 63) / 64) * ((C + 63) / 64)), 256, 0, st>>>(
                        f1bs, f2b, f1, f2s, wp1s, N, C, C);
                    wg_end();
                    KCHECK();
                }
            } else if (ro_fuse_mode == 1) {
                if constexpr (std::is_same<T, float>::value) {
                    nops::k_node_gemm<T, false, true><<<dim3(C / nops::CN, (N + nops::RM - 1) / nops::RM), 128, 0, st>>>(
                        f1bs, wp1, scbs, f2bs, N, C, C, C);
                    const int sk_wp = nops_opt ? (N >= 640 ? 36 : 12) : std::min(32, std::max(4, (N + 63) / 64));
                    nops::k_readout_rev_gemm<<<(N + 15) / 16, 128, 0, st>>>(
                        f2bs, wro1, u, wro2, Eb, b.ngraph, tbs, us, qn, f2s, act, scale, N,
                        do_scot ? d_scot : nullptr, (ei_on && frames.ei_ref) ? Enb : nullptr);
                    KCHECK();
                    wg_begin();
                    if (nops_opt)
                        nops::k_wro2s_par<T><<<N >= 640 ? 8 : 4, 256, 0, st>>>(tbs, tt, qn, scale, wro2s, N, hid);
                    else
                        nops::k_wro2s<T><<<1, 256, 0, st>>>(tbs, tt, qn, scale, wro2s, N, hid);
                    if (nops_opt && C == 64 && hid == 16)
                        nops::k_wgrad_pair_64x16<T><<<sk_wp, 64, 0, st>>>(f2bs, ub, f2, us, wro1s, N);
                    else
                        nops::k_wgrad_pair<T><<<dim3(sk_wp, ((C + 63) / 64) * ((hid + 63) / 64)), 256, 0, st>>>(
                            f2bs, ub, f2, us, wro1s, N, C, hid);
                    wg_end();
                    nops::k_node_gemm<T, true, false><<<dim3(C / nops::CN, (N + nops::RM - 1) / nops::RM), 128, 0, st>>>(
                        f2s, wp1, nullptr, f1s, N, C, C, C);
                    wg_begin();
                    nops::k_wgrad_pair<T><<<dim3(sk_wp, ((C + 63) / 64) * ((C + 63) / 64)), 256, 0, st>>>(
                        f1bs, f2b, f1, f2s, wp1s, N, C, C);
                    wg_end();
                    KCHECK();
                }
            } else {
                nops::k_node_gemm<T, false, true><<<dim3(C / nops::CN, (N + nops::RM - 1) / nops::RM), 128, 0, st>>>(
                    f1bs, wp1, scbs, f2bs, N, C, C, C);
                nops::k_node_gemm<T, false, false><<<dim3((hid + nops::CN - 1) / nops::CN, (N + nops::RM - 1) / nops::RM),
                                                     128, 0, st>>>(f2bs, wro1, nullptr, ubs, N, C, hid, hid);
                KCHECK();
                // (b)*, (a)* and the readout's forward reverse
                k_readout_rev<T><<<nblk((long)N * hid), TPB, 0, st>>>(u, ubs, wro2, Eb, b.ngraph, tbs, us, qn, act, scale, N,
                                                                      hid, do_scot ? d_scot : nullptr,
                                                                      (ei_on && frames.ei_ref) ? Enb : nullptr);
                KCHECK();
                wg_begin();
                if (nops_opt)
                    nops::k_wro2s_par<T><<<N >= 640 ? 8 : 4, 256, 0, st>>>(tbs, tt, qn, scale, wro2s, N, hid);
                else
                    nops::k_wro2s<T><<<1, 256, 0, st>>>(tbs, tt, qn, scale, wro2s, N, hid);
                KCHECK();
                wg_end();
                // ---------------- forward reverse
                const int sk_wp = nops_opt ? (N >= 640 ? 36 : 12) : std::min(32, std::max(4, (N + 63) / 64));
                nops::k_node_gemm<T, true, false><<<dim3(C / nops::CN, (N + nops::RM - 1) / nops::RM), 128, 0, st>>>(
                    us, wro1, nullptr, f2s, N, hid, C, hid);
                wg_begin();
                if (nops_opt && C == 64 && hid == 16)
                    nops::k_wgrad_pair_64x16<T><<<sk_wp, 64, 0, st>>>(f2bs, ub, f2, us, wro1s, N);
                else
                    nops::k_wgrad_pair<T><<<dim3(sk_wp, ((C + 63) / 64) * ((hid + 63) / 64)), 256, 0, st>>>(
                        f2bs, ub, f2, us, wro1s, N, C, hid);
                wg_end();
                nops::k_node_gemm<T, true, false><<<dim3(C / nops::CN, (N + nops::RM - 1) / nops::RM), 128, 0, st>>>(
                    f2s, wp1, nullptr, f1s, N, C, C, C);
                wg_begin();
                nops::k_wgrad_pair<T><<<dim3(sk_wp, ((C + 63) / 64) * ((C + 63) / 64)), 256, 0, st>>>(
                    f1bs, f2b, f1, f2s, wp1s, N, C, C);
                wg_end();
                KCHECK();
            }
        } else {
            k_copy<T><<<nblk(NC), TPB, 0, st>>>(f2bs, scbs, NC);
            KCHECK();
            // (d)*
            lgemm(false, false, N, C, C, T(1), f1bs, C, wp1, C, T(1), f2bs, C);
            wg_begin();
            lgemm(true, false, C, C, N, T(1), f1bs, C, f2b, C, T(0), wp1s, C);
            wg_end();
            // (c)*
            lgemm(false, false, N, hid, C, T(1), f2bs, C, wro1, hid, T(0), ubs, hid);
            wg_begin();
            lgemm(true, false, C, hid, N, T(1), f2bs, C, ub, hid, T(0), wro1s, hid);
            wg_end();
            // (b)*, (a)* and the readout's forward reverse
            k_readout_rev<T><<<nblk((long)N * hid), TPB, 0, st>>>(u, ubs, wro2, Eb, b.ngraph, tbs, us, qn, act, scale, N,
                                                                  hid, do_scot ? d_scot : nullptr,
                                                                  (ei_on && frames.ei_ref) ? Enb : nullptr);
            KCHECK();
            wg_begin();
            lgemm(true, false, hid, 1, N, scale, tbs, hid, ones, 1, T(0), wro2s, 1);
            lgemm(true, false, hid, 1, N, T(1), tt, hid, qn, 1, T(1), wro2s, 1);
            wg_end();
            // ---------------- forward reverse
            lgemm(false, true, N, C, hid, T(1), us, hid, wro1, hid, T(0), f2s, C);
            wg_begin();
            lgemm(true, false, C, hid, N, T(1), f2, C, us, hid, T(1), wro1s, hid);
            wg_end();
            lgemm(false, true, N, C, C, T(1), f2s, C, wp1, C, T(0), f1s, C);
            wg_begin();
            lgemm(true, false, C, C, N, T(1), f1, C, f2s, C, T(1), wp1s, C);
            wg_end();
        }
        if (prod_direct) {
            if (pdu)
                launch_prod_rev1_u(B1, W1, f1s, B1s, gW1);
            else if (nm && prod_spc)
                k_prod_rev1_dir_spc<T, true><<<dim3(prod_nsub, nops::spc_max_tiles32(N, S)), TPB, 0, st>>>(
                    B1, W1, b.spc_off, b.spc_idx, b.spc_tiles32, f1s, B1s, gW1, N, C, S);
            else if (nm)
                k_prod_rev1_dir<T, true><<<nblk(NC), TPB, 0, st>>>(B1, W1, b.spc, f1s, B1s, gW1, N, C);
            else
                k_prod_rev1_dir<T, false><<<nblk(NC), TPB, 0, st>>>(B1, W1, b.spc, f1s, B1s, gW1, N, C);
            KCHECK();
        } else {
            k_prod_rev1<T><<<nblk(NC), TPB, 0, st>>>(B1, W1, b.spc, f1s, B1s, Wp1, N, C);
            KCHECK();
            wg_begin();
            lgemm(true, false, S, K * C, N, T(1), b.onehot, S, Wp1, K * C, T(0), gW1, K * C);
            wg_end();
        }
        const int zsm1 = slot_zip_mode(1);
        per_l(B1s, C, wl1, (long)C * C, C, true, C, r_A1s, T(0), zsm1 ? Gn1 : nullptr, zsm1 ? GA_buf : nullptr, zsm1);
        const bool do_vgs = vg_split && fxd && d_mg_vg_early && d_mg_vg_t0 && wg_side;
        const bool do_mid = std::is_same<T, float>::value && (mid_defer > 0) && do_wgd && do_vgs && par() &&
                            t0s_dir_ok() && ms && mpb && fused_mlp && mlp_spline_small && mlp_sp8 && mlp_tc < 0 &&
                            !mtiny && !mlp_sp4 && (mlp_spv2 >= 1 && spline_M <= 66) &&
                            H == mspl::SP_H && O == mspl::SP_O && cz && qwgrad_dyn && wbal && wgrad_opt && !in_debug;
        const bool do_mid2 = do_mid && (mid_defer >= 2);
        cudaEvent_t e_wl1_done = nullptr;
        // layer 1's radial cotangents are final here: their slots and MLP 1's backward on the side stream
        if (fxd) {
            if (do_wgd) {
                // Wave 21/34 critical-first + deferred wgrads: launch slot_xd(1) FIRST so it enters SMs with 0 stall,
                // pack M0_wgrad(Gs0,B0b) + wl1_wgrad(Gs1,B1b) into its shadow on s3; under do_mid defer wl1_wgrad#2,
                // q_wgrad and mlp_bwd(1) out of the k_q_bwd_c32/prod_rev1/m0_bwd critical window!
                cudaEvent_t e_s1 = wg_record();
                slot_xd_sp<true>(1, Gn1, r_A1s, x1, x1bs, Yw, gyb, Xd1);
                wg_begin_ev(e_s1);
                launch_m0_wgrad(Gs0, B0b);
                launch_wl1_wgrad(Gs1, B1b, T(0));
                if (!do_mid) {
                    launch_wl1_wgrad(A1, B1s, T(1));
                } else {
                    e_wl1_done = evw[nevw++ % 32];
                    CK(cudaEventRecord(e_wl1_done, st));
                }
                wg_end();
            } else {
                wg_begin();
                launch_wl1_wgrad(A1, B1s, T(1));
                wg_end();
                slot_xd_sp<true>(1, Gn1, r_A1s, x1, x1bs, Yw, gyb, Xd1);
            }
            if (par() && !do_mid) {
                fork(2);
                OnSide side(this);
                mlp_bwd(1);
            }
        } else {
            if (par()) {
                fork(2);
                OnSide side(this);
                if (mf)
                    slot_sp<true>(1, Gn1, r_A1s, x1, x1bs, Yw, gyb);
                else
                    slot<true>(Gn1, r_A1s, x1, x1bs, Yw, gyb, Tb1, Rb1);
                mlp_bwd(1);
            }
            wg_begin();
            launch_wl1_wgrad(A1, B1s, T(1));
            wg_end();
            if (mf)
                conv_sp<false, true, 1, false, true>(1, nullptr, r_A1s, Y, nullptr, nullptr, nullptr, nullptr, Xd1);
            else
                conv<false, true, 1, false, true>(nullptr, r_A1s, R1, Y, nullptr, nullptr, nullptr, nullptr, nullptr, Xd1);  // x1s
        }
        if (cz) {
            const int sk_qw = std::min(16, std::max(4, (N + 127) / 128));
            if (do_q_split) {
                nops::k_q_bwd_r16<T><<<dim3(C / 32, 2 * nops::spc_max_tiles32(N, S)), 128, 0, st>>>(
                    qn, T(0), Xd1, f2s, Q, b.spc_off, b.spc_idx, b.spc_tiles32, f0s, C, ZW, S);
            } else if (nops_opt) {
                nops::k_q_bwd_c32<T><<<dim3(C / 32, nops::spc_max_tiles32(N, S)), 128, 0, st>>>(
                    qn, T(0), Xd1, f2s, Q, b.spc_off, b.spc_idx, b.spc_tiles32, f0s, C, ZW, S);
            } else {
                nops::k_q_bwd<T><<<dim3(C / nops::CN, nops::spc_max_tiles32(N, S)), 128, 0, st>>>(
                    qn, T(0), Xd1, f2s, Q, b.spc_off, b.spc_idx, b.spc_tiles32, f0s, C, ZW, S);
            }
            if (!do_mid) {
                wg_begin();
                if (qwgrad_dyn && wbal && wgrad_opt) {
                    const int sk_x = std::min(24, std::max(8, (N + 63) / 64));
                    nops::k_q_wgrad_dyn<T><<<dim3(sk_x + wtarget + S, (C / 64) * (C / 64)), 256, 0, st>>>(
                        f0bs, scale, x1b, f2b, f0, qn, Xd1, f2s, b.spc_off, b.spc_idx, b.spc_wsk, Qs, N, C, ZW, S, sk_x);
                } else {
                    nops::k_q_wgrad<T><<<dim3(sk_qw, (1 + S) * (C / 64) * (C / 64)), 256, 0, st>>>(
                        f0bs, scale, x1b, f2b, f0, qn, Xd1, f2s, b.spc_off, b.spc_idx, Qs, N, C, ZW, S);
                }
                wg_end();
            } else if (do_mid2) {
                // Wave 35 (MID=2): launch k_q_wgrad_dyn (only 13-17 CTAs!) on s2 RIGHT AFTER k_q_bwd_c32 so it runs
                // in free SMs alongside prod_rev1 + m0_bwd, finishing BEFORE slot_xd(0) and killing the +11.7 us tail spill!
                fork(2);
                {
                    OnSide side(this);
                    const int sk_x = std::min(24, std::max(8, (N + 63) / 64));
                    nops::k_q_wgrad_dyn<T><<<dim3(sk_x + wtarget + S, (C / 64) * (C / 64)), 256, 0, st>>>(
                        f0bs, scale, x1b, f2b, f0, qn, Xd1, f2s, b.spc_off, b.spc_idx, b.spc_wsk, Qs, N, C, ZW, S, sk_x);
                    if (vg_qsplit && !vg_merge && d_mg_vg_q && d_mg_vg_m0) {
                        launch_mgemm(d_mg_vg_q, mg_vg_q_tiles);
                    }
                    CK(cudaEventRecord(evf[3], st));
                }
            }
            KCHECK();
        } else {
            k_build_z<T><<<nblk((long)N * ZW), TPB, 0, st>>>(zs, qn, T(0), Xd1, f2s, b.spc, N, C, ZW);
            KCHECK();
            lgemm(false, true, N, C, ZW, T(1), zs, ZW, Q, ZW, T(0), f0s, C);
            wg_begin();
            lgemm(true, false, C, ZW, N, T(1), f0, C, zs, ZW, T(1), Qs, ZW);
            wg_end();
        }
        if (prod_direct) {
            if (pdu)
                launch_prod_rev1_u(B0, W0, f0s, B0s, gW0);
            else if (nm && prod_spc)
                k_prod_rev1_dir_spc<T, true><<<dim3(prod_nsub, nops::spc_max_tiles32(N, S)), TPB, 0, st>>>(
                    B0, W0, b.spc_off, b.spc_idx, b.spc_tiles32, f0s, B0s, gW0, N, C, S);
            else if (nm)
                k_prod_rev1_dir<T, true><<<nblk(NC), TPB, 0, st>>>(B0, W0, b.spc, f0s, B0s, gW0, N, C);
            else
                k_prod_rev1_dir<T, false><<<nblk(NC), TPB, 0, st>>>(B0, W0, b.spc, f0s, B0s, gW0, N, C);
            KCHECK();
        } else {
            k_prod_rev1<T><<<nblk(NC), TPB, 0, st>>>(B0, W0, b.spc, f0s, B0s, Wp0, N, C);
            KCHECK();
            wg_begin();
            lgemm(true, false, S, K * C, N, T(1), b.onehot, S, Wp0, K * C, T(0), gW0, K * C);
            wg_end();
        }
        cudaEvent_t e_s0_wg = nullptr;
        const int zsm0 = slot_zip_mode(0);
        if (m0_species) {
            launch_m0_bwd(B0s, r_A0s, zsm0, zsm0 ? Gn0 : nullptr, zsm0 ? GA_buf : nullptr);
            KCHECK();
            if (do_wgd) {
                e_s0_wg = wg_record();
            } else {
                wg_begin();
                launch_m0_wgrad(A0, B0s);
                if (do_vgs) launch_mgemm(d_mg_vg_early, mg_vg_early_tiles);
                KCHECK();
                wg_end();
            }
        } else {
            wg_join();   // tmp: the first M0s product has read it
            k_expand<T><<<nblk(16L * N * SC), TPB, 0, st>>>(B0s, b.spc, tmp, N, C, S);
            KCHECK();
            per_l(tmp, SC, M0, (long)C * SC, SC, true, C, r_A0s, T(0));
            wg_begin();
            per_l_wgrad(A0, C, tmp, SC, M0s, (long)C * SC, T(1));
            if (do_vgs) launch_mgemm(d_mg_vg_early, mg_vg_early_tiles);
            wg_end();
        }
        auto launch_view_grads = [&]() {
            wg_begin();
            if (!t0s_dir_ok()) {
                if (cz) {
                    nops::k_t0s_spc<T><<<dim3(8, S), C, 0, st>>>(Xd0, b.spc_off, b.spc_idx, T0s, C);
                    KCHECK();
                } else {
                    lgemm(true, false, S, C, N, T(1), b.onehot, S, Xd0, C, T(0), T0s, C);
                }
            }
            // ---------------- view gradients
            if (d_mg_vg) {
                mgemm::Table<T>* tbl = do_vgs ? d_mg_vg_t0 : d_mg_vg;
                const int tiles = do_vgs ? mg_vg_t0_tiles : mg_vg_tiles;
                launch_mgemm(tbl, tiles);
                KCHECK();
            } else {
                lgemm(false, true, S, C, C, T(1), T0s, C, wup0, C, T(0), gemb, C);
                lgemm(true, false, C, C, S, T(1), emb, C, T0s, C, T(0), gwup0, C);
                for (int l = 0; l < 4; ++l) {
                    lgemm(false, true, C, C, SC, T(1), M0s + (long)l * C * SC, SC, ws0 + (long)l * C * SC, SC, T(0),
                          gwl0 + (long)l * C * C, C);
                    lgemm(true, false, C, SC, C, T(1), wl0 + (long)l * C * C, C, M0s + (long)l * C * SC, SC, T(0),
                          gws0 + (long)l * C * SC, SC);
                }
                lgemm(false, true, C, C, ZW, T(1), Qs, ZW, Wcat, ZW, T(0), gwp0, C);
                lgemm(true, false, C, ZW, C, T(1), wp0, C, Qs, ZW, T(0), gWcat, ZW);
            }
            wg_end();
        };
        const bool tail_s2 = fxd && do_wgd && do_vgs && par() && t0s_dir_ok() && d_mg_vg_t0 && !in_debug;
        if (do_mid) {
            if constexpr (std::is_same<T, float>::value) {
                // Wave 34+37+38 zero-contention slot_xd(0) + zero-recompute dual MLP backward + critical-first tail:
                //   tail_prio=1:
                //     st: slot_xd(0) [rec e_s0_done] -> crt_dual [fork(1)] -> k_mlp_bwd_sp4_v2_dual_noz (18 CTAs FIRST!)
                //     s2: (after e_s0_wg) [q_wgrad] -> wl1_wgrad#2 -> (wait evf[1]) k_mlp_dw4_dual (64 CTAs AFTER bwd_noz!)
                //     s3: (after e_s0_wg) m0_wgrad#2 -> (wait evf[3]) vg_early (ZERO wait on slot_xd(0)!) -> (wait e_s0_done) vg_t0
                slot_xd_sp<false>(0, Gn0, r_A0s, x0, nullptr, Yw, gyb, Xd0);
                const bool do_tp = tail_prio && !vg_merge;
                const bool do_vgm = !do_tp && vg_merge && d_mg_vg_all != nullptr;
                cudaEvent_t e_s0_done = (do_tp || do_vgm) ? wg_record() : nullptr;
                CK(cudaStreamWaitEvent(s2, e_s0_wg, 0));
                {
                    OnSide side(this);
                    if (!do_mid2) {
                        const int sk_x = std::min(24, std::max(8, (N + 63) / 64));
                        nops::k_q_wgrad_dyn<T><<<dim3(sk_x + wtarget + S, (C / 64) * (C / 64)), 256, 0, st>>>(
                            f0bs, scale, x1b, f2b, f0, qn, Xd1, f2s, b.spc_off, b.spc_idx, b.spc_wsk, Qs, N, C, ZW, S, sk_x);
                        CK(cudaEventRecord(evf[3], st));
                    }
                    if (e_wl1_done) CK(cudaStreamWaitEvent(st, e_wl1_done, 0));
                    launch_wl1_wgrad(A1, B1s, T(1));
                }
                const bool do_vg_qs = vg_qsplit && do_mid2 && !vg_merge && d_mg_vg_q && d_mg_vg_m0;
                wg_begin_ev(e_s0_wg);
                launch_m0_wgrad(A0, B0s);
                if (!do_vg_qs) CK(cudaStreamWaitEvent(st, evf[3], 0));
                if (do_tp) {
                    launch_mgemm(do_vg_qs ? d_mg_vg_m0 : d_mg_vg_early, do_vg_qs ? mg_vg_m0_tiles : mg_vg_early_tiles);
                    if (e_s0_done) CK(cudaStreamWaitEvent(st, e_s0_done, 0));
                    launch_mgemm(d_mg_vg_t0, mg_vg_t0_tiles);
                } else if (do_vgm) {
                    if (e_s0_done) CK(cudaStreamWaitEvent(st, e_s0_done, 0));
                    launch_mgemm(d_mg_vg_all, mg_vg_all_tiles);
                } else {
                    launch_mgemm(do_vg_qs ? d_mg_vg_m0 : d_mg_vg_early, do_vg_qs ? mg_vg_m0_tiles : mg_vg_early_tiles);
                }
                KCHECK();
                wg_end();
                if (!do_tp && !do_vgm) {
                    fork(0);
                    OnSide side(this);
                    launch_mgemm(d_mg_vg_t0, mg_vg_t0_tiles);
                    KCHECK();
                }
                mspl::k_spline_c_to_rt_bwd_dual<float><<<dim3(spline_M, 2), O, 0, st>>>(
                    gC_k[0], gout_k[0], gout_k[0] + (long)spline_M * O, shell_rcut4(0),
                    gC_k[1], gout_k[1], gout_k[1] + (long)spline_M * O, shell_rcut4(1),
                    float(sp_h()), mlp_spline, C, float(sp_r_lo), float(shell_width));
                fork(1);
                const int ntiles = (spline_M + mspl::TP_V2 - 1) / mspl::TP_V2;
                const bool use_noz = mlp_savez && z1_k[0] && z2_k[0] && z3_k[0] && z1_k[1] && z2_k[1] && z3_k[1] &&
                                     ((mlp_spv2 == 1 && spline_M <= 34) || (mlp_spv2 == 2 && spline_M <= 260));
                const bool use_noz_t2 = use_noz && (mlp_t2 & 1) && spline_M <= 34;
                auto launch_bwd_dual = [&]() {
                    if (use_noz_t2) {
                        const int nt2 = (spline_M + mspl::TP_T2 - 1) / mspl::TP_T2;
                        mspl::k_mlp_bwd_sp4_v2_dual_noz_t2<<<dim3(nt2, 2), mspl::NTH_V2, mspl::SMEM_BWD_SPT2, st>>>(
                            EF_k, DEF_k,
                            mlpW[0][1], mlpW[0][2], mlpW[0][3], z1_k[0], z2_k[0], z3_k[0], gout_k[0], gmlp[0][0], gmlp[0][1], gmlp[0][2],
                            mlpW[1][1], mlpW[1][2], mlpW[1][3], z1_k[1], z2_k[1], z3_k[1], gout_k[1], gmlp[1][0], gmlp[1][1], gmlp[1][2],
                            float(md.act), spline_M);
                    } else if (use_noz) {
                        mspl::k_mlp_bwd_sp4_v2_dual_noz<<<dim3(ntiles, 2), mspl::NTH_V2, mspl::SMEM_BWD_SPV2, st>>>(
                            EF_k, DEF_k,
                            mlpW[0][1], mlpW[0][2], mlpW[0][3], z1_k[0], z2_k[0], z3_k[0], gout_k[0], gmlp[0][0], gmlp[0][1], gmlp[0][2],
                            mlpW[1][1], mlpW[1][2], mlpW[1][3], z1_k[1], z2_k[1], z3_k[1], gout_k[1], gmlp[1][0], gmlp[1][1], gmlp[1][2],
                            float(md.act), spline_M);
                    } else {
                        mspl::k_mlp_bwd_sp4_v2_dual<<<dim3(ntiles, 2), mspl::NTH_V2, mspl::SMEM_BWD_SPV2, st>>>(
                            EF_k, DEF_k,
                            mlpW[0][0], mlpW[0][1], mlpW[0][2], mlpW[0][3], gout_k[0], gmlp[0][0], gmlp[0][1], gmlp[0][2],
                            mlpW[1][0], mlpW[1][1], mlpW[1][2], mlpW[1][3], gout_k[1], gmlp[1][0], gmlp[1][1], gmlp[1][2],
                            float(md.act), spline_M);
                    }
                };
                if (do_tp) {
                    launch_bwd_dual();
                    {
                        OnSide side(this);
                        mspl::k_mlp_dw4_dual<<<dim3(8, 4, 2), 128, 0, st>>>(
                            h3_k[0], gout_k[0], gmlp[0][3], h3_k[1], gout_k[1], gmlp[1][3], 2 * spline_M);
                        KCHECK();
                    }
                } else {
                    {
                        OnSide side(this);
                        mspl::k_mlp_dw4_dual<<<dim3(8, 4, 2), 128, 0, st>>>(
                            h3_k[0], gout_k[0], gmlp[0][3], h3_k[1], gout_k[1], gmlp[1][3], 2 * spline_M);
                        KCHECK();
                    }
                    launch_bwd_dual();
                }
                KCHECK();
            }
            wg_join();
            join(0);
        } else {
            if (fxd) {
                slot_xd_sp<false>(0, Gn0, r_A0s, x0, nullptr, Yw, gyb, Xd0);
                if (do_wgd && m0_species) {
                    wg_begin_ev(e_s0_wg);
                    launch_m0_wgrad(A0, B0s);
                    if (do_vgs) launch_mgemm(d_mg_vg_early, mg_vg_early_tiles);
                    KCHECK();
                    wg_end();
                }
                if (tail_s2) {
                    join(3);  // mlp_bwd(1) on s2 finished long ago; reclaim s2 for vg_t0 + k_mlp_dw4(0)!
                    fork(0);
                    {
                        OnSide side(this);
                        launch_mgemm(d_mg_vg_t0, mg_vg_t0_tiles);
                        KCHECK();
                    }
                } else {
                    launch_view_grads();
                }
            } else {
                if (mf)
                    conv_sp<false, true, 1, false, true>(0, nullptr, r_A0s, Y, nullptr, nullptr, nullptr, nullptr, Xd0);
                else
                    conv<false, true, 1, false, true>(nullptr, r_A0s, R0, Y, nullptr, nullptr, nullptr, nullptr, nullptr, Xd0);  // x0s
                if (vg_early) launch_view_grads();
                // the radial weights' cotangents, each written once
                if (mf) {
                    slot_sp<false>(0, Gn0, r_A0s, x0, nullptr, Yw, gyb);
                    if (!par()) slot_sp<true>(1, Gn1, r_A1s, x1, x1bs, Yw, gyb);
                } else {
                    slot<false>(Gn0, r_A0s, x0, nullptr, Yw, gyb, Tb0, Rb0);
                    if (!par()) slot<true>(Gn1, r_A1s, x1, x1bs, Yw, gyb, Tb1, Rb1);
                }
                if (!vg_early) launch_view_grads();
            }
            if (!par() && mlp_dual_ok()) {
                mlp_bwd_dual();
            } else {
                mlp_bwd(0, tail_s2);
                if (!par()) mlp_bwd(1);
            }
            wg_join();
            join(tail_s2 ? 0 : 3);
        }
        par_mask = 3;
        // ---------------- raw gradients
        const T rC = T(1) / std::sqrt(T(C));
        const bool do_gfuse = grad_fuse && pdu;
        const bool uc_opt = (update && have_optim) ? (opt.update_clip > 0) : false;
        const bool do_gf_prep = do_gfuse && update && have_optim && gf_prep && !soap_cond_en;
        if (do_gfuse) {
            if (update && have_optim && !zero_fused) CK(cudaMemsetAsync(d_norm2, 0, sizeof(double), st));
            if (gf_flat && d_gftiles && n_gftiles > 0) {
                const int gf_grid = (gf_persist > 0) ? std::min(n_gftiles, std::max(1, nsm * gf_persist)) : n_gftiles;
                auto run_gf_flat = [&](auto f32_tag) {
                    constexpr bool F32 = decltype(f32_tag)::value;
                    if (do_gf_prep) {
                        k_grad_finalize_sumsq_flat<T, F32><<<gf_grid, TPB, 0, st>>>(
                            d_gftiles, n_gftiles, d_gseg, gW0, graw + poff[15], graw + poff[16], graw + poff[17], gW1,
                            graw + poff[19], graw + poff[20], graw + poff[21], K, md.K3, md.K2, C, gWcat, graw + poff[23],
                            graw + poff[8], graw + poff[14], rC, rC, T(1) / std::sqrt(T(C) * T(S)), ZW, d_norm2,
                            do_scot ? d_scot : nullptr, d_gf_done, d_pdesc, LST_N_PARAMS, d_lr, opt.max_grad_norm,
                            opt.beta1, opt.beta2, have_ema ? d_ema : nullptr, d_step, d_osc,
                            uc_opt ? d_urms : nullptr, adam_cautious ? d_caut : nullptr);
                    } else {
                        k_grad_finalize_sumsq_flat<T, F32><<<gf_grid, TPB, 0, st>>>(
                            d_gftiles, n_gftiles, d_gseg, gW0, graw + poff[15], graw + poff[16], graw + poff[17], gW1,
                            graw + poff[19], graw + poff[20], graw + poff[21], K, md.K3, md.K2, C, gWcat, graw + poff[23],
                            graw + poff[8], graw + poff[14], rC, rC, T(1) / std::sqrt(T(C) * T(S)), ZW,
                            (update && have_optim) ? d_norm2 : nullptr, do_scot ? d_scot : nullptr);
                    }
                };
                if constexpr (std::is_same<T, float>::value) {
                    if (gf_f32) run_gf_flat(std::true_type{});
                    else run_gf_flat(std::false_type{});
                } else {
                    run_gf_flat(std::false_type{});
                }
            } else {
                const int gf_gx = gf_prep ? 4 : 16;
                if (do_gf_prep) {
                    k_grad_finalize_sumsq<T><<<dim3(gf_gx, n_gseg + 3), TPB, 0, st>>>(
                        d_gseg, n_gseg, gW0, graw + poff[15], graw + poff[16], graw + poff[17], gW1, graw + poff[19],
                        graw + poff[20], graw + poff[21], S, K, md.K3, md.K2, C, gWcat, graw + poff[23], graw + poff[8],
                        graw + poff[14], rC, rC, T(1) / std::sqrt(T(C) * T(S)), ZW, d_norm2,
                        do_scot ? d_scot : nullptr, d_gf_done, d_pdesc, LST_N_PARAMS, d_lr, opt.max_grad_norm,
                        opt.beta1, opt.beta2, have_ema ? d_ema : nullptr, d_step, d_osc,
                        uc_opt ? d_urms : nullptr, adam_cautious ? d_caut : nullptr);
                } else {
                    k_grad_finalize_sumsq<T><<<dim3(gf_gx, n_gseg + 3), TPB, 0, st>>>(
                        d_gseg, n_gseg, gW0, graw + poff[15], graw + poff[16], graw + poff[17], gW1, graw + poff[19],
                        graw + poff[20], graw + poff[21], S, K, md.K3, md.K2, C, gWcat, graw + poff[23], graw + poff[8],
                        graw + poff[14], rC, rC, T(1) / std::sqrt(T(C) * T(S)), ZW, (update && have_optim) ? d_norm2 : nullptr,
                        do_scot ? d_scot : nullptr);
                }
            }
        } else {
            k_scale_multi<T><<<dim3(16, n_gseg), TPB, 0, st>>>(d_gseg);
            if (pdu) {
                k_uncat_gw_u_pair<T><<<dim3(nblk((long)S * K * C), 2), TPB, 0, st>>>(
                    gW0, graw + poff[15], graw + poff[16], graw + poff[17], gW1, graw + poff[19], graw + poff[20],
                    graw + poff[21], S, K, md.K3, md.K2, C);
            } else {
                k_cat_w<T><<<nblk((long)S * K * C), TPB, 0, st>>>(gW0, graw + poff[15], graw + poff[16], graw + poff[17], S,
                                                                 K, md.K3, md.K2, C, 1);
                k_cat_w<T><<<nblk((long)S * K * C), TPB, 0, st>>>(gW1, graw + poff[19], graw + poff[20], graw + poff[21], S,
                                                                 K, md.K3, md.K2, C, 1);
            }
            k_cat_wcat<T><<<nblk((long)C * ZW), TPB, 0, st>>>(gWcat, graw + poff[23], graw + poff[8], graw + poff[14], rC,
                                                             rC, T(1) / std::sqrt(T(C) * T(S)), C, ZW, 1);
        }
        KCHECK();
        // ---------------- optimizer
        if (update && have_optim) {
            if (!do_gfuse) {
                if (!zero_fused) CK(cudaMemsetAsync(d_norm2, 0, sizeof(double), st));
                k_sumsq<T><<<128, TPB, 0, st>>>(graw, ptotal, d_norm2);
            }
            const bool uc = uc_opt;
            const bool soap_fuse_g = soap_coal && n_mmats && soap_on && soap_smem_on && soap_smem_ok && soap_tc &&
                                     soap_ema_fuse && muon_wd == 0.0 && std::is_same<T, float>::value;
            const bool soap_par_adam = soap_fuse_g && par() && !uc;
#if CUDART_VERSION >= 12040
            cudaGraphConditionalHandle soap_cond_h = 0;
            const bool soap_cond_ok = soap_cond_en && n_mmats && soap_on && soap_eig && soap_smem_on && soap_smem_ok &&
                                      soap_jfuse && soap_tc && (soap_qr > 0) && std::is_same<T, float>::value && !in_debug;
            if (!do_gf_prep) {
                if (soap_cond_ok) {
                    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
                    cudaGraph_t cap_g = nullptr;
                    if (cudaStreamGetCaptureInfo(st, &cap_st, nullptr, &cap_g, nullptr, nullptr, nullptr) == cudaSuccess &&
                        cap_st == cudaStreamCaptureStatusActive && cap_g != nullptr) {
                        CK(cudaGraphConditionalHandleCreate(&soap_cond_h, cap_g, 0, 0));
                    }
                }
                k_adam_prep<T><<<1, 32, 0, st>>>(d_pdesc, LST_N_PARAMS, d_norm2, d_lr, opt.max_grad_norm, opt.beta1,
                                                 opt.beta2, have_ema ? d_ema : nullptr, d_step, d_osc,
                                                 uc ? d_urms : nullptr, adam_cautious ? d_caut : nullptr,
                                                 soap_cond_h, soap_every);
            }
#else
            if (!do_gf_prep) {
                k_adam_prep<T><<<1, 32, 0, st>>>(d_pdesc, LST_N_PARAMS, d_norm2, d_lr, opt.max_grad_norm, opt.beta1,
                                                 opt.beta2, have_ema ? d_ema : nullptr, d_step, d_osc,
                                                 uc ? d_urms : nullptr, adam_cautious ? d_caut : nullptr);
            }
#endif
            const int* d_step_opt = d_step + 1;  // Wave 34: saved s_cur from k_adam_prep (d_step[0] already incremented!)
            if (soap_par_adam) {
                // Wave 33: fork BEFORE k_caut_count; launch SOAP on st FIRST, hide caut+adam on s2!
                fork(0);
                if (use_geo_out && !(graph_pipe >= 3 && soap_mode == 1))
                    CK(cudaEventRecord(ev_geo_go, st));
            } else {
                if (adam_cautious)
                    k_caut_count<T><<<dim3(16, LST_N_PARAMS), TPB, 0, st>>>(d_pdesc, LST_N_PARAMS, d_osc, d_wd,
                                                                           T(opt.beta1), d_caut);
                if (uc)
                    k_urms<T><<<dim3(16, LST_N_PARAMS), TPB, 0, st>>>(d_pdesc, LST_N_PARAMS, d_osc, d_wd, T(opt.beta2),
                                                                     T(opt.eps), d_urms);
                k_adam<T><<<dim3(16, LST_N_PARAMS), TPB, 0, st>>>(d_pdesc, LST_N_PARAMS, d_osc, d_wd, T(opt.beta1),
                                                                  T(opt.beta2), T(opt.eps), uc ? d_urms : nullptr,
                                                                  opt.update_clip, T(soap_on ? soap_b1 : muon_mu),
                                                                  muon_nesterov, soap_on ? 1 : 0,
                                                                  adam_cautious ? d_caut : nullptr, d_lr, d_step_opt,
                                                                  muon_wd, soap_fuse_g ? 1 : 0);
            }
            KCHECK();
            if (n_mmats && soap_on) {
                const bool use_sm = soap_smem_on && soap_smem_ok;
#ifndef LST_LEAN
                const int use_tf = (std::is_same<T, float>::value && soap_tf32) ? 1 : 0;
#endif
                const T* ema_osc = (use_sm && have_ema && soap_ema_fuse) ? d_osc : nullptr;
                __nv_bfloat16* sqbf_ptr = (soap_qbf && soap_eig && soap_jfuse && (soap_qr > 0)) ? d_sqbf : nullptr;
                const int use_mrot = (soap_mrot > 0 && soap_coal && soap_fuse_g && soap_tc && soap_every > 1) ? soap_mrot : 0;
                const int stat_ev = (soap_stat_every > 1) ? soap_stat_every : 1;
                const float fb2_st = (stat_ev > 1) ? (float)std::pow(soap_b2, (double)stat_ev) : (float)soap_b2;
                if (use_sm) {
                    if constexpr (std::is_same<T, float>::value) {
                        if (soap_tc && soap_fast && soap_mode == 2 && use_mrot >= 2 && soap_coal && soap_fuse_g &&
                            sqbf_ptr != nullptr)
                            k_soap_a_fast_bf16<<<n_sunits, 256, 0, st>>>(d_smats, d_sunits, d_pdesc, d_lr, d_step_opt,
                                                                         d_sws, soap_mult, soap_b1, soap_b2, soap_eps,
                                                                         ema_osc, d_osc, sqbf_ptr, stat_ev, fb2_st);
                        else if (soap_tc)
                            k_soap_a_smem_bf16<<<n_sunits, 256, 0, st>>>(d_smats, d_sunits, d_pdesc, d_lr, d_step_opt, d_sws,
                                                                         soap_mult, soap_b1, soap_b2, soap_eps,
                                                                         soap_every, ema_osc, soap_coal ? 1 : 0,
                                                                         soap_fuse_g ? 1 : 0, d_osc, sqbf_ptr, use_mrot,
                                                                         stat_ev, fb2_st);
#ifndef LST_LEAN
                        else if (use_tf)
                            k_soap_a_smem_tf32<<<n_sunits, 256, soap_smem_a, st>>>(d_smats, d_sunits, d_pdesc, d_lr,
                                                                                   d_step_opt, d_sws, soap_mult, soap_b1,
                                                                                   soap_b2, soap_eps, soap_every,
                                                                                   ema_osc);
                        else if (soap_opt > 0)
                            k_soap_a_smem_opt<<<n_sunits, 256, soap_smem_a, st>>>(d_smats, d_sunits, d_pdesc, d_lr,
                                                                                  d_step_opt, d_sws, soap_mult, soap_b1,
                                                                                  soap_b2, soap_eps, soap_every,
                                                                                  ema_osc);
                        else
                            k_soap_a_smem_f<<<n_sunits, 256, soap_smem_a, st>>>(d_smats, d_sunits, d_pdesc, d_lr,
                                                                                d_step_opt, d_sws, soap_mult, soap_b1,
                                                                                soap_b2, soap_eps, soap_every, ema_osc);
#else
                        else
                            k_soap_a<T><<<n_sunits, 256, soap_smem, st>>>(d_smats, d_sunits, d_pdesc, d_lr, d_step_opt,
                                                                          d_sws, soap_mult, soap_b1, soap_b2, soap_eps,
                                                                          soap_every);
#endif
                    } else
                        k_soap_a_smem_d<<<n_sunits, 256, soap_smem_a, st>>>(d_smats, d_sunits, d_pdesc, d_lr, d_step_opt,
                                                                            d_sws, soap_mult, soap_b1, soap_b2,
                                                                            soap_eps, soap_every, ema_osc);
                } else {
                    k_soap_a<T><<<n_sunits, 256, soap_smem, st>>>(d_smats, d_sunits, d_pdesc, d_lr, d_step_opt, d_sws,
                                                                  soap_mult, soap_b1, soap_b2, soap_eps, soap_every);
                }
                if (soap_par_adam) {
                    if (adam_compact && d_adam_idx) {
                        if (n_adam_idx > 0)
                            k_caut_adam_compact<T><<<n_adam_idx, TPB, 0, s2>>>(
                                d_pdesc, d_adam_idx, LST_N_PARAMS, d_osc, d_wd, T(opt.beta1), T(opt.beta2), T(opt.eps),
                                adam_cautious ? 1 : 0);
                    } else {
                        if (adam_cautious)
                            k_caut_count<T><<<dim3(16, LST_N_PARAMS), TPB, 0, s2>>>(d_pdesc, LST_N_PARAMS, d_osc, d_wd,
                                                                                   T(opt.beta1), d_caut);
                        k_adam<T><<<dim3(16, LST_N_PARAMS), TPB, 0, s2>>>(d_pdesc, LST_N_PARAMS, d_osc, d_wd, T(opt.beta1),
                                                                          T(opt.beta2), T(opt.eps), nullptr,
                                                                          opt.update_clip, T(soap_b1),
                                                                          muon_nesterov, 1,
                                                                          adam_cautious ? d_caut : nullptr, d_lr, d_step_opt,
                                                                          muon_wd, 1);
                    }
                }
                if (use_geo_out) {
                    if constexpr (std::is_same<T, float>::value) {
                        if (graph_pipe >= 3 && soap_mode == 1) CK(cudaEventRecord(ev_geo_go, st));
                        CK(cudaStreamWaitEvent(s_cond, ev_geo_go, 0));
                        const long gmax_p = std::max<long>(std::max<long>(E, N + 1), std::max<long>(P, G1));
                        if (fast_g) {
                            const int nb_p = nblk(P), nb_e = nblk(E), nb_n = nblk(N + 1);
                            k_gather_basis_fast<<<nb_p + nb_e + nb_n + 1, TPB, 0, s_cond>>>(
                                d_step, d_fids, sh.B, ks_topo, ks_frames, b, N, E, P, G1, S, F, Y, UP, nb_p, nb_e, nb_n);
                        } else {
                            k_gather<T, true><<<nblk(gmax_p), TPB, 0, s_cond>>>(
                                d_step, d_fids, sh.B, use_ks ? ks_topo : topo, use_ks ? ks_frames : frames,
                                b, N, E, P, G1, S, fp_pair ? F : nullptr);
                            k_basis_pair<T, true><<<nblk(P), TPB, 0, s_cond>>>(
                                b.vec, b.pe0, Y, EF, DEF, UP, d_bw, T(md.prefactor),
                                T(md.r_max), md.p_cut, NB, P, b.sp_idx, b.sp_u,
                                sp_invh(), mlp_spline, T(sp_r_lo));
                        }
                        segg::k_spc_group<<<1, 256, 0, s_cond>>>(
                            b.spc, b.spc_off, b.spc_idx, b.spc_tiles, N, S, b.spc_tiles32,
                            b.spc_wsk, wtarget);
                        CK(cudaEventRecord(ev_geo_ready, s_cond));
                    }
                }
                if (soap_eig) {
                    if constexpr (std::is_same<T, float>::value) {
                        if (use_sm && soap_jfuse) {
                            if (soap_opt > 0 || soap_qr || soap_tc) {
                                bool cond_done = false;
#if CUDART_VERSION >= 12040
                                if (soap_cond_h && soap_qr > 0 && soap_tc) {
                                    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
                                    cudaGraph_t cap_g = nullptr;
                                    const cudaGraphNode_t* deps = nullptr;
                                    const cudaGraphEdgeData* ed = nullptr;
                                    size_t nd = 0;
                                    CK(cudaStreamGetCaptureInfo(st, &cap_st, nullptr, &cap_g, &deps, &ed, &nd));
                                    cudaGraphNodeParams np{};
                                    np.type = cudaGraphNodeTypeConditional;
                                    np.conditional.handle = soap_cond_h;
                                    np.conditional.type = cudaGraphCondTypeIf;
                                    np.conditional.size = 1;
                                    cudaGraphNode_t cnode = nullptr;
                                    CK(cudaGraphAddNode(&cnode, cap_g, deps, ed, nd, &np));
                                    CK(cudaStreamUpdateCaptureDependencies(st, &cnode, nullptr, 1,
                                                                           cudaStreamSetCaptureDependencies));
                                    cudaGraph_t body_g = np.conditional.phGraph_out[0];
                                    cudaGraphNode_t n_eig = nullptr, n_ap = nullptr;
                                    if (n_seunits) {
                                        const SoapMat* a_sm = d_smats;
                                        const int2* a_eu = d_seunits;
                                        const PDesc<float>* a_pd = d_pdesc;
                                        float* a_ws = d_sws;
                                        int a_ev = soap_every, a_sw = soap_sweeps, a_qr = soap_qr;
                                        double a_tol = soap_tol;
                                        __nv_bfloat16* a_qbf = sqbf_ptr;
                                        void* eig_args[] = {&a_sm, &a_eu, &a_pd, &a_ws, &a_ev, &a_tol, &a_sw, &a_qr, &a_qbf};
                                        cudaKernelNodeParams kp{};
                                        kp.func = (void*)k_soap_eig_qr_smem;
                                        kp.gridDim = dim3(n_seunits);
                                        kp.blockDim = dim3(256);
                                        kp.sharedMemBytes = (unsigned int)soap_smem_a;
                                        kp.kernelParams = eig_args;
                                        CK(cudaGraphAddKernelNode(&n_eig, body_g, nullptr, 0, &kp));
                                    }
                                    {
                                        const SoapMat* a_sm = d_smats;
                                        const int2* a_su = d_sunits;
                                        const PDesc<float>* a_pd = d_pdesc;
                                        const double* a_lr = d_lr;
                                        const int* a_st = d_step_opt;
                                        float* a_ws = d_sws;
                                        double a_mu = soap_mult, a_b1 = soap_b1, a_b2 = soap_b2, a_ep = soap_eps;
                                        int a_ev = soap_every;
                                        const float* a_eo = ema_osc;
                                        int a_co = soap_coal ? 1 : 0;
                                        int a_fg = soap_fuse_g ? 1 : 0;
                                        const float* a_go = d_osc;
                                        const __nv_bfloat16* a_qbf = sqbf_ptr;
                                        int a_mr = use_mrot;
                                        void* ap_args[] = {&a_sm, &a_su, &a_pd, &a_lr, &a_st, &a_ws, &a_mu, &a_b1,
                                                           &a_b2, &a_ep, &a_ev, &a_eo, &a_co, &a_fg, &a_go, &a_qbf, &a_mr};
                                        cudaKernelNodeParams kp{};
                                        kp.func = (void*)k_soap_ap_smem_bf16;
                                        kp.gridDim = dim3(n_sunits);
                                        kp.blockDim = dim3(256);
                                        kp.sharedMemBytes = 0;
                                        kp.kernelParams = ap_args;
                                        CK(cudaGraphAddKernelNode(&n_ap, body_g, n_eig ? &n_eig : nullptr,
                                                                  n_eig ? 1 : 0, &kp));
                                    }
                                    cond_done = true;
                                }
#endif
                                if (!cond_done && soap_mode != 2) {
                                    if (n_seunits) {
                                        if (soap_qr > 0)
                                            k_soap_eig_qr_smem<<<n_seunits, 256, soap_smem_a, st>>>(
                                                d_smats, d_seunits, d_pdesc, d_sws, soap_every, soap_tol, soap_sweeps,
                                                soap_qr, sqbf_ptr);
#ifndef LST_LEAN
                                        else if (soap_opt >= 2)
                                            k_soap_eig_smem_opt<true><<<n_seunits, 256, soap_smem_a, st>>>(
                                                d_smats, d_seunits, d_pdesc, d_sws, soap_every, soap_tol, soap_sweeps);
                                        else
                                            k_soap_eig_smem_opt<false><<<n_seunits, 256, soap_smem_a, st>>>(
                                                d_smats, d_seunits, d_pdesc, d_sws, soap_every, soap_tol, soap_sweeps);
#else
                                        else
                                            k_soap_eig<T><<<n_seunits, 256, soap_smem, st>>>(
                                                d_smats, d_seunits, d_pdesc, d_sws, soap_every, soap_tol, soap_sweeps,
                                                soap_jfuse);
#endif
                                    }
                                    if (soap_tc)
                                        k_soap_ap_smem_bf16<<<n_sunits, 256, 0, st>>>(
                                            d_smats, d_sunits, d_pdesc, d_lr, d_step_opt, d_sws, soap_mult, soap_b1, soap_b2,
                                            soap_eps, soap_every, ema_osc, soap_coal ? 1 : 0, soap_fuse_g ? 1 : 0, d_osc,
                                            sqbf_ptr, use_mrot);
#ifndef LST_LEAN
                                    else
                                        k_soap_ap_smem_opt<<<n_sunits, 256, soap_smem_a, st>>>(
                                            d_smats, d_sunits, d_pdesc, d_lr, d_step_opt, d_sws, soap_mult, soap_b1, soap_b2,
                                            soap_eps, soap_every, ema_osc);
#else
                                    else
                                        k_soap_ap<T><<<n_sunits, 256, soap_smem, st>>>(
                                            d_smats, d_sunits, d_pdesc, d_lr, d_step_opt, d_sws, soap_mult, soap_b1, soap_b2,
                                            soap_eps, soap_every, ema_osc);
#endif
                                }
                            } else {
#ifndef LST_LEAN
                                if (n_seunits)
                                    k_soap_eig_smem_f<<<n_seunits, 256, soap_smem_a, st>>>(
                                        d_smats, d_seunits, d_pdesc, d_sws, soap_every, soap_tol, soap_sweeps);
                                k_soap_ap_smem_f<<<n_sunits, 256, soap_smem_a, st>>>(
                                    d_smats, d_sunits, d_pdesc, d_lr, d_step_opt, d_sws, soap_mult, soap_b1, soap_b2,
                                    soap_eps, soap_every, ema_osc);
#else
                                if (n_seunits)
                                    k_soap_eig<T><<<n_seunits, 256, soap_smem, st>>>(
                                        d_smats, d_seunits, d_pdesc, d_sws, soap_every, soap_tol, soap_sweeps, soap_jfuse);
                                k_soap_ap<T><<<n_sunits, 256, soap_smem, st>>>(
                                    d_smats, d_sunits, d_pdesc, d_lr, d_step_opt, d_sws, soap_mult, soap_b1, soap_b2,
                                    soap_eps, soap_every, ema_osc);
#endif
                            }
                        } else {
                            if (n_seunits)
                                k_soap_eig<T><<<n_seunits, 256, soap_smem, st>>>(
                                    d_smats, d_seunits, d_pdesc, d_sws, soap_every, soap_tol, soap_sweeps, soap_jfuse);
                            k_soap_ap<T><<<n_sunits, 256, soap_smem, st>>>(
                                d_smats, d_sunits, d_pdesc, d_lr, d_step_opt, d_sws, soap_mult, soap_b1, soap_b2, soap_eps,
                                soap_every, ema_osc);
                        }
                    } else {
                        if (n_seunits)
                            k_soap_eig<T><<<n_seunits, 256, soap_smem, st>>>(
                                d_smats, d_seunits, d_pdesc, d_sws, soap_every, soap_tol, soap_sweeps, soap_jfuse);
                        k_soap_ap<T><<<n_sunits, 256, soap_smem, st>>>(
                            d_smats, d_sunits, d_pdesc, d_lr, d_step_opt, d_sws, soap_mult, soap_b1, soap_b2, soap_eps,
                            soap_every, ema_osc);
                    }
                } else {
#ifndef LST_LEAN
                    if constexpr (std::is_same<T, float>::value) {
                        if (use_sm && soap_jfuse)
                            k_soap_b_smem_f<<<n_soap, 256, soap_smem_a, st>>>(
                                d_smats, d_pdesc, d_lr, d_step_opt, d_sws, soap_mult, soap_b1, soap_b2, soap_eps,
                                soap_every, soap_tol, soap_sweeps, ema_osc);
                        else
                            k_soap_b<T><<<n_soap, 256, use_sm ? soap_smem_b : soap_smem, st>>>(
                                d_smats, d_pdesc, d_lr, d_step_opt, d_sws, soap_mult, soap_b1, soap_b2, soap_eps,
                                soap_every, soap_tol, soap_sweeps, soap_jfuse, use_sm ? 1 : 0, ema_osc);
                    } else {
                        k_soap_b<T><<<n_soap, 256, use_sm ? soap_smem_b : soap_smem, st>>>(
                            d_smats, d_pdesc, d_lr, d_step_opt, d_sws, soap_mult, soap_b1, soap_b2, soap_eps, soap_every,
                            soap_tol, soap_sweeps, soap_jfuse, use_sm ? 1 : 0, ema_osc);
                    }
#else
                    if (n_seunits)
                        k_soap_eig<T><<<n_seunits, 256, soap_smem, st>>>(
                            d_smats, d_seunits, d_pdesc, d_sws, soap_every, soap_tol, soap_sweeps, soap_jfuse);
                    k_soap_ap<T><<<n_sunits, 256, soap_smem, st>>>(
                        d_smats, d_sunits, d_pdesc, d_lr, d_step_opt, d_sws, soap_mult, soap_b1, soap_b2, soap_eps,
                        soap_every, ema_osc);
#endif
                }
                if (have_ema && !ema_osc) k_muon_ema<T><<<dim3(16, LST_N_PARAMS), TPB, 0, st>>>(d_pdesc, LST_N_PARAMS, d_osc);
                KCHECK();
            } else if (n_mmats) {
                bool done = false;
                if constexpr (std::is_same<T, float>::value) {
                    if (muon_fast) {
                        k_muon_ns_f<<<n_mmats, 256, mu::SMEM, st>>>(d_mmats, d_pdesc, d_lr, d_step_opt, d_mws,
                                                                    muon_scale, muon_steps, muon_polish, d_mnv,
                                                                    muon_nbeta);
                        done = true;
                    }
                }
                if (!done)
                    k_muon_ns<T><<<n_mmats, 256, 0, st>>>(d_mmats, d_pdesc, d_lr, d_step_opt, d_mws, muon_scale,
                                                          muon_steps, muon_polish, d_mnv, muon_nbeta);
                if (have_ema) k_muon_ema<T><<<dim3(16, LST_N_PARAMS), TPB, 0, st>>>(d_pdesc, LST_N_PARAMS, d_osc);
                KCHECK();
            }
            if (soap_par_adam) join(0);
        } else {
            k_step_end<<<1, 1, 0, st>>>(d_step);
        }
        KCHECK();
    }

    void check_ready() {
        if (!have_params || !have_frames) throw Error("parameters and frames must be set");
        if (!d_losses) throw Error("lst_begin_epoch first");
    }

    void run(int n, cudaStream_t user) override {
        check_ready();
        if (!have_optim) throw Error("lst_set_optim first");
        eval_views_valid = false;
        CK(cudaEventRecord(ev_in, user));
        CK(cudaStreamWaitEvent(st, ev_in, 0));
        const bool use_2g = soap_2graph && n_mmats && soap_on && soap_every > 1 && soap_eig && soap_smem_on &&
                            soap_smem_ok && soap_jfuse && soap_tc && (soap_qr > 0) && !soap_cond_en &&
                            std::is_same<T, float>::value;
        if (use_2g && opt_step_host < 0) {
            float hs[LST_N_PARAMS];
            for (int i = 0; i < LST_N_PARAMS; ++i)
                CK(cudaMemcpyAsync(&hs[i], h_pdesc[i].step, sizeof(float), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            const float s0 = hs[0];
            bool ok = (s0 >= 0.f && s0 <= 16777215.f && std::floor(s0) == s0);
            for (int i = 1; i < LST_N_PARAMS && ok; ++i)
                if (hs[i] != s0) ok = false;
            opt_steps_aligned = ok;
            opt_step_host = ok ? (long)s0 : 0;
        }
        auto capture_one = [&](cudaGraph_t& g_out, cudaGraphExec_t& e_out, int smode) {
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            try {
                body(true, false, false, smode);
            } catch (...) {
                cudaGraph_t gg;
                cudaStreamEndCapture(st, &gg);
                if (gg) cudaGraphDestroy(gg);
                throw;
            }
            CK(cudaStreamEndCapture(st, &g_out));
            CK(cudaGraphInstantiate(&e_out, g_out, 0));
        };
        auto capture_cycle = [&](int clen) {
            if (gexec_cycle) { cudaGraphExecDestroy(gexec_cycle); gexec_cycle = nullptr; }
            if (graph_cycle) { cudaGraphDestroy(graph_cycle); graph_cycle = nullptr; }
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            try {
                for (int k = 0; k < clen; ++k) {
                    const int sm = (k % soap_every == 0) ? 1 : 2;
                    const bool g_in = (graph_pipe == 4) ? (k > 0 && ((k - 1) % soap_every == 0)) : (k > 0);
                    const bool g_out = (graph_pipe == 4) ? ((k + 1 < clen) && (k % soap_every == 0)) : (k + 1 < clen);
                    body(true, false, false, sm, g_in, g_out);
                }
            } catch (...) {
                cudaGraph_t gg;
                cudaStreamEndCapture(st, &gg);
                if (gg) cudaGraphDestroy(gg);
                throw;
            }
            CK(cudaStreamEndCapture(st, &graph_cycle));
            CK(cudaGraphInstantiate(&gexec_cycle, graph_cycle, 0));
            graph_cycle_len = clen;
        };
        const bool can_cyc = (graph_pipe > 0) && use_2g && opt_steps_aligned && soap_every >= 2 && soap_every <= 20;
        const int cyc_len = can_cyc ? (soap_every * graph_cyc_mult) : 0;
        if (use_2g && opt_steps_aligned) {
            for (int i = 0; i < n;) {
                if (can_cyc && cyc_len >= 2 && (i + cyc_len <= n) && (opt_step_host % soap_every == 0)) {
                    if (!gexec_cycle || graph_cycle_len != cyc_len) capture_cycle(cyc_len);
                    CK(cudaGraphLaunch(gexec_cycle, st));
                    opt_step_host += cyc_len;
                    i += cyc_len;
                } else {
                    const bool is_qr = (opt_step_host % soap_every == 0);
                    if (is_qr) {
                        if (!gexec) capture_one(graph, gexec, 1);
                        CK(cudaGraphLaunch(gexec, st));
                    } else {
                        if (!gexec_fast) capture_one(graph_fast, gexec_fast, 2);
                        CK(cudaGraphLaunch(gexec_fast, st));
                    }
                    ++opt_step_host;
                    ++i;
                }
            }
        } else {
            if (!gexec) capture_one(graph, gexec, 0);
            for (int i = 0; i < n; ++i) CK(cudaGraphLaunch(gexec, st));
        }
        CK(cudaEventRecord(ev_out, st));
        CK(cudaStreamWaitEvent(user, ev_out, 0));
    }

    void losses(double* o, cudaStream_t user) override {
        CK(cudaEventRecord(ev_in, user));
        CK(cudaStreamWaitEvent(st, ev_in, 0));
        std::vector<T> h(n_epoch);
        CK(cudaMemcpyAsync(h.data(), d_losses, sizeof(T) * n_epoch, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int i = 0; i < n_epoch; ++i) o[i] = (double)h[i];
    }

    void debug_step(int batch, int update, cudaStream_t user) override {
        check_ready();
        eval_views_valid = false;
        if (update) {
            opt_step_host = -1;
            opt_steps_aligned = false;
        }
        CK(cudaEventRecord(ev_in, user));
        CK(cudaStreamWaitEvent(st, ev_in, 0));
        CK(cudaMemcpyAsync(d_step, &batch, sizeof(int), cudaMemcpyHostToDevice, st));
        in_debug = true;
        body(update != 0);
        in_debug = false;
        CK(cudaStreamSynchronize(st));
    }

    void evaluate(const int* ids, int n, void* e, void* f, void* s, cudaStream_t user) override {
        if (!have_params || !have_frames) throw Error("parameters and frames must be set");
        if (n < 1 || n > sh.B) throw Error("lst_evaluate: 1 .. B frames");
        std::vector<int> h(sh.B, -1);
        for (int k = 0; k < n; ++k) h[k] = ids[k];
        CK(cudaEventRecord(ev_in, user));
        CK(cudaStreamWaitEvent(st, ev_in, 0));
        // pageable source: staged before the call returns
        CK(cudaMemcpyAsync(d_efids, h.data(), sizeof(int) * sh.B, cudaMemcpyHostToDevice, st));
        const bool e_only = (f == nullptr && s == nullptr);
        const bool can_graph = eval_cache && eval_views_valid && mspl_active() && !in_debug;
        if (can_graph && e_only) {
            if (!eval_e_exec || eval_e_dst != e) {
                if (eval_e_exec) { cudaGraphExecDestroy(eval_e_exec); eval_e_exec = nullptr; }
                cudaGraph_t eg = nullptr;
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                body(false, true, true);
                k_eval_out<T><<<nblk((long)G1), TPB, 0, st>>>(
                    Eg, b.core_e, mv, b.volume, b.core_s, F, (T*)e, nullptr, nullptr, N, G1);
                CK(cudaStreamEndCapture(st, &eg));
                CK(cudaGraphInstantiate(&eval_e_exec, eg, 0));
                cudaGraphDestroy(eg);
                eval_e_dst = e;
            }
            CK(cudaGraphLaunch(eval_e_exec, st));
        } else if (can_graph && !e_only) {
            if (!eval_fs_exec || eval_e_dst != e || eval_f_dst != f || eval_s_dst != s) {
                if (eval_fs_exec) { cudaGraphExecDestroy(eval_fs_exec); eval_fs_exec = nullptr; }
                cudaGraph_t fg = nullptr;
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                body(false, true, false);
                k_eval_out<T><<<nblk(std::max<long>(3L * N, G1)), TPB, 0, st>>>(
                    Eg, b.core_e, mv, b.volume, b.core_s, F, (T*)e, (T*)f, (T*)s, N, G1);
                CK(cudaStreamEndCapture(st, &fg));
                CK(cudaGraphInstantiate(&eval_fs_exec, fg, 0));
                cudaGraphDestroy(fg);
                eval_e_dst = e; eval_f_dst = f; eval_s_dst = s;
            }
            CK(cudaGraphLaunch(eval_fs_exec, st));
        } else {
            body(false, true, e_only);
            k_eval_out<T><<<nblk(e_only ? (long)G1 : std::max<long>(3L * N, G1)), TPB, 0, st>>>(
                Eg, b.core_e, mv, b.volume, b.core_s, F, (T*)e, (T*)f, (T*)s, N, G1);
            KCHECK();
            if (eval_cache && mspl_active() && !in_debug) eval_views_valid = true;
        }
        CK(cudaEventRecord(ev_out, st));
        CK(cudaStreamWaitEvent(user, ev_out, 0));
    }

    long stage_size(const std::string& n) override {
        auto it = stages.find(n);
        return it == stages.end() ? -1 : it->second.second;
    }
    void stage_copy(const std::string& n, void* dst, cudaStream_t user) override {
        auto it = stages.find(n);
        if (it == stages.end()) throw Error("unknown stage " + n);
        CK(cudaEventRecord(ev_out, st));
        CK(cudaStreamWaitEvent(user, ev_out, 0));
        CK(cudaMemcpyAsync(dst, it->second.first, sizeof(T) * it->second.second, cudaMemcpyDeviceToDevice, user));
    }
    const void* stage_ptr(const std::string& n) override {
        auto it = stages.find(n);
        return it == stages.end() ? nullptr : (const void*)it->second.first;
    }
};

}  // namespace lst

// ============================================================================ C API
struct LstEngine {
    std::unique_ptr<lst::Base> e;
    std::string err;
};

static thread_local std::string g_create_err;

template <typename F>
static int guard(LstEngine* h, F&& f) {
    if (!h) return -1;
    try {
        f();
        return 0;
    } catch (const std::exception& ex) {
        h->err = ex.what();
        return -1;
    }
}

extern "C" {

int lst_abi_version(void) { return LST_ABI_VERSION; }

const char* lst_last_error(const LstEngine* h) { return h ? h->err.c_str() : g_create_err.c_str(); }

LstEngine* lst_create(const LstModelDesc* m, const LstShape* s, int device) {
    try {
        auto* h = new LstEngine();
        if (m->dtype == 0) h->e.reset(new lst::Engine<float>(*m, *s, device));
        else h->e.reset(new lst::Engine<double>(*m, *s, device));
        return h;
    } catch (const std::exception& ex) {
        g_create_err = ex.what();
        return nullptr;
    }
}

void lst_destroy(LstEngine* h) { delete h; }

int lst_set_params(LstEngine* h, void* const* p) { return guard(h, [&] { h->e->set_params(p); }); }
int lst_set_frames(LstEngine* h, const LstFrames* f) { return guard(h, [&] { h->e->set_frames(f); }); }
int lst_set_node_order(LstEngine* h, const int32_t* order) {
    return guard(h, [&] { h->e->set_node_order(order); });
}
int lst_set_node_energy(LstEngine* h, const void* ei_ref, double weight) {
    return guard(h, [&] { h->e->set_node_energy(ei_ref, weight); });
}
int lst_set_node_energy_steps(LstEngine* h, const double* w_ei_steps, int n_steps, void* stream) {
    return guard(h, [&] { h->e->set_node_energy_steps(w_ei_steps, n_steps, (cudaStream_t)stream); });
}
int lst_set_optim(LstEngine* h, const LstOptim* o) { return guard(h, [&] { h->e->set_optim(o); }); }
int lst_set_muon(LstEngine* h, const LstMuonBlock* b, int n, double momentum, int nesterov, double scale) {
    return guard(h, [&] { h->e->set_muon(b, n, momentum, nesterov, scale); });
}
int lst_set_muon_ns(LstEngine* h, int steps, int polish) {
    return guard(h, [&] { h->e->set_muon_ns(steps, polish); });
}
int lst_set_muon_norm(LstEngine* h, double beta) {
    return guard(h, [&] { h->e->set_muon_norm(beta); });
}
int lst_set_cautious(LstEngine* h, int on) {
    return guard(h, [&] { h->e->set_cautious(on); });
}
int lst_set_muon_wd(LstEngine* h, double wd_mult) {
    return guard(h, [&] { h->e->set_muon_wd(wd_mult); });
}
int lst_set_math(LstEngine* h, int gemm, int mlp) {
    return guard(h, [&] { h->e->set_math(gemm, mlp); });
}
int lst_set_soap(LstEngine* h, double b1, double b2, double eps, int every, int max_dim, double lr_mult) {
    return guard(h, [&] { h->e->set_soap(b1, b2, eps, every, max_dim, lr_mult); });
}
int lst_begin_epoch_steps(LstEngine* h, const LstBatches* b, const double* lr, const double* lw, const double* ema,
                          void* stream) {
    return guard(h, [&] { h->e->begin_epoch(b, lr, lw, ema, true, (cudaStream_t)stream); });
}

int lst_begin_epoch(LstEngine* h, const LstBatches* b, const double* lr, const double* lw, const double* ema,
                    void* stream) {
    return guard(h, [&] { h->e->begin_epoch(b, lr, lw, ema, false, (cudaStream_t)stream); });
}
int lst_run_steps(LstEngine* h, int n, void* stream) { return guard(h, [&] { h->e->run(n, (cudaStream_t)stream); }); }
int lst_epoch_losses(LstEngine* h, double* out, void* stream) {
    return guard(h, [&] { h->e->losses(out, (cudaStream_t)stream); });
}
int lst_debug_step(LstEngine* h, int batch, int update, void* stream) {
    return guard(h, [&] { h->e->debug_step(batch, update, (cudaStream_t)stream); });
}
int lst_evaluate(LstEngine* h, const int* frame_ids, int n, void* energy, void* forces, void* stress, void* stream) {
    return guard(h, [&] { h->e->evaluate(frame_ids, n, energy, forces, stress, (cudaStream_t)stream); });
}
int lst_set_eval_cache(LstEngine* h, int on) {
    return guard(h, [&] { h->e->set_eval_cache(on); });
}
long lst_stage_size(LstEngine* h, const char* name) { return h ? h->e->stage_size(name) : -1; }
int lst_stage_copy(LstEngine* h, const char* name, void* dst, void* stream) {
    return guard(h, [&] { h->e->stage_copy(name, dst, (cudaStream_t)stream); });
}
const void* lst_stage_ptr(LstEngine* h, const char* name) { return (h && name) ? h->e->stage_ptr(name) : nullptr; }
}
