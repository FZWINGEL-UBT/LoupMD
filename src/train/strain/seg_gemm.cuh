// The four per-degree products of per_l in one launch (FP32): rows [l^2 N, (l+1)^2 N) of A [16 N, K] times
// W_l = W + l * wstride ([K, Nc] row-major with leading dimension ldw, or its transpose when TW: W_l stored
// [Nc, K]), into the same rows of C [16 N, Nc] (C = A W_l + beta C). 64 x 64 tiles that never straddle a
// degree, 256 threads with 4 x 4 outputs each, K in steps of 32. Needs K % 32 == 0 and Nc % 64 == 0.
#pragma once
#include <type_traits>
#include "act_half.cuh"

namespace lst {
namespace segg {

constexpr int TM = 64, TN = 64, TK = 32;

template <bool TW>
__global__ void __launch_bounds__(256) k_seg_gemm(const float* __restrict__ A, int K, const float* __restrict__ W,
                                                  long wstride, int ldw, int Nc, float* __restrict__ C, float beta,
                                                  int N) {
    __shared__ __align__(16) float As[TK][TM + 4];
    __shared__ __align__(16) float Bs[TK][TN + 4];
    // degree of this row tile
    int t = blockIdx.y, l = 0, nr = N;
    for (;;) {
        nr = (2 * l + 1) * N;
        const int tl = (nr + TM - 1) / TM;
        if (t < tl || l == 3) break;
        t -= tl;
        ++l;
    }
    const long r0 = (long)l * l * N;
    const int m0 = t * TM, n0 = blockIdx.x * TN;
    const float* Al = A + r0 * K;
    const float* Wl = W + l * wstride;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        __syncthreads();
        // A tile: 64 rows x 32 k as 512 float4 (two per thread), stored k-major
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q, row = idx >> 3, kq = (idx & 7) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + row < nr) v = *reinterpret_cast<const float4*>(Al + (long)(m0 + row) * K + k0 + kq);
            As[kq][row] = v.x;
            As[kq + 1][row] = v.y;
            As[kq + 2][row] = v.z;
            As[kq + 3][row] = v.w;
        }
        // B tile: 32 k x 64 n
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q;
            if (!TW) {
                const int kk = idx >> 4, nq = (idx & 15) * 4;
                *reinterpret_cast<float4*>(&Bs[kk][nq]) =
                    *reinterpret_cast<const float4*>(Wl + (long)(k0 + kk) * ldw + n0 + nq);
            } else {
                const int nn = idx >> 3, kq = (idx & 7) * 4;
                const float4 v = *reinterpret_cast<const float4*>(Wl + (long)(n0 + nn) * ldw + k0 + kq);
                Bs[kq][nn] = v.x;
                Bs[kq + 1][nn] = v.y;
                Bs[kq + 2][nn] = v.z;
                Bs[kq + 3][nn] = v.w;
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < TK; ++kk) {
            const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
            const float4 b = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = m0 + ty * 4 + i;
        if (r >= nr) continue;
        float* cp = C + (r0 + r) * Nc + n0 + tx * 4;
        float4 v = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        if (beta != 0.f) {
            const float4 o = *reinterpret_cast<const float4*>(cp);
            v.x = fmaf(beta, o.x, v.x);
            v.y = fmaf(beta, o.y, v.y);
            v.z = fmaf(beta, o.z, v.z);
            v.w = fmaf(beta, o.w, v.w);
        }
        *reinterpret_cast<float4*>(cp) = v;
    }
}

inline int row_tiles(int N) {
    int s = 0;
    for (int l = 0; l < 4; ++l) s += ((2 * l + 1) * N + TM - 1) / TM;
    return s;
}

// ============================================================================ species-sliced M0 (LST_M0_SPECIES=1)
// Group atoms 0..N-1 by species into spc_idx[N] with per-species spans spc_off[S+1] and per-(l,s) tile prefix sums
// spc_tiles[4*S+1]. One block of 256 threads (< 1 us).
__global__ void k_spc_group(const int* __restrict__ spc, int* __restrict__ spc_off, int* __restrict__ spc_idx,
                            int* __restrict__ spc_tiles, int N, int S, int* __restrict__ spc_tiles32 = nullptr,
                            int* __restrict__ spc_wsk = nullptr, int wtarget = 0) {
    __shared__ int cnt[16];
    __shared__ int pos[16];
    const int t = threadIdx.x;
    if (t < S) cnt[t] = 0;
    __syncthreads();
    for (int i = t; i < N; i += blockDim.x) atomicAdd(&cnt[spc[i]], 1);
    __syncthreads();
    if (t == 0) {
        int acc = 0;
        for (int s = 0; s < S; ++s) {
            spc_off[s] = acc;
            pos[s] = acc;
            acc += cnt[s];
        }
        spc_off[S] = acc;
        int tt = 0;
        spc_tiles[0] = 0;
        for (int l = 0; l < 4; ++l) {
            for (int s = 0; s < S; ++s) {
                const int nr = (2 * l + 1) * cnt[s];
                tt += (nr + TM - 1) / TM;
                spc_tiles[l * S + s + 1] = tt;
            }
        }
        if (spc_tiles32) {
            int t32 = 0;
            spc_tiles32[0] = 0;
            for (int s = 0; s < S; ++s) {
                t32 += (cnt[s] + 31) / 32;
                spc_tiles32[s + 1] = t32;
            }
        }
        if (spc_wsk && wtarget > 0) {
            const int chunk = max(64, (N + wtarget - 1) / wtarget);
            int tw = 0;
            spc_wsk[0] = 0;
            for (int s = 0; s < S; ++s) {
                tw += max(1, (cnt[s] + chunk - 1) / chunk);
                spc_wsk[s + 1] = tw;
            }
        }
    }
    __syncthreads();
    for (int i = t; i < N; i += blockDim.x) {
        const int s = spc[i];
        const int k = atomicAdd(&pos[s], 1);
        spc_idx[k] = i;
    }
}

inline int spc_max_tiles(int N, int S) {
    return (16 * N + TM - 1) / TM + 4 * S;
}

// Species-sliced M0 forward/backward: C[m, n, :] = A[m, n, :] @ W_{l,s} (or W_{l,s}^T when TW),
// where W_{l,s} is the [K, K] submatrix M0[l, :, s*K : (s+1)*K] with leading dimension ldw = S * K (K = C).
// Eliminates k_expand, k_pick, and 2/3 of M0 GEMM FLOPs/bandwidth. Works for both float and double.
// NM=true: node-major [N, 16, K] layout instead of component-major [16, N, K].
template <typename T, bool TW, bool NM = false>
__global__ void __launch_bounds__(256) k_seg_gemm_spc(const T* __restrict__ A, const T* __restrict__ M0,
                                                      long wstride, int ldw, int K, T* __restrict__ C,
                                                      const int* __restrict__ spc_off, const int* __restrict__ spc_idx,
                                                      const int* __restrict__ spc_tiles, int N, int S) {
    const int tot = spc_tiles[4 * S];
    const int by = blockIdx.y;
    if (by >= tot) return;
    int ls = 0;
    while (ls + 1 < 4 * S && by >= spc_tiles[ls + 1]) ++ls;
    const int l = ls / S, s = ls % S;
    const int t = by - spc_tiles[ls];
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int deg = 2 * l + 1;
    const int nr = deg * cnt;
    const int m0 = t * TM, n0 = blockIdx.x * TN;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    __shared__ long row_map[TM];
    if (tid < TM) {
        const int r = m0 + tid;
        if (r < nr) {
            if constexpr (NM) {
                const int jn = r / deg, dm = r - jn * deg;
                row_map[tid] = (long)spc_idx[off + jn] * 16 + (l * l + dm);
            } else {
                const int dm = r / cnt, jn = r - dm * cnt;
                row_map[tid] = (long)(l * l + dm) * N + spc_idx[off + jn];
            }
        } else {
            row_map[tid] = -1L;
        }
    }
    __shared__ __align__(16) T As[TK][TM + 4];
    __shared__ __align__(16) T Bs[TK][TN + 4];
    const T* Wl = M0 + l * wstride + (long)s * K;
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    for (int k0 = 0; k0 < K; k0 += TK) {
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q, row = idx >> 3, kq = (idx & 7) * 4;
            const long rm = row_map[row];
            if constexpr (std::is_same<T, float>::value) {
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (rm >= 0) v = *reinterpret_cast<const float4*>(A + rm * K + k0 + kq);
                As[kq][row] = v.x;
                As[kq + 1][row] = v.y;
                As[kq + 2][row] = v.z;
                As[kq + 3][row] = v.w;
            } else {
                const T* ap = rm >= 0 ? A + rm * K + k0 + kq : nullptr;
                As[kq][row] = ap ? ap[0] : T(0);
                As[kq + 1][row] = ap ? ap[1] : T(0);
                As[kq + 2][row] = ap ? ap[2] : T(0);
                As[kq + 3][row] = ap ? ap[3] : T(0);
            }
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q;
            if (!TW) {
                const int kk = idx >> 4, nq = (idx & 15) * 4;
                if constexpr (std::is_same<T, float>::value) {
                    *reinterpret_cast<float4*>(&Bs[kk][nq]) =
                        *reinterpret_cast<const float4*>(Wl + (long)(k0 + kk) * ldw + n0 + nq);
                } else {
                    const T* wp = Wl + (long)(k0 + kk) * ldw + n0 + nq;
                    for (int u = 0; u < 4; ++u) Bs[kk][nq + u] = wp[u];
                }
            } else {
                const int nn = idx >> 3, kq = (idx & 7) * 4;
                if constexpr (std::is_same<T, float>::value) {
                    const float4 v = *reinterpret_cast<const float4*>(Wl + (long)(n0 + nn) * ldw + k0 + kq);
                    Bs[kq][nn] = v.x;
                    Bs[kq + 1][nn] = v.y;
                    Bs[kq + 2][nn] = v.z;
                    Bs[kq + 3][nn] = v.w;
                } else {
                    const T* wp = Wl + (long)(n0 + nn) * ldw + k0 + kq;
                    Bs[kq][nn] = wp[0];
                    Bs[kq + 1][nn] = wp[1];
                    Bs[kq + 2][nn] = wp[2];
                    Bs[kq + 3][nn] = wp[3];
                }
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < TK; ++kk) {
            T av[4], bv[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
                const float4 b = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                bv[0] = b.x; bv[1] = b.y; bv[2] = b.z; bv[3] = b.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    av[u] = As[kk][ty * 4 + u];
                    bv[u] = Bs[kk][tx * 4 + u];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const long rm = row_map[ty * 4 + i];
        if (rm < 0) continue;
        T* cp = C + rm * K + n0 + tx * 4;
        if constexpr (std::is_same<T, float>::value) {
            *reinterpret_cast<float4*>(cp) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) cp[j] = acc[i][j];
        }
    }
}

// Species-sliced M0s weight gradient: M0s[l, c, s*C + d] += sum_{m in l, n : spc[n]==s} A[m, n, c] * B[m, n, d].
// Grid: dim3(SK, 4 * S * ntiles_c * ntiles_c) with atomicAdd into zeroed M0s.
template <typename T>
__global__ void __launch_bounds__(256) k_seg_wgrad_spc(const T* __restrict__ A, const T* __restrict__ B,
                                                       T* __restrict__ M0s, long wstride, int ldw, int C,
                                                       const int* __restrict__ spc_off, const int* __restrict__ spc_idx,
                                                       int N, int S) {
    const int ntc = C / TM;
    const int sk = blockIdx.x, SK = gridDim.x;
    const int blk = blockIdx.y;
    const int ls = blk / (ntc * ntc), rem = blk % (ntc * ntc);
    const int tc = rem / ntc, td = rem % ntc;
    const int l = ls / S, s = ls % S;
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int nr = (2 * l + 1) * cnt;
    const int r_beg = (nr * sk) / SK, r_end = (nr * (sk + 1)) / SK;
    if (r_beg >= r_end) return;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int c0 = tc * TM, d0 = td * TN;
    __shared__ long row_map[TK];
    __shared__ __align__(16) T As[TK][TM + 4];
    __shared__ __align__(16) T Bs[TK][TN + 4];
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    for (int r0 = r_beg; r0 < r_end; r0 += TK) {
        if (tid < TK) {
            const int r = r0 + tid;
            if (r < r_end) {
                const int dm = r / cnt, jn = r - dm * cnt;
                row_map[tid] = (long)(l * l + dm) * N + spc_idx[off + jn];
            } else {
                row_map[tid] = -1L;
            }
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
            const long rm = row_map[kk];
            if constexpr (std::is_same<T, float>::value) {
                float4 va = make_float4(0.f, 0.f, 0.f, 0.f), vb = make_float4(0.f, 0.f, 0.f, 0.f);
                if (rm >= 0) {
                    va = *reinterpret_cast<const float4*>(A + rm * C + c0 + col4);
                    vb = *reinterpret_cast<const float4*>(B + rm * C + d0 + col4);
                }
                *reinterpret_cast<float4*>(&As[kk][col4]) = va;
                *reinterpret_cast<float4*>(&Bs[kk][col4]) = vb;
            } else {
                const T* ap = rm >= 0 ? A + rm * C + c0 + col4 : nullptr;
                const T* bp = rm >= 0 ? B + rm * C + d0 + col4 : nullptr;
                for (int u = 0; u < 4; ++u) {
                    As[kk][col4 + u] = ap ? ap[u] : T(0);
                    Bs[kk][col4 + u] = bp ? bp[u] : T(0);
                }
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < TK; ++kk) {
            T av[4], bv[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
                const float4 b = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                bv[0] = b.x; bv[1] = b.y; bv[2] = b.z; bv[3] = b.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    av[u] = As[kk][ty * 4 + u];
                    bv[u] = Bs[kk][tx * 4 + u];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }
    T* out = M0s + (long)l * wstride + (long)s * C;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            atomicAdd(out + (long)(c0 + ty * 4 + i) * ldw + d0 + tx * 4 + j, acc[i][j]);
}

// Node-major per_l for wl1 ([N, 16, C] -> [N, 16, C]): works for float and double (LST_NODE_MAJOR=1).
template <typename T, bool TW>
__global__ void __launch_bounds__(256) k_seg_gemm_nm(const T* __restrict__ A, int K, const T* __restrict__ W,
                                                     long wstride, int ldw, int Nc, T* __restrict__ C, T beta, int N) {
    int t = blockIdx.y, l = 0, nr = N;
    for (;;) {
        nr = (2 * l + 1) * N;
        const int tl = (nr + TM - 1) / TM;
        if (t < tl || l == 3) break;
        t -= tl;
        ++l;
    }
    const int deg = 2 * l + 1;
    const int m0 = t * TM, n0 = blockIdx.x * TN;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    __shared__ long row_map[TM];
    if (tid < TM) {
        const int r = m0 + tid;
        if (r < nr) {
            const int n = r / deg, dm = r - n * deg;
            row_map[tid] = (long)n * 16 + (l * l + dm);
        } else {
            row_map[tid] = -1L;
        }
    }
    __shared__ __align__(16) T As[TK][TM + 4];
    __shared__ __align__(16) T Bs[TK][TN + 4];
    const T* Wl = W + l * wstride;
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    for (int k0 = 0; k0 < K; k0 += TK) {
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q, row = idx >> 3, kq = (idx & 7) * 4;
            const long rm = row_map[row];
            if constexpr (std::is_same<T, float>::value) {
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (rm >= 0) v = *reinterpret_cast<const float4*>(A + rm * K + k0 + kq);
                As[kq][row] = v.x;
                As[kq + 1][row] = v.y;
                As[kq + 2][row] = v.z;
                As[kq + 3][row] = v.w;
            } else {
                const T* ap = rm >= 0 ? A + rm * K + k0 + kq : nullptr;
                As[kq][row] = ap ? ap[0] : T(0);
                As[kq + 1][row] = ap ? ap[1] : T(0);
                As[kq + 2][row] = ap ? ap[2] : T(0);
                As[kq + 3][row] = ap ? ap[3] : T(0);
            }
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q;
            if (!TW) {
                const int kk = idx >> 4, nq = (idx & 15) * 4;
                if constexpr (std::is_same<T, float>::value) {
                    *reinterpret_cast<float4*>(&Bs[kk][nq]) =
                        *reinterpret_cast<const float4*>(Wl + (long)(k0 + kk) * ldw + n0 + nq);
                } else {
                    const T* wp = Wl + (long)(k0 + kk) * ldw + n0 + nq;
                    for (int u = 0; u < 4; ++u) Bs[kk][nq + u] = wp[u];
                }
            } else {
                const int nn = idx >> 3, kq = (idx & 7) * 4;
                if constexpr (std::is_same<T, float>::value) {
                    const float4 v = *reinterpret_cast<const float4*>(Wl + (long)(n0 + nn) * ldw + k0 + kq);
                    Bs[kq][nn] = v.x;
                    Bs[kq + 1][nn] = v.y;
                    Bs[kq + 2][nn] = v.z;
                    Bs[kq + 3][nn] = v.w;
                } else {
                    const T* wp = Wl + (long)(n0 + nn) * ldw + k0 + kq;
                    Bs[kq][nn] = wp[0];
                    Bs[kq + 1][nn] = wp[1];
                    Bs[kq + 2][nn] = wp[2];
                    Bs[kq + 3][nn] = wp[3];
                }
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < TK; ++kk) {
            T av[4], bv[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
                const float4 b = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                bv[0] = b.x; bv[1] = b.y; bv[2] = b.z; bv[3] = b.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    av[u] = As[kk][ty * 4 + u];
                    bv[u] = Bs[kk][tx * 4 + u];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const long rm = row_map[ty * 4 + i];
        if (rm < 0) continue;
        T* cp = C + rm * Nc + n0 + tx * 4;
        if constexpr (std::is_same<T, float>::value) {
            float4 v = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            if (beta != 0.f) {
                const float4 o = *reinterpret_cast<const float4*>(cp);
                v.x = fmaf(beta, o.x, v.x);
                v.y = fmaf(beta, o.y, v.y);
                v.z = fmaf(beta, o.z, v.z);
                v.w = fmaf(beta, o.w, v.w);
            }
            *reinterpret_cast<float4*>(cp) = v;
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) cp[j] = beta != T(0) ? beta * cp[j] + acc[i][j] : acc[i][j];
        }
    }
}

// Balanced M0s wgrad (LST_WGRAD_BAL=1 / LST_NODE_MAJOR=1): one block per (m, s, sk), zero div/mod.
template <typename T, bool NM = false>
__global__ void __launch_bounds__(256) k_seg_wgrad_spc_bal(const T* __restrict__ A, const T* __restrict__ B,
                                                           T* __restrict__ M0s, long wstride, int ldw, int C,
                                                           const int* __restrict__ spc_off,
                                                           const int* __restrict__ spc_idx, int N, int S) {
    const int ntc = C / TM;
    const int sk = blockIdx.x, SK = gridDim.x;
    const int blk = blockIdx.y;
    const int ms = blk / (ntc * ntc), rem = blk % (ntc * ntc);
    const int tc = rem / ntc, td = rem % ntc;
    const int m = ms / S, s = ms % S;
    const int l = m == 0 ? 0 : m < 4 ? 1 : m < 9 ? 2 : 3;
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int j_beg = (cnt * sk) / SK, j_end = (cnt * (sk + 1)) / SK;
    if (j_beg >= j_end) return;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int c0 = tc * TM, d0 = td * TN;
    __shared__ long row_map[TK];
    __shared__ __align__(16) T As[TK][TM + 4];
    __shared__ __align__(16) T Bs[TK][TN + 4];
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    for (int j0 = j_beg; j0 < j_end; j0 += TK) {
        if (tid < TK) {
            const int j = j0 + tid;
            if (j < j_end) {
                const int n = spc_idx[off + j];
                row_map[tid] = NM ? (long)n * 16 + m : (long)m * N + n;
            } else {
                row_map[tid] = -1L;
            }
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
            const long rm = row_map[kk];
            if constexpr (std::is_same<T, float>::value) {
                float4 va = make_float4(0.f, 0.f, 0.f, 0.f), vb = make_float4(0.f, 0.f, 0.f, 0.f);
                if (rm >= 0) {
                    va = *reinterpret_cast<const float4*>(A + rm * C + c0 + col4);
                    vb = *reinterpret_cast<const float4*>(B + rm * C + d0 + col4);
                }
                *reinterpret_cast<float4*>(&As[kk][col4]) = va;
                *reinterpret_cast<float4*>(&Bs[kk][col4]) = vb;
            } else {
                const T* ap = rm >= 0 ? A + rm * C + c0 + col4 : nullptr;
                const T* bp = rm >= 0 ? B + rm * C + d0 + col4 : nullptr;
                for (int u = 0; u < 4; ++u) {
                    As[kk][col4 + u] = ap ? ap[u] : T(0);
                    Bs[kk][col4 + u] = bp ? bp[u] : T(0);
                }
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < TK; ++kk) {
            T av[4], bv[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
                const float4 b = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                bv[0] = b.x; bv[1] = b.y; bv[2] = b.z; bv[3] = b.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    av[u] = As[kk][ty * 4 + u];
                    bv[u] = Bs[kk][tx * 4 + u];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }
    T* out = M0s + (long)l * wstride + (long)s * C;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            atomicAdd(out + (long)(c0 + ty * 4 + i) * ldw + d0 + tx * 4 + j, acc[i][j]);
}

// Dynamic stoichiometry-balanced M0s wgrad (LST_WGRAD_OPT=1): allocates split-K proportional to cnt[s] via spc_wsk[S+1]
// so majority species (e.g. 60% O in SrTiO3) get 3x more blocks and every CTA processes equal atoms (~128 at b4).
// Grid: dim3(max_wsk, 16 * ntc * ntc), 256 threads.
template <typename T, bool NM = false>
__global__ void __launch_bounds__(256) k_seg_wgrad_spc_dyn(const T* __restrict__ A, const T* __restrict__ B,
                                                           T* __restrict__ M0s, long wstride, int ldw, int C,
                                                           const int* __restrict__ spc_off,
                                                           const int* __restrict__ spc_idx,
                                                           const int* __restrict__ spc_wsk, int N, int S) {
    const int tot_sk = spc_wsk[S];
    const int bx = blockIdx.x;
    if (bx >= tot_sk) return;
    int s = 0;
    while (s + 1 < S && bx >= spc_wsk[s + 1]) ++s;
    const int sk = bx - spc_wsk[s], SK = spc_wsk[s + 1] - spc_wsk[s];
    const int ntc = C / TM;
    const int blk = blockIdx.y;
    const int m = blk / (ntc * ntc), rem = blk % (ntc * ntc);
    const int tc = rem / ntc, td = rem % ntc;
    const int l = m == 0 ? 0 : m < 4 ? 1 : m < 9 ? 2 : 3;
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int j_beg = (cnt * sk) / SK, j_end = (cnt * (sk + 1)) / SK;
    if (j_beg >= j_end) return;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int c0 = tc * TM, d0 = td * TN;
    __shared__ long row_map[TK];
    __shared__ __align__(16) T As[TK][TM + 4];
    __shared__ __align__(16) T Bs[TK][TN + 4];
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    for (int j0 = j_beg; j0 < j_end; j0 += TK) {
        if (tid < TK) {
            const int j = j0 + tid;
            if (j < j_end) {
                const int n = spc_idx[off + j];
                row_map[tid] = NM ? (long)n * 16 + m : (long)m * N + n;
            } else {
                row_map[tid] = -1L;
            }
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
            const long rm = row_map[kk];
            if constexpr (std::is_same<T, float>::value) {
                float4 va = make_float4(0.f, 0.f, 0.f, 0.f), vb = make_float4(0.f, 0.f, 0.f, 0.f);
                if (rm >= 0) {
                    va = *reinterpret_cast<const float4*>(A + rm * C + c0 + col4);
                    vb = *reinterpret_cast<const float4*>(B + rm * C + d0 + col4);
                }
                *reinterpret_cast<float4*>(&As[kk][col4]) = va;
                *reinterpret_cast<float4*>(&Bs[kk][col4]) = vb;
            } else {
                const T* ap = rm >= 0 ? A + rm * C + c0 + col4 : nullptr;
                const T* bp = rm >= 0 ? B + rm * C + d0 + col4 : nullptr;
                for (int u = 0; u < 4; ++u) {
                    As[kk][col4 + u] = ap ? ap[u] : T(0);
                    Bs[kk][col4 + u] = bp ? bp[u] : T(0);
                }
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < TK; ++kk) {
            T av[4], bv[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
                const float4 b = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                bv[0] = b.x; bv[1] = b.y; bv[2] = b.z; bv[3] = b.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    av[u] = As[kk][ty * 4 + u];
                    bv[u] = Bs[kk][tx * 4 + u];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }
    T* out = M0s + (long)l * wstride + (long)s * C;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            atomicAdd(out + (long)(c0 + ty * 4 + i) * ldw + d0 + tx * 4 + j, acc[i][j]);
}

// Balanced wl1s wgrad (LST_WGRAD_BAL=1 / LST_NODE_MAJOR=1): one block per (m, sk), replaces 4 cuBLAS GEMMs + splitKreduce.
template <typename T, bool NM = false>
__global__ void __launch_bounds__(256) k_seg_wgrad_wl1(const T* __restrict__ A, const T* __restrict__ B,
                                                       T* __restrict__ Wg, long wstride, int C, int N) {
    const int ntc = C / TM;
    const int sk = blockIdx.x, SK = gridDim.x;
    const int blk = blockIdx.y;
    const int m = blk / (ntc * ntc), rem = blk % (ntc * ntc);
    const int tc = rem / ntc, td = rem % ntc;
    const int l = m == 0 ? 0 : m < 4 ? 1 : m < 9 ? 2 : 3;
    const int n_beg = (N * sk) / SK, n_end = (N * (sk + 1)) / SK;
    if (n_beg >= n_end) return;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int c0 = tc * TM, d0 = td * TN;
    __shared__ __align__(16) T As[TK][TM + 4];
    __shared__ __align__(16) T Bs[TK][TN + 4];
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    for (int n0 = n_beg; n0 < n_end; n0 += TK) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
            const int n = n0 + kk;
            const bool ok = n < n_end;
            const long rm = ok ? (NM ? (long)n * 16 + m : (long)m * N + n) : 0L;
            if constexpr (std::is_same<T, float>::value) {
                float4 va = make_float4(0.f, 0.f, 0.f, 0.f), vb = make_float4(0.f, 0.f, 0.f, 0.f);
                if (ok) {
                    va = *reinterpret_cast<const float4*>(A + rm * C + c0 + col4);
                    vb = *reinterpret_cast<const float4*>(B + rm * C + d0 + col4);
                }
                *reinterpret_cast<float4*>(&As[kk][col4]) = va;
                *reinterpret_cast<float4*>(&Bs[kk][col4]) = vb;
            } else {
                const T* ap = ok ? A + rm * C + c0 + col4 : nullptr;
                const T* bp = ok ? B + rm * C + d0 + col4 : nullptr;
                for (int u = 0; u < 4; ++u) {
                    As[kk][col4 + u] = ap ? ap[u] : T(0);
                    Bs[kk][col4 + u] = bp ? bp[u] : T(0);
                }
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < TK; ++kk) {
            T av[4], bv[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
                const float4 b = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                bv[0] = b.x; bv[1] = b.y; bv[2] = b.z; bv[3] = b.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    av[u] = As[kk][ty * 4 + u];
                    bv[u] = Bs[kk][tx * 4 + u];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }
    T* out = Wg + (long)l * wstride;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            atomicAdd(out + (long)(c0 + ty * 4 + i) * C + d0 + tx * 4 + j, acc[i][j]);
}

// ============================================================================ 16-bit Tensor Core seg_gemm / seg_wgrad
// (LST_ACT_HALF=1 BF16 / =2 FP16, C == 64, NM == true):
//   - k_seg_gemm_spc_h / k_seg_gemm_nm_h: loads full 64x64 A_h (uint2) & W (float4->H) into 18 KB smem (stride 72 ->
//     0-way bank conflicts!), 1 __syncthreads(), 16 mma.sync.m16n8k16 per warp, coalesced uint32_t (2xH) stores!
//   - k_seg_wgrad_spc_dyn_h / k_seg_wgrad_wl1_h: TK=32 chunks in 10 KB smem (stride 40 -> 0-way bank conflicts!),
//     8 mma.sync.m16n8k16 per chunk, FP32 atomicAdd into M0s / wl1s!

template <typename H, bool TW>
__global__ void __launch_bounds__(256, 4) k_seg_gemm_spc_h(const H* __restrict__ A, const float* __restrict__ M0,
                                                           long wstride, int ldw, H* __restrict__ C,
                                                           const int* __restrict__ spc_off,
                                                           const int* __restrict__ spc_idx,
                                                           const int* __restrict__ spc_tiles, int N, int S) {
    constexpr int K = 64, LDS = 72;
    const int tot = spc_tiles[4 * S];
    const int by = blockIdx.y;
    if (by >= tot) return;
    int ls = 0;
    while (ls + 1 < 4 * S && by >= spc_tiles[ls + 1]) ++ls;
    const int l = ls / S, s = ls % S;
    const int t = by - spc_tiles[ls];
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int deg = 2 * l + 1, nr = deg * cnt;
    const int m0 = t * 64;
    const int tid = threadIdx.x;
    __shared__ long row_map[64];
    if (tid < 64) {
        const int r = m0 + tid;
        if (r < nr) {
            const int jn = r / deg, dm = r - jn * deg;
            row_map[tid] = (long)spc_idx[off + jn] * 16 + (l * l + dm);
        } else {
            row_map[tid] = -1L;
        }
    }
    __shared__ __align__(16) H As[64][LDS];
    __shared__ __align__(16) H Bs[64][LDS];   // Bs[col][k] = W_eff[k, col]
    __syncthreads();
    // Load A_h [64, 64] -> As[r][k] (1024 uint2 = 4 per thread)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int idx = tid + 256 * q, row = idx >> 4, k4 = (idx & 15) * 4;
        const long rm = row_map[row];
        const uint2 v = (rm >= 0) ? *reinterpret_cast<const uint2*>(A + rm * K + k4) : make_uint2(0u, 0u);
        *reinterpret_cast<uint2*>(&As[row][k4]) = v;
    }
    // Load W [64, 64] FP32 -> Bs[col][k] (1024 float4 = 4 per thread)
    const float* Wl = M0 + l * wstride + (long)s * K;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int idx = tid + 256 * q;
        if constexpr (!TW) {
            const int kk = idx >> 4, c4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(Wl + (long)kk * ldw + c4);
            Bs[c4 + 0][kk] = ah::from_f<H>(v.x);
            Bs[c4 + 1][kk] = ah::from_f<H>(v.y);
            Bs[c4 + 2][kk] = ah::from_f<H>(v.z);
            Bs[c4 + 3][kk] = ah::from_f<H>(v.w);
        } else {
            const int col = idx >> 4, k4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(Wl + (long)col * ldw + k4);
            uint2 h4;
            h4.x = ah::pack2<H>(v.x, v.y);
            h4.y = ah::pack2<H>(v.z, v.w);
            *reinterpret_cast<uint2*>(&Bs[col][k4]) = h4;
        }
    }
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
    for (int k0 = 0; k0 < 64; k0 += 16) {
        const int c_lo = k0 + 2 * tl, c_hi = c_lo + 8;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&As[r0][c_lo]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&As[r1][c_lo]);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&As[r0][c_hi]);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&As[r1][c_hi]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = 32 * wn + 8 * j + g;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&Bs[col][c_lo]);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&Bs[col][c_hi]);
            ah::mma_16x8x16<H>(acc[j], a0, a1, a2, a3, b0, b1);
        }
    }
    const long rm0 = row_map[r0], rm1 = row_map[r1];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = 32 * wn + 8 * j + 2 * tl;
        if (rm0 >= 0) *reinterpret_cast<uint32_t*>(C + rm0 * K + c0) = ah::pack2<H>(acc[j][0], acc[j][1]);
        if (rm1 >= 0) *reinterpret_cast<uint32_t*>(C + rm1 * K + c0) = ah::pack2<H>(acc[j][2], acc[j][3]);
    }
}

template <typename H, bool TW>
__global__ void __launch_bounds__(256, 4) k_seg_gemm_nm_h(const H* __restrict__ A, const float* __restrict__ W,
                                                          long wstride, int ldw, H* __restrict__ C, float beta, int N) {
    constexpr int K = 64, LDS = 72;
    int t = blockIdx.y, l = 0, nr = N;
    for (;;) {
        nr = (2 * l + 1) * N;
        const int tl = (nr + 63) / 64;
        if (t < tl || l == 3) break;
        t -= tl;
        ++l;
    }
    const int deg = 2 * l + 1, m0 = t * 64;
    const int tid = threadIdx.x;
    __shared__ long row_map[64];
    if (tid < 64) {
        const int r = m0 + tid;
        if (r < nr) {
            const int n = r / deg, dm = r - n * deg;
            row_map[tid] = (long)n * 16 + (l * l + dm);
        } else {
            row_map[tid] = -1L;
        }
    }
    __shared__ __align__(16) H As[64][LDS];
    __shared__ __align__(16) H Bs[64][LDS];
    __syncthreads();
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int idx = tid + 256 * q, row = idx >> 4, k4 = (idx & 15) * 4;
        const long rm = row_map[row];
        const uint2 v = (rm >= 0) ? *reinterpret_cast<const uint2*>(A + rm * K + k4) : make_uint2(0u, 0u);
        *reinterpret_cast<uint2*>(&As[row][k4]) = v;
    }
    const float* Wl = W + l * wstride;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int idx = tid + 256 * q;
        if constexpr (!TW) {
            const int kk = idx >> 4, c4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(Wl + (long)kk * ldw + c4);
            Bs[c4 + 0][kk] = ah::from_f<H>(v.x);
            Bs[c4 + 1][kk] = ah::from_f<H>(v.y);
            Bs[c4 + 2][kk] = ah::from_f<H>(v.z);
            Bs[c4 + 3][kk] = ah::from_f<H>(v.w);
        } else {
            const int col = idx >> 4, k4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(Wl + (long)col * ldw + k4);
            uint2 h4;
            h4.x = ah::pack2<H>(v.x, v.y);
            h4.y = ah::pack2<H>(v.z, v.w);
            *reinterpret_cast<uint2*>(&Bs[col][k4]) = h4;
        }
    }
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
    for (int k0 = 0; k0 < 64; k0 += 16) {
        const int c_lo = k0 + 2 * tl, c_hi = c_lo + 8;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&As[r0][c_lo]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&As[r1][c_lo]);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&As[r0][c_hi]);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&As[r1][c_hi]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = 32 * wn + 8 * j + g;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&Bs[col][c_lo]);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&Bs[col][c_hi]);
            ah::mma_16x8x16<H>(acc[j], a0, a1, a2, a3, b0, b1);
        }
    }
    const long rm0 = row_map[r0], rm1 = row_map[r1];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = 32 * wn + 8 * j + 2 * tl;
        if (rm0 >= 0) {
            float v0 = acc[j][0], v1 = acc[j][1];
            if (beta != 0.f) {
                const float2 o = ah::unpack2<H>(*reinterpret_cast<const uint32_t*>(C + rm0 * K + c0));
                v0 = fmaf(beta, o.x, v0); v1 = fmaf(beta, o.y, v1);
            }
            *reinterpret_cast<uint32_t*>(C + rm0 * K + c0) = ah::pack2<H>(v0, v1);
        }
        if (rm1 >= 0) {
            float v0 = acc[j][2], v1 = acc[j][3];
            if (beta != 0.f) {
                const float2 o = ah::unpack2<H>(*reinterpret_cast<const uint32_t*>(C + rm1 * K + c0));
                v0 = fmaf(beta, o.x, v0); v1 = fmaf(beta, o.y, v1);
            }
            *reinterpret_cast<uint32_t*>(C + rm1 * K + c0) = ah::pack2<H>(v0, v1);
        }
    }
}

template <typename H>
__global__ void __launch_bounds__(256, 4) k_seg_wgrad_spc_dyn_h(const H* __restrict__ A, const H* __restrict__ B,
                                                                float* __restrict__ M0s, long wstride, int ldw,
                                                                const int* __restrict__ spc_off,
                                                                const int* __restrict__ spc_idx,
                                                                const int* __restrict__ spc_wsk, int S) {
    constexpr int C = 64, LDK = 40;
    const int tot_sk = spc_wsk[S];
    const int bx = blockIdx.x;
    if (bx >= tot_sk) return;
    int s = 0;
    while (s + 1 < S && bx >= spc_wsk[s + 1]) ++s;
    const int sk = bx - spc_wsk[s], SK = spc_wsk[s + 1] - spc_wsk[s];
    const int m = blockIdx.y;
    const int l = m == 0 ? 0 : m < 4 ? 1 : m < 9 ? 2 : 3;
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int j_beg = (cnt * sk) / SK, j_end = (cnt * (sk + 1)) / SK;
    if (j_beg >= j_end) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
    __shared__ long row_map[TK];
    __shared__ __align__(16) H As[64][LDK];   // As[c][kk]
    __shared__ __align__(16) H Bs[64][LDK];   // Bs[d][kk]
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    for (int j0 = j_beg; j0 < j_end; j0 += TK) {
        if (tid < TK) {
            const int j = j0 + tid;
            row_map[tid] = (j < j_end) ? ((long)spc_idx[off + j] * 16 + m) : -1L;
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
            const long rm = row_map[kk];
            uint2 va = make_uint2(0u, 0u), vb = make_uint2(0u, 0u);
            if (rm >= 0) {
                va = *reinterpret_cast<const uint2*>(A + rm * C + col4);
                vb = *reinterpret_cast<const uint2*>(B + rm * C + col4);
            }
            const H* ha = reinterpret_cast<const H*>(&va);
            const H* hb = reinterpret_cast<const H*>(&vb);
            As[col4 + 0][kk] = ha[0]; As[col4 + 1][kk] = ha[1]; As[col4 + 2][kk] = ha[2]; As[col4 + 3][kk] = ha[3];
            Bs[col4 + 0][kk] = hb[0]; Bs[col4 + 1][kk] = hb[1]; Bs[col4 + 2][kk] = hb[2]; Bs[col4 + 3][kk] = hb[3];
        }
        __syncthreads();
#pragma unroll
        for (int k0 = 0; k0 < TK; k0 += 16) {
            const int c_lo = k0 + 2 * tl, c_hi = c_lo + 8;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&As[r0][c_lo]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&As[r1][c_lo]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&As[r0][c_hi]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&As[r1][c_hi]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = 32 * wn + 8 * j + g;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&Bs[col][c_lo]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&Bs[col][c_hi]);
                ah::mma_16x8x16<H>(acc[j], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }
    float* out = M0s + (long)l * wstride + (long)s * C;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = 32 * wn + 8 * j + 2 * tl;
        atomicAdd(out + (long)r0 * ldw + c0, acc[j][0]);
        atomicAdd(out + (long)r0 * ldw + c0 + 1, acc[j][1]);
        atomicAdd(out + (long)r1 * ldw + c0, acc[j][2]);
        atomicAdd(out + (long)r1 * ldw + c0 + 1, acc[j][3]);
    }
}

template <typename H>
__global__ void __launch_bounds__(256, 4) k_seg_wgrad_wl1_h(const H* __restrict__ A, const H* __restrict__ B,
                                                            float* __restrict__ Wg, long wstride, int N) {
    constexpr int C = 64, LDK = 40;
    const int sk = blockIdx.x, SK = gridDim.x;
    const int m = blockIdx.y;
    const int l = m == 0 ? 0 : m < 4 ? 1 : m < 9 ? 2 : 3;
    const int n_beg = (N * sk) / SK, n_end = (N * (sk + 1)) / SK;
    if (n_beg >= n_end) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;
    __shared__ __align__(16) H As[64][LDK];
    __shared__ __align__(16) H Bs[64][LDK];
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    for (int n0 = n_beg; n0 < n_end; n0 += TK) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
            const int n = n0 + kk;
            uint2 va = make_uint2(0u, 0u), vb = make_uint2(0u, 0u);
            if (n < n_end) {
                const long rm = (long)n * 16 + m;
                va = *reinterpret_cast<const uint2*>(A + rm * C + col4);
                vb = *reinterpret_cast<const uint2*>(B + rm * C + col4);
            }
            const H* ha = reinterpret_cast<const H*>(&va);
            const H* hb = reinterpret_cast<const H*>(&vb);
            As[col4 + 0][kk] = ha[0]; As[col4 + 1][kk] = ha[1]; As[col4 + 2][kk] = ha[2]; As[col4 + 3][kk] = ha[3];
            Bs[col4 + 0][kk] = hb[0]; Bs[col4 + 1][kk] = hb[1]; Bs[col4 + 2][kk] = hb[2]; Bs[col4 + 3][kk] = hb[3];
        }
        __syncthreads();
#pragma unroll
        for (int k0 = 0; k0 < TK; k0 += 16) {
            const int c_lo = k0 + 2 * tl, c_hi = c_lo + 8;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&As[r0][c_lo]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&As[r1][c_lo]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&As[r0][c_hi]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&As[r1][c_hi]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = 32 * wn + 8 * j + g;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&Bs[col][c_lo]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&Bs[col][c_hi]);
                ah::mma_16x8x16<H>(acc[j], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }
    float* out = Wg + (long)l * wstride;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = 32 * wn + 8 * j + 2 * tl;
        atomicAdd(out + (long)r0 * C + c0, acc[j][0]);
        atomicAdd(out + (long)r0 * C + c0 + 1, acc[j][1]);
        atomicAdd(out + (long)r1 * C + c0, acc[j][2]);
        atomicAdd(out + (long)r1 * C + c0 + 1, acc[j][3]);
    }
}

// ============================================================================ Wave 17: seg_gemm v2 (LST_SEGG_V2=1)
// 1. k_seg_gemm_spc_h_v2 / k_seg_gemm_nm_h_v2:
//    - Loads W_l [64, 64] into Bs[64][72] ONCE per MT tiles of M (eliminates 50-75% of FP32->H W loads/converts!).
//    - Uses hardware ldmatrix.m8n8.x4 for both As and Bs (3 ldmatrix vs 12 scalar LDS.32 per k0 step!).
//    - Stages 64x64 H output through dead As[64][72] and writes 100% coalesced uint2 (16 threads x 8B = 128B line!).
// 2. k_seg_wgrad_spc_dyn_h_v2 / k_seg_wgrad_wl1_h_v2:
//    - Degree-grouped contiguous (atom, dm) rows: l=3 reads 7 contiguous 128B rows/atom (fixes 0% L1 -> >60% L1!).
//    - TK=64 chunks (halves loop iterations & barriers vs TK=32!), direct uint2 STS.64 row-major into As[64][72],
//      Bs[64][72] (zero scalar STS.16 transposes!), and hardware ldmatrix.m8n8.x4.trans!
//    - Zero row_map __syncthreads() (3 -> 2 barriers per 64-row chunk!).

// Wave 45: Exact 1-cycle unsigned magic division for deg in {1, 3, 5, 7} (verified exact for r < 2^30)
__device__ __forceinline__ int fast_div_deg(int r, int deg) {
    const unsigned ur = static_cast<unsigned>(r);
    if (deg == 1) return r;
    if (deg == 3) return static_cast<int>(__umulhi(ur, 0x55555556u));
    if (deg == 5) return static_cast<int>(__umulhi(ur, 0x33333334u));
    return static_cast<int>(__umulhi(ur, 0x24924925u));
}

// Wave 45 (LST_SEGG_H16=1): Pre-pack M0 [4, 64, S*64] and wl1 [4, 64, 64] FP32 -> FP16/BF16 ONCE per step
// into plane 0 (transposed for !TW) and plane 1 (direct for TW) so all 8 critical-path k_seg_gemm_*_h_v2 calls
// load Bs[64][72] via 2x LDG.128 + 2x STS.128 (0 FP32->H converts, 0 bank-conflicted scalar STS.16!).
// Grid: (4*S + 4) blocks x 256 threads (~1.5 us, hidden on s2 inside FWD L0's shadow!).
template <typename H>
__global__ void __launch_bounds__(256, 4) k_pack_segg_w_h(const float* __restrict__ M0, const float* __restrict__ wl1,
                                                          H* __restrict__ M0_h, H* __restrict__ Wl1_h, int S) {
    const int bx = blockIdx.x;
    const int n_m0 = 4 * S;
    const bool is_m0 = (bx < n_m0);
    const int tid = threadIdx.x;
    const int SC = S * 64;
    const float* Wl = is_m0 ? (M0 + (long)(bx / S) * 64L * SC + (long)(bx % S) * 64L)
                            : (wl1 + (long)(bx - n_m0) * 4096L);
    const int ldw = is_m0 ? SC : 64;
    H* dst_nt = is_m0 ? (M0_h + (long)bx * 4096L) : (Wl1_h + (long)(bx - n_m0) * 4096L);
    H* dst_tw = is_m0 ? (M0_h + (long)(n_m0 + bx) * 4096L) : (Wl1_h + (long)(4 + (bx - n_m0)) * 4096L);

    __shared__ __align__(16) H Ts[64][72];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int idx = tid + 256 * q, row = idx >> 4, c4 = (idx & 15) * 4;
        const float4 v = *reinterpret_cast<const float4*>(Wl + (long)row * ldw + c4);
        uint2 h4;
        h4.x = ah::pack2<H>(v.x, v.y);
        h4.y = ah::pack2<H>(v.z, v.w);
        *reinterpret_cast<uint2*>(dst_tw + (long)row * 64 + c4) = h4;
        *reinterpret_cast<uint2*>(&Ts[row][c4]) = h4;
    }
    __syncthreads();
    uint4* nt4 = reinterpret_cast<uint4*>(dst_nt);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int idx = tid + 256 * q, col = idx >> 3, k8 = (idx & 7) * 8;
        H tmp[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) tmp[j] = Ts[k8 + j][col];
        nt4[idx] = *reinterpret_cast<const uint4*>(tmp);
    }
}

template <typename H, bool TW, int ZIP = 0>
__global__ void __launch_bounds__(256, 4) k_seg_gemm_spc_h_v2(const H* __restrict__ A, const float* __restrict__ M0,
                                                              long wstride, int ldw, H* __restrict__ C,
                                                              const int* __restrict__ spc_off,
                                                              const int* __restrict__ spc_idx,
                                                              const int* __restrict__ spc_tiles, int N, int S, int mt,
                                                              const H* __restrict__ G_zip = nullptr,
                                                              uint32_t* __restrict__ GA_zip = nullptr,
                                                              const H* __restrict__ M0_h = nullptr) {
    constexpr int K = 64, LDS = 72;
    int rem = blockIdx.y, ls = 0, ntiles = 0;
    for (; ls < 4 * S; ++ls) {
        const int tl = spc_tiles[ls + 1] - spc_tiles[ls];
        const int bm = (tl + mt - 1) / mt;
        if (rem < bm) { ntiles = tl; break; }
        rem -= bm;
    }
    if (ls >= 4 * S) return;
    const int l = ls / S, s = ls % S;
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int deg = 2 * l + 1, nr = deg * cnt;
    const int t0 = rem * mt;
    const int t_end = min(ntiles, t0 + mt);

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;

    __shared__ long row_map[64];
    __shared__ __align__(16) H As[64][LDS];
    __shared__ __align__(16) H Bs[64][LDS];  // Bs[col][k] = W_eff[k, col]

    // Load W_{l,s} [64, 64] -> Bs[col][k] ONCE for all mt row tiles!
    if (M0_h) {
        const uint4* Wh = reinterpret_cast<const uint4*>(M0_h + ((long)(TW ? (4 * S) : 0) + ls) * 4096L);
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q, col = idx >> 3, k8 = (idx & 7) * 8;
            *reinterpret_cast<uint4*>(&Bs[col][k8]) = Wh[idx];
        }
    } else {
    const float* Wl = M0 + (long)l * wstride + (long)s * K;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int idx = tid + 256 * q;
        if constexpr (!TW) {
            const int kk = idx >> 4, c4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(Wl + (long)kk * ldw + c4);
            Bs[c4 + 0][kk] = ah::from_f<H>(v.x);
            Bs[c4 + 1][kk] = ah::from_f<H>(v.y);
            Bs[c4 + 2][kk] = ah::from_f<H>(v.z);
            Bs[c4 + 3][kk] = ah::from_f<H>(v.w);
        } else {
            const int col = idx >> 4, k4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(Wl + (long)col * ldw + k4);
            uint2 h4;
            h4.x = ah::pack2<H>(v.x, v.y);
            h4.y = ah::pack2<H>(v.z, v.w);
            *reinterpret_cast<uint2*>(&Bs[col][k4]) = h4;
        }
    }
    }

    const int a_row_ld = 16 * wm + (lane & 15);
    const int a_col_ld = (lane >> 4) * 8;
    const int b_col0 = 32 * wn + ((lane >> 4) * 8) + (lane & 7);
    const int b_koff = ((lane >> 3) & 1) * 8;

    for (int t = t0; t < t_end; ++t) {
        const int m0 = t * 64;
        if (t > t0) __syncthreads();
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + 256 * q, row = idx >> 4, k4 = (idx & 15) * 4;
            const int r = m0 + row;
            long rm = -1L;
            if (r < nr) {
                const int jn = fast_div_deg(r, deg), dm = r - jn * deg;
                rm = (long)spc_idx[off + jn] * 16 + (l * l + dm);
            }
            if (k4 == 0) row_map[row] = rm;
            const uint2 v = (rm >= 0) ? *reinterpret_cast<const uint2*>(A + rm * K + k4) : make_uint2(0u, 0u);
            *reinterpret_cast<uint2*>(&As[row][k4]) = v;
        }
        __syncthreads();
        float acc[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
        for (int k0 = 0; k0 < 64; k0 += 16) {
            uint32_t a0, a1, a2, a3;
            ah::ldmatrix_x4(a0, a1, a2, a3, &As[a_row_ld][k0 + a_col_ld]);
            uint32_t b0, b1, b2, b3;
            ah::ldmatrix_x4(b0, b1, b2, b3, &Bs[b_col0][k0 + b_koff]);
            ah::mma_16x8x16<H>(acc[0], a0, a1, a2, a3, b0, b1);
            ah::mma_16x8x16<H>(acc[1], a0, a1, a2, a3, b2, b3);
            ah::ldmatrix_x4(b0, b1, b2, b3, &Bs[b_col0 + 16][k0 + b_koff]);
            ah::mma_16x8x16<H>(acc[2], a0, a1, a2, a3, b0, b1);
            ah::mma_16x8x16<H>(acc[3], a0, a1, a2, a3, b2, b3);
        }
        __syncthreads();
        // Stage packed output into dead As[64][72], then write coalesced 128B cache-line uint2 rows!
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c0 = 32 * wn + 8 * j + 2 * tl;
            *reinterpret_cast<uint32_t*>(&As[r0][c0]) = ah::pack2<H>(acc[j][0], acc[j][1]);
            *reinterpret_cast<uint32_t*>(&As[r1][c0]) = ah::pack2<H>(acc[j][2], acc[j][3]);
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + 256 * q, row = idx >> 4, c4 = (idx & 15) * 4;
            const long rm = row_map[row];
            if constexpr (ZIP == 5 || ZIP == 6) {
                const bool valid = (rm >= 0);
                uint2 g4 = make_uint2(0u, 0u);
                uint2 a4 = make_uint2(0u, 0u);
                if (valid) {
                    a4 = *reinterpret_cast<const uint2*>(&As[row][c4]);
                    g4 = *reinterpret_cast<const uint2*>(G_zip + rm * K + c4);
                }
                const float2 g01 = ah::unpack2<H>(g4.x), g23 = ah::unpack2<H>(g4.y);
                const float2 a01 = ah::unpack2<H>(a4.x), a23 = ah::unpack2<H>(a4.y);
                float mg = fmaxf(fmaxf(fabsf(g01.x), fabsf(g01.y)), fmaxf(fabsf(g23.x), fabsf(g23.y)));
                float ma = fmaxf(fmaxf(fabsf(a01.x), fabsf(a01.y)), fmaxf(fabsf(a23.x), fabsf(a23.y)));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) {
                    mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, o));
                    ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, o));
                }
                if (valid) {
                    constexpr float QMAX = (ZIP == 5) ? 448.f : 127.f;
                    const float sg = mg * (1.f / QMAX), sa = ma * (1.f / QMAX);
                    const float ig = (mg > 0.f) ? (QMAX / mg) : 0.f;
                    const float ia = (ma > 0.f) ? (QMAX / ma) : 0.f;
                    uint16_t p0, p1, p2, p3;
                    if constexpr (ZIP == 5) {
                        p0 = ah::pack_ga_fp8(g01.x * ig, a01.x * ia);
                        p1 = ah::pack_ga_fp8(g01.y * ig, a01.y * ia);
                        p2 = ah::pack_ga_fp8(g23.x * ig, a23.x * ia);
                        p3 = ah::pack_ga_fp8(g23.y * ig, a23.y * ia);
                    } else {
                        p0 = ah::pack_ga_i8(g01.x * ig, a01.x * ia);
                        p1 = ah::pack_ga_i8(g01.y * ig, a01.y * ia);
                        p2 = ah::pack_ga_i8(g23.x * ig, a23.x * ia);
                        p3 = ah::pack_ga_i8(g23.y * ig, a23.y * ia);
                    }
                    const uint2 out4 = make_uint2(static_cast<uint32_t>(p0) | (static_cast<uint32_t>(p1) << 16),
                                                  static_cast<uint32_t>(p2) | (static_cast<uint32_t>(p3) << 16));
                    uint16_t* ga8 = reinterpret_cast<uint16_t*>(GA_zip);
                    *reinterpret_cast<uint2*>(ga8 + rm * K + c4) = out4;
                    if (c4 == 0) {
                        float2* ga_sc = reinterpret_cast<float2*>(GA_zip + 8L * N * K);
                        ga_sc[rm] = make_float2(sg, sa);
                    }
                }
            } else {
                if (rm < 0) continue;
                const uint2 a4 = *reinterpret_cast<const uint2*>(&As[row][c4]);
                if constexpr (ZIP == 1) {
                    const uint2 g4 = *reinterpret_cast<const uint2*>(G_zip + rm * K + c4);
                    uint4 ga;
                    ga.x = __byte_perm(g4.x, a4.x, 0x5410);
                    ga.y = __byte_perm(g4.x, a4.x, 0x7632);
                    ga.z = __byte_perm(g4.y, a4.y, 0x5410);
                    ga.w = __byte_perm(g4.y, a4.y, 0x7632);
                    *reinterpret_cast<uint4*>(GA_zip + rm * K + c4) = ga;
                } else if constexpr (ZIP == 2) {
                    *reinterpret_cast<uint2*>(C + rm * K + c4) = a4;
                    const uint2 g4 = *reinterpret_cast<const uint2*>(G_zip + rm * K + c4);
                    *reinterpret_cast<uint4*>(GA_zip + rm * K + c4) = make_uint4(a4.x, g4.x, a4.y, g4.y);
                } else if constexpr (ZIP == 4) {
                    const uint2 g4 = *reinterpret_cast<const uint2*>(G_zip + rm * K + c4);
                    const long n = rm >> 4, m = rm & 15;
                    const int l0 = c4 >> 1;
                    uint2* dst2 = reinterpret_cast<uint2*>(GA_zip);
                    const long base = ((n * 8 + (m >> 1)) * 32 + l0) * 2 + (m & 1);
                    dst2[base] = make_uint2(g4.x, a4.x);
                    dst2[base + 2] = make_uint2(g4.y, a4.y);
                } else {
                    *reinterpret_cast<uint2*>(C + rm * K + c4) = a4;
                }
            }
        }
    }
}

template <typename H, bool TW, int ZIP = 0>
__global__ void __launch_bounds__(256, 4) k_seg_gemm_nm_h_v2(const H* __restrict__ A, const float* __restrict__ W,
                                                             long wstride, int ldw, H* __restrict__ C, float beta,
                                                             int N, int mt, const H* __restrict__ G_zip = nullptr,
                                                             uint32_t* __restrict__ GA_zip = nullptr,
                                                             const H* __restrict__ W_h = nullptr) {
    constexpr int K = 64, LDS = 72;
    int rem = blockIdx.y, l = 0, deg = 1, nr = N, ntiles = 0;
    for (;;) {
        deg = 2 * l + 1;
        nr = deg * N;
        ntiles = (nr + 63) / 64;
        const int bm = (ntiles + mt - 1) / mt;
        if (rem < bm || l == 3) break;
        rem -= bm;
        ++l;
    }
    const int t0 = rem * mt;
    if (t0 >= ntiles) return;
    const int t_end = min(ntiles, t0 + mt);

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;

    __shared__ long row_map[64];
    __shared__ __align__(16) H As[64][LDS];
    __shared__ __align__(16) H Bs[64][LDS];

    if (W_h) {
        const uint4* Wh = reinterpret_cast<const uint4*>(W_h + ((long)(TW ? 4 : 0) + l) * 4096L);
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 256 * q, col = idx >> 3, k8 = (idx & 7) * 8;
            *reinterpret_cast<uint4*>(&Bs[col][k8]) = Wh[idx];
        }
    } else {
    const float* Wl = W + (long)l * wstride;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int idx = tid + 256 * q;
        if constexpr (!TW) {
            const int kk = idx >> 4, c4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(Wl + (long)kk * ldw + c4);
            Bs[c4 + 0][kk] = ah::from_f<H>(v.x);
            Bs[c4 + 1][kk] = ah::from_f<H>(v.y);
            Bs[c4 + 2][kk] = ah::from_f<H>(v.z);
            Bs[c4 + 3][kk] = ah::from_f<H>(v.w);
        } else {
            const int col = idx >> 4, k4 = (idx & 15) * 4;
            const float4 v = *reinterpret_cast<const float4*>(Wl + (long)col * ldw + k4);
            uint2 h4;
            h4.x = ah::pack2<H>(v.x, v.y);
            h4.y = ah::pack2<H>(v.z, v.w);
            *reinterpret_cast<uint2*>(&Bs[col][k4]) = h4;
        }
    }
    }

    const int a_row_ld = 16 * wm + (lane & 15);
    const int a_col_ld = (lane >> 4) * 8;
    const int b_col0 = 32 * wn + ((lane >> 4) * 8) + (lane & 7);
    const int b_koff = ((lane >> 3) & 1) * 8;

    for (int t = t0; t < t_end; ++t) {
        const int m0 = t * 64;
        if (t > t0) __syncthreads();
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + 256 * q, row = idx >> 4, k4 = (idx & 15) * 4;
            const int r = m0 + row;
            long rm = -1L;
            if (r < nr) {
                const int n = fast_div_deg(r, deg), dm = r - n * deg;
                rm = (long)n * 16 + (l * l + dm);
            }
            if (k4 == 0) row_map[row] = rm;
            const uint2 v = (rm >= 0) ? *reinterpret_cast<const uint2*>(A + rm * K + k4) : make_uint2(0u, 0u);
            *reinterpret_cast<uint2*>(&As[row][k4]) = v;
        }
        __syncthreads();
        float acc[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
        for (int k0 = 0; k0 < 64; k0 += 16) {
            uint32_t a0, a1, a2, a3;
            ah::ldmatrix_x4(a0, a1, a2, a3, &As[a_row_ld][k0 + a_col_ld]);
            uint32_t b0, b1, b2, b3;
            ah::ldmatrix_x4(b0, b1, b2, b3, &Bs[b_col0][k0 + b_koff]);
            ah::mma_16x8x16<H>(acc[0], a0, a1, a2, a3, b0, b1);
            ah::mma_16x8x16<H>(acc[1], a0, a1, a2, a3, b2, b3);
            ah::ldmatrix_x4(b0, b1, b2, b3, &Bs[b_col0 + 16][k0 + b_koff]);
            ah::mma_16x8x16<H>(acc[2], a0, a1, a2, a3, b0, b1);
            ah::mma_16x8x16<H>(acc[3], a0, a1, a2, a3, b2, b3);
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c0 = 32 * wn + 8 * j + 2 * tl;
            *reinterpret_cast<uint32_t*>(&As[r0][c0]) = ah::pack2<H>(acc[j][0], acc[j][1]);
            *reinterpret_cast<uint32_t*>(&As[r1][c0]) = ah::pack2<H>(acc[j][2], acc[j][3]);
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + 256 * q, row = idx >> 4, c4 = (idx & 15) * 4;
            const long rm = row_map[row];
            if constexpr (ZIP == 5 || ZIP == 6) {
                const bool valid = (rm >= 0);
                uint2 g4 = make_uint2(0u, 0u);
                uint2 v = make_uint2(0u, 0u);
                if (valid) {
                    v = *reinterpret_cast<const uint2*>(&As[row][c4]);
                    if (beta != 0.f) {
                        const uint2 o = *reinterpret_cast<const uint2*>(C + rm * K + c4);
                        const float2 vx = ah::unpack2<H>(v.x), ox = ah::unpack2<H>(o.x);
                        const float2 vy = ah::unpack2<H>(v.y), oy = ah::unpack2<H>(o.y);
                        v.x = ah::pack2<H>(fmaf(beta, ox.x, vx.x), fmaf(beta, ox.y, vx.y));
                        v.y = ah::pack2<H>(fmaf(beta, oy.x, vy.x), fmaf(beta, oy.y, vy.y));
                    }
                    g4 = *reinterpret_cast<const uint2*>(G_zip + rm * K + c4);
                }
                const float2 g01 = ah::unpack2<H>(g4.x), g23 = ah::unpack2<H>(g4.y);
                const float2 a01 = ah::unpack2<H>(v.x), a23 = ah::unpack2<H>(v.y);
                float mg = fmaxf(fmaxf(fabsf(g01.x), fabsf(g01.y)), fmaxf(fabsf(g23.x), fabsf(g23.y)));
                float ma = fmaxf(fmaxf(fabsf(a01.x), fabsf(a01.y)), fmaxf(fabsf(a23.x), fabsf(a23.y)));
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) {
                    mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, o));
                    ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, o));
                }
                if (valid) {
                    constexpr float QMAX = (ZIP == 5) ? 448.f : 127.f;
                    const float sg = mg * (1.f / QMAX), sa = ma * (1.f / QMAX);
                    const float ig = (mg > 0.f) ? (QMAX / mg) : 0.f;
                    const float ia = (ma > 0.f) ? (QMAX / ma) : 0.f;
                    uint16_t p0, p1, p2, p3;
                    if constexpr (ZIP == 5) {
                        p0 = ah::pack_ga_fp8(g01.x * ig, a01.x * ia);
                        p1 = ah::pack_ga_fp8(g01.y * ig, a01.y * ia);
                        p2 = ah::pack_ga_fp8(g23.x * ig, a23.x * ia);
                        p3 = ah::pack_ga_fp8(g23.y * ig, a23.y * ia);
                    } else {
                        p0 = ah::pack_ga_i8(g01.x * ig, a01.x * ia);
                        p1 = ah::pack_ga_i8(g01.y * ig, a01.y * ia);
                        p2 = ah::pack_ga_i8(g23.x * ig, a23.x * ia);
                        p3 = ah::pack_ga_i8(g23.y * ig, a23.y * ia);
                    }
                    const uint2 out4 = make_uint2(static_cast<uint32_t>(p0) | (static_cast<uint32_t>(p1) << 16),
                                                  static_cast<uint32_t>(p2) | (static_cast<uint32_t>(p3) << 16));
                    uint16_t* ga8 = reinterpret_cast<uint16_t*>(GA_zip);
                    *reinterpret_cast<uint2*>(ga8 + rm * K + c4) = out4;
                    if (c4 == 0) {
                        float2* ga_sc = reinterpret_cast<float2*>(GA_zip + 8L * N * K);
                        ga_sc[rm] = make_float2(sg, sa);
                    }
                }
            } else {
                if (rm < 0) continue;
                uint2 v = *reinterpret_cast<const uint2*>(&As[row][c4]);
                if (beta != 0.f) {
                    const uint2 o = *reinterpret_cast<const uint2*>(C + rm * K + c4);
                    const float2 vx = ah::unpack2<H>(v.x), ox = ah::unpack2<H>(o.x);
                    const float2 vy = ah::unpack2<H>(v.y), oy = ah::unpack2<H>(o.y);
                    v.x = ah::pack2<H>(fmaf(beta, ox.x, vx.x), fmaf(beta, ox.y, vx.y));
                    v.y = ah::pack2<H>(fmaf(beta, oy.x, vy.x), fmaf(beta, oy.y, vy.y));
                }
                if constexpr (ZIP == 1) {
                    const uint2 g4 = *reinterpret_cast<const uint2*>(G_zip + rm * K + c4);
                    uint4 ga;
                    ga.x = __byte_perm(g4.x, v.x, 0x5410);
                    ga.y = __byte_perm(g4.x, v.x, 0x7632);
                    ga.z = __byte_perm(g4.y, v.y, 0x5410);
                    ga.w = __byte_perm(g4.y, v.y, 0x7632);
                    *reinterpret_cast<uint4*>(GA_zip + rm * K + c4) = ga;
                } else if constexpr (ZIP == 4) {
                    const uint2 g4 = *reinterpret_cast<const uint2*>(G_zip + rm * K + c4);
                    const long n = rm >> 4, m = rm & 15;
                    const int l0 = c4 >> 1;
                    uint2* dst2 = reinterpret_cast<uint2*>(GA_zip);
                    const long base = ((n * 8 + (m >> 1)) * 32 + l0) * 2 + (m & 1);
                    dst2[base] = make_uint2(g4.x, v.x);
                    dst2[base + 2] = make_uint2(g4.y, v.y);
                } else {
                    *reinterpret_cast<uint2*>(C + rm * K + c4) = v;
                }
            }
        }
    }
}

template <typename H>
__global__ void __launch_bounds__(256, 4) k_seg_wgrad_spc_dyn_h_v2(const H* __restrict__ A, const H* __restrict__ B,
                                                                   float* __restrict__ M0s, long wstride, int ldw,
                                                                   const int* __restrict__ spc_off,
                                                                   const int* __restrict__ spc_idx,
                                                                   const int* __restrict__ spc_wsk, int S) {
    constexpr int C = 64, TK64 = 64, LDS = 72;
    const int tot_sk = spc_wsk[S];
    const int bx = blockIdx.x;
    if (bx >= tot_sk) return;
    int s = 0;
    while (s + 1 < S && bx >= spc_wsk[s + 1]) ++s;
    const int sk = bx - spc_wsk[s], SK = spc_wsk[s + 1] - spc_wsk[s];
    const int by = blockIdx.y;  // 0..15
    const int l = by == 0 ? 0 : by < 4 ? 1 : by < 9 ? 2 : 3;
    const int deg = 2 * l + 1;
    const int sub = by - l * l;
    const int part = sk * deg + sub, TOT = SK * deg;
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int nr = cnt * deg;
    const int r_beg = (nr * part) / TOT, r_end = (nr * (part + 1)) / TOT;
    if (r_beg >= r_end) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;

    __shared__ __align__(16) H As[TK64][LDS];  // row-major [kk][c] -> 100% coalesced uint2 STS.64!
    __shared__ __align__(16) H Bs[TK64][LDS];  // row-major [kk][d]
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

    const int a_koff = ((lane >> 4) * 8) + (lane & 7);
    const int a_col = 16 * wm + (((lane >> 3) & 1) * 8);
    const int b_koff = (((lane >> 3) & 1) * 8) + (lane & 7);
    const int b_col0 = 32 * wn + ((lane >> 4) * 8);

    for (int rb = r_beg; rb < r_end; rb += TK64) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + 256 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
            const int r = rb + kk;
            uint2 va = make_uint2(0u, 0u), vb = make_uint2(0u, 0u);
            if (r < r_end) {
                const int j = r / deg, dm = r - j * deg;
                const long rm = (long)spc_idx[off + j] * 16 + (l * l + dm);
                va = *reinterpret_cast<const uint2*>(A + rm * C + col4);
                vb = *reinterpret_cast<const uint2*>(B + rm * C + col4);
            }
            *reinterpret_cast<uint2*>(&As[kk][col4]) = va;
            *reinterpret_cast<uint2*>(&Bs[kk][col4]) = vb;
        }
        __syncthreads();
#pragma unroll
        for (int k0 = 0; k0 < TK64; k0 += 16) {
            uint32_t a0, a1, a2, a3;
            ah::ldmatrix_x4_trans(a0, a1, a2, a3, &As[k0 + a_koff][a_col]);
            uint32_t b0, b1, b2, b3;
            ah::ldmatrix_x4_trans(b0, b1, b2, b3, &Bs[k0 + b_koff][b_col0]);
            ah::mma_16x8x16<H>(acc[0], a0, a1, a2, a3, b0, b1);
            ah::mma_16x8x16<H>(acc[1], a0, a1, a2, a3, b2, b3);
            ah::ldmatrix_x4_trans(b0, b1, b2, b3, &Bs[k0 + b_koff][b_col0 + 16]);
            ah::mma_16x8x16<H>(acc[2], a0, a1, a2, a3, b0, b1);
            ah::mma_16x8x16<H>(acc[3], a0, a1, a2, a3, b2, b3);
        }
        __syncthreads();
    }
    float* out = M0s + (long)l * wstride + (long)s * C;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = 32 * wn + 8 * j + 2 * tl;
        atomicAdd(out + (long)r0 * ldw + c0, acc[j][0]);
        atomicAdd(out + (long)r0 * ldw + c0 + 1, acc[j][1]);
        atomicAdd(out + (long)r1 * ldw + c0, acc[j][2]);
        atomicAdd(out + (long)r1 * ldw + c0 + 1, acc[j][3]);
    }
}

template <typename H>
__global__ void __launch_bounds__(256, 4) k_seg_wgrad_wl1_h_v2(const H* __restrict__ A, const H* __restrict__ B,
                                                               float* __restrict__ Wg, long wstride, int N) {
    constexpr int C = 64, TK64 = 64, LDS = 72;
    const int sk = blockIdx.x, SK = gridDim.x;
    const int by = blockIdx.y;  // 0..15
    const int l = by == 0 ? 0 : by < 4 ? 1 : by < 9 ? 2 : 3;
    const int deg = 2 * l + 1;
    const int sub = by - l * l;
    const int part = sk * deg + sub, TOT = SK * deg;
    const int nr = N * deg;
    const int r_beg = (nr * part) / TOT, r_end = (nr * (part + 1)) / TOT;
    if (r_beg >= r_end) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, tl = lane & 3;
    const int r0 = 16 * wm + g, r1 = r0 + 8;

    __shared__ __align__(16) H As[TK64][LDS];
    __shared__ __align__(16) H Bs[TK64][LDS];
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

    const int a_koff = ((lane >> 4) * 8) + (lane & 7);
    const int a_col = 16 * wm + (((lane >> 3) & 1) * 8);
    const int b_koff = (((lane >> 3) & 1) * 8) + (lane & 7);
    const int b_col0 = 32 * wn + ((lane >> 4) * 8);

    for (int rb = r_beg; rb < r_end; rb += TK64) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + 256 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
            const int r = rb + kk;
            uint2 va = make_uint2(0u, 0u), vb = make_uint2(0u, 0u);
            if (r < r_end) {
                const int n = r / deg, dm = r - n * deg;
                const long rm = (long)n * 16 + (l * l + dm);
                va = *reinterpret_cast<const uint2*>(A + rm * C + col4);
                vb = *reinterpret_cast<const uint2*>(B + rm * C + col4);
            }
            *reinterpret_cast<uint2*>(&As[kk][col4]) = va;
            *reinterpret_cast<uint2*>(&Bs[kk][col4]) = vb;
        }
        __syncthreads();
#pragma unroll
        for (int k0 = 0; k0 < TK64; k0 += 16) {
            uint32_t a0, a1, a2, a3;
            ah::ldmatrix_x4_trans(a0, a1, a2, a3, &As[k0 + a_koff][a_col]);
            uint32_t b0, b1, b2, b3;
            ah::ldmatrix_x4_trans(b0, b1, b2, b3, &Bs[k0 + b_koff][b_col0]);
            ah::mma_16x8x16<H>(acc[0], a0, a1, a2, a3, b0, b1);
            ah::mma_16x8x16<H>(acc[1], a0, a1, a2, a3, b2, b3);
            ah::ldmatrix_x4_trans(b0, b1, b2, b3, &Bs[k0 + b_koff][b_col0 + 16]);
            ah::mma_16x8x16<H>(acc[2], a0, a1, a2, a3, b0, b1);
            ah::mma_16x8x16<H>(acc[3], a0, a1, a2, a3, b2, b3);
        }
        __syncthreads();
    }
    float* out = Wg + (long)l * wstride;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c0 = 32 * wn + 8 * j + 2 * tl;
        atomicAdd(out + (long)r0 * C + c0, acc[j][0]);
        atomicAdd(out + (long)r0 * C + c0 + 1, acc[j][1]);
        atomicAdd(out + (long)r1 * C + c0, acc[j][2]);
        atomicAdd(out + (long)r1 * C + c0 + 1, acc[j][3]);
    }
}

// Wave 27: Per-(node, degree l) FP8 E4M3 packer for GA_buf (QMAX=32.f -> safe for direct __half2 HFMA2!).
// Grid: N blocks, 64 threads (1 thread per channel c). Single-pass coalesced 128B reads/writes (~5 us at b4!).
// Output: uint16_t GA8[N, 16, 64] at GA_buf[0..8*N*64), float2 GAs[N, 4] at GA_buf + 8L * N * 64.
template <typename H>
__global__ void __launch_bounds__(64, 8) k_ga_pack_fp8_l(const H* __restrict__ G, const H* __restrict__ A,
                                                           uint32_t* __restrict__ GA_buf, int N) {
    const int n = blockIdx.x;
    if (n >= N) return;
    const int c = threadIdx.x;
    const int warp = c >> 5, lane = c & 31;
    const long nbase = (long)n * 16 * 64 + c;
    float gv[16], av[16];
    float mg[4] = {0.f, 0.f, 0.f, 0.f};
    float ma[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const float g = ah::to_f<H>(G[nbase + (long)m * 64]);
        const float a = ah::to_f<H>(A[nbase + (long)m * 64]);
        gv[m] = g;
        av[m] = a;
        const int l = (m == 0) ? 0 : (m < 4 ? 1 : (m < 9 ? 2 : 3));
        mg[l] = fmaxf(mg[l], fabsf(g));
        ma[l] = fmaxf(ma[l], fabsf(a));
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            mg[l] = fmaxf(mg[l], __shfl_xor_sync(0xffffffffu, mg[l], o));
            ma[l] = fmaxf(ma[l], __shfl_xor_sync(0xffffffffu, ma[l], o));
        }
    }
    __shared__ float sm[2][8];
    if (lane == 0) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            sm[warp][l] = mg[l];
            sm[warp][4 + l] = ma[l];
        }
    }
    __syncthreads();
    constexpr float QMAX = 32.f;
    float ig[4], ia[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const float gmax = fmaxf(sm[0][l], sm[1][l]);
        const float amax = fmaxf(sm[0][4 + l], sm[1][4 + l]);
        ig[l] = (gmax > 0.f) ? (QMAX / gmax) : 0.f;
        ia[l] = (amax > 0.f) ? (QMAX / amax) : 0.f;
        if (c == l) {
            float2* GAs = reinterpret_cast<float2*>(GA_buf + 8L * N * 64);
            GAs[(long)n * 4 + l] = make_float2(gmax * (1.f / QMAX), amax * (1.f / QMAX));
        }
    }
    uint16_t* GA8 = reinterpret_cast<uint16_t*>(GA_buf);
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const int l = (m == 0) ? 0 : (m < 4 ? 1 : (m < 9 ? 2 : 3));
        GA8[nbase + (long)m * 64] = ah::pack_ga_fp8(gv[m] * ig[l], av[m] * ia[l]);
    }
}

}  // namespace segg
}  // namespace lst
