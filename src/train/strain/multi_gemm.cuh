// Several small independent row-major GEMMs in one launch: C_i[M, N] = op(A_i) op(B_i) (beta 0), op(A) = A^T
// when ta (A stored K x M), op(B) = B^T when tb (B stored N x K). 32 x 32 output tiles, 256 threads with 4 outputs
// each; a block finds its GEMM from the tiles' prefix (tile0). For the weight-only products of prepare_views and
// of the view gradients (at most a few hundred columns, K <= a few hundred).
#pragma once
#include <cstdint>
#include <type_traits>

namespace lst {
namespace mgemm {

constexpr int TS = 32, MAXG = 16;

template <typename T>
struct Desc {
    const T* A;
    const T* B;
    T* C;
    int M, N, K, lda, ldb, ldc, ta, tb, tile0, sk;
};

template <typename T>
struct Table {
    Desc<T> d[MAXG];
    int n;
};

template <typename T>
__global__ void __launch_bounds__(256) k_multi_gemm(const Table<T>* __restrict__ tab) {
    __shared__ T As[TS][TS + 1];
    __shared__ T Bs[TS][TS + 1];
    const int ng = tab->n;
    int g = 0;
    while (g + 1 < ng && (int)blockIdx.x >= tab->d[g + 1].tile0) ++g;
    const Desc<T> q = tab->d[g];
    const int t = blockIdx.x - q.tile0, ntn = (q.N + TS - 1) / TS;
    const int m0 = (t / ntn) * TS, n0 = (t % ntn) * TS;
    const int tx = threadIdx.x % TS, ty = threadIdx.x / TS;   // ty 0..7: rows ty + 8 i
    T acc[4] = {T(0), T(0), T(0), T(0)};
    for (int k0 = 0; k0 < q.K; k0 += TS) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = ty + 8 * i;           // tile row (m or k)
            {   // As[m][k]
                const int m = m0 + r, k = k0 + tx;
                T v = T(0);
                if (m < q.M && k < q.K) v = q.ta ? q.A[(long)k * q.lda + m] : q.A[(long)m * q.lda + k];
                As[r][tx] = v;
            }
            {   // Bs[k][n]
                const int k = k0 + r, n = n0 + tx;
                T v = T(0);
                if (k < q.K && n < q.N) v = q.tb ? q.B[(long)n * q.ldb + k] : q.B[(long)k * q.ldb + n];
                Bs[r][tx] = v;
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < TS; ++k) {
            const T b = Bs[k][tx];
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] += As[ty + 8 * i][k] * b;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty + 8 * i, n = n0 + tx;
        if (m < q.M && n < q.N) q.C[(long)m * q.ldc + n] = acc[i];
    }
}

// Fast 64-thread (4x4 per thread) multi-GEMM over 32x32 tiles (LST_FAST_MGEMM=1): 4x lower smem/FMA ratio and 4x more
// CTAs per SM than the 256-thread 4x1 kernel above.
template <typename T>
__global__ void __launch_bounds__(64) k_multi_gemm_f(const Table<T>* __restrict__ tab) {
    __shared__ __align__(16) T As[TS][TS + 4];
    __shared__ __align__(16) T Bs[TS][TS + 4];
    const int ng = tab->n;
    int g = 0;
    while (g + 1 < ng && (int)blockIdx.x >= tab->d[g + 1].tile0) ++g;
    const Desc<T> q = tab->d[g];
    const int t = blockIdx.x - q.tile0, ntn = (q.N + TS - 1) / TS;
    const int m0 = (t / ntn) * TS, n0 = (t % ntn) * TS;
    const int tid = threadIdx.x, tx = tid & 7, ty = tid >> 3;
    const int r_ld = tid >> 3, c_ld = (tid & 7) * 4;
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    for (int k0 = 0; k0 < q.K; k0 += TS) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = r_ld + 8 * i;
            if (!q.ta) {
                const int m = m0 + row;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int k = k0 + c_ld + u;
                    As[c_ld + u][row] = (m < q.M && k < q.K) ? q.A[(long)m * q.lda + k] : T(0);
                }
            } else {
                const int k = k0 + row;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int m = m0 + c_ld + u;
                    As[row][c_ld + u] = (m < q.M && k < q.K) ? q.A[(long)k * q.lda + m] : T(0);
                }
            }
            if (!q.tb) {
                const int k = k0 + row;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int n = n0 + c_ld + u;
                    Bs[row][c_ld + u] = (k < q.K && n < q.N) ? q.B[(long)k * q.ldb + n] : T(0);
                }
            } else {
                const int n = n0 + row;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int k = k0 + c_ld + u;
                    Bs[c_ld + u][row] = (k < q.K && n < q.N) ? q.B[(long)n * q.ldb + k] : T(0);
                }
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < TS; ++k) {
            T av[4], bv[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 a = *reinterpret_cast<const float4*>(&As[k][ty * 4]);
                const float4 b = *reinterpret_cast<const float4*>(&Bs[k][tx * 4]);
                av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                bv[0] = b.x; bv[1] = b.y; bv[2] = b.z; bv[3] = b.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    av[u] = As[k][ty * 4 + u];
                    bv[u] = Bs[k][tx * 4 + u];
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
        const int m = m0 + ty * 4 + i;
        if (m >= q.M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx * 4 + j;
            if (n < q.N) q.C[(long)m * q.ldc + n] = acc[i][j];
        }
    }
}

// Wave 14.2: 16x32-tiled 64-thread (2x4 per thread) multi-GEMM (LST_NOPS_OPT=1):
//   - Doubles tile count vs 32x32 (~134-200 blocks -> fills ALL 36/48 SMs!)
//   - float2 LDS.64 on As + float4 LDS.128 on Bs (4x lower MIO pressure than k_multi_gemm!)
//   - Only 8 accumulators (~32 regs/thread, 6.9 KB smem -> max occupancy!).
constexpr int TM2 = 16, TN2 = 32, TK2 = 32;

template <typename T>
__global__ void __launch_bounds__(64) k_multi_gemm_v2(const Table<T>* __restrict__ tab) {
    __shared__ __align__(16) T As[TK2][TM2 + 4];
    __shared__ __align__(16) T Bs[TK2][TN2 + 4];
    const int ng = tab->n;
    int g = 0;
    while (g + 1 < ng && (int)blockIdx.x >= tab->d[g + 1].tile0) ++g;
    const Desc<T> q = tab->d[g];
    const int t = blockIdx.x - q.tile0;
    const int ntm = (q.M + TM2 - 1) / TM2, ntn = (q.N + TN2 - 1) / TN2;
    const int sk = q.sk > 1 ? q.sk : 1;
    const int nmn = ntm * ntn;
    const int s = (sk > 1) ? (t / nmn) : 0;
    const int mn = (sk > 1) ? (t % nmn) : t;
    const int m0 = (mn / ntn) * TM2, n0 = (mn % ntn) * TN2;
    const int ntk = (q.K + TK2 - 1) / TK2;
    const int k_beg = (sk > 1) ? (((s * ntk) / sk) * TK2) : 0;
    const int k_end = (sk > 1 && s + 1 < sk) ? ((((s + 1) * ntk) / sk) * TK2) : q.K;
    const int tid = threadIdx.x, tx = tid & 7, ty = tid >> 3;  // ty in 0..7 (2 rows), tx in 0..7 (4 cols)
    T acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    for (int k0 = k_beg; k0 < k_end; k0 += TK2) {
        __syncthreads();
        // Load A [16 x 32] = 512 elems / 64 threads = 8 elems/thread (2 chunks of 4)
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int idx = tid + 64 * p;
            if (!q.ta) {
                const int r = idx >> 3, c4 = (idx & 7) * 4;  // r in 0..15, c4 in 0..28
                const int m = m0 + r;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int k = k0 + c4 + u;
                    As[c4 + u][r] = (m < q.M && k < k_end) ? q.A[(long)m * q.lda + k] : T(0);
                }
            } else {
                const int rk = idx >> 2, c4 = (idx & 3) * 4;  // rk in 0..31, c4 in 0..12
                const int k = k0 + rk;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int m = m0 + c4 + u;
                    As[rk][c4 + u] = (m < q.M && k < k_end) ? q.A[(long)k * q.lda + m] : T(0);
                }
            }
        }
        // Load B [32 x 32] = 1024 elems / 64 threads = 16 elems/thread (4 chunks of 4)
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int idx = tid + 64 * p, r = idx >> 3, c4 = (idx & 7) * 4;  // r in 0..31, c4 in 0..28
            if (!q.tb) {
                const int k = k0 + r;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int n = n0 + c4 + u;
                    Bs[r][c4 + u] = (k < k_end && n < q.N) ? q.B[(long)k * q.ldb + n] : T(0);
                }
            } else {
                const int n = n0 + r;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int k = k0 + c4 + u;
                    Bs[c4 + u][r] = (k < k_end && n < q.N) ? q.B[(long)n * q.ldb + k] : T(0);
                }
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < TK2; ++k) {
            T a0 = As[k][2 * ty], a1 = As[k][2 * ty + 1];
            T bv[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 b4 = *reinterpret_cast<const float4*>(&Bs[k][4 * tx]);
                bv[0] = b4.x; bv[1] = b4.y; bv[2] = b4.z; bv[3] = b4.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) bv[u] = Bs[k][4 * tx + u];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[0][j] += a0 * bv[j];
                acc[1][j] += a1 * bv[j];
            }
        }
    }
    if (sk > 1) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int m = m0 + 2 * ty + i;
            if (m >= q.M) continue;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int n = n0 + 4 * tx + j;
                if (n < q.N) atomicAdd(&q.C[(long)m * q.ldc + n], acc[i][j]);
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int m = m0 + 2 * ty + i;
            if (m >= q.M) continue;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int n = n0 + 4 * tx + j;
                if (n < q.N) q.C[(long)m * q.ldc + n] = acc[i][j];
            }
        }
    }
}

// Wave 31 (LST_MGEMM_V3=1): 16x32-tiled 64-thread (2x4) exact-FP32 multi-GEMM with:
//   - coalesced float4 LDG.128 when aligned & in-bounds, warp-contiguous 32-wide fallback for odd strides (ZW=257)
//   - As[32][18] (8B aligned float2 LDS.64) + Bs[32][36] (16B aligned float4 LDS.128 & STS.128)
//   - exact klen tail (skips 90% of zero-padded FMAs on K=3 gwup0 and K=257 tail!)
//   - float4 STG.128 C stores when aligned & in-bounds
template <typename T>
__global__ void __launch_bounds__(64) k_multi_gemm_v3(const Table<T>* __restrict__ tab) {
    __shared__ __align__(16) T As[TK2][TM2 + 2];
    __shared__ __align__(16) T Bs[TK2][TN2 + 4];
    const int ng = tab->n;
    int g = 0;
    while (g + 1 < ng && (int)blockIdx.x >= tab->d[g + 1].tile0) ++g;
    const Desc<T> q = tab->d[g];
    const int t = blockIdx.x - q.tile0, ntn = (q.N + TN2 - 1) / TN2;
    const int m0 = (t / ntn) * TM2, n0 = (t % ntn) * TN2;
    const int tid = threadIdx.x, tx = tid & 7, ty = tid >> 3;
    const bool a_f4 = std::is_same<T, float>::value && ((q.lda & 3) == 0) &&
                      ((reinterpret_cast<uintptr_t>(q.A) & 15u) == 0);
    const bool b_f4 = std::is_same<T, float>::value && ((q.ldb & 3) == 0) &&
                      ((reinterpret_cast<uintptr_t>(q.B) & 15u) == 0);
    const bool c_f4 = std::is_same<T, float>::value && ((q.ldc & 3) == 0) &&
                      ((reinterpret_cast<uintptr_t>(q.C) & 15u) == 0);
    T acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);

    for (int k0 = 0; k0 < q.K; k0 += TK2) {
        const int klen = (k0 + TK2 <= q.K) ? TK2 : (q.K - k0);
        __syncthreads();
        // Load A [16 x 32]
        if (!q.ta) {
            if constexpr (std::is_same<T, float>::value) {
                if (a_f4 && klen == TK2) {
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 3, c4 = (idx & 7) * 4;
                        const int m = m0 + r;
                        const float4 v = (m < q.M) ? *reinterpret_cast<const float4*>(q.A + (long)m * q.lda + k0 + c4)
                                                   : make_float4(0.f, 0.f, 0.f, 0.f);
                        As[c4 + 0][r] = v.x;
                        As[c4 + 1][r] = v.y;
                        As[c4 + 2][r] = v.z;
                        As[c4 + 3][r] = v.w;
                    }
                } else {
#pragma unroll
                    for (int p = 0; p < 8; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                        const int m = m0 + r, k = k0 + c;
                        As[c][r] = (m < q.M && k < q.K) ? q.A[(long)m * q.lda + k] : T(0);
                    }
                }
            } else {
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                    const int m = m0 + r, k = k0 + c;
                    As[c][r] = (m < q.M && k < q.K) ? q.A[(long)m * q.lda + k] : T(0);
                }
            }
        } else {
            if constexpr (std::is_same<T, float>::value) {
                if (a_f4 && m0 + TM2 <= q.M) {
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        const int idx = tid + 64 * p, rk = idx >> 2, c4 = (idx & 3) * 4;
                        const int k = k0 + rk;
                        const float4 v = (k < q.K) ? *reinterpret_cast<const float4*>(q.A + (long)k * q.lda + m0 + c4)
                                                   : make_float4(0.f, 0.f, 0.f, 0.f);
                        As[rk][c4 + 0] = v.x;
                        As[rk][c4 + 1] = v.y;
                        As[rk][c4 + 2] = v.z;
                        As[rk][c4 + 3] = v.w;
                    }
                } else {
#pragma unroll
                    for (int p = 0; p < 8; ++p) {
                        const int idx = tid + 64 * p, rk = idx >> 4, cm = idx & 15;
                        const int k = k0 + rk, m = m0 + cm;
                        As[rk][cm] = (k < q.K && m < q.M) ? q.A[(long)k * q.lda + m] : T(0);
                    }
                }
            } else {
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    const int idx = tid + 64 * p, rk = idx >> 4, cm = idx & 15;
                    const int k = k0 + rk, m = m0 + cm;
                    As[rk][cm] = (k < q.K && m < q.M) ? q.A[(long)k * q.lda + m] : T(0);
                }
            }
        }
        // Load B [32 x 32]
        if (!q.tb) {
            if constexpr (std::is_same<T, float>::value) {
                if (b_f4 && n0 + TN2 <= q.N) {
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 3, c4 = (idx & 7) * 4;
                        const int k = k0 + r;
                        const float4 v = (k < q.K) ? *reinterpret_cast<const float4*>(q.B + (long)k * q.ldb + n0 + c4)
                                                   : make_float4(0.f, 0.f, 0.f, 0.f);
                        *reinterpret_cast<float4*>(&Bs[r][c4]) = v;
                    }
                } else {
#pragma unroll
                    for (int p = 0; p < 16; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                        const int k = k0 + r, n = n0 + c;
                        Bs[r][c] = (k < q.K && n < q.N) ? q.B[(long)k * q.ldb + n] : T(0);
                    }
                }
            } else {
#pragma unroll
                for (int p = 0; p < 16; ++p) {
                    const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                    const int k = k0 + r, n = n0 + c;
                    Bs[r][c] = (k < q.K && n < q.N) ? q.B[(long)k * q.ldb + n] : T(0);
                }
            }
        } else {
            if constexpr (std::is_same<T, float>::value) {
                if (b_f4 && klen == TK2) {
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 3, c4 = (idx & 7) * 4;
                        const int n = n0 + r;
                        const float4 v = (n < q.N) ? *reinterpret_cast<const float4*>(q.B + (long)n * q.ldb + k0 + c4)
                                                   : make_float4(0.f, 0.f, 0.f, 0.f);
                        Bs[c4 + 0][r] = v.x;
                        Bs[c4 + 1][r] = v.y;
                        Bs[c4 + 2][r] = v.z;
                        Bs[c4 + 3][r] = v.w;
                    }
                } else {
#pragma unroll
                    for (int p = 0; p < 16; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                        const int n = n0 + r, k = k0 + c;
                        Bs[c][r] = (n < q.N && k < q.K) ? q.B[(long)n * q.ldb + k] : T(0);
                    }
                }
            } else {
#pragma unroll
                for (int p = 0; p < 16; ++p) {
                    const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                    const int n = n0 + r, k = k0 + c;
                    Bs[c][r] = (n < q.N && k < q.K) ? q.B[(long)n * q.ldb + k] : T(0);
                }
            }
        }
        __syncthreads();
        auto fma_step = [&](int k) {
            T a0, a1, bv[4];
            if constexpr (std::is_same<T, float>::value) {
                const float2 a2 = *reinterpret_cast<const float2*>(&As[k][2 * ty]);
                const float4 b4 = *reinterpret_cast<const float4*>(&Bs[k][4 * tx]);
                a0 = a2.x; a1 = a2.y;
                bv[0] = b4.x; bv[1] = b4.y; bv[2] = b4.z; bv[3] = b4.w;
            } else {
                a0 = As[k][2 * ty]; a1 = As[k][2 * ty + 1];
#pragma unroll
                for (int u = 0; u < 4; ++u) bv[u] = Bs[k][4 * tx + u];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[0][j] += a0 * bv[j];
                acc[1][j] += a1 * bv[j];
            }
        };
        if (klen == TK2) {
#pragma unroll 8
            for (int k = 0; k < TK2; ++k) fma_step(k);
        } else {
            for (int k = 0; k < klen; ++k) fma_step(k);
        }
    }
    if constexpr (std::is_same<T, float>::value) {
        if (c_f4 && n0 + 4 * tx + 4 <= q.N) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int m = m0 + 2 * ty + i;
                if (m < q.M)
                    *reinterpret_cast<float4*>(q.C + (long)m * q.ldc + n0 + 4 * tx) =
                        make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            }
            return;
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int m = m0 + 2 * ty + i;
        if (m >= q.M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + 4 * tx + j;
            if (n < q.N) q.C[(long)m * q.ldc + n] = acc[i][j];
        }
    }
}

// Wave 31 (LST_MGEMM_V3=2): 32x32-tiled 64-thread (4x4) exact-FP32 multi-GEMM with coalesced float4 LDG/LDS/STG.
template <typename T>
__global__ void __launch_bounds__(64) k_multi_gemm_f4(const Table<T>* __restrict__ tab) {
    __shared__ __align__(16) T As[TS][TS + 4];
    __shared__ __align__(16) T Bs[TS][TS + 4];
    const int ng = tab->n;
    int g = 0;
    while (g + 1 < ng && (int)blockIdx.x >= tab->d[g + 1].tile0) ++g;
    const Desc<T> q = tab->d[g];
    const int t = blockIdx.x - q.tile0, ntn = (q.N + TS - 1) / TS;
    const int m0 = (t / ntn) * TS, n0 = (t % ntn) * TS;
    const int tid = threadIdx.x, tx = tid & 7, ty = tid >> 3;
    const bool a_f4 = std::is_same<T, float>::value && ((q.lda & 3) == 0) &&
                      ((reinterpret_cast<uintptr_t>(q.A) & 15u) == 0);
    const bool b_f4 = std::is_same<T, float>::value && ((q.ldb & 3) == 0) &&
                      ((reinterpret_cast<uintptr_t>(q.B) & 15u) == 0);
    const bool c_f4 = std::is_same<T, float>::value && ((q.ldc & 3) == 0) &&
                      ((reinterpret_cast<uintptr_t>(q.C) & 15u) == 0);
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);

    for (int k0 = 0; k0 < q.K; k0 += TS) {
        const int klen = (k0 + TS <= q.K) ? TS : (q.K - k0);
        __syncthreads();
        // Load A [32 x 32]
        if (!q.ta) {
            if constexpr (std::is_same<T, float>::value) {
                if (a_f4 && klen == TS) {
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 3, c4 = (idx & 7) * 4;
                        const int m = m0 + r;
                        const float4 v = (m < q.M) ? *reinterpret_cast<const float4*>(q.A + (long)m * q.lda + k0 + c4)
                                                   : make_float4(0.f, 0.f, 0.f, 0.f);
                        As[c4 + 0][r] = v.x;
                        As[c4 + 1][r] = v.y;
                        As[c4 + 2][r] = v.z;
                        As[c4 + 3][r] = v.w;
                    }
                } else {
#pragma unroll
                    for (int p = 0; p < 16; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                        const int m = m0 + r, k = k0 + c;
                        As[c][r] = (m < q.M && k < q.K) ? q.A[(long)m * q.lda + k] : T(0);
                    }
                }
            } else {
#pragma unroll
                for (int p = 0; p < 16; ++p) {
                    const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                    const int m = m0 + r, k = k0 + c;
                    As[c][r] = (m < q.M && k < q.K) ? q.A[(long)m * q.lda + k] : T(0);
                }
            }
        } else {
            if constexpr (std::is_same<T, float>::value) {
                if (a_f4 && m0 + TS <= q.M) {
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 3, c4 = (idx & 7) * 4;
                        const int k = k0 + r;
                        const float4 v = (k < q.K) ? *reinterpret_cast<const float4*>(q.A + (long)k * q.lda + m0 + c4)
                                                   : make_float4(0.f, 0.f, 0.f, 0.f);
                        *reinterpret_cast<float4*>(&As[r][c4]) = v;
                    }
                } else {
#pragma unroll
                    for (int p = 0; p < 16; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                        const int k = k0 + r, m = m0 + c;
                        As[r][c] = (k < q.K && m < q.M) ? q.A[(long)k * q.lda + m] : T(0);
                    }
                }
            } else {
#pragma unroll
                for (int p = 0; p < 16; ++p) {
                    const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                    const int k = k0 + r, m = m0 + c;
                    As[r][c] = (k < q.K && m < q.M) ? q.A[(long)k * q.lda + m] : T(0);
                }
            }
        }
        // Load B [32 x 32]
        if (!q.tb) {
            if constexpr (std::is_same<T, float>::value) {
                if (b_f4 && n0 + TS <= q.N) {
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 3, c4 = (idx & 7) * 4;
                        const int k = k0 + r;
                        const float4 v = (k < q.K) ? *reinterpret_cast<const float4*>(q.B + (long)k * q.ldb + n0 + c4)
                                                   : make_float4(0.f, 0.f, 0.f, 0.f);
                        *reinterpret_cast<float4*>(&Bs[r][c4]) = v;
                    }
                } else {
#pragma unroll
                    for (int p = 0; p < 16; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                        const int k = k0 + r, n = n0 + c;
                        Bs[r][c] = (k < q.K && n < q.N) ? q.B[(long)k * q.ldb + n] : T(0);
                    }
                }
            } else {
#pragma unroll
                for (int p = 0; p < 16; ++p) {
                    const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                    const int k = k0 + r, n = n0 + c;
                    Bs[r][c] = (k < q.K && n < q.N) ? q.B[(long)k * q.ldb + n] : T(0);
                }
            }
        } else {
            if constexpr (std::is_same<T, float>::value) {
                if (b_f4 && klen == TS) {
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 3, c4 = (idx & 7) * 4;
                        const int n = n0 + r;
                        const float4 v = (n < q.N) ? *reinterpret_cast<const float4*>(q.B + (long)n * q.ldb + k0 + c4)
                                                   : make_float4(0.f, 0.f, 0.f, 0.f);
                        Bs[c4 + 0][r] = v.x;
                        Bs[c4 + 1][r] = v.y;
                        Bs[c4 + 2][r] = v.z;
                        Bs[c4 + 3][r] = v.w;
                    }
                } else {
#pragma unroll
                    for (int p = 0; p < 16; ++p) {
                        const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                        const int n = n0 + r, k = k0 + c;
                        Bs[c][r] = (n < q.N && k < q.K) ? q.B[(long)n * q.ldb + k] : T(0);
                    }
                }
            } else {
#pragma unroll
                for (int p = 0; p < 16; ++p) {
                    const int idx = tid + 64 * p, r = idx >> 5, c = idx & 31;
                    const int n = n0 + r, k = k0 + c;
                    Bs[c][r] = (n < q.N && k < q.K) ? q.B[(long)n * q.ldb + k] : T(0);
                }
            }
        }
        __syncthreads();
        auto fma_step = [&](int k) {
            T av[4], bv[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 a4 = *reinterpret_cast<const float4*>(&As[k][4 * ty]);
                const float4 b4 = *reinterpret_cast<const float4*>(&Bs[k][4 * tx]);
                av[0] = a4.x; av[1] = a4.y; av[2] = a4.z; av[3] = a4.w;
                bv[0] = b4.x; bv[1] = b4.y; bv[2] = b4.z; bv[3] = b4.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    av[u] = As[k][4 * ty + u];
                    bv[u] = Bs[k][4 * tx + u];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        };
        if (klen == TS) {
#pragma unroll 8
            for (int k = 0; k < TS; ++k) fma_step(k);
        } else {
            for (int k = 0; k < klen; ++k) fma_step(k);
        }
    }
    if constexpr (std::is_same<T, float>::value) {
        if (c_f4 && n0 + 4 * tx + 4 <= q.N) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = m0 + 4 * ty + i;
                if (m < q.M)
                    *reinterpret_cast<float4*>(q.C + (long)m * q.ldc + n0 + 4 * tx) =
                        make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            }
            return;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + 4 * ty + i;
        if (m >= q.M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + 4 * tx + j;
            if (n < q.N) q.C[(long)m * q.ldc + n] = acc[i][j];
        }
    }
}

}  // namespace mgemm
}  // namespace lst
