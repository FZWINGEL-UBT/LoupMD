// Compact species-aware Q / z projections and single-launch small node GEMMs / wgrads (LST_COMPACT_Z=1).
// Eliminates k_build_z, k_split_z, k_copy, [N, 257] z/zb/zs traffic, 2/3 of Q FLOPs, T0s/wro2s cuBLAS, and all 18
// splitKreduce_kernel launches per step. Works for both float and double (FP64 parity verified to ~1e-15).
#pragma once
#include <type_traits>

namespace lst {
namespace nops {

constexpr int RM = 32, CN = 64, TK = 32;   // 32 rows x 64 cols per block (128 threads: 4x4 per thread)

// Per-species 32-row tile prefix sums spc_tiles32[S + 1] (called right after k_spc_group or inline).
__global__ void k_spc_tiles32(const int* __restrict__ spc_off, int* __restrict__ spc_tiles32, int S) {
    if (threadIdx.x != 0) return;
    int t = 0;
    spc_tiles32[0] = 0;
    for (int s = 0; s < S; ++s) {
        const int cnt = spc_off[s + 1] - spc_off[s];
        t += (cnt + RM - 1) / RM;
        spc_tiles32[s + 1] = t;
    }
}

inline int spc_max_tiles32(int N, int S) { return (N + RM - 1) / RM + S; }

// Compact Q forward: replaces z = f0 @ Q + k_split_z.
//   x1[n, :] = f0[n, :] @ Q[:, 1 .. C]
//   sc[n, :] = f0[n, :] @ Q[:, 1 + C + s*C .. 1 + C + (s+1)*C - 1]
//   if z0 != nullptr: z0[n] = f0[n, :] . Q[:, 0]
// Grid: dim3(2 * (C / CN), spc_max_tiles32(N, S)), 128 threads (4x4 per thread: ty = tid/16 in 0..7, tx = tid%16 in 0..15).
template <typename T>
__global__ void __launch_bounds__(128) k_q_fwd(const T* __restrict__ f0, const T* __restrict__ Q,
                                               const int* __restrict__ spc_off, const int* __restrict__ spc_idx,
                                               const int* __restrict__ spc_tiles32, T* __restrict__ z0,
                                               T* __restrict__ x1, T* __restrict__ sc, int C, int ZW, int S) {
    const int tot = spc_tiles32[S];
    const int by = blockIdx.y;
    if (by >= tot) return;
    int s = 0;
    while (s + 1 < S && by >= spc_tiles32[s + 1]) ++s;
    const int t = by - spc_tiles32[s];
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int m0 = t * RM;
    const int ntc = C / CN;
    const int is_sc = blockIdx.x / ntc;
    const int n0 = (blockIdx.x % ntc) * CN;
    const int q_col0 = is_sc ? (1 + C + s * C + n0) : (1 + n0);
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    __shared__ int row_map[RM];
    if (tid < RM) row_map[tid] = (m0 + tid < cnt) ? spc_idx[off + m0 + tid] : -1;
    __shared__ __align__(16) T As[TK][RM + 4];
    __shared__ __align__(16) T Bs[TK][CN + 4];
    __shared__ T q0s[TK];
    const bool do_z0 = (z0 != nullptr) && (is_sc == 0) && (n0 == 0);
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    T zacc[4] = {T(0), T(0), T(0), T(0)};
    for (int k0 = 0; k0 < C; k0 += TK) {
        __syncthreads();
        // A: 32 rows x 32 k = 1024 elems / 128 threads = 8 elems/thread (2 float4s)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 128 * q, r = idx >> 3, kq = (idx & 7) * 4;
            const int n = row_map[r];
            if constexpr (std::is_same<T, float>::value) {
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (n >= 0) v = *reinterpret_cast<const float4*>(f0 + (long)n * C + k0 + kq);
                As[kq][r] = v.x; As[kq + 1][r] = v.y; As[kq + 2][r] = v.z; As[kq + 3][r] = v.w;
            } else {
                const T* ap = n >= 0 ? f0 + (long)n * C + k0 + kq : nullptr;
                for (int u = 0; u < 4; ++u) As[kq + u][r] = ap ? ap[u] : T(0);
            }
        }
        // B: 32 k x 64 n = 2048 elems / 128 threads = 16 elems/thread (row stride ZW is odd -> scalar loads)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + 128 * q, kk = idx >> 4, nq = (idx & 15) * 4;
            const T* qp = Q + (long)(k0 + kk) * ZW + q_col0 + nq;
#pragma unroll
            for (int u = 0; u < 4; ++u) Bs[kk][nq + u] = qp[u];
        }
        if (do_z0 && tid < TK) q0s[tid] = Q[(long)(k0 + tid) * ZW];
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
                for (int u = 0; u < 4; ++u) { av[u] = As[kk][ty * 4 + u]; bv[u] = Bs[kk][tx * 4 + u]; }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
            if (do_z0 && tx == 0) {
                const T q0v = q0s[kk];
#pragma unroll
                for (int i = 0; i < 4; ++i) zacc[i] += av[i] * q0v;
            }
        }
    }
    T* out = is_sc ? sc : x1;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int n = row_map[ty * 4 + i];
        if (n < 0) continue;
        T* op = out + (long)n * C + n0 + tx * 4;
        if constexpr (std::is_same<T, float>::value) {
            *reinterpret_cast<float4*>(op) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) op[j] = acc[i][j];
        }
        if (do_z0 && tx == 0) z0[n] = zacc[i];
    }
}

// Compact Q backward: replaces k_build_z + out = zb @ Q^T.
//   out[n, c] = (v0 ? v0[n] : c0) * Q[c, 0] + x[n, :] @ Q[c, 1..C]^T + b[n, :] @ Q[c, 1+C+s*C ..]^T
// Grid: dim3(C / CN, spc_max_tiles32(N, S)), 128 threads.
template <typename T>
__global__ void __launch_bounds__(128) k_q_bwd(const T* __restrict__ v0, T c0, const T* __restrict__ x,
                                               const T* __restrict__ b, const T* __restrict__ Q,
                                               const int* __restrict__ spc_off, const int* __restrict__ spc_idx,
                                               const int* __restrict__ spc_tiles32, T* __restrict__ out, int C, int ZW,
                                               int S) {
    const int tot = spc_tiles32[S];
    const int by = blockIdx.y;
    if (by >= tot) return;
    int s = 0;
    while (s + 1 < S && by >= spc_tiles32[s + 1]) ++s;
    const int t = by - spc_tiles32[s];
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int m0 = t * RM, n0 = blockIdx.x * CN;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    __shared__ int row_map[RM];
    if (tid < RM) row_map[tid] = (m0 + tid < cnt) ? spc_idx[off + m0 + tid] : -1;
    __shared__ __align__(16) T As[TK][RM + 4];
    __shared__ __align__(16) T Bs[TK][CN + 4];
    T acc[4][4];
    // Initialize with (v0 ? v0[n] : c0) * Q[n0 + c, 0]
    __shared__ T q0s[CN];
    if (tid < CN) q0s[tid] = Q[(long)(n0 + tid) * ZW];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int n = row_map[ty * 4 + i];
        const T vn = n >= 0 ? (v0 ? v0[n] : c0) : T(0);
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = vn * q0s[tx * 4 + j];
    }
    // Two passes: pass 0 = x @ Qx^T (q_col0 = 1), pass 1 = b @ Qsc[s]^T (q_col0 = 1 + C + s * C)
    for (int pass = 0; pass < 2; ++pass) {
        const T* src = pass == 0 ? x : b;
        const int q_col0 = pass == 0 ? 1 : (1 + C + s * C);
        for (int k0 = 0; k0 < C; k0 += TK) {
            __syncthreads();
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int idx = tid + 128 * q, r = idx >> 3, kq = (idx & 7) * 4;
                const int n = row_map[r];
                if constexpr (std::is_same<T, float>::value) {
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (n >= 0) v = *reinterpret_cast<const float4*>(src + (long)n * C + k0 + kq);
                    As[kq][r] = v.x; As[kq + 1][r] = v.y; As[kq + 2][r] = v.z; As[kq + 3][r] = v.w;
                } else {
                    const T* ap = n >= 0 ? src + (long)n * C + k0 + kq : nullptr;
                    for (int u = 0; u < 4; ++u) As[kq + u][r] = ap ? ap[u] : T(0);
                }
            }
            // Q^T tile: Q[n0 + nn, q_col0 + k0 + kk] -> Bs[kk][nn] (64 nn x 32 kk = 2048 elems / 128 = 16/thread)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int idx = tid + 128 * q, nn = idx >> 3, kq = (idx & 7) * 4;
                const T* qp = Q + (long)(n0 + nn) * ZW + q_col0 + k0 + kq;
                Bs[kq][nn] = qp[0]; Bs[kq + 1][nn] = qp[1]; Bs[kq + 2][nn] = qp[2]; Bs[kq + 3][nn] = qp[3];
            }
            __syncthreads();
#pragma unroll 8
            for (int kk = 0; kk < TK; ++kk) {
                T av[4], bv[4];
                if constexpr (std::is_same<T, float>::value) {
                    const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
                    const float4 b4 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                    av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                    bv[0] = b4.x; bv[1] = b4.y; bv[2] = b4.z; bv[3] = b4.w;
                } else {
#pragma unroll
                    for (int u = 0; u < 4; ++u) { av[u] = As[kk][ty * 4 + u]; bv[u] = Bs[kk][tx * 4 + u]; }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int n = row_map[ty * 4 + i];
        if (n < 0) continue;
        T* op = out + (long)n * C + n0 + tx * 4;
        if constexpr (std::is_same<T, float>::value) {
            *reinterpret_cast<float4*>(op) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) op[j] = acc[i][j];
        }
    }
}

// Compact Qs weight gradient: replaces Qs = f0bs^T @ zb + f0^T @ zs (2 cuBLAS + 2 splitKreduce) in ONE launch!
// Atomic-adds into zeroed Qs [C, ZW]:
//   blk_col == 0: Qs[:, 1..C] += f0bs^T @ x1b + f0^T @ Xd1  (and Qs[:, 0] += sum_n (f0bs*scale + f0*qn))
//   blk_col == 1 + s (0 <= s < S): Qs[:, 1+C+s*C ..] += sum_{n: spc==s} (f0bs^T f2b + f0^T f2s)
// Grid: dim3(SK, (1 + S) * (C/64) * (C/64)), 256 threads (4x4 per thread over 64x64 output tile).
template <typename T>
__global__ void __launch_bounds__(256) k_q_wgrad(const T* __restrict__ f0bs, T scale, const T* __restrict__ x1b,
                                                 const T* __restrict__ f2b, const T* __restrict__ f0,
                                                 const T* __restrict__ qn, const T* __restrict__ Xd1,
                                                 const T* __restrict__ f2s, const int* __restrict__ spc_off,
                                                 const int* __restrict__ spc_idx, T* __restrict__ Qs, int N, int C,
                                                 int ZW, int S) {
    const int ntc = C / 64;
    const int sk = blockIdx.x, SK = gridDim.x;
    const int blk = blockIdx.y;
    const int col_grp = blk / (ntc * ntc), rem = blk % (ntc * ntc);
    const int tc = rem / ntc, td = rem % ntc;
    const int c0 = tc * 64, d0 = td * 64;
    const bool is_x = (col_grp == 0);
    const int s = col_grp - 1;
    const int off = is_x ? 0 : spc_off[s];
    const int cnt = is_x ? N : (spc_off[s + 1] - off);
    const int r_beg = (cnt * sk) / SK, r_end = (cnt * (sk + 1)) / SK;
    if (r_beg >= r_end) return;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    __shared__ int rmap[TK];
    __shared__ __align__(16) T As[TK][68];
    __shared__ __align__(16) T Bs[TK][68];
    __shared__ T qns[TK];
    const bool do_col0 = is_x && (td == 0);
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    T col0_acc[4] = {T(0), T(0), T(0), T(0)};
    // Two terms: term 0 = (f0bs, x1b/f2b, scale), term 1 = (f0, Xd1/f2s, qn)
    for (int term = 0; term < 2; ++term) {
        const T* A = term == 0 ? f0bs : f0;
        const T* B = is_x ? (term == 0 ? x1b : Xd1) : (term == 0 ? f2b : f2s);
        for (int r0 = r_beg; r0 < r_end; r0 += TK) {
            if (tid < TK) {
                const int r = r0 + tid;
                const int n = r < r_end ? (is_x ? r : spc_idx[off + r]) : -1;
                rmap[tid] = n;
                if (do_col0) qns[tid] = n >= 0 ? (term == 0 ? scale : qn[n]) : T(0);
            }
            __syncthreads();
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int idx = tid + 256 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
                const int n = rmap[kk];
                if constexpr (std::is_same<T, float>::value) {
                    float4 va = make_float4(0.f, 0.f, 0.f, 0.f), vb = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (n >= 0) {
                        va = *reinterpret_cast<const float4*>(A + (long)n * C + c0 + col4);
                        vb = *reinterpret_cast<const float4*>(B + (long)n * C + d0 + col4);
                    }
                    *reinterpret_cast<float4*>(&As[kk][col4]) = va;
                    *reinterpret_cast<float4*>(&Bs[kk][col4]) = vb;
                } else {
                    const T* ap = n >= 0 ? A + (long)n * C + c0 + col4 : nullptr;
                    const T* bp = n >= 0 ? B + (long)n * C + d0 + col4 : nullptr;
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
                    const float4 b4 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                    av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                    bv[0] = b4.x; bv[1] = b4.y; bv[2] = b4.z; bv[3] = b4.w;
                } else {
#pragma unroll
                    for (int u = 0; u < 4; ++u) { av[u] = As[kk][ty * 4 + u]; bv[u] = Bs[kk][tx * 4 + u]; }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
                if (do_col0 && tx == 0) {
                    const T qv = qns[kk];
#pragma unroll
                    for (int i = 0; i < 4; ++i) col0_acc[i] += av[i] * qv;
                }
            }
            __syncthreads();
        }
    }
    const int q_col0 = is_x ? (1 + d0) : (1 + C + s * C + d0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const long crow = (long)(c0 + ty * 4 + i) * ZW;
#pragma unroll
        for (int j = 0; j < 4; ++j) atomicAdd(Qs + crow + q_col0 + tx * 4 + j, acc[i][j]);
        if (do_col0 && tx == 0) atomicAdd(Qs + crow, col0_acc[i]);
    }
}

// Single-launch small node GEMM: out[N, No] = (ADD_SRC ? src[N, No] : 0) + in[N, Ki] @ op(W)
// Handles wp1 (Ki=64, No=64, with optional fused +sc replacing k_copy!) and wro1 (Ki=64, No=16 or Ki=16, No=64).
// Grid: dim3((No + 63) / 64, (N + 31) / 32), 128 threads (or 64 threads when No <= 32).
template <typename T, bool TB, bool ADD_SRC>
__global__ void __launch_bounds__(128) k_node_gemm(const T* __restrict__ A, const T* __restrict__ W,
                                                   const T* __restrict__ src, T* __restrict__ C_out, int N, int Ki,
                                                   int No, int ldw) {
    const int m0 = blockIdx.y * RM, n0 = blockIdx.x * CN;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    __shared__ __align__(16) T As[TK][RM + 4];
    __shared__ __align__(16) T Bs[TK][CN + 4];
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    for (int k0 = 0; k0 < Ki; k0 += TK) {
        const int klen = min(TK, Ki - k0);
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 128 * q, r = idx >> 3, kq = (idx & 7) * 4;
            const int row = m0 + r;
            if (row < N && kq < klen) {
                const T* ap = A + (long)row * Ki + k0 + kq;
                As[kq][r] = ap[0]; As[kq + 1][r] = ap[1]; As[kq + 2][r] = ap[2]; As[kq + 3][r] = ap[3];
            } else {
                As[kq][r] = T(0); As[kq + 1][r] = T(0); As[kq + 2][r] = T(0); As[kq + 3][r] = T(0);
            }
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + 128 * q;
            if constexpr (!TB) {
                const int kk = idx >> 4, nq = (idx & 15) * 4;
                const int col = n0 + nq;
                if (kk < klen && col < No) {
                    const T* wp = W + (long)(k0 + kk) * ldw + col;
                    Bs[kk][nq] = wp[0]; Bs[kk][nq + 1] = wp[1]; Bs[kk][nq + 2] = wp[2]; Bs[kk][nq + 3] = wp[3];
                } else {
                    Bs[kk][nq] = T(0); Bs[kk][nq + 1] = T(0); Bs[kk][nq + 2] = T(0); Bs[kk][nq + 3] = T(0);
                }
            } else {
                const int nn = idx >> 3, kq = (idx & 7) * 4;
                const int col = n0 + nn;
                if (col < No && kq < klen) {
                    const T* wp = W + (long)col * ldw + k0 + kq;
                    Bs[kq][nn] = wp[0]; Bs[kq + 1][nn] = wp[1]; Bs[kq + 2][nn] = wp[2]; Bs[kq + 3][nn] = wp[3];
                } else {
                    Bs[kq][nn] = T(0); Bs[kq + 1][nn] = T(0); Bs[kq + 2][nn] = T(0); Bs[kq + 3][nn] = T(0);
                }
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < TK; ++kk) {
            T av[4], bv[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
                const float4 b4 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                bv[0] = b4.x; bv[1] = b4.y; bv[2] = b4.z; bv[3] = b4.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) { av[u] = As[kk][ty * 4 + u]; bv[u] = Bs[kk][tx * 4 + u]; }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = m0 + ty * 4 + i;
        if (row >= N) continue;
        const int col = n0 + tx * 4;
        if (col >= No) continue;
        T* op = C_out + (long)row * No + col;
        if constexpr (ADD_SRC) {
            const T* sp = src + (long)row * No + col;
            for (int j = 0; j < 4; ++j) op[j] = sp[j] + acc[i][j];
        } else {
            for (int j = 0; j < 4; ++j) op[j] = acc[i][j];
        }
    }
}

// Fused pair wgrad: out[Ma, Nb] += A1^T @ B1 + A2^T @ B2 (atomicAdd into zeroed out; Ma, Nb multiples of 4, <= 64).
// Replaces BOTH wp1s GEMMs + 2 splitKreduce, and BOTH wro1s GEMMs + 2 splitKreduce!
// Grid: dim3(SK, ((Ma + 63)/64) * ((Nb + 63)/64)), 256 threads.
template <typename T>
__global__ void __launch_bounds__(256) k_wgrad_pair(const T* __restrict__ A1, const T* __restrict__ B1,
                                                    const T* __restrict__ A2, const T* __restrict__ B2,
                                                    T* __restrict__ out, int N, int Ma, int Nb) {
    const int ntb = (Nb + 63) / 64;
    const int sk = blockIdx.x, SK = gridDim.x;
    const int ta = blockIdx.y / ntb, tb = blockIdx.y % ntb;
    const int a0 = ta * 64, b0 = tb * 64;
    const int r_beg = (N * sk) / SK, r_end = (N * (sk + 1)) / SK;
    if (r_beg >= r_end) return;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    __shared__ __align__(16) T As[TK][68];
    __shared__ __align__(16) T Bs[TK][68];
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    for (int term = 0; term < 2; ++term) {
        const T* A = term == 0 ? A1 : A2;
        const T* B = term == 0 ? B1 : B2;
        for (int r0 = r_beg; r0 < r_end; r0 += TK) {
            __syncthreads();
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int idx = tid + 256 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
                const int r = r0 + kk;
                if constexpr (std::is_same<T, float>::value) {
                    float4 va = make_float4(0.f, 0.f, 0.f, 0.f), vb = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (r < r_end && a0 + col4 < Ma) va = *reinterpret_cast<const float4*>(A + (long)r * Ma + a0 + col4);
                    if (r < r_end && b0 + col4 < Nb) vb = *reinterpret_cast<const float4*>(B + (long)r * Nb + b0 + col4);
                    *reinterpret_cast<float4*>(&As[kk][col4]) = va;
                    *reinterpret_cast<float4*>(&Bs[kk][col4]) = vb;
                } else {
                    if (r < r_end && a0 + col4 < Ma) {
                        const T* ap = A + (long)r * Ma + a0 + col4;
                        As[kk][col4] = ap[0]; As[kk][col4 + 1] = ap[1]; As[kk][col4 + 2] = ap[2]; As[kk][col4 + 3] = ap[3];
                    } else {
                        As[kk][col4] = T(0); As[kk][col4 + 1] = T(0); As[kk][col4 + 2] = T(0); As[kk][col4 + 3] = T(0);
                    }
                    if (r < r_end && b0 + col4 < Nb) {
                        const T* bp = B + (long)r * Nb + b0 + col4;
                        Bs[kk][col4] = bp[0]; Bs[kk][col4 + 1] = bp[1]; Bs[kk][col4 + 2] = bp[2]; Bs[kk][col4 + 3] = bp[3];
                    } else {
                        Bs[kk][col4] = T(0); Bs[kk][col4 + 1] = T(0); Bs[kk][col4 + 2] = T(0); Bs[kk][col4 + 3] = T(0);
                    }
                }
            }
            __syncthreads();
#pragma unroll 8
            for (int kk = 0; kk < TK; ++kk) {
                T av[4], bv[4];
                if constexpr (std::is_same<T, float>::value) {
                    const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
                    const float4 b4 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                    av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                    bv[0] = b4.x; bv[1] = b4.y; bv[2] = b4.z; bv[3] = b4.w;
                } else {
#pragma unroll
                    for (int u = 0; u < 4; ++u) { av[u] = As[kk][ty * 4 + u]; bv[u] = Bs[kk][tx * 4 + u]; }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int a = a0 + ty * 4 + i;
        if (a >= Ma) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int b = b0 + tx * 4 + j;
            if (b < Nb) atomicAdd(out + (long)a * Nb + b, acc[i][j]);
        }
    }
}

// T0s[s, c] += sum_{n: spc[n] == s} Xd0[n, c] (split-K into zeroed T0s). Replaces onehot^T @ Xd0 cuBLAS + splitKreduce!
// Grid: dim3(SK, S), C threads (e.g. SK = 8 -> 24 blocks instead of 3).
template <typename T>
__global__ void k_t0s_spc(const T* __restrict__ Xd0, const int* __restrict__ spc_off, const int* __restrict__ spc_idx,
                          T* __restrict__ T0s, int C) {
    const int s = blockIdx.y;
    const int sk = blockIdx.x, SK = gridDim.x;
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int c = threadIdx.x;
    if (c >= C) return;
    const int j_beg = (cnt * sk) / SK, j_end = (cnt * (sk + 1)) / SK;
    if (j_beg >= j_end) return;
    T acc = T(0);
    for (int j = j_beg; j < j_end; ++j) acc += Xd0[(long)spc_idx[off + j] * C + c];
    atomicAdd(T0s + (long)s * C + c, acc);
}

// wro2s[h] = sum_n (scale * tbs[n, h] + tt[n, h] * qn[n]). Replaces both gemvNSP_kernel calls!
// Grid: 1 block of 256 threads (16 threads per h when hid=16).
template <typename T>
__global__ void k_wro2s(const T* __restrict__ tbs, const T* __restrict__ tt, const T* __restrict__ qn, T scale,
                        T* __restrict__ wro2s, int N, int hid) {
    __shared__ T sm[256];
    const int tid = threadIdx.x;
    const int h = tid % hid;
    const int lane = tid / hid;
    const int stride = blockDim.x / hid;
    T acc = T(0);
    if (lane < stride) {
        for (int n = lane; n < N; n += stride) acc += scale * tbs[(long)n * hid + h] + tt[(long)n * hid + h] * qn[n];
    }
    sm[tid] = acc;
    __syncthreads();
    if (tid < hid) {
        T s = T(0);
        for (int k = 0; k < stride; ++k) s += sm[k * hid + tid];
        wro2s[tid] = s;
    }
}

// Compact k_readout reading contiguous z0[n] (stride 1) instead of strided z[n * ZW].
template <typename T>
__global__ void k_readout_z0(const T* u, const T* z0, const T* wro2, const int* ngraph, T* tt, T* ne, T* Eg, T act,
                             T scale, int N, int hid) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    T e1 = T(0);
    for (int h = 0; h < hid; ++h) {
        T v = u[(long)n * hid + h];
        T t = act * v * (T(1) / (T(1) + exp(-v)));
        tt[(long)n * hid + h] = t;
        e1 += t * wro2[h];
    }
    T e = scale * (z0[n] + e1);
    ne[n] = e;
    atomicAdd(Eg + ngraph[n], e);
}

// ============================================================================ Wave 14.2: High-occupancy node_ops
// 1. k_q_bwd_c32: CN_B = 32 (doubles grid from 12/42 -> 24/84 blocks, halves Bs smem -> 2x occ over k_q_bwd!).
// Grid: dim3(C / 32, spc_max_tiles32(N, S)), 128 threads (2 rows x 4 cols per thread: ty = tid>>3 in 0..15, tx = tid&7 in 0..7).
template <typename T>
__global__ void __launch_bounds__(128) k_q_bwd_c32(const T* __restrict__ v0, T c0, const T* __restrict__ x,
                                                   const T* __restrict__ b, const T* __restrict__ Q,
                                                   const int* __restrict__ spc_off, const int* __restrict__ spc_idx,
                                                   const int* __restrict__ spc_tiles32, T* __restrict__ out, int C,
                                                   int ZW, int S) {
    const int tot = spc_tiles32[S];
    const int by = blockIdx.y;
    if (by >= tot) return;
    int s = 0;
    while (s + 1 < S && by >= spc_tiles32[s + 1]) ++s;
    const int t = by - spc_tiles32[s];
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int m0 = t * RM, n0 = blockIdx.x * 32;
    const int tid = threadIdx.x, tx = tid & 7, ty = tid >> 3;
    __shared__ int row_map[RM];
    if (tid < RM) row_map[tid] = (m0 + tid < cnt) ? spc_idx[off + m0 + tid] : -1;
    __shared__ __align__(16) T As[TK][RM + 4];
    __shared__ __align__(16) T Bs[TK][32 + 4];
    __shared__ T q0s[32];
    if (tid < 32) q0s[tid] = Q[(long)(n0 + tid) * ZW];
    __syncthreads();
    T acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int n = row_map[ty * 2 + i];
        const T vn = n >= 0 ? (v0 ? v0[n] : c0) : T(0);
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = vn * q0s[tx * 4 + j];
    }
    for (int pass = 0; pass < 2; ++pass) {
        const T* src = pass == 0 ? x : b;
        const int q_col0 = pass == 0 ? 1 : (1 + C + s * C);
        for (int k0 = 0; k0 < C; k0 += TK) {
            __syncthreads();
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int idx = tid + 128 * q, r = idx >> 3, kq = (idx & 7) * 4;
                const int n = row_map[r];
                if constexpr (std::is_same<T, float>::value) {
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (n >= 0) v = *reinterpret_cast<const float4*>(src + (long)n * C + k0 + kq);
                    As[kq][r] = v.x; As[kq + 1][r] = v.y; As[kq + 2][r] = v.z; As[kq + 3][r] = v.w;
                } else {
                    const T* ap = n >= 0 ? src + (long)n * C + k0 + kq : nullptr;
                    for (int u = 0; u < 4; ++u) As[kq + u][r] = ap ? ap[u] : T(0);
                }
            }
            // Q^T tile: 32 nn x 32 kk = 1024 elems / 128 = 8 elems/thread (2 chunks of 4)
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int idx = tid + 128 * q, nn = idx >> 3, kq = (idx & 7) * 4;
                const T* qp = Q + (long)(n0 + nn) * ZW + q_col0 + k0 + kq;
                Bs[kq][nn] = qp[0]; Bs[kq + 1][nn] = qp[1]; Bs[kq + 2][nn] = qp[2]; Bs[kq + 3][nn] = qp[3];
            }
            __syncthreads();
#pragma unroll 8
            for (int kk = 0; kk < TK; ++kk) {
                T av[2] = {As[kk][ty * 2], As[kk][ty * 2 + 1]};
                T bv[4];
                if constexpr (std::is_same<T, float>::value) {
                    const float4 b4 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                    bv[0] = b4.x; bv[1] = b4.y; bv[2] = b4.z; bv[3] = b4.w;
                } else {
#pragma unroll
                    for (int u = 0; u < 4; ++u) bv[u] = Bs[kk][tx * 4 + u];
                }
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int n = row_map[ty * 2 + i];
        if (n < 0) continue;
        T* op = out + (long)n * C + n0 + tx * 4;
        if constexpr (std::is_same<T, float>::value) {
            *reinterpret_cast<float4*>(op) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) op[j] = acc[i][j];
        }
    }
}

// 2. k_q_wgrad_bal: eliminates the 5x straggler on col_grp==0 (is_x, all N atoms) by assigning sk_x chunks to col_grp==0
// and sk_s chunks to each species s (Grid: dim3(sk_x + S * sk_s, ntc * ntc), 256 threads).
template <typename T>
__global__ void __launch_bounds__(256) k_q_wgrad_bal(const T* __restrict__ f0bs, T scale, const T* __restrict__ x1b,
                                                     const T* __restrict__ f2b, const T* __restrict__ f0,
                                                     const T* __restrict__ qn, const T* __restrict__ Xd1,
                                                     const T* __restrict__ f2s, const int* __restrict__ spc_off,
                                                     const int* __restrict__ spc_idx, T* __restrict__ Qs, int N, int C,
                                                     int ZW, int S, int sk_x, int sk_s) {
    const int ntc = C / 64;
    const int bx = blockIdx.x;
    const bool is_x = (bx < sk_x);
    const int s = is_x ? -1 : (bx - sk_x) / sk_s;
    const int sk = is_x ? bx : (bx - sk_x) % sk_s;
    const int SK = is_x ? sk_x : sk_s;
    if (!is_x && s >= S) return;
    const int tc = blockIdx.y / ntc, td = blockIdx.y % ntc;
    const int c0 = tc * 64, d0 = td * 64;
    const int off = is_x ? 0 : spc_off[s];
    const int cnt = is_x ? N : (spc_off[s + 1] - off);
    const int r_beg = (cnt * sk) / SK, r_end = (cnt * (sk + 1)) / SK;
    if (r_beg >= r_end) return;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    __shared__ int rmap[TK];
    __shared__ __align__(16) T As[TK][68];
    __shared__ __align__(16) T Bs[TK][68];
    __shared__ T qns[TK];
    const bool do_col0 = is_x && (td == 0);
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    T col0_acc[4] = {T(0), T(0), T(0), T(0)};
    for (int term = 0; term < 2; ++term) {
        const T* A = term == 0 ? f0bs : f0;
        const T* B = is_x ? (term == 0 ? x1b : Xd1) : (term == 0 ? f2b : f2s);
        for (int r0 = r_beg; r0 < r_end; r0 += TK) {
            if (tid < TK) {
                const int r = r0 + tid;
                const int n = r < r_end ? (is_x ? r : spc_idx[off + r]) : -1;
                rmap[tid] = n;
                if (do_col0) qns[tid] = n >= 0 ? (term == 0 ? scale : qn[n]) : T(0);
            }
            __syncthreads();
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int idx = tid + 256 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
                const int n = rmap[kk];
                if constexpr (std::is_same<T, float>::value) {
                    float4 va = make_float4(0.f, 0.f, 0.f, 0.f), vb = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (n >= 0) {
                        va = *reinterpret_cast<const float4*>(A + (long)n * C + c0 + col4);
                        vb = *reinterpret_cast<const float4*>(B + (long)n * C + d0 + col4);
                    }
                    *reinterpret_cast<float4*>(&As[kk][col4]) = va;
                    *reinterpret_cast<float4*>(&Bs[kk][col4]) = vb;
                } else {
                    const T* ap = n >= 0 ? A + (long)n * C + c0 + col4 : nullptr;
                    const T* bp = n >= 0 ? B + (long)n * C + d0 + col4 : nullptr;
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
                    const float4 b4 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                    av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                    bv[0] = b4.x; bv[1] = b4.y; bv[2] = b4.z; bv[3] = b4.w;
                } else {
#pragma unroll
                    for (int u = 0; u < 4; ++u) { av[u] = As[kk][ty * 4 + u]; bv[u] = Bs[kk][tx * 4 + u]; }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
                if (do_col0 && tx == 0) {
                    const T qv = qns[kk];
#pragma unroll
                    for (int i = 0; i < 4; ++i) col0_acc[i] += av[i] * qv;
                }
            }
            __syncthreads();
        }
    }
    const int q_col0 = is_x ? (1 + d0) : (1 + C + s * C + d0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const long crow = (long)(c0 + ty * 4 + i) * ZW;
#pragma unroll
        for (int j = 0; j < 4; ++j) atomicAdd(Qs + crow + q_col0 + tx * 4 + j, acc[i][j]);
        if (do_col0 && tx == 0) atomicAdd(Qs + crow, col0_acc[i]);
    }
}

// 3. Dedicated 64-thread 64x16 pair wgrad for wro1s (100% thread utilization vs 25% in 256-thread k_wgrad_pair!).
// Grid: SK blocks of 64 threads (ty = tid>>2 in 0..15 -> 4 rows, tx = tid&3 in 0..3 -> 4 cols).
template <typename T>
__global__ void __launch_bounds__(64) k_wgrad_pair_64x16(const T* __restrict__ A1, const T* __restrict__ B1,
                                                         const T* __restrict__ A2, const T* __restrict__ B2,
                                                         T* __restrict__ out, int N) {
    const int sk = blockIdx.x, SK = gridDim.x;
    const int r_beg = (N * sk) / SK, r_end = (N * (sk + 1)) / SK;
    if (r_beg >= r_end) return;
    const int tid = threadIdx.x, tx = tid & 3, ty = tid >> 2;
    __shared__ __align__(16) T As[TK][68];
    __shared__ __align__(16) T Bs[TK][20];
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    for (int term = 0; term < 2; ++term) {
        const T* A = term == 0 ? A1 : A2;
        const T* B = term == 0 ? B1 : B2;
        for (int r0 = r_beg; r0 < r_end; r0 += TK) {
            __syncthreads();
            // A: 32x64 = 2048 elems / 64 threads = 8 float4s per thread
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int idx = tid + 64 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
                const int r = r0 + kk;
                if constexpr (std::is_same<T, float>::value) {
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (r < r_end) v = *reinterpret_cast<const float4*>(A + (long)r * 64 + col4);
                    *reinterpret_cast<float4*>(&As[kk][col4]) = v;
                } else {
                    const T* ap = r < r_end ? A + (long)r * 64 + col4 : nullptr;
                    for (int u = 0; u < 4; ++u) As[kk][col4 + u] = ap ? ap[u] : T(0);
                }
            }
            // B: 32x16 = 512 elems / 64 threads = 2 float4s per thread
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int idx = tid + 64 * q, kk = idx >> 2, col4 = (idx & 3) * 4;
                const int r = r0 + kk;
                if constexpr (std::is_same<T, float>::value) {
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (r < r_end) v = *reinterpret_cast<const float4*>(B + (long)r * 16 + col4);
                    *reinterpret_cast<float4*>(&Bs[kk][col4]) = v;
                } else {
                    const T* bp = r < r_end ? B + (long)r * 16 + col4 : nullptr;
                    for (int u = 0; u < 4; ++u) Bs[kk][col4 + u] = bp ? bp[u] : T(0);
                }
            }
            __syncthreads();
#pragma unroll 8
            for (int kk = 0; kk < TK; ++kk) {
                T av[4], bv[4];
                if constexpr (std::is_same<T, float>::value) {
                    const float4 a = *reinterpret_cast<const float4*>(&As[kk][ty * 4]);
                    const float4 b4 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                    av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                    bv[0] = b4.x; bv[1] = b4.y; bv[2] = b4.z; bv[3] = b4.w;
                } else {
#pragma unroll
                    for (int u = 0; u < 4; ++u) { av[u] = As[kk][ty * 4 + u]; bv[u] = Bs[kk][tx * 4 + u]; }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) atomicAdd(out + (long)(ty * 4 + i) * 16 + tx * 4 + j, acc[i][j]);
}

// 4. Multi-block k_wro2s_par: 8 blocks (instead of 1 single block!) atomic-adding into zeroed wro2s[hid].
template <typename T>
__global__ void k_wro2s_par(const T* __restrict__ tbs, const T* __restrict__ tt, const T* __restrict__ qn, T scale,
                            T* __restrict__ wro2s, int N, int hid) {
    __shared__ T sm[256];
    const int tid = threadIdx.x;
    const int h = tid % hid;
    const int lane = tid / hid;
    const int stride = blockDim.x / hid;
    T acc = T(0);
    if (lane < stride) {
        const int step = gridDim.x * stride;
        for (int n = blockIdx.x * stride + lane; n < N; n += step)
            acc += scale * tbs[(long)n * hid + h] + tt[(long)n * hid + h] * qn[n];
    }
    sm[tid] = acc;
    __syncthreads();
    if (tid < hid) {
        T s = T(0);
        for (int k = 0; k < stride; ++k) s += sm[k * hid + tid];
        atomicAdd(wro2s + tid, s);
    }
}

// 5. Wave 17: Stoichiometry-balanced k_q_wgrad_dyn (LST_QWGRAD_DYN=1):
// Uses sk_x blocks for is_x (all N atoms) + existing spc_wsk[S+1] for species (proportional to cnt[s]!),
// so every CTA processes equal ~64 atoms (eliminates both k_q_wgrad's 5x is_x straggler AND k_q_wgrad_bal's O straggler!)
// AND removes the extra rmap __syncthreads() (3 -> 2 barriers per chunk!).
// Grid: dim3(sk_x + max_wsk, (C/64)*(C/64)), 256 threads.
template <typename T>
__global__ void __launch_bounds__(256) k_q_wgrad_dyn(const T* __restrict__ f0bs, T scale, const T* __restrict__ x1b,
                                                     const T* __restrict__ f2b, const T* __restrict__ f0,
                                                     const T* __restrict__ qn, const T* __restrict__ Xd1,
                                                     const T* __restrict__ f2s, const int* __restrict__ spc_off,
                                                     const int* __restrict__ spc_idx, const int* __restrict__ spc_wsk,
                                                     T* __restrict__ Qs, int N, int C, int ZW, int S, int sk_x) {
    const int ntc = C / 64;
    const int bx = blockIdx.x;
    const bool is_x = (bx < sk_x);
    int s = -1, sk = bx, SK = sk_x;
    if (!is_x) {
        const int sb = bx - sk_x;
        const int tot_sk = spc_wsk[S];
        if (sb >= tot_sk) return;
        s = 0;
        while (s + 1 < S && sb >= spc_wsk[s + 1]) ++s;
        sk = sb - spc_wsk[s];
        SK = spc_wsk[s + 1] - spc_wsk[s];
    }
    const int tc = blockIdx.y / ntc, td = blockIdx.y % ntc;
    const int c0 = tc * 64, d0 = td * 64;
    const int off = is_x ? 0 : spc_off[s];
    const int cnt = is_x ? N : (spc_off[s + 1] - off);
    const int r_beg = (cnt * sk) / SK, r_end = (cnt * (sk + 1)) / SK;
    if (r_beg >= r_end) return;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    __shared__ __align__(16) T As[TK][68];
    __shared__ __align__(16) T Bs[TK][68];
    __shared__ T qns[TK];
    const bool do_col0 = is_x && (td == 0);
    T acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    T col0_acc[4] = {T(0), T(0), T(0), T(0)};
    for (int term = 0; term < 2; ++term) {
        const T* A = term == 0 ? f0bs : f0;
        const T* B = is_x ? (term == 0 ? x1b : Xd1) : (term == 0 ? f2b : f2s);
        for (int r0 = r_beg; r0 < r_end; r0 += TK) {
            if (do_col0 && tid < TK) {
                const int r = r0 + tid;
                qns[tid] = r < r_end ? (term == 0 ? scale : qn[r]) : T(0);
            }
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int idx = tid + 256 * q, kk = idx >> 4, col4 = (idx & 15) * 4;
                const int r = r0 + kk;
                const int n = r < r_end ? (is_x ? r : spc_idx[off + r]) : -1;
                if constexpr (std::is_same<T, float>::value) {
                    float4 va = make_float4(0.f, 0.f, 0.f, 0.f), vb = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (n >= 0) {
                        va = *reinterpret_cast<const float4*>(A + (long)n * C + c0 + col4);
                        vb = *reinterpret_cast<const float4*>(B + (long)n * C + d0 + col4);
                    }
                    *reinterpret_cast<float4*>(&As[kk][col4]) = va;
                    *reinterpret_cast<float4*>(&Bs[kk][col4]) = vb;
                } else {
                    const T* ap = n >= 0 ? A + (long)n * C + c0 + col4 : nullptr;
                    const T* bp = n >= 0 ? B + (long)n * C + d0 + col4 : nullptr;
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
                    const float4 b4 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                    av[0] = a.x; av[1] = a.y; av[2] = a.z; av[3] = a.w;
                    bv[0] = b4.x; bv[1] = b4.y; bv[2] = b4.z; bv[3] = b4.w;
                } else {
#pragma unroll
                    for (int u = 0; u < 4; ++u) { av[u] = As[kk][ty * 4 + u]; bv[u] = Bs[kk][tx * 4 + u]; }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
                if (do_col0 && tx == 0) {
                    const T qv = qns[kk];
#pragma unroll
                    for (int i = 0; i < 4; ++i) col0_acc[i] += av[i] * qv;
                }
            }
            __syncthreads();
        }
    }
    const int q_col0 = is_x ? (1 + d0) : (1 + C + s * C + d0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const long crow = (long)(c0 + ty * 4 + i) * ZW;
#pragma unroll
        for (int j = 0; j < 4; ++j) atomicAdd(Qs + crow + q_col0 + tx * 4 + j, acc[i][j]);
        if (do_col0 && tx == 0) atomicAdd(Qs + crow, col0_acc[i]);
    }
}

// ============================================================================ Wave 35: Fused readout GEMM kernels (C=64, hid=16, FP32)
// 1. k_readout_fwd_bwd<DO_BWD>: fuses f2@wro1->u + readout_z0(tt,ne,Eg) + (if DO_BWD) k_ub(ub) + ub@wro1^T->f2b
//    into ONE launch (4 kernels -> 1!). Loads wro1[64,16] ONCE into 4.3 KB bank-conflict-free smem, reduces Eg per
//    CTA before atomicAdd (20 vs 320 atomics at N=320). 16 atoms/CTA, 128 threads/CTA, exact FP32 math.
template <bool DO_BWD>
__global__ void __launch_bounds__(128, 6) k_readout_fwd_bwd(
    const float* __restrict__ f2, const float* __restrict__ wro1, const float* __restrict__ z0,
    const float* __restrict__ wro2, const int* __restrict__ ngraph,
    float* __restrict__ u_out, float* __restrict__ tt, float* __restrict__ ne, float* __restrict__ Eg,
    float* __restrict__ ub_out, float* __restrict__ f2b, float act, float scale, int N) {
    __shared__ float s_w1[64][17];
    __shared__ float s_f2[16][65];
    __shared__ float s_ub[DO_BWD ? 16 : 1][DO_BWD ? 17 : 1];
    __shared__ float s_w2[16];
    __shared__ float s_e[16];
    __shared__ int s_g[16];

    const int tid = threadIdx.x;
    const int n0 = blockIdx.x * 16;

    // Load wro1[64][16] (1024 floats = 2 float4/thread) and wro2[16]
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int idx = tid + 128 * q;
        const int r = idx >> 2, h4 = (idx & 3) * 4;
        const float4 v = *reinterpret_cast<const float4*>(wro1 + r * 16 + h4);
        s_w1[r][h4 + 0] = v.x;
        s_w1[r][h4 + 1] = v.y;
        s_w1[r][h4 + 2] = v.z;
        s_w1[r][h4 + 3] = v.w;
    }
    if (tid < 16) s_w2[tid] = wro2[tid];

    // Load f2[n0..n0+15][64] (1024 floats = 2 float4/thread)
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int idx = tid + 128 * q;
        const int a = idx >> 4, c4 = (idx & 15) * 4;
        const int n = n0 + a;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n < N) v = *reinterpret_cast<const float4*>(f2 + (long)n * 64 + c4);
        s_f2[a][c4 + 0] = v.x;
        s_f2[a][c4 + 1] = v.y;
        s_f2[a][c4 + 2] = v.z;
        s_f2[a][c4 + 3] = v.w;
    }
    __syncthreads();

    // Step 1: 16 atoms x 8 h-pairs (h0 = 2*hh, h1 = 2*hh+1)
    const int a = tid >> 3, hh = tid & 7;
    const int h0 = 2 * hh, h1 = h0 + 1;
    const int n = n0 + a;
    float e1 = 0.f;
    if (n < N) {
        float u0 = 0.f, u1 = 0.f;
#pragma unroll 8
        for (int k = 0; k < 64; ++k) {
            const float fv = s_f2[a][k];
            u0 = fmaf(fv, s_w1[k][h0], u0);
            u1 = fmaf(fv, s_w1[k][h1], u1);
        }
        *reinterpret_cast<float2*>(u_out + (long)n * 16 + h0) = make_float2(u0, u1);
        const float sig0 = 1.f / (1.f + expf(-u0));
        const float sig1 = 1.f / (1.f + expf(-u1));
        const float t0 = act * u0 * sig0;
        const float t1 = act * u1 * sig1;
        *reinterpret_cast<float2*>(tt + (long)n * 16 + h0) = make_float2(t0, t1);
        const float w20 = s_w2[h0], w21 = s_w2[h1];
        e1 = t0 * w20 + t1 * w21;
        if constexpr (DO_BWD) {
            const float ub0 = scale * w20 * act * sig0 * (1.f + u0 * (1.f - sig0));
            const float ub1 = scale * w21 * act * sig1 * (1.f + u1 * (1.f - sig1));
            *reinterpret_cast<float2*>(ub_out + (long)n * 16 + h0) = make_float2(ub0, ub1);
            s_ub[a][h0] = ub0;
            s_ub[a][h1] = ub1;
        }
    }
    // Warp-convergent reduction across the 8 h-lanes of each atom (safe even when N is not a multiple of 16, e.g. N=321!)
    e1 += __shfl_xor_sync(0xffffffffu, e1, 4);
    e1 += __shfl_xor_sync(0xffffffffu, e1, 2);
    e1 += __shfl_xor_sync(0xffffffffu, e1, 1);
    if (hh == 0) {
        if (n < N) {
            const float e = scale * (z0[n] + e1);
            ne[n] = e;
            s_e[a] = e;
            s_g[a] = ngraph[n];
        } else {
            s_e[a] = 0.f;
            s_g[a] = -1;
        }
    }
    __syncthreads();

    // Warp-reduced Eg atomicAdd (1 atomic per distinct graph in CTA!)
    if (tid < 32) {
        const int g = (tid < 16) ? s_g[tid] : -1;
        const float ev = (tid < 16) ? s_e[tid] : 0.f;
        unsigned todo = __ballot_sync(0xffffffffu, g >= 0);
        while (todo) {
            const int lead = __ffs(todo) - 1;
            const int g_cur = __shfl_sync(0xffffffffu, g, lead);
            const unsigned grp = todo & __ballot_sync(0xffffffffu, g == g_cur);
            float sv = (g == g_cur) ? ev : 0.f;
            sv += __shfl_xor_sync(0xffffffffu, sv, 8);
            sv += __shfl_xor_sync(0xffffffffu, sv, 4);
            sv += __shfl_xor_sync(0xffffffffu, sv, 2);
            sv += __shfl_xor_sync(0xffffffffu, sv, 1);
            if (tid == lead) atomicAdd(Eg + g_cur, sv);
            todo &= ~grp;
        }
    }

    // Step 2: f2b = ub @ wro1^T (16 atoms x 8 lanes; each lane handles c0=4*hh and c1=32+4*hh -> 0 bank conflicts!)
    if constexpr (DO_BWD) {
        if (n < N) {
            const int c0 = 4 * hh, c1 = 32 + c0;
            float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
            float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
            for (int h = 0; h < 16; ++h) {
                const float ubv = s_ub[a][h];
                o0 = fmaf(ubv, s_w1[c0 + 0][h], o0);
                o1 = fmaf(ubv, s_w1[c0 + 1][h], o1);
                o2 = fmaf(ubv, s_w1[c0 + 2][h], o2);
                o3 = fmaf(ubv, s_w1[c0 + 3][h], o3);
                p0 = fmaf(ubv, s_w1[c1 + 0][h], p0);
                p1 = fmaf(ubv, s_w1[c1 + 1][h], p1);
                p2 = fmaf(ubv, s_w1[c1 + 2][h], p2);
                p3 = fmaf(ubv, s_w1[c1 + 3][h], p3);
            }
            *reinterpret_cast<float4*>(f2b + (long)n * 64 + c0) = make_float4(o0, o1, o2, o3);
            *reinterpret_cast<float4*>(f2b + (long)n * 64 + c1) = make_float4(p0, p1, p2, p3);
        }
    }
}

// 2. k_readout_rev_gemm: fuses f2bs@wro1->ubs(reg) + k_readout_rev(tbs,us,qn) + us@wro1^T->f2s (3 kernels -> 1!).
//    Never writes dead intermediate ubs to global memory; loads wro1[64,16] ONCE into 4.3 KB smem.
__global__ void __launch_bounds__(128, 6) k_readout_rev_gemm(
    const float* __restrict__ f2bs, const float* __restrict__ wro1, const float* __restrict__ u_in,
    const float* __restrict__ wro2, const float* __restrict__ Eb, const int* __restrict__ ngraph,
    float* __restrict__ tbs, float* __restrict__ us_out, float* __restrict__ qn, float* __restrict__ f2s,
    float act, float scale, int N, const float* __restrict__ d_scot = nullptr,
    const float* __restrict__ Enb = nullptr) {
    __shared__ float s_w1[64][17];
    __shared__ float s_f2[16][65];
    __shared__ float s_us[16][17];
    __shared__ float s_w2[16];

    const int tid = threadIdx.x;
    const int n0 = blockIdx.x * 16;

#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int idx = tid + 128 * q;
        const int r = idx >> 2, h4 = (idx & 3) * 4;
        const float4 v = *reinterpret_cast<const float4*>(wro1 + r * 16 + h4);
        s_w1[r][h4 + 0] = v.x;
        s_w1[r][h4 + 1] = v.y;
        s_w1[r][h4 + 2] = v.z;
        s_w1[r][h4 + 3] = v.w;
    }
    if (tid < 16) s_w2[tid] = wro2[tid];

#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int idx = tid + 128 * q;
        const int a = idx >> 4, c4 = (idx & 15) * 4;
        const int n = n0 + a;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n < N) v = *reinterpret_cast<const float4*>(f2bs + (long)n * 64 + c4);
        s_f2[a][c4 + 0] = v.x;
        s_f2[a][c4 + 1] = v.y;
        s_f2[a][c4 + 2] = v.z;
        s_f2[a][c4 + 3] = v.w;
    }
    __syncthreads();

    const int a = tid >> 3, hh = tid & 7;
    const int h0 = 2 * hh, h1 = h0 + 1;
    const int n = n0 + a;
    if (n < N) {
        float ubs0 = 0.f, ubs1 = 0.f;
#pragma unroll 8
        for (int k = 0; k < 64; ++k) {
            const float fv = s_f2[a][k];
            ubs0 = fmaf(fv, s_w1[k][h0], ubs0);
            ubs1 = fmaf(fv, s_w1[k][h1], ubs1);
        }
        const float2 uv = *reinterpret_cast<const float2*>(u_in + (long)n * 16 + h0);
        const float sc = d_scot ? d_scot[0] : 1.f;
        const float eb = Eb[ngraph[n]] + (Enb ? Enb[n] : 0.f);
        const float qv = scale * eb * sc;
        if (hh == 0) qn[n] = qv;
        const float s0 = 1.f / (1.f + expf(-uv.x));
        const float s1 = 1.f / (1.f + expf(-uv.y));
        const float d10 = s0 * (1.f + uv.x * (1.f - s0));
        const float d11 = s1 * (1.f + uv.y * (1.f - s1));
        const float d20 = s0 * (1.f - s0) * (2.f + uv.x * (1.f - 2.f * s0));
        const float d21 = s1 * (1.f - s1) * (2.f + uv.y * (1.f - 2.f * s1));
        *reinterpret_cast<float2*>(tbs + (long)n * 16 + h0) = make_float2(ubs0 * act * d10, ubs1 * act * d11);
        const float w20 = s_w2[h0], w21 = s_w2[h1];
        const float us0 = ubs0 * scale * w20 * act * d20 + qv * w20 * act * d10;
        const float us1 = ubs1 * scale * w21 * act * d21 + qv * w21 * act * d11;
        *reinterpret_cast<float2*>(us_out + (long)n * 16 + h0) = make_float2(us0, us1);
        s_us[a][h0] = us0;
        s_us[a][h1] = us1;
    }
    __syncthreads();

    if (n < N) {
        const int c0 = 4 * hh, c1 = 32 + c0;
        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
        for (int h = 0; h < 16; ++h) {
            const float usv = s_us[a][h];
            o0 = fmaf(usv, s_w1[c0 + 0][h], o0);
            o1 = fmaf(usv, s_w1[c0 + 1][h], o1);
            o2 = fmaf(usv, s_w1[c0 + 2][h], o2);
            o3 = fmaf(usv, s_w1[c0 + 3][h], o3);
            p0 = fmaf(usv, s_w1[c1 + 0][h], p0);
            p1 = fmaf(usv, s_w1[c1 + 1][h], p1);
            p2 = fmaf(usv, s_w1[c1 + 2][h], p2);
            p3 = fmaf(usv, s_w1[c1 + 3][h], p3);
        }
        *reinterpret_cast<float4*>(f2s + (long)n * 64 + c0) = make_float4(o0, o1, o2, o3);
        *reinterpret_cast<float4*>(f2s + (long)n * 64 + c1) = make_float4(p0, p1, p2, p3);
    }
}

// ============================================================================ Wave 36: Higher-occupancy Q projections & super-fused wp1+readout+wp1^T
// 1. k_q_fwd_c32: CN=32 (2x CTAs vs k_q_fwd -> 60 CTAs at N=321, full-GPU occ!), 128 threads (2 rows x 4 cols/thread).
// Grid: dim3(2 * (C / 32), spc_max_tiles32(N, S)), 128 threads.
template <typename T>
__global__ void __launch_bounds__(128, 6) k_q_fwd_c32(const T* __restrict__ f0, const T* __restrict__ Q,
                                                      const int* __restrict__ spc_off, const int* __restrict__ spc_idx,
                                                      const int* __restrict__ spc_tiles32, T* __restrict__ z0,
                                                      T* __restrict__ x1, T* __restrict__ sc, int C, int ZW, int S) {
    const int tot = spc_tiles32[S];
    const int by = blockIdx.y;
    if (by >= tot) return;
    int s = 0;
    while (s + 1 < S && by >= spc_tiles32[s + 1]) ++s;
    const int t = by - spc_tiles32[s];
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int m0 = t * RM;
    const int ntc = C / 32;
    const int is_sc = blockIdx.x / ntc;
    const int n0 = (blockIdx.x % ntc) * 32;
    const int q_col0 = is_sc ? (1 + C + s * C + n0) : (1 + n0);
    const int tid = threadIdx.x, tx = tid & 7, ty = tid >> 3;
    __shared__ int row_map[RM];
    if (tid < RM) row_map[tid] = (m0 + tid < cnt) ? spc_idx[off + m0 + tid] : -1;
    __shared__ __align__(16) T As[TK][RM + 4];
    __shared__ __align__(16) T Bs[TK][32 + 4];
    __shared__ T q0s[TK];
    const bool do_z0 = (z0 != nullptr) && (is_sc == 0) && (n0 == 0);
    T acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = T(0);
    T zacc[2] = {T(0), T(0)};
    for (int k0 = 0; k0 < C; k0 += TK) {
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 128 * q, r = idx >> 3, kq = (idx & 7) * 4;
            const int n = row_map[r];
            if constexpr (std::is_same<T, float>::value) {
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (n >= 0) v = *reinterpret_cast<const float4*>(f0 + (long)n * C + k0 + kq);
                As[kq][r] = v.x; As[kq + 1][r] = v.y; As[kq + 2][r] = v.z; As[kq + 3][r] = v.w;
            } else {
                const T* ap = n >= 0 ? f0 + (long)n * C + k0 + kq : nullptr;
                for (int u = 0; u < 4; ++u) As[kq + u][r] = ap ? ap[u] : T(0);
            }
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int idx = tid + 128 * q, kk = idx >> 3, nq = (idx & 7) * 4;
            const T* qp = Q + (long)(k0 + kk) * ZW + q_col0 + nq;
#pragma unroll
            for (int u = 0; u < 4; ++u) Bs[kk][nq + u] = qp[u];
        }
        if (do_z0 && tid < TK) q0s[tid] = Q[(long)(k0 + tid) * ZW];
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < TK; ++kk) {
            const T a0 = As[kk][ty * 2], a1 = As[kk][ty * 2 + 1];
            T bv[4];
            if constexpr (std::is_same<T, float>::value) {
                const float4 b4 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                bv[0] = b4.x; bv[1] = b4.y; bv[2] = b4.z; bv[3] = b4.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) bv[u] = Bs[kk][tx * 4 + u];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[0][j] += a0 * bv[j];
                acc[1][j] += a1 * bv[j];
            }
            if (do_z0 && tx == 0) {
                const T q0v = q0s[kk];
                zacc[0] += a0 * q0v;
                zacc[1] += a1 * q0v;
            }
        }
    }
    T* out = is_sc ? sc : x1;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int n = row_map[ty * 2 + i];
        if (n < 0) continue;
        T* op = out + (long)n * C + n0 + tx * 4;
        if constexpr (std::is_same<T, float>::value) {
            *reinterpret_cast<float4*>(op) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) op[j] = acc[i][j];
        }
        if (do_z0 && tx == 0) z0[n] = zacc[i];
    }
}

// 2. k_q_bwd_r16: RM=16, CN=32 (2x CTAs vs k_q_bwd_c32 using existing spc_tiles32 -> ~54 active CTAs, 7.3 KB smem!).
// Grid: dim3(C / 32, 2 * spc_max_tiles32(N, S)), 128 threads (1 row x 4 cols/thread).
template <typename T>
__global__ void __launch_bounds__(128, 6) k_q_bwd_r16(const T* __restrict__ v0, T c0, const T* __restrict__ x,
                                                      const T* __restrict__ b, const T* __restrict__ Q,
                                                      const int* __restrict__ spc_off, const int* __restrict__ spc_idx,
                                                      const int* __restrict__ spc_tiles32, T* __restrict__ out, int C,
                                                      int ZW, int S) {
    const int tot = spc_tiles32[S];
    const int by = blockIdx.y >> 1;
    if (by >= tot) return;
    int s = 0;
    while (s + 1 < S && by >= spc_tiles32[s + 1]) ++s;
    const int t = by - spc_tiles32[s];
    const int off = spc_off[s], cnt = spc_off[s + 1] - off;
    const int m0 = t * RM + (blockIdx.y & 1) * 16;
    if (m0 >= cnt) return;
    const int n0 = blockIdx.x * 32;
    const int tid = threadIdx.x, tx = tid & 7, ty = tid >> 3;
    __shared__ int row_map[16];
    if (tid < 16) row_map[tid] = (m0 + tid < cnt) ? spc_idx[off + m0 + tid] : -1;
    __shared__ __align__(16) T As[TK][16 + 4];
    __shared__ __align__(16) T Bs[TK][32 + 4];
    __shared__ T q0s[32];
    if (tid < 32) q0s[tid] = Q[(long)(n0 + tid) * ZW];
    __syncthreads();
    const int my_n = row_map[ty];
    const T vn = my_n >= 0 ? (v0 ? v0[my_n] : c0) : T(0);
    T acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = vn * q0s[tx * 4 + j];
    for (int pass = 0; pass < 2; ++pass) {
        const T* src = pass == 0 ? x : b;
        const int q_col0 = pass == 0 ? 1 : (1 + C + s * C);
        for (int k0 = 0; k0 < C; k0 += TK) {
            __syncthreads();
            // A: 16 rows x 32 k = 512 elems / 128 threads = 1 float4/thread
            {
                const int r = tid >> 3, kq = (tid & 7) * 4;
                const int n = row_map[r];
                if constexpr (std::is_same<T, float>::value) {
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (n >= 0) v = *reinterpret_cast<const float4*>(src + (long)n * C + k0 + kq);
                    As[kq][r] = v.x; As[kq + 1][r] = v.y; As[kq + 2][r] = v.z; As[kq + 3][r] = v.w;
                } else {
                    const T* ap = n >= 0 ? src + (long)n * C + k0 + kq : nullptr;
                    for (int u = 0; u < 4; ++u) As[kq + u][r] = ap ? ap[u] : T(0);
                }
            }
            // Q^T tile: 32 nn x 32 kk = 1024 elems / 128 = 2 chunks of 4
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int idx = tid + 128 * q, nn = idx >> 3, kq = (idx & 7) * 4;
                const T* qp = Q + (long)(n0 + nn) * ZW + q_col0 + k0 + kq;
                Bs[kq][nn] = qp[0]; Bs[kq + 1][nn] = qp[1]; Bs[kq + 2][nn] = qp[2]; Bs[kq + 3][nn] = qp[3];
            }
            __syncthreads();
#pragma unroll 8
            for (int kk = 0; kk < TK; ++kk) {
                const T av = As[kk][ty];
                if constexpr (std::is_same<T, float>::value) {
                    const float4 b4 = *reinterpret_cast<const float4*>(&Bs[kk][tx * 4]);
                    acc[0] += av * b4.x; acc[1] += av * b4.y; acc[2] += av * b4.z; acc[3] += av * b4.w;
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[j] += av * Bs[kk][tx * 4 + j];
                }
            }
        }
    }
    if (my_n >= 0) {
        T* op = out + (long)my_n * C + n0 + tx * 4;
        if constexpr (std::is_same<T, float>::value) {
            *reinterpret_cast<float4*>(op) = make_float4(acc[0], acc[1], acc[2], acc[3]);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) op[j] = acc[j];
        }
    }
}

// 3. k_wp1_readout_fwd_bwd<DO_BWD, NA>: super-fuses f1@wp1+sc->f2 + f2@wro1->u,tt,ne,Eg,ub + ub@wro1^T->f2b + f2b@wp1^T->f1b!
//    NA in {8, 12, 16} (128/192/256 threads): NA=12 (30 CTAs) or NA=16 (23 CTAs) fits in 1 wave on BOTH 36-SM 5060 Ti & 48-SM Ada,
//    halving wp1/wro1 L2 loads and doubling warps/CTA over NA=8!
//    Grid: (N + NA - 1) / NA, 16 * NA threads (NA atoms x 16 lanes/atom).
template <bool DO_BWD, int NA = 8>
__global__ void __launch_bounds__(16 * NA, NA <= 8 ? 4 : 3) k_wp1_readout_fwd_bwd(
    const float* __restrict__ f1, const float* __restrict__ wp1, const float* __restrict__ sc,
    float* __restrict__ f2, const float* __restrict__ wro1, const float* __restrict__ z0,
    const float* __restrict__ wro2, const int* __restrict__ ngraph,
    float* __restrict__ u_out, float* __restrict__ tt, float* __restrict__ ne, float* __restrict__ Eg,
    float* __restrict__ ub_out, float* __restrict__ f2b, float* __restrict__ f1b,
    float act, float scale, int N) {
    constexpr int NTH = 16 * NA;
    __shared__ float s_wp1[64][65];
    __shared__ float s_wro1[64][17];
    __shared__ float s_f[NA][65];
    __shared__ float s_u[DO_BWD ? NA : 1][DO_BWD ? 17 : 1];
    __shared__ float s_w2[16];
    __shared__ float s_e[NA];
    __shared__ int s_g[NA];

    const int tid = threadIdx.x;
    const int n0 = blockIdx.x * NA;

    // Load wp1[64][64] (1024 float4s)
#pragma unroll
    for (int idx = tid; idx < 1024; idx += NTH) {
        const int r = idx >> 4, c4 = (idx & 15) * 4;
        const float4 v = *reinterpret_cast<const float4*>(wp1 + r * 64 + c4);
        s_wp1[r][c4 + 0] = v.x; s_wp1[r][c4 + 1] = v.y; s_wp1[r][c4 + 2] = v.z; s_wp1[r][c4 + 3] = v.w;
    }
    // Load wro1[64][16] (256 float4s) & wro2[16]
#pragma unroll
    for (int idx = tid; idx < 256; idx += NTH) {
        const int r = idx >> 2, h4 = (idx & 3) * 4;
        const float4 v = *reinterpret_cast<const float4*>(wro1 + r * 16 + h4);
        s_wro1[r][h4 + 0] = v.x; s_wro1[r][h4 + 1] = v.y; s_wro1[r][h4 + 2] = v.z; s_wro1[r][h4 + 3] = v.w;
    }
    if (tid < 16) s_w2[tid] = wro2[tid];
    // Load f1[n0..n0+NA-1][64] (exact 1 float4/thread)
    {
        const int la = tid >> 4, lc4 = (tid & 15) * 4;
        const int ln = n0 + la;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (ln < N) v = *reinterpret_cast<const float4*>(f1 + (long)ln * 64 + lc4);
        s_f[la][lc4 + 0] = v.x; s_f[la][lc4 + 1] = v.y; s_f[la][lc4 + 2] = v.z; s_f[la][lc4 + 3] = v.w;
    }
    __syncthreads();

    const int a = tid >> 4, q = tid & 15;
    const int n = n0 + a;

    // Stage 1: f2 = f1 @ wp1 + sc (zero bank conflicts via strided q + 16*u)
    float y0 = 0.f, y1 = 0.f, y2 = 0.f, y3 = 0.f;
    if (n < N) {
        const long base = (long)n * 64;
        y0 = sc[base + q +  0]; y1 = sc[base + q + 16]; y2 = sc[base + q + 32]; y3 = sc[base + q + 48];
#pragma unroll 8
        for (int k = 0; k < 64; ++k) {
            const float fk = s_f[a][k];
            y0 = fmaf(fk, s_wp1[k][q +  0], y0);
            y1 = fmaf(fk, s_wp1[k][q + 16], y1);
            y2 = fmaf(fk, s_wp1[k][q + 32], y2);
            y3 = fmaf(fk, s_wp1[k][q + 48], y3);
        }
        f2[base + q +  0] = y0; f2[base + q + 16] = y1; f2[base + q + 32] = y2; f2[base + q + 48] = y3;
    }
    __syncthreads();
    if (n < N) {
        s_f[a][q +  0] = y0; s_f[a][q + 16] = y1; s_f[a][q + 32] = y2; s_f[a][q + 48] = y3;
    }
    __syncthreads();

    // Stage 2: u = f2 @ wro1, SiLU -> tt, ne, Eg, (ub)
    float e1 = 0.f;
    if (n < N) {
        float uv = 0.f;
#pragma unroll 8
        for (int k = 0; k < 64; ++k) uv = fmaf(s_f[a][k], s_wro1[k][q], uv);
        u_out[(long)n * 16 + q] = uv;
        const float sig = 1.f / (1.f + expf(-uv));
        const float tv = act * uv * sig;
        tt[(long)n * 16 + q] = tv;
        const float w2v = s_w2[q];
        e1 = tv * w2v;
        if constexpr (DO_BWD) {
            const float ubv = scale * w2v * act * sig * (1.f + uv * (1.f - sig));
            ub_out[(long)n * 16 + q] = ubv;
            s_u[a][q] = ubv;
        }
    }
    // Convergent 16-lane reduction (safe for any N, e.g. N=321)
    e1 += __shfl_xor_sync(0xffffffffu, e1, 8);
    e1 += __shfl_xor_sync(0xffffffffu, e1, 4);
    e1 += __shfl_xor_sync(0xffffffffu, e1, 2);
    e1 += __shfl_xor_sync(0xffffffffu, e1, 1);
    if (q == 0) {
        if (n < N) {
            const float e = scale * (z0[n] + e1);
            ne[n] = e;
            s_e[a] = e;
            s_g[a] = ngraph[n];
        } else {
            s_e[a] = 0.f;
            s_g[a] = -1;
        }
    }
    __syncthreads();
    if (tid < 32) {
        const int g = (tid < NA) ? s_g[tid] : -1;
        const float ev = (tid < NA) ? s_e[tid] : 0.f;
        unsigned todo = __ballot_sync(0xffffffffu, g >= 0);
        while (todo) {
            const int lead = __ffs(todo) - 1;
            const int g_cur = __shfl_sync(0xffffffffu, g, lead);
            const unsigned grp = todo & __ballot_sync(0xffffffffu, g == g_cur);
            float sv = (g == g_cur) ? ev : 0.f;
            if constexpr (NA > 8) sv += __shfl_xor_sync(0xffffffffu, sv, 8);
            sv += __shfl_xor_sync(0xffffffffu, sv, 4);
            sv += __shfl_xor_sync(0xffffffffu, sv, 2);
            sv += __shfl_xor_sync(0xffffffffu, sv, 1);
            if (tid == lead) atomicAdd(Eg + g_cur, sv);
            todo &= ~grp;
        }
    }

    if constexpr (DO_BWD) {
        // Stage 3: f2b = ub @ wro1^T
        if (n < N) {
            float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
#pragma unroll
            for (int h = 0; h < 16; ++h) {
                const float ubv = s_u[a][h];
                b0 = fmaf(ubv, s_wro1[q +  0][h], b0);
                b1 = fmaf(ubv, s_wro1[q + 16][h], b1);
                b2 = fmaf(ubv, s_wro1[q + 32][h], b2);
                b3 = fmaf(ubv, s_wro1[q + 48][h], b3);
            }
            const long base = (long)n * 64;
            f2b[base + q +  0] = b0; f2b[base + q + 16] = b1; f2b[base + q + 32] = b2; f2b[base + q + 48] = b3;
            s_f[a][q +  0] = b0; s_f[a][q + 16] = b1; s_f[a][q + 32] = b2; s_f[a][q + 48] = b3;
        }
        __syncthreads();
        // Stage 4: f1b = f2b @ wp1^T
        if (n < N) {
            float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
#pragma unroll 8
            for (int k = 0; k < 64; ++k) {
                const float bk = s_f[a][k];
                o0 = fmaf(bk, s_wp1[q +  0][k], o0);
                o1 = fmaf(bk, s_wp1[q + 16][k], o1);
                o2 = fmaf(bk, s_wp1[q + 32][k], o2);
                o3 = fmaf(bk, s_wp1[q + 48][k], o3);
            }
            const long base = (long)n * 64;
            f1b[base + q +  0] = o0; f1b[base + q + 16] = o1; f1b[base + q + 32] = o2; f1b[base + q + 48] = o3;
        }
    }
}

// 4. k_wp1_readout_rev_gemm<NA>: super-fuses f1bs@wp1+scbs->f2bs + f2bs@wro1->ubs(reg) + readout_rev(tbs,us,qn) + us@wro1^T->f2s + f2s@wp1^T->f1s!
//    Grid: (N + NA - 1) / NA, 16 * NA threads (NA atoms x 16 lanes/atom).
template <int NA = 8>
__global__ void __launch_bounds__(16 * NA, NA <= 8 ? 4 : 3) k_wp1_readout_rev_gemm(
    const float* __restrict__ f1bs, const float* __restrict__ wp1, const float* __restrict__ scbs,
    float* __restrict__ f2bs, const float* __restrict__ wro1, const float* __restrict__ u_in,
    const float* __restrict__ wro2, const float* __restrict__ Eb, const int* __restrict__ ngraph,
    float* __restrict__ tbs, float* __restrict__ us_out, float* __restrict__ qn,
    float* __restrict__ f2s, float* __restrict__ f1s,
    float act, float scale, int N, const float* __restrict__ d_scot = nullptr,
    const float* __restrict__ Enb = nullptr) {
    constexpr int NTH = 16 * NA;
    __shared__ float s_wp1[64][65];
    __shared__ float s_wro1[64][17];
    __shared__ float s_f[NA][65];
    __shared__ float s_u[NA][17];
    __shared__ float s_w2[16];

    const int tid = threadIdx.x;
    const int n0 = blockIdx.x * NA;

#pragma unroll
    for (int idx = tid; idx < 1024; idx += NTH) {
        const int r = idx >> 4, c4 = (idx & 15) * 4;
        const float4 v = *reinterpret_cast<const float4*>(wp1 + r * 64 + c4);
        s_wp1[r][c4 + 0] = v.x; s_wp1[r][c4 + 1] = v.y; s_wp1[r][c4 + 2] = v.z; s_wp1[r][c4 + 3] = v.w;
    }
#pragma unroll
    for (int idx = tid; idx < 256; idx += NTH) {
        const int r = idx >> 2, h4 = (idx & 3) * 4;
        const float4 v = *reinterpret_cast<const float4*>(wro1 + r * 16 + h4);
        s_wro1[r][h4 + 0] = v.x; s_wro1[r][h4 + 1] = v.y; s_wro1[r][h4 + 2] = v.z; s_wro1[r][h4 + 3] = v.w;
    }
    if (tid < 16) s_w2[tid] = wro2[tid];
    {
        const int la = tid >> 4, lc4 = (tid & 15) * 4;
        const int ln = n0 + la;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (ln < N) v = *reinterpret_cast<const float4*>(f1bs + (long)ln * 64 + lc4);
        s_f[la][lc4 + 0] = v.x; s_f[la][lc4 + 1] = v.y; s_f[la][lc4 + 2] = v.z; s_f[la][lc4 + 3] = v.w;
    }
    __syncthreads();

    const int a = tid >> 4, q = tid & 15;
    const int n = n0 + a;

    // Stage 1: f2bs = f1bs @ wp1 + scbs
    float y0 = 0.f, y1 = 0.f, y2 = 0.f, y3 = 0.f;
    if (n < N) {
        const long base = (long)n * 64;
        y0 = scbs[base + q +  0]; y1 = scbs[base + q + 16]; y2 = scbs[base + q + 32]; y3 = scbs[base + q + 48];
#pragma unroll 8
        for (int k = 0; k < 64; ++k) {
            const float fk = s_f[a][k];
            y0 = fmaf(fk, s_wp1[k][q +  0], y0);
            y1 = fmaf(fk, s_wp1[k][q + 16], y1);
            y2 = fmaf(fk, s_wp1[k][q + 32], y2);
            y3 = fmaf(fk, s_wp1[k][q + 48], y3);
        }
        f2bs[base + q +  0] = y0; f2bs[base + q + 16] = y1; f2bs[base + q + 32] = y2; f2bs[base + q + 48] = y3;
    }
    __syncthreads();
    if (n < N) {
        s_f[a][q +  0] = y0; s_f[a][q + 16] = y1; s_f[a][q + 32] = y2; s_f[a][q + 48] = y3;
    }
    __syncthreads();

    // Stage 2: ubs = f2bs @ wro1 -> tbs, us, qn
    if (n < N) {
        float ubsv = 0.f;
#pragma unroll 8
        for (int k = 0; k < 64; ++k) ubsv = fmaf(s_f[a][k], s_wro1[k][q], ubsv);
        const float uv = u_in[(long)n * 16 + q];
        const float scv = d_scot ? d_scot[0] : 1.f;
        const float eb = Eb[ngraph[n]] + (Enb ? Enb[n] : 0.f);
        const float qv = scale * eb * scv;
        if (q == 0) qn[n] = qv;
        const float sv = 1.f / (1.f + expf(-uv));
        const float d1 = sv * (1.f + uv * (1.f - sv));
        const float d2 = sv * (1.f - sv) * (2.f + uv * (1.f - 2.f * sv));
        tbs[(long)n * 16 + q] = ubsv * act * d1;
        const float w2v = s_w2[q];
        const float usv = ubsv * scale * w2v * act * d2 + qv * w2v * act * d1;
        us_out[(long)n * 16 + q] = usv;
        s_u[a][q] = usv;
    }
    __syncthreads();

    // Stage 3: f2s = us @ wro1^T
    if (n < N) {
        float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
#pragma unroll
        for (int h = 0; h < 16; ++h) {
            const float usv = s_u[a][h];
            b0 = fmaf(usv, s_wro1[q +  0][h], b0);
            b1 = fmaf(usv, s_wro1[q + 16][h], b1);
            b2 = fmaf(usv, s_wro1[q + 32][h], b2);
            b3 = fmaf(usv, s_wro1[q + 48][h], b3);
        }
        const long base = (long)n * 64;
        f2s[base + q +  0] = b0; f2s[base + q + 16] = b1; f2s[base + q + 32] = b2; f2s[base + q + 48] = b3;
        s_f[a][q +  0] = b0; s_f[a][q + 16] = b1; s_f[a][q + 32] = b2; s_f[a][q + 48] = b3;
    }
    __syncthreads();

    // Stage 4: f1s = f2s @ wp1^T
    if (n < N) {
        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
#pragma unroll 8
        for (int k = 0; k < 64; ++k) {
            const float bk = s_f[a][k];
            o0 = fmaf(bk, s_wp1[q +  0][k], o0);
            o1 = fmaf(bk, s_wp1[q + 16][k], o1);
            o2 = fmaf(bk, s_wp1[q + 32][k], o2);
            o3 = fmaf(bk, s_wp1[q + 48][k], o3);
        }
        const long base = (long)n * 64;
        f1s[base + q +  0] = o0; f1s[base + q + 16] = o1; f1s[base + q + 32] = o2; f1s[base + q + 48] = o3;
    }
}

}  // namespace nops
}  // namespace lst
