// Tensor-core variants of the fused radial MLPs (mlp_fused.cuh) for the reduced-precision modes (LST_MATH=tf32 /
// bf16): the same products with mma.sync (TF32 m16n8k8 or bf16 m16n8k16, operands rounded from FP32 shared memory
// at the fragment loads, FP32 accumulation); activations, storage and the weight-gradient atomics stay FP32.
//
// Tile: TP = 64 edge pairs, 128 rows. Physical row pr: warp m = pr / 16 owns rows 16 m .. 16 m + 15, in which
// rows 0..7 are the value rows of pairs 8 m .. 8 m + 7 and rows 8..15 their tangent rows. An m16n8 accumulator
// then holds a value element (c0, c1: row g = lane / 4) and the tangent element of the same pair and column
// (c2, c3: row g + 8), so the SiLU and its derivative stay in one thread as in the FP32 kernels.
// Shared tiles are [rows][64] FP32 with an XOR swizzle on the column (activations: by row, weights: by k).
#pragma once

#include <cuda_bf16.h>

namespace lst {
namespace mlpt {

constexpr int TP = 64, ROWS = 128, NTH = 256;
constexpr int NB = 8, H = 64, O = 256;
constexpr int TF32 = 0, BF16 = 1;

__device__ __forceinline__ int swa(int r, int c) { return r * 64 + (c ^ ((r & 7) << 2)); }   // activations
__device__ __forceinline__ int sww(int k, int n) { return k * 64 + (n ^ ((k & 3) << 3)); }   // weights [k][n]

__device__ __forceinline__ uint32_t tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ uint32_t bf2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ void mma_tf32(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// acc[j] (+)= A[16 m .., 0..K) W[0..K, 8 j ..] for the warp's 16 rows (A: swa layout, W: sww layout)
template <int KIND, int K>
__device__ __forceinline__ void warp_gemm(const float* __restrict__ A, const float* __restrict__ W, int m,
                                          float acc[8][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int r0 = 16 * m + g, r1 = r0 + 8;
    if constexpr (KIND == TF32) {
#pragma unroll
        for (int k0 = 0; k0 < K; k0 += 8) {
            const uint32_t a0 = tf32(A[swa(r0, k0 + t)]), a1 = tf32(A[swa(r1, k0 + t)]);
            const uint32_t a2 = tf32(A[swa(r0, k0 + t + 4)]), a3 = tf32(A[swa(r1, k0 + t + 4)]);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                mma_tf32(acc[j], a0, a1, a2, a3, tf32(W[sww(k0 + t, 8 * j + g)]), tf32(W[sww(k0 + t + 4, 8 * j + g)]));
        }
    } else {
#pragma unroll
        for (int k0 = 0; k0 < K; k0 += 16) {
            const int c = k0 + 2 * t;
            const uint32_t a0 = bf2(A[swa(r0, c)], A[swa(r0, c + 1)]), a1 = bf2(A[swa(r1, c)], A[swa(r1, c + 1)]);
            const uint32_t a2 = bf2(A[swa(r0, c + 8)], A[swa(r0, c + 9)]);
            const uint32_t a3 = bf2(A[swa(r1, c + 8)], A[swa(r1, c + 9)]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int n = 8 * j + g;
                mma_bf16(acc[j], a0, a1, a2, a3, bf2(W[sww(c, n)], W[sww(c + 1, n)]),
                         bf2(W[sww(c + 8, n)], W[sww(c + 9, n)]));
            }
        }
    }
}

// D[a][b] (+)= sum_rows S1[r][a] S2[r][b] (both swa layout, 128 rows): warp w -> a in 16 (w / 2) .., b in
// 32 (w % 2) .. (four n8 tiles)
template <int KIND>
__device__ __forceinline__ void warp_wgrad(const float* __restrict__ S1, const float* __restrict__ S2, float d[4][4]) {
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int a0c = 16 * (w >> 1) + g, b0c = 32 * (w & 1) + g;
    if constexpr (KIND == TF32) {
#pragma unroll 4
        for (int k0 = 0; k0 < ROWS; k0 += 8) {
            const int ra = k0 + t, rb = k0 + t + 4;
            const uint32_t a0 = tf32(S1[swa(ra, a0c)]), a1 = tf32(S1[swa(ra, a0c + 8)]);
            const uint32_t a2 = tf32(S1[swa(rb, a0c)]), a3 = tf32(S1[swa(rb, a0c + 8)]);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                mma_tf32(d[j], a0, a1, a2, a3, tf32(S2[swa(ra, b0c + 8 * j)]), tf32(S2[swa(rb, b0c + 8 * j)]));
        }
    } else {
#pragma unroll 2
        for (int k0 = 0; k0 < ROWS; k0 += 16) {
            const int ra = k0 + 2 * t, rb = ra + 8;
            const uint32_t a0 = bf2(S1[swa(ra, a0c)], S1[swa(ra + 1, a0c)]);
            const uint32_t a1 = bf2(S1[swa(ra, a0c + 8)], S1[swa(ra + 1, a0c + 8)]);
            const uint32_t a2 = bf2(S1[swa(rb, a0c)], S1[swa(rb + 1, a0c)]);
            const uint32_t a3 = bf2(S1[swa(rb, a0c + 8)], S1[swa(rb + 1, a0c + 8)]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int n = b0c + 8 * j;
                mma_bf16(d[j], a0, a1, a2, a3, bf2(S2[swa(ra, n)], S2[swa(ra + 1, n)]),
                         bf2(S2[swa(rb, n)], S2[swa(rb + 1, n)]));
            }
        }
    }
}

__device__ __forceinline__ void zero8(float a[8][4]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) a[j][0] = a[j][1] = a[j][2] = a[j][3] = 0.f;
}
__device__ __forceinline__ float sig(float z) { return 1.f / (1.f + __expf(-z)); }

// value (c0, c1) / tangent (c2, c3): h = a silu(z), dh = a silu'(z) dz, in place
__device__ __forceinline__ void act_fwd(float z[8][4], float a) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const float v = z[j][u], s = sig(v);
            z[j][u] = a * v * s;
            z[j][u + 2] = a * s * (1.f + v * (1.f - s)) * z[j][u + 2];
        }
}
// cotangents of z (value, tangent) from those of h (in g) and the pre-activations z
__device__ __forceinline__ void act_bwd(const float z[8][4], float g[8][4], float a) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const float v = z[j][u], dv = z[j][u + 2], s = sig(v);
            const float d1 = s * (1.f + v * (1.f - s));
            const float d2 = s * (1.f - s) * (2.f + v * (1.f - 2.f * s));
            const float gv = g[j][u], gt = g[j][u + 2];
            g[j][u] = a * (d1 * gv + d2 * dv * gt);
            g[j][u + 2] = a * d1 * gt;
        }
}
// the warp's rows of S (swa) from an accumulator; with ACT the activation of z instead of z
template <bool ACT>
__device__ __forceinline__ void store(float* S, const float z[8][4], float a) {
    const int m = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int r0 = 16 * m + g, r1 = r0 + 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float v0 = z[j][0], v1 = z[j][1], d0 = z[j][2], d1 = z[j][3];
        if (ACT) {
            const float s0 = sig(v0), s1 = sig(v1);
            d0 = a * s0 * (1.f + v0 * (1.f - s0)) * d0;
            d1 = a * s1 * (1.f + v1 * (1.f - s1)) * d1;
            v0 = a * v0 * s0;
            v1 = a * v1 * s1;
        }
        const int c = 8 * j + 2 * t;
        *reinterpret_cast<float2*>(S + swa(r0, c)) = make_float2(v0, v1);
        *reinterpret_cast<float2*>(S + swa(r1, c)) = make_float2(d0, d1);
    }
}
// global row of physical row pr of the tile (-1 past P)
__device__ __forceinline__ long grow(int pr, long p0, int P) {
    const long p = p0 + 8 * (pr >> 4) + (pr & 7);
    if (p >= P) return -1;
    return (pr & 8) ? P + p : p;
}
// x tile (cols 0..7 from EF / DEF of pe0[p], cols 8..15 zero for the bf16 k16 step)
__device__ __forceinline__ void load_x(float* S, const float* __restrict__ EF, const float* __restrict__ DEF,
                                       const int* __restrict__ pe0, long p0, int P) {
    for (int idx = threadIdx.x; idx < ROWS * 16; idx += NTH) {
        const int pr = idx >> 4, k = idx & 15;
        const long p = p0 + 8 * (pr >> 4) + (pr & 7);
        float v = 0.f;
        if (k < NB && p < P) {
            const long e = pe0[p];
            v = (pr & 8) ? DEF[e * NB + k] : EF[e * NB + k];
        }
        S[swa(pr, k)] = v;
    }
}
// W [K][ldw] columns c0 .. c0 + 63 -> Ws (sww); rows K .. KP - 1 zero
template <int K, int KP>
__device__ __forceinline__ void load_w(float* Ws, const float* __restrict__ W, int ldw, int c0) {
    for (int idx = threadIdx.x; idx < KP * 16; idx += NTH) {
        const int k = idx >> 4, q = idx & 15;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (k < K) v = *reinterpret_cast<const float4*>(W + (long)k * ldw + c0 + 4 * q);
        *reinterpret_cast<float4*>(Ws + sww(k, 4 * q)) = v;
    }
}
// 128 rows x 64 columns (c0 ..) of a [2P][ld] global array into S (swa); zeros past P
__device__ __forceinline__ void load_rows(float* S, const float* __restrict__ G, int ld, int c0, long p0, int P) {
    for (int idx = threadIdx.x; idx < ROWS * 16; idx += NTH) {
        const int pr = idx >> 4, q = idx & 15;
        const long r = grow(pr, p0, P);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r >= 0) v = *reinterpret_cast<const float4*>(G + r * ld + c0 + 4 * q);
        *reinterpret_cast<float4*>(S + swa(pr, 4 * q)) = v;
    }
}
// the warp's accumulator rows to a [2P][ld] global array at columns c0 ..
__device__ __forceinline__ void put_rows(float* G, int ld, int c0, const float z[8][4], long p0, int P) {
    const int m = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const long p = p0 + 8 * m + g;
    if (p >= P) return;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int c = c0 + 8 * j + 2 * t;
        *reinterpret_cast<float2*>(G + p * ld + c) = make_float2(z[j][0], z[j][1]);
        *reinterpret_cast<float2*>(G + (P + p) * ld + c) = make_float2(z[j][2], z[j][3]);
    }
}

constexpr int SMEM_FWD = (ROWS * 64 + 64 * 64) * 4;
constexpr int SMEM_BWD = (2 * ROWS * 64 + 64 * 64) * 4;

// out [2P, 256] (R rows, then T rows), h3 [2P, 64]: as mlpf::k_mlp_fwd
template <int KIND>
__global__ void __launch_bounds__(NTH, 2) k_fwd(const float* __restrict__ EF, const float* __restrict__ DEF,
                                                 const int* __restrict__ pe0, const float* __restrict__ W1,
                                                 const float* __restrict__ W2, const float* __restrict__ W3,
                                                 const float* __restrict__ W4, float* __restrict__ out,
                                                 float* __restrict__ h3, float act, int P) {
    constexpr int KX = KIND == TF32 ? 8 : 16;
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Ws = As + ROWS * 64;
    const int m = threadIdx.x >> 5;
    const long p0 = (long)blockIdx.x * TP;
    float z[8][4];
    load_x(As, EF, DEF, pe0, p0, P);
    load_w<NB, KX>(Ws, W1, 64, 0);
    __syncthreads();
    zero8(z);
    warp_gemm<KIND, KX>(As, Ws, m, z);
    act_fwd(z, act);
    __syncthreads();
    store<false>(As, z, act);
    load_w<64, 64>(Ws, W2, 64, 0);
    __syncthreads();
    zero8(z);
    warp_gemm<KIND, 64>(As, Ws, m, z);
    act_fwd(z, act);
    __syncthreads();
    store<false>(As, z, act);
    load_w<64, 64>(Ws, W3, 64, 0);
    __syncthreads();
    zero8(z);
    warp_gemm<KIND, 64>(As, Ws, m, z);
    act_fwd(z, act);
    put_rows(h3, H, 0, z, p0, P);
    __syncthreads();
    store<false>(As, z, act);
    for (int q = 0; q < O / 64; ++q) {
        __syncthreads();
        load_w<64, 64>(Ws, W4, O, 64 * q);
        __syncthreads();
        zero8(z);
        warp_gemm<KIND, 64>(As, Ws, m, z);
        put_rows(out, O, 64 * q, z, p0, P);
    }
}

// as mlpf::k_mlp_bwd<false>: gout [2P, 256] -> dW1, dW2, dW3 (atomics; the caller zeroes them), gh3 formed inside;
// WT = [W2^T, W3^T, W4^T] (mlpf::k_mlp_wt). Persistent blocks.
template <int KIND>
__global__ void __launch_bounds__(NTH, 1) k_bwd(const float* __restrict__ EF, const float* __restrict__ DEF,
                                                 const int* __restrict__ pe0, const float* __restrict__ W1,
                                                 const float* __restrict__ W2, const float* __restrict__ W3,
                                                 const float* __restrict__ WT, const float* __restrict__ gout,
                                                 float* dW1, float* dW2, float* dW3, float act, int P) {
    constexpr int KX = KIND == TF32 ? 8 : 16;
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS * 64;
    float* Ws = Bs + ROWS * 64;
    const int tid = threadIdx.x, m = tid >> 5;
    float d2[4][4], d3[4][4], d1[2];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int u = 0; u < 4; ++u) d2[j][u] = d3[j][u] = 0.f;
    d1[0] = d1[1] = 0.f;
    const int ntiles = (P + TP - 1) / TP;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const long p0 = (long)tile * TP;
        float z1[8][4], z2[8][4], z3[8][4], g[8][4];
        __syncthreads();
        load_x(As, EF, DEF, pe0, p0, P);
        load_w<NB, KX>(Ws, W1, 64, 0);
        __syncthreads();
        zero8(z1);
        warp_gemm<KIND, KX>(As, Ws, m, z1);
        __syncthreads();
        store<true>(As, z1, act);                 // h1
        load_w<64, 64>(Ws, W2, 64, 0);
        __syncthreads();
        zero8(z2);
        warp_gemm<KIND, 64>(As, Ws, m, z2);
        __syncthreads();
        store<true>(As, z2, act);                 // h2
        load_w<64, 64>(Ws, W3, 64, 0);
        __syncthreads();
        zero8(z3);
        warp_gemm<KIND, 64>(As, Ws, m, z3);
        zero8(g);                                 // gh3 = gout W4^T (four K slices); As keeps h2
        for (int q = 0; q < O / 64; ++q) {
            __syncthreads();
            load_rows(Bs, gout, O, 64 * q, p0, P);
            load_w<64, 64>(Ws, WT + 2 * H * H + 64 * q * H, 64, 0);
            __syncthreads();
            warp_gemm<KIND, 64>(Bs, Ws, m, g);
        }
        act_bwd(z3, g, act);                      // gz3
        __syncthreads();
        store<false>(Bs, g, act);
        load_w<64, 64>(Ws, WT + H * H, 64, 0);    // W3^T
        __syncthreads();
        warp_wgrad<KIND>(As, Bs, d3);             // dW3 += h2^T gz3
        zero8(g);
        warp_gemm<KIND, 64>(Bs, Ws, m, g);        // gh2
        act_bwd(z2, g, act);                      // gz2
        __syncthreads();
        store<false>(Bs, g, act);
        store<true>(As, z1, act);                 // h1
        load_w<64, 64>(Ws, WT, 64, 0);            // W2^T
        __syncthreads();
        warp_wgrad<KIND>(As, Bs, d2);             // dW2 += h1^T gz2
        zero8(g);
        warp_gemm<KIND, 64>(Bs, Ws, m, g);        // gh1
        act_bwd(z1, g, act);                      // gz1
        __syncthreads();
        store<false>(Bs, g, act);
        load_x(As, EF, DEF, pe0, p0, P);
        __syncthreads();
        {   // dW1 [8][64] += x^T gz1 (FP32): thread -> a = tid / 32, b = 2 (tid % 32) + {0, 1}
            const int a = tid >> 5, b0 = 2 * (tid & 31);
#pragma unroll 4
            for (int r = 0; r < ROWS; ++r) {
                const float xa = As[swa(r, a)];
                d1[0] += xa * Bs[swa(r, b0)];
                d1[1] += xa * Bs[swa(r, b0 + 1)];
            }
        }
    }
    const int w = tid >> 5, lane = tid & 31, gg = lane >> 2, t = lane & 3;
    const int ar = 16 * (w >> 1) + gg, bc = 32 * (w & 1) + 2 * t;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c = bc + 8 * j;
        atomicAdd(dW3 + ar * 64 + c, d3[j][0]);
        atomicAdd(dW3 + ar * 64 + c + 1, d3[j][1]);
        atomicAdd(dW3 + (ar + 8) * 64 + c, d3[j][2]);
        atomicAdd(dW3 + (ar + 8) * 64 + c + 1, d3[j][3]);
        atomicAdd(dW2 + ar * 64 + c, d2[j][0]);
        atomicAdd(dW2 + ar * 64 + c + 1, d2[j][1]);
        atomicAdd(dW2 + (ar + 8) * 64 + c, d2[j][2]);
        atomicAdd(dW2 + (ar + 8) * 64 + c + 1, d2[j][3]);
    }
    atomicAdd(dW1 + (tid >> 5) * 64 + 2 * (tid & 31), d1[0]);
    atomicAdd(dW1 + (tid >> 5) * 64 + 2 * (tid & 31) + 1, d1[1]);
}

}  // namespace mlpt
}  // namespace lst
