// Radial MLP backward (FP32) with 512 threads per tile of TP edge pairs: k_mlp_bwd's algorithm with each thread
// owning 2 value rows and their 2 tangent rows (instead of 4 + 4), so 16 warps per SM at <= 128 registers
// (k_mlp_bwd: 8 warps at 255 registers). Same shared memory, tiles, sums and atomics layout per weight entry.
#pragma once

#include "mlp_fused.cuh"

namespace lst {
namespace mlpf2 {

using mlpf::H;
using mlpf::LDB;
using mlpf::NB;
using mlpf::O;
using mlpf::ROWS;
using mlpf::SMEM_BWD;
using mlpf::TP;
constexpr int NTH = 512;

__device__ __forceinline__ int rowof(int rg, int i) { return i < 2 ? 2 * rg + i : TP + 2 * rg + i - 2; }

template <int K, int LDA>
__device__ __forceinline__ void gemm4(const float* __restrict__ A, const float* __restrict__ B, int rg, int cg,
                                      float acc[4][4]) {
#pragma unroll 4
    for (int k = 0; k < K; k += 4) {
        float4 b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) b[j] = *reinterpret_cast<const float4*>(B + (k + j) * 64 + 4 * cg);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 a = *reinterpret_cast<const float4*>(A + rowof(rg, i) * LDA + k);
            acc[i][0] += a.x * b[0].x + a.y * b[1].x + a.z * b[2].x + a.w * b[3].x;
            acc[i][1] += a.x * b[0].y + a.y * b[1].y + a.z * b[2].y + a.w * b[3].y;
            acc[i][2] += a.x * b[0].z + a.y * b[1].z + a.z * b[2].z + a.w * b[3].z;
            acc[i][3] += a.x * b[0].w + a.y * b[1].w + a.z * b[2].w + a.w * b[3].w;
        }
    }
}

__device__ __forceinline__ void zero4(float acc[4][4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
}

// as mlpf::act_bwd: rows i (value) and i + 2 (tangent)
__device__ __forceinline__ void act_bwd4(const float z[4][4], float g[4][4], float a) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float v = z[i][j], dv = z[i + 2][j], s = mlpf::sig(v);
            const float d1 = s * (1.f + v * (1.f - s));
            const float d2 = s * (1.f - s) * (2.f + v * (1.f - 2.f * s));
            const float gv = g[i][j], gt = g[i + 2][j];
            g[i][j] = a * (d1 * gv + d2 * dv * gt);
            g[i + 2][j] = a * d1 * gt;
        }
}

template <int LDA>
__device__ __forceinline__ void store_act4(float* S, const float z[4][4], float a, int rg, int cg) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        float hv[4], ht[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float v = z[i][j], s = mlpf::sig(v);
            hv[j] = a * v * s;
            ht[j] = a * s * (1.f + v * (1.f - s)) * z[i + 2][j];
        }
        *reinterpret_cast<float4*>(S + (2 * rg + i) * LDA + 4 * cg) = make_float4(hv[0], hv[1], hv[2], hv[3]);
        *reinterpret_cast<float4*>(S + (TP + 2 * rg + i) * LDA + 4 * cg) = make_float4(ht[0], ht[1], ht[2], ht[3]);
    }
}

template <int LDA>
__device__ __forceinline__ void store_tile4(float* S, const float v[4][4], int rg, int cg) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
        *reinterpret_cast<float4*>(S + rowof(rg, i) * LDA + 4 * cg) = make_float4(v[i][0], v[i][1], v[i][2], v[i][3]);
}

template <int LDA>
__device__ __forceinline__ void load_x(float* S, const float* __restrict__ EF, const float* __restrict__ DEF,
                                       const int* __restrict__ pe0, long p0, int P) {
    for (int idx = threadIdx.x; idx < ROWS * NB; idx += NTH) {
        const int row = idx / NB, k = idx % NB;
        const long p = p0 + row % TP;
        float v = 0.f;
        if (p < P) {
            const long e = pe0[p];
            v = row < TP ? EF[e * NB + k] : DEF[e * NB + k];
        }
        S[row * LDA + k] = v;
    }
}

template <int K>
__device__ __forceinline__ void load_w(float* Ws, const float* __restrict__ W, int ldw, int c0) {
    for (int idx = threadIdx.x; idx < K * 16; idx += NTH) {
        const int k = idx / 16, q = idx % 16;
        *reinterpret_cast<float4*>(Ws + k * 64 + 4 * q) =
            *reinterpret_cast<const float4*>(W + (long)k * ldw + c0 + 4 * q);
    }
}

// d[u][v] (+)= sum_rows S1[row][2 ag + u] S2[row][4 bg + v]
__device__ __forceinline__ void wgrad(const float* __restrict__ S1, const float* __restrict__ S2, int ag, int bg,
                                      float d[2][4]) {
#pragma unroll 4
    for (int r = 0; r < ROWS; ++r) {
        const float2 a = *reinterpret_cast<const float2*>(S1 + r * LDB + 2 * ag);
        const float4 b = *reinterpret_cast<const float4*>(S2 + r * LDB + 4 * bg);
        const float av[2] = {a.x, a.y}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) d[u][v] += av[u] * bv[v];
    }
}

__global__ void __launch_bounds__(NTH, 1) k_mlp_bwd2(const float* __restrict__ EF, const float* __restrict__ DEF,
                                                      const int* __restrict__ pe0, const float* __restrict__ W1,
                                                      const float* __restrict__ W2, const float* __restrict__ W3,
                                                      const float* __restrict__ WT, const float* __restrict__ gout,
                                                      float* dW1, float* dW2, float* dW3, float act, int P) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS * LDB;
    float* Ws = Bs + ROWS * LDB;
    const int t = threadIdx.x;
    const int rg = t / 16, cg = t % 16;
    float d2[2][4], d3[2][4], d1 = 0.f;
#pragma unroll
    for (int u = 0; u < 2; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) d2[u][v] = d3[u][v] = 0.f;
    const int ntiles = (P + TP - 1) / TP;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const long p0 = (long)tile * TP;
        float z1[4][4], z2[4][4], z3[4][4], g[4][4];
        __syncthreads();
        load_x<LDB>(As, EF, DEF, pe0, p0, P);
        load_w<NB>(Ws, W1, 64, 0);
        __syncthreads();
        zero4(z1);
        gemm4<NB, LDB>(As, Ws, rg, cg, z1);
        __syncthreads();
        store_act4<LDB>(As, z1, act, rg, cg);
        load_w<64>(Ws, W2, 64, 0);
        __syncthreads();
        zero4(z2);
        gemm4<64, LDB>(As, Ws, rg, cg, z2);
        __syncthreads();
        store_act4<LDB>(As, z2, act, rg, cg);
        load_w<64>(Ws, W3, 64, 0);
        __syncthreads();
        zero4(z3);
        gemm4<64, LDB>(As, Ws, rg, cg, z3);
        // gh3 = gout W4^T (K = 256 in four slices); As holds h2
        zero4(g);
        for (int q = 0; q < O / 64; ++q) {
            __syncthreads();
            for (int idx = t; idx < ROWS * 16; idx += NTH) {
                const int row = idx / 16, k4 = idx % 16;
                const long p = p0 + row % TP;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (p < P) {
                    const long grow = row < TP ? p : P + p;
                    v = *reinterpret_cast<const float4*>(gout + grow * O + 64 * q + 4 * k4);
                }
                *reinterpret_cast<float4*>(Bs + row * LDB + 4 * k4) = v;
            }
            load_w<64>(Ws, WT + 2 * H * H + 64 * q * H, 64, 0);
            __syncthreads();
            gemm4<64, LDB>(Bs, Ws, rg, cg, g);
        }
        act_bwd4(z3, g, act);                     // gz3
        __syncthreads();
        store_tile4<LDB>(Bs, g, rg, cg);
        load_w<64>(Ws, WT + H * H, 64, 0);
        __syncthreads();
        wgrad(As, Bs, rg, cg, d3);                // dW3 += h2^T gz3
        zero4(g);
        gemm4<64, LDB>(Bs, Ws, rg, cg, g);        // gh2 = gz3 W3^T
        act_bwd4(z2, g, act);                     // gz2
        __syncthreads();
        store_tile4<LDB>(Bs, g, rg, cg);
        store_act4<LDB>(As, z1, act, rg, cg);     // h1
        load_w<64>(Ws, WT, 64, 0);
        __syncthreads();
        wgrad(As, Bs, rg, cg, d2);                // dW2 += h1^T gz2
        zero4(g);
        gemm4<64, LDB>(Bs, Ws, rg, cg, g);        // gh1 = gz2 W2^T
        act_bwd4(z1, g, act);                     // gz1
        __syncthreads();
        store_tile4<LDB>(Bs, g, rg, cg);
        load_x<LDB>(As, EF, DEF, pe0, p0, P);
        __syncthreads();
        {   // dW1 [8][64] += x^T gz1: thread t -> (t / 64, t % 64)
            const int a = t / 64, b = t % 64;
#pragma unroll 4
            for (int r = 0; r < ROWS; ++r) d1 += As[r * LDB + a] * Bs[r * LDB + b];
        }
    }
#pragma unroll
    for (int u = 0; u < 2; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            atomicAdd(dW3 + (2 * rg + u) * 64 + 4 * cg + v, d3[u][v]);
            atomicAdd(dW2 + (2 * rg + u) * 64 + 4 * cg + v, d2[u][v]);
        }
    atomicAdd(dW1 + t, d1);
}

}  // namespace mlpf2
}  // namespace lst
