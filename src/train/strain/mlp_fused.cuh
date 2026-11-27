// Fused radial MLPs of the training library (FP32): value rows and their r-tangent rows of TP edge pairs per tile.
//   z1 = x W1, h1 = a silu(z1); z2 = h1 W2, h2 = ...; z3 = h2 W3, h3 = ...; out = h3 W4     (value rows)
//   dz_k = dh_{k-1} W_k, dh_k = a silu'(z_k) dz_k, dout = dh3 W4                              (tangent rows)
// Shapes: x [.., NB = 8], W1 [8, 64], W2 W3 [64, 64], W4 [64, 256]. Tile rows: value rows 0..TP-1, tangent rows
// TP..2 TP - 1. Thread (rg, cg) of 256 owns value rows 4 rg + i, tangent rows TP + 4 rg + i (i < 4) and columns
// 4 cg + j (j < 4) of every 64-wide product, so a value / tangent pair stays in one thread (activations fused).
#pragma once

namespace lst {
namespace mlpf {

constexpr int TP = 64;          // pairs per tile
constexpr int ROWS = 2 * TP;    // value + tangent rows
constexpr int NTH = 256;
constexpr int NB = 8, H = 64, O = 256;

// acc[i][j] (+)= sum_k A[row_i][k] B[k][4 cg + j], A row-major [ROWS][lda], B row-major [K][64]
template <int K, int LDA>
__device__ __forceinline__ void gemm_tile(const float* __restrict__ A, const float* __restrict__ B, int rg, int cg,
                                          float acc[8][4]) {
#pragma unroll 4
    for (int k = 0; k < K; k += 4) {
        float4 b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) b[j] = *reinterpret_cast<const float4*>(B + (k + j) * 64 + 4 * cg);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = i < 4 ? 4 * rg + i : TP + 4 * rg + i - 4;
            const float4 a = *reinterpret_cast<const float4*>(A + row * LDA + k);
            acc[i][0] += a.x * b[0].x + a.y * b[1].x + a.z * b[2].x + a.w * b[3].x;
            acc[i][1] += a.x * b[0].y + a.y * b[1].y + a.z * b[2].y + a.w * b[3].y;
            acc[i][2] += a.x * b[0].z + a.y * b[1].z + a.z * b[2].z + a.w * b[3].z;
            acc[i][3] += a.x * b[0].w + a.y * b[1].w + a.z * b[2].w + a.w * b[3].w;
        }
    }
}

__device__ __forceinline__ void zero(float acc[8][4]) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
}

__device__ __forceinline__ float sig(float z) { return 1.f / (1.f + __expf(-z)); }

// h = a silu(z) (value rows), dh = a silu'(z) dz (tangent rows), in place
__device__ __forceinline__ void act_fwd(float z[8][4], float a) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float v = z[i][j], s = sig(v);
            z[i][j] = a * v * s;
            z[i + 4][j] = a * s * (1.f + v * (1.f - s)) * z[i + 4][j];
        }
}

// cotangents of z (value, tangent) from those of h (in g, overwritten) and the pre-activations z
__device__ __forceinline__ void act_bwd(const float z[8][4], float g[8][4], float a) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float v = z[i][j], dv = z[i + 4][j], s = sig(v);
            const float d1 = s * (1.f + v * (1.f - s));
            const float d2 = s * (1.f - s) * (2.f + v * (1.f - 2.f * s));
            const float gv = g[i][j], gt = g[i + 4][j];
            g[i][j] = a * (d1 * gv + d2 * dv * gt);
            g[i + 4][j] = a * d1 * gt;
        }
}

// h of the tile (from z) into S [ROWS][LDA] columns 4 cg ..
template <int LDA>
__device__ __forceinline__ void store_act(float* S, const float z[8][4], float a, int rg, int cg) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float hv[4], ht[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float v = z[i][j], s = sig(v);
            hv[j] = a * v * s;
            ht[j] = a * s * (1.f + v * (1.f - s)) * z[i + 4][j];
        }
        *reinterpret_cast<float4*>(S + (4 * rg + i) * LDA + 4 * cg) = make_float4(hv[0], hv[1], hv[2], hv[3]);
        *reinterpret_cast<float4*>(S + (TP + 4 * rg + i) * LDA + 4 * cg) = make_float4(ht[0], ht[1], ht[2], ht[3]);
    }
}

template <int LDA>
__device__ __forceinline__ void store_tile(float* S, const float v[8][4], int rg, int cg) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = i < 4 ? 4 * rg + i : TP + 4 * rg + i - 4;
        *reinterpret_cast<float4*>(S + row * LDA + 4 * cg) = make_float4(v[i][0], v[i][1], v[i][2], v[i][3]);
    }
}

// x tile: value rows EF[pe0[p]], tangent rows DEF[pe0[p]] -> S [ROWS][LDA] columns 0..7 (zeros past P)
template <int LDA>
__device__ __forceinline__ void load_x(float* S, const float* __restrict__ EF, const float* __restrict__ DEF,
                                       const int* __restrict__ pe0, long p0, int P) {
    for (int idx = threadIdx.x; idx < ROWS * NB; idx += NTH) {
        const int row = idx / NB, k = idx % NB;
        const int pr = row % TP;
        const long p = p0 + pr;
        float v = 0.f;
        if (p < P) {
            const long e = pe0[p];
            v = row < TP ? EF[e * NB + k] : DEF[e * NB + k];
        }
        S[row * LDA + k] = v;
    }
}

// W [K][64] block (row stride ldw, column offset c0) -> Ws [K][64]
template <int K>
__device__ __forceinline__ void load_w(float* Ws, const float* __restrict__ W, int ldw, int c0) {
    for (int idx = threadIdx.x; idx < K * 16; idx += NTH) {
        const int k = idx / 16, q = idx % 16;
        *reinterpret_cast<float4*>(Ws + k * 64 + 4 * q) =
            *reinterpret_cast<const float4*>(W + (long)k * ldw + c0 + 4 * q);
    }
}

// Ws[k][c] = W[c][k0 + k] for W [64][ldw] (transposed block: the backward's W^T)
__device__ __forceinline__ void load_wt(float* Ws, const float* __restrict__ W, int ldw, int k0) {
    for (int idx = threadIdx.x; idx < 64 * 64; idx += NTH) {
        const int c = idx / 64, k = idx % 64;
        Ws[k * 64 + c] = W[(long)c * ldw + k0 + k];
    }
}

// WT2 [64][64], WT3 [64][64], WT4 [256][64] from W2, W3 [64][64], W4 [64][256]
__global__ void k_mlp_wt(const float* __restrict__ W2, const float* __restrict__ W3, const float* __restrict__ W4,
                         float* __restrict__ WT) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 2 * H * H) {
        const int m = i / (H * H), j = i % (H * H), c = j / H, r = j % H;
        WT[i] = (m ? W3 : W2)[r * H + c];
    } else if (i < 2 * H * H + O * H) {
        const int j = i - 2 * H * H, c = j / H, r = j % H;
        WT[i] = W4[r * O + c];
    }
}
constexpr int WT_SIZE = 2 * H * H + O * H;

// ---------------------------------------------------------------------------- forward
// out [2P, 256]: R rows [0, P), T rows [P, 2P); h3 [2P, 64] (value rows, then tangent rows) for dW4
constexpr int LDF = 64;
constexpr int SMEM_FWD = (ROWS * LDF + 64 * 64) * 4;

__global__ void __launch_bounds__(NTH, 2) k_mlp_fwd(const float* __restrict__ EF, const float* __restrict__ DEF,
                                                     const int* __restrict__ pe0, const float* __restrict__ W1,
                                                     const float* __restrict__ W2, const float* __restrict__ W3,
                                                     const float* __restrict__ W4, float* __restrict__ out,
                                                     float* __restrict__ h3, float act, int P) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Ws = As + ROWS * LDF;
    const int rg = threadIdx.x / 16, cg = threadIdx.x % 16;
    const long p0 = (long)blockIdx.x * TP;
    float z[8][4];
    load_x<LDF>(As, EF, DEF, pe0, p0, P);
    load_w<NB>(Ws, W1, 64, 0);
    __syncthreads();
    zero(z);
    gemm_tile<NB, LDF>(As, Ws, rg, cg, z);
    act_fwd(z, act);
    __syncthreads();
    store_tile<LDF>(As, z, rg, cg);
    load_w<64>(Ws, W2, 64, 0);
    __syncthreads();
    zero(z);
    gemm_tile<64, LDF>(As, Ws, rg, cg, z);
    act_fwd(z, act);
    __syncthreads();
    store_tile<LDF>(As, z, rg, cg);
    load_w<64>(Ws, W3, 64, 0);
    __syncthreads();
    zero(z);
    gemm_tile<64, LDF>(As, Ws, rg, cg, z);
    act_fwd(z, act);
    __syncthreads();
    store_tile<LDF>(As, z, rg, cg);
    // h3 to global
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const long p = p0 + 4 * rg + (i & 3);
        if (p < P) {
            const long row = i < 4 ? p : P + p;
            *reinterpret_cast<float4*>(h3 + row * H + 4 * cg) = make_float4(z[i][0], z[i][1], z[i][2], z[i][3]);
        }
    }
    for (int q = 0; q < O / 64; ++q) {
        if (q) __syncthreads();
        load_w<64>(Ws, W4, O, 64 * q);
        __syncthreads();
        zero(z);
        gemm_tile<64, LDF>(As, Ws, rg, cg, z);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const long p = p0 + 4 * rg + (i & 3);
            if (p < P) {
                const long row = i < 4 ? p : P + p;
                *reinterpret_cast<float4*>(out + row * O + 64 * q + 4 * cg) =
                    make_float4(z[i][0], z[i][1], z[i][2], z[i][3]);
            }
        }
    }
}

// Swizzled shared-memory helpers: XOR float4 column index with ((row >> 2) & 1) << 2 so even/odd rg halves
// of a warp access disjoint 16-bank halves without padding LDF beyond 64 (preserving 2 blocks/SM in k_mlp_fwd).
template <int K, int LDA, bool SWZ_B>
__device__ __forceinline__ void gemm_tile_swz(const float* __restrict__ A, const float* __restrict__ B, int rg, int cg,
                                              float acc[8][4]) {
    const int swzA = (rg & 1) << 2;
    const int cg0 = cg;
    const int cg1 = SWZ_B ? (cg ^ 4) : cg;
#pragma unroll 4
    for (int k = 0; k < K; k += 4) {
        float4 b[4];
        b[0] = *reinterpret_cast<const float4*>(B + (k + 0) * 64 + 4 * cg0);
        b[1] = *reinterpret_cast<const float4*>(B + (k + 1) * 64 + 4 * cg1);
        b[2] = *reinterpret_cast<const float4*>(B + (k + 2) * 64 + 4 * cg0);
        b[3] = *reinterpret_cast<const float4*>(B + (k + 3) * 64 + 4 * cg1);
        const int ka = ((k >> 2) ^ swzA) << 2;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = i < 4 ? 4 * rg + i : TP + 4 * rg + i - 4;
            const float4 a = *reinterpret_cast<const float4*>(A + row * LDA + ka);
            acc[i][0] += a.x * b[0].x + a.y * b[1].x + a.z * b[2].x + a.w * b[3].x;
            acc[i][1] += a.x * b[0].y + a.y * b[1].y + a.z * b[2].y + a.w * b[3].y;
            acc[i][2] += a.x * b[0].z + a.y * b[1].z + a.z * b[2].z + a.w * b[3].z;
            acc[i][3] += a.x * b[0].w + a.y * b[1].w + a.z * b[2].w + a.w * b[3].w;
        }
    }
}

template <int LDA>
__device__ __forceinline__ void store_tile_swz(float* S, const float v[8][4], int rg, int cg) {
    const int col = (cg ^ ((rg & 1) << 2)) << 2;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = i < 4 ? 4 * rg + i : TP + 4 * rg + i - 4;
        *reinterpret_cast<float4*>(S + row * LDA + col) = make_float4(v[i][0], v[i][1], v[i][2], v[i][3]);
    }
}

template <int LDA>
__device__ __forceinline__ void load_x_swz(float* S, const float* __restrict__ EF, const float* __restrict__ DEF,
                                           const int* __restrict__ pe0, long p0, int P) {
    const int row = threadIdx.x >> 1, h = threadIdx.x & 1;
    const long p = p0 + (row % TP);
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (p < P) {
        const long e = pe0[p];
        v = *reinterpret_cast<const float4*>((row < TP ? EF : DEF) + e * NB + 4 * h);
    }
    const int col = (h ^ (((row >> 2) & 1) << 2)) << 2;
    *reinterpret_cast<float4*>(S + row * LDA + col) = v;
}

template <int K>
__device__ __forceinline__ void load_w_swz(float* Ws, const float* __restrict__ W, int ldw, int c0) {
    for (int idx = threadIdx.x; idx < K * 16; idx += NTH) {
        const int k = idx >> 4, q = idx & 15;
        const int qs = q ^ ((k & 1) << 2);
        *reinterpret_cast<float4*>(Ws + k * 64 + 4 * qs) =
            *reinterpret_cast<const float4*>(W + (long)k * ldw + c0 + 4 * q);
    }
}

__global__ void __launch_bounds__(NTH, 2) k_mlp_fwd_swz(const float* __restrict__ EF, const float* __restrict__ DEF,
                                                         const int* __restrict__ pe0, const float* __restrict__ W1,
                                                         const float* __restrict__ W2, const float* __restrict__ W3,
                                                         const float* __restrict__ W4, float* __restrict__ out,
                                                         float* __restrict__ h3, float act, int P) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Ws = As + ROWS * LDF;
    const int rg = threadIdx.x / 16, cg = threadIdx.x % 16;
    const long p0 = (long)blockIdx.x * TP;
    float z[8][4];
    load_x_swz<LDF>(As, EF, DEF, pe0, p0, P);
    load_w_swz<NB>(Ws, W1, 64, 0);
    __syncthreads();
    zero(z);
    gemm_tile_swz<NB, LDF, true>(As, Ws, rg, cg, z);
    act_fwd(z, act);
    __syncthreads();
    store_tile_swz<LDF>(As, z, rg, cg);
    load_w_swz<64>(Ws, W2, 64, 0);
    __syncthreads();
    zero(z);
    gemm_tile_swz<64, LDF, true>(As, Ws, rg, cg, z);
    act_fwd(z, act);
    __syncthreads();
    store_tile_swz<LDF>(As, z, rg, cg);
    load_w_swz<64>(Ws, W3, 64, 0);
    __syncthreads();
    zero(z);
    gemm_tile_swz<64, LDF, true>(As, Ws, rg, cg, z);
    act_fwd(z, act);
    __syncthreads();
    store_tile_swz<LDF>(As, z, rg, cg);
    // h3 to global
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const long p = p0 + 4 * rg + (i & 3);
        if (p < P) {
            const long row = i < 4 ? p : P + p;
            *reinterpret_cast<float4*>(h3 + row * H + 4 * cg) = make_float4(z[i][0], z[i][1], z[i][2], z[i][3]);
        }
    }
    for (int q = 0; q < O / 64; ++q) {
        if (q) __syncthreads();
        load_w_swz<64>(Ws, W4, O, 64 * q);
        __syncthreads();
        zero(z);
        gemm_tile_swz<64, LDF, true>(As, Ws, rg, cg, z);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const long p = p0 + 4 * rg + (i & 3);
            if (p < P) {
                const long row = i < 4 ? p : P + p;
                *reinterpret_cast<float4*>(out + row * O + 64 * q + 4 * cg) =
                    make_float4(z[i][0], z[i][1], z[i][2], z[i][3]);
            }
        }
    }
}

// ---------------------------------------------------------------------------- backward
// gout [2P, 256] (cotangents of R rows, then T rows) -> dW1 [8, 64], dW2, dW3 [64, 64] (atomically added; the
// caller zeroes them); dW4 = h3^T gout is left to a GEMM. Persistent: each block loops over tiles and keeps its
// weight-gradient partials in registers.
constexpr int LDB = 68;
constexpr int SMEM_BWD = (2 * ROWS * LDB + 64 * 64) * 4;

// d[a][b] (+)= sum_rows S1[row][a] S2[row][b] for a = 4 ag + u, b = 4 bg + v (ag = t / 16, bg = t % 16)
__device__ __forceinline__ void wgrad64(const float* __restrict__ S1, const float* __restrict__ S2, int ag, int bg,
                                        float d[4][4]) {
#pragma unroll 4
    for (int r = 0; r < ROWS; ++r) {
        const float4 a = *reinterpret_cast<const float4*>(S1 + r * LDB + 4 * ag);
        const float4 b = *reinterpret_cast<const float4*>(S2 + r * LDB + 4 * bg);
        const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) d[u][v] += av[u] * bv[v];
    }
}

template <bool SPLIT>
__global__ void __launch_bounds__(NTH, 1) k_mlp_bwd(const float* __restrict__ EF, const float* __restrict__ DEF,
                                                     const int* __restrict__ pe0, const float* __restrict__ W1,
                                                     const float* __restrict__ W2, const float* __restrict__ W3,
                                                     const float* __restrict__ WT, const float* __restrict__ gout,
                                                     const float* __restrict__ gh3, float* dW1, float* dW2, float* dW3,
                                                     float act, int P, const float* W1b = nullptr,
                                                     const float* W2b = nullptr, const float* W3b = nullptr,
                                                     const float* WTb = nullptr, const float* goutb = nullptr,
                                                     float* dW1b = nullptr, float* dW2b = nullptr,
                                                     float* dW3b = nullptr) {
    extern __shared__ float4 smem4[];
    float* As = reinterpret_cast<float*>(smem4);
    float* Bs = As + ROWS * LDB;
    float* Ws = Bs + ROWS * LDB;
    const int t = threadIdx.x;
    const int rg = t / 16, cg = t % 16;
    float d2[4][4], d3[4][4], d1[2];
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) d2[u][v] = d3[u][v] = 0.f;
    d1[0] = d1[1] = 0.f;
    const int ntiles = (P + TP - 1) / TP;
    const int ntot = W1b ? 2 * ntiles : ntiles;
    bool cur_b = false;    // partials belong to the second MLP
    for (int tile = blockIdx.x; tile < ntot; tile += gridDim.x) {
        const bool sb = tile >= ntiles;
        if (sb && !cur_b) {   // crossing to MLP b: flush MLP a's partials
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    atomicAdd(dW3 + (4 * rg + u) * 64 + 4 * cg + v, d3[u][v]);
                    atomicAdd(dW2 + (4 * rg + u) * 64 + 4 * cg + v, d2[u][v]);
                    d3[u][v] = d2[u][v] = 0.f;
                }
            atomicAdd(dW1 + (t / 32) * 64 + 2 * (t % 32), d1[0]);
            atomicAdd(dW1 + (t / 32) * 64 + 2 * (t % 32) + 1, d1[1]);
            d1[0] = d1[1] = 0.f;
            cur_b = true;
        }
        const float* cW1 = sb ? W1b : W1;
        const float* cW2 = sb ? W2b : W2;
        const float* cW3 = sb ? W3b : W3;
        const float* cWT = sb ? WTb : WT;
        const float* cgout = sb ? goutb : gout;
        const long p0 = (long)(sb ? tile - ntiles : tile) * TP;
        float z1[8][4], z2[8][4], z3[8][4], g[8][4];
        // ---- recompute the forward, keeping z1, z2, z3
        __syncthreads();
        load_x<LDB>(As, EF, DEF, pe0, p0, P);
        load_w<NB>(Ws, cW1, 64, 0);
        __syncthreads();
        zero(z1);
        gemm_tile<NB, LDB>(As, Ws, rg, cg, z1);
        __syncthreads();
        store_act<LDB>(As, z1, act, rg, cg);
        load_w<64>(Ws, cW2, 64, 0);
        __syncthreads();
        zero(z2);
        gemm_tile<64, LDB>(As, Ws, rg, cg, z2);
        __syncthreads();
        store_act<LDB>(As, z2, act, rg, cg);
        load_w<64>(Ws, cW3, 64, 0);
        __syncthreads();
        zero(z3);
        gemm_tile<64, LDB>(As, Ws, rg, cg, z3);
        // ---- gh3 = gout W4^T (K = 256 in four slices); As still holds h2
        zero(g);
        if constexpr (SPLIT) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const long p = p0 + 4 * rg + (i & 3);
                if (p < P) {
                    const long row = i < 4 ? p : P + p;
                    const float4 v = *reinterpret_cast<const float4*>(gh3 + row * H + 4 * cg);
                    g[i][0] = v.x; g[i][1] = v.y; g[i][2] = v.z; g[i][3] = v.w;
                }
            }
        } else
        for (int q = 0; q < O / 64; ++q) {
            __syncthreads();
            for (int idx = t; idx < ROWS * 16; idx += NTH) {
                const int row = idx / 16, k4 = idx % 16;
                const long p = p0 + row % TP;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (p < P) {
                    const long grow = row < TP ? p : P + p;
                    v = *reinterpret_cast<const float4*>(cgout + grow * O + 64 * q + 4 * k4);
                }
                *reinterpret_cast<float4*>(Bs + row * LDB + 4 * k4) = v;
            }
            load_w<64>(Ws, cWT + 2 * H * H + 64 * q * H, 64, 0);
            __syncthreads();
            gemm_tile<64, LDB>(Bs, Ws, rg, cg, g);
        }
        act_bwd(z3, g, act);                      // g = gz3
        __syncthreads();
        store_tile<LDB>(Bs, g, rg, cg);
        load_w<64>(Ws, cWT + H * H, 64, 0);
        __syncthreads();
        wgrad64(As, Bs, rg, cg, d3);              // dW3 += h2^T gz3
        zero(g);
        gemm_tile<64, LDB>(Bs, Ws, rg, cg, g);    // gh2 = gz3 W3^T
        act_bwd(z2, g, act);                      // gz2
        __syncthreads();
        store_tile<LDB>(Bs, g, rg, cg);
        store_act<LDB>(As, z1, act, rg, cg);      // h1
        load_w<64>(Ws, cWT, 64, 0);
        __syncthreads();
        wgrad64(As, Bs, rg, cg, d2);              // dW2 += h1^T gz2
        zero(g);
        gemm_tile<64, LDB>(Bs, Ws, rg, cg, g);    // gh1 = gz2 W2^T
        act_bwd(z1, g, act);                      // gz1
        __syncthreads();
        store_tile<LDB>(Bs, g, rg, cg);
        load_x<LDB>(As, EF, DEF, pe0, p0, P);
        __syncthreads();
        {   // dW1 [8][64] += x^T gz1: thread t -> a = t / 32, b = 2 (t % 32) + {0, 1}
            const int a = t / 32, b0 = 2 * (t % 32);
#pragma unroll 4
            for (int r = 0; r < ROWS; ++r) {
                const float xa = As[r * LDB + a];
                d1[0] += xa * Bs[r * LDB + b0];
                d1[1] += xa * Bs[r * LDB + b0 + 1];
            }
        }
    }
    if (cur_b) {
        dW1 = dW1b;
        dW2 = dW2b;
        dW3 = dW3b;
    }
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            atomicAdd(dW3 + (4 * rg + u) * 64 + 4 * cg + v, d3[u][v]);
            atomicAdd(dW2 + (4 * rg + u) * 64 + 4 * cg + v, d2[u][v]);
        }
    atomicAdd(dW1 + (t / 32) * 64 + 2 * (t % 32), d1[0]);
    atomicAdd(dW1 + (t / 32) * 64 + 2 * (t % 32) + 1, d1[1]);
}

// gh3 [2P, 64] = gout W4^T; dW4 [64, 256] += h3^T gout (atomics; the caller zeroes it). Persistent blocks.
__global__ void __launch_bounds__(NTH, 1) k_mlp_top(const float* __restrict__ h3, const float* __restrict__ WT,
                                                     const float* __restrict__ gout, float* __restrict__ gh3,
                                                     float* dW4, int P) {
    extern __shared__ float4 smem4[];
    float* Hs = reinterpret_cast<float*>(smem4);
    float* Gs = Hs + ROWS * LDB;
    float* Ws = Gs + ROWS * LDB;
    const int t = threadIdx.x;
    const int rg = t / 16, cg = t % 16;
    float d4[4][4][4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) d4[q][u][v] = 0.f;
    const int ntiles = (P + TP - 1) / TP;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const long p0 = (long)tile * TP;
        float g[8][4];
        zero(g);
        __syncthreads();
        for (int idx = t; idx < ROWS * 16; idx += NTH) {
            const int row = idx / 16, k4 = idx % 16;
            const long p = p0 + row % TP;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (p < P) v = *reinterpret_cast<const float4*>(h3 + (row < TP ? p : P + p) * H + 4 * k4);
            *reinterpret_cast<float4*>(Hs + row * LDB + 4 * k4) = v;
        }
#pragma unroll
        for (int q = 0; q < O / 64; ++q) {
            if (q) __syncthreads();
            for (int idx = t; idx < ROWS * 16; idx += NTH) {
                const int row = idx / 16, k4 = idx % 16;
                const long p = p0 + row % TP;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (p < P) v = *reinterpret_cast<const float4*>(gout + (row < TP ? p : P + p) * O + 64 * q + 4 * k4);
                *reinterpret_cast<float4*>(Gs + row * LDB + 4 * k4) = v;
            }
            load_w<64>(Ws, WT + 2 * H * H + 64 * q * H, 64, 0);
            __syncthreads();
            gemm_tile<64, LDB>(Gs, Ws, rg, cg, g);
            wgrad64(Hs, Gs, rg, cg, d4[q]);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const long p = p0 + 4 * rg + (i & 3);
            if (p < P) {
                const long row = i < 4 ? p : P + p;
                *reinterpret_cast<float4*>(gh3 + row * H + 4 * cg) = make_float4(g[i][0], g[i][1], g[i][2], g[i][3]);
            }
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) atomicAdd(dW4 + (4 * rg + u) * O + 64 * q + 4 * cg + v, d4[q][u][v]);
}

}  // namespace mlpf
}  // namespace lst
