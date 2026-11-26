// Radial MLP backward (FP32), pipelined (LST_MLP_PIPE=1): k_mlp_bwd's sums per tile with gh3 = gout W4^T first,
// in 8 stages of 32 output columns (gout [128][32] and W4^T rows [32][64] per stage) through a 3-buffer cp.async
// pipeline over the activation buffers, the tile's x and W1 prefetched with the first stage; then the forward
// recompute with W2 / W3 prefetched into the gradient buffer and W3^T into the weight buffer.
#pragma once

namespace lst {
namespace mlpf {

constexpr int GST = ROWS * 32 + 32 * 64;   // one stage: gout [ROWS][32], W4^T rows [32][64]
constexpr int NSTG = 3;
constexpr int OFF_A = 64 * 64;                  // Ws [64][64] first
constexpr int OFF_B = OFF_A + ROWS * LDB;
constexpr int OFF_G = OFF_A;                    // the stages over As, Bs and the tail
constexpr int OFF_X = OFF_G + NSTG * GST;       // x [ROWS][NB]
constexpr int SMEM_BWDP = (OFF_X + ROWS * NB) * 4;
constexpr int OFF_W1 = OFF_X + ROWS * NB;       // W1 [NB][64] (LST_MLP_NOZ1=1)
constexpr int SMEM_BWDP_NOZ1 = (OFF_W1 + NB * 64) * 4;
static_assert(OFF_B + ROWS * LDB <= OFF_X, "stage buffers must cover As and Bs");
static_assert((GST * 4) % 16 == 0 && (OFF_X * 4) % 16 == 0 && (OFF_W1 * 4) % 16 == 0, "16-byte stage alignment");

__device__ __forceinline__ void cpa16(float* dst, const float* src, bool ok) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int n = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cpa_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// stage s (output columns 32 s ..) of gh3 = gout W4^T into S
__device__ __forceinline__ void bwdp_stage(float* S, const float* __restrict__ gout, const float* __restrict__ WT4,
                                           long p0, int P, int s, int t) {
#pragma unroll
    for (int i = 0; i < ROWS * 8 / NTH; ++i) {
        const int idx = t + i * NTH;
        const int row = idx >> 3, k4 = idx & 7;
        const long p = p0 + (row % TP);
        const bool ok = p < P;
        const long grow = row < TP ? p : P + p;
        cpa16(S + row * 32 + 4 * k4, ok ? gout + grow * O + 32 * s + 4 * k4 : gout, ok);
    }
#pragma unroll
    for (int i = 0; i < 32 * 16 / NTH; ++i) {
        const int idx = t + i * NTH;
        const int k = idx >> 4, q = idx & 15;
        cpa16(S + ROWS * 32 + k * 64 + 4 * q, WT4 + (32 * s + k) * 64 + 4 * q, true);
    }
}

// W [64][64] contiguous -> Sd
__device__ __forceinline__ void cpa_w64(float* Sd, const float* __restrict__ W, int t) {
#pragma unroll
    for (int i = 0; i < 64 * 16 / NTH; ++i) {
        const int idx = t + i * NTH;
        cpa16(Sd + 4 * idx, W + 4 * idx, true);
    }
}

__global__ void __launch_bounds__(NTH, 1) k_mlp_bwdp(const float* __restrict__ EF, const float* __restrict__ DEF,
                                                      const int* __restrict__ pe0, const float* __restrict__ W1,
                                                      const float* __restrict__ W2, const float* __restrict__ W3,
                                                      const float* __restrict__ WT, const float* __restrict__ gout,
                                                      float* dW1, float* dW2, float* dW3, float act, int P,
                                                      int pf = 0) {
    extern __shared__ float4 smem4[];
    float* sm = reinterpret_cast<float*>(smem4);
    float* Ws = sm;
    float* As = sm + OFF_A;
    float* Bs = sm + OFF_B;
    float* Gb = sm + OFF_G;
    float* Xs = sm + OFF_X;
    const int t = threadIdx.x;
    const int rg = t / 16, cg = t % 16;
    const float* WT4 = WT + 2 * H * H;
    float d2[4][4], d3[4][4], d1[2];
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) d2[u][v] = d3[u][v] = 0.f;
    d1[0] = d1[1] = 0.f;
    const int ntiles = (P + TP - 1) / TP;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const long p0 = (long)tile * TP;
        float z1[8][4], z2[8][4], z3[8][4], g[8][4];
        __syncthreads();   // the previous tile's reads of every buffer are done
        {   // group 0: x (value rows EF, tangent rows DEF), W1, stage 0
            const int row = t >> 1, h = t & 1;
            const long p = p0 + (row % TP);
            const bool ok = p < P;
            const float* src = ok ? (row < TP ? EF : DEF) + (long)pe0[p] * NB + 4 * h : EF;
            cpa16(Xs + row * NB + 4 * h, src, ok);
            if (t < NB * 16) cpa16(Ws + 4 * t, W1 + 4 * t, true);
        }
        bwdp_stage(Gb, gout, WT4, p0, P, 0, t);
        cpa_commit();
        bwdp_stage(Gb + GST, gout, WT4, p0, P, 1, t);
        cpa_commit();
        // ---- gh3 = gout W4^T
        zero(g);
#pragma unroll 1
        for (int s = 0; s < O / 32; ++s) {
            cpa_wait<1>();
            __syncthreads();   // stage s landed for everyone; stage s - 1's buffer is free
            if (s + 2 < O / 32) bwdp_stage(Gb + ((s + 2) % NSTG) * GST, gout, WT4, p0, P, s + 2, t);
            cpa_commit();
            const float* S = Gb + (s % NSTG) * GST;
            gemm_tile<32, 32>(S, S + ROWS * 32, rg, cg, g);
        }
        cpa_wait<0>();
        __syncthreads();   // stages read; x, W1 landed
        if (pf && tile + (int)gridDim.x < ntiles) {   // the next tile's gout rows into L2 (128-byte lines)
            const long q0 = (long)(tile + gridDim.x) * TP;
            const long nrow = min((long)TP, (long)P - q0);
            const int nl = (int)(nrow * O * 4 / 128);    // lines per half
            for (int i = t; i < 2 * nl; i += NTH) {
                const float* a = gout + (i < nl ? q0 : P + q0) * O + (long)(i % nl) * 32;
                asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
            }
        }
        // ---- forward recompute, keeping z1, z2, z3; W2, W3 into Bs (free until gz3)
        cpa_w64(Bs, W2, t);
        cpa_w64(Bs + 64 * 64, W3, t);
        cpa_commit();
        zero(z1);
        gemm_tile<NB, NB>(Xs, Ws, rg, cg, z1);
        store_act<LDB>(As, z1, act, rg, cg);   // h1
        cpa_wait<0>();
        __syncthreads();   // h1 stored; W2, W3 landed; W1 read
        cpa_w64(Ws, WT + H * H, t);   // W3^T
        cpa_commit();
        zero(z2);
        gemm_tile<64, LDB>(As, Bs, rg, cg, z2);
        __syncthreads();
        store_act<LDB>(As, z2, act, rg, cg);   // h2
        __syncthreads();
        zero(z3);
        gemm_tile<64, LDB>(As, Bs + 64 * 64, rg, cg, z3);
        act_bwd(z3, g, act);   // g = gz3
        cpa_wait<0>();
        __syncthreads();       // W3 read; W3^T landed
        store_tile<LDB>(Bs, g, rg, cg);
        __syncthreads();
        wgrad64(As, Bs, rg, cg, d3);              // dW3 += h2^T gz3
        zero(g);
        gemm_tile<64, LDB>(Bs, Ws, rg, cg, g);    // gh2 = gz3 W3^T
        act_bwd(z2, g, act);                      // gz2
        __syncthreads();
        cpa_w64(Ws, WT, t);                       // W2^T
        cpa_commit();
        store_tile<LDB>(Bs, g, rg, cg);
        store_act<LDB>(As, z1, act, rg, cg);      // h1
        cpa_wait<0>();
        __syncthreads();
        wgrad64(As, Bs, rg, cg, d2);              // dW2 += h1^T gz2
        zero(g);
        gemm_tile<64, LDB>(Bs, Ws, rg, cg, g);    // gh1 = gz2 W2^T
        act_bwd(z1, g, act);                      // gz1
        __syncthreads();
        store_tile<LDB>(Bs, g, rg, cg);
        __syncthreads();
        {   // dW1 [8][64] += x^T gz1: thread t -> a = t / 32, b = 2 (t % 32) + {0, 1}
            const int a = t / 32, b0 = 2 * (t % 32);
#pragma unroll 4
            for (int r = 0; r < ROWS; ++r) {
                const float xa = Xs[r * NB + a];
                d1[0] += xa * Bs[r * LDB + b0];
                d1[1] += xa * Bs[r * LDB + b0 + 1];
            }
        }
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

// Swizzled stage s (output columns 32 s ..) of gh3 = gout W4^T into S (conflict-free on 32-wide gout & WT4)
__device__ __forceinline__ void bwdp_stage_swz(float* S, const float* __restrict__ gout, const float* __restrict__ WT4,
                                               long p0, int P, int s, int t) {
#pragma unroll
    for (int i = 0; i < ROWS * 8 / NTH; ++i) {
        const int idx = t + i * NTH;
        const int row = idx >> 3, k4 = idx & 7;
        const int ks = k4 ^ (((row >> 2) & 1) << 2);
        const long p = p0 + (row % TP);
        const bool ok = p < P;
        const long grow = row < TP ? p : P + p;
        cpa16(S + row * 32 + 4 * ks, ok ? gout + grow * O + 32 * s + 4 * k4 : gout, ok);
    }
#pragma unroll
    for (int i = 0; i < 32 * 16 / NTH; ++i) {
        const int idx = t + i * NTH;
        const int k = idx >> 4, q = idx & 15;
        const int qs = q ^ ((k & 1) << 2);
        cpa16(S + ROWS * 32 + k * 64 + 4 * qs, WT4 + (32 * s + k) * 64 + 4 * q, true);
    }
}

// LST_MLP_NOZ1=1: keep W1 resident in smem (OFF_W1), drop z1 from live registers (3 arrays za/zb/g instead of 4),
// recompute z1 = Xs @ W1s (K=8) during W2^T cp.async, and swizzle gh3 stage buffers to avoid 32-stride bank conflicts.
__global__ void __launch_bounds__(NTH, 1) k_mlp_bwdp_noz1(const float* __restrict__ EF, const float* __restrict__ DEF,
                                                            const int* __restrict__ pe0, const float* __restrict__ W1,
                                                            const float* __restrict__ W2, const float* __restrict__ W3,
                                                            const float* __restrict__ WT, const float* __restrict__ gout,
                                                            float* dW1, float* dW2, float* dW3, float act, int P,
                                                            int pf = 0) {
    extern __shared__ float4 smem4[];
    float* sm = reinterpret_cast<float*>(smem4);
    float* Ws = sm;
    float* As = sm + OFF_A;
    float* Bs = sm + OFF_B;
    float* Gb = sm + OFF_G;
    float* Xs = sm + OFF_X;
    float* W1s = sm + OFF_W1;
    const int t = threadIdx.x;
    const int rg = t / 16, cg = t % 16;
    const float* WT4 = WT + 2 * H * H;
    if (t < NB * 16) *reinterpret_cast<float4*>(W1s + 4 * t) = *reinterpret_cast<const float4*>(W1 + 4 * t);
    float d2[4][4], d3[4][4], d1[2];
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) d2[u][v] = d3[u][v] = 0.f;
    d1[0] = d1[1] = 0.f;
    const int ntiles = (P + TP - 1) / TP;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const long p0 = (long)tile * TP;
        float za[8][4], zb[8][4], g[8][4];
        __syncthreads();   // tile 0: W1s visible; tile > 0: previous tile's reads done
        {   // group 0: x (value rows EF, tangent rows DEF) and stage 0
            const int row = t >> 1, h = t & 1;
            const long p = p0 + (row % TP);
            const bool ok = p < P;
            const float* src = ok ? (row < TP ? EF : DEF) + (long)pe0[p] * NB + 4 * h : EF;
            cpa16(Xs + row * NB + 4 * h, src, ok);
        }
        bwdp_stage_swz(Gb, gout, WT4, p0, P, 0, t);
        cpa_commit();
        bwdp_stage_swz(Gb + GST, gout, WT4, p0, P, 1, t);
        cpa_commit();
        // ---- gh3 = gout W4^T
        zero(g);
#pragma unroll 1
        for (int s = 0; s < O / 32; ++s) {
            cpa_wait<1>();
            __syncthreads();
            if (s + 2 < O / 32) bwdp_stage_swz(Gb + ((s + 2) % NSTG) * GST, gout, WT4, p0, P, s + 2, t);
            cpa_commit();
            const float* S = Gb + (s % NSTG) * GST;
            gemm_tile_swz<32, 32, true>(S, S + ROWS * 32, rg, cg, g);
        }
        cpa_wait<0>();
        __syncthreads();   // stages read; x landed
        if (pf && tile + (int)gridDim.x < ntiles) {
            const long q0 = (long)(tile + gridDim.x) * TP;
            const long nrow = min((long)TP, (long)P - q0);
            const int nl = (int)(nrow * O * 4 / 128);
            for (int i = t; i < 2 * nl; i += NTH) {
                const float* a = gout + (i < nl ? q0 : P + q0) * O + (long)(i % nl) * 32;
                asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
            }
        }
        // ---- forward recompute: za = z1 -> overwritten by z2, zb = z3; W2, W3 into Bs
        cpa_w64(Bs, W2, t);
        cpa_w64(Bs + 64 * 64, W3, t);
        cpa_commit();
        zero(za);
        gemm_tile<NB, NB>(Xs, W1s, rg, cg, za);
        store_act<LDB>(As, za, act, rg, cg);   // h1
        cpa_wait<0>();
        __syncthreads();   // h1 stored; W2, W3 landed
        cpa_w64(Ws, WT + H * H, t);   // W3^T
        cpa_commit();
        zero(za);
        gemm_tile<64, LDB>(As, Bs, rg, cg, za);   // z2
        __syncthreads();
        store_act<LDB>(As, za, act, rg, cg);   // h2
        __syncthreads();
        zero(zb);
        gemm_tile<64, LDB>(As, Bs + 64 * 64, rg, cg, zb);   // z3
        act_bwd(zb, g, act);   // g = gz3
        cpa_wait<0>();
        __syncthreads();       // W3 read; W3^T landed
        store_tile<LDB>(Bs, g, rg, cg);
        __syncthreads();
        wgrad64(As, Bs, rg, cg, d3);              // dW3 += h2^T gz3
        zero(g);
        gemm_tile<64, LDB>(Bs, Ws, rg, cg, g);    // gh2 = gz3 W3^T
        act_bwd(za, g, act);                      // gz2 (za = z2)
        __syncthreads();
        cpa_w64(Ws, WT, t);                       // W2^T
        cpa_commit();
        store_tile<LDB>(Bs, g, rg, cg);
        zero(za);
        gemm_tile<NB, NB>(Xs, W1s, rg, cg, za);   // recompute z1 while W2^T loads
        store_act<LDB>(As, za, act, rg, cg);      // h1
        cpa_wait<0>();
        __syncthreads();
        wgrad64(As, Bs, rg, cg, d2);              // dW2 += h1^T gz2
        zero(g);
        gemm_tile<64, LDB>(Bs, Ws, rg, cg, g);    // gh1 = gz2 W2^T
        act_bwd(za, g, act);                      // gz1 (za = z1)
        __syncthreads();
        store_tile<LDB>(Bs, g, rg, cg);
        __syncthreads();
        {   // dW1 [8][64] += x^T gz1
            const int a = t / 32, b0 = 2 * (t % 32);
#pragma unroll 4
            for (int r = 0; r < ROWS; ++r) {
                const float xa = Xs[r * NB + a];
                d1[0] += xa * Bs[r * LDB + b0];
                d1[1] += xa * Bs[r * LDB + b0 + 1];
            }
        }
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

}  // namespace mlpf
}  // namespace lst
