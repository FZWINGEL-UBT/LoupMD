// Row-major SIMT SGEMM C[M, N] = A[M, K] * B[K, N] + beta * C: 128 x 128 x 16 block tiles, 256 threads with 8 x 8
// outputs each, 3-stage cp.async pipeline (A kept m-major: a warp's fragment reads are 2-address broadcasts; B
// k-major), 126 registers and 48 KB shared memory, 2 blocks (16 warps) per SM. One more autotune candidate for
// sgemm_rm: on the tall-skinny products here (N = 128-384, K = 128-384) cuBLAS' SIMT kernels reach ~50% of the FMA
// pipe at 200+ registers and 8 warps per SM. Products with a transposed B use a transposed copy of B (always a model
// weight; see sgemm_rm). Every output is one in-order fmaf chain over k, as in cuBLAS' SIMT kernels (bit-identical
// results on the tested shapes).
#include <cuda_runtime.h>
#include <cstddef>

namespace loup {

namespace {

constexpr int kBK = 16, kStages = 3;
template <int BN>
constexpr int smem_bytes() { return kStages * (128 * kBK + kBK * BN) * (int)sizeof(float); }  // 48 KB / 36 KB

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool full) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    const int n = full ? 16 : 0;  // 0 source bytes: zero-fill (rows >= M, columns >= N)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_one_pending() { asm volatile("cp.async.wait_group %0;" ::"n"(kStages - 2)); }

// TN: 8 (128 x 128 tiles) or 4 (128 x 64 tiles, for N = 192 without padding). FULL_N: N % (16 TN) == 0, no
// column guards.
template <int TN, bool FULL_N>
__global__ void __launch_bounds__(256, 2) sgemm128_kernel(int M, int N, int K, const float* __restrict__ A,
                                                          const float* __restrict__ B, float* __restrict__ C,
                                                          float beta) {
    extern __shared__ __align__(16) float smem[];
    float(*As)[128][kBK] = reinterpret_cast<float(*)[128][kBK]>(smem);
    constexpr int BN = 16 * TN;
    float(*Bs)[kBK][BN] = reinterpret_cast<float(*)[kBK][BN]>(smem + kStages * 128 * kBK);
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.x * 128, n0 = blockIdx.y * BN;
    // Copy roles: A rows ar and ar + 64 (one 16-byte k chunk each); B rows bk (and bk + 8 for BN = 128), one 16-byte
    // n chunk each.
    const int ar = tid >> 2, ac = (tid & 3) * 4;
    const int bk = TN == 8 ? tid >> 5 : tid >> 4, bc = TN == 8 ? (tid & 31) * 4 : (tid & 15) * 4;
    const bool a_ok0 = m0 + ar < M, a_ok1 = m0 + ar + 64 < M;
    const bool b_ok = FULL_N || n0 + bc < N;
    const float* a0 = A + (size_t)(a_ok0 ? m0 + ar : 0) * K + ac;
    const float* a1 = A + (size_t)(a_ok1 ? m0 + ar + 64 : 0) * K + ac;
    const float* b0 = B + (size_t)bk * N + (b_ok ? n0 + bc : 0);
    const int nk = K / kBK;
    auto issue = [&](int kt) {
        const int st = kt % kStages;
        cp_async16(&As[st][ar][ac], a0 + kt * kBK, a_ok0);
        cp_async16(&As[st][ar + 64][ac], a1 + kt * kBK, a_ok1);
        cp_async16(&Bs[st][bk][bc], b0 + (size_t)kt * kBK * N, b_ok);
        if (TN == 8) cp_async16(&Bs[st][bk + 8][bc], b0 + (size_t)(kt * kBK + 8) * N, b_ok);
    };
#pragma unroll
    for (int st = 0; st < kStages - 1; ++st) {
        if (st < nk) issue(st);
        cp_async_commit();
    }
    float acc[8][TN];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    for (int kt = 0; kt < nk; ++kt) {
        cp_async_wait_one_pending();
        __syncthreads();  // stage kt landed for every thread; stage kt - 1 is free for reuse
        if (kt + kStages - 1 < nk) issue(kt + kStages - 1);
        cp_async_commit();
        const int st = kt % kStages;
#pragma unroll
        for (int kk = 0; kk < kBK; kk += 4) {
            float4 af[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) af[i] = *reinterpret_cast<const float4*>(&As[st][(i < 4 ? 0 : 60) + ty * 4 + i][kk]);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float bv[TN];
                const float4 bl = *reinterpret_cast<const float4*>(&Bs[st][kk + q][tx * 4]);
                bv[0] = bl.x; bv[1] = bl.y; bv[2] = bl.z; bv[3] = bl.w;
                if (TN == 8) {
                    const float4 bh = *reinterpret_cast<const float4*>(&Bs[st][kk + q][(TN == 8 ? 64 : 0) + tx * 4]);
                    bv[TN - 4] = bh.x; bv[TN - 3] = bh.y; bv[TN - 2] = bh.z; bv[TN - 1] = bh.w;
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float a = q == 0 ? af[i].x : q == 1 ? af[i].y : q == 2 ? af[i].z : af[i].w;
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a, bv[j], acc[i][j]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = m0 + (i < 4 ? 0 : 60) + ty * 4 + i;
        if (r >= M) continue;
#pragma unroll
        for (int hh = 0; hh < TN / 4; ++hh) {
            const int n = n0 + hh * 64 + tx * 4;
            if (!FULL_N && n >= N) continue;
            float* cp = C + (size_t)r * N + n;
            float4 v = make_float4(acc[i][4 * hh], acc[i][4 * hh + 1], acc[i][4 * hh + 2], acc[i][4 * hh + 3]);
            if (beta != 0.0f) {
                const float4 o = *reinterpret_cast<const float4*>(cp);
                v.x = fmaf(beta, o.x, v.x);
                v.y = fmaf(beta, o.y, v.y);
                v.z = fmaf(beta, o.z, v.z);
                v.w = fmaf(beta, o.w, v.w);
            }
            *reinterpret_cast<float4*>(cp) = v;
        }
    }
}

__global__ void transpose_kernel(int R, int Ccols, const float* __restrict__ in, float* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (c < Ccols) out[(size_t)c * R + r] = in[(size_t)r * Ccols + c];
}

}  // namespace

// Prefer the maximum shared-memory carveout so two blocks (48 / 36 KB) fit per SM.
void sgemm128_init() {
    cudaFuncSetAttribute(sgemm128_kernel<8, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(sgemm128_kernel<8, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(sgemm128_kernel<4, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

// Requirements (checked by the caller): K % 16 == 0, N % 4 == 0, lda = K, ldb = ldc = N, A / B / C 16-byte aligned.
void sgemm128(int M, int N, int K, const float* A, const float* B, float* C, float beta, cudaStream_t stream) {
    if ((N % 128) != 0 && (N % 64) == 0) {  // N = 192: 128 x 64 tiles instead of a half-empty 128 x 128 tile
        sgemm128_kernel<4, true><<<dim3((M + 127) / 128, N / 64), 256, smem_bytes<64>(), stream>>>(M, N, K, A, B, C, beta);
        return;
    }
    const dim3 grid((M + 127) / 128, (N + 127) / 128);
    if ((N % 128) == 0) sgemm128_kernel<8, true><<<grid, 256, smem_bytes<128>(), stream>>>(M, N, K, A, B, C, beta);
    else                sgemm128_kernel<8, false><<<grid, 256, smem_bytes<128>(), stream>>>(M, N, K, A, B, C, beta);
}

// out[C, R] = in[R, C]^T (row-major).
void transpose(int R, int Ccols, const float* in, float* out, cudaStream_t stream) {
    transpose_kernel<<<dim3((Ccols + 127) / 128, R), 128, 0, stream>>>(R, Ccols, in, out);
}

}  // namespace loup
