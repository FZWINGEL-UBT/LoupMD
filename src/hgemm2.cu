// LOUP_W2T_HALF: the reverse W_2 projections of a kind-4 gate (dgout_l = dmsgf_l W_2[l]^T, MH-1's eight 512 x 512
// adjoints) as two binary16 tensor-core products with FP32 accumulation, the gate's backward in the epilogue. The force
// audit's MH-1 recommendation (loupmd-force-audit-20261120/MH1.md, "two compensated FP16 products").
//
// Each operand X (scaled by a power of two s) is held as two binary16 panels: H = rn(X s) and the amplified
// correction V = rn(H + 64 (X s - H)). With P = H(A) H(B) and Q = V(A) V(B), both accumulated in FP32,
//   A B ~ (P + (Q - P) / 64) / (s_A s_B),
// which recovers both first-order rounding terms with one extra product (the residual is 63 dA dB plus rounding).
// The product kernels write A's panels directly (loup_store_h2 in the generated library source: s_A = 2^12, NaN for an
// element that overflows either half), the weights' panels are made once at load (split_half, s_B per matrix).
//
// Kernel: 128 x 64 block tiles, 4 warps of 64 x 32, K steps of 32 halves through a two-stage cp.async pipeline in
// XOR-swizzled shared memory (24 KB a stage, 48 KB: two blocks per SM), ldmatrix + mma.sync.m16n8k16 (f32 += f16 f16)
// into two accumulator sets. With the gate (DEG = 2l + 1), row tile t starts at atom t (128 / DEG), as sgemm128_gate's
// (sgemm.cu), so no per-atom sum crosses tiles; the combined tile is staged in shared memory and gate_bwd_rows
// (gate_epilogue.cuh) runs gate_bwd_kernel's arithmetic on it. DEG = 0: the plain FP32 store of the combined product
// (tests and timing only).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#include "gate_epilogue.cuh"
#include "generic.hpp"

namespace loup {

namespace {

constexpr int kHM = 128, kHN = 64, kHK = 32;
constexpr int kHTileA = kHM * kHK, kHTileB = kHN * kHK;  // halves of one panel's tile
constexpr int kHStage = 2 * kHTileA + 2 * kHTileB;       // halves of a stage: 12,288 (24 KB)
constexpr int kHSmem = 2 * kHStage * 2;                  // bytes, two stages: 48 KB
constexpr int kHSRow = 72;                               // staged epilogue tile: floats per row (64 + 8 padding)
static_assert(kHM * kHSRow * 4 + 2 * kHM * 4 <= kHSmem, "the staged tile and dD partials fit the pipeline buffers");

// Byte offset of 16-byte chunk j (0..3) of row r in a tile of 64-byte rows: the chunk index XORed with (r / 2) % 4,
// so the 8 rows an ldmatrix reads cover the 32 banks once.
__device__ __forceinline__ int swz(int r, int j) { return r * 64 + ((j ^ ((r >> 1) & 3)) << 4); }

__device__ __forceinline__ void cp16(unsigned s, const void* g, bool full) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(g), "r"(full ? 16 : 0));
}

__device__ __forceinline__ void ldsm4(unsigned addr, unsigned (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// C = A B^T from two-half panels: A [2][M][K] (H then V), B [2][N][K] (the weight W_2[l] as stored, [N][K]); out =
// (P + (Q - P) / 64) oscale. DEG 0: C [M][N] row-major; DEG 2l + 1: gate_bwd_rows on row tiles of whole atoms (C
// unused, M = g.natoms DEG, N = g.C). Requirements (checked by the host side): N % 64 == 0, K % 32 == 0, A and B
// 16-byte aligned.
template <int DEG>
__global__ void __launch_bounds__(128, 2) hgemm2_kernel(int M, int N, int K, const __half* __restrict__ A,
                                                        const __half* __restrict__ B, float oscale,
                                                        float* __restrict__ C, GateArgs g) {
    extern __shared__ __align__(128) unsigned char hsm[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(hsm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
    constexpr int MT = DEG ? kHM / DEG * DEG : kHM;  // rows a tile owns (whole atoms)
    const int m0 = blockIdx.y * MT, n0 = blockIdx.x * kHN;
    const int64_t planeA = (int64_t)M * K, planeB = (int64_t)N * K;

    // Copy roles per stage: A rows (tid >> 2) + 32 i (i = 0..3) and B rows (tid >> 2) + 32 i (i = 0..1), 16-byte
    // chunk tid & 3, both panels: 12 cp.async per thread.
    const int cr = tid >> 2, cj = tid & 3;
    const __half* ag[4];
    bool aok[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = m0 + cr + 32 * i;
        aok[i] = r < M;
        ag[i] = A + (int64_t)(aok[i] ? r : m0) * K + cj * 8;
    }
    const __half* bg = B + (int64_t)(n0 + cr) * K + cj * 8;
    auto stage_a = [&](int st, int p) { return sbase + (unsigned)((st * kHStage + p * kHTileA) * 2); };
    auto stage_b = [&](int st, int p) { return sbase + (unsigned)((st * kHStage + 2 * kHTileA + p * kHTileB) * 2); };
    auto issue = [&](int kt) {
        const int st = kt & 1, k0 = kt * kHK;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
#pragma unroll
            for (int i = 0; i < 4; ++i) cp16(stage_a(st, p) + swz(cr + 32 * i, cj), ag[i] + p * planeA + k0, aok[i]);
#pragma unroll
            for (int i = 0; i < 2; ++i)
                cp16(stage_b(st, p) + swz(cr + 32 * i, cj), bg + p * planeB + (int64_t)32 * i * K + k0, true);
        }
        asm volatile("cp.async.commit_group;");
    };

    float accP[4][4][4], accQ[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) accP[i][j][e] = accQ[i][j][e] = 0.0f;

    // ldmatrix row roles: A (rows lane & 15 of a 16-row tile, k chunk + (lane >> 4)); B (n rows of the matrix
    // q = lane >> 3: n tile (q >> 1), k chunk + (q & 1)).
    const int a_row = wm * 64 + (lane & 15), a_kc = lane >> 4;
    const int b_row = wn * 32 + ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;
    const int nk = K / kHK;
    issue(0);
    for (int kt = 0; kt < nk; ++kt) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();  // stage kt landed for every thread; every warp is done with stage kt - 1
        if (kt + 1 < nk) issue(kt + 1);
        const int st = kt & 1;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned bh[2][4], bv[2][4];  // [n tile pair][b0, b1 of tile 2j; b0, b1 of tile 2j + 1]
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                ldsm4(stage_b(st, 0) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                ldsm4(stage_b(st, 1) + swz(b_row + 16 * j, 2 * kk + b_kc), bv[j]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                unsigned ah[4], av[4];
                ldsm4(stage_a(st, 0) + swz(a_row + 16 * i, 2 * kk + a_kc), ah);
                ldsm4(stage_a(st, 1) + swz(a_row + 16 * i, 2 * kk + a_kc), av);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma16816(accP[i][j], ah, bh[j >> 1][2 * (j & 1)], bh[j >> 1][2 * (j & 1) + 1]);
                    mma16816(accQ[i][j], av, bv[j >> 1][2 * (j & 1)], bv[j >> 1][2 * (j & 1) + 1]);
                }
            }
        }
    }

    // Accumulator element e of (i, j): row wm 64 + 16 i + lane / 4 (+ 8 for e >= 2), column wn 32 + 8 j + 2 (lane % 4)
    // (+ 1 for odd e).
    const int er = wm * 64 + (lane >> 2), ec = wn * 32 + 2 * (lane & 3);
    auto comb = [&](float p, float q) { return fmaf(q - p, 0.015625f, p) * oscale; };
    if constexpr (DEG == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = m0 + er + 16 * i + 8 * h;
                if (r >= M) continue;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    *reinterpret_cast<float2*>(C + (int64_t)r * N + n0 + ec + 8 * j) =
                        make_float2(comb(accP[i][j][2 * h], accQ[i][j][2 * h]),
                                    comb(accP[i][j][2 * h + 1], accQ[i][j][2 * h + 1]));
            }
    } else {
        constexpr int APT = kHM / DEG;
        float* sC = reinterpret_cast<float*>(hsm);
        float* s_acc = sC + kHM * kHSRow;
        float* s_accq = s_acc + kHM;
        __syncthreads();  // every warp is done with the pipeline buffers (no copy in flight: the last wait was 0)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    *reinterpret_cast<float2*>(sC + (er + 16 * i + 8 * h) * kHSRow + ec + 8 * j) =
                        make_float2(comb(accP[i][j][2 * h], accQ[i][j][2 * h]),
                                    comb(accP[i][j][2 * h + 1], accQ[i][j][2 * h + 1]));
        if (tid < APT) s_acc[tid] = s_accq[tid] = 0.0f;
        const int a0 = m0 / DEG, na = min(APT, g.natoms - a0);
        __syncthreads();
        gate_bwd_rows<DEG, 2, kHSRow>(sC, s_acc, s_accq, a0, na, n0 + (tid & 63), tid & 63, tid >> 6, tid, g);
        __syncthreads();
        gate_bwd_dD(s_acc, s_accq, a0, na, tid, g);
    }
}

// The two panels of a weight: out[0 .. n) = rn(w scale), out[n .. 2n) = rn(H + 64 (w scale - H)).
__global__ void split_half_kernel(int64_t n, const float* __restrict__ w, float scale, __half* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = w[i] * scale;
    const __half h = __float2half_rn(a);
    const float hf = __half2float(h);
    out[i] = h;
    out[n + i] = __float2half_rn(fmaf(64.0f, a - hf, hf));
}

template <int DEG>
void launch_hgemm2(int M, int N, int K, const __half* A, const __half* B, float oscale, float* C, const GateArgs& g,
                   cudaStream_t stream) {
    const int rows = DEG ? kHM / DEG * DEG : kHM;
    const dim3 grid(N / kHN, (M + rows - 1) / rows);
    hgemm2_kernel<DEG><<<grid, 128, kHSmem, stream>>>(M, N, K, A, B, oscale, C, g);
}

}  // namespace

// Two blocks (2 x 48 KB) per SM need the largest shared-memory carveout.
void hgemm2_init() {
    cudaFuncSetAttribute(hgemm2_kernel<0>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2_kernel<1>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2_kernel<3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2_kernel<5>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2_kernel<7>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

// The exponent e of the weight panels' scale 2^e: the largest with max |w| 2^e <= 2^15, so both panels stay finite
// (|V| <= |H| (1 + 2^-5)) and as far from binary16's subnormals as that allows. 0 for an all-zero weight.
int hgemm2_weight_exponent(const float* w, int64_t n) {
    float amax = 0.0f;
    for (int64_t i = 0; i < n; ++i) amax = std::fmax(amax, std::fabs(w[i]));
    if (!(amax > 0.0f) || !std::isfinite(amax)) return 0;
    int e = (int)std::floor(std::log2(32768.0 / (double)amax));
    while (std::ldexp((double)amax, e) > 32768.0) --e;
    return e;
}

// out [2 n] halves: the panels of w [n] at scale 2^e (hgemm2_weight_exponent).
void hgemm2_split(int64_t n, const float* w, int e, void* out, cudaStream_t stream) {
    split_half_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(n, w, std::ldexp(1.0f, e),
                                                                       static_cast<__half*>(out));
}

// dgout_l = A B^T with degree l's gate backward (gate_bwd_rows) for M = g.natoms (2l + 1) rows; l = -1: the plain
// product into C (tests). A: two panels [2][M][K] at scale 2^12 (loup_store_h2), B: two panels [2][N][K] at 2^e_B;
// oscale = 2^-(12 + e_B).
void hgemm2_gate(int l, int M, int N, int K, const void* A, const void* B, float oscale, float* C, const GateArgs& g,
                 cudaStream_t stream) {
    const __half* a = static_cast<const __half*>(A);
    const __half* b = static_cast<const __half*>(B);
    switch (l) {
        case -1: launch_hgemm2<0>(M, N, K, a, b, oscale, C, g, stream); break;
        case 0: launch_hgemm2<1>(M, N, K, a, b, oscale, C, g, stream); break;
        case 1: launch_hgemm2<3>(M, N, K, a, b, oscale, C, g, stream); break;
        case 2: launch_hgemm2<5>(M, N, K, a, b, oscale, C, g, stream); break;
        default: launch_hgemm2<7>(M, N, K, a, b, oscale, C, g, stream); break;
    }
}

}  // namespace loup
