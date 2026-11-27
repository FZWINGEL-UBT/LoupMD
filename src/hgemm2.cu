// LOUP_W2T_HALF: the reverse W_2 projections of a kind-4 gate (dgout_l = dmsgf_l W_2[l]^T, MH-1's eight 512 x 512
// adjoints) as binary16 tensor-core products with explicit cross terms, the gate's backward in the epilogue. The force
// audit's MH-1 recommendation (loupmd-force-audit-20261120/MH1.md, "two compensated FP16 products"), in the form of the
// 20 November research list's idea 2 (ledger gen-w2t-two-half).
//
// Each operand x (scaled by a power of two s) is held as two binary16 panels: H = rn(x s) and its correction
// L = rn((x s - H) 2^11), |L| <= |H|. With
//   P = H(A) H(B)                              (FP32 accumulation),
//   X = H(A) [L(B) 2^-c] + L(A) [H(B) 2^-c]    (binary16 accumulation, X 2^(c - 11) added to P after every 32 of K),
//   A B ~ (P + X 2^(c - 11)) / (s_A s_B),
// dropping only L(A) L(B) (about 2^-22 of the product) and the cross sums' binary16 rounding. The shift c = 22 (kHC)
// is fixed by the worst case: |H(A)| <= 2^16 (the product kernels' overflow guard) and |H(B)| <= 2^15, so the 64 cross
// products of one 32-wide K step stay within 2^15 < 65,504 whatever the data. Both MMA accumulators restart from zero
// after every 32 of K, and that step's P + X 2^(c - 11) goes into an FP32 running sum with round-to-nearest adds: the
// tensor cores' FP32 accumulation truncates, so one MMA chain over the whole K shrank |result| coherently (MH-1's
// Sr-contact0.8 cell: mean signed error -5.5e-7 of |result| against FP32's +-1e-8, rms 1.6-1.7x FP32's, and the force
// gate failed at 2.0x its bound); restarted, -5.5e-8 and 0.35-0.41x (on the engine's own panels). What remains is
// the MMA's own final rounding toward zero (-0.8e-8 to -2.3e-8, emulated); P is jammed after each MMA (jam4), which
// makes that rounding unbiased (+1.8e-9 to +5.8e-9, emulated; FP32 +-0.4e-9 to 2.7e-9). The first form of
// this path (P and Q = V(A) V(B) with V = rn(H + 64 (x s - H)), both FP32-accumulated) had a 2^-16 floor, 36-49x
// FP32's (emulated, .tmp_o/screen_w2t2.py of the branch). On GeForce cards the binary16-accumulated MMA runs at twice
// the FP32-accumulated rate (5060 Ti: 105.3 / 52.7 TFLOP/s), so the three products take P and Q's tensor-core time,
// but not their energy: with the gate the kernels run at the 5060 Ti's 180 W limit, where this form clocks 2.68 GHz
// against P / Q's 2.78 and takes 4-6% longer (the plain product ties). Where both rates are equal (the professional
// cards, by NVIDIA's tables) the three products take 1.5x P and Q's time.
// The product kernels write A's panels directly (loup_store_h2 in the generated library source: s_A = 2^12, NaN for an
// element that overflows), the weights' panels are made once at load (split_half: H at s_B per matrix, and L(B) 2^-c);
// H(B) 2^-c is formed in registers (a power of two; binary16 subnormals are kept).
//
// Kernel: 128 x 64 block tiles, 4 warps of 64 x 32, K steps of 32 halves through a two-stage cp.async pipeline in
// XOR-swizzled shared memory (24 KB a stage, 48 KB: two blocks per SM), ldmatrix + mma.sync.m16n8k16 (f32 += f16 f16
// for P, f16 += f16 f16 for X) into an FP32 and a binary16 accumulator set, restarted every K step, and an FP32
// running sum (236-238 registers, two blocks per SM; restarting every 16 of K took 255 and gained little; list 26
// restarted H H alone at every MMA, as hgemm2f_kernel's RV 1: 255 registers and 2.8% slower, ledger
// ideas-20261122-list26). With the gate (DEG = 2l + 1), row tile t starts at atom t (128 / DEG), as sgemm128_gate's
// (sgemm.cu), so no per-atom sum crosses tiles; the combined tile is staged in shared memory and gate_bwd_tile runs
// gate_bwd_kernel's arithmetic on it. DEG = 0: the plain FP32 store of the combined product (tests and timing only).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "gate_epilogue.cuh"
#include "generic.hpp"
#ifdef LOUP_STUDENT_LAUNCH_BENCH
#include "../tools/dev/student_projection_group.cuh"
#endif

namespace loup {

namespace {

constexpr int kHM = 128, kHN = 64, kHK = 32;
constexpr int kHTileA = kHM * kHK, kHTileB = kHN * kHK;  // halves of one panel's tile
constexpr int kHStage = 2 * kHTileA + 2 * kHTileB;       // halves of a stage: 12,288 (24 KB)
constexpr int kHSmem = 2 * kHStage * 2;                  // bytes, two stages: 48 KB
constexpr int kHSRow = 72;                               // staged epilogue tile: floats per row (64 + 8 padding)
constexpr int kHC = 22;                                  // the cross term's shift c (see the top)
constexpr unsigned kHCScale2 = 0x00040004u;              // 2^-22 = 4 x 2^-24 (a binary16 subnormal) in both halves
static_assert(kHM * kHSRow * 4 + 2 * kHM * 4 <= kHSmem, "the staged tile and the atoms' s / res0 rows fit");

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

// The same MMA with binary16 accumulation (d: two f16x2 registers, rows lane / 4 and lane / 4 + 8).
__device__ __forceinline__ void mma16816h(unsigned (&d)[2], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The jam: the lowest significand bit of an FP32-accumulated MMA's result set (one LOP3 per element). The MMA rounds
// its sum toward zero (5060 Ti probe of 21 November 2026: the 16 products and c aligned to 2^(emax - 25), each
// truncated, then the sum rounded toward zero to FP32), so every result is short by half an ulp on average and a sum
// of many shrinks in one direction. Setting the bit makes the rounding unbiased (von Neumann rounding): the result moves
// up by an ulp exactly when the dropped bits were at least as likely above half as below. Applied to d itself, before
// it reaches the round-to-nearest running sum; a half-ulp correction added to that sum instead is rounded away.
__device__ __forceinline__ void jam4(float (&d)[4]) {
#pragma unroll
    for (int e = 0; e < 4; ++e) d[e] = __int_as_float(__float_as_int(d[e]) | 1);
}

__device__ __forceinline__ unsigned hmul2u(unsigned x, unsigned y) {
    const __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&x), *reinterpret_cast<const __half2*>(&y));
    return *reinterpret_cast<const unsigned*>(&r);
}

// hgemm2_kernel's gate backward is gate_bwd_tile (gate_epilogue.cuh; list 26 item 2, ledger gen-h2-gate-tile). The
// tile's dD after it: thread ai < na sums atom ai's 64 column partials (gate_bwd_tile_sum) and adds -beta s^2 part -
// beta s partq, one global atomic.
template <int DEG, int SROW>
__device__ __forceinline__ void gate_bwd_tile_dD(const float* sC, const float* sS, int a0, int na, int tid,
                                                 const GateArgs& g) {
    if (tid >= na) return;
    float p = 0.0f, q = 0.0f;
    gate_bwd_tile_sum<DEG, SROW>(sC, tid, g, p, q);
    const float s = sS[tid];
    atomicAdd(g.dD + a0 + tid, -g.beta * s * s * p - g.beta * s * q);
}

// C = A B^T from two-half panels: A [2][M][K] (H then L), B [2][N][K] (the weight W_2[l] as stored, [N][K]: H, then
// L 2^-c); out = (the running sum of P + X 2^(c - 11) over the K steps) oscale. DEG 0: C [M][N] row-major; DEG 2l + 1:
// gate_bwd_tile on row tiles of whole atoms (C unused, M = g.natoms DEG, N = g.C). Requirements (checked by the host
// side): N % 64 == 0, K % 32 == 0, A and B 16-byte aligned.
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

template <int DEG, bool RF = false>
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

    // acc: the running sum (FP32, round-to-nearest adds); accP, accX: H H and X of the current 32 of K in the MMAs' own
    // accumulators, restarted from zero after each (see the top: the tensor cores' FP32 accumulation truncates).
    float acc[4][4][4], accP[4][4][4];
    unsigned accX[4][4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = accP[i][j][e] = 0.0f;
            accX[i][j][0] = accX[i][j][1] = 0u;
        }

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
        if constexpr (DEG > 0) {
            // list 11: the gate backward's operands (the tile's lin_l / res_l rows, its atoms' lin0 / res0 l-slices) are
            // prefetched into L2 four K steps before the loop ends, so the epilogue's reads hit L2
            if (kt == (nk > 4 ? nk - 4 : 0)) {
                constexpr int l = DEG / 2;
                const int a0 = m0 / DEG, na = min(MT / DEG, g.natoms - a0), rows = min(MT, M - m0);
                if (DEG > 1)
                    for (int li = tid; li < 2 * rows; li += 128) {
                        const int64_t off = (int64_t)(m0 + (li >> 1)) * N + n0 + 32 * (li & 1);
                        if (g.lin_l) prefetch_l2(g.lin_l + off);
                        if (g.res_l) prefetch_l2(g.res_l + off);
                    }
                for (int li = tid; li < 2 * na; li += 128) {
                    const int64_t off = (int64_t)(a0 + (li >> 1)) * 4 * N + l * N + n0 + 32 * (li & 1);
                    prefetch_l2(g.lin0 + off);
                    if (g.res0 && !g.res0_sp) prefetch_l2(g.res0 + off);
                }
            }
        }
        const int st = kt & 1;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            // [n tile pair][b0, b1 of tile 2j; b0, b1 of tile 2j + 1]: H(B), L(B) 2^-c, H(B) 2^-c
            unsigned bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                ldsm4(stage_b(st, 0) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                ldsm4(stage_b(st, 1) + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHCScale2);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                unsigned ah[4], al[4];
                ldsm4(stage_a(st, 0) + swz(a_row + 16 * i, 2 * kk + a_kc), ah);
                ldsm4(stage_a(st, 1) + swz(a_row + 16 * i, 2 * kk + a_kc), al);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int jp = j >> 1, jb = 2 * (j & 1);
                    mma16816(accP[i][j], ah, bh[jp][jb], bh[jp][jb + 1]);
                    jam4(accP[i][j]);
                    mma16816h(accX[i][j], ah, bl[jp][jb], bl[jp][jb + 1]);
                    mma16816h(accX[i][j], al, bc[jp][jb], bc[jp][jb + 1]);
                }
            }
        }
        // After every 32 of K (the bound that fixes kHC): this step's H H + X 2^(c - 11) into the running sum, with
        // round-to-nearest adds; both MMA accumulators restart from zero.
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                    acc[i][j][2 * h] += fmaf(x.x, (float)(1 << (kHC - 11)), accP[i][j][2 * h]);
                    acc[i][j][2 * h + 1] += fmaf(x.y, (float)(1 << (kHC - 11)), accP[i][j][2 * h + 1]);
                    accP[i][j][2 * h] = accP[i][j][2 * h + 1] = 0.0f;
                }
                accX[i][j][0] = accX[i][j][1] = 0u;
            }
    }

    // Accumulator element e of (i, j): row wm 64 + 16 i + lane / 4 (+ 8 for e >= 2), column wn 32 + 8 j + 2 (lane % 4)
    // (+ 1 for odd e).
    const int er = wm * 64 + (lane >> 2), ec = wn * 32 + 2 * (lane & 3);
    auto comb = [&](float p) { return p * oscale; };
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
                        make_float2(comb(acc[i][j][2 * h]), comb(acc[i][j][2 * h + 1]));
            }
    } else {
        constexpr int APT = kHM / DEG;
        float* sC = reinterpret_cast<float*>(hsm);
        float* sS = sC + kHM * kHSRow;                    // the atoms' s
        int* sQ = reinterpret_cast<int*>(sS + kHM);  // and res0 rows
        __syncthreads();  // every warp is done with the pipeline buffers (no copy in flight: the last wait was 0)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    *reinterpret_cast<float2*>(sC + (er + 16 * i + 8 * h) * kHSRow + ec + 8 * j) =
                        make_float2(comb(acc[i][j][2 * h]), comb(acc[i][j][2 * h + 1]));
        const int a0 = m0 / DEG, na = min(APT, g.natoms - a0);
        if (tid < na) {
            const int a = a0 + tid;
            sS[tid] = 1.0f / (g.D[a] * g.beta + g.alpha);
            sQ[tid] = g.res0_sp ? g.res0_sp[a] : a;
        }
        __syncthreads();
        gate_bwd_tile<DEG, kHSRow, 2, RF>(sC, sS, sQ, a0, na, n0, tid, g);
        __syncthreads();
        gate_bwd_tile_dD<DEG, kHSRow>(sC, sS, a0, na, tid, g);
    }
}

// hgemm2f (forward projections, ledger gen-w2-fwd-two-half): C = A B^T with A in FP32 [M][K], split at the fragment
// loads into H = rn(a sA) and L' = rn(a sA - H) (the correction unscaled: L(A) 2^-11 of the panel form), so no
// producer changes and nothing else reads a panel. The cross term L' H(B) takes H(B) 2^-11 (formed in registers), so
// X keeps the panel form's scale and bound (|H(A)| <= 2^16, |H(B)| <= 2^15). A's tile goes to shared memory as FP32
// (cp.async, 16-byte chunks XOR-swizzled by row % 8: the quarter-warp LDS.64 of a fragment are conflict-free), the
// same 24 KB a stage as the panel kernel; each warp splits the rows it multiplies (the two warps of a row half both do).
// Plain FP32 store. Requirements as hgemm2_kernel's.
// Restart (RV): the forward's outputs cancel more than the reverse's (rms(result) 5-16% of sum |a b| on MH-1's contact
// cells), so the tensor cores' truncation biases them more. Restarted every 32 of K (RV 0, as hgemm2_kernel), the mean
// signed error on Sr-contact0.8's own gout was -1.3e-8 to -4.3e-8 of rms(result) against FP32's +-3e-9 (the
// representation alone, in float64, unbiased), and that cell's force rule failed at 0.9-1.4x. H H restarted at every MMA
// (RV 1, the default: each 16 of K from zero into the running sum with a round-to-nearest add; X every 32) brings it to
// -0.9e-8 to -2.9e-8 (the MMA's final rounding toward zero remains) and the cell passes (0.56); RV 2 (X every 16
// too) adds nothing and costs 4%. Still, with both kernels on, that cell's force ratio reached 0.51-1.10 over plan
// sets (FP32 path 0.17-0.72). The jam (jam4) on every FP32-accumulated MMA result, here and in hgemm2_kernel, takes the
// bias to +1e-9 to +5e-9 (emulated on the cell's own operands; FP32 +-0.4e-9 to 2.7e-9) and the cell to 0.16-0.38 over
// 16 plan sets x conv variants; it costs nothing measurable here (5060 Ti kernel bench) and about 1% in hgemm2_kernel.
// RV 0 with the jam (+4e-9 to +7e-9) is 3-5% slower than RV 1.
__device__ __forceinline__ void split2f(float2 x, float s, unsigned& h, unsigned& l) {
    const float a = x.x * s, b = x.y * s;
    const __half2 hh = __floats2half2_rn(a, b);
    const float2 hf = __half22float2(hh);
    const __half2 ll = __floats2half2_rn(a - hf.x, b - hf.y);
    h = *reinterpret_cast<const unsigned*>(&hh);
    l = *reinterpret_cast<const unsigned*>(&ll);
}

constexpr unsigned kHB11 = 0x10001000u;  // 2^-11 in both halves
constexpr int kHFwdVariant = 1;           // hgemm2_f32a's restart variant (hgemm2_f32a_variant)

// TAIL (LOUP_LIN_HALF's C0 shapes and beta): K % 16 == 0 and N % 16 == 0 suffice. B's panels are [2][Np][Kp] with Np,
// Kp rounded up to 64 / 32 and zeros in the padding (hgemm2_split_pad); A's chunks past K load as zeros; a warp computes
// and stores only its columns below N; beta != 0: C = beta C + A B^T. Without TAIL (N % 64 == 0, K % 32 == 0, beta 0)
// none of that is compiled in (the kernel as before the tails). DEG = 2l + 1 (l >= 1, residual-free): stores
// gate_fwd_kernel's gout_l = g (lin_l s), g = cs_g sigmoid(pre0[l C + c]), to g.out instead of lin_l (M = natoms DEG,
// N = g.C; C and beta unused), elementwise in the epilogue with the gate kernel's operations. SPLIT (list 22, DEG <= 0
// only): block z of gridDim.z takes its share of the 32-of-K steps and stores the raw sum into slice z of C (the
// workspace), for hgemm2_splitk_reduce; its own instances, so the unsplit ones keep their code (f7, with the split
// inside them, lost 1.5-1.9% on MH-1 SrTiO3, where nothing splits). RESID (list 27 item 3, LOUP_LIN_HALF_PARTS bit 64;
// DEG 3 and 5 only): the product is degree l's residual res_l, and the gated store adds it to lin_l s, lin_l read from
// ga.lin_l: gout_l = g (lin_l s + res_l), res_l never stored; the tile's lin_l rows are prefetched into L2 four K steps
// before the loop ends and each thread's 32 lin_l pairs are loaded before its first store (lists 23 and 25 read them
// one store at a time: the epilogue added 2.2 ms to OMOL-0 XL water 5,184, about what the gate kernel saved).
#ifdef LOUP_STUDENT_LAUNCH_BENCH
template <int RV, int DEG, bool TAIL, bool SPLIT = false, bool RESID = false, class Params = GateArgs>
#else
template <int RV, int DEG, bool TAIL, bool SPLIT = false, bool RESID = false>
#endif
__global__ void __launch_bounds__(128, 2) hgemm2f_kernel(int M, int N, int K, const float* __restrict__ A, float sA,
                                                         const __half* __restrict__ B, float oscale,
#ifdef LOUP_STUDENT_LAUNCH_BENCH
                                                         float* __restrict__ C, float beta, const __grid_constant__ Params ga) {
#else
                                                         float* __restrict__ C, float beta, GateArgs ga) {
#endif
    static_assert(!SPLIT || DEG <= 0, "only the plain and rows instances split");
    static_assert(!RESID || DEG == 3 || DEG == 5, "only degrees 1 and 2 have a residual");
    extern __shared__ __align__(128) unsigned char hsm[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(hsm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
    int row_tile = blockIdx.y;
#ifdef LOUP_STUDENT_LAUNCH_BENCH
    if constexpr (std::is_same_v<Params, StudentProjectionGroup>) {
        int i = ga.used - 1;
        while (i > 0 && row_tile < ga.matrix[i].tile_start) --i;
        const auto& p = ga.matrix[i];
        row_tile -= p.tile_start;
        M = p.rows; A = p.a; B = static_cast<const __half*>(p.b); C = p.c;
        sA = p.input_scale; oscale = p.output_scale;
    }
#endif
    const int m0 = row_tile * kHM, n0 = blockIdx.x * kHN;
    const int Kp = TAIL ? (K + kHK - 1) / kHK * kHK : K;
    const int64_t planeB = TAIL ? (int64_t)((N + kHN - 1) / kHN * kHN) * Kp : (int64_t)N * K;
    const int jn = TAIL ? min(4, max(0, (N - n0 - wn * 32) / 8)) : 4;  // this warp's 8-column groups below N
    const int fr = tid >> 3, fj = tid & 7;  // A copy roles: rows fr + 16 i (i = 0..7), 16-byte chunk fj
    const int cr = tid >> 2, cj = tid & 3;  // B copy roles as hgemm2_kernel's
    const __half* bg = B + (int64_t)(n0 + cr) * Kp + cj * 8;
    auto stage_b = [&](int st, int p) { return sbase + (unsigned)((st * kHStage + 2 * kHTileA + p * kHTileB) * 2); };
    auto issue = [&](int kt) {
        const int st = kt & 1, k0 = kt * kHK;
        const unsigned sa = sbase + (unsigned)(st * kHStage * 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = fr + 16 * i;
            const bool rok = m0 + r < M, kok = !TAIL || k0 + 4 * fj < K;
            cp16(sa + r * 128 + ((fj ^ (r & 7)) << 4), A + (int64_t)(rok ? m0 + r : m0) * K + (kok ? k0 + 4 * fj : 0),
                 rok && kok);
        }
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                cp16(stage_b(st, p) + swz(cr + 32 * i, cj), bg + p * planeB + (int64_t)32 * i * Kp + k0, true);
        asm volatile("cp.async.commit_group;");
    };

    float acc[4][4][4], accP[4][4][4];
    unsigned accX[4][4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = accP[i][j][e] = 0.0f;
            accX[i][j][0] = accX[i][j][1] = 0u;
        }

    const int g = lane >> 2, t = lane & 3;
    const int b_row = wn * 32 + ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;
    const int nk = Kp / kHK;
    int kt0 = 0, kt1 = nk;
    if constexpr (SPLIT) {  // list 22: this block's 32-of-K steps; the host leaves no slice empty
        const int nks = (nk + (int)gridDim.z - 1) / (int)gridDim.z;
        kt0 = (int)blockIdx.z * nks;
        kt1 = min(nk, kt0 + nks);
    }
    issue(kt0);
    for (int kt = kt0; kt < kt1; ++kt) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (kt + 1 < kt1) issue(kt + 1);
        if constexpr (RESID) {  // list 27 item 3: the tile's lin_l rows into L2 for the epilogue, as hgemm2_kernel's
            if (kt == (kt1 > 4 ? kt1 - 4 : 0)) {
                const int r = m0 + tid;
                if (r < M) {
                    prefetch_l2(ga.lin_l + (int64_t)r * N + n0);
                    if (n0 + 32 < N) prefetch_l2(ga.lin_l + (int64_t)r * N + n0 + 32);
                }
            }
        }
        if (TAIL && jn == 0) continue;  // a warp wholly past N (the last column tile) only copies
        const int st = kt & 1;
        const unsigned char* sa = hsm + st * kHStage * 2 + (wm * 64 + g) * 128 + (t & 1) * 8;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                ldsm4(stage_b(st, 0) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                ldsm4(stage_b(st, 1) + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
            }
            const int ja = 4 * kk + (t >> 1);
            const int o0 = (ja ^ g) << 4, o1 = ((ja + 2) ^ g) << 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned char* s = sa + 16 * i * 128;
                unsigned ah[4], al[4];
                split2f(*reinterpret_cast<const float2*>(s + o0), sA, ah[0], al[0]);
                split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o0), sA, ah[1], al[1]);
                split2f(*reinterpret_cast<const float2*>(s + o1), sA, ah[2], al[2]);
                split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o1), sA, ah[3], al[3]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j >= jn) continue;
                    const int jp = j >> 1, jb = 2 * (j & 1);
                    if constexpr (RV == 0) {
                        mma16816(accP[i][j], ah, bh[jp][jb], bh[jp][jb + 1]);
                        jam4(accP[i][j]);
                    } else {  // H H of this 16 of K from zero, jammed, into the running sum (round-to-nearest add)
                        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        mma16816(d, ah, bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[i][j][e] += d[e];
                    }
                    mma16816h(accX[i][j], ah, bl[jp][jb], bl[jp][jb + 1]);
                    mma16816h(accX[i][j], al, bc[jp][jb], bc[jp][jb + 1]);
                }
            }
            if constexpr (RV == 2) {  // X restarted every 16 of K too
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                            acc[i][j][2 * h] = fmaf(x.x, (float)(1 << (kHC - 11)), acc[i][j][2 * h]);
                            acc[i][j][2 * h + 1] = fmaf(x.y, (float)(1 << (kHC - 11)), acc[i][j][2 * h + 1]);
                        }
                        accX[i][j][0] = accX[i][j][1] = 0u;
                    }
            }
        }
        if constexpr (RV != 2) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        if constexpr (RV == 0) {
                            acc[i][j][2 * h] += fmaf(x.x, (float)(1 << (kHC - 11)), accP[i][j][2 * h]);
                            acc[i][j][2 * h + 1] += fmaf(x.y, (float)(1 << (kHC - 11)), accP[i][j][2 * h + 1]);
                            accP[i][j][2 * h] = accP[i][j][2 * h + 1] = 0.0f;
                        } else {
                            acc[i][j][2 * h] = fmaf(x.x, (float)(1 << (kHC - 11)), acc[i][j][2 * h]);
                            acc[i][j][2 * h + 1] = fmaf(x.y, (float)(1 << (kHC - 11)), acc[i][j][2 * h + 1]);
                        }
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
        }
    }

    const int er = wm * 64 + g, ec = wn * 32 + 2 * t;
    if constexpr (SPLIT) {  // list 22: the raw sum into slice z of the workspace C; hgemm2_splitk_reduce does the rest
        float* const W = C + (int64_t)blockIdx.z * M * N;
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = m0 + er + 16 * i + 8 * h;
                if (r >= M) continue;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j >= jn) continue;
                    *reinterpret_cast<float2*>(W + (int64_t)r * N + n0 + ec + 8 * j) =
                        make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                }
            }
        return;
    }
    if constexpr (DEG > 0) {
        // list 11: g = cg sigmoid(pre) and s are formed once per (atom, column) of the tile and staged in the pipeline's
        // dead shared memory (an atom's DEG rows share them), instead of per element: per row, the old store loaded
        // lin0 / res0 and evaluated the sigmoid DEG times over (at most 128 atoms x 72 floats = 36 KB of kHSmem)
        constexpr int l = DEG / 2;
        constexpr int SGS = kHN + 8;
        float* sG = reinterpret_cast<float*>(hsm);
        const int a0 = m0 / DEG, na = (min(m0 + kHM, M) - 1) / DEG - a0 + 1;
        float* sS = sG + na * SGS;
        int* sQ = reinterpret_cast<int*>(sS + na);  // list 26 item 3: the atoms' res0 rows
        // RESID (list 27 item 3): every lin_l pair this thread stores, loaded before the gate's staging (ga's pointers
        // are not restrict, so loads placed among the stores would each wait on the store before them)
        float2 lv[RESID ? 4 : 1][2][4];
        if constexpr (RESID) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = m0 + er + 16 * i + 8 * h;
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        lv[i][h][j] = r < M && j < jn
                                          ? *reinterpret_cast<const float2*>(ga.lin_l + (int64_t)r * N + n0 + ec + 8 * j)
                                          : make_float2(0.0f, 0.0f);
                }
        }
        __syncthreads();  // every warp is done with the pipeline buffers
        for (int ai = tid; ai < na; ai += 128) {
            const int a = a0 + ai;
            sS[ai] = 1.0f / (ga.D[a] * ga.beta + ga.alpha);
            sQ[ai] = ga.res0_sp ? ga.res0_sp[a] : a;
        }
        __syncthreads();
        {
            // list 26 item 3: thread (tid / 64, tid % 64) takes atoms tid / 64 + 2 j at column tid % 64, as the loop
            // over idx = tid + 128 k did, but four atoms' operand loads are issued before their sigmoids (one at a
            // time, the epilogue waited on each: ncu long_scoreboard 1.4-2.6 per issued instruction)
            constexpr int B = 4;
            const int cl = tid & 63, c = n0 + cl;
            for (int ai0 = tid >> 6; ai0 < na; ai0 += 2 * B) {
                float lv[B], rv[B];
#pragma unroll
                for (int j = 0; j < B; ++j) {
                    const int ai = ai0 + 2 * j;
                    const bool ok = ai < na && c < N;
                    const int aj = ok ? ai : 0;
                    lv[j] = ok ? ga.lin0[(int64_t)(a0 + aj) * 4 * N + l * N + c] : 0.0f;
                    rv[j] = ok && ga.res0 ? ga.res0[(int64_t)sQ[aj] * 4 * N + l * N + c] : 0.0f;
                }
#pragma unroll
                for (int j = 0; j < B; ++j) {
                    const int ai = ai0 + 2 * j;
                    if (ai < na && c < N) sG[ai * SGS + cl] = ga.cg * sigmoid_f(lv[j] * sS[ai] + rv[j]);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = m0 + er + 16 * i + 8 * h;
                if (r >= M) continue;
                const int ai = r / DEG - a0;
                const float s = sS[ai];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j >= jn) continue;
                    const int cl = ec + 8 * j;
                    const float2 gg = *reinterpret_cast<const float2*>(sG + ai * SGS + cl);
                    if constexpr (RESID) {  // gate_fwd_kernel's p = lin_l s + res_l, res_l the product
                        const float2 v = lv[RESID ? i : 0][h][j];
                        *reinterpret_cast<float2*>(ga.out + (int64_t)r * N + n0 + cl) =
                            make_float2(gg.x * fmaf(v.x, s, acc[i][j][2 * h] * oscale),
                                        gg.y * fmaf(v.y, s, acc[i][j][2 * h + 1] * oscale));
                    } else {
                        *reinterpret_cast<float2*>(ga.out + (int64_t)r * N + n0 + cl) = make_float2(
                            gg.x * ((acc[i][j][2 * h] * oscale) * s), gg.y * ((acc[i][j][2 * h + 1] * oscale) * s));
                    }
                }
            }
    } else if constexpr (DEG < 0) {
        // hgemm2_f32a_rows (list 13; beta 0): the row scale, and with c2 the column split (a 64-wide tile lies in one
        // part). An instance of its own, so the plain one below keeps its code.
        const bool part2 = ga.c2 && n0 >= ga.n1;
        const bool scaled = ga.rdeg && (!ga.c2 || part2);
        const int ldc = !ga.c2 ? N : (part2 ? N - ga.n1 : ga.n1);
        float* const Cb = part2 ? ga.c2 + (n0 - ga.n1) : C + n0;
        const float os = part2 ? ga.os2 : oscale;
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = m0 + er + 16 * i + 8 * h;
                if (r >= M) continue;
                const float rs = scaled ? fmaf(ga.D[r / ga.rdeg], ga.beta, ga.alpha) * os : os;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j >= jn) continue;
                    *reinterpret_cast<float2*>(Cb + (int64_t)r * ldc + ec + 8 * j) =
                        make_float2(acc[i][j][2 * h] * rs, acc[i][j][2 * h + 1] * rs);
                }
            }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = m0 + er + 16 * i + 8 * h;
                if (r >= M) continue;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j >= jn) continue;
                    float2* o = reinterpret_cast<float2*>(C + (int64_t)r * N + n0 + ec + 8 * j);
                    float2 v = make_float2(acc[i][j][2 * h] * oscale, acc[i][j][2 * h + 1] * oscale);
                    if (TAIL && beta != 0.0f) {
                        const float2 c = *o;
                        v.x = fmaf(beta, c.x, v.x);
                        v.y = fmaf(beta, c.y, v.y);
                    }
                    *o = v;
                }
            }
    }
}

// The two panels of a weight: out[0 .. n) = H = rn(w scale), out[n .. 2n) = rn((w scale - H) 2^(11 - c)), the
// correction already at the cross term's shift.
__global__ void split_half_kernel(int64_t n, const float* __restrict__ w, float scale, __half* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = w[i] * scale;
    const __half h = __float2half_rn(a);
    const float hf = __half2float(h);
    out[i] = h;
    out[n + i] = __float2half_rn((a - hf) * (1.0f / (float)(1 << (kHC - 11))));
}

// The panels of w^T for w [R][Cc] (out [2][Cc][R]): the forward projection's B (hgemm2_f32a takes B as [N][K]).
__global__ void split_half_t_kernel(int R, int Cc, const float* __restrict__ w, float scale, __half* __restrict__ out) {
    const int64_t n = (int64_t)R * Cc, i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t j = i / R, r = i % R;
    const float a = w[r * Cc + j] * scale;
    const __half h = __float2half_rn(a);
    out[i] = h;
    out[n + i] = __float2half_rn((a - __half2float(h)) * (1.0f / (float)(1 << (kHC - 11))));
}

// The panels of a weight padded to [2][Np][Kp] (Np = N rounded up to 64, Kp = K rounded up to 32; zeros in the padding)
// for hgemm2f_kernel's tails: element (r, k) is w[r][k] (w [Nr][Kc]) or, with trans, w[k][r] (w [Kc][Nr]).
__global__ void split_half_pad_kernel(int Nr, int Kc, int Np, int Kp, const float* __restrict__ w, int trans,
                                      float scale, __half* __restrict__ out) {
    const int64_t n = (int64_t)Np * Kp, i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int r = (int)(i / Kp), k = (int)(i % Kp);
    float a = 0.0f;
    if (r < Nr && k < Kc) a = (trans ? w[(int64_t)k * Nr + r] : w[(int64_t)r * Kc + k]) * scale;
    const __half h = __float2half_rn(a);
    out[i] = h;
    out[n + i] = __float2half_rn((a - __half2float(h)) * (1.0f / (float)(1 << (kHC - 11))));
}

// list 22: hgemm2f_kernel's split-K reduction. ws holds S slices [S][M][N] of raw sums; C = their sum in the order
// z = 0 .. S - 1, through the epilogue the split skipped: with `rows`, hgemm2_f32a_rows' row scale and column split
// (n1 a multiple of 64, so an element pair lies in one part), else the plain store (oscale, beta). One thread per pair.
__global__ void hgemm2_splitk_reduce_kernel(int M, int N, int S, const float* __restrict__ ws, float oscale,
                                            float* __restrict__ C, float beta, GateArgs ga, bool rows) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x, mn = (int64_t)M * N;
    if (2 * i >= mn) return;
    float2 s = reinterpret_cast<const float2*>(ws)[i];
    for (int z = 1; z < S; ++z) {
        const float2 p = reinterpret_cast<const float2*>(ws + z * mn)[i];
        s.x += p.x;
        s.y += p.y;
    }
    const int r = (int)(2 * i / N), c = (int)(2 * i % N);
    if (rows) {
        const bool part2 = ga.c2 && c >= ga.n1;
        const bool scaled = ga.rdeg && (!ga.c2 || part2);
        const int ldc = !ga.c2 ? N : (part2 ? N - ga.n1 : ga.n1);
        float* const Cb = part2 ? ga.c2 + (c - ga.n1) : C + c;
        const float os = part2 ? ga.os2 : oscale;
        const float rs = scaled ? fmaf(ga.D[r / ga.rdeg], ga.beta, ga.alpha) * os : os;
        *reinterpret_cast<float2*>(Cb + (int64_t)r * ldc) = make_float2(s.x * rs, s.y * rs);
    } else {
        float2* o = reinterpret_cast<float2*>(C + 2 * i);
        float2 v = make_float2(s.x * oscale, s.y * oscale);
        if (beta != 0.0f) {
            const float2 cv = *o;
            v.x = fmaf(beta, cv.x, v.x);
            v.y = fmaf(beta, cv.y, v.y);
        }
        *o = v;
    }
}

// list 22: the K slices for hgemm2f_kernel at M x N x K, 1 = no split. Split only where K >= 1,024 and there are at
// most as many tiles as SMs (half a wave at two blocks per SM; MH-1's degree-0 products with K = 2,048 on small cells:
// 6 and 12 tiles on NaCl216), up to the slots the tiles leave free, each slice at least 8 steps of 32, the slices
// within the workspace, none empty.
int hgemm2_split_k(int M, int N, int K, size_t ws_bytes) {
    const int nk = (K + kHK - 1) / kHK;
    if (nk < 32 || ws_bytes == 0 || N % 2 != 0) return 1;
    int dev = 0, sms = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
                                                 cudaSuccess || sms <= 0)
        return 1;
    const int64_t tiles = (int64_t)((N + kHN - 1) / kHN) * ((M + kHM - 1) / kHM);
    const int64_t slots = 2 * (int64_t)sms;
    if (2 * tiles > slots) return 1;
    int64_t s = std::min<int64_t>(nk / 8, slots / tiles);
    s = std::min<int64_t>(s, (int64_t)(ws_bytes / ((size_t)M * N * sizeof(float))));
    if (s < 2) return 1;
    const int nks = (nk + (int)s - 1) / (int)s;
    return (nk + nks - 1) / nks;
}

void hgemm2_splitk_reduce(int M, int N, int S, const float* ws, float oscale, float* C, float beta, const GateArgs& ga,
                          bool rows, cudaStream_t stream) {
    const int64_t pairs = (int64_t)M * N / 2;
    hgemm2_splitk_reduce_kernel<<<(unsigned)((pairs + 255) / 256), 256, 0, stream>>>(M, N, S, ws, oscale, C, beta, ga,
                                                                                     rows);
}

template <int DEG, bool RF = false>
void launch_hgemm2(int M, int N, int K, const __half* A, const __half* B, float oscale, float* C, const GateArgs& g,
                   cudaStream_t stream) {
    const int rows = DEG ? kHM / DEG * DEG : kHM;
    const dim3 grid(N / kHN, (M + rows - 1) / rows);
    hgemm2_kernel<DEG, RF><<<grid, 128, kHSmem, stream>>>(M, N, K, A, B, oscale, C, g);
}

}  // namespace

#ifdef LOUP_STUDENT_LAUNCH_BENCH
void launch_student_projection_group(const StudentProjectionGroup& p, cudaStream_t stream) {
    const dim3 grid((p.cols + kHN - 1) / kHN, p.row_tiles);
    if (p.cols % kHN || p.inner % kHK)
        hgemm2f_kernel<1, 0, true><<<grid, 128, kHSmem, stream>>>(
            0, p.cols, p.inner, nullptr, 0, nullptr, 0, nullptr, 0, p);
    else
        hgemm2f_kernel<1, 0, false><<<grid, 128, kHSmem, stream>>>(
            0, p.cols, p.inner, nullptr, 0, nullptr, 0, nullptr, 0, p);
}
#endif

// Two blocks (2 x 48 KB) per SM need the largest shared-memory carveout.
void hgemm2_init() {
    cudaFuncSetAttribute(hgemm2_kernel<0>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2_kernel<1>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2_kernel<3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2_kernel<5>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2_kernel<7>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<0, 0, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<1, 0, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<1, 0, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<2, 0, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<kHFwdVariant, 3, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<kHFwdVariant, 5, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<kHFwdVariant, 7, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<kHFwdVariant, 3, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<kHFwdVariant, 5, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<kHFwdVariant, 7, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<kHFwdVariant, -1, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<kHFwdVariant, -1, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<1, 0, false, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<1, 0, true, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2_kernel<3, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2_kernel<5, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<kHFwdVariant, 3, false, false, true>,
                         cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<kHFwdVariant, 5, false, false, true>,
                         cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<kHFwdVariant, 3, true, false, true>,
                         cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(hgemm2f_kernel<kHFwdVariant, 5, true, false, true>,
                         cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

namespace {
bool h2_tail(int N, int K, float beta) { return N % kHN != 0 || K % kHK != 0 || beta != 0.0f; }
}  // namespace

// C [M][N] = A B^T oscale: A FP32 [M][K] (split in the kernel at scale sA, a power of two with max |A| sA < 65,504 or
// the result is not finite), B two panels [2][N][K] (hgemm2_split); oscale = 1 / (sA 2^e_B).
// The restart variants (benches): 0 H H and X every 32 of K, 1 H H every MMA (16 of K) and X every 32, 2 both every 16;
// 0 and 2 without tails or beta.
void hgemm2_f32a_variant(int v, int M, int N, int K, const float* A, float sA, const void* B, float oscale, float* C,
                         cudaStream_t stream, float beta, float* ws, size_t ws_bytes) {
    const __half* b = static_cast<const __half*>(B);
    const GateArgs ga{};
    // list 22: split-K into ws for small M and long K (variant 1 only), then the reduction's plain store
    if (const int S = v == 1 && ws ? hgemm2_split_k(M, N, K, ws_bytes) : 1; S > 1) {
        const dim3 gs((N + kHN - 1) / kHN, (M + kHM - 1) / kHM, S);
        if (h2_tail(N, K, 0.0f))
            hgemm2f_kernel<1, 0, true, true><<<gs, 128, kHSmem, stream>>>(M, N, K, A, sA, b, oscale, ws, 0.0f, ga);
        else
            hgemm2f_kernel<1, 0, false, true><<<gs, 128, kHSmem, stream>>>(M, N, K, A, sA, b, oscale, ws, 0.0f, ga);
        hgemm2_splitk_reduce(M, N, S, ws, oscale, C, beta, ga, false, stream);
        return;
    }
    const dim3 grid((N + kHN - 1) / kHN, (M + kHM - 1) / kHM);
    if (v == 1 && h2_tail(N, K, beta))
        hgemm2f_kernel<1, 0, true><<<grid, 128, kHSmem, stream>>>(M, N, K, A, sA, b, oscale, C, beta, ga);
    else if (v == 1) hgemm2f_kernel<1, 0, false><<<grid, 128, kHSmem, stream>>>(M, N, K, A, sA, b, oscale, C, 0.0f, ga);
    else if (v == 0) hgemm2f_kernel<0, 0, false><<<grid, 128, kHSmem, stream>>>(M, N, K, A, sA, b, oscale, C, 0.0f, ga);
    else hgemm2f_kernel<2, 0, false><<<grid, 128, kHSmem, stream>>>(M, N, K, A, sA, b, oscale, C, 0.0f, ga);
}

void hgemm2_f32a_rows(int M, int N, int K, const float* A, float sA, const void* B, float oscale, float* C,
                      const GateArgs& rs, cudaStream_t stream, float* ws, size_t ws_bytes) {
    const __half* b = static_cast<const __half*>(B);
    // list 22: split-K as hgemm2_f32a (the split instances end before the DEG-specific epilogue, so the plain ones serve);
    // the reduction takes the row scale and the column split
    if (const int S = ws ? hgemm2_split_k(M, N, K, ws_bytes) : 1; S > 1) {
        const dim3 gs((N + kHN - 1) / kHN, (M + kHM - 1) / kHM, S);
        if (h2_tail(N, K, 0.0f))
            hgemm2f_kernel<kHFwdVariant, 0, true, true><<<gs, 128, kHSmem, stream>>>(M, N, K, A, sA, b, oscale, ws, 0.0f,
                                                                                    rs);
        else
            hgemm2f_kernel<kHFwdVariant, 0, false, true><<<gs, 128, kHSmem, stream>>>(M, N, K, A, sA, b, oscale, ws, 0.0f,
                                                                                     rs);
        hgemm2_splitk_reduce(M, N, S, ws, oscale, C, 0.0f, rs, true, stream);
        return;
    }
    const dim3 grid((N + kHN - 1) / kHN, (M + kHM - 1) / kHM);
    if (h2_tail(N, K, 0.0f))
        hgemm2f_kernel<kHFwdVariant, -1, true><<<grid, 128, kHSmem, stream>>>(M, N, K, A, sA, b, oscale, C, 0.0f, rs);
    else
        hgemm2f_kernel<kHFwdVariant, -1, false><<<grid, 128, kHSmem, stream>>>(M, N, K, A, sA, b, oscale, C, 0.0f, rs);
}

void hgemm2_f32a_gate(int l, int M, int N, int K, const float* A, float sA, const void* B, float oscale,
                      const GateArgs& g, cudaStream_t stream) {
    const dim3 grid((N + kHN - 1) / kHN, (M + kHM - 1) / kHM);
    const __half* b = static_cast<const __half*>(B);
    const auto go = [&](auto kernel) { kernel<<<grid, 128, kHSmem, stream>>>(M, N, K, A, sA, b, oscale, nullptr, 0.0f, g); };
    if (g.lin_l) {  // list 27 item 3: the residual degrees' form (l = 1, 2; the caller checks)
        if (h2_tail(N, K, 0.0f)) {
            if (l == 1) go(hgemm2f_kernel<kHFwdVariant, 3, true, false, true>);
            else go(hgemm2f_kernel<kHFwdVariant, 5, true, false, true>);
        } else {
            if (l == 1) go(hgemm2f_kernel<kHFwdVariant, 3, false, false, true>);
            else go(hgemm2f_kernel<kHFwdVariant, 5, false, false, true>);
        }
        return;
    }
    if (h2_tail(N, K, 0.0f)) {
        if (l == 1) go(hgemm2f_kernel<kHFwdVariant, 3, true>);
        else if (l == 2) go(hgemm2f_kernel<kHFwdVariant, 5, true>);
        else go(hgemm2f_kernel<kHFwdVariant, 7, true>);
    } else {
        if (l == 1) go(hgemm2f_kernel<kHFwdVariant, 3, false>);
        else if (l == 2) go(hgemm2f_kernel<kHFwdVariant, 5, false>);
        else go(hgemm2f_kernel<kHFwdVariant, 7, false>);
    }
}

void hgemm2_f32a(int M, int N, int K, const float* A, float sA, const void* B, float oscale, float* C,
                 cudaStream_t stream, float beta, float* ws, size_t ws_bytes) {
    hgemm2_f32a_variant(kHFwdVariant, M, N, K, A, sA, B, oscale, C, stream, beta, ws, ws_bytes);
}

int64_t hgemm2_pad_elems(int Nr, int Kc) {
    return (int64_t)((Nr + kHN - 1) / kHN * kHN) * ((Kc + kHK - 1) / kHK * kHK);
}

void hgemm2_split_pad(int Nr, int Kc, const float* w, bool trans, int e, void* out, cudaStream_t stream) {
    const int Np = (Nr + kHN - 1) / kHN * kHN, Kp = (Kc + kHK - 1) / kHK * kHK;
    const int64_t n = (int64_t)Np * Kp;
    split_half_pad_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(Nr, Kc, Np, Kp, w, trans ? 1 : 0,
                                                                           std::ldexp(1.0f, e),
                                                                           static_cast<__half*>(out));
}

// The exponent e of the weight panels' scale 2^e: the largest with max |w| 2^e <= 2^15, so both panels stay finite
// (|L| <= |H|) and as far from binary16's subnormals as that allows. 0 for an all-zero weight.
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

// out [2 R Cc] halves: the panels of w^T, w [R][Cc], at scale 2^e.
void hgemm2_split_t(int R, int Cc, const float* w, int e, void* out, cudaStream_t stream) {
    const int64_t n = (int64_t)R * Cc;
    split_half_t_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(R, Cc, w, std::ldexp(1.0f, e),
                                                                         static_cast<__half*>(out));
}

// dgout_l = A B^T with degree l's gate backward (gate_bwd_tile) for M = g.natoms (2l + 1) rows; l = -1: the plain
// product into C (tests). A: two panels [2][M][K] at scale 2^12 (loup_store_h2), B: two panels [2][N][K] at 2^e_B;
// oscale = 2^-(12 + e_B).
void hgemm2_gate(int l, int M, int N, int K, const void* A, const void* B, float oscale, float* C, const GateArgs& g,
                 cudaStream_t stream) {
    const __half* a = static_cast<const __half*>(A);
    const __half* b = static_cast<const __half*>(B);
    switch (l) {
        case -1: launch_hgemm2<0>(M, N, K, a, b, oscale, C, g, stream); break;
        case 0: launch_hgemm2<1>(M, N, K, a, b, oscale, C, g, stream); break;
        case 1:  // list 27 item 3: a residual degree whose forward gate ran in W_res's epilogue has its own instance
            if (g.rform) launch_hgemm2<3, true>(M, N, K, a, b, oscale, C, g, stream);
            else launch_hgemm2<3>(M, N, K, a, b, oscale, C, g, stream);
            break;
        case 2:
            if (g.rform) launch_hgemm2<5, true>(M, N, K, a, b, oscale, C, g, stream);
            else launch_hgemm2<5>(M, N, K, a, b, oscale, C, g, stream);
            break;
        default: launch_hgemm2<7>(M, N, K, a, b, oscale, C, g, stream); break;
    }
}


// ===================== Universal Per-Row Two-Half FP16 GEMM v2 (Plain + 64x128 + K128-Fused + Gate Bwd/Fwd) =====================
namespace {
__device__ __forceinline__ float h2r_pow2_f(int e) {
    return __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}
__device__ __forceinline__ int h2r_exp_ceil(float am) {
    const uint32_t u = __float_as_uint(am);
    int e = (int)((u >> 23) & 0xff) - 127 + 1;
    if (e < -108) e = -108;
    if (e > 108) e = 108;
    return e;
}

template <int TPR>
__global__ void h2r_row_exp_kernel(int M, int K, const float* __restrict__ A, int8_t* __restrict__ eA) {
    const int r = blockIdx.x * (blockDim.x / TPR) + (threadIdx.x / TPR);
    if (r >= M) return;
    const int lane = threadIdx.x & (TPR - 1), K4 = K >> 2;
    const float4* row4 = reinterpret_cast<const float4*>(A + (int64_t)r * K);
    float am = 0.0f;
    for (int k = lane; k < K4; k += TPR) {
        const float4 v = row4[k];
        am = fmaxf(am, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    if constexpr (TPR >= 32) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 16));
    if constexpr (TPR >= 16) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    if (lane == 0) eA[r] = (int8_t)((am > 0.0f && isfinite(am)) ? h2r_exp_ceil(am) : -127);
}

__global__ void h2r_pack_b_col_kernel(int Nr, int Kc, int Np, int Kp, const float* __restrict__ w, int trans,
                                      __half* __restrict__ out, float* __restrict__ out_inv_sB) {
    const int r = blockIdx.x;
    if (r >= Np) return;
    const int tid = threadIdx.x;
    float am = 0.0f;
    if (r < Nr) {
        for (int k = tid; k < Kc; k += blockDim.x) {
            const float v = trans ? w[(int64_t)k * Nr + r] : w[(int64_t)r * Kc + k];
            am = fmaxf(am, fabsf(v));
        }
    }
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    __shared__ float s_am[4];
    if ((tid & 31) == 0) s_am[tid >> 5] = am;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        for (int i = 0; i < 4; ++i) m = fmaxf(m, s_am[i]);
        s_am[0] = m;
    }
    __syncthreads();
    const float m = s_am[0];
    const int e = (m > 0.0f && isfinite(m)) ? h2r_exp_ceil(m) : -127;
    const float sB = (e > -127) ? h2r_pow2_f(12 - e) : 0.0f;
    if (tid == 0 && r < Nr) out_inv_sB[r] = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
    const int64_t plane = (int64_t)Np * Kp;
    for (int k = tid; k < Kp; k += blockDim.x) {
        float a = 0.0f;
        if (r < Nr && k < Kc) a = (trans ? w[(int64_t)k * Nr + r] : w[(int64_t)r * Kc + k]) * sB;
        const __half h = __float2half_rn(a);
        const int64_t idx = (int64_t)r * Kp + k;
        out[idx] = h;
        out[plane + idx] = __float2half_rn((a - __half2float(h)) * (1.0f / (float)(1 << (kHC - 11))));
    }
}

// Plain, Split-K, Gate-Bwd (DEG=1,3,5,7, GATE_FWD=false), and Gate-Fwd (DEG=3,5,7, GATE_FWD=true)
struct SiluEpiArgs {
    const int* species = nullptr;
    const float* bias_or_pre = nullptr;
    float* pre_out = nullptr;
    int8_t* eA_out = nullptr;
    bool per_sp = false;
};

template <int BM_T, bool TAIL, bool SPLIT, int DEG = 0, bool GATE_FWD = false, int SILU_EPI = 0>
__global__ void __launch_bounds__(128, BM_T == 64 ? 3 : 2)
hgemm2f_row_kernel(int M, int N, int K, const float* __restrict__ A, const int8_t* __restrict__ eA,
                   const __half* __restrict__ B, const float* __restrict__ inv_sB,
                   float* __restrict__ C, float beta, GateArgs ga, SiluEpiArgs sa = {}) {
    constexpr int NI = BM_T / 32;
    constexpr int MT = (DEG > 0 && !GATE_FWD) ? (BM_T / DEG * DEG) : BM_T;
    constexpr int kTileA = BM_T * kHK;
    constexpr int kStageH = 2 * kTileA + 2 * kHTileB;
    extern __shared__ __align__(128) unsigned char hsm[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(hsm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * MT, n0 = blockIdx.x * kHN;
    const int Kp = TAIL ? (K + kHK - 1) / kHK * kHK : K;
    const int64_t planeB = TAIL ? (int64_t)((N + kHN - 1) / kHN * kHN) * Kp : (int64_t)N * K;
    const int jn = TAIL ? min(4, max(0, (N - n0 - wn * 32) / 8)) : 4;
    const int fr = tid >> 3, fj = tid & 7;
    const int cr = tid >> 2, cj = tid & 3;
    const __half* bg = B + (int64_t)(n0 + cr) * Kp + cj * 8;
    const int g = lane >> 2, t = lane & 3;
    const int er = wm * (BM_T / 2) + g, ec = wn * 32 + 2 * t;

    __shared__ float s_sA[BM_T];
    if (tid < BM_T) {
        const int r = m0 + tid;
        const int e = (tid < MT && r < M) ? (int)eA[r] : -127;
        s_sA[tid] = (e > -127) ? h2r_pow2_f(12 - e) : 0.0f;
    }

    auto stage_b = [&](int st, int p) { return sbase + (unsigned)((st * kStageH + 2 * kTileA + p * kHTileB) * 2); };
    auto issue = [&](int kt) {
        const int st = kt & 1, k0 = kt * kHK;
        const unsigned sa = sbase + (unsigned)(st * kStageH * 2);
#pragma unroll
        for (int i = 0; i < (BM_T / 16); ++i) {
            const int r = fr + 16 * i;
            const bool rok = (r < MT) && (m0 + r < M), kok = !TAIL || k0 + 4 * fj < K;
            cp16(sa + r * 128 + ((fj ^ (r & 7)) << 4), A + (int64_t)(rok ? m0 + r : m0) * K + (kok ? k0 + 4 * fj : 0),
                 rok && kok);
        }
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                cp16(stage_b(st, p) + swz(cr + 32 * i, cj), bg + p * planeB + (int64_t)32 * i * Kp + k0, true);
        asm volatile("cp.async.commit_group;");
    };

    float acc[NI][4][4];
    unsigned accX[NI][4][2];
#pragma unroll
    for (int i = 0; i < NI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
            accX[i][j][0] = accX[i][j][1] = 0u;
        }

    const int b_row = wn * 32 + ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;
    const int nk = Kp / kHK;
    int kt0 = 0, kt1 = nk;
    if constexpr (SPLIT) {
        const int nks = (nk + (int)gridDim.z - 1) / (int)gridDim.z;
        kt0 = (int)blockIdx.z * nks;
        kt1 = min(nk, kt0 + nks);
    }
    issue(kt0);
    for (int kt = kt0; kt < kt1; ++kt) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (kt + 1 < kt1) issue(kt + 1);
        if constexpr (DEG > 0 && !GATE_FWD) {
            if (kt == (kt1 > 4 ? kt1 - 4 : 0)) {
                constexpr int l = DEG / 2;
                const int a0 = m0 / DEG, na = min(MT / DEG, ga.natoms - a0), rows = min(MT, M - m0);
                if (DEG > 1)
                    for (int li = tid; li < 2 * rows; li += 128) {
                        const int64_t off = (int64_t)(m0 + (li >> 1)) * N + n0 + 32 * (li & 1);
                        if (ga.lin_l) prefetch_l2(ga.lin_l + off);
                        if (ga.res_l) prefetch_l2(ga.res_l + off);
                    }
                for (int li = tid; li < 2 * na; li += 128) {
                    const int64_t off = (int64_t)(a0 + (li >> 1)) * 4 * N + l * N + n0 + 32 * (li & 1);
                    prefetch_l2(ga.lin0 + off);
                    if (ga.res0 && !ga.res0_sp) prefetch_l2(ga.res0 + off);
                }
            }
        }
        if (TAIL && jn == 0) continue;
        const int st = kt & 1;
        const unsigned char* sa = hsm + st * kStageH * 2 + er * 128 + (t & 1) * 8;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                ldsm4(stage_b(st, 0) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                ldsm4(stage_b(st, 1) + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
            }
            const int ja = 4 * kk + (t >> 1);
            const int o0 = (ja ^ g) << 4, o1 = ((ja + 2) ^ g) << 4;
#pragma unroll
            for (int i = 0; i < NI; ++i) {
                const unsigned char* s = sa + 16 * i * 128;
                unsigned ah[4], al[4];
                const float sc0 = s_sA[er + 16 * i], sc1 = s_sA[er + 16 * i + 8];
                split2f(*reinterpret_cast<const float2*>(s + o0), sc0, ah[0], al[0]);
                split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o0), sc1, ah[1], al[1]);
                split2f(*reinterpret_cast<const float2*>(s + o1), sc0, ah[2], al[2]);
                split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o1), sc1, ah[3], al[3]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j >= jn) continue;
                    const int jp = j >> 1, jb = 2 * (j & 1);
                    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    mma16816(d, ah, bh[jp][jb], bh[jp][jb + 1]);
                    jam4(d);
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][j][e] += d[e];
                    mma16816h(accX[i][j], ah, bl[jp][jb], bl[jp][jb + 1]);
                    mma16816h(accX[i][j], al, bc[jp][jb], bc[jp][jb + 1]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < NI; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                    acc[i][j][2 * h] = fmaf(x.x, (float)(1 << (kHC - 11)), acc[i][j][2 * h]);
                    acc[i][j][2 * h + 1] = fmaf(x.y, (float)(1 << (kHC - 11)), acc[i][j][2 * h + 1]);
                }
                accX[i][j][0] = accX[i][j][1] = 0u;
            }
    }

    if constexpr (SPLIT) {
        float* const W = C + (int64_t)blockIdx.z * M * N;
#pragma unroll
        for (int i = 0; i < NI; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = m0 + er + 16 * i + 8 * h;
                if (r >= M) continue;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j >= jn) continue;
                    *reinterpret_cast<float2*>(W + (int64_t)r * N + n0 + ec + 8 * j) =
                        make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                }
            }
        return;
    } else if constexpr (SILU_EPI == 1) {
        // Fused N==64 readout layer forward: pre = acc + bias; h = silu(pre); eA_out = row_exp(h)
        __syncthreads();
        float* s_am = reinterpret_cast<float*>(hsm);
#pragma unroll
        for (int i = 0; i < NI; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = er + 16 * i + 8 * h;
                const int r = m0 + lr;
                float am = 0.0f;
                if (r < M) {
                    const int e = (int)eA[r];
                    const float pa = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
                    const int sp = sa.per_sp ? __ldg(sa.species + r) : 0;
                    const float* bptr = sa.bias_or_pre + (int64_t)sp * 64;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int c0 = ec + 8 * j;
                        const float2 pb = *reinterpret_cast<const float2*>(inv_sB + c0);
                        const float2 bv = *reinterpret_cast<const float2*>(bptr + c0);
                        const float p0 = fmaf(acc[i][j][2 * h]     * pa, pb.x, bv.x);
                        const float p1 = fmaf(acc[i][j][2 * h + 1] * pa, pb.y, bv.y);
                        *reinterpret_cast<float2*>(sa.pre_out + (int64_t)r * 64 + c0) = make_float2(p0, p1);
                        const float h0 = p0 / (1.0f + __expf(-p0));
                        const float h1 = p1 / (1.0f + __expf(-p1));
                        *reinterpret_cast<float2*>(C + (int64_t)r * 64 + c0) = make_float2(h0, h1);
                        am = fmaxf(am, fmaxf(fabsf(h0), fabsf(h1)));
                    }
                }
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
                if (t == 0) s_am[lr * 2 + wn] = am;
            }
        __syncthreads();
        if (sa.eA_out && tid < BM_T && (m0 + tid) < M) {
            const float m_row = fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]);
            int8_t ev = (int8_t)-127;
            if (m_row > 0.0f && isfinite(m_row)) {
                int ei = (int)((__float_as_uint(m_row) >> 23) & 0xff) - 127 + 1;
                if (ei < -108) ei = -108;
                if (ei > 108) ei = 108;
                ev = (int8_t)ei;
            }
            sa.eA_out[m0 + tid] = ev;
        }
    } else if constexpr (SILU_EPI == 2) {
        // Fused N==64 readout layer backward: dh = acc; dpre = dh * silu'(pre_in); eA_out = row_exp(dpre)
        __syncthreads();
        float* s_am = reinterpret_cast<float*>(hsm);
#pragma unroll
        for (int i = 0; i < NI; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = er + 16 * i + 8 * h;
                const int r = m0 + lr;
                float am = 0.0f;
                if (r < M) {
                    const int e = (int)eA[r];
                    const float pa = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
                    const float* pptr = sa.bias_or_pre + (int64_t)r * 64;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int c0 = ec + 8 * j;
                        const float2 pb = *reinterpret_cast<const float2*>(inv_sB + c0);
                        const float2 pv = *reinterpret_cast<const float2*>(pptr + c0);
                        const float dh0 = (acc[i][j][2 * h]     * pa) * pb.x;
                        const float dh1 = (acc[i][j][2 * h + 1] * pa) * pb.y;
                        const float s0 = 1.0f / (1.0f + __expf(-pv.x));
                        const float s1 = 1.0f / (1.0f + __expf(-pv.y));
                        const float o0 = dh0 * (s0 * (1.0f + pv.x * (1.0f - s0)));
                        const float o1 = dh1 * (s1 * (1.0f + pv.y * (1.0f - s1)));
                        *reinterpret_cast<float2*>(C + (int64_t)r * 64 + c0) = make_float2(o0, o1);
                        am = fmaxf(am, fmaxf(fabsf(o0), fabsf(o1)));
                    }
                }
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
                am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
                if (t == 0) s_am[lr * 2 + wn] = am;
            }
        __syncthreads();
        if (sa.eA_out && tid < BM_T && (m0 + tid) < M) {
            const float m_row = fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]);
            int8_t ev = (int8_t)-127;
            if (m_row > 0.0f && isfinite(m_row)) {
                int ei = (int)((__float_as_uint(m_row) >> 23) & 0xff) - 127 + 1;
                if (ei < -108) ei = -108;
                if (ei > 108) ei = 108;
                ev = (int8_t)ei;
            }
            sa.eA_out[m0 + tid] = ev;
        }
    } else if constexpr (DEG == 0) {
#pragma unroll
        for (int i = 0; i < NI; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = m0 + er + 16 * i + 8 * h;
                if (r >= M) continue;
                const int e = (int)eA[r];
                const float pa = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j >= jn) continue;
                    const int c0 = n0 + ec + 8 * j;
                    const float2 pb = *reinterpret_cast<const float2*>(inv_sB + c0);
                    float2* o = reinterpret_cast<float2*>(C + (int64_t)r * N + c0);
                    float2 v = make_float2((acc[i][j][2 * h] * pa) * pb.x, (acc[i][j][2 * h + 1] * pa) * pb.y);
                    if (beta != 0.0f) {
                        const float2 cv = *o;
                        v.x = fmaf(beta, cv.x, v.x);
                        v.y = fmaf(beta, cv.y, v.y);
                    }
                    *o = v;
                }
            }
    } else if constexpr (!GATE_FWD) {
        constexpr int APT = BM_T / DEG;
        float* sC = reinterpret_cast<float*>(hsm);
        float* sS = sC + BM_T * kHSRow;
        int* sQ = reinterpret_cast<int*>(sS + BM_T);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < NI; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = er + 16 * i + 8 * h, r = m0 + lr;
                const int e = (lr < MT && r < M) ? (int)eA[r] : -127;
                const float pa = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int lc = ec + 8 * j, c0 = n0 + lc;
                    const float2 pb = *reinterpret_cast<const float2*>(inv_sB + c0);
                    *reinterpret_cast<float2*>(sC + lr * kHSRow + lc) =
                        make_float2((acc[i][j][2 * h] * pa) * pb.x, (acc[i][j][2 * h + 1] * pa) * pb.y);
                }
            }
        const int a0 = m0 / DEG, na = min(APT, ga.natoms - a0);
        if (tid < na) {
            const int a = a0 + tid;
            sS[tid] = 1.0f / (ga.D[a] * ga.beta + ga.alpha);
            sQ[tid] = ga.res0_sp ? ga.res0_sp[a] : a;
        }
        __syncthreads();
        gate_bwd_tile<DEG, kHSRow, 2, false>(sC, sS, sQ, a0, na, n0, tid, ga);
        __syncthreads();
        gate_bwd_tile_dD<DEG, kHSRow>(sC, sS, a0, na, tid, ga);
    } else {
        constexpr int l = DEG / 2;
        constexpr int SGS = kHN + 8;
        float* sG = reinterpret_cast<float*>(hsm);
        const int a0 = m0 / DEG, na = (min(m0 + BM_T, M) - 1) / DEG - a0 + 1;
        float* sS = sG + na * SGS;
        int* sQ = reinterpret_cast<int*>(sS + na);
        __syncthreads();
        for (int ai = tid; ai < na; ai += 128) {
            const int a = a0 + ai;
            sS[ai] = 1.0f / (ga.D[a] * ga.beta + ga.alpha);
            sQ[ai] = ga.res0_sp ? ga.res0_sp[a] : a;
        }
        __syncthreads();
        {
            constexpr int B = 4;
            const int cl = tid & 63, c = n0 + cl;
            for (int ai0 = tid >> 6; ai0 < na; ai0 += 2 * B) {
                float lv[B], rv[B];
#pragma unroll
                for (int j = 0; j < B; ++j) {
                    const int ai = ai0 + 2 * j;
                    const bool ok = ai < na && c < N;
                    const int aj = ok ? ai : 0;
                    lv[j] = ok ? ga.lin0[(int64_t)(a0 + aj) * 4 * N + l * N + c] : 0.0f;
                    rv[j] = ok && ga.res0 ? ga.res0[(int64_t)sQ[aj] * 4 * N + l * N + c] : 0.0f;
                }
#pragma unroll
                for (int j = 0; j < B; ++j) {
                    const int ai = ai0 + 2 * j;
                    if (ai < na && c < N) sG[ai * SGS + cl] = ga.cg * sigmoid_f(lv[j] * sS[ai] + rv[j]);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < NI; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = m0 + er + 16 * i + 8 * h;
                if (r >= M) continue;
                const int ai = r / DEG - a0;
                const float s = sS[ai];
                const int e = (int)eA[r];
                const float pa = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (j >= jn) continue;
                    const int cl = ec + 8 * j, c0 = n0 + cl;
                    const float2 pb = *reinterpret_cast<const float2*>(inv_sB + c0);
                    const float2 gg = *reinterpret_cast<const float2*>(sG + ai * SGS + cl);
                    *reinterpret_cast<float2*>(ga.out + (int64_t)r * N + c0) =
                        make_float2(gg.x * (((acc[i][j][2 * h] * pa) * pb.x) * s),
                                    gg.y * (((acc[i][j][2 * h + 1] * pa) * pb.y) * s));
                }
            }
    }
}

// Fused K=128 64x128 kernel (ZERO row_exp launch, 1x A DRAM read!)
__global__ void __launch_bounds__(256, 2)
h2f_64x128_k128_fused(int M, int N, const float* __restrict__ A, const __half* __restrict__ B,
                      const float* __restrict__ inv_sB, float* __restrict__ C, float beta) {
    constexpr int kStageB = 24576;
    extern __shared__ __align__(128) unsigned char sm[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 128;
    const int64_t planeB = (int64_t)N * 128;

    const int my_r = tid >> 2, my_g = tid & 3;
    const bool rok = (m0 + my_r) < M;
    const float4* arow = reinterpret_cast<const float4*>(A + (int64_t)(rok ? m0 + my_r : m0) * 128);
    float am = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const float4 v = rok ? __ldg(arow + s * 4 + my_g) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        am = fmaxf(am, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    const int my_e = (am > 0.0f && isfinite(am)) ? h2r_exp_ceil(am) : -127;
    const float my_sA = (my_e > -127) ? h2r_pow2_f(12 - my_e) : 0.0f;

    __shared__ float s_pa[64];
    if (my_g == 0) s_pa[my_r] = (my_e > -127) ? h2r_pow2_f(my_e - 12) : 0.0f;

    const int br = tid >> 2, bj = tid & 3;
    const __half* bg = B + (int64_t)(n0 + br) * 128 + bj * 8;
    auto sAH = [&](int st) { return sbase + (unsigned)(st * kStageB); };
    auto sAL = [&](int st) { return sbase + (unsigned)(st * kStageB + 4096); };
    auto sBH = [&](int st) { return sbase + (unsigned)(st * kStageB + 8192); };
    auto sBL = [&](int st) { return sbase + (unsigned)(st * kStageB + 16384); };

    auto issue_B = [&](int st, int kt) {
        const int k0 = kt * 32;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            cp16(sBH(st) + swz(br + 64 * i, bj), bg + (int64_t)64 * i * 128 + k0, true);
            cp16(sBL(st) + swz(br + 64 * i, bj), bg + planeB + (int64_t)64 * i * 128 + k0, true);
        }
        asm volatile("cp.async.commit_group;");
    };
    auto split_A_stage = [&](int st, int kt) {
        unsigned char* pAH = sm + st * kStageB;
        unsigned char* pAL = pAH + 4096;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const float4 v = rok ? __ldg(arow + (2 * kt + c) * 4 + my_g) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const int aj4 = c * 4 + my_g;
            unsigned h0, l0, h1, l1;
            split2f(make_float2(v.x, v.y), my_sA, h0, l0);
            split2f(make_float2(v.z, v.w), my_sA, h1, l1);
            const int off = swz(my_r, aj4 >> 1) + ((aj4 & 1) << 3);
            *reinterpret_cast<uint2*>(pAH + off) = make_uint2(h0, h1);
            *reinterpret_cast<uint2*>(pAL + off) = make_uint2(l0, l1);
        }
    };

    float acc[2][4][4] = {};
    unsigned accX[2][4][2] = {};
    const int a_row = wm * 32 + (lane & 15), a_kc = lane >> 4;
    const int b_row = wn * 32 + ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;

    issue_B(0, 0);
    split_A_stage(0, 0);
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
        const int st = kt & 1, nxt = st ^ 1;
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (kt + 1 < 4) issue_B(nxt, kt + 1);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned ah[2][4], al[2][4], bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                ldsm4(sAH(st) + swz(a_row + 16 * i, 2 * kk + a_kc), ah[i]);
                ldsm4(sAL(st) + swz(a_row + 16 * i, 2 * kk + a_kc), al[i]);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                ldsm4(sBH(st) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                ldsm4(sBL(st) + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int jp = j >> 1, jb = 2 * (j & 1);
                    float d[4] = {};
                    mma16816(d, ah[i], bh[jp][jb], bh[jp][jb + 1]);
                    jam4(d);
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][j][e] += d[e];
                    mma16816h(accX[i][j], ah[i], bl[jp][jb], bl[jp][jb + 1]);
                    mma16816h(accX[i][j], al[i], bc[jp][jb], bc[jp][jb + 1]);
                }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                    acc[i][j][2 * h] = fmaf(x.x, (float)(1 << (kHC - 11)), acc[i][j][2 * h]);
                    acc[i][j][2 * h + 1] = fmaf(x.y, (float)(1 << (kHC - 11)), acc[i][j][2 * h + 1]);
                }
                accX[i][j][0] = accX[i][j][1] = 0u;
            }
        if (kt + 1 < 4) split_A_stage(nxt, kt + 1);
    }

    const int g = lane >> 2, t = lane & 3;
    const int er = wm * 32 + g, ec = wn * 32 + 2 * t;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h, r = m0 + lr;
            if (r >= M) continue;
            const float pa = s_pa[lr];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = n0 + ec + 8 * j;
                const float2 pb = *reinterpret_cast<const float2*>(inv_sB + c0);
                float2* o = reinterpret_cast<float2*>(C + (int64_t)r * N + c0);
                float2 v = make_float2((acc[i][j][2 * h] * pa) * pb.x, (acc[i][j][2 * h + 1] * pa) * pb.y);
                if (beta != 0.0f) {
                    const float2 cv = *o;
                    v.x = fmaf(beta, cv.x, v.x);
                    v.y = fmaf(beta, cv.y, v.y);
                }
                *o = v;
            }
        }
}

// Fused ILMlip Readout Tail (depth==2, W==128, need_grad==true):
// Collapses 7 kernels (2x bias_silu, W1 fwd, energy_ro+ZBL, 2x silu_bwd, W1 bwd) into ONE 48 KB SMEM CTA!
template <bool ZBL_SW>
__global__ void __launch_bounds__(256, 2)
h2f_ilmlip_ro_tail_w128_kernel(
    int M, float inv_zc, const int* __restrict__ species,
    const float* __restrict__ pre0, const float* __restrict__ alpha_Z,
    const float* __restrict__ b1, const float* __restrict__ u_Z,
    const __half* __restrict__ B_fwd, const float* __restrict__ inv_sB_fwd,
    const __half* __restrict__ B_bwd, const float* __restrict__ inv_sB_bwd,
    const double* __restrict__ e_zbl, double* __restrict__ energy, double* __restrict__ atom_energy,
    float* __restrict__ w_zbl_out, float* __restrict__ dpre0, int8_t* __restrict__ eA_dpre0) {
    // Dynamic SMEM: 49,152 B = s_AH[4*4096] (0..16383) + s_AL[4*4096] (16384..32767) + s_BH[8192] (32768..40959) + s_BL[8192] (40960..49151)
    extern __shared__ __align__(128) unsigned char sm[];
    __shared__ float s_pa[64];
    __shared__ float s_sw[64];
    __shared__ int   s_sp[64];

    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;
    const int m0 = blockIdx.y * 64;
    constexpr int64_t planeB = 128 * 128;

    const int my_r = tid >> 2, my_g = tid & 3;
    const bool rok = (m0 + my_r) < M;
    if (my_g == 0) s_sp[my_r] = rok ? species[m0 + my_r] : 0;
    __syncthreads();
    const int sp_r = s_sp[my_r];

    // Step 1: Load pre0 + alpha_Z -> silu -> h0, compute row exp, split directly into s_AH/s_AL!
    const float4* p0_row = reinterpret_cast<const float4*>(pre0 + (int64_t)(rok ? m0 + my_r : m0) * 128);
    const float4* az_row = reinterpret_cast<const float4*>(alpha_Z + (int64_t)sp_r * 128);
    float4 h0_v[8];
    float am = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (rok) {
            const float4 p = __ldg(p0_row + s * 4 + my_g);
            const float4 a = __ldg(az_row + s * 4 + my_g);
            const float px = p.x + a.x, py = p.y + a.y, pz = p.z + a.z, pw = p.w + a.w;
            const float hx = px / (1.0f + __expf(-px));
            const float hy = py / (1.0f + __expf(-py));
            const float hz = pz / (1.0f + __expf(-pz));
            const float hw = pw / (1.0f + __expf(-pw));
            h0_v[s] = make_float4(hx, hy, hz, hw);
            am = fmaxf(am, fmaxf(fmaxf(fabsf(hx), fabsf(hy)), fmaxf(fabsf(hz), fabsf(hw))));
        } else {
            h0_v[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    const int my_e = (am > 0.0f && isfinite(am)) ? h2r_exp_ceil(am) : -127;
    const float my_sA = (my_e > -127) ? h2r_pow2_f(12 - my_e) : 0.0f;
    if (my_g == 0) s_pa[my_r] = (my_e > -127) ? h2r_pow2_f(my_e - 12) : 0.0f;

    const int br = tid >> 2, bj = tid & 3;
    const int a_row = wm * 32 + (lane & 15), a_kc = lane >> 4;
    const int b_row = wn * 32 + ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;

    // Step 2: Double-buffered Matmul 1 (pre1 = h0 @ W1) using 2x24 KB stages (h0_v[8] already in regs!)
    constexpr int kStageB = 24576;
    auto sAH_db = [&](int st) { return sbase + (unsigned)(st * kStageB); };
    auto sAL_db = [&](int st) { return sbase + (unsigned)(st * kStageB + 4096); };
    auto sBH_db = [&](int st) { return sbase + (unsigned)(st * kStageB + 8192); };
    auto sBL_db = [&](int st) { return sbase + (unsigned)(st * kStageB + 16384); };
    const __half* bg_f = B_fwd + (int64_t)br * 128 + bj * 8;
    auto issue_Bf = [&](int st, int kt) {
        const int k0 = kt * 32;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            cp16(sBH_db(st) + swz(br + 64 * i, bj), bg_f + (int64_t)64 * i * 128 + k0, true);
            cp16(sBL_db(st) + swz(br + 64 * i, bj), bg_f + planeB + (int64_t)64 * i * 128 + k0, true);
        }
        asm volatile("cp.async.commit_group;");
    };
    auto split_h0_stage = [&](int st, int kt) {
        unsigned char* pAH = sm + st * kStageB;
        unsigned char* pAL = pAH + 4096;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const float4 v = h0_v[2 * kt + c];
            const int aj4 = c * 4 + my_g;
            unsigned h0, l0, h1, l1;
            split2f(make_float2(v.x, v.y), my_sA, h0, l0);
            split2f(make_float2(v.z, v.w), my_sA, h1, l1);
            const int off = swz(my_r, aj4 >> 1) + ((aj4 & 1) << 3);
            *reinterpret_cast<uint2*>(pAH + off) = make_uint2(h0, h1);
            *reinterpret_cast<uint2*>(pAL + off) = make_uint2(l0, l1);
        }
    };

    float acc1[2][4][4] = {};
    {
        unsigned accX[2][4][2] = {};
        issue_Bf(0, 0);
        split_h0_stage(0, 0);
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const int st = kt & 1, nxt = st ^ 1;
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (kt + 1 < 4) issue_Bf(nxt, kt + 1);
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned ah[2][4], al[2][4], bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    ldsm4(sAH_db(st) + swz(a_row + 16 * i, 2 * kk + a_kc), ah[i]);
                    ldsm4(sAL_db(st) + swz(a_row + 16 * i, 2 * kk + a_kc), al[i]);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(sBH_db(st) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(sBL_db(st) + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int jp = j >> 1, jb = 2 * (j & 1);
                        float d[4] = {};
                        mma16816(d, ah[i], bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc1[i][j][e] += d[e];
                        mma16816h(accX[i][j], ah[i], bl[jp][jb], bl[jp][jb + 1]);
                        mma16816h(accX[i][j], al[i], bc[jp][jb], bc[jp][jb + 1]);
                    }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        acc1[i][j][2 * h]     = fmaf(x.x, (float)(1 << (kHC - 11)), acc1[i][j][2 * h]);
                        acc1[i][j][2 * h + 1] = fmaf(x.y, (float)(1 << (kHC - 11)), acc1[i][j][2 * h + 1]);
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
            if (kt + 1 < 4) split_h0_stage(nxt, kt + 1);
        }
    }
    __syncthreads();

    const unsigned sBH = sbase + 32768u;
    const unsigned sBL = sbase + 40960u;
    auto run_matmul = [&](const __half* Bmat, float acc[2][4][4]) {
        const __half* bg = Bmat + (int64_t)br * 128 + bj * 8;
        unsigned accX[2][4][2] = {};
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const int k0 = kt * 32;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                cp16(sBH + swz(br + 64 * i, bj), bg + (int64_t)64 * i * 128 + k0, true);
                cp16(sBL + swz(br + 64 * i, bj), bg + planeB + (int64_t)64 * i * 128 + k0, true);
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            const unsigned sAH_k = sbase + (unsigned)(kt * 4096);
            const unsigned sAL_k = sbase + (unsigned)(16384 + kt * 4096);
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned ah[2][4], al[2][4], bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    ldsm4(sAH_k + swz(a_row + 16 * i, 2 * kk + a_kc), ah[i]);
                    ldsm4(sAL_k + swz(a_row + 16 * i, 2 * kk + a_kc), al[i]);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(sBH + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(sBL + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int jp = j >> 1, jb = 2 * (j & 1);
                        float d[4] = {};
                        mma16816(d, ah[i], bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[i][j][e] += d[e];
                        mma16816h(accX[i][j], ah[i], bl[jp][jb], bl[jp][jb + 1]);
                        mma16816h(accX[i][j], al[i], bc[jp][jb], bc[jp][jb + 1]);
                    }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        acc[i][j][2 * h]     = fmaf(x.x, (float)(1 << (kHC - 11)), acc[i][j][2 * h]);
                        acc[i][j][2 * h + 1] = fmaf(x.y, (float)(1 << (kHC - 11)), acc[i][j][2 * h + 1]);
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
            __syncthreads();
        }
    };

    // Step 3: Fused bias_silu(pre1 + b1) + dot u_Z + ZBL switch + silu_bwd -> dpre1 split into s_AH/s_AL!
    // Reuse s_BH (sm + 32768, 8 KB free between matmuls) for s_dot[64][4], s_am[64][4], s_sA[64]
    float* s_dot = reinterpret_cast<float*>(sm + 32768);        // 256 floats
    float* s_am  = reinterpret_cast<float*>(sm + 32768 + 1024); // 256 floats
    float* s_sA  = reinterpret_cast<float*>(sm + 32768 + 2048); // 64 floats

    const int g = lane >> 2, t = lane & 3;
    const int er = wm * 32 + g, ec = wn * 32 + 2 * t;

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            const float pa = s_pa[lr];
            const int sp = s_sp[lr];
            const float* urow = u_Z + (int64_t)sp * 128;
            float dot_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ec + 8 * j;
                const float2 pb = *reinterpret_cast<const float2*>(inv_sB_fwd + c0);
                const float2 bv = *reinterpret_cast<const float2*>(b1 + c0);
                const float2 uv = *reinterpret_cast<const float2*>(urow + c0);
                const float p0 = fmaf(acc1[i][j][2 * h]     * pa, pb.x, bv.x);
                const float p1 = fmaf(acc1[i][j][2 * h + 1] * pa, pb.y, bv.y);
                const float s0 = 1.0f / (1.0f + __expf(-p0));
                const float s1 = 1.0f / (1.0f + __expf(-p1));
                dot_t = fmaf(p0 * s0, uv.x, fmaf(p1 * s1, uv.y, dot_t));
                // Save u * dsilu(p) directly in acc1 (will multiply by sw[lr] after ZBL switch!)
                acc1[i][j][2 * h]     = uv.x * (s0 * (1.0f + p0 * (1.0f - s0)));
                acc1[i][j][2 * h + 1] = uv.y * (s1 * (1.0f + p1 * (1.0f - s1)));
            }
            dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 2);
            dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 1);
            if (t == 0) s_dot[lr * 4 + wn] = (r < M) ? dot_t : 0.0f;
        }
    __syncthreads();

    if (tid < 64) {
        const int r = m0 + tid;
        double e_atom = 0.0;
        if (r < M) {
            const float e_ml = s_dot[tid * 4 + 0] + s_dot[tid * 4 + 1] + s_dot[tid * 4 + 2] + s_dot[tid * 4 + 3];
            float sw = 1.0f;
            float wz = 1.0f;
            double node_inter = (double)e_ml;
            if constexpr (ZBL_SW) {
                const double ez = e_zbl[r];
                const double inv_zc_d = (double)inv_zc;
                const double xc = ez * inv_zc_d;
                if (xc <= 0.0) {
                    sw = 1.0f; wz = 1.0f; node_inter = (double)e_ml + ez;
                } else if (xc >= 1.0) {
                    sw = 0.0f; wz = 1.0f; node_inter = ez;
                } else {
                    const double xc2 = xc * xc, xc3 = xc2 * xc;
                    const double sw_d = 1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2;
                    const double omx = 1.0 - xc;
                    sw = (float)sw_d;
                    wz = (float)(1.0 + (double)e_ml * (-30.0 * xc2 * omx * omx) * inv_zc_d);
                    node_inter = sw_d * (double)e_ml + ez;
                }
                w_zbl_out[r] = wz;
            }
            s_sw[tid] = sw;
            e_atom = node_inter;
            if (atom_energy) atom_energy[r] += e_atom;
        } else {
            s_sw[tid] = 0.0f;
        }
        // Warp-reduce CTA energy (only 2 atomics per 64 atoms instead of 64!)
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) e_atom += __shfl_xor_sync(0xffffffffu, e_atom, off);
        if ((tid & 31) == 0 && e_atom != 0.0) atomicAdd(energy, e_atom);
    }
    __syncthreads();

    // Scale dpre1 by sw[lr] and compute per-row max abs for FP16x2 split
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const float sw = s_sw[lr];
            float am_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float v0 = acc1[i][j][2 * h]     * sw;
                const float v1 = acc1[i][j][2 * h + 1] * sw;
                acc1[i][j][2 * h]     = v0;
                acc1[i][j][2 * h + 1] = v1;
                am_t = fmaxf(am_t, fmaxf(fabsf(v0), fabsf(v1)));
            }
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
            if (t == 0) s_am[lr * 4 + wn] = am_t;
        }
    __syncthreads();

    if (tid < 64) {
        const float am_r = fmaxf(fmaxf(s_am[tid * 4 + 0], s_am[tid * 4 + 1]),
                                 fmaxf(s_am[tid * 4 + 2], s_am[tid * 4 + 3]));
        const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
        s_pa[tid] = (erow > -127) ? h2r_pow2_f(erow - 12) : 0.0f;
        s_sA[tid] = (erow > -127) ? h2r_pow2_f(12 - erow) : 0.0f;
    }
    __syncthreads();

    // Split dpre1 directly into s_AH/s_AL using exact swz layout!
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const float sa = s_sA[lr];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ec + 8 * j;  // even in 0..126
                const int kt = c0 >> 5;
                const int kl = c0 & 31;
                unsigned h_pair, l_pair;
                split2f(make_float2(acc1[i][j][2 * h], acc1[i][j][2 * h + 1]), sa, h_pair, l_pair);
                const int off = swz(lr, kl >> 3) + ((kl & 7) << 1);
                *reinterpret_cast<unsigned*>(sm + kt * 4096 + off)         = h_pair;
                *reinterpret_cast<unsigned*>(sm + 16384 + kt * 4096 + off) = l_pair;
            }
        }
    __syncthreads();

    // Step 4: Matmul 2: dh0 = dpre1 @ W1^T
    float acc2[2][4][4] = {};
    run_matmul(B_bwd, acc2);

    // Step 5: Fused silu_bwd(dh0, pre0 + alpha_Z) -> dpre0 (+ optional per-row max-exp eA_dpre0)!
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            float am_t = 0.0f;
            if (r < M) {
                const float pa = s_pa[lr];
                const int sp = s_sp[lr];
                const float* p0_r = pre0 + (int64_t)r * 128;
                const float* az_r = alpha_Z + (int64_t)sp * 128;
                float* out_r = dpre0 + (int64_t)r * 128;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c0 = ec + 8 * j;
                    const float2 pb = *reinterpret_cast<const float2*>(inv_sB_bwd + c0);
                    const float2 p0v = *reinterpret_cast<const float2*>(p0_r + c0);
                    const float2 azv = *reinterpret_cast<const float2*>(az_r + c0);
                    const float px = p0v.x + azv.x, py = p0v.y + azv.y;
                    const float sx = 1.0f / (1.0f + __expf(-px));
                    const float sy = 1.0f / (1.0f + __expf(-py));
                    const float dhx = (acc2[i][j][2 * h]     * pa) * pb.x;
                    const float dhy = (acc2[i][j][2 * h + 1] * pa) * pb.y;
                    const float ox = dhx * (sx * (1.0f + px * (1.0f - sx)));
                    const float oy = dhy * (sy * (1.0f + py * (1.0f - sy)));
                    *reinterpret_cast<float2*>(out_r + c0) = make_float2(ox, oy);
                    am_t = fmaxf(am_t, fmaxf(fabsf(ox), fabsf(oy)));
                }
            }
            if (eA_dpre0) {
                am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
                am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
                if (t == 0) s_am[lr * 4 + wn] = am_t;
            }
        }
    if (eA_dpre0) {
        __syncthreads();
        if (tid < 64 && (m0 + tid) < M) {
            const float am_r = fmaxf(fmaxf(s_am[tid * 4 + 0], s_am[tid * 4 + 1]),
                                     fmaxf(s_am[tid * 4 + 2], s_am[tid * 4 + 3]));
            eA_dpre0[m0 + tid] = (int8_t)((am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127);
        }
    }
}

// =================================================================================================
// Fused ILMlip Readout Tail for depth==2, width==64, need_grad==true:
//   Collapses 7 kernels into ONE 128-thread CTA (24 KB SMEM -> 4 CTAs/SM)!
// =================================================================================================
template <bool ZBL_SW>
__global__ void __launch_bounds__(128, 4) h2f_ilmlip_ro_tail_w64_kernel(
    int M, float inv_zc, const int* __restrict__ species,
    const float* __restrict__ pre0, const float* __restrict__ alpha_Z,
    const float* __restrict__ b1, const float* __restrict__ u_Z,
    const __half* __restrict__ B_fwd, const float* __restrict__ inv_sB_fwd,
    const __half* __restrict__ B_bwd, const float* __restrict__ inv_sB_bwd,
    const double* __restrict__ e_zbl, double* __restrict__ energy, double* __restrict__ atom_energy,
    float* __restrict__ w_zbl_out, float* __restrict__ dpre0, int8_t* __restrict__ eA_dpre0) {
    extern __shared__ __align__(128) unsigned char sm[];
    __shared__ int   s_sp[64];
    __shared__ float s_pa[64];
    __shared__ float s_sw[64];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * 64;
    constexpr int64_t planeB = 64 * 64;

    const int my_r = tid >> 1, my_g = tid & 1;
    const bool rok = (m0 + my_r) < M;
    if (my_g == 0) s_sp[my_r] = rok ? species[m0 + my_r] : 0;
    __syncthreads();
    const int sp_r = s_sp[my_r];

    // Step 1: Load pre0 + alpha_Z -> silu -> h0, row exp, split directly into sAH[0..1] / sAL[0..1]
    const float4* p0_row = reinterpret_cast<const float4*>(pre0 + (int64_t)(rok ? m0 + my_r : m0) * 64);
    const float4* az_row = reinterpret_cast<const float4*>(alpha_Z + (int64_t)sp_r * 64);
    float4 h0_v[8];
    float am = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (rok) {
            const float4 p = __ldg(p0_row + s * 2 + my_g);
            const float4 a = __ldg(az_row + s * 2 + my_g);
            const float px = p.x + a.x, py = p.y + a.y, pz = p.z + a.z, pw = p.w + a.w;
            const float hx = px / (1.0f + __expf(-px));
            const float hy = py / (1.0f + __expf(-py));
            const float hz = pz / (1.0f + __expf(-pz));
            const float hw = pw / (1.0f + __expf(-pw));
            h0_v[s] = make_float4(hx, hy, hz, hw);
            am = fmaxf(am, fmaxf(fmaxf(fabsf(hx), fabsf(hy)), fmaxf(fabsf(hz), fabsf(hw))));
        } else {
            h0_v[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    const int my_e = (am > 0.0f && isfinite(am)) ? h2r_exp_ceil(am) : -127;
    const float my_sA = (my_e > -127) ? h2r_pow2_f(12 - my_e) : 0.0f;
    if (my_g == 0) s_pa[my_r] = (my_e > -127) ? h2r_pow2_f(my_e - 12) : 0.0f;

#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int col4 = s * 2 + my_g;
        const int kt = col4 >> 3;
        const int aj4 = col4 & 7;
        unsigned h0, l0, h1, l1;
        split2f(make_float2(h0_v[s].x, h0_v[s].y), my_sA, h0, l0);
        split2f(make_float2(h0_v[s].z, h0_v[s].w), my_sA, h1, l1);
        const int off = swz(my_r, aj4 >> 1) + ((aj4 & 1) << 3);
        *reinterpret_cast<uint2*>(sm + kt * 4096 + off)        = make_uint2(h0, h1);
        *reinterpret_cast<uint2*>(sm + 8192 + kt * 4096 + off) = make_uint2(l0, l1);
    }
    __syncthreads();

    const int br = tid >> 2, bj = tid & 3;
    const int a_row = wm * 32 + (lane & 15), a_kc = lane >> 4;
    const int b_row = wn * 32 + ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;
    const unsigned sBH = sbase + 16384u;
    const unsigned sBL = sbase + 20480u;

    auto run_matmul = [&](const __half* Bmat, float acc[2][4][4]) {
        const __half* bg = Bmat + (int64_t)br * 64 + bj * 8;
        unsigned accX[2][4][2] = {};
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            const int k0 = kt * 32;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                cp16(sBH + swz(br + 32 * i, bj), bg + (int64_t)32 * i * 64 + k0, true);
                cp16(sBL + swz(br + 32 * i, bj), bg + planeB + (int64_t)32 * i * 64 + k0, true);
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            const unsigned sAH_k = sbase + (unsigned)(kt * 4096);
            const unsigned sAL_k = sbase + (unsigned)(8192 + kt * 4096);
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned ah[2][4], al[2][4], bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    ldsm4(sAH_k + swz(a_row + 16 * i, 2 * kk + a_kc), ah[i]);
                    ldsm4(sAL_k + swz(a_row + 16 * i, 2 * kk + a_kc), al[i]);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(sBH + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(sBL + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int jp = j >> 1, jb = 2 * (j & 1);
                        float d[4] = {};
                        mma16816(d, ah[i], bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[i][j][e] += d[e];
                        mma16816h(accX[i][j], ah[i], bl[jp][jb], bl[jp][jb + 1]);
                        mma16816h(accX[i][j], al[i], bc[jp][jb], bc[jp][jb + 1]);
                    }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        acc[i][j][2 * h]     = fmaf(x.x, (float)(1 << (kHC - 11)), acc[i][j][2 * h]);
                        acc[i][j][2 * h + 1] = fmaf(x.y, (float)(1 << (kHC - 11)), acc[i][j][2 * h + 1]);
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
            __syncthreads();
        }
    };

    // Step 2: Matmul 1: pre1 = h0 @ W1
    float acc1[2][4][4] = {};
    run_matmul(B_fwd, acc1);

    // Step 3: Fused bias_silu + dot u_Z + ZBL switch + silu_bwd -> split dpre1 into sAH/sAL
    float* s_dot = reinterpret_cast<float*>(sm + 16384);        // 128 floats (512 B)
    float* s_am  = reinterpret_cast<float*>(sm + 16384 + 512);  // 128 floats (512 B)
    float* s_sA  = reinterpret_cast<float*>(sm + 16384 + 1024); // 64 floats (256 B)

    const int g = lane >> 2, t = lane & 3;
    const int er = wm * 32 + g, ec = wn * 32 + 2 * t;

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            const float pa = s_pa[lr];
            const int sp = s_sp[lr];
            const float* urow = u_Z + (int64_t)sp * 64;
            float dot_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ec + 8 * j;
                const float2 pb = *reinterpret_cast<const float2*>(inv_sB_fwd + c0);
                const float2 bv = *reinterpret_cast<const float2*>(b1 + c0);
                const float2 uv = *reinterpret_cast<const float2*>(urow + c0);
                const float p0 = fmaf(acc1[i][j][2 * h]     * pa, pb.x, bv.x);
                const float p1 = fmaf(acc1[i][j][2 * h + 1] * pa, pb.y, bv.y);
                const float s0 = 1.0f / (1.0f + __expf(-p0));
                const float s1 = 1.0f / (1.0f + __expf(-p1));
                dot_t = fmaf(p0 * s0, uv.x, fmaf(p1 * s1, uv.y, dot_t));
                acc1[i][j][2 * h]     = uv.x * (s0 * (1.0f + p0 * (1.0f - s0)));
                acc1[i][j][2 * h + 1] = uv.y * (s1 * (1.0f + p1 * (1.0f - s1)));
            }
            dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 2);
            dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 1);
            if (t == 0) s_dot[lr * 2 + wn] = (r < M) ? dot_t : 0.0f;
        }
    __syncthreads();

    if (tid < 64) {
        const int r = m0 + tid;
        double e_atom = 0.0;
        if (r < M) {
            const float e_ml = s_dot[tid * 2 + 0] + s_dot[tid * 2 + 1];
            float sw = 1.0f, wz = 1.0f;
            double node_inter = (double)e_ml;
            if constexpr (ZBL_SW) {
                const double ez = e_zbl[r];
                const double inv_zc_d = (double)inv_zc;
                const double xc = ez * inv_zc_d;
                if (xc <= 0.0) {
                    sw = 1.0f; wz = 1.0f; node_inter = (double)e_ml + ez;
                } else if (xc >= 1.0) {
                    sw = 0.0f; wz = 1.0f; node_inter = ez;
                } else {
                    const double xc2 = xc * xc, xc3 = xc2 * xc;
                    const double sw_d = 1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2;
                    const double omx = 1.0 - xc;
                    sw = (float)sw_d;
                    wz = (float)(1.0 + (double)e_ml * (-30.0 * xc2 * omx * omx) * inv_zc_d);
                    node_inter = sw_d * (double)e_ml + ez;
                }
                w_zbl_out[r] = wz;
            }
            s_sw[tid] = sw;
            e_atom = node_inter;
            if (atom_energy) atom_energy[r] += e_atom;
        } else {
            s_sw[tid] = 0.0f;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) e_atom += __shfl_xor_sync(0xffffffffu, e_atom, off);
        if ((tid & 31) == 0 && e_atom != 0.0) atomicAdd(energy, e_atom);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const float sw = s_sw[lr];
            float am_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float v0 = acc1[i][j][2 * h]     * sw;
                const float v1 = acc1[i][j][2 * h + 1] * sw;
                acc1[i][j][2 * h]     = v0;
                acc1[i][j][2 * h + 1] = v1;
                am_t = fmaxf(am_t, fmaxf(fabsf(v0), fabsf(v1)));
            }
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
            if (t == 0) s_am[lr * 2 + wn] = am_t;
        }
    __syncthreads();

    if (tid < 64) {
        const float am_r = fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]);
        const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
        s_pa[tid] = (erow > -127) ? h2r_pow2_f(erow - 12) : 0.0f;
        s_sA[tid] = (erow > -127) ? h2r_pow2_f(12 - erow) : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const float sa = s_sA[lr];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ec + 8 * j;
                const int kt = c0 >> 5;
                const int kl = c0 & 31;
                unsigned h_pair, l_pair;
                split2f(make_float2(acc1[i][j][2 * h], acc1[i][j][2 * h + 1]), sa, h_pair, l_pair);
                const int off = swz(lr, kl >> 3) + ((kl & 7) << 1);
                *reinterpret_cast<unsigned*>(sm + kt * 4096 + off)        = h_pair;
                *reinterpret_cast<unsigned*>(sm + 8192 + kt * 4096 + off) = l_pair;
            }
        }
    __syncthreads();

    // Step 4: Matmul 2: dh0 = dpre1 @ W1^T
    float acc2[2][4][4] = {};
    run_matmul(B_bwd, acc2);

    // Step 5: Fused silu_bwd(dh0, pre0 + alpha_Z) -> dpre0 + cached eA_dpre0 (eliminates h2r_row_exp_kernel!)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            float am_t = 0.0f;
            if (r < M) {
                const float pa = s_pa[lr];
                const int sp = s_sp[lr];
                const float* p0_r = pre0 + (int64_t)r * 64;
                const float* az_r = alpha_Z + (int64_t)sp * 64;
                float* out_r = dpre0 + (int64_t)r * 64;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c0 = ec + 8 * j;
                    const float2 pb = *reinterpret_cast<const float2*>(inv_sB_bwd + c0);
                    const float2 p0v = *reinterpret_cast<const float2*>(p0_r + c0);
                    const float2 azv = *reinterpret_cast<const float2*>(az_r + c0);
                    const float px = p0v.x + azv.x, py = p0v.y + azv.y;
                    const float sx = 1.0f / (1.0f + __expf(-px));
                    const float sy = 1.0f / (1.0f + __expf(-py));
                    const float dhx = (acc2[i][j][2 * h]     * pa) * pb.x;
                    const float dhy = (acc2[i][j][2 * h + 1] * pa) * pb.y;
                    const float ox = dhx * (sx * (1.0f + px * (1.0f - sx)));
                    const float oy = dhy * (sy * (1.0f + py * (1.0f - sy)));
                    *reinterpret_cast<float2*>(out_r + c0) = make_float2(ox, oy);
                    am_t = fmaxf(am_t, fmaxf(fabsf(ox), fabsf(oy)));
                }
            }
            if (eA_dpre0) {
                am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
                am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
                if (t == 0) s_am[lr * 2 + wn] = am_t;
            }
        }
    if (eA_dpre0) {
        __syncthreads();
        if (tid < 64 && m0 + tid < M) {
            const float am_r = fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]);
            eA_dpre0[m0 + tid] = (int8_t)((am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127);
        }
    }
}

// =================================================================================================
// Single-Kernel Full Fused ILMlip Readout for depth==1, width in {32, 64}, need_grad==true:
//   x[N,K] @ W0 -> +alpha_Z -> silu -> .u_Z (+ZBL) -> silu_bwd -> @ W0^T -> dx[N,K]
//   ZERO pre0/dpre0/eA_dpre0 DRAM traffic, 3 kernels -> 1 launch, 32 KB SMEM -> 3 CTAs/SM!
// =================================================================================================
template <int RW, bool ZBL_SW>
__global__ void __launch_bounds__(128, 3) h2f_ilmlip_ro1_fused_kernel(
    int M, int K, float inv_zc, const int* __restrict__ species,
    const float* __restrict__ x, const int8_t* __restrict__ eA_x,
    const float* __restrict__ alpha_Z, const float* __restrict__ u_Z,
    const __half* __restrict__ W0_B, const float* __restrict__ W0_inv_sB,
    const __half* __restrict__ W0_BT, const float* __restrict__ W0_inv_sBT,
    const double* __restrict__ e_zbl, double* __restrict__ energy, double* __restrict__ atom_energy,
    float* __restrict__ w_zbl_out, float* __restrict__ dx) {
    static_assert(RW == 32 || RW == 64, "RW must be 32 or 64");
    extern __shared__ __align__(128) unsigned char sm[];
    __shared__ int   s_sp[64];
    __shared__ float s_pa[64];
    __shared__ float s_sA[64];

    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * 64;
    const int g = lane >> 2, t = lane & 3;
    const int er = wm * 32 + g, ec = wn * 32 + 2 * t;

    if (tid < 64) {
        const int r = m0 + tid;
        const bool ok = (r < M);
        s_sp[tid] = ok ? species[r] : 0;
        const int e = ok ? (int)eA_x[r] : -127;
        s_sA[tid] = (e > -127) ? h2r_pow2_f(12 - e) : 0.0f;
        s_pa[tid] = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
    }

    // Phase 1: pre0[64, RW] = x[64, K] @ W0[K, RW] (K % 64 == 0, B_fwd [2][64][K])
    constexpr int kTileA = 64 * kHK;              // 2048 halves (4096 B); FP32 A tile = 2*kTileA halves = 8192 B
    constexpr int kStageH = 2 * kTileA + 2 * kHTileB;  // 8192 halves = 16384 B
    const int64_t planeB_fwd = (int64_t)64 * K;
    const int fr = tid >> 3, fj = tid & 7;
    const int cr = tid >> 2, cj = tid & 3;
    const __half* bg_f = W0_B + (int64_t)cr * K + cj * 8;
    const int b_row = wn * 32 + ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;

    auto stage_bf = [&](int st, int p) { return sbase + (unsigned)((st * kStageH + 2 * kTileA + p * kHTileB) * 2); };
    auto issue_fwd = [&](int kt) {
        const int st = kt & 1, k0 = kt * kHK;
        const unsigned sa = sbase + (unsigned)(st * kStageH * 2);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = fr + 16 * i;
            const bool rok = (m0 + r) < M;
            cp16(sa + r * 128 + ((fj ^ (r & 7)) << 4), x + (int64_t)(rok ? m0 + r : m0) * K + k0 + 4 * fj, rok);
        }
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                cp16(stage_bf(st, p) + swz(cr + 32 * i, cj), bg_f + p * planeB_fwd + (int64_t)32 * i * K + k0, true);
        asm volatile("cp.async.commit_group;");
    };

    float acc[2][4][4] = {};
    {
        unsigned accX[2][4][2] = {};
        const int nk = K / kHK;
        issue_fwd(0);
        for (int kt = 0; kt < nk; ++kt) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (kt + 1 < nk) issue_fwd(kt + 1);
            if constexpr (RW == 32) {
                if (wn == 1) continue;
            }
            const int st = kt & 1;
            const unsigned char* sa = sm + st * kStageH * 2 + er * 128 + (t & 1) * 8;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(stage_bf(st, 0) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(stage_bf(st, 1) + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
                const int ja = 4 * kk + (t >> 1);
                const int o0 = (ja ^ g) << 4, o1 = ((ja + 2) ^ g) << 4;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const unsigned char* s = sa + 16 * i * 128;
                    unsigned ah[4], al[4];
                    const float sc0 = s_sA[er + 16 * i], sc1 = s_sA[er + 16 * i + 8];
                    split2f(*reinterpret_cast<const float2*>(s + o0), sc0, ah[0], al[0]);
                    split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o0), sc1, ah[1], al[1]);
                    split2f(*reinterpret_cast<const float2*>(s + o1), sc0, ah[2], al[2]);
                    split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o1), sc1, ah[3], al[3]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int jp = j >> 1, jb = 2 * (j & 1);
                        float d[4] = {};
                        mma16816(d, ah, bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[i][j][e] += d[e];
                        mma16816h(accX[i][j], ah, bl[jp][jb], bl[jp][jb + 1]);
                        mma16816h(accX[i][j], al, bc[jp][jb], bc[jp][jb + 1]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        acc[i][j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), acc[i][j][2 * h]);
                        acc[i][j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), acc[i][j][2 * h + 1]);
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
        }
    }
    __syncthreads();

    // Phase 2: In-register bias+silu + dot u_Z + ZBL switch + silu_bwd -> split dpre0 into SMEM sAH/sAL!
    float* s_dot = reinterpret_cast<float*>(sm + 16384);
    float* s_am  = reinterpret_cast<float*>(sm + 16384 + 512);
    float* s_sw  = reinterpret_cast<float*>(sm + 16384 + 1024);

    if (RW == 64 || wn == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = er + 16 * i + 8 * h;
                const int r = m0 + lr;
                const float pa = s_pa[lr];
                const int sp = s_sp[lr];
                const float* azrow = alpha_Z + (int64_t)sp * RW;
                const float* urow  = u_Z + (int64_t)sp * RW;
                float dot_t = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c0 = ec + 8 * j;
                    const float2 pb = *reinterpret_cast<const float2*>(W0_inv_sB + c0);
                    const float2 az = *reinterpret_cast<const float2*>(azrow + c0);
                    const float2 uv = *reinterpret_cast<const float2*>(urow + c0);
                    const float p0 = fmaf(acc[i][j][2 * h]     * pa, pb.x, az.x);
                    const float p1 = fmaf(acc[i][j][2 * h + 1] * pa, pb.y, az.y);
                    const float s0 = 1.0f / (1.0f + __expf(-p0));
                    const float s1 = 1.0f / (1.0f + __expf(-p1));
                    dot_t = fmaf(p0 * s0, uv.x, fmaf(p1 * s1, uv.y, dot_t));
                    acc[i][j][2 * h]     = uv.x * (s0 * (1.0f + p0 * (1.0f - s0)));
                    acc[i][j][2 * h + 1] = uv.y * (s1 * (1.0f + p1 * (1.0f - s1)));
                }
                dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 2);
                dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 1);
                if (t == 0) s_dot[lr * 2 + wn] = (r < M) ? dot_t : 0.0f;
            }
    }
    __syncthreads();

    if (tid < 64) {
        const int r = m0 + tid;
        double e_atom = 0.0;
        if (r < M) {
            const float e_ml = (RW == 64) ? (s_dot[tid * 2 + 0] + s_dot[tid * 2 + 1]) : s_dot[tid * 2 + 0];
            float sw = 1.0f, wz = 1.0f;
            double node_inter = (double)e_ml;
            if constexpr (ZBL_SW) {
                const double ez = e_zbl[r];
                const double inv_zc_d = (double)inv_zc;
                const double xc = ez * inv_zc_d;
                if (xc <= 0.0) {
                    sw = 1.0f; wz = 1.0f; node_inter = (double)e_ml + ez;
                } else if (xc >= 1.0) {
                    sw = 0.0f; wz = 1.0f; node_inter = ez;
                } else {
                    const double xc2 = xc * xc, xc3 = xc2 * xc;
                    const double sw_d = 1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2;
                    const double omx = 1.0 - xc;
                    sw = (float)sw_d;
                    wz = (float)(1.0 + (double)e_ml * (-30.0 * xc2 * omx * omx) * inv_zc_d);
                    node_inter = sw_d * (double)e_ml + ez;
                }
                w_zbl_out[r] = wz;
            }
            s_sw[tid] = sw;
            e_atom = node_inter;
            if (atom_energy) atom_energy[r] += e_atom;
        } else {
            s_sw[tid] = 0.0f;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) e_atom += __shfl_xor_sync(0xffffffffu, e_atom, off);
        if ((tid & 31) == 0 && e_atom != 0.0) atomicAdd(energy, e_atom);
    }
    __syncthreads();

    if (RW == 64 || wn == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = er + 16 * i + 8 * h;
                const float sw = s_sw[lr];
                float am_t = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float v0 = acc[i][j][2 * h]     * sw;
                    const float v1 = acc[i][j][2 * h + 1] * sw;
                    acc[i][j][2 * h]     = v0;
                    acc[i][j][2 * h + 1] = v1;
                    am_t = fmaxf(am_t, fmaxf(fabsf(v0), fabsf(v1)));
                }
                am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
                am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
                if (t == 0) s_am[lr * 2 + wn] = am_t;
            }
    }
    __syncthreads();

    if (tid < 64) {
        const float am_r = (RW == 64) ? fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]) : s_am[tid * 2 + 0];
        const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
        s_pa[tid] = (erow > -127) ? h2r_pow2_f(erow - 12) : 0.0f;
        s_sA[tid] = (erow > -127) ? h2r_pow2_f(12 - erow) : 0.0f;
    }
    __syncthreads();

    if (RW == 64 || wn == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = er + 16 * i + 8 * h;
                const float sa = s_sA[lr];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c0 = ec + 8 * j;
                    const int kt = c0 >> 5;
                    const int kl = c0 & 31;
                    unsigned h_pair, l_pair;
                    split2f(make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]), sa, h_pair, l_pair);
                    const int off = swz(lr, kl >> 3) + ((kl & 7) << 1);
                    *reinterpret_cast<unsigned*>(sm + kt * 4096 + off)        = h_pair;
                    *reinterpret_cast<unsigned*>(sm + 8192 + kt * 4096 + off) = l_pair;
                }
            }
    }
    __syncthreads();

    // Phase 3: dx[64, K] = dpre0[64, RW] @ W0^T[RW, K] with double-buffered B_bwd in sm + 16384..32767!
    constexpr int NK_BWD = RW / 32;
    const int n_tiles = K / 64;
    const int total_steps = n_tiles * NK_BWD;
    const int64_t planeB_bwd = (int64_t)K * RW;
    const int a_row = wm * 32 + (lane & 15), a_kc = lane >> 4;

    auto issue_bwd = [&](int step) {
        const int st = step & 1;
        const int nt = (NK_BWD == 1) ? step : (step >> 1);
        const int kt = (NK_BWD == 1) ? 0    : (step & 1);
        const unsigned sBH = sbase + (unsigned)(16384 + st * 8192);
        const unsigned sBL = sBH + 4096u;
        const __half* bg = W0_BT + (int64_t)(nt * 64 + cr) * RW + kt * 32 + cj * 8;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            cp16(sBH + swz(cr + 32 * i, cj), bg + (int64_t)32 * i * RW, true);
            cp16(sBL + swz(cr + 32 * i, cj), bg + planeB_bwd + (int64_t)32 * i * RW, true);
        }
        asm volatile("cp.async.commit_group;");
    };

    float acc2[2][4][4] = {};
    unsigned accX2[2][4][2] = {};
    issue_bwd(0);
    for (int step = 0; step < total_steps; ++step) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (step + 1 < total_steps) issue_bwd(step + 1);
        const int st = step & 1;
        const int nt = (NK_BWD == 1) ? step : (step >> 1);
        const int kt = (NK_BWD == 1) ? 0    : (step & 1);
        const unsigned sAH_k = sbase + (unsigned)(kt * 4096);
        const unsigned sAL_k = sbase + (unsigned)(8192 + kt * 4096);
        const unsigned sBH   = sbase + (unsigned)(16384 + st * 8192);
        const unsigned sBL   = sBH + 4096u;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned ah[2][4], al[2][4], bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                ldsm4(sAH_k + swz(a_row + 16 * i, 2 * kk + a_kc), ah[i]);
                ldsm4(sAL_k + swz(a_row + 16 * i, 2 * kk + a_kc), al[i]);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                ldsm4(sBH + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                ldsm4(sBL + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int jp = j >> 1, jb = 2 * (j & 1);
                    float d[4] = {};
                    mma16816(d, ah[i], bh[jp][jb], bh[jp][jb + 1]);
                    jam4(d);
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc2[i][j][e] += d[e];
                    mma16816h(accX2[i][j], ah[i], bl[jp][jb], bl[jp][jb + 1]);
                    mma16816h(accX2[i][j], al[i], bc[jp][jb], bc[jp][jb + 1]);
                }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX2[i][j][h]));
                    acc2[i][j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), acc2[i][j][2 * h]);
                    acc2[i][j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), acc2[i][j][2 * h + 1]);
                }
                accX2[i][j][0] = accX2[i][j][1] = 0u;
            }
        if (kt == NK_BWD - 1) {
            const int n0 = nt * 64;
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int lr = er + 16 * i + 8 * h;
                    const int r = m0 + lr;
                    const float pa = s_pa[lr];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (r < M) {
                            const int c0 = n0 + ec + 8 * j;
                            const float2 pb = *reinterpret_cast<const float2*>(W0_inv_sBT + c0);
                            *reinterpret_cast<float2*>(dx + (int64_t)r * K + c0) =
                                make_float2((acc2[i][j][2 * h] * pa) * pb.x,
                                            (acc2[i][j][2 * h + 1] * pa) * pb.y);
                        }
                        acc2[i][j][2 * h] = acc2[i][j][2 * h + 1] = 0.0f;
                    }
                }
        }
    }
}

// =================================================================================================
// Zero-Idle-Warp Single-Kernel Full Fused ILMlip Readout for depth==1, width==32, need_grad==true:
//   4 warps x (16 atoms x 32 hidden) -> 100% warp utilization in ALL 3 phases, half W0_B loads,
//   24 KB SMEM -> 4 CTAs/SM!
// =================================================================================================
template <bool ZBL_SW>
__global__ void __launch_bounds__(128, 4) h2f_ilmlip_ro1_fused_w32_kernel(
    int M, int K, float inv_zc, const int* __restrict__ species,
    const float* __restrict__ x, const int8_t* __restrict__ eA_x,
    const float* __restrict__ alpha_Z, const float* __restrict__ u_Z,
    const __half* __restrict__ W0_B, const float* __restrict__ W0_inv_sB,
    const __half* __restrict__ W0_BT, const float* __restrict__ W0_inv_sBT,
    const double* __restrict__ e_zbl, double* __restrict__ energy, double* __restrict__ atom_energy,
    float* __restrict__ w_zbl_out, float* __restrict__ dx) {
    extern __shared__ __align__(128) unsigned char sm[];
    __shared__ int   s_sp[64];
    __shared__ float s_pa[64];
    __shared__ float s_sA[64];

    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int m0 = blockIdx.y * 64;
    const int g = lane >> 2, t = lane & 3;
    const int er = warp * 16 + g, ec = 2 * t;

    if (tid < 64) {
        const int r = m0 + tid;
        const bool ok = (r < M);
        s_sp[tid] = ok ? species[r] : 0;
        const int e = ok ? (int)eA_x[r] : -127;
        s_sA[tid] = (e > -127) ? h2r_pow2_f(12 - e) : 0.0f;
        s_pa[tid] = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
    }

    // Phase 1: pre0[64, 32] = x[64, K] @ W0[K, 32] (stage = 8192 B A + 4096 B B32 = 12288 B, 2x = 24576 B)
    constexpr int kStageBytes = 12288;
    const int64_t planeB_fwd = (int64_t)64 * K;
    const int fr = tid >> 3, fj = tid & 7;
    const int cr = tid >> 2, cj = tid & 3;
    const __half* bg_f = W0_B + (int64_t)cr * K + cj * 8;
    const int b_row = ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;

    auto stage_bf = [&](int st, int p) { return sbase + (unsigned)(st * kStageBytes + 8192 + p * 2048); };
    auto issue_fwd = [&](int kt) {
        const int st = kt & 1, k0 = kt * kHK;
        const unsigned sa = sbase + (unsigned)(st * kStageBytes);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = fr + 16 * i;
            const bool rok = (m0 + r) < M;
            cp16(sa + r * 128 + ((fj ^ (r & 7)) << 4), x + (int64_t)(rok ? m0 + r : m0) * K + k0 + 4 * fj, rok);
        }
        cp16(stage_bf(st, 0) + swz(cr, cj), bg_f + k0, true);
        cp16(stage_bf(st, 1) + swz(cr, cj), bg_f + planeB_fwd + k0, true);
        asm volatile("cp.async.commit_group;");
    };

    float acc[4][4] = {};
    {
        unsigned accX[4][2] = {};
        const int nk = K / kHK;
        issue_fwd(0);
        for (int kt = 0; kt < nk; ++kt) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (kt + 1 < nk) issue_fwd(kt + 1);
            const int st = kt & 1;
            const unsigned char* s = sm + st * kStageBytes + er * 128 + (t & 1) * 8;
            const float sc0 = s_sA[er], sc1 = s_sA[er + 8];
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(stage_bf(st, 0) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(stage_bf(st, 1) + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
                const int ja = 4 * kk + (t >> 1);
                const int o0 = (ja ^ g) << 4, o1 = ((ja + 2) ^ g) << 4;
                unsigned ah[4], al[4];
                split2f(*reinterpret_cast<const float2*>(s + o0), sc0, ah[0], al[0]);
                split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o0), sc1, ah[1], al[1]);
                split2f(*reinterpret_cast<const float2*>(s + o1), sc0, ah[2], al[2]);
                split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o1), sc1, ah[3], al[3]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int jp = j >> 1, jb = 2 * (j & 1);
                    float d[4] = {};
                    mma16816(d, ah, bh[jp][jb], bh[jp][jb + 1]);
                    jam4(d);
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[j][e] += d[e];
                    mma16816h(accX[j], ah, bl[jp][jb], bl[jp][jb + 1]);
                    mma16816h(accX[j], al, bc[jp][jb], bc[jp][jb + 1]);
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX[j][h]));
                    acc[j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), acc[j][2 * h]);
                    acc[j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), acc[j][2 * h + 1]);
                }
                accX[j][0] = accX[j][1] = 0u;
            }
        }
    }
    __syncthreads();

    // Phase 2: In-register bias+silu + dot u_Z + ZBL + silu_bwd -> split dpre0 into sAH[0..4095], sAL[4096..8191]!
    float* s_dot = reinterpret_cast<float*>(sm + 8192);
    float* s_am  = reinterpret_cast<float*>(sm + 8448);
    float* s_sw  = reinterpret_cast<float*>(sm + 8704);

#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int lr = er + 8 * h;
        const int r = m0 + lr;
        const float pa = s_pa[lr];
        const int sp = s_sp[lr];
        const float* azrow = alpha_Z + (int64_t)sp * 32;
        const float* urow  = u_Z + (int64_t)sp * 32;
        float dot_t = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c0 = ec + 8 * j;
            const float2 pb = *reinterpret_cast<const float2*>(W0_inv_sB + c0);
            const float2 az = *reinterpret_cast<const float2*>(azrow + c0);
            const float2 uv = *reinterpret_cast<const float2*>(urow + c0);
            const float p0 = fmaf(acc[j][2 * h]     * pa, pb.x, az.x);
            const float p1 = fmaf(acc[j][2 * h + 1] * pa, pb.y, az.y);
            const float s0 = 1.0f / (1.0f + __expf(-p0));
            const float s1 = 1.0f / (1.0f + __expf(-p1));
            dot_t = fmaf(p0 * s0, uv.x, fmaf(p1 * s1, uv.y, dot_t));
            acc[j][2 * h]     = uv.x * (s0 * (1.0f + p0 * (1.0f - s0)));
            acc[j][2 * h + 1] = uv.y * (s1 * (1.0f + p1 * (1.0f - s1)));
        }
        dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 2);
        dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 1);
        if (t == 0) s_dot[lr] = (r < M) ? dot_t : 0.0f;
    }
    __syncthreads();

    if (tid < 64) {
        const int r = m0 + tid;
        double e_atom = 0.0;
        if (r < M) {
            const float e_ml = s_dot[tid];
            float sw = 1.0f, wz = 1.0f;
            double node_inter = (double)e_ml;
            if constexpr (ZBL_SW) {
                const double ez = e_zbl[r];
                const double inv_zc_d = (double)inv_zc;
                const double xc = ez * inv_zc_d;
                if (xc <= 0.0) {
                    sw = 1.0f; wz = 1.0f; node_inter = (double)e_ml + ez;
                } else if (xc >= 1.0) {
                    sw = 0.0f; wz = 1.0f; node_inter = ez;
                } else {
                    const double xc2 = xc * xc, xc3 = xc2 * xc;
                    const double sw_d = 1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2;
                    const double omx = 1.0 - xc;
                    sw = (float)sw_d;
                    wz = (float)(1.0 + (double)e_ml * (-30.0 * xc2 * omx * omx) * inv_zc_d);
                    node_inter = sw_d * (double)e_ml + ez;
                }
                w_zbl_out[r] = wz;
            }
            s_sw[tid] = sw;
            e_atom = node_inter;
            if (atom_energy) atom_energy[r] += e_atom;
        } else {
            s_sw[tid] = 0.0f;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) e_atom += __shfl_xor_sync(0xffffffffu, e_atom, off);
        if ((tid & 31) == 0 && e_atom != 0.0) atomicAdd(energy, e_atom);
    }
    __syncthreads();

#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int lr = er + 8 * h;
        const float sw = s_sw[lr];
        float am_t = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float v0 = acc[j][2 * h]     * sw;
            const float v1 = acc[j][2 * h + 1] * sw;
            acc[j][2 * h]     = v0;
            acc[j][2 * h + 1] = v1;
            am_t = fmaxf(am_t, fmaxf(fabsf(v0), fabsf(v1)));
        }
        am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
        am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
        if (t == 0) s_am[lr] = am_t;
    }
    __syncthreads();

    if (tid < 64) {
        const float am_r = s_am[tid];
        const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
        s_pa[tid] = (erow > -127) ? h2r_pow2_f(erow - 12) : 0.0f;
        s_sA[tid] = (erow > -127) ? h2r_pow2_f(12 - erow) : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int lr = er + 8 * h;
        const float sa = s_sA[lr];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c0 = ec + 8 * j;
            unsigned h_pair, l_pair;
            split2f(make_float2(acc[j][2 * h], acc[j][2 * h + 1]), sa, h_pair, l_pair);
            const int off = swz(lr, c0 >> 3) + ((c0 & 7) << 1);
            *reinterpret_cast<unsigned*>(sm + off)        = h_pair;
            *reinterpret_cast<unsigned*>(sm + 4096 + off) = l_pair;
        }
    }
    __syncthreads();

    // Phase 3: dx[64, K] = dpre0[64, 32] @ W0^T[32, K] (double-buffered B_bwd in sm[8192..24575])
    const int n_tiles = K / 64;
    const int64_t planeB_bwd = (int64_t)K * 32;
    const int a_row = warp * 16 + (lane & 15), a_kc = lane >> 4;

    auto issue_bwd = [&](int nt) {
        const int st = nt & 1;
        const unsigned sBH = sbase + (unsigned)(8192 + st * 8192);
        const unsigned sBL = sBH + 4096u;
        const __half* bg = W0_BT + (int64_t)(nt * 64 + cr) * 32 + cj * 8;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            cp16(sBH + swz(cr + 32 * i, cj), bg + (int64_t)32 * i * 32, true);
            cp16(sBL + swz(cr + 32 * i, cj), bg + planeB_bwd + (int64_t)32 * i * 32, true);
        }
        asm volatile("cp.async.commit_group;");
    };

    issue_bwd(0);
    for (int nt = 0; nt < n_tiles; ++nt) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (nt + 1 < n_tiles) issue_bwd(nt + 1);
        const int st = nt & 1;
        const unsigned sBH = sbase + (unsigned)(8192 + st * 8192);
        const unsigned sBL = sBH + 4096u;
        float acc2[2][4][4] = {};
        unsigned accX2[2][4][2] = {};
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned ah[4], al[4];
            ldsm4(sbase + swz(a_row, 2 * kk + a_kc), ah);
            ldsm4(sbase + 4096u + swz(a_row, 2 * kk + a_kc), al);
#pragma unroll
            for (int cp = 0; cp < 2; ++cp) {
                unsigned bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(sBH + swz(cp * 32 + b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(sBL + swz(cp * 32 + b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int jp = j >> 1, jb = 2 * (j & 1);
                    float d[4] = {};
                    mma16816(d, ah, bh[jp][jb], bh[jp][jb + 1]);
                    jam4(d);
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc2[cp][j][e] += d[e];
                    mma16816h(accX2[cp][j], ah, bl[jp][jb], bl[jp][jb + 1]);
                    mma16816h(accX2[cp][j], al, bc[jp][jb], bc[jp][jb + 1]);
                }
            }
        }
        const int n0 = nt * 64;
#pragma unroll
        for (int cp = 0; cp < 2; ++cp)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = er + 8 * h;
                const int r = m0 + lr;
                const float pa = s_pa[lr];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (r < M) {
                        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX2[cp][j][h]));
                        const float v0 = fmaf(xv.x, (float)(1 << (kHC - 11)), acc2[cp][j][2 * h]);
                        const float v1 = fmaf(xv.y, (float)(1 << (kHC - 11)), acc2[cp][j][2 * h + 1]);
                        const int c0 = n0 + cp * 32 + ec + 8 * j;
                        const float2 pb = *reinterpret_cast<const float2*>(W0_inv_sBT + c0);
                        *reinterpret_cast<float2*>(dx + (int64_t)r * K + c0) =
                            make_float2((v0 * pa) * pb.x, (v1 * pa) * pb.y);
                    }
                }
            }
    }
}

// =================================================================================================
// Fused ILMlip Readout Forward + Tail for depth==1, width in {32, 64}, need_grad==true:
//   x[N,K] @ W0 -> +alpha_Z -> silu -> .u_Z (+ZBL) -> silu_bwd -> stores dpre0[N,RW] + eA_dpre0[N]
//   ZERO pre0/h0 DRAM traffic, 32 KB SMEM -> 3 CTAs/SM; followed by parallel 2D gemm_h2_row(W0^T)!
// =================================================================================================
template <int RW, bool ZBL_SW>
__global__ void __launch_bounds__(128, 3) h2f_ilmlip_ro1_fwd_tail_kernel(
    int M, int K, float inv_zc, const int* __restrict__ species,
    const float* __restrict__ x, const int8_t* __restrict__ eA_x,
    const float* __restrict__ alpha_Z, const float* __restrict__ u_Z,
    const __half* __restrict__ W0_B, const float* __restrict__ W0_inv_sB,
    const double* __restrict__ e_zbl, double* __restrict__ energy, double* __restrict__ atom_energy,
    float* __restrict__ w_zbl_out, float* __restrict__ dpre0, int8_t* __restrict__ eA_dpre0) {
    static_assert(RW == 32 || RW == 64, "RW must be 32 or 64");
    extern __shared__ __align__(128) unsigned char sm[];
    __shared__ int   s_sp[64];
    __shared__ float s_pa[64];
    __shared__ float s_sA[64];

    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * 64;
    const int g = lane >> 2, t = lane & 3;
    const int er = wm * 32 + g, ec = wn * 32 + 2 * t;

    if (tid < 64) {
        const int r = m0 + tid;
        const bool ok = (r < M);
        s_sp[tid] = ok ? species[r] : 0;
        const int e = ok ? (int)eA_x[r] : -127;
        s_sA[tid] = (e > -127) ? h2r_pow2_f(12 - e) : 0.0f;
        s_pa[tid] = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
    }

    constexpr int kTileA = 64 * kHK;
    constexpr int kStageH = 2 * kTileA + 2 * kHTileB;
    const int64_t planeB_fwd = (int64_t)64 * K;
    const int fr = tid >> 3, fj = tid & 7;
    const int cr = tid >> 2, cj = tid & 3;
    const __half* bg_f = W0_B + (int64_t)cr * K + cj * 8;
    const int b_row = wn * 32 + ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;

    auto stage_bf = [&](int st, int p) { return sbase + (unsigned)((st * kStageH + 2 * kTileA + p * kHTileB) * 2); };
    auto issue_fwd = [&](int kt) {
        const int st = kt & 1, k0 = kt * kHK;
        const unsigned sa = sbase + (unsigned)(st * kStageH * 2);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = fr + 16 * i;
            const bool rok = (m0 + r) < M;
            cp16(sa + r * 128 + ((fj ^ (r & 7)) << 4), x + (int64_t)(rok ? m0 + r : m0) * K + k0 + 4 * fj, rok);
        }
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                cp16(stage_bf(st, p) + swz(cr + 32 * i, cj), bg_f + p * planeB_fwd + (int64_t)32 * i * K + k0, true);
        asm volatile("cp.async.commit_group;");
    };

    float acc[2][4][4] = {};
    {
        unsigned accX[2][4][2] = {};
        const int nk = K / kHK;
        issue_fwd(0);
        for (int kt = 0; kt < nk; ++kt) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (kt + 1 < nk) issue_fwd(kt + 1);
            if constexpr (RW == 32) {
                if (wn == 1) continue;
            }
            const int st = kt & 1;
            const unsigned char* sa = sm + st * kStageH * 2 + er * 128 + (t & 1) * 8;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(stage_bf(st, 0) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(stage_bf(st, 1) + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
                const int ja = 4 * kk + (t >> 1);
                const int o0 = (ja ^ g) << 4, o1 = ((ja + 2) ^ g) << 4;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const unsigned char* s = sa + 16 * i * 128;
                    unsigned ah[4], al[4];
                    const float sc0 = s_sA[er + 16 * i], sc1 = s_sA[er + 16 * i + 8];
                    split2f(*reinterpret_cast<const float2*>(s + o0), sc0, ah[0], al[0]);
                    split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o0), sc1, ah[1], al[1]);
                    split2f(*reinterpret_cast<const float2*>(s + o1), sc0, ah[2], al[2]);
                    split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o1), sc1, ah[3], al[3]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int jp = j >> 1, jb = 2 * (j & 1);
                        float d[4] = {};
                        mma16816(d, ah, bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[i][j][e] += d[e];
                        mma16816h(accX[i][j], ah, bl[jp][jb], bl[jp][jb + 1]);
                        mma16816h(accX[i][j], al, bc[jp][jb], bc[jp][jb + 1]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        acc[i][j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), acc[i][j][2 * h]);
                        acc[i][j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), acc[i][j][2 * h + 1]);
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
        }
    }
    __syncthreads();

    float* s_dot = reinterpret_cast<float*>(sm + 16384);
    float* s_am  = reinterpret_cast<float*>(sm + 16384 + 512);
    float* s_sw  = reinterpret_cast<float*>(sm + 16384 + 1024);

    if (RW == 64 || wn == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = er + 16 * i + 8 * h;
                const int r = m0 + lr;
                const float pa = s_pa[lr];
                const int sp = s_sp[lr];
                const float* azrow = alpha_Z + (int64_t)sp * RW;
                const float* urow  = u_Z + (int64_t)sp * RW;
                float dot_t = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c0 = ec + 8 * j;
                    const float2 pb = *reinterpret_cast<const float2*>(W0_inv_sB + c0);
                    const float2 az = *reinterpret_cast<const float2*>(azrow + c0);
                    const float2 uv = *reinterpret_cast<const float2*>(urow + c0);
                    const float p0 = fmaf(acc[i][j][2 * h]     * pa, pb.x, az.x);
                    const float p1 = fmaf(acc[i][j][2 * h + 1] * pa, pb.y, az.y);
                    const float s0 = 1.0f / (1.0f + __expf(-p0));
                    const float s1 = 1.0f / (1.0f + __expf(-p1));
                    dot_t = fmaf(p0 * s0, uv.x, fmaf(p1 * s1, uv.y, dot_t));
                    acc[i][j][2 * h]     = uv.x * (s0 * (1.0f + p0 * (1.0f - s0)));
                    acc[i][j][2 * h + 1] = uv.y * (s1 * (1.0f + p1 * (1.0f - s1)));
                }
                dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 2);
                dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 1);
                if (t == 0) s_dot[lr * 2 + wn] = (r < M) ? dot_t : 0.0f;
            }
    }
    __syncthreads();

    if (tid < 64) {
        const int r = m0 + tid;
        double e_atom = 0.0;
        if (r < M) {
            const float e_ml = (RW == 64) ? (s_dot[tid * 2 + 0] + s_dot[tid * 2 + 1]) : s_dot[tid * 2 + 0];
            float sw = 1.0f, wz = 1.0f;
            double node_inter = (double)e_ml;
            if constexpr (ZBL_SW) {
                const double ez = e_zbl[r];
                const double inv_zc_d = (double)inv_zc;
                const double xc = ez * inv_zc_d;
                if (xc <= 0.0) {
                    sw = 1.0f; wz = 1.0f; node_inter = (double)e_ml + ez;
                } else if (xc >= 1.0) {
                    sw = 0.0f; wz = 1.0f; node_inter = ez;
                } else {
                    const double xc2 = xc * xc, xc3 = xc2 * xc;
                    const double sw_d = 1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2;
                    const double omx = 1.0 - xc;
                    sw = (float)sw_d;
                    wz = (float)(1.0 + (double)e_ml * (-30.0 * xc2 * omx * omx) * inv_zc_d);
                    node_inter = sw_d * (double)e_ml + ez;
                }
                w_zbl_out[r] = wz;
            }
            s_sw[tid] = sw;
            e_atom = node_inter;
            if (atom_energy) atom_energy[r] += e_atom;
        } else {
            s_sw[tid] = 0.0f;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) e_atom += __shfl_xor_sync(0xffffffffu, e_atom, off);
        if ((tid & 31) == 0 && e_atom != 0.0) atomicAdd(energy, e_atom);
    }
    __syncthreads();

    if (RW == 64 || wn == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = er + 16 * i + 8 * h;
                const int r = m0 + lr;
                const float sw = s_sw[lr];
                float am_t = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float v0 = acc[i][j][2 * h]     * sw;
                    const float v1 = acc[i][j][2 * h + 1] * sw;
                    if (r < M) {
                        const int c0 = ec + 8 * j;
                        *reinterpret_cast<float2*>(dpre0 + (int64_t)r * RW + c0) = make_float2(v0, v1);
                    }
                    am_t = fmaxf(am_t, fmaxf(fabsf(v0), fabsf(v1)));
                }
                am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
                am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
                if (t == 0) s_am[lr * 2 + wn] = am_t;
            }
    }
    __syncthreads();

    if (eA_dpre0 && tid < 64) {
        const int r = m0 + tid;
        if (r < M) {
            const float am_r = (RW == 64) ? fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]) : s_am[tid * 2 + 0];
            const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
            eA_dpre0[r] = (int8_t)erow;
        }
    }
}

// =================================================================================================
// Single-Kernel Full Fused ILMlip Readout for depth==2, width==64, need_grad==true:
//   x[N,K] @ W0 -> +alpha_Z -> silu -> @W1 -> +b1 -> silu -> .u_Z (+ZBL) -> silu_bwd -> @W1^T -> silu_bwd -> @W0^T -> dx
//   ZERO pre0/h0/pre1/dpre1/dpre0 DRAM traffic! 49,152 B SMEM (s_p0 lives in sm+32768..49151) -> 2 CTAs/SM!
// =================================================================================================
template <bool ZBL_SW>
__global__ void __launch_bounds__(128, 2) h2f_ilmlip_ro2_fused_w64_kernel(
    int M, int K, float inv_zc, const int* __restrict__ species,
    const float* __restrict__ x, const int8_t* __restrict__ eA_x,
    const float* __restrict__ alpha_Z, const float* __restrict__ b1, const float* __restrict__ u_Z,
    const __half* __restrict__ W0_B, const float* __restrict__ W0_inv_sB,
    const __half* __restrict__ W0_BT, const float* __restrict__ W0_inv_sBT,
    const __half* __restrict__ W1_B, const float* __restrict__ W1_inv_sB,
    const __half* __restrict__ W1_BT, const float* __restrict__ W1_inv_sBT,
    const double* __restrict__ e_zbl, double* __restrict__ energy, double* __restrict__ atom_energy,
    float* __restrict__ w_zbl_out, float* __restrict__ dx) {
    extern __shared__ __align__(128) unsigned char sm[];
    __shared__ int   s_sp[64];
    __shared__ float s_pa[64];
    __shared__ float s_sA[64];

    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * 64;
    const int g = lane >> 2, t = lane & 3;
    const int er = wm * 32 + g, ec = wn * 32 + 2 * t;
    const int fr = tid >> 3, fj = tid & 7;
    const int cr = tid >> 2, cj = tid & 3;
    const int a_row = wm * 32 + (lane & 15), a_kc = lane >> 4;
    const int b_row = wn * 32 + ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;

    if (tid < 64) {
        const int r = m0 + tid;
        const bool ok = (r < M);
        s_sp[tid] = ok ? species[r] : 0;
        const int e = ok ? (int)eA_x[r] : -127;
        s_sA[tid] = (e > -127) ? h2r_pow2_f(12 - e) : 0.0f;
        s_pa[tid] = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
    }

    // Step 1: pre0[64, 64] = x[64, K] @ W0[K, 64] (uses sm[0..32767])
    constexpr int kTileA = 64 * kHK;
    constexpr int kStageH = 2 * kTileA + 2 * kHTileB;
    const int64_t planeB_w0 = (int64_t)64 * K;
    const __half* bg_w0 = W0_B + (int64_t)cr * K + cj * 8;
    auto stage_bf = [&](int st, int p) { return sbase + (unsigned)((st * kStageH + 2 * kTileA + p * kHTileB) * 2); };
    auto issue_w0 = [&](int kt) {
        const int st = kt & 1, k0 = kt * kHK;
        const unsigned sa = sbase + (unsigned)(st * kStageH * 2);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = fr + 16 * i;
            const bool rok = (m0 + r) < M;
            cp16(sa + r * 128 + ((fj ^ (r & 7)) << 4), x + (int64_t)(rok ? m0 + r : m0) * K + k0 + 4 * fj, rok);
        }
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                cp16(stage_bf(st, p) + swz(cr + 32 * i, cj), bg_w0 + p * planeB_w0 + (int64_t)32 * i * K + k0, true);
        asm volatile("cp.async.commit_group;");
    };

    float acc0[2][4][4] = {};
    {
        unsigned accX[2][4][2] = {};
        const int nk = K / kHK;
        issue_w0(0);
        for (int kt = 0; kt < nk; ++kt) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (kt + 1 < nk) issue_w0(kt + 1);
            const int st = kt & 1;
            const unsigned char* sa = sm + st * kStageH * 2 + er * 128 + (t & 1) * 8;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(stage_bf(st, 0) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(stage_bf(st, 1) + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
                const int ja = 4 * kk + (t >> 1);
                const int o0 = (ja ^ g) << 4, o1 = ((ja + 2) ^ g) << 4;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const unsigned char* s = sa + 16 * i * 128;
                    unsigned ah[4], al[4];
                    const float sc0 = s_sA[er + 16 * i], sc1 = s_sA[er + 16 * i + 8];
                    split2f(*reinterpret_cast<const float2*>(s + o0), sc0, ah[0], al[0]);
                    split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o0), sc1, ah[1], al[1]);
                    split2f(*reinterpret_cast<const float2*>(s + o1), sc0, ah[2], al[2]);
                    split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o1), sc1, ah[3], al[3]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int jp = j >> 1, jb = 2 * (j & 1);
                        float d[4] = {};
                        mma16816(d, ah, bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc0[i][j][e] += d[e];
                        mma16816h(accX[i][j], ah, bl[jp][jb], bl[jp][jb + 1]);
                        mma16816h(accX[i][j], al, bc[jp][jb], bc[jp][jb + 1]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        acc0[i][j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), acc0[i][j][2 * h]);
                        acc0[i][j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), acc0[i][j][2 * h + 1]);
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
        }
    }
    __syncthreads();

    // Save p0 = pre0 + alpha_Z in sm[32768..49151] (16 KB) & split h0 = silu(p0) into sAH/sAL (sm[0..16383])
    float* s_p0  = reinterpret_cast<float*>(sm + 32768);
    float* s_dot = reinterpret_cast<float*>(sm + 16384);
    float* s_am  = reinterpret_cast<float*>(sm + 16384 + 512);
    float* s_sw  = reinterpret_cast<float*>(sm + 16384 + 1024);

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            const float pa = s_pa[lr];
            const int sp = s_sp[lr];
            const float* azrow = alpha_Z + (int64_t)sp * 64;
            float am_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ec + 8 * j;
                if (r < M) {
                    const float2 pb = *reinterpret_cast<const float2*>(W0_inv_sB + c0);
                    const float2 az = *reinterpret_cast<const float2*>(azrow + c0);
                    const float p0 = fmaf(acc0[i][j][2 * h]     * pa, pb.x, az.x);
                    const float p1 = fmaf(acc0[i][j][2 * h + 1] * pa, pb.y, az.y);
                    *reinterpret_cast<float2*>(s_p0 + lr * 64 + c0) = make_float2(p0, p1);
                    const float h0x = p0 / (1.0f + __expf(-p0));
                    const float h0y = p1 / (1.0f + __expf(-p1));
                    acc0[i][j][2 * h]     = h0x;
                    acc0[i][j][2 * h + 1] = h0y;
                    am_t = fmaxf(am_t, fmaxf(fabsf(h0x), fabsf(h0y)));
                } else {
                    acc0[i][j][2 * h] = acc0[i][j][2 * h + 1] = 0.0f;
                }
            }
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
            if (t == 0) s_am[lr * 2 + wn] = am_t;
        }
    __syncthreads();

    if (tid < 64) {
        const float am_r = fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]);
        const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
        s_pa[tid] = (erow > -127) ? h2r_pow2_f(erow - 12) : 0.0f;
        s_sA[tid] = (erow > -127) ? h2r_pow2_f(12 - erow) : 0.0f;
    }
    __syncthreads();

    auto split_acc_to_sA = [&](float a[2][4][4]) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = er + 16 * i + 8 * h;
                const float sa = s_sA[lr];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c0 = ec + 8 * j;
                    const int kt = c0 >> 5;
                    const int kl = c0 & 31;
                    unsigned h_pair, l_pair;
                    split2f(make_float2(a[i][j][2 * h], a[i][j][2 * h + 1]), sa, h_pair, l_pair);
                    const int off = swz(lr, kl >> 3) + ((kl & 7) << 1);
                    *reinterpret_cast<unsigned*>(sm + kt * 4096 + off)        = h_pair;
                    *reinterpret_cast<unsigned*>(sm + 8192 + kt * 4096 + off) = l_pair;
                }
            }
        __syncthreads();
    };
    split_acc_to_sA(acc0);

    // Helper: 64x64 @ 64x64 two-half MMA using double-buffered B in sm[16384..32767] (s_p0 intact!)
    auto run_w64 = [&](const __half* Bmat, float out_acc[2][4][4]) {
        constexpr int64_t planeB64 = 64 * 64;
        const __half* bg = Bmat + (int64_t)cr * 64 + cj * 8;
        unsigned accX[2][4][2] = {};
        auto iss_k = [&](int kt) {
            const unsigned sBH = sbase + (unsigned)(16384 + kt * 8192);
            const unsigned sBL = sBH + 4096u;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                cp16(sBH + swz(cr + 32 * i, cj), bg + (int64_t)32 * i * 64 + kt * 32, true);
                cp16(sBL + swz(cr + 32 * i, cj), bg + planeB64 + (int64_t)32 * i * 64 + kt * 32, true);
            }
            asm volatile("cp.async.commit_group;");
        };
        iss_k(0);
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (kt == 0) iss_k(1);
            const unsigned sAH_k = sbase + (unsigned)(kt * 4096);
            const unsigned sAL_k = sbase + (unsigned)(8192 + kt * 4096);
            const unsigned sBH   = sbase + (unsigned)(16384 + kt * 8192);
            const unsigned sBL   = sBH + 4096u;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned ah[2][4], al[2][4], bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    ldsm4(sAH_k + swz(a_row + 16 * i, 2 * kk + a_kc), ah[i]);
                    ldsm4(sAL_k + swz(a_row + 16 * i, 2 * kk + a_kc), al[i]);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(sBH + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(sBL + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int jp = j >> 1, jb = 2 * (j & 1);
                        float d[4] = {};
                        mma16816(d, ah[i], bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) out_acc[i][j][e] += d[e];
                        mma16816h(accX[i][j], ah[i], bl[jp][jb], bl[jp][jb + 1]);
                        mma16816h(accX[i][j], al[i], bc[jp][jb], bc[jp][jb + 1]);
                    }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        out_acc[i][j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), out_acc[i][j][2 * h]);
                        out_acc[i][j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), out_acc[i][j][2 * h + 1]);
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
        }
        __syncthreads();
    };

    // Step 2: pre1 = h0 @ W1
    float acc1[2][4][4] = {};
    run_w64(W1_B, acc1);

    // Step 3: bias_silu(pre1 + b1) + dot u_Z + ZBL + silu_bwd -> split dpre1 into sAH/sAL
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            const float pa = s_pa[lr];
            const int sp = s_sp[lr];
            const float* urow = u_Z + (int64_t)sp * 64;
            float dot_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ec + 8 * j;
                const float2 pb = *reinterpret_cast<const float2*>(W1_inv_sB + c0);
                const float2 bv = *reinterpret_cast<const float2*>(b1 + c0);
                const float2 uv = *reinterpret_cast<const float2*>(urow + c0);
                const float p0 = fmaf(acc1[i][j][2 * h]     * pa, pb.x, bv.x);
                const float p1 = fmaf(acc1[i][j][2 * h + 1] * pa, pb.y, bv.y);
                const float s0 = 1.0f / (1.0f + __expf(-p0));
                const float s1 = 1.0f / (1.0f + __expf(-p1));
                dot_t = fmaf(p0 * s0, uv.x, fmaf(p1 * s1, uv.y, dot_t));
                acc1[i][j][2 * h]     = uv.x * (s0 * (1.0f + p0 * (1.0f - s0)));
                acc1[i][j][2 * h + 1] = uv.y * (s1 * (1.0f + p1 * (1.0f - s1)));
            }
            dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 2);
            dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 1);
            if (t == 0) s_dot[lr * 2 + wn] = (r < M) ? dot_t : 0.0f;
        }
    __syncthreads();

    if (tid < 64) {
        const int r = m0 + tid;
        double e_atom = 0.0;
        if (r < M) {
            const float e_ml = s_dot[tid * 2 + 0] + s_dot[tid * 2 + 1];
            float sw = 1.0f, wz = 1.0f;
            double node_inter = (double)e_ml;
            if constexpr (ZBL_SW) {
                const double ez = e_zbl[r];
                const double inv_zc_d = (double)inv_zc;
                const double xc = ez * inv_zc_d;
                if (xc <= 0.0) {
                    sw = 1.0f; wz = 1.0f; node_inter = (double)e_ml + ez;
                } else if (xc >= 1.0) {
                    sw = 0.0f; wz = 1.0f; node_inter = ez;
                } else {
                    const double xc2 = xc * xc, xc3 = xc2 * xc;
                    const double sw_d = 1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2;
                    const double omx = 1.0 - xc;
                    sw = (float)sw_d;
                    wz = (float)(1.0 + (double)e_ml * (-30.0 * xc2 * omx * omx) * inv_zc_d);
                    node_inter = sw_d * (double)e_ml + ez;
                }
                w_zbl_out[r] = wz;
            }
            s_sw[tid] = sw;
            e_atom = node_inter;
            if (atom_energy) atom_energy[r] += e_atom;
        } else {
            s_sw[tid] = 0.0f;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) e_atom += __shfl_xor_sync(0xffffffffu, e_atom, off);
        if ((tid & 31) == 0 && e_atom != 0.0) atomicAdd(energy, e_atom);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const float sw = s_sw[lr];
            float am_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float v0 = acc1[i][j][2 * h]     * sw;
                const float v1 = acc1[i][j][2 * h + 1] * sw;
                acc1[i][j][2 * h]     = v0;
                acc1[i][j][2 * h + 1] = v1;
                am_t = fmaxf(am_t, fmaxf(fabsf(v0), fabsf(v1)));
            }
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
            if (t == 0) s_am[lr * 2 + wn] = am_t;
        }
    __syncthreads();

    if (tid < 64) {
        const float am_r = fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]);
        const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
        s_pa[tid] = (erow > -127) ? h2r_pow2_f(erow - 12) : 0.0f;
        s_sA[tid] = (erow > -127) ? h2r_pow2_f(12 - erow) : 0.0f;
    }
    __syncthreads();
    split_acc_to_sA(acc1);

    // Step 4: dh0 = dpre1 @ W1^T
    float acc2[2][4][4] = {};
    run_w64(W1_BT, acc2);

    // Step 5: dpre0 = dh0 * dsilu(s_p0) -> split dpre0 into sAH/sAL
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            const float pa = s_pa[lr];
            float am_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (r < M) {
                    const int c0 = ec + 8 * j;
                    const float2 pb = *reinterpret_cast<const float2*>(W1_inv_sBT + c0);
                    const float2 p0v = *reinterpret_cast<const float2*>(s_p0 + lr * 64 + c0);
                    const float sx = 1.0f / (1.0f + __expf(-p0v.x));
                    const float sy = 1.0f / (1.0f + __expf(-p0v.y));
                    const float ox = ((acc2[i][j][2 * h]     * pa) * pb.x) * (sx * (1.0f + p0v.x * (1.0f - sx)));
                    const float oy = ((acc2[i][j][2 * h + 1] * pa) * pb.y) * (sy * (1.0f + p0v.y * (1.0f - sy)));
                    acc2[i][j][2 * h]     = ox;
                    acc2[i][j][2 * h + 1] = oy;
                    am_t = fmaxf(am_t, fmaxf(fabsf(ox), fabsf(oy)));
                } else {
                    acc2[i][j][2 * h] = acc2[i][j][2 * h + 1] = 0.0f;
                }
            }
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
            if (t == 0) s_am[lr * 2 + wn] = am_t;
        }
    __syncthreads();

    if (tid < 64) {
        const float am_r = fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]);
        const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
        s_pa[tid] = (erow > -127) ? h2r_pow2_f(erow - 12) : 0.0f;
        s_sA[tid] = (erow > -127) ? h2r_pow2_f(12 - erow) : 0.0f;
    }
    __syncthreads();
    split_acc_to_sA(acc2);

    // Step 6: dx[64, K] = dpre0[64, 64] @ W0^T[64, K] (double-buffered B_bwd in sm[16384..32767])
    const int n_tiles = K / 64;
    const int total_steps = n_tiles * 2;
    const int64_t planeB_bwd = (int64_t)K * 64;
    auto issue_bwd = [&](int step) {
        const int st = step & 1;
        const int nt = step >> 1;
        const int kt = step & 1;
        const unsigned sBH = sbase + (unsigned)(16384 + st * 8192);
        const unsigned sBL = sBH + 4096u;
        const __half* bg = W0_BT + (int64_t)(nt * 64 + cr) * 64 + kt * 32 + cj * 8;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            cp16(sBH + swz(cr + 32 * i, cj), bg + (int64_t)32 * i * 64, true);
            cp16(sBL + swz(cr + 32 * i, cj), bg + planeB_bwd + (int64_t)32 * i * 64, true);
        }
        asm volatile("cp.async.commit_group;");
    };

    float acc3[2][4][4] = {};
    unsigned accX3[2][4][2] = {};
    issue_bwd(0);
    for (int step = 0; step < total_steps; ++step) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (step + 1 < total_steps) issue_bwd(step + 1);
        const int st = step & 1;
        const int nt = step >> 1;
        const int kt = step & 1;
        const unsigned sAH_k = sbase + (unsigned)(kt * 4096);
        const unsigned sAL_k = sbase + (unsigned)(8192 + kt * 4096);
        const unsigned sBH   = sbase + (unsigned)(16384 + st * 8192);
        const unsigned sBL   = sBH + 4096u;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned ah[2][4], al[2][4], bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                ldsm4(sAH_k + swz(a_row + 16 * i, 2 * kk + a_kc), ah[i]);
                ldsm4(sAL_k + swz(a_row + 16 * i, 2 * kk + a_kc), al[i]);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                ldsm4(sBH + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                ldsm4(sBL + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int jp = j >> 1, jb = 2 * (j & 1);
                    float d[4] = {};
                    mma16816(d, ah[i], bh[jp][jb], bh[jp][jb + 1]);
                    jam4(d);
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc3[i][j][e] += d[e];
                    mma16816h(accX3[i][j], ah[i], bl[jp][jb], bl[jp][jb + 1]);
                    mma16816h(accX3[i][j], al[i], bc[jp][jb], bc[jp][jb + 1]);
                }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX3[i][j][h]));
                    acc3[i][j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), acc3[i][j][2 * h]);
                    acc3[i][j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), acc3[i][j][2 * h + 1]);
                }
                accX3[i][j][0] = accX3[i][j][1] = 0u;
            }
        if (kt == 1) {
            const int n0 = nt * 64;
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int lr = er + 16 * i + 8 * h;
                    const int r = m0 + lr;
                    const float pa = s_pa[lr];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (r < M) {
                            const int c0 = n0 + ec + 8 * j;
                            const float2 pb = *reinterpret_cast<const float2*>(W0_inv_sBT + c0);
                            *reinterpret_cast<float2*>(dx + (int64_t)r * K + c0) =
                                make_float2((acc3[i][j][2 * h] * pa) * pb.x,
                                            (acc3[i][j][2 * h + 1] * pa) * pb.y);
                        }
                        acc3[i][j][2 * h] = acc3[i][j][2 * h + 1] = 0.0f;
                    }
                }
        }
    }
}

// =================================================================================================
// Fused ILMlip Readout Forward + Tail for depth==2, width==64, need_grad==true (for N > 4608):
//   x[N,K] @ W0 -> +alpha_Z -> silu -> @W1 -> +b1 -> silu -> .u_Z (+ZBL) -> silu_bwd -> @W1^T ->
//   silu_bwd -> stores dpre0[N,64] + eA_dpre0[N]; followed by parallel 2D gemm_h2_row(W0^T)!
// =================================================================================================
template <bool ZBL_SW>
__global__ void __launch_bounds__(128, 2) h2f_ilmlip_ro2_fwd_tail_w64_kernel(
    int M, int K, float inv_zc, const int* __restrict__ species,
    const float* __restrict__ x, const int8_t* __restrict__ eA_x,
    const float* __restrict__ alpha_Z, const float* __restrict__ b1, const float* __restrict__ u_Z,
    const __half* __restrict__ W0_B, const float* __restrict__ W0_inv_sB,
    const __half* __restrict__ W1_B, const float* __restrict__ W1_inv_sB,
    const __half* __restrict__ W1_BT, const float* __restrict__ W1_inv_sBT,
    const double* __restrict__ e_zbl, double* __restrict__ energy, double* __restrict__ atom_energy,
    float* __restrict__ w_zbl_out, float* __restrict__ dpre0, int8_t* __restrict__ eA_dpre0) {
    extern __shared__ __align__(128) unsigned char sm[];
    __shared__ int   s_sp[64];
    __shared__ float s_pa[64];
    __shared__ float s_sA[64];

    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * 64;
    const int g = lane >> 2, t = lane & 3;
    const int er = wm * 32 + g, ec = wn * 32 + 2 * t;
    const int fr = tid >> 3, fj = tid & 7;
    const int cr = tid >> 2, cj = tid & 3;
    const int a_row = wm * 32 + (lane & 15), a_kc = lane >> 4;
    const int b_row = wn * 32 + ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;

    if (tid < 64) {
        const int r = m0 + tid;
        const bool ok = (r < M);
        s_sp[tid] = ok ? species[r] : 0;
        const int e = ok ? (int)eA_x[r] : -127;
        s_sA[tid] = (e > -127) ? h2r_pow2_f(12 - e) : 0.0f;
        s_pa[tid] = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
    }

    constexpr int kTileA = 64 * kHK;
    constexpr int kStageH = 2 * kTileA + 2 * kHTileB;
    const int64_t planeB_w0 = (int64_t)64 * K;
    const __half* bg_w0 = W0_B + (int64_t)cr * K + cj * 8;
    auto stage_bf = [&](int st, int p) { return sbase + (unsigned)((st * kStageH + 2 * kTileA + p * kHTileB) * 2); };
    auto issue_w0 = [&](int kt) {
        const int st = kt & 1, k0 = kt * kHK;
        const unsigned sa = sbase + (unsigned)(st * kStageH * 2);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = fr + 16 * i;
            const bool rok = (m0 + r) < M;
            cp16(sa + r * 128 + ((fj ^ (r & 7)) << 4), x + (int64_t)(rok ? m0 + r : m0) * K + k0 + 4 * fj, rok);
        }
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                cp16(stage_bf(st, p) + swz(cr + 32 * i, cj), bg_w0 + p * planeB_w0 + (int64_t)32 * i * K + k0, true);
        asm volatile("cp.async.commit_group;");
    };

    float acc0[2][4][4] = {};
    {
        unsigned accX[2][4][2] = {};
        const int nk = K / kHK;
        issue_w0(0);
        for (int kt = 0; kt < nk; ++kt) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (kt + 1 < nk) issue_w0(kt + 1);
            const int st = kt & 1;
            const unsigned char* sa = sm + st * kStageH * 2 + er * 128 + (t & 1) * 8;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(stage_bf(st, 0) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(stage_bf(st, 1) + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
                const int ja = 4 * kk + (t >> 1);
                const int o0 = (ja ^ g) << 4, o1 = ((ja + 2) ^ g) << 4;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const unsigned char* s = sa + 16 * i * 128;
                    unsigned ah[4], al[4];
                    const float sc0 = s_sA[er + 16 * i], sc1 = s_sA[er + 16 * i + 8];
                    split2f(*reinterpret_cast<const float2*>(s + o0), sc0, ah[0], al[0]);
                    split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o0), sc1, ah[1], al[1]);
                    split2f(*reinterpret_cast<const float2*>(s + o1), sc0, ah[2], al[2]);
                    split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o1), sc1, ah[3], al[3]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int jp = j >> 1, jb = 2 * (j & 1);
                        float d[4] = {};
                        mma16816(d, ah, bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc0[i][j][e] += d[e];
                        mma16816h(accX[i][j], ah, bl[jp][jb], bl[jp][jb + 1]);
                        mma16816h(accX[i][j], al, bc[jp][jb], bc[jp][jb + 1]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        acc0[i][j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), acc0[i][j][2 * h]);
                        acc0[i][j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), acc0[i][j][2 * h + 1]);
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
        }
    }
    __syncthreads();

    float* s_p0  = reinterpret_cast<float*>(sm + 32768);
    float* s_dot = reinterpret_cast<float*>(sm + 16384);
    float* s_am  = reinterpret_cast<float*>(sm + 16384 + 512);
    float* s_sw  = reinterpret_cast<float*>(sm + 16384 + 1024);

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            const float pa = s_pa[lr];
            const int sp = s_sp[lr];
            const float* azrow = alpha_Z + (int64_t)sp * 64;
            float am_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ec + 8 * j;
                if (r < M) {
                    const float2 pb = *reinterpret_cast<const float2*>(W0_inv_sB + c0);
                    const float2 az = *reinterpret_cast<const float2*>(azrow + c0);
                    const float p0 = fmaf(acc0[i][j][2 * h]     * pa, pb.x, az.x);
                    const float p1 = fmaf(acc0[i][j][2 * h + 1] * pa, pb.y, az.y);
                    *reinterpret_cast<float2*>(s_p0 + lr * 64 + c0) = make_float2(p0, p1);
                    const float h0x = p0 / (1.0f + __expf(-p0));
                    const float h0y = p1 / (1.0f + __expf(-p1));
                    acc0[i][j][2 * h]     = h0x;
                    acc0[i][j][2 * h + 1] = h0y;
                    am_t = fmaxf(am_t, fmaxf(fabsf(h0x), fabsf(h0y)));
                } else {
                    acc0[i][j][2 * h] = acc0[i][j][2 * h + 1] = 0.0f;
                }
            }
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
            if (t == 0) s_am[lr * 2 + wn] = am_t;
        }
    __syncthreads();

    if (tid < 64) {
        const float am_r = fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]);
        const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
        s_pa[tid] = (erow > -127) ? h2r_pow2_f(erow - 12) : 0.0f;
        s_sA[tid] = (erow > -127) ? h2r_pow2_f(12 - erow) : 0.0f;
    }
    __syncthreads();

    auto split_acc_to_sA = [&](float a[2][4][4]) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = er + 16 * i + 8 * h;
                const float sa = s_sA[lr];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c0 = ec + 8 * j;
                    const int kt = c0 >> 5;
                    const int kl = c0 & 31;
                    unsigned h_pair, l_pair;
                    split2f(make_float2(a[i][j][2 * h], a[i][j][2 * h + 1]), sa, h_pair, l_pair);
                    const int off = swz(lr, kl >> 3) + ((kl & 7) << 1);
                    *reinterpret_cast<unsigned*>(sm + kt * 4096 + off)        = h_pair;
                    *reinterpret_cast<unsigned*>(sm + 8192 + kt * 4096 + off) = l_pair;
                }
            }
        __syncthreads();
    };
    split_acc_to_sA(acc0);

    auto run_w64 = [&](const __half* Bmat, float out_acc[2][4][4]) {
        constexpr int64_t planeB64 = 64 * 64;
        const __half* bg = Bmat + (int64_t)cr * 64 + cj * 8;
        unsigned accX[2][4][2] = {};
        auto iss_k = [&](int kt) {
            const unsigned sBH = sbase + (unsigned)(16384 + kt * 8192);
            const unsigned sBL = sBH + 4096u;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                cp16(sBH + swz(cr + 32 * i, cj), bg + (int64_t)32 * i * 64 + kt * 32, true);
                cp16(sBL + swz(cr + 32 * i, cj), bg + planeB64 + (int64_t)32 * i * 64 + kt * 32, true);
            }
            asm volatile("cp.async.commit_group;");
        };
        iss_k(0);
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (kt == 0) iss_k(1);
            const unsigned sAH_k = sbase + (unsigned)(kt * 4096);
            const unsigned sAL_k = sbase + (unsigned)(8192 + kt * 4096);
            const unsigned sBH   = sbase + (unsigned)(16384 + kt * 8192);
            const unsigned sBL   = sBH + 4096u;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned ah[2][4], al[2][4], bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    ldsm4(sAH_k + swz(a_row + 16 * i, 2 * kk + a_kc), ah[i]);
                    ldsm4(sAL_k + swz(a_row + 16 * i, 2 * kk + a_kc), al[i]);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(sBH + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(sBL + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int jp = j >> 1, jb = 2 * (j & 1);
                        float d[4] = {};
                        mma16816(d, ah[i], bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) out_acc[i][j][e] += d[e];
                        mma16816h(accX[i][j], ah[i], bl[jp][jb], bl[jp][jb + 1]);
                        mma16816h(accX[i][j], al[i], bc[jp][jb], bc[jp][jb + 1]);
                    }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        out_acc[i][j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), out_acc[i][j][2 * h]);
                        out_acc[i][j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), out_acc[i][j][2 * h + 1]);
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
        }
        __syncthreads();
    };

    float acc1[2][4][4] = {};
    run_w64(W1_B, acc1);

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            const float pa = s_pa[lr];
            const int sp = s_sp[lr];
            const float* urow = u_Z + (int64_t)sp * 64;
            float dot_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ec + 8 * j;
                const float2 pb = *reinterpret_cast<const float2*>(W1_inv_sB + c0);
                const float2 bv = *reinterpret_cast<const float2*>(b1 + c0);
                const float2 uv = *reinterpret_cast<const float2*>(urow + c0);
                const float p0 = fmaf(acc1[i][j][2 * h]     * pa, pb.x, bv.x);
                const float p1 = fmaf(acc1[i][j][2 * h + 1] * pa, pb.y, bv.y);
                const float s0 = 1.0f / (1.0f + __expf(-p0));
                const float s1 = 1.0f / (1.0f + __expf(-p1));
                dot_t = fmaf(p0 * s0, uv.x, fmaf(p1 * s1, uv.y, dot_t));
                acc1[i][j][2 * h]     = uv.x * (s0 * (1.0f + p0 * (1.0f - s0)));
                acc1[i][j][2 * h + 1] = uv.y * (s1 * (1.0f + p1 * (1.0f - s1)));
            }
            dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 2);
            dot_t += __shfl_xor_sync(0xffffffffu, dot_t, 1);
            if (t == 0) s_dot[lr * 2 + wn] = (r < M) ? dot_t : 0.0f;
        }
    __syncthreads();

    if (tid < 64) {
        const int r = m0 + tid;
        double e_atom = 0.0;
        if (r < M) {
            const float e_ml = s_dot[tid * 2 + 0] + s_dot[tid * 2 + 1];
            float sw = 1.0f, wz = 1.0f;
            double node_inter = (double)e_ml;
            if constexpr (ZBL_SW) {
                const double ez = e_zbl[r];
                const double inv_zc_d = (double)inv_zc;
                const double xc = ez * inv_zc_d;
                if (xc <= 0.0) {
                    sw = 1.0f; wz = 1.0f; node_inter = (double)e_ml + ez;
                } else if (xc >= 1.0) {
                    sw = 0.0f; wz = 1.0f; node_inter = ez;
                } else {
                    const double xc2 = xc * xc, xc3 = xc2 * xc;
                    const double sw_d = 1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2;
                    const double omx = 1.0 - xc;
                    sw = (float)sw_d;
                    wz = (float)(1.0 + (double)e_ml * (-30.0 * xc2 * omx * omx) * inv_zc_d);
                    node_inter = sw_d * (double)e_ml + ez;
                }
                w_zbl_out[r] = wz;
            }
            s_sw[tid] = sw;
            e_atom = node_inter;
            if (atom_energy) atom_energy[r] += e_atom;
        } else {
            s_sw[tid] = 0.0f;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) e_atom += __shfl_xor_sync(0xffffffffu, e_atom, off);
        if ((tid & 31) == 0 && e_atom != 0.0) atomicAdd(energy, e_atom);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const float sw = s_sw[lr];
            float am_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float v0 = acc1[i][j][2 * h]     * sw;
                const float v1 = acc1[i][j][2 * h + 1] * sw;
                acc1[i][j][2 * h]     = v0;
                acc1[i][j][2 * h + 1] = v1;
                am_t = fmaxf(am_t, fmaxf(fabsf(v0), fabsf(v1)));
            }
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
            if (t == 0) s_am[lr * 2 + wn] = am_t;
        }
    __syncthreads();

    if (tid < 64) {
        const float am_r = fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]);
        const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
        s_pa[tid] = (erow > -127) ? h2r_pow2_f(erow - 12) : 0.0f;
        s_sA[tid] = (erow > -127) ? h2r_pow2_f(12 - erow) : 0.0f;
    }
    __syncthreads();
    split_acc_to_sA(acc1);

    float acc2[2][4][4] = {};
    run_w64(W1_BT, acc2);

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            const float pa = s_pa[lr];
            float am_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (r < M) {
                    const int c0 = ec + 8 * j;
                    const float2 pb = *reinterpret_cast<const float2*>(W1_inv_sBT + c0);
                    const float2 p0v = *reinterpret_cast<const float2*>(s_p0 + lr * 64 + c0);
                    const float sx = 1.0f / (1.0f + __expf(-p0v.x));
                    const float sy = 1.0f / (1.0f + __expf(-p0v.y));
                    const float ox = ((acc2[i][j][2 * h]     * pa) * pb.x) * (sx * (1.0f + p0v.x * (1.0f - sx)));
                    const float oy = ((acc2[i][j][2 * h + 1] * pa) * pb.y) * (sy * (1.0f + p0v.y * (1.0f - sy)));
                    *reinterpret_cast<float2*>(dpre0 + (int64_t)r * 64 + c0) = make_float2(ox, oy);
                    am_t = fmaxf(am_t, fmaxf(fabsf(ox), fabsf(oy)));
                }
            }
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
            if (t == 0) s_am[lr * 2 + wn] = am_t;
        }
    __syncthreads();

    if (eA_dpre0 && tid < 64) {
        const int r = m0 + tid;
        if (r < M) {
            const float am_r = fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]);
            const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
            eA_dpre0[r] = (int8_t)erow;
        }
    }
}

// =================================================================================================
// v34 Fused 2-layer w=64 Site MLP Forward for LR-SK (32 KB dynamic SMEM -> 2-3 CTAs/SM):
//   pre0 = x[M,K] @ W0[K,64] + alpha_Z[sp] -> store pre0 -> h0 = silu(pre0) in SMEM -> pre1 = h0 @ W1
// =================================================================================================
__global__ void __launch_bounds__(128, 2) h2f_lrsk_ro2_fwd64_kernel(
    int M, int K, const int* __restrict__ species,
    const float* __restrict__ x, const int8_t* __restrict__ eA_x,
    const float* __restrict__ alpha_Z,
    const __half* __restrict__ W0_B, const float* __restrict__ W0_inv_sB,
    const __half* __restrict__ W1_B, const float* __restrict__ W1_inv_sB,
    float* __restrict__ pre0_out, float* __restrict__ pre1_out) {
    extern __shared__ __align__(128) unsigned char sm[];
    __shared__ int   s_sp[64];
    __shared__ float s_pa[64];
    __shared__ float s_sA[64];

    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * 64;
    const int g = lane >> 2, t = lane & 3;
    const int er = wm * 32 + g, ec = wn * 32 + 2 * t;
    const int fr = tid >> 3, fj = tid & 7;
    const int cr = tid >> 2, cj = tid & 3;
    const int a_row = wm * 32 + (lane & 15), a_kc = lane >> 4;
    const int b_row = wn * 32 + ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;

    if (tid < 64) {
        const int r = m0 + tid;
        const bool ok = (r < M);
        s_sp[tid] = ok ? species[r] : 0;
        const int e = ok ? (int)eA_x[r] : -127;
        s_sA[tid] = (e > -127) ? h2r_pow2_f(12 - e) : 0.0f;
        s_pa[tid] = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
    }

    constexpr int kTileA = 64 * kHK;
    constexpr int kStageH = 2 * kTileA + 2 * kHTileB;
    const int64_t planeB_w0 = (int64_t)64 * K;
    const __half* bg_w0 = W0_B + (int64_t)cr * K + cj * 8;
    auto stage_bf = [&](int st, int p) { return sbase + (unsigned)((st * kStageH + 2 * kTileA + p * kHTileB) * 2); };
    auto issue_w0 = [&](int kt) {
        const int st = kt & 1, k0 = kt * kHK;
        const unsigned sa = sbase + (unsigned)(st * kStageH * 2);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = fr + 16 * i;
            const bool rok = (m0 + r) < M;
            cp16(sa + r * 128 + ((fj ^ (r & 7)) << 4), x + (int64_t)(rok ? m0 + r : m0) * K + k0 + 4 * fj, rok);
        }
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                cp16(stage_bf(st, p) + swz(cr + 32 * i, cj), bg_w0 + p * planeB_w0 + (int64_t)32 * i * K + k0, true);
        asm volatile("cp.async.commit_group;");
    };

    float acc0[2][4][4] = {};
    {
        unsigned accX[2][4][2] = {};
        const int nk = K / kHK;
        issue_w0(0);
        for (int kt = 0; kt < nk; ++kt) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (kt + 1 < nk) issue_w0(kt + 1);
            const int st = kt & 1;
            const unsigned char* sa = sm + st * kStageH * 2 + er * 128 + (t & 1) * 8;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(stage_bf(st, 0) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(stage_bf(st, 1) + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
                const int ja = 4 * kk + (t >> 1);
                const int o0 = (ja ^ g) << 4, o1 = ((ja + 2) ^ g) << 4;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const unsigned char* s = sa + 16 * i * 128;
                    unsigned ah[4], al[4];
                    const float sc0 = s_sA[er + 16 * i], sc1 = s_sA[er + 16 * i + 8];
                    split2f(*reinterpret_cast<const float2*>(s + o0), sc0, ah[0], al[0]);
                    split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o0), sc1, ah[1], al[1]);
                    split2f(*reinterpret_cast<const float2*>(s + o1), sc0, ah[2], al[2]);
                    split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o1), sc1, ah[3], al[3]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int jp = j >> 1, jb = 2 * (j & 1);
                        float d[4] = {};
                        mma16816(d, ah, bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc0[i][j][e] += d[e];
                        mma16816h(accX[i][j], ah, bl[jp][jb], bl[jp][jb + 1]);
                        mma16816h(accX[i][j], al, bc[jp][jb], bc[jp][jb + 1]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        acc0[i][j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), acc0[i][j][2 * h]);
                        acc0[i][j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), acc0[i][j][2 * h + 1]);
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
        }
    }
    __syncthreads();

    float* s_am = reinterpret_cast<float*>(sm + 16384);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            const float pa = s_pa[lr];
            const int sp = s_sp[lr];
            const float* azrow = alpha_Z + (int64_t)sp * 64;
            float am_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ec + 8 * j;
                if (r < M) {
                    const float2 pb = *reinterpret_cast<const float2*>(W0_inv_sB + c0);
                    const float2 az = *reinterpret_cast<const float2*>(azrow + c0);
                    const float p0 = fmaf(acc0[i][j][2 * h]     * pa, pb.x, az.x);
                    const float p1 = fmaf(acc0[i][j][2 * h + 1] * pa, pb.y, az.y);
                    *reinterpret_cast<float2*>(pre0_out + (int64_t)r * 64 + c0) = make_float2(p0, p1);
                    const float h0x = p0 / (1.0f + __expf(-p0));
                    const float h0y = p1 / (1.0f + __expf(-p1));
                    acc0[i][j][2 * h]     = h0x;
                    acc0[i][j][2 * h + 1] = h0y;
                    am_t = fmaxf(am_t, fmaxf(fabsf(h0x), fabsf(h0y)));
                } else {
                    acc0[i][j][2 * h] = acc0[i][j][2 * h + 1] = 0.0f;
                }
            }
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
            if (t == 0) s_am[lr * 2 + wn] = am_t;
        }
    __syncthreads();

    if (tid < 64) {
        const float am_r = fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]);
        const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
        s_pa[tid] = (erow > -127) ? h2r_pow2_f(erow - 12) : 0.0f;
        s_sA[tid] = (erow > -127) ? h2r_pow2_f(12 - erow) : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const float sa = s_sA[lr];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ec + 8 * j;
                const int kt = c0 >> 5;
                const int kl = c0 & 31;
                unsigned h_pair, l_pair;
                split2f(make_float2(acc0[i][j][2 * h], acc0[i][j][2 * h + 1]), sa, h_pair, l_pair);
                const int off = swz(lr, kl >> 3) + ((kl & 7) << 1);
                *reinterpret_cast<unsigned*>(sm + kt * 4096 + off)        = h_pair;
                *reinterpret_cast<unsigned*>(sm + 8192 + kt * 4096 + off) = l_pair;
            }
        }
    __syncthreads();

    // Step 2: pre1[64, 64] = h0[64, 64] @ W1[64, 64] (double-buffered W1 in sm[16384..32767])
    float acc1[2][4][4] = {};
    {
        constexpr int64_t planeB64 = 64 * 64;
        const __half* bg = W1_B + (int64_t)cr * 64 + cj * 8;
        unsigned accX[2][4][2] = {};
        auto iss_k = [&](int kt) {
            const unsigned sBH = sbase + (unsigned)(16384 + kt * 8192);
            const unsigned sBL = sBH + 4096u;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                cp16(sBH + swz(cr + 32 * i, cj), bg + (int64_t)32 * i * 64 + kt * 32, true);
                cp16(sBL + swz(cr + 32 * i, cj), bg + planeB64 + (int64_t)32 * i * 64 + kt * 32, true);
            }
            asm volatile("cp.async.commit_group;");
        };
        iss_k(0);
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (kt == 0) iss_k(1);
            const unsigned sAH_k = sbase + (unsigned)(kt * 4096);
            const unsigned sAL_k = sbase + (unsigned)(8192 + kt * 4096);
            const unsigned sBH   = sbase + (unsigned)(16384 + kt * 8192);
            const unsigned sBL   = sBH + 4096u;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned ah[2][4], al[2][4], bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    ldsm4(sAH_k + swz(a_row + 16 * i, 2 * kk + a_kc), ah[i]);
                    ldsm4(sAL_k + swz(a_row + 16 * i, 2 * kk + a_kc), al[i]);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(sBH + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(sBL + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int jp = j >> 1, jb = 2 * (j & 1);
                        float d[4] = {};
                        mma16816(d, ah[i], bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc1[i][j][e] += d[e];
                        mma16816h(accX[i][j], ah[i], bl[jp][jb], bl[jp][jb + 1]);
                        mma16816h(accX[i][j], al[i], bc[jp][jb], bc[jp][jb + 1]);
                    }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        acc1[i][j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), acc1[i][j][2 * h]);
                        acc1[i][j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), acc1[i][j][2 * h + 1]);
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
        }
    }

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            if (r < M) {
                const float pa = s_pa[lr];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c0 = ec + 8 * j;
                    const float2 pb = *reinterpret_cast<const float2*>(W1_inv_sB + c0);
                    *reinterpret_cast<float2*>(pre1_out + (int64_t)r * 64 + c0) =
                        make_float2((acc1[i][j][2 * h] * pa) * pb.x,
                                    (acc1[i][j][2 * h + 1] * pa) * pb.y);
                }
            }
        }
}

// =================================================================================================
// v34 Fused 2-layer w=64 Site MLP Backward for LR-SK (32 KB dynamic SMEM -> 2-3 CTAs/SM):
//   dh0 = dpre1[M,64] @ W1^T -> dpre0 = dh0 * dsilu(pre0) in SMEM -> dx[M,K] = dpre0 @ W0^T
// =================================================================================================
__global__ void __launch_bounds__(128, 2) h2f_lrsk_ro2_bwd64_kernel(
    int M, int K,
    const float* __restrict__ dpre1, const int8_t* __restrict__ eA_dpre1,
    const float* __restrict__ pre0,
    const __half* __restrict__ W1_BT, const float* __restrict__ W1_inv_sBT,
    const __half* __restrict__ W0_BT, const float* __restrict__ W0_inv_sBT,
    float* __restrict__ dx) {
    extern __shared__ __align__(128) unsigned char sm[];
    __shared__ float s_pa[64];
    __shared__ float s_sA[64];

    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * 64;
    const int g = lane >> 2, t = lane & 3;
    const int er = wm * 32 + g, ec = wn * 32 + 2 * t;
    const int fr = tid >> 3, fj = tid & 7;
    const int cr = tid >> 2, cj = tid & 3;
    const int a_row = wm * 32 + (lane & 15), a_kc = lane >> 4;
    const int b_row = wn * 32 + ((lane >> 4) << 3) + (lane & 7), b_kc = (lane >> 3) & 1;

    if (tid < 64) {
        const int r = m0 + tid;
        const int e = (r < M) ? (int)eA_dpre1[r] : -127;
        s_sA[tid] = (e > -127) ? h2r_pow2_f(12 - e) : 0.0f;
        s_pa[tid] = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
    }

    // Step 1: dh0[64, 64] = dpre1[64, 64] @ W1^T[64, 64] (nk = 2, uses sm[0..32767])
    constexpr int kTileA = 64 * kHK;
    constexpr int kStageH = 2 * kTileA + 2 * kHTileB;
    constexpr int64_t planeB64 = 64 * 64;
    const __half* bg_w1 = W1_BT + (int64_t)cr * 64 + cj * 8;
    auto stage_bf = [&](int st, int p) { return sbase + (unsigned)((st * kStageH + 2 * kTileA + p * kHTileB) * 2); };
    auto issue_w1 = [&](int kt) {
        const int st = kt & 1, k0 = kt * kHK;
        const unsigned sa = sbase + (unsigned)(st * kStageH * 2);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = fr + 16 * i;
            const bool rok = (m0 + r) < M;
            cp16(sa + r * 128 + ((fj ^ (r & 7)) << 4), dpre1 + (int64_t)(rok ? m0 + r : m0) * 64 + k0 + 4 * fj, rok);
        }
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                cp16(stage_bf(st, p) + swz(cr + 32 * i, cj), bg_w1 + p * planeB64 + (int64_t)32 * i * 64 + k0, true);
        asm volatile("cp.async.commit_group;");
    };

    float acc1[2][4][4] = {};
    {
        unsigned accX[2][4][2] = {};
        issue_w1(0);
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (kt == 0) issue_w1(1);
            const int st = kt & 1;
            const unsigned char* sa = sm + st * kStageH * 2 + er * 128 + (t & 1) * 8;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    ldsm4(stage_bf(st, 0) + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                    ldsm4(stage_bf(st, 1) + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                    for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
                }
                const int ja = 4 * kk + (t >> 1);
                const int o0 = (ja ^ g) << 4, o1 = ((ja + 2) ^ g) << 4;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const unsigned char* s = sa + 16 * i * 128;
                    unsigned ah[4], al[4];
                    const float sc0 = s_sA[er + 16 * i], sc1 = s_sA[er + 16 * i + 8];
                    split2f(*reinterpret_cast<const float2*>(s + o0), sc0, ah[0], al[0]);
                    split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o0), sc1, ah[1], al[1]);
                    split2f(*reinterpret_cast<const float2*>(s + o1), sc0, ah[2], al[2]);
                    split2f(*reinterpret_cast<const float2*>(s + 8 * 128 + o1), sc1, ah[3], al[3]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int jp = j >> 1, jb = 2 * (j & 1);
                        float d[4] = {};
                        mma16816(d, ah, bh[jp][jb], bh[jp][jb + 1]);
                        jam4(d);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc1[i][j][e] += d[e];
                        mma16816h(accX[i][j], ah, bl[jp][jb], bl[jp][jb + 1]);
                        mma16816h(accX[i][j], al, bc[jp][jb], bc[jp][jb + 1]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX[i][j][h]));
                        acc1[i][j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), acc1[i][j][2 * h]);
                        acc1[i][j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), acc1[i][j][2 * h + 1]);
                    }
                    accX[i][j][0] = accX[i][j][1] = 0u;
                }
        }
    }
    __syncthreads();

    // Epilogue 1: dpre0 = dh0 * dsilu(pre0) -> split into sAH/sAL (sm[0..16383])
    float* s_am = reinterpret_cast<float*>(sm + 16384);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const int r = m0 + lr;
            const float pa = s_pa[lr];
            float am_t = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (r < M) {
                    const int c0 = ec + 8 * j;
                    const float2 pb  = *reinterpret_cast<const float2*>(W1_inv_sBT + c0);
                    const float2 p0v = __ldg(reinterpret_cast<const float2*>(pre0 + (int64_t)r * 64 + c0));
                    const float sx = 1.0f / (1.0f + __expf(-p0v.x));
                    const float sy = 1.0f / (1.0f + __expf(-p0v.y));
                    const float ox = ((acc1[i][j][2 * h]     * pa) * pb.x) * (sx * (1.0f + p0v.x * (1.0f - sx)));
                    const float oy = ((acc1[i][j][2 * h + 1] * pa) * pb.y) * (sy * (1.0f + p0v.y * (1.0f - sy)));
                    acc1[i][j][2 * h]     = ox;
                    acc1[i][j][2 * h + 1] = oy;
                    am_t = fmaxf(am_t, fmaxf(fabsf(ox), fabsf(oy)));
                } else {
                    acc1[i][j][2 * h] = acc1[i][j][2 * h + 1] = 0.0f;
                }
            }
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 2));
            am_t = fmaxf(am_t, __shfl_xor_sync(0xffffffffu, am_t, 1));
            if (t == 0) s_am[lr * 2 + wn] = am_t;
        }
    __syncthreads();

    if (tid < 64) {
        const float am_r = fmaxf(s_am[tid * 2 + 0], s_am[tid * 2 + 1]);
        const int erow = (am_r > 0.0f && isfinite(am_r)) ? h2r_exp_ceil(am_r) : -127;
        s_pa[tid] = (erow > -127) ? h2r_pow2_f(erow - 12) : 0.0f;
        s_sA[tid] = (erow > -127) ? h2r_pow2_f(12 - erow) : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lr = er + 16 * i + 8 * h;
            const float sa = s_sA[lr];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c0 = ec + 8 * j;
                const int kt = c0 >> 5;
                const int kl = c0 & 31;
                unsigned h_pair, l_pair;
                split2f(make_float2(acc1[i][j][2 * h], acc1[i][j][2 * h + 1]), sa, h_pair, l_pair);
                const int off = swz(lr, kl >> 3) + ((kl & 7) << 1);
                *reinterpret_cast<unsigned*>(sm + kt * 4096 + off)        = h_pair;
                *reinterpret_cast<unsigned*>(sm + 8192 + kt * 4096 + off) = l_pair;
            }
        }
    __syncthreads();

    // Step 2: dx[64, K] = dpre0[64, 64] @ W0^T[64, K] (double-buffered W0_BT in sm[16384..32767])
    const int n_tiles = K / 64;
    const int total_steps = n_tiles * 2;
    const int64_t planeB_bwd = (int64_t)K * 64;
    auto issue_bwd = [&](int step) {
        const int st = step & 1;
        const int nt = step >> 1;
        const int kt = step & 1;
        const unsigned sBH = sbase + (unsigned)(16384 + st * 8192);
        const unsigned sBL = sBH + 4096u;
        const __half* bg = W0_BT + (int64_t)(nt * 64 + cr) * 64 + kt * 32 + cj * 8;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            cp16(sBH + swz(cr + 32 * i, cj), bg + (int64_t)32 * i * 64, true);
            cp16(sBL + swz(cr + 32 * i, cj), bg + planeB_bwd + (int64_t)32 * i * 64, true);
        }
        asm volatile("cp.async.commit_group;");
    };

    float acc3[2][4][4] = {};
    unsigned accX3[2][4][2] = {};
    issue_bwd(0);
    for (int step = 0; step < total_steps; ++step) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (step + 1 < total_steps) issue_bwd(step + 1);
        const int st = step & 1;
        const int nt = step >> 1;
        const int kt = step & 1;
        const unsigned sAH_k = sbase + (unsigned)(kt * 4096);
        const unsigned sAL_k = sbase + (unsigned)(8192 + kt * 4096);
        const unsigned sBH   = sbase + (unsigned)(16384 + st * 8192);
        const unsigned sBL   = sBH + 4096u;
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned ah[2][4], al[2][4], bh[2][4], bl[2][4], bc[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                ldsm4(sAH_k + swz(a_row + 16 * i, 2 * kk + a_kc), ah[i]);
                ldsm4(sAL_k + swz(a_row + 16 * i, 2 * kk + a_kc), al[i]);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                ldsm4(sBH + swz(b_row + 16 * j, 2 * kk + b_kc), bh[j]);
                ldsm4(sBL + swz(b_row + 16 * j, 2 * kk + b_kc), bl[j]);
#pragma unroll
                for (int r = 0; r < 4; ++r) bc[j][r] = hmul2u(bh[j][r], kHB11);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int jp = j >> 1, jb = 2 * (j & 1);
                    float d[4] = {};
                    mma16816(d, ah[i], bh[jp][jb], bh[jp][jb + 1]);
                    jam4(d);
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc3[i][j][e] += d[e];
                    mma16816h(accX3[i][j], ah[i], bl[jp][jb], bl[jp][jb + 1]);
                    mma16816h(accX3[i][j], al[i], bc[jp][jb], bc[jp][jb + 1]);
                }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(&accX3[i][j][h]));
                    acc3[i][j][2 * h]     = fmaf(xv.x, (float)(1 << (kHC - 11)), acc3[i][j][2 * h]);
                    acc3[i][j][2 * h + 1] = fmaf(xv.y, (float)(1 << (kHC - 11)), acc3[i][j][2 * h + 1]);
                }
                accX3[i][j][0] = accX3[i][j][1] = 0u;
            }
        if (kt == 1) {
            const int n0 = nt * 64;
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int lr = er + 16 * i + 8 * h;
                    const int r = m0 + lr;
                    const float pa = s_pa[lr];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (r < M) {
                            const int c0 = n0 + ec + 8 * j;
                            const float2 pb = *reinterpret_cast<const float2*>(W0_inv_sBT + c0);
                            *reinterpret_cast<float2*>(dx + (int64_t)r * K + c0) =
                                make_float2((acc3[i][j][2 * h] * pa) * pb.x,
                                            (acc3[i][j][2 * h + 1] * pa) * pb.y);
                        }
                        acc3[i][j][2 * h] = acc3[i][j][2 * h + 1] = 0.0f;
                    }
                }
        }
    }
}

__global__ void h2r_splitk_reduce_kernel(int M, int N, int S, const float* __restrict__ ws,
                                         const int8_t* __restrict__ eA, const float* __restrict__ inv_sB,
                                         float* __restrict__ C, float beta) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x, mn = (int64_t)M * N;
    if (2 * i >= mn) return;
    float2 s = reinterpret_cast<const float2*>(ws)[i];
    for (int z = 1; z < S; ++z) {
        const float2 p = reinterpret_cast<const float2*>(ws + z * mn)[i];
        s.x += p.x;
        s.y += p.y;
    }
    const int r = (int)(2 * i / N), c = (int)(2 * i % N);
    const int e = (int)eA[r];
    const float pa = (e > -127) ? h2r_pow2_f(e - 12) : 0.0f;
    const float2 pb = *reinterpret_cast<const float2*>(inv_sB + c);
    float2* o = reinterpret_cast<float2*>(C + 2 * i);
    float2 v = make_float2((s.x * pa) * pb.x, (s.y * pa) * pb.y);
    if (beta != 0.0f) {
        const float2 cv = *o;
        v.x = fmaf(beta, cv.x, v.x);
        v.y = fmaf(beta, cv.y, v.y);
    }
    *o = v;
}

inline void launch_h2r_row_exp(int M, int K, const float* A, int8_t* eA, cudaStream_t stream) {
    if (K <= 256)
        h2r_row_exp_kernel<8><<<(M + 31) / 32, 256, 0, stream>>>(M, K, A, eA);
    else if (K <= 512)
        h2r_row_exp_kernel<16><<<(M + 15) / 16, 256, 0, stream>>>(M, K, A, eA);
    else
        h2r_row_exp_kernel<32><<<(M + 7) / 8, 256, 0, stream>>>(M, K, A, eA);
}

int h2r_split_k(int M, int N, int K, int bm_t, size_t ws_bytes) {
    const int nk = (K + kHK - 1) / kHK;
    if (nk < 8 || ws_bytes == 0 || N % 2 != 0) return 1;
    int dev = 0, sms = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0)
        return 1;
    const int64_t tiles = (int64_t)((N + kHN - 1) / kHN) * ((M + bm_t - 1) / bm_t);
    const int64_t slots = (bm_t == 64 ? 3 : 2) * (int64_t)sms;
    if (2 * tiles > slots) return 1;
    int64_t s = std::min<int64_t>(nk / 4, slots / tiles);
    s = std::min<int64_t>(s, (int64_t)(ws_bytes / ((size_t)M * N * sizeof(float))));
    if (s < 2) return 1;
    const int nks = (nk + (int)s - 1) / (int)s;
    return (nk + nks - 1) / nks;
}
}  // namespace

void hgemm2_row_init() {
    auto cfg = [](auto k, int smem) {
        if (smem >= 49152) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    };
    cfg(hgemm2f_row_kernel<128, false, false, 0, false>, kHSmem);
    cfg(hgemm2f_row_kernel<128, true,  false, 0, false>, kHSmem);
    cfg(hgemm2f_row_kernel<128, false, true,  0, false>, kHSmem);
    cfg(hgemm2f_row_kernel<128, true,  true,  0, false>, kHSmem);
    cfg(hgemm2f_row_kernel<64,  false, false, 0, false>, 32768);
    cfg(hgemm2f_row_kernel<64,  true,  false, 0, false>, 32768);
    cfg(hgemm2f_row_kernel<64,  false, true,  0, false>, 32768);
    cfg(hgemm2f_row_kernel<64,  true,  true,  0, false>, 32768);
    cfg(hgemm2f_row_kernel<64,  false, false, 0, false, 1>, 32768);
    cfg(hgemm2f_row_kernel<64,  true,  false, 0, false, 1>, 32768);
    cfg(hgemm2f_row_kernel<64,  false, false, 0, false, 2>, 32768);
    cfg(hgemm2f_row_kernel<64,  true,  false, 0, false, 2>, 32768);
    cfg(hgemm2f_row_kernel<128, false, false, 1, false>, kHSmem);
    cfg(hgemm2f_row_kernel<128, false, false, 3, false>, kHSmem);
    cfg(hgemm2f_row_kernel<128, false, false, 5, false>, kHSmem);
    cfg(hgemm2f_row_kernel<128, false, false, 7, false>, kHSmem);
    cfg(hgemm2f_row_kernel<128, false, false, 3, true>,  kHSmem);
    cfg(hgemm2f_row_kernel<128, false, false, 5, true>,  kHSmem);
    cfg(hgemm2f_row_kernel<128, false, false, 7, true>,  kHSmem);
    cfg(hgemm2f_row_kernel<128, true,  false, 3, true>,  kHSmem);
    cfg(hgemm2f_row_kernel<128, true,  false, 5, true>,  kHSmem);
    cfg(hgemm2f_row_kernel<128, true,  false, 7, true>,  kHSmem);
    cfg(h2f_64x128_k128_fused, 65536);
    cfg(h2f_ilmlip_ro_tail_w128_kernel<false>, 49152);
    cfg(h2f_ilmlip_ro_tail_w128_kernel<true>,  49152);
    cfg(h2f_ilmlip_ro_tail_w64_kernel<false>,  24576);
    cfg(h2f_ilmlip_ro_tail_w64_kernel<true>,   24576);
    cfg(h2f_ilmlip_ro1_fused_kernel<64, false>, 32768);
    cfg(h2f_ilmlip_ro1_fused_kernel<64, true>,  32768);
    cfg(h2f_ilmlip_ro1_fused_w32_kernel<false>, 24576);
    cfg(h2f_ilmlip_ro1_fused_w32_kernel<true>,  24576);
    cfg(h2f_ilmlip_ro1_fwd_tail_kernel<64, false>, 32768);
    cfg(h2f_ilmlip_ro1_fwd_tail_kernel<64, true>,  32768);
    cfg(h2f_ilmlip_ro1_fwd_tail_kernel<32, false>, 32768);
    cfg(h2f_ilmlip_ro1_fwd_tail_kernel<32, true>,  32768);
    cfg(h2f_ilmlip_ro2_fused_w64_kernel<false>, 49152);
    cfg(h2f_ilmlip_ro2_fused_w64_kernel<true>,  49152);
    cfg(h2f_ilmlip_ro2_fwd_tail_w64_kernel<false>, 49152);
    cfg(h2f_ilmlip_ro2_fwd_tail_w64_kernel<true>,  49152);
    cfg(h2f_lrsk_ro2_fwd64_kernel, 32768);
    cfg(h2f_lrsk_ro2_bwd64_kernel, 32768);
}

void hgemm2_ilmlip_ro1_fwd_tail(
    int N, int K, int rw, bool zbl_sw, float inv_zc, const int* species,
    const float* x, const int8_t* eA_x, const float* alpha_Z, const float* u_Z,
    const void* W0_B, const float* W0_inv_sB,
    const double* e_zbl, double* energy, double* atom_energy, float* w_zbl_out,
    float* dpre0, int8_t* eA_dpre0, void* ws, cudaStream_t stream) {
    const int8_t* ea = eA_x;
    if (!ea) {
        int8_t* ea_w = static_cast<int8_t*>(ws);
        launch_h2r_row_exp(N, K, x, ea_w, stream);
        ea = ea_w;
    }
    const dim3 grid(1, (N + 63) / 64);
    const __half* bf = static_cast<const __half*>(W0_B);
    if (rw == 32) {
        if (zbl_sw) {
            h2f_ilmlip_ro1_fwd_tail_kernel<32, true><<<grid, 128, 32768, stream>>>(
                N, K, inv_zc, species, x, ea, alpha_Z, u_Z, bf, W0_inv_sB,
                e_zbl, energy, atom_energy, w_zbl_out, dpre0, eA_dpre0);
        } else {
            h2f_ilmlip_ro1_fwd_tail_kernel<32, false><<<grid, 128, 32768, stream>>>(
                N, K, inv_zc, species, x, ea, alpha_Z, u_Z, bf, W0_inv_sB,
                e_zbl, energy, atom_energy, w_zbl_out, dpre0, eA_dpre0);
        }
    } else {
        if (zbl_sw) {
            h2f_ilmlip_ro1_fwd_tail_kernel<64, true><<<grid, 128, 32768, stream>>>(
                N, K, inv_zc, species, x, ea, alpha_Z, u_Z, bf, W0_inv_sB,
                e_zbl, energy, atom_energy, w_zbl_out, dpre0, eA_dpre0);
        } else {
            h2f_ilmlip_ro1_fwd_tail_kernel<64, false><<<grid, 128, 32768, stream>>>(
                N, K, inv_zc, species, x, ea, alpha_Z, u_Z, bf, W0_inv_sB,
                e_zbl, energy, atom_energy, w_zbl_out, dpre0, eA_dpre0);
        }
    }
}

void hgemm2_ilmlip_ro2_fwd_tail_w64(
    int N, int K, bool zbl_sw, float inv_zc, const int* species,
    const float* x, const int8_t* eA_x, const float* alpha_Z, const float* b1, const float* u_Z,
    const void* W0_B, const float* W0_inv_sB,
    const void* W1_B, const float* W1_inv_sB, const void* W1_BT, const float* W1_inv_sBT,
    const double* e_zbl, double* energy, double* atom_energy, float* w_zbl_out,
    float* dpre0, int8_t* eA_dpre0, void* ws, cudaStream_t stream) {
    const int8_t* ea = eA_x;
    if (!ea) {
        int8_t* ea_w = static_cast<int8_t*>(ws);
        launch_h2r_row_exp(N, K, x, ea_w, stream);
        ea = ea_w;
    }
    const dim3 grid(1, (N + 63) / 64);
    const __half* bf0 = static_cast<const __half*>(W0_B);
    const __half* bf1 = static_cast<const __half*>(W1_B);
    const __half* bb1 = static_cast<const __half*>(W1_BT);
    if (zbl_sw) {
        h2f_ilmlip_ro2_fwd_tail_w64_kernel<true><<<grid, 128, 49152, stream>>>(
            N, K, inv_zc, species, x, ea, alpha_Z, b1, u_Z, bf0, W0_inv_sB,
            bf1, W1_inv_sB, bb1, W1_inv_sBT, e_zbl, energy, atom_energy, w_zbl_out, dpre0, eA_dpre0);
    } else {
        h2f_ilmlip_ro2_fwd_tail_w64_kernel<false><<<grid, 128, 49152, stream>>>(
            N, K, inv_zc, species, x, ea, alpha_Z, b1, u_Z, bf0, W0_inv_sB,
            bf1, W1_inv_sB, bb1, W1_inv_sBT, e_zbl, energy, atom_energy, w_zbl_out, dpre0, eA_dpre0);
    }
}

void hgemm2_ilmlip_ro1_fused(
    int N, int K, int rw, bool zbl_sw, float inv_zc, const int* species,
    const float* x, const int8_t* eA_x, const float* alpha_Z, const float* u_Z,
    const void* W0_B, const float* W0_inv_sB, const void* W0_BT, const float* W0_inv_sBT,
    const double* e_zbl, double* energy, double* atom_energy, float* w_zbl_out, float* dx,
    void* ws, cudaStream_t stream) {
    const int8_t* ea = eA_x;
    if (!ea) {
        int8_t* ea_w = static_cast<int8_t*>(ws);
        launch_h2r_row_exp(N, K, x, ea_w, stream);
        ea = ea_w;
    }
    const dim3 grid(1, (N + 63) / 64);
    const __half* bf = static_cast<const __half*>(W0_B);
    const __half* bb = static_cast<const __half*>(W0_BT);
    if (rw == 32) {
        if (zbl_sw) {
            h2f_ilmlip_ro1_fused_w32_kernel<true><<<grid, 128, 24576, stream>>>(
                N, K, inv_zc, species, x, ea, alpha_Z, u_Z, bf, W0_inv_sB, bb, W0_inv_sBT,
                e_zbl, energy, atom_energy, w_zbl_out, dx);
        } else {
            h2f_ilmlip_ro1_fused_w32_kernel<false><<<grid, 128, 24576, stream>>>(
                N, K, inv_zc, species, x, ea, alpha_Z, u_Z, bf, W0_inv_sB, bb, W0_inv_sBT,
                e_zbl, energy, atom_energy, w_zbl_out, dx);
        }
    } else {
        if (zbl_sw) {
            h2f_ilmlip_ro1_fused_kernel<64, true><<<grid, 128, 32768, stream>>>(
                N, K, inv_zc, species, x, ea, alpha_Z, u_Z, bf, W0_inv_sB, bb, W0_inv_sBT,
                e_zbl, energy, atom_energy, w_zbl_out, dx);
        } else {
            h2f_ilmlip_ro1_fused_kernel<64, false><<<grid, 128, 32768, stream>>>(
                N, K, inv_zc, species, x, ea, alpha_Z, u_Z, bf, W0_inv_sB, bb, W0_inv_sBT,
                e_zbl, energy, atom_energy, w_zbl_out, dx);
        }
    }
}

void hgemm2_ilmlip_ro2_fused_w64(
    int N, int K, bool zbl_sw, float inv_zc, const int* species,
    const float* x, const int8_t* eA_x, const float* alpha_Z, const float* b1, const float* u_Z,
    const void* W0_B, const float* W0_inv_sB, const void* W0_BT, const float* W0_inv_sBT,
    const void* W1_B, const float* W1_inv_sB, const void* W1_BT, const float* W1_inv_sBT,
    const double* e_zbl, double* energy, double* atom_energy, float* w_zbl_out, float* dx,
    void* ws, cudaStream_t stream) {
    const int8_t* ea = eA_x;
    if (!ea) {
        int8_t* ea_w = static_cast<int8_t*>(ws);
        launch_h2r_row_exp(N, K, x, ea_w, stream);
        ea = ea_w;
    }
    const dim3 grid(1, (N + 63) / 64);
    const __half* bf0 = static_cast<const __half*>(W0_B);
    const __half* bb0 = static_cast<const __half*>(W0_BT);
    const __half* bf1 = static_cast<const __half*>(W1_B);
    const __half* bb1 = static_cast<const __half*>(W1_BT);
    if (zbl_sw) {
        h2f_ilmlip_ro2_fused_w64_kernel<true><<<grid, 128, 49152, stream>>>(
            N, K, inv_zc, species, x, ea, alpha_Z, b1, u_Z, bf0, W0_inv_sB, bb0, W0_inv_sBT,
            bf1, W1_inv_sB, bb1, W1_inv_sBT, e_zbl, energy, atom_energy, w_zbl_out, dx);
    } else {
        h2f_ilmlip_ro2_fused_w64_kernel<false><<<grid, 128, 49152, stream>>>(
            N, K, inv_zc, species, x, ea, alpha_Z, b1, u_Z, bf0, W0_inv_sB, bb0, W0_inv_sBT,
            bf1, W1_inv_sB, bb1, W1_inv_sBT, e_zbl, energy, atom_energy, w_zbl_out, dx);
    }
}

void hgemm2_ilmlip_ro_tail_w128(
    int N, bool zbl_sw, float inv_zc, const int* species,
    const float* pre0, const float* alpha_Z, const float* b1, const float* u_Z,
    const void* W1_B, const float* W1_inv_sB, const void* W1_BT, const float* W1_inv_sBT,
    const double* e_zbl, double* energy, double* atom_energy, float* w_zbl_out, float* dpre0,
    int8_t* eA_dpre0, cudaStream_t stream) {
    const dim3 grid(1, (N + 63) / 64);
    const __half* bf = static_cast<const __half*>(W1_B);
    const __half* bb = static_cast<const __half*>(W1_BT);
    if (zbl_sw) {
        h2f_ilmlip_ro_tail_w128_kernel<true><<<grid, 256, 49152, stream>>>(
            N, inv_zc, species, pre0, alpha_Z, b1, u_Z, bf, W1_inv_sB, bb, W1_inv_sBT,
            e_zbl, energy, atom_energy, w_zbl_out, dpre0, eA_dpre0);
    } else {
        h2f_ilmlip_ro_tail_w128_kernel<false><<<grid, 256, 49152, stream>>>(
            N, inv_zc, species, pre0, alpha_Z, b1, u_Z, bf, W1_inv_sB, bb, W1_inv_sBT,
            e_zbl, energy, atom_energy, w_zbl_out, dpre0, eA_dpre0);
    }
}

void hgemm2_ilmlip_ro_tail_w64(
    int N, bool zbl_sw, float inv_zc, const int* species,
    const float* pre0, const float* alpha_Z, const float* b1, const float* u_Z,
    const void* W1_B, const float* W1_inv_sB, const void* W1_BT, const float* W1_inv_sBT,
    const double* e_zbl, double* energy, double* atom_energy, float* w_zbl_out, float* dpre0,
    int8_t* eA_dpre0, cudaStream_t stream) {
    const dim3 grid(1, (N + 63) / 64);
    const __half* bf = static_cast<const __half*>(W1_B);
    const __half* bb = static_cast<const __half*>(W1_BT);
    if (zbl_sw) {
        h2f_ilmlip_ro_tail_w64_kernel<true><<<grid, 128, 24576, stream>>>(
            N, inv_zc, species, pre0, alpha_Z, b1, u_Z, bf, W1_inv_sB, bb, W1_inv_sBT,
            e_zbl, energy, atom_energy, w_zbl_out, dpre0, eA_dpre0);
    } else {
        h2f_ilmlip_ro_tail_w64_kernel<false><<<grid, 128, 24576, stream>>>(
            N, inv_zc, species, pre0, alpha_Z, b1, u_Z, bf, W1_inv_sB, bb, W1_inv_sBT,
            e_zbl, energy, atom_energy, w_zbl_out, dpre0, eA_dpre0);
    }
}

void hgemm2_row_exp(int M, int K, const float* A, int8_t* eA, cudaStream_t stream) {
    launch_h2r_row_exp(M, K, A, eA, stream);
}

void hgemm2_pack_b_col(int Nr, int Kc, const float* w, bool trans, void* out_panel, float* out_inv_sB,
                       cudaStream_t stream) {
    const int Np = (Nr + kHN - 1) / kHN * kHN, Kp = (Kc + kHK - 1) / kHK * kHK;
    h2r_pack_b_col_kernel<<<Np, 128, 0, stream>>>(Nr, Kc, Np, Kp, w, trans ? 1 : 0,
                                                  static_cast<__half*>(out_panel), out_inv_sB);
}

void hgemm2_f32a_row(int M, int N, int K, const float* A, const void* B_panel, const float* inv_sB,
                     float* C, float beta, void* ws, size_t ws_bytes, cudaStream_t stream,
                     const int8_t* eA_cached) {
    const __half* b = static_cast<const __half*>(B_panel);
    // Fast path: K == 128 && N % 128 == 0 -> fused in-kernel eA, ZERO row_exp launch, 1x A DRAM read!
    if (K == 128 && (N % 128) == 0 && M >= 128) {
        const dim3 grid(N / 128, (M + 63) / 64);
        h2f_64x128_k128_fused<<<grid, 256, 49152, stream>>>(M, N, A, b, inv_sB, C, beta);
        return;
    }
    constexpr size_t kEAOff = 256u << 10;
    const int8_t* eA = eA_cached;
    float* sk_ws = static_cast<float*>(ws);
    size_t sk_bytes = ws_bytes;
    if (!eA) {
        int8_t* eA_w = static_cast<int8_t*>(ws);
        launch_h2r_row_exp(M, K, A, eA_w, stream);
        eA = eA_w;
        if (ws && ws_bytes > kEAOff) {
            sk_ws = reinterpret_cast<float*>(static_cast<uint8_t*>(ws) + kEAOff);
            sk_bytes = ws_bytes - kEAOff;
        } else {
            sk_ws = nullptr;
            sk_bytes = 0;
        }
    }
    const bool use64 = (M <= 1600) || (N <= 128 && M <= 6000);
    const int bm_t = use64 ? 64 : 128;
    const int smem = use64 ? 32768 : kHSmem;
    const GateArgs ga{};
    const int S = sk_ws ? h2r_split_k(M, N, K, bm_t, sk_bytes) : 1;
    if (S > 1) {
        const dim3 gs((N + kHN - 1) / kHN, (M + bm_t - 1) / bm_t, S);
        const bool tail = h2_tail(N, K, 0.0f);
        if (use64) {
            if (tail) hgemm2f_row_kernel<64,  true,  true, 0, false><<<gs, 128, smem, stream>>>(M, N, K, A, eA, b, inv_sB, sk_ws, 0.0f, ga);
            else      hgemm2f_row_kernel<64,  false, true, 0, false><<<gs, 128, smem, stream>>>(M, N, K, A, eA, b, inv_sB, sk_ws, 0.0f, ga);
        } else {
            if (tail) hgemm2f_row_kernel<128, true,  true, 0, false><<<gs, 128, smem, stream>>>(M, N, K, A, eA, b, inv_sB, sk_ws, 0.0f, ga);
            else      hgemm2f_row_kernel<128, false, true, 0, false><<<gs, 128, smem, stream>>>(M, N, K, A, eA, b, inv_sB, sk_ws, 0.0f, ga);
        }
        const int64_t pairs = (int64_t)M * N / 2;
        h2r_splitk_reduce_kernel<<<(unsigned)((pairs + 255) / 256), 256, 0, stream>>>(M, N, S, sk_ws, eA, inv_sB, C, beta);
        return;
    }
    const dim3 grid((N + kHN - 1) / kHN, (M + bm_t - 1) / bm_t);
    const bool tail = (N % kHN != 0) || (K % kHK != 0);
    if (use64) {
        if (tail) hgemm2f_row_kernel<64,  true,  false, 0, false><<<grid, 128, smem, stream>>>(M, N, K, A, eA, b, inv_sB, C, beta, ga);
        else      hgemm2f_row_kernel<64,  false, false, 0, false><<<grid, 128, smem, stream>>>(M, N, K, A, eA, b, inv_sB, C, beta, ga);
    } else {
        if (tail) hgemm2f_row_kernel<128, true,  false, 0, false><<<grid, 128, smem, stream>>>(M, N, K, A, eA, b, inv_sB, C, beta, ga);
        else      hgemm2f_row_kernel<128, false, false, 0, false><<<grid, 128, smem, stream>>>(M, N, K, A, eA, b, inv_sB, C, beta, ga);
    }
}

void hgemm2_f32a_row_silu_fwd64(int M, int K, const float* A, const void* B_panel, const float* inv_sB,
                                bool per_sp_bias, const int* species, const float* bias,
                                float* pre_out, float* h_out, int8_t* eA_out,
                                const int8_t* eA_in, cudaStream_t stream) {
    const dim3 grid(1, (M + 63) / 64);
    const __half* b = static_cast<const __half*>(B_panel);
    const GateArgs ga{};
    const SiluEpiArgs sa{species, bias, pre_out, eA_out, per_sp_bias};
    if (K % kHK != 0)
        hgemm2f_row_kernel<64, true,  false, 0, false, 1><<<grid, 128, 32768, stream>>>(M, 64, K, A, eA_in, b, inv_sB, h_out, 0.0f, ga, sa);
    else
        hgemm2f_row_kernel<64, false, false, 0, false, 1><<<grid, 128, 32768, stream>>>(M, 64, K, A, eA_in, b, inv_sB, h_out, 0.0f, ga, sa);
}

void hgemm2_f32a_row_silu_bwd64(int M, int K, const float* A, const void* B_panel, const float* inv_sB,
                                const float* pre_in, float* dpre_out, int8_t* eA_out,
                                const int8_t* eA_in, cudaStream_t stream) {
    const dim3 grid(1, (M + 63) / 64);
    const __half* b = static_cast<const __half*>(B_panel);
    const GateArgs ga{};
    const SiluEpiArgs sa{nullptr, pre_in, nullptr, eA_out, false};
    if (K % kHK != 0)
        hgemm2f_row_kernel<64, true,  false, 0, false, 2><<<grid, 128, 32768, stream>>>(M, 64, K, A, eA_in, b, inv_sB, dpre_out, 0.0f, ga, sa);
    else
        hgemm2f_row_kernel<64, false, false, 0, false, 2><<<grid, 128, 32768, stream>>>(M, 64, K, A, eA_in, b, inv_sB, dpre_out, 0.0f, ga, sa);
}

void hgemm2_lrsk_ro2_fwd64(int M, int K, const int* species, const float* x, const int8_t* eA_x,
                           const float* alpha_Z, const void* W0_B, const float* W0_inv_sB,
                           const void* W1_B, const float* W1_inv_sB,
                           float* pre0_out, float* pre1_out, cudaStream_t stream) {
    const dim3 grid(1, (M + 63) / 64);
    h2f_lrsk_ro2_fwd64_kernel<<<grid, 128, 32768, stream>>>(
        M, K, species, x, eA_x, alpha_Z,
        static_cast<const __half*>(W0_B), W0_inv_sB,
        static_cast<const __half*>(W1_B), W1_inv_sB,
        pre0_out, pre1_out);
}

void hgemm2_lrsk_ro2_bwd64(int M, int K, const float* dpre1, const int8_t* eA_dpre1,
                           const float* pre0, const void* W1_BT, const float* W1_inv_sBT,
                           const void* W0_BT, const float* W0_inv_sBT,
                           float* dx_out, cudaStream_t stream) {
    const dim3 grid(1, (M + 63) / 64);
    h2f_lrsk_ro2_bwd64_kernel<<<grid, 128, 32768, stream>>>(
        M, K, dpre1, eA_dpre1, pre0,
        static_cast<const __half*>(W1_BT), W1_inv_sBT,
        static_cast<const __half*>(W0_BT), W0_inv_sBT,
        dx_out);
}

void hgemm2_f32a_row_gate_bwd(int l, int M, int N, int K, const float* A, const void* B_panel, const float* inv_sB,
                              const GateArgs& g, void* ws, cudaStream_t stream, const int8_t* eA_cached) {
    const int8_t* eA = eA_cached;
    if (!eA) {
        int8_t* eA_w = static_cast<int8_t*>(ws);
        launch_h2r_row_exp(M, K, A, eA_w, stream);
        eA = eA_w;
    }
    const int deg = 2 * l + 1, mt = kHM / deg * deg;
    const dim3 grid((N + kHN - 1) / kHN, (M + mt - 1) / mt);
    const __half* b = static_cast<const __half*>(B_panel);
    if (l == 0)      hgemm2f_row_kernel<128, false, false, 1, false><<<grid, 128, kHSmem, stream>>>(M, N, K, A, eA, b, inv_sB, nullptr, 0.0f, g);
    else if (l == 1) hgemm2f_row_kernel<128, false, false, 3, false><<<grid, 128, kHSmem, stream>>>(M, N, K, A, eA, b, inv_sB, nullptr, 0.0f, g);
    else if (l == 2) hgemm2f_row_kernel<128, false, false, 5, false><<<grid, 128, kHSmem, stream>>>(M, N, K, A, eA, b, inv_sB, nullptr, 0.0f, g);
    else             hgemm2f_row_kernel<128, false, false, 7, false><<<grid, 128, kHSmem, stream>>>(M, N, K, A, eA, b, inv_sB, nullptr, 0.0f, g);
}

void hgemm2_f32a_row_gate_fwd(int l, int M, int N, int K, const float* A, const void* B_panel, const float* inv_sB,
                              const GateArgs& g, void* ws, cudaStream_t stream, const int8_t* eA_cached) {
    const int8_t* eA = eA_cached;
    if (!eA) {
        int8_t* eA_w = static_cast<int8_t*>(ws);
        launch_h2r_row_exp(M, K, A, eA_w, stream);
        eA = eA_w;
    }
    const dim3 grid((N + kHN - 1) / kHN, (M + kHM - 1) / kHM);
    const __half* b = static_cast<const __half*>(B_panel);
    const bool tail = h2_tail(N, K, 0.0f);
    if (tail) {
        if (l == 1)      hgemm2f_row_kernel<128, true, false, 3, true><<<grid, 128, kHSmem, stream>>>(M, N, K, A, eA, b, inv_sB, nullptr, 0.0f, g);
        else if (l == 2) hgemm2f_row_kernel<128, true, false, 5, true><<<grid, 128, kHSmem, stream>>>(M, N, K, A, eA, b, inv_sB, nullptr, 0.0f, g);
        else             hgemm2f_row_kernel<128, true, false, 7, true><<<grid, 128, kHSmem, stream>>>(M, N, K, A, eA, b, inv_sB, nullptr, 0.0f, g);
    } else {
        if (l == 1)      hgemm2f_row_kernel<128, false, false, 3, true><<<grid, 128, kHSmem, stream>>>(M, N, K, A, eA, b, inv_sB, nullptr, 0.0f, g);
        else if (l == 2) hgemm2f_row_kernel<128, false, false, 5, true><<<grid, 128, kHSmem, stream>>>(M, N, K, A, eA, b, inv_sB, nullptr, 0.0f, g);
        else             hgemm2f_row_kernel<128, false, false, 7, true><<<grid, 128, kHSmem, stream>>>(M, N, K, A, eA, b, inv_sB, nullptr, 0.0f, g);
    }
}

}  // namespace loup
