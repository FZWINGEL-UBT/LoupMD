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

// hgemm2_kernel's gate backward (list 26 item 2, ledger gen-h2-gate-tile): gate_bwd_rows' per-element arithmetic in
// its order, with (a) the tile's per-atom scale s = 1 / (D beta + alpha) and res0 row staged once in shared memory
// (sS, sQ) instead of per thread and atom; (b) B atoms' operand loads issued before their arithmetic (DEG 1: 8 atoms,
// DEG 3: 2; DEG >= 5's lin_l / res_l rows already give a dozen independent loads an atom), since one atom at a time
// left DEG 1's epilogue waiting on its two loads (ncu: long_scoreboard 2.7 per issued instruction, the kernel 3x
// hgemm2f's time on the same grid); (c) each thread's dD partial of an (atom, column) stored in that atom's first
// staged row (and the vform partial in its second), which only this thread reads, and one thread per atom sums the
// 64 columns, instead of a 32-lane shuffle tree and a shared atomic per warp and atom. sgemm128's gate (sgemm.cu)
// keeps gate_bwd_rows, so only the order of dD's sum differs between the two.
template <int DEG, int SROW>
__device__ __forceinline__ void gate_bwd_tile(float* sC, const float* sS, const int* sQ, int a0, int na, int n0,
                                              int tid, const GateArgs& g) {
    const int N = g.C;
    const int cl = tid & 63, grp = tid >> 6, c = n0 + cl;
    constexpr int B = DEG == 1 ? 8 : (DEG == 3 ? 2 : 1);
    constexpr int l = DEG / 2;
    for (int ai0 = grp; ai0 < na; ai0 += 2 * B) {
        float l0[B], r0[B], li[B][DEG], rl[B][DEG];
#pragma unroll
        for (int j = 0; j < B; ++j) {
            const int ai = ai0 + 2 * j;
            const bool ok = ai < na;
            const int aj = ok ? ai : 0;
            const int64_t ro = (int64_t)(a0 + aj) * 4 * N + l * N + c;
            l0[j] = ok ? g.lin0[ro] : 0.0f;
            r0[j] = ok && g.res0 ? g.res0[(int64_t)sQ[aj] * 4 * N + l * N + c] : 0.0f;
            if constexpr (DEG > 1) {
#pragma unroll
                for (int m = 0; m < DEG; ++m) {
                    const int64_t idx = ((int64_t)(a0 + aj) * DEG + m) * N + c;
                    li[j][m] = ok ? g.lin_l[idx] : 0.0f;
                    rl[j][m] = ok && !g.vform && g.res_l ? g.res_l[idx] : 0.0f;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < B; ++j) {
            const int ai = ai0 + 2 * j;
            if (ai >= na) break;
            const int a = a0 + ai;
            const float s = sS[ai];
            const int64_t ro = (int64_t)a * 4 * N + l * N + c;
            float* go = sC + ai * DEG * SROW + cl;
            float part = 0.0f, partq = 0.0f;
            if constexpr (DEG == 1) {
                const float p = l0[j] * s + r0[j];
                const float d = go[0] * g.cs * dsilu_f(p);
                if (g.dp0) g.dp0[ro] = d;
                g.dl0[ro] = d * s;
                part = fmaf(d, l0[j], part);
            } else {
                const float gp = l0[j] * s + r0[j];
                const float sg = sigmoid_f(gp);
                const float gt = g.cg * sg;
                float dgp;
                if (g.vform) {
#pragma unroll
                    for (int m = 0; m < DEG; ++m) {
                        const int64_t idx = ((int64_t)a * DEG + m) * N + c;
                        const float o = go[m * SROW];
                        partq = fmaf(o, li[j][m], partq);
                        g.dl_l[idx] = o * gt * s;
                    }
                    dgp = (1.0f - sg) * partq;
                } else {
                    float dg = 0.0f;
#pragma unroll
                    for (int m = 0; m < DEG; ++m) {
                        const int64_t idx = ((int64_t)a * DEG + m) * N + c;
                        const float p = li[j][m] * s + rl[j][m];
                        const float o = go[m * SROW];
                        dg = fmaf(o, p, dg);
                        const float d = o * gt;
                        if (g.dp_l) g.dp_l[idx] = d;
                        g.dl_l[idx] = d * s;
                        part = fmaf(d, li[j][m], part);
                    }
                    dgp = dg * g.cg * sg * (1.0f - sg);
                }
                if (g.dp0) g.dp0[ro] = dgp;
                g.dl0[ro] = dgp * s;
                part = fmaf(dgp, l0[j], part);
                go[SROW] = partq;  // the atom's second staged row, read above by this thread only
            }
            go[0] = part;
        }
    }
}

// The tile's dD after gate_bwd_tile: thread ai < na sums atom ai's 64 column partials (rotated by 4 every 4 threads,
// so a quarter warp's 16-byte reads cover the banks once) and adds -beta s^2 part - beta s partq, one global atomic.
template <int DEG, int SROW>
__device__ __forceinline__ void gate_bwd_tile_dD(const float* sC, const float* sS, int a0, int na, int tid,
                                                 const GateArgs& g) {
    if (tid >= na) return;
    const float* row = sC + tid * DEG * SROW;
    float p = 0.0f, q = 0.0f;
    const bool vq = DEG > 1 && g.vform;
#pragma unroll
    for (int k = 0; k < 64; k += 4) {
        const int kk = (k + 4 * (tid >> 2)) & 63;
        const float4 v = *reinterpret_cast<const float4*>(row + kk);
        p += (v.x + v.y) + (v.z + v.w);
        if (vq) {
            const float4 w = *reinterpret_cast<const float4*>(row + SROW + kk);
            q += (w.x + w.y) + (w.z + w.w);
        }
    }
    const float s = sS[tid];
    atomicAdd(g.dD + a0 + tid, -g.beta * s * s * p - g.beta * s * q);
}

// C = A B^T from two-half panels: A [2][M][K] (H then L), B [2][N][K] (the weight W_2[l] as stored, [N][K]: H, then
// L 2^-c); out = (the running sum of P + X 2^(c - 11) over the K steps) oscale. DEG 0: C [M][N] row-major; DEG 2l + 1:
// gate_bwd_rows on row tiles of whole atoms (C unused, M = g.natoms DEG, N = g.C). Requirements (checked by the host
// side): N % 64 == 0, K % 32 == 0, A and B 16-byte aligned.
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

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
            // list 11: gate_bwd_rows' operands (the tile's lin_l / res_l rows, its atoms' lin0 / res0 l-slices) are
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
        gate_bwd_tile<DEG, kHSRow>(sC, sS, sQ, a0, na, n0, tid, g);
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
// inside them, lost 1.5-1.9% on MH-1 SrTiO3, where nothing splits).
template <int RV, int DEG, bool TAIL, bool SPLIT = false>
__global__ void __launch_bounds__(128, 2) hgemm2f_kernel(int M, int N, int K, const float* __restrict__ A, float sA,
                                                         const __half* __restrict__ B, float oscale,
                                                         float* __restrict__ C, float beta, GateArgs ga) {
    static_assert(!SPLIT || DEG <= 0, "only the plain and rows instances split");
    extern __shared__ __align__(128) unsigned char hsm[];
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(hsm);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * kHM, n0 = blockIdx.x * kHN;
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
                    *reinterpret_cast<float2*>(ga.out + (int64_t)r * N + n0 + cl) =
                        make_float2(gg.x * ((acc[i][j][2 * h] * oscale) * s), gg.y * ((acc[i][j][2 * h + 1] * oscale) * s));
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
