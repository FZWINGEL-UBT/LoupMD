#pragma once
// Elementwise, gate, readout, species-GEMM, force-scatter, loss, and AdamW+EMA+Anchor CUDA kernels
// for full pure-CUDA foundation MACE fine-tuning (fwd + 1st-bwd + exact 2nd-bwd double-backprop).
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "mace_ft_types.hpp"

namespace loup_ft {

// --------------------------------------------------------------------------- activation derivatives
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu_f(float x) { return x * sigmoid_f(x); }
__device__ __forceinline__ float dsilu_f(float x) {
    const float s = sigmoid_f(x);
    return s * (1.0f + x * (1.0f - s));
}
__device__ __forceinline__ float d2silu_f(float x) {
    const float s = sigmoid_f(x);
    const float oms = 1.0f - s;
    // d/dx [s + x*s*(1-s)] = s*(1-s)*(2 + x*(1 - 2*s))
    return s * oms * (2.0f + x * (1.0f - 2.0f * s));
}

// --------------------------------------------------------------------------- embedding gather / scatter
__global__ void gather_emb_fwd_kernel(int N, int C0, const int* __restrict__ species,
                                      const float* __restrict__ emb, float* __restrict__ h0) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)N * C0) return;
    const int a = (int)(idx / C0), c = (int)(idx % C0);
    h0[idx] = emb[(int64_t)species[a] * C0 + c];
}

__global__ void scatter_emb_bwd_kernel(int N, int C0, const int* __restrict__ species,
                                       const float* __restrict__ bar_h0, float* __restrict__ grad_emb) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)N * C0) return;
    const int a = (int)(idx / C0), c = (int)(idx % C0);
    atomicAdd(grad_emb + (int64_t)species[a] * C0 + c, bar_h0[idx]);
}

// --------------------------------------------------------------------------- per-species matmul helpers
// Forward:  Y[a, m, j] = beta * Y + sum_k X[a, m, k] * W[sp[a], k, j]
// Backward (dX): dX[a, m, k] = beta * dX + sum_j dY[a, m, j] * W[sp[a], k, j]
// GradW:    dW[s, k, j] += sum_{a: sp[a]==s, m} X[a, m, k] * dY[a, m, j]
// Used for W_skip_msg (kind 0/2) and sc_species==1 W_sc.
__global__ void species_mm_fwd_kernel(int N, int Mdeg, int K, int J, const int* __restrict__ species,
                                      const float* __restrict__ X, const float* __restrict__ W,
                                      float beta, float* __restrict__ Y) {
    const int row = blockIdx.y;  // 0 .. N*Mdeg - 1
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N * Mdeg || j >= J) return;
    const int a = row / Mdeg;
    const int s = species[a];
    const float* xr = X + (int64_t)row * K;
    const float* ws = W + (int64_t)s * K * J + j;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc = fmaf(xr[k], ws[(int64_t)k * J], acc);
    float* yr = Y + (int64_t)row * J + j;
    *yr = (beta == 0.0f) ? acc : fmaf(beta, *yr, acc);
}

__global__ void species_mm_bwd_x_kernel(int N, int Mdeg, int K, int J, const int* __restrict__ species,
                                        const float* __restrict__ dY, const float* __restrict__ W,
                                        float beta, float* __restrict__ dX) {
    const int row = blockIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N * Mdeg || k >= K) return;
    const int a = row / Mdeg;
    const int s = species[a];
    const float* dyr = dY + (int64_t)row * J;
    const float* ws = W + ((int64_t)s * K + k) * J;
    float acc = 0.0f;
    for (int j = 0; j < J; ++j) acc = fmaf(dyr[j], ws[j], acc);
    float* dxr = dX + (int64_t)row * K + k;
    *dxr = (beta == 0.0f) ? acc : fmaf(beta, *dxr, acc);
}

__global__ void species_mm_grad_w_kernel(int N, int Mdeg, int K, int J, const int* __restrict__ species,
                                         const float* __restrict__ X, const float* __restrict__ dY,
                                         float* __restrict__ dW) {
    // 1 thread per (s, k, j) or atomicAdd over rows; for small K*J use block per (k, j) tile?
    // Even simpler & fast: grid over (j, k), loop over rows a*Mdeg (zero atomics!).
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    const int s = blockIdx.z;
    if (j >= J || k >= K) return;
    float acc = 0.0f;
    for (int a = 0; a < N; ++a) {
        if (species[a] != s) continue;
        for (int m = 0; m < Mdeg; ++m) {
            const int64_t row = (int64_t)a * Mdeg + m;
            acc = fmaf(X[row * K + k], dY[row * J + j], acc);
        }
    }
    dW[((int64_t)s * K + k) * J + j] += acc;
}

// --------------------------------------------------------------------------- kind 4 nonlinear gate: fwd, bwd, bwd2
// Forward:
//   s = 1 / (D[a] * beta + alpha)
//   pre0[a, c] = lin0[a, c] * s + res0[a, c]                   (c = 0..4*C-1)
//   o0[a, c] = cs * silu(pre0[a, c])                           (c = 0..C-1)
//   for l = 1..3:
//     g_l = cg * sigmoid(pre0[a, l*C + c])
//     p_{l,m} = lin_l[a, m, c] * s + (l<=lx ? res_l[a, m, c] : 0)
//     o_l[a, m, c] = g_l * p_{l,m}
__global__ void ft_gate_fwd_kernel(
    int N, int C, int lx, const float* __restrict__ D, float alpha, float beta, float cs, float cg,
    const float* __restrict__ lin0, const float* __restrict__ lin1, const float* __restrict__ lin2, const float* __restrict__ lin3,
    const float* __restrict__ res0, const float* __restrict__ res1, const float* __restrict__ res2,
    float* __restrict__ o0, float* __restrict__ o1, float* __restrict__ o2, float* __restrict__ o3) {
    const int a = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (a >= N || c >= C) return;
    const float s = 1.0f / (D[a] * beta + alpha);
    const int64_t r0 = (int64_t)a * 4 * C;
    auto pre0 = [&](int col) { return lin0[r0 + col] * s + (res0 ? res0[r0 + col] : 0.0f); };
    o0[(int64_t)a * C + c] = cs * silu_f(pre0(c));
    const float* lins[3] = {lin1, lin2, lin3};
    const float* ress[3] = {res1, res2, nullptr};
    float* outs[3] = {o1, o2, o3};
    #pragma unroll
    for (int l = 1; l <= 3; ++l) {
        const float g = cg * sigmoid_f(pre0(l * C + c));
        const float* li = lins[l - 1];
        const float* re = (l <= lx) ? ress[l - 1] : nullptr;
        for (int m = 0; m < 2 * l + 1; ++m) {
            const int64_t idx = ((int64_t)a * (2 * l + 1) + m) * C + c;
            const float p = li[idx] * s + (re ? re[idx] : 0.0f);
            outs[l - 1][idx] = g * p;
        }
    }
}

// 1st-order backward of gate_fwd:
//   Given dg0..dg3 (dE/do_l):
//   dp0[a, c] = dg0 * cs * dsilu(pre0_c)
//   dl0[a, c] = dp0[a, c] * s
//   for l=1..3:
//     dg_l = sum_m dg_l[m] * p_{l,m}
//     dp0[a, l*C + c] = dg_l * cg * sg * (1 - sg)
//     dl0[a, l*C + c] = dp0[a, l*C + c] * s
//     dp_l[a, m, c] = dg_l[m] * g_l          (for l <= lx)
//     dl_l[a, m, c] = dg_l[m] * g_l * s
//   dD[a] += -beta * s^2 * sum_{l=0..3, m, c} dp_{l,m,c} * lin_{l,m,c}
__global__ void ft_gate_bwd_kernel(
    int N, int C, int lx, const float* __restrict__ D, float alpha, float beta, float cs, float cg,
    const float* __restrict__ lin0, const float* __restrict__ lin1, const float* __restrict__ lin2, const float* __restrict__ lin3,
    const float* __restrict__ res0, const float* __restrict__ res1, const float* __restrict__ res2,
    const float* __restrict__ dg0, const float* __restrict__ dg1, const float* __restrict__ dg2, const float* __restrict__ dg3,
    float* __restrict__ dp0, float* __restrict__ dp1, float* __restrict__ dp2,
    float* __restrict__ dl0, float* __restrict__ dl1, float* __restrict__ dl2, float* __restrict__ dl3,
    float* __restrict__ dD) {
    const int a = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (a >= N || c >= C) return;
    const float s = 1.0f / (D[a] * beta + alpha);
    const int64_t r0 = (int64_t)a * 4 * C;
    auto pre0 = [&](int col) { return lin0[r0 + col] * s + (res0 ? res0[r0 + col] : 0.0f); };

    float acc = 0.0f;
    {
        const float p = pre0(c);
        const float d = dg0[(int64_t)a * C + c] * cs * dsilu_f(p);
        dp0[r0 + c] = d;
        dl0[r0 + c] = d * s;
        acc = fmaf(d, lin0[r0 + c], acc);
    }
    const float* lins[3] = {lin1, lin2, lin3};
    const float* ress[3] = {res1, res2, nullptr};
    const float* dgs[3] = {dg1, dg2, dg3};
    float* dps[2] = {dp1, dp2};
    float* dls[3] = {dl1, dl2, dl3};
    #pragma unroll
    for (int l = 1; l <= 3; ++l) {
        const float gp = pre0(l * C + c);
        const float sg = sigmoid_f(gp);
        const float g = cg * sg;
        const float* li = lins[l - 1];
        const float* re = (l <= lx) ? ress[l - 1] : nullptr;
        float* dpl = (l <= lx && l <= 2) ? dps[l - 1] : nullptr;
        float dg_sum = 0.0f;
        for (int m = 0; m < 2 * l + 1; ++m) {
            const int64_t idx = ((int64_t)a * (2 * l + 1) + m) * C + c;
            const float p = li[idx] * s + (re ? re[idx] : 0.0f);
            const float go = dgs[l - 1][idx];
            dg_sum = fmaf(go, p, dg_sum);
            const float d = go * g;
            if (dpl) dpl[idx] = d;
            dls[l - 1][idx] = d * s;
            acc = fmaf(d, li[idx], acc);
        }
        const float dgp = dg_sum * cg * sg * (1.0f - sg);
        dp0[r0 + l * C + c] = dgp;
        dl0[r0 + l * C + c] = dgp * s;
        acc = fmaf(dgp, lin0[r0 + l * C + c], acc);
    }
    atomicAdd(dD + a, -beta * s * s * acc);
}

// Exact 2nd-order backward (double-backprop) of gate_fwd + gate_bwd!
// Inputs:
//   Forward saved: D, lin0..3, res0..2
//   1st-bwd saved: dg0..3 (dE/do_l)
//   Upstream adjoints of forward outputs: bar_o0..3 (dL/do_l)
//   Upstream tangents of 1st-bwd outputs: tan_dp0..2 (dL_F/ddp_l), tan_dl0..3 (dL_F/ddl_l), tan_dD (dL_F/ddD)
// Outputs:
//   tan_dg0..3 (dL_F / d(dg_l)) — propagates forward to W_2 and product!
//   bar_lin0..3 (dL / d(lin_l)) — propagates backward to W_lin / C0!
//   bar_res0..2 (dL / d(res_l)) — propagates backward to W_res!
//   bar_D (dL / dD) — accumulated atomically into bar_D[a]!
__global__ void ft_gate_bwd2_kernel(
    int N, int C, int lx, const float* __restrict__ D, float alpha, float beta, float cs, float cg,
    const float* __restrict__ lin0, const float* __restrict__ lin1, const float* __restrict__ lin2, const float* __restrict__ lin3,
    const float* __restrict__ res0, const float* __restrict__ res1, const float* __restrict__ res2,
    const float* __restrict__ dg0, const float* __restrict__ dg1, const float* __restrict__ dg2, const float* __restrict__ dg3,
    const float* __restrict__ bar_o0, const float* __restrict__ bar_o1, const float* __restrict__ bar_o2, const float* __restrict__ bar_o3,
    const float* __restrict__ tan_dp0, const float* __restrict__ tan_dp1, const float* __restrict__ tan_dp2,
    const float* __restrict__ tan_dl0, const float* __restrict__ tan_dl1, const float* __restrict__ tan_dl2, const float* __restrict__ tan_dl3,
    const float* __restrict__ tan_dD,
    float* __restrict__ tan_dg0, float* __restrict__ tan_dg1, float* __restrict__ tan_dg2, float* __restrict__ tan_dg3,
    float* __restrict__ bar_lin0, float* __restrict__ bar_lin1, float* __restrict__ bar_lin2, float* __restrict__ bar_lin3,
    float* __restrict__ bar_res0, float* __restrict__ bar_res1, float* __restrict__ bar_res2,
    float* __restrict__ bar_D) {
    const int a = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (a >= N || c >= C) return;

    const float s = 1.0f / (D[a] * beta + alpha);
    const float s2 = s * s;
    const float tdD = tan_dD ? tan_dD[a] : 0.0f;
    const float coeff_dD = -beta * s2 * tdD;  // dL_F / d(acc), where dD = -beta * s^2 * acc
    const int64_t r0 = (int64_t)a * 4 * C;
    auto pre0 = [&](int col) { return lin0[r0 + col] * s + (res0 ? res0[r0 + col] : 0.0f); };

    float bar_s_total = 0.0f;  // accumulates dL/ds; at end bar_D[a] += -beta * s^2 * bar_s_total

    // --- l = 0 scalar channel c ---
    {
        const float l0 = lin0[r0 + c];
        const float p0 = pre0(c);
        const float go0 = dg0[(int64_t)a * C + c];
        const float d1 = dsilu_f(p0);
        const float d2 = d2silu_f(p0);
        const float d = go0 * cs * d1;  // dp0[c]

        // Effective adjoint on dp0[c]: from tan_dp0, tan_dl0 (dl0 = dp0 * s), and dD (acc += dp0 * l0)
        const float tdp = (tan_dp0 ? tan_dp0[r0 + c] : 0.0f) + tan_dl0[r0 + c] * s + coeff_dD * l0;
        // Contribution to bar_s from dl0 = d * s and dD = -beta * s^2 * (d * l0)
        bar_s_total += tan_dl0[r0 + c] * d + (-2.0f * beta * s * tdD) * (d * l0);

        // tan_dg0 = d(dp0)/d(dg0) * tdp = cs * d1 * tdp
        tan_dg0[(int64_t)a * C + c] = cs * d1 * tdp;

        // bar_p0 = bar_o0 * cs * d1 + tdp * go0 * cs * d2
        const float bo0 = bar_o0 ? bar_o0[(int64_t)a * C + c] : 0.0f;
        const float bar_p0 = bo0 * cs * d1 + tdp * go0 * cs * d2;
        if (bar_res0) bar_res0[r0 + c] += bar_p0;
        bar_lin0[r0 + c] += bar_p0 * s + coeff_dD * d;
        bar_s_total += bar_p0 * l0;
    }

    // --- l = 1..3 gated equivariant channels ---
    const float* lins[3] = {lin1, lin2, lin3};
    const float* ress[3] = {res1, res2, nullptr};
    const float* dgs[3] = {dg1, dg2, dg3};
    const float* bar_os[3] = {bar_o1, bar_o2, bar_o3};
    const float* tan_dps[2] = {tan_dp1, tan_dp2};
    const float* tan_dls[3] = {tan_dl1, tan_dl2, tan_dl3};
    float* tan_dgs[3] = {tan_dg1, tan_dg2, tan_dg3};
    float* bar_lins[3] = {bar_lin1, bar_lin2, bar_lin3};
    float* bar_ress[2] = {bar_res1, bar_res2};

    #pragma unroll
    for (int l = 1; l <= 3; ++l) {
        const float l0_g = lin0[r0 + l * C + c];
        const float gp = pre0(l * C + c);
        const float sg = sigmoid_f(gp);
        const float oms = 1.0f - sg;
        const float g = cg * sg;
        const float dg_dgp = cg * sg * oms;
        const float d2g_dgp2 = cg * sg * oms * (1.0f - 2.0f * sg);

        const float* li = lins[l - 1];
        const float* re = (l <= lx) ? ress[l - 1] : nullptr;
        const float* tdpl_ptr = (l <= lx && l <= 2) ? tan_dps[l - 1] : nullptr;

        // First mini-pass over m: compute dg_sum = sum_m go_m * p_m
        float dg_sum = 0.0f;
        for (int m = 0; m < 2 * l + 1; ++m) {
            const int64_t idx = ((int64_t)a * (2 * l + 1) + m) * C + c;
            const float p = li[idx] * s + (re ? re[idx] : 0.0f);
            dg_sum = fmaf(dgs[l - 1][idx], p, dg_sum);
        }
        const float dgp = dg_sum * dg_dgp;  // dp0[l*C + c]

        // Effective adjoint on dgp = dp0[l*C + c]
        const float tdgp = (tan_dp0 ? tan_dp0[r0 + l * C + c] : 0.0f)
                         + tan_dl0[r0 + l * C + c] * s
                         + coeff_dD * l0_g;
        bar_s_total += tan_dl0[r0 + l * C + c] * dgp + (-2.0f * beta * s * tdD) * (dgp * l0_g);

        // From dgp = dg_sum * dg_dgp:
        const float bar_dg_sum = tdgp * dg_dgp;
        float bar_gp = tdgp * dg_sum * d2g_dgp2;  // will also receive bar_g * dg_dgp below

        float bar_g = 0.0f;
        for (int m = 0; m < 2 * l + 1; ++m) {
            const int64_t idx = ((int64_t)a * (2 * l + 1) + m) * C + c;
            const float lm = li[idx];
            const float pm = lm * s + (re ? re[idx] : 0.0f);
            const float go = dgs[l - 1][idx];
            const float dm = go * g;  // dp_l[idx]
            const float bo = bar_os[l - 1] ? bar_os[l - 1][idx] : 0.0f;

            // Effective adjoint on dm = dp_l[idx]: from tan_dpl, tan_dll (dl_l = dm * s), and dD (acc += dm * lm)
            const float tdm = (tdpl_ptr ? tdpl_ptr[idx] : 0.0f) + tan_dls[l - 1][idx] * s + coeff_dD * lm;
            bar_s_total += tan_dls[l - 1][idx] * dm + (-2.0f * beta * s * tdD) * (dm * lm);

            // tan_dg_l[idx] = d(dm)/d(go) * tdm + d(dg_sum)/d(go) * bar_dg_sum = g * tdm + pm * bar_dg_sum
            tan_dgs[l - 1][idx] = fmaf(g, tdm, pm * bar_dg_sum);

            // bar_g += bo * pm (from o_l = g * pm) + go * tdm (from dm = go * g)
            bar_g = fmaf(bo, pm, fmaf(go, tdm, bar_g));

            // bar_pm = bo * g (from o_l = g * pm) + go * bar_dg_sum (from dg_sum += go * pm)
            const float bar_pm = fmaf(bo, g, go * bar_dg_sum);
            if (l <= lx && l <= 2 && bar_ress[l - 1]) bar_ress[l - 1][idx] += bar_pm;
            bar_lins[l - 1][idx] += fmaf(bar_pm, s, coeff_dD * dm);
            bar_s_total += bar_pm * lm;
        }

        bar_gp += bar_g * dg_dgp;
        if (bar_res0) bar_res0[r0 + l * C + c] += bar_gp;
        bar_lin0[r0 + l * C + c] += fmaf(bar_gp, s, coeff_dD * dgp);
        bar_s_total += bar_gp * l0_g;
    }

    atomicAdd(bar_D + a, -beta * s2 * bar_s_total);
}

// --------------------------------------------------------------------------- readouts (kinds 0, 1, 2): fwd+bwd & bwd2
// Readout kind 0 (linear): E_g += sum_{a in g} h0[a] . w, dh0[a] += w
__global__ void ft_ro_linear_fwd_bwd_kernel(int N, int C, const int* __restrict__ atom_graph,
                                            const float* __restrict__ h0, const float* __restrict__ w,
                                            float* __restrict__ dh0, double* __restrict__ graph_E) {
    const int a = blockIdx.x, c = threadIdx.x;
    if (a >= N) return;
    double e = 0.0;
    for (int j = c; j < C; j += blockDim.x) {
        const float wj = w[j];
        e += (double)h0[(int64_t)a * C + j] * (double)wj;
        dh0[(int64_t)a * C + j] += wj;
    }
    // block reduce e
    __shared__ double sm[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) e += __shfl_xor_sync(0xffffffffu, e, off);
    if (lane == 0) sm[warp] = e;
    __syncthreads();
    if (warp == 0) {
        double tot = (lane < ((blockDim.x + 31) >> 5)) ? sm[lane] : 0.0;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, off);
        if (lane == 0) atomicAdd(graph_E + atom_graph[a], tot);
    }
}

// Readout kind 0 bwd2:
//   bar_h0[a, c] += bar_E[g(a)] * w[c]
//   grad_w[c]    += sum_a (bar_E[g(a)] * h0[a, c] + tan_dh0[a, c])
__global__ void ft_ro_linear_bwd2_kernel(int N, int C, const int* __restrict__ atom_graph,
                                         const float* __restrict__ bar_E, const float* __restrict__ h0,
                                         const float* __restrict__ w, const float* __restrict__ tan_dh0,
                                         float* __restrict__ bar_h0, float* __restrict__ grad_w) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float wc = w[c];
    float gw = 0.0f;
    for (int a = 0; a < N; ++a) {
        const float bE = bar_E[atom_graph[a]];
        bar_h0[(int64_t)a * C + c] += bE * wc;
        gw = fmaf(bE, h0[(int64_t)a * C + c], gw) + (tan_dh0 ? tan_dh0[(int64_t)a * C + c] : 0.0f);
    }
    if (grad_w) grad_w[c] += gw;
}

// Readout MLP activation + final dot:
//   pre[a, j] = hid[a, j] + b[j]
//   act[a, j] = c * silu(pre[a, j])
//   if final_stage:
//     graph_E[g(a)] += sum_j act[a, j] * W2[j]
//     dhid[a, j] = c * dsilu(pre[a, j]) * W2[j]
__global__ void ft_ro_mlp_fwd_bwd_kernel(int N, int H, float act_c, const int* __restrict__ atom_graph,
                                         const float* __restrict__ hid, const float* __restrict__ b,
                                         const float* __restrict__ W2, float* __restrict__ pre_out,
                                         float* __restrict__ act_out, float* __restrict__ dhid_out,
                                         double* __restrict__ graph_E, int final_stage) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)N * H) return;
    const int a = (int)(idx / H), j = (int)(idx % H);
    const float x = hid[idx] + b[j];
    const float y = act_c * silu_f(x);
    pre_out[idx] = x;
    act_out[idx] = y;
    if (final_stage) {
        const float w2 = W2[j];
        dhid_out[idx] = act_c * dsilu_f(x) * w2;
        atomicAdd(graph_E + atom_graph[a], (double)y * (double)w2);
    }
}

// Readout kind 2 stage-1 backward: dhid1 = (dhid2 @ Wm^T) * c * dsilu(pre1)
__global__ void ft_ro_act_mul_kernel(int64_t n, float act_c, const float* __restrict__ pre,
                                     const float* __restrict__ in_grad, float* __restrict__ out_grad) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out_grad[i] = in_grad[i] * act_c * dsilu_f(pre[i]);
}

// Readout MLP final-stage bwd2 (handles both kind 1 and stage 2 of kind 2):
//   E += act(pre) . W2,   dhid = c * dsilu(pre) * W2
//   Given bar_E[g(a)] and tan_dhid[a, j]:
//     bar_pre[a, j] = bar_E[g] * c * dsilu(pre) * W2[j] + tan_dhid[a, j] * c * d2silu(pre) * W2[j]
//     grad_b[j]    += sum_a bar_pre[a, j]
//     grad_W2[j]   += sum_a (bar_E[g] * act[a, j] + tan_dhid[a, j] * c * dsilu(pre))
__global__ void ft_ro_mlp_final_bwd2_kernel(int N, int H, float act_c, const int* __restrict__ atom_graph,
                                            const float* __restrict__ bar_E, const float* __restrict__ pre,
                                            const float* __restrict__ act, const float* __restrict__ W2,
                                            const float* __restrict__ tan_dhid,
                                            float* __restrict__ bar_pre, float* __restrict__ grad_b,
                                            float* __restrict__ grad_W2) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H) return;
    const float w2 = W2[j];
    float gb = 0.0f, gw2 = 0.0f;
    for (int a = 0; a < N; ++a) {
        const int64_t idx = (int64_t)a * H + j;
        const float bE = bar_E[atom_graph[a]];
        const float x = pre[idx];
        const float d1 = act_c * dsilu_f(x);
        const float d2 = act_c * d2silu_f(x);
        const float td = tan_dhid ? tan_dhid[idx] : 0.0f;
        const float bp = (bE * d1 + td * d2) * w2;
        bar_pre[idx] = bp;
        gb += bp;
        gw2 += bE * act[idx] + td * d1;
    }
    if (grad_b) grad_b[j] += gb;
    if (grad_W2) grad_W2[j] += gw2;
}

// Readout kind 2 stage-1 bwd2:
//   act1 = c * silu(pre1),   dhid1 = dh_mid * c * dsilu(pre1)
//   Given bar_act1 (from bar_pre2 @ Wm^T), dh_mid (from dhid2 @ Wm^T), tan_dhid1:
//     bar_pre1[i]   = bar_act1[i] * c * dsilu(pre1) + tan_dhid1[i] * dh_mid[i] * c * d2silu(pre1)
//     tan_dh_mid[i] = tan_dhid1[i] * c * dsilu(pre1)
//     grad_b1[j]   += sum_a bar_pre1[a, j]
__global__ void ft_ro_mlp_mid_bwd2_kernel(int N, int H, float act_c, const float* __restrict__ pre1,
                                          const float* __restrict__ dh_mid, const float* __restrict__ bar_act1,
                                          const float* __restrict__ tan_dhid1,
                                          float* __restrict__ bar_pre1, float* __restrict__ tan_dh_mid,
                                          float* __restrict__ grad_b1) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H) return;
    float gb = 0.0f;
    for (int a = 0; a < N; ++a) {
        const int64_t idx = (int64_t)a * H + j;
        const float x = pre1[idx];
        const float d1 = act_c * dsilu_f(x);
        const float d2 = act_c * d2silu_f(x);
        const float td = tan_dhid1 ? tan_dhid1[idx] : 0.0f;
        const float bp = bar_act1[idx] * d1 + td * dh_mid[idx] * d2;
        bar_pre1[idx] = bp;
        tan_dh_mid[idx] = td * d1;
        gb += bp;
    }
    if (grad_b1) grad_b1[j] += gb;
}

// --------------------------------------------------------------------------- E0 energy & force scatter/gather
__global__ void add_e0_kernel(int N, const int* __restrict__ species, const int* __restrict__ atom_graph,
                              const double* __restrict__ e0, double* __restrict__ graph_E) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < N) atomicAdd(graph_E + atom_graph[a], e0[species[a]]);
}

// LoupMD convention (src/layer0_kernels.cuh L408-413):
//   edge vector v_e = pos[rcv] - pos[snd] (+ shift)
//   edge_dvec[e] = dE / d(v_e)
//   Therefore:
//     dE / d(pos[rcv]) += +edge_dvec[e]  -> F[rcv] -= edge_dvec[e]
//     dE / d(pos[snd]) += -edge_dvec[e]  -> F[snd] += edge_dvec[e]
__global__ void scatter_forces_kernel(int E, const int* __restrict__ senders, const int* __restrict__ receivers,
                                      const float* __restrict__ edge_dvec, float* __restrict__ forces) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const int s = senders[e], r = receivers[e];
    const float fx = edge_dvec[3 * e + 0];
    const float fy = edge_dvec[3 * e + 1];
    const float fz = edge_dvec[3 * e + 2];
    atomicAdd(forces + 3 * r + 0, -fx);
    atomicAdd(forces + 3 * r + 1, -fy);
    atomicAdd(forces + 3 * r + 2, -fz);
    atomicAdd(forces + 3 * s + 0,  fx);
    atomicAdd(forces + 3 * s + 1,  fy);
    atomicAdd(forces + 3 * s + 2,  fz);
}

// Adjoint of scatter_forces_kernel:
//   L_F depends on F[a]; bar_F[a] = dL_F / dF[a]
//   bar_edge_dvec[e] = dL_F / d(edge_dvec[e]) = bar_F[snd] - bar_F[rcv]
__global__ void gather_bar_edge_dvec_kernel(int E, const int* __restrict__ senders, const int* __restrict__ receivers,
                                            const float* __restrict__ bar_F, float* __restrict__ bar_edge_dvec) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const int s = senders[e], r = receivers[e];
    bar_edge_dvec[3 * e + 0] = bar_F[3 * s + 0] - bar_F[3 * r + 0];
    bar_edge_dvec[3 * e + 1] = bar_F[3 * s + 1] - bar_F[3 * r + 1];
    bar_edge_dvec[3 * e + 2] = bar_F[3 * s + 2] - bar_F[3 * r + 2];
}

// --------------------------------------------------------------------------- loss & AdamW + EMA + Anchor
// Energy MSE (per-atom normalized): L_E = w_e / G * sum_g gw[g] * ((E_pred[g] - E_ref[g]) / N_g)^2
// Force MSE:                        L_F = w_f / (3*N) * sum_a gw[g(a)] * ||F_pred[a] - F_ref[a]||^2
__global__ void energy_loss_kernel(int G, float w_e, float inv_accum,
                                   const double* __restrict__ E_pred, const double* __restrict__ E_ref,
                                   const int* __restrict__ graph_natoms, const float* __restrict__ graph_w,
                                   float* __restrict__ bar_E, double* __restrict__ loss_acc) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;
    const double n = (double)graph_natoms[g];
    const double diff = (E_pred[g] - E_ref[g]) / n;
    const double gw = (double)graph_w[g];
    const double le = (double)w_e * gw * diff * diff / (double)G;
    atomicAdd(loss_acc, le);
    bar_E[g] = (float)(2.0 * (double)w_e * gw * diff / (n * (double)G) * (double)inv_accum);
}

__global__ void force_loss_kernel(int N, float w_f, float inv_accum,
                                  const float* __restrict__ F_pred, const float* __restrict__ F_ref,
                                  const int* __restrict__ atom_graph, const float* __restrict__ graph_w,
                                  float* __restrict__ bar_F, double* __restrict__ loss_acc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * N) return;
    const int a = i / 3;
    const double gw = (double)graph_w[atom_graph[a]];
    const double diff = (double)F_pred[i] - (double)F_ref[i];
    const double denom = 3.0 * (double)N;
    atomicAdd(loss_acc, (double)w_f * gw * diff * diff / denom);
    bar_F[i] = (float)(2.0 * (double)w_f * gw * diff / denom * (double)inv_accum);
}

__global__ void grad_norm_sq_kernel(int64_t n, float lr_scale, const float* __restrict__ g, double* __restrict__ acc) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    double v = (i < n) ? ((double)g[i] * (double)lr_scale) : 0.0;
    v = v * v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(acc, v);
}

__global__ void adamw_ema_anchor_kernel(
    int64_t n, float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, float g_scale, float anchor_coeff, float ema_decay,
    float* __restrict__ val, float* __restrict__ grad,
    float* __restrict__ m, float* __restrict__ v,
    float* __restrict__ ema, const float* __restrict__ anchor) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float w = val[i];
    float g = grad[i] * g_scale;
    if (anchor_coeff > 0.0f) g += anchor_coeff * (w - anchor[i]);
    const float mi = beta1 * m[i] + (1.0f - beta1) * g;
    const float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    const float m_hat = mi * bc1;
    const float v_hat = vi * bc2;
    const float w_new = w * (1.0f - lr * wd) - lr * m_hat / (sqrtf(v_hat) + eps);
    val[i] = w_new;
    ema[i] = ema_decay * ema[i] + (1.0f - ema_decay) * w_new;
    grad[i] = 0.0f;
}

__global__ void add_f64_to_f32_kernel(int64_t n, const double* __restrict__ src, float* __restrict__ dst) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += (float)src[i];
}

__global__ void add_scaled_by_barE_kernel(int N, int stride, const int* __restrict__ atom_graph,
                                          const float* __restrict__ bar_E, const float* __restrict__ dh_lr,
                                          float* __restrict__ bar_hout) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * stride) return;
    const int a = (int)(i / stride);
    bar_hout[i] += bar_E[atom_graph[a]] * dh_lr[i];
}

}  // namespace loup_ft
