#include "generic.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

#include "bundle.hpp"
#include "generated_generic.cuh"
#include "model.hpp"
#include "polar.hpp"

namespace loup {

// ------------------------------------------------------------------------------------------------- layer state
struct GenericLayer {
    // configuration (bundle meta)
    int kind = 0;  // 0 agnostic, 1 agnostic_res, 2 density, 3 density_res, 4 nonlinear
    int lmax_in = 0, lmax_x = 0, lmax_out = 0;
    int C_in = 0, C_x = 0, C_msg = 0, C_out = 0;
    int n_paths = 0, n_l3[4] = {0, 0, 0, 0};
    int intervals = 0, Q1 = 0, pair_sym = 0;
    int first_hidden = 0, J = 0;  // layer 0 as an outer sum over the radial chain's J columns (see first_fwd_kernel)
    float* C0[4] = {};            // [S (J - 1), lin_cols]
    float* S0[4] = {};            // [N, 2l+1, S (J - 1)] (the backward reuses it for dS0)
    int has_sc = 0, sc_species = 0, readout = -1, ro_H = 0, prod_species = 0, prod_segw = 0;
    int skip_folded = 0, ro_folded = 0;
    int lin_cols[4] = {0, 0, 0, 0};
    int sc_has[3] = {0, 0, 0};
    SplineGrid grid{};
    float alpha = 1.0f, beta = 0.0f, gate_cs = 1.0f, gate_cg = 1.0f, ro_act_c = 1.0f;
    const GProdKernels* prod = nullptr;
    // weights (device)
    float4* table = nullptr;
    float *W_up[3] = {}, *W_lin[4] = {}, *W_res[3] = {}, *W_2[4] = {}, *W_skip_msg[4] = {}, *W_sc[3] = {}, *W_p[3] = {};
    float* prod_w = nullptr;
    float *ro_w = nullptr, *ro_W1 = nullptr, *ro_W1_sc = nullptr, *ro_b1 = nullptr, *ro_W2 = nullptr,
          *ro_Wm = nullptr, *ro_bm = nullptr;
    std::vector<void*> weights;
    float* h_in[4] = {};  // not owned: previous layer's hout (or the embedding gather)
    float* dh_in[4] = {}; // not owned: previous layer's dhout (nullptr for layer 0)
    float *x[3] = {}, *mid[4] = {}, *D = nullptr, *lin[4] = {}, *res[3] = {}, *gout[4] = {}, *msgf[4] = {};
    float *y[3] = {}, *hout[4] = {}, *hid = nullptr, *hid2 = nullptr;
    float *dhout[4] = {}, *dx[3] = {}, *dD = nullptr, *dy[3] = {}, *dmsgf[4] = {}, *dgout[4] = {}, *dlin[4] = {},
          *dpre[4] = {}, *dhid = nullptr, *dhid2 = nullptr;

    ~GenericLayer() {
        for (void* p : weights) cudaFree(p);
    }
};

namespace {

template <typename T>
T* upload(std::vector<void*>& owner, const std::vector<T>& host) {
    T* dev = nullptr;
    CUDA_CHECK_THROW(cudaMalloc(&dev, std::max<size_t>(host.size(), 1) * sizeof(T)));
    if (!host.empty()) CUDA_CHECK_THROW(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    owner.push_back(dev);
    return dev;
}

inline int dimL(int l) { return 2 * l + 1; }

// ------------------------------------------------------------------------------------------------- kernels
__global__ void gather_rows_kernel(int N, int C, const int* __restrict__ sp, const float* __restrict__ table,
                                   float* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * C) return;
    const int a = (int)(i / C), c = (int)(i % C);
    out[i] = table[(int64_t)sp[a] * C + c];
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float dsilu_f(float x) {
    const float s = sigmoid_f(x);
    return s * (1.0f + x * (1.0f - s));
}

// Nonlinear block: pre_l = lin_l * s + res_l (s = 1 / (beta D + alpha)), then the gate: out_0 = cs silu(pre_0[:C]),
// out_l = cs_g sigmoid(pre_0[l C:(l+1) C]) * pre_l (l = 1..3). One block per atom, one thread per channel.
__global__ void gate_fwd_kernel(int N, int C, int lx, const float* __restrict__ D, float alpha, float beta, float cs,
                                float cg, const float* __restrict__ lin0, const float* __restrict__ lin1,
                                const float* __restrict__ lin2, const float* __restrict__ lin3,
                                const float* __restrict__ res0, const float* __restrict__ res1,
                                const float* __restrict__ res2, float* __restrict__ o0, float* __restrict__ o1,
                                float* __restrict__ o2, float* __restrict__ o3) {
    const int a = blockIdx.x, c = threadIdx.x;
    if (a >= N) return;
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

// Reverse of gate_fwd: dpre_l (the gradient of pre_l, also the gradient of res_l), dlin_l = dpre_l * s and
// dD += sum dpre . lin * (-beta s^2) (one atomic per warp).
__global__ void gate_bwd_kernel(int N, int C, int lx, const float* __restrict__ D, float alpha, float beta, float cs,
                                float cg, const float* __restrict__ lin0, const float* __restrict__ lin1,
                                const float* __restrict__ lin2, const float* __restrict__ lin3,
                                const float* __restrict__ res0, const float* __restrict__ res1,
                                const float* __restrict__ res2, const float* __restrict__ g0,
                                const float* __restrict__ g1, const float* __restrict__ g2,
                                const float* __restrict__ g3, float* __restrict__ dp0, float* __restrict__ dp1,
                                float* __restrict__ dp2, float* __restrict__ dl0, float* __restrict__ dl1,
                                float* __restrict__ dl2, float* __restrict__ dl3, float* __restrict__ dD) {
    const int a = blockIdx.x, c = threadIdx.x;
    if (a >= N) return;
    const float s = 1.0f / (D[a] * beta + alpha);
    const int64_t r0 = (int64_t)a * 4 * C;
    auto pre0 = [&](int col) { return lin0[r0 + col] * s + (res0 ? res0[r0 + col] : 0.0f); };
    float acc = 0.0f;  // sum dpre * lin
    {
        const float p = pre0(c);
        const float d = g0[(int64_t)a * C + c] * cs * dsilu_f(p);
        if (dp0) dp0[r0 + c] = d;
        dl0[r0 + c] = d * s;
        acc = fmaf(d, lin0[r0 + c], acc);
    }
    const float* lins[3] = {lin1, lin2, lin3};
    const float* ress[3] = {res1, res2, nullptr};
    const float* gs[3] = {g1, g2, g3};
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
        float dg = 0.0f;
        for (int m = 0; m < 2 * l + 1; ++m) {
            const int64_t idx = ((int64_t)a * (2 * l + 1) + m) * C + c;
            const float p = li[idx] * s + (re ? re[idx] : 0.0f);
            const float go = gs[l - 1][idx];
            dg = fmaf(go, p, dg);
            const float d = go * g;
            if (dpl) dpl[idx] = d;
            dls[l - 1][idx] = d * s;
            acc = fmaf(d, li[idx], acc);
        }
        const float dgp = dg * cg * sg * (1.0f - sg);
        if (dp0) dp0[r0 + l * C + c] = dgp;
        dl0[r0 + l * C + c] = dgp * s;
        acc = fmaf(dgp, lin0[r0 + l * C + c], acc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if ((c & 31) == 0) atomicAdd(dD + a, -beta * s * s * acc);
}

// Linear readout: E += sum_a h0[a] . w (double accumulation); its gradient dh0 += w.
__global__ void ro_linear_kernel(int N, int C, const float* __restrict__ h0, const float* __restrict__ w,
                                 float* __restrict__ dh0, double* __restrict__ energy) {
    const int a = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (a >= N) return;
    double e = 0.0;
    for (int c = lane; c < C; c += 32) {
        const float wc = w[c];
        e += (double)h0[(int64_t)a * C + c] * (double)wc;
        dh0[(int64_t)a * C + c] += wc;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) e += __shfl_xor_sync(0xffffffffu, e, off);
    if (lane == 0) atomicAdd(energy, e);
}

// MLP readout tail on hid [N, H] (= h0 @ W1, bias added here): kind 1: E += sum c silu(hid + b1) . W2;
// kind 2 (bias MLP): stage 1 writes act1 = c silu(hid + b1) into hid (for the mid GEMM) and keeps the
// pre-activation in pre; stage 2 runs on hid2 = act1 @ Wm. dhid receives dE/d(hid + b1) of the stage.
__global__ void ro_mlp_kernel(int N, int H, float c, float* __restrict__ hid, const float* __restrict__ b,
                              const float* __restrict__ W2, float* __restrict__ pre, float* __restrict__ dhid,
                              double* __restrict__ energy, int final_stage) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if (i < (int64_t)N * H) {
        const int j = (int)(i % H);
        const float x = hid[i] + b[j];
        if (final_stage) {
            const float w2 = W2[j];
            e = (double)(c * silu_f(x)) * (double)w2;
            dhid[i] = c * dsilu_f(x) * w2;
        } else {
            pre[i] = x;
            hid[i] = c * silu_f(x);
        }
    }
    if (final_stage) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) e += __shfl_xor_sync(0xffffffffu, e, off);
        if ((threadIdx.x & 31) == 0 && e != 0.0) atomicAdd(energy, e);
    }
}

// dhid *= c silu'(pre) (the first stage's activation, after the mid GEMM's backward)
__global__ void ro_act_bwd_kernel(int64_t n, float c, const float* __restrict__ pre, float* __restrict__ d) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= c * dsilu_f(pre[i]);
}

// ZBL per edge (ordered pair tables): energy and dE/dvec added to edge_dvec.
__global__ void zbl_kernel(const int* __restrict__ E_dev, int S, const float4* __restrict__ vec_len,
                           const int* __restrict__ senders, const int* __restrict__ receivers,
                           const int* __restrict__ species, const float* __restrict__ rc_t,
                           const float* __restrict__ inv_a_t, const float* __restrict__ pref_t, float c0, float c1,
                           float c2, float c3, float zp, float* __restrict__ edge_dvec, double* __restrict__ energy) {
    const int E = *E_dev;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    double en = 0.0;
    if (e < E) {
        const float4 vl = vec_len[e];
        const int p = species[senders[e]] * S + species[receivers[e]];
        const float rcf = rc_t[p];
        if (vl.w > 0.0f && vl.w < rcf) {
            const double r = vl.w, inv_a = inv_a_t[p], pref = pref_t[p], rc = rcf;
            const double u = r * inv_a;
            const double e0 = exp(-3.2 * u), e1 = exp(-0.9423 * u), e2 = exp(-0.4028 * u), e3 = exp(-0.2016 * u);
            const double phi = c0 * e0 + c1 * e1 + c2 * e2 + c3 * e3;
            const double dphi = -inv_a * (3.2 * c0 * e0 + 0.9423 * c1 * e1 + 0.4028 * c2 * e2 + 0.2016 * c3 * e3);
            const double v = pref / r * phi;
            const double dv = pref / r * (dphi - phi / r);
            const double s = r / rc;
            const double s2 = s * s;
            const double sp = zp == 5.0f ? s2 * s2 * s : pow(s, (double)zp);
            const double env = 1.0 - ((zp + 1.0) * (zp + 2.0) * 0.5) * sp + (zp * (zp + 2.0)) * sp * s -
                               (zp * (zp + 1.0) * 0.5) * sp * s * s;
            const double denv = -(zp * (zp + 1.0) * (zp + 2.0) * 0.5) * (sp / s) * (1.0 - s) * (1.0 - s) / rc;
            en = v * env;
            const float g = (float)((dv * env + v * denv) / r);
            edge_dvec[(int64_t)e * 3 + 0] += g * vl.x;
            edge_dvec[(int64_t)e * 3 + 1] += g * vl.y;
            edge_dvec[(int64_t)e * 3 + 2] += g * vl.z;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) en += __shfl_xor_sync(0xffffffffu, en, off);
    if ((threadIdx.x & 31) == 0 && en != 0.0) atomicAdd(energy, en);
}

// Forces and virial from dE/dvec per edge (vec = pos[rcv] - pos[snd] + shift).
__global__ void assemble_forces_kernel(const int* __restrict__ E_dev, const float4* __restrict__ vec_len,
                                       const int* __restrict__ senders, const int* __restrict__ receivers,
                                       const float* __restrict__ edge_dvec, int compute_stress,
                                       double* __restrict__ forces, double* __restrict__ virial) {
    const int E = *E_dev;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    double v[6] = {0, 0, 0, 0, 0, 0};
    if (e < E) {
        const double fx = edge_dvec[(int64_t)e * 3 + 0], fy = edge_dvec[(int64_t)e * 3 + 1],
                     fz = edge_dvec[(int64_t)e * 3 + 2];
        const int s = senders[e], r = receivers[e];
        atomicAdd(forces + r * 3 + 0, -fx);
        atomicAdd(forces + r * 3 + 1, -fy);
        atomicAdd(forces + r * 3 + 2, -fz);
        atomicAdd(forces + s * 3 + 0, fx);
        atomicAdd(forces + s * 3 + 1, fy);
        atomicAdd(forces + s * 3 + 2, fz);
        if (compute_stress) {
            const float4 vl = vec_len[e];
            v[0] = vl.x * fx;
            v[1] = 0.5 * (vl.x * fy + vl.y * fx);
            v[2] = 0.5 * (vl.x * fz + vl.z * fx);
            v[3] = vl.y * fy;
            v[4] = 0.5 * (vl.y * fz + vl.z * fy);
            v[5] = vl.z * fz;
        }
    }
    if (compute_stress) {
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            double x = v[k];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(0xffffffffu, x, off);
            if ((threadIdx.x & 31) == 0 && x != 0.0) atomicAdd(virial + k, x);
        }
    }
}

// Layer 0 (input = species embedding): S0_l[a][m][s (J-1) + j] = sum over the active edges of receiver a whose sender
// has species s of col_j(r) Y_lm, with col the radial chain's columns [h c, c] (j < J - 1); column J - 1 is the edge
// density, summed into D. One block per receiver, one thread per column; a receiver's edges are grouped by ascending
// sender species (the neighbour list's contract), so each species slot is written once (zeros if it has no edge).
__global__ void first_fwd_kernel(int N, int S, int I, int J, SplineGrid grid, int sh_perm, int pair_sym,
                                 const int* __restrict__ row_ptr, const int* __restrict__ senders,
                                 const float4* __restrict__ vec_len, const int* __restrict__ species,
                                 const float4* __restrict__ table, float* __restrict__ s0, float* __restrict__ s1,
                                 float* __restrict__ s2, float* __restrict__ s3, float* __restrict__ D) {
    const int a = blockIdx.x, j = threadIdx.x;
    if (a >= N || j >= J) return;
    const int Jp = J - 1, K = S * Jp, sa = species[a];
    float* outs[4] = {s0, s1, s2, s3};
    const int e_end = row_ptr[a + 1];
    int e = row_ptr[a];
    float dsum = 0.0f;
    for (int s = 0; s < S; ++s) {
        float acc[16];
        #pragma unroll
        for (int m = 0; m < 16; ++m) acc[m] = 0.0f;
        const int pr = pair_sym ? loup::sym_pair(s, sa, S) : s * S + sa;
        for (; e < e_end && species[senders[e]] == s; ++e) {
            const float4 vl = vec_len[e];
            int iv;
            float t, sxi;
            locate_spline(grid, vl.w, iv, t, sxi);
            const float4 cf = __ldg(table + ((int64_t)pr * I + iv) * J + j);
            const float v = fmaf(t, fmaf(t, fmaf(t, cf.w, cf.z), cf.y), cf.x);
            if (j == Jp) {
                dsum += v;
                continue;
            }
            float4 Yv[4];
            if (sh_perm) loup::edge_sh16(vl.y, vl.z, vl.x, vl.w, Yv);
            else loup::edge_sh16(vl.x, vl.y, vl.z, vl.w, Yv);
            const float Y[16] = {Yv[0].x, Yv[0].y, Yv[0].z, Yv[0].w, Yv[1].x, Yv[1].y, Yv[1].z, Yv[1].w,
                                 Yv[2].x, Yv[2].y, Yv[2].z, Yv[2].w, Yv[3].x, Yv[3].y, Yv[3].z, Yv[3].w};
            #pragma unroll
            for (int m = 0; m < 16; ++m) acc[m] = fmaf(v, Y[m], acc[m]);
        }
        if (j < Jp) {
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                #pragma unroll
                for (int m = 0; m < 2 * l + 1; ++m)
                    outs[l][((int64_t)a * (2 * l + 1) + m) * K + s * Jp + j] = acc[l * l + m];
        }
    }
    if (j == Jp) D[a] = dsum;
}

// Reverse of first_fwd_kernel: per edge, dE/dcol_j = sum_m dS0[j][m] Y_m gives the radial derivative through the
// table slope, dE/dY_m = sum_j dS0[j][m] col_j the angular part; each thread turns its share into dE/dvec, reduced
// over the warp and added once per warp.
__global__ void first_bwd_kernel(int N, int S, int I, int J, SplineGrid grid, int sh_perm, int pair_sym,
                                 const int* __restrict__ row_ptr, const int* __restrict__ senders,
                                 const float4* __restrict__ vec_len, const int* __restrict__ species,
                                 const float4* __restrict__ table, const float* __restrict__ g0,
                                 const float* __restrict__ g1, const float* __restrict__ g2,
                                 const float* __restrict__ g3, const float* __restrict__ dD,
                                 float* __restrict__ edge_dvec) {
    const int a = blockIdx.x, j = threadIdx.x, lane = threadIdx.x & 31;
    if (a >= N) return;
    const int Jp = J - 1, K = S * Jp, sa = species[a];
    const float* gs[4] = {g0, g1, g2, g3};
    const bool active = j < J;
    const float dDa = (dD && j == Jp) ? dD[a] : 0.0f;
    const int e_end = row_ptr[a + 1];
    int e = row_ptr[a];
    for (int s = 0; s < S; ++s) {
        float G[16];
        #pragma unroll
        for (int l = 0; l < 4; ++l)
            #pragma unroll
            for (int m = 0; m < 2 * l + 1; ++m)
                G[l * l + m] = (active && j < Jp) ? gs[l][((int64_t)a * (2 * l + 1) + m) * K + s * Jp + j] : 0.0f;
        const int pr = pair_sym ? loup::sym_pair(s, sa, S) : s * S + sa;
        // every thread of the block walks the same edges (warp-uniform loop bounds for the shuffles)
        for (; e < e_end && species[senders[e]] == s; ++e) {
            const float4 vl = vec_len[e];
            int iv;
            float t, sxi;
            locate_spline(grid, vl.w, iv, t, sxi);
            float v = 0.0f, dv = 0.0f;
            if (active) {
                const float4 cf = __ldg(table + ((int64_t)pr * I + iv) * J + j);
                v = fmaf(t, fmaf(t, fmaf(t, cf.w, cf.z), cf.y), cf.x);
                dv = fmaf(t, fmaf(t, 3.0f * cf.w, 2.0f * cf.z), cf.y) * sxi;
            }
            const float ux = sh_perm ? vl.y : vl.x, uy = sh_perm ? vl.z : vl.y, uz = sh_perm ? vl.x : vl.z;
            float4 Yv[4];
            loup::edge_sh16(ux, uy, uz, vl.w, Yv);
            const float Y[16] = {Yv[0].x, Yv[0].y, Yv[0].z, Yv[0].w, Yv[1].x, Yv[1].y, Yv[1].z, Yv[1].w,
                                 Yv[2].x, Yv[2].y, Yv[2].z, Yv[2].w, Yv[3].x, Yv[3].y, Yv[3].z, Yv[3].w};
            float gcol = 0.0f, dY[16];
            #pragma unroll
            for (int m = 0; m < 16; ++m) {
                gcol = fmaf(G[m], Y[m], gcol);
                dY[m] = G[m] * v;
            }
            float er = gcol * dv + dDa * dv;
            dY[0] = 0.0f;
            const float dd = vl.w > loup::EPS_SH ? vl.w : loup::EPS_SH;
            const float inv_d = 1.0f / dd;
            const float x = ux * inv_d, y = uy * inv_d, z = uz * inv_d;
            float gx, gy, gz;
            sh16_grad(x, y, z, dY, gx, gy, gz);
            const float dot = x * gx + y * gy + z * gz;
            float fx = x * er + (gx - x * dot) * inv_d;
            float fy = y * er + (gy - y * dot) * inv_d;
            float fz = z * er + (gz - z * dot) * inv_d;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                fx += __shfl_xor_sync(0xffffffffu, fx, off);
                fy += __shfl_xor_sync(0xffffffffu, fy, off);
                fz += __shfl_xor_sync(0xffffffffu, fz, off);
            }
            if (lane == 0) {
                if (sh_perm) {
                    const float tx = fx, ty = fy, tz = fz;
                    fx = tz;
                    fy = tx;
                    fz = ty;
                }
                atomicAdd(edge_dvec + (int64_t)e * 3 + 0, fx);
                atomicAdd(edge_dvec + (int64_t)e * 3 + 1, fy);
                atomicAdd(edge_dvec + (int64_t)e * 3 + 2, fz);
            }
        }
    }
}

__global__ void zero_edge_kernel(const int* __restrict__ E_dev, float* __restrict__ p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 3 * *E_dev) p[i] = 0.0f;
}

}  // namespace

// ------------------------------------------------------------------------------------------------- load
GenericModel::~GenericModel() {
    free_system();
    free_model();
}

void GenericModel::free_model() {
    delete lr_;
    lr_ = nullptr;
    for (auto* l : layers_) delete l;
    layers_.clear();
    auto freep = [](auto*& p) {
        if (p) cudaFree(p);
        p = nullptr;
    };
    freep(d_embedding_);
    freep(d_zbl_rc_);
    freep(d_zbl_inv_a_);
    freep(d_zbl_pref_);
}

void GenericModel::load(const Bundle& b) {
    free_model();
    if (b.meta("arch") != "generic") throw std::runtime_error("not a generic bundle");
    if (b.meta_int("exporter_version") != 2)
        throw std::runtime_error("generic bundle exporter version " + b.meta("exporter_version") + ", expected 2 (re-export)");
    S_ = b.meta_int("S");
    n_layers_ = b.meta_int("n_layers");
    sh_perm_ = b.meta_int("sh_permute");
    if (S_ < 1 || S_ > 127 || n_layers_ < 1 || n_layers_ > 8) throw std::runtime_error("generic bundle: bad sizes");
    Z_ = b.meta_ints("Z");
    if ((int)Z_.size() != S_) throw std::runtime_error("generic bundle: Z does not match S");
    r_max_ = (float)b.f64("r_max", {1})[0];
    e0_ = b.f64("e0", {S_});
    std::vector<void*> tmp_owner;
    {
        const BundleEntry& e = b.entry("embedding");
        if (e.dims.size() != 2 || (int)e.dims[0] != S_) throw std::runtime_error("generic bundle: bad embedding");
        C0_ = (int)e.dims[1];
        d_embedding_ = upload(tmp_owner, b.f32("embedding", {S_, C0_}));
    }
    const int P = S_ * S_;
    const bool has_lr = b.has_meta("has_lr") && b.meta_int("has_lr");
    auto fold_mat = [](const float* A, int M, int K, const float* B, int N) {
        std::vector<float> out((size_t)M * N);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                double s = 0.0;
                for (int k = 0; k < K; ++k) s += (double)A[(size_t)m * K + k] * (double)B[(size_t)k * N + n];
                out[(size_t)m * N + n] = (float)s;
            }
        return out;
    };
    auto fold_species = [&](const float* A, int M, int K, const float* B_s, int Ss, int N) {
        std::vector<float> out((size_t)Ss * M * N);
        for (int s = 0; s < Ss; ++s) {
            auto sub = fold_mat(A, M, K, B_s + (size_t)s * K * N, N);
            memcpy(out.data() + (size_t)s * M * N, sub.data(), sub.size() * sizeof(float));
        }
        return out;
    };
    for (int t = 0; t < n_layers_; ++t) {
        auto* L = new GenericLayer();
        layers_.push_back(L);
        const std::string p = "L" + std::to_string(t) + ".";
        auto mi = [&](const char* k) { return b.meta_int(p + k); };
        L->kind = mi("kind");
        L->lmax_in = mi("lmax_in");
        L->lmax_x = mi("lmax_x");
        L->lmax_out = mi("lmax_out");
        L->C_in = mi("C_in");
        L->C_x = mi("C_x");
        L->C_msg = mi("C_msg");
        L->C_out = mi("C_out");
        L->n_paths = mi("n_paths");
        L->intervals = mi("intervals");
        L->has_sc = mi("has_sc");
        L->pair_sym = b.has_meta(p + "pair_sym") ? mi("pair_sym") : 0;
        L->first_hidden = b.has_meta(p + "first_hidden") ? mi("first_hidden") : 0;
        if (L->first_hidden) {
            L->J = mi("J");
            if (t != 0 || L->lmax_x != 0 || L->J < 3 || L->J > 256)
                throw std::runtime_error("generic bundle: bad first-layer hidden table");
        }
        L->readout = mi("readout");
        if (L->kind < 0 || L->kind > 4 || L->lmax_x > 2 || L->lmax_in > 3 || L->lmax_out > 2 || L->lmax_x > L->lmax_in)
            throw std::runtime_error("generic bundle layer " + std::to_string(t) + ": unsupported shape");
        for (int c : {L->C_in, L->C_x, L->C_msg, L->C_out})
            if (c < 32 || c > 512 || c % 32) throw std::runtime_error("generic bundle: channel counts must be 32..512, multiples of 32");
        if (L->C_msg != L->C_out) throw std::runtime_error("generic bundle: product channels differ from the hidden channels");
        if (t == 0 && (L->lmax_in != 0 || L->C_in != C0_)) throw std::runtime_error("generic bundle: layer 0 input");
        if (t > 0) {
            const GenericLayer* prev = layers_[t - 1];
            if (L->lmax_in != prev->lmax_out || L->C_in != prev->C_out)
                throw std::runtime_error("generic bundle: layer " + std::to_string(t) + " input does not match layer " +
                                         std::to_string(t - 1) + " output");
        }
        // paths per output degree (export_generic.conv_paths)
        static const int n_l3_tab[3][4] = {{1, 1, 1, 1}, {2, 3, 3, 2}, {3, 5, 5, 4}};
        for (int l = 0; l < 4; ++l) L->n_l3[l] = n_l3_tab[L->lmax_x][l];
        const int npaths = L->n_l3[0] + L->n_l3[1] + L->n_l3[2] + L->n_l3[3];
        if (npaths != L->n_paths) throw std::runtime_error("generic bundle: path count mismatch");
        const int Q = L->n_paths * L->C_x;
        L->Q1 = L->first_hidden ? L->J : Q + 1;
        L->skip_folded = (L->kind == 0 || L->kind == 2);
        L->ro_folded = (t == n_layers_ - 1 && !has_lr && L->lmax_out == 0 && L->readout > 0);
        auto& W = L->weights;
        for (int l = 0; l <= L->lmax_x; ++l)
            L->W_up[l] = upload(W, b.f32(p + "W_up_" + std::to_string(l), {L->C_in, L->C_x}));
        // message linear widths
        const bool nl = L->kind == 4;
        for (int l = 0; l < 4; ++l) L->lin_cols[l] = (nl && l == 0) ? 4 * L->C_msg : L->C_msg;
        if (L->skip_folded) {
            for (int l = 0; l < 4; ++l) {
                const auto wsk = b.f32(p + "W_skip_msg_" + std::to_string(l), {S_, L->C_msg, L->C_msg});
                const auto wlin = b.f32(p + "W_lin_" + std::to_string(l), {L->n_l3[l] * L->C_x, L->C_msg});
                L->W_lin[l] = upload(W, fold_species(wlin.data(), L->n_l3[l] * L->C_x, L->C_msg, wsk.data(), S_, L->C_msg));
                if (L->first_hidden) {
                    const auto c0 = b.f32(p + "first_C0_" + std::to_string(l), {S_ * (L->J - 1), L->C_msg});
                    L->C0[l] = upload(W, fold_species(c0.data(), S_ * (L->J - 1), L->C_msg, wsk.data(), S_, L->C_msg));
                }
            }
        } else {
            for (int l = 0; l < 4; ++l)
                L->W_lin[l] = upload(W, b.f32(p + "W_lin_" + std::to_string(l), {L->n_l3[l] * L->C_x, L->lin_cols[l]}));
            if (L->first_hidden)
                for (int l = 0; l < 4; ++l)
                    L->C0[l] = upload(W, b.f32(p + "first_C0_" + std::to_string(l), {S_ * (L->J - 1), L->lin_cols[l]}));
        }
        if (nl) {
            const std::vector<double> ab = b.f64(p + "alpha_beta", {2});
            L->alpha = (float)ab[0];
            L->beta = (float)ab[1];
            const std::vector<double> gc = b.f64(p + "gate_c", {2});
            L->gate_cs = (float)gc[0];
            L->gate_cg = (float)gc[1];
            if (b.meta_int(p + "gate_C") != L->C_msg) throw std::runtime_error("generic bundle: gate width");
            for (int l = 0; l <= L->lmax_x; ++l)
                L->W_res[l] = upload(W, b.f32(p + "W_res_" + std::to_string(l), {L->C_x, L->lin_cols[l]}));
            for (int l = 0; l < 4; ++l)
                L->W_2[l] = upload(W, b.f32(p + "W_2_" + std::to_string(l), {L->C_msg, L->C_msg}));
        }
        std::vector<float> h_wsc0;
        if (L->has_sc) {
            L->sc_species = mi("sc_species");
            const int Ss = L->sc_species ? S_ : 1;
            for (int l = 0; l <= std::min(L->lmax_out, L->lmax_in); ++l) {
                const std::string key = p + "W_sc_" + std::to_string(l);
                if (b.has(key)) {
                    auto wsc = b.f32(key, {Ss, L->C_in, L->C_out});
                    if (L->ro_folded && l == 0) h_wsc0 = std::move(wsc);
                    else L->W_sc[l] = upload(W, wsc);
                    L->sc_has[l] = 1;
                }
            }
        }
        {
            const std::vector<double> g = b.f64(p + "grid", {-1, 3});
            const int nb = (int)(g.size() / 3);
            if (nb < 1 || nb > 8) throw std::runtime_error("generic bundle: spline bands");
            int off = 0;
            for (int k = 0; k < 8; ++k) {
                if (k < nb) {
                    const double a0 = g[3 * k], b0 = g[3 * k + 1];
                    const int n = (int)g[3 * k + 2];
                    if (!(b0 > a0) || n < 1) throw std::runtime_error("generic bundle: bad spline band");
                    L->grid.starts[k] = (float)a0;
                    L->grid.ends[k] = (float)b0;
                    L->grid.inv_h[k] = (float)(n / (b0 - a0));
                    L->grid.offsets[k] = off;
                    L->grid.counts[k] = n;
                    off += n;
                } else {
                    L->grid.starts[k] = L->grid.ends[k] = L->grid.inv_h[k] = 0.0f;
                    L->grid.offsets[k] = L->grid.counts[k] = 0;
                }
            }
            L->grid.n_bands = nb;
            L->grid.total_intervals = off;
            if (off != L->intervals) throw std::runtime_error("generic bundle: spline intervals mismatch");
            const int Ptab = L->pair_sym ? S_ * (S_ + 1) / 2 : P;
            const std::vector<float> tab = b.f32(p + "spline", {Ptab, L->intervals, L->Q1, 4});
            std::vector<float4> host(tab.size() / 4);
            memcpy(host.data(), tab.data(), tab.size() * sizeof(float));
            L->table = upload(W, host);
        }
        // product
        {
            const std::string sig = b.meta(p + "prod_sig");
            for (const auto& k : kGProd)
                if (sig == k.sig) L->prod = &k;
            if (!L->prod) throw std::runtime_error("generic bundle: no compiled product kernel for signature " + sig);
            if (L->prod->lmax_out != L->lmax_out) throw std::runtime_error("generic bundle: product degree mismatch");
            L->prod_species = mi("prod_species");
            L->prod_segw = mi("prod_segments_w");
            if (L->prod_segw != L->prod->segments_w) throw std::runtime_error("generic bundle: product segments mismatch");
            L->prod_w = upload(W, b.f32(p + "prod_w", {L->prod_species ? S_ : 1, L->prod_segw, L->C_msg}));
            std::vector<float> h_wp0;
            for (int l = 0; l <= L->lmax_out; ++l) {
                auto wp = b.f32(p + "W_p_" + std::to_string(l), {L->C_out, L->C_out});
                if (L->ro_folded && l == 0) h_wp0 = std::move(wp);
                else L->W_p[l] = upload(W, wp);
            }
            // readout
            if (L->readout == 0) {
                L->ro_w = upload(W, b.f32(p + "ro_w", {L->C_out}));
            } else if (L->readout == 1 || L->readout == 2) {
                L->ro_H = mi("ro_H");
                const auto row1 = b.f32(p + "ro_W1", {L->C_out, L->ro_H});
                if (L->ro_folded) {
                    L->ro_W1 = upload(W, fold_mat(h_wp0.data(), L->C_out, L->C_out, row1.data(), L->ro_H));
                    if (L->sc_has[0]) {
                        const int Ss = L->sc_species ? S_ : 1;
                        std::vector<float> fsc((size_t)Ss * L->C_in * L->ro_H);
                        for (int s = 0; s < Ss; ++s) {
                            auto sub = fold_mat(h_wsc0.data() + (size_t)s * L->C_in * L->C_out, L->C_in, L->C_out,
                                                row1.data(), L->ro_H);
                            memcpy(fsc.data() + (size_t)s * L->C_in * L->ro_H, sub.data(), sub.size() * sizeof(float));
                        }
                        L->ro_W1_sc = upload(W, fsc);
                    }
                } else {
                    L->ro_W1 = upload(W, row1);
                }
                L->ro_b1 = upload(W, b.f32(p + "ro_b1", {L->ro_H}));
                L->ro_W2 = upload(W, b.f32(p + "ro_W2", {L->ro_H}));
                L->ro_act_c = (float)b.f64(p + "ro_act_c", {1})[0];
                if (L->readout == 2) {
                    L->ro_Wm = upload(W, b.f32(p + "ro_Wm", {L->ro_H, L->ro_H}));
                    L->ro_bm = upload(W, b.f32(p + "ro_bm", {L->ro_H}));
                }
            } else if (L->readout != -1) {
                throw std::runtime_error("generic bundle: unknown readout kind");
            }
        }
    }
    if (layers_.back()->readout < 0) throw std::runtime_error("generic bundle: the last layer has no readout");
    if (has_lr) {
        lr_ = new PolarLR();
        lr_->load(b, S_, n_layers_);
        for (auto* L : layers_)
            if (L->lmax_out != b.meta_int("LR.lmax") || L->C_out != b.meta_int("LR.C"))
                throw std::runtime_error("generic bundle: the long-range part needs every layer's output irreps equal");
    }
    has_zbl_ = b.meta_int("has_zbl");
    if (has_zbl_) {
        const std::vector<double> c = b.f64("zbl_c", {4});
        for (int i = 0; i < 4; ++i) zbl_c_[i] = (float)c[i];
        zbl_p_ = (float)b.f64("zbl_p", {1})[0];
        std::vector<float> rc = b.f32("zbl_r_cut", {P});
        zbl_rmax_ = *std::max_element(rc.begin(), rc.end());
        std::vector<void*> own;
        d_zbl_rc_ = upload(own, rc);
        d_zbl_inv_a_ = upload(own, b.f32("zbl_inv_a", {P}));
        d_zbl_pref_ = upload(own, b.f32("zbl_pref", {P}));
    }
}

// ------------------------------------------------------------------------------------------------- buffers
bool GenericModel::set_cell(const double* cell) { return lr_ ? lr_->set_cell(cell) : false; }

void GenericModel::free_system() {
    if (lr_) lr_->free_system();
    for (void* p : sys_allocs_) cudaFree(p);
    sys_allocs_.clear();
    if (d_zero_slab_) cudaFree(d_zero_slab_);
    d_zero_slab_ = nullptr;
    zero_slab_bytes_ = 0;
    if (d_edge_dvec_) cudaFree(d_edge_dvec_);
    d_edge_dvec_ = nullptr;
    E_cap_ = 0;
    N_ = 0;
    for (auto* L : layers_) {
        // keep configuration and weights; drop buffer pointers
        std::fill(std::begin(L->h_in), std::end(L->h_in), nullptr);
        std::fill(std::begin(L->dh_in), std::end(L->dh_in), nullptr);
        std::fill(std::begin(L->dx), std::end(L->dx), nullptr);
        std::fill(std::begin(L->dhout), std::end(L->dhout), nullptr);
        L->dD = nullptr;
    }
}

void GenericModel::alloc_system(int N) {
    free_system();
    auto alloc = [&](size_t floats) -> float* {
        float* p = nullptr;
        CUDA_CHECK_THROW(cudaMalloc(&p, std::max<size_t>(floats, 1) * sizeof(float)));
        sys_allocs_.push_back(p);
        return p;
    };
    const size_t n = (size_t)N;
    auto align256 = [](size_t b) { return (b + 255) & ~size_t(255); };
    // Pack every zero-initialized per-step gradient accumulator into one contiguous slab (1 cudaMemsetAsync/step).
    size_t zbytes = 0;
    for (int t = 0; t < n_layers_; ++t) {
        GenericLayer* L = layers_[t];
        if (!L->ro_folded)
            for (int l = 0; l <= L->lmax_out; ++l) zbytes += align256(n * dimL(l) * L->C_out * sizeof(float));
        if (t > 0)
            for (int l = 0; l <= L->lmax_x; ++l) zbytes += align256(n * dimL(l) * L->C_x * sizeof(float));
        if (L->kind >= 2) zbytes += align256(n * sizeof(float));
    }
    if (zbytes) {
        CUDA_CHECK_THROW(cudaMalloc(&d_zero_slab_, zbytes));
        zero_slab_bytes_ = zbytes;
    }
    char* zbase = static_cast<char*>(d_zero_slab_);
    size_t zoff = 0;
    auto zalloc = [&](size_t floats) -> float* {
        float* p = reinterpret_cast<float*>(zbase + zoff);
        zoff += align256(floats * sizeof(float));
        return p;
    };
    d_h0_ = alloc(n * C0_);
    for (int t = 0; t < n_layers_; ++t) {
        GenericLayer* L = layers_[t];
        for (int l = 0; l <= L->lmax_x; ++l) {
            L->x[l] = alloc(n * dimL(l) * L->C_x);
            L->dx[l] = (t > 0) ? zalloc(n * dimL(l) * L->C_x) : nullptr;
        }
        if (L->first_hidden)
            for (int l = 0; l < 4; ++l) L->S0[l] = alloc(n * dimL(l) * (size_t)S_ * (L->J - 1));
        for (int l = 0; l < 4; ++l) {
            L->mid[l] = L->first_hidden ? nullptr : alloc(n * dimL(l) * L->n_l3[l] * L->C_x);
            if (L->skip_folded) {
                L->lin[l] = nullptr;
                L->dlin[l] = nullptr;
                L->msgf[l] = alloc(n * dimL(l) * L->C_msg);
            } else {
                L->lin[l] = alloc(n * dimL(l) * L->lin_cols[l]);
                L->dlin[l] = alloc(n * dimL(l) * L->lin_cols[l]);
                L->msgf[l] = (L->kind == 1 || L->kind == 3) ? L->lin[l] : alloc(n * dimL(l) * L->C_msg);
            }
            L->dmsgf[l] = alloc(n * dimL(l) * L->C_msg);
            if (L->kind == 4) {
                L->gout[l] = alloc(n * dimL(l) * L->C_msg);
                L->dgout[l] = alloc(n * dimL(l) * L->C_msg);
                L->dpre[l] = (t > 0 && l <= L->lmax_x) ? alloc(n * dimL(l) * L->lin_cols[l]) : nullptr;
            }
        }
        if (L->kind == 4)
            for (int l = 0; l <= L->lmax_x; ++l) L->res[l] = alloc(n * dimL(l) * L->lin_cols[l]);
        L->D = alloc(n);
        L->dD = (L->kind >= 2) ? zalloc(n) : nullptr;
        for (int l = 0; l <= L->lmax_out; ++l) {
            L->y[l] = alloc(n * dimL(l) * L->C_out);
            L->dy[l] = alloc(n * dimL(l) * L->C_out);
            L->hout[l] = L->ro_folded ? nullptr : alloc(n * dimL(l) * L->C_out);
            L->dhout[l] = L->ro_folded ? nullptr : zalloc(n * dimL(l) * L->C_out);
        }
        if (L->ro_H) {
            L->hid = alloc(n * L->ro_H);
            L->hid2 = alloc(n * L->ro_H);
            L->dhid = alloc(n * L->ro_H);
            L->dhid2 = alloc(n * L->ro_H);
        }
    }
    for (int t = 0; t < n_layers_; ++t) {
        GenericLayer* L = layers_[t];
        if (t == 0) {
            L->h_in[0] = d_h0_;
        } else {
            for (int l = 0; l <= L->lmax_in; ++l) {
                L->h_in[l] = layers_[t - 1]->hout[l];
                L->dh_in[l] = layers_[t - 1]->dhout[l];
            }
        }
    }
    if (lr_) lr_->alloc_system(N);
    N_ = N;
}

void GenericModel::ensure_edges(int E_cap) {
    if (E_cap <= E_cap_) return;
    if (d_edge_dvec_) cudaFree(d_edge_dvec_);
    d_edge_dvec_ = nullptr;
    CUDA_CHECK_THROW(cudaMalloc(&d_edge_dvec_, (size_t)E_cap * 3 * sizeof(float)));
    E_cap_ = E_cap;
}

// ------------------------------------------------------------------------------------------------- force pass
namespace {

void launch_conv_fwd(int lx, dim3 grid, dim3 block, cudaStream_t st, int N, int C, int S, int I, int Q1,
                     const SplineGrid& sg, int perm, int psym, const int* rp, const int* snd, const float4* vl, const int* sp,
                     const float4* tab, float* const* x, float* const* mid, float* D) {
    if (lx == 0)
        gconv_fwd_L0<<<grid, block, 0, st>>>(N, C, S, I, Q1, sg, perm, psym, rp, snd, vl, sp, tab, x[0], mid[0], mid[1],
                                             mid[2], mid[3], D);
    else if (lx == 1)
        gconv_fwd_L1<<<grid, block, 0, st>>>(N, C, S, I, Q1, sg, perm, psym, rp, snd, vl, sp, tab, x[0], x[1], mid[0],
                                             mid[1], mid[2], mid[3], D);
    else
        gconv_fwd_L2<<<grid, block, 0, st>>>(N, C, S, I, Q1, sg, perm, psym, rp, snd, vl, sp, tab, x[0], x[1], x[2],
                                             mid[0], mid[1], mid[2], mid[3], D);
}

void launch_conv_bwd(int lx, dim3 grid, dim3 block, cudaStream_t st, int N, int C, int S, int I, int Q1,
                     const SplineGrid& sg, int perm, int psym, const int* rp, const int* snd, const float4* vl, const int* sp,
                     const float4* tab, float* const* x, float* const* dmid, const float* dD, float* const* dx,
                     int want_dx, float* edge_dvec) {
    if (lx == 0)
        gconv_bwd_L0<<<grid, block, 0, st>>>(N, C, S, I, Q1, sg, perm, psym, rp, snd, vl, sp, tab, x[0], dmid[0], dmid[1],
                                             dmid[2], dmid[3], dD, dx[0], want_dx, edge_dvec);
    else if (lx == 1)
        gconv_bwd_L1<<<grid, block, 0, st>>>(N, C, S, I, Q1, sg, perm, psym, rp, snd, vl, sp, tab, x[0], x[1], dmid[0],
                                             dmid[1], dmid[2], dmid[3], dD, dx[0], dx[1], want_dx, edge_dvec);
    else
        gconv_bwd_L2<<<grid, block, 0, st>>>(N, C, S, I, Q1, sg, perm, psym, rp, snd, vl, sp, tab, x[0], x[1], x[2],
                                             dmid[0], dmid[1], dmid[2], dmid[3], dD, dx[0], dx[1], dx[2], want_dx, edge_dvec);
}

void launch_prod(const GProdKernels* k, bool backward, cudaStream_t st, int N, int C, float* const* m, const float* D,
                 const float* w, int64_t w_stride, const int* sp, float* const* y_or_g, float* const* dm, float* dD) {
    std::vector<void*> args;
    args.push_back(&N);
    args.push_back(&C);
    for (int l = 0; l < 4; ++l) args.push_back((void*)&m[l]);
    args.push_back(&D);
    args.push_back(&w);
    args.push_back(&w_stride);
    args.push_back(&sp);
    for (int l = 0; l <= k->lmax_out; ++l) args.push_back((void*)&y_or_g[l]);
    if (backward) {
        for (int l = 0; l < 4; ++l) args.push_back((void*)&dm[l]);
        args.push_back(&dD);
    }
    CUDA_CHECK_THROW(cudaLaunchKernel(backward ? k->bwd : k->fwd, dim3(N, C / 32), dim3(32), args.data(), 0, st));
}

}  // namespace

void GenericModel::record(const GenericPassInputs& in) {
    const int N = in.N;
    cudaStream_t st = in.stream;
    const auto& start = *in.species_start;
    const auto& count = *in.species_count;
    const int S = S_;
    auto fork = [&] { if (in.gemm_fork) in.gemm_fork(); };
    auto lane = [&](int k) { if (in.gemm_lane) in.gemm_lane(k); };
    auto join = [&] { if (in.gemm_join) in.gemm_join(); };
    // species-grouped GEMM: rows of degree l for each species with its own weight block
    auto species_gemm = [&](bool transB, int l, int K, int Nc, const float* A, const float* W, int64_t w_stride,
                            float beta, float* C, int lane_base = -1) {
        const int d = dimL(l);
        for (int s = 0; s < S; ++s) {
            if (count[s] == 0) continue;
            if (lane_base >= 0) lane(lane_base + s);
            const int64_t row0 = (int64_t)start[s] * d;
            in.gemm(transB, count[s] * d, Nc, K, 1.0f, A + row0 * K, K, W + (w_stride ? s * w_stride : 0),
                    transB ? K : Nc, beta, C + row0 * Nc, Nc);
        }
    };

    // in.energy / in.forces / in.virial live in engine.cu's d_zero_slab and are already zeroed by record_force_pass.
    const bool has_edges = in.E_cap > 0;
    if (has_edges) zero_edge_kernel<<<(3 * in.E_cap + 255) / 256, 256, 0, st>>>(in.E_dev, d_edge_dvec_);
    if (zero_slab_bytes_) CUDA_CHECK_THROW(cudaMemsetAsync(d_zero_slab_, 0, zero_slab_bytes_, st));

    gather_rows_kernel<<<(unsigned)(((int64_t)N * C0_ + 255) / 256), 256, 0, st>>>(N, C0_, in.species, d_embedding_, d_h0_);

    // ---------------- forward
    for (int t = 0; t < n_layers_; ++t) {
        GenericLayer* L = layers_[t];
        const int Cx = L->C_x, Cm = L->C_msg, Co = L->C_out;
        if (!L->first_hidden || L->kind == 4) {  // the hidden-table layer 0 needs x only for the nonlinear residual
            if (L->lmax_x > 0) fork();
            for (int l = 0; l <= L->lmax_x; ++l) {
                if (L->lmax_x > 0) lane(l);
                in.gemm(false, N * dimL(l), Cx, L->C_in, 1.0f, L->h_in[l], L->C_in, L->W_up[l], Cx, 0.0f, L->x[l], Cx);
            }
            if (L->lmax_x > 0) join();
        }
        if (L->first_hidden) {
            const int Kh = S * (L->J - 1);
            first_fwd_kernel<<<N, ((L->J + 31) / 32) * 32, 0, st>>>(N, S, L->intervals, L->J, L->grid, sh_perm_,
                                                                   L->pair_sym, in.act_row_ptr, in.act_senders,
                                                                   in.act_vec_len, in.species, L->table, L->S0[0],
                                                                   L->S0[1], L->S0[2], L->S0[3], L->D);
            if (L->skip_folded) {
                fork();
                for (int l = 0; l < 4; ++l)
                    species_gemm(false, l, Kh, Cm, L->S0[l], L->C0[l], (int64_t)Kh * Cm, 0.0f, L->msgf[l], 4 * l);
                join();
            } else {
                fork();
                for (int l = 0; l < 4; ++l) {
                    lane(l);
                    in.gemm(false, N * dimL(l), L->lin_cols[l], Kh, 1.0f, L->S0[l], Kh, L->C0[l], L->lin_cols[l],
                            0.0f, L->lin[l], L->lin_cols[l]);
                }
                if (L->kind == 4) {
                    for (int l = 0; l <= L->lmax_x; ++l) {
                        lane(4 + l);
                        in.gemm(false, N * dimL(l), L->lin_cols[l], Cx, 1.0f, L->x[l], Cx, L->W_res[l],
                                L->lin_cols[l], 0.0f, L->res[l], L->lin_cols[l]);
                    }
                }
                join();
            }
        } else {
            launch_conv_fwd(L->lmax_x, dim3(N), dim3(Cx), st, N, Cx, S, L->intervals, L->Q1, L->grid, sh_perm_,
                            L->pair_sym, in.act_row_ptr, in.act_senders, in.act_vec_len, in.species, L->table, L->x,
                            L->mid, L->D);
            if (L->skip_folded) {
                fork();
                for (int l = 0; l < 4; ++l)
                    species_gemm(false, l, L->n_l3[l] * Cx, Cm, L->mid[l], L->W_lin[l],
                                 (int64_t)L->n_l3[l] * Cx * Cm, 0.0f, L->msgf[l], 4 * l);
                join();
            } else {
                fork();
                for (int l = 0; l < 4; ++l) {
                    lane(l);
                    in.gemm(false, N * dimL(l), L->lin_cols[l], L->n_l3[l] * Cx, 1.0f, L->mid[l], L->n_l3[l] * Cx,
                            L->W_lin[l], L->lin_cols[l], 0.0f, L->lin[l], L->lin_cols[l]);
                }
                if (L->kind == 4) {
                    for (int l = 0; l <= L->lmax_x; ++l) {
                        lane(4 + l);
                        in.gemm(false, N * dimL(l), L->lin_cols[l], Cx, 1.0f, L->x[l], Cx, L->W_res[l],
                                L->lin_cols[l], 0.0f, L->res[l], L->lin_cols[l]);
                    }
                }
                join();
            }
        }
        if (L->kind == 4) {
            gate_fwd_kernel<<<N, Cm, 0, st>>>(N, Cm, L->lmax_x, L->D, L->alpha, L->beta, L->gate_cs, L->gate_cg,
                                              L->lin[0], L->lin[1], L->lin[2], L->lin[3], L->res[0],
                                              L->lmax_x >= 1 ? L->res[1] : nullptr,
                                              L->lmax_x >= 2 ? L->res[2] : nullptr, L->gout[0], L->gout[1],
                                              L->gout[2], L->gout[3]);
            fork();
            for (int l = 0; l < 4; ++l) {
                lane(l);
                in.gemm(false, N * dimL(l), Cm, Cm, 1.0f, L->gout[l], Cm, L->W_2[l], Cm, 0.0f, L->msgf[l], Cm);
            }
            join();
        }
        const float* Dp = (L->kind == 2 || L->kind == 3) ? L->D : nullptr;
        launch_prod(L->prod, false, st, N, Cm, L->msgf, Dp, L->prod_w,
                    L->prod_species ? (int64_t)L->prod_segw * Cm : 0, in.species, L->y, nullptr, nullptr);
        if (!L->ro_folded) {
            if (L->lmax_out > 0) fork();
            for (int l = 0; l <= L->lmax_out; ++l) {
                if (L->lmax_out > 0) lane(l);
                in.gemm(false, N * dimL(l), Co, Co, 1.0f, L->y[l], Co, L->W_p[l], Co, 0.0f, L->hout[l], Co);
                if (L->sc_has[l]) {
                    if (L->sc_species)
                        species_gemm(false, l, L->C_in, Co, L->h_in[l], L->W_sc[l], (int64_t)L->C_in * Co, 1.0f,
                                     L->hout[l]);
                    else
                        in.gemm(false, N * dimL(l), Co, L->C_in, 1.0f, L->h_in[l], L->C_in, L->W_sc[l], Co, 1.0f,
                                L->hout[l], Co);
                }
            }
            if (L->lmax_out > 0) join();
        }
        // readout (energy) and its gradient seed into dhout[0] (or dhid directly when ro_folded)
        if (L->readout == 0) {
            ro_linear_kernel<<<(N + 7) / 8, 256, 0, st>>>(N, Co, L->hout[0], L->ro_w, L->dhout[0], in.energy);
        } else if (L->readout > 0) {
            const int H = L->ro_H;
            const unsigned gH = (unsigned)(((int64_t)N * H + 255) / 256);
            if (L->ro_folded) {
                in.gemm(false, N, H, Co, 1.0f, L->y[0], Co, L->ro_W1, H, 0.0f, L->hid, H);
                if (L->sc_has[0]) {
                    if (L->sc_species) {
                        fork();
                        species_gemm(false, 0, L->C_in, H, L->h_in[0], L->ro_W1_sc, (int64_t)L->C_in * H, 1.0f,
                                     L->hid, 0);
                        join();
                    } else {
                        in.gemm(false, N, H, L->C_in, 1.0f, L->h_in[0], L->C_in, L->ro_W1_sc, H, 1.0f, L->hid, H);
                    }
                }
            } else {
                in.gemm(false, N, H, Co, 1.0f, L->hout[0], Co, L->ro_W1, H, 0.0f, L->hid, H);
            }
            if (L->readout == 1) {
                ro_mlp_kernel<<<gH, 256, 0, st>>>(N, H, L->ro_act_c, L->hid, L->ro_b1, L->ro_W2, nullptr, L->dhid,
                                                  in.energy, 1);
            } else {
                // stage 1: hid -> act1 (pre kept in dhid2 as scratch), stage 2 on hid2 = act1 @ Wm
                ro_mlp_kernel<<<gH, 256, 0, st>>>(N, H, L->ro_act_c, L->hid, L->ro_b1, nullptr, L->dhid2, nullptr,
                                                  nullptr, 0);
                in.gemm(false, N, H, H, 1.0f, L->hid, H, L->ro_Wm, H, 0.0f, L->hid2, H);
                ro_mlp_kernel<<<gH, 256, 0, st>>>(N, H, L->ro_act_c, L->hid2, L->ro_bm, L->ro_W2, nullptr, L->dhid,
                                                  in.energy, 1);
                // dhid = (dhid @ Wm^T) * c silu'(pre1)
                in.gemm(true, N, H, H, 1.0f, L->dhid, H, L->ro_Wm, H, 0.0f, L->hid2, H);
                ro_act_bwd_kernel<<<gH, 256, 0, st>>>((int64_t)N * H, L->ro_act_c, L->dhid2, L->hid2);
                CUDA_CHECK_THROW(cudaMemcpyAsync(L->dhid, L->hid2, (size_t)N * H * sizeof(float),
                                                 cudaMemcpyDeviceToDevice, st));
            }
            if (!L->ro_folded)
                in.gemm(true, N, Co, H, 1.0f, L->dhid, H, L->ro_W1, H, 1.0f, L->dhout[0], Co);
        }
    }
    if (has_zbl_ && has_edges) {
        zbl_kernel<<<(in.E_cap + 255) / 256, 256, 0, st>>>(in.E_dev, S, in.act_vec_len, in.act_senders,
                                                          in.act_receivers, in.species, d_zbl_rc_, d_zbl_inv_a_,
                                                          d_zbl_pref_, zbl_c_[0], zbl_c_[1], zbl_c_[2], zbl_c_[3],
                                                          zbl_p_, d_edge_dvec_, in.energy);
    }

    // ---------------- long range: energy, k-space forces and its gradient seeds into every layer's dhout
    if (lr_) {
        std::vector<std::vector<float*>> hs, dhs;
        for (auto* L : layers_) {
            hs.emplace_back(L->hout, L->hout + L->lmax_out + 1);
            dhs.emplace_back(L->dhout, L->dhout + L->lmax_out + 1);
        }
        lr_->record(st, N, in.gemm, in.gemm_fork, in.gemm_lane, in.gemm_join, in.species, in.pos64, hs, dhs,
                    in.energy, in.forces, in.compute_stress, in.virial);
    }

    // ---------------- backward
    for (int t = n_layers_ - 1; t >= 0; --t) {
        GenericLayer* L = layers_[t];
        const int Cx = L->C_x, Cm = L->C_msg, Co = L->C_out;
        if (L->ro_folded) {
            const int H = L->ro_H;
            const bool fork_sc = L->sc_has[0] && t > 0;
            if (fork_sc) fork();
            if (fork_sc) lane(0);
            in.gemm(true, N, Co, H, 1.0f, L->dhid, H, L->ro_W1, H, 0.0f, L->dy[0], Co);
            if (fork_sc) {
                if (L->sc_species)
                    species_gemm(true, 0, H, L->C_in, L->dhid, L->ro_W1_sc, (int64_t)L->C_in * H, 1.0f,
                                 L->dh_in[0], 1);
                else {
                    lane(1);
                    in.gemm(true, N, L->C_in, H, 1.0f, L->dhid, H, L->ro_W1_sc, H, 1.0f, L->dh_in[0], L->C_in);
                }
                join();
            }
        } else {
            const bool fork_p = L->lmax_out > 0 || (L->has_sc && t > 0);
            if (fork_p) fork();
            for (int l = 0; l <= L->lmax_out; ++l) {
                if (fork_p) lane(l);
                in.gemm(true, N * dimL(l), Co, Co, 1.0f, L->dhout[l], Co, L->W_p[l], Co, 0.0f, L->dy[l], Co);
                if (L->sc_has[l] && t > 0) {
                    if (L->sc_species)
                        species_gemm(true, l, Co, L->C_in, L->dhout[l], L->W_sc[l], (int64_t)L->C_in * Co, 1.0f,
                                     L->dh_in[l], 4);
                    else {
                        lane(4 + l);
                        in.gemm(true, N * dimL(l), L->C_in, Co, 1.0f, L->dhout[l], Co, L->W_sc[l], Co, 1.0f,
                                L->dh_in[l], L->C_in);
                    }
                }
            }
            if (fork_p) join();
        }
        const float* Dp = (L->kind == 2 || L->kind == 3) ? L->D : nullptr;
        float* dDp = (L->kind == 2 || L->kind == 3) ? L->dD : nullptr;
        // product backward: gradient of the product input
        float* dm[4];
        for (int l = 0; l < 4; ++l) dm[l] = (L->kind == 1 || L->kind == 3) ? L->dlin[l] : L->dmsgf[l];
        launch_prod(L->prod, true, st, N, Cm, L->msgf, Dp, L->prod_w,
                    L->prod_species ? (int64_t)L->prod_segw * Cm : 0, in.species, L->dy, dm, dDp);
        if (L->kind == 4) {
            fork();
            for (int l = 0; l < 4; ++l) {
                lane(l);
                in.gemm(true, N * dimL(l), Cm, Cm, 1.0f, L->dmsgf[l], Cm, L->W_2[l], Cm, 0.0f, L->dgout[l], Cm);
            }
            join();
            gate_bwd_kernel<<<N, Cm, 0, st>>>(N, Cm, L->lmax_x, L->D, L->alpha, L->beta, L->gate_cs, L->gate_cg,
                                              L->lin[0], L->lin[1], L->lin[2], L->lin[3], L->res[0],
                                              L->lmax_x >= 1 ? L->res[1] : nullptr,
                                              L->lmax_x >= 2 ? L->res[2] : nullptr, L->dgout[0], L->dgout[1],
                                              L->dgout[2], L->dgout[3], t > 0 ? L->dpre[0] : nullptr,
                                              (t > 0 && L->lmax_x >= 1) ? L->dpre[1] : nullptr,
                                              (t > 0 && L->lmax_x >= 2) ? L->dpre[2] : nullptr, L->dlin[0], L->dlin[1],
                                              L->dlin[2], L->dlin[3], L->dD);
        }
        const bool density = L->kind >= 2;
        if (L->first_hidden) {
            // dS0 (aliases S0) = dlin @ C0^T (or dmsgf @ C0_folded^T when skip_folded)
            const int Kh = S * (L->J - 1);
            if (L->skip_folded) {
                fork();
                for (int l = 0; l < 4; ++l)
                    species_gemm(true, l, Cm, Kh, L->dmsgf[l], L->C0[l], (int64_t)Kh * Cm, 0.0f, L->S0[l], 4 * l);
                join();
            } else {
                fork();
                for (int l = 0; l < 4; ++l) {
                    lane(l);
                    in.gemm(true, N * dimL(l), Kh, L->lin_cols[l], 1.0f, L->dlin[l], L->lin_cols[l], L->C0[l],
                            L->lin_cols[l], 0.0f, L->S0[l], Kh);
                }
                if (L->kind == 4 && t > 0) {
                    for (int l = 0; l <= L->lmax_x; ++l) {
                        lane(4 + l);
                        in.gemm(true, N * dimL(l), Cx, L->lin_cols[l], 1.0f, L->dpre[l], L->lin_cols[l], L->W_res[l],
                                L->lin_cols[l], 1.0f, L->dx[l], Cx);
                    }
                }
                join();
            }
            first_bwd_kernel<<<N, ((L->J + 31) / 32) * 32, 0, st>>>(N, S, L->intervals, L->J, L->grid, sh_perm_,
                                                                   L->pair_sym, in.act_row_ptr, in.act_senders,
                                                                   in.act_vec_len, in.species, L->table, L->S0[0],
                                                                   L->S0[1], L->S0[2], L->S0[3],
                                                                   density ? L->dD : nullptr, d_edge_dvec_);
            continue;
        }
        // dmid (aliases mid: the forward values are dead) = dlin @ W_lin^T (or dmsgf @ W_lin_folded^T)
        if (L->skip_folded) {
            fork();
            for (int l = 0; l < 4; ++l)
                species_gemm(true, l, Cm, L->n_l3[l] * Cx, L->dmsgf[l], L->W_lin[l],
                             (int64_t)L->n_l3[l] * Cx * Cm, 0.0f, L->mid[l], 4 * l);
            join();
        } else {
            fork();
            for (int l = 0; l < 4; ++l) {
                lane(l);
                in.gemm(true, N * dimL(l), L->n_l3[l] * Cx, L->lin_cols[l], 1.0f, L->dlin[l], L->lin_cols[l],
                        L->W_lin[l], L->lin_cols[l], 0.0f, L->mid[l], L->n_l3[l] * Cx);
            }
            if (L->kind == 4 && t > 0) {
                for (int l = 0; l <= L->lmax_x; ++l) {
                    lane(4 + l);
                    in.gemm(true, N * dimL(l), Cx, L->lin_cols[l], 1.0f, L->dpre[l], L->lin_cols[l], L->W_res[l],
                            L->lin_cols[l], 1.0f, L->dx[l], Cx);
                }
            }
            join();
        }
        launch_conv_bwd(L->lmax_x, dim3(N), dim3(Cx), st, N, Cx, S, L->intervals, L->Q1, L->grid, sh_perm_,
                        L->pair_sym, in.act_row_ptr, in.act_senders, in.act_vec_len, in.species, L->table, L->x, L->mid,
                        density ? L->dD : nullptr, L->dx, t > 0 ? 1 : 0, d_edge_dvec_);
        if (t > 0) {
            if (L->lmax_x > 0) fork();
            for (int l = 0; l <= L->lmax_x; ++l) {
                if (L->lmax_x > 0) lane(l);
                in.gemm(true, N * dimL(l), L->C_in, Cx, 1.0f, L->dx[l], Cx, L->W_up[l], Cx, 1.0f, L->dh_in[l], L->C_in);
            }
            if (L->lmax_x > 0) join();
        }
    }
    if (has_edges)
        assemble_forces_kernel<<<(in.E_cap + 255) / 256, 256, 0, st>>>(in.E_dev, in.act_vec_len, in.act_senders,
                                                                      in.act_receivers, d_edge_dvec_, in.compute_stress,
                                                                      in.forces, in.virial);
}

}  // namespace loup
