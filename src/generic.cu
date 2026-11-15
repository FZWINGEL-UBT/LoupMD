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
    int first_hidden = 0, J = 0, K0 = 0;  // layer 0 outer sum over J columns; K0 = S * (J - 1) padded to 16
    float* C0[4] = {};                    // [K0, lin_cols] (per species when skip_folded)
    float* S0[4] = {};                    // [N, 2l+1, K0] (the backward reuses it for dS0)
    int has_sc = 0, sc_species = 0, readout = -1, ro_H = 0, prod_species = 0, prod_segw = 0;
    int skip_folded = 0, ro_folded = 0;
    int lin_cols[4] = {0, 0, 0, 0};
    int sc_has[3] = {0, 0, 0};
    SplineGrid grid{};
    float alpha = 1.0f, beta = 0.0f, gate_cs = 1.0f, gate_cg = 1.0f, ro_act_c = 1.0f;
    const GProdKernels* prod = nullptr;
    // weights (device)
    float4* table = nullptr;
    uint2* packed_table = nullptr;
    float4* density_table = nullptr;
    unsigned cubic_bias_shift = 0;
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
    // S57 skip-chain fold (GenericModel::setup_chain_fold): this layer's hout is never formed; per degree l, y[l]
    // feeds every linear consumer of hout downstream through one composed weight per consumer. dest_layer >= 0: that
    // layer's x[l] (its linear_up input, through the skips in between); -1: the last layer's readout hidden (hid).
    // Wh (layer 0 only): the same consumers of the layer-0 skip h_in[0] W_sc[0]; Th instead with the S59 species
    // tables: [S, cols] = embedding W_sc[0] M, gathered per atom (always the destination's first write).
    struct FoldTarget {
        int dest_layer = 0, cols = 0;
        float* Wy = nullptr;  // [C_out, cols]
        float* Wh = nullptr;  // [C_in, cols] or nullptr
        float* Th = nullptr;  // [S, cols] or nullptr
    };
    std::vector<FoldTarget> fold[3];
    // S59 (layer 0): [S, lin_cols[0]] species table of the residual (embedding W_up[0] W_res[0]), or nullptr
    float* res0_table = nullptr;

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

// ZBL per edge (ordered pair tables), float32 as in MACE's float32 model (the gate's yardstick; FP64 runs at 1/64
// rate on GeForce parts). Callers skip edges beyond the largest ZBL cutoff (ZblParams::r_max, 0 without ZBL)
// before reading the tables. Returns false outside the pair's cutoff; otherwise en = the pair energy, dE/dvec = g vec.
struct ZblParams {
    const float *rc = nullptr, *inv_a = nullptr, *pref = nullptr;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0, p = 5.0f, r_max = 0.0f;
};
__device__ __forceinline__ bool zbl_pair(const ZblParams& z, int p, float r, float& en, float& g) {
    const float rc = z.rc[p];
    if (!(r > 0.0f && r < rc)) return false;
    const float inv_a = z.inv_a[p], pref = z.pref[p], zp = z.p;
    const float u = r * inv_a;
    const float e0 = expf(-3.2f * u), e1 = expf(-0.9423f * u), e2 = expf(-0.4028f * u), e3 = expf(-0.2016f * u);
    const float phi = z.c0 * e0 + z.c1 * e1 + z.c2 * e2 + z.c3 * e3;
    const float dphi = -inv_a * (3.2f * z.c0 * e0 + 0.9423f * z.c1 * e1 + 0.4028f * z.c2 * e2 + 0.2016f * z.c3 * e3);
    const float inv_r = 1.0f / r;
    const float vz = pref * inv_r * phi;
    const float dv = pref * inv_r * (dphi - phi * inv_r);
    const float sr = r / rc;
    const float s2 = sr * sr;
    const float sp = zp == 5.0f ? s2 * s2 * sr : powf(sr, zp);
    const float env = 1.0f - ((zp + 1.0f) * (zp + 2.0f) * 0.5f) * sp + (zp * (zp + 2.0f)) * sp * sr -
                      (zp * (zp + 1.0f) * 0.5f) * sp * sr * sr;
    const float denv = -(zp * (zp + 1.0f) * (zp + 2.0f) * 0.5f) * (sp / sr) * (1.0f - sr) * (1.0f - sr) / rc;
    en = vz * env;
    g = (dv * env + vz * denv) * inv_r;
    return true;
}

// Per edge, the last pass of the backward: ZBL, then forces and virial from the total dE/dvec
// (vec = pos[rcv] - pos[snd] + shift). ZBL used to be a separate pass over all edges before the backward.
template <bool ZBL, bool STRESS>
__global__ void assemble_forces_kernel(const int* __restrict__ E_dev, int S, const float4* __restrict__ vec_len,
                                       const int* __restrict__ senders, const int* __restrict__ receivers,
                                       const int* __restrict__ species, ZblParams z,
                                       const float* __restrict__ edge_dvec, double* __restrict__ forces,
                                       double* __restrict__ virial, double* __restrict__ energy) {
    const int E = *E_dev;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    double v[6] = {0, 0, 0, 0, 0, 0};
    double en = 0.0;
    if (e < E) {
        float fxf = edge_dvec[(int64_t)e * 3 + 0], fyf = edge_dvec[(int64_t)e * 3 + 1],
              fzf = edge_dvec[(int64_t)e * 3 + 2];
        const int s = senders[e], r = receivers[e];
        float4 vl = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (ZBL || STRESS) vl = vec_len[e];
        float zen, g;
        if (ZBL && vl.w < z.r_max && zbl_pair(z, species[s] * S + species[r], vl.w, zen, g)) {
            en = zen;
            fxf += g * vl.x;
            fyf += g * vl.y;
            fzf += g * vl.z;
        }
        const double fx = fxf, fy = fyf, fz = fzf;
        atomicAdd(forces + r * 3 + 0, -fx);
        atomicAdd(forces + r * 3 + 1, -fy);
        atomicAdd(forces + r * 3 + 2, -fz);
        atomicAdd(forces + s * 3 + 0, fx);
        atomicAdd(forces + s * 3 + 1, fy);
        atomicAdd(forces + s * 3 + 2, fz);
        if (STRESS) {
            v[0] = vl.x * fx;
            v[1] = 0.5 * (vl.x * fy + vl.y * fx);
            v[2] = 0.5 * (vl.x * fz + vl.z * fx);
            v[3] = vl.y * fy;
            v[4] = 0.5 * (vl.y * fz + vl.z * fy);
            v[5] = vl.z * fz;
        }
    }
    if (ZBL) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) en += __shfl_xor_sync(0xffffffffu, en, off);
        if ((threadIdx.x & 31) == 0 && en != 0.0) atomicAdd(energy, en);
    }
    if (STRESS) {
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
__global__ void __launch_bounds__(256) first_fwd_kernel(
    int N, int S, int I, int J, int K0, SplineGrid grid, int sh_perm, int pair_sym,
    const int* __restrict__ row_ptr, const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species, const float4* __restrict__ table,
    float* __restrict__ s0, float* __restrict__ s1, float* __restrict__ s2, float* __restrict__ s3, float* __restrict__ D) {
    // s_ee alternates by species parity: with no edge of species s there is no barrier between this iteration's read
    // and thread 0's write for s + 1, but the next barrier bounds thread 0's lead to one slot.
    __shared__ int s_iv[32], s_ee[2];
    __shared__ float s_t[32];
    __shared__ __align__(16) float s_sh[32 * 16];
    const int a = blockIdx.x, tid = threadIdx.x;
    if (a >= N) return;
    const int Jp = J - 1, K = S * Jp, sa = species[a];
    const int e_end = row_ptr[a + 1];
    int e_cur = row_ptr[a];
    float dsum = 0.0f;
    for (int s = 0; s < S; ++s) {
        if (tid < 32) {
            int p = e_cur;
            while (p < e_end) {
                const int idx = p + tid;
                const bool ok = (idx < e_end) && (species[senders[idx]] == s);
                const unsigned mask = __ballot_sync(0xffffffffu, ok);
                if (mask != 0xffffffffu) { p += __ffs(~mask) - 1; break; }
                p += 32;
            }
            if (p > e_end) p = e_end;
            if (tid == 0) s_ee[s & 1] = p;
        }
        __syncthreads();
        const int e_s_end = s_ee[s & 1];
        float acc[16];
        #pragma unroll
        for (int m = 0; m < 16; ++m) acc[m] = 0.0f;
        const int pr = pair_sym ? loup::sym_pair(s, sa, S) : s * S + sa;
        for (int base = e_cur; base < e_s_end; base += 32) {
            const int cnt = (e_s_end - base < 32) ? (e_s_end - base) : 32;
            if (tid < cnt) {
                const float4 vl = __ldg(vec_len + base + tid);
                int iv; float t, sxi;
                locate_spline(grid, vl.w, iv, t, sxi);
                s_iv[tid] = iv;
                s_t[tid] = t;
            }
            if (tid >= 32 && tid - 32 < cnt) {
                const float4 vl = __ldg(vec_len + base + (tid - 32));
                if (sh_perm) loup::edge_sh16(vl.y, vl.z, vl.x, vl.w, reinterpret_cast<float4*>(s_sh + (tid - 32) * 16));
                else         loup::edge_sh16(vl.x, vl.y, vl.z, vl.w, reinterpret_cast<float4*>(s_sh + (tid - 32) * 16));
            }
            __syncthreads();
            if (tid < J) {
                for (int k = 0; k < cnt; ++k) {
                    const float4 cf = __ldg(table + ((int64_t)pr * I + s_iv[k]) * J + tid);
                    const float t = s_t[k];
                    const float v = fmaf(t, fmaf(t, fmaf(t, cf.w, cf.z), cf.y), cf.x);
                    if (tid == Jp) {
                        dsum += v;
                    } else {
                        const float* sh = s_sh + k * 16;
                        #pragma unroll
                        for (int m = 0; m < 16; ++m) acc[m] = fmaf(v, sh[m], acc[m]);
                    }
                }
            }
            __syncthreads();
        }
        if (tid < Jp) {
            s0[(int64_t)a * K0 + s * Jp + tid] = acc[0];
            #pragma unroll
            for (int m = 0; m < 3; ++m) s1[((int64_t)a * 3 + m) * K0 + s * Jp + tid] = acc[1 + m];
            #pragma unroll
            for (int m = 0; m < 5; ++m) s2[((int64_t)a * 5 + m) * K0 + s * Jp + tid] = acc[4 + m];
            #pragma unroll
            for (int m = 0; m < 7; ++m) s3[((int64_t)a * 7 + m) * K0 + s * Jp + tid] = acc[9 + m];
        }
        e_cur = e_s_end;
    }
    if (tid < K0 - K) {
        s0[(int64_t)a * K0 + K + tid] = 0.0f;
        #pragma unroll
        for (int m = 0; m < 3; ++m) s1[((int64_t)a * 3 + m) * K0 + K + tid] = 0.0f;
        #pragma unroll
        for (int m = 0; m < 5; ++m) s2[((int64_t)a * 5 + m) * K0 + K + tid] = 0.0f;
        #pragma unroll
        for (int m = 0; m < 7; ++m) s3[((int64_t)a * 7 + m) * K0 + K + tid] = 0.0f;
    }
    if (tid == Jp) D[a] = dsum;
}

// STAGE_G: the receiver's gradient rows G[m][s Jp + j] (16 x K floats, the same for all its edges) are staged once in
// shared memory, m innermost in float4 chunks swizzled by (k >> 1) & 3 so a warp's column-consecutive float4 reads
// are bank-conflict free: 4 shared loads per column instead of 16 global ones. Without it (K too large for shared
// memory) the rows are read from global memory per edge.
template <bool STAGE_G>
__global__ void __launch_bounds__(256) first_bwd_kernel(
    int N, int S, int I, int J, int K0, SplineGrid grid, int sh_perm, int pair_sym,
    const int* __restrict__ row_ptr, const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species, const float4* __restrict__ table,
    const float* __restrict__ g0, const float* __restrict__ g1, const float* __restrict__ g2, const float* __restrict__ g3,
    const float* __restrict__ dD, float* __restrict__ edge_dvec) {
    __shared__ float s_dy[8][32];
    extern __shared__ float4 s_G4[];
    const int a = blockIdx.x, tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, n_warps = blockDim.x >> 5;
    if (a >= N) return;
    const int Jp = J - 1, K = S * Jp, sa = species[a];
    const float dDa = dD ? __ldg(dD + a) : 0.0f;
    const int eb = row_ptr[a], ee = row_ptr[a + 1];
    if (STAGE_G) {
        float* s_G = reinterpret_cast<float*>(s_G4);
        for (int idx = tid; idx < 16 * K; idx += blockDim.x) {
            const int m = idx / K, k = idx - m * K;
            const float v = m == 0 ? __ldg(g0 + (int64_t)a * K0 + k)
                          : m < 4 ? __ldg(g1 + ((int64_t)a * 3 + m - 1) * K0 + k)
                          : m < 9 ? __ldg(g2 + ((int64_t)a * 5 + m - 4) * K0 + k)
                                  : __ldg(g3 + ((int64_t)a * 7 + m - 9) * K0 + k);
            s_G[(k * 4 + ((m >> 2) ^ ((k >> 1) & 3))) * 4 + (m & 3)] = v;
        }
        __syncthreads();
    }
    // Each half-warp owns one edge and strides the J columns by 16: J = 66 takes 5 passes per two edges instead of 3
    // per edge (the third with 2 of 32 lanes), and the serial tail runs on lanes 0 and 16 together. Edges are taken in
    // pairs so the loop and its shuffles stay warp-convergent; an odd last edge leaves its partner half idle.
    const int half = lane >> 4, hl = lane & 15;
    for (int base = eb + 2 * wid; base < ee; base += 2 * n_warps) {
        const bool valid = base + half < ee;
        const int e = valid ? base + half : base;
        const float4 vl = __ldg(vec_len + e);
        const int s = species[senders[e]];
        const int pr = pair_sym ? loup::sym_pair(s, sa, S) : s * S + sa;
        int iv; float t, sxi;
        locate_spline(grid, vl.w, iv, t, sxi);
        const float ux = sh_perm ? vl.y : vl.x, uy = sh_perm ? vl.z : vl.y, uz = sh_perm ? vl.x : vl.z;
        float4 Yv[4];
        loup::edge_sh16(ux, uy, uz, vl.w, Yv);
        const float Y[16] = {Yv[0].x, Yv[0].y, Yv[0].z, Yv[0].w, Yv[1].x, Yv[1].y, Yv[1].z, Yv[1].w,
                             Yv[2].x, Yv[2].y, Yv[2].z, Yv[2].w, Yv[3].x, Yv[3].y, Yv[3].z, Yv[3].w};
        float dY[16];
        #pragma unroll
        for (int m = 0; m < 16; ++m) dY[m] = 0.0f;
        float er = 0.0f;
        for (int j = hl; j < J; j += 16) {
            const float4 cf = __ldg(table + ((int64_t)pr * I + iv) * J + j);
            const float v = fmaf(t, fmaf(t, fmaf(t, cf.w, cf.z), cf.y), cf.x);
            const float dv = fmaf(t, fmaf(t, 3.0f * cf.w, 2.0f * cf.z), cf.y) * sxi;
            if (j == Jp) {
                er = fmaf(dDa, dv, er);
            } else {
                float G[16];
                if (STAGE_G) {
                    const int k = s * Jp + j, sw = (k >> 1) & 3;
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const float4 c = s_G4[k * 4 + (q ^ sw)];
                        G[4 * q] = c.x; G[4 * q + 1] = c.y; G[4 * q + 2] = c.z; G[4 * q + 3] = c.w;
                    }
                } else {
                    G[0] = __ldg(g0 + (int64_t)a * K0 + s * Jp + j);
                    #pragma unroll
                    for (int m = 0; m < 3; ++m) G[1 + m] = __ldg(g1 + ((int64_t)a * 3 + m) * K0 + s * Jp + j);
                    #pragma unroll
                    for (int m = 0; m < 5; ++m) G[4 + m] = __ldg(g2 + ((int64_t)a * 5 + m) * K0 + s * Jp + j);
                    #pragma unroll
                    for (int m = 0; m < 7; ++m) G[9 + m] = __ldg(g3 + ((int64_t)a * 7 + m) * K0 + s * Jp + j);
                }
                float gcol = 0.0f;
                #pragma unroll
                for (int m = 0; m < 16; ++m) {
                    gcol = fmaf(G[m], Y[m], gcol);
                    dY[m] = fmaf(G[m], v, dY[m]);
                }
                er = fmaf(gcol, dv, er);
            }
        }
        // Fold dY[0..15] over each half's 16 lanes (xor 1, 2, 4, 8 stay inside a half): lane hl ends with dY[hl].
        dY[0] = 0.0f;
        float v8[8];
        const bool p0 = (lane & 1) != 0;
        #pragma unroll
        for (int k = 0; k < 8; ++k) v8[k] = (p0 ? dY[2 * k + 1] : dY[2 * k]) + __shfl_xor_sync(0xffffffffu, p0 ? dY[2 * k] : dY[2 * k + 1], 1);
        float v4[4];
        const bool p1 = (lane & 2) != 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k) v4[k] = (p1 ? v8[2 * k + 1] : v8[2 * k]) + __shfl_xor_sync(0xffffffffu, p1 ? v8[2 * k] : v8[2 * k + 1], 2);
        float v2[2];
        const bool p2 = (lane & 4) != 0;
        #pragma unroll
        for (int k = 0; k < 2; ++k) v2[k] = (p2 ? v4[2 * k + 1] : v4[2 * k]) + __shfl_xor_sync(0xffffffffu, p2 ? v4[2 * k] : v4[2 * k + 1], 4);
        const bool p3 = (lane & 8) != 0;
        const float v0 = (p3 ? v2[1] : v2[0]) + __shfl_xor_sync(0xffffffffu, p3 ? v2[0] : v2[1], 8);
        s_dy[wid][lane] = v0;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) er += __shfl_xor_sync(0xffffffffu, er, off);
        __syncwarp();
        if (hl == 0 && valid) {
            const float dd = vl.w > loup::EPS_SH ? vl.w : loup::EPS_SH;
            const float inv_d = 1.0f / dd;
            const float x = ux * inv_d, y = uy * inv_d, z = uz * inv_d;
            float gx, gy, gz;
            ::sh16_grad(x, y, z, &s_dy[wid][half * 16], gx, gy, gz);
            const float dot = x * gx + y * gy + z * gz;
            float fx = x * er + (gx - x * dot) * inv_d;
            float fy = y * er + (gy - y * dot) * inv_d;
            float fz = z * er + (gz - z * dot) * inv_d;
            if (sh_perm) { const float tx = fx, ty = fy, tz = fz; fx = tz; fy = tx; fz = ty; }
            edge_dvec[(int64_t)e * 3 + 0] += fx;
            edge_dvec[(int64_t)e * 3 + 1] += fy;
            edge_dvec[(int64_t)e * 3 + 2] += fz;
        }
        __syncwarp();
    }
}

// Edge-parallel layer-0 backward + ZBL + force/virial assembly (tabled_outer_sum0_backward_kernel design):
// 4 warps x 32 edges/warp, COLS columns/lane (2 for Jp <= 64, 3 for Jp <= 96); caches dS0 across consecutive
// active edges with the same (rcv, snd_sp), folds SH gradients in s_edge[4][32][20], then all 32 lanes assemble
// forces, ZBL and virial in parallel. Latency-bound (serial shuffle folds): on sm_120, COLS = 2 is capped at 80
// registers (6 blocks / SM, no spills; 92-96 registers uncapped). COLS = 3 would spill under the cap, and other
// architectures (not measured) keep the uncapped kernel.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
#define LOUP_FIRST_BWD_EDGE_MINB2 6
#else
#define LOUP_FIRST_BWD_EDGE_MINB2 0
#endif
template <int COLS, bool HAS_ZBL, bool STRESS>
__global__ void __launch_bounds__(128, COLS == 2 ? LOUP_FIRST_BWD_EDGE_MINB2 : 0) first_bwd_edge_kernel(
    const int* __restrict__ E_dev, int S, int I, int J, int K0, SplineGrid grid, int sh_perm, int pair_sym,
    const int* __restrict__ senders, const int* __restrict__ receivers, const float4* __restrict__ vec_len,
    const int* __restrict__ species, const float4* __restrict__ table,
    const float* __restrict__ g0, const float* __restrict__ g1, const float* __restrict__ g2, const float* __restrict__ g3,
    const float* __restrict__ dD, const float* __restrict__ edge_dvec, ZblParams zbl,
    double* __restrict__ energy, double* __restrict__ forces, double* __restrict__ virial) {
    constexpr int CHUNK = 32;
    const int E = *E_dev;
    const int tid = threadIdx.x, warp = tid >> 5, warp_id = blockIdx.x * 4 + warp, lane = tid & 31;
    const int start = warp_id * CHUNK;
    const int Jp = J - 1;

    __shared__ __align__(16) float s_edge[4][CHUNK][20];
    double v00 = 0.0, v01 = 0.0, v02 = 0.0, v11 = 0.0, v12 = 0.0, v22 = 0.0;
    double e_zbl_thread = 0.0;

    if (start < E) {
        const int stop = (start + CHUNK < E) ? (start + CHUNK) : E;
        const int cnt = stop - start;
        int m_iv = 0, m_pr = 0, m_rcv = -1, m_snd = -1, m_sp = -1, m_rcv_sp = -1;
        float m_t = 0.0f, m_sxi = 0.0f;
        if (lane < cnt) {
            const int e = start + lane;
            const float4 vl = __ldg(vec_len + e);
            const float ux = sh_perm ? vl.y : vl.x, uy = sh_perm ? vl.z : vl.y, uz = sh_perm ? vl.x : vl.z;
            loup::edge_sh16(ux, uy, uz, vl.w, reinterpret_cast<float4*>(s_edge[warp][lane]));
            locate_spline(grid, vl.w, m_iv, m_t, m_sxi);
            m_rcv = receivers[e];
            m_snd = senders[e];
            m_sp = species[m_snd];
            m_rcv_sp = species[m_rcv];
            m_pr = pair_sym ? loup::sym_pair(m_sp, m_rcv_sp, S) : m_sp * S + m_rcv_sp;
        }
        __syncwarp();

        float blk[COLS][16];
        #pragma unroll
        for (int c = 0; c < COLS; ++c)
            #pragma unroll
            for (int d = 0; d < 16; ++d) blk[c][d] = 0.0f;
        float dD_r = 0.0f;
        int cur_rcv = -1, cur_snd_sp = -1;

        for (int k = 0; k < cnt; ++k) {
            const int rcv = __shfl_sync(0xffffffffu, m_rcv, k);
            const int snd_sp = __shfl_sync(0xffffffffu, m_sp, k);
            if (rcv != cur_rcv || snd_sp != cur_snd_sp) {
                cur_rcv = rcv;
                cur_snd_sp = snd_sp;
                if (COLS == 2 && (Jp & 1) == 0 && (K0 & 1) == 0) {
                    const int off = snd_sp * Jp + 2 * lane;
                    const bool ok = 2 * lane + 1 < Jp;
                    auto ld2 = [&](const float* base, int64_t row, int d) {
                        if (ok) {
                            const float2 v = __ldg(reinterpret_cast<const float2*>(base + row * K0 + off));
                            blk[0][d] = v.x;
                            blk[1][d] = v.y;
                        } else {
                            blk[0][d] = 2 * lane < Jp ? __ldg(base + row * K0 + off) : 0.0f;
                            blk[1][d] = 0.0f;
                        }
                    };
                    ld2(g0, rcv, 0);
                    #pragma unroll
                    for (int d = 0; d < 3; ++d) ld2(g1, (int64_t)rcv * 3 + d, 1 + d);
                    #pragma unroll
                    for (int d = 0; d < 5; ++d) ld2(g2, (int64_t)rcv * 5 + d, 4 + d);
                    #pragma unroll
                    for (int d = 0; d < 7; ++d) ld2(g3, (int64_t)rcv * 7 + d, 9 + d);
                } else {
                    #pragma unroll
                    for (int c = 0; c < COLS; ++c) {
                        const int col = (COLS == 2) ? (2 * lane + c) : (c * 32 + lane);
                        if (col < Jp) {
                            const int off = snd_sp * Jp + col;
                            blk[c][0] = __ldg(g0 + (int64_t)rcv * K0 + off);
                            #pragma unroll
                            for (int d = 0; d < 3; ++d) blk[c][1 + d] = __ldg(g1 + ((int64_t)rcv * 3 + d) * K0 + off);
                            #pragma unroll
                            for (int d = 0; d < 5; ++d) blk[c][4 + d] = __ldg(g2 + ((int64_t)rcv * 5 + d) * K0 + off);
                            #pragma unroll
                            for (int d = 0; d < 7; ++d) blk[c][9 + d] = __ldg(g3 + ((int64_t)rcv * 7 + d) * K0 + off);
                        }
                    }
                }
                if (dD) dD_r = __ldg(dD + rcv);
            }

            const int iv = __shfl_sync(0xffffffffu, m_iv, k);
            const int pr = __shfl_sync(0xffffffffu, m_pr, k);
            const float t = __shfl_sync(0xffffffffu, m_t, k);
            const float sxi = __shfl_sync(0xffffffffu, m_sxi, k);
            const float4* tb = table + ((int64_t)pr * I + iv) * J;
            float pa[COLS], pd[COLS];
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                const int col = (COLS == 2) ? (2 * lane + c) : (c * 32 + lane);
                if (col < Jp) {
                    const float4 cf = __ldg(tb + col);
                    pa[c] = fmaf(t, fmaf(t, fmaf(t, cf.w, cf.z), cf.y), cf.x);
                    pd[c] = fmaf(t, fmaf(t, 3.0f * cf.w, 2.0f * cf.z), cf.y) * sxi;
                } else {
                    pa[c] = 0.0f;
                    pd[c] = 0.0f;
                }
            }

            const float4* shv = reinterpret_cast<const float4*>(s_edge[warp][k]);
            const float4 Y0 = shv[0], Y1 = shv[1], Y2 = shv[2], Y3 = shv[3];
            const float shk[16] = {Y0.x, Y0.y, Y0.z, Y0.w, Y1.x, Y1.y, Y1.z, Y1.w,
                                   Y2.x, Y2.y, Y2.z, Y2.w, Y3.x, Y3.y, Y3.z, Y3.w};
            float grad_a[COLS];
            #pragma unroll
            for (int c = 0; c < COLS; ++c) grad_a[c] = 0.0f;
            float gy[16];
            #pragma unroll
            for (int d = 0; d < 16; ++d) {
                const float sd = shk[d];
                float gsum = 0.0f;
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    grad_a[c] = fmaf(blk[c][d], sd, grad_a[c]);
                    gsum = fmaf(blk[c][d], pa[c], gsum);
                }
                gy[d] = gsum;
            }

            // 16-shuffle fold of gy[0..15] across 32 lanes -> lane 1..15 holds dsh[lane]
            float g8[8];
            const bool odd = (lane & 1) != 0;
            #pragma unroll
            for (int q = 0; q < 8; ++q)
                g8[q] = (odd ? gy[2 * q + 1] : gy[2 * q]) + __shfl_xor_sync(0xffffffffu, odd ? gy[2 * q] : gy[2 * q + 1], 1);
            float g4[4];
            const bool h2 = (lane & 2) != 0;
            #pragma unroll
            for (int q = 0; q < 4; ++q)
                g4[q] = (h2 ? g8[2 * q + 1] : g8[2 * q]) + __shfl_xor_sync(0xffffffffu, h2 ? g8[2 * q] : g8[2 * q + 1], 2);
            float g2[2];
            const bool h4 = (lane & 4) != 0;
            #pragma unroll
            for (int q = 0; q < 2; ++q)
                g2[q] = (h4 ? g4[2 * q + 1] : g4[2 * q]) + __shfl_xor_sync(0xffffffffu, h4 ? g4[2 * q] : g4[2 * q + 1], 4);
            const bool h8 = (lane & 8) != 0;
            float g1 = (h8 ? g2[1] : g2[0]) + __shfl_xor_sync(0xffffffffu, h8 ? g2[0] : g2[1], 8);
            g1 += __shfl_xor_sync(0xffffffffu, g1, 16);
            if (lane >= 1 && lane < 16) s_edge[warp][k][lane] = g1;

            float dr_v = 0.0f;
            #pragma unroll
            for (int c = 0; c < COLS; ++c) dr_v = fmaf(grad_a[c], pd[c], dr_v);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dr_v += __shfl_xor_sync(0xffffffffu, dr_v, off);
            if (lane == 0) {
                if (dD && dD_r != 0.0f) {
                    const float4 cd = __ldg(tb + Jp);
                    const float dvd = fmaf(t, fmaf(t, 3.0f * cd.w, 2.0f * cd.z), cd.y) * sxi;
                    dr_v = fmaf(dD_r, dvd, dr_v);
                }
                s_edge[warp][k][0] = dr_v;
            }
        }
        __syncwarp();

        if (lane < cnt) {
            const int e = start + lane;
            const float4 vl = __ldg(vec_len + e);
            const float dd = vl.w > loup::EPS_SH ? vl.w : loup::EPS_SH;
            const float inv_d = 1.0f / dd;
            const float ux = sh_perm ? vl.y : vl.x, uy = sh_perm ? vl.z : vl.y, uz = sh_perm ? vl.x : vl.z;
            const float x = ux * inv_d, y = uy * inv_d, z = uz * inv_d;
            float gx, gy, gz;
            ::sh16_grad(x, y, z, s_edge[warp][lane], gx, gy, gz);
            const float er = s_edge[warp][lane][0];
            const float dot = x * gx + y * gy + z * gz;
            float fx = x * er + (gx - x * dot) * inv_d;
            float fy = y * er + (gy - y * dot) * inv_d;
            float fz = z * er + (gz - z * dot) * inv_d;
            if (sh_perm) { const float tx = fx, ty = fy, tz = fz; fx = tz; fy = tx; fz = ty; }
            fx += edge_dvec[(int64_t)e * 3 + 0];
            fy += edge_dvec[(int64_t)e * 3 + 1];
            fz += edge_dvec[(int64_t)e * 3 + 2];
            if (HAS_ZBL && vl.w < zbl.r_max) {
                float zen, zg;
                if (zbl_pair(zbl, m_sp * S + m_rcv_sp, vl.w, zen, zg)) {
                    e_zbl_thread += zen;
                    fx += zg * vl.x;
                    fy += zg * vl.y;
                    fz += zg * vl.z;
                }
            }
            atomicAdd(forces + m_rcv * 3 + 0, -(double)fx);
            atomicAdd(forces + m_rcv * 3 + 1, -(double)fy);
            atomicAdd(forces + m_rcv * 3 + 2, -(double)fz);
            atomicAdd(forces + m_snd * 3 + 0, (double)fx);
            atomicAdd(forces + m_snd * 3 + 1, (double)fy);
            atomicAdd(forces + m_snd * 3 + 2, (double)fz);
            if (STRESS) {
                v00 = (double)vl.x * (double)fx;
                v01 = 0.5 * ((double)vl.x * (double)fy + (double)vl.y * (double)fx);
                v02 = 0.5 * ((double)vl.x * (double)fz + (double)vl.z * (double)fx);
                v11 = (double)vl.y * (double)fy;
                v12 = 0.5 * ((double)vl.y * (double)fz + (double)vl.z * (double)fy);
                v22 = (double)vl.z * (double)fz;
            }
        }
    }
    if (HAS_ZBL && __any_sync(0xffffffffu, e_zbl_thread != 0.0)) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) e_zbl_thread += __shfl_xor_sync(0xffffffffu, e_zbl_thread, off);
        if (lane == 0) atomicAdd(energy, e_zbl_thread);
    }
    if (STRESS) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v00 += __shfl_xor_sync(0xffffffffu, v00, off);
            v01 += __shfl_xor_sync(0xffffffffu, v01, off);
            v02 += __shfl_xor_sync(0xffffffffu, v02, off);
            v11 += __shfl_xor_sync(0xffffffffu, v11, off);
            v12 += __shfl_xor_sync(0xffffffffu, v12, off);
            v22 += __shfl_xor_sync(0xffffffffu, v22, off);
        }
        __shared__ double s_v[4][6];
        if (lane == 0) {
            s_v[warp][0] = v00; s_v[warp][1] = v01; s_v[warp][2] = v02;
            s_v[warp][3] = v11; s_v[warp][4] = v12; s_v[warp][5] = v22;
        }
        __syncthreads();
        if (tid < 6) {
            double sum = 0.0;
            #pragma unroll
            for (int w = 0; w < 4; ++w) sum += s_v[w][tid];
            if (sum != 0.0) atomicAdd(virial + tid, sum);
        }
    }
}
__global__ void zero_edge_kernel(const int* __restrict__ E_dev, float* __restrict__ p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 3 * *E_dev) p[i] = 0.0f;
}

// C[M, N] = A[M, K] B[K, N] in double, one thread per output: load-time weight composition (S57 / S59).
__global__ void dmm_kernel(int M, int N, int K, const double* __restrict__ A, const double* __restrict__ B,
                           double* __restrict__ C) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (n >= N || m >= M) return;
    double s = 0.0;
    for (int k = 0; k < K; ++k) s = fma(A[(size_t)m * K + k], B[(size_t)k * N + n], s);
    C[(size_t)m * N + n] = s;
}

__global__ void d2f_kernel(int64_t n, const double* __restrict__ in, float* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
}

// Load-time weight composition in double on the device (S57 skip-chain fold, S59 species tables). Temporaries live
// until the object goes; float results belong to `owner` (a layer's weight list).
struct DevCompose {
    std::vector<void*> tmp;
    ~DevCompose() {
        cudaDeviceSynchronize();  // the kernels reading the temporaries
        for (void* q : tmp) cudaFree(q);
    }
    const double* dup(const std::vector<float>& h) {
        std::vector<double> d(h.begin(), h.end());
        double* q = nullptr;
        CUDA_CHECK_THROW(cudaMalloc(&q, std::max<size_t>(d.size(), 1) * sizeof(double)));
        tmp.push_back(q);
        CUDA_CHECK_THROW(cudaMemcpy(q, d.data(), d.size() * sizeof(double), cudaMemcpyHostToDevice));
        return q;
    }
    const double* mm(const double* A, int M, int K, const double* B, int N) {
        double* C = nullptr;
        CUDA_CHECK_THROW(cudaMalloc(&C, (size_t)M * N * sizeof(double)));
        tmp.push_back(C);
        dmm_kernel<<<dim3((unsigned)((N + 127) / 128), (unsigned)M), 128>>>(M, N, K, A, B, C);
        CUDA_CHECK_THROW(cudaGetLastError());
        return C;
    }
    float* to_f32(const double* D, int64_t n, std::vector<void*>& owner) {
        float* f = nullptr;
        CUDA_CHECK_THROW(cudaMalloc(&f, std::max<int64_t>(n, 1) * sizeof(float)));
        owner.push_back(f);
        d2f_kernel<<<(unsigned)((n + 255) / 256), 256>>>(n, D, f);
        CUDA_CHECK_THROW(cudaGetLastError());
        return f;
    }
};

// The load-time folds (S57, S59) were measured on compute capability 12.0 only; other GPUs keep the plain path.
bool device_is_cc120() {
    int dev = 0, major = 0, minor = 0;
    CUDA_CHECK_THROW(cudaGetDevice(&dev));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    return major == 12 && minor == 0;
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
        // Hidden widths (C_in / C_msg / C_out: GEMMs, gates with one thread per channel, products on 128-channel
        // blocks) up to 1024; the conv width C_x up to 512 (the conv kernels size threads and shared memory by it),
        // except in the hidden-table layer 0, which runs no conv (C_x feeds only its GEMMs).
        for (int c : {L->C_in, L->C_msg, L->C_out})
            if (c < 32 || c > 1024 || c % 32) throw std::runtime_error("generic bundle: hidden channel counts must be 32..1024, multiples of 32");
        if (L->C_x < 32 || L->C_x > (L->first_hidden ? 1024 : 512) || L->C_x % 32)
            throw std::runtime_error("generic bundle: conv channel count must be 32..512, multiples of 32");
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
        std::vector<float> c0_pad[4];
        if (L->first_hidden) {
            const int Jp = L->J - 1;
            std::vector<float> c0_raw[4];
            bool bias_zero = L->J > 2;
            for (int l = 0; l < 4; ++l) {
                const int nc = L->lin_cols[l];
                c0_raw[l] = b.f32(p + "first_C0_" + std::to_string(l), {S_ * Jp, nc});
                for (int s = 0; s < S_ && bias_zero; ++s)
                    for (int j = 0; j < nc; ++j)
                        if (c0_raw[l][((size_t)s * Jp + Jp - 1) * nc + j] != 0.0f) bias_zero = false;
            }
            const int Jp_new = bias_zero ? Jp - 1 : Jp;
            if (bias_zero) L->J = Jp_new + 1;
            L->K0 = ((S_ * Jp_new + 15) / 16) * 16;
            for (int l = 0; l < 4; ++l) {
                const int nc = L->lin_cols[l];
                c0_pad[l].assign((size_t)L->K0 * nc, 0.0f);
                for (int s = 0; s < S_; ++s)
                    memcpy(&c0_pad[l][(size_t)s * Jp_new * nc], &c0_raw[l][(size_t)s * Jp * nc],
                           (size_t)Jp_new * nc * sizeof(float));
            }
        }
        if (L->skip_folded) {
            for (int l = 0; l < 4; ++l) {
                const auto wsk = b.f32(p + "W_skip_msg_" + std::to_string(l), {S_, L->C_msg, L->C_msg});
                const auto wlin = b.f32(p + "W_lin_" + std::to_string(l), {L->n_l3[l] * L->C_x, L->C_msg});
                L->W_lin[l] = upload(W, fold_species(wlin.data(), L->n_l3[l] * L->C_x, L->C_msg, wsk.data(), S_, L->C_msg));
                if (L->first_hidden)
                    L->C0[l] = upload(W, fold_species(c0_pad[l].data(), L->K0, L->C_msg, wsk.data(), S_, L->C_msg));
            }
        } else {
            for (int l = 0; l < 4; ++l)
                L->W_lin[l] = upload(W, b.f32(p + "W_lin_" + std::to_string(l), {L->n_l3[l] * L->C_x, L->lin_cols[l]}));
            if (L->first_hidden)
                for (int l = 0; l < 4; ++l) L->C0[l] = upload(W, c0_pad[l]);
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
                    // locate_spline assumes ascending, contiguous bands that end at r_max
                    if (k > 0 && std::abs(a0 - g[3 * (k - 1) + 1]) > 1e-6)
                        throw std::runtime_error("generic bundle: spline bands not contiguous");
                    if (k == nb - 1 && std::abs(b0 - (double)r_max_) > 1e-4)
                        throw std::runtime_error("generic bundle: last spline band does not end at r_max");
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
            if (L->first_hidden) {
                const size_t PI = (size_t)Ptab * L->intervals;
                const float4* src = reinterpret_cast<const float4*>(tab.data());
                std::vector<float4> host(PI * L->J);
                if (L->J == L->Q1) {
                    memcpy(host.data(), src, PI * L->J * sizeof(float4));
                } else {
                    for (size_t pi = 0; pi < PI; ++pi) {
                        memcpy(&host[pi * L->J], &src[pi * L->Q1], (size_t)(L->J - 1) * sizeof(float4));
                        host[pi * L->J + L->J - 1] = src[pi * L->Q1 + L->Q1 - 1];
                    }
                }
                L->table = upload(W, host);
            } else {
                const int Q = L->n_paths * L->C_x;
                const size_t PI = (size_t)Ptab * L->intervals;
                const size_t sp_n = PI * Q;
                const float4* src = reinterpret_cast<const float4*>(tab.data());
                if (L->kind >= 2) {
                    std::vector<float4> dens(PI);
                    for (size_t pi = 0; pi < PI; ++pi) dens[pi] = src[pi * L->Q1 + Q];
                    L->density_table = upload(W, dens);
                }
                int emin = 1000, emax = -1000;
                std::vector<int> exps(sp_n, 0);
                for (size_t pi = 0; pi < PI; ++pi) {
                    for (int q = 0; q < Q; ++q) {
                        const float4 cf = src[pi * L->Q1 + q];
                        double a0 = (double)cf.x + 0.5 * (double)cf.y + 0.25 * (double)cf.z + 0.125 * (double)cf.w;
                        double a1 = (double)cf.y + (double)cf.z + 0.75 * (double)cf.w;
                        double a2 = (double)cf.z + 1.5 * (double)cf.w;
                        double a3 = (double)cf.w;
                        double top = std::max({std::abs(a0), std::abs(a1) * 256.0, std::abs(a2) * 16384.0,
                                               std::abs(a3) * 524288.0});
                        const size_t idx = pi * Q + q;
                        if (top > 0.0) {
                            int e = (int)std::floor(std::log2(top)) + 1;
                            exps[idx] = e;
                            if (e < emin) emin = e;
                            if (e > emax) emax = e;
                        } else {
                            exps[idx] = 1000000;
                        }
                    }
                }
                if (emin > emax) emin = emax = 0;
                if (emax - emin > 31) emin = emax - 31;
                if (emin < -108 || emax > 125)
                    throw std::runtime_error("generic spline coefficients outside CUBIC_PACKED8 exponent range");
                L->cubic_bias_shift = (unsigned)(128 + emin) << 23;
                std::vector<uint2> host_cubic(sp_n);
                auto qval = [](double v, int ei, int sh, int nb) -> uint32_t {
                    int64_t lim = (int64_t(1) << (nb - 1)) - 1;
                    double step = std::ldexp(1.0, ei - sh - nb + 1);
                    int64_t q = std::llround(v / step);
                    if (q > lim) q = lim;
                    else if (q < -lim) q = -lim;
                    return (uint32_t)(q + (int64_t(1) << (nb - 1)));
                };
                auto rotr = [](uint32_t v, int k) { return (v >> k) | (v << (32 - k)); };
                auto deq = [](uint32_t u, int ei, int sh, int nb) {
                    return (double)((int64_t)u - (int64_t(1) << (nb - 1))) * std::ldexp(1.0, ei - sh - nb + 1);
                };
                // Packing error per column (path x channel), exact maxima over every interval and pair, value and
                // d/dr, over the whole table and over the intervals reaching the physical region (PackErrTracker).
                std::vector<double> ih(L->intervals);
                std::vector<double> r0(L->intervals);
                for (int k = 0; k < L->grid.n_bands; ++k)
                    for (int i = 0; i < L->grid.counts[k]; ++i) {
                        ih[L->grid.offsets[k] + i] = L->grid.inv_h[k];
                        r0[L->grid.offsets[k] + i] = L->grid.starts[k] + i / (double)L->grid.inv_h[k];
                    }
                loup::PackErrTracker perr(Q);
                for (size_t pi = 0; pi < PI; ++pi) {
                    for (int q = 0; q < Q; ++q) {
                        const float4 cf = src[pi * L->Q1 + q];
                        const size_t idx = pi * Q + q;
                        double a0 = (double)cf.x + 0.5 * (double)cf.y + 0.25 * (double)cf.z + 0.125 * (double)cf.w;
                        double a1 = (double)cf.y + (double)cf.z + 0.75 * (double)cf.w;
                        double a2 = (double)cf.z + 1.5 * (double)cf.w;
                        double a3 = (double)cf.w;
                        int ei = (exps[idx] == 1000000 || exps[idx] < emin) ? emin : exps[idx];
                        uint32_t u0 = qval(a0, ei, 0, 20);
                        uint32_t u1 = qval(a1, ei, 8, 19);
                        uint32_t u2 = qval(a2, ei, 14, 12);
                        uint32_t u3 = qval(a3, ei, 19, 8);
                        uint32_t eb_code = (uint32_t)(ei - emin);
                        host_cubic[idx].x = (u0 << 3) | rotr(u2 << 11, 20);
                        host_cubic[idx].y = (u1 << 4) | rotr(u3 << 15, 19) | (eb_code << 23);
                        const double av[4] = {a0, a1, a2, a3};
                        const double dv[4] = {deq(u0, ei, 0, 20) - a0, deq(u1, ei, 8, 19) - a1,
                                              deq(u2, ei, 14, 12) - a2, deq(u3, ei, 19, 8) - a3};
                        const size_t iv = pi % (size_t)L->intervals;
                        perr.add(q, r0[iv], ih[iv], av, dv);
                    }
                }
                const double rel = perr.rel();
                if (std::getenv("LOUP_DEBUG_PACK")) fprintf(stderr, "pack L%d: %.3e\n", t, rel);
                if (rel > loup::kPackedTableMaxRelErr)
                    throw std::runtime_error("generic bundle layer " + std::to_string(t) + ": 8-byte table error " +
                                             std::to_string(rel) + " of the column maximum exceeds " +
                                             std::to_string(loup::kPackedTableMaxRelErr));
                L->packed_table = upload(W, host_cubic);
            }
        }
        // product
        {
            const std::string sig = b.meta(p + "prod_sig");
            L->prod = find_gprod(sig.c_str());
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
    setup_species_tables(b);
    setup_chain_fold(b);
}

// S59 species tables. Layer 0's input h_in[0] is the species embedding row, so every product of it alone is a
// per-species constant: the gated block's residual res[0] = (h_in[0] W_up[0]) W_res[0] (with x[0] used for nothing
// else in the hidden-table layer) and, under the S57 fold, the skip terms h_in[0] Wh. They become [S, cols] tables
// composed in double at load and gathered per atom: no x / res GEMMs in layer 0 and no Wh GEMMs. Layer 0 takes no
// gradient into h_in, so the backward is unchanged. Not with the long-range block: POLAR-1-L measured no win in the
// tables-first order (three ABBA pairs), so the POLAR models keep the GEMMs. sm_120 only; LOUP_SPECIES_TABLES=0 keeps
// the GEMMs for testing.
void GenericModel::setup_species_tables(const Bundle& b) {
    species_tables_ = false;
    GenericLayer* L = layers_.empty() ? nullptr : layers_[0];
    if (lr_ || !L || !L->first_hidden || L->kind != 4 || L->skip_folded || L->lmax_x != 0 || !device_is_cc120()) return;
    if (const char* e = std::getenv("LOUP_SPECIES_TABLES"); e && e[0] == '0') return;
    DevCompose dc;
    const double* E = dc.dup(b.f32("embedding", {S_, C0_}));
    const double* Wup = dc.dup(b.f32("L0.W_up_0", {L->C_in, L->C_x}));
    const double* Wres = dc.dup(b.f32("L0.W_res_0", {L->C_x, L->lin_cols[0]}));
    L->res0_table = dc.to_f32(dc.mm(dc.mm(E, S_, C0_, Wup, L->C_x), S_, L->C_x, Wres, L->lin_cols[0]),
                              (int64_t)S_ * L->lin_cols[0], L->weights);
    species_tables_ = true;
}

// S57 skip-chain fold. A non-last layer's output hout_t = y_t W_p_t + h_in_t W_sc_t is read only through linear maps:
// the next layer's linear_up (x_{t+1} = hout_t W_up_{t+1}) and skip (hout_{t+1} += hout_t W_sc_{t+1}), down to the
// last layer, whose skip already sits in its readout (hid += h_in ro_W1_sc, ro_W1_sc = W_sc ro_W1). So hout_t is never
// formed: per degree l, y_t[l] (and for layer 0 the species-only h_in[0]) multiplies one composed weight per consumer,
// W_p_t[l] W_sc_{t+1}[l] ... W_up_s[l] (or ... ro_W1), composed in double at load; the backward forms dy_t from the
// consumers' gradients through the same weights. Removes the full-width W_p / W_sc products of every non-last layer in
// both directions (and their hout / dhout buffers). Applies when the model has no long range, every layer is a gated
// (nonlinear) block, no non-last layer has a readout, no skip is per species, and the last layer's readout is folded.
// With the S59 species tables, layer 0's skip terms are [S, cols] tables (E W_sc_0 M) instead of Wh GEMMs.
// sm_120 only (where it was measured); LOUP_CHAIN_FOLD=0 keeps the plain path for testing.
void GenericModel::setup_chain_fold(const Bundle& b) {
    chain_fold_ = false;
    const int nL = n_layers_;
    if (lr_ || nL < 2 || !device_is_cc120()) return;
    if (const char* e = std::getenv("LOUP_CHAIN_FOLD"); e && e[0] == '0') return;
    const GenericLayer* last = layers_.back();
    if (!last->ro_folded || last->readout < 1) return;
    for (int t = 0; t < nL; ++t) {
        const GenericLayer* L = layers_[t];
        if (L->kind != 4 || (L->has_sc && L->sc_species)) return;
        if (t < nL - 1) {
            if (L->readout != -1) return;
            for (int l = 0; l <= L->lmax_out; ++l)
                if (l > layers_[t + 1]->lmax_x) return;  // every degree of y_t reaches the next linear_up
        }
    }
    DevCompose dc;
    struct Target {
        int dest, cols;
        const double* M;  // [rows, cols]: hout_t's (or h_in's) consumer map
    };
    std::vector<Target> next[3];  // the targets of hout_{t+1}
    std::vector<GenericLayer::FoldTarget> out[8][3];
    for (int t = nL - 2; t >= 0; --t) {
        GenericLayer* L = layers_[t];
        const GenericLayer* Ln = layers_[t + 1];
        const std::string pn = "L" + std::to_string(t + 1) + ".";
        std::vector<Target> cur[3];
        for (int l = 0; l <= L->lmax_out; ++l) {
            cur[l].push_back({t + 1, Ln->C_x, dc.dup(b.f32(pn + "W_up_" + std::to_string(l), {Ln->C_in, Ln->C_x}))});
            if (l <= 2 && Ln->sc_has[l]) {
                const double* Wsc = dc.dup(b.f32(pn + "W_sc_" + std::to_string(l), {1, Ln->C_in, Ln->C_out}));
                if (t + 1 == nL - 1) {
                    if (l == 0) {  // the last layer's skip, into its readout hidden
                        const double* W1 = dc.dup(b.f32(pn + "ro_W1", {Ln->C_out, Ln->ro_H}));
                        cur[0].push_back({-1, Ln->ro_H, dc.mm(Wsc, Ln->C_in, Ln->C_out, W1, Ln->ro_H)});
                    }
                } else {
                    for (const Target& d : next[l])
                        cur[l].push_back({d.dest, d.cols, dc.mm(Wsc, Ln->C_in, Ln->C_out, d.M, d.cols)});
                }
            }
        }
        const std::string p = "L" + std::to_string(t) + ".";
        const double* E = (t == 0 && species_tables_) ? dc.dup(b.f32("embedding", {S_, C0_})) : nullptr;
        for (int l = 0; l <= L->lmax_out; ++l) {
            const double* Wp = dc.dup(b.f32(p + "W_p_" + std::to_string(l), {L->C_out, L->C_out}));
            const double* Wsc0 = (t == 0 && l == 0 && L->sc_has[0])
                                     ? dc.dup(b.f32(p + "W_sc_0", {1, L->C_in, L->C_out})) : nullptr;
            for (const Target& d : cur[l]) {
                GenericLayer::FoldTarget ft;
                ft.dest_layer = d.dest;
                ft.cols = d.cols;
                ft.Wy = dc.to_f32(dc.mm(Wp, L->C_out, L->C_out, d.M, d.cols), (int64_t)L->C_out * d.cols, L->weights);
                if (Wsc0) {
                    const double* Wh = dc.mm(Wsc0, L->C_in, L->C_out, d.M, d.cols);
                    if (E) ft.Th = dc.to_f32(dc.mm(E, S_, C0_, Wh, d.cols), (int64_t)S_ * d.cols, L->weights);
                    else ft.Wh = dc.to_f32(Wh, (int64_t)L->C_in * d.cols, L->weights);
                }
                out[t][l].push_back(ft);
            }
        }
        for (int l = 0; l < 3; ++l) next[l] = cur[l];
    }
    for (int t = 0; t < nL - 1; ++t)
        for (int l = 0; l < 3; ++l) layers_[t]->fold[l] = std::move(out[t][l]);
    chain_fold_ = true;
}
std::string GenericModel::kernel_signature() const {
    std::string out;
    for (const GenericLayer* L : layers_) {
        if (!out.empty()) out += ';';
        out += (L && L->prod && L->prod->sig) ? L->prod->sig : "?";
    }
    return out;
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
    sys_bytes_ = 0;
    auto alloc = [&](size_t floats) -> float* {
        float* p = nullptr;
        CUDA_CHECK_THROW(cudaMalloc(&p, std::max<size_t>(floats, 1) * sizeof(float)));
        sys_bytes_ += std::max<size_t>(floats, 1) * sizeof(float);
        sys_allocs_.push_back(p);
        return p;
    };
    const size_t n = (size_t)N;
    auto align256 = [](size_t b) { return (b + 255) & ~size_t(255); };
    // Pack every zero-initialized per-step gradient accumulator into one contiguous slab (1 cudaMemsetAsync/step).
    size_t zbytes = 0;
    for (int t = 0; t < n_layers_; ++t) {
        GenericLayer* L = layers_[t];
        if (!L->ro_folded && !(chain_fold_ && t < n_layers_ - 1))
            for (int l = 0; l <= L->lmax_out; ++l) zbytes += align256(n * dimL(l) * L->C_out * sizeof(float));
        if (t > 0)
            for (int l = 0; l <= L->lmax_x; ++l) zbytes += align256(n * dimL(l) * L->C_x * sizeof(float));
        if (L->kind >= 2) zbytes += align256(n * sizeof(float));
    }
    if (zbytes) {
        CUDA_CHECK_THROW(cudaMalloc(&d_zero_slab_, zbytes));
        zero_slab_bytes_ = zbytes;
        sys_bytes_ += zbytes;
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
            for (int l = 0; l < 4; ++l) L->S0[l] = alloc(n * dimL(l) * (size_t)L->K0);
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
            const bool no_hout = L->ro_folded || (chain_fold_ && t < n_layers_ - 1);  // S57: never formed
            L->hout[l] = no_hout ? nullptr : alloc(n * dimL(l) * L->C_out);
            L->dhout[l] = no_hout ? nullptr : zalloc(n * dimL(l) * L->C_out);
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
    E_cap_ = 0;  // a failed allocation must not leave the old capacity on a null array
    CUDA_CHECK_THROW(cudaMalloc(&d_edge_dvec_, (size_t)E_cap * 3 * sizeof(float)));
    E_cap_ = E_cap;
    edge_bytes_ = (size_t)E_cap * 3 * sizeof(float);
}

// ------------------------------------------------------------------------------------------------- force pass

static const GenericLayer* pick_preloc_layer(const std::vector<GenericLayer*>& layers) {
    for (size_t t = 1; t < layers.size(); ++t)
        if (!layers[t]->first_hidden && layers[t]->C_x <= 256) return layers[t];
    if (!layers.empty() && !layers[0]->first_hidden && layers[0]->C_x <= 256) return layers[0];
    return nullptr;
}
const SplineGrid* GenericModel::preloc_grid() const {
    const GenericLayer* L = pick_preloc_layer(layers_);
    return L ? &L->grid : nullptr;
}
int GenericModel::preloc_pair_sym() const {
    const GenericLayer* L = pick_preloc_layer(layers_);
    return L ? L->pair_sym : 1;
}
void GenericModel::record(const GenericPassInputs& in) {
    const SplineGrid* pg = preloc_grid();
    const int pg_psym = preloc_pair_sym();
    auto can_preloc = [&](const GenericLayer* L) {
        return pg != nullptr && in.act_slab != nullptr && in.act_t_sx != nullptr && !L->first_hidden &&
               L->pair_sym == pg_psym && memcmp(&L->grid, pg, sizeof(SplineGrid)) == 0;
    };
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
    ZblParams zbl_p;  // r_max 0 without ZBL
    if (has_zbl_) {
        zbl_p.rc = d_zbl_rc_; zbl_p.inv_a = d_zbl_inv_a_; zbl_p.pref = d_zbl_pref_;
        zbl_p.c0 = zbl_c_[0]; zbl_p.c1 = zbl_c_[1]; zbl_p.c2 = zbl_c_[2]; zbl_p.c3 = zbl_c_[3];
        zbl_p.p = zbl_p_; zbl_p.r_max = zbl_rmax_;
    }
    if (has_edges) zero_edge_kernel<<<(3 * in.E_cap + 255) / 256, 256, 0, st>>>(in.E_dev, d_edge_dvec_);
    if (zero_slab_bytes_) CUDA_CHECK_THROW(cudaMemsetAsync(d_zero_slab_, 0, zero_slab_bytes_, st));

    gather_rows_kernel<<<(unsigned)(((int64_t)N * C0_ + 255) / 256), 256, 0, st>>>(N, C0_, in.species, d_embedding_, d_h0_);

    // ---------------- forward
    std::vector<const float*> written;  // S57 fold destinations already assigned in this pass
    for (int t = 0; t < n_layers_; ++t) {
        GenericLayer* L = layers_[t];
        const int Cx = L->C_x, Cm = L->C_msg, Co = L->C_out;
        // the hidden-table layer 0 needs x only for the nonlinear residual (none with the S59 residual table); with
        // the S57 fold, later layers' x comes from the earlier layers' folded consumers instead
        if ((!L->first_hidden || L->kind == 4) && !(chain_fold_ && t > 0) && !L->res0_table) {
            if (L->lmax_x > 0) fork();
            for (int l = 0; l <= L->lmax_x; ++l) {
                if (L->lmax_x > 0) lane(l);
                in.gemm(false, N * dimL(l), Cx, L->C_in, 1.0f, L->h_in[l], L->C_in, L->W_up[l], Cx, 0.0f, L->x[l], Cx);
            }
            if (L->lmax_x > 0) join();
        }
        if (L->first_hidden) {
            const int Kh = L->K0;
            first_fwd_kernel<<<N, ((std::max(L->J, 64) + 31) / 32) * 32, 0, st>>>(
                N, S, L->intervals, L->J, L->K0, L->grid, sh_perm_, L->pair_sym, in.act_row_ptr, in.act_senders,
                in.act_vec_len, in.species, L->table, L->S0[0], L->S0[1], L->S0[2], L->S0[3], L->D);
            if (L->skip_folded) {
                fork();
                for (int l = 0; l < 4; ++l)
                    species_gemm(false, l, Kh, Cm, L->S0[l], L->C0[l], (int64_t)Kh * Cm, 0.0f, L->msgf[l], 4 * l);
                join();
            } else {
                if (L->res0_table)  // S59: the residual per species (lmax_x 0), before the lanes fork
                    gather_rows_kernel<<<(unsigned)(((int64_t)N * L->lin_cols[0] + 255) / 256), 256, 0, st>>>(
                        N, L->lin_cols[0], in.species, L->res0_table, L->res[0]);
                fork();
                for (int l = 0; l < 4; ++l) {
                    lane(l);
                    in.gemm(false, N * dimL(l), L->lin_cols[l], Kh, 1.0f, L->S0[l], Kh, L->C0[l], L->lin_cols[l],
                            0.0f, L->lin[l], L->lin_cols[l]);
                }
                if (L->kind == 4 && !L->res0_table) {
                    for (int l = 0; l <= L->lmax_x; ++l) {
                        lane(4 + l);
                        in.gemm(false, N * dimL(l), L->lin_cols[l], Cx, 1.0f, L->x[l], Cx, L->W_res[l],
                                L->lin_cols[l], 0.0f, L->res[l], L->lin_cols[l]);
                    }
                }
                join();
            }
        } else {
            launch_conv_fwd(L->lmax_x, st, N, Cx, S, L->intervals, L->grid, sh_perm_, L->pair_sym, in.act_row_ptr,
                            in.act_senders, in.act_vec_len, in.act_slab, in.act_t_sx, can_preloc(L), in.species,
                            L->packed_table, L->density_table, L->cubic_bias_shift, L->x, L->mid, L->D);
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
        // the fused product + readout kernel runs one thread per channel (<= 512)
        const bool ro1_fused = L->ro_folded && L->readout == 1 && L->ro_H == 16 && L->prod->ro1_128 != nullptr &&
                               L->C_out <= 512;
        if (ro1_fused) {
            if (L->sc_has[0] && !chain_fold_) {  // with the S57 fold, hid already holds the skip's share
                if (L->sc_species) {
                    fork();
                    species_gemm(false, 0, L->C_in, 16, L->h_in[0], L->ro_W1_sc, (int64_t)L->C_in * 16, 0.0f, L->hid, 0);
                    join();
                } else {
                    in.gemm(false, N, 16, L->C_in, 1.0f, L->h_in[0], L->C_in, L->ro_W1_sc, 16, 0.0f, L->hid, 16);
                }
            }
            float* dm[4];
            for (int l = 0; l < 4; ++l) dm[l] = (L->kind == 1 || L->kind == 3) ? L->dlin[l] : L->dmsgf[l];
            float* dDp = (L->kind == 2 || L->kind == 3) ? L->dD : nullptr;
            launch_prod_ro1_fused(L->prod, st, N, Cm, L->msgf, Dp, L->prod_w,
                                  L->prod_species ? (int64_t)L->prod_segw * Cm : 0, in.species,
                                  L->sc_has[0] ? L->hid : nullptr, L->ro_W1, L->ro_b1, L->ro_W2, L->ro_act_c,
                                  L->dhid, in.energy, dm, dDp);
            continue;
        }
        launch_prod(L->prod, false, st, N, Cm, L->msgf, Dp, L->prod_w,
                    L->prod_species ? (int64_t)L->prod_segw * Cm : 0, in.species, L->y, nullptr, nullptr);
        if (chain_fold_ && t < n_layers_ - 1) {
            // S57: y_t (and layer 0's species-only skip input) into every downstream consumer of hout_t; the first
            // write of a destination in this pass assigns it, later ones accumulate (program order: race-free).
            // S59: the species-table skip terms (layer 0, l 0; each its destination's first write) are gathered on
            // the main stream before the lanes fork, so the lane GEMMs accumulate onto them.
            for (const auto& ft : L->fold[0]) {
                if (!ft.Th) continue;
                float* dst = ft.dest_layer >= 0 ? layers_[ft.dest_layer]->x[0] : layers_.back()->hid;
                if (std::find(written.begin(), written.end(), dst) != written.end())
                    throw std::logic_error("generic: species-table fold term is not a first write");
                gather_rows_kernel<<<(unsigned)(((int64_t)N * ft.cols + 255) / 256), 256, 0, st>>>(
                    N, ft.cols, in.species, ft.Th, dst);
                written.push_back(dst);
            }
            fork();
            int lane_i = 0;
            for (int l = 0; l <= L->lmax_out; ++l)
                for (const auto& ft : L->fold[l]) {
                    lane(lane_i++);
                    float* dst = ft.dest_layer >= 0 ? layers_[ft.dest_layer]->x[l] : layers_.back()->hid;
                    float beta = std::find(written.begin(), written.end(), dst) != written.end() ? 1.0f : 0.0f;
                    if (ft.Wh) {
                        in.gemm(false, N * dimL(l), ft.cols, L->C_in, 1.0f, L->h_in[l], L->C_in, ft.Wh, ft.cols, beta,
                                dst, ft.cols);
                        beta = 1.0f;
                    }
                    in.gemm(false, N * dimL(l), ft.cols, Co, 1.0f, L->y[l], Co, ft.Wy, ft.cols, beta, dst, ft.cols);
                    written.push_back(dst);
                }
            join();
        } else if (!L->ro_folded) {
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
                const bool hid_folded = chain_fold_ &&
                                        std::find(written.begin(), written.end(), L->hid) != written.end();
                in.gemm(false, N, H, Co, 1.0f, L->y[0], Co, L->ro_W1, H, hid_folded ? 1.0f : 0.0f, L->hid, H);
                if (L->sc_has[0] && !chain_fold_) {
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
        const bool ro1_fused = L->ro_folded && L->readout == 1 && L->ro_H == 16 && L->prod->ro1_128 != nullptr &&
                               L->C_out <= 512;
        if (ro1_fused) {
            if (L->sc_has[0] && t > 0 && !chain_fold_) {  // S57: the fold's backward carries this gradient
                if (L->sc_species) {
                    fork();
                    species_gemm(true, 0, 16, L->C_in, L->dhid, L->ro_W1_sc, (int64_t)L->C_in * 16, 1.0f,
                                 L->dh_in[0], 0);
                    join();
                } else {
                    in.gemm(true, N, L->C_in, 16, 1.0f, L->dhid, 16, L->ro_W1_sc, 16, 1.0f, L->dh_in[0], L->C_in);
                }
            }
        } else {
            if (L->ro_folded) {
                const int H = L->ro_H;
                const bool fork_sc = L->sc_has[0] && t > 0 && !chain_fold_;
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
            } else if (chain_fold_) {
                // S57: dy_t[l] from every downstream consumer's gradient through the composed weights (all final:
                // later layers and the readout ran first); one lane per degree, its terms in order
                fork();
                for (int l = 0; l <= L->lmax_out; ++l) {
                    lane(l);
                    float beta = 0.0f;
                    for (const auto& ft : L->fold[l]) {
                        const float* dsrc = ft.dest_layer >= 0 ? layers_[ft.dest_layer]->dx[l] : layers_.back()->dhid;
                        in.gemm(true, N * dimL(l), Co, ft.cols, 1.0f, dsrc, ft.cols, ft.Wy, ft.cols, beta, L->dy[l], Co);
                        beta = 1.0f;
                    }
                }
                join();
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
        }
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
            const int Kh = L->K0;
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
            if (L->J - 1 <= 96) {
                if (has_edges) {
                    using EdgeBwd = void (*)(const int*, int, int, int, int, SplineGrid, int, int, const int*, const int*,
                                             const float4*, const int*, const float4*, const float*, const float*, const float*,
                                             const float*, const float*, const float*, ZblParams, double*, double*, double*);
                    const EdgeBwd tbl[2][2][2] = {
                        {{first_bwd_edge_kernel<2, false, false>, first_bwd_edge_kernel<2, false, true>},
                         {first_bwd_edge_kernel<2, true, false>, first_bwd_edge_kernel<2, true, true>}},
                        {{first_bwd_edge_kernel<3, false, false>, first_bwd_edge_kernel<3, false, true>},
                         {first_bwd_edge_kernel<3, true, false>, first_bwd_edge_kernel<3, true, true>}}};
                    tbl[L->J - 1 > 64 ? 1 : 0][has_zbl_ ? 1 : 0][in.compute_stress ? 1 : 0]<<<(in.E_cap + 127) / 128, 128, 0, st>>>(
                        in.E_dev, S, L->intervals, L->J, L->K0, L->grid, sh_perm_, L->pair_sym, in.act_senders,
                        in.act_receivers, in.act_vec_len, in.species, L->table, L->S0[0], L->S0[1], L->S0[2], L->S0[3],
                        density ? L->dD : nullptr, d_edge_dvec_, zbl_p, in.energy, in.forces, in.virial);
                }
            } else {
                const size_t g_smem = (size_t)16 * S * (L->J - 1) * sizeof(float);
                if (g_smem <= 46 * 1024)  // + 2 KB static shared memory within the 48 KB default
                    first_bwd_kernel<true><<<N, 256, g_smem, st>>>(
                        N, S, L->intervals, L->J, L->K0, L->grid, sh_perm_, L->pair_sym, in.act_row_ptr, in.act_senders,
                        in.act_vec_len, in.species, L->table, L->S0[0], L->S0[1], L->S0[2], L->S0[3],
                        density ? L->dD : nullptr, d_edge_dvec_);
                else
                    first_bwd_kernel<false><<<N, 256, 0, st>>>(
                        N, S, L->intervals, L->J, L->K0, L->grid, sh_perm_, L->pair_sym, in.act_row_ptr, in.act_senders,
                        in.act_vec_len, in.species, L->table, L->S0[0], L->S0[1], L->S0[2], L->S0[3],
                        density ? L->dD : nullptr, d_edge_dvec_);
            }
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
        launch_conv_bwd(L->lmax_x, st, N, Cx, S, L->intervals, L->grid, sh_perm_, L->pair_sym, in.act_row_ptr,
                        in.act_senders, in.act_vec_len, in.act_slab, in.act_t_sx, can_preloc(L), in.species,
                        L->packed_table, L->density_table, L->cubic_bias_shift, L->x, L->mid,
                        density ? L->dD : nullptr, L->dx, t > 0, d_edge_dvec_);
        if (t > 0 && !chain_fold_) {  // S57: the earlier layers' fold reads dx directly
            if (L->lmax_x > 0) fork();
            for (int l = 0; l <= L->lmax_x; ++l) {
                if (L->lmax_x > 0) lane(l);
                in.gemm(true, N * dimL(l), L->C_in, Cx, 1.0f, L->dx[l], Cx, L->W_up[l], Cx, 1.0f, L->dh_in[l], L->C_in);
            }
            if (L->lmax_x > 0) join();
        }
    }
    const bool edge_done = layers_[0]->first_hidden && layers_[0]->J - 1 <= 96;
    if (has_edges && !edge_done) {
        auto kern = has_zbl_ ? (in.compute_stress ? assemble_forces_kernel<true, true> : assemble_forces_kernel<true, false>)
                             : (in.compute_stress ? assemble_forces_kernel<false, true> : assemble_forces_kernel<false, false>);
        kern<<<(in.E_cap + 255) / 256, 256, 0, st>>>(in.E_dev, S, in.act_vec_len, in.act_senders, in.act_receivers,
                                                    in.species, zbl_p, d_edge_dvec_, in.forces, in.virial, in.energy);
    }
}

}  // namespace loup
