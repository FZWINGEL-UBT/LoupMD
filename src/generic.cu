#include "generic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <dlfcn.h>

#include <string>

#include "bundle.hpp"
#include "generated_generic.cuh"
#include "kernel_plugin.h"
#include "model.hpp"
#include "polar.hpp"
#include "jit.hpp"
#include "kernel_ir.hpp"
#include "layer0_kernels.cuh"

namespace loup {

// ------------------------------------------------------------------------------------------------- kernel plugins
// The product kernels of `sig` from a plugin library (kernel_plugin.h), or NULL with the reason in `why`.
// LOUP_DEBUG_HIDE_KERNEL=<sig> (testing) makes the library's own kernels of that signature invisible, so that a
// plugin for a model the library does have can be exercised. A loaded plugin stays loaded.
static const GProdKernels* find_plugin_gprod(const std::string& sig, std::string& why) {
    const char* path = std::getenv("LOUP_KERNEL_PATH");
    if (!path || !*path) {
        why = "LOUP_KERNEL_PATH names no folder to look for a kernel plugin in";
        return nullptr;
    }
    const std::string dirs(path);
    size_t start = 0;
    why = "no loupmd-kernels-" + sig + ".so in " + dirs;
    while (start <= dirs.size()) {
        const size_t end = std::min(dirs.find(':', start), dirs.size());
        const std::string file = dirs.substr(start, end - start) + "/loupmd-kernels-" + sig + ".so";
        start = end + 1;
        void* lib = dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!lib) continue;
        typedef int (*abi_fn)();
        typedef const GProdKernels* (*find_fn)(const char*);
        abi_fn abi = reinterpret_cast<abi_fn>(dlsym(lib, "loup_kernel_abi"));
        find_fn find = reinterpret_cast<find_fn>(dlsym(lib, "loup_plugin_gprod"));
        if (!abi || !find) {
            why = file + " is not a kernel plugin (loup_kernel_abi or loup_plugin_gprod is missing)";
            continue;
        }
        if (abi() != LOUP_KERNEL_ABI) {
            why = file + " was built for kernel ABI " + std::to_string(abi()) + ", this library has " +
                  std::to_string(LOUP_KERNEL_ABI) + ": build it again with `loupmd build-kernels`";
            continue;
        }
        if (const GProdKernels* k = find(sig.c_str())) return k;
        why = file + " does not hold the kernels of signature " + sig;
    }
    return nullptr;
}

// The library's own kernels of a signature, else a kernel plugin's; null, with `why` no plugin was found, when
// neither exists (the caller then turns to the bundle's source, JIT_DESIGN.md stage 3).
static const GProdKernels* find_product_kernels(const std::string& sig, std::string& why) {
    const char* hidden = std::getenv("LOUP_DEBUG_HIDE_KERNEL");
    if (!(hidden && sig == hidden)) {
        if (const GProdKernels* k = find_gprod(sig.c_str())) return k;
    }
    return find_plugin_gprod(sig, why);
}

// ------------------------------------------------------------------------------------------------- layer state
struct GenericLayer {
    // configuration (bundle meta)
    int kind = 0;  // 0 agnostic, 1 agnostic_res, 2 density, 3 density_res, 4 nonlinear
    int lmax_in = 0, lmax_x = 0, lmax_out = 0;
    int C_in = 0, C_x = 0, C_msg = 0, C_out = 0;
    int n_paths = 0, n_l3[4] = {0, 0, 0, 0};
    // pair_sym: radial tables per unordered pair (1), per ordered pair (0), or one table for every pair (2, S116: the
    // bundle's per-pair tables are byte-identical, as in the MACE-OFF bundles; loup::table_pair)
    int intervals = 0, Q1 = 0, pair_sym = 0;
    int first_hidden = 0, J = 0, K0 = 0;  // layer 0 outer sum over J columns; K0 = S * (J - 1) padded to 16
    int Jt = 0;  // S128: layer 0's table row stride (float4): J, or J - 1 without the density column (kinds 0 / 1)
    float* C0[4] = {};                    // [K0, lin_cols] (per species when skip_folded)
    float* S0[4] = {};                    // [N, 2l+1, K0] (the backward reuses it for dS0)
    int has_sc = 0, sc_species = 0, readout = -1, ro_H = 0, prod_species = 0, prod_segw = 0;
    int skip_folded = 0, ro_folded = 0;
    int fwd_var = 0;           // S64: conv forward variant (launch_conv_fwd; tune_conv_variants picks it per system)
    bool last_preloc = false;  // S64: the preloc flag of the last recorded conv forward (for the re-launch)
    // S125: W_lin_1..3 exactly zero (certified at load), so lin_1..3 are exactly zero: the pruned lx 1 conv kernels
    // (gconv_*_L1s, the output-degree-0 paths only) run, and the l >= 1 projections are skipped in both directions
    bool conv_prune = false;
    // S130 / S132: compact live channels of a kind-1 layer (compact_layout, certified at load): lin_cols[l] are the
    // compact projection widths (degree 0: the kept channels, angular first; degree l >= 1: only its live columns, d_pos
    // mapping each angular position to its column there), n_ang the angular channels, kept the live ones; the terminal
    // layer's fused product + readout kernel gprod_ro1c and a non-terminal layer's gprod_fwdc / bwdc (through d_order,
    // compact position -> channel) run on them (LOUP_LIN_COMPACT=0: dense)
    bool lin_compact = false;
    int n_ang = 0, kept = 0;
    int *d_order = nullptr, *d_pos = nullptr;
    int lin_cols[4] = {0, 0, 0, 0};
    int sc_has[3] = {0, 0, 0};
    SplineGrid grid{};
    // JIT_DESIGN.md stage 1 (LOUP_JIT=1): layer 0's kernels compiled at load with this layer's constants (kernel_ir.hpp);
    // null: the ahead-of-time kernels
    const void* jit_fwd = nullptr;
    const void* jit_bwd_edge[2] = {nullptr, nullptr};  // [stress]
    // stage 2: the conv kernels of a width without a built-in instance (the launchers' run-time-width branches)
    ConvJit jit_conv;
    bool jit_conv_on = false;
    float alpha = 1.0f, beta = 0.0f, gate_cs = 1.0f, gate_cg = 1.0f, ro_act_c = 1.0f;
    const GProdKernels* prod = nullptr;
    // LOUP_W2T_HALF (setup_w2t_half, hgemm2.cu): the library's _h2 kernels of this product (null: none, or the product
    // runs a plugin's or compiled-at-load kernels); for a layer that runs the path (h2), prod's table with bwd / ro1_*
    // replaced by them, the W_2[l] panels ([2][C_msg][C_msg] halves at scale 2^w2h_exp[l]) and the output scales
    const GProdH2Kernels* prod_h2src = nullptr;
    GProdKernels prod_h2{};
    void* w2h[4] = {};
    int w2h_exp[4] = {};
    float w2h_oscale[4] = {};
    bool h2 = false;
    // LOUP_W2F_HALF (setup_w2f_half): the panels of W_2[l]^T ([2][C_msg][C_msg] halves at 2^w2h_exp[l]), gout's split
    // scale and the output scales
    void* w2fh[4] = {};
    float w2f_sA[4] = {};
    float w2f_oscale[4] = {};
    bool f2 = false;
    // LOUP_LIN_HALF (setup_lin_half): a conv layer's plain projections on hgemm2f_kernel, forward lin_l = mid_l W_lin[l]
    // (layer 0: S0_l C0_l) and res_l = x_l W_res[l], backward dmid_l = dlin_l W_lin[l]^T (layer 0: dS0_l = dlin_l
    // C0_l^T) and dx_l += dpre_l W_res[l]^T: B's panels [2][Np][Kp] (null: the FP32 GEMM), A's split scale and the
    // output scale
    struct H2Proj {
        void* B = nullptr;
        float sA = 0.0f, os = 0.0f;
    };
    H2Proj hl_flin[4], hl_fres[3], hl_blin[4], hl_bres[3];
    H2Proj hl_bcat[3];  // LOUP_RES_BWD 3: hl_blin's and hl_bres's panels stacked, [2][n_l3 Cx + Cx][lin_cols]
    // list 24 item 4 (LOUP_LIN_HALF_PARTS bit 128): a skip-folded kind 1 / 3 layer 0's species-grouped projections per
    // degree and species, forward msgf_l = S0_l C0_l[s] and backward dS0_l = dmsgf_l C0_l[s]^T (empty: FP32 GEMMs)
    std::vector<H2Proj> hs_flin[4], hs_blin[4];
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
    // layer's x[l] (its linear_up input, through the skips in between); -1: the last layer's readout hidden (hid);
    // -2 (l 0, cols 1): the energy, a linear readout of this or a later non-last layer (seeds dy[0] in the forward).
    // ns = S when a per-species skip lies on the way (Wy, Wh per species, species-grouped GEMMs), else 1.
    // Wh (layer 0 only): the same consumers of the layer-0 skip h_in[0] W_sc[0]; Th instead (S59 species tables, and
    // always for the energy): [S, cols] = embedding W_sc[0] M, gathered per atom (always the destination's first
    // write) or, for the energy, added per atom.
    struct FoldTarget {
        int dest_layer = 0, cols = 0, ns = 1, ns_h = 1;
        H2Proj hf, hb;        // LOUP_LIN_HALF_PARTS bit 256 (list 25 item 1): panels of Wy (y Wy) and of Wy^T (dsrc Wy^T)
        float* Wy = nullptr;  // [ns, C_out, cols]
        float* Wh = nullptr;  // [ns_h, C_in, cols] or nullptr
        float* Th = nullptr;  // [S, cols] or nullptr
    };
    std::vector<FoldTarget> fold[3];
    // S59 (layer 0): [S, lin_cols[0]] species table of the residual (embedding W_up[0] W_res[0]), or nullptr
    float* res0_table = nullptr;
    // Long-range hidden fold (GenericModel::setup_lr_fold): hfold[l], degree l of hout is never formed: hout[l]
    // aliases y[l] and dhout[l] aliases dy[l] (in the zero slab), and every consumer's weight (the next layer's W_up /
    // W_sc, the long-range mixer and source) is composed at load with W_p[l] and the skips in between. x_extra /
    // sc_extra: further terms of this layer's x[l] / skip sum, y_src[l] W from an earlier folded layer src (backward:
    // dy_src[l] += dx[l] W^T / dhout[l] W^T).
    bool hfold[3] = {false, false, false};
    struct YTerm {
        int src = 0;
        float* W = nullptr;
    };
    std::vector<YTerm> x_extra[3], sc_extra[3];

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

// JIT_DESIGN.md stage 3: a product's kernels compiled from the bundle's source, one table per (signature, source) for
// the process (the modules are never unloaded, jit.hpp). Null on failure, `err` says why.
const GProdKernels* jit_product(const std::string& sig, int lmax_out, int segw, const std::string& src,
                                std::string* err) {
    struct Entry {
        std::string sig;
        GProdKernels k;
    };
    static std::mutex mu;
    static std::map<std::string, std::unique_ptr<Entry>> made;
    const std::string key = sig + ":" + std::to_string(std::hash<std::string>{}(src));
    std::lock_guard<std::mutex> lk(mu);
    if (auto it = made.find(key); it != made.end()) return &it->second->k;
    const std::vector<std::string> names = product_jit_names(sig, lmax_out);
    const std::vector<const void*> h = jit::build("prod", product_jit_source(src), jit::embedded_headers(), names, err);
    if (h.size() != names.size()) return nullptr;
    auto e = std::make_unique<Entry>();
    e->sig = sig;
    GProdKernels& k = e->k;
    k = GProdKernels{};
    k.sig = e->sig.c_str();
    k.lmax_out = lmax_out;
    k.segments_w = segw;
    size_t i = 0;
    k.fwd = h[i++];
    k.bwd = h[i++];
    if (lmax_out == 0) {
        for (const void** f : {&k.ro1_128, &k.ro1_256, &k.ro1_512, &k.ro1s_128, &k.ro1s_256, &k.ro1s_512, &k.ro1c_128,
                               &k.ro1c_256, &k.ro1c_512})
            *f = h[i++];
    }
    k.fwdc = h[i++];
    k.bwdc = h[i++];
    const GProdKernels* out = &k;
    made.emplace(key, std::move(e));
    return out;
}

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
// out_l = cs_g sigmoid(pre_0[l C:(l+1) C]) * pre_l (l = 1..3). One block per atom, one thread per channel. With res0_sp,
// res0 is the S59 [S, 4 C] species table and atom a reads row res0_sp[a] (no per-atom copy). Degrees l with bit l of
// `fused` set are skipped (their projection's epilogue wrote out_l: sgemm128_gate, forward).
__global__ void gate_fwd_kernel(int N, int C, int lx, const float* __restrict__ D, float alpha, float beta, float cs,
                                float cg, const float* __restrict__ lin0, const float* __restrict__ lin1,
                                const float* __restrict__ lin2, const float* __restrict__ lin3,
                                const float* __restrict__ res0, const int* __restrict__ res0_sp,
                                const float* __restrict__ res1, const float* __restrict__ res2, float* __restrict__ o0,
                                float* __restrict__ o1, float* __restrict__ o2, float* __restrict__ o3, unsigned fused) {
    const int a = blockIdx.x, c = threadIdx.x;
    if (a >= N) return;
    const float s = 1.0f / (D[a] * beta + alpha);
    const int64_t r0 = (int64_t)a * 4 * C, q0 = res0_sp ? (int64_t)res0_sp[a] * 4 * C : r0;
    auto pre0 = [&](int col) { return lin0[r0 + col] * s + (res0 ? res0[q0 + col] : 0.0f); };
    o0[(int64_t)a * C + c] = cs * silu_f(pre0(c));
    const float* lins[3] = {lin1, lin2, lin3};
    const float* ress[3] = {res1, res2, nullptr};
    float* outs[3] = {o1, o2, o3};
    #pragma unroll
    for (int l = 1; l <= 3; ++l) {
        if (fused >> l & 1u) continue;
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
// dD += sum dpre . lin * (-beta s^2) (one atomic per warp). res0 / res0_sp as in gate_fwd. Degrees l with bit l of
// `fused` set are skipped (their W_2^T GEMM ran sgemm128_gate, which did their share). Degrees with bit l of `vform`
// (residual-free, their forward fused: lin_l is never stored) pass the saved out_l = v as lin_l: with q = sum dgout v,
// d pre_0[l C + c] = (1 - sigmoid) q and dD += -beta s q (the same terms; v = g (lin s)). Degrees with bit l of `rform`
// (list 27 item 3: a residual degree whose forward gate ran in W_res's epilogue) pass the saved out_l = g pre_l as
// res_l: d pre_0[l C + c] = (1 - sigmoid) sum dgout out_l; lin_l as usual.
__global__ void gate_bwd_kernel(int N, int C, int lx, const float* __restrict__ D, float alpha, float beta, float cs,
                                float cg, const float* __restrict__ lin0, const float* __restrict__ lin1,
                                const float* __restrict__ lin2, const float* __restrict__ lin3,
                                const float* __restrict__ res0, const int* __restrict__ res0_sp,
                                const float* __restrict__ res1, const float* __restrict__ res2,
                                const float* __restrict__ g0, const float* __restrict__ g1, const float* __restrict__ g2,
                                const float* __restrict__ g3, float* __restrict__ dp0, float* __restrict__ dp1,
                                float* __restrict__ dp2, float* __restrict__ dl0, float* __restrict__ dl1,
                                float* __restrict__ dl2, float* __restrict__ dl3, float* __restrict__ dD,
                                unsigned fused, unsigned vform, unsigned rform) {
    const int a = blockIdx.x, c = threadIdx.x;
    if (a >= N) return;
    const float s = 1.0f / (D[a] * beta + alpha);
    const int64_t r0 = (int64_t)a * 4 * C, q0 = res0_sp ? (int64_t)res0_sp[a] * 4 * C : r0;
    auto pre0 = [&](int col) { return lin0[r0 + col] * s + (res0 ? res0[q0 + col] : 0.0f); };
    float acc = 0.0f, accq = 0.0f;  // sum dpre * lin; sum dgout * v (vform degrees)
    if (!(fused & 1u)) {
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
        if (fused >> l & 1u) continue;
        const float gp = pre0(l * C + c);
        const float sg = sigmoid_f(gp);
        const float g = cg * sg;
        const float* li = lins[l - 1];
        const float* re = (l <= lx) ? ress[l - 1] : nullptr;
        float* dpl = (l <= lx && l <= 2) ? dps[l - 1] : nullptr;
        float dgp;
        if (vform >> l & 1u) {
            float q = 0.0f;
            for (int m = 0; m < 2 * l + 1; ++m) {
                const int64_t idx = ((int64_t)a * (2 * l + 1) + m) * C + c;
                const float go = gs[l - 1][idx];
                q = fmaf(go, li[idx], q);
                dls[l - 1][idx] = go * g * s;
            }
            dgp = (1.0f - sg) * q;
            accq += q;
        } else {
            const bool rf = rform >> l & 1u;  // re holds out_l = g pre_l
            float dg = 0.0f;
            for (int m = 0; m < 2 * l + 1; ++m) {
                const int64_t idx = ((int64_t)a * (2 * l + 1) + m) * C + c;
                const float p = rf ? re[idx] : li[idx] * s + (re ? re[idx] : 0.0f);
                const float go = gs[l - 1][idx];
                dg = fmaf(go, p, dg);
                const float d = go * g;
                if (dpl) dpl[idx] = d;
                dls[l - 1][idx] = d * s;
                acc = fmaf(d, li[idx], acc);
            }
            dgp = rf ? (1.0f - sg) * dg : dg * cg * sg * (1.0f - sg);
        }
        if (dp0) dp0[r0 + l * C + c] = dgp;
        dl0[r0 + l * C + c] = dgp * s;
        acc = fmaf(dgp, lin0[r0 + l * C + c], acc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
        accq += __shfl_xor_sync(0xffffffffu, accq, off);
    }
    if ((c & 31) == 0) atomicAdd(dD + a, -beta * s * s * acc - beta * s * accq);
}

// E += sum_a c[species[a]] (S73: a folded linear readout's share of the species-only layer-0 skip).
__global__ void species_energy_kernel(int N, const int* __restrict__ sp, const float* __restrict__ c,
                                      double* __restrict__ energy, double* __restrict__ atom_energy = nullptr) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    double e = a < N ? (double)c[sp[a]] : 0.0;
    if (atom_energy && a < N) atom_energy[a] += e;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) e += __shfl_xor_sync(0xffffffffu, e, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(energy, e);
}

// Linear readout: E += sum_a h0[a] . w (double accumulation); its gradient dh0 = w (assign: the pass's first write of
// dh0) or dh0 += w.
__global__ void ro_linear_kernel(int N, int C, const float* __restrict__ h0, const float* __restrict__ w,
                                 float* __restrict__ dh0, double* __restrict__ energy, bool assign,
                                 double* __restrict__ atom_energy = nullptr) {
    const int a = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (a >= N) return;
    double e = 0.0;
    for (int c = lane; c < C; c += 32) {
        const float wc = w[c];
        e += (double)h0[(int64_t)a * C + c] * (double)wc;
        if (assign)
            dh0[(int64_t)a * C + c] = wc;
        else
            dh0[(int64_t)a * C + c] += wc;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) e += __shfl_xor_sync(0xffffffffu, e, off);
    if (lane == 0) {
        atomicAdd(energy, e);
        if (atom_energy) atom_energy[a] += e;
    }
}

// MLP readout tail on hid [N, H] (= h0 @ W1, bias added here): kind 1: E += sum c silu(hid + b1) . W2;
// kind 2 (bias MLP): stage 1 writes act1 = c silu(hid + b1) into hid (for the mid GEMM) and keeps the
// pre-activation in pre; stage 2 runs on hid2 = act1 @ Wm. dhid receives dE/d(hid + b1) of the stage.
__global__ void ro_mlp_kernel(int N, int H, float c, float* __restrict__ hid, const float* __restrict__ b,
                              const float* __restrict__ W2, float* __restrict__ pre, float* __restrict__ dhid,
                              double* __restrict__ energy, int final_stage,
                              double* __restrict__ atom_energy = nullptr) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if (i < (int64_t)N * H) {
        const int j = (int)(i % H);
        const float x = hid[i] + b[j];
        if (final_stage) {
            const float w2 = W2[j];
            e = (double)(c * silu_f(x)) * (double)w2;
            dhid[i] = c * dsilu_f(x) * w2;
            if (atom_energy && e != 0.0) atomicAdd(&atom_energy[i / H], e);
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


// The ZBL pair energy alone, per edge as assemble_forces_kernel and first_bwd_edge_kernel add it in the full pass,
// for the energy-only pass.
__global__ void zbl_energy_kernel(const int* __restrict__ E_dev, int S, const float4* __restrict__ vec_len,
                                  const int* __restrict__ senders, const int* __restrict__ receivers,
                                  const int* __restrict__ species, ZblParams z, double* __restrict__ energy,
                                  double* __restrict__ atom_energy = nullptr) {
    const int E = *E_dev;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    double en = 0.0;
    if (e < E) {
        const float4 vl = vec_len[e];
        float zen, g;
        if (vl.w < z.r_max && zbl_pair(z, species[senders[e]] * S + species[receivers[e]], vl.w, zen, g)) en = zen;
        if (atom_energy && en != 0.0) atomicAdd(&atom_energy[receivers[e]], en);  // MACE: the receiver's node energy
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) en += __shfl_xor_sync(0xffffffffu, en, off);
    if ((threadIdx.x & 31) == 0 && en != 0.0) atomicAdd(energy, en);
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


// STAGE_G: the receiver's gradient rows G[m][s Jp + j] (16 x K floats, the same for all its edges) are staged once in
// shared memory, m innermost in float4 chunks swizzled by (k >> 1) & 3 so a warp's column-consecutive float4 reads
// are bank-conflict free: 4 shared loads per column instead of 16 global ones. Without it (K too large for shared
// memory) the rows are read from global memory per edge.
template <bool STAGE_G>
__global__ void __launch_bounds__(256) first_bwd_kernel(
    int N, int S, int I, int J, int Jt, int K0, SplineGrid grid, int sh_perm, int pair_sym,
    const int* __restrict__ row_ptr, const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species, const float4* __restrict__ table,
    const float* __restrict__ g0, const float* __restrict__ g1, const float* __restrict__ g2, const float* __restrict__ g3,
    const float* __restrict__ dD, float* __restrict__ edge_dvec) {
    __shared__ float s_dy[8][32];
    extern __shared__ float4 s_G4[];
    const int a = blockIdx.x, tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, n_warps = blockDim.x >> 5;
    if (a >= N) return;
    const int Jp = J - 1, Jh = (Jp + 1) >> 1, K = S * Jp, sa = species[a];
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
        const int pr = loup::table_pair(pair_sym, s, sa, S);
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
        for (int j = hl; j < Jt; j += 16) {  // S128: j == Jp (the density) only when the table keeps it
            const float4 cf = __ldg(table + ((int64_t)pr * I + iv) * Jt + (j < Jp ? first_col_pos(j, Jh) : j));
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

__global__ void zero_edge_kernel(const int* __restrict__ E_dev, float* __restrict__ p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 3 * *E_dev) p[i] = 0.0f;
}

// C[M, N] = A[M, K] B[K, N] (+ D[M, N]) in double, one thread per output: load-time weight composition (S57 / S59 /
// the long-range hidden fold).
__global__ void dmm_kernel(int M, int N, int K, const double* __restrict__ A, const double* __restrict__ B,
                           const double* __restrict__ D, double* __restrict__ C) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (n >= N || m >= M) return;
    double s = D ? D[(size_t)m * N + n] : 0.0;
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
    // C = A[M, K] B[K, N] (+ D[M, N]); into `out` [M, N] if given, else a new temporary
    const double* mm(const double* A, int M, int K, const double* B, int N, const double* D = nullptr,
                     double* out = nullptr) {
        double* C = out;
        if (!C) {
            CUDA_CHECK_THROW(cudaMalloc(&C, (size_t)M * N * sizeof(double)));
            tmp.push_back(C);
        }
        dmm_kernel<<<dim3((unsigned)((N + 127) / 128), (unsigned)M), 128>>>(M, N, K, A, B, D, C);
        CUDA_CHECK_THROW(cudaGetLastError());
        return C;
    }
    double* alloc(int64_t n) {
        double* q = nullptr;
        CUDA_CHECK_THROW(cudaMalloc(&q, std::max<int64_t>(n, 1) * sizeof(double)));
        tmp.push_back(q);
        return q;
    }
    // Per species: C[s] = A[s or 0] B[s or 0], s < max(nsA, nsB) (each operand either one block or one per species).
    const double* mm_s(const double* A, int nsA, int M, int K, const double* B, int nsB, int N) {
        const int ns = std::max(nsA, nsB);
        double* C = nullptr;
        CUDA_CHECK_THROW(cudaMalloc(&C, (size_t)ns * M * N * sizeof(double)));
        tmp.push_back(C);
        for (int s = 0; s < ns; ++s)
            dmm_kernel<<<dim3((unsigned)((N + 127) / 128), (unsigned)M), 128>>>(
                M, N, K, A + (nsA > 1 ? (size_t)s * M * K : 0), B + (nsB > 1 ? (size_t)s * K * N : 0), nullptr,
                C + (size_t)s * M * N);
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

// The load-time folds (S57, S59, S73) were measured on compute capability 12.0 (5060 Ti), 8.0 (A100, S74a: faster on
// every family, gated in S75g), 8.9 (RTX 4000 Ada, List 30: 1.14-1.84x across families, gated on leto/leeloo) and
// 8.6 (RTX A4000, List 33: 0.60-0.92 of the plain path's step with the rest of the tuned set, gated on
// trillian/coraline); other GPUs keep the plain path.
bool folds_measured_here() {
    int dev = 0, major = 0, minor = 0;
    CUDA_CHECK_THROW(cudaGetDevice(&dev));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    return (major == 12 && minor == 0) || (major == 8 && (minor == 0 || minor == 6 || minor == 9));
}

// Compute capabilities where the two-half paths, POLAR species tables / hidden fold, and per-system conv timing
// were measured and gated (5060 Ti sm_120 and RTX 4000 Ada sm_89).
bool device_is_tuned_cc() {
    int dev = 0, major = 0, minor = 0;
    CUDA_CHECK_THROW(cudaGetDevice(&dev));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    return (major == 12 && minor == 0) || (major == 8 && (minor == 6 || minor == 9));
}

// The same set plus compute capability 8.0 (A100, list 37: the two-half package and the conv timing 0.81-0.98 of
// the step beside the TF32 + FP16 cubin, gated on the generic families). MACE-POLAR-1's own defaults
// (setup_polar_h2, the species tables, the hidden fold) follow since the second part of list 37 (0.57-0.84 of the
// step, gated); only its per-system conv timing, not measured there, keeps device_is_tuned_cc().
static bool device_is_tuned_generic_cc() {
    if (device_is_tuned_cc()) return true;
    int dev = 0, major = 0, minor = 0;
    CUDA_CHECK_THROW(cudaGetDevice(&dev));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    return major == 8 && minor == 0;
}

// S130 / S132: compact live channels of a kind-1 layer (the projections feed the product directly), certified from exact
// zeros of the loaded float32 weights. Channel c is dead when its product weights (every species) or every projection
// column (C0_l or W_lin_l [:, c]) are zero, or, on the fused terminal path, its folded readout row; it is angular when
// some projection column with l >= 1 is nonzero, else scalar-only (x1..x15 exactly zero: the S127 program). The
// positions hold the angular channels, then the scalar-only ones, then (all_channels: non-terminal layers, whose dead
// channels write zero outputs) the dead ones, each group in channel order. Degree 0's compact projection has a column
// per kept position; degree l >= 1 keeps only its live columns (pos: an angular position's column there, or -1). The
// bundle holds the system's species only, so the sets depend on them: MPtrj-2022.9 on SrTiO3 keeps 205 / 1 / 2 / 12 of
// 256 in layer 1 and 251 / 190 / 138 / 123 in layer 0 (NbCl5: 173 and 244 kept). Widths are padded to multiples of 8
// with zero columns (against 4: MPtrj-2022.9 2.0% / 0.9% faster per step on SrTiO3 / NbCl5, MP-0 medium SrTiO3 0.35%
// slower). False when the degree-1..3 projection work does not shrink by at least a quarter (the dense kernels stay).
struct CompactLayout {
    std::vector<int> order;  // compact position -> channel: the kept channels (then the dead ones with all_channels)
    std::vector<int> pos;    // [3][n_ang]: angular position i's column in degree l's compact projection, or -1
    int kept = 0, n_ang = 0;
    int width[4] = {0, 0, 0, 0};
};
bool compact_layout(int C, const std::vector<float>* proj, const int* K, const std::vector<float>& prodw, int Sp,
                    int segw, const std::vector<float>* roW1, int H, bool all_channels, CompactLayout& out) {
    std::vector<unsigned char> nz[4];
    for (int l = 0; l < 4; ++l) {
        nz[l].assign(C, 0);
        for (int64_t k = 0; k < K[l]; ++k)
            for (int c = 0; c < C; ++c)
                if (proj[l][k * C + c] != 0.0f) nz[l][c] = 1;
    }
    std::vector<int> ang, sca, dead;
    for (int c = 0; c < C; ++c) {
        bool pw = false, ro = roW1 == nullptr;
        for (int64_t r = 0; r < (int64_t)Sp * segw && !pw; ++r) pw = prodw[r * C + c] != 0.0f;
        for (int j = 0; j < H && !ro; ++j) ro = (*roW1)[(int64_t)c * H + j] != 0.0f;
        const bool angular = nz[1][c] || nz[2][c] || nz[3][c];
        if (!pw || !ro || !(nz[0][c] || angular)) {  // dead: its product output is zero or never reaches the energy
            dead.push_back(c);
            continue;
        }
        (angular ? ang : sca).push_back(c);
    }
    out.order = ang;
    out.order.insert(out.order.end(), sca.begin(), sca.end());
    out.kept = (int)out.order.size();
    out.n_ang = (int)ang.size();
    if (all_channels) out.order.insert(out.order.end(), dead.begin(), dead.end());
    auto pad8 = [](int v) { return std::max(8, (v + 7) / 8 * 8); };
    out.width[0] = pad8(out.kept);
    out.pos.assign((size_t)3 * out.n_ang, -1);
    int work = 0;
    for (int l = 1; l < 4; ++l) {
        int w = 0;
        for (int i = 0; i < out.n_ang; ++i)
            if (nz[l][out.order[i]]) out.pos[(size_t)(l - 1) * out.n_ang + i] = w++;
        out.width[l] = pad8(w);
        work += (2 * l + 1) * out.width[l];
    }
    return 4 * work <= 3 * 15 * C;
}

}  // namespace

// ------------------------------------------------------------------------------------------------- load
void GenericModel::setup_polar_h2() {
    polar_h2_ = false;
    if (!lr_) return;
    const char* e = std::getenv("LOUP_POLAR_H2");
    if (e && (e[0] == '0' || e[0] == '1')) {
        if (e[0] == '0') return;
    } else if (!device_is_tuned_generic_cc()) {
        return;
    }
    if (const char* mm = std::getenv("LOUP_POLAR_H2_MINM"); mm && *mm) polar_h2_min_m_ = std::atoi(mm);
    auto reg = [&](const float* W, int rows_w, int cols_w) {
        if (!W || rows_w < 32 || cols_w < 32 || rows_w % 16 != 0 || cols_w % 16 != 0) return;
        if (ph2_map_.count(W)) return;
        H2RowW hw; hw.rows = rows_w; hw.cols = cols_w;
        CUDA_CHECK_THROW(cudaMalloc(&hw.B, 2 * hgemm2_pad_elems(cols_w, rows_w) * 2));
        CUDA_CHECK_THROW(cudaMalloc(&hw.pB, (size_t)cols_w * sizeof(float)));
        hgemm2_pack_b_col(cols_w, rows_w, W, true, hw.B, hw.pB, nullptr);
        ph2_allocs_.push_back(hw.B); ph2_allocs_.push_back(hw.pB);
        CUDA_CHECK_THROW(cudaMalloc(&hw.BT, 2 * hgemm2_pad_elems(rows_w, cols_w) * 2));
        CUDA_CHECK_THROW(cudaMalloc(&hw.pBT, (size_t)rows_w * sizeof(float)));
        hgemm2_pack_b_col(rows_w, cols_w, W, false, hw.BT, hw.pBT, nullptr);
        ph2_allocs_.push_back(hw.BT); ph2_allocs_.push_back(hw.pBT);
        ph2_map_[W] = hw;
    };
    for (auto* L : layers_) {
        const int Cx = L->C_x, Cm = L->C_msg, Co = L->C_out;
        for (int l = 0; l < 4; ++l) {
            if (l <= L->lmax_x) {
                reg(L->W_up[l], L->C_in, Cx);
                reg(L->W_res[l], Cx, L->lin_cols[l]);
                for (const auto& ex : L->x_extra[l]) reg(ex.W, L->C_in, Cx);
            }
            if (l < 3) {
                reg(L->W_p[l], Co, Co);
                if (L->sc_has[l] && !L->sc_species) reg(L->W_sc[l], L->C_in, Co);
                for (const auto& ex : L->sc_extra[l]) reg(ex.W, L->C_in, Co);
            }
            if (L->first_hidden && !L->skip_folded) reg(L->C0[l], L->K0, L->lin_cols[l]);
            else if (!L->first_hidden) reg(L->W_lin[l], L->n_l3[l] * Cx, L->lin_cols[l]);
            if (L->kind == 4) reg(L->W_2[l], Cm, Cm);
        }
    }
    if (!ph2_map_.empty()) {
        CUDA_CHECK_THROW(cudaStreamSynchronize(nullptr));
        hgemm2_row_init();
        polar_h2_ = true;
    }
}

GenericModel::~GenericModel() {
    for (void* p : ph2_allocs_) cudaFree(p);
    ph2_allocs_.clear();
    ph2_map_.clear();
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
    freep(d_energy_sink_);
    freep(d_zbl_rc_);
    freep(d_lin_amax_);
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
        // S125: MP-0 medium's second interaction has W_lin_1..3 exactly zero (about 5.5e-94 in the float64 checkpoint,
        // 0 after its float32 conversion), so only 2 of its 10 conv paths reach the answer. Certified here from the
        // loaded weights, so other checkpoints keep every path. LOUP_CONV_PRUNE=0 disables it (tests, A/B).
        if (!L->first_hidden && L->lmax_x == 1) {
            const char* pe = std::getenv("LOUP_CONV_PRUNE");
            bool dead = !(pe && pe[0] == '0');
            for (int l = 1; l < 4 && dead; ++l)
                for (float v : b.f32(p + "W_lin_" + std::to_string(l), {L->n_l3[l] * L->C_x, L->lin_cols[l]}))
                    if (v != 0.0f) { dead = false; break; }
            L->conv_prune = dead;
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
            for (int l = 0; l < 4; ++l) {
                const std::vector<float> w2 = b.f32(p + "W_2_" + std::to_string(l), {L->C_msg, L->C_msg});
                L->w2h_exp[l] = hgemm2_weight_exponent(w2.data(), (int64_t)w2.size());  // LOUP_W2T_HALF's scale
                L->W_2[l] = upload(W, w2);
            }
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
            // One table for every pair when the bundle's per-pair tables are byte-identical (S116): the radial chain
            // does not read the species (MACE-OFF), so the copies only cost memory and cache footprint.
            const size_t pair_floats = (size_t)L->intervals * L->Q1 * 4;
            bool one_table = Ptab > 1;
            for (int pr = 1; pr < Ptab && one_table; ++pr)
                one_table = memcmp(tab.data(), tab.data() + pr * pair_floats, pair_floats * sizeof(float)) == 0;
            if (one_table) L->pair_sym = 2;
            const int Pt = one_table ? 1 : Ptab;
            if (L->first_hidden) {
                // S128: the density column (the last) only where the layer reads the density (kinds 2-4); the
                // agnostic kinds never form D, so their rows hold the J - 1 hidden columns alone
                L->Jt = L->kind >= 2 ? L->J : L->J - 1;
                const size_t PI = (size_t)Pt * L->intervals;
                const float4* src = reinterpret_cast<const float4*>(tab.data());
                std::vector<float4> host(PI * L->Jt);
                for (size_t pi = 0; pi < PI; ++pi) {
                    const int Jp = L->J - 1, Jh = (Jp + 1) / 2;  // S131: even columns first, then the odd ones
                    for (int j = 0; j < Jp; ++j)
                        host[pi * L->Jt + ((j & 1) ? Jh + (j >> 1) : (j >> 1))] = src[pi * L->Q1 + j];
                    if (L->Jt == L->J) host[pi * L->Jt + L->J - 1] = src[pi * L->Q1 + L->Q1 - 1];
                }
                L->table = upload(W, host);
            } else {
                const int Q = L->n_paths * L->C_x;
                const size_t PI = (size_t)Pt * L->intervals;
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
            L->prod_species = mi("prod_species");
            L->prod_segw = mi("prod_segments_w");
            // The library's own kernels, else a kernel plugin (loupmd build-kernels), else, JIT_DESIGN.md stage 3,
            // the bundle's source compiled here (LOUP_JIT_PRODUCTS=1: every product so, for tests and gates).
            std::string no_plugin;
            L->prod = find_product_kernels(sig, no_plugin);
            const char* fp = std::getenv("LOUP_JIT_PRODUCTS");
            const bool force = fp && fp[0] == '1';
            std::string why;
            if ((!L->prod || force) && jit::level() > 0 && b.has(p + "prod_src")) {
                const auto& raw = b.entry(p + "prod_src").data;
                const GProdKernels* k =
                    jit_product(sig, L->lmax_out, L->prod_segw, std::string(raw.begin(), raw.end()), &why);
                if (k) L->prod = k;
                else if (L->prod)
                    fprintf(stderr, "[loupmd] JIT (LOUP_JIT=0 turns it off): %s; using the built-in kernels\n",
                            why.c_str());
            } else if (!L->prod) {
                why = !b.has(p + "prod_src") ? "the bundle has no product source (exported before JIT_DESIGN.md stage 3)"
                                             : "LOUP_JIT=0";
            }
            if (!L->prod)
                throw std::runtime_error("generic bundle: no compiled product kernel for signature " + sig +
                                         ": this library was not built with this model's kernels, " + no_plugin +
                                         ", and the run-time compiler did not build them (" + why +
                                         "). `loupmd build-kernels MODEL` builds them (needs nvcc)");
            if (L->prod == find_gprod(sig.c_str())) L->prod_h2src = find_gprod_h2(sig.c_str());
            if (L->prod->lmax_out != L->lmax_out) throw std::runtime_error("generic bundle: product degree mismatch");
            if (L->prod_segw != L->prod->segments_w) throw std::runtime_error("generic bundle: product segments mismatch");
            const int Sp = L->prod_species ? S_ : 1;
            std::vector<float> h_prodw = b.f32(p + "prod_w", {Sp, L->prod_segw, L->C_msg});
            std::vector<float> h_wp0, h_roW1;
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
                    h_roW1 = fold_mat(h_wp0.data(), L->C_out, L->C_out, row1.data(), L->ro_H);
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
                    h_roW1 = row1;
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
            // S130 / S132: compact live channels of a kind-1 layer: the terminal one on the fused product + readout path
            // (gprod_ro1c; the dead channels dropped) or a non-terminal one (gprod_fwdc / bwdc: outputs written in the
            // dense layout, so the consumers keep their weights). The projection weights (C0 for layer 0, else W_lin)
            // were uploaded dense above and are replaced by their compact columns.
            const char* lc = std::getenv("LOUP_LIN_COMPACT");
            const bool fused = L->ro_folded && L->readout == 1 && L->ro_H == 16 && L->C_out <= 512;
            const bool have_kernels = fused ? (L->prod->ro1c_128 != nullptr && !L->first_hidden)
                                            : (!L->ro_folded && L->prod->fwdc != nullptr);
            CompactLayout cl;
            if (!(lc && lc[0] == '0') && L->kind == 1 && !L->conv_prune && have_kernels) {
                const int C = L->C_msg;
                std::vector<float> proj[4];
                int K[4];
                for (int l = 0; l < 4; ++l) {
                    K[l] = L->first_hidden ? L->K0 : L->n_l3[l] * L->C_x;
                    proj[l] = L->first_hidden ? c0_pad[l] : b.f32(p + "W_lin_" + std::to_string(l), {K[l], C});
                }
                if (compact_layout(C, proj, K, h_prodw, Sp, L->prod_segw, fused ? &h_roW1 : nullptr, L->ro_H, !fused,
                                   cl)) {
                    const int kept = cl.kept, W0 = cl.width[0], ncols = fused ? W0 : C, npw = fused ? kept : C;
                    std::vector<float> pw((size_t)Sp * L->prod_segw * ncols, 0.0f);
                    for (size_t r = 0; r < (size_t)Sp * L->prod_segw; ++r)
                        for (int j = 0; j < npw; ++j) pw[r * ncols + j] = h_prodw[r * C + cl.order[j]];
                    h_prodw.swap(pw);
                    if (fused) {
                        // readout rows up to the kernel's blockDim (W0 rounded up to 32), zero beyond the kept channels
                        std::vector<float> rw((size_t)((W0 + 31) / 32 * 32) * L->ro_H, 0.0f);
                        for (int j = 0; j < kept; ++j)
                            memcpy(&rw[(size_t)j * L->ro_H], &h_roW1[(size_t)cl.order[j] * L->ro_H],
                                   L->ro_H * sizeof(float));
                        h_roW1.swap(rw);
                    } else {
                        L->d_order = upload(W, cl.order);
                    }
                    L->d_pos = upload(W, cl.pos);
                    for (int l = 0; l < 4; ++l) {
                        const int Wl = cl.width[l];
                        std::vector<float> wc((size_t)K[l] * Wl, 0.0f);
                        for (int j = 0; j < (l == 0 ? kept : cl.n_ang); ++j) {
                            const int col = l == 0 ? j : cl.pos[(size_t)(l - 1) * cl.n_ang + j];
                            if (col < 0) continue;
                            for (int64_t k = 0; k < K[l]; ++k) wc[k * Wl + col] = proj[l][k * C + cl.order[j]];
                        }
                        float*& dst = L->first_hidden ? L->C0[l] : L->W_lin[l];
                        auto it = std::find(W.begin(), W.end(), (void*)dst);
                        if (it != W.end()) W.erase(it);
                        CUDA_CHECK_THROW(cudaFree(dst));
                        dst = upload(W, wc);
                        L->lin_cols[l] = Wl;
                    }
                    L->n_ang = cl.n_ang;
                    L->kept = kept;
                    L->lin_compact = true;
                    if (std::getenv("LOUP_DEBUG_COMPACT"))
                        fprintf(stderr, "compact L%d (%s): kept %d of %d (angular %d), widths %d / %d / %d / %d\n", t,
                                fused ? "fused readout" : "dense outputs", kept, C, cl.n_ang, cl.width[0], cl.width[1],
                                cl.width[2], cl.width[3]);
                }
            }
            L->prod_w = upload(W, h_prodw);
            if (!h_roW1.empty()) L->ro_W1 = upload(W, h_roW1);
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
    // S64 test override: LOUP_CONV_FWD_VARIANT=v forces conv forward
    // variant v on every layer that has it, without tuning (gates of each variant; see tune_conv_variants)
    conv_var_override_ = -1;
    if (const char* e = std::getenv("LOUP_CONV_FWD_VARIANT"); e && e[0] >= '0' && e[0] <= '9') conv_var_override_ = e[0] - '0';
    // LOUP_CONV_TUNE, read here (per engine) like the forced variant: 0 never times, 1 times on any GPU (tests)
    conv_tune_env_ = -1;
    if (const char* e = std::getenv("LOUP_CONV_TUNE"); e && (e[0] == '0' || e[0] == '1')) conv_tune_env_ = e[0] - '0';
    setup_chain_fold(b);
    setup_lr_fold(b);
    setup_jit();
    setup_polar_h2();
    setup_w2t_half();
    setup_w2f_half();
    setup_lin_half();
    if (std::getenv("LOUP_SWEEP_REPLAY")) {
        if (lr_)
            throw std::runtime_error("LOUP_SWEEP_REPLAY: the sweep schedule is not available for long-range models");
        sweep_replay_ = true;
        chain_fold_ = chain_fold_built_ = false;  // the sweeps hand on every layer's output, which the fold never forms
    }
    // tile-shared-layers (alloc_system): LOUP_TILE_SHARE_LAYERS=0 keeps every layer's buffers its own (tests, A/B)
    share_layers_env_ = true;
    if (const char* e = std::getenv("LOUP_TILE_SHARE_LAYERS"); e && e[0] == '0') share_layers_env_ = false;
    if (const char* e = std::getenv("LOUP_SWEEP_ZERO"); e && e[0] == '0') sweep_zero_ = false;
    if (const char* e = std::getenv("LOUP_GATE_FWD_TILED"); e && e[0] == '0') gate_fwd_tiled_ = false;
    CUDA_CHECK_THROW(cudaMalloc(&d_energy_sink_, sizeof(double)));
    update_h0_read();
}

// S97: whether a pass reads layer 0's input rows h_in[0] (the species embedding gathered per atom): the x products of
// a hidden-table layer 0 without the S59 residual table, the skip / readout terms of an unfolded layer 0, and the S57
// fold's skip terms without the S59 tables (Wh). With the fold and the tables every consumer reads a species table
// instead, and the gather is skipped.
void GenericModel::update_h0_read() {
    const GenericLayer* L0 = layers_[0];
    h0_read_ = n_layers_ == 1 || ((!L0->first_hidden || L0->kind == 4) && !L0->res0_table);
    if (chain_fold_) {
        for (const auto& ft : L0->fold[0]) h0_read_ = h0_read_ || ft.Wh != nullptr;
    } else {
        h0_read_ = h0_read_ || L0->sc_has[0];
    }
}

void GenericModel::use_chain_fold(bool on, bool sweeps) {
    if (N_ > 0) throw std::logic_error("generic: use_chain_fold with a system allocated");
    // list 20: a domain folds only with two layers, whose sweeps then hand layer 1's x and the readout hidden over
    chain_fold_ = on && chain_fold_built_ && (!sweeps || n_layers_ == 2);
    fold_sweeps_ = sweeps && chain_fold_;
    update_h0_read();
}

void GenericModel::share_layer_buffers(bool on) {
    if (N_ > 0) throw std::logic_error("generic: share_layer_buffers with a system allocated");
    share_layers_ = on;
}

// JIT_DESIGN.md stage 1: with LOUP_JIT=1, layer 0's edge kernels compiled for this model (S, intervals, J, Jt, K0,
// sh_perm, pair_sym and the spline grid as constants, kernel_ir.cu), launched instead of the ahead-of-time ones with
// the same arguments. Any failure (no NVRTC, a compile error, a cubin the driver refuses) keeps the ahead-of-time
// kernels, with one message on stderr.
// LOUP_W2T_HALF (hgemm2.cu; generic.hpp; default on compute capability 12.0, where it was measured, 0 / 1 force it):
// every kind-4 layer whose product has the library's own _h2 kernels, with neither the compact layout nor the long
// range and C_msg % 64 == 0, gets its W_2[l] panels and the _h2 table; record then stores its dmsgf as two binary16
// panels and runs its W_2^T GEMMs through GenericPassInputs::gemm_h2_gate.
static bool two_half_switch(const char* name) {
    const char* e = std::getenv(name);
    if (e && (e[0] == '0' || e[0] == '1')) return e[0] == '1';
    return device_is_tuned_generic_cc();
}
void GenericModel::setup_w2t_half() {
    w2t_half_ = false;
    if ((lr_ && !polar_h2_) || !two_half_switch("LOUP_W2T_HALF")) return;
    for (auto* L : layers_) {
        const GProdH2Kernels* k = L->prod_h2src;
        if (L->kind != 4 || !k || L->lin_compact || L->C_msg % 64 != 0 || (L->lmax_out == 0 && !k->ro1_128)) continue;
        L->prod_h2 = *L->prod;
        L->prod_h2.bwd = k->bwd;
        L->prod_h2.ro1_128 = k->ro1_128;
        L->prod_h2.ro1_256 = k->ro1_256;
        L->prod_h2.ro1_512 = k->ro1_512;
        const int64_t n = (int64_t)L->C_msg * L->C_msg;
        for (int l = 0; l < 4; ++l) {
            CUDA_CHECK_THROW(cudaMalloc(&L->w2h[l], 2 * n * sizeof(uint16_t)));
            L->weights.push_back(L->w2h[l]);
            hgemm2_split(n, L->W_2[l], L->w2h_exp[l], L->w2h[l], nullptr);
            L->w2h_oscale[l] = std::ldexp(1.0f, -(12 + L->w2h_exp[l]));  // 2^12: loup_store_h2's scale of dmsgf
        }
        L->h2 = true;
        w2t_half_ = true;
    }
    CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaStreamSynchronize(nullptr));
    if (w2t_half_) hgemm2_init();
}

void GenericModel::debug_w2t_half(int N) const {
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    for (size_t t = 0; t < layers_.size(); ++t) {
        const GenericLayer* L = layers_[t];
        if (!L->h2) continue;
        for (int l = 0; l < 4; ++l) {
            const int64_t n = (int64_t)N * dimL(l) * L->C_msg;  // one panel; the second follows it
            std::vector<__half> p((size_t)(2 * n));
            CUDA_CHECK_THROW(cudaMemcpy(p.data(), L->dmsgf[l], (size_t)(2 * n) * sizeof(__half), cudaMemcpyDeviceToHost));
            float mh = 0.0f, mv = 0.0f;
            int64_t bad = 0;
            for (int64_t i = 0; i < n; ++i) {
                const float a = __half2float(p[(size_t)i]), b = __half2float(p[(size_t)(n + i)]);
                if (!std::isfinite(a) || !std::isfinite(b)) {
                    ++bad;
                    continue;
                }
                mh = std::max(mh, std::fabs(a));
                mv = std::max(mv, std::fabs(b));
            }
            std::fprintf(stderr,
                         "loupmd: LOUP_W2T_HALF layer %zu degree %d: max|dmsgf| %.5g (binary16 limit %.5g at 2^12), "
                         "largest panel element %.5g of 65504, %lld non-finite\n",
                         t, l, mh / 4096.0f, 65504.0f / 4096.0f, std::max(mh, mv), (long long)bad);
        }
    }
}

bool GenericModel::drop_w2t_half() {
    if (!w2t_half_ && !w2f_half_ && !lin_half_ && !polar_h2_) return false;
    for (auto* L : layers_) {
        L->h2 = L->f2 = false;
        for (auto& p : L->hl_flin) p.B = nullptr;
        for (auto& p : L->hl_fres) p.B = nullptr;
        for (auto& p : L->hl_blin) p.B = nullptr;
        for (auto& p : L->hl_bres) p.B = nullptr;
        for (auto& p : L->hl_bcat) p.B = nullptr;
        for (auto& v : L->hs_flin) v.clear();
        for (auto& v : L->hs_blin) v.clear();
        for (auto& v : L->fold)
            for (auto& ft : v) ft.hf.B = ft.hb.B = nullptr;
    }
    w2t_half_ = w2f_half_ = lin_half_ = polar_h2_ = false;
    if (lr_) lr_->disable_h2_row();
    return true;
}

// LOUP_LIN_HALF (hgemm2.cu, hgemm2_f32a; ledger gen-lin-two-half; default as LOUP_W2T_HALF, on compute capability 12.0
// since 21 November 2026, 0 / 1 force it): the conv layers' plain projections of kind-4 layers (MH-1's and OMOL-0's
// later layers; neither compact, folded, pruned nor long-range) and of later kind 1 / 3 layers on the cross-term
// two-half product of LOUP_W2F_HALF, A read as FP32 and split in the kernel at a fixed scale per projection kind
// (kLinHalfExp: forward lin_l from mid_l, forward res_l from x_l, backward dmid_l from dlin_l, then layer 0's S0,
// dpre, layer 0's dlin, kind 1 / 3 layers' mid and dlin; LOUP_LIN_HALF_EXP=a,b,c,d,e,f,g,h overrides a prefix), B's
// panels made here. Layer 0 (first hidden, kind 4, not folded): forward lin_l = S0_l C0_l and backward dS0_l = dlin_l
// C0_l^T (K0 and N0 = K0 need the kernel's tails: N % 16, K % 16, padded panels), with their own scales
// (kLinHalfExp[3] forward, [5] backward); later kind-4 layers also the backward dx_l = dpre_l W_res[l]^T (kLinHalfExp[4];
// LOUP_RES_BWD in setup_lin_half: by default from dlin_l instead). An element with |A| 2^e >= 65,520
// makes the pass non-finite, and the engine then drops every two-half path (drop_w2t_half).
// LOUP_DEBUG_LIN_AMAX=1 records each such projection's largest |A| (lin_amax_record), FP32 path included.
// The scales leave at least 2x headroom over the 21 November census (MH-1 physics headroom set with its contact cells,
// the seven, the three heads' gate sets; OMOL-0 XL organics): largest |mid| 451 (overflow at 1,024), |x| 7.5 (16),
// |dlin| 0.17 (0.5); layer 0 |S0| 62 (1,024) and |dlin| 0.33 (1.0); |dpre| 3.5 (16).
// Later kind 1 / 3 layers (list 13 idea 4; OMAT-0, MP-0b3, MPA-0, MatPES, the OFF families: neither compact, pruned
// nor first hidden), forward lin_l and backward dmid_l only, at scales of their own (kLinHalfExp[6], [7]) from the
// census of their gate sets and of the physics headroom cells (OMAT-0 medium / small, MPA-0, MatPES): |mid| 29.7 / 131
// (MPA-0; overflow at 512), |dlin| 1.56 / 10.3 (OMAT-0 small; overflow at 32).
// A skip-folded kind 1 / 3 layer 0's species-grouped S0 C0 and dS0 (list 24 item 4, bit 128): kLinHalfExp[8] (S0) and
// [9] (dmsgf), from list 24's census (the kind 1 / 3 families' gate sets, the crystal families' physics headroom cells
// and the seven): largest |S0| 126 (OMAT-0 medium headroom; overflow at 1,024), |dmsgf| 47 (OMAT-0 small headroom;
// overflow at 128); the OFF families' gate sets 18 and 3.8.
// The chain fold's composed products (list 25 item 1, bit 256): kLinHalfExp[10] (y_t, forward) and [11] (the targets'
// gradients dx, backward), from list 25's census (every chain-folded family's gate sets; the physics headroom cells and
// the seven of MH-1, OMAT-0 medium / small, MPA-0, MatPES): largest |y| 83 (MH-1 headroom; overflow at 256), |dx| 402
// (OMAT-0 small headroom; overflow at 1,024); the gate sets 14 and 2.8.
static constexpr int kLinHalfExp[12] = {6, 12, 17, 6, 12, 16, 7, 11, 6, 9, 8, 6};  // mid x dlin S0 dpre, layer 0 dlin; kind 1/3 mid dlin; folded layer 0 S0 dmsgf; fold y, dx
// Bit 128's fewest rows (atoms of the species x (2l + 1)) on hgemm2f_kernel: one 128-row tile. Fewer keep the FP32
// GEMM (OMAT-0 medium NaCl216 in the MD loop, 16 runs: 0 rows 1.009 by medians, 128 rows 0.993, the null A / B 0.989).
static constexpr int kLinHalfSpeciesRows = 128;
static int weight_exponent_dev(const float* w, int64_t n) {
    std::vector<float> h((size_t)n);
    CUDA_CHECK_THROW(cudaMemcpy(h.data(), w, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    return hgemm2_weight_exponent(h.data(), n);
}
void GenericModel::setup_lin_half() {
    lin_half_ = false;
    // LOUP_RES_BWD (list 13 idea 1; tests, benches): the backward dx_l of a later kind-4 layer's residual. 0: the
    // forward conv kernel zeroes dx's rows and W_res^T adds dpre_l W_res[l]^T to them (beta 1); 1: W_res^T assigns
    // (beta 0) and the forward leaves dx alone; 2: as 1, and where every degree's W_res^T has its two-half panels,
    // from dlin_l (dpre = dlin (D beta + alpha) per atom, in the product's epilogue), so the gate backward writes no
    // dpre; 3 (default): as 2, with dmid_l and dx_l of a degree in one product dlin_l [W_lin[l] | W_res[l]]^T
    // (hl_bcat, aligned shapes).
    if (const char* rb = std::getenv("LOUP_RES_BWD"); rb && *rb) res_bwd_ = std::atoi(rb);
    if (std::getenv("LOUP_DEBUG_LIN_AMAX") && !d_lin_amax_) {
        CUDA_CHECK_THROW(cudaMalloc(&d_lin_amax_, kLinAmaxSlots * sizeof(unsigned)));
        CUDA_CHECK_THROW(cudaMemset(d_lin_amax_, 0, kLinAmaxSlots * sizeof(unsigned)));
    }
    if (lr_ || !two_half_switch("LOUP_LIN_HALF")) return;
    int ex[12];
    for (int i = 0; i < 12; ++i) ex[i] = kLinHalfExp[i];
    if (const char* xe = std::getenv("LOUP_LIN_HALF_EXP"); xe && *xe)
        std::sscanf(xe, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", &ex[0], &ex[1], &ex[2], &ex[3], &ex[4], &ex[5], &ex[6],
                    &ex[7], &ex[8], &ex[9], &ex[10], &ex[11]);
    // LOUP_LIN_HALF_PARTS (tests, benches): bit 0 the later layers' projections, bit 1 layer 0's backward dS0, bit 2
    // the backward W_res^T, bit 3 layer 0's forward S0 C0, bit 4 the gated store on hgemm2f_kernel (without it a
    // degree with panels runs the plain product and gate_fwd_kernel), bit 5 the kind 1 / 3 layers' projections, bit 7
    // a skip-folded kind 1 / 3 layer 0's species-grouped S0 C0 and dS0 (ledger gen-lin-half-l0-species), bit 8 (256,
    // list 25) the chain fold's composed products (ledger gen-lin-half-fold); bit 6 (64, list 27 item 3, rform v3:
    // batched lin_l loads + L2 prefetch in W_res's gated epilogue, replacing lists 23 / 25's rejected scalar loads).
    // Default 511 (every bit) since list 27 item 3 (5060 Ti only: MH-1 SrTiO3 0.9924, OMOL-0 XL water216 MD 0.9899,
    // MH-1 NaCl216 MD 0.9995); 447 in list 25 (OMAT-0 medium SrTiO3 0.989, OMOL-0 XL water216's MD loop 0.995, MH-1
    // NaCl216's a tie with the tile floor); 191 in list 24 (the MD loop of OMAT-0 medium NbCl5 0.963, NaCl 0.971,
    // SrTiO3 0.993, OFF23-M / -S water216 0.977 / 0.976); 63 in list 23 (MH-1 SrTiO3 0.997, OMOL-0 XL 0.997, NaCl216's
    // MD loop a tie); until then 55, without bit 3, whose tails had cost MH-1 SrTiO3 1.4% against sgemm128's gated
    // store (ledger gen-lin-two-half)
    int parts = 511;
    if (const char* pe = std::getenv("LOUP_LIN_HALF_PARTS"); pe && *pe) parts = std::atoi(pe);
    lin_half_parts_ = parts;
    // bit 128 runs a species' rows of a degree on hgemm2f_kernel only from this many rows; fewer keep the FP32 GEMM
    // (LOUP_LIN_HALF_SPECIES_ROWS overrides)
    lin_half_srows_ = kLinHalfSpeciesRows;
    if (const char* se = std::getenv("LOUP_LIN_HALF_SPECIES_ROWS"); se && *se) lin_half_srows_ = std::atoi(se);
    auto make = [&](GenericLayer::H2Proj& p, const float* W, int Nc, int K, bool transpose, int ea) {
        if (!W || Nc % 16 != 0 || K % 16 != 0) return;
        const int ew = weight_exponent_dev(W, (int64_t)Nc * K);
        CUDA_CHECK_THROW(cudaMalloc(&p.B, 2 * hgemm2_pad_elems(Nc, K) * sizeof(uint16_t)));
        hgemm2_split_pad(Nc, K, W, transpose, ew, p.B, nullptr);  // transpose: W [K][Nc], the panels of W^T
        p.sA = std::ldexp(1.0f, ea);
        p.os = std::ldexp(1.0f, -(ea + ew));
        lin_half_ = true;
    };
    for (auto* L : layers_) {
        // bit 7 (128, list 24 item 4): a skip-folded kind 1 / 3 layer 0 (OMAT-0 and the other per-species-skip
        // families), its species-grouped S0_l C0_l[s] and dS0_l = dmsgf_l C0_l[s]^T, one panel set per species
        if ((parts & 128) && L->skip_folded && L->first_hidden && !L->lin_compact && L->kind != 4) {
            const int Kh = L->K0, Cm = L->C_msg;
            for (int l = 0; l < 4; ++l) {
                if (!L->C0[l]) continue;
                L->hs_flin[l].assign(S_, GenericLayer::H2Proj{});
                L->hs_blin[l].assign(S_, GenericLayer::H2Proj{});
                for (int s = 0; s < S_; ++s) {
                    const float* W = L->C0[l] + (size_t)s * Kh * Cm;
                    make(L->hs_flin[l][s], W, Cm, Kh, true, ex[8]);
                    make(L->hs_blin[l][s], W, Kh, Cm, false, ex[9]);
                }
            }
            if (d_lin_amax_)
                std::fprintf(stderr, "loupmd: lin half: layer 0's species-grouped projections (bit 128), K0 %d, C %d, "
                                     "%d species\n", Kh, Cm, S_);
            continue;
        }
        if (L->lin_compact || L->skip_folded || L->conv_prune) continue;
        if (L->kind != 4) {
            // bit 5 (list 13 idea 4): a kind 1 / 3 layer's lin_l = mid_l W_lin[l] (msgf_l) and dmid_l = dlin_l
            // W_lin[l]^T; these layers have no gate and no W_res
            if ((parts & 32) && !L->first_hidden && (L->kind == 1 || L->kind == 3))
                for (int l = 0; l < 4; ++l) {
                    const int Kl = L->n_l3[l] * L->C_x;
                    make(L->hl_flin[l], L->W_lin[l], L->lin_cols[l], Kl, true, ex[6]);
                    make(L->hl_blin[l], L->W_lin[l], Kl, L->lin_cols[l], false, ex[7]);
                }
            continue;
        }
        const int Cx = L->C_x;
        if (L->first_hidden) {
            for (int l = 0; l < 4; ++l) {
                if (parts & 8) make(L->hl_flin[l], L->C0[l], L->lin_cols[l], L->K0, true, ex[3]);
                if (parts & 2) make(L->hl_blin[l], L->C0[l], L->K0, L->lin_cols[l], false, ex[5]);
            }
            continue;
        }
        for (int l = 0; l < 4; ++l) {
            const int Kl = L->n_l3[l] * Cx;
            if (parts & 1) {
                make(L->hl_flin[l], L->W_lin[l], L->lin_cols[l], Kl, true, ex[0]);
                make(L->hl_blin[l], L->W_lin[l], Kl, L->lin_cols[l], false, ex[2]);
                if (l <= L->lmax_x && l < 3) make(L->hl_fres[l], L->W_res[l], L->lin_cols[l], Cx, true, ex[1]);
            }
            if ((parts & 4) && l <= L->lmax_x && l < 3) make(L->hl_bres[l], L->W_res[l], Cx, L->lin_cols[l], false, ex[4]);
            // LOUP_RES_BWD 3: one product dlin_l [W_lin[l] | W_res[l]]^T for dmid_l and dx_l, from the two panel sets
            // (shapes without the kernel's tails only: the panels are unpadded, so they stack row-wise)
            const int Kc = L->lin_cols[l];
            if (res_bwd_ >= 3 && l <= L->lmax_x && l < 3 && L->hl_blin[l].B && L->hl_bres[l].B && Kl % 64 == 0 &&
                Cx % 64 == 0 && Kc % 32 == 0) {
                GenericLayer::H2Proj& c = L->hl_bcat[l];
                const size_t pl = (size_t)Kl * Kc, pr = (size_t)Cx * Kc, h = sizeof(uint16_t);
                CUDA_CHECK_THROW(cudaMalloc(&c.B, 2 * (pl + pr) * h));
                auto* d = static_cast<char*>(c.B);
                const auto* bl = static_cast<const char*>(L->hl_blin[l].B);
                const auto* br = static_cast<const char*>(L->hl_bres[l].B);
                for (int q = 0; q < 2; ++q) {
                    CUDA_CHECK_THROW(cudaMemcpy(d + q * (pl + pr) * h, bl + q * pl * h, pl * h, cudaMemcpyDeviceToDevice));
                    CUDA_CHECK_THROW(cudaMemcpy(d + (q * (pl + pr) + pl) * h, br + q * pr * h, pr * h,
                                                cudaMemcpyDeviceToDevice));
                }
                c.sA = L->hl_blin[l].sA;
                c.os = L->hl_blin[l].os;
            }
        }
    }
    for (auto* L : layers_)
        for (auto* ps : {&L->hl_flin[0], &L->hl_flin[1], &L->hl_flin[2], &L->hl_flin[3], &L->hl_blin[0],
                         &L->hl_blin[1], &L->hl_blin[2], &L->hl_blin[3], &L->hl_fres[0], &L->hl_fres[1],
                         &L->hl_fres[2], &L->hl_bres[0], &L->hl_bres[1], &L->hl_bres[2], &L->hl_bcat[0],
                         &L->hl_bcat[1], &L->hl_bcat[2]})
            if (ps->B) L->weights.push_back(ps->B);
    // bit 8 (256, list 25 item 1): the chain fold's composed products (S57) of every folded layer, forward dst +=
    // y_t[l] Wy and backward dy_t[l] += dsrc Wy^T, for each target with one weight (ns 1) and at least 64 columns (the
    // readout hidden's 16 stay FP32)
    if (parts & 256) {
        // a product goes to the kernel only from LOUP_LIN_HALF_FOLD_TILES of its 128 x 64 tiles (default the SM count:
        // MH-1 NaCl216's fold products, 216-648 rows, lost 1% in the MD loop on a few tiles); fewer keep the FP32 GEMM
        int dev = 0, sms = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
            sms = 0;
        lin_half_fold_tiles_ = sms;
        if (const char* te = std::getenv("LOUP_LIN_HALF_FOLD_TILES"); te && *te) lin_half_fold_tiles_ = std::atoi(te);
        int nf = 0;
        for (int t = 0; t < (int)layers_.size(); ++t) {
            GenericLayer* L = layers_[t];
            for (int l = 0; l < 3; ++l)
                for (auto& ft : L->fold[l]) {
                    if (ft.dest_layer == -2 || ft.ns != 1 || !ft.Wy || ft.cols < 64) continue;
                    make(ft.hf, ft.Wy, ft.cols, L->C_out, true, ex[10]);
                    make(ft.hb, ft.Wy, L->C_out, ft.cols, false, ex[11]);
                    nf += ft.hf.B != nullptr;
                    if (d_lin_amax_ && ft.hf.B)
                        std::fprintf(stderr, "loupmd: lin half: fold layer %d degree %d -> %d: %d columns, C_out %d\n",
                                     t, l, ft.dest_layer, ft.cols, L->C_out);
                }
        }
        if (d_lin_amax_)
            std::fprintf(stderr, "loupmd: lin half: %d chain-fold products (bit 256), from %d tiles\n", nf,
                         lin_half_fold_tiles_);
    }
    for (auto* L : layers_) {
        for (int l = 0; l < 4; ++l)
            for (auto* v : {&L->hs_flin[l], &L->hs_blin[l]})
                for (const auto& p : *v)
                    if (p.B) L->weights.push_back(p.B);
        for (const auto& v : L->fold)
            for (const auto& ft : v)
                for (const auto* p : {&ft.hf, &ft.hb})
                    if (p->B) L->weights.push_back(p->B);
    }
    CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaStreamSynchronize(nullptr));
    if (lin_half_) hgemm2_init();
}

__global__ void lin_amax_kernel(const float* __restrict__ a, int64_t n, unsigned* __restrict__ out) {
    float m = 0.0f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(a[i]));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(m));
}

void GenericModel::lin_amax_record(const float* A, int64_t n, int slot, cudaStream_t st) const {
    if (d_lin_amax_ && slot >= 0 && slot < kLinAmaxSlots) lin_amax_kernel<<<128, 256, 0, st>>>(A, n, d_lin_amax_ + slot);
}

void GenericModel::debug_lin_amax() const {
    if (!d_lin_amax_) return;
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    std::vector<unsigned> v(kLinAmaxSlots);
    CUDA_CHECK_THROW(cudaMemcpy(v.data(), d_lin_amax_, kLinAmaxSlots * sizeof(unsigned), cudaMemcpyDeviceToHost));
    static const char* kind[4] = {"forward lin (mid; layer 0 S0)", "forward res (x)", "backward dmid (dlin)",
                                  "backward dx (dpre)"};
    for (int s = 0; s < kLinAmaxSlots; ++s) {
        if (!v[s]) continue;
        float m;
        std::memcpy(&m, &v[s], sizeof m);
        if (s >= 128)  // list 25 item 1: the chain fold's products, 128 + 16 t + l (y_t) and + 8 (dsrc)
            std::fprintf(stderr, "loupmd: lin amax layer %d fold %s degree %d: %.5g\n", (s - 128) / 16,
                         s % 16 < 8 ? "forward (y)" : "backward (dx)", s % 8, m);
        else
            std::fprintf(stderr, "loupmd: lin amax layer %d %s degree %d: %.5g\n", s / 16, kind[s % 16 / 4], s % 4, m);
    }
}

// LOUP_W2F_HALF (hgemm2.cu, hgemm2_f32a; ledger gen-w2-fwd-two-half; default as LOUP_W2T_HALF): every kind-4 layer
// with neither the compact layout nor the long range and C_msg % 64 == 0 runs its forward W_2 GEMMs
// (msgf_l = gout_l W_2[l]) as the cross-term two-half product with gout_l read as FP32 and split in the kernel at the fixed scale 2^kW2fExp
// (LOUP_W2F_HALF_EXP overrides), against the panels of W_2[l]^T made here. Nothing else changes: the producers of
// gout and the gate backward, which reads it, keep FP32. An element with |gout| 2^e >= 65,520 makes the pass
// non-finite, and the engine then drops the path with LOUP_W2T_HALF's (drop_w2t_half).
static constexpr int kW2fExp = 10;
void GenericModel::setup_w2f_half() {
    w2f_half_ = false;
    if ((lr_ && !polar_h2_) || !two_half_switch("LOUP_W2F_HALF")) return;
    const char* xe = std::getenv("LOUP_W2F_HALF_EXP");
    const int ea = xe && *xe ? std::atoi(xe) : kW2fExp;
    for (auto* L : layers_) {
        if (L->kind != 4 || L->lin_compact || L->C_msg % 64 != 0) continue;
        const int64_t n = (int64_t)L->C_msg * L->C_msg;
        for (int l = 0; l < 4; ++l) {
            CUDA_CHECK_THROW(cudaMalloc(&L->w2fh[l], 2 * n * sizeof(uint16_t)));
            L->weights.push_back(L->w2fh[l]);
            hgemm2_split_t(L->C_msg, L->C_msg, L->W_2[l], L->w2h_exp[l], L->w2fh[l], nullptr);
            L->w2f_sA[l] = std::ldexp(1.0f, ea);
            L->w2f_oscale[l] = std::ldexp(1.0f, -(ea + L->w2h_exp[l]));
        }
        L->f2 = true;
        w2f_half_ = true;
    }
    CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaStreamSynchronize(nullptr));
    if (w2f_half_) hgemm2_init();
}

void GenericModel::debug_gout(int N) const {
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    for (size_t t = 0; t < layers_.size(); ++t) {
        const GenericLayer* L = layers_[t];
        if (L->kind != 4 || !L->gout[0]) continue;
        for (int l = 0; l < 4; ++l) {
            const int64_t n = (int64_t)N * dimL(l) * L->C_msg;
            std::vector<float> p((size_t)n);
            CUDA_CHECK_THROW(cudaMemcpy(p.data(), L->gout[l], (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
            float mx = 0.0f;
            double ss = 0.0;
            int64_t bad = 0;
            for (float x : p) {
                if (!std::isfinite(x)) {
                    ++bad;
                    continue;
                }
                mx = std::max(mx, std::fabs(x));
                ss += (double)x * x;
            }
            std::fprintf(stderr, "loupmd: gout layer %zu degree %d: max %.5g rms %.4g (%s), %lld non-finite\n", t, l, mx,
                         std::sqrt(ss / std::max<int64_t>(n, 1)), L->f2 ? "two-half forward" : "FP32 forward",
                         (long long)bad);
        }
    }
}

void GenericModel::setup_jit() {
    if (jit::level() == 0) return;
    const bool dbg = [] {
        const char* e = std::getenv("LOUP_DEBUG_JIT");
        return e && e[0] == '1';
    }();
    for (GenericLayer* L : layers_) {
        if (!L->first_hidden) {  // stage 2: conv layers whose width has no built-in instance
            ConvIR ci;
            ci.lx = L->lmax_x;
            ci.C = L->C_x;
            ci.prune = L->conv_prune;
            ci.fwd = conv_fwd_generic(ci.lx, ci.C);
            ci.bwd = conv_bwd_generic(ci.lx, ci.C, ci.prune);
            if (const char* e = std::getenv("LOUP_JIT_PAIR"); e && e[0] == '0') ci.pair = false;  // lab knob
            if (dbg)
                fprintf(stderr, "[loupmd jit] conv IR: lx %d C %d prune %d run-time-width fwd %d bwd %d\n", ci.lx, ci.C,
                        ci.prune ? 1 : 0, ci.fwd ? 1 : 0, ci.bwd ? 1 : 0);
            if (!ci.fwd && !ci.bwd) continue;
            int nf = 0, nb = 0;
            const std::vector<std::string> names = conv_jit_names(ci, &nf, &nb);
            std::string err;
            const std::vector<const void*> k =
                jit::build("conv", conv_jit_source(), jit::embedded_headers(), names, &err);
            if (k.size() != names.size()) {
                static bool told = false;
                if (!told) fprintf(stderr, "[loupmd] JIT (LOUP_JIT=0 turns it off): %s; using the built-in kernels\n", err.c_str());
                told = true;
                continue;
            }
            size_t i = 0;
            for (int f = 0; f < nf; ++f)
                for (int p = 0; p < 2; ++p) L->jit_conv.fwd[f][p] = k[i++];
            for (int b = 0; b < nb; ++b)
                for (int w = 0; w < 2; ++w)
                    for (int p = 0; p < 2; ++p) L->jit_conv.bwd[b][w][p] = k[i++];
            L->jit_conv_on = true;
            continue;
        }
        if (!jit::enabled() || L->J - 1 > 96) continue;  // stage 1 (LOUP_JIT=1); above 96 columns: first_bwd_kernel
        Layer0IR ir;
        ir.S = S_;
        ir.I = L->intervals;
        ir.J = L->J;
        ir.Jt = L->Jt;
        ir.K0 = L->K0;
        ir.sh_perm = sh_perm_;
        ir.pair_sym = L->pair_sym;
        ir.zbl = has_zbl_ != 0;
        ir.grid = L->grid;
        if (const char* e = std::getenv("LOUP_JIT_BAKE_GRID"); e && e[0] == '1') ir.bake_grid = true;  // lab knob
        if (const char* e = std::getenv("LOUP_DEBUG_JIT"); e && e[0] == '1')
            fprintf(stderr, "[loupmd jit] layer0 IR: S %d I %d J %d Jt %d K0 %d sh_perm %d pair_sym %d zbl %d bands %d\n",
                    ir.S, ir.I, ir.J, ir.Jt, ir.K0, ir.sh_perm, ir.pair_sym, ir.zbl ? 1 : 0, ir.grid.n_bands);
        std::string err;
        const std::vector<const void*> k =
            jit::build("layer0", layer0_jit_source(ir), jit::embedded_headers(), layer0_jit_names(ir), &err);
        if (k.size() != 3) {
            static bool told = false;
            if (!told) fprintf(stderr, "[loupmd] JIT (LOUP_JIT=0 turns it off): %s; using the built-in kernels\n", err.c_str());
            told = true;
            continue;
        }
        L->jit_fwd = k[0];
        L->jit_bwd_edge[0] = k[1];
        L->jit_bwd_edge[1] = k[2];
    }
}

// Launches the JIT kernel `k` with the parameter list of the ahead-of-time kernel `aot` (the same source), so the
// arguments convert exactly as at a <<< >>> launch.
template <typename T>
struct jit_arg_ident {
    using type = T;
};
template <typename... P>
static void launch_jit_as(void (*aot)(P...), const void* k, dim3 grid, dim3 block, size_t smem, cudaStream_t st,
                          typename jit_arg_ident<P>::type... a) {
    (void)aot;
    void* args[] = {(void*)&a...};
    CUDA_CHECK_THROW(cudaLaunchKernel(k, grid, block, args, smem, st));
}

// S59 species tables. Layer 0's input h_in[0] is the species embedding row, so every product of it alone is a
// per-species constant: the gated block's residual res[0] = (h_in[0] W_up[0]) W_res[0] (with x[0] used for nothing
// else in the hidden-table layer) and, under the S57 fold, the skip terms h_in[0] Wh. They become [S, cols] tables
// composed in double at load: the gate kernels read the residual row of each atom's species directly and the skip
// tables are gathered per atom: no x / res GEMMs in layer 0 and no Wh GEMMs. Layer 0 takes no
// gradient into h_in, so the backward is unchanged. The long-range models (POLAR-1) on cc 12.0 only (list 28, 5060 Ti
// only): S59 kept them on the GEMMs (POLAR-1-L measured no win in the tables-first order) before layer 0's gated store
// (vform0_ask), which runs only with the table. sm_120 and cc 8.0 by default (the long-range models sm_120);
// LOUP_SPECIES_TABLES=0 keeps the GEMMs, =1 builds the tables on any GPU (tests, A/B on other GPUs).
void GenericModel::setup_species_tables(const Bundle& b) {
    species_tables_ = false;
    GenericLayer* L = layers_.empty() ? nullptr : layers_[0];
    if (!L || !L->first_hidden || L->kind != 4 || L->skip_folded || L->lmax_x != 0) return;
    const char* env = std::getenv("LOUP_SPECIES_TABLES");
    if (env && env[0] == '0') return;
    if (!(lr_ ? device_is_tuned_generic_cc() : folds_measured_here()) && !(env && env[0] == '1')) return;
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
// last layer, whose skip already sits in its readout (hid += h_in ro_W1_sc, ro_W1_sc = W_sc ro_W1), and (S73) a linear
// readout of a non-last layer (E += hout_t[0] . ro_w). So hout_t is never formed: per degree l, y_t[l] (and for layer
// 0 the species-only h_in[0]) multiplies one composed weight per consumer, W_p_t[l] W_sc_{t+1}[l] ... W_up_s[l] (or
// ... ro_W1, or ... ro_w), composed in double at load; the backward forms dy_t from the consumers' gradients through
// the same weights (a linear readout's gradient is the constant composed ro_w, seeded into dy_t[0] by the forward).
// Removes the full-width W_p / W_sc products of every non-last layer in both directions (and their hout / dhout
// buffers). Applies when the model has no long range, no non-last layer has a nonlinear readout, and the last layer's
// readout is folded; per-species skips make the composed weights per species (species-grouped GEMMs). S57 covered the
// all-gated, readout-free, species-free models (OMOL-0); S73 any layer kind, linear non-last readouts and per-species
// skips (MH-1, OMAT-0 / MP-0 / MPtrj / OFF on the generic path). Layer 0's skip terms are [S, cols] tables
// (E W_sc_0 M) with the S59 species tables, else Wh GEMMs; their energy share is always a per-species constant.
// sm_120 and cc 8.0 by default (where it was measured); LOUP_CHAIN_FOLD=0 keeps the plain path, =1 folds on any GPU
// (tests, A/B on other GPUs).
void GenericModel::setup_chain_fold(const Bundle& b) {
    chain_fold_ = chain_fold_built_ = false;
    const int nL = n_layers_;
    if (lr_ || nL < 2) return;
    const char* env = std::getenv("LOUP_CHAIN_FOLD");
    if (env && env[0] == '0') return;
    if (!folds_measured_here() && !(env && env[0] == '1')) return;
    const GenericLayer* last = layers_.back();
    if (!last->ro_folded || last->readout < 1) return;
    for (int t = 0; t < nL - 1; ++t) {
        const GenericLayer* L = layers_[t];
        if (L->readout != -1 && L->readout != 0) return;
        for (int l = 0; l <= L->lmax_out; ++l)
            if (l > layers_[t + 1]->lmax_x) return;  // every degree of y_t reaches the next linear_up
    }
    DevCompose dc;
    struct Target {
        int dest, cols, ns;
        const double* M;  // [ns, rows, cols]: hout_t's (or h_in's) consumer map
    };
    auto wsc = [&](const GenericLayer* Lk, const std::string& p, int l) {
        return dc.dup(b.f32(p + "W_sc_" + std::to_string(l), {Lk->sc_species ? S_ : 1, Lk->C_in, Lk->C_out}));
    };
    std::vector<Target> next[3];  // the targets of hout_{t+1}
    std::vector<GenericLayer::FoldTarget> out[8][3];
    for (int t = nL - 2; t >= 0; --t) {
        GenericLayer* L = layers_[t];
        const GenericLayer* Ln = layers_[t + 1];
        const std::string pn = "L" + std::to_string(t + 1) + ".";
        const std::string p = "L" + std::to_string(t) + ".";
        std::vector<Target> cur[3];
        if (L->readout == 0) cur[0].push_back({-2, 1, 1, dc.dup(b.f32(p + "ro_w", {L->C_out}))});
        for (int l = 0; l <= L->lmax_out; ++l) {
            cur[l].push_back({t + 1, Ln->C_x, 1, dc.dup(b.f32(pn + "W_up_" + std::to_string(l), {Ln->C_in, Ln->C_x}))});
            if (l <= 2 && Ln->sc_has[l]) {
                const int nsc = Ln->sc_species ? S_ : 1;
                const double* Wsc = wsc(Ln, pn, l);
                if (t + 1 == nL - 1) {
                    if (l == 0) {  // the last layer's skip, into its readout hidden
                        const double* W1 = dc.dup(b.f32(pn + "ro_W1", {Ln->C_out, Ln->ro_H}));
                        cur[0].push_back({-1, Ln->ro_H, nsc, dc.mm_s(Wsc, nsc, Ln->C_in, Ln->C_out, W1, 1, Ln->ro_H)});
                    }
                } else {
                    for (const Target& d : next[l])
                        cur[l].push_back({d.dest, d.cols, std::max(nsc, d.ns),
                                          dc.mm_s(Wsc, nsc, Ln->C_in, Ln->C_out, d.M, d.ns, d.cols)});
                }
            }
        }
        for (const Target& d : cur[0])
            if (d.dest == -2 && d.ns > 1) return;  // a linear readout behind a per-species skip: not folded
        const double* E = t == 0 ? dc.dup(b.f32("embedding", {S_, C0_})) : nullptr;
        for (int l = 0; l <= L->lmax_out; ++l) {
            const double* Wp = dc.dup(b.f32(p + "W_p_" + std::to_string(l), {L->C_out, L->C_out}));
            const int nsc0 = L->sc_species ? S_ : 1;
            const double* Wsc0 = (t == 0 && l == 0 && L->sc_has[0]) ? wsc(L, p, 0) : nullptr;
            for (const Target& d : cur[l]) {
                GenericLayer::FoldTarget ft;
                ft.dest_layer = d.dest;
                ft.cols = d.cols;
                ft.ns = d.ns;
                ft.Wy = dc.to_f32(dc.mm_s(Wp, 1, L->C_out, L->C_out, d.M, d.ns, d.cols),
                                  (int64_t)d.ns * L->C_out * d.cols, L->weights);
                if (Wsc0) {
                    const int nsh = std::max(nsc0, d.ns);
                    const double* Wh = dc.mm_s(Wsc0, nsc0, L->C_in, L->C_out, d.M, d.ns, d.cols);
                    if (species_tables_ || d.dest == -2)
                        ft.Th = dc.to_f32(dc.mm_s(E, S_, 1, C0_, Wh, nsh, d.cols), (int64_t)S_ * d.cols, L->weights);
                    else {
                        ft.Wh = dc.to_f32(Wh, (int64_t)nsh * L->C_in * d.cols, L->weights);
                        ft.ns_h = nsh;
                    }
                }
                out[t][l].push_back(ft);
            }
        }
        for (int l = 0; l < 3; ++l) next[l] = cur[l];
    }
    for (int t = 0; t < nL - 1; ++t)
        for (int l = 0; l < 3; ++l) layers_[t]->fold[l] = std::move(out[t][l]);
    chain_fold_ = chain_fold_built_ = true;
}
size_t GenericModel::system_bytes() const { return sys_bytes_ + (lr_ ? lr_->system_bytes() : 0); }
float* GenericModel::layer_out(int t, int l) const { return layers_[t]->hout[l]; }
float* GenericModel::layer_out_grad(int t, int l) const { return layers_[t]->dhout[l]; }
int GenericModel::out_width(int t) const { return layers_[t]->C_out; }
int GenericModel::out_lmax(int t) const { return layers_[t]->lmax_out; }

namespace {
// O31 / list 19: the offsets (floats) of layer 0's kept forward results in GenericPassInputs::keep0 for N receivers:
// off[0..3] msgf_l, off[4] D; for the gated layer 0 (kind 4) also off[5 + l] lin_l (gout_l for the degrees whose gated
// store runs in the projection's epilogue, gate_vform) and off[9] res_0 without the S59 table (else 0); each slot from
// a 256-byte boundary (the gated GEMMs' 16-byte operands). Returns the floats of the whole.
size_t keep0_layout(const GenericLayer* L, int N, size_t off[10]) {
    const size_t n = (size_t)N;
    size_t o = 0;
    auto put = [&](int i, size_t floats) {
        off[i] = o;
        o += (floats + 63) & ~size_t(63);
    };
    for (int i = 0; i < 10; ++i) off[i] = 0;
    for (int l = 0; l < 4; ++l) put(l, n * (2 * l + 1) * L->C_msg);
    put(4, n);
    if (L->kind == 4) {
        for (int l = 0; l < 4; ++l) put(5 + l, n * (2 * l + 1) * std::max(L->lin_cols[l], L->C_msg));
        if (!L->res0_table) put(9, n * L->lin_cols[0]);
    }
    return o;
}
}  // namespace

bool GenericModel::sweep_keep_layer0() const {
    // list 20: with the fold in sweeps the backward forms dy from the fold targets' gradients, and the kept forward
    // results are the same
    if (layers_.empty() || (chain_fold_ && !fold_sweeps_) || lr_) return false;
    const GenericLayer* L = layers_[0];
    if (!L->first_hidden || L->readout != 0 || L->ro_folded || L->lmax_x != 0 || !L->ro_w) return false;
    if (L->kind == 2)
        return L->skip_folded && !L->res0_table && !L->sc_has[0] && !L->sc_has[1] && !L->sc_has[2];
    // list 19: the gated layer 0 (MH-1, OMOL-0); its backward reads msgf, D, lin (or gout), res_0, never the skip
    return L->kind == 4 && !L->skip_folded && !L->conv_prune && !L->lin_compact && !L->hfold[0];
}

size_t GenericModel::sweep_keep_floats(int N) const {
    size_t off[10];
    return layers_.empty() ? 0 : keep0_layout(layers_[0], N, off);
}

namespace {
// dh[a, c] += w[c]: the linear readout's gradient seed (ro_linear_kernel's) without its energy
__global__ void add_row_vector_kernel(int N, int C, const float* __restrict__ w, float* __restrict__ dh) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)N * C) dh[i] += w[i % C];
}
}  // namespace

std::string GenericModel::kernel_signature() const {
    std::string out;
    for (const GenericLayer* L : layers_) {
        if (!out.empty()) out += ';';
        out += (L && L->prod && L->prod->sig) ? L->prod->sig : "?";
    }
    return out;
}


// Long-range hidden fold. The POLAR models form every layer's hout (S57's chain fold excludes them) because the long-range
// block reads it: the multipole sources (qsrc += hout_t[l] src_t[l]), the feature mixer (fm[l] += hout_t[l] mix_t[l]),
// the Fukui MLP (the last layer's l = 0) and the electron energy (the last layer's l <= le_lmax()). The sources and the
// mixer are linear, as are the next layer's linear_up and skip. For a degree l >= 1, a prefix of k layers whose
// degree-l hout only those linear maps read (layer 0 has no skip into l; the last layer only if the electron energy
// does not read l) never forms it: hout_u[l] = sum_{v <= u} y_v[l] G_{v,u} (G_{u,u} = W_p_u, G_{v,u} = G_{v,u-1}
// W_sc_u), so y_v[l] feeds every consumer through one weight composed in double at load and rounded once: the next
// layer's x (W_up' = G W_up, earlier layers' terms as x_extra), the first unfolded layer's skip (W_sc' = G W_sc,
// sc_extra), the mixer W_p_v B_v (B_{k-1} = M_{k-1}, B_u = M_u + W_sc_{u+1} B_{u+1}) and the source (alike, per m
// block). hout_v[l] / dhout_v[l] alias y_v[l] / dy_v[l]: the long-range block and the next layer read the same
// pointers with the composed weights, and the backward accumulates every consumer's gradient straight into dy_v[l]
// (zeroed with the zero slab). Degree 1 folds the first layer (h_0 = y_0 W_p_0; POLAR-1-M / -L: no W_p_0 product in
// either direction), degree 2 every layer it can (POLAR-1-L: all three, its degree-2 sources and electron-energy
// blocks being exactly zero; 14 fewer GEMMs per force call). Exact to float32 rounding (reassociation only). sm_120
// by default; LOUP_POLAR_HFOLD=0 disables it, 1 / 2 / 3 (bit l - 1: degree l) choose the folded degrees on any GPU.
void GenericModel::setup_lr_fold(const Bundle& b) {
    if (!lr_) return;
    int mask = device_is_tuned_generic_cc() ? 3 : 0;
    if (const char* e = std::getenv("LOUP_POLAR_HFOLD"); e && e[0] >= '0' && e[0] <= '3') mask = e[0] - '0';
    const int nL = n_layers_, C = layers_[0]->C_out, le_l = lr_->le_lmax();
    for (int l = 1; l <= std::min(layers_[0]->lmax_out, 2); ++l) {
        if (!((mask >> (l - 1)) & 1)) continue;
        int k = l == 1 ? 1 : nL;
        if (k == nL && le_l >= l) --k;  // the electron energy reads the last layer's hout[l]
        if (k < 1) continue;
        // every layer's degree-l output (and later layers' input) C wide, species-independent skips on the way
        bool ok = !layers_[0]->sc_has[l];
        for (int u = 0; u < nL; ++u) {
            const GenericLayer* Lu = layers_[u];
            if (Lu->lmax_out < l || Lu->C_out != C || Lu->ro_folded) ok = false;
            if (u > 0 && (Lu->lmax_in < l || Lu->C_in != C || (u <= k && Lu->sc_has[l] && Lu->sc_species))) ok = false;
        }
        if (!ok || chain_fold_) continue;
        DevCompose dc;
        auto key = [&](int u, const char* w) { return "L" + std::to_string(u) + "." + w + std::to_string(l); };
        std::vector<const double*> P(k), S(nL, nullptr);
        for (int u = 0; u < k; ++u) P[u] = dc.dup(b.f32(key(u, "W_p_"), {C, C}));
        for (int u = 1; u <= std::min(k, nL - 1); ++u)
            if (layers_[u]->sc_has[l]) S[u] = dc.dup(b.f32(key(u, "W_sc_"), {1, C, C}));
        // G_{v, s-1} W = W_p_v W_sc_{v+1} ... W_sc_{s-1} W for W [C, cols]; nullptr when a skip on the way is absent
        auto chain = [&](int v, int s, const double* W, int cols) -> const double* {
            const double* T = W;
            for (int u = s - 1; u > v; --u) {
                if (!S[u]) return nullptr;
                T = dc.mm(S[u], C, C, T, cols);
            }
            return dc.mm(P[v], C, C, T, cols);
        };
        // layer s's x (s <= k) and, for the first unfolded layer, its skip: the term of layer s - 1 replaces the
        // weight (h_in aliases y_{s-1}), the earlier layers' are extra terms
        for (int s = 1; s <= std::min(k, nL - 1); ++s) {
            GenericLayer* Ls = layers_[s];
            auto fold_into = [&](const double* W, int cols, float*& primary, std::vector<GenericLayer::YTerm>& extra) {
                for (int v = s - 1; v >= 0; --v) {
                    const double* G = chain(v, s, W, cols);
                    if (!G) continue;
                    float* f = dc.to_f32(G, (int64_t)C * cols, Ls->weights);
                    if (v == s - 1) primary = f;
                    else extra.push_back({v, f});
                }
            };
            if (Ls->lmax_x >= l)
                fold_into(dc.dup(b.f32(key(s, "W_up_"), {C, Ls->C_x})), Ls->C_x, Ls->W_up[l], Ls->x_extra[l]);
            if (s == k && S[s]) fold_into(S[s], C, Ls->W_sc[l], Ls->sc_extra[l]);
        }
        // mixer and source of each folded layer v: W_p_v B_v; a source block is m-major, (2l + 1) blocks [C, 8]
        const int lm = layers_[0]->lmax_out, rows = dimL(l) * C;
        const double *B = nullptr, *Bs = nullptr;
        for (int v = k - 1; v >= 0; --v) {
            const std::string t = std::to_string(v);
            const double* M = dc.dup(b.f32("LR.mix_" + t + "_" + std::to_string(l), {C, C}));
            const std::vector<float> src = b.f32("LR.src_" + t, {(lm + 1) * (lm + 1) * C, 8});
            const double* Rv = dc.dup(std::vector<float>(src.begin() + (size_t)l * l * C * 8,
                                                         src.begin() + ((size_t)l * l * C + rows) * 8));
            if (v == k - 1 || !S[v + 1]) {
                B = M;
                Bs = Rv;
            } else {
                B = dc.mm(S[v + 1], C, C, B, C, M);
                double* nb = dc.alloc((int64_t)rows * 8);
                for (int m = 0; m < dimL(l); ++m)
                    dc.mm(S[v + 1], C, C, Bs + (size_t)m * C * 8, 8, Rv + (size_t)m * C * 8, nb + (size_t)m * C * 8);
                Bs = nb;
            }
            double* sv = dc.alloc((int64_t)rows * 8);
            for (int m = 0; m < dimL(l); ++m) dc.mm(P[v], C, C, Bs + (size_t)m * C * 8, 8, nullptr, sv + (size_t)m * C * 8);
            lr_->set_hidden_fold(v, l, dc.to_f32(dc.mm(P[v], C, C, B, C), (int64_t)C * C, layers_[v]->weights),
                                 dc.to_f32(sv, (int64_t)rows * 8, dc.tmp));
        }
        for (int u = 0; u < k; ++u) layers_[u]->hfold[l] = true;
    }
}
// ------------------------------------------------------------------------------------------------- buffers
bool GenericModel::set_cell(const double* cell) { return lr_ ? lr_->set_cell(cell) : false; }

// S64: the conv forward variants (launch_conv_fwd's var: for lx 1 at C 128 the one-receiver paired block, the
// unpaired one, and five receivers per 640-thread block) rank differently per system (5060 Ti, gconv_fwd_L1 us per
// step R1p / R1 / R5: OMAT-0 medium SrTiO3 2,200 / 2,176 / 1,742, NbCl5 346 / 327 / 321; OFF24-M water 2,723 /
// 2,686 / 2,437; MP-0 medium SrTiO3 2,385 / 2,273 / 2,136, NbCl5 354 / 348 / 357), so each layer's are timed once
// per system on the last force pass's inputs (from tune_rebuild_variants at the first rebuild, and from
// tune_early_variants after 50 passes without one); a variant replaces the current one only if >= 2% faster. Its outputs (mid, D) are
// rewritten by every force pass. sm_120 by default; LOUP_CONV_TUNE=0 keeps the default variant, LOUP_CONV_TUNE=1
// times them on any GPU (tests). Returns -1 before a pass has been recorded for this system (call again later), 0 if
// nothing changed, 1 if a variant changed.
int GenericModel::tune_conv_variants(cudaStream_t st) {
    if (conv_tune_env_ == 0) return 0;
    if (!(lr_ ? device_is_tuned_cc() : device_is_tuned_generic_cc()) && conv_tune_env_ != 1) return 0;
    if (conv_var_override_ >= 0) return 0;  // forced variant: nothing to time
    if (!conv_args_.valid) return -1;
    const ConvTuneArgs& a = conv_args_;
    cudaEvent_t e0, e1;
    CUDA_CHECK_THROW(cudaEventCreate(&e0));
    CUDA_CHECK_THROW(cudaEventCreate(&e1));
    bool changed = false;
    for (size_t t = 0; t < layers_.size(); ++t) {
        GenericLayer* L = layers_[t];
        if (L->first_hidden) continue;
        const int nv = conv_fwd_variants(L->lmax_x, L->C_x);
        if (nv <= 1) continue;
        float best[8] = {};
        for (int v = 0; v < nv; ++v) {
            best[v] = 1e30f;
            for (int round = 0; round < 4; ++round) {  // round 0 warms up
                CUDA_CHECK_THROW(cudaEventRecord(e0, st));
                launch_conv_fwd(L->lmax_x, st, a.N, L->C_x, S_, L->intervals, L->grid, sh_perm_, L->pair_sym, a.row_ptr,
                                a.senders, a.vec_len, a.slab, a.t_sx, L->last_preloc, a.species, L->packed_table,
                                L->density_table, L->cubic_bias_shift, L->x, L->mid, L->D, v, nullptr, L->conv_prune,
                                L->jit_conv_on ? &L->jit_conv : nullptr);
                CUDA_CHECK_THROW(cudaEventRecord(e1, st));
                CUDA_CHECK_THROW(cudaEventSynchronize(e1));
                float ms = 0.0f;
                CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, e0, e1));
                if (round > 0) best[v] = std::min(best[v], ms);
            }
        }
        int fast = L->fwd_var;
        for (int v = 0; v < nv; ++v)
            if (best[v] < best[fast]) fast = v;
        if (fast != L->fwd_var && best[fast] < 0.98f * best[L->fwd_var]) {
            L->fwd_var = fast;
            changed = true;
        }
        if (std::getenv("LOUP_DEBUG_SPLINE")) {
            fprintf(stderr, "generic layer %zu conv forward us:", t);
            for (int v = 0; v < nv; ++v) fprintf(stderr, " %d: %.1f", v, best[v] * 1e3f);
            fprintf(stderr, " -> %d\n", L->fwd_var);
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return changed ? 1 : 0;
}

void GenericModel::free_system() {
    conv_args_ = ConvTuneArgs{};
    for (auto* L : layers_) L->fwd_var = 0;
    if (lr_) lr_->free_system();
    for (void* p : sys_allocs_) cudaFree(p);
    sys_allocs_.clear();
    sys_alloc_bytes_.clear();
    sys_zero_once_.clear();
    hand_.clear();
    layers_shared_ = false;
    sys_bytes_unshared_ = 0;
    sys_max_alloc_ = 0;
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
    sys_max_alloc_ = 0;
    auto alloc = [&](size_t floats) -> float* {
        float* p = nullptr;
        sys_max_alloc_ = std::max(sys_max_alloc_, floats * sizeof(float));
        CUDA_CHECK_THROW(device_malloc(&p, std::max<size_t>(floats, 1) * sizeof(float)));
        sys_bytes_ += std::max<size_t>(floats, 1) * sizeof(float);
        sys_allocs_.push_back(p);
        sys_alloc_bytes_.push_back(std::max<size_t>(floats, 1) * sizeof(float));
        return p;
    };
    const size_t n = (size_t)N;
    auto align256 = [](size_t b) { return (b + 255) & ~size_t(255); };
    // Pack every zero-initialized per-step gradient accumulator into one contiguous slab (1 cudaMemsetAsync/step).
    // S97: a folded linear readout (S73) assigns dy[0] in the forward, the pass's first write of it, so dy[0] is not
    // one of them; S99: neither is a conv layer's dx, which its forward conv kernel zeroes per receiver.
    size_t zbytes = 0;
    for (int t = 0; t < n_layers_; ++t) {
        GenericLayer* L = layers_[t];
        if (!L->ro_folded && !(chain_fold_ && t < n_layers_ - 1))
            for (int l = 0; l <= L->lmax_out; ++l) zbytes += align256(n * dimL(l) * L->C_out * sizeof(float));
        if (L->kind >= 2) zbytes += align256(n * sizeof(float));
    }
    if (zbytes) {
        CUDA_CHECK_THROW(device_malloc(&d_zero_slab_, zbytes));
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
    // tile-shared-layers: a layer's own buffers come from `own`. Where the system only sweeps (share_layer_buffers,
    // LOUP_SWEEP_REPLAY) they are carved from one arena, each layer from its start: a sweep runs one layer, and of
    // another layer it reads and writes only the output (hout) and its gradient (dhout, in the zero slab), which
    // keep buffers of their own. A first walk of the layers sizes the arena (the largest layer), a second assigns.
    // list 20: with the fold in sweeps the fold's targets (the last layer's x, dx, hid, dhid) are what crosses between
    // the layers' sweeps, so they keep buffers of their own (hand_buf) and the rest shares the arena
    layers_shared_ = (share_layers_ || sweep_replay_) && share_layers_env_ && n_layers_ >= 2 && !lr_ &&
                     (!chain_fold_ || fold_sweeps_);
    char* arena = nullptr;
    size_t arena_off = 0, arena_bytes = 0, layers_sum = 0;
    bool sizing = false;
    auto own = [&](size_t floats) -> float* {
        if (!layers_shared_) return alloc(floats);
        if (sizing) sys_max_alloc_ = std::max(sys_max_alloc_, floats * sizeof(float));
        float* p = sizing ? nullptr : reinterpret_cast<float*>(arena + arena_off);
        arena_off += align256(std::max<size_t>(floats, 1) * sizeof(float));
        return p;
    };
    // Buffers whose zeros are written once, below, and read in every pass (S125: a pruned conv layer's skipped
    // projections; S130 / S132: the pad columns of a compact layer's angular gradients) keep memory of their own:
    // in the arena another layer's sweep would overwrite them.
    auto own_zero = [&](size_t floats) -> float* {
        if (sizing) return nullptr;
        float* p = alloc(floats);
        sys_zero_once_.push_back(p);
        return p;
    };
    // list 20: the fold's targets in a folding domain (the last layer's x and dx of the degrees layer 0 hands on, its
    // hid and dhid): outside the arena, which layer 0's sweeps overwrite
    auto hand_buf = [&](size_t floats, bool hand) -> float* {
        if (!hand || !layers_shared_) return own(floats);
        return sizing ? nullptr : alloc(floats);
    };
    auto layout_layer = [&](int t) {
        GenericLayer* L = layers_[t];
        const bool hand_t = fold_sweeps_ && t == n_layers_ - 1;
        for (int l = 0; l <= L->lmax_x; ++l) {
            const bool hl = hand_t && l <= layers_[0]->lmax_out;
            L->x[l] = hand_buf(n * dimL(l) * L->C_x, hl);
            L->dx[l] = (t > 0) ? hand_buf(n * dimL(l) * L->C_x, hl) : nullptr;
        }
        if (L->first_hidden)
            for (int l = 0; l < 4; ++l) L->S0[l] = own(n * dimL(l) * (size_t)L->K0);
        for (int l = 0; l < 4; ++l) {
            L->mid[l] = L->first_hidden ? nullptr : own(n * dimL(l) * L->n_l3[l] * L->C_x);
            const bool zero_fwd = L->conv_prune && l >= 1, zero_bwd = L->lin_compact && l >= 1;
            if (L->skip_folded) {
                L->lin[l] = nullptr;
                L->dlin[l] = nullptr;
                L->msgf[l] = zero_fwd ? own_zero(n * dimL(l) * L->C_msg) : own(n * dimL(l) * L->C_msg);
            } else {
                L->lin[l] = zero_fwd ? own_zero(n * dimL(l) * L->lin_cols[l]) : own(n * dimL(l) * L->lin_cols[l]);
                L->dlin[l] = zero_bwd ? own_zero(n * dimL(l) * L->lin_cols[l]) : own(n * dimL(l) * L->lin_cols[l]);
                L->msgf[l] = (L->kind == 1 || L->kind == 3) ? L->lin[l] : own(n * dimL(l) * L->C_msg);
            }
            L->dmsgf[l] = own(n * dimL(l) * L->C_msg);
            if (L->kind == 4) {
                L->gout[l] = own(n * dimL(l) * L->C_msg);
                L->dgout[l] = own(n * dimL(l) * L->C_msg);
                L->dpre[l] = (t > 0 && l <= L->lmax_x) ? own(n * dimL(l) * L->lin_cols[l]) : nullptr;
            }
        }
        if (L->conv_prune && !sizing)  // S125: the skipped projections' outputs, read by the product (or the gate), stay zero
            for (int l = 1; l < 4; ++l) {
                float* z = L->skip_folded ? L->msgf[l] : L->lin[l];
                CUDA_CHECK_THROW(cudaMemset(z, 0, n * dimL(l) * (L->skip_folded ? L->C_msg : L->lin_cols[l]) * sizeof(float)));
            }
        if (L->lin_compact && !sizing)  // S130 / S132: the pad columns of the compact angular gradients are never written
            for (int l = 1; l < 4; ++l)
                CUDA_CHECK_THROW(cudaMemset(L->dlin[l], 0, n * dimL(l) * L->lin_cols[l] * sizeof(float)));
        if (L->kind == 4)  // with the S59 table the gates read res[0] from it
            for (int l = 0; l <= L->lmax_x; ++l)
                L->res[l] = (l == 0 && L->res0_table) ? nullptr : own(n * dimL(l) * L->lin_cols[l]);
        L->D = own(n);
        for (int l = 0; l <= L->lmax_out; ++l) {
            L->y[l] = own(n * dimL(l) * L->C_out);
            L->dy[l] = L->hfold[l] ? nullptr : own(n * dimL(l) * L->C_out);  // folded: in the zero slab (below)
        }
        if (L->ro_H) {
            L->hid = hand_buf(n * L->ro_H, hand_t);
            L->hid2 = own(n * L->ro_H);
            L->dhid = hand_buf(n * L->ro_H, hand_t);
            L->dhid2 = own(n * L->ro_H);
        }
    };
    std::vector<size_t> arena_layer((size_t)n_layers_, 0);  // list 20: each layer's part of the arena (fold_delta_)
    if (layers_shared_) {
        sizing = true;
        for (int t = 0; t < n_layers_; ++t) {
            arena_off = 0;
            layout_layer(t);
            arena_layer[(size_t)t] = arena_off;
            arena_bytes = std::max(arena_bytes, arena_off);
            layers_sum += arena_off;
        }
        sizing = false;
        const size_t largest = sys_max_alloc_;  // the arena is no buffer of a GEMM
        arena = reinterpret_cast<char*>(alloc((arena_bytes + sizeof(float) - 1) / sizeof(float)));
        sys_max_alloc_ = largest;
    }
    for (int t = 0; t < n_layers_; ++t) {
        GenericLayer* L = layers_[t];
        arena_off = 0;
        layout_layer(t);
        // What sweeps carry from layer to layer, and the zero slab's accumulators: never in the arena.
        L->dD = (L->kind >= 2) ? zalloc(n) : nullptr;
        for (int l = 0; l <= L->lmax_out; ++l) {
            if (L->hfold[l]) {  // long-range hidden fold: hout aliases y, dhout dy (in dhout's zero-slab place)
                L->dy[l] = zalloc(n * dimL(l) * L->C_out);
                L->hout[l] = L->y[l];
                L->dhout[l] = L->dy[l];
                continue;
            }
            const bool no_hout = L->ro_folded || (chain_fold_ && t < n_layers_ - 1);  // S57: never formed
            L->hout[l] = no_hout ? nullptr : alloc(n * dimL(l) * L->C_out);
            L->dhout[l] = no_hout ? nullptr : zalloc(n * dimL(l) * L->C_out);
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
    sys_bytes_unshared_ = sys_bytes_ + (layers_shared_ ? layers_sum - arena_bytes : 0);
    // list 20: what each layer's sweeps hand on (hand_count): its output, or with the fold in sweeps (two layers)
    // the fold's targets, layer 1's x[l] and the last layer's readout hidden
    hand_.assign((size_t)n_layers_, {});
    for (int t = 0; t + 1 < n_layers_; ++t) {
        const GenericLayer* L = layers_[t];
        if (fold_sweeps_) {
            const GenericLayer* Ln = layers_[t + 1];
            const GenericLayer* La = layers_.back();
            bool hid = false;
            for (int l = 0; l <= L->lmax_out; ++l) {
                hand_[(size_t)t].push_back({Ln->x[l], Ln->dx[l], dimL(l) * Ln->C_x, false});
                for (const auto& ft : L->fold[l]) hid = hid || ft.dest_layer == -1;
            }
            if (hid) hand_[(size_t)t].push_back({La->hid, La->dhid, La->ro_H, true});
        } else {
            for (int l = 0; l <= L->lmax_out; ++l)
                hand_[(size_t)t].push_back({L->hout[l], L->dhout[l], dimL(l) * L->C_out, false});
        }
    }
    // list 20 (the memory plan, which sizes a folding plain order from an unfolded probe): what this unfolded
    // two-layer system would hold with the fold in its sweeps beyond what it holds. Less layer 0's hout and dhout (its
    // own buffer and the zero slab's); with the shared arena the fold's targets leave the last layer's part of it and
    // take buffers of their own. Unshared (system_bytes_unshared's count) the targets are counted either way.
    fold_delta_[0] = fold_delta_[1] = 0.0;
    fold_hand_w_.clear();
    if (!chain_fold_ && chain_fold_built_ && n_layers_ == 2 && !lr_) {
        const GenericLayer* L0 = layers_[0];
        const GenericLayer* L1 = layers_[1];
        double hout_b = 0.0, H = 0.0;
        bool hid = false;
        for (int l = 0; l <= L0->lmax_out; ++l) {
            const size_t fo = n * dimL(l) * L0->C_out, fx = n * dimL(l) * L1->C_x;
            hout_b += (double)(std::max<size_t>(fo, 1) * sizeof(float)) + (double)align256(fo * sizeof(float));
            H += 2.0 * (double)align256(std::max<size_t>(fx, 1) * sizeof(float));
            fold_hand_w_.push_back(dimL(l) * L1->C_x);
            for (const auto& ft : L0->fold[l]) hid = hid || ft.dest_layer == -1;
        }
        if (hid) {
            H += 2.0 * (double)align256(std::max<size_t>(n * L1->ro_H, 1) * sizeof(float));
            fold_hand_w_.push_back(L1->ro_H);
        }
        fold_delta_[0] = fold_delta_[1] = -hout_b;
        if (layers_shared_) {
            const double a0 = (double)arena_layer[0], a1 = (double)arena_layer[1];
            fold_delta_[1] += std::max(a0, a1 - H) + H - std::max(a0, a1);
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
    CUDA_CHECK_THROW(device_malloc(&d_edge_dvec_, (size_t)E_cap * 3 * sizeof(float)));
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
    const int Nx = in.N + in.n_ghost;  // rows of what senders provide: h_in, x = h_in W_up, dx, dh_in
    // S64: what tune_conv_variants needs to re-launch the conv forwards (valid until the next rebuild)
    conv_args_ = ConvTuneArgs{true, N, in.act_row_ptr, in.act_senders, in.act_vec_len, in.act_slab, in.act_t_sx,
                              in.species};
    cudaStream_t st = in.stream;
    const auto& start = *in.species_start;
    const auto& count = *in.species_count;
    const int S = S_;
    auto fork = [&] { if (in.gemm_fork) in.gemm_fork(); };
    auto lane = [&](int k) { if (in.gemm_lane) in.gemm_lane(k); };
    auto join = [&] { if (in.gemm_join) in.gemm_join(); };
    // LOUP_LIN_HALF (setup_lin_half): C [M][Nc] = A W (transB false; W [K][Nc]) or A W^T (W [Nc][K]) on hgemm2f_kernel
    // where the layer has the projection's panels, else the FP32 GEMM; slot: LOUP_DEBUG_LIN_AMAX's record
    auto run_gemm = [&](bool transB, int M, int Nc, int Kc, float alpha, const float* A, int lda,
                        const float* W, int ldb, float beta, float* C, int ldc) {
        if (polar_h2_ && in.gemm_h2_row && alpha == 1.0f && lda == Kc && ldc == Nc &&
            (transB ? ldb == Kc : ldb == Nc) && M >= polar_h2_min_m_ && Nc >= 64 && Kc >= 64) {
            auto it = ph2_map_.find(W);
            if (it != ph2_map_.end()) {
                const auto& hw = it->second;
                void* panel = transB ? hw.BT : hw.B;
                float* pb = transB ? hw.pBT : hw.pB;
                if (panel && (transB ? (hw.rows == Nc && hw.cols == Kc) : (hw.rows == Kc && hw.cols == Nc))) {
                    in.gemm_h2_row(M, Nc, Kc, A, panel, pb, C, beta, nullptr);
                    return;
                }
            }
        }
        in.gemm(transB, M, Nc, Kc, alpha, A, lda, W, ldb, beta, C, ldc);
    };
    auto proj = [&](const GenericLayer::H2Proj& p, int slot, bool transB, int M, int Nc, int K, const float* A,
                    const float* W, float* C, float beta = 0.0f) {
        if (d_lin_amax_) lin_amax_record(A, (int64_t)M * K, slot, st);
        if (p.B && in.gemm_h2_fwd)
            in.gemm_h2_fwd(M, Nc, K, A, p.sA, p.B, p.os, C, beta);
        else
            run_gemm(transB, M, Nc, K, 1.0f, A, K, W, transB ? K : Nc, beta, C, Nc);
    };
    // LOUP_RES_BWD (setup_lin_half): layer t's backward W_res^T assigns dx (the forward conv kernel leaves dx's rows
    // alone), and also reads dlin instead of dpre (the gate backward writes no dpre)
    auto res_assigns = [&](const GenericLayer* L, int t) {
        return res_bwd_ >= 1 && L->kind == 4 && t > 0 && !L->first_hidden && !L->skip_folded;
    };
    auto res_from_dlin = [&](const GenericLayer* L, int t) {
        if (res_bwd_ < 2 || !res_assigns(L, t) || !in.gemm_h2_rows || L->lmax_x > 2) return false;
        for (int l = 0; l <= L->lmax_x; ++l)
            if (!L->hl_bres[l].B || !L->hl_blin[l].B) return false;
        return true;
    };
    // LOUP_RES_BWD 3: degree l's dmid and dx in one product from dlin (hl_bcat)
    auto res_fused = [&](const GenericLayer* L, int t, int l) {
        return res_bwd_ >= 3 && l <= L->lmax_x && l < 3 && L->hl_bcat[l].B && res_from_dlin(L, t);
    };
    // the row scale D beta + alpha of dpre = dlin (D beta + alpha), degree l
    auto res_rows = [&](const GenericLayer* L, int l) {
        GateArgs rs;
        rs.D = L->D;
        rs.alpha = L->alpha;
        rs.beta = L->beta;
        rs.rdeg = dimL(l);
        return rs;
    };
    // species-grouped GEMM: rows of degree l for each species with its own weight block
    auto species_gemm = [&](bool transB, int l, int K, int Nc, const float* A, const float* W, int64_t w_stride,
                            float beta, float* C, int lane_base = -1) {
        const int d = dimL(l);
        for (int s = 0; s < S; ++s) {
            if (count[s] == 0) continue;
            if (lane_base >= 0) lane(lane_base + s);
            const int64_t row0 = (int64_t)start[s] * d;
            run_gemm(transB, count[s] * d, Nc, K, 1.0f, A + row0 * K, K, W + (w_stride ? s * w_stride : 0),
                    transB ? K : Nc, beta, C + row0 * Nc, Nc);
        }
    };
    // list 24 item 4: species_gemm (beta 0) on hgemm2f_kernel for each species with panels in ps (LOUP_LIN_HALF_PARTS
    // bit 128), else the FP32 GEMM; slot: LOUP_DEBUG_LIN_AMAX's record of A over every row of the degree
    auto species_proj = [&](const std::vector<GenericLayer::H2Proj>& ps, int slot, bool transB, int l, int K, int Nc,
                            const float* A, const float* W, int64_t w_stride, float* C, int lane_base) {
        const int d = dimL(l);
        if (d_lin_amax_) lin_amax_record(A, (int64_t)N * d * K, slot, st);
        for (int s = 0; s < S; ++s) {
            if (count[s] == 0) continue;
            if (lane_base >= 0) lane(lane_base + s);
            const int64_t row0 = (int64_t)start[s] * d;
            const GenericLayer::H2Proj* p = (size_t)s < ps.size() ? &ps[s] : nullptr;
            if (p && p->B && in.gemm_h2_fwd && (int64_t)count[s] * d >= lin_half_srows_)
                in.gemm_h2_fwd(count[s] * d, Nc, K, A + row0 * K, p->sA, p->B, p->os, C + row0 * Nc, 0.0f);
            else
                run_gemm(transB, count[s] * d, Nc, K, 1.0f, A + row0 * K, K, W + (w_stride ? s * w_stride : 0),
                        transB ? K : Nc, 0.0f, C + row0 * Nc, Nc);
        }
    };
    // S100: a folded linear readout of y_t[0] (the S73 energy target) runs in the layer's product forward (E +=
    // y_t[0] . Wy) and its gradient seed Wy is added in the product backward, when some other fold target's GEMM is
    // the first write of dy_t[0] (else ro_linear_kernel assigns the seed as before). Not in a per-atom energy pass:
    // the product kernels do not split the energy by atom.
    auto fused_readout = [&](int t) -> const float* {
        const GenericLayer* L = layers_[t];
        if (!chain_fold_ || t >= n_layers_ - 1 || in.atom_energy) return nullptr;
        const float* w = nullptr;
        bool other = false;
        for (const auto& ft : L->fold[0]) {
            if (ft.dest_layer == -2) w = ft.Wy;
            else other = true;
        }
        return other ? w : nullptr;
    };

    // in.energy / in.forces / in.virial live in engine.cu's d_zero_slab and are already zeroed by record_force_pass.
    const bool has_edges = in.E_cap > 0;
    ZblParams zbl_p;  // r_max 0 without ZBL
    if (has_zbl_) {
        zbl_p.rc = d_zbl_rc_; zbl_p.inv_a = d_zbl_inv_a_; zbl_p.pref = d_zbl_pref_;
        zbl_p.c0 = zbl_c_[0]; zbl_p.c1 = zbl_c_[1]; zbl_p.c2 = zbl_c_[2]; zbl_p.c3 = zbl_c_[3];
        zbl_p.p = zbl_p_; zbl_p.r_max = zbl_rmax_;
    }
    const int sweep = in.sweep, sweep_t = in.sweep_layer;
    // S98: the top layer's conv backward (the first in the pass) assigns the per-edge vectors; only a model without
    // conv layers (layer 0 alone) needs them zeroed for the layer-0 backward's accumulation, and so does a backward
    // sweep, whose first conv backward is a lower layer's.
    if (has_edges && (layers_.back()->first_hidden || sweep == GenericPassInputs::SweepBackward))
        zero_edge_kernel<<<(3 * in.E_cap + 255) / 256, 256, 0, st>>>(in.E_dev, d_edge_dvec_);
    if (zero_slab_bytes_ && (sweep == GenericPassInputs::SweepAll || !sweep_zero_)) {
        CUDA_CHECK_THROW(cudaMemsetAsync(d_zero_slab_, 0, zero_slab_bytes_, st));
    } else if (zero_slab_bytes_) {
        // list 19 (LOUP_SWEEP_ZERO=0: the whole slab): a sweep zeroes only the accumulators it adds into and hands on
        // or reads: a forward sweep's layer t none it keeps (dD for a fused readout's backward only), the last layer's
        // sweep its dD, dhout and the layer below's dhout (dh_in), a backward sweep of layer t its dD and layer t - 1's
        // dhout (its own dhout is the adjoint, copied in)
        auto zero_layer = [&](int t, bool dD, bool dhout) {
            const GenericLayer* K = layers_[t];
            if (dD && K->dD) CUDA_CHECK_THROW(cudaMemsetAsync(K->dD, 0, (size_t)N_ * sizeof(float), st));
            if (dhout)
                for (int l = 0; l <= K->lmax_out; ++l)
                    if (K->dhout[l])
                        CUDA_CHECK_THROW(cudaMemsetAsync(K->dhout[l], 0,
                                                         (size_t)N_ * dimL(l) * K->C_out * sizeof(float), st));
        };
        if (sweep == GenericPassInputs::SweepForward) {
            zero_layer(sweep_t, true, false);
        } else if (sweep == GenericPassInputs::SweepLast) {
            zero_layer(n_layers_ - 1, true, true);
            if (n_layers_ > 1) zero_layer(n_layers_ - 2, false, true);
        } else {
            zero_layer(sweep_t, true, false);
            if (sweep_t > 0) zero_layer(sweep_t - 1, false, true);
        }
    }
    // S99: a conv layer's forward kernel zeroes its dx rows per receiver; the ghost rows of a domain (senders only,
    // which the backward accumulates into) are zeroed here.
    if (in.n_ghost > 0)
        for (int t = 1; t < n_layers_; ++t) {
            const GenericLayer* L = layers_[t];
            for (int l = 0; l <= L->lmax_x; ++l)
                if (L->dx[l])
                    CUDA_CHECK_THROW(cudaMemsetAsync(L->dx[l] + (int64_t)in.N * dimL(l) * L->C_x, 0,
                                                     (size_t)in.n_ghost * dimL(l) * L->C_x * sizeof(float), st));
        }

    if (sweep != GenericPassInputs::SweepAll) {
        if (!sweeps_available() || in.energy_only)
            throw std::logic_error("generic: a sweep was asked of a model or a pass that has none");
        const bool layer_ok = sweep == GenericPassInputs::SweepLast || (sweep_t >= 0 && sweep_t < n_layers_ - 1);
        if (!layer_ok) throw std::logic_error("generic: sweep layer out of range");
    } else if (layers_shared_ && !sweep_replay_ && !in.energy_only) {
        // tile-shared-layers: a layer's backward would read forward results a later layer's forward overwrote
        throw std::logic_error("generic: the whole pass was asked of a system whose layers share their buffers");
    }
    const bool runs_layer0 = sweep == GenericPassInputs::SweepAll ||
                             (sweep == GenericPassInputs::SweepLast ? n_layers_ == 1 : sweep_t == 0);
    // O31: layer 0's forward results kept in the domain's buffer (keep0) by its forward sweep for its backward sweep,
    // which then neither replays the forward nor reads h_in
    const bool keep0 = in.keep0 && sweep_t == 0 &&
                       (sweep == GenericPassInputs::SweepForward || sweep == GenericPassInputs::SweepBackward);
    if (runs_layer0 && h0_read_ && !(keep0 && sweep == GenericPassInputs::SweepBackward))
        gather_rows_kernel<<<(unsigned)(((int64_t)Nx * C0_ + 255) / 256), 256, 0, st>>>(Nx, C0_, in.species, d_embedding_, d_h0_);

    // ---------------- forward of layer t: everything from its linear_up to its readout. `energy` takes the
    // readout's energy (a sink when a backward sweep replays the layer).
    std::vector<const float*> written;  // S57 fold destinations already assigned in this pass
    // list 20: in a folding domain's last-layer sweep the fold's targets came in from layer 0's sweeps (hand), so the
    // last layer's accumulations onto them (the readout hidden) add instead of assigning
    if (fold_sweeps_ && sweep == GenericPassInputs::SweepLast)
        for (const Hand& e : hand_[(size_t)n_layers_ - 2]) written.push_back(e.p);
    std::vector<unsigned> gate_vform(n_layers_, 0u);  // per layer: degrees whose forward gate ran in the projection
    std::vector<unsigned> gate_rform(n_layers_, 0u);  // per layer: residual degrees whose gate ran in W_res's (list 27)
    // bit 256's floor: an M x Nc fold product on hgemm2f_kernel only from lin_half_fold_tiles_ of its 128 x 64 tiles
    auto fold_h2_fills = [&](int M, int Nc) {
        return (int64_t)((M + 127) / 128) * ((Nc + 63) / 64) >= (int64_t)lin_half_fold_tiles_;
    };
    // A hidden-table layer's projection l with the gated store in its epilogue (gf.out = gout_l): LOUP_LIN_HALF's
    // panels on hgemm2f_kernel (gemm_h2_gate_fwd), else sgemm128_gate. A null: only asks whether it would run.
    auto gate0_call = [&](GenericLayer* L, int t, int l, const float* A, const GateArgs& gf) -> bool {
        const auto& p = L->hl_flin[l];
        if (A && d_lin_amax_) lin_amax_record(A, (int64_t)N * dimL(l) * L->K0, 16 * t + l, st);
        if (p.B)
            return (lin_half_parts_ & 16) && in.gemm_h2_gate_fwd &&
                   in.gemm_h2_gate_fwd(l, N * dimL(l), L->lin_cols[l], L->K0, A, p.sA, p.B, p.os, gf, false);
        if (polar_h2_ && in.gemm_h2_row_gate_fwd && L->lin_cols[l] >= 64 && L->K0 >= 64 &&
            N * dimL(l) >= polar_h2_min_m_) {
            auto it = ph2_map_.find(L->C0[l]);
            if (it != ph2_map_.end() && it->second.B) {
                if (A) in.gemm_h2_row_gate_fwd(l, N * dimL(l), L->lin_cols[l], L->K0, A, it->second.B, it->second.pB, gf, nullptr);
                return true;
            }
        }
        return in.gemm_gate_fwd(l, N * dimL(l), L->lin_cols[l], L->K0, A, L->C0[l], gf);
    };
    // the degrees of such a layer whose gated store would run in the projection (asks only; gf: natoms and C)
    auto vform0_ask = [&](GenericLayer* L, int t, const GateArgs& gf) -> unsigned {
        unsigned v = 0;
        if (L->first_hidden && !L->skip_folded && L->kind == 4 && L->res0_table && in.gemm_gate_fwd)
            for (int l = L->lmax_x + 1; l < 4; ++l)
                if (gate0_call(L, t, l, nullptr, gf)) v |= 1u << l;
        return v;
    };
    auto forward_layer = [&](int t, double* energy) {
        GenericLayer* L = layers_[t];
        const int Cx = L->C_x, Cm = L->C_msg, Co = L->C_out;
        // the hidden-table layer 0 needs x only for the nonlinear residual (none with the S59 residual table); with
        // the S57 fold, later layers' x comes from the earlier layers' folded consumers instead
        if ((!L->first_hidden || L->kind == 4) && !(chain_fold_ && t > 0) && !L->res0_table) {
            if (L->lmax_x > 0) fork();
            for (int l = 0; l <= L->lmax_x; ++l) {
                if (L->lmax_x > 0) lane(l);
                run_gemm(false, Nx * dimL(l), Cx, L->C_in, 1.0f, L->h_in[l], L->C_in, L->W_up[l], Cx, 0.0f, L->x[l], Cx);
                for (const auto& e : L->x_extra[l])  // long-range hidden fold: earlier layers' y terms
                    run_gemm(false, Nx * dimL(l), Cx, L->C_in, 1.0f, layers_[e.src]->y[l], L->C_in, e.W, Cx, 1.0f,
                            L->x[l], Cx);
            }
            if (L->lmax_x > 0) join();
        }
        if (L->first_hidden) {
            const int Kh = L->K0;
            const int fwd_blk = ((std::max(L->J - 1, 64) + 31) / 32) * 32;
            if (L->jit_fwd)
                launch_jit_as(first_fwd_kernel, L->jit_fwd, N, fwd_blk, 0, st, N, S, L->intervals, L->J, L->Jt, L->K0,
                              L->grid, sh_perm_, L->pair_sym, in.act_row_ptr, in.act_senders, in.act_vec_len,
                              in.species, L->table, L->S0[0], L->S0[1], L->S0[2], L->S0[3], L->D);
            else
                first_fwd_kernel<<<N, fwd_blk, 0, st>>>(
                    N, S, L->intervals, L->J, L->Jt, L->K0, L->grid, sh_perm_, L->pair_sym, in.act_row_ptr,
                    in.act_senders, in.act_vec_len, in.species, L->table, L->S0[0], L->S0[1], L->S0[2], L->S0[3], L->D);
            if (L->skip_folded) {
                fork();
                for (int l = 0; l < 4; ++l)
                    species_proj(L->hs_flin[l], 16 * t + l, false, l, Kh, Cm, L->S0[l], L->C0[l], (int64_t)Kh * Cm,
                                 L->msgf[l], 4 * l);
                join();
            } else {
                // Residual-free degrees l >= 1 (l > lmax_x; res0 from the S59 table) whose projection can take the gate
                // into its epilogue store gout_l instead of lin_l (gate_vform): the l = 0 projection, whose gate logits
                // those epilogues read, runs first.
                GateArgs gf;
                gf.natoms = N;
                gf.C = Cm;
                gf.D = L->D;
                gf.alpha = L->alpha;
                gf.beta = L->beta;
                gf.cg = L->gate_cg;
                gf.lin0 = L->lin[0];
                gf.res0 = L->res0_table;
                gf.res0_sp = in.species;
                auto gate0 = [&](int l, const float* A) { return gate0_call(L, t, l, A, gf); };
                gate_vform[t] = vform0_ask(L, t, gf);
                if (gate_vform[t])
                    proj(L->hl_flin[0], 16 * t, false, N, L->lin_cols[0], Kh, L->S0[0], L->C0[0], L->lin[0]);
                fork();
                for (int l = gate_vform[t] ? 1 : 0; l < 4; ++l) {
                    lane(l);
                    gf.out = L->gout[l];
                    if (!(gate_vform[t] >> l & 1u) || !gate0(l, L->S0[l])) {
                        gate_vform[t] &= ~(1u << l);
                        proj(L->hl_flin[l], 16 * t + l, false, N * dimL(l), L->lin_cols[l], Kh, L->S0[l], L->C0[l],
                             L->lin[l]);
                    }
                }
                if (L->kind == 4 && !L->res0_table) {
                    for (int l = 0; l <= L->lmax_x; ++l) {
                        lane(4 + l);
                        run_gemm(false, N * dimL(l), L->lin_cols[l], Cx, 1.0f, L->x[l], Cx, L->W_res[l],
                                L->lin_cols[l], 0.0f, L->res[l], L->lin_cols[l]);
                    }
                }
                join();
            }
        } else {
            L->last_preloc = can_preloc(L);
            const int var = (conv_var_override_ >= 0 && conv_var_override_ < conv_fwd_variants(L->lmax_x, Cx))
                                ? conv_var_override_ : L->fwd_var;
            launch_conv_fwd(L->lmax_x, st, N, Cx, S, L->intervals, L->grid, sh_perm_, L->pair_sym, in.act_row_ptr,
                            in.act_senders, in.act_vec_len, in.act_slab, in.act_t_sx, L->last_preloc, in.species,
                            L->packed_table, L->density_table, L->cubic_bias_shift, L->x, L->mid, L->D, var,
                            res_assigns(L, t) ? nullptr : L->dx, L->conv_prune, L->jit_conv_on ? &L->jit_conv : nullptr);
            const int n_proj = L->conv_prune ? 1 : 4;  // S125: lin_1..3 stay zero (W_lin_1..3 = 0)
            if (L->skip_folded) {
                fork();
                for (int l = 0; l < n_proj; ++l)
                    species_gemm(false, l, L->n_l3[l] * Cx, Cm, L->mid[l], L->W_lin[l],
                                 (int64_t)L->n_l3[l] * Cx * Cm, 0.0f, L->msgf[l], 4 * l);
                join();
            } else {
                // gemm_gate_fwd_l1 (LOUP_GATE_FWD_L1, default 1 on sm_120; whole passes only): a later kind-4
                // layer's residual-free degrees l > lmax_x store gout_l from their lin_l GEMM's epilogue (gate_vform;
                // 1: own sgemm128 forced above 1,600 rows, none below, 2: where it is the plan), as layer 0; the l = 0
                // projection and residual, whose sum the epilogues' gate logits read, first.
                GateArgs gf;
                gf.natoms = N;
                gf.C = Cm;
                gf.D = L->D;
                gf.alpha = L->alpha;
                gf.beta = L->beta;
                gf.cg = L->gate_cg;
                gf.lin0 = L->lin[0];
                gf.res0 = L->res[0];
                gf.res0_sp = nullptr;
                // LOUP_LIN_HALF's panels: the gated store on hgemm2f_kernel (gemm_h2_gate_fwd), else sgemm128_gate
                auto gate1 = [&](int l, const float* A) {
                    const auto& p = L->hl_flin[l];
                    const int M = N * dimL(l), K = L->n_l3[l] * Cx;
                    if (A && d_lin_amax_) lin_amax_record(A, (int64_t)M * K, 16 * t + l, st);
                    if (p.B)
                        return (lin_half_parts_ & 16) && in.gemm_h2_gate_fwd &&
                               in.gemm_h2_gate_fwd(l, M, L->lin_cols[l], K, A, p.sA, p.B, p.os, gf, true);
                    if (polar_h2_ && in.gemm_h2_row_gate_fwd && L->lin_cols[l] >= 64 && K >= 64 &&
                        M >= polar_h2_min_m_) {
                        auto it = ph2_map_.find(L->W_lin[l]);
                        if (it != ph2_map_.end() && it->second.B) {
                            if (A) in.gemm_h2_row_gate_fwd(l, M, L->lin_cols[l], K, A, it->second.B, it->second.pB, gf, nullptr);
                            return true;
                        }
                    }
                    return in.gemm_gate_fwd_l1(l, M, L->lin_cols[l], K, A, L->W_lin[l], gf);
                };
                // whole passes, and the tiled call's last-layer sweep, which runs the layer's forward and backward
                // in one recording (list 20, gen-gate-fwd-tiled; LOUP_GATE_FWD_TILED=0: whole passes only)
                const bool gate1_sweep = sweep == GenericPassInputs::SweepAll ||
                                         (gate_fwd_tiled_ && sweep == GenericPassInputs::SweepLast && t == n_layers_ - 1);
                // list 27 item 3 (LOUP_LIN_HALF_PARTS bit 64): a residual degree l = 1 .. min(lmax_x, 2) stores gout_l
                // = g (lin_l s + res_l) from its W_res projection's epilogue (gate_rform), which reads lin_l: lin_l runs
                // first on the same lane, and res_l is never stored (the backward reads gout_l in its place)
                auto gater = [&](int l, const float* A) {
                    const auto& p = L->hl_fres[l];
                    if (!p.B || !(lin_half_parts_ & 64) || !in.gemm_h2_gate_fwd) return false;
                    const int M = N * dimL(l);
                    if (A && d_lin_amax_) lin_amax_record(A, (int64_t)M * Cx, 16 * t + 4 + l, st);
                    GateArgs gr = gf;
                    gr.lin_l = L->lin[l];
                    return in.gemm_h2_gate_fwd(l, M, L->lin_cols[l], Cx, A, p.sA, p.B, p.os, gr, true);
                };
                if (L->kind == 4 && in.gemm_gate_fwd_l1 && n_proj == 4 && gate1_sweep) {
                    for (int l = L->lmax_x + 1; l < 4; ++l)  // A null: only asks
                        if (gate1(l, nullptr)) gate_vform[t] |= 1u << l;
                    for (int l = 1; l <= std::min(L->lmax_x, 2); ++l)
                        if (gater(l, nullptr)) gate_rform[t] |= 1u << l;
                }
                const int l_first = (gate_vform[t] | gate_rform[t]) ? 1 : 0;
                if (gate_vform[t] | gate_rform[t]) {
                    fork();
                    lane(0);
                    proj(L->hl_flin[0], 16 * t, false, N, L->lin_cols[0], L->n_l3[0] * Cx, L->mid[0], L->W_lin[0],
                         L->lin[0]);
                    lane(4);
                    proj(L->hl_fres[0], 16 * t + 4, false, N, L->lin_cols[0], Cx, L->x[0], L->W_res[0], L->res[0]);
                    join();
                }
                fork();
                for (int l = l_first; l < n_proj; ++l) {
                    lane(l);
                    gf.out = L->gout[l];
                    if (!(gate_vform[t] >> l & 1u) || !gate1(l, L->mid[l])) {
                        gate_vform[t] &= ~(1u << l);
                        proj(L->hl_flin[l], 16 * t + l, false, N * dimL(l), L->lin_cols[l], L->n_l3[l] * Cx,
                             L->mid[l], L->W_lin[l], L->lin[l]);
                    }
                    if ((gate_rform[t] >> l & 1u) && !gater(l, L->x[l]))  // declined: res_l on its own lane below
                        gate_rform[t] &= ~(1u << l);
                }
                if (L->kind == 4) {
                    for (int l = l_first; l <= L->lmax_x; ++l) {
                        if (gate_rform[t] >> l & 1u) continue;
                        lane(4 + l);
                        proj(L->hl_fres[l], 16 * t + 4 + l, false, N * dimL(l), L->lin_cols[l], Cx, L->x[l],
                             L->W_res[l], L->res[l]);
                    }
                }
                join();
            }
        }
        if (L->kind == 4) {
            const float* res0 = L->res0_table ? L->res0_table : L->res[0];
            const int* res0_sp = L->res0_table ? in.species : nullptr;
            gate_fwd_kernel<<<N, Cm, 0, st>>>(N, Cm, L->lmax_x, L->D, L->alpha, L->beta, L->gate_cs, L->gate_cg,
                                              L->lin[0], L->lin[1], L->lin[2], L->lin[3], res0, res0_sp,
                                              L->lmax_x >= 1 ? L->res[1] : nullptr,
                                              L->lmax_x >= 2 ? L->res[2] : nullptr, L->gout[0], L->gout[1],
                                              L->gout[2], L->gout[3], gate_vform[t] | gate_rform[t]);
            fork();
            for (int l = 0; l < 4; ++l) {
                lane(l);
                if (L->f2 && in.gemm_h2_fwd)  // LOUP_W2F_HALF (setup_w2f_half)
                    in.gemm_h2_fwd(N * dimL(l), Cm, Cm, L->gout[l], L->w2f_sA[l], L->w2fh[l], L->w2f_oscale[l],
                                   L->msgf[l], 0.0f);
                else
                    run_gemm(false, N * dimL(l), Cm, Cm, 1.0f, L->gout[l], Cm, L->W_2[l], Cm, 0.0f, L->msgf[l], Cm);
            }
            join();
        }
        const float* Dp = (L->kind == 2 || L->kind == 3) ? L->D : nullptr;
        // the fused product + readout kernel runs one thread per channel (<= 512)
        const bool ro1_fused = L->ro_folded && L->readout == 1 && L->ro_H == 16 && L->prod->ro1_128 != nullptr &&
                               L->C_out <= 512 && !in.atom_energy;
        if (ro1_fused) {
            if (L->sc_has[0] && !chain_fold_) {  // with the S57 fold, hid already holds the skip's share
                if (L->sc_species) {
                    fork();
                    species_gemm(false, 0, L->C_in, 16, L->h_in[0], L->ro_W1_sc, (int64_t)L->C_in * 16, 0.0f, L->hid, 0);
                    join();
                } else {
                    run_gemm(false, N, 16, L->C_in, 1.0f, L->h_in[0], L->C_in, L->ro_W1_sc, 16, 0.0f, L->hid, 16);
                }
            }
            float* dm[4];
            for (int l = 0; l < 4; ++l) dm[l] = (L->kind == 1 || L->kind == 3) ? L->dlin[l] : L->dmsgf[l];
            float* dDp = (L->kind == 2 || L->kind == 3) ? L->dD : nullptr;
            if (L->lin_compact)  // S130
                launch_prod_ro1c_fused(L->prod, st, N, L->lin_cols[0], L->n_ang, L->lin_cols, L->d_pos, L->msgf, Dp,
                                       L->prod_w,
                                       L->prod_species ? (int64_t)L->prod_segw * L->lin_cols[0] : 0, in.species,
                                       L->sc_has[0] ? L->hid : nullptr, L->ro_W1, L->ro_b1, L->ro_W2, L->ro_act_c,
                                       L->dhid, energy, dm, dDp);
            else
                launch_prod_ro1_fused(L->h2 && in.gemm_h2_gate ? &L->prod_h2 : L->prod, st, N, Cm, L->msgf, Dp,
                                      L->prod_w,
                                      L->prod_species ? (int64_t)L->prod_segw * Cm : 0, in.species,
                                      L->sc_has[0] ? L->hid : nullptr, L->ro_W1, L->ro_b1, L->ro_W2, L->ro_act_c,
                                      L->dhid, energy, dm, dDp,
                                      L->conv_prune && L->kind != 4);  // S127: m_1..3 = lin_1..3 = 0 (not for kind 4)
            return;
        }
        const float* ro_fw = fused_readout(t);
        if (L->lin_compact)  // S132
            launch_prod_c(L->prod, false, st, N, Cm, L->kept, L->n_ang, L->lin_cols, L->d_order, L->d_pos, L->msgf, Dp,
                          L->prod_w,
                          L->prod_species ? (int64_t)L->prod_segw * Cm : 0, in.species, L->y, nullptr, nullptr, ro_fw,
                          ro_fw ? energy : nullptr);
        else
            launch_prod(L->prod, false, st, N, Cm, L->msgf, Dp, L->prod_w,
                        L->prod_species ? (int64_t)L->prod_segw * Cm : 0, in.species, L->y, nullptr, nullptr, ro_fw,
                        ro_fw ? energy : nullptr);
        if (chain_fold_ && t < n_layers_ - 1) {
            // S57: y_t (and layer 0's species-only skip input) into every downstream consumer of hout_t; the first
            // write of a destination in this pass assigns it, later ones accumulate (program order: race-free).
            // S59: the species-table skip terms (layer 0, l 0; each its destination's first write) are gathered on
            // the main stream before the lanes fork, so the lane GEMMs accumulate onto them. S73: the energy targets
            // (folded linear readouts) also run there: E += y_t[0] . Wy (assigning the seed dy_t[0] = Wy; S100: in
            // the product kernels instead, see fused_readout) and the per-species constant of the layer-0 skip.
            for (const auto& ft : L->fold[0]) {
                if (ft.dest_layer == -2) {
                    if (!ro_fw)
                        ro_linear_kernel<<<(N + 7) / 8, 256, 0, st>>>(N, Co, L->y[0], ft.Wy, L->dy[0], energy, true,
                                                                      in.atom_energy);
                    if (ft.Th)
                        species_energy_kernel<<<(N + 255) / 256, 256, 0, st>>>(N, in.species, ft.Th, energy,
                                                                               in.atom_energy);
                    continue;
                }
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
                    if (ft.dest_layer == -2) continue;
                    lane(lane_i++);
                    float* dst = ft.dest_layer >= 0 ? layers_[ft.dest_layer]->x[l] : layers_.back()->hid;
                    float beta = std::find(written.begin(), written.end(), dst) != written.end() ? 1.0f : 0.0f;
                    if (ft.Wh) {
                        if (ft.ns_h > 1)
                            species_gemm(false, l, L->C_in, ft.cols, L->h_in[l], ft.Wh, (int64_t)L->C_in * ft.cols,
                                         beta, dst);
                        else
                            run_gemm(false, N * dimL(l), ft.cols, L->C_in, 1.0f, L->h_in[l], L->C_in, ft.Wh, ft.cols,
                                    beta, dst, ft.cols);
                        beta = 1.0f;
                    }
                    if (ft.ns > 1) {
                        species_gemm(false, l, Co, ft.cols, L->y[l], ft.Wy, (int64_t)Co * ft.cols, beta, dst);
                    } else {
                        // list 25 item 1 (LOUP_LIN_HALF_PARTS bit 256): on hgemm2f_kernel where the target has panels
                        if (d_lin_amax_) lin_amax_record(L->y[l], (int64_t)N * dimL(l) * Co, 128 + 16 * t + l, st);
                        if (ft.hf.B && in.gemm_h2_fwd && fold_h2_fills(N * dimL(l), ft.cols))
                            in.gemm_h2_fwd(N * dimL(l), ft.cols, Co, L->y[l], ft.hf.sA, ft.hf.B, ft.hf.os, dst, beta);
                        else
                            run_gemm(false, N * dimL(l), ft.cols, Co, 1.0f, L->y[l], Co, ft.Wy, ft.cols, beta, dst,
                                    ft.cols);
                    }
                    written.push_back(dst);
                }
            join();
        } else if (!L->ro_folded) {
            if (L->lmax_out > 0) fork();
            for (int l = 0; l <= L->lmax_out; ++l) {
                if (L->hfold[l]) continue;  // long-range hidden fold: hout[l] is y[l], its consumers composed
                if (L->lmax_out > 0) lane(l);
                run_gemm(false, N * dimL(l), Co, Co, 1.0f, L->y[l], Co, L->W_p[l], Co, 0.0f, L->hout[l], Co);
                if (L->sc_has[l]) {
                    if (L->sc_species)
                        species_gemm(false, l, L->C_in, Co, L->h_in[l], L->W_sc[l], (int64_t)L->C_in * Co, 1.0f,
                                     L->hout[l]);
                    else
                        run_gemm(false, N * dimL(l), Co, L->C_in, 1.0f, L->h_in[l], L->C_in, L->W_sc[l], Co, 1.0f,
                                L->hout[l], Co);
                    for (const auto& e : L->sc_extra[l])
                        run_gemm(false, N * dimL(l), Co, L->C_in, 1.0f, layers_[e.src]->y[l], L->C_in, e.W, Co, 1.0f,
                                L->hout[l], Co);
                }
            }
            if (L->lmax_out > 0) join();
        }
        // readout (energy) and its gradient seed into dhout[0] (or dhid directly when ro_folded); S73: a folded
        // layer's linear readout ran with its fold targets
        if (L->readout == 0) {
            if (!(chain_fold_ && t < n_layers_ - 1))
                ro_linear_kernel<<<(N + 7) / 8, 256, 0, st>>>(N, Co, L->hout[0], L->ro_w, L->dhout[0], energy, false,
                                                              in.atom_energy);
        } else if (L->readout > 0) {
            const int H = L->ro_H;
            const unsigned gH = (unsigned)(((int64_t)N * H + 255) / 256);
            if (L->ro_folded) {
                const bool hid_folded = chain_fold_ &&
                                        std::find(written.begin(), written.end(), L->hid) != written.end();
                run_gemm(false, N, H, Co, 1.0f, L->y[0], Co, L->ro_W1, H, hid_folded ? 1.0f : 0.0f, L->hid, H);
                if (L->sc_has[0] && !chain_fold_) {
                    if (L->sc_species) {
                        fork();
                        species_gemm(false, 0, L->C_in, H, L->h_in[0], L->ro_W1_sc, (int64_t)L->C_in * H, 1.0f,
                                     L->hid, 0);
                        join();
                    } else {
                        run_gemm(false, N, H, L->C_in, 1.0f, L->h_in[0], L->C_in, L->ro_W1_sc, H, 1.0f, L->hid, H);
                    }
                }
            } else {
                run_gemm(false, N, H, Co, 1.0f, L->hout[0], Co, L->ro_W1, H, 0.0f, L->hid, H);
            }
            if (L->readout == 1) {
                ro_mlp_kernel<<<gH, 256, 0, st>>>(N, H, L->ro_act_c, L->hid, L->ro_b1, L->ro_W2, nullptr, L->dhid,
                                                  energy, 1, in.atom_energy);
            } else {
                // stage 1: hid -> act1 (pre kept in dhid2 as scratch), stage 2 on hid2 = act1 @ Wm
                ro_mlp_kernel<<<gH, 256, 0, st>>>(N, H, L->ro_act_c, L->hid, L->ro_b1, nullptr, L->dhid2, nullptr,
                                                  nullptr, 0);
                run_gemm(false, N, H, H, 1.0f, L->hid, H, L->ro_Wm, H, 0.0f, L->hid2, H);
                ro_mlp_kernel<<<gH, 256, 0, st>>>(N, H, L->ro_act_c, L->hid2, L->ro_bm, L->ro_W2, nullptr, L->dhid,
                                                  energy, 1, in.atom_energy);
                // dhid = (dhid @ Wm^T) * c silu'(pre1)
                run_gemm(true, N, H, H, 1.0f, L->dhid, H, L->ro_Wm, H, 0.0f, L->hid2, H);
                ro_act_bwd_kernel<<<gH, 256, 0, st>>>((int64_t)N * H, L->ro_act_c, L->dhid2, L->hid2);
                CUDA_CHECK_THROW(cudaMemcpyAsync(L->dhid, L->hid2, (size_t)N * H * sizeof(float),
                                                 cudaMemcpyDeviceToDevice, st));
            }
            if (!L->ro_folded)
                run_gemm(true, N, Co, H, 1.0f, L->dhid, H, L->ro_W1, H, 1.0f, L->dhout[0], Co);
        }
    };

    // ---------------- backward of layer t: from the gradient of its output (dhout, or the fused readout's dhid)
    // to its edge gradients and, for t > 0, the gradient of the previous layer's output (dh_in).
    auto backward_layer = [&](int t) {
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
                    run_gemm(true, N, L->C_in, 16, 1.0f, L->dhid, 16, L->ro_W1_sc, 16, 1.0f, L->dh_in[0], L->C_in);
                }
            }
        } else {
            if (L->ro_folded) {
                const int H = L->ro_H;
                const bool fork_sc = L->sc_has[0] && t > 0 && !chain_fold_;
                if (fork_sc) fork();
                if (fork_sc) lane(0);
                run_gemm(true, N, Co, H, 1.0f, L->dhid, H, L->ro_W1, H, 0.0f, L->dy[0], Co);
                if (fork_sc) {
                    if (L->sc_species)
                        species_gemm(true, 0, H, L->C_in, L->dhid, L->ro_W1_sc, (int64_t)L->C_in * H, 1.0f,
                                     L->dh_in[0], 1);
                    else {
                        lane(1);
                        run_gemm(true, N, L->C_in, H, 1.0f, L->dhid, H, L->ro_W1_sc, H, 1.0f, L->dh_in[0], L->C_in);
                    }
                    join();
                }
            } else if (chain_fold_) {
                // S57: dy_t[l] from every downstream consumer's gradient through the composed weights (all final:
                // later layers and the readout ran first); one lane per degree, its terms in order. S73: a folded
                // linear readout seeded dy_t[0] in the forward, so every term accumulates onto it (S100: unless the
                // product backward adds the seed, then the first term assigns).
                const bool seed_in_prod = fused_readout(t) != nullptr;
                fork();
                for (int l = 0; l <= L->lmax_out; ++l) {
                    lane(l);
                    float beta = 0.0f;
                    for (const auto& ft : L->fold[l])
                        if (ft.dest_layer == -2 && !seed_in_prod) beta = 1.0f;
                    for (const auto& ft : L->fold[l]) {
                        if (ft.dest_layer == -2) continue;
                        const float* dsrc = ft.dest_layer >= 0 ? layers_[ft.dest_layer]->dx[l] : layers_.back()->dhid;
                        if (ft.ns > 1) {
                            species_gemm(true, l, ft.cols, Co, dsrc, ft.Wy, (int64_t)Co * ft.cols, beta, L->dy[l]);
                        } else {
                            // list 25 item 1 (bit 256): dy_t[l] += dsrc Wy^T on hgemm2f_kernel
                            if (d_lin_amax_)
                                lin_amax_record(dsrc, (int64_t)N * dimL(l) * ft.cols, 128 + 16 * t + 8 + l, st);
                            if (ft.hb.B && in.gemm_h2_fwd && fold_h2_fills(N * dimL(l), Co))
                                in.gemm_h2_fwd(N * dimL(l), Co, ft.cols, dsrc, ft.hb.sA, ft.hb.B, ft.hb.os, L->dy[l],
                                               beta);
                            else
                                run_gemm(true, N * dimL(l), Co, ft.cols, 1.0f, dsrc, ft.cols, ft.Wy, ft.cols, beta,
                                        L->dy[l], Co);
                        }
                        beta = 1.0f;
                    }
                }
                join();
            } else {
                const bool fork_p = L->lmax_out > 0 || (L->has_sc && t > 0);
                if (fork_p) fork();
                for (int l = 0; l <= L->lmax_out; ++l) {
                    if (L->hfold[l]) continue;  // long-range hidden fold: dy[l] (= dhout[l]) accumulated in place
                    if (fork_p) lane(l);
                    run_gemm(true, N * dimL(l), Co, Co, 1.0f, L->dhout[l], Co, L->W_p[l], Co, 0.0f, L->dy[l], Co);
                    if (L->sc_has[l] && t > 0) {
                        if (L->sc_species)
                            species_gemm(true, l, Co, L->C_in, L->dhout[l], L->W_sc[l], (int64_t)L->C_in * Co, 1.0f,
                                         L->dh_in[l], 4);
                        else {
                            lane(4 + l);
                            run_gemm(true, N * dimL(l), L->C_in, Co, 1.0f, L->dhout[l], Co, L->W_sc[l], Co, 1.0f,
                                    L->dh_in[l], L->C_in);
                            for (const auto& e : L->sc_extra[l])
                                run_gemm(true, N * dimL(l), L->C_in, Co, 1.0f, L->dhout[l], Co, e.W, Co, 1.0f,
                                        layers_[e.src]->dy[l], L->C_in);
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
            if (L->lin_compact)  // S132
                launch_prod_c(L->prod, true, st, N, Cm, L->kept, L->n_ang, L->lin_cols, L->d_order, L->d_pos, L->msgf,
                              Dp,
                              L->prod_w, L->prod_species ? (int64_t)L->prod_segw * Cm : 0, in.species, L->dy, dm, dDp,
                              fused_readout(t));
            else
                launch_prod(L->h2 && in.gemm_h2_gate ? &L->prod_h2 : L->prod, true, st, N, Cm, L->msgf, Dp, L->prod_w,
                            L->prod_species ? (int64_t)L->prod_segw * Cm : 0, in.species, L->dy, dm, dDp,
                            fused_readout(t));
        }
        if (L->kind == 4) {
            const float* res0 = L->res0_table ? L->res0_table : L->res[0];
            const int* res0_sp = L->res0_table ? in.species : nullptr;
            const bool rdl = res_from_dlin(L, t);  // LOUP_RES_BWD 2: no dpre
            float* dpre0 = t > 0 && !rdl ? L->dpre[0] : nullptr;
            const unsigned vform = gate_vform[t];  // these degrees saved gout_l instead of lin_l
            const unsigned rform = gate_rform[t];  // these saved gout_l instead of res_l (list 27 item 3)
            const float* lins[4];
            const float* ress[3] = {nullptr, nullptr, nullptr};
            for (int l = 0; l < 4; ++l) lins[l] = (vform >> l & 1u) ? L->gout[l] : L->lin[l];
            for (int l = 1; l <= std::min(L->lmax_x, 2); ++l) ress[l] = (rform >> l & 1u) ? L->gout[l] : L->res[l];
            // dgout_l = dmsgf_l @ W_2[l]^T; degrees whose GEMM takes the gate backward into its epilogue never write
            // dgout_l, gate_bwd_kernel does the others. LOUP_W2T_HALF (L->h2): dmsgf_l holds the product kernel's two
            // binary16 panels, the two-half tensor-core GEMM takes every degree's gate
            const bool h2 = L->h2 && in.gemm_h2_gate;
            unsigned fused = 0;
            fork();
            for (int l = 0; l < 4; ++l) {
                lane(l);
                bool done = false;
                if (in.gemm_gate_bwd || h2) {
                    GateArgs g;
                    g.natoms = N;
                    g.C = Cm;
                    g.D = L->D;
                    g.alpha = L->alpha;
                    g.beta = L->beta;
                    g.cs = L->gate_cs;
                    g.cg = L->gate_cg;
                    g.lin0 = L->lin[0];
                    g.lin_l = lins[l];
                    g.res0 = res0;
                    g.res0_sp = res0_sp;
                    g.res_l = (l >= 1 && l <= 2) ? ress[l] : nullptr;
                    g.dp0 = dpre0;
                    g.dp_l = (l >= 1 && t > 0 && !rdl && l <= L->lmax_x && l <= 2) ? L->dpre[l] : nullptr;
                    g.dl0 = L->dlin[0];
                    g.dl_l = L->dlin[l];
                    g.dD = L->dD;
                    g.vform = vform >> l & 1u;
                    g.rform = rform >> l & 1u;
                    if (h2) {
                        in.gemm_h2_gate(l, N * dimL(l), Cm, Cm, L->dmsgf[l], L->w2h[l], L->w2h_oscale[l], g);
                        done = true;
                    } else if (polar_h2_ && in.gemm_h2_row_gate_bwd && Cm >= 64 && Cm % 64 == 0 &&
                               N * dimL(l) >= polar_h2_min_m_) {
                        auto it = ph2_map_.find(L->W_2[l]);
                        if (it != ph2_map_.end() && it->second.BT) {
                            in.gemm_h2_row_gate_bwd(l, N * dimL(l), Cm, Cm, L->dmsgf[l], it->second.BT, it->second.pBT, g, nullptr);
                            done = true;
                        } else {
                            done = in.gemm_gate_bwd(l, N * dimL(l), Cm, Cm, L->dmsgf[l], L->W_2[l], g);
                        }
                    } else {
                        done = in.gemm_gate_bwd(l, N * dimL(l), Cm, Cm, L->dmsgf[l], L->W_2[l], g);
                    }
                }
                if (done) fused |= 1u << l;
                else run_gemm(true, N * dimL(l), Cm, Cm, 1.0f, L->dmsgf[l], Cm, L->W_2[l], Cm, 0.0f, L->dgout[l], Cm);
            }
            join();
            if (fused != 15u)
                gate_bwd_kernel<<<N, Cm, 0, st>>>(N, Cm, L->lmax_x, L->D, L->alpha, L->beta, L->gate_cs, L->gate_cg,
                                                  L->lin[0], lins[1], lins[2], lins[3], res0, res0_sp, ress[1],
                                                  ress[2], L->dgout[0], L->dgout[1], L->dgout[2], L->dgout[3], dpre0,
                                                  (t > 0 && !rdl && L->lmax_x >= 1) ? L->dpre[1] : nullptr,
                                                  (t > 0 && !rdl && L->lmax_x >= 2) ? L->dpre[2] : nullptr, L->dlin[0],
                                                  L->dlin[1], L->dlin[2], L->dlin[3], L->dD, fused, vform, rform);
        }
        const bool density = L->kind >= 2;
        if (L->first_hidden) {
            // dS0 (aliases S0) = dlin @ C0^T (or dmsgf @ C0_folded^T when skip_folded)
            const int Kh = L->K0;
            if (L->skip_folded) {
                fork();
                for (int l = 0; l < 4; ++l)
                    species_proj(L->hs_blin[l], 16 * t + 8 + l, true, l, Cm, Kh, L->dmsgf[l], L->C0[l],
                                 (int64_t)Kh * Cm, L->S0[l], 4 * l);
                join();
            } else {
                fork();
                for (int l = 0; l < 4; ++l) {
                    lane(l);
                    proj(L->hl_blin[l], 16 * t + 8 + l, true, N * dimL(l), Kh, L->lin_cols[l], L->dlin[l], L->C0[l],
                         L->S0[l]);
                }
                if (L->kind == 4 && t > 0) {
                    for (int l = 0; l <= L->lmax_x; ++l) {
                        lane(4 + l);
                        run_gemm(true, N * dimL(l), Cx, L->lin_cols[l], 1.0f, L->dpre[l], L->lin_cols[l], L->W_res[l],
                                L->lin_cols[l], 1.0f, L->dx[l], Cx);
                    }
                }
                join();
            }
            if (L->J - 1 <= 96) {
                if (has_edges) {
                    using EdgeBwd = void (*)(const int*, int, int, int, int, int, SplineGrid, int, int, const int*, const int*,
                                             const float4*, const int*, const float4*, const float*, const float*, const float*,
                                             const float*, const float*, const float*, ZblParams, double*, double*, double*);
                    const EdgeBwd tbl[4][2][2] = {
                        {{first_bwd_edge_kernel<2, false, false>, first_bwd_edge_kernel<2, false, true>},
                         {first_bwd_edge_kernel<2, true, false>, first_bwd_edge_kernel<2, true, true>}},
                        {{first_bwd_edge_kernel<3, false, false>, first_bwd_edge_kernel<3, false, true>},
                         {first_bwd_edge_kernel<3, true, false>, first_bwd_edge_kernel<3, true, true>}},
                        {{first_bwd_edge_kernel<2, false, false, 65>, first_bwd_edge_kernel<2, false, true, 65>},
                         {first_bwd_edge_kernel<2, true, false, 65>, first_bwd_edge_kernel<2, true, true, 65>}},
                        {{first_bwd_edge_kernel<2, false, false, 66>, first_bwd_edge_kernel<2, false, true, 66>},
                         {first_bwd_edge_kernel<2, true, false, 66>, first_bwd_edge_kernel<2, true, true, 66>}}};
                    // 2: S142's fixed 64-column kernel; 3: list 11's 64 + 1 columns (J 66: MH-1)
                    const int cols = L->J == 66 ? 3 : (L->J - 1 > 64 ? 1 : (L->J == 65 ? 2 : 0));
                    const EdgeBwd aot = tbl[cols][has_zbl_ ? 1 : 0][in.compute_stress ? 1 : 0];
                    const float* dD_arg = density ? L->dD : nullptr;
                    const unsigned nblk = (unsigned)((in.E_cap + 127) / 128);
                    if (const void* jk = L->jit_bwd_edge[in.compute_stress ? 1 : 0])
                        launch_jit_as(aot, jk, nblk, 128, 0, st, in.E_dev, S, L->intervals, L->J, L->Jt, L->K0,
                                      L->grid, sh_perm_, L->pair_sym, in.act_senders, in.act_receivers,
                                      in.act_vec_len, in.species, L->table, L->S0[0], L->S0[1], L->S0[2], L->S0[3],
                                      dD_arg, d_edge_dvec_, zbl_p, in.energy, in.forces, in.virial);
                    else
                        aot<<<nblk, 128, 0, st>>>(
                            in.E_dev, S, L->intervals, L->J, L->Jt, L->K0, L->grid, sh_perm_, L->pair_sym,
                            in.act_senders, in.act_receivers, in.act_vec_len, in.species, L->table, L->S0[0],
                            L->S0[1], L->S0[2], L->S0[3], dD_arg, d_edge_dvec_, zbl_p, in.energy, in.forces,
                            in.virial);
                }
            } else {
                const size_t g_smem = (size_t)16 * S * (L->J - 1) * sizeof(float);
                if (g_smem <= 46 * 1024)  // + 2 KB static shared memory within the 48 KB default
                    first_bwd_kernel<true><<<N, 256, g_smem, st>>>(
                        N, S, L->intervals, L->J, L->Jt, L->K0, L->grid, sh_perm_, L->pair_sym, in.act_row_ptr, in.act_senders,
                        in.act_vec_len, in.species, L->table, L->S0[0], L->S0[1], L->S0[2], L->S0[3],
                        density ? L->dD : nullptr, d_edge_dvec_);
                else
                    first_bwd_kernel<false><<<N, 256, 0, st>>>(
                        N, S, L->intervals, L->J, L->Jt, L->K0, L->grid, sh_perm_, L->pair_sym, in.act_row_ptr, in.act_senders,
                        in.act_vec_len, in.species, L->table, L->S0[0], L->S0[1], L->S0[2], L->S0[3],
                        density ? L->dD : nullptr, d_edge_dvec_);
            }
            return;
        }
        // dmid (aliases mid: the forward values are dead) = dlin @ W_lin^T (or dmsgf @ W_lin_folded^T); S125: dmid_0
        // only (the pruned conv backward reads no other)
        const int n_proj = L->conv_prune ? 1 : 4;
        if (L->skip_folded) {
            fork();
            for (int l = 0; l < n_proj; ++l)
                species_gemm(true, l, Cm, L->n_l3[l] * Cx, L->dmsgf[l], L->W_lin[l],
                             (int64_t)L->n_l3[l] * Cx * Cm, 0.0f, L->mid[l], 4 * l);
            join();
        } else {
            fork();
            for (int l = 0; l < n_proj; ++l) {
                lane(l);
                if (res_fused(L, t, l)) {
                    const GenericLayer::H2Proj &c = L->hl_bcat[l], &r = L->hl_bres[l];
                    GateArgs rs = res_rows(L, l);
                    rs.c2 = L->dx[l];
                    rs.n1 = L->n_l3[l] * Cx;
                    rs.os2 = r.os * (r.sA / c.sA);
                    if (d_lin_amax_)
                        lin_amax_record(L->dlin[l], (int64_t)N * dimL(l) * L->lin_cols[l], 16 * t + 8 + l, st);
                    in.gemm_h2_rows(N * dimL(l), rs.n1 + Cx, L->lin_cols[l], L->dlin[l], c.sA, c.B, c.os, L->mid[l],
                                    rs);
                    continue;
                }
                proj(L->hl_blin[l], 16 * t + 8 + l, true, N * dimL(l), L->n_l3[l] * Cx, L->lin_cols[l], L->dlin[l],
                     L->W_lin[l], L->mid[l]);
            }
            if (L->kind == 4 && t > 0) {
                // LOUP_RES_BWD: dx_l = dpre_l W_res[l]^T assigned (1) and from dlin_l (2), else added to the rows the
                // forward conv kernel zeroed
                const bool rdl = res_from_dlin(L, t);
                const float rbeta = res_assigns(L, t) ? 0.0f : 1.0f;
                for (int l = 0; l <= L->lmax_x; ++l) {
                    if (res_fused(L, t, l)) continue;  // ran with dmid_l
                    lane(4 + l);
                    if (rdl) {
                        const GenericLayer::H2Proj& p = L->hl_bres[l];
                        const float sA = L->hl_blin[l].sA;  // dlin's scale; the panels' 2^-e_B kept
                        const GateArgs rs = res_rows(L, l);
                        if (d_lin_amax_)
                            lin_amax_record(L->dlin[l], (int64_t)N * dimL(l) * L->lin_cols[l], 16 * t + 12 + l, st);
                        in.gemm_h2_rows(N * dimL(l), Cx, L->lin_cols[l], L->dlin[l], sA, p.B, p.os * (p.sA / sA),
                                        L->dx[l], rs);
                    } else {
                        proj(L->hl_bres[l], 16 * t + 12 + l, true, N * dimL(l), Cx, L->lin_cols[l], L->dpre[l],
                             L->W_res[l], L->dx[l], rbeta);
                    }
                }
            }
            join();
        }
        launch_conv_bwd(L->lmax_x, st, N, Cx, S, L->intervals, L->grid, sh_perm_, L->pair_sym, in.act_row_ptr,
                        in.act_senders, in.act_vec_len, in.act_slab, in.act_t_sx, can_preloc(L), in.species,
                        L->packed_table, L->density_table, L->cubic_bias_shift, L->x, L->mid,
                        density ? L->dD : nullptr, L->dx, t > 0, d_edge_dvec_, t == n_layers_ - 1, L->conv_prune,
                        L->jit_conv_on ? &L->jit_conv : nullptr, in.n_ghost == 0);
        if (t > 0 && !chain_fold_) {  // S57: the earlier layers' fold reads dx directly
            if (L->lmax_x > 0) fork();
            for (int l = 0; l <= L->lmax_x; ++l) {
                if (L->lmax_x > 0) lane(l);
                run_gemm(true, Nx * dimL(l), L->C_in, Cx, 1.0f, L->dx[l], Cx, L->W_up[l], Cx, 1.0f, L->dh_in[l], L->C_in);
                for (const auto& e : L->x_extra[l])
                    run_gemm(true, Nx * dimL(l), L->C_in, Cx, 1.0f, L->dx[l], Cx, e.W, Cx, 1.0f, layers_[e.src]->dy[l],
                            L->C_in);
            }
            if (L->lmax_x > 0) join();
        }
    };

    // Per-edge forces and virial from the edge gradients gathered so far, with or without ZBL.
    auto assemble_edges = [&](bool with_zbl) {
        if (!has_edges) return;
        auto kern = (with_zbl && has_zbl_)
                        ? (in.compute_stress ? assemble_forces_kernel<true, true> : assemble_forces_kernel<true, false>)
                        : (in.compute_stress ? assemble_forces_kernel<false, true> : assemble_forces_kernel<false, false>);
        kern<<<(in.E_cap + 255) / 256, 256, 0, st>>>(in.E_dev, S, in.act_vec_len, in.act_senders, in.act_receivers,
                                                    in.species, zbl_p, d_edge_dvec_, in.forces, in.virial, in.energy);
    };

    // ---------------- one sweep of a domain (GenericPassInputs::Sweep)
    // O31: layer 0's msgf and D (list 19, the gated layer 0: also lin_l, or gout_l for its gate_vform degrees, and res_0)
    // in the domain's own buffer (keep0, keep0_layout) for its forward and backward sweeps of layer 0; the layer's
    // pointers are given back when the pass returns
    if (keep0 && !sweep_keep_layer0()) throw std::logic_error("generic: keep0 given for a layer 0 that cannot keep");
    struct Keep0 {
        GenericLayer* L = nullptr;
        float* msgf[4] = {};
        float* lin[4] = {};
        float* gout[4] = {};
        float* D = nullptr;
        float* res0 = nullptr;
        ~Keep0() {
            if (!L) return;
            for (int l = 0; l < 4; ++l) {
                L->msgf[l] = msgf[l];
                L->lin[l] = lin[l];
                L->gout[l] = gout[l];
            }
            L->D = D;
            L->res[0] = res0;
        }
    } keep0_guard;
    unsigned keep0_vform = 0;  // the gated layer 0's gate_vform, as its forward asks (the backward runs no forward)
    if (keep0) {
        GenericLayer* L0 = layers_[0];
        keep0_guard.L = L0;
        for (int l = 0; l < 4; ++l) {
            keep0_guard.msgf[l] = L0->msgf[l];
            keep0_guard.lin[l] = L0->lin[l];
            keep0_guard.gout[l] = L0->gout[l];
        }
        keep0_guard.D = L0->D;
        keep0_guard.res0 = L0->res[0];
        size_t off[10];
        keep0_layout(L0, N, off);
        for (int l = 0; l < 4; ++l) L0->msgf[l] = in.keep0 + off[l];
        L0->D = in.keep0 + off[4];
        if (L0->kind == 4) {
            // the forward sweep asks and leaves its answer for the domain's backward sweep, which must read the slots
            // as they were written (a plan for the gated GEMM may be tuned in between, so an ask there may differ)
            if (sweep == GenericPassInputs::SweepForward) {
                GateArgs ga;
                ga.natoms = N;
                ga.C = L0->C_msg;
                keep0_vform = vform0_ask(L0, 0, ga);
                if (keep0_vform_.size() > (1u << 20)) keep0_vform_.clear();  // forward sweeps without their backward
                keep0_vform_[in.keep0] = keep0_vform;
            } else {
                const auto it = keep0_vform_.find(in.keep0);
                if (it == keep0_vform_.end())
                    throw std::logic_error("generic: keep0's backward sweep before its forward sweep");
                keep0_vform = it->second;
                keep0_vform_.erase(it);
            }
            for (int l = 0; l < 4; ++l) {
                if (keep0_vform >> l & 1u) L0->gout[l] = in.keep0 + off[5 + l];  // lin_l is never written
                else L0->lin[l] = in.keep0 + off[5 + l];
            }
            if (!L0->res0_table) L0->res[0] = in.keep0 + off[9];
        }
    }
    if (sweep == GenericPassInputs::SweepForward) {
        forward_layer(sweep_t, in.energy);
        if (keep0 && gate_vform[0] != keep0_vform)  // a gated store declined after its ask: gout_l went to the arena
            throw std::logic_error("generic: keep0's gated stores differ from their asks");
        return;
    }
    if (sweep == GenericPassInputs::SweepLast) {
        forward_layer(n_layers_ - 1, in.energy);
        backward_layer(n_layers_ - 1);
        if (n_layers_ > 1) {
            assemble_edges(false);
            return;
        }
        // a single layer: its sweep is the whole pass, and the final assembly below applies
    } else if (sweep == GenericPassInputs::SweepBackward) {
        GenericLayer* L = layers_[sweep_t];
        const auto& hd = hand_[(size_t)sweep_t];  // list 20: the output's gradient, or the fold targets' (hand)
        if (hd.size() > 4) throw std::logic_error("generic: more than four hand-over arrays");
        for (size_t i = 0; i < hd.size(); ++i) {
            if (!in.adjoint_in[i]) throw std::logic_error("generic: a backward sweep needs adjoint_in");
            CUDA_CHECK_THROW(cudaMemcpyAsync(hd[i].g, in.adjoint_in[i], (size_t)N * hd[i].w * sizeof(float),
                                             cudaMemcpyDeviceToDevice, st));
        }
        if (keep0) {  // layer 0's forward results kept from the forward sweep: only the readout's seed onto the adjoint
            if (!fold_sweeps_) {
                add_row_vector_kernel<<<(unsigned)(((int64_t)N * L->C_out + 255) / 256), 256, 0, st>>>(
                    N, L->C_out, L->ro_w, L->dhout[0]);
            } else if (!fused_readout(0)) {  // list 20: the folded linear readout's seed, which its forward assigns
                for (const auto& ft : L->fold[0])
                    if (ft.dest_layer == -2) {
                        CUDA_CHECK_THROW(cudaMemsetAsync(L->dy[0], 0, (size_t)N * L->C_out * sizeof(float), st));
                        add_row_vector_kernel<<<(unsigned)(((int64_t)N * L->C_out + 255) / 256), 256, 0, st>>>(
                            N, L->C_out, ft.Wy, L->dy[0]);
                    }
            }
            gate_vform[0] = keep0_vform;
        } else {
            forward_layer(sweep_t, d_energy_sink_);  // adds the readout's seed onto the adjoint; energy already counted
        }
        backward_layer(sweep_t);
        if (sweep_t > 0) {
            assemble_edges(false);
            return;
        }
        // layer 0: the final assembly below applies (ZBL included)
    }

    // ---------------- the pass
    if (in.energy_only) {
        // The readouts of every layer, the long range and ZBL. Nothing is differentiated, except inside the two
        // calls that fuse their backward with the forward (the last layer's product + readout kernel, the long
        // range): their gradients are not used.
        for (int t = 0; t < n_layers_; ++t) forward_layer(t, in.energy);
        if (lr_) {
            std::vector<std::vector<float*>> hs, dhs;
            for (auto* L : layers_) {
                hs.emplace_back(L->hout, L->hout + L->lmax_out + 1);
                dhs.emplace_back(L->dhout, L->dhout + L->lmax_out + 1);
            }
            lr_->record(st, N, in.gemm, in.gemm_fork, in.gemm_lane, in.gemm_join, in.species, in.pos64, hs, dhs,
                        in.energy, in.forces, 0, in.virial, in.gemm_h2_row);
        }
        if (has_edges && has_zbl_)
            zbl_energy_kernel<<<(in.E_cap + 255) / 256, 256, 0, st>>>(in.E_dev, S, in.act_vec_len, in.act_senders,
                                                                      in.act_receivers, in.species, zbl_p, in.energy,
                                                                      in.atom_energy);
        return;
    }
    if (sweep != GenericPassInputs::SweepAll) {
        // A sweep that reaches this point (the only layer's, or layer 0's backward) has run above; what is left is
        // the final edge assembly at the end of this function.
    } else if (!sweep_replay_) {
        // Whole system: every layer forward, the long range, every layer backward, all activations kept.
        for (int t = 0; t < n_layers_; ++t) forward_layer(t, in.energy);
        // long range: energy, k-space forces and its gradient seeds into every layer's dhout
        if (lr_) {
            std::vector<std::vector<float*>> hs, dhs;
            for (auto* L : layers_) {
                hs.emplace_back(L->hout, L->hout + L->lmax_out + 1);
                dhs.emplace_back(L->dhout, L->dhout + L->lmax_out + 1);
            }
            lr_->record(st, N, in.gemm, in.gemm_fork, in.gemm_lane, in.gemm_join, in.species, in.pos64, hs, dhs,
                        in.energy, in.forces, in.compute_stress, in.virial, in.gemm_h2_row);
        }
        for (int t = n_layers_ - 1; t >= 0; --t) backward_layer(t);
    } else {
        // The sweep schedule of a domain with ghost atoms (PLAN.md 4.4), on the whole system. Between sweeps only
        // each layer's output (hout) and the gradient of it that later layers added (dhout) survive:
        //   forward sweeps:  layer t forward; its readout's own seed in dhout is dropped again.
        //   last layer:      forward and backward together.
        //   backward sweeps: everything but the stored outputs (and the zeros alloc_system wrote once, which no
        //                    sweep writes) is overwritten with NaN, layer t is replayed forward
        //                    (which adds the readout's seed onto dhout; its energy goes to the sink), then backward.
        for (int t = 0; t < n_layers_; ++t) {
            forward_layer(t, in.energy);
            if (t == n_layers_ - 1) break;
            const GenericLayer* L = layers_[t];
            for (int l = 0; l <= L->lmax_out; ++l)
                CUDA_CHECK_THROW(cudaMemsetAsync(L->dhout[l], 0, (size_t)N * dimL(l) * L->C_out * sizeof(float), st));
        }
        backward_layer(n_layers_ - 1);
        for (int t = n_layers_ - 2; t >= 0; --t) {
            // A sweep that differentiates leaves no edge gradients behind: it assembles its own edge forces (ZBL
            // comes with layer 0's sweep), and the next one starts from zero.
            assemble_edges(false);
            if (has_edges) zero_edge_kernel<<<(3 * in.E_cap + 255) / 256, 256, 0, st>>>(in.E_dev, d_edge_dvec_);
            for (size_t i = 0; i < sys_allocs_.size(); ++i) {
                bool keep = sys_allocs_[i] == d_h0_ ||
                            std::find(sys_zero_once_.begin(), sys_zero_once_.end(), sys_allocs_[i]) != sys_zero_once_.end();
                for (const GenericLayer* K : layers_)
                    for (const float* p : K->hout) keep = keep || (p != nullptr && p == sys_allocs_[i]);
                if (!keep) CUDA_CHECK_THROW(cudaMemsetAsync(sys_allocs_[i], 0xFF, sys_alloc_bytes_[i], st));
            }
            forward_layer(t, d_energy_sink_);
            backward_layer(t);
        }
    }
    // Layer 0's edge kernel, where it runs, has assembled the edge forces and ZBL itself.
    const bool edge_done = layers_[0]->first_hidden && layers_[0]->J - 1 <= 96;
    if (!edge_done) assemble_edges(true);
}

}  // namespace loup
