#pragma once
// Unfolded pure-CUDA foundation MACE fine-tuning types and parameter descriptors.
// Supports all 18 LoupMD foundation models (generic arch kinds 0..4 + optional PolarLR + ZBL).
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "kernels_common.cuh"

namespace loup_ft {

struct ParamTensor {
    std::string bundle_key;  // exact .loup entry name (empty for derived/special params like radial delta)
    int64_t numel = 0;
    float* d_val = nullptr;
    float* d_grad = nullptr;
    float* d_m = nullptr;
    float* d_v = nullptr;
    float* d_ema = nullptr;
    float* d_anchor = nullptr;
    bool is_head = false;
    bool trainable = false;
    float lr_scale = 1.0f;  // compensates for .loup folded 1/(avg_num_neighbors*sqrt(C)) scale
};

struct LayerFT {
    int t = 0;
    int kind = 0;  // 0 agnostic, 1 agnostic_res, 2 density, 3 density_res, 4 nonlinear
    int lmax_in = 0, lmax_x = 0, lmax_out = 0;
    int C_in = 0, C_x = 0, C_msg = 0, C_out = 0;
    int n_paths = 0, n_l3[4] = {0, 0, 0, 0};
    int lin_cols[4] = {0, 0, 0, 0};
    int intervals = 0, Q1 = 0, pair_sym = 0, Ptab = 0;
    int first_hidden = 0, J = 0, Jp = 0;  // Jp = J - 1 (raw bundle first_C0 rows = S * Jp)
    int has_sc = 0, sc_species = 0, sc_has[3] = {0, 0, 0};
    int prod_species = 0, prod_segw = 0;
    int readout = -1, ro_H = 0;
    float alpha = 1.0f, beta = 0.0f, gate_cs = 1.0f, gate_cg = 1.0f, ro_act_c = 1.0f;

    SplineGrid grid{};
    std::string prod_sig;
    std::string prod_src;  // raw CUDA source stored in bundle (empty if built-in only)

    // Frozen/base FP32 spline coefficients on GPU: [Ptab * intervals * Q1] float4 (a0, a1, a2, a3)
    float4* d_spline = nullptr;
    // Separate density spline on GPU if kind >= 2: [Ptab * intervals] float4 (last column of spline)
    float4* d_density_spline = nullptr;

    // Parameter indices into ModelFT::params (-1 if absent)
    int p_W_up[3] = {-1, -1, -1};
    int p_first_C0[4] = {-1, -1, -1, -1};
    int p_W_lin[4] = {-1, -1, -1, -1};
    int p_W_skip_msg[4] = {-1, -1, -1, -1};  // kind 0 or 2: [S, C_msg, C_msg]
    int p_W_res[3] = {-1, -1, -1};           // kind 4: [C_x, lin_cols[l]]
    int p_W_2[4] = {-1, -1, -1, -1};         // kind 4: [C_msg, C_msg]
    int p_prod_w = -1;                       // [Sp, prod_segw, C_msg]
    int p_W_p[3] = {-1, -1, -1};             // [C_out, C_out]
    int p_W_sc[3] = {-1, -1, -1};            // [Ss, C_in, C_out]
    int p_ro_w = -1;                         // readout == 0: [C_out]
    int p_ro_W1 = -1;                        // readout >= 1: [C_out, ro_H]
    int p_ro_b1 = -1;                        // readout >= 1: [ro_H]
    int p_ro_Wm = -1;                        // readout == 2: [ro_H, ro_H]
    int p_ro_bm = -1;                        // readout == 2: [ro_H]
    int p_ro_W2 = -1;                        // readout >= 1: [ro_H]
};

struct ZblFT {
    int enabled = 0;
    float c[4] = {0, 0, 0, 0};
    float p = 0.0f;
    float rmax = 0.0f;
    float* d_rc = nullptr;     // [S*S]
    float* d_inv_a = nullptr;  // [S*S]
    float* d_pref = nullptr;   // [S*S]
};

// Forward/backward activation workspace for a concatenated batch of N_tot atoms and E_tot edges.
struct LayerActFT {
    // Forward intermediates (all FP32, kept alive for 1st and 2nd backward)
    float* x[3] = {};        // [N, 2l+1, C_x]
    float* S0[4] = {};       // [N, 2l+1, S*Jp] (layer 0 first_hidden)
    float* mid[4] = {};      // [N, 2l+1, n_l3[l]*C_x]
    float* D = nullptr;      // [N]
    float* lin[4] = {};      // [N, 2l+1, lin_cols[l]]
    float* res[3] = {};      // [N, 2l+1, lin_cols[l]] (kind 4)
    float* gout[4] = {};     // [N, 2l+1, C_msg] (kind 4)
    float* msgf[4] = {};     // [N, 2l+1, C_msg] (input to product)
    float* y[3] = {};        // [N, 2l+1, C_out] (output of product)
    float* hout[3] = {};     // [N, 2l+1, C_out] (W_p + skip)
    float* hid = nullptr;    // [N, ro_H] (pre-act1 or act1)
    float* pre1 = nullptr;   // [N, ro_H] (readout >= 1: h0 @ W1 + b1)
    float* act1 = nullptr;   // [N, ro_H] (readout >= 1: c * silu(pre1))
    float* pre2 = nullptr;   // [N, ro_H] (readout == 2: act1 @ Wm + bm)
    float* act2 = nullptr;   // [N, ro_H] (readout == 2: c * silu(pre2))

    // 1st-order backward (dE/d*) — kept alive for 2nd-order backward (force loss)!
    float* dhout[3] = {};
    float* dy[3] = {};
    float* dmsgf[4] = {};
    float* dgout[4] = {};
    float* dlin[4] = {};
    float* dpre[3] = {};     // kind 4: dE/d(pre_l) = dE/d(res_l)
    float* dD = nullptr;
    float* dmid[4] = {};
    float* dS0[4] = {};
    float* dx[3] = {};
    float* dhid1 = nullptr;  // dE/d(pre1)
    float* dhid2 = nullptr;  // dE/d(pre2) (readout == 2)

    // 2nd-order backward cotangents (dL_F / d(1st-bwd) = tangent of forward activations,
    // and dL_total / d(fwd) = adjoint of forward activations).
    // Notation:
    //   bar_*  = dL_total / d(*)           (adjoint of forward tensor *)
    //   tan_*  = dL_F     / d(d*)          (adjoint of 1st-bwd gradient d*, propagates forward-like!)
    float* bar_hout[3] = {};
    float* bar_y[3] = {};
    float* bar_msgf[4] = {};
    float* bar_gout[4] = {};
    float* bar_lin[4] = {};
    float* bar_res[3] = {};
    float* bar_D = nullptr;
    float* bar_mid[4] = {};
    float* bar_S0[4] = {};
    float* bar_x[3] = {};

    float* tan_x[3] = {};
    float* tan_S0[4] = {};
    float* tan_mid[4] = {};
    float* tan_D = nullptr;
    float* tan_lin[4] = {};
    float* tan_pre[3] = {};
    float* tan_gout[4] = {};
    float* tan_msgf[4] = {};
    float* tan_y[3] = {};
    float* tan_hout[3] = {};
    float* tan_dhid1 = nullptr;
    float* tan_dhid2 = nullptr;
};

struct ModelFT {
    std::string base_bundle_path;
    int S = 0;
    int n_layers = 0;
    int sh_perm = 0;
    int C0 = 0;
    int has_lr = 0;
    float r_max = 0.0f;
    std::vector<int> Z;
    std::vector<double> e0;
    double* d_e0 = nullptr;

    int p_embedding = -1;  // [S, C0]
    std::vector<LayerFT> layers;
    ZblFT zbl;

    std::vector<ParamTensor> params;
    std::vector<void*> aux_allocs;  // device memory freed in destroy()
};

inline int dimL(int l) { return 2 * l + 1; }

}  // namespace loup_ft
