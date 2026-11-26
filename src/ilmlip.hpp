#pragma once
// Native CUDA inference engine for arch=ilmlip (ILMlip-Student, arXiv:2611.33538v1 adapted for
// Loup-S single-material distillation).
//
// Zero radial MLPs, zero per-edge global buffers: pair-FiLM Chebyshev radials are folded at export
// into compact per-pair tables C1_eff / C2_eff / C_bf, with receiver-centric warp-cooperative edge
// passes, channel-wise ACE nu<=3 products, pre-contracted exact order-1 back-flow removal, Layer-2
// Gram + low-rank bispectrum invariants, standardized shared SiLU readout MLP, and fused 0.65 R_cov
// C^2 ZBL core switch.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "generic.hpp"

namespace loup {

class Bundle;

class IlmlipModel {
public:
    IlmlipModel() = default;
    ~IlmlipModel();
    IlmlipModel(const IlmlipModel&) = delete;
    IlmlipModel& operator=(const IlmlipModel&) = delete;

    void load(const Bundle& b);
    void alloc_system(int N);
    void free_system();
    void record(const GenericPassInputs& in);

    int S() const { return S_; }
    float r_max() const { return r_max_; }
    const std::vector<int>& atomic_numbers() const { return Z_; }
    const std::vector<double>& e0() const { return e0_; }
    size_t system_bytes() const { return sys_bytes_; }
    size_t largest_buffer_bytes() const { return sys_max_alloc_; }
    std::string kernel_signature() const;

    static constexpr int kMaxL = 4;
    static constexpr int kMaxDepth = 4;

private:
    void free_model();

    // Model dimensions & flags
    int S_ = 0;
    float r_max_ = 0.0f;
    std::vector<int> Z_;
    std::vector<double> e0_;

    int lmax_ = 2, lfeat_ = 1, lmsg_ = 1, lfeat2_ = 1, lout2_ = 2;
    int n_l_[kMaxL] = {0, 0, 0, 0};
    int l1_off_[kMaxL] = {0, 0, 0, 0};
    int n_tot1_ = 0, max_nl_ = 0;
    int n_base_ = 0, n_msg_ = 0, r_gram_ = 0, r_bisp_ = 0;
    int ro_depth_ = 0, ro_width_ = 0;
    int remove_backflow_ = 1, use_zbl_switch_ = 1;
    bool fast_l1_node_lf1_ = false;
    bool fast_bisp_l2_ = false;

    int n_nu2_ = 0, n_nu3_ = 0, n_l2_paths_ = 0, n_bisp_ = 0;
    int cg_count_ = 0, bisp_idx_count_ = 0, r_gb_ = 0;
    int n_inv1_ = 0, n_inv2_ = 0, n_feat_ = 0, n_feat_pad_ = 0;
    int D_A1_ = 0, D_A2_ = 0;
    int b2_scalars_ = 0, total_l2_out_ = 0;
    int msg_in_dim_[kMaxL] = {0, 0, 0, 0};
    int l2_npaths_[kMaxL] = {0, 0, 0, 0};
    int l2_c_[kMaxL] = {0, 0, 0, 0};
    int gram_off_[kMaxL] = {0, 0, 0, 0};
    int bisp_off_ = 0;

    float sigma0_ = 1.0f, sigma1_ = 1.0f, a0_ = 0.05f, kappa_ = 16.0f, z_c_ = 10.0f;
    float max_rs_ = 0.0f;

    // Device weights & metadata (owned by model_allocs_)
    std::vector<void*> model_allocs_;
    float* d_C1_ = nullptr;       // [S*S, n_base, n_tot1]
    float* d_C1_zick_ = nullptr;  // [S, n_tot1, S*n_base] (coalesced over (zj,k) for zero-shuffle L1 bwd!)
    float* d_C2_ = nullptr;       // [S*S, n_base, (lfeat2+1)*n_msg]
    float* d_Cbf_ = nullptr;      // [S*S, n_base, (lmsg+1)*n_msg]
    double* d_rs_pair_ = nullptr; // [S*S]
    float* d_rs_pair_f32_ = nullptr; // [S*S] (rs_pair + 1e-3f for fast FP32 ZBL trigger check)
    double* d_zz_pair_ = nullptr; // [S*S]
    double* d_au_pair_ = nullptr; // [S*S]

    int32_t* d_nu2_ = nullptr;    // [n_nu2, 9]: l1, l2, L, c, cg_off, cg_cnt, b2_off, x1_off, bcat_off
    int32_t* d_nu3_ = nullptr;    // [n_nu3, 8]: p2_idx, l3, L, c, cg_off, cg_cnt, x1_off, bcat_off
    int32_t* d_l2p_ = nullptr;    // [n_l2_paths, 7]: l1, l2, L, p_in_L, cg_off, cg_cnt, reg_off
    int32_t* d_cg_idx_ = nullptr;
    float* d_cg_val_ = nullptr;
    int32_t* d_bisp_ = nullptr;   // [n_bisp, 5]: L1, L2, L3, w3_off, w3_cnt
    int32_t* d_bisp_idx_ = nullptr;
    float* d_bisp_val_ = nullptr;
    int32_t* d_triu_i_ = nullptr;
    int32_t* d_triu_j_ = nullptr;

    float* d_W_msg_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};    // [msg_in_dim[l], n_msg]
    float* d_W_msg_T_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};  // [n_msg, msg_in_dim[l]] (coalesced for fused L1 node bwd)
    float* d_W_GB_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};   // L>=1: [l2_c[L], r_gram + r_bisp]; L==0: d_W_B_[0]
    float* d_W_B0_ = nullptr;                                       // [l2_c[0], r_bisp] (if r_bisp > 0)
    float* d_mu_x_ = nullptr;       // [n_feat]
    float* d_inv_std_x_ = nullptr;  // [n_feat]
    float* d_ro_W_[kMaxDepth] = {nullptr, nullptr, nullptr, nullptr};
    float* d_ro_b_[kMaxDepth] = {nullptr, nullptr, nullptr, nullptr};
    void*  d_ro_h2_B_[kMaxDepth]  = {nullptr, nullptr, nullptr, nullptr};
    float* d_ro_h2_pB_[kMaxDepth] = {nullptr, nullptr, nullptr, nullptr};
    void*  d_ro_h2_BT_[kMaxDepth]  = {nullptr, nullptr, nullptr, nullptr};
    float* d_ro_h2_pBT_[kMaxDepth] = {nullptr, nullptr, nullptr, nullptr};
    bool   use_ro_h2_ = false;
    float* d_alpha_Z_ = nullptr;    // [S, ro_width] (includes ro_b_0)
    float* d_u_Z_ = nullptr;        // [S, ro_width] (pre-scaled by energy_scale)

    // Per-system buffers (owned by sys_allocs_ + d_zero_slab_)
    int N_ = 0;
    size_t sys_bytes_ = 0, sys_max_alloc_ = 0;
    std::vector<void*> sys_allocs_;
    void* d_zero_slab_ = nullptr;
    size_t zero_slab_bytes_ = 0;

    float* d_A1_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};
    float* d_dA1_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};
    float* d_f0_ = nullptr;
    double* d_e_zbl_ = nullptr;
    float* d_w_zbl_ = nullptr;
    int* d_zbl_flag_ = nullptr;  // [1] device flag set when any edge has r < max_rs_
    float* d_df0_bf_ = nullptr;  // in zero slab

    float* d_Bcat_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};
    float* d_dBcat_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};
    float* d_I_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};
    float4* d_I_pack4_ = nullptr;  // [N, n_msg] packed (I0, I1_0, I1_1, I1_2) -> 1x LDG.128 in L2 fwd!
    float* d_dI_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};  // in zero slab

    float* d_A2_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};
    float* d_dA2_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};
    float* d_dA2_pack_ = nullptr;  // [N, 14, n_msg] pre-packed (g0..g6, g7c, w00..w22) for sender-owned L2 bwd
    float* d_f1_ = nullptr;

    float* d_PQ_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};   // L>=1: [N*(2L+1), r_gb] (P and Q interleaved)
    float* d_dPQ_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};
    float* d_Q0_ = nullptr;                                        // L==0: [N, r_bisp]
    float* d_dQ0_ = nullptr;

    float* d_x_ = nullptr;   // [N, n_feat]
    float* d_dx_ = nullptr;  // [N, n_feat]
    int8_t* d_eA_x_ = nullptr;  // [N] cached row exponent of d_x_ (eliminates h2r_row_exp_kernel!)
    float* d_pre_[kMaxDepth] = {nullptr, nullptr, nullptr, nullptr};
    float* d_h_[kMaxDepth] = {nullptr, nullptr, nullptr, nullptr};
    float* d_dh_[2] = {nullptr, nullptr};  // ping-pong [N, ro_width]
};

}  // namespace loup
