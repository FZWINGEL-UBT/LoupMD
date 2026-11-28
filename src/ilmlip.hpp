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
    float r_max2() const { return r_max2_; }
    const std::vector<int>& atomic_numbers() const { return Z_; }
    const std::vector<double>& e0() const { return e0_; }
    size_t system_bytes() const { return sys_bytes_; }
    size_t largest_buffer_bytes() const { return sys_max_alloc_; }
    std::string kernel_signature() const;
    float* forces_f32() const { return d_forces_f32_; }
    bool supports_pack_vs() const {
        const bool nb1_ok = (n_base_ == 8 || n_base_ == 12 || n_base_ == 14 || n_base_ == 16);
        const bool l1_ok = (S_ <= 4 && nb1_ok &&
                            n_l_[0] <= 64 && n_l_[1] <= 32 && n_l_[2] <= 32 && n_l_[3] <= 32 &&
                            (lmax_ == 2 || lmax_ == 3) &&
                            S_ * n_base_ <= 64 && D_A1_ <= ((lmax_ == 2) ? 256 : 416));
        return has_lrsk_ && lrsk_fast_sym_ && l1_ok;
    }

    static constexpr int kMaxL = 4;
    static constexpr int kMaxDepth = 4;

private:
    void free_model();

    // Model dimensions & flags
    int S_ = 0;
    float r_max_ = 0.0f;
    float r_max2_ = 0.0f;
    float radial_warp_ = 0.0f;
    std::vector<int> Z_;
    std::vector<double> e0_;

    int lmax_ = 2, lfeat_ = 1, lmsg_ = 1, lfeat2_ = 1, lout2_ = 2;
    int n_l_[kMaxL] = {0, 0, 0, 0};
    int l1_off_[kMaxL] = {0, 0, 0, 0};
    int n_tot1_ = 0, max_nl_ = 0;
    int n_base_ = 0, n_base2_ = 0, n_msg_ = 0, r_gram_ = 0, r_bisp_ = 0;
    int ro_depth_ = 0, ro_width_ = 0;
    int remove_backflow_ = 1, use_zbl_switch_ = 1;
    bool has_l2_ = true;
    bool has_lrsk_ = false;
    int lrsk_sym_ = 0;
    bool lrsk_fast_sym_ = false;
    int lrsk_c0_ = 0, lrsk_c1_ = 0, lrsk_c2_ = 0, lrsk_i_stride_ = 0;
    int lrsk_pol_field_ = 0, lrsk_pol_width_ = 0;
    int lrsk_pol_gram_ = 0, lrsk_sw_mp_ = 0, lrsk_bo_width_ = 0;
    float lrsk_mp_cap_ = 0.0f, lrsk_ef_cap_ = 0.0f;
    int lrsk_pol_in_pad_ = 0, lrsk_pol_w_pad_ = 0;
    bool lrsk_pol_h2_ = false;
    int lrsk_n_terms_ = 0, lrsk_n_terms_pad_ = 0, lrsk_n_head_out_ = 0;
    float energy_scale_ = 1.0f;
    bool fast_l1_node_lf1_ = false;
    bool fast_l1_node_lf1_s0_ = false;
    bool fast_l1_node_1L_lf1_ = false;
    bool fast_l1_node_lf2_ = false;
    bool fast_l1_node_1L_lf2_ = false;
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
    void*  d_W_msg_h2_B_[kMaxL]  = {nullptr, nullptr, nullptr, nullptr};
    float* d_W_msg_h2_pB_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};
    void*  d_W_msg_h2_BT_[kMaxL]  = {nullptr, nullptr, nullptr, nullptr};
    float* d_W_msg_h2_pBT_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};
    float* d_W_GB_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};   // L>=1: [l2_c[L], r_gram + r_bisp]; L==0: d_W_B_[0]
    void*  d_W_GB_h2_B_[kMaxL]  = {nullptr, nullptr, nullptr, nullptr};
    float* d_W_GB_h2_pB_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};
    void*  d_W_GB_h2_BT_[kMaxL]  = {nullptr, nullptr, nullptr, nullptr};
    float* d_W_GB_h2_pBT_[kMaxL] = {nullptr, nullptr, nullptr, nullptr};
    float* d_W_B0_ = nullptr;                                       // [l2_c[0], r_bisp] (if r_bisp > 0)
    void*  d_W_B0_h2_B_  = nullptr;
    float* d_W_B0_h2_pB_ = nullptr;
    void*  d_W_B0_h2_BT_  = nullptr;
    float* d_W_B0_h2_pBT_ = nullptr;
    bool   use_node_h2_ = false;
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
    // LR-SK model weights
    float*  d_lrsk_W_p1_ = nullptr;        // [msg_in_dim[1], (sym ? c1 : 2*c1)]
    float*  d_lrsk_W_p2_ = nullptr;        // [msg_in_dim[2], (sym ? c2 : 2*c2)]
    float*  d_lrsk_W_p1_T_ = nullptr;      // [i_stride, msg_in_dim[1]] (for fused L1 node!)
    float*  d_lrsk_W_p2_T_ = nullptr;      // [i_stride, msg_in_dim[2]] (for fused L1 node!)
    float*  d_lrsk_W0_zero_ = nullptr;     // [i_stride, msg_in_dim[0]] (zero dummy for fused lf2_msg)
    float*  d_lrsk_W_head_ = nullptr;      // [S, ro_width, n_head_out]
    float*  d_lrsk_W_head_T_ = nullptr;    // [S, n_head_out, ro_width]
    float*  d_lrsk_b_head_ = nullptr;      // [S, n_head_out]
    float*  d_lrsk_W_head_p16_ = nullptr;  // [S, ro_width, 16] (SMEM-staged fast sym heads)
    float*  d_lrsk_b_head_p16_ = nullptr;  // [S, 16]
    float4* d_lrsk_W_rad4_ = nullptr;      // [S*S, NB2_4, n_terms_pad]
    float4* d_lrsk_W_fast4_ = nullptr;     // [S*S, NB2_4, N_T4, 4] (4-lane fast sym edge!)
    float*  d_lrsk_bo_Wx_fwd_ = nullptr;   // [28, H_pad]
    float4* d_lrsk_bo_Wx_bwd4_ = nullptr;  // [7, H4, 4]
    float4* d_lrsk_bo_Wr4_ = nullptr;      // [S*S, NB2_4, H_pad]
    float*  d_lrsk_bo_bv_ = nullptr;       // [S*S, 2, H_pad]
    float*  d_lrsk_pol_W0_ = nullptr;      // [f_dim + ro_width, pol_width]
    float*  d_lrsk_pol_W0_pad_ = nullptr;  // [pol_in_pad, pol_w_pad]
    uint4*  d_lrsk_pol_cHL_ = nullptr;     // [K_TILES * 4 * 32] pre-split coalesced fwd panels
    uint4*  d_lrsk_pol_rHL_ = nullptr;     // [K_TILES * 4 * 32] pre-split coalesced bwd panels
    float   lrsk_pol_inv_sW_ = 0.0f;       // host-cached inv_sW for pre-split pol_W0
    float*  d_lrsk_pol_b0_ = nullptr;      // [pol_width]
    float*  d_lrsk_pol_u_ = nullptr;       // [S, pol_width] (pre-scaled by energy_scale)
    void*   d_lrsk_pol_h2_B_ = nullptr;
    float*  d_lrsk_pol_h2_pB_ = nullptr;
    void*   d_lrsk_pol_h2_BT_ = nullptr;
    float*  d_lrsk_pol_h2_pBT_ = nullptr;

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
    float* d_am_l1_ = nullptr;  // [N] max|x_L1| per atom for fused L1 standardization
    int8_t* d_eA_x_ = nullptr;  // [N] cached row exponent of d_x_ (eliminates h2r_row_exp_kernel!)
    float* d_pre_[kMaxDepth] = {nullptr, nullptr, nullptr, nullptr};
    float* d_h_[kMaxDepth] = {nullptr, nullptr, nullptr, nullptr};
    float* d_dh_[2] = {nullptr, nullptr};  // ping-pong [N, ro_width]
    float* d_forces_f32_ = nullptr;        // [N, 3] L2-resident FP32 ML force accumulator (reset on flush)
    // LR-SK per-system buffers
    float* d_lrsk_I1_ = nullptr;
    float* d_lrsk_I2_ = nullptr;
    float* d_lrsk_dI1_ = nullptr;
    float* d_lrsk_dI2_ = nullptr;
    float* d_lrsk_qA_ = nullptr;
    float* d_lrsk_qB_ = nullptr;
    float* d_lrsk_pA_ = nullptr;
    float* d_lrsk_pB_ = nullptr;
    float* d_lrsk_QA_ = nullptr;
    float* d_lrsk_QB_ = nullptr;
    float* d_lrsk_g1A_ = nullptr;
    float* d_lrsk_g1B_ = nullptr;
    float* d_lrsk_g2A_ = nullptr;
    float* d_lrsk_g2B_ = nullptr;
    float* d_lrsk_q_raw_ = nullptr;
    float* d_lrsk_es_ = nullptr;
    float* d_lrsk_elr_ = nullptr;
    float* d_lrsk_sw_ = nullptr;
    float* d_lrsk_dqA_ = nullptr;
    float* d_lrsk_dqB_ = nullptr;
    float* d_lrsk_dpA_ = nullptr;
    float* d_lrsk_dpB_ = nullptr;
    float* d_lrsk_dQA_ = nullptr;
    float* d_lrsk_dQB_ = nullptr;
    float* d_lrsk_Phi_ = nullptr;
    float* d_lrsk_Ef_ = nullptr;
    float* d_lrsk_dPhi_ = nullptr;
    float* d_lrsk_dEf_ = nullptr;
    // LR-SK-Fast symmetric buffers
    float*  d_lrsk_I0_dum_ = nullptr;
    float*  d_lrsk_dI0_dum_ = nullptr;
    float4* d_lrsk_qp4_ = nullptr;
    float4* d_lrsk_Q03_ = nullptr;
    float*  d_lrsk_Q4_ = nullptr;
    float4* d_lrsk_dqp4_ = nullptr;
    float4* d_lrsk_dQ03_ = nullptr;
    float*  d_lrsk_dQ4_ = nullptr;
    float4* d_lrsk_pol_ef4_ = nullptr;
    float4* d_lrsk_dpol_ef4_ = nullptr;
    float4* d_lrsk_qp_dpol8_ = nullptr;
    float*  d_lrsk_pol_in_ = nullptr;
    float*  d_lrsk_pol_pre_ = nullptr;
    float*  d_lrsk_pol_din_ = nullptr;
    int8_t* d_lrsk_eA_pol_ = nullptr;

    // Dedicated side stream + events for zero-latency graph-parallel ZBL fwd/bwd overlap
    cudaStream_t zbl_stream_ = nullptr;
    cudaEvent_t zbl_ev_fork_ = nullptr;
    cudaEvent_t zbl_ev_join_ = nullptr;

    // Built-in D3(BJ) dispersion: mode 0=off, 1=Option B (d3bj_fixed), 2=Option C (d3bj_cn), 3=Option D (d3bj_far)
    int disp_mode_ = 0;
    bool disp_use_tail_ = false;
    float disp_rcut_ = 0.0f;
    float disp_s6_ = 1.0f, disp_s8_ = 0.0f;
    float* d_disp_pair_ = nullptr;        // mode 1/3: [S*S, 6] float32: c6, c8, f6, f8, e_shift, f_shift
    // Mode 2 (Option C: CN-dependent C6) model tables
    int* d_disp_cn_nref_ = nullptr;       // [S]
    float* d_disp_cn_cnref_ = nullptr;    // [S, 5]
    float* d_disp_cn_rcov_ = nullptr;     // [S]
    float* d_disp_cn_pair_ = nullptr;     // [S*S, 5]: qq_A2, f6, f8, g_cut, dg_cut
    float* d_disp_cn_c6ref_ = nullptr;    // [S*S, 25]
    // Mode 2 per-system scratch (owned by sys_allocs_)
    float* d_disp_w_ = nullptr;           // [N, 5]
    float* d_disp_dw_ = nullptr;          // [N, 5]
    float* d_disp_dedcn_ = nullptr;       // [N]
    // Mode 3 (Option D: explicit far-pair spatial cell-bin scratch, owned by model_allocs_ / sys_allocs_)
    void* d_disp_far_grid_ = nullptr;     // DispFarGrid struct on device
    int* d_disp_far_cnt_ = nullptr;       // [8192]
    int* d_disp_far_ptr_ = nullptr;       // [8193]
    int* d_disp_far_cur_ = nullptr;       // [8192]
    float4* d_disp_far_pos4_ = nullptr;   // [N] (x, y, z, int_as_float(sp)) sorted by bin
    int* d_disp_far_idx_ = nullptr;       // [N] original atom index sorted by bin (mode 4)
    int* d_disp_far_bin_ = nullptr;       // [N] bin index per atom
    // Analytical K/V tail
    std::vector<double> disp_J_st_;       // [S*S] eV*A^3 (host, for K_tail = 0.5 * N^T J N)
    double* d_disp_tail_per_sp_ = nullptr; // [S] 0.5 * (J @ N) eV*A^3 per species
    std::vector<int> disp_cached_counts_;
    double disp_K_tail_ = 0.0;
    cudaStream_t disp_stream_ = nullptr;
    cudaEvent_t disp_ev_fork_ = nullptr;
    cudaEvent_t disp_ev_join_ = nullptr;
};

}  // namespace loup
