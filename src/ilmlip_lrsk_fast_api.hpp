#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace loup {
namespace lrsk_fast {

struct ChebParamF {
    float r_max;
    float q0;
    float q1;
    float warp;
};

struct Cfg {
    int c0, c1, c2, c01, c02, c12;
    int n_head_out;
    int pol_width;
    int pol_gram;       // 0: diagonal (|E|^2, p.E); 1: full Gram (E_a.E_b, p_a.E_b)
    int sw_mp;          // 0: raw multipoles; 1: multiply q, p, Q (and Phi, Ef if ef_cap>0) by ZBL switch sw_i
    int bo_width;       // 0: bilinear only; >0 (<=32): BO-LRSK v4 pair-symmetric Bond-Order MLP width
    float half_esc;
    float inv_zc;
    float inv_mp_cap2;  // 0: uncapped; >0: 1/(mp_cap^2) smooth C^inf O(3) soft-cap
    float inv_ef_cap2;  // 0: uncapped; >0: 1/(ef_cap^2) smooth C^inf O(3) soft-cap on Phi & Ef
};

struct BoWeights {
    const float*  Wx_fwd;   // [28, H_pad]: ((ch*7 + m) * H_pad + h)
    const float4* Wx_bwd4;  // [7, H4, 4_ch]: float4 over 4 consecutive h per (m, h4, ch)
    const float4* Wr4;      // [S*S, NB2_4, H_pad]: float4 radial weights per (pair, blk, h)
    const float*  bv;       // [S*S, 2, H_pad]: bias (0) and output weights v (1) per (pair, h)
};

struct Bufs {
    float4* qp_pack4;     // [N, 4]: (q, px, py, pz)
    float4* Q_pack03;     // [N, 4]: (Q0, Q1, Q2, Q3) (nullptr if c2==0)
    float*  Q_pack4;      // [N, 4]: Q4 (nullptr if c2==0)
    float*  g1;           // [N, 4]
    float*  g2;           // [N, 4]
    float*  q_raw;        // [N, 4] (pre-cap/pre-sw q for exact sw_mp / mp_cap backward)
    float*  e_site;       // [N]
    float*  e_lr;         // [N]
    float*  sw;           // [N]
    float4* dqp_pack4;    // [N, 4]: (dq, dpx, dpy, dpz)
    float4* dQ_pack03;    // [N, 4]
    float*  dQ_pack4;     // [N, 4]
    float4* pol_ef4;      // [N, 4]: (Phi, Efx, Efy, Efz)
    float4* dpol_ef4;     // [N, 4]: (dPhi, dEfx, dEfy, dEfz)
    float4* qp_dpol_pack8;// [N, 8]: interleaved [qp[0..3], dpol_ef[0..3]] in 1x 128B cache line/atom
    const int* zbl_flag;  // optional device flag (*zbl_flag == 0 => all e_zbl==0, sw==1!)
};

// Zero-s_bc0/s_bc1 16-lane/32-lane L1 Node Fwd & Bwd specialized for LR-SK-Fast (lfeat==1, c1<=4, c2==0)
// Writes standardized x_out + eA_x directly (eliminates lrsk_am_to_eA_kernel launch!)
void launch_l1_node_fwd_lf1_lrsk(
    cudaStream_t st, int lmax, bool nl16,
    int N, int n_feat_pad, int n_msg,
    int nl0, int nl1, int nl2, int nl3, int m1d,
    const float* A0, const float* A1, const float* A2, const float* A3,
    const float* W1T, const float* mu_x, const float* inv_std_x,
    float* x_out, int8_t* eA_x_out, float* I1_out);

void launch_l1_node_bwd_lf1_lrsk(
    cudaStream_t st, int lmax, bool nl16,
    int N, int n_feat_pad, int n_msg,
    int nl0, int nl1, int nl2, int nl3, int m1d,
    const float* A0, const float* A1, const float* A2, const float* A3,
    const float* inv_std_x, const float* dx, const float* dI1, const float* W1T,
    float* dA0, float* dA1, float* dA2, float* dA3);

// Fused readout bias + SiLU + optional hgemm2 row exponent (eA_out)
void launch_bias_silu_exp(
    cudaStream_t st, int N, int W, bool per_species,
    const int* species, const float* bias, float* pre, float* h, int8_t* eA_out);

// Fused readout SiLU backward + optional hgemm2 row exponent (eA_out)
void launch_silu_bwd_exp(
    cudaStream_t st, int N, int W,
    const float* pre, const float* dh, float* dpre, int8_t* eA_out);

// Grid-stride SMEM-resident Heads Forward + FUSED last-layer bias + SiLU + optional sw_mp/mp_cap (read-only pre_last!)
void launch_heads_fwd_silu(
    cudaStream_t st, int N, int S, int ro_width, int i_stride,
    bool zbl_sw, bool has_c2, bool per_species_bias, Cfg cfg,
    const int* species, const float* bias_last,
    const float* pre_last, float* h_last,
    const float* W_head_p16, const float* b_head_p16,
    const float* I_raw1, const float* I_raw2,
    const double* e_zbl, Bufs buf);

// Grid-stride SMEM-resident Heads Backward + FUSED last-layer bias+SiLU bwd (float4 vectorized!) + sw_mp/mp_cap + eA_dpre
void launch_heads_bwd_silu(
    cudaStream_t st, int N, int S, int ro_width, int i_stride,
    bool pol_field, bool has_c2, bool zbl_sw, bool per_species_bias, Cfg cfg,
    const int* species, const float* bias_last, const float* W_head_p16,
    const float* I_raw1, const float* I_raw2,
    const float* pre_last, const float* dh_pol_in,
    const double* e_zbl, float* w_zbl_io,
    Bufs buf, float* dI_raw1, float* dI_raw2,
    float* dpre_out, int8_t* eA_dpre);

// 4-lane (32 atoms/CTA) symmetric edge kernels (NB2 <= 12, optional BO-LRSK v4 pair MLP)
void launch_edge_sym_pol_fwd(
    cudaStream_t st, int nb2, bool has_c2, bool skip_bo,
    int N, int S, Cfg cfg, ChebParamF r_max2,
    const int* row_ptr, const int* row_end,
    const int* senders, const float4* vec_len,
    const int* species, const float4* W_fast4, BoWeights bo, Bufs buf);

void launch_edge_sym_1p(
    cudaStream_t st, int nb2, bool has_c2, bool stress,
    int N, int S, Cfg cfg, ChebParamF r_max2,
    const int* row_ptr, const int* row_end,
    const int* senders, const float4* vec_len,
    const int* species, const float4* W_fast4, BoWeights bo, Bufs buf,
    double* forces, float* forces_f32, double* virial);

void launch_edge_sym_pol_bwd(
    cudaStream_t st, int nb2, bool has_c2, bool stress,
    int N, int S, Cfg cfg, ChebParamF r_max2,
    const int* row_ptr, const int* row_end,
    const int* senders, const float4* vec_len,
    const int* species, const float4* W_fast4, BoWeights bo, Bufs buf,
    const double* e_zbl, double* energy_out, double* atom_energy_out, float* w_zbl_io,
    double* forces, float* forces_f32, double* virial);

// One-time load-phase pre-split of pol_W0 into L1-resident coalesced uint4 panels (pol_cHL, pol_rHL)
void launch_pack_pol_w0_hl(
    cudaStream_t st, bool pol_gram, int in_dim, int pw, int pol_w_pad,
    const float* pol_W0_pad, uint4* pol_cHL, uint4* pol_rHL, float* inv_sW_dev);

// Fused FP16x2 Tensor-Core (mma.sync.m16n8k16) Pol-MLP + Energy + ZBL kernel (15.6 KB SMEM -> 6 CTAs/SM!)
// Supports pol_width <= 32, S <= 8, ro_width <= 64, BOTH pol_gram=0 (in_dim<=80) and pol_gram=1 (in_dim<=96)!
void launch_pol_mlp_fused(
    cudaStream_t st, bool zbl_sw, bool need_grad, bool pol_gram,
    int N, int S, int ro_width, float inv_sW, Cfg cfg,
    const int* species, const float* h_last,
    const uint4* pol_cHL, const uint4* pol_rHL, const float* pol_b0, const float* pol_u,
    const double* e_zbl, Bufs buf,
    double* energy_out, double* atom_energy_out,
    float* w_zbl_out, float* dh_pol_out);

// Fallback split Pol-MLP helpers (when pol_width > 32 or ro_width > 64)
void launch_pol_pack_in_sym(
    cudaStream_t st, int N, int ro_width, int in_pad, Cfg cfg,
    const float* h, Bufs buf, float* pol_in, int8_t* eA_pol);

void launch_pol_act_energy_zbl(
    cudaStream_t st, bool zbl_sw, bool need_grad,
    int N, int pw, int pw_pad, Cfg cfg,
    const int* species, const float* pol_b0, const float* pol_u,
    const float* e_site, const float* e_lr, const float* sw_arr,
    const double* e_zbl, float* pre_io, int8_t* eA_dpre,
    double* energy_out, double* atom_energy_out, float* w_zbl_out);

void launch_pol_unpack_grad_sym(
    cudaStream_t st, bool zbl_sw, int N, int ro_width, int in_pad, Cfg cfg,
    const float* dpol_in, const double* e_zbl, float* w_zbl_io, Bufs buf, float* dh_pol_out);

// Energy + ZBL switch for pure (pol=False) mode
void launch_energy_zbl_pure(
    cudaStream_t st, bool zbl_sw, bool need_grad,
    int N, Cfg cfg, const float* e_site, const float* e_lr,
    const double* e_zbl, double* energy_out, double* atom_energy_out, float* w_zbl_out);

// Tiny am -> hgemm2 exponent converter
void launch_am_to_eA(cudaStream_t st, int N, const float* am_l1, int8_t* eA_x);

}  // namespace lrsk_fast
}  // namespace loup
