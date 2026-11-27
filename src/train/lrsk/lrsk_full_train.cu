// General Full Pure-CUDA Trainer for FastLRSKStudent (src/train/lrsk_full_train.cu)
// Runs ENTIRE training epochs asynchronously on GPU with ZERO Python/PyTorch autograd in the loop!
//
// Features:
//  - Works for ANY atomic structure (crystals, liquids, molecules, variable graph sizes, any S).
//  - Links src/train/lrsk_train.cu (exact 1st & 2nd-order L1 + LRSK pair kernels) + cuBLAS (NO TF32).
//  - GPU Block-Jacobi SOAP + Compact Cautious AMSGrad-Adam + Grad-Clip + Fused EMA (`lrsk_full_soap.cuh`).
//  - Instantiated for BOTH float32 (ultra-fast training) AND float64 (exact <1e-12 gradcheck parity!).
//  - Clean extern "C" API (ctypes-callable from Python, zero PyTorch C++ ABI dependency).

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <random>

#include "lrsk_full_kernels.cuh"
#include "lrsk_full_soap.cuh"

// External C API declarations from src/train/lrsk_train.cu
extern "C" {
int l1_force_fwd_f32(void*, int, int, int, int, int, double, double, int,
    const int*, const int*, const float*, const int*, const int*, const float*, float*, float*);
int l1_force_fwd_f64(void*, int, int, int, int, int, double, double, int,
    const int*, const int*, const double*, const int*, const int*, const double*, double*, double*);

int l1_force_bwd2_f32(void*, int, int, int, int, double, double, int,
    const int*, const int*, const float*, const int*, const int*, const float*, const float*, float*);
int l1_force_bwd2_f64(void*, int, int, int, int, double, double, int,
    const int*, const int*, const double*, const int*, const int*, const double*, const double*, double*);

int lrsk_pair_fwd_f32(void*, int, int, int, int, int, int, int, double, double, double,
    const int*, const int*, const float*, const int*, const float*, const float*, const float*, const float*,
    float*, float*, float*);
int lrsk_pair_fwd_f64(void*, int, int, int, int, int, int, int, double, double, double,
    const int*, const int*, const double*, const int*, const double*, const double*, const double*, const double*,
    double*, double*, double*);

int lrsk_pair_bwd1_f32(void*, int, int, int, int, int, int, int, int, double, double, double,
    int, int, int, const int*, const int*, const float*, const int*, const int*,
    const float*, const float*, const float*, const float*, const float*, const float*, const float*,
    float*, float*, float*, float*, float*, float*);
int lrsk_pair_bwd1_f64(void*, int, int, int, int, int, int, int, int, double, double, double,
    int, int, int, const int*, const int*, const double*, const int*, const int*,
    const double*, const double*, const double*, const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, double*, double*);

int lrsk_pair_bwd2_f32(void*, int, int, int, int, int, int, int, double, double, double, int, int,
    const int*, const int*, const float*, const int*, const int*,
    const float*, const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*,
    float*, float*, float*, float*, float*, float*, float*);
int lrsk_pair_bwd2_f64(void*, int, int, int, int, int, int, int, double, double, double, int, int,
    const int*, const int*, const double*, const int*, const int*,
    const double*, const double*, const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, double*, double*, double*);
}

namespace lrsk_ft {

#define CUDA_CHK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return (int)_e; \
    } \
} while (0)

#define CUBLAS_CHK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__, (int)_s); \
        return 1000 + (int)_s; \
    } \
} while (0)

template <typename T>
inline cublasStatus_t gemm_t(
    cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc
);
template <>
inline cublasStatus_t gemm_t<float>(
    cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc
) {
    return cublasSgemm(h, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
template <>
inline cublasStatus_t gemm_t<double>(
    cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc
) {
    return cublasDgemm(h, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

// Templated dispatches to lrsk_train.cu
template <typename T>
inline int call_l1_force_fwd(cudaStream_t st, int N, int G, int S, int K1, int lmax, double r_max, double warp,
    int cv, const int* rp, const int* snd, const T* vec, const int* z, const int* ab, const T* Gb, T* F, T* V);
template <> inline int call_l1_force_fwd<float>(cudaStream_t st, int N, int G, int S, int K1, int lmax, double r_max, double warp,
    int cv, const int* rp, const int* snd, const float* vec, const int* z, const int* ab, const float* Gb, float* F, float* V) {
    return l1_force_fwd_f32(st, N, G, S, K1, lmax, r_max, warp, cv, rp, snd, vec, z, ab, Gb, F, V);
}
template <> inline int call_l1_force_fwd<double>(cudaStream_t st, int N, int G, int S, int K1, int lmax, double r_max, double warp,
    int cv, const int* rp, const int* snd, const double* vec, const int* z, const int* ab, const double* Gb, double* F, double* V) {
    return l1_force_fwd_f64(st, N, G, S, K1, lmax, r_max, warp, cv, rp, snd, vec, z, ab, Gb, F, V);
}

template <typename T>
inline int call_l1_force_bwd2(cudaStream_t st, int N, int S, int K1, int lmax, double r_max, double warp,
    int hv, const int* rp, const int* snd, const T* vec, const int* z, const int* ab, const T* dF, const T* dV, T* dG);
template <> inline int call_l1_force_bwd2<float>(cudaStream_t st, int N, int S, int K1, int lmax, double r_max, double warp,
    int hv, const int* rp, const int* snd, const float* vec, const int* z, const int* ab, const float* dF, const float* dV, float* dG) {
    return l1_force_bwd2_f32(st, N, S, K1, lmax, r_max, warp, hv, rp, snd, vec, z, ab, dF, dV, dG);
}
template <> inline int call_l1_force_bwd2<double>(cudaStream_t st, int N, int S, int K1, int lmax, double r_max, double warp,
    int hv, const int* rp, const int* snd, const double* vec, const int* z, const int* ab, const double* dF, const double* dV, double* dG) {
    return l1_force_bwd2_f64(st, N, S, K1, lmax, r_max, warp, hv, rp, snd, vec, z, ab, dF, dV, dG);
}

template <typename T>
inline int call_lrsk_pair_fwd(cudaStream_t st, int N, int S, int K2, int c0, int c1, int c2, int pf,
    double rm2, double w, double es, const int* rp, const int* snd, const T* vec, const int* z,
    const T* q, const T* p, const T* Q, const T* We, T* elr, T* Phi, T* Ef);
template <> inline int call_lrsk_pair_fwd<float>(cudaStream_t st, int N, int S, int K2, int c0, int c1, int c2, int pf,
    double rm2, double w, double es, const int* rp, const int* snd, const float* vec, const int* z,
    const float* q, const float* p, const float* Q, const float* We, float* elr, float* Phi, float* Ef) {
    return lrsk_pair_fwd_f32(st, N, S, K2, c0, c1, c2, pf, rm2, w, es, rp, snd, vec, z, q, p, Q, We, elr, Phi, Ef);
}
template <> inline int call_lrsk_pair_fwd<double>(cudaStream_t st, int N, int S, int K2, int c0, int c1, int c2, int pf,
    double rm2, double w, double es, const int* rp, const int* snd, const double* vec, const int* z,
    const double* q, const double* p, const double* Q, const double* We, double* elr, double* Phi, double* Ef) {
    return lrsk_pair_fwd_f64(st, N, S, K2, c0, c1, c2, pf, rm2, w, es, rp, snd, vec, z, q, p, Q, We, elr, Phi, Ef);
}

template <typename T>
inline int call_lrsk_pair_bwd1(cudaStream_t st, int N, int G, int S, int K2, int c0, int c1, int c2, int pf,
    double rm2, double w, double es, int cf, int cv, int cw,
    const int* rp, const int* snd, const T* vec, const int* z, const int* ab,
    const T* q, const T* p, const T* Q, const T* celr, const T* cPhi, const T* cE, const T* We,
    T* dq, T* dp, T* dQ, T* F, T* V, T* gW);
template <> inline int call_lrsk_pair_bwd1<float>(cudaStream_t st, int N, int G, int S, int K2, int c0, int c1, int c2, int pf,
    double rm2, double w, double es, int cf, int cv, int cw,
    const int* rp, const int* snd, const float* vec, const int* z, const int* ab,
    const float* q, const float* p, const float* Q, const float* celr, const float* cPhi, const float* cE, const float* We,
    float* dq, float* dp, float* dQ, float* F, float* V, float* gW) {
    return lrsk_pair_bwd1_f32(st, N, G, S, K2, c0, c1, c2, pf, rm2, w, es, cf, cv, cw, rp, snd, vec, z, ab, q, p, Q, celr, cPhi, cE, We, dq, dp, dQ, F, V, gW);
}
template <> inline int call_lrsk_pair_bwd1<double>(cudaStream_t st, int N, int G, int S, int K2, int c0, int c1, int c2, int pf,
    double rm2, double w, double es, int cf, int cv, int cw,
    const int* rp, const int* snd, const double* vec, const int* z, const int* ab,
    const double* q, const double* p, const double* Q, const double* celr, const double* cPhi, const double* cE, const double* We,
    double* dq, double* dp, double* dQ, double* F, double* V, double* gW) {
    return lrsk_pair_bwd1_f64(st, N, G, S, K2, c0, c1, c2, pf, rm2, w, es, cf, cv, cw, rp, snd, vec, z, ab, q, p, Q, celr, cPhi, cE, We, dq, dp, dQ, F, V, gW);
}

template <typename T>
inline int call_lrsk_pair_bwd2(cudaStream_t st, int N, int S, int K2, int c0, int c1, int c2, int pf,
    double rm2, double w, double es, int hf, int hv,
    const int* rp, const int* snd, const T* vec, const int* z, const int* ab,
    const T* q, const T* p, const T* Q, const T* celr, const T* cPhi, const T* cE, const T* We,
    const T* ddq, const T* ddp, const T* ddQ, const T* dF, const T* dV,
    T* gq, T* gp, T* gQ, T* gcelr, T* gcPhi, T* gcE, T* gW);
template <> inline int call_lrsk_pair_bwd2<float>(cudaStream_t st, int N, int S, int K2, int c0, int c1, int c2, int pf,
    double rm2, double w, double es, int hf, int hv,
    const int* rp, const int* snd, const float* vec, const int* z, const int* ab,
    const float* q, const float* p, const float* Q, const float* celr, const float* cPhi, const float* cE, const float* We,
    const float* ddq, const float* ddp, const float* ddQ, const float* dF, const float* dV,
    float* gq, float* gp, float* gQ, float* gcelr, float* gcPhi, float* gcE, float* gW) {
    return lrsk_pair_bwd2_f32(st, N, S, K2, c0, c1, c2, pf, rm2, w, es, hf, hv, rp, snd, vec, z, ab, q, p, Q, celr, cPhi, cE, We, ddq, ddp, ddQ, dF, dV, gq, gp, gQ, gcelr, gcPhi, gcE, gW);
}
template <> inline int call_lrsk_pair_bwd2<double>(cudaStream_t st, int N, int S, int K2, int c0, int c1, int c2, int pf,
    double rm2, double w, double es, int hf, int hv,
    const int* rp, const int* snd, const double* vec, const int* z, const int* ab,
    const double* q, const double* p, const double* Q, const double* celr, const double* cPhi, const double* cE, const double* We,
    const double* ddq, const double* ddp, const double* ddQ, const double* dF, const double* dV,
    double* gq, double* gp, double* gQ, double* gcelr, double* gcPhi, double* gcE, double* gW) {
    return lrsk_pair_bwd2_f64(st, N, S, K2, c0, c1, c2, pf, rm2, w, es, hf, hv, rp, snd, vec, z, ab, q, p, Q, celr, cPhi, cE, We, ddq, ddp, ddQ, dF, dV, gq, gp, gQ, gcelr, gcPhi, gcE, gW);
}

// =============================================================================
// Helper: GPU Device Vector
// =============================================================================
template <typename U>
struct DevBuf {
    U* ptr = nullptr;
    size_t cap = 0;
    void alloc(size_t n) {
        if (n == 0) return;
        if (n > cap) {
            if (ptr) cudaFree(ptr);
            cudaMalloc(&ptr, n * sizeof(U));
            cap = n;
        }
    }
    void zero(size_t n, cudaStream_t st) {
        if (n > 0 && ptr) cudaMemsetAsync(ptr, 0, n * sizeof(U), st);
    }
    void upload(const U* host, size_t n) {
        alloc(n);
        if (n > 0) cudaMemcpy(ptr, host, n * sizeof(U), cudaMemcpyHostToDevice);
    }
    void download(U* host, size_t n) const {
        if (n > 0 && ptr) cudaMemcpy(host, ptr, n * sizeof(U), cudaMemcpyDeviceToHost);
    }
    ~DevBuf() { if (ptr) cudaFree(ptr); }
};

// =============================================================================
// General Pure-CUDA LRSK Trainer Class (float or double)
// =============================================================================
template <typename T>
class LrskFullTrainer {
public:
    ModelDims D{};
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;

    // CG & sym metadata on GPU
    DevBuf<Nu2Path> d_nu2;
    DevBuf<Nu3Path> d_nu3;
    DevBuf<CgEntry<T>> d_cg;
    DevBuf<int> d_lr_sym_mask;

    // Calibration buffers on GPU
    DevBuf<T> d_mu_x, d_inv_std_x, d_inv_std_b1, d_inv_std_b2;
    DevBuf<T> d_sp_w;
    int has_sp_w = 0;

    // Flat trainable parameter storage + grads + EMA + optimizer states
    struct ParamSlot {
        int id;
        long off, n;
        int ndim;
        int shape[3];
        int is_soap;
        int is_ro;
    };
    std::vector<ParamSlot> slots;
    long total_params = 0;

    // Named indices into `slots` (-1 if absent)
    int idx_C1_R = -1, idx_C1_B = -1, idx_gamma1 = -1, idx_beta1 = -1, idx_eta1 = -1, idx_C1 = -1;
    std::vector<int> idx_ro_W, idx_ro_b;
    int idx_alpha_Z = -1, idx_u_Z = -1;
    int idx_W_qA = -1, idx_b_qA = -1;
    int idx_W_p1A = -1, idx_W_g1A = -1, idx_b_g1A = -1;
    int idx_W_p2A = -1, idx_W_g2A = -1, idx_b_g2A = -1;
    int idx_W_lr_rad = -1;
    int idx_pol_W0_w = -1, idx_pol_W0_b = -1, idx_pol_u = -1;

    DevBuf<T> d_p_flat, d_g_flat, d_ema_flat, d_m_flat, d_v_flat, d_vmax_flat, d_dir_flat;
    DevBuf<lrsk_opt::PDesc<T>> d_pdesc;
    DevBuf<int> d_soap_pids, d_adam_pids;
    int n_soap_pids = 0, n_adam_pids = 0;

    // SOAP metadata & workspace
    std::vector<lrsk_opt::SoapMat> h_soap_mats;
    DevBuf<lrsk_opt::SoapMat> d_soap_mats;
    DevBuf<int2> d_soap_units, d_soap_eig_units;
    int n_soap_units = 0, n_soap_eig_units = 0;
    DevBuf<T> d_soap_ws;
    DevBuf<double> d_norm2, d_loss_acc, d_lf_acc, d_fsq_g_tmp;
    DevBuf<float> d_frame_err;
    DevBuf<int> d_frame_seen;
    DevBuf<T> d_opt_sc; // [6]
    int prof_enabled = 0;

    // Optimizer hyperparameters
    double soap_b1 = 0.98, soap_b2 = 0.98, soap_eps = 1e-8, soap_wd = 2e-4, soap_mult = 1.0;
    int soap_every = 8, soap_max_dim = 128;
    size_t soap_eig_smem_bytes = 0;
    double adam_b1 = 0.85, adam_b2 = 0.96, adam_eps = 1e-8;
    double ema_dmax = 0.96, grad_clip = 1.0;
    double w_lr_reg = 2e-4, rad_smooth = 0.0, wd_ro = 0.0;
    int step_count = 0;

    // GPU-resident full dataset
    int ds_G = 0, ds_split_r2 = 0, ds_has_stress = 0, ds_has_ei = 0, ds_has_mol = 0, ds_any_zbl = 0;
    std::vector<int> h_n_atoms, h_n_e1, h_n_e2, h_n_zbl, h_n_mol;
    DevBuf<int> d_ds_atom_off, d_ds_e1_off, d_ds_e2_off, d_ds_ez_off;
    DevBuf<int> d_ds_z, d_ds_deg1, d_ds_deg2, d_ds_snd1, d_ds_snd2;
    DevBuf<T> d_ds_vec1, d_ds_vec2, d_ds_G1, d_ds_f, d_ds_E, d_ds_vol, d_ds_S, d_ds_gw;
    DevBuf<T> d_ds_ezbl, d_ds_sw, d_ds_dsw, d_ds_ei, d_ds_mol_arms;
    DevBuf<int> d_ds_zsnd, d_ds_zrcv, d_ds_mol_id, d_ds_n_atoms, d_ds_n_mol;
    DevBuf<T> d_ds_zgv, d_ds_zvec;

    // Mini-batch scratch buffers (auto-grow up to max_N, max_E1, max_E2, max_Ez, max_B, max_M)
    int cap_N = 0, cap_E1 = 0, cap_E2 = 0, cap_Ez = 0, cap_B = 0, cap_M = 0;
    DevBuf<int> b_z, b_batch, b_deg1, b_deg2, b_rp1, b_rp2, b_snd1, b_snd2;
    DevBuf<int> b_zsnd, b_zrcv, b_zeb, b_mol_id, b_nmol, b_mol_g;
    DevBuf<T> b_vec1, b_vec2, b_G1, b_fref, b_Eref, b_vol, b_Sref, b_gw, b_na_f;
    DevBuf<T> b_ezbl, b_sw, b_dsw, b_zgv, b_zvec, b_ei, b_mol_arms;

    // Forward / Bwd1 / Bwd2 activation & adjoint scratch
    DevBuf<T> s_C1_eff, s_g_C1_eff, s_W_eff, s_g_Weff1, s_g_Weff2;
    DevBuf<T> s_A1_raw, s_A1, s_f0, s_k1, s_B2, s_Bcat0, s_Bcat1, s_Bcat2, s_x_norm;
    std::vector<DevBuf<T>> s_ro_pre, s_ro_h, s_dot_ro_pre, s_dot_ro_h;
    DevBuf<T> s_e_site, s_q_raw, s_q_cap, s_q, s_v1, s_g1, s_p_raw, s_p_cap, s_p;
    DevBuf<T> s_v2, s_g2, s_Q_raw, s_Q_cap, s_Q;
    DevBuf<T> s_e_lr, s_Phi_raw, s_Ef_raw, s_Phi_cap, s_Ef_cap, s_pol_in, s_pol_pre, s_pol_h, s_e_pol;
    DevBuf<T> s_e_ml, s_e_node, s_c_sw, s_E_pred;
    DevBuf<T> s_c_pol_pre, s_c_pol_in, s_c_Phi_raw, s_c_Ef_raw, s_c_p_pol, s_c_h_pol;
    DevBuf<T> s_dq_pair, s_dp_pair, s_dQ_pair, s_c_p_tot, s_F_lr, s_V_lr;
    DevBuf<T> s_c_h, s_c_ro_tmp1, s_c_ro_tmp2, s_c_Bcat0, s_c_Bcat1, s_c_Bcat2, s_g_B2;
    DevBuf<T> s_g_A1, s_g_A1raw, s_G_bwd, s_F_l1, s_V_l1, s_F_tot, s_V_tot;
    DevBuf<T> s_bar_enode, s_dF, s_dV, s_gs_sum, s_gs_cnt, s_gs_gsum, s_df_net, s_df_trq;
    DevBuf<T> s_dG_bwd, s_dd_gA1raw, s_dd_c_sw, s_bar_eml;
    DevBuf<T> s_dot_A1, s_dot_B2, s_dot_Bcat0, s_dot_Bcat1, s_dot_Bcat2, s_dot_xnorm;
    DevBuf<T> s_dot_q_raw, s_ddq, s_dot_v1, s_dot_g1, s_dot_p_raw, s_ddp;
    DevBuf<T> s_dot_v2, s_dot_g2, s_dot_Q_raw, s_ddQ;
    DevBuf<T> s_g_q2, s_g_p2, s_g_Q2, s_g_celr2, s_g_cPhi2, s_g_cE2;
    DevBuf<T> s_dot_pol_in, s_dot_pol_pre, s_bar_pol_pre, s_hat_pol_pre, s_bar_pol_in, s_hat_pol_in;
    DevBuf<T> s_bar_Phi_raw, s_bar_Ef_raw, s_bar_p_pol, s_bar_h_pol;
    DevBuf<T> s_bar_q1, s_bar_p1, s_bar_Q1, s_bar_q_tot, s_bar_p_tot, s_bar_Q_tot;
    DevBuf<T> s_bar_h, s_hat_h, s_bar_tmp1, s_hat_tmp1, s_bar_tmp2, s_hat_tmp2;
    DevBuf<T> s_bar_Bcat0, s_hat_Bcat0, s_bar_Bcat1, s_hat_Bcat1, s_bar_Bcat2, s_hat_Bcat2;
    DevBuf<T> s_bar_B2, s_hat_B2, s_bar_A1, s_hat_A1, s_bar_A1raw;
    // Zero-atomic head reduction per-atom buffers
    DevBuf<T> s_dot_pol_h;
    DevBuf<T> s_nb1_bq, s_nb1_hq, s_nb1_bg1, s_nb1_hg1, s_nb1_bv1, s_nb1_hv1;
    DevBuf<T> s_nb1_bg2, s_nb1_hg2, s_nb1_bv2, s_nb1_hv2;

    T* p_ptr(int idx) { return (idx >= 0) ? (d_p_flat.ptr + slots[idx].off) : nullptr; }
    T* g_ptr(int idx) { return (idx >= 0) ? (d_g_flat.ptr + slots[idx].off) : nullptr; }

    int add_param(int ndim, int d0, int d1, int d2, int force_adam, int is_ro) {
        ParamSlot s{};
        s.id = (int)slots.size();
        s.off = total_params;
        s.ndim = ndim;
        s.shape[0] = d0; s.shape[1] = d1; s.shape[2] = d2;
        s.n = (long)d0 * (ndim >= 2 ? d1 : 1) * (ndim >= 3 ? d2 : 1);
        int soap = 0;
        if (!force_adam) {
            if (ndim == 2 && std::min(d0, d1) >= 4 && std::max(d0, d1) <= 2048) soap = 1;
            else if (ndim == 3 && std::min(d1, d2) >= 4 && std::max(d1, d2) <= 2048) soap = 1;
        }
        s.is_soap = soap;
        s.is_ro = is_ro;
        // Align each param to 16 floats/doubles for clean vectorization
        long padded = (s.n + 15L) & ~15L;
        total_params += padded;
        slots.push_back(s);
        return s.id;
    }

    int init(
        const ModelDims& dims,
        const Nu2Path* nu2_h, const Nu3Path* nu3_h,
        const CgEntry<T>* cg_h, int n_cg,
        const int* lr_sym_mask_h,
        const T* mu_x_h, const T* std_x_h, const T* std_b1_h, const T* std_b2_h,
        int soap_max_dim_in
    ) {
        D = dims;
        soap_max_dim = soap_max_dim_in;
        CUDA_CHK(cudaStreamCreate(&stream));
        CUBLAS_CHK(cublasCreate(&cublas));
        CUBLAS_CHK(cublasSetStream(cublas, stream));
        CUBLAS_CHK(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH));

        d_nu2.upload(nu2_h, D.n_nu2);
        d_nu3.upload(nu3_h, D.n_nu3);
        d_cg.upload(cg_h, n_cg);
        d_lr_sym_mask.upload(lr_sym_mask_h, D.T_lr);

        d_mu_x.upload(mu_x_h, D.bcat_dim[0]);
        std::vector<T> inv_sx(D.bcat_dim[0]);
        for (int i = 0; i < D.bcat_dim[0]; ++i) inv_sx[i] = T(1) / (std_x_h[i] > T(1e-7) ? std_x_h[i] : T(1e-7));
        d_inv_std_x.upload(inv_sx.data(), D.bcat_dim[0]);
        if (D.c1 > 0) {
            std::vector<T> inv_s1(D.bcat_dim[1]);
            for (int i = 0; i < D.bcat_dim[1]; ++i) inv_s1[i] = T(1) / (std_b1_h[i] > T(1e-7) ? std_b1_h[i] : T(1e-7));
            d_inv_std_b1.upload(inv_s1.data(), D.bcat_dim[1]);
        }
        if (D.c2 > 0) {
            std::vector<T> inv_s2(D.bcat_dim[2]);
            for (int i = 0; i < D.bcat_dim[2]; ++i) inv_s2[i] = T(1) / (std_b2_h[i] > T(1e-7) ? std_b2_h[i] : T(1e-7));
            d_inv_std_b2.upload(inv_s2.data(), D.bcat_dim[2]);
        }

        // Register all parameters in canonical order (matches Python state_dict export!)
        if (D.film_radials) {
            idx_C1_R = add_param(2, D.n_tot1, D.K1, 1, 0, 0);
            idx_C1_B = add_param(2, D.n_tot1, D.K1, 1, 0, 0);
            idx_gamma1 = add_param(2, D.S * D.S, D.n_tot1, 1, 0, 0);
            idx_beta1 = add_param(2, D.S * D.S, D.n_tot1, 1, 0, 0);
            idx_eta1 = add_param(2, D.S, D.n_tot1, 1, 0, 0);
        } else {
            idx_C1 = add_param(3, D.S * D.S, D.n_tot1, D.K1, 0, 0);
        }
        idx_ro_W.resize(D.ro_depth);
        idx_ro_b.resize(D.ro_depth);
        for (int d = 0; d < D.ro_depth; ++d) {
            int din = (d == 0) ? D.bcat_dim[0] : D.ro_width;
            idx_ro_W[d] = add_param(2, D.ro_width, din, 1, 0, 1);
            idx_ro_b[d] = add_param(1, D.ro_width, 1, 1, 1, 1);
        }
        idx_alpha_Z = add_param(2, D.S, D.ro_width, 1, 1, 0);
        idx_u_Z     = add_param(2, D.S, D.ro_width, 1, 1, 1);
        if (D.c0 > 0) {
            idx_W_qA = add_param(3, D.S, D.c0, D.ro_width, 0, 1);
            idx_b_qA = add_param(2, D.S, D.c0, 1, 0, 0);
        }
        if (D.c1 > 0) {
            idx_W_p1A = add_param(2, D.c1, D.bcat_dim[1], 1, 0, 0);
            idx_W_g1A = add_param(3, D.S, D.c1, D.ro_width, 0, 1);
            idx_b_g1A = add_param(2, D.S, D.c1, 1, 0, 0);
        }
        if (D.c2 > 0) {
            idx_W_p2A = add_param(2, D.c2, D.bcat_dim[2], 1, 0, 0);
            idx_W_g2A = add_param(3, D.S, D.c2, D.ro_width, 0, 1);
            idx_b_g2A = add_param(2, D.S, D.c2, 1, 0, 0);
        }
        idx_W_lr_rad = add_param(3, D.S * D.S, D.T_lr, D.K2, 0, 0);
        if (D.pol_field) {
            idx_pol_W0_w = add_param(2, D.pol_width, D.D_pol, 1, 0, 1);
            idx_pol_W0_b = add_param(1, D.pol_width, 1, 1, 1, 1);
            idx_pol_u    = add_param(2, D.S, D.pol_width, 1, 0, 1);
        }

        // Allocate parameter + optimizer flat buffers
        d_p_flat.alloc(total_params); d_p_flat.zero(total_params, stream);
        d_g_flat.alloc(total_params); d_g_flat.zero(total_params, stream);
        d_ema_flat.alloc(total_params); d_ema_flat.zero(total_params, stream);
        d_m_flat.alloc(total_params); d_m_flat.zero(total_params, stream);
        d_v_flat.alloc(total_params); d_v_flat.zero(total_params, stream);
        d_vmax_flat.alloc(total_params); d_vmax_flat.zero(total_params, stream);
        d_dir_flat.alloc(total_params); d_dir_flat.zero(total_params, stream);

        // Build PDesc & SOAP plan
        std::vector<lrsk_opt::PDesc<T>> h_pd(slots.size());
        std::vector<int> h_spids, h_apids;
        std::vector<int2> h_units, h_eunits;
        long ws_off = 0;
        auto align16 = [](long x) { return (x + 15L) & ~15L; };

        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& s = slots[i];
            lrsk_opt::PDesc<T> pd{};
            pd.p = d_p_flat.ptr + s.off;
            pd.g = d_g_flat.ptr + s.off;
            pd.ema = d_ema_flat.ptr + s.off;
            pd.m = d_m_flat.ptr + s.off;
            pd.v = d_v_flat.ptr + s.off;
            pd.vmax = d_vmax_flat.ptr + s.off;
            pd.dir = d_dir_flat.ptr + s.off;
            pd.n = s.n;
            pd.is_soap = s.is_soap;
            pd.is_ro = s.is_ro;
            h_pd[i] = pd;

            if (s.is_soap) {
                h_spids.push_back((int)i);
                int B_blk = (s.ndim == 3) ? s.shape[0] : 1;
                int rows = (s.ndim == 3) ? s.shape[1] : s.shape[0];
                int cols = (s.ndim == 3) ? s.shape[2] : s.shape[1];
                for (int b = 0; b < B_blk; ++b) {
                    lrsk_opt::SoapMat mt{};
                    mt.param = (int)i;
                    mt.off = (long)b * rows * cols;
                    if (rows <= cols) {
                        mt.m = rows; mt.n = cols; mt.xi = cols; mt.xj = 1;
                    } else {
                        mt.m = cols; mt.n = rows; mt.xi = 1; mt.xj = cols;
                    }
                    mt.pm = (soap_max_dim == 0 || mt.m <= soap_max_dim) ? 1 : 0;
                    mt.pn = (soap_max_dim == 0 || mt.n <= soap_max_dim) ? 1 : 0;
                    mt.use_smem = (mt.m <= lrsk_opt::TS && (!mt.pn || mt.n <= lrsk_opt::TS)) ? 1 : 0;
                    mt.nch = (mt.use_smem && !mt.pn && mt.n > lrsk_opt::TS) ? ((mt.n + lrsk_opt::TS - 1) / lrsk_opt::TS) : 1;

                    if (mt.pm) {
                        mt.Lm = ws_off; ws_off = align16(ws_off + (long)mt.m * mt.m);
                        mt.Qm = ws_off; ws_off = align16(ws_off + (long)mt.m * mt.m);
                        mt.Bm = ws_off; ws_off = align16(ws_off + (long)mt.m * mt.m);
                        if (mt.nch > 1) {
                            mt.Lp = ws_off; ws_off = align16(ws_off + (long)mt.nch * mt.m * mt.m);
                        }
                    }
                    if (mt.pn) {
                        mt.Rn = ws_off; ws_off = align16(ws_off + (long)mt.n * mt.n);
                        mt.Qn = ws_off; ws_off = align16(ws_off + (long)mt.n * mt.n);
                        mt.Bn = ws_off; ws_off = align16(ws_off + (long)mt.n * mt.n);
                    }
                    mt.V = ws_off; ws_off = align16(ws_off + (long)mt.m * mt.n);
                    if (!mt.use_smem) {
                        long mn = std::max((long)mt.m * mt.n, std::max((long)mt.m * mt.m, (long)mt.n * mt.n));
                        mt.W1 = ws_off; ws_off = align16(ws_off + mn);
                        mt.W2 = ws_off; ws_off = align16(ws_off + mn);
                        mt.W3 = ws_off; ws_off = align16(ws_off + mn);
                        mt.W4 = ws_off; ws_off = align16(ws_off + mn);
                    }
                    int mid = (int)h_soap_mats.size();
                    h_soap_mats.push_back(mt);
                    for (int ch = 0; ch < mt.nch; ++ch) h_units.push_back(make_int2(mid, ch));
                    if (mt.pm) h_eunits.push_back(make_int2(mid, 0));
                    if (mt.pn) h_eunits.push_back(make_int2(mid, 1));
                }
            } else {
                h_apids.push_back((int)i);
            }
        }
        d_pdesc.upload(h_pd.data(), h_pd.size());
        n_soap_pids = (int)h_spids.size();
        n_adam_pids = (int)h_apids.size();
        if (n_soap_pids > 0) d_soap_pids.upload(h_spids.data(), n_soap_pids);
        if (n_adam_pids > 0) d_adam_pids.upload(h_apids.data(), n_adam_pids);
        if (!h_soap_mats.empty()) d_soap_mats.upload(h_soap_mats.data(), h_soap_mats.size());
        n_soap_units = (int)h_units.size();
        n_soap_eig_units = (int)h_eunits.size();
        int max_eig_d = 0;
        for (const auto& eu : h_eunits) {
            const auto& mt = h_soap_mats[eu.x];
            max_eig_d = std::max(max_eig_d, eu.y == 0 ? mt.m : mt.n);
        }
        soap_eig_smem_bytes = 2u * lrsk_opt::TS * lrsk_opt::LDV * sizeof(T);
        if constexpr (std::is_same<T, float>::value) {
            if (max_eig_d > lrsk_opt::TS && max_eig_d <= 128) {
                soap_eig_smem_bytes = std::max(soap_eig_smem_bytes, (size_t)128u * 129u * sizeof(T));
                CUDA_CHK(cudaFuncSetAttribute(
                    lrsk_opt::k_soap_eig<T>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)soap_eig_smem_bytes
                ));
            }
        }
        if (n_soap_units > 0) d_soap_units.upload(h_units.data(), n_soap_units);
        if (n_soap_eig_units > 0) d_soap_eig_units.upload(h_eunits.data(), n_soap_eig_units);
        if (ws_off > 0) {
            d_soap_ws.alloc(ws_off);
            d_soap_ws.zero(ws_off, stream);
            // Initialize Qm, Qn = Identity
            for (const auto& mt : h_soap_mats) {
                if (mt.pm) {
                    int b = (mt.m * mt.m + 127) / 128;
                    lrsk_opt::k_init_eye<T><<<b, 128, 0, stream>>>(d_soap_ws.ptr + mt.Qm, mt.m);
                }
                if (mt.pn) {
                    int b = (mt.n * mt.n + 127) / 128;
                    lrsk_opt::k_init_eye<T><<<b, 128, 0, stream>>>(d_soap_ws.ptr + mt.Qn, mt.n);
                }
            }
        }
        d_norm2.alloc(1); d_loss_acc.alloc(1); d_lf_acc.alloc(1); d_fsq_g_tmp.alloc(MAX_BATCH); d_opt_sc.alloc(6);
        const char* pe = getenv("LOUP_FT_PROFILE");
        prof_enabled = (pe && pe[0] == '1') ? 1 : 0;

        // Fixed-size weight scratch
        s_C1_eff.alloc((size_t)D.S * D.S * D.n_tot1 * D.K1);
        s_g_C1_eff.alloc((size_t)D.S * D.S * D.n_tot1 * D.K1);
        s_W_eff.alloc((size_t)D.S * D.S * D.T_lr * D.K2);
        s_g_Weff1.alloc((size_t)D.S * D.S * D.T_lr * D.K2);
        s_g_Weff2.alloc((size_t)D.S * D.S * D.T_lr * D.K2);
        s_ro_pre.resize(D.ro_depth); s_ro_h.resize(D.ro_depth);
        s_dot_ro_pre.resize(D.ro_depth); s_dot_ro_h.resize(D.ro_depth);

        CUDA_CHK(cudaStreamSynchronize(stream));
        return 0;
    }

    ~LrskFullTrainer() {
        if (cublas) cublasDestroy(cublas);
        if (stream) cudaStreamDestroy(stream);
    }

    void ensure_scratch(int N, int E1, int E2, int Ez, int B, int M) {
        if (N > cap_N) {
            cap_N = std::max(N, (int)(cap_N * 3 / 2 + 64));
            int g1s = D.S * D.K1 * D.n_sh;
            b_z.alloc(cap_N); b_batch.alloc(cap_N); b_deg1.alloc(cap_N); b_deg2.alloc(cap_N);
            b_rp1.alloc(cap_N + 1); b_rp2.alloc(cap_N + 1);
            b_G1.alloc((size_t)cap_N * g1s); b_fref.alloc((size_t)cap_N * 3);
            b_ezbl.alloc(cap_N); b_sw.alloc(cap_N); b_dsw.alloc(cap_N);
            b_ei.alloc(cap_N); b_mol_id.alloc(cap_N); b_mol_arms.alloc((size_t)cap_N * 3);

            s_A1_raw.alloc((size_t)cap_N * D.D_A1); s_A1.alloc((size_t)cap_N * D.D_A1);
            s_f0.alloc(cap_N); s_k1.alloc(cap_N);
            s_B2.alloc((size_t)cap_N * std::max(1, D.B2_tot));
            s_Bcat0.alloc((size_t)cap_N * D.bcat_dim[0]); s_x_norm.alloc((size_t)cap_N * D.bcat_dim[0]);
            if (D.c1 > 0) {
                s_Bcat1.alloc((size_t)cap_N * D.bcat_dim[1] * 3);
                s_c_Bcat1.alloc((size_t)cap_N * D.bcat_dim[1] * 3);
                s_dot_Bcat1.alloc((size_t)cap_N * D.bcat_dim[1] * 3);
                s_bar_Bcat1.alloc((size_t)cap_N * D.bcat_dim[1] * 3);
                s_hat_Bcat1.alloc((size_t)cap_N * D.bcat_dim[1] * 3);
            }
            if (D.c2 > 0) {
                s_Bcat2.alloc((size_t)cap_N * D.bcat_dim[2] * 5);
                s_c_Bcat2.alloc((size_t)cap_N * D.bcat_dim[2] * 5);
                s_dot_Bcat2.alloc((size_t)cap_N * D.bcat_dim[2] * 5);
                s_bar_Bcat2.alloc((size_t)cap_N * D.bcat_dim[2] * 5);
                s_hat_Bcat2.alloc((size_t)cap_N * D.bcat_dim[2] * 5);
            }
            for (int d = 0; d < D.ro_depth; ++d) {
                s_ro_pre[d].alloc((size_t)cap_N * D.ro_width);
                s_ro_h[d].alloc((size_t)cap_N * D.ro_width);
                s_dot_ro_pre[d].alloc((size_t)cap_N * D.ro_width);
                s_dot_ro_h[d].alloc((size_t)cap_N * D.ro_width);
            }
            s_e_site.alloc(cap_N);
            if (D.c0 > 0) {
                size_t nc0 = (size_t)cap_N * D.c0;
                s_q_raw.alloc(nc0); s_q_cap.alloc(nc0); s_q.alloc(nc0);
                s_Phi_raw.alloc(nc0); s_Phi_cap.alloc(nc0); s_c_Phi_raw.alloc(nc0);
                s_dq_pair.alloc(nc0); s_dot_q_raw.alloc(nc0); s_ddq.alloc(nc0);
                s_g_q2.alloc(nc0); s_g_cPhi2.alloc(nc0); s_bar_Phi_raw.alloc(nc0);
                s_bar_q1.alloc(nc0); s_bar_q_tot.alloc(nc0);
                s_nb1_bq.alloc(nc0); s_nb1_hq.alloc(nc0);
            }
            if (D.c1 > 0) {
                size_t nc1 = (size_t)cap_N * D.c1, nc13 = nc1 * 3;
                s_v1.alloc(nc13); s_g1.alloc(nc1); s_p_raw.alloc(nc13); s_p_cap.alloc(nc13); s_p.alloc(nc13);
                s_Ef_raw.alloc(nc13); s_Ef_cap.alloc(nc13); s_c_Ef_raw.alloc(nc13); s_c_p_pol.alloc(nc13);
                s_dp_pair.alloc(nc13); s_c_p_tot.alloc(nc13);
                s_dot_v1.alloc(nc13); s_dot_g1.alloc(nc1); s_dot_p_raw.alloc(nc13); s_ddp.alloc(nc13);
                s_g_p2.alloc(nc13); s_g_cE2.alloc(nc13); s_bar_Ef_raw.alloc(nc13); s_bar_p_pol.alloc(nc13);
                s_bar_p1.alloc(nc13); s_bar_p_tot.alloc(nc13);
                s_nb1_bg1.alloc(nc1); s_nb1_hg1.alloc(nc1); s_nb1_bv1.alloc(nc13); s_nb1_hv1.alloc(nc13);
            }
            if (D.c2 > 0) {
                size_t nc2 = (size_t)cap_N * D.c2, nc25 = nc2 * 5;
                s_v2.alloc(nc25); s_g2.alloc(nc2); s_Q_raw.alloc(nc25); s_Q_cap.alloc(nc25); s_Q.alloc(nc25);
                s_dQ_pair.alloc(nc25);
                s_dot_v2.alloc(nc25); s_dot_g2.alloc(nc2); s_dot_Q_raw.alloc(nc25); s_ddQ.alloc(nc25);
                s_g_Q2.alloc(nc25); s_bar_Q1.alloc(nc25); s_bar_Q_tot.alloc(nc25);
                s_nb1_bg2.alloc(nc2); s_nb1_hg2.alloc(nc2); s_nb1_bv2.alloc(nc25); s_nb1_hv2.alloc(nc25);
            }
            s_e_lr.alloc(cap_N); s_e_pol.alloc(cap_N); s_e_ml.alloc(cap_N); s_e_node.alloc(cap_N);
            s_c_sw.alloc(cap_N); s_bar_enode.alloc(cap_N); s_dd_c_sw.alloc(cap_N); s_bar_eml.alloc(cap_N);
            s_g_celr2.alloc(cap_N);
            if (D.pol_field) {
                s_pol_in.alloc((size_t)cap_N * D.D_pol); s_pol_pre.alloc((size_t)cap_N * D.pol_width);
                s_pol_h.alloc((size_t)cap_N * D.pol_width); s_dot_pol_h.alloc((size_t)cap_N * D.pol_width);
                s_c_pol_pre.alloc((size_t)cap_N * D.pol_width);
                s_c_pol_in.alloc((size_t)cap_N * D.D_pol); s_c_h_pol.alloc((size_t)cap_N * D.ro_width);
                s_dot_pol_in.alloc((size_t)cap_N * D.D_pol); s_dot_pol_pre.alloc((size_t)cap_N * D.pol_width);
                s_bar_pol_pre.alloc((size_t)cap_N * D.pol_width); s_hat_pol_pre.alloc((size_t)cap_N * D.pol_width);
                s_bar_pol_in.alloc((size_t)cap_N * D.D_pol); s_hat_pol_in.alloc((size_t)cap_N * D.D_pol);
                s_bar_h_pol.alloc((size_t)cap_N * D.ro_width);
            }
            int max_ro = std::max(D.ro_width, D.bcat_dim[0]);
            s_c_h.alloc((size_t)cap_N * D.ro_width);
            s_c_ro_tmp1.alloc((size_t)cap_N * max_ro); s_c_ro_tmp2.alloc((size_t)cap_N * max_ro);
            s_c_Bcat0.alloc((size_t)cap_N * D.bcat_dim[0]);
            s_g_B2.alloc((size_t)cap_N * std::max(1, D.B2_tot));
            s_g_A1.alloc((size_t)cap_N * D.D_A1); s_g_A1raw.alloc((size_t)cap_N * D.D_A1);
            s_G_bwd.alloc((size_t)cap_N * g1s); s_dG_bwd.alloc((size_t)cap_N * g1s);
            s_dd_gA1raw.alloc((size_t)cap_N * D.D_A1);
            s_F_l1.alloc((size_t)cap_N * 3); s_F_lr.alloc((size_t)cap_N * 3);
            s_F_tot.alloc((size_t)cap_N * 3); s_dF.alloc((size_t)cap_N * 3);
            s_dot_A1.alloc((size_t)cap_N * D.D_A1);
            s_dot_B2.alloc((size_t)cap_N * std::max(1, D.B2_tot));
            s_dot_Bcat0.alloc((size_t)cap_N * D.bcat_dim[0]); s_dot_xnorm.alloc((size_t)cap_N * D.bcat_dim[0]);
            s_bar_h.alloc((size_t)cap_N * D.ro_width); s_hat_h.alloc((size_t)cap_N * D.ro_width);
            s_bar_tmp1.alloc((size_t)cap_N * max_ro); s_hat_tmp1.alloc((size_t)cap_N * max_ro);
            s_bar_tmp2.alloc((size_t)cap_N * max_ro); s_hat_tmp2.alloc((size_t)cap_N * max_ro);
            s_bar_Bcat0.alloc((size_t)cap_N * D.bcat_dim[0]); s_hat_Bcat0.alloc((size_t)cap_N * D.bcat_dim[0]);
            s_bar_B2.alloc((size_t)cap_N * std::max(1, D.B2_tot)); s_hat_B2.alloc((size_t)cap_N * std::max(1, D.B2_tot));
            s_bar_A1.alloc((size_t)cap_N * D.D_A1); s_hat_A1.alloc((size_t)cap_N * D.D_A1);
            s_bar_A1raw.alloc((size_t)cap_N * D.D_A1);
        }
        if (E1 > cap_E1) { cap_E1 = std::max(E1, cap_E1 * 3 / 2 + 256); b_snd1.alloc(cap_E1); b_vec1.alloc((size_t)cap_E1 * 3); }
        if (E2 > cap_E2) { cap_E2 = std::max(E2, cap_E2 * 3 / 2 + 256); b_snd2.alloc(cap_E2); b_vec2.alloc((size_t)cap_E2 * 3); }
        if (Ez > cap_Ez) {
            cap_Ez = std::max(Ez, cap_Ez * 3 / 2 + 64);
            b_zsnd.alloc(cap_Ez); b_zrcv.alloc(cap_Ez); b_zeb.alloc(cap_Ez);
            b_zgv.alloc((size_t)cap_Ez * 3); b_zvec.alloc((size_t)cap_Ez * 3);
        }
        if (B > cap_B) {
            cap_B = std::max(B, 64);
            b_Eref.alloc(cap_B); b_vol.alloc(cap_B); b_Sref.alloc((size_t)cap_B * 9);
            b_gw.alloc(cap_B); b_na_f.alloc(cap_B); b_nmol.alloc(cap_B);
            s_E_pred.alloc(cap_B); s_V_l1.alloc((size_t)cap_B * 9); s_V_lr.alloc((size_t)cap_B * 9);
            s_V_tot.alloc((size_t)cap_B * 9); s_dV.alloc((size_t)cap_B * 9);
            s_gs_sum.alloc((size_t)cap_B * D.S); s_gs_cnt.alloc((size_t)cap_B * D.S); s_gs_gsum.alloc((size_t)cap_B * D.S);
        }
        if (M > cap_M) {
            cap_M = std::max(M, cap_M * 3 / 2 + 64);
            b_mol_g.alloc(cap_M); s_df_net.alloc((size_t)cap_M * 3); s_df_trq.alloc((size_t)cap_M * 3);
        }
    }

    // =========================================================================
    // Execute 1 Complete Step: Gather -> Fwd+Bwd1 -> Loss -> Exact 2nd-Order Bwd2 -> Regs
    // =========================================================================
    int compute_batch_grads(
        const BatchDesc& bd, int n_mol_g,
        double we, double wf, double ws, double wei,
        double w_mol, double net_s, double trq_s,
        int do_regs, int per_graph_f_norm = 0, int update_ferr = 0,
        cudaEvent_t* pev = nullptr
    ) {
        const int N = bd.N, B = bd.B;
        const int compute_stress = (ws > 0.0 && ds_has_stress) ? 1 : 0;
        ensure_scratch(N, bd.E1, bd.E2, std::max(1, bd.Ez), B, std::max(1, bd.M_mol));
        auto blk = [](int n, int t = 128) { return (n + t - 1) / t; };
        const int g1s = D.S * D.K1 * D.n_sh;
        auto mark = [&](int idx) { if (pev) cudaEventRecord(pev[idx], stream); };

        mark(0);
        // 0. Zero param grads + gather mini-batch on GPU (ZERO CPU sync!)
        d_g_flat.zero(total_params, stream);
        k_gather_atoms_noG1<T><<<blk(N), 128, 0, stream>>>(
            bd, ds_has_ei, ds_has_mol, d_ds_atom_off.ptr, d_ds_z.ptr, d_ds_f.ptr,
            d_ds_ezbl.ptr, d_ds_sw.ptr, d_ds_dsw.ptr, d_ds_ei.ptr,
            d_ds_mol_id.ptr, d_ds_mol_arms.ptr, d_ds_deg1.ptr, d_ds_deg2.ptr, ds_split_r2,
            b_z.ptr, b_batch.ptr, b_fref.ptr, b_ezbl.ptr, b_sw.ptr, b_dsw.ptr,
            b_ei.ptr, b_mol_id.ptr, b_mol_arms.ptr, b_deg1.ptr, b_deg2.ptr
        );
        k_gather_G1_fast<T><<<N, 128, 0, stream>>>(bd, g1s, d_ds_atom_off.ptr, d_ds_G1.ptr, b_G1.ptr);
        k_prefix_sum_i32<<<1, 256, 0, stream>>>(N, b_deg1.ptr, b_rp1.ptr);
        if (bd.E1 > 0) {
            k_gather_edges<T><<<blk(bd.E1, 256), 256, 0, stream>>>(bd, 0, d_ds_e1_off.ptr, d_ds_snd1.ptr, d_ds_vec1.ptr, b_snd1.ptr, b_vec1.ptr);
        }
        const int* rp2 = b_rp1.ptr;
        const int* snd2 = b_snd1.ptr;
        const T* vec2 = b_vec1.ptr;
        if (ds_split_r2) {
            k_prefix_sum_i32<<<1, 256, 0, stream>>>(N, b_deg2.ptr, b_rp2.ptr);
            if (bd.E2 > 0) {
                k_gather_edges<T><<<blk(bd.E2, 256), 256, 0, stream>>>(bd, 1, d_ds_e2_off.ptr, d_ds_snd2.ptr, d_ds_vec2.ptr, b_snd2.ptr, b_vec2.ptr);
            }
            rp2 = b_rp2.ptr; snd2 = b_snd2.ptr; vec2 = b_vec2.ptr;
        }
        if (bd.Ez > 0) {
            k_gather_zbl<T><<<blk(bd.Ez), 128, 0, stream>>>(
                bd, d_ds_ez_off.ptr, d_ds_zsnd.ptr, d_ds_zrcv.ptr, d_ds_zgv.ptr, d_ds_zvec.ptr,
                b_zsnd.ptr, b_zrcv.ptr, b_zeb.ptr, b_zgv.ptr, b_zvec.ptr
            );
        }
        k_gather_graphs<T><<<blk(B, 32), 32, 0, stream>>>(
            bd, ds_has_stress, ds_has_mol, d_ds_E.ptr, d_ds_vol.ptr, d_ds_S.ptr, d_ds_gw.ptr,
            d_ds_n_atoms.ptr, d_ds_n_mol.ptr, b_Eref.ptr, b_vol.ptr, b_Sref.ptr, b_gw.ptr, b_na_f.ptr, b_nmol.ptr, b_mol_g.ptr
        );
        mark(1);

        // 1.1 Weight Prep: C1_eff & W_eff
        const int n_c1 = D.S * D.S * D.n_tot1 * D.K1;
        const T* C1_eff = nullptr;
        if (D.film_radials) {
            k_c1_film_fwd<T><<<blk(n_c1), 128, 0, stream>>>(
                D.S, D.n_tot1, D.K1, p_ptr(idx_C1_R), p_ptr(idx_C1_B), p_ptr(idx_gamma1), p_ptr(idx_beta1), p_ptr(idx_eta1), s_C1_eff.ptr
            );
            C1_eff = s_C1_eff.ptr;
        } else {
            C1_eff = p_ptr(idx_C1);
        }
        const int n_wlr = D.S * D.S * D.T_lr * D.K2;
        k_weff_sym_fwd<T><<<blk(n_wlr), 128, 0, stream>>>(
            D.S, D.T_lr, D.K2, D.sym, d_lr_sym_mask.ptr, p_ptr(idx_W_lr_rad), s_W_eff.ptr
        );

        // 1.2 L1 Radial Contraction: G1 -> A1_raw
        k_l1_contract_fwd<T><<<blk(N * D.D_A1, 256), 256, 0, stream>>>(N, D, b_z.ptr, C1_eff, b_G1.ptr, s_A1_raw.ptr);

        // 1.3 NodeBlock1 Forward
        k_soft_norm_fwd<T><<<blk(N, 4), 128, 0, stream>>>(N, D, s_A1_raw.ptr, s_A1.ptr, s_f0.ptr, s_k1.ptr);
        k_cg_fwd<T><<<blk(N * D.max_c_cg), 128, 0, stream>>>(
            N, D, d_nu2.ptr, d_nu3.ptr, d_cg.ptr, s_A1.ptr, s_B2.ptr, s_Bcat0.ptr, s_Bcat1.ptr, s_Bcat2.ptr
        );
        k_norm_x1_fwd<T><<<blk(N * D.bcat_dim[0], 256), 256, 0, stream>>>(
            N, D.bcat_dim[0], s_Bcat0.ptr, d_mu_x.ptr, d_inv_std_x.ptr, s_x_norm.ptr
        );
        for (int d = 0; d < D.ro_depth; ++d) {
            const int din = (d == 0) ? D.bcat_dim[0] : D.ro_width;
            const T* X = (d == 0) ? s_x_norm.ptr : s_ro_h[d - 1].ptr;
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D.ro_width, N, din,
                T(1), p_ptr(idx_ro_W[d]), din, X, din, T(0), s_ro_pre[d].ptr, D.ro_width));
            k_bias_alpha_silu_fwd<T><<<blk(N * D.ro_width, 256), 256, 0, stream>>>(
                N, D.ro_width, b_z.ptr, p_ptr(idx_ro_b[d]), (d == 0) ? p_ptr(idx_alpha_Z) : nullptr,
                s_ro_pre[d].ptr, s_ro_h[d].ptr
            );
        }
        const T* h_last = s_ro_h[D.ro_depth - 1].ptr;
        k_nb1_heads_fwd<T><<<blk(N), 128, 0, stream>>>(
            N, D, b_z.ptr, h_last, s_Bcat1.ptr, s_Bcat2.ptr, d_inv_std_b1.ptr, d_inv_std_b2.ptr, b_sw.ptr,
            p_ptr(idx_u_Z), p_ptr(idx_W_qA), p_ptr(idx_b_qA),
            p_ptr(idx_W_p1A), p_ptr(idx_W_g1A), p_ptr(idx_b_g1A),
            p_ptr(idx_W_p2A), p_ptr(idx_W_g2A), p_ptr(idx_b_g2A),
            s_e_site.ptr, s_q_raw.ptr, s_q_cap.ptr, s_q.ptr,
            s_v1.ptr, s_g1.ptr, s_p_raw.ptr, s_p_cap.ptr, s_p.ptr,
            s_v2.ptr, s_g2.ptr, s_Q_raw.ptr, s_Q_cap.ptr, s_Q.ptr
        );

        // 1.4 LRSK Pair Forward
        mark(2);
        int rc = call_lrsk_pair_fwd<T>(
            stream, N, D.S, D.K2, D.c0, D.c1, D.c2, D.pol_field, D.r_max2, D.radial_warp, D.energy_scale,
            rp2, snd2, vec2, b_z.ptr, s_q.ptr, s_p.ptr, s_Q.ptr, s_W_eff.ptr,
            s_e_lr.ptr, s_Phi_raw.ptr, s_Ef_raw.ptr
        );
        if (rc) return rc;
        mark(3);

        // 1.5 NodeBlock2 Forward + Combine E_node + NodeBlock2 Bwd1
        if (D.pol_field) {
            k_nb2_pack_fwd<T><<<blk(N), 128, 0, stream>>>(
                N, D, s_Phi_raw.ptr, s_Ef_raw.ptr, s_p.ptr, h_last, b_sw.ptr,
                s_Phi_cap.ptr, s_Ef_cap.ptr, s_pol_in.ptr
            );
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D.pol_width, N, D.D_pol,
                T(1), p_ptr(idx_pol_W0_w), D.D_pol, s_pol_in.ptr, D.D_pol, T(0), s_pol_pre.ptr, D.pol_width));
            k_bias_alpha_silu_fwd<T><<<blk(N * D.pol_width, 256), 256, 0, stream>>>(
                N, D.pol_width, b_z.ptr, p_ptr(idx_pol_W0_b), nullptr, s_pol_pre.ptr, s_pol_h.ptr
            );
            k_nb2_epol_and_pre_bwd1<T><<<blk(N), 128, 0, stream>>>(
                N, D, b_z.ptr, s_pol_pre.ptr, s_pol_h.ptr, p_ptr(idx_pol_u), b_sw.ptr,
                s_e_pol.ptr, s_c_pol_pre.ptr
            );
        }
        s_E_pred.zero(B, stream);
        k_combine_enode<T><<<blk(N), 128, 0, stream>>>(
            N, D.use_zbl_switch, D.pol_field, s_e_site.ptr, s_e_lr.ptr, s_e_pol.ptr,
            b_sw.ptr, b_ezbl.ptr, b_batch.ptr, s_e_ml.ptr, s_e_node.ptr, s_c_sw.ptr, s_E_pred.ptr
        );
        if (D.pol_field) {
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D.D_pol, N, D.pol_width,
                T(1), p_ptr(idx_pol_W0_w), D.D_pol, s_c_pol_pre.ptr, D.pol_width, T(0), s_c_pol_in.ptr, D.D_pol));
            k_nb2_unpack_bwd1<T><<<blk(N), 128, 0, stream>>>(
                N, D, s_c_pol_in.ptr, s_Phi_raw.ptr, s_Phi_cap.ptr, s_Ef_raw.ptr, s_Ef_cap.ptr,
                s_p.ptr, b_sw.ptr, s_c_Phi_raw.ptr, s_c_Ef_raw.ptr, s_c_p_pol.ptr, s_c_h_pol.ptr, s_c_sw.ptr
            );
        }

        // 1.6 LRSK Pair Bwd1 (Forces + Virial + dq, dp, dQ)
        // Note: c_elr = sw (when use_zbl_switch else 1; b_sw is already 1.0 when !use_zbl_switch!)
        mark(4);
        rc = call_lrsk_pair_bwd1<T>(
            stream, N, B, D.S, D.K2, D.c0, D.c1, D.c2, D.pol_field, D.r_max2, D.radial_warp, D.energy_scale,
            1, compute_stress, 0, rp2, snd2, vec2, b_z.ptr, b_batch.ptr,
            s_q.ptr, s_p.ptr, s_Q.ptr, b_sw.ptr,
            D.pol_field ? s_c_Phi_raw.ptr : nullptr,
            D.pol_field ? s_c_Ef_raw.ptr : nullptr,
            s_W_eff.ptr, s_dq_pair.ptr, s_dp_pair.ptr, s_dQ_pair.ptr,
            s_F_lr.ptr, compute_stress ? s_V_lr.ptr : nullptr, nullptr
        );
        if (rc) return rc;
        mark(5);

        const T* c_p_use = s_dp_pair.ptr;
        if (D.c1 > 0 && D.pol_field) {
            k_add_vec<T><<<blk(N * D.c1 * 3, 256), 256, 0, stream>>>(
                (long)N * D.c1 * 3, s_dp_pair.ptr, s_c_p_pol.ptr, s_c_p_tot.ptr
            );
            c_p_use = s_c_p_tot.ptr;
        }

        // 1.7 NodeBlock1 Bwd1 -> g_A1raw -> G_bwd -> L1 Forces+Virial
        k_nb1_heads_bwd1<T><<<blk(N), 128, 0, stream>>>(
            N, D, b_z.ptr, b_sw.ptr, b_sw.ptr, D.pol_field ? s_c_h_pol.ptr : nullptr,
            s_dq_pair.ptr, c_p_use, s_dQ_pair.ptr,
            s_q_raw.ptr, s_q_cap.ptr, s_v1.ptr, s_g1.ptr, s_p_raw.ptr, s_p_cap.ptr,
            s_v2.ptr, s_g2.ptr, s_Q_raw.ptr, s_Q_cap.ptr,
            d_inv_std_b1.ptr, d_inv_std_b2.ptr,
            p_ptr(idx_u_Z), p_ptr(idx_W_qA), p_ptr(idx_W_p1A), p_ptr(idx_W_g1A), p_ptr(idx_W_p2A), p_ptr(idx_W_g2A),
            s_c_h.ptr, s_c_Bcat1.ptr, s_c_Bcat2.ptr, s_c_sw.ptr
        );
        T* cur_ch = s_c_h.ptr;
        for (int d = D.ro_depth - 1; d >= 0; --d) {
            const int din = (d == 0) ? D.bcat_dim[0] : D.ro_width;
            T* cpre = s_c_ro_tmp1.ptr;
            T* cnext = (d == 0) ? s_c_ro_tmp2.ptr : s_c_h.ptr;
            k_silu_bwd_mul<T><<<blk(N * D.ro_width, 256), 256, 0, stream>>>(N, D.ro_width, s_ro_pre[d].ptr, cur_ch, cpre);
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, din, N, D.ro_width,
                T(1), p_ptr(idx_ro_W[d]), din, cpre, D.ro_width, T(0), cnext, din));
            cur_ch = cnext;
        }
        k_scale_by_vec<T><<<blk(N * D.bcat_dim[0], 256), 256, 0, stream>>>(N, D.bcat_dim[0], cur_ch, d_inv_std_x.ptr, s_c_Bcat0.ptr);
        k_cg_bwd1<T><<<blk(N * D.max_c_cg), 128, 0, stream>>>(
            N, D, d_nu2.ptr, d_nu3.ptr, d_cg.ptr, s_A1.ptr, s_B2.ptr,
            s_c_Bcat0.ptr, s_c_Bcat1.ptr, s_c_Bcat2.ptr, s_g_B2.ptr, s_g_A1.ptr
        );
        k_soft_norm_bwd1<T><<<blk(N, 4), 128, 0, stream>>>(N, D, s_A1_raw.ptr, s_f0.ptr, s_k1.ptr, s_g_A1.ptr, s_g_A1raw.ptr);
        k_l1_contract_bwd_G<T><<<blk(N * g1s, 256), 256, 0, stream>>>(N, D, b_z.ptr, C1_eff, s_g_A1raw.ptr, s_G_bwd.ptr);

        rc = call_l1_force_fwd<T>(
            stream, N, B, D.S, D.K1, D.lmax, D.r_max, D.radial_warp, compute_stress,
            b_rp1.ptr, b_snd1.ptr, b_vec1.ptr, b_z.ptr, b_batch.ptr, s_G_bwd.ptr,
            s_F_l1.ptr, compute_stress ? s_V_l1.ptr : nullptr
        );
        if (rc) return rc;

        // 1.8 Total F, V + ZBL
        k_add_vec<T><<<blk(N * 3, 256), 256, 0, stream>>>((long)N * 3, s_F_l1.ptr, s_F_lr.ptr, s_F_tot.ptr);
        if (compute_stress) {
            k_add_vec<T><<<blk(B * 9, 64), 64, 0, stream>>>((long)B * 9, s_V_l1.ptr, s_V_lr.ptr, s_V_tot.ptr);
        }
        if (D.use_zbl_switch && bd.Ez > 0) {
            k_zbl_force_fwd<T><<<blk(bd.Ez), 128, 0, stream>>>(
                bd.Ez, compute_stress, b_zsnd.ptr, b_zrcv.ptr, b_zeb.ptr, b_zgv.ptr, b_zvec.ptr,
                b_dsw.ptr, s_c_sw.ptr, s_F_tot.ptr, compute_stress ? s_V_tot.ptr : nullptr
            );
        }

        // 2. Loss & Incoming Adjoints (bar_enode, dF, dV)
        mark(6);
        if (per_graph_f_norm) {
            d_fsq_g_tmp.zero(B, stream);
            k_loss_ef_pergraph_n3<T><<<blk(N, 256), 256, 0, stream>>>(
                bd, we, wf, b_batch.ptr, b_na_f.ptr, s_E_pred.ptr, b_Eref.ptr,
                s_F_tot.ptr, b_fref.ptr, s_bar_enode.ptr, s_dF.ptr,
                d_fsq_g_tmp.ptr, d_loss_acc.ptr, d_lf_acc.ptr
            );
            k_update_frame_err_n3<T><<<1, 1, 0, stream>>>(
                bd, we, wf, d_fsq_g_tmp.ptr, s_E_pred.ptr, b_Eref.ptr, b_na_f.ptr,
                update_ferr ? d_frame_err.ptr : nullptr,
                update_ferr ? d_frame_seen.ptr : nullptr,
                d_loss_acc.ptr, d_lf_acc.ptr
            );
        } else {
            k_loss_ef_s<T><<<blk(std::max(N, B), 256), 256, 0, stream>>>(
                N, B, compute_stress, has_sp_w, we, wf, ws,
                b_batch.ptr, b_z.ptr, b_gw.ptr, has_sp_w ? d_sp_w.ptr : nullptr,
                b_na_f.ptr, b_vol.ptr, s_E_pred.ptr, b_Eref.ptr,
                s_F_tot.ptr, b_fref.ptr, compute_stress ? s_V_tot.ptr : nullptr, compute_stress ? b_Sref.ptr : nullptr,
                s_bar_enode.ptr, s_dF.ptr, compute_stress ? s_dV.ptr : nullptr, d_loss_acc.ptr
            );
        }
        if (wei > 0.0 && ds_has_ei) {
            s_gs_sum.zero((size_t)B * D.S, stream);
            s_gs_cnt.zero((size_t)B * D.S, stream);
            s_gs_gsum.zero((size_t)B * D.S, stream);
            k_loss_ei_stats<T><<<blk(N), 128, 0, stream>>>(N, D.S, b_batch.ptr, b_z.ptr, s_e_node.ptr, s_gs_sum.ptr, s_gs_cnt.ptr);
            k_loss_ei_grad1<T><<<blk(N), 128, 0, stream>>>(
                N, D.S, wei, b_batch.ptr, b_z.ptr, b_gw.ptr, s_e_node.ptr, b_ei.ptr,
                s_gs_sum.ptr, s_gs_cnt.ptr, s_gs_gsum.ptr, s_bar_enode.ptr, d_loss_acc.ptr
            );
            k_loss_ei_grad2<T><<<blk(N), 128, 0, stream>>>(N, D.S, b_batch.ptr, b_z.ptr, s_gs_gsum.ptr, s_bar_enode.ptr);
        }
        if (w_mol > 0.0 && ds_has_mol && n_mol_g > 0 && bd.M_mol > 0) {
            s_df_net.zero((size_t)bd.M_mol * 3, stream);
            s_df_trq.zero((size_t)bd.M_mol * 3, stream);
            k_loss_mol_accum<T><<<blk(N), 128, 0, stream>>>(
                N, b_mol_id.ptr, b_mol_arms.ptr, s_F_tot.ptr, b_fref.ptr, s_df_net.ptr, s_df_trq.ptr
            );
            k_loss_mol_loss<T><<<blk(bd.M_mol), 128, 0, stream>>>(
                bd.M_mol, n_mol_g, w_mol, net_s, trq_s, b_mol_g.ptr, b_nmol.ptr, b_gw.ptr,
                s_df_net.ptr, s_df_trq.ptr, d_loss_acc.ptr
            );
            k_loss_mol_grad<T><<<blk(N), 128, 0, stream>>>(
                N, n_mol_g, w_mol, net_s, trq_s, b_batch.ptr, b_mol_id.ptr, b_nmol.ptr, b_gw.ptr,
                b_mol_arms.ptr, s_df_net.ptr, s_df_trq.ptr, s_dF.ptr
            );
        }

        // 3.1 L1 Force Bwd2 + ZBL Bwd2
        rc = call_l1_force_bwd2<T>(
            stream, N, D.S, D.K1, D.lmax, D.r_max, D.radial_warp, compute_stress,
            b_rp1.ptr, b_snd1.ptr, b_vec1.ptr, b_z.ptr, b_batch.ptr,
            s_dF.ptr, compute_stress ? s_dV.ptr : nullptr, s_dG_bwd.ptr
        );
        if (rc) return rc;
        k_l1_contract_fwd<T><<<blk(N * D.D_A1, 256), 256, 0, stream>>>(N, D, b_z.ptr, C1_eff, s_dG_bwd.ptr, s_dd_gA1raw.ptr);

        const T* dd_csw_ptr = nullptr;
        if (D.use_zbl_switch && bd.Ez > 0) {
            s_dd_c_sw.zero(N, stream);
            k_zbl_bwd2<T><<<blk(bd.Ez), 128, 0, stream>>>(
                bd.Ez, compute_stress, b_zsnd.ptr, b_zrcv.ptr, b_zeb.ptr, b_zgv.ptr, b_zvec.ptr,
                b_dsw.ptr, s_dF.ptr, compute_stress ? s_dV.ptr : nullptr, s_dd_c_sw.ptr
            );
            dd_csw_ptr = s_dd_c_sw.ptr;
        }
        k_bar_eml_from_enode<T><<<blk(N), 128, 0, stream>>>(N, D.use_zbl_switch, b_sw.ptr, s_bar_enode.ptr, dd_csw_ptr, s_bar_eml.ptr);

        // 3.2 NodeBlock1 JVP (Dual Forward Pass)
        k_soft_norm_dual_fwd<T><<<blk(N, 4), 128, 0, stream>>>(N, D, s_A1_raw.ptr, s_dd_gA1raw.ptr, s_f0.ptr, s_k1.ptr, s_dot_A1.ptr);
        k_cg_dual_fwd<T><<<blk(N * D.max_c_cg), 128, 0, stream>>>(
            N, D, d_nu2.ptr, d_nu3.ptr, d_cg.ptr, s_A1.ptr, s_dot_A1.ptr, s_B2.ptr,
            s_dot_B2.ptr, s_dot_Bcat0.ptr, s_dot_Bcat1.ptr, s_dot_Bcat2.ptr
        );
        k_scale_by_vec<T><<<blk(N * D.bcat_dim[0], 256), 256, 0, stream>>>(N, D.bcat_dim[0], s_dot_Bcat0.ptr, d_inv_std_x.ptr, s_dot_xnorm.ptr);
        for (int d = 0; d < D.ro_depth; ++d) {
            const int din = (d == 0) ? D.bcat_dim[0] : D.ro_width;
            const T* dX = (d == 0) ? s_dot_xnorm.ptr : s_dot_ro_h[d - 1].ptr;
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D.ro_width, N, din,
                T(1), p_ptr(idx_ro_W[d]), din, dX, din, T(0), s_dot_ro_pre[d].ptr, D.ro_width));
            k_silu_bwd_mul<T><<<blk(N * D.ro_width, 256), 256, 0, stream>>>(
                N, D.ro_width, s_ro_pre[d].ptr, s_dot_ro_pre[d].ptr, s_dot_ro_h[d].ptr
            );
        }
        const T* dot_h_last = s_dot_ro_h[D.ro_depth - 1].ptr;
        k_nb1_heads_dual_fwd<T><<<blk(N), 128, 0, stream>>>(
            N, D, b_z.ptr, dot_h_last, s_dot_Bcat1.ptr, s_dot_Bcat2.ptr, d_inv_std_b1.ptr, d_inv_std_b2.ptr,
            b_sw.ptr, dd_csw_ptr, s_q_raw.ptr, s_q_cap.ptr,
            s_v1.ptr, s_g1.ptr, s_p_raw.ptr, s_p_cap.ptr,
            s_v2.ptr, s_g2.ptr, s_Q_raw.ptr, s_Q_cap.ptr,
            p_ptr(idx_W_qA), p_ptr(idx_W_p1A), p_ptr(idx_W_g1A), p_ptr(idx_W_p2A), p_ptr(idx_W_g2A),
            s_dot_q_raw.ptr, s_ddq.ptr, s_dot_v1.ptr, s_dot_g1.ptr, s_dot_p_raw.ptr, s_ddp.ptr,
            s_dot_v2.ptr, s_dot_g2.ptr, s_dot_Q_raw.ptr, s_ddQ.ptr
        );

        // 3.3 LRSK Pair Bwd2 + NodeBlock2 Dual Fwd & Dual Bwd
        mark(7);
        rc = call_lrsk_pair_bwd2<T>(
            stream, N, D.S, D.K2, D.c0, D.c1, D.c2, D.pol_field, D.r_max2, D.radial_warp, D.energy_scale,
            1, compute_stress, rp2, snd2, vec2, b_z.ptr, b_batch.ptr,
            s_q.ptr, s_p.ptr, s_Q.ptr, b_sw.ptr,
            D.pol_field ? s_c_Phi_raw.ptr : nullptr, D.pol_field ? s_c_Ef_raw.ptr : nullptr, s_W_eff.ptr,
            s_ddq.ptr, s_ddp.ptr, s_ddQ.ptr, s_dF.ptr, compute_stress ? s_dV.ptr : nullptr,
            s_g_q2.ptr, s_g_p2.ptr, s_g_Q2.ptr, s_g_celr2.ptr,
            D.pol_field ? s_g_cPhi2.ptr : nullptr, D.pol_field ? s_g_cE2.ptr : nullptr, s_g_Weff2.ptr
        );
        if (rc) return rc;
        mark(8);

        if (D.pol_field) {
            k_nb2_pack_dual_fwd<T><<<blk(N), 128, 0, stream>>>(
                N, D, s_Phi_raw.ptr, s_Phi_cap.ptr, s_g_cPhi2.ptr,
                s_Ef_raw.ptr, s_Ef_cap.ptr, s_g_cE2.ptr,
                s_p.ptr, s_ddp.ptr, dot_h_last, b_sw.ptr, dd_csw_ptr, s_dot_pol_in.ptr
            );
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, D.pol_width, N, D.D_pol,
                T(1), p_ptr(idx_pol_W0_w), D.D_pol, s_dot_pol_in.ptr, D.D_pol, T(0), s_dot_pol_pre.ptr, D.pol_width));
            k_nb2_dual_bwd_pre<T><<<blk(N), 128, 0, stream>>>(
                N, D, b_z.ptr, s_pol_pre.ptr, s_dot_pol_pre.ptr, p_ptr(idx_pol_u),
                s_bar_eml.ptr, b_sw.ptr, s_bar_pol_pre.ptr, s_hat_pol_pre.ptr, s_dot_pol_h.ptr
            );
            auto launch_sp_head = [&](int C_h, int W_h, T scale_h, const T* ba, const T* ha, const T* hh, const T* dhh, T* gW, T* gb) {
                const int nw = (W_h + 31) / 32;
                const int nsp = (D.S * C_h * nw < 32 && N >= 256) ? 4 : 1;
                k_reduce_sp_head_W<T><<<dim3(D.S * C_h, nw, nsp), dim3(32, 8), 0, stream>>>(
                    N, D.S, C_h, W_h, scale_h, b_z.ptr, ba, ha, hh, dhh, gW, gb
                );
            };
            launch_sp_head(1, D.pol_width, (T)D.energy_scale, s_bar_eml.ptr, b_sw.ptr, s_pol_h.ptr, s_dot_pol_h.ptr, g_ptr(idx_pol_u), nullptr);
            // g_pol_W0_w += bar_pol_pre^T @ pol_in + hat_pol_pre^T @ dot_pol_in
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D.D_pol, D.pol_width, N,
                T(1), s_pol_in.ptr, D.D_pol, s_bar_pol_pre.ptr, D.pol_width, T(1), g_ptr(idx_pol_W0_w), D.D_pol));
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_T, D.D_pol, D.pol_width, N,
                T(1), s_dot_pol_in.ptr, D.D_pol, s_hat_pol_pre.ptr, D.pol_width, T(1), g_ptr(idx_pol_W0_w), D.D_pol));
            k_reduce_bias_alpha<T><<<D.pol_width, 128, 0, stream>>>(
                N, D.S, D.pol_width, b_z.ptr, s_bar_pol_pre.ptr, g_ptr(idx_pol_W0_b), nullptr
            );
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D.D_pol, N, D.pol_width,
                T(1), p_ptr(idx_pol_W0_w), D.D_pol, s_bar_pol_pre.ptr, D.pol_width, T(0), s_bar_pol_in.ptr, D.D_pol));
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D.D_pol, N, D.pol_width,
                T(1), p_ptr(idx_pol_W0_w), D.D_pol, s_hat_pol_pre.ptr, D.pol_width, T(0), s_hat_pol_in.ptr, D.D_pol));
            k_nb2_unpack_dual_bwd<T><<<blk(N), 128, 0, stream>>>(
                N, D, s_bar_pol_in.ptr, s_hat_pol_in.ptr,
                s_Phi_raw.ptr, s_Phi_cap.ptr, s_g_cPhi2.ptr,
                s_Ef_raw.ptr, s_Ef_cap.ptr, s_g_cE2.ptr,
                s_p.ptr, s_ddp.ptr, b_sw.ptr, dd_csw_ptr,
                s_bar_Phi_raw.ptr, s_bar_Ef_raw.ptr, s_bar_p_pol.ptr, s_bar_h_pol.ptr
            );
        }

        // 3.4 Primal LRSK Pair Bwd1 (for bar_eml, bar_Phi_raw, bar_Ef_raw -> bar_q1, bar_p1, bar_Q1, g_Weff1)
        rc = call_lrsk_pair_bwd1<T>(
            stream, N, B, D.S, D.K2, D.c0, D.c1, D.c2, D.pol_field, D.r_max2, D.radial_warp, D.energy_scale,
            0, 0, 1, rp2, snd2, vec2, b_z.ptr, b_batch.ptr,
            s_q.ptr, s_p.ptr, s_Q.ptr, s_bar_eml.ptr,
            D.pol_field ? s_bar_Phi_raw.ptr : nullptr, D.pol_field ? s_bar_Ef_raw.ptr : nullptr,
            s_W_eff.ptr, s_bar_q1.ptr, s_bar_p1.ptr, s_bar_Q1.ptr, nullptr, nullptr, s_g_Weff1.ptr
        );
        if (rc) return rc;
        mark(9);
        k_weff_sym_bwd<T><<<blk(n_wlr), 128, 0, stream>>>(
            D.S, D.T_lr, D.K2, D.sym, d_lr_sym_mask.ptr, s_g_Weff1.ptr, s_g_Weff2.ptr, g_ptr(idx_W_lr_rad)
        );

        if (D.c0 > 0) k_add_vec<T><<<blk(N * D.c0, 256), 256, 0, stream>>>((long)N * D.c0, s_bar_q1.ptr, s_g_q2.ptr, s_bar_q_tot.ptr);
        if (D.c1 > 0) {
            if (D.pol_field) {
                k_add3_vec<T><<<blk(N * D.c1 * 3, 256), 256, 0, stream>>>(
                    (long)N * D.c1 * 3, s_bar_p1.ptr, s_g_p2.ptr, s_bar_p_pol.ptr, s_bar_p_tot.ptr
                );
            } else {
                k_add_vec<T><<<blk(N * D.c1 * 3, 256), 256, 0, stream>>>(
                    (long)N * D.c1 * 3, s_bar_p1.ptr, s_g_p2.ptr, s_bar_p_tot.ptr
                );
            }
        }
        if (D.c2 > 0) k_add_vec<T><<<blk(N * D.c2 * 5, 256), 256, 0, stream>>>((long)N * D.c2 * 5, s_bar_Q1.ptr, s_g_Q2.ptr, s_bar_Q_tot.ptr);

        // 3.5 NodeBlock1 Dual Backward (Zero-Atomic Heads -> Readout MLP -> CG -> Soft-Norm)
        k_nb1_heads_dual_bwd_noatom<T><<<blk(N), 128, 0, stream>>>(
            N, D, b_z.ptr,
            d_inv_std_b1.ptr, d_inv_std_b2.ptr, b_sw.ptr, dd_csw_ptr,
            s_bar_eml.ptr, b_sw.ptr,
            D.pol_field ? s_bar_h_pol.ptr : nullptr, D.pol_field ? s_c_h_pol.ptr : nullptr,
            s_bar_q_tot.ptr, s_dq_pair.ptr, s_bar_p_tot.ptr, c_p_use, s_bar_Q_tot.ptr, s_dQ_pair.ptr,
            s_q_raw.ptr, s_dot_q_raw.ptr,
            s_v1.ptr, s_dot_v1.ptr, s_g1.ptr, s_dot_g1.ptr, s_p_raw.ptr, s_dot_p_raw.ptr,
            s_v2.ptr, s_dot_v2.ptr, s_g2.ptr, s_dot_g2.ptr, s_Q_raw.ptr, s_dot_Q_raw.ptr,
            p_ptr(idx_u_Z), p_ptr(idx_W_qA), p_ptr(idx_W_p1A), p_ptr(idx_W_g1A), p_ptr(idx_W_p2A), p_ptr(idx_W_g2A),
            s_bar_h.ptr, s_hat_h.ptr, s_bar_Bcat1.ptr, s_hat_Bcat1.ptr, s_bar_Bcat2.ptr, s_hat_Bcat2.ptr,
            s_nb1_bq.ptr, s_nb1_hq.ptr,
            s_nb1_bg1.ptr, s_nb1_hg1.ptr, s_nb1_bv1.ptr, s_nb1_hv1.ptr,
            s_nb1_bg2.ptr, s_nb1_hg2.ptr, s_nb1_bv2.ptr, s_nb1_hv2.ptr
        );
        auto launch_sp_head_nb1 = [&](int C_h, int W_h, T scale_h, const T* ba, const T* ha, const T* hh, const T* dhh, T* gW, T* gb) {
            const int nw = (W_h + 31) / 32;
            const int nsp = (D.S * C_h * nw < 32 && N >= 256) ? 4 : 1;
            k_reduce_sp_head_W<T><<<dim3(D.S * C_h, nw, nsp), dim3(32, 8), 0, stream>>>(
                N, D.S, C_h, W_h, scale_h, b_z.ptr, ba, ha, hh, dhh, gW, gb
            );
        };
        launch_sp_head_nb1(1, D.ro_width, (T)D.energy_scale, s_bar_eml.ptr, b_sw.ptr, h_last, dot_h_last, g_ptr(idx_u_Z), nullptr);
        if (D.c0 > 0) {
            launch_sp_head_nb1(D.c0, D.ro_width, T(1), s_nb1_bq.ptr, s_nb1_hq.ptr, h_last, dot_h_last, g_ptr(idx_W_qA), g_ptr(idx_b_qA));
        }
        if (D.c1 > 0) {
            launch_sp_head_nb1(D.c1, D.ro_width, T(1), s_nb1_bg1.ptr, s_nb1_hg1.ptr, h_last, dot_h_last, g_ptr(idx_W_g1A), g_ptr(idx_b_g1A));
            const int nsp_wp1 = (N >= 256) ? 4 : 1;
            k_reduce_Wp_grad<T, 3><<<dim3((D.bcat_dim[1] + 31) / 32, D.c1, nsp_wp1), dim3(32, 8), 0, stream>>>(
                N, D.c1, D.bcat_dim[1], d_inv_std_b1.ptr,
                s_nb1_bv1.ptr, s_nb1_hv1.ptr, s_Bcat1.ptr, s_dot_Bcat1.ptr, g_ptr(idx_W_p1A)
            );
        }
        if (D.c2 > 0) {
            launch_sp_head_nb1(D.c2, D.ro_width, T(1), s_nb1_bg2.ptr, s_nb1_hg2.ptr, h_last, dot_h_last, g_ptr(idx_W_g2A), g_ptr(idx_b_g2A));
            const int nsp_wp2 = (N >= 256) ? 4 : 1;
            k_reduce_Wp_grad<T, 5><<<dim3((D.bcat_dim[2] + 31) / 32, D.c2, nsp_wp2), dim3(32, 8), 0, stream>>>(
                N, D.c2, D.bcat_dim[2], d_inv_std_b2.ptr,
                s_nb1_bv2.ptr, s_nb1_hv2.ptr, s_Bcat2.ptr, s_dot_Bcat2.ptr, g_ptr(idx_W_p2A)
            );
        }

        T* cur_bh = s_bar_h.ptr;
        T* cur_hh = s_hat_h.ptr;
        for (int d = D.ro_depth - 1; d >= 0; --d) {
            const int din = (d == 0) ? D.bcat_dim[0] : D.ro_width;
            const T* X = (d == 0) ? s_x_norm.ptr : s_ro_h[d - 1].ptr;
            const T* dX = (d == 0) ? s_dot_xnorm.ptr : s_dot_ro_h[d - 1].ptr;
            T* bpre = s_bar_tmp1.ptr;
            T* hpre = s_hat_tmp1.ptr;
            T* bnext = (d == 0) ? s_bar_tmp2.ptr : s_bar_h.ptr;
            T* hnext = (d == 0) ? s_hat_tmp2.ptr : s_hat_h.ptr;
            k_silu_dual_bwd<T><<<blk(N * D.ro_width, 256), 256, 0, stream>>>(
                N, D.ro_width, s_ro_pre[d].ptr, s_dot_ro_pre[d].ptr, cur_bh, cur_hh, bpre, hpre
            );
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_T, din, D.ro_width, N,
                T(1), X, din, bpre, D.ro_width, T(1), g_ptr(idx_ro_W[d]), din));
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_T, din, D.ro_width, N,
                T(1), dX, din, hpre, D.ro_width, T(1), g_ptr(idx_ro_W[d]), din));
            k_reduce_bias_alpha<T><<<D.ro_width, 128, 0, stream>>>(
                N, D.S, D.ro_width, b_z.ptr, bpre, g_ptr(idx_ro_b[d]), (d == 0) ? g_ptr(idx_alpha_Z) : nullptr
            );
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, din, N, D.ro_width,
                T(1), p_ptr(idx_ro_W[d]), din, bpre, D.ro_width, T(0), bnext, din));
            CUBLAS_CHK(gemm_t<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, din, N, D.ro_width,
                T(1), p_ptr(idx_ro_W[d]), din, hpre, D.ro_width, T(0), hnext, din));
            cur_bh = bnext;
            cur_hh = hnext;
        }
        k_scale_by_vec<T><<<blk(N * D.bcat_dim[0], 256), 256, 0, stream>>>(N, D.bcat_dim[0], cur_bh, d_inv_std_x.ptr, s_bar_Bcat0.ptr);
        k_scale_by_vec<T><<<blk(N * D.bcat_dim[0], 256), 256, 0, stream>>>(N, D.bcat_dim[0], cur_hh, d_inv_std_x.ptr, s_hat_Bcat0.ptr);
        k_cg_dual_bwd<T><<<blk(N * D.max_c_cg), 128, 0, stream>>>(
            N, D, d_nu2.ptr, d_nu3.ptr, d_cg.ptr, s_A1.ptr, s_dot_A1.ptr, s_B2.ptr, s_dot_B2.ptr,
            s_bar_Bcat0.ptr, s_hat_Bcat0.ptr, s_bar_Bcat1.ptr, s_hat_Bcat1.ptr, s_bar_Bcat2.ptr, s_hat_Bcat2.ptr,
            s_bar_B2.ptr, s_hat_B2.ptr, s_bar_A1.ptr, s_hat_A1.ptr
        );
        k_soft_norm_dual_bwd<T><<<blk(N, 4), 128, 0, stream>>>(
            N, D, s_A1_raw.ptr, s_dd_gA1raw.ptr, s_f0.ptr, s_k1.ptr, s_bar_A1.ptr, s_hat_A1.ptr, s_bar_A1raw.ptr
        );

        // 3.6 C1_eff Grad (v3 coalesced SMEM-staged!) + FiLM Bwd
        {
            const int n_c1_blks = D.S * D.S * (D.lmax + 1);
            const int nsp_c1 = (N >= 256) ? std::max(2, std::min(8, 96 / std::max(1, n_c1_blks))) : 1;
            if (D.film_radials && nsp_c1 > 1) s_g_C1_eff.zero(n_c1, stream);
            k_l1_contract_grad_C1<T><<<dim3(n_c1_blks, nsp_c1), 256, 0, stream>>>(
                N, D, b_z.ptr, s_bar_A1raw.ptr, b_G1.ptr, s_g_A1raw.ptr, s_dG_bwd.ptr,
                D.film_radials ? s_g_C1_eff.ptr : g_ptr(idx_C1)
            );
        }
        if (D.film_radials) {
            k_c1_film_bwd<T><<<blk(D.n_tot1, 64), 64, 0, stream>>>(
                D.S, D.n_tot1, D.K1, p_ptr(idx_C1_R), p_ptr(idx_C1_B), p_ptr(idx_gamma1), p_ptr(idx_beta1), p_ptr(idx_eta1),
                s_g_C1_eff.ptr, g_ptr(idx_C1_R), g_ptr(idx_C1_B), g_ptr(idx_gamma1), g_ptr(idx_beta1), g_ptr(idx_eta1)
            );
        }

        // 3.7 Regularization Gradients
        if (do_regs) {
            if (w_lr_reg > 0.0) {
                int rows = D.S * D.S * D.T_lr;
                T coef = (T)(2.0 * w_lr_reg / (double)(rows * D.K2));
                lrsk_opt::k_add_reg_kpow<T><<<blk(rows * D.K2), 128, 0, stream>>>(
                    g_ptr(idx_W_lr_rad), p_ptr(idx_W_lr_rad), rows, D.K2, 4, coef
                );
            }
            if (rad_smooth > 0.0) {
                if (D.film_radials) {
                    int tot = D.n_tot1 * D.K1;
                    T coef = (T)(2.0 * rad_smooth / (double)tot);
                    lrsk_opt::k_add_reg_kpow<T><<<blk(tot), 128, 0, stream>>>(g_ptr(idx_C1_R), p_ptr(idx_C1_R), D.n_tot1, D.K1, 2, coef);
                    lrsk_opt::k_add_reg_kpow<T><<<blk(tot), 128, 0, stream>>>(g_ptr(idx_C1_B), p_ptr(idx_C1_B), D.n_tot1, D.K1, 2, coef);
                } else {
                    int rows = D.S * D.S * D.n_tot1;
                    T coef = (T)(2.0 * rad_smooth / (double)(rows * D.K1));
                    lrsk_opt::k_add_reg_kpow<T><<<blk(rows * D.K1), 128, 0, stream>>>(g_ptr(idx_C1), p_ptr(idx_C1), rows, D.K1, 2, coef);
                }
            }
            if (wd_ro > 0.0) {
                for (const auto& s : slots) {
                    if (!s.is_ro) continue;
                    T coef = (T)(2.0 * wd_ro / (double)s.n);
                    lrsk_opt::k_add_reg_l2<T><<<blk((int)s.n), 128, 0, stream>>>(
                        d_g_flat.ptr + s.off, d_p_flat.ptr + s.off, s.n, coef
                    );
                }
            }
        }
        mark(10);
        return 0;
    }

    // =========================================================================
    // Execute 1 Optimizer Step (Grad-Norm Clip + Cautious-Adam + GPU Block-Jacobi SOAP + EMA)
    // =========================================================================
    int optimizer_step(double lr) {
        step_count += 1;
        d_norm2.zero(1, stream);
        dim3 gn_grid(4, (unsigned)slots.size());
        lrsk_opt::k_grad_norm2<T><<<gn_grid, 256, 0, stream>>>(d_pdesc.ptr, (int)slots.size(), d_norm2.ptr);
        lrsk_opt::k_opt_prep<T><<<1, 1, 0, stream>>>(
            d_norm2.ptr, grad_clip, lr, adam_b1, adam_b2, soap_b1, soap_b2, ema_dmax, step_count, d_opt_sc.ptr
        );
        if (n_adam_pids > 0) {
            lrsk_opt::k_caut_adam_compact<T><<<n_adam_pids, 256, 0, stream>>>(
                d_pdesc.ptr, d_adam_pids.ptr, d_opt_sc.ptr, adam_b1, adam_b2, adam_eps, 1
            );
        }
        if (n_soap_pids > 0 && n_soap_units > 0) {
            dim3 sp_grid(4, (unsigned)n_soap_pids);
            lrsk_opt::k_soap_prep_params<T><<<sp_grid, 256, 0, stream>>>(
                d_pdesc.ptr, d_soap_pids.ptr, d_opt_sc.ptr, lr, soap_mult, soap_wd, soap_b1
            );
            const int do_refresh = ((step_count - 1) % soap_every == 0) ? 1 : 0;
            const size_t smem_bytes = 2u * lrsk_opt::TS * lrsk_opt::LDV * sizeof(T);
            lrsk_opt::k_soap_a<T><<<n_soap_units, 256, smem_bytes, stream>>>(
                d_soap_mats.ptr, d_soap_units.ptr, d_pdesc.ptr, d_opt_sc.ptr, d_soap_ws.ptr,
                lr * soap_mult, soap_b2, soap_eps, do_refresh
            );
            if (do_refresh) {
                const int maxsw = (step_count <= 1) ? 30 : 4;
                if (n_soap_eig_units > 0) {
                    lrsk_opt::k_soap_eig<T><<<n_soap_eig_units, 256, soap_eig_smem_bytes, stream>>>(
                        d_soap_mats.ptr, d_soap_eig_units.ptr, d_soap_ws.ptr, maxsw, 1e-4
                    );
                }
                lrsk_opt::k_soap_ap<T><<<n_soap_units, 256, smem_bytes, stream>>>(
                    d_soap_mats.ptr, d_soap_units.ptr, d_pdesc.ptr, d_opt_sc.ptr, d_soap_ws.ptr,
                    lr * soap_mult, soap_b2, soap_eps
                );
            }
        }
        return 0;
    }

    // =========================================================================
    // Run an ENTIRE Epoch on GPU (ZERO Python, ZERO mid-epoch CPU-GPU sync!)
    // =========================================================================
    int run_epoch_impl(
        const int* perm_h, const double* dyn_gw_h, int n_steps, int batch_size,
        const double* lr_per_step, const double* we_per_step, const double* wf_per_step,
        const double* ws_per_step, const double* wei_per_step, const double* wmol_per_step,
        double net_s, double trq_s, int per_graph_f_norm, int update_ferr,
        double* out_mean_loss, double* out_mean_lf
    ) {
        d_loss_acc.zero(1, stream);
        if (out_mean_lf) d_lf_acc.zero(1, stream);
        for (int st = 0; st < n_steps; ++st) {
            BatchDesc bd{};
            bd.B = batch_size;
            int n_mol_g = 0;
            int a = 0, e1 = 0, e2 = 0, ez = 0, mm = 0;
            for (int k = 0; k < batch_size; ++k) {
                int g = perm_h[st * batch_size + k];
                bd.g_id[k] = g;
                bd.a_off[k] = a; a += h_n_atoms[g];
                bd.e1_off[k] = e1; e1 += h_n_e1[g];
                bd.e2_off[k] = e2; e2 += h_n_e2[g];
                bd.ez_off[k] = ez; ez += h_n_zbl[g];
                bd.m_off[k] = mm;
                bd.dyn_gw[k] = dyn_gw_h ? dyn_gw_h[st * batch_size + k] : 1.0;
                int nm = h_n_mol[g];
                if (nm > 0) { mm += nm; ++n_mol_g; }
            }
            bd.N = a; bd.E1 = e1; bd.E2 = e2; bd.Ez = ez; bd.M_mol = mm;
            const bool do_prof = (prof_enabled && (step_count == 7 || step_count == 8));
            cudaEvent_t ev[12];
            if (do_prof) { for (int i = 0; i < 12; ++i) cudaEventCreate(&ev[i]); }
            int rc = compute_batch_grads(
                bd, n_mol_g, we_per_step[st], wf_per_step[st], ws_per_step[st], wei_per_step[st],
                wmol_per_step[st], net_s, trq_s, 1, per_graph_f_norm, update_ferr, do_prof ? ev : nullptr
            );
            if (rc) return rc;
            rc = optimizer_step(lr_per_step[st]);
            if (rc) return rc;
            if (do_prof) {
                cudaEventRecord(ev[11], stream);
                CUDA_CHK(cudaStreamSynchronize(stream));
                const char* names[11] = {
                    "gather", "L1+NB1_f", "LR_f", "NB2_fb1", "LR_b1",
                    "NB1_b1+L1F", "loss+L1b2+df", "LR_b2", "NB2db+LRpb1", "NB1db+C1+reg", "opt"
                };
                float tot_ms = 0.0f;
                cudaEventElapsedTime(&tot_ms, ev[0], ev[11]);
                printf("  [LOUP_FT_PROFILE step=%d N=%d E1=%d E2=%d tot=%.3fms]", step_count, bd.N, bd.E1, bd.E2, tot_ms);
                for (int i = 0; i < 11; ++i) {
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, ev[i], ev[i + 1]);
                    printf(" %s=%.3f", names[i], ms);
                }
                printf("\n");
                fflush(stdout);
                for (int i = 0; i < 12; ++i) cudaEventDestroy(ev[i]);
            }
        }
        CUDA_CHK(cudaStreamSynchronize(stream));
        double tot_loss = 0.0;
        d_loss_acc.download(&tot_loss, 1);
        if (out_mean_loss) *out_mean_loss = tot_loss / std::max(1, n_steps);
        if (out_mean_lf) {
            double tot_lf = 0.0;
            d_lf_acc.download(&tot_lf, 1);
            *out_mean_lf = tot_lf / std::max(1, n_steps);
        }
        return 0;
    }

    int run_epoch(
        const int* perm_h, int n_steps, int batch_size,
        const double* lr_per_step, const double* we_per_step, const double* wf_per_step,
        const double* ws_per_step, const double* wei_per_step, const double* wmol_per_step,
        double net_s, double trq_s, double* out_mean_loss
    ) {
        return run_epoch_impl(
            perm_h, nullptr, n_steps, batch_size, lr_per_step, we_per_step, wf_per_step,
            ws_per_step, wei_per_step, wmol_per_step, net_s, trq_s, 0, 0, out_mean_loss, nullptr
        );
    }

    int run_epoch_weighted(
        const int* perm_h, const double* dyn_gw_h, int n_steps, int batch_size,
        const double* lr_per_step, const double* we_per_step, const double* wf_per_step,
        int update_ferr, double* out_mean_loss, double* out_mean_lf
    ) {
        std::vector<double> zeros(n_steps, 0.0);
        return run_epoch_impl(
            perm_h, dyn_gw_h, n_steps, batch_size, lr_per_step, we_per_step, wf_per_step,
            zeros.data(), zeros.data(), zeros.data(), 0.0, 0.0, 1, update_ferr, out_mean_loss, out_mean_lf
        );
    }

    int get_frame_err(float* out_ferr) const {
        if (ds_G <= 0 || !out_ferr) return 0;
        d_frame_err.download(out_ferr, ds_G);
        return (int)cudaGetLastError();
    }

    // Single-batch gradcheck helper (computes E, F, V, loss, and exact param grads WITHOUT stepping!)
    int gradcheck_batch(
        const int* g_ids, int B,
        double we, double wf, double ws, double wei,
        double w_mol, double net_s, double trq_s, int do_regs,
        T* E_out, T* F_out, T* V_out, double* loss_out,
        const double* dyn_gw_h = nullptr, int per_graph_f_norm = 0
    ) {
        d_loss_acc.zero(1, stream);
        BatchDesc bd{};
        bd.B = B;
        int n_mol_g = 0;
        int a = 0, e1 = 0, e2 = 0, ez = 0, mm = 0;
        for (int k = 0; k < B; ++k) {
            int g = g_ids[k];
            bd.g_id[k] = g;
            bd.a_off[k] = a; a += h_n_atoms[g];
            bd.e1_off[k] = e1; e1 += h_n_e1[g];
            bd.e2_off[k] = e2; e2 += h_n_e2[g];
            bd.ez_off[k] = ez; ez += h_n_zbl[g];
            bd.m_off[k] = mm;
            bd.dyn_gw[k] = dyn_gw_h ? dyn_gw_h[k] : 1.0;
            int nm = h_n_mol[g];
            if (nm > 0) { mm += nm; ++n_mol_g; }
        }
        bd.N = a; bd.E1 = e1; bd.E2 = e2; bd.Ez = ez; bd.M_mol = mm;
        int rc = compute_batch_grads(bd, n_mol_g, we, wf, ws, wei, w_mol, net_s, trq_s, do_regs, per_graph_f_norm, 0, nullptr);
        if (rc) return rc;
        CUDA_CHK(cudaStreamSynchronize(stream));
        if (E_out) s_E_pred.download(E_out, B);
        if (F_out) s_F_tot.download(F_out, (size_t)bd.N * 3);
        if (V_out && ws > 0.0 && ds_has_stress) s_V_tot.download(V_out, (size_t)B * 9);
        if (loss_out) d_loss_acc.download(loss_out, 1);
        return 0;
    }
};

} // namespace lrsk_ft

// =============================================================================
// C API Exports (float32 + float64)
// =============================================================================
using namespace lrsk_ft;

#define DEFINE_FT_API(SFX, T) \
extern "C" { \
void* lrsk_ft_create_##SFX( \
    const ModelDims* dims, const Nu2Path* nu2, const Nu3Path* nu3, \
    const CgEntry<T>* cg, int n_cg, const int* lr_sym_mask, \
    const T* mu_x, const T* std_x, const T* std_b1, const T* std_b2, int soap_max_dim \
) { \
    auto* tr = new LrskFullTrainer<T>(); \
    if (tr->init(*dims, nu2, nu3, cg, n_cg, lr_sym_mask, mu_x, std_x, std_b1, std_b2, soap_max_dim) != 0) { \
        delete tr; return nullptr; \
    } \
    return tr; \
} \
void lrsk_ft_destroy_##SFX(void* h) { delete reinterpret_cast<LrskFullTrainer<T>*>(h); } \
int lrsk_ft_num_params_##SFX(void* h) { return (int)reinterpret_cast<LrskFullTrainer<T>*>(h)->slots.size(); } \
long lrsk_ft_param_len_##SFX(void* h, int pid) { return reinterpret_cast<LrskFullTrainer<T>*>(h)->slots[pid].n; } \
int lrsk_ft_set_param_##SFX(void* h, int pid, const T* src, int init_ema) { \
    auto* tr = reinterpret_cast<LrskFullTrainer<T>*>(h); \
    const auto& s = tr->slots[pid]; \
    cudaMemcpy(tr->d_p_flat.ptr + s.off, src, s.n * sizeof(T), cudaMemcpyHostToDevice); \
    if (init_ema) cudaMemcpy(tr->d_ema_flat.ptr + s.off, src, s.n * sizeof(T), cudaMemcpyHostToDevice); \
    return (int)cudaGetLastError(); \
} \
int lrsk_ft_get_param_##SFX(void* h, int pid, T* dst, int from_ema) { \
    auto* tr = reinterpret_cast<LrskFullTrainer<T>*>(h); \
    const auto& s = tr->slots[pid]; \
    const T* src = (from_ema ? tr->d_ema_flat.ptr : tr->d_p_flat.ptr) + s.off; \
    cudaMemcpy(dst, src, s.n * sizeof(T), cudaMemcpyDeviceToHost); \
    return (int)cudaGetLastError(); \
} \
int lrsk_ft_get_grad_##SFX(void* h, int pid, T* dst) { \
    auto* tr = reinterpret_cast<LrskFullTrainer<T>*>(h); \
    const auto& s = tr->slots[pid]; \
    cudaMemcpy(dst, tr->d_g_flat.ptr + s.off, s.n * sizeof(T), cudaMemcpyDeviceToHost); \
    return (int)cudaGetLastError(); \
} \
int lrsk_ft_set_opt_##SFX( \
    void* h, double soap_b1, double soap_b2, int soap_every, double soap_wd, double soap_mult, \
    double adam_b1, double adam_b2, double ema_dmax, double grad_clip, \
    double w_lr_reg, double rad_smooth, double wd_ro, const T* sp_w, int has_sp_w \
) { \
    auto* tr = reinterpret_cast<LrskFullTrainer<T>*>(h); \
    tr->soap_b1 = soap_b1; tr->soap_b2 = soap_b2; tr->soap_every = soap_every; \
    tr->soap_wd = soap_wd; tr->soap_mult = soap_mult; \
    tr->adam_b1 = adam_b1; tr->adam_b2 = adam_b2; tr->ema_dmax = ema_dmax; tr->grad_clip = grad_clip; \
    tr->w_lr_reg = w_lr_reg; tr->rad_smooth = rad_smooth; tr->wd_ro = wd_ro; \
    tr->has_sp_w = has_sp_w; \
    if (has_sp_w && sp_w) tr->d_sp_w.upload(sp_w, tr->D.S); \
    return 0; \
} \
int lrsk_ft_upload_dataset_##SFX( \
    void* h, int G, int tot_N, int tot_E1, int tot_E2, int tot_Ez, int split_r2, \
    int has_stress, int has_ei, int has_mol, \
    const int* atom_off, const int* e1_off, const int* e2_off, const int* ez_off, \
    const int* n_atoms, const int* n_e1, const int* n_e2, const int* n_zbl, const int* n_mol, \
    const int* z_idx, const int* deg1, const int* deg2, const int* snd1, const int* snd2, \
    const T* vec1, const T* vec2, const T* G1, const T* f_ref, \
    const T* E_ref, const T* vol, const T* S_ref, const T* gw, \
    const T* ezbl, const T* sw, const T* dsw, \
    const int* zsnd, const int* zrcv, const T* zgv, const T* zvec, \
    const T* ei_ref, const int* mol_id, const T* mol_arms \
) { \
    auto* tr = reinterpret_cast<LrskFullTrainer<T>*>(h); \
    tr->ds_G = G; tr->ds_split_r2 = split_r2; tr->ds_has_stress = has_stress; \
    tr->ds_has_ei = has_ei; tr->ds_has_mol = has_mol; tr->ds_any_zbl = (tot_Ez > 0) ? 1 : 0; \
    tr->h_n_atoms.assign(n_atoms, n_atoms + G); \
    tr->h_n_e1.assign(n_e1, n_e1 + G); \
    tr->h_n_e2.assign(n_e2, n_e2 + G); \
    tr->h_n_zbl.assign(n_zbl, n_zbl + G); \
    tr->h_n_mol.assign(n_mol, n_mol + G); \
    int g1s = tr->D.S * tr->D.K1 * tr->D.n_sh; \
    tr->d_ds_atom_off.upload(atom_off, G + 1); \
    tr->d_ds_e1_off.upload(e1_off, G + 1); \
    tr->d_ds_e2_off.upload(e2_off, G + 1); \
    tr->d_ds_ez_off.upload(ez_off, G + 1); \
    tr->d_ds_n_atoms.upload(n_atoms, G); \
    tr->d_ds_n_mol.upload(n_mol, G); \
    tr->d_ds_z.upload(z_idx, tot_N); \
    tr->d_ds_deg1.upload(deg1, tot_N); \
    if (split_r2) tr->d_ds_deg2.upload(deg2, tot_N); \
    tr->d_ds_snd1.upload(snd1, tot_E1); \
    tr->d_ds_vec1.upload(vec1, (size_t)tot_E1 * 3); \
    if (split_r2) { \
        tr->d_ds_snd2.upload(snd2, tot_E2); \
        tr->d_ds_vec2.upload(vec2, (size_t)tot_E2 * 3); \
    } \
    tr->d_ds_G1.upload(G1, (size_t)tot_N * g1s); \
    tr->d_ds_f.upload(f_ref, (size_t)tot_N * 3); \
    tr->d_ds_E.upload(E_ref, G); \
    tr->d_ds_vol.upload(vol, G); \
    if (has_stress && S_ref) tr->d_ds_S.upload(S_ref, (size_t)G * 9); \
    if (gw) tr->d_ds_gw.upload(gw, G); \
    tr->d_ds_ezbl.upload(ezbl, tot_N); \
    tr->d_ds_sw.upload(sw, tot_N); \
    tr->d_ds_dsw.upload(dsw, tot_N); \
    if (tot_Ez > 0) { \
        tr->d_ds_zsnd.upload(zsnd, tot_Ez); \
        tr->d_ds_zrcv.upload(zrcv, tot_Ez); \
        tr->d_ds_zgv.upload(zgv, (size_t)tot_Ez * 3); \
        tr->d_ds_zvec.upload(zvec, (size_t)tot_Ez * 3); \
    } \
    if (has_ei && ei_ref) tr->d_ds_ei.upload(ei_ref, tot_N); \
    if (has_mol && mol_id && mol_arms) { \
        tr->d_ds_mol_id.upload(mol_id, tot_N); \
        tr->d_ds_mol_arms.upload(mol_arms, (size_t)tot_N * 3); \
    } \
    std::vector<float> fe_init(G, 1.0f); \
    std::vector<int> fs_init(G, 0); \
    tr->d_frame_err.upload(fe_init.data(), G); \
    tr->d_frame_seen.upload(fs_init.data(), G); \
    return (int)cudaGetLastError(); \
} \
int lrsk_ft_run_epoch_##SFX( \
    void* h, const int* perm, int n_steps, int batch_size, \
    const double* lr_s, const double* we_s, const double* wf_s, \
    const double* ws_s, const double* wei_s, const double* wmol_s, \
    double net_s, double trq_s, double* out_mean_loss \
) { \
    return reinterpret_cast<LrskFullTrainer<T>*>(h)->run_epoch( \
        perm, n_steps, batch_size, lr_s, we_s, wf_s, ws_s, wei_s, wmol_s, net_s, trq_s, out_mean_loss \
    ); \
} \
int lrsk_ft_run_epoch_weighted_##SFX( \
    void* h, const int* perm, const double* dyn_gw, int n_steps, int batch_size, \
    const double* lr_s, const double* we_s, const double* wf_s, \
    int update_ferr, double* out_mean_loss, double* out_mean_lf \
) { \
    return reinterpret_cast<LrskFullTrainer<T>*>(h)->run_epoch_weighted( \
        perm, dyn_gw, n_steps, batch_size, lr_s, we_s, wf_s, update_ferr, out_mean_loss, out_mean_lf \
    ); \
} \
int lrsk_ft_get_frame_err_##SFX(void* h, float* out_ferr) { \
    return reinterpret_cast<LrskFullTrainer<T>*>(h)->get_frame_err(out_ferr); \
} \
int lrsk_ft_gradcheck_##SFX( \
    void* h, const int* g_ids, int B, \
    double we, double wf, double ws, double wei, \
    double w_mol, double net_s, double trq_s, int do_regs, \
    T* E_out, T* F_out, T* V_out, double* loss_out \
) { \
    return reinterpret_cast<LrskFullTrainer<T>*>(h)->gradcheck_batch( \
        g_ids, B, we, wf, ws, wei, w_mol, net_s, trq_s, do_regs, E_out, F_out, V_out, loss_out \
    ); \
} \
int lrsk_ft_gradcheck_weighted_##SFX( \
    void* h, const int* g_ids, const double* dyn_gw, int B, \
    double we, double wf, int do_regs, \
    T* E_out, T* F_out, double* loss_out \
) { \
    return reinterpret_cast<LrskFullTrainer<T>*>(h)->gradcheck_batch( \
        g_ids, B, we, wf, 0.0, 0.0, 0.0, 0.0, 0.0, do_regs, E_out, F_out, nullptr, loss_out, dyn_gw, 1 \
    ); \
} \
}

DEFINE_FT_API(f32, float)
DEFINE_FT_API(f64, double)
