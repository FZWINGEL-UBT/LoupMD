#pragma once
// The generic MACE path: any bundle with arch=generic (python/export_generic.py) — MACE-OMAT-0 small/medium,
// MACE-OFF, MACE-MH-1 and MACE-POLAR-1's local part. Layer by layer: linear_up GEMMs, the fused conv tensor
// product with radial splines (generated_generic.cuh), message linears, density / avg-neighbour / gated
// normalisations, the factored symmetric contraction, skip connections and readouts; then the same in reverse for
// forces and stress. engine.cu owns the neighbour list, the MD loop and the GEMM autotuning; it calls in here.
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "kernels_common.cuh"

namespace loup {

class Bundle;

// row-major C[M, N] = alpha * A[M, K] @ op(B) + beta * C, op(B) = B [K, N] or B^T for B [N, K]
using GemmFn = std::function<void(bool transB, int M, int N, int K, float alpha, const float* A, int lda,
                                  const float* B, int ldb, float beta, float* C, int ldc)>;

// One degree l of the nonlinear (kind 4) block's gate in the epilogue of an own-SGEMM projection (sgemm.cu):
// - forward (gemm_gate_fwd: lin_l = A @ B for a degree l >= 1 without residual): writes gate_fwd_kernel's gout_l =
//   cs_g sigmoid(pre0[l C + c]) * (lin_l * s) to `out` instead of storing lin_l; reads D, lin0, res0 / res0_sp;
// - backward (gemm_gate_bwd: dgout_l = A @ W_2[l]^T): what gate_bwd_kernel computes for degree l's columns from
//   dgout_l, without writing dgout_l. lin_l / res_l / dp_l / dl_l are degree l's arrays (lin0 / res0 / dp0 / dl0 for
//   l = 0); res_l, dp0 and dp_l may be null. vform: lin_l is the forward's saved gout_l (a degree whose lin_l the
//   forward never stored; no residual), the reverse taken through it.
struct GateArgs {
    int natoms = 0, C = 0;
    const float* D = nullptr;
    float alpha = 1.0f, beta = 0.0f, cs = 1.0f, cg = 1.0f;
    const float* lin0 = nullptr;
    const float* lin_l = nullptr;
    const float* res0 = nullptr;
    const int* res0_sp = nullptr;
    const float* res_l = nullptr;
    float* dp0 = nullptr;
    float* dp_l = nullptr;
    float* dl0 = nullptr;
    float* dl_l = nullptr;
    float* dD = nullptr;  // += the degree's terms of dE/dD, atomically
    float* out = nullptr;
    int vform = 0;
    // hgemm2_f32a_rows (list 13): rows of rdeg per atom; the plain product's row r scaled by D[r / rdeg] beta + alpha.
    // With c2 (LOUP_RES_BWD 3, n1 % 64 == 0): columns n1 .. N - 1 go to c2 (row length N - n1) at os2, scaled so;
    // columns below n1 to C (row length n1) at oscale, unscaled.
    int rdeg = 0;
    float* c2 = nullptr;
    int n1 = 0;
    float os2 = 1.0f;
    // rform (list 27 item 3, LOUP_LIN_HALF_PARTS bit 64; last, so the other fields keep their offsets in the kernels'
    // parameters): a degree l in 1 .. 2 with a residual whose forward gate ran in its W_res projection's epilogue
    // (hgemm2_f32a_gate with lin_l), so res_l is never stored: res_l holds the saved gout_l = g (lin_l s + res_l), the
    // gate logit's reverse taken through it (d gp = (1 - sg) sum dgout gout_l)
    int rform = 0;
};
// Runs the projection fused with degree l's gate (M = natoms (2l + 1), N = C) when the GEMM's plan and shape allow;
// false: nothing launched (the caller runs the plain GEMM and the gate kernel for l). Backward: dgout_l = A[M, K] @
// B[N, K]^T; forward: lin_l = A[M, K] @ B[K, N]. A null: only reports whether it would run.
using GemmGateFn = std::function<bool(int l, int M, int N, int K, const float* A, const float* B, const GateArgs& g)>;
// LOUP_W2T_HALF (hgemm2.cu): dgout_l = A @ W_2[l]^T from the two binary16 panels of A (the product kernel's _h2
// store) and of the weight (made at load), FP32 accumulation, degree l's gate backward in the epilogue, on the
// current GEMM lane (M = natoms (2l + 1), N = C); oscale undoes the panels' scales.
using GemmH2GateFn = std::function<void(int l, int M, int N, int K, const void* A, const void* B, float oscale,
                                        const GateArgs& g)>;
void hgemm2_init();
int hgemm2_weight_exponent(const float* w, int64_t n);
void hgemm2_split(int64_t n, const float* w, int e, void* out, cudaStream_t stream);
void hgemm2_gate(int l, int M, int N, int K, const void* A, const void* B, float oscale, float* C, const GateArgs& g,
                 cudaStream_t stream);
// LOUP_W2F_HALF (hgemm2.cu, hgemm2f_kernel): C = A B^T oscale with A in FP32, split in the kernel at scale sA; B the
// panels of the transposed weight (hgemm2_split_t).
// beta != 0: C = beta C + A B^T oscale. N % 16 == 0 and K % 16 == 0 with B's panels from hgemm2_split_pad (padded to
// [2][Np][Kp]); N % 64 == 0 and K % 32 == 0 need no padding (hgemm2_split, hgemm2_split_t give the same layout) and,
// with beta 0, run the kernel's instance without tails.
// ws (ws_bytes): a workspace on the launch's stream for list 22's split-K, which runs where M is small and K >= 1,024
// (hgemm2_split_k); null: never split.
void hgemm2_f32a(int M, int N, int K, const float* A, float sA, const void* B, float oscale, float* C,
                 cudaStream_t stream, float beta = 0.0f, float* ws = nullptr, size_t ws_bytes = 0);
void hgemm2_f32a_variant(int v, int M, int N, int K, const float* A, float sA, const void* B, float oscale, float* C,
                         cudaStream_t stream, float beta = 0.0f, float* ws = nullptr, size_t ws_bytes = 0);
void hgemm2_split_t(int R, int Cc, const float* w, int e, void* out, cudaStream_t stream);
int64_t hgemm2_pad_elems(int Nr, int Kc);  // halves of one padded panel (two per weight)
void hgemm2_split_pad(int Nr, int Kc, const float* w, bool trans, int e, void* out, cudaStream_t stream);
void hgemm2_row_init();
void hgemm2_row_exp(int M, int K, const float* A, int8_t* eA, cudaStream_t stream);
void hgemm2_pack_b_col(int Nr, int Kc, const float* w, bool trans, void* out_panel, float* out_inv_sB,
                       cudaStream_t stream);
void hgemm2_f32a_row(int M, int N, int K, const float* A, const void* B_panel, const float* inv_sB,
                     float* C, float beta, void* ws, size_t ws_bytes, cudaStream_t stream,
                     const int8_t* eA_cached = nullptr);
// Fused w=64 readout epilogues for LR-SK:
//   fwd: pre_out = A @ B + bias; h_out = silu(pre_out); eA_out = row_exp(h_out)
//   bwd: dh = A @ B; dpre_out = dh * silu'(pre_in); eA_out = row_exp(dpre_out)
void hgemm2_f32a_row_silu_fwd64(int M, int K, const float* A, const void* B_panel, const float* inv_sB,
                                bool per_sp_bias, const int* species, const float* bias,
                                float* pre_out, float* h_out, int8_t* eA_out,
                                const int8_t* eA_in, cudaStream_t stream);
void hgemm2_f32a_row_silu_bwd64(int M, int K, const float* A, const void* B_panel, const float* inv_sB,
                                const float* pre_in, float* dpre_out, int8_t* eA_out,
                                const int8_t* eA_in, cudaStream_t stream);
// Fused 2-layer w=64 site MLP forward & backward for LR-SK (32 KB SMEM -> 2-3 CTAs/SM, zero h0/dpre0 DRAM!):
//   fwd: pre0 = x @ W0 + alpha_Z[sp]; h0 = silu(pre0); pre1 = h0 @ W1 (pre-bias, consumed by lrsk_heads_fwd_silu)
//   bwd: dh0 = dpre1 @ W1^T; dpre0 = dh0 * silu'(pre0); dx = dpre0 @ W0^T
void hgemm2_lrsk_ro2_fwd64(int M, int K, const int* species, const float* x, const int8_t* eA_x,
                           const float* alpha_Z, const void* W0_B, const float* W0_inv_sB,
                           const void* W1_B, const float* W1_inv_sB,
                           float* pre0_out, float* pre1_out, cudaStream_t stream);
void hgemm2_lrsk_ro2_bwd64(int M, int K, const float* dpre1, const int8_t* eA_dpre1,
                           const float* pre0, const void* W1_BT, const float* W1_inv_sBT,
                           const void* W0_BT, const float* W0_inv_sBT,
                           float* dx_out, cudaStream_t stream);
void hgemm2_f32a_row_gate_bwd(int l, int M, int N, int K, const float* A, const void* B_panel, const float* inv_sB,
                              const GateArgs& g, void* ws, cudaStream_t stream, const int8_t* eA_cached = nullptr);
void hgemm2_f32a_row_gate_fwd(int l, int M, int N, int K, const float* A, const void* B_panel, const float* inv_sB,
                              const GateArgs& g, void* ws, cudaStream_t stream, const int8_t* eA_cached = nullptr);
// Fused ILMlip readout tail (depth==2, width==128, need_grad==true):
// pre0[N,128] -> +alpha_Z -> silu -> @W1 -> +b1 -> silu -> .u_Z (+ZBL switch) -> silu_bwd -> @W1^T -> silu_bwd -> dpre0[N,128]
void hgemm2_ilmlip_ro_tail_w128(
    int N, bool zbl_sw, float inv_zc, const int* species,
    const float* pre0, const float* alpha_Z, const float* b1, const float* u_Z,
    const void* W1_B, const float* W1_inv_sB, const void* W1_BT, const float* W1_inv_sBT,
    const double* e_zbl, double* energy, double* atom_energy, float* w_zbl_out, float* dpre0,
    int8_t* eA_dpre0, cudaStream_t stream);
void hgemm2_ilmlip_ro_tail_w64(
    int N, bool zbl_sw, float inv_zc, const int* species,
    const float* pre0, const float* alpha_Z, const float* b1, const float* u_Z,
    const void* W1_B, const float* W1_inv_sB, const void* W1_BT, const float* W1_inv_sBT,
    const double* e_zbl, double* energy, double* atom_energy, float* w_zbl_out, float* dpre0,
    int8_t* eA_dpre0, cudaStream_t stream);
void hgemm2_ilmlip_ro1_fused(
    int N, int K, int rw, bool zbl_sw, float inv_zc, const int* species,
    const float* x, const int8_t* eA_x, const float* alpha_Z, const float* u_Z,
    const void* W0_B, const float* W0_inv_sB, const void* W0_BT, const float* W0_inv_sBT,
    const double* e_zbl, double* energy, double* atom_energy, float* w_zbl_out, float* dx,
    void* ws, cudaStream_t stream);
void hgemm2_ilmlip_ro1_fwd_tail(
    int N, int K, int rw, bool zbl_sw, float inv_zc, const int* species,
    const float* x, const int8_t* eA_x, const float* alpha_Z, const float* u_Z,
    const void* W0_B, const float* W0_inv_sB,
    const double* e_zbl, double* energy, double* atom_energy, float* w_zbl_out,
    float* dpre0, int8_t* eA_dpre0, void* ws, cudaStream_t stream);
void hgemm2_ilmlip_ro2_fused_w64(
    int N, int K, bool zbl_sw, float inv_zc, const int* species,
    const float* x, const int8_t* eA_x, const float* alpha_Z, const float* b1, const float* u_Z,
    const void* W0_B, const float* W0_inv_sB, const void* W0_BT, const float* W0_inv_sBT,
    const void* W1_B, const float* W1_inv_sB, const void* W1_BT, const float* W1_inv_sBT,
    const double* e_zbl, double* energy, double* atom_energy, float* w_zbl_out, float* dx,
    void* ws, cudaStream_t stream);
void hgemm2_ilmlip_ro2_fwd_tail_w64(
    int N, int K, bool zbl_sw, float inv_zc, const int* species,
    const float* x, const int8_t* eA_x, const float* alpha_Z, const float* b1, const float* u_Z,
    const void* W0_B, const float* W0_inv_sB,
    const void* W1_B, const float* W1_inv_sB, const void* W1_BT, const float* W1_inv_sBT,
    const double* e_zbl, double* energy, double* atom_energy, float* w_zbl_out,
    float* dpre0, int8_t* eA_dpre0, void* ws, cudaStream_t stream);
using GemmH2RowFn = std::function<void(int M, int N, int K, const float* A, const void* B_panel,
                                       const float* inv_sB, float* C, float beta, const int8_t* eA_cached)>;
using GemmH2RowGateFn = std::function<void(int l, int M, int N, int K, const float* A, const void* B_panel,
                                           const float* inv_sB, const GateArgs& g, const int8_t* eA_cached)>;
using GemmH2FwdFn = std::function<void(int M, int N, int K, const float* A, float sA, const void* B, float oscale,
                                       float* C, float beta)>;
// The same product (beta 0) with row r scaled by rs.D[r / rs.rdeg] rs.beta + rs.alpha (list 13, LOUP_RES_BWD 2: the
// backward dx_l = dpre_l W_res[l]^T from dlin_l, dpre = dlin (D beta + alpha) per atom).
void hgemm2_f32a_rows(int M, int N, int K, const float* A, float sA, const void* B, float oscale, float* C,
                      const GateArgs& rs, cudaStream_t stream, float* ws = nullptr, size_t ws_bytes = 0);
using GemmH2RowsFn = std::function<void(int M, int N, int K, const float* A, float sA, const void* B, float oscale,
                                        float* C, const GateArgs& rs)>;
// The same product for a residual-free degree l >= 1 (M = g.natoms (2l + 1), N = g.C) with gate_fwd's gated store:
// gout_l = g (lin_l s) to g.out instead of lin_l. With g.lin_l (list 27 item 3, l = 1, 2 only): the product is degree
// l's residual res_l, and the store is gout_l = g (lin_l s + res_l), lin_l read from g.lin_l; res_l is not stored.
void hgemm2_f32a_gate(int l, int M, int N, int K, const float* A, float sA, const void* B, float oscale,
                      const GateArgs& g, cudaStream_t stream);
// LOUP_LIN_HALF's projections keep the forward gate fusion (LOUP_GATE_FUSE_FWD; `later`: a later layer's, under
// LOUP_GATE_FWD_L1 and its row rule). False: nothing launched. A null: only reports whether it would run.
using GemmH2GateFwdFn = std::function<bool(int l, int M, int N, int K, const float* A, float sA, const void* B,
                                           float oscale, const GateArgs& g, bool later)>;

struct GenericPassInputs {
    cudaStream_t stream;
    int N;                     // receivers: the atoms of a periodic system, the local atoms of a domain
    int n_ghost = 0;           // a domain's ghost atoms, rows N .. N + n_ghost - 1: they only send. Their rows of a
                               // layer's input and of its gradient exist; nothing else is computed for them
    const int* E_dev;          // device: number of active edges (act_row_ptr[N])
    int E_cap;                 // capacity of the edge arrays
    const int* act_row_ptr;    // [N + 1], receiver rows
    const int* act_row_end = nullptr;  // [N], exclusive end of receiver row i; nullptr => act_row_ptr + 1
    const int* act_row_end2 = nullptr; // [N], exclusive end of r < r_max2 edges (arch=ilmlip dual-cutoff); nullptr => act_row_end
    const int* act_senders;    // [E]
    const int* act_receivers;  // [E]
    const float4* act_vec_len; // [E] (pos[rcv] - pos[snd] + shift, |v|)
    const int* act_slab;       // [E] prelocated (pair * I + iv) for preloc_grid()/preloc_pair_sym(), or nullptr
    const float2* act_t_sx;    // [E] prelocated (t, +/- sx_inv)
    const int* species;        // [N], atoms grouped by species
    const std::vector<int>* species_start;
    const std::vector<int>* species_count;
    double* energy;            // [1] accumulated (zeroed here)
    double* forces;            // [N, 3] (zeroed here)
    double* virial;            // [6] (zeroed here)
    const double* pos64;       // [N, 3] positions (device order; the long-range k-space terms read them)
    int compute_stress;
    int energy_only = 0;       // forward of every layer and the ZBL energy only: no forces, no stress
    bool defer_f32_flush = false;  // when true, arch=ilmlip skips ilmlip_add_forces_f32_kernel (fused into MD 2nd half!)
    double* atom_energy = nullptr;  // energy_only (S6-T08): also each row's readout and ZBL energy [N], added
                                    // to (the caller zeroes it); the fused product + readout kernel is not used
    const double* cell_vol_dev = nullptr;  // live device cell volume (variable-cell NPT), or nullptr
    double cell_vol_host = 0.0;            // host cell volume (used when cell_vol_dev == nullptr)
    const double* cell_h_dev = nullptr;    // live device 3x3 cell (variable-cell NPT), or nullptr
    double cell_h_host[9] = {0};           // host 3x3 row-major cell (used when cell_h_dev == nullptr)
    // Which part of the pass record() runs (PLAN.md 4.4). SweepAll is the whole pass. The others are one sweep of a
    // domain with ghost atoms; the caller carries the layer outputs and their gradients from sweep to sweep:
    //   SweepForward t (t < L - 1): needs layer t - 1's output rows (receivers and ghosts) in layer_out(t - 1, l);
    //                               leaves layer t's output for the receivers in layer_out(t, l); adds the
    //                               readout's energy.
    //   SweepLast:                  layer L - 1 forward and backward; needs layer L - 2's output as above; leaves
    //                               the gradient of it (receivers and ghosts) in layer_out_grad(L - 2, l), and
    //                               this sweep's forces.
    //   SweepBackward t:            replays layer t forward and runs its backward; needs layer t - 1's output and,
    //                               in adjoint_in[l], the receivers' rows of the gradient of layer t's output that
    //                               the later sweeps left (summed over every domain that holds the atom); leaves
    //                               the gradient of layer t - 1's output and this sweep's forces. ZBL comes with
    //                               t = 0.
    // A sweep writes the forces of its own edges only (receiver and sender rows), zeroing nothing of `forces`,
    // `virial` and `energy`: the caller zeroes them once.
    enum Sweep { SweepAll = 0, SweepForward = 1, SweepLast = 2, SweepBackward = 3 };
    int sweep = SweepAll;
    int sweep_layer = 0;
    const float* adjoint_in[4] = {nullptr, nullptr, nullptr, nullptr};
    // O31 (tiled): a buffer of GenericModel::sweep_keep_floats(N) floats the caller keeps per domain for layer 0's
    // forward results that its backward reads. SweepForward 0 writes them there; SweepBackward 0 reads them there
    // and does not replay layer 0's forward. Only where sweep_keep_layer0(); nullptr: the replay.
    float* keep0 = nullptr;
    GemmFn gemm;
    std::function<void()> gemm_fork;
    std::function<void(int)> gemm_lane;
    std::function<void()> gemm_join;
    GemmGateFn gemm_gate_bwd;  // empty: never fused (LOUP_GATE_FUSE=0)
    GemmGateFn gemm_gate_fwd;  // empty: never fused (LOUP_GATE_FUSE_FWD=0)
    GemmGateFn gemm_gate_fwd_l1;  // empty: later layers' forward gate off (LOUP_GATE_FWD_L1=0)
    GemmH2GateFn gemm_h2_gate; // empty: the FP32 W_2^T path (the model also needs LOUP_W2T_HALF at load)
    GemmH2FwdFn gemm_h2_fwd;
    GemmH2RowFn gemm_h2_row;
    GemmH2RowGateFn gemm_h2_row_gate_bwd;
    GemmH2RowGateFn gemm_h2_row_gate_fwd;   // empty: the FP32 forward W_2 path (the model also needs LOUP_W2F_HALF at load)
    GemmH2RowsFn gemm_h2_rows; // empty: W_res^T from dpre (LOUP_RES_BWD 1)
    GemmH2GateFwdFn gemm_h2_gate_fwd;  // empty: LOUP_GATE_FUSE_FWD=0
};

struct GenericLayer;

class PolarLR;
class IlmlipModel;

class GenericModel {
public:
    GenericModel() = default;
    ~GenericModel();
    GenericModel(const GenericModel&) = delete;
    GenericModel& operator=(const GenericModel&) = delete;

    void load(const Bundle& b);
    int S() const { return S_; }
    float r_max() const { return r_max_; }
    const std::vector<int>& atomic_numbers() const { return Z_; }
    const std::vector<double>& e0() const { return e0_; }
    int n_layers() const { return n_layers_; }
    // Bytes of the per-system buffers (alloc_system) and of the edge buffers (ensure_edges), for loup_plan_memory,
    // the long-range part's (polar.cu) included.
    size_t system_bytes() const;
    // The same with every layer's buffers its own: what a whole system of as many rows allocates where this one
    // shares them (share_layer_buffers); equal to system_bytes() otherwise.
    size_t system_bytes_unshared() const { return sys_bytes_unshared_; }
    // The largest single buffer of the system: the first pass times every GEMM shape into a scratch copy of its
    // output, so a system's first call needs that much more for a moment (the engine's memory plans).
    size_t largest_buffer_bytes() const { return sys_max_alloc_; }
    size_t edge_bytes() const { return edge_bytes_; }
    // For a sweep driver: layer t's output rows of degree l ([rows, 2l + 1, out_width(t)]; nullptr when the model
    // does not form them) and the buffer the next layer's sweep leaves their gradient in.
    float* layer_out(int t, int l) const;
    float* layer_out_grad(int t, int l) const;
    int out_width(int t) const;
    int out_lmax(int t) const;
    // For a sweep driver: what layer t's sweeps hand to layer t + 1's, entries i < hand_count(t): rows of
    // hand_width(t, i) floats in hand(t, i), their gradient in hand_grad(t, i). Without the fold these are layer t's
    // output rows (layer_out(t, l), i = l <= out_lmax(t), (2l + 1) out_width(t) floats); with the fold in sweeps (list
    // 20: two-layer models) layer 1's x[l] ((2l + 1) C_x floats, l <= layer 0's lmax_out) and, last, the last layer's
    // readout hidden hid (ro_H floats), of which only the receivers' rows count (hand_locals). Set by alloc_system.
    int hand_count(int t) const { return (int)hand_[(size_t)t].size(); }
    float* hand(int t, int i) const { return hand_[(size_t)t][(size_t)i].p; }
    float* hand_grad(int t, int i) const { return hand_[(size_t)t][(size_t)i].g; }
    int hand_width(int t, int i) const { return hand_[(size_t)t][(size_t)i].w; }
    bool hand_locals(int t, int i) const { return hand_[(size_t)t][(size_t)i].locals; }
    // Sweeps need every layer's output to exist (not with the long range) or, with the skip-chain fold, the fold's
    // hand-over (use_chain_fold(on, true)).
    bool sweeps_available() const { return lr_ == nullptr && ilmlip_ == nullptr && (!chain_fold_ || fold_sweeps_); }
    // Whether the model can sweep once the fold is off (use_chain_fold(false)): every model without long range.
    bool sweeps_possible() const { return lr_ == nullptr && ilmlip_ == nullptr; }
    // The skip-chain fold (S57, S73) of the systems allocated from now on: on only where setup_chain_fold built it.
    // The fold's composed weights sit beside the plain ones, so a handle switches per system: whole systems fold,
    // tile domains (which sweep) only with `sweeps` (list 20: two-layer models, whose sweeps then hand on the fold's
    // targets, hand_count). Call with no system allocated (before alloc_system).
    void use_chain_fold(bool on, bool sweeps = false);
    bool chain_fold_built() const { return chain_fold_built_; }
    bool chain_fold_on() const { return chain_fold_; }
    // list 20 (the engine's memory plan): for an unfolded two-layer system whose model folds, the bytes its system
    // would differ by with the fold in its sweeps (shared: with its layers' arena as allocated; else the count of
    // system_bytes_unshared), and the widths of the fold's hand-over rows (hand_width's, by entry); 0 and empty for
    // any other system
    double fold_delta_bytes(bool shared) const { return fold_delta_[shared ? 1 : 0]; }
    const std::vector<int>& fold_hand_widths() const { return fold_hand_w_; }
    // LOUP_W2T_HALF at load (hgemm2.cu; default on compute capability 12.0, 0 / 1 force it): the kind-4 layers whose
    // product has the library's own _h2 kernels (find_gprod_h2) store dmsgf as two binary16 panels and run their W_2^T
    // GEMMs as FP16 tensor-core products with the gate in the epilogue. Not with the long range, compact layers or
    // C_msg % 64 != 0. True while
    // some layer runs it; drop_w2t_half() turns it off for good (the engine, after a non-finite result: an element
    // overflowed binary16) and returns whether it was on (captured graphs must then be re-recorded).
    bool w2t_half() const { return w2t_half_; }
    bool drop_w2t_half();
    // LOUP_DEBUG_W2T_HALF=1 (the engine, after an evaluate): each such layer's largest |dmsgf| over N atoms against
    // binary16's range at loup_store_h2's scale, and the non-finite panel elements, on stderr. Synchronises the device.
    void debug_w2t_half(int N) const;
    // LOUP_W2F_HALF at load (hgemm2.cu, hgemm2_f32a; default as LOUP_W2T_HALF): the kind-4 layers' forward W_2 GEMMs
    // from FP32 gout split in the kernel at a fixed scale (setup_w2f_half). drop_w2t_half() drops it too.
    bool w2f_half() const { return w2f_half_; }
    // LOUP_DEBUG_GOUT=1 (the engine, after an evaluate): each kind-4 layer's largest |gout_l| over N atoms on stderr.
    void debug_gout(int N) const;
    // LOUP_LIN_HALF at load (setup_lin_half; default as LOUP_W2T_HALF): the kind-4 conv layers' plain projections on the
    // two-half product. drop_w2t_half() drops it too. LOUP_DEBUG_LIN_AMAX=1 (the engine, after an evaluate): the largest
    // |A| each such projection has read since load, on stderr.
    bool lin_half() const { return lin_half_; }
    void debug_lin_amax() const;
    // The layers' own buffers of the systems allocated from now on in one shared arena (tile-shared-layers): a sweep
    // runs one layer and needs nothing of another layer but its output and that output's gradient, which stay
    // buffers of their own, so the arena is as large as the largest layer instead of all of them. Such a system runs
    // sweeps and energy-only passes, not the whole pass (its backward of a layer would read what a later layer's
    // forward overwrote). Only without the fold and the long range, from two layers; LOUP_TILE_SHARE_LAYERS=0 keeps
    // every buffer its own. Call with no system allocated (before alloc_system).
    void share_layer_buffers(bool on);
    bool layers_shared() const { return layers_shared_; }
    // O31: whether layer 0's backward sweep reads nothing of its forward but msgf[0..3] and D (the density layer 0
    // with the hidden table, skip folded, linear readout: MACE-MP-0b3 / MPA-0 / OMAT-0 / MatPES) or, list 19, also
    // the projections lin_l (gout_l where the gate ran in their epilogue) and res_0 (the gated layer 0 with a linear
    // readout: MH-1, OMOL-0), so a domain can keep them (GenericPassInputs::keep0) instead of the replay; and their
    // floats for N receivers (keep0_layout in generic.cu).
    bool sweep_keep_layer0() const;
    size_t sweep_keep_floats(int N) const;
    // The compiled product kernels this bundle uses, one signature per layer, joined by ';' (loup_get_info).
    std::string kernel_signature() const;

    // Per-system buffers (N atoms); edge buffers grow with ensure_edges (call outside graph capture).
    void alloc_system(int N);
    void free_system();
    void ensure_edges(int E_cap);
    // The long-range (POLAR) k-vectors follow the cell: call on every cell change, outside stream capture; true if
    // captured graphs must be re-recorded.
    bool set_cell(const double* cell);
    bool has_lr() const { return lr_ != nullptr; }
    bool is_ilmlip() const { return ilmlip_ != nullptr; }
    float* ilmlip_forces_f32() const;
    float ilmlip_r_max1() const;
    float ilmlip_r_max2() const;
    bool ilmlip_supports_pack_vs() const;
    const SplineGrid* preloc_grid() const;
    int preloc_pair_sym() const;
    // Records the whole force pass on in.stream (capturable; no host syncs, no allocation).
    void record(const GenericPassInputs& in);
    // S64: times each layer's conv forward variants on the last recorded pass's inputs (outside capture; sm_120) and
    // keeps the fastest. -1: no pass recorded for this system yet (call again later); 0: unchanged; 1: a variant
    // changed (captured graphs must be re-recorded).
    int tune_conv_variants(cudaStream_t st);
private:
    struct ConvTuneArgs {
        bool valid = false;
        int N = 0;
        const int* row_ptr = nullptr;
        const int* senders = nullptr;
        const float4* vec_len = nullptr;
        const int* slab = nullptr;
        const float2* t_sx = nullptr;
        const int* species = nullptr;
    };
    ConvTuneArgs conv_args_;
    int conv_var_override_ = -1;  // S64: LOUP_CONV_FWD_VARIANT (test override), -1 none
    int conv_tune_env_ = -1;      // LOUP_CONV_TUNE at load: -1 unset (sm_120 only), 0 off, 1 any GPU (tests)
    void free_model();
    // S57 skip-chain fold (sm_120, cc 8.0; LOUP_CHAIN_FOLD=0 disables): see setup_chain_fold in generic.cu.
    void setup_chain_fold(const Bundle& b);
    bool chain_fold_ = false;        // the fold of the allocated system (use_chain_fold)
    bool w2t_half_ = false;          // LOUP_W2T_HALF: some layer has its weight panels (setup_w2t_half)
    void setup_w2t_half();
    bool w2f_half_ = false;          // LOUP_W2F_HALF: some layer has its transposed weight panels (setup_w2f_half)
    void setup_w2f_half();
    bool lin_half_ = false;
    bool polar_h2_ = false;
    int polar_h2_min_m_ = 128;
    struct H2RowW {
        void* B = nullptr;    float* pB = nullptr;
        void* BT = nullptr;   float* pBT = nullptr;
        int rows = 0, cols = 0;
    };
    std::unordered_map<const float*, H2RowW> ph2_map_;
    std::vector<void*> ph2_allocs_;
    void setup_polar_h2();          // LOUP_LIN_HALF: some conv layer's projection has its panels (setup_lin_half)
    int lin_half_parts_ = 0;         // LOUP_LIN_HALF_PARTS (setup_lin_half)
    int lin_half_srows_ = 0;         // LOUP_LIN_HALF_SPECIES_ROWS: bit 128's fewest rows per species and degree
    int lin_half_fold_tiles_ = 0;    // LOUP_LIN_HALF_FOLD_TILES: bit 256's fewest 128 x 64 tiles per product (SMs)
    int res_bwd_ = 3;                // LOUP_RES_BWD (setup_lin_half; generic.cu's backward W_res^T)
    void setup_lin_half();
    static constexpr int kLinAmaxSlots = 256;  // LOUP_DEBUG_LIN_AMAX: layer t, kind k, degree l at 16 t + 4 k + l;
                                               // the chain fold's at 128 + 16 t + l (y) and 128 + 16 t + 8 + l (dx)
    unsigned* d_lin_amax_ = nullptr;
    void lin_amax_record(const float* A, int64_t n, int slot, cudaStream_t st) const;
    bool chain_fold_built_ = false;  // setup_chain_fold composed the fold's weights
    bool share_layers_ = false;      // asked for the systems to come (share_layer_buffers)
    bool share_layers_env_ = true;   // LOUP_TILE_SHARE_LAYERS != 0, read at load
    bool layers_shared_ = false;     // the allocated system's layers share their arena
    size_t sys_bytes_unshared_ = 0;  // sys_bytes_ with every layer's buffers its own
    size_t sys_max_alloc_ = 0;       // the largest single buffer of alloc_system
    void update_h0_read();           // h0_read_ for the current chain_fold_ (S97)
    // S59 layer-0 species tables (sm_120, cc 8.0; LOUP_SPECIES_TABLES=0 disables): see setup_species_tables.
    void setup_species_tables(const Bundle& b);
    bool species_tables_ = false;
    // Long-range hidden fold (sm_120; LOUP_POLAR_HFOLD=0 disables, 1 / 2 / 3 picks the degrees): see setup_lr_fold.
    void setup_lr_fold(const Bundle& b);
    // JIT_DESIGN.md stage 1 (LOUP_JIT=1): layer 0's kernels compiled for this model at load; see setup_jit in generic.cu.
    void setup_jit();
    int S_ = 0, n_layers_ = 0, sh_perm_ = 0, has_zbl_ = 0;
    float r_max_ = 0.0f;
    std::vector<int> Z_;
    std::vector<double> e0_;
    float* d_embedding_ = nullptr;  // [S, C0]
    int C0_ = 0;
    std::vector<GenericLayer*> layers_;
    PolarLR* lr_ = nullptr;
    IlmlipModel* ilmlip_ = nullptr;
    // ZBL per ordered pair
    float zbl_c_[4] = {0, 0, 0, 0};
    float zbl_p_ = 5.0f, zbl_rmax_ = 0.0f;
    float *d_zbl_rc_ = nullptr, *d_zbl_inv_a_ = nullptr, *d_zbl_pref_ = nullptr;
    // system buffers
    int N_ = 0, E_cap_ = 0;
    size_t sys_bytes_ = 0, edge_bytes_ = 0;
    // LOUP_SWEEP_REPLAY=1 (testing): record() runs the per-layer sweep schedule of a domain with ghost atoms on the
    // whole system; see record(). sys_alloc_bytes_ parallels sys_allocs_.
    bool sweep_replay_ = false;
    bool sweep_zero_ = true;  // list 19: a sweep zeroes only its accumulators (LOUP_SWEEP_ZERO=0: the whole slab)
    bool gate_fwd_tiled_ = true;  // list 20: LOUP_GATE_FWD_L1's store in the last-layer sweep (LOUP_GATE_FWD_TILED=0)
    bool fold_sweeps_ = false;    // list 20: chain_fold_ in a domain's sweeps (use_chain_fold(on, true))
    struct Hand {
        float* p;
        float* g;
        int w;
        bool locals;
    };
    std::vector<std::vector<Hand>> hand_;  // list 20: per layer, what its sweeps hand on (hand_count)
    double fold_delta_[2] = {0.0, 0.0};    // list 20: fold_delta_bytes (unshared, shared)
    std::vector<int> fold_hand_w_;         // list 20: fold_hand_widths
    double* d_energy_sink_ = nullptr;  // a replayed forward adds its readout's energy here: it is already counted
    // list 19: per keep0 buffer, the gated layer 0's gate_vform of its forward sweep, which the backward sweep of that
    // domain takes (an ask there may differ: a GEMM plan tuned in between); one entry per domain in flight
    std::unordered_map<const float*, unsigned> keep0_vform_;
    std::vector<size_t> sys_alloc_bytes_;
    std::vector<void*> sys_allocs_;
    // The buffers whose zeros alloc_system writes once and every pass reads (a pruned conv layer's skipped
    // projections, a compact layer's pad columns): no sweep writes them, so the sweep schedule's test leaves them.
    std::vector<const void*> sys_zero_once_;
    float* d_edge_dvec_ = nullptr;  // [E_cap, 3]
    void* d_zero_slab_ = nullptr;
    size_t zero_slab_bytes_ = 0;
    float* d_h0_ = nullptr;         // [N, C0] layer-0 node features
    bool h0_read_ = true;           // S97: some consumer reads d_h0_ (else record skips its gather)
    float* d_scratch_ = nullptr;    // gradient scratch
    size_t scratch_floats_ = 0;
};

}  // namespace loup
