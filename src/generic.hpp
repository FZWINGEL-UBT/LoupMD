#pragma once
// The generic MACE path: any bundle with arch=generic (python/export_generic.py) — MACE-OMAT-0 small/medium,
// MACE-OFF, MACE-MH-1 and MACE-POLAR-1's local part. Layer by layer: linear_up GEMMs, the fused conv tensor
// product with radial splines (generated_generic.cuh), message linears, density / avg-neighbour / gated
// normalisations, the factored symmetric contraction, skip connections and readouts; then the same in reverse for
// forces and stress. engine.cu owns the neighbour list, the MD loop and the GEMM autotuning; it calls in here.
#include <cstdint>
#include <functional>
#include <string>
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
};
// Runs the projection fused with degree l's gate (M = natoms (2l + 1), N = C) when the GEMM's plan and shape allow;
// false: nothing launched (the caller runs the plain GEMM and the gate kernel for l). Backward: dgout_l = A[M, K] @
// B[N, K]^T; forward: lin_l = A[M, K] @ B[K, N]. A null: only reports whether it would run.
using GemmGateFn = std::function<bool(int l, int M, int N, int K, const float* A, const float* B, const GateArgs& g)>;

struct GenericPassInputs {
    cudaStream_t stream;
    int N;
    const int* E_dev;          // device: number of active edges (act_row_ptr[N])
    int E_cap;                 // capacity of the edge arrays
    const int* act_row_ptr;    // [N + 1], receiver rows
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
    GemmFn gemm;
    std::function<void()> gemm_fork;
    std::function<void(int)> gemm_lane;
    std::function<void()> gemm_join;
    GemmGateFn gemm_gate_bwd;  // empty: never fused (LOUP_GATE_FUSE=0)
    GemmGateFn gemm_gate_fwd;  // empty: never fused (LOUP_GATE_FUSE_FWD=0)
};

struct GenericLayer;

class PolarLR;

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

    // Per-system buffers (N atoms); edge buffers grow with ensure_edges (call outside graph capture).
    void alloc_system(int N);
    void free_system();
    void ensure_edges(int E_cap);
    // The long-range (POLAR) k-vectors follow the cell: call on every cell change, outside stream capture; true if
    // captured graphs must be re-recorded.
    bool set_cell(const double* cell);
    bool has_lr() const { return lr_ != nullptr; }
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
    bool chain_fold_ = false;
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
    // ZBL per ordered pair
    float zbl_c_[4] = {0, 0, 0, 0};
    float zbl_p_ = 5.0f, zbl_rmax_ = 0.0f;
    float *d_zbl_rc_ = nullptr, *d_zbl_inv_a_ = nullptr, *d_zbl_pref_ = nullptr;
    // system buffers
    int N_ = 0, E_cap_ = 0;
    std::vector<void*> sys_allocs_;
    float* d_edge_dvec_ = nullptr;  // [E_cap, 3]
    void* d_zero_slab_ = nullptr;
    size_t zero_slab_bytes_ = 0;
    float* d_h0_ = nullptr;         // [N, C0] layer-0 node features
    bool h0_read_ = true;           // S97: some consumer reads d_h0_ (else record skips its gather)
    float* d_scratch_ = nullptr;    // gradient scratch
    size_t scratch_floats_ = 0;
};

}  // namespace loup
