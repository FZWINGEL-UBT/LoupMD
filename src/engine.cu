#include "loupmd.h"
#include "model.hpp"
#include "kernels_common.cuh"
#include "generated_kernels.cuh"
#include "generated_kernels_signature.h"
#include "generic.hpp"
#include "bundle.hpp"
#include "thermo.hpp"
#include "cell.hpp"
#include "tiled.hpp"
#include "dispersion/term.hpp"
#include <cstddef>

#include <cub/cub.cuh>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda.h>
#include <cublasLt.h>
#include <type_traits>
#include <map>
#include <memory>
#include <chrono>
#include <functional>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>

// Defined in neighbors.cu & integrator.cu (compiled into same or linked translation units;
// we include or forward-declare cleanly here).
namespace loup {
struct CellMat64 {
    double c[9];
    double inv[9];
};
void compute_cell_mat(const double* cell_row_major, CellMat64& out, double& out_vol, double heights[3]);

__global__ void wrap_and_bin_kernel(int N, CellMat64 cell, int nx, int ny, int nz,
                                    const double* __restrict__ pos64, double* __restrict__ pos_wrapped,
                                    int* __restrict__ atom_wrap, int* __restrict__ atom_bin,
                                    int* __restrict__ bin_counts);
__global__ void atom_order_keys_kernel(int N, int nx, int ny, int nz, const int* __restrict__ species,
                                       const int* __restrict__ atom_bin, unsigned long long* __restrict__ keys,
                                       int* __restrict__ vals);
__global__ void gather_rows_f64_kernel(int n, int width, const int* __restrict__ order,
                                       const double* __restrict__ src, double* __restrict__ dst);
__global__ void scatter_bins_kernel(int N, const double* __restrict__ pos_wrapped,
                                    const int* __restrict__ atom_wrap, const int* __restrict__ species,
                                    const int* __restrict__ atom_bin, int* __restrict__ bin_cursor,
                                    int* __restrict__ bin_atom_id, double* __restrict__ bin_pos, float4* __restrict__ bin_pos_f32,
                                    int* __restrict__ bin_wrap, int* __restrict__ bin_sp);
__global__ void count_neighbors_kernel(int N, CellMat64 cell, int nx, int ny, int nz,
                                       int rx, int ry, int rz, double r_list2, float r_lo2_f32, float r_hi2_f32,
                                       const double* __restrict__ pos_wrapped, const int* __restrict__ atom_bin,
                                       const int* __restrict__ bin_offsets, const int* __restrict__ bin_atom_id,
                                       const double* __restrict__ bin_pos, const float4* __restrict__ bin_pos_f32, int* __restrict__ deg);
__global__ void fill_neighbors_kernel(int N, int S, CellMat64 cell, int nx, int ny, int nz,
                                      int rx, int ry, int rz, double r_list2, float r_lo2_f32, float r_hi2_f32,
                                      const double* __restrict__ pos_wrapped, const int* __restrict__ atom_wrap,
                                      const int* __restrict__ species, const int* __restrict__ atom_bin,
                                      const int* __restrict__ bin_offsets, const int* __restrict__ bin_atom_id,
                                      const double* __restrict__ bin_pos, const float4* __restrict__ bin_pos_f32,
                                      const int* __restrict__ bin_wrap,
                                      const int* __restrict__ bin_sp, const int* __restrict__ row_ptr,
                                      int* __restrict__ tmp_snd, int* __restrict__ tmp_rcv,
                                      int* __restrict__ tmp_shifts, int* __restrict__ tmp_pairs,
                                     uint32_t* __restrict__ sort_keys, int* __restrict__ sort_vals);
__global__ void count_neighbors_scalar_kernel(int N, CellMat64 cell, int nx, int ny, int nz,
                                              int rx, int ry, int rz, double r_list2, float r_lo2_f32, float r_hi2_f32,
                                              const double* __restrict__ pos_wrapped, const int* __restrict__ atom_bin,
                                              const int* __restrict__ bin_offsets, const int* __restrict__ bin_atom_id,
                                              const double* __restrict__ bin_pos, const float4* __restrict__ bin_pos_f32,
                                              int* __restrict__ deg);
__global__ void fill_neighbors_scalar_kernel(int N, int S, CellMat64 cell, int nx, int ny, int nz,
                                             int rx, int ry, int rz, double r_list2, float r_lo2_f32, float r_hi2_f32,
                                             const double* __restrict__ pos_wrapped, const int* __restrict__ atom_wrap,
                                             const int* __restrict__ species, const int* __restrict__ atom_bin,
                                             const int* __restrict__ bin_offsets, const int* __restrict__ bin_atom_id,
                                             const double* __restrict__ bin_pos, const float4* __restrict__ bin_pos_f32,
                                             const int* __restrict__ bin_wrap,
                                             const int* __restrict__ bin_sp, const int* __restrict__ row_ptr,
                                             int* __restrict__ tmp_snd, int* __restrict__ tmp_rcv,
                                             int* __restrict__ tmp_shifts, int* __restrict__ tmp_pairs,
                                             uint32_t* __restrict__ sort_keys, int* __restrict__ sort_vals);
__global__ void gather_sorted_edges_kernel(int E, CellMat64 cell, const int* __restrict__ sort_vals,
                                           const int* __restrict__ tmp_snd, const int* __restrict__ tmp_rcv,
                                           const int* __restrict__ tmp_shifts, const int* __restrict__ tmp_pairs,
                                           int* __restrict__ senders, int* __restrict__ receivers,
                                           double* __restrict__ shift_cart, int* __restrict__ pairs);
__global__ void check_row_species_order_kernel(int E, const int* __restrict__ senders,
                                               const int* __restrict__ receivers, const int* __restrict__ species,
                                               int* __restrict__ err);
__global__ void compute_and_count_active_kernel(
    int N, float r_max, const double* __restrict__ pos64,
    const int* __restrict__ row_ptr, const int* __restrict__ senders,
    const int* __restrict__ receivers, const double* __restrict__ shift_cart,
    float4* __restrict__ tmp_vec_len, int* __restrict__ act_deg);
__global__ void scatter_active_edges_kernel(
    int N, float r_max, SplineGrid grid, int S, int pair_sym, const int* __restrict__ species,
    const int* __restrict__ row_ptr, const int* __restrict__ act_row_ptr,
    const int* __restrict__ senders, const int* __restrict__ receivers, const int* __restrict__ pairs,
    const float4* __restrict__ tmp_vec_len,
    int* __restrict__ act_senders, int* __restrict__ act_receivers, int* __restrict__ act_pairs,
    float4* __restrict__ act_vec_len,
    int* __restrict__ act_slab, float2* __restrict__ act_t_sx);
__global__ void check_skin_kernel(int N, double half_skin, int lag, int lag_switch, const double* __restrict__ pos64,
                                  const double* __restrict__ ref_pos64, const unsigned int* __restrict__ max_step_disp2,
                                  int* __restrict__ need_rebuild);

__global__ void tabled_outer_sum0_forward_kernel(
    int N, int S, int pieces0, int n_bins0, float r_lo0, float bin_scale0,
    const int* __restrict__ act_row_ptr, const float4* __restrict__ vec_len,
    const int* __restrict__ senders, const int* __restrict__ species,
    const float* __restrict__ coef0, const float* __restrict__ starts0,
    const float* __restrict__ scales0, const int32_t* __restrict__ bins0,
    float* __restrict__ S0_0, float* __restrict__ S0_1, float* __restrict__ S0_2, float* __restrict__ S0_3,
    float* __restrict__ D0);

__global__ void tabled_outer_sum0_backward_kernel(
    const int* __restrict__ E_ptr, int S, int pieces0, int n_bins0, float r_lo0, float bin_scale0, float r_max,
    int compute_stress, float zbl_r_max_max, float zbl_p,
    float zbl_c0, float zbl_c1, float zbl_c2, float zbl_c3,
    const float4* __restrict__ vec_len,
    const int* __restrict__ senders, const int* __restrict__ receivers, const int* __restrict__ pairs,
    const int* __restrict__ species, const float* __restrict__ pcoef0, const float* __restrict__ starts0,
    const float* __restrict__ scales0, const int32_t* __restrict__ bins0,
    const float* __restrict__ dS0_0, const float* __restrict__ dS0_1,
    const float* __restrict__ dS0_2, const float* __restrict__ dS0_3,
    const float* __restrict__ dD0,
    const float* __restrict__ zbl_r_cut, const float* __restrict__ zbl_inv_a, const float* __restrict__ zbl_pref,
    const float* __restrict__ edge_force_1,
    double* __restrict__ energy_f64, double* __restrict__ forces_f64, double* __restrict__ virial_grad_f64);
__global__ void zbl_energy0_kernel(
    const int* __restrict__ E_ptr, float zbl_r_max_max, float zbl_p,
    float zbl_c0, float zbl_c1, float zbl_c2, float zbl_c3,
    const float4* __restrict__ vec_len, const int* __restrict__ pairs,
    const float* __restrict__ zbl_r_cut, const float* __restrict__ zbl_inv_a, const float* __restrict__ zbl_pref,
    double* __restrict__ energy_f64);
constexpr double ASE_FS = 0.09822694788464063;
constexpr double ASE_KB = 8.617330337217213e-5;
constexpr double GPA_PER_EV_A3 = 160.21766208;  // ASE: 1 / units.GPa

__global__ void nve_first_half_kernel(int N, double dt_ase, const double* __restrict__ forces,
                                      const double* __restrict__ inv_masses,
                                      double* __restrict__ pos64, double* __restrict__ vel64,
                                      unsigned int* __restrict__ max_step_disp2);
__global__ void nve_second_half_kernel(int N, double dt_ase, const double* __restrict__ forces,
                                       const double* __restrict__ inv_masses, double* __restrict__ vel64);
__global__ void rotate_step_disp_kernel(unsigned int* __restrict__ max_step_disp2, int* __restrict__ need_rebuild);
__global__ void langevin_first_half_kernel(
    int N, double dt_ase, double c1, double c2,
    const double* __restrict__ target, unsigned long long seed, const unsigned long long* __restrict__ step_counter,
    const double* __restrict__ forces, const double* __restrict__ inv_masses,
    const double* __restrict__ sqrt_inv_masses, double* __restrict__ pos64,
    double* __restrict__ vel64, double* __restrict__ rnd_vel_save, unsigned int* __restrict__ max_step_disp2);
__global__ void langevin_second_half_kernel(
    int N, double c1, double c2, const double* __restrict__ forces,
    const double* __restrict__ inv_masses, const double* __restrict__ rnd_vel_save,
    double* __restrict__ vel64, unsigned long long* __restrict__ step_counter, double* __restrict__ target,
    double a3, double a4, double a5, double two_fr_kb);
__global__ void langevin_noise_kernel(
    int N, const double* __restrict__ target,
    unsigned long long seed, const unsigned long long* __restrict__ step_counter,
    const double* __restrict__ masses, const double* __restrict__ sqrt_inv_masses,
    double* __restrict__ rnd_pos, double* __restrict__ rnd_vel, double* __restrict__ com);
__global__ void langevin_first_half_fixcom_kernel(
    int N, double dt_ase, double c1, double c2, double inv_total_mass, const double* __restrict__ com,
    const double* __restrict__ forces, const double* __restrict__ inv_masses, const double* __restrict__ rnd_pos,
    double* __restrict__ rnd_vel, double* __restrict__ pos64, double* __restrict__ vel64,
    unsigned int* __restrict__ max_step_disp2);
__global__ void momentum_sum_kernel(int N, const double* __restrict__ masses, const double* __restrict__ vel64,
                                    double* __restrict__ com);
__global__ void momentum_remove_kernel(int N, double inv_total_mass, const double* __restrict__ com,
                                       double* __restrict__ vel64);
__global__ void kinetic_energy_kernel(int N, const double* __restrict__ masses,
                                      const double* __restrict__ vel64, double* __restrict__ out_ke);
__global__ void min_gradient_kernel(
    int N, int have_previous, const double* __restrict__ forces, const double* __restrict__ inv_masses,
    const double* __restrict__ pos64, const double* __restrict__ x0, const double* __restrict__ g0,
    double* __restrict__ g, double* __restrict__ s_row, double* __restrict__ y_row,
    unsigned long long* __restrict__ out_fmax2);
__global__ void lbfgs_products_kernel(
    int n3, int stride, const double* __restrict__ S, const double* __restrict__ Y, const double* __restrict__ g,
    const double* __restrict__ s_new, const double* __restrict__ y_new, double* __restrict__ out);
__global__ void lbfgs_direction_kernel(
    int N, int rows, int stride, double h0, const int* __restrict__ row_index, const double* __restrict__ u,
    const double* __restrict__ w, const double* __restrict__ S, const double* __restrict__ Y,
    const double* __restrict__ g, double* __restrict__ p, unsigned long long* __restrict__ out_len2);
__global__ void lbfgs_move_kernel(
    int N, double maxstep, double damping, const unsigned long long* __restrict__ len2,
    const double* __restrict__ p, const double* __restrict__ g, double* __restrict__ pos64,
    double* __restrict__ x0, double* __restrict__ g0);
__global__ void max_displacement2_kernel(int N, const double* __restrict__ pos64,
                                         const double* __restrict__ ref_pos64,
                                         unsigned long long* __restrict__ out_disp2);
__global__ void csvr_noise_debug_kernel(unsigned long long seed, unsigned long long first_step, int n, int ndof,
                                        double* __restrict__ normal, double* __restrict__ chi2);
__global__ void csvr_alpha_kernel(int ndof, double exp_term, double kb, unsigned long long seed,
                                  unsigned long long* __restrict__ step_counter, double* __restrict__ target,
                                  double* __restrict__ work, double* __restrict__ state);
__global__ void nhc_half_kernel(int ndof, int tchain, int tloop, double delta, double q, double kb, int advance,
                                double* __restrict__ target, double* __restrict__ work, double* __restrict__ state);
__global__ void scale_velocities_kernel(int N, const double* __restrict__ work, double* __restrict__ vel64);
__global__ void baoab_cell_first_kernel(
    int full, double dt, double gamma, double ou_unit, double gamma_b, double ou_b_unit, double mass, int ndof,
    double kb, unsigned long long seed, const unsigned long long* __restrict__ step_counter,
    double* __restrict__ target, double* __restrict__ state, double* __restrict__ work, CellDev* __restrict__ cell);
__global__ void baoab_first_kernel(
    int N, double dt, const double* __restrict__ work, unsigned long long seed,
    const unsigned long long* __restrict__ step_counter, const double* __restrict__ forces,
    const double* __restrict__ inv_masses, const double* __restrict__ sqrt_inv_masses,
    double* __restrict__ pos64, double* __restrict__ vel64, unsigned int* __restrict__ max_step_disp2);
__global__ void kinetic_tensor_kernel(int N, const double* __restrict__ masses, const double* __restrict__ vel64,
                                      double* __restrict__ out);
__global__ void baoab_cell_second_kernel(
    int full, int kick, int advance, double dt, int ndof, double ext, const double* __restrict__ virial,
    const CellDev* __restrict__ cell, double* __restrict__ work, double* __restrict__ state,
    unsigned long long* __restrict__ step_counter);
__global__ void mtk_first_kernel(
    int full, double dt, int tchain, int tloop, int pchain, int ploop, int ndof, double q_t, double r_p, double w,
    double kb, double pressure, const double* __restrict__ target, double* __restrict__ state,
    double* __restrict__ work, const double* __restrict__ virial, CellDev* __restrict__ cell);
__global__ void mtk_first_atoms_kernel(
    int N, double dt, const double* __restrict__ work, const double* __restrict__ forces,
    const double* __restrict__ inv_masses, double* __restrict__ pos64, double* __restrict__ vel64,
    unsigned int* __restrict__ max_step_disp2);
__global__ void mtk_kick_kernel(int N, double dt, const double* __restrict__ work, const double* __restrict__ forces,
                                const double* __restrict__ inv_masses, double* __restrict__ vel64);
__global__ void mtk_second_kernel(
    int full, double dt, int tchain, int tloop, int pchain, int ploop, int ndof, double q_t, double r_p, double w,
    double kb, double pressure, double* __restrict__ target, double* __restrict__ state, double* __restrict__ work,
    const double* __restrict__ virial, const CellDev* __restrict__ cell);
__global__ void berendsen_scale_kernel(int ndof, double dt_over_tau, double kb, double* __restrict__ target,
                                       double* __restrict__ work);
__global__ void csvr_first_half_kernel(
    int N, double dt_ase, const double* __restrict__ work, const double* __restrict__ forces,
    const double* __restrict__ inv_masses, double* __restrict__ pos64, double* __restrict__ vel64,
    unsigned int* __restrict__ max_step_disp2);
}  // namespace loup

// Scalars an integrator carries from step to step and a restart needs (the state block holds all of them):
// CSVR: [0] the energy the thermostat has transferred. Nose-Hoover chain: [0] minus the chain's energy, then eta
// and p_eta of each thermostat (nhc_half_kernel). The other slots are for the integrators to come.
constexpr int kIntStateSlots = 64;
constexpr int kIntWorkSlots = 64;  // per-step scratch: CSVR's kinetic energy and alpha; BAOAB NPT's matrices

// loup_minimize's LBFGS (src/minimize.cu). The history and the step's vectors are on the device in the handle's
// atom order; the host keeps what the two triangular solves need. It is valid for the positions the last call
// left (pos_epoch), with the settings of that call.
struct MinState {
    bool valid = false;
    unsigned long long epoch = 0;   // LoupHandle::pos_epoch the state belongs to
    double maxstep = 0.0, damping = 0.0, alpha = 0.0;
    int memory = 0;
    int n3 = 0;                     // 3 N of the buffers
    int count = 0;                  // pairs in the history (rows 0 .. count - 1)
    int head = 0;                   // the row the next pair goes to
    long long iteration = 0;        // steps taken
    std::vector<int> order;         // history rows, oldest first
    std::vector<double> rho;        // [memory]: 1 / (y_i . s_i)
    std::vector<double> gram;       // [memory, memory]: s_i . y_j
    std::vector<double> yy;         // [memory, memory]: y_i . y_j
    double* d_S = nullptr;          // [memory, n3]
    double* d_Y = nullptr;          // [memory, n3]
    double* d_vec = nullptr;        // [4, n3]: g, g0, x0, p
    double* d_small = nullptr;      // [7 memory]: the five products per row, then u and w
    int* d_rows = nullptr;          // [memory]: the rows of u and w
    // [3], bits of doubles: the largest squared force, the largest squared length of the last direction, the
    // largest squared displacement from the neighbour list's reference after the last move
    unsigned long long* d_max = nullptr;
    // Pinned host memory, so that no copy waits for the host: d_max's three values, the energy, the five products
    // per row, then u and w; and the rows of u and w.
    double* h_pin = nullptr;        // [4 + 7 memory]
    int* h_rows = nullptr;          // [memory]
    // What the list check of the next move goes by (minimize_impl): the largest displacement from the list's
    // reference at the current positions and the longest step of an atom in the last move; -1: not known.
    double disp_known = -1.0, step_known = -1.0;
    void release() {
        auto freep = [](auto*& p) {
            if (p) { cudaFree(p); p = nullptr; }
        };
        freep(d_S); freep(d_Y); freep(d_vec); freep(d_small); freep(d_rows); freep(d_max);
        if (h_pin) { cudaFreeHost(h_pin); h_pin = nullptr; }
        if (h_rows) { cudaFreeHost(h_rows); h_rows = nullptr; }
        valid = false;
        n3 = 0;
    }
};

// FNV-1a over a file's bytes (0 if unreadable).
static uint64_t fnv1a_file(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    uint64_t hsh = 1469598103934665603ULL;
    std::vector<unsigned char> buf(1 << 16);
    size_t n;
    while ((n = fread(buf.data(), 1, buf.size(), fp)) > 0)
        for (size_t i = 0; i < n; ++i) hsh = (hsh ^ buf[i]) * 1099511628211ULL;
    fclose(fp);
    return hsh ? hsh : 1;
}

// Tensor-core GEMM cubin (GEMM autotune candidates). Two formats: the shipped FastMACE builds (cubins/tf32x3_hyb_*,
// fm_tf32x3_info: tiles 64x64, 32x64, 64x32, beta 0 only) and LoupMD's own (cubins/src/gemm_tc.cu,
// loup_gemm_tc_info: math, beta support and up to kMaxTiles tiles; through LOUP_HYBRID_CUBIN for now). Tiles 0-2 are
// the same shapes in both, so a stored plan's tile index means the same kernel shape.
struct HybCubin {
    static constexpr int kMaxTiles = 8;
    bool loaded = false;
    bool own_format = false;  // loup_gemm_tc_info: entries take beta
    bool beta_ok = false;     // C = A B + beta C may use it
    int ntiles = 0;
    uint64_t cubin_hash = 0;  // the loaded cubin file's FNV-1a hash (keys the GEMM plan cache)
    std::string path_loaded;  // for diagnostics (the cubin's math: its build, see cubins/src)
    void* lib = nullptr;
    CUmodule mod = nullptr;
    CUfunction fn[kMaxTiles][2] = {};  // [tile][transB ? 1 : 0]
    int threads[kMaxTiles] = {};
    int smem[kMaxTiles][2] = {};
    int tile_m[kMaxTiles] = {};
    int tile_n[kMaxTiles] = {};
    CUresult (*p_cuModuleLoad)(CUmodule*, const char*) = nullptr;
    CUresult (*p_cuModuleUnload)(CUmodule) = nullptr;
    CUresult (*p_cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*p_cuModuleGetGlobal)(CUdeviceptr*, size_t*, CUmodule, const char*) = nullptr;
    CUresult (*p_cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult (*p_cuFuncSetAttribute)(CUfunction, CUfunction_attribute, int) = nullptr;
    CUresult (*p_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**) = nullptr;

    void init(int device_id) {
        const char* no_hyb = std::getenv("LOUP_NO_HYBRID");
        if (no_hyb && no_hyb[0] == '1') return;
        int maj = 0, min = 0;
        if (cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, device_id) != cudaSuccess) return;
        if (cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, device_id) != cudaSuccess) return;
        const char* cname = nullptr;
        if (maj == 8) cname = "tf32x3_hyb_sm80.cubin";
        else if (maj == 9) cname = "tf32x3_hyb_sm90.cubin";
        else if (maj == 12) cname = "tf32x3_hyb_sm120.cubin";
        else return;

        std::string path;
        if (const char* envp = std::getenv("LOUP_HYBRID_CUBIN")) {
            path = envp;
        } else {
            Dl_info dli{};
            if (dladdr((void*)&loup_create, &dli) && dli.dli_fname) {
                std::string so(dli.dli_fname);
                auto slash = so.find_last_of('/');
                std::string dir = (slash == std::string::npos) ? "." : so.substr(0, slash);
                path = dir + "/cubins/" + cname;
            }
        }
        if (path.empty()) return;
        cubin_hash = fnv1a_file(path.c_str());
        path_loaded = path;
        FILE* chk = fopen(path.c_str(), "rb");
        if (!chk) return;
        fclose(chk);

        lib = dlopen("libcuda.so.1", RTLD_LAZY);
        if (!lib) return;
        auto sym = [&](auto& dst, const char* name) {
            dst = reinterpret_cast<std::decay_t<decltype(dst)>>(dlsym(lib, name));
            return dst != nullptr;
        };
        if (!sym(p_cuModuleLoad, "cuModuleLoad") ||
            !sym(p_cuModuleUnload, "cuModuleUnload") ||
            !sym(p_cuModuleGetFunction, "cuModuleGetFunction") ||
            !sym(p_cuModuleGetGlobal, "cuModuleGetGlobal_v2") ||
            !sym(p_cuMemcpyDtoH, "cuMemcpyDtoH_v2") ||
            !sym(p_cuFuncSetAttribute, "cuFuncSetAttribute") ||
            !sym(p_cuLaunchKernel, "cuLaunchKernel")) {
            return;
        }
        if (p_cuModuleLoad(&mod, path.c_str()) != CUDA_SUCCESS) { mod = nullptr; return; }
        CUdeviceptr gptr = 0; size_t gsz = 0;
        if (p_cuModuleGetGlobal(&gptr, &gsz, mod, "loup_gemm_tc_info") == CUDA_SUCCESS && gsz >= 4 * sizeof(int)) {
            std::vector<int> info(gsz / sizeof(int));
            if (p_cuMemcpyDtoH(info.data(), gptr, gsz) != CUDA_SUCCESS || info[0] != 1) { destroy(); return; }
            own_format = true;
            beta_ok = info[2] != 0;
            ntiles = info[3];
            if (ntiles < 1 || ntiles > kMaxTiles || (int)info.size() < 4 + 5 * ntiles) { destroy(); return; }
            for (int t = 0; t < ntiles; ++t) {
                const int* ti = &info[4 + 5 * t];
                tile_m[t] = ti[0]; tile_n[t] = ti[1]; threads[t] = ti[2]; smem[t][0] = ti[3]; smem[t][1] = ti[4];
                for (int b = 0; b < 2; ++b) {
                    char name[48];
                    snprintf(name, sizeof(name), "loup_gemm_tc_%s_%d", b ? "rc" : "rr", t);
                    if (p_cuModuleGetFunction(&fn[t][b], mod, name) != CUDA_SUCCESS) { destroy(); return; }
                }
            }
        } else {
            if (p_cuModuleGetGlobal(&gptr, &gsz, mod, "fm_tf32x3_info") != CUDA_SUCCESS || gsz < 40) {
                destroy(); return;
            }
            int info[10] = {0};
            if (p_cuMemcpyDtoH(info, gptr, sizeof(info)) != CUDA_SUCCESS) { destroy(); return; }
            ntiles = 3;
            tile_m[0] = 64; tile_n[0] = 64; tile_m[1] = 32; tile_n[1] = 64; tile_m[2] = 64; tile_n[2] = 32;
            threads[0] = info[0]; smem[0][0] = info[1]; smem[0][1] = info[2];
            threads[1] = info[5]; smem[1][0] = info[6]; smem[1][1] = info[7];
            threads[2] = info[5]; smem[2][0] = info[8]; smem[2][1] = info[9];
            const char* names[3][2] = {
                {"fm_tf32x3_rr_64x64", "fm_tf32x3_rc_64x64"},
                {"fm_tf32x3_rr_32x64", "fm_tf32x3_rc_32x64"},
                {"fm_tf32x3_rr_64x32", "fm_tf32x3_rc_64x32"},
            };
            for (int t = 0; t < 3; ++t)
                for (int b = 0; b < 2; ++b)
                    if (p_cuModuleGetFunction(&fn[t][b], mod, names[t][b]) != CUDA_SUCCESS) { destroy(); return; }
        }
        for (int t = 0; t < ntiles; ++t)
            for (int b = 0; b < 2; ++b)
                if (smem[t][b] > 48 * 1024)
                    p_cuFuncSetAttribute(fn[t][b], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem[t][b]);
        loaded = true;
    }

    // Whether a product with this beta may run on the cubin (alpha 1 always).
    bool takes(float beta) const { return loaded && (beta == 0.0f || beta_ok); }

    void launch(int tile, bool transB, int M, int N, int K,
                const float* A, int lda, const float* B, int ldb, float* C, int ldc, float beta,
                cudaStream_t stream) const {
        void* args[10] = {
            (void*)&M, (void*)&N, (void*)&K,
            (void*)&A, (void*)&lda, (void*)&B, (void*)&ldb, (void*)&C, (void*)&ldc, (void*)&beta
        };
        unsigned gx = (unsigned)((M + tile_m[tile] - 1) / tile_m[tile]);
        unsigned gy = (unsigned)((N + tile_n[tile] - 1) / tile_n[tile]);
        CUresult rc = p_cuLaunchKernel(
            fn[tile][transB ? 1 : 0], gx, gy, 1,
            (unsigned)threads[tile], 1, 1, (unsigned)smem[tile][transB ? 1 : 0],
            (CUstream)stream, args, nullptr);  // the legacy entries take the first 9 arguments only
        if (rc != CUDA_SUCCESS) throw std::runtime_error("cuLaunchKernel hybrid failed");
    }

    void destroy() {
        if (mod && p_cuModuleUnload) { p_cuModuleUnload(mod); mod = nullptr; }
        if (lib) { dlclose(lib); lib = nullptr; }
        loaded = false;
    }
};

// FNV-1a over this library's own file: GEMM plans are keyed by it, so a rebuilt library (new kernels, other tile
// code) never reuses plans timed against an older build.
// Test overrides (LOUP_SPLINE_VARIANT=R1|R1p|R4|R4p|R5, LOUP_WPRIME_THREADS=256|512, LOUP_WPRIME_RCV=0|1,
// LOUP_REORDER=0|1): pin a kernel variant or path that tuning or a size threshold would otherwise pick, so the
// gate can cover every one. Parsed by hand (no strtol: the library must link against older glibc).
struct VariantOverrides {
    int spline_var = -1;  // index into tune_rebuild_variants' table
    int wp_threads = 0;
    int wprime_rcv = -1;
    int reorder = -1;
    VariantOverrides() {
        static const char* const names[5] = {"R1", "R1p", "R4", "R4p", "R5"};
        if (const char* s = std::getenv("LOUP_SPLINE_VARIANT"))
            for (int v = 0; v < 5; ++v)
                if (strcmp(s, names[v]) == 0) spline_var = v;
        if (const char* s = std::getenv("LOUP_WPRIME_THREADS")) {
            if (strcmp(s, "256") == 0) wp_threads = 256;
            if (strcmp(s, "512") == 0) wp_threads = 512;
        }
        if (const char* s = std::getenv("LOUP_WPRIME_RCV")) wprime_rcv = (s[0] == '1') ? 1 : (s[0] == '0') ? 0 : -1;
        if (const char* s = std::getenv("LOUP_REORDER")) reorder = (s[0] == '1') ? 1 : (s[0] == '0') ? 0 : -1;
    }
};
static constexpr int kSplineVar = 5;
static constexpr int kSplineVarR[kSplineVar] = {1, 1, 4, 4, 5};
static constexpr bool kSplineVarPair[kSplineVar] = {false, true, false, true, false};

static uint64_t library_build_hash() {
    static uint64_t cached = 0;
    if (cached) return cached;
    Dl_info info{};
    if (dladdr((void*)&library_build_hash, &info) && info.dli_fname) cached = fnv1a_file(info.dli_fname);
    if (!cached) cached = 1;
    return cached;
}

static uint64_t fnv1a64(const void* data, size_t n, uint64_t hsh = 1469598103934665603ULL) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) hsh = (hsh ^ p[i]) * 1099511628211ULL;
    return hsh;
}

namespace {
struct TiledState;
}
struct LoupHandle;
static void release_tiled_state(LoupHandle* h);

struct LoupHandle {
    std::string last_error;  // text of the last failed call on this handle (loup_last_error); empty after a success
    int device_id = 0;
    int l2_bytes = 0;
    bool is_sm120 = false;
    // The own-SGEMM extensions measured on the 5060 Ti (sgemm64 / sgemm32 + split-K candidates, the narrow-N rules,
    // N-fastest rasterisation and the N = 96 kernel of sgemm128): on for sm_120; LOUP_OWN_EXT=0|1 forces them (tests).
    bool own_ext = false;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cublasLtHandle_t cublas_lt = nullptr;
    int* h_reb_pinned = nullptr;
    cudaEvent_t ev_reb[2] = {nullptr, nullptr};
    void* d_cublas_ws = nullptr;
    HybCubin hyb;
    size_t cublas_ws_bytes = 4 << 20;
    // Independent GEMM groups (per l, per species) run on side streams, captured as parallel graph branches, so
    // small products fill the GPU together instead of one under-filled wave after another. Each side stream has its
    // own cuBLASLt workspace. Disabled during the uncaptured autotune pass so timings are not disturbed.
    static constexpr int kSide = 8;
    cudaStream_t side[kSide] = {};
    void* side_ws[kSide] = {};
    cudaEvent_t ev_fork = nullptr;
    cudaEvent_t ev_join[kSide] = {};
    cudaStream_t gemm_stream = nullptr;  // stream sgemm_rm launches on (main stream outside fork groups)
    void* gemm_ws = nullptr;
    bool fork_ok = true;
    struct LtPlan {
        cublasLtMatmulDesc_t desc = nullptr;
        cublasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr;
        cublasLtMatmulAlgo_t algo{};
        bool use_hyb = false;
        int hyb_tile = 0;
        int own_mode = 0;      // 1 = sgemm128, 2 = sgemm64, 3..5 = sgemm32 (split_k = 1, 2, 4)
    };
    std::map<std::tuple<int, int, int, bool, bool>, LtPlan> lt_cache;
    std::map<std::tuple<const float*, int, int>, float*> bt_cache;  // [K, N] copies of transposed weights (weight_kn)
    int spline_rpb_occ = 1;      // receivers per spline_forward_inter1 block by occupancy (choose_spline_rpb)
    int spline_rpb = 1;          // in use: spline_rpb_occ until tune_rebuild_variants has timed the variants
    bool spline_pair = true;     // paired-edge decode in use
    int wp_threads = 256;        // wprime_backward_inter1_rcv block size (256 or 512; tune_rebuild_variants)
    VariantOverrides ovr;        // test overrides from the environment, read at loup_create
    void apply_overrides() {
        if (ovr.spline_var >= 0) {
            spline_rpb = kSplineVarR[ovr.spline_var];
            spline_pair = kSplineVarPair[ovr.spline_var];
        }
        if (ovr.wp_threads) wp_threads = ovr.wp_threads;
    }
    bool use_wprime_rcv() const { return ovr.wprime_rcv >= 0 ? ovr.wprime_rcv == 1 : N >= 512; }
    bool variants_tuned = false;  // tune_rebuild_variants ran for this system
    struct DiskPlan {
        int M, N, K;
        uint8_t transB, beta_nz, use_hyb, hyb_tile, use_own;
        cublasLtMatmulAlgo_t algo;
    };
    std::string plan_cache_path;
    std::map<std::tuple<int, int, int, bool, bool>, DiskPlan> disk_plans;
    // The tiled call (plan task S5-T10): each tile's row count is a GEMM M of its own and changes with every list
    // build, so a decision timed for one M is taken for every M of its size bucket (four per octave, m_bucket) while
    // the handle holds a tile workspace. Kept in memory for the handle's life (not written to the plan file);
    // LOUP_TILE_GEMM_BUCKETS=0 turns it off (every M timed, as the whole-system path).
    std::map<std::tuple<int, int, int, bool, bool>, DiskPlan> bucket_plans;
    bool gemm_buckets = true;
    bool disk_dirty = false;
    void load_disk_plans(int device_id) {
        const char* cdir = std::getenv("LOUP_CACHE_DIR");
        std::string dir;
        if (cdir && cdir[0]) dir = cdir;
        else if (const char* hm = std::getenv("HOME")) dir = std::string(hm) + "/.cache/loupmd";
        else return;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) return;
        std::string gname;
        for (const char* p = prop.name; *p; ++p) {
            unsigned char c = (unsigned char)*p;
            gname.push_back((isalnum(c) || c == '_') ? (char)c : '_');
        }
        char buf[512];
        // The loaded cubin keys the plans too: a replaced cubin file (other tiles, math or beta support) is re-timed.
        char cub[24] = "";
        if (hyb.loaded) snprintf(cub, sizeof(cub), "_c%016llx", (unsigned long long)hyb.cubin_hash);
        snprintf(buf, sizeof(buf), "%s/plans_%s_sm%d%d_lt%zu_hyb%d_%016llx%s.bin",
                 dir.c_str(), gname.c_str(), prop.major, prop.minor,
                 cublasLtGetVersion(), hyb.loaded ? 1 : 0, (unsigned long long)library_build_hash(), cub);
        plan_cache_path = buf;
        if (std::getenv("LOUP_RETUNE")) return;  // re-time everything; the new plans still overwrite the file
        FILE* fp = fopen(plan_cache_path.c_str(), "rb");
        if (!fp) return;
        char magic[8];
        uint32_t cnt = 0;
        if (fread(magic, 1, 8, fp) == 8 && memcmp(magic, "LOUPPLN4", 8) == 0 && fread(&cnt, 4, 1, fp) == 1) {
            for (uint32_t i = 0; i < cnt && i < 4096; ++i) {
                DiskPlan dp{};
                if (fread(&dp, sizeof(dp), 1, fp) != 1) break;
                disk_plans[std::make_tuple(dp.M, dp.N, dp.K, dp.transB != 0, dp.beta_nz != 0)] = dp;
            }
        }
        fclose(fp);
    }
    void save_disk_plans() {
        if (!disk_dirty || plan_cache_path.empty()) return;
        disk_dirty = false;
        for (size_t pos = 1; (pos = plan_cache_path.find('/', pos)) != std::string::npos; ++pos) {
            mkdir(plan_cache_path.substr(0, pos).c_str(), 0755);
        }
        std::string tmp = plan_cache_path + ".tmp." + std::to_string((long long)getpid());
        FILE* fp = fopen(tmp.c_str(), "wb");
        if (!fp) return;
        uint32_t cnt = (uint32_t)disk_plans.size();
        bool ok = fwrite("LOUPPLN4", 1, 8, fp) == 8 && fwrite(&cnt, 4, 1, fp) == 1;
        for (const auto& kv : disk_plans) {
            if (!ok) break;
            ok = fwrite(&kv.second, sizeof(DiskPlan), 1, fp) == 1;
        }
        fclose(fp);
        if (ok) rename(tmp.c_str(), plan_cache_path.c_str());
        else remove(tmp.c_str());
    }
    void clear_lt_cache() {
        for (auto& kv : lt_cache) {
            if (kv.second.layA) cublasLtMatrixLayoutDestroy(kv.second.layA);
            if (kv.second.layB) cublasLtMatrixLayoutDestroy(kv.second.layB);
            if (kv.second.layC) cublasLtMatrixLayoutDestroy(kv.second.layC);
            if (kv.second.desc) cublasLtMatmulDescDestroy(kv.second.desc);
        }
        lt_cache.clear();
    }
    LoupModelGPU model;                       // the specialised OMAT-0-medium path
    std::unique_ptr<loup::GenericModel> gen;  // the generic path (arch=generic bundles)
    bool generic() const { return gen != nullptr; }
    int n_species() const { return gen ? gen->S() : model.S; }
    float cutoff() const { return gen ? gen->r_max() : model.r_max; }
    int species_z(int s) const { return gen ? gen->atomic_numbers()[s] : model.atomic_numbers[s]; }
    double species_e0(int s) const { return gen ? gen->e0()[s] : model.atomic_energies[s]; }

    // System topology (atoms grouped by species 0..S-1)
    int N = 0;
    // An open domain (loup_debug_set_domain): the device rows N .. N + n_ghost - 1 are ghost atoms. They take part
    // in the neighbour build as senders and have rows in what senders provide; they receive nothing. 0 for a
    // periodic system, which is all the public calls accept.
    int n_ghost = 0;
    // The rows (locals and ghosts) the per-system buffers were allocated for: a tile workspace (reshape_domain)
    // takes any smaller domain without allocating again.
    int atom_cap = 0;
    bool tile_workspace = false;  // the handle holds the tiled call's workspace (set up by it, kept between calls)
    // loup_set_execution (kept by loup_set_system): mode, receiver cap (0: the plan's), fraction of free device
    // memory a plan may use; and how the last tiled call ran (loup_get_execution).
    int exec_mode = LOUP_EXEC_AUTO;
    int exec_receivers = 0;
    double exec_fraction = 0.9;
    LoupExecution last_exec{};
    TiledState* tiled_state = nullptr;  // the tiled call's plan, kept lists and state buffers (S5-T06)
    long long tile_list_builds = 0;     // tiled calls that built their lists, since the handle was created
    double skin = 0.5;
    double e0_total = 0.0;
    std::vector<int> perm;          // device (species-grouped, spatially ordered) index -> caller index
    std::vector<double> atom_e0;    // caller index -> the atom's E0 (0 for ghosts): per-atom energies (S6-T08)
    std::vector<int> h_order;       // scratch: new device index -> old device index at an atom reorder
    std::vector<int> species_start; // [S+1]
    std::vector<int> species_count; // [S]
    loup::CellMat64 cell{};
    double cell_in[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // the cell as the caller last gave it (state blob)
    uint64_t species_hash = 0;                        // of the atomic numbers in caller order (state blob)
    double cell_vol = 0.0;
    double cell_heights[3] = {0, 0, 0};
    // The variable cell (cell.hpp), from the handle's first variable-cell run on: the cell on the device, which the
    // captured step reads and writes, and the edges' integer image shifts in the edges' order, kept in the
    // rebuild's scratch arrays (d_tmp_snd, d_tmp_rcv, d_tmp_pairs). Every rebuild refreshes both from the host's
    // cell; the host's cell follows the device's before a rebuild inside such a run and at the end of the call.
    bool var_cell = false;
    loup::CellDev* d_celldev = nullptr;
    // The dispersion term (S6-D04, loup_set_dispersion), added to every whole-system force pass. It reads the cell
    // on the device: d_disp_cell, which follows the host's cell, or the device's own cell inside a variable-cell
    // step (disp_cell_dev while that step is recorded). disp_ready: prepared for this system (the first positions
    // after loup_set_system or loup_set_dispersion prepare it).
    std::unique_ptr<loup::dispersion::Term> disp;
    double* d_disp_cell = nullptr;
    bool disp_ready = false;
    bool disp_cell_dev = false;
    loup::CellDev celldev_host{};
    double cell_v0 = 0.0;          // the volume at loup_set_state / loup_import_state: the cell limits refer to it
    bool cell_ref_stale = false;   // the cell has changed since the list was built (a variable-cell run ended so):
                                   // whoever checks the list without the device's cell rebuilds first
    bool req_var_cell = false;     // this loup_run call goes through the variable-cell step
    double req_cell_rate = 0.0;    // testing: the cell and the positions scaled by 1 + this in every step

    // Atom device buffers
    int* d_species = nullptr;
    double* d_pos64 = nullptr;
    double* d_ref_pos64 = nullptr;
    double* d_vel64 = nullptr;
    double* d_masses = nullptr;
    double* d_inv_masses = nullptr;
    double* d_sqrt_inv_masses = nullptr;
    double* d_rnd_vel = nullptr;
    int* d_need_rebuild = nullptr;
    // Maximum of the squared per-step displacements (float bits) in two windows, [0] current and [1] previous;
    // readers take the larger. disp_window_steps: steps since the current window began.
    unsigned int* d_max_step_disp2 = nullptr;
    int disp_window_steps = 0;
    unsigned long long* d_step_counter = nullptr;
    double* d_langevin_target = nullptr;  // [5]: the thermostat's target of the next step in K, its change per step,
                                          // and that step's noise coefficients (c3, c4, c5 without the mass factor)
    double* d_int_state = nullptr;        // [kIntStateSlots]: the integrator's scalars (see kIntStateSlots)
    double* d_int_work = nullptr;         // [kIntWorkSlots]
    int int_state_kind = -1;              // the LOUP_* kind whose scalars d_int_state holds; -1: none (all zero)
    MinState min_state;                   // loup_minimize
    unsigned long long pos_epoch = 0;     // counts what moves positions or reorders atoms outside loup_minimize
    bool suppress_reorder = false;        // loup_minimize: its vectors are in the atom order of its start
    // Fixed atoms (loup_set_fixed_atoms): their inverse masses on the device are zero, so no kernel moves them.
    int n_fixed = 0;
    std::vector<double> masses_caller;        // caller order: restores the inverse mass of an atom set free
    std::vector<unsigned char> fixed_caller;  // caller order
    // fix_com (LoupIntegrator.fix_com) of the run in progress or run last: three degrees of freedom fewer.
    bool fix_com = false;
    double total_mass = 0.0;
    double* d_rnd_pos = nullptr;  // [N, 3]: the position noise of a fix_com Langevin step (allocated on first use)
    double* d_com = nullptr;      // [6]: mass-weighted sums of the two noise terms, or the total momentum
    int kinetic_dof() const { return 3 * (N - n_fixed) - (fix_com ? 3 : 0); }
    double* d_ke_f64 = nullptr;

    // Neighbour build scratch
    int max_bins = 0;
    double* d_pos_wrapped = nullptr;
    int* d_atom_wrap = nullptr;
    int* d_atom_bin = nullptr;
    unsigned long long* d_ro_keys = nullptr;  // [2N] atom reorder sort keys (in, out)
    int* d_ro_vals = nullptr;                 // [2N] atom reorder indices (in, out)
    double* d_ro_tmp = nullptr;               // [3N] gather scratch
    int* d_bin_counts = nullptr;
    int* d_bin_offsets = nullptr;
    int* d_bin_cursor = nullptr;
    int* d_bin_atom_id = nullptr;
    double* d_bin_pos = nullptr;
    float4* d_bin_pos_f32 = nullptr;
    int* d_bin_wrap = nullptr;
    int* d_bin_sp = nullptr;
    int* d_deg = nullptr;
    unsigned long long* d_edge_total = nullptr;  // the degrees' sum in 64 bits (rebuild_neighbors)
    // The most edges a whole-system list may have: the edge arrays are indexed e * 3 and e * 4 in int.
    // LOUP_DEBUG_EDGE_LIMIT lowers it for the test of the refusal (read by loup_set_system).
    long long edge_limit = 536870911;
    int* d_row_ptr = nullptr;
    int* d_act_deg = nullptr;
    int* d_act_row_ptr = nullptr;
    void* d_cub_tmp = nullptr;
    size_t cub_tmp_bytes = 0;

    // Edge buffers (capacity >= E_cap >= E_graph >= E_skin)
    int E_skin = 0;
    int E_graph = 0;
    int E_cap = 0;
    int* d_senders = nullptr;
    int* d_receivers = nullptr;
    double* d_shift_cart = nullptr;  // [E, 3] shift @ cell of each skin edge (float64), set at rebuild
    int* d_pairs = nullptr;
    int* d_tmp_snd = nullptr;
    int* d_tmp_rcv = nullptr;
    int* d_tmp_shifts = nullptr;
    int* d_tmp_pairs = nullptr;
    uint32_t* d_sort_keys = nullptr;
    int* d_sort_vals = nullptr;
    float4* d_act_vec_len = nullptr;
    int* d_act_senders = nullptr;
    int* d_act_receivers = nullptr;
    int* d_act_pairs = nullptr;
    int* d_act_slab = nullptr;
    float2* d_act_t_sx = nullptr;
    float4* d_tmp_vec_len = nullptr;
    float* d_edge_force_1 = nullptr;

    // Zero-initialized per-step slab + non-zero activation slab
    void* d_zero_slab = nullptr;
    size_t zero_slab_bytes = 0;
    size_t zero_tail_off = 0;  // S0/D0 are fully written by the forward; only [zero_tail_off, end) is zero-filled
    float* d_S0_0 = nullptr;
    float* d_S0_1 = nullptr;
    float* d_S0_2 = nullptr;
    float* d_S0_3 = nullptr;
    float* d_D0 = nullptr;
    float* d_dx0 = nullptr;
    float* d_dx1 = nullptr;
    double* d_energy_f64 = nullptr;
    double* d_forces_f64 = nullptr;
    double* d_virial_grad_f64 = nullptr;

    // Non-zero activation buffers
    float* d_m0_0 = nullptr; float* d_m0_1 = nullptr; float* d_m0_2 = nullptr; float* d_m0_3 = nullptr;
    float* d_h0_0 = nullptr; float* d_h0_1 = nullptr;
    float* d_sc_skip16 = nullptr;
    float* d_x0 = nullptr;   float* d_x1 = nullptr;
    float* d_m1_0 = nullptr; float* d_m1_1 = nullptr; float* d_m1_2 = nullptr; float* d_m1_3 = nullptr;
    float* d_D1 = nullptr;
    float* d_b0 = nullptr;   float* d_b1 = nullptr;   float* d_b2 = nullptr;   float* d_b3 = nullptr;
    float* d_y1 = nullptr;
    float* d_hidden16 = nullptr;  // d(energy)/d(hidden16) [N, 16], written by sc1_fused for the skip backward
    float* d_db0 = nullptr;  float* d_db1 = nullptr;  float* d_db2 = nullptr;  float* d_db3 = nullptr;
    float* d_dD1 = nullptr;
    float* d_dm1_0 = nullptr; float* d_dm1_1 = nullptr; float* d_dm1_2 = nullptr; float* d_dm1_3 = nullptr;
    float* d_dh0_0 = nullptr; float* d_dh0_1 = nullptr;
    float* d_dm0_0 = nullptr; float* d_dm0_1 = nullptr; float* d_dm0_2 = nullptr; float* d_dm0_3 = nullptr;
    float* d_dD0 = nullptr;
    float* d_dS0_0 = nullptr; float* d_dS0_1 = nullptr; float* d_dS0_2 = nullptr; float* d_dS0_3 = nullptr;

    // Graph caches
    cudaGraph_t force_graph = nullptr;
    cudaGraphExec_t force_graph_exec = nullptr;
    int force_graph_stress = -1;
    // The energy-only pass (loup_evaluate2 with LOUP_EVAL_ENERGY_ONLY), kept beside the force graph so that
    // alternating the two (Monte Carlo between MD steps) captures neither again.
    cudaGraph_t energy_graph = nullptr;
    cudaGraphExec_t energy_graph_exec = nullptr;
    double* pass_atom_energy = nullptr;  // a per-atom energy pass (S6-T08) is being recorded: its [N] buffer

    cudaGraph_t md_graph = nullptr;  // pipelined loop: move + check + forces + second half
    cudaGraphExec_t md_graph_exec = nullptr;
    // Unpipelined loop: move + check, then (after a rebuild if flagged) forces + second half.
    cudaGraph_t md_move_graph = nullptr, md_rest_graph = nullptr;
    cudaGraphExec_t md_move_exec = nullptr, md_rest_exec = nullptr;
    int md_graph_kind = -1;  // 0=NVE, 1=Langevin
    double md_graph_dt = 0.0;
    double md_graph_fr = 0.0;
    double md_graph_tau = 0.0;
    double md_graph_mass_temp = 0.0;
    int md_graph_chain = 0;   // 1000 chain_length + chain_loops
    unsigned long long md_graph_seed = 0;
    bool md_graph_thermo = false;  // the graphs carry the two thermo kernels
    bool md_graph_fixcom = false;  // the graphs are the fix_com variant (and its degrees of freedom)
    bool md_graph_varcell = false; // the graphs are the variable-cell step
    double md_graph_cellrate = 0.0;
    double md_graph_npt[7] = {0, 0, 0, 0, 0, 0, 0};  // the barostat's settings the graphs bake in (NptParams)
    bool last_pass_stress = false; // the last force pass carried stress: the virial on the device is current with
                                   // the forces (a barostat's first step needs it)

    // Thermo buffer (thermo.hpp), allocated by the first run that asks for rows. The host knows the step count and
    // the number of unread rows exactly and writes the device's control block at the start of every such run.
    loup::ThermoCtrl* d_thermo_ctrl = nullptr;
    LoupThermoRow* d_thermo_rows = nullptr;
    long long md_steps = 0;       // MD steps completed since the last set_state / import_state
    int thermo_pending = 0;       // unread rows on the device

    void invalidate_graphs() {
        if (force_graph_exec) { cudaGraphExecDestroy(force_graph_exec); force_graph_exec = nullptr; }
        if (force_graph) { cudaGraphDestroy(force_graph); force_graph = nullptr; }
        force_graph_stress = -1;
        if (energy_graph_exec) { cudaGraphExecDestroy(energy_graph_exec); energy_graph_exec = nullptr; }
        if (energy_graph) { cudaGraphDestroy(energy_graph); energy_graph = nullptr; }
        invalidate_md_graphs();
    }
    void invalidate_md_graphs() {
        if (md_graph_exec) { cudaGraphExecDestroy(md_graph_exec); md_graph_exec = nullptr; }
        if (md_graph) { cudaGraphDestroy(md_graph); md_graph = nullptr; }
        if (md_move_exec) { cudaGraphExecDestroy(md_move_exec); md_move_exec = nullptr; }
        if (md_move_graph) { cudaGraphDestroy(md_move_graph); md_move_graph = nullptr; }
        if (md_rest_exec) { cudaGraphExecDestroy(md_rest_exec); md_rest_exec = nullptr; }
        if (md_rest_graph) { cudaGraphDestroy(md_rest_graph); md_rest_graph = nullptr; }
        md_graph_kind = -1;
    }

    int n_rebuilds = 0;     // rebuilds since set_system (atom reorder cadence)
    int* d_order_err = nullptr;
    bool nl_valid = false;  // neighbour list built for the current system (ref positions and shifts are set)
    // d_forces_f64 and d_energy_f64 belong to the positions in d_pos64: the last call was a force evaluation or a
    // completed run, with no new positions since. A run then starts without a force pass of its own.
    bool forces_current = false;
    // Bytes of the buffers that scale with the system, counted where they are allocated (loup_plan_memory scales
    // them to another size): per atom (set_system_impl), per edge of capacity (ensure_edge_capacity), and the
    // [K, N] weight copies of the GEMMs, which do not scale.
    size_t sys_atom_bytes = 0, sys_edge_bytes = 0, weight_copy_bytes = 0;
    size_t sys_fixed_bytes = 0;  // buffers of set_system_impl whose size does not depend on the atom count

    void free_system() {
        release_tiled_state(this);
        if (gen) gen->free_system();
        invalidate_graphs();
        clear_lt_cache();
        spline_rpb = spline_rpb_occ;
        spline_pair = true;
        wp_threads = 256;
        apply_overrides();
        variants_tuned = false;
        nl_valid = false;
        forces_current = false;
        disp_window_steps = 0;
        n_ghost = 0;
        atom_cap = 0;
        tile_workspace = false;
        sys_atom_bytes = 0;
        sys_edge_bytes = 0;
        sys_fixed_bytes = 0;
        n_rebuilds = 0;
        N = 0;
        auto freep = [](auto*& p) {
            if (p) { cudaFree(p); p = nullptr; }
        };
        freep(d_species); freep(d_pos64); freep(d_ref_pos64); freep(d_vel64);
        freep(d_masses); freep(d_inv_masses); freep(d_sqrt_inv_masses); freep(d_rnd_vel);
        freep(d_need_rebuild); freep(d_max_step_disp2); freep(d_step_counter); freep(d_langevin_target);
        freep(d_int_state); freep(d_int_work);
        int_state_kind = -1;
        min_state.release();
        freep(d_rnd_pos); freep(d_com);
        freep(d_ke_f64);
        freep(d_thermo_ctrl); freep(d_thermo_rows);
        md_steps = 0;
        thermo_pending = 0;
        freep(d_pos_wrapped); freep(d_atom_wrap); freep(d_atom_bin);
        freep(d_ro_keys); freep(d_ro_vals); freep(d_ro_tmp); freep(d_order_err);
        freep(d_bin_counts); freep(d_bin_offsets); freep(d_bin_cursor);
        freep(d_bin_atom_id); freep(d_bin_pos); freep(d_bin_pos_f32); freep(d_bin_wrap); freep(d_bin_sp);
        freep(d_deg); freep(d_edge_total); freep(d_row_ptr); freep(d_cub_tmp); cub_tmp_bytes = 0;
        freep(d_act_deg); freep(d_act_row_ptr);

        freep(d_senders); freep(d_receivers); freep(d_shift_cart); freep(d_pairs);
        freep(d_tmp_snd); freep(d_tmp_rcv); freep(d_tmp_shifts); freep(d_tmp_pairs);
        freep(d_sort_keys); freep(d_sort_vals);
        freep(d_act_vec_len);
        freep(d_act_senders); freep(d_act_receivers); freep(d_act_pairs);
        freep(d_act_slab); freep(d_act_t_sx);
        freep(d_tmp_vec_len);
        freep(d_edge_force_1);
        E_cap = 0; E_skin = 0;

        freep(d_zero_slab); zero_slab_bytes = 0;
        freep(d_m0_0); freep(d_m0_1); freep(d_m0_2); freep(d_m0_3);
        freep(d_h0_0); freep(d_h0_1); freep(d_sc_skip16);
        freep(d_x0); freep(d_x1);
        freep(d_m1_0); freep(d_m1_1); freep(d_m1_2); freep(d_m1_3); freep(d_D1);
        freep(d_b0); freep(d_b1); freep(d_b2); freep(d_b3);
        freep(d_y1); freep(d_hidden16);
        freep(d_db0); freep(d_db1); freep(d_db2); freep(d_db3); freep(d_dD1);
        d_dm1_0 = d_dm1_1 = d_dm1_2 = d_dm1_3 = nullptr;  // aliased onto d_m1_*
        d_dh0_0 = d_dh0_1 = nullptr;                      // aliased onto d_h0_*
        freep(d_dm0_0); freep(d_dm0_1); freep(d_dm0_2); freep(d_dm0_3); freep(d_dD0);
        d_dS0_0 = d_dS0_1 = d_dS0_2 = d_dS0_3 = nullptr;  // aliased onto d_S0_*
    }
};

namespace loup {
void sgemm128_init();
void sgemm128(int M, int N, int K, const float* A, const float* B, float* C, float beta, bool n_fastest,
              cudaStream_t stream);
void sgemm64(int M, int N, int K, const float* A, const float* B, float* C, float beta, cudaStream_t stream);
void sgemm32(int M, int N, int K, const float* A, const float* B, float* C, float* ws, float beta, int split_k,
             cudaStream_t stream);
void transpose(int R, int Ccols, const float* in, float* out, cudaStream_t stream);
}

// Scope guards for the tuning paths (a throw from a timed launch must not leak device memory or events).
struct DeviceBufferGuard {
    void* p = nullptr;
    ~DeviceBufferGuard() { if (p) cudaFree(p); }
};
struct EventPairGuard {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    EventPairGuard() {
        CUDA_CHECK_THROW(cudaEventCreate(&e0));
        CUDA_CHECK_THROW(cudaEventCreate(&e1));
    }
    ~EventPairGuard() {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
    }
};

// loup::sgemm128 takes B as [K, N]; a transposed operand of sgemm_rm is always a model weight, so its [K, N] copy is
// made once (on first use, in the uncaptured autotune pass) and kept for the handle's lifetime.
static const float* weight_kn(LoupHandle* h, const float* B, int N, int K, cudaStream_t strm) {
    const auto key = std::make_tuple(B, N, K);
    auto it = h->bt_cache.find(key);
    if (it != h->bt_cache.end()) return it->second;
    float* bt = nullptr;
    CUDA_CHECK_THROW(cudaMalloc(&bt, (size_t)N * K * sizeof(float)));
    h->weight_copy_bytes += (size_t)N * K * sizeof(float);
    loup::transpose(N, K, B, bt, strm);
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    CUDA_CHECK_THROW(cudaStreamIsCapturing(strm, &cap));
    if (cap == cudaStreamCaptureStatusNone) CUDA_CHECK_THROW(cudaStreamSynchronize(strm));
    h->bt_cache.emplace(key, bt);
    return bt;
}

// The legacy handle serves every stream; cublasSetStream resets its workspace to cuBLAS's shared default pool, so the
// lane's own workspace is bound with it (cuBLAS: a workspace per stream, else results may differ from run to run when
// streams run concurrently). Each workspace serves one stream, where cuBLASLt and legacy calls run in order.
static void cublas_bind(LoupHandle* h, cudaStream_t s, void* ws) {
    CUBLAS_CHECK_THROW(cublasSetStream(h->cublas, s));
    CUBLAS_CHECK_THROW(cublasSetWorkspace(h->cublas, ws, h->cublas_ws_bytes));
}

// The size bucket of a GEMM's M in the tiled call: four per octave (M in [2^e, 2^e * 5/4) is one bucket, and so on),
// so a bucket spans at most 25% of its smallest M. Below 64 rows every M is its own bucket.
static inline int m_bucket(int M) {
    if (M < 64) return -M;
    const int e = 31 - __builtin_clz((unsigned)M);
    return e * 4 + (int)(((unsigned)M >> (e - 2)) & 3u);
}

static inline void sgemm_rm(
    LoupHandle* h, bool transA, bool transB,
    int M, int N, int K,
    float alpha, const float* A, int lda,
    const float* B, int ldb,
    float beta, float* C, int ldc
) {
    cudaStream_t strm = h->gemm_stream ? h->gemm_stream : h->stream;
    cublasLtHandle_t lt_h = h->cublas_lt;
    void* ws_p = h->gemm_ws ? h->gemm_ws : h->d_cublas_ws;
    if (transA || lda != K || ldc != N || ldb != (transB ? K : N)) {
        cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublas_bind(h, strm, ws_p);
        CUBLAS_CHECK_THROW(cublasSgemm(h->cublas, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
        cublas_bind(h, h->stream, h->d_cublas_ws);
        return;
    }
    auto key = std::make_tuple(M, N, K, transB, beta != 0.0f);
    const bool own_ok = alpha == 1.0f && (K % 16) == 0 && (N % 4) == 0 && ((((uintptr_t)A | (uintptr_t)B) & 15) == 0);
    auto own_b = [&] { return transB ? weight_kn(h, B, N, K, strm) : B; };
    auto it = h->lt_cache.find(key);
    if (it == h->lt_cache.end()) {
        LoupHandle::LtPlan plan{};
        cublasOperation_t opA_lt = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB_lt = CUBLAS_OP_N;
        CUBLAS_CHECK_THROW(cublasLtMatmulDescCreate(&plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK_THROW(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA_lt, sizeof(opA_lt)));
        CUBLAS_CHECK_THROW(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB_lt, sizeof(opB_lt)));
        if (!transB) CUBLAS_CHECK_THROW(cublasLtMatrixLayoutCreate(&plan.layA, CUDA_R_32F, N, K, N));
        else         CUBLAS_CHECK_THROW(cublasLtMatrixLayoutCreate(&plan.layA, CUDA_R_32F, K, N, K));
        CUBLAS_CHECK_THROW(cublasLtMatrixLayoutCreate(&plan.layB, CUDA_R_32F, K, M, K));
        CUBLAS_CHECK_THROW(cublasLtMatrixLayoutCreate(&plan.layC, CUDA_R_32F, N, M, N));

        cublasLtMatmulPreference_t pref = nullptr;
        CUBLAS_CHECK_THROW(cublasLtMatmulPreferenceCreate(&pref));
        CUBLAS_CHECK_THROW(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &h->cublas_ws_bytes, sizeof(h->cublas_ws_bytes)));
        // No split-K reduction: those algorithms zero a workspace semaphore with a memset node before each GEMM, and
        // the captured memsets left the GPU idle ~120 us per step (5060 Ti, the beta = 1 skip GEMMs); they are also
        // order-nondeterministic. K <= 384 here, so split-K has nothing to gain.
        const uint32_t no_reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
        CUBLAS_CHECK_THROW(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &no_reduction, sizeof(no_reduction)));
        cublasLtMatmulHeuristicResult_t res[32];
        int n_res = 0;
        CUBLAS_CHECK_THROW(cublasLtMatmulAlgoGetHeuristic(
            lt_h, plan.desc, plan.layA, plan.layB, plan.layC, plan.layC, pref, 32, res, &n_res));
        cublasLtMatmulPreferenceDestroy(pref);
        if (n_res <= 0) {
            cublasLtMatrixLayoutDestroy(plan.layA); cublasLtMatrixLayoutDestroy(plan.layB);
            cublasLtMatrixLayoutDestroy(plan.layC); cublasLtMatmulDescDestroy(plan.desc);
            cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublas_bind(h, strm, ws_p);
            CUBLAS_CHECK_THROW(cublasSgemm(h->cublas, opB, CUBLAS_OP_N, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
            cublas_bind(h, h->stream, h->d_cublas_ws);
            return;
        }
        plan.algo = res[0].algo;
        bool hit_disk = false;
        const bool bucketed = h->tile_workspace && h->gemm_buckets;
        const auto bkey = std::make_tuple(m_bucket(M), N, K, transB, beta != 0.0f);
        const LoupHandle::DiskPlan* found = nullptr;
        auto dit = h->disk_plans.find(key);
        if (dit != h->disk_plans.end()) {
            found = &dit->second;
        } else if (bucketed) {
            auto bit = h->bucket_plans.find(bkey);
            if (bit != h->bucket_plans.end()) found = &bit->second;
        }
        if (found) {
            const auto& dp = *found;
            auto splitk_ok = [&](uint8_t m) {
                if (m <= 2) return true;
                if (m > 5) return false;
                const int sk = (m == 3) ? 1 : (m == 4 ? 2 : 4);
                return sk == 1 || (K >= 64 * sk && (size_t)sk * M * N * sizeof(float) <= h->cublas_ws_bytes);
            };
            auto narrow_ok = [&](uint8_t m) {
                if (M <= 1600) return true;
                if (m == 2) return N <= 64 || (M <= 2500 && N <= 160);
                return N <= 32 || (M <= 2500 && N <= 160);
            };
            if (dp.use_own == 1 ||
                (dp.use_own >= 2 && dp.use_own <= 5 && h->own_ext && narrow_ok(dp.use_own) && splitk_ok(dp.use_own))) {
                if (own_ok) {
                    plan.own_mode = dp.use_own;
                    hit_disk = true;
                }
            } else if (dp.use_hyb) {
                if (dp.hyb_tile < h->hyb.ntiles && alpha == 1.0f && h->hyb.takes(beta) && (K & 3) == 0 && (N & 3) == 0 &&
                    ((((uintptr_t)A | (uintptr_t)B) & 15) == 0)) {
                    plan.use_hyb = true;
                    plan.hyb_tile = dp.hyb_tile;
                    hit_disk = true;
                }
            } else {
                cublasLtMatmulHeuristicResult_t chk{};
                if (cublasLtMatmulAlgoCheck(lt_h, plan.desc, plan.layA, plan.layB, plan.layC, plan.layC,
                                            &dp.algo, &chk) == CUBLAS_STATUS_SUCCESS &&
                    chk.state == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= h->cublas_ws_bytes) {
                    plan.algo = dp.algo;
                    plan.use_hyb = false;
                    hit_disk = true;
                }
            }
        }
        cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(strm, &cap_st);
        if (!hit_disk && cap_st == cudaStreamCaptureStatusNone) {
            // An A larger than half the L2 was written by the preceding kernel among tens of MB of other data and
            // streams from DRAM in the force pass, so it is timed with a cold L2 (flushed before every launch): a
            // hot-L2 loop favours tiles that re-read A per column tile (on Ada the 64x32 hybrid tile won hot for the
            // lin1 l=2/3 products and ran ~2x slower in the pass). Smaller operands stay resident and are timed hot.
            DeviceBufferGuard tmpC_buf, flush_buf;
            float* d_tmpC = nullptr;
            void* d_flush = nullptr;
            const size_t flush_bytes = (size_t)std::max(h->l2_bytes, 1 << 20) * 2;
            const bool cold = (size_t)M * K * sizeof(float) > (size_t)h->l2_bytes / 2;
            if (cudaMalloc(&tmpC_buf.p, (size_t)M * N * sizeof(float)) == cudaSuccess &&
                (!cold || cudaMalloc(&flush_buf.p, flush_bytes) == cudaSuccess)) {
                d_tmpC = static_cast<float*>(tmpC_buf.p);
                d_flush = flush_buf.p;
                cudaMemsetAsync(d_tmpC, 0, (size_t)M * N * sizeof(float), strm);
                EventPairGuard ev;
                cudaEvent_t e0 = ev.e0, e1 = ev.e1;
                auto time_min = [&](auto&& launch) -> float {
                    for (int w = 0; w < 2; ++w) launch();
                    float best = 1e30f;
                    const int per_round = cold ? 1 : 5;
                    for (int round = 0; round < 5; ++round) {
                        if (cold) cudaMemsetAsync(d_flush, round, flush_bytes, strm);
                        cudaEventRecord(e0, strm);
                        for (int it_t = 0; it_t < per_round; ++it_t) launch();
                        cudaEventRecord(e1, strm);
                        CUDA_CHECK_THROW(cudaEventSynchronize(e1));
                        float ms = 1e30f;
                        cudaEventElapsedTime(&ms, e0, e1);
                        best = std::min(best, ms / (float)per_round);
                    }
                    return best;
                };
                float best_ms = 1e30f, second_ms = 1e30f;
                auto consider = [&](float ms) {
                    if (ms < best_ms) { second_ms = best_ms; best_ms = ms; return true; }
                    second_ms = std::min(second_ms, ms);
                    return false;
                };
                for (int r = 0; r < n_res; ++r) {
                    cublasStatus_t st = cublasLtMatmul(
                        lt_h, plan.desc, &alpha, B, plan.layA, A, plan.layB, &beta,
                        d_tmpC, plan.layC, d_tmpC, plan.layC, &res[r].algo, ws_p, h->cublas_ws_bytes, strm);
                    if (st != CUBLAS_STATUS_SUCCESS) continue;
                    const float ms = time_min([&] {
                        cublasLtMatmul(
                            lt_h, plan.desc, &alpha, B, plan.layA, A, plan.layB, &beta,
                            d_tmpC, plan.layC, d_tmpC, plan.layC, &res[r].algo, ws_p, h->cublas_ws_bytes, strm);
                    });
                    if (consider(ms)) { plan.algo = res[r].algo; plan.use_hyb = false; }
                }
                if (alpha == 1.0f && h->hyb.takes(beta) && (K & 3) == 0 && (N & 3) == 0 &&
                    ((((uintptr_t)A | (uintptr_t)B | (uintptr_t)d_tmpC) & 15) == 0)) {
                    for (int t = 0; t < h->hyb.ntiles; ++t) {
                        const float ms = time_min([&] { h->hyb.launch(t, transB, M, N, K, A, lda, B, ldb, d_tmpC, ldc, beta, strm); });
                        if (consider(ms)) { plan.use_hyb = true; plan.hyb_tile = t; }
                    }
                }
                if (own_ok && (((uintptr_t)d_tmpC & 15) == 0)) {
                    const float* Bkn = own_b();
                    const float ms128 = time_min([&] { loup::sgemm128(M, N, K, A, Bkn, d_tmpC, beta, h->own_ext, strm); });
                    if (consider(ms128)) { plan.own_mode = 1; plan.use_hyb = false; }
                    if (h->own_ext) {
                        if (M <= 1600 || N <= 64 || (M <= 2500 && N <= 160)) {
                            const float ms64 = time_min([&] { loup::sgemm64(M, N, K, A, Bkn, d_tmpC, beta, strm); });
                            if (consider(ms64)) { plan.own_mode = 2; plan.use_hyb = false; }
                        }
                        if (M <= 1600 || N <= 32 || (M <= 2500 && N <= 160)) {
                            for (int idx = 0; idx < 3; ++idx) {
                                const int sk = 1 << idx;
                                if (sk > 1 && (K < 64 * sk || (size_t)sk * M * N * sizeof(float) > h->cublas_ws_bytes)) continue;
                                const float ms32 = time_min([&] {
                                    loup::sgemm32(M, N, K, A, Bkn, d_tmpC, static_cast<float*>(ws_p), beta, sk, strm);
                                });
                                if (consider(ms32)) { plan.own_mode = 3 + idx; plan.use_hyb = false; }
                            }
                        }
                    }
                }
                if (getenv("LOUP_DEBUG_GEMM")) {
                    const char* tag = plan.own_mode >= 3 ? "own32_"
                                    : plan.own_mode == 2 ? "own64_"
                                    : plan.own_mode == 1 ? "own"
                                    : plan.use_hyb       ? "hyb"
                                                         : "lt";
                    const int sub = plan.own_mode >= 3 ? (1 << (plan.own_mode - 3)) : (plan.use_hyb ? plan.hyb_tile : 0);
                    fprintf(stderr, "gemm M=%d N=%d K=%d transB=%d beta=%g cold=%d -> %s%d %.1f us\n", M, N, K,
                            (int)transB, beta, (int)cold, tag, sub, best_ms * 1e3f);
                }
                LoupHandle::DiskPlan dp{};
                dp.M = M; dp.N = N; dp.K = K;
                dp.transB = transB ? 1 : 0; dp.beta_nz = (beta != 0.0f) ? 1 : 0;
                dp.use_hyb = plan.use_hyb ? 1 : 0; dp.hyb_tile = (uint8_t)plan.hyb_tile;
                dp.use_own = (uint8_t)plan.own_mode;
                dp.algo = plan.algo;
                // A winner within 3% of the runner-up is timing noise: use it in this process but re-time next time
                // instead of pinning it for every later run on this GPU.
                if (best_ms < 0.97f * second_ms) {
                    h->disk_plans[key] = dp;
                    h->disk_dirty = true;
                }
                if (bucketed) h->bucket_plans[bkey] = dp;   // the bucket takes this decision, noise or not
                if (bucketed && getenv("LOUP_DEBUG_GEMM_BUCKETS"))
                    fprintf(stderr, "gemm timed M=%d N=%d K=%d (bucket %d)\n", M, N, K, m_bucket(M));
            }
        }
        it = h->lt_cache.emplace(key, plan).first;
    }
    const auto& plan = it->second;
    if (plan.own_mode && own_ok && (((uintptr_t)C & 15) == 0)) {
        if (plan.own_mode >= 3) {
            loup::sgemm32(M, N, K, A, own_b(), C, static_cast<float*>(ws_p), beta, 1 << (plan.own_mode - 3), strm);
        } else if (plan.own_mode == 2) {
            loup::sgemm64(M, N, K, A, own_b(), C, beta, strm);
        } else {
            loup::sgemm128(M, N, K, A, own_b(), C, beta, h->own_ext, strm);
        }
        return;
    }
    if (plan.use_hyb && alpha == 1.0f && h->hyb.takes(beta) &&
        ((((uintptr_t)A | (uintptr_t)B | (uintptr_t)C) & 15) == 0)) {
        h->hyb.launch(plan.hyb_tile, transB, M, N, K, A, lda, B, ldb, C, ldc, beta, strm);
        return;
    }
    CUBLAS_CHECK_THROW(cublasLtMatmul(
        lt_h, plan.desc, &alpha, B, plan.layA, A, plan.layB, &beta,
        C, plan.layC, C, plan.layC, &plan.algo, ws_p, h->cublas_ws_bytes, strm));
}

static int ensure_edge_capacity(LoupHandle* h, int needed) {
    if (needed <= h->E_cap) return 0;
    int cap = std::max(needed + needed / 4 + 1024, 4096);
    // Capacity 0 and no graphs until every array is reallocated: after a failed allocation a later call grows again
    // instead of running on freed or null arrays.
    h->invalidate_graphs();
    h->E_cap = 0;
    h->sys_edge_bytes = 0;
    auto reallocp = [&](auto*& p, size_t elem_sz) {
        if (p) cudaFree(p);
        p = nullptr;
        h->sys_edge_bytes += (size_t)cap * elem_sz;
        return cudaMalloc(&p, (size_t)cap * elem_sz);
    };
    CUDA_CHECK(reallocp(h->d_senders, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_receivers, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_shift_cart, 3 * sizeof(double)));
    CUDA_CHECK(reallocp(h->d_pairs, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_tmp_snd, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_tmp_rcv, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_tmp_shifts, 3 * sizeof(int)));
    CUDA_CHECK(reallocp(h->d_tmp_pairs, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_sort_keys, sizeof(uint32_t)));
    CUDA_CHECK(reallocp(h->d_sort_vals, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_act_vec_len, sizeof(float4)));
    CUDA_CHECK(reallocp(h->d_act_senders, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_act_receivers, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_act_pairs, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_act_slab, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_act_t_sx, sizeof(float2)));
    CUDA_CHECK(reallocp(h->d_tmp_vec_len, sizeof(float4)));
    CUDA_CHECK(reallocp(h->d_edge_force_1, 3 * sizeof(float)));
    if (h->gen) h->gen->ensure_edges(cap);
    h->E_cap = cap;
    return 0;
}

static int ensure_cub_tmp(LoupHandle* h, size_t bytes) {
    if (bytes <= h->cub_tmp_bytes) return 0;
    // Queued work (a scan earlier in this rebuild, or the in-flight step of the pipelined loop) may still use the old
    // buffer: wait for it before the free. Growth is rare (only the first rebuilds of a system reach it).
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    if (h->d_cub_tmp) cudaFree(h->d_cub_tmp);
    h->d_cub_tmp = nullptr;
    h->cub_tmp_bytes = 0;
    CUDA_CHECK(cudaMalloc(&h->d_cub_tmp, bytes));
    h->cub_tmp_bytes = bytes;
    h->invalidate_graphs();  // the force pass's scan bakes d_cub_tmp into the captured graph
    return 0;
}

// The device's cell from the host's, with the list's reference at this cell (at the end of a rebuild).
static int upload_cell_dev(LoupHandle* h) {
    loup::CellDev& c = h->celldev_host;
    for (int i = 0; i < 9; ++i) {
        c.h[i] = h->cell.c[i];
        c.ref_inv[i] = h->cell.inv[i];
        c.f[i] = c.move[i] = (i % 4 == 0) ? 1.0 : 0.0;
    }
    c.row_scale[0] = c.row_scale[1] = c.row_scale[2] = 1.0;
    c.r_max = (double)h->cutoff();
    c.r_list = c.r_max + h->skin;
    c.thr = 0.5 * h->skin;
    c.volume = h->cell_vol;
    c.v0 = h->cell_v0 > 0.0 ? h->cell_v0 : h->cell_vol;
    c.has_move = 0;
    c.pad = 0;
    CUDA_CHECK(cudaMemcpyAsync(h->d_celldev, &c, sizeof(c), cudaMemcpyHostToDevice, h->stream));
    return 0;
}

// The host's cell from the device's: before a rebuild inside a variable-cell run and at the end of one.
static int download_cell_dev(LoupHandle* h, loup::CellDev* out = nullptr) {
    loup::CellDev c;
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    CUDA_CHECK(cudaMemcpy(&c, h->d_celldev, sizeof(c), cudaMemcpyDeviceToHost));
    bool finite = true;
    for (int i = 0; i < 9; ++i) finite = finite && std::isfinite(c.h[i]);
    if (finite) {
        memcpy(h->cell_in, c.h, sizeof(h->cell_in));
        loup::compute_cell_mat(c.h, h->cell, h->cell_vol, h->cell_heights);
        if (h->d_disp_cell)
            CUDA_CHECK(cudaMemcpy(h->d_disp_cell, h->cell_in, sizeof(h->cell_in), cudaMemcpyHostToDevice));
    }
    if (out) *out = c;
    return 0;
}

// The dispersion term for the handle's system: the atomic numbers in the device's row order (rows stay grouped by
// species, so they do not change at a rebuild) and the host's cell. Outside capture.
static int prepare_dispersion(LoupHandle* h) {
    if (!h->disp || h->N <= 0) return 0;
    if (h->n_ghost > 0) {
        loup::set_error("the dispersion term has no open domains (PLAN.md S6-D09)");
        return -1;
    }
    std::vector<int32_t> z(h->N);
    for (int s = 0; s < h->n_species(); ++s)
        for (int k = h->species_start[s]; k < h->species_start[s + 1]; ++k) z[k] = h->species_z(s);
    if (!h->d_disp_cell) CUDA_CHECK(cudaMalloc(&h->d_disp_cell, 9 * sizeof(double)));
    try {
        h->disp->prepare(h->N, z.data(), h->cell_in, h->stream);
    } catch (const std::exception& e) {
        loup::set_error("%s", e.what());
        return -1;
    }
    CUDA_CHECK(cudaMemcpyAsync(h->d_disp_cell, h->cell_in, sizeof(h->cell_in), cudaMemcpyHostToDevice, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    h->disp_ready = true;
    h->invalidate_graphs();
    return 0;
}

// After the host's cell changed: the term's copy, and the first preparation for a new system.
static int dispersion_cell_changed(LoupHandle* h) {
    if (!h->disp) return 0;
    if (!h->disp_ready) return prepare_dispersion(h);
    CUDA_CHECK(cudaMemcpyAsync(h->d_disp_cell, h->cell_in, sizeof(h->cell_in), cudaMemcpyHostToDevice, h->stream));
    return 0;
}

extern "C++" {
// The sum of n int degrees in 64 bits (*out is zeroed by the caller).
__global__ void sum_degrees_kernel(int n, const int* __restrict__ deg, unsigned long long* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long d = i < n ? (unsigned long long)(unsigned)deg[i] : 0ull;
    for (int o = 16; o > 0; o >>= 1) d += __shfl_down_sync(0xffffffffu, d, o);
    if ((threadIdx.x & 31) == 0 && d) atomicAdd(out, d);
}
}  // extern "C++"

static int rebuild_neighbors(LoupHandle* h) {
    h->nl_valid = false;  // set again only when the rebuild completes
    const int N = h->N;
    const int S = h->n_species();
    const double r_list = (double)h->cutoff() + h->skin;
    const double r_list2 = r_list * r_list;

    int nx = std::max(1, (int)std::floor(h->cell_heights[0] / 3.25));
    int ny = std::max(1, (int)std::floor(h->cell_heights[1] / 3.25));
    int nz = std::max(1, (int)std::floor(h->cell_heights[2] / 3.25));
    while ((int64_t)nx * ny * nz > h->max_bins) {
        if (nx >= ny && nx >= nz && nx > 1) --nx;
        else if (ny >= nx && ny >= nz && ny > 1) --ny;
        else if (nz > 1) --nz;
        else break;
    }
    const int n_bins = nx * ny * nz;
    const int rx = (int)std::ceil(r_list * nx / h->cell_heights[0]);
    const int ry = (int)std::ceil(r_list * ny / h->cell_heights[1]);
    const int rz = (int)std::ceil(r_list * nz / h->cell_heights[2]);

    // An open domain's ghosts (rows N .. NT - 1) are binned and found as senders like every atom; only the first N
    // rows are receivers (the count, fill and sort below).
    const int NT = N + h->n_ghost;
    const int blk = 128;
    const int gridN = (NT + blk - 1) / blk;

    CUDA_CHECK(cudaMemsetAsync(h->d_bin_counts, 0, n_bins * sizeof(int), h->stream));
    loup::wrap_and_bin_kernel<<<gridN, blk, 0, h->stream>>>(
        NT, h->cell, nx, ny, nz, h->d_pos64, h->d_pos_wrapped, h->d_atom_wrap, h->d_atom_bin, h->d_bin_counts);

    // Reorder atoms inside each species group by the Morton code of their bin, once the per-atom sender working set
    // (x0/x1 rows and their gradients, ~4 KB per atom) outgrows L2: then sender rows and sender-gradient atomics stay
    // in L2 only if neighbours in space are neighbours in memory (a shuffled 40,000-atom input is ~30 % slower
    // otherwise). Below that size the caller's order costs nothing and the reorder's extra sync is pure overhead.
    // Atoms near a bin boundary flip bins thermally, so the order is refreshed only on every 16th rebuild (diffusion
    // is slow on that scale). Per-atom state (positions, velocities, forces, masses) is gathered in place, so every
    // buffer address and every captured graph stays valid; only the host permutation changes.
    const bool reorder = h->n_ghost == 0 &&
                         (h->ovr.reorder >= 0 ? h->ovr.reorder == 1 : (int64_t)N * 4096 > (int64_t)h->l2_bytes);
    if (reorder && !h->suppress_reorder && (h->n_rebuilds++ % 16) == 0) {
        unsigned long long* keys_in = h->d_ro_keys;
        unsigned long long* keys_out = h->d_ro_keys + N;
        int* vals_in = h->d_ro_vals;
        int* vals_out = h->d_ro_vals + N;
        loup::atom_order_keys_kernel<<<gridN, blk, 0, h->stream>>>(
            N, nx, ny, nz, h->d_species, h->d_atom_bin, keys_in, vals_in);
        int species_bits = 0;
        while ((1 << species_bits) < S) ++species_bits;
        const int end_bit = 48 + species_bits;  // Morton code in bits 0..47, species above
        size_t sort_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys_in, keys_out, vals_in, vals_out, N, 0, end_bit, h->stream);
        if (ensure_cub_tmp(h, sort_bytes) != 0) return -1;
        cub::DeviceRadixSort::SortPairs(h->d_cub_tmp, sort_bytes, keys_in, keys_out, vals_in, vals_out, N, 0, end_bit, h->stream);
        CUDA_CHECK(cudaMemcpyAsync(h->h_order.data(), vals_out, N * sizeof(int), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        bool identity = true;
        for (int i = 0; i < N && identity; ++i) identity = h->h_order[i] == i;
        if (!identity) {
            auto gather = [&](double* arr, int width) -> int {
                loup::gather_rows_f64_kernel<<<(N * width + 255) / 256, 256, 0, h->stream>>>(
                    N, width, vals_out, arr, h->d_ro_tmp);
                CUDA_CHECK(cudaMemcpyAsync(arr, h->d_ro_tmp, (size_t)N * width * sizeof(double),
                                           cudaMemcpyDeviceToDevice, h->stream));
                return 0;
            };
            // d_rnd_vel: a rebuild in an unpipelined Langevin step falls between the half-kicks, and the second one
            // must reuse the noise the first drew for the same atom.
            if (gather(h->d_pos64, 3) || gather(h->d_vel64, 3) || gather(h->d_forces_f64, 3) ||
                gather(h->d_masses, 1) || gather(h->d_inv_masses, 1) || gather(h->d_sqrt_inv_masses, 1) ||
                gather(h->d_rnd_vel, 3)) return -1;
            std::vector<int> old_perm = h->perm;
            for (int i = 0; i < N; ++i) h->perm[i] = old_perm[h->h_order[i]];
            ++h->pos_epoch;  // per-atom state kept outside this function (loup_minimize's) is in the old order
            CUDA_CHECK(cudaMemsetAsync(h->d_bin_counts, 0, n_bins * sizeof(int), h->stream));
            loup::wrap_and_bin_kernel<<<gridN, blk, 0, h->stream>>>(
                N, h->cell, nx, ny, nz, h->d_pos64, h->d_pos_wrapped, h->d_atom_wrap, h->d_atom_bin, h->d_bin_counts);
        }
    }

    size_t scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, h->d_bin_counts, h->d_bin_offsets, n_bins + 1, h->stream);
    if (ensure_cub_tmp(h, scan_bytes) != 0) return -1;
    // Set bin_counts[n_bins] = 0 so exclusive sum gives total N in bin_offsets[n_bins]
    CUDA_CHECK(cudaMemsetAsync(h->d_bin_counts + n_bins, 0, sizeof(int), h->stream));
    cub::DeviceScan::ExclusiveSum(h->d_cub_tmp, scan_bytes, h->d_bin_counts, h->d_bin_offsets, n_bins + 1, h->stream);
    CUDA_CHECK(cudaMemcpyAsync(h->d_bin_cursor, h->d_bin_offsets, n_bins * sizeof(int), cudaMemcpyDeviceToDevice, h->stream));

    const double max_h = std::max({h->cell_heights[0], h->cell_heights[1], h->cell_heights[2]});
    const float margin = std::max(0.01f, 1e-4f * (float)max_h);
    const float r_lo_f = std::max(0.0f, (float)r_list - margin);
    const float r_hi_f = (float)r_list + margin;
    const float r_lo2_f32 = r_lo_f * r_lo_f;
    const float r_hi2_f32 = r_hi_f * r_hi_f;
    // count / fill: on sm_120 kNeighborLanes lanes per atom (128-thread blocks hold whole lane groups; fill keeps one
    // nscan-int bin-offset table per atom in shared memory); elsewhere, or when that table would exceed 48 KB, the
    // one-thread-per-atom kernels (32-thread blocks), which the lane-group kernels were not measured against.
    const int nscan = (2 * rx + 1) * (2 * ry + 1) * (2 * rz + 1);
    const size_t nb_smem = (size_t)(128 / loup::kNeighborLanes) * nscan * sizeof(int);
    const bool nb_groups = h->is_sm120 && nb_smem <= 48 * 1024;
    const int blkN = nb_groups ? 128 : 32;
    const int gridNb = (int)(((int64_t)N * (nb_groups ? loup::kNeighborLanes : 1) + blkN - 1) / blkN);
    loup::scatter_bins_kernel<<<gridN, blk, 0, h->stream>>>(
        NT, h->d_pos_wrapped, h->d_atom_wrap, h->d_species, h->d_atom_bin, h->d_bin_cursor,
        h->d_bin_atom_id, h->d_bin_pos, h->d_bin_pos_f32, h->d_bin_wrap, h->d_bin_sp);

    if (nb_groups)
        loup::count_neighbors_kernel<<<gridNb, blkN, 0, h->stream>>>(
            N, h->cell, nx, ny, nz, rx, ry, rz, r_list2, r_lo2_f32, r_hi2_f32, h->d_pos_wrapped, h->d_atom_bin,
            h->d_bin_offsets, h->d_bin_atom_id, h->d_bin_pos, h->d_bin_pos_f32, h->d_deg);
    else
        loup::count_neighbors_scalar_kernel<<<gridNb, blkN, 0, h->stream>>>(
            N, h->cell, nx, ny, nz, rx, ry, rz, r_list2, r_lo2_f32, r_hi2_f32, h->d_pos_wrapped, h->d_atom_bin,
            h->d_bin_offsets, h->d_bin_atom_id, h->d_bin_pos, h->d_bin_pos_f32, h->d_deg);

    CUDA_CHECK(cudaMemsetAsync(h->d_deg + N, 0, sizeof(int), h->stream));
    if (!h->d_edge_total) CUDA_CHECK(cudaMalloc(&h->d_edge_total, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemsetAsync(h->d_edge_total, 0, sizeof(unsigned long long), h->stream));
    sum_degrees_kernel<<<(N + 255) / 256, 256, 0, h->stream>>>(N, h->d_deg, h->d_edge_total);
    size_t scanN_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scanN_bytes, h->d_deg, h->d_row_ptr, N + 1, h->stream);
    if (ensure_cub_tmp(h, scanN_bytes) != 0) return -1;
    cub::DeviceScan::ExclusiveSum(h->d_cub_tmp, scanN_bytes, h->d_deg, h->d_row_ptr, N + 1, h->stream);

    int E_new = 0;
    unsigned long long E_total = 0;
    CUDA_CHECK(cudaMemcpyAsync(&E_new, h->d_row_ptr + N, sizeof(int), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(&E_total, h->d_edge_total, sizeof(E_total), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    if (E_total > (unsigned long long)h->edge_limit) {
        h->nl_valid = false;
        loup::set_error("the neighbour list has %llu edges, more than the %lld one call can index; run the system "
                        "in tiles (capacity 'auto' or 'tiled')", E_total, h->edge_limit);
        return -1;
    }

    h->E_skin = E_new;
    if (E_new > 0) {
        if (ensure_edge_capacity(h, E_new) != 0) return -1;
        if (E_new > h->E_graph || h->E_graph > E_new + 32768) {
            h->E_graph = std::min(h->E_cap, ((E_new + 8191) & ~8191));
            if (h->E_graph < E_new) h->E_graph = E_new;
            h->invalidate_graphs();
        }
        if (nb_groups)
            loup::fill_neighbors_kernel<<<gridNb, blkN, nb_smem, h->stream>>>(
                N, S, h->cell, nx, ny, nz, rx, ry, rz, r_list2, r_lo2_f32, r_hi2_f32,
                h->d_pos_wrapped, h->d_atom_wrap, h->d_species, h->d_atom_bin,
                h->d_bin_offsets, h->d_bin_atom_id, h->d_bin_pos, h->d_bin_pos_f32, h->d_bin_wrap, h->d_bin_sp,
                h->d_row_ptr, h->d_tmp_snd, h->d_tmp_rcv, h->d_tmp_shifts, h->d_tmp_pairs,
                h->d_sort_keys, h->d_sort_vals);
        else
            loup::fill_neighbors_scalar_kernel<<<gridNb, blkN, 0, h->stream>>>(
                N, S, h->cell, nx, ny, nz, rx, ry, rz, r_list2, r_lo2_f32, r_hi2_f32,
                h->d_pos_wrapped, h->d_atom_wrap, h->d_species, h->d_atom_bin,
                h->d_bin_offsets, h->d_bin_atom_id, h->d_bin_pos, h->d_bin_pos_f32, h->d_bin_wrap, h->d_bin_sp,
                h->d_row_ptr, h->d_tmp_snd, h->d_tmp_rcv, h->d_tmp_shifts, h->d_tmp_pairs,
                h->d_sort_keys, h->d_sort_vals);

        // Sort each receiver's row by key (sym_pair << 19 | length bits). Contract: for a fixed receiver sym_pair grows
        // with the sender species, so every row ends up grouped by ascending sender species, which the receiver-owned
        // tabled_outer_sum0_forward_kernel relies on (it stores one slot per species run). Rows are ~100 edges:
        // DeviceSegmentedSort sorts small segments in warps/blocks instead of radix passes over every segment;
        // stable, so ties keep fill order.
        size_t sort_bytes = 0;
        // Key scratch: edge_force_1 ([E, 3] floats) is per-step data, free during a rebuild.
        cub::DoubleBuffer<uint32_t> d_keys(h->d_sort_keys, reinterpret_cast<uint32_t*>(h->d_edge_force_1));
        cub::DoubleBuffer<int> d_vals(h->d_sort_vals, reinterpret_cast<int*>(h->d_act_vec_len));
        cub::DeviceSegmentedSort::StableSortPairs(
            nullptr, sort_bytes, d_keys, d_vals, E_new, N, h->d_row_ptr, h->d_row_ptr + 1, h->stream);
        if (ensure_cub_tmp(h, sort_bytes) != 0) return -1;
        cub::DeviceSegmentedSort::StableSortPairs(
            h->d_cub_tmp, sort_bytes, d_keys, d_vals, E_new, N, h->d_row_ptr, h->d_row_ptr + 1, h->stream);
        const int gridE = (E_new + 255) / 256;
        loup::gather_sorted_edges_kernel<<<gridE, 256, 0, h->stream>>>(
            E_new, h->cell, d_vals.Current(), h->d_tmp_snd, h->d_tmp_rcv, h->d_tmp_shifts, h->d_tmp_pairs,
            h->d_senders, h->d_receivers, h->d_shift_cart, h->d_pairs);
        // Enforce the contract above (a changed key or sym_pair would otherwise corrupt S0 silently).
        CUDA_CHECK(cudaMemsetAsync(h->d_order_err, 0, sizeof(int), h->stream));
        loup::check_row_species_order_kernel<<<gridE, 256, 0, h->stream>>>(
            E_new, h->d_senders, h->d_receivers, h->d_species, h->d_order_err);
        int order_err = 0;
        CUDA_CHECK(cudaMemcpyAsync(&order_err, h->d_order_err, sizeof(int), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        if (order_err) {
            loup::set_error("internal error: neighbour rows are not grouped by ascending sender species");
            return -1;
        }
        if (h->var_cell) {  // the integer shifts in the edges' order; the scratch arrays are free until the next rebuild
            loup::sorted_shifts_kernel<<<gridE, 256, 0, h->stream>>>(
                E_new, d_vals.Current(), h->d_tmp_shifts, h->d_tmp_snd, h->d_tmp_rcv, h->d_tmp_pairs);
        }
    } else if (h->E_graph != 0) {
        h->E_graph = 0;
        h->invalidate_graphs();
    }
    CUDA_CHECK(cudaMemcpyAsync(h->d_ref_pos64, h->d_pos64, (size_t)NT * 3 * sizeof(double), cudaMemcpyDeviceToDevice, h->stream));
    CUDA_CHECK(cudaMemsetAsync(h->d_need_rebuild, 0, sizeof(int), h->stream));
    if (h->var_cell && upload_cell_dev(h) != 0) return -1;
    h->cell_ref_stale = false;
    CUDA_CHECK(cudaGetLastError());
    h->nl_valid = true;
    return 0;
}

// Fork: side streams wait for everything queued on the main stream so far.
static void gemm_fork(LoupHandle* h) {
    if (!h->fork_ok) return;
    CUDA_CHECK_THROW(cudaEventRecord(h->ev_fork, h->stream));
    for (int i = 0; i < LoupHandle::kSide; ++i) CUDA_CHECK_THROW(cudaStreamWaitEvent(h->side[i], h->ev_fork, 0));
}

// Route the following sgemm_rm calls to side stream `lane` (calls on one lane stay ordered).
static void gemm_lane(LoupHandle* h, int lane) {
    if (!h->fork_ok) return;
    h->gemm_stream = h->side[lane % LoupHandle::kSide];
    h->gemm_ws = h->side_ws[lane % LoupHandle::kSide];
}

// Join: the main stream waits for every side stream; GEMMs go back to the main stream.
static void gemm_join(LoupHandle* h) {
    if (!h->fork_ok) return;
    for (int i = 0; i < LoupHandle::kSide; ++i) {
        CUDA_CHECK_THROW(cudaEventRecord(h->ev_join[i], h->side[i]));
        CUDA_CHECK_THROW(cudaStreamWaitEvent(h->stream, h->ev_join[i], 0));
    }
    h->gemm_stream = nullptr;
    h->gemm_ws = nullptr;
}

static void launch_wprime_rcv(LoupHandle* h, int threads) {
    const int N = h->N;
    wprime_backward_inter1_rcv<<<N, threads, 0, h->stream>>>(
        N, h->d_act_row_ptr + N, h->d_x0, h->d_x1, h->d_act_vec_len,
        h->d_act_senders, h->d_act_row_ptr,
        h->d_act_slab, h->d_act_t_sx,
        h->d_dm1_0, h->d_dm1_1, h->d_dm1_2, h->d_dm1_3, h->d_dD1,
        h->model.d_cubic_packed8_1, h->model.d_density_table_1, h->model.cubic_bias_shift,
        h->d_dx0, h->d_dx1, h->d_edge_force_1);
}

static void launch_spline(LoupHandle* h, int rpb, bool pair) {
    const int N = h->N;
    auto* kernel = rpb == 5 ? spline_forward_inter1<5, false>
                 : rpb == 4 ? (pair ? spline_forward_inter1<4, true> : spline_forward_inter1<4, false>)
                            : (pair ? spline_forward_inter1<1, true> : spline_forward_inter1<1, false>);
    kernel<<<(N + rpb - 1) / rpb, 128 * rpb, 0, h->stream>>>(
        N, h->d_act_row_ptr + N, h->d_x0, h->d_x1, h->d_act_vec_len, h->d_act_senders, h->d_act_row_ptr,
        h->d_act_slab, h->d_act_t_sx, h->model.d_cubic_packed8_1, h->model.d_density_table_1,
        h->model.cubic_bias_shift,
        h->d_m1_0, h->d_m1_1, h->d_m1_2, h->d_m1_3, h->d_D1);
}

// Kernel variants whose ranking depends on the system, timed once per system at the first MD rebuild (thermalised
// positions, before the rebuild touches the edge arrays) on the last force pass's inputs; the fastest is adopted if it
// beats the current one by >= 2% (LOUP_DEBUG_SPLINE=1 prints the timings). Outputs of the extra launches are
// rewritten (m1, D1, edge forces) or re-zeroed (dx0, dx1) by every force pass before use. The w' launches read
// dm1/dD1, which alias m1/D1 and so hold the spline timings' forward values by then: harmless for timing (same
// shapes, same access pattern), and nothing reads w' outputs from these launches. The time spent here is reported as
// LoupMDStats::tune_ms and excluded from step_ms.
//   spline: the 4- and 5-receiver blocks win where consecutive receivers share table slabs and lose where they rarely
//   do (5060 Ti: SrTiO3 -8%, NbCl5 +6.8%); the paired-edge decode depends on how many consecutive edges share a slab.
//   The ideal starting lattice of a run shows neither.
//   w': 512-thread blocks halve the shared q staging per SM (more L1): Ada -3%, 5060 Ti +0.8% (SrTiO3).
static float time_best_of_3(LoupHandle* h, cudaEvent_t e0, cudaEvent_t e1, const std::function<void()>& launch) {
    float best = 1e30f;
    for (int round = 0; round < 4; ++round) {
        CUDA_CHECK_THROW(cudaEventRecord(e0, h->stream));
        launch();
        CUDA_CHECK_THROW(cudaEventRecord(e1, h->stream));
        CUDA_CHECK_THROW(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK_THROW(cudaEventElapsedTime(&ms, e0, e1));
        if (round > 0) best = std::min(best, ms);  // round 0 warms up
    }
    return best;
}

static void tune_rebuild_variants(LoupHandle* h) {
    if (h->variants_tuned) return;
    // No pair inside r_max + skin: the active-edge arrays were never written. Time later, on a list with edges.
    if (h->E_graph == 0) return;
    if (h->generic()) {  // S64: the generic layers' conv forward variants (GenericModel::tune_conv_variants)
        const int r = h->gen->tune_conv_variants(h->stream);
        if (r < 0) return;  // no force pass recorded for this system yet: time at a later rebuild
        h->variants_tuned = true;
        if (r > 0) h->invalidate_graphs();
        return;
    }
    h->variants_tuned = true;
    EventPairGuard ev;
    cudaEvent_t e0 = ev.e0, e1 = ev.e1;
    bool changed = false;

    float best[kSplineVar] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (h->ovr.spline_var < 0) {
        for (int v = 0; v < kSplineVar; ++v)
            best[v] = time_best_of_3(h, e0, e1, [&] { launch_spline(h, kSplineVarR[v], kSplineVarPair[v]); });
        int cur = 0;
        for (int v = 0; v < kSplineVar; ++v)
            if (kSplineVarR[v] == h->spline_rpb && kSplineVarPair[v] == h->spline_pair) cur = v;
        int fast = cur;
        for (int v = 0; v < kSplineVar; ++v)
            if (best[v] < best[fast]) fast = v;
        if (best[fast] < 0.98f * best[cur]) {
            h->spline_rpb = kSplineVarR[fast];
            h->spline_pair = kSplineVarPair[fast];
            changed = true;
        }
    }

    float wp[2] = {0.0f, 0.0f};
    if (h->use_wprime_rcv() && !h->ovr.wp_threads) {  // the receiver-owned w' (else the edge-tiled kernel)
        wp[0] = time_best_of_3(h, e0, e1, [&] { launch_wprime_rcv(h, 256); });
        wp[1] = time_best_of_3(h, e0, e1, [&] { launch_wprime_rcv(h, 512); });
        const int cur_t = h->wp_threads == 512 ? 1 : 0;
        if (wp[1 - cur_t] < 0.98f * wp[cur_t]) {
            h->wp_threads = cur_t == 1 ? 256 : 512;
            changed = true;
        }
    }
    if (changed) h->invalidate_graphs();
    if (getenv("LOUP_DEBUG_SPLINE"))
        fprintf(stderr, "spline us: R1 %.1f, R1p %.1f, R4 %.1f, R4p %.1f, R5 %.1f -> R=%d paired=%d; w' us: 256 %.1f, 512 %.1f -> %d\n",
                best[0] * 1e3, best[1] * 1e3, best[2] * 1e3, best[3] * 1e3, best[4] * 1e3, h->spline_rpb,
                (int)h->spline_pair, wp[0] * 1e3, wp[1] * 1e3, h->wp_threads);
}

// One sweep of the generic pass (GenericPassInputs::Sweep), for a driver that runs a domain sweep by sweep.
struct DomainSweep {
    int kind = 0, layer = 0;
    const float* adjoint[4] = {nullptr, nullptr, nullptr, nullptr};
};

static_assert(offsetof(loup::CellDev, h) == 0, "record_dispersion reads the device cell at the start of CellDev");

// The dispersion term's share of a whole-system pass, after the model's (which zeroes the accumulators first).
static void record_dispersion(LoupHandle* h, int compute_stress, bool energy_only) {
    if (!h->disp) return;
    if (!h->disp_ready) throw std::logic_error("the dispersion term was not prepared for this system");
    loup::dispersion::Inputs in{h->stream, h->d_pos64,
                                h->disp_cell_dev ? reinterpret_cast<const double*>(h->d_celldev)   // CellDev::h is its first field
                                                  : static_cast<const double*>(h->d_disp_cell),
                                !energy_only, compute_stress != 0 && !energy_only};
    loup::dispersion::Outputs out{h->d_energy_f64, energy_only ? h->pass_atom_energy : nullptr,
                                  energy_only ? nullptr : h->d_forces_f64,
                                  compute_stress && !energy_only ? h->d_virial_grad_f64 : nullptr};
    h->disp->record(in, out);
}

static void record_force_pass(LoupHandle* h, int compute_stress, bool energy_only = false,
                              const DomainSweep* sweep = nullptr) {
    if (h->disp && sweep) throw std::logic_error("the dispersion term has no domain sweeps (PLAN.md S6-D09)");
    const int N = h->N;
    const int E = h->E_graph;
    const int S = h->n_species();
    const float r_max = h->cutoff();

    // S0 and D0 are fully written by the interaction-0 forward; only the accumulators after them need zeroing. A
    // sweep adds to energy, forces and virial: its driver zeroes them once.
    if (!sweep)
        CUDA_CHECK_THROW(cudaMemsetAsync(static_cast<char*>(h->d_zero_slab) + h->zero_tail_off, 0,
                                         h->zero_slab_bytes - h->zero_tail_off, h->stream));
    // No pair within r_max + skin. The specialised model then has no interaction energy at all. A generic model can
    // still have one per atom (residual first block, readout biases, the long-range term: MACE-MH-1 0.56 eV and
    // MACE-POLAR-1-S 0.64 eV per isolated atom), so its pass runs on the empty edge list.
    if (E == 0 && !h->generic()) {
        record_dispersion(h, compute_stress, energy_only);
        return;
    }

    const int gridE64 = (E + 63) / 64;
    const int gridE128 = (E + 127) / 128;

    SplineGrid sg{};
    int sg_psym = 1;
    if (h->generic()) {
        if (const SplineGrid* pg = h->gen->preloc_grid()) {
            sg = *pg;
            sg_psym = h->gen->preloc_pair_sym();
        } else {
            sg.n_bands = 1;
            sg.total_intervals = 1;
            sg.counts[0] = 1;
            sg.ends[0] = r_max;
            sg.inv_h[0] = 1.0f / r_max;
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            sg.starts[i] = h->model.grid1.starts[i];
            sg.ends[i] = h->model.grid1.ends[i];
            sg.inv_h[i] = h->model.grid1.inv_h[i];
            sg.offsets[i] = h->model.grid1.offsets[i];
            sg.counts[i] = h->model.grid1.counts[i];
        }
        sg.n_bands = h->model.grid1.n_bands;
        sg.total_intervals = h->model.grid1.total_intervals;
    }

    // 1. Compute edge vectors, compact active edges (r < r_max) with their spline locator
    const int warpGridN = (N + 3) / 4;  // 4 warps per 128-thread block (1 warp per atom)
    loup::compute_and_count_active_kernel<<<warpGridN, 128, 0, h->stream>>>(
        N, r_max, h->d_pos64, h->d_row_ptr, h->d_senders, h->d_receivers, h->d_shift_cart,
        h->d_tmp_vec_len, h->d_act_deg);
    size_t act_scan_bytes = h->cub_tmp_bytes;
    cub::DeviceScan::ExclusiveSum(
        h->d_cub_tmp, act_scan_bytes, h->d_act_deg, h->d_act_row_ptr, N + 1, h->stream);
    loup::scatter_active_edges_kernel<<<warpGridN, 128, 0, h->stream>>>(
        N, r_max, sg, S, sg_psym, h->d_species, h->d_row_ptr, h->d_act_row_ptr, h->d_senders, h->d_receivers,
        h->d_pairs, h->d_tmp_vec_len, h->d_act_senders, h->d_act_receivers, h->d_act_pairs, h->d_act_vec_len,
        h->d_act_slab, h->d_act_t_sx);

    const int* d_E_act = h->d_act_row_ptr + N;
    if (h->generic()) {
        loup::GenericPassInputs in{};
        in.stream = h->stream;
        in.N = N;
        in.n_ghost = h->n_ghost;
        if (sweep) {
            in.sweep = sweep->kind;
            in.sweep_layer = sweep->layer;
            for (int l = 0; l < 4; ++l) in.adjoint_in[l] = sweep->adjoint[l];
        }
        in.E_dev = d_E_act;
        in.E_cap = h->E_cap;
        in.act_row_ptr = h->d_act_row_ptr;
        in.act_senders = h->d_act_senders;
        in.act_receivers = h->d_act_receivers;
        in.act_vec_len = h->d_act_vec_len;
        in.act_slab = h->d_act_slab;
        in.act_t_sx = h->d_act_t_sx;
        in.species = h->d_species;
        in.species_start = &h->species_start;
        in.species_count = &h->species_count;
        in.energy = h->d_energy_f64;
        in.forces = h->d_forces_f64;
        in.virial = h->d_virial_grad_f64;
        in.compute_stress = compute_stress;
        in.energy_only = energy_only ? 1 : 0;
        in.atom_energy = energy_only ? h->pass_atom_energy : nullptr;
        in.pos64 = h->d_pos64;
        in.gemm = [h](bool transB, int M, int Nc, int K, float alpha, const float* A, int lda, const float* B, int ldb,
                      float beta, float* C, int ldc) {
            sgemm_rm(h, false, transB, M, Nc, K, alpha, A, lda, B, ldb, beta, C, ldc);
        };
        in.gemm_fork = [h] { gemm_fork(h); };
        in.gemm_lane = [h](int lane) { gemm_lane(h, lane); };
        in.gemm_join = [h] { gemm_join(h); };
        h->gen->record(in);
        if (!sweep) record_dispersion(h, compute_stress, energy_only);
        return;
    }

    // 2. Interaction 0: tabled outer sum forward + COMPOSE_SKIP species GEMMs
    loup::tabled_outer_sum0_forward_kernel<<<(N + 3) / 4, 128, 0, h->stream>>>(
        N, S, h->model.pieces0, h->model.n_bins0, h->model.r_lo0, h->model.bin_scale0,
        h->d_act_row_ptr, h->d_act_vec_len, h->d_act_senders, h->d_species,
        h->model.d_coef0, h->model.d_starts0, h->model.d_scales0, h->model.d_bins0,
        h->d_S0_0, h->d_S0_1, h->d_S0_2, h->d_S0_3, h->d_D0);
    const int K0 = S * 64;
    float* S0_ptrs[4] = {h->d_S0_0, h->d_S0_1, h->d_S0_2, h->d_S0_3};
    float* m0_ptrs[4] = {h->d_m0_0, h->d_m0_1, h->d_m0_2, h->d_m0_3};
    const int dims[4] = {1, 3, 5, 7};
    gemm_fork(h);
    for (int s = 0; s < S; ++s) {
        int ns = h->species_count[s];
        if (ns == 0) continue;
        int st = h->species_start[s];
        for (int l = 0; l < 4; ++l) {
            int d = dims[l];
            gemm_lane(h, 4 * s + l);
            sgemm_rm(h, false, false, ns * d, 128, K0,
                     1.0f, S0_ptrs[l] + (size_t)st * d * K0, K0,
                     h->model.d_C0[l] + (size_t)s * K0 * 128, 128,
                     0.0f, m0_ptrs[l] + (size_t)st * d * 128, 128);
        }
    }
    gemm_join(h);
    // 3. Product 0 + Readout 0
    sc0_forward<<<N, 128, 0, h->stream>>>(
        N, h->d_m0_0, h->d_m0_1, h->d_m0_2, h->d_m0_3,
        h->d_D0, h->model.d_w0_factored, h->d_species, h->model.d_w_ro0,
        h->d_h0_0, h->d_h0_1, h->d_energy_f64);

    // 4. Interaction 1: skip_tp + linear_up + spline_forward1 + linear1
    gemm_fork(h);
    for (int s = 0; s < S; ++s) {
        int ns = h->species_count[s];
        if (ns == 0) continue;
        int st = h->species_start[s];
        gemm_lane(h, 2 + (s & 1));
        sgemm_rm(h, false, false, ns, 16, 128,
                 1.0f, h->d_h0_0 + (size_t)st * 128, 128,
                 h->model.d_W_skip1_16 + (size_t)s * 128 * 16, 16,
                 0.0f, h->d_sc_skip16 + (size_t)st * 16, 16);
    }
    gemm_lane(h, 0);
    sgemm_rm(h, false, false, N, 128, 128, 1.0f, h->d_h0_0, 128, h->model.d_W_up_0, 128, 0.0f, h->d_x0, 128);
    gemm_lane(h, 1);
    sgemm_rm(h, false, false, N * 3, 128, 128, 1.0f, h->d_h0_1, 128, h->model.d_W_up_1, 128, 0.0f, h->d_x1, 128);
    gemm_join(h);

    launch_spline(h, h->spline_rpb, h->spline_pair);

    gemm_fork(h);
    gemm_lane(h, 0);
    sgemm_rm(h, false, false, N, 128, 256, 1.0f, h->d_m1_0, 256, h->model.d_W_lin1[0], 128, 0.0f, h->d_b0, 128);
    gemm_lane(h, 1);
    sgemm_rm(h, false, false, N * 3, 128, 384, 1.0f, h->d_m1_1, 384, h->model.d_W_lin1[1], 128, 0.0f, h->d_b1, 128);
    gemm_lane(h, 2);
    sgemm_rm(h, false, false, N * 5, 128, 384, 1.0f, h->d_m1_2, 384, h->model.d_W_lin1[2], 128, 0.0f, h->d_b2, 128);
    gemm_lane(h, 3);
    sgemm_rm(h, false, false, N * 7, 128, 256, 1.0f, h->d_m1_3, 256, h->model.d_W_lin1[3], 128, 0.0f, h->d_b3, 128);
    gemm_join(h);

    // 5-6. Product 1 + Readout 1 forward and backward (one block per atom)
    sc1_fused<<<N, 128, 0, h->stream>>>(
        N, h->d_b0, h->d_b1, h->d_b2, h->d_b3, h->d_D1,
        h->model.d_w1_factored, h->d_species,
        h->d_sc_skip16, h->model.d_W_p1_ro1, h->model.d_W_ro1_2,
        h->d_y1, h->d_hidden16, h->d_energy_f64,
        h->d_db0, h->d_db1, h->d_db2, h->d_db3, h->d_dD1);

    if (energy_only) {
        // Both readouts have added their energy (sc1_fused also ran its part of the backward: not used). What is
        // left of the energy is ZBL, which the full pass takes in its last edge kernel.
        if (h->model.has_zbl)
            loup::zbl_energy0_kernel<<<(E + 255) / 256, 256, 0, h->stream>>>(
                d_E_act, h->model.zbl_r_max_max, h->model.zbl_p, h->model.zbl_c[0], h->model.zbl_c[1],
                h->model.zbl_c[2], h->model.zbl_c[3], h->d_act_vec_len, h->d_act_pairs, h->model.d_zbl_r_cut,
                h->model.d_zbl_inv_a, h->model.d_zbl_pref, h->d_energy_f64);
        record_dispersion(h, compute_stress, true);
        return;
    }
    // 7. Backward: Interaction 1
    gemm_fork(h);
    gemm_lane(h, 0);
    sgemm_rm(h, false, true, N, 256, 128, 1.0f, h->d_db0, 128, h->model.d_W_lin1[0], 128, 0.0f, h->d_dm1_0, 256);
    gemm_lane(h, 1);
    sgemm_rm(h, false, true, N * 3, 384, 128, 1.0f, h->d_db1, 128, h->model.d_W_lin1[1], 128, 0.0f, h->d_dm1_1, 384);
    gemm_lane(h, 2);
    sgemm_rm(h, false, true, N * 5, 384, 128, 1.0f, h->d_db2, 128, h->model.d_W_lin1[2], 128, 0.0f, h->d_dm1_2, 384);
    gemm_lane(h, 3);
    sgemm_rm(h, false, true, N * 7, 256, 128, 1.0f, h->d_db3, 128, h->model.d_W_lin1[3], 128, 0.0f, h->d_dm1_3, 256);
    gemm_join(h);

    if (!h->use_wprime_rcv()) {
        wprime_backward_inter1<<<gridE64, 256, 0, h->stream>>>(
            N, d_E_act, h->d_x0, h->d_x1, h->d_act_vec_len,
            h->d_act_senders, h->d_act_receivers,
            h->d_act_slab, h->d_act_t_sx,
            h->d_dm1_0, h->d_dm1_1, h->d_dm1_2, h->d_dm1_3, h->d_dD1,
            h->model.d_cubic_packed8_1, h->model.d_density_table_1, h->model.cubic_bias_shift,
            h->d_dx0, h->d_dx1, h->d_edge_force_1);
    } else {
        launch_wprime_rcv(h, h->wp_threads);
    }

    // dh0_0 = dx0 @ W_up_0^T, then the skip backward accumulates into it (same lane keeps that order)
    gemm_fork(h);
    gemm_lane(h, 0);
    sgemm_rm(h, false, true, N, 128, 128, 1.0f, h->d_dx0, 128, h->model.d_W_up_0, 128, 0.0f, h->d_dh0_0, 128);
    for (int s = 0; s < S; ++s) {
        int ns = h->species_count[s];
        if (ns == 0) continue;
        int st = h->species_start[s];
        sgemm_rm(h, false, true, ns, 128, 16,
                 1.0f, h->d_hidden16 + (size_t)st * 16, 16,
                 h->model.d_W_skip1_16 + (size_t)s * 128 * 16, 16,
                 1.0f, h->d_dh0_0 + (size_t)st * 128, 128);
    }
    gemm_lane(h, 1);
    sgemm_rm(h, false, true, N * 3, 128, 128, 1.0f, h->d_dx1, 128, h->model.d_W_up_1, 128, 0.0f, h->d_dh0_1, 128);
    gemm_join(h);

    // 8. Backward: Product 0 + Readout 0 + Interaction 0
    sc0_backward<<<N, 128, 0, h->stream>>>(
        N, h->d_m0_0, h->d_m0_1, h->d_m0_2, h->d_m0_3,
        h->d_D0, h->model.d_w0_factored, h->d_species, h->model.d_w_ro0,
        h->d_dh0_0, h->d_dh0_1,
        h->d_dm0_0, h->d_dm0_1, h->d_dm0_2, h->d_dm0_3, h->d_dD0);

    float* dm0_ptrs[4] = {h->d_dm0_0, h->d_dm0_1, h->d_dm0_2, h->d_dm0_3};
    float* dS0_ptrs[4] = {h->d_dS0_0, h->d_dS0_1, h->d_dS0_2, h->d_dS0_3};
    gemm_fork(h);
    for (int s = 0; s < S; ++s) {
        int ns = h->species_count[s];
        if (ns == 0) continue;
        int st = h->species_start[s];
        for (int l = 0; l < 4; ++l) {
            int d = dims[l];
            gemm_lane(h, 4 * s + l);
            sgemm_rm(h, false, true, ns * d, K0, 128,
                     1.0f, dm0_ptrs[l] + (size_t)st * d * 128, 128,
                     h->model.d_C0[l] + (size_t)s * K0 * 128, 128,
                     0.0f, dS0_ptrs[l] + (size_t)st * d * K0, K0);
        }
    }
    gemm_join(h);

    // 8b-9. Interaction 0 tabled backward + fused SH pullback, ZBL, and force/virial assembly
    loup::tabled_outer_sum0_backward_kernel<<<gridE128, 128, 0, h->stream>>>(
        d_E_act, S, h->model.pieces0, h->model.n_bins0, h->model.r_lo0, h->model.bin_scale0, r_max,
        compute_stress, h->model.has_zbl ? h->model.zbl_r_max_max : 0.0f, h->model.zbl_p,
        h->model.zbl_c[0], h->model.zbl_c[1], h->model.zbl_c[2], h->model.zbl_c[3],
        h->d_act_vec_len, h->d_act_senders, h->d_act_receivers, h->d_act_pairs, h->d_species,
        h->model.d_pcoef0, h->model.d_starts0, h->model.d_scales0, h->model.d_bins0,
        h->d_dS0_0, h->d_dS0_1, h->d_dS0_2, h->d_dS0_3, h->d_dD0,
        h->model.d_zbl_r_cut, h->model.d_zbl_inv_a, h->model.d_zbl_pref,
        h->d_edge_force_1, h->d_energy_f64, h->d_forces_f64, h->d_virial_grad_f64);
    record_dispersion(h, compute_stress, false);
}

static int ensure_force_graph(LoupHandle* h, int compute_stress) {
    h->last_pass_stress = compute_stress != 0;  // every launch of the force graph comes through here first
    if (h->force_graph_exec && h->force_graph_stress == compute_stress) return 0;
    if (h->force_graph_exec) { cudaGraphExecDestroy(h->force_graph_exec); h->force_graph_exec = nullptr; }
    if (h->force_graph) { cudaGraphDestroy(h->force_graph); h->force_graph = nullptr; }

    // One uncaptured pass per system so cuBLAS allocates its internal workspaces and sgemm_rm autotunes every GEMM
    // shape (shapes depend only on N and the species counts; lt_cache is cleared with the system).
    if (h->lt_cache.empty()) {
        struct NoFork {  // autotune timings must not overlap other GEMMs
            LoupHandle* h;
            bool saved;
            explicit NoFork(LoupHandle* h_) : h(h_), saved(h_->fork_ok) { h->fork_ok = false; }
            ~NoFork() { h->fork_ok = saved; }
        } no_fork(h);
        record_force_pass(h, compute_stress);
        CUDA_CHECK(cudaGetLastError());  // launch errors (e.g. no kernel image for this GPU) surface here, named
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        h->save_disk_plans();
    }

    CUDA_CHECK(cudaStreamBeginCapture(h->stream, cudaStreamCaptureModeGlobal));
    record_force_pass(h, compute_stress);
    CUDA_CHECK(cudaStreamEndCapture(h->stream, &h->force_graph));
    CUDA_CHECK(cudaGraphInstantiate(&h->force_graph_exec, h->force_graph, nullptr, nullptr, 0));
    h->force_graph_stress = compute_stress;
    return 0;
}

// The energy-only graph. The uncaptured tuning pass is the full one of ensure_force_graph (the energy-only pass has
// a subset of its GEMM shapes).
static int ensure_energy_graph(LoupHandle* h) {
    if (h->energy_graph_exec) return 0;
    if (h->lt_cache.empty() && ensure_force_graph(h, 0) != 0) return -1;
    if (h->energy_graph) { cudaGraphDestroy(h->energy_graph); h->energy_graph = nullptr; }
    CUDA_CHECK(cudaStreamBeginCapture(h->stream, cudaStreamCaptureModeGlobal));
    record_force_pass(h, 0, true);
    CUDA_CHECK(cudaStreamEndCapture(h->stream, &h->energy_graph));
    CUDA_CHECK(cudaGraphInstantiate(&h->energy_graph_exec, h->energy_graph, nullptr, nullptr, 0));
    return 0;
}

// The error channel (loup_last_error). A failing call leaves "<entry point>: <reason>" in its handle, or in this
// thread-local string when there is no handle (loup_create, a NULL handle). The reason is the message a refusal or
// a CUDA_CHECK recorded with loup::set_error, or the text of the exception. Nothing goes to stderr unless
// LOUP_DEBUG_ERRORS is set.
static thread_local std::string tl_null_error;

static void store_error(LoupHandle* h, const char* name) {
    std::string& pending = loup::pending_error();
    std::string msg = std::string(name) + ": " + (pending.empty() ? "failed without a message" : pending);
    pending.clear();
    if (std::getenv("LOUP_DEBUG_ERRORS")) fprintf(stderr, "%s\n", msg.c_str());
    (h ? h->last_error : tl_null_error) = std::move(msg);
}

// Every exported entry point runs through this: a C++ exception (CUDA/cuBLAS failure inside record_force_pass,
// sgemm_rm or HybCubin::launch) must not unwind into the C caller, and a throw during capture must not leave the
// stream in capture mode.
template <class F>
static int guarded(LoupHandle* h, const char* name, F&& f) {
    loup::pending_error().clear();
    try {
        if (!h) loup::set_error("the handle is NULL");
        const int rc = h ? f() : -1;
        if (rc == 0) {
            h->last_error.clear();
            return 0;
        }
        store_error(h, name);
        return rc;
    } catch (const std::exception& e) {
        loup::pending_error() = e.what();
    } catch (...) {
        loup::pending_error() = "unknown exception";
    }
    store_error(h, name);
    if (h && h->stream) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(h->stream, &st) == cudaSuccess && st != cudaStreamCaptureStatusNone) {
            cudaGraph_t g = nullptr;
            cudaStreamEndCapture(h->stream, &g);
            if (g) cudaGraphDestroy(g);
        }
        h->invalidate_graphs();
        h->gemm_stream = nullptr;  // a throw between gemm_lane and gemm_join leaves the routing on a side stream
        h->gemm_ws = nullptr;
    }
    return -1;
}

// Per-arch L1 tuning of the gather-heavy kernels. Their table slabs and sender rows are reused across resident
// blocks, so a smaller shared-memory carveout (more L1) cuts L2 traffic; the driver's default configuration leaves
// most of the unified array as unused shared memory (64 KB for spline_forward_inter1, 100 KB for the i0 kernels).
// The driver maps the carveout percent to a configuration non-proportionally and per kernel, so the percents are
// measured per arch (sm_120: ncu launch__shared_mem_config_size + duration; sm_89: nsys durations, SrTiO3 5k):
//   spline, 1 receiver per block: sm_120 20% (32 KB) 2.28 -> 2.20 ms; sm_89 10% 3.03 -> 2.85 ms.
//   spline, 4 receivers per block: 0% (smallest configuration, 16 KB) on both; sm_120 1.99 ms vs 2.05-2.08 ms at
//     64 KB; sm_89 2.47 ms at 0-10%, 2.51 at 25%, 2.60 at 50% (SrTiO3).
//   sm_120 (RTX 5060 Ti): i0 forward 40% (64 KB) 305 -> 245 us; i0 backward 59% (64 KB) 702 -> 672 us (before its
//                         SH staging; now equal at 64 and 100 KB).
//   sm_89 (RTX 4000 Ada): the i0 kernels are fastest at the driver default.
// Other archs keep the defaults (unmeasured).
static void tune_l1(int device_id) {
    int maj = 0, min = 0;
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, device_id));
    CUDA_CHECK_THROW(cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, device_id));
    auto carve = [](const void* kernel, int pct) {
        CUDA_CHECK_THROW(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, pct));
    };
    if (maj == 12) {
        carve((const void*)spline_forward_inter1<1, true>, 20);
        carve((const void*)spline_forward_inter1<1, false>, 20);
        carve((const void*)spline_forward_inter1<4, true>, 0);
        carve((const void*)spline_forward_inter1<4, false>, 0);
        carve((const void*)spline_forward_inter1<5, false>, 0);
        carve((const void*)loup::tabled_outer_sum0_forward_kernel, 40);
        carve((const void*)loup::tabled_outer_sum0_backward_kernel, 59);
    } else if (maj == 8 && min == 9) {
        carve((const void*)spline_forward_inter1<1, true>, 10);
        carve((const void*)spline_forward_inter1<1, false>, 10);
        carve((const void*)spline_forward_inter1<4, true>, 0);
        carve((const void*)spline_forward_inter1<4, false>, 0);
        carve((const void*)spline_forward_inter1<5, false>, 0);
    }
}

// Default spline variant until tune_rebuild_variants has measured the system (and for evaluate()-only use): 4 receivers
// per block share table slabs between the receivers' warps through L1, but the block holds all 16 warps' registers
// until its slowest receiver finishes, so it is the default only where it keeps the resident warp count of the
// 1-receiver kernel (sm_89: 16 either way; sm_120: 16 vs 20, where it loses ~7% on NbCl5).
static int choose_spline_rpb() {
    int b1 = 0, b4 = 0;
    CUDA_CHECK_THROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b1, spline_forward_inter1<1, true>, 128, 0));
    CUDA_CHECK_THROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b4, spline_forward_inter1<4, true>, 512, 0));
    return b4 * 4 >= b1 ? 4 : 1;
}

extern "C" {

LoupHandle* loup_create(const char* loup_path, int device_id) {
    LoupHandle* h = nullptr;
    loup::pending_error().clear();
    tl_null_error.clear();
    try {
        if (!loup_path) throw std::runtime_error("loup_path is NULL");
        CUDA_CHECK_THROW(cudaSetDevice(device_id));
        h = new LoupHandle();
        h->device_id = device_id;
        CUDA_CHECK_THROW(cudaDeviceGetAttribute(&h->l2_bytes, cudaDevAttrL2CacheSize, device_id));
        {
            int cc_maj = 0, cc_min = 0;
            CUDA_CHECK_THROW(cudaDeviceGetAttribute(&cc_maj, cudaDevAttrComputeCapabilityMajor, device_id));
            CUDA_CHECK_THROW(cudaDeviceGetAttribute(&cc_min, cudaDevAttrComputeCapabilityMinor, device_id));
            h->is_sm120 = (cc_maj == 12 && cc_min == 0);
            h->own_ext = h->is_sm120;
            if (const char* e = std::getenv("LOUP_OWN_EXT"); e && (e[0] == '0' || e[0] == '1')) h->own_ext = e[0] == '1';
        }
        CUDA_CHECK_THROW(cudaStreamCreate(&h->stream));
        CUBLAS_CHECK_THROW(cublasCreate(&h->cublas));
        CUBLAS_CHECK_THROW(cublasLtCreate(&h->cublas_lt));
        CUDA_CHECK_THROW(cudaMalloc(&h->d_cublas_ws, h->cublas_ws_bytes));
        cublas_bind(h, h->stream, h->d_cublas_ws);
        CUDA_CHECK_THROW(cudaMallocHost(&h->h_reb_pinned, 4 * sizeof(int)));  // [0..1] flags, [2..3] step-disp bits
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&h->ev_reb[0], cudaEventDisableTiming));
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&h->ev_reb[1], cudaEventDisableTiming));
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&h->ev_fork, cudaEventDisableTiming));
        for (int i = 0; i < LoupHandle::kSide; ++i) {
            CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&h->side[i], cudaStreamNonBlocking));
            CUDA_CHECK_THROW(cudaEventCreateWithFlags(&h->ev_join[i], cudaEventDisableTiming));
            CUDA_CHECK_THROW(cudaMalloc(&h->side_ws[i], h->cublas_ws_bytes));
        }
        {
            const loup::Bundle bundle(loup_path);
            if (bundle.meta("arch") == "generic") {
                h->gen = std::make_unique<loup::GenericModel>();
                h->gen->load(bundle);
            } else {
                h->model.load(bundle, loup_path);
            }
        }
        // MACE-POLAR-1's charge updates amplify GEMM rounding: with the 3xTF32 tensor-core GEMMs (~4x float32's
        // rounding per product) the RTX 4000 Ada's float32 stress and NaNbO3 forces failed the 3x gate that plain
        // float32 GEMMs pass, so long-range models run on float32 SIMT GEMMs only.
        // LOUP_LR_HYBRID=1 (tests): load the GEMM cubin for long-range models too (e.g. a true 3xTF32 build through
        // LOUP_HYBRID_CUBIN, gated before any adoption).
        const char* lr_hyb = std::getenv("LOUP_LR_HYBRID");
        if (!(h->gen && h->gen->has_lr()) || (lr_hyb && lr_hyb[0] == '1')) h->hyb.init(device_id);
        h->load_disk_plans(device_id);
        if (const char* b = std::getenv("LOUP_TILE_GEMM_BUCKETS")) h->gemm_buckets = std::atoi(b) != 0;
        tune_l1(device_id);
        loup::sgemm128_init();
        h->spline_rpb_occ = h->spline_rpb = choose_spline_rpb();
        h->apply_overrides();
        return h;
    } catch (const std::exception& e) {
        loup::pending_error() = e.what();
    } catch (...) {
        loup::pending_error() = "unknown exception";
    }
    store_error(nullptr, "loup_create");
    if (h) loup_destroy(h);
    return nullptr;
}

void loup_destroy(LoupHandle* h) {
    if (!h) return;
    cudaSetDevice(h->device_id);
    h->free_system();
    h->model.free_gpu();
    if (h->ev_reb[0]) cudaEventDestroy(h->ev_reb[0]);
    if (h->ev_reb[1]) cudaEventDestroy(h->ev_reb[1]);
    if (h->ev_fork) cudaEventDestroy(h->ev_fork);
    for (int i = 0; i < LoupHandle::kSide; ++i) {
        if (h->ev_join[i]) cudaEventDestroy(h->ev_join[i]);
        if (h->side_ws[i]) cudaFree(h->side_ws[i]);
        if (h->side[i]) cudaStreamDestroy(h->side[i]);
    }
    if (h->h_reb_pinned) cudaFreeHost(h->h_reb_pinned);
    if (h->d_celldev) cudaFree(h->d_celldev);
    h->disp.reset();
    if (h->d_disp_cell) cudaFree(h->d_disp_cell);
    h->hyb.destroy();
    if (h->d_cublas_ws) cudaFree(h->d_cublas_ws);
    for (auto& kv : h->bt_cache) cudaFree(kv.second);
    if (h->cublas_lt) cublasLtDestroy(h->cublas_lt);
    if (h->cublas) cublasDestroy(h->cublas);
    if (h->stream) cudaStreamDestroy(h->stream);
    delete h;
}

// Every cudaMalloc inside set_system_impl is counted into the handle's sys_atom_bytes (for loup_plan_memory): the
// macro spares its sixty allocation lines an edit each. It ends right after the function.
static cudaError_t counted_malloc(size_t& counter, void** p, size_t bytes) {
    const cudaError_t err = cudaMalloc(p, bytes);
    if (err == cudaSuccess) counter += bytes;
    return err;
}
#define cudaMalloc(p, bytes) counted_malloc(h->sys_atom_bytes, (void**)(p), (bytes))
// n_ghost > 0 makes the last n_ghost atoms ghosts of an open domain (see LoupHandle::n_ghost): the locals are grouped
// by species as always, the ghosts keep their order behind them, and every per-atom buffer has a row for each.
static int set_system_impl(LoupHandle* h, int n_atoms, const int* atomic_numbers, const double* masses_amu, double skin,
                           int n_ghost = 0) {
    if (!h) return -1;
    if (n_atoms <= 0) { loup::set_error("n_atoms must be positive, got %d", n_atoms); return -1; }
    if (!atomic_numbers) { loup::set_error("atomic_numbers is NULL"); return -1; }
    if (!(skin >= 0.0) || !std::isfinite(skin)) {
        loup::set_error("skin must be finite and not negative, got %g A", skin);
        return -1;
    }
    if (n_ghost < 0 || n_ghost >= n_atoms) {
        loup::set_error("n_ghost must be between 0 and the atom count less one, got %d of %d", n_ghost, n_atoms);
        return -1;
    }
    {
        const char* lim = std::getenv("LOUP_DEBUG_EDGE_LIMIT");
        h->edge_limit = (lim && *lim) ? std::max(1LL, std::atoll(lim)) : 536870911LL;
    }
    if (n_ghost > 0 && !(h->gen && h->gen->sweeps_available())) {
        loup::set_error("an open domain needs a bundle of the generic path without long range, and "
                        "LOUP_CHAIN_FOLD=0 where the skip-chain fold applies");
        return -1;
    }
    const int n_local = n_atoms - n_ghost;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    const int S = h->n_species();

    // Validate everything before touching the handle: on any failure the previous system is gone only if the
    // failure came from allocation, and then N stays 0 so every later call is refused.
    std::vector<int> sp_orig(n_atoms);
    double e0_total = 0.0;
    for (int i = 0; i < n_atoms; ++i) {
        int z = atomic_numbers[i];
        int found = -1;
        for (int s = 0; s < S; ++s) {
            if (h->species_z(s) == z) { found = s; break; }
        }
        if (found < 0) {
            std::string elements;
            for (int s = 0; s < S; ++s) elements += (s ? ", " : "") + std::to_string(h->species_z(s));
            loup::set_error("atomic number %d (atom %d) is not in the bundle, which was exported for %s", z, i,
                            elements.c_str());
            return -1;
        }
        if (masses_amu && !(masses_amu[i] > 0.0 && std::isfinite(masses_amu[i]))) {
            loup::set_error("atom %d has mass %g amu; masses must be finite and positive", i, masses_amu[i]);
            return -1;
        }
        sp_orig[i] = found;
        if (i < n_local) e0_total += h->species_e0(found);
    }
    h->free_system();  // sets N = 0
    h->disp_ready = false;
    h->skin = skin;
    h->e0_total = e0_total;
    h->atom_e0.assign(n_atoms, 0.0);
    for (int i = 0; i < n_local; ++i) h->atom_e0[i] = h->species_e0(sp_orig[i]);
    h->species_hash = fnv1a64(atomic_numbers, (size_t)n_atoms * sizeof(int), fnv1a64(&n_atoms, sizeof(n_atoms)));

    // Build stable species-grouping permutation (the spatial order inside each group is set at every rebuild)
    h->perm.resize(n_atoms);
    h->h_order.resize(n_atoms);
    h->masses_caller.assign(n_atoms, 1.0);
    h->fixed_caller.assign(n_atoms, 0);
    h->n_fixed = 0;
    h->fix_com = false;
    h->total_mass = 0.0;
    h->species_start.assign(S + 1, 0);
    h->species_count.assign(S, 0);
    for (int i = 0; i < n_local; ++i) h->species_count[sp_orig[i]]++;
    for (int s = 0; s < S; ++s) h->species_start[s + 1] = h->species_start[s] + h->species_count[s];
    std::vector<int> cursor = h->species_start;
    std::vector<int> sp_sorted(n_atoms);
    std::vector<double> m_sorted(n_atoms), inv_m_sorted(n_atoms), sim_sorted(n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        int dst = i < n_local ? cursor[sp_orig[i]]++ : i;
        h->perm[dst] = i;
        sp_sorted[dst] = sp_orig[i];
        double m = masses_amu ? masses_amu[i] : 1.0;
        h->masses_caller[i] = m;
        if (i < n_local) h->total_mass += m;
        m_sorted[dst] = m;
        inv_m_sorted[dst] = 1.0 / m;
        sim_sorted[dst] = 1.0 / sqrt(m);
    }

    // Allocate atom device buffers
    CUDA_CHECK(cudaMalloc(&h->d_species, n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(h->d_species, sp_sorted.data(), n_atoms * sizeof(int), cudaMemcpyHostToDevice, h->stream));
    CUDA_CHECK(cudaMalloc(&h->d_pos64, (size_t)n_atoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&h->d_ref_pos64, (size_t)n_atoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&h->d_vel64, (size_t)n_atoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&h->d_masses, n_atoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&h->d_inv_masses, n_atoms * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&h->d_sqrt_inv_masses, n_atoms * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(h->d_masses, m_sorted.data(), n_atoms * sizeof(double), cudaMemcpyHostToDevice, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(h->d_inv_masses, inv_m_sorted.data(), n_atoms * sizeof(double), cudaMemcpyHostToDevice, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(h->d_sqrt_inv_masses, sim_sorted.data(), n_atoms * sizeof(double), cudaMemcpyHostToDevice, h->stream));
    CUDA_CHECK(cudaMalloc(&h->d_rnd_vel, (size_t)n_atoms * 3 * sizeof(double)));
    // Only set_state writes velocities: an evaluate-then-run sequence starts from rest, not from stale memory.
    CUDA_CHECK(cudaMemsetAsync(h->d_vel64, 0, (size_t)n_atoms * 3 * sizeof(double), h->stream));
    CUDA_CHECK(cudaMemsetAsync(h->d_rnd_vel, 0, (size_t)n_atoms * 3 * sizeof(double), h->stream));
    CUDA_CHECK(cudaMalloc(&h->d_need_rebuild, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_max_step_disp2, 2 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemsetAsync(h->d_max_step_disp2, 0, 2 * sizeof(unsigned int), h->stream));
    CUDA_CHECK(cudaMalloc(&h->d_step_counter, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemsetAsync(h->d_step_counter, 0, sizeof(unsigned long long), h->stream));
    CUDA_CHECK(cudaMalloc(&h->d_langevin_target, 5 * sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(h->d_langevin_target, 0, 5 * sizeof(double), h->stream));
    CUDA_CHECK(cudaMalloc(&h->d_int_state, kIntStateSlots * sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(h->d_int_state, 0, kIntStateSlots * sizeof(double), h->stream));
    CUDA_CHECK(cudaMalloc(&h->d_int_work, kIntWorkSlots * sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(h->d_int_work, 0, kIntWorkSlots * sizeof(double), h->stream));
    CUDA_CHECK(cudaMalloc(&h->d_ke_f64, sizeof(double)));

    // Neighbour scratch
    h->max_bins = 262144;
    CUDA_CHECK(cudaMalloc(&h->d_pos_wrapped, (size_t)n_atoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&h->d_atom_wrap, (size_t)n_atoms * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_atom_bin, n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_ro_keys, (size_t)n_atoms * 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&h->d_ro_vals, (size_t)n_atoms * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_ro_tmp, (size_t)n_atoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&h->d_order_err, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_bin_counts, (h->max_bins + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_bin_offsets, (h->max_bins + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_bin_cursor, (h->max_bins + 1) * sizeof(int)));
    // The three bin arrays above have a fixed size: loup_plan_memory must not scale them with the atom count.
    h->sys_fixed_bytes = 3 * (size_t)(h->max_bins + 1) * sizeof(int);
    h->sys_atom_bytes -= h->sys_fixed_bytes;
    CUDA_CHECK(cudaMalloc(&h->d_bin_atom_id, n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_bin_pos, (size_t)n_atoms * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&h->d_bin_pos_f32, (size_t)n_atoms * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&h->d_bin_wrap, (size_t)n_atoms * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_bin_sp, n_atoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_deg, (n_atoms + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_row_ptr, (n_atoms + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_act_deg, (n_atoms + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_act_row_ptr, (n_atoms + 1) * sizeof(int)));
    {
        size_t scanN_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scanN_bytes, h->d_act_deg, h->d_act_row_ptr, n_atoms + 1, h->stream);
        if (ensure_cub_tmp(h, scanN_bytes) != 0) return -1;
    }

    if (h->generic()) {
        // energy, forces and virial only: the generic path keeps its own buffers
        auto align256 = [](size_t b) { return (b + 255) & ~size_t(255); };
        const size_t off_f64 = align256(sizeof(double));
        const size_t off_v64 = off_f64 + align256((size_t)n_atoms * 3 * sizeof(double));
        h->zero_slab_bytes = off_v64 + align256(6 * sizeof(double));
        h->zero_tail_off = 0;
        CUDA_CHECK(cudaMalloc(&h->d_zero_slab, h->zero_slab_bytes));
        auto* base = static_cast<char*>(h->d_zero_slab);
        h->d_energy_f64 = reinterpret_cast<double*>(base);
        h->d_forces_f64 = reinterpret_cast<double*>(base + off_f64);
        h->d_virial_grad_f64 = reinterpret_cast<double*>(base + off_v64);
        h->gen->alloc_system(n_atoms);
        if (h->E_cap > 0) h->gen->ensure_edges(h->E_cap);
        h->N = n_local;
        h->n_ghost = n_ghost;
        h->atom_cap = n_atoms;
        return 0;
    }

    // Zero-initialized per-step slab
    const size_t K0 = (size_t)S * 64;
    const size_t nN = (size_t)n_atoms;
    auto align256 = [](size_t b) { return (b + 255) & ~size_t(255); };
    size_t off_S0_0 = 0;
    size_t off_S0_1 = off_S0_0 + align256(nN * 1 * K0 * sizeof(float));
    size_t off_S0_2 = off_S0_1 + align256(nN * 3 * K0 * sizeof(float));
    size_t off_S0_3 = off_S0_2 + align256(nN * 5 * K0 * sizeof(float));
    size_t off_D0   = off_S0_3 + align256(nN * 7 * K0 * sizeof(float));
    size_t off_dx0  = off_D0   + align256(nN * sizeof(float));
    size_t off_dx1  = off_dx0  + align256(nN * 128 * sizeof(float));
    size_t off_en   = off_dx1  + align256(nN * 3 * 128 * sizeof(float));
    size_t off_f64  = off_en   + align256(sizeof(double));
    size_t off_v64  = off_f64  + align256(nN * 3 * sizeof(double));
    h->zero_slab_bytes = off_v64 + align256(6 * sizeof(double));
    h->zero_tail_off = off_dx0;
    CUDA_CHECK(cudaMalloc(&h->d_zero_slab, h->zero_slab_bytes));
    auto* base = static_cast<char*>(h->d_zero_slab);
    h->d_S0_0 = reinterpret_cast<float*>(base + off_S0_0);
    h->d_S0_1 = reinterpret_cast<float*>(base + off_S0_1);
    h->d_S0_2 = reinterpret_cast<float*>(base + off_S0_2);
    h->d_S0_3 = reinterpret_cast<float*>(base + off_S0_3);
    h->d_D0   = reinterpret_cast<float*>(base + off_D0);
    h->d_dx0  = reinterpret_cast<float*>(base + off_dx0);
    h->d_dx1  = reinterpret_cast<float*>(base + off_dx1);
    h->d_energy_f64 = reinterpret_cast<double*>(base + off_en);
    h->d_forces_f64 = reinterpret_cast<double*>(base + off_f64);
    h->d_virial_grad_f64 = reinterpret_cast<double*>(base + off_v64);

    // Non-zero activation buffers
    auto allocf = [&](float*& p, size_t cnt) { return cudaMalloc(&p, cnt * sizeof(float)); };
    CUDA_CHECK(allocf(h->d_m0_0, nN * 1 * 128));
    CUDA_CHECK(allocf(h->d_m0_1, nN * 3 * 128));
    CUDA_CHECK(allocf(h->d_m0_2, nN * 5 * 128));
    CUDA_CHECK(allocf(h->d_m0_3, nN * 7 * 128));
    CUDA_CHECK(allocf(h->d_h0_0, nN * 128));
    CUDA_CHECK(allocf(h->d_h0_1, nN * 3 * 128));
    CUDA_CHECK(allocf(h->d_sc_skip16, nN * 16));
    CUDA_CHECK(allocf(h->d_x0, nN * 128));
    CUDA_CHECK(allocf(h->d_x1, nN * 3 * 128));
    CUDA_CHECK(allocf(h->d_m1_0, nN * 1 * 256));
    CUDA_CHECK(allocf(h->d_m1_1, nN * 3 * 384));
    CUDA_CHECK(allocf(h->d_m1_2, nN * 5 * 384));
    CUDA_CHECK(allocf(h->d_m1_3, nN * 7 * 256));
    CUDA_CHECK(allocf(h->d_D1, nN));
    CUDA_CHECK(allocf(h->d_b0, nN * 1 * 128));
    CUDA_CHECK(allocf(h->d_b1, nN * 3 * 128));
    CUDA_CHECK(allocf(h->d_b2, nN * 5 * 128));
    CUDA_CHECK(allocf(h->d_b3, nN * 7 * 128));
    CUDA_CHECK(allocf(h->d_y1, nN * 128));
    CUDA_CHECK(allocf(h->d_hidden16, nN * 16));
    CUDA_CHECK(allocf(h->d_db0, nN * 1 * 128));
    CUDA_CHECK(allocf(h->d_db1, nN * 3 * 128));
    CUDA_CHECK(allocf(h->d_db2, nN * 5 * 128));
    CUDA_CHECK(allocf(h->d_db3, nN * 7 * 128));
    CUDA_CHECK(allocf(h->d_dD1, nN));
    // Alias backward buffers whose forward counterparts are dead before backward writes them (~34 KB/atom saved).
    h->d_dm1_0 = h->d_m1_0;
    h->d_dm1_1 = h->d_m1_1;
    h->d_dm1_2 = h->d_m1_2;
    h->d_dm1_3 = h->d_m1_3;
    h->d_dh0_0 = h->d_h0_0;
    h->d_dh0_1 = h->d_h0_1;
    CUDA_CHECK(allocf(h->d_dm0_0, nN * 1 * 128));
    CUDA_CHECK(allocf(h->d_dm0_1, nN * 3 * 128));
    CUDA_CHECK(allocf(h->d_dm0_2, nN * 5 * 128));
    CUDA_CHECK(allocf(h->d_dm0_3, nN * 7 * 128));
    CUDA_CHECK(allocf(h->d_dD0, nN));
    h->d_dS0_0 = h->d_S0_0;
    h->d_dS0_1 = h->d_S0_1;
    h->d_dS0_2 = h->d_S0_2;
    h->d_dS0_3 = h->d_S0_3;
    h->N = n_atoms;  // last: a failure above leaves N = 0
    return 0;
}
#undef cudaMalloc

// Upload positions (caller order) and the cell. Returns 1 if the cell changed (the neighbour list's Cartesian
// shifts depend on it, so the list must be rebuilt), 0 if not, -1 on error.
// The engine is always fully periodic: the cell must be finite and non-degenerate (ASE's zero cell for a molecule
// is rejected; put molecules in a box larger than 2 (r_max + skin)), and positions finite.
static bool valid_input(int n, const double* positions, const double* cell, const double* velocities) {
    if (!positions || !cell) {
        loup::set_error("%s is NULL", positions ? "cell" : "positions");
        return false;
    }
    for (int i = 0; i < 9; ++i) {
        if (!std::isfinite(cell[i])) {
            loup::set_error("non-finite cell entry [%d, %d]", i / 3, i % 3);
            return false;
        }
    }
    const double det = cell[0] * (cell[4] * cell[8] - cell[5] * cell[7]) - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6]) +
                       cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
    double scale = 0.0;
    for (int i = 0; i < 9; ++i) scale = std::max(scale, std::fabs(cell[i]));
    if (!(std::fabs(det) > 1e-9 * scale * scale * scale) || scale == 0.0) {
        loup::set_error("singular or zero cell (det = %g A^3); the engine is fully periodic", det);
        return false;
    }
    for (size_t i = 0; i < (size_t)n * 3; ++i) {
        if (!std::isfinite(positions[i])) {
            loup::set_error("non-finite position (atom %zu)", i / 3);
            return false;
        }
        if (velocities && !std::isfinite(velocities[i])) {
            loup::set_error("non-finite velocity (atom %zu)", i / 3);
            return false;
        }
    }
    return true;
}

static int upload_positions(LoupHandle* h, const double* positions, const double* cell) {
    const int NT = h->N + h->n_ghost;
    if (!valid_input(NT, positions, cell, nullptr)) return -1;
    h->forces_current = false;
    ++h->pos_epoch;
    memcpy(h->cell_in, cell, sizeof(h->cell_in));
    loup::CellMat64 old_cell = h->cell;
    loup::compute_cell_mat(cell, h->cell, h->cell_vol, h->cell_heights);
    const int cell_changed = memcmp(&old_cell, &h->cell, sizeof(old_cell)) != 0 ? 1 : 0;
    if (h->gen && h->gen->set_cell(cell)) h->invalidate_graphs();
    if (dispersion_cell_changed(h) != 0) return -1;

    std::vector<double> p_sorted((size_t)NT * 3);
    for (int dst = 0; dst < NT; ++dst) {
        int src = h->perm[dst];
        p_sorted[dst * 3 + 0] = positions[src * 3 + 0];
        p_sorted[dst * 3 + 1] = positions[src * 3 + 1];
        p_sorted[dst * 3 + 2] = positions[src * 3 + 2];
    }
    CUDA_CHECK(cudaMemcpyAsync(h->d_pos64, p_sorted.data(), (size_t)NT * 3 * sizeof(double), cudaMemcpyHostToDevice, h->stream));
    return cell_changed;
}

// The refusal shared by every call that needs a system.
static bool has_system(const LoupHandle* h) {
    if (h && h->N > 0 && h->n_ghost == 0) return true;
    if (h && h->n_ghost > 0)
        loup::set_error("the handle holds an open domain (loup_debug_set_domain): only loup_debug_evaluate_domain "
                        "runs on it");
    else
        loup::set_error("no system is set (call loup_set_system first)");
    return false;
}

static int set_state_impl(LoupHandle* h, const double* positions, const double* velocities, const double* cell) {
    if (!has_system(h)) return -1;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    if (!valid_input(h->N, positions, cell, velocities)) return -1;
    if (upload_positions(h, positions, cell) < 0) return -1;
    h->cell_v0 = h->cell_vol;  // a new trajectory: the cell limits of a variable-cell run refer to this volume
    CUDA_CHECK(cudaMemsetAsync(h->d_max_step_disp2, 0, 2 * sizeof(unsigned int), h->stream));
    CUDA_CHECK(cudaMemsetAsync(h->d_int_state, 0, kIntStateSlots * sizeof(double), h->stream));  // a new trajectory
    h->int_state_kind = -1;
    h->disp_window_steps = 0;
    h->md_steps = 0;
    h->thermo_pending = 0;
    if (velocities) {
        std::vector<double> v_sorted((size_t)h->N * 3);
        for (int dst = 0; dst < h->N; ++dst) {
            int src = h->perm[dst];
            v_sorted[dst * 3 + 0] = velocities[src * 3 + 0];
            v_sorted[dst * 3 + 1] = velocities[src * 3 + 1];
            v_sorted[dst * 3 + 2] = velocities[src * 3 + 2];
            if (h->fixed_caller[src]) v_sorted[dst * 3 + 0] = v_sorted[dst * 3 + 1] = v_sorted[dst * 3 + 2] = 0.0;
        }
        CUDA_CHECK(cudaMemcpyAsync(h->d_vel64, v_sorted.data(), (size_t)h->N * 3 * sizeof(double), cudaMemcpyHostToDevice, h->stream));
    } else {
        CUDA_CHECK(cudaMemsetAsync(h->d_vel64, 0, (size_t)h->N * 3 * sizeof(double), h->stream));
    }
    return rebuild_neighbors(h);
}

// loup_set_fixed_atoms: a fixed atom gets zero inverse mass (and zero 1/sqrt(m), the thermostat's noise factor) and
// zero velocity, so every integrator leaves it where it is without knowing about it. Masses stay, so the kinetic
// energy of the others is unchanged.
static int set_fixed_impl(LoupHandle* h, const int* fixed) {
    if (!has_system(h)) return -1;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    const int N = h->N;
    int n_fixed = 0;
    if (fixed) {
        for (int i = 0; i < N; ++i) n_fixed += fixed[i] != 0;
    }
    if (n_fixed == N) {
        loup::set_error("every atom is fixed (%d atoms): nothing would move", N);
        return -1;
    }
    std::vector<double> inv_m((size_t)N), sim((size_t)N), vel((size_t)N * 3);
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    CUDA_CHECK(cudaMemcpy(vel.data(), h->d_vel64, vel.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (int dst = 0; dst < N; ++dst) {
        const int src = h->perm[dst];
        const bool held = fixed && fixed[src] != 0;
        const double m = h->masses_caller[src];
        inv_m[dst] = held ? 0.0 : 1.0 / m;
        sim[dst] = held ? 0.0 : 1.0 / sqrt(m);
        if (held) vel[(size_t)dst * 3 + 0] = vel[(size_t)dst * 3 + 1] = vel[(size_t)dst * 3 + 2] = 0.0;
    }
    CUDA_CHECK(cudaMemcpy(h->d_inv_masses, inv_m.data(), inv_m.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_sqrt_inv_masses, sim.data(), sim.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h->d_vel64, vel.data(), vel.size() * sizeof(double), cudaMemcpyHostToDevice));
    for (int i = 0; i < N; ++i) h->fixed_caller[i] = fixed && fixed[i] != 0;
    h->n_fixed = n_fixed;
    ++h->pos_epoch;  // loup_minimize's gradient leaves fixed atoms out
    h->invalidate_md_graphs();  // the thermo row's degrees of freedom are baked into the captured step
    return 0;
}

// The TF32 + FP16 hybrid GEMMs (gemm-hybrid-cubin) overflow to inf when a GEMM operand element reaches 32,752. A
// non-finite energy, force or stress with the hybrid loaded drops it for this engine: later GEMM plans are re-tuned
// without it (disk plans that chose it are skipped while it is unloaded) and the captured graphs are re-recorded.
// The notice goes to stderr: the call that triggered it goes on and can succeed, so it is not an error of the error
// channel.
static bool drop_hybrid(LoupHandle* h) {
    if (!h->hyb.loaded) return false;
    fprintf(stderr, "loupmd: non-finite energy / forces with the tensor-core GEMM cubin %s: disabled for this engine "
                    "(GEMMs re-tuned without it; the TF32+FP16 hybrid builds overflow at |operand| >= 32,752)\n",
            h->hyb.path_loaded.c_str());
    h->hyb.loaded = false;
    h->clear_lt_cache();
    // The plans in memory and the file were timed with the hybrid: drop them and save nothing from this engine, so a
    // later process (hybrid loaded) still times the hybrid for these shapes.
    h->disk_plans.clear();
    h->bucket_plans.clear();
    h->disk_dirty = false;
    h->plan_cache_path.clear();
    h->invalidate_graphs();
    return true;
}

// What every evaluation starts with: the new positions and cell on the device and a neighbour list that is complete
// for them.
static int prepare_evaluate(LoupHandle* h, const double* positions, const double* cell) {
    if (!has_system(h)) return -1;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    // Positions only: MD velocities are kept. The neighbour list is rebuilt only when the cell changed or an atom
    // moved more than skin/2 since the last build; the force graph is reused unless the rebuild outgrew it.
    const int cell_changed = upload_positions(h, positions, cell);
    if (cell_changed < 0) return -1;
    bool rebuild = cell_changed != 0 || !h->nl_valid || h->cell_ref_stale;
    if (!rebuild) {
        const double half_skin = 0.5 * h->skin;
        CUDA_CHECK(cudaMemsetAsync(h->d_need_rebuild, 0, sizeof(int), h->stream));
        loup::check_skin_kernel<<<(h->N + 255) / 256, 256, 0, h->stream>>>(
            h->N, half_skin, 0, 0, h->d_pos64, h->d_ref_pos64, h->d_max_step_disp2, h->d_need_rebuild);
        int flag = 0;
        CUDA_CHECK(cudaMemcpyAsync(&flag, h->d_need_rebuild, sizeof(int), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        rebuild = flag != 0;
    }
    if (rebuild && rebuild_neighbors(h) != 0) return -1;
    return 0;
}

static int evaluate_once(
    LoupHandle* h, const double* positions, const double* cell, int compute_stress,
    double* out_energy_ev, double* out_forces, double* out_stress, bool* finite
) {
    if (prepare_evaluate(h, positions, cell) != 0) return -1;
    if (ensure_force_graph(h, compute_stress) != 0) return -1;

    CUDA_CHECK(cudaGraphLaunch(h->force_graph_exec, h->stream));
    double inter_e = 0.0;
    std::vector<double> f_sorted((size_t)h->N * 3);
    double v_grad[6] = {0};
    CUDA_CHECK(cudaMemcpyAsync(&inter_e, h->d_energy_f64, sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(f_sorted.data(), h->d_forces_f64, (size_t)h->N * 3 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    if (compute_stress && out_stress) {
        CUDA_CHECK(cudaMemcpyAsync(v_grad, h->d_virial_grad_f64, 6 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(h->stream));

    *finite = std::isfinite(inter_e);
    for (size_t i = 0; *finite && i < f_sorted.size(); ++i) *finite = std::isfinite(f_sorted[i]);
    for (int i = 0; *finite && i < 6; ++i) *finite = std::isfinite(v_grad[i]);
    h->forces_current = *finite;  // a run may start from these forces only if they are usable
    if (out_energy_ev) *out_energy_ev = h->e0_total + inter_e;
    if (out_forces) {
        for (int dst = 0; dst < h->N; ++dst) {
            int orig = h->perm[dst];
            out_forces[orig * 3 + 0] = f_sorted[dst * 3 + 0];
            out_forces[orig * 3 + 1] = f_sorted[dst * 3 + 1];
            out_forces[orig * 3 + 2] = f_sorted[dst * 3 + 2];
        }
    }
    if (compute_stress && out_stress) {
        double inv_v = 1.0 / h->cell_vol;
        out_stress[0] = v_grad[0] * inv_v; out_stress[1] = v_grad[1] * inv_v; out_stress[2] = v_grad[2] * inv_v;
        out_stress[3] = v_grad[1] * inv_v; out_stress[4] = v_grad[3] * inv_v; out_stress[5] = v_grad[4] * inv_v;
        out_stress[6] = v_grad[2] * inv_v; out_stress[7] = v_grad[4] * inv_v; out_stress[8] = v_grad[5] * inv_v;
    }
    return 0;
}

static int evaluate_impl(
    LoupHandle* h, const double* positions, const double* cell, int compute_stress,
    double* out_energy_ev, double* out_forces, double* out_stress
) {
    bool finite = true;
    int rc = evaluate_once(h, positions, cell, compute_stress, out_energy_ev, out_forces, out_stress, &finite);
    if (rc == 0 && !finite && drop_hybrid(h)) {
        rc = evaluate_once(h, positions, cell, compute_stress, out_energy_ev, out_forces, out_stress, &finite);
    }
    if (rc == 0 && !finite) {
        loup::set_error("non-finite energy, forces or stress (model output; the configuration may be unphysical)");
        return -1;
    }
    return rc;
}

// Per-atom energies (S6-T08, PLAN.md D70) of the positions the last evaluation uploaded: one uncaptured energy-only
// pass whose readouts and ZBL term also add each row's share into an [N] buffer (MACE's node energies: the readout
// of every layer, and half of each ZBL pair energy on its receiver), plus each atom's E0. They sum to the total
// energy up to float64 rounding. The pass reuses the GEMM plans of the energy graph and leaves the device forces
// stale.
static int per_atom_energies(LoupHandle* h, double* out) {
    if (ensure_energy_graph(h) != 0) return -1;
    const int rows = h->N + h->n_ghost;
    double* d_ae = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ae, (size_t)rows * sizeof(double)));
    struct Pass {  // the buffer goes, and the next recorded pass is a plain one, whatever happens
        LoupHandle* h;
        double* d;
        bool saved_fork;
        Pass(LoupHandle* h_, double* d_) : h(h_), d(d_), saved_fork(h_->fork_ok) {
            h->pass_atom_energy = d;
            h->fork_ok = false;
        }
        ~Pass() {
            h->pass_atom_energy = nullptr;
            h->fork_ok = saved_fork;
            cudaFree(d);
        }
    } pass(h, d_ae);
    CUDA_CHECK(cudaMemsetAsync(d_ae, 0, (size_t)rows * sizeof(double), h->stream));
    record_force_pass(h, 0, true);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> ae(rows);
    CUDA_CHECK(cudaMemcpyAsync(ae.data(), d_ae, (size_t)rows * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    h->forces_current = false;
    for (int r = 0; r < rows; ++r) {
        const int orig = h->perm[r];
        out[orig] = r < h->N ? ae[r] + h->atom_e0[orig] : 0.0;
        if (!std::isfinite(out[orig])) {
            loup::set_error("non-finite per-atom energy (atom %d)", orig);
            return -1;
        }
    }
    return 0;
}

static int evaluate2_impl(
    LoupHandle* h, const double* positions, const double* cell, int flags, double* out_energy_ev,
    double* out_forces, double* out_stress, double* out_atom_energies
) {
    const int known = LOUP_EVAL_STRESS | LOUP_EVAL_ENERGY_ONLY | LOUP_EVAL_PER_ATOM;
    if (flags & ~known) { loup::set_error("unknown evaluation flags 0x%x", flags & ~known); return -1; }
    if (flags & LOUP_EVAL_PER_ATOM) {
        if (!out_atom_energies) { loup::set_error("LOUP_EVAL_PER_ATOM needs out_atom_energies"); return -1; }
        if (!h->generic()) {
            loup::set_error("per-atom energies are built for the generic kernels only; this engine runs the "
                            "specialised MACE-OMAT-0 medium kernels (load the bundle with generic=True)");
            return -1;
        }
        if (h->gen->has_lr()) {
            loup::set_error("per-atom energies are not built for long-range models (the k-space energy is not "
                            "split by atom)");
            return -1;
        }
        if (evaluate2_impl(h, positions, cell, flags & ~LOUP_EVAL_PER_ATOM, out_energy_ev, out_forces, out_stress,
                           nullptr) != 0)
            return -1;
        if (per_atom_energies(h, out_atom_energies) != 0) return -1;
        // The energy of a per-atom call is the sum of its per-atom energies (the energy-only pass), so the two
        // agree to the rounding of the sum; it differs from the plain call's energy in the float32 rounding of the
        // readouts (measured up to 6e-7 relative on a compressed cell).
        if (out_energy_ev) {
            double sum = 0.0;
            for (int r = 0; r < h->N; ++r) sum += out_atom_energies[h->perm[r]];
            *out_energy_ev = sum;
        }
        return 0;
    }
    (void)out_atom_energies;
    if (!(flags & LOUP_EVAL_ENERGY_ONLY)) {
        return evaluate_impl(h, positions, cell, (flags & LOUP_EVAL_STRESS) ? 1 : 0, out_energy_ev, out_forces,
                             (flags & LOUP_EVAL_STRESS) ? out_stress : nullptr);
    }
    if (flags & LOUP_EVAL_STRESS) {
        loup::set_error("LOUP_EVAL_ENERGY_ONLY excludes LOUP_EVAL_STRESS: the energy-only pass has no gradients");
        return -1;
    }
    // prepare_evaluate has marked the forces on the device as stale; they stay so (a run that follows starts with
    // a force pass). A non-finite energy with the hybrid GEMMs loaded drops them and evaluates once more, as the
    // full call does.
    double inter_e = 0.0;
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (prepare_evaluate(h, positions, cell) != 0) return -1;
        if (ensure_energy_graph(h) != 0) return -1;
        CUDA_CHECK(cudaGraphLaunch(h->energy_graph_exec, h->stream));
        CUDA_CHECK(cudaMemcpyAsync(&inter_e, h->d_energy_f64, sizeof(double), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        if (std::isfinite(inter_e) || !drop_hybrid(h)) break;
    }
    if (!std::isfinite(inter_e)) {
        loup::set_error("non-finite energy (model output; the configuration may be unphysical)");
        return -1;
    }
    if (out_energy_ev) *out_energy_ev = h->e0_total + inter_e;
    return 0;
}

// A barostat's settings (Berendsen NPT, kind 5): pressure in eV/A^3, compressibility in A^3/eV.
struct NptParams {
    double pressure = 0.0, tau_p_fs = 0.0, compressibility = 0.0;
    int full = 0;
    int mask[3] = {1, 1, 1};
    int pchain = 0, ploop = 0;  // MTK: the barostat's chain
};

static int run_md_loop(
    LoupHandle* h, int kind, int n_steps, double dt_fs,
    double temp_k, double temp_rate, double friction_per_fs, unsigned long long seed, int thermo_every,
    bool fix_com, LoupMDStats* out_stats, double tau_fs = 0.0, int tchain = 0, int tloop = 1,
    double mass_temp_k = 0.0, const NptParams* npt = nullptr
) {
    // nl_valid: positions and a neighbour list exist (set_state or evaluate ran and its rebuild completed).
    if (!has_system(h)) return -1;
    if (!h->nl_valid) {
        loup::set_error("no positions or neighbour list (call loup_set_state or loup_evaluate first)");
        return -1;
    }
    if (n_steps <= 0) { loup::set_error("n_steps must be positive, got %d", n_steps); return -1; }
    if (!std::isfinite(dt_fs) || !(dt_fs > 0.0)) {
        loup::set_error("dt_fs must be finite and positive, got %g fs", dt_fs);
        return -1;
    }
    if (thermo_every < 0) { loup::set_error("thermo_every must not be negative, got %d", thermo_every); return -1; }
    CUDA_CHECK(cudaSetDevice(h->device_id));
    const int N = h->N;
    const int gridN = (N + 255) / 256;
    const double dt_ase = dt_fs * loup::ASE_FS;
    const double half_skin = 0.5 * h->skin;
    // The variable-cell step (cell.hpp): the cell on the device, the Cartesian shifts formed from the integer ones
    // in every step, stress in every force pass, the list check against the deformed reference. One host wait per
    // step: it does not pipeline. The first such run of a handle rebuilds the list once, which brings the cell and
    // the sorted integer shifts onto the device; a fixed-cell run after a cell change rebuilds for its plain check.
    const bool var_cell = h->req_var_cell;
    const double cell_rate = var_cell ? h->req_cell_rate : 0.0;
    if (var_cell && !h->var_cell) {
        if (!h->d_celldev) CUDA_CHECK(cudaMalloc(&h->d_celldev, sizeof(loup::CellDev)));
        h->var_cell = true;
        if (rebuild_neighbors(h) != 0) return -1;
    } else if (!var_cell && h->cell_ref_stale) {
        if (rebuild_neighbors(h) != 0) return -1;
    }
    // Thermo rows of this call: counted on the host, so a run that would overflow the buffer is refused here.
    int thermo_rows = 0;
    if (thermo_every > 0) {
        thermo_rows = (int)((h->md_steps + n_steps) / thermo_every - h->md_steps / thermo_every);
        if (h->thermo_pending + thermo_rows > loup::kThermoCapacity) {
            loup::set_error("this call would write %d thermo rows and the buffer has room for %d of %d: read the "
                            "rows first (loup_read_thermo) or run fewer steps per call", thermo_rows,
                            loup::kThermoCapacity - h->thermo_pending, loup::kThermoCapacity);
            return -1;
        }
        if (!h->d_thermo_ctrl) CUDA_CHECK(cudaMalloc(&h->d_thermo_ctrl, sizeof(loup::ThermoCtrl)));
        if (!h->d_thermo_rows) {
            CUDA_CHECK(cudaMalloc(&h->d_thermo_rows, (size_t)loup::kThermoCapacity * sizeof(LoupThermoRow)));
        }
        loup::ThermoCtrl ctrl{};
        ctrl.step = (unsigned long long)h->md_steps;
        ctrl.every = thermo_every;
        ctrl.count = h->thermo_pending;
        ctrl.capacity = loup::kThermoCapacity;
        CUDA_CHECK(cudaMemcpy(h->d_thermo_ctrl, &ctrl, sizeof(ctrl), cudaMemcpyHostToDevice));
    }
    // Two loop modes. Unpipelined: each step's skin check is read before its force pass, so the force always uses a
    // list built within skin/2 of the current positions (exact for any skin, including 0), at one host sync per step.
    // Pipelined: step k + 1 is launched before step k's check is read, so two force calls follow each check; the
    // check lowers its threshold by three times the largest per-step displacement seen (two for the lag, one of
    // margin for a displacement larger than any so far: check_skin_kernel). That is only sound while the per-step
    // displacement is well below the skin, so the loop starts unpipelined, switches to pipelining once
    // 4 x max step displacement < skin/2, and drops back if that stops holding or a step is found to have run on an
    // incomplete list (counted in dangerous_builds).
    const bool allow_pipeline = (h->skin > 0.1 && n_steps > 1) && !var_cell;
    constexpr int kPipeLag = 3;  // steps by which the pipelined check lowers its threshold (two of lag, one of margin)
    auto bits_to_disp = [](int bits) {
        float d2;
        memcpy(&d2, &bits, sizeof(d2));
        return std::sqrt((double)d2);
    };
    // The step-displacement maximum decays in two windows (integrator.cu). A window ends at a neighbour rebuild once
    // it holds kDispWindowMin steps, so the previous window always stands for that many steps of every atom, and
    // after kDispWindowMax steps without a rebuild, so a system too cold to rebuild forgets a fast start as well.
    constexpr int kDispWindowMin = 32, kDispWindowMax = 512;
    auto rotate_disp = [&] {
        loup::rotate_step_disp_kernel<<<1, 1, 0, h->stream>>>(h->d_max_step_disp2, h->d_need_rebuild);
        h->disp_window_steps = 0;
    };
    auto note_step = [&] {
        if (++h->disp_window_steps >= kDispWindowMax) rotate_disp();
    };

    // Forces at step 0. They are already on the device when the last call on this handle was a force evaluation or
    // a completed run at these positions, so a run made of many short calls pays one force pass per step and none
    // per call. (lt_cache: the uncaptured pass of ensure_force_graph, which the MD graphs' capture relies on.)
    const bool debug_graphs = std::getenv("LOUP_DEBUG_GRAPHS") != nullptr;
    if (!h->forces_current || h->lt_cache.empty() || (var_cell && !h->last_pass_stress)) {
        if (ensure_force_graph(h, var_cell ? 1 : 0) != 0) return -1;
        if (debug_graphs) fputs("loupmd: initial force pass\n", stderr);
        CUDA_CHECK(cudaGraphLaunch(h->force_graph_exec, h->stream));
    }
    // The Langevin step counter persists across run calls (reset only in loup_set_system), so chunked runs with
    // one seed continue the noise sequence instead of replaying it.
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    h->forces_current = false;  // until this call completes: a failure below leaves positions ahead of the forces
    ++h->pos_epoch;

    // ASE's Langevin coefficients. c3, c4, c5 carry sigma = sqrt(2 gamma kB T), which the first half-kick forms from
    // the target on the device; here are their factors of sigma.
    double c1 = 0, c2 = 0, a3 = 0, a4 = 0, a5 = 0, two_fr_kb = 0;
    if (kind == 1) {
        const double fr = friction_per_fs / loup::ASE_FS;  // 1/fs -> 1/(ASE time unit), as ASE's friction=f/units.fs
        c1 = dt_ase / 2.0 - dt_ase * dt_ase * fr / 8.0;
        c2 = dt_ase * fr / 2.0 - dt_ase * dt_ase * fr * fr / 8.0;
        two_fr_kb = 2.0 * fr * loup::ASE_KB;
        a3 = sqrt(dt_ase) / 2.0 - pow(dt_ase, 1.5) * fr / 8.0;
        a5 = pow(dt_ase, 1.5) / (2.0 * sqrt(3.0));
        a4 = (fr / 2.0) * a5;
        // The target of this call's first step (its end point), the change per step, and the first step's noise
        // coefficients; the second half-kick advances all of them.
        const double first = temp_k + temp_rate;
        const double sigma = sqrt(two_fr_kb * (first > 0.0 ? first : 0.0));
        const double target[5] = {first, temp_rate, a3 * sigma, a4 * sigma, a5 * sigma};
        CUDA_CHECK(cudaMemcpy(h->d_langevin_target, target, sizeof(target), cudaMemcpyHostToDevice));
    }
    // CSVR (kind 2): ASE's exp(-dt / taut). The target of the call's first step and its change per step go into the
    // thermostat target's first two slots; the scaling's own kernel advances them. The transferred energy carries
    // on from the last CSVR call on this trajectory.
    double csvr_exp = 0.0;
    if (kind == 2) {
        csvr_exp = exp(-dt_fs / tau_fs);
        const double target[5] = {temp_k + temp_rate, temp_rate, 0.0, 0.0, 0.0};
        CUDA_CHECK(cudaMemcpy(h->d_langevin_target, target, sizeof(target), cudaMemcpyHostToDevice));
        if (h->int_state_kind != LOUP_CSVR) {
            CUDA_CHECK(cudaMemsetAsync(h->d_int_state, 0, kIntStateSlots * sizeof(double), h->stream));
            h->int_state_kind = LOUP_CSVR;
        }
        CUDA_CHECK(cudaMemsetAsync(h->d_int_work, 0, kIntWorkSlots * sizeof(double), h->stream));
    }
    // Berendsen (kind 4, and 5 with its barostat): the target as for CSVR; it has no state of its own.
    if (kind == 4 || kind == 5) {
        const double target[5] = {temp_k + temp_rate, temp_rate, 0.0, 0.0, 0.0};
        CUDA_CHECK(cudaMemcpy(h->d_langevin_target, target, sizeof(target), cudaMemcpyHostToDevice));
        const int state_kind = kind == 4 ? LOUP_BERENDSEN_NVT : LOUP_BERENDSEN_NPT;
        if (h->int_state_kind != state_kind) {
            CUDA_CHECK(cudaMemsetAsync(h->d_int_state, 0, kIntStateSlots * sizeof(double), h->stream));
            h->int_state_kind = state_kind;
        }
        CUDA_CHECK(cudaMemsetAsync(h->d_int_work, 0, kIntWorkSlots * sizeof(double), h->stream));
    }
    // Langevin BAOAB NPT (kind 6): ASE's constants. T_tau = 1 / friction; P_tau = tau_p (0: ASE's default of
    // 20 T_tau); the barostat's mass from ASE's heuristic for a period of P_tau / 4; its thermostat's rate 1 / P_tau.
    // The external stress is minus the target pressure. The barostat's momentum and the cell force carry on from
    // the last such call on this trajectory; a first call forms the cell force from the state it finds.
    double bao_gamma = 0.0, bao_ou = 0.0, bao_gamma_b = 0.0, bao_ou_b = 0.0, bao_mass = 0.0;
    const int bao_ndof = 3 * N;
    bool bao_start = false;
    if (kind == 6) {
        bao_gamma = friction_per_fs / loup::ASE_FS;
        bao_ou = sqrt(1.0 - exp(-2.0 * bao_gamma * dt_ase));
        const double p_tau = (npt->tau_p_fs > 0.0 ? npt->tau_p_fs : 20.0 / friction_per_fs) * loup::ASE_FS;
        const double c = 1.66 * loup::ASE_FS;
        bao_mass = pow((p_tau / 4.0) / c * pow((double)N, 1.0 / 6.0), 2.0);
        bao_gamma_b = 1.0 / p_tau;
        bao_ou_b = sqrt(1.0 - exp(-2.0 * bao_gamma_b * dt_ase)) * sqrt(bao_mass);
        const double target[5] = {temp_k + temp_rate, temp_rate, 0.0, 0.0, 0.0};
        CUDA_CHECK(cudaMemcpy(h->d_langevin_target, target, sizeof(target), cudaMemcpyHostToDevice));
        if (h->int_state_kind != LOUP_BAOAB_NPT) {
            CUDA_CHECK(cudaMemsetAsync(h->d_int_state, 0, kIntStateSlots * sizeof(double), h->stream));
            h->int_state_kind = LOUP_BAOAB_NPT;
            bao_start = true;
        }
        CUDA_CHECK(cudaMemsetAsync(h->d_int_work, 0, kIntWorkSlots * sizeof(double), h->stream));
    }
    // Martyna-Tobias-Klein NPT (kind 7): ASE's masses from the temperature the masses are made for: the
    // thermostat's unit kT tdamp^2, the barostat chain's kT pdamp^2, the cell's W = (N_f + 3) kT pdamp^2
    // (isotropic) or (N_f / 3 + 1) kT pdamp^2 (full cell). Chains and cell momentum carry on from the last such
    // call on this trajectory.
    double mtk_q = 0.0, mtk_r = 0.0, mtk_w = 0.0;
    if (kind == 7) {
        const double tdamp = tau_fs * loup::ASE_FS, pdamp = npt->tau_p_fs * loup::ASE_FS;
        const double kt_mass = loup::ASE_KB * mass_temp_k;
        mtk_q = kt_mass * tdamp * tdamp;
        mtk_r = kt_mass * pdamp * pdamp;
        mtk_w = (npt->full ? (double)h->kinetic_dof() / 3.0 + 1.0 : (double)h->kinetic_dof() + 3.0) * mtk_r;
        const double target[5] = {temp_k + temp_rate, temp_rate, 0.0, 0.0, 0.0};
        CUDA_CHECK(cudaMemcpy(h->d_langevin_target, target, sizeof(target), cudaMemcpyHostToDevice));
        if (h->int_state_kind != LOUP_MTK_NPT) {
            CUDA_CHECK(cudaMemsetAsync(h->d_int_state, 0, kIntStateSlots * sizeof(double), h->stream));
            h->int_state_kind = LOUP_MTK_NPT;
        }
        CUDA_CHECK(cudaMemsetAsync(h->d_int_work, 0, kIntWorkSlots * sizeof(double), h->stream));
    }
    // Nose-Hoover chain (kind 3): the thermostat masses' unit kT_mass tdamp^2 (Q_0 = N_f q, the others q), and the
    // target as for CSVR. The chain's variables carry on from the last Nose-Hoover call on this trajectory.
    double nhc_q = 0.0;
    if (kind == 3) {
        const double tdamp = tau_fs * loup::ASE_FS;
        nhc_q = loup::ASE_KB * mass_temp_k * tdamp * tdamp;
        const double target[5] = {temp_k + temp_rate, temp_rate, 0.0, 0.0, 0.0};
        CUDA_CHECK(cudaMemcpy(h->d_langevin_target, target, sizeof(target), cudaMemcpyHostToDevice));
        if (h->int_state_kind != LOUP_NHC) {
            CUDA_CHECK(cudaMemsetAsync(h->d_int_state, 0, kIntStateSlots * sizeof(double), h->stream));
            h->int_state_kind = LOUP_NHC;
        }
        CUDA_CHECK(cudaMemsetAsync(h->d_int_work, 0, kIntWorkSlots * sizeof(double), h->stream));
    }

    // fix_com: the buffers of the two-pass noise, and for NVE the total momentum removed once (the forces of a
    // periodic system sum to zero to rounding, so it stays removed).
    const double inv_total_mass = fix_com ? 1.0 / h->total_mass : 0.0;
    if (fix_com) {
        if (!h->d_com) CUDA_CHECK(cudaMalloc(&h->d_com, 6 * sizeof(double)));
        if (kind == 1 && !h->d_rnd_pos) CUDA_CHECK(cudaMalloc(&h->d_rnd_pos, (size_t)N * 3 * sizeof(double)));
        if (kind != 1) {  // NVE and CSVR (whose scaling keeps a zero total momentum zero)
            CUDA_CHECK(cudaMemsetAsync(h->d_com, 0, 6 * sizeof(double), h->stream));
            loup::momentum_sum_kernel<<<(N + 255) / 256, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64, h->d_com);
            loup::momentum_remove_kernel<<<(N + 255) / 256, 256, 0, h->stream>>>(N, inv_total_mass, h->d_com,
                                                                                  h->d_vel64);
        }
    }
    h->fix_com = fix_com;

    auto record_move = [&](int lag) {
        if (kind == 1 && fix_com) {
            // (No CUDA_CHECK inside this lambda: a failed node fails the capture, which is checked.)
            (void)cudaMemsetAsync(h->d_com, 0, 6 * sizeof(double), h->stream);
            loup::langevin_noise_kernel<<<(N + 31) / 32, 32, 0, h->stream>>>(
                N, h->d_langevin_target, seed, h->d_step_counter, h->d_masses, h->d_sqrt_inv_masses, h->d_rnd_pos,
                h->d_rnd_vel, h->d_com);
            loup::langevin_first_half_fixcom_kernel<<<gridN, 256, 0, h->stream>>>(
                N, dt_ase, c1, c2, inv_total_mass, h->d_com, h->d_forces_f64, h->d_inv_masses, h->d_rnd_pos,
                h->d_rnd_vel, h->d_pos64, h->d_vel64, h->d_max_step_disp2);
        } else if (kind == 0) {
            loup::nve_first_half_kernel<<<gridN, 256, 0, h->stream>>>(
                N, dt_ase, h->d_forces_f64, h->d_inv_masses, h->d_pos64, h->d_vel64, h->d_max_step_disp2);
        } else if (kind == 4) {
            // The kinetic temperature, lambda from it, then the half-kick on the scaled velocities.
            loup::kinetic_energy_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64, h->d_int_work);
            loup::berendsen_scale_kernel<<<1, 1, 0, h->stream>>>(h->kinetic_dof(), dt_fs / tau_fs, loup::ASE_KB,
                                                                 h->d_langevin_target, h->d_int_work);
            loup::csvr_first_half_kernel<<<gridN, 256, 0, h->stream>>>(
                N, dt_ase, h->d_int_work, h->d_forces_f64, h->d_inv_masses, h->d_pos64, h->d_vel64,
                h->d_max_step_disp2);
        } else if (kind == 7) {
            // MTK NPT: the kinetic tensor, one thread for both chains, the cell kick and the step's matrices,
            // then every atom's scaling, kick and drift (integrator.cu).
            loup::kinetic_tensor_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64,
                                                                      h->d_int_work + 44);
            loup::mtk_first_kernel<<<1, 1, 0, h->stream>>>(
                npt->full, dt_ase, tchain, tloop, npt->pchain, npt->ploop, h->kinetic_dof(), mtk_q, mtk_r, mtk_w,
                loup::ASE_KB, npt->pressure, h->d_langevin_target, h->d_int_state, h->d_int_work,
                h->d_virial_grad_f64, h->d_celldev);
            loup::mtk_first_atoms_kernel<<<gridN, 256, 0, h->stream>>>(
                N, dt_ase, h->d_int_work, h->d_forces_f64, h->d_inv_masses, h->d_pos64, h->d_vel64,
                h->d_max_step_disp2);
        } else if (kind == 6) {
            // Langevin BAOAB NPT: one thread for the barostat and the step's matrices, then every atom's B, A, O, A
            // with the cell's two half steps around them (integrator.cu).
            loup::baoab_cell_first_kernel<<<1, 1, 0, h->stream>>>(
                npt->full, dt_ase, bao_gamma, bao_ou, bao_gamma_b, bao_ou_b, bao_mass, bao_ndof, loup::ASE_KB, seed,
                h->d_step_counter, h->d_langevin_target, h->d_int_state, h->d_int_work, h->d_celldev);
            loup::baoab_first_kernel<<<(N + 31) / 32, 32, 0, h->stream>>>(
                N, dt_ase, h->d_int_work, seed, h->d_step_counter, h->d_forces_f64, h->d_inv_masses,
                h->d_sqrt_inv_masses, h->d_pos64, h->d_vel64, h->d_max_step_disp2);
        } else if (kind == 5) {
            // Berendsen NPT: lambda from the kinetic energy; the barostat's factors from the stress of the last
            // force pass and the scaled velocities' ideal-gas part; the atoms scaled with the cell (the cell
            // itself follows in cell_derive_kernel below); then the half-kick on the scaled velocities and the
            // drift. One force call per step, as FastMACE's port: ASE's class kicks with forces of the scaled cell.
            loup::kinetic_energy_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64, h->d_int_work);
            if (npt->full) {
                loup::kinetic_diag_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64,
                                                                        h->d_int_work + 2);
            }
            loup::berendsen_scale_kernel<<<1, 1, 0, h->stream>>>(h->kinetic_dof(), dt_fs / tau_fs, loup::ASE_KB,
                                                                 h->d_langevin_target, h->d_int_work);
            loup::berendsen_npt_kernel<<<1, 1, 0, h->stream>>>(
                npt->full, npt->mask[0], npt->mask[1], npt->mask[2], npt->pressure, dt_fs, npt->tau_p_fs,
                npt->compressibility, h->d_int_work, h->d_virial_grad_f64, h->d_celldev);
            loup::cell_move_positions_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_celldev, h->d_pos64);
            loup::csvr_first_half_kernel<<<gridN, 256, 0, h->stream>>>(
                N, dt_ase, h->d_int_work, h->d_forces_f64, h->d_inv_masses, h->d_pos64, h->d_vel64,
                h->d_max_step_disp2);
        } else if (kind == 3) {
            // The chain for half a step from the kinetic energy; the half-kick on the velocities it scales.
            loup::kinetic_energy_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64, h->d_int_work);
            loup::nhc_half_kernel<<<1, 1, 0, h->stream>>>(h->kinetic_dof(), tchain, tloop, 0.5 * dt_ase, nhc_q,
                                                          loup::ASE_KB, 0, h->d_langevin_target, h->d_int_work,
                                                          h->d_int_state);
            loup::csvr_first_half_kernel<<<gridN, 256, 0, h->stream>>>(
                N, dt_ase, h->d_int_work, h->d_forces_f64, h->d_inv_masses, h->d_pos64, h->d_vel64,
                h->d_max_step_disp2);
        } else if (kind == 2) {
            // The kinetic energy, alpha from it and this step's noise, then the half-kick on the scaled velocities.
            loup::kinetic_energy_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64, h->d_int_work);
            loup::csvr_alpha_kernel<<<1, 1, 0, h->stream>>>(h->kinetic_dof(), csvr_exp, loup::ASE_KB, seed,
                                                            h->d_step_counter, h->d_langevin_target, h->d_int_work,
                                                            h->d_int_state);
            loup::csvr_first_half_kernel<<<gridN, 256, 0, h->stream>>>(
                N, dt_ase, h->d_int_work, h->d_forces_f64, h->d_inv_masses, h->d_pos64, h->d_vel64,
                h->d_max_step_disp2);
        } else {
            // 32-thread blocks: the FP64 Box-Muller (3 curand_normal2_double per atom) is FP64-throughput bound on
            // consumer GPUs, so spread the atoms over every SM instead of N / 256 of them.
            loup::langevin_first_half_kernel<<<(N + 31) / 32, 32, 0, h->stream>>>(
                N, dt_ase, c1, c2, h->d_langevin_target, seed, h->d_step_counter,
                h->d_forces_f64, h->d_inv_masses, h->d_sqrt_inv_masses,
                h->d_pos64, h->d_vel64, h->d_rnd_vel, h->d_max_step_disp2);
        }
        if (var_cell) {
            // Whoever moves the cell sets this step's deformation; the atoms follow it, then the cell itself and
            // what the step needs from it. The list check measures from the deformed reference.
            if (cell_rate != 0.0) {
                loup::cell_debug_rate_kernel<<<1, 1, 0, h->stream>>>(cell_rate, h->d_celldev);
                loup::cell_move_positions_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_celldev, h->d_pos64);
            }
            loup::cell_derive_kernel<<<1, 1, 0, h->stream>>>(h->d_celldev, h->d_need_rebuild);
            loup::check_skin_cell_kernel<<<gridN, 256, 0, h->stream>>>(
                N, lag, 0, h->d_pos64, h->d_ref_pos64, h->d_celldev, h->d_max_step_disp2, h->d_need_rebuild);
            return;
        }
        // The unpipelined check (lag 0) also reports whether the list is still young enough for the pipelined loop.
        loup::check_skin_kernel<<<gridN, 256, 0, h->stream>>>(
            N, half_skin, lag, lag == 0 ? kPipeLag : 0, h->d_pos64, h->d_ref_pos64, h->d_max_step_disp2,
            h->d_need_rebuild);
    };
    auto record_rest = [&]() {
        if (var_cell && h->E_graph > 0) {  // the edges' Cartesian shifts in the cell of this step
            loup::shift_cart_kernel<<<(h->E_graph + 255) / 256, 256, 0, h->stream>>>(
                h->d_row_ptr + N, h->d_celldev, h->d_tmp_snd, h->d_tmp_rcv, h->d_tmp_pairs, h->d_shift_cart);
        }
        h->disp_cell_dev = var_cell;
        record_force_pass(h, var_cell ? 1 : 0);
        h->disp_cell_dev = false;
        if (kind == 6) {  // the cell force from the velocities before the last kick, then barostat B
            loup::kinetic_tensor_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64, h->d_int_work);
            loup::baoab_cell_second_kernel<<<1, 1, 0, h->stream>>>(
                npt->full, 1, 1, dt_ase, bao_ndof, -npt->pressure, h->d_virial_grad_f64, h->d_celldev, h->d_int_work,
                h->d_int_state, h->d_step_counter);
        }
        if (kind == 7) {  // the atom kick, the cell kick with the new stress, both chains
            loup::mtk_kick_kernel<<<gridN, 256, 0, h->stream>>>(N, dt_ase, h->d_int_work, h->d_forces_f64,
                                                                h->d_inv_masses, h->d_vel64);
            loup::kinetic_tensor_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64,
                                                                      h->d_int_work + 44);
            loup::mtk_second_kernel<<<1, 1, 0, h->stream>>>(
                npt->full, dt_ase, tchain, tloop, npt->pchain, npt->ploop, h->kinetic_dof(), mtk_q, mtk_r, mtk_w,
                loup::ASE_KB, npt->pressure, h->d_langevin_target, h->d_int_state, h->d_int_work,
                h->d_virial_grad_f64, h->d_celldev);
            loup::scale_velocities_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_int_work, h->d_vel64);
        } else if (kind != 1) {
            loup::nve_second_half_kernel<<<gridN, 256, 0, h->stream>>>(
                N, dt_ase, h->d_forces_f64, h->d_inv_masses, h->d_vel64);
        } else {
            loup::langevin_second_half_kernel<<<gridN, 256, 0, h->stream>>>(
                N, c1, c2, h->d_forces_f64, h->d_inv_masses, h->d_rnd_vel, h->d_vel64, h->d_step_counter,
                h->d_langevin_target, a3, a4, a5, two_fr_kb);
        }
        if (kind == 3) {  // the chain's second half, which also moves the target on
            loup::kinetic_energy_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64, h->d_int_work);
            loup::nhc_half_kernel<<<1, 1, 0, h->stream>>>(h->kinetic_dof(), tchain, tloop, 0.5 * dt_ase, nhc_q,
                                                          loup::ASE_KB, 1, h->d_langevin_target, h->d_int_work,
                                                          h->d_int_state);
            loup::scale_velocities_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_int_work, h->d_vel64);
        }
        if (thermo_every > 0) {  // the row cadence lives on the device: only on / off is baked into the graph
            loup::thermo_kinetic_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64, h->d_thermo_ctrl);
            loup::thermo_row_kernel<<<1, 1, 0, h->stream>>>(h->kinetic_dof(), h->e0_total, loup::ASE_KB,
                                                            h->d_energy_f64,
                                                            h->d_thermo_ctrl, h->d_thermo_rows,
                                                            kind >= 2 ? h->d_int_state : nullptr,
                                                            var_cell ? h->d_celldev : nullptr,
                                                            var_cell ? h->d_virial_grad_f64 : nullptr);
        }
    };
    const double npt_key[7] = {npt ? npt->pressure : 0.0, npt ? npt->tau_p_fs : 0.0, npt ? npt->compressibility : 0.0,
                               npt ? (double)npt->full : 0.0,
                               npt ? (double)(npt->mask[0] + 2 * npt->mask[1] + 4 * npt->mask[2]) : 0.0,
                               npt ? (double)npt->pchain : 0.0, npt ? (double)npt->ploop : 0.0};
    auto capture = [&](cudaGraph_t& g, cudaGraphExec_t& ge, auto&& body) -> int {
        if (ge) { cudaGraphExecDestroy(ge); ge = nullptr; }
        if (g) { cudaGraphDestroy(g); g = nullptr; }
        if (debug_graphs) fputs("loupmd: MD graph capture\n", stderr);
        CUDA_CHECK(cudaStreamBeginCapture(h->stream, cudaStreamCaptureModeGlobal));
        body();
        CUDA_CHECK(cudaStreamEndCapture(h->stream, &g));
        CUDA_CHECK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        return 0;
    };
    auto note_params = [&] {
        h->md_graph_kind = kind;
        h->md_graph_dt = dt_fs;
        h->md_graph_fr = friction_per_fs;
        h->md_graph_tau = tau_fs;
        h->md_graph_mass_temp = mass_temp_k;
        h->md_graph_chain = 1000 * tchain + tloop;
        h->md_graph_seed = seed;
        h->md_graph_thermo = thermo_every > 0;
        h->md_graph_fixcom = fix_com;
        h->md_graph_varcell = var_cell;
        h->md_graph_cellrate = cell_rate;
        memcpy(h->md_graph_npt, npt_key, sizeof(npt_key));
    };
    auto ensure_piped = [&]() -> int {
        if (h->md_graph_exec) return 0;
        if (capture(h->md_graph, h->md_graph_exec, [&] { record_move(kPipeLag); record_rest(); }) != 0) return -1;
        note_params();
        return 0;
    };
    auto ensure_unpiped = [&]() -> int {
        if (h->md_move_exec && h->md_rest_exec) return 0;
        if (capture(h->md_move_graph, h->md_move_exec, [&] { record_move(0); }) != 0) return -1;
        if (capture(h->md_rest_graph, h->md_rest_exec, record_rest) != 0) return -1;
        note_params();
        return 0;
    };
    // The graphs bake in dt, friction, time constant and seed: those of an earlier call are kept when all are the same
    // (review item S4) and dropped otherwise. The thermostat's temperature is not baked in (d_langevin_target), so a
    // schedule's calls reuse the graphs. Everything else a graph bakes in (edge bucket, scan workspace, kernel
    // variants, the long-range part's cell) drops the graphs where it changes: invalidate_graphs.
    if (h->md_graph_kind != kind || h->md_graph_dt != dt_fs ||
        h->md_graph_fr != friction_per_fs || h->md_graph_tau != tau_fs || h->md_graph_seed != seed ||
        h->md_graph_mass_temp != mass_temp_k || h->md_graph_chain != 1000 * tchain + tloop ||
        h->md_graph_thermo != (thermo_every > 0) ||
        h->md_graph_fixcom != fix_com || h->md_graph_varcell != var_cell || h->md_graph_cellrate != cell_rate ||
        memcmp(h->md_graph_npt, npt_key, sizeof(npt_key)) != 0) {
        h->invalidate_md_graphs();
    }

    int rebuilds = 0, dangerous = 0;
    double tune_ms = 0.0;
    // The per-system variant timing (first rebuild only) is not part of the steps: it is timed and excluded.
    auto rebuild = [&](int flag) -> int {
        ++rebuilds;
        if (flag & 2) ++dangerous;
        if (!h->variants_tuned) {
            const auto t0 = std::chrono::steady_clock::now();
            tune_rebuild_variants(h);
            tune_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        if (rebuild_neighbors(h) != 0) return -1;
        if (h->disp_window_steps >= kDispWindowMin) rotate_disp();
        return 0;
    };
    int* flags = h->h_reb_pinned;
    auto copy_disp = [&] {
        CUDA_CHECK(cudaMemcpyAsync(&flags[2], h->d_max_step_disp2, 2 * sizeof(int), cudaMemcpyDeviceToHost, h->stream));
        return 0;
    };
    auto step_disp = [&] {  // of the last copy_disp: the larger of the two windows
        return bits_to_disp((unsigned)flags[2] > (unsigned)flags[3] ? flags[2] : flags[3]);
    };

    if (bao_start) {  // LangevinBAOAB.__init__: the cell force of the forces, stress and velocities it starts from
        loup::kinetic_tensor_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64, h->d_int_work);
        loup::baoab_cell_second_kernel<<<1, 1, 0, h->stream>>>(
            npt->full, 0, 0, dt_ase, bao_ndof, -npt->pressure, h->d_virial_grad_f64, h->d_celldev, h->d_int_work,
            h->d_int_state, h->d_step_counter);
    }
    EventPairGuard evg;
    cudaEvent_t ev0 = evg.e0, ev1 = evg.e1;
    CUDA_CHECK(cudaEventRecord(ev0, h->stream));

    int launched = 0, pipelined_steps = 0;
    bool piped = false;
    while (launched < n_steps) {
        if (!piped) {
            if (ensure_unpiped() != 0) return -1;
            CUDA_CHECK(cudaGraphLaunch(h->md_move_exec, h->stream));
            flags[0] = 0;
            CUDA_CHECK(cudaMemcpyAsync(&flags[0], h->d_need_rebuild, sizeof(int), cudaMemcpyDeviceToHost, h->stream));
            copy_disp();
            CUDA_CHECK(cudaStreamSynchronize(h->stream));
            if (var_cell && (flags[0] & 8)) {  // cell_derive_kernel: the volume left its limits
                loup::CellDev c;
                if (download_cell_dev(h, &c) != 0) return -1;
                h->cell_ref_stale = true;
                loup::set_error("the cell diverged: its volume is %g A^3, outside [V0 / 20, 20 V0] with V0 = %g A^3 "
                                "(step %lld of the trajectory); the state on the handle is that step's",
                                c.volume, c.v0, (long long)(h->md_steps + launched + 1));
                return -1;
            }
            // Rebuilt before this step's forces, so bit 2 (already beyond skin/2) is not a violation here.
            if (flags[0] & 1) {
                if (var_cell && download_cell_dev(h) != 0) return -1;  // the list is built in the cell of now
                if (rebuild(1) != 0) return -1;
                if (ensure_unpiped() != 0) return -1;
            }
            CUDA_CHECK(cudaGraphLaunch(h->md_rest_exec, h->stream));
            ++launched;
            note_step();
            // The first pipelined force calls run on the current list before any flag is read. They are covered
            // only if the list was rebuilt in this step or no atom is beyond the pipelined (lowered) threshold
            // (bit 4). Without this, a list aged by unpipelined steps (single-step calls, evaluate()) was used up to
            // two steps past skin/2: 9 such steps in 60 switches of tests/test_engine.py's switch test.
            const bool list_young = (flags[0] & 1) || !(flags[0] & 4);
            piped = allow_pipeline && n_steps - launched >= 2 && 4.0 * step_disp() < half_skin && list_young;
            continue;
        }
        if (ensure_piped() != 0) return -1;
        int prev = 0;
        flags[0] = 0;
        CUDA_CHECK(cudaGraphLaunch(h->md_graph_exec, h->stream));
        CUDA_CHECK(cudaMemcpyAsync(&flags[0], h->d_need_rebuild, sizeof(int), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaEventRecord(h->ev_reb[0], h->stream));
        ++launched;
        ++pipelined_steps;
        note_step();
        bool leave = false;
        while (launched < n_steps && !leave) {
            const int cur = prev ^ 1;
            flags[cur] = 0;
            CUDA_CHECK(cudaGraphLaunch(h->md_graph_exec, h->stream));
            CUDA_CHECK(cudaMemcpyAsync(&flags[cur], h->d_need_rebuild, sizeof(int), cudaMemcpyDeviceToHost, h->stream));
            CUDA_CHECK(cudaEventRecord(h->ev_reb[cur], h->stream));
            ++launched;
            ++pipelined_steps;
            note_step();
            CUDA_CHECK(cudaEventSynchronize(h->ev_reb[prev]));
            if (flags[prev] & 1) {
                const int before = dangerous;
                if (rebuild(flags[prev]) != 0) return -1;
                // rebuild_neighbors synchronised the stream, so the in-flight step's flag copy has landed: that step
                // also ran on the old list.
                if ((flags[cur] & 2) && !(flags[prev] & 2)) ++dangerous;
                flags[cur] = 0;
                copy_disp();
                CUDA_CHECK(cudaStreamSynchronize(h->stream));
                leave = dangerous != before || 4.0 * step_disp() >= half_skin;
                if (!leave && ensure_piped() != 0) return -1;
            }
            prev = cur;
        }
        if (!leave) {
            CUDA_CHECK(cudaEventSynchronize(h->ev_reb[prev]));
            if ((flags[prev] & 1) && rebuild(flags[prev]) != 0) return -1;
        }
        piped = false;
    }

    CUDA_CHECK(cudaEventRecord(ev1, h->stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    // The block's last energy (forces feed it every step): non-finite means the run is lost; with the hybrid GEMMs
    // loaded they are dropped first, so a run restarted from a saved state does not hit the same overflow.
    double inter_e = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&inter_e, h->d_energy_f64, sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    if (!std::isfinite(inter_e)) {
        drop_hybrid(h);
        loup::set_error("non-finite potential energy after the MD block: the state is lost; restart from a saved "
                        "state");
        return -1;
    }
    h->last_pass_stress = var_cell;  // the steps' force passes, not the force graph's
    if (var_cell) {  // the host's cell follows; a cell that moved since the list was built makes the plain check stale
        loup::CellDev c;
        if (download_cell_dev(h, &c) != 0) return -1;
        bool moved = false;
        for (int i = 0; i < 9; ++i) moved = moved || c.f[i] != ((i % 4 == 0) ? 1.0 : 0.0);
        h->cell_ref_stale = moved;
    }
    if (out_stats) {
        double ke = 0.0;
        CUDA_CHECK(cudaMemsetAsync(h->d_ke_f64, 0, sizeof(double), h->stream));
        loup::kinetic_energy_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_masses, h->d_vel64, h->d_ke_f64);
        CUDA_CHECK(cudaMemcpyAsync(&ke, h->d_ke_f64, sizeof(double), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        out_stats->step_ms = std::max(0.0, (double)ms - tune_ms) / n_steps;
        out_stats->rebuilds = rebuilds;
        out_stats->final_potential_energy_ev = h->e0_total + inter_e;
        out_stats->final_kinetic_energy_ev = ke;
        out_stats->final_temperature_k = (2.0 * ke) / ((double)h->kinetic_dof() * loup::ASE_KB);
        out_stats->dangerous_builds = dangerous;
        out_stats->tune_ms = tune_ms;
        out_stats->pipelined_steps = pipelined_steps;
    }
    h->forces_current = true;
    h->md_steps += n_steps;
    h->thermo_pending += thermo_rows;
    return 0;
}

static int read_thermo_impl(LoupHandle* h, LoupThermoRow* rows, int max_rows, int* n_rows) {
    if (!has_system(h)) return -1;
    if (!n_rows) { loup::set_error("n_rows is NULL"); return -1; }
    if (max_rows < 0 || (!rows && max_rows > 0)) {
        loup::set_error("rows is NULL or max_rows is negative (max_rows = %d)", max_rows);
        return -1;
    }
    const int have = h->thermo_pending;
    if (!rows) { *n_rows = have; return 0; }
    const int take = std::min(have, max_rows);
    *n_rows = take;
    if (take == 0) return 0;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    CUDA_CHECK(cudaMemcpy(rows, h->d_thermo_rows, (size_t)take * sizeof(LoupThermoRow), cudaMemcpyDeviceToHost));
    if (take < have) {  // keep the unread rest at the front
        std::vector<LoupThermoRow> rest((size_t)(have - take));
        CUDA_CHECK(cudaMemcpy(rest.data(), h->d_thermo_rows + take, rest.size() * sizeof(LoupThermoRow), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h->d_thermo_rows, rest.data(), rest.size() * sizeof(LoupThermoRow), cudaMemcpyHostToDevice));
    }
    h->thermo_pending = have - take;  // the device's count is written from this at the start of the next run
    return 0;
}

// Testing only: the noise CSVR draws in steps first_step .. first_step + n_steps - 1 for `ndof` degrees of freedom,
// as the captured step draws it.
static int csvr_noise_impl(LoupHandle* h, unsigned long long seed, unsigned long long first_step, int n_steps,
                           int ndof, double* out_normal, double* out_chi2) {
    if (!h) { loup::set_error("the handle is NULL"); return -1; }
    if (n_steps <= 0 || ndof <= 0 || !out_normal || !out_chi2) {
        loup::set_error("n_steps and ndof must be positive and both outputs given (n_steps = %d, ndof = %d)", n_steps,
                        ndof);
        return -1;
    }
    CUDA_CHECK(cudaSetDevice(h->device_id));
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n_steps * 2 * sizeof(double)));
    loup::csvr_noise_debug_kernel<<<(n_steps + 255) / 256, 256>>>(seed, first_step, n_steps, ndof, d_out,
                                                                  d_out + n_steps);
    cudaError_t rc = cudaMemcpy(out_normal, d_out, (size_t)n_steps * sizeof(double), cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess) {
        rc = cudaMemcpy(out_chi2, d_out + n_steps, (size_t)n_steps * sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_out);
    CUDA_CHECK(rc);
    return 0;
}

// loup_minimize: LBFGS on the device, see src/minimize.cu for the algebra. Per step: one pass over the history
// for every product the host needs, the two triangular solves on the host, one pass for the direction, the move,
// the list check and the force pass. The host waits for the device once per step (for the products) and a second
// time only where the move may have outdated the neighbour list.
static int minimize_impl(LoupHandle* h, const LoupMinimizer* caller, int max_steps, LoupMinStats* out_stats) {
    if (!has_system(h)) return -1;
    if (!caller) { loup::set_error("the minimizer is NULL"); return -1; }
    if (caller->size != sizeof(LoupMinimizer)) {
        loup::set_error("LoupMinimizer.size is %u, and this library's layout has %zu bytes (set size to "
                        "sizeof(LoupMinimizer) of the header the library was built from)", caller->size,
                        sizeof(LoupMinimizer));
        return -1;
    }
    const LoupMinimizer m = *caller;
    if (m.kind != LOUP_MIN_LBFGS) { loup::set_error("unknown minimizer kind %d", m.kind); return -1; }
    if (!h->nl_valid) {
        loup::set_error("no positions or neighbour list (call loup_set_state or loup_evaluate first)");
        return -1;
    }
    if (max_steps < 0) { loup::set_error("max_steps must not be negative, got %d", max_steps); return -1; }
    if (!std::isfinite(m.fmax_ev_a) || !(m.fmax_ev_a > 0.0)) {
        loup::set_error("fmax_ev_a must be finite and positive, got %g eV/A", m.fmax_ev_a);
        return -1;
    }
    if (!std::isfinite(m.maxstep_a) || !(m.maxstep_a > 0.0) || !std::isfinite(m.damping) || !(m.damping > 0.0) ||
        !std::isfinite(m.alpha) || !(m.alpha > 0.0)) {
        loup::set_error("maxstep_a, damping and alpha must be finite and positive, got %g A, %g and %g", m.maxstep_a,
                        m.damping, m.alpha);
        return -1;
    }
    if (m.memory < 1 || m.memory > 1000) {
        loup::set_error("memory must be from 1 to 1000 history pairs, got %d", m.memory);
        return -1;
    }
    CUDA_CHECK(cudaSetDevice(h->device_id));
    const int N = h->N;
    const int n3 = 3 * N;
    const int gridN = (N + 255) / 256;
    const int mem = m.memory;
    MinState& st = h->min_state;
    // A new relaxation unless this call continues the last one: same settings, and nothing moved the atoms since.
    if (!st.valid || st.epoch != h->pos_epoch || st.n3 != n3 || st.memory != mem || st.maxstep != m.maxstep_a ||
        st.damping != m.damping || st.alpha != m.alpha) {
        if (st.n3 != n3 || st.memory != mem) {
            st.release();
            CUDA_CHECK(cudaMalloc(&st.d_S, (size_t)mem * n3 * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&st.d_Y, (size_t)mem * n3 * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&st.d_vec, (size_t)4 * n3 * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&st.d_small, (size_t)7 * mem * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&st.d_rows, (size_t)mem * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&st.d_max, 3 * sizeof(unsigned long long)));
            CUDA_CHECK(cudaMallocHost((void**)&st.h_pin, ((size_t)4 + 7 * (size_t)mem) * sizeof(double)));
            CUDA_CHECK(cudaMallocHost((void**)&st.h_rows, (size_t)mem * sizeof(int)));
            st.n3 = n3;
            st.memory = mem;
        }
        st.maxstep = m.maxstep_a;
        st.damping = m.damping;
        st.alpha = m.alpha;
        st.count = 0;
        st.head = 0;
        st.iteration = 0;
        st.order.clear();
        st.rho.assign((size_t)mem, 0.0);
        st.gram.assign((size_t)mem * mem, 0.0);
        st.yy.assign((size_t)mem * mem, 0.0);
        st.disp_known = st.step_known = -1.0;
        st.valid = true;
    }
    st.epoch = h->pos_epoch;
    double* d_g = st.d_vec;
    double* d_g0 = st.d_vec + n3;
    double* d_x0 = st.d_vec + 2 * (size_t)n3;
    double* d_p = st.d_vec + 3 * (size_t)n3;
    double* d_prod = st.d_small;
    double* d_u = st.d_small + 5 * (size_t)mem;
    double* d_w = st.d_small + 6 * (size_t)mem;
    const double h0 = 1.0 / m.alpha;
    const double half_skin = 0.5 * h->skin;

    struct ReorderOff {  // the optimiser's vectors stay in the atom order of the call
        LoupHandle* h;
        explicit ReorderOff(LoupHandle* handle) : h(handle) { h->suppress_reorder = true; }
        ~ReorderOff() { h->suppress_reorder = false; }
    } reorder_off(h);
    // The optimiser's own time is measured between events on the stream: from the end of a force pass to the end of
    // the history products (ev_products), and from the host's solve to the end of the move (ev_step). Everything
    // else of the call is the force evaluation: force passes, list checks and rebuilds, graph captures.
    EventPairGuard ev_products, ev_step;
    double optimizer_ms = 0.0;
    int rebuilds = 0, passes = 0, repeated = 0;
    auto force_pass = [&]() -> int {
        if (ensure_force_graph(h, 0) != 0) return -1;
        CUDA_CHECK(cudaGraphLaunch(h->force_graph_exec, h->stream));
        ++passes;
        return 0;
    };
    auto add_elapsed = [&](const EventPairGuard& ev) -> int {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.e0, ev.e1));
        optimizer_ms += (double)ms;
        return 0;
    };
    const auto t_start = std::chrono::steady_clock::now();
    if (h->cell_ref_stale && rebuild_neighbors(h) != 0) return -1;  // after a variable-cell run: see run_md_loop
    if (!h->forces_current || h->lt_cache.empty()) {
        if (force_pass() != 0) return -1;
    }

    double* pin = st.h_pin;                      // [0..2] d_max's values, [3] the energy
    double* prod = pin + 4;                      // five products per history row
    double* u = pin + 4 + 5 * (size_t)mem;       // u, then w: one upload
    double* w = u + mem;
    std::vector<double> a((size_t)mem), yz((size_t)mem);
    int taken = 0;
    bool converged = false;
    double fmax = 0.0, energy = 0.0;
    // The list check of a move, as an evaluation at new positions makes it: a rebuild once an atom is further than
    // half the skin from where the list was built. Where the displacement before the move plus twice the last
    // step's length stays inside that, the force pass is launched without waiting, and the displacement after the
    // move arrives with the next step's downloads (`unchecked`); should the list have been outdated after all,
    // that force pass is thrown away and run again with a new list. Otherwise (a call's first move of a new
    // relaxation, or a move that may cross the bound) the host waits for the measurement before the force pass.
    // LOUP_MIN_AHEAD=0|1 (testing) forces one of the two ways.
    const char* ahead_env = getenv("LOUP_MIN_AHEAD");
    const int ahead_forced = (ahead_env && *ahead_env) ? (atoi(ahead_env) != 0 ? 1 : 0) : -1;
    bool unchecked = false, step_timed = false;
    for (;;) {
        // The gradient and the largest force; from the second step on the newest pair goes into its row, and one
        // pass over the history gives every product with it and with the gradient.
        const bool have_previous = st.iteration > 0;
        const int row = st.head;
        const int n_rows = have_previous ? std::min(st.count + 1, mem) : 0;  // the history with the new pair
        CUDA_CHECK(cudaEventRecord(ev_products.e0, h->stream));
        CUDA_CHECK(cudaMemsetAsync(st.d_max, 0, sizeof(unsigned long long), h->stream));
        loup::min_gradient_kernel<<<gridN, 256, 0, h->stream>>>(
            N, have_previous ? 1 : 0, h->d_forces_f64, h->d_inv_masses, h->d_pos64, d_x0, d_g0, d_g,
            st.d_S + (size_t)row * n3, st.d_Y + (size_t)row * n3, st.d_max);
        if (have_previous) {
            CUDA_CHECK(cudaMemsetAsync(d_prod, 0, (size_t)5 * mem * sizeof(double), h->stream));
            const dim3 grid((unsigned)((n3 + 256 * 16 - 1) / (256 * 16)), (unsigned)n_rows);
            loup::lbfgs_products_kernel<<<grid, 256, 0, h->stream>>>(
                n3, n3, st.d_S, st.d_Y, d_g, st.d_S + (size_t)row * n3, st.d_Y + (size_t)row * n3, d_prod);
            CUDA_CHECK(cudaMemcpyAsync(prod, d_prod, (size_t)5 * n_rows * sizeof(double), cudaMemcpyDeviceToHost,
                                       h->stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(pin, st.d_max, 3 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaMemcpyAsync(pin + 3, h->d_energy_f64, sizeof(double), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaEventRecord(ev_products.e1, h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        if (add_elapsed(ev_products) != 0) return -1;
        if (step_timed) {
            if (add_elapsed(ev_step) != 0) return -1;
            step_timed = false;
        }
        if (taken > 0) st.step_known = m.damping * std::min(std::sqrt(pin[1]), m.maxstep_a);
        if (unchecked) {
            unchecked = false;
            if (pin[2] > half_skin * half_skin) {
                // The move outdated the list, so the force pass that followed it does not count: again, with a
                // new list.
                if (rebuild_neighbors(h) != 0) return -1;
                ++rebuilds;
                ++repeated;
                st.disp_known = 0.0;
                if (force_pass() != 0) return -1;
                continue;
            }
            st.disp_known = std::sqrt(pin[2]);
        }
        h->forces_current = true;
        const double inter_e = pin[3];
        if (!std::isfinite(inter_e)) {
            drop_hybrid(h);
            loup::set_error("non-finite potential energy during the relaxation: the step led to a configuration "
                            "the model cannot evaluate");
            return -1;
        }
        fmax = std::sqrt(pin[0]);
        energy = h->e0_total + inter_e;
        converged = fmax < m.fmax_ev_a;
        if (converged || taken >= max_steps) break;

        // The history gains the pair (ASE: from the second step on), the oldest one leaving once it is full.
        CUDA_CHECK(cudaEventRecord(ev_step.e0, h->stream));
        int n = 0;
        if (have_previous) {
            const double sy = prod[(size_t)5 * row + 2];  // s . y of the new pair
            if (!std::isfinite(sy) || sy == 0.0) {
                loup::set_error("the relaxation cannot go on: the last step changed neither positions nor forces "
                                "(s . y = %g)", sy);
                return -1;
            }
            if (st.count == mem) st.order.erase(st.order.begin());
            else ++st.count;
            st.order.push_back(row);
            st.head = (row + 1) % mem;
            st.rho[row] = 1.0 / sy;
            for (int j = 0; j < n_rows; ++j) {
                st.gram[(size_t)j * mem + row] = prod[(size_t)5 * j + 2];  // s_j . y_new
                st.gram[(size_t)row * mem + j] = prod[(size_t)5 * j + 3];  // s_new . y_j
                st.yy[(size_t)j * mem + row] = st.yy[(size_t)row * mem + j] = prod[(size_t)5 * j + 4];
            }
            n = st.count;
            // (I + R triu(G, 1)) a = R S g, newest to oldest; then Y z = H0 (Y g - (Y Y^T) a), and
            // (I + R tril(G^T, -1)) c = a - R Y z, oldest to newest. u = H0 a and w = -c weigh Y and S in p.
            for (int k = n - 1; k >= 0; --k) {
                const int i = st.order[k];
                double sum = prod[(size_t)5 * i + 0];
                for (int l = k + 1; l < n; ++l) sum -= st.gram[(size_t)i * mem + st.order[l]] * a[l];
                a[k] = st.rho[i] * sum;
            }
            for (int k = 0; k < n; ++k) {
                const int i = st.order[k];
                double sum = prod[(size_t)5 * i + 1];
                for (int l = 0; l < n; ++l) sum -= st.yy[(size_t)i * mem + st.order[l]] * a[l];
                yz[k] = h0 * sum;
            }
            for (int k = 0; k < n; ++k) {
                const int i = st.order[k];
                double sum = yz[k];
                for (int l = 0; l < k; ++l) sum -= st.gram[(size_t)st.order[l] * mem + i] * w[l];  // w = -c
                w[k] = -(a[k] - st.rho[i] * sum);
                u[k] = h0 * a[k];
                st.h_rows[k] = i;
            }
            // The host leaves u, w and the rows alone until the next wait, so the copies may run when they will.
            CUDA_CHECK(cudaMemcpyAsync(d_u, u, (size_t)(mem + n) * sizeof(double), cudaMemcpyHostToDevice,
                                       h->stream));
            CUDA_CHECK(cudaMemcpyAsync(st.d_rows, st.h_rows, (size_t)n * sizeof(int), cudaMemcpyHostToDevice,
                                       h->stream));
        }
        CUDA_CHECK(cudaMemsetAsync(st.d_max + 1, 0, 2 * sizeof(unsigned long long), h->stream));
        loup::lbfgs_direction_kernel<<<gridN, 256, 0, h->stream>>>(
            N, n, n3, h0, st.d_rows, d_u, d_w, st.d_S, st.d_Y, d_g, d_p, st.d_max + 1);
        loup::lbfgs_move_kernel<<<gridN, 256, 0, h->stream>>>(
            N, m.maxstep_a, m.damping, st.d_max + 1, d_p, d_g, h->d_pos64, d_x0, d_g0);
        CUDA_CHECK(cudaEventRecord(ev_step.e1, h->stream));
        step_timed = true;
        h->forces_current = false;
        ++st.iteration;
        ++taken;
        // The list at the new positions, then their forces.
        loup::max_displacement2_kernel<<<gridN, 256, 0, h->stream>>>(N, h->d_pos64, h->d_ref_pos64, st.d_max + 2);
        const bool ahead = half_skin > 0.0 &&
                           (ahead_forced >= 0 ? ahead_forced == 1
                                              : st.disp_known >= 0.0 && st.step_known >= 0.0 &&
                                                    st.disp_known + 2.0 * st.step_known < half_skin);
        if (ahead) {
            unchecked = true;
        } else {
            CUDA_CHECK(cudaMemcpyAsync(pin + 2, st.d_max + 2, sizeof(double), cudaMemcpyDeviceToHost, h->stream));
            CUDA_CHECK(cudaStreamSynchronize(h->stream));
            if (add_elapsed(ev_step) != 0) return -1;
            step_timed = false;
            if (!(half_skin > 0.0) || pin[2] > half_skin * half_skin) {
                if (rebuild_neighbors(h) != 0) return -1;
                ++rebuilds;
                st.disp_known = 0.0;
            } else {
                st.disp_known = std::sqrt(pin[2]);
            }
        }
        if (force_pass() != 0) return -1;
    }
    if (out_stats) {
        const double total_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
        out_stats->steps = taken;
        out_stats->converged = converged ? 1 : 0;
        out_stats->total_steps = (int)st.iteration;
        out_stats->rebuilds = rebuilds;
        out_stats->force_passes = passes;
        out_stats->repeated_passes = repeated;
        out_stats->max_force_ev_a = fmax;
        out_stats->potential_energy_ev = energy;
        out_stats->force_ms = std::max(0.0, total_ms - optimizer_ms);
        out_stats->optimizer_ms = optimizer_ms;
    }
    return 0;
}

// loup_set_cell: a new cell for the positions on the handle. With scale_atoms the atoms keep their fractional
// coordinates (x <- x old^-1 new, as ASE's set_cell(scale_atoms=True)). Velocities stay.
static int set_cell_impl(LoupHandle* h, const double* cell, int scale_atoms) {
    if (!has_system(h)) return -1;
    if (!h->nl_valid) {
        loup::set_error("no positions (call loup_set_state or loup_evaluate first)");
        return -1;
    }
    if (!cell) { loup::set_error("cell is NULL"); return -1; }
    if (!valid_input(0, cell, cell, nullptr)) return -1;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    if (scale_atoms) {
        loup::Mat3 m;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                m.m[r * 3 + c] = h->cell.inv[r * 3 + 0] * cell[0 * 3 + c] + h->cell.inv[r * 3 + 1] * cell[1 * 3 + c] +
                                 h->cell.inv[r * 3 + 2] * cell[2 * 3 + c];
        loup::positions_times_matrix_kernel<<<(h->N + 255) / 256, 256, 0, h->stream>>>(h->N, m, h->d_pos64);
    }
    memcpy(h->cell_in, cell, sizeof(h->cell_in));
    loup::compute_cell_mat(cell, h->cell, h->cell_vol, h->cell_heights);
    if (h->gen && h->gen->set_cell(cell)) h->invalidate_graphs();
    if (dispersion_cell_changed(h) != 0) return -1;
    h->forces_current = false;
    ++h->pos_epoch;
    return rebuild_neighbors(h);
}

// loup_set_reference_volume: the V0 of the cell limits, for a run that goes on in another process.
static int set_reference_volume_impl(LoupHandle* h, double volume_a3) {
    if (!has_system(h)) return -1;
    if (!std::isfinite(volume_a3) || !(volume_a3 > 0.0)) {
        loup::set_error("the reference volume must be finite and positive, got %g A^3", volume_a3);
        return -1;
    }
    h->cell_v0 = volume_a3;
    if (h->var_cell && h->d_celldev) {  // only this field: the device's cell may be ahead of the list's reference
        CUDA_CHECK(cudaSetDevice(h->device_id));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<char*>(h->d_celldev) + offsetof(loup::CellDev, v0), &volume_a3,
                              sizeof(double), cudaMemcpyHostToDevice));
    }
    return 0;
}

static int get_cell_impl(LoupHandle* h, double* out_cell) {
    if (!has_system(h)) return -1;
    if (!h->nl_valid) {
        loup::set_error("no positions (call loup_set_state or loup_evaluate first)");
        return -1;
    }
    if (!out_cell) { loup::set_error("out_cell is NULL"); return -1; }
    memcpy(out_cell, h->cell_in, sizeof(h->cell_in));
    return 0;
}

static int step_displacement_impl(LoupHandle* h, double* out_max_a) {
    if (!has_system(h)) return -1;
    if (!out_max_a) { loup::set_error("out_max_a is NULL"); return -1; }
    CUDA_CHECK(cudaSetDevice(h->device_id));
    unsigned int bits[2] = {0u, 0u};
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    CUDA_CHECK(cudaMemcpy(bits, h->d_max_step_disp2, sizeof(bits), cudaMemcpyDeviceToHost));
    const unsigned int larger = bits[0] > bits[1] ? bits[0] : bits[1];
    float d2 = 0.0f;
    memcpy(&d2, &larger, sizeof(d2));
    *out_max_a = std::sqrt((double)d2);
    return 0;
}

// loup_run: every integrator enters here; loup_run_nve and loup_run_langevin fill the struct and call it.
static const char* integrator_name(int kind) {
    switch (kind) {
        case LOUP_NVE: return "LOUP_NVE";
        case LOUP_LANGEVIN: return "LOUP_LANGEVIN";
        case LOUP_CSVR: return "LOUP_CSVR";
        case LOUP_NHC: return "LOUP_NHC";
        case LOUP_BERENDSEN_NVT: return "LOUP_BERENDSEN_NVT";
        case LOUP_BERENDSEN_NPT: return "LOUP_BERENDSEN_NPT";
        case LOUP_BAOAB_NPT: return "LOUP_BAOAB_NPT";
        case LOUP_MTK_NPT: return "LOUP_MTK_NPT";
        default: return nullptr;
    }
}

static int run_impl(LoupHandle* h, const LoupIntegrator* caller, int n_steps, LoupMDStats* out_stats) {
    if (!caller) { loup::set_error("the integrator is NULL"); return -1; }
    // Size-versioned: this is the first released layout, so nothing smaller exists, and a larger struct comes from
    // a header this library does not know.
    if (caller->size != sizeof(LoupIntegrator)) {
        loup::set_error("LoupIntegrator.size is %u, and this library's layout has %zu bytes (set size to "
                        "sizeof(LoupIntegrator) of the header the library was built from)", caller->size,
                        sizeof(LoupIntegrator));
        return -1;
    }
    const LoupIntegrator it = *caller;
    const char* name = integrator_name(it.kind);
    if (!name) { loup::set_error("unknown integrator kind %d", it.kind); return -1; }
    (void)name;  // every kind of the enum is built
    if (it.fix_com != 0 && h && h->n_fixed > 0) {
        loup::set_error("fix_com with fixed atoms: %d fixed atoms pin the frame already", h->n_fixed);
        return -1;
    }
    // The variable-cell step. No integrator of this version moves the cell, so it runs for tests only:
    // LOUP_VARIABLE_CELL=1 sends this call through it, and LOUP_DEBUG_CELL_RATE scales the cell and the positions
    // by 1 + rate in every step.
    if (h) {
        h->req_var_cell = false;
        h->req_cell_rate = 0.0;
        if (const char* e = std::getenv("LOUP_VARIABLE_CELL"); e && e[0] == '1') {
            if (h->gen && h->gen->has_lr()) {
                loup::set_error("the variable cell is not built for long-range models (MACE-POLAR-1): their k-space "
                                "terms are set up for one cell");
                return -1;
            }
            h->req_var_cell = true;
            if (const char* r = std::getenv("LOUP_DEBUG_CELL_RATE")) {
                const double rate = atof(r);
                if (!std::isfinite(rate) || std::fabs(rate) > 0.1) {
                    loup::set_error("LOUP_DEBUG_CELL_RATE must be within -0.1 to 0.1 per step, got %s", r);
                    return -1;
                }
                h->req_cell_rate = rate;
            }
        }
    }
    if (it.kind == LOUP_NVE) {
        return run_md_loop(h, 0, n_steps, it.dt_fs, 0.0, 0.0, 0.0, 0, it.thermo_every, it.fix_com != 0, out_stats);
    }
    if (!std::isfinite(it.temp_k) || it.temp_k < 0.0) {
        loup::set_error("temp_k must be finite and not negative, got %g K", it.temp_k);
        return -1;
    }
    if (it.kind == LOUP_LANGEVIN && (!std::isfinite(it.friction_per_fs) || it.friction_per_fs < 0.0)) {
        loup::set_error("friction_per_fs must be finite and not negative, got %g /fs", it.friction_per_fs);
        return -1;
    }
    if (it.kind == LOUP_BAOAB_NPT) {
        // ASE's LangevinBAOAB with a barostat: T_tau = 1 / friction_per_fs, P_tau = tau_p_fs (0: 20 T_tau), the
        // external stress minus pressure_gpa, hydrostatic or the full cell.
        if (!std::isfinite(it.friction_per_fs) || !(it.friction_per_fs > 0.0)) {
            loup::set_error("friction_per_fs must be finite and positive for LOUP_BAOAB_NPT (1 / T_tau), got %g /fs",
                            it.friction_per_fs);
            return -1;
        }
        if (!std::isfinite(it.tau_p_fs) || it.tau_p_fs < 0.0) {
            loup::set_error("tau_p_fs must be finite and not negative (0 takes 20 / friction_per_fs), got %g fs",
                            it.tau_p_fs);
            return -1;
        }
        if (!std::isfinite(it.pressure_gpa)) {
            loup::set_error("pressure_gpa must be finite, got %g GPa", it.pressure_gpa);
            return -1;
        }
        if (it.cell_mode != LOUP_CELL_HYDROSTATIC && it.cell_mode != LOUP_CELL_FULL) {
            loup::set_error("unknown cell_mode %d (LOUP_CELL_HYDROSTATIC or LOUP_CELL_FULL)", it.cell_mode);
            return -1;
        }
        if (it.fix_com != 0) {
            loup::set_error("fix_com is not part of LOUP_BAOAB_NPT: ASE's LangevinBAOAB leaves the centre of mass "
                            "to its thermostat");
            return -1;
        }
        if (!std::isfinite(it.temp_rate_k_per_step) ||
            it.temp_k + it.temp_rate_k_per_step * (double)n_steps < 0.0) {
            loup::set_error("temp_rate_k_per_step must be finite and must not take the target below 0 K: %g K + %d "
                            "steps of %g K", it.temp_k, n_steps, it.temp_rate_k_per_step);
            return -1;
        }
        if (h && h->n_fixed > 0) {
            loup::set_error("a barostat with fixed atoms: the cell's scaling would move the %d fixed atoms", h->n_fixed);
            return -1;
        }
        if (h && h->gen && h->gen->has_lr()) {
            loup::set_error("the variable cell is not built for long-range models (MACE-POLAR-1): their k-space "
                            "terms are set up for one cell");
            return -1;
        }
        NptParams npt;
        npt.pressure = it.pressure_gpa / loup::GPA_PER_EV_A3;
        npt.tau_p_fs = it.tau_p_fs;
        npt.full = it.cell_mode == LOUP_CELL_FULL ? 1 : 0;
        if (h) h->req_var_cell = true;
        return run_md_loop(h, 6, n_steps, it.dt_fs, it.temp_k, it.temp_rate_k_per_step, it.friction_per_fs, it.seed,
                           it.thermo_every, false, out_stats, 0.0, 0, 1, 0.0, &npt);
    }
    if ((it.kind == LOUP_CSVR || it.kind == LOUP_NHC || it.kind == LOUP_BERENDSEN_NVT ||
         it.kind == LOUP_BERENDSEN_NPT || it.kind == LOUP_MTK_NPT) &&
        (!std::isfinite(it.tau_t_fs) || !(it.tau_t_fs > 0.0))) {
        loup::set_error("tau_t_fs must be finite and positive, got %g fs", it.tau_t_fs);
        return -1;
    }
    if (!std::isfinite(it.temp_rate_k_per_step) || it.temp_k + it.temp_rate_k_per_step * (double)n_steps < 0.0) {
        loup::set_error("temp_rate_k_per_step must be finite and must not take the target below 0 K: %g K + %d steps "
                        "of %g K", it.temp_k, n_steps, it.temp_rate_k_per_step);
        return -1;
    }
    if (it.kind == LOUP_CSVR) {
        return run_md_loop(h, 2, n_steps, it.dt_fs, it.temp_k, it.temp_rate_k_per_step, 0.0, it.seed,
                           it.thermo_every, it.fix_com != 0, out_stats, it.tau_t_fs);
    }
    if (it.kind == LOUP_MTK_NPT) {
        // ASE's IsotropicMTKNPT (LOUP_CELL_HYDROSTATIC) and MTKNPT (LOUP_CELL_FULL): tdamp = tau_t_fs, pdamp =
        // tau_p_fs, tchain / tloop = chain_length / chain_loops, pchain / ploop = p_chain_length / p_chain_loops
        // (0: ASE's 3 and 1).
        if (!std::isfinite(it.tau_p_fs) || !(it.tau_p_fs > 0.0)) {
            loup::set_error("tau_p_fs must be finite and positive, got %g fs", it.tau_p_fs);
            return -1;
        }
        if (!std::isfinite(it.pressure_gpa)) {
            loup::set_error("pressure_gpa must be finite, got %g GPa", it.pressure_gpa);
            return -1;
        }
        const int tchain = it.chain_length == 0 ? 3 : it.chain_length;
        const int pchain = it.p_chain_length == 0 ? 3 : it.p_chain_length;
        if (tchain < 1 || tchain > 8) {
            loup::set_error("chain_length must be from 1 to 8 thermostats for LOUP_MTK_NPT, got %d", it.chain_length);
            return -1;
        }
        if (pchain < 1 || pchain > 8) {
            loup::set_error("p_chain_length must be from 1 to 8 barostat links, got %d", it.p_chain_length);
            return -1;
        }
        if (it.chain_loops < 0 || it.chain_loops > 100 || it.p_chain_loops < 0 || it.p_chain_loops > 100) {
            loup::set_error("%s must be from 0 to 100 (0 is 1), got %d",
                            it.chain_loops < 0 || it.chain_loops > 100 ? "chain_loops" : "p_chain_loops",
                            it.chain_loops < 0 || it.chain_loops > 100 ? it.chain_loops : it.p_chain_loops);
            return -1;
        }
        if (it.cell_mode != LOUP_CELL_HYDROSTATIC && it.cell_mode != LOUP_CELL_FULL) {
            loup::set_error("unknown cell_mode %d (LOUP_CELL_HYDROSTATIC or LOUP_CELL_FULL)", it.cell_mode);
            return -1;
        }
        const double mass_temp = it.mass_temp_k == 0.0 ? it.temp_k : it.mass_temp_k;
        if (!std::isfinite(mass_temp) || !(mass_temp > 0.0) || !(it.temp_k > 0.0)) {
            loup::set_error("temp_k must be positive for LOUP_MTK_NPT (the thermostat's and the barostat's masses "
                            "are made for it), got %g K with mass_temp_k = %g K", it.temp_k, it.mass_temp_k);
            return -1;
        }
        if (h && h->n_fixed > 0) {
            loup::set_error("a barostat with fixed atoms: the cell's scaling would move the %d fixed atoms", h->n_fixed);
            return -1;
        }
        if (h && h->gen && h->gen->has_lr()) {
            loup::set_error("the variable cell is not built for long-range models (MACE-POLAR-1): their k-space "
                            "terms are set up for one cell");
            return -1;
        }
        NptParams npt;
        npt.pressure = it.pressure_gpa / loup::GPA_PER_EV_A3;
        npt.tau_p_fs = it.tau_p_fs;
        npt.full = it.cell_mode == LOUP_CELL_FULL ? 1 : 0;
        npt.pchain = pchain;
        npt.ploop = it.p_chain_loops == 0 ? 1 : it.p_chain_loops;
        if (h) h->req_var_cell = true;
        return run_md_loop(h, 7, n_steps, it.dt_fs, it.temp_k, it.temp_rate_k_per_step, 0.0, 0, it.thermo_every,
                           it.fix_com != 0, out_stats, it.tau_t_fs, tchain, it.chain_loops == 0 ? 1 : it.chain_loops,
                           mass_temp, &npt);
    }
    if (it.kind == LOUP_BERENDSEN_NVT || it.kind == LOUP_BERENDSEN_NPT) {
        if (it.tau_t_fs < it.dt_fs) {  // sqrt(1 + (T0 / T - 1) dt / taut) needs dt / taut <= 1
            loup::set_error("tau_t_fs must be at least the timestep for the Berendsen thermostat: %g fs against "
                            "dt_fs = %g fs", it.tau_t_fs, it.dt_fs);
            return -1;
        }
        if (it.kind == LOUP_BERENDSEN_NVT) {
            return run_md_loop(h, 4, n_steps, it.dt_fs, it.temp_k, it.temp_rate_k_per_step, 0.0, 0,
                               it.thermo_every, it.fix_com != 0, out_stats, it.tau_t_fs);
        }
        // The barostat (ASE's NPTBerendsen; with LOUP_CELL_FULL its Inhomogeneous_NPTBerendsen and the mask).
        if (!std::isfinite(it.tau_p_fs) || !(it.tau_p_fs > 0.0)) {
            loup::set_error("tau_p_fs must be finite and positive, got %g fs", it.tau_p_fs);
            return -1;
        }
        if (!std::isfinite(it.pressure_gpa)) {
            loup::set_error("pressure_gpa must be finite, got %g GPa", it.pressure_gpa);
            return -1;
        }
        if (!std::isfinite(it.compressibility_per_gpa) || !(it.compressibility_per_gpa > 0.0)) {
            loup::set_error("compressibility_per_gpa must be finite and positive, got %g per GPa",
                            it.compressibility_per_gpa);
            return -1;
        }
        if (it.cell_mode != LOUP_CELL_HYDROSTATIC && it.cell_mode != LOUP_CELL_FULL) {
            loup::set_error("unknown cell_mode %d (LOUP_CELL_HYDROSTATIC or LOUP_CELL_FULL)", it.cell_mode);
            return -1;
        }
        NptParams npt;
        npt.pressure = it.pressure_gpa / loup::GPA_PER_EV_A3;
        npt.tau_p_fs = it.tau_p_fs;
        npt.compressibility = it.compressibility_per_gpa * loup::GPA_PER_EV_A3;
        npt.full = it.cell_mode == LOUP_CELL_FULL ? 1 : 0;
        if (npt.full) {
            int freed = 0;
            for (int a = 0; a < 3; ++a) {
                if (it.cell_mask[a] != 0 && it.cell_mask[a] != 1) {
                    loup::set_error("cell_mask entries must be 0 or 1, got (%d, %d, %d)", it.cell_mask[0],
                                    it.cell_mask[1], it.cell_mask[2]);
                    return -1;
                }
                npt.mask[a] = it.cell_mask[a];
                freed += it.cell_mask[a];
            }
            if (freed == 0) {
                loup::set_error("cell_mask frees no axis: (0, 0, 0) is a run at fixed cell (LOUP_BERENDSEN_NVT)");
                return -1;
            }
        }
        if (h && h->n_fixed > 0) {
            loup::set_error("a barostat with fixed atoms: the cell's scaling would move the %d fixed atoms", h->n_fixed);
            return -1;
        }
        if (h && h->gen && h->gen->has_lr()) {
            loup::set_error("the variable cell is not built for long-range models (MACE-POLAR-1): their k-space "
                            "terms are set up for one cell");
            return -1;
        }
        if (h) h->req_var_cell = true;
        return run_md_loop(h, 5, n_steps, it.dt_fs, it.temp_k, it.temp_rate_k_per_step, 0.0, 0, it.thermo_every,
                           it.fix_com != 0, out_stats, it.tau_t_fs, 0, 1, 0.0, &npt);
    }
    if (it.kind == LOUP_NHC) {
        if (it.chain_length < 1 || it.chain_length > 16) {
            loup::set_error("chain_length must be from 1 to 16 thermostats, got %d", it.chain_length);
            return -1;
        }
        if (it.chain_loops < 0 || it.chain_loops > 100) {
            loup::set_error("chain_loops must be from 0 (one) to 100 sub-steps, got %d", it.chain_loops);
            return -1;
        }
        const double mass_temp = it.mass_temp_k != 0.0 ? it.mass_temp_k : it.temp_k;
        if (!std::isfinite(mass_temp) || !(mass_temp > 0.0)) {
            loup::set_error("the thermostat masses of the Nose-Hoover chain need a temperature above 0 K: "
                            "mass_temp_k is %g K and temp_k %g K", it.mass_temp_k, it.temp_k);
            return -1;
        }
        return run_md_loop(h, 3, n_steps, it.dt_fs, it.temp_k, it.temp_rate_k_per_step, 0.0, 0, it.thermo_every,
                           it.fix_com != 0, out_stats, it.tau_t_fs, it.chain_length,
                           it.chain_loops > 0 ? it.chain_loops : 1, mass_temp);
    }
    return run_md_loop(h, 1, n_steps, it.dt_fs, it.temp_k, it.temp_rate_k_per_step, it.friction_per_fs, it.seed,
                       it.thermo_every, it.fix_com != 0, out_stats);
}

static int run_nve_impl(LoupHandle* h, int n_steps, double dt_fs, LoupMDStats* out_stats) {
    LoupIntegrator it{};
    it.size = sizeof(it);
    it.kind = LOUP_NVE;
    it.dt_fs = dt_fs;
    return run_impl(h, &it, n_steps, out_stats);
}

static int run_langevin_impl(
    LoupHandle* h, int n_steps, double dt_fs, double temp_k,
    double friction_per_fs, unsigned long long seed, LoupMDStats* out_stats
) {
    LoupIntegrator it{};
    it.size = sizeof(it);
    it.kind = LOUP_LANGEVIN;
    it.dt_fs = dt_fs;
    it.temp_k = temp_k;
    it.friction_per_fs = friction_per_fs;
    it.seed = seed;
    return run_impl(h, &it, n_steps, out_stats);
}

static int get_state_impl(LoupHandle* h, double* out_positions, double* out_velocities, double* out_forces, double* out_energy_ev) {
    if (!has_system(h)) return -1;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    std::vector<double> tmp((size_t)h->N * 3);
    if (out_positions) {
        CUDA_CHECK(cudaMemcpyAsync(tmp.data(), h->d_pos64, (size_t)h->N * 3 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        for (int dst = 0; dst < h->N; ++dst) {
            int orig = h->perm[dst];
            out_positions[orig * 3 + 0] = tmp[dst * 3 + 0];
            out_positions[orig * 3 + 1] = tmp[dst * 3 + 1];
            out_positions[orig * 3 + 2] = tmp[dst * 3 + 2];
        }
    }
    if (out_velocities) {
        CUDA_CHECK(cudaMemcpyAsync(tmp.data(), h->d_vel64, (size_t)h->N * 3 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        for (int dst = 0; dst < h->N; ++dst) {
            int orig = h->perm[dst];
            out_velocities[orig * 3 + 0] = tmp[dst * 3 + 0];
            out_velocities[orig * 3 + 1] = tmp[dst * 3 + 1];
            out_velocities[orig * 3 + 2] = tmp[dst * 3 + 2];
        }
    }
    if (out_forces) {
        CUDA_CHECK(cudaMemcpyAsync(tmp.data(), h->d_forces_f64, (size_t)h->N * 3 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        for (int dst = 0; dst < h->N; ++dst) {
            int orig = h->perm[dst];
            out_forces[orig * 3 + 0] = tmp[dst * 3 + 0];
            out_forces[orig * 3 + 1] = tmp[dst * 3 + 1];
            out_forces[orig * 3 + 2] = tmp[dst * 3 + 2];
        }
    }
    if (out_energy_ev) {
        double inter_e = 0.0;
        CUDA_CHECK(cudaMemcpyAsync(&inter_e, h->d_energy_f64, sizeof(double), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        *out_energy_ev = h->e0_total + inter_e;
    }
    return 0;
}

// The state blob (loup_export_state / loup_import_state): what a restart needs beyond the system itself, in the
// caller's atom order: positions, velocities, the step counter (the offset of the noise streams) and the
// integrator's scalars (kIntStateSlots: CSVR's transferred energy) with the kind they belong to. Forces and the
// neighbour list are not stored: import rebuilds the list, and the next run starts with a force pass.
//   StateHeader | positions [N, 3] f64 | velocities [N, 3] f64 | FNV-1a of everything before, u64
struct StateHeader {
    char magic[8];            // "LOUPSTA1"
    uint32_t version;         // 1
    uint32_t header_bytes;    // sizeof(StateHeader)
    uint64_t n_atoms;
    uint64_t species_hash;    // atomic numbers in caller order: a blob only fits the system it was written for
    uint64_t step_counter;
    uint32_t max_step_disp2[2];  // the displacement tracker: current and previous window (float bits)
    double cell[9];
    int32_t integrator_kind;     // the LOUP_* kind whose scalars follow; -1: none
    uint32_t reserved;
    double integrator_state[kIntStateSlots];
};
static_assert(sizeof(StateHeader) == 128 + 8 * kIntStateSlots, "StateHeader has padding");
static const char kStateMagic[8] = {'L', 'O', 'U', 'P', 'S', 'T', 'A', '1'};
static size_t state_blob_bytes(int n_atoms) {
    return sizeof(StateHeader) + (size_t)n_atoms * 6 * sizeof(double) + sizeof(uint64_t);
}

static int export_state_impl(LoupHandle* h, void* buffer, size_t capacity, size_t* written) {
    if (!has_system(h)) return -1;
    if (!h->nl_valid) {
        loup::set_error("no positions to export (call loup_set_state or loup_evaluate first)");
        return -1;
    }
    const size_t need = state_blob_bytes(h->N);
    if (written) *written = need;
    if (!buffer && capacity == 0) return 0;  // size query
    if (!buffer || capacity < need) {
        loup::set_error("the buffer holds %zu bytes, and the state of %d atoms needs %zu", buffer ? capacity : (size_t)0,
                        h->N, need);
        return -1;
    }
    CUDA_CHECK(cudaSetDevice(h->device_id));
    const size_t n3 = (size_t)h->N * 3;
    std::vector<double> p_sorted(n3), v_sorted(n3);
    StateHeader hd{};
    memcpy(hd.magic, kStateMagic, 8);
    hd.version = 1;
    hd.header_bytes = sizeof(StateHeader);
    hd.n_atoms = (uint64_t)h->N;
    hd.species_hash = h->species_hash;
    memcpy(hd.cell, h->cell_in, sizeof(hd.cell));
    CUDA_CHECK(cudaMemcpyAsync(p_sorted.data(), h->d_pos64, n3 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(v_sorted.data(), h->d_vel64, n3 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(&hd.step_counter, h->d_step_counter, sizeof(hd.step_counter), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(hd.max_step_disp2, h->d_max_step_disp2, sizeof(hd.max_step_disp2), cudaMemcpyDeviceToHost, h->stream));
    hd.integrator_kind = h->int_state_kind;
    CUDA_CHECK(cudaMemcpyAsync(hd.integrator_state, h->d_int_state, sizeof(hd.integrator_state), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    std::vector<double> body(2 * n3);
    for (int dst = 0; dst < h->N; ++dst) {
        const size_t orig = (size_t)h->perm[dst];
        for (int k = 0; k < 3; ++k) {
            body[orig * 3 + k] = p_sorted[(size_t)dst * 3 + k];
            body[n3 + orig * 3 + k] = v_sorted[(size_t)dst * 3 + k];
        }
    }
    unsigned char* out = static_cast<unsigned char*>(buffer);
    memcpy(out, &hd, sizeof(hd));
    memcpy(out + sizeof(hd), body.data(), body.size() * sizeof(double));
    const uint64_t sum = fnv1a64(out, need - sizeof(uint64_t));
    memcpy(out + need - sizeof(uint64_t), &sum, sizeof(sum));
    return 0;
}

static int import_state_impl(LoupHandle* h, const void* buffer, size_t size) {
    if (!has_system(h)) return -1;
    if (!buffer) { loup::set_error("the buffer is NULL"); return -1; }
    StateHeader hd;
    if (size < sizeof(hd)) {
        loup::set_error("the state blob has %zu bytes, fewer than its header (%zu)", size, sizeof(hd));
        return -1;
    }
    const unsigned char* in = static_cast<const unsigned char*>(buffer);
    memcpy(&hd, in, sizeof(hd));
    if (memcmp(hd.magic, kStateMagic, 8) != 0) { loup::set_error("not a LoupMD state blob"); return -1; }
    if (hd.version != 1) {
        loup::set_error("the state blob has version %u, and this library reads version 1", hd.version);
        return -1;
    }
    if (hd.header_bytes != sizeof(StateHeader)) {
        loup::set_error("the state blob's header has %u bytes, and this library's has %zu (written by another "
                        "build of the library)", hd.header_bytes, sizeof(StateHeader));
        return -1;
    }
    if (hd.n_atoms != (uint64_t)h->N) {
        loup::set_error("the state blob holds %llu atoms, and the handle's system has %d",
                        (unsigned long long)hd.n_atoms, h->N);
        return -1;
    }
    const size_t need = state_blob_bytes(h->N);
    if (size != need) {
        loup::set_error("the state blob has %zu bytes, and %zu are expected for %d atoms (truncated or padded)", size,
                        need, h->N);
        return -1;
    }
    uint64_t sum = 0;
    memcpy(&sum, in + need - sizeof(sum), sizeof(sum));
    if (sum != fnv1a64(in, need - sizeof(sum))) {
        loup::set_error("the state blob is damaged (its checksum does not match)");
        return -1;
    }
    if (hd.species_hash != h->species_hash) {
        loup::set_error("the state blob was written for other atomic numbers or another atom order");
        return -1;
    }
    const size_t n3 = (size_t)h->N * 3;
    std::vector<double> body(2 * n3);  // the caller's buffer need not be aligned
    memcpy(body.data(), in + sizeof(hd), body.size() * sizeof(double));
    const double* positions = body.data();
    const double* velocities = body.data() + n3;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    if (!valid_input(h->N, positions, hd.cell, velocities)) return -1;
    if (upload_positions(h, positions, hd.cell) < 0) return -1;
    if (!(h->cell_v0 > 0.0)) h->cell_v0 = h->cell_vol;  // a handle's first state (loup_set_reference_volume: a resumed run's)
    std::vector<double> v_sorted(n3);
    for (int dst = 0; dst < h->N; ++dst) {
        const size_t src = (size_t)h->perm[dst];
        for (int k = 0; k < 3; ++k) v_sorted[(size_t)dst * 3 + k] = h->fixed_caller[src] ? 0.0 : velocities[src * 3 + k];
    }
    CUDA_CHECK(cudaMemcpyAsync(h->d_vel64, v_sorted.data(), n3 * sizeof(double), cudaMemcpyHostToDevice, h->stream));
    if (rebuild_neighbors(h) != 0) return -1;
    CUDA_CHECK(cudaMemcpyAsync(h->d_step_counter, &hd.step_counter, sizeof(hd.step_counter), cudaMemcpyHostToDevice, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(h->d_max_step_disp2, hd.max_step_disp2, sizeof(hd.max_step_disp2), cudaMemcpyHostToDevice, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(h->d_int_state, hd.integrator_state, sizeof(hd.integrator_state), cudaMemcpyHostToDevice, h->stream));
    h->int_state_kind = hd.integrator_kind;
    h->disp_window_steps = 0;
    h->md_steps = 0;
    h->thermo_pending = 0;
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    return 0;
}

// ---- Open domains (plan task S1-T09), testing only ----
// A domain whose ghosts are images of its own local atoms needs no other domain to supply its ghost rows: after a
// sweep the ghosts' rows are copied from the locals they are images of, and what a sweep adds to ghost rows (the
// gradient of a layer's output, forces) is folded back onto those locals. With the periodic images of a cell as
// ghosts this must reproduce the periodic call.
extern "C++" {  // kernels, one of them a template: not inside this file's C linkage block
__global__ void ghost_copy_rows_kernel(int n_local, int n_ghost, int width, const int* __restrict__ ghost_of,
                                       float* rows) {
    const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (int64_t)n_ghost * width) return;
    const int g = (int)(k / width), c = (int)(k % width);
    rows[(int64_t)(n_local + g) * width + c] = rows[(int64_t)ghost_of[g] * width + c];
}

template <typename T>
__global__ void ghost_fold_rows_kernel(int n_local, int n_ghost, int width, const int* __restrict__ ghost_of, T* rows) {
    const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (int64_t)n_ghost * width) return;
    const int g = (int)(k / width), c = (int)(k % width);
    atomicAdd(&rows[(int64_t)ghost_of[g] * width + c], rows[(int64_t)(n_local + g) * width + c]);
}
}  // extern "C++"

static int set_domain_impl(LoupHandle* h, int n_local, int n_ghost, const int* atomic_numbers, double skin) {
    if (n_local <= 0 || n_ghost < 0) {
        loup::set_error("n_local must be positive and n_ghost not negative, got %d and %d", n_local, n_ghost);
        return -1;
    }
    if (!(h->gen && h->gen->sweeps_available())) {
        loup::set_error("an open domain needs a bundle of the generic path without long range, and "
                        "LOUP_CHAIN_FOLD=0 where the skip-chain fold applies");
        return -1;
    }
    return set_system_impl(h, n_local + n_ghost, atomic_numbers, nullptr, skin, n_ghost);
}

static int evaluate_domain_impl(LoupHandle* h, const double* positions, const double* cell, const int* ghost_of,
                                double* out_energy_ev, double* out_forces, double* out_stress) {
    if (!h || h->N <= 0) { loup::set_error("no domain is set (call loup_debug_set_domain first)"); return -1; }
    if (h->disp) { loup::set_error("the dispersion term has no open domains (PLAN.md S6-D09)"); return -1; }
    if (!(h->gen && h->gen->sweeps_available())) {
        loup::set_error("the handle's model has no sweeps (generic path without long range and chain fold)");
        return -1;
    }
    const int N = h->N, G = h->n_ghost;
    if (G > 0 && !ghost_of) { loup::set_error("ghost_of is NULL"); return -1; }
    CUDA_CHECK(cudaSetDevice(h->device_id));
    if (upload_positions(h, positions, cell) < 0) return -1;
    if (rebuild_neighbors(h) != 0) return -1;
    h->invalidate_graphs();  // nothing below is captured, and no graph of another call may outlive this state

    // ghost_of names locals in the caller's order; the device rows of the locals are grouped by species
    std::vector<int> row_of(N);
    for (int dst = 0; dst < N; ++dst) row_of[h->perm[dst]] = dst;
    std::vector<int> map((size_t)std::max(G, 1), 0);
    for (int g = 0; g < G; ++g) {
        if (ghost_of[g] < 0 || ghost_of[g] >= N) {
            loup::set_error("ghost_of[%d] = %d is not a local atom (0 .. %d)", g, ghost_of[g], N - 1);
            return -1;
        }
        map[g] = row_of[ghost_of[g]];
    }
    DeviceBufferGuard d_map;
    CUDA_CHECK(cudaMalloc(&d_map.p, map.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_map.p, map.data(), map.size() * sizeof(int), cudaMemcpyHostToDevice));
    const int* dmap = static_cast<const int*>(d_map.p);

    const int L = h->gen->n_layers();
    DeviceBufferGuard adj[4];  // the locals' rows of a layer output's gradient, per degree, between two sweeps
    for (int l = 0; l < 4; ++l) {
        size_t floats = 0;
        for (int t = 0; t + 1 < L; ++t)
            if (l <= h->gen->out_lmax(t)) floats = std::max(floats, (size_t)N * (2 * l + 1) * h->gen->out_width(t));
        if (floats) CUDA_CHECK(cudaMalloc(&adj[l].p, floats * sizeof(float)));
    }
    struct NoFork {  // the GEMMs are tuned on first use here, and their timings must not overlap
        LoupHandle* h;
        bool saved;
        explicit NoFork(LoupHandle* h_) : h(h_), saved(h_->fork_ok) { h->fork_ok = false; }
        ~NoFork() { h->fork_ok = saved; }
    } no_fork(h);
    const int stress = out_stress ? 1 : 0;
    auto grid = [](int64_t n) { return (unsigned)((n + 255) / 256); };

    CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(h->d_zero_slab) + h->zero_tail_off, 0,
                               h->zero_slab_bytes - h->zero_tail_off, h->stream));
    for (int t = 0; t + 1 < L; ++t) {  // forward sweeps: each leaves its layer's output; the ghosts copy theirs
        DomainSweep sw;
        sw.kind = loup::GenericPassInputs::SweepForward;
        sw.layer = t;
        record_force_pass(h, stress, false, &sw);
        for (int l = 0; G > 0 && l <= h->gen->out_lmax(t); ++l) {
            const int width = (2 * l + 1) * h->gen->out_width(t);
            ghost_copy_rows_kernel<<<grid((int64_t)G * width), 256, 0, h->stream>>>(N, G, width, dmap,
                                                                                    h->gen->layer_out(t, l));
        }
    }
    {
        DomainSweep sw;
        sw.kind = loup::GenericPassInputs::SweepLast;
        record_force_pass(h, stress, false, &sw);
    }
    for (int b = L - 2; b >= 0; --b) {  // backward sweeps: fold the ghosts' gradient rows, hand the locals' on
        DomainSweep sw;
        sw.kind = loup::GenericPassInputs::SweepBackward;
        sw.layer = b;
        for (int l = 0; l <= h->gen->out_lmax(b); ++l) {
            const int width = (2 * l + 1) * h->gen->out_width(b);
            float* grad = h->gen->layer_out_grad(b, l);
            if (G > 0)
                ghost_fold_rows_kernel<float><<<grid((int64_t)G * width), 256, 0, h->stream>>>(N, G, width, dmap, grad);
            CUDA_CHECK(cudaMemcpyAsync(adj[l].p, grad, (size_t)N * width * sizeof(float), cudaMemcpyDeviceToDevice,
                                       h->stream));
            sw.adjoint[l] = static_cast<const float*>(adj[l].p);
        }
        record_force_pass(h, stress, false, &sw);
    }
    if (G > 0)
        ghost_fold_rows_kernel<double><<<grid((int64_t)G * 3), 256, 0, h->stream>>>(N, G, 3, dmap, h->d_forces_f64);

    double inter_e = 0.0, v_grad[6] = {0, 0, 0, 0, 0, 0};
    std::vector<double> f_sorted((size_t)N * 3);
    CUDA_CHECK(cudaMemcpyAsync(&inter_e, h->d_energy_f64, sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(f_sorted.data(), h->d_forces_f64, (size_t)N * 3 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(v_grad, h->d_virial_grad_f64, 6 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    CUDA_CHECK(cudaGetLastError());
    h->save_disk_plans();
    if (out_energy_ev) *out_energy_ev = h->e0_total + inter_e;
    if (out_forces) {
        for (int dst = 0; dst < N; ++dst) {
            const int orig = h->perm[dst];
            for (int k = 0; k < 3; ++k) out_forces[orig * 3 + k] = f_sorted[(size_t)dst * 3 + k];
        }
    }
    if (out_stress) {  // per volume of the cell given, which for an open domain is only its box
        const double inv_v = 1.0 / h->cell_vol;
        out_stress[0] = v_grad[0] * inv_v; out_stress[1] = v_grad[1] * inv_v; out_stress[2] = v_grad[2] * inv_v;
        out_stress[3] = v_grad[1] * inv_v; out_stress[4] = v_grad[3] * inv_v; out_stress[5] = v_grad[4] * inv_v;
        out_stress[6] = v_grad[2] * inv_v; out_stress[7] = v_grad[4] * inv_v; out_stress[8] = v_grad[5] * inv_v;
    }
    return 0;
}

// ---- A tile's neighbour list (plan task S5-T02), testing only ----
// The list of cutoff plus skin the handle builds for `positions` (a periodic system, or an open domain whose ghosts
// are a tile's ghost candidates), copied out in the caller's order: receivers are locals, senders locals or ghosts
// (n_local + g for ghost g), shifts the Cartesian offsets of the engine's convention (edge vector = receiver -
// sender + shift). ghost_used[g] (optional) is 1 for each ghost that sends along some edge: the others are candidates
// the tile can drop. With capacity below the count nothing but the count is written.
extern "C++" {
__global__ void mark_ghost_senders_kernel(int E, int n_local, const int* __restrict__ senders,
                                          unsigned char* __restrict__ used) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < E && senders[e] >= n_local) used[senders[e] - n_local] = 1;
}
}  // extern "C++"

static int domain_edges_impl(LoupHandle* h, const double* positions, const double* cell, long long capacity,
                             int* out_receivers, int* out_senders, double* out_shifts, long long* out_count,
                             unsigned char* out_ghost_used) {
    if (!h || h->N <= 0) { loup::set_error("no system is set"); return -1; }
    if (!out_count) { loup::set_error("out_count is NULL"); return -1; }
    const int N = h->N, G = h->n_ghost;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    if (upload_positions(h, positions, cell) < 0) return -1;
    if (rebuild_neighbors(h) != 0) return -1;
    const int E = h->E_skin;
    *out_count = E;
    if (out_ghost_used && G > 0) {
        DeviceBufferGuard used;
        CUDA_CHECK(cudaMalloc(&used.p, G));
        CUDA_CHECK(cudaMemsetAsync(used.p, 0, G, h->stream));
        if (E > 0)
            mark_ghost_senders_kernel<<<(E + 255) / 256, 256, 0, h->stream>>>(
                E, N, h->d_senders, static_cast<unsigned char*>(used.p));
        CUDA_CHECK(cudaMemcpyAsync(out_ghost_used, used.p, G, cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
    }
    if (capacity < E || E == 0) return 0;
    std::vector<int> rcv(E), snd(E);
    std::vector<double> sh((size_t)E * 3);
    CUDA_CHECK(cudaMemcpyAsync(rcv.data(), h->d_receivers, (size_t)E * sizeof(int), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(snd.data(), h->d_senders, (size_t)E * sizeof(int), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(sh.data(), h->d_shift_cart, (size_t)E * 3 * sizeof(double), cudaMemcpyDeviceToHost,
                               h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    for (int e = 0; e < E; ++e) {
        if (out_receivers) out_receivers[e] = h->perm[rcv[e]];
        if (out_senders) out_senders[e] = h->perm[snd[e]];
        if (out_shifts)
            for (int k = 0; k < 3; ++k) out_shifts[(size_t)e * 3 + k] = sh[(size_t)e * 3 + k];
    }
    return 0;
}

// ---- The tiled call (plan task S5-T03), testing only until S5-T05 makes it a mode of the public calls ----
// A whole periodic system evaluated tile by tile (src/tiled.hpp): every tile is an open domain of its locals and the
// ghost candidates that send along some edge of its list. The per-layer sweeps run over all tiles in turn, and the
// state between sweeps lives in whole-system arrays on the device: each forward sweep writes its locals' rows of
// the layer's output, the next sweep gathers the rows of its locals and ghosts; the last sweep and each backward
// sweep scatter-add the gradient rows of every row (ghosts fold onto the atom they are an image of), and the
// backward sweep of a layer takes its locals' gradient rows as its adjoint. Forces fold the same way.
// Plain launches, no graphs; every tile sweep sets its domain up again (S5-T06 keeps lists, S5-T10 the speed).
extern "C++" {
__global__ void tile_gather_rows_kernel(int rows, int width, const int* __restrict__ row_atom,
                                        const float* __restrict__ whole, float* __restrict__ tile) {
    const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (int64_t)rows * width) return;
    const int r = (int)(k / width), c = (int)(k % width);
    tile[k] = whole[(int64_t)row_atom[r] * width + c];
}

__global__ void tile_scatter_rows_kernel(int rows, int width, const int* __restrict__ row_atom,
                                         const float* __restrict__ tile, float* __restrict__ whole) {
    const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (int64_t)rows * width) return;
    const int r = (int)(k / width), c = (int)(k % width);
    whole[(int64_t)row_atom[r] * width + c] = tile[k];
}

template <typename T>
__global__ void tile_scatter_add_rows_kernel(int rows, int width, const int* __restrict__ row_atom,
                                             const T* __restrict__ tile, T* __restrict__ whole) {
    const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (int64_t)rows * width) return;
    const int r = (int)(k / width), c = (int)(k % width);
    atomicAdd(&whole[(int64_t)row_atom[r] * width + c], tile[k]);
}

// The wavefront's row pool (S5-T08): zero rows of a pool array, and map a tile's rows to their atoms' pool rows.
__global__ void tile_zero_rows_kernel(int rows, int width, const int* __restrict__ row_slot, float* __restrict__ pool) {
    const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (int64_t)rows * width) return;
    const int r = (int)(k / width), c = (int)(k % width);
    pool[(int64_t)row_slot[r] * width + c] = 0.0f;
}

__global__ void tile_map_rows_kernel(int rows, const int* __restrict__ row_atom, const int* __restrict__ slot_of_atom,
                                     int* __restrict__ row_slot) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) row_slot[r] = slot_of_atom[row_atom[r]];
}
}  // extern "C++"

extern "C++" {
// A kept tile's device positions from the whole system's: row r is atom row_atom[r] moved by its fixed image offset.
__global__ void tile_place_rows_kernel(int rows, const int* __restrict__ row_atom, const double* __restrict__ offset,
                                       const double* __restrict__ whole, double* __restrict__ pos) {
    const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (int64_t)rows * 3) return;
    const int r = (int)(k / 3), c = (int)(k % 3);
    pos[k] = whole[(int64_t)row_atom[r] * 3 + c] + offset[k];
}

// The largest squared displacement of `pos` from `ref` [n, 3], as the bits of a non-negative double (which order
// as unsigned integers).
__global__ void max_disp2_kernel(int n, const double* __restrict__ pos, const double* __restrict__ ref,
                                 unsigned long long* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double d2 = 0.0;
    if (i < n) {
        const double dx = pos[i * 3] - ref[i * 3], dy = pos[i * 3 + 1] - ref[i * 3 + 1],
                     dz = pos[i * 3 + 2] - ref[i * 3 + 2];
        d2 = dx * dx + dy * dy + dz * dz;
    }
    for (int o = 16; o > 0; o >>= 1) d2 = fmax(d2, __shfl_down_sync(0xffffffffu, d2, o));
    if ((threadIdx.x & 31) == 0) atomicMax(out, (unsigned long long)__double_as_longlong(d2));
}
}  // extern "C++"

namespace {
struct TileDomain {        // one tile as an open domain: locals, then the ghosts it keeps
    int n_local = 0;
    std::vector<int> atom;           // [n_local + n_ghost] the whole system's atom of each row (caller order)
    std::vector<int> z;
    std::vector<double> place;       // [rows, 3] positions in the domain's box
    double box[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int n_ghost() const { return (int)atom.size() - n_local; }
};
}  // namespace

// Tile k of `plan` as an open domain on the host: its locals, then all its ghost candidates, placed in a box with
// room for the list's reach on every side.
static void host_domain(const loup::TilePlan& plan, int64_t k, const int* atomic_numbers, const double* positions,
                        const double* cell, double r_list, TileDomain& all) {
    const int64_t l0 = plan.local_ptr[k], l1 = plan.local_ptr[k + 1];
    const int64_t g0 = plan.ghost_ptr[k], g1 = plan.ghost_ptr[k + 1];
    all.n_local = (int)(l1 - l0);
    auto add = [&](int a, const int32_t* s) {
        all.atom.push_back(a);
        all.z.push_back(atomic_numbers[a]);
        for (int c = 0; c < 3; ++c)
            all.place.push_back(positions[(int64_t)a * 3 + c] + s[0] * cell[c] + s[1] * cell[3 + c] +
                                s[2] * cell[6 + c]);
    };
    for (int64_t i = l0; i < l1; ++i) add(plan.local[i], &plan.local_shift[i * 3]);
    for (int64_t i = g0; i < g1; ++i) add(plan.ghost[i], &plan.ghost_shift[i * 3]);
    double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t r = 0; r < all.atom.size(); ++r)
        for (int c = 0; c < 3; ++c) {
            lo[c] = std::min(lo[c], all.place[r * 3 + c]);
            hi[c] = std::max(hi[c], all.place[r * 3 + c]);
        }
    const double pad = r_list + 1.0;  // the box's own images then lie beyond the list's reach
    for (size_t r = 0; r < all.atom.size(); ++r)
        for (int c = 0; c < 3; ++c) all.place[r * 3 + c] -= lo[c] - pad;
    for (int c = 0; c < 3; ++c) all.box[c * 4] = hi[c] - lo[c] + 2.0 * pad;
}

// Device bytes of the handle's per-system buffers (released when another system or workspace is set up).
static double handle_system_bytes(const LoupHandle* h) {
    return (double)h->sys_atom_bytes + (double)h->sys_edge_bytes + (double)h->sys_fixed_bytes +
           (double)h->cub_tmp_bytes + (h->gen ? (double)h->gen->system_bytes() + (double)h->gen->edge_bytes() : 0.0);
}

// A tile workspace: the handle holds an open domain whose buffers have room for `atom_cap` rows. Another domain
// of at most that many rows takes them over: only the species grouping, the per-atom constants and the row counts
// change. Buffers, the edge capacity, the GEMM plans and the cuBLASLt cache stay (loup_set_system would free and
// allocate them all, and clear the cache).
static int reshape_domain(LoupHandle* h, int n_local, int n_ghost, const int* atomic_numbers) {
    const int n_atoms = n_local + n_ghost;
    if (!h->gen || h->N <= 0 || n_local <= 0 || n_ghost < 0 || n_atoms > h->atom_cap) {
        loup::set_error("internal error: a tile of %d rows does not fit the workspace of %d", n_atoms, h->atom_cap);
        return -1;
    }
    const int S = h->n_species();
    std::vector<int> sp_orig(n_atoms);
    double e0_total = 0.0;
    for (int i = 0; i < n_atoms; ++i) {
        int found = -1;
        for (int s = 0; s < S; ++s)
            if (h->species_z(s) == atomic_numbers[i]) { found = s; break; }
        if (found < 0) {
            loup::set_error("atomic number %d (atom %d) is not in the bundle", atomic_numbers[i], i);
            return -1;
        }
        sp_orig[i] = found;
        if (i < n_local) e0_total += h->species_e0(found);
    }
    h->perm.resize(n_atoms);
    h->h_order.resize(n_atoms);
    h->species_start.assign(S + 1, 0);
    h->species_count.assign(S, 0);
    for (int i = 0; i < n_local; ++i) h->species_count[sp_orig[i]]++;
    for (int s = 0; s < S; ++s) h->species_start[s + 1] = h->species_start[s] + h->species_count[s];
    std::vector<int> cursor = h->species_start, sp_sorted(n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        const int dst = i < n_local ? cursor[sp_orig[i]]++ : i;
        h->perm[dst] = i;
        sp_sorted[dst] = sp_orig[i];
    }
    CUDA_CHECK(cudaMemcpyAsync(h->d_species, sp_sorted.data(), (size_t)n_atoms * sizeof(int), cudaMemcpyHostToDevice,
                               h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));  // `sp_sorted` goes out of scope
    h->e0_total = e0_total;
    h->atom_e0.assign(n_atoms, 0.0);
    for (int i = 0; i < n_local; ++i) h->atom_e0[i] = h->species_e0(sp_orig[i]);
    h->N = n_local;
    h->n_ghost = n_ghost;
    h->nl_valid = false;
    h->forces_current = false;
    h->n_rebuilds = 0;
    h->invalidate_graphs();
    return 0;
}

// A tile's list, kept on the device for the sweeps of one call: everything a sweep reads of the handle's domain.
namespace {
struct TileList {
    int n_local = 0, rows = 0, E = 0;
    double e0 = 0.0, box[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> perm, species_start, species_count;
    DeviceBufferGuard species, pos, row_ptr, senders, receivers, pairs, shift, row_atom;
    DeviceBufferGuard offset;  // [rows, 3] a row's place in the tile less its atom's position at the build
    DeviceBufferGuard row_slot;  // the wavefront (S5-T08): [rows] each row's atom's row in the pool of x and dx
};

// The tiled call's state kept on the handle between calls (S5-T06). Never copied: its buffers free themselves.
struct TiledState {
    int n = 0, receivers = 0, max_rows = 0, max_edges = 0;
    int64_t T = 0;
    double skin = 0.0, cell[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double bytes = 0.0;  // device bytes of everything below
    std::vector<int> z;
    std::vector<TileList> lists;
    DeviceBufferGuard whole_pos, ref_pos, disp2;  // [n, 3] this call's and the build's positions; one 64-bit word
    std::vector<DeviceBufferGuard> x, dx;         // [(L - 1) * 4] layer outputs and their gradients, whole system
    DeviceBufferGuard forces, adj[4];
    // The wavefront (S5-T08, two-layer models): the sweeps in the order of loup::wavefront_schedule, and x and dx
    // in a pool of `pool_rows` rows instead of n (each tile's row_slot gives its rows' places).
    bool wave = false;
    int64_t pool_rows = 0;
    std::vector<loup::SweepStep> steps;
};
}  // namespace

extern "C++" {
static void release_tiled_state(LoupHandle* h) {
    delete h->tiled_state;
    h->tiled_state = nullptr;
}
}  // extern "C++"

// Sets the workspace up as `t`'s domain, builds its list and keeps a copy of it in `out`.
static int build_tile_list(LoupHandle* h, const TileDomain& t, const double* positions, TileList& out) {
    if (reshape_domain(h, t.n_local, t.n_ghost(), t.z.data()) != 0) return -1;
    if (upload_positions(h, t.place.data(), t.box) < 0) return -1;
    if (rebuild_neighbors(h) != 0) return -1;
    const int rows = (int)t.atom.size(), N = t.n_local, E = h->E_skin;
    out.n_local = N; out.rows = rows; out.E = E; out.e0 = h->e0_total;
    std::copy(t.box, t.box + 9, out.box);
    out.perm = h->perm; out.species_start = h->species_start; out.species_count = h->species_count;
    auto keep = [&](DeviceBufferGuard& g, const void* src, size_t bytes) -> int {
        CUDA_CHECK(cudaMalloc(&g.p, std::max<size_t>(bytes, 1)));
        if (bytes) CUDA_CHECK(cudaMemcpyAsync(g.p, src, bytes, cudaMemcpyDeviceToDevice, h->stream));
        return 0;
    };
    if (keep(out.species, h->d_species, (size_t)rows * sizeof(int)) ||
        keep(out.pos, h->d_pos64, (size_t)rows * 3 * sizeof(double)) ||
        keep(out.row_ptr, h->d_row_ptr, (size_t)(N + 1) * sizeof(int)) ||
        keep(out.senders, h->d_senders, (size_t)E * sizeof(int)) ||
        keep(out.receivers, h->d_receivers, (size_t)E * sizeof(int)) ||
        keep(out.pairs, h->d_pairs, (size_t)E * sizeof(int)) ||
        keep(out.shift, h->d_shift_cart, (size_t)E * 3 * sizeof(double)))
        return -1;
    std::vector<int> map(rows);
    std::vector<double> offset((size_t)rows * 3);
    for (int dst = 0; dst < rows; ++dst) {
        const int src = h->perm[dst];
        map[dst] = t.atom[src];
        for (int c = 0; c < 3; ++c)
            offset[(size_t)dst * 3 + c] = t.place[(size_t)src * 3 + c] - positions[(int64_t)map[dst] * 3 + c];
    }
    CUDA_CHECK(cudaMalloc(&out.row_atom.p, (size_t)rows * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(out.row_atom.p, map.data(), (size_t)rows * sizeof(int), cudaMemcpyHostToDevice,
                               h->stream));
    CUDA_CHECK(cudaMalloc(&out.offset.p, offset.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(out.offset.p, offset.data(), offset.size() * sizeof(double), cudaMemcpyHostToDevice,
                               h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));  // `map` and `offset` go out of scope
    return 0;
}

// Puts a kept tile list back into the workspace: device-to-device copies on the stream, no host sync.
static int restore_tile_list(LoupHandle* h, const TileList& t) {
    h->N = t.n_local;
    h->n_ghost = t.rows - t.n_local;
    h->perm = t.perm;
    h->species_start = t.species_start;
    h->species_count = t.species_count;
    h->e0_total = t.e0;
    loup::compute_cell_mat(t.box, h->cell, h->cell_vol, h->cell_heights);
    memcpy(h->cell_in, t.box, sizeof(h->cell_in));
    auto put = [&](void* dst, const DeviceBufferGuard& g, size_t bytes) -> int {
        if (bytes) CUDA_CHECK(cudaMemcpyAsync(dst, g.p, bytes, cudaMemcpyDeviceToDevice, h->stream));
        return 0;
    };
    if (put(h->d_species, t.species, (size_t)t.rows * sizeof(int)) ||
        put(h->d_pos64, t.pos, (size_t)t.rows * 3 * sizeof(double)) ||
        put(h->d_row_ptr, t.row_ptr, (size_t)(t.n_local + 1) * sizeof(int)) ||
        put(h->d_senders, t.senders, (size_t)t.E * sizeof(int)) ||
        put(h->d_receivers, t.receivers, (size_t)t.E * sizeof(int)) ||
        put(h->d_pairs, t.pairs, (size_t)t.E * sizeof(int)) ||
        put(h->d_shift_cart, t.shift, (size_t)t.E * 3 * sizeof(double)))
        return -1;
    h->E_skin = t.E;
    h->nl_valid = true;
    h->forces_current = false;
    CUDA_CHECK(cudaMemsetAsync(h->d_forces_f64, 0, (size_t)t.rows * 3 * sizeof(double), h->stream));
    return 0;
}

static bool can_tile(const LoupHandle* h) { return h->gen && h->gen->sweeps_available() && h->gen->n_layers() >= 2; }

// loup_plan_execution (see the header).
// The wavefront (S5-T08) for a tiled call of n atoms: LOUP_TILE_WAVEFRONT=1|0 forces it on or off, else the
// wavefront mode asks for it, else it is on from 2^20 atoms (with 12 layers of tiles along the first axis, which only
// the plan can tell: wave_pool_rows). Two-layer models only.
static bool wave_try(const LoupHandle* h, int64_t n_atoms, bool* forced) {
    const char* env = std::getenv("LOUP_TILE_WAVEFRONT");
    const int ask = (env && env[0]) ? std::atoi(env) : (h->exec_mode == LOUP_EXEC_WAVEFRONT ? 1 : -1);
    *forced = ask == 1;
    return h->gen->n_layers() == 2 && (ask == 1 || (ask < 0 && n_atoms >= (1 << 20)));
}

// The row pool of the wavefront on the plan of `receivers` per tile, or -1 when that plan gets the plain sweeps.
static int64_t wave_pool_rows(const double* positions, int n_atoms, const double* cell, int receivers, double reach,
                              bool forced) {
    const loup::TilePlan plan = loup::plan_tiles(positions, n_atoms, cell, receivers, 0, nullptr, reach, nullptr, reach);
    const int64_t T = plan.tiles();
    if (!(forced || plan.grid[0] >= 12) || T < 2) return -1;
    std::vector<int64_t> needs_ptr, needs, ready, counts((size_t)T);
    for (int64_t k = 0; k < T; ++k) counts[(size_t)k] = plan.local_ptr[(size_t)k + 1] - plan.local_ptr[(size_t)k];
    loup::tile_dependencies(plan, needs_ptr, needs, ready);
    int64_t peak = 0;
    loup::wavefront_schedule(T, needs_ptr.data(), needs.data(), ready.data(), counts.data(), 1, &peak);
    return peak;
}

static int plan_execution_impl(LoupHandle* h, int n_atoms, const int* atomic_numbers, const double* positions,
                               const double* cell, double skin, LoupMemoryPlan* out) {
    if (!h) return -1;
    if (!out) { loup::set_error("out is NULL"); return -1; }
    if (n_atoms <= 0 || !atomic_numbers || !positions || !cell) {
        loup::set_error("n_atoms must be positive and atomic_numbers, positions and cell given");
        return -1;
    }
    if (!(skin >= 0.0) || !std::isfinite(skin)) { loup::set_error("skin must be finite and not negative"); return -1; }
    if ((h->exec_mode == LOUP_EXEC_TILED || h->exec_mode == LOUP_EXEC_WAVEFRONT) && !can_tile(h)) {
        loup::set_error("the tiled mode needs a bundle of the generic path without long range and with two or more "
                        "layers, and LOUP_CHAIN_FOLD=0 where the skip-chain fold applies");
        return -1;
    }
    if (h->exec_mode == LOUP_EXEC_WAVEFRONT && h->gen->n_layers() != 2) {
        loup::set_error("the wavefront mode needs a model of two layers (this one has %d); use the tiled mode",
                        h->gen->n_layers());
        return -1;
    }
    CUDA_CHECK(cudaSetDevice(h->device_id));
    LoupMemoryPlan plan{};
    size_t free_b = 0, total_b = 0;
    if (h->exec_mode == LOUP_EXEC_WHOLE || !can_tile(h)) {  // the whole-system call: asked, or all the bundle has
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        plan.free_bytes = free_b;
        plan.mode = LOUP_EXEC_WHOLE;
        *out = plan;
        return 0;
    }
    // The probe: the tile with the most rows of a plan at 4,096 receivers, set up and listed on the handle.
    constexpr int kMinReceivers = 4096;
    const double r_list = (double)h->gen->r_max() + skin;
    const int c0 = std::min(kMinReceivers, n_atoms);
    loup::TilePlan tp;
    try {
        tp = loup::plan_tiles(positions, n_atoms, cell, c0, 0, nullptr, r_list + 1e-3, nullptr);
    } catch (const std::exception& e) {
        loup::set_error("tile plan: %s", e.what());
        return -1;
    }
    int64_t widest = 0, widest_rows = -1;
    for (int64_t k = 0; k < tp.tiles(); ++k) {
        const int64_t rows = tp.local_ptr[k + 1] - tp.local_ptr[k] + tp.ghost_ptr[k + 1] - tp.ghost_ptr[k];
        if (rows > widest_rows) { widest_rows = rows; widest = k; }
    }
    TileDomain d;
    host_domain(tp, widest, atomic_numbers, positions, cell, r_list, d);
    if (set_domain_impl(h, d.n_local, d.n_ghost(), d.z.data(), skin) != 0) return -1;
    h->tile_workspace = true;
    if (upload_positions(h, d.place.data(), d.box) < 0) return -1;
    if (rebuild_neighbors(h) != 0) return -1;
    // Bytes per row (locals and ghosts alike), per edge of capacity, and those that do not scale; kPlanAllowance as
    // in loup_plan_memory.
    constexpr double kPlanAllowance = 32.0 * 1024.0 * 1024.0;
    const double row_b = ((double)h->sys_atom_bytes + (double)h->gen->system_bytes() + (double)h->cub_tmp_bytes) /
                         h->atom_cap;
    const double edge_b = ((double)h->sys_edge_bytes + (double)h->gen->edge_bytes()) / h->E_cap;
    const double fixed_b = (double)h->sys_fixed_bytes + (double)h->weight_copy_bytes + kPlanAllowance;
    const double epl = (double)h->E_skin / d.n_local;  // edges per receiver
    auto edge_cap = [](double e) { return std::max(e + std::floor(e / 4.0) + 1024.0, 4096.0); };
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const double budget = (double)free_b + handle_system_bytes(h);  // the probe's buffers go when the call sets up
    const double allowed = h->exec_fraction * budget;
    const double n = n_atoms;
    const double whole = row_b * n + edge_b * edge_cap(n * epl) + fixed_b;
    // The tiled call: the whole system's layer outputs, their gradients and the forces; each tile's kept list
    // (rows: species, positions, row map; edges: senders, receivers, pairs, shifts); the workspace for the widest
    // domain, whose ghost rows grow with a tile's surface; the adjoint buffers.
    // With the wavefront (S5-T08) the layer outputs and their gradients take the schedule's row pool instead of the
    // whole system, and each kept row a pool row more.
    const int L = h->gen->n_layers();
    double layer_per_atom = 0.0, adj_per_row = 0.0;
    for (int l = 0; l < 4; ++l) {
        double widest_l = 0.0;
        for (int t = 0; t + 1 < L; ++t)
            if (l <= h->gen->out_lmax(t)) {
                const double w = (double)(2 * l + 1) * h->gen->out_width(t);
                layer_per_atom += 2.0 * w * sizeof(float);
                widest_l = std::max(widest_l, w);
            }
        adj_per_row += widest_l * sizeof(float);
    }
    bool wave_forced = false;
    const bool wave = wave_try(h, n_atoms, &wave_forced);
    const double ghost0 = (double)widest_rows - c0;
    auto tiled_at = [&](double c, double& global, double& ws) -> int {
        c = std::min(c, n);
        const double rows = c + ghost0 * std::pow(c / c0, 2.0 / 3.0), T = std::ceil(n / c);
        ws = row_b * rows + edge_b * edge_cap(c * epl) + fixed_b + adj_per_row * rows;
        double kept = 32.0 * (n + T * (rows - c)) + 4.0 * (n + T) + 36.0 * n * epl;
        double layer_rows = n;
        if (wave) {
            int64_t pool = -1;
            try {
                pool = wave_pool_rows(positions, n_atoms, cell, (int)c, r_list + 1e-3, wave_forced);
            } catch (const std::exception& e) {
                loup::set_error("tile wavefront: %s", e.what());
                return -1;
            }
            if (pool >= 0) {
                layer_rows = (double)pool;
                kept += 4.0 * (n + T * (rows - c));
            }
        }
        global = n * 3.0 * sizeof(double) + layer_rows * layer_per_atom + kept;
        return 0;
    };
    static const int kCaps[] = {65536, 49152, 32768, 24576, 16384, 12288, 8192, 6144, 4096};
    int cap = 0;
    double global = 0.0, ws = 0.0;
    if (h->exec_receivers > 0) {
        cap = h->exec_receivers;
        if (tiled_at(cap, global, ws) != 0) return -1;
    } else {
        for (int c : kCaps) {
            if (tiled_at(c, global, ws) != 0) return -1;
            if (global + ws <= allowed && std::min((double)c, n) * epl * 1.1 <= (double)h->edge_limit) {
                cap = c;  // the tile's own list stays within the edge limit, with 10% for its densest region
                break;
            }
        }
    }
    plan.whole_bytes = (size_t)whole;
    plan.tiled_global_bytes = (size_t)global;
    plan.tile_workspace_bytes = (size_t)ws;
    plan.free_bytes = free_b;
    plan.receivers = cap;
    const bool whole_fits = whole <= allowed && n * epl <= (double)h->edge_limit;
    plan.mode = (h->exec_mode == LOUP_EXEC_AUTO && whole_fits) ? LOUP_EXEC_WHOLE : LOUP_EXEC_TILED;
    if (plan.mode == LOUP_EXEC_TILED && cap == 0) {
        loup::set_error("%d atoms do not fit the device even in tiles of %d receivers: %.2f GiB of state and "
                        "%.2f GiB of workspace against %.2f GiB allowed (%.0f%% of %.2f GiB); the tiled state in "
                        "host memory is plan task S5-T07", n_atoms, kMinReceivers, global / 1073741824.0,
                        ws / 1073741824.0, allowed / 1073741824.0, 100.0 * h->exec_fraction, budget / 1073741824.0);
        return -1;
    }
    *out = plan;
    return 0;
}

static int evaluate_tiled_impl(LoupHandle* h, int n_atoms, const int* atomic_numbers, const double* positions,
                               const double* cell, int receivers, double skin, double* out_energy_ev,
                               double* out_forces, double* out_stress, long long* out_tiles) {
    if (!h) return -1;
    if (h->disp) {
        loup::set_error("the tiled call has no dispersion term yet (PLAN.md S6-D09)");
        return -1;
    }
    if (n_atoms <= 0 || !atomic_numbers || !positions || !cell) {
        loup::set_error("n_atoms must be positive and atomic_numbers, positions and cell given");
        return -1;
    }
    if (!(h->gen && h->gen->sweeps_available())) {
        loup::set_error("the tiled call needs a bundle of the generic path without long range, and "
                        "LOUP_CHAIN_FOLD=0 where the skip-chain fold applies");
        return -1;
    }
    const int L = h->gen->n_layers();
    if (L < 2) { loup::set_error("the tiled call needs a model of two or more layers, this one has %d", L); return -1; }
    if (h->exec_mode == LOUP_EXEC_WAVEFRONT && L != 2) {
        loup::set_error("the wavefront mode needs a model of two layers (this one has %d); use the tiled mode", L);
        return -1;
    }
    if (!(skin >= 0.0) || !std::isfinite(skin)) { loup::set_error("skin must be finite and not negative"); return -1; }
    CUDA_CHECK(cudaSetDevice(h->device_id));
    auto blocks = [](int64_t n) { return (unsigned)((n + 255) / 256); };
    auto width = [&](int t, int l) { return (2 * l + 1) * h->gen->out_width(t); };
    // The tiled state of an earlier call stays while the system, cell, skin and cap are the same and no atom has
    // moved skin/2 since its lists were built (S5-T06): only the tiles' positions are refreshed then.
    TiledState* st = h->tiled_state;
    const bool same_system = st && h->tile_workspace && st->n == n_atoms && st->skin == skin &&
                             memcmp(st->cell, cell, sizeof(st->cell)) == 0 &&
                             std::equal(st->z.begin(), st->z.end(), atomic_numbers);
    if (same_system && receivers <= 0) receivers = st->receivers;  // a rebuild of the same system keeps its cap
    bool keep = same_system && receivers == st->receivers;
    if (keep) {
        CUDA_CHECK(cudaMemcpyAsync(st->whole_pos.p, positions, (size_t)n_atoms * 3 * sizeof(double),
                                   cudaMemcpyHostToDevice, h->stream));
        CUDA_CHECK(cudaMemsetAsync(st->disp2.p, 0, sizeof(unsigned long long), h->stream));
        max_disp2_kernel<<<blocks(n_atoms), 256, 0, h->stream>>>(n_atoms, static_cast<const double*>(st->whole_pos.p),
                                                                 static_cast<const double*>(st->ref_pos.p),
                                                                 static_cast<unsigned long long*>(st->disp2.p));
        unsigned long long bits = 0;
        CUDA_CHECK(cudaMemcpyAsync(&bits, st->disp2.p, sizeof(bits), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        double d2 = 0.0;
        memcpy(&d2, &bits, sizeof(d2));
        keep = std::isfinite(d2) && d2 <= 0.25 * skin * skin;
    }
    if (!keep) {
        release_tiled_state(h);
        st = nullptr;
    }
    if (!keep && receivers <= 0) {  // the memory plan's cap, the mode taken as tiled
        struct Mode {
            LoupHandle* h;
            int saved;
            ~Mode() { h->exec_mode = saved; }
        } mode{h, h->exec_mode};
        if (h->exec_mode != LOUP_EXEC_WAVEFRONT) h->exec_mode = LOUP_EXEC_TILED;
        LoupMemoryPlan mp{};
        if (plan_execution_impl(h, n_atoms, atomic_numbers, positions, cell, skin, &mp) != 0) return -1;
        receivers = mp.receivers;
    }
    size_t free_b = 0, total_b = 0, least_free = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const double budget = (double)free_b + handle_system_bytes(h) + (st ? st->bytes : 0.0);
    least_free = free_b;
    auto note_free = [&]() -> int {
        size_t f = 0, t = 0;
        CUDA_CHECK(cudaMemGetInfo(&f, &t));
        least_free = std::min(least_free, f);
        return 0;
    };
    const double r_list = (double)h->gen->r_max() + skin;
    const bool tile_times = std::getenv("LOUP_DEBUG_TILE_TIMES") && std::getenv("LOUP_DEBUG_TILE_TIMES")[0] == '1';
    auto tick_last = std::chrono::steady_clock::now();
    auto tick = [&](const char* what) {
        if (!tile_times) return;
        cudaStreamSynchronize(h->stream);
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "tile times: %-28s %9.2f ms\n", what,
                     std::chrono::duration<double, std::milli>(now - tick_last).count());
        tick_last = now;
    };
    if (!keep) {
        // The wavefront (S5-T08): two-layer models only (three sweeps). LOUP_TILE_WAVEFRONT=1 forces it, 0 turns it
        // off; otherwise on from 2^20 atoms and 12 layers of tiles along the first axis (the plan's rule). Its plan
        // splits the first and last layer at the faces (boundary skins), so only they stay in the pool all call.
        bool wave_forced = false;
        const bool wave_on = wave_try(h, n_atoms, &wave_forced);
        loup::TilePlan plan;
        try {
            plan = loup::plan_tiles(positions, n_atoms, cell, receivers, 0, nullptr, r_list + 1e-3, nullptr,
                                    wave_on ? r_list + 1e-3 : 0.0);
        } catch (const std::exception& e) {
            loup::set_error("tile plan: %s", e.what());
            return -1;
        }
        // the rule of wave_pool_rows, which the memory plan used
        const bool wave = wave_on && (wave_forced || plan.grid[0] >= 12) && plan.tiles() >= 2;
        const int64_t T = plan.tiles();
        // The tiles' domains; the candidates that send along no edge are dropped (S5-T02).
        std::vector<TileDomain> alls((size_t)T), tiles((size_t)T);
        int max_rows = 0;
        size_t widest = 0;
        for (int64_t k = 0; k < T; ++k) {
            host_domain(plan, k, atomic_numbers, positions, cell, r_list, alls[(size_t)k]);
            if (alls[(size_t)k].atom.size() > alls[widest].atom.size()) widest = (size_t)k;
        }
        tick("plan and host domains");
        // One workspace for every tile: the buffers of the widest domain (all its candidates), set up once. A
        // workspace of an earlier call stays when it is large enough and has the same skin: loup_set_system's path
        // would also empty the cuBLASLt cache, and every tile's GEMM shapes would be looked up again.
        const bool reuse = h->tile_workspace && h->skin == skin && (int)alls[widest].atom.size() <= h->atom_cap;
        if (!reuse) {
            if (set_domain_impl(h, alls[widest].n_local, alls[widest].n_ghost(), alls[widest].z.data(), skin) != 0)
                return -1;
            h->tile_workspace = true;
        }
        tick("workspace set-up");
        for (int64_t k = 0; k < T; ++k) {
            TileDomain& all = alls[(size_t)k];
            TileDomain& t = tiles[(size_t)k];
            const int G = all.n_ghost();
            if (G == 0) { t = std::move(all); max_rows = std::max(max_rows, t.n_local); continue; }
            if (reshape_domain(h, all.n_local, G, all.z.data()) != 0) return -1;
            if (upload_positions(h, all.place.data(), all.box) < 0) return -1;
            if (rebuild_neighbors(h) != 0) return -1;
            std::vector<unsigned char> used((size_t)G, 0);
            DeviceBufferGuard dm;
            CUDA_CHECK(cudaMalloc(&dm.p, G));
            CUDA_CHECK(cudaMemsetAsync(dm.p, 0, G, h->stream));
            if (h->E_skin > 0)
                mark_ghost_senders_kernel<<<(h->E_skin + 255) / 256, 256, 0, h->stream>>>(
                    h->E_skin, h->N, h->d_senders, static_cast<unsigned char*>(dm.p));
            CUDA_CHECK(cudaMemcpyAsync(used.data(), dm.p, G, cudaMemcpyDeviceToHost, h->stream));
            CUDA_CHECK(cudaStreamSynchronize(h->stream));
            t.n_local = all.n_local;
            std::copy(all.box, all.box + 9, t.box);
            for (int r = 0; r < (int)all.atom.size(); ++r) {
                if (r >= all.n_local && !used[(size_t)(r - all.n_local)]) continue;
                t.atom.push_back(all.atom[r]);
                t.z.push_back(all.z[r]);
                t.place.insert(t.place.end(), &all.place[(size_t)r * 3], &all.place[(size_t)r * 3 + 3]);
            }
            max_rows = std::max(max_rows, (int)t.atom.size());
            TileDomain().atom.swap(all.atom);  // the candidates' host copy is no longer needed
            all.place.clear(); all.place.shrink_to_fit();
        }
        tick("candidate builds");
        // Every tile's list, built once and kept on the device with the rest of the state
        st = h->tiled_state = new TiledState();
        st->lists = std::vector<TileList>((size_t)T);
        int max_edges = 0;
        for (int64_t k = 0; k < T; ++k) {
            TileList& tl = st->lists[(size_t)k];
            if (build_tile_list(h, tiles[(size_t)k], positions, tl) != 0) { release_tiled_state(h); return -1; }
            max_edges = std::max(max_edges, tl.E);
            st->bytes += (double)tl.rows * (4 + 24 + 4 + 24) + (double)(tl.n_local + 1) * 4 + (double)tl.E * 36;
        }
        if (max_edges > h->E_cap) {
            release_tiled_state(h);
            loup::set_error("internal error: a kept tile list outgrew the edge capacity");
            return -1;
        }
        tick("kept list builds");
        st->n = n_atoms; st->receivers = receivers; st->T = T; st->skin = skin;
        st->max_rows = max_rows; st->max_edges = max_edges;
        if (wave) {
            // The schedule from the plan's locals and candidates (a superset of the kept rows), then a pool row for
            // every atom: the host replays the schedule, taking rows at a tile's sweep 1 and giving them back at its
            // sweep 3. Every atom is a local of one tile, so it has one pool row for the whole call.
            std::vector<int64_t> needs_ptr, needs, ready, counts((size_t)T);
            for (int64_t k = 0; k < T; ++k) counts[(size_t)k] = plan.local_ptr[(size_t)k + 1] - plan.local_ptr[(size_t)k];
            int64_t peak = 0;
            try {
                loup::tile_dependencies(plan, needs_ptr, needs, ready);
                st->steps = loup::wavefront_schedule(T, needs_ptr.data(), needs.data(), ready.data(), counts.data(), 1,
                                                     &peak);
            } catch (const std::exception& e) {
                release_tiled_state(h);
                loup::set_error("tile wavefront: %s", e.what());
                return -1;
            }
            std::vector<int> slot_of_atom((size_t)n_atoms, -1), free_rows;
            std::vector<std::vector<int>> taken((size_t)T);
            int next = 0;
            for (const loup::SweepStep& s : st->steps) {
                const int64_t a = plan.local_ptr[(size_t)s.tile], b = plan.local_ptr[(size_t)s.tile + 1];
                if (s.sweep == 1) {
                    for (int64_t i = a; i < b; ++i) {
                        int row;
                        if (!free_rows.empty()) { row = free_rows.back(); free_rows.pop_back(); }
                        else row = next++;
                        slot_of_atom[(size_t)plan.local[(size_t)i]] = row;
                        taken[(size_t)s.tile].push_back(row);
                    }
                } else if (s.sweep == 3) {
                    free_rows.insert(free_rows.end(), taken[(size_t)s.tile].begin(), taken[(size_t)s.tile].end());
                    std::vector<int>().swap(taken[(size_t)s.tile]);
                }
            }
            if (next != peak) {
                release_tiled_state(h);
                loup::set_error("internal error: the row pool took %d rows, the schedule's peak is %lld", next,
                                (long long)peak);
                return -1;
            }
            DeviceBufferGuard d_slot;
            CUDA_CHECK(cudaMalloc(&d_slot.p, (size_t)n_atoms * sizeof(int)));
            CUDA_CHECK(cudaMemcpyAsync(d_slot.p, slot_of_atom.data(), (size_t)n_atoms * sizeof(int),
                                       cudaMemcpyHostToDevice, h->stream));
            for (TileList& tl : st->lists) {
                CUDA_CHECK(cudaMalloc(&tl.row_slot.p, (size_t)tl.rows * sizeof(int)));
                tile_map_rows_kernel<<<(tl.rows + 255) / 256, 256, 0, h->stream>>>(
                    tl.rows, static_cast<const int*>(tl.row_atom.p), static_cast<const int*>(d_slot.p),
                    static_cast<int*>(tl.row_slot.p));
                st->bytes += (double)tl.rows * 4;
            }
            CUDA_CHECK(cudaStreamSynchronize(h->stream));  // `slot_of_atom` and `d_slot` go out of scope
            st->wave = true;
            st->pool_rows = peak;
            tick("wavefront schedule");
            if (tile_times)
                std::fprintf(stderr, "tile wavefront: %lld tiles, grid %d x %d x %d, row pool %lld of %lld atoms\n",
                             (long long)T, plan.grid[0], plan.grid[1], plan.grid[2], (long long)peak,
                             (long long)n_atoms);
        }
        const int64_t layer_rows = st->wave ? st->pool_rows : n_atoms;
        std::copy(cell, cell + 9, st->cell);
        st->z.assign(atomic_numbers, atomic_numbers + n_atoms);
        auto alloc = [&](DeviceBufferGuard& g, size_t bytes) -> int {
            CUDA_CHECK(cudaMalloc(&g.p, std::max<size_t>(bytes, 1)));
            st->bytes += (double)bytes;
            return 0;
        };
        const size_t pos_bytes = (size_t)n_atoms * 3 * sizeof(double);
        // The whole system's positions, layer outputs, their gradients and the forces, on the device
        st->x = std::vector<DeviceBufferGuard>((size_t)(L - 1) * 4);
        st->dx = std::vector<DeviceBufferGuard>((size_t)(L - 1) * 4);
        int fail = alloc(st->whole_pos, pos_bytes) | alloc(st->ref_pos, pos_bytes) |
                   alloc(st->disp2, sizeof(unsigned long long)) | alloc(st->forces, pos_bytes);
        for (int t = 0; t + 1 < L; ++t)
            for (int l = 0; l <= h->gen->out_lmax(t); ++l) {
                const size_t bytes = (size_t)layer_rows * width(t, l) * sizeof(float);
                fail |= alloc(st->x[(size_t)t * 4 + l], bytes) | alloc(st->dx[(size_t)t * 4 + l], bytes);
            }
        for (int l = 0; l < 4; ++l) {
            size_t floats = 0;
            for (int t = 0; t + 1 < L; ++t)
                if (l <= h->gen->out_lmax(t)) floats = std::max(floats, (size_t)max_rows * width(t, l));
            if (floats) fail |= alloc(st->adj[l], floats * sizeof(float));
        }
        if (fail) { release_tiled_state(h); return -1; }
        CUDA_CHECK(cudaMemcpyAsync(st->whole_pos.p, positions, pos_bytes, cudaMemcpyHostToDevice, h->stream));
        CUDA_CHECK(cudaMemcpyAsync(st->ref_pos.p, st->whole_pos.p, pos_bytes, cudaMemcpyDeviceToDevice, h->stream));
        ++h->tile_list_builds;
    } else {
        for (TileList& t : st->lists)
            tile_place_rows_kernel<<<blocks((int64_t)t.rows * 3), 256, 0, h->stream>>>(
                t.rows, static_cast<const int*>(t.row_atom.p), static_cast<const double*>(t.offset.p),
                static_cast<const double*>(st->whole_pos.p), static_cast<double*>(t.pos.p));
        tick("kept lists refreshed");
    }
    h->E_graph = st->max_edges;  // the edge kernels' grid: every tile's edges
    h->invalidate_graphs();
    const int64_t T = st->T;
    if (out_tiles) *out_tiles = T;
    std::vector<TileList>& lists = st->lists;
    std::vector<DeviceBufferGuard>& x = st->x;
    std::vector<DeviceBufferGuard>& dx = st->dx;
    DeviceBufferGuard& d_forces = st->forces;
    DeviceBufferGuard* adj = st->adj;
    const bool wave = st->wave;  // the pool's rows are zeroed tile by tile at each tile's sweep 1
    for (int t = 0; !wave && t + 1 < L; ++t)
        for (int l = 0; l <= h->gen->out_lmax(t); ++l)
            CUDA_CHECK(cudaMemsetAsync(dx[(size_t)t * 4 + l].p, 0, (size_t)n_atoms * width(t, l) * sizeof(float),
                                       h->stream));
    CUDA_CHECK(cudaMemsetAsync(d_forces.p, 0, (size_t)n_atoms * 3 * sizeof(double), h->stream));
    auto fx = [&](int t, int l) { return static_cast<float*>(x[(size_t)t * 4 + l].p); };
    auto fdx = [&](int t, int l) { return static_cast<float*>(dx[(size_t)t * 4 + l].p); };
    auto grid = [](int64_t n) { return (unsigned)((n + 255) / 256); };
    struct NoFork {
        LoupHandle* h;
        bool saved;
        explicit NoFork(LoupHandle* h_) : h(h_), saved(h_->fork_ok) { h->fork_ok = false; }
        ~NoFork() { h->fork_ok = saved; }
    } no_fork(h);
    const int stress = out_stress ? 1 : 0;
    double energy = 0.0, virial[6] = {0, 0, 0, 0, 0, 0};
    // Energy and virial accumulate on the device over every tile sweep; each tile's forces are folded after it.
    CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(h->d_zero_slab) + h->zero_tail_off, 0,
                               h->zero_slab_bytes - h->zero_tail_off, h->stream));
    auto row_atom = [](const TileList& t) { return static_cast<const int*>(t.row_atom.p); };
    // The rows of x and dx: the atoms' own rows, or with the wavefront their rows in the pool
    auto row_x = [&](const TileList& t) { return static_cast<const int*>(wave ? t.row_slot.p : t.row_atom.p); };
    auto fold_forces = [&](const TileList& t) {
        tile_scatter_add_rows_kernel<double><<<grid((int64_t)t.rows * 3), 256, 0, h->stream>>>(
            t.rows, 3, row_atom(t), h->d_forces_f64, static_cast<double*>(d_forces.p));
    };
    auto gather_input = [&](const TileList& t, int layer) {  // layer `layer`'s output rows of locals and ghosts
        for (int l = 0; layer >= 0 && l <= h->gen->out_lmax(layer); ++l)
            tile_gather_rows_kernel<<<grid((int64_t)t.rows * width(layer, l)), 256, 0, h->stream>>>(
                t.rows, width(layer, l), row_x(t), fx(layer, l), h->gen->layer_out(layer, l));
    };
    auto scatter_grad = [&](const TileList& t, int layer) {  // the gradient of layer `layer`'s output, every row
        for (int l = 0; layer >= 0 && l <= h->gen->out_lmax(layer); ++l)
            tile_scatter_add_rows_kernel<float><<<grid((int64_t)t.rows * width(layer, l)), 256, 0, h->stream>>>(
                t.rows, width(layer, l), row_x(t), h->gen->layer_out_grad(layer, l), fdx(layer, l));
    };
    auto forward = [&](const TileList& t, int layer) -> int {
        if (restore_tile_list(h, t) != 0) return -1;
        if (layer == 0) energy += t.e0;
        gather_input(t, layer - 1);
        DomainSweep sw;
        sw.kind = loup::GenericPassInputs::SweepForward;
        sw.layer = layer;
        record_force_pass(h, stress, false, &sw);
        for (int l = 0; l <= h->gen->out_lmax(layer); ++l) {
            tile_scatter_rows_kernel<<<grid((int64_t)t.n_local * width(layer, l)), 256, 0, h->stream>>>(
                t.n_local, width(layer, l), row_x(t), h->gen->layer_out(layer, l), fx(layer, l));
            if (wave)  // the pool rows its locals take may hold an earlier tile's gradients
                tile_zero_rows_kernel<<<grid((int64_t)t.n_local * width(layer, l)), 256, 0, h->stream>>>(
                    t.n_local, width(layer, l), row_x(t), fdx(layer, l));
        }
        return 0;
    };
    auto last = [&](const TileList& t) -> int {
        if (restore_tile_list(h, t) != 0) return -1;
        gather_input(t, L - 2);
        DomainSweep sw;
        sw.kind = loup::GenericPassInputs::SweepLast;
        record_force_pass(h, stress, false, &sw);
        scatter_grad(t, L - 2);
        fold_forces(t);
        return 0;
    };
    auto backward = [&](const TileList& t, int layer) -> int {
        if (restore_tile_list(h, t) != 0) return -1;
        gather_input(t, layer - 1);
        DomainSweep sw;
        sw.kind = loup::GenericPassInputs::SweepBackward;
        sw.layer = layer;
        for (int l = 0; l <= h->gen->out_lmax(layer); ++l) {
            float* a = static_cast<float*>(adj[l].p);
            tile_gather_rows_kernel<<<grid((int64_t)t.n_local * width(layer, l)), 256, 0, h->stream>>>(
                t.n_local, width(layer, l), row_x(t), fdx(layer, l), a);
            sw.adjoint[l] = a;
        }
        record_force_pass(h, stress, false, &sw);
        scatter_grad(t, layer - 1);
        fold_forces(t);
        return 0;
    };

    if (note_free() != 0) return -1;
    tick("state buffers");
    if (wave) {  // two-layer models: sweep 1 is layer 0 forward, 2 the last layer, 3 layer 0 backward
        for (const loup::SweepStep& s : st->steps) {
            const TileList& t = lists[(size_t)s.tile];
            if ((s.sweep == 1 ? forward(t, 0) : s.sweep == 2 ? last(t) : backward(t, 0)) != 0) return -1;
        }
        tick("wavefront sweeps");
    } else {
        for (int layer = 0; layer + 1 < L; ++layer)  // forward sweeps
            for (const TileList& t : lists)
                if (forward(t, layer) != 0) return -1;
        tick("forward sweeps");
        for (const TileList& t : lists)  // the last layer
            if (last(t) != 0) return -1;
        tick("last-layer sweeps");
        for (int layer = L - 2; layer >= 0; --layer)  // backward sweeps
            for (const TileList& t : lists)
                if (backward(t, layer) != 0) return -1;
        tick("backward sweeps");
    }
    if (note_free() != 0) return -1;
    h->last_exec = LoupExecution{wave ? LOUP_EXEC_WAVEFRONT : LOUP_EXEC_TILED, receivers, (long long)T, (size_t)budget,
                                 (size_t)std::max(0.0, budget - (double)least_free), keep ? 1 : 0,
                                 h->tile_list_builds};
    {
        double e = 0.0;
        CUDA_CHECK(cudaMemcpyAsync(&e, h->d_energy_f64, sizeof(double), cudaMemcpyDeviceToHost, h->stream));
        CUDA_CHECK(cudaMemcpyAsync(virial, h->d_virial_grad_f64, 6 * sizeof(double), cudaMemcpyDeviceToHost,
                                   h->stream));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        CUDA_CHECK(cudaGetLastError());
        energy += e;
    }

    if (out_energy_ev) *out_energy_ev = energy;
    if (out_forces)
        CUDA_CHECK(cudaMemcpyAsync(out_forces, d_forces.p, (size_t)n_atoms * 3 * sizeof(double),
                                   cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    if (out_stress) {
        const double vol = std::fabs(cell[0] * (cell[4] * cell[8] - cell[5] * cell[7]) -
                                     cell[1] * (cell[3] * cell[8] - cell[5] * cell[6]) +
                                     cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]));
        const double inv_v = 1.0 / vol;
        const double* v = virial;
        out_stress[0] = v[0] * inv_v; out_stress[1] = v[1] * inv_v; out_stress[2] = v[2] * inv_v;
        out_stress[3] = v[1] * inv_v; out_stress[4] = v[3] * inv_v; out_stress[5] = v[4] * inv_v;
        out_stress[6] = v[2] * inv_v; out_stress[7] = v[4] * inv_v; out_stress[8] = v[5] * inv_v;
    }
    h->save_disk_plans();
    return 0;
}

// loup_plan_memory: the device memory a system of n_atoms would take, scaled from the buffers the handle's current
// system has. Every per-system buffer is proportional to the atom count or to the edge capacity, so the scaling is
// exact up to alignment; what it leaves out is small and does not scale (captured graphs, cuBLAS internals).
static int plan_memory_impl(LoupHandle* h, int n_atoms, double edges_per_atom, LoupMemoryPlan* out) {
    if (!out) { loup::set_error("out is NULL"); return -1; }
    if (!has_system(h)) return -1;
    if (!h->nl_valid || h->E_cap <= 0) {
        loup::set_error("no neighbour list to scale from (call loup_set_state or loup_evaluate on a system with "
                        "neighbours first; a small cell of the same material is enough)");
        return -1;
    }
    if (n_atoms <= 0) { loup::set_error("n_atoms must be positive, got %d", n_atoms); return -1; }
    const double epa = edges_per_atom > 0.0 ? edges_per_atom : (double)h->E_skin / h->N;
    if (!std::isfinite(epa)) { loup::set_error("edges_per_atom must be finite, got %g", edges_per_atom); return -1; }
    CUDA_CHECK(cudaSetDevice(h->device_id));
    const double scale = (double)n_atoms / h->N;
    const double atom_now = (double)h->sys_atom_bytes + (h->gen ? (double)h->gen->system_bytes() : 0.0);
    const double edge_now = (double)h->sys_edge_bytes + (h->gen ? (double)h->gen->edge_bytes() : 0.0);
    const double edges = std::ceil((double)n_atoms * epa);
    const double cap = std::max(edges + std::floor(edges / 4.0) + 1024.0, 4096.0);  // ensure_edge_capacity's rule
    // kPlanAllowance stands for what is not counted (captured graphs, the allocator's rounding of sixty buffers):
    // on SrTiO3 with OMAT-0 medium the counted bytes were 58, -1 and 25 MB below the measured memory at 5,000, 20,480
    // and 40,000 atoms (RTX 5060 Ti, 15 November 2026).
    constexpr double kPlanAllowance = 32.0 * 1024.0 * 1024.0;
    const double whole = atom_now * scale + (double)h->sys_fixed_bytes + edge_now / h->E_cap * cap +
                         (double)h->cub_tmp_bytes * scale + (double)h->weight_copy_bytes + kPlanAllowance;
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    LoupMemoryPlan plan{};
    plan.whole_bytes = (size_t)whole;
    plan.free_bytes = free_b;
    // The planned system would replace the current one, so its buffers count as free; 10% stay as headroom. Edge
    // indices are 32-bit: beyond that the whole-system call cannot run whatever the memory.
    const double budget = (double)free_b + atom_now + (double)h->sys_fixed_bytes + edge_now + (double)h->cub_tmp_bytes;
    plan.mode = (whole <= 0.9 * budget && edges <= (double)h->edge_limit) ? LOUP_EXEC_WHOLE : LOUP_EXEC_TILED;
    *out = plan;
    return 0;
}

static int profile_force_impl(LoupHandle* h, int warmup, int n_calls, double* out_total_us) {
    if (!has_system(h)) return -1;
    if (!h->nl_valid) {
        loup::set_error("no positions or neighbour list (call loup_set_state or loup_evaluate first)");
        return -1;
    }
    if (warmup < 0 || n_calls <= 0) {
        loup::set_error("warmup must not be negative and n_calls must be positive, got %d and %d", warmup, n_calls);
        return -1;
    }
    CUDA_CHECK(cudaSetDevice(h->device_id));
    if (ensure_force_graph(h, 0) != 0) return -1;
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaGraphLaunch(h->force_graph_exec, h->stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    EventPairGuard evg;
    cudaEvent_t ev0 = evg.e0, ev1 = evg.e1;
    CUDA_CHECK(cudaEventRecord(ev0, h->stream));
    for (int i = 0; i < n_calls; ++i) {
        CUDA_CHECK(cudaGraphLaunch(h->force_graph_exec, h->stream));
    }
    CUDA_CHECK(cudaEventRecord(ev1, h->stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    if (out_total_us) *out_total_us = (double)ms * 1000.0 / n_calls;
    return 0;
}

int loup_set_dispersion(LoupHandle* h, const LoupD3Settings* settings, const LoupD3Tables* tables) {
    return guarded(h, "loup_set_dispersion", [&]() -> int {
        CUDA_CHECK(cudaSetDevice(h->device_id));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
        h->invalidate_graphs();
        h->disp.reset();
        h->disp_ready = false;
        if (!settings) return 0;
        if (!tables) { loup::set_error("tables is NULL"); return -1; }
        if (settings->damping < 0 || settings->damping > 3) {
            loup::set_error("damping must be 0 (BJ), 1 (zero), 2 (modified BJ) or 3 (modified zero), got %d",
                            settings->damping);
            return -1;
        }
        loup::dispersion::Settings s;
        s.damping = static_cast<loup::dispersion::Damping>(settings->damping);
        s.s6 = settings->s6; s.s8 = settings->s8; s.a1 = settings->a1; s.a2 = settings->a2;
        s.rs6 = settings->rs6; s.rs8 = settings->rs8; s.alpha = settings->alpha; s.beta = settings->beta;
        s.pair_cutoff_bohr = settings->pair_cutoff_bohr;
        s.cn_cutoff_bohr = settings->cn_cutoff_bohr;
        s.smoothing_bohr = settings->smoothing_bohr;
        loup::dispersion::Tables t{tables->nref, tables->cnref, tables->rcov, tables->r2r4, tables->c6ref,
                                   tables->r0ab};
        const char* why = nullptr;
        h->disp.reset(loup::dispersion::make_term(s, t, &why));
        if (!h->disp) { loup::set_error("%s", why ? why : "the dispersion term was refused"); return -1; }
        // a system with positions: prepare now (its cell is known); else at its first positions
        if (h->N > 0 && h->nl_valid) return prepare_dispersion(h);
        return 0;
    });
}

int loup_set_system(LoupHandle* h, int n_atoms, const int* atomic_numbers, const double* masses_amu, double skin) {
    return guarded(h, "loup_set_system", [&] { return set_system_impl(h, n_atoms, atomic_numbers, masses_amu, skin); });
}

int loup_set_state(LoupHandle* h, const double* positions, const double* velocities, const double* cell) {
    return guarded(h, "loup_set_state", [&] { return set_state_impl(h, positions, velocities, cell); });
}

int loup_set_fixed_atoms(LoupHandle* h, const int* fixed) {
    return guarded(h, "loup_set_fixed_atoms", [&] { return set_fixed_impl(h, fixed); });
}

int loup_evaluate(
    LoupHandle* h, const double* positions, const double* cell, int compute_stress,
    double* out_energy_ev, double* out_forces, double* out_stress
) {
    return guarded(h, "loup_evaluate", [&] {
        return evaluate_impl(h, positions, cell, compute_stress, out_energy_ev, out_forces, out_stress);
    });
}

int loup_evaluate2(
    LoupHandle* h, const double* positions, const double* cell, int flags, double* out_energy_ev,
    double* out_forces, double* out_stress, double* out_atom_energies
) {
    return guarded(h, "loup_evaluate2", [&] {
        return evaluate2_impl(h, positions, cell, flags, out_energy_ev, out_forces, out_stress, out_atom_energies);
    });
}

int loup_run_nve(LoupHandle* h, int n_steps, double dt_fs, LoupMDStats* out_stats) {
    return guarded(h, "loup_run_nve", [&] { return run_nve_impl(h, n_steps, dt_fs, out_stats); });
}

int loup_run_langevin(
    LoupHandle* h, int n_steps, double dt_fs, double temp_k,
    double friction_per_fs, unsigned long long seed, LoupMDStats* out_stats
) {
    return guarded(h, "loup_run_langevin", [&] {
        return run_langevin_impl(h, n_steps, dt_fs, temp_k, friction_per_fs, seed, out_stats);
    });
}

int loup_run(LoupHandle* h, const LoupIntegrator* it, int n_steps, LoupMDStats* out_stats) {
    return guarded(h, "loup_run", [&] { return run_impl(h, it, n_steps, out_stats); });
}

int loup_get_state(LoupHandle* h, double* out_positions, double* out_velocities, double* out_forces, double* out_energy_ev) {
    return guarded(h, "loup_get_state", [&] {
        return get_state_impl(h, out_positions, out_velocities, out_forces, out_energy_ev);
    });
}

int loup_profile_force(LoupHandle* h, int warmup, int n_calls, double* out_total_us) {
    return guarded(h, "loup_profile_force", [&] { return profile_force_impl(h, warmup, n_calls, out_total_us); });
}

int loup_export_state(LoupHandle* h, void* buffer, size_t capacity, size_t* written) {
    return guarded(h, "loup_export_state", [&] { return export_state_impl(h, buffer, capacity, written); });
}

int loup_import_state(LoupHandle* h, const void* buffer, size_t size) {
    return guarded(h, "loup_import_state", [&] { return import_state_impl(h, buffer, size); });
}

int loup_debug_set_domain(LoupHandle* h, int n_local, int n_ghost, const int* atomic_numbers, double skin) {
    return guarded(h, "loup_debug_set_domain", [&] { return set_domain_impl(h, n_local, n_ghost, atomic_numbers, skin); });
}

int loup_debug_evaluate_domain(LoupHandle* h, const double* positions, const double* cell, const int* ghost_of,
                               double* out_energy_ev, double* out_forces, double* out_stress) {
    return guarded(h, "loup_debug_evaluate_domain", [&] {
        return evaluate_domain_impl(h, positions, cell, ghost_of, out_energy_ev, out_forces, out_stress);
    });
}

int loup_debug_domain_edges(LoupHandle* h, const double* positions, const double* cell, long long capacity,
                            int* out_receivers, int* out_senders, double* out_shifts, long long* out_count,
                            unsigned char* out_ghost_used) {
    return guarded(h, "loup_debug_domain_edges", [&] {
        return domain_edges_impl(h, positions, cell, capacity, out_receivers, out_senders, out_shifts, out_count,
                                 out_ghost_used);
    });
}

int loup_debug_evaluate_tiled(LoupHandle* h, int n_atoms, const int* atomic_numbers, const double* positions,
                              const double* cell, int receivers, double skin, double* out_energy_ev, double* out_forces,
                              double* out_stress, long long* out_tiles) {
    return guarded(h, "loup_debug_evaluate_tiled", [&] {
        return evaluate_tiled_impl(h, n_atoms, atomic_numbers, positions, cell, receivers, skin, out_energy_ev,
                                   out_forces, out_stress, out_tiles);
    });
}

int loup_set_execution(LoupHandle* h, int mode, int receivers, double memory_fraction) {
    return guarded(h, "loup_set_execution", [&] {
        if (!h) return -1;
        if (mode < LOUP_EXEC_AUTO || mode > LOUP_EXEC_WAVEFRONT) { loup::set_error("unknown mode %d", mode); return -1; }
        if (receivers < 0) { loup::set_error("receivers must not be negative, got %d", receivers); return -1; }
        if (!(memory_fraction > 0.0 && memory_fraction <= 1.0)) {
            loup::set_error("memory_fraction must be above 0 and at most 1, got %g", memory_fraction);
            return -1;
        }
        h->exec_mode = mode;
        h->exec_receivers = receivers;
        h->exec_fraction = memory_fraction;
        return 0;
    });
}

int loup_plan_execution(LoupHandle* h, int n_atoms, const int* atomic_numbers, const double* positions,
                        const double* cell, double skin, LoupMemoryPlan* out) {
    return guarded(h, "loup_plan_execution", [&] {
        return plan_execution_impl(h, n_atoms, atomic_numbers, positions, cell, skin, out);
    });
}

int loup_get_execution(LoupHandle* h, LoupExecution* out) {
    return guarded(h, "loup_get_execution", [&] {
        if (!h) return -1;
        if (!out) { loup::set_error("out is NULL"); return -1; }
        *out = h->last_exec;
        return 0;
    });
}

int loup_read_thermo(LoupHandle* h, LoupThermoRow* rows, int max_rows, int* n_rows) {
    return guarded(h, "loup_read_thermo", [&] { return read_thermo_impl(h, rows, max_rows, n_rows); });
}

int loup_minimize(LoupHandle* h, const LoupMinimizer* minimizer, int max_steps, LoupMinStats* out_stats) {
    return guarded(h, "loup_minimize", [&] { return minimize_impl(h, minimizer, max_steps, out_stats); });
}

int loup_get_step_displacement(LoupHandle* h, double* out_max_a) {
    return guarded(h, "loup_get_step_displacement", [&] { return step_displacement_impl(h, out_max_a); });
}

int loup_set_cell(LoupHandle* h, const double* cell, int scale_atoms) {
    return guarded(h, "loup_set_cell", [&] { return set_cell_impl(h, cell, scale_atoms); });
}

int loup_get_cell(LoupHandle* h, double* out_cell) {
    return guarded(h, "loup_get_cell", [&] { return get_cell_impl(h, out_cell); });
}

int loup_set_reference_volume(LoupHandle* h, double volume_a3) {
    return guarded(h, "loup_set_reference_volume", [&] { return set_reference_volume_impl(h, volume_a3); });
}

int loup_debug_csvr_noise(LoupHandle* h, unsigned long long seed, unsigned long long first_step, int n_steps,
                          int ndof, double* out_normal, double* out_chi2) {
    return guarded(h, "loup_debug_csvr_noise", [&] {
        return csvr_noise_impl(h, seed, first_step, n_steps, ndof, out_normal, out_chi2);
    });
}

int loup_plan_memory(LoupHandle* h, int n_atoms, double edges_per_atom, LoupMemoryPlan* out) {
    return guarded(h, "loup_plan_memory", [&] { return plan_memory_impl(h, n_atoms, edges_per_atom, out); });
}

const char* loup_last_error(const LoupHandle* h) {
    return h ? h->last_error.c_str() : tl_null_error.c_str();
}

int loup_abi_version(void) { return LOUP_ABI_VERSION; }

const char* loup_version(void) {
    static const std::string version = [] {
        char buf[96];
        snprintf(buf, sizeof(buf), LOUP_VERSION "+%016llx", (unsigned long long)library_build_hash());
        return std::string(buf);
    }();
    return version.c_str();
}

static int get_info_impl(const LoupHandle* h, LoupInfo* out) {
    if (!out) { loup::set_error("out is NULL"); return -1; }
    LoupInfo info{};
    info.abi = LOUP_ABI_VERSION;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, h->device_id));
    info.compute_capability = prop.major * 10 + prop.minor;
    memcpy(info.gpu_name, prop.name, std::min(strnlen(prop.name, sizeof(prop.name)), sizeof(info.gpu_name) - 1));
    info.n_layers = h->gen ? h->gen->n_layers() : 2;
    info.has_long_range = (h->gen && h->gen->has_lr()) ? 1 : 0;
    info.specialised = h->gen ? 0 : 1;
    info.n_species = h->n_species();
    info.r_max = (double)h->cutoff();
    const std::string sig = h->gen ? h->gen->kernel_signature() : std::string(LOUP_KERNEL_SIGNATURE);
    const size_t room = sizeof(info.kernel_signature) - 1;
    if (sig.size() <= room) {
        memcpy(info.kernel_signature, sig.data(), sig.size());
    } else {
        memcpy(info.kernel_signature, sig.data(), room - 3);
        memcpy(info.kernel_signature + room - 3, "...", 3);
    }
    *out = info;
    return 0;
}

int loup_get_info(const LoupHandle* h, LoupInfo* out) {
    // The error channel is the only state of the handle this call touches.
    return guarded(const_cast<LoupHandle*>(h), "loup_get_info", [&] { return get_info_impl(h, out); });
}

}  // extern "C"
