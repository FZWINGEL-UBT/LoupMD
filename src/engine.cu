#include "loupmd.h"
#include "model.hpp"
#include "kernels_common.cuh"
#include "generic.hpp"
#include "bundle.hpp"

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
                                      int* __restrict__ tmp_snd, int* __restrict__ tmp_shifts,
                                      uint32_t* __restrict__ sort_keys, const int* __restrict__ go);
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
                                             int* __restrict__ tmp_snd, int* __restrict__ tmp_shifts,
                                             uint32_t* __restrict__ sort_keys, const int* __restrict__ go);
__global__ void max_degree_kernel(int N, const int* __restrict__ deg, int* __restrict__ out);
__global__ void sort_gather_rows_kernel(
    CellMat64 cell, const int* __restrict__ row_ptr, const uint32_t* __restrict__ sort_keys,
    const int* __restrict__ tmp_snd, const int* __restrict__ tmp_shifts, const int* __restrict__ species,
    int* __restrict__ senders, double* __restrict__ shift_cart,
    int* __restrict__ order_err, const int* __restrict__ go);
__global__ void sort_gather_rows_global_kernel(
    CellMat64 cell, const int* __restrict__ row_ptr, const uint32_t* __restrict__ sort_keys,
    const int* __restrict__ tmp_snd, const int* __restrict__ tmp_shifts, const int* __restrict__ species,
    int* __restrict__ senders, double* __restrict__ shift_cart,
    int* __restrict__ order_err, unsigned long long* __restrict__ scratch);
__global__ void compute_and_count_active_kernel(
    int N, float r_max, const double* __restrict__ pos64,
    const int* __restrict__ row_ptr, const int* __restrict__ senders, const double* __restrict__ shift_cart,
    float4* __restrict__ tmp_vec_len, int* __restrict__ act_deg);
__global__ void scatter_active_edges_kernel(
    int N, float r_max, SplineGrid grid, int S, int pair_sym, const int* __restrict__ species,
    const int* __restrict__ row_ptr, const int* __restrict__ act_row_ptr,
    const int* __restrict__ senders, const float4* __restrict__ tmp_vec_len,
    int* __restrict__ act_senders, int* __restrict__ act_receivers,
    float4* __restrict__ act_vec_len,
    int* __restrict__ act_slab, float2* __restrict__ act_t_sx);
__global__ void check_skin_kernel(int N, double half_skin, const double* __restrict__ pos64,
                                  const double* __restrict__ ref_pos64, int* __restrict__ need_rebuild);
// Publications to mapped pinned host memory: the next value of a device counter selects one of four slots (so two
// publications of one MD step cannot overwrite each other before the host reads them), the slot gets the three
// values and then the counter value, which the host polls (wait_published) instead of calling cudaStreamSynchronize,
// whose wake-up leaves the GPU idle for 0.1-0.2 ms under WSL2. One thread; a stream-ordered (or graph-ordered)
// launch sees every earlier write of the stream.
__device__ __forceinline__ void publish3(unsigned int* counter, int* pub, int a, int b, int c) {
    const unsigned int seq = *counter + 1u;
    *counter = seq;
    volatile int* p = pub + 4 * (seq & 3u);
    p[1] = a;
    p[2] = b;
    p[3] = c;
    __threadfence_system();
    reinterpret_cast<volatile unsigned int*>(p)[0] = seq;
}

__global__ void publish_kernel(const int* __restrict__ a, const int* __restrict__ b, const int* __restrict__ c,
                               unsigned int* __restrict__ counter, int* __restrict__ pub) {
    publish3(counter, pub, a ? *a : 0, b ? *b : 0, c ? *c : 0);
}

// The MD step's hinge (run_md_loop): publishes the skin-check flag and the last rebuild's row-order flag, and picks the
// body of the step graph's SWITCH node: 0 the rest (force pass + second half) when no rebuild is needed, 1 the rebuild
// branch (has_rebuild) when one is, none when the rows are invalid or the host rebuilds. The host queues the next step
// while this one's forces run instead of reading the flag at an idle GPU. One SWITCH node rather than an IF node per
// body: every conditional node a step evaluates costs time even when its body does not run.
__global__ void guard_step_kernel(cudaGraphConditionalHandle cond, int has_rebuild, const int* __restrict__ need_rebuild,
                                  const int* __restrict__ order_err, unsigned int* __restrict__ counter,
                                  int* __restrict__ pub) {
    const int flag = *need_rebuild;
    const int bad_rows = *order_err;
    publish3(counter, pub, flag, bad_rows, 0);
    cudaGraphSetConditional(cond, bad_rows ? 2u : !flag ? 0u : has_rebuild ? 1u : 2u);
}

// The step graph's rebuild branch after nl_count: publishes the edge count, the longest row and whether the new list
// fits what the graph was captured for (edge capacity / bucket, row-sort width), and stores that verdict in *go: the
// branch's fill, sort and commit kernels run only then (one nesting level less than an IF node around them);
// otherwise the host finishes the rebuild.
__global__ void rebuild_gate_kernel(const int* __restrict__ row_ptr_end, const int* __restrict__ max_deg, int E_lim,
                                    int P_cap, int* __restrict__ go, unsigned int* __restrict__ counter,
                                    int* __restrict__ pub) {
    const int E = *row_ptr_end;
    const int m = *max_deg;
    const int fits = E > 0 && E <= E_lim && m <= P_cap;
    *go = fits;
    publish3(counter, pub, E, m, fits);
}

// After a rebuild in MD: the force pass (the IF node's body) runs only if the rebuild's rows passed the order check,
// which the host reads with the next step's publication, and (go, the step graph's rebuild branch) were built.
__global__ void rest_guard_kernel(cudaGraphConditionalHandle cond, const int* __restrict__ order_err,
                                  const int* __restrict__ go) {
    cudaGraphSetConditional(cond, (*order_err || (go && !*go)) ? 0u : 1u);
}

// Neighbour-rebuild bookkeeping (nl_bin, nl_count, nl_commit) as kernels rather than memset / memcpy nodes.
__global__ void nl_reset_kernel(int n_bins, int* __restrict__ bin_counts, int* __restrict__ deg_end,
                                int* __restrict__ order_err, int* __restrict__ max_deg) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n_bins) bin_counts[i] = 0;  // [n_bins] too: the bin scan's total
    if (i == 0) {
        *deg_end = 0;  // the row scan's total
        *order_err = 0;
        *max_deg = 0;
    }
}

__global__ void copy_ints_kernel(int n, const int* __restrict__ src, int* __restrict__ dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void nl_commit_kernel(int n, const double* __restrict__ pos, double* __restrict__ ref_pos,
                                 int* __restrict__ need_rebuild, const int* __restrict__ go) {
    if (go && !*go) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ref_pos[i] = pos[i];
    if (i == 0) *need_rebuild = 0;
}

constexpr double ASE_FS = 0.09822694788464063;
constexpr double ASE_KB = 8.617330337217213e-5;

__global__ void nve_first_half_kernel(int N, double dt_ase, const double* __restrict__ forces,
                                      const double* __restrict__ inv_masses,
                                      double* __restrict__ pos64, double* __restrict__ vel64);
__global__ void nve_second_half_kernel(int N, double dt_ase, const double* __restrict__ forces,
                                       const double* __restrict__ inv_masses, double* __restrict__ vel64);
__global__ void langevin_first_half_kernel(
    int N, double dt_ase, double c1, double c2, double k_c3, double k_c4, double k_c5,
    unsigned long long seed, const unsigned long long* __restrict__ step_counter,
    const double* __restrict__ forces, const double* __restrict__ inv_masses,
    const double* __restrict__ sqrt_inv_masses, double* __restrict__ pos64,
    double* __restrict__ vel64, double* __restrict__ rnd_vel_save);
__global__ void langevin_second_half_kernel(
    int N, double c1, double c2, const double* __restrict__ forces,
    const double* __restrict__ inv_masses, const double* __restrict__ rnd_vel_save,
    double* __restrict__ vel64, unsigned long long* __restrict__ step_counter);
__global__ void kinetic_energy_kernel(int N, const double* __restrict__ masses,
                                      const double* __restrict__ vel64, double* __restrict__ out_ke);
}  // namespace loup

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
// the same shapes in both, so a stored plan's tile index means the same kernel shape. Own format 1 passes beta at run
// time to every entry; format 2 fixes it at 0 in the plain entries (tiles 0-2 SASS-identical to the FastMACE ones)
// and adds loup_gemm_tc_{rrb,rcb}_<t> for beta != 0.
struct HybCubin {
    static constexpr int kMaxTiles = 8;
    bool loaded = false;
    bool own_format = false;  // loup_gemm_tc_info
    bool beta_ok = false;     // C = A B + beta C may use it
    int ntiles = 0;
    uint64_t cubin_hash = 0;  // the loaded cubin file's FNV-1a hash (keys the GEMM plan cache)
    std::string path_loaded;  // for diagnostics (the cubin's math: its build, see cubins/src)
    void* lib = nullptr;
    CUmodule mod = nullptr;
    CUfunction fn[kMaxTiles][2] = {};   // [tile][transB ? 1 : 0]: C = A B (format 1: any beta)
    CUfunction fnb[kMaxTiles][2] = {};  // beta != 0 (format 1: the fn entries)
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
        // cc 8.0 (A100): LoupMD's own format-2 cubin (cubins/src/gemm_tc.cu: the shipped three tiles SASS-identical
        // plus three larger ones and compile-time beta variants), faster or equal on every family measured (S74a) and
        // gated (S75g); other 8.x GPUs, not measured with it, keep the FastMACE cubin.
        if (maj == 8) cname = (min == 0) ? "gemm_tc_hyb_sm80.cubin" : "tf32x3_hyb_sm80.cubin";
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
            if (p_cuMemcpyDtoH(info.data(), gptr, gsz) != CUDA_SUCCESS || (info[0] != 1 && info[0] != 2)) {
                destroy();
                return;
            }
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
                    fnb[t][b] = fn[t][b];
                    if (beta_ok && info[0] == 2) {
                        snprintf(name, sizeof(name), "loup_gemm_tc_%s_%d", b ? "rcb" : "rrb", t);
                        if (p_cuModuleGetFunction(&fnb[t][b], mod, name) != CUDA_SUCCESS) { destroy(); return; }
                    }
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
                    for (CUfunction f : {fn[t][b], fnb[t][b]})
                        if (f) p_cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem[t][b]);
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
            (beta != 0.0f ? fnb : fn)[tile][transB ? 1 : 0], gx, gy, 1,
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
// Test overrides (LOUP_REORDER=0|1, LOUP_ROW_SORT=global, LOUP_MD_REBUILD=host|fallback): pin a path that a size
// threshold or the cadence would otherwise pick, so the tests can cover every one.
struct VariantOverrides {
    int reorder = -1;
    bool row_sort_global = false;  // neighbour rows sorted in global scratch (else only rows over 4096 edges)
    int md_rebuild = 0;  // MD rebuilds: 0 on the GPU in the step graph, 1 host-driven, 2 counted on the GPU, finished on the host
    VariantOverrides() {
        if (const char* s = std::getenv("LOUP_REORDER")) reorder = (s[0] == '1') ? 1 : (s[0] == '0') ? 0 : -1;
        if (const char* s = std::getenv("LOUP_ROW_SORT")) row_sort_global = strcmp(s, "global") == 0;
        if (const char* s = std::getenv("LOUP_MD_REBUILD")) md_rebuild = strcmp(s, "host") == 0 ? 1 : strcmp(s, "fallback") == 0 ? 2 : 0;
    }
};

static uint64_t library_build_hash() {
    static uint64_t cached = 0;
    if (cached) return cached;
    Dl_info info{};
    if (dladdr((void*)&library_build_hash, &info) && info.dli_fname) cached = fnv1a_file(info.dli_fname);
    if (!cached) cached = 1;
    return cached;
}

struct LoupHandle {
    int device_id = 0;
    int l2_bytes = 0;
    bool is_sm120 = false;
    // The own-SGEMM extensions (sgemm64 / sgemm32 + split-K candidates, the narrow-N rules, N-fastest rasterisation and
    // the N = 96 kernel of sgemm128): on for sm_120 (5060 Ti), and on compute capability 8.0 when no tensor-core cubin
    // is loaded (long-range models, float32 SIMT GEMMs: A100, S82: POLAR-1-S NaCl216 0.870, POLAR-1-M 0.959); with
    // the cubin they lose there (S80: OFF23-S water216 1.034, OMOL-0 1.008). LOUP_OWN_EXT=0|1 forces them (tests).
    bool own_ext = false;
    // The lane-group neighbour count / fill kernels: on for sm_120 and compute capability 8.0 (A100, S80: OFF23-S
    // water216 0.843, OFF23-M water216 0.885, MH-1 NaCl216 0.965, large cells 0.990-0.995; the same lists, so the same
    // forces); LOUP_NB_GROUPS=0|1 forces them (tests).
    bool nb_groups_ok = false;
    // LOUP_FORCE_TC=1 (tests): every GEMM the tensor-core cubin can take runs its fastest cubin tile (no cuBLASLt or own
    // SGEMM candidates), so gates cover the cubin's arithmetic on every eligible shape; no plans are loaded or saved.
    bool force_tc = false;
    // The generic path's gate in the epilogues of own-sgemm128 projections (sgemm128_gate, gemm_gate): the backward in
    // the W_2^T GEMMs, the forward of residual-free degrees l >= 1 in layer 0's lin_l GEMMs. LOUP_GATE_FUSE=0 /
    // LOUP_GATE_FUSE_FWD=0 disable them (tests).
    bool gate_fuse = true, gate_fuse_fwd = true;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cublasLtHandle_t cublas_lt = nullptr;
    // Host polling of stream positions (loup::publish_kernel, wait_published): mapped pinned words [0] = counter,
    // [1..3] = published values; the device counter's mirror on the host.
    int* h_pub = nullptr;
    int* d_pub = nullptr;
    unsigned int* d_pub_counter = nullptr;
    unsigned int pub_count = 0;
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
    VariantOverrides ovr;         // test overrides from the environment, read at loup_create
    bool variants_tuned = false;  // tune_rebuild_variants ran for this system
    struct DiskPlan {
        int M, N, K;
        uint8_t transB, beta_nz, use_hyb, hyb_tile, use_own;
        cublasLtMatmulAlgo_t algo;
    };
    std::string plan_cache_path;
    std::map<std::tuple<int, int, int, bool, bool>, DiskPlan> disk_plans;
    bool disk_dirty = false;
    void load_disk_plans(int device_id) {
        if (force_tc) return;  // forced picks must not reach (or come from) the plan cache
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
    std::unique_ptr<loup::GenericModel> gen;  // the model (every bundle is arch=generic since JIT_DESIGN.md stage 5)
    int n_species() const { return gen->S(); }
    float cutoff() const { return gen->r_max(); }
    int species_z(int s) const { return gen->atomic_numbers()[s]; }
    double species_e0(int s) const { return gen->e0()[s]; }

    // System topology (atoms grouped by species 0..S-1)
    int N = 0;
    double skin = 0.5;
    double e0_total = 0.0;
    std::vector<int> perm;          // device (species-grouped, spatially ordered) index -> caller index
    std::vector<int> h_order;       // scratch: new device index -> old device index at an atom reorder
    std::vector<int> species_start; // [S+1]
    std::vector<int> species_count; // [S]
    loup::CellMat64 cell{};
    double cell_vol = 0.0;
    double cell_heights[3] = {0, 0, 0};

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
    unsigned long long* d_step_counter = nullptr;
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
    int* d_row_ptr = nullptr;
    int* d_act_deg = nullptr;
    int* d_act_row_ptr = nullptr;
    void* d_cub_tmp = nullptr;
    size_t cub_tmp_bytes = 0;

    // Edge buffers (capacity >= E_cap >= E_graph >= E_skin)
    int E_skin = 0;
    int E_graph = 0;
    int E_cap = 0;
    // Candidate edges (within r_max + skin), row i holding the edges whose receiver is i (S113: no per-edge receiver
    // or pair index)
    int* d_senders = nullptr;
    double* d_shift_cart = nullptr;  // [E, 3] shift @ cell of each skin edge (float64), set at rebuild
    int* d_tmp_snd = nullptr;
    int* d_tmp_shifts = nullptr;
    uint32_t* d_sort_keys = nullptr;
    float4* d_act_vec_len = nullptr;
    int* d_act_senders = nullptr;
    int* d_act_receivers = nullptr;
    int* d_act_slab = nullptr;
    float2* d_act_t_sx = nullptr;
    float4* d_tmp_vec_len = nullptr;
    float* d_edge_force_1 = nullptr;

    // Zero-initialized per-step slab
    void* d_zero_slab = nullptr;
    size_t zero_slab_bytes = 0;
    double* d_energy_f64 = nullptr;
    double* d_forces_f64 = nullptr;
    double* d_virial_grad_f64 = nullptr;

    // Graph caches
    cudaGraph_t force_graph = nullptr;
    cudaGraphExec_t force_graph_exec = nullptr;
    int force_graph_stress = -1;

    // MD step: first half + skin check + guard_step_kernel + IF (no rebuild, valid rows) {forces + second half}, and
    // when md_dev_rebuild, IF (rebuild, valid rows) {count + rebuild_gate_kernel + IF (fits) {fill + commit + checked
    // rest}}: the rebuild runs on the GPU without a host round trip unless the new list outgrows the captured sizes.
    cudaGraph_t md_step_graph = nullptr;
    cudaGraphExec_t md_step_exec = nullptr;
    bool md_dev_rebuild = false;
    // Checked rest, launched after a host-driven rebuild: rest_guard_kernel + IF (valid rows) {forces + second half}.
    cudaGraph_t md_checked_graph = nullptr;
    cudaGraphExec_t md_checked_exec = nullptr;

    void invalidate_graphs() {
        if (force_graph_exec) { cudaGraphExecDestroy(force_graph_exec); force_graph_exec = nullptr; }
        if (force_graph) { cudaGraphDestroy(force_graph); force_graph = nullptr; }
        force_graph_stress = -1;
        invalidate_md_graphs();
    }
    void invalidate_md_graphs() {
        if (md_step_exec) { cudaGraphExecDestroy(md_step_exec); md_step_exec = nullptr; }
        if (md_step_graph) { cudaGraphDestroy(md_step_graph); md_step_graph = nullptr; }
        if (md_checked_exec) { cudaGraphExecDestroy(md_checked_exec); md_checked_exec = nullptr; }
        if (md_checked_graph) { cudaGraphDestroy(md_checked_graph); md_checked_graph = nullptr; }
        md_dev_rebuild = false;
        md_graph_key = -1;
    }
    // Which step the MD graphs hold: -1 none, -2 an integrator's (they bake in dt, thermostat and seed: md_int_*, so
    // a run call with other values drops them and one with the same values, e.g. a run in blocks, reuses them),
    // 2 * kind + compute_stress the external steps of loup_compute_device / loup_evaluate, kept across calls.
    int md_graph_key = -1;
    int md_int_kind = -1;
    double md_int_dt = 0.0, md_int_temp = 0.0, md_int_fric = 0.0;
    unsigned long long md_int_seed = 0;

    // loup_compute_device (host codes whose atoms stay on the GPU, e.g. LAMMPS Kokkos): the caller's rows are gathered
    // into the engine's order and the forces added back through a device copy of the inverse permutation.
    int* d_inv_perm = nullptr;    // [N] caller atom -> device index
    int perm_epoch = 0;           // bumped whenever perm changes (set_system, an atom reorder)
    int inv_perm_epoch = -1;      // the perm_epoch d_inv_perm was written for
    int* d_io_bad = nullptr;      // set by io_gather_kernel when an id is outside 0 .. N-1
    // loup_evaluate's host I/O: page-locked staging [3N positions | 3N forces | energy | 6 virial], so the uploads and
    // downloads are asynchronous copies and the host learns that the results have landed from one publication instead
    // of blocking transfers and a stream synchronisation (a host round trip each, 100-200 us under WSL 2).
    double* h_stage = nullptr;
    double* stage_pos() const { return h_stage; }
    double* stage_f() const { return h_stage + (size_t)3 * N; }
    double* stage_ev() const { return h_stage + (size_t)6 * N; }
    cudaEvent_t ev_io_in = nullptr, ev_io_out = nullptr;

    int n_rebuilds = 0;     // rebuilds since set_system (atom reorder cadence)
    int* d_order_err = nullptr;  // set by sort_gather_rows_kernel when a row breaks the species-run contract
    int* d_max_deg = nullptr;    // [0] longest neighbour row of the current rebuild, [1] the MD rebuild gate's verdict
    int max_deg_last = 0;        // longest row of the last rebuild (sizes the MD step graph's row sort)
    bool nl_valid = false;  // neighbour list built for the current system (ref positions and shifts are set)

    void free_system() {
        if (gen) gen->free_system();
        invalidate_graphs();
        clear_lt_cache();
        variants_tuned = false;
        nl_valid = false;
        n_rebuilds = 0;
        N = 0;
        auto freep = [](auto*& p) {
            if (p) { cudaFree(p); p = nullptr; }
        };
        freep(d_species); freep(d_pos64); freep(d_ref_pos64); freep(d_vel64);
        freep(d_masses); freep(d_inv_masses); freep(d_sqrt_inv_masses); freep(d_rnd_vel);
        freep(d_need_rebuild); freep(d_step_counter); freep(d_ke_f64);
        freep(d_inv_perm); freep(d_io_bad); inv_perm_epoch = -1;
        if (h_stage) { cudaFreeHost(h_stage); h_stage = nullptr; }
        freep(d_pos_wrapped); freep(d_atom_wrap); freep(d_atom_bin);
        freep(d_ro_keys); freep(d_ro_vals); freep(d_ro_tmp); freep(d_order_err); freep(d_max_deg);
        freep(d_bin_counts); freep(d_bin_offsets); freep(d_bin_cursor);
        freep(d_bin_atom_id); freep(d_bin_pos); freep(d_bin_pos_f32); freep(d_bin_wrap); freep(d_bin_sp);
        freep(d_deg); freep(d_row_ptr); freep(d_cub_tmp); cub_tmp_bytes = 0;
        freep(d_act_deg); freep(d_act_row_ptr);

        freep(d_senders); freep(d_shift_cart);
        // views into d_tmp_vec_len / d_edge_force_1 (ensure_edge_capacity), freed with them
        d_tmp_snd = d_tmp_shifts = nullptr;
        d_sort_keys = nullptr;
        freep(d_act_vec_len);
        freep(d_act_senders); freep(d_act_receivers);
        freep(d_act_slab); freep(d_act_t_sx);
        freep(d_tmp_vec_len);
        freep(d_edge_force_1);
        E_cap = 0; E_skin = 0;

        freep(d_zero_slab); zero_slab_bytes = 0;
    }
};

namespace loup {
void sgemm128_init();
void sgemm128(int M, int N, int K, const float* A, const float* B, float* C, float beta, bool n_fastest,
              cudaStream_t stream);
void sgemm64(int M, int N, int K, const float* A, const float* B, float* C, float beta, cudaStream_t stream);
void sgemm32(int M, int N, int K, const float* A, const float* B, float* C, float* ws, float beta, int split_k,
             cudaStream_t stream);
void sgemm128_gate(bool fwd, int l, int M, int N, int K, const float* A, const float* B, const GateArgs& g,
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
        auto dit = h->disk_plans.find(key);
        if (dit != h->disk_plans.end()) {
            const auto& dp = dit->second;
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
                const bool tc_ok = alpha == 1.0f && h->hyb.takes(beta) && (K & 3) == 0 && (N & 3) == 0 &&
                                   ((((uintptr_t)A | (uintptr_t)B | (uintptr_t)d_tmpC) & 15) == 0);
                const bool only_tc = h->force_tc && tc_ok;
                for (int r = 0; r < (only_tc ? 0 : n_res); ++r) {
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
                if (tc_ok) {
                    for (int t = 0; t < h->hyb.ntiles; ++t) {
                        const float ms = time_min([&] { h->hyb.launch(t, transB, M, N, K, A, lda, B, ldb, d_tmpC, ldc, beta, strm); });
                        if (consider(ms)) { plan.use_hyb = true; plan.hyb_tile = t; }
                    }
                }
                if (!only_tc && own_ok && (((uintptr_t)d_tmpC & 15) == 0)) {
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

// A projection fused with degree l's gate (sgemm128_gate): backward dgout_l = A @ B^T (B [N, K], a model weight),
// forward lin_l = A @ B (B [K, N]), when sgemm_rm's plan for that GEMM is the own sgemm128 with N tiles fastest (whose
// tiles and arithmetic the fused kernel shares) and the shape fits it. False: nothing launched (no plan yet, as in the
// uncaptured autotune pass, or another implementation won). A null: only reports whether it would run.
static bool gemm_gate(LoupHandle* h, bool fwd, int l, int M, int N, int K, const float* A, const float* B,
                      const loup::GateArgs& g) {
    if (!h->own_ext || l < (fwd ? 1 : 0) || l > 3 || M != g.natoms * (2 * l + 1) || N != g.C || (N % 128) != 0 ||
        (K % 16) != 0 || ((((uintptr_t)A | (uintptr_t)B) & 15) != 0))
        return false;
    const auto it = h->lt_cache.find(std::make_tuple(M, N, K, !fwd, false));
    if (it == h->lt_cache.end() || it->second.own_mode != 1) return false;
    if (!A) return true;
    cudaStream_t strm = h->gemm_stream ? h->gemm_stream : h->stream;
    loup::sgemm128_gate(fwd, l, M, N, K, A, fwd ? B : weight_kn(h, B, N, K, strm), g, strm);
    return true;
}

static int ensure_edge_capacity(LoupHandle* h, int needed) {
    if (needed <= h->E_cap) return 0;
    int cap = std::max(needed + needed / 4 + 1024, 4096);
    // Capacity 0 and no graphs until every array is reallocated: after a failed allocation a later call grows again
    // instead of running on freed or null arrays.
    h->invalidate_graphs();
    h->E_cap = 0;
    auto reallocp = [&](auto*& p, size_t elem_sz) {
        if (p) cudaFree(p);
        p = nullptr;
        return cudaMalloc(&p, (size_t)cap * elem_sz);
    };
    CUDA_CHECK(reallocp(h->d_senders, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_shift_cart, 3 * sizeof(double)));
    // The rebuild's fill / sort temporaries are views into arrays a force pass rewrites before reading and that are
    // dead during a rebuild: senders and sort keys in d_tmp_vec_len (8 of its 16 B per edge), the shifts in
    // d_edge_force_1.
    h->d_tmp_snd = h->d_tmp_shifts = nullptr;
    h->d_sort_keys = nullptr;
    CUDA_CHECK(reallocp(h->d_act_vec_len, sizeof(float4)));
    CUDA_CHECK(reallocp(h->d_act_senders, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_act_receivers, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_act_slab, sizeof(int)));
    CUDA_CHECK(reallocp(h->d_act_t_sx, sizeof(float2)));
    CUDA_CHECK(reallocp(h->d_tmp_vec_len, sizeof(float4)));
    CUDA_CHECK(reallocp(h->d_edge_force_1, 3 * sizeof(float)));
    int* tmp4 = reinterpret_cast<int*>(h->d_tmp_vec_len);
    h->d_tmp_snd = tmp4;
    h->d_sort_keys = reinterpret_cast<uint32_t*>(tmp4 + (size_t)cap);
    h->d_tmp_shifts = reinterpret_cast<int*>(h->d_edge_force_1);
    h->gen->ensure_edges(cap);
    h->E_cap = cap;
    return 0;
}

static int ensure_cub_tmp(LoupHandle* h, size_t bytes) {
    if (bytes <= h->cub_tmp_bytes) return 0;
    // Queued work (a scan earlier in this rebuild, or the MD step still in flight) may still use the old
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

// Queue loup::publish_kernel(a, b, c) on the main stream (outside capture); returns the counter value it will post.
static unsigned int publish(LoupHandle* h, const int* a, const int* b, const int* c) {
    loup::publish_kernel<<<1, 1, 0, h->stream>>>(a, b, c, h->d_pub_counter, h->d_pub);
    return ++h->pub_count;
}

// Polls the mapped publish slot of publication `want` until it has landed, then copies its values to out[0..2]. The
// stream is queried every few thousand polls so a failed kernel or an emptied queue cannot hang the host.
static int wait_published(LoupHandle* h, unsigned int want, int out[3]) {
    volatile int* p = h->h_pub + 4 * (want & 3u);
    volatile unsigned int* seq = reinterpret_cast<volatile unsigned int*>(p);
    for (unsigned int spins = 1; *seq != want; ++spins) {
        if ((spins & 4095) == 0) {
            const cudaError_t e = cudaStreamQuery(h->stream);
            if (e == cudaSuccess && *seq != want) {
                fprintf(stderr, "loupmd: publish %u never landed (slot has %u)\n", want, *seq);
                return -1;
            }
            if (e != cudaSuccess && e != cudaErrorNotReady) CUDA_CHECK(e);
        }
        __builtin_ia32_pause();
    }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    out[0] = p[1];
    out[1] = p[2];
    out[2] = p[3];
    return 0;
}

// Reads up to three device ints at the current end of the main stream (polling the mapped publish words).
static int read_device_ints(LoupHandle* h, const int* a, const int* b, const int* c, int out[3]) {
    return wait_published(h, publish(h, a, b, c), out);
}

static int order_error() {
    fprintf(stderr, "LoupMD: neighbour rows are not grouped by ascending sender species\n");
    return -1;
}

// Geometry of one neighbour-list build for the current cell, skin and N: bin grid, scan ranges, kernel shapes.
struct NlGeom {
    int nx, ny, nz, n_bins, rx, ry, rz, gridN, blkN, gridNb;
    bool nb_groups;
    size_t nb_smem;
    double r_list2;
    float r_lo2_f32, r_hi2_f32;
};
static constexpr int kNlBlk = 128;

static NlGeom nl_geom(const LoupHandle* h) {
    NlGeom g;
    const int N = h->N;
    const double r_list = (double)h->cutoff() + h->skin;
    g.r_list2 = r_list * r_list;
    g.nx = std::max(1, (int)std::floor(h->cell_heights[0] / 3.25));
    g.ny = std::max(1, (int)std::floor(h->cell_heights[1] / 3.25));
    g.nz = std::max(1, (int)std::floor(h->cell_heights[2] / 3.25));
    while ((int64_t)g.nx * g.ny * g.nz > h->max_bins) {
        if (g.nx >= g.ny && g.nx >= g.nz && g.nx > 1) --g.nx;
        else if (g.ny >= g.nx && g.ny >= g.nz && g.ny > 1) --g.ny;
        else if (g.nz > 1) --g.nz;
        else break;
    }
    g.n_bins = g.nx * g.ny * g.nz;
    g.rx = (int)std::ceil(r_list * g.nx / h->cell_heights[0]);
    g.ry = (int)std::ceil(r_list * g.ny / h->cell_heights[1]);
    g.rz = (int)std::ceil(r_list * g.nz / h->cell_heights[2]);
    g.gridN = (N + kNlBlk - 1) / kNlBlk;
    const double max_h = std::max({h->cell_heights[0], h->cell_heights[1], h->cell_heights[2]});
    const float margin = std::max(0.01f, 1e-4f * (float)max_h);
    const float r_lo_f = std::max(0.0f, (float)r_list - margin);
    const float r_hi_f = (float)r_list + margin;
    g.r_lo2_f32 = r_lo_f * r_lo_f;
    g.r_hi2_f32 = r_hi_f * r_hi_f;
    // count / fill: on sm_120 kNeighborLanes lanes per atom (128-thread blocks hold whole lane groups; fill keeps one
    // nscan-int bin-offset table per atom in shared memory); elsewhere, or when that table would exceed 48 KB, the
    // one-thread-per-atom kernels (32-thread blocks), which the lane-group kernels were not measured against.
    const int nscan = (2 * g.rx + 1) * (2 * g.ry + 1) * (2 * g.rz + 1);
    g.nb_smem = (size_t)(128 / loup::kNeighborLanes) * nscan * sizeof(int);
    g.nb_groups = h->nb_groups_ok && g.nb_smem <= 48 * 1024;
    g.blkN = g.nb_groups ? 128 : 32;
    g.gridNb = (int)(((int64_t)N * (g.nb_groups ? loup::kNeighborLanes : 1) + g.blkN - 1) / g.blkN);
    return g;
}

// Scratch of nl_count's two scans.
static size_t nl_cub_bytes(LoupHandle* h, const NlGeom& g) {
    size_t scan_bytes = 0, scanN_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, h->d_bin_counts, h->d_bin_offsets, g.n_bins + 1, h->stream);
    cub::DeviceScan::ExclusiveSum(nullptr, scanN_bytes, h->d_deg, h->d_row_ptr, h->N + 1, h->stream);
    return std::max(scan_bytes, scanN_bytes);
}

// Wrapped positions and per-bin atom counts; zeroes the scans' totals, the row-order flag and the longest-row slot
// (kernels, not memset / memcpy nodes: the MD step graph launches this branch from the device, where copy nodes cost
// tens of microseconds each under WSL2).
static int nl_bin(LoupHandle* h, const NlGeom& g) {
    loup::nl_reset_kernel<<<(g.n_bins + 1 + 255) / 256, 256, 0, h->stream>>>(
        g.n_bins, h->d_bin_counts, h->d_deg + h->N, h->d_order_err, h->d_max_deg);
    loup::wrap_and_bin_kernel<<<g.gridN, kNlBlk, 0, h->stream>>>(
        h->N, h->cell, g.nx, g.ny, g.nz, h->d_pos64, h->d_pos_wrapped, h->d_atom_wrap, h->d_atom_bin, h->d_bin_counts);
    return 0;
}

// Bin offsets and slots, neighbour counts, row_ptr (edge count in row_ptr[N]) and the longest row. No host round
// trip (d_cub_tmp must hold nl_cub_bytes), so run_md_loop captures it.
static int nl_count(LoupHandle* h, const NlGeom& g) {
    const int N = h->N;
    size_t scan_bytes = h->cub_tmp_bytes;
    cub::DeviceScan::ExclusiveSum(h->d_cub_tmp, scan_bytes, h->d_bin_counts, h->d_bin_offsets, g.n_bins + 1, h->stream);
    loup::copy_ints_kernel<<<(g.n_bins + 255) / 256, 256, 0, h->stream>>>(g.n_bins, h->d_bin_offsets, h->d_bin_cursor);
    loup::scatter_bins_kernel<<<g.gridN, kNlBlk, 0, h->stream>>>(
        N, h->d_pos_wrapped, h->d_atom_wrap, h->d_species, h->d_atom_bin, h->d_bin_cursor,
        h->d_bin_atom_id, h->d_bin_pos, h->d_bin_pos_f32, h->d_bin_wrap, h->d_bin_sp);
    if (g.nb_groups)
        loup::count_neighbors_kernel<<<g.gridNb, g.blkN, 0, h->stream>>>(
            N, h->cell, g.nx, g.ny, g.nz, g.rx, g.ry, g.rz, g.r_list2, g.r_lo2_f32, g.r_hi2_f32, h->d_pos_wrapped,
            h->d_atom_bin, h->d_bin_offsets, h->d_bin_atom_id, h->d_bin_pos, h->d_bin_pos_f32, h->d_deg);
    else
        loup::count_neighbors_scalar_kernel<<<g.gridNb, g.blkN, 0, h->stream>>>(
            N, h->cell, g.nx, g.ny, g.nz, g.rx, g.ry, g.rz, g.r_list2, g.r_lo2_f32, g.r_hi2_f32, h->d_pos_wrapped,
            h->d_atom_bin, h->d_bin_offsets, h->d_bin_atom_id, h->d_bin_pos, h->d_bin_pos_f32, h->d_deg);
    size_t scanN_bytes = h->cub_tmp_bytes;
    cub::DeviceScan::ExclusiveSum(h->d_cub_tmp, scanN_bytes, h->d_deg, h->d_row_ptr, N + 1, h->stream);
    loup::max_degree_kernel<<<(N + 255) / 256, 256, 0, h->stream>>>(N, h->d_deg, h->d_max_deg);
    return 0;
}

// Smallest power of two >= n (the row sort's width).
static int pow2_at_least(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Fills the rows into the edge arrays (capacity >= the edge count), then sorts each receiver's row by key
// (sym_pair << 19 | length bits), ties in fill order, and gathers it. Contract: for a fixed receiver sym_pair grows
// with the sender species, so every row ends up grouped by ascending sender species, which the receiver-owned
// first_fwd_kernel (layer0_kernels.cuh) relies on (it writes each species slot once); the sort kernel flags a row that
// breaks it (a changed key or sym_pair would otherwise corrupt S0 silently). P: the sort width, a power of two >= the
// longest row; rows of P <= 4096 sort in shared memory. go: the MD step graph's rebuild verdict (nullptr: always).
static int nl_fill(LoupHandle* h, const NlGeom& g, int P, const int* go = nullptr) {
    const int N = h->N;
    const int S = h->n_species();
    if (g.nb_groups)
        loup::fill_neighbors_kernel<<<g.gridNb, g.blkN, g.nb_smem, h->stream>>>(
            N, S, h->cell, g.nx, g.ny, g.nz, g.rx, g.ry, g.rz, g.r_list2, g.r_lo2_f32, g.r_hi2_f32,
            h->d_pos_wrapped, h->d_atom_wrap, h->d_species, h->d_atom_bin,
            h->d_bin_offsets, h->d_bin_atom_id, h->d_bin_pos, h->d_bin_pos_f32, h->d_bin_wrap, h->d_bin_sp,
            h->d_row_ptr, h->d_tmp_snd, h->d_tmp_shifts, h->d_sort_keys, go);
    else
        loup::fill_neighbors_scalar_kernel<<<g.gridNb, g.blkN, 0, h->stream>>>(
            N, S, h->cell, g.nx, g.ny, g.nz, g.rx, g.ry, g.rz, g.r_list2, g.r_lo2_f32, g.r_hi2_f32,
            h->d_pos_wrapped, h->d_atom_wrap, h->d_species, h->d_atom_bin,
            h->d_bin_offsets, h->d_bin_atom_id, h->d_bin_pos, h->d_bin_pos_f32, h->d_bin_wrap, h->d_bin_sp,
            h->d_row_ptr, h->d_tmp_snd, h->d_tmp_shifts, h->d_sort_keys, go);
    const int threads = std::min(256, std::max(32, P / 2));
    if (P <= 4096 && !h->ovr.row_sort_global)
        loup::sort_gather_rows_kernel<<<N, threads, P * sizeof(unsigned long long), h->stream>>>(
            h->cell, h->d_row_ptr, h->d_sort_keys, h->d_tmp_snd, h->d_tmp_shifts, h->d_species, h->d_senders,
            h->d_shift_cart, h->d_order_err, go);
    else  // scratch: act_vec_len (16 B per edge) is per-step data, free during a rebuild
        loup::sort_gather_rows_global_kernel<<<N, threads, 0, h->stream>>>(
            h->cell, h->d_row_ptr, h->d_sort_keys, h->d_tmp_snd, h->d_tmp_shifts, h->d_species, h->d_senders,
            h->d_shift_cart, h->d_order_err,
            reinterpret_cast<unsigned long long*>(h->d_act_vec_len));
    return 0;
}

// The list is now the one built at the current positions: they become the reference, and the skin flag clears.
static int nl_commit(LoupHandle* h, const int* go = nullptr) {
    loup::nl_commit_kernel<<<(h->N * 3 + 255) / 256, 256, 0, h->stream>>>(h->N * 3, h->d_pos64, h->d_ref_pos64,
                                                                         h->d_need_rebuild, go);
    return 0;
}

// Host decisions after nl_count (edge capacity, the graphs' edge bucket), then fill, sort and commit.
static int nl_finish(LoupHandle* h, const NlGeom& g, int E_new, int max_deg, bool check_order) {
    h->E_skin = E_new;
    h->max_deg_last = max_deg;
    if (E_new > 0) {
        if (ensure_edge_capacity(h, E_new) != 0) return -1;
        if (E_new > h->E_graph || h->E_graph > E_new + 32768) {
            h->E_graph = std::min(h->E_cap, ((E_new + 8191) & ~8191));
            if (h->E_graph < E_new) h->E_graph = E_new;
            h->invalidate_graphs();
        }
        if (nl_fill(h, g, pow2_at_least(max_deg)) != 0) return -1;
        if (check_order) {
            int vals[3];
            if (read_device_ints(h, h->d_order_err, nullptr, nullptr, vals) != 0) return -1;
            if (vals[0]) return order_error();
        }
    } else if (h->E_graph != 0) {
        h->E_graph = 0;
        h->invalidate_graphs();
    }
    if (nl_commit(h) != 0) return -1;
    CUDA_CHECK(cudaGetLastError());
    h->nl_valid = true;
    return 0;
}

// Whether rebuilds reorder the atoms spatially (see rebuild_neighbors): a host round trip, so such systems keep the
// host-driven rebuild in MD.
static bool reorder_atoms(const LoupHandle* h) {
    return h->ovr.reorder >= 0 ? h->ovr.reorder == 1 : (int64_t)h->N * 4096 > (int64_t)h->l2_bytes;
}

// check_order = false (run_md_loop) leaves the sort's contract flag in d_order_err for the caller's next read.
static int rebuild_neighbors(LoupHandle* h, bool check_order = true) {
    h->nl_valid = false;  // set again only when the rebuild completes
    const int N = h->N;
    const int S = h->n_species();
    const NlGeom g = nl_geom(h);
    if (ensure_cub_tmp(h, nl_cub_bytes(h, g)) != 0) return -1;
    if (nl_bin(h, g) != 0) return -1;

    // Reorder atoms inside each species group by the Morton code of their bin, once the per-atom sender working set
    // (x0/x1 rows and their gradients, ~4 KB per atom) outgrows L2: then sender rows and sender-gradient atomics stay
    // in L2 only if neighbours in space are neighbours in memory (a shuffled 40,000-atom input is ~30 % slower
    // otherwise). Below that size the caller's order costs nothing and the reorder's extra sync is pure overhead.
    // Atoms near a bin boundary flip bins thermally, so the order is refreshed only on every 16th rebuild (diffusion
    // is slow on that scale). Per-atom state (positions, velocities, forces, masses) is gathered in place, so every
    // buffer address and every captured graph stays valid; only the host permutation changes.
    if (reorder_atoms(h) && (h->n_rebuilds++ % 16) == 0) {
        unsigned long long* keys_in = h->d_ro_keys;
        unsigned long long* keys_out = h->d_ro_keys + N;
        int* vals_in = h->d_ro_vals;
        int* vals_out = h->d_ro_vals + N;
        loup::atom_order_keys_kernel<<<g.gridN, kNlBlk, 0, h->stream>>>(
            N, g.nx, g.ny, g.nz, h->d_species, h->d_atom_bin, keys_in, vals_in);
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
            // d_rnd_vel: a rebuild in a Langevin step falls between the half-kicks, and the second one
            // must reuse the noise the first drew for the same atom.
            if (gather(h->d_pos64, 3) || gather(h->d_vel64, 3) || gather(h->d_forces_f64, 3) ||
                gather(h->d_masses, 1) || gather(h->d_inv_masses, 1) || gather(h->d_sqrt_inv_masses, 1) ||
                gather(h->d_rnd_vel, 3)) return -1;
            std::vector<int> old_perm = h->perm;
            for (int i = 0; i < N; ++i) h->perm[i] = old_perm[h->h_order[i]];
            ++h->perm_epoch;
            if (nl_bin(h, g) != 0) return -1;
        }
    }

    if (nl_count(h, g) != 0) return -1;
    int vals[3];
    if (read_device_ints(h, h->d_row_ptr + N, h->d_max_deg, nullptr, vals) != 0) return -1;
    return nl_finish(h, g, vals[0], vals[1], check_order);
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

// Kernel variants whose ranking depends on the system, timed once per system at the first MD rebuild (thermalised
// positions) on the last force pass's inputs: the generic layers' conv forward variants
// (GenericModel::tune_conv_variants). The time spent is reported as LoupMDStats::tune_ms and excluded from step_ms.
static void tune_rebuild_variants(LoupHandle* h) {
    if (h->variants_tuned) return;
    // No pair inside r_max + skin: the active-edge arrays were never written. Time later, on a list with edges.
    if (h->E_graph == 0) return;
    const int r = h->gen->tune_conv_variants(h->stream);
    if (r < 0) return;  // no force pass recorded for this system yet: time at a later rebuild
    h->variants_tuned = true;
    if (r > 0) h->invalidate_graphs();
}

static void record_force_pass(LoupHandle* h, int compute_stress) {
    const int N = h->N;
    const int S = h->n_species();
    const float r_max = h->cutoff();

    CUDA_CHECK_THROW(cudaMemsetAsync(h->d_zero_slab, 0, h->zero_slab_bytes, h->stream));
    // With no pair within r_max + skin the pass still runs, on the empty edge list: a model can have an energy per
    // isolated atom (residual first block, readout biases, the long-range term: MACE-MH-1 0.56 eV and MACE-POLAR-1-S
    // 0.64 eV).

    SplineGrid sg{};
    int sg_psym = 1;
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

    // 1. Compute edge vectors, compact active edges (r < r_max) with their spline locator
    const int warpGridN = (N + 3) / 4;  // 4 warps per 128-thread block (1 warp per atom)
    loup::compute_and_count_active_kernel<<<warpGridN, 128, 0, h->stream>>>(
        N, r_max, h->d_pos64, h->d_row_ptr, h->d_senders, h->d_shift_cart,
        h->d_tmp_vec_len, h->d_act_deg);
    size_t act_scan_bytes = h->cub_tmp_bytes;
    cub::DeviceScan::ExclusiveSum(
        h->d_cub_tmp, act_scan_bytes, h->d_act_deg, h->d_act_row_ptr, N + 1, h->stream);
    loup::scatter_active_edges_kernel<<<warpGridN, 128, 0, h->stream>>>(
        N, r_max, sg, S, sg_psym, h->d_species, h->d_row_ptr, h->d_act_row_ptr, h->d_senders,
        h->d_tmp_vec_len, h->d_act_senders, h->d_act_receivers, h->d_act_vec_len,
        h->d_act_slab, h->d_act_t_sx);

    const int* d_E_act = h->d_act_row_ptr + N;
    loup::GenericPassInputs in{};
    in.stream = h->stream;
    in.N = N;
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
    in.pos64 = h->d_pos64;
    in.gemm = [h](bool transB, int M, int Nc, int K, float alpha, const float* A, int lda, const float* B, int ldb,
                  float beta, float* C, int ldc) {
        sgemm_rm(h, false, transB, M, Nc, K, alpha, A, lda, B, ldb, beta, C, ldc);
    };
    in.gemm_fork = [h] { gemm_fork(h); };
    in.gemm_lane = [h](int lane) { gemm_lane(h, lane); };
    in.gemm_join = [h] { gemm_join(h); };
    if (h->gate_fuse)
        in.gemm_gate_bwd = [h](int l, int M, int Nc, int K, const float* A, const float* B,
                               const loup::GateArgs& g) { return gemm_gate(h, false, l, M, Nc, K, A, B, g); };
    if (h->gate_fuse_fwd)
        in.gemm_gate_fwd = [h](int l, int M, int Nc, int K, const float* A, const float* B,
                               const loup::GateArgs& g) { return gemm_gate(h, true, l, M, Nc, K, A, B, g); };
    h->gen->record(in);
}

static int ensure_force_graph(LoupHandle* h, int compute_stress) {
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

// Every exported entry point runs through this: a C++ exception (CUDA/cuBLAS failure inside record_force_pass,
// sgemm_rm or HybCubin::launch) must not unwind into the C caller, and a throw during capture must not leave the
// stream in capture mode.
template <class F>
static int guarded(LoupHandle* h, const char* name, F&& f) {
    try {
        return f();
    } catch (const std::exception& e) {
        fprintf(stderr, "%s failed: %s\n", name, e.what());
    } catch (...) {
        fprintf(stderr, "%s failed: unknown exception\n", name);
    }
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

extern "C" {

LoupHandle* loup_create(const char* loup_path, int device_id) {
    LoupHandle* h = nullptr;
    try {
        CUDA_CHECK_THROW(cudaSetDevice(device_id));
        h = new LoupHandle();
        h->device_id = device_id;
        CUDA_CHECK_THROW(cudaDeviceGetAttribute(&h->l2_bytes, cudaDevAttrL2CacheSize, device_id));
        bool own_ext_env = false, cc80 = false;
        {
            int cc_maj = 0, cc_min = 0;
            CUDA_CHECK_THROW(cudaDeviceGetAttribute(&cc_maj, cudaDevAttrComputeCapabilityMajor, device_id));
            CUDA_CHECK_THROW(cudaDeviceGetAttribute(&cc_min, cudaDevAttrComputeCapabilityMinor, device_id));
            h->is_sm120 = (cc_maj == 12 && cc_min == 0);
            cc80 = cc_maj == 8 && cc_min == 0;
            h->own_ext = h->is_sm120;
            if (const char* e = std::getenv("LOUP_OWN_EXT"); e && (e[0] == '0' || e[0] == '1')) {
                h->own_ext = e[0] == '1';
                own_ext_env = true;
            }
            h->nb_groups_ok = h->is_sm120 || cc80;
            if (const char* e = std::getenv("LOUP_NB_GROUPS"); e && (e[0] == '0' || e[0] == '1')) h->nb_groups_ok = e[0] == '1';
            if (const char* e = std::getenv("LOUP_FORCE_TC"); e && e[0] == '1') h->force_tc = true;
            if (const char* e = std::getenv("LOUP_GATE_FUSE"); e && e[0] == '0') h->gate_fuse = false;
            if (const char* e = std::getenv("LOUP_GATE_FUSE_FWD"); e && e[0] == '0') h->gate_fuse_fwd = false;
        }
        CUDA_CHECK_THROW(cudaStreamCreate(&h->stream));
        CUBLAS_CHECK_THROW(cublasCreate(&h->cublas));
        CUBLAS_CHECK_THROW(cublasLtCreate(&h->cublas_lt));
        CUDA_CHECK_THROW(cudaMalloc(&h->d_cublas_ws, h->cublas_ws_bytes));
        cublas_bind(h, h->stream, h->d_cublas_ws);
        // Mapped pinned words for polling stream positions (publish_kernel, guard_step_kernel).
        CUDA_CHECK_THROW(cudaHostAlloc(&h->h_pub, 16 * sizeof(int), cudaHostAllocMapped));
        memset(h->h_pub, 0, 16 * sizeof(int));
        CUDA_CHECK_THROW(cudaHostGetDevicePointer(reinterpret_cast<void**>(&h->d_pub), h->h_pub, 0));
        CUDA_CHECK_THROW(cudaMalloc(&h->d_pub_counter, sizeof(unsigned int)));
        CUDA_CHECK_THROW(cudaMemset(h->d_pub_counter, 0, sizeof(unsigned int)));
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&h->ev_fork, cudaEventDisableTiming));
        for (int i = 0; i < LoupHandle::kSide; ++i) {
            CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&h->side[i], cudaStreamNonBlocking));
            CUDA_CHECK_THROW(cudaEventCreateWithFlags(&h->ev_join[i], cudaEventDisableTiming));
            CUDA_CHECK_THROW(cudaMalloc(&h->side_ws[i], h->cublas_ws_bytes));
        }
        {
            const loup::Bundle bundle(loup_path);
            if (bundle.meta("arch") != "generic")  // e.g. omat0_medium_spec: the specialised path, retired (stage 5)
                throw std::runtime_error(std::string(loup_path) + ": bundle arch '" + bundle.meta("arch") +
                                         "' is no longer supported; re-export it (every model exports as arch=generic)");
            h->gen = std::make_unique<loup::GenericModel>();
            h->gen->load(bundle);
        }
        // Tensor-core GEMM implementations have failed full-model POLAR accuracy gates. Their rounding depends on
        // the operands and MMA accumulation, not a fixed multiple of FP32 error. Keep long-range models on FP32
        // until a candidate clears both accuracy and performance qualification (see gemm-tc-rounded-polar in the ledger).
        // LOUP_LR_HYBRID=1 (tests): load the GEMM cubin for long-range models too (e.g. a true 3xTF32 build through
        // LOUP_HYBRID_CUBIN, gated before any adoption).
        const char* lr_hyb = std::getenv("LOUP_LR_HYBRID");
        if (!h->gen->has_lr() || (lr_hyb && lr_hyb[0] == '1')) h->hyb.init(device_id);
        if (cc80 && !h->hyb.loaded && !own_ext_env) h->own_ext = true;  // float32 SIMT GEMMs only (see own_ext)
        h->load_disk_plans(device_id);
        loup::sgemm128_init();
        return h;
    } catch (const std::exception& e) {
        fprintf(stderr, "loup_create failed: %s\n", e.what());
        if (h) loup_destroy(h);
        return nullptr;
    }
}

void loup_destroy(LoupHandle* h) {
    if (!h) return;
    cudaSetDevice(h->device_id);
    h->free_system();
    if (h->ev_fork) cudaEventDestroy(h->ev_fork);
    if (h->ev_io_in) cudaEventDestroy(h->ev_io_in);
    if (h->ev_io_out) cudaEventDestroy(h->ev_io_out);
    for (int i = 0; i < LoupHandle::kSide; ++i) {
        if (h->ev_join[i]) cudaEventDestroy(h->ev_join[i]);
        if (h->side_ws[i]) cudaFree(h->side_ws[i]);
        if (h->side[i]) cudaStreamDestroy(h->side[i]);
    }
    if (h->h_pub) cudaFreeHost(h->h_pub);
    if (h->d_pub_counter) cudaFree(h->d_pub_counter);
    h->hyb.destroy();
    if (h->d_cublas_ws) cudaFree(h->d_cublas_ws);
    for (auto& kv : h->bt_cache) cudaFree(kv.second);
    if (h->cublas_lt) cublasLtDestroy(h->cublas_lt);
    if (h->cublas) cublasDestroy(h->cublas);
    if (h->stream) cudaStreamDestroy(h->stream);
    delete h;
}

static int set_system_impl(LoupHandle* h, int n_atoms, const int* atomic_numbers, const double* masses_amu, double skin) {
    if (!h || n_atoms <= 0 || !atomic_numbers || !(skin >= 0.0) || !std::isfinite(skin)) return -1;
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
            fprintf(stderr, "loup_set_system: Z=%d not in exported .loup model\n", z);
            return -1;
        }
        if (masses_amu && !(masses_amu[i] > 0.0 && std::isfinite(masses_amu[i]))) {
            fprintf(stderr, "loup_set_system: atom %d has mass %g\n", i, masses_amu[i]);
            return -1;
        }
        sp_orig[i] = found;
        e0_total += h->species_e0(found);
    }
    h->free_system();  // sets N = 0
    h->skin = skin;
    h->e0_total = e0_total;

    // Build stable species-grouping permutation (the spatial order inside each group is set at every rebuild)
    ++h->perm_epoch;
    h->perm.resize(n_atoms);
    h->h_order.resize(n_atoms);
    h->species_start.assign(S + 1, 0);
    h->species_count.assign(S, 0);
    for (int i = 0; i < n_atoms; ++i) h->species_count[sp_orig[i]]++;
    for (int s = 0; s < S; ++s) h->species_start[s + 1] = h->species_start[s] + h->species_count[s];
    std::vector<int> cursor = h->species_start;
    std::vector<int> sp_sorted(n_atoms);
    std::vector<double> m_sorted(n_atoms), inv_m_sorted(n_atoms), sim_sorted(n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        int dst = cursor[sp_orig[i]]++;
        h->perm[dst] = i;
        sp_sorted[dst] = sp_orig[i];
        double m = masses_amu ? masses_amu[i] : 1.0;
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
    CUDA_CHECK(cudaMalloc(&h->d_step_counter, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemsetAsync(h->d_step_counter, 0, sizeof(unsigned long long), h->stream));
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
    CUDA_CHECK(cudaMalloc(&h->d_max_deg, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_bin_counts, (h->max_bins + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_bin_offsets, (h->max_bins + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&h->d_bin_cursor, (h->max_bins + 1) * sizeof(int)));
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

    // energy, forces and virial (zeroed per force pass); the model keeps its own buffers
    auto align256 = [](size_t b) { return (b + 255) & ~size_t(255); };
    const size_t off_f64 = align256(sizeof(double));
    const size_t off_v64 = off_f64 + align256((size_t)n_atoms * 3 * sizeof(double));
    h->zero_slab_bytes = off_v64 + align256(6 * sizeof(double));
    CUDA_CHECK(cudaMalloc(&h->d_zero_slab, h->zero_slab_bytes));
    auto* base = static_cast<char*>(h->d_zero_slab);
    h->d_energy_f64 = reinterpret_cast<double*>(base);
    h->d_forces_f64 = reinterpret_cast<double*>(base + off_f64);
    h->d_virial_grad_f64 = reinterpret_cast<double*>(base + off_v64);
    h->gen->alloc_system(n_atoms);
    if (h->E_cap > 0) h->gen->ensure_edges(h->E_cap);
    CUDA_CHECK(cudaHostAlloc(&h->h_stage, ((size_t)6 * n_atoms + 8) * sizeof(double), cudaHostAllocDefault));
    h->N = n_atoms;
    return 0;
}

// Upload positions (caller order) and the cell. Returns 1 if the cell changed (the neighbour list's Cartesian
// shifts depend on it, so the list must be rebuilt), 0 if not, -1 on error.
// The engine is always fully periodic: the cell must be finite and non-degenerate (ASE's zero cell for a molecule
// is rejected; put molecules in a box larger than 2 (r_max + skin)), and positions finite.
static bool valid_cell(const double* cell) {
    if (!cell) return false;
    for (int i = 0; i < 9; ++i)
        if (!std::isfinite(cell[i])) return false;
    const double det = cell[0] * (cell[4] * cell[8] - cell[5] * cell[7]) - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6]) +
                       cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
    double scale = 0.0;
    for (int i = 0; i < 9; ++i) scale = std::max(scale, std::fabs(cell[i]));
    if (!(std::fabs(det) > 1e-9 * scale * scale * scale) || scale == 0.0) {
        fprintf(stderr, "loupmd: singular or zero cell (det = %g); the engine is fully periodic\n", det);
        return false;
    }
    return true;
}

static bool valid_input(int n, const double* positions, const double* cell, const double* velocities) {
    if (!positions || !valid_cell(cell)) return false;
    for (size_t i = 0; i < (size_t)n * 3; ++i) {
        if (!std::isfinite(positions[i]) || (velocities && !std::isfinite(velocities[i]))) {
            fprintf(stderr, "loupmd: non-finite position or velocity (atom %zu)\n", i / 3);
            return false;
        }
    }
    return true;
}

// Takes the cell and writes the positions, in the engine's order, to the page-locked staging. Returns 1 if the cell
// changed, 0 if not, -1 on error.
static int stage_positions(LoupHandle* h, const double* positions, const double* cell) {
    if (!valid_input(h->N, positions, cell, nullptr)) return -1;
    loup::CellMat64 old_cell = h->cell;
    loup::compute_cell_mat(cell, h->cell, h->cell_vol, h->cell_heights);
    const int cell_changed = memcmp(&old_cell, &h->cell, sizeof(old_cell)) != 0 ? 1 : 0;
    if (h->gen->set_cell(cell)) h->invalidate_graphs();

    // The staging is free here: every call that fills it has read its results back before returning.
    double* p_sorted = h->stage_pos();
    for (int dst = 0; dst < h->N; ++dst) {
        int src = h->perm[dst];
        p_sorted[dst * 3 + 0] = positions[src * 3 + 0];
        p_sorted[dst * 3 + 1] = positions[src * 3 + 1];
        p_sorted[dst * 3 + 2] = positions[src * 3 + 2];
    }
    return cell_changed;
}

static int upload_positions(LoupHandle* h, const double* positions, const double* cell) {
    const int cell_changed = stage_positions(h, positions, cell);
    if (cell_changed < 0) return -1;
    CUDA_CHECK(cudaMemcpyAsync(h->d_pos64, h->stage_pos(), (size_t)h->N * 3 * sizeof(double), cudaMemcpyHostToDevice, h->stream));
    return cell_changed;
}

// Queues (or, during capture, records) the copies of the energy, the forces and with `stress` the virial into the
// page-locked staging, and behind them a publication of the neighbour rows' order flag: once the host has read that
// publication the results have landed, with no blocking transfer and no stream synchronisation.
static int queue_results(LoupHandle* h, int stress) {
    double* ev = h->stage_ev();
    CUDA_CHECK(cudaMemcpyAsync(ev, h->d_energy_f64, sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(h->stage_f(), h->d_forces_f64, (size_t)h->N * 3 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    if (stress)
        CUDA_CHECK(cudaMemcpyAsync(ev + 1, h->d_virial_grad_f64, 6 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    loup::publish_kernel<<<1, 1, 0, h->stream>>>(h->d_order_err, nullptr, nullptr, h->d_pub_counter, h->d_pub);
    return 0;
}

static int run_md_loop(LoupHandle* h, int kind, int n_steps, double dt_fs, double temp_k, double friction_per_fs,
                       unsigned long long seed, LoupMDStats* out_stats);

static int set_state_impl(LoupHandle* h, const double* positions, const double* velocities, const double* cell) {
    if (!h || h->N <= 0) return -1;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    if (!valid_input(h->N, positions, cell, velocities)) return -1;
    if (upload_positions(h, positions, cell) < 0) return -1;
    if (velocities) {
        std::vector<double> v_sorted((size_t)h->N * 3);
        for (int dst = 0; dst < h->N; ++dst) {
            int src = h->perm[dst];
            v_sorted[dst * 3 + 0] = velocities[src * 3 + 0];
            v_sorted[dst * 3 + 1] = velocities[src * 3 + 1];
            v_sorted[dst * 3 + 2] = velocities[src * 3 + 2];
        }
        CUDA_CHECK(cudaMemcpyAsync(h->d_vel64, v_sorted.data(), (size_t)h->N * 3 * sizeof(double), cudaMemcpyHostToDevice, h->stream));
    } else {
        CUDA_CHECK(cudaMemsetAsync(h->d_vel64, 0, (size_t)h->N * 3 * sizeof(double), h->stream));
    }
    return rebuild_neighbors(h);
}

// The TF32 + FP16 hybrid GEMMs (gemm-hybrid-cubin) overflow to inf when a GEMM operand element reaches 32,752. A
// non-finite energy, force or stress with the hybrid loaded drops it for this engine: later GEMM plans are re-tuned
// without it (disk plans that chose it are skipped while it is unloaded) and the captured graphs are re-recorded.
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
    h->disk_dirty = false;
    h->plan_cache_path.clear();
    h->invalidate_graphs();
    return true;
}

static int evaluate_once(
    LoupHandle* h, const double* positions, const double* cell, int compute_stress,
    double* out_energy_ev, double* out_forces, double* out_stress, bool* finite
) {
    if (!h || h->N <= 0) return -1;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    // Positions only: MD velocities are kept. The neighbour list is rebuilt only when the cell changed or an atom
    // moved more than skin/2 since the last build; the force graph is reused unless the rebuild outgrew it.
    const int cell_changed = stage_positions(h, positions, cell);
    if (cell_changed < 0) return -1;
    const int stress = compute_stress && out_stress ? 1 : 0;
    const double* f_sorted = h->stage_f();
    double* ev = h->stage_ev();
    for (int i = 0; i < 7; ++i) ev[i] = 0.0;
    if (cell_changed != 0 || !h->nl_valid) {
        // A new cell (the list's Cartesian shifts depend on it) or no list yet: the host-driven rebuild, then the pass.
        CUDA_CHECK(cudaMemcpyAsync(h->d_pos64, h->stage_pos(), (size_t)h->N * 3 * sizeof(double), cudaMemcpyHostToDevice, h->stream));
        if (rebuild_neighbors(h) != 0) return -1;
        if (ensure_force_graph(h, stress) != 0) return -1;
        CUDA_CHECK(cudaGraphLaunch(h->force_graph_exec, h->stream));
        int fin[3];
        if (queue_results(h, stress) != 0 || wait_published(h, ++h->pub_count, fin) != 0) return -1;
    } else if (run_md_loop(h, 3, 1, 1.0, 0.0, 0.0, (unsigned long long)stress, nullptr) != 0) {
        // One launch of the host step graph: upload, skin check, guard, force pass or on-device rebuild, downloads.
        return -1;
    }
    const double inter_e = ev[0];
    const double* v_grad = ev + 1;

    *finite = std::isfinite(inter_e);
    for (size_t i = 0; *finite && i < (size_t)h->N * 3; ++i) *finite = std::isfinite(f_sorted[i]);
    for (int i = 0; *finite && i < 6; ++i) *finite = std::isfinite(v_grad[i]);
    if (out_energy_ev) *out_energy_ev = h->e0_total + inter_e;
    if (out_forces) {
        for (int dst = 0; dst < h->N; ++dst) {
            int orig = h->perm[dst];
            out_forces[orig * 3 + 0] = f_sorted[dst * 3 + 0];
            out_forces[orig * 3 + 1] = f_sorted[dst * 3 + 1];
            out_forces[orig * 3 + 2] = f_sorted[dst * 3 + 2];
        }
    }
    if (stress) {
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
        fprintf(stderr, "loupmd: non-finite energy / forces (model output; the configuration may be unphysical)\n");
        return -1;
    }
    return rc;
}

static int run_md_loop(LoupHandle* h, int kind, int n_steps, double dt_fs, double temp_k, double friction_per_fs,
                       unsigned long long seed, LoupMDStats* out_stats) {
    // nl_valid: positions and a neighbour list exist (set_state or evaluate ran and its rebuild completed).
    if (!h || h->N <= 0 || !h->nl_valid || n_steps <= 0 || !std::isfinite(dt_fs) || !(dt_fs > 0.0)) return -1;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    // kind 2, the external step of loup_compute_device: the caller integrates and has written the positions, so one
    // step is the skin check and the force pass (with the virial when `seed` is 1), with the same guard, on-device
    // rebuild and host fallback as an MD step. It queues the step and returns: nothing below waits for the forces.
    // kind 3, the host step of loup_evaluate: the same, with the positions' upload from the page-locked staging at the
    // head of the step graph and the results' copies and their publication (queue_results) at its end, so a call is
    // one graph launch. The host reads the guard's publication and then that last one.
    const bool ext = kind == 2 || kind == 3;
    const bool host_io = kind == 3;
    const int ext_stress = ext && seed ? 1 : 0;
    const int graph_key = ext ? 2 * kind + ext_stress : -1;
    const int N = h->N;
    const int gridN = (N + 255) / 256;
    const double dt_ase = dt_fs * loup::ASE_FS;
    const double half_skin = 0.5 * h->skin;
    // One loop. Each step is one launch of the step graph: first half + skin check + guard_step_kernel + an IF node
    // whose body is the rest (force pass + second half). The guard publishes the check's flag to mapped host memory
    // and lets the force pass run only while the list is complete, i.e. no atom has moved more than skin/2 since the
    // last rebuild (exact for any skin, including 0). The host polls the flag, which lands before the force pass
    // starts, and queues the next step while the force pass runs, so the GPU waits for the host only at a rebuild:
    // the guard has skipped the force pass, the host rebuilds the list at the new positions and launches the checked
    // rest graph (rest_guard_kernel + IF (valid rows) rest).

    // Initial force evaluation at step 0
    if (!ext) {
        if (ensure_force_graph(h, 0) != 0) return -1;
        CUDA_CHECK(cudaGraphLaunch(h->force_graph_exec, h->stream));
        // The Langevin step counter persists across run calls (reset only in loup_set_system), so chunked runs with
        // one seed continue the noise sequence instead of replaying it.
        CUDA_CHECK(cudaStreamSynchronize(h->stream));
    }

    double c1 = 0, c2 = 0, k_c3 = 0, k_c4 = 0, k_c5 = 0;
    if (kind == 1) {
        const double fr = friction_per_fs / loup::ASE_FS;  // 1/fs -> 1/(ASE time unit), as ASE's friction=f/units.fs
        c1 = dt_ase / 2.0 - dt_ase * dt_ase * fr / 8.0;
        c2 = dt_ase * fr / 2.0 - dt_ase * dt_ase * fr * fr / 8.0;
        double sigma_k = sqrt(2.0 * fr * loup::ASE_KB * temp_k);
        k_c3 = sqrt(dt_ase) * sigma_k / 2.0 - pow(dt_ase, 1.5) * fr * sigma_k / 8.0;
        k_c5 = pow(dt_ase, 1.5) * sigma_k / (2.0 * sqrt(3.0));
        k_c4 = (fr / 2.0) * k_c5;
    }

    auto record_move = [&] {
        if (kind == 0) {
            loup::nve_first_half_kernel<<<gridN, 256, 0, h->stream>>>(
                N, dt_ase, h->d_forces_f64, h->d_inv_masses, h->d_pos64, h->d_vel64);
        } else if (kind == 1) {
            // 32 atoms per block: the FP64 Box-Muller (3 curand_normal2_double per atom) is FP64-throughput bound on
            // consumer GPUs, so spread the atoms over every SM instead of N / 256 of them; S106: one thread per draw.
            loup::langevin_first_half_kernel<<<(N + 31) / 32, 96, 0, h->stream>>>(
                N, dt_ase, c1, c2, k_c3, k_c4, k_c5, seed, h->d_step_counter,
                h->d_forces_f64, h->d_inv_masses, h->d_sqrt_inv_masses,
                h->d_pos64, h->d_vel64, h->d_rnd_vel);
        }
        if (host_io)
            CUDA_CHECK_THROW(cudaMemcpyAsync(h->d_pos64, h->stage_pos(), (size_t)N * 3 * sizeof(double),
                                             cudaMemcpyHostToDevice, h->stream));
        loup::check_skin_kernel<<<gridN, 256, 0, h->stream>>>(N, half_skin, h->d_pos64, h->d_ref_pos64,
                                                              h->d_need_rebuild);
    };
    auto record_rest = [&]() {
        record_force_pass(h, ext_stress);
        if (kind == 0) {
            loup::nve_second_half_kernel<<<gridN, 256, 0, h->stream>>>(
                N, dt_ase, h->d_forces_f64, h->d_inv_masses, h->d_vel64);
        } else if (kind == 1) {
            loup::langevin_second_half_kernel<<<gridN, 256, 0, h->stream>>>(
                N, c1, c2, h->d_forces_f64, h->d_inv_masses, h->d_rnd_vel, h->d_vel64, h->d_step_counter);
        }
    };
    // Captures into graph g: prologue(conds) with one new conditional handle per body, then one IF node per handle
    // after the prologue, then each body into its IF node's graph (a body may nest this call). Each level's capture
    // ends before its bodies are captured, as the force pass records on h->stream.
    using Prologue = std::function<void(const cudaGraphConditionalHandle*)>;
    using Body = std::function<int(cudaGraph_t)>;
    std::function<int(cudaGraph_t, const Prologue&, const std::vector<Body>&)> capture_ifs =
        [&](cudaGraph_t g, const Prologue& prologue, const std::vector<Body>& bodies) -> int {
        const size_t n = bodies.size();
        std::vector<cudaGraphConditionalHandle> conds(n);
        for (size_t i = 0; i < n; ++i) CUDA_CHECK(cudaGraphConditionalHandleCreate(&conds[i], g, 0, 0));
        CUDA_CHECK(cudaStreamBeginCaptureToGraph(h->stream, g, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
        prologue(conds.data());
        cudaStreamCaptureStatus status;
        const cudaGraphNode_t* deps_p = nullptr;
        size_t n_deps = 0;
        CUDA_CHECK(cudaStreamGetCaptureInfo(h->stream, &status, nullptr, nullptr, &deps_p, nullptr, &n_deps));
        const std::vector<cudaGraphNode_t> deps(deps_p, deps_p + n_deps);
        std::vector<cudaGraphNode_t> nodes(n);
        std::vector<cudaGraph_t> body_graphs(n);
        for (size_t i = 0; i < n; ++i) {
            cudaGraphNodeParams cp = {};
            cp.type = cudaGraphNodeTypeConditional;
            cp.conditional.handle = conds[i];
            cp.conditional.type = cudaGraphCondTypeIf;
            cp.conditional.size = 1;
            CUDA_CHECK(cudaGraphAddNode(&nodes[i], g, deps.data(), nullptr, deps.size(), &cp));
            body_graphs[i] = cp.conditional.phGraph_out[0];
        }
        CUDA_CHECK(cudaStreamUpdateCaptureDependencies(h->stream, nodes.data(), nullptr, n,
                                                       cudaStreamSetCaptureDependencies));
        cudaGraph_t captured = nullptr;
        CUDA_CHECK(cudaStreamEndCapture(h->stream, &captured));
        for (size_t i = 0; i < n; ++i)
            if (bodies[i](body_graphs[i]) != 0) return -1;
        return 0;
    };
    const Body rest_body = [&](cudaGraph_t g) -> int {
        CUDA_CHECK(cudaStreamBeginCaptureToGraph(h->stream, g, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
        record_rest();
        cudaGraph_t captured = nullptr;
        CUDA_CHECK(cudaStreamEndCapture(h->stream, &captured));
        return 0;
    };
    // The checked rest: rest_guard_kernel + IF (valid rows) rest.
    const Body checked_rest_body = [&](cudaGraph_t g) -> int {
        return capture_ifs(g, [&](const cudaGraphConditionalHandle* c) {
            loup::rest_guard_kernel<<<1, 1, 0, h->stream>>>(c[0], h->d_order_err, nullptr);
        }, {rest_body});
    };
    auto instantiate = [&](cudaGraph_t& g, cudaGraphExec_t& ge, const Body& body) -> int {
        if (ge) { cudaGraphExecDestroy(ge); ge = nullptr; }
        if (g) { cudaGraphDestroy(g); g = nullptr; }
        CUDA_CHECK(cudaGraphCreate(&g, 0));
        if (body(g) != 0) return -1;
        CUDA_CHECK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        return 0;
    };
    // The step graph's rebuild branch needs fixed sizes: the edge bucket and a row-sort width with a quarter of headroom
    // over the last rebuild's longest row. Systems whose rebuilds reorder the atoms (a host round trip) or sort rows in
    // global scratch keep the host-driven rebuild, as does a system's first rebuild (it times kernel variants).
    const NlGeom nl_g = nl_geom(h);
    auto rebuild_width = [&] { return pow2_at_least(std::max(32, h->max_deg_last + h->max_deg_last / 4 + 1)); };
    auto device_rebuild_ok = [&] {
        return h->ovr.md_rebuild != 1 && h->variants_tuned && h->E_graph > 0 && !reorder_atoms(h) &&
               !h->ovr.row_sort_global && rebuild_width() <= 4096;
    };
    auto ensure_graphs = [&]() -> int {
        if (h->md_step_exec && h->md_checked_exec) return 0;
        const bool dev = device_rebuild_ok();
        if (dev && ensure_cub_tmp(h, nl_cub_bytes(h, nl_g)) != 0) return -1;
        const int E_lim = h->ovr.md_rebuild == 2 ? 0 : h->E_graph;  // fallback (tests): never fits
        const int P_cap = rebuild_width();
        // The rebuild branch: count, gate (its verdict in go = d_max_deg + 1), then fill, sort and commit, which skip
        // themselves on a negative verdict, and the checked rest, also gated by the verdict.
        const int* go = h->d_max_deg + 1;
        const Body rebuild_body = [&](cudaGraph_t g) -> int {
            return capture_ifs(g, [&](const cudaGraphConditionalHandle* c) {
                nl_bin(h, nl_g);
                nl_count(h, nl_g);
                loup::rebuild_gate_kernel<<<1, 1, 0, h->stream>>>(h->d_row_ptr + N, h->d_max_deg, E_lim, P_cap,
                                                                   h->d_max_deg + 1, h->d_pub_counter, h->d_pub);
                nl_fill(h, nl_g, P_cap, go);
                nl_commit(h, go);
                loup::rest_guard_kernel<<<1, 1, 0, h->stream>>>(c[0], h->d_order_err, go);
            }, {rest_body});
        };
        // The step graph: move + check + guard_step_kernel, then one SWITCH node (body 0 the rest, body 1 the rebuild
        // branch when dev).
        const Body step_body = [&](cudaGraph_t g) -> int {
            cudaGraphConditionalHandle c;
            CUDA_CHECK(cudaGraphConditionalHandleCreate(&c, g, 0, 0));
            CUDA_CHECK(cudaStreamBeginCaptureToGraph(h->stream, g, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
            record_move();
            loup::guard_step_kernel<<<1, 1, 0, h->stream>>>(c, dev ? 1 : 0, h->d_need_rebuild, h->d_order_err,
                                                             h->d_pub_counter, h->d_pub);
            cudaStreamCaptureStatus status;
            const cudaGraphNode_t* deps_p = nullptr;
            size_t n_deps = 0;
            CUDA_CHECK(cudaStreamGetCaptureInfo(h->stream, &status, nullptr, nullptr, &deps_p, nullptr, &n_deps));
            const std::vector<cudaGraphNode_t> deps(deps_p, deps_p + n_deps);
            cudaGraphNodeParams cp = {};
            cp.type = cudaGraphNodeTypeConditional;
            cp.conditional.handle = c;
            cp.conditional.type = cudaGraphCondTypeSwitch;
            cp.conditional.size = dev ? 2 : 1;
            cudaGraphNode_t node;
            CUDA_CHECK(cudaGraphAddNode(&node, g, deps.data(), nullptr, deps.size(), &cp));
            CUDA_CHECK(cudaStreamUpdateCaptureDependencies(h->stream, &node, nullptr, 1,
                                                           cudaStreamSetCaptureDependencies));
            // After the SWITCH node, whichever body ran (or none): the results' copies and their publication.
            if (host_io && queue_results(h, ext_stress) != 0) return -1;
            cudaGraph_t captured = nullptr;
            CUDA_CHECK(cudaStreamEndCapture(h->stream, &captured));
            if (rest_body(cp.conditional.phGraph_out[0]) != 0) return -1;
            if (dev && rebuild_body(cp.conditional.phGraph_out[1]) != 0) return -1;
            return 0;
        };
        if (instantiate(h->md_step_graph, h->md_step_exec, step_body) != 0) return -1;
        if (instantiate(h->md_checked_graph, h->md_checked_exec, checked_rest_body) != 0) return -1;
        h->md_dev_rebuild = dev;
        h->md_graph_key = ext ? graph_key : -2;
        if (!ext) {
            h->md_int_kind = kind;
            h->md_int_dt = dt_fs;
            h->md_int_temp = temp_k;
            h->md_int_fric = friction_per_fs;
            h->md_int_seed = seed;
        }
        return 0;
    };
    // The graphs bake in this call's dt, thermostat and seed: drop those of a call with other values (the Langevin
    // noise continues through the device step counter, so equal values give the same graphs). The external steps
    // bake in only whether the virial is formed, so their graphs are kept from call to call.
    const bool same_run = !ext && h->md_graph_key == -2 && h->md_int_kind == kind && h->md_int_dt == dt_fs &&
                          h->md_int_temp == temp_k && h->md_int_fric == friction_per_fs && h->md_int_seed == seed;
    if (!(ext && h->md_graph_key == graph_key) && !same_run) h->invalidate_md_graphs();

    int rebuilds = 0;
    double tune_ms = 0.0;
    // One step: launch the step graph, read the guard's publication, and on a rebuild that the graph could not finish
    // on the GPU, finish it on the host and launch the checked rest.
    auto one_step = [&]() -> int {
        if (ensure_graphs() != 0) return -1;
        CUDA_CHECK(cudaGraphLaunch(h->md_step_exec, h->stream));
        int v[3];
        if (wait_published(h, ++h->pub_count, v) != 0) return -1;
        if (v[1]) return order_error();
        // host_io: the step graph ends with the results' copies and their publication `fin` (the rows' order flag),
        // whichever branch ran; the staging holds this step's results once it has landed, if the force pass ran.
        int fin[3] = {0, 0, 0};
        if (!v[0]) {
            if (host_io && wait_published(h, ++h->pub_count, fin) != 0) return -1;
            return fin[0] ? order_error() : 0;
        }
        ++rebuilds;
        if (h->md_dev_rebuild) {
            // The step graph's rebuild branch has counted the new rows; it went on to fill them and run the checked
            // rest on the GPU if they fit its captured sizes.
            int w[3];
            if (wait_published(h, ++h->pub_count, w) != 0) return -1;
            h->E_skin = w[0];
            h->max_deg_last = w[1];
            if (host_io && wait_published(h, ++h->pub_count, fin) != 0) return -1;
            if (w[2]) return fin[0] ? order_error() : 0;
            // They did not: once the step graph has drained, finish the rebuild on the host from the counted rows and
            // recapture the graphs for the new sizes.
            int z[3];
            if (read_device_ints(h, nullptr, nullptr, nullptr, z) != 0) return -1;
            h->nl_valid = false;
            if (nl_finish(h, nl_g, w[0], w[1], false) != 0) return -1;
            h->invalidate_md_graphs();
        } else {
            if (host_io && wait_published(h, ++h->pub_count, fin) != 0) return -1;  // of the pass the guard skipped
            // The per-system variant timing (first rebuild only) is not part of the steps: it is timed and excluded.
            if (!h->variants_tuned) {
                const auto t0 = std::chrono::steady_clock::now();
                tune_rebuild_variants(h);
                tune_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            }
            // The row-order flag is read with the next step's publication (or at the end of the block); until then
            // the checked rest's guard keeps the force pass off rows that failed it.
            if (rebuild_neighbors(h, false) != 0) return -1;
            if (device_rebuild_ok()) h->invalidate_md_graphs();  // the step graph gains its rebuild branch
        }
        if (ensure_graphs() != 0) return -1;
        CUDA_CHECK(cudaGraphLaunch(h->md_checked_exec, h->stream));
        if (host_io) {
            if (queue_results(h, ext_stress) != 0 || wait_published(h, ++h->pub_count, fin) != 0) return -1;
            if (fin[0]) return order_error();
        }
        return 0;
    };
    if (ext) return one_step();

    EventPairGuard evg;
    cudaEvent_t ev0 = evg.e0, ev1 = evg.e1;
    CUDA_CHECK(cudaEventRecord(ev0, h->stream));
    for (int step = 0; step < n_steps; ++step)
        if (one_step() != 0) return -1;
    CUDA_CHECK(cudaEventRecord(ev1, h->stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    // The block's last energy (forces feed it every step): non-finite means the run is lost; with the hybrid GEMMs
    // loaded they are dropped first, so a run restarted from a saved state does not hit the same overflow.
    double inter_e = 0.0;
    int order_err = 0;  // a rebuild in the block's last step is not followed by a flag read
    CUDA_CHECK(cudaMemcpyAsync(&inter_e, h->d_energy_f64, sizeof(double), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(&order_err, h->d_order_err, sizeof(int), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    if (order_err) return order_error();
    if (!std::isfinite(inter_e)) {
        drop_hybrid(h);
        fprintf(stderr, "loupmd: non-finite potential energy after the MD block: the state is lost; restart from a "
                        "saved state\n");
        return -1;
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
        out_stats->final_temperature_k = (2.0 * ke) / (3.0 * N * loup::ASE_KB);
        out_stats->tune_ms = tune_ms;
    }
    return 0;
}

static int run_nve_impl(LoupHandle* h, int n_steps, double dt_fs, LoupMDStats* out_stats) {
    return run_md_loop(h, 0, n_steps, dt_fs, 0.0, 0.0, 0, out_stats);
}

// loup_compute_device: the caller's rows (row r holds engine atom ids[r], or r without ids) to the engine's atom
// order and back. An id outside 0 .. n-1 sets bit 0 of *bad, a non-finite force bit 1.
static __global__ void io_gather_kernel(int n, const double* __restrict__ pos, long long rs, long long cs,
                                        const int* __restrict__ ids, const int* __restrict__ inv_perm,
                                        double* __restrict__ pos_out, int* bad) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const int a = ids ? ids[r] : r;
    if ((unsigned)a >= (unsigned)n) {
        atomicOr(bad, 1);
        return;
    }
    const int d = inv_perm[a];
    const double* p = pos + (long long)r * rs;
    pos_out[3 * d + 0] = p[0];
    pos_out[3 * d + 1] = p[cs];
    pos_out[3 * d + 2] = p[2 * cs];
}

static __global__ void io_scatter_kernel(int n, const int* __restrict__ ids, const int* __restrict__ inv_perm,
                                         const double* __restrict__ f, double* __restrict__ out, long long rs,
                                         long long cs, int* bad) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const int a = ids ? ids[r] : r;
    if ((unsigned)a >= (unsigned)n) return;
    const int d = inv_perm[a];
    const double fx = f[3 * d + 0], fy = f[3 * d + 1], fz = f[3 * d + 2];
    if (!(isfinite(fx) && isfinite(fy) && isfinite(fz))) atomicOr(bad, 2);
    double* o = out + (long long)r * rs;
    o[0] += fx;
    o[cs] += fy;
    o[2 * cs] += fz;
}

// One force evaluation for a host code that keeps its atoms on the GPU and integrates itself (LAMMPS Kokkos,
// LAMMPS_KOKKOS_PLAN.md stage B). Everything is queued behind the work already on the caller's stream, and the caller's
// stream is made to wait for the forces; the host waits only for the step graph's guard (which lands before the force
// pass starts), for a neighbour rebuild the graph could not finish on the GPU, and for the energy / virial when asked.
static int compute_device_impl(LoupHandle* h, const LoupDeviceIO* io, const double* cell, void* stream,
                               double* out_energy_ev, double* out_virial) {
    if (!h || h->N <= 0 || !io) return -1;
    CUDA_CHECK(cudaSetDevice(h->device_id));
    const int N = h->N;
    if (io->n_rows != N || !io->positions || io->pos_row_stride <= 0 || io->pos_comp_stride <= 0 ||
        (io->forces && (io->f_row_stride <= 0 || io->f_comp_stride <= 0))) {
        fprintf(stderr, "loup_compute_device: %d rows for a system of %d atoms, positions %s, strides %lld %lld / %lld %lld\n",
                io->n_rows, N, io->positions ? "set" : "null", io->pos_row_stride, io->pos_comp_stride,
                io->f_row_stride, io->f_comp_stride);
        return -1;
    }
    if ((io->flags & LOUP_IO_ATOM_ENERGY) || io->atom_energy) {
        fprintf(stderr, "loup_compute_device: per-atom energies are not available\n");
        return -1;
    }
    if (!valid_cell(cell)) return -1;
    const bool want_e = (io->flags & LOUP_IO_ENERGY) && out_energy_ev;
    const bool want_v = (io->flags & LOUP_IO_VIRIAL) && out_virial;
    // The virial is formed in every pass: the steps that ask for it (thermo output, barostats) then run the same
    // captured graphs as the others instead of re-capturing them.
    const int stress = 1;
    const cudaStream_t cs = static_cast<cudaStream_t>(stream);

    if (!h->ev_io_in) {
        CUDA_CHECK(cudaEventCreateWithFlags(&h->ev_io_in, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&h->ev_io_out, cudaEventDisableTiming));
    }
    if (!h->d_inv_perm) {
        CUDA_CHECK(cudaMalloc(&h->d_inv_perm, (size_t)N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h->d_io_bad, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(h->d_io_bad, 0, sizeof(int), h->stream));
    }
    auto sync_inv_perm = [&]() -> int {
        if (h->inv_perm_epoch == h->perm_epoch) return 0;
        std::vector<int> inv(N);
        for (int d = 0; d < N; ++d) inv[h->perm[d]] = d;
        CUDA_CHECK(cudaMemcpyAsync(h->d_inv_perm, inv.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice, h->stream));
        h->inv_perm_epoch = h->perm_epoch;
        return 0;
    };
    if (sync_inv_perm() != 0) return -1;

    // The engine's work starts after everything the caller has queued on its stream.
    CUDA_CHECK(cudaEventRecord(h->ev_io_in, cs));
    CUDA_CHECK(cudaStreamWaitEvent(h->stream, h->ev_io_in, 0));

    loup::CellMat64 old_cell = h->cell;
    loup::compute_cell_mat(cell, h->cell, h->cell_vol, h->cell_heights);
    const bool cell_changed = memcmp(&old_cell, &h->cell, sizeof(old_cell)) != 0;
    if (h->gen->set_cell(cell)) h->invalidate_graphs();

    const int grid = (N + 255) / 256;
    io_gather_kernel<<<grid, 256, 0, h->stream>>>(N, io->positions, io->pos_row_stride, io->pos_comp_stride, io->ids,
                                                  h->d_inv_perm, h->d_pos64, h->d_io_bad);
    if (cell_changed || !h->nl_valid) {
        // A new cell (the list's Cartesian shifts depend on it) or no list yet: the host-driven rebuild.
        if (rebuild_neighbors(h) != 0) return -1;
        if (ensure_force_graph(h, stress) != 0) return -1;
        CUDA_CHECK(cudaGraphLaunch(h->force_graph_exec, h->stream));
    } else if (run_md_loop(h, 2, 1, 1.0, 0.0, 0.0, (unsigned long long)stress, nullptr) != 0) {
        return -1;
    }

    double inter_e = 0.0, v_grad[6] = {0, 0, 0, 0, 0, 0};
    if (want_e || want_v) {
        int flags[2] = {0, 0};
        auto fetch = [&]() -> int {
            CUDA_CHECK(cudaMemcpyAsync(&inter_e, h->d_energy_f64, sizeof(double), cudaMemcpyDeviceToHost, h->stream));
            if (want_v)
                CUDA_CHECK(cudaMemcpyAsync(v_grad, h->d_virial_grad_f64, 6 * sizeof(double), cudaMemcpyDeviceToHost, h->stream));
            CUDA_CHECK(cudaMemcpyAsync(&flags[0], h->d_io_bad, sizeof(int), cudaMemcpyDeviceToHost, h->stream));
            CUDA_CHECK(cudaMemcpyAsync(&flags[1], h->d_order_err, sizeof(int), cudaMemcpyDeviceToHost, h->stream));
            CUDA_CHECK(cudaStreamSynchronize(h->stream));
            return 0;
        };
        auto finite = [&] {
            bool ok = std::isfinite(inter_e);
            for (int i = 0; ok && i < 6; ++i) ok = std::isfinite(v_grad[i]);
            return ok;
        };
        if (fetch() != 0) return -1;
        if (flags[0] & 1) {
            fprintf(stderr, "loup_compute_device: an atom id is outside 0 .. %d\n", N - 1);
            CUDA_CHECK(cudaMemsetAsync(h->d_io_bad, 0, sizeof(int), h->stream));  // a corrected call is accepted
            return -1;
        }
        if (flags[1]) return order_error();
        // As loup_evaluate: with the tensor-core GEMM cubin loaded, a non-finite result drops it and the pass is run
        // again (the positions are still in place; no force has been added to the caller's rows yet).
        if ((!finite() || (flags[0] & 2)) && drop_hybrid(h)) {
            CUDA_CHECK(cudaMemsetAsync(h->d_io_bad, 0, sizeof(int), h->stream));
            if (ensure_force_graph(h, stress) != 0) return -1;
            CUDA_CHECK(cudaGraphLaunch(h->force_graph_exec, h->stream));
            if (fetch() != 0) return -1;
        }
        if (!finite() || (flags[0] & 2)) {
            fprintf(stderr, "loupmd: non-finite energy / forces (model output; the configuration may be unphysical)\n");
            return -1;
        }
    }

    if (sync_inv_perm() != 0) return -1;  // an atom reorder in this step's rebuild changed the permutation
    if (io->forces)
        io_scatter_kernel<<<grid, 256, 0, h->stream>>>(N, io->ids, h->d_inv_perm, h->d_forces_f64, io->forces,
                                                       io->f_row_stride, io->f_comp_stride, h->d_io_bad);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(h->ev_io_out, h->stream));
    CUDA_CHECK(cudaStreamWaitEvent(cs, h->ev_io_out, 0));

    if (want_e) *out_energy_ev = h->e0_total + inter_e;
    if (want_v) {
        // v_grad = dE/d(strain) as [xx, xy, xz, yy, yz, zz]; the virial is its negative, in the order xx yy zz xy xz yz.
        out_virial[0] = -v_grad[0];
        out_virial[1] = -v_grad[3];
        out_virial[2] = -v_grad[5];
        out_virial[3] = -v_grad[1];
        out_virial[4] = -v_grad[2];
        out_virial[5] = -v_grad[4];
    }
    return 0;
}

static int run_langevin_impl(
    LoupHandle* h, int n_steps, double dt_fs, double temp_k,
    double friction_per_fs, unsigned long long seed, LoupMDStats* out_stats
) {
    if (!std::isfinite(temp_k) || temp_k < 0.0 || !std::isfinite(friction_per_fs) || friction_per_fs < 0.0) return -1;
    return run_md_loop(h, 1, n_steps, dt_fs, temp_k, friction_per_fs, seed, out_stats);
}

static int get_state_impl(LoupHandle* h, double* out_positions, double* out_velocities, double* out_forces, double* out_energy_ev) {
    if (!h || h->N <= 0) return -1;
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

static int profile_force_impl(LoupHandle* h, int warmup, int n_calls, double* out_total_us) {
    if (!h || h->N <= 0 || !h->nl_valid || warmup < 0 || n_calls <= 0) return -1;
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

int loup_set_system(LoupHandle* h, int n_atoms, const int* atomic_numbers, const double* masses_amu, double skin) {
    return guarded(h, "loup_set_system", [&] { return set_system_impl(h, n_atoms, atomic_numbers, masses_amu, skin); });
}

int loup_set_state(LoupHandle* h, const double* positions, const double* velocities, const double* cell) {
    return guarded(h, "loup_set_state", [&] { return set_state_impl(h, positions, velocities, cell); });
}

int loup_evaluate(
    LoupHandle* h, const double* positions, const double* cell, int compute_stress,
    double* out_energy_ev, double* out_forces, double* out_stress
) {
    return guarded(h, "loup_evaluate", [&] {
        return evaluate_impl(h, positions, cell, compute_stress, out_energy_ev, out_forces, out_stress);
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

int loup_get_state(LoupHandle* h, double* out_positions, double* out_velocities, double* out_forces, double* out_energy_ev) {
    return guarded(h, "loup_get_state", [&] {
        return get_state_impl(h, out_positions, out_velocities, out_forces, out_energy_ev);
    });
}

double loup_cutoff(const LoupHandle* h) { return h && h->gen ? (double)h->cutoff() : 0.0; }

int loup_species(const LoupHandle* h, int* out_atomic_numbers, int max) {
    if (!h || !h->gen) return 0;
    const int S = h->n_species();
    for (int s = 0; out_atomic_numbers && s < S && s < max; ++s) out_atomic_numbers[s] = h->species_z(s);
    return S;
}

int loup_compute_device(LoupHandle* h, const LoupDeviceIO* io, const double* cell, void* stream,
                        double* out_energy_ev, double* out_virial) {
    return guarded(h, "loup_compute_device", [&] {
        return compute_device_impl(h, io, cell, stream, out_energy_ev, out_virial);
    });
}

int loup_profile_force(LoupHandle* h, int warmup, int n_calls, double* out_total_us) {
    return guarded(h, "loup_profile_force", [&] { return profile_force_impl(h, warmup, n_calls, out_total_us); });
}

}  // extern "C"
