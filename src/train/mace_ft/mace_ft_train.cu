// Full Pure-CUDA Foundation MACE Fine-Tuning Engine (src/train/mace_ft_train.cu -> libloupmd_ft.so).
// Zero PyTorch dependency: reads any arch=generic .loup bundle, runs exact forward + 1st-order force backward +
// 2nd-order double-backprop (d(L_E + L_F)/dW) across all layer kinds 0..4 (OMAT-0, MP-0, MPtrj, MatPES, MH-1,
// OFF23/OFF24, OMOL-0, and local trunk of POLAR-1), updates weights with GPU AdamW + Anchor + EMA, and writes
// a verified .loup bundle directly.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "mace_ft_bundle.cuh"
#include "mace_ft_gconv.cuh"
#include "mace_ft_kernels.cuh"
#include "mace_ft_prod_jit.cuh"
#include "mace_ft_types.hpp"
#include "polar.hpp"

namespace loup {
cudaError_t device_malloc(void** p, size_t bytes) { return cudaMalloc(p, bytes); }
}  // namespace loup

namespace loup_ft {

#define FT_CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) \
        throw std::runtime_error("cuBLAS error status=" + std::to_string((int)_s)); \
} while (0)

struct BatchGraphGPU {
    int G = 0;  // number of frames in batch
    int N = 0;  // total atoms
    int E = 0;  // total directed edges
    int* d_species = nullptr;      // [N]
    int* d_atom_graph = nullptr;   // [N]
    int* d_graph_natoms = nullptr; // [G]
    float* d_graph_w = nullptr;    // [G]
    int* d_senders = nullptr;      // [E]
    int* d_receivers = nullptr;    // [E]
    float4* d_vec_len = nullptr;   // [E] (rx-sx, ry-sy, rz-sz, r)
    double* d_E_ref = nullptr;     // [G]
    float* d_F_ref = nullptr;      // [3*N]
    double* d_pos64 = nullptr;     // [3*N] (PolarLR)
    std::vector<int> h_graph_natoms;
    std::vector<double> h_cells;   // [9*G] (PolarLR)
};

struct TrainerFT {
    ModelFT model;
    std::unique_ptr<loup::PolarLR> polar_lr;
    int polar_alloc_N = 0;
    cublasHandle_t cublas = nullptr;
    cudaStream_t stream = nullptr;
    int device_id = 0;

    // Pre-allocated workspace for current max (N_cap, E_cap, G_cap)
    int N_cap = 0, E_cap = 0, G_cap = 0;
    std::vector<LayerActFT> acts;
    std::vector<std::vector<float*>> dh_lr;  // [n_layers][lmax_out+1] when has_lr
    double* d_F_lr = nullptr;                // [3*N] when has_lr
    float* d_h0 = nullptr;         // [N, C0]
    float* d_bar_h0 = nullptr;     // [N, C0]
    float* d_edge_dvec = nullptr;  // [3*E]
    float* d_bar_edge_dvec = nullptr; // [3*E]
    double* d_E_pred = nullptr;    // [G]
    float* d_F_pred = nullptr;     // [3*N]
    float* d_bar_E = nullptr;      // [G]
    float* d_bar_F = nullptr;      // [3*N]
    double* d_loss_acc = nullptr;  // [1]
    double* d_gnorm_acc = nullptr; // [1]
    std::vector<void*> ws_allocs;

    // Cached batch graph upload buffers
    BatchGraphGPU bg;
    std::vector<void*> bg_allocs;

    // Pre-compiled product kernels per layer
    std::vector<const ProdFTKernels*> prod_k;

    int opt_step = 0;
    std::string last_err;

    TrainerFT() = default;
    ~TrainerFT() { destroy(); }

    void free_ws() {
        for (void* p : ws_allocs) if (p) cudaFree(p);
        ws_allocs.clear();
        acts.clear();
        dh_lr.clear();
        d_F_lr = nullptr;
        d_h0 = d_bar_h0 = d_edge_dvec = d_bar_edge_dvec = d_F_pred = d_bar_E = d_bar_F = nullptr;
        d_E_pred = d_loss_acc = d_gnorm_acc = nullptr;
        N_cap = E_cap = G_cap = 0;
    }

    void free_bg() {
        for (void* p : bg_allocs) if (p) cudaFree(p);
        bg_allocs.clear();
        bg = BatchGraphGPU{};
    }

    void destroy() {
        polar_lr.reset();
        polar_alloc_N = 0;
        free_ws();
        free_bg();
        free_model_ft(model);
        if (cublas) { cublasDestroy(cublas); cublas = nullptr; }
    }

    template <typename T>
    T* ws_alloc(int64_t count) {
        T* p = nullptr;
        FT_CUDA_CHECK(cudaMalloc(&p, std::max<int64_t>(count, 1) * sizeof(T)));
        ws_allocs.push_back(p);
        return p;
    }

    void ensure_workspace(int N, int E, int G) {
        if (N <= N_cap && E <= E_cap && G <= G_cap) return;
        free_ws();
        N_cap = std::max(N, 64);
        E_cap = std::max(E, 256);
        G_cap = std::max(G, 8);
        const int64_t Nc = N_cap, Ec = E_cap, Gc = G_cap;

        d_h0 = ws_alloc<float>(Nc * model.C0);
        d_bar_h0 = ws_alloc<float>(Nc * model.C0);
        d_edge_dvec = ws_alloc<float>(3 * Ec);
        d_bar_edge_dvec = ws_alloc<float>(3 * Ec);
        d_E_pred = ws_alloc<double>(Gc);
        d_F_pred = ws_alloc<float>(3 * Nc);
        d_bar_E = ws_alloc<float>(Gc);
        d_bar_F = ws_alloc<float>(3 * Nc);
        d_loss_acc = ws_alloc<double>(1);
        d_gnorm_acc = ws_alloc<double>(1);
        if (model.has_lr) {
            d_F_lr = ws_alloc<double>(3 * Nc);
            dh_lr.resize(model.n_layers);
            for (int t = 0; t < model.n_layers; ++t) {
                const LayerFT& L = model.layers[t];
                dh_lr[t].resize(L.lmax_out + 1, nullptr);
                for (int l = 0; l <= L.lmax_out; ++l)
                    dh_lr[t][l] = ws_alloc<float>(Nc * dimL(l) * L.C_out);
            }
        }

        acts.resize(model.n_layers);
        for (int t = 0; t < model.n_layers; ++t) {
            const LayerFT& L = model.layers[t];
            LayerActFT& A = acts[t];
            for (int l = 0; l <= L.lmax_x; ++l) {
                const int64_t sz = Nc * dimL(l) * L.C_x;
                A.x[l] = ws_alloc<float>(sz);
                A.dx[l] = ws_alloc<float>(sz);
                A.bar_x[l] = ws_alloc<float>(sz);
                A.tan_x[l] = ws_alloc<float>(sz);
            }
            if (L.first_hidden) {
                const int64_t K = (int64_t)model.S * L.Jp;
                for (int l = 0; l < 4; ++l) {
                    const int64_t sz = Nc * dimL(l) * K;
                    A.S0[l] = ws_alloc<float>(sz);
                    A.dS0[l] = ws_alloc<float>(sz);
                    A.bar_S0[l] = ws_alloc<float>(sz);
                    A.tan_S0[l] = ws_alloc<float>(sz);
                }
            } else {
                for (int l = 0; l < 4; ++l) {
                    const int64_t sz = Nc * dimL(l) * L.n_l3[l] * L.C_x;
                    A.mid[l] = ws_alloc<float>(sz);
                    A.dmid[l] = ws_alloc<float>(sz);
                    A.bar_mid[l] = ws_alloc<float>(sz);
                    A.tan_mid[l] = ws_alloc<float>(sz);
                }
            }
            if (L.kind >= 2) {
                A.D = ws_alloc<float>(Nc);
                A.dD = ws_alloc<float>(Nc);
                A.bar_D = ws_alloc<float>(Nc);
                A.tan_D = ws_alloc<float>(Nc);
            }
            for (int l = 0; l < 4; ++l) {
                const int64_t sz = Nc * dimL(l) * L.lin_cols[l];
                A.lin[l] = ws_alloc<float>(sz);
                A.dlin[l] = ws_alloc<float>(sz);
                A.bar_lin[l] = ws_alloc<float>(sz);
                A.tan_lin[l] = ws_alloc<float>(sz);
            }
            if (L.kind == 4) {
                for (int l = 0; l <= L.lmax_x; ++l) {
                    const int64_t sz = Nc * dimL(l) * L.lin_cols[l];
                    A.res[l] = ws_alloc<float>(sz);
                    A.dpre[l] = ws_alloc<float>(sz);
                    A.bar_res[l] = ws_alloc<float>(sz);
                    A.tan_pre[l] = ws_alloc<float>(sz);
                }
                for (int l = 0; l < 4; ++l) {
                    const int64_t sz = Nc * dimL(l) * L.C_msg;
                    A.gout[l] = ws_alloc<float>(sz);
                    A.dgout[l] = ws_alloc<float>(sz);
                    A.bar_gout[l] = ws_alloc<float>(sz);
                    A.tan_gout[l] = ws_alloc<float>(sz);
                }
            }
            for (int l = 0; l < 4; ++l) {
                const int64_t sz = Nc * dimL(l) * L.C_msg;
                A.msgf[l] = ws_alloc<float>(sz);
                A.dmsgf[l] = ws_alloc<float>(sz);
                A.bar_msgf[l] = ws_alloc<float>(sz);
                A.tan_msgf[l] = ws_alloc<float>(sz);
            }
            for (int l = 0; l <= L.lmax_out; ++l) {
                const int64_t sz = Nc * dimL(l) * L.C_out;
                A.y[l] = ws_alloc<float>(sz);
                A.dy[l] = ws_alloc<float>(sz);
                A.bar_y[l] = ws_alloc<float>(sz);
                A.tan_y[l] = ws_alloc<float>(sz);
                A.hout[l] = ws_alloc<float>(sz);
                A.dhout[l] = ws_alloc<float>(sz);
                A.bar_hout[l] = ws_alloc<float>(sz);
                A.tan_hout[l] = ws_alloc<float>(sz);
            }
            if (L.readout >= 1) {
                const int64_t sz = Nc * L.ro_H;
                A.hid = ws_alloc<float>(sz);
                A.pre1 = ws_alloc<float>(sz);
                A.act1 = ws_alloc<float>(sz);
                A.dhid1 = ws_alloc<float>(sz);
                A.tan_dhid1 = ws_alloc<float>(sz);
                if (L.readout == 2) {
                    A.pre2 = ws_alloc<float>(sz);
                    A.act2 = ws_alloc<float>(sz);
                    A.dhid2 = ws_alloc<float>(sz);
                    A.tan_dhid2 = ws_alloc<float>(sz);
                }
            }
        }
    }

    // Row-major cuBLAS GEMM wrappers (NO TF32 — pedantic FP32)
    void gemm_nn(int M, int N, int K, const float* A, const float* B, float* C, float beta = 0.0f) {
        const float one = 1.0f;
        FT_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &beta, C, N));
    }
    void gemm_nt(int M, int N, int K, const float* A, const float* B, float* C, float beta = 0.0f) {
        const float one = 1.0f;
        FT_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, B, K, A, K, &beta, C, N));
    }
    void gemm_tn(int M, int N, int K, const float* A, const float* B, float* C, float beta = 0.0f) {
        const float one = 1.0f;
        FT_CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &one, B, N, A, M, &beta, C, N));
    }

    void species_mm_fwd(int N, int Mdeg, int K, int J, const float* X, const float* W, float* Y, float beta = 0.0f) {
        dim3 blk(std::min(256, ((J + 31) / 32) * 32));
        dim3 grd((J + blk.x - 1) / blk.x, N * Mdeg);
        species_mm_fwd_kernel<<<grd, blk, 0, stream>>>(N, Mdeg, K, J, bg.d_species, X, W, beta, Y);
    }
    void species_mm_bwd_x(int N, int Mdeg, int K, int J, const float* dY, const float* W, float* dX, float beta = 0.0f) {
        dim3 blk(std::min(256, ((K + 31) / 32) * 32));
        dim3 grd((K + blk.x - 1) / blk.x, N * Mdeg);
        species_mm_bwd_x_kernel<<<grd, blk, 0, stream>>>(N, Mdeg, K, J, bg.d_species, dY, W, beta, dX);
    }
    void species_mm_grad_w(int N, int Mdeg, int K, int J, const float* X, const float* dY, float* dW) {
        dim3 blk(std::min(256, ((J + 31) / 32) * 32));
        dim3 grd((J + blk.x - 1) / blk.x, K, model.S);
        species_mm_grad_w_kernel<<<grd, blk, 0, stream>>>(N, Mdeg, K, J, bg.d_species, X, dY, dW);
    }

    const float* pval(int idx) const { return model.params[idx].d_val; }
    float* pgrad(int idx) const { return model.params[idx].d_grad; }
    bool ptrain(int idx) const { return idx >= 0 && model.params[idx].trainable; }

    // Launchers for JIT product kernels
    void launch_prod_fwd(int t, int N) {
        const LayerFT& L = model.layers[t];
        LayerActFT& A = acts[t];
        const ProdFTKernels* pk = prod_k[t];
        int C = L.C_msg;
        const float* Dp = (L.kind == 2 || L.kind == 3) ? A.D : nullptr;
        const float* w = pval(L.p_prod_w);
        int64_t ws = L.prod_species ? (int64_t)L.prod_segw * C : 0;
        const int* sp = bg.d_species;
        const float* ro_null = nullptr;
        double* en_null = nullptr;
        dim3 grd(N, (C + 127) / 128), blk(128);
        if (L.lmax_out == 0) {
            void* args[] = {&N, &C, &A.msgf[0], &A.msgf[1], &A.msgf[2], &A.msgf[3], &Dp, &w, &ws, &sp, &A.y[0], &ro_null, &en_null};
            FT_CUDA_CHECK(cudaLaunchKernel(pk->fwd, grd, blk, args, 0, stream));
        } else if (L.lmax_out == 1) {
            void* args[] = {&N, &C, &A.msgf[0], &A.msgf[1], &A.msgf[2], &A.msgf[3], &Dp, &w, &ws, &sp, &A.y[0], &A.y[1], &ro_null, &en_null};
            FT_CUDA_CHECK(cudaLaunchKernel(pk->fwd, grd, blk, args, 0, stream));
        } else {
            void* args[] = {&N, &C, &A.msgf[0], &A.msgf[1], &A.msgf[2], &A.msgf[3], &Dp, &w, &ws, &sp, &A.y[0], &A.y[1], &A.y[2], &ro_null, &en_null};
            FT_CUDA_CHECK(cudaLaunchKernel(pk->fwd, grd, blk, args, 0, stream));
        }
    }

    void launch_prod_bwd(int t, int N, float* const gin[3], float* dm[4], float* dD_out) {
        const LayerFT& L = model.layers[t];
        LayerActFT& A = acts[t];
        const ProdFTKernels* pk = prod_k[t];
        int C = L.C_msg;
        const float* Dp = (L.kind == 2 || L.kind == 3) ? A.D : nullptr;
        const float* w = pval(L.p_prod_w);
        int64_t ws = L.prod_species ? (int64_t)L.prod_segw * C : 0;
        const int* sp = bg.d_species;
        const float* gseed_null = nullptr;
        float* g0 = gin[0]; float* g1 = gin[1]; float* g2 = gin[2];
        dim3 grd(N, (C + 127) / 128), blk(128);
        if (L.lmax_out == 0) {
            void* args[] = {&N, &C, &A.msgf[0], &A.msgf[1], &A.msgf[2], &A.msgf[3], &Dp, &w, &ws, &sp, &g0, &dm[0], &dm[1], &dm[2], &dm[3], &dD_out, &gseed_null};
            FT_CUDA_CHECK(cudaLaunchKernel(pk->bwd, grd, blk, args, 0, stream));
        } else if (L.lmax_out == 1) {
            void* args[] = {&N, &C, &A.msgf[0], &A.msgf[1], &A.msgf[2], &A.msgf[3], &Dp, &w, &ws, &sp, &g0, &g1, &dm[0], &dm[1], &dm[2], &dm[3], &dD_out, &gseed_null};
            FT_CUDA_CHECK(cudaLaunchKernel(pk->bwd, grd, blk, args, 0, stream));
        } else {
            void* args[] = {&N, &C, &A.msgf[0], &A.msgf[1], &A.msgf[2], &A.msgf[3], &Dp, &w, &ws, &sp, &g0, &g1, &g2, &dm[0], &dm[1], &dm[2], &dm[3], &dD_out, &gseed_null};
            FT_CUDA_CHECK(cudaLaunchKernel(pk->bwd, grd, blk, args, 0, stream));
        }
    }

    void launch_prod_gradw(int t, int N) {
        const LayerFT& L = model.layers[t];
        if (!ptrain(L.p_prod_w)) return;
        LayerActFT& A = acts[t];
        const ProdFTKernels* pk = prod_k[t];
        int C = L.C_msg;
        const float* Dp = (L.kind == 2 || L.kind == 3) ? A.D : nullptr;
        int64_t ws = L.prod_species ? (int64_t)L.prod_segw * C : 0;
        const int* sp = bg.d_species;
        float* gw = pgrad(L.p_prod_w);
        dim3 grd(N, (C + 127) / 128), blk(128);
        if (L.lmax_out == 0) {
            void* args[] = {&N, &C, &A.msgf[0], &A.msgf[1], &A.msgf[2], &A.msgf[3], &Dp, &ws, &sp, &A.bar_y[0], &gw};
            FT_CUDA_CHECK(cudaLaunchKernel(pk->gradw, grd, blk, args, 0, stream));
        } else if (L.lmax_out == 1) {
            void* args[] = {&N, &C, &A.msgf[0], &A.msgf[1], &A.msgf[2], &A.msgf[3], &Dp, &ws, &sp, &A.bar_y[0], &A.bar_y[1], &gw};
            FT_CUDA_CHECK(cudaLaunchKernel(pk->gradw, grd, blk, args, 0, stream));
        } else {
            void* args[] = {&N, &C, &A.msgf[0], &A.msgf[1], &A.msgf[2], &A.msgf[3], &Dp, &ws, &sp, &A.bar_y[0], &A.bar_y[1], &A.bar_y[2], &gw};
            FT_CUDA_CHECK(cudaLaunchKernel(pk->gradw, grd, blk, args, 0, stream));
        }
    }

    void launch_prod_bwd2(int t, int N) {
        const LayerFT& L = model.layers[t];
        LayerActFT& A = acts[t];
        const ProdFTKernels* pk = prod_k[t];
        int C = L.C_msg;
        const float* Dp = (L.kind == 2 || L.kind == 3) ? A.D : nullptr;
        const float* tdD = (L.kind == 2 || L.kind == 3) ? A.tan_D : nullptr;
        float* bD = (L.kind == 2 || L.kind == 3) ? A.bar_D : nullptr;
        const float* w = pval(L.p_prod_w);
        float* gw = ptrain(L.p_prod_w) ? pgrad(L.p_prod_w) : nullptr;
        int64_t ws = L.prod_species ? (int64_t)L.prod_segw * C : 0;
        const int* sp = bg.d_species;
        dim3 grd(N, (C + 127) / 128), blk(128);
        if (L.lmax_out == 0) {
            void* args[] = {&N, &C, &A.msgf[0], &A.msgf[1], &A.msgf[2], &A.msgf[3], &Dp, &w, &ws, &sp,
                            &A.dy[0], &A.tan_msgf[0], &A.tan_msgf[1], &A.tan_msgf[2], &A.tan_msgf[3], &tdD,
                            &A.tan_y[0], &A.bar_msgf[0], &A.bar_msgf[1], &A.bar_msgf[2], &A.bar_msgf[3], &bD, &gw};
            FT_CUDA_CHECK(cudaLaunchKernel(pk->bwd2, grd, blk, args, 0, stream));
        } else if (L.lmax_out == 1) {
            void* args[] = {&N, &C, &A.msgf[0], &A.msgf[1], &A.msgf[2], &A.msgf[3], &Dp, &w, &ws, &sp,
                            &A.dy[0], &A.dy[1], &A.tan_msgf[0], &A.tan_msgf[1], &A.tan_msgf[2], &A.tan_msgf[3], &tdD,
                            &A.tan_y[0], &A.tan_y[1], &A.bar_msgf[0], &A.bar_msgf[1], &A.bar_msgf[2], &A.bar_msgf[3], &bD, &gw};
            FT_CUDA_CHECK(cudaLaunchKernel(pk->bwd2, grd, blk, args, 0, stream));
        } else {
            void* args[] = {&N, &C, &A.msgf[0], &A.msgf[1], &A.msgf[2], &A.msgf[3], &Dp, &w, &ws, &sp,
                            &A.dy[0], &A.dy[1], &A.dy[2], &A.tan_msgf[0], &A.tan_msgf[1], &A.tan_msgf[2], &A.tan_msgf[3], &tdD,
                            &A.tan_y[0], &A.tan_y[1], &A.tan_y[2], &A.bar_msgf[0], &A.bar_msgf[1], &A.bar_msgf[2], &A.bar_msgf[3], &bD, &gw};
            FT_CUDA_CHECK(cudaLaunchKernel(pk->bwd2, grd, blk, args, 0, stream));
        }
    }

    // ----------------------------------------------------------------------- PASS 1 (Forward) + PASS 2 (1st-order Force Backward)
    void forward_and_force_backward() {
        const int N = bg.N, E = bg.E, G = bg.G;
        FT_CUDA_CHECK(cudaMemsetAsync(d_E_pred, 0, (size_t)G * sizeof(double), stream));
        FT_CUDA_CHECK(cudaMemsetAsync(d_F_pred, 0, (size_t)3 * N * sizeof(float), stream));
        FT_CUDA_CHECK(cudaMemsetAsync(d_edge_dvec, 0, (size_t)3 * E * sizeof(float), stream));

        // E0 energy
        add_e0_kernel<<<(N + 255) / 256, 256, 0, stream>>>(N, bg.d_species, bg.d_atom_graph, model.d_e0, d_E_pred);
        // Embedding gather -> d_h0 [N, C0]
        gather_emb_fwd_kernel<<<((int64_t)N * model.C0 + 255) / 256, 256, 0, stream>>>(
            N, model.C0, bg.d_species, pval(model.p_embedding), d_h0);

        // Frozen ZBL pair contribution
        if (model.zbl.enabled && E > 0) {
            ft_zbl_kernel<<<(E + 255) / 256, 256, 0, stream>>>(
                E, model.S, model.zbl, bg.d_senders, bg.d_receivers, bg.d_vec_len,
                bg.d_species, bg.d_atom_graph, d_E_pred, d_edge_dvec);
        }

        // Zero dhout before forward readouts seed it
        for (int t = 0; t < model.n_layers; ++t) {
            const LayerFT& L = model.layers[t];
            LayerActFT& A = acts[t];
            for (int l = 0; l <= L.lmax_out; ++l)
                FT_CUDA_CHECK(cudaMemsetAsync(A.dhout[l], 0, (size_t)N * dimL(l) * L.C_out * sizeof(float), stream));
            if (L.kind >= 2) {
                FT_CUDA_CHECK(cudaMemsetAsync(A.D, 0, (size_t)N * sizeof(float), stream));
                FT_CUDA_CHECK(cudaMemsetAsync(A.dD, 0, (size_t)N * sizeof(float), stream));
            }
        }

        // ---- PASS 1: Forward t = 0 .. n_layers-1 ----
        for (int t = 0; t < model.n_layers; ++t) {
            const LayerFT& L = model.layers[t];
            LayerActFT& A = acts[t];
            float* h_in[3] = {
                t == 0 ? d_h0 : acts[t - 1].hout[0],
                t == 0 ? nullptr : (L.lmax_in >= 1 ? acts[t - 1].hout[1] : nullptr),
                t == 0 ? nullptr : (L.lmax_in >= 2 ? acts[t - 1].hout[2] : nullptr),
            };

            // 1) linear_up: x[l] = h_in[l] @ W_up[l] (needed when !first_hidden, or kind==4 for W_res)
            if (!L.first_hidden || L.kind == 4) {
                for (int l = 0; l <= L.lmax_x; ++l)
                    gemm_nn(N * dimL(l), L.C_x, L.C_in, h_in[l], pval(L.p_W_up[l]), A.x[l], 0.0f);
            }

            // 2) conv -> mid (or first_hidden -> S0) + D, then W_lin / first_C0 -> lin
            if (L.first_hidden) {
                const int K = model.S * L.Jp;
                for (int l = 0; l < 4; ++l)
                    FT_CUDA_CHECK(cudaMemsetAsync(A.S0[l], 0, (size_t)N * dimL(l) * K * sizeof(float), stream));
                if (E > 0) {
                    const int blk = std::min(256, std::max(32, ((L.Jp + 31) / 32) * 32));
                    ft_first_fwd_kernel<<<E, blk, 0, stream>>>(
                        E, model.S, L.intervals, L.J, L.Jp, L.grid, model.sh_perm, L.pair_sym,
                        bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline,
                        L.kind >= 2 ? 1 : 0, A.S0[0], A.S0[1], A.S0[2], A.S0[3], L.kind >= 2 ? A.D : nullptr);
                }
                for (int l = 0; l < 4; ++l)
                    gemm_nn(N * dimL(l), L.lin_cols[l], K, A.S0[l], pval(L.p_first_C0[l]), A.lin[l], 0.0f);
            } else {
                for (int l = 0; l < 4; ++l)
                    FT_CUDA_CHECK(cudaMemsetAsync(A.mid[l], 0, (size_t)N * dimL(l) * L.n_l3[l] * L.C_x * sizeof(float), stream));
                if (E > 0) {
                    const int blk = std::min(256, std::max(32, ((L.C_x + 31) / 32) * 32));
                    if (L.lmax_x == 0)
                        ft_gconv_fwd_kernel<0><<<E, blk, 0, stream>>>(
                            E, L.C_x, model.S, L.intervals, L.Q1, L.grid, model.sh_perm, L.pair_sym,
                            bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, L.kind >= 2 ? 1 : 0,
                            A.x[0], nullptr, nullptr, A.mid[0], A.mid[1], A.mid[2], A.mid[3], L.kind >= 2 ? A.D : nullptr);
                    else if (L.lmax_x == 1)
                        ft_gconv_fwd_kernel<1><<<E, blk, 0, stream>>>(
                            E, L.C_x, model.S, L.intervals, L.Q1, L.grid, model.sh_perm, L.pair_sym,
                            bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, L.kind >= 2 ? 1 : 0,
                            A.x[0], A.x[1], nullptr, A.mid[0], A.mid[1], A.mid[2], A.mid[3], L.kind >= 2 ? A.D : nullptr);
                    else
                        ft_gconv_fwd_kernel<2><<<E, blk, 0, stream>>>(
                            E, L.C_x, model.S, L.intervals, L.Q1, L.grid, model.sh_perm, L.pair_sym,
                            bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, L.kind >= 2 ? 1 : 0,
                            A.x[0], A.x[1], A.x[2], A.mid[0], A.mid[1], A.mid[2], A.mid[3], L.kind >= 2 ? A.D : nullptr);
                }
                for (int l = 0; l < 4; ++l)
                    gemm_nn(N * dimL(l), L.lin_cols[l], L.n_l3[l] * L.C_x, A.mid[l], pval(L.p_W_lin[l]), A.lin[l], 0.0f);
            }

            // 3) message transform: lin -> msgf
            if (L.kind == 0 || L.kind == 2) {
                for (int l = 0; l < 4; ++l)
                    species_mm_fwd(N, dimL(l), L.C_msg, L.C_msg, A.lin[l], pval(L.p_W_skip_msg[l]), A.msgf[l], 0.0f);
            } else if (L.kind == 1 || L.kind == 3) {
                for (int l = 0; l < 4; ++l)
                    FT_CUDA_CHECK(cudaMemcpyAsync(A.msgf[l], A.lin[l], (size_t)N * dimL(l) * L.C_msg * sizeof(float),
                                                  cudaMemcpyDeviceToDevice, stream));
            } else {  // kind == 4
                for (int l = 0; l <= L.lmax_x; ++l)
                    gemm_nn(N * dimL(l), L.lin_cols[l], L.C_x, A.x[l], pval(L.p_W_res[l]), A.res[l], 0.0f);
                dim3 blk(std::min(256, L.C_msg));
                dim3 grd(N, (L.C_msg + blk.x - 1) / blk.x);
                ft_gate_fwd_kernel<<<grd, blk, 0, stream>>>(
                    N, L.C_msg, L.lmax_x, A.D, L.alpha, L.beta, L.gate_cs, L.gate_cg,
                    A.lin[0], A.lin[1], A.lin[2], A.lin[3],
                    A.res[0], L.lmax_x >= 1 ? A.res[1] : nullptr, L.lmax_x >= 2 ? A.res[2] : nullptr,
                    A.gout[0], A.gout[1], A.gout[2], A.gout[3]);
                for (int l = 0; l < 4; ++l)
                    gemm_nn(N * dimL(l), L.C_msg, L.C_msg, A.gout[l], pval(L.p_W_2[l]), A.msgf[l], 0.0f);
            }

            // 4) symmetric contraction product: msgf (+ D for kind 2/3) -> y
            launch_prod_fwd(t, N);

            // 5) post-product W_p + skip W_sc -> hout
            for (int l = 0; l <= L.lmax_out; ++l) {
                gemm_nn(N * dimL(l), L.C_out, L.C_out, A.y[l], pval(L.p_W_p[l]), A.hout[l], 0.0f);
                if (L.sc_has[l]) {
                    if (L.sc_species)
                        species_mm_fwd(N, dimL(l), L.C_in, L.C_out, h_in[l], pval(L.p_W_sc[l]), A.hout[l], 1.0f);
                    else
                        gemm_nn(N * dimL(l), L.C_out, L.C_in, h_in[l], pval(L.p_W_sc[l]), A.hout[l], 1.0f);
                }
            }

            // 6) readout -> graph_E & seed dhout[0]
            if (L.readout == 0) {
                const int blk = std::min(256, std::max(32, ((L.C_out + 31) / 32) * 32));
                ft_ro_linear_fwd_bwd_kernel<<<N, blk, 0, stream>>>(
                    N, L.C_out, bg.d_atom_graph, A.hout[0], pval(L.p_ro_w), A.dhout[0], d_E_pred);
            } else if (L.readout == 1) {
                gemm_nn(N, L.ro_H, L.C_out, A.hout[0], pval(L.p_ro_W1), A.hid, 0.0f);
                ft_ro_mlp_fwd_bwd_kernel<<<((int64_t)N * L.ro_H + 255) / 256, 256, 0, stream>>>(
                    N, L.ro_H, L.ro_act_c, bg.d_atom_graph, A.hid, pval(L.p_ro_b1), pval(L.p_ro_W2),
                    A.pre1, A.act1, A.dhid1, d_E_pred, 1);
                gemm_nt(N, L.C_out, L.ro_H, A.dhid1, pval(L.p_ro_W1), A.dhout[0], 1.0f);
            } else if (L.readout == 2) {
                const int H = L.ro_H;
                gemm_nn(N, H, L.C_out, A.hout[0], pval(L.p_ro_W1), A.hid, 0.0f);
                ft_ro_mlp_fwd_bwd_kernel<<<((int64_t)N * H + 255) / 256, 256, 0, stream>>>(
                    N, H, L.ro_act_c, bg.d_atom_graph, A.hid, pval(L.p_ro_b1), nullptr,
                    A.pre1, A.act1, nullptr, nullptr, 0);
                gemm_nn(N, H, H, A.act1, pval(L.p_ro_Wm), A.hid, 0.0f);  // reuse A.hid for pre2_raw
                ft_ro_mlp_fwd_bwd_kernel<<<((int64_t)N * H + 255) / 256, 256, 0, stream>>>(
                    N, H, L.ro_act_c, bg.d_atom_graph, A.hid, pval(L.p_ro_bm), pval(L.p_ro_W2),
                    A.pre2, A.act2, A.dhid2, d_E_pred, 1);
                // dhid1 = (dhid2 @ Wm^T) * c * dsilu(pre1)
                gemm_nt(N, H, H, A.dhid2, pval(L.p_ro_Wm), A.hid, 0.0f);  // A.hid now holds dh_mid = dhid2 @ Wm^T!
                ft_ro_act_mul_kernel<<<((int64_t)N * H + 255) / 256, 256, 0, stream>>>(
                    (int64_t)N * H, L.ro_act_c, A.pre1, A.hid, A.dhid1);
                gemm_nt(N, L.C_out, H, A.dhid1, pval(L.p_ro_W1), A.dhout[0], 1.0f);
            }
        }

        // ---- Optional PolarLR (POLAR-1-S/M/L): exact E_lr, F_lr, and dh_lr[t][l] per frame ----
        if (polar_lr && bg.d_pos64 && !bg.h_cells.empty()) {
            FT_CUDA_CHECK(cudaMemsetAsync(d_F_lr, 0, (size_t)3 * N * sizeof(double), stream));
            for (int t = 0; t < model.n_layers; ++t) {
                const LayerFT& L = model.layers[t];
                for (int l = 0; l <= L.lmax_out; ++l)
                    FT_CUDA_CHECK(cudaMemsetAsync(dh_lr[t][l], 0, (size_t)N * dimL(l) * L.C_out * sizeof(float), stream));
            }
            auto noop = []() {};
            auto noop_i = [](int) {};
            auto gemm_cb = [&](bool transB, int M, int Nc, int Kc, float alpha, const float* A, int lda,
                               const float* B, int ldb, float beta, float* C_out, int ldc) {
                FT_CUBLAS_CHECK(cublasSgemm(cublas, transB ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
                                            Nc, M, Kc, &alpha, B, ldb, A, lda, &beta, C_out, ldc));
            };
            int a0 = 0;
            for (int g = 0; g < G; ++g) {
                const int Ng = bg.h_graph_natoms[g];
                polar_lr->set_cell(bg.h_cells.data() + 9 * g);
                if (Ng != polar_alloc_N) {
                    polar_lr->alloc_system(Ng);
                    polar_alloc_N = Ng;
                }
                std::vector<std::vector<float*>> h_sub(model.n_layers), dh_sub(model.n_layers);
                for (int t = 0; t < model.n_layers; ++t) {
                    const LayerFT& L = model.layers[t];
                    h_sub[t].resize(L.lmax_out + 1);
                    dh_sub[t].resize(L.lmax_out + 1);
                    for (int l = 0; l <= L.lmax_out; ++l) {
                        const int64_t off = (int64_t)a0 * dimL(l) * L.C_out;
                        h_sub[t][l] = acts[t].hout[l] + off;
                        dh_sub[t][l] = dh_lr[t][l] + off;
                    }
                }
                polar_lr->record(stream, Ng, gemm_cb, noop, noop_i, noop,
                                 bg.d_species + a0, bg.d_pos64 + 3 * a0,
                                 h_sub, dh_sub, d_E_pred + g, d_F_lr + 3 * a0, 0, nullptr, nullptr);
                a0 += Ng;
            }
            const float one = 1.0f;
            for (int t = 0; t < model.n_layers; ++t) {
                const LayerFT& L = model.layers[t];
                for (int l = 0; l <= L.lmax_out; ++l)
                    FT_CUBLAS_CHECK(cublasSaxpy(cublas, N * dimL(l) * L.C_out, &one, dh_lr[t][l], 1, acts[t].dhout[l], 1));
            }
            add_f64_to_f32_kernel<<<(3 * N + 255) / 256, 256, 0, stream>>>(3 * N, d_F_lr, d_F_pred);
        }

        // ---- PASS 2: 1st-order Backward t = n_layers-1 .. 0 -> edge_dvec -> F_pred ----
        for (int t = model.n_layers - 1; t >= 0; --t) {
            const LayerFT& L = model.layers[t];
            LayerActFT& A = acts[t];
            float* dh_in[3] = {
                t == 0 ? nullptr : acts[t - 1].dhout[0],
                t == 0 ? nullptr : (L.lmax_in >= 1 ? acts[t - 1].dhout[1] : nullptr),
                t == 0 ? nullptr : (L.lmax_in >= 2 ? acts[t - 1].dhout[2] : nullptr),
            };

            // 1) W_p^T -> dy, W_sc^T -> dh_in
            for (int l = 0; l <= L.lmax_out; ++l) {
                gemm_nt(N * dimL(l), L.C_out, L.C_out, A.dhout[l], pval(L.p_W_p[l]), A.dy[l], 0.0f);
                if (L.sc_has[l] && t > 0) {
                    if (L.sc_species)
                        species_mm_bwd_x(N, dimL(l), L.C_in, L.C_out, A.dhout[l], pval(L.p_W_sc[l]), dh_in[l], 1.0f);
                    else
                        gemm_nt(N * dimL(l), L.C_in, L.C_out, A.dhout[l], pval(L.p_W_sc[l]), dh_in[l], 1.0f);
                }
            }

            // 2) product backward: dy -> dmsgf (+ dD for kind 2/3)
            launch_prod_bwd(t, N, A.dy, A.dmsgf, (L.kind == 2 || L.kind == 3) ? A.dD : nullptr);

            // 3) message transform backward: dmsgf -> dlin (+ dpre, dD for kind 4)
            if (L.kind == 0 || L.kind == 2) {
                for (int l = 0; l < 4; ++l)
                    species_mm_bwd_x(N, dimL(l), L.C_msg, L.C_msg, A.dmsgf[l], pval(L.p_W_skip_msg[l]), A.dlin[l], 0.0f);
            } else if (L.kind == 1 || L.kind == 3) {
                for (int l = 0; l < 4; ++l)
                    FT_CUDA_CHECK(cudaMemcpyAsync(A.dlin[l], A.dmsgf[l], (size_t)N * dimL(l) * L.C_msg * sizeof(float),
                                                  cudaMemcpyDeviceToDevice, stream));
            } else {  // kind == 4
                for (int l = 0; l < 4; ++l)
                    gemm_nt(N * dimL(l), L.C_msg, L.C_msg, A.dmsgf[l], pval(L.p_W_2[l]), A.dgout[l], 0.0f);
                dim3 blk(std::min(256, L.C_msg));
                dim3 grd(N, (L.C_msg + blk.x - 1) / blk.x);
                ft_gate_bwd_kernel<<<grd, blk, 0, stream>>>(
                    N, L.C_msg, L.lmax_x, A.D, L.alpha, L.beta, L.gate_cs, L.gate_cg,
                    A.lin[0], A.lin[1], A.lin[2], A.lin[3],
                    A.res[0], L.lmax_x >= 1 ? A.res[1] : nullptr, L.lmax_x >= 2 ? A.res[2] : nullptr,
                    A.dgout[0], A.dgout[1], A.dgout[2], A.dgout[3],
                    A.dpre[0], L.lmax_x >= 1 ? A.dpre[1] : nullptr, L.lmax_x >= 2 ? A.dpre[2] : nullptr,
                    A.dlin[0], A.dlin[1], A.dlin[2], A.dlin[3], A.dD);
            }

            // 4) W_lin^T / first_C0^T -> dmid / dS0, then edge_dvec + dx
            if (L.first_hidden) {
                const int K = model.S * L.Jp;
                for (int l = 0; l < 4; ++l)
                    gemm_nt(N * dimL(l), K, L.lin_cols[l], A.dlin[l], pval(L.p_first_C0[l]), A.dS0[l], 0.0f);
                if (E > 0) {
                    const int blk = std::min(256, std::max(32, ((L.Jp + 31) / 32) * 32));
                    ft_first_bwd_kernel<<<E, blk, 0, stream>>>(
                        E, model.S, L.intervals, L.J, L.Jp, L.grid, model.sh_perm, L.pair_sym,
                        bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, L.kind >= 2 ? 1 : 0,
                        A.dS0[0], A.dS0[1], A.dS0[2], A.dS0[3], L.kind >= 2 ? A.dD : nullptr, d_edge_dvec);
                }
            } else {
                for (int l = 0; l < 4; ++l)
                    gemm_nt(N * dimL(l), L.n_l3[l] * L.C_x, L.lin_cols[l], A.dlin[l], pval(L.p_W_lin[l]), A.dmid[l], 0.0f);
                for (int l = 0; l <= L.lmax_x; ++l)
                    FT_CUDA_CHECK(cudaMemsetAsync(A.dx[l], 0, (size_t)N * dimL(l) * L.C_x * sizeof(float), stream));
                if (L.kind == 4) {
                    for (int l = 0; l <= L.lmax_x; ++l)
                        gemm_nt(N * dimL(l), L.C_x, L.lin_cols[l], A.dpre[l], pval(L.p_W_res[l]), A.dx[l], 1.0f);
                }
                if (E > 0) {
                    const int blk = std::min(256, std::max(32, ((L.C_x + 31) / 32) * 32));
                    float* dx0 = t > 0 ? A.dx[0] : nullptr;
                    float* dx1 = (t > 0 && L.lmax_x >= 1) ? A.dx[1] : nullptr;
                    float* dx2 = (t > 0 && L.lmax_x >= 2) ? A.dx[2] : nullptr;
                    if (L.lmax_x == 0)
                        ft_gconv_bwd_kernel<0><<<E, blk, 0, stream>>>(
                            E, L.C_x, model.S, L.intervals, L.Q1, L.grid, model.sh_perm, L.pair_sym,
                            bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, L.kind >= 2 ? 1 : 0,
                            A.x[0], nullptr, nullptr, A.dmid[0], A.dmid[1], A.dmid[2], A.dmid[3],
                            L.kind >= 2 ? A.dD : nullptr, dx0, nullptr, nullptr, d_edge_dvec);
                    else if (L.lmax_x == 1)
                        ft_gconv_bwd_kernel<1><<<E, blk, 0, stream>>>(
                            E, L.C_x, model.S, L.intervals, L.Q1, L.grid, model.sh_perm, L.pair_sym,
                            bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, L.kind >= 2 ? 1 : 0,
                            A.x[0], A.x[1], nullptr, A.dmid[0], A.dmid[1], A.dmid[2], A.dmid[3],
                            L.kind >= 2 ? A.dD : nullptr, dx0, dx1, nullptr, d_edge_dvec);
                    else
                        ft_gconv_bwd_kernel<2><<<E, blk, 0, stream>>>(
                            E, L.C_x, model.S, L.intervals, L.Q1, L.grid, model.sh_perm, L.pair_sym,
                            bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, L.kind >= 2 ? 1 : 0,
                            A.x[0], A.x[1], A.x[2], A.dmid[0], A.dmid[1], A.dmid[2], A.dmid[3],
                            L.kind >= 2 ? A.dD : nullptr, dx0, dx1, dx2, d_edge_dvec);
                }
                if (t > 0) {
                    for (int l = 0; l <= L.lmax_x; ++l)
                        gemm_nt(N * dimL(l), L.C_in, L.C_x, A.dx[l], pval(L.p_W_up[l]), dh_in[l], 1.0f);
                }
            }
        }

        // Scatter edge_dvec -> atomic forces F_pred
        if (E > 0) {
            scatter_forces_kernel<<<(E + 255) / 256, 256, 0, stream>>>(
                E, bg.d_senders, bg.d_receivers, d_edge_dvec, d_F_pred);
        }
    }

    // ----------------------------------------------------------------------- PASS 3 (Exact Double-Backprop: Phase 3a UP + Phase 3b DOWN)
    void double_backward(bool has_force_loss) {
        const int N = bg.N, E = bg.E;

        // Zero all bar_* and tan_* accumulators
        FT_CUDA_CHECK(cudaMemsetAsync(d_bar_h0, 0, (size_t)N * model.C0 * sizeof(float), stream));
        for (int t = 0; t < model.n_layers; ++t) {
            const LayerFT& L = model.layers[t];
            LayerActFT& A = acts[t];
            for (int l = 0; l <= L.lmax_x; ++l) {
                const size_t sz = (size_t)N * dimL(l) * L.C_x * sizeof(float);
                FT_CUDA_CHECK(cudaMemsetAsync(A.bar_x[l], 0, sz, stream));
                FT_CUDA_CHECK(cudaMemsetAsync(A.tan_x[l], 0, sz, stream));
            }
            if (L.first_hidden) {
                const size_t sz = (size_t)N * model.S * L.Jp * sizeof(float);
                for (int l = 0; l < 4; ++l) {
                    FT_CUDA_CHECK(cudaMemsetAsync(A.bar_S0[l], 0, dimL(l) * sz, stream));
                    FT_CUDA_CHECK(cudaMemsetAsync(A.tan_S0[l], 0, dimL(l) * sz, stream));
                }
            } else {
                for (int l = 0; l < 4; ++l) {
                    const size_t sz = (size_t)N * dimL(l) * L.n_l3[l] * L.C_x * sizeof(float);
                    FT_CUDA_CHECK(cudaMemsetAsync(A.bar_mid[l], 0, sz, stream));
                    FT_CUDA_CHECK(cudaMemsetAsync(A.tan_mid[l], 0, sz, stream));
                }
            }
            if (L.kind >= 2) {
                FT_CUDA_CHECK(cudaMemsetAsync(A.bar_D, 0, (size_t)N * sizeof(float), stream));
                FT_CUDA_CHECK(cudaMemsetAsync(A.tan_D, 0, (size_t)N * sizeof(float), stream));
            }
            for (int l = 0; l < 4; ++l) {
                const size_t sz = (size_t)N * dimL(l) * L.lin_cols[l] * sizeof(float);
                FT_CUDA_CHECK(cudaMemsetAsync(A.bar_lin[l], 0, sz, stream));
                FT_CUDA_CHECK(cudaMemsetAsync(A.tan_lin[l], 0, sz, stream));
            }
            if (L.kind == 4) {
                for (int l = 0; l <= L.lmax_x; ++l) {
                    const size_t sz = (size_t)N * dimL(l) * L.lin_cols[l] * sizeof(float);
                    FT_CUDA_CHECK(cudaMemsetAsync(A.bar_res[l], 0, sz, stream));
                    FT_CUDA_CHECK(cudaMemsetAsync(A.tan_pre[l], 0, sz, stream));
                }
                for (int l = 0; l < 4; ++l) {
                    const size_t sz = (size_t)N * dimL(l) * L.C_msg * sizeof(float);
                    FT_CUDA_CHECK(cudaMemsetAsync(A.bar_gout[l], 0, sz, stream));
                    FT_CUDA_CHECK(cudaMemsetAsync(A.tan_gout[l], 0, sz, stream));
                }
            }
            for (int l = 0; l < 4; ++l) {
                const size_t sz = (size_t)N * dimL(l) * L.C_msg * sizeof(float);
                FT_CUDA_CHECK(cudaMemsetAsync(A.bar_msgf[l], 0, sz, stream));
                FT_CUDA_CHECK(cudaMemsetAsync(A.tan_msgf[l], 0, sz, stream));
            }
            for (int l = 0; l <= L.lmax_out; ++l) {
                const size_t sz = (size_t)N * dimL(l) * L.C_out * sizeof(float);
                FT_CUDA_CHECK(cudaMemsetAsync(A.bar_y[l], 0, sz, stream));
                FT_CUDA_CHECK(cudaMemsetAsync(A.tan_y[l], 0, sz, stream));
                FT_CUDA_CHECK(cudaMemsetAsync(A.bar_hout[l], 0, sz, stream));
                FT_CUDA_CHECK(cudaMemsetAsync(A.tan_hout[l], 0, sz, stream));
            }
            if (L.readout >= 1) {
                FT_CUDA_CHECK(cudaMemsetAsync(A.tan_dhid1, 0, (size_t)N * L.ro_H * sizeof(float), stream));
                if (L.readout == 2)
                    FT_CUDA_CHECK(cudaMemsetAsync(A.tan_dhid2, 0, (size_t)N * L.ro_H * sizeof(float), stream));
            }
        }

        // ---- PHASE 3a (UP: t = 0 .. n_layers-1): Reverse of Pass 2 (force loss) ----
        if (has_force_loss && E > 0) {
            gather_bar_edge_dvec_kernel<<<(E + 255) / 256, 256, 0, stream>>>(
                E, bg.d_senders, bg.d_receivers, d_bar_F, d_bar_edge_dvec);

            for (int t = 0; t < model.n_layers; ++t) {
                const LayerFT& L = model.layers[t];
                LayerActFT& A = acts[t];
                float* tan_hin[3] = {
                    t == 0 ? nullptr : acts[t - 1].tan_hout[0],
                    t == 0 ? nullptr : (L.lmax_in >= 1 ? acts[t - 1].tan_hout[1] : nullptr),
                    t == 0 ? nullptr : (L.lmax_in >= 2 ? acts[t - 1].tan_hout[2] : nullptr),
                };

                // 1) Reverse of W_up^T + gconv_bwd + W_lin^T (or first_bwd + first_C0^T)
                if (L.first_hidden) {
                    const int K = model.S * L.Jp;
                    const int blk = std::min(256, std::max(32, ((L.Jp + 31) / 32) * 32));
                    ft_first_bwd2_kernel<<<E, blk, 0, stream>>>(
                        E, model.S, L.intervals, L.J, L.Jp, L.grid, model.sh_perm, L.pair_sym,
                        bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, L.kind >= 2 ? 1 : 0,
                        d_bar_edge_dvec, A.tan_S0[0], A.tan_S0[1], A.tan_S0[2], A.tan_S0[3],
                        L.kind >= 2 ? A.tan_D : nullptr);
                    for (int l = 0; l < 4; ++l) {
                        gemm_nn(N * dimL(l), L.lin_cols[l], K, A.tan_S0[l], pval(L.p_first_C0[l]), A.tan_lin[l], 1.0f);
                        if (ptrain(L.p_first_C0[l]))
                            gemm_tn(K, L.lin_cols[l], N * dimL(l), A.tan_S0[l], A.dlin[l], pgrad(L.p_first_C0[l]), 1.0f);
                    }
                } else {
                    // Reverse of dh_in += dx @ W_up^T (for t > 0)
                    if (t > 0) {
                        for (int l = 0; l <= L.lmax_x; ++l) {
                            gemm_nn(N * dimL(l), L.C_x, L.C_in, tan_hin[l], pval(L.p_W_up[l]), A.tan_x[l], 1.0f);
                            if (ptrain(L.p_W_up[l]))
                                gemm_tn(L.C_in, L.C_x, N * dimL(l), tan_hin[l], A.dx[l], pgrad(L.p_W_up[l]), 1.0f);
                        }
                    }
                    // Reverse of dx += dpre @ W_res^T (for kind == 4)
                    if (L.kind == 4) {
                        for (int l = 0; l <= L.lmax_x; ++l) {
                            gemm_nn(N * dimL(l), L.lin_cols[l], L.C_x, A.tan_x[l], pval(L.p_W_res[l]), A.tan_pre[l], 1.0f);
                            if (ptrain(L.p_W_res[l]))
                                gemm_tn(L.C_x, L.lin_cols[l], N * dimL(l), A.tan_x[l], A.dpre[l], pgrad(L.p_W_res[l]), 1.0f);
                        }
                    }
                    // Reverse of gconv_bwd -> tan_mid, tan_D, and force-loss part of bar_x!
                    const int blk = std::min(256, std::max(32, ((L.C_x + 31) / 32) * 32));
                    if (L.lmax_x == 0)
                        ft_gconv_bwd2_kernel<0><<<E, blk, 0, stream>>>(
                            E, L.C_x, model.S, L.intervals, L.Q1, L.grid, model.sh_perm, L.pair_sym,
                            bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, L.kind >= 2 ? 1 : 0,
                            A.x[0], nullptr, nullptr, A.tan_x[0], nullptr, nullptr,
                            A.dmid[0], A.dmid[1], A.dmid[2], A.dmid[3],
                            nullptr, nullptr, nullptr, nullptr, d_bar_edge_dvec,
                            A.tan_mid[0], A.tan_mid[1], A.tan_mid[2], A.tan_mid[3],
                            A.bar_x[0], nullptr, nullptr, L.kind >= 2 ? A.tan_D : nullptr);
                    else if (L.lmax_x == 1)
                        ft_gconv_bwd2_kernel<1><<<E, blk, 0, stream>>>(
                            E, L.C_x, model.S, L.intervals, L.Q1, L.grid, model.sh_perm, L.pair_sym,
                            bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, L.kind >= 2 ? 1 : 0,
                            A.x[0], A.x[1], nullptr, A.tan_x[0], A.tan_x[1], nullptr,
                            A.dmid[0], A.dmid[1], A.dmid[2], A.dmid[3],
                            nullptr, nullptr, nullptr, nullptr, d_bar_edge_dvec,
                            A.tan_mid[0], A.tan_mid[1], A.tan_mid[2], A.tan_mid[3],
                            A.bar_x[0], A.bar_x[1], nullptr, L.kind >= 2 ? A.tan_D : nullptr);
                    else
                        ft_gconv_bwd2_kernel<2><<<E, blk, 0, stream>>>(
                            E, L.C_x, model.S, L.intervals, L.Q1, L.grid, model.sh_perm, L.pair_sym,
                            bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, L.kind >= 2 ? 1 : 0,
                            A.x[0], A.x[1], A.x[2], A.tan_x[0], A.tan_x[1], A.tan_x[2],
                            A.dmid[0], A.dmid[1], A.dmid[2], A.dmid[3],
                            nullptr, nullptr, nullptr, nullptr, d_bar_edge_dvec,
                            A.tan_mid[0], A.tan_mid[1], A.tan_mid[2], A.tan_mid[3],
                            A.bar_x[0], A.bar_x[1], A.bar_x[2], L.kind >= 2 ? A.tan_D : nullptr);

                    // Reverse of dmid = dlin @ W_lin^T
                    for (int l = 0; l < 4; ++l) {
                        const int Kl = L.n_l3[l] * L.C_x;
                        gemm_nn(N * dimL(l), L.lin_cols[l], Kl, A.tan_mid[l], pval(L.p_W_lin[l]), A.tan_lin[l], 1.0f);
                        if (ptrain(L.p_W_lin[l]))
                            gemm_tn(Kl, L.lin_cols[l], N * dimL(l), A.tan_mid[l], A.dlin[l], pgrad(L.p_W_lin[l]), 1.0f);
                    }
                }

                // 2) Reverse of message transform backward: tan_lin (+ tan_pre, tan_D) -> tan_msgf
                if (L.kind == 0 || L.kind == 2) {
                    for (int l = 0; l < 4; ++l) {
                        species_mm_fwd(N, dimL(l), L.C_msg, L.C_msg, A.tan_lin[l], pval(L.p_W_skip_msg[l]), A.tan_msgf[l], 1.0f);
                        if (ptrain(L.p_W_skip_msg[l]))
                            species_mm_grad_w(N, dimL(l), L.C_msg, L.C_msg, A.tan_lin[l], A.dmsgf[l], pgrad(L.p_W_skip_msg[l]));
                    }
                } else if (L.kind == 1 || L.kind == 3) {
                    for (int l = 0; l < 4; ++l)
                        FT_CUDA_CHECK(cudaMemcpyAsync(A.tan_msgf[l], A.tan_lin[l], (size_t)N * dimL(l) * L.C_msg * sizeof(float),
                                                      cudaMemcpyDeviceToDevice, stream));
                } else {  // kind == 4
                    dim3 blk(std::min(256, L.C_msg));
                    dim3 grd(N, (L.C_msg + blk.x - 1) / blk.x);
                    ft_gate_bwd2_kernel<<<grd, blk, 0, stream>>>(
                        N, L.C_msg, L.lmax_x, A.D, L.alpha, L.beta, L.gate_cs, L.gate_cg,
                        A.lin[0], A.lin[1], A.lin[2], A.lin[3],
                        A.res[0], L.lmax_x >= 1 ? A.res[1] : nullptr, L.lmax_x >= 2 ? A.res[2] : nullptr,
                        A.dgout[0], A.dgout[1], A.dgout[2], A.dgout[3],
                        nullptr, nullptr, nullptr, nullptr,
                        A.tan_pre[0], L.lmax_x >= 1 ? A.tan_pre[1] : nullptr, L.lmax_x >= 2 ? A.tan_pre[2] : nullptr,
                        A.tan_lin[0], A.tan_lin[1], A.tan_lin[2], A.tan_lin[3], A.tan_D,
                        A.tan_gout[0], A.tan_gout[1], A.tan_gout[2], A.tan_gout[3],
                        A.bar_lin[0], A.bar_lin[1], A.bar_lin[2], A.bar_lin[3],
                        A.bar_res[0], L.lmax_x >= 1 ? A.bar_res[1] : nullptr, L.lmax_x >= 2 ? A.bar_res[2] : nullptr,
                        A.bar_D);
                    // Reverse of dgout = dmsgf @ W_2^T
                    for (int l = 0; l < 4; ++l) {
                        gemm_nn(N * dimL(l), L.C_msg, L.C_msg, A.tan_gout[l], pval(L.p_W_2[l]), A.tan_msgf[l], 1.0f);
                        if (ptrain(L.p_W_2[l]))
                            gemm_tn(L.C_msg, L.C_msg, N * dimL(l), A.tan_gout[l], A.dmsgf[l], pgrad(L.p_W_2[l]), 1.0f);
                    }
                }

                // 3) Reverse of product backward (Dual-number HVP): tan_msgf (+ tan_D) -> tan_y, bar_msgf, bar_D, grad(prod_w)
                launch_prod_bwd2(t, N);

                // 4) Reverse of dy = dhout @ W_p^T and dh_in += dhout @ W_sc^T -> tan_hout
                for (int l = 0; l <= L.lmax_out; ++l) {
                    gemm_nn(N * dimL(l), L.C_out, L.C_out, A.tan_y[l], pval(L.p_W_p[l]), A.tan_hout[l], 1.0f);
                    if (ptrain(L.p_W_p[l]))
                        gemm_tn(L.C_out, L.C_out, N * dimL(l), A.tan_y[l], A.dhout[l], pgrad(L.p_W_p[l]), 1.0f);
                    if (L.sc_has[l] && t > 0) {
                        if (L.sc_species) {
                            species_mm_fwd(N, dimL(l), L.C_in, L.C_out, tan_hin[l], pval(L.p_W_sc[l]), A.tan_hout[l], 1.0f);
                            if (ptrain(L.p_W_sc[l]))
                                species_mm_grad_w(N, dimL(l), L.C_in, L.C_out, tan_hin[l], A.dhout[l], pgrad(L.p_W_sc[l]));
                        } else {
                            gemm_nn(N * dimL(l), L.C_out, L.C_in, tan_hin[l], pval(L.p_W_sc[l]), A.tan_hout[l], 1.0f);
                            if (ptrain(L.p_W_sc[l]))
                                gemm_tn(L.C_in, L.C_out, N * dimL(l), tan_hin[l], A.dhout[l], pgrad(L.p_W_sc[l]), 1.0f);
                        }
                    }
                }

                // 5) Reverse of readout 1st-bwd: tan_hout[0] -> tan_dhid1 (and grad_W1 from force loss)
                if (L.readout >= 1) {
                    gemm_nn(N, L.ro_H, L.C_out, A.tan_hout[0], pval(L.p_ro_W1), A.tan_dhid1, 1.0f);
                    if (ptrain(L.p_ro_W1))
                        gemm_tn(L.C_out, L.ro_H, N, A.tan_hout[0], A.dhid1, pgrad(L.p_ro_W1), 1.0f);
                }
            }
        }

        // ---- PHASE 3b (DOWN: t = n_layers-1 .. 0): Reverse of Pass 1 (energy loss + force-loss bar_act) ----
        if (polar_lr && !dh_lr.empty()) {
            for (int t = 0; t < model.n_layers; ++t) {
                const LayerFT& L = model.layers[t];
                for (int l = 0; l <= L.lmax_out; ++l) {
                    const int stride = dimL(l) * L.C_out;
                    add_scaled_by_barE_kernel<<<((int64_t)N * stride + 255) / 256, 256, 0, stream>>>(
                        N, stride, bg.d_atom_graph, d_bar_E, dh_lr[t][l], acts[t].bar_hout[l]);
                }
            }
        }
        for (int t = model.n_layers - 1; t >= 0; --t) {
            const LayerFT& L = model.layers[t];
            LayerActFT& A = acts[t];
            float* h_in[3] = {
                t == 0 ? d_h0 : acts[t - 1].hout[0],
                t == 0 ? nullptr : (L.lmax_in >= 1 ? acts[t - 1].hout[1] : nullptr),
                t == 0 ? nullptr : (L.lmax_in >= 2 ? acts[t - 1].hout[2] : nullptr),
            };
            float* bar_hin[3] = {
                t == 0 ? d_bar_h0 : acts[t - 1].bar_hout[0],
                t == 0 ? nullptr : (L.lmax_in >= 1 ? acts[t - 1].bar_hout[1] : nullptr),
                t == 0 ? nullptr : (L.lmax_in >= 2 ? acts[t - 1].bar_hout[2] : nullptr),
            };

            // 1) Readout bwd2 (combines bar_E AND tan_dhout/tan_dhid1!) -> bar_hout[0] & readout grad_W
            if (L.readout == 0) {
                ft_ro_linear_bwd2_kernel<<<(L.C_out + 255) / 256, 256, 0, stream>>>(
                    N, L.C_out, bg.d_atom_graph, d_bar_E, A.hout[0], pval(L.p_ro_w),
                    has_force_loss ? A.tan_hout[0] : nullptr, A.bar_hout[0],
                    ptrain(L.p_ro_w) ? pgrad(L.p_ro_w) : nullptr);
            } else if (L.readout == 1) {
                const int H = L.ro_H;
                // Reuse A.hid as bar_pre1
                ft_ro_mlp_final_bwd2_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
                    N, H, L.ro_act_c, bg.d_atom_graph, d_bar_E, A.pre1, A.act1, pval(L.p_ro_W2),
                    has_force_loss ? A.tan_dhid1 : nullptr, A.hid,
                    ptrain(L.p_ro_b1) ? pgrad(L.p_ro_b1) : nullptr,
                    ptrain(L.p_ro_W2) ? pgrad(L.p_ro_W2) : nullptr);
                gemm_nt(N, L.C_out, H, A.hid, pval(L.p_ro_W1), A.bar_hout[0], 1.0f);
                if (ptrain(L.p_ro_W1))
                    gemm_tn(L.C_out, H, N, A.hout[0], A.hid, pgrad(L.p_ro_W1), 1.0f);
            } else if (L.readout == 2) {
                const int H = L.ro_H;
                // Note: A.hid currently holds dh_mid = dhid2 @ Wm^T from Pass 1!
                // Stage-1 force-loss tangent -> tan_dh_mid (put in A.tan_dhid2 temporarily) & bar_pre1 (need bar_act1 first!)
                // Wait: tan_dhid2 comes from tan_dh_mid @ Wm!
                // Step 1: tan_dh_mid = tan_dhid1 * c * dsilu(pre1) -> put in A.tan_dhid2 (Wait: we need bar_act1 too)
                // Let's do it cleanly in 2 steps:
                //   a) tan_dh_mid = tan_dhid1 * c * dsilu(pre1) -> stored in A.dhid1 (no longer needed!)
                //   b) tan_dhid2 = tan_dh_mid @ Wm; grad_Wm += tan_dh_mid^T @ dhid2
                //   c) final_bwd2(bar_E, pre2, act2, W2, tan_dhid2) -> bar_pre2 (stored in A.pre2), grad_bm, grad_W2
                //   d) bar_act1 = bar_pre2 @ Wm^T; grad_Wm += act1^T @ bar_pre2
                //   e) mid_bwd2(pre1, dh_mid=A.hid, bar_act1=A.act2, tan_dhid1) -> bar_pre1 (stored in A.pre1), grad_b1
                //   f) bar_hout[0] += bar_pre1 @ W1^T; grad_W1 += hout[0]^T @ bar_pre1
                if (has_force_loss) {
                    ft_ro_act_mul_kernel<<<((int64_t)N * H + 255) / 256, 256, 0, stream>>>(
                        (int64_t)N * H, L.ro_act_c, A.pre1, A.tan_dhid1, A.dhid1);  // A.dhid1 = tan_dh_mid
                    gemm_nn(N, H, H, A.dhid1, pval(L.p_ro_Wm), A.tan_dhid2, 0.0f);
                    if (ptrain(L.p_ro_Wm))
                        gemm_tn(H, H, N, A.dhid1, A.dhid2, pgrad(L.p_ro_Wm), 1.0f);
                }
                // Reuse A.dhid2 as bar_pre2
                ft_ro_mlp_final_bwd2_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
                    N, H, L.ro_act_c, bg.d_atom_graph, d_bar_E, A.pre2, A.act2, pval(L.p_ro_W2),
                    has_force_loss ? A.tan_dhid2 : nullptr, A.dhid2,
                    ptrain(L.p_ro_bm) ? pgrad(L.p_ro_bm) : nullptr,
                    ptrain(L.p_ro_W2) ? pgrad(L.p_ro_W2) : nullptr);
                // bar_act1 -> A.act2
                gemm_nt(N, H, H, A.dhid2, pval(L.p_ro_Wm), A.act2, 0.0f);
                if (ptrain(L.p_ro_Wm))
                    gemm_tn(H, H, N, A.act1, A.dhid2, pgrad(L.p_ro_Wm), 1.0f);
                // mid_bwd2 -> bar_pre1 in A.dhid1, grad_b1
                ft_ro_mlp_mid_bwd2_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
                    N, H, L.ro_act_c, A.pre1, A.hid, A.act2,
                    has_force_loss ? A.tan_dhid1 : nullptr, A.dhid1, A.tan_dhid2,
                    ptrain(L.p_ro_b1) ? pgrad(L.p_ro_b1) : nullptr);
                gemm_nt(N, L.C_out, H, A.dhid1, pval(L.p_ro_W1), A.bar_hout[0], 1.0f);
                if (ptrain(L.p_ro_W1))
                    gemm_tn(L.C_out, H, N, A.hout[0], A.dhid1, pgrad(L.p_ro_W1), 1.0f);
            }

            // 2) Reverse of hout = y @ W_p + h_in @ W_sc
            for (int l = 0; l <= L.lmax_out; ++l) {
                gemm_nt(N * dimL(l), L.C_out, L.C_out, A.bar_hout[l], pval(L.p_W_p[l]), A.bar_y[l], 1.0f);
                if (ptrain(L.p_W_p[l]))
                    gemm_tn(L.C_out, L.C_out, N * dimL(l), A.y[l], A.bar_hout[l], pgrad(L.p_W_p[l]), 1.0f);
                if (L.sc_has[l]) {
                    if (L.sc_species) {
                        species_mm_bwd_x(N, dimL(l), L.C_in, L.C_out, A.bar_hout[l], pval(L.p_W_sc[l]), bar_hin[l], 1.0f);
                        if (ptrain(L.p_W_sc[l]))
                            species_mm_grad_w(N, dimL(l), L.C_in, L.C_out, h_in[l], A.bar_hout[l], pgrad(L.p_W_sc[l]));
                    } else {
                        gemm_nt(N * dimL(l), L.C_in, L.C_out, A.bar_hout[l], pval(L.p_W_sc[l]), bar_hin[l], 1.0f);
                        if (ptrain(L.p_W_sc[l]))
                            gemm_tn(L.C_in, L.C_out, N * dimL(l), h_in[l], A.bar_hout[l], pgrad(L.p_W_sc[l]), 1.0f);
                    }
                }
            }

            // 3) Reverse of product forward: bar_y -> grad(prod_w) + bar_msgf_from_E (+ bar_D_from_E)
            launch_prod_gradw(t, N);
            {
                // gprod_bwd overwrites dm (doesn't +=), so write into A.dmsgf (now dead!) and add into A.bar_msgf!
                launch_prod_bwd(t, N, A.bar_y, A.dmsgf, (L.kind == 2 || L.kind == 3) ? A.bar_D : nullptr);
                for (int l = 0; l < 4; ++l) {
                    const float one = 1.0f;
                    FT_CUBLAS_CHECK(cublasSaxpy(cublas, N * dimL(l) * L.C_msg, &one, A.dmsgf[l], 1, A.bar_msgf[l], 1));
                }
            }

            // 4) Reverse of message transform forward: bar_msgf -> bar_lin (+ bar_res, bar_D for kind 4)
            if (L.kind == 0 || L.kind == 2) {
                for (int l = 0; l < 4; ++l) {
                    species_mm_bwd_x(N, dimL(l), L.C_msg, L.C_msg, A.bar_msgf[l], pval(L.p_W_skip_msg[l]), A.bar_lin[l], 1.0f);
                    if (ptrain(L.p_W_skip_msg[l]))
                        species_mm_grad_w(N, dimL(l), L.C_msg, L.C_msg, A.lin[l], A.bar_msgf[l], pgrad(L.p_W_skip_msg[l]));
                }
            } else if (L.kind == 1 || L.kind == 3) {
                for (int l = 0; l < 4; ++l) {
                    const float one = 1.0f;
                    FT_CUBLAS_CHECK(cublasSaxpy(cublas, N * dimL(l) * L.C_msg, &one, A.bar_msgf[l], 1, A.bar_lin[l], 1));
                }
            } else {  // kind == 4
                for (int l = 0; l < 4; ++l) {
                    gemm_nt(N * dimL(l), L.C_msg, L.C_msg, A.bar_msgf[l], pval(L.p_W_2[l]), A.bar_gout[l], 1.0f);
                    if (ptrain(L.p_W_2[l]))
                        gemm_tn(L.C_msg, L.C_msg, N * dimL(l), A.gout[l], A.bar_msgf[l], pgrad(L.p_W_2[l]), 1.0f);
                }
                // Reuse A.dpre and A.dlin (dead now!) for gate_bwd(bar_gout), then axpy into bar_res & bar_lin!
                dim3 blk(std::min(256, L.C_msg));
                dim3 grd(N, (L.C_msg + blk.x - 1) / blk.x);
                ft_gate_bwd_kernel<<<grd, blk, 0, stream>>>(
                    N, L.C_msg, L.lmax_x, A.D, L.alpha, L.beta, L.gate_cs, L.gate_cg,
                    A.lin[0], A.lin[1], A.lin[2], A.lin[3],
                    A.res[0], L.lmax_x >= 1 ? A.res[1] : nullptr, L.lmax_x >= 2 ? A.res[2] : nullptr,
                    A.bar_gout[0], A.bar_gout[1], A.bar_gout[2], A.bar_gout[3],
                    A.dpre[0], L.lmax_x >= 1 ? A.dpre[1] : nullptr, L.lmax_x >= 2 ? A.dpre[2] : nullptr,
                    A.dlin[0], A.dlin[1], A.dlin[2], A.dlin[3], A.bar_D);
                const float one = 1.0f;
                for (int l = 0; l < 4; ++l)
                    FT_CUBLAS_CHECK(cublasSaxpy(cublas, N * dimL(l) * L.lin_cols[l], &one, A.dlin[l], 1, A.bar_lin[l], 1));
                for (int l = 0; l <= L.lmax_x; ++l)
                    FT_CUBLAS_CHECK(cublasSaxpy(cublas, N * dimL(l) * L.lin_cols[l], &one, A.dpre[l], 1, A.bar_res[l], 1));
            }

            // 5) Reverse of lin = mid @ W_lin (or S0 @ first_C0) and res = x @ W_res
            if (L.first_hidden) {
                const int K = model.S * L.Jp;
                for (int l = 0; l < 4; ++l) {
                    if (ptrain(L.p_first_C0[l]))
                        gemm_tn(K, L.lin_cols[l], N * dimL(l), A.S0[l], A.bar_lin[l], pgrad(L.p_first_C0[l]), 1.0f);
                }
                if (L.kind == 4) {
                    for (int l = 0; l <= L.lmax_x; ++l) {
                        gemm_nt(N * dimL(l), L.C_x, L.lin_cols[l], A.bar_res[l], pval(L.p_W_res[l]), A.bar_x[l], 1.0f);
                        if (ptrain(L.p_W_res[l]))
                            gemm_tn(L.C_x, L.lin_cols[l], N * dimL(l), A.x[l], A.bar_res[l], pgrad(L.p_W_res[l]), 1.0f);
                    }
                    // Also backprop bar_x through W_up[0] into bar_hin[0] (embedding)!
                    for (int l = 0; l <= L.lmax_x; ++l) {
                        gemm_nt(N * dimL(l), L.C_in, L.C_x, A.bar_x[l], pval(L.p_W_up[l]), bar_hin[l], 1.0f);
                        if (ptrain(L.p_W_up[l]))
                            gemm_tn(L.C_in, L.C_x, N * dimL(l), h_in[l], A.bar_x[l], pgrad(L.p_W_up[l]), 1.0f);
                    }
                }
            } else {
                for (int l = 0; l < 4; ++l) {
                    const int Kl = L.n_l3[l] * L.C_x;
                    gemm_nt(N * dimL(l), Kl, L.lin_cols[l], A.bar_lin[l], pval(L.p_W_lin[l]), A.bar_mid[l], 1.0f);
                    if (ptrain(L.p_W_lin[l]))
                        gemm_tn(Kl, L.lin_cols[l], N * dimL(l), A.mid[l], A.bar_lin[l], pgrad(L.p_W_lin[l]), 1.0f);
                }
                if (L.kind == 4) {
                    for (int l = 0; l <= L.lmax_x; ++l) {
                        gemm_nt(N * dimL(l), L.C_x, L.lin_cols[l], A.bar_res[l], pval(L.p_W_res[l]), A.bar_x[l], 1.0f);
                        if (ptrain(L.p_W_res[l]))
                            gemm_tn(L.C_x, L.lin_cols[l], N * dimL(l), A.x[l], A.bar_res[l], pgrad(L.p_W_res[l]), 1.0f);
                    }
                }
                // Reverse of gconv forward: bar_mid -> bar_x += G^T(R*Y; bar_mid)
                if (E > 0) {
                    const int blk = std::min(256, std::max(32, ((L.C_x + 31) / 32) * 32));
                    if (L.lmax_x == 0)
                        ft_gconv_bwd2_kernel<0><<<E, blk, 0, stream>>>(
                            E, L.C_x, model.S, L.intervals, L.Q1, L.grid, model.sh_perm, L.pair_sym,
                            bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, 0,
                            A.x[0], nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr,
                            A.bar_mid[0], A.bar_mid[1], A.bar_mid[2], A.bar_mid[3], nullptr,
                            nullptr, nullptr, nullptr, nullptr,
                            A.bar_x[0], nullptr, nullptr, nullptr);
                    else if (L.lmax_x == 1)
                        ft_gconv_bwd2_kernel<1><<<E, blk, 0, stream>>>(
                            E, L.C_x, model.S, L.intervals, L.Q1, L.grid, model.sh_perm, L.pair_sym,
                            bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, 0,
                            A.x[0], A.x[1], nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr,
                            A.bar_mid[0], A.bar_mid[1], A.bar_mid[2], A.bar_mid[3], nullptr,
                            nullptr, nullptr, nullptr, nullptr,
                            A.bar_x[0], A.bar_x[1], nullptr, nullptr);
                    else
                        ft_gconv_bwd2_kernel<2><<<E, blk, 0, stream>>>(
                            E, L.C_x, model.S, L.intervals, L.Q1, L.grid, model.sh_perm, L.pair_sym,
                            bg.d_senders, bg.d_receivers, bg.d_vec_len, bg.d_species, L.d_spline, 0,
                            A.x[0], A.x[1], A.x[2], nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr,
                            A.bar_mid[0], A.bar_mid[1], A.bar_mid[2], A.bar_mid[3], nullptr,
                            nullptr, nullptr, nullptr, nullptr,
                            A.bar_x[0], A.bar_x[1], A.bar_x[2], nullptr);
                }
                // Reverse of x = h_in @ W_up
                for (int l = 0; l <= L.lmax_x; ++l) {
                    gemm_nt(N * dimL(l), L.C_in, L.C_x, A.bar_x[l], pval(L.p_W_up[l]), bar_hin[l], 1.0f);
                    if (ptrain(L.p_W_up[l]))
                        gemm_tn(L.C_in, L.C_x, N * dimL(l), h_in[l], A.bar_x[l], pgrad(L.p_W_up[l]), 1.0f);
                }
            }
        }

        // Embedding gradient
        if (ptrain(model.p_embedding)) {
            scatter_emb_bwd_kernel<<<((int64_t)N * model.C0 + 255) / 256, 256, 0, stream>>>(
                N, model.C0, bg.d_species, d_bar_h0, pgrad(model.p_embedding));
        }
    }

    void configure_strategy(const std::string& strategy) {
        for (auto& p : model.params) p.trainable = false;
        const int last = model.n_layers - 1;
        for (int t = 0; t < model.n_layers; ++t) {
            const LayerFT& L = model.layers[t];
            auto set = [&](int idx) { if (idx >= 0) model.params[idx].trainable = true; };
            if (strategy == "readout") {
                set(L.p_ro_w); set(L.p_ro_W1); set(L.p_ro_b1); set(L.p_ro_Wm); set(L.p_ro_bm); set(L.p_ro_W2);
                if (t == last) {
                    for (int l = 0; l <= L.lmax_out; ++l) set(L.p_W_p[l]);
                }
            } else if (strategy == "top") {
                set(L.p_ro_w); set(L.p_ro_W1); set(L.p_ro_b1); set(L.p_ro_Wm); set(L.p_ro_bm); set(L.p_ro_W2);
                for (int l = 0; l <= L.lmax_out; ++l) set(L.p_W_p[l]);
                if (t == last) {
                    for (int l = 0; l <= L.lmax_x; ++l) { set(L.p_W_up[l]); set(L.p_W_res[l]); }
                    for (int l = 0; l < 4; ++l) { set(L.p_first_C0[l]); set(L.p_W_lin[l]); set(L.p_W_skip_msg[l]); set(L.p_W_2[l]); }
                    set(L.p_prod_w);
                    for (int l = 0; l < 3; ++l) set(L.p_W_sc[l]);
                }
            } else if (strategy == "full") {
                set(model.p_embedding);
                for (int l = 0; l <= L.lmax_x; ++l) { set(L.p_W_up[l]); set(L.p_W_res[l]); }
                for (int l = 0; l < 4; ++l) { set(L.p_first_C0[l]); set(L.p_W_lin[l]); set(L.p_W_skip_msg[l]); set(L.p_W_2[l]); }
                set(L.p_prod_w);
                for (int l = 0; l <= L.lmax_out; ++l) set(L.p_W_p[l]);
                for (int l = 0; l < 3; ++l) set(L.p_W_sc[l]);
                set(L.p_ro_w); set(L.p_ro_W1); set(L.p_ro_b1); set(L.p_ro_Wm); set(L.p_ro_bm); set(L.p_ro_W2);
            } else {
                throw std::runtime_error("mace_ft: unknown strategy " + strategy);
            }
        }
    }

    void upload_batch(int G, int N, int E, const int* species, const int* atom_graph, const int* graph_natoms,
                      const float* graph_w, const int* senders, const int* receivers, const float* vec_len_4,
                      const double* E_ref, const float* F_ref,
                      const double* pos64 = nullptr, const double* cells = nullptr) {
        free_bg();
        bg.G = G; bg.N = N; bg.E = E;
        auto up = [&](const void* src, size_t bytes) -> void* {
            void* p = nullptr;
            FT_CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(bytes, 16)));
            if (src && bytes) FT_CUDA_CHECK(cudaMemcpyAsync(p, src, bytes, cudaMemcpyHostToDevice, stream));
            bg_allocs.push_back(p);
            return p;
        };
        bg.d_species = (int*)up(species, (size_t)N * sizeof(int));
        bg.d_atom_graph = (int*)up(atom_graph, (size_t)N * sizeof(int));
        bg.d_graph_natoms = (int*)up(graph_natoms, (size_t)G * sizeof(int));
        bg.d_graph_w = (float*)up(graph_w, (size_t)G * sizeof(float));
        bg.d_senders = (int*)up(senders, (size_t)E * sizeof(int));
        bg.d_receivers = (int*)up(receivers, (size_t)E * sizeof(int));
        bg.d_vec_len = (float4*)up(vec_len_4, (size_t)E * sizeof(float4));
        bg.d_E_ref = (double*)up(E_ref, (size_t)G * sizeof(double));
        bg.d_F_ref = (float*)up(F_ref, (size_t)3 * N * sizeof(float));
        bg.h_graph_natoms.assign(graph_natoms, graph_natoms + G);
        if (pos64) bg.d_pos64 = (double*)up(pos64, (size_t)3 * N * sizeof(double));
        if (cells) bg.h_cells.assign(cells, cells + 9 * G);
        ensure_workspace(N, E, G);
    }

    void zero_grads() {
        for (auto& p : model.params) {
            if (p.trainable && p.numel > 0)
                FT_CUDA_CHECK(cudaMemsetAsync(p.d_grad, 0, (size_t)p.numel * sizeof(float), stream));
        }
    }

    double compute_loss_and_backward(float w_e, float w_f, float inv_accum) {
        forward_and_force_backward();
        FT_CUDA_CHECK(cudaMemsetAsync(d_loss_acc, 0, sizeof(double), stream));
        FT_CUDA_CHECK(cudaMemsetAsync(d_bar_E, 0, (size_t)bg.G * sizeof(float), stream));
        FT_CUDA_CHECK(cudaMemsetAsync(d_bar_F, 0, (size_t)3 * bg.N * sizeof(float), stream));

        if (w_e > 0.0f) {
            energy_loss_kernel<<<(bg.G + 63) / 64, 64, 0, stream>>>(
                bg.G, w_e, inv_accum, d_E_pred, bg.d_E_ref, bg.d_graph_natoms, bg.d_graph_w, d_bar_E, d_loss_acc);
        }
        if (w_f > 0.0f) {
            force_loss_kernel<<<(3 * bg.N + 255) / 256, 256, 0, stream>>>(
                bg.N, w_f, inv_accum, d_F_pred, bg.d_F_ref, bg.d_atom_graph, bg.d_graph_w, d_bar_F, d_loss_acc);
        }
        double_backward(w_f > 0.0f);
        double loss_h = 0.0;
        FT_CUDA_CHECK(cudaMemcpyAsync(&loss_h, d_loss_acc, sizeof(double), cudaMemcpyDeviceToHost, stream));
        FT_CUDA_CHECK(cudaStreamSynchronize(stream));
        return loss_h;
    }

    void optimizer_step(float lr_head, float lr_trunk, float max_grad_norm, float anchor_weight, float ema_decay_cap) {
        ++opt_step;
        FT_CUDA_CHECK(cudaMemsetAsync(d_gnorm_acc, 0, sizeof(double), stream));
        int64_t total_trainable = 0;
        for (const auto& p : model.params) {
            if (!p.trainable || p.numel == 0) continue;
            total_trainable += p.numel;
            grad_norm_sq_kernel<<<(unsigned)((p.numel + 255) / 256), 256, 0, stream>>>(p.numel, p.lr_scale, p.d_grad, d_gnorm_acc);
        }
        double gnorm_sq = 0.0;
        FT_CUDA_CHECK(cudaMemcpyAsync(&gnorm_sq, d_gnorm_acc, sizeof(double), cudaMemcpyDeviceToHost, stream));
        FT_CUDA_CHECK(cudaStreamSynchronize(stream));
        const float gnorm = (float)std::sqrt(std::max(0.0, gnorm_sq));
        const float g_scale = (max_grad_norm > 0.0f && gnorm > max_grad_norm) ? (max_grad_norm / (gnorm + 1e-8f)) : 1.0f;

        const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 1e-6f;
        const float bc1 = 1.0f / (1.0f - std::pow(beta1, (float)opt_step));
        const float bc2 = 1.0f / (1.0f - std::pow(beta2, (float)opt_step));
        const float ema_d = std::min(ema_decay_cap, (1.0f + (float)opt_step) / (10.0f + (float)opt_step));
        const float anchor_coeff = (anchor_weight > 0.0f && total_trainable > 0)
                                       ? (2.0f * anchor_weight / (float)total_trainable)
                                       : 0.0f;

        for (auto& p : model.params) {
            if (!p.trainable || p.numel == 0) continue;
            const float lr = (p.is_head ? lr_head : lr_trunk) * p.lr_scale;
            adamw_ema_anchor_kernel<<<(unsigned)((p.numel + 255) / 256), 256, 0, stream>>>(
                p.numel, lr, beta1, beta2, eps, wd, bc1, bc2, g_scale, anchor_coeff, ema_d,
                p.d_val, p.d_grad, p.d_m, p.d_v, p.d_ema, p.d_anchor);
        }
    }
};

}  // namespace loup_ft

// =========================================================================== C ABI (ctypes-callable, zero PyTorch!)
extern "C" {

void* mace_ft_create(const char* bundle_path, const char* strategy, int device_id, char* err_buf, int err_len) {
    try {
        FT_CUDA_CHECK(cudaSetDevice(device_id));
        auto* tr = new loup_ft::TrainerFT();
        tr->device_id = device_id;
        FT_CUBLAS_CHECK(cublasCreate(&tr->cublas));
        FT_CUBLAS_CHECK(cublasSetMathMode(tr->cublas, CUBLAS_PEDANTIC_MATH));
        loup_ft::load_model_ft(bundle_path, tr->model);
        if (tr->model.has_lr) {
            loup::Bundle b(bundle_path);
            tr->polar_lr = std::make_unique<loup::PolarLR>();
            tr->polar_lr->load(b, tr->model.S, tr->model.n_layers);
            tr->polar_lr->disable_h2_row();
        }
        tr->prod_k.resize(tr->model.n_layers);
        for (int t = 0; t < tr->model.n_layers; ++t)
            tr->prod_k[t] = &loup_ft::get_prod_ft_kernels(tr->model.layers[t]);
        tr->configure_strategy(strategy ? strategy : "full");
        return tr;
    } catch (const std::exception& e) {
        if (err_buf && err_len > 0) snprintf(err_buf, (size_t)err_len, "%s", e.what());
        return nullptr;
    }
}

void mace_ft_destroy(void* handle) {
    delete reinterpret_cast<loup_ft::TrainerFT*>(handle);
}

int mace_ft_info(void* handle, int* out_S, int* out_n_layers, int* out_has_lr, int* out_n_params,
                 int64_t* out_trainable_elems, int64_t* out_total_elems, double* out_rmax) {
    auto* tr = reinterpret_cast<loup_ft::TrainerFT*>(handle);
    if (!tr) return -1;
    if (out_S) *out_S = tr->model.S;
    if (out_n_layers) *out_n_layers = tr->model.n_layers;
    if (out_has_lr) *out_has_lr = tr->model.has_lr;
    if (out_n_params) *out_n_params = (int)tr->model.params.size();
    int64_t tr_el = 0, tot_el = 0;
    for (const auto& p : tr->model.params) {
        tot_el += p.numel;
        if (p.trainable) tr_el += p.numel;
    }
    if (out_trainable_elems) *out_trainable_elems = tr_el;
    if (out_total_elems) *out_total_elems = tot_el;
    if (out_rmax) *out_rmax = (double)tr->model.r_max;
    return 0;
}

int mace_ft_set_e0(void* handle, const double* e0_host) {
    auto* tr = reinterpret_cast<loup_ft::TrainerFT*>(handle);
    if (!tr) return -1;
    try {
        tr->model.e0.assign(e0_host, e0_host + tr->model.S);
        FT_CUDA_CHECK(cudaMemcpy(tr->model.d_e0, e0_host, (size_t)tr->model.S * sizeof(double), cudaMemcpyHostToDevice));
        return 0;
    } catch (...) { return -1; }
}

int mace_ft_get_e0(void* handle, double* e0_host) {
    auto* tr = reinterpret_cast<loup_ft::TrainerFT*>(handle);
    if (!tr) return -1;
    memcpy(e0_host, tr->model.e0.data(), (size_t)tr->model.S * sizeof(double));
    return 0;
}

int mace_ft_eval_batch(void* handle, int G, int N, int E,
                       const int* species, const int* atom_graph, const int* graph_natoms,
                       const float* graph_w, const int* senders, const int* receivers, const float* vec_len_4,
                       const double* pos64, const double* cells,
                       int use_ema, double* out_E_pred, float* out_F_pred) {
    auto* tr = reinterpret_cast<loup_ft::TrainerFT*>(handle);
    if (!tr) return -1;
    try {
        // Optional EMA swap
        std::vector<float*> orig_vals;
        if (use_ema) {
            orig_vals.reserve(tr->model.params.size());
            for (auto& p : tr->model.params) { orig_vals.push_back(p.d_val); p.d_val = p.d_ema; }
        }
        tr->upload_batch(G, N, E, species, atom_graph, graph_natoms, graph_w, senders, receivers, vec_len_4,
                         nullptr, nullptr, pos64, cells);
        tr->forward_and_force_backward();
        FT_CUDA_CHECK(cudaMemcpy(out_E_pred, tr->d_E_pred, (size_t)G * sizeof(double), cudaMemcpyDeviceToHost));
        FT_CUDA_CHECK(cudaMemcpy(out_F_pred, tr->d_F_pred, (size_t)3 * N * sizeof(float), cudaMemcpyDeviceToHost));
        if (use_ema) {
            for (size_t i = 0; i < tr->model.params.size(); ++i) tr->model.params[i].d_val = orig_vals[i];
        }
        return 0;
    } catch (const std::exception& e) {
        tr->last_err = e.what();
        return -1;
    }
}

int mace_ft_zero_grad(void* handle) {
    auto* tr = reinterpret_cast<loup_ft::TrainerFT*>(handle);
    if (!tr) return -1;
    try { tr->zero_grads(); return 0; } catch (...) { return -1; }
}

int mace_ft_backward_batch(void* handle, int G, int N, int E,
                           const int* species, const int* atom_graph, const int* graph_natoms,
                           const float* graph_w, const int* senders, const int* receivers, const float* vec_len_4,
                           const double* pos64, const double* cells,
                           const double* E_ref, const float* F_ref,
                           float w_e, float w_f, float inv_accum, double* out_loss) {
    auto* tr = reinterpret_cast<loup_ft::TrainerFT*>(handle);
    if (!tr) return -1;
    try {
        tr->upload_batch(G, N, E, species, atom_graph, graph_natoms, graph_w, senders, receivers, vec_len_4,
                         E_ref, F_ref, pos64, cells);
        const double l = tr->compute_loss_and_backward(w_e, w_f, inv_accum);
        if (out_loss) *out_loss = l;
        return 0;
    } catch (const std::exception& e) {
        tr->last_err = e.what();
        return -1;
    }
}

int mace_ft_step(void* handle, float lr_head, float lr_trunk, float max_grad_norm, float anchor_weight, float ema_decay_cap) {
    auto* tr = reinterpret_cast<loup_ft::TrainerFT*>(handle);
    if (!tr) return -1;
    try { tr->optimizer_step(lr_head, lr_trunk, max_grad_norm, anchor_weight, ema_decay_cap); return 0; }
    catch (const std::exception& e) { tr->last_err = e.what(); return -1; }
}

int mace_ft_save(void* handle, const char* out_path, int use_ema) {
    auto* tr = reinterpret_cast<loup_ft::TrainerFT*>(handle);
    if (!tr) return -1;
    try { loup_ft::save_model_ft(tr->model, out_path, use_ema != 0); return 0; }
    catch (const std::exception& e) { tr->last_err = e.what(); return -1; }
}

// Introspection helpers for exact finite-difference gradient verification!
int mace_ft_param_meta(void* handle, int idx, char* name_buf, int name_len, int64_t* out_numel, int* out_trainable) {
    auto* tr = reinterpret_cast<loup_ft::TrainerFT*>(handle);
    if (!tr || idx < 0 || idx >= (int)tr->model.params.size()) return -1;
    const auto& p = tr->model.params[idx];
    if (name_buf && name_len > 0) snprintf(name_buf, (size_t)name_len, "%s", p.bundle_key.c_str());
    if (out_numel) *out_numel = p.numel;
    if (out_trainable) *out_trainable = p.trainable ? 1 : 0;
    return 0;
}

int mace_ft_param_get_elem(void* handle, int idx, int64_t elem, float* out_val, float* out_grad) {
    auto* tr = reinterpret_cast<loup_ft::TrainerFT*>(handle);
    if (!tr || idx < 0 || idx >= (int)tr->model.params.size()) return -1;
    const auto& p = tr->model.params[idx];
    if (elem < 0 || elem >= p.numel) return -1;
    if (out_val) FT_CUDA_CHECK(cudaMemcpy(out_val, p.d_val + elem, sizeof(float), cudaMemcpyDeviceToHost));
    if (out_grad) FT_CUDA_CHECK(cudaMemcpy(out_grad, p.d_grad + elem, sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

int mace_ft_param_set_elem(void* handle, int idx, int64_t elem, float val) {
    auto* tr = reinterpret_cast<loup_ft::TrainerFT*>(handle);
    if (!tr || idx < 0 || idx >= (int)tr->model.params.size()) return -1;
    const auto& p = tr->model.params[idx];
    if (elem < 0 || elem >= p.numel) return -1;
    FT_CUDA_CHECK(cudaMemcpy(p.d_val + elem, &val, sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

const char* mace_ft_last_error(void* handle) {
    auto* tr = reinterpret_cast<loup_ft::TrainerFT*>(handle);
    return tr ? tr->last_err.c_str() : "null handle";
}

}  // extern "C"
