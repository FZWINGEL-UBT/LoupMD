#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err__ = (call);                                                                     \
        if (err__ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__));\
            return -1;                                                                                  \
        }                                                                                               \
    } while (0)

#define CUDA_CHECK_THROW(call)                                                                          \
    do {                                                                                                \
        cudaError_t err__ = (call);                                                                     \
        if (err__ != cudaSuccess) {                                                                     \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__));          \
        }                                                                                               \
    } while (0)

#define CUBLAS_CHECK(call)                                                                              \
    do {                                                                                                \
        cublasStatus_t st__ = (call);                                                                   \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                                            \
            fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, __LINE__, (int)st__);       \
            return -1;                                                                                  \
        }                                                                                               \
    } while (0)

#define CUBLAS_CHECK_THROW(call)                                                                        \
    do {                                                                                                \
        cublasStatus_t st__ = (call);                                                                   \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                                            \
            throw std::runtime_error("cuBLAS error");                                                   \
        }                                                                                               \
    } while (0)

namespace loup { class Bundle; }

struct SplineGridHost {
    float starts[8];
    float ends[8];
    float inv_h[8];
    int offsets[8];
    int counts[8];
    int n_bands;
    int total_intervals;
};

struct LoupModelGPU {
    int S = 0;
    int P_sym = 0;
    float r_max = 6.0f;
    float scale = 1.0f;
    std::vector<int32_t> atomic_numbers;
    std::vector<double> atomic_energies;

    // Interaction 0 radial table
    int pieces0 = 0;
    int n_bins0 = 0;
    float r_lo0 = 0.0f;
    float bin_scale0 = 0.0f;
    float* d_starts0 = nullptr;
    float* d_scales0 = nullptr;
    int32_t* d_bins0 = nullptr;
    float* d_coef0 = nullptr;   // [P_sym, pieces0, 6, 66]
    float* d_pcoef0 = nullptr;  // [P_sym, pieces0, 6, 66]

    // Interaction 0 COMPOSE_SKIP weights: 4 degrees x [S, S*64, 128]
    float* d_C0[4] = {nullptr, nullptr, nullptr, nullptr};

    // Prod 0 + folded readout 0 + inter 1 linear_up / skip_tp
    float* d_w0_factored = nullptr;  // [S, 29, 128]
    float* d_w_ro0 = nullptr;        // [128]
    float* d_W_up_0 = nullptr;       // [128, 128]
    float* d_W_up_1 = nullptr;       // [128, 128]
    float* d_W_skip1_16 = nullptr;   // [S, 128, 16]

    // Interaction 1 spline tables
    SplineGridHost grid1{};
    uint32_t packed8_bias = 0;
    unsigned bias_shift = 0;
    unsigned cubic_bias_shift = 0;
    uint2* d_cubic_packed8_1 = nullptr;   // [P_sym * intervals1, 1280] (8-byte cubic packed)
    uint2* d_packed8_table_1 = nullptr;   // [P_sym * intervals1, 1280]
    float4* d_density_table_1 = nullptr;  // [P_sym * intervals1]

    // Interaction 1 linear block weights
    float* d_W_lin1[4] = {nullptr, nullptr, nullptr, nullptr};  // [256,128], [384,128], [384,128], [256,128]

    // Prod 1 + folded readout 1
    float* d_w1_factored = nullptr;  // [S, 13, 128]
    float* d_W_p1_ro1 = nullptr;     // [128, 16]
    float* d_W_ro1_2 = nullptr;      // [16]

    // ZBL pair repulsion
    int has_zbl = 0;
    float zbl_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float zbl_p = 5.0f;
    float zbl_r_max_max = 0.0f;
    float* d_zbl_r_cut = nullptr;    // [P_sym]
    float* d_zbl_inv_a = nullptr;    // [P_sym]
    float* d_zbl_pref = nullptr;     // [P_sym]

    void load(const loup::Bundle& bundle, const char* path);
    void free_gpu();
};
