#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "error.hpp"

#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err__ = (call);                                                                     \
        if (err__ != cudaSuccess) {                                                                     \
            loup::set_error("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err__));  \
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
            loup::set_error("cuBLAS error at %s:%d: status=%d", __FILE__, __LINE__, (int)st__);         \
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

namespace loup {
class Bundle;
// Largest accepted error of the 8-byte cubic tables (CUBIC_PACKED8), per column relative to the column's largest
// value or slope (d/dr) over all intervals and pairs. The bundles gated so far measure 2.7e-5 (POLAR-1-L) to 1.7e-4
// (OFF23 medium); OMAT-0 small's 1.0e-4 costs about 1.2 of the gate's 3 units. A guard against tables far outside what
// has been gated, not a certificate: the gate stays the acceptance test. Measured over the whole table and,
// separately, over the intervals lying wholly in r >= kPackedPhysicalR (the export certificate's second region, where MD
// edges lie), so large short-range values cannot hide a larger relative error in the physical range.
constexpr double kPackedTableMaxRelErr = 5e-4;
constexpr double kPackedPhysicalR = 1.0;

// max |c0 + c1 s + c2 s^2 + c3 s^3| over s in [lo, hi]: the endpoints and the real roots of the derivative.
inline double max_abs_cubic(double c0, double c1, double c2, double c3, double lo, double hi) {
    auto f = [&](double s) { return std::abs(c0 + s * (c1 + s * (c2 + s * c3))); };
    double m = std::max(f(lo), f(hi));
    const double A = 3.0 * c3, B = 2.0 * c2, C = c1;  // derivative A s^2 + B s + C
    if (A != 0.0) {
        const double disc = B * B - 4.0 * A * C;
        if (disc >= 0.0) {
            const double r = std::sqrt(disc);
            for (const double s : {(-B + r) / (2.0 * A), (-B - r) / (2.0 * A)})
                if (s > lo && s < hi) m = std::max(m, f(s));
        }
    } else if (B != 0.0) {
        const double s = -C / B;
        if (s > lo && s < hi) m = std::max(m, f(s));
    }
    return m;
}
// max |d0 + d1 s + d2 s^2| over s in [lo, hi]: the endpoints and the vertex.
inline double max_abs_quad(double d0, double d1, double d2, double lo, double hi) {
    auto f = [&](double s) { return std::abs(d0 + s * (d1 + s * d2)); };
    double m = std::max(f(lo), f(hi));
    if (d2 != 0.0) {
        const double s = -d1 / (2.0 * d2);
        if (s > lo && s < hi) m = std::max(m, f(s));
    }
    return m;
}

// Per-column maxima of |value|, |d/dr| and of their packing errors, over the whole table and over the physical
// region r >= kPackedPhysicalR (an interval straddling it contributes its part above); rel() is the worst
// error / maximum ratio of either region (see kPackedTableMaxRelErr).
struct PackErrTracker {
    int cols;
    std::vector<double> m[2][4];  // [all, physical][value, slope, value error, slope error]
    explicit PackErrTracker(int n) : cols(n) {
        for (auto& region : m)
            for (auto& v : region) v.assign((size_t)n, 0.0);
    }
    // a: original cubic and d: decoded - original, both in s = t - 1/2 on [-1/2, 1/2]; inv_h: 1 / interval width (A);
    // r0: the interval's start (A). The physical part is s >= (kPackedPhysicalR - r0) inv_h - 1/2.
    void add(int col, double r0, double inv_h, const double a[4], const double d[4]) {
        const double s_phys = std::max(-0.5, (kPackedPhysicalR - r0) * inv_h - 0.5);
        const int regions = s_phys < 0.5 ? 2 : 1;
        for (int r = 0; r < regions; ++r) {
            const double lo = r == 0 ? -0.5 : s_phys;
            const double x[4] = {max_abs_cubic(a[0], a[1], a[2], a[3], lo, 0.5),
                                 inv_h * max_abs_quad(a[1], 2 * a[2], 3 * a[3], lo, 0.5),
                                 max_abs_cubic(d[0], d[1], d[2], d[3], lo, 0.5),
                                 inv_h * max_abs_quad(d[1], 2 * d[2], 3 * d[3], lo, 0.5)};
            for (int k = 0; k < 4; ++k) m[r][k][(size_t)col] = std::max(m[r][k][(size_t)col], x[k]);
        }
    }
    double rel() const {
        double w = 0.0;
        for (const auto& region : m)
            for (int c = 0; c < cols; ++c) {
                if (region[0][(size_t)c] > 0.0) w = std::max(w, region[2][(size_t)c] / region[0][(size_t)c]);
                if (region[1][(size_t)c] > 0.0) w = std::max(w, region[3][(size_t)c] / region[1][(size_t)c]);
            }
        return w;
    }
};
}  // namespace loup

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
    unsigned cubic_bias_shift = 0;
    uint2* d_cubic_packed8_1 = nullptr;   // [P_sym * intervals1, 1280] (8-byte cubic packed)
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
