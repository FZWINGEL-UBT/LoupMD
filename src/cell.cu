// The variable cell's kernels (cell.hpp).
#include "cell.hpp"

#include <cmath>

namespace loup {

__global__ void cell_debug_rate_kernel(double rate, CellDev* __restrict__ cell) {
    const double s = 1.0 + rate;
    for (int i = 0; i < 9; ++i) cell->move[i] = (i % 4 == 0) ? s : 0.0;
    cell->has_move = 1;
}

__global__ void cell_move_positions_kernel(int N, const CellDev* __restrict__ cell, double* __restrict__ pos64) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !cell->has_move) return;
    const double* m = cell->move;
    const double x = pos64[i * 3 + 0], y = pos64[i * 3 + 1], z = pos64[i * 3 + 2];
    pos64[i * 3 + 0] = x * m[0] + y * m[3] + z * m[6];
    pos64[i * 3 + 1] = x * m[1] + y * m[4] + z * m[7];
    pos64[i * 3 + 2] = x * m[2] + y * m[5] + z * m[8];
}

__device__ __forceinline__ void mat3_mul(const double* a, const double* b, double* out) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out[r * 3 + c] = a[r * 3 + 0] * b[0 * 3 + c] + a[r * 3 + 1] * b[1 * 3 + c] + a[r * 3 + 2] * b[2 * 3 + c];
}

__global__ void cell_derive_kernel(CellDev* __restrict__ cell, int* __restrict__ need_rebuild) {
    double h[9];
    if (cell->has_move) {
        mat3_mul(cell->h, cell->move, h);
        for (int i = 0; i < 9; ++i) cell->h[i] = h[i];
        cell->has_move = 0;
    } else {
        for (int i = 0; i < 9; ++i) h[i] = cell->h[i];
    }
    double f[9];
    mat3_mul(cell->ref_inv, h, f);
    double dev2 = 0.0;
    for (int i = 0; i < 9; ++i) {
        cell->f[i] = f[i];
        const double e = f[i] - ((i % 4 == 0) ? 1.0 : 0.0);
        dev2 += e * e;
    }
    cell->thr = 0.5 * (cell->r_list * (1.0 - sqrt(dev2)) - cell->r_max);
    const double det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                       h[2] * (h[3] * h[7] - h[4] * h[6]);
    const double volume = fabs(det);
    cell->volume = volume;
    if (!(volume >= cell->v0 / 20.0 && volume <= 20.0 * cell->v0)) atomicOr(need_rebuild, 8);
}

__global__ void check_skin_cell_kernel(
    int N, int lag, int lag_switch,
    const double* __restrict__ pos64, const double* __restrict__ ref_pos64, const CellDev* __restrict__ cell,
    const unsigned int* __restrict__ max_step_disp2, int* __restrict__ need_rebuild
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double* f = cell->f;
    const double x0 = ref_pos64[i * 3 + 0], y0 = ref_pos64[i * 3 + 1], z0 = ref_pos64[i * 3 + 2];
    const double dx = pos64[i * 3 + 0] - (x0 * f[0] + y0 * f[3] + z0 * f[6]);
    const double dy = pos64[i * 3 + 1] - (x0 * f[1] + y0 * f[4] + z0 * f[7]);
    const double dz = pos64[i * 3 + 2] - (x0 * f[2] + y0 * f[5] + z0 * f[8]);
    const double d2 = dx * dx + dy * dy + dz * dz;
    const unsigned int step_bits = max_step_disp2[0] > max_step_disp2[1] ? max_step_disp2[0] : max_step_disp2[1];
    const double step = sqrt((double)__uint_as_float(step_bits));
    const double limit = cell->thr;
    const double thr = limit - lag * step;
    int flag = 0;
    if (thr <= 0.0 || d2 > thr * thr) flag |= 1;
    if (limit <= 0.0 || d2 > limit * limit) flag |= 2;
    if (lag_switch > 0) {
        const double thr_switch = limit - lag_switch * step;
        if (thr_switch <= 0.0 || d2 > thr_switch * thr_switch) flag |= 4;
    }
    if (flag) atomicOr(need_rebuild, flag);
}

__global__ void sorted_shifts_kernel(
    int E, const int* __restrict__ sort_vals, const int* __restrict__ tmp_shifts,
    int* __restrict__ sx, int* __restrict__ sy, int* __restrict__ sz
) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const int src = sort_vals[e];
    sx[e] = tmp_shifts[src * 3 + 0];
    sy[e] = tmp_shifts[src * 3 + 1];
    sz[e] = tmp_shifts[src * 3 + 2];
}

__global__ void shift_cart_kernel(
    const int* __restrict__ n_edges, const CellDev* __restrict__ cell,
    const int* __restrict__ sx, const int* __restrict__ sy, const int* __restrict__ sz,
    double* __restrict__ shift_cart
) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges[0]) return;
    const double* c = cell->h;
    const double x = (double)sx[e], y = (double)sy[e], z = (double)sz[e];
    shift_cart[e * 3 + 0] = x * c[0] + y * c[3] + z * c[6];
    shift_cart[e * 3 + 1] = x * c[1] + y * c[4] + z * c[7];
    shift_cart[e * 3 + 2] = x * c[2] + y * c[5] + z * c[8];
}

__global__ void positions_times_matrix_kernel(int N, Mat3 mat, double* __restrict__ pos64) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double* m = mat.m;
    const double x = pos64[i * 3 + 0], y = pos64[i * 3 + 1], z = pos64[i * 3 + 2];
    pos64[i * 3 + 0] = x * m[0] + y * m[3] + z * m[6];
    pos64[i * 3 + 1] = x * m[1] + y * m[4] + z * m[7];
    pos64[i * 3 + 2] = x * m[2] + y * m[5] + z * m[8];
}

}  // namespace loup
