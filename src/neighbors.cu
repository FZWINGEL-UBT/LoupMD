#include "kernels_common.cuh"
#include "model.hpp"
#include <cub/cub.cuh>
#include <cmath>

namespace loup {

struct CellMat64 {
    double c[9];     // row-major: row k is lattice vector a_k
    double inv[9];   // row-major inverse: frac = pos @ inv
};

__device__ __forceinline__ int floor_div(int a, int b) {
    int q = a / b;
    int r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) --q;
    return q;
}

__device__ __forceinline__ uint32_t float_to_ordered_uint(float f) {
    uint32_t u = __float_as_uint(f);
    uint32_t mask = -int32_t(u >> 31) | 0x80000000u;
    return u ^ mask;
}

__global__ void wrap_and_bin_kernel(
    int N, CellMat64 cell, int nx, int ny, int nz,
    const double* __restrict__ pos64,
    double* __restrict__ pos_wrapped,
    int* __restrict__ atom_wrap,
    int* __restrict__ atom_bin,
    int* __restrict__ bin_counts
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double x = pos64[i * 3 + 0], y = pos64[i * 3 + 1], z = pos64[i * 3 + 2];
    double fx = x * cell.inv[0] + y * cell.inv[3] + z * cell.inv[6];
    double fy = x * cell.inv[1] + y * cell.inv[4] + z * cell.inv[7];
    double fz = x * cell.inv[2] + y * cell.inv[5] + z * cell.inv[8];
    int wx = (int)floor(fx);
    int wy = (int)floor(fy);
    int wz = (int)floor(fz);
    fx -= (double)wx;
    fy -= (double)wy;
    fz -= (double)wz;
    // Handle tiny negative rounding like -1e-16 -> wrap
    if (fx >= 1.0) { fx -= 1.0; wx += 1; } else if (fx < 0.0) { fx += 1.0; wx -= 1; }
    if (fy >= 1.0) { fy -= 1.0; wy += 1; } else if (fy < 0.0) { fy += 1.0; wy -= 1; }
    if (fz >= 1.0) { fz -= 1.0; wz += 1; } else if (fz < 0.0) { fz += 1.0; wz -= 1; }

    pos_wrapped[i * 3 + 0] = fx * cell.c[0] + fy * cell.c[3] + fz * cell.c[6];
    pos_wrapped[i * 3 + 1] = fx * cell.c[1] + fy * cell.c[4] + fz * cell.c[7];
    pos_wrapped[i * 3 + 2] = fx * cell.c[2] + fy * cell.c[5] + fz * cell.c[8];
    atom_wrap[i * 3 + 0] = wx;
    atom_wrap[i * 3 + 1] = wy;
    atom_wrap[i * 3 + 2] = wz;

    int bx = min(nx - 1, max(0, (int)(fx * nx)));
    int by = min(ny - 1, max(0, (int)(fy * ny)));
    int bz = min(nz - 1, max(0, (int)(fz * nz)));
    int bid = (bx * ny + by) * nz + bz;
    atom_bin[i] = bid;
    atomicAdd(bin_counts + bid, 1);
}

__device__ __forceinline__ unsigned long long spread_bits16(unsigned v) {
    unsigned long long x = v & 0xFFFFu;
    x = (x | (x << 16)) & 0x0000FF0000FFull;
    x = (x | (x << 8)) & 0x00F00F00F00Full;
    x = (x | (x << 4)) & 0x0C30C30C30C3ull;
    x = (x | (x << 2)) & 0x249249249249ull;
    return x;
}

// Atom order key: species in the top bits (keeps the species groups the GEMMs rely on), then the Morton code of the
// atom's cell-list bin, so atoms close in space are close in memory (sender rows and sender-gradient atomics of an
// edge tile then stay in L2 for large or shuffled systems).
__global__ void atom_order_keys_kernel(
    int N, int nx, int ny, int nz,
    const int* __restrict__ species,
    const int* __restrict__ atom_bin,
    unsigned long long* __restrict__ keys,
    int* __restrict__ vals
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int bid = atom_bin[i];
    const unsigned bz = (unsigned)(bid % nz);
    const unsigned by = (unsigned)((bid / nz) % ny);
    const unsigned bx = (unsigned)(bid / (ny * nz));
    const unsigned long long m = spread_bits16(min(bx, 0xFFFFu)) | (spread_bits16(min(by, 0xFFFFu)) << 1) |
                                 (spread_bits16(min(bz, 0xFFFFu)) << 2);
    keys[i] = ((unsigned long long)species[i] << 48) | m;
    vals[i] = i;
}

// dst[i, :] = src[order[i], :] for a row-major [n, width] float64 array.
__global__ void gather_rows_f64_kernel(int n, int width, const int* __restrict__ order,
                                       const double* __restrict__ src, double* __restrict__ dst) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n * width) return;
    const int i = k / width, c = k - i * width;
    dst[k] = src[(int64_t)order[i] * width + c];
}

__global__ void scatter_bins_kernel(
    int N,
    const double* __restrict__ pos_wrapped,
    const int* __restrict__ atom_wrap,
    const int* __restrict__ species,
    const int* __restrict__ atom_bin,
    int* __restrict__ bin_cursor,
    int* __restrict__ bin_atom_id,
    double* __restrict__ bin_pos,
    float4* __restrict__ bin_pos_f32,
    int* __restrict__ bin_wrap,
    int* __restrict__ bin_sp
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int bid = atom_bin[i];
    int slot = atomicAdd(bin_cursor + bid, 1);
    const double px = pos_wrapped[i * 3 + 0];
    const double py = pos_wrapped[i * 3 + 1];
    const double pz = pos_wrapped[i * 3 + 2];
    bin_atom_id[slot] = i;
    bin_pos[slot * 3 + 0] = px;
    bin_pos[slot * 3 + 1] = py;
    bin_pos[slot * 3 + 2] = pz;
    bin_pos_f32[slot] = make_float4((float)px, (float)py, (float)pz, 0.0f);
    bin_wrap[slot * 3 + 0] = atom_wrap[i * 3 + 0];
    bin_wrap[slot * 3 + 1] = atom_wrap[i * 3 + 1];
    bin_wrap[slot * 3 + 2] = atom_wrap[i * 3 + 2];
    bin_sp[slot] = species[i];
}

__global__ void count_neighbors_kernel(
    int N, CellMat64 cell, int nx, int ny, int nz,
    int rx, int ry, int rz, double r_list2, float r_lo2_f32, float r_hi2_f32,
    const double* __restrict__ pos_wrapped,
    const int* __restrict__ atom_bin,
    const int* __restrict__ bin_offsets,
    const int* __restrict__ bin_atom_id,
    const double* __restrict__ bin_pos,
    const float4* __restrict__ bin_pos_f32,
    int* __restrict__ deg
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double xi = pos_wrapped[i * 3 + 0];
    double yi = pos_wrapped[i * 3 + 1];
    double zi = pos_wrapped[i * 3 + 2];
    float fxi = (float)xi, fyi = (float)yi, fzi = (float)zi;
    int bid = atom_bin[i];
    int bxi = bid / (ny * nz);
    int rem = bid - bxi * (ny * nz);
    int byi = rem / nz;
    int bzi = rem - byi * nz;

    int count = 0;
    for (int dx = -rx; dx <= rx; ++dx) {
        int raw_x = bxi + dx;
        int ix = floor_div(raw_x, nx);
        int bxj = raw_x - ix * nx;
        double sx_x = (double)ix * cell.c[0];
        double sx_y = (double)ix * cell.c[1];
        double sx_z = (double)ix * cell.c[2];
        for (int dy = -ry; dy <= ry; ++dy) {
            int raw_y = byi + dy;
            int iy = floor_div(raw_y, ny);
            int byj = raw_y - iy * ny;
            double sxy_x = sx_x + (double)iy * cell.c[3];
            double sxy_y = sx_y + (double)iy * cell.c[4];
            double sxy_z = sx_z + (double)iy * cell.c[5];
            for (int dz = -rz; dz <= rz; ++dz) {
                int raw_z = bzi + dz;
                int iz = floor_div(raw_z, nz);
                int bzj = raw_z - iz * nz;
                double s_x = sxy_x + (double)iz * cell.c[6];
                double s_y = sxy_y + (double)iz * cell.c[7];
                double s_z = sxy_z + (double)iz * cell.c[8];
                float fs_x = (float)s_x, fs_y = (float)s_y, fs_z = (float)s_z;
                int nbid = (bxj * ny + byj) * nz + bzj;
                int b0 = bin_offsets[nbid];
                int b1 = bin_offsets[nbid + 1];
                bool zero_img = (ix == 0 && iy == 0 && iz == 0);
                for (int slot = b0; slot < b1; ++slot) {
                    const float4 pj = __ldg(bin_pos_f32 + slot);
                    const float fx = fxi - (pj.x + fs_x);
                    const float fy = fyi - (pj.y + fs_y);
                    const float fz = fzi - (pj.z + fs_z);
                    const float fr2 = fx * fx + fy * fy + fz * fz;
                    if (fr2 >= r_hi2_f32) continue;
                    if (zero_img && bin_atom_id[slot] == i) continue;
                    if (fr2 < r_lo2_f32) {
                        ++count;
                    } else {
                        double vx = xi - (bin_pos[slot * 3 + 0] + s_x);
                        double vy = yi - (bin_pos[slot * 3 + 1] + s_y);
                        double vz = zi - (bin_pos[slot * 3 + 2] + s_z);
                        if (vx * vx + vy * vy + vz * vz < r_list2) ++count;
                    }
                }
            }
        }
    }
    deg[i] = count;
}

__global__ void fill_neighbors_kernel(
    int N, int S, CellMat64 cell, int nx, int ny, int nz,
    int rx, int ry, int rz, double r_list2, float r_lo2_f32, float r_hi2_f32,
    const double* __restrict__ pos_wrapped,
    const int* __restrict__ atom_wrap,
    const int* __restrict__ species,
    const int* __restrict__ atom_bin,
    const int* __restrict__ bin_offsets,
    const int* __restrict__ bin_atom_id,
    const double* __restrict__ bin_pos,
    const float4* __restrict__ bin_pos_f32,
    const int* __restrict__ bin_wrap,
    const int* __restrict__ bin_sp,
    const int* __restrict__ row_ptr,
    int* __restrict__ tmp_snd,
    int* __restrict__ tmp_rcv,
    int* __restrict__ tmp_shifts,
    int* __restrict__ tmp_pairs,
    uint32_t* __restrict__ sort_keys,
    int* __restrict__ sort_vals
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double xi = pos_wrapped[i * 3 + 0];
    double yi = pos_wrapped[i * 3 + 1];
    double zi = pos_wrapped[i * 3 + 2];
    float fxi = (float)xi, fyi = (float)yi, fzi = (float)zi;
    int wi_x = atom_wrap[i * 3 + 0];
    int wi_y = atom_wrap[i * 3 + 1];
    int wi_z = atom_wrap[i * 3 + 2];
    int sp_i = species[i];

    int bid = atom_bin[i];
    int bxi = bid / (ny * nz);
    int rem = bid - bxi * (ny * nz);
    int byi = rem / nz;
    int bzi = rem - byi * nz;

    int cursor = row_ptr[i];
    const int row_end = row_ptr[i + 1];
    for (int dx = -rx; dx <= rx; ++dx) {
        int raw_x = bxi + dx;
        int ix = floor_div(raw_x, nx);
        int bxj = raw_x - ix * nx;
        double sx_x = (double)ix * cell.c[0];
        double sx_y = (double)ix * cell.c[1];
        double sx_z = (double)ix * cell.c[2];
        for (int dy = -ry; dy <= ry; ++dy) {
            int raw_y = byi + dy;
            int iy = floor_div(raw_y, ny);
            int byj = raw_y - iy * ny;
            double sxy_x = sx_x + (double)iy * cell.c[3];
            double sxy_y = sx_y + (double)iy * cell.c[4];
            double sxy_z = sx_z + (double)iy * cell.c[5];
            for (int dz = -rz; dz <= rz; ++dz) {
                int raw_z = bzi + dz;
                int iz = floor_div(raw_z, nz);
                int bzj = raw_z - iz * nz;
                double s_x = sxy_x + (double)iz * cell.c[6];
                double s_y = sxy_y + (double)iz * cell.c[7];
                double s_z = sxy_z + (double)iz * cell.c[8];
                float fs_x = (float)s_x, fs_y = (float)s_y, fs_z = (float)s_z;
                int nbid = (bxj * ny + byj) * nz + bzj;
                int b0 = bin_offsets[nbid];
                int b1 = bin_offsets[nbid + 1];
                bool zero_img = (ix == 0 && iy == 0 && iz == 0);
                for (int slot = b0; slot < b1; ++slot) {
                    const float4 pj = __ldg(bin_pos_f32 + slot);
                    const float fx = fxi - (pj.x + fs_x);
                    const float fy = fyi - (pj.y + fs_y);
                    const float fz = fzi - (pj.z + fs_z);
                    const float fr2 = fx * fx + fy * fy + fz * fz;
                    if (fr2 >= r_hi2_f32) continue;
                    int j = bin_atom_id[slot];
                    if (zero_img && j == i) continue;
                    float r;
                    if (fr2 < r_lo2_f32) {
                        r = sqrtf(fr2);
                    } else {
                        double vx = xi - (bin_pos[slot * 3 + 0] + s_x);
                        double vy = yi - (bin_pos[slot * 3 + 1] + s_y);
                        double vz = zi - (bin_pos[slot * 3 + 2] + s_z);
                        double r2 = vx * vx + vy * vy + vz * vz;
                        if (!(r2 < r_list2)) continue;
                        r = (float)sqrt(r2);
                    }
                    if (cursor < row_end) {
                        int shx = bin_wrap[slot * 3 + 0] - wi_x - ix;
                        int shy = bin_wrap[slot * 3 + 1] - wi_y - iy;
                        int shz = bin_wrap[slot * 3 + 2] - wi_z - iz;
                        int sp_j = bin_sp[slot];
                        int p = sym_pair(sp_j, sp_i, S);
                        tmp_snd[cursor] = j;
                        tmp_rcv[cursor] = i;
                        tmp_shifts[cursor * 3 + 0] = shx;
                        tmp_shifts[cursor * 3 + 1] = shy;
                        tmp_shifts[cursor * 3 + 2] = shz;
                        tmp_pairs[cursor] = p;
                        sort_keys[cursor] = ((uint32_t)p << 19) | (__float_as_uint(r) >> 12);
                        sort_vals[cursor] = cursor;
                        ++cursor;
                    }
                }
            }
        }
    }
}

// Gathers the sorted skin edges and turns each integer image shift into its Cartesian offset shift @ cell (float64),
// so the per-step edge-vector kernel does six float64 additions instead of the full shift-cell product.
__global__ void gather_sorted_edges_kernel(
    int E, CellMat64 cell,
    const int* __restrict__ sort_vals,
    const int* __restrict__ tmp_snd,
    const int* __restrict__ tmp_rcv,
    const int* __restrict__ tmp_shifts,
    const int* __restrict__ tmp_pairs,
    int* __restrict__ senders,
    int* __restrict__ receivers,
    double* __restrict__ shift_cart,
    int* __restrict__ pairs
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    int src = sort_vals[e];
    senders[e] = tmp_snd[src];
    receivers[e] = tmp_rcv[src];
    const double sx = (double)tmp_shifts[src * 3 + 0];
    const double sy = (double)tmp_shifts[src * 3 + 1];
    const double sz = (double)tmp_shifts[src * 3 + 2];
    shift_cart[e * 3 + 0] = sx * cell.c[0] + sy * cell.c[3] + sz * cell.c[6];
    shift_cart[e * 3 + 1] = sx * cell.c[1] + sy * cell.c[4] + sz * cell.c[7];
    shift_cart[e * 3 + 2] = sx * cell.c[2] + sy * cell.c[5] + sz * cell.c[8];
    pairs[e] = tmp_pairs[src];
}

// Flags any row (consecutive edges with the same receiver) whose sender species decreases.
__global__ void check_row_species_order_kernel(int E, const int* __restrict__ senders,
                                               const int* __restrict__ receivers, const int* __restrict__ species,
                                               int* __restrict__ err) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e <= 0 || e >= E) return;
    if (receivers[e] == receivers[e - 1] && species[senders[e]] < species[senders[e - 1]]) *err = 1;
}

__global__ void compute_and_count_active_kernel(
    int N, float r_max,
    const double* __restrict__ pos64,
    const int* __restrict__ row_ptr,
    const int* __restrict__ senders,
    const int* __restrict__ receivers,
    const double* __restrict__ shift_cart,
    float4* __restrict__ tmp_vec_len,
    int* __restrict__ act_deg
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = tid >> 5;
    const int lane = tid & 31;
    if (i >= N) return;
    if (i == 0 && lane == 0) act_deg[N] = 0;
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    const double rx = pos64[i * 3 + 0];
    const double ry = pos64[i * 3 + 1];
    const double rz = pos64[i * 3 + 2];
    int count = 0;
    for (int base = eb; base < ee; base += 32) {
        const int e = base + lane;
        bool act = false;
        if (e < ee) {
            const int s = senders[e];
            const double vx = (rx - pos64[s * 3 + 0]) + shift_cart[e * 3 + 0];
            const double vy = (ry - pos64[s * 3 + 1]) + shift_cart[e * 3 + 1];
            const double vz = (rz - pos64[s * 3 + 2]) + shift_cart[e * 3 + 2];
            const float fx = (float)vx, fy = (float)vy, fz = (float)vz;
            const float len = sqrtf(fx * fx + fy * fy + fz * fz);
            tmp_vec_len[e] = make_float4(fx, fy, fz, len);
            act = (len < r_max);
        }
        const unsigned mask = __ballot_sync(0xffffffffu, act);
        count += __popc(mask);
    }
    if (lane == 0) act_deg[i] = count;
}

// Compacts the active edges (r < r_max) of each receiver row, in row order, with their spline locator. The SH of an
// edge are not stored: its consumers compute them from act_vec_len where they stage the edge (edge_sh16).
__global__ void scatter_active_edges_kernel(
    int N, float r_max, SplineGrid grid,
    const int* __restrict__ row_ptr,
    const int* __restrict__ act_row_ptr,
    const int* __restrict__ senders,
    const int* __restrict__ receivers,
    const int* __restrict__ pairs,
    const float4* __restrict__ tmp_vec_len,
    int* __restrict__ act_senders,
    int* __restrict__ act_receivers,
    int* __restrict__ act_pairs,
    float4* __restrict__ act_vec_len,
    int* __restrict__ act_slab,
    float2* __restrict__ act_t_sx
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = tid >> 5;
    const int lane = tid & 31;
    if (i >= N) return;
    const int eb = row_ptr[i];
    const int ee = row_ptr[i + 1];
    int out_cursor = act_row_ptr[i];
    for (int base = eb; base < ee; base += 32) {
        const int e = base + lane;
        bool act = false;
        float4 vl = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (e < ee) {
            vl = __ldg(tmp_vec_len + e);
            act = (vl.w < r_max);
        }
        const unsigned mask = __ballot_sync(0xffffffffu, act);
        if (act) {
            const int dst = out_cursor + __popc(mask & ((1u << lane) - 1u));
            const int p = pairs[e];
            act_senders[dst] = senders[e];
            act_receivers[dst] = receivers[e];
            act_pairs[dst] = p;
            act_vec_len[dst] = vl;
            // Fused spline locator (shared by spline_forward_inter1 and wprime_backward_inter1)
            int iv;
            float t, sx_inv;
            locate_spline(grid, vl.w, iv, t, sx_inv);
            act_slab[dst] = p * grid.total_intervals + iv;
            act_t_sx[dst] = make_float2(t, (vl.w >= grid.starts[0]) ? sx_inv : -sx_inv);
        }
        out_cursor += __popc(mask);
    }
}

// The list built at ref_pos64 holds every pair within r_max + skin then, so it stays complete while every atom has
// moved at most skin/2. Forces computed with it `lag` steps after this check (0: the caller rebuilds before the next
// force call; 3: the pipelined loop, whose next step is already running, plus one step of margin) are safe if the
// threshold is lowered by lag times the largest per-step displacement seen (max_step_disp2, maintained by the
// first-half kernels).
// Bit 1: rebuild now. Bit 2: some atom is already beyond skin/2, so a force call since the last check may have used
// an incomplete list (a "dangerous build", counted in LoupMDStats). Bit 4 (only with lag_switch > 0, the unpipelined
// loop's check): some atom is beyond the threshold a check with lag = lag_switch would use, so the list is too old to
// switch to the pipelined loop without a rebuild. The flag is cleared only by a rebuild, so bit 4 stays set until then.
__global__ void check_skin_kernel(
    int N, double half_skin, int lag, int lag_switch,
    const double* __restrict__ pos64,
    const double* __restrict__ ref_pos64,
    const unsigned int* __restrict__ max_step_disp2,
    int* __restrict__ need_rebuild
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double dx = pos64[i * 3 + 0] - ref_pos64[i * 3 + 0];
    double dy = pos64[i * 3 + 1] - ref_pos64[i * 3 + 1];
    double dz = pos64[i * 3 + 2] - ref_pos64[i * 3 + 2];
    const double d2 = dx * dx + dy * dy + dz * dz;
    const double step = sqrt((double)__uint_as_float(*max_step_disp2));
    const double thr = half_skin - lag * step;
    int flag = 0;
    if (thr <= 0.0 || d2 > thr * thr) flag |= 1;
    if (d2 > half_skin * half_skin) flag |= 2;
    if (lag_switch > 0) {
        const double thr_switch = half_skin - lag_switch * step;
        if (thr_switch <= 0.0 || d2 > thr_switch * thr_switch) flag |= 4;
    }
    if (flag) atomicOr(need_rebuild, flag);
}

void compute_cell_mat(const double* cell_row_major, CellMat64& out, double& out_vol, double heights[3]) {
    for (int i = 0; i < 9; ++i) out.c[i] = cell_row_major[i];
    const double* a = out.c;
    // Cofactors for transpose inverse
    double c00 = a[4] * a[8] - a[5] * a[7];
    double c01 = a[5] * a[6] - a[3] * a[8];
    double c02 = a[3] * a[7] - a[4] * a[6];
    double c10 = a[7] * a[2] - a[8] * a[1];
    double c11 = a[8] * a[0] - a[6] * a[2];
    double c12 = a[6] * a[1] - a[7] * a[0];
    double c20 = a[1] * a[5] - a[2] * a[4];
    double c21 = a[2] * a[3] - a[0] * a[5];
    double c22 = a[0] * a[4] - a[1] * a[3];
    double det = a[0] * c00 + a[1] * c01 + a[2] * c02;
    out_vol = fabs(det);
    double inv_det = 1.0 / det;
    // out.inv = a^{-1} = cof^T / det
    out.inv[0] = c00 * inv_det; out.inv[1] = c10 * inv_det; out.inv[2] = c20 * inv_det;
    out.inv[3] = c01 * inv_det; out.inv[4] = c11 * inv_det; out.inv[5] = c21 * inv_det;
    out.inv[6] = c02 * inv_det; out.inv[7] = c12 * inv_det; out.inv[8] = c22 * inv_det;

    double area0 = sqrt(c00 * c00 + c01 * c01 + c02 * c02);
    double area1 = sqrt(c10 * c10 + c11 * c11 + c12 * c12);
    double area2 = sqrt(c20 * c20 + c21 * c21 + c22 * c22);
    heights[0] = out_vol / area0;
    heights[1] = out_vol / area1;
    heights[2] = out_vol / area2;
}

}  // namespace loup
