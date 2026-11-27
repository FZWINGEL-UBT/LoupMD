#pragma once
// Exact forward, 1st-order backward (dE/dx, dE/dvec), and 2nd-order backward (double-backprop for force loss)
// for layer-0 first_hidden (S0), general equivariant convolutions (lmax_x in {0, 1, 2}, any C_x), and frozen ZBL.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "kernels_common.cuh"
#include "generated_sh16.cuh"
#include "mace_ft_sh_cg.cuh"
#include "mace_ft_types.hpp"

namespace loup_ft {

__device__ __forceinline__ float eval_cubic(const float4& cf, float t) {
    return fmaf(t, fmaf(t, fmaf(t, cf.w, cf.z), cf.y), cf.x);
}

__device__ __forceinline__ float eval_cubic_deriv(const float4& cf, float t, float sx_inv) {
    return fmaf(t, fmaf(3.0f * cf.w, t, 2.0f * cf.z), cf.y) * sx_inv;
}

// Compute unit vector (x,y,z) in SH frame, radial projection v_r = bar_vec . u, and
// directional SH derivative dY_v[16] = bar_vec . grad_vec Y(vec/|vec|).
__device__ __forceinline__ void edge_geom_and_jvp(
    const float4& vl, int sh_perm, float bx, float by, float bz,
    float Y[16], float dY_v[16], float& v_r) {
    const float ux = sh_perm ? vl.y : vl.x;
    const float uy = sh_perm ? vl.z : vl.y;
    const float uz = sh_perm ? vl.x : vl.z;
    const float vx = sh_perm ? by : bx;
    const float vy = sh_perm ? bz : by;
    const float vz = sh_perm ? bx : bz;

    const float dd = vl.w > loup::EPS_SH ? vl.w : loup::EPS_SH;
    const float inv_d = 1.0f / dd;
    const float x = ux * inv_d, y = uy * inv_d, z = uz * inv_d;

    loup::edge_sh16(ux, uy, uz, vl.w, reinterpret_cast<float4*>(Y));
    v_r = vx * x + vy * y + vz * z;
    // Unit-sphere tangent: du = (v - (v.u)*u) / d
    const float dux = (vx - v_r * x) * inv_d;
    const float duy = (vy - v_r * y) * inv_d;
    const float duz = (vz - v_r * z) * inv_d;
    sh16_jvp(x, y, z, dux, duy, duz, dY_v);
}

// Convert (er = dE/dr, dY[16] = dE/dY) into dE/dvec in world frame and add to edge_dvec[e].
__device__ __forceinline__ void add_edge_dvec(
    const float4& vl, int sh_perm, float er, const float dY[16], float* edge_dvec_e) {
    const float ux = sh_perm ? vl.y : vl.x;
    const float uy = sh_perm ? vl.z : vl.y;
    const float uz = sh_perm ? vl.x : vl.z;
    const float dd = vl.w > loup::EPS_SH ? vl.w : loup::EPS_SH;
    const float inv_d = 1.0f / dd;
    const float x = ux * inv_d, y = uy * inv_d, z = uz * inv_d;
    float gx, gy, gz;
    ::sh16_grad(x, y, z, dY, gx, gy, gz);
    const float dot = x * gx + y * gy + z * gz;
    float fx = x * er + (gx - x * dot) * inv_d;
    float fy = y * er + (gy - y * dot) * inv_d;
    float fz = z * er + (gz - z * dot) * inv_d;
    if (sh_perm) { const float tx = fx, ty = fy, tz = fz; fx = tz; fy = tx; fz = ty; }
    edge_dvec_e[0] += fx;
    edge_dvec_e[1] += fy;
    edge_dvec_e[2] += fz;
}

// ===========================================================================
// 1. Layer 0 first_hidden == 1 (S0[l]: [N, 2l+1, S*Jp], D: [N])
// ===========================================================================

// Forward: S0[rcv, m, s_snd*Jp + j] += R_j(r_e) * Y_m(u_e); D[rcv] += rho(r_e)
__global__ void ft_first_fwd_kernel(
    int E, int S, int I, int J, int Jp, SplineGrid grid, int sh_perm, int pair_sym,
    const int* __restrict__ senders, const int* __restrict__ receivers, const float4* __restrict__ vec_len,
    const int* __restrict__ species, const float4* __restrict__ spline, int has_density,
    float* __restrict__ S0_0, float* __restrict__ S0_1, float* __restrict__ S0_2, float* __restrict__ S0_3,
    float* __restrict__ D) {
    const int e = blockIdx.x;
    const int j = threadIdx.x;
    if (e >= E) return;

    __shared__ float s_Y[16];
    __shared__ int s_iv, s_pr, s_snd_sp, s_rcv;
    __shared__ float s_t;
    if (j == 0) {
        const int snd = senders[e], rcv = receivers[e];
        const int ss = species[snd], sa = species[rcv];
        const float4 vl = vec_len[e];
        int iv; float t, sxi;
        locate_spline(grid, vl.w, iv, t, sxi);
        s_iv = iv; s_t = t; s_snd_sp = ss; s_rcv = rcv;
        s_pr = loup::table_pair(pair_sym, ss, sa, S);
        if (sh_perm) loup::edge_sh16(vl.y, vl.z, vl.x, vl.w, reinterpret_cast<float4*>(s_Y));
        else         loup::edge_sh16(vl.x, vl.y, vl.z, vl.w, reinterpret_cast<float4*>(s_Y));
        if (has_density && D) {
            const float4 dc = spline[((int64_t)s_pr * I + iv) * J + Jp];
            atomicAdd(D + rcv, eval_cubic(dc, t));
        }
    }
    __syncthreads();

    const int K = S * Jp;
    const int rcv = s_rcv, ss = s_snd_sp;
    const float4* row = spline + ((int64_t)s_pr * I + s_iv) * J;
    for (int col = j; col < Jp; col += blockDim.x) {
        const float rv = eval_cubic(row[col], s_t);
        const int k = ss * Jp + col;
        atomicAdd(S0_0 + (int64_t)rcv * K + k, rv * s_Y[0]);
        #pragma unroll
        for (int m = 0; m < 3; ++m) atomicAdd(S0_1 + ((int64_t)rcv * 3 + m) * K + k, rv * s_Y[1 + m]);
        #pragma unroll
        for (int m = 0; m < 5; ++m) atomicAdd(S0_2 + ((int64_t)rcv * 5 + m) * K + k, rv * s_Y[4 + m]);
        #pragma unroll
        for (int m = 0; m < 7; ++m) atomicAdd(S0_3 + ((int64_t)rcv * 7 + m) * K + k, rv * s_Y[9 + m]);
    }
}

// 1st-order backward: edge_dvec[e] += sum_{m,j} dS0[rcv, m, ss*Jp+j] * grad_vec(R_j * Y_m) + dD[rcv] * grad_vec(rho)
__global__ void ft_first_bwd_kernel(
    int E, int S, int I, int J, int Jp, SplineGrid grid, int sh_perm, int pair_sym,
    const int* __restrict__ senders, const int* __restrict__ receivers, const float4* __restrict__ vec_len,
    const int* __restrict__ species, const float4* __restrict__ spline, int has_density,
    const float* __restrict__ dS0_0, const float* __restrict__ dS0_1,
    const float* __restrict__ dS0_2, const float* __restrict__ dS0_3,
    const float* __restrict__ dD, float* __restrict__ edge_dvec) {
    const int e = blockIdx.x;
    const int tid = threadIdx.x;
    if (e >= E) return;

    __shared__ float s_Y[16];
    __shared__ float s_dY[16];
    __shared__ float s_er;
    __shared__ int s_iv, s_pr, s_snd_sp, s_rcv;
    __shared__ float s_t, s_sxi;

    if (tid < 16) s_dY[tid] = 0.0f;
    if (tid == 0) {
        const int snd = senders[e], rcv = receivers[e];
        const int ss = species[snd], sa = species[rcv];
        const float4 vl = vec_len[e];
        int iv; float t, sxi;
        locate_spline(grid, vl.w, iv, t, sxi);
        s_iv = iv; s_t = t; s_sxi = sxi; s_snd_sp = ss; s_rcv = rcv;
        s_pr = loup::table_pair(pair_sym, ss, sa, S);
        if (sh_perm) loup::edge_sh16(vl.y, vl.z, vl.x, vl.w, reinterpret_cast<float4*>(s_Y));
        else         loup::edge_sh16(vl.x, vl.y, vl.z, vl.w, reinterpret_cast<float4*>(s_Y));
        float er0 = 0.0f;
        if (has_density && dD) {
            const float4 dc = spline[((int64_t)s_pr * I + iv) * J + Jp];
            er0 = dD[rcv] * eval_cubic_deriv(dc, t, sxi);
        }
        s_er = er0;
    }
    __syncthreads();

    const int K = S * Jp;
    const int rcv = s_rcv, ss = s_snd_sp;
    const float4* row = spline + ((int64_t)s_pr * I + s_iv) * J;

    float er_loc = 0.0f;
    float dY_loc[16];
    #pragma unroll
    for (int m = 0; m < 16; ++m) dY_loc[m] = 0.0f;

    for (int col = tid; col < Jp; col += blockDim.x) {
        const float4 cf = row[col];
        const float rv = eval_cubic(cf, s_t);
        const float dr = eval_cubic_deriv(cf, s_t, s_sxi);
        const int k = ss * Jp + col;
        float g[16];
        g[0] = dS0_0[(int64_t)rcv * K + k];
        #pragma unroll
        for (int m = 0; m < 3; ++m) g[1 + m] = dS0_1[((int64_t)rcv * 3 + m) * K + k];
        #pragma unroll
        for (int m = 0; m < 5; ++m) g[4 + m] = dS0_2[((int64_t)rcv * 5 + m) * K + k];
        #pragma unroll
        for (int m = 0; m < 7; ++m) g[9 + m] = dS0_3[((int64_t)rcv * 7 + m) * K + k];
        float dot_gY = 0.0f;
        #pragma unroll
        for (int m = 0; m < 16; ++m) {
            dot_gY = fmaf(g[m], s_Y[m], dot_gY);
            dY_loc[m] = fmaf(g[m], rv, dY_loc[m]);
        }
        er_loc = fmaf(dot_gY, dr, er_loc);
    }

    atomicAdd(&s_er, er_loc);
    #pragma unroll
    for (int m = 0; m < 16; ++m) if (dY_loc[m] != 0.0f) atomicAdd(&s_dY[m], dY_loc[m]);
    __syncthreads();

    if (tid == 0) add_edge_dvec(vec_len[e], sh_perm, s_er, s_dY, edge_dvec + (int64_t)e * 3);
}

// 2nd-order backward for layer 0 first_hidden:
//   Given bar_edge_dvec[e, 3]:
//   tan_S0[rcv, m, ss*Jp+j] += v_r * R'_j * Y_m + R_j * dY_v[m]
//   tan_D[rcv]              += v_r * rho'
__global__ void ft_first_bwd2_kernel(
    int E, int S, int I, int J, int Jp, SplineGrid grid, int sh_perm, int pair_sym,
    const int* __restrict__ senders, const int* __restrict__ receivers, const float4* __restrict__ vec_len,
    const int* __restrict__ species, const float4* __restrict__ spline, int has_density,
    const float* __restrict__ bar_edge_dvec,
    float* __restrict__ tan_S0_0, float* __restrict__ tan_S0_1,
    float* __restrict__ tan_S0_2, float* __restrict__ tan_S0_3,
    float* __restrict__ tan_D) {
    const int e = blockIdx.x;
    const int tid = threadIdx.x;
    if (e >= E) return;

    __shared__ float s_Y[16], s_dYv[16];
    __shared__ int s_iv, s_pr, s_snd_sp, s_rcv;
    __shared__ float s_t, s_sxi, s_vr;

    if (tid == 0) {
        const int snd = senders[e], rcv = receivers[e];
        const int ss = species[snd], sa = species[rcv];
        const float4 vl = vec_len[e];
        int iv; float t, sxi;
        locate_spline(grid, vl.w, iv, t, sxi);
        s_iv = iv; s_t = t; s_sxi = sxi; s_snd_sp = ss; s_rcv = rcv;
        s_pr = loup::table_pair(pair_sym, ss, sa, S);
        float vr = 0.0f;
        if (bar_edge_dvec) {
            edge_geom_and_jvp(vl, sh_perm,
                              bar_edge_dvec[3 * e + 0], bar_edge_dvec[3 * e + 1], bar_edge_dvec[3 * e + 2],
                              s_Y, s_dYv, vr);
        } else {
            if (sh_perm) loup::edge_sh16(vl.y, vl.z, vl.x, vl.w, reinterpret_cast<float4*>(s_Y));
            else         loup::edge_sh16(vl.x, vl.y, vl.z, vl.w, reinterpret_cast<float4*>(s_Y));
            for (int m = 0; m < 16; ++m) s_dYv[m] = 0.0f;
        }
        s_vr = vr;
        if (has_density && tan_D) {
            const float4 dc = spline[((int64_t)s_pr * I + iv) * J + Jp];
            atomicAdd(tan_D + rcv, vr * eval_cubic_deriv(dc, t, sxi));
        }
    }
    __syncthreads();

    const int K = S * Jp;
    const int rcv = s_rcv, ss = s_snd_sp;
    const float4* row = spline + ((int64_t)s_pr * I + s_iv) * J;
    for (int col = tid; col < Jp; col += blockDim.x) {
        const float4 cf = row[col];
        const float rv = eval_cubic(cf, s_t);
        const float vdr = s_vr * eval_cubic_deriv(cf, s_t, s_sxi);
        const int k = ss * Jp + col;
        atomicAdd(tan_S0_0 + (int64_t)rcv * K + k, fmaf(vdr, s_Y[0], rv * s_dYv[0]));
        #pragma unroll
        for (int m = 0; m < 3; ++m)
            atomicAdd(tan_S0_1 + ((int64_t)rcv * 3 + m) * K + k, fmaf(vdr, s_Y[1 + m], rv * s_dYv[1 + m]));
        #pragma unroll
        for (int m = 0; m < 5; ++m)
            atomicAdd(tan_S0_2 + ((int64_t)rcv * 5 + m) * K + k, fmaf(vdr, s_Y[4 + m], rv * s_dYv[4 + m]));
        #pragma unroll
        for (int m = 0; m < 7; ++m)
            atomicAdd(tan_S0_3 + ((int64_t)rcv * 7 + m) * K + k, fmaf(vdr, s_Y[9 + m], rv * s_dYv[9 + m]));
    }
}

// ===========================================================================
// 2. General equivariant convolution (lmax_x in {0, 1, 2}, any C_x)
// ===========================================================================

template <int LX>
__device__ __forceinline__ const PathMeta* get_paths(int& n_paths) {
    if constexpr (LX == 0) { n_paths = 4;  return kPaths_L0; }
    if constexpr (LX == 1) { n_paths = 10; return kPaths_L1; }
    if constexpr (LX == 2) { n_paths = 17; return kPaths_L2; }
}

template <int LX>
__device__ __forceinline__ const CGEntry* get_cg() {
    if constexpr (LX == 0) return kCG_L0;
    if constexpr (LX == 1) return kCG_L1;
    if constexpr (LX == 2) return kCG_L2;
}

// Forward: mid[l3][rcv, m3, slot*C + c] += sum_{e=(rcv,snd)} sum_{m1,m2} CG * R_{e,p,c} * Y[l2^2+m2] * x[l1][snd, m1, c]
template <int LX>
__global__ void ft_gconv_fwd_kernel(
    int E, int C, int S, int I, int Q1, SplineGrid grid, int sh_perm, int pair_sym,
    const int* __restrict__ senders, const int* __restrict__ receivers, const float4* __restrict__ vec_len,
    const int* __restrict__ species, const float4* __restrict__ spline, int has_density,
    const float* __restrict__ x0, const float* __restrict__ x1, const float* __restrict__ x2,
    float* __restrict__ mid0, float* __restrict__ mid1, float* __restrict__ mid2, float* __restrict__ mid3,
    float* __restrict__ D) {
    const int e = blockIdx.x;
    const int tid = threadIdx.x;
    if (e >= E) return;

    __shared__ float s_Y[16];
    __shared__ int s_iv, s_pr, s_snd, s_rcv;
    __shared__ float s_t;

    if (tid == 0) {
        const int snd = senders[e], rcv = receivers[e];
        const float4 vl = vec_len[e];
        int iv; float t, sxi;
        locate_spline(grid, vl.w, iv, t, sxi);
        s_iv = iv; s_t = t; s_snd = snd; s_rcv = rcv;
        s_pr = loup::table_pair(pair_sym, species[snd], species[rcv], S);
        if (sh_perm) loup::edge_sh16(vl.y, vl.z, vl.x, vl.w, reinterpret_cast<float4*>(s_Y));
        else         loup::edge_sh16(vl.x, vl.y, vl.z, vl.w, reinterpret_cast<float4*>(s_Y));
        if (has_density && D) {
            const float4 dc = spline[((int64_t)s_pr * I + iv) * Q1 + (Q1 - 1)];
            atomicAdd(D + rcv, eval_cubic(dc, t));
        }
    }
    __syncthreads();

    int n_paths = 0;
    const PathMeta* paths = get_paths<LX>(n_paths);
    const CGEntry* cg = get_cg<LX>();
    const float* xs[3] = {x0, x1, x2};
    float* mids[4] = {mid0, mid1, mid2, mid3};
    static constexpr int n_l3[4] = {
        LX == 0 ? 1 : (LX == 1 ? 2 : 3),
        LX == 0 ? 1 : (LX == 1 ? 3 : 5),
        LX == 0 ? 1 : (LX == 1 ? 3 : 5),
        LX == 0 ? 1 : (LX == 1 ? 2 : 4)
    };

    const float4* sp_row = spline + ((int64_t)s_pr * I + s_iv) * Q1;
    const int snd = s_snd, rcv = s_rcv;

    for (int c = tid; c < C; c += blockDim.x) {
        // Load sender x for this channel c
        float xv[9];  // l=0 (1), l=1 (3), l=2 (5) -> 9 total
        xv[0] = xs[0][(int64_t)snd * C + c];
        if constexpr (LX >= 1) {
            #pragma unroll
            for (int m = 0; m < 3; ++m) xv[1 + m] = xs[1][((int64_t)snd * 3 + m) * C + c];
        }
        if constexpr (LX >= 2) {
            #pragma unroll
            for (int m = 0; m < 5; ++m) xv[4 + m] = xs[2][((int64_t)snd * 5 + m) * C + c];
        }

        for (int p = 0; p < n_paths; ++p) {
            const PathMeta pm = paths[p];
            const float R = eval_cubic(sp_row[p * C + c], s_t);
            const int x_off = pm.l1 * pm.l1;
            const int y_off = pm.l2 * pm.l2;
            const int d3 = 2 * pm.l3 + 1;
            float out[7] = {0, 0, 0, 0, 0, 0, 0};
            for (int k = 0; k < pm.cg_len; ++k) {
                const CGEntry ce = cg[pm.cg_off + k];
                out[ce.m3] = fmaf(ce.val, xv[x_off + ce.m1] * s_Y[y_off + ce.m2], out[ce.m3]);
            }
            float* dst = mids[pm.l3];
            const int row_w = n_l3[pm.l3] * C;
            for (int m3 = 0; m3 < d3; ++m3) {
                atomicAdd(dst + ((int64_t)rcv * d3 + m3) * row_w + pm.l3_slot * C + c, R * out[m3]);
            }
        }
    }
}

// 1st-order backward: computes dx[snd] and accumulates edge_dvec[e]
template <int LX>
__global__ void ft_gconv_bwd_kernel(
    int E, int C, int S, int I, int Q1, SplineGrid grid, int sh_perm, int pair_sym,
    const int* __restrict__ senders, const int* __restrict__ receivers, const float4* __restrict__ vec_len,
    const int* __restrict__ species, const float4* __restrict__ spline, int has_density,
    const float* __restrict__ x0, const float* __restrict__ x1, const float* __restrict__ x2,
    const float* __restrict__ dm0, const float* __restrict__ dm1, const float* __restrict__ dm2, const float* __restrict__ dm3,
    const float* __restrict__ dD,
    float* __restrict__ dx0, float* __restrict__ dx1, float* __restrict__ dx2,
    float* __restrict__ edge_dvec) {
    const int e = blockIdx.x;
    const int tid = threadIdx.x;
    if (e >= E) return;

    __shared__ float s_Y[16], s_dY[16];
    __shared__ float s_er;
    __shared__ int s_iv, s_pr, s_snd, s_rcv;
    __shared__ float s_t, s_sxi;

    if (tid < 16) s_dY[tid] = 0.0f;
    if (tid == 0) {
        const int snd = senders[e], rcv = receivers[e];
        const float4 vl = vec_len[e];
        int iv; float t, sxi;
        locate_spline(grid, vl.w, iv, t, sxi);
        s_iv = iv; s_t = t; s_sxi = sxi; s_snd = snd; s_rcv = rcv;
        s_pr = loup::table_pair(pair_sym, species[snd], species[rcv], S);
        if (sh_perm) loup::edge_sh16(vl.y, vl.z, vl.x, vl.w, reinterpret_cast<float4*>(s_Y));
        else         loup::edge_sh16(vl.x, vl.y, vl.z, vl.w, reinterpret_cast<float4*>(s_Y));
        float er0 = 0.0f;
        if (has_density && dD) {
            const float4 dc = spline[((int64_t)s_pr * I + iv) * Q1 + (Q1 - 1)];
            er0 = dD[rcv] * eval_cubic_deriv(dc, t, sxi);
        }
        s_er = er0;
    }
    __syncthreads();

    int n_paths = 0;
    const PathMeta* paths = get_paths<LX>(n_paths);
    const CGEntry* cg = get_cg<LX>();
    const float* xs[3] = {x0, x1, x2};
    const float* dms[4] = {dm0, dm1, dm2, dm3};
    float* dxs[3] = {dx0, dx1, dx2};
    static constexpr int n_l3[4] = {
        LX == 0 ? 1 : (LX == 1 ? 2 : 3),
        LX == 0 ? 1 : (LX == 1 ? 3 : 5),
        LX == 0 ? 1 : (LX == 1 ? 3 : 5),
        LX == 0 ? 1 : (LX == 1 ? 2 : 4)
    };

    const float4* sp_row = spline + ((int64_t)s_pr * I + s_iv) * Q1;
    const int snd = s_snd, rcv = s_rcv;

    float er_loc = 0.0f;
    float dY_loc[16];
    #pragma unroll
    for (int m = 0; m < 16; ++m) dY_loc[m] = 0.0f;

    for (int c = tid; c < C; c += blockDim.x) {
        float xv[9], dxv[9];
        #pragma unroll
        for (int i = 0; i < 9; ++i) dxv[i] = 0.0f;
        xv[0] = xs[0][(int64_t)snd * C + c];
        if constexpr (LX >= 1) {
            #pragma unroll
            for (int m = 0; m < 3; ++m) xv[1 + m] = xs[1][((int64_t)snd * 3 + m) * C + c];
        }
        if constexpr (LX >= 2) {
            #pragma unroll
            for (int m = 0; m < 5; ++m) xv[4 + m] = xs[2][((int64_t)snd * 5 + m) * C + c];
        }

        for (int p = 0; p < n_paths; ++p) {
            const PathMeta pm = paths[p];
            const float4 cf = sp_row[p * C + c];
            const float R = eval_cubic(cf, s_t);
            const float dR = eval_cubic_deriv(cf, s_t, s_sxi);
            const int x_off = pm.l1 * pm.l1;
            const int y_off = pm.l2 * pm.l2;
            const int d3 = 2 * pm.l3 + 1;
            const int row_w = n_l3[pm.l3] * C;
            const float* dm_src = dms[pm.l3];
            float g3[7];
            for (int m3 = 0; m3 < d3; ++m3)
                g3[m3] = dm_src[((int64_t)rcv * d3 + m3) * row_w + pm.l3_slot * C + c];

            float er_p = 0.0f;
            for (int k = 0; k < pm.cg_len; ++k) {
                const CGEntry ce = cg[pm.cg_off + k];
                const float c_g = ce.val * g3[ce.m3];
                const float x_m1 = xv[x_off + ce.m1];
                const float y_m2 = s_Y[y_off + ce.m2];
                dxv[x_off + ce.m1] = fmaf(c_g * R, y_m2, dxv[x_off + ce.m1]);
                dY_loc[y_off + ce.m2] = fmaf(c_g * R, x_m1, dY_loc[y_off + ce.m2]);
                er_p = fmaf(c_g * x_m1, y_m2, er_p);
            }
            er_loc = fmaf(er_p, dR, er_loc);
        }

        if (dxs[0]) atomicAdd(dxs[0] + (int64_t)snd * C + c, dxv[0]);
        if constexpr (LX >= 1) {
            if (dxs[1])
                for (int m = 0; m < 3; ++m) atomicAdd(dxs[1] + ((int64_t)snd * 3 + m) * C + c, dxv[1 + m]);
        }
        if constexpr (LX >= 2) {
            if (dxs[2])
                for (int m = 0; m < 5; ++m) atomicAdd(dxs[2] + ((int64_t)snd * 5 + m) * C + c, dxv[4 + m]);
        }
    }

    atomicAdd(&s_er, er_loc);
    #pragma unroll
    for (int m = 0; m < 16; ++m) if (dY_loc[m] != 0.0f) atomicAdd(&s_dY[m], dY_loc[m]);
    __syncthreads();

    if (tid == 0) add_edge_dvec(vec_len[e], sh_perm, s_er, s_dY, edge_dvec + (int64_t)e * 3);
}

// 2nd-order backward (double-backprop) for general equivariant conv:
//   tan_mid = G(R*Y; tan_x) + G(v_r*dR*Y + R*dY_v; x)
//   bar_x  += G^T(R*Y; bar_mid) + G^T(v_r*dR*Y + R*dY_v; dmid)
//   tan_D  += v_r * rho'
template <int LX>
__global__ void ft_gconv_bwd2_kernel(
    int E, int C, int S, int I, int Q1, SplineGrid grid, int sh_perm, int pair_sym,
    const int* __restrict__ senders, const int* __restrict__ receivers, const float4* __restrict__ vec_len,
    const int* __restrict__ species, const float4* __restrict__ spline, int has_density,
    const float* __restrict__ x0, const float* __restrict__ x1, const float* __restrict__ x2,
    const float* __restrict__ tan_x0, const float* __restrict__ tan_x1, const float* __restrict__ tan_x2,
    const float* __restrict__ dm0, const float* __restrict__ dm1, const float* __restrict__ dm2, const float* __restrict__ dm3,
    const float* __restrict__ bar_mid0, const float* __restrict__ bar_mid1, const float* __restrict__ bar_mid2, const float* __restrict__ bar_mid3,
    const float* __restrict__ bar_edge_dvec,
    float* __restrict__ tan_mid0, float* __restrict__ tan_mid1, float* __restrict__ tan_mid2, float* __restrict__ tan_mid3,
    float* __restrict__ bar_x0, float* __restrict__ bar_x1, float* __restrict__ bar_x2,
    float* __restrict__ tan_D) {
    const int e = blockIdx.x;
    const int tid = threadIdx.x;
    if (e >= E) return;

    __shared__ float s_Y[16], s_dYv[16];
    __shared__ int s_iv, s_pr, s_snd, s_rcv;
    __shared__ float s_t, s_sxi, s_vr;

    if (tid == 0) {
        const int snd = senders[e], rcv = receivers[e];
        const float4 vl = vec_len[e];
        int iv; float t, sxi;
        locate_spline(grid, vl.w, iv, t, sxi);
        s_iv = iv; s_t = t; s_sxi = sxi; s_snd = snd; s_rcv = rcv;
        s_pr = loup::table_pair(pair_sym, species[snd], species[rcv], S);
        float vr = 0.0f;
        if (bar_edge_dvec) {
            edge_geom_and_jvp(vl, sh_perm,
                              bar_edge_dvec[3 * e + 0], bar_edge_dvec[3 * e + 1], bar_edge_dvec[3 * e + 2],
                              s_Y, s_dYv, vr);
        } else {
            if (sh_perm) loup::edge_sh16(vl.y, vl.z, vl.x, vl.w, reinterpret_cast<float4*>(s_Y));
            else         loup::edge_sh16(vl.x, vl.y, vl.z, vl.w, reinterpret_cast<float4*>(s_Y));
            for (int m = 0; m < 16; ++m) s_dYv[m] = 0.0f;
        }
        s_vr = vr;
        if (has_density && tan_D) {
            const float4 dc = spline[((int64_t)s_pr * I + iv) * Q1 + (Q1 - 1)];
            atomicAdd(tan_D + rcv, vr * eval_cubic_deriv(dc, t, sxi));
        }
    }
    __syncthreads();

    int n_paths = 0;
    const PathMeta* paths = get_paths<LX>(n_paths);
    const CGEntry* cg = get_cg<LX>();
    const float* xs[3] = {x0, x1, x2};
    const float* txs[3] = {tan_x0, tan_x1, tan_x2};
    const float* dms[4] = {dm0, dm1, dm2, dm3};
    const float* bms[4] = {bar_mid0, bar_mid1, bar_mid2, bar_mid3};
    float* tms[4] = {tan_mid0, tan_mid1, tan_mid2, tan_mid3};
    float* bxs[3] = {bar_x0, bar_x1, bar_x2};
    static constexpr int n_l3[4] = {
        LX == 0 ? 1 : (LX == 1 ? 2 : 3),
        LX == 0 ? 1 : (LX == 1 ? 3 : 5),
        LX == 0 ? 1 : (LX == 1 ? 3 : 5),
        LX == 0 ? 1 : (LX == 1 ? 2 : 4)
    };

    const float4* sp_row = spline + ((int64_t)s_pr * I + s_iv) * Q1;
    const int snd = s_snd, rcv = s_rcv;

    for (int c = tid; c < C; c += blockDim.x) {
        float xv[9], txv[9], bxv[9];
        #pragma unroll
        for (int i = 0; i < 9; ++i) { txv[i] = 0.0f; bxv[i] = 0.0f; }
        xv[0] = xs[0][(int64_t)snd * C + c];
        if (txs[0]) txv[0] = txs[0][(int64_t)snd * C + c];
        if constexpr (LX >= 1) {
            #pragma unroll
            for (int m = 0; m < 3; ++m) {
                xv[1 + m] = xs[1][((int64_t)snd * 3 + m) * C + c];
                if (txs[1]) txv[1 + m] = txs[1][((int64_t)snd * 3 + m) * C + c];
            }
        }
        if constexpr (LX >= 2) {
            #pragma unroll
            for (int m = 0; m < 5; ++m) {
                xv[4 + m] = xs[2][((int64_t)snd * 5 + m) * C + c];
                if (txs[2]) txv[4 + m] = txs[2][((int64_t)snd * 5 + m) * C + c];
            }
        }

        for (int p = 0; p < n_paths; ++p) {
            const PathMeta pm = paths[p];
            const float4 cf = sp_row[p * C + c];
            const float R = eval_cubic(cf, s_t);
            const float vdR = s_vr * eval_cubic_deriv(cf, s_t, s_sxi);
            const int x_off = pm.l1 * pm.l1;
            const int y_off = pm.l2 * pm.l2;
            const int d3 = 2 * pm.l3 + 1;
            const int row_w = n_l3[pm.l3] * C;

            // Load dm and optional bar_mid for this path
            float g_dm[7], g_bm[7];
            for (int m3 = 0; m3 < d3; ++m3) {
                const int64_t idx = ((int64_t)rcv * d3 + m3) * row_w + pm.l3_slot * C + c;
                g_dm[m3] = dms[pm.l3] ? dms[pm.l3][idx] : 0.0f;
                g_bm[m3] = bms[pm.l3] ? bms[pm.l3][idx] : 0.0f;
            }

            float tm_out[7] = {0, 0, 0, 0, 0, 0, 0};
            for (int k = 0; k < pm.cg_len; ++k) {
                const CGEntry ce = cg[pm.cg_off + k];
                const float ry = R * s_Y[y_off + ce.m2];
                const float dy_tot = fmaf(vdR, s_Y[y_off + ce.m2], R * s_dYv[y_off + ce.m2]);
                // tan_mid += CG * (ry * tan_x + dy_tot * x)
                tm_out[ce.m3] = fmaf(ce.val, fmaf(ry, txv[x_off + ce.m1], dy_tot * xv[x_off + ce.m1]), tm_out[ce.m3]);
                // bar_x += CG * (ry * bar_mid + dy_tot * dm)
                bxv[x_off + ce.m1] = fmaf(ce.val, fmaf(ry, g_bm[ce.m3], dy_tot * g_dm[ce.m3]), bxv[x_off + ce.m1]);
            }

            if (tms[pm.l3]) {
                for (int m3 = 0; m3 < d3; ++m3) {
                    const int64_t idx = ((int64_t)rcv * d3 + m3) * row_w + pm.l3_slot * C + c;
                    atomicAdd(tms[pm.l3] + idx, tm_out[m3]);
                }
            }
        }

        if (bxs[0]) atomicAdd(bxs[0] + (int64_t)snd * C + c, bxv[0]);
        if constexpr (LX >= 1) {
            if (bxs[1])
                for (int m = 0; m < 3; ++m) atomicAdd(bxs[1] + ((int64_t)snd * 3 + m) * C + c, bxv[1 + m]);
        }
        if constexpr (LX >= 2) {
            if (bxs[2])
                for (int m = 0; m < 5; ++m) atomicAdd(bxs[2] + ((int64_t)snd * 5 + m) * C + c, bxv[4 + m]);
        }
    }
}

// ===========================================================================
// 3. Frozen ZBL pair term (energy + edge_dvec)
// ===========================================================================
__global__ void ft_zbl_kernel(
    int E, int S, ZblFT z, const int* __restrict__ senders, const int* __restrict__ receivers,
    const float4* __restrict__ vec_len, const int* __restrict__ species, const int* __restrict__ atom_graph,
    double* __restrict__ graph_E, float* __restrict__ edge_dvec) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const float4 vl = vec_len[e];
    if (!(vl.w > 0.0f && vl.w < z.rmax)) return;
    const int snd = senders[e], rcv = receivers[e];
    const int p = species[snd] * S + species[rcv];
    const float rc = z.d_rc[p];
    if (!(vl.w < rc)) return;
    const float inv_a = z.d_inv_a[p], pref = z.d_pref[p], zp = z.p;
    const float u = vl.w * inv_a;
    const float e0 = expf(-3.2f * u), e1 = expf(-0.9423f * u), e2 = expf(-0.4028f * u), e3 = expf(-0.2016f * u);
    const float phi = z.c[0] * e0 + z.c[1] * e1 + z.c[2] * e2 + z.c[3] * e3;
    const float dphi = -inv_a * (3.2f * z.c[0] * e0 + 0.9423f * z.c[1] * e1 + 0.4028f * z.c[2] * e2 + 0.2016f * z.c[3] * e3);
    const float inv_r = 1.0f / vl.w;
    const float vz = pref * inv_r * phi;
    const float dv = pref * inv_r * (dphi - phi * inv_r);
    const float sr = vl.w / rc;
    const float s2 = sr * sr;
    const float sp = zp == 5.0f ? s2 * s2 * sr : powf(sr, zp);
    const float env = 1.0f - ((zp + 1.0f) * (zp + 2.0f) * 0.5f) * sp + (zp * (zp + 2.0f)) * sp * sr -
                      (zp * (zp + 1.0f) * 0.5f) * sp * sr * sr;
    const float denv = -(zp * (zp + 1.0f) * (zp + 2.0f) * 0.5f) * (sp / sr) * (1.0f - sr) * (1.0f - sr) / rc;
    const float en = vz * env;
    const float g = (dv * env + vz * denv) * inv_r;
    atomicAdd(graph_E + atom_graph[rcv], (double)en);
    edge_dvec[3 * e + 0] += g * vl.x;
    edge_dvec[3 * e + 1] += g * vl.y;
    edge_dvec[3 * e + 2] += g * vl.z;
}

}  // namespace loup_ft
