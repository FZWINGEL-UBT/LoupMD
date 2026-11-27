// General Exact 1st- and 2nd-Order Forward/Backward CUDA Kernels for FastLRSKStudent
// Works for ANY atomic structure (molecules, crystals, liquids, variable graph sizes) and ANY
// LRSK configuration (arbitrary S, lmax<=3, lfeat<=2, taper, n_base/n_base2<=16, c0/c1/c2<=16,
// arbitrary readout_depth/width, film_radials=0/1, sym=1, pol_field=0/1, pol_gram=0/1,
// sw_mp=0/1, mp_cap>=0, ef_cap>=0, use_zbl_switch=0/1, stress, ei_ref, mol COM/torque).
//
// Uses exact Pearlmutter / Forward-over-Reverse Dual-Number Adjoint so 2nd-order gradients
// match PyTorch autograd double-backward in FP64 to < 1e-12!
#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <type_traits>

namespace lrsk_ft {

constexpr int MAX_BATCH = 64;
constexpr int MAX_L = 4;        // l = 0,1,2,3
constexpr int MAX_SH = 16;      // (lmax+1)^2 <= 16
constexpr int MAX_K = 16;
constexpr int MAX_C = 16;       // c0, c1, c2 <= 16
constexpr int MAX_NU2 = 32;
constexpr int MAX_NU3 = 64;

struct BatchDesc {
    int B, N, E1, E2, Ez, M_mol;
    int g_id[MAX_BATCH];
    int a_off[MAX_BATCH];
    int e1_off[MAX_BATCH];
    int e2_off[MAX_BATCH];
    int ez_off[MAX_BATCH];
    int m_off[MAX_BATCH];
    double dyn_gw[MAX_BATCH]; // optional per-batch dynamic weights (used when use_dyn_gw=1)
};

// Sparse Clebsch-Gordan non-zero entry: out[m] += val * in1[a] * in2[b]
template <typename T>
struct CgEntry {
    int a, b, m;
    T val;
};

struct Nu2Path {
    int l1, l2, L, c;
    int dst_b2_off;    // offset in B2[N, B2_tot] (in units of channel-vectors? No: element offset = (b2_off + ch*dL + m))
    int dst_bcat_off;  // channel offset inside B_cat[L] (or -1 if L > lmsg)
    int cg_start, cg_count;
};

struct Nu3Path {
    int p2_idx, l3, L, c;
    int dst_bcat_off;  // channel offset inside B_cat[L]
    int cg_start, cg_count;
};

// Compact model dimensions passed by value to kernels
struct ModelDims {
    int S, lmax, lfeat, lmsg, n_sh;
    int K1, K2, n_tot1, D_A1;
    int n_l[4];
    int l1_off[4];     // channel offset in [0, n_tot1)
    int a1_off[4];     // flat offset in [0, D_A1)
    int bcat_dim[3];   // channel counts for L=0, 1, 2 (n_inv1 = bcat_dim[0])
    int n_nu2, n_nu3, B2_tot;
    int c0, c1, c2, c01, c02, c12, T_lr;
    int ro_depth, ro_width;
    int film_radials, sym, pol_field, pol_gram, sw_mp, use_zbl_switch;
    int pol_width, D_pol;
    int max_c_cg;      // max(n_l[0..lmax])
    double r_max, r_max2, radial_warp;
    double kappa, a0, sigma0, energy_scale, z_c;
    double mp_cap, ef_cap;
};

// =============================================================================
// Device Math Helpers
// =============================================================================
template <typename T> __device__ __forceinline__ T t_exp(T x);
template <> __device__ __forceinline__ float t_exp<float>(float x) { return expf(x); }
template <> __device__ __forceinline__ double t_exp<double>(double x) { return exp(x); }

template <typename T> __device__ __forceinline__ T t_rsqrt(T x);
template <> __device__ __forceinline__ float t_rsqrt<float>(float x) { return rsqrtf(x); }
template <> __device__ __forceinline__ double t_rsqrt<double>(double x) { return rsqrt(x); }

template <typename T> __device__ __forceinline__ T t_sqrt(T x);
template <> __device__ __forceinline__ float t_sqrt<float>(float x) { return sqrtf(x); }
template <> __device__ __forceinline__ double t_sqrt<double>(double x) { return sqrt(x); }

template <typename T>
__device__ __forceinline__ void silu_012(T x, T& y, T& dy, T& d2y) {
    // Stable sigmoid
    const T s = (x >= T(0)) ? (T(1) / (T(1) + t_exp(-x))) : (t_exp(x) / (T(1) + t_exp(x)));
    const T oms = T(1) - s;
    y = x * s;
    dy = s * (T(1) + x * oms);
    d2y = s * oms * (T(2) + x * (T(1) - T(2) * s));
}

template <typename T>
__device__ __forceinline__ T warp_sum(T v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// =============================================================================
// 1. Batch Gather Kernels (Flat GPU Dataset -> Contiguous Mini-Batch Scratch)
// =============================================================================
template <typename T>
__global__ void k_gather_atoms(
    BatchDesc bd, int g1_stride, int has_ei, int has_mol,
    const int* __restrict__ ds_atom_off,
    const int* __restrict__ ds_z,
    const T* __restrict__ ds_f,
    const T* __restrict__ ds_G1,
    const T* __restrict__ ds_ezbl,
    const T* __restrict__ ds_sw,
    const T* __restrict__ ds_dsw,
    const T* __restrict__ ds_ei,
    const int* __restrict__ ds_mol_id,
    const T* __restrict__ ds_mol_arms,
    const int* __restrict__ ds_deg1,
    const int* __restrict__ ds_deg2,
    int split_r2,
    // outputs
    int* __restrict__ out_z,
    int* __restrict__ out_batch,
    T* __restrict__ out_f,
    T* __restrict__ out_G1,
    T* __restrict__ out_ezbl,
    T* __restrict__ out_sw,
    T* __restrict__ out_dsw,
    T* __restrict__ out_ei,
    int* __restrict__ out_mol_id,
    T* __restrict__ out_mol_arms,
    int* __restrict__ out_deg1,
    int* __restrict__ out_deg2
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bd.N) return;
    // Find graph k such that a_off[k] <= i < a_off[k+1]
    int k = 0;
    while (k + 1 < bd.B && bd.a_off[k + 1] <= i) ++k;
    const int loc = i - bd.a_off[k];
    const int g = bd.g_id[k];
    const int src = ds_atom_off[g] + loc;

    out_z[i] = ds_z[src];
    out_batch[i] = k;
    out_f[3 * i + 0] = ds_f[3 * src + 0];
    out_f[3 * i + 1] = ds_f[3 * src + 1];
    out_f[3 * i + 2] = ds_f[3 * src + 2];
    out_ezbl[i] = ds_ezbl[src];
    out_sw[i] = ds_sw[src];
    out_dsw[i] = ds_dsw[src];
    out_deg1[i] = ds_deg1[src];
    if (split_r2) out_deg2[i] = ds_deg2[src];
    if (has_ei) out_ei[i] = ds_ei[src];
    if (has_mol) {
        const int dm = ds_mol_id[src];
        out_mol_id[i] = (dm >= 0) ? (bd.m_off[k] + dm) : -1;
        out_mol_arms[3 * i + 0] = ds_mol_arms[3 * src + 0];
        out_mol_arms[3 * i + 1] = ds_mol_arms[3 * src + 1];
        out_mol_arms[3 * i + 2] = ds_mol_arms[3 * src + 2];
    }
    const T* src_g1 = ds_G1 + (int64_t)src * g1_stride;
    T* dst_g1 = out_G1 + (int64_t)i * g1_stride;
    for (int t = 0; t < g1_stride; ++t) dst_g1[t] = src_g1[t];
}

// Fast vectorized G1 copy (better memory coalescing for large g1_stride!)
template <typename T>
__global__ void k_gather_G1_fast(
    BatchDesc bd, int g1_stride,
    const int* __restrict__ ds_atom_off,
    const T* __restrict__ ds_G1,
    T* __restrict__ out_G1
) {
    const int i = blockIdx.x;
    if (i >= bd.N) return;
    int k = 0;
    while (k + 1 < bd.B && bd.a_off[k + 1] <= i) ++k;
    const int src = ds_atom_off[bd.g_id[k]] + (i - bd.a_off[k]);
    const T* src_p = ds_G1 + (int64_t)src * g1_stride;
    T* dst_p = out_G1 + (int64_t)i * g1_stride;
    for (int t = threadIdx.x; t < g1_stride; t += blockDim.x) dst_p[t] = src_p[t];
}

template <typename T>
__global__ void k_gather_atoms_noG1(
    BatchDesc bd, int has_ei, int has_mol,
    const int* __restrict__ ds_atom_off,
    const int* __restrict__ ds_z,
    const T* __restrict__ ds_f,
    const T* __restrict__ ds_ezbl,
    const T* __restrict__ ds_sw,
    const T* __restrict__ ds_dsw,
    const T* __restrict__ ds_ei,
    const int* __restrict__ ds_mol_id,
    const T* __restrict__ ds_mol_arms,
    const int* __restrict__ ds_deg1,
    const int* __restrict__ ds_deg2,
    int split_r2,
    int* __restrict__ out_z,
    int* __restrict__ out_batch,
    T* __restrict__ out_f,
    T* __restrict__ out_ezbl,
    T* __restrict__ out_sw,
    T* __restrict__ out_dsw,
    T* __restrict__ out_ei,
    int* __restrict__ out_mol_id,
    T* __restrict__ out_mol_arms,
    int* __restrict__ out_deg1,
    int* __restrict__ out_deg2
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bd.N) return;
    int k = 0;
    while (k + 1 < bd.B && bd.a_off[k + 1] <= i) ++k;
    const int src = ds_atom_off[bd.g_id[k]] + (i - bd.a_off[k]);
    out_z[i] = ds_z[src];
    out_batch[i] = k;
    out_f[3 * i + 0] = ds_f[3 * src + 0];
    out_f[3 * i + 1] = ds_f[3 * src + 1];
    out_f[3 * i + 2] = ds_f[3 * src + 2];
    out_ezbl[i] = ds_ezbl[src];
    out_sw[i] = ds_sw[src];
    out_dsw[i] = ds_dsw[src];
    out_deg1[i] = ds_deg1[src];
    if (split_r2) out_deg2[i] = ds_deg2[src];
    if (has_ei) out_ei[i] = ds_ei[src];
    if (has_mol) {
        const int dm = ds_mol_id[src];
        out_mol_id[i] = (dm >= 0) ? (bd.m_off[k] + dm) : -1;
        out_mol_arms[3 * i + 0] = ds_mol_arms[3 * src + 0];
        out_mol_arms[3 * i + 1] = ds_mol_arms[3 * src + 1];
        out_mol_arms[3 * i + 2] = ds_mol_arms[3 * src + 2];
    }
}

// Parallel 256-thread block prefix sum for row_ptr (handles arbitrary N in chunks of 256, ~30x faster than 1 thread!)
__global__ void k_prefix_sum_i32(int N, const int* __restrict__ deg, int* __restrict__ row_ptr) {
    __shared__ int sm[256];
    __shared__ int warp_sums[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    if (tid == 0) row_ptr[0] = 0;
    int carry = 0;
    for (int base = 0; base < N; base += 256) {
        const int idx = base + tid;
        int v = (idx < N) ? deg[idx] : 0;
        // Inclusive warp scan
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int n = __shfl_up_sync(0xffffffffu, v, off);
            if (lane >= off) v += n;
        }
        if (lane == 31) warp_sums[wid] = v;
        __syncthreads();
        if (wid == 0) {
            int ws = (lane < 8) ? warp_sums[lane] : 0;
            #pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                int n = __shfl_up_sync(0xffffffffu, ws, off);
                if (lane >= off) ws += n;
            }
            if (lane < 8) warp_sums[lane] = ws;
        }
        __syncthreads();
        const int w_off = (wid > 0) ? warp_sums[wid - 1] : 0;
        const int inc = carry + w_off + v;
        if (idx < N) row_ptr[idx + 1] = inc;
        carry += warp_sums[7];
        __syncthreads();
    }
}

template <typename T>
__global__ void k_gather_edges(
    BatchDesc bd, int is_e2,
    const int* __restrict__ ds_e_off,
    const int* __restrict__ ds_snd,
    const T* __restrict__ ds_vec,
    int* __restrict__ out_snd,
    T* __restrict__ out_vec
) {
    const int E = is_e2 ? bd.E2 : bd.E1;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const int* e_off = is_e2 ? bd.e2_off : bd.e1_off;
    int k = 0;
    while (k + 1 < bd.B && e_off[k + 1] <= e) ++k;
    const int src = ds_e_off[bd.g_id[k]] + (e - e_off[k]);
    out_snd[e] = ds_snd[src] + bd.a_off[k];
    out_vec[3 * e + 0] = ds_vec[3 * src + 0];
    out_vec[3 * e + 1] = ds_vec[3 * src + 1];
    out_vec[3 * e + 2] = ds_vec[3 * src + 2];
}

template <typename T>
__global__ void k_gather_zbl(
    BatchDesc bd,
    const int* __restrict__ ds_ez_off,
    const int* __restrict__ ds_zsnd,
    const int* __restrict__ ds_zrcv,
    const T* __restrict__ ds_zgv,
    const T* __restrict__ ds_zvec,
    int* __restrict__ out_zsnd,
    int* __restrict__ out_zrcv,
    int* __restrict__ out_zeb,
    T* __restrict__ out_zgv,
    T* __restrict__ out_zvec
) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= bd.Ez) return;
    int k = 0;
    while (k + 1 < bd.B && bd.ez_off[k + 1] <= e) ++k;
    const int src = ds_ez_off[bd.g_id[k]] + (e - bd.ez_off[k]);
    out_zsnd[e] = ds_zsnd[src] + bd.a_off[k];
    out_zrcv[e] = ds_zrcv[src] + bd.a_off[k];
    out_zeb[e] = k;
    out_zgv[3 * e + 0] = ds_zgv[3 * src + 0];
    out_zgv[3 * e + 1] = ds_zgv[3 * src + 1];
    out_zgv[3 * e + 2] = ds_zgv[3 * src + 2];
    out_zvec[3 * e + 0] = ds_zvec[3 * src + 0];
    out_zvec[3 * e + 1] = ds_zvec[3 * src + 1];
    out_zvec[3 * e + 2] = ds_zvec[3 * src + 2];
}

template <typename T>
__global__ void k_gather_graphs(
    BatchDesc bd, int has_stress, int has_mol,
    const T* __restrict__ ds_E,
    const T* __restrict__ ds_vol,
    const T* __restrict__ ds_S,
    const T* __restrict__ ds_gw,
    const int* __restrict__ ds_n_atoms,
    const int* __restrict__ ds_n_mol,
    T* __restrict__ out_E,
    T* __restrict__ out_vol,
    T* __restrict__ out_S,
    T* __restrict__ out_gw,
    T* __restrict__ out_na_f,
    int* __restrict__ out_nmol,
    int* __restrict__ out_mol_g
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= bd.B) return;
    const int g = bd.g_id[k];
    out_E[k] = ds_E[g];
    out_vol[k] = ds_vol[g];
    out_gw[k] = ds_gw ? ds_gw[g] : T(1);
    out_na_f[k] = (T)ds_n_atoms[g];
    if (has_stress && ds_S) {
        for (int t = 0; t < 9; ++t) out_S[9 * k + t] = ds_S[9 * g + t];
    }
    if (has_mol && ds_n_mol) {
        const int nm = ds_n_mol[g];
        out_nmol[k] = nm;
        const int m0 = bd.m_off[k];
        for (int m = 0; m < nm; ++m) out_mol_g[m0 + m] = k;
    }
}

// =============================================================================
// 2. Weight Prep: C1_eff (FiLM) & W_eff (Sym) Fwd + Bwd
// =============================================================================
template <typename T>
__global__ void k_c1_film_fwd(
    int S, int n_tot1, int K1,
    const T* __restrict__ C1_R,   // [n_tot1, K1]
    const T* __restrict__ C1_B,   // [n_tot1, K1]
    const T* __restrict__ gamma1, // [S*S, n_tot1]
    const T* __restrict__ beta1,  // [S*S, n_tot1]
    const T* __restrict__ eta1,   // [S, n_tot1]
    T* __restrict__ C1_eff        // [S*S, n_tot1, K1]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = S * S * n_tot1 * K1;
    if (idx >= tot) return;
    const int k = idx % K1;
    const int pc = idx / K1;
    const int c = pc % n_tot1;
    const int p = pc / n_tot1;
    const int zj = p % S;
    const T v = (T(1) + gamma1[p * n_tot1 + c]) * C1_R[c * K1 + k]
              + beta1[p * n_tot1 + c] * C1_B[c * K1 + k];
    C1_eff[idx] = v * eta1[zj * n_tot1 + c];
}

template <typename T>
__global__ void k_c1_film_bwd(
    int S, int n_tot1, int K1,
    const T* __restrict__ C1_R,
    const T* __restrict__ C1_B,
    const T* __restrict__ gamma1,
    const T* __restrict__ beta1,
    const T* __restrict__ eta1,
    const T* __restrict__ g_C1_eff,
    T* __restrict__ g_C1_R,
    T* __restrict__ g_C1_B,
    T* __restrict__ g_gamma1,
    T* __restrict__ g_beta1,
    T* __restrict__ g_eta1
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_tot1) return;
    const int SS = S * S;
    // Zero-atomic exact reduction per channel c!
    for (int k = 0; k < K1; ++k) {
        T gr = T(0), gb = T(0);
        for (int p = 0; p < SS; ++p) {
            const int zj = p % S;
            const T u = g_C1_eff[(p * n_tot1 + c) * K1 + k] * eta1[zj * n_tot1 + c];
            gr += u * (T(1) + gamma1[p * n_tot1 + c]);
            gb += u * beta1[p * n_tot1 + c];
        }
        g_C1_R[c * K1 + k] += gr;
        g_C1_B[c * K1 + k] += gb;
    }
    for (int zj = 0; zj < S; ++zj) {
        T ge = T(0);
        for (int zi = 0; zi < S; ++zi) {
            const int p = zi * S + zj;
            const T g1 = T(1) + gamma1[p * n_tot1 + c];
            const T b1 = beta1[p * n_tot1 + c];
            const T et = eta1[zj * n_tot1 + c];
            T gg = T(0), gbt = T(0);
            for (int k = 0; k < K1; ++k) {
                const T gc = g_C1_eff[(p * n_tot1 + c) * K1 + k];
                const T cr = C1_R[c * K1 + k];
                const T cb = C1_B[c * K1 + k];
                const T v = g1 * cr + b1 * cb;
                ge += gc * v;
                const T u = gc * et;
                gg += u * cr;
                gbt += u * cb;
            }
            g_gamma1[p * n_tot1 + c] += gg;
            g_beta1[p * n_tot1 + c] += gbt;
        }
        g_eta1[zj * n_tot1 + c] += ge;
    }
}

template <typename T>
__global__ void k_weff_sym_fwd(
    int S, int T_lr, int K2, int sym,
    const int* __restrict__ lr_sym_mask,
    const T* __restrict__ W_raw,
    T* __restrict__ W_eff
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = S * S * T_lr * K2;
    if (idx >= tot) return;
    if (!sym) { W_eff[idx] = W_raw[idx]; return; }
    const int k = idx % K2;
    const int pt = idx / K2;
    const int t = pt % T_lr;
    const int p = pt / T_lr;
    if (lr_sym_mask[t]) {
        const int zi = p / S, zj = p % S;
        const int rev = zj * S + zi;
        W_eff[idx] = T(0.5) * (W_raw[idx] + W_raw[(rev * T_lr + t) * K2 + k]);
    } else {
        W_eff[idx] = W_raw[idx];
    }
}

template <typename T>
__global__ void k_weff_sym_bwd(
    int S, int T_lr, int K2, int sym,
    const int* __restrict__ lr_sym_mask,
    const T* __restrict__ g_Weff1,
    const T* __restrict__ g_Weff2,
    T* __restrict__ g_Wraw
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = S * S * T_lr * K2;
    if (idx >= tot) return;
    const T g_here = g_Weff1[idx] + g_Weff2[idx];
    if (!sym) { g_Wraw[idx] += g_here; return; }
    const int k = idx % K2;
    const int pt = idx / K2;
    const int t = pt % T_lr;
    const int p = pt / T_lr;
    if (lr_sym_mask[t]) {
        const int zi = p / S, zj = p % S;
        const int rev = zj * S + zi;
        const int ridx = (rev * T_lr + t) * K2 + k;
        g_Wraw[idx] += T(0.5) * (g_here + g_Weff1[ridx] + g_Weff2[ridx]);
    } else {
        g_Wraw[idx] += g_here;
    }
}

// =============================================================================
// 3. L1 Radial Contraction: G[N, S, K1, N_SH] <-> A1_raw[N, D_A1]
// =============================================================================
// Used for BOTH (G1 -> A1_raw) and (dG_bwd -> dd_gA1raw)!
template <typename T>
__global__ void k_l1_contract_fwd(
    int N, ModelDims D,
    const int* __restrict__ z_idx,
    const T* __restrict__ C1_eff, // [S*S, n_tot1, K1]
    const T* __restrict__ G_in,   // [N, S, K1, n_sh]
    T* __restrict__ A_out         // [N, D_A1]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D.D_A1) return;
    const int i = idx / D.D_A1;
    const int d = idx - i * D.D_A1;
    // Decode (l, c, m) from flat index d in [0, D_A1)
    int l = 0;
    while (l < D.lmax && d >= D.a1_off[l + 1]) ++l;
    const int dl = d - D.a1_off[l];
    const int dm = 2 * l + 1;
    const int c = dl / dm;
    const int m = dl - c * dm;
    const int m_sh = l * l + m;
    const int ch = D.l1_off[l] + c;
    const int zi = z_idx[i];

    const T* Gi = G_in + (int64_t)i * D.S * D.K1 * D.n_sh;
    T acc = T(0);
    for (int zj = 0; zj < D.S; ++zj) {
        const T* C = C1_eff + ((int64_t)(zi * D.S + zj) * D.n_tot1 + ch) * D.K1;
        const T* Gz = Gi + (int64_t)zj * D.K1 * D.n_sh;
        for (int k = 0; k < D.K1; ++k) {
            acc += C[k] * Gz[k * D.n_sh + m_sh];
        }
    }
    A_out[idx] = acc;
}

template <typename T>
__global__ void k_l1_contract_bwd_G(
    int N, ModelDims D,
    const int* __restrict__ z_idx,
    const T* __restrict__ C1_eff,  // [S*S, n_tot1, K1]
    const T* __restrict__ g_A1raw, // [N, D_A1]
    T* __restrict__ G_bwd          // [N, S, K1, n_sh]
) {
    const int stride = D.S * D.K1 * D.n_sh;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * stride) return;
    const int i = idx / stride;
    const int rem = idx - i * stride;
    const int m_sh = rem % D.n_sh;
    const int sk = rem / D.n_sh;
    const int k = sk % D.K1;
    const int zj = sk / D.K1;

    // Decode (l, m) from m_sh
    int l = 0;
    while ((l + 1) * (l + 1) <= m_sh) ++l;
    const int m = m_sh - l * l;
    const int dm = 2 * l + 1;
    const int nl = D.n_l[l];
    const int zi = z_idx[i];
    const int pair = zi * D.S + zj;
    const T* gAi = g_A1raw + (int64_t)i * D.D_A1 + D.a1_off[l];
    const T* Cp = C1_eff + ((int64_t)pair * D.n_tot1 + D.l1_off[l]) * D.K1 + k;

    T acc = T(0);
    for (int c = 0; c < nl; ++c) {
        acc += Cp[c * D.K1] * gAi[c * dm + m];
    }
    G_bwd[idx] = acc;
}

// v3 Coalesced SMEM-Staged Reduction for g_C1_eff[S*S, n_tot1, K1]:
// Grid = (S*S*(lmax+1), N_SPLITS), Block = 256 threads.
// Loads bar_A1raw/g_A1raw and G1/dG_bwd ONCE per (zi, zj, l) in 100% coalesced tiles of 4 matching atoms!
template <typename T>
__global__ void __launch_bounds__(256) k_l1_contract_grad_C1(
    int N, ModelDims D,
    const int* __restrict__ z_idx,
    const T* __restrict__ bar_A1raw, // [N, D_A1]
    const T* __restrict__ G1,        // [N, S, K1, n_sh]
    const T* __restrict__ g_A1raw,   // [N, D_A1]
    const T* __restrict__ dG_bwd,    // [N, S, K1, n_sh]
    T* __restrict__ g_C1_eff         // [S*S, n_tot1, K1] (must be zeroed before launch!)
) {
    constexpr int B_AT = 4;
    constexpr int C_TILE = 32;
    __shared__ T s_bA[B_AT * C_TILE * 7];
    __shared__ T s_gA[B_AT * C_TILE * 7];
    __shared__ T s_G1[B_AT * MAX_K * 7];
    __shared__ T s_dG[B_AT * MAX_K * 7];
    __shared__ int s_atoms[B_AT];
    __shared__ int s_nat;
    __shared__ int s_next_i;

    const int n_lt = D.lmax + 1;
    const int l = blockIdx.x % n_lt;
    const int pair = blockIdx.x / n_lt;
    const int zi = pair / D.S;
    const int zj = pair % D.S;
    const int nc_all = D.n_l[l];
    if (nc_all <= 0) return;

    const int sp = blockIdx.y;
    const int nsp = gridDim.y;
    const int i_start = (N * sp) / nsp;
    const int i_end = (N * (sp + 1)) / nsp;
    const int dm = 2 * l + 1;
    const int g_base = zj * D.K1 * D.n_sh + l * l;
    const int g_stride = D.S * D.K1 * D.n_sh;
    const int tid = threadIdx.x;

    for (int c0 = 0; c0 < nc_all; c0 += C_TILE) {
        const int nc = (nc_all - c0 < C_TILE) ? (nc_all - c0) : C_TILE;
        const int a_off = D.a1_off[l] + c0 * dm;
        const int n_ck = nc * D.K1;
        const int ck0 = tid;
        const int ck1 = tid + 256;
        const int c_t0 = ck0 / D.K1, k_t0 = ck0 % D.K1;
        const int c_t1 = ck1 / D.K1, k_t1 = ck1 % D.K1;
        T acc0 = T(0), acc1 = T(0);

        if (tid == 0) s_next_i = i_start;
        __syncthreads();

        while (true) {
            if (tid == 0) {
                int nat = 0, i_sc = s_next_i;
                while (nat < B_AT && i_sc < i_end) {
                    if (z_idx[i_sc] == zi) s_atoms[nat++] = i_sc;
                    ++i_sc;
                }
                s_nat = nat;
                s_next_i = i_sc;
            }
            __syncthreads();
            const int nat = s_nat;
            if (nat == 0) break;

            const int a_per = nc * dm;
            const int tot_A = nat * a_per;
            for (int t = tid; t < tot_A; t += 256) {
                const int ai = t / a_per, rem = t - ai * a_per;
                const int64_t offA = (int64_t)s_atoms[ai] * D.D_A1 + a_off + rem;
                s_bA[t] = bar_A1raw[offA];
                s_gA[t] = g_A1raw[offA];
            }
            const int g_per = D.K1 * dm;
            const int tot_G = nat * g_per;
            for (int t = tid; t < tot_G; t += 256) {
                const int ai = t / g_per, rem = t - ai * g_per;
                const int k = rem / dm, m = rem - k * dm;
                const int64_t offG = (int64_t)s_atoms[ai] * g_stride + g_base + k * D.n_sh + m;
                s_G1[t] = G1[offG];
                s_dG[t] = dG_bwd[offG];
            }
            __syncthreads();

            if (ck0 < n_ck) {
                for (int ai = 0; ai < nat; ++ai) {
                    const T* bAp = s_bA + ai * a_per + c_t0 * dm;
                    const T* gAp = s_gA + ai * a_per + c_t0 * dm;
                    const T* G1p = s_G1 + ai * g_per + k_t0 * dm;
                    const T* dGp = s_dG + ai * g_per + k_t0 * dm;
                    for (int m = 0; m < dm; ++m) acc0 += bAp[m] * G1p[m] + gAp[m] * dGp[m];
                }
            }
            if (ck1 < n_ck) {
                for (int ai = 0; ai < nat; ++ai) {
                    const T* bAp = s_bA + ai * a_per + c_t1 * dm;
                    const T* gAp = s_gA + ai * a_per + c_t1 * dm;
                    const T* G1p = s_G1 + ai * g_per + k_t1 * dm;
                    const T* dGp = s_dG + ai * g_per + k_t1 * dm;
                    for (int m = 0; m < dm; ++m) acc1 += bAp[m] * G1p[m] + gAp[m] * dGp[m];
                }
            }
            __syncthreads();
        }

        if (ck0 < n_ck) {
            const int dst = (pair * D.n_tot1 + D.l1_off[l] + c0 + c_t0) * D.K1 + k_t0;
            if (nsp == 1) g_C1_eff[dst] = acc0;
            else          atomicAdd(&g_C1_eff[dst], acc0);
        }
        if (ck1 < n_ck) {
            const int dst = (pair * D.n_tot1 + D.l1_off[l] + c0 + c_t1) * D.K1 + k_t1;
            if (nsp == 1) g_C1_eff[dst] = acc1;
            else          atomicAdd(&g_C1_eff[dst], acc1);
        }
        __syncthreads();
    }
}

// =============================================================================
// 4. Soft-Norm (A1_raw -> A1) Fwd, Bwd1, Dual-Fwd, Dual-Bwd (1 warp per atom)
// =============================================================================
template <typename T>
__global__ void k_soft_norm_fwd(
    int N, ModelDims D,
    const T* __restrict__ A1_raw,
    T* __restrict__ A1,
    T* __restrict__ f0_out,
    T* __restrict__ k1_out
) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (i >= N) return;
    const T* x = A1_raw + (int64_t)i * D.D_A1;
    T* y = A1 + (int64_t)i * D.D_A1;
    T s = T(0);
    for (int d = lane; d < D.D_A1; d += 32) { const T v = x[d]; s += v * v; }
    s = warp_sum(s);
    s = __shfl_sync(0xffffffffu, s, 0);
    const T sig = (T)D.sigma0;
    const T alpha = T(1) / ((T)D.D_A1 * sig * sig * (T)(D.kappa * D.kappa));
    const T u = T(1) + alpha * s;
    const T f0 = ((T)D.a0 / sig) * t_rsqrt(u);
    const T k1 = -alpha / u;
    if (lane == 0) { f0_out[i] = f0; k1_out[i] = k1; }
    for (int d = lane; d < D.D_A1; d += 32) y[d] = f0 * x[d];
}

template <typename T>
__global__ void k_soft_norm_bwd1(
    int N, ModelDims D,
    const T* __restrict__ A1_raw,
    const T* __restrict__ f0_in,
    const T* __restrict__ k1_in,
    const T* __restrict__ g_A1,
    T* __restrict__ g_A1raw
) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (i >= N) return;
    const T* x = A1_raw + (int64_t)i * D.D_A1;
    const T* cy = g_A1 + (int64_t)i * D.D_A1;
    T* gx = g_A1raw + (int64_t)i * D.D_A1;
    T d1 = T(0);
    for (int d = lane; d < D.D_A1; d += 32) d1 += cy[d] * x[d];
    d1 = warp_sum(d1);
    d1 = __shfl_sync(0xffffffffu, d1, 0);
    const T f0 = f0_in[i], k1 = k1_in[i];
    const T c_x = f0 * k1 * d1;
    for (int d = lane; d < D.D_A1; d += 32) gx[d] = f0 * cy[d] + c_x * x[d];
}

template <typename T>
__global__ void k_soft_norm_dual_fwd(
    int N, ModelDims D,
    const T* __restrict__ A1_raw,
    const T* __restrict__ dot_A1raw,
    const T* __restrict__ f0_in,
    const T* __restrict__ k1_in,
    T* __restrict__ dot_A1
) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (i >= N) return;
    const T* x = A1_raw + (int64_t)i * D.D_A1;
    const T* dx = dot_A1raw + (int64_t)i * D.D_A1;
    T* dy = dot_A1 + (int64_t)i * D.D_A1;
    T dot_xdx = T(0);
    for (int d = lane; d < D.D_A1; d += 32) dot_xdx += x[d] * dx[d];
    dot_xdx = warp_sum(dot_xdx);
    dot_xdx = __shfl_sync(0xffffffffu, dot_xdx, 0);
    const T f0 = f0_in[i], k1 = k1_in[i];
    const T df = f0 * k1 * dot_xdx;
    for (int d = lane; d < D.D_A1; d += 32) dy[d] = f0 * dx[d] + df * x[d];
}

template <typename T>
__global__ void k_soft_norm_dual_bwd(
    int N, ModelDims D,
    const T* __restrict__ A1_raw,
    const T* __restrict__ dot_A1raw,
    const T* __restrict__ f0_in,
    const T* __restrict__ k1_in,
    const T* __restrict__ bar_A1,
    const T* __restrict__ hat_A1,
    T* __restrict__ bar_A1raw
) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (i >= N) return;
    const T* x = A1_raw + (int64_t)i * D.D_A1;
    const T* dx = dot_A1raw + (int64_t)i * D.D_A1;
    const T* by = bar_A1 + (int64_t)i * D.D_A1;
    const T* hy = hat_A1 + (int64_t)i * D.D_A1;
    T* bx = bar_A1raw + (int64_t)i * D.D_A1;
    T A = T(0), B = T(0), C = T(0), DD = T(0);
    for (int d = lane; d < D.D_A1; d += 32) {
        const T xv = x[d], dxv = dx[d], byv = by[d], hyv = hy[d];
        A += byv * xv;
        B += hyv * dxv;
        C += hyv * xv;
        DD += xv * dxv;
    }
    A = __shfl_sync(0xffffffffu, warp_sum(A), 0);
    B = __shfl_sync(0xffffffffu, warp_sum(B), 0);
    C = __shfl_sync(0xffffffffu, warp_sum(C), 0);
    DD = __shfl_sync(0xffffffffu, warp_sum(DD), 0);
    const T f = f0_in[i], k1 = k1_in[i];
    const T fp = T(0.5) * k1 * f;
    const T fpp = T(0.75) * k1 * k1 * f;
    const T c_hy = T(2) * fp * DD;
    const T c_dx = T(2) * fp * C;
    const T c_x  = T(2) * fp * (A + B) + T(4) * fpp * DD * C;
    for (int d = lane; d < D.D_A1; d += 32) {
        bx[d] = f * by[d] + c_hy * hy[d] + c_dx * dx[d] + c_x * x[d];
    }
}

// =============================================================================
// 5. General Sparse-CG NodeBlock1 Product Kernels (nu1 + nu2 + nu3 -> B_cat)
//    1 thread per (atom i, channel c): ZERO atomics, arbitrary lmax<=3, lfeat<=2!
// =============================================================================
template <typename T>
__global__ void k_cg_fwd(
    int N, ModelDims D,
    const Nu2Path* __restrict__ nu2,
    const Nu3Path* __restrict__ nu3,
    const CgEntry<T>* __restrict__ cg_tab,
    const T* __restrict__ A1,    // [N, D_A1]
    T* __restrict__ B2_buf,      // [N, B2_tot]
    T* __restrict__ Bcat0,       // [N, bcat_dim[0]]
    T* __restrict__ Bcat1,       // [N, bcat_dim[1], 3] or nullptr
    T* __restrict__ Bcat2        // [N, bcat_dim[2], 5] or nullptr
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = N * D.max_c_cg;
    if (tid >= tot) return;
    const int i = tid / D.max_c_cg;
    const int c = tid - i * D.max_c_cg;

    const T* Ai = A1 + (int64_t)i * D.D_A1;
    T* B2i = B2_buf + (int64_t)i * D.B2_tot;

    // Load A1[l][c][0..2l] into registers (at most 16 scalars!)
    T a_reg[MAX_SH];
    #pragma unroll
    for (int m = 0; m < MAX_SH; ++m) a_reg[m] = T(0);
    for (int l = 0; l <= D.lmax; ++l) {
        if (c < D.n_l[l]) {
            const int dm = 2 * l + 1;
            const T* src = Ai + D.a1_off[l] + c * dm;
            for (int m = 0; m < dm; ++m) a_reg[l * l + m] = src[m];
            // Write nu1 part of B_cat[l]
            if (l == 0) {
                Bcat0[(int64_t)i * D.bcat_dim[0] + c] = src[0];
            } else if (l == 1 && D.lmsg >= 1 && Bcat1) {
                T* dst = Bcat1 + ((int64_t)i * D.bcat_dim[1] + c) * 3;
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            } else if (l == 2 && D.lmsg >= 2 && Bcat2) {
                T* dst = Bcat2 + ((int64_t)i * D.bcat_dim[2] + c) * 5;
                for (int m = 0; m < 5; ++m) dst[m] = src[m];
            }
        }
    }

    // nu2 paths
    for (int p = 0; p < D.n_nu2; ++p) {
        const Nu2Path pt = nu2[p];
        if (c >= pt.c) continue;
        const int dL = 2 * pt.L + 1;
        const T* u = a_reg + pt.l1 * pt.l1;
        const T* v = a_reg + pt.l2 * pt.l2;
        T out[5] = {T(0), T(0), T(0), T(0), T(0)};
        for (int e = pt.cg_start; e < pt.cg_start + pt.cg_count; ++e) {
            const CgEntry<T> ce = cg_tab[e];
            out[ce.m] += ce.val * u[ce.a] * v[ce.b];
        }
        T* b2_dst = B2i + pt.dst_b2_off + c * dL;
        for (int m = 0; m < dL; ++m) b2_dst[m] = out[m];
        if (pt.dst_bcat_off >= 0) {
            if (pt.L == 0) {
                Bcat0[(int64_t)i * D.bcat_dim[0] + pt.dst_bcat_off + c] = out[0];
            } else if (pt.L == 1 && Bcat1) {
                T* dst = Bcat1 + ((int64_t)i * D.bcat_dim[1] + pt.dst_bcat_off + c) * 3;
                dst[0] = out[0]; dst[1] = out[1]; dst[2] = out[2];
            } else if (pt.L == 2 && Bcat2) {
                T* dst = Bcat2 + ((int64_t)i * D.bcat_dim[2] + pt.dst_bcat_off + c) * 5;
                for (int m = 0; m < 5; ++m) dst[m] = out[m];
            }
        }
    }

    // nu3 paths
    for (int p = 0; p < D.n_nu3; ++p) {
        const Nu3Path pt = nu3[p];
        if (c >= pt.c) continue;
        const Nu2Path p2 = nu2[pt.p2_idx];
        const int d12 = 2 * p2.L + 1;
        const int dL = 2 * pt.L + 1;
        const T* u = B2i + p2.dst_b2_off + c * d12;
        const T* v = a_reg + pt.l3 * pt.l3;
        T out[5] = {T(0), T(0), T(0), T(0), T(0)};
        for (int e = pt.cg_start; e < pt.cg_start + pt.cg_count; ++e) {
            const CgEntry<T> ce = cg_tab[e];
            out[ce.m] += ce.val * u[ce.a] * v[ce.b];
        }
        if (pt.L == 0) {
            Bcat0[(int64_t)i * D.bcat_dim[0] + pt.dst_bcat_off + c] = out[0];
        } else if (pt.L == 1 && Bcat1) {
            T* dst = Bcat1 + ((int64_t)i * D.bcat_dim[1] + pt.dst_bcat_off + c) * 3;
            dst[0] = out[0]; dst[1] = out[1]; dst[2] = out[2];
        } else if (pt.L == 2 && Bcat2) {
            T* dst = Bcat2 + ((int64_t)i * D.bcat_dim[2] + pt.dst_bcat_off + c) * 5;
            for (int m = 0; m < 5; ++m) dst[m] = out[m];
        }
    }
}

template <typename T>
__global__ void k_cg_bwd1(
    int N, ModelDims D,
    const Nu2Path* __restrict__ nu2,
    const Nu3Path* __restrict__ nu3,
    const CgEntry<T>* __restrict__ cg_tab,
    const T* __restrict__ A1,
    const T* __restrict__ B2_buf,
    const T* __restrict__ g_Bcat0,
    const T* __restrict__ g_Bcat1,
    const T* __restrict__ g_Bcat2,
    T* __restrict__ g_B2_scratch, // [N, B2_tot]
    T* __restrict__ g_A1          // [N, D_A1]
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * D.max_c_cg) return;
    const int i = tid / D.max_c_cg;
    const int c = tid - i * D.max_c_cg;

    const T* Ai = A1 + (int64_t)i * D.D_A1;
    const T* B2i = B2_buf + (int64_t)i * D.B2_tot;
    T* gB2i = g_B2_scratch + (int64_t)i * D.B2_tot;

    T a_reg[MAX_SH], ga_reg[MAX_SH];
    #pragma unroll
    for (int m = 0; m < MAX_SH; ++m) { a_reg[m] = T(0); ga_reg[m] = T(0); }
    for (int l = 0; l <= D.lmax; ++l) {
        if (c < D.n_l[l]) {
            const int dm = 2 * l + 1;
            const T* src = Ai + D.a1_off[l] + c * dm;
            for (int m = 0; m < dm; ++m) a_reg[l * l + m] = src[m];
            if (l == 0) {
                ga_reg[0] = g_Bcat0[(int64_t)i * D.bcat_dim[0] + c];
            } else if (l == 1 && D.lmsg >= 1 && g_Bcat1) {
                const T* s = g_Bcat1 + ((int64_t)i * D.bcat_dim[1] + c) * 3;
                ga_reg[1] = s[0]; ga_reg[2] = s[1]; ga_reg[3] = s[2];
            } else if (l == 2 && D.lmsg >= 2 && g_Bcat2) {
                const T* s = g_Bcat2 + ((int64_t)i * D.bcat_dim[2] + c) * 5;
                for (int m = 0; m < 5; ++m) ga_reg[4 + m] = s[m];
            }
        }
    }

    // Initialize g_B2 from direct nu2 contributions to B_cat
    for (int p = 0; p < D.n_nu2; ++p) {
        const Nu2Path pt = nu2[p];
        if (c >= pt.c) continue;
        const int dL = 2 * pt.L + 1;
        T* dst = gB2i + pt.dst_b2_off + c * dL;
        if (pt.dst_bcat_off >= 0) {
            if (pt.L == 0) {
                dst[0] = g_Bcat0[(int64_t)i * D.bcat_dim[0] + pt.dst_bcat_off + c];
            } else if (pt.L == 1 && g_Bcat1) {
                const T* s = g_Bcat1 + ((int64_t)i * D.bcat_dim[1] + pt.dst_bcat_off + c) * 3;
                dst[0] = s[0]; dst[1] = s[1]; dst[2] = s[2];
            } else if (pt.L == 2 && g_Bcat2) {
                const T* s = g_Bcat2 + ((int64_t)i * D.bcat_dim[2] + pt.dst_bcat_off + c) * 5;
                for (int m = 0; m < 5; ++m) dst[m] = s[m];
            } else {
                for (int m = 0; m < dL; ++m) dst[m] = T(0);
            }
        } else {
            for (int m = 0; m < dL; ++m) dst[m] = T(0);
        }
    }

    // Reverse nu3
    for (int p = 0; p < D.n_nu3; ++p) {
        const Nu3Path pt = nu3[p];
        if (c >= pt.c) continue;
        const Nu2Path p2 = nu2[pt.p2_idx];
        const int d12 = 2 * p2.L + 1;
        T gw[5] = {T(0), T(0), T(0), T(0), T(0)};
        if (pt.L == 0) {
            gw[0] = g_Bcat0[(int64_t)i * D.bcat_dim[0] + pt.dst_bcat_off + c];
        } else if (pt.L == 1 && g_Bcat1) {
            const T* s = g_Bcat1 + ((int64_t)i * D.bcat_dim[1] + pt.dst_bcat_off + c) * 3;
            gw[0] = s[0]; gw[1] = s[1]; gw[2] = s[2];
        } else if (pt.L == 2 && g_Bcat2) {
            const T* s = g_Bcat2 + ((int64_t)i * D.bcat_dim[2] + pt.dst_bcat_off + c) * 5;
            for (int m = 0; m < 5; ++m) gw[m] = s[m];
        }
        const T* u = B2i + p2.dst_b2_off + c * d12;
        T* gu = gB2i + p2.dst_b2_off + c * d12;
        const T* v = a_reg + pt.l3 * pt.l3;
        T* gv = ga_reg + pt.l3 * pt.l3;
        for (int e = pt.cg_start; e < pt.cg_start + pt.cg_count; ++e) {
            const CgEntry<T> ce = cg_tab[e];
            const T cw = ce.val * gw[ce.m];
            gu[ce.a] += cw * v[ce.b];
            gv[ce.b] += cw * u[ce.a];
        }
    }

    // Reverse nu2
    for (int p = 0; p < D.n_nu2; ++p) {
        const Nu2Path pt = nu2[p];
        if (c >= pt.c) continue;
        const int dL = 2 * pt.L + 1;
        const T* gw = gB2i + pt.dst_b2_off + c * dL;
        const T* u = a_reg + pt.l1 * pt.l1;
        const T* v = a_reg + pt.l2 * pt.l2;
        T* gu = ga_reg + pt.l1 * pt.l1;
        T* gv = ga_reg + pt.l2 * pt.l2;
        for (int e = pt.cg_start; e < pt.cg_start + pt.cg_count; ++e) {
            const CgEntry<T> ce = cg_tab[e];
            const T cw = ce.val * gw[ce.m];
            gu[ce.a] += cw * v[ce.b];
            gv[ce.b] += cw * u[ce.a];
        }
    }

    // Write g_A1
    T* gAi = g_A1 + (int64_t)i * D.D_A1;
    for (int l = 0; l <= D.lmax; ++l) {
        if (c < D.n_l[l]) {
            const int dm = 2 * l + 1;
            T* dst = gAi + D.a1_off[l] + c * dm;
            for (int m = 0; m < dm; ++m) dst[m] = ga_reg[l * l + m];
        }
    }
}

template <typename T>
__global__ void k_cg_dual_fwd(
    int N, ModelDims D,
    const Nu2Path* __restrict__ nu2,
    const Nu3Path* __restrict__ nu3,
    const CgEntry<T>* __restrict__ cg_tab,
    const T* __restrict__ A1,
    const T* __restrict__ dot_A1,
    const T* __restrict__ B2_buf,
    T* __restrict__ dot_B2_buf,
    T* __restrict__ dot_Bcat0,
    T* __restrict__ dot_Bcat1,
    T* __restrict__ dot_Bcat2
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * D.max_c_cg) return;
    const int i = tid / D.max_c_cg;
    const int c = tid - i * D.max_c_cg;

    const T* Ai = A1 + (int64_t)i * D.D_A1;
    const T* dAi = dot_A1 + (int64_t)i * D.D_A1;
    const T* B2i = B2_buf + (int64_t)i * D.B2_tot;
    T* dB2i = dot_B2_buf + (int64_t)i * D.B2_tot;

    T a_reg[MAX_SH], da_reg[MAX_SH];
    #pragma unroll
    for (int m = 0; m < MAX_SH; ++m) { a_reg[m] = T(0); da_reg[m] = T(0); }
    for (int l = 0; l <= D.lmax; ++l) {
        if (c < D.n_l[l]) {
            const int dm = 2 * l + 1;
            const T* s = Ai + D.a1_off[l] + c * dm;
            const T* ds = dAi + D.a1_off[l] + c * dm;
            for (int m = 0; m < dm; ++m) { a_reg[l * l + m] = s[m]; da_reg[l * l + m] = ds[m]; }
            if (l == 0) {
                dot_Bcat0[(int64_t)i * D.bcat_dim[0] + c] = ds[0];
            } else if (l == 1 && D.lmsg >= 1 && dot_Bcat1) {
                T* dst = dot_Bcat1 + ((int64_t)i * D.bcat_dim[1] + c) * 3;
                dst[0] = ds[0]; dst[1] = ds[1]; dst[2] = ds[2];
            } else if (l == 2 && D.lmsg >= 2 && dot_Bcat2) {
                T* dst = dot_Bcat2 + ((int64_t)i * D.bcat_dim[2] + c) * 5;
                for (int m = 0; m < 5; ++m) dst[m] = ds[m];
            }
        }
    }

    for (int p = 0; p < D.n_nu2; ++p) {
        const Nu2Path pt = nu2[p];
        if (c >= pt.c) continue;
        const int dL = 2 * pt.L + 1;
        const T* u = a_reg + pt.l1 * pt.l1;
        const T* du = da_reg + pt.l1 * pt.l1;
        const T* v = a_reg + pt.l2 * pt.l2;
        const T* dv = da_reg + pt.l2 * pt.l2;
        T dout[5] = {T(0), T(0), T(0), T(0), T(0)};
        for (int e = pt.cg_start; e < pt.cg_start + pt.cg_count; ++e) {
            const CgEntry<T> ce = cg_tab[e];
            dout[ce.m] += ce.val * (du[ce.a] * v[ce.b] + u[ce.a] * dv[ce.b]);
        }
        T* dst_b2 = dB2i + pt.dst_b2_off + c * dL;
        for (int m = 0; m < dL; ++m) dst_b2[m] = dout[m];
        if (pt.dst_bcat_off >= 0) {
            if (pt.L == 0) {
                dot_Bcat0[(int64_t)i * D.bcat_dim[0] + pt.dst_bcat_off + c] = dout[0];
            } else if (pt.L == 1 && dot_Bcat1) {
                T* dst = dot_Bcat1 + ((int64_t)i * D.bcat_dim[1] + pt.dst_bcat_off + c) * 3;
                dst[0] = dout[0]; dst[1] = dout[1]; dst[2] = dout[2];
            } else if (pt.L == 2 && dot_Bcat2) {
                T* dst = dot_Bcat2 + ((int64_t)i * D.bcat_dim[2] + pt.dst_bcat_off + c) * 5;
                for (int m = 0; m < 5; ++m) dst[m] = dout[m];
            }
        }
    }

    for (int p = 0; p < D.n_nu3; ++p) {
        const Nu3Path pt = nu3[p];
        if (c >= pt.c) continue;
        const Nu2Path p2 = nu2[pt.p2_idx];
        const int d12 = 2 * p2.L + 1;
        const T* u = B2i + p2.dst_b2_off + c * d12;
        const T* du = dB2i + p2.dst_b2_off + c * d12;
        const T* v = a_reg + pt.l3 * pt.l3;
        const T* dv = da_reg + pt.l3 * pt.l3;
        T dout[5] = {T(0), T(0), T(0), T(0), T(0)};
        for (int e = pt.cg_start; e < pt.cg_start + pt.cg_count; ++e) {
            const CgEntry<T> ce = cg_tab[e];
            dout[ce.m] += ce.val * (du[ce.a] * v[ce.b] + u[ce.a] * dv[ce.b]);
        }
        if (pt.L == 0) {
            dot_Bcat0[(int64_t)i * D.bcat_dim[0] + pt.dst_bcat_off + c] = dout[0];
        } else if (pt.L == 1 && dot_Bcat1) {
            T* dst = dot_Bcat1 + ((int64_t)i * D.bcat_dim[1] + pt.dst_bcat_off + c) * 3;
            dst[0] = dout[0]; dst[1] = dout[1]; dst[2] = dout[2];
        } else if (pt.L == 2 && dot_Bcat2) {
            T* dst = dot_Bcat2 + ((int64_t)i * D.bcat_dim[2] + pt.dst_bcat_off + c) * 5;
            for (int m = 0; m < 5; ++m) dst[m] = dout[m];
        }
    }
}

template <typename T>
__global__ void k_cg_dual_bwd(
    int N, ModelDims D,
    const Nu2Path* __restrict__ nu2,
    const Nu3Path* __restrict__ nu3,
    const CgEntry<T>* __restrict__ cg_tab,
    const T* __restrict__ A1,
    const T* __restrict__ dot_A1,
    const T* __restrict__ B2_buf,
    const T* __restrict__ dot_B2_buf,
    const T* __restrict__ bar_Bcat0, const T* __restrict__ hat_Bcat0,
    const T* __restrict__ bar_Bcat1, const T* __restrict__ hat_Bcat1,
    const T* __restrict__ bar_Bcat2, const T* __restrict__ hat_Bcat2,
    T* __restrict__ bar_B2_s, T* __restrict__ hat_B2_s,
    T* __restrict__ bar_A1,   T* __restrict__ hat_A1
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * D.max_c_cg) return;
    const int i = tid / D.max_c_cg;
    const int c = tid - i * D.max_c_cg;

    const T* Ai = A1 + (int64_t)i * D.D_A1;
    const T* dAi = dot_A1 + (int64_t)i * D.D_A1;
    const T* B2i = B2_buf + (int64_t)i * D.B2_tot;
    const T* dB2i = dot_B2_buf + (int64_t)i * D.B2_tot;
    T* bB2i = bar_B2_s + (int64_t)i * D.B2_tot;
    T* hB2i = hat_B2_s + (int64_t)i * D.B2_tot;

    T a_reg[MAX_SH], da_reg[MAX_SH], ba_reg[MAX_SH], ha_reg[MAX_SH];
    #pragma unroll
    for (int m = 0; m < MAX_SH; ++m) { a_reg[m] = da_reg[m] = ba_reg[m] = ha_reg[m] = T(0); }

    for (int l = 0; l <= D.lmax; ++l) {
        if (c < D.n_l[l]) {
            const int dm = 2 * l + 1;
            const T* s = Ai + D.a1_off[l] + c * dm;
            const T* ds = dAi + D.a1_off[l] + c * dm;
            for (int m = 0; m < dm; ++m) { a_reg[l * l + m] = s[m]; da_reg[l * l + m] = ds[m]; }
            if (l == 0) {
                ba_reg[0] = bar_Bcat0[(int64_t)i * D.bcat_dim[0] + c];
                ha_reg[0] = hat_Bcat0[(int64_t)i * D.bcat_dim[0] + c];
            } else if (l == 1 && D.lmsg >= 1 && bar_Bcat1) {
                const int off = ((int64_t)i * D.bcat_dim[1] + c) * 3;
                for (int m = 0; m < 3; ++m) { ba_reg[1 + m] = bar_Bcat1[off + m]; ha_reg[1 + m] = hat_Bcat1[off + m]; }
            } else if (l == 2 && D.lmsg >= 2 && bar_Bcat2) {
                const int off = ((int64_t)i * D.bcat_dim[2] + c) * 5;
                for (int m = 0; m < 5; ++m) { ba_reg[4 + m] = bar_Bcat2[off + m]; ha_reg[4 + m] = hat_Bcat2[off + m]; }
            }
        }
    }

    for (int p = 0; p < D.n_nu2; ++p) {
        const Nu2Path pt = nu2[p];
        if (c >= pt.c) continue;
        const int dL = 2 * pt.L + 1;
        T* bd = bB2i + pt.dst_b2_off + c * dL;
        T* hd = hB2i + pt.dst_b2_off + c * dL;
        for (int m = 0; m < dL; ++m) { bd[m] = T(0); hd[m] = T(0); }
        if (pt.dst_bcat_off >= 0) {
            if (pt.L == 0) {
                const int off = (int64_t)i * D.bcat_dim[0] + pt.dst_bcat_off + c;
                bd[0] = bar_Bcat0[off]; hd[0] = hat_Bcat0[off];
            } else if (pt.L == 1 && bar_Bcat1) {
                const int off = ((int64_t)i * D.bcat_dim[1] + pt.dst_bcat_off + c) * 3;
                for (int m = 0; m < 3; ++m) { bd[m] = bar_Bcat1[off + m]; hd[m] = hat_Bcat1[off + m]; }
            } else if (pt.L == 2 && bar_Bcat2) {
                const int off = ((int64_t)i * D.bcat_dim[2] + pt.dst_bcat_off + c) * 5;
                for (int m = 0; m < 5; ++m) { bd[m] = bar_Bcat2[off + m]; hd[m] = hat_Bcat2[off + m]; }
            }
        }
    }

    for (int p = 0; p < D.n_nu3; ++p) {
        const Nu3Path pt = nu3[p];
        if (c >= pt.c) continue;
        const Nu2Path p2 = nu2[pt.p2_idx];
        const int d12 = 2 * p2.L + 1;
        T bw[5] = {T(0)}, hw[5] = {T(0)};
        if (pt.L == 0) {
            const int off = (int64_t)i * D.bcat_dim[0] + pt.dst_bcat_off + c;
            bw[0] = bar_Bcat0[off]; hw[0] = hat_Bcat0[off];
        } else if (pt.L == 1 && bar_Bcat1) {
            const int off = ((int64_t)i * D.bcat_dim[1] + pt.dst_bcat_off + c) * 3;
            for (int m = 0; m < 3; ++m) { bw[m] = bar_Bcat1[off + m]; hw[m] = hat_Bcat1[off + m]; }
        } else if (pt.L == 2 && bar_Bcat2) {
            const int off = ((int64_t)i * D.bcat_dim[2] + pt.dst_bcat_off + c) * 5;
            for (int m = 0; m < 5; ++m) { bw[m] = bar_Bcat2[off + m]; hw[m] = hat_Bcat2[off + m]; }
        }
        const T* u = B2i + p2.dst_b2_off + c * d12;
        const T* du = dB2i + p2.dst_b2_off + c * d12;
        T* bu = bB2i + p2.dst_b2_off + c * d12;
        T* hu = hB2i + p2.dst_b2_off + c * d12;
        const T* v = a_reg + pt.l3 * pt.l3;
        const T* dv = da_reg + pt.l3 * pt.l3;
        T* bv = ba_reg + pt.l3 * pt.l3;
        T* hv = ha_reg + pt.l3 * pt.l3;
        for (int e = pt.cg_start; e < pt.cg_start + pt.cg_count; ++e) {
            const CgEntry<T> ce = cg_tab[e];
            const T cb = ce.val * bw[ce.m];
            const T ch = ce.val * hw[ce.m];
            bu[ce.a] += cb * v[ce.b] + ch * dv[ce.b];
            hu[ce.a] += ch * v[ce.b];
            bv[ce.b] += cb * u[ce.a] + ch * du[ce.a];
            hv[ce.b] += ch * u[ce.a];
        }
    }

    for (int p = 0; p < D.n_nu2; ++p) {
        const Nu2Path pt = nu2[p];
        if (c >= pt.c) continue;
        const int dL = 2 * pt.L + 1;
        const T* bw = bB2i + pt.dst_b2_off + c * dL;
        const T* hw = hB2i + pt.dst_b2_off + c * dL;
        const T* u = a_reg + pt.l1 * pt.l1;
        const T* du = da_reg + pt.l1 * pt.l1;
        const T* v = a_reg + pt.l2 * pt.l2;
        const T* dv = da_reg + pt.l2 * pt.l2;
        T* bu = ba_reg + pt.l1 * pt.l1;
        T* hu = ha_reg + pt.l1 * pt.l1;
        T* bv = ba_reg + pt.l2 * pt.l2;
        T* hv = ha_reg + pt.l2 * pt.l2;
        for (int e = pt.cg_start; e < pt.cg_start + pt.cg_count; ++e) {
            const CgEntry<T> ce = cg_tab[e];
            const T cb = ce.val * bw[ce.m];
            const T ch = ce.val * hw[ce.m];
            bu[ce.a] += cb * v[ce.b] + ch * dv[ce.b];
            hu[ce.a] += ch * v[ce.b];
            bv[ce.b] += cb * u[ce.a] + ch * du[ce.a];
            hv[ce.b] += ch * u[ce.a];
        }
    }

    T* bAi = bar_A1 + (int64_t)i * D.D_A1;
    T* hAi = hat_A1 + (int64_t)i * D.D_A1;
    for (int l = 0; l <= D.lmax; ++l) {
        if (c < D.n_l[l]) {
            const int dm = 2 * l + 1;
            T* bd = bAi + D.a1_off[l] + c * dm;
            T* hd = hAi + D.a1_off[l] + c * dm;
            for (int m = 0; m < dm; ++m) { bd[m] = ba_reg[l * l + m]; hd[m] = ha_reg[l * l + m]; }
        }
    }
}

// =============================================================================
// 6. Normalization & Elementwise SiLU Kernels (Readout MLP + Pol MLP)
// =============================================================================
template <typename T>
__global__ void k_norm_x1_fwd(
    int N, int D0, const T* __restrict__ x1, const T* __restrict__ mu, const T* __restrict__ inv_std,
    T* __restrict__ x_norm
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D0) return;
    const int d = idx % D0;
    x_norm[idx] = (x1[idx] - mu[d]) * inv_std[d];
}

template <typename T>
__global__ void k_scale_by_vec(int N, int D0, const T* __restrict__ in, const T* __restrict__ sc, T* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D0) return;
    out[idx] = in[idx] * sc[idx % D0];
}

template <typename T>
__global__ void k_bias_alpha_silu_fwd(
    int N, int W, const int* __restrict__ z_idx,
    const T* __restrict__ bias, const T* __restrict__ alpha_Z, // alpha_Z optional
    T* __restrict__ pre, T* __restrict__ act
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * W) return;
    const int i = idx / W, w = idx - i * W;
    T v = pre[idx] + bias[w];
    if (alpha_Z) v += alpha_Z[z_idx[i] * W + w];
    pre[idx] = v;
    T y, dy, d2y;
    silu_012(v, y, dy, d2y);
    act[idx] = y;
}

template <typename T>
__global__ void k_silu_bwd_mul(
    int N, int W, const T* __restrict__ pre, const T* __restrict__ g_act, T* __restrict__ g_pre
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * W) return;
    T y, dy, d2y;
    silu_012(pre[idx], y, dy, d2y);
    g_pre[idx] = g_act[idx] * dy;
}

template <typename T>
__global__ void k_silu_dual_bwd(
    int N, int W,
    const T* __restrict__ pre, const T* __restrict__ dot_pre,
    const T* __restrict__ bar_act, const T* __restrict__ hat_act,
    T* __restrict__ bar_pre, T* __restrict__ hat_pre
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * W) return;
    T y, dy, d2y;
    silu_012(pre[idx], y, dy, d2y);
    const T ha = hat_act[idx];
    hat_pre[idx] = dy * ha;
    bar_pre[idx] = dy * bar_act[idx] + d2y * dot_pre[idx] * ha;
}

// Parallel zero-atomic reduction: 1 block of 128 threads per column w in [0, W)
template <typename T>
__global__ void k_reduce_bias_alpha(
    int N, int S, int W, const int* __restrict__ z_idx,
    const T* __restrict__ bar_pre,
    T* __restrict__ g_bias, T* __restrict__ g_alpha_Z
) {
    __shared__ T sm[4];
    const int w = blockIdx.x;
    if (w >= W) return;
    if (g_alpha_Z) {
        T per_s[32];
        for (int s = 0; s < S; ++s) per_s[s] = T(0);
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            per_s[z_idx[i]] += bar_pre[(int64_t)i * W + w];
        }
        T tot_all = T(0);
        for (int s = 0; s < S; ++s) {
            T wv = warp_sum(per_s[s]);
            if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = wv;
            __syncthreads();
            if (threadIdx.x == 0) {
                T sv = sm[0] + sm[1] + sm[2] + sm[3];
                g_alpha_Z[s * W + w] += sv;
                tot_all += sv;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) g_bias[w] += tot_all;
    } else {
        T acc = T(0);
        for (int i = threadIdx.x; i < N; i += blockDim.x) acc += bar_pre[(int64_t)i * W + w];
        acc = warp_sum(acc);
        if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x == 0) g_bias[w] += sm[0] + sm[1] + sm[2] + sm[3];
    }
}

// =============================================================================
// 7. Generic Vector Soft-Cap Device Helpers (used by mp_cap & ef_cap!)
// =============================================================================
template <typename T, int DIM>
__device__ __forceinline__ void vcap_fwd(const T* x, T inv_cap2, T* y) {
    if (inv_cap2 <= T(0)) {
        #pragma unroll
        for (int m = 0; m < DIM; ++m) y[m] = x[m];
        return;
    }
    T s = T(0);
    #pragma unroll
    for (int m = 0; m < DIM; ++m) s += x[m] * x[m];
    const T f = t_rsqrt(T(1) + s * inv_cap2);
    #pragma unroll
    for (int m = 0; m < DIM; ++m) y[m] = f * x[m];
}

template <typename T, int DIM>
__device__ __forceinline__ void vcap_bwd1(const T* x, const T* cy, T inv_cap2, T* gx) {
    if (inv_cap2 <= T(0)) {
        #pragma unroll
        for (int m = 0; m < DIM; ++m) gx[m] = cy[m];
        return;
    }
    T s = T(0), d1 = T(0);
    #pragma unroll
    for (int m = 0; m < DIM; ++m) { s += x[m] * x[m]; d1 += cy[m] * x[m]; }
    const T u = T(1) + s * inv_cap2;
    const T f = t_rsqrt(u);
    const T k1 = -inv_cap2 / u;
    const T cx = f * k1 * d1;
    #pragma unroll
    for (int m = 0; m < DIM; ++m) gx[m] = f * cy[m] + cx * x[m];
}

template <typename T, int DIM>
__device__ __forceinline__ void vcap_dual_fwd(const T* x, const T* dx, T inv_cap2, T* y, T* dy) {
    if (inv_cap2 <= T(0)) {
        #pragma unroll
        for (int m = 0; m < DIM; ++m) { y[m] = x[m]; dy[m] = dx[m]; }
        return;
    }
    T s = T(0), dd = T(0);
    #pragma unroll
    for (int m = 0; m < DIM; ++m) { s += x[m] * x[m]; dd += x[m] * dx[m]; }
    const T u = T(1) + s * inv_cap2;
    const T f = t_rsqrt(u);
    const T df = f * (-inv_cap2 / u) * dd;
    #pragma unroll
    for (int m = 0; m < DIM; ++m) { y[m] = f * x[m]; dy[m] = f * dx[m] + df * x[m]; }
}

template <typename T, int DIM>
__device__ __forceinline__ void vcap_dual_bwd(
    const T* x, const T* dx, const T* by, const T* hy, T inv_cap2, T* bx, T* hx
) {
    if (inv_cap2 <= T(0)) {
        #pragma unroll
        for (int m = 0; m < DIM; ++m) { bx[m] = by[m]; hx[m] = hy[m]; }
        return;
    }
    T s = T(0), A = T(0), B = T(0), C = T(0), DD = T(0);
    #pragma unroll
    for (int m = 0; m < DIM; ++m) {
        s += x[m] * x[m]; A += by[m] * x[m]; B += hy[m] * dx[m];
        C += hy[m] * x[m]; DD += x[m] * dx[m];
    }
    const T u = T(1) + s * inv_cap2;
    const T f = t_rsqrt(u);
    const T k1 = -inv_cap2 / u;
    const T fp = T(0.5) * k1 * f;
    const T fpp = T(0.75) * k1 * k1 * f;
    const T c_hy = T(2) * fp * DD;
    const T c_dx = T(2) * fp * C;
    const T c_x  = T(2) * fp * (A + B) + T(4) * fpp * DD * C;
    #pragma unroll
    for (int m = 0; m < DIM; ++m) {
        hx[m] = f * hy[m] + c_dx * x[m];
        bx[m] = f * by[m] + c_hy * hy[m] + c_dx * dx[m] + c_x * x[m];
    }
}

// =============================================================================
// 8. NodeBlock1 Heads: e_site, q, p, Q (Fwd, Bwd1, Dual-Fwd, Dual-Bwd)
// =============================================================================
template <typename T>
__global__ void k_nb1_heads_fwd(
    int N, ModelDims D,
    const int* __restrict__ z_idx,
    const T* __restrict__ h,       // [N, W]
    const T* __restrict__ Bcat1,   // [N, b1, 3]
    const T* __restrict__ Bcat2,   // [N, b2, 5]
    const T* __restrict__ inv_sb1, // [b1]
    const T* __restrict__ inv_sb2, // [b2]
    const T* __restrict__ sw,      // [N]
    const T* __restrict__ u_Z,     // [S, W]
    const T* __restrict__ W_qA, const T* __restrict__ b_qA,
    const T* __restrict__ W_p1A, const T* __restrict__ W_g1A, const T* __restrict__ b_g1A,
    const T* __restrict__ W_p2A, const T* __restrict__ W_g2A, const T* __restrict__ b_g2A,
    T* __restrict__ e_site,
    T* __restrict__ q_raw, T* __restrict__ q_cap, T* __restrict__ q_out,
    T* __restrict__ v1_out, T* __restrict__ g1_out, T* __restrict__ p_raw, T* __restrict__ p_cap, T* __restrict__ p_out,
    T* __restrict__ v2_out, T* __restrict__ g2_out, T* __restrict__ Q_raw, T* __restrict__ Q_cap, T* __restrict__ Q_out
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int z = z_idx[i];
    const int W = D.ro_width;
    const T* hi = h + (int64_t)i * W;
    const T es = (T)D.energy_scale;
    const T inv_cap2 = (D.mp_cap > 0.0) ? (T)(1.0 / (D.mp_cap * D.mp_cap)) : T(0);
    const T sw_m = (D.use_zbl_switch && D.sw_mp) ? sw[i] : T(1);

    // e_site
    {
        const T* uz = u_Z + z * W;
        T acc = T(0);
        for (int w = 0; w < W; ++w) acc += hi[w] * uz[w];
        e_site[i] = acc * es;
    }

    // q
    for (int c = 0; c < D.c0; ++c) {
        const T* wq = W_qA + (z * D.c0 + c) * W;
        T v = b_qA[z * D.c0 + c];
        for (int w = 0; w < W; ++w) v += wq[w] * hi[w];
        T vc;
        vcap_fwd<T, 1>(&v, inv_cap2, &vc);
        q_raw[(int64_t)i * D.c0 + c] = v;
        q_cap[(int64_t)i * D.c0 + c] = vc;
        q_out[(int64_t)i * D.c0 + c] = sw_m * vc;
    }

    // p
    if (D.c1 > 0) {
        const int b1 = D.bcat_dim[1];
        const T* B1i = Bcat1 + (int64_t)i * b1 * 3;
        for (int c = 0; c < D.c1; ++c) {
            const T* wg = W_g1A + (z * D.c1 + c) * W;
            T g1 = T(1) + b_g1A[z * D.c1 + c];
            for (int w = 0; w < W; ++w) g1 += wg[w] * hi[w];
            const T* wp = W_p1A + c * b1;
            T vv[3] = {T(0), T(0), T(0)};
            for (int k = 0; k < b1; ++k) {
                const T wk = wp[k] * inv_sb1[k];
                vv[0] += wk * B1i[3 * k + 0];
                vv[1] += wk * B1i[3 * k + 1];
                vv[2] += wk * B1i[3 * k + 2];
            }
            T pr[3] = {vv[0] * g1, vv[1] * g1, vv[2] * g1};
            T pc[3];
            vcap_fwd<T, 3>(pr, inv_cap2, pc);
            const int64_t off = ((int64_t)i * D.c1 + c) * 3;
            g1_out[(int64_t)i * D.c1 + c] = g1;
            for (int m = 0; m < 3; ++m) {
                v1_out[off + m] = vv[m];
                p_raw[off + m] = pr[m];
                p_cap[off + m] = pc[m];
                p_out[off + m] = sw_m * pc[m];
            }
        }
    }

    // Q
    if (D.c2 > 0) {
        const int b2 = D.bcat_dim[2];
        const T* B2i = Bcat2 + (int64_t)i * b2 * 5;
        for (int c = 0; c < D.c2; ++c) {
            const T* wg = W_g2A + (z * D.c2 + c) * W;
            T g2 = T(1) + b_g2A[z * D.c2 + c];
            for (int w = 0; w < W; ++w) g2 += wg[w] * hi[w];
            const T* wp = W_p2A + c * b2;
            T vv[5] = {T(0), T(0), T(0), T(0), T(0)};
            for (int k = 0; k < b2; ++k) {
                const T wk = wp[k] * inv_sb2[k];
                for (int m = 0; m < 5; ++m) vv[m] += wk * B2i[5 * k + m];
            }
            T qr[5], qc[5];
            for (int m = 0; m < 5; ++m) qr[m] = vv[m] * g2;
            vcap_fwd<T, 5>(qr, inv_cap2, qc);
            const int64_t off = ((int64_t)i * D.c2 + c) * 5;
            g2_out[(int64_t)i * D.c2 + c] = g2;
            for (int m = 0; m < 5; ++m) {
                v2_out[off + m] = vv[m];
                Q_raw[off + m] = qr[m];
                Q_cap[off + m] = qc[m];
                Q_out[off + m] = sw_m * qc[m];
            }
        }
    }
}

template <typename T>
__global__ void k_nb1_heads_bwd1(
    int N, ModelDims D,
    const int* __restrict__ z_idx,
    const T* __restrict__ sw,
    const T* __restrict__ c_esite, // [N]
    const T* __restrict__ c_h_pol, // [N, W] or nullptr
    const T* __restrict__ c_q, const T* __restrict__ c_p, const T* __restrict__ c_Q,
    const T* __restrict__ q_raw, const T* __restrict__ q_cap,
    const T* __restrict__ v1, const T* __restrict__ g1, const T* __restrict__ p_raw, const T* __restrict__ p_cap,
    const T* __restrict__ v2, const T* __restrict__ g2, const T* __restrict__ Q_raw, const T* __restrict__ Q_cap,
    const T* __restrict__ inv_sb1, const T* __restrict__ inv_sb2,
    const T* __restrict__ u_Z,
    const T* __restrict__ W_qA,
    const T* __restrict__ W_p1A, const T* __restrict__ W_g1A,
    const T* __restrict__ W_p2A, const T* __restrict__ W_g2A,
    T* __restrict__ c_h_out,       // [N, W]
    T* __restrict__ c_Bcat1_out,   // [N, b1, 3]
    T* __restrict__ c_Bcat2_out,   // [N, b2, 5]
    T* __restrict__ c_sw_io        // [N], adds sw_mp multipole contributions if use_zbl_switch && sw_mp
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int z = z_idx[i];
    const int W = D.ro_width;
    const T es = (T)D.energy_scale;
    const T inv_cap2 = (D.mp_cap > 0.0) ? (T)(1.0 / (D.mp_cap * D.mp_cap)) : T(0);
    const bool do_sw = (D.use_zbl_switch && D.sw_mp);
    const T sw_m = do_sw ? sw[i] : T(1);

    T* chi = c_h_out + (int64_t)i * W;
    const T ce = c_esite[i] * es;
    const T* uz = u_Z + z * W;
    const T* chp = c_h_pol ? (c_h_pol + (int64_t)i * W) : nullptr;
    for (int w = 0; w < W; ++w) chi[w] = ce * uz[w] + (chp ? chp[w] : T(0));

    T csw_add = T(0);

    // q
    for (int c = 0; c < D.c0; ++c) {
        const T cq = c_q[(int64_t)i * D.c0 + c];
        if (do_sw) csw_add += cq * q_cap[(int64_t)i * D.c0 + c];
        const T cq_cap = sw_m * cq;
        T cq_raw;
        vcap_bwd1<T, 1>(q_raw + (int64_t)i * D.c0 + c, &cq_cap, inv_cap2, &cq_raw);
        const T* wq = W_qA + (z * D.c0 + c) * W;
        for (int w = 0; w < W; ++w) chi[w] += cq_raw * wq[w];
    }

    // p
    if (D.c1 > 0) {
        const int b1 = D.bcat_dim[1];
        T* cB1 = c_Bcat1_out + (int64_t)i * b1 * 3;
        for (int k = 0; k < b1 * 3; ++k) cB1[k] = T(0);
        for (int c = 0; c < D.c1; ++c) {
            const int64_t off = ((int64_t)i * D.c1 + c) * 3;
            T cp_cap[3], cp_raw[3];
            for (int m = 0; m < 3; ++m) {
                const T cp = c_p[off + m];
                if (do_sw) csw_add += cp * p_cap[off + m];
                cp_cap[m] = sw_m * cp;
            }
            vcap_bwd1<T, 3>(p_raw + off, cp_cap, inv_cap2, cp_raw);
            T cg1 = T(0);
            const T g1v = g1[(int64_t)i * D.c1 + c];
            T cv1[3];
            for (int m = 0; m < 3; ++m) {
                cg1 += cp_raw[m] * v1[off + m];
                cv1[m] = cp_raw[m] * g1v;
            }
            const T* wg = W_g1A + (z * D.c1 + c) * W;
            for (int w = 0; w < W; ++w) chi[w] += cg1 * wg[w];
            const T* wp = W_p1A + c * b1;
            for (int k = 0; k < b1; ++k) {
                const T wk = wp[k] * inv_sb1[k];
                cB1[3 * k + 0] += wk * cv1[0];
                cB1[3 * k + 1] += wk * cv1[1];
                cB1[3 * k + 2] += wk * cv1[2];
            }
        }
    }

    // Q
    if (D.c2 > 0) {
        const int b2 = D.bcat_dim[2];
        T* cB2 = c_Bcat2_out + (int64_t)i * b2 * 5;
        for (int k = 0; k < b2 * 5; ++k) cB2[k] = T(0);
        for (int c = 0; c < D.c2; ++c) {
            const int64_t off = ((int64_t)i * D.c2 + c) * 5;
            T cQ_cap[5], cQ_raw[5];
            for (int m = 0; m < 5; ++m) {
                const T cQ = c_Q[off + m];
                if (do_sw) csw_add += cQ * Q_cap[off + m];
                cQ_cap[m] = sw_m * cQ;
            }
            vcap_bwd1<T, 5>(Q_raw + off, cQ_cap, inv_cap2, cQ_raw);
            T cg2 = T(0);
            const T g2v = g2[(int64_t)i * D.c2 + c];
            T cv2[5];
            for (int m = 0; m < 5; ++m) {
                cg2 += cQ_raw[m] * v2[off + m];
                cv2[m] = cQ_raw[m] * g2v;
            }
            const T* wg = W_g2A + (z * D.c2 + c) * W;
            for (int w = 0; w < W; ++w) chi[w] += cg2 * wg[w];
            const T* wp = W_p2A + c * b2;
            for (int k = 0; k < b2; ++k) {
                const T wk = wp[k] * inv_sb2[k];
                for (int m = 0; m < 5; ++m) cB2[5 * k + m] += wk * cv2[m];
            }
        }
    }

    if (do_sw && c_sw_io) c_sw_io[i] += csw_add;
}

template <typename T>
__global__ void k_nb1_heads_dual_fwd(
    int N, ModelDims D,
    const int* __restrict__ z_idx,
    const T* __restrict__ dot_h,
    const T* __restrict__ dot_Bcat1,
    const T* __restrict__ dot_Bcat2,
    const T* __restrict__ inv_sb1, const T* __restrict__ inv_sb2,
    const T* __restrict__ sw, const T* __restrict__ dd_c_sw,
    const T* __restrict__ q_raw, const T* __restrict__ q_cap,
    const T* __restrict__ v1, const T* __restrict__ g1, const T* __restrict__ p_raw, const T* __restrict__ p_cap,
    const T* __restrict__ v2, const T* __restrict__ g2, const T* __restrict__ Q_raw, const T* __restrict__ Q_cap,
    const T* __restrict__ W_qA,
    const T* __restrict__ W_p1A, const T* __restrict__ W_g1A,
    const T* __restrict__ W_p2A, const T* __restrict__ W_g2A,
    T* __restrict__ dot_q_raw, T* __restrict__ ddq_out,
    T* __restrict__ dot_v1, T* __restrict__ dot_g1, T* __restrict__ dot_p_raw, T* __restrict__ ddp_out,
    T* __restrict__ dot_v2, T* __restrict__ dot_g2, T* __restrict__ dot_Q_raw, T* __restrict__ ddQ_out
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int z = z_idx[i];
    const int W = D.ro_width;
    const T* dhi = dot_h + (int64_t)i * W;
    const T inv_cap2 = (D.mp_cap > 0.0) ? (T)(1.0 / (D.mp_cap * D.mp_cap)) : T(0);
    const bool do_sw = (D.use_zbl_switch && D.sw_mp);
    const T sw_m = do_sw ? sw[i] : T(1);
    const T dsw_c = (do_sw && dd_c_sw) ? dd_c_sw[i] : T(0);

    for (int c = 0; c < D.c0; ++c) {
        const T* wq = W_qA + (z * D.c0 + c) * W;
        T dq = T(0);
        for (int w = 0; w < W; ++w) dq += wq[w] * dhi[w];
        dot_q_raw[(int64_t)i * D.c0 + c] = dq;
        T dummy, dqc;
        vcap_dual_fwd<T, 1>(q_raw + (int64_t)i * D.c0 + c, &dq, inv_cap2, &dummy, &dqc);
        ddq_out[(int64_t)i * D.c0 + c] = sw_m * dqc + dsw_c * q_cap[(int64_t)i * D.c0 + c];
    }

    if (D.c1 > 0) {
        const int b1 = D.bcat_dim[1];
        const T* dB1 = dot_Bcat1 + (int64_t)i * b1 * 3;
        for (int c = 0; c < D.c1; ++c) {
            const int64_t off = ((int64_t)i * D.c1 + c) * 3;
            const T* wg = W_g1A + (z * D.c1 + c) * W;
            T dg1 = T(0);
            for (int w = 0; w < W; ++w) dg1 += wg[w] * dhi[w];
            dot_g1[(int64_t)i * D.c1 + c] = dg1;
            const T* wp = W_p1A + c * b1;
            T dvv[3] = {T(0), T(0), T(0)};
            for (int k = 0; k < b1; ++k) {
                const T wk = wp[k] * inv_sb1[k];
                dvv[0] += wk * dB1[3 * k + 0];
                dvv[1] += wk * dB1[3 * k + 1];
                dvv[2] += wk * dB1[3 * k + 2];
            }
            const T g1v = g1[(int64_t)i * D.c1 + c];
            T dpr[3], dummy[3], dpc[3];
            for (int m = 0; m < 3; ++m) {
                dot_v1[off + m] = dvv[m];
                dpr[m] = dvv[m] * g1v + v1[off + m] * dg1;
                dot_p_raw[off + m] = dpr[m];
            }
            vcap_dual_fwd<T, 3>(p_raw + off, dpr, inv_cap2, dummy, dpc);
            for (int m = 0; m < 3; ++m) {
                ddp_out[off + m] = sw_m * dpc[m] + dsw_c * p_cap[off + m];
            }
        }
    }

    if (D.c2 > 0) {
        const int b2 = D.bcat_dim[2];
        const T* dB2 = dot_Bcat2 + (int64_t)i * b2 * 5;
        for (int c = 0; c < D.c2; ++c) {
            const int64_t off = ((int64_t)i * D.c2 + c) * 5;
            const T* wg = W_g2A + (z * D.c2 + c) * W;
            T dg2 = T(0);
            for (int w = 0; w < W; ++w) dg2 += wg[w] * dhi[w];
            dot_g2[(int64_t)i * D.c2 + c] = dg2;
            const T* wp = W_p2A + c * b2;
            T dvv[5] = {T(0), T(0), T(0), T(0), T(0)};
            for (int k = 0; k < b2; ++k) {
                const T wk = wp[k] * inv_sb2[k];
                for (int m = 0; m < 5; ++m) dvv[m] += wk * dB2[5 * k + m];
            }
            const T g2v = g2[(int64_t)i * D.c2 + c];
            T dQr[5], dummy[5], dQc[5];
            for (int m = 0; m < 5; ++m) {
                dot_v2[off + m] = dvv[m];
                dQr[m] = dvv[m] * g2v + v2[off + m] * dg2;
                dot_Q_raw[off + m] = dQr[m];
            }
            vcap_dual_fwd<T, 5>(Q_raw + off, dQr, inv_cap2, dummy, dQc);
            for (int m = 0; m < 5; ++m) {
                ddQ_out[off + m] = sw_m * dQc[m] + dsw_c * Q_cap[off + m];
            }
        }
    }
}

// Zero-atomic NodeBlock1 Heads Dual Backward: writes per-atom scalars (bq, hq, bg1, hg1, bv1, hv1, bg2, hg2, bv2, hv2),
// then dedicated 1-CTA-per-param-row reduction kernels compute exact g_u_Z, g_W_qA, g_b_qA, g_W_p1A, g_W_g1A, g_b_g1A, etc.!
template <typename T>
__global__ void k_nb1_heads_dual_bwd_noatom(
    int N, ModelDims D,
    const int* __restrict__ z_idx,
    const T* __restrict__ inv_sb1, const T* __restrict__ inv_sb2,
    const T* __restrict__ sw, const T* __restrict__ dd_c_sw,
    const T* __restrict__ bar_esite, const T* __restrict__ hat_esite,
    const T* __restrict__ bar_h_pol, const T* __restrict__ hat_h_pol,
    const T* __restrict__ bar_q, const T* __restrict__ hat_q,
    const T* __restrict__ bar_p, const T* __restrict__ hat_p,
    const T* __restrict__ bar_Q, const T* __restrict__ hat_Q,
    const T* __restrict__ q_raw, const T* __restrict__ dot_q_raw,
    const T* __restrict__ v1, const T* __restrict__ dot_v1, const T* __restrict__ g1, const T* __restrict__ dot_g1,
    const T* __restrict__ p_raw, const T* __restrict__ dot_p_raw,
    const T* __restrict__ v2, const T* __restrict__ dot_v2, const T* __restrict__ g2, const T* __restrict__ dot_g2,
    const T* __restrict__ Q_raw, const T* __restrict__ dot_Q_raw,
    const T* __restrict__ u_Z,
    const T* __restrict__ W_qA,
    const T* __restrict__ W_p1A, const T* __restrict__ W_g1A,
    const T* __restrict__ W_p2A, const T* __restrict__ W_g2A,
    // outputs
    T* __restrict__ bar_h_out, T* __restrict__ hat_h_out,
    T* __restrict__ bar_Bcat1, T* __restrict__ hat_Bcat1,
    T* __restrict__ bar_Bcat2, T* __restrict__ hat_Bcat2,
    // per-atom scratch for zero-atomic param reductions
    T* __restrict__ out_bq_raw, T* __restrict__ out_hq_raw,
    T* __restrict__ out_bg1, T* __restrict__ out_hg1, T* __restrict__ out_bv1, T* __restrict__ out_hv1,
    T* __restrict__ out_bg2, T* __restrict__ out_hg2, T* __restrict__ out_bv2, T* __restrict__ out_hv2
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int z = z_idx[i];
    const int W = D.ro_width;
    const T es = (T)D.energy_scale;
    const T inv_cap2 = (D.mp_cap > 0.0) ? (T)(1.0 / (D.mp_cap * D.mp_cap)) : T(0);
    const bool do_sw = (D.use_zbl_switch && D.sw_mp);
    const T sw_m = do_sw ? sw[i] : T(1);
    const T dsw_c = (do_sw && dd_c_sw) ? dd_c_sw[i] : T(0);

    T* bhi = bar_h_out + (int64_t)i * W;
    T* hhi = hat_h_out + (int64_t)i * W;
    const T* uz = u_Z + z * W;
    const T be = bar_esite[i] * es;
    const T he = hat_esite[i] * es;
    const T* bhp = bar_h_pol ? (bar_h_pol + (int64_t)i * W) : nullptr;
    const T* hhp = hat_h_pol ? (hat_h_pol + (int64_t)i * W) : nullptr;
    for (int w = 0; w < W; ++w) {
        bhi[w] = be * uz[w] + (bhp ? bhp[w] : T(0));
        hhi[w] = he * uz[w] + (hhp ? hhp[w] : T(0));
    }

    for (int c = 0; c < D.c0; ++c) {
        const int64_t idx = (int64_t)i * D.c0 + c;
        const T bq_cap = sw_m * bar_q[idx] + dsw_c * hat_q[idx];
        const T hq_cap = sw_m * hat_q[idx];
        T bq_raw, hq_raw;
        vcap_dual_bwd<T, 1>(q_raw + idx, dot_q_raw + idx, &bq_cap, &hq_cap, inv_cap2, &bq_raw, &hq_raw);
        out_bq_raw[idx] = bq_raw;
        out_hq_raw[idx] = hq_raw;
        const T* wq = W_qA + (z * D.c0 + c) * W;
        for (int w = 0; w < W; ++w) { bhi[w] += bq_raw * wq[w]; hhi[w] += hq_raw * wq[w]; }
    }

    if (D.c1 > 0) {
        const int b1 = D.bcat_dim[1];
        T* bB1 = bar_Bcat1 + (int64_t)i * b1 * 3;
        T* hB1 = hat_Bcat1 + (int64_t)i * b1 * 3;
        for (int k = 0; k < b1 * 3; ++k) { bB1[k] = T(0); hB1[k] = T(0); }
        for (int c = 0; c < D.c1; ++c) {
            const int64_t idx = (int64_t)i * D.c1 + c;
            const int64_t off = idx * 3;
            T bp_cap[3], hp_cap[3], bp_raw[3], hp_raw[3];
            for (int m = 0; m < 3; ++m) {
                bp_cap[m] = sw_m * bar_p[off + m] + dsw_c * hat_p[off + m];
                hp_cap[m] = sw_m * hat_p[off + m];
            }
            vcap_dual_bwd<T, 3>(p_raw + off, dot_p_raw + off, bp_cap, hp_cap, inv_cap2, bp_raw, hp_raw);
            const T g1v = g1[idx], dg1v = dot_g1[idx];
            T bg1 = T(0), hg1 = T(0), bv1[3], hv1[3];
            for (int m = 0; m < 3; ++m) {
                bg1 += bp_raw[m] * v1[off + m] + hp_raw[m] * dot_v1[off + m];
                hg1 += hp_raw[m] * v1[off + m];
                bv1[m] = bp_raw[m] * g1v + hp_raw[m] * dg1v;
                hv1[m] = hp_raw[m] * g1v;
                out_bv1[off + m] = bv1[m];
                out_hv1[off + m] = hv1[m];
            }
            out_bg1[idx] = bg1; out_hg1[idx] = hg1;
            const T* wg = W_g1A + (z * D.c1 + c) * W;
            for (int w = 0; w < W; ++w) { bhi[w] += bg1 * wg[w]; hhi[w] += hg1 * wg[w]; }
            const T* wp = W_p1A + c * b1;
            for (int k = 0; k < b1; ++k) {
                const T wk = wp[k] * inv_sb1[k];
                for (int m = 0; m < 3; ++m) { bB1[3 * k + m] += wk * bv1[m]; hB1[3 * k + m] += wk * hv1[m]; }
            }
        }
    }

    if (D.c2 > 0) {
        const int b2 = D.bcat_dim[2];
        T* bB2 = bar_Bcat2 + (int64_t)i * b2 * 5;
        T* hB2 = hat_Bcat2 + (int64_t)i * b2 * 5;
        for (int k = 0; k < b2 * 5; ++k) { bB2[k] = T(0); hB2[k] = T(0); }
        for (int c = 0; c < D.c2; ++c) {
            const int64_t idx = (int64_t)i * D.c2 + c;
            const int64_t off = idx * 5;
            T bQ_cap[5], hQ_cap[5], bQ_raw[5], hQ_raw[5];
            for (int m = 0; m < 5; ++m) {
                bQ_cap[m] = sw_m * bar_Q[off + m] + dsw_c * hat_Q[off + m];
                hQ_cap[m] = sw_m * hat_Q[off + m];
            }
            vcap_dual_bwd<T, 5>(Q_raw + off, dot_Q_raw + off, bQ_cap, hQ_cap, inv_cap2, bQ_raw, hQ_raw);
            const T g2v = g2[idx], dg2v = dot_g2[idx];
            T bg2 = T(0), hg2 = T(0), bv2[5], hv2[5];
            for (int m = 0; m < 5; ++m) {
                bg2 += bQ_raw[m] * v2[off + m] + hQ_raw[m] * dot_v2[off + m];
                hg2 += hQ_raw[m] * v2[off + m];
                bv2[m] = bQ_raw[m] * g2v + hQ_raw[m] * dg2v;
                hv2[m] = hQ_raw[m] * g2v;
                out_bv2[off + m] = bv2[m];
                out_hv2[off + m] = hv2[m];
            }
            out_bg2[idx] = bg2; out_hg2[idx] = hg2;
            const T* wg = W_g2A + (z * D.c2 + c) * W;
            for (int w = 0; w < W; ++w) { bhi[w] += bg2 * wg[w]; hhi[w] += hg2 * wg[w]; }
            const T* wp = W_p2A + c * b2;
            for (int k = 0; k < b2; ++k) {
                const T wk = wp[k] * inv_sb2[k];
                for (int m = 0; m < 5; ++m) { bB2[5 * k + m] += wk * bv2[m]; hB2[5 * k + m] += wk * hv2[m]; }
            }
        }
    }
}

// v3 Coalesced 2D Warp-Tile species-grouped weight+bias reduction:
// Grid = (S*C, ceil(W/32), N_SPLITS), Block = (32, 8) (256 thr): contiguous w across each warp!
template <typename T>
__global__ void __launch_bounds__(256) k_reduce_sp_head_W(
    int N, int S, int C, int W, T scale,
    const int* __restrict__ z_idx,
    const T* __restrict__ b_arr, const T* __restrict__ hc_arr, // [N, C]
    const T* __restrict__ h, const T* __restrict__ dot_h,      // [N, W]
    T* __restrict__ g_W, T* __restrict__ g_b                   // g_b optional [S, C]
) {
    __shared__ T sm_w[8][32];
    __shared__ T sm_b[8];
    const int sc = blockIdx.x;
    const int c = sc % C;
    const int s = sc / C;
    const int lane = threadIdx.x;
    const int wid = threadIdx.y;
    const int w = blockIdx.y * 32 + lane;
    const bool do_b = (g_b != nullptr && blockIdx.y == 0 && lane == 0);
    const int nsp = gridDim.z;
    const int sp = blockIdx.z;
    const int i_start = (N * sp) / nsp;
    const int i_end = (N * (sp + 1)) / nsp;

    T aw = T(0), ab = T(0);
    for (int i = i_start + wid; i < i_end; i += 8) {
        if (z_idx[i] != s) continue;
        const T bv = b_arr[(int64_t)i * C + c];
        const T hv = hc_arr[(int64_t)i * C + c];
        if (w < W) {
            const int64_t iw = (int64_t)i * W + w;
            aw += bv * h[iw] + hv * dot_h[iw];
        }
        if (do_b) ab += bv;
    }
    sm_w[wid][lane] = aw;
    if (do_b) sm_b[wid] = ab;
    __syncthreads();
    if (wid == 0) {
        if (w < W) {
            T tot = sm_w[0][lane] + sm_w[1][lane] + sm_w[2][lane] + sm_w[3][lane]
                  + sm_w[4][lane] + sm_w[5][lane] + sm_w[6][lane] + sm_w[7][lane];
            if (nsp == 1) g_W[sc * W + w] += scale * tot;
            else          atomicAdd(&g_W[sc * W + w], scale * tot);
        }
        if (do_b) {
            T totb = sm_b[0] + sm_b[1] + sm_b[2] + sm_b[3]
                   + sm_b[4] + sm_b[5] + sm_b[6] + sm_b[7];
            if (nsp == 1) g_b[sc] += scale * totb;
            else          atomicAdd(&g_b[sc], scale * totb);
        }
    }
}

// v3 Coalesced SMEM-staged reduction for W_p1A (DM=3) and W_p2A (DM=5):
// Grid = (ceil(Bdim/32), C, N_SPLITS), Block = (32, 8) (256 thr): contiguous 32*DM warp loads!
template <typename T, int DM>
__global__ void __launch_bounds__(256) k_reduce_Wp_grad(
    int N, int C, int Bdim,
    const T* __restrict__ inv_sb,
    const T* __restrict__ bv, const T* __restrict__ hv,       // [N, C, DM]
    const T* __restrict__ Bcat, const T* __restrict__ dot_Bcat, // [N, Bdim, DM]
    T* __restrict__ g_Wp                                       // [C, Bdim]
) {
    __shared__ T sb[8][32 * DM];
    __shared__ T sdb[8][32 * DM];
    __shared__ T sm[8][32];
    const int k0 = blockIdx.x * 32;
    const int c = blockIdx.y;
    const int sp = blockIdx.z;
    const int nsp = gridDim.z;
    const int lane = threadIdx.x;
    const int wid = threadIdx.y;
    const int k = k0 + lane;
    const int nk_loc = (Bdim - k0 < 32) ? (Bdim - k0) : 32;
    const int n_km = nk_loc * DM;
    const int i_start = (N * sp) / nsp;
    const int i_end = (N * (sp + 1)) / nsp;

    T acc = T(0);
    for (int i = i_start + wid; i < i_end; i += 8) {
        const T* bi_base = Bcat + ((int64_t)i * Bdim + k0) * DM;
        const T* dbi_base = dot_Bcat + ((int64_t)i * Bdim + k0) * DM;
        for (int t = lane; t < n_km; t += 32) {
            sb[wid][t] = bi_base[t];
            sdb[wid][t] = dbi_base[t];
        }
        __syncwarp();
        if (lane < nk_loc) {
            const T* bvi = bv + ((int64_t)i * C + c) * DM;
            const T* hvi = hv + ((int64_t)i * C + c) * DM;
            const T* bkm = sb[wid] + lane * DM;
            const T* dbkm = sdb[wid] + lane * DM;
            #pragma unroll
            for (int m = 0; m < DM; ++m) acc += bvi[m] * bkm[m] + hvi[m] * dbkm[m];
        }
        __syncwarp();
    }
    sm[wid][lane] = acc;
    __syncthreads();
    if (wid == 0 && k < Bdim) {
        T tot = (sm[0][lane] + sm[1][lane] + sm[2][lane] + sm[3][lane]
               + sm[4][lane] + sm[5][lane] + sm[6][lane] + sm[7][lane]) * inv_sb[k];
        if (nsp == 1) g_Wp[c * Bdim + k] += tot;
        else          atomicAdd(&g_Wp[c * Bdim + k], tot);
    }
}

// =============================================================================
// 9. NodeBlock2 (Pol-Field): Pack/Unpack Fwd, Bwd1, Dual-Fwd, Dual-Bwd + Combine E_node
// =============================================================================
template <typename T>
__global__ void k_nb2_pack_fwd(
    int N, ModelDims D,
    const T* __restrict__ Phi_raw, // [N, c0]
    const T* __restrict__ Ef_raw,  // [N, c1, 3]
    const T* __restrict__ p,       // [N, c1, 3]
    const T* __restrict__ h,       // [N, W]
    const T* __restrict__ sw,
    T* __restrict__ Phi_cap, T* __restrict__ Ef_cap,
    T* __restrict__ pol_in         // [N, D_pol]
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const T inv_ef2 = (D.ef_cap > 0.0) ? (T)(1.0 / (D.ef_cap * D.ef_cap)) : T(0);
    const T sw_e = (D.use_zbl_switch && D.sw_mp && D.ef_cap > 0.0) ? sw[i] : T(1);
    T* out = pol_in + (int64_t)i * D.D_pol;
    int col = 0;

    for (int c = 0; c < D.c0; ++c) {
        T pc;
        vcap_fwd<T, 1>(Phi_raw + (int64_t)i * D.c0 + c, inv_ef2, &pc);
        Phi_cap[(int64_t)i * D.c0 + c] = pc;
        out[col++] = sw_e * pc;
    }

    T ef_u[MAX_C][3];
    for (int c = 0; c < D.c1; ++c) {
        const int64_t off = ((int64_t)i * D.c1 + c) * 3;
        T ec[3];
        vcap_fwd<T, 3>(Ef_raw + off, inv_ef2, ec);
        for (int m = 0; m < 3; ++m) {
            Ef_cap[off + m] = ec[m];
            ef_u[c][m] = sw_e * ec[m];
        }
    }

    const T* pi = p + (int64_t)i * D.c1 * 3;
    if (D.pol_gram) {
        for (int a = 0; a < D.c1; ++a) {
            for (int b = a; b < D.c1; ++b) {
                out[col++] = ef_u[a][0] * ef_u[b][0] + ef_u[a][1] * ef_u[b][1] + ef_u[a][2] * ef_u[b][2];
            }
        }
        for (int a = 0; a < D.c1; ++a) {
            for (int b = 0; b < D.c1; ++b) {
                out[col++] = pi[3 * a + 0] * ef_u[b][0] + pi[3 * a + 1] * ef_u[b][1] + pi[3 * a + 2] * ef_u[b][2];
            }
        }
    } else {
        for (int a = 0; a < D.c1; ++a) {
            out[col++] = ef_u[a][0] * ef_u[a][0] + ef_u[a][1] * ef_u[a][1] + ef_u[a][2] * ef_u[a][2];
        }
        for (int a = 0; a < D.c1; ++a) {
            out[col++] = pi[3 * a + 0] * ef_u[a][0] + pi[3 * a + 1] * ef_u[a][1] + pi[3 * a + 2] * ef_u[a][2];
        }
    }
    const T* hi = h + (int64_t)i * D.ro_width;
    for (int w = 0; w < D.ro_width; ++w) out[col++] = hi[w];
}

template <typename T>
__global__ void k_nb2_epol_and_pre_bwd1(
    int N, ModelDims D,
    const int* __restrict__ z_idx,
    const T* __restrict__ pol_pre,  // [N, W_pol]
    const T* __restrict__ pol_h,    // [N, W_pol]
    const T* __restrict__ pol_u,    // [S, W_pol]
    const T* __restrict__ sw,
    T* __restrict__ e_pol_out,      // [N]
    T* __restrict__ c_pol_pre       // [N, W_pol]
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int z = z_idx[i];
    const int Wp = D.pol_width;
    const T es = (T)D.energy_scale;
    const T cep = D.use_zbl_switch ? sw[i] : T(1);
    const T* pu = pol_u + z * Wp;
    const T* pre = pol_pre + (int64_t)i * Wp;
    const T* act = pol_h + (int64_t)i * Wp;
    T* cpre = c_pol_pre + (int64_t)i * Wp;
    T acc = T(0);
    for (int k = 0; k < Wp; ++k) {
        acc += act[k] * pu[k];
        T y, dy, d2y;
        silu_012(pre[k], y, dy, d2y);
        cpre[k] = cep * es * pu[k] * dy;
    }
    e_pol_out[i] = acc * es;
}

template <typename T>
__global__ void k_nb2_unpack_bwd1(
    int N, ModelDims D,
    const T* __restrict__ c_pol_in, // [N, D_pol]
    const T* __restrict__ Phi_raw, const T* __restrict__ Phi_cap,
    const T* __restrict__ Ef_raw,  const T* __restrict__ Ef_cap,
    const T* __restrict__ p,
    const T* __restrict__ sw,
    T* __restrict__ c_Phi_raw,
    T* __restrict__ c_Ef_raw,
    T* __restrict__ c_p_pol,
    T* __restrict__ c_h_pol,
    T* __restrict__ c_sw_io
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const T inv_ef2 = (D.ef_cap > 0.0) ? (T)(1.0 / (D.ef_cap * D.ef_cap)) : T(0);
    const bool do_swe = (D.use_zbl_switch && D.sw_mp && D.ef_cap > 0.0);
    const T sw_e = do_swe ? sw[i] : T(1);
    const T* cin = c_pol_in + (int64_t)i * D.D_pol;
    int col = 0;
    T csw_add = T(0);

    for (int c = 0; c < D.c0; ++c) {
        const T c_u = cin[col++];
        if (do_swe) csw_add += c_u * Phi_cap[(int64_t)i * D.c0 + c];
        const T c_cap = sw_e * c_u;
        vcap_bwd1<T, 1>(Phi_raw + (int64_t)i * D.c0 + c, &c_cap, inv_ef2, c_Phi_raw + (int64_t)i * D.c0 + c);
    }

    T ef_u[MAX_C][3], cef_u[MAX_C][3], cp_l[MAX_C][3];
    for (int c = 0; c < D.c1; ++c) {
        const int64_t off = ((int64_t)i * D.c1 + c) * 3;
        for (int m = 0; m < 3; ++m) {
            ef_u[c][m] = sw_e * Ef_cap[off + m];
            cef_u[c][m] = T(0);
            cp_l[c][m] = T(0);
        }
    }

    const T* pi = p + (int64_t)i * D.c1 * 3;
    if (D.pol_gram) {
        for (int a = 0; a < D.c1; ++a) {
            for (int b = a; b < D.c1; ++b) {
                const T g = cin[col++];
                for (int m = 0; m < 3; ++m) {
                    cef_u[a][m] += g * ef_u[b][m];
                    cef_u[b][m] += g * ef_u[a][m];
                }
            }
        }
        for (int a = 0; a < D.c1; ++a) {
            for (int b = 0; b < D.c1; ++b) {
                const T g = cin[col++];
                for (int m = 0; m < 3; ++m) {
                    cp_l[a][m] += g * ef_u[b][m];
                    cef_u[b][m] += g * pi[3 * a + m];
                }
            }
        }
    } else {
        for (int a = 0; a < D.c1; ++a) {
            const T g = cin[col++];
            for (int m = 0; m < 3; ++m) cef_u[a][m] += T(2) * g * ef_u[a][m];
        }
        for (int a = 0; a < D.c1; ++a) {
            const T g = cin[col++];
            for (int m = 0; m < 3; ++m) {
                cp_l[a][m] += g * ef_u[a][m];
                cef_u[a][m] += g * pi[3 * a + m];
            }
        }
    }

    for (int c = 0; c < D.c1; ++c) {
        const int64_t off = ((int64_t)i * D.c1 + c) * 3;
        T c_cap[3];
        for (int m = 0; m < 3; ++m) {
            if (do_swe) csw_add += cef_u[c][m] * Ef_cap[off + m];
            c_cap[m] = sw_e * cef_u[c][m];
            c_p_pol[off + m] = cp_l[c][m];
        }
        vcap_bwd1<T, 3>(Ef_raw + off, c_cap, inv_ef2, c_Ef_raw + off);
    }

    T* chp = c_h_pol + (int64_t)i * D.ro_width;
    for (int w = 0; w < D.ro_width; ++w) chp[w] = cin[col++];
    if (do_swe && c_sw_io) c_sw_io[i] += csw_add;
}

template <typename T>
__global__ void k_nb2_pack_dual_fwd(
    int N, ModelDims D,
    const T* __restrict__ Phi_raw, const T* __restrict__ Phi_cap, const T* __restrict__ dot_Phi_raw,
    const T* __restrict__ Ef_raw,  const T* __restrict__ Ef_cap,  const T* __restrict__ dot_Ef_raw,
    const T* __restrict__ p, const T* __restrict__ ddp,
    const T* __restrict__ dot_h,
    const T* __restrict__ sw, const T* __restrict__ dd_c_sw,
    T* __restrict__ dot_pol_in
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const T inv_ef2 = (D.ef_cap > 0.0) ? (T)(1.0 / (D.ef_cap * D.ef_cap)) : T(0);
    const bool do_swe = (D.use_zbl_switch && D.sw_mp && D.ef_cap > 0.0);
    const T sw_e = do_swe ? sw[i] : T(1);
    const T dsw_c = (do_swe && dd_c_sw) ? dd_c_sw[i] : T(0);
    T* out = dot_pol_in + (int64_t)i * D.D_pol;
    int col = 0;

    for (int c = 0; c < D.c0; ++c) {
        T dummy, dpc;
        vcap_dual_fwd<T, 1>(Phi_raw + (int64_t)i * D.c0 + c, dot_Phi_raw + (int64_t)i * D.c0 + c, inv_ef2, &dummy, &dpc);
        out[col++] = sw_e * dpc + dsw_c * Phi_cap[(int64_t)i * D.c0 + c];
    }

    T ef_u[MAX_C][3], def_u[MAX_C][3];
    for (int c = 0; c < D.c1; ++c) {
        const int64_t off = ((int64_t)i * D.c1 + c) * 3;
        T dummy[3], dec[3];
        vcap_dual_fwd<T, 3>(Ef_raw + off, dot_Ef_raw + off, inv_ef2, dummy, dec);
        for (int m = 0; m < 3; ++m) {
            ef_u[c][m] = sw_e * Ef_cap[off + m];
            def_u[c][m] = sw_e * dec[m] + dsw_c * Ef_cap[off + m];
        }
    }

    const T* pi = p + (int64_t)i * D.c1 * 3;
    const T* dpi = ddp + (int64_t)i * D.c1 * 3;
    if (D.pol_gram) {
        for (int a = 0; a < D.c1; ++a) {
            for (int b = a; b < D.c1; ++b) {
                out[col++] = (def_u[a][0] * ef_u[b][0] + ef_u[a][0] * def_u[b][0])
                           + (def_u[a][1] * ef_u[b][1] + ef_u[a][1] * def_u[b][1])
                           + (def_u[a][2] * ef_u[b][2] + ef_u[a][2] * def_u[b][2]);
            }
        }
        for (int a = 0; a < D.c1; ++a) {
            for (int b = 0; b < D.c1; ++b) {
                out[col++] = (dpi[3 * a + 0] * ef_u[b][0] + pi[3 * a + 0] * def_u[b][0])
                           + (dpi[3 * a + 1] * ef_u[b][1] + pi[3 * a + 1] * def_u[b][1])
                           + (dpi[3 * a + 2] * ef_u[b][2] + pi[3 * a + 2] * def_u[b][2]);
            }
        }
    } else {
        for (int a = 0; a < D.c1; ++a) {
            out[col++] = T(2) * (ef_u[a][0] * def_u[a][0] + ef_u[a][1] * def_u[a][1] + ef_u[a][2] * def_u[a][2]);
        }
        for (int a = 0; a < D.c1; ++a) {
            out[col++] = (dpi[3 * a + 0] * ef_u[a][0] + pi[3 * a + 0] * def_u[a][0])
                       + (dpi[3 * a + 1] * ef_u[a][1] + pi[3 * a + 1] * def_u[a][1])
                       + (dpi[3 * a + 2] * ef_u[a][2] + pi[3 * a + 2] * def_u[a][2]);
        }
    }

    const T* dhi = dot_h + (int64_t)i * D.ro_width;
    for (int w = 0; w < D.ro_width; ++w) out[col++] = dhi[w];
}

// Zero-atomic NodeBlock2 Dual Bwd Pre (writes dot_pol_h for zero-atomic g_pol_u reduction!)
template <typename T>
__global__ void k_nb2_dual_bwd_pre(
    int N, ModelDims D,
    const int* __restrict__ z_idx,
    const T* __restrict__ pol_pre, const T* __restrict__ dot_pol_pre,
    const T* __restrict__ pol_u,
    const T* __restrict__ bar_epol, const T* __restrict__ sw,
    T* __restrict__ bar_pol_pre, T* __restrict__ hat_pol_pre,
    T* __restrict__ dot_pol_h_out
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int z = z_idx[i];
    const int Wp = D.pol_width;
    const T es = (T)D.energy_scale;
    const T be = bar_epol[i] * es;
    const T he = (D.use_zbl_switch ? sw[i] : T(1)) * es;
    const T* pu = pol_u + z * Wp;
    const T* pre = pol_pre + (int64_t)i * Wp;
    const T* dpre = dot_pol_pre + (int64_t)i * Wp;
    T* bpre = bar_pol_pre + (int64_t)i * Wp;
    T* hpre = hat_pol_pre + (int64_t)i * Wp;
    T* dact_out = dot_pol_h_out + (int64_t)i * Wp;
    for (int k = 0; k < Wp; ++k) {
        T y, dy, d2y;
        silu_012(pre[k], y, dy, d2y);
        dact_out[k] = dy * dpre[k];
        const T ba = be * pu[k];
        const T ha = he * pu[k];
        hpre[k] = dy * ha;
        bpre[k] = dy * ba + d2y * dpre[k] * ha;
    }
}

template <typename T>
__global__ void k_nb2_unpack_dual_bwd(
    int N, ModelDims D,
    const T* __restrict__ bar_pol_in, const T* __restrict__ hat_pol_in,
    const T* __restrict__ Phi_raw, const T* __restrict__ Phi_cap, const T* __restrict__ dot_Phi_raw,
    const T* __restrict__ Ef_raw,  const T* __restrict__ Ef_cap,  const T* __restrict__ dot_Ef_raw,
    const T* __restrict__ p, const T* __restrict__ ddp,
    const T* __restrict__ sw, const T* __restrict__ dd_c_sw,
    T* __restrict__ bar_Phi_raw,
    T* __restrict__ bar_Ef_raw,
    T* __restrict__ bar_p_pol,
    T* __restrict__ bar_h_pol
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const T inv_ef2 = (D.ef_cap > 0.0) ? (T)(1.0 / (D.ef_cap * D.ef_cap)) : T(0);
    const bool do_swe = (D.use_zbl_switch && D.sw_mp && D.ef_cap > 0.0);
    const T sw_e = do_swe ? sw[i] : T(1);
    const T dsw_c = (do_swe && dd_c_sw) ? dd_c_sw[i] : T(0);
    const T* bin = bar_pol_in + (int64_t)i * D.D_pol;
    const T* hin = hat_pol_in + (int64_t)i * D.D_pol;
    int col = 0;

    for (int c = 0; c < D.c0; ++c) {
        const T bu = bin[col], hu = hin[col]; ++col;
        const T b_cap = sw_e * bu + dsw_c * hu;
        const T h_cap = sw_e * hu;
        T b_raw, h_raw;
        vcap_dual_bwd<T, 1>(Phi_raw + (int64_t)i * D.c0 + c, dot_Phi_raw + (int64_t)i * D.c0 + c, &b_cap, &h_cap, inv_ef2, &b_raw, &h_raw);
        bar_Phi_raw[(int64_t)i * D.c0 + c] = b_raw;
    }

    T ef_u[MAX_C][3], def_u[MAX_C][3], bef_u[MAX_C][3], hef_u[MAX_C][3], bp_l[MAX_C][3];
    for (int c = 0; c < D.c1; ++c) {
        const int64_t off = ((int64_t)i * D.c1 + c) * 3;
        T dummy[3], dec[3];
        vcap_dual_fwd<T, 3>(Ef_raw + off, dot_Ef_raw + off, inv_ef2, dummy, dec);
        for (int m = 0; m < 3; ++m) {
            ef_u[c][m] = sw_e * Ef_cap[off + m];
            def_u[c][m] = sw_e * dec[m] + dsw_c * Ef_cap[off + m];
            bef_u[c][m] = T(0); hef_u[c][m] = T(0); bp_l[c][m] = T(0);
        }
    }

    const T* pi = p + (int64_t)i * D.c1 * 3;
    const T* dpi = ddp + (int64_t)i * D.c1 * 3;
    if (D.pol_gram) {
        for (int a = 0; a < D.c1; ++a) {
            for (int b = a; b < D.c1; ++b) {
                const T bg = bin[col], hg = hin[col]; ++col;
                for (int m = 0; m < 3; ++m) {
                    bef_u[a][m] += bg * ef_u[b][m] + hg * def_u[b][m];
                    hef_u[a][m] += hg * ef_u[b][m];
                    bef_u[b][m] += bg * ef_u[a][m] + hg * def_u[a][m];
                    hef_u[b][m] += hg * ef_u[a][m];
                }
            }
        }
        for (int a = 0; a < D.c1; ++a) {
            for (int b = 0; b < D.c1; ++b) {
                const T bg = bin[col], hg = hin[col]; ++col;
                for (int m = 0; m < 3; ++m) {
                    bp_l[a][m]  += bg * ef_u[b][m] + hg * def_u[b][m];
                    bef_u[b][m] += bg * pi[3 * a + m] + hg * dpi[3 * a + m];
                    hef_u[b][m] += hg * pi[3 * a + m];
                }
            }
        }
    } else {
        for (int a = 0; a < D.c1; ++a) {
            const T bg = bin[col], hg = hin[col]; ++col;
            for (int m = 0; m < 3; ++m) {
                bef_u[a][m] += T(2) * (bg * ef_u[a][m] + hg * def_u[a][m]);
                hef_u[a][m] += T(2) * hg * ef_u[a][m];
            }
        }
        for (int a = 0; a < D.c1; ++a) {
            const T bg = bin[col], hg = hin[col]; ++col;
            for (int m = 0; m < 3; ++m) {
                bp_l[a][m]  += bg * ef_u[a][m] + hg * def_u[a][m];
                bef_u[a][m] += bg * pi[3 * a + m] + hg * dpi[3 * a + m];
                hef_u[a][m] += hg * pi[3 * a + m];
            }
        }
    }

    for (int c = 0; c < D.c1; ++c) {
        const int64_t off = ((int64_t)i * D.c1 + c) * 3;
        T b_cap[3], h_cap[3], b_raw[3], h_raw[3];
        for (int m = 0; m < 3; ++m) {
            b_cap[m] = sw_e * bef_u[c][m] + dsw_c * hef_u[c][m];
            h_cap[m] = sw_e * hef_u[c][m];
            bar_p_pol[off + m] = bp_l[c][m];
        }
        vcap_dual_bwd<T, 3>(Ef_raw + off, dot_Ef_raw + off, b_cap, h_cap, inv_ef2, b_raw, h_raw);
        for (int m = 0; m < 3; ++m) bar_Ef_raw[off + m] = b_raw[m];
    }

    T* bhp = bar_h_pol + (int64_t)i * D.ro_width;
    for (int w = 0; w < D.ro_width; ++w) bhp[w] = bin[col++];
}

// =============================================================================
// 10. Combine Node Energies, ZBL Force/Virial Fwd & Bwd2, and General Loss Kernel
// =============================================================================
template <typename T>
__global__ void k_combine_enode(
    int N, int use_zbl, int has_pol,
    const T* __restrict__ e_site, const T* __restrict__ e_lr, const T* __restrict__ e_pol,
    const T* __restrict__ sw, const T* __restrict__ e_zbl,
    const int* __restrict__ atom_batch,
    T* __restrict__ e_ml_out, T* __restrict__ e_node_out, T* __restrict__ c_sw_out,
    T* __restrict__ E_graph_out // zero-initialized [G]
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const T eml = e_site[i] + e_lr[i] + (has_pol ? e_pol[i] : T(0));
    const T en = use_zbl ? (sw[i] * eml + e_zbl[i]) : eml;
    e_ml_out[i] = eml;
    e_node_out[i] = en;
    if (c_sw_out) c_sw_out[i] = eml;
    atomicAdd(E_graph_out + atom_batch[i], en);
}

template <typename T>
__global__ void k_zbl_force_fwd(
    int Ez, int compute_virial,
    const int* __restrict__ z_snd, const int* __restrict__ z_rcv, const int* __restrict__ z_eb,
    const T* __restrict__ z_gv, const T* __restrict__ z_vec,
    const T* __restrict__ dsw, const T* __restrict__ c_sw,
    T* __restrict__ F_tot, T* __restrict__ V_tot
) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= Ez) return;
    const int j = z_snd[e], i = z_rcv[e];
    const T ce = T(1) + dsw[i] * c_sw[i];
    const T gx = ce * z_gv[3 * e + 0];
    const T gy = ce * z_gv[3 * e + 1];
    const T gz = ce * z_gv[3 * e + 2];
    atomicAdd(F_tot + 3 * j + 0, gx); atomicAdd(F_tot + 3 * j + 1, gy); atomicAdd(F_tot + 3 * j + 2, gz);
    atomicAdd(F_tot + 3 * i + 0, -gx); atomicAdd(F_tot + 3 * i + 1, -gy); atomicAdd(F_tot + 3 * i + 2, -gz);
    if (compute_virial && V_tot) {
        const T rx = z_vec[3 * e + 0], ry = z_vec[3 * e + 1], rz = z_vec[3 * e + 2];
        T* vg = V_tot + (int64_t)z_eb[e] * 9;
        const T xy = T(0.5) * (rx * gy + ry * gx);
        const T xz = T(0.5) * (rx * gz + rz * gx);
        const T yz = T(0.5) * (ry * gz + rz * gy);
        atomicAdd(vg + 0, rx * gx); atomicAdd(vg + 1, xy); atomicAdd(vg + 2, xz);
        atomicAdd(vg + 3, xy);      atomicAdd(vg + 4, ry * gy); atomicAdd(vg + 5, yz);
        atomicAdd(vg + 6, xz);      atomicAdd(vg + 7, yz);      atomicAdd(vg + 8, rz * gz);
    }
}

template <typename T>
__global__ void k_zbl_bwd2(
    int Ez, int has_dV,
    const int* __restrict__ z_snd, const int* __restrict__ z_rcv, const int* __restrict__ z_eb,
    const T* __restrict__ z_gv, const T* __restrict__ z_vec,
    const T* __restrict__ dsw,
    const T* __restrict__ dF, const T* __restrict__ dV,
    T* __restrict__ dd_c_sw // zero-initialized [N]
) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= Ez) return;
    const int j = z_snd[e], i = z_rcv[e];
    T wx = dF[3 * j + 0] - dF[3 * i + 0];
    T wy = dF[3 * j + 1] - dF[3 * i + 1];
    T wz = dF[3 * j + 2] - dF[3 * i + 2];
    if (has_dV && dV) {
        const T* dv = dV + (int64_t)z_eb[e] * 9;
        const T rx = z_vec[3 * e + 0], ry = z_vec[3 * e + 1], rz = z_vec[3 * e + 2];
        const T m01 = T(0.5) * (dv[1] + dv[3]);
        const T m02 = T(0.5) * (dv[2] + dv[6]);
        const T m12 = T(0.5) * (dv[5] + dv[7]);
        wx += dv[0] * rx + m01 * ry + m02 * rz;
        wy += m01 * rx + dv[4] * ry + m12 * rz;
        wz += m02 * rx + m12 * ry + dv[8] * rz;
    }
    const T dot = wx * z_gv[3 * e + 0] + wy * z_gv[3 * e + 1] + wz * z_gv[3 * e + 2];
    atomicAdd(dd_c_sw + i, dsw[i] * dot);
}

template <typename T>
__global__ void k_add_vec(long n, const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

template <typename T>
__global__ void k_add3_vec(long n, const T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ c, T* __restrict__ out) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i] + c[i];
}

template <typename T>
__global__ void k_bar_eml_from_enode(
    int N, int use_zbl,
    const T* __restrict__ sw, const T* __restrict__ bar_enode, const T* __restrict__ dd_c_sw,
    T* __restrict__ bar_eml
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    bar_eml[i] = (use_zbl ? (sw[i] * bar_enode[i]) : bar_enode[i]) + (dd_c_sw ? dd_c_sw[i] : T(0));
}

// Per-graph normalized force loss + GPU frame_err EMA update (exact match to train_n3tbu_lrsk_v4_fast.py!)
template <typename T>
__global__ void k_loss_ef_pergraph_n3(
    BatchDesc bd, double we, double wf,
    const int* __restrict__ atom_batch,
    const T* __restrict__ na_f,
    const T* __restrict__ E_pred, const T* __restrict__ E_ref,
    const T* __restrict__ F_pred, const T* __restrict__ F_ref,
    T* __restrict__ bar_enode,
    T* __restrict__ dF_out,
    double* __restrict__ f_sq_g_tmp, // zero-initialized [B]
    double* __restrict__ loss_acc,
    double* __restrict__ lf_acc      // unweighted weighted-mean force MSE accumulator for trF logging
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bd.N) {
        const int g = atom_batch[i];
        const T w_g = (T)bd.dyn_gw[g];
        const T inv_na = T(1) / na_f[g];
        const T de = (E_pred[g] - E_ref[g]) * inv_na;
        bar_enode[i] = (T)(2.0 * we / (double)bd.B) * w_g * de * inv_na;

        const T f_coef = (T)(wf / (double)bd.B) * w_g * (inv_na / T(3));
        const T g_coef = T(2) * f_coef;
        double fsq_i = 0.0;
        for (int a = 0; a < 3; ++a) {
            const T df = F_pred[3 * i + a] - F_ref[3 * i + a];
            dF_out[3 * i + a] = g_coef * df;
            fsq_i += (double)(df * df);
        }
        atomicAdd(f_sq_g_tmp + g, fsq_i / (3.0 * (double)na_f[g]));
    }
}

template <typename T>
__global__ void k_update_frame_err_n3(
    BatchDesc bd, double we, double wf,
    const double* __restrict__ f_sq_g_tmp,
    const T* __restrict__ E_pred, const T* __restrict__ E_ref, const T* __restrict__ na_f,
    float* __restrict__ frame_err, int* __restrict__ frame_seen,
    double* __restrict__ loss_acc, double* __restrict__ lf_acc
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double l_sum = 0.0, lf_sum = 0.0;
    const double inv_B = 1.0 / (double)bd.B;
    for (int k = 0; k < bd.B; ++k) {
        const int gid = bd.g_id[k];
        const double wg = bd.dyn_gw[k];
        const double fsq = f_sq_g_tmp[k];
        if (frame_err && frame_seen) {
            const float v = (float)fsq;
            frame_err[gid] = frame_seen[gid] ? (0.5f * frame_err[gid] + 0.5f * v) : v;
            frame_seen[gid] = 1;
        }
        const double de = ((double)E_pred[k] - (double)E_ref[k]) / (double)na_f[k];
        l_sum += inv_B * (we * wg * de * de + wf * wg * fsq);
        lf_sum += inv_B * wg * fsq;
    }
    *loss_acc += l_sum;
    if (lf_acc) *lf_acc += lf_sum;
}

// =============================================================================
// 11. General Loss & Adjoint Kernels (E + F + S + Ei + Mol COM/Torque + Graph/Species Weights)
// =============================================================================
template <typename T>
__global__ void k_loss_ef_s(
    int N, int G, int has_stress, int has_spw,
    double we, double wf, double ws,
    const int* __restrict__ atom_batch, const int* __restrict__ z_idx,
    const T* __restrict__ gw, const T* __restrict__ sp_w,
    const T* __restrict__ na_f, const T* __restrict__ vol,
    const T* __restrict__ E_pred, const T* __restrict__ E_ref,
    const T* __restrict__ F_pred, const T* __restrict__ F_ref,
    const T* __restrict__ V_pred, const T* __restrict__ S_ref,
    T* __restrict__ bar_enode,
    T* __restrict__ dF_out,
    T* __restrict__ dV_out,
    double* __restrict__ loss_acc // scalar GPU loss accumulator
) {
    // 1. Per-atom energy + force terms
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const int g = atom_batch[i];
        const T w_g = gw ? gw[g] : T(1);
        const T inv_na = T(1) / na_f[g];
        const T de = (E_pred[g] - E_ref[g]) * inv_na;
        bar_enode[i] = (T)(2.0 * we / (double)G) * w_g * de * inv_na;

        const T aw = (has_spw && sp_w) ? sp_w[z_idx[i]] : T(1);
        const T f_coef = (T)(wf / (double)(3 * N)) * w_g * aw;
        const T g_coef = T(2) * f_coef;
        double lf_i = 0.0;
        for (int a = 0; a < 3; ++a) {
            const T df = F_pred[3 * i + a] - F_ref[3 * i + a];
            dF_out[3 * i + a] = g_coef * df;
            lf_i += (double)(f_coef * df * df);
        }
        atomicAdd(loss_acc, lf_i);
    }
    // 2. Per-graph energy loss & stress terms
    if (i < G) {
        const T w_g = gw ? gw[i] : T(1);
        const T de = (E_pred[i] - E_ref[i]) / na_f[i];
        atomicAdd(loss_acc, (we / (double)G) * (double)(w_g * de * de));
        if (has_stress && V_pred && S_ref && dV_out) {
            const T inv_vol = T(1) / vol[i];
            const T s_coef = (T)(ws / (double)(9 * G)) * w_g;
            const T g_coef = T(2) * s_coef * inv_vol;
            double ls_g = 0.0;
            for (int t = 0; t < 9; ++t) {
                const T ds = V_pred[9 * i + t] * inv_vol - S_ref[9 * i + t];
                dV_out[9 * i + t] = g_coef * ds;
                ls_g += (double)(s_coef * ds * ds);
            }
            atomicAdd(loss_acc, ls_g);
        }
    }
}

// Gauge-centered per-(graph, species) E_i loss (2 tiny passes: accumulate sums/cnts, then compute loss + bar_enode)
template <typename T>
__global__ void k_loss_ei_stats(
    int N, int S, const int* __restrict__ atom_batch, const int* __restrict__ z_idx,
    const T* __restrict__ e_node, T* __restrict__ gs_sum, T* __restrict__ gs_cnt
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int gs = atom_batch[i] * S + z_idx[i];
    atomicAdd(gs_sum + gs, e_node[i]);
    atomicAdd(gs_cnt + gs, T(1));
}

template <typename T>
__global__ void k_loss_ei_grad1(
    int N, int S, double wei, const int* __restrict__ atom_batch, const int* __restrict__ z_idx,
    const T* __restrict__ gw, const T* __restrict__ e_node, const T* __restrict__ ei_ref,
    const T* __restrict__ gs_sum, const T* __restrict__ gs_cnt,
    T* __restrict__ gs_grad_sum, // zero-initialized [G*S]
    T* __restrict__ bar_enode,
    double* __restrict__ loss_acc
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int g = atom_batch[i];
    const int gs = g * S + z_idx[i];
    const T cnt = gs_cnt[gs] > T(1) ? gs_cnt[gs] : T(1);
    const T mean = gs_sum[gs] / cnt;
    const T diff = (e_node[i] - mean) - ei_ref[i];
    const T w_g = gw ? gw[g] : T(1);
    const T coef = (T)(wei / (double)N) * w_g;
    const T g_raw = T(2) * coef * diff;
    bar_enode[i] += g_raw;
    atomicAdd(gs_grad_sum + gs, g_raw / cnt);
    atomicAdd(loss_acc, (double)(coef * diff * diff));
}

template <typename T>
__global__ void k_loss_ei_grad2(
    int N, int S, const int* __restrict__ atom_batch, const int* __restrict__ z_idx,
    const T* __restrict__ gs_grad_sum, T* __restrict__ bar_enode
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    bar_enode[i] -= gs_grad_sum[atom_batch[i] * S + z_idx[i]];
}

// Molecular COM + Torque loss (2 tiny passes: accumulate per-molecule df_net & df_trq, then scatter exact grad to dF!)
template <typename T>
__global__ void k_loss_mol_accum(
    int N, const int* __restrict__ mol_id,
    const T* __restrict__ mol_arms, const T* __restrict__ F_pred, const T* __restrict__ F_ref,
    T* __restrict__ df_net, T* __restrict__ df_trq // zero-initialized [M_mol, 3]
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int m = mol_id[i];
    if (m < 0) return;
    const T dx = F_pred[3 * i + 0] - F_ref[3 * i + 0];
    const T dy = F_pred[3 * i + 1] - F_ref[3 * i + 1];
    const T dz = F_pred[3 * i + 2] - F_ref[3 * i + 2];
    const T ax = mol_arms[3 * i + 0], ay = mol_arms[3 * i + 1], az = mol_arms[3 * i + 2];
    atomicAdd(df_net + 3 * m + 0, dx);
    atomicAdd(df_net + 3 * m + 1, dy);
    atomicAdd(df_net + 3 * m + 2, dz);
    atomicAdd(df_trq + 3 * m + 0, ay * dz - az * dy);
    atomicAdd(df_trq + 3 * m + 1, az * dx - ax * dz);
    atomicAdd(df_trq + 3 * m + 2, ax * dy - ay * dx);
}

template <typename T>
__global__ void k_loss_mol_loss(
    int M_mol, int n_mol_g, double w_mol, double net_s, double trq_s,
    const int* __restrict__ mol_g, const int* __restrict__ n_mol_per_g,
    const T* __restrict__ gw, const T* __restrict__ df_net, const T* __restrict__ df_trq,
    double* __restrict__ loss_acc
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M_mol) return;
    const int g = mol_g[m];
    const T w_g = gw ? gw[g] : T(1);
    const double sc = (w_mol / (double)n_mol_g) / (double)(3 * n_mol_per_g[g]) * (double)w_g;
    double n2 = 0.0, t2 = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double fn = (double)df_net[3 * m + a];
        const double ft = (double)df_trq[3 * m + a];
        n2 += fn * fn; t2 += ft * ft;
    }
    atomicAdd(loss_acc, sc * (net_s * n2 + trq_s * t2));
}

template <typename T>
__global__ void k_loss_mol_grad(
    int N, int n_mol_g, double w_mol, double net_s, double trq_s,
    const int* __restrict__ atom_batch, const int* __restrict__ mol_id,
    const int* __restrict__ n_mol_per_g, const T* __restrict__ gw,
    const T* __restrict__ mol_arms, const T* __restrict__ df_net, const T* __restrict__ df_trq,
    T* __restrict__ dF_out
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int g = atom_batch[i];
    const int m = mol_id[i];
    if (m < 0) return;
    const T w_g = gw ? gw[g] : T(1);
    const T sc = (T)(2.0 * (w_mol / (double)n_mol_g) / (double)(3 * n_mol_per_g[g])) * w_g;
    const T ns = sc * (T)net_s;
    const T ts = sc * (T)trq_s;
    const T nx = df_net[3 * m + 0], ny = df_net[3 * m + 1], nz = df_net[3 * m + 2];
    const T tx = df_trq[3 * m + 0], ty = df_trq[3 * m + 1], tz = df_trq[3 * m + 2];
    const T ax = mol_arms[3 * i + 0], ay = mol_arms[3 * i + 1], az = mol_arms[3 * i + 2];
    // d/df_i <df_trq, arms x f_i> = df_trq x arms
    dF_out[3 * i + 0] += ns * nx + ts * (ty * az - tz * ay);
    dF_out[3 * i + 1] += ns * ny + ts * (tz * ax - tx * az);
    dF_out[3 * i + 2] += ns * nz + ts * (tx * ay - ty * ax);
}

} // namespace lrsk_ft
