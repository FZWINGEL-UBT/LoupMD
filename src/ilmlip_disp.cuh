#pragma once
// Built-in Option B force-shifted fixed-C6 D3(BJ) dispersion kernel + analytical K/V tail
// for arch=ilmlip students. Reuses the student's existing active neighbour list (works in
// periodic, tiled, open-domain, and LAMMPS ghost-atom modes) and runs on a dedicated side
// stream overlapped with layer-1 forward/GEMMs.
#include <cstdint>
#include <cuda_runtime.h>

namespace loup {

__device__ __forceinline__ double disp_warp_sum_d(double v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Per-species-pair packed constants: [S*S, 6] float32 = (c6, c8, f6, f8, e_shift, f_shift)
// in Angstrom / eV units:
//   u(r)   = -c6 / (r^6 + f6) - c8 / (r^8 + f8)
//   V(r)   = u(r) - e_shift - (r - r_cut) * f_shift     (r < r_cut, else 0)
//   V'(r)  = 6*c6*r^5/(r^6+f6)^2 + 8*c8*r^7/(r^8+f8)^2 - f_shift
//
// Receiver-only force accumulation (no per-edge sender atomics) + block-level energy & virial
// reduction (at most 1 atomicAdd to energy and 6 atomicAdds to virial per thread block).
template <bool STRESS, bool PACK_VS, bool ENERGY_ONLY>
__global__ void __launch_bounds__(128) ilmlip_disp_pair_kernel(
    int N, int S, float r_cut,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species, const float* __restrict__ disp_pair,
    double* __restrict__ energy, double* __restrict__ atom_energy,
    double* __restrict__ forces, double* __restrict__ virial) {
    extern __shared__ float sm_pair[];
    const int n_pair_f = S * S * 6;
    const bool use_smem = (S <= 16);
    if (use_smem) {
        for (int idx = threadIdx.x; idx < n_pair_f; idx += blockDim.x)
            sm_pair[idx] = __ldg(disp_pair + idx);
    }
    __shared__ double sm_e[4];
    __shared__ double sm_v[4][6];
    __syncthreads();

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int stride = gridDim.x * 4;

    double blk_e = 0.0;
    double blk_v0 = 0.0, blk_v1 = 0.0, blk_v2 = 0.0, blk_v3 = 0.0, blk_v4 = 0.0, blk_v5 = 0.0;

    for (int i = blockIdx.x * 4 + wid; i < N; i += stride) {
        const int z_i = __ldg(species + i);
        const int eb = __ldg(row_ptr + i);
        const int ee = __ldg(row_end + i);
        double e_acc = 0.0;
        double fx_acc = 0.0, fy_acc = 0.0, fz_acc = 0.0;
        double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

        for (int e = eb + lane; e < ee; e += 32) {
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                const float r2_raw = fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z));
                r = sqrtf(fmaxf(r2_raw, 1e-16f));
            } else {
                j = __ldg(senders + e);
                r = fmaxf(vl.w, 1e-8f);
            }
            if (r < r_cut && r > 1e-7f) {
                const int pbase = (z_i * S + __ldg(species + j)) * 6;
                const float* pp = use_smem ? (sm_pair + pbase) : (disp_pair + pbase);
                const float c6 = pp[0];
                const float c8 = pp[1];
                const float f6 = pp[2];
                const float f8 = pp[3];
                const float es = pp[4];
                const float fs = pp[5];

                const float r2 = r * r;
                const float r4 = r2 * r2;
                const float r6 = r4 * r2;
                const float r8 = r4 * r4;
                const float inv_d6 = 1.0f / (r6 + f6);
                const float inv_d8 = 1.0f / (r8 + f8);

                const float u = -c6 * inv_d6 - c8 * inv_d8;
                const float v_pair = u - es - (r - r_cut) * fs;
                e_acc += 0.5 * (double)v_pair;

                if constexpr (!ENERGY_ONLY) {
                    const float r5 = r4 * r;
                    const float r7 = r6 * r;
                    const float du = (6.0f * c6 * r5) * (inv_d6 * inv_d6) +
                                     (8.0f * c8 * r7) * (inv_d8 * inv_d8);
                    const float dv = du - fs;
                    const float coef = dv / r;
                    fx_acc -= (double)coef * (double)vl.x;
                    fy_acc -= (double)coef * (double)vl.y;
                    fz_acc -= (double)coef * (double)vl.z;

                    if constexpr (STRESS) {
                        const double vc = 0.5 * (double)coef;
                        const double vx = (double)vl.x;
                        const double vy = (double)vl.y;
                        const double vz = (double)vl.z;
                        v0 += vc * vx * vx;
                        v1 += vc * vx * vy;
                        v2 += vc * vx * vz;
                        v3 += vc * vy * vy;
                        v4 += vc * vy * vz;
                        v5 += vc * vz * vz;
                    }
                }
            }
        }

        e_acc = disp_warp_sum_d(e_acc);
        if constexpr (!ENERGY_ONLY) {
            fx_acc = disp_warp_sum_d(fx_acc);
            fy_acc = disp_warp_sum_d(fy_acc);
            fz_acc = disp_warp_sum_d(fz_acc);
            if constexpr (STRESS) {
                v0 = disp_warp_sum_d(v0); v1 = disp_warp_sum_d(v1); v2 = disp_warp_sum_d(v2);
                v3 = disp_warp_sum_d(v3); v4 = disp_warp_sum_d(v4); v5 = disp_warp_sum_d(v5);
            }
        }
        if (lane == 0) {
            blk_e += e_acc;
            if (atom_energy) atom_energy[i] += e_acc;
            if constexpr (!ENERGY_ONLY) {
                if (fx_acc != 0.0 || fy_acc != 0.0 || fz_acc != 0.0) {
                    atomicAdd(forces + (int64_t)i * 3 + 0, fx_acc);
                    atomicAdd(forces + (int64_t)i * 3 + 1, fy_acc);
                    atomicAdd(forces + (int64_t)i * 3 + 2, fz_acc);
                }
                if constexpr (STRESS) {
                    blk_v0 += v0; blk_v1 += v1; blk_v2 += v2;
                    blk_v3 += v3; blk_v4 += v4; blk_v5 += v5;
                }
            }
        }
    }

    if (lane == 0) {
        sm_e[wid] = blk_e;
        if constexpr (!ENERGY_ONLY && STRESS) {
            sm_v[wid][0] = blk_v0; sm_v[wid][1] = blk_v1; sm_v[wid][2] = blk_v2;
            sm_v[wid][3] = blk_v3; sm_v[wid][4] = blk_v4; sm_v[wid][5] = blk_v5;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const double be = sm_e[0] + sm_e[1] + sm_e[2] + sm_e[3];
        if (be != 0.0) atomicAdd(energy, be);
        if constexpr (!ENERGY_ONLY && STRESS) {
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                const double bv = sm_v[0][k] + sm_v[1][k] + sm_v[2][k] + sm_v[3][k];
                if (bv != 0.0) atomicAdd(virial + k, bv);
            }
        }
    }
}

// Analytical uniform-density tail: E_tail = K_tail / V, virial_diag -= E_tail.
// Reads live device volume during variable-cell NPT (cell_vol_dev != nullptr), else cell_vol_host.
template <bool STRESS>
__global__ void ilmlip_disp_tail_kernel(
    int N, double K_tail, const double* __restrict__ cell_vol_dev, double cell_vol_host,
    const int* __restrict__ species, const double* __restrict__ tail_per_sp,
    double* __restrict__ energy, double* __restrict__ atom_energy, double* __restrict__ virial) {
    const double vol = cell_vol_dev ? (*cell_vol_dev) : cell_vol_host;
    if (vol <= 1e-12) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const double e_tail = K_tail / vol;
        atomicAdd(energy, e_tail);
        if constexpr (STRESS) {
            atomicAdd(virial + 0, -e_tail);
            atomicAdd(virial + 3, -e_tail);
            atomicAdd(virial + 5, -e_tail);
        }
    }
    if (atom_energy && tail_per_sp) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) {
            atom_energy[i] += tail_per_sp[__ldg(species + i)] / vol;
        }
    }
}

// =================================================================================================
// Option C: Coordination-dependent C6 D3(BJ) + exact CN chain-rule forces/virial (3 passes over
// the student's active neighbour list).
// =================================================================================================
template <bool PACK_VS>
__global__ void __launch_bounds__(128) ilmlip_disp_cn1_kernel(
    int N, float r_cut,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const int* __restrict__ nref_sp, const float* __restrict__ cnref_sp,
    const float* __restrict__ rcov_sp,
    float* __restrict__ w_out, float* __restrict__ dw_out) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int stride = gridDim.x * 4;
    for (int i = blockIdx.x * 4 + wid; i < N; i += stride) {
        const int z_i = __ldg(species + i);
        const float rcov_i = __ldg(rcov_sp + z_i);
        const int eb = __ldg(row_ptr + i);
        const int ee = __ldg(row_end + i);
        double cn = 0.0;
        for (int e = eb + lane; e < ee; e += 32) {
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                r = sqrtf(fmaxf(fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z)), 1e-16f));
            } else {
                j = __ldg(senders + e);
                r = fmaxf(vl.w, 1e-8f);
            }
            if (r < r_cut && r > 1e-7f) {
                const float rco = rcov_i + __ldg(rcov_sp + __ldg(species + j));
                cn += 1.0 / (1.0 + exp(-16.0 * ((double)rco / (double)r - 1.0)));
            }
        }
        cn = disp_warp_sum_d(cn);
        if (lane == 0) {
            const int nr = __ldg(nref_sp + z_i);
            float* wi = w_out + (int64_t)i * 5;
            float* dwi = dw_out + (int64_t)i * 5;
            double lw[5], dcn[5];
            double mx = -1e300;
            for (int a = 0; a < nr; ++a) {
                dcn[a] = cn - (double)__ldg(cnref_sp + z_i * 5 + a);
                lw[a] = -4.0 * dcn[a] * dcn[a];
                if (lw[a] > mx) mx = lw[a];
            }
            double ea[5], zsum = 0.0;
            for (int a = 0; a < nr; ++a) {
                ea[a] = exp(lw[a] - mx);
                zsum += ea[a];
            }
            double wa[5], gmean = 0.0;
            for (int a = 0; a < nr; ++a) {
                wa[a] = ea[a] / zsum;
                gmean += wa[a] * (-8.0 * dcn[a]);
            }
            for (int a = 0; a < 5; ++a) {
                if (a < nr) {
                    wi[a] = (float)wa[a];
                    dwi[a] = (float)(wa[a] * (-8.0 * dcn[a] - gmean));
                } else {
                    wi[a] = 0.0f;
                    dwi[a] = 0.0f;
                }
            }
        }
    }
}

template <bool STRESS, bool PACK_VS, bool ENERGY_ONLY>
__global__ void __launch_bounds__(128) ilmlip_disp_cn2_kernel(
    int N, int S, float r_cut, float s6, float s8,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const int* __restrict__ nref_sp,
    const float* __restrict__ cn_pair,   // [S*S, 5]: qq_A2, f6, f8, g_cut, dg_cut
    const float* __restrict__ c6ref_ev,  // [S*S, 25]
    const float* __restrict__ w_in, const float* __restrict__ dw_in,
    float* __restrict__ dedcn_out,
    double* __restrict__ energy, double* __restrict__ atom_energy,
    double* __restrict__ forces, double* __restrict__ virial) {
    __shared__ double sm_e[4];
    __shared__ double sm_v[4][6];

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int stride = gridDim.x * 4;

    double blk_e = 0.0;
    double blk_v0 = 0.0, blk_v1 = 0.0, blk_v2 = 0.0, blk_v3 = 0.0, blk_v4 = 0.0, blk_v5 = 0.0;

    for (int i = blockIdx.x * 4 + wid; i < N; i += stride) {
        const int z_i = __ldg(species + i);
        const int nr_i = __ldg(nref_sp + z_i);
        float wi[5], dwi[5];
        #pragma unroll
        for (int a = 0; a < 5; ++a) {
            wi[a] = __ldg(w_in + (int64_t)i * 5 + a);
            dwi[a] = __ldg(dw_in + (int64_t)i * 5 + a);
        }
        const int eb = __ldg(row_ptr + i);
        const int ee = __ldg(row_end + i);
        double e_acc = 0.0, dedcn_acc = 0.0;
        double fx_acc = 0.0, fy_acc = 0.0, fz_acc = 0.0;
        double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

        for (int e = eb + lane; e < ee; e += 32) {
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                r = sqrtf(fmaxf(fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z)), 1e-16f));
            } else {
                j = __ldg(senders + e);
                r = fmaxf(vl.w, 1e-8f);
            }
            if (r < r_cut && r > 1e-7f) {
                const int z_j = __ldg(species + j);
                const int nr_j = __ldg(nref_sp + z_j);
                const int pair = z_i * S + z_j;
                const float* c6m = c6ref_ev + pair * 25;
                float c6_ij = 0.0f, dc6_i = 0.0f;
                for (int b = 0; b < nr_j; ++b) {
                    const float wjb = __ldg(w_in + (int64_t)j * 5 + b);
                    float s_c6 = 0.0f, s_dc6 = 0.0f;
                    for (int a = 0; a < nr_i; ++a) {
                        const float cab = __ldg(c6m + a * 5 + b);
                        s_c6 = fmaf(wi[a], cab, s_c6);
                        s_dc6 = fmaf(dwi[a], cab, s_dc6);
                    }
                    c6_ij = fmaf(s_c6, wjb, c6_ij);
                    dc6_i = fmaf(s_dc6, wjb, dc6_i);
                }

                const float* pp = cn_pair + pair * 5;
                const float qq = __ldg(pp + 0);
                const float f6 = __ldg(pp + 1);
                const float f8 = __ldg(pp + 2);
                const float gc = __ldg(pp + 3);
                const float dgc = __ldg(pp + 4);

                const float r2 = r * r;
                const float r4 = r2 * r2;
                const float r6 = r4 * r2;
                const float r8 = r4 * r4;
                const float inv_d6 = 1.0f / (r6 + f6);
                const float inv_d8 = 1.0f / (r8 + f8);

                const float g_raw = -s6 * inv_d6 - (s8 * qq) * inv_d8;
                const float g_pair = g_raw - gc - (r - r_cut) * dgc;
                e_acc += 0.5 * (double)c6_ij * (double)g_pair;

                if constexpr (!ENERGY_ONLY) {
                    dedcn_acc += (double)g_pair * (double)dc6_i;
                    const float r5 = r4 * r;
                    const float r7 = r6 * r;
                    const float dg_raw = (6.0f * s6 * r5) * (inv_d6 * inv_d6) +
                                         (8.0f * s8 * qq * r7) * (inv_d8 * inv_d8);
                    const float dg_pair = dg_raw - dgc;
                    const float coef = (c6_ij * dg_pair) / r;
                    fx_acc -= (double)coef * (double)vl.x;
                    fy_acc -= (double)coef * (double)vl.y;
                    fz_acc -= (double)coef * (double)vl.z;

                    if constexpr (STRESS) {
                        const double vc = 0.5 * (double)coef;
                        const double vx = (double)vl.x, vy = (double)vl.y, vz = (double)vl.z;
                        v0 += vc * vx * vx; v1 += vc * vx * vy; v2 += vc * vx * vz;
                        v3 += vc * vy * vy; v4 += vc * vy * vz; v5 += vc * vz * vz;
                    }
                }
            }
        }

        e_acc = disp_warp_sum_d(e_acc);
        if constexpr (!ENERGY_ONLY) {
            dedcn_acc = disp_warp_sum_d(dedcn_acc);
            fx_acc = disp_warp_sum_d(fx_acc);
            fy_acc = disp_warp_sum_d(fy_acc);
            fz_acc = disp_warp_sum_d(fz_acc);
            if constexpr (STRESS) {
                v0 = disp_warp_sum_d(v0); v1 = disp_warp_sum_d(v1); v2 = disp_warp_sum_d(v2);
                v3 = disp_warp_sum_d(v3); v4 = disp_warp_sum_d(v4); v5 = disp_warp_sum_d(v5);
            }
        }
        if (lane == 0) {
            blk_e += e_acc;
            if (atom_energy) atom_energy[i] += e_acc;
            if constexpr (!ENERGY_ONLY) {
                dedcn_out[i] = (float)dedcn_acc;
                if (fx_acc != 0.0 || fy_acc != 0.0 || fz_acc != 0.0) {
                    atomicAdd(forces + (int64_t)i * 3 + 0, fx_acc);
                    atomicAdd(forces + (int64_t)i * 3 + 1, fy_acc);
                    atomicAdd(forces + (int64_t)i * 3 + 2, fz_acc);
                }
                if constexpr (STRESS) {
                    blk_v0 += v0; blk_v1 += v1; blk_v2 += v2;
                    blk_v3 += v3; blk_v4 += v4; blk_v5 += v5;
                }
            }
        }
    }

    if (lane == 0) {
        sm_e[wid] = blk_e;
        if constexpr (!ENERGY_ONLY && STRESS) {
            sm_v[wid][0] = blk_v0; sm_v[wid][1] = blk_v1; sm_v[wid][2] = blk_v2;
            sm_v[wid][3] = blk_v3; sm_v[wid][4] = blk_v4; sm_v[wid][5] = blk_v5;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const double be = sm_e[0] + sm_e[1] + sm_e[2] + sm_e[3];
        if (be != 0.0) atomicAdd(energy, be);
        if constexpr (!ENERGY_ONLY && STRESS) {
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                const double bv = sm_v[0][k] + sm_v[1][k] + sm_v[2][k] + sm_v[3][k];
                if (bv != 0.0) atomicAdd(virial + k, bv);
            }
        }
    }
}

template <bool STRESS, bool PACK_VS>
__global__ void __launch_bounds__(128) ilmlip_disp_cn3_kernel(
    int N, float r_cut,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species, const float* __restrict__ rcov_sp,
    const float* __restrict__ dedcn_in,
    double* __restrict__ forces, double* __restrict__ virial) {
    __shared__ double sm_v[4][6];
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int stride = gridDim.x * 4;
    double blk_v0 = 0.0, blk_v1 = 0.0, blk_v2 = 0.0, blk_v3 = 0.0, blk_v4 = 0.0, blk_v5 = 0.0;

    for (int i = blockIdx.x * 4 + wid; i < N; i += stride) {
        const int z_i = __ldg(species + i);
        const float rcov_i = __ldg(rcov_sp + z_i);
        const float di = __ldg(dedcn_in + i);
        const int eb = __ldg(row_ptr + i);
        const int ee = __ldg(row_end + i);
        double fx_acc = 0.0, fy_acc = 0.0, fz_acc = 0.0;
        double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

        for (int e = eb + lane; e < ee; e += 32) {
            const float4 vl = __ldg(vec_len + e);
            int j;
            float r;
            if constexpr (PACK_VS) {
                j = __float_as_int(vl.w);
                r = sqrtf(fmaxf(fmaf(vl.x, vl.x, fmaf(vl.y, vl.y, vl.z * vl.z)), 1e-16f));
            } else {
                j = __ldg(senders + e);
                r = fmaxf(vl.w, 1e-8f);
            }
            if (r < r_cut && r > 1e-7f) {
                const float rco = rcov_i + __ldg(rcov_sp + __ldg(species + j));
                const float dj = __ldg(dedcn_in + j);
                const double sig = 1.0 / (1.0 + exp(-16.0 * ((double)rco / (double)r - 1.0)));
                const double dcn_dr = sig * (1.0 - sig) * (-16.0 * (double)rco / ((double)r * (double)r));
                const double coef = (((double)di + (double)dj) * dcn_dr) / (double)r;
                fx_acc -= coef * (double)vl.x;
                fy_acc -= coef * (double)vl.y;
                fz_acc -= coef * (double)vl.z;
                if constexpr (STRESS) {
                    const double vc = 0.5 * coef;
                    const double vx = (double)vl.x, vy = (double)vl.y, vz = (double)vl.z;
                    v0 += vc * vx * vx; v1 += vc * vx * vy; v2 += vc * vx * vz;
                    v3 += vc * vy * vy; v4 += vc * vy * vz; v5 += vc * vz * vz;
                }
            }
        }
        fx_acc = disp_warp_sum_d(fx_acc);
        fy_acc = disp_warp_sum_d(fy_acc);
        fz_acc = disp_warp_sum_d(fz_acc);
        if constexpr (STRESS) {
            v0 = disp_warp_sum_d(v0); v1 = disp_warp_sum_d(v1); v2 = disp_warp_sum_d(v2);
            v3 = disp_warp_sum_d(v3); v4 = disp_warp_sum_d(v4); v5 = disp_warp_sum_d(v5);
        }
        if (lane == 0) {
            if (fx_acc != 0.0 || fy_acc != 0.0 || fz_acc != 0.0) {
                atomicAdd(forces + (int64_t)i * 3 + 0, fx_acc);
                atomicAdd(forces + (int64_t)i * 3 + 1, fy_acc);
                atomicAdd(forces + (int64_t)i * 3 + 2, fz_acc);
            }
            if constexpr (STRESS) {
                blk_v0 += v0; blk_v1 += v1; blk_v2 += v2;
                blk_v3 += v3; blk_v4 += v4; blk_v5 += v5;
            }
        }
    }
    if constexpr (STRESS) {
        if (lane == 0) {
            sm_v[wid][0] = blk_v0; sm_v[wid][1] = blk_v1; sm_v[wid][2] = blk_v2;
            sm_v[wid][3] = blk_v3; sm_v[wid][4] = blk_v4; sm_v[wid][5] = blk_v5;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                const double bv = sm_v[0][k] + sm_v[1][k] + sm_v[2][k] + sm_v[3][k];
                if (bv != 0.0) atomicAdd(virial + k, bv);
            }
        }
    }
}

// =================================================================================================
// Option D: Explicit fixed-C6 force-shifted D3(BJ) out to arbitrary r_disp (e.g. 9-12 A) via
// compact on-device spatial cell-binning (works for slabs, fibres in vacuum, clusters, and bulk).
// =================================================================================================
constexpr int kDispFarMaxAxis = 20;
constexpr int kDispFarMaxBins = kDispFarMaxAxis * kDispFarMaxAxis * kDispFarMaxAxis;  // 8000 <= 8192

struct DispFarGrid {
    int nb[3];
    int nbins;
    int reach[3];
    int periodic;
    double h[9];
    double inv[9];
};

struct DispCell9 {
    double h[9];
};

__global__ void ilmlip_disp_far_grid_kernel(const double* __restrict__ cell_h_dev, DispCell9 cell_h_host,
                                            float r_disp, DispFarGrid* __restrict__ g) {
    double c[9];
    for (int k = 0; k < 9; ++k) c[k] = cell_h_dev ? cell_h_dev[k] : cell_h_host.h[k];
    const double det = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) +
                       c[2] * (c[3] * c[7] - c[4] * c[6]);
    const int is_per = (fabs(det) > 1e-12) ? 1 : 0;
    g->periodic = is_per;
    if (!is_per) {
        for (int k = 0; k < 9; ++k) { g->h[k] = 0.0; g->inv[k] = 0.0; }
        g->nb[0] = g->nb[1] = g->nb[2] = 1;
        g->reach[0] = g->reach[1] = g->reach[2] = 0;
        g->nbins = 1;
        return;
    }
    double inv[9];
    const double idet = 1.0 / det;
    inv[0] = (c[4] * c[8] - c[5] * c[7]) * idet; inv[1] = (c[2] * c[7] - c[1] * c[8]) * idet;
    inv[2] = (c[1] * c[5] - c[2] * c[4]) * idet; inv[3] = (c[5] * c[6] - c[3] * c[8]) * idet;
    inv[4] = (c[0] * c[8] - c[2] * c[6]) * idet; inv[5] = (c[2] * c[3] - c[0] * c[5]) * idet;
    inv[6] = (c[3] * c[7] - c[4] * c[6]) * idet; inv[7] = (c[1] * c[6] - c[0] * c[7]) * idet;
    inv[8] = (c[0] * c[4] - c[1] * c[3]) * idet;
    const double vol = fabs(det);
    for (int k = 0; k < 9; ++k) { g->h[k] = c[k]; g->inv[k] = inv[k]; }
    int nbins = 1;
    for (int ax = 0; ax < 3; ++ax) {
        const double* u = c + 3 * ((ax + 1) % 3);
        const double* w = c + 3 * ((ax + 2) % 3);
        const double cx = u[1] * w[2] - u[2] * w[1];
        const double cy = u[2] * w[0] - u[0] * w[2];
        const double cz = u[0] * w[1] - u[1] * w[0];
        const double area = sqrt(cx * cx + cy * cy + cz * cz);
        const double height = (area > 1e-12) ? (vol / area) : 1.0;
        int nb = (int)floor(height / 3.0);
        if (nb < 1) nb = 1;
        if (nb > kDispFarMaxAxis) nb = kDispFarMaxAxis;
        g->nb[ax] = nb;
        g->reach[ax] = (int)ceil((double)r_disp / (height / (double)nb));
        if (g->reach[ax] < 1) g->reach[ax] = 1;
        nbins *= nb;
    }
    g->nbins = nbins;
}

__global__ void ilmlip_disp_far_count_kernel(int N, const double* __restrict__ pos64,
                                             const DispFarGrid* __restrict__ g,
                                             int* __restrict__ atom_bin, int* __restrict__ bin_cnt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (!g->periodic) {
        atom_bin[i] = 0;
        atomicAdd(bin_cnt, 1);
        return;
    }
    const double x = pos64[i * 3 + 0], y = pos64[i * 3 + 1], z = pos64[i * 3 + 2];
    const double* inv = g->inv;
    double f0 = x * inv[0] + y * inv[3] + z * inv[6];
    double f1 = x * inv[1] + y * inv[4] + z * inv[7];
    double f2 = x * inv[2] + y * inv[5] + z * inv[8];
    f0 -= floor(f0); f1 -= floor(f1); f2 -= floor(f2);
    int b0 = (int)(f0 * g->nb[0]); if (b0 >= g->nb[0]) b0 = g->nb[0] - 1; if (b0 < 0) b0 = 0;
    int b1 = (int)(f1 * g->nb[1]); if (b1 >= g->nb[1]) b1 = g->nb[1] - 1; if (b1 < 0) b1 = 0;
    int b2 = (int)(f2 * g->nb[2]); if (b2 >= g->nb[2]) b2 = g->nb[2] - 1; if (b2 < 0) b2 = 0;
    const int b = (b0 * g->nb[1] + b1) * g->nb[2] + b2;
    atom_bin[i] = b;
    atomicAdd(bin_cnt + b, 1);
}

__global__ void ilmlip_disp_far_scan_kernel(const DispFarGrid* __restrict__ g,
                                            const int* __restrict__ bin_cnt,
                                            int* __restrict__ bin_ptr, int* __restrict__ bin_cur) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const int nb = g->nbins;
        int acc = 0;
        for (int b = 0; b < nb; ++b) {
            bin_ptr[b] = acc;
            bin_cur[b] = acc;
            acc += bin_cnt[b];
        }
        bin_ptr[nb] = acc;
    }
}

__global__ void ilmlip_disp_far_scatter_kernel(int N, const double* __restrict__ pos64,
                                               const int* __restrict__ species,
                                               const DispFarGrid* __restrict__ g,
                                               const int* __restrict__ atom_bin,
                                               int* __restrict__ bin_cur,
                                               float4* __restrict__ pos4_sorted,
                                               int* __restrict__ idx_sorted = nullptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double x = pos64[i * 3 + 0], y = pos64[i * 3 + 1], z = pos64[i * 3 + 2];
    float wx, wy, wz;
    if (g->periodic) {
        const double* inv = g->inv;
        const double* h = g->h;
        double f0 = x * inv[0] + y * inv[3] + z * inv[6];
        double f1 = x * inv[1] + y * inv[4] + z * inv[7];
        double f2 = x * inv[2] + y * inv[5] + z * inv[8];
        f0 -= floor(f0); f1 -= floor(f1); f2 -= floor(f2);
        wx = (float)(f0 * h[0] + f1 * h[3] + f2 * h[6]);
        wy = (float)(f0 * h[1] + f1 * h[4] + f2 * h[7]);
        wz = (float)(f0 * h[2] + f1 * h[5] + f2 * h[8]);
    } else {
        wx = (float)x; wy = (float)y; wz = (float)z;
    }
    const int slot = atomicAdd(bin_cur + atom_bin[i], 1);
    pos4_sorted[slot] = make_float4(wx, wy, wz, __int_as_float(species[i]));
    if (idx_sorted) idx_sorted[slot] = i;
}

template <bool STRESS, bool ENERGY_ONLY>
__global__ void __launch_bounds__(128) ilmlip_disp_far_walk_kernel(
    int N, int S, float r_disp,
    const DispFarGrid* __restrict__ g,
    const int* __restrict__ bin_ptr,
    const float4* __restrict__ pos4_sorted,
    const int* __restrict__ atom_bin,
    const double* __restrict__ pos64,
    const int* __restrict__ species,
    const float* __restrict__ disp_pair,
    double* __restrict__ energy, double* __restrict__ atom_energy,
    double* __restrict__ forces, double* __restrict__ virial) {
    extern __shared__ float sm_pair[];
    const int n_pair_f = S * S * 6;
    const bool use_smem = (S <= 16);
    if (use_smem) {
        for (int idx = threadIdx.x; idx < n_pair_f; idx += blockDim.x)
            sm_pair[idx] = __ldg(disp_pair + idx);
    }
    __shared__ double sm_e[4];
    __shared__ double sm_v[4][6];
    __syncthreads();

    const int nb0 = g->nb[0], nb1 = g->nb[1], nb2 = g->nb[2];
    const int r0 = g->reach[0], r1 = g->reach[1], r2 = g->reach[2];
    const float h00 = (float)g->h[0], h01 = (float)g->h[1], h02 = (float)g->h[2];
    const float h10 = (float)g->h[3], h11 = (float)g->h[4], h12 = (float)g->h[5];
    const float h20 = (float)g->h[6], h21 = (float)g->h[7], h22 = (float)g->h[8];
    const float rcut2 = r_disp * r_disp;

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int stride = gridDim.x * 4;
    double blk_e = 0.0;
    double blk_v0 = 0.0, blk_v1 = 0.0, blk_v2 = 0.0, blk_v3 = 0.0, blk_v4 = 0.0, blk_v5 = 0.0;

    for (int i = blockIdx.x * 4 + wid; i < N; i += stride) {
        const int z_i = __ldg(species + i);
        const double x = pos64[i * 3 + 0], y = pos64[i * 3 + 1], z = pos64[i * 3 + 2];
        float px, py, pz;
        if (g->periodic) {
            const double* inv = g->inv;
            double f0 = x * inv[0] + y * inv[3] + z * inv[6];
            double f1 = x * inv[1] + y * inv[4] + z * inv[7];
            double f2 = x * inv[2] + y * inv[5] + z * inv[8];
            f0 -= floor(f0); f1 -= floor(f1); f2 -= floor(f2);
            px = (float)(f0 * g->h[0] + f1 * g->h[3] + f2 * g->h[6]);
            py = (float)(f0 * g->h[1] + f1 * g->h[4] + f2 * g->h[7]);
            pz = (float)(f0 * g->h[2] + f1 * g->h[5] + f2 * g->h[8]);
        } else {
            px = (float)x; py = (float)y; pz = (float)z;
        }
        const int b_i = atom_bin[i];
        const int bi2 = b_i % nb2;
        const int bi1 = (b_i / nb2) % nb1;
        const int bi0 = b_i / (nb1 * nb2);

        double e_acc = 0.0;
        double fx_acc = 0.0, fy_acc = 0.0, fz_acc = 0.0;
        double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

        for (int o0 = -r0; o0 <= r0; ++o0) {
            int t0 = bi0 + o0;
            int img0 = (t0 >= 0) ? (t0 / nb0) : -((-t0 + nb0 - 1) / nb0);
            int m0 = t0 - img0 * nb0;
            for (int o1 = -r1; o1 <= r1; ++o1) {
                int t1 = bi1 + o1;
                int img1 = (t1 >= 0) ? (t1 / nb1) : -((-t1 + nb1 - 1) / nb1);
                int m1 = t1 - img1 * nb1;
                for (int o2 = -r2; o2 <= r2; ++o2) {
                    int t2 = bi2 + o2;
                    int img2 = (t2 >= 0) ? (t2 / nb2) : -((-t2 + nb2 - 1) / nb2);
                    int m2 = t2 - img2 * nb2;
                    const int bj = (m0 * nb1 + m1) * nb2 + m2;
                    const int s0 = __ldg(bin_ptr + bj);
                    const int s1 = __ldg(bin_ptr + bj + 1);
                    if (s0 >= s1) continue;
                    // Shift of sender j relative to primary box: img @ h -> vec = p_i - (p_j + img @ h)
                    const float sx = px - ((float)img0 * h00 + (float)img1 * h10 + (float)img2 * h20);
                    const float sy = py - ((float)img0 * h01 + (float)img1 * h11 + (float)img2 * h21);
                    const float sz = pz - ((float)img0 * h02 + (float)img1 * h12 + (float)img2 * h22);
                    for (int k = s0 + lane; k < s1; k += 32) {
                        const float4 pj = __ldg(pos4_sorted + k);
                        const float vx = sx - pj.x;
                        const float vy = sy - pj.y;
                        const float vz = sz - pj.z;
                        const float r2_v = fmaf(vx, vx, fmaf(vy, vy, vz * vz));
                        if (r2_v < rcut2 && r2_v > 1e-10f) {
                            const float r = sqrtf(r2_v);
                            const int z_j = __float_as_int(pj.w);
                            const int pbase = (z_i * S + z_j) * 6;
                            const float* pp = use_smem ? (sm_pair + pbase) : (disp_pair + pbase);
                            const float c6 = pp[0], c8 = pp[1], f6 = pp[2], f8 = pp[3], es = pp[4], fs = pp[5];
                            const float r4 = r2_v * r2_v;
                            const float r6 = r4 * r2_v;
                            const float r8 = r4 * r4;
                            const float inv_d6 = 1.0f / (r6 + f6);
                            const float inv_d8 = 1.0f / (r8 + f8);
                            const float u = -c6 * inv_d6 - c8 * inv_d8;
                            const float v_pair = u - es - (r - r_disp) * fs;
                            e_acc += 0.5 * (double)v_pair;
                            if constexpr (!ENERGY_ONLY) {
                                const float r5 = r4 * r;
                                const float r7 = r6 * r;
                                const float du = (6.0f * c6 * r5) * (inv_d6 * inv_d6) +
                                                 (8.0f * c8 * r7) * (inv_d8 * inv_d8);
                                const float dv = du - fs;
                                const float coef = dv / r;
                                fx_acc -= (double)coef * (double)vx;
                                fy_acc -= (double)coef * (double)vy;
                                fz_acc -= (double)coef * (double)vz;
                                if constexpr (STRESS) {
                                    const double vc = 0.5 * (double)coef;
                                    v0 += vc * (double)vx * (double)vx;
                                    v1 += vc * (double)vx * (double)vy;
                                    v2 += vc * (double)vx * (double)vz;
                                    v3 += vc * (double)vy * (double)vy;
                                    v4 += vc * (double)vy * (double)vz;
                                    v5 += vc * (double)vz * (double)vz;
                                }
                            }
                        }
                    }
                }
            }
        }

        e_acc = disp_warp_sum_d(e_acc);
        if constexpr (!ENERGY_ONLY) {
            fx_acc = disp_warp_sum_d(fx_acc);
            fy_acc = disp_warp_sum_d(fy_acc);
            fz_acc = disp_warp_sum_d(fz_acc);
            if constexpr (STRESS) {
                v0 = disp_warp_sum_d(v0); v1 = disp_warp_sum_d(v1); v2 = disp_warp_sum_d(v2);
                v3 = disp_warp_sum_d(v3); v4 = disp_warp_sum_d(v4); v5 = disp_warp_sum_d(v5);
            }
        }
        if (lane == 0) {
            blk_e += e_acc;
            if (atom_energy) atom_energy[i] += e_acc;
            if constexpr (!ENERGY_ONLY) {
                if (fx_acc != 0.0 || fy_acc != 0.0 || fz_acc != 0.0) {
                    atomicAdd(forces + (int64_t)i * 3 + 0, fx_acc);
                    atomicAdd(forces + (int64_t)i * 3 + 1, fy_acc);
                    atomicAdd(forces + (int64_t)i * 3 + 2, fz_acc);
                }
                if constexpr (STRESS) {
                    blk_v0 += v0; blk_v1 += v1; blk_v2 += v2;
                    blk_v3 += v3; blk_v4 += v4; blk_v5 += v5;
                }
            }
        }
    }

    if (lane == 0) {
        sm_e[wid] = blk_e;
        if constexpr (!ENERGY_ONLY && STRESS) {
            sm_v[wid][0] = blk_v0; sm_v[wid][1] = blk_v1; sm_v[wid][2] = blk_v2;
            sm_v[wid][3] = blk_v3; sm_v[wid][4] = blk_v4; sm_v[wid][5] = blk_v5;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const double be = sm_e[0] + sm_e[1] + sm_e[2] + sm_e[3];
        if (be != 0.0) atomicAdd(energy, be);
        if constexpr (!ENERGY_ONLY && STRESS) {
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                const double bv = sm_v[0][k] + sm_v[1][k] + sm_v[2][k] + sm_v[3][k];
                if (bv != 0.0) atomicAdd(virial + k, bv);
            }
        }
    }
}

// =================================================================================================
// Option C+D: Coordination-dependent C6 D3(BJ) out to arbitrary r_disp (e.g. 10 A) via the spatial
// cell-binning far-pair grid (handles both periodic cells and non-periodic clusters in vacuum).
// =================================================================================================
__global__ void __launch_bounds__(128) ilmlip_disp_cn_far1_kernel(
    int N, float r_disp,
    const DispFarGrid* __restrict__ g,
    const int* __restrict__ bin_ptr,
    const float4* __restrict__ pos4_sorted,
    const int* __restrict__ atom_bin,
    const double* __restrict__ pos64,
    const int* __restrict__ species,
    const int* __restrict__ nref_sp, const float* __restrict__ cnref_sp,
    const float* __restrict__ rcov_sp,
    float* __restrict__ w_out, float* __restrict__ dw_out) {
    const int nb0 = g->nb[0], nb1 = g->nb[1], nb2 = g->nb[2];
    const int r0 = g->reach[0], r1 = g->reach[1], r2 = g->reach[2];
    const float h00 = (float)g->h[0], h01 = (float)g->h[1], h02 = (float)g->h[2];
    const float h10 = (float)g->h[3], h11 = (float)g->h[4], h12 = (float)g->h[5];
    const float h20 = (float)g->h[6], h21 = (float)g->h[7], h22 = (float)g->h[8];
    const float rcut2 = r_disp * r_disp;

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int stride = gridDim.x * 4;
    for (int i = blockIdx.x * 4 + wid; i < N; i += stride) {
        const int z_i = __ldg(species + i);
        const float rcov_i = __ldg(rcov_sp + z_i);
        const double x = pos64[i * 3 + 0], y = pos64[i * 3 + 1], z = pos64[i * 3 + 2];
        float px, py, pz;
        if (g->periodic) {
            const double* inv = g->inv;
            double f0 = x * inv[0] + y * inv[3] + z * inv[6];
            double f1 = x * inv[1] + y * inv[4] + z * inv[7];
            double f2 = x * inv[2] + y * inv[5] + z * inv[8];
            f0 -= floor(f0); f1 -= floor(f1); f2 -= floor(f2);
            px = (float)(f0 * g->h[0] + f1 * g->h[3] + f2 * g->h[6]);
            py = (float)(f0 * g->h[1] + f1 * g->h[4] + f2 * g->h[7]);
            pz = (float)(f0 * g->h[2] + f1 * g->h[5] + f2 * g->h[8]);
        } else {
            px = (float)x; py = (float)y; pz = (float)z;
        }
        const int bi = __ldg(atom_bin + i);
        const int bi2 = bi % nb2;
        const int bi1 = (bi / nb2) % nb1;
        const int bi0 = bi / (nb1 * nb2);
        double cn = 0.0;
        for (int o0 = -r0; o0 <= r0; ++o0) {
            int t0 = bi0 + o0;
            int img0 = (t0 >= 0) ? (t0 / nb0) : -((-t0 + nb0 - 1) / nb0);
            int m0 = t0 - img0 * nb0;
            for (int o1 = -r1; o1 <= r1; ++o1) {
                int t1 = bi1 + o1;
                int img1 = (t1 >= 0) ? (t1 / nb1) : -((-t1 + nb1 - 1) / nb1);
                int m1 = t1 - img1 * nb1;
                for (int o2 = -r2; o2 <= r2; ++o2) {
                    int t2 = bi2 + o2;
                    int img2 = (t2 >= 0) ? (t2 / nb2) : -((-t2 + nb2 - 1) / nb2);
                    int m2 = t2 - img2 * nb2;
                    const int bj = (m0 * nb1 + m1) * nb2 + m2;
                    const int s0 = __ldg(bin_ptr + bj);
                    const int s1 = __ldg(bin_ptr + bj + 1);
                    if (s0 >= s1) continue;
                    const float sx = px - ((float)img0 * h00 + (float)img1 * h10 + (float)img2 * h20);
                    const float sy = py - ((float)img0 * h01 + (float)img1 * h11 + (float)img2 * h21);
                    const float sz = pz - ((float)img0 * h02 + (float)img1 * h12 + (float)img2 * h22);
                    for (int k = s0 + lane; k < s1; k += 32) {
                        const float4 pj = __ldg(pos4_sorted + k);
                        const float vx = sx - pj.x, vy = sy - pj.y, vz = sz - pj.z;
                        const float r2_v = fmaf(vx, vx, fmaf(vy, vy, vz * vz));
                        if (r2_v < rcut2 && r2_v > 1e-10f) {
                            const float r = sqrtf(r2_v);
                            const float rco = rcov_i + __ldg(rcov_sp + __float_as_int(pj.w));
                            cn += 1.0 / (1.0 + exp(-16.0 * ((double)rco / (double)r - 1.0)));
                        }
                    }
                }
            }
        }
        cn = disp_warp_sum_d(cn);
        if (lane == 0) {
            const int nr = __ldg(nref_sp + z_i);
            float* wi = w_out + (int64_t)i * 5;
            float* dwi = dw_out + (int64_t)i * 5;
            double lw[5], dcn[5];
            double mx = -1e300;
            for (int a = 0; a < nr; ++a) {
                dcn[a] = cn - (double)__ldg(cnref_sp + z_i * 5 + a);
                lw[a] = -4.0 * dcn[a] * dcn[a];
                if (lw[a] > mx) mx = lw[a];
            }
            double ea[5], zsum = 0.0;
            for (int a = 0; a < nr; ++a) { ea[a] = exp(lw[a] - mx); zsum += ea[a]; }
            double wa[5], gmean = 0.0;
            for (int a = 0; a < nr; ++a) { wa[a] = ea[a] / zsum; gmean += wa[a] * (-8.0 * dcn[a]); }
            for (int a = 0; a < 5; ++a) {
                if (a < nr) {
                    wi[a] = (float)wa[a];
                    dwi[a] = (float)(wa[a] * (-8.0 * dcn[a] - gmean));
                } else {
                    wi[a] = 0.0f;
                    dwi[a] = 0.0f;
                }
            }
        }
    }
}

template <bool STRESS, bool ENERGY_ONLY>
__global__ void __launch_bounds__(128) ilmlip_disp_cn_far2_kernel(
    int N, int S, float r_disp, float s6, float s8,
    const DispFarGrid* __restrict__ g,
    const int* __restrict__ bin_ptr,
    const float4* __restrict__ pos4_sorted,
    const int* __restrict__ idx_sorted,
    const int* __restrict__ atom_bin,
    const double* __restrict__ pos64,
    const int* __restrict__ species,
    const int* __restrict__ nref_sp,
    const float* __restrict__ cn_pair,
    const float* __restrict__ c6ref_ev,
    const float* __restrict__ w_in, const float* __restrict__ dw_in,
    float* __restrict__ dedcn_out,
    double* __restrict__ energy, double* __restrict__ atom_energy,
    double* __restrict__ forces, double* __restrict__ virial) {
    __shared__ double sm_e[4];
    __shared__ double sm_v[4][6];
    const int nb0 = g->nb[0], nb1 = g->nb[1], nb2 = g->nb[2];
    const int r0 = g->reach[0], r1 = g->reach[1], r2 = g->reach[2];
    const float h00 = (float)g->h[0], h01 = (float)g->h[1], h02 = (float)g->h[2];
    const float h10 = (float)g->h[3], h11 = (float)g->h[4], h12 = (float)g->h[5];
    const float h20 = (float)g->h[6], h21 = (float)g->h[7], h22 = (float)g->h[8];
    const float rcut2 = r_disp * r_disp;

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int stride = gridDim.x * 4;
    double blk_e = 0.0;
    double blk_v0 = 0.0, blk_v1 = 0.0, blk_v2 = 0.0, blk_v3 = 0.0, blk_v4 = 0.0, blk_v5 = 0.0;

    for (int i = blockIdx.x * 4 + wid; i < N; i += stride) {
        const int z_i = __ldg(species + i);
        const int nr_i = __ldg(nref_sp + z_i);
        float wi[5], dwi[5];
        #pragma unroll
        for (int a = 0; a < 5; ++a) {
            wi[a] = __ldg(w_in + (int64_t)i * 5 + a);
            dwi[a] = __ldg(dw_in + (int64_t)i * 5 + a);
        }
        const double x = pos64[i * 3 + 0], y = pos64[i * 3 + 1], z = pos64[i * 3 + 2];
        float px, py, pz;
        if (g->periodic) {
            const double* inv = g->inv;
            double f0 = x * inv[0] + y * inv[3] + z * inv[6];
            double f1 = x * inv[1] + y * inv[4] + z * inv[7];
            double f2 = x * inv[2] + y * inv[5] + z * inv[8];
            f0 -= floor(f0); f1 -= floor(f1); f2 -= floor(f2);
            px = (float)(f0 * g->h[0] + f1 * g->h[3] + f2 * g->h[6]);
            py = (float)(f0 * g->h[1] + f1 * g->h[4] + f2 * g->h[7]);
            pz = (float)(f0 * g->h[2] + f1 * g->h[5] + f2 * g->h[8]);
        } else {
            px = (float)x; py = (float)y; pz = (float)z;
        }
        const int bi = __ldg(atom_bin + i);
        const int bi2 = bi % nb2;
        const int bi1 = (bi / nb2) % nb1;
        const int bi0 = bi / (nb1 * nb2);
        double e_acc = 0.0, dedcn_acc = 0.0;
        double fx_acc = 0.0, fy_acc = 0.0, fz_acc = 0.0;
        double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

        for (int o0 = -r0; o0 <= r0; ++o0) {
            int t0 = bi0 + o0;
            int img0 = (t0 >= 0) ? (t0 / nb0) : -((-t0 + nb0 - 1) / nb0);
            int m0 = t0 - img0 * nb0;
            for (int o1 = -r1; o1 <= r1; ++o1) {
                int t1 = bi1 + o1;
                int img1 = (t1 >= 0) ? (t1 / nb1) : -((-t1 + nb1 - 1) / nb1);
                int m1 = t1 - img1 * nb1;
                for (int o2 = -r2; o2 <= r2; ++o2) {
                    int t2 = bi2 + o2;
                    int img2 = (t2 >= 0) ? (t2 / nb2) : -((-t2 + nb2 - 1) / nb2);
                    int m2 = t2 - img2 * nb2;
                    const int bj = (m0 * nb1 + m1) * nb2 + m2;
                    const int s0 = __ldg(bin_ptr + bj);
                    const int s1 = __ldg(bin_ptr + bj + 1);
                    if (s0 >= s1) continue;
                    const float sx = px - ((float)img0 * h00 + (float)img1 * h10 + (float)img2 * h20);
                    const float sy = py - ((float)img0 * h01 + (float)img1 * h11 + (float)img2 * h21);
                    const float sz = pz - ((float)img0 * h02 + (float)img1 * h12 + (float)img2 * h22);
                    for (int k = s0 + lane; k < s1; k += 32) {
                        const float4 pj = __ldg(pos4_sorted + k);
                        const float vx = sx - pj.x, vy = sy - pj.y, vz = sz - pj.z;
                        const float r2_v = fmaf(vx, vx, fmaf(vy, vy, vz * vz));
                        if (r2_v < rcut2 && r2_v > 1e-10f) {
                            const float r = sqrtf(r2_v);
                            const int j = __ldg(idx_sorted + k);
                            const int z_j = __float_as_int(pj.w);
                            const int nr_j = __ldg(nref_sp + z_j);
                            const int pair = z_i * S + z_j;
                            const float* c6m = c6ref_ev + pair * 25;
                            float c6_ij = 0.0f, dc6_i = 0.0f;
                            for (int b = 0; b < nr_j; ++b) {
                                const float wjb = __ldg(w_in + (int64_t)j * 5 + b);
                                float s_c6 = 0.0f, s_dc6 = 0.0f;
                                for (int a = 0; a < nr_i; ++a) {
                                    const float cab = __ldg(c6m + a * 5 + b);
                                    s_c6 = fmaf(wi[a], cab, s_c6);
                                    s_dc6 = fmaf(dwi[a], cab, s_dc6);
                                }
                                c6_ij = fmaf(s_c6, wjb, c6_ij);
                                dc6_i = fmaf(s_dc6, wjb, dc6_i);
                            }
                            const float* pp = cn_pair + pair * 5;
                            const float qq = __ldg(pp + 0), f6 = __ldg(pp + 1), f8 = __ldg(pp + 2);
                            const float gc = __ldg(pp + 3), dgc = __ldg(pp + 4);
                            const float r4 = r2_v * r2_v;
                            const float r6 = r4 * r2_v;
                            const float r8 = r4 * r4;
                            const float inv_d6 = 1.0f / (r6 + f6);
                            const float inv_d8 = 1.0f / (r8 + f8);
                            const float g_raw = -s6 * inv_d6 - (s8 * qq) * inv_d8;
                            const float g_pair = g_raw - gc - (r - r_disp) * dgc;
                            e_acc += 0.5 * (double)c6_ij * (double)g_pair;
                            if constexpr (!ENERGY_ONLY) {
                                dedcn_acc += (double)g_pair * (double)dc6_i;
                                const float r5 = r4 * r;
                                const float r7 = r6 * r;
                                const float dg_raw = (6.0f * s6 * r5) * (inv_d6 * inv_d6) +
                                                     (8.0f * s8 * qq * r7) * (inv_d8 * inv_d8);
                                const float dg_pair = dg_raw - dgc;
                                const float coef = (c6_ij * dg_pair) / r;
                                fx_acc -= (double)coef * (double)vx;
                                fy_acc -= (double)coef * (double)vy;
                                fz_acc -= (double)coef * (double)vz;
                                if constexpr (STRESS) {
                                    const double vc = 0.5 * (double)coef;
                                    v0 += vc * (double)vx * (double)vx; v1 += vc * (double)vx * (double)vy;
                                    v2 += vc * (double)vx * (double)vz; v3 += vc * (double)vy * (double)vy;
                                    v4 += vc * (double)vy * (double)vz; v5 += vc * (double)vz * (double)vz;
                                }
                            }
                        }
                    }
                }
            }
        }
        e_acc = disp_warp_sum_d(e_acc);
        if constexpr (!ENERGY_ONLY) {
            dedcn_acc = disp_warp_sum_d(dedcn_acc);
            fx_acc = disp_warp_sum_d(fx_acc); fy_acc = disp_warp_sum_d(fy_acc); fz_acc = disp_warp_sum_d(fz_acc);
            if constexpr (STRESS) {
                v0 = disp_warp_sum_d(v0); v1 = disp_warp_sum_d(v1); v2 = disp_warp_sum_d(v2);
                v3 = disp_warp_sum_d(v3); v4 = disp_warp_sum_d(v4); v5 = disp_warp_sum_d(v5);
            }
        }
        if (lane == 0) {
            blk_e += e_acc;
            if (atom_energy) atom_energy[i] += e_acc;
            if constexpr (!ENERGY_ONLY) {
                dedcn_out[i] = (float)dedcn_acc;
                if (fx_acc != 0.0 || fy_acc != 0.0 || fz_acc != 0.0) {
                    atomicAdd(forces + (int64_t)i * 3 + 0, fx_acc);
                    atomicAdd(forces + (int64_t)i * 3 + 1, fy_acc);
                    atomicAdd(forces + (int64_t)i * 3 + 2, fz_acc);
                }
                if constexpr (STRESS) {
                    blk_v0 += v0; blk_v1 += v1; blk_v2 += v2;
                    blk_v3 += v3; blk_v4 += v4; blk_v5 += v5;
                }
            }
        }
    }
    if (lane == 0) {
        sm_e[wid] = blk_e;
        if constexpr (!ENERGY_ONLY && STRESS) {
            sm_v[wid][0] = blk_v0; sm_v[wid][1] = blk_v1; sm_v[wid][2] = blk_v2;
            sm_v[wid][3] = blk_v3; sm_v[wid][4] = blk_v4; sm_v[wid][5] = blk_v5;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const double be = sm_e[0] + sm_e[1] + sm_e[2] + sm_e[3];
        if (be != 0.0) atomicAdd(energy, be);
        if constexpr (!ENERGY_ONLY && STRESS) {
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                const double bv = sm_v[0][k] + sm_v[1][k] + sm_v[2][k] + sm_v[3][k];
                if (bv != 0.0) atomicAdd(virial + k, bv);
            }
        }
    }
}

template <bool STRESS>
__global__ void __launch_bounds__(128) ilmlip_disp_cn_far3_kernel(
    int N, float r_disp,
    const DispFarGrid* __restrict__ g,
    const int* __restrict__ bin_ptr,
    const float4* __restrict__ pos4_sorted,
    const int* __restrict__ idx_sorted,
    const int* __restrict__ atom_bin,
    const double* __restrict__ pos64,
    const int* __restrict__ species,
    const float* __restrict__ rcov_sp,
    const float* __restrict__ dedcn_in,
    double* __restrict__ forces, double* __restrict__ virial) {
    __shared__ double sm_v[4][6];
    const int nb0 = g->nb[0], nb1 = g->nb[1], nb2 = g->nb[2];
    const int r0 = g->reach[0], r1 = g->reach[1], r2 = g->reach[2];
    const float h00 = (float)g->h[0], h01 = (float)g->h[1], h02 = (float)g->h[2];
    const float h10 = (float)g->h[3], h11 = (float)g->h[4], h12 = (float)g->h[5];
    const float h20 = (float)g->h[6], h21 = (float)g->h[7], h22 = (float)g->h[8];
    const float rcut2 = r_disp * r_disp;

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int stride = gridDim.x * 4;
    double blk_v0 = 0.0, blk_v1 = 0.0, blk_v2 = 0.0, blk_v3 = 0.0, blk_v4 = 0.0, blk_v5 = 0.0;

    for (int i = blockIdx.x * 4 + wid; i < N; i += stride) {
        const int z_i = __ldg(species + i);
        const float rcov_i = __ldg(rcov_sp + z_i);
        const double de_i = (double)__ldg(dedcn_in + i);
        const double x = pos64[i * 3 + 0], y = pos64[i * 3 + 1], z = pos64[i * 3 + 2];
        float px, py, pz;
        if (g->periodic) {
            const double* inv = g->inv;
            double f0 = x * inv[0] + y * inv[3] + z * inv[6];
            double f1 = x * inv[1] + y * inv[4] + z * inv[7];
            double f2 = x * inv[2] + y * inv[5] + z * inv[8];
            f0 -= floor(f0); f1 -= floor(f1); f2 -= floor(f2);
            px = (float)(f0 * g->h[0] + f1 * g->h[3] + f2 * g->h[6]);
            py = (float)(f0 * g->h[1] + f1 * g->h[4] + f2 * g->h[7]);
            pz = (float)(f0 * g->h[2] + f1 * g->h[5] + f2 * g->h[8]);
        } else {
            px = (float)x; py = (float)y; pz = (float)z;
        }
        const int bi = __ldg(atom_bin + i);
        const int bi2 = bi % nb2;
        const int bi1 = (bi / nb2) % nb1;
        const int bi0 = bi / (nb1 * nb2);
        double fx_acc = 0.0, fy_acc = 0.0, fz_acc = 0.0;
        double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

        for (int o0 = -r0; o0 <= r0; ++o0) {
            int t0 = bi0 + o0;
            int img0 = (t0 >= 0) ? (t0 / nb0) : -((-t0 + nb0 - 1) / nb0);
            int m0 = t0 - img0 * nb0;
            for (int o1 = -r1; o1 <= r1; ++o1) {
                int t1 = bi1 + o1;
                int img1 = (t1 >= 0) ? (t1 / nb1) : -((-t1 + nb1 - 1) / nb1);
                int m1 = t1 - img1 * nb1;
                for (int o2 = -r2; o2 <= r2; ++o2) {
                    int t2 = bi2 + o2;
                    int img2 = (t2 >= 0) ? (t2 / nb2) : -((-t2 + nb2 - 1) / nb2);
                    int m2 = t2 - img2 * nb2;
                    const int bj = (m0 * nb1 + m1) * nb2 + m2;
                    const int s0 = __ldg(bin_ptr + bj);
                    const int s1 = __ldg(bin_ptr + bj + 1);
                    if (s0 >= s1) continue;
                    const float sx = px - ((float)img0 * h00 + (float)img1 * h10 + (float)img2 * h20);
                    const float sy = py - ((float)img0 * h01 + (float)img1 * h11 + (float)img2 * h21);
                    const float sz = pz - ((float)img0 * h02 + (float)img1 * h12 + (float)img2 * h22);
                    for (int k = s0 + lane; k < s1; k += 32) {
                        const float4 pj = __ldg(pos4_sorted + k);
                        const float vx = sx - pj.x, vy = sy - pj.y, vz = sz - pj.z;
                        const float r2_v = fmaf(vx, vx, fmaf(vy, vy, vz * vz));
                        if (r2_v < rcut2 && r2_v > 1e-10f) {
                            const float r = sqrtf(r2_v);
                            const float rco = rcov_i + __ldg(rcov_sp + __float_as_int(pj.w));
                            const double ex = exp(-16.0 * ((double)rco / (double)r - 1.0));
                            const double sig = 1.0 / (1.0 + ex);
                            const double dcn_dr = -16.0 * (double)rco * sig * (1.0 - sig) / ((double)r * (double)r);
                            const int j = __ldg(idx_sorted + k);
                            const double coef = (de_i + (double)__ldg(dedcn_in + j)) * dcn_dr / (double)r;
                            fx_acc -= coef * (double)vx;
                            fy_acc -= coef * (double)vy;
                            fz_acc -= coef * (double)vz;
                            if constexpr (STRESS) {
                                const double vc = 0.5 * coef;
                                v0 += vc * (double)vx * (double)vx; v1 += vc * (double)vx * (double)vy;
                                v2 += vc * (double)vx * (double)vz; v3 += vc * (double)vy * (double)vy;
                                v4 += vc * (double)vy * (double)vz; v5 += vc * (double)vz * (double)vz;
                            }
                        }
                    }
                }
            }
        }
        fx_acc = disp_warp_sum_d(fx_acc); fy_acc = disp_warp_sum_d(fy_acc); fz_acc = disp_warp_sum_d(fz_acc);
        if constexpr (STRESS) {
            v0 = disp_warp_sum_d(v0); v1 = disp_warp_sum_d(v1); v2 = disp_warp_sum_d(v2);
            v3 = disp_warp_sum_d(v3); v4 = disp_warp_sum_d(v4); v5 = disp_warp_sum_d(v5);
        }
        if (lane == 0) {
            if (fx_acc != 0.0 || fy_acc != 0.0 || fz_acc != 0.0) {
                atomicAdd(forces + (int64_t)i * 3 + 0, fx_acc);
                atomicAdd(forces + (int64_t)i * 3 + 1, fy_acc);
                atomicAdd(forces + (int64_t)i * 3 + 2, fz_acc);
            }
            if constexpr (STRESS) {
                blk_v0 += v0; blk_v1 += v1; blk_v2 += v2;
                blk_v3 += v3; blk_v4 += v4; blk_v5 += v5;
            }
        }
    }
    if constexpr (STRESS) {
        if (lane == 0) {
            sm_v[wid][0] = blk_v0; sm_v[wid][1] = blk_v1; sm_v[wid][2] = blk_v2;
            sm_v[wid][3] = blk_v3; sm_v[wid][4] = blk_v4; sm_v[wid][5] = blk_v5;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                const double bv = sm_v[0][k] + sm_v[1][k] + sm_v[2][k] + sm_v[3][k];
                if (bv != 0.0) atomicAdd(virial + k, bv);
            }
        }
    }
}

}  // namespace loup
