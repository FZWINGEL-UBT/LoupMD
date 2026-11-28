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

}  // namespace loup
