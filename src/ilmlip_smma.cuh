#pragma once
// Specialized ultra-fast CUDA kernels for SMMA-ILMlip (Scalar-Message Multipole Aggregation):
//   lmsg == 0, lfeat2 in {1, 2}, lout2 == lfeat2, n_msg <= 16, n_base2 == 8, S <= 4.
//   Supports BOTH SW == 8 (n_msg <= 8, 16 atoms/CTA!) and SW == 16 (9..16, 8 atoms/CTA!).
//   v3 upgrades:
//   - Bank-conflict-free float4 LDS.128 SMEM layout s_c4[pair][2*(LF2+1)][SW] (6 LDS.128 vs 24 4-way LDS.32!)
//   - Register-cached my_vl & my_inb in sender bwd (eliminates s_vl SMEM = -2 KB/CTA!)
//   - Inline l<=1 / l<=2 SH (ZERO l=3 work) + in-place EdgeBuf SMEM reuse + minimal shuffle butterfly + FP32 forces!

__device__ __forceinline__ void smma_sh_l1(float vx, float vy, float vz, float len, float3& y1) {
    const float d = len > loup::EPS_SH ? len : loup::EPS_SH;
    const float s = loup::SH_C1 / d;
    y1 = make_float3(vx * s, vy * s, vz * s);
}

__device__ __forceinline__ void smma_sh_l2(
    float vx, float vy, float vz, float len, float3& y1, float4& y2a, float& y2_4) {
    const float d = len > loup::EPS_SH ? len : loup::EPS_SH;
    const float inv_d = 1.0f / d;
    const float x = vx * inv_d, y = vy * inv_d, z = vz * inv_d;
    y1 = make_float3(x * loup::SH_C1, y * loup::SH_C1, z * loup::SH_C1);
    const float s20 = loup::SH_S3 * x * z;
    const float s21 = loup::SH_S3 * x * y;
    const float y2 = y * y;
    const float x2z2 = x * x + z * z;
    const float s22 = y2 - 0.5f * x2z2;
    const float s23 = loup::SH_S3 * y * z;
    const float s24 = (0.5f * loup::SH_S3) * (z * z - x * x);
    y2a  = make_float4(s20 * loup::SH_C2, s21 * loup::SH_C2, s22 * loup::SH_C2, s23 * loup::SH_C2);
    y2_4 = s24 * loup::SH_C2;
}

template <int SW>
__device__ __forceinline__ float smma_sw_sum_f(float v, unsigned mask) {
    if constexpr (SW == 8) return qwarp_sum_f(v, mask);
    else                   return half_warp_sum_f(v, mask);
}

template <int SW>
__device__ __forceinline__ double smma_sw_sum_d(double v, unsigned mask) {
    if constexpr (SW == 8) return qwarp_sum_d(v, mask);
    else                   return half_warp_sum_d(v, mask);
}

__device__ __forceinline__ float smma_dot8(float4 ca, float4 cb, float4 ga, float4 gb) {
    return fmaf(ca.x, ga.x, fmaf(ca.y, ga.y, fmaf(ca.z, ga.z, fmaf(ca.w, ga.w,
           fmaf(cb.x, gb.x, fmaf(cb.y, gb.y, fmaf(cb.z, gb.z, cb.w * gb.w)))))));
}

template <int SW, int LF2, bool REM_BF>
__global__ void __launch_bounds__(128, 4) ilmlip_l2_edge_fwd_smma_kernel(
    int N, int S, int n_msg,
    ChebParam r_max, float a0_over_sig1, float inv_DA2_sig1_kap_sq,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2, const float* __restrict__ Cbf,
    const float* __restrict__ f0, const float* __restrict__ I0,
    Ptr4 A2, float* __restrict__ f1_out) {
    constexpr int ATOMS = 128 / SW;
    constexpr int N_F4 = 2 * (LF2 + 1);
    extern __shared__ float4 smem_c4[];
    __shared__ float4 s_g03[ATOMS][8];
    __shared__ float4 s_g47[ATOMS][8];
    __shared__ float3 s_Y1[ATOMS][8];
    __shared__ float4 s_Y2a[LF2 == 2 ? ATOMS : 1][LF2 == 2 ? 8 : 1];
    __shared__ float  s_Y2_4[LF2 == 2 ? ATOMS : 1][LF2 == 2 ? 8 : 1];
    __shared__ int2   s_jp[ATOMS][8];  // .x = j, .y = pair
    __shared__ float  s_f0[REM_BF ? ATOMS : 1][REM_BF ? 8 : 1];

    // Transpose C2 [pair, 8, LF2+1, n_msg] -> float4 s_c4[pair, 2*(LF2+1), SW] for zero-conflict LDS.128!
    const int l2_tot_cols = (LF2 + 1) * n_msg;
    const int n_slots = S * S * (LF2 + 1) * SW;
    for (int idx = threadIdx.x; idx < n_slots; idx += 128) {
        const int q = idx & (SW - 1);
        const int pl = idx / SW;
        const int pair = pl / (LF2 + 1);
        const int l = pl - pair * (LF2 + 1);
        float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 v1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (q < n_msg) {
            const float* src = C2 + (pair * 8) * l2_tot_cols + l * n_msg + q;
            v0 = make_float4(__ldg(src + 0 * l2_tot_cols), __ldg(src + 1 * l2_tot_cols),
                             __ldg(src + 2 * l2_tot_cols), __ldg(src + 3 * l2_tot_cols));
            v1 = make_float4(__ldg(src + 4 * l2_tot_cols), __ldg(src + 5 * l2_tot_cols),
                             __ldg(src + 6 * l2_tot_cols), __ldg(src + 7 * l2_tot_cols));
        }
        smem_c4[(pair * N_F4 + 2 * l + 0) * SW + q] = v0;
        smem_c4[(pair * N_F4 + 2 * l + 1) * SW + q] = v1;
    }
    float4* s_bf4 = smem_c4 + S * S * N_F4 * SW;
    if constexpr (REM_BF) {
        const int bf_slots = S * S * SW;
        for (int idx = threadIdx.x; idx < bf_slots; idx += 128) {
            const int q = idx & (SW - 1);
            const int pair = idx / SW;
            float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 v1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (q < n_msg) {
                const float* src = Cbf + (pair * 8) * n_msg + q;
                v0 = make_float4(__ldg(src + 0 * n_msg), __ldg(src + 1 * n_msg),
                                 __ldg(src + 2 * n_msg), __ldg(src + 3 * n_msg));
                v1 = make_float4(__ldg(src + 4 * n_msg), __ldg(src + 5 * n_msg),
                                 __ldg(src + 6 * n_msg), __ldg(src + 7 * n_msg));
            }
            s_bf4[(pair * 2 + 0) * SW + q] = v0;
            s_bf4[(pair * 2 + 1) * SW + q] = v1;
        }
    }
    __syncthreads();

    const int sid = (SW == 8) ? (threadIdx.x >> 3) : (threadIdx.x >> 4);
    const int slane = threadIdx.x & (SW - 1);
    const unsigned smask = (SW == 8)
        ? (0xffu << ((sid & 3) * 8))
        : ((sid & 1) ? 0xffff0000u : 0x0000ffffu);
    const int i = blockIdx.x * ATOMS + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const bool active_q = (slane < n_msg);

    float a0 = 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;

    for (int eb_b = eb; eb_b < ee; eb_b += 8) {
        const int bcnt = (ee - eb_b < 8) ? (ee - eb_b) : 8;
        if (slane < bcnt) {
            const int e = eb_b + slane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const float r = fmaxf(vl.w, 1e-8f);
            s_jp[sid][slane] = make_int2(j, z_i * S + __ldg(species + j));
            if constexpr (REM_BF) s_f0[sid][slane] = __ldg(f0 + j);
            float4 g03, g47, d0, d1;
            eval_cheb8_f4<false>(r, r_max, g03, g47, d0, d1);
            s_g03[sid][slane] = g03;
            s_g47[sid][slane] = g47;
            float3 y1;
            if constexpr (LF2 == 1) {
                smma_sh_l1(vl.x, vl.y, vl.z, r, y1);
                s_Y1[sid][slane] = y1;
            } else {
                float4 y2a; float y24;
                smma_sh_l2(vl.x, vl.y, vl.z, r, y1, y2a, y24);
                s_Y1[sid][slane] = y1;
                s_Y2a[sid][slane] = y2a;
                s_Y2_4[sid][slane] = y24;
            }
        }
        __syncwarp(smask);

        if (active_q) {
            for (int bi = 0; bi < bcnt; ++bi) {
                const int2 jp = s_jp[sid][bi];
                const int j = jp.x, pair = jp.y;
                const float4 g03 = s_g03[sid][bi];
                const float4 g47 = s_g47[sid][bi];
                const float4* cp4 = smem_c4 + (pair * N_F4) * SW + slane;
                const float r0 = smma_dot8(cp4[0 * SW], cp4[1 * SW], g03, g47);
                const float r1 = smma_dot8(cp4[2 * SW], cp4[3 * SW], g03, g47);
                float i0 = __ldg(I0 + (int64_t)j * n_msg + slane);
                if constexpr (REM_BF) {
                    const float4* bp4 = s_bf4 + (pair * 2) * SW + slane;
                    const float rbf = smma_dot8(bp4[0], bp4[SW], g03, g47);
                    i0 -= s_f0[sid][bi] * rbf;
                }
                a0 = fmaf(r0, i0, a0);
                const float3 Y1 = s_Y1[sid][bi];
                const float s1 = r1 * i0;
                a1_0 = fmaf(s1, Y1.x, a1_0);
                a1_1 = fmaf(s1, Y1.y, a1_1);
                a1_2 = fmaf(s1, Y1.z, a1_2);
                if constexpr (LF2 == 2) {
                    const float r2 = smma_dot8(cp4[4 * SW], cp4[5 * SW], g03, g47);
                    const float4 Y2a = s_Y2a[sid][bi];
                    const float y24 = s_Y2_4[sid][bi];
                    const float s2 = r2 * i0;
                    a2_0 = fmaf(s2, Y2a.x, a2_0);
                    a2_1 = fmaf(s2, Y2a.y, a2_1);
                    a2_2 = fmaf(s2, Y2a.z, a2_2);
                    a2_3 = fmaf(s2, Y2a.w, a2_3);
                    a2_4 = fmaf(s2, y24,   a2_4);
                }
            }
        }
        __syncwarp(smask);
    }

    float sq_l = 0.0f;
    if (active_q) {
        sq_l = a0 * a0 + a1_0 * a1_0 + a1_1 * a1_1 + a1_2 * a1_2;
        if constexpr (LF2 == 2) {
            sq_l += a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4;
        }
    }
    const float sq_tot = smma_sw_sum_f<SW>(sq_l, smask);
    const float f1 = a0_over_sig1 / sqrtf(1.0f + sq_tot * inv_DA2_sig1_kap_sq);
    if (slane == 0) f1_out[i] = f1;
    if (active_q) {
        A2.p[0][(int64_t)i * n_msg + slane] = a0 * f1;
        float* d1 = A2.p[1] + (int64_t)i * 3 * n_msg + slane;
        d1[0] = a1_0 * f1; d1[n_msg] = a1_1 * f1; d1[2 * n_msg] = a1_2 * f1;
        if constexpr (LF2 == 2) {
            float* d2 = A2.p[2] + (int64_t)i * 5 * n_msg + slane;
            d2[0] = a2_0 * f1; d2[n_msg] = a2_1 * f1; d2[2 * n_msg] = a2_2 * f1;
            d2[3 * n_msg] = a2_3 * f1; d2[4 * n_msg] = a2_4 * f1;
        }
    }
}

// Fused SMMA L2 Projections (W_G[1], W_G[2]) + Gram P^T P + x Standardization + eA_x Caching
template <int SW, int LF2>
__global__ void __launch_bounds__(128, 4) ilmlip_smma_inv_fwd_kernel(
    int N, int n_msg, int r_gram, int n_inv1, int n_feat_pad, Int4 gram_off,
    const int32_t* __restrict__ triu_i, const int32_t* __restrict__ triu_j,
    const float* __restrict__ W_G1, const float* __restrict__ W_G2,
    CPtr4 A2, const float* __restrict__ mu_x, const float* __restrict__ inv_std_x,
    float* __restrict__ x, int8_t* __restrict__ eA_x) {
    constexpr int ATOMS = 128 / SW;
    constexpr int WSTR = SW + 1;
    __shared__ float sW1[SW][WSTR];
    __shared__ float sW2[LF2 == 2 ? SW : 1][LF2 == 2 ? WSTR : 1];
    __shared__ float sA[ATOMS][LF2 == 2 ? 5 : 3][WSTR];
    __shared__ float sP[ATOMS][LF2 == 2 ? 5 : 3][WSTR];

    const int n_wg = n_msg * r_gram;
    for (int idx = threadIdx.x; idx < n_wg; idx += 128) {
        const int q = idx / r_gram, r = idx - q * r_gram;
        sW1[q][r] = __ldg(W_G1 + idx);
        if constexpr (LF2 == 2) sW2[q][r] = __ldg(W_G2 + idx);
    }
    __syncthreads();

    const int sid = (SW == 8) ? (threadIdx.x >> 3) : (threadIdx.x >> 4);
    const int slane = threadIdx.x & (SW - 1);
    const unsigned smask = (SW == 8)
        ? (0xffu << ((sid & 3) * 8))
        : ((sid & 1) ? 0xffff0000u : 0x0000ffffu);
    const int i = blockIdx.x * ATOMS + sid;
    if (i >= N) return;

    float* xi = x + (int64_t)i * n_feat_pad;
    float am = 0.0f;
    for (int f = slane; f < n_inv1; f += SW) {
        const float v = (xi[f] - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
        xi[f] = v;
        am = fmaxf(am, fabsf(v));
    }
    if (slane < n_msg) {
        const int f = n_inv1 + slane;
        const float v = (A2.p[0][(int64_t)i * n_msg + slane] - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
        xi[f] = v;
        am = fmaxf(am, fabsf(v));
    }

    const int n_tri = (r_gram * (r_gram + 1)) >> 1;

    if (slane < n_msg) {
        const float* p1 = A2.p[1] + (int64_t)i * 3 * n_msg + slane;
        sA[sid][0][slane] = p1[0]; sA[sid][1][slane] = p1[n_msg]; sA[sid][2][slane] = p1[2 * n_msg];
    }
    __syncwarp(smask);
    if (slane < r_gram) {
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f;
        for (int q = 0; q < n_msg; ++q) {
            const float w = sW1[q][slane];
            p0 = fmaf(sA[sid][0][q], w, p0); p1 = fmaf(sA[sid][1][q], w, p1); p2 = fmaf(sA[sid][2][q], w, p2);
        }
        sP[sid][0][slane] = p0; sP[sid][1][slane] = p1; sP[sid][2][slane] = p2;
    }
    __syncwarp(smask);
    {
        constexpr float inv_s1 = 0.57735026918962576451f;
        const int goff1 = gram_off.v[1];
        for (int idx = slane; idx < n_tri; idx += SW) {
            const int a = __ldg(triu_i + idx), b = __ldg(triu_j + idx);
            const float s = fmaf(sP[sid][0][a], sP[sid][0][b],
                            fmaf(sP[sid][1][a], sP[sid][1][b], sP[sid][2][a] * sP[sid][2][b]));
            const int f = goff1 + idx;
            const float v = (s * inv_s1 - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
            xi[f] = v;
            am = fmaxf(am, fabsf(v));
        }
    }

    if constexpr (LF2 == 2) {
        __syncwarp(smask);
        if (slane < n_msg) {
            const float* p2 = A2.p[2] + (int64_t)i * 5 * n_msg + slane;
            sA[sid][0][slane] = p2[0]; sA[sid][1][slane] = p2[n_msg]; sA[sid][2][slane] = p2[2 * n_msg];
            sA[sid][3][slane] = p2[3 * n_msg]; sA[sid][4][slane] = p2[4 * n_msg];
        }
        __syncwarp(smask);
        if (slane < r_gram) {
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f, p4 = 0.0f;
            for (int q = 0; q < n_msg; ++q) {
                const float w = sW2[q][slane];
                p0 = fmaf(sA[sid][0][q], w, p0); p1 = fmaf(sA[sid][1][q], w, p1);
                p2 = fmaf(sA[sid][2][q], w, p2); p3 = fmaf(sA[sid][3][q], w, p3); p4 = fmaf(sA[sid][4][q], w, p4);
            }
            sP[sid][0][slane] = p0; sP[sid][1][slane] = p1; sP[sid][2][slane] = p2;
            sP[sid][3][slane] = p3; sP[sid][4][slane] = p4;
        }
        __syncwarp(smask);
        constexpr float inv_s2 = 0.44721359549995793928f;
        const int goff2 = gram_off.v[2];
        for (int idx = slane; idx < n_tri; idx += SW) {
            const int a = __ldg(triu_i + idx), b = __ldg(triu_j + idx);
            const float s = fmaf(sP[sid][0][a], sP[sid][0][b],
                            fmaf(sP[sid][1][a], sP[sid][1][b],
                            fmaf(sP[sid][2][a], sP[sid][2][b],
                            fmaf(sP[sid][3][a], sP[sid][3][b], sP[sid][4][a] * sP[sid][4][b]))));
            const int f = goff2 + idx;
            const float v = (s * inv_s2 - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
            xi[f] = v;
            am = fmaxf(am, fabsf(v));
        }
    }

    if (eA_x) {
        if constexpr (SW == 16) am = fmaxf(am, __shfl_xor_sync(smask, am, 8));
        am = fmaxf(am, __shfl_xor_sync(smask, am, 4));
        am = fmaxf(am, __shfl_xor_sync(smask, am, 2));
        am = fmaxf(am, __shfl_xor_sync(smask, am, 1));
        if (slane == 0) eA_x[i] = ilmlip_exp_from_am(am);
    }
}

// Fused SMMA Backward: dx inv_std_x + Gram bwd + W_G[1,2]^T + f1 RMS Gate bwd + float4 dA2_pack!
template <int SW, int LF2>
__global__ void __launch_bounds__(128, 3) ilmlip_smma_inv_gate_bwd_pack_kernel(
    int N, int n_msg, int r_gram, int n_inv1, int n_feat_pad, float inv_DA2_a0_kap_sq, Int4 gram_off,
    const int32_t* __restrict__ triu_i, const int32_t* __restrict__ triu_j,
    const float* __restrict__ W_G1, const float* __restrict__ W_G2,
    const float* __restrict__ inv_std_x, float* __restrict__ dx,
    CPtr4 A2, const float* __restrict__ f1_arr, float* __restrict__ dA2_pack) {
    constexpr int ATOMS = 128 / SW;
    constexpr int WSTR = SW + 1;
    __shared__ float sW1[SW][WSTR];
    __shared__ float sW2[LF2 == 2 ? SW : 1][LF2 == 2 ? WSTR : 1];
    __shared__ float sA[ATOMS][LF2 == 2 ? 5 : 3][WSTR];
    __shared__ float sP[ATOMS][LF2 == 2 ? 5 : 3][WSTR];
    __shared__ float sG[ATOMS][SW][WSTR];

    const int n_wg = n_msg * r_gram;
    for (int idx = threadIdx.x; idx < n_wg; idx += 128) {
        const int q = idx / r_gram, r = idx - q * r_gram;
        sW1[q][r] = __ldg(W_G1 + idx);
        if constexpr (LF2 == 2) sW2[q][r] = __ldg(W_G2 + idx);
    }
    __syncthreads();

    const int sid = (SW == 8) ? (threadIdx.x >> 3) : (threadIdx.x >> 4);
    const int slane = threadIdx.x & (SW - 1);
    const unsigned smask = (SW == 8)
        ? (0xffu << ((sid & 3) * 8))
        : ((sid & 1) ? 0xffff0000u : 0x0000ffffu);
    const int i = blockIdx.x * ATOMS + sid;
    if (i >= N) return;

    float* dxi = dx + (int64_t)i * n_feat_pad;
    for (int f = slane; f < n_inv1; f += SW) dxi[f] *= __ldg(inv_std_x + f);

    const bool active_q = (slane < n_msg);
    const int n_tri = (r_gram * (r_gram + 1)) >> 1;

    float a0 = 0.0f, da0 = 0.0f;
    if (active_q) {
        a0 = A2.p[0][(int64_t)i * n_msg + slane];
        da0 = dxi[n_inv1 + slane] * __ldg(inv_std_x + n_inv1 + slane);
    }

    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
    if (active_q) {
        const float* p1 = A2.p[1] + (int64_t)i * 3 * n_msg + slane;
        a1_0 = p1[0]; a1_1 = p1[n_msg]; a1_2 = p1[2 * n_msg];
        sA[sid][0][slane] = a1_0; sA[sid][1][slane] = a1_1; sA[sid][2][slane] = a1_2;
    }
    __syncwarp(smask);
    if (slane < r_gram) {
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f;
        for (int q = 0; q < n_msg; ++q) {
            const float w = sW1[q][slane];
            p0 = fmaf(sA[sid][0][q], w, p0); p1 = fmaf(sA[sid][1][q], w, p1); p2 = fmaf(sA[sid][2][q], w, p2);
        }
        sP[sid][0][slane] = p0; sP[sid][1][slane] = p1; sP[sid][2][slane] = p2;
    }
    {
        constexpr float inv_s1 = 0.57735026918962576451f;
        const int goff1 = gram_off.v[1];
        for (int idx = slane; idx < n_tri; idx += SW) {
            const int a = __ldg(triu_i + idx), b = __ldg(triu_j + idx);
            const int f = goff1 + idx;
            const float g = dxi[f] * __ldg(inv_std_x + f) * inv_s1;
            if (a == b) sG[sid][a][a] = 2.0f * g;
            else { sG[sid][a][b] = g; sG[sid][b][a] = g; }
        }
    }
    __syncwarp(smask);
    float dp1_0 = 0.0f, dp1_1 = 0.0f, dp1_2 = 0.0f;
    if (slane < r_gram) {
        for (int b = 0; b < r_gram; ++b) {
            const float gb = sG[sid][slane][b];
            dp1_0 = fmaf(gb, sP[sid][0][b], dp1_0); dp1_1 = fmaf(gb, sP[sid][1][b], dp1_1); dp1_2 = fmaf(gb, sP[sid][2][b], dp1_2);
        }
    }
    __syncwarp(smask);
    if (slane < r_gram) {
        sP[sid][0][slane] = dp1_0; sP[sid][1][slane] = dp1_1; sP[sid][2][slane] = dp1_2;
    }
    __syncwarp(smask);
    float da1_0 = 0.0f, da1_1 = 0.0f, da1_2 = 0.0f;
    if (active_q) {
        for (int r = 0; r < r_gram; ++r) {
            const float w = sW1[slane][r];
            da1_0 = fmaf(sP[sid][0][r], w, da1_0); da1_1 = fmaf(sP[sid][1][r], w, da1_1); da1_2 = fmaf(sP[sid][2][r], w, da1_2);
        }
    }

    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    float da2_0 = 0.0f, da2_1 = 0.0f, da2_2 = 0.0f, da2_3 = 0.0f, da2_4 = 0.0f;
    if constexpr (LF2 == 2) {
        __syncwarp(smask);
        if (active_q) {
            const float* p2 = A2.p[2] + (int64_t)i * 5 * n_msg + slane;
            a2_0 = p2[0]; a2_1 = p2[n_msg]; a2_2 = p2[2 * n_msg]; a2_3 = p2[3 * n_msg]; a2_4 = p2[4 * n_msg];
            sA[sid][0][slane] = a2_0; sA[sid][1][slane] = a2_1; sA[sid][2][slane] = a2_2;
            sA[sid][3][slane] = a2_3; sA[sid][4][slane] = a2_4;
        }
        __syncwarp(smask);
        if (slane < r_gram) {
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f, p4 = 0.0f;
            for (int q = 0; q < n_msg; ++q) {
                const float w = sW2[q][slane];
                p0 = fmaf(sA[sid][0][q], w, p0); p1 = fmaf(sA[sid][1][q], w, p1);
                p2 = fmaf(sA[sid][2][q], w, p2); p3 = fmaf(sA[sid][3][q], w, p3); p4 = fmaf(sA[sid][4][q], w, p4);
            }
            sP[sid][0][slane] = p0; sP[sid][1][slane] = p1; sP[sid][2][slane] = p2;
            sP[sid][3][slane] = p3; sP[sid][4][slane] = p4;
        }
        {
            constexpr float inv_s2 = 0.44721359549995793928f;
            const int goff2 = gram_off.v[2];
            for (int idx = slane; idx < n_tri; idx += SW) {
                const int a = __ldg(triu_i + idx), b = __ldg(triu_j + idx);
                const int f = goff2 + idx;
                const float g = dxi[f] * __ldg(inv_std_x + f) * inv_s2;
                if (a == b) sG[sid][a][a] = 2.0f * g;
                else { sG[sid][a][b] = g; sG[sid][b][a] = g; }
            }
        }
        __syncwarp(smask);
        float dp2_0 = 0.0f, dp2_1 = 0.0f, dp2_2 = 0.0f, dp2_3 = 0.0f, dp2_4 = 0.0f;
        if (slane < r_gram) {
            for (int b = 0; b < r_gram; ++b) {
                const float gb = sG[sid][slane][b];
                dp2_0 = fmaf(gb, sP[sid][0][b], dp2_0); dp2_1 = fmaf(gb, sP[sid][1][b], dp2_1);
                dp2_2 = fmaf(gb, sP[sid][2][b], dp2_2); dp2_3 = fmaf(gb, sP[sid][3][b], dp2_3);
                dp2_4 = fmaf(gb, sP[sid][4][b], dp2_4);
            }
        }
        __syncwarp(smask);
        if (slane < r_gram) {
            sP[sid][0][slane] = dp2_0; sP[sid][1][slane] = dp2_1; sP[sid][2][slane] = dp2_2;
            sP[sid][3][slane] = dp2_3; sP[sid][4][slane] = dp2_4;
        }
        __syncwarp(smask);
        if (active_q) {
            for (int r = 0; r < r_gram; ++r) {
                const float w = sW2[slane][r];
                da2_0 = fmaf(sP[sid][0][r], w, da2_0); da2_1 = fmaf(sP[sid][1][r], w, da2_1);
                da2_2 = fmaf(sP[sid][2][r], w, da2_2); da2_3 = fmaf(sP[sid][3][r], w, da2_3);
                da2_4 = fmaf(sP[sid][4][r], w, da2_4);
            }
        }
    }

    float dot_l = 0.0f;
    if (active_q) {
        dot_l = a0 * da0 + a1_0 * da1_0 + a1_1 * da1_1 + a1_2 * da1_2;
        if constexpr (LF2 == 2) dot_l += a2_0 * da2_0 + a2_1 * da2_1 + a2_2 * da2_2 + a2_3 * da2_3 + a2_4 * da2_4;
    }
    const float dot = smma_sw_sum_f<SW>(dot_l, smask);
    const float f1 = f1_arr[i];
    const float sc = (dot * f1) * inv_DA2_a0_kap_sq;
    if (active_q) {
        if constexpr (LF2 == 1) {
            reinterpret_cast<float4*>(dA2_pack + (int64_t)i * (4 * SW))[slane] = make_float4(
                fmaf(f1, da0, -sc * a0), fmaf(f1, da1_0, -sc * a1_0),
                fmaf(f1, da1_1, -sc * a1_1), fmaf(f1, da1_2, -sc * a1_2));
        } else {
            float* dst = dA2_pack + (int64_t)i * (9 * SW);
            reinterpret_cast<float4*>(dst)[slane] = make_float4(
                fmaf(f1, da0, -sc * a0), fmaf(f1, da1_0, -sc * a1_0),
                fmaf(f1, da1_1, -sc * a1_1), fmaf(f1, da1_2, -sc * a1_2));
            reinterpret_cast<float4*>(dst + 4 * SW)[slane] = make_float4(
                fmaf(f1, da2_0, -sc * a2_0), fmaf(f1, da2_1, -sc * a2_1),
                fmaf(f1, da2_2, -sc * a2_2), fmaf(f1, da2_3, -sc * a2_3));
            dst[8 * SW + slane] = fmaf(f1, da2_4, -sc * a2_4);
        }
    }
}

// Generic gate_bwd + pack for SMMA (fallback when r_bisp > 0 or r_gram > SW)
template <int SW, int LF2>
__global__ void __launch_bounds__(128) ilmlip_gate_bwd_pack_smma_kernel(
    int N, int n_msg, float inv_DA2_a0_kap_sq,
    CPtr4 A2, CPtr4 dA2, const float* __restrict__ f1_arr, float* __restrict__ dA2_pack) {
    constexpr int ATOMS = 128 / SW;
    const int sid = (SW == 8) ? (threadIdx.x >> 3) : (threadIdx.x >> 4);
    const int slane = threadIdx.x & (SW - 1);
    const unsigned smask = (SW == 8)
        ? (0xffu << ((sid & 3) * 8))
        : ((sid & 1) ? 0xffff0000u : 0x0000ffffu);
    const int i = blockIdx.x * ATOMS + sid;
    if (i >= N) return;
    const bool active_q = (slane < n_msg);
    float a0=0, da0=0, a1_0=0, a1_1=0, a1_2=0, da1_0=0, da1_1=0, da1_2=0;
    float a2_0=0, a2_1=0, a2_2=0, a2_3=0, a2_4=0, da2_0=0, da2_1=0, da2_2=0, da2_3=0, da2_4=0;
    float dot_l = 0.0f;
    if (active_q) {
        a0 = A2.p[0][(int64_t)i * n_msg + slane]; da0 = dA2.p[0][(int64_t)i * n_msg + slane];
        const float* pA1 = A2.p[1] + (int64_t)i * 3 * n_msg + slane;
        const float* pd1 = dA2.p[1] + (int64_t)i * 3 * n_msg + slane;
        a1_0 = pA1[0]; a1_1 = pA1[n_msg]; a1_2 = pA1[2*n_msg];
        da1_0 = pd1[0]; da1_1 = pd1[n_msg]; da1_2 = pd1[2*n_msg];
        dot_l = a0*da0 + a1_0*da1_0 + a1_1*da1_1 + a1_2*da1_2;
        if constexpr (LF2 == 2) {
            const float* pA2 = A2.p[2] + (int64_t)i * 5 * n_msg + slane;
            const float* pd2 = dA2.p[2] + (int64_t)i * 5 * n_msg + slane;
            a2_0 = pA2[0]; a2_1 = pA2[n_msg]; a2_2 = pA2[2*n_msg]; a2_3 = pA2[3*n_msg]; a2_4 = pA2[4*n_msg];
            da2_0 = pd2[0]; da2_1 = pd2[n_msg]; da2_2 = pd2[2*n_msg]; da2_3 = pd2[3*n_msg]; da2_4 = pd2[4*n_msg];
            dot_l += a2_0*da2_0 + a2_1*da2_1 + a2_2*da2_2 + a2_3*da2_3 + a2_4*da2_4;
        }
    }
    const float dot = smma_sw_sum_f<SW>(dot_l, smask);
    const float f1 = f1_arr[i];
    const float sc = (dot * f1) * inv_DA2_a0_kap_sq;
    if (active_q) {
        if constexpr (LF2 == 1) {
            reinterpret_cast<float4*>(dA2_pack + (int64_t)i * (4 * SW))[slane] = make_float4(
                fmaf(f1, da0, -sc * a0), fmaf(f1, da1_0, -sc * a1_0),
                fmaf(f1, da1_1, -sc * a1_1), fmaf(f1, da1_2, -sc * a1_2));
        } else {
            float* dst = dA2_pack + (int64_t)i * (9 * SW);
            reinterpret_cast<float4*>(dst)[slane] = make_float4(
                fmaf(f1, da0, -sc * a0), fmaf(f1, da1_0, -sc * a1_0),
                fmaf(f1, da1_1, -sc * a1_1), fmaf(f1, da1_2, -sc * a1_2));
            reinterpret_cast<float4*>(dst + 4 * SW)[slane] = make_float4(
                fmaf(f1, da2_0, -sc * a2_0), fmaf(f1, da2_1, -sc * a2_1),
                fmaf(f1, da2_2, -sc * a2_2), fmaf(f1, da2_3, -sc * a2_3));
            dst[8 * SW + slane] = fmaf(f1, da2_4, -sc * a2_4);
        }
    }
}

// Sender-Owned Zero-Atomic SMMA L2 Edge Backward (v3: zero-conflict float4 LDS.128 C2 + register my_vl/my_inb!)
template <int SW, int LF2, bool REM_BF, bool STRESS, bool F32_F>
__global__ void __launch_bounds__(128, 4) ilmlip_l2_edge_bwd_smma_sender_kernel(
    int N, int S, int n_msg, ChebParam r_max,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2, const float* __restrict__ Cbf,
    const float* __restrict__ f0, const float* __restrict__ I0, const float* __restrict__ dA2_pack,
    float* __restrict__ dI0_out, float* __restrict__ df0_bf,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    using FT = typename std::conditional<F32_F, float, double>::type;
    constexpr int ATOMS = 128 / SW;
    constexpr int N_F4 = 2 * (LF2 + 1);
    union SmmaEdgeBuf {
        struct { float4 g03, g47, dg03, dg47; } g;
        float ed[16];
    };
    extern __shared__ float4 smem_c4[];
    __shared__ SmmaEdgeBuf s_eb[ATOMS][8];
    __shared__ float3 s_Y1[ATOMS][8];
    __shared__ float4 s_Y2a[LF2 == 2 ? ATOMS : 1][LF2 == 2 ? 8 : 1];
    __shared__ float  s_Y2_4[LF2 == 2 ? ATOMS : 1][LF2 == 2 ? 8 : 1];
    __shared__ int2   s_ip[ATOMS][8];  // .x = i_nb, .y = pair
    __shared__ FT     s_sf[ATOMS][3];

    // Transpose C2 -> float4 s_c4[pair, 2*(LF2+1), SW] for zero-conflict LDS.128!
    const int l2_tot_cols = (LF2 + 1) * n_msg;
    const int n_slots = S * S * (LF2 + 1) * SW;
    for (int idx = threadIdx.x; idx < n_slots; idx += 128) {
        const int q = idx & (SW - 1);
        const int pl = idx / SW;
        const int pair = pl / (LF2 + 1);
        const int l = pl - pair * (LF2 + 1);
        float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 v1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (q < n_msg) {
            const float* src = C2 + (pair * 8) * l2_tot_cols + l * n_msg + q;
            v0 = make_float4(__ldg(src + 0 * l2_tot_cols), __ldg(src + 1 * l2_tot_cols),
                             __ldg(src + 2 * l2_tot_cols), __ldg(src + 3 * l2_tot_cols));
            v1 = make_float4(__ldg(src + 4 * l2_tot_cols), __ldg(src + 5 * l2_tot_cols),
                             __ldg(src + 6 * l2_tot_cols), __ldg(src + 7 * l2_tot_cols));
        }
        smem_c4[(pair * N_F4 + 2 * l + 0) * SW + q] = v0;
        smem_c4[(pair * N_F4 + 2 * l + 1) * SW + q] = v1;
    }
    float4* s_bf4 = smem_c4 + S * S * N_F4 * SW;
    if constexpr (REM_BF) {
        const int bf_slots = S * S * SW;
        for (int idx = threadIdx.x; idx < bf_slots; idx += 128) {
            const int q = idx & (SW - 1);
            const int pair = idx / SW;
            float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 v1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (q < n_msg) {
                const float* src = Cbf + (pair * 8) * n_msg + q;
                v0 = make_float4(__ldg(src + 0 * n_msg), __ldg(src + 1 * n_msg),
                                 __ldg(src + 2 * n_msg), __ldg(src + 3 * n_msg));
                v1 = make_float4(__ldg(src + 4 * n_msg), __ldg(src + 5 * n_msg),
                                 __ldg(src + 6 * n_msg), __ldg(src + 7 * n_msg));
            }
            s_bf4[(pair * 2 + 0) * SW + q] = v0;
            s_bf4[(pair * 2 + 1) * SW + q] = v1;
        }
    }
    __syncthreads();

    const int sid = (SW == 8) ? (threadIdx.x >> 3) : (threadIdx.x >> 4);
    const int slane = threadIdx.x & (SW - 1);
    const unsigned smask = (SW == 8)
        ? (0xffu << ((sid & 3) * 8))
        : ((sid & 1) ? 0xffff0000u : 0x0000ffffu);
    const int j = blockIdx.x * ATOMS + sid;
    if (j >= N) return;

    const int z_j = species[j];
    const int eb = row_ptr[j], ee = row_end[j];
    const bool active_q = (slane < n_msg);
    const float f0_j = REM_BF ? __ldg(f0 + j) : 0.0f;
    const float i0_raw = active_q ? __ldg(I0 + (int64_t)j * n_msg + slane) : 0.0f;
    if (slane < 3) s_sf[sid][slane] = (FT)0;

    float acc_di0 = 0.0f, acc_df0 = 0.0f;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += 8) {
        const int bcnt = (ee - eb_b < 8) ? (ee - eb_b) : 8;
        float4 my_vl = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        int my_inb = 0;
        if (slane < bcnt) {
            const int e = eb_b + slane;
            const float4 vl_rev = __ldg(vec_len + e);
            my_vl = make_float4(-vl_rev.x, -vl_rev.y, -vl_rev.z, vl_rev.w);
            my_inb = __ldg(senders + e);
            const float r = fmaxf(vl_rev.w, 1e-8f);
            s_ip[sid][slane] = make_int2(my_inb, __ldg(species + my_inb) * S + z_j);
            float4 g03, g47, dg03, dg47;
            eval_cheb8_f4<true>(r, r_max, g03, g47, dg03, dg47);
            s_eb[sid][slane].g.g03 = g03;   s_eb[sid][slane].g.g47 = g47;
            s_eb[sid][slane].g.dg03 = dg03; s_eb[sid][slane].g.dg47 = dg47;
            float3 y1;
            if constexpr (LF2 == 1) {
                smma_sh_l1(my_vl.x, my_vl.y, my_vl.z, r, y1);
                s_Y1[sid][slane] = y1;
            } else {
                float4 y2a; float y24;
                smma_sh_l2(my_vl.x, my_vl.y, my_vl.z, r, y1, y2a, y24);
                s_Y1[sid][slane] = y1;
                s_Y2a[sid][slane] = y2a;
                s_Y2_4[sid][slane] = y24;
            }
        }
        __syncwarp(smask);

        for (int bi = 0; bi < bcnt; ++bi) {
            float er_l = 0.0f, dY1 = 0.0f, dY2 = 0.0f, dY3 = 0.0f;
            float dY4 = 0.0f, dY5 = 0.0f, dY6 = 0.0f, dY7 = 0.0f, dY8 = 0.0f;
            if (active_q) {
                const int2 ip = s_ip[sid][bi];
                const int i_nb = ip.x, pair = ip.y;
                const float4 g03  = s_eb[sid][bi].g.g03,  g47  = s_eb[sid][bi].g.g47;
                const float4 dg03 = s_eb[sid][bi].g.dg03, dg47 = s_eb[sid][bi].g.dg47;
                const float4* cp4 = smem_c4 + (pair * N_F4) * SW + slane;
                const float4 c0a = cp4[0 * SW], c0b = cp4[1 * SW];
                const float4 c1a = cp4[2 * SW], c1b = cp4[3 * SW];
                const float r0  = smma_dot8(c0a, c0b, g03,  g47);
                const float dr0 = smma_dot8(c0a, c0b, dg03, dg47);
                const float r1  = smma_dot8(c1a, c1b, g03,  g47);
                const float dr1 = smma_dot8(c1a, c1b, dg03, dg47);
                float rbf = 0.0f, drbf = 0.0f;
                if constexpr (REM_BF) {
                    const float4* bp4 = s_bf4 + (pair * 2) * SW + slane;
                    const float4 b0 = bp4[0], b1 = bp4[SW];
                    rbf  = smma_dot8(b0, b1, g03,  g47);
                    drbf = smma_dot8(b0, b1, dg03, dg47);
                }
                const float i0 = REM_BF ? (i0_raw - f0_j * rbf) : i0_raw;
                const float3 Y1 = s_Y1[sid][bi];
                const float4 q0 = (LF2 == 1)
                    ? __ldg(reinterpret_cast<const float4*>(dA2_pack + (int64_t)i_nb * (4 * SW)) + slane)
                    : __ldg(reinterpret_cast<const float4*>(dA2_pack + (int64_t)i_nb * (9 * SW)) + slane);
                const float t1 = q0.y * Y1.x + q0.z * Y1.y + q0.w * Y1.z;
                float di0 = fmaf(q0.x, r0, t1 * r1);
                float dr_dot = fmaf(q0.x, dr0, t1 * dr1);
                const float sr1 = i0 * r1;
                dY1 = sr1 * q0.y; dY2 = sr1 * q0.z; dY3 = sr1 * q0.w;
                if constexpr (LF2 == 2) {
                    const float4 c2a = cp4[4 * SW], c2b = cp4[5 * SW];
                    const float r2  = smma_dot8(c2a, c2b, g03,  g47);
                    const float dr2 = smma_dot8(c2a, c2b, dg03, dg47);
                    const float* p2 = dA2_pack + (int64_t)i_nb * (9 * SW);
                    const float4 q1 = __ldg(reinterpret_cast<const float4*>(p2 + 4 * SW) + slane);
                    const float q24 = __ldg(p2 + 8 * SW + slane);
                    const float4 Y2a = s_Y2a[sid][bi];
                    const float y24 = s_Y2_4[sid][bi];
                    const float t2 = q1.x * Y2a.x + q1.y * Y2a.y + q1.z * Y2a.z + q1.w * Y2a.w + q24 * y24;
                    di0 = fmaf(t2, r2, di0);
                    dr_dot = fmaf(t2, dr2, dr_dot);
                    const float sr2 = i0 * r2;
                    dY4 = sr2 * q1.x; dY5 = sr2 * q1.y; dY6 = sr2 * q1.z; dY7 = sr2 * q1.w; dY8 = sr2 * q24;
                }
                acc_di0 += di0;
                er_l = i0 * dr_dot;
                if constexpr (REM_BF) {
                    acc_df0 = fmaf(-di0, rbf, acc_df0);
                    er_l = fmaf(-di0 * f0_j, drbf, er_l);
                }
            }

            if constexpr (LF2 == 1) {
                const bool p0 = (slane & 1) != 0;
                const float u0 = (p0 ? dY1 : er_l) + __shfl_xor_sync(smask, p0 ? er_l : dY1, 1);
                const float u1 = (p0 ? dY3 : dY2)  + __shfl_xor_sync(smask, p0 ? dY2 : dY3, 1);
                const bool p1 = (slane & 2) != 0;
                float v = (p1 ? u1 : u0) + __shfl_xor_sync(smask, p1 ? u0 : u1, 2);
                v += __shfl_xor_sync(smask, v, 4);
                if constexpr (SW == 16) v += __shfl_xor_sync(smask, v, 8);
                __syncwarp(smask);
                if (slane < 4) s_eb[sid][bi].ed[slane] = v;
            } else if constexpr (SW == 8) {
                const bool p0 = (slane & 1) != 0;
                const float a0 = (p0 ? dY1 : er_l) + __shfl_xor_sync(smask, p0 ? er_l : dY1, 1);
                const float a1 = (p0 ? dY3 : dY2)  + __shfl_xor_sync(smask, p0 ? dY2 : dY3, 1);
                const float a2 = (p0 ? dY5 : dY4)  + __shfl_xor_sync(smask, p0 ? dY4 : dY5, 1);
                const float a3 = (p0 ? dY7 : dY6)  + __shfl_xor_sync(smask, p0 ? dY6 : dY7, 1);
                float a8 = dY8 + __shfl_xor_sync(smask, dY8, 1);
                const bool p1 = (slane & 2) != 0;
                const float b0 = (p1 ? a1 : a0) + __shfl_xor_sync(smask, p1 ? a0 : a1, 2);
                const float b1 = (p1 ? a3 : a2) + __shfl_xor_sync(smask, p1 ? a2 : a3, 2);
                a8 += __shfl_xor_sync(smask, a8, 2);
                const bool p2 = (slane & 4) != 0;
                const float r_out = (p2 ? b1 : b0) + __shfl_xor_sync(smask, p2 ? b0 : b1, 4);
                a8 += __shfl_xor_sync(smask, a8, 4);
                __syncwarp(smask);
                s_eb[sid][bi].ed[slane] = r_out;
                if (slane == 0) s_eb[sid][bi].ed[8] = a8;
            } else {
                const bool p0 = (slane & 1) != 0;
                const float a0 = (p0 ? dY1 : er_l) + __shfl_xor_sync(smask, p0 ? er_l : dY1, 1);
                const float a1 = (p0 ? dY3 : dY2)  + __shfl_xor_sync(smask, p0 ? dY2 : dY3, 1);
                const float a2 = (p0 ? dY5 : dY4)  + __shfl_xor_sync(smask, p0 ? dY4 : dY5, 1);
                const float a3 = (p0 ? dY7 : dY6)  + __shfl_xor_sync(smask, p0 ? dY6 : dY7, 1);
                const float a4 = (p0 ? 0.0f : dY8) + __shfl_xor_sync(smask, p0 ? dY8 : 0.0f, 1);
                const bool p1 = (slane & 2) != 0;
                const float b0 = (p1 ? a1 : a0)    + __shfl_xor_sync(smask, p1 ? a0 : a1, 2);
                const float b1 = (p1 ? a3 : a2)    + __shfl_xor_sync(smask, p1 ? a2 : a3, 2);
                const float b2 = (p1 ? 0.0f : a4)  + __shfl_xor_sync(smask, p1 ? a4 : 0.0f, 2);
                const bool p2 = (slane & 4) != 0;
                const float c0 = (p2 ? b1 : b0)    + __shfl_xor_sync(smask, p2 ? b0 : b1, 4);
                const float c1 = (p2 ? 0.0f : b2)  + __shfl_xor_sync(smask, p2 ? b2 : 0.0f, 4);
                const bool p3 = (slane & 8) != 0;
                const float r_out = (p3 ? c1 : c0) + __shfl_xor_sync(smask, p3 ? c0 : c1, 8);
                __syncwarp(smask);
                if (slane < 9) s_eb[sid][bi].ed[slane] = r_out;
            }
        }
        __syncwarp(smask);

        FT bfx = (FT)0, bfy = (FT)0, bfz = (FT)0;
        if (slane < bcnt) {
            const float r = fmaxf(my_vl.w, 1e-8f);
            const float inv_d = 1.0f / r;
            const float ux = my_vl.x * inv_d, uy = my_vl.y * inv_d, uz = my_vl.z * inv_d;
            const float* ed = s_eb[sid][slane].ed;
            float gx, gy, gz;
            if constexpr (LF2 == 1) {
                sh_grad_l1(ed[1], ed[2], ed[3], gx, gy, gz);
            } else {
                sh_grad_l2(ux, uy, uz, ed[1], ed[2], ed[3], ed[4], ed[5], ed[6], ed[7], ed[8], gx, gy, gz);
            }
            const float dot = ux * gx + uy * gy + uz * gz;
            bfx = (FT)(ux * ed[0] + (gx - ux * dot) * inv_d);
            bfy = (FT)(uy * ed[0] + (gy - uy * dot) * inv_d);
            bfz = (FT)(uz * ed[0] + (gz - uz * dot) * inv_d);
            if constexpr (F32_F) {
                atomicAdd(forces_f32 + (int64_t)my_inb * 3 + 0, -bfx);
                atomicAdd(forces_f32 + (int64_t)my_inb * 3 + 1, -bfy);
                atomicAdd(forces_f32 + (int64_t)my_inb * 3 + 2, -bfz);
            } else {
                atomicAdd(forces + (int64_t)my_inb * 3 + 0, -bfx);
                atomicAdd(forces + (int64_t)my_inb * 3 + 1, -bfy);
                atomicAdd(forces + (int64_t)my_inb * 3 + 2, -bfz);
            }
            if constexpr (STRESS) {
                v0 += (double)my_vl.x * (double)bfx;
                v1 += 0.5 * ((double)my_vl.x * (double)bfy + (double)my_vl.y * (double)bfx);
                v2 += 0.5 * ((double)my_vl.x * (double)bfz + (double)my_vl.z * (double)bfx);
                v3 += (double)my_vl.y * (double)bfy;
                v4 += 0.5 * ((double)my_vl.y * (double)bfz + (double)my_vl.z * (double)bfy);
                v5 += (double)my_vl.z * (double)bfz;
            }
        }
        if constexpr (F32_F) {
            bfx = smma_sw_sum_f<SW>(bfx, smask); bfy = smma_sw_sum_f<SW>(bfy, smask); bfz = smma_sw_sum_f<SW>(bfz, smask);
        } else {
            bfx = smma_sw_sum_d<SW>(bfx, smask); bfy = smma_sw_sum_d<SW>(bfy, smask); bfz = smma_sw_sum_d<SW>(bfz, smask);
        }
        if (slane == 0) { s_sf[sid][0] += bfx; s_sf[sid][1] += bfy; s_sf[sid][2] += bfz; }
        __syncwarp(smask);
    }

    if (active_q) dI0_out[(int64_t)j * n_msg + slane] = acc_di0;
    if constexpr (REM_BF) {
        const float df0_tot = smma_sw_sum_f<SW>(acc_df0, smask);
        if (slane == 0) df0_bf[j] = df0_tot;
    }
    if constexpr (STRESS) {
        v0 = smma_sw_sum_d<SW>(v0, smask); v1 = smma_sw_sum_d<SW>(v1, smask); v2 = smma_sw_sum_d<SW>(v2, smask);
        v3 = smma_sw_sum_d<SW>(v3, smask); v4 = smma_sw_sum_d<SW>(v4, smask); v5 = smma_sw_sum_d<SW>(v5, smask);
    }
    if (slane == 0) {
        const FT sfx = s_sf[sid][0], sfy = s_sf[sid][1], sfz = s_sf[sid][2];
        if (sfx != (FT)0 || sfy != (FT)0 || sfz != (FT)0) {
            if constexpr (F32_F) {
                atomicAdd(forces_f32 + (int64_t)j * 3 + 0, sfx);
                atomicAdd(forces_f32 + (int64_t)j * 3 + 1, sfy);
                atomicAdd(forces_f32 + (int64_t)j * 3 + 2, sfz);
            } else {
                atomicAdd(forces + (int64_t)j * 3 + 0, sfx);
                atomicAdd(forces + (int64_t)j * 3 + 1, sfy);
                atomicAdd(forces + (int64_t)j * 3 + 2, sfz);
            }
        }
        if constexpr (STRESS) {
            if (v0 != 0.0) atomicAdd(virial + 0, v0); if (v1 != 0.0) atomicAdd(virial + 1, v1);
            if (v2 != 0.0) atomicAdd(virial + 2, v2); if (v3 != 0.0) atomicAdd(virial + 3, v3);
            if (v4 != 0.0) atomicAdd(virial + 4, v4); if (v5 != 0.0) atomicAdd(virial + 5, v5);
        }
    }
}
