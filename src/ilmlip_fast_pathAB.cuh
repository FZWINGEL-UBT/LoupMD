#pragma once
// Specialized ultra-fast CUDA kernels for arch=ilmlip Path A (1-layer, n_msg==0)
// and Path B (lean 2-layer: scalar-L2 s0 & vector-8 v1_n8), targeting <= 0.05 us/atom-step!

// 4-lane subwarp reductions (8 atoms per warp for n_msg <= 4 -> 32 atoms/CTA!)
__device__ __forceinline__ float q4_sum_f(float v, unsigned mask) {
    v += __shfl_xor_sync(mask, v, 2);
    v += __shfl_xor_sync(mask, v, 1);
    return v;
}
__device__ __forceinline__ double q4_sum_d(double v, unsigned mask) {
    v += __shfl_xor_sync(mask, v, 2);
    v += __shfl_xor_sync(mask, v, 1);
    return v;
}

// 8-lane subwarp reductions (4 atoms per warp for n_msg <= 8)
__device__ __forceinline__ float qwarp_sum_f(float v, unsigned mask) {
    v += __shfl_xor_sync(mask, v, 4);
    v += __shfl_xor_sync(mask, v, 2);
    v += __shfl_xor_sync(mask, v, 1);
    return v;
}
__device__ __forceinline__ double qwarp_sum_d(double v, unsigned mask) {
    v += __shfl_xor_sync(mask, v, 4);
    v += __shfl_xor_sync(mask, v, 2);
    v += __shfl_xor_sync(mask, v, 1);
    return v;
}

__device__ __forceinline__ int8_t ilmlip_exp_from_am(float am) {
    if (am > 0.0f && isfinite(am)) {
        int e = (int)((__float_as_uint(am) >> 23) & 0xff) - 127 + 1;
        if (e < -108) e = -108;
        if (e > 108) e = 108;
        return (int8_t)e;
    }
    return (int8_t)-127;
}

// =================================================================================================
// Generic 1-layer standardization & backward fallback (for arbitrary 1-layer shapes)
// =================================================================================================
__global__ void __launch_bounds__(128) ilmlip_norm_x_1L_kernel(
    int N, int n_feat, int n_feat_pad,
    const float* __restrict__ mu_x, const float* __restrict__ inv_std_x,
    float* __restrict__ x, int8_t* __restrict__ eA_x) {
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;
    float* xi = x + (int64_t)i * n_feat_pad;
    float am = 0.0f;
    for (int f = lane; f < n_feat; f += 32) {
        const float v = (xi[f] - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
        xi[f] = v;
        am = fmaxf(am, fabsf(v));
    }
    if (eA_x) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
        if (lane == 0) eA_x[i] = ilmlip_exp_from_am(am);
    }
}

__global__ void __launch_bounds__(128) ilmlip_scale_dx_1L_kernel(
    int N, int n_feat, int n_feat_pad,
    const float* __restrict__ inv_std_x, float* __restrict__ dx) {
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;
    float* dxi = dx + (int64_t)i * n_feat_pad;
    for (int f = lane; f < n_feat; f += 32) dxi[f] *= __ldg(inv_std_x + f);
}

// =================================================================================================
// PATH A: Zero-SMEM 1-Layer Specialized L1 Node Forward & Backward (lfeat==1, LMAX in {2,3}, n_msg==0)
// Computes ONLY L=0 invariants, fuses inline standardization + eA_x caching! Zero Bcat traffic!
// =================================================================================================
template <int LMAX>
__global__ void __launch_bounds__(256) ilmlip_l1_node_fwd_1L_lf1_kernel(
    int N, int n_feat_pad, Int4 n_l, CPtr4 A1,
    const float* __restrict__ mu_x, const float* __restrict__ inv_std_x,
    float* __restrict__ x_out, int8_t* __restrict__ eA_x) {
    const int lx = threadIdx.x;
    const int c = blockIdx.y * 32 + lx;
    const int i = blockIdx.x * 8 + threadIdx.y;
    if (i >= N) return;

    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];
    float* xi = x_out + (int64_t)i * n_feat_pad;
    float am = 0.0f;
    auto store_x = [&](int f, float v) {
        const float vn = (v - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
        xi[f] = vn;
        am = fmaxf(am, fabsf(vn));
    };

    const float a0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
    if (c < nl1) { const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c; a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1]; }
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    if (c < nl2) { const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c; a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2]; }
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c;
            a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3];
            a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3];
        }
    }

    const int c01 = nl0 < nl1 ? nl0 : nl1;
    const int c12 = nl1 < nl2 ? nl1 : nl2;
    const int c02 = nl0 < nl2 ? nl0 : nl2;
    const int c23 = (LMAX == 3) ? (nl2 < nl3 ? nl2 : nl3) : 0;
    const int c123 = (LMAX == 3) ? (c23 < nl1 ? c23 : nl1) : 0;
    const int c03 = (LMAX == 3) ? (nl0 < nl3 ? nl0 : nl3) : 0;

    if (c < nl0) store_x(c, a0);
    int xoff = nl0;

    const float b0 = a0 * a0;
    if (c < nl0) store_x(xoff + c, b0);
    xoff += nl0;

    const float b1_0 = a0 * a1_0, b1_1 = a0 * a1_1, b1_2 = a0 * a1_2;

    constexpr float C110 = 0.5773502588272095f;
    const float b2 = C110 * (a1_0 * a1_0 + a1_1 * a1_1 + a1_2 * a1_2);
    if (c < nl1) store_x(xoff + c, b2);
    xoff += nl1;

    constexpr float CS310 = 0.547722578048706f, CS110 = 0.3162277638912201f, CS410 = 0.6324555277824402f;
    const float b3_0 = CS310 * (a1_2 * a2_0 + a1_1 * a2_1 - a1_0 * a2_4) - CS110 * a1_0 * a2_2;
    const float b3_1 = CS310 * (a1_0 * a2_1 + a1_2 * a2_3) + CS410 * a1_1 * a2_2;
    const float b3_2 = CS310 * (a1_0 * a2_0 + a1_1 * a2_3 + a1_2 * a2_4) - CS110 * a1_2 * a2_2;

    constexpr float C220 = 0.4472135901451111f;
    const float b4 = C220 * (a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4);
    if (c < nl2) store_x(xoff + c, b4);
    xoff += nl2;

    float b5_0 = 0.0f, b5_1 = 0.0f, b5_2 = 0.0f, b6 = 0.0f;
    if constexpr (LMAX == 3) {
        constexpr float K1 = 0.4629100561141968f, K2 = 0.37796446681022644f, K3 = 0.11952286213636398f;
        constexpr float K4 = 0.47809144854545593f, K5 = 0.29277002811431885f, K6 = 0.41403934359550476f;
        constexpr float K7 = 0.5070925354957581f, C330 = 0.37796446681022644f;
        b5_0 = K1 * (a2_4 * a3_0 - a2_0 * a3_6) + K2 * (a2_3 * a3_1 - a2_1 * a3_5) +
               K3 * (a2_4 * a3_2 - a2_0 * a3_4) - K5 * a2_1 * a3_3 + K6 * a2_2 * a3_2;
        b5_1 = K2 * (a2_0 * a3_1 + a2_4 * a3_5) + K4 * (a2_1 * a3_2 + a2_3 * a3_4) + K7 * a2_2 * a3_3;
        b5_2 = K1 * (a2_0 * a3_0 + a2_4 * a3_6) + K2 * (a2_1 * a3_1 + a2_3 * a3_5) -
               K3 * (a2_0 * a3_2 + a2_4 * a3_4) - K5 * a2_3 * a3_3 + K6 * a2_2 * a3_4;
        b6 = C330 * (a3_0 * a3_0 + a3_1 * a3_1 + a3_2 * a3_2 + a3_3 * a3_3 + a3_4 * a3_4 + a3_5 * a3_5 + a3_6 * a3_6);
        if (c < nl3) store_x(xoff + c, b6);
        xoff += nl3;
    }

    if (c < nl0) store_x(xoff + c, b0 * a0);
    xoff += nl0;
    if (c < c01) store_x(xoff + c, C110 * (b1_0 * a1_0 + b1_1 * a1_1 + b1_2 * a1_2));
    xoff += c01;
    if (c < c12) store_x(xoff + c, C110 * (b3_0 * a1_0 + b3_1 * a1_1 + b3_2 * a1_2));
    xoff += c12;
    if (c < c02) store_x(xoff + c, b4 * a0);
    xoff += c02;
    if constexpr (LMAX == 3) {
        if (c < c123) store_x(xoff + c, C110 * (b5_0 * a1_0 + b5_1 * a1_1 + b5_2 * a1_2));
        xoff += c123;
        if (c < c03) store_x(xoff + c, b6 * a0);
    }

    if (eA_x && gridDim.y == 1) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
        if (lx == 0) eA_x[i] = ilmlip_exp_from_am(am);
    }
}

template <int LMAX>
__global__ void __launch_bounds__(256) ilmlip_l1_node_bwd_1L_lf1_kernel(
    int N, int n_feat_pad, Int4 n_l, CPtr4 A1,
    const float* __restrict__ inv_std_x, const float* __restrict__ dx, Ptr4 dA1) {
    const int c = blockIdx.y * 32 + threadIdx.x;
    const int i = blockIdx.x * 8 + threadIdx.y;
    if (i >= N) return;

    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];
    const float* dxi = dx + (int64_t)i * n_feat_pad;
    auto rdx = [&](int f) { return dxi[f] * __ldg(inv_std_x + f); };

    const float a0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
    if (c < nl1) { const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c; a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1]; }
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    if (c < nl2) { const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c; a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2]; }
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c;
            a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3];
            a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3];
        }
    }

    const int c01 = nl0 < nl1 ? nl0 : nl1;
    const int c12 = nl1 < nl2 ? nl1 : nl2;
    const int c02 = nl0 < nl2 ? nl0 : nl2;
    const int c23 = (LMAX == 3) ? (nl2 < nl3 ? nl2 : nl3) : 0;
    const int c123 = (LMAX == 3) ? (c23 < nl1 ? c23 : nl1) : 0;
    const int c03 = (LMAX == 3) ? (nl0 < nl3 ? nl0 : nl3) : 0;

    constexpr float C110 = 0.5773502588272095f;
    constexpr float CS310 = 0.547722578048706f, CS110 = 0.3162277638912201f, CS410 = 0.6324555277824402f;
    constexpr float C220 = 0.4472135901451111f;
    constexpr float K1 = 0.4629100561141968f, K2 = 0.37796446681022644f, K3 = 0.11952286213636398f;
    constexpr float K4 = 0.47809144854545593f, K5 = 0.29277002811431885f, K6 = 0.41403934359550476f;
    constexpr float K7 = 0.5070925354957581f, C330 = 0.37796446681022644f;

    const float b0 = a0 * a0;
    const float b1_0 = a0 * a1_0, b1_1 = a0 * a1_1, b1_2 = a0 * a1_2;
    const float b3_0 = CS310 * (a1_2 * a2_0 + a1_1 * a2_1 - a1_0 * a2_4) - CS110 * a1_0 * a2_2;
    const float b3_1 = CS310 * (a1_0 * a2_1 + a1_2 * a2_3) + CS410 * a1_1 * a2_2;
    const float b3_2 = CS310 * (a1_0 * a2_0 + a1_1 * a2_3 + a1_2 * a2_4) - CS110 * a1_2 * a2_2;
    const float b4 = C220 * (a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4);
    float b5_0 = 0.0f, b5_1 = 0.0f, b5_2 = 0.0f, b6 = 0.0f;
    if constexpr (LMAX == 3) {
        b5_0 = K1 * (a2_4 * a3_0 - a2_0 * a3_6) + K2 * (a2_3 * a3_1 - a2_1 * a3_5) +
               K3 * (a2_4 * a3_2 - a2_0 * a3_4) - K5 * a2_1 * a3_3 + K6 * a2_2 * a3_2;
        b5_1 = K2 * (a2_0 * a3_1 + a2_4 * a3_5) + K4 * (a2_1 * a3_2 + a2_3 * a3_4) + K7 * a2_2 * a3_3;
        b5_2 = K1 * (a2_0 * a3_0 + a2_4 * a3_6) + K2 * (a2_1 * a3_1 + a2_3 * a3_5) -
               K3 * (a2_0 * a3_2 + a2_4 * a3_4) - K5 * a2_3 * a3_3 + K6 * a2_2 * a3_4;
        b6 = C330 * (a3_0 * a3_0 + a3_1 * a3_1 + a3_2 * a3_2 + a3_3 * a3_3 + a3_4 * a3_4 + a3_5 * a3_5 + a3_6 * a3_6);
    }

    int xoff = 0;
    float da0 = (c < nl0) ? rdx(c) : 0.0f; xoff += nl0;
    float db0 = (c < nl0) ? rdx(xoff + c) : 0.0f; xoff += nl0;
    float db2 = (c < nl1) ? rdx(xoff + c) : 0.0f; xoff += nl1;
    float db4 = (c < nl2) ? rdx(xoff + c) : 0.0f; xoff += nl2;
    float db6 = 0.0f;
    if constexpr (LMAX == 3) { db6 = (c < nl3) ? rdx(xoff + c) : 0.0f; xoff += nl3; }

    float da1_0 = 0.0f, da1_1 = 0.0f, da1_2 = 0.0f;
    float db1_0 = 0.0f, db1_1 = 0.0f, db1_2 = 0.0f;
    float db3_0 = 0.0f, db3_1 = 0.0f, db3_2 = 0.0f;
    float db5_0 = 0.0f, db5_1 = 0.0f, db5_2 = 0.0f;

    // nu=3 L=0 backprop
    if (c < nl0) { const float g = rdx(xoff + c); db0 = fmaf(g, a0, db0); da0 = fmaf(g, b0, da0); } xoff += nl0;
    if (c < c01) {
        const float gc = rdx(xoff + c) * C110;
        db1_0 = fmaf(gc, a1_0, db1_0); db1_1 = fmaf(gc, a1_1, db1_1); db1_2 = fmaf(gc, a1_2, db1_2);
        da1_0 = fmaf(gc, b1_0, da1_0); da1_1 = fmaf(gc, b1_1, da1_1); da1_2 = fmaf(gc, b1_2, da1_2);
    } xoff += c01;
    if (c < c12) {
        const float gc = rdx(xoff + c) * C110;
        db3_0 = fmaf(gc, a1_0, db3_0); db3_1 = fmaf(gc, a1_1, db3_1); db3_2 = fmaf(gc, a1_2, db3_2);
        da1_0 = fmaf(gc, b3_0, da1_0); da1_1 = fmaf(gc, b3_1, da1_1); da1_2 = fmaf(gc, b3_2, da1_2);
    } xoff += c12;
    if (c < c02) { const float g = rdx(xoff + c); db4 = fmaf(g, a0, db4); da0 = fmaf(g, b4, da0); } xoff += c02;
    if constexpr (LMAX == 3) {
        if (c < c123) {
            const float gc = rdx(xoff + c) * C110;
            db5_0 = fmaf(gc, a1_0, db5_0); db5_1 = fmaf(gc, a1_1, db5_1); db5_2 = fmaf(gc, a1_2, db5_2);
            da1_0 = fmaf(gc, b5_0, da1_0); da1_1 = fmaf(gc, b5_1, da1_1); da1_2 = fmaf(gc, b5_2, da1_2);
        } xoff += c123;
        if (c < c03) { const float g = rdx(xoff + c); db6 = fmaf(g, a0, db6); da0 = fmaf(g, b6, da0); }
    }

    // nu=2 -> da
    da0 = fmaf(2.0f * db0, a0, da0);
    if (c < c01) {
        da0 = fmaf(db1_0, a1_0, fmaf(db1_1, a1_1, fmaf(db1_2, a1_2, da0)));
        da1_0 = fmaf(db1_0, a0, da1_0); da1_1 = fmaf(db1_1, a0, da1_1); da1_2 = fmaf(db1_2, a0, da1_2);
    }
    const float sc1 = 2.0f * C110 * db2;
    da1_0 = fmaf(sc1, a1_0, da1_0); da1_1 = fmaf(sc1, a1_1, da1_1); da1_2 = fmaf(sc1, a1_2, da1_2);

    const float sc2 = 2.0f * C220 * db4;
    float da2_0 = sc2 * a2_0, da2_1 = sc2 * a2_1, da2_2 = sc2 * a2_2, da2_3 = sc2 * a2_3, da2_4 = sc2 * a2_4;
    if (c < c12) {
        da1_0 = fmaf(db3_0, -CS110 * a2_2 - CS310 * a2_4, fmaf(db3_1, CS310 * a2_1, fmaf(db3_2, CS310 * a2_0, da1_0)));
        da1_1 = fmaf(db3_0,  CS310 * a2_1, fmaf(db3_1, CS410 * a2_2, fmaf(db3_2, CS310 * a2_3, da1_1)));
        da1_2 = fmaf(db3_0,  CS310 * a2_0, fmaf(db3_1, CS310 * a2_3, fmaf(db3_2, CS310 * a2_4 - CS110 * a2_2, da1_2)));
        da2_0 = fmaf(CS310, db3_0 * a1_2 + db3_2 * a1_0, da2_0);
        da2_1 = fmaf(CS310, db3_0 * a1_1 + db3_1 * a1_0, da2_1);
        da2_2 = fmaf(db3_1, CS410 * a1_1, fmaf(-CS110, db3_0 * a1_0 + db3_2 * a1_2, da2_2));
        da2_3 = fmaf(CS310, db3_1 * a1_2 + db3_2 * a1_1, da2_3);
        da2_4 = fmaf(CS310, db3_2 * a1_2 - db3_0 * a1_0, da2_4);
    }

    if constexpr (LMAX == 3) {
        const float sc3 = 2.0f * C330 * db6;
        float da3_0 = sc3 * a3_0, da3_1 = sc3 * a3_1, da3_2 = sc3 * a3_2, da3_3 = sc3 * a3_3;
        float da3_4 = sc3 * a3_4, da3_5 = sc3 * a3_5, da3_6 = sc3 * a3_6;
        if (c < c23) {
            da2_0 = fmaf(db5_0, -K1 * a3_6 - K3 * a3_4, fmaf(db5_1, K2 * a3_1, fmaf(db5_2, K1 * a3_0 - K3 * a3_2, da2_0)));
            da2_1 = fmaf(db5_0, -K2 * a3_5 - K5 * a3_3, fmaf(db5_1, K4 * a3_2, fmaf(db5_2, K2 * a3_1, da2_1)));
            da2_2 = fmaf(db5_0,  K6 * a3_2,             fmaf(db5_1, K7 * a3_3, fmaf(db5_2, K6 * a3_4, da2_2)));
            da2_3 = fmaf(db5_0,  K2 * a3_1,             fmaf(db5_1, K4 * a3_4, fmaf(db5_2, K2 * a3_5 - K5 * a3_3, da2_3)));
            da2_4 = fmaf(db5_0,  K1 * a3_0 + K3 * a3_2, fmaf(db5_1, K2 * a3_5, fmaf(db5_2, K1 * a3_6 - K3 * a3_4, da2_4)));
            da3_0 = fmaf(K1, db5_0 * a2_4 + db5_2 * a2_0, da3_0);
            da3_1 = fmaf(K2, db5_0 * a2_3 + db5_1 * a2_0 + db5_2 * a2_1, da3_1);
            da3_2 = fmaf(K3, db5_0 * a2_4 - db5_2 * a2_0, fmaf(K4, db5_1 * a2_1, fmaf(K6, db5_0 * a2_2, da3_2)));
            da3_3 = fmaf(-K5, db5_0 * a2_1 + db5_2 * a2_3, fmaf(K7, db5_1 * a2_2, da3_3));
            da3_4 = fmaf(-K3, db5_0 * a2_0 + db5_2 * a2_4, fmaf(K4, db5_1 * a2_3, fmaf(K6, db5_2 * a2_2, da3_4)));
            da3_5 = fmaf(K2, -db5_0 * a2_1 + db5_1 * a2_4 + db5_2 * a2_3, da3_5);
            da3_6 = fmaf(K1, -db5_0 * a2_0 + db5_2 * a2_4, da3_6);
        }
        if (c < nl3) {
            float* d3 = dA1.p[3] + (int64_t)i * 7 * nl3 + c;
            d3[0] = da3_0; d3[nl3] = da3_1; d3[2*nl3] = da3_2; d3[3*nl3] = da3_3;
            d3[4*nl3] = da3_4; d3[5*nl3] = da3_5; d3[6*nl3] = da3_6;
        }
    }
    if (c < nl0) dA1.p[0][(int64_t)i * nl0 + c] = da0;
    if (c < nl1) { float* d1 = dA1.p[1] + (int64_t)i * 3 * nl1 + c; d1[0] = da1_0; d1[nl1] = da1_1; d1[2 * nl1] = da1_2; }
    if (c < nl2) { float* d2 = dA1.p[2] + (int64_t)i * 5 * nl2 + c; d2[0] = da2_0; d2[nl2] = da2_1; d2[2 * nl2] = da2_2; d2[3 * nl2] = da2_3; d2[4 * nl2] = da2_4; }
}

// =================================================================================================
// PATH B1 (Scalar-L2 2-Hop: lfeat==1, lmsg==0, lfeat2==0, lout2==0, n_msg<=16, n_base==8)
// 1) Fused L1 Node + Scalar W_msg_0 Forward & Backward (ZERO Bcat/dBcat DRAM!)
// 2) Scalar-L2 Edge Forward + Fused x Standardization + eA_x Caching (ZERO SH, ZERO l2_inv_fwd!)
// 3) Fused dx inv_std_x + f1 Gate Backward (ZERO l2_inv_bwd, ZERO dA2_pack!)
// 4) Sender-Owned Zero-Atomic Scalar-L2 Edge Backward (ZERO SH, FP32 RED.E.ADD.F32 forces!)
// =================================================================================================
template <int LMAX, bool M8 = false>
__global__ void __launch_bounds__(256) ilmlip_l1_node_fwd_lf1_s0_kernel(
    int N, int n_feat_pad, int n_msg, Int4 n_l, int m0d,
    CPtr4 A1, const float* __restrict__ W0, float* __restrict__ x_out, float* __restrict__ I0_out) {
    extern __shared__ float sW0[];
    const int tid = threadIdx.x;
    const int nW0 = m0d * n_msg;
    for (int idx = tid; idx < nW0; idx += 256) sW0[idx] = __ldg(W0 + idx);
    __syncthreads();

    const int wid = tid >> 5, lane = tid & 31;
    const int i = blockIdx.x * 8 + wid;
    if (i >= N) return;
    const int grp = M8 ? (lane >> 3) : (lane >> 4);
    const int q = M8 ? (lane & 7) : (lane & 15);
    const bool active_q = (q < n_msg);
    const int c = lane;
    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];
    float* xi = x_out + (int64_t)i * n_feat_pad;

    const float a0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
    if (c < nl1) { const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c; a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1]; }
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    if (c < nl2) { const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c; a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2]; }
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c;
            a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3];
            a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3];
        }
    }

    const int c01 = nl0 < nl1 ? nl0 : nl1;
    const int c12 = nl1 < nl2 ? nl1 : nl2;
    const int c02 = nl0 < nl2 ? nl0 : nl2;
    const int c23 = (LMAX == 3) ? (nl2 < nl3 ? nl2 : nl3) : 0;
    const int c123 = (LMAX == 3) ? (c23 < nl1 ? c23 : nl1) : 0;
    const int c03 = (LMAX == 3) ? (nl0 < nl3 ? nl0 : nl3) : 0;

    float acc_i0 = 0.0f;
    auto add_b0 = [&](float val, int b0off, int lim) {
        const float* wbase = sW0 + b0off * n_msg + q;
        constexpr int STEP = M8 ? 4 : 2;
        for (int k = 0; k < lim; k += STEP) {
            const int kk = k + grp;
            const float v = __shfl_sync(0xffffffffu, val, (kk < lim) ? kk : 0);
            if (active_q && kk < lim) acc_i0 = fmaf(v, wbase[kk * n_msg], acc_i0);
        }
    };

    if (c < nl0) xi[c] = a0;
    add_b0(a0, 0, nl0);
    int xoff = nl0;

    const float b0 = a0 * a0;
    if (c < nl0) xi[xoff + c] = b0;
    add_b0(b0, xoff, nl0);
    xoff += nl0;

    const float b1_0 = a0 * a1_0, b1_1 = a0 * a1_1, b1_2 = a0 * a1_2;
    constexpr float C110 = 0.5773502588272095f;
    const float b2 = C110 * (a1_0 * a1_0 + a1_1 * a1_1 + a1_2 * a1_2);
    if (c < nl1) xi[xoff + c] = b2;
    add_b0(b2, xoff, nl1);
    xoff += nl1;

    constexpr float CS310 = 0.547722578048706f, CS110 = 0.3162277638912201f, CS410 = 0.6324555277824402f;
    const float b3_0 = CS310 * (a1_2 * a2_0 + a1_1 * a2_1 - a1_0 * a2_4) - CS110 * a1_0 * a2_2;
    const float b3_1 = CS310 * (a1_0 * a2_1 + a1_2 * a2_3) + CS410 * a1_1 * a2_2;
    const float b3_2 = CS310 * (a1_0 * a2_0 + a1_1 * a2_3 + a1_2 * a2_4) - CS110 * a1_2 * a2_2;

    constexpr float C220 = 0.4472135901451111f;
    const float b4 = C220 * (a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4);
    if (c < nl2) xi[xoff + c] = b4;
    add_b0(b4, xoff, nl2);
    xoff += nl2;

    float b5_0 = 0.0f, b5_1 = 0.0f, b5_2 = 0.0f, b6 = 0.0f;
    if constexpr (LMAX == 3) {
        constexpr float K1 = 0.4629100561141968f, K2 = 0.37796446681022644f, K3 = 0.11952286213636398f;
        constexpr float K4 = 0.47809144854545593f, K5 = 0.29277002811431885f, K6 = 0.41403934359550476f;
        constexpr float K7 = 0.5070925354957581f, C330 = 0.37796446681022644f;
        b5_0 = K1 * (a2_4 * a3_0 - a2_0 * a3_6) + K2 * (a2_3 * a3_1 - a2_1 * a3_5) +
               K3 * (a2_4 * a3_2 - a2_0 * a3_4) - K5 * a2_1 * a3_3 + K6 * a2_2 * a3_2;
        b5_1 = K2 * (a2_0 * a3_1 + a2_4 * a3_5) + K4 * (a2_1 * a3_2 + a2_3 * a3_4) + K7 * a2_2 * a3_3;
        b5_2 = K1 * (a2_0 * a3_0 + a2_4 * a3_6) + K2 * (a2_1 * a3_1 + a2_3 * a3_5) -
               K3 * (a2_0 * a3_2 + a2_4 * a3_4) - K5 * a2_3 * a3_3 + K6 * a2_2 * a3_4;
        b6 = C330 * (a3_0 * a3_0 + a3_1 * a3_1 + a3_2 * a3_2 + a3_3 * a3_3 + a3_4 * a3_4 + a3_5 * a3_5 + a3_6 * a3_6);
        if (c < nl3) xi[xoff + c] = b6;
        add_b0(b6, xoff, nl3);
        xoff += nl3;
    }

    { const float v = b0 * a0; if (c < nl0) xi[xoff + c] = v; add_b0(v, xoff, nl0); xoff += nl0; }
    { const float v = C110 * (b1_0 * a1_0 + b1_1 * a1_1 + b1_2 * a1_2); if (c < c01) xi[xoff + c] = v; add_b0(v, xoff, c01); xoff += c01; }
    { const float v = C110 * (b3_0 * a1_0 + b3_1 * a1_1 + b3_2 * a1_2); if (c < c12) xi[xoff + c] = v; add_b0(v, xoff, c12); xoff += c12; }
    { const float v = b4 * a0; if (c < c02) xi[xoff + c] = v; add_b0(v, xoff, c02); xoff += c02; }
    if constexpr (LMAX == 3) {
        { const float v = C110 * (b5_0 * a1_0 + b5_1 * a1_1 + b5_2 * a1_2); if (c < c123) xi[xoff + c] = v; add_b0(v, xoff, c123); xoff += c123; }
        { const float v = b6 * a0; if (c < c03) xi[xoff + c] = v; add_b0(v, xoff, c03); }
    }

    acc_i0 += __shfl_xor_sync(0xffffffffu, acc_i0, 16);
    if constexpr (M8) acc_i0 += __shfl_xor_sync(0xffffffffu, acc_i0, 8);
    if (grp == 0 && active_q) I0_out[(int64_t)i * n_msg + q] = acc_i0;
}

template <int LMAX, bool M8 = false>
__global__ void __launch_bounds__(256) ilmlip_l1_node_bwd_lf1_s0_kernel(
    int N, int n_feat_pad, int n_msg, Int4 n_l, int m0d,
    CPtr4 A1, const float* __restrict__ dx, const float* __restrict__ dI0,
    const float* __restrict__ W0_T, Ptr4 dA1) {
    extern __shared__ float sW0T[];
    const int tid = threadIdx.x;
    const int nW0 = n_msg * m0d;
    for (int idx = tid; idx < nW0; idx += 256) sW0T[idx] = __ldg(W0_T + idx);
    __syncthreads();

    const int wid = tid >> 5, c = tid & 31;
    const int i = blockIdx.x * 8 + wid;
    if (i >= N) return;

    __shared__ float s_dI0[8][16];
    if (c < n_msg) s_dI0[wid][c] = __ldg(dI0 + (int64_t)i * n_msg + c);
    else if constexpr (M8) { if (c < 8) s_dI0[wid][c] = 0.0f; }
    __syncwarp();

    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];
    const float* dxi = dx + (int64_t)i * n_feat_pad;
    auto g0_at = [&](int off, int lim) -> float {
        if (c >= lim) return 0.0f;
        float s = dxi[off + c];
        const float* wp = sW0T + off + c;
        if constexpr (M8) {
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                if (q < n_msg) s = fmaf(s_dI0[wid][q], wp[q * m0d], s);
            }
        } else {
            for (int q = 0; q < n_msg; ++q) s = fmaf(s_dI0[wid][q], wp[q * m0d], s);
        }
        return s;
    };

    const float a0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
    if (c < nl1) { const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c; a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1]; }
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    if (c < nl2) { const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c; a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2]; }
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;
    if constexpr (LMAX == 3) {
        if (c < nl3) {
            const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c;
            a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3];
            a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3];
        }
    }

    const int c01 = nl0 < nl1 ? nl0 : nl1;
    const int c12 = nl1 < nl2 ? nl1 : nl2;
    const int c02 = nl0 < nl2 ? nl0 : nl2;
    const int c23 = (LMAX == 3) ? (nl2 < nl3 ? nl2 : nl3) : 0;
    const int c123 = (LMAX == 3) ? (c23 < nl1 ? c23 : nl1) : 0;
    const int c03 = (LMAX == 3) ? (nl0 < nl3 ? nl0 : nl3) : 0;

    constexpr float C110 = 0.5773502588272095f;
    constexpr float CS310 = 0.547722578048706f, CS110 = 0.3162277638912201f, CS410 = 0.6324555277824402f;
    constexpr float C220 = 0.4472135901451111f;
    constexpr float K1 = 0.4629100561141968f, K2 = 0.37796446681022644f, K3 = 0.11952286213636398f;
    constexpr float K4 = 0.47809144854545593f, K5 = 0.29277002811431885f, K6 = 0.41403934359550476f;
    constexpr float K7 = 0.5070925354957581f, C330 = 0.37796446681022644f;

    const float b0 = a0 * a0;
    const float b1_0 = a0 * a1_0, b1_1 = a0 * a1_1, b1_2 = a0 * a1_2;
    const float b3_0 = CS310 * (a1_2 * a2_0 + a1_1 * a2_1 - a1_0 * a2_4) - CS110 * a1_0 * a2_2;
    const float b3_1 = CS310 * (a1_0 * a2_1 + a1_2 * a2_3) + CS410 * a1_1 * a2_2;
    const float b3_2 = CS310 * (a1_0 * a2_0 + a1_1 * a2_3 + a1_2 * a2_4) - CS110 * a1_2 * a2_2;
    const float b4 = C220 * (a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4);
    float b5_0 = 0.0f, b5_1 = 0.0f, b5_2 = 0.0f, b6 = 0.0f;
    if constexpr (LMAX == 3) {
        b5_0 = K1 * (a2_4 * a3_0 - a2_0 * a3_6) + K2 * (a2_3 * a3_1 - a2_1 * a3_5) +
               K3 * (a2_4 * a3_2 - a2_0 * a3_4) - K5 * a2_1 * a3_3 + K6 * a2_2 * a3_2;
        b5_1 = K2 * (a2_0 * a3_1 + a2_4 * a3_5) + K4 * (a2_1 * a3_2 + a2_3 * a3_4) + K7 * a2_2 * a3_3;
        b5_2 = K1 * (a2_0 * a3_0 + a2_4 * a3_6) + K2 * (a2_1 * a3_1 + a2_3 * a3_5) -
               K3 * (a2_0 * a3_2 + a2_4 * a3_4) - K5 * a2_3 * a3_3 + K6 * a2_2 * a3_4;
        b6 = C330 * (a3_0 * a3_0 + a3_1 * a3_1 + a3_2 * a3_2 + a3_3 * a3_3 + a3_4 * a3_4 + a3_5 * a3_5 + a3_6 * a3_6);
    }

    int off = 0;
    float da0 = g0_at(off, nl0); off += nl0;
    float db0 = g0_at(off, nl0); off += nl0;
    float db2 = g0_at(off, nl1); off += nl1;
    float db4 = g0_at(off, nl2); off += nl2;
    float db6 = 0.0f;
    if constexpr (LMAX == 3) { db6 = g0_at(off, nl3); off += nl3; }

    float da1_0 = 0.0f, da1_1 = 0.0f, da1_2 = 0.0f;
    float db1_0 = 0.0f, db1_1 = 0.0f, db1_2 = 0.0f;
    float db3_0 = 0.0f, db3_1 = 0.0f, db3_2 = 0.0f;
    float db5_0 = 0.0f, db5_1 = 0.0f, db5_2 = 0.0f;

    { const float g = g0_at(off, nl0); db0 = fmaf(g, a0, db0); da0 = fmaf(g, b0, da0); off += nl0; }
    { const float gc = g0_at(off, c01) * C110;
      db1_0 = fmaf(gc, a1_0, db1_0); db1_1 = fmaf(gc, a1_1, db1_1); db1_2 = fmaf(gc, a1_2, db1_2);
      da1_0 = fmaf(gc, b1_0, da1_0); da1_1 = fmaf(gc, b1_1, da1_1); da1_2 = fmaf(gc, b1_2, da1_2); off += c01; }
    { const float gc = g0_at(off, c12) * C110;
      db3_0 = fmaf(gc, a1_0, db3_0); db3_1 = fmaf(gc, a1_1, db3_1); db3_2 = fmaf(gc, a1_2, db3_2);
      da1_0 = fmaf(gc, b3_0, da1_0); da1_1 = fmaf(gc, b3_1, da1_1); da1_2 = fmaf(gc, b3_2, da1_2); off += c12; }
    { const float g = g0_at(off, c02); db4 = fmaf(g, a0, db4); da0 = fmaf(g, b4, da0); off += c02; }
    if constexpr (LMAX == 3) {
        { const float gc = g0_at(off, c123) * C110;
          db5_0 = fmaf(gc, a1_0, db5_0); db5_1 = fmaf(gc, a1_1, db5_1); db5_2 = fmaf(gc, a1_2, db5_2);
          da1_0 = fmaf(gc, b5_0, da1_0); da1_1 = fmaf(gc, b5_1, da1_1); da1_2 = fmaf(gc, b5_2, da1_2); off += c123; }
        { const float g = g0_at(off, c03); db6 = fmaf(g, a0, db6); da0 = fmaf(g, b6, da0); }
    }

    da0 = fmaf(2.0f * db0, a0, da0);
    if (c < c01) {
        da0 = fmaf(db1_0, a1_0, fmaf(db1_1, a1_1, fmaf(db1_2, a1_2, da0)));
        da1_0 = fmaf(db1_0, a0, da1_0); da1_1 = fmaf(db1_1, a0, da1_1); da1_2 = fmaf(db1_2, a0, da1_2);
    }
    const float sc1 = 2.0f * C110 * db2;
    da1_0 = fmaf(sc1, a1_0, da1_0); da1_1 = fmaf(sc1, a1_1, da1_1); da1_2 = fmaf(sc1, a1_2, da1_2);
    const float sc2 = 2.0f * C220 * db4;
    float da2_0 = sc2 * a2_0, da2_1 = sc2 * a2_1, da2_2 = sc2 * a2_2, da2_3 = sc2 * a2_3, da2_4 = sc2 * a2_4;
    if (c < c12) {
        da1_0 = fmaf(db3_0, -CS110 * a2_2 - CS310 * a2_4, fmaf(db3_1, CS310 * a2_1, fmaf(db3_2, CS310 * a2_0, da1_0)));
        da1_1 = fmaf(db3_0,  CS310 * a2_1, fmaf(db3_1, CS410 * a2_2, fmaf(db3_2, CS310 * a2_3, da1_1)));
        da1_2 = fmaf(db3_0,  CS310 * a2_0, fmaf(db3_1, CS310 * a2_3, fmaf(db3_2, CS310 * a2_4 - CS110 * a2_2, da1_2)));
        da2_0 = fmaf(CS310, db3_0 * a1_2 + db3_2 * a1_0, da2_0);
        da2_1 = fmaf(CS310, db3_0 * a1_1 + db3_1 * a1_0, da2_1);
        da2_2 = fmaf(db3_1, CS410 * a1_1, fmaf(-CS110, db3_0 * a1_0 + db3_2 * a1_2, da2_2));
        da2_3 = fmaf(CS310, db3_1 * a1_2 + db3_2 * a1_1, da2_3);
        da2_4 = fmaf(CS310, db3_2 * a1_2 - db3_0 * a1_0, da2_4);
    }
    if constexpr (LMAX == 3) {
        const float sc3 = 2.0f * C330 * db6;
        float da3_0 = sc3 * a3_0, da3_1 = sc3 * a3_1, da3_2 = sc3 * a3_2, da3_3 = sc3 * a3_3;
        float da3_4 = sc3 * a3_4, da3_5 = sc3 * a3_5, da3_6 = sc3 * a3_6;
        if (c < c23) {
            da2_0 = fmaf(db5_0, -K1 * a3_6 - K3 * a3_4, fmaf(db5_1, K2 * a3_1, fmaf(db5_2, K1 * a3_0 - K3 * a3_2, da2_0)));
            da2_1 = fmaf(db5_0, -K2 * a3_5 - K5 * a3_3, fmaf(db5_1, K4 * a3_2, fmaf(db5_2, K2 * a3_1, da2_1)));
            da2_2 = fmaf(db5_0,  K6 * a3_2,             fmaf(db5_1, K7 * a3_3, fmaf(db5_2, K6 * a3_4, da2_2)));
            da2_3 = fmaf(db5_0,  K2 * a3_1,             fmaf(db5_1, K4 * a3_4, fmaf(db5_2, K2 * a3_5 - K5 * a3_3, da2_3)));
            da2_4 = fmaf(db5_0,  K1 * a3_0 + K3 * a3_2, fmaf(db5_1, K2 * a3_5, fmaf(db5_2, K1 * a3_6 - K3 * a3_4, da2_4)));
            da3_0 = fmaf(K1, db5_0 * a2_4 + db5_2 * a2_0, da3_0);
            da3_1 = fmaf(K2, db5_0 * a2_3 + db5_1 * a2_0 + db5_2 * a2_1, da3_1);
            da3_2 = fmaf(K3, db5_0 * a2_4 - db5_2 * a2_0, fmaf(K4, db5_1 * a2_1, fmaf(K6, db5_0 * a2_2, da3_2)));
            da3_3 = fmaf(-K5, db5_0 * a2_1 + db5_2 * a2_3, fmaf(K7, db5_1 * a2_2, da3_3));
            da3_4 = fmaf(-K3, db5_0 * a2_0 + db5_2 * a2_4, fmaf(K4, db5_1 * a2_3, fmaf(K6, db5_2 * a2_2, da3_4)));
            da3_5 = fmaf(K2, -db5_0 * a2_1 + db5_1 * a2_4 + db5_2 * a2_3, da3_5);
            da3_6 = fmaf(K1, -db5_0 * a2_0 + db5_2 * a2_4, da3_6);
        }
        if (c < nl3) {
            float* d3 = dA1.p[3] + (int64_t)i * 7 * nl3 + c;
            d3[0] = da3_0; d3[nl3] = da3_1; d3[2*nl3] = da3_2; d3[3*nl3] = da3_3;
            d3[4*nl3] = da3_4; d3[5*nl3] = da3_5; d3[6*nl3] = da3_6;
        }
    }
    if (c < nl0) dA1.p[0][(int64_t)i * nl0 + c] = da0;
    if (c < nl1) { float* d1 = dA1.p[1] + (int64_t)i * 3 * nl1 + c; d1[0] = da1_0; d1[nl1] = da1_1; d1[2 * nl1] = da1_2; }
    if (c < nl2) { float* d2 = dA1.p[2] + (int64_t)i * 5 * nl2 + c; d2[0] = da2_0; d2[nl2] = da2_1; d2[2 * nl2] = da2_2; d2[3 * nl2] = da2_3; d2[4 * nl2] = da2_4; }
}

template <bool REM_BF>
__global__ void __launch_bounds__(128, 6) ilmlip_l2_edge_fwd_s0_norm_kernel(
    int N, int S, int n_msg, int n_inv1, int n_feat_pad,
    ChebParam r_max, float a0_over_sig1, float inv_DA2_sig1_kap_sq,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2, const float* __restrict__ Cbf,
    const float* __restrict__ f0, const float* __restrict__ I0,
    const float* __restrict__ mu_x, const float* __restrict__ inv_std_x,
    float* __restrict__ A2_0, float* __restrict__ f1_out, float* __restrict__ x, int8_t* __restrict__ eA_x) {
    extern __shared__ float smem_c[];
    __shared__ float4 s_g03[8][9];
    __shared__ float4 s_g47[8][9];
    __shared__ int    s_j[8][9];
    __shared__ int    s_pair[8][9];
    __shared__ float  s_f0[8][9];

    const int c2_elems = S * S * 8 * n_msg;
    for (int idx = threadIdx.x; idx < c2_elems; idx += 128) smem_c[idx] = __ldg(C2 + idx);
    if constexpr (REM_BF) {
        float* sbf = smem_c + c2_elems;
        for (int idx = threadIdx.x; idx < c2_elems; idx += 128) sbf[idx] = __ldg(Cbf + idx);
    }
    __syncthreads();

    const int sid = threadIdx.x >> 4, slane = threadIdx.x & 15;
    const unsigned smask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const bool active_q = (slane < n_msg);
    const float* sbf_base = REM_BF ? (smem_c + c2_elems) : nullptr;
    float acc = 0.0f;

    for (int eb_b = eb; eb_b < ee; eb_b += 8) {
        const int bcnt = (ee - eb_b < 8) ? (ee - eb_b) : 8;
        if (slane < bcnt) {
            const int e = eb_b + slane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const float r = fmaxf(vl.w, 1e-8f);
            s_j[sid][slane] = j;
            s_pair[sid][slane] = z_i * S + __ldg(species + j);
            if constexpr (REM_BF) s_f0[sid][slane] = __ldg(f0 + j);
            float4 g03, g47, d0, d1;
            eval_cheb8_f4<false>(r, r_max, g03, g47, d0, d1);
            s_g03[sid][slane] = g03;
            s_g47[sid][slane] = g47;
        }
        __syncwarp(smask);

        if (active_q) {
            for (int bi = 0; bi < bcnt; ++bi) {
                const int j = s_j[sid][bi];
                const int pair = s_pair[sid][bi];
                const float4 g03 = s_g03[sid][bi];
                const float4 g47 = s_g47[sid][bi];
                const float* cp = smem_c + (pair * 8) * n_msg + slane;
                const float r2 = fmaf(cp[7 * n_msg], g47.w, fmaf(cp[6 * n_msg], g47.z, fmaf(cp[5 * n_msg], g47.y, fmaf(cp[4 * n_msg], g47.x,
                                 fmaf(cp[3 * n_msg], g03.w, fmaf(cp[2 * n_msg], g03.z, fmaf(cp[1 * n_msg], g03.y, cp[0] * g03.x)))))));
                float i0 = __ldg(I0 + (int64_t)j * n_msg + slane);
                if constexpr (REM_BF) {
                    const float* bp = sbf_base + (pair * 8) * n_msg + slane;
                    const float rbf = fmaf(bp[7 * n_msg], g47.w, fmaf(bp[6 * n_msg], g47.z, fmaf(bp[5 * n_msg], g47.y, fmaf(bp[4 * n_msg], g47.x,
                                      fmaf(bp[3 * n_msg], g03.w, fmaf(bp[2 * n_msg], g03.z, fmaf(bp[1 * n_msg], g03.y, bp[0] * g03.x)))))));
                    i0 -= s_f0[sid][bi] * rbf;
                }
                acc = fmaf(r2, i0, acc);
            }
        }
        __syncwarp(smask);
    }

    const float sq_tot = half_warp_sum_f(active_q ? (acc * acc) : 0.0f, smask);
    const float f1 = a0_over_sig1 / sqrtf(1.0f + sq_tot * inv_DA2_sig1_kap_sq);
    if (slane == 0) f1_out[i] = f1;

    float* xi = x + (int64_t)i * n_feat_pad;
    float am = 0.0f;
    for (int f = slane; f < n_inv1; f += 16) {
        const float v = (xi[f] - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
        xi[f] = v;
        am = fmaxf(am, fabsf(v));
    }
    if (active_q) {
        const float a2 = acc * f1;
        A2_0[(int64_t)i * n_msg + slane] = a2;
        const int f = n_inv1 + slane;
        const float v = (a2 - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
        xi[f] = v;
        am = fmaxf(am, fabsf(v));
    }
    if (eA_x) {
        am = fmaxf(am, __shfl_xor_sync(smask, am, 8));
        am = fmaxf(am, __shfl_xor_sync(smask, am, 4));
        am = fmaxf(am, __shfl_xor_sync(smask, am, 2));
        am = fmaxf(am, __shfl_xor_sync(smask, am, 1));
        if (slane == 0) eA_x[i] = ilmlip_exp_from_am(am);
    }
}

__global__ void __launch_bounds__(128) ilmlip_gate_bwd_s0_norm_kernel(
    int N, int n_msg, int n_inv1, int n_feat_pad, float inv_DA2_a0_kap_sq,
    const float* __restrict__ inv_std_x, float* __restrict__ dx,
    const float* __restrict__ A2_0, const float* __restrict__ f1_arr, float* __restrict__ dA2_0) {
    const int sid = threadIdx.x >> 4, slane = threadIdx.x & 15;
    const unsigned smask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;
    float* dxi = dx + (int64_t)i * n_feat_pad;
    for (int f = slane; f < n_inv1; f += 16) dxi[f] *= __ldg(inv_std_x + f);
    float av = 0.0f, dv = 0.0f;
    if (slane < n_msg) {
        av = A2_0[(int64_t)i * n_msg + slane];
        dv = dxi[n_inv1 + slane] * __ldg(inv_std_x + n_inv1 + slane);
    }
    const float dot = half_warp_sum_f(av * dv, smask);
    const float f1 = f1_arr[i];
    const float sc = (dot * f1) * inv_DA2_a0_kap_sq;
    if (slane < n_msg) dA2_0[(int64_t)i * n_msg + slane] = fmaf(f1, dv, -sc * av);
}

template <bool REM_BF, bool STRESS, bool F32_F>
__global__ void __launch_bounds__(128, 6) ilmlip_l2_edge_bwd_s0_sender_kernel(
    int N, int S, int n_msg, ChebParam r_max,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2, const float* __restrict__ Cbf,
    const float* __restrict__ f0, const float* __restrict__ I0, const float* __restrict__ dA2_0,
    float* __restrict__ dI0_out, float* __restrict__ df0_bf,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    using FT = typename std::conditional<F32_F, float, double>::type;
    extern __shared__ float smem_c[];
    __shared__ float4 s_g03[8][9];
    __shared__ float4 s_g47[8][9];
    __shared__ float4 s_dg03[8][9];
    __shared__ float4 s_dg47[8][9];
    __shared__ float4 s_vl[8][9];
    __shared__ int    s_i[8][9];
    __shared__ int    s_pair[8][9];
    __shared__ float  s_er[8][9];
    __shared__ FT     s_sf[8][3];

    const int c2_elems = S * S * 8 * n_msg;
    for (int idx = threadIdx.x; idx < c2_elems; idx += 128) smem_c[idx] = __ldg(C2 + idx);
    if constexpr (REM_BF) {
        float* sbf = smem_c + c2_elems;
        for (int idx = threadIdx.x; idx < c2_elems; idx += 128) sbf[idx] = __ldg(Cbf + idx);
    }
    __syncthreads();

    const int sid = threadIdx.x >> 4, slane = threadIdx.x & 15;
    const unsigned smask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int j = blockIdx.x * 8 + sid;
    if (j >= N) return;

    const int z_j = species[j];
    const int eb = row_ptr[j], ee = row_end[j];
    const bool active_q = (slane < n_msg);
    const float f0_j = REM_BF ? __ldg(f0 + j) : 0.0f;
    const float i0_raw = active_q ? __ldg(I0 + (int64_t)j * n_msg + slane) : 0.0f;
    const float* sbf_base = REM_BF ? (smem_c + c2_elems) : nullptr;
    if (slane < 3) s_sf[sid][slane] = (FT)0;

    float acc_di0 = 0.0f, acc_df0 = 0.0f;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += 8) {
        const int bcnt = (ee - eb_b < 8) ? (ee - eb_b) : 8;
        if (slane < bcnt) {
            const int e = eb_b + slane;
            const float4 vl_rev = __ldg(vec_len + e);
            const int i_nb = __ldg(senders + e);
            const float r = fmaxf(vl_rev.w, 1e-8f);
            s_vl[sid][slane] = make_float4(-vl_rev.x, -vl_rev.y, -vl_rev.z, vl_rev.w);
            s_i[sid][slane] = i_nb;
            s_pair[sid][slane] = __ldg(species + i_nb) * S + z_j;
            float4 g03, g47, dg03, dg47;
            eval_cheb8_f4<true>(r, r_max, g03, g47, dg03, dg47);
            s_g03[sid][slane] = g03;   s_g47[sid][slane] = g47;
            s_dg03[sid][slane] = dg03; s_dg47[sid][slane] = dg47;
        }
        __syncwarp(smask);

        for (int bi = 0; bi < bcnt; ++bi) {
            float er_lane = 0.0f;
            if (active_q) {
                const int i_nb = s_i[sid][bi];
                const int pair = s_pair[sid][bi];
                const float4 g03 = s_g03[sid][bi], g47 = s_g47[sid][bi];
                const float4 dg03 = s_dg03[sid][bi], dg47 = s_dg47[sid][bi];
                const float* cp = smem_c + (pair * 8) * n_msg + slane;
                const float c0 = cp[0], c1 = cp[n_msg], c2 = cp[2*n_msg], c3 = cp[3*n_msg];
                const float c4 = cp[4*n_msg], c5 = cp[5*n_msg], c6 = cp[6*n_msg], c7 = cp[7*n_msg];
                const float r2  = fmaf(c7, g47.w, fmaf(c6, g47.z, fmaf(c5, g47.y, fmaf(c4, g47.x, fmaf(c3, g03.w, fmaf(c2, g03.z, fmaf(c1, g03.y, c0 * g03.x)))))));
                const float dr2 = fmaf(c7, dg47.w, fmaf(c6, dg47.z, fmaf(c5, dg47.y, fmaf(c4, dg47.x, fmaf(c3, dg03.w, fmaf(c2, dg03.z, fmaf(c1, dg03.y, c0 * dg03.x)))))));
                const float g = __ldg(dA2_0 + (int64_t)i_nb * n_msg + slane);
                const float di0 = g * r2;
                acc_di0 += di0;
                if constexpr (REM_BF) {
                    const float* bp = sbf_base + (pair * 8) * n_msg + slane;
                    const float b0 = bp[0], b1 = bp[n_msg], b2 = bp[2*n_msg], b3 = bp[3*n_msg];
                    const float b4 = bp[4*n_msg], b5 = bp[5*n_msg], b6 = bp[6*n_msg], b7 = bp[7*n_msg];
                    const float rbf  = fmaf(b7, g47.w, fmaf(b6, g47.z, fmaf(b5, g47.y, fmaf(b4, g47.x, fmaf(b3, g03.w, fmaf(b2, g03.z, fmaf(b1, g03.y, b0 * g03.x)))))));
                    const float drbf = fmaf(b7, dg47.w, fmaf(b6, dg47.z, fmaf(b5, dg47.y, fmaf(b4, dg47.x, fmaf(b3, dg03.w, fmaf(b2, dg03.z, fmaf(b1, dg03.y, b0 * dg03.x)))))));
                    const float i0 = i0_raw - f0_j * rbf;
                    er_lane = fmaf(g * i0, dr2, -di0 * f0_j * drbf);
                    acc_df0 = fmaf(-di0, rbf, acc_df0);
                } else {
                    er_lane = (g * i0_raw) * dr2;
                }
            }
            const float er = half_warp_sum_f(er_lane, smask);
            if (slane == 0) s_er[sid][bi] = er;
        }
        __syncwarp(smask);

        FT bfx = (FT)0, bfy = (FT)0, bfz = (FT)0;
        if (slane < bcnt) {
            const float4 vl = s_vl[sid][slane];
            const int i_nb = s_i[sid][slane];
            const float sc = s_er[sid][slane] / fmaxf(vl.w, 1e-8f);
            bfx = (FT)(vl.x * sc); bfy = (FT)(vl.y * sc); bfz = (FT)(vl.z * sc);
            if constexpr (F32_F) {
                atomicAdd(forces_f32 + (int64_t)i_nb * 3 + 0, -bfx);
                atomicAdd(forces_f32 + (int64_t)i_nb * 3 + 1, -bfy);
                atomicAdd(forces_f32 + (int64_t)i_nb * 3 + 2, -bfz);
            } else {
                atomicAdd(forces + (int64_t)i_nb * 3 + 0, -bfx);
                atomicAdd(forces + (int64_t)i_nb * 3 + 1, -bfy);
                atomicAdd(forces + (int64_t)i_nb * 3 + 2, -bfz);
            }
            if constexpr (STRESS) {
                v0 += (double)vl.x * (double)bfx;
                v1 += 0.5 * ((double)vl.x * (double)bfy + (double)vl.y * (double)bfx);
                v2 += 0.5 * ((double)vl.x * (double)bfz + (double)vl.z * (double)bfx);
                v3 += (double)vl.y * (double)bfy;
                v4 += 0.5 * ((double)vl.y * (double)bfz + (double)vl.z * (double)bfy);
                v5 += (double)vl.z * (double)bfz;
            }
        }
        if constexpr (F32_F) {
            bfx = half_warp_sum_f(bfx, smask); bfy = half_warp_sum_f(bfy, smask); bfz = half_warp_sum_f(bfz, smask);
        } else {
            bfx = half_warp_sum_d(bfx, smask); bfy = half_warp_sum_d(bfy, smask); bfz = half_warp_sum_d(bfz, smask);
        }
        if (slane == 0) { s_sf[sid][0] += bfx; s_sf[sid][1] += bfy; s_sf[sid][2] += bfz; }
        __syncwarp(smask);
    }

    if (active_q) dI0_out[(int64_t)j * n_msg + slane] = acc_di0;
    if constexpr (REM_BF) {
        const float df0_tot = half_warp_sum_f(acc_df0, smask);
        if (slane == 0) df0_bf[j] = df0_tot;
    }
    if constexpr (STRESS) {
        v0 = half_warp_sum_d(v0, smask); v1 = half_warp_sum_d(v1, smask); v2 = half_warp_sum_d(v2, smask);
        v3 = half_warp_sum_d(v3, smask); v4 = half_warp_sum_d(v4, smask); v5 = half_warp_sum_d(v5, smask);
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

// =================================================================================================
// PATH B2 (Unified Vector & Quadrupole L2 2-Hop: lmsg==1, lfeat2==1, lout2 in {1,2}, n_msg<=8,
//          NB2 in {8,12,14,16}, SUBWARP=8 -> 16 atoms/CTA!)
// v6 upgrades:
//  - Unified <int NB2, bool L2_QUAD, bool REM_BF> supporting BOTH vec8 (lout2=1) and sp_n8 (lout2=2)
//    at NB2 in {8, 12, 14, 16} with bank-conflict-free float4 LDS.128 smem_c4!
//  - Fused ilmlip_v1_n8_inv_fwd_kernel<L2_QUAD> & ilmlip_v1_n8_inv_gate_bwd_pack_kernel<L2_QUAD>
//    (fuses W_G[1] + W_G[2] + L=1,2 Gram + eA_x + f1 gate bwd + traceless 3x3 W pack -> 7 kernels -> 2!)
//  - Zero-spill __launch_bounds__(128, 5) sender-owned zero-atomic L2 backward!
// =================================================================================================
__device__ __forceinline__ float v1_dot4(float4 c, float4 g) {
    return fmaf(c.x, g.x, fmaf(c.y, g.y, fmaf(c.z, g.z, c.w * g.w)));
}

template <int NB, bool WITH_DERIV>
__device__ __forceinline__ void eval_cheb_nb_f4(
    float r_f, ChebParam cp, float4 g[(NB + 3) / 4], float4 dg[(NB + 3) / 4]) {
    constexpr int NB4 = (NB + 3) / 4;
    if constexpr (NB == 8) {
        eval_cheb8_f4<WITH_DERIV>(r_f, cp, g[0], g[1], dg[0], dg[1]);
        return;
    }
    float t = cheb_t(r_f, cp);
    if (t < 0.0f) t = 0.0f;
    if (t >= 1.0f) {
        #pragma unroll
        for (int k = 0; k < NB4; ++k) {
            g[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if constexpr (WITH_DERIV) dg[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }
    const float inv_rmax = cp.q0;
    const float t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
    const float env = 1.0f - 21.0f * t5 + 35.0f * t5 * t - 15.0f * t5 * t2;
    const float omt = 1.0f - t;
    const float denv_dt = WITH_DERIV ? (-105.0f * t4 * omt * omt) : 0.0f;
    float u = t;
    float dx_dt = 2.0f;
    if (cp.warp > 0.0f) {
        const float one_p_a = 1.0f + cp.warp;
        const float inv_d = 1.0f / fmaf(cp.warp, t, 1.0f);
        u = (one_p_a * t) * inv_d;
        if constexpr (WITH_DERIV) dx_dt = (2.0f * one_p_a) * (inv_d * inv_d);
    }
    const float env_dx = WITH_DERIV ? (dx_dt * env) : 0.0f;
    const float x = fmaf(2.0f, u, -1.0f);
    const float x2 = 2.0f * x;

    float gv[NB4 * 4];
    float dgv[WITH_DERIV ? (NB4 * 4) : 1];
    float T0 = 1.0f, T1 = x;
    gv[0] = env;
    gv[1] = T1 * env;
    float dT0 = 0.0f, dT1 = 1.0f;
    if constexpr (WITH_DERIV) {
        dgv[0] = denv_dt * inv_rmax;
        dgv[1] = fmaf(env_dx, dT1, T1 * denv_dt) * inv_rmax;
    }
    #pragma unroll
    for (int k = 2; k < NB; ++k) {
        const float Tk = fmaf(x2, T1, -T0);
        gv[k] = Tk * env;
        if constexpr (WITH_DERIV) {
            const float dTk = fmaf(x2, dT1, fmaf(2.0f, T1, -dT0));
            dgv[k] = fmaf(env_dx, dTk, Tk * denv_dt) * inv_rmax;
            dT0 = dT1; dT1 = dTk;
        }
        T0 = T1; T1 = Tk;
    }
    #pragma unroll
    for (int k = NB; k < NB4 * 4; ++k) {
        gv[k] = 0.0f;
        if constexpr (WITH_DERIV) dgv[k] = 0.0f;
    }
    #pragma unroll
    for (int b = 0; b < NB4; ++b) {
        g[b] = make_float4(gv[4 * b + 0], gv[4 * b + 1], gv[4 * b + 2], gv[4 * b + 3]);
        if constexpr (WITH_DERIV)
            dg[b] = make_float4(dgv[4 * b + 0], dgv[4 * b + 1], dgv[4 * b + 2], dgv[4 * b + 3]);
    }
}

template <int NB, bool WITH_DERIV>
__device__ __forceinline__ void eval_cheb_nb_store(
    float r_f, ChebParam cp, float4* __restrict__ dst_g, float4* __restrict__ dst_dg) {
    constexpr int NB4 = (NB + 3) / 4;
    if constexpr (NB == 8) {
        float4 g0, g1, dg0, dg1;
        eval_cheb8_f4<WITH_DERIV>(r_f, cp, g0, g1, dg0, dg1);
        dst_g[0] = g0; dst_g[1] = g1;
        if constexpr (WITH_DERIV) { dst_dg[0] = dg0; dst_dg[1] = dg1; }
        return;
    }
    float t = cheb_t(r_f, cp);
    if (t < 0.0f) t = 0.0f;
    if (t >= 1.0f) {
        #pragma unroll
        for (int b = 0; b < NB4; ++b) {
            dst_g[b] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if constexpr (WITH_DERIV) dst_dg[b] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }
    const float inv_rmax = cp.q0;
    const float t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
    const float env = 1.0f - 21.0f * t5 + 35.0f * t5 * t - 15.0f * t5 * t2;
    const float omt = 1.0f - t;
    const float denv_r = WITH_DERIV ? ((-105.0f * t4 * omt * omt) * inv_rmax) : 0.0f;
    float u = t;
    float dx_dt = 2.0f;
    if (cp.warp > 0.0f) {
        const float one_p_a = 1.0f + cp.warp;
        const float inv_d = 1.0f / fmaf(cp.warp, t, 1.0f);
        u = (one_p_a * t) * inv_d;
        if constexpr (WITH_DERIV) dx_dt = (2.0f * one_p_a) * (inv_d * inv_d);
    }
    const float env2_r = WITH_DERIV ? ((dx_dt * env) * inv_rmax) : 0.0f;
    const float x = fmaf(2.0f, u, -1.0f);
    const float x2 = 2.0f * x;

    float T0 = 1.0f, T1 = x;
    float dT0 = 0.0f, dT1 = 1.0f;
    // Block b = 0: k = 0, 1, 2, 3
    const float g0 = env, g1 = T1 * env;
    const float dg0 = WITH_DERIV ? denv_r : 0.0f;
    const float dg1 = WITH_DERIV ? fmaf(env2_r, 1.0f, T1 * denv_r) : 0.0f;
    const float T2 = fmaf(x2, T1, -T0);
    const float dT2 = WITH_DERIV ? fmaf(x2, dT1, fmaf(2.0f, T1, -dT0)) : 0.0f;
    const float g2 = T2 * env;
    const float dg2 = WITH_DERIV ? fmaf(env2_r, dT2, T2 * denv_r) : 0.0f;
    T0 = T1; T1 = T2; if constexpr (WITH_DERIV) { dT0 = dT1; dT1 = dT2; }
    const float T3 = fmaf(x2, T1, -T0);
    const float dT3 = WITH_DERIV ? fmaf(x2, dT1, fmaf(2.0f, T1, -dT0)) : 0.0f;
    const float g3 = T3 * env;
    const float dg3 = WITH_DERIV ? fmaf(env2_r, dT3, T3 * denv_r) : 0.0f;
    T0 = T1; T1 = T3; if constexpr (WITH_DERIV) { dT0 = dT1; dT1 = dT3; }
    dst_g[0] = make_float4(g0, g1, g2, g3);
    if constexpr (WITH_DERIV) dst_dg[0] = make_float4(dg0, dg1, dg2, dg3);

    #pragma unroll 1
    for (int b = 1; b < NB4; ++b) {
        const int k0 = b * 4;
        float vg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float vdg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (k0 + s < NB) {
                const float Tk = fmaf(x2, T1, -T0);
                vg[s] = Tk * env;
                if constexpr (WITH_DERIV) {
                    const float dTk = fmaf(x2, dT1, fmaf(2.0f, T1, -dT0));
                    vdg[s] = fmaf(env2_r, dTk, Tk * denv_r);
                    dT0 = dT1; dT1 = dTk;
                }
                T0 = T1; T1 = Tk;
            }
        }
        dst_g[b] = make_float4(vg[0], vg[1], vg[2], vg[3]);
        if constexpr (WITH_DERIV) dst_dg[b] = make_float4(vdg[0], vdg[1], vdg[2], vdg[3]);
    }
}

template <int NB2, int NB1, bool SAME_G, bool L2_QUAD, bool REM_BF>
__global__ void __launch_bounds__(128, SAME_G ? 5 : 4) ilmlip_l2_edge_fwd_v1_n8_kernel(
    int N, int S, int n_msg, ChebParam r_max2, ChebParam r_max1, float a0_over_sig1, float inv_DA2_sig1_kap_sq,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2, const float* __restrict__ Cbf,
    const float* __restrict__ f0, const float4* __restrict__ I_pack4, CPtr4 I_node,
    Ptr4 A2, float* __restrict__ f1_out) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int NB1_4 = (NB1 + 3) / 4;
    constexpr int EB = SAME_G ? 8 : 4;
    constexpr bool DUAL_G = REM_BF && !SAME_G;
    extern __shared__ float4 smem_c4[];
    __shared__ float4 s_g[16][EB][NB2_4];
    __shared__ float4 s_g1[DUAL_G ? 16 : 1][DUAL_G ? EB : 1][DUAL_G ? NB1_4 : 1];
    __shared__ float3 s_Y1[16][EB];
    __shared__ int2   s_jp[16][EB];  // .x = j, .y = pair
    __shared__ float  s_f0[REM_BF ? 16 : 1][REM_BF ? EB : 1];

    // Transpose C2 [pair, NB2, 2*n_msg] -> float4 smem_c4[pair, 2*NB2_4, 8] for zero-conflict LDS.128!
    const int l2_tot_cols = 2 * n_msg;
    const int n_slots = S * S * 2 * 8;
    for (int idx = threadIdx.x; idx < n_slots; idx += 128) {
        const int q = idx & 7;
        const int pl = idx >> 3;
        const int pair = pl >> 1;
        const int l = pl & 1;
        #pragma unroll
        for (int b = 0; b < NB2_4; ++b) {
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (q < n_msg) {
                const int k0 = b * 4;
                const float* src = C2 + (pair * NB2 + k0) * l2_tot_cols + l * n_msg + q;
                v.x = (k0 + 0 < NB2) ? __ldg(src + 0 * l2_tot_cols) : 0.0f;
                v.y = (k0 + 1 < NB2) ? __ldg(src + 1 * l2_tot_cols) : 0.0f;
                v.z = (k0 + 2 < NB2) ? __ldg(src + 2 * l2_tot_cols) : 0.0f;
                v.w = (k0 + 3 < NB2) ? __ldg(src + 3 * l2_tot_cols) : 0.0f;
            }
            smem_c4[(pair * (2 * NB2_4) + l * NB2_4 + b) * 8 + q] = v;
        }
    }
    float4* s_bf4 = smem_c4 + S * S * (2 * NB2_4) * 8;
    if constexpr (REM_BF) {
        for (int idx = threadIdx.x; idx < n_slots; idx += 128) {
            const int q = idx & 7;
            const int pl = idx >> 3;
            const int pair = pl >> 1;
            const int l = pl & 1;
            #pragma unroll
            for (int b = 0; b < NB1_4; ++b) {
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (q < n_msg) {
                    const int k0 = b * 4;
                    const float* src = Cbf + (pair * NB1 + k0) * l2_tot_cols + l * n_msg + q;
                    v.x = (k0 + 0 < NB1) ? __ldg(src + 0 * l2_tot_cols) : 0.0f;
                    v.y = (k0 + 1 < NB1) ? __ldg(src + 1 * l2_tot_cols) : 0.0f;
                    v.z = (k0 + 2 < NB1) ? __ldg(src + 2 * l2_tot_cols) : 0.0f;
                    v.w = (k0 + 3 < NB1) ? __ldg(src + 3 * l2_tot_cols) : 0.0f;
                }
                s_bf4[(pair * (2 * NB1_4) + l * NB1_4 + b) * 8 + q] = v;
            }
        }
    }
    __syncthreads();

    const int sid = threadIdx.x >> 3, slane = threadIdx.x & 7;
    const unsigned qmask = 0xffu << ((sid & 3) * 8);
    const int i = blockIdx.x * 16 + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const bool active_q = (slane < n_msg);
    constexpr float C110   = 0.57735026918962576451f;
    constexpr float C_SQ2  = 0.70710678118654752440f;
    constexpr float C_SQ6  = 0.40824829046386301637f;
    constexpr float C_2SQ6 = 0.81649658092772603273f;

    float a00 = 0.0f, a01 = 0.0f;
    float a10_0 = 0.0f, a10_1 = 0.0f, a10_2 = 0.0f;
    float a11_0 = 0.0f, a11_1 = 0.0f, a11_2 = 0.0f;
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (slane < bcnt) {
            const int e = eb_b + slane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const float r = fmaxf(vl.w, 1e-8f);
            const float sc = 1.7320508075688772935f / r;
            s_jp[sid][slane] = make_int2(j, z_i * S + __ldg(species + j));
            if constexpr (REM_BF) s_f0[sid][slane] = __ldg(f0 + j);
            s_Y1[sid][slane] = make_float3(vl.x * sc, vl.y * sc, vl.z * sc);
            eval_cheb_nb_store<NB2, false>(r, r_max2, s_g[sid][slane], nullptr);
        } else if constexpr (DUAL_G) {
            if (slane >= 4 && slane < 4 + bcnt) {
                const int bi = slane - 4;
                const float4 vl = __ldg(vec_len + eb_b + bi);
                const float r = fmaxf(vl.w, 1e-8f);
                eval_cheb_nb_store<NB1, false>(r, r_max1, s_g1[sid][bi], nullptr);
            }
        }
        __syncwarp(qmask);

        if (active_q) {
            for (int bi = 0; bi < bcnt; ++bi) {
                const int2 jp = s_jp[sid][bi];
                const int j = jp.x, pair = jp.y;
                const float3 Y1 = s_Y1[sid][bi];
                const float4* cp4 = smem_c4 + (pair * (2 * NB2_4)) * 8 + slane;
                float r2_0 = 0.0f, r2_1 = 0.0f;
                #pragma unroll
                for (int b = 0; b < NB2_4; ++b) {
                    const float4 gv = s_g[sid][bi][b];
                    r2_0 += v1_dot4(cp4[b * 8], gv);
                    r2_1 += v1_dot4(cp4[(NB2_4 + b) * 8], gv);
                }

                float i0, i1_0, i1_1, i1_2;
                if (I_pack4) {
                    const float4 ir = __ldg(I_pack4 + (int64_t)j * n_msg + slane);
                    i0 = ir.x; i1_0 = ir.y; i1_1 = ir.z; i1_2 = ir.w;
                } else {
                    i0 = __ldg(I_node.p[0] + (int64_t)j * n_msg + slane);
                    const float* pI1 = I_node.p[1] + (int64_t)j * 3 * n_msg + slane;
                    i1_0 = __ldg(pI1); i1_1 = __ldg(pI1 + n_msg); i1_2 = __ldg(pI1 + 2 * n_msg);
                }
                if constexpr (REM_BF) {
                    const float4* bp4 = s_bf4 + (pair * (2 * NB1_4)) * 8 + slane;
                    float rb0 = 0.0f, rb1 = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < NB1_4; ++b) {
                        const float4 gv = DUAL_G ? s_g1[sid][bi][b] : s_g[sid][bi][b];
                        rb0 += v1_dot4(bp4[b * 8], gv);
                        rb1 += v1_dot4(bp4[(NB1_4 + b) * 8], gv);
                    }
                    const float f0j = s_f0[sid][bi];
                    i0 -= f0j * rb0;
                    const float b1s = f0j * rb1;
                    i1_0 -= b1s * Y1.x; i1_1 -= b1s * Y1.y; i1_2 -= b1s * Y1.z;
                }
                const float p1_0 = r2_1 * Y1.x, p1_1 = r2_1 * Y1.y, p1_2 = r2_1 * Y1.z;
                a00 = fmaf(r2_0, i0, a00);
                a01 = fmaf(C110, p1_0 * i1_0 + p1_1 * i1_1 + p1_2 * i1_2, a01);
                a10_0 = fmaf(r2_0, i1_0, a10_0); a10_1 = fmaf(r2_0, i1_1, a10_1); a10_2 = fmaf(r2_0, i1_2, a10_2);
                a11_0 = fmaf(p1_0, i0, a11_0);   a11_1 = fmaf(p1_1, i0, a11_1);   a11_2 = fmaf(p1_2, i0, a11_2);
                if constexpr (L2_QUAD) {
                    a2_0 = fmaf(C_SQ2, p1_0 * i1_2 + p1_2 * i1_0, a2_0);
                    a2_1 = fmaf(C_SQ2, p1_0 * i1_1 + p1_1 * i1_0, a2_1);
                    a2_2 += (-C_SQ6) * p1_0 * i1_0 + C_2SQ6 * p1_1 * i1_1 - C_SQ6 * p1_2 * i1_2;
                    a2_3 = fmaf(C_SQ2, p1_1 * i1_2 + p1_2 * i1_1, a2_3);
                    a2_4 = fmaf(C_SQ2, p1_2 * i1_2 - p1_0 * i1_0, a2_4);
                }
            }
        }
        __syncwarp(qmask);
    }

    float sq_l = 0.0f;
    if (active_q) {
        sq_l = a00*a00 + a01*a01 + a10_0*a10_0 + a10_1*a10_1 + a10_2*a10_2 + a11_0*a11_0 + a11_1*a11_1 + a11_2*a11_2;
        if constexpr (L2_QUAD) {
            sq_l += a2_0*a2_0 + a2_1*a2_1 + a2_2*a2_2 + a2_3*a2_3 + a2_4*a2_4;
        }
    }
    const float sq_tot = qwarp_sum_f(sq_l, qmask);
    const float f1 = a0_over_sig1 / sqrtf(1.0f + sq_tot * inv_DA2_sig1_kap_sq);
    if (slane == 0) f1_out[i] = f1;
    if (active_q) {
        const int c0 = 2 * n_msg, c1 = 2 * n_msg;
        float* d0 = A2.p[0] + (int64_t)i * c0 + slane;
        d0[0] = a00 * f1; d0[n_msg] = a01 * f1;
        float* d1 = A2.p[1] + (int64_t)i * 3 * c1 + slane;
        d1[0] = a10_0 * f1; d1[c1] = a10_1 * f1; d1[2*c1] = a10_2 * f1;
        d1[n_msg] = a11_0 * f1; d1[c1 + n_msg] = a11_1 * f1; d1[2*c1 + n_msg] = a11_2 * f1;
        if constexpr (L2_QUAD) {
            float* d2 = A2.p[2] + (int64_t)i * 5 * n_msg + slane;
            d2[0] = a2_0 * f1; d2[n_msg] = a2_1 * f1; d2[2*n_msg] = a2_2 * f1;
            d2[3*n_msg] = a2_3 * f1; d2[4*n_msg] = a2_4 * f1;
        }
    }
}

// V7 Register-Only (<1 KB SMEM, 100% occ!) Fused v1_n8/sp_n8 Forward:
// W_G[1] (+ W_G[2]) + L=1(,2) Gram + x norm + eA_x — ZERO sA/sP, ZERO triu_i/triu_j loads!
template <bool L2_QUAD>
__global__ void __launch_bounds__(128, 10) ilmlip_v1_n8_inv_fwd_kernel(
    int N, int n_msg, int r_gram, int n_inv1, int n_feat_pad, Int4 gram_off,
    const int32_t* __restrict__ /*triu_i*/, const int32_t* __restrict__ /*triu_j*/,
    const float* __restrict__ W_G1, const float* __restrict__ W_G2,
    CPtr4 A2, const float* __restrict__ mu_x, const float* __restrict__ inv_std_x,
    float* __restrict__ x, int8_t* __restrict__ eA_x,
    const float* __restrict__ am_l1 = nullptr) {
    __shared__ float sW1[16][9];
    __shared__ float sW2[L2_QUAD ? 8 : 1][L2_QUAD ? 9 : 1];

    const int c1 = 2 * n_msg;
    const int n_wg1 = c1 * r_gram;
    for (int idx = threadIdx.x; idx < n_wg1; idx += 128) {
        const int q = idx / r_gram, r = idx - q * r_gram;
        sW1[q][r] = __ldg(W_G1 + idx);
    }
    if constexpr (L2_QUAD) {
        const int n_wg2 = n_msg * r_gram;
        for (int idx = threadIdx.x; idx < n_wg2; idx += 128) {
            const int q = idx / r_gram, r = idx - q * r_gram;
            sW2[q][r] = __ldg(W_G2 + idx);
        }
    }
    __syncthreads();

    const int sid = threadIdx.x >> 3, slane = threadIdx.x & 7;
    const unsigned qmask = 0xffu << ((sid & 3) * 8);
    const int i = blockIdx.x * 16 + sid;
    if (i >= N) return;

    float* xi = x + (int64_t)i * n_feat_pad;
    float am = 0.0f;
    if (am_l1) {
        if (slane == 0) am = __ldg(am_l1 + i);
    } else {
        for (int f = slane; f < n_inv1; f += 8) {
            const float v = (xi[f] - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
            xi[f] = v;
            am = fmaxf(am, fabsf(v));
        }
    }

    float a10_0 = 0.0f, a10_1 = 0.0f, a10_2 = 0.0f;
    float a11_0 = 0.0f, a11_1 = 0.0f, a11_2 = 0.0f;
    if (slane < n_msg) {
        const float* p0 = A2.p[0] + (int64_t)i * c1 + slane;
        const int f0 = n_inv1 + slane;
        const float v0 = (p0[0] - __ldg(mu_x + f0)) * __ldg(inv_std_x + f0);
        xi[f0] = v0; am = fmaxf(am, fabsf(v0));
        const int f1 = n_inv1 + n_msg + slane;
        const float v1 = (p0[n_msg] - __ldg(mu_x + f1)) * __ldg(inv_std_x + f1);
        xi[f1] = v1; am = fmaxf(am, fabsf(v1));

        const float* p1 = A2.p[1] + (int64_t)i * 3 * c1 + slane;
        a10_0 = p1[0]; a10_1 = p1[c1]; a10_2 = p1[2 * c1];
        a11_0 = p1[n_msg]; a11_1 = p1[c1 + n_msg]; a11_2 = p1[2 * c1 + n_msg];
    }

    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f;
    const bool active_r = (slane < r_gram);
    for (int q = 0; q < n_msg; ++q) {
        const float b00 = __shfl_sync(qmask, a10_0, q, 8);
        const float b01 = __shfl_sync(qmask, a10_1, q, 8);
        const float b02 = __shfl_sync(qmask, a10_2, q, 8);
        const float b10 = __shfl_sync(qmask, a11_0, q, 8);
        const float b11 = __shfl_sync(qmask, a11_1, q, 8);
        const float b12 = __shfl_sync(qmask, a11_2, q, 8);
        if (active_r) {
            const float w0 = sW1[q][slane];
            const float w1 = sW1[n_msg + q][slane];
            p0 = fmaf(b00, w0, fmaf(b10, w1, p0));
            p1 = fmaf(b01, w0, fmaf(b11, w1, p1));
            p2 = fmaf(b02, w0, fmaf(b12, w1, p2));
        }
    }

    constexpr float inv_s1 = 0.57735026918962576451f;
    const int goff1 = gram_off.v[1];
    int row_off = goff1;
    for (int a = 0; a < r_gram; ++a) {
        const float pa0 = __shfl_sync(qmask, p0, a, 8);
        const float pa1 = __shfl_sync(qmask, p1, a, 8);
        const float pa2 = __shfl_sync(qmask, p2, a, 8);
        if (slane >= a && active_r) {
            const float s = fmaf(pa0, p0, fmaf(pa1, p1, pa2 * p2));
            const int f = row_off + (slane - a);
            const float v = (s * inv_s1 - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
            xi[f] = v;
            am = fmaxf(am, fabsf(v));
        }
        row_off += (r_gram - a);
    }

    if constexpr (L2_QUAD) {
        float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
        if (slane < n_msg) {
            const float* p2_g = A2.p[2] + (int64_t)i * 5 * n_msg + slane;
            a2_0 = p2_g[0]; a2_1 = p2_g[n_msg]; a2_2 = p2_g[2 * n_msg];
            a2_3 = p2_g[3 * n_msg]; a2_4 = p2_g[4 * n_msg];
        }
        float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f, q4 = 0.0f;
        for (int q = 0; q < n_msg; ++q) {
            const float c0 = __shfl_sync(qmask, a2_0, q, 8);
            const float c1v = __shfl_sync(qmask, a2_1, q, 8);
            const float c2 = __shfl_sync(qmask, a2_2, q, 8);
            const float c3 = __shfl_sync(qmask, a2_3, q, 8);
            const float c4 = __shfl_sync(qmask, a2_4, q, 8);
            if (active_r) {
                const float w = sW2[q][slane];
                q0 = fmaf(c0, w, q0); q1 = fmaf(c1v, w, q1); q2 = fmaf(c2, w, q2);
                q3 = fmaf(c3, w, q3); q4 = fmaf(c4, w, q4);
            }
        }
        constexpr float inv_s2 = 0.44721359549995793928f;
        int row_off2 = gram_off.v[2];
        for (int a = 0; a < r_gram; ++a) {
            const float qa0 = __shfl_sync(qmask, q0, a, 8);
            const float qa1 = __shfl_sync(qmask, q1, a, 8);
            const float qa2 = __shfl_sync(qmask, q2, a, 8);
            const float qa3 = __shfl_sync(qmask, q3, a, 8);
            const float qa4 = __shfl_sync(qmask, q4, a, 8);
            if (slane >= a && active_r) {
                const float s = fmaf(qa0, q0, fmaf(qa1, q1, fmaf(qa2, q2, fmaf(qa3, q3, qa4 * q4))));
                const int f = row_off2 + (slane - a);
                const float v = (s * inv_s2 - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
                xi[f] = v;
                am = fmaxf(am, fabsf(v));
            }
            row_off2 += (r_gram - a);
        }
    }

    if (eA_x) {
        am = fmaxf(am, __shfl_xor_sync(qmask, am, 4));
        am = fmaxf(am, __shfl_xor_sync(qmask, am, 2));
        am = fmaxf(am, __shfl_xor_sync(qmask, am, 1));
        if (slane == 0) eA_x[i] = ilmlip_exp_from_am(am);
    }
}

// V7 Register-Only (<1 KB SMEM, 100% occ!) Fused v1_n8/sp_n8 Backward:
// Gram bwd + W_G[1](,2)^T + f1 gate bwd + float4 dA2_pack — ZERO sA/sP/sG, ZERO bank conflicts!
template <bool L2_QUAD>
__global__ void __launch_bounds__(128, 8) ilmlip_v1_n8_inv_gate_bwd_pack_kernel(
    int N, int n_msg, int r_gram, int n_inv1, int n_feat_pad, float inv_DA2_a0_kap_sq, Int4 gram_off,
    const int32_t* __restrict__ /*triu_i*/, const int32_t* __restrict__ /*triu_j*/,
    const float* __restrict__ W_G1, const float* __restrict__ W_G2,
    const float* __restrict__ inv_std_x, float* __restrict__ dx,
    CPtr4 A2, const float* __restrict__ f1_arr, float* __restrict__ dA2_pack,
    bool skip_inv1 = false) {
    __shared__ float sW1[16][9];
    __shared__ float sW2[L2_QUAD ? 8 : 1][L2_QUAD ? 9 : 1];

    const int c1 = 2 * n_msg;
    const int n_wg1 = c1 * r_gram;
    for (int idx = threadIdx.x; idx < n_wg1; idx += 128) {
        const int q = idx / r_gram, r = idx - q * r_gram;
        sW1[q][r] = __ldg(W_G1 + idx);
    }
    if constexpr (L2_QUAD) {
        const int n_wg2 = n_msg * r_gram;
        for (int idx = threadIdx.x; idx < n_wg2; idx += 128) {
            const int q = idx / r_gram, r = idx - q * r_gram;
            sW2[q][r] = __ldg(W_G2 + idx);
        }
    }
    __syncthreads();

    const int sid = threadIdx.x >> 3, slane = threadIdx.x & 7;
    const unsigned qmask = 0xffu << ((sid & 3) * 8);
    const int i = blockIdx.x * 16 + sid;
    if (i >= N) return;

    float* dxi = dx + (int64_t)i * n_feat_pad;
    if (!skip_inv1) {
        for (int f = slane; f < n_inv1; f += 8) dxi[f] *= __ldg(inv_std_x + f);
    }

    const bool active_q = (slane < n_msg);
    const bool active_r = (slane < r_gram);
    float a00 = 0.0f, a01 = 0.0f, d00 = 0.0f, d01 = 0.0f;
    float a10_0 = 0.0f, a10_1 = 0.0f, a10_2 = 0.0f, a11_0 = 0.0f, a11_1 = 0.0f, a11_2 = 0.0f;
    if (active_q) {
        const float* p0 = A2.p[0] + (int64_t)i * c1 + slane;
        a00 = p0[0]; a01 = p0[n_msg];
        d00 = dxi[n_inv1 + slane] * __ldg(inv_std_x + n_inv1 + slane);
        d01 = dxi[n_inv1 + n_msg + slane] * __ldg(inv_std_x + n_inv1 + n_msg + slane);
        const float* p1 = A2.p[1] + (int64_t)i * 3 * c1 + slane;
        a10_0 = p1[0]; a10_1 = p1[c1]; a10_2 = p1[2 * c1];
        a11_0 = p1[n_msg]; a11_1 = p1[c1 + n_msg]; a11_2 = p1[2 * c1 + n_msg];
    }

    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f;
    for (int q = 0; q < n_msg; ++q) {
        const float b00 = __shfl_sync(qmask, a10_0, q, 8);
        const float b01 = __shfl_sync(qmask, a10_1, q, 8);
        const float b02 = __shfl_sync(qmask, a10_2, q, 8);
        const float b10 = __shfl_sync(qmask, a11_0, q, 8);
        const float b11 = __shfl_sync(qmask, a11_1, q, 8);
        const float b12 = __shfl_sync(qmask, a11_2, q, 8);
        if (active_r) {
            const float w0 = sW1[q][slane];
            const float w1 = sW1[n_msg + q][slane];
            p0 = fmaf(b00, w0, fmaf(b10, w1, p0));
            p1 = fmaf(b01, w0, fmaf(b11, w1, p1));
            p2 = fmaf(b02, w0, fmaf(b12, w1, p2));
        }
    }

    float dp0 = 0.0f, dp1 = 0.0f, dp2 = 0.0f;
    {
        constexpr float inv_s1 = 0.57735026918962576451f;
        const int goff1 = gram_off.v[1];
        for (int a = 0; a < r_gram; ++a) {
            const float pa0 = __shfl_sync(qmask, p0, a, 8);
            const float pa1 = __shfl_sync(qmask, p1, a, 8);
            const float pa2 = __shfl_sync(qmask, p2, a, 8);
            if (active_r) {
                const int lo = (a < slane) ? a : slane;
                const int hi = (a < slane) ? slane : a;
                const int f = goff1 + lo * r_gram - ((lo * (lo - 1)) >> 1) + (hi - lo);
                float g = dxi[f] * __ldg(inv_std_x + f) * inv_s1;
                if (a == slane) g *= 2.0f;
                dp0 = fmaf(g, pa0, dp0);
                dp1 = fmaf(g, pa1, dp1);
                dp2 = fmaf(g, pa2, dp2);
            }
        }
    }

    float d10_0 = 0.0f, d10_1 = 0.0f, d10_2 = 0.0f, d11_0 = 0.0f, d11_1 = 0.0f, d11_2 = 0.0f;
    for (int r = 0; r < r_gram; ++r) {
        const float r0 = __shfl_sync(qmask, dp0, r, 8);
        const float r1 = __shfl_sync(qmask, dp1, r, 8);
        const float r2 = __shfl_sync(qmask, dp2, r, 8);
        if (active_q) {
            const float w0 = sW1[slane][r];
            const float w1 = sW1[n_msg + slane][r];
            d10_0 = fmaf(r0, w0, d10_0); d10_1 = fmaf(r1, w0, d10_1); d10_2 = fmaf(r2, w0, d10_2);
            d11_0 = fmaf(r0, w1, d11_0); d11_1 = fmaf(r1, w1, d11_1); d11_2 = fmaf(r2, w1, d11_2);
        }
    }
    float dot_l = 0.0f;
    if (active_q) {
        dot_l = a00*d00 + a01*d01 + a10_0*d10_0 + a10_1*d10_1 + a10_2*d10_2 + a11_0*d11_0 + a11_1*d11_1 + a11_2*d11_2;
    }

    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    float d2_0 = 0.0f, d2_1 = 0.0f, d2_2 = 0.0f, d2_3 = 0.0f, d2_4 = 0.0f;
    if constexpr (L2_QUAD) {
        if (active_q) {
            const float* p2_g = A2.p[2] + (int64_t)i * 5 * n_msg + slane;
            a2_0 = p2_g[0]; a2_1 = p2_g[n_msg]; a2_2 = p2_g[2 * n_msg];
            a2_3 = p2_g[3 * n_msg]; a2_4 = p2_g[4 * n_msg];
        }
        float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f, q4 = 0.0f;
        for (int q = 0; q < n_msg; ++q) {
            const float c0 = __shfl_sync(qmask, a2_0, q, 8);
            const float c1v = __shfl_sync(qmask, a2_1, q, 8);
            const float c2 = __shfl_sync(qmask, a2_2, q, 8);
            const float c3 = __shfl_sync(qmask, a2_3, q, 8);
            const float c4 = __shfl_sync(qmask, a2_4, q, 8);
            if (active_r) {
                const float w = sW2[q][slane];
                q0 = fmaf(c0, w, q0); q1 = fmaf(c1v, w, q1); q2 = fmaf(c2, w, q2);
                q3 = fmaf(c3, w, q3); q4 = fmaf(c4, w, q4);
            }
        }
        float dq0 = 0.0f, dq1 = 0.0f, dq2 = 0.0f, dq3 = 0.0f, dq4 = 0.0f;
        {
            constexpr float inv_s2 = 0.44721359549995793928f;
            const int goff2 = gram_off.v[2];
            for (int a = 0; a < r_gram; ++a) {
                const float qa0 = __shfl_sync(qmask, q0, a, 8);
                const float qa1 = __shfl_sync(qmask, q1, a, 8);
                const float qa2 = __shfl_sync(qmask, q2, a, 8);
                const float qa3 = __shfl_sync(qmask, q3, a, 8);
                const float qa4 = __shfl_sync(qmask, q4, a, 8);
                if (active_r) {
                    const int lo = (a < slane) ? a : slane;
                    const int hi = (a < slane) ? slane : a;
                    const int f = goff2 + lo * r_gram - ((lo * (lo - 1)) >> 1) + (hi - lo);
                    float g = dxi[f] * __ldg(inv_std_x + f) * inv_s2;
                    if (a == slane) g *= 2.0f;
                    dq0 = fmaf(g, qa0, dq0); dq1 = fmaf(g, qa1, dq1); dq2 = fmaf(g, qa2, dq2);
                    dq3 = fmaf(g, qa3, dq3); dq4 = fmaf(g, qa4, dq4);
                }
            }
        }
        for (int r = 0; r < r_gram; ++r) {
            const float r0 = __shfl_sync(qmask, dq0, r, 8);
            const float r1 = __shfl_sync(qmask, dq1, r, 8);
            const float r2 = __shfl_sync(qmask, dq2, r, 8);
            const float r3 = __shfl_sync(qmask, dq3, r, 8);
            const float r4 = __shfl_sync(qmask, dq4, r, 8);
            if (active_q) {
                const float w = sW2[slane][r];
                d2_0 = fmaf(r0, w, d2_0); d2_1 = fmaf(r1, w, d2_1); d2_2 = fmaf(r2, w, d2_2);
                d2_3 = fmaf(r3, w, d2_3); d2_4 = fmaf(r4, w, d2_4);
            }
        }
        if (active_q) {
            dot_l += a2_0*d2_0 + a2_1*d2_1 + a2_2*d2_2 + a2_3*d2_3 + a2_4*d2_4;
        }
    }

    const float dot = qwarp_sum_f(dot_l, qmask);
    const float f1 = f1_arr[i];
    const float sc = (dot * f1) * inv_DA2_a0_kap_sq;
    if (active_q) {
        constexpr float C110 = 0.57735026918962576451f;
        const int64_t atom_stride = L2_QUAD ? 128 : 64;
        float* base = dA2_pack + (int64_t)i * atom_stride;
        float4* dst4 = reinterpret_cast<float4*>(base) + slane;
        dst4[0] = make_float4(fmaf(f1, d00, -sc * a00), C110 * fmaf(f1, d01, -sc * a01),
                              fmaf(f1, d10_0, -sc * a10_0), fmaf(f1, d10_1, -sc * a10_1));
        dst4[8] = make_float4(fmaf(f1, d10_2, -sc * a10_2), fmaf(f1, d11_0, -sc * a11_0),
                              fmaf(f1, d11_1, -sc * a11_1), fmaf(f1, d11_2, -sc * a11_2));
        if constexpr (L2_QUAD) {
            constexpr float C_SQ2 = 0.70710678118654752440f;
            constexpr float C_SQ6 = 0.40824829046386301637f;
            const float g8  = fmaf(f1, d2_0, -sc * a2_0);
            const float g9  = fmaf(f1, d2_1, -sc * a2_1);
            const float g10 = fmaf(f1, d2_2, -sc * a2_2);
            const float g11 = fmaf(f1, d2_3, -sc * a2_3);
            const float g12 = fmaf(f1, d2_4, -sc * a2_4);
            const float w00 = -C_SQ6 * g10 - C_SQ2 * g12;
            const float w01 =  C_SQ2 * g9;
            const float w02 =  C_SQ2 * g8;
            const float w12 =  C_SQ2 * g11;
            const float w22 = -C_SQ6 * g10 + C_SQ2 * g12;
            dst4[16] = make_float4(w00, w01, w02, w12);
            base[96 + slane] = w22;
        }
    }
}

template <bool L2_QUAD>
__global__ void __launch_bounds__(128) ilmlip_gate_bwd_pack_l2_v1_n8_kernel(
    int N, int n_msg, float inv_DA2_a0_kap_sq,
    CPtr4 A2, CPtr4 dA2, const float* __restrict__ f1_arr, float* __restrict__ dA2_pack) {
    const int sid = threadIdx.x >> 3, slane = threadIdx.x & 7;
    const unsigned qmask = 0xffu << ((sid & 3) * 8);
    const int i = blockIdx.x * 16 + sid;
    if (i >= N) return;
    const bool active_q = (slane < n_msg);
    float a00=0,a01=0,a10_0=0,a10_1=0,a10_2=0,a11_0=0,a11_1=0,a11_2=0;
    float d00=0,d01=0,d10_0=0,d10_1=0,d10_2=0,d11_0=0,d11_1=0,d11_2=0;
    float a2_0=0,a2_1=0,a2_2=0,a2_3=0,a2_4=0,d2_0=0,d2_1=0,d2_2=0,d2_3=0,d2_4=0;
    float dot_l = 0.0f;
    if (active_q) {
        const int c = 2 * n_msg;
        const float* pA0 = A2.p[0] + (int64_t)i * c + slane;
        const float* pd0 = dA2.p[0] + (int64_t)i * c + slane;
        const float* pA1 = A2.p[1] + (int64_t)i * 3 * c + slane;
        const float* pd1 = dA2.p[1] + (int64_t)i * 3 * c + slane;
        a00 = pA0[0]; a01 = pA0[n_msg]; d00 = pd0[0]; d01 = pd0[n_msg];
        a10_0 = pA1[0]; a10_1 = pA1[c]; a10_2 = pA1[2*c];
        d10_0 = pd1[0]; d10_1 = pd1[c]; d10_2 = pd1[2*c];
        a11_0 = pA1[n_msg]; a11_1 = pA1[c + n_msg]; a11_2 = pA1[2*c + n_msg];
        d11_0 = pd1[n_msg]; d11_1 = pd1[c + n_msg]; d11_2 = pd1[2*c + n_msg];
        dot_l = a00*d00 + a01*d01 + a10_0*d10_0 + a10_1*d10_1 + a10_2*d10_2 + a11_0*d11_0 + a11_1*d11_1 + a11_2*d11_2;
        if constexpr (L2_QUAD) {
            const float* pA2 = A2.p[2] + (int64_t)i * 5 * n_msg + slane;
            const float* pd2 = dA2.p[2] + (int64_t)i * 5 * n_msg + slane;
            a2_0 = pA2[0]; a2_1 = pA2[n_msg]; a2_2 = pA2[2*n_msg]; a2_3 = pA2[3*n_msg]; a2_4 = pA2[4*n_msg];
            d2_0 = pd2[0]; d2_1 = pd2[n_msg]; d2_2 = pd2[2*n_msg]; d2_3 = pd2[3*n_msg]; d2_4 = pd2[4*n_msg];
            dot_l += a2_0*d2_0 + a2_1*d2_1 + a2_2*d2_2 + a2_3*d2_3 + a2_4*d2_4;
        }
    }
    const float dot = qwarp_sum_f(dot_l, qmask);
    const float f1 = f1_arr[i];
    const float sc = (dot * f1) * inv_DA2_a0_kap_sq;
    if (active_q) {
        constexpr float C110 = 0.57735026918962576451f;
        const int64_t atom_stride = L2_QUAD ? 128 : 64;
        float* base = dA2_pack + (int64_t)i * atom_stride;
        float4* dst4 = reinterpret_cast<float4*>(base) + slane;
        dst4[0] = make_float4(fmaf(f1, d00, -sc * a00), C110 * fmaf(f1, d01, -sc * a01),
                              fmaf(f1, d10_0, -sc * a10_0), fmaf(f1, d10_1, -sc * a10_1));
        dst4[8] = make_float4(fmaf(f1, d10_2, -sc * a10_2), fmaf(f1, d11_0, -sc * a11_0),
                              fmaf(f1, d11_1, -sc * a11_1), fmaf(f1, d11_2, -sc * a11_2));
        if constexpr (L2_QUAD) {
            constexpr float C_SQ2 = 0.70710678118654752440f;
            constexpr float C_SQ6 = 0.40824829046386301637f;
            const float g8  = fmaf(f1, d2_0, -sc * a2_0);
            const float g9  = fmaf(f1, d2_1, -sc * a2_1);
            const float g10 = fmaf(f1, d2_2, -sc * a2_2);
            const float g11 = fmaf(f1, d2_3, -sc * a2_3);
            const float g12 = fmaf(f1, d2_4, -sc * a2_4);
            dst4[16] = make_float4(-C_SQ6 * g10 - C_SQ2 * g12, C_SQ2 * g9, C_SQ2 * g8, C_SQ2 * g11);
            base[96 + slane] = -C_SQ6 * g10 + C_SQ2 * g12;
        }
    }
}

template <int NB2, int NB1, bool SAME_G, bool L2_QUAD, bool REM_BF, bool STRESS, bool F32_F>
__global__ void __launch_bounds__(128, SAME_G ? 5 : 4) ilmlip_l2_edge_bwd_v1_n8_sender_kernel(
    int N, int S, int n_msg, ChebParam r_max2, ChebParam r_max1,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2, const float* __restrict__ Cbf,
    const float* __restrict__ f0, const float4* __restrict__ I_pack4, CPtr4 I_node, const float* __restrict__ dA2_pack,
    Ptr4 dI_node, float* __restrict__ df0_bf,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    using FT = typename std::conditional<F32_F, float, double>::type;
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int NB1_4 = (NB1 + 3) / 4;
    constexpr int EB = SAME_G ? 8 : 4;
    constexpr bool DUAL_G = REM_BF && !SAME_G;
    extern __shared__ float4 smem_c4[];
    __shared__ float4 s_g[16][EB][NB2_4];
    __shared__ float4 s_dg[16][EB][NB2_4];  // s_dg[sid][bi][0] overwritten after read with (er, dY1, dY2, dY3)!
    __shared__ float4 s_g1[DUAL_G ? 16 : 1][DUAL_G ? EB : 1][DUAL_G ? NB1_4 : 1];
    __shared__ float4 s_dg1[DUAL_G ? 16 : 1][DUAL_G ? EB : 1][DUAL_G ? NB1_4 : 1];
    __shared__ float4 s_Y1[16][EB];       // (.x,.y,.z) = S3*u_fwd, .w = inv_r (zero 2nd 1/r or vec_len load!)
    __shared__ int2   s_ip[16][EB];       // .x = i_nb, .y = pair
    __shared__ FT     s_sf[16][3];

    const int l2_tot_cols = 2 * n_msg;
    const int n_slots = S * S * 2 * 8;
    for (int idx = threadIdx.x; idx < n_slots; idx += 128) {
        const int q = idx & 7;
        const int pl = idx >> 3;
        const int pair = pl >> 1;
        const int l = pl & 1;
        #pragma unroll
        for (int b = 0; b < NB2_4; ++b) {
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (q < n_msg) {
                const int k0 = b * 4;
                const float* src = C2 + (pair * NB2 + k0) * l2_tot_cols + l * n_msg + q;
                v.x = (k0 + 0 < NB2) ? __ldg(src + 0 * l2_tot_cols) : 0.0f;
                v.y = (k0 + 1 < NB2) ? __ldg(src + 1 * l2_tot_cols) : 0.0f;
                v.z = (k0 + 2 < NB2) ? __ldg(src + 2 * l2_tot_cols) : 0.0f;
                v.w = (k0 + 3 < NB2) ? __ldg(src + 3 * l2_tot_cols) : 0.0f;
            }
            smem_c4[(pair * (2 * NB2_4) + l * NB2_4 + b) * 8 + q] = v;
        }
    }
    float4* s_bf4 = smem_c4 + S * S * (2 * NB2_4) * 8;
    if constexpr (REM_BF) {
        for (int idx = threadIdx.x; idx < n_slots; idx += 128) {
            const int q = idx & 7;
            const int pl = idx >> 3;
            const int pair = pl >> 1;
            const int l = pl & 1;
            #pragma unroll
            for (int b = 0; b < NB1_4; ++b) {
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (q < n_msg) {
                    const int k0 = b * 4;
                    const float* src = Cbf + (pair * NB1 + k0) * l2_tot_cols + l * n_msg + q;
                    v.x = (k0 + 0 < NB1) ? __ldg(src + 0 * l2_tot_cols) : 0.0f;
                    v.y = (k0 + 1 < NB1) ? __ldg(src + 1 * l2_tot_cols) : 0.0f;
                    v.z = (k0 + 2 < NB1) ? __ldg(src + 2 * l2_tot_cols) : 0.0f;
                    v.w = (k0 + 3 < NB1) ? __ldg(src + 3 * l2_tot_cols) : 0.0f;
                }
                s_bf4[(pair * (2 * NB1_4) + l * NB1_4 + b) * 8 + q] = v;
            }
        }
    }
    __syncthreads();

    const int sid = threadIdx.x >> 3, slane = threadIdx.x & 7;
    const unsigned qmask = 0xffu << ((sid & 3) * 8);
    const int j = blockIdx.x * 16 + sid;
    if (j >= N) return;

    const int z_j = species[j];
    const int eb = row_ptr[j], ee = row_end[j];
    const bool active_q = (slane < n_msg);
    const float f0_j = REM_BF ? __ldg(f0 + j) : 0.0f;
    float i0_raw = 0.0f, i1_0_raw = 0.0f, i1_1_raw = 0.0f, i1_2_raw = 0.0f;
    if (active_q) {
        if (I_pack4) {
            const float4 ir = __ldg(I_pack4 + (int64_t)j * n_msg + slane);
            i0_raw = ir.x; i1_0_raw = ir.y; i1_1_raw = ir.z; i1_2_raw = ir.w;
        } else {
            i0_raw = __ldg(I_node.p[0] + (int64_t)j * n_msg + slane);
            const float* pI1 = I_node.p[1] + (int64_t)j * 3 * n_msg + slane;
            i1_0_raw = __ldg(pI1); i1_1_raw = __ldg(pI1 + n_msg); i1_2_raw = __ldg(pI1 + 2 * n_msg);
        }
    }
    if (slane < 3) s_sf[sid][slane] = (FT)0;

    float acc_di0 = 0.0f, acc_di1_0 = 0.0f, acc_di1_1 = 0.0f, acc_di1_2 = 0.0f, acc_df0 = 0.0f;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (slane < bcnt) {
            const int e = eb_b + slane;
            const float4 vl_rev = __ldg(vec_len + e);
            const int i_nb = __ldg(senders + e);
            const float r = fmaxf(vl_rev.w, 1e-8f);
            const float inv_d = 1.0f / r;
            const float sc = -1.7320508075688772935f * inv_d;
            s_ip[sid][slane] = make_int2(i_nb, __ldg(species + i_nb) * S + z_j);
            s_Y1[sid][slane] = make_float4(vl_rev.x * sc, vl_rev.y * sc, vl_rev.z * sc, inv_d);
            eval_cheb_nb_store<NB2, true>(r, r_max2, s_g[sid][slane], s_dg[sid][slane]);
        } else if constexpr (DUAL_G) {
            if (slane >= 4 && slane < 4 + bcnt) {
                const int bi = slane - 4;
                const float4 vl_rev = __ldg(vec_len + eb_b + bi);
                const float r = fmaxf(vl_rev.w, 1e-8f);
                eval_cheb_nb_store<NB1, true>(r, r_max1, s_g1[sid][bi], s_dg1[sid][bi]);
            }
        }
        __syncwarp(qmask);

        for (int bi = 0; bi < bcnt; ++bi) {
            float er_lane = 0.0f, dY1 = 0.0f, dY2 = 0.0f, dY3 = 0.0f;
            if (active_q) {
                const int2 ip = s_ip[sid][bi];
                const int i_nb = ip.x, pair = ip.y;
                const float4 Y1 = s_Y1[sid][bi];
                const float4* cp4 = smem_c4 + (pair * (2 * NB2_4)) * 8 + slane;
                float r2_0 = 0.0f, dr2_0 = 0.0f, r2_1 = 0.0f, dr2_1 = 0.0f;
                #pragma unroll
                for (int b = 0; b < NB2_4; ++b) {
                    const float4 gv = s_g[sid][bi][b], dgv = s_dg[sid][bi][b];
                    const float4 c0 = cp4[b * 8];
                    r2_0  += v1_dot4(c0, gv);
                    dr2_0 += v1_dot4(c0, dgv);
                    const float4 c1 = cp4[(NB2_4 + b) * 8];
                    r2_1  += v1_dot4(c1, gv);
                    dr2_1 += v1_dot4(c1, dgv);
                }

                float rb0 = 0.0f, rb1 = 0.0f, drb0 = 0.0f, drb1 = 0.0f;
                if constexpr (REM_BF) {
                    const float4* bp4 = s_bf4 + (pair * (2 * NB1_4)) * 8 + slane;
                    #pragma unroll
                    for (int b = 0; b < NB1_4; ++b) {
                        const float4 gv  = DUAL_G ? s_g1[sid][bi][b]  : s_g[sid][bi][b];
                        const float4 dgv = DUAL_G ? s_dg1[sid][bi][b] : s_dg[sid][bi][b];
                        const float4 b0 = bp4[b * 8];
                        rb0  += v1_dot4(b0, gv);
                        drb0 += v1_dot4(b0, dgv);
                        const float4 b1 = bp4[(NB1_4 + b) * 8];
                        rb1  += v1_dot4(b1, gv);
                        drb1 += v1_dot4(b1, dgv);
                    }
                }
                const float i0   = i0_raw   - (REM_BF ? (f0_j * rb0) : 0.0f);
                const float b1s  = REM_BF ? (f0_j * rb1) : 0.0f;
                const float i1_0 = i1_0_raw - b1s * Y1.x;
                const float i1_1 = i1_1_raw - b1s * Y1.y;
                const float i1_2 = i1_2_raw - b1s * Y1.z;

                const int64_t atom_stride = L2_QUAD ? 128 : 64;
                const float* base = dA2_pack + (int64_t)i_nb * atom_stride;
                const float4* p4 = reinterpret_cast<const float4*>(base) + slane;
                const float4 q0 = __ldg(p4);      // (g00, g01*C110, g10_0, g10_1)
                const float4 q1 = __ldg(p4 + 8);  // (g10_2, g11_0, g11_1, g11_2)

                const float dp0 = fmaf(q0.x, i0, q0.z * i1_0 + q0.w * i1_1 + q1.x * i1_2);
                float dp1_0 = fmaf(q0.y, i1_0, q1.y * i0);
                float dp1_1 = fmaf(q0.y, i1_1, q1.z * i0);
                float dp1_2 = fmaf(q0.y, i1_2, q1.w * i0);

                const float di0   = fmaf(q0.x, r2_0, r2_1 * (q1.y * Y1.x + q1.z * Y1.y + q1.w * Y1.z));
                const float g01r1 = q0.y * r2_1;
                float di1_0 = fmaf(q0.z, r2_0, g01r1 * Y1.x);
                float di1_1 = fmaf(q0.w, r2_0, g01r1 * Y1.y);
                float di1_2 = fmaf(q1.x, r2_0, g01r1 * Y1.z);

                if constexpr (L2_QUAD) {
                    const float4 q2 = __ldg(p4 + 16);        // (w00, w01, w02, w12)
                    const float w22 = __ldg(base + 96 + slane);
                    const float w11 = -(q2.x + w22);         // traceless symmetric 3x3!
                    dp1_0 += q2.x * i1_0 + q2.y * i1_1 + q2.z * i1_2;
                    dp1_1 += q2.y * i1_0 + w11  * i1_1 + q2.w * i1_2;
                    dp1_2 += q2.z * i1_0 + q2.w * i1_1 + w22  * i1_2;
                    di1_0 = fmaf(r2_1, q2.x * Y1.x + q2.y * Y1.y + q2.z * Y1.z, di1_0);
                    di1_1 = fmaf(r2_1, q2.y * Y1.x + w11  * Y1.y + q2.w * Y1.z, di1_1);
                    di1_2 = fmaf(r2_1, q2.z * Y1.x + q2.w * Y1.y + w22  * Y1.z, di1_2);
                }

                const float d_r2_1 = dp1_0 * Y1.x + dp1_1 * Y1.y + dp1_2 * Y1.z;
                er_lane = fmaf(dp0, dr2_0, d_r2_1 * dr2_1);
                dY1 = dp1_0 * r2_1; dY2 = dp1_1 * r2_1; dY3 = dp1_2 * r2_1;
                acc_di0 += di0; acc_di1_0 += di1_0; acc_di1_1 += di1_1; acc_di1_2 += di1_2;
                if constexpr (REM_BF) {
                    acc_df0 = fmaf(-di0, rb0, acc_df0);
                    er_lane = fmaf(-di0 * f0_j, drb0, er_lane);
                    const float ydot1 = di1_0 * Y1.x + di1_1 * Y1.y + di1_2 * Y1.z;
                    acc_df0 = fmaf(-ydot1, rb1, acc_df0);
                    er_lane = fmaf(-ydot1 * f0_j, drb1, er_lane);
                    const float bf1s = -f0_j * rb1;
                    dY1 = fmaf(di1_0, bf1s, dY1); dY2 = fmaf(di1_1, bf1s, dY2); dY3 = fmaf(di1_2, bf1s, dY3);
                }
            }
            const bool p0 = (slane & 1) != 0;
            const float u0 = (p0 ? dY1 : er_lane) + __shfl_xor_sync(qmask, p0 ? er_lane : dY1, 1);
            const float u1 = (p0 ? dY3 : dY2)     + __shfl_xor_sync(qmask, p0 ? dY2 : dY3, 1);
            const bool p1 = (slane & 2) != 0;
            float v = (p1 ? u1 : u0) + __shfl_xor_sync(qmask, p1 ? u0 : u1, 2);
            v += __shfl_xor_sync(qmask, v, 4);
            // Reuse s_dg[sid][bi][0] (already dead!) as (er, dY1, dY2, dY3)
            if (slane < 4) reinterpret_cast<float*>(&s_dg[sid][bi][0])[slane] = v;
        }
        __syncwarp(qmask);

        FT bfx = (FT)0, bfy = (FT)0, bfz = (FT)0;
        if (slane < bcnt) {
            const int my_inb = s_ip[sid][slane].x;
            const float4 y1i = s_Y1[sid][slane];
            constexpr float S3 = 1.7320508075688772935f;
            constexpr float INV_S3 = 0.57735026918962576451f;
            const float ux = y1i.x * INV_S3, uy = y1i.y * INV_S3, uz = y1i.z * INV_S3;
            const float inv_d = y1i.w;
            const float4 ed = s_dg[sid][slane][0];
            const float gx = S3 * ed.y, gy = S3 * ed.z, gz = S3 * ed.w;
            const float dot = ux * gx + uy * gy + uz * gz;
            bfx = (FT)(ux * ed.x + (gx - ux * dot) * inv_d);
            bfy = (FT)(uy * ed.x + (gy - uy * dot) * inv_d);
            bfz = (FT)(uz * ed.x + (gz - uz * dot) * inv_d);
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
                const float4 vl_rev = __ldg(vec_len + eb_b + slane);
                const float vx = -vl_rev.x, vy = -vl_rev.y, vz = -vl_rev.z;
                v0 += (double)vx * (double)bfx;
                v1 += 0.5 * ((double)vx * (double)bfy + (double)vy * (double)bfx);
                v2 += 0.5 * ((double)vx * (double)bfz + (double)vz * (double)bfx);
                v3 += (double)vy * (double)bfy;
                v4 += 0.5 * ((double)vy * (double)bfz + (double)vz * (double)bfy);
                v5 += (double)vz * (double)bfz;
            }
        }
        if constexpr (F32_F) {
            bfx = qwarp_sum_f(bfx, qmask); bfy = qwarp_sum_f(bfy, qmask); bfz = qwarp_sum_f(bfz, qmask);
        } else {
            bfx = qwarp_sum_d(bfx, qmask); bfy = qwarp_sum_d(bfy, qmask); bfz = qwarp_sum_d(bfz, qmask);
        }
        if (slane == 0) { s_sf[sid][0] += bfx; s_sf[sid][1] += bfy; s_sf[sid][2] += bfz; }
        __syncwarp(qmask);
    }

    if (active_q) {
        dI_node.p[0][(int64_t)j * n_msg + slane] = acc_di0;
        float* dI1 = dI_node.p[1] + (int64_t)j * 3 * n_msg + slane;
        dI1[0] = acc_di1_0; dI1[n_msg] = acc_di1_1; dI1[2 * n_msg] = acc_di1_2;
    }
    if constexpr (REM_BF) {
        const float df0_tot = qwarp_sum_f(acc_df0, qmask);
        if (slane == 0) df0_bf[j] = df0_tot;
    }
    if constexpr (STRESS) {
        v0 = qwarp_sum_d(v0, qmask); v1 = qwarp_sum_d(v1, qmask); v2 = qwarp_sum_d(v2, qmask);
        v3 = qwarp_sum_d(v3, qmask); v4 = qwarp_sum_d(v4, qmask); v5 = qwarp_sum_d(v5, qmask);
    }
    if (slane == 0) {
        const FT sfx_acc = s_sf[sid][0], sfy_acc = s_sf[sid][1], sfz_acc = s_sf[sid][2];
        if (sfx_acc != (FT)0 || sfy_acc != (FT)0 || sfz_acc != (FT)0) {
            if constexpr (F32_F) {
                atomicAdd(forces_f32 + (int64_t)j * 3 + 0, sfx_acc);
                atomicAdd(forces_f32 + (int64_t)j * 3 + 1, sfy_acc);
                atomicAdd(forces_f32 + (int64_t)j * 3 + 2, sfz_acc);
            } else {
                atomicAdd(forces + (int64_t)j * 3 + 0, sfx_acc);
                atomicAdd(forces + (int64_t)j * 3 + 1, sfy_acc);
                atomicAdd(forces + (int64_t)j * 3 + 2, sfz_acc);
            }
        }
        if constexpr (STRESS) {
            if (v0 != 0.0) atomicAdd(virial + 0, v0); if (v1 != 0.0) atomicAdd(virial + 1, v1);
            if (v2 != 0.0) atomicAdd(virial + 2, v2); if (v3 != 0.0) atomicAdd(virial + 3, v3);
            if (v4 != 0.0) atomicAdd(virial + 4, v4); if (v5 != 0.0) atomicAdd(virial + 5, v5);
        }
    }
}

// =================================================================================================
// V6 ULTRA-FAST FLOAT4 LDS.128 2-ATOMS/WARP L1 NODE + W_MSG (n_msg <= 8, lfeat == 1, lmsg == 1)
// - 8 atoms/CTA ((N+7)/8 CTAs -> 2x more CTAs than v5), transposed float4-aligned sW0T/sW1T!
// - 100% coalesced float4 LDS.128/LDG.128/STG.128: ONLY 2 iters for W_msg0 + x-norm and 1 iter for W_msg1!
// - Parallel 9-shuffle butterfly warp-reduce for all 8 q's at once!
// =================================================================================================
__device__ __forceinline__ float warp_reduce4_to_q(const float a[4], int lane) {
    const bool p0 = (lane & 1) != 0;
    const float u0 = (p0 ? a[1] : a[0]) + __shfl_xor_sync(0xffffffffu, p0 ? a[0] : a[1], 1);
    const float u1 = (p0 ? a[3] : a[2]) + __shfl_xor_sync(0xffffffffu, p0 ? a[2] : a[3], 1);
    const bool p1 = (lane & 2) != 0;
    float res = (p1 ? u1 : u0) + __shfl_xor_sync(0xffffffffu, p1 ? u0 : u1, 2);
    res += __shfl_xor_sync(0xffffffffu, res, 4);
    res += __shfl_xor_sync(0xffffffffu, res, 8);
    res += __shfl_xor_sync(0xffffffffu, res, 16);
    return res;
}

__device__ __forceinline__ float half_warp_reduce4_to_q(const float a[4], int lx, unsigned mask) {
    const bool p0 = (lx & 1) != 0;
    const float u0 = (p0 ? a[1] : a[0]) + __shfl_xor_sync(mask, p0 ? a[0] : a[1], 1);
    const float u1 = (p0 ? a[3] : a[2]) + __shfl_xor_sync(mask, p0 ? a[2] : a[3], 1);
    const bool p1 = (lx & 2) != 0;
    float res = (p1 ? u1 : u0) + __shfl_xor_sync(mask, p1 ? u0 : u1, 2);
    res += __shfl_xor_sync(mask, res, 4);
    res += __shfl_xor_sync(mask, res, 8);
    return res;
}

__device__ __forceinline__ float warp_reduce8_to_q(const float a[8], int lane) {
    const bool p0 = (lane & 1) != 0;
    const float u0 = (p0 ? a[1] : a[0]) + __shfl_xor_sync(0xffffffffu, p0 ? a[0] : a[1], 1);
    const float u1 = (p0 ? a[3] : a[2]) + __shfl_xor_sync(0xffffffffu, p0 ? a[2] : a[3], 1);
    const float u2 = (p0 ? a[5] : a[4]) + __shfl_xor_sync(0xffffffffu, p0 ? a[4] : a[5], 1);
    const float u3 = (p0 ? a[7] : a[6]) + __shfl_xor_sync(0xffffffffu, p0 ? a[6] : a[7], 1);
    const bool p1 = (lane & 2) != 0;
    const float v0 = (p1 ? u1 : u0) + __shfl_xor_sync(0xffffffffu, p1 ? u0 : u1, 2);
    const float v1 = (p1 ? u3 : u2) + __shfl_xor_sync(0xffffffffu, p1 ? u2 : u3, 2);
    const bool p2 = (lane & 4) != 0;
    float res = (p2 ? v1 : v0) + __shfl_xor_sync(0xffffffffu, p2 ? v0 : v1, 4);
    res += __shfl_xor_sync(0xffffffffu, res, 8);
    res += __shfl_xor_sync(0xffffffffu, res, 16);
    return res;
}

template <int LMAX, bool NORM, bool M4 = false>
__global__ void __launch_bounds__(128, 6) ilmlip_l1_node_fwd_lf1_msg_v5_kernel(
    int N, int n_feat_pad, int n_msg, Int4 n_l, Int4 msg_in_dim,
    CPtr4 A1, const float* __restrict__ W_msg0_T, const float* __restrict__ W_msg1_T,
    const float* __restrict__ mu_x, const float* __restrict__ inv_std_x,
    float* __restrict__ x_out, float* __restrict__ am_l1_out,
    float4* __restrict__ I_pack4, Ptr4 I_out) {
    constexpr int QMAX = M4 ? 4 : 8;
    extern __shared__ float smem[];
    const int m0d = msg_in_dim.v[0], m1d = msg_in_dim.v[1];
    const int m0p = (m0d + 3) & ~3;
    const int m1p = (m1d + 3) & ~3;
    const int m0p4 = m0p >> 2;
    const int m1p4 = m1p >> 2;
    float* sW0T  = smem;
    float* sW1T  = sW0T + QMAX * m0p;
    float* s_bc0 = sW1T + QMAX * m1p;
    float* s_bc1 = s_bc0 + 4 * m0p;

    const int tid = threadIdx.x;
    if (W_msg0_T) {
        for (int idx = tid; idx < QMAX * m0p; idx += 128) {
            const int q = idx / m0p, k = idx - q * m0p;
            sW0T[idx] = (q < n_msg && k < m0d) ? __ldg(W_msg0_T + q * m0d + k) : 0.0f;
        }
    }
    for (int idx = tid; idx < QMAX * m1p; idx += 128) {
        const int q = idx / m1p, k = idx - q * m1p;
        sW1T[idx] = (q < n_msg && k < m1d) ? __ldg(W_msg1_T + q * m1d + k) : 0.0f;
    }
    __syncthreads();

    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int c = lane;
    float* bc0 = s_bc0 + wid * m0p;
    float* bc1_0 = s_bc1 + wid * 3 * m1p;
    float* bc1_1 = bc1_0 + m1p;
    float* bc1_2 = bc1_1 + m1p;
    if (m0d + lane < m0p) bc0[m0d + lane] = 0.0f;
    if (m1d + lane < m1p) { bc1_0[m1d + lane] = 0.0f; bc1_1[m1d + lane] = 0.0f; bc1_2[m1d + lane] = 0.0f; }

    const float4* sW0T4 = reinterpret_cast<const float4*>(sW0T);
    const float4* sW1T4 = reinterpret_cast<const float4*>(sW1T);
    const float4* bc0_4 = reinterpret_cast<const float4*>(bc0);
    const float4* b10_4 = reinterpret_cast<const float4*>(bc1_0);
    const float4* b11_4 = reinterpret_cast<const float4*>(bc1_1);
    const float4* b12_4 = reinterpret_cast<const float4*>(bc1_2);
    const bool x_vec4 = ((n_feat_pad & 3) == 0);

    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];
    const int c01 = nl0 < nl1 ? nl0 : nl1;
    const int c12 = nl1 < nl2 ? nl1 : nl2;
    const int c02 = nl0 < nl2 ? nl0 : nl2;
    const int c012 = c12 < nl0 ? c12 : nl0;
    const int c23 = (LMAX == 3) ? (nl2 < nl3 ? nl2 : nl3) : 0;
    const int c123 = (LMAX == 3) ? (c23 < nl1 ? c23 : nl1) : 0;
    const int c023 = (LMAX == 3) ? (c23 < nl0 ? c23 : nl0) : 0;
    const int c03 = (LMAX == 3) ? (nl0 < nl3 ? nl0 : nl3) : 0;
    const int c13 = (LMAX == 3) ? (nl1 < nl3 ? nl1 : nl3) : 0;

    constexpr float C110  = 0.5773502588272095f;
    constexpr float CS310 = 0.547722578048706f, CS110 = 0.3162277638912201f, CS410 = 0.6324555277824402f;
    constexpr float C220  = 0.4472135901451111f;
    constexpr float K1 = 0.4629100561141968f, K2 = 0.37796446681022644f, K3 = 0.11952286213636398f;
    constexpr float K4 = 0.47809144854545593f, K5 = 0.29277002811431885f, K6 = 0.41403934359550476f;
    constexpr float K7 = 0.5070925354957581f, C330 = 0.37796446681022644f;

    const int i_base = blockIdx.x * 8 + wid * 2;
    for (int ia = 0; ia < 2; ++ia) {
        const int i = i_base + ia;
        if (i >= N) break;

        auto emit0 = [&](int off, int lim, float val) { if (c < lim) bc0[off + c] = val; };
        auto emit1 = [&](int off, int lim, float v0, float v1, float v2) {
            if (c < lim) { const int f = off + c; bc1_0[f] = v0; bc1_1[f] = v1; bc1_2[f] = v2; }
        };

        const float a0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;
        float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
        if (c < nl1) {
            const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c;
            a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1];
        }
        float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
        if (c < nl2) {
            const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c;
            a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2];
        }
        float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;
        if constexpr (LMAX == 3) {
            if (c < nl3) {
                const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c;
                a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3];
                a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3];
            }
        }

        int xoff = 0, b1off = 0;
        emit0(xoff, nl0, a0); xoff += nl0;
        emit1(b1off, nl1, a1_0, a1_1, a1_2); b1off += nl1;

        const float b0 = a0 * a0;
        emit0(xoff, nl0, b0); xoff += nl0;

        const float b1_0 = a0 * a1_0, b1_1 = a0 * a1_1, b1_2 = a0 * a1_2;
        emit1(b1off, c01, b1_0, b1_1, b1_2); b1off += c01;

        const float b2 = C110 * (a1_0 * a1_0 + a1_1 * a1_1 + a1_2 * a1_2);
        emit0(xoff, nl1, b2); xoff += nl1;

        const float b3_0 = CS310 * (a1_2 * a2_0 + a1_1 * a2_1 - a1_0 * a2_4) - CS110 * a1_0 * a2_2;
        const float b3_1 = CS310 * (a1_0 * a2_1 + a1_2 * a2_3) + CS410 * a1_1 * a2_2;
        const float b3_2 = CS310 * (a1_0 * a2_0 + a1_1 * a2_3 + a1_2 * a2_4) - CS110 * a1_2 * a2_2;
        emit1(b1off, c12, b3_0, b3_1, b3_2); b1off += c12;

        const float b4 = C220 * (a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4);
        emit0(xoff, nl2, b4); xoff += nl2;

        float b5_0 = 0.0f, b5_1 = 0.0f, b5_2 = 0.0f, b6 = 0.0f;
        if constexpr (LMAX == 3) {
            b5_0 = K1 * (a2_4 * a3_0 - a2_0 * a3_6) + K2 * (a2_3 * a3_1 - a2_1 * a3_5) +
                   K3 * (a2_4 * a3_2 - a2_0 * a3_4) - K5 * a2_1 * a3_3 + K6 * a2_2 * a3_2;
            b5_1 = K2 * (a2_0 * a3_1 + a2_4 * a3_5) + K4 * (a2_1 * a3_2 + a2_3 * a3_4) + K7 * a2_2 * a3_3;
            b5_2 = K1 * (a2_0 * a3_0 + a2_4 * a3_6) + K2 * (a2_1 * a3_1 + a2_3 * a3_5) -
                   K3 * (a2_0 * a3_2 + a2_4 * a3_4) - K5 * a2_3 * a3_3 + K6 * a2_2 * a3_4;
            emit1(b1off, c23, b5_0, b5_1, b5_2); b1off += c23;

            b6 = C330 * (a3_0 * a3_0 + a3_1 * a3_1 + a3_2 * a3_2 + a3_3 * a3_3 + a3_4 * a3_4 + a3_5 * a3_5 + a3_6 * a3_6);
            emit0(xoff, nl3, b6); xoff += nl3;
        }

        emit0(xoff, nl0, b0 * a0); xoff += nl0;
        emit0(xoff, c01, C110 * (b1_0 * a1_0 + b1_1 * a1_1 + b1_2 * a1_2)); xoff += c01;
        emit0(xoff, c12, C110 * (b3_0 * a1_0 + b3_1 * a1_1 + b3_2 * a1_2)); xoff += c12;
        emit0(xoff, c02, b4 * a0); xoff += c02;
        if constexpr (LMAX == 3) {
            emit0(xoff, c123, C110 * (b5_0 * a1_0 + b5_1 * a1_1 + b5_2 * a1_2)); xoff += c123;
            emit0(xoff, c03, b6 * a0);
        }

        emit1(b1off, c01,  b0 * a1_0, b0 * a1_1, b0 * a1_2); b1off += c01;
        emit1(b1off, nl1,  b2 * a1_0, b2 * a1_1, b2 * a1_2); b1off += nl1;
        emit1(b1off, c012, b3_0 * a0, b3_1 * a0, b3_2 * a0); b1off += c012;
        emit1(b1off, c12,  b4 * a1_0, b4 * a1_1, b4 * a1_2); b1off += c12;
        if constexpr (LMAX == 3) {
            emit1(b1off, c023, b5_0 * a0, b5_1 * a0, b5_2 * a0); b1off += c023;
            emit1(b1off, c13,  b6 * a1_0, b6 * a1_1, b6 * a1_2);
        }
        __syncwarp();

        float* xi = x_out + (int64_t)i * n_feat_pad;
        float4* xi4 = reinterpret_cast<float4*>(xi);
        const float4* mu4p = reinterpret_cast<const float4*>(mu_x);
        const float4* is4p = reinterpret_cast<const float4*>(inv_std_x);
        float am = 0.0f;
        float r0[QMAX] = {};
        for (int k4 = lane; k4 < m0p4; k4 += 32) {
            const float4 b4v = bc0_4[k4];
            if (W_msg0_T) {
                #pragma unroll
                for (int q = 0; q < QMAX; ++q) {
                    r0[q] += v1_dot4(b4v, sW0T4[q * m0p4 + k4]);
                }
            }
            const int f = k4 << 2;
            if (x_vec4 && f + 3 < m0d) {
                if constexpr (NORM) {
                    const float4 m4 = __ldg(mu4p + k4);
                    const float4 s4 = __ldg(is4p + k4);
                    const float4 vn = make_float4((b4v.x - m4.x) * s4.x, (b4v.y - m4.y) * s4.y,
                                                  (b4v.z - m4.z) * s4.z, (b4v.w - m4.w) * s4.w);
                    xi4[k4] = vn;
                    am = fmaxf(am, fmaxf(fmaxf(fabsf(vn.x), fabsf(vn.y)), fmaxf(fabsf(vn.z), fabsf(vn.w))));
                } else {
                    xi4[k4] = b4v;
                }
            } else {
                const float bv[4] = {b4v.x, b4v.y, b4v.z, b4v.w};
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (f + t < m0d) {
                        if constexpr (NORM) {
                            const float vn = (bv[t] - __ldg(mu_x + f + t)) * __ldg(inv_std_x + f + t);
                            xi[f + t] = vn;
                            am = fmaxf(am, fabsf(vn));
                        } else {
                            xi[f + t] = bv[t];
                        }
                    }
                }
            }
        }
        float out_i0 = 0.0f;
        if constexpr (M4) out_i0 = warp_reduce4_to_q(r0, lane);
        else              out_i0 = warp_reduce8_to_q(r0, lane);

        float r1_0[QMAX] = {}, r1_1[QMAX] = {}, r1_2[QMAX] = {};
        for (int k4 = lane; k4 < m1p4; k4 += 32) {
            const float4 v0 = b10_4[k4], v1 = b11_4[k4], v2 = b12_4[k4];
            #pragma unroll
            for (int q = 0; q < QMAX; ++q) {
                const float4 w = sW1T4[q * m1p4 + k4];
                r1_0[q] += v1_dot4(v0, w);
                r1_1[q] += v1_dot4(v1, w);
                r1_2[q] += v1_dot4(v2, w);
            }
        }
        float out_i1_0 = 0.0f, out_i1_1 = 0.0f, out_i1_2 = 0.0f;
        if constexpr (M4) {
            out_i1_0 = warp_reduce4_to_q(r1_0, lane);
            out_i1_1 = warp_reduce4_to_q(r1_1, lane);
            out_i1_2 = warp_reduce4_to_q(r1_2, lane);
        } else {
            out_i1_0 = warp_reduce8_to_q(r1_0, lane);
            out_i1_1 = warp_reduce8_to_q(r1_1, lane);
            out_i1_2 = warp_reduce8_to_q(r1_2, lane);
        }

        if (lane < n_msg) {
            if (I_pack4) {
                I_pack4[(int64_t)i * n_msg + lane] = make_float4(out_i0, out_i1_0, out_i1_1, out_i1_2);
            } else {
                if (I_out.p[0]) I_out.p[0][(int64_t)i * n_msg + lane] = out_i0;
                float* d1 = I_out.p[1] + (int64_t)i * 3 * n_msg + lane;
                d1[0] = out_i1_0; d1[n_msg] = out_i1_1; d1[2 * n_msg] = out_i1_2;
            }
        }
        if constexpr (NORM) {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
            if (lane == 0) am_l1_out[i] = am;
        }
        __syncwarp();
    }
}

template <int LMAX, bool SCALE_DX, bool M4 = false>
__global__ void __launch_bounds__(128, 6) ilmlip_l1_node_bwd_lf1_msg_v5_kernel(
    int N, int n_feat_pad, int n_msg, Int4 n_l, Int4 msg_in_dim,
    CPtr4 A1, const float* __restrict__ inv_std_x, const float* __restrict__ dx, CPtr4 dI,
    const float* __restrict__ W_msg0_T, const float* __restrict__ W_msg1_T, Ptr4 dA1) {
    constexpr int QMAX = M4 ? 4 : 8;
    extern __shared__ float smem[];
    const int m0d = msg_in_dim.v[0], m1d = msg_in_dim.v[1];
    const int m0p = (m0d + 3) & ~3;
    const int m1p = (m1d + 3) & ~3;
    const int m0p4 = m0p >> 2;
    const int m1p4 = m1p >> 2;
    float* sW0T   = smem;
    float* sW1T   = sW0T + QMAX * m0p;
    float* s_gbc0 = sW1T + QMAX * m1p;
    float* s_gbc1 = s_gbc0 + 4 * m0p;

    const int tid = threadIdx.x;
    if (W_msg0_T) {
        for (int idx = tid; idx < QMAX * m0p; idx += 128) {
            const int q = idx / m0p, k = idx - q * m0p;
            sW0T[idx] = (q < n_msg && k < m0d) ? __ldg(W_msg0_T + q * m0d + k) : 0.0f;
        }
    }
    for (int idx = tid; idx < QMAX * m1p; idx += 128) {
        const int q = idx / m1p, k = idx - q * m1p;
        sW1T[idx] = (q < n_msg && k < m1d) ? __ldg(W_msg1_T + q * m1d + k) : 0.0f;
    }
    __syncthreads();

    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int c = lane;
    float* gbc0 = s_gbc0 + wid * m0p;
    float* gbc1_0 = s_gbc1 + wid * 3 * m1p;
    float* gbc1_1 = gbc1_0 + m1p;
    float* gbc1_2 = gbc1_1 + m1p;

    const float4* sW0T4 = reinterpret_cast<const float4*>(sW0T);
    const float4* sW1T4 = reinterpret_cast<const float4*>(sW1T);
    float4* gbc0_4 = reinterpret_cast<float4*>(gbc0);
    float4* g10_4  = reinterpret_cast<float4*>(gbc1_0);
    float4* g11_4  = reinterpret_cast<float4*>(gbc1_1);
    float4* g12_4  = reinterpret_cast<float4*>(gbc1_2);
    const bool x_vec4 = ((n_feat_pad & 3) == 0);

    const int nl0 = n_l.v[0], nl1 = n_l.v[1], nl2 = n_l.v[2], nl3 = n_l.v[3];
    const int c01 = nl0 < nl1 ? nl0 : nl1;
    const int c12 = nl1 < nl2 ? nl1 : nl2;
    const int c02 = nl0 < nl2 ? nl0 : nl2;
    const int c012 = c12 < nl0 ? c12 : nl0;
    const int c23 = (LMAX == 3) ? (nl2 < nl3 ? nl2 : nl3) : 0;
    const int c123 = (LMAX == 3) ? (c23 < nl1 ? c23 : nl1) : 0;
    const int c023 = (LMAX == 3) ? (c23 < nl0 ? c23 : nl0) : 0;
    const int c03 = (LMAX == 3) ? (nl0 < nl3 ? nl0 : nl3) : 0;
    const int c13 = (LMAX == 3) ? (nl1 < nl3 ? nl1 : nl3) : 0;

    constexpr float C110  = 0.5773502588272095f;
    constexpr float CS310 = 0.547722578048706f, CS110 = 0.3162277638912201f, CS410 = 0.6324555277824402f;
    constexpr float C220  = 0.4472135901451111f;
    constexpr float K1 = 0.4629100561141968f, K2 = 0.37796446681022644f, K3 = 0.11952286213636398f;
    constexpr float K4 = 0.47809144854545593f, K5 = 0.29277002811431885f, K6 = 0.41403934359550476f;
    constexpr float K7 = 0.5070925354957581f, C330 = 0.37796446681022644f;

    const int i_base = blockIdx.x * 8 + wid * 2;
    for (int ia = 0; ia < 2; ++ia) {
        const int i = i_base + ia;
        if (i >= N) break;

        {
            const float d0_l = (W_msg0_T && dI.p[0] && lane < n_msg) ? __ldg(dI.p[0] + (int64_t)i * n_msg + lane) : 0.0f;
            float d0[QMAX];
            if (W_msg0_T) {
                #pragma unroll
                for (int q = 0; q < QMAX; ++q) d0[q] = __shfl_sync(0xffffffffu, d0_l, q);
            }
            const float* dxi = dx + (int64_t)i * n_feat_pad;
            const float4* dxi4 = reinterpret_cast<const float4*>(dxi);
            const float4* is4p = reinterpret_cast<const float4*>(inv_std_x);
            for (int k4 = lane; k4 < m0p4; k4 += 32) {
                const int f = k4 << 2;
                float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (x_vec4 && f + 3 < m0d) {
                    s = __ldg(dxi4 + k4);
                    if constexpr (SCALE_DX) {
                        const float4 sc = __ldg(is4p + k4);
                        s.x *= sc.x; s.y *= sc.y; s.z *= sc.z; s.w *= sc.w;
                    }
                } else {
                    if (f + 0 < m0d) s.x = dxi[f + 0] * (SCALE_DX ? __ldg(inv_std_x + f + 0) : 1.0f);
                    if (f + 1 < m0d) s.y = dxi[f + 1] * (SCALE_DX ? __ldg(inv_std_x + f + 1) : 1.0f);
                    if (f + 2 < m0d) s.z = dxi[f + 2] * (SCALE_DX ? __ldg(inv_std_x + f + 2) : 1.0f);
                    if (f + 3 < m0d) s.w = dxi[f + 3] * (SCALE_DX ? __ldg(inv_std_x + f + 3) : 1.0f);
                }
                if (W_msg0_T) {
                    #pragma unroll
                    for (int q = 0; q < QMAX; ++q) {
                        const float4 w = sW0T4[q * m0p4 + k4];
                        s.x = fmaf(d0[q], w.x, s.x);
                        s.y = fmaf(d0[q], w.y, s.y);
                        s.z = fmaf(d0[q], w.z, s.z);
                        s.w = fmaf(d0[q], w.w, s.w);
                    }
                }
                gbc0_4[k4] = s;
            }
        }
        {
            float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f;
            if (lane < n_msg) {
                const float* p1 = dI.p[1] + (int64_t)i * 3 * n_msg + lane;
                l0 = __ldg(p1); l1 = __ldg(p1 + n_msg); l2 = __ldg(p1 + 2 * n_msg);
            }
            float d1_0[QMAX], d1_1[QMAX], d1_2[QMAX];
            #pragma unroll
            for (int q = 0; q < QMAX; ++q) {
                d1_0[q] = __shfl_sync(0xffffffffu, l0, q);
                d1_1[q] = __shfl_sync(0xffffffffu, l1, q);
                d1_2[q] = __shfl_sync(0xffffffffu, l2, q);
            }
            for (int k4 = lane; k4 < m1p4; k4 += 32) {
                float4 r0 = make_float4(0.f, 0.f, 0.f, 0.f);
                float4 r1 = make_float4(0.f, 0.f, 0.f, 0.f);
                float4 r2 = make_float4(0.f, 0.f, 0.f, 0.f);
                #pragma unroll
                for (int q = 0; q < QMAX; ++q) {
                    const float4 w = sW1T4[q * m1p4 + k4];
                    r0.x = fmaf(d1_0[q], w.x, r0.x); r0.y = fmaf(d1_0[q], w.y, r0.y); r0.z = fmaf(d1_0[q], w.z, r0.z); r0.w = fmaf(d1_0[q], w.w, r0.w);
                    r1.x = fmaf(d1_1[q], w.x, r1.x); r1.y = fmaf(d1_1[q], w.y, r1.y); r1.z = fmaf(d1_1[q], w.z, r1.z); r1.w = fmaf(d1_1[q], w.w, r1.w);
                    r2.x = fmaf(d1_2[q], w.x, r2.x); r2.y = fmaf(d1_2[q], w.y, r2.y); r2.z = fmaf(d1_2[q], w.z, r2.z); r2.w = fmaf(d1_2[q], w.w, r2.w);
                }
                g10_4[k4] = r0; g11_4[k4] = r1; g12_4[k4] = r2;
            }
        }
        __syncwarp();

        // Step 3: Exact nu=1,2,3 backprop reading single-cycle SMEM gbc0 / gbc1!
        const float a0 = (c < nl0) ? A1.p[0][(int64_t)i * nl0 + c] : 0.0f;
        float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f;
        if (c < nl1) {
            const float* s1 = A1.p[1] + (int64_t)i * 3 * nl1 + c;
            a1_0 = s1[0]; a1_1 = s1[nl1]; a1_2 = s1[2 * nl1];
        }
        float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
        if (c < nl2) {
            const float* s2 = A1.p[2] + (int64_t)i * 5 * nl2 + c;
            a2_0 = s2[0]; a2_1 = s2[nl2]; a2_2 = s2[2 * nl2]; a2_3 = s2[3 * nl2]; a2_4 = s2[4 * nl2];
        }
        float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f, a3_4 = 0.0f, a3_5 = 0.0f, a3_6 = 0.0f;
        if constexpr (LMAX == 3) {
            if (c < nl3) {
                const float* s3 = A1.p[3] + (int64_t)i * 7 * nl3 + c;
                a3_0 = s3[0]; a3_1 = s3[nl3]; a3_2 = s3[2 * nl3]; a3_3 = s3[3 * nl3];
                a3_4 = s3[4 * nl3]; a3_5 = s3[5 * nl3]; a3_6 = s3[6 * nl3];
            }
        }

        float da0 = (c < nl0) ? gbc0[c] : 0.0f;
        float da1_0 = (c < nl1) ? gbc1_0[c] : 0.0f;
        float da1_1 = (c < nl1) ? gbc1_1[c] : 0.0f;
        float da1_2 = (c < nl1) ? gbc1_2[c] : 0.0f;
        float da2_0 = 0.0f, da2_1 = 0.0f, da2_2 = 0.0f, da2_3 = 0.0f, da2_4 = 0.0f;
        float da3_0 = 0.0f, da3_1 = 0.0f, da3_2 = 0.0f, da3_3 = 0.0f, da3_4 = 0.0f, da3_5 = 0.0f, da3_6 = 0.0f;

        int b0off = nl0, b1off = nl1;
        const float b0 = a0 * a0;
        float db0 = (c < nl0) ? gbc0[b0off + c] : 0.0f;
        b0off += nl0;

        const float b1_0 = a0 * a1_0, b1_1 = a0 * a1_1, b1_2 = a0 * a1_2;
        float db1_0 = (c < c01) ? gbc1_0[b1off + c] : 0.0f;
        float db1_1 = (c < c01) ? gbc1_1[b1off + c] : 0.0f;
        float db1_2 = (c < c01) ? gbc1_2[b1off + c] : 0.0f;
        b1off += c01;

        const float b2 = C110 * (a1_0 * a1_0 + a1_1 * a1_1 + a1_2 * a1_2);
        float db2 = (c < nl1) ? gbc0[b0off + c] : 0.0f;
        b0off += nl1;

        const float b3_0 = CS310 * (a1_2 * a2_0 + a1_1 * a2_1 - a1_0 * a2_4) - CS110 * a1_0 * a2_2;
        const float b3_1 = CS310 * (a1_0 * a2_1 + a1_2 * a2_3) + CS410 * a1_1 * a2_2;
        const float b3_2 = CS310 * (a1_0 * a2_0 + a1_1 * a2_3 + a1_2 * a2_4) - CS110 * a1_2 * a2_2;
        float db3_0 = (c < c12) ? gbc1_0[b1off + c] : 0.0f;
        float db3_1 = (c < c12) ? gbc1_1[b1off + c] : 0.0f;
        float db3_2 = (c < c12) ? gbc1_2[b1off + c] : 0.0f;
        b1off += c12;

        const float b4 = C220 * (a2_0 * a2_0 + a2_1 * a2_1 + a2_2 * a2_2 + a2_3 * a2_3 + a2_4 * a2_4);
        float db4 = (c < nl2) ? gbc0[b0off + c] : 0.0f;
        b0off += nl2;

        float b5_0 = 0.0f, b5_1 = 0.0f, b5_2 = 0.0f, db5_0 = 0.0f, db5_1 = 0.0f, db5_2 = 0.0f;
        float b6 = 0.0f, db6 = 0.0f;
        if constexpr (LMAX == 3) {
            b5_0 = K1 * (a2_4 * a3_0 - a2_0 * a3_6) + K2 * (a2_3 * a3_1 - a2_1 * a3_5) +
                   K3 * (a2_4 * a3_2 - a2_0 * a3_4) - K5 * a2_1 * a3_3 + K6 * a2_2 * a3_2;
            b5_1 = K2 * (a2_0 * a3_1 + a2_4 * a3_5) + K4 * (a2_1 * a3_2 + a2_3 * a3_4) + K7 * a2_2 * a3_3;
            b5_2 = K1 * (a2_0 * a3_0 + a2_4 * a3_6) + K2 * (a2_1 * a3_1 + a2_3 * a3_5) -
                   K3 * (a2_0 * a3_2 + a2_4 * a3_4) - K5 * a2_3 * a3_3 + K6 * a2_2 * a3_4;
            if (c < c23) { db5_0 = gbc1_0[b1off + c]; db5_1 = gbc1_1[b1off + c]; db5_2 = gbc1_2[b1off + c]; }
            b1off += c23;
            b6 = C330 * (a3_0 * a3_0 + a3_1 * a3_1 + a3_2 * a3_2 + a3_3 * a3_3 + a3_4 * a3_4 + a3_5 * a3_5 + a3_6 * a3_6);
            if (c < nl3) db6 = gbc0[b0off + c];
            b0off += nl3;
        }

        // Backprop nu=3 L=0
        if (c < nl0) { const float g = gbc0[b0off + c]; db0 = fmaf(g, a0, db0); da0 = fmaf(g, b0, da0); }
        b0off += nl0;
        if (c < c01) {
            const float g = gbc0[b0off + c] * C110;
            db1_0 = fmaf(g, a1_0, db1_0); db1_1 = fmaf(g, a1_1, db1_1); db1_2 = fmaf(g, a1_2, db1_2);
            da1_0 = fmaf(g, b1_0, da1_0); da1_1 = fmaf(g, b1_1, da1_1); da1_2 = fmaf(g, b1_2, da1_2);
        }
        b0off += c01;
        if (c < c12) {
            const float g = gbc0[b0off + c] * C110;
            db3_0 = fmaf(g, a1_0, db3_0); db3_1 = fmaf(g, a1_1, db3_1); db3_2 = fmaf(g, a1_2, db3_2);
            da1_0 = fmaf(g, b3_0, da1_0); da1_1 = fmaf(g, b3_1, da1_1); da1_2 = fmaf(g, b3_2, da1_2);
        }
        b0off += c12;
        if (c < c02) { const float g = gbc0[b0off + c]; db4 = fmaf(g, a0, db4); da0 = fmaf(g, b4, da0); }
        b0off += c02;
        if constexpr (LMAX == 3) {
            if (c < c123) {
                const float g = gbc0[b0off + c] * C110;
                db5_0 = fmaf(g, a1_0, db5_0); db5_1 = fmaf(g, a1_1, db5_1); db5_2 = fmaf(g, a1_2, db5_2);
                da1_0 = fmaf(g, b5_0, da1_0); da1_1 = fmaf(g, b5_1, da1_1); da1_2 = fmaf(g, b5_2, da1_2);
            }
            b0off += c123;
            if (c < c03) { const float g = gbc0[b0off + c]; db6 = fmaf(g, a0, db6); da0 = fmaf(g, b6, da0); }
        }

        // Backprop nu=3 L=1
        if (c < c01) {
            const float g0 = gbc1_0[b1off + c], g1 = gbc1_1[b1off + c], g2 = gbc1_2[b1off + c];
            db0 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db0)));
            da1_0 = fmaf(g0, b0, da1_0); da1_1 = fmaf(g1, b0, da1_1); da1_2 = fmaf(g2, b0, da1_2);
        }
        b1off += c01;
        if (c < nl1) {
            const float g0 = gbc1_0[b1off + c], g1 = gbc1_1[b1off + c], g2 = gbc1_2[b1off + c];
            db2 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db2)));
            da1_0 = fmaf(g0, b2, da1_0); da1_1 = fmaf(g1, b2, da1_1); da1_2 = fmaf(g2, b2, da1_2);
        }
        b1off += nl1;
        if (c < c012) {
            const float g0 = gbc1_0[b1off + c], g1 = gbc1_1[b1off + c], g2 = gbc1_2[b1off + c];
            db3_0 = fmaf(g0, a0, db3_0); db3_1 = fmaf(g1, a0, db3_1); db3_2 = fmaf(g2, a0, db3_2);
            da0 = fmaf(g0, b3_0, fmaf(g1, b3_1, fmaf(g2, b3_2, da0)));
        }
        b1off += c012;
        if (c < c12) {
            const float g0 = gbc1_0[b1off + c], g1 = gbc1_1[b1off + c], g2 = gbc1_2[b1off + c];
            db4 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db4)));
            da1_0 = fmaf(g0, b4, da1_0); da1_1 = fmaf(g1, b4, da1_1); da1_2 = fmaf(g2, b4, da1_2);
        }
        b1off += c12;
        if constexpr (LMAX == 3) {
            if (c < c023) {
                const float g0 = gbc1_0[b1off + c], g1 = gbc1_1[b1off + c], g2 = gbc1_2[b1off + c];
                db5_0 = fmaf(g0, a0, db5_0); db5_1 = fmaf(g1, a0, db5_1); db5_2 = fmaf(g2, a0, db5_2);
                da0 = fmaf(g0, b5_0, fmaf(g1, b5_1, fmaf(g2, b5_2, da0)));
            }
            b1off += c023;
            if (c < c13) {
                const float g0 = gbc1_0[b1off + c], g1 = gbc1_1[b1off + c], g2 = gbc1_2[b1off + c];
                db6 = fmaf(g0, a1_0, fmaf(g1, a1_1, fmaf(g2, a1_2, db6)));
                da1_0 = fmaf(g0, b6, da1_0); da1_1 = fmaf(g1, b6, da1_1); da1_2 = fmaf(g2, b6, da1_2);
            }
        }

        // Backprop nu=2 -> da0..da3
        if (c < nl0) da0 = fmaf(2.0f * db0, a0, da0);
        if (c < c01) {
            da0 = fmaf(db1_0, a1_0, fmaf(db1_1, a1_1, fmaf(db1_2, a1_2, da0)));
            da1_0 = fmaf(db1_0, a0, da1_0); da1_1 = fmaf(db1_1, a0, da1_1); da1_2 = fmaf(db1_2, a0, da1_2);
        }
        if (c < nl1) {
            const float g = 2.0f * C110 * db2;
            da1_0 = fmaf(g, a1_0, da1_0); da1_1 = fmaf(g, a1_1, da1_1); da1_2 = fmaf(g, a1_2, da1_2);
        }
        if (c < c12) {
            da1_0 += CS310 * (db3_2 * a2_0 + db3_1 * a2_1 - db3_0 * a2_4) - CS110 * db3_0 * a2_2;
            da1_1 += CS310 * (db3_0 * a2_1 + db3_2 * a2_3) + CS410 * db3_1 * a2_2;
            da1_2 += CS310 * (db3_0 * a2_0 + db3_1 * a2_3 + db3_2 * a2_4) - CS110 * db3_2 * a2_2;
            da2_0 += CS310 * (db3_2 * a1_0 + db3_0 * a1_2);
            da2_1 += CS310 * (db3_1 * a1_0 + db3_0 * a1_1);
            da2_2 += -CS110 * (db3_0 * a1_0 + db3_2 * a1_2) + CS410 * db3_1 * a1_1;
            da2_3 += CS310 * (db3_2 * a1_1 + db3_1 * a1_2);
            da2_4 += CS310 * (db3_2 * a1_2 - db3_0 * a1_0);
        }
        if (c < nl2) {
            const float g = 2.0f * C220 * db4;
            da2_0 = fmaf(g, a2_0, da2_0); da2_1 = fmaf(g, a2_1, da2_1); da2_2 = fmaf(g, a2_2, da2_2);
            da2_3 = fmaf(g, a2_3, da2_3); da2_4 = fmaf(g, a2_4, da2_4);
        }
        if constexpr (LMAX == 3) {
            if (c < c23) {
                da2_0 += db5_2 * (K1 * a3_0 - K3 * a3_2) + db5_1 * (K2 * a3_1) + db5_0 * (-K3 * a3_4 - K1 * a3_6);
                da2_1 += db5_2 * (K2 * a3_1) + db5_1 * (K4 * a3_2) + db5_0 * (-K5 * a3_3 - K2 * a3_5);
                da2_2 += db5_0 * (K6 * a3_2) + db5_1 * (K7 * a3_3) + db5_2 * (K6 * a3_4);
                da2_3 += db5_0 * (K2 * a3_1) + db5_2 * (-K5 * a3_3 + K2 * a3_5) + db5_1 * (K4 * a3_4);
                da2_4 += db5_0 * (K1 * a3_0 + K3 * a3_2) + db5_2 * (-K3 * a3_4 + K1 * a3_6) + db5_1 * (K2 * a3_5);
                da3_0 += K1 * (db5_2 * a2_0 + db5_0 * a2_4);
                da3_1 += K2 * (db5_1 * a2_0 + db5_2 * a2_1 + db5_0 * a2_3);
                da3_2 += -K3 * db5_2 * a2_0 + K4 * db5_1 * a2_1 + K6 * db5_0 * a2_2 + K3 * db5_0 * a2_4;
                da3_3 += -K5 * db5_0 * a2_1 + K7 * db5_1 * a2_2 - K5 * db5_2 * a2_3;
                da3_4 += -K3 * db5_0 * a2_0 + K6 * db5_2 * a2_2 + K4 * db5_1 * a2_3 - K3 * db5_2 * a2_4;
                da3_5 += -K2 * db5_0 * a2_1 + K2 * db5_2 * a2_3 + K2 * db5_1 * a2_4;
                da3_6 += K1 * (-db5_0 * a2_0 + db5_2 * a2_4);
            }
            if (c < nl3) {
                const float g = 2.0f * C330 * db6;
                da3_0 = fmaf(g, a3_0, da3_0); da3_1 = fmaf(g, a3_1, da3_1); da3_2 = fmaf(g, a3_2, da3_2);
                da3_3 = fmaf(g, a3_3, da3_3); da3_4 = fmaf(g, a3_4, da3_4); da3_5 = fmaf(g, a3_5, da3_5);
                da3_6 = fmaf(g, a3_6, da3_6);
            }
        }

        if (c < nl0) dA1.p[0][(int64_t)i * nl0 + c] = da0;
        if (c < nl1) {
            float* d1 = dA1.p[1] + (int64_t)i * 3 * nl1 + c;
            d1[0] = da1_0; d1[nl1] = da1_1; d1[2 * nl1] = da1_2;
        }
        if (c < nl2) {
            float* d2 = dA1.p[2] + (int64_t)i * 5 * nl2 + c;
            d2[0] = da2_0; d2[nl2] = da2_1; d2[2 * nl2] = da2_2; d2[3 * nl2] = da2_3; d2[4 * nl2] = da2_4;
        }
        if constexpr (LMAX == 3) {
            if (c < nl3) {
                float* d3 = dA1.p[3] + (int64_t)i * 7 * nl3 + c;
                d3[0] = da3_0; d3[nl3] = da3_1; d3[2 * nl3] = da3_2; d3[3 * nl3] = da3_3;
                d3[4 * nl3] = da3_4; d3[5 * nl3] = da3_5; d3[6 * nl3] = da3_6;
            }
        }
        __syncwarp();
    }
}

// =================================================================================================
// PATH B3 (32-Atoms/CTA SUBWARP=4 Ultra-Fast L2 2-Hop: lmsg==1, lfeat2==1, lout2 in {1,2},
//          n_msg <= 4, NB2 in {8,12,14,16} -> 2x throughput over v1_n8, 128B/256B cache-aligned dA2_pack!)
// =================================================================================================
template <int NB2, int NB1, bool SAME_G, bool L2_QUAD, bool REM_BF>
__global__ void __launch_bounds__(128, SAME_G ? 5 : 4) ilmlip_l2_edge_fwd_v1_n4_kernel(
    int N, int S, int n_msg, ChebParam r_max2, ChebParam r_max1, float a0_over_sig1, float inv_DA2_sig1_kap_sq,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2, const float* __restrict__ Cbf,
    const float* __restrict__ f0, const float4* __restrict__ I_pack4, CPtr4 I_node,
    Ptr4 A2, float* __restrict__ f1_out) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int NB1_4 = (NB1 + 3) / 4;
    constexpr int EB = SAME_G ? 4 : 2;
    constexpr bool DUAL_G = REM_BF && !SAME_G;
    extern __shared__ float4 smem_c4[];
    __shared__ float4 s_g[32][EB][NB2_4];
    __shared__ float4 s_g1[DUAL_G ? 32 : 1][DUAL_G ? EB : 1][DUAL_G ? NB1_4 : 1];
    __shared__ float3 s_Y1[32][EB];
    __shared__ int2   s_jp[32][EB];
    __shared__ float  s_f0[REM_BF ? 32 : 1][REM_BF ? EB : 1];

    const int l2_tot_cols = 2 * n_msg;
    const int n_slots = S * S * 2 * 4;
    for (int idx = threadIdx.x; idx < n_slots; idx += 128) {
        const int q = idx & 3;
        const int pl = idx >> 2;
        const int pair = pl >> 1;
        const int l = pl & 1;
        #pragma unroll
        for (int b = 0; b < NB2_4; ++b) {
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (q < n_msg) {
                const int k0 = b * 4;
                const float* src = C2 + (pair * NB2 + k0) * l2_tot_cols + l * n_msg + q;
                v.x = (k0 + 0 < NB2) ? __ldg(src + 0 * l2_tot_cols) : 0.0f;
                v.y = (k0 + 1 < NB2) ? __ldg(src + 1 * l2_tot_cols) : 0.0f;
                v.z = (k0 + 2 < NB2) ? __ldg(src + 2 * l2_tot_cols) : 0.0f;
                v.w = (k0 + 3 < NB2) ? __ldg(src + 3 * l2_tot_cols) : 0.0f;
            }
            smem_c4[(pair * (2 * NB2_4) + l * NB2_4 + b) * 4 + q] = v;
        }
    }
    float4* s_bf4 = smem_c4 + S * S * (2 * NB2_4) * 4;
    if constexpr (REM_BF) {
        for (int idx = threadIdx.x; idx < n_slots; idx += 128) {
            const int q = idx & 3;
            const int pl = idx >> 2;
            const int pair = pl >> 1;
            const int l = pl & 1;
            #pragma unroll
            for (int b = 0; b < NB1_4; ++b) {
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (q < n_msg) {
                    const int k0 = b * 4;
                    const float* src = Cbf + (pair * NB1 + k0) * l2_tot_cols + l * n_msg + q;
                    v.x = (k0 + 0 < NB1) ? __ldg(src + 0 * l2_tot_cols) : 0.0f;
                    v.y = (k0 + 1 < NB1) ? __ldg(src + 1 * l2_tot_cols) : 0.0f;
                    v.z = (k0 + 2 < NB1) ? __ldg(src + 2 * l2_tot_cols) : 0.0f;
                    v.w = (k0 + 3 < NB1) ? __ldg(src + 3 * l2_tot_cols) : 0.0f;
                }
                s_bf4[(pair * (2 * NB1_4) + l * NB1_4 + b) * 4 + q] = v;
            }
        }
    }
    __syncthreads();

    const int sid = threadIdx.x >> 2, slane = threadIdx.x & 3;
    const int qbase = (sid & 7) * 4;
    const unsigned qmask = 0xfu << qbase;
    const int i = blockIdx.x * 32 + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const bool active_q = (slane < n_msg);
    constexpr float C110   = 0.57735026918962576451f;
    constexpr float C_SQ2  = 0.70710678118654752440f;
    constexpr float C_SQ6  = 0.40824829046386301637f;
    constexpr float C_2SQ6 = 0.81649658092772603273f;

    float a00 = 0.0f, a01 = 0.0f;
    float a10_0 = 0.0f, a10_1 = 0.0f, a10_2 = 0.0f;
    float a11_0 = 0.0f, a11_1 = 0.0f, a11_2 = 0.0f;
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (slane < bcnt) {
            const int e = eb_b + slane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const float r = fmaxf(vl.w, 1e-8f);
            const float sc = 1.7320508075688772935f / r;
            s_jp[sid][slane] = make_int2(j, z_i * S + __ldg(species + j));
            if constexpr (REM_BF) s_f0[sid][slane] = __ldg(f0 + j);
            s_Y1[sid][slane] = make_float3(vl.x * sc, vl.y * sc, vl.z * sc);
            eval_cheb_nb_store<NB2, false>(r, r_max2, s_g[sid][slane], nullptr);
        } else if constexpr (DUAL_G) {
            if (slane >= 2 && slane < 2 + bcnt) {
                const int bi = slane - 2;
                const float4 vl = __ldg(vec_len + eb_b + bi);
                const float r = fmaxf(vl.w, 1e-8f);
                eval_cheb_nb_store<NB1, false>(r, r_max1, s_g1[sid][bi], nullptr);
            }
        }
        __syncwarp(qmask);

        if (active_q) {
            for (int bi = 0; bi < bcnt; ++bi) {
                const int2 jp = s_jp[sid][bi];
                const int j = jp.x, pair = jp.y;
                const float3 Y1 = s_Y1[sid][bi];
                const float4* cp4 = smem_c4 + (pair * (2 * NB2_4)) * 4 + slane;
                float r2_0 = 0.0f, r2_1 = 0.0f;
                #pragma unroll
                for (int b = 0; b < NB2_4; ++b) {
                    const float4 gv = s_g[sid][bi][b];
                    r2_0 += v1_dot4(cp4[b * 4], gv);
                    r2_1 += v1_dot4(cp4[(NB2_4 + b) * 4], gv);
                }

                float i0, i1_0, i1_1, i1_2;
                if (I_pack4) {
                    const float4 ir = __ldg(I_pack4 + (int64_t)j * n_msg + slane);
                    i0 = ir.x; i1_0 = ir.y; i1_1 = ir.z; i1_2 = ir.w;
                } else {
                    i0 = __ldg(I_node.p[0] + (int64_t)j * n_msg + slane);
                    const float* pI1 = I_node.p[1] + (int64_t)j * 3 * n_msg + slane;
                    i1_0 = __ldg(pI1); i1_1 = __ldg(pI1 + n_msg); i1_2 = __ldg(pI1 + 2 * n_msg);
                }
                if constexpr (REM_BF) {
                    const float4* bp4 = s_bf4 + (pair * (2 * NB1_4)) * 4 + slane;
                    float rb0 = 0.0f, rb1 = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < NB1_4; ++b) {
                        const float4 gv = DUAL_G ? s_g1[sid][bi][b] : s_g[sid][bi][b];
                        rb0 += v1_dot4(bp4[b * 4], gv);
                        rb1 += v1_dot4(bp4[(NB1_4 + b) * 4], gv);
                    }
                    const float f0j = s_f0[sid][bi];
                    i0 -= f0j * rb0;
                    const float b1s = f0j * rb1;
                    i1_0 -= b1s * Y1.x; i1_1 -= b1s * Y1.y; i1_2 -= b1s * Y1.z;
                }
                const float p1_0 = r2_1 * Y1.x, p1_1 = r2_1 * Y1.y, p1_2 = r2_1 * Y1.z;
                a00 = fmaf(r2_0, i0, a00);
                a01 = fmaf(C110, p1_0 * i1_0 + p1_1 * i1_1 + p1_2 * i1_2, a01);
                a10_0 = fmaf(r2_0, i1_0, a10_0); a10_1 = fmaf(r2_0, i1_1, a10_1); a10_2 = fmaf(r2_0, i1_2, a10_2);
                a11_0 = fmaf(p1_0, i0, a11_0);   a11_1 = fmaf(p1_1, i0, a11_1);   a11_2 = fmaf(p1_2, i0, a11_2);
                if constexpr (L2_QUAD) {
                    a2_0 = fmaf(C_SQ2, p1_0 * i1_2 + p1_2 * i1_0, a2_0);
                    a2_1 = fmaf(C_SQ2, p1_0 * i1_1 + p1_1 * i1_0, a2_1);
                    a2_2 += (-C_SQ6) * p1_0 * i1_0 + C_2SQ6 * p1_1 * i1_1 - C_SQ6 * p1_2 * i1_2;
                    a2_3 = fmaf(C_SQ2, p1_1 * i1_2 + p1_2 * i1_1, a2_3);
                    a2_4 = fmaf(C_SQ2, p1_2 * i1_2 - p1_0 * i1_0, a2_4);
                }
            }
        }
        __syncwarp(qmask);
    }

    float sq_l = 0.0f;
    if (active_q) {
        sq_l = a00*a00 + a01*a01 + a10_0*a10_0 + a10_1*a10_1 + a10_2*a10_2 + a11_0*a11_0 + a11_1*a11_1 + a11_2*a11_2;
        if constexpr (L2_QUAD) sq_l += a2_0*a2_0 + a2_1*a2_1 + a2_2*a2_2 + a2_3*a2_3 + a2_4*a2_4;
    }
    const float sq_tot = q4_sum_f(sq_l, qmask);
    const float f1 = a0_over_sig1 / sqrtf(1.0f + sq_tot * inv_DA2_sig1_kap_sq);
    if (slane == 0) f1_out[i] = f1;
    if (active_q) {
        const int c0 = 2 * n_msg, c1 = 2 * n_msg;
        float* d0 = A2.p[0] + (int64_t)i * c0 + slane;
        d0[0] = a00 * f1; d0[n_msg] = a01 * f1;
        float* d1 = A2.p[1] + (int64_t)i * 3 * c1 + slane;
        d1[0] = a10_0 * f1; d1[c1] = a10_1 * f1; d1[2*c1] = a10_2 * f1;
        d1[n_msg] = a11_0 * f1; d1[c1 + n_msg] = a11_1 * f1; d1[2*c1 + n_msg] = a11_2 * f1;
        if constexpr (L2_QUAD) {
            float* d2 = A2.p[2] + (int64_t)i * 5 * n_msg + slane;
            d2[0] = a2_0 * f1; d2[n_msg] = a2_1 * f1; d2[2*n_msg] = a2_2 * f1;
            d2[3*n_msg] = a2_3 * f1; d2[4*n_msg] = a2_4 * f1;
        }
    }
}

template <bool L2_QUAD>
__global__ void __launch_bounds__(128, 10) ilmlip_v1_n4_inv_fwd_kernel(
    int N, int n_msg, int r_gram, int n_inv1, int n_feat_pad, Int4 gram_off,
    const float* __restrict__ W_G1, const float* __restrict__ W_G2,
    CPtr4 A2, const float* __restrict__ mu_x, const float* __restrict__ inv_std_x,
    float* __restrict__ x, int8_t* __restrict__ eA_x,
    const float* __restrict__ am_l1 = nullptr) {
    __shared__ float sW1[8][5];
    __shared__ float sW2[L2_QUAD ? 4 : 1][L2_QUAD ? 5 : 1];

    const int c1 = 2 * n_msg;
    const int n_wg1 = c1 * r_gram;
    for (int idx = threadIdx.x; idx < n_wg1; idx += 128) {
        const int q = idx / r_gram, r = idx - q * r_gram;
        sW1[q][r] = __ldg(W_G1 + idx);
    }
    if constexpr (L2_QUAD) {
        const int n_wg2 = n_msg * r_gram;
        for (int idx = threadIdx.x; idx < n_wg2; idx += 128) {
            const int q = idx / r_gram, r = idx - q * r_gram;
            sW2[q][r] = __ldg(W_G2 + idx);
        }
    }
    __syncthreads();

    const int sid = threadIdx.x >> 2, slane = threadIdx.x & 3;
    const int qbase = (sid & 7) * 4;
    const unsigned qmask = 0xfu << qbase;
    const int i = blockIdx.x * 32 + sid;
    if (i >= N) return;

    float* xi = x + (int64_t)i * n_feat_pad;
    float am = 0.0f;
    if (am_l1) {
        if (slane == 0) am = __ldg(am_l1 + i);
    } else {
        for (int f = slane; f < n_inv1; f += 4) {
            const float v = (xi[f] - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
            xi[f] = v;
            am = fmaxf(am, fabsf(v));
        }
    }

    float a10_0 = 0.0f, a10_1 = 0.0f, a10_2 = 0.0f;
    float a11_0 = 0.0f, a11_1 = 0.0f, a11_2 = 0.0f;
    if (slane < n_msg) {
        const float* p0 = A2.p[0] + (int64_t)i * c1 + slane;
        const int f0 = n_inv1 + slane;
        const float v0 = (p0[0] - __ldg(mu_x + f0)) * __ldg(inv_std_x + f0);
        xi[f0] = v0; am = fmaxf(am, fabsf(v0));
        const int f1 = n_inv1 + n_msg + slane;
        const float v1 = (p0[n_msg] - __ldg(mu_x + f1)) * __ldg(inv_std_x + f1);
        xi[f1] = v1; am = fmaxf(am, fabsf(v1));

        const float* p1 = A2.p[1] + (int64_t)i * 3 * c1 + slane;
        a10_0 = p1[0]; a10_1 = p1[c1]; a10_2 = p1[2 * c1];
        a11_0 = p1[n_msg]; a11_1 = p1[c1 + n_msg]; a11_2 = p1[2 * c1 + n_msg];
    }

    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f;
    const bool active_r = (slane < r_gram);
    for (int q = 0; q < n_msg; ++q) {
        const float b00 = __shfl_sync(qmask, a10_0, q, 4);
        const float b01 = __shfl_sync(qmask, a10_1, q, 4);
        const float b02 = __shfl_sync(qmask, a10_2, q, 4);
        const float b10 = __shfl_sync(qmask, a11_0, q, 4);
        const float b11 = __shfl_sync(qmask, a11_1, q, 4);
        const float b12 = __shfl_sync(qmask, a11_2, q, 4);
        if (active_r) {
            const float w0 = sW1[q][slane];
            const float w1 = sW1[n_msg + q][slane];
            p0 = fmaf(b00, w0, fmaf(b10, w1, p0));
            p1 = fmaf(b01, w0, fmaf(b11, w1, p1));
            p2 = fmaf(b02, w0, fmaf(b12, w1, p2));
        }
    }

    constexpr float inv_s1 = 0.57735026918962576451f;
    int row_off = gram_off.v[1];
    for (int a = 0; a < r_gram; ++a) {
        const float pa0 = __shfl_sync(qmask, p0, a, 4);
        const float pa1 = __shfl_sync(qmask, p1, a, 4);
        const float pa2 = __shfl_sync(qmask, p2, a, 4);
        if (slane >= a && active_r) {
            const float s = fmaf(pa0, p0, fmaf(pa1, p1, pa2 * p2));
            const int f = row_off + (slane - a);
            const float v = (s * inv_s1 - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
            xi[f] = v;
            am = fmaxf(am, fabsf(v));
        }
        row_off += (r_gram - a);
    }

    if constexpr (L2_QUAD) {
        float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
        if (slane < n_msg) {
            const float* p2_g = A2.p[2] + (int64_t)i * 5 * n_msg + slane;
            a2_0 = p2_g[0]; a2_1 = p2_g[n_msg]; a2_2 = p2_g[2 * n_msg];
            a2_3 = p2_g[3 * n_msg]; a2_4 = p2_g[4 * n_msg];
        }
        float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f, q4 = 0.0f;
        for (int q = 0; q < n_msg; ++q) {
            const float c0 = __shfl_sync(qmask, a2_0, q, 4);
            const float c1v = __shfl_sync(qmask, a2_1, q, 4);
            const float c2 = __shfl_sync(qmask, a2_2, q, 4);
            const float c3 = __shfl_sync(qmask, a2_3, q, 4);
            const float c4 = __shfl_sync(qmask, a2_4, q, 4);
            if (active_r) {
                const float w = sW2[q][slane];
                q0 = fmaf(c0, w, q0); q1 = fmaf(c1v, w, q1); q2 = fmaf(c2, w, q2);
                q3 = fmaf(c3, w, q3); q4 = fmaf(c4, w, q4);
            }
        }
        constexpr float inv_s2 = 0.44721359549995793928f;
        int row_off2 = gram_off.v[2];
        for (int a = 0; a < r_gram; ++a) {
            const float qa0 = __shfl_sync(qmask, q0, a, 4);
            const float qa1 = __shfl_sync(qmask, q1, a, 4);
            const float qa2 = __shfl_sync(qmask, q2, a, 4);
            const float qa3 = __shfl_sync(qmask, q3, a, 4);
            const float qa4 = __shfl_sync(qmask, q4, a, 4);
            if (slane >= a && active_r) {
                const float s = fmaf(qa0, q0, fmaf(qa1, q1, fmaf(qa2, q2, fmaf(qa3, q3, qa4 * q4))));
                const int f = row_off2 + (slane - a);
                const float v = (s * inv_s2 - __ldg(mu_x + f)) * __ldg(inv_std_x + f);
                xi[f] = v;
                am = fmaxf(am, fabsf(v));
            }
            row_off2 += (r_gram - a);
        }
    }

    if (eA_x) {
        am = fmaxf(am, __shfl_xor_sync(qmask, am, 2));
        am = fmaxf(am, __shfl_xor_sync(qmask, am, 1));
        if (slane == 0) eA_x[i] = ilmlip_exp_from_am(am);
    }
}

template <bool L2_QUAD>
__global__ void __launch_bounds__(128, 8) ilmlip_v1_n4_inv_gate_bwd_pack_kernel(
    int N, int n_msg, int r_gram, int n_inv1, int n_feat_pad, float inv_DA2_a0_kap_sq, Int4 gram_off,
    const float* __restrict__ W_G1, const float* __restrict__ W_G2,
    const float* __restrict__ inv_std_x, float* __restrict__ dx,
    CPtr4 A2, const float* __restrict__ f1_arr, float* __restrict__ dA2_pack,
    bool skip_inv1 = false) {
    __shared__ float sW1[8][5];
    __shared__ float sW2[L2_QUAD ? 4 : 1][L2_QUAD ? 5 : 1];

    const int c1 = 2 * n_msg;
    const int n_wg1 = c1 * r_gram;
    for (int idx = threadIdx.x; idx < n_wg1; idx += 128) {
        const int q = idx / r_gram, r = idx - q * r_gram;
        sW1[q][r] = __ldg(W_G1 + idx);
    }
    if constexpr (L2_QUAD) {
        const int n_wg2 = n_msg * r_gram;
        for (int idx = threadIdx.x; idx < n_wg2; idx += 128) {
            const int q = idx / r_gram, r = idx - q * r_gram;
            sW2[q][r] = __ldg(W_G2 + idx);
        }
    }
    __syncthreads();

    const int sid = threadIdx.x >> 2, slane = threadIdx.x & 3;
    const int qbase = (sid & 7) * 4;
    const unsigned qmask = 0xfu << qbase;
    const int i = blockIdx.x * 32 + sid;
    if (i >= N) return;

    float* dxi = dx + (int64_t)i * n_feat_pad;
    if (!skip_inv1) {
        for (int f = slane; f < n_inv1; f += 4) dxi[f] *= __ldg(inv_std_x + f);
    }

    const bool active_q = (slane < n_msg);
    const bool active_r = (slane < r_gram);
    float a00 = 0.0f, a01 = 0.0f, d00 = 0.0f, d01 = 0.0f;
    float a10_0 = 0.0f, a10_1 = 0.0f, a10_2 = 0.0f, a11_0 = 0.0f, a11_1 = 0.0f, a11_2 = 0.0f;
    if (active_q) {
        const float* p0 = A2.p[0] + (int64_t)i * c1 + slane;
        a00 = p0[0]; a01 = p0[n_msg];
        d00 = dxi[n_inv1 + slane] * __ldg(inv_std_x + n_inv1 + slane);
        d01 = dxi[n_inv1 + n_msg + slane] * __ldg(inv_std_x + n_inv1 + n_msg + slane);
        const float* p1 = A2.p[1] + (int64_t)i * 3 * c1 + slane;
        a10_0 = p1[0]; a10_1 = p1[c1]; a10_2 = p1[2 * c1];
        a11_0 = p1[n_msg]; a11_1 = p1[c1 + n_msg]; a11_2 = p1[2 * c1 + n_msg];
    }

    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f;
    for (int q = 0; q < n_msg; ++q) {
        const float b00 = __shfl_sync(qmask, a10_0, q, 4);
        const float b01 = __shfl_sync(qmask, a10_1, q, 4);
        const float b02 = __shfl_sync(qmask, a10_2, q, 4);
        const float b10 = __shfl_sync(qmask, a11_0, q, 4);
        const float b11 = __shfl_sync(qmask, a11_1, q, 4);
        const float b12 = __shfl_sync(qmask, a11_2, q, 4);
        if (active_r) {
            const float w0 = sW1[q][slane];
            const float w1 = sW1[n_msg + q][slane];
            p0 = fmaf(b00, w0, fmaf(b10, w1, p0));
            p1 = fmaf(b01, w0, fmaf(b11, w1, p1));
            p2 = fmaf(b02, w0, fmaf(b12, w1, p2));
        }
    }

    float dp0 = 0.0f, dp1 = 0.0f, dp2 = 0.0f;
    {
        constexpr float inv_s1 = 0.57735026918962576451f;
        const int goff1 = gram_off.v[1];
        for (int a = 0; a < r_gram; ++a) {
            const float pa0 = __shfl_sync(qmask, p0, a, 4);
            const float pa1 = __shfl_sync(qmask, p1, a, 4);
            const float pa2 = __shfl_sync(qmask, p2, a, 4);
            if (active_r) {
                const int lo = (a < slane) ? a : slane;
                const int hi = (a < slane) ? slane : a;
                const int f = goff1 + lo * r_gram - ((lo * (lo - 1)) >> 1) + (hi - lo);
                float g = dxi[f] * __ldg(inv_std_x + f) * inv_s1;
                if (a == slane) g *= 2.0f;
                dp0 = fmaf(g, pa0, dp0); dp1 = fmaf(g, pa1, dp1); dp2 = fmaf(g, pa2, dp2);
            }
        }
    }

    float d10_0 = 0.0f, d10_1 = 0.0f, d10_2 = 0.0f, d11_0 = 0.0f, d11_1 = 0.0f, d11_2 = 0.0f;
    for (int r = 0; r < r_gram; ++r) {
        const float r0 = __shfl_sync(qmask, dp0, r, 4);
        const float r1 = __shfl_sync(qmask, dp1, r, 4);
        const float r2 = __shfl_sync(qmask, dp2, r, 4);
        if (active_q) {
            const float w0 = sW1[slane][r];
            const float w1 = sW1[n_msg + slane][r];
            d10_0 = fmaf(r0, w0, d10_0); d10_1 = fmaf(r1, w0, d10_1); d10_2 = fmaf(r2, w0, d10_2);
            d11_0 = fmaf(r0, w1, d11_0); d11_1 = fmaf(r1, w1, d11_1); d11_2 = fmaf(r2, w1, d11_2);
        }
    }
    float dot_l = 0.0f;
    if (active_q) {
        dot_l = a00*d00 + a01*d01 + a10_0*d10_0 + a10_1*d10_1 + a10_2*d10_2 + a11_0*d11_0 + a11_1*d11_1 + a11_2*d11_2;
    }

    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f, a2_4 = 0.0f;
    float d2_0 = 0.0f, d2_1 = 0.0f, d2_2 = 0.0f, d2_3 = 0.0f, d2_4 = 0.0f;
    if constexpr (L2_QUAD) {
        if (active_q) {
            const float* p2_g = A2.p[2] + (int64_t)i * 5 * n_msg + slane;
            a2_0 = p2_g[0]; a2_1 = p2_g[n_msg]; a2_2 = p2_g[2 * n_msg];
            a2_3 = p2_g[3 * n_msg]; a2_4 = p2_g[4 * n_msg];
        }
        float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f, q4 = 0.0f;
        for (int q = 0; q < n_msg; ++q) {
            const float c0 = __shfl_sync(qmask, a2_0, q, 4);
            const float c1v = __shfl_sync(qmask, a2_1, q, 4);
            const float c2 = __shfl_sync(qmask, a2_2, q, 4);
            const float c3 = __shfl_sync(qmask, a2_3, q, 4);
            const float c4 = __shfl_sync(qmask, a2_4, q, 4);
            if (active_r) {
                const float w = sW2[q][slane];
                q0 = fmaf(c0, w, q0); q1 = fmaf(c1v, w, q1); q2 = fmaf(c2, w, q2);
                q3 = fmaf(c3, w, q3); q4 = fmaf(c4, w, q4);
            }
        }
        float dq0 = 0.0f, dq1 = 0.0f, dq2 = 0.0f, dq3 = 0.0f, dq4 = 0.0f;
        {
            constexpr float inv_s2 = 0.44721359549995793928f;
            const int goff2 = gram_off.v[2];
            for (int a = 0; a < r_gram; ++a) {
                const float qa0 = __shfl_sync(qmask, q0, a, 4);
                const float qa1 = __shfl_sync(qmask, q1, a, 4);
                const float qa2 = __shfl_sync(qmask, q2, a, 4);
                const float qa3 = __shfl_sync(qmask, q3, a, 4);
                const float qa4 = __shfl_sync(qmask, q4, a, 4);
                if (active_r) {
                    const int lo = (a < slane) ? a : slane;
                    const int hi = (a < slane) ? slane : a;
                    const int f = goff2 + lo * r_gram - ((lo * (lo - 1)) >> 1) + (hi - lo);
                    float g = dxi[f] * __ldg(inv_std_x + f) * inv_s2;
                    if (a == slane) g *= 2.0f;
                    dq0 = fmaf(g, qa0, dq0); dq1 = fmaf(g, qa1, dq1); dq2 = fmaf(g, qa2, dq2);
                    dq3 = fmaf(g, qa3, dq3); dq4 = fmaf(g, qa4, dq4);
                }
            }
        }
        for (int r = 0; r < r_gram; ++r) {
            const float r0 = __shfl_sync(qmask, dq0, r, 4);
            const float r1 = __shfl_sync(qmask, dq1, r, 4);
            const float r2 = __shfl_sync(qmask, dq2, r, 4);
            const float r3 = __shfl_sync(qmask, dq3, r, 4);
            const float r4 = __shfl_sync(qmask, dq4, r, 4);
            if (active_q) {
                const float w = sW2[slane][r];
                d2_0 = fmaf(r0, w, d2_0); d2_1 = fmaf(r1, w, d2_1); d2_2 = fmaf(r2, w, d2_2);
                d2_3 = fmaf(r3, w, d2_3); d2_4 = fmaf(r4, w, d2_4);
            }
        }
        if (active_q) dot_l += a2_0*d2_0 + a2_1*d2_1 + a2_2*d2_2 + a2_3*d2_3 + a2_4*d2_4;
    }

    const float dot = q4_sum_f(dot_l, qmask);
    const float f1 = f1_arr[i];
    const float sc = (dot * f1) * inv_DA2_a0_kap_sq;
    if (active_q) {
        constexpr float C110 = 0.57735026918962576451f;
        const int64_t atom_stride = L2_QUAD ? 64 : 32;  // 256 B or 128 B (100% 128B cache-line aligned!)
        float* base = dA2_pack + (int64_t)i * atom_stride;
        float4* dst4 = reinterpret_cast<float4*>(base) + slane;
        dst4[0] = make_float4(fmaf(f1, d00, -sc * a00), C110 * fmaf(f1, d01, -sc * a01),
                              fmaf(f1, d10_0, -sc * a10_0), fmaf(f1, d10_1, -sc * a10_1));
        dst4[4] = make_float4(fmaf(f1, d10_2, -sc * a10_2), fmaf(f1, d11_0, -sc * a11_0),
                              fmaf(f1, d11_1, -sc * a11_1), fmaf(f1, d11_2, -sc * a11_2));
        if constexpr (L2_QUAD) {
            constexpr float C_SQ2 = 0.70710678118654752440f;
            constexpr float C_SQ6 = 0.40824829046386301637f;
            const float g8  = fmaf(f1, d2_0, -sc * a2_0);
            const float g9  = fmaf(f1, d2_1, -sc * a2_1);
            const float g10 = fmaf(f1, d2_2, -sc * a2_2);
            const float g11 = fmaf(f1, d2_3, -sc * a2_3);
            const float g12 = fmaf(f1, d2_4, -sc * a2_4);
            dst4[8] = make_float4(-C_SQ6 * g10 - C_SQ2 * g12, C_SQ2 * g9, C_SQ2 * g8, C_SQ2 * g11);
            base[48 + slane] = -C_SQ6 * g10 + C_SQ2 * g12;
        }
    }
}

template <bool L2_QUAD>
__global__ void __launch_bounds__(128) ilmlip_gate_bwd_pack_l2_v1_n4_kernel(
    int N, int n_msg, float inv_DA2_a0_kap_sq,
    CPtr4 A2, CPtr4 dA2, const float* __restrict__ f1_arr, float* __restrict__ dA2_pack) {
    const int sid = threadIdx.x >> 2, slane = threadIdx.x & 3;
    const int qbase = (sid & 7) * 4;
    const unsigned qmask = 0xfu << qbase;
    const int i = blockIdx.x * 32 + sid;
    if (i >= N) return;
    const bool active_q = (slane < n_msg);
    float a00=0,a01=0,a10_0=0,a10_1=0,a10_2=0,a11_0=0,a11_1=0,a11_2=0;
    float d00=0,d01=0,d10_0=0,d10_1=0,d10_2=0,d11_0=0,d11_1=0,d11_2=0;
    float a2_0=0,a2_1=0,a2_2=0,a2_3=0,a2_4=0,d2_0=0,d2_1=0,d2_2=0,d2_3=0,d2_4=0;
    float dot_l = 0.0f;
    if (active_q) {
        const int c = 2 * n_msg;
        const float* pA0 = A2.p[0] + (int64_t)i * c + slane;
        const float* pd0 = dA2.p[0] + (int64_t)i * c + slane;
        const float* pA1 = A2.p[1] + (int64_t)i * 3 * c + slane;
        const float* pd1 = dA2.p[1] + (int64_t)i * 3 * c + slane;
        a00 = pA0[0]; a01 = pA0[n_msg]; d00 = pd0[0]; d01 = pd0[n_msg];
        a10_0 = pA1[0]; a10_1 = pA1[c]; a10_2 = pA1[2*c];
        d10_0 = pd1[0]; d10_1 = pd1[c]; d10_2 = pd1[2*c];
        a11_0 = pA1[n_msg]; a11_1 = pA1[c + n_msg]; a11_2 = pA1[2*c + n_msg];
        d11_0 = pd1[n_msg]; d11_1 = pd1[c + n_msg]; d11_2 = pd1[2*c + n_msg];
        dot_l = a00*d00 + a01*d01 + a10_0*d10_0 + a10_1*d10_1 + a10_2*d10_2 + a11_0*d11_0 + a11_1*d11_1 + a11_2*d11_2;
        if constexpr (L2_QUAD) {
            const float* pA2 = A2.p[2] + (int64_t)i * 5 * n_msg + slane;
            const float* pd2 = dA2.p[2] + (int64_t)i * 5 * n_msg + slane;
            a2_0 = pA2[0]; a2_1 = pA2[n_msg]; a2_2 = pA2[2*n_msg]; a2_3 = pA2[3*n_msg]; a2_4 = pA2[4*n_msg];
            d2_0 = pd2[0]; d2_1 = pd2[n_msg]; d2_2 = pd2[2*n_msg]; d2_3 = pd2[3*n_msg]; d2_4 = pd2[4*n_msg];
            dot_l += a2_0*d2_0 + a2_1*d2_1 + a2_2*d2_2 + a2_3*d2_3 + a2_4*d2_4;
        }
    }
    const float dot = q4_sum_f(dot_l, qmask);
    const float f1 = f1_arr[i];
    const float sc = (dot * f1) * inv_DA2_a0_kap_sq;
    if (active_q) {
        constexpr float C110 = 0.57735026918962576451f;
        const int64_t atom_stride = L2_QUAD ? 64 : 32;
        float* base = dA2_pack + (int64_t)i * atom_stride;
        float4* dst4 = reinterpret_cast<float4*>(base) + slane;
        dst4[0] = make_float4(fmaf(f1, d00, -sc * a00), C110 * fmaf(f1, d01, -sc * a01),
                              fmaf(f1, d10_0, -sc * a10_0), fmaf(f1, d10_1, -sc * a10_1));
        dst4[4] = make_float4(fmaf(f1, d10_2, -sc * a10_2), fmaf(f1, d11_0, -sc * a11_0),
                              fmaf(f1, d11_1, -sc * a11_1), fmaf(f1, d11_2, -sc * a11_2));
        if constexpr (L2_QUAD) {
            constexpr float C_SQ2 = 0.70710678118654752440f;
            constexpr float C_SQ6 = 0.40824829046386301637f;
            const float g8  = fmaf(f1, d2_0, -sc * a2_0);
            const float g9  = fmaf(f1, d2_1, -sc * a2_1);
            const float g10 = fmaf(f1, d2_2, -sc * a2_2);
            const float g11 = fmaf(f1, d2_3, -sc * a2_3);
            const float g12 = fmaf(f1, d2_4, -sc * a2_4);
            dst4[8] = make_float4(-C_SQ6 * g10 - C_SQ2 * g12, C_SQ2 * g9, C_SQ2 * g8, C_SQ2 * g11);
            base[48 + slane] = -C_SQ6 * g10 + C_SQ2 * g12;
        }
    }
}

template <int NB2, int NB1, bool SAME_G, bool L2_QUAD, bool REM_BF, bool STRESS, bool F32_F>
__global__ void __launch_bounds__(128, SAME_G ? 5 : 4) ilmlip_l2_edge_bwd_v1_n4_sender_kernel(
    int N, int S, int n_msg, ChebParam r_max2, ChebParam r_max1,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end, const int* __restrict__ senders,
    const float4* __restrict__ vec_len, const int* __restrict__ species,
    const float* __restrict__ C2, const float* __restrict__ Cbf,
    const float* __restrict__ f0, const float4* __restrict__ I_pack4, CPtr4 I_node, const float* __restrict__ dA2_pack,
    Ptr4 dI_node, float* __restrict__ df0_bf,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    using FT = typename std::conditional<F32_F, float, double>::type;
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int NB1_4 = (NB1 + 3) / 4;
    constexpr int EB = SAME_G ? 4 : 2;
    constexpr bool DUAL_G = REM_BF && !SAME_G;
    extern __shared__ float4 smem_c4[];
    __shared__ float4 s_g[32][EB][NB2_4];
    __shared__ float4 s_dg[32][EB][NB2_4];
    __shared__ float4 s_g1[DUAL_G ? 32 : 1][DUAL_G ? EB : 1][DUAL_G ? NB1_4 : 1];
    __shared__ float4 s_dg1[DUAL_G ? 32 : 1][DUAL_G ? EB : 1][DUAL_G ? NB1_4 : 1];
    __shared__ float4 s_Y1[32][EB];
    __shared__ int2   s_ip[32][EB];
    __shared__ FT     s_sf[32][3];

    const int l2_tot_cols = 2 * n_msg;
    const int n_slots = S * S * 2 * 4;
    for (int idx = threadIdx.x; idx < n_slots; idx += 128) {
        const int q = idx & 3;
        const int pl = idx >> 2;
        const int pair = pl >> 1;
        const int l = pl & 1;
        #pragma unroll
        for (int b = 0; b < NB2_4; ++b) {
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (q < n_msg) {
                const int k0 = b * 4;
                const float* src = C2 + (pair * NB2 + k0) * l2_tot_cols + l * n_msg + q;
                v.x = (k0 + 0 < NB2) ? __ldg(src + 0 * l2_tot_cols) : 0.0f;
                v.y = (k0 + 1 < NB2) ? __ldg(src + 1 * l2_tot_cols) : 0.0f;
                v.z = (k0 + 2 < NB2) ? __ldg(src + 2 * l2_tot_cols) : 0.0f;
                v.w = (k0 + 3 < NB2) ? __ldg(src + 3 * l2_tot_cols) : 0.0f;
            }
            smem_c4[(pair * (2 * NB2_4) + l * NB2_4 + b) * 4 + q] = v;
        }
    }
    float4* s_bf4 = smem_c4 + S * S * (2 * NB2_4) * 4;
    if constexpr (REM_BF) {
        for (int idx = threadIdx.x; idx < n_slots; idx += 128) {
            const int q = idx & 3;
            const int pl = idx >> 2;
            const int pair = pl >> 1;
            const int l = pl & 1;
            #pragma unroll
            for (int b = 0; b < NB1_4; ++b) {
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (q < n_msg) {
                    const int k0 = b * 4;
                    const float* src = Cbf + (pair * NB1 + k0) * l2_tot_cols + l * n_msg + q;
                    v.x = (k0 + 0 < NB1) ? __ldg(src + 0 * l2_tot_cols) : 0.0f;
                    v.y = (k0 + 1 < NB1) ? __ldg(src + 1 * l2_tot_cols) : 0.0f;
                    v.z = (k0 + 2 < NB1) ? __ldg(src + 2 * l2_tot_cols) : 0.0f;
                    v.w = (k0 + 3 < NB1) ? __ldg(src + 3 * l2_tot_cols) : 0.0f;
                }
                s_bf4[(pair * (2 * NB1_4) + l * NB1_4 + b) * 4 + q] = v;
            }
        }
    }
    __syncthreads();

    const int sid = threadIdx.x >> 2, slane = threadIdx.x & 3;
    const int qbase = (sid & 7) * 4;
    const unsigned qmask = 0xfu << qbase;
    const int j = blockIdx.x * 32 + sid;
    if (j >= N) return;

    const int z_j = species[j];
    const int eb = row_ptr[j], ee = row_end[j];
    const bool active_q = (slane < n_msg);
    const float f0_j = REM_BF ? __ldg(f0 + j) : 0.0f;
    float i0_raw = 0.0f, i1_0_raw = 0.0f, i1_1_raw = 0.0f, i1_2_raw = 0.0f;
    if (active_q) {
        if (I_pack4) {
            const float4 ir = __ldg(I_pack4 + (int64_t)j * n_msg + slane);
            i0_raw = ir.x; i1_0_raw = ir.y; i1_1_raw = ir.z; i1_2_raw = ir.w;
        } else {
            i0_raw = __ldg(I_node.p[0] + (int64_t)j * n_msg + slane);
            const float* pI1 = I_node.p[1] + (int64_t)j * 3 * n_msg + slane;
            i1_0_raw = __ldg(pI1); i1_1_raw = __ldg(pI1 + n_msg); i1_2_raw = __ldg(pI1 + 2 * n_msg);
        }
    }
    if (slane < 3) s_sf[sid][slane] = (FT)0;

    float acc_di0 = 0.0f, acc_di1_0 = 0.0f, acc_di1_1 = 0.0f, acc_di1_2 = 0.0f, acc_df0 = 0.0f;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (slane < bcnt) {
            const int e = eb_b + slane;
            const float4 vl_rev = __ldg(vec_len + e);
            const int i_nb = __ldg(senders + e);
            const float r = fmaxf(vl_rev.w, 1e-8f);
            const float inv_d = 1.0f / r;
            const float sc = -1.7320508075688772935f * inv_d;
            s_ip[sid][slane] = make_int2(i_nb, __ldg(species + i_nb) * S + z_j);
            s_Y1[sid][slane] = make_float4(vl_rev.x * sc, vl_rev.y * sc, vl_rev.z * sc, inv_d);
            eval_cheb_nb_store<NB2, true>(r, r_max2, s_g[sid][slane], s_dg[sid][slane]);
        } else if constexpr (DUAL_G) {
            if (slane >= 2 && slane < 2 + bcnt) {
                const int bi = slane - 2;
                const float4 vl_rev = __ldg(vec_len + eb_b + bi);
                const float r = fmaxf(vl_rev.w, 1e-8f);
                eval_cheb_nb_store<NB1, true>(r, r_max1, s_g1[sid][bi], s_dg1[sid][bi]);
            }
        }
        __syncwarp(qmask);

        for (int bi = 0; bi < bcnt; ++bi) {
            float er_lane = 0.0f, dY1 = 0.0f, dY2 = 0.0f, dY3 = 0.0f;
            if (active_q) {
                const int2 ip = s_ip[sid][bi];
                const int i_nb = ip.x, pair = ip.y;
                const float4 Y1 = s_Y1[sid][bi];
                const float4* cp4 = smem_c4 + (pair * (2 * NB2_4)) * 4 + slane;
                float r2_0 = 0.0f, dr2_0 = 0.0f, r2_1 = 0.0f, dr2_1 = 0.0f;
                #pragma unroll
                for (int b = 0; b < NB2_4; ++b) {
                    const float4 gv = s_g[sid][bi][b], dgv = s_dg[sid][bi][b];
                    const float4 c0 = cp4[b * 4];
                    r2_0  += v1_dot4(c0, gv);
                    dr2_0 += v1_dot4(c0, dgv);
                    const float4 c1 = cp4[(NB2_4 + b) * 4];
                    r2_1  += v1_dot4(c1, gv);
                    dr2_1 += v1_dot4(c1, dgv);
                }

                float rb0 = 0.0f, rb1 = 0.0f, drb0 = 0.0f, drb1 = 0.0f;
                if constexpr (REM_BF) {
                    const float4* bp4 = s_bf4 + (pair * (2 * NB1_4)) * 4 + slane;
                    #pragma unroll
                    for (int b = 0; b < NB1_4; ++b) {
                        const float4 gv  = DUAL_G ? s_g1[sid][bi][b]  : s_g[sid][bi][b];
                        const float4 dgv = DUAL_G ? s_dg1[sid][bi][b] : s_dg[sid][bi][b];
                        const float4 b0 = bp4[b * 4];
                        rb0  += v1_dot4(b0, gv);
                        drb0 += v1_dot4(b0, dgv);
                        const float4 b1 = bp4[(NB1_4 + b) * 4];
                        rb1  += v1_dot4(b1, gv);
                        drb1 += v1_dot4(b1, dgv);
                    }
                }
                const float i0   = i0_raw   - (REM_BF ? (f0_j * rb0) : 0.0f);
                const float b1s  = REM_BF ? (f0_j * rb1) : 0.0f;
                const float i1_0 = i1_0_raw - b1s * Y1.x;
                const float i1_1 = i1_1_raw - b1s * Y1.y;
                const float i1_2 = i1_2_raw - b1s * Y1.z;

                const int64_t atom_stride = L2_QUAD ? 64 : 32;
                const float* base = dA2_pack + (int64_t)i_nb * atom_stride;
                const float4* p4 = reinterpret_cast<const float4*>(base) + slane;
                const float4 q0 = __ldg(p4);
                const float4 q1 = __ldg(p4 + 4);

                const float dp0 = fmaf(q0.x, i0, q0.z * i1_0 + q0.w * i1_1 + q1.x * i1_2);
                float dp1_0 = fmaf(q0.y, i1_0, q1.y * i0);
                float dp1_1 = fmaf(q0.y, i1_1, q1.z * i0);
                float dp1_2 = fmaf(q0.y, i1_2, q1.w * i0);

                const float di0   = fmaf(q0.x, r2_0, r2_1 * (q1.y * Y1.x + q1.z * Y1.y + q1.w * Y1.z));
                const float g01r1 = q0.y * r2_1;
                float di1_0 = fmaf(q0.z, r2_0, g01r1 * Y1.x);
                float di1_1 = fmaf(q0.w, r2_0, g01r1 * Y1.y);
                float di1_2 = fmaf(q1.x, r2_0, g01r1 * Y1.z);

                if constexpr (L2_QUAD) {
                    const float4 q2 = __ldg(p4 + 8);
                    const float w22 = __ldg(base + 48 + slane);
                    const float w11 = -(q2.x + w22);
                    dp1_0 += q2.x * i1_0 + q2.y * i1_1 + q2.z * i1_2;
                    dp1_1 += q2.y * i1_0 + w11  * i1_1 + q2.w * i1_2;
                    dp1_2 += q2.z * i1_0 + q2.w * i1_1 + w22  * i1_2;
                    di1_0 = fmaf(r2_1, q2.x * Y1.x + q2.y * Y1.y + q2.z * Y1.z, di1_0);
                    di1_1 = fmaf(r2_1, q2.y * Y1.x + w11  * Y1.y + q2.w * Y1.z, di1_1);
                    di1_2 = fmaf(r2_1, q2.z * Y1.x + q2.w * Y1.y + w22  * Y1.z, di1_2);
                }

                const float d_r2_1 = dp1_0 * Y1.x + dp1_1 * Y1.y + dp1_2 * Y1.z;
                er_lane = fmaf(dp0, dr2_0, d_r2_1 * dr2_1);
                dY1 = dp1_0 * r2_1; dY2 = dp1_1 * r2_1; dY3 = dp1_2 * r2_1;
                acc_di0 += di0; acc_di1_0 += di1_0; acc_di1_1 += di1_1; acc_di1_2 += di1_2;
                if constexpr (REM_BF) {
                    acc_df0 = fmaf(-di0, rb0, acc_df0);
                    er_lane = fmaf(-di0 * f0_j, drb0, er_lane);
                    const float ydot1 = di1_0 * Y1.x + di1_1 * Y1.y + di1_2 * Y1.z;
                    acc_df0 = fmaf(-ydot1, rb1, acc_df0);
                    er_lane = fmaf(-ydot1 * f0_j, drb1, er_lane);
                    const float bf1s = -f0_j * rb1;
                    dY1 = fmaf(di1_0, bf1s, dY1); dY2 = fmaf(di1_1, bf1s, dY2); dY3 = fmaf(di1_2, bf1s, dY3);
                }
            }
            const bool p0 = (slane & 1) != 0;
            const float u0 = (p0 ? dY1 : er_lane) + __shfl_xor_sync(qmask, p0 ? er_lane : dY1, 1);
            const float u1 = (p0 ? dY3 : dY2)     + __shfl_xor_sync(qmask, p0 ? dY2 : dY3, 1);
            const bool p1 = (slane & 2) != 0;
            const float v = (p1 ? u1 : u0) + __shfl_xor_sync(qmask, p1 ? u0 : u1, 2);
            reinterpret_cast<float*>(&s_dg[sid][bi][0])[slane] = v;
        }
        __syncwarp(qmask);

        FT bfx = (FT)0, bfy = (FT)0, bfz = (FT)0;
        if (slane < bcnt) {
            const int my_inb = s_ip[sid][slane].x;
            const float4 y1i = s_Y1[sid][slane];
            constexpr float S3 = 1.7320508075688772935f;
            constexpr float INV_S3 = 0.57735026918962576451f;
            const float ux = y1i.x * INV_S3, uy = y1i.y * INV_S3, uz = y1i.z * INV_S3;
            const float inv_d = y1i.w;
            const float4 ed = s_dg[sid][slane][0];
            const float gx = S3 * ed.y, gy = S3 * ed.z, gz = S3 * ed.w;
            const float dot = ux * gx + uy * gy + uz * gz;
            bfx = (FT)(ux * ed.x + (gx - ux * dot) * inv_d);
            bfy = (FT)(uy * ed.x + (gy - uy * dot) * inv_d);
            bfz = (FT)(uz * ed.x + (gz - uz * dot) * inv_d);
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
                const float4 vl_rev = __ldg(vec_len + eb_b + slane);
                const float vx = -vl_rev.x, vy = -vl_rev.y, vz = -vl_rev.z;
                v0 += (double)vx * (double)bfx;
                v1 += 0.5 * ((double)vx * (double)bfy + (double)vy * (double)bfx);
                v2 += 0.5 * ((double)vx * (double)bfz + (double)vz * (double)bfx);
                v3 += (double)vy * (double)bfy;
                v4 += 0.5 * ((double)vy * (double)bfz + (double)vz * (double)bfy);
                v5 += (double)vz * (double)bfz;
            }
        }
        if constexpr (F32_F) {
            bfx = q4_sum_f(bfx, qmask); bfy = q4_sum_f(bfy, qmask); bfz = q4_sum_f(bfz, qmask);
        } else {
            bfx = q4_sum_d(bfx, qmask); bfy = q4_sum_d(bfy, qmask); bfz = q4_sum_d(bfz, qmask);
        }
        if (slane == 0) { s_sf[sid][0] += bfx; s_sf[sid][1] += bfy; s_sf[sid][2] += bfz; }
        __syncwarp(qmask);
    }

    if (active_q) {
        dI_node.p[0][(int64_t)j * n_msg + slane] = acc_di0;
        float* dI1 = dI_node.p[1] + (int64_t)j * 3 * n_msg + slane;
        dI1[0] = acc_di1_0; dI1[n_msg] = acc_di1_1; dI1[2 * n_msg] = acc_di1_2;
    }
    if constexpr (REM_BF) {
        const float df0_tot = q4_sum_f(acc_df0, qmask);
        if (slane == 0) df0_bf[j] = df0_tot;
    }
    if constexpr (STRESS) {
        v0 = q4_sum_d(v0, qmask); v1 = q4_sum_d(v1, qmask); v2 = q4_sum_d(v2, qmask);
        v3 = q4_sum_d(v3, qmask); v4 = q4_sum_d(v4, qmask); v5 = q4_sum_d(v5, qmask);
    }
    if (slane == 0) {
        const FT sfx_acc = s_sf[sid][0], sfy_acc = s_sf[sid][1], sfz_acc = s_sf[sid][2];
        if (sfx_acc != (FT)0 || sfy_acc != (FT)0 || sfz_acc != (FT)0) {
            if constexpr (F32_F) {
                atomicAdd(forces_f32 + (int64_t)j * 3 + 0, sfx_acc);
                atomicAdd(forces_f32 + (int64_t)j * 3 + 1, sfy_acc);
                atomicAdd(forces_f32 + (int64_t)j * 3 + 2, sfz_acc);
            } else {
                atomicAdd(forces + (int64_t)j * 3 + 0, sfx_acc);
                atomicAdd(forces + (int64_t)j * 3 + 1, sfy_acc);
                atomicAdd(forces + (int64_t)j * 3 + 2, sfz_acc);
            }
        }
        if constexpr (STRESS) {
            if (v0 != 0.0) atomicAdd(virial + 0, v0); if (v1 != 0.0) atomicAdd(virial + 1, v1);
            if (v2 != 0.0) atomicAdd(virial + 2, v2); if (v3 != 0.0) atomicAdd(virial + 3, v3);
            if (v4 != 0.0) atomicAdd(virial + 4, v4); if (v5 != 0.0) atomicAdd(virial + 5, v5);
        }
    }
}
