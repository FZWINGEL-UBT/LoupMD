#pragma once
// Ultra-fast 4-lane subwarp (32 atoms/CTA!) symmetric LR-SK-Fast CUDA kernels:
//   - Shared multipoles (sym=True, c0<=4, c1<=4, c2<=4) packed into 1x float4 (64 B/atom -> 1 LDG.128/edge!)
//   - Pre-symmetrized pair weights (w2b, w0, w1s, w1p, w2s, w2p, w2d) + pre-packed rev asymmetric weights
//     (w01_r, w02_r, w12s_r, w12p_r) in bank-conflict-free SMEM [pair, NB2_4, N_T4, 4]
//   - Zero-atomic forces and multipole gradients in ONE 4-lane pass (pol=False) or 2 fast passes (pol=True)
//   - SMEM-staged heads fwd/bwd + hgemm2-accelerated polarization MLP!

namespace {

struct LrskFastBufs {
    float4* qp_pack4;     // [N, 4]: (q, px, py, pz)
    float4* Q_pack03;     // [N, 4]: (Q0, Q1, Q2, Q3) (nullptr if c2==0)
    float*  Q_pack4;      // [N, 4]: Q4 (nullptr if c2==0)
    float*  g1;           // [N, 4]
    float*  g2;           // [N, 4]
    float*  e_site;       // [N]
    float*  e_lr;         // [N]
    float*  sw;           // [N]
    float4* dqp_pack4;    // [N, 4]: (dq, dpx, dpy, dpz)
    float4* dQ_pack03;    // [N, 4]
    float*  dQ_pack4;     // [N, 4]
    float4* pol_ef4;      // [N, 4]: (Phi, Efx, Efy, Efz)
    float4* dpol_ef4;     // [N, 4]: (dPhi, dEfx, dEfy, dEfz)
};

// =================================================================================================
// Kernel F1: SMEM-Staged Fast Symmetric Heads Forward -> e_site, sw, qp_pack4, Q_pack03/Q_pack4
//   16 lanes/atom (8 atoms/CTA). SMEM caches W_head[S, ro_width, 16] (<=16 KB for S<=4, ro_width<=64).
// =================================================================================================
template <bool ZBL_SW, bool HAS_C2, bool SMEM_W>
__global__ void __launch_bounds__(128) lrsk_heads_fwd_sym_kernel(
    int N, int S, int ro_width, int i_stride, LrskCfg cfg,
    const int* __restrict__ species,
    const float* __restrict__ h,          // [N, ro_width]
    const float* __restrict__ W_head_p16, // [S, ro_width, 16]
    const float* __restrict__ b_head_p16, // [S, 16]
    const float* __restrict__ I_raw1,     // [N, 3, i_stride]
    const float* __restrict__ I_raw2,     // [N, 5, i_stride]
    const double* __restrict__ e_zbl,
    LrskFastBufs buf) {
    extern __shared__ float s_Wh[];       // [S * ro_width * 16] if SMEM_W
    __shared__ float s_h[8][128];
    __shared__ float s_q[8][4];

    const int tid = threadIdx.x;
    if constexpr (SMEM_W) {
        const int tot = S * ro_width * 16;
        for (int idx = tid; idx < tot; idx += 128) s_Wh[idx] = __ldg(W_head_p16 + idx);
        __syncthreads();
    }
    const float* Wp_base = SMEM_W ? s_Wh : W_head_p16;

    const int sid = tid >> 4;
    const int r = tid & 15;
    const unsigned smask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;

    const float* hi = h + (int64_t)i * ro_width;
    if ((ro_width & 3) == 0 && (ro_width <= 64)) {
        const int rw4 = ro_width >> 2;
        if (r < rw4) reinterpret_cast<float4*>(s_h[sid])[r] = __ldg(reinterpret_cast<const float4*>(hi) + r);
    } else {
        for (int k = r; k < ro_width; k += 16) s_h[sid][k] = __ldg(hi + k);
    }
    if (r < 4) s_q[sid][r] = 0.0f;
    __syncwarp(smask);

    const int z = species[i];
    const int nh = cfg.n_head_out;
    float acc = 0.0f;
    if (r < nh) {
        acc = __ldg(b_head_p16 + (int64_t)z * 16 + r);
        const float* Wz = Wp_base + (int64_t)z * ro_width * 16 + r;
        #pragma unroll 4
        for (int k = 0; k < ro_width; ++k) {
            const float w = SMEM_W ? Wz[k * 16] : __ldg(Wz + k * 16);
            acc = fmaf(s_h[sid][k], w, acc);
        }
    }

    if (r == 0) {
        buf.e_site[i] = acc;
        if constexpr (ZBL_SW) {
            double xc = e_zbl[i] * (double)cfg.inv_zc;
            if (xc < 0.0) xc = 0.0;
            if (xc > 1.0) xc = 1.0;
            const double xc2 = xc * xc, xc3 = xc2 * xc;
            buf.sw[i] = (float)(1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2);
        } else {
            buf.sw[i] = 1.0f;
        }
    } else if (r >= 1 && r < 1 + cfg.c0) {
        s_q[sid][r - 1] = acc;
    }

    const int g1_off = 1 + cfg.c0;
    const int g2_off = g1_off + cfg.c1;
    // Write g1 into buffer on its owning lane so all 4 channels are ready for qp_pack4
    float my_g1 = 0.0f;
    if (r >= g1_off && r < g2_off) {
        const int c = r - g1_off;
        buf.g1[(int64_t)i * 4 + c] = acc;
    }
    if constexpr (HAS_C2) {
        if (r >= g2_off && r < g2_off + cfg.c2) {
            const int c = r - g2_off;
            buf.g2[(int64_t)i * 4 + c] = acc;
            const float* src = I_raw2 + (int64_t)i * 5 * i_stride + c;
            buf.Q_pack03[(int64_t)i * 4 + c] = make_float4(
                acc * __ldg(src + 0 * i_stride),
                acc * __ldg(src + 1 * i_stride),
                acc * __ldg(src + 2 * i_stride),
                acc * __ldg(src + 3 * i_stride));
            buf.Q_pack4[(int64_t)i * 4 + c] = acc * __ldg(src + 4 * i_stride);
        } else if (r >= g2_off + cfg.c2 && r < g2_off + 4) {
            const int c = r - g2_off;
            buf.Q_pack03[(int64_t)i * 4 + c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            buf.Q_pack4[(int64_t)i * 4 + c] = 0.0f;
        }
    }
    __syncwarp(smask);

    if (r < 4) {
        const float qv = s_q[sid][r];
        float px = 0.0f, py = 0.0f, pz = 0.0f;
        if (r < cfg.c1) {
            my_g1 = buf.g1[(int64_t)i * 4 + r];
            const float* src = I_raw1 + (int64_t)i * 3 * i_stride + r;
            px = my_g1 * __ldg(src + 0 * i_stride);
            py = my_g1 * __ldg(src + 1 * i_stride);
            pz = my_g1 * __ldg(src + 2 * i_stride);
        }
        buf.qp_pack4[(int64_t)i * 4 + r] = make_float4(qv, px, py, pz);
    }
}

// =================================================================================================
// Kernel F2: Pure 1-Pass 4-Lane Subwarp (32 atoms/CTA!) Symmetric LR-SK-Fast Edge Kernel (pol=False)
//   Slot layout per (pair, b, lane c):
//     0: w0 (sym), 1: w01_f, 2: w01_r, 3: w1s (sym), 4: w1p (sym), 5: w2b (lane 0, sym)
//     If HAS_C2: 6: w02_f, 7: w02_r, 8: w12s_f, 9: w12s_r, 10: w12p_f, 11: w12p_r, 12: w2s, 13: w2p, 14: w2d
// =================================================================================================
template <int NB2, bool HAS_C2, bool STRESS>
__global__ void __launch_bounds__(128, HAS_C2 ? 3 : 5) lrsk_edge_sym_1p_kernel(
    int N, int S, LrskCfg cfg, ChebParam r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_fast4,   // [S*S, NB2_4, N_T4, 4]
    LrskFastBufs buf,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int N_T4 = HAS_C2 ? 15 : 6;
    constexpr int EB = (HAS_C2 || NB2_4 > 2) ? 2 : 4;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[32][EB][NB2_4];
    __shared__ float4 s_dg[32][EB][NB2_4];
    __shared__ float4 s_u[32][EB];
    __shared__ int2   s_jp[32][EB];
    __shared__ float  s_swj[32][EB];

    const int tot_w4 = S * S * NB2_4 * N_T4 * 4;
    for (int idx = threadIdx.x; idx < tot_w4; idx += 128) s_W4[idx] = __ldg(W_fast4 + idx);
    __syncthreads();
    const float4* W_ptr = s_W4;

    const int sid = threadIdx.x >> 2;
    const int c = threadIdx.x & 3;
    const unsigned qmask = 0xfu << ((sid & 7) * 4);
    const int i = blockIdx.x * 32 + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float c_i = cfg.half_esc * buf.sw[i];

    const float4 qpi = __ldg(buf.qp_pack4 + (int64_t)i * 4 + c);
    const float qi = qpi.x, pix = qpi.y, piy = qpi.z, piz = qpi.w;
    float Qi0 = 0.0f, Qi1 = 0.0f, Qi2 = 0.0f, Qi3 = 0.0f, Qi4 = 0.0f;
    if constexpr (HAS_C2) {
        const float4 Q03 = __ldg(buf.Q_pack03 + (int64_t)i * 4 + c);
        Qi0 = Q03.x; Qi1 = Q03.y; Qi2 = Q03.z; Qi3 = Q03.w;
        Qi4 = __ldg(buf.Q_pack4 + (int64_t)i * 4 + c);
    }

    float e_lr_acc = 0.0f;
    float dqi = 0.0f, dpix = 0.0f, dpiy = 0.0f, dpiz = 0.0f;
    float dQi0 = 0.0f, dQi1 = 0.0f, dQi2 = 0.0f, dQi3 = 0.0f, dQi4 = 0.0f;
    double rfx = 0.0, rfy = 0.0, rfz = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (c < bcnt) {
            const int e = eb_b + c;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const int z_j = __ldg(species + j);
            const float r = fmaxf(vl.w, 1e-8f);
            const float inv_r = 1.0f / r;
            s_jp[sid][c] = make_int2(j, z_i * S + z_j);
            s_u[sid][c] = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            s_swj[sid][c] = __ldg(buf.sw + j);
            eval_cheb_nb_store<NB2, true>(r, r_max2, s_g[sid][c], s_dg[sid][c]);
        }
        __syncwarp(qmask);

        for (int bi = 0; bi < bcnt; ++bi) {
            const int2 jp = s_jp[sid][bi];
            const int j = jp.x, pair = jp.y;
            const float4 u4 = s_u[sid][bi];
            const float ux = u4.x, uy = u4.y, uz = u4.z, inv_r = u4.w;
            const float c_j = cfg.half_esc * s_swj[sid][bi];
            const float c_sum = c_i + c_j;
            const float4* Wp = W_ptr + (int64_t)pair * (NB2_4 * N_T4 * 4) + c;

            auto eval_w = [&](int slot, float& w, float& dw) {
                w = 0.0f; dw = 0.0f;
                #pragma unroll
                for (int b = 0; b < NB2_4; ++b) {
                    const float4 wv = Wp[(b * N_T4 + slot) * 4];
                    w  += v1_dot4(wv, s_g[sid][bi][b]);
                    dw += v1_dot4(wv, s_dg[sid][bi][b]);
                }
            };

            const float4 qpj = __ldg(buf.qp_pack4 + (int64_t)j * 4 + c);
            const float qj = qpj.x, pjx = qpj.y, pjy = qpj.z, pjz = qpj.w;

            float er_lane = 0.0f, dux = 0.0f, duy = 0.0f, duz = 0.0f;

            // 2-body baseline (slot 5, lane 0 only)
            if (c == 0) {
                float w2b, dw2b;
                eval_w(5, w2b, dw2b);
                e_lr_acc += w2b;
                er_lane = fmaf(c_sum, dw2b, er_lane);
            }

            // 0-0 (slot 0, symmetric)
            {
                float w0, dw0;
                eval_w(0, w0, dw0);
                e_lr_acc = fmaf(w0, qi * qj, e_lr_acc);
                er_lane = fmaf(c_sum * dw0, qi * qj, er_lane);
                dqi = fmaf(c_sum * w0, qj, dqi);
            }

            const float piu = fmaf(pix, ux, fmaf(piy, uy, piz * uz));
            const float pju = fmaf(pjx, ux, fmaf(pjy, uy, pjz * uz));

            // 0-1 (slot 1 = fwd, slot 2 = rev)
            {
                float w01f, dw01f, w01r, dw01r;
                eval_w(1, w01f, dw01f);
                eval_w(2, w01r, dw01r);
                e_lr_acc = fmaf(w01f, qi * pju, e_lr_acc);
                const float cf_qi = c_i * qi, cr_qj = c_j * qj;
                er_lane = fmaf(dw01f, cf_qi * pju, fmaf(-dw01r, cr_qj * piu, er_lane));
                const float sf = w01f * cf_qi, sr = w01r * cr_qj;
                dux = fmaf(sf, pjx, fmaf(-sr, pix, dux));
                duy = fmaf(sf, pjy, fmaf(-sr, piy, duy));
                duz = fmaf(sf, pjz, fmaf(-sr, piz, duz));
                dqi = fmaf(c_i * w01f, pju, dqi);
                dpix = fmaf(-sr, ux, dpix); dpiy = fmaf(-sr, uy, dpiy); dpiz = fmaf(-sr, uz, dpiz);
            }

            // 1-1 (slot 3 = w1s, slot 4 = w1p, symmetric)
            {
                float w1s, dw1s, w1p, dw1p;
                eval_w(3, w1s, dw1s);
                eval_w(4, w1p, dw1p);
                const float p_dot = fmaf(pix, pjx, fmaf(piy, pjy, piz * pjz));
                e_lr_acc = fmaf(w1s, piu * pju, fmaf(w1p, p_dot, e_lr_acc));
                const float cs_w1s = c_sum * w1s, cs_w1p = c_sum * w1p;
                er_lane = fmaf(c_sum * dw1s, piu * pju, fmaf(c_sum * dw1p, p_dot, er_lane));
                const float su_i = cs_w1s * pju, su_j = cs_w1s * piu;
                dux = fmaf(su_i, pix, fmaf(su_j, pjx, dux));
                duy = fmaf(su_i, piy, fmaf(su_j, pjy, duy));
                duz = fmaf(su_i, piz, fmaf(su_j, pjz, duz));
                dpix = fmaf(su_i, ux, fmaf(cs_w1p, pjx, dpix));
                dpiy = fmaf(su_i, uy, fmaf(cs_w1p, pjy, dpiy));
                dpiz = fmaf(su_i, uz, fmaf(cs_w1p, pjz, dpiz));
            }

            // Quadrupole terms (0-2, 1-2, 2-2) when HAS_C2
            if constexpr (HAS_C2) {
                const float4 Qj03 = __ldg(buf.Q_pack03 + (int64_t)j * 4 + c);
                const float Qj0 = Qj03.x, Qj1 = Qj03.y, Qj2 = Qj03.z, Qj3 = Qj03.w;
                const float Qj4 = __ldg(buf.Q_pack4 + (int64_t)j * 4 + c);
                float muix, muiy, muiz, s2i, mujx, mujy, mujz, s2j;
                lrsk_l2_matvec(Qi0, Qi1, Qi2, Qi3, Qi4, ux, uy, uz, muix, muiy, muiz, s2i);
                lrsk_l2_matvec(Qj0, Qj1, Qj2, Qj3, Qj4, ux, uy, uz, mujx, mujy, mujz, s2j);

                float g_s2i = 0.0f, g_muix = 0.0f, g_muiy = 0.0f, g_muiz = 0.0f;
                float g_s2j = 0.0f, g_mujx = 0.0f, g_mujy = 0.0f, g_mujz = 0.0f;

                {
                    float w02f, dw02f, w02r, dw02r;
                    eval_w(6, w02f, dw02f); eval_w(7, w02r, dw02r);
                    e_lr_acc = fmaf(w02f, qi * s2j, e_lr_acc);
                    const float cf_qi = c_i * qi, cr_qj = c_j * qj;
                    er_lane = fmaf(dw02f, cf_qi * s2j, fmaf(dw02r, cr_qj * s2i, er_lane));
                    dqi = fmaf(c_i * w02f, s2j, dqi);
                    g_s2j = fmaf(w02f, cf_qi, g_s2j);
                    g_s2i = fmaf(w02r, cr_qj, g_s2i);
                }
                {
                    float w12sf, dw12sf, w12sr, dw12sr, w12pf, dw12pf, w12pr, dw12pr;
                    eval_w(8, w12sf, dw12sf); eval_w(9, w12sr, dw12sr);
                    eval_w(10, w12pf, dw12pf); eval_w(11, w12pr, dw12pr);
                    const float pi_muj = fmaf(pix, mujx, fmaf(piy, mujy, piz * mujz));
                    const float pj_mui = fmaf(pjx, muix, fmaf(pjy, muiy, pjz * muiz));
                    e_lr_acc = fmaf(w12sf, piu * s2j, fmaf(w12pf, pi_muj, e_lr_acc));
                    const float cw12sf = c_i * w12sf, cw12pf = c_i * w12pf;
                    const float cw12sr = c_j * w12sr, cw12pr = c_j * w12pr;
                    er_lane = fmaf(c_i * dw12sf, piu * s2j, fmaf(c_i * dw12pf, pi_muj,
                              fmaf(-c_j * dw12sr, pju * s2i, fmaf(-c_j * dw12pr, pj_mui, er_lane))));
                    const float sf12 = cw12sf * s2j, sr12 = cw12sr * s2i;
                    dux = fmaf(sf12, pix, fmaf(-sr12, pjx, dux));
                    duy = fmaf(sf12, piy, fmaf(-sr12, pjy, duy));
                    duz = fmaf(sf12, piz, fmaf(-sr12, pjz, duz));
                    dpix = fmaf(sf12, ux, fmaf(cw12pf, mujx, dpix));
                    dpiy = fmaf(sf12, uy, fmaf(cw12pf, mujy, dpiy));
                    dpiz = fmaf(sf12, uz, fmaf(cw12pf, mujz, dpiz));
                    g_s2j = fmaf(cw12sf, piu, g_s2j);
                    g_mujx = fmaf(cw12pf, pix, g_mujx); g_mujy = fmaf(cw12pf, piy, g_mujy); g_mujz = fmaf(cw12pf, piz, g_mujz);
                    g_s2i = fmaf(-cw12sr, pju, g_s2i);
                    g_muix = fmaf(-cw12pr, pjx, g_muix); g_muiy = fmaf(-cw12pr, pjy, g_muiy); g_muiz = fmaf(-cw12pr, pjz, g_muiz);
                }
                {
                    float w2s, dw2s, w2p, dw2p, w2d, dw2d;
                    eval_w(12, w2s, dw2s); eval_w(13, w2p, dw2p); eval_w(14, w2d, dw2d);
                    const float mu_dot = fmaf(muix, mujx, fmaf(muiy, mujy, muiz * mujz));
                    const float Q_dot = fmaf(Qi0, Qj0, fmaf(Qi1, Qj1, fmaf(Qi2, Qj2, fmaf(Qi3, Qj3, Qi4 * Qj4))));
                    e_lr_acc = fmaf(w2s, s2i * s2j, fmaf(w2p, mu_dot, fmaf(w2d, Q_dot, e_lr_acc)));
                    const float cw2s = c_sum * w2s, cw2p = c_sum * w2p, cw2d = c_sum * w2d;
                    er_lane = fmaf(c_sum * dw2s, s2i * s2j, fmaf(c_sum * dw2p, mu_dot, fmaf(c_sum * dw2d, Q_dot, er_lane)));
                    g_s2i = fmaf(cw2s, s2j, g_s2i);
                    g_muix = fmaf(cw2p, mujx, g_muix); g_muiy = fmaf(cw2p, mujy, g_muiy); g_muiz = fmaf(cw2p, mujz, g_muiz);
                    g_s2j = fmaf(cw2s, s2i, g_s2j);
                    g_mujx = fmaf(cw2p, muix, g_mujx); g_mujy = fmaf(cw2p, muiy, g_mujy); g_mujz = fmaf(cw2p, muiz, g_mujz);

                    float tux, tuy, tuz;
                    lrsk_l2_matvec_only(Qi0, Qi1, Qi2, Qi3, Qi4, g_muix, g_muiy, g_muiz, tux, tuy, tuz);
                    dux = fmaf(2.0f * g_s2i, muix, dux + tux);
                    duy = fmaf(2.0f * g_s2i, muiy, duy + tuy);
                    duz = fmaf(2.0f * g_s2i, muiz, duz + tuz);
                    lrsk_l2_matvec_only(Qj0, Qj1, Qj2, Qj3, Qj4, g_mujx, g_mujy, g_mujz, tux, tuy, tuz);
                    dux = fmaf(2.0f * g_s2j, mujx, dux + tux);
                    duy = fmaf(2.0f * g_s2j, mujy, duy + tuy);
                    duz = fmaf(2.0f * g_s2j, mujz, duz + tuz);

                    dQi0 = fmaf(cw2d, Qj0, dQi0); dQi1 = fmaf(cw2d, Qj1, dQi1); dQi2 = fmaf(cw2d, Qj2, dQi2);
                    dQi3 = fmaf(cw2d, Qj3, dQi3); dQi4 = fmaf(cw2d, Qj4, dQi4);
                    lrsk_l2_dq_acc(g_muix + g_s2i * ux, g_muiy + g_s2i * uy, g_muiz + g_s2i * uz,
                                   ux, uy, uz, dQi0, dQi1, dQi2, dQi3, dQi4);
                }
            }

            const bool p0 = (c & 1) != 0;
            const float r0 = (p0 ? dux : er_lane) + __shfl_xor_sync(qmask, p0 ? er_lane : dux, 1);
            const float r1 = (p0 ? duz : duy)     + __shfl_xor_sync(qmask, p0 ? duy : duz, 1);
            const bool p1 = (c & 2) != 0;
            const float rv = (p1 ? r1 : r0) + __shfl_xor_sync(qmask, p1 ? r0 : r1, 2);
            const float er_t  = __shfl_sync(qmask, rv, 0, 4);
            const float dux_t = __shfl_sync(qmask, rv, 1, 4);
            const float duy_t = __shfl_sync(qmask, rv, 2, 4);
            const float duz_t = __shfl_sync(qmask, rv, 3, 4);
            if (c == 0) {
                const float udot = fmaf(ux, dux_t, fmaf(uy, duy_t, uz * duz_t));
                const float gx = fmaf(er_t, ux, (dux_t - udot * ux) * inv_r);
                const float gy = fmaf(er_t, uy, (duy_t - udot * uy) * inv_r);
                const float gz = fmaf(er_t, uz, (duz_t - udot * uz) * inv_r);
                rfx -= (double)gx; rfy -= (double)gy; rfz -= (double)gz;
                if constexpr (STRESS) {
                    const float4 vl = __ldg(vec_len + eb_b + bi);
                    v0 += 0.5 * (double)vl.x * (double)gx;
                    v1 += 0.25 * ((double)vl.x * (double)gy + (double)vl.y * (double)gx);
                    v2 += 0.25 * ((double)vl.x * (double)gz + (double)vl.z * (double)gx);
                    v3 += 0.5 * (double)vl.y * (double)gy;
                    v4 += 0.25 * ((double)vl.y * (double)gz + (double)vl.z * (double)gy);
                    v5 += 0.5 * (double)vl.z * (double)gz;
                }
            }
        }
        __syncwarp(qmask);
    }

    const float e_tot = q4_sum_f(e_lr_acc, qmask);
    if (c == 0) buf.e_lr[i] = cfg.half_esc * e_tot;

    buf.dqp_pack4[(int64_t)i * 4 + c] = make_float4(dqi, dpix, dpiy, dpiz);
    if constexpr (HAS_C2) {
        buf.dQ_pack03[(int64_t)i * 4 + c] = make_float4(dQi0, dQi1, dQi2, dQi3);
        buf.dQ_pack4[(int64_t)i * 4 + c] = dQi4;
    }
    if (c == 0) {
        if (forces_f32) {
            forces_f32[(int64_t)i * 3 + 0] += (float)rfx;
            forces_f32[(int64_t)i * 3 + 1] += (float)rfy;
            forces_f32[(int64_t)i * 3 + 2] += (float)rfz;
        } else {
            forces[(int64_t)i * 3 + 0] += rfx;
            forces[(int64_t)i * 3 + 1] += rfy;
            forces[(int64_t)i * 3 + 2] += rfz;
        }
        if constexpr (STRESS) {
            atomicAdd(virial + 0, v0); atomicAdd(virial + 1, v1); atomicAdd(virial + 2, v2);
            atomicAdd(virial + 3, v3); atomicAdd(virial + 4, v4); atomicAdd(virial + 5, v5);
        }
    }
}

// =================================================================================================
// Kernel F3: 4-Lane Subwarp (32 atoms/CTA!) Symmetric LR-SK-Fast Polarization Field Forward (Pass 1)
//            Also serves energy-only evaluation for pol=False when buf.pol_ef4 == nullptr!
// =================================================================================================
template <int NB2, bool HAS_C2>
__global__ void __launch_bounds__(128, HAS_C2 ? 4 : 6) lrsk_edge_sym_pol_fwd_kernel(
    int N, int S, LrskCfg cfg, ChebParam r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_fast4,
    LrskFastBufs buf) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int N_T4 = HAS_C2 ? 15 : 6;
    constexpr int EB = (HAS_C2 || NB2_4 > 2) ? 2 : 4;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[32][EB][NB2_4];
    __shared__ float4 s_u[32][EB];
    __shared__ int2   s_jp[32][EB];

    const int tot_w4 = S * S * NB2_4 * N_T4 * 4;
    for (int idx = threadIdx.x; idx < tot_w4; idx += 128) s_W4[idx] = __ldg(W_fast4 + idx);
    __syncthreads();
    const float4* W_ptr = s_W4;

    const int sid = threadIdx.x >> 2;
    const int c = threadIdx.x & 3;
    const unsigned qmask = 0xfu << ((sid & 7) * 4);
    const int i = blockIdx.x * 32 + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float4 qpi = __ldg(buf.qp_pack4 + (int64_t)i * 4 + c);
    const float qi = qpi.x, pix = qpi.y, piy = qpi.z, piz = qpi.w;
    float Qi0 = 0.0f, Qi1 = 0.0f, Qi2 = 0.0f, Qi3 = 0.0f, Qi4 = 0.0f;
    if constexpr (HAS_C2) {
        const float4 Q03 = __ldg(buf.Q_pack03 + (int64_t)i * 4 + c);
        Qi0 = Q03.x; Qi1 = Q03.y; Qi2 = Q03.z; Qi3 = Q03.w;
        Qi4 = __ldg(buf.Q_pack4 + (int64_t)i * 4 + c);
    }

    float e_lr_acc = 0.0f, phi_acc = 0.0f, Efx = 0.0f, Efy = 0.0f, Efz = 0.0f;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (c < bcnt) {
            const int e = eb_b + c;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const float r = fmaxf(vl.w, 1e-8f);
            const float inv_r = 1.0f / r;
            s_jp[sid][c] = make_int2(j, z_i * S + __ldg(species + j));
            s_u[sid][c] = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            eval_cheb_nb_store<NB2, false>(r, r_max2, s_g[sid][c], nullptr);
        }
        __syncwarp(qmask);

        for (int bi = 0; bi < bcnt; ++bi) {
            const int2 jp = s_jp[sid][bi];
            const int j = jp.x, pair = jp.y;
            const float4 u4 = s_u[sid][bi];
            const float ux = u4.x, uy = u4.y, uz = u4.z;
            const float4* Wp = W_ptr + (int64_t)pair * (NB2_4 * N_T4 * 4) + c;

            auto eval_wf = [&](int slot) -> float {
                float w = 0.0f;
                #pragma unroll
                for (int b = 0; b < NB2_4; ++b) {
                    const float4 wv = Wp[(b * N_T4 + slot) * 4];
                    w += v1_dot4(wv, s_g[sid][bi][b]);
                }
                return w;
            };

            const float4 qpj = __ldg(buf.qp_pack4 + (int64_t)j * 4 + c);
            const float qj = qpj.x, pjx = qpj.y, pjy = qpj.z, pjz = qpj.w;

            if (c == 0) e_lr_acc += eval_wf(5);

            const float w0 = eval_wf(0);
            const float t0 = w0 * qj;
            e_lr_acc = fmaf(qi, t0, e_lr_acc);
            phi_acc += t0;

            const float piu = fmaf(pix, ux, fmaf(piy, uy, piz * uz));
            const float pju = fmaf(pjx, ux, fmaf(pjy, uy, pjz * uz));

            const float w01f = eval_wf(1);
            const float t01p = w01f * pju;
            e_lr_acc = fmaf(qi, t01p, e_lr_acc);
            phi_acc += t01p;
            const float t01q = w01f * qj;
            Efx = fmaf(t01q, ux, Efx); Efy = fmaf(t01q, uy, Efy); Efz = fmaf(t01q, uz, Efz);

            const float w1s = eval_wf(3);
            const float w1p = eval_wf(4);
            const float w1s_pju = w1s * pju;
            e_lr_acc = fmaf(piu, w1s_pju, fmaf(w1p, fmaf(pix, pjx, fmaf(piy, pjy, piz * pjz)), e_lr_acc));
            Efx = fmaf(w1s_pju, ux, fmaf(w1p, pjx, Efx));
            Efy = fmaf(w1s_pju, uy, fmaf(w1p, pjy, Efy));
            Efz = fmaf(w1s_pju, uz, fmaf(w1p, pjz, Efz));

            if constexpr (HAS_C2) {
                const float4 Qj03 = __ldg(buf.Q_pack03 + (int64_t)j * 4 + c);
                const float Qj0 = Qj03.x, Qj1 = Qj03.y, Qj2 = Qj03.z, Qj3 = Qj03.w;
                const float Qj4 = __ldg(buf.Q_pack4 + (int64_t)j * 4 + c);
                float muix, muiy, muiz, s2i, mujx, mujy, mujz, s2j;
                lrsk_l2_matvec(Qi0, Qi1, Qi2, Qi3, Qi4, ux, uy, uz, muix, muiy, muiz, s2i);
                lrsk_l2_matvec(Qj0, Qj1, Qj2, Qj3, Qj4, ux, uy, uz, mujx, mujy, mujz, s2j);
                e_lr_acc = fmaf(eval_wf(6), qi * s2j, e_lr_acc);
                const float pi_muj = fmaf(pix, mujx, fmaf(piy, mujy, piz * mujz));
                e_lr_acc = fmaf(eval_wf(8), piu * s2j, fmaf(eval_wf(10), pi_muj, e_lr_acc));
                const float mu_dot = fmaf(muix, mujx, fmaf(muiy, mujy, muiz * mujz));
                const float Q_dot = fmaf(Qi0, Qj0, fmaf(Qi1, Qj1, fmaf(Qi2, Qj2, fmaf(Qi3, Qj3, Qi4 * Qj4))));
                e_lr_acc = fmaf(eval_wf(12), s2i * s2j, fmaf(eval_wf(13), mu_dot, fmaf(eval_wf(14), Q_dot, e_lr_acc)));
            }
        }
        __syncwarp(qmask);
    }

    const float e_tot = q4_sum_f(e_lr_acc, qmask);
    if (c == 0) buf.e_lr[i] = cfg.half_esc * e_tot;
    if (buf.pol_ef4) buf.pol_ef4[(int64_t)i * 4 + c] = make_float4(phi_acc, Efx, Efy, Efz);
}

// =================================================================================================
// Kernel F4a/b/c: Tensor-Core-Friendly Polarization MLP Helpers (used with hgemm2 for pol_W0!)
// =================================================================================================
__global__ void __launch_bounds__(128) lrsk_pol_pack_in_sym_kernel(
    int N, int ro_width, int in_pad, LrskCfg cfg,
    const float* __restrict__ h,
    LrskFastBufs buf,
    float* __restrict__ pol_in,
    int8_t* __restrict__ eA_pol) {
    const int sid = threadIdx.x >> 4, slane = threadIdx.x & 15;
    const unsigned smask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;

    const int f_dim = cfg.c0 + 2 * cfg.c1;
    const int in_raw = f_dim + ro_width;
    float* dst = pol_in + (int64_t)i * in_pad;
    float am = 0.0f;

    if (slane < 4) {
        const float4 ef = __ldg(buf.pol_ef4 + (int64_t)i * 4 + slane);
        if (slane < cfg.c0) {
            dst[slane] = ef.x;
            am = fmaxf(am, fabsf(ef.x));
        }
        if (slane < cfg.c1) {
            const float4 qp = __ldg(buf.qp_pack4 + (int64_t)i * 4 + slane);
            const float esq = fmaf(ef.y, ef.y, fmaf(ef.z, ef.z, ef.w * ef.w));
            const float pde = fmaf(qp.y, ef.y, fmaf(qp.z, ef.z, qp.w * ef.w));
            dst[cfg.c0 + slane] = esq;
            dst[cfg.c0 + cfg.c1 + slane] = pde;
            am = fmaxf(am, fmaxf(fabsf(esq), fabsf(pde)));
        }
    }
    const float* hi = h + (int64_t)i * ro_width;
    for (int k = slane; k < ro_width; k += 16) {
        const float v = __ldg(hi + k);
        dst[f_dim + k] = v;
        am = fmaxf(am, fabsf(v));
    }
    for (int k = in_raw + slane; k < in_pad; k += 16) dst[k] = 0.0f;

    if (eA_pol) {
        am = fmaxf(am, __shfl_xor_sync(smask, am, 8));
        am = fmaxf(am, __shfl_xor_sync(smask, am, 4));
        am = fmaxf(am, __shfl_xor_sync(smask, am, 2));
        am = fmaxf(am, __shfl_xor_sync(smask, am, 1));
        if (slane == 0) eA_pol[i] = ilmlip_exp_from_am(am);
    }
}

template <bool ZBL_SW, bool NEED_GRAD>
__global__ void __launch_bounds__(128) lrsk_pol_act_energy_zbl_kernel(
    int N, int pw, int pw_pad, LrskCfg cfg,
    const int* __restrict__ species,
    const float* __restrict__ pol_b0,
    const float* __restrict__ pol_u,
    const float* __restrict__ e_site,
    const float* __restrict__ e_lr,
    const float* __restrict__ sw_arr,
    const double* __restrict__ e_zbl,
    float* __restrict__ pre_io,        // [N, pw_pad] input: pre, output: dpre (if NEED_GRAD)
    int8_t* __restrict__ eA_dpre,
    double* __restrict__ energy_out,
    double* __restrict__ atom_energy_out,
    float* __restrict__ w_zbl_out) {
    const int sid = threadIdx.x >> 4, slane = threadIdx.x & 15;
    const unsigned smask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;

    const int z = species[i];
    const float* pu_z = pol_u + (int64_t)z * pw;
    const float sw_i = sw_arr[i];
    float* pr = pre_io + (int64_t)i * pw_pad;

    float e_pol_l = 0.0f, am = 0.0f;
    for (int w = slane; w < pw_pad; w += 16) {
        if (w < pw) {
            const float pre = pr[w] + __ldg(pol_b0 + w);
            const float sig = 1.0f / (1.0f + __expf(-pre));
            const float act = pre * sig;
            const float uw = __ldg(pu_z + w);
            e_pol_l = fmaf(act, uw, e_pol_l);
            if constexpr (NEED_GRAD) {
                const float dsilu = sig * (1.0f + pre * (1.0f - sig));
                const float dp = sw_i * uw * dsilu;
                pr[w] = dp;
                am = fmaxf(am, fabsf(dp));
            }
        } else if constexpr (NEED_GRAD) {
            pr[w] = 0.0f;
        }
    }
    const float e_pol = half_warp_sum_f(e_pol_l, smask);
    if constexpr (NEED_GRAD) {
        if (eA_dpre) {
            am = fmaxf(am, __shfl_xor_sync(smask, am, 8));
            am = fmaxf(am, __shfl_xor_sync(smask, am, 4));
            am = fmaxf(am, __shfl_xor_sync(smask, am, 2));
            am = fmaxf(am, __shfl_xor_sync(smask, am, 1));
            if (slane == 0) eA_dpre[i] = ilmlip_exp_from_am(am);
        }
    }

    double val = 0.0;
    if (slane == 0) {
        const double e_ml = (double)e_site[i] + (double)e_lr[i] + (double)e_pol;
        if constexpr (ZBL_SW) {
            const double ez = e_zbl[i];
            double xc = ez * (double)cfg.inv_zc;
            double sw, dsw_dx;
            if (xc <= 0.0) { sw = 1.0; dsw_dx = 0.0; }
            else if (xc >= 1.0) { sw = 0.0; dsw_dx = 0.0; }
            else {
                const double xc2 = xc * xc, xc3 = xc2 * xc;
                sw = 1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2;
                dsw_dx = -30.0 * xc2 + 60.0 * xc3 - 30.0 * xc2 * xc2;
            }
            val = sw * e_ml + ez;
            if constexpr (NEED_GRAD) w_zbl_out[i] = (float)(1.0 + dsw_dx * (double)cfg.inv_zc * e_ml);
        } else {
            val = e_ml;
        }
        if (atom_energy_out) atom_energy_out[i] = val;
    }
    const double wsum = warp_sum_d(val);
    if ((threadIdx.x & 31) == 0 && wsum != 0.0) atomicAdd(energy_out, wsum);
}

__global__ void __launch_bounds__(128) lrsk_pol_unpack_grad_sym_kernel(
    int N, int ro_width, int in_pad, LrskCfg cfg,
    const float* __restrict__ dpol_in,   // [N, in_pad]
    LrskFastBufs buf,
    float* __restrict__ dh_pol_out) {
    const int sid = threadIdx.x >> 4, slane = threadIdx.x & 15;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;

    const int f_dim = cfg.c0 + 2 * cfg.c1;
    const float* dp = dpol_in + (int64_t)i * in_pad;
    if (slane < 4) {
        const float dphi = (slane < cfg.c0) ? dp[slane] : 0.0f;
        float defx = 0.0f, defy = 0.0f, defz = 0.0f;
        float dpx = 0.0f, dpy = 0.0f, dpz = 0.0f;
        if (slane < cfg.c1) {
            const float d_esq = dp[cfg.c0 + slane];
            const float d_pe  = dp[cfg.c0 + cfg.c1 + slane];
            const float4 ef = __ldg(buf.pol_ef4 + (int64_t)i * 4 + slane);
            const float4 qp = __ldg(buf.qp_pack4 + (int64_t)i * 4 + slane);
            defx = fmaf(2.0f * d_esq, ef.y, d_pe * qp.y);
            defy = fmaf(2.0f * d_esq, ef.z, d_pe * qp.z);
            defz = fmaf(2.0f * d_esq, ef.w, d_pe * qp.w);
            dpx = d_pe * ef.y; dpy = d_pe * ef.z; dpz = d_pe * ef.w;
        }
        buf.dpol_ef4[(int64_t)i * 4 + slane] = make_float4(dphi, defx, defy, defz);
        buf.dqp_pack4[(int64_t)i * 4 + slane] = make_float4(0.0f, dpx, dpy, dpz);  // seeds dp_pol!
    }
    float* dho = dh_pol_out + (int64_t)i * ro_width;
    for (int k = slane; k < ro_width; k += 16) dho[k] = dp[f_dim + k];
}

// =================================================================================================
// Kernel F5: 4-Lane Subwarp (32 atoms/CTA!) Symmetric LR-SK-Fast Polarization Edge Backward (Pass 2)
//   Only 2x LDG.128 per neighbor j (qp_pack4 + dpol_ef4), ZERO atomics on forces/multipole grads!
// =================================================================================================
template <int NB2, bool HAS_C2, bool STRESS>
__global__ void __launch_bounds__(128, HAS_C2 ? 3 : 5) lrsk_edge_sym_pol_bwd_kernel(
    int N, int S, LrskCfg cfg, ChebParam r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_fast4,
    LrskFastBufs buf,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int N_T4 = HAS_C2 ? 15 : 6;
    constexpr int EB = (HAS_C2 || NB2_4 > 2) ? 2 : 4;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[32][EB][NB2_4];
    __shared__ float4 s_dg[32][EB][NB2_4];
    __shared__ float4 s_u[32][EB];
    __shared__ int2   s_jp[32][EB];
    __shared__ float  s_swj[32][EB];

    const int tot_w4 = S * S * NB2_4 * N_T4 * 4;
    for (int idx = threadIdx.x; idx < tot_w4; idx += 128) s_W4[idx] = __ldg(W_fast4 + idx);
    __syncthreads();
    const float4* W_ptr = s_W4;

    const int sid = threadIdx.x >> 2;
    const int c = threadIdx.x & 3;
    const unsigned qmask = 0xfu << ((sid & 7) * 4);
    const int i = blockIdx.x * 32 + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float c_i = cfg.half_esc * buf.sw[i];

    const float4 qpi = __ldg(buf.qp_pack4 + (int64_t)i * 4 + c);
    const float qi = qpi.x, pix = qpi.y, piy = qpi.z, piz = qpi.w;
    const float4 dpi_pol = __ldg(buf.dpol_ef4 + (int64_t)i * 4 + c);
    const float dPhi_i = dpi_pol.x, dEfix = dpi_pol.y, dEfiy = dpi_pol.z, dEfiz = dpi_pol.w;
    const float qa0_i = fmaf(c_i, qi, dPhi_i);
    const float pa1_ix = fmaf(c_i, pix, dEfix);
    const float pa1_iy = fmaf(c_i, piy, dEfiy);
    const float pa1_iz = fmaf(c_i, piz, dEfiz);

    float Qi0 = 0.0f, Qi1 = 0.0f, Qi2 = 0.0f, Qi3 = 0.0f, Qi4 = 0.0f;
    if constexpr (HAS_C2) {
        const float4 Q03 = __ldg(buf.Q_pack03 + (int64_t)i * 4 + c);
        Qi0 = Q03.x; Qi1 = Q03.y; Qi2 = Q03.z; Qi3 = Q03.w;
        Qi4 = __ldg(buf.Q_pack4 + (int64_t)i * 4 + c);
    }

    const float4 seed_dqp = __ldg(buf.dqp_pack4 + (int64_t)i * 4 + c);
    float dqi = 0.0f, dpix = seed_dqp.y, dpiy = seed_dqp.z, dpiz = seed_dqp.w;
    float dQi0 = 0.0f, dQi1 = 0.0f, dQi2 = 0.0f, dQi3 = 0.0f, dQi4 = 0.0f;
    double rfx = 0.0, rfy = 0.0, rfz = 0.0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (c < bcnt) {
            const int e = eb_b + c;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const int z_j = __ldg(species + j);
            const float r = fmaxf(vl.w, 1e-8f);
            const float inv_r = 1.0f / r;
            s_jp[sid][c] = make_int2(j, z_i * S + z_j);
            s_u[sid][c] = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            s_swj[sid][c] = __ldg(buf.sw + j);
            eval_cheb_nb_store<NB2, true>(r, r_max2, s_g[sid][c], s_dg[sid][c]);
        }
        __syncwarp(qmask);

        for (int bi = 0; bi < bcnt; ++bi) {
            const int2 jp = s_jp[sid][bi];
            const int j = jp.x, pair = jp.y;
            const float4 u4 = s_u[sid][bi];
            const float ux = u4.x, uy = u4.y, uz = u4.z, inv_r = u4.w;
            const float c_j = cfg.half_esc * s_swj[sid][bi];
            const float c_sum = c_i + c_j;
            const float4* Wp = W_ptr + (int64_t)pair * (NB2_4 * N_T4 * 4) + c;

            auto eval_w = [&](int slot, float& w, float& dw) {
                w = 0.0f; dw = 0.0f;
                #pragma unroll
                for (int b = 0; b < NB2_4; ++b) {
                    const float4 wv = Wp[(b * N_T4 + slot) * 4];
                    w  += v1_dot4(wv, s_g[sid][bi][b]);
                    dw += v1_dot4(wv, s_dg[sid][bi][b]);
                }
            };

            const float4 qpj = __ldg(buf.qp_pack4 + (int64_t)j * 4 + c);
            const float qj = qpj.x, pjx = qpj.y, pjy = qpj.z, pjz = qpj.w;
            const float4 dpj_pol = __ldg(buf.dpol_ef4 + (int64_t)j * 4 + c);
            const float dPhi_j = dpj_pol.x, dEfjx = dpj_pol.y, dEfjy = dpj_pol.z, dEfjz = dpj_pol.w;
            const float qb0_j = fmaf(c_j, qj, dPhi_j);
            const float pb1_jx = fmaf(c_j, pjx, dEfjx);
            const float pb1_jy = fmaf(c_j, pjy, dEfjy);
            const float pb1_jz = fmaf(c_j, pjz, dEfjz);

            float er_lane = 0.0f, dux = 0.0f, duy = 0.0f, duz = 0.0f;

            // 2-body (slot 5, lane 0)
            if (c == 0) {
                float w2b, dw2b;
                eval_w(5, w2b, dw2b);
                er_lane = fmaf(c_sum, dw2b, er_lane);
            }

            // 0-0 (slot 0, symmetric)
            {
                float w0, dw0;
                eval_w(0, w0, dw0);
                er_lane = fmaf(dw0, fmaf(qa0_i, qj, qb0_j * qi), er_lane);
                dqi = fmaf(w0, fmaf(c_i, qj, qb0_j), dqi);
            }

            const float piu = fmaf(pix, ux, fmaf(piy, uy, piz * uz));
            const float pju = fmaf(pjx, ux, fmaf(pjy, uy, pjz * uz));

            // 0-1 (slot 1 = fwd, slot 2 = rev)
            {
                float w01f, dw01f, w01r, dw01r;
                eval_w(1, w01f, dw01f);
                eval_w(2, w01r, dw01r);
                const float defi_u = fmaf(dEfix, ux, fmaf(dEfiy, uy, dEfiz * uz));
                const float defj_u = fmaf(dEfjx, ux, fmaf(dEfjy, uy, dEfjz * uz));
                er_lane = fmaf(dw01f, fmaf(qa0_i, pju, qj * defi_u),
                          fmaf(-dw01r, fmaf(qb0_j, piu, qi * defj_u), er_lane));
                const float sf_a = w01f * qa0_i, sf_b = w01f * qj;
                const float sr_a = w01r * qb0_j, sr_b = w01r * qi;
                dux = fmaf(sf_a, pjx, fmaf(sf_b, dEfix, fmaf(-sr_a, pix, fmaf(-sr_b, dEfjx, dux))));
                duy = fmaf(sf_a, pjy, fmaf(sf_b, dEfiy, fmaf(-sr_a, piy, fmaf(-sr_b, dEfjy, duy))));
                duz = fmaf(sf_a, pjz, fmaf(sf_b, dEfiz, fmaf(-sr_a, piz, fmaf(-sr_b, dEfjz, duz))));
                dqi  = fmaf(c_i * w01f, pju, fmaf(-w01r, defj_u, dqi));
                dpix = fmaf(-sr_a, ux, dpix); dpiy = fmaf(-sr_a, uy, dpiy); dpiz = fmaf(-sr_a, uz, dpiz);
            }

            // 1-1 (slot 3 = w1s, slot 4 = w1p, symmetric)
            {
                float w1s, dw1s, w1p, dw1p;
                eval_w(3, w1s, dw1s);
                eval_w(4, w1p, dw1p);
                const float pa1_u = fmaf(pa1_ix, ux, fmaf(pa1_iy, uy, pa1_iz * uz));
                const float pb1_u = fmaf(pb1_jx, ux, fmaf(pb1_jy, uy, pb1_jz * uz));
                const float p1_dot = fmaf(pa1_ix, pjx, fmaf(pa1_iy, pjy, pa1_iz * pjz)) +
                                     fmaf(pb1_jx, pix, fmaf(pb1_jy, piy, pb1_jz * piz));
                er_lane = fmaf(dw1s, fmaf(pa1_u, pju, pb1_u * piu), fmaf(dw1p, p1_dot, er_lane));
                const float sf_a = w1s * pju, sf_b = w1s * pa1_u;
                const float sr_a = w1s * piu, sr_b = w1s * pb1_u;
                dux = fmaf(sf_a, pa1_ix, fmaf(sf_b, pjx, fmaf(sr_a, pb1_jx, fmaf(sr_b, pix, dux))));
                duy = fmaf(sf_a, pa1_iy, fmaf(sf_b, pjy, fmaf(sr_a, pb1_jy, fmaf(sr_b, piy, duy))));
                duz = fmaf(sf_a, pa1_iz, fmaf(sf_b, pjz, fmaf(sr_a, pb1_jz, fmaf(sr_b, piz, duz))));
                const float peff_x = fmaf(c_i, pjx, pb1_jx);
                const float peff_y = fmaf(c_i, pjy, pb1_jy);
                const float peff_z = fmaf(c_i, pjz, pb1_jz);
                const float peff_u = fmaf(c_i, pju, pb1_u);
                dpix = fmaf(w1s * peff_u, ux, fmaf(w1p, peff_x, dpix));
                dpiy = fmaf(w1s * peff_u, uy, fmaf(w1p, peff_y, dpiy));
                dpiz = fmaf(w1s * peff_u, uz, fmaf(w1p, peff_z, dpiz));
            }

            // Quadrupole terms (exact same as 1-pass — pol_field does not depend on Q!)
            if constexpr (HAS_C2) {
                const float4 Qj03 = __ldg(buf.Q_pack03 + (int64_t)j * 4 + c);
                const float Qj0 = Qj03.x, Qj1 = Qj03.y, Qj2 = Qj03.z, Qj3 = Qj03.w;
                const float Qj4 = __ldg(buf.Q_pack4 + (int64_t)j * 4 + c);
                float muix, muiy, muiz, s2i, mujx, mujy, mujz, s2j;
                lrsk_l2_matvec(Qi0, Qi1, Qi2, Qi3, Qi4, ux, uy, uz, muix, muiy, muiz, s2i);
                lrsk_l2_matvec(Qj0, Qj1, Qj2, Qj3, Qj4, ux, uy, uz, mujx, mujy, mujz, s2j);

                float g_s2i = 0.0f, g_muix = 0.0f, g_muiy = 0.0f, g_muiz = 0.0f;
                float g_s2j = 0.0f, g_mujx = 0.0f, g_mujy = 0.0f, g_mujz = 0.0f;

                {
                    float w02f, dw02f, w02r, dw02r;
                    eval_w(6, w02f, dw02f); eval_w(7, w02r, dw02r);
                    const float cf_qi = c_i * qi, cr_qj = c_j * qj;
                    er_lane = fmaf(dw02f, cf_qi * s2j, fmaf(dw02r, cr_qj * s2i, er_lane));
                    dqi = fmaf(c_i * w02f, s2j, dqi);
                    g_s2j = fmaf(w02f, cf_qi, g_s2j);
                    g_s2i = fmaf(w02r, cr_qj, g_s2i);
                }
                {
                    float w12sf, dw12sf, w12sr, dw12sr, w12pf, dw12pf, w12pr, dw12pr;
                    eval_w(8, w12sf, dw12sf); eval_w(9, w12sr, dw12sr);
                    eval_w(10, w12pf, dw12pf); eval_w(11, w12pr, dw12pr);
                    const float pi_muj = fmaf(pix, mujx, fmaf(piy, mujy, piz * mujz));
                    const float pj_mui = fmaf(pjx, muix, fmaf(pjy, muiy, pjz * muiz));
                    const float cw12sf = c_i * w12sf, cw12pf = c_i * w12pf;
                    const float cw12sr = c_j * w12sr, cw12pr = c_j * w12pr;
                    er_lane = fmaf(c_i * dw12sf, piu * s2j, fmaf(c_i * dw12pf, pi_muj,
                              fmaf(-c_j * dw12sr, pju * s2i, fmaf(-c_j * dw12pr, pj_mui, er_lane))));
                    const float sf12 = cw12sf * s2j, sr12 = cw12sr * s2i;
                    dux = fmaf(sf12, pix, fmaf(-sr12, pjx, dux));
                    duy = fmaf(sf12, piy, fmaf(-sr12, pjy, duy));
                    duz = fmaf(sf12, piz, fmaf(-sr12, pjz, duz));
                    dpix = fmaf(sf12, ux, fmaf(cw12pf, mujx, dpix));
                    dpiy = fmaf(sf12, uy, fmaf(cw12pf, mujy, dpiy));
                    dpiz = fmaf(sf12, uz, fmaf(cw12pf, mujz, dpiz));
                    g_s2j = fmaf(cw12sf, piu, g_s2j);
                    g_mujx = fmaf(cw12pf, pix, g_mujx); g_mujy = fmaf(cw12pf, piy, g_mujy); g_mujz = fmaf(cw12pf, piz, g_mujz);
                    g_s2i = fmaf(-cw12sr, pju, g_s2i);
                    g_muix = fmaf(-cw12pr, pjx, g_muix); g_muiy = fmaf(-cw12pr, pjy, g_muiy); g_muiz = fmaf(-cw12pr, pjz, g_muiz);
                }
                {
                    float w2s, dw2s, w2p, dw2p, w2d, dw2d;
                    eval_w(12, w2s, dw2s); eval_w(13, w2p, dw2p); eval_w(14, w2d, dw2d);
                    const float mu_dot = fmaf(muix, mujx, fmaf(muiy, mujy, muiz * mujz));
                    const float Q_dot = fmaf(Qi0, Qj0, fmaf(Qi1, Qj1, fmaf(Qi2, Qj2, fmaf(Qi3, Qj3, Qi4 * Qj4))));
                    const float cw2s = c_sum * w2s, cw2p = c_sum * w2p, cw2d = c_sum * w2d;
                    er_lane = fmaf(c_sum * dw2s, s2i * s2j, fmaf(c_sum * dw2p, mu_dot, fmaf(c_sum * dw2d, Q_dot, er_lane)));
                    g_s2i = fmaf(cw2s, s2j, g_s2i);
                    g_muix = fmaf(cw2p, mujx, g_muix); g_muiy = fmaf(cw2p, mujy, g_muiy); g_muiz = fmaf(cw2p, mujz, g_muiz);
                    g_s2j = fmaf(cw2s, s2i, g_s2j);
                    g_mujx = fmaf(cw2p, muix, g_mujx); g_mujy = fmaf(cw2p, muiy, g_mujy); g_mujz = fmaf(cw2p, muiz, g_mujz);

                    float tux, tuy, tuz;
                    lrsk_l2_matvec_only(Qi0, Qi1, Qi2, Qi3, Qi4, g_muix, g_muiy, g_muiz, tux, tuy, tuz);
                    dux = fmaf(2.0f * g_s2i, muix, dux + tux);
                    duy = fmaf(2.0f * g_s2i, muiy, duy + tuy);
                    duz = fmaf(2.0f * g_s2i, muiz, duz + tuz);
                    lrsk_l2_matvec_only(Qj0, Qj1, Qj2, Qj3, Qj4, g_mujx, g_mujy, g_mujz, tux, tuy, tuz);
                    dux = fmaf(2.0f * g_s2j, mujx, dux + tux);
                    duy = fmaf(2.0f * g_s2j, mujy, duy + tuy);
                    duz = fmaf(2.0f * g_s2j, mujz, duz + tuz);

                    dQi0 = fmaf(cw2d, Qj0, dQi0); dQi1 = fmaf(cw2d, Qj1, dQi1); dQi2 = fmaf(cw2d, Qj2, dQi2);
                    dQi3 = fmaf(cw2d, Qj3, dQi3); dQi4 = fmaf(cw2d, Qj4, dQi4);
                    lrsk_l2_dq_acc(g_muix + g_s2i * ux, g_muiy + g_s2i * uy, g_muiz + g_s2i * uz,
                                   ux, uy, uz, dQi0, dQi1, dQi2, dQi3, dQi4);
                }
            }

            const bool p0 = (c & 1) != 0;
            const float r0 = (p0 ? dux : er_lane) + __shfl_xor_sync(qmask, p0 ? er_lane : dux, 1);
            const float r1 = (p0 ? duz : duy)     + __shfl_xor_sync(qmask, p0 ? duy : duz, 1);
            const bool p1 = (c & 2) != 0;
            const float rv = (p1 ? r1 : r0) + __shfl_xor_sync(qmask, p1 ? r0 : r1, 2);
            const float er_t  = __shfl_sync(qmask, rv, 0, 4);
            const float dux_t = __shfl_sync(qmask, rv, 1, 4);
            const float duy_t = __shfl_sync(qmask, rv, 2, 4);
            const float duz_t = __shfl_sync(qmask, rv, 3, 4);
            if (c == 0) {
                const float udot = fmaf(ux, dux_t, fmaf(uy, duy_t, uz * duz_t));
                const float gx = fmaf(er_t, ux, (dux_t - udot * ux) * inv_r);
                const float gy = fmaf(er_t, uy, (duy_t - udot * uy) * inv_r);
                const float gz = fmaf(er_t, uz, (duz_t - udot * uz) * inv_r);
                rfx -= (double)gx; rfy -= (double)gy; rfz -= (double)gz;
                if constexpr (STRESS) {
                    const float4 vl = __ldg(vec_len + eb_b + bi);
                    v0 += 0.5 * (double)vl.x * (double)gx;
                    v1 += 0.25 * ((double)vl.x * (double)gy + (double)vl.y * (double)gx);
                    v2 += 0.25 * ((double)vl.x * (double)gz + (double)vl.z * (double)gx);
                    v3 += 0.5 * (double)vl.y * (double)gy;
                    v4 += 0.25 * ((double)vl.y * (double)gz + (double)vl.z * (double)gy);
                    v5 += 0.5 * (double)vl.z * (double)gz;
                }
            }
        }
        __syncwarp(qmask);
    }

    buf.dqp_pack4[(int64_t)i * 4 + c] = make_float4(dqi, dpix, dpiy, dpiz);
    if constexpr (HAS_C2) {
        buf.dQ_pack03[(int64_t)i * 4 + c] = make_float4(dQi0, dQi1, dQi2, dQi3);
        buf.dQ_pack4[(int64_t)i * 4 + c] = dQi4;
    }
    if (c == 0) {
        if (forces_f32) {
            forces_f32[(int64_t)i * 3 + 0] += (float)rfx;
            forces_f32[(int64_t)i * 3 + 1] += (float)rfy;
            forces_f32[(int64_t)i * 3 + 2] += (float)rfz;
        } else {
            forces[(int64_t)i * 3 + 0] += rfx;
            forces[(int64_t)i * 3 + 1] += rfy;
            forces[(int64_t)i * 3 + 2] += rfz;
        }
        if constexpr (STRESS) {
            atomicAdd(virial + 0, v0); atomicAdd(virial + 1, v1); atomicAdd(virial + 2, v2);
            atomicAdd(virial + 3, v3); atomicAdd(virial + 4, v4); atomicAdd(virial + 5, v5);
        }
    }
}

// =================================================================================================
// Kernel F6: SMEM-Staged Fast Symmetric Heads Backward -> dI_raw1, dI_raw2, total dh
// =================================================================================================
template <bool POL_FIELD, bool HAS_C2, bool SMEM_W>
__global__ void __launch_bounds__(128) lrsk_heads_bwd_sym_kernel(
    int N, int S, int ro_width, int i_stride, LrskCfg cfg,
    const int* __restrict__ species,
    const float* __restrict__ W_head_p16, // [S, ro_width, 16]
    const float* __restrict__ I_raw1,     // [N, 3, i_stride]
    const float* __restrict__ I_raw2,     // [N, 5, i_stride]
    LrskFastBufs buf,
    float* __restrict__ dI_raw1,          // [N, 3, i_stride]
    float* __restrict__ dI_raw2,          // [N, 5, i_stride]
    float* __restrict__ dh_io) {
    extern __shared__ float s_Wh[];       // [S * ro_width * 16] if SMEM_W
    __shared__ float s_gh[8][16];

    const int tid = threadIdx.x;
    if constexpr (SMEM_W) {
        const int tot = S * ro_width * 16;
        for (int idx = tid; idx < tot; idx += 128) s_Wh[idx] = __ldg(W_head_p16 + idx);
        __syncthreads();
    }
    const float* Wp_base = SMEM_W ? s_Wh : W_head_p16;

    const int sid = tid >> 4;
    const int r = tid & 15;
    const unsigned smask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;

    const int nh = cfg.n_head_out;
    const int g1_off = 1 + cfg.c0;
    const int g2_off = g1_off + cfg.c1;

    float gh = 0.0f;
    if (r == 0) {
        gh = buf.sw[i];
    } else if (r < g1_off) {
        const int c = r - 1;
        gh = __ldg(buf.dqp_pack4 + (int64_t)i * 4 + c).x;
    } else if (r < g2_off) {
        const int c = r - g1_off;
        const float g = buf.g1[(int64_t)i * 4 + c];
        const float4 dqp = __ldg(buf.dqp_pack4 + (int64_t)i * 4 + c);
        const float* src = I_raw1 + (int64_t)i * 3 * i_stride + c;
        float* dst = dI_raw1 + (int64_t)i * 3 * i_stride + c;
        gh = fmaf(dqp.y, __ldg(src + 0 * i_stride),
             fmaf(dqp.z, __ldg(src + 1 * i_stride), dqp.w * __ldg(src + 2 * i_stride)));
        dst[0 * i_stride] = dqp.y * g;
        dst[1 * i_stride] = dqp.z * g;
        dst[2 * i_stride] = dqp.w * g;
    } else if constexpr (HAS_C2) {
        if (r < g2_off + cfg.c2) {
            const int c = r - g2_off;
            const float g = buf.g2[(int64_t)i * 4 + c];
            const float4 dQ03 = __ldg(buf.dQ_pack03 + (int64_t)i * 4 + c);
            const float dQ4 = __ldg(buf.dQ_pack4 + (int64_t)i * 4 + c);
            const float* src = I_raw2 + (int64_t)i * 5 * i_stride + c;
            float* dst = dI_raw2 + (int64_t)i * 5 * i_stride + c;
            gh = fmaf(dQ03.x, __ldg(src + 0 * i_stride),
                 fmaf(dQ03.y, __ldg(src + 1 * i_stride),
                 fmaf(dQ03.z, __ldg(src + 2 * i_stride),
                 fmaf(dQ03.w, __ldg(src + 3 * i_stride), dQ4 * __ldg(src + 4 * i_stride)))));
            dst[0 * i_stride] = dQ03.x * g;
            dst[1 * i_stride] = dQ03.y * g;
            dst[2 * i_stride] = dQ03.z * g;
            dst[3 * i_stride] = dQ03.w * g;
            dst[4 * i_stride] = dQ4 * g;
        }
    }
    // Zero any padded channels in dI_raw1 / dI_raw2 when c1 < i_stride or c2 < i_stride
    if (r < i_stride) {
        if (dI_raw1 && r >= cfg.c1) {
            float* dst = dI_raw1 + (int64_t)i * 3 * i_stride + r;
            dst[0] = 0.0f; dst[i_stride] = 0.0f; dst[2 * i_stride] = 0.0f;
        }
        if constexpr (HAS_C2) {
            if (dI_raw2 && r >= cfg.c2) {
                float* dst = dI_raw2 + (int64_t)i * 5 * i_stride + r;
                dst[0] = 0.0f; dst[i_stride] = 0.0f; dst[2 * i_stride] = 0.0f; dst[3 * i_stride] = 0.0f; dst[4 * i_stride] = 0.0f;
            }
        }
    }
    s_gh[sid][r] = gh;
    __syncwarp(smask);

    const int z = species[i];
    const float* Wz = Wp_base + (int64_t)z * ro_width * 16;
    float* dhi = dh_io + (int64_t)i * ro_width;
    for (int k = r; k < ro_width; k += 16) {
        float acc = POL_FIELD ? dhi[k] : 0.0f;
        const float* Wk = Wz + k * 16;
        #pragma unroll
        for (int h_idx = 0; h_idx < 13; ++h_idx) {
            if (h_idx < nh) {
                const float w = SMEM_W ? Wk[h_idx] : __ldg(Wk + h_idx);
                acc = fmaf(s_gh[sid][h_idx], w, acc);
            }
        }
        dhi[k] = acc;
    }
}

// Tiny helper: convert am_l1 [N] -> eA_x [N] (0.5 us!) after fused L1 node lf2_msg NORM=true
__global__ void __launch_bounds__(128) lrsk_am_to_eA_kernel(
    int N, const float* __restrict__ am_l1, int8_t* __restrict__ eA_x) {
    const int i = blockIdx.x * 128 + threadIdx.x;
    if (i < N) eA_x[i] = ilmlip_exp_from_am(am_l1[i]);
}

}  // namespace
