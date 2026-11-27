#pragma once
// Native CUDA kernels for Nonlinear-Gated Full-Range Bilinear Slater-Koster Multipole Student (LR-SK).
// Reuses 100% of ILMlip Layer-1 edge/node kernels, hgemm2 Site MLP, and decoupled ZBL switch.
// Zero Layer-2 message passing, zero atomicAdd on forces or multipole gradients (receiver-owned fwd+rev!).

namespace {

struct LrskCfg {
    int c0, c1, c2, c01, c02, c12;
    int off_w0, off_w01, off_w1s, off_w1p, off_w02, off_w12s, off_w12p, off_w2s, off_w2p, off_w2d;
    int n_terms_pad;
    int n_head_out;
    int head_qA_off, head_qB_off, head_g1A_off, head_g1B_off, head_g2A_off, head_g2B_off;
    int pol_width;
    float half_esc;
    float inv_zc;
};

struct LrskBufs {
    float* qA;   // [N, 16]
    float* qB;   // [N, 16]
    float* pA;   // [N, 3, 16]
    float* pB;   // [N, 3, 16]
    float* QA;   // [N, 5, 8]
    float* QB;   // [N, 5, 8]
    float* g1A;  // [N, 16]
    float* g1B;  // [N, 16]
    float* g2A;  // [N, 8]
    float* g2B;  // [N, 8]
    float* e_site;  // [N]
    float* e_lr;    // [N]
    float* sw;      // [N]
    float* dqA;  // [N, 16]
    float* dqB;  // [N, 16]
    float* dpA;  // [N, 3, 16]
    float* dpB;  // [N, 3, 16]
    float* dQA;  // [N, 5, 8]
    float* dQB;  // [N, 5, 8]
    float* Phi;  // [N, 16]   (pol_field)
    float* Ef;   // [N, 3, 16] (pol_field)
    float* dPhi; // [N, 16]   (pol_field)
    float* dEf;  // [N, 3, 16] (pol_field)
};

__device__ __forceinline__ void lrsk_l2_matvec(
    float q0, float q1, float q2, float q3, float q4,
    float ux, float uy, float uz,
    float& mux, float& muy, float& muz, float& s2) {
    constexpr float INV_S2 = 0.70710678118654752440f;
    constexpr float INV_S6 = 0.40824829046386301637f;
    const float m_xx = -INV_S6 * q2 - INV_S2 * q4;
    const float m_yy = (2.0f * INV_S6) * q2;
    const float m_zz = -INV_S6 * q2 + INV_S2 * q4;
    const float m_xy = INV_S2 * q1;
    const float m_xz = INV_S2 * q0;
    const float m_yz = INV_S2 * q3;
    mux = fmaf(m_xx, ux, fmaf(m_xy, uy, m_xz * uz));
    muy = fmaf(m_xy, ux, fmaf(m_yy, uy, m_yz * uz));
    muz = fmaf(m_xz, ux, fmaf(m_yz, uy, m_zz * uz));
    s2  = fmaf(mux, ux, fmaf(muy, uy, muz * uz));
}

__device__ __forceinline__ void lrsk_l2_matvec_only(
    float q0, float q1, float q2, float q3, float q4,
    float vx, float vy, float vz,
    float& outx, float& outy, float& outz) {
    constexpr float INV_S2 = 0.70710678118654752440f;
    constexpr float INV_S6 = 0.40824829046386301637f;
    const float m_xx = -INV_S6 * q2 - INV_S2 * q4;
    const float m_yy = (2.0f * INV_S6) * q2;
    const float m_zz = -INV_S6 * q2 + INV_S2 * q4;
    const float m_xy = INV_S2 * q1;
    const float m_xz = INV_S2 * q0;
    const float m_yz = INV_S2 * q3;
    outx = fmaf(m_xx, vx, fmaf(m_xy, vy, m_xz * vz));
    outy = fmaf(m_xy, vx, fmaf(m_yy, vy, m_yz * vz));
    outz = fmaf(m_xz, vx, fmaf(m_yz, vy, m_zz * vz));
}

__device__ __forceinline__ void lrsk_l2_dq_acc(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float& dq0, float& dq1, float& dq2, float& dq3, float& dq4) {
    constexpr float INV_S2 = 0.70710678118654752440f;
    constexpr float INV_S6 = 0.40824829046386301637f;
    dq0 = fmaf(INV_S2, ax * bz + az * bx, dq0);
    dq1 = fmaf(INV_S2, ax * by + ay * bx, dq1);
    dq2 = fmaf(INV_S6, 2.0f * ay * by - ax * bx - az * bz, dq2);
    dq3 = fmaf(INV_S2, ay * bz + az * by, dq3);
    dq4 = fmaf(INV_S2, az * bz - ax * bx, dq4);
}

// =================================================================================================
// Kernel 1: Site Multipole Heads Forward (e_site, qA/qB, g1A/g1B -> pA/pB, g2A/g2B -> QA/QB, sw)
// =================================================================================================
template <bool ZBL_SW>
__global__ void __launch_bounds__(128) lrsk_heads_fwd_kernel(
    int N, int ro_width, LrskCfg cfg,
    const int* __restrict__ species,
    const float* __restrict__ h,          // [N, ro_width]
    const float* __restrict__ W_head,     // [S, ro_width, n_head_out]
    const float* __restrict__ b_head,     // [S, n_head_out]
    const float* __restrict__ I_raw1,     // [N, 3, 2*c1] (nullptr if c1==0)
    const float* __restrict__ I_raw2,     // [N, 5, 2*c2] (nullptr if c2==0)
    const double* __restrict__ e_zbl,
    LrskBufs buf) {
    __shared__ float s_h[4][128];
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const float* hi = h + (int64_t)i * ro_width;
    for (int k = lane; k < ro_width; k += 32) s_h[wid][k] = __ldg(hi + k);
    __syncwarp();

    const int z = species[i];
    const int nh = cfg.n_head_out;
    const float* Wh_z = W_head + (int64_t)z * ro_width * nh;
    const float* bh_z = b_head + (int64_t)z * nh;

    // Zero-pad unused lanes once so edge kernels read clean zeros without branch divergence
    if (lane < 16) {
        if (lane >= cfg.c0) {
            buf.qA[(int64_t)i * 16 + lane] = 0.0f;
            buf.qB[(int64_t)i * 16 + lane] = 0.0f;
        }
        if (lane >= cfg.c1) {
            #pragma unroll
            for (int m = 0; m < 3; ++m) {
                buf.pA[((int64_t)i * 3 + m) * 16 + lane] = 0.0f;
                buf.pB[((int64_t)i * 3 + m) * 16 + lane] = 0.0f;
            }
        }
    }
    if (lane < 8 && lane >= cfg.c2) {
        #pragma unroll
        for (int m = 0; m < 5; ++m) {
            buf.QA[((int64_t)i * 5 + m) * 8 + lane] = 0.0f;
            buf.QB[((int64_t)i * 5 + m) * 8 + lane] = 0.0f;
        }
    }

    for (int r = lane; r < nh; r += 32) {
        float acc = __ldg(bh_z + r);
        for (int k = 0; k < ro_width; ++k) acc = fmaf(s_h[wid][k], __ldg(Wh_z + (int64_t)k * nh + r), acc);
        if (r == 0) {
            buf.e_site[i] = acc;
        } else if (r < cfg.head_qB_off) {
            buf.qA[(int64_t)i * 16 + (r - cfg.head_qA_off)] = acc;
        } else if (r < cfg.head_g1A_off) {
            buf.qB[(int64_t)i * 16 + (r - cfg.head_qB_off)] = acc;
        } else if (r < cfg.head_g1B_off) {
            const int c = r - cfg.head_g1A_off;
            buf.g1A[(int64_t)i * 16 + c] = acc;
            const int str = 2 * cfg.c1;
            const float* src = I_raw1 + (int64_t)i * 3 * str + c;
            #pragma unroll
            for (int m = 0; m < 3; ++m) buf.pA[((int64_t)i * 3 + m) * 16 + c] = acc * __ldg(src + m * str);
        } else if (r < cfg.head_g2A_off) {
            const int c = r - cfg.head_g1B_off;
            buf.g1B[(int64_t)i * 16 + c] = acc;
            const int str = 2 * cfg.c1;
            const float* src = I_raw1 + (int64_t)i * 3 * str + cfg.c1 + c;
            #pragma unroll
            for (int m = 0; m < 3; ++m) buf.pB[((int64_t)i * 3 + m) * 16 + c] = acc * __ldg(src + m * str);
        } else if (r < cfg.head_g2B_off) {
            const int c = r - cfg.head_g2A_off;
            buf.g2A[(int64_t)i * 8 + c] = acc;
            const int str = 2 * cfg.c2;
            const float* src = I_raw2 + (int64_t)i * 5 * str + c;
            #pragma unroll
            for (int m = 0; m < 5; ++m) buf.QA[((int64_t)i * 5 + m) * 8 + c] = acc * __ldg(src + m * str);
        } else {
            const int c = r - cfg.head_g2B_off;
            buf.g2B[(int64_t)i * 8 + c] = acc;
            const int str = 2 * cfg.c2;
            const float* src = I_raw2 + (int64_t)i * 5 * str + cfg.c2 + c;
            #pragma unroll
            for (int m = 0; m < 5; ++m) buf.QB[((int64_t)i * 5 + m) * 8 + c] = acc * __ldg(src + m * str);
        }
    }

    if (lane == 0) {
        if constexpr (ZBL_SW) {
            double xc = e_zbl[i] * (double)cfg.inv_zc;
            if (xc < 0.0) xc = 0.0;
            if (xc > 1.0) xc = 1.0;
            const double xc2 = xc * xc, xc3 = xc2 * xc;
            buf.sw[i] = (float)(1.0 - 10.0 * xc3 + 15.0 * xc3 * xc - 6.0 * xc3 * xc2);
        } else {
            buf.sw[i] = 1.0f;
        }
    }
}

// =================================================================================================
// Kernel 2: Pure 1-Pass Receiver-Owned Zero-Atomic LR-SK Edge Forward + Backward (pol_field=False)
//   32 lanes (1 warp) per atom i (4 atoms/block):
//     - Lanes  0..15: forward edge (i -> j), channel c = lane, computes e_lr, dqA/dpA/dQA, f_fwd, virial
//     - Lanes 16..31: reverse edge (j -> i), channel c = lane-16, computes dqB/dpB/dQB, f_rev_ji
//   Force on atom i = sum_j (f_rev_ji - f_fwd) -> ZERO atomicAdd on forces or multipole grads!
// =================================================================================================
template <int NB2, bool SMEM_W, bool STRESS, bool F32_F, bool NEED_GRAD>
__global__ void __launch_bounds__(128) lrsk_edge_fwd_bwd_kernel(
    int N, int S, LrskCfg cfg, ChebParam r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_rad4,  // [S*S, NB2_4, n_terms_pad]
    LrskBufs buf,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    using FT = typename std::conditional<F32_F, float, double>::type;
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int EB = 8;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[4][EB][NB2_4];
    __shared__ float4 s_dg[NEED_GRAD ? 4 : 1][NEED_GRAD ? EB : 1][NEED_GRAD ? NB2_4 : 1];
    __shared__ float4 s_u[4][EB];
    __shared__ int    s_j[4][EB];
    __shared__ int    s_pf[4][EB];
    __shared__ int    s_pr[4][EB];
    __shared__ float  s_swj[4][EB];

    const int n_tp = cfg.n_terms_pad;
    if constexpr (SMEM_W) {
        const int tot_w4 = S * S * NB2_4 * n_tp;
        for (int idx = threadIdx.x; idx < tot_w4; idx += 128) s_W4[idx] = __ldg(W_rad4 + idx);
        __syncthreads();
    }
    const float4* W_ptr = SMEM_W ? s_W4 : W_rad4;

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const bool is_rev = (lane >= 16);
    const int c = lane & 15;
    const unsigned hmask = is_rev ? 0xffff0000u : 0x0000ffffu;
    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float c_i = cfg.half_esc * buf.sw[i];

    // Preload atom i's multipole channel c once into registers!
    float qi = 0.0f;
    float pix = 0.0f, piy = 0.0f, piz = 0.0f;
    float Qi0 = 0.0f, Qi1 = 0.0f, Qi2 = 0.0f, Qi3 = 0.0f, Qi4 = 0.0f;
    if (!is_rev) {
        if (c < cfg.c0) qi = __ldg(buf.qA + (int64_t)i * 16 + c);
        if (c < cfg.c1) {
            const float* p = buf.pA + (int64_t)i * 48 + c;
            pix = __ldg(p); piy = __ldg(p + 16); piz = __ldg(p + 32);
        }
        if (c < cfg.c2) {
            const float* Q = buf.QA + (int64_t)i * 40 + c;
            Qi0 = __ldg(Q); Qi1 = __ldg(Q + 8); Qi2 = __ldg(Q + 16); Qi3 = __ldg(Q + 24); Qi4 = __ldg(Q + 32);
        }
    } else if constexpr (NEED_GRAD) {
        if (c < cfg.c0) qi = __ldg(buf.qB + (int64_t)i * 16 + c);
        if (c < cfg.c1) {
            const float* p = buf.pB + (int64_t)i * 48 + c;
            pix = __ldg(p); piy = __ldg(p + 16); piz = __ldg(p + 32);
        }
        if (c < cfg.c2) {
            const float* Q = buf.QB + (int64_t)i * 40 + c;
            Qi0 = __ldg(Q); Qi1 = __ldg(Q + 8); Qi2 = __ldg(Q + 16); Qi3 = __ldg(Q + 24); Qi4 = __ldg(Q + 32);
        }
    }

    float e_lr_acc = 0.0f;
    float dqi = 0.0f, dpix = 0.0f, dpiy = 0.0f, dpiz = 0.0f;
    float dQi0 = 0.0f, dQi1 = 0.0f, dQi2 = 0.0f, dQi3 = 0.0f, dQi4 = 0.0f;
    FT rfx = (FT)0, rfy = (FT)0, rfz = (FT)0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (lane < bcnt) {
            const int e = eb_b + lane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const int z_j = __ldg(species + j);
            const float r = fmaxf(vl.w, 1e-8f);
            const float inv_r = 1.0f / r;
            s_j[wid][lane] = j;
            s_pf[wid][lane] = z_i * S + z_j;
            s_pr[wid][lane] = z_j * S + z_i;
            s_swj[wid][lane] = __ldg(buf.sw + j);
            s_u[wid][lane] = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            if constexpr (NEED_GRAD) {
                eval_cheb_nb_store<NB2, true>(r, r_max2, s_g[wid][lane], s_dg[wid][lane]);
            } else {
                eval_cheb_nb_store<NB2, false>(r, r_max2, s_g[wid][lane], nullptr);
            }
        }
        __syncwarp();

        for (int bi = 0; bi < bcnt; ++bi) {
            const float4 u4 = s_u[wid][bi];
            const float inv_r = u4.w;
            float er_lane = 0.0f, dux = 0.0f, duy = 0.0f, duz = 0.0f;

            if (!is_rev || NEED_GRAD) {
                const int j = s_j[wid][bi];
                const int pair = is_rev ? s_pr[wid][bi] : s_pf[wid][bi];
                const float ux = is_rev ? -u4.x : u4.x;
                const float uy = is_rev ? -u4.y : u4.y;
                const float uz = is_rev ? -u4.z : u4.z;
                const float c_edge = is_rev ? (cfg.half_esc * s_swj[wid][bi]) : c_i;
                const float4* Wp = W_ptr + (int64_t)pair * NB2_4 * n_tp;

                auto eval_w = [&](int t_idx, float& w, float& dw) {
                    w = 0.0f;
                    dw = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < NB2_4; ++b) {
                        const float4 wv = SMEM_W ? Wp[b * n_tp + t_idx] : __ldg(Wp + b * n_tp + t_idx);
                        w += v1_dot4(wv, s_g[wid][bi][b]);
                        if constexpr (NEED_GRAD) dw += v1_dot4(wv, s_dg[wid][bi][b]);
                    }
                };

                // 2-body baseline on channel 0
                if (c == 0) {
                    float w2b, dw2b;
                    eval_w(0, w2b, dw2b);
                    if (!is_rev) e_lr_acc += w2b;
                    if constexpr (NEED_GRAD) er_lane = fmaf(c_edge, dw2b, er_lane);
                }

                // Load left (a) and right (b) multipoles for channel c!
                // When !is_rev: a=i (qi,pi,Qi), b=j (qB,pB,QB). When is_rev: a=j (qA,pA,QA), b=i (qi,pi,Qi).
                float qa = 0.0f, qb = 0.0f;
                if (c < cfg.c0) {
                    if (!is_rev) { qa = qi; qb = __ldg(buf.qB + (int64_t)j * 16 + c); }
                    else         { qa = __ldg(buf.qA + (int64_t)j * 16 + c); qb = qi; }
                    float w0, dw0;
                    eval_w(cfg.off_w0 + c, w0, dw0);
                    if (!is_rev) e_lr_acc = fmaf(w0, qa * qb, e_lr_acc);
                    if constexpr (NEED_GRAD) {
                        const float qac = c_edge * qa;
                        er_lane = fmaf(dw0, qac * qb, er_lane);
                        if (!is_rev) dqi = fmaf(c_edge * w0, qb, dqi);
                        else         dqi = fmaf(w0, qac, dqi);
                    }
                }

                float pax = 0.0f, pay = 0.0f, paz = 0.0f, pbx = 0.0f, pby = 0.0f, pbz = 0.0f;
                float pau = 0.0f, pbu = 0.0f;
                if (c < cfg.c1) {
                    if (!is_rev) {
                        pax = pix; pay = piy; paz = piz;
                        const float* pb = buf.pB + (int64_t)j * 48 + c;
                        pbx = __ldg(pb); pby = __ldg(pb + 16); pbz = __ldg(pb + 32);
                    } else {
                        const float* pa = buf.pA + (int64_t)j * 48 + c;
                        pax = __ldg(pa); pay = __ldg(pa + 16); paz = __ldg(pa + 32);
                        pbx = pix; pby = piy; pbz = piz;
                    }
                    pau = fmaf(pax, ux, fmaf(pay, uy, paz * uz));
                    pbu = fmaf(pbx, ux, fmaf(pby, uy, pbz * uz));

                    // 0-1 term
                    if (c < cfg.c01) {
                        float w01, dw01;
                        eval_w(cfg.off_w01 + c, w01, dw01);
                        if (!is_rev) e_lr_acc = fmaf(w01, qa * pbu, e_lr_acc);
                        if constexpr (NEED_GRAD) {
                            const float qac = c_edge * qa;
                            const float s01 = w01 * qac;
                            er_lane = fmaf(dw01, qac * pbu, er_lane);
                            dux = fmaf(s01, pbx, dux); duy = fmaf(s01, pby, duy); duz = fmaf(s01, pbz, duz);
                            if (!is_rev) {
                                dqi = fmaf(c_edge * w01, pbu, dqi);
                            } else {
                                dpix = fmaf(s01, ux, dpix); dpiy = fmaf(s01, uy, dpiy); dpiz = fmaf(s01, uz, dpiz);
                            }
                        }
                    }

                    // 1-1 terms (sigma + pi)
                    float w1s, dw1s, w1p, dw1p;
                    eval_w(cfg.off_w1s + c, w1s, dw1s);
                    eval_w(cfg.off_w1p + c, w1p, dw1p);
                    const float p_dot = fmaf(pax, pbx, fmaf(pay, pby, paz * pbz));
                    if (!is_rev) e_lr_acc = fmaf(w1s, pau * pbu, fmaf(w1p, p_dot, e_lr_acc));
                    if constexpr (NEED_GRAD) {
                        const float cw1s = c_edge * w1s;
                        const float cw1p = c_edge * w1p;
                        er_lane = fmaf(c_edge * dw1s, pau * pbu, fmaf(c_edge * dw1p, p_dot, er_lane));
                        const float su_a = cw1s * pbu, su_b = cw1s * pau;
                        dux = fmaf(su_a, pax, fmaf(su_b, pbx, dux));
                        duy = fmaf(su_a, pay, fmaf(su_b, pby, duy));
                        duz = fmaf(su_a, paz, fmaf(su_b, pbz, duz));
                        if (!is_rev) {
                            dpix = fmaf(su_a, ux, fmaf(cw1p, pbx, dpix));
                            dpiy = fmaf(su_a, uy, fmaf(cw1p, pby, dpiy));
                            dpiz = fmaf(su_a, uz, fmaf(cw1p, pbz, dpiz));
                        } else {
                            dpix = fmaf(su_b, ux, fmaf(cw1p, pax, dpix));
                            dpiy = fmaf(su_b, uy, fmaf(cw1p, pay, dpiy));
                            dpiz = fmaf(su_b, uz, fmaf(cw1p, paz, dpiz));
                        }
                    }
                }

                // Quadrupole terms (0-2, 1-2, 2-2) on c < c2
                if (c < cfg.c2) {
                    float Qa0, Qa1, Qa2, Qa3, Qa4, Qb0, Qb1, Qb2, Qb3, Qb4;
                    if (!is_rev) {
                        Qa0 = Qi0; Qa1 = Qi1; Qa2 = Qi2; Qa3 = Qi3; Qa4 = Qi4;
                        const float* Qb = buf.QB + (int64_t)j * 40 + c;
                        Qb0 = __ldg(Qb); Qb1 = __ldg(Qb + 8); Qb2 = __ldg(Qb + 16); Qb3 = __ldg(Qb + 24); Qb4 = __ldg(Qb + 32);
                    } else {
                        const float* Qa = buf.QA + (int64_t)j * 40 + c;
                        Qa0 = __ldg(Qa); Qa1 = __ldg(Qa + 8); Qa2 = __ldg(Qa + 16); Qa3 = __ldg(Qa + 24); Qa4 = __ldg(Qa + 32);
                        Qb0 = Qi0; Qb1 = Qi1; Qb2 = Qi2; Qb3 = Qi3; Qb4 = Qi4;
                    }
                    float mua_x, mua_y, mua_z, s2a, mub_x, mub_y, mub_z, s2b;
                    lrsk_l2_matvec(Qa0, Qa1, Qa2, Qa3, Qa4, ux, uy, uz, mua_x, mua_y, mua_z, s2a);
                    lrsk_l2_matvec(Qb0, Qb1, Qb2, Qb3, Qb4, ux, uy, uz, mub_x, mub_y, mub_z, s2b);

                    float g_s2a = 0.0f, g_mua_x = 0.0f, g_mua_y = 0.0f, g_mua_z = 0.0f;
                    float g_s2b = 0.0f, g_mub_x = 0.0f, g_mub_y = 0.0f, g_mub_z = 0.0f;

                    if (c < cfg.c02) {
                        float w02, dw02;
                        eval_w(cfg.off_w02 + c, w02, dw02);
                        if (!is_rev) e_lr_acc = fmaf(w02, qa * s2b, e_lr_acc);
                        if constexpr (NEED_GRAD) {
                            const float qac = c_edge * qa;
                            er_lane = fmaf(dw02, qac * s2b, er_lane);
                            if (!is_rev) dqi = fmaf(c_edge * w02, s2b, dqi);
                            g_s2b = fmaf(w02, qac, g_s2b);
                        }
                    }

                    if (c < cfg.c12) {
                        float w12s, dw12s, w12p, dw12p;
                        eval_w(cfg.off_w12s + c, w12s, dw12s);
                        eval_w(cfg.off_w12p + c, w12p, dw12p);
                        const float pa_mub = fmaf(pax, mub_x, fmaf(pay, mub_y, paz * mub_z));
                        if (!is_rev) e_lr_acc = fmaf(w12s, pau * s2b, fmaf(w12p, pa_mub, e_lr_acc));
                        if constexpr (NEED_GRAD) {
                            const float cw12s = c_edge * w12s;
                            const float cw12p = c_edge * w12p;
                            er_lane = fmaf(c_edge * dw12s, pau * s2b, fmaf(c_edge * dw12p, pa_mub, er_lane));
                            const float s12 = cw12s * s2b;
                            dux = fmaf(s12, pax, dux); duy = fmaf(s12, pay, duy); duz = fmaf(s12, paz, duz);
                            if (!is_rev) {
                                dpix = fmaf(s12, ux, fmaf(cw12p, mub_x, dpix));
                                dpiy = fmaf(s12, uy, fmaf(cw12p, mub_y, dpiy));
                                dpiz = fmaf(s12, uz, fmaf(cw12p, mub_z, dpiz));
                            }
                            g_s2b = fmaf(cw12s, pau, g_s2b);
                            g_mub_x = fmaf(cw12p, pax, g_mub_x);
                            g_mub_y = fmaf(cw12p, pay, g_mub_y);
                            g_mub_z = fmaf(cw12p, paz, g_mub_z);
                        }
                    }

                    // 2-2 terms (sigma, pi, delta)
                    float w2s, dw2s, w2p, dw2p, w2d, dw2d;
                    eval_w(cfg.off_w2s + c, w2s, dw2s);
                    eval_w(cfg.off_w2p + c, w2p, dw2p);
                    eval_w(cfg.off_w2d + c, w2d, dw2d);
                    const float mu_dot = fmaf(mua_x, mub_x, fmaf(mua_y, mub_y, mua_z * mub_z));
                    const float Q_dot = fmaf(Qa0, Qb0, fmaf(Qa1, Qb1, fmaf(Qa2, Qb2, fmaf(Qa3, Qb3, Qa4 * Qb4))));
                    if (!is_rev) e_lr_acc = fmaf(w2s, s2a * s2b, fmaf(w2p, mu_dot, fmaf(w2d, Q_dot, e_lr_acc)));
                    if constexpr (NEED_GRAD) {
                        const float cw2s = c_edge * w2s, cw2p = c_edge * w2p, cw2d = c_edge * w2d;
                        er_lane = fmaf(c_edge * dw2s, s2a * s2b, fmaf(c_edge * dw2p, mu_dot, fmaf(c_edge * dw2d, Q_dot, er_lane)));
                        g_s2a = fmaf(cw2s, s2b, g_s2a);
                        g_mua_x = fmaf(cw2p, mub_x, g_mua_x); g_mua_y = fmaf(cw2p, mub_y, g_mua_y); g_mua_z = fmaf(cw2p, mub_z, g_mua_z);
                        g_s2b = fmaf(cw2s, s2a, g_s2b);
                        g_mub_x = fmaf(cw2p, mua_x, g_mub_x); g_mub_y = fmaf(cw2p, mua_y, g_mub_y); g_mub_z = fmaf(cw2p, mua_z, g_mub_z);

                        float tux, tuy, tuz;
                        lrsk_l2_matvec_only(Qa0, Qa1, Qa2, Qa3, Qa4, g_mua_x, g_mua_y, g_mua_z, tux, tuy, tuz);
                        dux = fmaf(2.0f * g_s2a, mua_x, dux + tux);
                        duy = fmaf(2.0f * g_s2a, mua_y, duy + tuy);
                        duz = fmaf(2.0f * g_s2a, mua_z, duz + tuz);
                        lrsk_l2_matvec_only(Qb0, Qb1, Qb2, Qb3, Qb4, g_mub_x, g_mub_y, g_mub_z, tux, tuy, tuz);
                        dux = fmaf(2.0f * g_s2b, mub_x, dux + tux);
                        duy = fmaf(2.0f * g_s2b, mub_y, duy + tuy);
                        duz = fmaf(2.0f * g_s2b, mub_z, duz + tuz);

                        if (!is_rev) {
                            dQi0 = fmaf(cw2d, Qb0, dQi0); dQi1 = fmaf(cw2d, Qb1, dQi1); dQi2 = fmaf(cw2d, Qb2, dQi2);
                            dQi3 = fmaf(cw2d, Qb3, dQi3); dQi4 = fmaf(cw2d, Qb4, dQi4);
                            lrsk_l2_dq_acc(g_mua_x + g_s2a * ux, g_mua_y + g_s2a * uy, g_mua_z + g_s2a * uz,
                                           ux, uy, uz, dQi0, dQi1, dQi2, dQi3, dQi4);
                        } else {
                            dQi0 = fmaf(cw2d, Qa0, dQi0); dQi1 = fmaf(cw2d, Qa1, dQi1); dQi2 = fmaf(cw2d, Qa2, dQi2);
                            dQi3 = fmaf(cw2d, Qa3, dQi3); dQi4 = fmaf(cw2d, Qa4, dQi4);
                            lrsk_l2_dq_acc(g_mub_x + g_s2b * ux, g_mub_y + g_s2b * uy, g_mub_z + g_s2b * uz,
                                           ux, uy, uz, dQi0, dQi1, dQi2, dQi3, dQi4);
                        }
                    }
                }
            }

            if constexpr (NEED_GRAD) {
                // Parallel 4-component half-warp reduction (both halves in lockstep!)
                const bool p0 = (c & 1) != 0;
                const float r0 = (p0 ? dux : er_lane) + __shfl_xor_sync(hmask, p0 ? er_lane : dux, 1);
                const float r1 = (p0 ? duz : duy)     + __shfl_xor_sync(hmask, p0 ? duy : duz, 1);
                const bool p1 = (c & 2) != 0;
                float rv = (p1 ? r1 : r0) + __shfl_xor_sync(hmask, p1 ? r0 : r1, 2);
                rv += __shfl_xor_sync(hmask, rv, 4);
                rv += __shfl_xor_sync(hmask, rv, 8);
                // Now within each half-warp: c==0 -> er, c==1 -> dux, c==2 -> duy, c==3 -> duz!
                const float er_h  = __shfl_sync(hmask, rv, is_rev ? 16 : 0);
                const float dux_h = __shfl_sync(hmask, rv, is_rev ? 17 : 1);
                const float duy_h = __shfl_sync(hmask, rv, is_rev ? 18 : 2);
                const float duz_h = __shfl_sync(hmask, rv, is_rev ? 19 : 3);

                float fx_h = 0.0f, fy_h = 0.0f, fz_h = 0.0f;
                if (c == 0) {
                    const float ux = is_rev ? -u4.x : u4.x;
                    const float uy = is_rev ? -u4.y : u4.y;
                    const float uz = is_rev ? -u4.z : u4.z;
                    const float udot = fmaf(ux, dux_h, fmaf(uy, duy_h, uz * duz_h));
                    fx_h = fmaf(er_h, ux, (dux_h - udot * ux) * inv_r);
                    fy_h = fmaf(er_h, uy, (duy_h - udot * uy) * inv_r);
                    fz_h = fmaf(er_h, uz, (duz_h - udot * uz) * inv_r);
                }
                const float fr_x = __shfl_sync(0xffffffffu, fx_h, 16);
                const float fr_y = __shfl_sync(0xffffffffu, fy_h, 16);
                const float fr_z = __shfl_sync(0xffffffffu, fz_h, 16);
                if (lane == 0) {
                    rfx += (FT)(fr_x - fx_h);
                    rfy += (FT)(fr_y - fy_h);
                    rfz += (FT)(fr_z - fz_h);
                    if constexpr (STRESS) {
                        const float4 vl = __ldg(vec_len + eb_b + bi);
                        v0 += (double)vl.x * (double)fx_h;
                        v1 += 0.5 * ((double)vl.x * (double)fy_h + (double)vl.y * (double)fx_h);
                        v2 += 0.5 * ((double)vl.x * (double)fz_h + (double)vl.z * (double)fx_h);
                        v3 += (double)vl.y * (double)fy_h;
                        v4 += 0.5 * ((double)vl.y * (double)fz_h + (double)vl.z * (double)fy_h);
                        v5 += (double)vl.z * (double)fz_h;
                    }
                }
            }
        }
        __syncwarp();
    }

    if (!is_rev) {
        const float e_tot = half_warp_sum_f(e_lr_acc, 0x0000ffffu);
        if (lane == 0) buf.e_lr[i] = cfg.half_esc * e_tot;
    }

    if constexpr (NEED_GRAD) {
        if (!is_rev) {
            if (c < cfg.c0) buf.dqA[(int64_t)i * 16 + c] = dqi;
            if (c < cfg.c1) {
                float* dp = buf.dpA + (int64_t)i * 48 + c;
                dp[0] = dpix; dp[16] = dpiy; dp[32] = dpiz;
            }
            if (c < cfg.c2) {
                float* dQ = buf.dQA + (int64_t)i * 40 + c;
                dQ[0] = dQi0; dQ[8] = dQi1; dQ[16] = dQi2; dQ[24] = dQi3; dQ[32] = dQi4;
            }
            if (lane == 0) {
                if constexpr (F32_F) {
                    forces_f32[(int64_t)i * 3 + 0] += rfx;
                    forces_f32[(int64_t)i * 3 + 1] += rfy;
                    forces_f32[(int64_t)i * 3 + 2] += rfz;
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
        } else {
            if (c < cfg.c0) buf.dqB[(int64_t)i * 16 + c] = dqi;
            if (c < cfg.c1) {
                float* dp = buf.dpB + (int64_t)i * 48 + c;
                dp[0] = dpix; dp[16] = dpiy; dp[32] = dpiz;
            }
            if (c < cfg.c2) {
                float* dQ = buf.dQB + (int64_t)i * 40 + c;
                dQ[0] = dQi0; dQ[8] = dQi1; dQ[16] = dQi2; dQ[24] = dQi3; dQ[32] = dQi4;
            }
        }
    }
}

// =================================================================================================
// Kernel 2b: Energy + ZBL Switch Finalization for Pure 1-Pass LR-SK (pol_field=False)
// =================================================================================================
template <bool ZBL_SW, bool NEED_GRAD>
__global__ void __launch_bounds__(128) lrsk_energy_zbl_kernel(
    int N, LrskCfg cfg,
    const float* __restrict__ e_site, const float* __restrict__ e_lr,
    const double* __restrict__ e_zbl,
    double* __restrict__ energy_out, double* __restrict__ atom_energy_out,
    float* __restrict__ w_zbl_out) {
    const int i = blockIdx.x * 128 + threadIdx.x;
    double val = 0.0;
    if (i < N) {
        const double e_ml = (double)e_site[i] + (double)e_lr[i];
        if constexpr (ZBL_SW) {
            const double ez = e_zbl[i];
            double xc = ez * (double)cfg.inv_zc;
            double sw, dsw_dx;
            if (xc <= 0.0) {
                sw = 1.0; dsw_dx = 0.0;
            } else if (xc >= 1.0) {
                sw = 0.0; dsw_dx = 0.0;
            } else {
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

// =================================================================================================
// Kernel 3: LR-SK Polarization Field Forward (pol_field=True, Pass 1 — 16 lanes/atom, fwd only!)
// =================================================================================================
template <int NB2, bool SMEM_W>
__global__ void __launch_bounds__(128) lrsk_edge_fwd_field_kernel(
    int N, int S, LrskCfg cfg, ChebParam r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_rad4,
    LrskBufs buf) {
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int EB = 4;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[8][EB][NB2_4];
    __shared__ float4 s_u[8][EB];
    __shared__ int    s_j[8][EB];
    __shared__ int    s_pf[8][EB];

    const int n_tp = cfg.n_terms_pad;
    if constexpr (SMEM_W) {
        const int tot_w4 = S * S * NB2_4 * n_tp;
        for (int idx = threadIdx.x; idx < tot_w4; idx += 128) s_W4[idx] = __ldg(W_rad4 + idx);
        __syncthreads();
    }
    const float4* W_ptr = SMEM_W ? s_W4 : W_rad4;

    const int sid = threadIdx.x >> 4;
    const int c = threadIdx.x & 15;
    const unsigned hmask = (sid & 1) ? 0xffff0000u : 0x0000ffffu;
    const int i = blockIdx.x * 8 + sid;
    if (i >= N) return;

    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float qa = (c < cfg.c0) ? __ldg(buf.qA + (int64_t)i * 16 + c) : 0.0f;
    float pax = 0.0f, pay = 0.0f, paz = 0.0f;
    if (c < cfg.c1) {
        const float* p = buf.pA + (int64_t)i * 48 + c;
        pax = __ldg(p); pay = __ldg(p + 16); paz = __ldg(p + 32);
    }
    float Qa0 = 0.0f, Qa1 = 0.0f, Qa2 = 0.0f, Qa3 = 0.0f, Qa4 = 0.0f;
    if (c < cfg.c2) {
        const float* Q = buf.QA + (int64_t)i * 40 + c;
        Qa0 = __ldg(Q); Qa1 = __ldg(Q + 8); Qa2 = __ldg(Q + 16); Qa3 = __ldg(Q + 24); Qa4 = __ldg(Q + 32);
    }

    float e_lr_acc = 0.0f;
    float phi_acc = 0.0f;
    float Efx = 0.0f, Efy = 0.0f, Efz = 0.0f;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (c < bcnt) {
            const int e = eb_b + c;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const float r = fmaxf(vl.w, 1e-8f);
            const float inv_r = 1.0f / r;
            s_j[sid][c] = j;
            s_pf[sid][c] = z_i * S + __ldg(species + j);
            s_u[sid][c] = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            eval_cheb_nb_store<NB2, false>(r, r_max2, s_g[sid][c], nullptr);
        }
        __syncwarp(hmask);

        for (int bi = 0; bi < bcnt; ++bi) {
            const int j = s_j[sid][bi];
            const int pair = s_pf[sid][bi];
            const float4 u4 = s_u[sid][bi];
            const float ux = u4.x, uy = u4.y, uz = u4.z;
            const float4* Wp = W_ptr + (int64_t)pair * NB2_4 * n_tp;

            auto eval_wf = [&](int t_idx) -> float {
                float w = 0.0f;
                #pragma unroll
                for (int b = 0; b < NB2_4; ++b) {
                    const float4 wv = SMEM_W ? Wp[b * n_tp + t_idx] : __ldg(Wp + b * n_tp + t_idx);
                    w += v1_dot4(wv, s_g[sid][bi][b]);
                }
                return w;
            };

            if (c == 0) e_lr_acc += eval_wf(0);

            float qb = 0.0f;
            if (c < cfg.c0) {
                qb = __ldg(buf.qB + (int64_t)j * 16 + c);
                const float w0 = eval_wf(cfg.off_w0 + c);
                const float t0 = w0 * qb;
                e_lr_acc = fmaf(qa, t0, e_lr_acc);
                phi_acc += t0;
            }
            if (c < cfg.c1) {
                const float* pb = buf.pB + (int64_t)j * 48 + c;
                const float pbx = __ldg(pb), pby = __ldg(pb + 16), pbz = __ldg(pb + 32);
                const float pau = fmaf(pax, ux, fmaf(pay, uy, paz * uz));
                const float pbu = fmaf(pbx, ux, fmaf(pby, uy, pbz * uz));
                if (c < cfg.c01) {
                    const float w01 = eval_wf(cfg.off_w01 + c);
                    const float t01p = w01 * pbu;
                    e_lr_acc = fmaf(qa, t01p, e_lr_acc);
                    phi_acc += t01p;
                    const float t01q = w01 * qb;
                    Efx = fmaf(t01q, ux, Efx); Efy = fmaf(t01q, uy, Efy); Efz = fmaf(t01q, uz, Efz);
                }
                const float w1s = eval_wf(cfg.off_w1s + c);
                const float w1p = eval_wf(cfg.off_w1p + c);
                const float w1s_pbu = w1s * pbu;
                e_lr_acc = fmaf(pau, w1s_pbu, fmaf(w1p, fmaf(pax, pbx, fmaf(pay, pby, paz * pbz)), e_lr_acc));
                Efx = fmaf(w1s_pbu, ux, fmaf(w1p, pbx, Efx));
                Efy = fmaf(w1s_pbu, uy, fmaf(w1p, pby, Efy));
                Efz = fmaf(w1s_pbu, uz, fmaf(w1p, pbz, Efz));
            }
            if (c < cfg.c2) {
                const float* Qb = buf.QB + (int64_t)j * 40 + c;
                const float Qb0 = __ldg(Qb), Qb1 = __ldg(Qb + 8), Qb2 = __ldg(Qb + 16), Qb3 = __ldg(Qb + 24), Qb4 = __ldg(Qb + 32);
                float mua_x, mua_y, mua_z, s2a, mub_x, mub_y, mub_z, s2b;
                lrsk_l2_matvec(Qa0, Qa1, Qa2, Qa3, Qa4, ux, uy, uz, mua_x, mua_y, mua_z, s2a);
                lrsk_l2_matvec(Qb0, Qb1, Qb2, Qb3, Qb4, ux, uy, uz, mub_x, mub_y, mub_z, s2b);
                if (c < cfg.c02) e_lr_acc = fmaf(eval_wf(cfg.off_w02 + c), qa * s2b, e_lr_acc);
                if (c < cfg.c12) {
                    const float pau = fmaf(pax, ux, fmaf(pay, uy, paz * uz));
                    const float pa_mub = fmaf(pax, mub_x, fmaf(pay, mub_y, paz * mub_z));
                    e_lr_acc = fmaf(eval_wf(cfg.off_w12s + c), pau * s2b, fmaf(eval_wf(cfg.off_w12p + c), pa_mub, e_lr_acc));
                }
                const float mu_dot = fmaf(mua_x, mub_x, fmaf(mua_y, mub_y, mua_z * mub_z));
                const float Q_dot = fmaf(Qa0, Qb0, fmaf(Qa1, Qb1, fmaf(Qa2, Qb2, fmaf(Qa3, Qb3, Qa4 * Qb4))));
                e_lr_acc = fmaf(eval_wf(cfg.off_w2s + c), s2a * s2b,
                           fmaf(eval_wf(cfg.off_w2p + c), mu_dot,
                           fmaf(eval_wf(cfg.off_w2d + c), Q_dot, e_lr_acc)));
            }
        }
        __syncwarp(hmask);
    }

    const float e_tot = half_warp_sum_f(e_lr_acc, hmask);
    if (c == 0) buf.e_lr[i] = cfg.half_esc * e_tot;
    if (c < cfg.c0) buf.Phi[(int64_t)i * 16 + c] = phi_acc;
    if (c < cfg.c1) {
        float* ef = buf.Ef + (int64_t)i * 48 + c;
        ef[0] = Efx; ef[16] = Efy; ef[32] = Efz;
    }
}

// =================================================================================================
// Kernel 4: Polarization MLP Fwd + Bwd + Total Energy + ZBL Switch (pol_field=True)
//   Also seeds dpA with dpA_pol (from pA . Ef) and writes dh_pol, dPhi, dEf!
// =================================================================================================
template <bool ZBL_SW, bool NEED_GRAD>
__global__ void __launch_bounds__(128) lrsk_pol_mlp_fwd_bwd_kernel(
    int N, int ro_width, LrskCfg cfg,
    const int* __restrict__ species,
    const float* __restrict__ h,       // [N, ro_width]
    const float* __restrict__ pol_W0,  // [in_dim, pol_width]
    const float* __restrict__ pol_b0,  // [pol_width]
    const float* __restrict__ pol_u,   // [S, pol_width] (pre-scaled by energy_scale)
    const double* __restrict__ e_zbl,
    LrskBufs buf,
    double* __restrict__ energy_out, double* __restrict__ atom_energy_out,
    float* __restrict__ w_zbl_out, float* __restrict__ dh_pol_out) {
    __shared__ float s_pin[4][192];
    __shared__ float s_dpre[4][64];
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int f_dim = cfg.c0 + 2 * cfg.c1;
    const int in_dim = f_dim + ro_width;
    const int pw = cfg.pol_width;

    // Assemble pol_in = [Phi (c0), |Ef|^2 (c1), pA.Ef (c1), h (ro_width)]
    if (lane < cfg.c0) s_pin[wid][lane] = buf.Phi[(int64_t)i * 16 + lane];
    if (lane < cfg.c1) {
        const float* ef = buf.Ef + (int64_t)i * 48 + lane;
        const float* pa = buf.pA + (int64_t)i * 48 + lane;
        const float ex = ef[0], ey = ef[16], ez = ef[32];
        const float px = pa[0], py = pa[16], pz = pa[32];
        s_pin[wid][cfg.c0 + lane]          = fmaf(ex, ex, fmaf(ey, ey, ez * ez));
        s_pin[wid][cfg.c0 + cfg.c1 + lane] = fmaf(px, ex, fmaf(py, ey, pz * ez));
    }
    const float* hi = h + (int64_t)i * ro_width;
    for (int k = lane; k < ro_width; k += 32) s_pin[wid][f_dim + k] = __ldg(hi + k);
    __syncwarp();

    const int z = species[i];
    const float* pu_z = pol_u + (int64_t)z * pw;
    const float sw_i = buf.sw[i];
    float e_pol_lane = 0.0f;
    for (int w = lane; w < pw; w += 32) {
        float pre = __ldg(pol_b0 + w);
        for (int k = 0; k < in_dim; ++k) pre = fmaf(s_pin[wid][k], __ldg(pol_W0 + (int64_t)k * pw + w), pre);
        const float sig = 1.0f / (1.0f + __expf(-pre));
        const float act = pre * sig;
        const float uw = __ldg(pu_z + w);
        e_pol_lane = fmaf(act, uw, e_pol_lane);
        if constexpr (NEED_GRAD) {
            const float dsilu = sig * (1.0f + pre * (1.0f - sig));
            s_dpre[wid][w] = sw_i * uw * dsilu;
        }
    }
    const float e_pol = warp_sum_f(e_pol_lane);
    if (lane == 0) {
        const double e_ml = (double)buf.e_site[i] + (double)buf.e_lr[i] + (double)e_pol;
        double val;
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
        atomicAdd(energy_out, val);
    }

    if constexpr (NEED_GRAD) {
        __syncwarp();
        // Backprop s_dpre @ pol_W0^T into d_pin -> dPhi, dEf, dpA_pol, dh_pol!
        for (int k = lane; k < in_dim; k += 32) {
            const float* Wrow = pol_W0 + (int64_t)k * pw;
            float g = 0.0f;
            for (int w = 0; w < pw; ++w) g = fmaf(s_dpre[wid][w], __ldg(Wrow + w), g);
            s_pin[wid][k] = g;
        }
        __syncwarp();
        if (lane < 16) {
            buf.dPhi[(int64_t)i * 16 + lane] = (lane < cfg.c0) ? s_pin[wid][lane] : 0.0f;
            if (lane < cfg.c1) {
                const float d_esq = s_pin[wid][cfg.c0 + lane];
                const float d_pe  = s_pin[wid][cfg.c0 + cfg.c1 + lane];
                const float* ef = buf.Ef + (int64_t)i * 48 + lane;
                const float* pa = buf.pA + (int64_t)i * 48 + lane;
                float* def = buf.dEf + (int64_t)i * 48 + lane;
                float* dpa = buf.dpA + (int64_t)i * 48 + lane;
                #pragma unroll
                for (int m = 0; m < 3; ++m) {
                    const float em = ef[m * 16];
                    def[m * 16] = fmaf(2.0f * d_esq, em, d_pe * pa[m * 16]);
                    dpa[m * 16] = d_pe * em;
                }
            } else {
                float* def = buf.dEf + (int64_t)i * 48 + lane;
                def[0] = 0.0f; def[16] = 0.0f; def[32] = 0.0f;
            }
        }
        float* dho = dh_pol_out + (int64_t)i * ro_width;
        for (int k = lane; k < ro_width; k += 32) dho[k] = s_pin[wid][f_dim + k];
    }
}

// =================================================================================================
// Kernel 5: LR-SK Polarization Edge Backward (pol_field=True, Pass 2 — 32 lanes/atom, ZERO atomics!)
//   Combines e_lr + Phi + Ef edge backward in ONE pass using qA_0 = c_a*qA + dPhi, pA_1 = c_a*pA + dEf!
// =================================================================================================
template <int NB2, bool SMEM_W, bool STRESS, bool F32_F>
__global__ void __launch_bounds__(128) lrsk_edge_bwd_pol_kernel(
    int N, int S, LrskCfg cfg, ChebParam r_max2,
    const int* __restrict__ row_ptr, const int* __restrict__ row_end,
    const int* __restrict__ senders, const float4* __restrict__ vec_len,
    const int* __restrict__ species,
    const float4* __restrict__ W_rad4,
    LrskBufs buf,
    double* __restrict__ forces, float* __restrict__ forces_f32, double* __restrict__ virial) {
    using FT = typename std::conditional<F32_F, float, double>::type;
    constexpr int NB2_4 = (NB2 + 3) / 4;
    constexpr int EB = 8;

    extern __shared__ float4 s_W4[];
    __shared__ float4 s_g[4][EB][NB2_4];
    __shared__ float4 s_dg[4][EB][NB2_4];
    __shared__ float4 s_u[4][EB];
    __shared__ int    s_j[4][EB];
    __shared__ int    s_pf[4][EB];
    __shared__ int    s_pr[4][EB];
    __shared__ float  s_swj[4][EB];

    const int n_tp = cfg.n_terms_pad;
    if constexpr (SMEM_W) {
        const int tot_w4 = S * S * NB2_4 * n_tp;
        for (int idx = threadIdx.x; idx < tot_w4; idx += 128) s_W4[idx] = __ldg(W_rad4 + idx);
        __syncthreads();
    }
    const float4* W_ptr = SMEM_W ? s_W4 : W_rad4;

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const bool is_rev = (lane >= 16);
    const int c = lane & 15;
    const unsigned hmask = is_rev ? 0xffff0000u : 0x0000ffffu;
    const int z_i = species[i];
    const int eb = row_ptr[i], ee = row_end[i];
    const float c_i = cfg.half_esc * buf.sw[i];

    float qi = 0.0f, dPhi_i = 0.0f;
    float pix = 0.0f, piy = 0.0f, piz = 0.0f;
    float dEf_ix = 0.0f, dEf_iy = 0.0f, dEf_iz = 0.0f;
    float Qi0 = 0.0f, Qi1 = 0.0f, Qi2 = 0.0f, Qi3 = 0.0f, Qi4 = 0.0f;
    if (!is_rev) {
        if (c < cfg.c0) {
            qi = __ldg(buf.qA + (int64_t)i * 16 + c);
            dPhi_i = __ldg(buf.dPhi + (int64_t)i * 16 + c);
        }
        if (c < cfg.c1) {
            const float* p = buf.pA + (int64_t)i * 48 + c;
            pix = __ldg(p); piy = __ldg(p + 16); piz = __ldg(p + 32);
            const float* de = buf.dEf + (int64_t)i * 48 + c;
            dEf_ix = __ldg(de); dEf_iy = __ldg(de + 16); dEf_iz = __ldg(de + 32);
        }
        if (c < cfg.c2) {
            const float* Q = buf.QA + (int64_t)i * 40 + c;
            Qi0 = __ldg(Q); Qi1 = __ldg(Q + 8); Qi2 = __ldg(Q + 16); Qi3 = __ldg(Q + 24); Qi4 = __ldg(Q + 32);
        }
    } else {
        if (c < cfg.c0) qi = __ldg(buf.qB + (int64_t)i * 16 + c);
        if (c < cfg.c1) {
            const float* p = buf.pB + (int64_t)i * 48 + c;
            pix = __ldg(p); piy = __ldg(p + 16); piz = __ldg(p + 32);
        }
        if (c < cfg.c2) {
            const float* Q = buf.QB + (int64_t)i * 40 + c;
            Qi0 = __ldg(Q); Qi1 = __ldg(Q + 8); Qi2 = __ldg(Q + 16); Qi3 = __ldg(Q + 24); Qi4 = __ldg(Q + 32);
        }
    }

    float dqi = 0.0f, dpix = 0.0f, dpiy = 0.0f, dpiz = 0.0f;
    float dQi0 = 0.0f, dQi1 = 0.0f, dQi2 = 0.0f, dQi3 = 0.0f, dQi4 = 0.0f;
    FT rfx = (FT)0, rfy = (FT)0, rfz = (FT)0;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;

    for (int eb_b = eb; eb_b < ee; eb_b += EB) {
        const int bcnt = (ee - eb_b < EB) ? (ee - eb_b) : EB;
        if (lane < bcnt) {
            const int e = eb_b + lane;
            const float4 vl = __ldg(vec_len + e);
            const int j = __ldg(senders + e);
            const int z_j = __ldg(species + j);
            const float r = fmaxf(vl.w, 1e-8f);
            const float inv_r = 1.0f / r;
            s_j[wid][lane] = j;
            s_pf[wid][lane] = z_i * S + z_j;
            s_pr[wid][lane] = z_j * S + z_i;
            s_swj[wid][lane] = __ldg(buf.sw + j);
            s_u[wid][lane] = make_float4(vl.x * inv_r, vl.y * inv_r, vl.z * inv_r, inv_r);
            eval_cheb_nb_store<NB2, true>(r, r_max2, s_g[wid][lane], s_dg[wid][lane]);
        }
        __syncwarp();

        for (int bi = 0; bi < bcnt; ++bi) {
            const float4 u4 = s_u[wid][bi];
            const float inv_r = u4.w;
            const int j = s_j[wid][bi];
            const int pair = is_rev ? s_pr[wid][bi] : s_pf[wid][bi];
            const float ux = is_rev ? -u4.x : u4.x;
            const float uy = is_rev ? -u4.y : u4.y;
            const float uz = is_rev ? -u4.z : u4.z;
            const float c_edge = is_rev ? (cfg.half_esc * s_swj[wid][bi]) : c_i;
            const float4* Wp = W_ptr + (int64_t)pair * NB2_4 * n_tp;

            auto eval_w = [&](int t_idx, float& w, float& dw) {
                w = 0.0f; dw = 0.0f;
                #pragma unroll
                for (int b = 0; b < NB2_4; ++b) {
                    const float4 wv = SMEM_W ? Wp[b * n_tp + t_idx] : __ldg(Wp + b * n_tp + t_idx);
                    w  += v1_dot4(wv, s_g[wid][bi][b]);
                    dw += v1_dot4(wv, s_dg[wid][bi][b]);
                }
            };

            float er_lane = 0.0f, dux = 0.0f, duy = 0.0f, duz = 0.0f;

            if (c == 0) {
                float w2b, dw2b;
                eval_w(0, w2b, dw2b);
                er_lane = fmaf(c_edge, dw2b, er_lane);
            }

            float qa = 0.0f, qb = 0.0f, qac = 0.0f, qa_0 = 0.0f;
            if (c < cfg.c0) {
                float dphi_a;
                if (!is_rev) {
                    qa = qi; dphi_a = dPhi_i;
                    qb = __ldg(buf.qB + (int64_t)j * 16 + c);
                } else {
                    qa = __ldg(buf.qA + (int64_t)j * 16 + c);
                    dphi_a = __ldg(buf.dPhi + (int64_t)j * 16 + c);
                    qb = qi;
                }
                qac = c_edge * qa;
                qa_0 = qac + dphi_a;
                float w0, dw0;
                eval_w(cfg.off_w0 + c, w0, dw0);
                er_lane = fmaf(dw0, qa_0 * qb, er_lane);
                if (!is_rev) dqi = fmaf(c_edge * w0, qb, dqi);
                else         dqi = fmaf(w0, qa_0, dqi);
            }

            float pax = 0.0f, pay = 0.0f, paz = 0.0f, pbx = 0.0f, pby = 0.0f, pbz = 0.0f;
            float pau = 0.0f, pbu = 0.0f;
            if (c < cfg.c1) {
                float def_ax, def_ay, def_az;
                if (!is_rev) {
                    pax = pix; pay = piy; paz = piz;
                    def_ax = dEf_ix; def_ay = dEf_iy; def_az = dEf_iz;
                    const float* pb = buf.pB + (int64_t)j * 48 + c;
                    pbx = __ldg(pb); pby = __ldg(pb + 16); pbz = __ldg(pb + 32);
                } else {
                    const float* pa = buf.pA + (int64_t)j * 48 + c;
                    pax = __ldg(pa); pay = __ldg(pa + 16); paz = __ldg(pa + 32);
                    const float* de = buf.dEf + (int64_t)j * 48 + c;
                    def_ax = __ldg(de); def_ay = __ldg(de + 16); def_az = __ldg(de + 32);
                    pbx = pix; pby = piy; pbz = piz;
                }
                pau = fmaf(pax, ux, fmaf(pay, uy, paz * uz));
                pbu = fmaf(pbx, ux, fmaf(pby, uy, pbz * uz));

                if (c < cfg.c01) {
                    float w01, dw01;
                    eval_w(cfg.off_w01 + c, w01, dw01);
                    const float def_u = fmaf(def_ax, ux, fmaf(def_ay, uy, def_az * uz));
                    er_lane = fmaf(dw01, fmaf(qa_0, pbu, qb * def_u), er_lane);
                    const float s01_a = w01 * qa_0;
                    const float s01_b = w01 * qb;
                    dux = fmaf(s01_a, pbx, fmaf(s01_b, def_ax, dux));
                    duy = fmaf(s01_a, pby, fmaf(s01_b, def_ay, duy));
                    duz = fmaf(s01_a, pbz, fmaf(s01_b, def_az, duz));
                    if (!is_rev) {
                        dqi = fmaf(c_edge * w01, pbu, dqi);
                    } else {
                        dqi  = fmaf(w01, def_u, dqi);
                        dpix = fmaf(s01_a, ux, dpix);
                        dpiy = fmaf(s01_a, uy, dpiy);
                        dpiz = fmaf(s01_a, uz, dpiz);
                    }
                }

                // 1-1: pA_1 = c_edge * pA_a + dEf_a
                float w1s, dw1s, w1p, dw1p;
                eval_w(cfg.off_w1s + c, w1s, dw1s);
                eval_w(cfg.off_w1p + c, w1p, dw1p);
                const float pa1x = fmaf(c_edge, pax, def_ax);
                const float pa1y = fmaf(c_edge, pay, def_ay);
                const float pa1z = fmaf(c_edge, paz, def_az);
                const float pa1u = fmaf(pa1x, ux, fmaf(pa1y, uy, pa1z * uz));
                const float p1_dot = fmaf(pa1x, pbx, fmaf(pa1y, pby, pa1z * pbz));
                er_lane = fmaf(dw1s, pa1u * pbu, fmaf(dw1p, p1_dot, er_lane));
                const float su_a = w1s * pbu, su_b = w1s * pa1u;
                dux = fmaf(su_a, pa1x, fmaf(su_b, pbx, dux));
                duy = fmaf(su_a, pa1y, fmaf(su_b, pby, duy));
                duz = fmaf(su_a, pa1z, fmaf(su_b, pbz, duz));
                if (!is_rev) {
                    const float c_su_a = c_edge * su_a, cw1p = c_edge * w1p;
                    dpix = fmaf(c_su_a, ux, fmaf(cw1p, pbx, dpix));
                    dpiy = fmaf(c_su_a, uy, fmaf(cw1p, pby, dpiy));
                    dpiz = fmaf(c_su_a, uz, fmaf(cw1p, pbz, dpiz));
                } else {
                    dpix = fmaf(su_b, ux, fmaf(w1p, pa1x, dpix));
                    dpiy = fmaf(su_b, uy, fmaf(w1p, pa1y, dpiy));
                    dpiz = fmaf(su_b, uz, fmaf(w1p, pa1z, dpiz));
                }
            }

            if (c < cfg.c2) {
                float Qa0, Qa1, Qa2, Qa3, Qa4, Qb0, Qb1, Qb2, Qb3, Qb4;
                if (!is_rev) {
                    Qa0 = Qi0; Qa1 = Qi1; Qa2 = Qi2; Qa3 = Qi3; Qa4 = Qi4;
                    const float* Qb = buf.QB + (int64_t)j * 40 + c;
                    Qb0 = __ldg(Qb); Qb1 = __ldg(Qb + 8); Qb2 = __ldg(Qb + 16); Qb3 = __ldg(Qb + 24); Qb4 = __ldg(Qb + 32);
                } else {
                    const float* Qa = buf.QA + (int64_t)j * 40 + c;
                    Qa0 = __ldg(Qa); Qa1 = __ldg(Qa + 8); Qa2 = __ldg(Qa + 16); Qa3 = __ldg(Qa + 24); Qa4 = __ldg(Qa + 32);
                    Qb0 = Qi0; Qb1 = Qi1; Qb2 = Qi2; Qb3 = Qi3; Qb4 = Qi4;
                }
                float mua_x, mua_y, mua_z, s2a, mub_x, mub_y, mub_z, s2b;
                lrsk_l2_matvec(Qa0, Qa1, Qa2, Qa3, Qa4, ux, uy, uz, mua_x, mua_y, mua_z, s2a);
                lrsk_l2_matvec(Qb0, Qb1, Qb2, Qb3, Qb4, ux, uy, uz, mub_x, mub_y, mub_z, s2b);

                float g_s2a = 0.0f, g_mua_x = 0.0f, g_mua_y = 0.0f, g_mua_z = 0.0f;
                float g_s2b = 0.0f, g_mub_x = 0.0f, g_mub_y = 0.0f, g_mub_z = 0.0f;

                if (c < cfg.c02) {
                    float w02, dw02;
                    eval_w(cfg.off_w02 + c, w02, dw02);
                    er_lane = fmaf(dw02, qac * s2b, er_lane);
                    if (!is_rev) dqi = fmaf(c_edge * w02, s2b, dqi);
                    g_s2b = fmaf(w02, qac, g_s2b);
                }
                if (c < cfg.c12) {
                    float w12s, dw12s, w12p, dw12p;
                    eval_w(cfg.off_w12s + c, w12s, dw12s);
                    eval_w(cfg.off_w12p + c, w12p, dw12p);
                    const float pa_mub = fmaf(pax, mub_x, fmaf(pay, mub_y, paz * mub_z));
                    const float cw12s = c_edge * w12s, cw12p = c_edge * w12p;
                    er_lane = fmaf(c_edge * dw12s, pau * s2b, fmaf(c_edge * dw12p, pa_mub, er_lane));
                    const float s12 = cw12s * s2b;
                    dux = fmaf(s12, pax, dux); duy = fmaf(s12, pay, duy); duz = fmaf(s12, paz, duz);
                    if (!is_rev) {
                        dpix = fmaf(s12, ux, fmaf(cw12p, mub_x, dpix));
                        dpiy = fmaf(s12, uy, fmaf(cw12p, mub_y, dpiy));
                        dpiz = fmaf(s12, uz, fmaf(cw12p, mub_z, dpiz));
                    }
                    g_s2b = fmaf(cw12s, pau, g_s2b);
                    g_mub_x = fmaf(cw12p, pax, g_mub_x);
                    g_mub_y = fmaf(cw12p, pay, g_mub_y);
                    g_mub_z = fmaf(cw12p, paz, g_mub_z);
                }
                float w2s, dw2s, w2p, dw2p, w2d, dw2d;
                eval_w(cfg.off_w2s + c, w2s, dw2s);
                eval_w(cfg.off_w2p + c, w2p, dw2p);
                eval_w(cfg.off_w2d + c, w2d, dw2d);
                const float mu_dot = fmaf(mua_x, mub_x, fmaf(mua_y, mub_y, mua_z * mub_z));
                const float Q_dot = fmaf(Qa0, Qb0, fmaf(Qa1, Qb1, fmaf(Qa2, Qb2, fmaf(Qa3, Qb3, Qa4 * Qb4))));
                const float cw2s = c_edge * w2s, cw2p = c_edge * w2p, cw2d = c_edge * w2d;
                er_lane = fmaf(c_edge * dw2s, s2a * s2b, fmaf(c_edge * dw2p, mu_dot, fmaf(c_edge * dw2d, Q_dot, er_lane)));
                g_s2a = fmaf(cw2s, s2b, g_s2a);
                g_mua_x = fmaf(cw2p, mub_x, g_mua_x); g_mua_y = fmaf(cw2p, mub_y, g_mua_y); g_mua_z = fmaf(cw2p, mub_z, g_mua_z);
                g_s2b = fmaf(cw2s, s2a, g_s2b);
                g_mub_x = fmaf(cw2p, mua_x, g_mub_x); g_mub_y = fmaf(cw2p, mua_y, g_mub_y); g_mub_z = fmaf(cw2p, mua_z, g_mub_z);

                float tux, tuy, tuz;
                lrsk_l2_matvec_only(Qa0, Qa1, Qa2, Qa3, Qa4, g_mua_x, g_mua_y, g_mua_z, tux, tuy, tuz);
                dux = fmaf(2.0f * g_s2a, mua_x, dux + tux);
                duy = fmaf(2.0f * g_s2a, mua_y, duy + tuy);
                duz = fmaf(2.0f * g_s2a, mua_z, duz + tuz);
                lrsk_l2_matvec_only(Qb0, Qb1, Qb2, Qb3, Qb4, g_mub_x, g_mub_y, g_mub_z, tux, tuy, tuz);
                dux = fmaf(2.0f * g_s2b, mub_x, dux + tux);
                duy = fmaf(2.0f * g_s2b, mub_y, duy + tuy);
                duz = fmaf(2.0f * g_s2b, mub_z, duz + tuz);

                if (!is_rev) {
                    dQi0 = fmaf(cw2d, Qb0, dQi0); dQi1 = fmaf(cw2d, Qb1, dQi1); dQi2 = fmaf(cw2d, Qb2, dQi2);
                    dQi3 = fmaf(cw2d, Qb3, dQi3); dQi4 = fmaf(cw2d, Qb4, dQi4);
                    lrsk_l2_dq_acc(g_mua_x + g_s2a * ux, g_mua_y + g_s2a * uy, g_mua_z + g_s2a * uz,
                                   ux, uy, uz, dQi0, dQi1, dQi2, dQi3, dQi4);
                } else {
                    dQi0 = fmaf(cw2d, Qa0, dQi0); dQi1 = fmaf(cw2d, Qa1, dQi1); dQi2 = fmaf(cw2d, Qa2, dQi2);
                    dQi3 = fmaf(cw2d, Qa3, dQi3); dQi4 = fmaf(cw2d, Qa4, dQi4);
                    lrsk_l2_dq_acc(g_mub_x + g_s2b * ux, g_mub_y + g_s2b * uy, g_mub_z + g_s2b * uz,
                                   ux, uy, uz, dQi0, dQi1, dQi2, dQi3, dQi4);
                }
            }

            const bool p0 = (c & 1) != 0;
            const float r0 = (p0 ? dux : er_lane) + __shfl_xor_sync(hmask, p0 ? er_lane : dux, 1);
            const float r1 = (p0 ? duz : duy)     + __shfl_xor_sync(hmask, p0 ? duy : duz, 1);
            const bool p1 = (c & 2) != 0;
            float rv = (p1 ? r1 : r0) + __shfl_xor_sync(hmask, p1 ? r0 : r1, 2);
            rv += __shfl_xor_sync(hmask, rv, 4);
            rv += __shfl_xor_sync(hmask, rv, 8);
            const float er_h  = __shfl_sync(hmask, rv, is_rev ? 16 : 0);
            const float dux_h = __shfl_sync(hmask, rv, is_rev ? 17 : 1);
            const float duy_h = __shfl_sync(hmask, rv, is_rev ? 18 : 2);
            const float duz_h = __shfl_sync(hmask, rv, is_rev ? 19 : 3);

            float fx_h = 0.0f, fy_h = 0.0f, fz_h = 0.0f;
            if (c == 0) {
                const float udot = fmaf(ux, dux_h, fmaf(uy, duy_h, uz * duz_h));
                fx_h = fmaf(er_h, ux, (dux_h - udot * ux) * inv_r);
                fy_h = fmaf(er_h, uy, (duy_h - udot * uy) * inv_r);
                fz_h = fmaf(er_h, uz, (duz_h - udot * uz) * inv_r);
            }
            const float fr_x = __shfl_sync(0xffffffffu, fx_h, 16);
            const float fr_y = __shfl_sync(0xffffffffu, fy_h, 16);
            const float fr_z = __shfl_sync(0xffffffffu, fz_h, 16);
            if (lane == 0) {
                rfx += (FT)(fr_x - fx_h);
                rfy += (FT)(fr_y - fy_h);
                rfz += (FT)(fr_z - fz_h);
                if constexpr (STRESS) {
                    const float4 vl = __ldg(vec_len + eb_b + bi);
                    v0 += (double)vl.x * (double)fx_h;
                    v1 += 0.5 * ((double)vl.x * (double)fy_h + (double)vl.y * (double)fx_h);
                    v2 += 0.5 * ((double)vl.x * (double)fz_h + (double)vl.z * (double)fx_h);
                    v3 += (double)vl.y * (double)fy_h;
                    v4 += 0.5 * ((double)vl.y * (double)fz_h + (double)vl.z * (double)fy_h);
                    v5 += (double)vl.z * (double)fz_h;
                }
            }
        }
        __syncwarp();
    }

    if (!is_rev) {
        if (c < cfg.c0) buf.dqA[(int64_t)i * 16 + c] = dqi;
        if (c < cfg.c1) {
            float* dp = buf.dpA + (int64_t)i * 48 + c;
            dp[0] += dpix; dp[16] += dpiy; dp[32] += dpiz;  // adds onto dpA_pol seeded by Kernel 4!
        }
        if (c < cfg.c2) {
            float* dQ = buf.dQA + (int64_t)i * 40 + c;
            dQ[0] = dQi0; dQ[8] = dQi1; dQ[16] = dQi2; dQ[24] = dQi3; dQ[32] = dQi4;
        }
        if (lane == 0) {
            if constexpr (F32_F) {
                forces_f32[(int64_t)i * 3 + 0] += rfx;
                forces_f32[(int64_t)i * 3 + 1] += rfy;
                forces_f32[(int64_t)i * 3 + 2] += rfz;
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
    } else {
        if (c < cfg.c0) buf.dqB[(int64_t)i * 16 + c] = dqi;
        if (c < cfg.c1) {
            float* dp = buf.dpB + (int64_t)i * 48 + c;
            dp[0] = dpix; dp[16] = dpiy; dp[32] = dpiz;
        }
        if (c < cfg.c2) {
            float* dQ = buf.dQB + (int64_t)i * 40 + c;
            dQ[0] = dQi0; dQ[8] = dQi1; dQ[16] = dQi2; dQ[24] = dQi3; dQ[32] = dQi4;
        }
    }
}

// =================================================================================================
// Kernel 6: Site Multipole Heads Backward (computes dI_raw1, dI_raw2, and total dh for Readout MLP!)
// =================================================================================================
template <bool POL_FIELD>
__global__ void __launch_bounds__(128) lrsk_heads_bwd_kernel(
    int N, int ro_width, LrskCfg cfg,
    const int* __restrict__ species,
    const float* __restrict__ W_head_T,  // [S, n_head_out, ro_width]
    const float* __restrict__ I_raw1,    // [N, 3, 2*c1]
    const float* __restrict__ I_raw2,    // [N, 5, 2*c2]
    LrskBufs buf,
    float* __restrict__ dI_raw1,         // [N, 3, 2*c1]
    float* __restrict__ dI_raw2,         // [N, 5, 2*c2]
    float* __restrict__ dh_io) {         // [N, ro_width] (input dh_pol if POL_FIELD, output total dh)
    __shared__ float s_gh[4][96];
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * 4 + wid;
    if (i >= N) return;

    const int nh = cfg.n_head_out;
    for (int r = lane; r < nh; r += 32) {
        if (r == 0) {
            s_gh[wid][0] = buf.sw[i];
        } else if (r < cfg.head_qB_off) {
            s_gh[wid][r] = buf.dqA[(int64_t)i * 16 + (r - cfg.head_qA_off)];
        } else if (r < cfg.head_g1A_off) {
            s_gh[wid][r] = buf.dqB[(int64_t)i * 16 + (r - cfg.head_qB_off)];
        } else if (r < cfg.head_g1B_off) {
            const int c = r - cfg.head_g1A_off;
            const float g = buf.g1A[(int64_t)i * 16 + c];
            const int str = 2 * cfg.c1;
            const float* src = I_raw1 + (int64_t)i * 3 * str + c;
            float* dst = dI_raw1 + (int64_t)i * 3 * str + c;
            const float* dp = buf.dpA + (int64_t)i * 48 + c;
            float dg = 0.0f;
            #pragma unroll
            for (int m = 0; m < 3; ++m) {
                const float dpm = dp[m * 16];
                dg = fmaf(dpm, __ldg(src + m * str), dg);
                dst[m * str] = dpm * g;
            }
            s_gh[wid][r] = dg;
        } else if (r < cfg.head_g2A_off) {
            const int c = r - cfg.head_g1B_off;
            const float g = buf.g1B[(int64_t)i * 16 + c];
            const int str = 2 * cfg.c1;
            const float* src = I_raw1 + (int64_t)i * 3 * str + cfg.c1 + c;
            float* dst = dI_raw1 + (int64_t)i * 3 * str + cfg.c1 + c;
            const float* dp = buf.dpB + (int64_t)i * 48 + c;
            float dg = 0.0f;
            #pragma unroll
            for (int m = 0; m < 3; ++m) {
                const float dpm = dp[m * 16];
                dg = fmaf(dpm, __ldg(src + m * str), dg);
                dst[m * str] = dpm * g;
            }
            s_gh[wid][r] = dg;
        } else if (r < cfg.head_g2B_off) {
            const int c = r - cfg.head_g2A_off;
            const float g = buf.g2A[(int64_t)i * 8 + c];
            const int str = 2 * cfg.c2;
            const float* src = I_raw2 + (int64_t)i * 5 * str + c;
            float* dst = dI_raw2 + (int64_t)i * 5 * str + c;
            const float* dQ = buf.dQA + (int64_t)i * 40 + c;
            float dg = 0.0f;
            #pragma unroll
            for (int m = 0; m < 5; ++m) {
                const float dQm = dQ[m * 8];
                dg = fmaf(dQm, __ldg(src + m * str), dg);
                dst[m * str] = dQm * g;
            }
            s_gh[wid][r] = dg;
        } else {
            const int c = r - cfg.head_g2B_off;
            const float g = buf.g2B[(int64_t)i * 8 + c];
            const int str = 2 * cfg.c2;
            const float* src = I_raw2 + (int64_t)i * 5 * str + cfg.c2 + c;
            float* dst = dI_raw2 + (int64_t)i * 5 * str + cfg.c2 + c;
            const float* dQ = buf.dQB + (int64_t)i * 40 + c;
            float dg = 0.0f;
            #pragma unroll
            for (int m = 0; m < 5; ++m) {
                const float dQm = dQ[m * 8];
                dg = fmaf(dQm, __ldg(src + m * str), dg);
                dst[m * str] = dQm * g;
            }
            s_gh[wid][r] = dg;
        }
    }
    __syncwarp();

    const int z = species[i];
    const float* WhT_z = W_head_T + (int64_t)z * nh * ro_width;
    float* dhi = dh_io + (int64_t)i * ro_width;
    for (int k = lane; k < ro_width; k += 32) {
        float acc = POL_FIELD ? dhi[k] : 0.0f;
        for (int r = 0; r < nh; ++r) acc = fmaf(s_gh[wid][r], __ldg(WhT_z + (int64_t)r * ro_width + k), acc);
        dhi[k] = acc;
    }
}

}  // namespace
