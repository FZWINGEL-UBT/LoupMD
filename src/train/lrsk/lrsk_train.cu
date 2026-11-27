// Fast native CUDA kernels for exact 1st- and 2nd-order training of ILMlip-LRSK (FastLRSKStudent).
// Pure C API (ctypes-callable from PyTorch with zero PyTorch C++ ABI dependency) + fatbin for sm_80/86/89/120.
// Supports BOTH float32 (training) and float64 (exact <1e-12 autograd gradcheck).
//
// Kernels:
//   1. l1_precompute_G1  : precomputes constant G1[N, S, K1, N_SH] once per dataset (L1 fwd becomes 1 tiny GEMM!)
//   2. l1_force_fwd      : G_bwd[N, S, K1, N_SH] -> (F_l1[N, 3], V_l1[G, 3, 3])
//   3. l1_force_bwd2     : (dF_l1[N, 3], dV_l1[G, 3, 3]) -> dG_bwd[N, S, K1, N_SH] (ZERO atomics!)
//   4. lrsk_pair_fwd     : (q, p, [Q], W_eff) -> (e_lr[N], Phi[N, c0], Ef[N, c1, 3])
//   5. lrsk_pair_bwd1    : (q, p, [Q], c_elr, c_Phi, c_E, W_eff) -> (dq, dp, [dQ], [F_lr], [V_lr], [g_Weff])
//   6. lrsk_pair_bwd2    : exact 2nd-order adjoint of lrsk_pair_bwd1 (dual-number JVP adjoint, ZERO [E,...] tensors!)

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace loup_lrsk_train {

static constexpr int MAX_K = 16;
static constexpr int MAX_SH = 16;

// -----------------------------------------------------------------------------
// Math helpers (float / double templated)
// -----------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T t_sqrt(T x);
template <>
__device__ __forceinline__ float t_sqrt<float>(float x) { return sqrtf(x); }
template <>
__device__ __forceinline__ double t_sqrt<double>(double x) { return sqrt(x); }

template <typename T>
__device__ __forceinline__ T t_max(T a, T b) { return a > b ? a : b; }
template <typename T>
__device__ __forceinline__ T t_min(T a, T b) { return a < b ? a : b; }

template <typename T>
__device__ __forceinline__ T warp_sum(T v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

template <typename T>
struct Vec3 {
    T x, y, z;
    __device__ __forceinline__ Vec3() : x(0), y(0), z(0) {}
    __device__ __forceinline__ Vec3(T a, T b, T c) : x(a), y(b), z(c) {}
    __device__ __forceinline__ Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    __device__ __forceinline__ Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    __device__ __forceinline__ Vec3 operator*(T s) const { return Vec3(x * s, y * s, z * s); }
    __device__ __forceinline__ Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
};

template <typename T>
__device__ __forceinline__ T dot3(const Vec3<T>& a, const Vec3<T>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T>
__device__ __forceinline__ void atomicAddVec3(T* dst, const Vec3<T>& v) {
    atomicAdd(dst + 0, v.x);
    atomicAdd(dst + 1, v.y);
    atomicAdd(dst + 2, v.z);
}

// Exact Chebyshev radial basis + smooth polynomial envelope & exact d/dr
template <typename T>
__device__ __forceinline__ void cheb_val_deriv(
    T r, T r_cut, T warp, int K, T* __restrict__ g, T* __restrict__ dg
) {
    const T inv_rc = T(1) / r_cut;
    const T t_raw = r * inv_rc;
    if (t_raw >= T(1)) {
        for (int k = 0; k < K; ++k) {
            g[k] = T(0);
            if (dg) dg[k] = T(0);
        }
        return;
    }
    const T t = t_raw > T(0) ? t_raw : T(0);
    const T t2 = t * t;
    const T t4 = t2 * t2;
    const T t5 = t4 * t;
    const T env = T(1) - T(21) * t5 + T(35) * t5 * t - T(15) * t5 * t2;
    const T om = T(1) - t;
    const T denv_dt = T(-105) * t4 * om * om;

    T u, du_dt;
    if (warp > T(0)) {
        const T denom = T(1) + warp * t;
        const T inv_d = T(1) / denom;
        u = (T(1) + warp) * t * inv_d;
        du_dt = (T(1) + warp) * inv_d * inv_d;
    } else {
        u = t;
        du_dt = T(1);
    }
    const T x = T(2) * u - T(1);
    const T dx_dt = T(2) * du_dt;

    T T0 = T(1), T1 = x;
    T dT0 = T(0), dT1 = T(1);
    if (K >= 1) {
        g[0] = env;
        if (dg) dg[0] = inv_rc * denv_dt;
    }
    if (K >= 2) {
        g[1] = x * env;
        if (dg) dg[1] = inv_rc * (dx_dt * env + x * denv_dt);
    }
    for (int k = 2; k < K; ++k) {
        const T Tk = T(2) * x * T1 - T0;
        const T dTk = T(2) * T1 + T(2) * x * dT1 - dT0;
        g[k] = Tk * env;
        if (dg) dg[k] = inv_rc * (dTk * dx_dt * env + Tk * denv_dt);
        T0 = T1; T1 = Tk;
        dT0 = dT1; dT1 = dTk;
    }
}

// Exact e3nn component-normalized real spherical harmonics Y_m(u) and Cartesian dY_m/du for lmax <= 3
template <typename T>
__device__ __forceinline__ void eval_sh_and_grad(
    const Vec3<T>& u, int lmax, T* __restrict__ Y, Vec3<T>* __restrict__ dY
) {
    const T x = u.x, y = u.y, z = u.z;
    const T S3  = T(1.73205080756887729353);
    const T S5  = T(2.23606797749978969641);
    const T S7  = T(2.64575131106459059050);
    const T S56 = T(0.91287092917527685576); // sqrt(5/6)
    const T S38 = T(0.61237243569579452455); // sqrt(3/8)

    Y[0] = T(1);
    if (dY) dY[0] = Vec3<T>(T(0), T(0), T(0));
    if (lmax >= 1) {
        Y[1] = S3 * x; Y[2] = S3 * y; Y[3] = S3 * z;
        if (dY) {
            dY[1] = Vec3<T>(S3, T(0), T(0));
            dY[2] = Vec3<T>(T(0), S3, T(0));
            dY[3] = Vec3<T>(T(0), T(0), S3);
        }
    }
    if (lmax >= 2) {
        const T y2 = y * y;
        const T x2z2 = x * x + z * z;
        const T s20 = S3 * x * z;
        const T s21 = S3 * x * y;
        const T s22 = y2 - T(0.5) * x2z2;
        const T s23 = S3 * y * z;
        const T s24 = (T(0.5) * S3) * (z * z - x * x);

        Y[4] = S5 * s20;
        Y[5] = S5 * s21;
        Y[6] = S5 * s22;
        Y[7] = S5 * s23;
        Y[8] = S5 * s24;

        const T ds20_x = S3 * z, ds20_z = S3 * x;
        const T ds24_x = -S3 * x, ds24_z = S3 * z;

        if (dY) {
            dY[4] = Vec3<T>(S5 * ds20_x, T(0), S5 * ds20_z);
            dY[5] = Vec3<T>(S5 * S3 * y, S5 * S3 * x, T(0));
            dY[6] = Vec3<T>(-S5 * x, T(2) * S5 * y, -S5 * z);
            dY[7] = Vec3<T>(T(0), S5 * S3 * z, S5 * S3 * y);
            dY[8] = Vec3<T>(S5 * ds24_x, T(0), S5 * ds24_z);
        }
        if (lmax >= 3) {
            const T t4 = T(4) * y2 - x2z2;
            const T s30 = S56 * (s20 * z + s24 * x);
            const T s31 = S5 * (s20 * y);
            const T s32 = S38 * (t4 * x);
            const T s33 = T(0.5) * y * (T(2) * y2 - T(3) * x2z2);
            const T s34 = S38 * (z * t4);
            const T s35 = S5 * (s24 * y);
            const T s36 = S56 * (s24 * z - s20 * x);

            Y[9]  = S7 * s30;
            Y[10] = S7 * s31;
            Y[11] = S7 * s32;
            Y[12] = S7 * s33;
            Y[13] = S7 * s34;
            Y[14] = S7 * s35;
            Y[15] = S7 * s36;

            if (dY) {
                const T dt4_x = T(-2) * x, dt4_y = T(8) * y, dt4_z = T(-2) * z;
                const T c56 = S7 * S56;
                const T c5  = S7 * S5;
                const T c38 = S7 * S38;

                dY[9]  = Vec3<T>(c56 * (ds20_x * z + ds24_x * x + s24), T(0), c56 * (ds20_z * z + s20 + ds24_z * x));
                dY[10] = Vec3<T>(c5 * ds20_x * y, c5 * s20, c5 * ds20_z * y);
                dY[11] = Vec3<T>(c38 * (t4 + x * dt4_x), c38 * x * dt4_y, c38 * x * dt4_z);
                dY[12] = Vec3<T>(S7 * T(-3) * x * y, S7 * (T(3) * y2 - T(1.5) * x2z2), S7 * T(-3) * y * z);
                dY[13] = Vec3<T>(c38 * z * dt4_x, c38 * z * dt4_y, c38 * (t4 + z * dt4_z));
                dY[14] = Vec3<T>(c5 * ds24_x * y, c5 * s24, c5 * ds24_z * y);
                dY[15] = Vec3<T>(c56 * (ds24_x * z - ds20_x * x - s20), T(0), c56 * (ds24_z * z + s24 - ds20_z * x));
            }
        }
    }
}

// L=2 traceless symmetric 3x3 matvec helpers
template <typename T>
__device__ __forceinline__ Vec3<T> l2_matvec_only(const T* __restrict__ q, const Vec3<T>& v) {
    const T inv_s2 = T(0.70710678118654752440);
    const T inv_s6 = T(0.40824829046386301637);
    const T q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3], q4 = q[4];
    const T m_xx = -inv_s6 * q2 - inv_s2 * q4;
    const T m_yy = (T(2) * inv_s6) * q2;
    const T m_zz = -inv_s6 * q2 + inv_s2 * q4;
    const T m_xy = inv_s2 * q1;
    const T m_xz = inv_s2 * q0;
    const T m_yz = inv_s2 * q3;
    return Vec3<T>(
        m_xx * v.x + m_xy * v.y + m_xz * v.z,
        m_xy * v.x + m_yy * v.y + m_yz * v.z,
        m_xz * v.x + m_yz * v.y + m_zz * v.z
    );
}

// Accumulates scale * d(a^T M(Q) b)/dQ into dQ[0..4] (non-atomic, register or local ptr)
template <typename T>
__device__ __forceinline__ void l2_add_outer_reg(T* __restrict__ dQ, const Vec3<T>& a, const Vec3<T>& b, T sc) {
    const T inv_s2 = T(0.70710678118654752440) * sc;
    const T inv_s6 = T(0.40824829046386301637) * sc;
    dQ[0] += inv_s2 * (a.x * b.z + a.z * b.x);
    dQ[1] += inv_s2 * (a.x * b.y + a.y * b.x);
    dQ[2] += inv_s6 * (T(2) * a.y * b.y - a.x * b.x - a.z * b.z);
    dQ[3] += inv_s2 * (a.y * b.z + a.z * b.y);
    dQ[4] += inv_s2 * (a.z * b.z - a.x * b.x);
}

template <typename T>
__device__ __forceinline__ void l2_add_outer_atomic(T* __restrict__ dQ, const Vec3<T>& a, const Vec3<T>& b, T sc) {
    const T inv_s2 = T(0.70710678118654752440) * sc;
    const T inv_s6 = T(0.40824829046386301637) * sc;
    atomicAdd(dQ + 0, inv_s2 * (a.x * b.z + a.z * b.x));
    atomicAdd(dQ + 1, inv_s2 * (a.x * b.y + a.y * b.x));
    atomicAdd(dQ + 2, inv_s6 * (T(2) * a.y * b.y - a.x * b.x - a.z * b.z));
    atomicAdd(dQ + 3, inv_s2 * (a.y * b.z + a.z * b.y));
    atomicAdd(dQ + 4, inv_s2 * (a.z * b.z - a.x * b.x));
}

// =============================================================================
// 1. Precompute constant G1[N, S, K1, N_SH] once per dataset
// =============================================================================
template <typename T>
__global__ void l1_precompute_G1_kernel(
    int N, int S, int K1, int lmax, T r_max, T warp,
    const int* __restrict__ row_ptr,
    const int* __restrict__ snd,
    const T* __restrict__ vec,
    const int* __restrict__ z_idx,
    T* __restrict__ G1 // [N, S, K1, N_SH], zero-initialized
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int n_sh = (lmax + 1) * (lmax + 1);
    const int e0 = row_ptr[i], e1 = row_ptr[i + 1];
    T* dst_i = G1 + (int64_t)i * S * K1 * n_sh;
    T g[MAX_K], Y[MAX_SH];
    for (int e = e0; e < e1; ++e) {
        const int j = snd[e];
        const int zj = z_idx[j];
        const Vec3<T> v(vec[3 * e + 0], vec[3 * e + 1], vec[3 * e + 2]);
        const T r = t_max(t_sqrt(dot3(v, v)), T(1e-8));
        const Vec3<T> u = v * (T(1) / r);
        cheb_val_deriv<T>(r, r_max, warp, K1, g, nullptr);
        eval_sh_and_grad<T>(u, lmax, Y, nullptr);
        T* dst_z = dst_i + (int64_t)zj * K1 * n_sh;
        for (int k = 0; k < K1; ++k) {
            const T gk = g[k];
            T* dst_k = dst_z + k * n_sh;
            for (int m = 0; m < n_sh; ++m) {
                dst_k[m] += gk * Y[m];
            }
        }
    }
}

// =============================================================================
// 2. L1 Force Forward: G_bwd[N, S, K1, N_SH] -> (F_l1[N, 3], V_l1[G, 3, 3])
//    1 warp per receiver atom i (32 lanes iterate over edges e in row(i)).
// =============================================================================
template <typename T>
__global__ void l1_force_fwd_kernel(
    int N, int S, int K1, int lmax, T r_max, T warp, int compute_virial,
    const int* __restrict__ row_ptr,
    const int* __restrict__ snd,
    const T* __restrict__ vec,
    const int* __restrict__ z_idx,
    const int* __restrict__ atom_batch,
    const T* __restrict__ G_bwd, // [N, S, K1, N_SH]
    T* __restrict__ F_out,       // [N, 3] (zero-initialized before launch)
    T* __restrict__ V_out        // [G, 9] (zero-initialized before launch, optional)
) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= N) return;
    const int i = wid;
    const int n_sh = (lmax + 1) * (lmax + 1);
    const int e0 = row_ptr[i], e1 = row_ptr[i + 1];
    const T* Gi = G_bwd + (int64_t)i * S * K1 * n_sh;

    Vec3<T> fi_acc(0, 0, 0);
    T v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;

    T g[MAX_K], dg[MAX_K], Y[MAX_SH];
    Vec3<T> dY[MAX_SH];

    for (int e = e0 + lane; e < e1; e += 32) {
        const int j = snd[e];
        const int zj = z_idx[j];
        const Vec3<T> rv(vec[3 * e + 0], vec[3 * e + 1], vec[3 * e + 2]);
        const T r = t_max(t_sqrt(dot3(rv, rv)), T(1e-8));
        const T inv_r = T(1) / r;
        const Vec3<T> u = rv * inv_r;

        cheb_val_deriv<T>(r, r_max, warp, K1, g, dg);
        eval_sh_and_grad<T>(u, lmax, Y, dY);

        const T* Gz = Gi + (int64_t)zj * K1 * n_sh;
        T grad_r = T(0);
        Vec3<T> grad_u(0, 0, 0);
        for (int m = 0; m < n_sh; ++m) {
            T sum_g = T(0), sum_dg = T(0);
            for (int k = 0; k < K1; ++k) {
                const T c = Gz[k * n_sh + m];
                sum_g += c * g[k];
                sum_dg += c * dg[k];
            }
            grad_r += sum_dg * Y[m];
            grad_u.x += sum_g * dY[m].x;
            grad_u.y += sum_g * dY[m].y;
            grad_u.z += sum_g * dY[m].z;
        }
        const T du = dot3(grad_u, u);
        const Vec3<T> gv(
            grad_r * u.x + (grad_u.x - du * u.x) * inv_r,
            grad_r * u.y + (grad_u.y - du * u.y) * inv_r,
            grad_r * u.z + (grad_u.z - du * u.z) * inv_r
        );
        fi_acc.x -= gv.x;
        fi_acc.y -= gv.y;
        fi_acc.z -= gv.z;
        atomicAddVec3(F_out + (int64_t)j * 3, gv);

        if (compute_virial) {
            v_xx += rv.x * gv.x;
            v_yy += rv.y * gv.y;
            v_zz += rv.z * gv.z;
            v_xy += T(0.5) * (rv.x * gv.y + rv.y * gv.x);
            v_xz += T(0.5) * (rv.x * gv.z + rv.z * gv.x);
            v_yz += T(0.5) * (rv.y * gv.z + rv.z * gv.y);
        }
    }

    fi_acc.x = warp_sum(fi_acc.x);
    fi_acc.y = warp_sum(fi_acc.y);
    fi_acc.z = warp_sum(fi_acc.z);
    if (lane == 0) {
        atomicAddVec3(F_out + (int64_t)i * 3, fi_acc);
    }
    if (compute_virial) {
        v_xx = warp_sum(v_xx); v_yy = warp_sum(v_yy); v_zz = warp_sum(v_zz);
        v_xy = warp_sum(v_xy); v_xz = warp_sum(v_xz); v_yz = warp_sum(v_yz);
        if (lane == 0) {
            const int g_id = atom_batch[i];
            T* vg = V_out + (int64_t)g_id * 9;
            atomicAdd(vg + 0, v_xx); atomicAdd(vg + 1, v_xy); atomicAdd(vg + 2, v_xz);
            atomicAdd(vg + 3, v_xy); atomicAdd(vg + 4, v_yy); atomicAdd(vg + 5, v_yz);
            atomicAdd(vg + 6, v_xz); atomicAdd(vg + 7, v_yz); atomicAdd(vg + 8, v_zz);
        }
    }
}

// =============================================================================
// 3. L1 Force 2nd-Order Backward (ZERO ATOMICS!):
//    (dF_l1[N, 3], dV_l1[G, 9]) -> dG_bwd[N, S, K1, N_SH]
//    1 warp per receiver atom i: processes edges in chunks of 4; all 32 lanes
//    accumulate their disjoint (k, m) indices with ZERO atomics & 100% coalesced writes!
// =============================================================================
template <typename T>
__global__ void l1_force_bwd2_kernel(
    int N, int S, int K1, int lmax, T r_max, T warp, int has_dV,
    const int* __restrict__ row_ptr,
    const int* __restrict__ snd,
    const T* __restrict__ vec,
    const int* __restrict__ z_idx,
    const int* __restrict__ atom_batch,
    const T* __restrict__ dF,    // [N, 3]
    const T* __restrict__ dV,    // [G, 9] or nullptr
    T* __restrict__ dG_out       // [N, S, K1, N_SH]
) {
    constexpr int WPB = 4;
    constexpr int CHUNK = 8;
    const int local_w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * WPB + local_w;
    if (i >= N) return;

    const int n_sh = (lmax + 1) * (lmax + 1);
    const int km_tot = K1 * n_sh;
    const int tot_el = S * km_tot;
    const int e0 = row_ptr[i], e1 = row_ptr[i + 1];

    const Vec3<T> dFi(dF[3 * i + 0], dF[3 * i + 1], dF[3 * i + 2]);
    T m00 = 0, m01 = 0, m02 = 0, m11 = 0, m12 = 0, m22 = 0;
    if (has_dV && dV != nullptr) {
        const T* dvg = dV + (int64_t)atom_batch[i] * 9;
        m00 = dvg[0];
        m11 = dvg[4];
        m22 = dvg[8];
        m01 = T(0.5) * (dvg[1] + dvg[3]);
        m02 = T(0.5) * (dvg[2] + dvg[6]);
        m12 = T(0.5) * (dvg[5] + dvg[7]);
    }

    // Small per-warp SMEM staging for CHUNK edges:
    __shared__ T s_g[WPB][CHUNK][MAX_K];
    __shared__ T s_dgw[WPB][CHUNK][MAX_K];
    __shared__ T s_Y[WPB][CHUNK][MAX_SH];
    __shared__ T s_dYwt[WPB][CHUNK][MAX_SH];
    __shared__ int s_zj[WPB][CHUNK];

    // Zero output slice for atom i (disjoint across lanes, 100% coalesced!)
    T* dst_i = dG_out + (int64_t)i * tot_el;
    for (int idx = lane; idx < tot_el; idx += 32) {
        dst_i[idx] = T(0);
    }

    // Also keep fast register accumulators when tot_el <= 256*4 = 1024 -> <=32 regs/lane?
    // Even cleaner: keep SMEM buffer of size 1024 floats/warp (4KB*4 = 16KB) when tot_el <= 1024 (float)
    // or <= 512 (double), else accumulate directly in dst_i (still ZERO atomics & L1-cached!).
    constexpr int MAX_SMEM_EL = (sizeof(T) == 4) ? 1024 : 512;
    __shared__ T s_acc[WPB][MAX_SMEM_EL];
    const bool use_smem = (tot_el <= MAX_SMEM_EL);
    if (use_smem) {
        for (int idx = lane; idx < tot_el; idx += 32) {
            s_acc[local_w][idx] = T(0);
        }
    }
    __syncwarp();

    for (int eb = e0; eb < e1; eb += CHUNK) {
        const int cnt = (eb + CHUNK <= e1) ? CHUNK : (e1 - eb);
        if (lane < cnt) {
            const int e = eb + lane;
            const int j = snd[e];
            s_zj[local_w][lane] = z_idx[j];
            const Vec3<T> rv(vec[3 * e + 0], vec[3 * e + 1], vec[3 * e + 2]);
            const T r = t_max(t_sqrt(dot3(rv, rv)), T(1e-8));
            const T inv_r = T(1) / r;
            const Vec3<T> u = rv * inv_r;

            Vec3<T> w(dF[3 * j + 0] - dFi.x, dF[3 * j + 1] - dFi.y, dF[3 * j + 2] - dFi.z);
            if (has_dV && dV != nullptr) {
                w.x += m00 * rv.x + m01 * rv.y + m02 * rv.z;
                w.y += m01 * rv.x + m11 * rv.y + m12 * rv.z;
                w.z += m02 * rv.x + m12 * rv.y + m22 * rv.z;
            }
            const T wr = dot3(w, u);
            const Vec3<T> wt = (w - u * wr) * inv_r;

            T g[MAX_K], dg[MAX_K], Y[MAX_SH];
            Vec3<T> dY[MAX_SH];
            cheb_val_deriv<T>(r, r_max, warp, K1, g, dg);
            eval_sh_and_grad<T>(u, lmax, Y, dY);
            for (int k = 0; k < K1; ++k) {
                s_g[local_w][lane][k] = g[k];
                s_dgw[local_w][lane][k] = wr * dg[k];
            }
            for (int m = 0; m < n_sh; ++m) {
                s_Y[local_w][lane][m] = Y[m];
                s_dYwt[local_w][lane][m] = dot3(wt, dY[m]);
            }
        }
        __syncwarp();

        for (int c = 0; c < cnt; ++c) {
            const int zj = s_zj[local_w][c];
            const int base = zj * km_tot;
            T* acc_z = use_smem ? (s_acc[local_w] + base) : (dst_i + base);
            for (int km = lane; km < km_tot; km += 32) {
                const int k = km / n_sh;
                const int m = km - k * n_sh;
                acc_z[km] += s_dgw[local_w][c][k] * s_Y[local_w][c][m]
                           + s_g[local_w][c][k]   * s_dYwt[local_w][c][m];
            }
        }
        __syncwarp();
    }

    if (use_smem) {
        for (int idx = lane; idx < tot_el; idx += 32) {
            dst_i[idx] = s_acc[local_w][idx];
        }
    }
}

// =============================================================================
// LRSK Pair descriptor layout helper
// =============================================================================
struct LrskDims {
    int S, K2, c0, c1, c2, c01, c02, c12, pol_field;
    int off_w0, off_w01, off_w1s, off_w1p, off_w02, off_w12s, off_w12p, off_w2s, off_w2p, off_w2d, T_lr;
    __host__ __device__ static LrskDims make(int S_, int K2_, int c0_, int c1_, int c2_, int pol_) {
        LrskDims d;
        d.S = S_; d.K2 = K2_; d.c0 = c0_; d.c1 = c1_; d.c2 = c2_; d.pol_field = pol_;
        d.c01 = c0_ < c1_ ? c0_ : c1_;
        d.c02 = c0_ < c2_ ? c0_ : c2_;
        d.c12 = c1_ < c2_ ? c1_ : c2_;
        int o = 1;
        d.off_w0   = o; o += d.c0;
        d.off_w01  = o; o += d.c01;
        d.off_w1s  = o; o += d.c1;
        d.off_w1p  = o; o += d.c1;
        d.off_w02  = o; o += d.c02;
        d.off_w12s = o; o += d.c12;
        d.off_w12p = o; o += d.c12;
        d.off_w2s  = o; o += d.c2;
        d.off_w2p  = o; o += d.c2;
        d.off_w2d  = o; o += d.c2;
        d.T_lr = o;
        return d;
    }
};

template <typename T>
__device__ __forceinline__ T eval_w(const T* __restrict__ Wp, int t, int K2, const T* __restrict__ g) {
    const T* wt = Wp + t * K2;
    T s = T(0);
    for (int k = 0; k < K2; ++k) s += wt[k] * g[k];
    return s;
}

template <typename T>
__device__ __forceinline__ void eval_w_dw(
    const T* __restrict__ Wp, int t, int K2, const T* __restrict__ g, const T* __restrict__ dg, T& w, T& dw
) {
    const T* wt = Wp + t * K2;
    T sw = T(0), sdw = T(0);
    for (int k = 0; k < K2; ++k) {
        const T wk = wt[k];
        sw += wk * g[k];
        sdw += wk * dg[k];
    }
    w = sw;
    dw = sdw;
}

// =============================================================================
// 4. LRSK Pair Forward: (q, p, [Q], W_eff) -> (e_lr[N], Phi[N, c0], Ef[N, c1, 3])
//    1 warp per receiver atom i -> ZERO atomics!
// =============================================================================
template <typename T>
__global__ void lrsk_pair_fwd_kernel(
    int N, LrskDims D, T r_max2, T warp, T half_es,
    const int* __restrict__ row_ptr,
    const int* __restrict__ snd,
    const T* __restrict__ vec,
    const int* __restrict__ z_idx,
    const T* __restrict__ q,
    const T* __restrict__ p,
    const T* __restrict__ Q,
    const T* __restrict__ W_eff, // [S*S, T_lr, K2]
    T* __restrict__ e_lr_out,    // [N]
    T* __restrict__ Phi_out,     // [N, c0] (optional if !pol_field)
    T* __restrict__ Ef_out       // [N, c1, 3] (optional if !pol_field)
) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (i >= N) return;

    const int zi = z_idx[i];
    const int e0 = row_ptr[i], e1 = row_ptr[i + 1];
    const T* qi = (D.c0 > 0) ? (q + (int64_t)i * D.c0) : nullptr;
    const T* pi = (D.c1 > 0) ? (p + (int64_t)i * D.c1 * 3) : nullptr;
    const T* Qi = (D.c2 > 0) ? (Q + (int64_t)i * D.c2 * 5) : nullptr;

    T e_acc = T(0);
    // Support up to c0 <= 16, c1 <= 16 cleanly in registers
    T phi_acc[16];
    Vec3<T> ef_acc[16];
    if (D.pol_field) {
        for (int c = 0; c < D.c0; ++c) phi_acc[c] = T(0);
        for (int c = 0; c < D.c1; ++c) ef_acc[c] = Vec3<T>(0, 0, 0);
    }

    T g2[MAX_K];
    for (int e = e0 + lane; e < e1; e += 32) {
        const int j = snd[e];
        const int pair = zi * D.S + z_idx[j];
        const T* Wp = W_eff + (int64_t)pair * D.T_lr * D.K2;

        const Vec3<T> rv(vec[3 * e + 0], vec[3 * e + 1], vec[3 * e + 2]);
        const T r = t_max(t_sqrt(dot3(rv, rv)), T(1e-8));
        const Vec3<T> u = rv * (T(1) / r);
        cheb_val_deriv<T>(r, r_max2, warp, D.K2, g2, nullptr);

        T eij = eval_w(Wp, 0, D.K2, g2);
        const T* qj = (D.c0 > 0) ? (q + (int64_t)j * D.c0) : nullptr;
        const T* pj = (D.c1 > 0) ? (p + (int64_t)j * D.c1 * 3) : nullptr;
        const T* Qj = (D.c2 > 0) ? (Q + (int64_t)j * D.c2 * 5) : nullptr;

        for (int c = 0; c < D.c0; ++c) {
            const T w0 = eval_w(Wp, D.off_w0 + c, D.K2, g2);
            eij += w0 * qi[c] * qj[c];
            if (D.pol_field) phi_acc[c] += w0 * qj[c];
        }
        for (int c = 0; c < D.c01; ++c) {
            const T w01 = eval_w(Wp, D.off_w01 + c, D.K2, g2);
            const Vec3<T> pjc(pj[3 * c + 0], pj[3 * c + 1], pj[3 * c + 2]);
            const T pju = dot3(pjc, u);
            eij += w01 * qi[c] * pju;
            if (D.pol_field) {
                phi_acc[c] += w01 * pju;
                ef_acc[c] += u * (w01 * qj[c]);
            }
        }
        for (int c = 0; c < D.c1; ++c) {
            const T w1s = eval_w(Wp, D.off_w1s + c, D.K2, g2);
            const T w1p = eval_w(Wp, D.off_w1p + c, D.K2, g2);
            const Vec3<T> pic(pi[3 * c + 0], pi[3 * c + 1], pi[3 * c + 2]);
            const Vec3<T> pjc(pj[3 * c + 0], pj[3 * c + 1], pj[3 * c + 2]);
            const T piu = dot3(pic, u);
            const T pju = dot3(pjc, u);
            eij += w1s * piu * pju + w1p * dot3(pic, pjc);
            if (D.pol_field) {
                ef_acc[c] += u * (w1s * pju) + pjc * w1p;
            }
        }
        if (D.c2 > 0) {
            Vec3<T> Mui[16], Muj[16];
            T s2i[16], s2j[16];
            for (int c = 0; c < D.c2; ++c) {
                Mui[c] = l2_matvec_only(Qi + 5 * c, u);
                s2i[c] = dot3(Mui[c], u);
                Muj[c] = l2_matvec_only(Qj + 5 * c, u);
                s2j[c] = dot3(Muj[c], u);
            }
            for (int c = 0; c < D.c02; ++c) {
                eij += eval_w(Wp, D.off_w02 + c, D.K2, g2) * qi[c] * s2j[c];
            }
            for (int c = 0; c < D.c12; ++c) {
                const T w12s = eval_w(Wp, D.off_w12s + c, D.K2, g2);
                const T w12p = eval_w(Wp, D.off_w12p + c, D.K2, g2);
                const Vec3<T> pic(pi[3 * c + 0], pi[3 * c + 1], pi[3 * c + 2]);
                eij += w12s * dot3(pic, u) * s2j[c] + w12p * dot3(pic, Muj[c]);
            }
            for (int c = 0; c < D.c2; ++c) {
                const T w2s = eval_w(Wp, D.off_w2s + c, D.K2, g2);
                const T w2p = eval_w(Wp, D.off_w2p + c, D.K2, g2);
                const T w2d = eval_w(Wp, D.off_w2d + c, D.K2, g2);
                T qdot = T(0);
                for (int m = 0; m < 5; ++m) qdot += Qi[5 * c + m] * Qj[5 * c + m];
                eij += w2s * s2i[c] * s2j[c] + w2p * dot3(Mui[c], Muj[c]) + w2d * qdot;
            }
        }
        e_acc += eij;
    }

    e_acc = warp_sum(e_acc);
    if (lane == 0) e_lr_out[i] = half_es * e_acc;
    if (D.pol_field) {
        for (int c = 0; c < D.c0; ++c) {
            const T v = warp_sum(phi_acc[c]);
            if (lane == 0) Phi_out[(int64_t)i * D.c0 + c] = v;
        }
        for (int c = 0; c < D.c1; ++c) {
            const T vx = warp_sum(ef_acc[c].x);
            const T vy = warp_sum(ef_acc[c].y);
            const T vz = warp_sum(ef_acc[c].z);
            if (lane == 0) {
                T* dst = Ef_out + ((int64_t)i * D.c1 + c) * 3;
                dst[0] = vx; dst[1] = vy; dst[2] = vz;
            }
        }
    }
}

template <typename T>
__device__ __forceinline__ void add_gw_rot(T* __restrict__ dst, int K2, int lane, const T* __restrict__ g, T val) {
    for (int kk = 0; kk < K2; ++kk) {
        int k = kk + lane;
        k -= (k / K2) * K2;
        atomicAdd(dst + k, g[k] * val);
    }
}

template <typename T>
__device__ __forceinline__ void add_gw2_rot(
    T* __restrict__ dst, int K2, int lane, const T* __restrict__ dgw, const T* __restrict__ g2, T M, T Mp
) {
    for (int kk = 0; kk < K2; ++kk) {
        int k = kk + lane;
        k -= (k / K2) * K2;
        atomicAdd(dst + k, dgw[k] * M + g2[k] * Mp);
    }
}

// =============================================================================
// 5. LRSK Pair 1st-Order Backward (serves BOTH Force+Virial+dq/dp/dQ AND FwdFn.backward g_Weff!)
// =============================================================================
template <typename T>
__global__ void lrsk_pair_bwd1_kernel(
    int N, LrskDims D, T r_max2, T warp, T half_es,
    int compute_force, int compute_virial, int compute_wgrad,
    const int* __restrict__ row_ptr,
    const int* __restrict__ snd,
    const T* __restrict__ vec,
    const int* __restrict__ z_idx,
    const int* __restrict__ atom_batch,
    const T* __restrict__ q,
    const T* __restrict__ p,
    const T* __restrict__ Q,
    const T* __restrict__ c_elr,   // [N]
    const T* __restrict__ c_Phi,   // [N, c0] or nullptr
    const T* __restrict__ c_E,     // [N, c1, 3] or nullptr
    const T* __restrict__ W_eff,   // [S*S, T_lr, K2]
    T* __restrict__ dq_out,        // [N, c0] zero-initialized
    T* __restrict__ dp_out,        // [N, c1, 3] zero-initialized
    T* __restrict__ dQ_out,        // [N, c2, 5] zero-initialized
    T* __restrict__ F_out,         // [N, 3] zero-initialized (optional)
    T* __restrict__ V_out,         // [G, 9] zero-initialized (optional)
    T* __restrict__ g_Weff_out     // [S*S, T_lr, K2] zero-initialized (optional)
) {
    extern __shared__ unsigned char smem_raw[];
    const int wpb = blockDim.x >> 5;
    const int local_w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * wpb + local_w;
    if (i >= N) return;

    const int w_per_zi = D.S * D.T_lr * D.K2;
    const bool use_warp_gw = (compute_wgrad != 0) && ((size_t)wpb * (size_t)w_per_zi * sizeof(T) <= 32768u);
    T* s_gW_warp = use_warp_gw ? (reinterpret_cast<T*>(smem_raw) + (size_t)local_w * w_per_zi) : nullptr;
    if (use_warp_gw) {
        for (int idx = lane; idx < w_per_zi; idx += 32) s_gW_warp[idx] = T(0);
        __syncwarp();
    }

    const int zi = z_idx[i];
    const int e0 = row_ptr[i], e1 = row_ptr[i + 1];
    const T ai = half_es * c_elr[i];
    const T* qi = (D.c0 > 0) ? (q + (int64_t)i * D.c0) : nullptr;
    const T* pi = (D.c1 > 0) ? (p + (int64_t)i * D.c1 * 3) : nullptr;
    const T* Qi = (D.c2 > 0) ? (Q + (int64_t)i * D.c2 * 5) : nullptr;

    T qti[16], dqi_acc[16];
    Vec3<T> pti[16], cEi[16], dpi_acc[16];
    T dQi_acc[16][5];
    for (int c = 0; c < D.c0; ++c) {
        const T cp = (D.pol_field && c_Phi) ? c_Phi[(int64_t)i * D.c0 + c] : T(0);
        qti[c] = ai * qi[c] + cp;
        dqi_acc[c] = T(0);
    }
    for (int c = 0; c < D.c1; ++c) {
        const Vec3<T> pic(pi[3 * c + 0], pi[3 * c + 1], pi[3 * c + 2]);
        if (D.pol_field && c_E) {
            const T* ce = c_E + ((int64_t)i * D.c1 + c) * 3;
            cEi[c] = Vec3<T>(ce[0], ce[1], ce[2]);
        } else {
            cEi[c] = Vec3<T>(0, 0, 0);
        }
        pti[c] = pic * ai + cEi[c];
        dpi_acc[c] = Vec3<T>(0, 0, 0);
    }
    for (int c = 0; c < D.c2; ++c) {
        for (int m = 0; m < 5; ++m) dQi_acc[c][m] = T(0);
    }

    Vec3<T> fi_acc(0, 0, 0);
    T v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;
    T g2[MAX_K], dg2[MAX_K];

    for (int e = e0 + lane; e < e1; e += 32) {
        const int j = snd[e];
        const int zj = z_idx[j];
        const int pair = zi * D.S + zj;
        const int p_off = pair * D.T_lr * D.K2;
        const T* Wp = W_eff + p_off;
        T* gWp = compute_wgrad ? (use_warp_gw ? (s_gW_warp + zj * D.T_lr * D.K2) : (g_Weff_out + p_off)) : nullptr;

        const Vec3<T> rv(vec[3 * e + 0], vec[3 * e + 1], vec[3 * e + 2]);
        const T r = t_max(t_sqrt(dot3(rv, rv)), T(1e-8));
        const T inv_r = T(1) / r;
        const Vec3<T> u = rv * inv_r;
        cheb_val_deriv<T>(r, r_max2, warp, D.K2, g2, compute_force ? dg2 : nullptr);

        const T* qj = (D.c0 > 0) ? (q + (int64_t)j * D.c0) : nullptr;
        const T* pj = (D.c1 > 0) ? (p + (int64_t)j * D.c1 * 3) : nullptr;
        const T* Qj = (D.c2 > 0) ? (Q + (int64_t)j * D.c2 * 5) : nullptr;

        T grad_r = T(0);
        Vec3<T> grad_u(0, 0, 0);

        // Base term t=0
        if (compute_force) {
            T w_b, dw_b;
            eval_w_dw(Wp, 0, D.K2, g2, dg2, w_b, dw_b);
            grad_r += dw_b * ai;
        }
        if (compute_wgrad) add_gw_rot(gWp, D.K2, lane, g2, ai);

        // w0
        for (int c = 0; c < D.c0; ++c) {
            T w, dw = T(0);
            if (compute_force) eval_w_dw(Wp, D.off_w0 + c, D.K2, g2, dg2, w, dw);
            else w = eval_w(Wp, D.off_w0 + c, D.K2, g2);
            const T qjc = qj[c];
            const T M = qti[c] * qjc;
            if (compute_force) grad_r += dw * M;
            if (compute_wgrad) add_gw_rot(gWp + (D.off_w0 + c) * D.K2, D.K2, lane, g2, M);
            dqi_acc[c] += w * ai * qjc;
            atomicAdd(dq_out + (int64_t)j * D.c0 + c, w * qti[c]);
        }

        // w01
        for (int c = 0; c < D.c01; ++c) {
            T w, dw = T(0);
            if (compute_force) eval_w_dw(Wp, D.off_w01 + c, D.K2, g2, dg2, w, dw);
            else w = eval_w(Wp, D.off_w01 + c, D.K2, g2);
            const Vec3<T> pjc(pj[3 * c + 0], pj[3 * c + 1], pj[3 * c + 2]);
            const T pju = dot3(pjc, u);
            const T ceu = dot3(cEi[c], u);
            const T qjc = qj[c];
            const T M = qti[c] * pju + ceu * qjc;
            if (compute_force) {
                grad_r += dw * M;
                grad_u += (pjc * qti[c] + cEi[c] * qjc) * w;
            }
            if (compute_wgrad) add_gw_rot(gWp + (D.off_w01 + c) * D.K2, D.K2, lane, g2, M);
            dqi_acc[c] += w * ai * pju;
            atomicAdd(dq_out + (int64_t)j * D.c0 + c, w * ceu);
            atomicAddVec3(dp_out + ((int64_t)j * D.c1 + c) * 3, u * (w * qti[c]));
        }

        // w1s & w1p
        for (int c = 0; c < D.c1; ++c) {
            T w1s, dw1s = T(0), w1p, dw1p = T(0);
            if (compute_force) {
                eval_w_dw(Wp, D.off_w1s + c, D.K2, g2, dg2, w1s, dw1s);
                eval_w_dw(Wp, D.off_w1p + c, D.K2, g2, dg2, w1p, dw1p);
            } else {
                w1s = eval_w(Wp, D.off_w1s + c, D.K2, g2);
                w1p = eval_w(Wp, D.off_w1p + c, D.K2, g2);
            }
            const Vec3<T> pjc(pj[3 * c + 0], pj[3 * c + 1], pj[3 * c + 2]);
            const T ptiu = dot3(pti[c], u);
            const T pju = dot3(pjc, u);
            const T Ms = ptiu * pju;
            const T Mp = dot3(pti[c], pjc);
            if (compute_force) {
                grad_r += dw1s * Ms + dw1p * Mp;
                grad_u += (pti[c] * pju + pjc * ptiu) * w1s;
            }
            if (compute_wgrad) {
                add_gw_rot(gWp + (D.off_w1s + c) * D.K2, D.K2, lane, g2, Ms);
                add_gw_rot(gWp + (D.off_w1p + c) * D.K2, D.K2, lane, g2, Mp);
            }
            dpi_acc[c] += u * (w1s * ai * pju) + pjc * (w1p * ai);
            atomicAddVec3(dp_out + ((int64_t)j * D.c1 + c) * 3, u * (w1s * ptiu) + pti[c] * w1p);
        }

        // c2 > 0 terms
        if (D.c2 > 0) {
            Vec3<T> Mui[16], Muj[16];
            T s2i[16], s2j[16];
            for (int c = 0; c < D.c2; ++c) {
                Mui[c] = l2_matvec_only(Qi + 5 * c, u);
                s2i[c] = dot3(Mui[c], u);
                Muj[c] = l2_matvec_only(Qj + 5 * c, u);
                s2j[c] = dot3(Muj[c], u);
            }
            for (int c = 0; c < D.c02; ++c) {
                T w, dw = T(0);
                if (compute_force) eval_w_dw(Wp, D.off_w02 + c, D.K2, g2, dg2, w, dw);
                else w = eval_w(Wp, D.off_w02 + c, D.K2, g2);
                const T qic = qi[c];
                const T M = ai * qic * s2j[c];
                if (compute_force) {
                    grad_r += dw * M;
                    grad_u += Muj[c] * (T(2) * w * ai * qic);
                }
                if (compute_wgrad) add_gw_rot(gWp + (D.off_w02 + c) * D.K2, D.K2, lane, g2, M);
                dqi_acc[c] += w * ai * s2j[c];
                l2_add_outer_atomic(dQ_out + ((int64_t)j * D.c2 + c) * 5, u, u, w * ai * qic);
            }
            for (int c = 0; c < D.c12; ++c) {
                T ws, dws = T(0), wp, dwp = T(0);
                if (compute_force) {
                    eval_w_dw(Wp, D.off_w12s + c, D.K2, g2, dg2, ws, dws);
                    eval_w_dw(Wp, D.off_w12p + c, D.K2, g2, dg2, wp, dwp);
                } else {
                    ws = eval_w(Wp, D.off_w12s + c, D.K2, g2);
                    wp = eval_w(Wp, D.off_w12p + c, D.K2, g2);
                }
                const Vec3<T> pic(pi[3 * c + 0], pi[3 * c + 1], pi[3 * c + 2]);
                const T piu = dot3(pic, u);
                const T Ms = ai * piu * s2j[c];
                const T Mp = ai * dot3(pic, Muj[c]);
                if (compute_force) {
                    grad_r += dws * Ms + dwp * Mp;
                    grad_u += (pic * s2j[c] + Muj[c] * (T(2) * piu)) * (ws * ai);
                    grad_u += l2_matvec_only(Qj + 5 * c, pic) * (wp * ai);
                }
                if (compute_wgrad) {
                    add_gw_rot(gWp + (D.off_w12s + c) * D.K2, D.K2, lane, g2, Ms);
                    add_gw_rot(gWp + (D.off_w12p + c) * D.K2, D.K2, lane, g2, Mp);
                }
                dpi_acc[c] += u * (ws * ai * s2j[c]) + Muj[c] * (wp * ai);
                l2_add_outer_atomic(dQ_out + ((int64_t)j * D.c2 + c) * 5, u, u, ws * ai * piu);
                l2_add_outer_atomic(dQ_out + ((int64_t)j * D.c2 + c) * 5, pic, u, wp * ai);
            }
            for (int c = 0; c < D.c2; ++c) {
                T ws, dws = T(0), wp, dwp = T(0), wd, dwd = T(0);
                if (compute_force) {
                    eval_w_dw(Wp, D.off_w2s + c, D.K2, g2, dg2, ws, dws);
                    eval_w_dw(Wp, D.off_w2p + c, D.K2, g2, dg2, wp, dwp);
                    eval_w_dw(Wp, D.off_w2d + c, D.K2, g2, dg2, wd, dwd);
                } else {
                    ws = eval_w(Wp, D.off_w2s + c, D.K2, g2);
                    wp = eval_w(Wp, D.off_w2p + c, D.K2, g2);
                    wd = eval_w(Wp, D.off_w2d + c, D.K2, g2);
                }
                const T* Qic = Qi + 5 * c;
                const T* Qjc = Qj + 5 * c;
                T qdot = T(0);
                for (int m = 0; m < 5; ++m) qdot += Qic[m] * Qjc[m];
                const T Ms = ai * s2i[c] * s2j[c];
                const T Mp = ai * dot3(Mui[c], Muj[c]);
                const T Md = ai * qdot;
                if (compute_force) {
                    grad_r += dws * Ms + dwp * Mp + dwd * Md;
                    grad_u += (Mui[c] * s2j[c] + Muj[c] * s2i[c]) * (T(2) * ws * ai);
                    grad_u += (l2_matvec_only(Qic, Muj[c]) + l2_matvec_only(Qjc, Mui[c])) * (wp * ai);
                }
                if (compute_wgrad) {
                    add_gw_rot(gWp + (D.off_w2s + c) * D.K2, D.K2, lane, g2, Ms);
                    add_gw_rot(gWp + (D.off_w2p + c) * D.K2, D.K2, lane, g2, Mp);
                    add_gw_rot(gWp + (D.off_w2d + c) * D.K2, D.K2, lane, g2, Md);
                }
                l2_add_outer_reg(dQi_acc[c], u, u, ws * ai * s2j[c]);
                l2_add_outer_reg(dQi_acc[c], Muj[c], u, wp * ai);
                for (int m = 0; m < 5; ++m) dQi_acc[c][m] += wd * ai * Qjc[m];

                T* dQj = dQ_out + ((int64_t)j * D.c2 + c) * 5;
                l2_add_outer_atomic(dQj, u, u, ws * ai * s2i[c]);
                l2_add_outer_atomic(dQj, Mui[c], u, wp * ai);
                for (int m = 0; m < 5; ++m) atomicAdd(dQj + m, wd * ai * Qic[m]);
            }
        }

        if (compute_force) {
            const T du = dot3(grad_u, u);
            const Vec3<T> gv(
                grad_r * u.x + (grad_u.x - du * u.x) * inv_r,
                grad_r * u.y + (grad_u.y - du * u.y) * inv_r,
                grad_r * u.z + (grad_u.z - du * u.z) * inv_r
            );
            fi_acc.x -= gv.x; fi_acc.y -= gv.y; fi_acc.z -= gv.z;
            atomicAddVec3(F_out + (int64_t)j * 3, gv);
            if (compute_virial) {
                v_xx += rv.x * gv.x; v_yy += rv.y * gv.y; v_zz += rv.z * gv.z;
                v_xy += T(0.5) * (rv.x * gv.y + rv.y * gv.x);
                v_xz += T(0.5) * (rv.x * gv.z + rv.z * gv.x);
                v_yz += T(0.5) * (rv.y * gv.z + rv.z * gv.y);
            }
        }
    }

    // Reduce receiver accumulators across warp and flush
    for (int c = 0; c < D.c0; ++c) {
        const T v = warp_sum(dqi_acc[c]);
        if (lane == 0) atomicAdd(dq_out + (int64_t)i * D.c0 + c, v);
    }
    for (int c = 0; c < D.c1; ++c) {
        const T vx = warp_sum(dpi_acc[c].x);
        const T vy = warp_sum(dpi_acc[c].y);
        const T vz = warp_sum(dpi_acc[c].z);
        if (lane == 0) atomicAddVec3(dp_out + ((int64_t)i * D.c1 + c) * 3, Vec3<T>(vx, vy, vz));
    }
    for (int c = 0; c < D.c2; ++c) {
        for (int m = 0; m < 5; ++m) {
            const T v = warp_sum(dQi_acc[c][m]);
            if (lane == 0) atomicAdd(dQ_out + ((int64_t)i * D.c2 + c) * 5 + m, v);
        }
    }
    if (compute_force) {
        fi_acc.x = warp_sum(fi_acc.x);
        fi_acc.y = warp_sum(fi_acc.y);
        fi_acc.z = warp_sum(fi_acc.z);
        if (lane == 0) atomicAddVec3(F_out + (int64_t)i * 3, fi_acc);
        if (compute_virial) {
            v_xx = warp_sum(v_xx); v_yy = warp_sum(v_yy); v_zz = warp_sum(v_zz);
            v_xy = warp_sum(v_xy); v_xz = warp_sum(v_xz); v_yz = warp_sum(v_yz);
            if (lane == 0) {
                T* vg = V_out + (int64_t)atom_batch[i] * 9;
                atomicAdd(vg + 0, v_xx); atomicAdd(vg + 1, v_xy); atomicAdd(vg + 2, v_xz);
                atomicAdd(vg + 3, v_xy); atomicAdd(vg + 4, v_yy); atomicAdd(vg + 5, v_yz);
                atomicAdd(vg + 6, v_xz); atomicAdd(vg + 7, v_yz); atomicAdd(vg + 8, v_zz);
            }
        }
    }

    if (use_warp_gw) {
        __syncwarp();
        T* g_dst = g_Weff_out + (int64_t)zi * w_per_zi;
        for (int idx = lane; idx < w_per_zi; idx += 32) {
            const T v = s_gW_warp[idx];
            if (v != T(0)) atomicAdd(g_dst + idx, v);
        }
    }
}

// =============================================================================
// 6. LRSK Pair 2nd-Order Backward (exact adjoint of lrsk_pair_bwd1!)
//    Warp-private SMEM + lane-rotated k accumulation -> eliminates 128-way s_gW serialization!
// =============================================================================
template <typename T>
__global__ void lrsk_pair_bwd2_kernel(
    int N, LrskDims D, T r_max2, T warp, T half_es,
    int has_dF, int has_dV,
    const int* __restrict__ row_ptr,
    const int* __restrict__ snd,
    const T* __restrict__ vec,
    const int* __restrict__ z_idx,
    const int* __restrict__ atom_batch,
    const T* __restrict__ q,
    const T* __restrict__ p,
    const T* __restrict__ Q,
    const T* __restrict__ c_elr,
    const T* __restrict__ c_Phi,
    const T* __restrict__ c_E,
    const T* __restrict__ W_eff,
    const T* __restrict__ ddq,    // [N, c0] or nullptr
    const T* __restrict__ ddp,    // [N, c1, 3] or nullptr
    const T* __restrict__ ddQ,    // [N, c2, 5] or nullptr
    const T* __restrict__ dF,     // [N, 3] or nullptr
    const T* __restrict__ dV,     // [G, 9] or nullptr
    T* __restrict__ g_q_out,      // [N, c0] zero-initialized
    T* __restrict__ g_p_out,      // [N, c1, 3] zero-initialized
    T* __restrict__ g_Q_out,      // [N, c2, 5] zero-initialized
    T* __restrict__ g_celr_out,   // [N]
    T* __restrict__ g_cPhi_out,   // [N, c0] (optional if !pol_field)
    T* __restrict__ g_cE_out,     // [N, c1, 3] (optional if !pol_field)
    T* __restrict__ g_Weff_out    // [S*S, T_lr, K2] zero-initialized
) {
    extern __shared__ unsigned char smem_raw[];
    const int wpb = blockDim.x >> 5;
    const int local_w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * wpb + local_w;
    if (i >= N) return;

    const int w_per_zi = D.S * D.T_lr * D.K2;
    const bool use_warp_gw = ((size_t)wpb * (size_t)w_per_zi * sizeof(T) <= 32768u);
    T* s_gW_warp = use_warp_gw ? (reinterpret_cast<T*>(smem_raw) + (size_t)local_w * w_per_zi) : nullptr;
    if (use_warp_gw) {
        for (int idx = lane; idx < w_per_zi; idx += 32) s_gW_warp[idx] = T(0);
        __syncwarp();
    }

    const int zi = z_idx[i];
    const int e0 = row_ptr[i], e1 = row_ptr[i + 1];
    const T ai = half_es * c_elr[i];

    const T* qi = (D.c0 > 0) ? (q + (int64_t)i * D.c0) : nullptr;
    const T* pi = (D.c1 > 0) ? (p + (int64_t)i * D.c1 * 3) : nullptr;
    const T* Qi = (D.c2 > 0) ? (Q + (int64_t)i * D.c2 * 5) : nullptr;
    const T* ddqi = (D.c0 > 0 && ddq) ? (ddq + (int64_t)i * D.c0) : nullptr;
    const T* ddpi = (D.c1 > 0 && ddp) ? (ddp + (int64_t)i * D.c1 * 3) : nullptr;
    const T* ddQi = (D.c2 > 0 && ddQ) ? (ddQ + (int64_t)i * D.c2 * 5) : nullptr;

    const Vec3<T> dFi = (has_dF && dF) ? Vec3<T>(dF[3 * i + 0], dF[3 * i + 1], dF[3 * i + 2]) : Vec3<T>(0, 0, 0);
    T m00 = 0, m01 = 0, m02 = 0, m11 = 0, m12 = 0, m22 = 0;
    if (has_dV && dV) {
        const T* dvg = dV + (int64_t)atom_batch[i] * 9;
        m00 = dvg[0]; m11 = dvg[4]; m22 = dvg[8];
        m01 = T(0.5) * (dvg[1] + dvg[3]);
        m02 = T(0.5) * (dvg[2] + dvg[6]);
        m12 = T(0.5) * (dvg[5] + dvg[7]);
    }

    T qti[16], qtip[16], g_qi_acc[16], g_cPhi_acc[16];
    Vec3<T> pti[16], ptip[16], cEi[16], g_pi_acc[16], g_cE_acc[16];
    T g_Qi_acc[16][5];
    T g_ai_acc = T(0);

    for (int c = 0; c < D.c0; ++c) {
        const T cp = (D.pol_field && c_Phi) ? c_Phi[(int64_t)i * D.c0 + c] : T(0);
        const T dqic = ddqi ? ddqi[c] : T(0);
        qti[c] = ai * qi[c] + cp;
        qtip[c] = ai * dqic;
        g_qi_acc[c] = T(0);
        g_cPhi_acc[c] = T(0);
    }
    for (int c = 0; c < D.c1; ++c) {
        const Vec3<T> pic(pi[3 * c + 0], pi[3 * c + 1], pi[3 * c + 2]);
        const Vec3<T> dpic = ddpi ? Vec3<T>(ddpi[3 * c + 0], ddpi[3 * c + 1], ddpi[3 * c + 2]) : Vec3<T>(0, 0, 0);
        if (D.pol_field && c_E) {
            const T* ce = c_E + ((int64_t)i * D.c1 + c) * 3;
            cEi[c] = Vec3<T>(ce[0], ce[1], ce[2]);
        } else {
            cEi[c] = Vec3<T>(0, 0, 0);
        }
        pti[c] = pic * ai + cEi[c];
        ptip[c] = dpic * ai;
        g_pi_acc[c] = Vec3<T>(0, 0, 0);
        g_cE_acc[c] = Vec3<T>(0, 0, 0);
    }
    for (int c = 0; c < D.c2; ++c) {
        for (int m = 0; m < 5; ++m) g_Qi_acc[c][m] = T(0);
    }

    T g2[MAX_K], dg2[MAX_K], dgw[MAX_K];
    for (int e = e0 + lane; e < e1; e += 32) {
        const int j = snd[e];
        const int zj = z_idx[j];
        const int pair = zi * D.S + zj;
        const int p_off = pair * D.T_lr * D.K2;
        const T* Wp = W_eff + p_off;
        T* gWp = use_warp_gw ? (s_gW_warp + zj * D.T_lr * D.K2) : (g_Weff_out + p_off);

        const Vec3<T> rv(vec[3 * e + 0], vec[3 * e + 1], vec[3 * e + 2]);
        const T r = t_max(t_sqrt(dot3(rv, rv)), T(1e-8));
        const T inv_r = T(1) / r;
        const Vec3<T> u = rv * inv_r;

        Vec3<T> w(0, 0, 0);
        if (has_dF && dF) {
            w.x += dF[3 * j + 0] - dFi.x;
            w.y += dF[3 * j + 1] - dFi.y;
            w.z += dF[3 * j + 2] - dFi.z;
        }
        if (has_dV && dV) {
            w.x += m00 * rv.x + m01 * rv.y + m02 * rv.z;
            w.y += m01 * rv.x + m11 * rv.y + m12 * rv.z;
            w.z += m02 * rv.x + m12 * rv.y + m22 * rv.z;
        }
        const T wr = dot3(w, u);
        const Vec3<T> wt = (w - u * wr) * inv_r;

        cheb_val_deriv<T>(r, r_max2, warp, D.K2, g2, dg2);
        for (int k = 0; k < D.K2; ++k) dgw[k] = wr * dg2[k];

        const T* qj = (D.c0 > 0) ? (q + (int64_t)j * D.c0) : nullptr;
        const T* pj = (D.c1 > 0) ? (p + (int64_t)j * D.c1 * 3) : nullptr;
        const T* Qj = (D.c2 > 0) ? (Q + (int64_t)j * D.c2 * 5) : nullptr;
        const T* ddqj = (D.c0 > 0 && ddq) ? (ddq + (int64_t)j * D.c0) : nullptr;
        const T* ddpj = (D.c1 > 0 && ddp) ? (ddp + (int64_t)j * D.c1 * 3) : nullptr;
        const T* ddQj = (D.c2 > 0 && ddQ) ? (ddQ + (int64_t)j * D.c2 * 5) : nullptr;

        // 1. Base term t=0: M=ai, Mp=0
        {
            T Wv, Wpv;
            eval_w_dw(Wp, 0, D.K2, g2, dgw, Wv, Wpv);
            g_ai_acc += Wpv;
            add_gw_rot(gWp, D.K2, lane, dgw, ai);
        }

        // 2. w0
        for (int c = 0; c < D.c0; ++c) {
            T Wv, Wpv;
            eval_w_dw(Wp, D.off_w0 + c, D.K2, g2, dgw, Wv, Wpv);
            const T qic = qi[c], qjc = qj[c];
            const T dqic = ddqi ? ddqi[c] : T(0);
            const T dqjc = ddqj ? ddqj[c] : T(0);
            const T M  = qti[c] * qjc;
            const T Mp = qtip[c] * qjc + qti[c] * dqjc;
            g_ai_acc += Wpv * qic * qjc + Wv * (dqic * qjc + qic * dqjc);
            const T c_i = Wpv * qjc + Wv * dqjc;
            g_cPhi_acc[c] += c_i;
            g_qi_acc[c] += ai * c_i;
            atomicAdd(g_q_out + (int64_t)j * D.c0 + c, Wpv * qti[c] + Wv * qtip[c]);
            add_gw2_rot(gWp + (D.off_w0 + c) * D.K2, D.K2, lane, dgw, g2, M, Mp);
        }

        // 3. w01
        for (int c = 0; c < D.c01; ++c) {
            T Wv, Wpv;
            eval_w_dw(Wp, D.off_w01 + c, D.K2, g2, dgw, Wv, Wpv);
            const T qic = qi[c], qjc = qj[c];
            const T dqic = ddqi ? ddqi[c] : T(0);
            const T dqjc = ddqj ? ddqj[c] : T(0);
            const Vec3<T> pjc(pj[3 * c + 0], pj[3 * c + 1], pj[3 * c + 2]);
            const Vec3<T> dpjc = ddpj ? Vec3<T>(ddpj[3 * c + 0], ddpj[3 * c + 1], ddpj[3 * c + 2]) : Vec3<T>(0, 0, 0);
            const T sj = dot3(pjc, u);
            const T sjp = dot3(dpjc, u) + dot3(pjc, wt);
            const T ei = dot3(cEi[c], u);
            const T eip = dot3(cEi[c], wt);
            const T M  = qti[c] * sj + ei * qjc;
            const T Mp = qtip[c] * sj + qti[c] * sjp + eip * qjc + ei * dqjc;

            g_ai_acc += Wpv * qic * sj + Wv * (dqic * sj + qic * sjp);
            const T c_i = Wpv * sj + Wv * sjp;
            g_cPhi_acc[c] += c_i;
            g_qi_acc[c] += ai * c_i;
            g_cE_acc[c] += u * (Wpv * qjc + Wv * dqjc) + wt * (Wv * qjc);
            atomicAdd(g_q_out + (int64_t)j * D.c0 + c, Wpv * ei + Wv * eip);
            atomicAddVec3(g_p_out + ((int64_t)j * D.c1 + c) * 3,
                          u * (Wpv * qti[c] + Wv * qtip[c]) + wt * (Wv * qti[c]));
            add_gw2_rot(gWp + (D.off_w01 + c) * D.K2, D.K2, lane, dgw, g2, M, Mp);
        }

        // 4. w1s & 5. w1p
        for (int c = 0; c < D.c1; ++c) {
            T Ws, Wps, Wp_, Wpp;
            eval_w_dw(Wp, D.off_w1s + c, D.K2, g2, dgw, Ws, Wps);
            eval_w_dw(Wp, D.off_w1p + c, D.K2, g2, dgw, Wp_, Wpp);

            const Vec3<T> pic(pi[3 * c + 0], pi[3 * c + 1], pi[3 * c + 2]);
            const Vec3<T> pjc(pj[3 * c + 0], pj[3 * c + 1], pj[3 * c + 2]);
            const Vec3<T> dpic = ddpi ? Vec3<T>(ddpi[3 * c + 0], ddpi[3 * c + 1], ddpi[3 * c + 2]) : Vec3<T>(0, 0, 0);
            const Vec3<T> dpjc = ddpj ? Vec3<T>(ddpj[3 * c + 0], ddpj[3 * c + 1], ddpj[3 * c + 2]) : Vec3<T>(0, 0, 0);

            // w1s
            const T piu = dot3(pic, u);
            const T piup = dot3(dpic, u) + dot3(pic, wt);
            const T ti = dot3(pti[c], u);
            const T tip = dot3(ptip[c], u) + dot3(pti[c], wt);
            const T sj = dot3(pjc, u);
            const T sjp = dot3(dpjc, u) + dot3(pjc, wt);
            const T Ms = ti * sj;
            const T Msp = tip * sj + ti * sjp;

            const T Ci = Wps * sj + Ws * sjp;
            const T Di = Ws * sj;
            g_ai_acc += Ci * piu + Di * piup;
            const Vec3<T> gi_vec = u * Ci + wt * Di;
            g_cE_acc[c] += gi_vec;
            g_pi_acc[c] += gi_vec * ai;
            const Vec3<T> gj_s = u * (Wps * ti + Ws * tip) + wt * (Ws * ti);

            // w1p
            const T Mp = dot3(pti[c], pjc);
            const T Mpp = dot3(ptip[c], pjc) + dot3(pti[c], dpjc);
            g_ai_acc += Wpp * dot3(pic, pjc) + Wp_ * (dot3(dpic, pjc) + dot3(pic, dpjc));
            const Vec3<T> gp_vec = pjc * Wpp + dpjc * Wp_;
            g_cE_acc[c] += gp_vec;
            g_pi_acc[c] += gp_vec * ai;
            const Vec3<T> gj_p = pti[c] * Wpp + ptip[c] * Wp_;

            atomicAddVec3(g_p_out + ((int64_t)j * D.c1 + c) * 3, gj_s + gj_p);

            add_gw2_rot(gWp + (D.off_w1s + c) * D.K2, D.K2, lane, dgw, g2, Ms, Msp);
            add_gw2_rot(gWp + (D.off_w1p + c) * D.K2, D.K2, lane, dgw, g2, Mp, Mpp);
        }

        // 6. c2 > 0 terms
        if (D.c2 > 0) {
            Vec3<T> Mui[16], Muip[16], Muj[16], Mujp[16];
            T s2i[16], s2ip[16], s2j[16], s2jp[16];
            for (int c = 0; c < D.c2; ++c) {
                const T* Qic = Qi + 5 * c;
                const T* Qjc = Qj + 5 * c;
                Mui[c] = l2_matvec_only(Qic, u);
                s2i[c] = dot3(Mui[c], u);
                Vec3<T> dMui = ddQi ? l2_matvec_only(ddQi + 5 * c, u) : Vec3<T>(0, 0, 0);
                Muip[c] = dMui + l2_matvec_only(Qic, wt);
                s2ip[c] = dot3(dMui, u) + T(2) * dot3(Mui[c], wt);

                Muj[c] = l2_matvec_only(Qjc, u);
                s2j[c] = dot3(Muj[c], u);
                Vec3<T> dMuj = ddQj ? l2_matvec_only(ddQj + 5 * c, u) : Vec3<T>(0, 0, 0);
                Mujp[c] = dMuj + l2_matvec_only(Qjc, wt);
                s2jp[c] = dot3(dMuj, u) + T(2) * dot3(Muj[c], wt);
            }
            for (int c = 0; c < D.c02; ++c) {
                T Wv, Wpv;
                eval_w_dw(Wp, D.off_w02 + c, D.K2, g2, dgw, Wv, Wpv);
                const T qic = qi[c];
                const T dqic = ddqi ? ddqi[c] : T(0);
                const T M  = ai * qic * s2j[c];
                const T Mp = ai * (dqic * s2j[c] + qic * s2jp[c]);
                g_ai_acc += Wpv * qic * s2j[c] + Wv * (dqic * s2j[c] + qic * s2jp[c]);
                g_qi_acc[c] += ai * (Wpv * s2j[c] + Wv * s2jp[c]);
                T* gQj = g_Q_out + ((int64_t)j * D.c2 + c) * 5;
                l2_add_outer_atomic(gQj, u, u, ai * (Wpv * qic + Wv * dqic));
                l2_add_outer_atomic(gQj, wt, u, T(2) * ai * Wv * qic);
                add_gw2_rot(gWp + (D.off_w02 + c) * D.K2, D.K2, lane, dgw, g2, M, Mp);
            }
            for (int c = 0; c < D.c12; ++c) {
                T Ws, Wps, Wp_, Wpp;
                eval_w_dw(Wp, D.off_w12s + c, D.K2, g2, dgw, Ws, Wps);
                eval_w_dw(Wp, D.off_w12p + c, D.K2, g2, dgw, Wp_, Wpp);
                const Vec3<T> pic(pi[3 * c + 0], pi[3 * c + 1], pi[3 * c + 2]);
                const Vec3<T> dpic = ddpi ? Vec3<T>(ddpi[3 * c + 0], ddpi[3 * c + 1], ddpi[3 * c + 2]) : Vec3<T>(0, 0, 0);
                const T piu = dot3(pic, u);
                const T piup = dot3(dpic, u) + dot3(pic, wt);
                const T Ms = ai * piu * s2j[c];
                const T Msp = ai * (piup * s2j[c] + piu * s2jp[c]);
                const T Mp = ai * dot3(pic, Muj[c]);
                const T Mpp = ai * (dot3(dpic, Muj[c]) + dot3(pic, Mujp[c]));

                g_ai_acc += Wps * piu * s2j[c] + Ws * (piup * s2j[c] + piu * s2jp[c]);
                g_ai_acc += Wpp * dot3(pic, Muj[c]) + Wp_ * (dot3(dpic, Muj[c]) + dot3(pic, Mujp[c]));

                g_pi_acc[c] += (u * (Wps * s2j[c] + Ws * s2jp[c]) + wt * (Ws * s2j[c])) * ai;
                g_pi_acc[c] += (Muj[c] * Wpp + Mujp[c] * Wp_) * ai;

                T* gQj = g_Q_out + ((int64_t)j * D.c2 + c) * 5;
                l2_add_outer_atomic(gQj, u, u, ai * (Wps * piu + Ws * piup));
                l2_add_outer_atomic(gQj, wt, u, T(2) * ai * Ws * piu);
                l2_add_outer_atomic(gQj, (pic * Wpp + dpic * Wp_) * ai, u, T(1));
                l2_add_outer_atomic(gQj, pic * (ai * Wp_), wt, T(1));

                add_gw2_rot(gWp + (D.off_w12s + c) * D.K2, D.K2, lane, dgw, g2, Ms, Msp);
                add_gw2_rot(gWp + (D.off_w12p + c) * D.K2, D.K2, lane, dgw, g2, Mp, Mpp);
            }
            for (int c = 0; c < D.c2; ++c) {
                T Ws, Wps, Wp_, Wpp, Wd, Wpd;
                eval_w_dw(Wp, D.off_w2s + c, D.K2, g2, dgw, Ws, Wps);
                eval_w_dw(Wp, D.off_w2p + c, D.K2, g2, dgw, Wp_, Wpp);
                eval_w_dw(Wp, D.off_w2d + c, D.K2, g2, dgw, Wd, Wpd);

                const T* Qic = Qi + 5 * c;
                const T* Qjc = Qj + 5 * c;
                const T* dQic = ddQi ? (ddQi + 5 * c) : nullptr;
                const T* dQjc = ddQj ? (ddQj + 5 * c) : nullptr;

                T qdot = T(0), qdotp = T(0);
                for (int m = 0; m < 5; ++m) {
                    const T dqi_m = dQic ? dQic[m] : T(0);
                    const T dqj_m = dQjc ? dQjc[m] : T(0);
                    qdot += Qic[m] * Qjc[m];
                    qdotp += dqi_m * Qjc[m] + Qic[m] * dqj_m;
                }
                const T Ms = ai * s2i[c] * s2j[c];
                const T Msp = ai * (s2ip[c] * s2j[c] + s2i[c] * s2jp[c]);
                const T Mp = ai * dot3(Mui[c], Muj[c]);
                const T Mpp = ai * (dot3(Muip[c], Muj[c]) + dot3(Mui[c], Mujp[c]));
                const T Md = ai * qdot;
                const T Mdp = ai * qdotp;

                g_ai_acc += Wps * s2i[c] * s2j[c] + Ws * (s2ip[c] * s2j[c] + s2i[c] * s2jp[c]);
                g_ai_acc += Wpp * dot3(Mui[c], Muj[c]) + Wp_ * (dot3(Muip[c], Muj[c]) + dot3(Mui[c], Mujp[c]));
                g_ai_acc += Wpd * qdot + Wd * qdotp;

                l2_add_outer_reg(g_Qi_acc[c], u, u, ai * (Wps * s2j[c] + Ws * s2jp[c]));
                l2_add_outer_reg(g_Qi_acc[c], wt, u, T(2) * ai * Ws * s2j[c]);
                l2_add_outer_reg(g_Qi_acc[c], (Muj[c] * Wpp + Mujp[c] * Wp_) * ai, u, T(1));
                l2_add_outer_reg(g_Qi_acc[c], Muj[c] * (ai * Wp_), wt, T(1));
                for (int m = 0; m < 5; ++m) {
                    const T dqj_m = dQjc ? dQjc[m] : T(0);
                    g_Qi_acc[c][m] += ai * (Wpd * Qjc[m] + Wd * dqj_m);
                }

                T* gQj = g_Q_out + ((int64_t)j * D.c2 + c) * 5;
                l2_add_outer_atomic(gQj, u, u, ai * (Wps * s2i[c] + Ws * s2ip[c]));
                l2_add_outer_atomic(gQj, wt, u, T(2) * ai * Ws * s2i[c]);
                l2_add_outer_atomic(gQj, (Mui[c] * Wpp + Muip[c] * Wp_) * ai, u, T(1));
                l2_add_outer_atomic(gQj, Mui[c] * (ai * Wp_), wt, T(1));
                for (int m = 0; m < 5; ++m) {
                    const T dqi_m = dQic ? dQic[m] : T(0);
                    atomicAdd(gQj + m, ai * (Wpd * Qic[m] + Wd * dqi_m));
                }

                add_gw2_rot(gWp + (D.off_w2s + c) * D.K2, D.K2, lane, dgw, g2, Ms, Msp);
                add_gw2_rot(gWp + (D.off_w2p + c) * D.K2, D.K2, lane, dgw, g2, Mp, Mpp);
                add_gw2_rot(gWp + (D.off_w2d + c) * D.K2, D.K2, lane, dgw, g2, Md, Mdp);
            }
        }
    }

    // Reduce receiver accumulators across warp
    const T g_ai_tot = warp_sum(g_ai_acc);
    if (lane == 0) g_celr_out[i] = half_es * g_ai_tot;

    for (int c = 0; c < D.c0; ++c) {
        const T gq = warp_sum(g_qi_acc[c]);
        if (lane == 0) atomicAdd(g_q_out + (int64_t)i * D.c0 + c, gq);
        if (D.pol_field && g_cPhi_out) {
            const T gp = warp_sum(g_cPhi_acc[c]);
            if (lane == 0) g_cPhi_out[(int64_t)i * D.c0 + c] = gp;
        }
    }
    for (int c = 0; c < D.c1; ++c) {
        const T gx = warp_sum(g_pi_acc[c].x);
        const T gy = warp_sum(g_pi_acc[c].y);
        const T gz = warp_sum(g_pi_acc[c].z);
        if (lane == 0) atomicAddVec3(g_p_out + ((int64_t)i * D.c1 + c) * 3, Vec3<T>(gx, gy, gz));
        if (D.pol_field && g_cE_out) {
            const T ex = warp_sum(g_cE_acc[c].x);
            const T ey = warp_sum(g_cE_acc[c].y);
            const T ez = warp_sum(g_cE_acc[c].z);
            if (lane == 0) {
                T* dst = g_cE_out + ((int64_t)i * D.c1 + c) * 3;
                dst[0] = ex; dst[1] = ey; dst[2] = ez;
            }
        }
    }
    for (int c = 0; c < D.c2; ++c) {
        for (int m = 0; m < 5; ++m) {
            const T gQ = warp_sum(g_Qi_acc[c][m]);
            if (lane == 0) atomicAdd(g_Q_out + ((int64_t)i * D.c2 + c) * 5 + m, gQ);
        }
    }

    if (use_warp_gw) {
        __syncwarp();
        T* g_dst = g_Weff_out + (int64_t)zi * w_per_zi;
        for (int idx = lane; idx < w_per_zi; idx += 32) {
            const T v = s_gW_warp[idx];
            if (v != T(0)) atomicAdd(g_dst + idx, v);
        }
    }
}

} // namespace loup_lrsk_train

using namespace loup_lrsk_train;

// =============================================================================
// C API Exports (float32 + float64)
// =============================================================================
extern "C" {

#define DEFINE_C_API(SUFFIX, T) \
int l1_precompute_G1_##SUFFIX( \
    void* stream_ptr, int N, int S, int K1, int lmax, double r_max, double warp, \
    const int* row_ptr, const int* snd, const T* vec, const int* z_idx, T* G1 \
) { \
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_ptr); \
    int n_sh = (lmax + 1) * (lmax + 1); \
    cudaMemsetAsync(G1, 0, (size_t)N * S * K1 * n_sh * sizeof(T), st); \
    int thr = 128; \
    int blk = (N + thr - 1) / thr; \
    if (blk > 0) l1_precompute_G1_kernel<T><<<blk, thr, 0, st>>>(N, S, K1, lmax, (T)r_max, (T)warp, row_ptr, snd, vec, z_idx, G1); \
    return (int)cudaGetLastError(); \
} \
int l1_force_fwd_##SUFFIX( \
    void* stream_ptr, int N, int G, int S, int K1, int lmax, double r_max, double warp, int compute_virial, \
    const int* row_ptr, const int* snd, const T* vec, const int* z_idx, const int* atom_batch, \
    const T* G_bwd, T* F_out, T* V_out \
) { \
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_ptr); \
    cudaMemsetAsync(F_out, 0, (size_t)N * 3 * sizeof(T), st); \
    if (compute_virial && V_out) cudaMemsetAsync(V_out, 0, (size_t)G * 9 * sizeof(T), st); \
    int wpb = 4; \
    int blk = (N + wpb - 1) / wpb; \
    if (blk > 0) l1_force_fwd_kernel<T><<<blk, wpb * 32, 0, st>>>(N, S, K1, lmax, (T)r_max, (T)warp, compute_virial, row_ptr, snd, vec, z_idx, atom_batch, G_bwd, F_out, V_out); \
    return (int)cudaGetLastError(); \
} \
int l1_force_bwd2_##SUFFIX( \
    void* stream_ptr, int N, int S, int K1, int lmax, double r_max, double warp, int has_dV, \
    const int* row_ptr, const int* snd, const T* vec, const int* z_idx, const int* atom_batch, \
    const T* dF, const T* dV, T* dG_out \
) { \
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_ptr); \
    int wpb = 4; \
    int blk = (N + wpb - 1) / wpb; \
    if (blk > 0) l1_force_bwd2_kernel<T><<<blk, wpb * 32, 0, st>>>(N, S, K1, lmax, (T)r_max, (T)warp, has_dV, row_ptr, snd, vec, z_idx, atom_batch, dF, dV, dG_out); \
    return (int)cudaGetLastError(); \
} \
int lrsk_pair_fwd_##SUFFIX( \
    void* stream_ptr, int N, int S, int K2, int c0, int c1, int c2, int pol_field, \
    double r_max2, double warp, double energy_scale, \
    const int* row_ptr, const int* snd, const T* vec, const int* z_idx, \
    const T* q, const T* p, const T* Q, const T* W_eff, \
    T* e_lr_out, T* Phi_out, T* Ef_out \
) { \
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_ptr); \
    LrskDims D = LrskDims::make(S, K2, c0, c1, c2, pol_field); \
    int wpb = 4; \
    int blk = (N + wpb - 1) / wpb; \
    if (blk > 0) lrsk_pair_fwd_kernel<T><<<blk, wpb * 32, 0, st>>>(N, D, (T)r_max2, (T)warp, (T)(0.5 * energy_scale), row_ptr, snd, vec, z_idx, q, p, Q, W_eff, e_lr_out, Phi_out, Ef_out); \
    return (int)cudaGetLastError(); \
} \
int lrsk_pair_bwd1_##SUFFIX( \
    void* stream_ptr, int N, int G, int S, int K2, int c0, int c1, int c2, int pol_field, \
    double r_max2, double warp, double energy_scale, \
    int compute_force, int compute_virial, int compute_wgrad, \
    const int* row_ptr, const int* snd, const T* vec, const int* z_idx, const int* atom_batch, \
    const T* q, const T* p, const T* Q, const T* c_elr, const T* c_Phi, const T* c_E, const T* W_eff, \
    T* dq_out, T* dp_out, T* dQ_out, T* F_out, T* V_out, T* g_Weff_out \
) { \
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_ptr); \
    LrskDims D = LrskDims::make(S, K2, c0, c1, c2, pol_field); \
    if (c0 > 0 && dq_out) cudaMemsetAsync(dq_out, 0, (size_t)N * c0 * sizeof(T), st); \
    if (c1 > 0 && dp_out) cudaMemsetAsync(dp_out, 0, (size_t)N * c1 * 3 * sizeof(T), st); \
    if (c2 > 0 && dQ_out) cudaMemsetAsync(dQ_out, 0, (size_t)N * c2 * 5 * sizeof(T), st); \
    if (compute_force && F_out) cudaMemsetAsync(F_out, 0, (size_t)N * 3 * sizeof(T), st); \
    if (compute_force && compute_virial && V_out) cudaMemsetAsync(V_out, 0, (size_t)G * 9 * sizeof(T), st); \
    size_t w_bytes = (size_t)S * S * D.T_lr * K2 * sizeof(T); \
    if (compute_wgrad && g_Weff_out) cudaMemsetAsync(g_Weff_out, 0, w_bytes, st); \
    int wpb = 4; \
    size_t warp_bytes = (size_t)wpb * (size_t)S * D.T_lr * K2 * sizeof(T); \
    size_t smem = (compute_wgrad && warp_bytes <= 32768u) ? warp_bytes : 0u; \
    int blk = (N + wpb - 1) / wpb; \
    if (blk > 0) lrsk_pair_bwd1_kernel<T><<<blk, wpb * 32, smem, st>>>(N, D, (T)r_max2, (T)warp, (T)(0.5 * energy_scale), compute_force, compute_virial, compute_wgrad, row_ptr, snd, vec, z_idx, atom_batch, q, p, Q, c_elr, c_Phi, c_E, W_eff, dq_out, dp_out, dQ_out, F_out, V_out, g_Weff_out); \
    return (int)cudaGetLastError(); \
} \
int lrsk_pair_bwd2_##SUFFIX( \
    void* stream_ptr, int N, int S, int K2, int c0, int c1, int c2, int pol_field, \
    double r_max2, double warp, double energy_scale, int has_dF, int has_dV, \
    const int* row_ptr, const int* snd, const T* vec, const int* z_idx, const int* atom_batch, \
    const T* q, const T* p, const T* Q, const T* c_elr, const T* c_Phi, const T* c_E, const T* W_eff, \
    const T* ddq, const T* ddp, const T* ddQ, const T* dF, const T* dV, \
    T* g_q_out, T* g_p_out, T* g_Q_out, T* g_celr_out, T* g_cPhi_out, T* g_cE_out, T* g_Weff_out \
) { \
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_ptr); \
    LrskDims D = LrskDims::make(S, K2, c0, c1, c2, pol_field); \
    if (c0 > 0 && g_q_out) cudaMemsetAsync(g_q_out, 0, (size_t)N * c0 * sizeof(T), st); \
    if (c1 > 0 && g_p_out) cudaMemsetAsync(g_p_out, 0, (size_t)N * c1 * 3 * sizeof(T), st); \
    if (c2 > 0 && g_Q_out) cudaMemsetAsync(g_Q_out, 0, (size_t)N * c2 * 5 * sizeof(T), st); \
    size_t w_bytes = (size_t)S * S * D.T_lr * K2 * sizeof(T); \
    cudaMemsetAsync(g_Weff_out, 0, w_bytes, st); \
    int wpb = 4; \
    size_t warp_bytes = (size_t)wpb * (size_t)S * D.T_lr * K2 * sizeof(T); \
    size_t smem = (warp_bytes <= 32768u) ? warp_bytes : 0u; \
    int blk = (N + wpb - 1) / wpb; \
    if (blk > 0) lrsk_pair_bwd2_kernel<T><<<blk, wpb * 32, smem, st>>>(N, D, (T)r_max2, (T)warp, (T)(0.5 * energy_scale), has_dF, has_dV, row_ptr, snd, vec, z_idx, atom_batch, q, p, Q, c_elr, c_Phi, c_E, W_eff, ddq, ddp, ddQ, dF, dV, g_q_out, g_p_out, g_Q_out, g_celr_out, g_cPhi_out, g_cE_out, g_Weff_out); \
    return (int)cudaGetLastError(); \
}

DEFINE_C_API(f32, float)
DEFINE_C_API(f64, double)

} // extern "C"
