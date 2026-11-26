// Split-K "weight gradient" GEMMs of the training library: C (+)= alpha A^T B with A [K][M], B [K][N] row-major,
// small M x N and long K (the node rows), up to four segments (degrees) in one launch. cuBLAS runs these on one to
// three blocks; here each segment's K is cut into chunks whose partial products go to a workspace, and a second
// kernel sums them in a fixed order (deterministic) and applies alpha and beta.
#pragma once

namespace lst {

template <typename T>
struct TnSeg {
    const T* A;
    const T* B;
    T* C;
    int K;       // rows of this segment
    int kc;      // rows per chunk (multiple of 32)
    int nchunk;
    int chunk0;  // index of its first chunk
};

template <typename T>
struct TnArgs {
    TnSeg<T> s[4];
    int nseg, M, N, lda, ldb, ldc;
    T alpha, beta;
    T* ws;  // [total chunks][M][N]
};

constexpr int TN_KS = 32;

__device__ __forceinline__ void tn_ld4(const float* p, float v[4]) {
    const float4 q = *reinterpret_cast<const float4*>(p);
    v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
}
__device__ __forceinline__ void tn_ld4(const double* p, double v[4]) {
    const double2 q0 = *reinterpret_cast<const double2*>(p), q1 = *reinterpret_cast<const double2*>(p + 2);
    v[0] = q0.x; v[1] = q0.y; v[2] = q1.x; v[3] = q1.y;
}

template <typename T>
__global__ void __launch_bounds__(256) k_gemm_tn(TnArgs<T> a) {
    __shared__ __align__(16) T As[TN_KS][64];
    __shared__ __align__(16) T Bs[TN_KS][64];
    const int z = blockIdx.z;
    int si = 0;
    while (si + 1 < a.nseg && z >= a.s[si + 1].chunk0) ++si;
    const TnSeg<T> sg = a.s[si];
    const int ch = z - sg.chunk0;
    const int k0 = ch * sg.kc, k1 = min(sg.K, k0 + sg.kc);
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const int t = threadIdx.x, tx = t % 16, ty = t / 16;
    T acc[4][4] = {};
    for (int kb = k0; kb < k1; kb += TN_KS) {
        for (int i = t; i < TN_KS * 64; i += 256) {
            const int r = i / 64, c = i % 64;
            const int k = kb + r;
            const bool kin = k < k1;
            As[r][c] = (kin && m0 + c < a.M) ? sg.A[(long)k * a.lda + m0 + c] : T(0);
            Bs[r][c] = (kin && n0 + c < a.N) ? sg.B[(long)k * a.ldb + n0 + c] : T(0);
        }
        __syncthreads();
#pragma unroll 8
        for (int r = 0; r < TN_KS; ++r) {
            T av[4], bv[4];
            tn_ld4(&As[r][4 * ty], av);
            tn_ld4(&Bs[r][4 * tx], bv);
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) acc[u][v] += av[u] * bv[v];
        }
        __syncthreads();
    }
    T* w = a.ws + (long)z * a.M * a.N;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int m = m0 + 4 * ty + u;
        if (m >= a.M) continue;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int n = n0 + 4 * tx + v;
            if (n < a.N) w[(long)m * a.N + n] = acc[u][v];
        }
    }
}

template <typename T>
__global__ void k_gemm_tn_reduce(TnArgs<T> a) {
    const long MN = (long)a.M * a.N;
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MN * a.nseg) return;
    const int si = (int)(i / MN);
    const long j = i % MN;
    const TnSeg<T> sg = a.s[si];
    T sum = T(0);
    for (int c = 0; c < sg.nchunk; ++c) sum += a.ws[(long)(sg.chunk0 + c) * MN + j];
    const int m = (int)(j / a.N), n = (int)(j % a.N);
    T* cp = sg.C + (long)m * a.ldc + n;
    *cp = a.beta == T(0) ? a.alpha * sum : a.alpha * sum + a.beta * *cp;
}

}  // namespace lst
