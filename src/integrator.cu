#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdint>

namespace loup {

constexpr double ASE_FS = 0.09822694788464063;
constexpr double ASE_KB = 8.617330337217213e-5;

__global__ void nve_first_half_kernel(
    int N, double dt_ase,
    const double* __restrict__ forces,
    const double* __restrict__ inv_masses,
    double* __restrict__ pos64,
    double* __restrict__ vel64
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double im = inv_masses[i];
    double half_dt_im = 0.5 * dt_ase * im;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        int idx = i * 3 + c;
        double v = vel64[idx] + half_dt_im * forces[idx];
        vel64[idx] = v;
        const double dx = dt_ase * v;
        pos64[idx] += dx;
    }
}

__global__ void nve_second_half_kernel(
    int N, double dt_ase,
    const double* __restrict__ forces,
    const double* __restrict__ inv_masses,
    double* __restrict__ vel64
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double half_dt_im = 0.5 * dt_ase * inv_masses[i];
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        int idx = i * 3 + c;
        vel64[idx] += half_dt_im * forces[idx];
    }
}

// S106: 32 atoms per 96-thread block. Thread (atom i, c) draws the c-th normal pair of atom i at Philox offset
// step * 16 + 4 c: the same 32-bit words, and so the same noise, as the three sequential curand_normal2_double calls
// of the former one-thread-per-atom kernel (each consumes 4 words from an aligned position), in a third of the
// double-precision latency chain. The same thread then updates component c of atom i (contiguous across the block).
// Offset counts 32-bit outputs; the three pairs consume 3 x 4 = 12, so 16 per step keeps consecutive steps on
// disjoint words.
__global__ void __launch_bounds__(96) langevin_first_half_kernel(
    int N, double dt_ase, double c1, double c2,
    double k_c3, double k_c4, double k_c5,
    unsigned long long seed, const unsigned long long* __restrict__ step_counter,
    const double* __restrict__ forces,
    const double* __restrict__ inv_masses,
    const double* __restrict__ sqrt_inv_masses,
    double* __restrict__ pos64,
    double* __restrict__ vel64,
    double* __restrict__ rnd_vel_save
) {
    __shared__ double s_n[32][6];  // per atom: xi[0..2], eta[0..2] = the three pairs in draw order
    const int t = threadIdx.x, a = t / 3, c = t - 3 * a;
    const int i = blockIdx.x * 32 + a;
    if (i < N) {
        curandStatePhilox4_32_10_t rng;
        curand_init(seed, (unsigned long long)i, *step_counter * 16ULL + 4ULL * c, &rng);
        const double2 n = curand_normal2_double(&rng);
        s_n[a][2 * c] = n.x;
        s_n[a][2 * c + 1] = n.y;
    }
    __syncthreads();
    if (i >= N) return;
    const double xi = s_n[a][c], eta = s_n[a][3 + c];
    const double im = inv_masses[i];
    const double sim = sqrt_inv_masses[i];
    const double c3 = k_c3 * sim;
    const double c4 = k_c4 * sim;
    const double c5 = k_c5 * sim;
    const int idx = i * 3 + c;
    const double rnd_p = c5 * eta;
    const double rnd_v = c3 * xi - c4 * eta;
    rnd_vel_save[idx] = rnd_v;
    const double v = vel64[idx] + c1 * forces[idx] * im - c2 * vel64[idx] + rnd_v;
    const double dx = dt_ase * v + rnd_p;
    pos64[idx] += dx;
    vel64[idx] = (dx - rnd_p) / dt_ase;
}

// Also advances the step counter: every first-half read of it for this step has completed.
__global__ void langevin_second_half_kernel(
    int N, double c1, double c2,
    const double* __restrict__ forces,
    const double* __restrict__ inv_masses,
    const double* __restrict__ rnd_vel_save,
    double* __restrict__ vel64,
    unsigned long long* __restrict__ step_counter
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) (*step_counter) += 1ULL;
    if (i >= N) return;
    double im = inv_masses[i];
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        int idx = i * 3 + c;
        vel64[idx] += c1 * forces[idx] * im - c2 * vel64[idx] + rnd_vel_save[idx];
    }
}

__global__ void kinetic_energy_kernel(
    int N,
    const double* __restrict__ masses,
    const double* __restrict__ vel64,
    double* __restrict__ out_ke
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double ke = 0.0;
    if (i < N) {
        double vx = vel64[i * 3 + 0];
        double vy = vel64[i * 3 + 1];
        double vz = vel64[i * 3 + 2];
        ke = 0.5 * masses[i] * (vx * vx + vy * vy + vz * vz);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ke += __shfl_xor_sync(0xffffffffu, ke, off);
    __shared__ double s_ke[8];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) s_ke[warp] = ke;
    __syncthreads();
    if (threadIdx.x == 0) {
        double sum = 0.0;
        int nw = blockDim.x >> 5;
        for (int w = 0; w < nw; ++w) sum += s_ke[w];
        atomicAdd(out_ke, sum);
    }
}

}  // namespace loup
