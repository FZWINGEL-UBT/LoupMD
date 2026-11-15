#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdint>

namespace loup {

constexpr double ASE_FS = 0.09822694788464063;
constexpr double ASE_KB = 8.617330337217213e-5;

// Maximum over atoms and steps of the squared per-step displacement, as float bits (ordered like the floats
// themselves for non-negative values). check_skin_kernel lowers the rebuild threshold by it. It decays in windows:
// max_step_disp2[0] is the maximum of the current window, [1] that of the previous one, and readers take the larger,
// so one fast step is forgotten two windows later (review item S2) while the value never falls below what the
// current window has seen. rotate_step_disp_kernel ends a window (run_md_loop decides when).
__device__ __forceinline__ void record_step_disp2(double d2, unsigned int* max_step_disp2) {
    const unsigned int bits = __float_as_uint((float)d2);
    const unsigned int m = __reduce_max_sync(0xffffffffu, bits);
    if ((threadIdx.x & 31) == 0 && m != 0u) atomicMax(max_step_disp2, m);
}

// Ending a window also clears bit 4 of the skin check's flag word ("list too old to switch to the pipelined loop"):
// that bit was set against the old maximum and otherwise stays until a rebuild, which a cold system never has. The
// next unpipelined check sets it again if it still holds. Bits 1 and 2 are not touched.
__global__ void rotate_step_disp_kernel(unsigned int* __restrict__ max_step_disp2, int* __restrict__ need_rebuild) {
    max_step_disp2[1] = max_step_disp2[0];
    max_step_disp2[0] = 0u;
    need_rebuild[0] &= ~4;
}

__global__ void nve_first_half_kernel(
    int N, double dt_ase,
    const double* __restrict__ forces,
    const double* __restrict__ inv_masses,
    double* __restrict__ pos64,
    double* __restrict__ vel64,
    unsigned int* __restrict__ max_step_disp2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double d2 = 0.0;
    if (i < N) {
        double im = inv_masses[i];
        double half_dt_im = 0.5 * dt_ase * im;
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            int idx = i * 3 + c;
            double v = vel64[idx] + half_dt_im * forces[idx];
            vel64[idx] = v;
            const double dx = dt_ase * v;
            pos64[idx] += dx;
            d2 += dx * dx;
        }
    }
    record_step_disp2(d2, max_step_disp2);
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

// The noise coefficients of this step come from the device (target[2..4]: ASE's c3, c4, c5 without the mass
// factor, for the step's target temperature), so a ramp can move them every step and a changed temperature needs
// no new graph. The second half-kick's first thread forms them for the next step: per atom this kernel does what
// it did when they were baked in.
__global__ void langevin_first_half_kernel(
    int N, double dt_ase, double c1, double c2, const double* __restrict__ target,
    unsigned long long seed, const unsigned long long* __restrict__ step_counter,
    const double* __restrict__ forces,
    const double* __restrict__ inv_masses,
    const double* __restrict__ sqrt_inv_masses,
    double* __restrict__ pos64,
    double* __restrict__ vel64,
    double* __restrict__ rnd_vel_save,
    unsigned int* __restrict__ max_step_disp2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double d2 = 0.0;
    if (i < N) {
    unsigned long long step = *step_counter;
    curandStatePhilox4_32_10_t rng;
    // Offset counts 32-bit outputs; the three curand_normal2_double calls consume 3 x 4 = 12, so 16 per step keeps
    // consecutive steps on disjoint words.
    curand_init(seed, (unsigned long long)i, step * 16ULL, &rng);
    double2 n0 = curand_normal2_double(&rng);
    double2 n1 = curand_normal2_double(&rng);
    double2 n2 = curand_normal2_double(&rng);
    double xi[3]  = {n0.x, n0.y, n1.x};
    double eta[3] = {n1.y, n2.x, n2.y};

    double im = inv_masses[i];
    double sim = sqrt_inv_masses[i];
    double c3 = target[2] * sim;
    double c4 = target[3] * sim;
    double c5 = target[4] * sim;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        int idx = i * 3 + c;
        double rnd_p = c5 * eta[c];
        double rnd_v = c3 * xi[c] - c4 * eta[c];
        rnd_vel_save[idx] = rnd_v;
        double v = vel64[idx] + c1 * forces[idx] * im - c2 * vel64[idx] + rnd_v;
        double dx = dt_ase * v + rnd_p;
        pos64[idx] += dx;
        vel64[idx] = (dx - rnd_p) / dt_ase;
        d2 += dx * dx;
    }
    }
    record_step_disp2(d2, max_step_disp2);
}

// ---- fix_com: the noise of a step conditioned on a centre of mass at rest ----
// Both random terms of an atom have a covariance proportional to 1/m_i, so their conditional mean, given that the
// mass-weighted sum over all atoms is zero, is the same mass-weighted mean for every atom. The noise is therefore
// drawn in a pass of its own that also sums m_i * rnd over the atoms (com[0..2]: positions' term, com[3..5]:
// velocities'); the half-kick then subtracts com / M. Same random numbers as langevin_first_half_kernel.
__global__ void langevin_noise_kernel(
    int N, const double* __restrict__ target,
    unsigned long long seed, const unsigned long long* __restrict__ step_counter,
    const double* __restrict__ masses,
    const double* __restrict__ sqrt_inv_masses,
    double* __restrict__ rnd_pos,
    double* __restrict__ rnd_vel,
    double* __restrict__ com
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double sums[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (i < N) {
        unsigned long long step = *step_counter;
        curandStatePhilox4_32_10_t rng;
        curand_init(seed, (unsigned long long)i, step * 16ULL, &rng);
        double2 n0 = curand_normal2_double(&rng);
        double2 n1 = curand_normal2_double(&rng);
        double2 n2 = curand_normal2_double(&rng);
        double xi[3]  = {n0.x, n0.y, n1.x};
        double eta[3] = {n1.y, n2.x, n2.y};
        double sim = sqrt_inv_masses[i];
        double c3 = target[2] * sim;
        double c4 = target[3] * sim;
        double c5 = target[4] * sim;
        double m = masses[i];
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            double rnd_p = c5 * eta[c];
            double rnd_v = c3 * xi[c] - c4 * eta[c];
            rnd_pos[i * 3 + c] = rnd_p;
            rnd_vel[i * 3 + c] = rnd_v;
            sums[c] = m * rnd_p;
            sums[3 + c] = m * rnd_v;
        }
    }
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        double s = sums[k];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if ((threadIdx.x & 31) == 0 && s != 0.0) atomicAdd(com + k, s);
    }
}

// The first half-kick with the noise of langevin_noise_kernel minus its mass-weighted mean. Writes the corrected
// velocity noise back for the second half-kick.
__global__ void langevin_first_half_fixcom_kernel(
    int N, double dt_ase, double c1, double c2, double inv_total_mass,
    const double* __restrict__ com,
    const double* __restrict__ forces,
    const double* __restrict__ inv_masses,
    const double* __restrict__ rnd_pos,
    double* __restrict__ rnd_vel,
    double* __restrict__ pos64,
    double* __restrict__ vel64,
    unsigned int* __restrict__ max_step_disp2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double d2 = 0.0;
    if (i < N) {
        double im = inv_masses[i];
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            int idx = i * 3 + c;
            double rnd_p = rnd_pos[idx] - com[c] * inv_total_mass;
            double rnd_v = rnd_vel[idx] - com[3 + c] * inv_total_mass;
            rnd_vel[idx] = rnd_v;
            double v = vel64[idx] + c1 * forces[idx] * im - c2 * vel64[idx] + rnd_v;
            double dx = dt_ase * v + rnd_p;
            pos64[idx] += dx;
            vel64[idx] = (dx - rnd_p) / dt_ase;
            d2 += dx * dx;
        }
    }
    record_step_disp2(d2, max_step_disp2);
}

// The total momentum, summed into com[0..2] (zeroed by the caller), and its removal: v -= P / M.
__global__ void momentum_sum_kernel(int N, const double* __restrict__ masses, const double* __restrict__ vel64,
                                    double* __restrict__ com) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        double s = i < N ? masses[i] * vel64[i * 3 + c] : 0.0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if ((threadIdx.x & 31) == 0 && s != 0.0) atomicAdd(com + c, s);
    }
}

__global__ void momentum_remove_kernel(int N, double inv_total_mass, const double* __restrict__ com,
                                       double* __restrict__ vel64) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    #pragma unroll
    for (int c = 0; c < 3; ++c) vel64[i * 3 + c] -= com[c] * inv_total_mass;
}

// Also advances the step counter and the thermostat's target: target[0] += target[1] (the ramp's change per step),
// then the next step's noise coefficients target[2..4] = a3, a4, a5 times sigma = sqrt(2 gamma kB T). Every
// first-half read of them for this step has completed.
__global__ void langevin_second_half_kernel(
    int N, double c1, double c2,
    const double* __restrict__ forces,
    const double* __restrict__ inv_masses,
    const double* __restrict__ rnd_vel_save,
    double* __restrict__ vel64,
    unsigned long long* __restrict__ step_counter,
    double* __restrict__ target, double a3, double a4, double a5, double two_fr_kb
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        (*step_counter) += 1ULL;
        const double next = target[0] + target[1];
        const double sigma = sqrt(two_fr_kb * fmax(next, 0.0));
        target[0] = next;
        target[2] = a3 * sigma;
        target[3] = a4 * sigma;
        target[4] = a5 * sigma;
    }
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
