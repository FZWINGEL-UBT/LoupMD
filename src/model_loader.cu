#include "model.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "bundle.hpp"
#include "generated_kernels_signature.h"

namespace {
template <typename T>
T* upload(const std::vector<T>& host) {
    T* dev = nullptr;
    CUDA_CHECK_THROW(cudaMalloc(&dev, std::max<size_t>(host.size(), 1) * sizeof(T)));
    if (!host.empty()) CUDA_CHECK_THROW(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return dev;
}
}  // namespace

void LoupModelGPU::load(const loup::Bundle& b, const char* path) {
    free_gpu();
    // Any throw below leaves the partially uploaded arrays to the caller's free_gpu (loup_create destroys the handle).
    if (b.meta("arch") != "omat0_medium_spec")
        throw std::runtime_error(std::string(path) + ": architecture '" + b.meta("arch") + "' is not supported by this build");
    if (b.meta_int("exporter_version") != 4)
        throw std::runtime_error(std::string(path) + ": exporter version " + b.meta("exporter_version") + ", expected 4 (re-export)");
    if (b.meta("kernel_signature") != LOUP_KERNEL_SIGNATURE)
        throw std::runtime_error(std::string(path) + ": bundle kernel signature '" + b.meta("kernel_signature") +
                                 "' does not match the compiled kernels '" LOUP_KERNEL_SIGNATURE "'");
    S = b.meta_int("S");
    P_sym = b.meta_int("P_sym");
    if (S < 1 || S > 127 || P_sym != S * (S + 1) / 2)
        throw std::runtime_error(std::string(path) + ": bad species count");
    r_max = (float)b.f64("r_max", {1})[0];
    scale = (float)b.f64("scale", {1})[0];
    {
        std::vector<int> z = b.meta_ints("Z");
        if ((int)z.size() != S) throw std::runtime_error(std::string(path) + ": Z list does not match S");
        atomic_numbers.assign(z.begin(), z.end());
    }
    atomic_energies = b.f64("e0", {S});

    pieces0 = b.meta_int("pieces0");
    n_bins0 = b.meta_int("n_bins0");
    r_lo0 = (float)b.f64("r_lo0", {1})[0];
    bin_scale0 = (float)b.f64("bin_scale0", {1})[0];
    d_starts0 = upload(b.f32("starts0", {pieces0}));
    d_scales0 = upload(b.f32("scales0", {pieces0}));
    {
        std::vector<int32_t> bins = b.i32("bins0", {n_bins0});
        for (int v : bins)
            if (v < 0 || v >= pieces0) throw std::runtime_error(std::string(path) + ": bins0 entry out of range");
        d_bins0 = upload(bins);
    }
    d_coef0 = upload(b.f32("coef0", {P_sym, pieces0, 6, 66}));
    d_pcoef0 = upload(b.f32("pcoef0", {P_sym, pieces0, 6, 66}));
    for (int l = 0; l < 4; ++l) d_C0[l] = upload(b.f32("C0_" + std::to_string(l), {S, S * 64, 128}));

    d_w0_factored = upload(b.f32("w0_factored", {S, 29, 128}));
    d_w_ro0 = upload(b.f32("w_ro0", {128}));
    d_W_up_0 = upload(b.f32("W_up_0", {128, 128}));
    d_W_up_1 = upload(b.f32("W_up_1", {128, 128}));
    d_W_skip1_16 = upload(b.f32("W_skip1_16", {S, 128, 16}));

    grid1.n_bands = b.meta_int("n_bands1");
    grid1.total_intervals = b.meta_int("intervals1");
    packed8_bias = (uint32_t)b.meta_int("packed8_bias");
    if (grid1.n_bands < 1 || grid1.n_bands > 8 || grid1.total_intervals < 1 || packed8_bias > 255)
        throw std::runtime_error(std::string(path) + ": bad interaction-1 spline grid header");
    bias_shift = (128u - packed8_bias) << 23;
    {
        const std::vector<float> st = b.f32("starts1", {8}), en = b.f32("ends1", {8}), ih = b.f32("inv_h1", {8});
        const std::vector<int32_t> of = b.i32("offsets1", {8}), co = b.i32("counts1", {8});
        int total = 0;
        for (int i = 0; i < 8; ++i) {
            grid1.starts[i] = st[i];
            grid1.ends[i] = en[i];
            grid1.inv_h[i] = ih[i];
            grid1.offsets[i] = of[i];
            grid1.counts[i] = co[i];
            if (i < grid1.n_bands) {
                if (of[i] != total || co[i] < 1 || !(en[i] > st[i]) || !(ih[i] > 0.0f))
                    throw std::runtime_error(std::string(path) + ": inconsistent interaction-1 spline bands");
                total += co[i];
            }
        }
        if (total != grid1.total_intervals) throw std::runtime_error(std::string(path) + ": spline band counts do not sum to the intervals");
    }
    const int64_t PI = (int64_t)P_sym * grid1.total_intervals;
    const size_t sp1_n = (size_t)PI * 1280;
    {
        const std::vector<float> flat = b.f32("spline_table_1", {PI, 1280, 4});
        std::vector<float4> host_sp(sp1_n);
        memcpy(host_sp.data(), flat.data(), sp1_n * sizeof(float4));
        int emin = 1000, emax = -1000;
        std::vector<int> exps(sp1_n, 0);
        for (size_t i = 0; i < sp1_n; ++i) {
            const float4 cf = host_sp[i];
            double a0 = (double)cf.x + 0.5 * (double)cf.y + 0.25 * (double)cf.z + 0.125 * (double)cf.w;
            double a1 = (double)cf.y + (double)cf.z + 0.75 * (double)cf.w;
            double a2 = (double)cf.z + 1.5 * (double)cf.w;
            double a3 = (double)cf.w;
            double top = std::max({std::abs(a0), std::abs(a1) * 256.0, std::abs(a2) * 16384.0, std::abs(a3) * 524288.0});
            if (top > 0.0) {
                int e = (int)std::floor(std::log2(top)) + 1;
                exps[i] = e;
                if (e < emin) emin = e;
                if (e > emax) emax = e;
            } else {
                exps[i] = 1000000;
            }
        }
        // 5-bit per-entry exponent: entries more than 2^31 below the largest are quantised on the coarser grid of
        // emax - 31 (absolute error unchanged, relative error larger only for those negligible entries).
        if (emin > emax) emin = emax = 0;
        if (emax - emin > 31) emin = emax - 31;
        if (emin < -108 || emax > 125)
            throw std::runtime_error("Interaction-1 spline coefficients outside the float range of the 8-byte cubic table");
        cubic_bias_shift = (unsigned)(128 + emin) << 23;
        std::vector<uint2> host_cubic(sp1_n);
        auto qval = [](double v, int ei, int sh, int nb) -> uint32_t {
            int64_t lim = (int64_t(1) << (nb - 1)) - 1;
            double step = std::ldexp(1.0, ei - sh - nb + 1);
            int64_t q = std::llround(v / step);
            if (q > lim) q = lim;
            else if (q < -lim) q = -lim;
            return (uint32_t)(q + (int64_t(1) << (nb - 1)));
        };
        auto deq = [](uint32_t u, int ei, int sh, int nb) {
            return (double)((int64_t)u - (int64_t(1) << (nb - 1))) * std::ldexp(1.0, ei - sh - nb + 1);
        };
        // Packing error per column, as GenericModel::load measures it (PackErrTracker, kPackedTableMaxRelErr).
        std::vector<double> ih(grid1.total_intervals);
        std::vector<char> phys(grid1.total_intervals);
        for (int k = 0; k < grid1.n_bands; ++k)
            for (int i = 0; i < grid1.counts[k]; ++i) {
                ih[grid1.offsets[k] + i] = grid1.inv_h[k];
                phys[grid1.offsets[k] + i] = grid1.starts[k] + (i + 1) / grid1.inv_h[k] > loup::kPackedPhysicalR;
            }
        loup::PackErrTracker perr(1280);
        for (size_t i = 0; i < sp1_n; ++i) {
            const float4 cf = host_sp[i];
            double a0 = (double)cf.x + 0.5 * (double)cf.y + 0.25 * (double)cf.z + 0.125 * (double)cf.w;
            double a1 = (double)cf.y + (double)cf.z + 0.75 * (double)cf.w;
            double a2 = (double)cf.z + 1.5 * (double)cf.w;
            double a3 = (double)cf.w;
            int ei = (exps[i] == 1000000 || exps[i] < emin) ? emin : exps[i];
            uint32_t u0 = qval(a0, ei, 0, 20);
            uint32_t u1 = qval(a1, ei, 8, 19);
            uint32_t u2 = qval(a2, ei, 14, 12);
            uint32_t u3 = qval(a3, ei, 19, 8);
            uint32_t eb_code = (uint32_t)(ei - emin);
            // Fields placed so the kernel extracts each with at most one funnel shift: u0 at x[22:3] and u1 at
            // y[22:4] (mantissa positions), u2 rotated so rotl(x, 20) puts it at [22:11], u3 rotated so
            // rotl(y, 19) puts it at [22:15], and the exponent code at y[27:23].
            auto rotr = [](uint32_t v, int k) { return (v >> k) | (v << (32 - k)); };
            host_cubic[i].x = (u0 << 3) | rotr(u2 << 11, 20);
            host_cubic[i].y = (u1 << 4) | rotr(u3 << 15, 19) | (eb_code << 23);
            const size_t iv = (i / 1280) % (size_t)grid1.total_intervals;
            const double av[4] = {a0, a1, a2, a3};
            const double dv[4] = {deq(u0, ei, 0, 20) - a0, deq(u1, ei, 8, 19) - a1, deq(u2, ei, 14, 12) - a2,
                                  deq(u3, ei, 19, 8) - a3};
            perr.add((int)(i % 1280), phys[iv] != 0, ih[iv], av, dv);
        }
        const double rel = perr.rel();
        if (std::getenv("LOUP_DEBUG_PACK")) fprintf(stderr, "pack interaction 1: %.3e\n", rel);
        if (rel > loup::kPackedTableMaxRelErr)
            throw std::runtime_error(std::string(path) + ": 8-byte interaction-1 table error " + std::to_string(rel) +
                                     " of the column maximum exceeds " + std::to_string(loup::kPackedTableMaxRelErr));
        d_cubic_packed8_1 = upload(host_cubic);
    }
    {
        const std::vector<int32_t> packed = b.i32("packed8_table_1", {PI, 1280, 2});
        std::vector<uint2> host(sp1_n);
        memcpy(host.data(), packed.data(), sp1_n * sizeof(uint2));
        d_packed8_table_1 = upload(host);
    }
    {
        const std::vector<float> dens = b.f32("density_table_1", {P_sym, grid1.total_intervals, 4});
        std::vector<float4> host((size_t)PI);
        memcpy(host.data(), dens.data(), host.size() * sizeof(float4));
        d_density_table_1 = upload(host);
    }

    const int lin_rows[4] = {256, 384, 384, 256};
    for (int l = 0; l < 4; ++l) d_W_lin1[l] = upload(b.f32("W_lin1_" + std::to_string(l), {lin_rows[l], 128}));

    d_w1_factored = upload(b.f32("w1_factored", {S, 13, 128}));
    d_W_p1_ro1 = upload(b.f32("W_p1_ro1", {128, 16}));
    d_W_ro1_2 = upload(b.f32("W_ro1_2", {16}));

    has_zbl = b.meta_int("has_zbl");
    {
        const std::vector<float> c = b.f32("zbl_c", {4});
        for (int i = 0; i < 4; ++i) zbl_c[i] = c[i];
    }
    zbl_p = (float)b.f64("zbl_p", {1})[0];
    zbl_r_max_max = (float)b.f64("zbl_r_max_max", {1})[0];
    d_zbl_r_cut = upload(b.f32("zbl_r_cut", {P_sym}));
    d_zbl_inv_a = upload(b.f32("zbl_inv_a", {P_sym}));
    d_zbl_pref = upload(b.f32("zbl_pref", {P_sym}));
    if (!(r_max > 0.0f) || !std::isfinite(r_max)) throw std::runtime_error(std::string(path) + ": bad r_max");
}

void LoupModelGPU::free_gpu() {
    auto freep = [](auto*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };
    freep(d_starts0);
    freep(d_scales0);
    freep(d_bins0);
    freep(d_coef0);
    freep(d_pcoef0);
    for (int l = 0; l < 4; ++l) freep(d_C0[l]);
    freep(d_w0_factored);
    freep(d_w_ro0);
    freep(d_W_up_0);
    freep(d_W_up_1);
    freep(d_W_skip1_16);
    freep(d_cubic_packed8_1);
    freep(d_packed8_table_1);
    freep(d_density_table_1);
    for (int l = 0; l < 4; ++l) freep(d_W_lin1[l]);
    freep(d_w1_factored);
    freep(d_W_p1_ro1);
    freep(d_W_ro1_2);
    freep(d_zbl_r_cut);
    freep(d_zbl_inv_a);
    freep(d_zbl_pref);
}
