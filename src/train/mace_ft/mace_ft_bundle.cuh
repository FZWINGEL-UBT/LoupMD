#pragma once
// LOUPC001 .loup bundle reader (into unfolded ModelFT) and writer (updated weights back to .loup).
// Zero PyTorch dependency: reads/writes the exact binary format of src/bundle.cu / loupmd/bundle.py.
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bundle.hpp"
#include "mace_ft_types.hpp"

namespace loup_ft {

#define FT_CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e)); \
} while (0)

namespace detail {

struct RawEntry {
    loup::DType dtype = loup::DType::U8;
    std::vector<uint64_t> dims;
    std::vector<uint8_t> data;
};

struct FileCloser {
    void operator()(FILE* f) const { if (f) fclose(f); }
};

inline size_t dtype_bytes(loup::DType d) {
    switch (d) {
        case loup::DType::F32: return 4;
        case loup::DType::F64: return 8;
        case loup::DType::I32: return 4;
        case loup::DType::U8:  return 1;
    }
    return 0;
}

inline std::map<std::string, RawEntry> read_raw_bundle(const std::string& path) {
    std::unique_ptr<FILE, FileCloser> fp(fopen(path.c_str(), "rb"));
    if (!fp) throw std::runtime_error("Cannot open .loup file: " + path);
    if (fseek(fp.get(), 0, SEEK_END) != 0) throw std::runtime_error("Cannot seek .loup file: " + path);
    const long size = ftell(fp.get());
    rewind(fp.get());
    if (size < 0) throw std::runtime_error("Cannot size .loup file: " + path);
    uint64_t rem = (uint64_t)size;
    auto rd = [&](void* dst, size_t n) {
        if (n > rem || (n && fread(dst, 1, n, fp.get()) != n))
            throw std::runtime_error(path + ": truncated .loup file");
        rem -= n;
    };
    char magic[8];
    rd(magic, 8);
    if (memcmp(magic, "LOUPC001", 8) != 0)
        throw std::runtime_error(path + ": not a LOUPC001 bundle");
    uint32_t n_entries = 0;
    rd(&n_entries, 4);
    std::map<std::string, RawEntry> out;
    for (uint32_t e = 0; e < n_entries; ++e) {
        uint16_t name_len = 0;
        rd(&name_len, 2);
        std::string name(name_len, '\0');
        rd(&name[0], name_len);
        uint8_t dt = 0, ndim = 0;
        rd(&dt, 1);
        rd(&ndim, 1);
        RawEntry ent;
        ent.dtype = (loup::DType)dt;
        ent.dims.resize(ndim);
        if (ndim) rd(ent.dims.data(), 8 * (size_t)ndim);
        uint64_t bytes = dtype_bytes(ent.dtype);
        for (uint64_t d : ent.dims) bytes *= d;
        ent.data.resize((size_t)bytes);
        rd(ent.data.data(), (size_t)bytes);
        out.emplace(std::move(name), std::move(ent));
    }
    char trailer[8];
    rd(trailer, 8);
    if (memcmp(trailer, "LOUPEND\0", 8) != 0 || rem != 0)
        throw std::runtime_error(path + ": corrupt LOUPC001 trailer");
    return out;
}

inline void write_raw_bundle(const std::string& path, const std::map<std::string, RawEntry>& entries) {
    const std::string tmp = path + ".tmp";
    {
        std::unique_ptr<FILE, FileCloser> fp(fopen(tmp.c_str(), "wb"));
        if (!fp) throw std::runtime_error("Cannot open output .loup file: " + tmp);
        auto wr = [&](const void* src, size_t n) {
            if (n && fwrite(src, 1, n, fp.get()) != n)
                throw std::runtime_error("Write failed for .loup file: " + tmp);
        };
        wr("LOUPC001", 8);
        const uint32_t n_entries = (uint32_t)entries.size();
        wr(&n_entries, 4);
        for (const auto& kv : entries) {
            const std::string& name = kv.first;
            const RawEntry& ent = kv.second;
            const uint16_t nlen = (uint16_t)name.size();
            wr(&nlen, 2);
            wr(name.data(), nlen);
            const uint8_t dt = (uint8_t)ent.dtype;
            const uint8_t ndim = (uint8_t)ent.dims.size();
            wr(&dt, 1);
            wr(&ndim, 1);
            if (ndim) wr(ent.dims.data(), 8 * (size_t)ndim);
            wr(ent.data.data(), ent.data.size());
        }
        wr("LOUPEND\0", 8);
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        remove(tmp.c_str());
        throw std::runtime_error("Failed to rename " + tmp + " -> " + path);
    }
}

template <typename T>
T* alloc_upload(std::vector<void*>& owner, const std::vector<T>& host) {
    T* dev = nullptr;
    FT_CUDA_CHECK(cudaMalloc(&dev, std::max<size_t>(host.size(), 1) * sizeof(T)));
    if (!host.empty())
        FT_CUDA_CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    owner.push_back(dev);
    return dev;
}

inline int register_param(ModelFT& m, const std::string& key, const std::vector<float>& host, bool is_head) {
    ParamTensor p;
    p.bundle_key = key;
    p.numel = (int64_t)host.size();
    p.is_head = is_head;
    p.trainable = false;
    if (!host.empty()) {
        double sum_sq = 0.0;
        for (float v : host) sum_sq += (double)v * (double)v;
        const float rms = (float)std::sqrt(sum_sq / (double)host.size());
        p.lr_scale = (rms > 0.0f && rms < 0.1f) ? (rms / 0.1f) : 1.0f;
    }
    const size_t bytes = std::max<size_t>(host.size(), 1) * sizeof(float);
    FT_CUDA_CHECK(cudaMalloc(&p.d_val, bytes));
    FT_CUDA_CHECK(cudaMalloc(&p.d_grad, bytes));
    FT_CUDA_CHECK(cudaMalloc(&p.d_m, bytes));
    FT_CUDA_CHECK(cudaMalloc(&p.d_v, bytes));
    FT_CUDA_CHECK(cudaMalloc(&p.d_ema, bytes));
    FT_CUDA_CHECK(cudaMalloc(&p.d_anchor, bytes));
    if (!host.empty()) {
        FT_CUDA_CHECK(cudaMemcpy(p.d_val, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
        FT_CUDA_CHECK(cudaMemcpy(p.d_ema, p.d_val, host.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        FT_CUDA_CHECK(cudaMemcpy(p.d_anchor, p.d_val, host.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    FT_CUDA_CHECK(cudaMemset(p.d_grad, 0, bytes));
    FT_CUDA_CHECK(cudaMemset(p.d_m, 0, bytes));
    FT_CUDA_CHECK(cudaMemset(p.d_v, 0, bytes));
    const int idx = (int)m.params.size();
    m.params.push_back(std::move(p));
    return idx;
}

}  // namespace detail

inline void free_model_ft(ModelFT& m) {
    for (auto& p : m.params) {
        if (p.d_val) cudaFree(p.d_val);
        if (p.d_grad) cudaFree(p.d_grad);
        if (p.d_m) cudaFree(p.d_m);
        if (p.d_v) cudaFree(p.d_v);
        if (p.d_ema) cudaFree(p.d_ema);
        if (p.d_anchor) cudaFree(p.d_anchor);
    }
    m.params.clear();
    for (void* ptr : m.aux_allocs) {
        if (ptr) cudaFree(ptr);
    }
    m.aux_allocs.clear();
    m.layers.clear();
    m.d_e0 = nullptr;
    m.zbl = ZblFT{};
}

inline void load_model_ft(const std::string& bundle_path, ModelFT& m) {
    free_model_ft(m);
    m.base_bundle_path = bundle_path;
    loup::Bundle b(bundle_path.c_str());
    if (b.meta("arch") != "generic")
        throw std::runtime_error("mace_ft: expected arch=generic bundle, got " + b.meta("arch"));
    if (b.meta_int("exporter_version") != 2)
        throw std::runtime_error("mace_ft: expected exporter_version=2");

    m.S = b.meta_int("S");
    m.n_layers = b.meta_int("n_layers");
    m.sh_perm = b.meta_int("sh_permute");
    m.Z = b.meta_ints("Z");
    m.r_max = (float)b.f64("r_max", {1})[0];
    m.e0 = b.f64("e0", {m.S});
    m.d_e0 = detail::alloc_upload(m.aux_allocs, m.e0);
    m.has_lr = (b.has_meta("has_lr") && b.meta_int("has_lr")) ? 1 : 0;

    {
        const auto& e = b.entry("embedding");
        m.C0 = (int)e.dims[1];
        m.p_embedding = detail::register_param(m, "embedding", b.f32("embedding", {m.S, m.C0}), false);
    }

    const int P = m.S * m.S;
    static const int n_l3_tab[3][4] = {{1, 1, 1, 1}, {2, 3, 3, 2}, {3, 5, 5, 4}};

    m.layers.resize(m.n_layers);
    for (int t = 0; t < m.n_layers; ++t) {
        LayerFT& L = m.layers[t];
        L.t = t;
        const std::string p = "L" + std::to_string(t) + ".";
        auto mi = [&](const char* k) { return b.meta_int(p + k); };

        L.kind = mi("kind");
        L.lmax_in = mi("lmax_in");
        L.lmax_x = mi("lmax_x");
        L.lmax_out = mi("lmax_out");
        L.C_in = mi("C_in");
        L.C_x = mi("C_x");
        L.C_msg = mi("C_msg");
        L.C_out = mi("C_out");
        L.n_paths = mi("n_paths");
        L.intervals = mi("intervals");
        L.has_sc = mi("has_sc");
        L.pair_sym = b.has_meta(p + "pair_sym") ? mi("pair_sym") : 0;
        L.first_hidden = b.has_meta(p + "first_hidden") ? mi("first_hidden") : 0;
        if (L.first_hidden) {
            L.J = mi("J");
            L.Jp = L.J - 1;
        }
        L.readout = mi("readout");
        for (int l = 0; l < 4; ++l) L.n_l3[l] = n_l3_tab[L.lmax_x][l];
        const int Q = L.n_paths * L.C_x;
        L.Q1 = L.first_hidden ? L.J : Q + 1;
        const bool nl = (L.kind == 4);
        for (int l = 0; l < 4; ++l) L.lin_cols[l] = (nl && l == 0) ? 4 * L.C_msg : L.C_msg;
        const bool is_top_layer = (t == m.n_layers - 1);

        // W_up
        for (int l = 0; l <= L.lmax_x; ++l) {
            const std::string k = p + "W_up_" + std::to_string(l);
            L.p_W_up[l] = detail::register_param(m, k, b.f32(k, {L.C_in, L.C_x}), false);
        }
        // first_C0 (layer 0 first_hidden)
        if (L.first_hidden) {
            for (int l = 0; l < 4; ++l) {
                const std::string k = p + "first_C0_" + std::to_string(l);
                L.p_first_C0[l] = detail::register_param(m, k, b.f32(k, {m.S * L.Jp, L.lin_cols[l]}), false);
            }
        }
        // W_lin
        for (int l = 0; l < 4; ++l) {
            const std::string k = p + "W_lin_" + std::to_string(l);
            L.p_W_lin[l] = detail::register_param(m, k, b.f32(k, {L.n_l3[l] * L.C_x, L.lin_cols[l]}), false);
        }
        // W_skip_msg (kind 0 or 2)
        if (L.kind == 0 || L.kind == 2) {
            for (int l = 0; l < 4; ++l) {
                const std::string k = p + "W_skip_msg_" + std::to_string(l);
                L.p_W_skip_msg[l] = detail::register_param(m, k, b.f32(k, {m.S, L.C_msg, L.C_msg}), false);
            }
        }
        // kind 4 nonlinear gate weights
        if (nl) {
            const auto ab = b.f64(p + "alpha_beta", {2});
            L.alpha = (float)ab[0];
            L.beta = (float)ab[1];
            const auto gc = b.f64(p + "gate_c", {2});
            L.gate_cs = (float)gc[0];
            L.gate_cg = (float)gc[1];
            for (int l = 0; l <= L.lmax_x; ++l) {
                const std::string k = p + "W_res_" + std::to_string(l);
                L.p_W_res[l] = detail::register_param(m, k, b.f32(k, {L.C_x, L.lin_cols[l]}), false);
            }
            for (int l = 0; l < 4; ++l) {
                const std::string k = p + "W_2_" + std::to_string(l);
                L.p_W_2[l] = detail::register_param(m, k, b.f32(k, {L.C_msg, L.C_msg}), false);
            }
        }
        // Skip connection W_sc
        if (L.has_sc) {
            L.sc_species = mi("sc_species");
            const int Ss = L.sc_species ? m.S : 1;
            for (int l = 0; l <= std::min(L.lmax_out, L.lmax_in); ++l) {
                const std::string k = p + "W_sc_" + std::to_string(l);
                if (b.has(k)) {
                    L.p_W_sc[l] = detail::register_param(m, k, b.f32(k, {Ss, L.C_in, L.C_out}), is_top_layer);
                    L.sc_has[l] = 1;
                }
            }
        }
        // Spline grid + FP32 spline coefficients
        {
            const auto g = b.f64(p + "grid", {-1, 3});
            const int nb = (int)(g.size() / 3);
            int off = 0;
            for (int k = 0; k < 8; ++k) {
                if (k < nb) {
                    const double a0 = g[3 * k], b0 = g[3 * k + 1];
                    const int n = (int)g[3 * k + 2];
                    L.grid.starts[k] = (float)a0;
                    L.grid.ends[k] = (float)b0;
                    L.grid.inv_h[k] = (float)(n / (b0 - a0));
                    L.grid.offsets[k] = off;
                    L.grid.counts[k] = n;
                    off += n;
                } else {
                    L.grid.starts[k] = L.grid.ends[k] = L.grid.inv_h[k] = 0.0f;
                    L.grid.offsets[k] = L.grid.counts[k] = 0;
                }
            }
            L.grid.n_bands = nb;
            L.grid.total_intervals = off;
            L.Ptab = L.pair_sym ? m.S * (m.S + 1) / 2 : P;
            const auto tab = b.f32(p + "spline", {L.Ptab, L.intervals, L.Q1, 4});
            const size_t n4 = (size_t)L.Ptab * L.intervals * L.Q1;
            std::vector<float4> sp4(n4);
            memcpy(sp4.data(), tab.data(), n4 * sizeof(float4));
            L.d_spline = detail::alloc_upload(m.aux_allocs, sp4);
            if (L.kind >= 2) {
                const size_t pi_n = (size_t)L.Ptab * L.intervals;
                std::vector<float4> dens(pi_n);
                for (size_t pi = 0; pi < pi_n; ++pi) dens[pi] = sp4[pi * L.Q1 + (L.Q1 - 1)];
                L.d_density_spline = detail::alloc_upload(m.aux_allocs, dens);
            }
        }
        // Product
        {
            L.prod_sig = b.meta(p + "prod_sig");
            L.prod_species = mi("prod_species");
            L.prod_segw = mi("prod_segments_w");
            if (b.has(p + "prod_src")) {
                const auto& raw = b.entry(p + "prod_src").data;
                L.prod_src.assign(raw.begin(), raw.end());
            }
            const int Sp = L.prod_species ? m.S : 1;
            const std::string k = p + "prod_w";
            L.p_prod_w = detail::register_param(m, k, b.f32(k, {Sp, L.prod_segw, L.C_msg}), is_top_layer);
        }
        // Post-product linear W_p
        for (int l = 0; l <= L.lmax_out; ++l) {
            const std::string k = p + "W_p_" + std::to_string(l);
            L.p_W_p[l] = detail::register_param(m, k, b.f32(k, {L.C_out, L.C_out}), is_top_layer);
        }
        // Readout
        if (L.readout == 0) {
            const std::string k = p + "ro_w";
            L.p_ro_w = detail::register_param(m, k, b.f32(k, {L.C_out}), true);
        } else if (L.readout == 1 || L.readout == 2) {
            L.ro_H = mi("ro_H");
            L.ro_act_c = (float)b.f64(p + "ro_act_c", {1})[0];
            L.p_ro_W1 = detail::register_param(m, p + "ro_W1", b.f32(p + "ro_W1", {L.C_out, L.ro_H}), true);
            L.p_ro_b1 = detail::register_param(m, p + "ro_b1", b.f32(p + "ro_b1", {L.ro_H}), true);
            L.p_ro_W2 = detail::register_param(m, p + "ro_W2", b.f32(p + "ro_W2", {L.ro_H}), true);
            if (L.readout == 2) {
                L.p_ro_Wm = detail::register_param(m, p + "ro_Wm", b.f32(p + "ro_Wm", {L.ro_H, L.ro_H}), true);
                L.p_ro_bm = detail::register_param(m, p + "ro_bm", b.f32(p + "ro_bm", {L.ro_H}), true);
            }
        }
    }

    // Optional frozen ZBL pair repulsion
    m.zbl.enabled = b.has_meta("has_zbl") ? b.meta_int("has_zbl") : 0;
    if (m.zbl.enabled) {
        const auto c = b.f64("zbl_c", {4});
        for (int i = 0; i < 4; ++i) m.zbl.c[i] = (float)c[i];
        m.zbl.p = (float)b.f64("zbl_p", {1})[0];
        const auto rc = b.f32("zbl_r_cut", {P});
        m.zbl.rmax = 0.0f;
        for (float v : rc) if (v > m.zbl.rmax) m.zbl.rmax = v;
        m.zbl.d_rc = detail::alloc_upload(m.aux_allocs, rc);
        m.zbl.d_inv_a = detail::alloc_upload(m.aux_allocs, b.f32("zbl_inv_a", {P}));
        m.zbl.d_pref = detail::alloc_upload(m.aux_allocs, b.f32("zbl_pref", {P}));
    }
}

inline void save_model_ft(const ModelFT& m, const std::string& out_path, bool use_ema = true) {
    auto entries = detail::read_raw_bundle(m.base_bundle_path);

    // Update e0 (f64 [S])
    {
        auto it = entries.find("e0");
        if (it == entries.end() || it->second.dtype != loup::DType::F64 ||
            it->second.data.size() != (size_t)m.S * sizeof(double))
            throw std::runtime_error("save_model_ft: e0 entry mismatch");
        memcpy(it->second.data.data(), m.e0.data(), (size_t)m.S * sizeof(double));
    }

    // Update every registered float32 parameter tensor
    for (const auto& p : m.params) {
        if (p.bundle_key.empty()) continue;
        auto it = entries.find(p.bundle_key);
        if (it == entries.end())
            throw std::runtime_error("save_model_ft: missing bundle entry " + p.bundle_key);
        if (it->second.dtype != loup::DType::F32 ||
            it->second.data.size() != (size_t)p.numel * sizeof(float))
            throw std::runtime_error("save_model_ft: shape/dtype mismatch for " + p.bundle_key);
        const float* src = use_ema ? p.d_ema : p.d_val;
        FT_CUDA_CHECK(cudaMemcpy(it->second.data.data(), src, (size_t)p.numel * sizeof(float), cudaMemcpyDeviceToHost));
    }

    detail::write_raw_bundle(out_path, entries);
}

}  // namespace loup_ft
