#pragma once
// NVRTC JIT compiler for exact forward, 1st-order backward, grad_w, and 2nd-order backward (Dual-number HVP)
// of ANY MACE factored symmetric contraction stored in L<t>.prod_src!
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "jit.hpp"
#include "mace_ft_types.hpp"

namespace loup_ft {

struct ProdFTKernels {
    const void* fwd = nullptr;      // gprod_fwd_<sig>
    const void* bwd = nullptr;      // gprod_bwd_<sig>
    const void* gradw = nullptr;    // ft_gprod_gradw_<sig>  (Phase 3b: dL/dw from bar_y)
    const void* bwd2 = nullptr;     // ft_gprod_bwd2_<sig>   (Phase 3a: Dual-number HVP -> tan_g, bar_m, bar_D, grad_w)
};

namespace detail {

inline std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// Extract the body between the opening '{' and matching '}' of a kernel named `fn_name` in `src`.
inline std::string extract_kernel_body(const std::string& src, const std::string& fn_name) {
    const size_t p = src.find(fn_name);
    if (p == std::string::npos)
        throw std::runtime_error("mace_ft_prod_jit: missing kernel " + fn_name + " in prod_src");
    const size_t ob = src.find('{', p);
    if (ob == std::string::npos)
        throw std::runtime_error("mace_ft_prod_jit: malformed kernel " + fn_name);
    int depth = 0;
    for (size_t i = ob; i < src.size(); ++i) {
        if (src[i] == '{') ++depth;
        else if (src[i] == '}') {
            if (--depth == 0) return src.substr(ob + 1, i - ob - 1);
        }
    }
    throw std::runtime_error("mace_ft_prod_jit: unclosed kernel " + fn_name);
}

// Build ft_gprod_gradw_<sig> and ft_gprod_bwd2_<sig> by mechanically lifting the exact SSA lines
// of gprod_fwd_<sig> and gprod_bwd_<sig> to Dual numbers.
inline std::string build_prod_ft_source(const std::string& sig, int lo, const std::string& prod_src) {
    const std::string fwd_body = extract_kernel_body(prod_src, "gprod_fwd_" + sig);
    const std::string bwd_body = extract_kernel_body(prod_src, "gprod_bwd_" + sig);

    // Parse lines from fwd_body:
    //  1) SSA lines: "  const float v... = ...;" (after loads, before "{ const float wb = ...")
    //  2) Feature lines: blocks "{ const float wb = __ldg(wr + <b> * C); o<cc> = fmaf(wb, <expr>, o<cc>); ... }"
    std::vector<std::string> fwd_ssa;
    struct FeatBlock {
        int b = -1;
        std::vector<std::pair<int, std::string>> items;  // (cc, expr)
    };
    std::vector<FeatBlock> feat_blocks;

    {
        std::istringstream iss(fwd_body);
        std::string line;
        bool in_active = false;
        bool seen_feat = false;
        while (std::getline(iss, line)) {
            if (line.find("if (active) {") != std::string::npos) { in_active = true; continue; }
            if (!in_active) continue;
            if (line.find("const float x") != std::string::npos) continue;  // loads
            if (line.find("const float wb = __ldg(wr + ") != std::string::npos) {
                seen_feat = true;
                FeatBlock fb;
                const size_t p0 = line.find("wr + ") + 5;
                const size_t p1 = line.find(" * C", p0);
                fb.b = std::stoi(line.substr(p0, p1 - p0));
                feat_blocks.push_back(std::move(fb));
                continue;
            }
            if (seen_feat) {
                if (line.find("fmaf(wb, ") != std::string::npos) {
                    // "    o<cc> = fmaf(wb, <expr>, o<cc>);"
                    const size_t o_pos = line.find('o');
                    const size_t eq_pos = line.find(" = ", o_pos);
                    const int cc = std::stoi(line.substr(o_pos + 1, eq_pos - (o_pos + 1)));
                    const size_t e0 = line.find("fmaf(wb, ") + 9;
                    const size_t e1 = line.rfind(", o");
                    feat_blocks.back().items.emplace_back(cc, line.substr(e0, e1 - e0));
                }
                if (line.find("y0[") != std::string::npos) break;
                continue;
            }
            if (line.find("const float ") != std::string::npos) {
                fwd_ssa.push_back(line);
            }
        }
    }

    // Parse adjoint lines from bwd_body (after SSA lines, between "const float w" and "dot = fmaf")
    std::vector<std::string> bwd_adj;
    std::string dx_exprs[16];
    for (int i = 0; i < 16; ++i) dx_exprs[i] = "Dual(0.0f, 0.0f)";
    {
        std::istringstream iss(bwd_body);
        std::string line;
        bool in_adj = false;
        while (std::getline(iss, line)) {
            if (line.find("const float w") != std::string::npos && line.find("__ldg(wr + ") != std::string::npos) {
                in_adj = true;
            }
            if (!in_adj) continue;
            if (line.find("dot = fmaf(") != std::string::npos) {
                // "  dot = fmaf(<dx>, x<i>, dot);"
                const size_t p0 = line.find("fmaf(") + 5;
                const size_t p1 = line.find(", x", p0);
                const size_t p2 = line.find(", dot", p1);
                const int idx = std::stoi(line.substr(p1 + 3, p2 - (p1 + 3)));
                std::string expr = line.substr(p0, p1 - p0);
                if (expr == "0.0f") expr = "Dual(0.0f, 0.0f)";
                dx_exprs[idx] = expr;
                continue;
            }
            if (line.find("dm0[") != std::string::npos || line.find("dm1[") != std::string::npos ||
                line.find("dm2[") != std::string::npos || line.find("dm3[") != std::string::npos) {
                continue;
            }
            if (line.find("if (dD)") != std::string::npos) break;
            if (line.find("}") != std::string::npos && line.find("{") == std::string::npos) break;
            bwd_adj.push_back(line);
        }
    }

    auto lift_dual = [](std::string line) {
        line = replace_all(line, "const float ", "const Dual ");
        line = replace_all(line, "  float ", "  Dual ");
        line = replace_all(line, "fmaf(", "dfmaf(");
        return line;
    };

    const int seg_out = (lo + 1) * (lo + 1);
    std::ostringstream oss;

    // 1) Prelude: Dual arithmetic + original gprod_fwd & gprod_bwd
    oss << R"DUAL(
#ifdef __CUDACC_RTC__
typedef long int64_t;
typedef unsigned long uint64_t;
typedef int int32_t;
typedef unsigned int uint32_t;
#else
#include <cuda_runtime.h>
#include <cstdint>
#endif

struct Dual {
  float v, d;
  __device__ __forceinline__ Dual() : v(0.0f), d(0.0f) {}
  __device__ __forceinline__ Dual(float a) : v(a), d(0.0f) {}
  __device__ __forceinline__ Dual(float a, float b) : v(a), d(b) {}
};
__device__ __forceinline__ Dual operator+(Dual a, Dual b) { return {a.v + b.v, a.d + b.d}; }
__device__ __forceinline__ Dual operator-(Dual a, Dual b) { return {a.v - b.v, a.d - b.d}; }
__device__ __forceinline__ Dual operator-(Dual a) { return {-a.v, -a.d}; }
__device__ __forceinline__ Dual operator*(Dual a, Dual b) { return {a.v * b.v, fmaf(a.v, b.d, a.d * b.v)}; }
__device__ __forceinline__ Dual operator*(float a, Dual b) { return {a * b.v, a * b.d}; }
__device__ __forceinline__ Dual operator*(Dual a, float b) { return {a.v * b, a.d * b}; }
__device__ __forceinline__ Dual dfmaf(Dual a, Dual b, Dual c) { return {fmaf(a.v, b.v, c.v), fmaf(a.v, b.d, fmaf(a.d, b.v, c.d))}; }
__device__ __forceinline__ Dual dfmaf(float a, Dual b, Dual c) { return {fmaf(a, b.v, c.v), fmaf(a, b.d, c.d)}; }
__device__ __forceinline__ Dual dfmaf(Dual a, float b, Dual c) { return {fmaf(a.v, b, c.v), fmaf(a.d, b, c.d)}; }
__device__ __forceinline__ Dual dfmaf(float a, float b, Dual c) { return {fmaf(a, b, c.v), c.d}; }

)DUAL";

    // Include original gprod_fwd_<sig> and gprod_bwd_<sig> directly
    // (Strip everything after gprod_bwd_<sig> to keep compile fast)
    {
        const size_t bpos = prod_src.find("gprod_bwd_" + sig);
        const size_t ob = prod_src.find('{', bpos);
        int depth = 0;
        size_t end_pos = prod_src.size();
        for (size_t i = ob; i < prod_src.size(); ++i) {
            if (prod_src[i] == '{') ++depth;
            else if (prod_src[i] == '}' && --depth == 0) { end_pos = i + 1; break; }
        }
        oss << prod_src.substr(0, end_pos) << "\n\n";
    }

    // 2) Phase 3b weight gradient kernel: ft_gprod_gradw_<sig>
    //    grad_w[sp[a], b, c] += sum_cc bar_y[a, cc, c] * phi_{b, cc}(x)
    oss << "extern \"C\" __global__ void __launch_bounds__(128) ft_gprod_gradw_" << sig << "(\n"
        << "    int N, int C, const float* __restrict__ m0, const float* __restrict__ m1,\n"
        << "    const float* __restrict__ m2, const float* __restrict__ m3,\n"
        << "    const float* __restrict__ D, int64_t w_stride, const int* __restrict__ sp,\n";
    for (int l = 0; l <= lo; ++l)
        oss << "    const float* __restrict__ by" << l << ",\n";
    oss << "    float* __restrict__ grad_w) {\n"
        << "  const int a = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;\n"
        << "  if (c >= C) return;\n"
        << "  const float s = D ? 1.0f / (D[a] + 1.0f) : 1.0f;\n"
        << "  float* gw = grad_w + (int64_t)sp[a] * w_stride + c;\n";
    for (int l = 0; l < 4; ++l)
        for (int m = 0; m < 2 * l + 1; ++m)
            oss << "  const float x" << (l * l + m) << " = __ldg(m" << l << " + ((int64_t)a * " << (2 * l + 1) << " + " << m << ") * C + c) * s;\n";
    for (int l = 0; l <= lo; ++l)
        for (int m = 0; m < 2 * l + 1; ++m)
            oss << "  const float g" << (l * l + m) << " = __ldg(by" << l << " + ((int64_t)a * " << (2 * l + 1) << " + " << m << ") * C + c);\n";
    for (const auto& ssa : fwd_ssa) oss << ssa << "\n";
    for (const auto& fb : feat_blocks) {
        if (fb.items.empty()) continue;
        oss << "  { float acc = 0.0f;\n";
        for (const auto& kv : fb.items)
            oss << "    acc = fmaf(g" << kv.first << ", " << kv.second << ", acc);\n";
        oss << "    atomicAdd(gw + " << fb.b << " * C, acc);\n  }\n";
    }
    oss << "}\n\n";

    // 3) Phase 3a 2nd-order backward kernel: ft_gprod_bwd2_<sig> (exact Dual-number HVP!)
    oss << "extern \"C\" __global__ void __launch_bounds__(128) ft_gprod_bwd2_" << sig << "(\n"
        << "    int N, int C, const float* __restrict__ m0, const float* __restrict__ m1,\n"
        << "    const float* __restrict__ m2, const float* __restrict__ m3,\n"
        << "    const float* __restrict__ D, const float* __restrict__ w, int64_t w_stride, const int* __restrict__ sp,\n";
    for (int l = 0; l <= lo; ++l)
        oss << "    const float* __restrict__ gin" << l << ",\n";
    oss << "    const float* __restrict__ tdm0, const float* __restrict__ tdm1,\n"
        << "    const float* __restrict__ tdm2, const float* __restrict__ tdm3,\n"
        << "    const float* __restrict__ tan_dD,\n";
    for (int l = 0; l <= lo; ++l)
        oss << "    float* __restrict__ tan_g" << l << ",\n";
    oss << "    float* __restrict__ bar_m0, float* __restrict__ bar_m1,\n"
        << "    float* __restrict__ bar_m2, float* __restrict__ bar_m3,\n"
        << "    float* __restrict__ bar_D, float* __restrict__ grad_w) {\n"
        << "  const int a = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;\n"
        << "  const bool active = c < C;\n"
        << "  const float s = D ? 1.0f / (D[a] + 1.0f) : 1.0f;\n"
        << "  const float tdD = (D && tan_dD) ? tan_dD[a] : 0.0f;\n"
        << "  const float* wr = w + (int64_t)sp[a] * w_stride + c;\n"
        << "  float* gw = grad_w ? (grad_w + (int64_t)sp[a] * w_stride + c) : nullptr;\n"
        << "  float bar_s_acc = 0.0f;\n"
        << "  if (active) {\n";

    // Load m_i, tdm_i, build Dual x_i = (m_i * s, s * tdm_i - s * tdD * x_i.v)
    for (int l = 0; l < 4; ++l) {
        for (int m = 0; m < 2 * l + 1; ++m) {
            const int i = l * l + m;
            const std::string idx = "((int64_t)a * " + std::to_string(2 * l + 1) + " + " + std::to_string(m) + ") * C + c";
            oss << "  const float mv" << i << " = __ldg(m" << l << " + " << idx << ");\n"
                << "  const float xv" << i << " = mv" << i << " * s;\n"
                << "  const float tdmv" << i << " = __ldg(tdm" << l << " + " << idx << ");\n"
                << "  const Dual x" << i << "(xv" << i << ", s * tdmv" << i << " - s * tdD * xv" << i << ");\n";
        }
    }
    for (int l = 0; l <= lo; ++l) {
        for (int m = 0; m < 2 * l + 1; ++m) {
            const int cc = l * l + m;
            const std::string idx = "((int64_t)a * " + std::to_string(2 * l + 1) + " + " + std::to_string(m) + ") * C + c";
            oss << "  const Dual g" << cc << "(__ldg(gin" << l << " + " << idx << "), 0.0f);\n";
        }
    }
    // Dual forward SSA
    for (const auto& ssa : fwd_ssa) oss << lift_dual(ssa) << "\n";

    // Accumulate tan_g[cc] = sum_b w_b * phi_{b,cc}.d  AND  grad_w[b] += sum_cc g_cc * phi_{b,cc}.d
    oss << "  float tg[" << seg_out << "];\n"
        << "  #pragma unroll\n"
        << "  for (int i = 0; i < " << seg_out << "; ++i) tg[i] = 0.0f;\n";
    for (const auto& fb : feat_blocks) {
        if (fb.items.empty()) continue;
        oss << "  { const float wb = __ldg(wr + " << fb.b << " * C);\n"
            << "    float gwb = 0.0f;\n";
        for (const auto& kv : fb.items) {
            oss << "    { const Dual ph = " << kv.second << ";\n"
                << "      tg[" << kv.first << "] = fmaf(wb, ph.d, tg[" << kv.first << "]);\n"
                << "      gwb = fmaf(g" << kv.first << ".v, ph.d, gwb); }\n";
        }
        oss << "    if (gw) atomicAdd(gw + " << fb.b << " * C, gwb);\n  }\n";
    }
    for (int l = 0; l <= lo; ++l) {
        for (int m = 0; m < 2 * l + 1; ++m) {
            const int cc = l * l + m;
            const std::string idx = "((int64_t)a * " + std::to_string(2 * l + 1) + " + " + std::to_string(m) + ") * C + c";
            oss << "  tan_g" << l << "[" << idx << "] = tg[" << cc << "];\n";
        }
    }

    // Dual backward adjoint -> dx_dual[i] = (dx_i, sum_j t_j * d^2f/(dx_i dx_j))
    for (const auto& line : bwd_adj) oss << lift_dual(line) << "\n";

    // Accumulate bar_m[i] += s * (dx_dual.d - s * tdD * dx_dual.v) and bar_s_acc
    for (int l = 0; l < 4; ++l) {
        for (int m = 0; m < 2 * l + 1; ++m) {
            const int i = l * l + m;
            const std::string idx = "((int64_t)a * " + std::to_string(2 * l + 1) + " + " + std::to_string(m) + ") * C + c";
            oss << "  { const Dual dx = " << dx_exprs[i] << ";\n"
                << "    const float bx = dx.d - s * tdD * dx.v;\n"
                << "    bar_m" << l << "[" << idx << "] += s * bx;\n"
                << "    bar_s_acc = fmaf(bx, mv" << i << ", fmaf(tdmv" << i << ", dx.v, fmaf(-tdD, dx.v * xv" << i << ", bar_s_acc))); }\n";
        }
    }
    oss << "  }\n"
        << "  if (D && bar_D) {\n"
        << "    #pragma unroll\n"
        << "    for (int off = 16; off > 0; off >>= 1) bar_s_acc += __shfl_xor_sync(0xffffffffu, bar_s_acc, off);\n"
        << "    if ((threadIdx.x & 31) == 0) atomicAdd(bar_D + a, -s * s * bar_s_acc);\n"
        << "  }\n"
        << "}\n";

    return oss.str();
}

}  // namespace detail

inline const ProdFTKernels& get_prod_ft_kernels(const LayerFT& L) {
    static std::mutex mu;
    static std::map<std::string, ProdFTKernels> cache;
    const std::string key = L.prod_sig + ":" + std::to_string(L.lmax_out);
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    if (L.prod_src.empty())
        throw std::runtime_error("mace_ft: bundle layer " + std::to_string(L.t) + " missing prod_src (re-export with current loupmd.export)");

    const std::string full_src = detail::build_prod_ft_source(L.prod_sig, L.lmax_out, L.prod_src);
    const std::vector<std::string> names = {
        "gprod_fwd_" + L.prod_sig,
        "gprod_bwd_" + L.prod_sig,
        "ft_gprod_gradw_" + L.prod_sig,
        "ft_gprod_bwd2_" + L.prod_sig,
    };
    std::string err;
    const std::vector<const void*> handles = loup::jit::build("mace_ft_prod", full_src, {}, names, &err);
    if (handles.size() != names.size())
        throw std::runtime_error("mace_ft NVRTC compile failed for product " + L.prod_sig + ": " + err);

    ProdFTKernels k;
    k.fwd = handles[0];
    k.bwd = handles[1];
    k.gradw = handles[2];
    k.bwd2 = handles[3];
    return cache.emplace(key, k).first->second;
}

}  // namespace loup_ft
