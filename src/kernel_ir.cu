// Lowering of the kernel IR to NVRTC sources (kernel_ir.hpp, JIT_DESIGN.md).
#include "kernel_ir.hpp"

#include <cstdio>

namespace loup {

namespace {

// An exact float literal (hexadecimal: the value is the float's own, no decimal rounding)
std::string flit(float v) {
    char b[48];
    snprintf(b, sizeof(b), "%af", (double)v);
    return b;
}

// k == 0 ? v[0] : k == 1 ? v[1] : ... over the grid's bands
template <typename T, typename F>
std::string by_band(int nb, const T* v, F lit) {
    std::string s = lit(v[nb - 1]);
    for (int k = nb - 2; k >= 0; --k) s = "(k == " + std::to_string(k) + " ? " + lit(v[k]) + " : " + s + ")";
    return s;
}

}  // namespace

std::string layer0_jit_source(const Layer0IR& ir) {
    std::string s;
    s += "// LoupMD layer-0 kernels, JIT (kernel_ir.cu)\n";
    s += "#define LOUP_JIT_L0 1\n";
    s += "#define LOUP_JIT_L0_S " + std::to_string(ir.S) + "\n";
    s += "#define LOUP_JIT_L0_I " + std::to_string(ir.I) + "\n";
    s += "#define LOUP_JIT_L0_J " + std::to_string(ir.J) + "\n";
    s += "#define LOUP_JIT_L0_Jt " + std::to_string(ir.Jt) + "\n";
    s += "#define LOUP_JIT_L0_K0 " + std::to_string(ir.K0) + "\n";
    s += "#define LOUP_JIT_L0_sh_perm " + std::to_string(ir.sh_perm) + "\n";
    s += "#define LOUP_JIT_L0_pair_sym " + std::to_string(ir.pair_sym) + "\n";
    s += "#include \"kernels_common.cuh\"\n";
    if (!ir.bake_grid) {
        s += "#include \"layer0_kernels.cuh\"\n";
        return s;
    }
    s += "#define LOUP_JIT_L0_GRID 1\n";
    // locate_spline (kernels_common.cuh) on this grid: the same band test, operations and order
    const SplineGrid& g = ir.grid;
    const int nb = g.n_bands < 1 ? 1 : (g.n_bands > 8 ? 8 : g.n_bands);
    auto ilit = [](int v) { return std::to_string(v); };
    int cm1[8];
    for (int k = 0; k < 8; ++k) cm1[k] = g.counts[k] - 1;
    s += "__device__ __forceinline__ void loup_jit_l0_locate(float r, int& iv, float& t, float& sx_inv) {\n";
    s += "    int k = 0;\n";
    for (int i = 0; i < nb - 1 && i < 7; ++i)
        s += "    if (r >= " + flit(g.ends[i]) + ") k = " + std::to_string(i + 1) + ";\n";
    s += "    (void)k;\n";
    s += "    sx_inv = " + by_band(nb, g.inv_h, flit) + ";\n";
    s += "    float u = (r - " + by_band(nb, g.starts, flit) + ") * sx_inv;\n";
    s += "    u = u > 0.0f ? u : 0.0f;\n";
    s += "    int il = (int)u;\n";
    s += "    const int mx = " + by_band(nb, cm1, ilit) + ";\n";
    s += "    il = il < mx ? il : mx;\n";
    s += "    t = u - (float)il;\n";
    s += "    iv = " + by_band(nb, g.offsets, ilit) + " + il;\n";
    s += "}\n";
    s += "#include \"layer0_kernels.cuh\"\n";
    return s;
}

std::vector<std::string> layer0_jit_names(const Layer0IR& ir) {
    const std::string c = std::to_string(layer0_edge_cols(ir.J)), z = ir.zbl ? "true" : "false";
    return {"&first_fwd_kernel", "first_bwd_edge_kernel<" + c + ", " + z + ", false, 0>",
            "first_bwd_edge_kernel<" + c + ", " + z + ", true, 0>"};
}

}  // namespace loup
