#pragma once
// The typed kernel IR (JIT_DESIGN.md): per layer, the constants a kernel can have compiled in, lowered from the bundle
// at load. Stage 1 covers layer 0's edge kernels (layer0_kernels.cuh); later stages add the conv, product and
// epilogue records to this header.
#include <string>
#include <vector>

#include "kernels_common.cuh"

namespace loup {

struct Layer0IR {
    int S = 0;         // species
    int I = 0;         // spline intervals per table
    int J = 0;         // table columns (J - 1 radial, then the density)
    int Jt = 0;        // the table's row stride in float4 (J, or J - 1 without the density column)
    int K0 = 0;        // S0 row stride (S (J - 1) padded to 16)
    int sh_perm = 0;   // harmonics in (y, z, x) order
    int pair_sym = 0;  // table per ordered pair (0), unordered pair (1), one for all (2)
    bool zbl = false;
    SplineGrid grid{};
    bool bake_grid = false;  // the grid's numbers as literals (select chains over the bands) instead of the argument
};

// first_bwd_edge_kernel's columns per lane for a table of J columns (J - 1 <= 96; above, first_bwd_kernel runs)
inline int layer0_edge_cols(int J) { return J - 1 > 64 ? 3 : 2; }

// The NVRTC source of layer 0's kernels for one model: the LOUP_JIT_L0_* constants, the grid lookup with the grid's
// numbers as literals, then layer0_kernels.cuh.
std::string layer0_jit_source(const Layer0IR& ir);

// NVRTC name expressions of the kernels layer0_jit_source builds, in the order jit::build returns them:
// first_fwd_kernel, then first_bwd_edge_kernel without and with stress (ZBL as the IR says).
std::vector<std::string> layer0_jit_names(const Layer0IR& ir);

}  // namespace loup
