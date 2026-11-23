// Benchmark-only, enabled by -DLOUP_STUDENT_LAUNCH_BENCH; not a runtime engine feature.
#pragma once
#include <cuda_runtime.h>

namespace loup {
struct StudentProjectionGroup {
    struct Matrix {
        const float* a;
        const void* b;
        float* c;
        int rows, tile_start;
        float input_scale, output_scale;
    } matrix[12];  // 4 angular degrees x 3 occupied species in the benchmark
    int used{}, row_tiles{}, cols{}, inner{};
};
void launch_student_projection_group(const StudentProjectionGroup&, cudaStream_t);
}  // namespace loup
