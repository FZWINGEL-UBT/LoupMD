## Force call at equal skin (ms, median of 100; pass 1 / pass 2)

skin 0.5 A

| implementation | 1080 atoms | 5000 atoms | 13720 atoms |
|---|---|---|---|
| loup | 2.48 / 2.59 | 8.87 / 8.88 | 23.85 / 23.77 |
| loup-generic | 2.46 / 2.51 (0.98x) | 8.99 / 9.08 (1.02x) | 23.85 / 24.31 (1.01x) |
| sx-f32 | 3.19 / 3.27 (1.27x) | 13.73 / 13.83 (1.55x) | 36.31 / 36.24 (1.52x) |
| sx-mix | 2.82 / 2.77 (1.10x) | 11.96 / 11.84 (1.34x) | 32.03 / 31.62 (1.34x) |

skin 1.0 A

| implementation | 1080 atoms | 5000 atoms | 13720 atoms |
|---|---|---|---|
| loup | 2.49 / 2.57 | 8.90 / 9.10 | 24.04 / 24.10 |
| loup-generic | 2.63 / 2.51 (1.02x) | 8.93 / 8.96 (0.99x) | 24.05 / 24.64 (1.01x) |
| sx-f32 | 3.35 / 3.29 (1.31x) | 13.81 / 13.77 (1.53x) | 36.50 / 36.39 (1.51x) |
| sx-mix | 2.85 / 2.74 (1.10x) | 11.94 / 11.78 (1.32x) | 32.14 / 31.67 (1.33x) |

## Kernel profile of one force call (nsys, SrTiO3 5,000 atoms)

### loup: 20 calls, device span 8612 us, kernels 8675 us, memcpy / memset 30 us per call

| phase | us per call |
|---|---|
| layer 1 conv backward | 2923 |
| GEMMs | 2348 |
| layer 1 conv forward | 1924 |
| other | 1088 |
| products | 337 |
| neighbour list / skin / edges | 54 |

| kernel | launches per call | us per call |
|---|---|---|
| `wprime_backward_inter1_rcv` | 1 | 2923 |
| `void spline_forward_inter1<(int)1, (bool)1>(int, const int *, const float *, const float *, const float4 *,...` | 1 | 1924 |
| `void loup::<unnamed>::sgemm128_kernel<(int)8, (bool)1, (bool)1>(int, int, int, const float *, const float *...` | 13 | 1238 |
| `loup::tabled_outer_sum0_backward_kernel(const int *, int, int, int, float, float, float, int, float, float,...` | 1 | 570 |
| `void loup::<unnamed>::sgemm128_kernel<(int)4, (bool)1, (bool)1>(int, int, int, const float *, const float *...` | 10 | 459 |
| `loup::tabled_outer_sum0_forward_kernel(int, int, int, int, float, float, const int *, const float4 *, const...` | 1 | 246 |
| `sc0_backward` | 1 | 229 |
| `sc1_fused` | 1 | 226 |
| `fm_tf32x3_rr_64x64` | 5 | 207 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>(T1::Params)` | 1 | 179 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(T1::Params)` | 3 | 138 |
| `sc0_forward` | 1 | 108 |
| `void loup::<unnamed>::sgemm32_kernel<(bool)1>(int, int, int, const float *, const float *, float *, float *...` | 4 | 63 |
| `loup::scatter_active_edges_kernel(int, float, SplineGrid, int, int, const int *, const int *, const int *, ...` | 1 | 49 |
| `loup::compute_and_count_active_kernel(int, float, const double *, const int *, const int *, const int *, co...` | 1 | 44 |
| `void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (int)4>(in...` | 1 | 31 |
| `void loup::<unnamed>::sgemm32_kernel<(bool)0>(int, int, int, const float *, const float *, float *, float *...` | 2 | 16 |
| `fm_tf32x3_rr_64x32` | 1 | 12 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(T1::Params)` | 2 | 6 |

### loup-generic: 20 calls, device span 8468 us, kernels 8620 us, memcpy / memset 22 us per call

| phase | us per call |
|---|---|
| layer 1 conv backward | 2872 |
| GEMMs | 2324 |
| layer 1 conv forward | 2002 |
| neighbour list / skin / edges | 600 |
| products | 559 |
| layer 0 conv forward | 220 |
| other | 43 |

| kernel | launches per call | us per call |
|---|---|---|
| `void gconv_bwd_L1<(bool)1, (int)128, (bool)1>(int, int, int, int, SplineGrid, int, int, const int *, const ...` | 1 | 2872 |
| `void gconv_fwd_L1<(int)1, (bool)1, (int)256, (bool)1, (int)128>(int, int, int, int, SplineGrid, int, int, c...` | 1 | 2002 |
| `void loup::<unnamed>::sgemm128_kernel<(int)8, (bool)1, (bool)1>(int, int, int, const float *, const float *...` | 12 | 1213 |
| `void loup::<unnamed>::first_bwd_edge_kernel<(int)2, (bool)1, (bool)1>(const int *, int, int, int, int, Spli...` | 1 | 546 |
| `void loup::<unnamed>::sgemm128_kernel<(int)4, (bool)1, (bool)1>(int, int, int, const float *, const float *...` | 10 | 423 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 1 | 240 |
| `void gprod_ro1_fused_ae0a447b65f4fcd4<(int)128, (int)4>(int, int, const float *, const float *, const float...` | 1 | 221 |
| `loup::<unnamed>::first_fwd_kernel(int, int, int, int, int, SplineGrid, int, int, const int *, const int *, ...` | 1 | 220 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>(T1::Params)` | 1 | 188 |
| `fm_tf32x3_rr_64x64` | 5 | 185 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(T1::Params)` | 3 | 135 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 1 | 98 |
| `void loup::<unnamed>::sgemm32_kernel<(bool)1>(int, int, int, const float *, const float *, float *, float *...` | 4 | 66 |
| `loup::scatter_active_edges_kernel(int, float, SplineGrid, int, int, const int *, const int *, const int *, ...` | 1 | 50 |
| `fm_tf32x3_rc_64x64` | 1 | 48 |
| `loup::compute_and_count_active_kernel(int, float, const double *, const int *, const int *, const int *, co...` | 1 | 42 |
| `void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (int)4>(in...` | 1 | 32 |
| `fm_tf32x3_rr_64x32` | 1 | 18 |
| `void loup::<unnamed>::sgemm32_kernel<(bool)0>(int, int, int, const float *, const float *, float *, float *...` | 2 | 10 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_align1>(T1::Params)` | 2 | 6 |

### sx-f32: 20 calls, device span 13615 us, kernels 13181 us, memcpy / memset 48 us per call

| phase | us per call |
|---|---|
| layer 1 conv backward (Phi1_reverse) | 4562 |
| layer 1 conv forward (Phi1) | 4065 |
| GEMMs | 2561 |
| products backward (k_M_reverse) | 453 |
| layer 0 conv backward (Phi0_reverse) | 446 |
| neighbour edges (count, scan, fill) | 326 |
| layer 0 conv forward (Phi0) | 256 |
| readouts, ZBL, outputs | 223 |
| products forward (k_M) | 202 |
| edge -> atom forces | 59 |
| layer 1 density (scale1) | 29 |

| kernel | launches per call | us per call |
|---|---|---|
| `void <unnamed>::k_Phi1_reverse<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, (bool)0, float>(mace_...` | 1 | 4562 |
| `void <unnamed>::k_Phi1<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, (bool)0, float>(mace_foundati...` | 1 | 4065 |
| `void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (int)4>(in...` | 14 | 1116 |
| `void magma_sgemmEx_kernel<float, float, float, (bool)0, (bool)0, (int)6, (int)3, (int)5, (int)3, (int)3>(in...` | 4 | 829 |
| `void <unnamed>::k_Phi0_reverse<(int)64, float>(mace_foundation::Model<T2>, mace_foundation::Graph<T2>)` | 1 | 446 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(T1::Params)` | 10 | 440 |
| `void <unnamed>::k_Phi0<(int)64, float>(mace_foundation::Model<T2>, mace_foundation::Graph<T2>)` | 1 | 256 |
| `void <unnamed>::k_M_reverse<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, (int)0, float>(mace_foun...` | 1 | 248 |
| `void <unnamed>::k_fill<float>(mace_foundation::Model<T1>, mace_foundation::Graph<T1>, <unnamed>::Atoms)` | 1 | 238 |
| `void <unnamed>::k_M_reverse<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, (int)1, float>(mace_foun...` | 1 | 205 |
| `void <unnamed>::k_sgemm<(int)64, (int)32>(<unnamed>::GemmBatch)` | 2 | 140 |
| `void <unnamed>::k_outputs<float>(mace_foundation::Graph<T1>, double *, double *)` | 1 | 113 |
| `void <unnamed>::k_M<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, (int)1, float>(mace_foundation::...` | 1 | 102 |
| `void <unnamed>::k_M<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, (int)0, float>(mace_foundation::...` | 1 | 99 |
| `void <unnamed>::k_count<float>(mace_foundation::Model<T1>, mace_foundation::Graph<T1>, <unnamed>::Atoms)` | 1 | 86 |
| `void <unnamed>::k_readouts<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, float>(mace_foundation::M...` | 1 | 70 |
| `void <unnamed>::k_forces<float>(mace_foundation::Model<T1>, mace_foundation::Graph<T1>, <unnamed>::Atoms)` | 1 | 59 |
| `void <unnamed>::k_zbl<float>(mace_foundation::Model<T1>, mace_foundation::Graph<T1>)` | 1 | 40 |
| `void <unnamed>::k_scale1<float>(mace_foundation::Model<T1>, mace_foundation::Graph<T1>)` | 1 | 29 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(T1::Params)` | 2 | 19 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(T1::Params)` | 2 | 17 |

### sx-mix: 20 calls, device span 11769 us, kernels 11375 us, memcpy / memset 36 us per call

| phase | us per call |
|---|---|
| layer 1 conv backward (Phi1_reverse) | 3901 |
| layer 1 conv forward (Phi1) | 3167 |
| GEMMs | 2214 |
| products backward (k_M_reverse) | 487 |
| layer 0 conv backward (Phi0_reverse) | 454 |
| neighbour edges (count, scan, fill) | 331 |
| layer 0 conv forward (Phi0) | 277 |
| readouts, ZBL, outputs | 238 |
| products forward (k_M) | 194 |
| edge -> atom forces | 70 |
| layer 1 density (scale1) | 28 |
| gates, biases, amax | 14 |

| kernel | launches per call | us per call |
|---|---|---|
| `void <unnamed>::k_Phi1_reverse<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, (bool)1, float>(mace_...` | 1 | 3901 |
| `void <unnamed>::k_Phi1<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, (bool)1, float>(mace_foundati...` | 1 | 3167 |
| `void <unnamed>::k_gemm<(int)128, (int)128, (int)4, (int)2>(<unnamed>::GemmBatch, const unsigned int *)` | 5 | 1781 |
| `void <unnamed>::k_Phi0_reverse<(int)64, float>(mace_foundation::Model<T2>, mace_foundation::Graph<T2>)` | 1 | 454 |
| `void <unnamed>::k_gemm<(int)128, (int)64, (int)4, (int)2>(<unnamed>::GemmBatch, const unsigned int *)` | 1 | 433 |
| `void <unnamed>::k_Phi0<(int)64, float>(mace_foundation::Model<T2>, mace_foundation::Graph<T2>)` | 1 | 277 |
| `void <unnamed>::k_M_reverse<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, (int)0, float>(mace_foun...` | 1 | 264 |
| `void <unnamed>::k_fill<float>(mace_foundation::Model<T1>, mace_foundation::Graph<T1>, <unnamed>::Atoms)` | 1 | 241 |
| `void <unnamed>::k_M_reverse<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, (int)1, float>(mace_foun...` | 1 | 223 |
| `void <unnamed>::k_outputs<float>(mace_foundation::Graph<T1>, double *, double *)` | 1 | 134 |
| `void <unnamed>::k_M<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, (int)0, float>(mace_foundation::...` | 1 | 102 |
| `void <unnamed>::k_M<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, (int)1, float>(mace_foundation::...` | 1 | 93 |
| `void <unnamed>::k_count<float>(mace_foundation::Model<T1>, mace_foundation::Graph<T1>, <unnamed>::Atoms)` | 1 | 87 |
| `void <unnamed>::k_readouts<<unnamed>::Shape<(int)128, (int)1, (int)128, (bool)0>, float>(mace_foundation::M...` | 1 | 81 |
| `void <unnamed>::k_forces<float>(mace_foundation::Model<T1>, mace_foundation::Graph<T1>, <unnamed>::Atoms)` | 1 | 70 |
| `void <unnamed>::k_scale1<float>(mace_foundation::Model<T1>, mace_foundation::Graph<T1>)` | 1 | 28 |
| `void <unnamed>::k_zbl<float>(mace_foundation::Model<T1>, mace_foundation::Graph<T1>)` | 1 | 23 |
| `void <unnamed>::k_amax<float>(mace_foundation::Model<T1>, const T1 *, long, int)` | 1 | 14 |

| implementation | calls | device span per call (us) | kernel time per call (us) | memcpy / memset (us) |
|---|---|---|---|---|
| loup | 20 | 8612 | 8675 | 30 |
| loup-generic | 20 | 8468 | 8620 | 22 |
| sx-f32 | 20 | 13615 | 13181 | 48 |
| sx-mix | 20 | 11769 | 11375 | 36 |
