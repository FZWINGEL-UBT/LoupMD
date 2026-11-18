# Kernel time per MD step (nsys)

| Model | System | Atoms | Steps | Rebuilds | ms / step |
|---|---|---:|---:|---:|---:|
| mace-omat-0-medium.model | srtio3 | 5000 | 64 | 4 | 7.9836 |
| mace-omat-0-medium.model | srtio3 | 5000 | 64 | 4 | 7.8038 |
| 2023-12-03-mace-128-L1_epoch-199.model | srtio3 | 5000 | 64 | 4 | 3.8149 |
| 2023-12-03-mace-128-L1_epoch-199.model | nbcl5 | 1944 | 128 | 5 | 0.8902 |
| MACE_MPtrj_2022.9.model | srtio3 | 5000 | 64 | 3 | 10.8188 |
| mace-omat-0-small.model | srtio3 | 5000 | 64 | 4 | 4.4954 |
| mace-mh-1.model | srtio3 | 5000 | 16 | 1 | 33.0429 |
| mace-mh-1.model | nacl216 | 216 | 128 | 5 | 1.4183 |
| MACE-OFF23_small.model | water216 | 648 | 256 | 62 | 0.3974 |
| MACE-OFF23_medium.model | water216 | 648 | 256 | 61 | 0.9172 |
| MACE-OFF24_medium.model | water | 5184 | 32 | 10 | 9.1206 |
| MACE-OFF23_large.model | water216 | 648 | 64 | 16 | 3.0445 |
| MACE-omol-0-extra-large-1024.model | water216 | 648 | 32 | 8 | 22.6052 |
| MACE-POLAR-1-S.model | nacl216 | 216 | 128 | 5 | 1.2375 |
| MACE-POLAR-1-S.model | nbcl5 | 1944 | 32 | 2 | 7.9641 |
| MACE-POLAR-1-M.model | nacl216 | 216 | 128 | 5 | 2.9416 |
| MACE-POLAR-1-M.model | nbcl5 | 1944 | 16 | 1 | 22.0890 |
| MACE-POLAR-1-L.model | nacl216 | 216 | 64 | 3 | 5.8282 |

## All cases: share of kernel time per category

| Case | kernel us / step | launches / step | conv bwd | conv fwd | GEMM | layer 0 | product | gates | POLAR k | POLAR dense | rest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mh1_nacl216 | 2,326 | 70 | 3.0% | 2.3% | 83.5% | 1.4% | 3.3% | 5.5% | - | - | 1.0% |
| mh1_srtio3 | 37,575 | 70 | 8.2% | 4.7% | 60.8% | 2.3% | 7.3% | 16.4% | - | - | 0.3% |
| mp0m_nbcl5 | 961 | 40 | 16.9% | 12.0% | 43.4% | 10.7% | 12.9% | - | - | - | 4.1% |
| mp0m_srtio3 | 4,021 | 45 | 26.8% | 16.4% | 26.2% | 16.8% | 10.7% | - | - | - | 3.1% |
| mptrj_srtio3 | 11,627 | 50 | 21.2% | 15.9% | 48.1% | 3.6% | 10.6% | - | - | - | 0.7% |
| off23l_water216 | 3,654 | 57 | 29.0% | 20.0% | 44.3% | 2.0% | 3.2% | - | - | - | 1.6% |
| off23m_water216 | 1,186 | 51 | 23.1% | 17.4% | 44.8% | 6.0% | 4.5% | - | - | - | 4.2% |
| off23s_water216 | 521 | 49 | 11.9% | 9.0% | 55.5% | 9.4% | 5.5% | - | - | - | 8.8% |
| off24m_water | 9,856 | 53 | 36.0% | 25.0% | 22.3% | 7.6% | 6.6% | - | - | - | 2.4% |
| omat0m_gen_srtio3 | 8,530 | 59 | 35.0% | 19.8% | 28.5% | 8.5% | 6.9% | - | - | - | 1.3% |
| omat0m_spec_srtio3 | 8,648 | 59 | 34.4% | 20.6% | 27.5% | 9.4% | 6.5% | - | - | - | 1.5% |
| omat0s_srtio3 | 4,950 | 57 | 22.7% | 14.9% | 34.5% | 14.8% | 11.0% | - | - | - | 2.1% |
| omol0xl_water216 | 28,474 | 140 | 7.2% | 4.5% | 74.1% | 0.9% | 4.3% | 8.7% | - | - | 0.3% |
| polar1l_nacl216 | 10,134 | 397 | 2.9% | 2.0% | 86.4% | 0.3% | 1.7% | 2.2% | 1.8% | 2.3% | 0.4% |
| polar1m_nacl216 | 4,761 | 300 | 1.5% | 1.2% | 83.4% | 0.7% | 1.7% | 2.8% | 3.8% | 4.2% | 0.7% |
| polar1m_nbcl5 | 26,400 | 298 | 2.0% | 1.3% | 67.6% | 0.6% | 4.5% | 9.0% | 7.2% | 7.5% | 0.4% |
| polar1s_nacl216 | 1,599 | 192 | - | - | 76.5% | 2.2% | 1.5% | 3.9% | 6.9% | 7.1% | 1.8% |
| polar1s_nbcl5 | 8,944 | 185 | - | - | 64.8% | 1.6% | 6.0% | 12.1% | 9.3% | 5.4% | 0.8% |

## mh1_nacl216

Kernel time 2,326 us per step (summed over streams), 70 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 1,944 | 83.5% |
| nonlinear gates | 128 | 5.5% |
| symmetric contraction / product | 78 | 3.3% |
| conv backward (interaction >= 1) | 69 | 3.0% |
| conv forward (interaction >= 1) | 54 | 2.3% |
| layer 0 / interaction 0 | 32 | 1.4% |
| neighbour list / edges | 8 | 0.3% |
| other | 7 | 0.3% |
| forces / integrator / guards | 7 | 0.3% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm64_kernel<` | 1,549.9 | 66.6% | 30.2 |
| `sgemm32_kernel<` | 244.6 | 10.5% | 10.1 |
| `sgemm128_kernel<` | 138.5 | 6.0% | 2.0 |
| `gate_bwd_kernel` | 101.2 | 4.3% | 2.0 |
| `gconv_bwd_L1<` | 69.3 | 3.0% | 1.0 |
| `gconv_fwd_L1<` | 53.7 | 2.3% | 1.0 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 32.5 | 1.4% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 31.5 | 1.4% | 1.0 |
| `gate_fwd_kernel` | 26.5 | 1.1% | 2.0 |
| `first_bwd_edge_kernel<` | 23.6 | 1.0% | 1.0 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 13.6 | 0.6% | 1.0 |
| `splitk_reduce_kernel` | 10.7 | 0.5% | 5.0 |
| `first_fwd_kernel` | 8.4 | 0.4% | 1.0 |
| `langevin_first_half_kernel` | 3.5 | 0.1% | 1.0 |

## mh1_srtio3

Kernel time 37,575 us per step (summed over streams), 70 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 22,832 | 60.8% |
| nonlinear gates | 6,157 | 16.4% |
| conv backward (interaction >= 1) | 3,100 | 8.2% |
| symmetric contraction / product | 2,729 | 7.3% |
| conv forward (interaction >= 1) | 1,760 | 4.7% |
| layer 0 / interaction 0 | 880 | 2.3% |
| neighbour list / edges | 88 | 0.2% |
| other | 15 | 0.0% |
| forces / integrator / guards | 14 | 0.0% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 13,755.9 | 36.6% | 27.6 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>` | 7,226.2 | 19.2% | 9.6 |
| `gate_bwd_kernel` | 3,881.1 | 10.3% | 2.1 |
| `gconv_bwd_L1<` | 3,099.9 | 8.2% | 1.1 |
| `gate_fwd_kernel` | 2,275.5 | 6.1% | 2.1 |
| `gconv_fwd_L1<` | 1,759.8 | 4.7% | 1.1 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 1,153.3 | 3.1% | 1.1 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 994.8 | 2.6% | 1.1 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 756.3 | 2.0% | 4.2 |
| `magma_sgemmEx_kernel<float, float, float,` | 605.6 | 1.6% | 1.1 |
| `first_bwd_edge_kernel<` | 588.4 | 1.6% | 1.1 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 580.9 | 1.5% | 1.1 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 405.7 | 1.1% | 1.1 |
| `first_fwd_kernel` | 292.0 | 0.8% | 1.1 |

## mp0m_nbcl5

Kernel time 961 us per step (summed over streams), 40 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 417 | 43.4% |
| conv backward (interaction >= 1) | 163 | 16.9% |
| symmetric contraction / product | 124 | 12.9% |
| conv forward (interaction >= 1) | 116 | 12.0% |
| layer 0 / interaction 0 | 103 | 10.7% |
| neighbour list / edges | 21 | 2.2% |
| forces / integrator / guards | 9 | 0.9% |
| other | 9 | 0.9% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 188.7 | 19.6% | 6.0 |
| `gconv_bwd_L1s<` | 162.9 | 16.9% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 133.8 | 13.9% | 5.0 |
| `gconv_fwd_L1s<` | 115.6 | 12.0% | 1.0 |
| `first_bwd_edge_kernel<` | 71.2 | 7.4% | 1.0 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 70.0 | 7.3% | 1.0 |
| `sgemm64_kernel<` | 47.3 | 4.9% | 4.0 |
| `sgemm32_kernel<` | 46.8 | 4.9% | 7.1 |
| `first_fwd_kernel` | 31.4 | 3.3% | 1.0 |
| `gprod_ro1s_fused_ae0a447b65f4fcd4<` | 28.4 | 3.0% | 1.0 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 25.8 | 2.7% | 1.0 |
| `compute_and_count_active_kernel` | 11.4 | 1.2% | 1.0 |
| `langevin_first_half_kernel` | 4.9 | 0.5% | 1.0 |
| `scatter_active_edges_kernel` | 4.8 | 0.5% | 1.0 |

## mp0m_srtio3

Kernel time 4,021 us per step (summed over streams), 45 launches per step.

| Category | us / step | share |
|---|---:|---:|
| conv backward (interaction >= 1) | 1,076 | 26.8% |
| GEMM | 1,054 | 26.2% |
| layer 0 / interaction 0 | 677 | 16.8% |
| conv forward (interaction >= 1) | 660 | 16.4% |
| symmetric contraction / product | 429 | 10.7% |
| neighbour list / edges | 97 | 2.4% |
| forces / integrator / guards | 14 | 0.4% |
| other | 14 | 0.3% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `gconv_bwd_L1s<` | 1,076.2 | 26.8% | 1.0 |
| `sgemm128_kernel<` | 790.5 | 19.7% | 10.2 |
| `gconv_fwd_L1s<` | 660.3 | 16.4% | 1.0 |
| `first_bwd_edge_kernel<` | 443.8 | 11.0% | 1.0 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 239.2 | 5.9% | 1.0 |
| `first_fwd_kernel` | 233.1 | 5.8% | 1.0 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 115.3 | 2.9% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 94.9 | 2.4% | 3.0 |
| `gprod_ro1s_fused_ae0a447b65f4fcd4<` | 74.3 | 1.8% | 1.0 |
| `fm_tf32x3_rr_64x64` | 67.7 | 1.7% | 2.0 |
| `compute_and_count_active_kernel` | 41.0 | 1.0% | 1.0 |
| `sgemm32_kernel<` | 36.8 | 0.9% | 5.1 |
| `scatter_active_edges_kernel` | 33.3 | 0.8% | 1.0 |
| `fm_tf32x3_rc_64x64` | 24.6 | 0.6% | 1.0 |

## mptrj_srtio3

Kernel time 11,627 us per step (summed over streams), 50 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 5,591 | 48.1% |
| conv backward (interaction >= 1) | 2,465 | 21.2% |
| conv forward (interaction >= 1) | 1,845 | 15.9% |
| symmetric contraction / product | 1,235 | 10.6% |
| layer 0 / interaction 0 | 416 | 3.6% |
| neighbour list / edges | 44 | 0.4% |
| other | 18 | 0.2% |
| forces / integrator / guards | 15 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 4,610.6 | 39.7% | 18.3 |
| `gconv_bwd_L1<` | 2,464.6 | 21.2% | 1.0 |
| `gconv_fwd_L1<` | 1,844.9 | 15.9% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>` | 575.2 | 4.9% | 1.0 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 515.1 | 4.4% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 460.5 | 4.0% | 1.0 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 259.0 | 2.2% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 257.5 | 2.2% | 3.0 |
| `first_bwd_edge_kernel<` | 229.2 | 2.0% | 1.0 |
| `first_fwd_kernel` | 186.8 | 1.6% | 1.0 |
| `sgemm32_kernel<` | 112.2 | 1.0% | 8.1 |
| `sgemm64_kernel<` | 35.2 | 0.3% | 2.0 |
| `compute_and_count_active_kernel` | 21.6 | 0.2% | 1.0 |
| `gather_rows_kernel` | 11.4 | 0.1% | 1.0 |

## off23l_water216

Kernel time 3,654 us per step (summed over streams), 57 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 1,617 | 44.3% |
| conv backward (interaction >= 1) | 1,061 | 29.0% |
| conv forward (interaction >= 1) | 730 | 20.0% |
| symmetric contraction / product | 118 | 3.2% |
| layer 0 / interaction 0 | 72 | 2.0% |
| neighbour list / edges | 43 | 1.2% |
| forces / integrator / guards | 8 | 0.2% |
| other | 6 | 0.2% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 843.8 | 23.1% | 8.1 |
| `gconv_bwd_L2<` | 572.8 | 15.7% | 1.0 |
| `gconv_bwd_L2lp<` | 488.0 | 13.4% | 1.0 |
| `gconv_fwd_L2a<` | 402.4 | 11.0% | 1.0 |
| `gconv_fwd_L2b<` | 327.2 | 9.0% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 289.7 | 7.9% | 3.0 |
| `sgemm32_kernel<` | 254.4 | 7.0% | 12.2 |
| `sgemm64_kernel<` | 118.0 | 3.2% | 6.1 |
| `fm_tf32x3_rc_64x32` | 64.3 | 1.8% | 2.0 |
| `first_bwd_edge_kernel<` | 54.0 | 1.5% | 1.0 |
| `gprod_bwd_00dc9085957f7969` | 48.0 | 1.3% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 39.2 | 1.1% | 1.0 |
| `gprod_fwd_00dc9085957f7969` | 30.4 | 0.8% | 1.0 |
| `fm_tf32x3_rc_64x64` | 25.1 | 0.7% | 1.0 |

## off23m_water216

Kernel time 1,186 us per step (summed over streams), 51 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 531 | 44.8% |
| conv backward (interaction >= 1) | 274 | 23.1% |
| conv forward (interaction >= 1) | 206 | 17.4% |
| layer 0 / interaction 0 | 71 | 6.0% |
| symmetric contraction / product | 54 | 4.5% |
| neighbour list / edges | 36 | 3.1% |
| forces / integrator / guards | 8 | 0.7% |
| other | 6 | 0.5% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `gconv_bwd_L1<` | 273.7 | 23.1% | 1.0 |
| `gconv_fwd_L1<` | 205.9 | 17.4% | 1.0 |
| `sgemm64_kernel<` | 199.3 | 16.8% | 13.1 |
| `sgemm128_kernel<` | 138.5 | 11.7% | 4.0 |
| `first_bwd_edge_kernel<` | 53.2 | 4.5% | 1.0 |
| `fm_tf32x3_rr_64x64` | 48.0 | 4.0% | 4.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 47.5 | 4.0% | 1.0 |
| `sgemm32_kernel<` | 41.3 | 3.5% | 6.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 34.4 | 2.9% | 2.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 22.1 | 1.9% | 1.0 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 22.0 | 1.9% | 1.0 |
| `fm_tf32x3_rc_64x64` | 18.6 | 1.6% | 1.0 |
| `first_fwd_kernel` | 18.0 | 1.5% | 1.0 |
| `fill_neighbors_kernel` | 17.0 | 1.4% | 0.2 |

## off23s_water216

Kernel time 521 us per step (summed over streams), 49 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 289 | 55.5% |
| conv backward (interaction >= 1) | 62 | 11.9% |
| layer 0 / interaction 0 | 49 | 9.4% |
| conv forward (interaction >= 1) | 47 | 9.0% |
| neighbour list / edges | 33 | 6.3% |
| symmetric contraction / product | 28 | 5.5% |
| forces / integrator / guards | 7 | 1.4% |
| other | 5 | 1.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm32_kernel<` | 84.2 | 16.2% | 12.0 |
| `gconv_bwd_L0<` | 62.0 | 11.9% | 1.0 |
| `sgemm96_kernel` | 50.0 | 9.6% | 4.0 |
| `gconv_fwd_L0<` | 46.9 | 9.0% | 1.0 |
| `sgemm64_kernel<` | 35.6 | 6.8% | 3.0 |
| `first_bwd_edge_kernel<` | 34.5 | 6.6% | 1.0 |
| `magma_sgemmEx_kernel<float, float, float,` | 25.5 | 4.9% | 2.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>` | 21.0 | 4.0% | 2.0 |
| `fm_tf32x3_rr_64x32` | 19.2 | 3.7% | 2.0 |
| `fill_neighbors_kernel` | 16.0 | 3.1% | 0.2 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 14.9 | 2.9% | 1.0 |
| `fm_tf32x3_rc_64x64` | 14.6 | 2.8% | 1.0 |
| `first_fwd_kernel` | 14.4 | 2.8% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 14.0 | 2.7% | 1.0 |

## off24m_water

Kernel time 9,856 us per step (summed over streams), 53 launches per step.

| Category | us / step | share |
|---|---:|---:|
| conv backward (interaction >= 1) | 3,550 | 36.0% |
| conv forward (interaction >= 1) | 2,463 | 25.0% |
| GEMM | 2,199 | 22.3% |
| layer 0 / interaction 0 | 753 | 7.6% |
| symmetric contraction / product | 650 | 6.6% |
| neighbour list / edges | 220 | 2.2% |
| forces / integrator / guards | 15 | 0.1% |
| other | 7 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `gconv_bwd_L1<` | 3,549.8 | 36.0% | 1.0 |
| `gconv_fwd_L1<` | 2,463.4 | 25.0% | 1.0 |
| `sgemm128_kernel<` | 1,534.0 | 15.6% | 16.5 |
| `first_bwd_edge_kernel<` | 535.6 | 5.4% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 302.9 | 3.1% | 3.1 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 260.9 | 2.6% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 241.7 | 2.5% | 1.0 |
| `first_fwd_kernel` | 217.2 | 2.2% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 169.7 | 1.7% | 4.1 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 147.2 | 1.5% | 1.0 |
| `fm_tf32x3_rc_64x64` | 121.2 | 1.2% | 4.1 |
| `fill_neighbors_kernel` | 77.2 | 0.8% | 0.3 |
| `sgemm32_kernel<` | 56.4 | 0.6% | 4.1 |
| `compute_and_count_active_kernel` | 43.3 | 0.4% | 1.0 |

## omat0m_gen_srtio3

Kernel time 8,530 us per step (summed over streams), 59 launches per step.

| Category | us / step | share |
|---|---:|---:|
| conv backward (interaction >= 1) | 2,986 | 35.0% |
| GEMM | 2,429 | 28.5% |
| conv forward (interaction >= 1) | 1,693 | 19.8% |
| layer 0 / interaction 0 | 721 | 8.5% |
| symmetric contraction / product | 589 | 6.9% |
| neighbour list / edges | 91 | 1.1% |
| forces / integrator / guards | 17 | 0.2% |
| other | 5 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `gconv_bwd_L1<` | 2,985.7 | 35.0% | 1.0 |
| `sgemm128_kernel<` | 1,861.3 | 21.8% | 28.4 |
| `gconv_fwd_L1<` | 1,692.9 | 19.8% | 1.0 |
| `first_bwd_edge_kernel<` | 479.7 | 5.6% | 1.0 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 252.7 | 3.0% | 1.0 |
| `first_fwd_kernel` | 241.5 | 2.8% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 227.6 | 2.7% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 188.6 | 2.2% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 147.2 | 1.7% | 3.0 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 109.0 | 1.3% | 1.0 |
| `fm_tf32x3_rr_64x64` | 108.7 | 1.3% | 2.0 |
| `fm_tf32x3_rr_32x64` | 58.3 | 0.7% | 2.0 |
| `compute_and_count_active_kernel` | 41.1 | 0.5% | 1.0 |
| `fm_tf32x3_rc_64x64` | 29.7 | 0.3% | 1.0 |

## omat0m_spec_srtio3

Kernel time 8,648 us per step (summed over streams), 59 launches per step.

| Category | us / step | share |
|---|---:|---:|
| conv backward (interaction >= 1) | 2,978 | 34.4% |
| GEMM | 2,380 | 27.5% |
| conv forward (interaction >= 1) | 1,785 | 20.6% |
| layer 0 / interaction 0 | 812 | 9.4% |
| symmetric contraction / product | 560 | 6.5% |
| neighbour list / edges | 114 | 1.3% |
| forces / integrator / guards | 15 | 0.2% |
| other | 5 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `wprime_backward_inter1_rcv` | 2,978.2 | 34.4% | 1.0 |
| `spline_forward_inter1<` | 1,784.6 | 20.6% | 1.0 |
| `sgemm128_kernel<` | 1,726.0 | 20.0% | 25.4 |
| `tabled_outer_sum0_backward_kernel` | 555.2 | 6.4% | 1.0 |
| `tabled_outer_sum0_forward_kernel` | 256.8 | 3.0% | 1.0 |
| `fm_tf32x3_rr_64x64` | 239.2 | 2.8% | 6.1 |
| `sc0_backward` | 230.1 | 2.7% | 1.0 |
| `sc1_fused` | 220.7 | 2.6% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 219.8 | 2.5% | 1.0 |
| `sc0_forward` | 108.8 | 1.3% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 90.1 | 1.0% | 2.0 |
| `fm_tf32x3_rr_64x32` | 59.7 | 0.7% | 3.0 |
| `compute_and_count_active_kernel` | 47.3 | 0.5% | 1.0 |
| `scatter_active_edges_kernel` | 44.0 | 0.5% | 1.0 |

## omat0s_srtio3

Kernel time 4,950 us per step (summed over streams), 57 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 1,707 | 34.5% |
| conv backward (interaction >= 1) | 1,125 | 22.7% |
| conv forward (interaction >= 1) | 737 | 14.9% |
| layer 0 / interaction 0 | 733 | 14.8% |
| symmetric contraction / product | 543 | 11.0% |
| neighbour list / edges | 86 | 1.7% |
| forces / integrator / guards | 14 | 0.3% |
| other | 5 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 1,317.8 | 26.6% | 26.4 |
| `gconv_bwd_L0<` | 1,124.8 | 22.7% | 1.0 |
| `gconv_fwd_L0<` | 737.2 | 14.9% | 1.0 |
| `first_bwd_edge_kernel<` | 492.4 | 9.9% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 242.2 | 4.9% | 1.0 |
| `first_fwd_kernel` | 240.4 | 4.9% | 1.0 |
| `gprod_bwd_ae0a447b65f4fcd4` | 207.8 | 4.2% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 130.0 | 2.6% | 3.0 |
| `fm_tf32x3_rr_64x64` | 110.8 | 2.2% | 2.0 |
| `gprod_fwd_ae0a447b65f4fcd4` | 92.8 | 1.9% | 1.0 |
| `sgemm64_kernel<` | 60.7 | 1.2% | 2.0 |
| `fm_tf32x3_rc_64x64` | 59.3 | 1.2% | 2.0 |
| `compute_and_count_active_kernel` | 40.6 | 0.8% | 1.0 |
| `scatter_active_edges_kernel` | 22.7 | 0.5% | 1.0 |

## omol0xl_water216

Kernel time 28,474 us per step (summed over streams), 140 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 21,107 | 74.1% |
| nonlinear gates | 2,474 | 8.7% |
| conv backward (interaction >= 1) | 2,058 | 7.2% |
| conv forward (interaction >= 1) | 1,269 | 4.5% |
| symmetric contraction / product | 1,224 | 4.3% |
| layer 0 / interaction 0 | 249 | 0.9% |
| neighbour list / edges | 46 | 0.2% |
| forces / integrator / guards | 28 | 0.1% |
| other | 21 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 13,664.3 | 48.0% | 45.4 |
| `sgemm64_kernel<` | 3,174.1 | 11.1% | 22.7 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>` | 2,587.4 | 9.1% | 6.2 |
| `gate_bwd_kernel` | 1,594.8 | 5.6% | 3.1 |
| `gconv_bwd_L2<` | 1,095.3 | 3.8% | 2.1 |
| `gconv_bwd_L2lp<` | 962.8 | 3.4% | 2.1 |
| `sgemm32_kernel<` | 952.6 | 3.3% | 10.3 |
| `gate_fwd_kernel` | 878.7 | 3.1% | 3.1 |
| `gconv_fwd_L2a<` | 694.4 | 2.4% | 2.1 |
| `gconv_fwd_L2b<` | 574.7 | 2.0% | 2.1 |
| `gprod_bwd_6f194247389dfa5c` | 563.1 | 2.0% | 2.1 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 495.8 | 1.7% | 3.1 |
| `gprod_fwd_6f194247389dfa5c` | 332.2 | 1.2% | 2.1 |
| `gprod_bwd_4715eda4cc3ebda5` | 236.8 | 0.8% | 1.0 |

## polar1l_nacl216

Kernel time 10,134 us per step (summed over streams), 397 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 8,758 | 86.4% |
| conv backward (interaction >= 1) | 299 | 2.9% |
| POLAR dense / elementwise | 229 | 2.3% |
| nonlinear gates | 227 | 2.2% |
| conv forward (interaction >= 1) | 200 | 2.0% |
| POLAR k-space | 180 | 1.8% |
| symmetric contraction / product | 171 | 1.7% |
| layer 0 / interaction 0 | 33 | 0.3% |
| other | 21 | 0.2% |
| neighbour list / edges | 9 | 0.1% |
| forces / integrator / guards | 7 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm64_kernel<` | 5,506.3 | 54.3% | 100.5 |
| `sgemm128_kernel<` | 1,905.5 | 18.8% | 30.5 |
| `sgemm32_kernel<` | 728.7 | 7.2% | 53.8 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 205.9 | 2.0% | 12.2 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>` | 195.2 | 1.9% | 14.2 |
| `gate_bwd_kernel` | 191.5 | 1.9% | 3.0 |
| `gconv_bwd_L2<` | 165.0 | 1.6% | 2.0 |
| `gconv_bwd_L2lp<` | 133.8 | 1.3% | 2.0 |
| `gprod_bwd_00dc9085957f7969` | 123.6 | 1.2% | 3.0 |
| `gconv_fwd_L2a<` | 107.8 | 1.1% | 2.0 |
| `gconv_fwd_L2b<` | 91.7 | 0.9% | 2.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>` | 81.2 | 0.8% | 3.0 |
| `sf_kernel_tiled<` | 57.2 | 0.6% | 5.1 |
| `inv_bwd_kernel` | 56.8 | 0.6% | 10.2 |

## polar1m_nacl216

Kernel time 4,761 us per step (summed over streams), 300 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 3,971 | 83.4% |
| POLAR dense / elementwise | 200 | 4.2% |
| POLAR k-space | 183 | 3.8% |
| nonlinear gates | 134 | 2.8% |
| symmetric contraction / product | 81 | 1.7% |
| conv backward (interaction >= 1) | 71 | 1.5% |
| conv forward (interaction >= 1) | 58 | 1.2% |
| layer 0 / interaction 0 | 32 | 0.7% |
| other | 16 | 0.3% |
| neighbour list / edges | 8 | 0.2% |
| forces / integrator / guards | 7 | 0.2% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm64_kernel<` | 2,656.2 | 55.8% | 61.5 |
| `sgemm128_kernel<` | 473.0 | 9.9% | 8.1 |
| `sgemm32_kernel<` | 401.1 | 8.4% | 43.3 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>` | 208.3 | 4.4% | 15.1 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 129.7 | 2.7% | 12.1 |
| `gate_bwd_kernel` | 109.0 | 2.3% | 2.0 |
| `gconv_bwd_L1<` | 70.5 | 1.5% | 1.0 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 61.1 | 1.3% | 2.0 |
| `sf_kernel_tiled<` | 58.0 | 1.2% | 5.0 |
| `gconv_fwd_L1<` | 58.0 | 1.2% | 1.0 |
| `grad_pass_kernel<` | 53.3 | 1.1% | 3.0 |
| `cublasLt::splitKreduce_kernel<` | 33.5 | 0.7% | 12.1 |
| `splitk_reduce_kernel` | 31.8 | 0.7% | 19.1 |
| `inv_bwd_kernel` | 30.8 | 0.6% | 8.1 |

## polar1m_nbcl5

Kernel time 26,400 us per step (summed over streams), 298 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 17,838 | 67.6% |
| nonlinear gates | 2,377 | 9.0% |
| POLAR dense / elementwise | 1,977 | 7.5% |
| POLAR k-space | 1,904 | 7.2% |
| symmetric contraction / product | 1,181 | 4.5% |
| conv backward (interaction >= 1) | 522 | 2.0% |
| conv forward (interaction >= 1) | 346 | 1.3% |
| layer 0 / interaction 0 | 155 | 0.6% |
| other | 67 | 0.3% |
| neighbour list / edges | 25 | 0.1% |
| forces / integrator / guards | 9 | 0.0% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 8,925.7 | 33.8% | 56.3 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>` | 5,559.8 | 21.1% | 32.9 |
| `gate_bwd_kernel` | 1,472.0 | 5.6% | 2.1 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 1,060.4 | 4.0% | 3.2 |
| `sf_kernel_tiled<` | 1,047.2 | 4.0% | 5.3 |
| `gate_fwd_kernel` | 904.8 | 3.4% | 2.1 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 813.3 | 3.1% | 2.1 |
| `inv_bwd_kernel` | 803.5 | 3.0% | 8.5 |
| `sgemm64_kernel<` | 765.1 | 2.9% | 21.2 |
| `grad_pass_kernel<` | 548.8 | 2.1% | 3.2 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>` | 523.9 | 2.0% | 12.8 |
| `gconv_bwd_L1<` | 521.6 | 2.0% | 1.1 |
| `sgemm32_kernel<` | 370.4 | 1.4% | 8.5 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 368.1 | 1.4% | 2.1 |

## polar1s_nacl216

Kernel time 1,599 us per step (summed over streams), 192 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 1,224 | 76.5% |
| POLAR dense / elementwise | 114 | 7.1% |
| POLAR k-space | 110 | 6.9% |
| nonlinear gates | 63 | 3.9% |
| layer 0 / interaction 0 | 36 | 2.2% |
| symmetric contraction / product | 24 | 1.5% |
| other | 12 | 0.8% |
| forces / integrator / guards | 8 | 0.5% |
| neighbour list / edges | 8 | 0.5% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm64_kernel<` | 689.8 | 43.1% | 20.2 |
| `sgemm32_kernel<` | 226.0 | 14.1% | 34.3 |
| `sgemm128_kernel<` | 171.0 | 10.7% | 3.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 80.1 | 5.0% | 1.0 |
| `gate_bwd_kernel` | 49.3 | 3.1% | 1.0 |
| `grad_pass_kernel<` | 27.9 | 1.7% | 3.0 |
| `first_bwd_edge_kernel<` | 27.5 | 1.7% | 1.0 |
| `sf_kernel_tiled<` | 23.2 | 1.5% | 5.0 |
| `bias_ln_silu_fwd_kernel` | 22.6 | 1.4% | 9.1 |
| `feat_coef_kernel` | 18.5 | 1.2% | 2.0 |
| `gprod_bwd_ae0a447b65f4fcd4` | 18.1 | 1.1% | 1.0 |
| `bwd_coef_kernel` | 17.1 | 1.1% | 3.0 |
| `splitk_reduce_kernel` | 14.6 | 0.9% | 12.1 |
| `gate_fwd_kernel` | 13.7 | 0.9% | 1.0 |

## polar1s_nbcl5

Kernel time 8,944 us per step (summed over streams), 185 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 5,791 | 64.8% |
| nonlinear gates | 1,079 | 12.1% |
| POLAR k-space | 836 | 9.3% |
| symmetric contraction / product | 540 | 6.0% |
| POLAR dense / elementwise | 483 | 5.4% |
| layer 0 / interaction 0 | 142 | 1.6% |
| other | 37 | 0.4% |
| neighbour list / edges | 24 | 0.3% |
| forces / integrator / guards | 11 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 2,965.5 | 33.2% | 21.7 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>` | 1,329.0 | 14.9% | 12.4 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 757.4 | 8.5% | 5.2 |
| `gate_bwd_kernel` | 647.1 | 7.2% | 1.0 |
| `gate_fwd_kernel` | 431.7 | 4.8% | 1.0 |
| `gprod_bwd_ae0a447b65f4fcd4` | 384.3 | 4.3% | 1.0 |
| `sf_kernel_tiled<` | 358.5 | 4.0% | 5.2 |
| `sgemm32_kernel<` | 332.0 | 3.7% | 12.4 |
| `sgemm64_kernel<` | 314.8 | 3.5% | 10.3 |
| `grad_pass_kernel<` | 295.7 | 3.3% | 3.1 |
| `gprod_fwd_ae0a447b65f4fcd4` | 156.1 | 1.7% | 1.0 |
| `feat_pass_kernel<` | 117.2 | 1.3% | 2.1 |
| `inv_bwd_kernel` | 100.2 | 1.1% | 4.1 |
| `first_bwd_edge_kernel<` | 93.6 | 1.0% | 1.0 |

# ncu, one force call (top kernels, sections + raw metrics)

## mh1_srtio3

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sgemm128_kernel<8, 1, 1>` | 23 | 11,443.8 | 71 | 24 | 30 | 32 | 128 | 71 | 0 | not_selected 2.39, selected 1.00, dispatch_stall 0.94 |
| `gate_bwd_kernel` | 2 | 3,579.9 | 9 | 19 | 87 | 53 | 48 | 3 | 6 | long_scoreboard 83.82, wait 1.60, selected 1.00 |
| `gconv_bwd_L1<1, 128, 1>` | 1 | 2,935.4 | 76 | 82 | 23 | 64 | 62 | 94 | 11 | not_selected 3.95, dispatch_stall 1.38, long_scoreboard 1.29 |
| `gate_fwd_kernel` | 2 | 1,958.2 | 9 | 23 | 88 | 83 | 40 | 0 | 5 | long_scoreboard 128.81, lg_throttle 1.95, wait 1.65 |
| `sgemm128_kernel<8, 0, 1>` | 4 | 1,910.8 | 68 | 18 | 29 | 31 | 128 | 67 | 1 | not_selected 2.31, selected 1.00, dispatch_stall 0.87 |
| `gconv_fwd_L1<1, 1, 256, 1, 128>` | 1 | 1,899.8 | 51 | 82 | 17 | 33 | 102 | 95 | 5 | long_scoreboard 3.55, selected 1.00, not_selected 0.83 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<512, 1>` | 1 | 944.4 | 18 | 20 | 82 | 39 | 110 | 17 | 79 | lg_throttle 4.92, barrier 4.77, long_scoreboard 4.51 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 1 | 901.6 | 32 | 20 | 88 | 26 | 160 | 2 | 44 | long_scoreboard 7.02, selected 1.00, not_selected 0.65 |
| `first_bwd_edge_kernel<3, 1, 0>` | 1 | 575.4 | 59 | 39 | 32 | 32 | 119 | 84 | 19 | long_scoreboard 2.14, not_selected 1.17, selected 1.00 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 1 | 487.5 | 34 | 29 | 93 | 55 | 72 | 2 | 59 | long_scoreboard 12.16, barrier 2.60, wait 1.25 |
| `magma_sgemmEx_kernel<float, float, float, 1, 0, 6, 4, 6, 3, 4>` | 1 | 305.7 | 55 | 26 | 51 | 30 | 80 | 70 | 10 | not_selected 1.69, selected 1.00, long_scoreboard 0.60 |
| `first_fwd_kernel` | 1 | 301.4 | 72 | 54 | 56 | 72 | 52 | 86 | 33 | long_scoreboard 3.02, barrier 3.00, short_scoreboard 2.12 |
| `sgemm32_kernel<0>` | 1 | 37.6 | 26 | 24 | 65 | 26 | 77 | 31 | 4 | long_scoreboard 8.25, selected 1.00, short_scoreboard 0.60 |

## mp0m_srtio3

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gconv_bwd_L1s<1, 128, 1>` | 1 | 1,724.2 | 50 | 89 | 21 | 64 | 64 | 98 | 12 | long_scoreboard 14.00, wait 1.34, selected 1.00 |
| `gconv_fwd_L1s<1, 1, 256, 1, 128>` | 1 | 756.8 | 47 | 83 | 16 | 98 | 40 | 97 | 1 | long_scoreboard 20.06, mio_throttle 5.23, short_scoreboard 2.47 |
| `first_bwd_edge_kernel<2, 0, 0>` | 1 | 465.9 | 74 | 51 | 40 | 47 | 80 | 84 | 47 | long_scoreboard 2.19, short_scoreboard 1.93, not_selected 1.55 |
| `sgemm128_kernel<8, 1, 1>` | 8 | 459.5 | 58 | 23 | 55 | 30 | 128 | 36 | 1 | not_selected 2.05, selected 1.00, dispatch_stall 0.84 |
| `sgemm128_kernel<4, 1, 1>` | 4 | 333.9 | 66 | 35 | 49 | 32 | 105 | 56 | 0 | not_selected 1.86, selected 1.00, dispatch_stall 0.69 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 1 | 243.4 | 30 | 19 | 86 | 24 | 160 | 3 | 44 | long_scoreboard 4.61, selected 1.00, not_selected 0.78 |
| `first_fwd_kernel` | 1 | 207.2 | 71 | 83 | 66 | 71 | 52 | 79 | 22 | long_scoreboard 5.76, short_scoreboard 2.00, barrier 1.40 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 1 | 109.1 | 39 | 34 | 92 | 56 | 72 | 4 | 59 | long_scoreboard 10.55, short_scoreboard 1.36, wait 1.27 |
| `gprod_ro1s_fused_ae0a447b65f4fcd4<128, 4>` | 1 | 78.1 | 30 | 6 | 21 | 79 | 48 | 31 | 98 | barrier 19.74, mio_throttle 18.77, lg_throttle 14.34 |
| `fm_tf32x3_rr_32x64` | 1 | 32.1 | 72 | 39 | 34 | 15 | 128 | 75 | 0 | math_pipe_throttle 3.19, selected 1.00, wait 0.76 |
| `sgemm32_kernel<0>` | 5 | 29.0 | 12 | 15 | 36 | 6 | 77 | 30 | 6 | long_scoreboard 1.62, selected 1.00, short_scoreboard 0.54 |
| `fm_tf32x3_rc_64x64` | 1 | 22.7 | 68 | 31 | 37 | 21 | 132 | 58 | 0 | math_pipe_throttle 4.45, selected 1.00, long_scoreboard 0.92 |
| `fm_tf32x3_rr_64x32` | 1 | 8.1 | 28 | 18 | 56 | 6 | 128 | 36 | 0 | math_pipe_throttle 1.27, selected 1.00, long_scoreboard 0.92 |

## off23l_water216

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gconv_bwd_L2<1, 224, 1>` | 1 | 592.8 | 67 | 58 | 12 | 23 | 126 | 96 | 3 | not_selected 1.20, selected 1.00, dispatch_stall 0.54 |
| `sgemm128_kernel<8, 0, 1>` | 7 | 527.6 | 66 | 21 | 25 | 26 | 128 | 64 | 0 | not_selected 1.76, selected 1.00, dispatch_stall 0.72 |
| `gconv_bwd_L2lp<1, 224, 1>` | 1 | 519.4 | 70 | 80 | 19 | 57 | 62 | 96 | 4 | not_selected 3.38, long_scoreboard 1.62, dispatch_stall 1.20 |
| `gconv_fwd_L2a<1, 1, 256, 1, 224>` | 1 | 408.2 | 44 | 78 | 15 | 29 | 102 | 92 | 2 | long_scoreboard 3.49, selected 1.00, not_selected 0.81 |
| `gconv_fwd_L2b<1, 1, 256, 1, 224>` | 1 | 354.1 | 49 | 73 | 26 | 28 | 103 | 92 | 3 | long_scoreboard 2.39, not_selected 1.11, selected 1.00 |
| `sgemm128_kernel<8, 1, 1>` | 1 | 138.8 | 71 | 24 | 14 | 31 | 128 | 71 | 0 | not_selected 2.28, selected 1.00, dispatch_stall 0.91 |
| `sgemm32_kernel<1>` | 11 | 111.8 | 35 | 37 | 22 | 17 | 77 | 79 | 1 | selected 1.00, short_scoreboard 0.93, long_scoreboard 0.66 |
| `first_bwd_edge_kernel<2, 0, 0>` | 1 | 64.7 | 49 | 35 | 24 | 39 | 80 | 78 | 47 | long_scoreboard 3.47, short_scoreboard 2.03, not_selected 1.37 |
| `sgemm64_kernel<0>` | 5 | 58.9 | 47 | 28 | 19 | 20 | 101 | 71 | 0 | selected 1.00, not_selected 0.94, dispatch_stall 0.48 |
| `gprod_bwd_00dc9085957f7969` | 1 | 51.0 | 49 | 25 | 72 | 20 | 222 | 12 | 49 | long_scoreboard 1.33, selected 1.00, not_selected 0.54 |
| `fm_tf32x3_rc_64x32` | 2 | 46.3 | 74 | 38 | 22 | 13 | 136 | 80 | 0 | math_pipe_throttle 2.99, selected 1.00, wait 0.84 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<256, 2>` | 1 | 45.8 | 21 | 24 | 63 | 27 | 110 | 10 | 81 | long_scoreboard 6.25, barrier 4.20, lg_throttle 1.92 |
| `gprod_fwd_00dc9085957f7969` | 1 | 45.2 | 32 | 22 | 55 | 31 | 128 | 15 | 60 | long_scoreboard 6.73, short_scoreboard 1.49, wait 1.05 |
| `fm_tf32x3_rc_64x64` | 1 | 22.9 | 71 | 27 | 31 | 18 | 132 | 68 | 0 | math_pipe_throttle 4.30, selected 1.00, wait 0.77 |
| `first_fwd_kernel` | 1 | 21.4 | 60 | 65 | 24 | 67 | 52 | 76 | 32 | long_scoreboard 6.49, short_scoreboard 2.12, mio_throttle 1.54 |
| `sgemm64_kernel<1>` | 1 | 13.6 | 50 | 33 | 35 | 16 | 101 | 58 | 0 | selected 1.00, not_selected 0.59, short_scoreboard 0.39 |

## omat0m_gen_srtio3

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gconv_bwd_L1<1, 128, 1>` | 1 | 2,750.2 | 76 | 81 | 20 | 64 | 62 | 95 | 11 | not_selected 3.98, dispatch_stall 1.38, long_scoreboard 1.30 |
| `gconv_fwd_L1<1, 1, 256, 1, 128>` | 1 | 1,860.4 | 52 | 82 | 24 | 33 | 102 | 95 | 6 | long_scoreboard 3.48, selected 1.00, not_selected 0.84 |
| `sgemm128_kernel<8, 1, 1>` | 16 | 1,258.2 | 61 | 24 | 46 | 30 | 128 | 44 | 0 | not_selected 2.10, selected 1.00, dispatch_stall 0.86 |
| `first_bwd_edge_kernel<2, 1, 0>` | 1 | 483.2 | 71 | 43 | 40 | 47 | 80 | 84 | 52 | long_scoreboard 2.69, short_scoreboard 2.13, not_selected 1.34 |
| `sgemm128_kernel<4, 1, 1>` | 12 | 365.0 | 60 | 34 | 39 | 30 | 105 | 55 | 0 | not_selected 1.72, selected 1.00, dispatch_stall 0.65 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 1 | 253.6 | 29 | 24 | 84 | 21 | 160 | 4 | 44 | long_scoreboard 8.54, selected 1.00, not_selected 0.83 |
| `first_fwd_kernel` | 1 | 209.1 | 75 | 67 | 63 | 72 | 52 | 78 | 38 | long_scoreboard 3.71, barrier 2.27, short_scoreboard 2.15 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<128, 4>` | 1 | 192.7 | 24 | 24 | 80 | 32 | 110 | 10 | 81 | long_scoreboard 5.86, barrier 4.93, lg_throttle 1.02 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 1 | 120.7 | 35 | 34 | 93 | 57 | 72 | 3 | 59 | long_scoreboard 11.41, not_selected 1.55, wait 1.32 |
| `fm_tf32x3_rr_64x64` | 2 | 62.2 | 73 | 30 | 35 | 21 | 128 | 63 | 0 | math_pipe_throttle 4.26, selected 1.00, wait 0.76 |
| `fm_tf32x3_rc_64x64` | 1 | 23.1 | 66 | 30 | 37 | 22 | 132 | 57 | 0 | math_pipe_throttle 4.42, long_scoreboard 1.08, selected 1.00 |
| `sgemm64_kernel<1>` | 2 | 16.3 | 33 | 20 | 30 | 8 | 101 | 62 | 0 | selected 1.00, short_scoreboard 0.30, long_scoreboard 0.21 |
| `sgemm32_kernel<0>` | 2 | 8.9 | 11 | 11 | 28 | 4 | 77 | 34 | 4 | selected 1.00, long_scoreboard 0.53, short_scoreboard 0.48 |
| `fm_tf32x3_rr_64x32` | 1 | 7.8 | 29 | 19 | 46 | 5 | 128 | 35 | 0 | math_pipe_throttle 1.27, selected 1.00, wait 0.85 |

## omat0m_spec_srtio3

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `wprime_backward_inter1_rcv` | 1 | 2,735.8 | 75 | 81 | 18 | 64 | 64 | 94 | 12 | not_selected 3.90, long_scoreboard 1.58, dispatch_stall 1.43 |
| `spline_forward_inter1<1, 1>` | 1 | 1,771.7 | 58 | 85 | 22 | 41 | 96 | 95 | 8 | long_scoreboard 3.85, not_selected 1.16, selected 1.00 |
| `sgemm128_kernel<8, 1, 1>` | 16 | 1,259.7 | 61 | 24 | 49 | 30 | 128 | 44 | 0 | not_selected 2.10, selected 1.00, dispatch_stall 0.86 |
| `tabled_outer_sum0_backward_kernel` | 1 | 526.6 | 73 | 49 | 41 | 37 | 80 | 86 | 36 | long_scoreboard 2.58, wait 1.87, short_scoreboard 1.50 |
| `sgemm128_kernel<4, 1, 1>` | 12 | 365.9 | 60 | 37 | 45 | 30 | 105 | 54 | 0 | not_selected 1.72, selected 1.00, dispatch_stall 0.65 |
| `tabled_outer_sum0_forward_kernel` | 1 | 230.5 | 56 | 61 | 55 | 40 | 96 | 78 | 59 | long_scoreboard 5.34, short_scoreboard 1.50, wait 1.10 |
| `sc0_backward` | 1 | 213.7 | 31 | 22 | 88 | 35 | 128 | 40 | 63 | long_scoreboard 6.32, not_selected 1.04, selected 1.00 |
| `sc1_fused` | 1 | 191.8 | 24 | 25 | 82 | 29 | 110 | 46 | 88 | long_scoreboard 8.25, barrier 4.97, selected 1.00 |
| `sc0_forward` | 1 | 126.2 | 29 | 29 | 91 | 64 | 72 | 4 | 59 | long_scoreboard 12.42, short_scoreboard 1.06, wait 1.05 |
| `fm_tf32x3_rr_64x64` | 2 | 62.2 | 73 | 29 | 40 | 21 | 128 | 66 | 0 | math_pipe_throttle 4.27, selected 1.00, wait 0.76 |
| `fm_tf32x3_rc_64x32` | 1 | 23.5 | 66 | 37 | 26 | 14 | 136 | 73 | 0 | math_pipe_throttle 2.87, selected 1.00, wait 0.82 |
| `sgemm64_kernel<1>` | 2 | 16.2 | 34 | 22 | 33 | 8 | 101 | 59 | 0 | selected 1.00, short_scoreboard 0.31, long_scoreboard 0.20 |
| `sgemm32_kernel<0>` | 2 | 8.8 | 11 | 14 | 30 | 4 | 77 | 30 | 4 | selected 1.00, short_scoreboard 0.48, long_scoreboard 0.40 |
| `fm_tf32x3_rr_64x32` | 1 | 7.8 | 29 | 18 | 45 | 5 | 128 | 40 | 0 | math_pipe_throttle 1.27, selected 1.00, wait 0.85 |

## polar1m_nbcl5

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sgemm128_kernel<8, 1, 1>` | 49 | 7,814.9 | 63 | 22 | 27 | 30 | 128 | 71 | 3 | not_selected 2.18, selected 1.00, dispatch_stall 0.88 |
| `gconv_fwd_L1<1, 1, 256, 1, 128>` | 1 | 951.1 | 38 | 70 | 47 | 32 | 102 | 90 | 2 | long_scoreboard 4.24, selected 1.00, not_selected 0.84 |
| `sgemm64_kernel<1>` | 18 | 716.2 | 35 | 21 | 37 | 13 | 101 | 57 | 1 | selected 1.00, not_selected 0.38, short_scoreboard 0.36 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 2 | 677.7 | 33 | 23 | 88 | 23 | 160 | 3 | 43 | long_scoreboard 6.70, selected 1.00, not_selected 0.78 |
| `sf_kernel_tiled<16, 8, 16>` | 2 | 662.4 | 70 | 8 | 1 | 46 | 80 | 86 | 10 | short_scoreboard 9.17, wait 3.53, tex_throttle 2.29 |
| `inv_bwd_kernel` | 8 | 599.3 | 22 | 34 | 94 | 85 | 20 | 24 | 15 | long_scoreboard 42.67, wait 2.80, short_scoreboard 1.66 |
| `grad_pass_kernel<2, 1, 0>` | 2 | 536.8 | 34 | 5 | 1 | 17 | 80 | 69 | 45 | long_scoreboard 1.65, short_scoreboard 1.44, wait 1.13 |
| `gconv_bwd_L1<1, 128, 1>` | 1 | 469.3 | 70 | 82 | 31 | 58 | 62 | 90 | 3 | not_selected 2.96, long_scoreboard 2.04, dispatch_stall 1.08 |
| `sgemm128_kernel<4, 0, 1>` | 4 | 468.7 | 68 | 26 | 40 | 31 | 105 | 70 | 0 | not_selected 1.83, selected 1.00, dispatch_stall 0.70 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 2 | 363.9 | 35 | 31 | 95 | 56 | 72 | 3 | 58 | long_scoreboard 13.19, not_selected 1.16, selected 1.00 |
| `sf_kernel_tiled<8, 8, 8>` | 2 | 346.5 | 67 | 5 | 8 | 61 | 64 | 81 | 54 | short_scoreboard 6.80, tex_throttle 3.78, wait 3.04 |
| `inv_kernel` | 4 | 335.6 | 11 | 33 | 95 | 96 | 30 | 0 | 5 | long_scoreboard 86.44, wait 2.69, selected 1.00 |
| `feat_pass_kernel<0>` | 2 | 301.4 | 35 | 6 | 1 | 17 | 80 | 60 | 37 | long_scoreboard 1.85, wait 1.03, short_scoreboard 1.01 |
| `sgemm32_kernel<0>` | 6 | 164.8 | 22 | 22 | 59 | 19 | 77 | 22 | 1 | long_scoreboard 6.13, selected 1.00, short_scoreboard 0.58 |
| `ro_bwd_kernel` | 2 | 157.6 | 11 | 4 | 13 | 47 | 66 | 13 | 98 | mio_throttle 58.54, lg_throttle 46.28, short_scoreboard 11.85 |
| `sgemm64_kernel<0>` | 2 | 145.6 | 60 | 27 | 30 | 21 | 101 | 77 | 0 | not_selected 1.01, selected 1.00, dispatch_stall 0.51 |

