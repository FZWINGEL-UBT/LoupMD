# Kernel time per MD step (nsys)

| Model | System | Atoms | Steps | Rebuilds | ms / step |
|---|---|---:|---:|---:|---:|
| mace-omat-0-medium.model | srtio3 | 5000 | 64 | 4 | 8.0194 |
| mace-omat-0-medium.model | srtio3 | 5000 | 64 | 4 | 7.8856 |
| 2023-12-03-mace-128-L1_epoch-199.model | srtio3 | 5000 | 64 | 4 | 3.1858 |
| 2023-12-03-mace-128-L1_epoch-199.model | nbcl5 | 1944 | 128 | 5 | 0.6953 |
| MACE_MPtrj_2022.9.model | srtio3 | 5000 | 64 | 3 | 8.3988 |
| MACE_MPtrj_2022.9.model | nbcl5 | 1944 | 128 | 5 | 2.1097 |
| mace-omat-0-small.model | srtio3 | 5000 | 64 | 4 | 4.3751 |
| mace-mh-1.model | srtio3 | 5000 | 16 | 1 | 30.9072 |
| mace-mh-1.model | nacl216 | 216 | 128 | 5 | 1.3829 |
| MACE-OFF23_small.model | water216 | 648 | 256 | 62 | 0.3725 |
| MACE-OFF23_medium.model | water216 | 648 | 256 | 61 | 0.9348 |
| MACE-OFF24_medium.model | water | 5184 | 32 | 10 | 9.0639 |
| MACE-OFF23_large.model | water216 | 648 | 64 | 16 | 3.0309 |
| MACE-omol-0-extra-large-1024.model | water216 | 648 | 32 | 8 | 21.7300 |
| MACE-POLAR-1-S.model | nacl216 | 216 | 128 | 5 | 1.2532 |
| MACE-POLAR-1-S.model | nbcl5 | 1944 | 32 | 2 | 7.8729 |
| MACE-POLAR-1-M.model | nacl216 | 216 | 128 | 5 | 2.8628 |
| MACE-POLAR-1-M.model | nbcl5 | 1944 | 16 | 1 | 21.1296 |
| MACE-POLAR-1-L.model | nacl216 | 216 | 64 | 3 | 5.1848 |

## All cases: share of kernel time per category

| Case | kernel us / step | launches / step | conv bwd | conv fwd | GEMM | layer 0 | product | gates | POLAR k | POLAR dense | rest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mh1_nacl216 | 2,365 | 70 | 3.1% | 2.3% | 85.3% | 1.4% | 3.1% | 3.9% | - | - | 0.9% |
| mh1_srtio3 | 35,309 | 67 | 9.0% | 5.1% | 71.5% | 2.5% | 7.7% | 3.8% | - | - | 0.3% |
| mp0m_nbcl5 | 773 | 40 | 14.0% | 15.1% | 40.1% | 12.2% | 13.1% | - | - | - | 5.5% |
| mp0m_srtio3 | 3,367 | 45 | 23.9% | 19.9% | 25.7% | 18.4% | 8.7% | - | - | - | 3.4% |
| mptrj_nbcl5 | 2,524 | 46 | 18.5% | 14.1% | 51.7% | 2.4% | 12.0% | - | - | - | 1.3% |
| mptrj_srtio3 | 8,924 | 50 | 27.4% | 21.0% | 33.6% | 4.3% | 12.8% | - | - | - | 0.9% |
| off23l_water216 | 3,654 | 57 | 29.2% | 20.1% | 44.3% | 1.8% | 3.2% | - | - | - | 1.4% |
| off23m_water216 | 1,186 | 51 | 22.4% | 17.5% | 46.0% | 5.5% | 4.3% | - | - | - | 4.3% |
| off23s_water216 | 486 | 49 | 12.7% | 9.7% | 53.1% | 9.3% | 5.5% | - | - | - | 9.7% |
| off24m_water | 9,783 | 53 | 36.3% | 25.2% | 22.6% | 7.3% | 6.1% | - | - | - | 2.6% |
| omat0m_gen_srtio3 | 8,624 | 59 | 34.9% | 19.9% | 28.8% | 8.1% | 7.0% | - | - | - | 1.3% |
| omat0m_spec_srtio3 | 8,739 | 59 | 34.0% | 20.8% | 27.8% | 9.4% | 6.5% | - | - | - | 1.5% |
| omat0s_srtio3 | 4,861 | 57 | 23.5% | 15.3% | 33.6% | 14.3% | 11.2% | - | - | - | 2.2% |
| omol0xl_water216 | 27,604 | 140 | 7.6% | 4.8% | 79.0% | 1.0% | 4.2% | 3.1% | - | - | 0.3% |
| polar1l_nacl216 | 9,024 | 379 | 3.4% | 2.3% | 85.4% | 0.4% | 1.9% | 1.7% | 2.1% | 2.5% | 0.4% |
| polar1m_nacl216 | 4,682 | 294 | 1.5% | 1.2% | 84.4% | 0.7% | 1.7% | 1.9% | 3.7% | 4.2% | 0.7% |
| polar1m_nbcl5 | 25,091 | 294 | 1.9% | 1.4% | 71.8% | 0.6% | 4.8% | 3.5% | 7.8% | 7.9% | 0.3% |
| polar1s_nacl216 | 1,621 | 189 | - | - | 77.1% | 2.0% | 1.5% | 3.4% | 7.1% | 7.2% | 1.7% |
| polar1s_nbcl5 | 8,585 | 183 | - | - | 70.9% | 1.7% | 6.1% | 5.0% | 9.6% | 5.9% | 0.8% |

## mh1_nacl216

Kernel time 2,365 us per step (summed over streams), 70 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 2,017 | 85.3% |
| nonlinear gates | 91 | 3.9% |
| conv backward (interaction >= 1) | 74 | 3.1% |
| symmetric contraction / product | 72 | 3.1% |
| conv forward (interaction >= 1) | 54 | 2.3% |
| layer 0 / interaction 0 | 33 | 1.4% |
| neighbour list / edges | 8 | 0.3% |
| forces / integrator / guards | 7 | 0.3% |
| other | 7 | 0.3% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm64_kernel<` | 1,275.8 | 53.9% | 25.2 |
| `sgemm128_kernel<` | 475.9 | 20.1% | 7.1 |
| `sgemm32_kernel<` | 254.1 | 10.7% | 10.1 |
| `gconv_bwd_L1<` | 74.2 | 3.1% | 1.0 |
| `gate_bwd_kernel` | 66.4 | 2.8% | 2.0 |
| `gconv_fwd_L1<` | 54.1 | 2.3% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 31.5 | 1.3% | 1.0 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 27.4 | 1.2% | 1.0 |
| `gate_fwd_kernel` | 25.1 | 1.1% | 2.0 |
| `first_bwd_edge_kernel<` | 21.6 | 0.9% | 1.0 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 13.5 | 0.6% | 1.0 |
| `first_fwd_kernel` | 11.6 | 0.5% | 1.0 |
| `splitk_reduce_kernel` | 11.5 | 0.5% | 5.0 |
| `langevin_first_half_kernel` | 3.5 | 0.1% | 1.0 |

## mh1_srtio3

Kernel time 35,309 us per step (summed over streams), 67 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 25,259 | 71.5% |
| conv backward (interaction >= 1) | 3,169 | 9.0% |
| symmetric contraction / product | 2,712 | 7.7% |
| conv forward (interaction >= 1) | 1,801 | 5.1% |
| nonlinear gates | 1,348 | 3.8% |
| layer 0 / interaction 0 | 898 | 2.5% |
| neighbour list / edges | 91 | 0.3% |
| other | 15 | 0.0% |
| forces / integrator / guards | 15 | 0.0% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 15,584.3 | 44.1% | 26.6 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>` | 7,456.8 | 21.1% | 9.6 |
| `gconv_bwd_L1<` | 3,169.3 | 9.0% | 1.1 |
| `gconv_fwd_L1<` | 1,801.1 | 5.1% | 1.1 |
| `gate_fwd_kernel` | 1,348.4 | 3.8% | 2.1 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 1,156.1 | 3.3% | 1.1 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 991.1 | 2.8% | 1.1 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 840.1 | 2.4% | 4.2 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 671.6 | 1.9% | 2.1 |
| `magma_sgemmEx_kernel<float, float, float,` | 606.4 | 1.7% | 1.1 |
| `first_bwd_edge_kernel<` | 601.0 | 1.7% | 1.1 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 564.7 | 1.6% | 1.1 |
| `first_fwd_kernel` | 297.0 | 0.8% | 1.1 |
| `sgemm32_kernel<` | 99.6 | 0.3% | 1.1 |

## mp0m_nbcl5

Kernel time 773 us per step (summed over streams), 40 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 310 | 40.1% |
| conv forward (interaction >= 1) | 117 | 15.1% |
| conv backward (interaction >= 1) | 108 | 14.0% |
| symmetric contraction / product | 101 | 13.1% |
| layer 0 / interaction 0 | 94 | 12.2% |
| neighbour list / edges | 25 | 3.2% |
| forces / integrator / guards | 9 | 1.2% |
| other | 9 | 1.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `gconv_fwd_L1s<` | 116.8 | 15.1% | 1.0 |
| `gconv_bwd_L1sr<` | 107.8 | 14.0% | 1.0 |
| `sgemm64_kernel<` | 87.9 | 11.4% | 6.0 |
| `first_bwd_edge_kernel<` | 62.4 | 8.1% | 1.0 |
| `sgemm128_kernel<` | 60.5 | 7.8% | 3.0 |
| `sgemm32_kernel<` | 59.0 | 7.6% | 7.1 |
| `fm_tf32x3_rc_64x64` | 49.0 | 6.3% | 3.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 46.1 | 6.0% | 2.0 |
| `gprod_bwdc_dd662f9d3a2b5b2b` | 45.4 | 5.9% | 1.0 |
| `first_fwd_kernel` | 31.9 | 4.1% | 1.0 |
| `gprod_ro1s_fused_ae0a447b65f4fcd4<` | 28.8 | 3.7% | 1.0 |
| `gprod_fwdc_dd662f9d3a2b5b2b` | 26.8 | 3.5% | 1.0 |
| `compute_and_count_active_kernel` | 15.0 | 1.9% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>` | 7.4 | 1.0% | 1.0 |

## mp0m_srtio3

Kernel time 3,367 us per step (summed over streams), 45 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 864 | 25.7% |
| conv backward (interaction >= 1) | 806 | 23.9% |
| conv forward (interaction >= 1) | 670 | 19.9% |
| layer 0 / interaction 0 | 620 | 18.4% |
| symmetric contraction / product | 293 | 8.7% |
| neighbour list / edges | 86 | 2.5% |
| forces / integrator / guards | 15 | 0.4% |
| other | 13 | 0.4% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `gconv_bwd_L1sr<` | 806.1 | 23.9% | 1.0 |
| `gconv_fwd_L1s<` | 669.9 | 19.9% | 1.0 |
| `first_bwd_edge_kernel<` | 381.8 | 11.3% | 1.0 |
| `fm_tf32x3_rc_64x64` | 244.6 | 7.3% | 3.0 |
| `first_fwd_kernel` | 238.4 | 7.1% | 1.0 |
| `fm_tf32x3_rr_64x64` | 215.4 | 6.4% | 3.0 |
| `sgemm128_kernel<` | 184.0 | 5.5% | 6.1 |
| `gprod_bwdc_dd662f9d3a2b5b2b` | 144.5 | 4.3% | 1.0 |
| `gprod_ro1s_fused_ae0a447b65f4fcd4<` | 81.2 | 2.4% | 1.0 |
| `sgemm64_kernel<` | 75.3 | 2.2% | 3.0 |
| `gprod_fwdc_dd662f9d3a2b5b2b` | 67.4 | 2.0% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 52.6 | 1.6% | 2.0 |
| `sgemm32_kernel<` | 51.1 | 1.5% | 6.1 |
| `compute_and_count_active_kernel` | 41.9 | 1.2% | 1.0 |

## mptrj_nbcl5

Kernel time 2,524 us per step (summed over streams), 46 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 1,306 | 51.7% |
| conv backward (interaction >= 1) | 467 | 18.5% |
| conv forward (interaction >= 1) | 355 | 14.1% |
| symmetric contraction / product | 304 | 12.0% |
| layer 0 / interaction 0 | 60 | 2.4% |
| neighbour list / edges | 13 | 0.5% |
| other | 11 | 0.4% |
| forces / integrator / guards | 9 | 0.4% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 637.6 | 25.3% | 15.1 |
| `gconv_bwd_L1<` | 467.2 | 18.5% | 1.0 |
| `gconv_fwd_L1<` | 355.3 | 14.1% | 1.0 |
| `fm_tf32x3_rr_64x32` | 321.1 | 12.7% | 3.0 |
| `gprod_bwdc_dd662f9d3a2b5b2b` | 150.5 | 6.0% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_tn_align1>` | 106.3 | 4.2% | 1.0 |
| `fm_tf32x3_rc_64x64` | 96.2 | 3.8% | 2.0 |
| `gprod_ro1c_fused_ae0a447b65f4fcd4<` | 81.0 | 3.2% | 1.0 |
| `fm_tf32x3_rr_64x64` | 77.8 | 3.1% | 1.0 |
| `gprod_fwdc_dd662f9d3a2b5b2b` | 72.2 | 2.9% | 1.0 |
| `sgemm32_kernel<` | 64.1 | 2.5% | 5.0 |
| `first_bwd_edge_kernel<` | 38.4 | 1.5% | 1.0 |
| `first_fwd_kernel` | 21.3 | 0.8% | 1.0 |
| `compute_and_count_active_kernel` | 5.7 | 0.2% | 1.0 |

## mptrj_srtio3

Kernel time 8,924 us per step (summed over streams), 50 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 3,001 | 33.6% |
| conv backward (interaction >= 1) | 2,444 | 27.4% |
| conv forward (interaction >= 1) | 1,873 | 21.0% |
| symmetric contraction / product | 1,145 | 12.8% |
| layer 0 / interaction 0 | 385 | 4.3% |
| neighbour list / edges | 43 | 0.5% |
| other | 18 | 0.2% |
| forces / integrator / guards | 15 | 0.2% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `gconv_bwd_L1<` | 2,444.1 | 27.4% | 1.0 |
| `gconv_fwd_L1<` | 1,873.4 | 21.0% | 1.0 |
| `sgemm128_kernel<` | 1,326.0 | 14.9% | 12.2 |
| `fm_tf32x3_rr_64x32` | 668.4 | 7.5% | 3.0 |
| `gprod_bwdc_dd662f9d3a2b5b2b` | 559.9 | 6.3% | 1.0 |
| `fm_tf32x3_rc_64x64` | 438.2 | 4.9% | 2.0 |
| `gprod_fwdc_dd662f9d3a2b5b2b` | 293.3 | 3.3% | 1.0 |
| `gprod_ro1c_fused_ae0a447b65f4fcd4<` | 292.0 | 3.3% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 276.5 | 3.1% | 5.1 |
| `first_bwd_edge_kernel<` | 212.0 | 2.4% | 1.0 |
| `first_fwd_kernel` | 172.6 | 1.9% | 1.0 |
| `fm_tf32x3_rc_64x32` | 146.1 | 1.6% | 1.0 |
| `sgemm32_kernel<` | 86.8 | 1.0% | 6.1 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>` | 59.1 | 0.7% | 3.0 |

## off23l_water216

Kernel time 3,654 us per step (summed over streams), 57 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 1,618 | 44.3% |
| conv backward (interaction >= 1) | 1,068 | 29.2% |
| conv forward (interaction >= 1) | 734 | 20.1% |
| symmetric contraction / product | 118 | 3.2% |
| layer 0 / interaction 0 | 65 | 1.8% |
| neighbour list / edges | 37 | 1.0% |
| forces / integrator / guards | 8 | 0.2% |
| other | 6 | 0.2% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 904.8 | 24.8% | 10.2 |
| `gconv_bwd_L2<` | 579.4 | 15.9% | 1.0 |
| `gconv_bwd_L2lp<` | 488.4 | 13.4% | 1.0 |
| `gconv_fwd_L2a<` | 410.2 | 11.2% | 1.0 |
| `gconv_fwd_L2b<` | 324.0 | 8.9% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 281.9 | 7.7% | 3.0 |
| `sgemm32_kernel<` | 236.3 | 6.5% | 10.2 |
| `sgemm64_kernel<` | 145.0 | 4.0% | 8.1 |
| `gprod_bwd_00dc9085957f7969` | 47.9 | 1.3% | 1.0 |
| `first_bwd_edge_kernel<` | 45.5 | 1.2% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 40.0 | 1.1% | 1.0 |
| `fm_tf32x3_rc_64x32` | 39.6 | 1.1% | 1.0 |
| `gprod_fwd_00dc9085957f7969` | 30.4 | 0.8% | 1.0 |
| `first_fwd_kernel` | 19.9 | 0.5% | 1.0 |

## off23m_water216

Kernel time 1,186 us per step (summed over streams), 51 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 546 | 46.0% |
| conv backward (interaction >= 1) | 265 | 22.4% |
| conv forward (interaction >= 1) | 208 | 17.5% |
| layer 0 / interaction 0 | 65 | 5.5% |
| symmetric contraction / product | 51 | 4.3% |
| neighbour list / edges | 38 | 3.2% |
| forces / integrator / guards | 8 | 0.7% |
| other | 6 | 0.5% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `gconv_bwd_L1<` | 265.5 | 22.4% | 1.0 |
| `gconv_fwd_L1<` | 207.9 | 17.5% | 1.0 |
| `sgemm64_kernel<` | 127.5 | 10.8% | 6.0 |
| `fm_tf32x3_rc_64x64` | 105.8 | 8.9% | 5.0 |
| `sgemm128_kernel<` | 96.6 | 8.1% | 3.0 |
| `sgemm32_kernel<` | 82.9 | 7.0% | 10.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 52.1 | 4.4% | 1.0 |
| `first_bwd_edge_kernel<` | 45.0 | 3.8% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>` | 22.7 | 1.9% | 2.0 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 22.6 | 1.9% | 1.0 |
| `first_fwd_kernel` | 19.7 | 1.7% | 1.0 |
| `fm_tf32x3_rc_32x64` | 19.0 | 1.6% | 2.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 18.6 | 1.6% | 1.0 |
| `fill_neighbors_kernel` | 18.5 | 1.6% | 0.2 |

## off23s_water216

Kernel time 486 us per step (summed over streams), 49 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 258 | 53.1% |
| conv backward (interaction >= 1) | 62 | 12.7% |
| conv forward (interaction >= 1) | 47 | 9.7% |
| layer 0 / interaction 0 | 45 | 9.3% |
| neighbour list / edges | 34 | 7.1% |
| symmetric contraction / product | 27 | 5.5% |
| forces / integrator / guards | 7 | 1.5% |
| other | 6 | 1.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm32_kernel<` | 73.5 | 15.1% | 14.1 |
| `sgemm64_kernel<` | 67.6 | 13.9% | 6.0 |
| `gconv_bwd_L0<` | 61.9 | 12.7% | 1.0 |
| `sgemm96_kernel` | 54.9 | 11.3% | 4.0 |
| `gconv_fwd_L0<` | 47.0 | 9.7% | 1.0 |
| `first_bwd_edge_kernel<` | 29.9 | 6.2% | 1.0 |
| `fm_tf32x3_rr_64x32` | 29.4 | 6.1% | 3.0 |
| `fill_neighbors_kernel` | 16.7 | 3.4% | 0.2 |
| `first_fwd_kernel` | 15.3 | 3.2% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 13.5 | 2.8% | 1.0 |
| `fm_tf32x3_rc_32x64` | 12.0 | 2.5% | 1.0 |
| `fm_tf32x3_rc_64x64` | 11.5 | 2.4% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>` | 8.9 | 1.8% | 1.0 |
| `gprod_fwd_ae0a447b65f4fcd4` | 7.0 | 1.4% | 1.0 |

## off24m_water

Kernel time 9,783 us per step (summed over streams), 53 launches per step.

| Category | us / step | share |
|---|---:|---:|
| conv backward (interaction >= 1) | 3,548 | 36.3% |
| conv forward (interaction >= 1) | 2,466 | 25.2% |
| GEMM | 2,213 | 22.6% |
| layer 0 / interaction 0 | 710 | 7.3% |
| symmetric contraction / product | 596 | 6.1% |
| neighbour list / edges | 229 | 2.3% |
| forces / integrator / guards | 15 | 0.2% |
| other | 7 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `gconv_bwd_L1<` | 3,547.6 | 36.3% | 1.0 |
| `gconv_fwd_L1<` | 2,465.9 | 25.2% | 1.0 |
| `sgemm128_kernel<` | 1,417.1 | 14.5% | 14.4 |
| `first_bwd_edge_kernel<` | 461.9 | 4.7% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 450.7 | 4.6% | 5.2 |
| `first_fwd_kernel` | 248.3 | 2.5% | 1.0 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 245.3 | 2.5% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 236.8 | 2.4% | 1.0 |
| `fm_tf32x3_rc_64x64` | 116.3 | 1.2% | 4.1 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 114.1 | 1.2% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 109.3 | 1.1% | 2.1 |
| `fill_neighbors_kernel` | 89.9 | 0.9% | 0.3 |
| `fm_tf32x3_rr_64x64` | 51.6 | 0.5% | 2.1 |
| `compute_and_count_active_kernel` | 48.6 | 0.5% | 1.0 |

## omat0m_gen_srtio3

Kernel time 8,624 us per step (summed over streams), 59 launches per step.

| Category | us / step | share |
|---|---:|---:|
| conv backward (interaction >= 1) | 3,013 | 34.9% |
| GEMM | 2,484 | 28.8% |
| conv forward (interaction >= 1) | 1,718 | 19.9% |
| layer 0 / interaction 0 | 697 | 8.1% |
| symmetric contraction / product | 601 | 7.0% |
| neighbour list / edges | 91 | 1.1% |
| forces / integrator / guards | 15 | 0.2% |
| other | 5 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `gconv_bwd_L1<` | 3,013.1 | 34.9% | 1.0 |
| `sgemm128_kernel<` | 1,842.1 | 21.4% | 26.4 |
| `gconv_fwd_L1<` | 1,717.9 | 19.9% | 1.0 |
| `first_bwd_edge_kernel<` | 454.5 | 5.3% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 257.9 | 3.0% | 2.0 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 253.0 | 2.9% | 1.0 |
| `first_fwd_kernel` | 243.0 | 2.8% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 236.2 | 2.7% | 1.0 |
| `fm_tf32x3_rr_64x64` | 118.7 | 1.4% | 2.0 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 111.4 | 1.3% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 102.8 | 1.2% | 2.0 |
| `sgemm32_kernel<` | 65.5 | 0.8% | 5.1 |
| `sgemm64_kernel<` | 61.4 | 0.7% | 2.0 |
| `compute_and_count_active_kernel` | 41.4 | 0.5% | 1.0 |

## omat0m_spec_srtio3

Kernel time 8,739 us per step (summed over streams), 59 launches per step.

| Category | us / step | share |
|---|---:|---:|
| conv backward (interaction >= 1) | 2,969 | 34.0% |
| GEMM | 2,431 | 27.8% |
| conv forward (interaction >= 1) | 1,818 | 20.8% |
| layer 0 / interaction 0 | 823 | 9.4% |
| symmetric contraction / product | 567 | 6.5% |
| neighbour list / edges | 110 | 1.3% |
| forces / integrator / guards | 15 | 0.2% |
| other | 5 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `wprime_backward_inter1_rcv` | 2,969.3 | 34.0% | 1.0 |
| `spline_forward_inter1<` | 1,818.1 | 20.8% | 1.0 |
| `sgemm128_kernel<` | 1,720.0 | 19.7% | 23.4 |
| `tabled_outer_sum0_backward_kernel` | 546.4 | 6.3% | 1.0 |
| `tabled_outer_sum0_forward_kernel` | 277.0 | 3.2% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 273.6 | 3.1% | 2.0 |
| `sc1_fused` | 236.0 | 2.7% | 1.0 |
| `fm_tf32x3_rr_64x64` | 222.1 | 2.5% | 5.1 |
| `sc0_backward` | 221.9 | 2.5% | 1.0 |
| `sc0_forward` | 109.5 | 1.3% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 80.2 | 0.9% | 2.0 |
| `sgemm32_kernel<` | 59.6 | 0.7% | 5.1 |
| `compute_and_count_active_kernel` | 45.7 | 0.5% | 1.0 |
| `scatter_active_edges_kernel` | 41.6 | 0.5% | 1.0 |

## omat0s_srtio3

Kernel time 4,861 us per step (summed over streams), 57 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 1,634 | 33.6% |
| conv backward (interaction >= 1) | 1,141 | 23.5% |
| conv forward (interaction >= 1) | 742 | 15.3% |
| layer 0 / interaction 0 | 694 | 14.3% |
| symmetric contraction / product | 544 | 11.2% |
| neighbour list / edges | 87 | 1.8% |
| forces / integrator / guards | 14 | 0.3% |
| other | 5 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `gconv_bwd_L0<` | 1,140.8 | 23.5% | 1.0 |
| `sgemm128_kernel<` | 1,042.1 | 21.4% | 20.3 |
| `gconv_fwd_L0<` | 742.2 | 15.3% | 1.0 |
| `first_bwd_edge_kernel<` | 450.7 | 9.3% | 1.0 |
| `fm_tf32x3_rr_64x64` | 256.9 | 5.3% | 7.1 |
| `first_fwd_kernel` | 243.3 | 5.0% | 1.0 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<` | 241.2 | 5.0% | 1.0 |
| `gprod_bwd_ae0a447b65f4fcd4` | 212.4 | 4.4% | 1.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 173.9 | 3.6% | 2.0 |
| `gprod_fwd_ae0a447b65f4fcd4` | 90.3 | 1.9% | 1.0 |
| `fm_tf32x3_rc_64x64` | 60.7 | 1.2% | 2.0 |
| `sgemm64_kernel<` | 49.9 | 1.0% | 2.0 |
| `sgemm32_kernel<` | 44.4 | 0.9% | 5.1 |
| `compute_and_count_active_kernel` | 41.3 | 0.8% | 1.0 |

## omol0xl_water216

Kernel time 27,604 us per step (summed over streams), 140 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 21,813 | 79.0% |
| conv backward (interaction >= 1) | 2,105 | 7.6% |
| conv forward (interaction >= 1) | 1,312 | 4.8% |
| symmetric contraction / product | 1,160 | 4.2% |
| nonlinear gates | 862 | 3.1% |
| layer 0 / interaction 0 | 263 | 1.0% |
| neighbour list / edges | 46 | 0.2% |
| forces / integrator / guards | 28 | 0.1% |
| other | 16 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 14,597.5 | 52.9% | 45.4 |
| `sgemm64_kernel<` | 3,357.6 | 12.2% | 23.7 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>` | 2,582.6 | 9.4% | 6.2 |
| `gconv_bwd_L2<` | 1,111.7 | 4.0% | 2.1 |
| `gconv_bwd_L2lp<` | 993.2 | 3.6% | 2.1 |
| `sgemm32_kernel<` | 905.7 | 3.3% | 9.3 |
| `gate_fwd_kernel` | 765.0 | 2.8% | 3.1 |
| `gconv_fwd_L2a<` | 714.5 | 2.6% | 2.1 |
| `gconv_fwd_L2b<` | 597.0 | 2.2% | 2.1 |
| `gprod_bwd_6f194247389dfa5c` | 531.7 | 1.9% | 2.1 |
| `gprod_fwd_6f194247389dfa5c` | 300.0 | 1.1% | 2.1 |
| `gprod_bwd_4715eda4cc3ebda5` | 234.9 | 0.9% | 1.0 |
| `first_bwd_kernel<` | 193.8 | 0.7% | 1.0 |
| `fm_tf32x3_rc_64x32` | 165.3 | 0.6% | 1.0 |

## polar1l_nacl216

Kernel time 9,024 us per step (summed over streams), 379 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 7,703 | 85.4% |
| conv backward (interaction >= 1) | 303 | 3.4% |
| POLAR dense / elementwise | 224 | 2.5% |
| conv forward (interaction >= 1) | 211 | 2.3% |
| POLAR k-space | 187 | 2.1% |
| symmetric contraction / product | 170 | 1.9% |
| nonlinear gates | 155 | 1.7% |
| layer 0 / interaction 0 | 33 | 0.4% |
| other | 21 | 0.2% |
| neighbour list / edges | 9 | 0.1% |
| forces / integrator / guards | 7 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm64_kernel<` | 4,225.9 | 46.8% | 94.5 |
| `sgemm128_kernel<` | 1,715.6 | 19.0% | 22.3 |
| `sgemm32_kernel<` | 691.7 | 7.7% | 45.7 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 647.2 | 7.2% | 17.3 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>` | 234.8 | 2.6% | 17.3 |
| `gconv_bwd_L2<` | 162.1 | 1.8% | 2.0 |
| `gconv_bwd_L2lp<` | 140.5 | 1.6% | 2.0 |
| `gprod_bwd_00dc9085957f7969` | 122.9 | 1.4% | 3.0 |
| `gconv_fwd_L2a<` | 117.2 | 1.3% | 2.0 |
| `gate_bwd_kernel` | 116.2 | 1.3% | 3.0 |
| `gconv_fwd_L2b<` | 94.1 | 1.0% | 2.0 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>` | 92.3 | 1.0% | 3.0 |
| `sf_kernel_tiled<` | 57.7 | 0.6% | 5.1 |
| `inv_bwd_kernel` | 54.3 | 0.6% | 10.2 |

## polar1m_nacl216

Kernel time 4,682 us per step (summed over streams), 294 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 3,951 | 84.4% |
| POLAR dense / elementwise | 196 | 4.2% |
| POLAR k-space | 175 | 3.7% |
| nonlinear gates | 90 | 1.9% |
| symmetric contraction / product | 78 | 1.7% |
| conv backward (interaction >= 1) | 72 | 1.5% |
| conv forward (interaction >= 1) | 55 | 1.2% |
| layer 0 / interaction 0 | 34 | 0.7% |
| other | 16 | 0.3% |
| neighbour list / edges | 8 | 0.2% |
| forces / integrator / guards | 7 | 0.2% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm64_kernel<` | 2,510.8 | 53.6% | 59.5 |
| `sgemm128_kernel<` | 543.8 | 11.6% | 9.1 |
| `sgemm32_kernel<` | 437.3 | 9.3% | 37.3 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>` | 212.9 | 4.5% | 16.1 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>` | 128.2 | 2.7% | 12.1 |
| `gconv_bwd_L1<` | 72.4 | 1.5% | 1.0 |
| `gate_bwd_kernel` | 68.9 | 1.5% | 2.0 |
| `sf_kernel_tiled<` | 58.1 | 1.2% | 5.0 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 58.1 | 1.2% | 2.0 |
| `gconv_fwd_L1<` | 55.5 | 1.2% | 1.0 |
| `grad_pass_kernel<` | 46.3 | 1.0% | 3.0 |
| `cublasLt::splitKreduce_kernel<` | 33.6 | 0.7% | 12.1 |
| `inv_bwd_kernel` | 30.5 | 0.7% | 8.1 |
| `ro_bwd_kernel` | 26.5 | 0.6% | 2.0 |

## polar1m_nbcl5

Kernel time 25,091 us per step (summed over streams), 294 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 18,013 | 71.8% |
| POLAR dense / elementwise | 1,973 | 7.9% |
| POLAR k-space | 1,960 | 7.8% |
| symmetric contraction / product | 1,197 | 4.8% |
| nonlinear gates | 883 | 3.5% |
| conv backward (interaction >= 1) | 479 | 1.9% |
| conv forward (interaction >= 1) | 345 | 1.4% |
| layer 0 / interaction 0 | 156 | 0.6% |
| other | 50 | 0.2% |
| neighbour list / edges | 25 | 0.1% |
| forces / integrator / guards | 9 | 0.0% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 9,126.0 | 36.4% | 47.8 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>` | 5,046.0 | 20.1% | 30.8 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 1,470.2 | 5.9% | 8.5 |
| `sf_kernel_tiled<` | 1,088.8 | 4.3% | 5.3 |
| `gate_fwd_kernel` | 883.3 | 3.5% | 2.1 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 825.8 | 3.3% | 2.1 |
| `inv_bwd_kernel` | 799.7 | 3.2% | 8.5 |
| `sgemm64_kernel<` | 747.9 | 3.0% | 21.2 |
| `grad_pass_kernel<` | 551.6 | 2.2% | 3.2 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>` | 534.5 | 2.1% | 12.8 |
| `gconv_bwd_L1<` | 479.3 | 1.9% | 1.1 |
| `bcast_mul_bwd_kernel` | 383.7 | 1.5% | 4.2 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 371.3 | 1.5% | 2.1 |
| `sgemm32_kernel<` | 358.1 | 1.4% | 9.6 |

## polar1s_nacl216

Kernel time 1,621 us per step (summed over streams), 189 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 1,249 | 77.1% |
| POLAR dense / elementwise | 117 | 7.2% |
| POLAR k-space | 116 | 7.1% |
| nonlinear gates | 56 | 3.4% |
| layer 0 / interaction 0 | 32 | 2.0% |
| symmetric contraction / product | 24 | 1.5% |
| other | 12 | 0.7% |
| neighbour list / edges | 8 | 0.5% |
| forces / integrator / guards | 8 | 0.5% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm64_kernel<` | 699.2 | 43.1% | 22.2 |
| `sgemm32_kernel<` | 336.1 | 20.7% | 30.2 |
| `sgemm128_kernel<` | 160.1 | 9.9% | 3.0 |
| `gate_bwd_kernel` | 45.0 | 2.8% | 1.0 |
| `grad_pass_kernel<` | 30.6 | 1.9% | 3.0 |
| `first_bwd_edge_kernel<` | 23.5 | 1.5% | 1.0 |
| `sf_kernel_tiled<` | 23.3 | 1.4% | 5.0 |
| `feat_coef_kernel` | 21.4 | 1.3% | 2.0 |
| `bias_ln_silu_fwd_kernel` | 21.2 | 1.3% | 9.1 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>` | 20.9 | 1.3% | 8.1 |
| `gprod_bwd_ae0a447b65f4fcd4` | 17.5 | 1.1% | 1.0 |
| `bwd_coef_kernel` | 17.2 | 1.1% | 3.0 |
| `ln_silu_bwd_kernel` | 15.3 | 0.9% | 9.1 |
| `feat_pass_kernel<` | 11.9 | 0.7% | 2.0 |

## polar1s_nbcl5

Kernel time 8,585 us per step (summed over streams), 183 launches per step.

| Category | us / step | share |
|---|---:|---:|
| GEMM | 6,089 | 70.9% |
| POLAR k-space | 822 | 9.6% |
| symmetric contraction / product | 524 | 6.1% |
| POLAR dense / elementwise | 508 | 5.9% |
| nonlinear gates | 426 | 5.0% |
| layer 0 / interaction 0 | 144 | 1.7% |
| other | 37 | 0.4% |
| neighbour list / edges | 24 | 0.3% |
| forces / integrator / guards | 11 | 0.1% |

| Kernel | us / step | share | launches / step |
|---|---:|---:|---:|
| `sgemm128_kernel<` | 3,261.0 | 38.0% | 21.7 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>` | 1,345.5 | 15.7% | 13.4 |
| `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_align1>` | 740.7 | 8.6% | 2.1 |
| `gate_fwd_kernel` | 425.8 | 5.0% | 1.0 |
| `gprod_bwd_ae0a447b65f4fcd4` | 362.3 | 4.2% | 1.0 |
| `sf_kernel_tiled<` | 345.6 | 4.0% | 5.2 |
| `sgemm32_kernel<` | 299.1 | 3.5% | 8.2 |
| `grad_pass_kernel<` | 295.2 | 3.4% | 3.1 |
| `sgemm64_kernel<` | 284.7 | 3.3% | 13.4 |
| `gprod_fwd_ae0a447b65f4fcd4` | 161.7 | 1.9% | 1.0 |
| `feat_pass_kernel<` | 117.3 | 1.4% | 2.1 |
| `inv_bwd_kernel` | 112.1 | 1.3% | 4.1 |
| `first_bwd_edge_kernel<` | 92.9 | 1.1% | 1.0 |
| `bcast_mul_bwd_kernel` | 79.1 | 0.9% | 2.1 |

# ncu, one force call (top kernels, sections + raw metrics)

## mh1_srtio3

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sgemm128_kernel<8, 1, 1, 0, 0>` | 12 | 4,117.0 | 68 | 23 | 37 | 32 | 128 | 67 | 1 | not_selected 2.33, selected 1.00, dispatch_stall 0.93 |
| `sgemm128_kernel<8, 1, 1, 7, 0>` | 2 | 2,990.0 | 67 | 23 | 39 | 33 | 128 | 67 | 2 | not_selected 2.13, dispatch_stall 1.13, selected 1.00 |
| `gconv_bwd_L1<1, 128, 1>` | 1 | 2,774.9 | 76 | 80 | 22 | 64 | 62 | 96 | 10 | not_selected 3.95, dispatch_stall 1.37, long_scoreboard 1.33 |
| `sgemm128_kernel<8, 1, 1, 5, 0>` | 2 | 2,330.6 | 61 | 22 | 40 | 33 | 128 | 65 | 3 | not_selected 2.12, dispatch_stall 1.12, selected 1.00 |
| `sgemm128_kernel<8, 0, 1, 0, 0>` | 4 | 1,915.6 | 68 | 20 | 34 | 31 | 128 | 61 | 1 | not_selected 2.31, selected 1.00, dispatch_stall 0.87 |
| `sgemm128_kernel<8, 1, 1, 3, 0>` | 2 | 1,610.3 | 54 | 21 | 43 | 33 | 128 | 59 | 4 | not_selected 2.07, long_scoreboard 1.42, dispatch_stall 1.09 |
| `gconv_fwd_L1<5, 0, 640, 1, 128>` | 1 | 1,536.2 | 67 | 61 | 23 | 41 | 96 | 91 | 54 | not_selected 1.83, long_scoreboard 1.68, selected 1.00 |
| `gate_fwd_kernel` | 2 | 1,211.3 | 9 | 23 | 85 | 88 | 40 | 0 | 3 | long_scoreboard 135.65, wait 1.87, short_scoreboard 1.03 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 1 | 1,039.1 | 27 | 18 | 85 | 25 | 160 | 2 | 44 | long_scoreboard 7.37, selected 1.00, not_selected 0.60 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<512, 1>` | 1 | 919.2 | 19 | 21 | 82 | 36 | 110 | 17 | 79 | long_scoreboard 5.31, lg_throttle 4.93, barrier 4.60 |
| `sgemm128_kernel<8, 1, 1, 1, 0>` | 2 | 691.0 | 46 | 17 | 24 | 32 | 128 | 55 | 14 | not_selected 1.99, long_scoreboard 1.60, selected 1.00 |
| `sgemm128_kernel<8, 1, 1, 7, 1>` | 1 | 616.2 | 68 | 23 | 39 | 33 | 128 | 65 | 4 | not_selected 2.19, dispatch_stall 1.13, selected 1.00 |
| `first_bwd_edge_kernel<3, 1, 0>` | 1 | 528.9 | 54 | 43 | 44 | 31 | 118 | 82 | 23 | long_scoreboard 2.27, selected 1.00, wait 0.99 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 1 | 517.3 | 31 | 29 | 92 | 50 | 72 | 2 | 59 | long_scoreboard 14.33, barrier 2.90, wait 1.23 |
| `sgemm128_kernel<8, 1, 1, 5, 1>` | 1 | 480.9 | 64 | 23 | 44 | 33 | 128 | 61 | 6 | not_selected 2.11, dispatch_stall 1.09, selected 1.00 |
| `sgemm128_kernel<8, 1, 1, 3, 1>` | 1 | 327.6 | 57 | 20 | 47 | 33 | 128 | 61 | 9 | not_selected 2.11, long_scoreboard 1.35, dispatch_stall 1.09 |

## mp0m_srtio3

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gconv_bwd_L1sr<4, 1, 1>` | 1 | 829.1 | 46 | 59 | 17 | 41 | 96 | 93 | 15 | long_scoreboard 6.01, wait 1.18, selected 1.00 |
| `gconv_fwd_L1s<5, 0, 640, 1, 128>` | 1 | 642.8 | 47 | 71 | 13 | 41 | 48 | 96 | 31 | long_scoreboard 9.29, wait 1.64, selected 1.00 |
| `first_bwd_edge_kernel<2, 0, 0>` | 1 | 390.5 | 66 | 60 | 54 | 47 | 80 | 85 | 12 | long_scoreboard 2.04, not_selected 1.71, short_scoreboard 1.32 |
| `first_fwd_kernel` | 1 | 214.1 | 68 | 81 | 59 | 75 | 54 | 81 | 20 | long_scoreboard 4.51, mio_throttle 3.13, short_scoreboard 2.55 |
| `fm_tf32x3_rr_64x64` | 3 | 191.9 | 85 | 31 | 68 | 23 | 128 | 51 | 1 | math_pipe_throttle 4.54, selected 1.00, long_scoreboard 0.87 |
| `fm_tf32x3_rc_64x64` | 4 | 180.8 | 82 | 45 | 55 | 23 | 132 | 50 | 0 | math_pipe_throttle 3.44, selected 1.00, wait 0.73 |
| `gprod_bwdc_dd662f9d3a2b5b2b` | 1 | 156.0 | 32 | 14 | 48 | 15 | 149 | 21 | 60 | long_scoreboard 2.57, selected 1.00, wait 0.74 |
| `sgemm128_kernel<8, 1, 1, 0, 0>` | 4 | 136.0 | 49 | 23 | 41 | 28 | 128 | 35 | 3 | not_selected 1.89, selected 1.00, dispatch_stall 0.79 |
| `gprod_ro1s_fused_ae0a447b65f4fcd4<128, 4>` | 1 | 78.9 | 30 | 7 | 13 | 79 | 48 | 27 | 98 | barrier 19.44, mio_throttle 18.54, lg_throttle 14.11 |
| `gprod_fwdc_dd662f9d3a2b5b2b` | 1 | 76.8 | 42 | 33 | 61 | 53 | 72 | 36 | 68 | long_scoreboard 5.13, barrier 3.84, wait 1.73 |
| `sgemm64_kernel<0>` | 3 | 56.7 | 38 | 21 | 54 | 23 | 101 | 30 | 5 | long_scoreboard 2.16, selected 1.00, not_selected 0.44 |
| `sgemm128_kernel<4, 1, 1, 0, 0>` | 1 | 25.3 | 54 | 28 | 26 | 29 | 105 | 58 | 0 | not_selected 1.62, selected 1.00, dispatch_stall 0.63 |
| `sgemm32_kernel<0>` | 4 | 22.7 | 17 | 18 | 48 | 9 | 77 | 32 | 5 | long_scoreboard 1.31, selected 1.00, short_scoreboard 0.61 |
| `fm_tf32x3_rr_64x32` | 1 | 15.2 | 61 | 34 | 24 | 14 | 128 | 73 | 0 | math_pipe_throttle 2.58, selected 1.00, wait 0.87 |
| `sgemm64_kernel<1>` | 2 | 12.5 | 29 | 20 | 35 | 9 | 101 | 55 | 0 | selected 1.00, short_scoreboard 0.32, long_scoreboard 0.28 |

## mptrj_srtio3

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gconv_bwd_L1<1, 256, 1>` | 1 | 2,367.5 | 72 | 82 | 41 | 58 | 62 | 92 | 8 | not_selected 3.45, long_scoreboard 1.43, dispatch_stall 1.22 |
| `gconv_fwd_L1<1, 0, 256, 1, 256>` | 1 | 1,818.1 | 49 | 85 | 40 | 50 | 80 | 90 | 12 | long_scoreboard 7.14, wait 1.18, selected 1.00 |
| `sgemm128_kernel<8, 1, 1, 0, 0>` | 6 | 659.8 | 64 | 23 | 31 | 31 | 128 | 53 | 0 | not_selected 2.22, selected 1.00, dispatch_stall 0.90 |
| `gprod_bwdc_dd662f9d3a2b5b2b` | 1 | 549.0 | 31 | 17 | 63 | 20 | 149 | 20 | 62 | long_scoreboard 4.83, selected 1.00, wait 0.68 |
| `fm_tf32x3_rr_64x32` | 3 | 503.8 | 57 | 37 | 92 | 15 | 128 | 39 | 0 | long_scoreboard 4.38, math_pipe_throttle 2.15, selected 1.00 |
| `sgemm128_kernel<4, 1, 1, 0, 0>` | 5 | 497.2 | 67 | 33 | 44 | 32 | 105 | 60 | 0 | not_selected 1.86, selected 1.00, dispatch_stall 0.69 |
| `gprod_ro1c_fused_ae0a447b65f4fcd4<256, 2>` | 1 | 325.9 | 16 | 4 | 13 | 24 | 96 | 27 | 98 | barrier 8.13, long_scoreboard 4.30, lg_throttle 2.98 |
| `gprod_fwdc_dd662f9d3a2b5b2b` | 1 | 268.4 | 35 | 20 | 72 | 59 | 72 | 24 | 69 | long_scoreboard 7.33, barrier 2.48, wait 1.34 |
| `fm_tf32x3_rc_64x64` | 1 | 210.4 | 26 | 32 | 71 | 15 | 132 | 31 | 0 | long_scoreboard 5.45, short_scoreboard 1.04, selected 1.00 |
| `first_bwd_edge_kernel<2, 0, 0>` | 1 | 208.4 | 55 | 53 | 82 | 48 | 80 | 70 | 14 | long_scoreboard 2.43, not_selected 1.59, short_scoreboard 1.45 |
| `first_fwd_kernel` | 1 | 175.3 | 38 | 41 | 66 | 56 | 54 | 61 | 29 | long_scoreboard 12.97, barrier 3.69, short_scoreboard 1.63 |
| `sgemm128_kernel<4, 0, 1, 0, 0>` | 1 | 153.5 | 68 | 30 | 54 | 32 | 105 | 62 | 0 | not_selected 1.93, selected 1.00, dispatch_stall 0.72 |
| `sgemm32_kernel<0>` | 6 | 52.1 | 18 | 19 | 52 | 7 | 77 | 31 | 3 | selected 1.00, long_scoreboard 0.90, short_scoreboard 0.56 |
| `sgemm64_kernel<1>` | 4 | 43.3 | 36 | 25 | 30 | 15 | 101 | 69 | 12 | long_scoreboard 3.50, selected 1.00, not_selected 0.53 |

## off23l_water216

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gconv_bwd_L2<1, 224, 1>` | 1 | 602.3 | 66 | 57 | 21 | 23 | 126 | 96 | 3 | not_selected 1.20, selected 1.00, dispatch_stall 0.54 |
| `sgemm128_kernel<8, 0, 1, 0, 0>` | 7 | 528.0 | 66 | 22 | 24 | 26 | 128 | 62 | 0 | not_selected 1.76, selected 1.00, dispatch_stall 0.72 |
| `gconv_bwd_L2lp<1, 224, 1>` | 1 | 520.0 | 70 | 80 | 20 | 57 | 62 | 96 | 4 | not_selected 3.42, long_scoreboard 1.60, dispatch_stall 1.21 |
| `gconv_fwd_L2a<1, 1, 256, 1, 224>` | 1 | 410.4 | 44 | 76 | 18 | 29 | 102 | 94 | 3 | long_scoreboard 3.49, selected 1.00, not_selected 0.81 |
| `gconv_fwd_L2b<1, 1, 256, 1, 224>` | 1 | 346.7 | 49 | 72 | 10 | 28 | 103 | 94 | 3 | long_scoreboard 2.38, not_selected 1.12, selected 1.00 |
| `sgemm128_kernel<8, 1, 1, 0, 0>` | 1 | 138.3 | 71 | 24 | 14 | 31 | 128 | 71 | 0 | not_selected 2.27, selected 1.00, dispatch_stall 0.91 |
| `sgemm32_kernel<1>` | 9 | 97.6 | 36 | 37 | 23 | 17 | 77 | 80 | 0 | selected 1.00, short_scoreboard 0.93, mio_throttle 0.64 |
| `gprod_bwd_00dc9085957f7969` | 1 | 55.7 | 45 | 25 | 83 | 16 | 222 | 11 | 49 | long_scoreboard 1.89, selected 1.00, not_selected 0.57 |
| `first_bwd_edge_kernel<2, 0, 0>` | 1 | 55.6 | 43 | 38 | 29 | 39 | 80 | 86 | 13 | long_scoreboard 3.02, not_selected 1.54, short_scoreboard 1.46 |
| `sgemm64_kernel<0>` | 4 | 50.1 | 48 | 28 | 23 | 21 | 101 | 72 | 0 | not_selected 1.01, selected 1.00, dispatch_stall 0.50 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<256, 2>` | 1 | 49.8 | 20 | 21 | 60 | 27 | 110 | 10 | 81 | long_scoreboard 5.08, barrier 4.25, lg_throttle 1.82 |
| `gprod_fwd_00dc9085957f7969` | 1 | 46.5 | 31 | 18 | 54 | 31 | 128 | 16 | 60 | long_scoreboard 5.95, short_scoreboard 1.43, wait 1.05 |
| `fm_tf32x3_rr_64x32` | 1 | 36.9 | 83 | 42 | 21 | 15 | 128 | 82 | 0 | math_pipe_throttle 2.94, selected 1.00, wait 0.89 |
| `fm_tf32x3_rc_64x32` | 1 | 36.8 | 83 | 42 | 33 | 15 | 136 | 83 | 0 | math_pipe_throttle 3.39, selected 1.00, wait 0.84 |
| `sgemm64_kernel<1>` | 3 | 31.6 | 39 | 24 | 34 | 12 | 101 | 61 | 0 | selected 1.00, short_scoreboard 0.34, dispatch_stall 0.28 |
| `first_fwd_kernel` | 1 | 23.2 | 54 | 62 | 15 | 60 | 54 | 80 | 26 | long_scoreboard 4.46, mio_throttle 3.64, short_scoreboard 2.49 |

## omat0m_gen_srtio3

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gconv_bwd_L1<1, 128, 1>` | 1 | 2,749.6 | 76 | 80 | 19 | 64 | 62 | 95 | 11 | not_selected 3.99, dispatch_stall 1.38, long_scoreboard 1.28 |
| `gconv_fwd_L1<5, 0, 640, 1, 128>` | 1 | 1,521.3 | 68 | 58 | 29 | 41 | 96 | 92 | 56 | not_selected 1.88, long_scoreboard 1.58, selected 1.00 |
| `sgemm128_kernel<8, 1, 1, 0, 0>` | 16 | 1,258.1 | 61 | 24 | 46 | 30 | 128 | 44 | 0 | not_selected 2.10, selected 1.00, dispatch_stall 0.86 |
| `first_bwd_edge_kernel<2, 1, 0>` | 1 | 452.9 | 59 | 46 | 46 | 46 | 80 | 83 | 22 | long_scoreboard 3.67, not_selected 1.30, wait 1.23 |
| `sgemm128_kernel<4, 1, 1, 0, 0>` | 10 | 346.8 | 62 | 33 | 32 | 31 | 105 | 57 | 0 | not_selected 1.77, selected 1.00, dispatch_stall 0.67 |
| `first_fwd_kernel` | 1 | 232.1 | 65 | 62 | 60 | 72 | 54 | 77 | 37 | long_scoreboard 3.43, mio_throttle 3.23, barrier 2.80 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 1 | 219.1 | 34 | 25 | 86 | 22 | 160 | 4 | 44 | long_scoreboard 6.18, selected 1.00, not_selected 0.82 |
| `gprod_ro1_fused_ae0a447b65f4fcd4<128, 4>` | 1 | 197.1 | 23 | 24 | 81 | 38 | 110 | 10 | 81 | long_scoreboard 6.55, barrier 4.71, lg_throttle 1.05 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 1 | 107.5 | 40 | 35 | 91 | 55 | 72 | 4 | 59 | long_scoreboard 9.47, not_selected 1.43, short_scoreboard 1.35 |
| `fm_tf32x3_rr_64x64` | 2 | 62.4 | 73 | 28 | 35 | 21 | 128 | 91 | 0 | math_pipe_throttle 4.31, selected 1.00, wait 0.76 |
| `fm_tf32x3_rc_64x64` | 1 | 22.7 | 68 | 28 | 42 | 21 | 132 | 63 | 0 | math_pipe_throttle 4.39, selected 1.00, long_scoreboard 0.94 |
| `sgemm32_kernel<1>` | 2 | 16.9 | 36 | 39 | 24 | 20 | 77 | 79 | 0 | short_scoreboard 1.00, selected 1.00, mio_throttle 0.76 |
| `sgemm64_kernel<1>` | 2 | 16.4 | 33 | 22 | 31 | 8 | 101 | 58 | 0 | selected 1.00, short_scoreboard 0.30, long_scoreboard 0.25 |
| `sgemm32_kernel<0>` | 3 | 15.7 | 16 | 17 | 40 | 7 | 77 | 29 | 4 | selected 1.00, long_scoreboard 0.77, short_scoreboard 0.58 |

## omat0m_spec_srtio3

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `wprime_backward_inter1_rcv` | 1 | 3,163.2 | 72 | 78 | 18 | 64 | 64 | 95 | 12 | not_selected 3.90, long_scoreboard 1.57, dispatch_stall 1.42 |
| `spline_forward_inter1<5, 0>` | 1 | 1,639.5 | 66 | 51 | 25 | 41 | 96 | 91 | 59 | long_scoreboard 1.60, not_selected 1.46, wait 1.29 |
| `sgemm128_kernel<8, 1, 1, 0, 0>` | 14 | 1,249.0 | 62 | 24 | 52 | 30 | 128 | 43 | 0 | not_selected 2.15, selected 1.00, dispatch_stall 0.87 |
| `tabled_outer_sum0_backward_kernel` | 1 | 525.7 | 72 | 49 | 38 | 37 | 80 | 86 | 36 | long_scoreboard 2.51, wait 1.87, short_scoreboard 1.50 |
| `sgemm128_kernel<4, 1, 1, 0, 0>` | 10 | 347.2 | 62 | 32 | 40 | 31 | 105 | 57 | 0 | not_selected 1.77, selected 1.00, dispatch_stall 0.67 |
| `tabled_outer_sum0_forward_kernel` | 1 | 227.5 | 56 | 60 | 57 | 40 | 96 | 79 | 60 | long_scoreboard 4.57, short_scoreboard 1.50, wait 1.10 |
| `sc1_fused` | 1 | 202.4 | 22 | 22 | 85 | 38 | 110 | 52 | 88 | long_scoreboard 6.91, barrier 4.36, selected 1.00 |
| `sc0_backward` | 1 | 190.1 | 35 | 25 | 88 | 38 | 128 | 46 | 63 | long_scoreboard 7.11, selected 1.00, not_selected 0.87 |
| `fm_tf32x3_rr_64x64` | 5 | 123.4 | 70 | 29 | 33 | 20 | 128 | 63 | 0 | math_pipe_throttle 4.02, selected 1.00, wait 0.75 |
| `sc0_forward` | 1 | 113.5 | 32 | 36 | 92 | 54 | 72 | 4 | 59 | long_scoreboard 14.95, not_selected 1.21, short_scoreboard 1.17 |
| `fm_tf32x3_rc_64x64` | 1 | 22.7 | 68 | 30 | 27 | 21 | 132 | 58 | 0 | math_pipe_throttle 4.41, long_scoreboard 1.03, selected 1.00 |
| `sgemm32_kernel<1>` | 2 | 17.0 | 36 | 39 | 23 | 20 | 77 | 77 | 0 | short_scoreboard 1.01, selected 1.00, mio_throttle 0.75 |
| `sgemm64_kernel<1>` | 2 | 16.3 | 33 | 21 | 25 | 8 | 101 | 61 | 0 | selected 1.00, short_scoreboard 0.31, long_scoreboard 0.23 |
| `sgemm32_kernel<0>` | 3 | 15.4 | 16 | 16 | 41 | 6 | 77 | 31 | 4 | selected 1.00, long_scoreboard 0.88, short_scoreboard 0.56 |

## polar1m_nbcl5

| Kernel | launches | us | SM % | L2 % | DRAM % | warps % | regs | L2 hit % | L1 hit % | top stalls (warps per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sgemm128_kernel<8, 1, 1, 0, 0>` | 38 | 4,452.7 | 63 | 22 | 27 | 30 | 128 | 70 | 5 | not_selected 2.10, selected 1.00, dispatch_stall 0.86 |
| `sgemm128_kernel<8, 1, 1, 7, 0>` | 2 | 1,220.5 | 65 | 22 | 42 | 33 | 128 | 65 | 1 | not_selected 2.19, dispatch_stall 1.15, selected 1.00 |
| `sgemm128_kernel<8, 1, 1, 5, 0>` | 2 | 972.2 | 59 | 20 | 42 | 32 | 128 | 64 | 2 | not_selected 2.17, dispatch_stall 1.13, selected 1.00 |
| `gprod_bwd_dd662f9d3a2b5b2b` | 2 | 804.5 | 28 | 18 | 84 | 22 | 160 | 3 | 43 | long_scoreboard 5.64, selected 1.00, not_selected 0.65 |
| `sgemm128_kernel<8, 1, 1, 3, 0>` | 2 | 714.4 | 49 | 19 | 34 | 31 | 128 | 57 | 2 | not_selected 2.13, long_scoreboard 1.22, dispatch_stall 1.11 |
| `inv_bwd_kernel` | 8 | 704.4 | 18 | 29 | 91 | 92 | 20 | 24 | 15 | long_scoreboard 42.53, wait 2.80, short_scoreboard 1.66 |
| `sf_kernel_tiled<16, 8, 16>` | 2 | 659.0 | 71 | 7 | 2 | 46 | 80 | 97 | 10 | short_scoreboard 9.19, wait 3.53, tex_throttle 2.30 |
| `grad_pass_kernel<2, 1, 0>` | 2 | 534.1 | 34 | 3 | 3 | 17 | 80 | 74 | 45 | long_scoreboard 1.67, short_scoreboard 1.45, wait 1.13 |
| `gconv_bwd_L1<1, 128, 1>` | 1 | 467.8 | 70 | 80 | 34 | 58 | 62 | 92 | 3 | not_selected 2.94, long_scoreboard 2.10, dispatch_stall 1.08 |
| `sgemm128_kernel<4, 0, 1, 0, 0>` | 3 | 467.4 | 68 | 27 | 30 | 31 | 105 | 68 | 0 | not_selected 1.84, selected 1.00, dispatch_stall 0.70 |
| `sgemm64_kernel<1>` | 19 | 422.6 | 44 | 26 | 46 | 12 | 101 | 57 | 1 | selected 1.00, short_scoreboard 0.33, dispatch_stall 0.29 |
| `gconv_fwd_L1<5, 0, 640, 1, 128>` | 1 | 345.5 | 46 | 77 | 35 | 40 | 96 | 90 | 16 | long_scoreboard 5.51, selected 1.00, wait 0.91 |
| `sf_kernel_tiled<8, 8, 8>` | 2 | 345.1 | 68 | 6 | 1 | 61 | 64 | 95 | 51 | short_scoreboard 6.82, tex_throttle 3.87, wait 3.04 |
| `gprod_fwd_dd662f9d3a2b5b2b` | 2 | 343.9 | 37 | 33 | 95 | 56 | 72 | 3 | 58 | long_scoreboard 13.51, not_selected 1.05, selected 1.00 |
| `inv_kernel` | 4 | 321.3 | 12 | 37 | 95 | 97 | 30 | 0 | 5 | long_scoreboard 95.39, wait 2.68, selected 1.00 |
| `feat_pass_kernel<0>` | 2 | 295.5 | 36 | 5 | 9 | 17 | 80 | 89 | 36 | long_scoreboard 1.92, wait 1.03, short_scoreboard 1.00 |

