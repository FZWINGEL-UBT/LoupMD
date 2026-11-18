# In-depth profiles, 18 November 2026 (after S122-S128)

Profiles of `round6-hardening` `f799b54` (library `libloupmd_s128b.so`, sha256 `e67297afde5d...`; the later commits
change documentation only) on the local **RTX 5060 Ti** (sm_120, 36 SMs, FP32 peak about 23.7 TFLOP/s). They are the
input for the next round of ideas ([PROMPT.md](PROMPT.md)). Every number below is from this GPU unless marked.

## Contents

| File | What |
| --- | --- |
| [summary.md](summary.md) | Generated tables: MD step times, the share of kernel time per category for every case, each case's top kernels, and the ncu tables |
| `nsys/<case>_cuda_gpu_kern_sum.csv` | Per-kernel totals of an MD segment (`nsys stats` kernel summary), `<case>.steps` the segment's step count |
| `gemm/<case>.txt` | Every GEMM shape of one MD step with the autotune's pick and its tuned time (`LOUP_DEBUG_GEMM=1`, fresh plans) |
| `ncu/<case>.csv.gz`, `ncu/<case>_raw.csv.gz` | ncu sections (details page, with ncu's rule notes) and every collected metric (raw page) of the top kernels of one force call (gzip; `zcat` to read) |
| [edge_stats.md](edge_stats.md) | Edges per atom, distance bands and species-pair shares of the benchmark systems (CPU) |
| [weight_structure.md](weight_structure.md) | Live output columns and input rows of every family's projection weights (CPU, from the cached bundles) |
| `scripts/` | `job_profiles.sh` (nsys / gemm / ncu parts, `ONLY=` label filter), `run_chunks.sh` (the parts in short local-GPU reservations), `md_prof.py` (MD segment), `prof_eval.py` (one force call), `summarize.py`, `edge_stats.py`, `weight_structure.py` |

## Method

- **nsys** (`job_profiles.sh nsys`): `md_prof.py` runs 100 warm-up Langevin steps (300 K, 1 fs, friction 0.01 / fs,
  skin 0.5 A, forces only), then captures STEPS more (16-256 by size) inside `cudaProfilerStart / Stop`, with
  `--cuda-graph-trace=node`. Times are per MD step, rebuilds included (their share is the "neighbour list" row).
  Kernel time is **summed over streams**: forked GEMM lanes overlap, so it exceeds the step's wall time.
- **GEMM shapes** (`job_profiles.sh gemm`): 4 MD steps with fresh plans and `LOUP_DEBUG_GEMM=1`; each line is a
  shape, the kernel the autotune picked (`own*` the SIMT kernels, `lt*` cuBLASLt, `hyb*` the 3xTF32 hybrid cubin) and
  its tuned time (one call, `cold=1`: timed with a cold L2).
- **ncu** (`job_profiles.sh ncu`): one force call (`prof_eval.py`: a rattled cell, 20 unprofiled warm-up calls, then
  one call in `cudaProfilerStart / Stop`), sections SpeedOfLight, MemoryWorkloadAnalysis (+ tables), WarpStateStats,
  SchedulerStats, Occupancy, LaunchStats, InstructionStats for the kernels matching each case's regex. Kernels are
  serialized and replayed, so their durations are a little above the in-graph ones.
- **CPU**: `edge_stats.py` (ASE neighbour lists of the same structures), `weight_structure.py` (the bundles'
  float32 weights).

## MD step time

| Family | System | Atoms | ms / step | Kernel time per step (us) | Launches per step |
| --- | --- | ---: | ---: | ---: | ---: |
| OMAT-0 medium, specialised | SrTiO3 | 5,000 | 7.98 | 8,648 | 59 |
| OMAT-0 medium, generic | SrTiO3 | 5,000 | 7.80 | 8,530 | 59 |
| MP-0 medium | SrTiO3 | 5,000 | **3.81** (8.20 at the start of the round) | 4,021 | 45 |
| MP-0 medium | NbCl5 | 1,944 | **0.89** (1.72) | 961 | 40 |
| MPtrj-2022.9 | SrTiO3 | 5,000 | 10.82 | 11,627 | 50 |
| OMAT-0 small | SrTiO3 | 5,000 | 4.50 | 4,950 | 57 |
| MH-1 (`omat_pbe`) | SrTiO3 | 5,000 | 33.04 | 37,575 | 70 |
| MH-1 (`omat_pbe`) | NaCl216 | 216 | 1.42 | 2,326 | 70 |
| OFF23-S | water216 | 648 | 0.40 | 521 | 49 |
| OFF23-M | water216 | 648 | 0.92 | 1,186 | 51 |
| OFF24-M | water | 5,184 | 9.12 | 9,856 | 53 |
| OFF23-L | water216 | 648 | 3.04 | 3,654 | 57 |
| OMOL-0 XL | water216 | 648 | 22.61 | 28,474 | 140 |
| POLAR-1-S | NaCl216 | 216 | 1.24 | 1,599 | 192 |
| POLAR-1-S | NbCl5 | 1,944 | 7.96 | 8,944 | 185 |
| POLAR-1-M | NaCl216 | 216 | 2.94 | 4,761 | 300 |
| POLAR-1-M | NbCl5 | 1,944 | 22.09 | 26,400 | 298 |
| POLAR-1-L | NaCl216 | 216 | 5.83 | 10,134 | 397 |

## Where the time goes (share of summed kernel time)

| Case | conv bwd | conv fwd | GEMM | layer 0 | product | gates | POLAR k-space | POLAR dense | rest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OMAT-0 M specialised SrTiO3 | 34.4% | 20.6% | 27.5% | 9.4% | 6.5% | - | - | - | 1.5% |
| OMAT-0 M generic SrTiO3 | 35.0% | 19.8% | 28.5% | 8.5% | 6.9% | - | - | - | 1.3% |
| MP-0 M SrTiO3 | 26.8% | 16.4% | 26.2% | 16.8% | 10.7% | - | - | - | 3.1% |
| MP-0 M NbCl5 | 16.9% | 12.0% | 43.4% | 10.7% | 12.9% | - | - | - | 4.1% |
| MPtrj SrTiO3 | 21.2% | 15.9% | 48.1% | 3.6% | 10.6% | - | - | - | 0.7% |
| OMAT-0 S SrTiO3 | 22.7% | 14.9% | 34.5% | 14.8% | 11.0% | - | - | - | 2.1% |
| MH-1 SrTiO3 | 8.2% | 4.7% | 60.8% | 2.3% | 7.3% | 16.4% | - | - | 0.3% |
| MH-1 NaCl216 | 3.0% | 2.3% | 83.5% | 1.4% | 3.3% | 5.5% | - | - | 1.0% |
| OFF23-S water216 | 11.9% | 9.0% | 55.5% | 9.4% | 5.5% | - | - | - | 8.8% |
| OFF23-M water216 | 23.1% | 17.4% | 44.8% | 6.0% | 4.5% | - | - | - | 4.2% |
| OFF24-M water (5,184) | 36.0% | 25.0% | 22.3% | 7.6% | 6.6% | - | - | - | 2.4% |
| OFF23-L water216 | 29.0% | 20.0% | 44.3% | 2.0% | 3.2% | - | - | - | 1.6% |
| OMOL-0 XL water216 | 7.2% | 4.5% | 74.1% | 0.9% | 4.3% | 8.7% | - | - | 0.3% |
| POLAR-1-S NaCl216 | - | - | 76.5% | 2.2% | 1.5% | 3.9% | 6.9% | 7.1% | 1.8% |
| POLAR-1-S NbCl5 | - | - | 64.8% | 1.6% | 6.0% | 12.1% | 9.3% | 5.4% | 0.8% |
| POLAR-1-M NaCl216 | 1.5% | 1.2% | 83.4% | 0.7% | 1.7% | 2.8% | 3.8% | 4.2% | 0.7% |
| POLAR-1-M NbCl5 | 2.0% | 1.3% | 67.6% | 0.6% | 4.5% | 9.0% | 7.2% | 7.5% | 0.4% |
| POLAR-1-L NaCl216 | 2.9% | 2.0% | 86.4% | 0.3% | 1.7% | 2.2% | 1.8% | 2.3% | 0.4% |

"conv" = the interaction >= 1 convolutions (`gconv_*`, specialised `spline_forward_inter1` / `wprime_backward_inter1_rcv`);
"layer 0" = the species-embedding layer (`first_fwd` / `first_bwd_edge`, specialised `tabled_outer_sum0_*`);
"product" = the symmetric contractions (`gprod_*`, `sc0_*`, `sc1_fused`); "gates" = the nonlinear blocks' gate kernels
(MH-1, OMOL-0, POLAR). Per-kernel tables: [summary.md](summary.md).

## GEMMs

The autotuned GEMMs of the large cases run at **12-17 TFLOP/s** in FP32 (`gemm/*.txt`; 2 M N K / tuned time), i.e.
55-72% of this GPU's FP32 peak: they are compute-bound, so a GEMM idea has to remove FLOPs (or move them to tensor
cores without losing FP32 accuracy; see the ledger's `gemm-hybrid-cubin` for what passed and what failed). Largest
shapes, one call each (M = atoms x (2l + 1) rows; T = transposed B):

| Case | Largest shapes (us per call) | Sum of the step's shapes |
| --- | --- | ---: |
| MH-1 SrTiO3 | 35,000 x 512 x 512 T 1,157 / N 1,115; 25,000 x 512 x 512 T 826 / N 774; 25,000 x 384 x 512 T 646; 25,000 x 512 x 384 N 619 | 13.5 ms |
| OMOL-0 XL water216 | 4,536 x 1,024 x 1,024 T 600 / N 552; 3,240 x 1,024 x 1,024 T 437 / N 436 | 6.7 ms |
| POLAR-1-M NbCl5 | 13,608 x 512 x 512 T 455 / N 427; 9,720 x 512 x 512 T 341 / N 331; 1,944 x 2,048 x 512 N 273 | 6.5 ms |
| OMAT-0 M generic SrTiO3 | 25,000 x 128 x 384 N 195; 35,000 x 128 x 256 N 178; 25,000 x 384 x 128 T 176 | 1.7 ms |
| OFF23-L water216 | 3,240 x 224 x 1,120 N 142; 4,536 x 224 x 896 N 128 | 1.0 ms |
| MP-0 M SrTiO3 | 35,000 x 128 x 192 N 134 / 35,000 x 192 x 128 T 131 | 0.9 ms |

(The tuned sums are one call per shape and exclude lane overlap, so they differ from the nsys GEMM row.)

## ncu: what the top kernels are limited by

One force call per case (OMAT-0 medium specialised and generic SrTiO3, MP-0 medium SrTiO3, MH-1 SrTiO3, OFF23-L
water216, POLAR-1-M NbCl5); full tables in [summary.md](summary.md). ncu replays each kernel with cold caches, so its
durations exceed the in-graph ones (MP-0's pruned conv backward: 1.72 ms in ncu, 1.08 ms per MD step in nsys).

| Kernel class | Limiter (ncu) | Evidence |
| --- | --- | --- |
| Interaction-1 conv forward (`gconv_fwd_L1`, `spline_forward_inter1`) | L2 throughput | 78-85% of L2 peak, 95% L2 hit, long-scoreboard stalls 3.5-3.9 (33-41% warps active); MP-0's pruned forward 83% L2, 98% warps active |
| Interaction-1 conv backward (`gconv_bwd_L1`, `wprime_backward_inter1_rcv`) | Issue and L2 together | 81-82% of L2 peak, SM 75-76%, not-selected 3.9-4.0 (64% warps active, 62-64 registers). Per-edge atomic reductions into the sender's x-gradient rows: 64 L2 sectors per edge, 36% of all L2 sectors in MP-0's pruned backward (`lts__t_sectors_srcunit_tex_op_red.sum`), which is L2-latency-bound (89% L2, long scoreboard 14) |
| Degree-2 conv backward (`gconv_bwd_L2`, OFF23-L) | Occupancy | 126 registers, 23% warps active, L2 only 58%; the companion `gconv_bwd_L2lp` (62 registers) is at 80% L2 |
| Layer 0 (`first_fwd`, `first_bwd_edge`, `tabled_outer_sum0_*`) | Mixed: latency and L2 / DRAM | `first_bwd_edge` SM 71-74%, L2 43-51%, DRAM 40%; `first_fwd` L2 54-83%, DRAM 56-66% |
| GEMMs (`sgemm128`, cuBLASLt, cutlass SIMT) | FP32 issue | SM 58-71%, not-selected 1.7-2.4 (issue-saturated), L2 18-37%; 12-17 TFLOP/s |
| Products (`gprod_*`, `sc0_*`, `sc1_fused`) and gates (`gate_fwd` / `gate_bwd`) | DRAM bandwidth | 80-95% of DRAM peak, L2 hit 0-17%; gates show long-scoreboard stalls of 84-129 |
| POLAR elementwise (`inv_kernel`, `inv_bwd_kernel`) | DRAM bandwidth | 94-95% of DRAM peak |
| POLAR k-space (`sf_kernel_tiled`, `grad_pass_kernel`, `feat_pass_kernel`) | Latency at low occupancy | `sf_kernel_tiled`: SM 67-70%, short-scoreboard 6.8-9.2, tex-throttle 2.3-3.8, L2 / DRAM < 10%; `grad_pass` / `feat_pass`: 17% warps active, SM 34-35%, L2 5-6% |
| Fused readouts (`gprod_ro1s_fused`, `ro_bwd_kernel`) | Shared memory and barriers | barrier 19.7 / mio-throttle 18.8 (MP-0 `ro1s`); mio-throttle 58.5 / lg-throttle 46.3 (POLAR `ro_bwd`) |

## Structure in the inputs

- [edge_stats.md](edge_stats.md): SrTiO3 has 73.9 edges per atom at 6 A (369,266 directed edges), 91% of them at
  2.75 A or more and 32% beyond 5 A; NbCl5 27.0; NaCl216 32.0; water216 / water 84.6 at 6 A and 53.7 at 5 A (the OFF
  cutoff). Same-species edges are 44% (SrTiO3) to 65% (NbCl5).
- [weight_structure.md](weight_structure.md): live output columns of every projection matrix. Exact zeros made
  S125 possible (MP-0 medium's second-interaction W_lin_1..3 are entirely zero). Other near-empty matrices:
  - MP-0 medium layer 0: W_lin_1..3 have 33 / 37 / 35 nonzero columns of 128 (exact zeros in the others).
  - MPtrj-2022.9 layer 1: W_lin_1..3 have 1 / 2 / 12 nonzero columns of 256, and only 1 / 1 / 2 above 1e-6 of the
    layer's largest W_lin_0 entry; layer 0: W_lin_1..3 82 / 82 / 69 of 256 columns above that level; layer 1's W_lin_0
    86 of 256. The remaining entries are 1e-17 to 1e-20.
  - OMAT-0, MP-0b3, MPA-0, MatPES, MH-1, OFF, OMOL-0, POLAR-1: every column of W_lin / W_res live.

## Caveats

- One GPU (sm_120). The workstations (sm_89, sm_86) have no performance-counter permission (`ERR_NVGPUCTRPERM`), and
  the A100 (Alex) was under maintenance; earlier A100 profiles are in the ledger (`meas-*` entries).
- Kernel time is summed over streams (see Method); shares are of that sum.
- The ncu force call is a rattled cell (not an MD frame); its edge counts match the MD frames' to within a few percent.
