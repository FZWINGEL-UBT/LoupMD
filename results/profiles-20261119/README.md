# In-depth profiles, 19 November 2026 (after S129-S140)

Profiles of `round6-hardening` `93e47fd` (library `libloupmd_p1119.so`, sha256 `fe7d3ebcb11d...`, built from that
commit) on the local **RTX 5060 Ti** (sm_120, 36 SMs, FP32 peak about 23.7 TFLOP/s), with the method of
[profiles-20261118](../profiles-20261118/README.md) so that the two sets compare case by case. Every number below is
from this GPU unless marked.

## Contents

| File | What |
| --- | --- |
| [summary.md](summary.md) | Generated tables: MD step times, the share of kernel time per category for every case, each case's top kernels, and the ncu tables |
| `nsys/<case>_cuda_gpu_kern_sum.csv` | Per-kernel totals of an MD segment (`nsys stats` kernel summary), `<case>.steps` the segment's step count, `steps.txt` the step times |
| `gemm/<case>.txt` | Every GEMM shape of one MD step with the autotune's pick and its tuned time (`LOUP_DEBUG_GEMM=1`, fresh plans) |
| `ncu/<case>.csv.gz`, `ncu/<case>_raw.csv.gz` | ncu sections (details page) and every collected metric (raw page) of the top kernels of one force call (gzip); `ncu/variants.txt` the interaction-1 forward variant each case ran |
| `scripts/` | `job_profiles.sh` (nsys / gemm / ncu parts, `ONLY=` label filter), `run_chunks.sh` (the parts in short local-GPU reservations), `md_prof.py`, `prof_eval.py`, `summarize.py`, `compare.py` (this set against another) |

The CPU tables of the inputs ([edge_stats.md](../profiles-20261118/edge_stats.md),
[weight_structure.md](../profiles-20261118/weight_structure.md)) describe the structures and the models' weights, which
have not changed; they stay in the 18 November set.

## Method

As in [profiles-20261118](../profiles-20261118/README.md#method): nsys over an MD segment after 100 warm-up Langevin
steps (300 K, 1 fs, skin 0.5 A, forces only, rebuilds included), kernel time summed over streams; GEMM shapes from 4 MD
steps with fresh plans; ncu sections of one force call after 20 warm-up calls. Two differences:

- **ncu runs the MD variant.** A lone force call does not run the per-system timing of the interaction-1 forward
  variants that an MD rebuild does, so the 18 November ncu of OMAT-0 generic profiled the untuned default (R1 paired)
  while MD ran R5. Here each ncu case pins the variant its own MD profile ran (`ncu/variants.txt`: R5 for OMAT-0 medium,
  MP-0, MH-1 and POLAR-1-M; MPtrj's C = 256 layers have a single variant, so the pin changes nothing there).
- **Added cases:** MPtrj-2022.9 NbCl5 (nsys) and MPtrj-2022.9 SrTiO3 (GEMM shapes, ncu), for the compacted kernels
  of S130 / S132.

## MD step time

One profiled run per case (`scripts/compare.py ../profiles-20261118 .`). Single runs scatter by 1-2%, so the
OMAT-0 medium, OFF23-M and POLAR-1-S NaCl216 rows (+0.4% to +1.9%) are within noise; the library ABBA comparisons of
every change are in the ledger (`OPTIMISATION_LEDGER.md`, S129-S140).

| Case | Atoms | ms / step 18 Nov | ms / step 19 Nov | 19 / 18 Nov | kernel us / step 18 Nov | 19 Nov |
|---|---:|---:|---:|---:|---:|---:|
| OMAT-0 medium, specialised, SrTiO3 | 5,000 | 7.98 | 8.02 | 1.004 | 8,648 | 8,739 |
| OMAT-0 medium, generic, SrTiO3 | 5,000 | 7.80 | 7.89 | 1.010 | 8,530 | 8,624 |
| MP-0 medium SrTiO3 | 5,000 | 3.81 | **3.19** | 0.835 | 4,021 | 3,367 |
| MP-0 medium NbCl5 | 1,944 | 0.89 | **0.70** | 0.781 | 961 | 773 |
| MPtrj-2022.9 SrTiO3 | 5,000 | 10.82 | **8.40** | 0.776 | 11,627 | 8,924 |
| MPtrj-2022.9 NbCl5 | 1,944 | - | 2.11 | - | - | 2,524 |
| OMAT-0 small SrTiO3 | 5,000 | 4.50 | 4.38 | 0.973 | 4,950 | 4,861 |
| MH-1 (`omat_pbe`) SrTiO3 | 5,000 | 33.04 | **30.91** | 0.935 | 37,575 | 35,309 |
| MH-1 (`omat_pbe`) NaCl216 | 216 | 1.42 | 1.38 | 0.975 | 2,326 | 2,365 |
| OFF23-S water216 | 648 | 0.40 | 0.37 | 0.937 | 521 | 486 |
| OFF23-M water216 | 648 | 0.92 | 0.93 | 1.019 | 1,186 | 1,186 |
| OFF24-M water | 5,184 | 9.12 | 9.06 | 0.994 | 9,856 | 9,783 |
| OFF23-L water216 | 648 | 3.04 | 3.03 | 0.996 | 3,654 | 3,654 |
| OMOL-0 XL water216 | 648 | 22.61 | **21.73** | 0.961 | 28,474 | 27,604 |
| POLAR-1-S NaCl216 | 216 | 1.24 | 1.25 | 1.013 | 1,599 | 1,621 |
| POLAR-1-S NbCl5 | 1,944 | 7.96 | 7.87 | 0.989 | 8,944 | 8,585 |
| POLAR-1-M NaCl216 | 216 | 2.94 | 2.86 | 0.973 | 4,761 | 4,682 |
| POLAR-1-M NbCl5 | 1,944 | 22.09 | **21.13** | 0.957 | 26,400 | 25,091 |
| POLAR-1-L NaCl216 | 216 | 5.83 | **5.18** | 0.890 | 10,134 | 9,024 |

## Where the time goes (share of summed kernel time)

| Case | conv bwd | conv fwd | GEMM | layer 0 | product | gates | POLAR k-space | POLAR dense | rest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OMAT-0 M specialised SrTiO3 | 34.0% | 20.8% | 27.8% | 9.4% | 6.5% | - | - | - | 1.5% |
| OMAT-0 M generic SrTiO3 | 34.9% | 19.9% | 28.8% | 8.1% | 7.0% | - | - | - | 1.3% |
| MP-0 M SrTiO3 | 23.9% | 19.9% | 25.7% | 18.4% | 8.7% | - | - | - | 3.4% |
| MP-0 M NbCl5 | 14.0% | 15.1% | 40.1% | 12.2% | 13.1% | - | - | - | 5.5% |
| MPtrj SrTiO3 | 27.4% | 21.0% | 33.6% | 4.3% | 12.8% | - | - | - | 0.9% |
| MPtrj NbCl5 | 18.5% | 14.1% | 51.7% | 2.4% | 12.0% | - | - | - | 1.3% |
| OMAT-0 S SrTiO3 | 23.5% | 15.3% | 33.6% | 14.3% | 11.2% | - | - | - | 2.2% |
| MH-1 SrTiO3 | 9.0% | 5.1% | 71.5% | 2.5% | 7.7% | 3.8% | - | - | 0.3% |
| MH-1 NaCl216 | 3.1% | 2.3% | 85.3% | 1.4% | 3.1% | 3.9% | - | - | 0.9% |
| OFF23-S water216 | 12.7% | 9.7% | 53.1% | 9.3% | 5.5% | - | - | - | 9.7% |
| OFF23-M water216 | 22.4% | 17.5% | 46.0% | 5.5% | 4.3% | - | - | - | 4.3% |
| OFF24-M water (5,184) | 36.3% | 25.2% | 22.6% | 7.3% | 6.1% | - | - | - | 2.6% |
| OFF23-L water216 | 29.2% | 20.1% | 44.3% | 1.8% | 3.2% | - | - | - | 1.4% |
| OMOL-0 XL water216 | 7.6% | 4.8% | 79.0% | 1.0% | 4.2% | 3.1% | - | - | 0.3% |
| POLAR-1-S NaCl216 | - | - | 77.1% | 2.0% | 1.5% | 3.4% | 7.1% | 7.2% | 1.7% |
| POLAR-1-S NbCl5 | - | - | 70.9% | 1.7% | 6.1% | 5.0% | 9.6% | 5.9% | 0.8% |
| POLAR-1-M NaCl216 | 1.5% | 1.2% | 84.4% | 0.7% | 1.7% | 1.9% | 3.7% | 4.2% | 0.7% |
| POLAR-1-M NbCl5 | 1.9% | 1.4% | 71.8% | 0.6% | 4.8% | 3.5% | 7.8% | 7.9% | 0.3% |
| POLAR-1-L NaCl216 | 3.4% | 2.3% | 85.4% | 0.4% | 1.9% | 1.7% | 2.1% | 2.5% | 0.4% |

Categories as in the 18 November set. The gate fusion (S137) moved the kind-4 gate backward into the `W_2^T` GEMMs'
epilogue, so that work now counts as GEMM.

## What changed since 18 November (kernel us per step, 19 Nov minus 18 Nov)

| Case | conv bwd | conv fwd | layer 0 | product | gates | GEMM | rest | Changes behind it |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MP-0 M SrTiO3 | -270 | +10 | -57 | -136 | 0 | -190 | -11 | S135 sender-owned pruned backward; S132 layer-0 compaction; S131 / S140 layer 0 |
| MP-0 M NbCl5 | -55 | +1 | -8 | -23 | 0 | -107 | +4 | as above |
| MPtrj SrTiO3 | -21 | +28 | -31 | -89 | 0 | -2,590 | 0 | S130 / S132 live-channel compaction (thin projections) |
| MH-1 SrTiO3 | +69 | +41 | +18 | -17 | -4,808 | +2,427 | +4 | S137: gate backward in the GEMM epilogue, layer 0's gated store |
| OMOL-0 XL water216 | +47 | +42 | +14 | -64 | -1,612 | +706 | -5 | S137 |
| POLAR-1-M NbCl5 | -42 | 0 | +1 | +16 | -1,494 | +175 | +35 | S137; S136 degree-1 fold |
| POLAR-1-S NbCl5 | 0 | 0 | +1 | -16 | -653 | +297 | +12 | S137 |
| POLAR-1-L NaCl216 | +4 | +12 | 0 | -1 | -72 | -1,054 | +2 | S136 hidden folds and zero-source skips |
| OMAT-0 S SrTiO3 | +16 | +5 | -39 | +1 | 0 | -74 | +2 | S131 / S140 layer 0 |
| OFF24-M water | -2 | +3 | -43 | -54 | 0 | +14 | +10 | S131 / S140 layer 0 |
| OMAT-0 M generic SrTiO3 | +27 | +25 | -24 | +11 | 0 | +56 | -2 | S140 layer 0; the rest single-run scatter and plan picks |

The full per-case table is printed by `scripts/compare.py`.

## GEMMs

Largest shapes of one MD step (us per call, the autotune's pick; M = atoms x (2l + 1) rows, T = transposed B):

| Case | Largest shapes | Sum of the step's shapes |
| --- | --- | ---: |
| MH-1 SrTiO3 | 35,000 x 512 x 512 T 1,157 / N 1,110; 25,000 x 512 x 512 T 828 / N 773 | 13.5 ms |
| OMOL-0 XL water216 | 4,536 x 1,024 x 1,024 T 600 / N 552; 3,240 x 1,024 x 1,024 N 441 / T 441 | 6.8 ms |
| POLAR-1-M NbCl5 | 13,608 x 512 x 512 T 455 / N 423; 9,720 x 512 x 512 T 341 / N 335 | 6.4 ms |
| MPtrj SrTiO3 | 25,000 x 8 x 768 N 208; 35,000 x 16 x 512 N 189; 25,000 x 768 x 8 T 187; 35,000 x 512 x 16 T 175 | 2.4 ms (not logged on 18 Nov; the nsys GEMM time fell by 2,590 us per step) |
| OMAT-0 M generic SrTiO3 | 25,000 x 128 x 384 N 193; 35,000 x 128 x 256 N 179; 25,000 x 384 x 128 T 176 | 1.7 ms |
| OFF23-L water216 | 3,240 x 224 x 1,120 N 143; 4,536 x 224 x 896 N 129 | 1.0 ms |
| MP-0 M SrTiO3 | 35,000 x 40 x 192 N 91; 25,000 x 40 x 192 N 70; 35,000 x 192 x 40 T 66 | 0.6 ms (0.9 ms on 18 Nov) |

The tuned sums are one call per shape and exclude lane overlap. The large square shapes run at 12-17 TFLOP/s, as
before. MPtrj's compacted projections are thin now: 25,000 x 8 x 768 in 208 us reads the conv's 77 MB of mid rows at
about 370 GB/s, i.e. they are bound by that read, not by arithmetic.

## ncu: what the top kernels are limited by

One force call per case (OMAT-0 medium specialised and generic SrTiO3, MP-0 medium SrTiO3, MPtrj SrTiO3, MH-1
SrTiO3, OFF23-L water216, POLAR-1-M NbCl5), with the MD variants; full tables in [summary.md](summary.md).

| Kernel class | Limiter (ncu) | Evidence |
| --- | --- | --- |
| Interaction-1 conv forward, five-receiver R5 on the 5,000-atom cells (`gconv_fwd_L1<5,...>` OMAT-0 generic and MH-1, `spline_forward_inter1<5,0>`) | Issue and latency, not L2 | SM 66-68%, L2 51-61%, L1 hit 54-59% (the five receivers share table rows in L1), not-selected 1.5-1.9, long scoreboard 1.6-1.7; the one-receiver R1 paired profiled on 18 Nov ran at L2 81%, L1 hit 5.6% |
| The same R5 on POLAR-1-M NbCl5 (1,944 atoms) and MP-0's pruned R5 (`gconv_fwd_L1s`, two paths) | L2 and latency | POLAR-1-M NbCl5: L2 77%, L1 hit 16%, long scoreboard 5.5; MP-0 pruned: L2 71%, L1 hit 31%, long scoreboard 9.3 |
| One-receiver conv forward at C = 256 (MPtrj, `gconv_fwd_L1<1,0,256,...>`: no multi-receiver variant at C = 256) | L2 throughput | L2 85%, L1 hit 12%, long scoreboard 7.1 |
| Interaction-1 conv backward (`gconv_bwd_L1`, `wprime_backward_inter1_rcv`) | Issue and L2 together | SM 70-76%, L2 78-82%, not-selected 2.9-4.0 (58-64% warps active) |
| MP-0's sender-owned pruned backward (`gconv_bwd_L1sr`, S135) | Latency | SM 46%, L2 59%, long scoreboard 6.0, 41% warps active (96 registers) |
| Layer-0 edge backward (`first_bwd_edge`, after S140) | Latency, then the LSU pipe | LSU pipe 59% (OMAT-0 generic) / 66% (MP-0), down from 71% / 74%; long scoreboard 2.0-3.7 the top stall; FMA pipe 30-34% |
| GEMMs, including MH-1's / POLAR's gate-fused `sgemm128<8,1,1,2l+1,...>` (S137) | FP32 issue | SM 46-68%, not-selected 2.0-2.2, 128 registers, 33% warps active, L2 17-23% |
| Gate forward (`gate_fwd_kernel`, MH-1: layer 0's degree 0 and all of layer 1, whose projections are cuBLASLt and keep the kernel) | DRAM bandwidth | 1.2 ms over 2 launches per force call, DRAM 85%, long scoreboard 136 |
| Compact products (`gprod_fwdc` / `gprod_bwdc`, S132) | DRAM and latency | DRAM 48-72%, 149 registers in the backward (15-20% warps active), long scoreboard 2.6-7.3 |
| Compact fused readout (`gprod_ro1c_fused`, S130) | Barriers | SM 16%, barrier 8.1, long scoreboard 4.3 |
| POLAR k-space and elementwise | as on 18 Nov | see summary.md |

## Caveats

- One GPU (sm_120); the workstations (sm_89, sm_86) have no performance-counter permission.
- One run per case (see MD step time); kernel time is summed over streams; ncu replays each kernel with cold caches,
  so its durations differ from the in-graph ones.
- The ncu force call is a rattled cell, not an MD frame.
