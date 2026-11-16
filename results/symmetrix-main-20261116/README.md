# symmetrix main (b907f04) vs LoupMD: force-call timing and 3x gate, MACE-OMAT-0 medium, RTX 5060 Ti

16 November 2026.

## Versions under test

- **symmetrix** `main` at b907f04 (10 November 2026), which is v0.3.0 plus build fixes.
  - Its MACE foundation code is the 1771c84 commit ("add foundation model machinery") plus a JSON `format_version` 3 check.
  - Only the `_mace_foundation` Python module was built (`scripts/sx_build.sh`, no Kokkos).
  - Flags are those of `symmetrix/source/cpp/mace_foundation/CMakeLists.txt` in a Release build:
    - C++20, `-O3 -march=native -fopenmp`;
    - `-ffast-math -fno-finite-math-only` for `cpu.cpp`;
    - nvcc `--expt-relaxed-constexpr --use_fast_math -arch=sm_120`.
  - Module sha256 `c1cd8d1a…`.
- **LoupMD** installed `libloupmd.so` = `libloupmd_s106a.so` (sha256 `9e866c32…`), branch `round6-hardening`.
- **Hardware and software:** RTX 5060 Ti (driver 621.14), CUDA 13.3, mace-torch 0.3.16, torch 2.14.1.

## Model extraction

`scripts/sx_extract.py` runs the steps of `symmetrix_extract_mace`: `extract_mace_data` with `r_min` 0.5, no basis compression, written with `indent=4`. Each structure family gets its own species set.

Spline points:
- **1024:** the default of both the CLI and `extract_mace_data()` in the code.
- **256:** the default that `pair_symmetrix/README.md` states.

| Species (Z) | JSON at 1024 points | JSON at 256 points |
|---|---|---|
| 8, 22, 38 | 849 MB | 226 MB |
| 11, 17 | 431 MB | 119 MB |
| 17, 41 | 431 MB | 119 MB |

## Timing (`scripts/job_speed.sh`, `sx_speed.py`)

**Systems:** SrTiO3 6³, 10³ and 14³, each a snapshot after 500 LoupMD Langevin steps at 300 K.

**One force call** is:
- the library's Python entry point: LoupMD `LoupEngine.evaluate(x, cell, compute_stress=True)`, symmetrix `MACEFoundation.compute(x, Z, cell, pbc)`;
- positions host→device, then energy, forces and virial/stress device→host;
- symmetrix replays its own CUDA graph.

**Protocol:**
- 20 untimed warm-up calls.
- 100 timed calls, each on the snapshot plus fresh 0.003 Å Gaussian noise, so no neighbour-list rebuild and no caching.
- Reported time is the median.
- Every configuration runs in its own process, in one order and then in the reverse order.

| Implementation | 1,080 atoms ms | 5,000 atoms ms | 13,720 atoms ms | Ratio to LoupMD default (1,080 / 5,000 / 13,720) |
|---|---|---|---|---|
| LoupMD default (specialised) | 2.18 | 8.69 | 23.65 | 1.00 |
| LoupMD generic | 1.98 | 8.56 | 24.08 | 0.91 / 0.99 / 1.02 |
| symmetrix float32, 1024 points | 3.11 | 13.91 | 36.82 | 1.42 / 1.60 / 1.56 |
| symmetrix float32-mixed, 1024 points | 2.57 | 11.62 | 31.95 | 1.18 / 1.34 / 1.35 |
| symmetrix float32, 256 points | 3.01 | 13.52 | 36.11 | 1.38 / 1.56 / 1.53 |
| symmetrix float32-mixed, 256 points | 2.61 | 11.81 | 31.86 | 1.19 / 1.36 / 1.35 |

- **Repeatability:** the two passes agree within 0.7 ms (pass 1 / pass 2 per row in `speed_summary.md`).
- **Spline points barely change symmetrix's time:** 1024 and 256 are within 3.5%.
- **Cross-check:** symmetrix forces agree with LoupMD's within 3.5e-5 to 8.0e-5 eV/Å (max abs), and energies within 0.003 meV/atom.
- **LoupMD context, same snapshots:**

  | LoupMD | 1,080 atoms ms | 5,000 atoms ms | 13,720 atoms ms |
  |---|---|---|---|
  | Device-only force pass | 1.8–1.9 | 8.3–8.5 | 23.1–23.3 |
  | MD step (Langevin) | 1.8 | 8.1–8.3 | 22.6–23.0 |

- **Not measured:** symmetrix's MD loop. It has none outside LAMMPS, and the `symmetrix/mace/foundation` LAMMPS pair style was not built.

## 3x float32 gate (`scripts/job_gate.sh`, `sx_gate.py`)

**Set and rule:**
- `python/test_gate.py`'s oxide set: NaCl 2×2×2, SrTiO3 3×3×3 and NbCl5 2×2×2, two strained and rattled configurations each.
- Its cached MACE float64/float32 references.
- Its rule: dE ≤ max(3 dE32, 0.01 meV/atom), dF ≤ max(3 dF32, 1e-5), dS ≤ max(3 dS32, 1e-7).

**How each was run:**
- symmetrix through its ASE calculator (`symmetrix.experimental.MACEFoundation`, skin 1 Å).
- LoupMD through `LoupEngine.evaluate`.

| Candidate | Worst dF / dF32 | Worst dS / dS32 | Worst dE (meV/atom) | 3x gate |
|---|---|---|---|---|
| LoupMD default | 1.66 | 2.49 | 0.00224 | pass |
| LoupMD generic | 2.00 | 1.80 | 0.00189 | pass |
| symmetrix float64, 1024 | 0.03 | 0.02 | 0.00000 | pass |
| symmetrix float32, 1024 | 0.22 | 0.23 | 0.00012 | pass |
| symmetrix float32-mixed, 1024 | 0.75 | 0.85 | 0.00013 | pass |
| symmetrix float64, 256 | 2.08 | 1.30 | 0.00010 | pass |
| symmetrix float32, 256 | 2.12 | 1.63 | 0.00019 | pass |
| symmetrix float32-mixed, 256 | 2.81 | 1.52 | 0.00004 | pass |

Per-configuration rows are in `gate_summary.txt`.

- **At 1024 points, symmetrix's float32 path is roughly 8x closer to MACE float64 than LoupMD on this set:**
  - its worst ratios are 0.22 (force) and 0.23 (stress), against LoupMD's 1.66–2.00 and 1.80–2.49;
  - it is closer than MACE's own float32 (ratio 1);
  - its mixed mode stays below 1.
- **At 256 points the spline sets the error:** float64 already reaches 2.08. The mixed mode then reaches 2.81 of the 3.0 limit.
- **LoupMD's ratios are mostly its own approximations:** S103 showed the 8-byte packed tables account for most of OMAT-0's margin.
- **Not covered:** the gate structures have no pair below about 1.8 Å, so short-range spline error is not tested here.

## Round 2: equal-skin force calls and kernel profiles (`scripts/job_apples.sh`, `job_ncu.sh`, `sx_apples.py`)

### What makes it apples to apples

- Same 300 K snapshots and the same noise sequence for both libraries.
- Same neighbour-list skin, each value applied to both: 0.5 Å (LoupMD's default) and 1.0 Å (symmetrix's default).
- Same outputs: energy, forces and virial/stress.
- Same host pattern per call, from the nsys CUDA API trace: four async copies, one CUDA-graph launch, one stream sync.
- symmetrix at 1024 spline points, its code default.

### Force call (ms, median of 100; pass 1 / pass 2; ratio to LoupMD default)

| Skin | Implementation | 1,080 atoms | 5,000 atoms | 13,720 atoms |
|---|---|---|---|---|
| 0.5 Å | LoupMD default | 2.48 / 2.59 | 8.87 / 8.88 | 23.85 / 23.77 |
| 0.5 Å | LoupMD generic | 2.46 / 2.51 (0.98×) | 8.99 / 9.08 (1.02×) | 23.85 / 24.31 (1.01×) |
| 0.5 Å | symmetrix float32 | 3.19 / 3.27 (1.27×) | 13.73 / 13.83 (1.55×) | 36.31 / 36.24 (1.52×) |
| 0.5 Å | symmetrix float32-mixed | 2.82 / 2.77 (1.10×) | 11.96 / 11.84 (1.34×) | 32.03 / 31.62 (1.34×) |
| 1.0 Å | LoupMD default | 2.49 / 2.57 | 8.90 / 9.10 | 24.04 / 24.10 |
| 1.0 Å | LoupMD generic | 2.63 / 2.51 (1.02×) | 8.93 / 8.96 (0.99×) | 24.05 / 24.64 (1.01×) |
| 1.0 Å | symmetrix float32 | 3.35 / 3.29 (1.31×) | 13.81 / 13.77 (1.53×) | 36.50 / 36.39 (1.51×) |
| 1.0 Å | symmetrix float32-mixed | 2.85 / 2.74 (1.10×) | 11.94 / 11.78 (1.32×) | 32.14 / 31.67 (1.33×) |

- Skin changes neither library by more than 3%.
- LoupMD at 1,080 atoms ran 2.5 ms here against 2.2 ms in round 1, so small-cell times vary by about 10% between runs.

### Device time per force call (nsys, 5,000 atoms, skin 0.5 Å, 20 calls)

Phases are mapped by hand from the kernel lists in `prof/`; times in µs per call.

| Phase | LoupMD default | symmetrix float32 | symmetrix float32-mixed |
|---|---|---|---|
| Per-call edge selection | 99 (`compute_and_count_active`, `scatter_active_edges`, skin check, scan) | 326 (`k_count`, scan, `k_fill`) | 331 |
| Layer 0 edge forward | 246 (`tabled_outer_sum0_forward`) | 256 (`k_Phi0`) | 277 |
| Layer 1 convolution forward | 1,924 (`spline_forward_inter1`) | 4,094 (`k_Phi1` + `k_scale1`) | 3,194 |
| Products and readouts, forward and backward | 563 (`sc0_forward`, `sc1_fused`, `sc0_backward`) | 724 (`k_M`, `k_readouts`, `k_M_reverse`) | 763 |
| Layer 1 convolution backward | 2,923 (`wprime_backward_inter1_rcv`) | 4,562 (`k_Phi1_reverse`) | 3,902 |
| Layer 0 edge backward, ZBL, force and virial assembly | 570 (one fused kernel) | 658 (`k_Phi0_reverse`, `k_zbl`, `k_forces`, `k_outputs`) | 682 |
| GEMMs (22.3 GFLOP in both, see below) | 2,348 (42 launches) | 2,561 (cuBLAS, 34 launches) | 2,228 (6 batched launches, plus a 14 µs amax pass) |
| Kernel total | 8,675 | 13,181 | 11,375 |
| Device span of the call | 8,612 | 13,615 | 11,769 |

- **Idle time inside the call:** symmetrix's span exceeds its kernel total by about 0.4 ms, idle gaps inside its graph. LoupMD's span is 63 µs below its kernel total because its GEMM lanes overlap.
- **LoupMD generic** spends about the same: 8,468 µs span. Its convolution kernels take 2,002 (forward) and 2,872 µs (backward); `first_fwd` 220, `first_bwd_edge` 546, products 559, GEMMs 2,324.
- **GEMM work is identical.** Both libraries factor layer 0 through the 64-function radial basis per species pair, followed by GEMMs with K = 192. They also share the `linear_up` GEMMs (128×128 per irrep) and the post-convolution linears (K = 256/384/384/256 per l), with the same backward transposes. That is 22.3 GFLOP per call for 5,000 atoms; LoupMD adds 0.04 GFLOP of skip GEMMs (`src/engine.cu` force pass; symmetrix `compute_first_layer`…`reverse_first_layer`).
- **GEMM rate:**

  | Implementation | TFLOP/s |
  |---|---|
  | LoupMD | 9.5 |
  | symmetrix float32 | 8.7 |
  | symmetrix float32-mixed (3×FP16 tensor cores, one amax scale per GEMM; amax pass included) | 10.0 |

### Convolution kernels (ncu, one force call, 5,000 atoms; 371,624 directed edges)

| Kernel | ms (ncu) | L2 read sectors per edge | L2 read bandwidth | L2 reduction sectors per edge | Warp instructions | Warps active | Registers |
|---|---|---|---|---|---|---|---|
| LoupMD `spline_forward_inter1` | 1.99 | 313 | 1.87 TB/s | 0 | 382 M | 41% | 96 |
| LoupMD generic `gconv_fwd_L1` | 2.09 | 316 | 1.79 TB/s | 0 | 384 M | 33% | 104 |
| symmetrix `k_Phi1` float32 | 4.29 | 727 | 2.02 TB/s | 0 | 411 M | 41% | 96 |
| symmetrix `k_Phi1` mixed | 3.25 | 548 | 2.01 TB/s | 0.05 | 431 M | 41% | 96 |
| LoupMD `wprime_backward_inter1_rcv` | 2.96 | 368 | 1.48 TB/s | 64 | 717 M | 64% | 64 |
| LoupMD generic `gconv_bwd_L1` | 2.99 | 370 | 1.47 TB/s | 64 | 732 M | 63% | 62 |
| symmetrix `k_Phi1_reverse` float32 | 5.21 | 719 | 1.64 TB/s | 69 | 735 M | 25% | 128 |
| symmetrix `k_Phi1_reverse` mixed | 3.92 | 558 | 1.70 TB/s | 69 | 744 M | 25% | 128 |
| LoupMD `tabled_outer_sum0_backward` (fused ZBL and force/virial assembly) | 0.63 | 52 | 0.97 TB/s | 5.7 | 102 M | 41% | 80 |
| symmetrix `k_Phi0_reverse` (layer 0 only) | 0.48 | 32 | 0.79 TB/s | 2.5 | 116 M | 64% | 63 |

- **Same instruction count, different bytes.** Both libraries execute about the same instructions per convolution. symmetrix reads more L2 bytes per edge: 2.3× (float32) or 1.75× (mixed) in the forward, 1.95× or 1.5× in the backward. That is its 16-byte or 12-byte table entries against LoupMD's 8-byte packed ones, and each implementation's time follows its bytes.
- **symmetrix streams L2 faster per byte:** about 8% faster in the forward and 11–15% in the backward, even with 25% warps active in its backward.
  - This is the one measurable gap in LoupMD's convolutions.
  - The S103 ncu numbers put LoupMD's convolutions at about 82% of L2 peak; that would put symmetrix's forward near 88% [inference: same peak assumed].

## Rerun

```bash
bash scripts/sx_build.sh                     # clone ~/src/symmetrix-b907f04, build ~/src/symmetrix-eval/b907f04
# then, under the local-GPU reservation:
bash scripts/job_extract.sh SrTiO3           # ~/src/symmetrix-eval/json
bash scripts/queue_all.sh                    # job_speed.sh, then job_gate.sh (which extracts NaCl and NbCl5)
bash scripts/queue_apples.sh                 # round 2: equal-skin timing and nsys profiles
bash scripts/queue_ncu.sh                    # round 2: ncu convolution kernels
```

## Files

- `speed.jsonl`, `speed_summary.md`: round 1 timing.
- `gate_summary.txt`: gate output.
- `apples_time.jsonl`, `apples_report.md`: round 2 timing and nsys tables (the report's automatic phase grouping is coarser than the table above).
- `prof/`: nsys kernel summaries per implementation.
- `ncu/`: ncu metrics per convolution kernel, and the edge count.
- `logs/`: job output.
- `scripts/`: build, extraction, timing, gate and profiling scripts.
