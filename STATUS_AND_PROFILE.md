# LoupMD Current State, Kernel Profiles, and Open Ideas

Snapshot date: 2026-11-13 (seventh update: generic-path and POLAR speed work S7, 5060 Ti only, section 5.3 and
the ledger's S7 entries; the workstations were ruled out by the owner. Sections 1-2 are from the fifth update.)
Every optimisation tried so far, adopted or not, is indexed in [OPTIMISATION_LEDGER.md](OPTIMISATION_LEDGER.md);
a new attempt adds its ledger row and entry together with the numbers recorded here.
Target model of the specialised kernels (sections 1-4): MACE-OMAT-0 medium (float32, 128 channels, l <= 3,
correlation 3, cutoff 6.0 A). Every other supported family (MACE-OMAT-0 small, MACE-OFF23/23b/24, MACE-MH-1,
MACE-POLAR-1 S/M/L) runs on the generic path: section 5.
Verified GPUs:
- **RTX 5060 Ti** (`sm_120`, Windows/WSL2 WDDM, CUDA 13, 36 SMs, 32 MB L2)
- **RTX 4000 Ada** (`sm_89`, `leeloo`, Linux, 48 SMs, 48 MB L2; no performance-counter permission, so `ncu` numbers are 5060 Ti only)

---

## 1. Benchmark & Verification Summary

Standard MD benchmark: median of 6 x 256 Langevin steps (300 K, 1 fs, skin 0.5 A, friction 0.01/fs) after 100 warmup steps (`python/bench_md.py`). Every change below was A/B tested in both build orders on both GPUs (`bench_md.py ... LOUP_LIB=<A> LOUP_LIB=<B>` and reversed): on Ada the first run of an ABBA sequence is systematically fastest, which biases order A by ~+0.6% against the new build, so a change is adopted only if it wins or ties in both orders.
Standard single-point gate: 3x MACE float32-vs-float64 rule (`python/test_gate.py` + `NbCl5` 576-atom strained, rattled cell).

| Metric | RTX 5060 Ti (`sm_120`) | RTX 4000 Ada (`sm_89`) |
|---|---|---|
| `NbCl5` 3x3x3 (1,944 atoms, S=2) Langevin | **1.849 ms/step** | **2.054 ms/step** |
| `SrTiO3` 10x10x10 (5,000 atoms, S=3) Langevin | **8.458 ms/step** | **9.27 ms/step** |
| `test_gate.py` (4 configs: `NaCl` c0/c1, `SrTiO3` c0/c1) | PASS (force ratio <= 1.25) | PASS (force ratio <= 1.20) |
| `NbCl5` single-config gate (576 atoms, 1% strain, 0.05 A rattle) | PASS (force 0.51, stress 0.83) | PASS (force 0.49, stress 0.69) |
| `SrTiO3` NVE (10x10x10, 1 fs, 300 K, 1 ps after 200 Langevin steps) | drift -0.00011 meV/at/ps (std 0.00036) | drift -0.00012 meV/at/ps (std 0.00037) |

`bench_md` numbers are the mean of two runs of the final build (Ada run-to-run spread on `SrTiO3` is ~2%: 9.17 / 9.37). The `SrTiO3` 20x20x20 and single-point `evaluate()` timings of the previous snapshot were not re-measured.

System-size scaling (5060 Ti, `SrTiO3` n x n x n, Langevin, after 60 warm-up steps; us per atom-step): 135 atoms 3.00 (0.405 ms/step), 320 atoms 2.18, 1,080 atoms 1.73, 5,000 atoms 1.66-1.68, 20,480 atoms 1.69-1.71, 40,000 atoms 1.71-1.72 (the rebuild-time choices stay R5 spline; `w'` picks 512 threads at 40k atoms). Large systems scale linearly; below ~1,000 atoms fixed latencies dominate (section 4.1).

Timeline on the two standard systems (ms/step, `bench_md.py`):

| Stage | 5060 Ti `NbCl5` | 5060 Ti `SrTiO3` | Ada `NbCl5` | Ada `SrTiO3` |
|---|---|---|---|---|
| Session baseline (pre-CUBIC_PACKED8) | 2.384 | 11.380 | 2.533 | 12.985 |
| + `CUBIC_PACKED8` + `hyb_sm120` + `E_graph` bucketing | 2.187 | 10.458 | 2.353 | 11.764 |
| + receiver-owned i0 forward, i0 bwd prefetch, `DeviceSegmentedSort`, `shift_cart`, `sc1_fused`, spatial reorder (>L2), forked GEMMs | 2.089 | 9.939 | 2.260 | 11.184 |
| + items 12-26 | 1.930 | 8.858 | 2.090 | 10.021 |
| + items 27-29 (shift-free decode, 2^8 radial derivative, 4-receiver spline on `sm_89`) | 1.904 | 8.646 | 2.088 | 9.576 |
| + items 29-31 completed (per-system spline / `w'` variants, own SGEMM) | **1.849** | **8.458** | **2.054** | **9.27** |

---

## 2. Detailed Kernel Profile (Current Build)

Captured with `nsys --capture-range=cudaProfilerApi --cuda-graph-trace=node` over 100 Langevin steps after 50 warmup steps (profiler range opened with `cudaProfilerStart` around `run_langevin`); stage spans are medians over steps of the gaps between the custom kernels (rebuild steps excluded by the median). "Span" is wall-clock GPU time of a stage (GEMM groups overlap on side streams). The spline / `w'` variants are already chosen at the first rebuild of the warm-up. The profiled span is within ~3% of unprofiled `bench_md` (5060 Ti `SrTiO3` 8.23 vs 8.46 ms).

### 2.1 Stage Spans (steady-state non-rebuild step, us)

| Stage | 5060 Ti `SrTiO3` | 5060 Ti `NbCl5` | Ada `SrTiO3` | Ada `NbCl5` |
|---|---:|---:|---:|---:|
| `i0_fwd` (`tabled_outer_sum0_forward_kernel`) | 259.0 | 43.0 | 312.4 | 58.6 |
| `compose0_fwd` GEMMs | 299.7 | 88.8 | 351.7 | 88.7 |
| `sc0_forward` | 105.1 | 24.4 | 156.3 | 29.6 |
| `skip16` + `W_up` GEMMs | 66.5 | 28.7 | 66.0 | 27.2 |
| `spline_forward_inter1` (chosen: R5 / R1 paired / R5 / R4 paired) | 1,927.5 | 314.5 | 2,382.3 | 442.7 |
| `lin1` forward GEMMs | 468.6 | 200.6 | 562.9 | 193.4 |
| `sc1_fused` | 222.4 | 58.0 | 232.7 | 68.3 |
| `lin1` backward GEMMs | 549.7 | 175.7 | 557.8 | 196.7 |
| `wprime_backward_inter1_rcv` | 2,961.0 | 443.3 | 3,168.1 | 510.9 |
| `W_up^T` + skip backward GEMMs | 72.0 | 36.9 | 74.2 | 35.7 |
| `sc0_backward` | 230.9 | 63.0 | 273.8 | 84.8 |
| `compose0_bwd` GEMMs | 368.4 | 87.9 | 337.1 | 88.3 |
| `i0_bwd` (+ fused force/virial assembly) | 523.9 | 109.7 | 645.8 | 135.6 |
| Integrator, active-edge compaction, memset, gaps | 126.2 | 45.1 | 140.8 | 60.4 |
| **Step span** | **8,232.0** | **1,723.6** | **9,267.6** | **2,025.5** |
| GEMM groups (sum of spans) | 1,825.0 (22.2%) | 618.6 (35.9%) | 1,949.7 (21.0%) | 630.0 (31.1%) |

GPU idle inside a step is now ~20 us (5060 Ti `SrTiO3`); the 120 us/step of idle caused by split-K memset nodes is gone (item 22).

### 2.2 Top Kernels by Time per Step (us, kernel sum incl. side streams)

| Kernel | 5060 Ti `SrTiO3` | 5060 Ti `NbCl5` | Ada `SrTiO3` | Ada `NbCl5` |
|---|---:|---:|---:|---:|
| `wprime_backward_inter1_rcv` | 3,009 | 462 | 3,188 | 512 |
| `spline_forward_inter1` | 2,008 | 331 | 2,381 | 442 |
| own `sgemm128_kernel` | 1,507 | 424 | 226 | 0 |
| `fm_tf32x3_rc_*` / `fm_tf32x3_rr_*` (hybrid) | 417 / 288 | 137 / 57 | 1,179 / 750 | 434 / 491 |
| cuBLAS SIMT `Kernel2` / `ampere_sgemm` / `magma` | 352 / 0 / 33 | 467 / 0 / 0 | 0 / 538 / 0 | 0 / 60 / 0 |
| `tabled_outer_sum0_backward_kernel` | 537 | 111 | 649 | 134 |
| `tabled_outer_sum0_forward_kernel` | 272 | 50 | 314 | 58 |
| `sc0_backward` / `sc1_fused` / `sc0_forward` | 238 / 226 / 110 | 64 / 60 / 25 | 277 / 235 / 158 | 86 / 68 / 30 |
| `scatter_active_edges` + `compute_and_count_active` | 54 + 47 | 7 + 12 | 40 + 51 | 7 + 16 |
| `langevin_first_half` | 10.6 | 7.8 | 13.4 | 8.1 |
| **Kernel sum / step** | **9,151** | **2,232** | **10,060** | **2,370** |

### 2.3 Neighbour Rebuild (kernel medians per rebuild)

Cadence at skin 0.5 A, 300 K, 1 fs: ~1 rebuild every 15 steps on `SrTiO3`, every ~22 on `NbCl5`, every ~4 on water (5,184 atoms; never pipelined at this skin, measured S34).

| Kernel | 5060 Ti `SrTiO3` | Ada `SrTiO3` | 5060 Ti `NbCl5` | Ada `NbCl5` | 5060 Ti water |
|---|---:|---:|---:|---:|---:|
| `fill_neighbors_kernel` (S34 lane groups on `sm_120`; was 283 / 169 / 273) | 215 | 401 | 76 | 186 | 226 |
| `count_neighbors_kernel` (S34; was 150 / 115 / 166) | 77 | 209 | 34 | 150 | 76 |
| `gather_sorted_edges_kernel` | 37 | 46 | 7 | 9 | 21 |

Ada columns are the one-thread-per-atom kernels (`*_scalar_kernel` since S34), measured before S34.

### 2.4 Kernel Resource Usage (`cuobjdump -res-usage`; shared includes the 1 KB `sm_120` driver reserve)

| Kernel | `sm_120` Reg / Stack / Smem | `sm_89` Reg / Stack / Smem |
|---|---|---|
| `wprime_backward_inter1_rcv` (`512, 2`; 256 or 512 threads) | 64 / 8 / 23,552 B (4 B spill outside the channel loop) | 64 / 0 / 22,528 B |
| `wprime_backward_inter1` (`256, 4`, N < 512 fallback) | 64 / 0 / 24,388 B | 64 / 0 / 23,364 B |
| `spline_forward_inter1<1, paired / unpaired>` (`128`) | 96 / 79 regs, 3,584 B | 122 / 72 regs, 2,560 B |
| `spline_forward_inter1<4, paired / unpaired>` (`512`) | 123 / 94 regs, 11,264 B | 126 / 93 regs, 10,240 B |
| `spline_forward_inter1<5, unpaired>` (`640`) | 96 / 0 / 13,824 B | 95 / 0 / 12,800 B |
| `sgemm128_kernel` (`256, 2`, own SGEMM; + 48 KB dynamic) | 128 / 0 / 1,024 B | 128 / 0 / 0 B |
| `tabled_outer_sum0_forward_kernel` (`128`, warp per receiver) | 96 / 0 / 9,216 B | 80 / 0 / 8,192 B |
| `tabled_outer_sum0_backward_kernel` (`128`) | 80 / 0 / 11,456 B | 80 / 8 / 10,432 B |
| `sc1_fused` (`128, 4`) | 113 / 0 / 2,128 B | 104 / 0 / 1,104 B |
| `sc0_forward` (`128`) | 72 / 0 / 1,056 B | 72 / 0 / 32 B |
| `sc0_backward` (`128, 4`) | 128 / 0 / 1,040 B | 128 / 32 / 16 B |
| `scatter_active_edges_kernel` (`128`) | 40 / 0 / 0 B | 40 / 0 / 0 B |
| `compute_and_count_active_kernel` (`128`) | 40 / 0 / 0 B | 40 / 0 / 0 B |
| `fill_neighbors_kernel` / `count_neighbors_kernel` (`128`, 8 lanes per atom, `sm_120`) | 62 / 44, fill `16 x nscan` ints dynamic | – |
| `fill_neighbors_scalar_kernel` / `count_neighbors_scalar_kernel` (`32`, other GPUs) | 64 / 48 | 64 / 52 |

The `sm_89` stacks are frames without spills (`ptxas -v`: 0 bytes spill stores/loads).

### 2.5 Nsight Compute Facts (`SrTiO3` 5k, RTX 5060 Ti, current kernels)

| Kernel | Time | Occupancy | Key counters |
|---|---:|---:|---|
| `wprime_backward_inter1_rcv` | 3.01 ms | 63% | issue slots 80% busy (issue-bound, `not_selected` dominant); 760 M instructions (818 M before items 27-28); L2 5.3 GB |
| `spline_forward_inter1<1>` | 2.04 ms | 41% (96 regs) | L2 83%, issue 71%; L2 traffic 3.92 GB, L1 hit 5%. `<4>`: 33% occupancy, L1 hit 36%, L2 2.66 GB, 1.96 ms (`SrTiO3`) but `NbCl5` +7-10% |
| `tabled_outer_sum0_backward_kernel` | 0.56 ms | 37% | L1 requests per edge ~29 -> ~11 (SH staged, `float2` channel pairs); equal time at 64 and 100 KB carveout |
| `tabled_outer_sum0_forward_kernel` | 0.25 ms | 39% | L1 hit 53% at the 64 KB configuration (33% at the default 100 KB, 305 us) |
| `scatter_active_edges_kernel` | 0.05 ms | 80% | no SH output (item 24); was 0.11-0.16 ms |

---

## 3. Current Architecture & Hardening Status

### 3.1 Implemented Optimisations
1. **`CUBIC_PACKED8` (`d_cubic_packed8_1`)**: `wprime_backward_inter1` reads an 8-byte exact cubic table (`20/19/12/8` signed quantized coefficients around `s = t - 0.5`, 5-bit shared exponent clamped to `emax - 31`) instead of 16-byte `float4`.
2. **Hybrid `tf32x3` + `fp16` cubin for `sm_120`**: `cubins/tf32x3_hyb_sm120.cubin`; `sm_80` cubin used on Ada (`sm_89`).
3. **`E_graph` bucketing + `cub::DeviceSegmentedSort::StableSortPairs`**: edge grid rounded to 8192; rebuild sort is ~16 us instead of ~2.3 ms.
4. **Receiver-owned i0 forward**: register accumulation per sender-species run, zero writes for absent species; eliminates atomics and the 61 MB per-step `S0/D0` zero-fill. Guarded at rebuild end by `check_row_species_order_kernel`.
5. **Lane-parallel metadata prefetch in i0 backward**: the chunk's metadata is loaded across lanes and broadcast via shuffles.
6. **`sc1_fused`**: fuses `sc1_forward`, `128->16->1->16->128` readout forward/backward, and `sc1_backward` into one `N x 128` launch.
7. **Cartesian float64 shifts (`d_shift_cart`)**: precomputed at rebuild; removes cell dependence from the force graph.
8. **L2-gated spatial atom reorder**: above `N * 4096 > l2_bytes`, Morton reorder within species groups every 16th rebuild.
9. **Side-stream GEMM fork/join**: 8 non-blocking side streams with per-stream 4 MB cuBLASLt workspaces.
10. **Non-destructive `evaluate()`**: keeps velocities, skips rebuilds when `max_disp < skin/2` and cell unchanged.
11. **GEMM autotuning**: up to 32 cuBLASLt candidates + 3 hybrid tiles; fork disabled during autotune.
12. **1-receiver-per-block `wprime_backward_inter1_rcv`** (`N >= 512`; edge-tiled kernel kept for `N < 512`): stages the receiver's q once per block; -170 us on both GPUs.
13. **Packed edge arrays**: `act_vec_len` (`float4` x, y, z, r) and `act_t_sx` (`float2`).
14. **i0 backward fused with force/virial assembly (T5)**: `force_virial_assemble_kernel`, `d_dsh0`, `d_dr0` removed; 32-edge warp chunks.
15. **Float32 pre-test + 32-thread blocks in `count`/`fill_neighbors` (T3)**: borderline pairs re-checked in float64; rebuild 1,222 -> 510 us (5060 Ti), 1,121 -> 684 us (Ada).
16. **Memory aliasing + persistent GEMM plan cache**: `dm1 -> m1`, `dh0 -> h0`, `dS0 -> S0`; plans in `~/.cache/loupmd/plans_<gpu>_sm<cc>_lt<ver>_hyb<0|1>.bin` (`LOUP_RETUNE`, `LOUP_CACHE_DIR`); cold-start `evaluate` 1.3 s -> 0.21 s (5060 Ti), 0.76 s -> 0.05 s (Ada).
17. **Per-arch L1 carveouts** (`tune_l1`): smallest shared-memory configuration that keeps occupancy for the gather kernels (values measured per arch; see `engine.cu`).
18. **Selective cold-L2 GEMM autotune**: operands larger than half the L2 are timed with a flushed L2 (they stream from DRAM in the force pass); Ada's `lin1_fwd` group 672 -> 575 us.
19. **Coalesced SH stores** in `scatter_active_edges` (superseded by item 24).
20. **Paired-edge `spline_forward_inter1`**: two consecutive edges of a receiver that read the same table slab (sorted by `(pair, r)`) decode it once: table loads per edge 1 -> 0.805 (`SrTiO3`), 0.927 (`NbCl5`); spline -7% (5060 Ti), -3% (Ada) on `SrTiO3`.
21. **Channel-major staged q in `wprime_backward_inter1_rcv`**: `s_q[channel * 44 + slot]` (11 x 16 B, odd) so each lane reads its 40 q values as 10 conflict-free `LDS.128`; w' -1.7% (5060 Ti), -2.5% (Ada).
22. **No split-K cuBLASLt algorithms** (`CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = NONE`; plan cache `LOUPPLN3`): their per-launch workspace memsets left the 5060 Ti idle ~120 us per step; also deterministic.
23. **i0 backward SH staging + `float2` channel pairs**: the chunk's SH are staged in the warp's result rows (stride 20, `LDS.128` broadcast), coefficients and dS0 rows load as `float2`; 685 -> 564 us (5060 Ti), 736 -> 668 us (Ada).
24. **SH computed where staged, not stored** (`edge_sh16`): `scatter_active_edges` no longer writes 64 B/edge (24 MB per step at 5k atoms); i0 forward/backward and spline forward compute the 16 values from `act_vec_len` while staging. Dead `edge_sh3_forward_kernel` and `common_kernels.cu` removed; rebuild sort key scratch moved to `d_edge_force_1`.
25. **Warp-per-receiver i0 forward**: 4 receivers per 128-thread block, 2 channels per lane (`float2` coefficient loads, one SH broadcast per two channels), per-edge D0 term at staging; 390 -> 245 us (5060 Ti, 64 KB carveout), 415 -> 304 us (Ada).
26. **Langevin integrator**: first half launched with 32-thread blocks (its FP64 Box-Muller is FP64-throughput bound on consumer GPUs: 14.9 -> 8.0 us on `NbCl5`), step-counter increment folded into the second half (one graph node less); random stream bit-identical.
27. **Shift-free 8-byte cubic decode in `w'`**: the loader packs each coefficient's mantissa bits at their float position (`x = u0<<3 | rotr(u2<<11, 20)`, `y = u1<<4 | rotr(u3<<15, 19) | eb<<23`), so a coefficient decodes as one `LOP3` (mask + OR with the biased exponent) or one funnel shift + `LOP3`; inner loop 480 -> 448 static instructions, `w'` 3,177 -> 3,065 us (5060 Ti). MD: 5060 Ti -0.5% / -1.2% (A), -0.8% / -1.3% (B); Ada +0.4% / -0.8% (A), -0.6% / -1.1% (B) (`NbCl5` / `SrTiO3`).
28. **`w'` radial derivative scaled by 2^8**: `radial_dt = fma(ds3', d3, fma(ds2', d2, d1))` with `ds2' = 2^8 ds2`, `ds3' = 2^8 ds3` and one `2^-8` at the edge epilogue (bit-identical: power-of-two scaling), dropping one `FMUL` per path; `w'` 3,069 -> 3,010 us, 774 -> 760 M instructions (5060 Ti). MD: 5060 Ti -0.1% / -0.5% (A), -0.5% / -0.6% (B); Ada tie.
29. **Multi-receiver spline blocks, chosen per system** (`spline_forward_inter1<R, PAIR>`, `choose_spline_rpb`, `tune_rebuild_variants`): with R consecutive receivers per 128R-thread block (per-receiver named barriers, smallest shared-memory carveout), receivers of a symmetric crystal walk the same slabs at the same time and the later reads hit L1 (`SrTiO3` 5060 Ti, R = 4: L1 hit 5% -> 36%, L2 3.92 -> 2.66 GB). The block keeps all its warps' registers until its slowest receiver finishes, so the gain depends on the system, and so does the paired-edge decode (item 13; it costs 17-30 registers). Five variants are compiled: R = 1 and R = 4 with and without pairing, and R = 5 unpaired (640 threads at 94-96 registers: 20 warps). The default is R = 4 paired where it keeps the resident warp count of R = 1 paired (`sm_89`), else R = 1 paired; at the first MD rebuild of a system (thermalised positions; the ideal starting lattice favours sharing everywhere) all five are timed on the last force pass's inputs and the fastest is adopted if >= 2% faster than the current one (`LOUP_DEBUG_SPLINE=1` prints the timings).
    Spline us at the first rebuild (R1 / R1 paired / R4 / R4 paired / R5 -> choice): 5060 Ti `SrTiO3` 2,029 / 1,953 / 1,828 / 1,847 / 1,798 -> R5; 5060 Ti `NbCl5` 317 / 316 / 330 / 344 / 319 -> R1 paired; Ada `SrTiO3` 2,601 / 2,593 / 2,260 / 2,270 / 2,187 -> R5; Ada `NbCl5` 431 / 418 / 411 / 423 / 419 -> R4.
    MD, step by step (A / B orders): occupancy rule only (R1 paired vs R4 paired) vs item 28: Ada `NbCl5` +1.5% / +0.3%, `SrTiO3` -3.1% / -3.6% (5060 Ti unchanged). + rebuild-time choice between those two: 5060 Ti `NbCl5` +0.1% / -0.3%, `SrTiO3` -0.8% / -0.8%; Ada tie. + unpaired and R5 variants: 5060 Ti `NbCl5` +0.4% / -0.2%, `SrTiO3` -0.1% / -0.3%; Ada `NbCl5` +0.8% / -0.2%, `SrTiO3` -0.4% / -0.9% (Ada order A carries the ~+0.6% first-run bias).
30. **Own SIMT SGEMM as an autotune candidate** (`src/sgemm.cu`, `loup::sgemm128`): 128 x 128 x 16 block tiles, 256 threads with 8 x 8 outputs each, 3-stage `cp.async` pipeline (A kept m-major, so a warp's A fragment reads are 2-address broadcasts; B k-major), 126 registers, 48 KB shared memory, 2 blocks (16 warps) per SM where cuBLAS' SIMT kernels run 8 warps at 200-210 registers. A transposed B operand is always a model weight, so `weight_kn` keeps a [K, N] copy per weight (made in the uncaptured autotune pass). Each output is one in-order `fmaf` chain over k (bit-identical to `cublasSgemm` on all tested shapes). `sgemm_rm` times it next to the cuBLASLt heuristics and the hybrid cubins (plan cache `LOUPPLN4`, `use_own`; `LOUP_DEBUG_GEMM=1` prints every choice). Cold-L2 microbenchmark (5060 Ti, us): `lin1` l=1 / l=2 / l=3 forward 132 / 196 / 181 (cuBLASLt's pick 136 / 207 / 175, `cublasSgemm` 236 / 370 / 185); `compose0` forward (M = 48,000, K = 192) 195 vs 210. It is picked for the large `lin1` / `lin1^T` / `compose0` forward products on the 5060 Ti. N = 192 (`compose0` backward) runs 128 x 64 tiles instead of a half-empty 128 x 128 tile (111 registers, 36 KB): the 5060 Ti autotune then picks it over the hybrid cubin for all seven `SrTiO3` products (sum 297 -> 248 us); MD tie on both GPUs (5060 Ti `NbCl5` +0.5% / +0.0%, `SrTiO3` -0.1% / -0.4%; Ada +1.2% / +1.0% (A), +0.0% / -0.2% (B)).
    MD: first version (register-staged, 2 stages) vs item 29: 5060 Ti `NbCl5` -1.6% / -2.0%, `SrTiO3` -0.6% / -0.7% (A / B); Ada +0.8% / +0.2% (A), -0.2% / -0.3% (B). `cp.async` version vs the first: 5060 Ti `NbCl5` -1.4% / -1.3%, `SrTiO3` -0.1% / -0.4%; Ada +1.1% / +0.5% (A), -0.3% / +0.0% (B).
31. **`w'` block size chosen per system** (`wprime_backward_inter1_rcv` at `(512, 2)` launch bounds, warp stride `blockDim.x / 32`, launched with 256 or 512 threads; timed with the spline variants in `tune_rebuild_variants`): 512-thread blocks stage one receiver's q per 16 warps, halving the shared memory per SM (more L1: 5060 Ti L1 hit 5.9% -> 10.5%, L2 5.27 -> 5.05 GB). Ada `SrTiO3` `w'` 3,230 -> 3,130 us (picked); 5060 Ti 3,009 -> 3,034 us and Ada `NbCl5` 519 -> 511 us (within 2%: 256 kept). MD: 5060 Ti tie (+0.6% / -0.1% `NbCl5`, -0.1% / -0.1% `SrTiO3`); Ada `SrTiO3` -1.0% / -1.2%, `NbCl5` +1.4% (A, first-run bias) / +0.2% (B).

32. **`w'` cubic decode with the bias folded into the polynomials** (`python/gen_kernels.py`, `LOUP_GEN_DECODE_FOLD`, on by default since 13 Nov): the four `raw_k - hb` subtractions per path become two FFMA with per-edge bias factors; `wprime_backward_inter1_rcv` 1,000 -> 976 instructions on `sm_120` (the 8-byte stack spill disappears), 1,016 -> 1,000 on `sm_89`. `bench_md.py`, a-b-b-a runs: 5060 Ti `SrTiO3` 0.9929 / 0.9948 / 0.9925 and `NbCl5` 0.9951 / 0.9938 / 0.9965 (three runs; about 8.22 and 1.80 ms/step); Ada a tie (1.0015 / 0.9955, two runs). The default gate passes on both GPUs with ratios inside its own run-to-run spread; 47 tests pass on the 5060 Ti and 46 on the Ada (OMAT-0 small skipped there), and the full gate passes for all eight model sets on the 5060 Ti (78 rows, worst force ratio 1.45, stress 2.27). Ledger: [wp-decode-bias-fold](OPTIMISATION_LEDGER.md#wp-decode-bias-fold).

### 3.2 Correctness / API Fixes Applied (from code review)
- **H1**: Langevin Philox offset advanced by `step * 16` (12 outputs consumed/step; previously overlapped at `step * 2`).
- **H2**: Langevin `friction_per_fs` converted via `/ ASE_FS` to match ASE units; header documents ASE velocity units.
- **H3**: `d_step_counter` zeroed only in `loup_set_system` (persists across `run_*` calls).
- **H4**: Cubic-table exponent span > 31 clamped to `emax - 31`; unused `float4` fallback removed.
- **H5**: All C entry points wrapped in `guarded(...)` (closes open stream captures, resets `gemm_stream`/`gemm_ws`, returns `-1`).
- **L3**: `ensure_cub_tmp` invalidates graphs on growth; `cudaGetLastError()` checked after warm-up pass and each rebuild.
- **L4**: `fill_neighbors_kernel` guards its cursor with `cursor < row_ptr[i + 1]`.
- **Review round 5** (`loupmd-review-20261112.md`, fifth pass):
  - F1: `test_gate.py` adds the 576-atom `NbCl5` cell (receiver-owned `w'`) and gates every forced kernel path: `LOUP_SPLINE_VARIANT=R1|R1p|R4|R4p|R5`, `LOUP_WPRIME_THREADS=512`, `LOUP_WPRIME_RCV=1` on the small cells, `LOUP_REORDER=0|1` (reorder on vs off agree to 5e-7 eV/A).
  - F2: `ensure_cub_tmp` synchronises the stream before freeing. F3: the per-system variant timing is reported as `LoupMDStats::tune_ms` and excluded from `step_ms` (aliasing of the timed buffers documented). F4: the plan cache file name carries an FNV hash of the library file; plans whose winner leads by < 3% are not persisted; `LOUP_CACHE_DIR` / `LOUP_RETUNE` documented in `include/loupmd.h`. F5: `loup_debug_tensors` removed (unused, model-specific). F6: scope guards for the autotune buffers and events. F7: the spline's named-barrier conditions are stated in the generator, with `static_assert(R <= 15)`.
  - M1: the MD loop starts unpipelined (each step's skin check is read before its force pass: exact for any skin, including 0) and switches to the pipelined loop once 4 x the largest per-step displacement (tracked on the device by the first-half kernels) is below skin/2; the pipelined check lowers its threshold by three times that displacement (two steps of lag, one of margin for a step longer than any so far: with only the lag, 1 of ~1,600 steps of the 300 K benchmarks ran on an incomplete list), and any step that still did is counted in `LoupMDStats::dangerous_builds` (and drops back to the unpipelined loop; 0 in every benchmark of section 5 on both GPUs). MD speed unchanged (`bench_md.py`: 5060 Ti `NbCl5` 1.836, `SrTiO3` 8.454 ms/step, 1.832 / 8.441 with the two-step margin; Ada 2.056 / 9.262).
  - M2: `loup_set_system` validates Z, masses and skin before freeing the old system. M3: singular / zero / non-finite cells and non-finite positions or velocities are rejected; the engine is documented as fully periodic.
  - M6: bundles are a self-describing container (`LOUPC001`, `python/bundle.py`, `src/bundle.cu`): named, typed, shaped arrays plus a meta text entry (architecture, exporter version, model sha256, kernel signature); every array is shape-checked, truncated files and trailing bytes are rejected, files are closed on every path; `ScaleShift` shift is folded into E0; the bundle cache key is the model's content hash + species + head + path + exporter version; the specialised bundle must match the compiled kernels' signature (`generated_kernels_signature.h`).
  - M7: `tests/` (pytest): NVE energy conservation, skin list vs rebuild-every-step trajectory, hot small-skin Langevin without dangerous builds, `tune_ms` reporting, evaluate-after-MD, and the error paths (unknown species, bad masses / skin / cell / positions / velocities, MD without a system, truncated / padded / missing bundles); `tests/test_gate_wrapper.py` runs the gate. 24 tests pass on both GPUs.
- **Review round 6** (`loupmd-review-20261112.md`, sixth pass; fixed 13 Nov 14:00, each with a test that failed first). Verified on both GPUs: 35 tests pass; the gate passes for every family on the 5060 Ti (worst force ratio 1.31, stress 1.95) and for 7 of 8 sets on the Ada (1.31 / 1.49; OMAT-0 small skipped, its model file is not on `leeloo`); `bench_md.py` unchanged on the 5060 Ti (NbCl5 1.81, SrTiO3 8.33 ms/step; not re-timed on the Ada).
  - S1: the unpipelined skin check also reports whether any atom is beyond the pipelined (three-step) threshold (bit 4 of the flag, `check_skin_kernel`); the loop switches to pipelining only if that bit is clear or the list was rebuilt in the same step. Before: 9 force calls on an incomplete list in 60 switches of `test_switch_to_pipelined_loop_never_uses_an_incomplete_list`; now 0. `LoupMDStats::pipelined_steps` reports how many steps ran pipelined.
  - S5: `PolarLR::set_cell` reports every cell change, so the force graph is re-captured and no longer carries the first cell's volume factors (NaCl 2x2x2, POLAR-1-S, strain 3e-4 on the same engine: 1e-6 eV/atom, 9e-5 eV/A and 3e-7 eV/A^3 against a fresh engine before; identical now). `test_polar_energy_follows_a_small_cell_change_on_the_same_engine`.
  - S6: `python/loupmd.py` stores the atom count only after `loup_set_system` succeeded and refuses arrays of the wrong shape (before: a rejected `set_system` of another size made the next `evaluate` write past its arrays; the test process segfaulted).
  - S7: a generic model's pass also runs when no pair lies within the cutoff (isolated atoms: MACE-MH-1 was off by 0.56 eV per atom, MACE-POLAR-1-S by 0.64 eV; MACE-OFF23 small and OMAT-0 small were unaffected). `test_generic_model_without_neighbour_pairs_matches_mace`.
- **Review round 6, second batch** (13 Nov 14:20-14:50; S8, S3 and the build layout). Verified on both GPUs: 47 tests pass on the 5060 Ti and 46 on the Ada (OMAT-0 small skipped there: its model file is not on `leeloo`); the gate passes for all eight model sets on the 5060 Ti (78 rows, worst ratios 1.45 / 2.27) and for seven of eight on the Ada with freshly exported bundles (72 rows, 1.31 / 1.18).
  - S8, certificates: `export_omat0.py` now refuses a bundle whose interaction-0 table or interaction-1 spline is less accurate than the float32 network (`rad.certify` and the `rspline.fit` certificate, both <= 1; the SrTiO3 set gives 0.28 and 0.49). `rad.fit`'s `fit_error` is not the criterion: the real table has pieces at the deepest level with 4e-5 against a tolerance of 1e-7 and still certifies.
  - S8, kernel signature: the signature now ends in the hashes of the two factored product programs (`:p0=dd662f9d3a2b5b2b:p1=ae0a447b65f4fcd4`), written by `gen_kernels.py` and by the exporter and compared by the loader, so a change of FastMACE's factoring between generation and export is refused. Regenerating the kernels gives a byte-identical `generated_kernels.cuh`; `gen_kernels.py` needs the `mace_cueq_oeq` environment (openequivariance). Exporter version 4.
  - S8, cache key: `loupmd.exporter_fingerprint()` (hash of the four exporters, `bundle.py` and the signature header) replaces the manual `EXPORT_VERSION`; every cached bundle is re-exported once.
  - S3, tests: a replica test for five families (a cell against its periodic replica of about 2,000 atoms; Ada: energy within 8e-7 eV/atom, forces within 5e-6 eV/A for the local models and 7e-4 eV/A for POLAR-1-S) and NVE conservation for MH-1, OFF23 small and POLAR-1-S (drift 0.003, -0.05 and -0.0002 meV/atom/ps). `tests/test_export.py` covers the certificates, the signature and the cache key.
  - Build: one object directory per architecture (`build/<CUDA_ARCH>/`); `make CUDA_ARCH=sm_89 objects` compiles for linking on `leeloo`.

### 3.3 Ruled Out (Measured)
- **T2 / 4-byte far-edge table rows**: slower on both GPUs (5060 Ti `w'` 3,286 -> 3,316 us, spline 2,211 -> 2,242 us; Ada `w'` 3,561 -> 3,628 us, spline 2,974 -> 3,075 us).
- **T1 / 2-channel vectorized `w'`**: -1.6% at `(256, 3)` on 5060 Ti, +2.3% on Ada.
- **Pre-transposed forward GEMM weights (old idea 1)**: 5060 Ti tie, Ada +1.0% / +0.3%.
- **`sh` shuffle broadcast in i0 backward (old idea 3)**: tie; superseded by item 23 (staged SH).
- **`spline_forward` `(128, 7)` / min-blocks 5 on `sm_89`**: spills; Ada +17% / +3%.
- **T4 grouped species GEMM (`fm_tf32x3_species_rr_64x64`)** for `compose0_fwd` + `skip16`: 5060 Ti +1.7% (`NbCl5`) / +0.8% (`SrTiO3`); the grouped launch is slower than the forked per-species GEMMs.
- **T6 per-atom energy array**: +0.5% / +0.1% (5060 Ti); the extra reduction launch costs more than the removed atomics.
- **T7 receiver-owned i0 backward**: +1.8% / +1.1% (5060 Ti); per-chunk edge parallelism beats per-receiver runs.
- **Paired edges in `w'`**: instructions -3% to -6% but 64 -> 80/128 registers (spills at 64); 3.23 -> 3.28/3.35 ms.
- **2/4-pass channel staging in `w'`** (smaller `s_dout`, more L1): 3.23 -> 3.40 / 3.80 ms.
- **Interaction-1 chain split into two receiver halves on two streams** (spline/GEMM/w' overlap) and high-priority GEMM streams: 5060 Ti +0.9% (`NbCl5`) / +0.7% (`SrTiO3`).
- **Global cold-L2 autotune** (all GEMMs timed cold): Ada `NbCl5` +2.9%; kept only for operands larger than half the L2 (item 18).
- **Receiver remap for spline L1 locality** (blocks strided over the receiver range): tie.
- **`spline_forward` 16-edge tiles**: tie.
- **Single-pass active-edge compaction** (decoupled look-back over 4-row tiles, pass-1 vectors cached in shared memory): kernel 96 -> 80 us on 5060 Ti `SrTiO3`, equal on `NbCl5`; MD tie on both GPUs in both orders, so the simpler count + scan + scatter path stays.
- **Interaction-1 chain over L2-sized receiver chunks** (spline + `lin1` + `sc1` + `lin1^T` + `w'` per chunk so `m1`/`dm1` stay in L2): 5060 Ti 1,700-atom chunks +2.4% / +0.2%, 1,000 +2.4% / +1.1%, 650 +2.6% / +2.8% (`NbCl5` / `SrTiO3`); per-chunk launch and tail costs exceed the DRAM savings (the `lin1` GEMMs run at ~50% DRAM, latency-limited).
- **`w'` scalar-path `x0` hoisting** (accumulate the `x0`-scaled pull terms of the four `a = 0` paths, multiply once per channel): -2 M instructions, but spills at 64 registers (3,016 -> 3,086 us) and is still slower at `(256, 3)` / 80 registers (2,992 -> 3,018 us).
- **`w'` `(256, 3)` launch bounds** (80 registers, 50% occupancy): 5060 Ti 3,013 -> 2,992 us, Ada 3,184 -> 3,184 / 3,221 us; below A/B resolution.
- **Paired 5-receiver spline blocks** (640 threads at 96 registers): spill 48-72 B and lose to R = 1 / R = 4 (the unpaired R = 5 of item 29 fits without spills). Timing R = 1 vs R = 4 in the uncaptured setup pass picks R = 4 for `NbCl5` too (the ideal starting lattice shares slabs), hence the rebuild-time choice of item 29. Lockstep R = 2/4 (block-wide barriers, max tile count) is slower than per-receiver named barriers.
- **SM-local persistent `w'`** (grid = resident blocks, receivers handed out from per-SM contiguous ranges via `%smid` and atomic cursors, stealing at the end, so co-resident blocks work on neighbouring receivers): L1 hit 5.7% -> 12%, L2 bytes unchanged (5.3 -> 5.4 GB), 40 B stack from the extra loop state; 3,014 -> 3,240 us (5060 Ti).
- **Reversed block order in the kernels between GEMMs** (spline, `sc0_forward`, `sc1_fused`, `w'`, `sc0_backward`, i0 forward/backward walk receivers from the top, so they first read the rows the preceding GEMM wrote last and leave the rows the next GEMM reads first in L2): step span 8,274 -> 8,285 us (5060 Ti `SrTiO3`), no stage gained more than 6 us.
- **SGEMM variants** (microbenchmark / MD, 5060 Ti): `cp.async` 4 stages with BK = 8: 1.2-1.5x slower than BK = 16 (twice the barriers, 32-byte row segments); register-staged with BK = 8 or 1 block per SM: 5-30% slower; extra 64 x 128 and 128 x 64 tiles as autotune candidates (for small M and N = 192): picked in isolation but `NbCl5` +1.6% / +0.8% in the pass; 64 x 128 tiles for every N = 128 product (cold microbenchmark 7-8% faster than 128 x 128 on the 5060 Ti: `lin1` l=1/2/3 forward 115 / 181 / 172 vs 126 / 196 / 185 us; on Ada still behind `cublasSgemm` for l=1/2): MD `NbCl5` +1.7% / +1.8%, `SrTiO3` +0.5% / +0.8% (5060 Ti), Ada +1.1% / +0.2% (A), -0.0% / -0.2% (B). Isolated cold timings of these tile shapes do not carry over to the concurrent GEMM groups.
- **Row-chunked hybrid GEMM for cold operands** (launch the 64-wide hybrid tiles per quarter-L2 row chunk so the second column tile re-reads A from L2): never chosen by the cold autotune on either GPU (Ada `lin1` forward stays on cuBLASLt at 148 / 228 / 209 us).
- **i0 backward micro-variants** (5060 Ti `SrTiO3`; SASS stall sampling puts ~14% of the samples on the first FFMA consuming the edge's coefficient loads): next edge's coefficients prefetched into registers (96 registers): 551 -> 571 us; coefficient loads moved to the top of the edge iteration (80 registers): 554 -> 564 us; `grad_a` summed in 4 interleaved partial sums: 554 -> 626 us (+4 M instructions, worse schedule).
- **`w'` table loads without L1 allocation** (`ld.global.nc.L1::no_allocate`, to keep L1 for the sender rows): L1 hit 9.7% -> 1.3%, L2 5.06 -> 5.53 GB, 3,022 -> 3,120 us (5060 Ti); adjacent warps do re-read table rows through L1.
- **GEMM groups serialised** (no side streams): 5060 Ti `NbCl5` +5.7% / +5.0%, `SrTiO3` +2.1% / +2.0%: the concurrent lanes matter.
- **Per-group own-SGEMM policy tuned in context** (products of a fork group collected; at a second uncaptured pass each group is timed whole under three policies: as autotuned per product / own wherever eligible / never own; the fastest kept per group signature). Group timings differ a lot from the per-product picks (e.g. `SrTiO3` `lin1^T` group 637 -> 552 us with own everywhere, timed as a captured graph with a warm L2), but MD gained only 5060 Ti `NbCl5` -0.4% to -0.8%, `SrTiO3` +0.1% / +0.2%, Ada tie in both variants (stream-timed, graph-timed); with a cold L2 per launch the policies mostly tie and the timings get noisy. Not worth the collection / replay machinery.
- **`sc0` per species overlapped with the other species' compose0 products** (forward: `sc0_forward` of species s on its l = 3 lane after its four products; backward: `sc0_backward` of species s, then its four products, while the next species' `sc0_backward` runs; 16 side streams): 5060 Ti `SrTiO3` i0 forward end -> spline start 470 -> 489 us, `w'` end -> i0 backward start 600 -> 647 us (the DRAM-bound `sc0` kernels compete with the products); `NbCl5` -4 / +1 us.
- **Other earlier drops**: receiver-warp segmented float64 force reduction, 2-edge unroll in spline (pre-pairing), `sc0_backward` `(128, 3)`, overlapping `skip16` GEMMs across spline, per-species i0 forward, cubic table in forward, Morton reorder below L2.

---

## 4. Open Ideas & Priorities

### 4.1 Remaining Performance Opportunities

All 13 ideas of the previous list are resolved: adopted (2, 5, 7, 9, 10, 13 = items 12-16), ruled out (1, 3, 4, 6, 8, 12; section 3.3), or deferred (11, below).

| Priority | Idea | Target & Estimate | Notes / Falsifier |
|---|---|---|---|
| **1** | **Fuse `lin1` into the interaction-1 kernels** | `m1`/`dm1` are `N x 4864` floats (97 MB at 5k atoms) written by spline / `lin1^T` and read back by `lin1` / `w'`; the `lin1` groups are 529 + 597 us on 5060 Ti `SrTiO3` | Not feasible inside the current one-receiver-per-block spline: each block would read all 655 KB of `W_lin1` (~3.3 GB extra L2 traffic per step for an already L2-bound kernel), and sharing it across R receivers needs R x 38 accumulator registers (spline is at 96-122). Needs a different decomposition (e.g. a receiver-tiled tensor-core GEMM consuming spline output tiles from shared memory). |
| **2** | **Ada performance counters** | all Ada tuning so far is timing-only | `ERR_NVGPUCTRPERM` on `leeloo`; needs `NVreg_RestrictProfilingToAdminUsers=0` (admin) before Ada-specific `ncu` work (`w'` is 9% slower on Ada than on the 5060 Ti although Ada has 33% more SMs; spline gained 10% from L1 slab sharing there, so Ada looks L2-sensitive). |
| **3** | **Small-system latency (< ~1,000 atoms)** | 135-atom `SrTiO3` step 358 us (median span): GEMM groups 115 us (32%; ~25 tiny products, each ~6-13 us in the graph regardless of size), edge-tiled `w'` 100 us, spline 58 us (135 one-receiver blocks), i0 backward 28 us | Split-K for the own SGEMM (partials per K slice, last block of a tile sums them in z order) was tried: the isolated autotune cannot rank such products (every candidate times 6-14 us, launch-bound), and forcing split 2 / 4 for every eligible product raised the 135-atom GEMM groups 123 -> 147 / 158 us (step 367 -> 392 / 409 us), so the per-GEMM floor in the graph is launch/completion latency, not the K pipeline. Grouped launches were tried too: products of a fork group collected and, when each has fewer tiles than SMs, issued as one own-SGEMM launch (problem table, 64 x 64 tiles below two blocks per SM): 135-atom GEMM groups 114 -> 101 us (step 355 -> 345 us on the 5060 Ti, wall time -3% to -5%; 320 atoms within noise; Ada 135 atoms +2%), but the refactored kernel cost the 5060 Ti benchmarks +1.2% / +0.9% (`NbCl5`) and +0.3% / +0.2% (`SrTiO3`); dropped. The spline is not the problem: even at 135 atoms its 135 blocks keep ~15 warps per SM issuing (R4 picked, 64 us); an edge-split variant (one receiver per block, edges over 2 / 4 thread groups, fixed-order shared-memory reduction) measured 98 / 82 us there and 1.4-1.6x slower at 5,000 atoms. |
| **4** | **Larger / adaptive skin (`T10`)** | 0.5-1.5% at 5,000 atoms, ~10% at 135 atoms | Changes the benchmark configuration; owner decision. |
| **5** | **Generic path (section 5): gate in the GEMM epilogue** | MH-1 SrTiO3 (5060 Ti): gate forward + backward 15% of the step, ~1 GB of DRAM traffic per call; GEMMs ~50% | The gate reads the `linear_2`-side rows a GEMM just wrote and writes rows the next GEMM reads; fusing it into the own SGEMM's epilogue / prologue removes one round trip per direction. Falsifier: gate kernels already at ~DRAM bandwidth, so the ceiling is their full time (~7 ms of 47); stop if the fused GEMM loses more than it saves. |

### 4.2 Open Review Hardening Items (Correctness / Robustness)
M1-M3, M6, M7 and F1-F7 are resolved, and S1, S3, S5-S8 of the sixth pass (section 3.2). Open from the sixth pass:
- **S2**: the running maximum of the step displacement never decays, so one fast step keeps the loop unpipelined until the next `set_state` (hydrogen at 300 K: always unpipelined at skin 0.5).
- **S4, S9 (small)**: MD graphs re-captured on every `run` call; `temp_k` / `friction_per_fs` not validated; force below the first spline knot on the generic path; unchecked negative `lmax` and meta integers in bundles; three k-space buffers never freed and `set_cell` not exception-safe in `polar.cu`; exporter accepts models by class name only.

---

## 5. Model Families (Generic Path)

Supported, as in FastMACE (every label of its model table; local files match its SHA-256): MACE-MP-0 medium and large
(`2023-12-03-mace-128-L1_epoch-199`, `MACE_MPtrj_2022.9`), MACE-MP-0b3 medium, MACE-MPA-0 medium, MACE-OMAT-0
small and medium, MACE-MatPES PBE and r2SCAN (`MACE-matpes-*-omat-ft`), MACE-OFF23 small / medium / large,
MACE-OFF23b medium, MACE-OFF24 medium, MACE-MH-1 (any head, one per bundle; default `omat_pbe`), MACE-POLAR-1
S / M / L and MACE-OMOL-0 extra large (`MACE-omol-0-extra-large-1024`: 1024 hidden channels; total charge and spin
multiplicity per bundle, `ensure_loup_bundle(..., charge=Q, spin_multiplicity=M)`, default 0 and 1). MACE-OMAT-0
medium, MP-0b3 medium, MPA-0 medium and both MatPES models (the same density architecture) run the specialised
kernels of sections 1-4 (`LOUP_GENERIC=1` forces the generic path, used to cross-check it); the rest the generic path.

### 5.1 Pipeline
- `python/export.py` picks the exporter: `export_omat0.py` (specialised, `arch=omat0_medium_spec`) when
  `check_architecture` accepts the model, else `export_generic.py` (`arch=generic`). Both write the `LOUPC001`
  container (section 3.2, M6); `ensure_loup_bundle(model, Z, head=...)` caches by model content hash, species, head,
  path and exporter version.
- `export_generic.py` extracts every linear map by probing the float64 e3nn modules with basis inputs (`probe_linear`,
  `probe_species_linear`), so e3nn path weights, `1/avg_num_neighbors`, the normalize2mom constants of SiLU / gates,
  ScaleShift (scale into the readouts and ZBL, shift into E0) and the selected head (E0 row, readout column,
  `mask_head` block) are folded numerically. Conv tensor-product paths are checked to be multiples of
  `wigner_3j(l1, l2, l3)`; the factor goes into the radial table. Radial functions (Bessel + cutoff + Agnesi
  transform + MLP, for the nonlinear blocks with the source / target species embeddings, and the edge density) are
  fitted per species pair (symmetric pairs unless the block embeds sender and receiver separately) with FastMACE's
  certified cubic-Hermite spline (`spline.fit`; the export refuses a certificate above 1, i.e. a spline error
  above the float32 network's own rounding error on FastMACE's `radial.certify` points). The conv layers' tables are
  stored as 8-byte cubics (CUBIC_PACKED8); the export then refines every band whose packing error would exceed half
  the loader's bound (`refine_for_packing`: 2.5e-4 of the column's largest value or r-slope, whole table and
  r >= 1 A) and re-certifies the refined grid. The gated bundles of the other families already pack within it and
  are unchanged; OMOL-0's steeper radial functions needed it (layer 1 2.1e-3 before: refused by the loader).
  Products: FastMACE's factored symmetric contraction (`sc_factored.factor` / `fold_weights`), kernels keyed by a
  signature of the factored program.
- MACE-OMOL-0's joint embedding (total charge, spin multiplicity) is one per-system vector added to every atom's
  layer-0 features, so the export folds it into each species' embedding row (everything downstream of layer 0 stays
  species-only) and its embedding readout energy into E0; charge and spin are export arguments (`export.py
  --charge Q --spin-multiplicity M`, part of the bundle cache key) and are refused for models without the embedding.
  Hidden widths up to 1024 (gates with one thread per channel, products on 128-channel blocks); the conv width stays
  at most 512, and the fused product + readout kernel runs only for widths up to 512.
- `python/ref_generic.py`: a float64 PyTorch executor of the bundle (same algorithm as the CUDA path, autograd forces
  and stress), used to check exports against MACE float64 and to localise CUDA-vs-model differences.
- `python/gen_generic.py` -> `src/generated_generic.cuh`: conv forward / backward for x irreps up to 0e+1o+2e
  (one block per receiver, one thread per channel, radial spline per edge; the backward reduces
  each edge's dE/dvec per warp, sender-feature gradients by atomics), the factored products (3 configurations) and
  the SH gradient (sympy).
- `src/generic.cu`: layer loop forward and reverse with GEMMs through the engine's autotuned `sgemm_rm`: linear_up,
  conv, message linear, density / `avg_num_neighbors` / `(beta D + alpha)` normalisation, the nonlinear blocks'
  residual linear, gate and linear_2, species skip connections, product + product linear, readouts (linear, SiLU MLP,
  bias MLP), ZBL, force / virial assembly. Layer 0 (input = species embedding) runs as an outer sum over the radial
  chain's 66 columns per sender species, with the conv weights, the layer input, the path factors and the message
  linear folded into one matrix per degree (`first_fwd_kernel` / `first_bwd_kernel`): its table is 66 columns
  instead of 4 C + 1.
- `src/polar.cu` (MACE-POLAR-1): per-layer spin-charge sources, the layer feature mixer, the Fukui constraint, two
  SCF steps of GTO electrostatic features (k-space: structure factors per k over atoms, feature / gradient passes per
  atom over k with tiled k data, double accumulation, exact fixed-point phases n . s mod 1) and the one-body charge
  update (embeddings, sparse uvu products as GEMMs, LayerNorm MLP, gated readout), the local electron energy and the
  k-space Coulomb energy, forward and reverse. Neutral singlet systems in zero external field, fully periodic.
  The stress includes the k-space terms' cell dependence (MACE PR 1718, as FastMACE): the installed PolarMACE
  (0.3.16) hands those terms the unstrained geometry, so its autograd stress misses up to 2e-5 eV/A^3 (NaCl); the
  gate's stress reference for POLAR is the float64 energy's finite difference.

### 5.2 Gates (`python/test_gate.py MODEL --set ...`; 3x MACE float32-vs-float64 rule, cells strained / rattled)
Sets as FastMACE's: `oxide` (NaCl, SrTiO3), `organic` (urea 3x3x3, naphthalene 2x2x2, paracetamol 2x2x2,
water_216), `crystals` (NaCl, NaNbO3, NbCl5 R2SCAN cells); two configurations each. `--ref-device cpu` where the
float64 reference does not fit the GPU (OFF23-L / 23b / 24 on the 16 GB 5060 Ti, OFF23-L and POLAR-1-L on the
20 GB Ada).

| Model | Set | RTX 5060 Ti | RTX 4000 Ada |
|---|---|---|---|
| OMAT-0 small | oxide | PASS | PASS |
| OFF23 small / medium | organic | PASS / PASS | PASS / PASS |
| OFF23 large, OFF23b medium, OFF24 medium | organic | PASS (CPU refs) | PASS (worst 1.46 / 1.23 / 0.27; OFF23-L CPU refs) |
| MH-1 `omat_pbe`, `matpes_r2scan`, `mp_pbe_refit_add`, `oc20_usemppbe` | crystals | PASS (S59 build: worst force 0.87 / 0.86 / 0.80 / 0.84, stress 0.27 / 0.32 / 0.25 / 0.26) | PASS (first two) |
| MH-1 `omol`, `spice_wB97M` | organic | PASS (S59 build: worst force 0.32 / 0.28, stress 0.63 / 0.38) | PASS (`omol`) |
| MP-0 medium, MP-0 large | oxide | PASS (S59 build: worst force 0.72 / 1.31, stress 2.97 / 0.94) | not run |
| MP-0b3 medium, MPA-0 medium (specialised, every forced variant) | oxide | PASS (S59 build, 30 / 30 rows: worst force 1.06 / 1.68, stress 0.87 / 0.89) | not run |
| MatPES PBE, r2SCAN (specialised, every forced variant) | crystals | PASS (S59 build, 18 / 18 rows: worst force 0.68 / 1.57, stress 1.10 / 1.04) | not run |
| OMOL-0 extra large | organic: neutral singlet, +1 doublet, -1 doublet, neutral triplet | PASS (S57 path; worst force 0.86 / 0.31 / 0.82 / 0.79, stress 0.79 / 0.19 / 0.63 / 0.47) | not run |
| POLAR-1-S | crystals | PASS (worst ratio 1.30) | PASS (worst 0.99) |
| POLAR-1-M | crystals | PASS (worst 0.97) | PASS (worst 1.14) |
| POLAR-1-L | crystals | PASS (worst 1.16) | PASS (CPU refs; worst 2.69, force NaNbO3_c0) |

POLAR-1-S NaNbO3_c0 is far from float32-exact in MACE itself (MACE float32 vs float64: 27-90 eV/A force,
0.38-1.15 eV/A^3 stress; FastMACE excludes the configuration); LoupMD's error there is 0.15-0.34 of that.
Its energy row is not reproducible run to run: LoupMD's energy there takes one of a few discrete values per
process (dE/atom 0.074, 0.167, 0.679, 0.988 or 2.320 meV against float64; identical within a process, the same set
for the S7 and S8 builds; a discrete set points to the per-process GEMM plan choice, not verified), and MACE's own float32 error, the gate's
bound, also changes between runs (0.44 meV in one). With 2.320 the row failed once for S7 and once for S8 and
passed for S8 in another run; force and stress ratios stayed at or below 0.29.
Since S9 the gate leaves the configuration out (`--include-unstable` gates it; FastMACE excludes it too). A cuBLAS
workspace per lane (review U4) left the per-process values as they were, with plans re-timed per process
(inconclusive). Both loaders also measure the 8-byte tables'
error at load (exact maxima, value and d/dr, whole table and r >= 1 A clipped exactly) and refuse above 5e-4 of the column maximum;
current bundles measure 2.7e-5 to 1.7e-4 (ledger rv8-fixes).
`tests/test_gate_wrapper.py` runs the gate for OMAT-0 small, OFF23-S / M, MH-1 (two sets) and POLAR-1-S / M.
The MACE references are cached per model, set, head and reference GPU (`LOUP_GATE_CACHE`, default
`~/.cache/loupmd/gate_refs`; `--no-ref-cache` recomputes):
a repeat gate costs only LoupMD's evaluations (POLAR-1-S crystals 40 -> 6 s; the ~30 min of the POLAR-1-L CPU-reference
gate is the references), and the float32 bound no longer changes between runs.

### 5.3 MD Speed (`python/bench_family.py`: Langevin 300 K, 1 fs, skin 0.5 A, 100 warm-up + median of 6 x 256 steps)

| Model | System (atoms) | 5060 Ti ms/step (S7) | 5060 Ti before S7 | Ada (before S7) | FastMACE (documented, 5060 Ti unless noted) |
|---|---|---|---|---|---|
| OMAT-0 medium (specialised) | NbCl5 (1,944) / SrTiO3 (5,000) | 1.834 / 8.376 | 1.843 / 8.488 | 2.085 / 9.384 | |
| OMAT-0 small | NbCl5 / SrTiO3 | **1.432 / 6.416** | 2.175 / 10.014 | 2.263 / 10.824 | |
| OFF23 small / medium / large | water 2x2x2 (5,184) | **3.869 / 9.311 / 33.722** | 5.593 / 13.093 / 47.829 | 5.960 / 14.620 / 59.736 | |
| OFF23b medium / OFF24 medium | water 2x2x2 | **13.585 / 13.518** | 19.680 / 19.602 | 22.060 / 22.010 | |
| OFF23 medium | water_216 (648) | **1.553** | 2.121 | 1.969 | H200 1.80 |
| MH-1 `omat_pbe` | NaCl 3x3x3 (216) | **1.780** | 2.334 | 2.001 | 2.65-2.90 (`matpes_r2scan`) |
| MH-1 `omat_pbe` | NbCl5 / SrTiO3 | **13.781 / 41.388** | 15.017 / 46.386 | 16.233 / 51.176 | NbCl5 18.7-21.3 |
| MH-1 `omol` | water 2x2x2 | **43.302** | 50.345 | 54.863 | |
| POLAR-1 S / M / L | NaCl 3x3x3 (216) | **2.167 / 4.671 / 8.907** | 2.492 / 5.273 / 10.076 | 2.291 / 4.928 / 10.227 | 2.19-2.23 / 4.51-4.59 / 9.74-9.81; Ada M 4.88-4.90 |
| POLAR-1 S / M / L | NaCl 6x6x6 (1,728) | **11.637 / 25.418 / 51.876** | 15.975 / 30.484 / 59.997 | 16.149 / 33.263 / 71.207 | |
| POLAR-1-M | NbCl5 | **29.739** | 37.682 | 39.984 | 31.1-31.5; Ada 37.4 |

S7 (13 November, ledger entries pol-sf-f32, gen-fork-zero-slab, gen-skip-fold, gen-conv-v2): GEMM lanes and one zero
slab in the generic and long-range passes, the skip-message and readout-tail folds, conv kernels with the
specialised path's 8-byte cubic tables, staged edges, multi-receiver forward and late pull-back backward, and
float32-chunked structure factors. Every gate passes on the 5060 Ti (section 5.2 sets); 47 tests pass. The Ada
column is the fifth update's (the workstations were not used for S7). Against FastMACE's raw A/B lines of 11
November on the same 3x3x3 cells, LoupMD's POLAR-1 is now 1-3% faster (S), 2-4% slower (M, NaCl 216), 4-6% faster
(M, NbCl5) and 9% faster (L); MH-1 NaCl 216 is 33-39% faster than FastMACE's documented `matpes_r2scan` range.
These are FastMACE's adopted configurations; its non-adopted POLAR-1-L pair-`w'` variant runs NaCl 216 in
9.11-9.19 ms (slower on NbCl5), which LoupMD's 8.907 still beats by 2-3%.
The FastMACE column is its documented numbers (`docs/benchmarks.md`, `docs/optimisation-ledger.md`,
`results/ada-5060-screen-20261110/`), not re-measured here. POLAR-1 rows run with float32 SIMT GEMMs (5.4).
POLAR-1-S / -L on NbCl5 3x3x3 are not listed: the model is unstable there (5.5).

S8 (13 November, 5060 Ti only; ledger gen-conv-l2-split, gen-first-bwd-v2, gen-conv-fwd-r1, gen-conv-fwd-l2-split,
gen-first-fwd-race). Times are the S8 side of ABBA runs against S7 (`ab_fam.py`, same protocol as above):

| Model | System | S7 | S8 | b/a |
|---|---|---|---|---|
| OMAT-0 small | SrTiO3 / NbCl5 | 6.511 / 1.456 | **6.109 / 1.381** | 0.938 / 0.948 |
| OFF23 small / medium / large | water 2x2x2 | 3.869 / 9.427 / 34.198 | **3.793 / 8.927 / 29.714** | 0.980 / 0.947 / 0.869 |
| OFF23 medium | water_216 | 1.549 | **1.431** | 0.924 |
| OFF24 medium | water 2x2x2 | 13.685 | **12.762** | 0.933 |
| MH-1 `omat_pbe` | NaCl 3x3x3 | 1.841 | **1.792** | 0.974 |
| POLAR-1 S / M / L | NaCl 3x3x3 | 2.140 / 4.787 / 8.856 | **2.136 / 4.638 / 8.739** | 0.998 / 0.969 / 0.987 |
| POLAR-1-M | NbCl5 | 30.031 | **30.068** | 1.001 |

POLAR-1-S / -M NaCl rows are with freshly tuned GEMM plans on both sides (ledger meas-plan-variance: with each
build's cached plans they read 1.019 / 1.013). Against FastMACE (same cells, adopted configurations): POLAR-1-S 2-4%
faster, POLAR-1-M NaCl 1-3% slower, NbCl5 3-5% faster, POLAR-1-L 10-11% faster; MH-1 33-38% faster. Gates: every
set in 5.2 passes except the POLAR-1-S NaNbO3_c0 energy row in one of six S8 runs (and one of five S7 runs; see
5.2); 55 tests pass.

S8 + POLAR k split (ledger pol-k-split: the per-atom k-space passes split over k chunks with a fixed-order
reduction), ABBA against S8 with fresh GEMM plans on both sides, ms/step:

| Model | System | S8 | S8 + k split | b/a | FastMACE (adopted) |
|---|---|---|---|---|---|
| POLAR-1-S | NaCl 3x3x3 / 6x6x6 | 2.148 / 11.833 | **1.946 / 10.363** | 0.906 / 0.876 | 2.19-2.23 / – |
| POLAR-1-M | NaCl 3x3x3 / NbCl5 | 4.801 / 29.917 | **4.598 / 27.751** | 0.958 / 0.928 | 4.51-4.59 / 31.1-31.5 |
| POLAR-1-L | NaCl 3x3x3 | 8.758 | **8.563** | 0.978 | 9.74-9.81 |

Against FastMACE's adopted configurations LoupMD's POLAR-1 is now 11-13% faster (S), level to 2% slower (M, NaCl
216), 11-12% faster (M, NbCl5) and 12-13% faster (L). POLAR-1-S / -M / -L crystals gates pass (POLAR-1-L with CPU
references).

S10 (ZBL in the force-assembly pass, `gen-zbl-in-assembly`): OMAT-0 small `SrTiO3` 6.06 -> **5.87 ms/step** (0.969); MH-1 `NbCl5` 13.33 -> 13.27, OFF23 medium water 8.88 -> 8.85, POLAR-1-M `NbCl5` 27.28 -> 27.34.
S11 (layer-0 `K0` alignment, `gen-first-hidden-k0-align`): OMAT-0 small `SrTiO3` 5.88 -> **5.62 ms/step** (0.956); OFF23-M water 8.85 -> **8.67** (0.979); OMAT-0 medium generic `SrTiO3` 10.56 -> **10.30** (0.976); MH-1 `NbCl5` 13.27 -> 13.16 (0.992).
S12 (edge-parallel layer-0 backward + ZBL + assembly, `gen-first-bwd-edge`): OMAT-0 small `SrTiO3` 5.62 -> **5.13 ms/step** (0.913); OFF23-M water 8.65 -> **8.28** (0.957); OMAT-0 medium generic `SrTiO3` 10.30 -> **9.80** (0.951); MH-1 `NbCl5` 13.20 -> **13.07** (0.990); POLAR-1-M `NbCl5` 27.40 -> **27.27** (0.995).
S13 (fused product + `ro1` readout + 128-thread `gprod`, `gen-gprod-ro1-fused`): OMAT-0 small `SrTiO3` 5.12 -> **4.99 ms/step** (0.976, down from 6.06 before S10); OFF23-M water 8.29 -> **8.16** (0.984); OMAT-0 medium generic `SrTiO3` 9.81 -> **9.67** (0.986, kernel gap vs specialised cut from +2,144 to +913 us); POLAR-1-M `NbCl5` 27.28 -> **27.14** (0.995).
S14 (`C_FIX` + prelocated `act_slab`/`act_t_sx` on `gconv_fwd`/`gconv_bwd` + 256 thr `L1`, `gen-gconv-cfix-preloc`): OMAT-0 small `SrTiO3` 4.91 -> **4.73 ms/step** (0.964, down from 6.06 pre-S10); OFF23-M water 8.00 -> **7.43** (0.928, down from 8.88 pre-S10); MH-1 `NbCl5` 12.82 -> **12.50** (0.976); POLAR-1-M `NbCl5` 27.20 -> **27.06** (0.995); OMAT-0 medium generic `SrTiO3` 9.67 -> **8.91** (`1.051x` of specialised `8.31 ms`, down from `1.273x` pre-S10).
S15 (`pol-sf-tiled`, `spl-v6-cubic-cg-fold`, `gen-conv-fwd-cg-fold`, 14 November, 5060 Ti only):
- `sf_kernel_tiled` (`src/polar.cu`, `c24b6c4`): `POLAR-1-M` `NbCl5` 27.00 -> **25.00 ms/step** (`0.926`, `sf_kernel<16/8/4>` 900/455/233 -> 316/150/84 us), `NaCl` 216 4.41 -> **4.36** (`0.990`), `POLAR-1-S` `NaCl` 1.92 -> **1.86** (`0.966`), `POLAR-1-L` `NaCl` 8.36 -> **8.31** (`0.993`).
- `spline_forward_inter1` v6 (`ca23b3c`, OMAT-0 medium): `SrTiO3` 8.40 -> **8.28 ms/step** (`0.985`, frozen `R1p` 1,403 -> 1,306 us), `NbCl5` 1.863 -> **1.853** (`0.994`); all forced `LOUP_SPLINE_VARIANT` gates pass. Removed dead `packed8_table_1` / `packed8_bias` (~45 MB) from `export_omat0.py` and `model_loader.cu`.
- `emit_conv_fwd` CG linear-form fold + `s_pw`: kernel profiles `OFF23-M` `gconv_fwd_L1` `1,688 -> 1,572 us`, `OFF23-L` `gconv_fwd_L2b` `3,029 -> 2,898 us`; ABBA `OMAT-0` small `SrTiO3` 4.82 -> **4.77 ms/step** (`0.989`), `OFF23-L` water 28.30 -> **28.00** (`0.989`), `OFF23-M` tie (`7.53 -> 7.51`, `0.997`), `POLAR-1-M` `NbCl5` tie (`25.02 -> 24.99`, `0.999`), `MH-1` `NaCl` `1.75 -> 1.71` (`0.981`, inconclusive due to `1.682/1.746` rep spread). Cached 3x gates pass across `OMAT-0` medium/small, `OFF23-S/M/L`, `OFF24-M`, `MH-1`, `POLAR-1-S/M`; pytest suite passes.
S16 (`gemm-own-n144`, `gemm-own-sgemm64`, 14 November, 5060 Ti only):
- `sgemm128_kernel<4, false>` partial-`N` `128 x 64` tiles + `sgemm64` (`64 x 64 x 16` `cp.async`, `128` threads, autotuned for `sm_120` and `M <= 1600`): fresh-plan ABBA `POLAR-1-S` `NaCl216` 1.85 -> **1.61 ms/step** (`0.873`, vs FastMACE `2.19-2.23`), `POLAR-1-M` `NaCl216` 4.30 -> **3.61** (`0.839`, vs FastMACE `4.51-4.59`), `POLAR-1-L` `NaCl216` 8.24 -> **7.22** (`0.876`, vs FastMACE `9.74-9.81`), `MH-1` `NaCl216` 1.71 -> **1.57** (`0.919`), `OFF23-M` `water216` 1.39 -> **1.30** (`0.938`), `MH-1` `NbCl5` tie (`12.59 -> 12.59`, `1.000`). Gates pass on `POLAR-1-S/M`, `MH-1` (crystals + organic), `OMAT-0-S/M`, `OFF23-M`; `POLAR-1-L` `cmp_libs` `dF = 2.07e-7`; pytest suite passes.
S17 (`gemm-own-sgemm32`, 14 November, `sm_120` only):
- `sgemm32_kernel` (`32 x 32 x 16` `cp.async`, `64` threads) + deterministic `split_k` (`1, 2, 4` into per-stream `4 MB` workspace, autotuned for `sm_120` and `M <= 1600`): fresh-plan ABBA `POLAR-1-S` `NaCl216` 1.65 -> **1.50 ms/step** (`0.909`), `POLAR-1-M` `NaCl216` 3.70 -> **3.46** (`0.936`), `POLAR-1-L` `NaCl216` 7.28 -> **6.89** (`0.947`), `MH-1` `NaCl216` 1.61 -> **1.53** (`0.948`), `OFF23-M` `water216` tie (`1.34 -> 1.35`, `1.006`), `MH-1` `NbCl5` tie (`12.55 -> 12.59`, `1.003`). Gates pass across `POLAR-1-S/M`, `MH-1`, `OMAT-0-S/M`, `OFF23-M`; `67 passed`.
S18 (`gemm-own-narrow-n`, 14 November, `sm_120` only):
- Narrow-`N` `sgemm64` / `sgemm32` above `M = 1600` (`N <= 64` / `N <= 32`, or `M <= 2500 && N <= 160`): fresh-plan ABBA `POLAR-1-M` `NbCl5` 24.66 -> **24.48 ms/step** (`0.9925`), `MH-1` `NbCl5` 12.62 -> **12.51** (`0.9907`), `OFF23-M` `water` tie (`7.41 -> 7.42`), `OMAT-0-S` `SrTiO3` tie (`4.71 -> 4.71`). Cached 3x gates + `576`-atom `NbCl5` `POLAR-1-M` full force+FD-stress 3x gate (`dF 0.66x`/`dS 0.00x`) + `1,944`-atom `NbCl5` `MH-1` full 3x gate (`dF 0.31x`/`dS 1.14x`) + `POLAR-1-M` force-only 3x check (`dF 0.47x`, no stress check) + `67 passed` pass.
S19 (`gen-conv-l2-cfix-preloc`, 14 November, 5060 Ti only):
- `C_FIX` (`128/224`) on the lx 2 convs, `PRELOC` on `gconv_bwd_L2`, prelocated slabs for `pair_sym = 0` models: fresh-plan ABBA `OFF23-L` water 27.81 -> **27.16 ms/step** (`0.977`), water216 3.85 -> **3.79** (`0.985`), `POLAR-1-L` `NaCl216` 6.83 -> **6.77** (`0.991`), `POLAR-1-M` / `MH-1` `NbCl5` tie. The `OFF23-L` gain is the lx 2 `C_FIX` / `PRELOC` (it was already prelocated); `pair_sym = 0` prelocation alone is a tie on `NbCl5`.
S20 (`gemm-own-raster`, 14 November, 5060 Ti only):
- Own SGEMMs rasterised `N`-tile-fastest (`sgemm128` via an `N_FASTEST` template only on `sm_120`; other GPUs run SASS identical to S19), which improves observed cold-`A` reuse (timing only, no DRAM-byte counters; `13608x512x512` cold 529 -> 425 us): fresh-plan ABBA `MH-1` `NbCl5` 12.52 -> **12.22 ms/step** (`0.977`), `POLAR-1-M` `NbCl5` 24.50 -> **24.19** (`0.987`), `OMAT-0-M` `SrTiO3` 8.18 -> **8.08** (`0.989`), `OFF23-M` water 7.41 -> **7.35** (`0.993`), `OMAT-0-S` / `POLAR-1-S` tie.
S21 (`gen-first-bwd-edge-minb`, 14 November, 5060 Ti only):
- `first_bwd_edge_kernel<COLS = 2>` capped at 80 registers in sm_120 device code (latency-bound, 41.7% occupancy at 94 registers): kernel `-6%` (`OMAT-0-S` 510 -> 480 us, `OFF23-M` 358 -> 334 us); step, both argument orders combined: `OMAT-0-S` `0.9954`, `OFF23-M` `0.9979` (one `1.003` run). Lane-per-edge rewrite rejected (`-43%` instructions, but 168 registers: `+13%`). Own `sgemm128` and the `L1` convs profiled near their limits (`s21-limiters` in the ledger).
S29 (`gen-conv-bwd-l1-c256`, 14 November, 5060 Ti only):
- `C_FIX = 256` on `gconv_bwd_L1` for `MACE_MPtrj_2022.9` (`C = 256`; runtime `C` held 128 registers, now 62): final installed build vs rebuilt parent, fresh-plan ABBA `SrTiO3` (5,000 atoms) `12.05 -> 11.94 ms/step` (`0.9913`; reversed `0.9912`). 3x gate (oxide) passes; `pytest` 67 passed. Found by a sweep over every installed model; only `MPtrj` uses the `C = 256` lx 1 branches.
S30 (`gen-conv-fwd-l1-c256`, 14 November, 5060 Ti only):
- `C_FIX = 256` (`PAIR = false`, 80 registers) on `gconv_fwd_L1` for `MPtrj`: kernel 1960 -> 1860 us, ABBA `SrTiO3` `12.04 -> 11.94 ms/step` (`0.9921`; reversed `0.9910`). 3x gate (oxide) passes; `pytest` 67 passed. S29's `PAIR + C_FIX` forward (104 registers) was a tie.
S31 (`gen-conv-bwd-512thr`, 14 November, 5060 Ti only):
- 512-thread blocks for `gconv_bwd_L1` at `C = 256` and `gconv_bwd_L2lp` at `C = 224` (their 39-45 KB `s_q` allows 2 blocks per SM: 16 warps at 256 threads, 32 at 512; measured achieved occupancy 29% -> 59% on `MPtrj`): final binary vs S30, fresh-plan ABBA `MPtrj` `SrTiO3` `11.93 -> 11.58 ms/step` (`0.9703`; reversed `0.9707`), `OFF23-L` water216 `3.83 -> 3.76` (`0.9838`; reversed `0.9786`); scratch build `OFF23-L` water (5,184 atoms) `26.95 -> 26.48` (`0.9825` / `0.9839`). 3x gates (`MPtrj` oxide, `OFF23-L` organic) pass; `pytest` 67 passed.
S34 (`nl-lane-groups`, 14 November, 5060 Ti only):
- Neighbour count / fill with 8 lanes per atom (bins split round-robin) instead of one thread per atom (162 warps for 5,184 atoms, 8% occupancy); fill is designed to reproduce the one-thread row order (per-bin offsets in shared memory); edge arrays were not compared directly, but `MH-1` `NbCl5` (deterministic path) matches the old build to `dE 3.6e-12` and the `POLAR-1-S` replicated-cell test passes. Per rebuild: `SrTiO3` 433 -> 292 us, `NbCl5` 284 -> 110, water 439 -> 302. Water rebuilds every ~4 steps: ABBA `OFF23-S` water `0.9945` / `0.9856`, `OFF23-M` `0.9961` / `0.9943`, `OMAT-0-S` `0.9961` / `0.9972`. Other GPUs keep the one-thread kernels (SASS identical to S33). Gates (`OFF23-S` organic, `OMAT-0-S` oxide, `MH-1` / `POLAR-1-S` crystals) pass; `pytest` 67 passed.
S38 (`gemm-own-n96`, 14 November, 5060 Ti only):
- `128 x 96` own-SGEMM tile for `N = 96` products (the `C = 96` models; `sgemm128_kernel<8>` spent a quarter of its FMAs on padding): standalone `0.79-0.84` per shape, bit-identical; final binary, fresh-plan ABBA `OFF23-S` water `3.088 -> 3.046 ms/step` (`0.9866`), reversed `3.101 -> 3.026` (`0.9758`). Other widths and other GPUs unchanged (SASS identical). Gates (`OFF23-S` organic, `OMAT-0-S` oxide) pass; `pytest` 67 passed.
S40 (`gen-conv-bwd-l0-red`, 14 November, 5060 Ti only):
- `gconv_bwd_L0`'s per-edge `edge_dvec` update as a fire-and-forget atomic add (one writer per edge; `ncu` attributed ~4% of its stall samples to the read-modify-write's load): kernel `-8%` (`OMAT-0-S`) / `-16%` (`OFF23-S`); final binary, fresh-plan ABBA `OMAT-0-S` `SrTiO3` `4.689 -> 4.601 ms/step` (`0.9813`), reversed `0.9768`; `OFF23-S` water `3.052 -> 2.966` (`0.9718`), reversed `0.9763`. lx 1 / 2 kernels and other GPUs unchanged (SASS identical). Gates (`OFF23-S` organic, `OMAT-0-S` / `MPA-0` oxide) pass; `pytest` 67 passed.
S48 (`gen-conv-bwd-l2-pullr`, 14 November, 5060 Ti only):
- `gconv_bwd_L2` (x-degree-2 backward convs of the lx 2 models) takes its radial pull from the x-gradient contraction (`sum_m1 x[m1] Cg[m1]`) instead of the full sum over every `A x Y` term: SASS 1624 -> 1464, kernel `-16%` (`OFF23-L` water) / `-18%` (water216); fresh-plan ABBA `OFF23-L` water `26.47 -> 25.47 ms/step` (`0.9624`), reversed `0.9609`; water216 `0.9666` / `0.9639`; `POLAR-1-L` `NaCl216` tie (`1.0024` / `0.978`). Other GPUs and other kernels unchanged (SASS identical). Gates (`OFF23-L` organic, `POLAR-1-L` crystals; CPU references) pass; `pytest` 67 passed.
S52 (`pol-dot-fold`, 14 November, 5060 Ti only):
- POLAR SCF and electron-energy weights composed with `dot^T` at load (the two linear maps had nothing between them): one C x C GEMM per SCF step and l in each direction instead of two, none for the electron energy (`POLAR-1-M` `NaCl216` GEMM launches 165.6 -> 149.5 per step). Final binary, fresh-plan ABBA: `POLAR-1-S` `NaCl216` `1.511 -> 1.377 ms/step` (`0.911`), reversed `0.914`; `POLAR-1-M` `NaCl216` `3.461 -> 3.073` (`0.888`), reversed `0.922`; `POLAR-1-M` `NbCl5` `24.39 -> 22.11` (`0.906`), reversed `0.906`; `POLAR-1-L` `NaCl216` `6.908 -> 6.122` (`0.886`), reversed `0.895`. Only on `sm_120` (other GPUs run the unfolded GEMMs; existing kernels SASS-identical on `sm_89` / `sm_90` / `sm_120`); `LOUP_POLAR_FOLD=0` selects the unfolded path for testing. Gates (crystals; `POLAR-1-L` CPU references) pass on both paths; `pytest` 67 passed.
S57 (`gen-chain-fold`, 14 November, 5060 Ti only):
- OMOL-0 XL: the non-last layers' outputs are never formed. Each layer's product output (and layer 0's species-only skip input) goes straight into every later consumer (the next layers' `linear_up` and the last readout) through weights composed in double at load (`W_p W_sc ... W_up`), and the backward follows the same weights; the 1024 x 1024 `W_p` / `W_sc` products of layers 0 and 1 and the 1024-wide `linear_up` of layers 1 and 2 go in both directions. Fold against `LOUP_CHAIN_FOLD=0` on one build: water216 `29.02 -> 23.83 ms/step` (`0.821`), reversed `0.821`; water `217.9 -> 179.8` (`0.825`), reversed `0.825`. Exact to float32 rounding against the plain path (water216 `dF` 1.8e-6 eV/A); organic gates pass on the fold path (neutral worst 0.86 / 0.79, +1 doublet 0.31 / 0.19). Only OMOL-0 qualifies among the supported families (all layers gated, no readout before the last, no per-species skip, no long range); other GPUs keep the plain path (`generic.cu` kernels SASS identical for `sm_89` / `sm_90` / `sm_120`).
- OMOL-0 XL after S57, `nsys` water216 (installed library, default plan cache): kernel sum 29.2 ms/step (35.8 before), GEMM-like 73% (21.3 ms): `sgemm128<8>` 12.6 ms (41.4 launches, mean 305 us), `sgemm64` 3.5, CUTLASS SIMT 2.4, `magma` 0.8, `sgemm32` 1.1; `gate_bwd` 1.6, `gate_fwd` 0.9; all `gconv` kernels together 3.6 ms; products 1.1 ms. The 1024-wide message (`W_2`) and linear GEMMs are the limiter.
- OMOL-0 XL against FastMACE (same 5060 Ti, 14 November; FastMACE `fastmace` 1.0.0 at the current FastMACE checkout, float32, graph tier, `radial_tables=off`: its default plan here falls back, and the fallback's float64 reference does not fit 16 GB; FastMACE's own OMOL-0 5060 Ti number was measured that way too). MD under `bench_family`'s protocol on both sides (`.tmp_o/fm_md.py` for FastMACE): water216 LoupMD **23.86 ms/step**, FastMACE 74.15 (3.1x); `NbCl5` 3x3x3 (1,944 atoms) **60.37**, FastMACE 137.58 (2.3x; FastMACE documented 141.5). Accuracy against MACE float64 on the gate's organic configurations (`.tmp_o/acc_om.py`). FastMACE through its one-off `ForceField.evaluate` with `radial_tables=off` and its rule-13 re-plan applied (`reassociate_first` / `segmented_first_sum` off: their setup self-check fails on urea, and the one-off path raises instead of re-planning); the timed MD plans above kept those two rewrites on, so these FastMACE errors are of that variant plan, not of the timed graph path: neutral worst force error LoupMD 6.6e-5 eV/A (RMS 2.4-7.6e-6), FastMACE 1.23e-4 (RMS 1.0-2.0e-5), MACE float32 up to 1.0e-4; charge +1 doublet LoupMD 1.85e-4 (RMS up to 2.7e-5), FastMACE 1.04e-3 (RMS up to 2.4e-4), MACE float32 up to 9.8e-4. Worst ratios to MACE float32's error (force / stress): LoupMD 0.88 / 0.79 and 0.31 / 0.22, FastMACE 1.21 / 1.52 and 1.33 / 2.49; both within the 3x rule.
S59 (`gen-species-tables`, 14 November, 5060 Ti only):
- Layer 0's species-only products (its input is the embedding row) become per-species tables composed in double at load and gathered per atom: the gated block's residual and, under S57, the skip terms. OMOL-0 XL water216 `23.78 -> 23.26 ms/step` (`0.978`), reversed `0.977`; MH-1 NaCl216 `0.988` / `0.991`, NbCl5 `0.997` / `0.996`. POLAR-1-L showed no win in the tables-first order, so the long-range models keep the GEMMs. Gates pass (OMOL-0 at four charge / spin settings, MH-1 crystals both heads and `omol` organic); kernels SASS-identical; `pytest` 71 passed; `LOUP_SPECIES_TABLES=0` disables.
S64 (`gen-conv-fwd-variants`, 15 November, 5060 Ti only):
- The generic `gconv_fwd_L1` at `C = 128` now has the specialised spline forward's receivers-per-block choice: R1 paired (old), R1 unpaired, R5 (640 threads), timed per layer and system at the first MD rebuild. Kernel up to -21% (OMAT-0 medium generic SrTiO3 2,200 -> 1,742 us; OFF24-M water 2,723 -> 2,437). Full-step ABBA tuned / old, both orders: MP-0 medium SrTiO3 `0.968` / `0.967`, OFF24-M water `0.978` / `0.978`, OFF23-M water `0.975` / `0.980`, MH-1 SrTiO3 `0.988` / `0.991`; small cells (216-648 atoms) and POLAR-1-M tie. Gates pass for both forced variants on all seven families; existing kernels SASS-identical; `LOUP_CONV_TUNE=0` disables.

### 5.4 Fixes Found by the Family Work
- **POLAR LayerNorm-MLP reverse wrote its GEMM output over its own input** (`mlp_bwd`: after the first hidden layer
  the destination buffer was the one holding dz). Harmless while one GEMM tile covered all rows; from ~1,700 atoms
  the tiles raced: perfect NaCl 6x6x6 with POLAR-1-S had 3-11 eV/A forces (random per call) and MD went to NaN.
  Smaller cells changed at the rounding level (perfect NaCl 7x4x4: residual force 1.1e-4 -> 2.5e-6 eV/A). Fixed
  (alternating scratch buffers, LayerNorm reverse in place).
- **k-space phases as exact fixed point**: fractional coordinates are stored as 32-bit fixed point and n . s mod 1
  comes out of wrapping integer arithmetic, so the phase error no longer grows with |n| (elongated NaCl 10x4x4,
  perfect lattice: residual force 3.0e-4 -> 8.7e-5 eV/A).
- **Long-range models use float32 SIMT GEMMs only**: with the 3xTF32 hybrid GEMMs the RTX 4000 Ada failed the
  POLAR-1-S / -L stress gates (NaCl stress ratios 3.4-5.2, POLAR-1-L NaNbO3 up to 6.3) and POLAR-1-S NaNbO3_c0
  returned NaN; with SIMT GEMMs every POLAR configuration passes on both GPUs (5.2). Other families keep the
  hybrid GEMMs (their gates pass on both GPUs).
- **Pipelined MD margin** (section 3.2, M1): 1 of ~1,600 steps of three 300 K benchmarks had run on an
  incomplete list with a two-step margin; three steps: 0 in every run.

### 5.5 Known Limitations
- MACE-POLAR-1: neutral, singlet, zero field, fully periodic (`include/loupmd.h`).
- MACE-POLAR-1 itself is not stable in 300 K MD on some larger cells: NbCl5 3x3x3 (S, L) and SrTiO3 from 6x6x6
  (M). MACE's own float64 forces after 5 fs of NVE from the same start agree with LoupMD (SrTiO3 6x6x6, POLAR-1-M:
  max |F| 84.83 eV/A in both, difference 7e-4 eV/A; NbCl5 3x3x3, POLAR-1-S: 6.74 eV/A after 1 fs in both, from
  0.61 eV/A at the unrelaxed ideal cell). The same models are stable on NaCl (all sizes run) and NbCl5 2x2x2.
  `bench_family.py` marks such runs `UNSTABLE`.
- Bundles of models with nonlinear (ordered-pair) radial blocks are large: MH-1 74-200 MB, POLAR-1-L 790 MB for
  3 species.
- Generic-path hot spots (5060 Ti, nsys, SrTiO3 5,000 atoms): MH-1 47 ms/step = SIMT SGEMMs ~50% (cuBLAS CUTLASS
  SIMT and the own SGEMM, chosen by the autotune), conv backward / forward of layer 1 12% / 8%, gate backward /
  forward 10% / 5% (estimated ~1 GB of traffic per backward call: near DRAM bandwidth), first-layer outer sum 5%, products 5%.

---

## 6. Release branch (`release/v1`): Stage 1 of the release plan

Measured on 15 November 2026 on the RTX 5060 Ti only (`sm_120`, WSL2, driver 621.14); the workstations were not
used. Plan and task ids: `PLAN.md`. Baseline: `plans/baseline-20261115.md` (the tree of `bac354c`, whose sources
equal `round6-hardening` `11778cd`; library sha256 `df58a9bd2623a502...`). Head: `release/v1` at `a68cc04`
(library `2e7e904abac64b91...`). Every GPU job ran after the card had been idle and under FastMACE's local GPU
reservation, with its own plan cache (`LOUP_CACHE_DIR=$HOME/.cache/loupmd-release`).

| Check | Baseline | Head |
|---|---|---|
| `pytest -q` | 71 passed | 205 passed (at `7bafa23`) |
| `python/test_gate.py` (OMAT-0 medium, oxide set, every forced variant) | PASS, worst force 1.56, stress 1.71 | PASS, worst force 1.55, stress 1.71 |
| `bench_md.py` A/B, baseline first (`NbCl5` / `SrTiO3`, ms/step) | 1.812 / 8.052 | 1.811 / 8.056 (0.9994 / 1.0006) |
| `bench_md.py` A/B, head first | 1.809 / 8.065 | 1.807 / 8.045 (0.9986 / 0.9975) |
| 100 run calls of 10 steps against one call of 1,000 (`SrTiO3` 5,000 atoms, wall time) | 1.1133, 1.1132 | 1.0106, 1.0056 |
| `thermo_every = 20` against off (`NbCl5` / `SrTiO3`) | not applicable | 1.0005, 1.0041 / 1.0003, 1.0007 |

- The standard benchmarks are a tie in both orders: Stage 1 changes no kernel of the default step.
- A run made of short calls cost 11% more than one long call because every call re-captured its graphs and ran
  a force pass before its first step; both are gone (ledger `md-graph-reuse`).
- Memory plan (`loup_plan_memory`, from a 135-atom probe): predicted 487.8 / 1,878.2 / 3,631.7 MB for 5,000 /
  20,480 / 40,000 `SrTiO3` atoms (OMAT-0 medium) against 513.8 / 1,851.8 / 3,636.5 MB measured (-5.1%, +1.4%,
  -0.1%); OMAT-0 small on the generic path at 5,000 atoms 467.3 against 507.5 MB (-7.9%).
- Open domain (testing only, `loup_debug_evaluate_domain`): a periodic cell with its images as ghosts against the
  periodic call, energy / forces / virial stress: OMAT-0 small 3.1e-7 eV/atom, 2.0e-6 eV/A, 2.9e-8 eV/A^3; OMAT-0
  medium on the generic path 1.7e-7, 2.1e-6, 3.2e-8; MH-1 1.5e-8, 1.4e-6, 4.2e-9; OFF23 medium 1.0e-8, 7.4e-7,
  1.1e-9; OMOL-0 extra large (three layers, chain fold off) 7.9e-10, 7.8e-7, 4.3e-10. The periodic call on other
  GEMM kernels (`LOUP_NO_HYBRID=1`) changes by 3.9e-10 to 6.3e-8 eV/atom and 5.5e-7 to 2.5e-6 eV/A. The energy
  deviation of the two OMAT-0 models is larger than that and its cause is not isolated (`PLAN.md` §1.3 D16).
- Energy-only pass (`loup_evaluate2`): equal to the full call's energy within 1e-11 relative on five model paths,
  relaxed and compressed; its speed was not measured.
- Not measured: anything on the RTX 4000 Ada; `nsys` kernel lists; the family benchmarks (`bench_family.py`).

### 6.1 After the merge of `round6-hardening` `e21faa2` (15 November 2026, 04:30 to 05:15)

RTX 5060 Ti only, same conditions as above. Head: `release/v1` at `11edd3c` (library sha256 `9f89f02c26e7792a...`,
`loup_version()` `1.0.0a1+2929486ab9da008c`); the later commits change Python only. Tip: `round6-hardening`
`e21faa2` built from a `git archive` (library `5cbfaca706805301...`). Each library has its own GEMM plan files.

| Check | Tip `e21faa2` | Head |
|---|---|---|
| `pytest -q` | not run | 212 passed (331 s) |
| `python/test_gate.py` (OMAT-0 medium, oxide set, every forced variant) | not run | PASS, 30 rows and 2 reorder checks, worst force 1.52, stress 1.68 |
| `bench_md.py`, tip first, twice (`NbCl5` / `SrTiO3` ms/step) | 1.805 / 8.008; 1.790 / 8.036 | 1.812 / 8.043 (1.0040 / 1.0044); 1.804 / 8.050 (1.0080 / 1.0017) |
| `bench_md.py`, head first, twice | 1.798 / 8.037; 1.802 / 8.038 | 1.807 / 8.019 (1.0052 / 0.9977); 1.793 / 8.030 (0.9946 / 0.9990) |
| `bench_md.py`, tip first, three fresh plan directories a side | 1.822 / 8.029; 1.808 / 8.057; 1.818 / 8.079 | 1.813 / 8.034 (0.9955 / 1.0006); 1.820 / 8.029 (1.0069 / 0.9965); 1.830 / 8.064 (1.0067 / 0.9981) |
| mean of the seven ratios | | 1.0030 / 0.9997 |
| `bench_family.py` OMAT-0 small `SrTiO3` (r6 head head r6) | 4.541 | 4.531 (0.9977) |
| `bench_family.py` OMAT-0 small `NbCl5` (six runs a side) | 1.058 (1.046 to 1.071) | 1.065 (1.043 to 1.115): 1.0066 |
| `bench_family.py` MH-1 `SrTiO3` | 34.978 | 35.010 (1.0009) |
| `bench_family.py` OFF23 medium `water216` (six runs a side) | 1.234 (1.217 to 1.244) | 1.250 (1.203 to 1.289): 1.0127 |
| the same, per block (18 blocks a side, `.tmp_o/water_blocks.py`) | 1.230 (1.127 to 1.317) | 1.223 (1.126 to 1.292): 0.9944 |
| 100 run calls of 10 steps against one call of 1,000 (`SrTiO3`) | not run | 1.0106 |

- Single `NbCl5` runs of the head spread from 1.783 to 1.839 ms (3%) across these jobs, the tip's from 1.781 to
  1.834 ms. The seven ratios' mean for `NbCl5` is 1.0030 with a standard error of 0.0021; `SrTiO3` 0.9997 with
  0.0010. Counted as a tie
  (`PLAN.md` §1.3 D21). The MD loop of the head differs from the tip's by Stage 1's host-side counters only.
- OFF23 medium on the 648-atom water box: no step is pipelined and the rebuild counts are equal in both builds
  (60, 52, 51, ... per 256 steps); its blocks spread by 13 to 15% in both, so the six-run means differ by noise.
  Observation, both builds: the final temperatures of these twelve runs are 272 to 295 K (mean 284 K) for a 300 K
  thermostat; not investigated here.
- Release tools on the same build (`h11` / `h12` snapshots: Python of `93f296e` / `9521e71`):
  `python -m loupmd bench --model omat0-medium` 1.795 / 7.997 and 1.830 / 8.040 ms against `bench_md.py` 1.807 /
  8.034 in between (-0.7% / -0.5% and +1.3% / +0.1%); `--quick` (`SrTiO3` 6x6x6, 1,080 atoms, 2 x 128 steps) 1.787
  and 1.767 ms; `--suite families` for `omat0-small` 4.540 ms; `--quick --compare mace` 395.8 ms/step for
  mace-torch 0.3.16's ASE calculator (float32, 3 x 20 steps) against 1.767: 224 times LoupMD's step;
  `--quick --ab - LOUP_NO_HYBRID=1` 1.774 against 1.781 ms (1.0042). Every record validates against
  `loupmd.bench/1`; the GPU read 1% utilisation before each run.
- `python -m loupmd validate --model omat0-medium`: PASSED, worst ratios energy 0.66, force 1.53, stress 1.67, from
  the reference file `python/test_gate.py` wrote (same key); `test_gate.py` on the same build PASS, 1.52 / 1.68.
  `--structure` on `SrTiO3` 3x3x3 (two copies rattled by 0.05 A, references computed on the GPU): PASSED, 0.89 /
  1.98 / 1.43.
- `lm.calc` (`tests/test_ase_calculator_gpu.py`): ASE `VelocityVerlet`, 1,000 steps of 1 fs on `SrTiO3` 3x3x3 from
  300 K: drift -0.0006 meV/atom/ps, fluctuation 0.0035 meV/atom, 137 K at the end.
- Not measured: anything on the RTX 4000 Ada; `nsys` kernel lists; the gates of the generic families on the merged
  code (`--set organic`, `--set crystals`); `loupmd bench` for models other than OMAT-0 medium and small.

### 6.2 Stage 3 front end and the thermostat target on the device (15 November 2026, 05:20 to 06:10)

RTX 5060 Ti only. Engine build of `e11370e` (library sha256 `6bd216e368ca63f0...`, `loup_version()`
`1.0.0a1+a4baf9f088519beb`) against the build of §6.1 (`9f89f02c26e7792a...`). The change: the Langevin target is
a device scalar that the first half-kick reads and the second advances (ledger `md-langevin-target`).

| Check | Result |
|---|---|
| `bench_md.py`, old build first (`NbCl5` / `SrTiO3`, new / old) | 1.817 / 8.015 against 1.802 / 8.024 ms: 1.0085 / 0.9989 |
| `bench_md.py`, new build first | 1.792 / 8.009 against 1.803 / 8.059 ms: 0.9943 / 0.9938 |
| `python/test_gate.py` | PASS, 30 rows and 2 reorder checks, worst force 1.56, stress 1.73 |
| `pytest -q` (tests of `cb8b020`) | 450 tests, 449 passed and 1 failed (`tests/test_sweeps.py`, the replay schedule on OFF23 medium; it then passed in 12 isolated repeats and had passed in the four full runs since the merge; its message was not kept) |
| ramp 200 to 400 K at 100 K/ps, 5,000 atoms, friction 0.02/fs (`tests/test_schedule_gpu.py`) | mean kinetic temperature 296.6 K, mean target 301.0 K (0.9855); last ten rows 387.4 K for 391.0 K; 7.985 ms/step against 7.839 on the 200 K hold before it |
| `Simulation`, rows and frames every 20 steps against none, 5,000 atoms, wall time of `run()` | 8.078, 8.123 against 7.977, 8.013 ms/step: 1.0132 (1.0132 on the build before, too) |
| resume after 500 of 1,000 NVE steps, 135 atoms | 7.3e-7 A from the uninterrupted run; two uninterrupted runs differ by 9.4e-7 A (7.0e-7 and 7.5e-7 on the build before) |
| `lm.calc`, ASE `VelocityVerlet`, 1 ps, 135 atoms | drift -0.0005 meV/atom/ps |
| `check="gate"` on `SrTiO3` 3x3x3 | passed, worst ratios energy 0.91, force 1.80, stress 1.44 |
| the same with `spline_table_1` of the bundle scaled by 1.02 | `GateFailed`, worst ratios 6,172 / 6,637 / 19,809; no step ran |
| `loupmd md NbCl5_R2SCAN.cif --model omat0-medium --temperature 300 --steps 2000 --timestep 1 --repeat 3 3 3` | 1.883 ms/step with output every 20 steps on `/mnt/d` (1.785 in the engine), 45.9 ns/day, 88 rebuilds, 0 dangerous |
| `loupmd init md ... --profile ultrafast --repeat 2 2 2`, then `loupmd run` (576 atoms, 9,000 steps) | 0.636 ms/step (0.619 in the engine), 135.8 ns/day |
| `lm.md`, `SrTiO3` 4x4x4 (320 atoms), 200 steps | 0.776 ms/step with output (0.675 in the engine) |

- The A/B is a tie: the two orders give 1.0085 and 0.9943 for `NbCl5`, 0.9989 and 0.9938 for `SrTiO3`. The kernel
  now does one FP64 square root and three multiplications more per atom and reads one more value.
- `loupmd md`'s Langevin friction is FastMACE's default, 0.001965/fs; `bench_md.py` uses 0.01/fs, so its 1.80 ms
  and the 1.785 ms here are not the same run.
- Not measured: anything on the RTX 4000 Ada; a schedule on a model of the generic path; the gate inside a run for
  a structure whose float64 reference does not fit the GPU; `loupmd run --resume` on the GPU.

### 6.3 Fixed atoms and the host-side optimisers (15 November 2026, 06:15 to 06:50)

RTX 5060 Ti only. Engine build of `06230cf` (library sha256 `6c2af2c880580942...`) against the build of §6.2
(`6bd216e368ca63f0...`). The engine change adds `loup_set_fixed_atoms` and makes the reported temperature count
3 (N - n_fixed) degrees of freedom; no kernel of the default step changed.

| Check | Result |
|---|---|
| `bench_md.py`, old build first, twice (`NbCl5` / `SrTiO3`, new / old) | 1.0124 / 1.0011 and 1.0050 / 1.0008 |
| `bench_md.py`, new build first, twice | 1.0009 / 0.9986 and 0.9946 / 0.9973 |
| mean of the four | 1.0032 / 0.9995 |
| `python/test_gate.py` | PASS, 30 rows and 2 reorder checks, worst force 1.54, stress 1.68 |
| `pytest -q` at `06230cf` | 458 passed |
| `pytest -q` at `4f20442` | 491 passed, 1 failed: the optimiser-cost test (FIRE 10.5%, LBFGS 87% of the force calls), fixed in the next commit |
| fixed atoms (`tests/test_fixed_atoms_gpu.py`) | 27 of 135 atoms fixed: positions bitwise unchanged after 1,000 Langevin steps; NVE with 20 fixed atoms drifts under 2e-5 eV/atom in 0.6 ps |
| LBFGS and FIRE against ASE's with `lm.calc`, rattled `SrTiO3` 3x3x3, `fmax` 0.05 | 15 steps (ASE 15) and 44 steps (ASE 44); positions within 2.7e-7 and 7.3e-8 A of ASE's |
| the optimiser's own time at 5,000 atoms, 40 steps | FIRE 0.077 ms a step (0.90% of the 8.4 ms force calls); LBFGS 1.407 ms (17.25%) |
| cell relaxation of `SrTiO3` 3x3x3 strained by 2%, `fmax` 0.002 | 157 steps (ASE's `FrechetCellFilter` 157); lattice constants within 1.0e-5 A; stress left 8.1e-5 eV/A^3 |

- The first version of the optimisers used `np.dot` / `np.vdot` on 15,000-element vectors, where NumPy's threaded
  BLAS call cost 0.03 to 0.8 ms each on this machine; plain sums (`einsum`) cost 5 us. LBFGS's two loops are now
  six matrix-vector products and two small triangular solves a step.
- At `1d7380e` the GPU tests of the optimisers and of fixed atoms pass (8 passed; FIRE 0.87%, LBFGS 18.96% of the
  force calls this time). `loupmd relax` on rattled `SrTiO3` 3x3x3: converged after 25 steps, 26 force calls in
  1.31 s. `loupmd md --schedule ramp:0:600:0.5,ramp:600:300:0.5` (135 atoms, 0.420 ms/step with output) and then
  `loupmd quench --min-K 300 --bin-K 150 --per-bin 1`: five frames, 27 to 33 LBFGS steps each, quenched energies
  -7.989531 to -7.989537 eV/atom.
- Not measured: anything on the RTX 4000 Ada; a cell relaxation with a model of the generic path; the full suite
  at `1d7380e` (the last full run, at `4f20442`, had 491 passed and the optimiser-cost test failing).

### 6.4 Centre of mass at rest (`fix_com`) and the current build against the merge build (15 November 2026, 06:55 to 07:35)

RTX 5060 Ti only. Engine build of `42cd77d` (library sha256 `7c835aab4e62167d...`). Two changes: `fix_com`
(the Langevin noise in two passes with FastMACE's mass-weighted projection, used only when it is on; NVE's total
momentum removed at the start of a call), and the Langevin noise coefficients formed once per step by one thread
of the second half-kick instead of per atom (the per-atom square root came in with the ramp of §6.2).

| Check | Result |
|---|---|
| `bench_md.py` against the merge build of §6.1 (`9f89f02c26e7792a`), merge build first, twice (`NbCl5` / `SrTiO3`, new / merge) | 1.0064 / 0.9982 and 1.0047 / 1.0005 |
| the same, new build first, twice | 0.9965 / 0.9964 (one `SrTiO3` run of the new build was disturbed, 8.763 ms, and is left out) and 0.9919 / 0.9976 |
| mean of the four | 0.9999 / 0.9982 |
| `python/test_gate.py` | PASS, 30 rows and 2 reorder checks, worst force 1.55, stress 1.64 |
| `pytest -q` | 496 passed |
| total momentum after 500 Langevin steps, 135 atoms, in units of sqrt(M kB T) | 1.6e-15 with `fix_com`, 1.40 without |
| NVE with `fix_com` from a start with a net drift | the total momentum falls below 1e-7 of its start value; without `fix_com` it is conserved to 1e-6 |
| Langevin with `fix_com` against without, 5,000 atoms, through `lm.Simulation` (a b b a) | 7.992, 7.986 against 7.961, 8.003 ms/step: 1.0009 (0.9995 on the build before) |

- Within one `bench_md.py` job the first run is the fastest for `NbCl5` (1.796 to 1.809 ms) and the later ones are
  slower (1.811 to 1.825), whichever library runs; the two orders cancel this. Across the four jobs the current
  build equals the merge build: nothing of Stage 1 to Stage 4 so far costs speed on the standard benchmarks.
- The build before this one (`fix_com` without the per-step coefficients, library `019f60731d9f9ad0`) read 1.0160 /
  1.0024 and 0.9958 / 0.9982 against the build of §6.3; its full suite: 496 passed and one outdated test.
- Not measured: `fix_com`'s cost on the 1,944-atom `NbCl5` cell; anything on the RTX 4000 Ada.

### 6.5 CSVR, the layout move and the skin chosen from the step size (15 November 2026, 12:05 to 13:10)

RTX 5060 Ti only. Engine builds: `6437f8e` (CSVR; library sha256 `cebdfa27f1af83de...`) and `705af4c` (the layout
move, `loup_get_step_displacement`; `885c3f49f6693412...`). Paths below are those after the move: the developer
scripts are `tools/dev/bench_md.py` and `tools/dev/test_gate.py`.

| Check | Result |
|---|---|
| CSVR's noise, 20,000 draws for 2, 3, 4, 405 and 15,000 degrees of freedom | Kolmogorov-Smirnov p = 0.80 against the normal; 0.80, 0.79, 0.90, 0.84, 0.84 against chi-square |
| CSVR against ASE 3.27's `Bussi` driven by `lm.calc`, the engine's noise fed to ASE, 200 steps, 135 atoms | positions within 2.5e-7 to 2.8e-7 A; transferred energy 2.955326 eV in both |
| CSVR, 20,000 steps of 1 fs at 300 K, tau 10 fs, 135 atoms, three runs | mean 300.5, 301.0, 300.5 K; Kolmogorov-Smirnov p = 0.886, 0.605, 0.639 against the canonical kinetic-energy distribution (190 samples 100 fs apart); conserved quantity drift 0.0000, 0.0000, +0.0003 meV/atom/ps, range 0.03 meV/atom while the thermostat moved 24 to 26 meV/atom |
| CSVR target ramp 100 to 500 K in 2 ps, tau 20 fs | 177 K at 0.5 ps, 284 K at 1 ps, 520 K at the end; held at 500 K: mean 527 K over 1 ps |
| CSVR against Langevin, 5,000 atoms at 300 K, through `lm.Simulation` (a b b a) | 7.988, 8.006 against 7.962, 7.986 ms/step: 1.0029 |
| `loupmd md NbCl5_R2SCAN.cif --repeat 3 3 3 --thermostat csvr --tau 100 --steps 2000` | 1.926 ms/step with output (1.818 in the engine), 96 rebuilds, 0 dangerous |
| gate, CSVR build and move build | PASS, worst force 1.57 / stress 1.55; PASS, 1.52 / 1.72 |
| `bench_md.py`, CSVR build against the build of §6.4, both orders (`NbCl5` / `SrTiO3`, new / old) | 1.0015 / 1.0137 and 0.9910 / 0.9990 |
| the same, move build against the build of §6.4 | 1.0051 / 1.0005 and 0.9987 / 0.9997; mean of these two 1.0019 / 1.0001 |
| `pytest tests/test_[a-e]*.py` on the move build | 240 passed |
| water 216 (MACE-OFF23 small), Langevin 300 K, 1 fs, skin 0.5 A | 0 of 800 steps in the pipelined loop, 0.676 ms/step |
| the same with `skin="auto"`: estimated 0.8 A, largest step measured 0.079 A, skin 0.8 A | 799 of 800 steps pipelined, 0.556 ms/step (0.82 of the fixed skin) |

- The four CSVR-or-later A/Bs against the build of §6.4 average 0.9991 (`NbCl5`) and 1.0032 (`SrTiO3`); the
  1.0137 comes from one job whose last `SrTiO3` run of the old build read 7.934 ms. The Langevin step of the
  benchmark is the same captured graph in all three builds.
- CSVR against NVE from the same start read 1.0137: NVE loses half its kinetic energy to the lattice and then
  rebuilds its neighbour list less often, so it is not the comparison for the thermostat's cost.
- Not run: the full suite on these builds. The run of the CSVR build ended with the crash of 12:32 (bug check
  0xD1 in `nvlddmkm.sys` while two sessions used the GPU), and the WSL distribution was shut down repeatedly from
  13:12 to 13:31 (PLAN.md D40). Anything on the RTX 4000 Ada: the workstations are down (O12).

### 6.6 Nose-Hoover chain, Berendsen NVT, LBFGS on the device and the generic path as the default (15 November 2026, 13:30 to 17:40)

RTX 5060 Ti only. Engine build of `763aaa5` (library sha256 `53abaad3e9aca96c...`); the generic default is
`5d7f363`. Where a row names `7b31011e...`, it was measured on the build before the last change to
`loup_minimize`, which differs in that function only.

| Check | Result |
|---|---|
| Nose-Hoover chain against ASE 3.27's `NoseHooverChainNVT` driven by `lm.calc`, 200 steps, 135 atoms, tdamp 50 fs | positions within 2.1e-7 A (3 thermostats x 1 loop), 2.2e-7 A (1 x 1), 2.5e-7 A (5 x 2); the chain's energy -3.919513 / -5.646661 / -3.204603 eV against ASE's -3.919514 / -5.646663 / -3.204602 |
| Nose-Hoover chain, 20,000 steps of 1 fs, tdamp 100 fs, 135 atoms, three runs | mean 299.6, 300.3, 298.0 K; conserved quantity drift +0.0002 meV/atom/ps, range 0.026 meV/atom while potential plus kinetic moved 29 to 36 meV/atom |
| Nose-Hoover target ramp 100 to 500 K in 2 ps, tdamp 25 fs, thermostat masses of 300 K | 196 K at 0.5 ps, 238 K at 1 ps, 494 K at the end; held at 500 K: mean 497 K |
| Nose-Hoover chain against Langevin, 5,000 atoms at 300 K, `lm.Simulation`, a b b a twice, 400 steps each | 8.157 to 8.223 against 8.119 to 8.159 ms/step, ratio of the medians 1.0067 (single a b b a runs before: 1.0077 and 1.0218) |
| Berendsen against ASE's `NVTBerendsen`, 200 steps, 135 atoms | positions within 2.0e-7 A (tau 50 fs from 300 K) and 2.1e-7 A (tau 2 fs from 900 K, the scale at its bound of 0.9) |
| Berendsen from 100 K to a 300 K target, tau 20 fs; then a ramp to 500 K in 2 ps | 285 K after 0.1 ps, 297 K after 1 ps; 394 K at a target of 400 K, 489 K at the end |
| LBFGS on the device against ASE's `LBFGS` with `lm.calc`, rattled `SrTiO3` 3x3x3 to 0.05 eV/A | 15 steps, ASE 15; positions within 2.5e-7 A |
| LBFGS on the device against the host LBFGS, to 0.02 eV/A, memory 100 and memory 4 | 22 steps each way, positions within 2.8e-7 and 2.2e-7 A; one step per call: 22 steps |
| the optimisers' own time at 5,000 atoms, 40 steps | LBFGS on the device 0.141 ms a step, 1.71% of the force calls (8.04 ms each); FIRE on the host 0.111 ms, 1.14% |
| the same before pinned buffers and the list check after the pass (`7b31011e...`) | LBFGS 0.825 ms a step, 9.70% (force calls 8.29 ms); on the host (§6.3) 1.4 ms, 17% |
| LBFGS with a skin of 0.2 A, rattle 0.12 A: list check before the force pass, after it, and the engine's choice | 30 steps and 5 rebuilds each; 31, 36 and 31 force passes (0, 5 and 0 repeated); positions within 3.7e-7 A |
| cell relaxation on the host (Frechet filter), strained `SrTiO3` 3x3x3 | 157 steps, ASE 157; lattice constants within 7.3e-6 A |
| gate, default (generic) and `--specialised` | PASS and PASS (exit 0 each); worst ratios in the rows the log kept: force 1.91, stress 1.10 and force 1.56, stress 1.53 |
| `bench_md.py`, this build against the build before CSVR (§6.4), both orders (`NbCl5` / `SrTiO3`, new / old), specialised kernels | 1.0035 / 0.9993 and 0.9954 / 0.9992 |
| `bench_md.py`, `LOUP_GENERIC=1` against `LOUP_GENERIC=0` in one library (`7b31011e...`), both orders | `NbCl5` 3x3x3 1.0409 and 1.0399; `SrTiO3` 10x10x10 1.0184 and 1.0144 |
| `loupmd bench --model omat0-medium`, generic path, two runs each | `NbCl5` 3x3x3 1.876, 1.886; `SrTiO3` 10x10x10 8.236, 8.250; `SrTiO3` 6x6x6 (quick) 1.830, 1.837 ms/step |
| the library of `make wheel-lib` (six architectures, 32 MB) on this card, `loupmd bench --quick` | 1.833 ms/step on `SrTiO3` 6x6x6 |
| `pytest tests/test_[a-e]*.py`; `test_f*.py test_gate_cache.py`; `test_[m-q]*.py`; `test_[r-z]*.py` on the tree of `5d7f363` less its last change | 247 passed; exit 0; exit 0; one failure (the sweep replay's energy, 1.6e-8 against a floor of 1e-8 eV/atom) |
| `pytest tests/test_sweeps.py tests/test_run_api.py` with the floor at 1e-7 | 25 passed |
| `tests/test_gate_wrapper.py` (19 minutes), on `7b31011e...` | 10 passed |
| open domain against the periodic call, today's build, two runs of the test and two of a probe | energy within 3.9e-9 (OMAT-0 small), 1.9e-8 (medium), 3.8e-8 (MH-1), 3.4e-9 (OFF23 medium), 1.8e-9 eV/atom (OMOL-0); forces within 1.8e-6 eV/A; the same periodic call with its atoms in another order: 2e-9 to 3.4e-8 eV/atom |
| the same on the two builds of the morning (`2e7e904a`, `9f89f02c`), default GEMMs and `LOUP_NO_HYBRID=1` | OMAT-0 small 2.9e-7 against 1.5e-8 and 4.6e-9; medium 1.6e-7 and 2.2e-7 against 2.6e-8 and 5.9e-9 eV/atom |

- The open domain's energy difference (PLAN.md D16) is the rounding of the tensor-core hybrid GEMMs: on the
  morning's builds it is there with the default GEMMs and gone with `LOUP_NO_HYBRID=1` on every engine. The
  domain's GEMMs have 843 rows where the periodic call has 135, and the autotune picks a kernel per shape from
  timings. Which kernels today's plan files pick was not read out; the difference is small today and can return
  with other plan files.
- The generic path costs 4.0% on `NbCl5` and 1.4 to 1.8% on `SrTiO3` against the specialised kernels in the same
  library. S63 (14 November) had 2.9% and 7.4%; the per-system conv variants (S64) came between.
- The graph-capture tests failed by chance on the generic path: the conv variants are timed at an engine's first
  MD rebuild and a faster one drops the graphs. On 135 atoms the three are near ties, so one of two engines doing
  the same steps captured three graphs more in 4 of 8 repeats. The tests now switch the timing off
  (`LOUP_CONV_TUNE=0`). The behaviour itself is unchanged: such a run captures its graphs twice instead of once.
- Not run: the suite as one run on the last tree (its parts ran on the tree before the sweep floor changed, and the
  two changed files after); `tests/test_gate_wrapper.py` on the last build; the family gates (`--set organic`,
  `--set crystals`); anything on the RTX 4000 Ada or an A100 (O12).

### 6.7 The variable cell and Berendsen NPT (15 November 2026, 17:20 to 18:45)

RTX 5060 Ti only. Engine builds: `1fe8f24` (the variable cell; library sha256 `fbbaee33a44c9267...`), `fd731a9`
(Berendsen NPT; `0e89b6d050da93ff...`) and `7df6798` (`loup_set_reference_volume`; `c640e85347aeb330...`).

| Check | Result |
|---|---|
| NVE through the variable-cell step with a constant cell, 135 atoms: forces after one step against the fixed-cell step, two runs | within 7.5e-7 and 1.6e-6 eV/A; two fixed-cell runs differ by 1.2e-6 |
| the same after 40 steps | positions within 4.3e-8 A, forces within 4.7e-6 eV/A; two fixed-cell runs: 4.9e-8 A, 4.5e-6 eV/A |
| a row's pressure against the stress of an evaluation at the same state plus the ideal-gas part | 6.3236 GPa both |
| cell and positions scaled by 1 - 2e-4 per step for 150 steps (0.9704 in length), then a new engine at the final cell | forces within 2.1e-6 eV/A (1.4e-6 in a second run), energy within 3.7e-8 eV/atom, 6 rebuilds, pressure 26.755 GPa both |
| the same with 1 + 2e-4 per step (1.0305) | forces within 1.8e-6 eV/A, energy within 3.9e-8 eV/atom, 6 rebuilds, pressure -6.329 GPa both |
| Langevin, 5,000 atoms, through the variable-cell step (constant cell) against the pipelined fixed-cell step, a b b a twice | 8.377 to 8.416 against 8.142 to 8.202 ms/step: 1.0296; a second job 8.511 to 8.562 against 8.329 to 8.361: 1.0263 |
| Berendsen NPT, `SrTiO3` 3x3x3 compressed by 1% (11.65 GPa at step 0), target 0 GPa, taup 100 fs, 200 steps | volume 1560.03 -> 1655.083 A^3; pressure 7.05 GPa at step 50, 1.78 GPa at step 200 |
| against ASE 3.27's `NPTBerendsen` taken with one force call per step, three jobs | volume within 3.4e-10, 2.2e-8 and 1.3e-9 relative; positions within 1.8e-7 to 3.0e-7 A |
| against ASE's own class (two force calls per step) | volume within 1.1e-5 relative; 3.2e-6 from a start 0.5 GPa off the target |
| mask (1, 1, 0) against ASE's `Inhomogeneous_NPTBerendsen` with one force call per step | cell lengths 11.86512, 11.86363, 11.59785 A from 11.59785; within 1.0e-9 relative; z unchanged bit for bit |
| to 0 GPa with taup 50 fs, 800 steps | 9.21 GPa at step 10; mean of the last 200 steps -0.022 GPa (scatter 0.14); volume 1678.07 A^3 |
| eight calls of 100 steps against one of 800 | volume within 2.2e-8 relative |
| `lm.Simulation` with `lm.integrators.Berendsen(pressure_GPa=0)`, 200 steps, against one engine call; the `NPTBerendsen` drop-in | volume within 5.8e-10 relative; 1.5e-9; resumed from the restart for 100 more steps: 1666.495 A^3 |
| gate on `1fe8f24`, default (generic) and `--specialised` | PASS, 18 rows, worst force 1.87, stress 1.26; PASS, 30 rows, 1.55 and 1.69 |
| gate on `7df6798` | PASS, 18 rows, 1.88 and 1.20; PASS, 30 rows, 1.55 and 1.71 |
| `bench_md.py`, `1fe8f24` against the build of §6.6, both orders, new / old: specialised kernels | `NbCl5` 1.0105 and 0.9973; `SrTiO3` 1.0028 and 0.9995 |
| the same on the generic path | `NbCl5` 0.9998 and 1.0007; `SrTiO3` 1.0015 and 1.0001 |
| `bench_md.py`, `7df6798` against the build of §6.6, generic path, both orders | `NbCl5` 1.0110 and 1.0000; `SrTiO3` 1.0016 and 0.9979 |
| suite on `1fe8f24`: `test_[a-e]*.py`; `test_f*.py test_gate_cache.py`; `test_[m-q]*.py`; `test_[r-z]*.py` | 247, 38, 150 and 115 passed, 1 skipped |
| suite on `7df6798` less its last change to `lm.md`: the first three parts | 252, 38 and 150 passed |

- The fixed-cell step is the same captured graph as before: the rebuild's two extra kernels and the device cell
  exist only on a handle that has run a variable-cell step.
- The variable-cell step's extra 2.6 to 3.0% is the sum of three things that were not measured apart: stress in
  every force pass, the Cartesian shifts formed per step (24 bytes written per edge), and one host wait per
  step in place of the pipelined loop.
- Berendsen NPT's distance from ASE's own class is PLAN.md D48.
- `test_[r-z]*.py` on `7df6798` ran after this was first written: 120 passed, 1 skipped.
- Not run: `tests/test_gate_wrapper.py` on these builds; the barostat on other systems than `SrTiO3` 3x3x3; its step time;
  anything on the RTX 4000 Ada or an A100 (O12).

### 6.8 Langevin BAOAB NPT, MTK NPT, and what two builds differ by in the benchmark (15 November 2026, 18:45 to 20:25)

RTX 5060 Ti only. Engine builds: `e689cf7` (BAOAB; library sha256 `342ca4cb0ca213fc...`) and `a8a3e06` (MTK;
`433ffe6558b55fa0...`, the build of every later commit of this section). With these every integrator kind of
`include/loupmd.h` is built.

| Check | Result |
|---|---|
| BAOAB NPT at a target of 0 K (nothing is drawn), hydrostatic, 100 steps from `SrTiO3` 3x3x3 compressed by 1%, against ASE 3.27's `LangevinBAOAB` driven by `lm.calc` | volume 1560.03 -> 1599.148 A^3; cell within 2.4e-9 of its length (3.7e-10 in the first run), positions within 9.6e-8 A, velocities within 7.1e-8 |
| the same with the full cell, against ASE's class with the identity in its drag | volume 1726.976 A^3; cell within 7.2e-10 of its length, positions within 1.1e-7 A |
| BAOAB NPT, `SrTiO3` 5x5x5 (625 atoms), 300 K, 1 GPa, T_tau 50 fs, P_tau 500 fs, 20,000 steps after 5,000, rows every 10 steps, blocks of 1,000 steps | pressure 1.000 +- 0.001 GPa, temperature 299.3 +- 0.4 K, volume 12.3305 A^3/atom, relative scatter 2.0e-3 |
| BAOAB NPT, 200 steps in one call, in two calls, from a state block imported into another engine after 100 | volume within 3.7e-9 and 3.0e-9 relative |
| `lm.md(ensemble="NPT")` at 300 K and 1 GPa from the compressed `SrTiO3` 3x3x3, then resumed from the restart | volume 1560.03 -> 1639.64 A^3 after 400 steps, 1707.65 after 600; mean pressure of steps 300 to 600 1.57 GPa from 11.65 at step 0; mean temperature 244 K (still expanding) |
| MTK NPT, isotropic, chains 3x1 and 3x1, 200 steps, against ASE's `IsotropicMTKNPT` | volume 1560.03 -> 1614.760 A^3, ASE the same to 3.5e-10 relative; cell within 1.4e-9 A, positions within 3.0e-7 A; the conserved quantity beside E_pot + E_kin -0.192892 eV in both |
| MTK NPT, full cell, against ASE's `MTKNPT` | volume 1614.554 A^3, relative 3.0e-10; cell within 5.4e-8 A, positions within 2.1e-7 A; -0.208253 eV in both |
| MTK NPT, isotropic, chains 2x2 and 4x3 | volume relative 7.6e-9; positions within 2.2e-7 A; 0.377626 against 0.377625 eV |
| MTK NPT, 20,000 steps of 1 fs at 300 K and 1 GPa, 135 atoms: conserved quantity | isotropic: drift +0.0002 meV/atom/ps, range 0.022 meV/atom; full cell: -0.0002 and 0.026; potential plus kinetic moved 27 to 28 meV/atom |
| the same runs, second half | isotropic 297.8 K, 0.980 GPa, 12.3395 A^3/atom; full cell 299.5 K, 1.004 GPa, 12.3433 A^3/atom |
| MTK NPT in one call, two calls, from an imported state block | volume within 7.3e-11 and 6.4e-9 relative |
| `lm.integrators.MTK` through `lm.Simulation`, and the `IsotropicMTKNPT` drop-in, against one engine call | volume within 6.0e-9 and 1.9e-9 relative |
| gate on `e689cf7`, default (generic) and `--specialised` | PASS, 18 rows, worst force 1.82, stress 1.41; PASS, 30 rows, 1.50 and 1.83 |
| gate on `a8a3e06` | PASS, 18 rows, 1.90 and 1.21; PASS, 30 rows, 1.57 and 1.71 |
| suite on `e689cf7` with the BAOAB front end: `test_[a-e]*.py`; `test_f*.py test_gate_cache.py`; `test_[m-q]*.py`; `test_[r-z]*.py` | 257, 38, 152 and 123 passed, 1 skipped |
| suite on `a8a3e06` with the MTK front end | 258, 38, 160 and 123 passed, 1 skipped |
| `bench_md.py`, generic path, `e689cf7` against the build of §6.6, each with its own stored plans, three pairs (new / old; old first, new first) | `NbCl5` 1.0127 and 1.0026, 1.0109 and 1.0032 (the first of these while CPU tests ran), 1.0099 and 1.0025; `SrTiO3` 0.9994 and 1.0009, 1.0093 and 1.0007, 1.0027 and 1.0003 |
| the same, `a8a3e06` against §6.6's build, old new new old twice | `NbCl5` 1.855, 1.852, 1.855, 1.852 against 1.832, 1.850, 1.841, 1.844 ms/step (1.0064); `SrTiO3` 8.049 against 8.041 (1.0010) |
| `e689cf7` against §6.6's build with a fresh plan directory per run, three runs of two each | `NbCl5` 1.853 to 1.864 (mean 1.855) against 1.828 to 1.860 (mean 1.849); `SrTiO3` 8.054 against 8.043 |
| four builds (§6.6's, `1fe8f24`, `7df6798`, `a8a3e06`), each with its own script, a fresh plan directory per run, four rounds | `NbCl5` means 1.853, 1.857, 1.861, 1.858 ms/step (single runs 1.839 to 1.864); `SrTiO3` 8.042, 8.055, 8.052, 8.047 (8.024 to 8.060); the last build against the first 1.0027 and 1.0006, and 1.001 and 1.000 without the job's first run (1.839, 8.024) |

- The fixed-cell benchmark has not changed: with fresh plans the four builds agree within 0.4% on `NbCl5` and 0.2% on `SrTiO3`, which is less than one build's own run-to-run spread (0.6 to 1.3%) and not in the order of the commits. The pairs with stored plans read up to 1.3% apart because each library keeps the GEMM plans it timed once, and because the first run of a job is about 1% faster than the later ones: it was the old build in every old-first pair and in round 1 of the four-build job.
- BAOAB's and MTK's own step times were not measured; both run in the variable-cell step of §6.7 (1.026 to 1.030
  of the pipelined fixed-cell step) with one single-thread kernel and one or two kinetic sums more.
- Not run: `tests/test_gate_wrapper.py` on these builds; the family gates (`--set organic`, `--set crystals`);
  the NPT integrators on any other system than `SrTiO3`; anything on the RTX 4000 Ada or an A100 (O12).

### 6.9 The family gates on the last integrator build, and kernel plugins (15 November 2026, 20:15 to 21:10)

RTX 5060 Ti only. Gates: `tools/dev/test_gate.py MODEL [--set ...]` on the build of `a8a3e06` (library sha256
`433ffe6558b55fa0...`), every model on the generic path, references from the cache except OMOL-0 with charge 0
and multiplicity 1 (computed on the CPU in this run). Each row: gate rows, failures, the worst force and stress
ratios (a quantity passes up to 3).

| Model, set | Result |
|---|---|
| MACE-OFF23 small, medium, large; OFF23b medium; OFF24 medium (`--set organic`; large and the last two with `--ref-device cpu`) | 8, 24, 8, 24, 24 rows, 0 failed; force 0.62, 0.33, 0.43, 0.94, 0.69; stress 1.88, 0.62, 0.91, 0.53, 1.03 |
| MACE-MH-1; MACE-POLAR-1 S, M, L (`--set crystals`; L with `--ref-device cpu`) | 18, 5, 18, 6 rows, 0 failed; force 0.90, 0.70, 2.10, 2.20; stress 0.31, 0.59, 0.73, 0.83 |
| MACE-OMAT-0 small; MP-0b3 medium; MPA-0 medium; MP-0 large (`MACE_MPtrj_2022.9`); MP-0 medium (default set) | 6, 18, 18, 6, 18 rows, 0 failed; force 1.34, 1.15, 2.02, 1.30, 0.63; stress 1.33, 0.83, 2.19, 0.99, 0.38 |
| MATPES PBE and r2SCAN (`--set crystals`) | 18 and 18 rows, 0 failed; force 1.02 and 1.41; stress 1.02 and 0.62 |
| MACE-OMOL-0 extra large (`--set organic --charge 0 --spin-multiplicity 1 --ref-device cpu`) | 8 rows, 0 failed; force 0.83, stress 0.73 |
| MACE-OMAT-0 medium, default set, generic and `--specialised`, on the build of `8b858e0` (`9d39c8e75f8571b1...`) | 18 and 30 rows, 0 failed |

Kernel plugins (`8b858e0`, PLAN.md S2-T06 and D57):

| Check | Result |
|---|---|
| `kernels.build_plugins` on `mace-omat-0-medium.model`, forced (nvcc 13, `-arch=sm_120`, the shared CUDA runtime) | two plugins, 76,368 and 157,720 bytes |
| `SrTiO3` 3x3x3 with the library's own kernels of one signature hidden, so the plugin's run; three jobs | energy within 1.1e-8 eV/atom, forces within 9.2e-7 to 1.2e-6 eV/A, stress within 5.1e-9 eV/A^3 of the library's own kernels |
| the same with nvcc's default static runtime in the plugin | `loup_evaluate: invalid resource handle`: kernels registered with a second copy of the runtime |
| a hidden signature and no plugin | `no compiled product kernel for signature ...` naming the folder searched and `loupmd build-kernels` |
| `loupmd build-kernels omat0-small` | "nothing to build: every kernel of mace-omat-0-small.model is there" |
| `codegen/gen_generic.py` after the change, against the tracked `generated_generic.cu`, `.cuh` and manifest | byte-identical |

- Not run: `loupmd validate --set organic | crystals` (the same library code as `test_gate.py`, through the
  command); the gate wrapper tests on these builds; a model whose kernels the library really lacks; a plugin built
  for another architecture than `sm_120`; anything on the RTX 4000 Ada or an A100 (O12).

### 6.10 The re-homed tests, NPT profiles and the pinned CUDA runtime (15 November 2026, 21:00 to 21:25)

RTX 5060 Ti only; the library is the build of `8b858e0` (sha256 `9d39c8e75f8571b1...`), unchanged since §6.9.

| Check | Result |
|---|---|
| `pytest -q tests/cpu` in the development environment, GPU hidden (after `fba7376`) | 327 passed, 1 skipped, 56 s |
| `pytest -q tests/cpu` in a venv with numpy, scipy, ase and pytest only (no torch, mace, e3nn, cuequivariance), no `nvidia-smi` on the PATH, no library in the tree, `LOUPMD_MODELS` unset (at `a80f166`) | 319 passed, 7 skipped, 23 s |
| `pytest tests/gpu -m "not acceptance" --durations=0` (at `a80f166`, before the full tier was marked) | 262 passed, 10 deselected, 612 s |
| the tiers by collection (at `7875527`) | 242 fast, 20 full, 10 acceptance |
| NPT profiles (`fba7376`, O23) on the fake engine: `loupmd md --profile` with a profile at 0.5 GPa | Langevin BAOAB at 0.5 GPa; `--friction` and `--pressure` keep or replace the profile's pressure |

The pinned CUDA runtime (PLAN.md D58). `tools/dev/bench_md.py`, generic path, a fresh plan directory per run, four
rounds alternating; "system" is `/usr/local/cuda-13.3` (cudart 13.3.29, cuBLAS 13.5.1.27, what every figure of
this file so far ran on), "wheels" the NVIDIA wheels the package pins (cuda-toolkit 13.0.3: cudart 13.0.96, cuBLAS
13.1.1.3), put first with `LD_LIBRARY_PATH`; the loaded libraries were read from `/proc/self/maps` on each side.

| Runtime | `NbCl5` 3x3x3, ms/step (mean; range) | `SrTiO3` 10x10x10, ms/step (mean; range) |
|---|---|---|
| system 13.3 | 1.862; 1.847 to 1.868 | 8.042; 8.013 to 8.061 |
| wheels 13.0.3 | 1.864; 1.859 to 1.866 | 8.047; 8.043 to 8.050 |
| wheels / system | 1.0012 | 1.0007 |

- A tie: the difference is inside the spread of either side.
- All 69 CUDA symbols `libloupmd.so` imports (`nm -D`) are defined in the wheels' `libcudart.so.13`,
  `libcublas.so.13` and `libcublasLt.so.13`. The local build needs GLIBC 2.38 and GLIBCXX 3.4.32, so the wheel's
  library has to come from the container build.
- Not run: the fast tier on its own (about 320 s of the 612 s run, by subtraction); the acceptance tier after the
  move; an NPT profile on the GPU; the wheel build (its scripts are written, the image is not built).

### 6.11 The tile planner, the fast tier alone and the version 0.1.0a1 (15 November 2026, 22:00 to 22:30)

RTX 5060 Ti only. The library is the build of the working tree with the planner and the version `0.1.0a1` (sha256
`929a8f5a57d5a5f4...`); the planner is host code that nothing in the step calls yet.

| Check | Result |
|---|---|
| `pytest tests/cpu/test_tile_plan.py` (g++ build of `src/tile_plan.cu`) | 14 passed; 1,026,895 atoms planned in 72 ms (criterion 0.5 s) |
| `pytest tests/gpu` fast tier alone (default selection) | 243 passed, 30 deselected, 284 s |
| `pytest tests/gpu -m full` | 20 passed, 253 deselected, 239 s |
| `tools/dev/test_gate.py`, default and `--specialised` | default: 18 rows PASS, 0 FAIL; `--specialised`: 32 PASS, 0 FAIL |

A/B against the build before (`9d39c8e75f8571b1...`, §6.10), `tools/dev/bench_md.py`, generic path, a fresh plan
directory per run, four rounds alternating:

| Build | `NbCl5` 3x3x3, ms/step (mean; range) | `SrTiO3` 10x10x10, ms/step (mean; range) |
|---|---|---|
| before (`9d39c8e7`) | 1.859; 1.850 to 1.869 | 8.037; 8.006 to 8.061 |
| with the planner (`929a8f5a`) | 1.862; 1.858 to 1.868 | 8.046; 8.041 to 8.058 |
| ratio | 1.0019 | 1.0012 |

- A tie: each difference is inside the other side's range, as expected for code the step does not run.
- Not run: the wheel (Docker Desktop does not start, PLAN.md D59), the acceptance tier, anything on another GPU.

### 6.12 Per-tile neighbour lists (15 November 2026, 22:35)

RTX 5060 Ti only; library of the working tree after `40bdf96` with `loup_debug_domain_edges` (sha256
`d289193f6b995db7...`). `pytest -s tests/gpu/test_tile_lists.py`: `SrTiO3` 20x20x20 (40,000 atoms; rattled by
0.05 A, sheared by up to 1%), MACE-OMAT-0 medium on the generic path, skin 0.5 A, planner reach = cutoff + skin +
0.01 A.

| Receiver cap | Tiles (grid) | Edges of the whole system | Edges in one list only | Ghost candidates | Sending along no edge |
|---|---|---|---|---|---|
| 12,000 | 4 (1x1x1) | 3,969,958 | 0 | 89,790 | 38,879 (43.3%) |
| 3,000 | 16 (2x2x2) | 3,969,958 | 0 | 103,230 | 28,654 (27.8%) |

- The candidates without an edge are those of the corners of a tile's bounding box (S5-T01 lists every image
  within reach of the box, not of the atoms): a tile that keeps them carries up to 43% more ghost rows than it needs.
- Not measured: the time of a tile's list build, and of a second build without the dropped candidates.
- `pytest tests/gpu` with no marker filter (fast, full and acceptance tiers, with the two tests above) on this
  build: 275 passed in 1,458 s. The acceptance tier after the move of S2-T11 thus passes as well.

### 6.13 The tiled call (15 November 2026, 23:30)

RTX 5060 Ti only; library of the working tree after `9abd36d` with `loup_debug_evaluate_tiled` (sha256
`d7ad0d9be7f8ce99...`). `pytest -s tests/gpu/test_tiled.py`, skin 0.5 A; each line compares the tiled call with
the whole-system call of the same build (energy per atom, largest force and stress component).

| Model | Atoms | Tiles | Default GEMMs: dE eV/atom, dF eV/A, dS eV/A^3 | No hybrids on either side | Whole system, default against no hybrids |
|---|---|---|---|---|---|
| OMAT-0 medium (generic) | 40,000 `SrTiO3` | 8 | 6.8e-7, 5.8e-6, 2.8e-8 | 1.2e-9, 1.8e-6, 2.1e-10 | 1.3e-8, 1.3e-6, 3.2e-9 |
| OMAT-0 small | 40,000 `SrTiO3` | 8 | 4.5e-7, 6.7e-6, 4.7e-8 | 1.2e-9, 1.7e-6, 1.6e-10 | 1.7e-10, 1.1e-6, 2.8e-10 |
| MH-1 | 24,565 `SrTiO3` | 8 | 4.4e-8, 5.5e-6, 4.3e-9 | 2.5e-10, 1.7e-6, 5.6e-10 | 4.4e-8, 1.6e-6, 1.6e-9 |
| OFF23 medium | 41,472 water | 8 | 4.3e-8, 2.9e-6, 1.5e-9 | 2.6e-10, 1.2e-6, 8.3e-11 | 4.0e-9, 1.6e-6, 6.4e-10 |
| OMOL-0 extra large (five sweeps) | 1,296 water | 4 | 5.9e-9, 1.2e-6, 1.4e-9 | 8.1e-10, 1.5e-6, 6.1e-10 | 1.9e-9, 1.2e-6, 1.1e-9 |

- Without hybrids the tiled call is the whole-system call to float32 rounding; with the default GEMMs the
  difference is the hybrid kernels' rounding where a tile's GEMM shape gets another kernel (PLAN.md D16, D60).
- MH-1's whole-system call runs out of memory at 40,000 atoms on 16 GB (24,565 fit): the case for tiling.
- No device memory is left behind: three tiled and two whole-system calls in one process leave 1,129 MiB in use
  before and after (`cudaMemGetInfo`).
- `pytest tests/gpu -m "not acceptance"`: 270 passed, 1 failed in 653 s. The failure, `test_the_calculator_equals_evaluate` (two engines of one bundle 3.0e-7 eV/atom apart against its bound of 2e-7), passed five runs in a row on this build and three on the build before; its bound is now 1e-6 eV/atom (PLAN.md D61).
- A/B against the build of §6.12 (the tiled code is not on the step path): four rounds alternating, a fresh plan directory per run: `NbCl5` 1.898 ms/step before and 1.894 after, `SrTiO3` 8.230 and 8.233 (ratios 0.998 and 1.0004), a tie. Both sides ran about 2% slower than the runs of §6.11 earlier that evening; only the alternating comparison counts.
- Not measured: the tiled call's time (every tile sweep sets its domain up again; S5-T06, S5-T10).
