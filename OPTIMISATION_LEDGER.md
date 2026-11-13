# LoupMD Optimisation Ledger

Status: compiled on 13 November 2026 from the LoupMD working tree and
[STATUS_AND_PROFILE.md][status] at its sixth update (13 November, 14:15). Modelled on FastMACE's
`docs/optimisation-ledger.md`. The tree is under git since 14:20 that day (first commit `4477d66`); entries before
that have no commit to cite.

This file is the index and the verdict list for every optimisation LoupMD has tried, adopted or not, so that
work is not repeated. It does not replace the evidence: `STATUS_AND_PROFILE.md` stays authoritative for numbers,
profiles and methods, and each entry links the section it comes from. Numbers are copied as recorded; "not
recorded" means no source gives one. Before 12 November the sources give no dates, only the stage of the
[chronology](#chronology) at which an item landed (S0-S5).

Maintenance rule: every new attempt adds its index row and its appendix entry together with the
`STATUS_AND_PROFILE.md` update that records its numbers. Rows are never deleted: a reverted or superseded attempt
changes its status.

Conventions:

- **MD A/B**: `python/bench_md.py` (median of 6 x 256 Langevin steps, 300 K, 1 fs, skin 0.5 A, friction 0.01/fs,
  after 100 warm-up steps), old and new library in the same process, run in both build orders: "A / B" are the two
  orders. Percentages are the change in MD step time; **negative is faster**. A change is adopted only if it wins or
  ties in both orders on both GPUs. On the Ada the first run of a sequence is systematically ~0.6% fast, which biases
  order A against the new build.
- **Kernel times** are nsys kernel sums per step (us) unless stated; "span" is wall-clock GPU time of a stage.
- **GPUs**: 5060 Ti = RTX 5060 Ti 16 GB (`sm_120`, WSL2, the desktop shares the GPU); Ada = RTX 4000 Ada 20 GB
  (`sm_89`, `leeloo`, no performance-counter permission, so every `ncu` number is from the 5060 Ti).
- **Systems**: `NbCl5` 3x3x3 (1,944 atoms, 2 species), `SrTiO3` 10x10x10 (5,000 atoms, 3 species, ~370k edges).
  Two values "x / y" without other labels are `NbCl5` / `SrTiO3`.
- **Path**: spec = the specialised MACE-OMAT-0 medium kernels (sections 1-4 of the status file); gen = the generic
  path for the other families (section 5); POLAR = MACE-POLAR-1's long-range part (`src/polar.cu`).
- **Status**: DEFAULT (in the build and on); DEFAULT (scoped: ...); REJECTED (built and measured, not adopted);
  SUPERSEDED; NOT BUILT (dropped on paper or deferred); PENDING (owner decision); BLOCKED; MEASUREMENT (profile or
  check, no switch); FIX (a correctness defect found while optimising).
- **Accuracy gate**: `python/test_gate.py`, 3x MACE float32-vs-float64 rule (dE <= max(3 dE32, 0.01 meV/atom),
  dF <= max(3 dF32, 1e-5 eV/A), dS <= max(3 dS32, 1e-7 eV/A^3)). Every adopted item passes it on both GPUs.

## Index by area

### Interaction-1 radial spline forward (`spline_forward_inter1`)

| ID | Idea | Switch/flag | Path | Status | Stage | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [spl-paired-edges](#spl-paired-edges) | Decode a table slab once for two consecutive edges of a receiver | template `PAIR` | spec | DEFAULT (per-system choice) | S3 | table loads per edge 1 -> 0.805 (`SrTiO3`); spline -7% (5060 Ti), -3% (Ada) | [§3.1 #20][s31] |
| [spl-multi-receiver](#spl-multi-receiver) | R consecutive receivers per block, variant timed at the first rebuild | `spline_forward_inter1<R, PAIR>`, `LOUP_SPLINE_VARIANT` | spec | DEFAULT | S4-S5 | L1 hit 5% -> 36%, L2 3.92 -> 2.66 GB; Ada `SrTiO3` -3.1% / -3.6% | [§3.1 #29][s31] |
| [spl-launch-128-7](#spl-launch-128-7) | `(128, 7)` / min-blocks 5 on `sm_89` | launch bounds | spec | REJECTED | – | spills; Ada +17% / +3% | [§3.3][s33] |
| [spl-16-edge-tiles](#spl-16-edge-tiles) | 16-edge tiles | – | spec | REJECTED | – | tie | [§3.3][s33] |
| [spl-receiver-remap](#spl-receiver-remap) | Blocks strided over the receiver range for L1 locality | – | spec | REJECTED | – | tie | [§3.3][s33] |
| [spl-paired-r5](#spl-paired-r5) | Paired 5-receiver blocks; lockstep R = 2 / 4 | – | spec | REJECTED | S4-S5 | spills 48-72 B; lockstep slower than named barriers | [§3.3][s33] |
| [spl-edge-split](#spl-edge-split) | One receiver's edges over 2 / 4 thread groups (small systems) | – | spec | REJECTED | S5 | 135 atoms 98 / 82 us vs 64 us; 1.4-1.6x slower at 5,000 atoms | [§4.1][s41] |
| [spl-unroll2](#spl-unroll2) | 2-edge unroll (before pairing) | – | spec | REJECTED | – | not recorded | [§3.3][s33] |
| [spl-cubic-forward](#spl-cubic-forward) | The 8-byte cubic table in the forward too | – | spec | REJECTED | – | not recorded | [§3.3][s33] |
| [tab-far-edge-4byte](#tab-far-edge-4byte) | 4-byte table rows for far edges (T2) | – | spec | REJECTED | – | `w'` +0.9% / +1.9%, spline +1.4% / +3.4% (5060 Ti / Ada) | [§3.3][s33] |

### Interaction-1 chain as a whole

| ID | Idea | Switch/flag | Path | Status | Stage | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [int1-two-halves](#int1-two-halves) | Chain split into two receiver halves on two streams; high-priority GEMM streams | – | spec | REJECTED | – | 5060 Ti +0.9% / +0.7% | [§3.3][s33] |
| [int1-l2-chunks](#int1-l2-chunks) | Whole chain per L2-sized receiver chunk so `m1` / `dm1` stay in L2 | – | spec | REJECTED | – | +0.2% to +2.8% at 650-1,700-atom chunks | [§3.3][s33] |
| [int1-lin1-fusion](#int1-lin1-fusion) | Fuse `lin1` into the spline / `w'` kernels | – | spec | NOT BUILT | – | ceiling: `lin1` groups 529 + 597 us (`SrTiO3`) | [§4.1 #1][s41] |
| [int1-radial-low-rank](#int1-radial-low-rank) | Tabulate fewer radial functions: the 64 hidden activations, or a few basis functions per channel | – | spec | NOT BUILT (closed by measurement) | – | rank 64 of 64; per channel 10 of 10 paths | this file |

### Interaction-1 backward (`w'`, `wprime_backward_inter1`)

| ID | Idea | Switch/flag | Path | Status | Stage | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [wp-cubic-packed8](#wp-cubic-packed8) | 8-byte exact quantised cubic table instead of `float4` | `d_cubic_packed8_1` | spec | DEFAULT | S1 | part of S0 -> S1: 2.384 -> 2.187 / 11.380 -> 10.458 ms (5060 Ti) | [§3.1 #1][s31], [§1][s1] |
| [wp-receiver-owned](#wp-receiver-owned) | One receiver per block, q staged once (N >= 512) | `wprime_backward_inter1_rcv`, `LOUP_WPRIME_RCV` | spec | DEFAULT | S3 | -170 us on both GPUs | [§3.1 #12][s31] |
| [wp-channel-major-q](#wp-channel-major-q) | Channel-major staged q, conflict-free `LDS.128` | – | spec | DEFAULT | S3 | `w'` -1.7% (5060 Ti), -2.5% (Ada) | [§3.1 #21][s31] |
| [wp-shift-free-decode](#wp-shift-free-decode) | Coefficient bits packed at their float position: one `LOP3` per decode | – | spec | DEFAULT | S4 | `w'` 3,177 -> 3,065 us; MD up to -1.3% | [§3.1 #27][s31] |
| [wp-dt-scale](#wp-dt-scale) | Radial derivative scaled by 2^8 (one `FMUL` less per path) | – | spec | DEFAULT | S4 | `w'` 3,069 -> 3,010 us; MD -0.1% to -0.6% | [§3.1 #28][s31] |
| [wp-block-size](#wp-block-size) | 256 or 512 threads per block, chosen per system | `LOUP_WPRIME_THREADS` | spec | DEFAULT | S5 | Ada `SrTiO3` -1.0% / -1.2%; 5060 Ti tie | [§3.1 #31][s31] |
| [wp-vector2](#wp-vector2) | 2-channel vectorised `w'` (T1) | – | spec | REJECTED | – | -1.6% (5060 Ti), +2.3% (Ada) | [§3.3][s33] |
| [wp-paired-edges](#wp-paired-edges) | Paired edges as in the spline | – | spec | REJECTED | – | 3.23 -> 3.28 / 3.35 ms (spills) | [§3.3][s33] |
| [wp-channel-passes](#wp-channel-passes) | 2 / 4-pass channel staging (smaller `s_dout`) | – | spec | REJECTED | – | 3.23 -> 3.40 / 3.80 ms | [§3.3][s33] |
| [wp-x0-hoist](#wp-x0-hoist) | Hoist the `x0` scale of the four scalar paths | – | spec | REJECTED | – | 3,016 -> 3,086 us (spills) | [§3.3][s33] |
| [wp-launch-256-3](#wp-launch-256-3) | `(256, 3)` launch bounds, 80 registers | – | spec | REJECTED | – | below A/B resolution | [§3.3][s33] |
| [wp-persistent](#wp-persistent) | SM-local persistent blocks with `%smid` cursors | – | spec | REJECTED | – | 3,014 -> 3,240 us | [§3.3][s33] |
| [wp-no-allocate](#wp-no-allocate) | Table loads with `L1::no_allocate` | – | spec | REJECTED | – | 3,022 -> 3,120 us | [§3.3][s33] |
| [wp-decode-bias-fold](#wp-decode-bias-fold) | Fold the four `- hb` subtractions of the cubic decode into the polynomials | `LOUP_GEN_DECODE_FOLD=1` (generator, off) | spec | BLOCKED (5060 Ti not measured); Ada: tie | – | Ada `SrTiO3` 1.0015 / 0.9955; -24 / -16 instructions; stress ratio 0.76 -> 0.94 | this file |

### Interaction 0 and the symmetric contractions

| ID | Idea | Switch/flag | Path | Status | Stage | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [i0-receiver-owned-fwd](#i0-receiver-owned-fwd) | Receiver-owned i0 forward, register accumulation per sender-species run | `check_row_species_order_kernel` guard | spec | DEFAULT | S2 | removes atomics and the 61 MB per-step zero-fill | [§3.1 #4][s31] |
| [i0-bwd-prefetch](#i0-bwd-prefetch) | Lane-parallel chunk metadata in i0 backward | – | spec | DEFAULT | S2 | not recorded alone | [§3.1 #5][s31] |
| [i0-bwd-fused-assembly](#i0-bwd-fused-assembly) | i0 backward fused with force / virial assembly (T5) | – | spec | DEFAULT | S3 | removes `d_dsh0`, `d_dr0` | [§3.1 #14][s31] |
| [i0-bwd-sh-staging](#i0-bwd-sh-staging) | SH staged in the warp's rows, `float2` channel pairs | – | spec | DEFAULT | S3 | 685 -> 564 us (5060 Ti), 736 -> 668 us (Ada) | [§3.1 #23][s31] |
| [i0-fwd-warp-per-receiver](#i0-fwd-warp-per-receiver) | Warp per receiver, 2 channels per lane | – | spec | DEFAULT | S3 | 390 -> 245 us (5060 Ti), 415 -> 304 us (Ada) | [§3.1 #25][s31] |
| [i0-sh-shuffle](#i0-sh-shuffle) | SH shuffle broadcast in i0 backward (old idea 3) | – | spec | SUPERSEDED | – | tie; superseded by i0-bwd-sh-staging | [§3.3][s33] |
| [i0-bwd-receiver-owned](#i0-bwd-receiver-owned) | Receiver-owned i0 backward (T7) | – | spec | REJECTED | – | +1.8% / +1.1% (5060 Ti) | [§3.3][s33] |
| [i0-bwd-micro](#i0-bwd-micro) | Coefficient prefetch, early loads, 4 partial sums | – | spec | REJECTED | – | 551 -> 571, 554 -> 564, 554 -> 626 us | [§3.3][s33] |
| [i0-per-species-fwd](#i0-per-species-fwd) | Per-species i0 forward | – | spec | REJECTED | – | not recorded | [§3.3][s33] |
| [sc-sc1-fused](#sc-sc1-fused) | `sc1` forward + readout forward / backward + `sc1` backward in one launch | `sc1_fused` | spec | DEFAULT | S2 | not recorded alone | [§3.1 #6][s31] |
| [sc-sc0-bwd-launch](#sc-sc0-bwd-launch) | `sc0_backward` at `(128, 3)` | – | spec | REJECTED | – | not recorded | [§3.3][s33] |
| [sc-sc0-overlap](#sc-sc0-overlap) | `sc0` per species beside the other species' `compose0` products | – | spec | REJECTED | – | +19 / +47 us on the `SrTiO3` stage gaps | [§3.3][s33] |
| [frc-segmented-f64](#frc-segmented-f64) | Receiver-warp segmented float64 force reduction | – | spec | REJECTED | – | not recorded | [§3.3][s33] |
| [frc-per-atom-energy](#frc-per-atom-energy) | Per-atom energy array instead of atomics (T6) | – | spec | REJECTED | – | +0.5% / +0.1% (5060 Ti) | [§3.3][s33] |

### GEMMs and the autotune

| ID | Idea | Switch/flag | Path | Status | Stage | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [gemm-hybrid-cubin](#gemm-hybrid-cubin) | 3xTF32 hybrid tensor-core GEMM cubins as autotune candidates | `cubins/tf32x3_hyb_sm*.cubin`, `LOUP_NO_HYBRID` | both | DEFAULT (scoped: not for long-range models since S6) | S1 | part of S0 -> S1 | [§3.1 #2][s31] |
| [gemm-autotune](#gemm-autotune) | Time up to 32 cuBLASLt heuristics + 3 hybrid tiles (+ own SGEMM) per shape | `sgemm_rm`, `LOUP_DEBUG_GEMM` | both | DEFAULT | S2 | not recorded alone | [§3.1 #11][s31] |
| [gemm-cold-l2](#gemm-cold-l2) | Operands larger than L2/2 timed with a flushed L2 | – | both | DEFAULT | S3 | Ada `lin1_fwd` group 672 -> 575 us | [§3.1 #18][s31] |
| [gemm-cold-l2-all](#gemm-cold-l2-all) | Every GEMM timed cold | – | spec | REJECTED | S3 | Ada `NbCl5` +2.9% | [§3.3][s33] |
| [gemm-no-split-k](#gemm-no-split-k) | Exclude split-K cuBLASLt algorithms | `REDUCTION_SCHEME_MASK = NONE` | both | DEFAULT | S3 | 5060 Ti idle -120 us per step; deterministic | [§3.1 #22][s31] |
| [gemm-plan-cache](#gemm-plan-cache) | Persistent per-GPU plan cache | `LOUP_CACHE_DIR`, `LOUP_RETUNE` | both | DEFAULT | S3; S6 | cold-start `evaluate` 1.3 -> 0.21 s (5060 Ti), 0.76 -> 0.05 s (Ada) | [§3.1 #16][s31], [§3.2 F4][s32] |
| [gemm-own-sgemm](#gemm-own-sgemm) | Own 128 x 128 x 16 `cp.async` SIMT SGEMM as a candidate | `loup::sgemm128` | both | DEFAULT | S5 | 5060 Ti `NbCl5` -1.6% / -2.0% (first version), -1.4% / -1.3% (`cp.async`) | [§3.1 #30][s31] |
| [gemm-own-n192](#gemm-own-n192) | 128 x 64 tiles for N = 192 | – | spec | DEFAULT (tie) | S5 | 7 products 297 -> 248 us; MD tie | [§3.1 #30][s31] |
| [gemm-sgemm-variants](#gemm-sgemm-variants) | 4 stages BK = 8; register-staged BK = 8; 1 block / SM; extra 64 x 128 / 128 x 64 tiles | – | spec | REJECTED | S5 | 5-50% slower, or `NbCl5` +0.8% to +1.8% in the pass | [§3.3][s33] |
| [gemm-pretransposed](#gemm-pretransposed) | Pre-transposed forward weights (old idea 1) | – | spec | REJECTED | – | 5060 Ti tie, Ada +1.0% / +0.3% | [§3.3][s33] |
| [gemm-grouped-species](#gemm-grouped-species) | One grouped species GEMM for `compose0_fwd` + `skip16` (T4) | `fm_tf32x3_species_rr_64x64` | spec | REJECTED | – | 5060 Ti +1.7% / +0.8% | [§3.3][s33] |
| [gemm-row-chunked](#gemm-row-chunked) | Hybrid GEMM per quarter-L2 row chunk for cold operands | – | spec | REJECTED | – | never chosen by the autotune | [§3.3][s33] |
| [gemm-group-policy](#gemm-group-policy) | Own-SGEMM policy timed per fork group in context | – | spec | REJECTED | S5 | 5060 Ti `NbCl5` -0.4% to -0.8%, `SrTiO3` +0.1% / +0.2%, Ada tie | [§3.3][s33] |
| [gemm-split-k-own](#gemm-split-k-own) | Split-K own SGEMM for small systems | – | spec | REJECTED | S5 | 135 atoms: GEMM groups 123 -> 147 / 158 us | [§4.1 #3][s41] |
| [gemm-grouped-launch](#gemm-grouped-launch) | A fork group's small products in one own-SGEMM launch | – | spec | REJECTED | S5 | 135 atoms step 355 -> 345 us; benchmarks +0.2% to +1.2% | [§4.1 #3][s41] |
| [gemm-skip16-overlap](#gemm-skip16-overlap) | `skip16` GEMMs overlapped across the spline | – | spec | REJECTED | – | not recorded | [§3.3][s33] |
| [gemm-simt-for-lr](#gemm-simt-for-lr) | Float32 SIMT GEMMs only for long-range models | `GenericModel::has_lr` | POLAR | DEFAULT (scoped: POLAR) | S6 | gate: Ada POLAR-S / -L failures -> all pass; cost +0% to +7% | [§5.4][s54] |
| [gen-gate-epilogue](#gen-gate-epilogue) | Gate in the own SGEMM's epilogue / prologue | – | gen | NOT BUILT | – | ceiling ~7 of 47 ms (MH-1 `SrTiO3`, 5060 Ti) | [§4.1 #5][s41] |

### Scheduling, memory and edge data

| ID | Idea | Switch/flag | Path | Status | Stage | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [sched-side-streams](#sched-side-streams) | GEMM fork / join on 8 side streams with own workspaces | – | both | DEFAULT | S2 | serialising them costs +2.0% to +5.7% | [§3.1 #9][s31], [§3.3][s33] |
| [sched-serialised](#sched-serialised) | GEMM groups serialised (check of the above) | – | spec | REJECTED | – | 5060 Ti +5.7% / +5.0%, +2.1% / +2.0% | [§3.3][s33] |
| [sched-reversed-order](#sched-reversed-order) | Kernels between GEMMs walk receivers from the top (L2 reuse) | – | spec | REJECTED | – | step span 8,274 -> 8,285 us | [§3.3][s33] |
| [mem-aliasing](#mem-aliasing) | Gradient buffers alias forward ones (`dm1 -> m1`, `dh0 -> h0`, `dS0 -> S0`) | – | spec | DEFAULT | S3 | not recorded alone | [§3.1 #16][s31] |
| [mem-l1-carveout](#mem-l1-carveout) | Per-arch smallest carveout keeping occupancy | `tune_l1` | both | DEFAULT | S3 | i0 forward 305 -> ~250 us at 64 KB (5060 Ti) | [§3.1 #17][s31], [§2.5][s2] |
| [edge-packed](#edge-packed) | Packed edge arrays (`float4` x, y, z, r; `float2`) | `act_vec_len`, `act_t_sx` | both | DEFAULT | S3 | not recorded alone | [§3.1 #13][s31] |
| [edge-sh-coalesced](#edge-sh-coalesced) | Coalesced SH stores in `scatter_active_edges` | – | spec | SUPERSEDED | S3 | superseded by edge-sh-recompute | [§3.1 #19][s31] |
| [edge-sh-recompute](#edge-sh-recompute) | SH computed where staged, never stored | `edge_sh16` | spec | DEFAULT | S3 | -24 MB of stores per step; `scatter_active_edges` 0.11-0.16 -> 0.05 ms | [§3.1 #24][s31], [§2.5][s2] |
| [edge-compaction-1pass](#edge-compaction-1pass) | Single-pass active-edge compaction (decoupled look-back) | – | spec | REJECTED | – | kernel 96 -> 80 us; MD tie | [§3.3][s33] |
| [mem-reorder](#mem-reorder) | Morton reorder within species groups every 16th rebuild, above L2 | `LOUP_REORDER` | both | DEFAULT (scoped: N x 4 KB > L2) | S2 | part of S1 -> S2 | [§3.1 #8][s31] |
| [mem-reorder-small](#mem-reorder-small) | Morton reorder below L2 size | – | spec | REJECTED | – | not recorded | [§3.3][s33] |
| [mem-shift-cart](#mem-shift-cart) | Cartesian float64 shifts precomputed at rebuild | `d_shift_cart` | both | DEFAULT | S2 | removes the cell from the force graph | [§3.1 #7][s31] |

### Neighbour list, integrator and MD loop

| ID | Idea | Switch/flag | Path | Status | Stage | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [nl-bucketing-sort](#nl-bucketing-sort) | Edge grid rounded to 8,192; `cub::DeviceSegmentedSort` | – | both | DEFAULT | S1-S2 | rebuild sort ~2.3 ms -> ~16 us | [§3.1 #3][s31] |
| [nl-f32-pretest](#nl-f32-pretest) | Float32 pre-test, float64 re-check of borderline pairs; 32-thread blocks (T3) | – | both | DEFAULT | S3 | rebuild 1,222 -> 510 us (5060 Ti), 1,121 -> 684 us (Ada) | [§3.1 #15][s31] |
| [int-langevin-32](#int-langevin-32) | Langevin first half in 32-thread blocks; counter folded into the second half | – | both | DEFAULT | S3 | first half 14.9 -> 8.0 us (`NbCl5`) | [§3.1 #26][s31] |
| [md-evaluate-reuse](#md-evaluate-reuse) | Non-destructive `evaluate()` reusing the list | – | both | DEFAULT | – | not recorded | [§3.1 #10][s31] |
| [md-adaptive-pipeline](#md-adaptive-pipeline) | Start unpipelined, pipeline once the step displacement allows | `LoupMDStats::dangerous_builds`, `pipelined_steps` | both | DEFAULT | S6 | MD unchanged; dangerous builds 1 / ~1,600 steps -> 0 with a 3-step margin | [§3.2 M1, S1][s32], [§5.4][s54] |
| [md-skin](#md-skin) | Larger / adaptive skin (T10) | – | both | PENDING (owner) | – | 0.5-1.5% at 5,000 atoms, ~10% at 135 atoms | [§4.1 #4][s41] |
| [md-small-systems](#md-small-systems) | Small-system latency (< ~1,000 atoms) | – | spec | NOT BUILT (open) | – | 135 atoms: GEMM groups 32% of 358 us | [§4.1 #3][s41] |

### Generic path (other model families, 12-13 November 2026)

| ID | Idea | Switch/flag | Path | Status | Stage | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [gen-probe-export](#gen-probe-export) | Extract every linear map by probing the float64 modules; fold constants numerically | `export_generic.py` | gen | DEFAULT | S6 | design: one exporter for every family | [§5.1][s51] |
| [gen-pair-sym](#gen-pair-sym) | Symmetric pair tables for blocks that do not embed sender and receiver separately | bundle `pair_sym` | gen | DEFAULT | S6 | S(S+1)/2 instead of S^2 tables; size not recorded | [§5.1][s51] |
| [gen-first-layer](#gen-first-layer) | Layer 0 as an outer sum over the radial chain's columns | `first_fwd_kernel`, `first_bwd_kernel`, bundle `first_hidden` | gen | DEFAULT | S6 | 66-column table instead of 4C + 1; OMAT-0 small bundle 40 MB | [§5.1][s51] |
| [gen-factored-products](#gen-factored-products) | FastMACE's factored symmetric contraction, kernels by program signature | `gprod_*` | gen | DEFAULT | S6 | design | [§5.1][s51] |
| [gen-certified-spline](#gen-certified-spline) | FastMACE's certified cubic-Hermite radial spline per pair | `spline.fit` | gen | DEFAULT | S6 | export refuses certificate > 1 | [§5.1][s51] |
| [gen-profile](#gen-profile) | nsys profile of MH-1 `SrTiO3` | – | gen | MEASUREMENT | S6 | SGEMMs ~50%, conv 20%, gate 15% | [§5.5][s55] |

### MACE-POLAR-1 long range (`src/polar.cu`)

| ID | Idea | Switch/flag | Path | Status | Stage | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [pol-double-sf](#pol-double-sf) | Structure factors and feature passes accumulated in double | – | POLAR | DEFAULT | S6 | not recorded alone | [§5.1][s51] |
| [pol-fixed-point-phase](#pol-fixed-point-phase) | Fractional coordinates as 32-bit fixed point; exact n . s mod 1 | `kphase` | POLAR | DEFAULT | S6 | perfect NaCl 10x4x4 residual force 3.0e-4 -> 8.7e-5 eV/A | [§5.4][s54] |
| [pol-mlp-aliasing](#pol-mlp-aliasing) | LayerNorm-MLP reverse wrote its GEMM output over its input | `mlp_bwd` | POLAR | FIX | S6 | NaCl 6x6x6 forces 3-11 eV/A -> 1e-5 eV/A | [§5.4][s54] |
| [pol-corrected-stress](#pol-corrected-stress) | Stress with the k-space terms' cell dependence (MACE PR 1718) | `lr_virial_kernel` | POLAR | DEFAULT | S6 | installed PolarMACE misses up to 2e-5 eV/A^3 | [§5.1][s51] |
| [pol-precise-math](#pol-precise-math) | Precise `expf` / `sqrt` for the float32 stress error | – | POLAR | REJECTED | S6 | no change | this file |
| [pol-stress-diagnosis](#pol-stress-diagnosis) | Where POLAR's float32 stress error comes from | – | POLAR | MEASUREMENT | S6 | local electron energy term; k-space virial to 1e-10 | this file |
| [pol-set-cell](#pol-set-cell) | Re-capture the force graph on every cell change | `PolarLR::set_cell` | POLAR | FIX | round 6 | strained cell on a reused engine: 1e-6 eV/atom, 9e-5 eV/A off -> identical | [§3.2 S5][s32] |

### Measurements without a switch

| ID | What | Status | Result | Evidence |
| --- | --- | --- | --- | --- |
| [meas-stage-spans](#meas-stage-spans) | Stage spans and top kernels, both GPUs | MEASUREMENT | `w'` 36%, spline 23% of the `SrTiO3` step (5060 Ti) | [§2][s2] |
| [meas-ncu](#meas-ncu) | Nsight Compute facts, 5060 Ti | MEASUREMENT | `w'` issue-bound (80% issue slots); spline L2-bound (83%) | [§2][s2] |
| [meas-size-scaling](#meas-size-scaling) | System-size scaling, `SrTiO3` 135-40,000 atoms | MEASUREMENT | linear above ~1,000 atoms: 1.66-1.72 us/atom-step | [§1][s1] |
| [meas-nve-drift](#meas-nve-drift) | NVE drift, `SrTiO3`, 1 ps | MEASUREMENT | -0.00011 / -0.00012 meV/atom/ps | [§1][s1] |
| [meas-ada-counters](#meas-ada-counters) | Ada performance counters | BLOCKED | `ERR_NVGPUCTRPERM`, needs admin | [§4.1 #2][s41] |
| [meas-family-bench](#meas-family-bench) | MD speed of every family, both GPUs, against FastMACE's documented numbers | MEASUREMENT | POLAR-1 level to 20% slower than FastMACE (11 Nov lines) | [§5][s5] |

## Chronology

Step times are `bench_md.py` OMAT-0 medium MD steps, `NbCl5` / `SrTiO3`, ms. Rows are successive builds of the
same benchmark, not A/B pairs. "Tried" lists only attempts whose stage is recorded; most rejected items of
section 3.3 of the status file carry no stage.

| Stage | Tried | Adopted | 5060 Ti | Ada |
| --- | --- | --- | --- | --- |
| S0 | session baseline (before `CUBIC_PACKED8`) | – | 2.384 / 11.380 | 2.533 / 12.985 |
| S1 | 8-byte cubic table, hybrid cubin for `sm_120`, edge-grid bucketing | wp-cubic-packed8, gemm-hybrid-cubin, nl-bucketing-sort | 2.187 / 10.458 | 2.353 / 11.764 |
| S2 | receiver-owned i0 forward, i0 backward prefetch, segmented sort, Cartesian shifts, `sc1_fused`, spatial reorder, forked GEMMs | all of these | 2.089 / 9.939 | 2.260 / 11.184 |
| S3 | receiver-owned `w'`, packed edges, fused assembly, float32 pre-test, aliasing, carveouts, cold-L2 autotune, paired edges, staged q, no split-K, SH staging and recompute, warp-per-receiver i0, Langevin blocks | items 12-26 | 1.930 / 8.858 | 2.090 / 10.021 |
| S4 | shift-free decode, 2^8 derivative, 4-receiver spline on `sm_89` | items 27-29 (partly) | 1.904 / 8.646 | 2.088 / 9.576 |
| S5 | per-system spline / `w'` variants, own SGEMM; SGEMM variants, group policy, split-K, grouped launch, edge-split spline, paired R5 | items 29-31 | **1.849 / 8.458** | **2.054 / 9.27** |
| S6 (12-13 Nov) | review round 5 (adaptive pipeline), generic path for every FastMACE family, POLAR long range, SIMT GEMMs for POLAR, fixed-point phases | md-adaptive-pipeline, gen-*, pol-*, gemm-simt-for-lr | 1.836 / 8.454 | 2.056 / 9.262 |
| round 6 (13 Nov) | review round 6 fixes (S1, S5, S6, S7) | – (correctness) | 1.81 / 8.33 | not run |

## Closed directions

Do not retry these unless the stated condition changes.

| Direction | Evidence | Reopen only if |
| --- | --- | --- |
| Splitting the interaction-1 chain over streams or L2-sized receiver chunks | [int1-two-halves](#int1-two-halves) +0.7% to +0.9%; [int1-l2-chunks](#int1-l2-chunks) +0.2% to +2.8%: per-chunk launch and tail costs exceed the DRAM savings | `lin1` is fused into the chain ([int1-lin1-fusion](#int1-lin1-fusion)) so a chunk no longer pays two GEMM launches |
| Register-raising `w'` variants at 64 registers | [wp-paired-edges](#wp-paired-edges), [wp-x0-hoist](#wp-x0-hoist), [wp-vector2](#wp-vector2) spill or lose on one GPU; [wp-launch-256-3](#wp-launch-256-3) below resolution | `w'` is restructured so it is no longer issue-bound at 63% occupancy |
| Cache-placement schedules for `w'` and the spline | [wp-persistent](#wp-persistent), [wp-no-allocate](#wp-no-allocate), [sched-reversed-order](#sched-reversed-order), [spl-receiver-remap](#spl-receiver-remap): ties or losses | Ada counters ([meas-ada-counters](#meas-ada-counters)) show an L2 limit the 5060 Ti does not have |
| More GEMM tile shapes, split-K, grouped or per-group policies | [gemm-sgemm-variants](#gemm-sgemm-variants), [gemm-split-k-own](#gemm-split-k-own), [gemm-grouped-launch](#gemm-grouped-launch), [gemm-group-policy](#gemm-group-policy): isolated timings do not carry over to concurrent GEMM groups; the small-product floor is launch latency | the graph launch model changes (e.g. programmatic dependent launch measured exact and faster) |
| Split-K cuBLASLt algorithms | [gemm-no-split-k](#gemm-no-split-k): captured memset nodes idle the GPU, non-deterministic | never |
| 3xTF32 GEMMs for MACE-POLAR-1 | [gemm-simt-for-lr](#gemm-simt-for-lr): Ada stress ratios 3.4-6.3 and a NaN | the accuracy gate changes |
| Atomics replaced by extra reduction launches | [frc-per-atom-energy](#frc-per-atom-energy) +0.5%, [i0-bwd-receiver-owned](#i0-bwd-receiver-owned) +1.8%, [frc-segmented-f64](#frc-segmented-f64) | a reproducibility (bitwise) mode is required |

## Open or untried

| Item | State | Where |
| --- | --- | --- |
| Fuse `lin1` into interaction 1 | needs a receiver-tiled tensor-core GEMM consuming spline tiles from shared memory | [int1-lin1-fusion](#int1-lin1-fusion) |
| Ada performance counters | blocked on admin rights on `leeloo` | [meas-ada-counters](#meas-ada-counters) |
| Small-system latency | GEMM launch floor; split-K, grouped launch and edge-split spline rejected | [md-small-systems](#md-small-systems) |
| Larger / adaptive skin | owner decision (changes the benchmark configuration) | [md-skin](#md-skin) |
| Gate fused into the generic path's GEMMs | not built; ceiling ~15% of the MH-1 step | [gen-gate-epilogue](#gen-gate-epilogue) |
| Generic-path GEMMs (~50% of MH-1) | not screened beyond the shared autotune | [gen-profile](#gen-profile) |
| Decaying step-displacement maximum | open review item S2: one fast step keeps the loop unpipelined | [§4.2][s42] |
| `w'` path order and shared scalars | untried: emit the six `x1` paths before the four `x0` paths (shorter live ranges at the 64-register limit) and add the three `xx / xy / xz` pull terms of paths 1, 4, 7 through one summed scalar (6 FMA per channel). Differs from [wp-x0-hoist](#wp-x0-hoist), which added a live accumulator. Test: SASS count and spill bytes first, then A/B | external suggestion, 13 Nov |
| Shift-free decode for the forward's PACKED8 table | untried: the layout change of `w'` (S4) applied to `spline_forward_inter1`; the forward is L2-bound on the 5060 Ti, so expect little there | external suggestion, 13 Nov |
| Exponent field without the add | untried: if all table exponents fall in one 32-wide block, `(c.y & mask) \| prefix` is one LOP3 instead of AND + IADD per path in `w'` | external suggestion, 13 Nov |
| Dead `y1` store in `sc1_fused` | untried, trivial: `d_y1` ([N, 128]) is written every step and read nowhere since `loup_debug_tensors` was removed | external suggestion, 13 Nov |

## Appendix: entries

Each entry has the same fields. Unless stated, the system is OMAT-0 medium on the specialised path.

### Interaction-1 radial spline forward

<a id="spl-paired-edges"></a>
#### spl-paired-edges: paired-edge decode
- Switch: template parameter `PAIR` of `spline_forward_inter1`; chosen per system with spl-multi-receiver.
- Stage: S3 (item 20).
- What: two consecutive edges of a receiver that read the same table slab (edges sorted by `(pair, r)`) decode it once.
- Status: DEFAULT (one of the per-system variants).
- Measured: table loads per edge 1 -> 0.805 (`SrTiO3`), 0.927 (`NbCl5`); spline -7% (5060 Ti), -3% (Ada) on `SrTiO3`. Costs 17-30 registers.
- Why: the spline is L2-bound; fewer slab loads per edge.
- Sources: [§3.1 #20][s31].

<a id="spl-multi-receiver"></a>
#### spl-multi-receiver: R receivers per block, chosen at the first rebuild
- Switch: `spline_forward_inter1<R, PAIR>`, `choose_spline_rpb`, `tune_rebuild_variants`; `LOUP_SPLINE_VARIANT=R1|R1p|R4|R4p|R5` forces one; `LOUP_DEBUG_SPLINE=1` prints the timings.
- Stage: S4-S5 (item 29).
- What: R consecutive receivers per 128R-thread block with per-receiver named barriers and the smallest carveout, so receivers of a symmetric crystal walk the same slabs together and later reads hit L1. Five variants (R1, R1 paired, R4, R4 paired, R5 unpaired) are timed on the last force pass's inputs at the first MD rebuild; the fastest is adopted if >= 2% faster.
- Status: DEFAULT.
- Measured: 5060 Ti `SrTiO3` R4: L1 hit 5% -> 36%, L2 3.92 -> 2.66 GB. Choices: 5060 Ti `SrTiO3` R5, `NbCl5` R1 paired; Ada `SrTiO3` R5, `NbCl5` R4. MD: occupancy rule alone Ada `NbCl5` +1.5% / +0.3%, `SrTiO3` -3.1% / -3.6%; + rebuild-time choice 5060 Ti `SrTiO3` -0.8% / -0.8%; + unpaired and R5 variants 5060 Ti -0.1% to -0.3% (`SrTiO3`), Ada `SrTiO3` -0.4% / -0.9%.
- Why: the gain depends on the system (registers are held until the slowest receiver finishes), hence the per-system timing. Timing at setup on the ideal lattice picked R4 everywhere, which is wrong for thermalised `NbCl5`.
- Sources: [§3.1 #29][s31]; gated per variant by review F1 ([§3.2][s32]).

<a id="spl-launch-128-7"></a>
#### spl-launch-128-7: tighter launch bounds on `sm_89`
- Stage: not recorded.
- What: `(128, 7)` / min-blocks 5 for `spline_forward` on the Ada.
- Status: REJECTED.
- Measured: spills; Ada +17% / +3%.
- Sources: [§3.3][s33].

<a id="spl-16-edge-tiles"></a>
#### spl-16-edge-tiles: 16-edge tiles in the spline
- Status: REJECTED. Measured: tie. Sources: [§3.3][s33].

<a id="spl-receiver-remap"></a>
#### spl-receiver-remap: strided receiver remap
- What: blocks strided over the receiver range for L1 locality.
- Status: REJECTED. Measured: tie. Sources: [§3.3][s33].

<a id="spl-paired-r5"></a>
#### spl-paired-r5: paired 5-receiver blocks and lockstep blocks
- Stage: S4-S5.
- What: R = 5 with pairing (640 threads at 96 registers); lockstep R = 2 / 4 with block-wide barriers.
- Status: REJECTED.
- Measured: paired R5 spills 48-72 B and loses to R1 / R4 (the unpaired R5 fits without spills and is a variant). Lockstep blocks are slower than per-receiver named barriers.
- Sources: [§3.3][s33].

<a id="spl-edge-split"></a>
#### spl-edge-split: edge-split spline for small systems
- Stage: S5.
- What: one receiver per block with its edges over 2 / 4 thread groups and a fixed-order shared-memory reduction.
- Status: REJECTED.
- Measured: 135-atom `SrTiO3` 98 / 82 us against 64 us for the R4 variant; 1.4-1.6x slower at 5,000 atoms.
- Why: even at 135 atoms the 135 blocks keep ~15 warps per SM issuing; the spline is not the small-system limit.
- Sources: [§4.1 #3][s41].

<a id="spl-unroll2"></a>
#### spl-unroll2: 2-edge unroll (before pairing)
- Status: REJECTED. Measured: not recorded. Sources: [§3.3][s33] ("other earlier drops").

<a id="spl-cubic-forward"></a>
#### spl-cubic-forward: the 8-byte cubic table in the forward
- Status: REJECTED. Measured: not recorded. Sources: [§3.3][s33] ("other earlier drops").

<a id="tab-far-edge-4byte"></a>
#### tab-far-edge-4byte: 4-byte table rows for far edges (T2)
- Status: REJECTED.
- Measured: 5060 Ti `w'` 3,286 -> 3,316 us, spline 2,211 -> 2,242 us; Ada `w'` 3,561 -> 3,628 us, spline 2,974 -> 3,075 us.
- Sources: [§3.3][s33].

### Interaction-1 chain as a whole

<a id="int1-radial-low-rank"></a>
#### int1-radial-low-rank: fewer tabulated radial functions
- What: the interaction-1 radial MLP ends in a bias-free linear layer 64 -> 1,280 (10 paths x 128 channels). Two
  proposals (external suggestion, 13 November): (a) tabulate the 64 hidden activations per pair and contract with
  the pair-independent 64 x 1,280 matrix; (b) per channel, tabulate a few basis functions that span its 10 paths.
- Status: NOT BUILT, closed by measurement.
- Measured (CPU, `mace-omat-0-medium.model`, SVD of the last layer's weight, 13 November):
  - whole layer: rank 64 of 64, smallest / largest singular value 0.098. Nothing below the hidden width.
  - per channel, its 10 paths: rank 10 of 10 in all 128 channels (smallest / largest 0.085 to 0.19). A rank-5 basis
    leaves a relative error of 37% (median), rank 8 leaves 18%.
- Why (a) does not pay: 64 FMA per weight against about 17 instructions for one table decode, in a kernel that is
  issue-bound; the first interaction can use the hidden basis only because its input is one scalar per sender
  species, which lets the sum over edges be taken before the contraction.
- Same finding as FastMACE's low-rank table entry (rank 65 of 65) and as the weight statistics of
  `loupmd-review-20261112.md` (fourth pass).
- Sources: this file.

<a id="int1-two-halves"></a>
#### int1-two-halves: two receiver halves on two streams
- What: the chain split into two receiver halves on two streams (spline / GEMM / `w'` overlap), and high-priority GEMM streams.
- Status: REJECTED.
- Measured: 5060 Ti +0.9% / +0.7%.
- Sources: [§3.3][s33].

<a id="int1-l2-chunks"></a>
#### int1-l2-chunks: interaction 1 per L2-sized receiver chunk
- What: spline + `lin1` + `sc1` + `lin1^T` + `w'` per receiver chunk so `m1` / `dm1` stay in L2.
- Status: REJECTED.
- Measured: 5060 Ti 1,700-atom chunks +2.4% / +0.2%, 1,000 +2.4% / +1.1%, 650 +2.6% / +2.8%.
- Why: per-chunk launch and tail costs exceed the DRAM savings; the `lin1` GEMMs run at ~50% DRAM, latency-limited.
- Sources: [§3.3][s33].

<a id="int1-lin1-fusion"></a>
#### int1-lin1-fusion: `lin1` inside the interaction-1 kernels
- What: avoid writing and re-reading `m1` / `dm1` (`N x 4864` floats, 97 MB at 5,000 atoms).
- Status: NOT BUILT.
- Measured: ceiling: the `lin1` groups are 529 + 597 us (5060 Ti `SrTiO3`).
- Why: in the one-receiver-per-block spline each block would read all 655 KB of `W_lin1` (~3.3 GB more L2 traffic per step for an L2-bound kernel), and sharing it over R receivers needs R x 38 accumulator registers. Needs a different decomposition (a receiver-tiled tensor-core GEMM consuming spline tiles from shared memory).
- Sources: [§4.1 #1][s41].

### Interaction-1 backward (`w'`)

<a id="wp-decode-bias-fold"></a>
#### wp-decode-bias-fold: bias of the cubic decode folded into the polynomials
- Switch: `LOUP_GEN_DECODE_FOLD=1` when running `python/gen_kernels.py` (off by default; it changes both `w'` kernels).
- What: each coefficient is decoded as `raw_k - hb` with `hb = 1.5 * 2^e`. Substituting into `w` and `radial_dt`
  gives `w = raw0 + s1 raw1 + s2 raw2 + s3 raw3 - 1.5 (1 + s1 + s2 + s3) 2^e` (and likewise the slope), with the two
  bias factors computed once per edge: 4 FADD + 1 LOP3 per path become 2 FFMA. External suggestion, 13 November.
- Status: BLOCKED on a quiet RTX 5060 Ti (another process held that GPU on 13 November). On the Ada it is a tie, so
  it is not adopted there on its own.
- Measured (13 November, both libraries built from the same tree, differing only in this switch):
  - SASS, `wprime_backward_inter1_rcv`: `sm_120` 1,000 -> 976 instructions and the 8-byte stack spill disappears;
    `sm_89` 1,016 -> 1,000; 64 registers on both, before and after.
  - RTX 4000 Ada, `bench_md.py`, order a b b a, twice: `SrTiO3` 9.316 -> 9.331 ms (1.0015) and 9.455 -> 9.413 ms
    (0.9955); `NbCl5` 1.0002 and 0.9925. Within the run-to-run spread, and the run drifts upward by about 1.5%
    from first to last side, which favours neither.
  - Accuracy, Ada: 46 tests pass; the default gate passes all 30 rows, but its ratios rise: force mean 0.690 -> 0.713
    (max 1.31 -> 1.37), stress mean 0.764 -> 0.940 (max 1.14 -> 1.37). The cause is the rounding of the bias factor
    (about 2^-23 of each entry's scale, the same for every weight of an edge), against exact subtractions before.
- To decide: A/B on the 5060 Ti with `libloupmd_fold.so` (kept beside `libloupmd.so` on both machines). Adopt only if
  it wins there in both orders; a tie on both GPUs with a worse stress ratio is a rejection.
- Sources: this file.

<a id="wp-cubic-packed8"></a>
#### wp-cubic-packed8: 8-byte exact cubic table
- Switch: `d_cubic_packed8_1`.
- Stage: S1 (item 1).
- What: 20 / 19 / 12 / 8-bit signed quantised coefficients around `s = t - 0.5` with a 5-bit shared exponent (clamped to `emax - 31`, review H4) instead of a 16-byte `float4`.
- Status: DEFAULT.
- Measured: alone not recorded; S0 -> S1 (with the hybrid cubin and bucketing) 2.384 -> 2.187 / 11.380 -> 10.458 ms (5060 Ti), 2.533 -> 2.353 / 12.985 -> 11.764 ms (Ada).
- Sources: [§3.1 #1][s31], [§1][s1].

<a id="wp-receiver-owned"></a>
#### wp-receiver-owned: one receiver per block
- Switch: `wprime_backward_inter1_rcv` for N >= 512 (the edge-tiled kernel stays for N < 512); `LOUP_WPRIME_RCV` forces it.
- Stage: S3 (item 12).
- Status: DEFAULT.
- Measured: -170 us on both GPUs.
- Why: the receiver's q is staged once per block.
- Sources: [§3.1 #12][s31].

<a id="wp-channel-major-q"></a>
#### wp-channel-major-q: channel-major staged q
- Stage: S3 (item 21).
- What: `s_q[channel * 44 + slot]` (11 x 16 B, odd stride), so each lane reads its 40 q values as 10 conflict-free `LDS.128`.
- Status: DEFAULT.
- Measured: `w'` -1.7% (5060 Ti), -2.5% (Ada).
- Sources: [§3.1 #21][s31].

<a id="wp-shift-free-decode"></a>
#### wp-shift-free-decode: coefficients packed at their float bit position
- Stage: S4 (item 27).
- What: each coefficient's mantissa bits sit at their float position, so a coefficient decodes as one `LOP3` (or a funnel shift + `LOP3`).
- Status: DEFAULT.
- Measured: inner loop 480 -> 448 static instructions; `w'` 3,177 -> 3,065 us (5060 Ti). MD 5060 Ti -0.5% / -1.2% (A), -0.8% / -1.3% (B); Ada +0.4% / -0.8% (A), -0.6% / -1.1% (B).
- Sources: [§3.1 #27][s31].

<a id="wp-dt-scale"></a>
#### wp-dt-scale: radial derivative scaled by 2^8
- Stage: S4 (item 28).
- What: `fma(ds3', d3, fma(ds2', d2, d1))` with `ds2' = 2^8 ds2`, `ds3' = 2^8 ds3` and one `2^-8` at the epilogue; bit-identical (power-of-two scaling), one `FMUL` less per path.
- Status: DEFAULT.
- Measured: `w'` 3,069 -> 3,010 us, 774 -> 760 M instructions (5060 Ti). MD 5060 Ti -0.1% / -0.5% (A), -0.5% / -0.6% (B); Ada tie.
- Sources: [§3.1 #28][s31].

<a id="wp-block-size"></a>
#### wp-block-size: 256 or 512 threads, chosen per system
- Switch: `(512, 2)` launch bounds, warp stride `blockDim.x / 32`; timed with the spline variants; `LOUP_WPRIME_THREADS=512` forces it.
- Stage: S5 (item 31).
- Status: DEFAULT.
- Measured: 512 threads halve the shared memory per SM (5060 Ti L1 hit 5.9% -> 10.5%, L2 5.27 -> 5.05 GB). Ada `SrTiO3` 3,230 -> 3,130 us (picked); 5060 Ti 3,009 -> 3,034 us and Ada `NbCl5` 519 -> 511 us (within 2%: 256 kept). MD 5060 Ti tie; Ada `SrTiO3` -1.0% / -1.2%.
- Sources: [§3.1 #31][s31].

<a id="wp-vector2"></a>
#### wp-vector2: 2-channel vectorised `w'` (T1)
- Status: REJECTED. Measured: -1.6% at `(256, 3)` on the 5060 Ti, +2.3% on the Ada. Sources: [§3.3][s33].

<a id="wp-paired-edges"></a>
#### wp-paired-edges: paired edges in `w'`
- Status: REJECTED.
- Measured: instructions -3% to -6%, but 64 -> 80 / 128 registers (spills at 64); 3.23 -> 3.28 / 3.35 ms.
- Sources: [§3.3][s33].

<a id="wp-channel-passes"></a>
#### wp-channel-passes: 2 / 4-pass channel staging
- What: smaller `s_dout`, more L1.
- Status: REJECTED. Measured: 3.23 -> 3.40 / 3.80 ms. Sources: [§3.3][s33].

<a id="wp-x0-hoist"></a>
#### wp-x0-hoist: scalar-path `x0` hoisting
- What: accumulate the `x0`-scaled pull terms of the four `a = 0` paths and multiply once per channel.
- Status: REJECTED.
- Measured: -2 M instructions, but spills at 64 registers (3,016 -> 3,086 us); still slower at `(256, 3)` / 80 registers (2,992 -> 3,018 us).
- Sources: [§3.3][s33].

<a id="wp-launch-256-3"></a>
#### wp-launch-256-3: `(256, 3)` launch bounds
- Status: REJECTED.
- Measured: 80 registers, 50% occupancy: 5060 Ti 3,013 -> 2,992 us, Ada 3,184 -> 3,184 / 3,221 us; below A/B resolution.
- Sources: [§3.3][s33].

<a id="wp-persistent"></a>
#### wp-persistent: SM-local persistent `w'`
- What: grid = resident blocks; receivers handed out from per-SM contiguous ranges (`%smid`, atomic cursors, stealing at the end), so co-resident blocks work on neighbouring receivers.
- Status: REJECTED.
- Measured: L1 hit 5.7% -> 12%, L2 bytes 5.3 -> 5.4 GB, 40 B stack; 3,014 -> 3,240 us (5060 Ti).
- Sources: [§3.3][s33].

<a id="wp-no-allocate"></a>
#### wp-no-allocate: table loads without L1 allocation
- What: `ld.global.nc.L1::no_allocate` for the table, to keep L1 for sender rows.
- Status: REJECTED.
- Measured: L1 hit 9.7% -> 1.3%, L2 5.06 -> 5.53 GB, 3,022 -> 3,120 us (5060 Ti).
- Why: adjacent warps do re-read table rows through L1.
- Sources: [§3.3][s33].

### Interaction 0 and the symmetric contractions

<a id="i0-receiver-owned-fwd"></a>
#### i0-receiver-owned-fwd: receiver-owned i0 forward
- Stage: S2 (item 4).
- What: register accumulation per sender-species run, zero writes for absent species; guarded at rebuild end by `check_row_species_order_kernel`.
- Status: DEFAULT.
- Measured: removes the atomics and the 61 MB per-step `S0` / `D0` zero-fill; alone not recorded.
- Sources: [§3.1 #4][s31].

<a id="i0-bwd-prefetch"></a>
#### i0-bwd-prefetch: lane-parallel metadata in i0 backward
- Stage: S2 (item 5). Status: DEFAULT. Measured: not recorded alone. Sources: [§3.1 #5][s31].

<a id="i0-bwd-fused-assembly"></a>
#### i0-bwd-fused-assembly: i0 backward with force / virial assembly (T5)
- Stage: S3 (item 14).
- What: `force_virial_assemble_kernel` folded in; `d_dsh0`, `d_dr0` removed; 32-edge warp chunks.
- Status: DEFAULT. Measured: not recorded alone. Sources: [§3.1 #14][s31].

<a id="i0-bwd-sh-staging"></a>
#### i0-bwd-sh-staging: staged SH and `float2` channel pairs in i0 backward
- Stage: S3 (item 23).
- Status: DEFAULT.
- Measured: 685 -> 564 us (5060 Ti), 736 -> 668 us (Ada); L1 requests per edge ~29 -> ~11.
- Sources: [§3.1 #23][s31], [§2][s2].

<a id="i0-fwd-warp-per-receiver"></a>
#### i0-fwd-warp-per-receiver: warp-per-receiver i0 forward
- Stage: S3 (item 25).
- What: 4 receivers per 128-thread block, 2 channels per lane (`float2` coefficient loads, one SH broadcast per two channels), per-edge D0 term at staging.
- Status: DEFAULT.
- Measured: 390 -> 245 us (5060 Ti, 64 KB carveout), 415 -> 304 us (Ada).
- Sources: [§3.1 #25][s31].

<a id="i0-sh-shuffle"></a>
#### i0-sh-shuffle: SH shuffle broadcast in i0 backward (old idea 3)
- Status: SUPERSEDED. Measured: tie; replaced by i0-bwd-sh-staging. Sources: [§3.3][s33].

<a id="i0-bwd-receiver-owned"></a>
#### i0-bwd-receiver-owned: receiver-owned i0 backward (T7)
- Status: REJECTED.
- Measured: +1.8% / +1.1% (5060 Ti).
- Why: per-chunk edge parallelism beats per-receiver runs.
- Sources: [§3.3][s33].

<a id="i0-bwd-micro"></a>
#### i0-bwd-micro: i0 backward micro-variants
- What: SASS stall sampling put ~14% of samples on the first FFMA consuming the edge's coefficients. Tried: next edge's coefficients prefetched into registers (96 registers); coefficient loads at the top of the iteration (80 registers); `grad_a` in 4 interleaved partial sums.
- Status: REJECTED.
- Measured: 551 -> 571 us; 554 -> 564 us; 554 -> 626 us (+4 M instructions).
- Sources: [§3.3][s33].

<a id="i0-per-species-fwd"></a>
#### i0-per-species-fwd: per-species i0 forward
- Status: REJECTED. Measured: not recorded. Sources: [§3.3][s33] ("other earlier drops").

<a id="sc-sc1-fused"></a>
#### sc-sc1-fused: fused `sc1` forward, readout and backward
- Stage: S2 (item 6).
- What: `sc1_forward`, the `128 -> 16 -> 1 -> 16 -> 128` readout forward / backward and `sc1_backward` in one `N x 128` launch.
- Status: DEFAULT. Measured: not recorded alone. Sources: [§3.1 #6][s31].

<a id="sc-sc0-bwd-launch"></a>
#### sc-sc0-bwd-launch: `sc0_backward` at `(128, 3)`
- Status: REJECTED. Measured: not recorded. Sources: [§3.3][s33].

<a id="sc-sc0-overlap"></a>
#### sc-sc0-overlap: `sc0` per species beside the other species' products
- What: forward `sc0_forward` of species s on its l = 3 lane after its four products; backward `sc0_backward` of s, then its products, while the next species' `sc0_backward` runs (16 side streams).
- Status: REJECTED.
- Measured: 5060 Ti `SrTiO3` i0 forward end -> spline start 470 -> 489 us, `w'` end -> i0 backward start 600 -> 647 us; `NbCl5` -4 / +1 us.
- Why: the DRAM-bound `sc0` kernels compete with the products.
- Sources: [§3.3][s33].

<a id="frc-segmented-f64"></a>
#### frc-segmented-f64: receiver-warp segmented float64 force reduction
- Status: REJECTED. Measured: not recorded. Sources: [§3.3][s33].

<a id="frc-per-atom-energy"></a>
#### frc-per-atom-energy: per-atom energy array (T6)
- Status: REJECTED.
- Measured: +0.5% / +0.1% (5060 Ti).
- Why: the extra reduction launch costs more than the removed atomics.
- Sources: [§3.3][s33].

### GEMMs and the autotune

<a id="gemm-hybrid-cubin"></a>
#### gemm-hybrid-cubin: 3xTF32 hybrid GEMM cubins
- Switch: `cubins/tf32x3_hyb_sm120.cubin` on `sm_120`, the `sm_80` cubin on `sm_89`; three tiles (64 x 64, 32 x 64, 64 x 32) as autotune candidates; `LOUP_NO_HYBRID=1` disables them.
- Stage: S1 (item 2); scoped in S6 (gemm-simt-for-lr).
- Status: DEFAULT (scoped: every model without long-range terms).
- Measured: alone not recorded (part of S0 -> S1). Ada uses them for most products ([§2][s2] top kernels: `fm_tf32x3_*` 1,179 / 750 us on `SrTiO3`).
- Sources: [§3.1 #2][s31].

<a id="gemm-autotune"></a>
#### gemm-autotune: per-shape GEMM autotune
- Switch: `sgemm_rm`; `LOUP_DEBUG_GEMM=1` prints every choice.
- What: up to 32 cuBLASLt heuristics, 3 hybrid tiles and the own SGEMM timed per shape in an uncaptured pass; the fork is disabled while timing. A winner within 3% of the runner-up is used but not persisted (review F4).
- Status: DEFAULT. Measured: not recorded alone. Sources: [§3.1 #11][s31], [§3.2 F4][s32].

<a id="gemm-cold-l2"></a>
#### gemm-cold-l2: selective cold-L2 timing
- Stage: S3 (item 18).
- What: operands larger than half the L2 are timed with a flushed L2, because they stream from DRAM in the force pass.
- Status: DEFAULT.
- Measured: Ada `lin1_fwd` group 672 -> 575 us (the 64 x 32 hybrid tile won hot and ran ~2x slower in the pass).
- Sources: [§3.1 #18][s31].

<a id="gemm-cold-l2-all"></a>
#### gemm-cold-l2-all: every GEMM timed cold
- Status: REJECTED. Measured: Ada `NbCl5` +2.9%. Sources: [§3.3][s33].

<a id="gemm-no-split-k"></a>
#### gemm-no-split-k: no split-K cuBLASLt algorithms
- Switch: `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = NONE` (plan cache `LOUPPLN3` at the time).
- Stage: S3 (item 22).
- Status: DEFAULT.
- Measured: their captured per-launch workspace memsets left the 5060 Ti idle ~120 us per step; GPU idle inside a step is now ~20 us.
- Why: also order-deterministic; K <= 384, so split-K has nothing to gain.
- Sources: [§3.1 #22][s31], [§2][s2].

<a id="gemm-plan-cache"></a>
#### gemm-plan-cache: persistent GEMM plan cache
- Switch: `~/.cache/loupmd/plans_<gpu>_sm<cc>_lt<ver>_hyb<0|1>_<library hash>.bin`; `LOUP_CACHE_DIR`, `LOUP_RETUNE`.
- Stage: S3 (item 16); S6: library FNV hash in the name and the 3% persistence rule (review F4).
- Status: DEFAULT.
- Measured: cold-start `evaluate` 1.3 -> 0.21 s (5060 Ti), 0.76 -> 0.05 s (Ada).
- Sources: [§3.1 #16][s31], [§3.2 F4][s32].

<a id="gemm-own-sgemm"></a>
#### gemm-own-sgemm: own SIMT SGEMM
- Switch: `src/sgemm.cu`, `loup::sgemm128`; plan field `use_own` (cache `LOUPPLN4`).
- Stage: S5 (item 30).
- What: 128 x 128 x 16 block tiles, 256 threads with 8 x 8 outputs, 3-stage `cp.async` pipeline, 126 registers, 48 KB shared memory, 2 blocks (16 warps) per SM where cuBLAS' SIMT kernels run 8 warps at 200-210 registers. Transposed weights get a [K, N] copy. Bit-identical to `cublasSgemm` on every tested shape.
- Status: DEFAULT (an autotune candidate; picked for the large `lin1` / `lin1^T` / `compose0` forward products on the 5060 Ti).
- Measured: cold microbenchmark (5060 Ti) `lin1` l = 1 / 2 / 3 forward 132 / 196 / 181 us (cuBLASLt's pick 136 / 207 / 175); MD first version vs item 29: 5060 Ti -1.6% / -2.0% (`NbCl5`), -0.6% / -0.7% (`SrTiO3`), Ada +0.8% / +0.2% (A), -0.2% / -0.3% (B); `cp.async` vs first version: 5060 Ti -1.4% / -1.3%, -0.1% / -0.4%; Ada +1.1% / +0.5% (A), -0.3% / +0.0% (B).
- Sources: [§3.1 #30][s31].

<a id="gemm-own-n192"></a>
#### gemm-own-n192: 128 x 64 tiles for N = 192
- Stage: S5.
- Status: DEFAULT (an MD tie, kept).
- Measured: 111 registers, 36 KB; the 5060 Ti autotune picks it for all seven `SrTiO3` `compose0` backward products (297 -> 248 us); MD 5060 Ti `NbCl5` +0.5% / +0.0%, `SrTiO3` -0.1% / -0.4%; Ada +1.2% / +1.0% (A), +0.0% / -0.2% (B).
- Sources: [§3.1 #30][s31].

<a id="gemm-sgemm-variants"></a>
#### gemm-sgemm-variants: other own-SGEMM shapes and pipelines
- Stage: S5.
- Status: REJECTED.
- Measured (5060 Ti): `cp.async` 4 stages with BK = 8: 1.2-1.5x slower than BK = 16; register-staged with BK = 8 or 1 block per SM: 5-30% slower; 64 x 128 and 128 x 64 tiles as extra candidates: picked in isolation, MD `NbCl5` +1.6% / +0.8%; 64 x 128 for every N = 128 product (cold microbenchmark 7-8% faster): MD `NbCl5` +1.7% / +1.8%, `SrTiO3` +0.5% / +0.8%, Ada +1.1% / +0.2% (A).
- Why: isolated cold timings of tile shapes do not carry over to the concurrent GEMM groups.
- Sources: [§3.3][s33].

<a id="gemm-pretransposed"></a>
#### gemm-pretransposed: pre-transposed forward weights (old idea 1)
- Status: REJECTED. Measured: 5060 Ti tie, Ada +1.0% / +0.3%. Sources: [§3.3][s33].

<a id="gemm-grouped-species"></a>
#### gemm-grouped-species: grouped species GEMM (T4)
- Switch: `fm_tf32x3_species_rr_64x64` for `compose0_fwd` + `skip16`.
- Status: REJECTED.
- Measured: 5060 Ti +1.7% / +0.8%; the grouped launch is slower than the forked per-species GEMMs.
- Sources: [§3.3][s33].

<a id="gemm-row-chunked"></a>
#### gemm-row-chunked: row-chunked hybrid GEMM for cold operands
- What: the 64-wide hybrid tiles launched per quarter-L2 row chunk so the second column tile re-reads A from L2.
- Status: REJECTED. Measured: never chosen by the cold autotune on either GPU (Ada `lin1` forward stays on cuBLASLt at 148 / 228 / 209 us). Sources: [§3.3][s33].

<a id="gemm-group-policy"></a>
#### gemm-group-policy: own-SGEMM policy per fork group, timed in context
- Stage: S5.
- What: each fork group timed whole at a second uncaptured pass under three policies (as autotuned / own wherever eligible / never own); the fastest kept per group signature.
- Status: REJECTED.
- Measured: group timings differ from the per-product picks (`SrTiO3` `lin1^T` group 637 -> 552 us with own everywhere, warm L2), but MD 5060 Ti `NbCl5` -0.4% to -0.8%, `SrTiO3` +0.1% / +0.2%, Ada tie (stream-timed and graph-timed variants).
- Why: not worth the collection / replay machinery.
- Sources: [§3.3][s33].

<a id="gemm-split-k-own"></a>
#### gemm-split-k-own: split-K own SGEMM for small systems
- Stage: S5.
- What: partials per K slice, the last block of a tile sums them in z order.
- Status: REJECTED.
- Measured: the isolated autotune cannot rank such products (every candidate 6-14 us, launch-bound); forcing split 2 / 4: 135-atom GEMM groups 123 -> 147 / 158 us (step 367 -> 392 / 409 us).
- Why: the per-GEMM floor in the graph is launch / completion latency, not the K pipeline.
- Sources: [§4.1 #3][s41].

<a id="gemm-grouped-launch"></a>
#### gemm-grouped-launch: one own-SGEMM launch per fork group of small products
- Stage: S5.
- What: products of a fork group collected and, when each has fewer tiles than SMs, issued as one launch over a problem table (64 x 64 tiles).
- Status: REJECTED.
- Measured: 135-atom GEMM groups 114 -> 101 us (step 355 -> 345 us, wall -3% to -5%, 5060 Ti); 320 atoms within noise; Ada 135 atoms +2%; the refactored kernel cost the benchmarks +1.2% / +0.9% (`NbCl5`), +0.3% / +0.2% (`SrTiO3`).
- Sources: [§4.1 #3][s41].

<a id="gemm-skip16-overlap"></a>
#### gemm-skip16-overlap: `skip16` GEMMs across the spline
- Status: REJECTED. Measured: not recorded. Sources: [§3.3][s33].

<a id="gemm-simt-for-lr"></a>
#### gemm-simt-for-lr: float32 SIMT GEMMs only for long-range models
- Switch: `loup_create` skips `HybCubin::init` when `GenericModel::has_lr()`; `LOUP_NO_HYBRID=1` does it for any model.
- Stage: S6 (13 November 2026).
- What: MACE-POLAR-1's SCF charge updates amplify GEMM rounding; 3xTF32 products have about 4x float32's rounding per product.
- Status: DEFAULT (scoped: long-range models).
- Measured: gate on the Ada with hybrid GEMMs: POLAR-1-S NaCl stress ratios 5.16 / 3.43, NaNbO3_c0 NaN; POLAR-1-L NaCl_c1 stress 3.52, NaNbO3 force 3.13, stress 6.27 / 3.63. With SIMT GEMMs (`LOUP_NO_HYBRID=1`, then the scoped build): every POLAR configuration passes on both GPUs (Ada worst 0.99 / 1.14 / 2.69 for S / M / L; 5060 Ti 1.30 / 0.97 / 1.16). Cost, separate `bench_family.py` runs (not an A/B): 5060 Ti NaCl 216 atoms +1.8% to +3.0%, 1,728 atoms +0.0% to +0.4%; Ada NaCl 1,728 atoms +5.4% / +7.2% / +6.9% (S / M / L), POLAR-1-M `NbCl5` +5.1% (the Ada's earlier run also predates pol-fixed-point-phase).
- Why: correctness first; the other families' gates pass with the hybrid GEMMs on both GPUs.
- Sources: [§5.4][s54], [§5.2][s5].

<a id="gen-gate-epilogue"></a>
#### gen-gate-epilogue: generic gate fused into the GEMMs
- What: the nonlinear blocks' gate reads the rows a GEMM just wrote and writes rows the next GEMM reads; fuse it into the own SGEMM's epilogue / prologue.
- Status: NOT BUILT.
- Measured: ceiling = the gate kernels' time, ~7 of 47 ms (MH-1 `SrTiO3`, 5060 Ti: gate backward 10%, forward 5%).
- Falsifier: stop if the fused GEMM loses more than the gate kernels cost.
- Sources: [§4.1 #5][s41], [§5.5][s55].

### Scheduling, memory and edge data

<a id="sched-side-streams"></a>
#### sched-side-streams: side-stream GEMM fork / join
- Stage: S2 (item 9).
- What: 8 non-blocking side streams, each with its own 4 MB cuBLASLt workspace, inside the captured graph.
- Status: DEFAULT.
- Measured: alone not recorded; serialising them costs +2.0% to +5.7% (sched-serialised).
- Sources: [§3.1 #9][s31].

<a id="sched-serialised"></a>
#### sched-serialised: GEMM groups without side streams
- Status: REJECTED (a check of sched-side-streams).
- Measured: 5060 Ti `NbCl5` +5.7% / +5.0%, `SrTiO3` +2.1% / +2.0%.
- Sources: [§3.3][s33].

<a id="sched-reversed-order"></a>
#### sched-reversed-order: reversed block order between GEMMs
- What: spline, `sc0_forward`, `sc1_fused`, `w'`, `sc0_backward` and i0 forward / backward walk receivers from the top, so they first read the rows the preceding GEMM wrote last and leave the rows the next GEMM reads first in L2.
- Status: REJECTED.
- Measured: step span 8,274 -> 8,285 us (5060 Ti `SrTiO3`); no stage gained more than 6 us.
- Sources: [§3.3][s33].

<a id="mem-aliasing"></a>
#### mem-aliasing: gradient buffers aliased onto forward buffers
- Stage: S3 (item 16). What: `dm1 -> m1`, `dh0 -> h0`, `dS0 -> S0`. Status: DEFAULT. Measured: not recorded alone. Sources: [§3.1 #16][s31].

<a id="mem-l1-carveout"></a>
#### mem-l1-carveout: per-arch L1 carveouts
- Switch: `tune_l1`.
- Stage: S3 (item 17).
- What: the smallest shared-memory configuration that keeps occupancy for the gather kernels, per arch.
- Status: DEFAULT.
- Measured: i0 forward L1 hit 53% at the 64 KB configuration against 33% at the default 100 KB (305 us there).
- Sources: [§3.1 #17][s31], [§2][s2].

<a id="edge-packed"></a>
#### edge-packed: packed edge arrays
- Stage: S3 (item 13). What: `act_vec_len` (`float4` x, y, z, r), `act_t_sx` (`float2`). Status: DEFAULT. Measured: not recorded alone. Sources: [§3.1 #13][s31].

<a id="edge-sh-coalesced"></a>
#### edge-sh-coalesced: coalesced SH stores
- Stage: S3 (item 19). Status: SUPERSEDED by edge-sh-recompute. Sources: [§3.1 #19][s31].

<a id="edge-sh-recompute"></a>
#### edge-sh-recompute: SH computed where staged
- Switch: `edge_sh16`.
- Stage: S3 (item 24).
- What: `scatter_active_edges` no longer writes 64 B per edge; i0 forward / backward and the spline compute the 16 values from `act_vec_len` while staging.
- Status: DEFAULT.
- Measured: 24 MB fewer stores per step at 5,000 atoms; `scatter_active_edges` 0.11-0.16 -> 0.05 ms.
- Sources: [§3.1 #24][s31], [§2][s2].

<a id="edge-compaction-1pass"></a>
#### edge-compaction-1pass: single-pass active-edge compaction
- What: decoupled look-back over 4-row tiles, pass-1 vectors cached in shared memory.
- Status: REJECTED.
- Measured: kernel 96 -> 80 us (5060 Ti `SrTiO3`), equal on `NbCl5`; MD tie on both GPUs in both orders.
- Why: the simpler count + scan + scatter path stays.
- Sources: [§3.3][s33].

<a id="mem-reorder"></a>
#### mem-reorder: L2-gated spatial atom reorder
- Switch: on when `N * 4096 > l2_bytes`; `LOUP_REORDER=0|1` forces it.
- Stage: S2 (item 8).
- What: Morton reorder within species groups every 16th rebuild.
- Status: DEFAULT (scoped: above L2 size). Gate: reorder on vs off agree to 5e-7 eV/A (review F1).
- Sources: [§3.1 #8][s31], [§3.2 F1][s32].

<a id="mem-reorder-small"></a>
#### mem-reorder-small: Morton reorder below L2 size
- Status: REJECTED. Measured: not recorded. Why: the reorder's extra sync is pure overhead when the system fits L2. Sources: [§3.3][s33].

<a id="mem-shift-cart"></a>
#### mem-shift-cart: Cartesian float64 shifts
- Switch: `d_shift_cart`. Stage: S2 (item 7). What: precomputed at rebuild; removes the cell from the force graph. Status: DEFAULT. Sources: [§3.1 #7][s31].

### Neighbour list, integrator and MD loop

<a id="nl-bucketing-sort"></a>
#### nl-bucketing-sort: edge-grid bucketing and segmented sort
- Stage: S1 (bucketing), S2 (`cub::DeviceSegmentedSort::StableSortPairs`).
- What: the edge grid rounded to 8,192 so graphs survive edge-count changes; the rebuild sort as a segmented sort.
- Status: DEFAULT.
- Measured: rebuild sort ~2.3 ms -> ~16 us.
- Sources: [§3.1 #3][s31].

<a id="nl-f32-pretest"></a>
#### nl-f32-pretest: float32 pair pre-test (T3)
- Stage: S3 (item 15).
- What: float32 distance test in `count` / `fill_neighbors` with borderline pairs re-checked in float64; 32-thread blocks.
- Status: DEFAULT.
- Measured: rebuild 1,222 -> 510 us (5060 Ti), 1,121 -> 684 us (Ada).
- Sources: [§3.1 #15][s31].

<a id="int-langevin-32"></a>
#### int-langevin-32: Langevin first half in 32-thread blocks
- Stage: S3 (item 26).
- What: the first half's float64 Box-Muller is FP64-throughput bound on consumer GPUs; the step-counter increment moves into the second half (one graph node less). Random stream bit-identical.
- Status: DEFAULT.
- Measured: 14.9 -> 8.0 us (`NbCl5`).
- Sources: [§3.1 #26][s31].

<a id="md-evaluate-reuse"></a>
#### md-evaluate-reuse: non-destructive `evaluate()`
- What: keeps velocities, skips rebuilds when `max_disp < skin/2` and the cell is unchanged.
- Status: DEFAULT. Measured: not recorded. Sources: [§3.1 #10][s31].

<a id="md-adaptive-pipeline"></a>
#### md-adaptive-pipeline: adaptive pipelined MD loop
- Switch: `LoupMDStats::dangerous_builds`, `LoupMDStats::pipelined_steps`; `check_skin_kernel`, `d_max_step_disp2`.
- Stage: S6 (review round 5, M1, 12 November 2026); margin raised 13 November; switch rule tightened in review round 6 (S1).
- What: pipelining reads step k's skin check after step k + 1 is launched, so two force calls follow each check. The loop starts unpipelined (exact for any skin) and pipelines once 4 x the largest per-step displacement (tracked on the device) is below skin/2; the pipelined check lowers its threshold by three times that displacement (two steps of lag, one of margin). Round 6: the unpipelined check also flags atoms beyond the pipelined threshold, and the loop switches only when none is.
- Status: DEFAULT.
- Measured: MD speed unchanged (5060 Ti 1.836 / 8.454 ms, Ada 2.056 / 9.262 ms). With a two-step margin, 1 of ~1,600 steps of three 300 K benchmarks ran on an incomplete list; with three, 0 in every benchmark on both GPUs. Round 6: 9 incomplete-list force calls in 60 switches of the regression test -> 0.
- Sources: [§3.2 M1, S1][s32], [§5.4][s54].

<a id="md-skin"></a>
#### md-skin: larger / adaptive skin (T10)
- Status: PENDING (owner decision: it changes the benchmark configuration).
- Measured: estimate 0.5-1.5% at 5,000 atoms, ~10% at 135 atoms.
- Sources: [§4.1 #4][s41].

<a id="md-small-systems"></a>
#### md-small-systems: small-system latency
- Status: NOT BUILT (open).
- Measured: 135-atom `SrTiO3` step 358 us: GEMM groups 115 us (32%; ~25 tiny products at ~6-13 us each regardless of size), edge-tiled `w'` 100 us, spline 58 us, i0 backward 28 us. Tried and rejected: gemm-split-k-own, gemm-grouped-launch, spl-edge-split.
- Sources: [§4.1 #3][s41].

### Generic path

Families: MACE-OMAT-0 small, MACE-OFF23 S / M / L, OFF23b M, OFF24 M, MACE-MH-1 (any head), MACE-POLAR-1 S / M / L.
These items are design decisions of the first version (12-13 November 2026); none was A/B-timed against an
alternative.

<a id="gen-probe-export"></a>
#### gen-probe-export: probe-based weight extraction
- Switch: `python/export_generic.py` (exporter version 2), `python/export.py` dispatcher.
- What: every linear map extracted by probing the float64 e3nn modules with basis inputs, so path weights, `1/avg_num_neighbors`, normalize2mom constants, ScaleShift and the head are folded numerically; conv paths checked to be multiples of `wigner_3j` with the factor folded into the radial table.
- Status: DEFAULT.
- Sources: [§5.1][s51].

<a id="gen-pair-sym"></a>
#### gen-pair-sym: symmetric radial pair tables
- Switch: bundle meta `pair_sym`.
- What: blocks whose radial MLP does not embed sender and receiver separately get one table per unordered species pair.
- Status: DEFAULT.
- Measured: bundle size not recorded against the ordered-pair version. Models with nonlinear (ordered-pair) blocks stay large: MH-1 74-200 MB, POLAR-1-L 790 MB for 3 species.
- Sources: [§5.1][s51], [§5.5][s55].

<a id="gen-first-layer"></a>
#### gen-first-layer: layer 0 as an outer sum over the radial chain's columns
- Switch: `first_fwd_kernel` / `first_bwd_kernel`; bundle meta `first_hidden`, `J = H + 2`.
- What: layer 0's input is the species embedding, so the conv weights, the layer input, the path factors and the message linear fold into one matrix per degree; the per-edge table holds the radial chain's columns (66) instead of 4C + 1 weights.
- Status: DEFAULT.
- Measured: OMAT-0 small bundle 40 MB; speed against the plain conv not measured. In the MH-1 profile the first-layer kernels are 5% of the step.
- Sources: [§5.1][s51], [§5.5][s55].

<a id="gen-factored-products"></a>
#### gen-factored-products: factored symmetric contractions
- Switch: `gprod_fwd_<sig>` / `gprod_bwd_<sig>` (three configurations generated by `python/gen_generic.py`).
- What: FastMACE's factored program (`sc_factored.factor` / `fold_weights`), one kernel per program signature.
- Status: DEFAULT. Measured: products 5% of the MH-1 step. Sources: [§5.1][s51], [§5.5][s55].

<a id="gen-certified-spline"></a>
#### gen-certified-spline: certified radial splines
- What: FastMACE's cubic-Hermite spline fit per species pair; the export refuses a certificate above 1 (spline error above the float32 network's own rounding error on `radial.certify`'s points).
- Status: DEFAULT. Sources: [§5.1][s51].

<a id="gen-profile"></a>
#### gen-profile: generic-path profile
- What: nsys over 200 Langevin steps, MH-1 `omat_pbe`, `SrTiO3` 5,000 atoms, 5060 Ti, 46.9 ms per step.
- Status: MEASUREMENT.
- Measured: SIMT SGEMMs ~50% (cuBLAS CUTLASS SIMT kernels 39%, own SGEMM 8%, magma 2%), `gconv_bwd_L1` 11.7%, `gconv_fwd_L1` 7.9%, gate backward 9.6%, gate forward 4.7%, first-layer kernels 4.7%, products 4.6%.
- Sources: [§5.5][s55].

### MACE-POLAR-1 long range

<a id="pol-double-sf"></a>
#### pol-double-sf: double accumulation in k-space
- What: structure factors over atoms and the per-atom feature / gradient passes over k accumulate in double; k data tiled in shared memory.
- Status: DEFAULT. Measured: not recorded alone. Sources: [§5.1][s51].

<a id="pol-fixed-point-phase"></a>
#### pol-fixed-point-phase: exact fixed-point k-space phases
- Switch: `frac_kernel` (32-bit fixed point), `kphase`.
- Stage: S6 (13 November 2026).
- What: fractional coordinates are stored as `s * 2^32 mod 2^32`; `n . s mod 1` comes out of wrapping integer arithmetic, then `sincospif` on [-0.5, 0.5). Before: float fractional coordinates, a phase error growing with |n|.
- Status: DEFAULT.
- Measured: perfect NaCl 10x4x4 (POLAR-1-S), residual force 3.0e-4 -> 8.7e-5 eV/A.
- Sources: [§5.4][s54].

<a id="pol-mlp-aliasing"></a>
#### pol-mlp-aliasing: MLP reverse writing over its own input
- Switch: `mlp_bwd` in `src/polar.cu`; `ln_silu_bwd_kernel` now runs in place.
- Stage: S6 (13 November 2026).
- What: after the first hidden layer the reverse GEMM's destination was the buffer holding its input dz. Found while chasing NaN temperatures in POLAR MD.
- Status: FIX.
- Measured: perfect NaCl 6x6x6 (1,728 atoms), POLAR-1-S: max force 4.8 eV/A (2.99 and 10.8 on repeated calls) -> 1.1e-5 eV/A; NVE no longer heats (T 302 -> 429 K in 6 fs before, 300 -> 290 K after). Smaller cells at rounding level: NaCl 7x4x4 residual 1.1e-4 -> 2.5e-6 eV/A.
- Why: harmless while one GEMM tile covered all rows; from ~1,700 atoms the tiles raced.
- Sources: [§5.4][s54].

<a id="pol-corrected-stress"></a>
#### pol-corrected-stress: stress with the k-space cell dependence
- Switch: `lr_virial_kernel`.
- What: the k-space energy's dependence on the cell (MACE PR 1718, as FastMACE). The gate's POLAR stress reference is the float64 energy's finite difference.
- Status: DEFAULT.
- Measured: the installed PolarMACE (0.3.16) hands the k-space terms the unstrained geometry, so its autograd stress misses up to 2e-5 eV/A^3 (NaCl). Reference vs finite difference 2.4e-9, LoupMD vs finite difference 8e-8 eV/A^3.
- Sources: [§5.1][s51].

<a id="pol-precise-math"></a>
#### pol-precise-math: precise `expf` / `sqrt` in POLAR
- Stage: S6 (12 November 2026).
- What: an attempt to cut the float32 stress error on symmetric crystals (marginal failures on the Ada at the time).
- Status: REJECTED.
- Measured: the stress error did not change (numbers not kept). The failures were later traced to the 3xTF32 GEMMs (gemm-simt-for-lr).
- Sources: session record of 12 November 2026 (not in the status file).

<a id="pol-stress-diagnosis"></a>
#### pol-stress-diagnosis: origin of POLAR's float32 stress error
- Stage: S6 (12 November 2026).
- Status: MEASUREMENT.
- Measured: the error is an isotropic diagonal and comes from the local electron energy term (it contributes -0.043 eV/A^3 to the stress; LoupMD's relative error there ~3e-6); the k-space virial matches the reference to 1e-10; without that term the error is ~2e-8. MACE float32's own error on the same cells varies 1e-8 to 1.8e-7.
- Sources: session record of 12 November 2026 (not in the status file).

<a id="pol-set-cell"></a>
#### pol-set-cell: force graph re-captured on cell changes
- Switch: `PolarLR::set_cell` reports every cell change.
- Stage: review round 6 (S5, 13 November 2026).
- Status: FIX.
- Measured: NaCl 2x2x2, POLAR-1-S, strain 3e-4 on the same engine: 1e-6 eV/atom, 9e-5 eV/A and 3e-7 eV/A^3 against a fresh engine before; identical now.
- Sources: [§3.2 S5][s32].

### Measurements without a switch

<a id="meas-stage-spans"></a>
#### meas-stage-spans: stage spans and top kernels
- What: nsys with `--cuda-graph-trace=node`, 100 Langevin steps after 50 warm-up; medians over non-rebuild steps; both GPUs.
- Measured: 5060 Ti `SrTiO3` step span 8,232 us: `w'` 2,961, spline 1,928, GEMM groups 1,825 (22%), i0 backward 524. Ada 9,268 us.
- Sources: [§2][s2].

<a id="meas-ncu"></a>
#### meas-ncu: Nsight Compute facts (5060 Ti)
- Measured: `w'` 3.01 ms, 63% occupancy, issue slots 80% busy (issue-bound, `not_selected` dominant), 760 M instructions, L2 5.3 GB; spline R1 2.04 ms, L2 83%, L1 hit 5%; R4 L1 hit 36%.
- Sources: [§2][s2].

<a id="meas-size-scaling"></a>
#### meas-size-scaling: system-size scaling
- Measured (5060 Ti, `SrTiO3` n^3, us per atom-step): 135 atoms 3.00, 320 atoms 2.18, 1,080 1.73, 5,000 1.66-1.68, 20,480 1.69-1.71, 40,000 1.71-1.72.
- Sources: [§1][s1].

<a id="meas-nve-drift"></a>
#### meas-nve-drift: NVE energy drift
- Measured: `SrTiO3` 10^3, 1 fs, 300 K, 1 ps after 200 Langevin steps: -0.00011 meV/atom/ps (5060 Ti), -0.00012 (Ada).
- Sources: [§1][s1].

<a id="meas-ada-counters"></a>
#### meas-ada-counters: Ada performance counters
- Status: BLOCKED (`ERR_NVGPUCTRPERM` on `leeloo`; needs `NVreg_RestrictProfilingToAdminUsers=0`).
- Why it matters: `w'` is 9% slower on the Ada than on the 5060 Ti although the Ada has 33% more SMs; the spline gained 10% from L1 slab sharing there, so the Ada looks L2-sensitive.
- Sources: [§4.1 #2][s41].

<a id="meas-family-bench"></a>
#### meas-family-bench: family benchmarks against FastMACE
- What: `python/bench_family.py` (Langevin 300 K, 1 fs, skin 0.5 A, median of 6 x 256 steps) for every family on both GPUs.
- Measured: see [§5][s5]. Against FastMACE's raw A/B lines of 11 November (same 3x3x3 cells) LoupMD's POLAR-1 is level to 20% slower (S 1.13, M 1.16 / 1.20, L 1.03 on the 5060 Ti; M 1.01 / 1.07 on the Ada). MH-1 NaCl 216 atoms 2.33 ms against FastMACE's documented 2.65-2.90 ms (5060 Ti, `matpes_r2scan`).
- Correction: an earlier version of the status table compared with FastMACE's October baseline (`SPEC.md` 3.3) and made LoupMD's POLAR-1 look 1.5-1.9x faster; that comparison was wrong.
- Sources: [§5][s5].

[status]: STATUS_AND_PROFILE.md
[s1]: STATUS_AND_PROFILE.md#1-benchmark--verification-summary
[s2]: STATUS_AND_PROFILE.md#2-detailed-kernel-profile-current-build
[s31]: STATUS_AND_PROFILE.md#31-implemented-optimisations
[s32]: STATUS_AND_PROFILE.md#32-correctness--api-fixes-applied-from-code-review
[s33]: STATUS_AND_PROFILE.md#33-ruled-out-measured
[s41]: STATUS_AND_PROFILE.md#41-remaining-performance-opportunities
[s42]: STATUS_AND_PROFILE.md#42-open-review-hardening-items-correctness--robustness
[s5]: STATUS_AND_PROFILE.md#5-model-families-generic-path
[s51]: STATUS_AND_PROFILE.md#51-pipeline
[s54]: STATUS_AND_PROFILE.md#54-fixes-found-by-the-family-work
[s55]: STATUS_AND_PROFILE.md#55-known-limitations
