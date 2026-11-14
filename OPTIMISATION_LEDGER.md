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
| [spl-receiver-order](#spl-receiver-order) | Receiver / sublattice ordering in `spline` and `w'` for L1/L2 slab locality | – | spec | NOT BUILT | – | untried (`SrTiO3` O sublattices interleaved; Morton off < L2) | `OMAT0_SPEEDUP_BRIEF.md` |
| [spl-r6-r8](#spl-r6-r8) | Unpaired `R=6` or `R=8` spline blocks if register pressure drops <= 80 | – | spec | NOT BUILT | – | untried (`R=5` at 96 regs already won `SrTiO3`) | `OMAT0_SPEEDUP_BRIEF.md` |
| [spl-v6-cubic-cg-fold](#spl-v6-cubic-cg-fold) | `w'` cubic decode (`d_cubic_packed8_1`) + staged per-edge CG linear forms in `s_sh` (`ED = 36`, `TILE = 16`) | `_emit_spline_forward1` | spec | DEFAULT (5060 Ti only) | S15 | 5060 Ti `SrTiO3` 8.40 -> 8.28 ms (`0.985`, `R1p` 1,403 -> 1,306 us), `NbCl5` 1.863 -> 1.853 (`0.994`); all `LOUP_SPLINE_VARIANT` gates pass | this file |

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
| [wp-decode-bias-fold](#wp-decode-bias-fold) | Fold the four `- hb` subtractions of the cubic decode into the polynomials | `LOUP_GEN_DECODE_FOLD` (generator, on) | spec | DEFAULT (13 Nov) | – | 5060 Ti `SrTiO3` 0.9929 / 0.9948 / 0.9925, `NbCl5` 0.9951 / 0.9938 / 0.9965; Ada tie; -24 / -16 instructions | this file |

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
| [i0-bwd-rcv-warp-agg](#i0-bwd-rcv-warp-agg) | Warp-aggregate receiver `atomicAdd`s in `tabled_outer_sum0_backward_kernel` | – | spec | NOT BUILT | – | untried (1-2 receivers/warp -> ~2x fewer force atomics) | `OMAT0_SPEEDUP_BRIEF.md` |

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
| [gemm-own-n144](#gemm-own-n144) | `sgemm128_kernel<4, false>` partial-`N` `128 x 64` tiles when `(N % 128) <= 64` (e.g. `N = 144`) | `loup::sgemm128` | both | DEFAULT (5060 Ti only) | S16 | replaces `magma_sgemmEx` on `N = 144` (`POLAR-1-M` `NbCl5` `170.7 -> 76.6 us` in `prof_k`); tie on `M >= 1944` MD | this file |
| [gemm-own-sgemm64](#gemm-own-sgemm64) | Own `64 x 64 x 16` `cp.async` SIMT SGEMM (`128` threads) scoped to `M <= 1600` | `loup::sgemm64`, `DiskPlan::use_own == 2` | both (`M <= 1600`) | DEFAULT (5060 Ti only) | S16 | fresh-plan ABBA: `POLAR-1-S` `NaCl216` `1.85 -> 1.61 ms` (`0.873`), `POLAR-1-M` `NaCl216` `4.30 -> 3.61` (`0.839`), `POLAR-1-L` `NaCl216` `8.24 -> 7.22` (`0.876`), `MH-1` `NaCl216` `1.71 -> 1.57` (`0.919`), `OFF23-M` `water216` `1.39 -> 1.30` (`0.938`), `MH-1` `NbCl5` tie (`1.0002`) | this file |
| [gemm-own-sgemm32](#gemm-own-sgemm32) | Own `32 x 32 x 16` `cp.async` SIMT SGEMM (`64` threads) + deterministic split-K (`1, 2, 4`) into per-stream `4 MB` workspace, scoped to `sm_120` and `M <= 1600` | `loup::sgemm32`, `DiskPlan::use_own` in `3..5` | both (`sm_120`, `M <= 1600`) | DEFAULT (5060 Ti only) | S17 | fresh-plan ABBA: `POLAR-1-S` `NaCl216` `1.65 -> 1.50 ms` (`0.909`), `POLAR-1-M` `NaCl216` `3.70 -> 3.46` (`0.936`), `POLAR-1-L` `NaCl216` `7.28 -> 6.89` (`0.947`), `MH-1` `NaCl216` `1.61 -> 1.53` (`0.948`), `MH-1` `NbCl5` tie (`1.003`) | this file |
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
| [gemm-cold-l2-ada-fix](#gemm-cold-l2-ada-fix) | Fix Ada `22 MiB < L2/2` cold-L2 rule for `lin1_fwd l=1` + wider 3xTF32 `mma` tiles | – | both | NOT BUILT | – | untried (`lin1_fwd l=1` timed hot on Ada despite 97 MB `m1`) | `OMAT0_SPEEDUP_BRIEF.md` |

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
| [mem-l2-persist-x](#mem-l2-persist-x) | CUDA Graph `AccessPolicyWindow` (~10-12 MiB) on `x0`+`x1` in `spline`/`w'` | – | spec | NOT BUILT | – | untried (`x0+x1` = 9.77 MiB on `SrTiO3`; check `ncu` hit rate first) | `OMAT0_SPEEDUP_BRIEF.md` |
| [mem-l1-carveout-retune](#mem-l1-carveout-retune) | Re-check `tune_l1` shared/L1 carveouts on `w'` and `i0_bwd` after items 23-31 | `tune_l1` | spec | NOT BUILT | – | untried (carveouts tuned before later kernel rewrites) | `OMAT0_SPEEDUP_BRIEF.md` |

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
| [comp-active-edge-tighten](#comp-active-edge-tighten) | Tighten active-edge compaction (`compute_and_count_active` + `scatter_active_edges`) | – | both | NOT BUILT | – | untried (101 us/step on `SrTiO3` 5060 Ti, `tmp_vec_len` round-trip) | `OMAT0_SPEEDUP_BRIEF.md` |

### Generic path (other model families, 12-13 November 2026)

| ID | Idea | Switch/flag | Path | Status | Stage | Effect | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [gen-probe-export](#gen-probe-export) | Extract every linear map by probing the float64 modules; fold constants numerically | `export_generic.py` | gen | DEFAULT | S6 | design: one exporter for every family | [§5.1][s51] |
| [gen-pair-sym](#gen-pair-sym) | Symmetric pair tables for blocks that do not embed sender and receiver separately | bundle `pair_sym` | gen | DEFAULT | S6 | S(S+1)/2 instead of S^2 tables; size not recorded | [§5.1][s51] |
| [gen-first-layer](#gen-first-layer) | Layer 0 as an outer sum over the radial chain's columns | `first_fwd_kernel`, `first_bwd_kernel`, bundle `first_hidden` | gen | DEFAULT | S6 | 66-column table instead of 4C + 1; OMAT-0 small bundle 40 MB | [§5.1][s51] |
| [gen-factored-products](#gen-factored-products) | FastMACE's factored symmetric contraction, kernels by program signature | `gprod_*` | gen | DEFAULT | S6 | design | [§5.1][s51] |
| [gen-certified-spline](#gen-certified-spline) | FastMACE's certified cubic-Hermite radial spline per pair | `spline.fit` | gen | DEFAULT | S6 | export refuses certificate > 1 | [§5.1][s51] |
| [gen-profile](#gen-profile) | nsys profile of MH-1 `SrTiO3` | – | gen | MEASUREMENT | S6 | SGEMMs ~50%, conv 20%, gate 15% | [§5.5][s55] |
| [gen-fork-zero-slab](#gen-fork-zero-slab) | GEMM fork / join lanes in `GenericModel` and `PolarLR`, one zero slab per pass, dead gate-backward stores cut | `GenericPassInputs::gemm_fork/lane/join` | gen + POLAR | DEFAULT (5060 Ti only) | S7 | 5060 Ti ABBA b/a 0.914-0.990 (all families) | this file |
| [gen-cublas-stream](#gen-cublas-stream) | `cublasSgemm` fallback ignored the GEMM lane's stream | `sgemm_rm` | both | FIX | S7 | POLAR-1-M gate ratios 1e4-4e5 -> PASS | this file |
| [gen-skip-fold](#gen-skip-fold) | `W_skip_msg` folded into `W_lin` / `C0` per species; last layer's `W_p` / `W_sc` folded into `ro_W1` | `GenericLayer::skip_folded`, `ro_folded` | gen | DEFAULT (5060 Ti only) | S7 | ABBA b/a 0.939-0.991 (OMAT-S, OFF, MH-1) | this file |
| [gen-conv-v2](#gen-conv-v2) | Conv kernels with 8-byte cubic tables, staged edges, R-receiver forward, late pull-back backward; warp-staged first layer | `python/gen_generic.py`, `first_fwd/bwd_kernel` | gen | DEFAULT (5060 Ti only) | S7 | ABBA b/a 0.688 (OMAT-S `SrTiO3`) to 0.987 (POLAR-1-S) | this file |
| [gen-skinny-gemm](#gen-skinny-gemm) | Own kernels for products with N <= 16 or K <= 16 (warp K-split, thread per output) in place of cuBLAS | – | gen, POLAR, OMAT-M | REJECTED | S7 | ABBA b/a 1.040 (OMAT-S `SrTiO3`), 1.009 (POLAR-1-M `NbCl5`), 1.016 (OMAT-M `SrTiO3`) | this file |
| [gen-first-bwd-tail](#gen-first-bwd-tail) | First-layer backward: the per-edge `sh16_grad` tail batched one edge per lane | – | gen | REJECTED | S7 | ABBA b/a 1.040 (OMAT-S `SrTiO3`), 1.017 (OFF23-M water) | this file |
| [gen-conv-l2-split](#gen-conv-l2-split) | lx 2 conv backward split: the 10 paths with x degree <= 1 in the late pull-back form, the 7 with x degree 2 kept explicit | `gconv_bwd_L2lp` + `gconv_bwd_L2` | gen (OFF23-L, POLAR-1-L) | DEFAULT (5060 Ti only) | S8 | ABBA b/a 0.925 (OFF23-L water 2x2x2) | this file |
| [gen-first-fwd-race](#gen-first-fwd-race) | `first_fwd_kernel`: species-segment end in two parity slots | – | gen | FIX | S8 | no timing change expected; forces at noise level | this file |
| [gen-first-bwd-v2](#gen-first-bwd-v2) | First-layer backward: receiver's `G` rows staged once in swizzled shared memory; one edge per half-warp | `first_bwd_kernel<STAGE_G>` | gen | DEFAULT (5060 Ti only) | S8 | ABBA b/a 0.948 (OMAT-S `SrTiO3`), 0.982 (OFF23-M water), 0.986 (OMAT-S `NbCl5`), 0.996 (MH-1) | this file |
| [gen-conv-fwd-r1](#gen-conv-fwd-r1) | Conv forward at C = 128 with one receiver per block (was 4 for lx 0, 2 for lx 1) | `launch_conv_fwd` | gen | DEFAULT (5060 Ti only) | S8 | ABBA b/a 0.949-0.989 | this file |
| [gen-conv-fwd-l2-split](#gen-conv-fwd-l2-split) | lx 2 conv forward as two kernels by x degree | `gconv_fwd_L2a` / `L2b` | gen (OFF23-L, POLAR-1-L) | DEFAULT (5060 Ti only) | S8 | OFF23-L water forward 7.6 -> 3.1 + 3.1 ms | this file |
| [gen-conv-bwd-128](#gen-conv-bwd-128) | Conv backward (lx 0 / 1) with 128-thread blocks | – | gen | REJECTED | S8 | ABBA b/a 0.997 (OMAT-S), 0.992 (OFF23-M): below resolution, halves per-receiver warps on small cells | this file |
| [gen-conv-fwd-l1-split](#gen-conv-fwd-l1-split) | lx 1 conv forward as two kernels by x degree | – | gen | REJECTED | S8 | 1.70 -> 1.78-1.87 ms (x degree 1 kernel alone at 128 registers) | this file |
| [rv8-fixes](#rv8-fixes) | Eighth review pass: packed-table error check at load, cuBLAS workspace per lane, reorder carries Langevin noise, MD needs a state, safe edge growth, band / plan / input checks, unstable gate row opt-in | `kPackedTableMaxRelErr`, `cublas_bind`, `--include-unstable` | all | FIX | S9 | accuracy and robustness, no speed change intended | this file |
| [gen-zbl-in-assembly](#gen-zbl-in-assembly) | ZBL computed in the force-assembly pass (float32, skip beyond the largest ZBL cutoff) instead of its own pass | `assemble_forces_kernel(ZblParams)` | gen (ZBL models: OMAT-S, MH-1) | DEFAULT (5060 Ti only) | S10 | OMAT-S `SrTiO3`: 389 -> 221 us per call | this file |
| [gen-first-bwd-assembly](#gen-first-bwd-assembly) | Force / virial assembly (and ZBL) folded into the first-layer backward | – | gen | REJECTED | S10 | 990-1,029 -> 1,113-2,243 us | this file |
| [gen-first-hidden-k0-align](#gen-first-hidden-k0-align) | Layer 0 `first_hidden` `K` alignment (drop zero bias col 64, pad nonzero `K` to 16) to unlock `sgemm128` + `3xTF32` | `GenericLayer::K0` | gen (`first_hidden`) | DEFAULT (5060 Ti only) | S11 | OMAT-S `SrTiO3` 5.88 -> 5.62 ms/step (0.956); OFF23-M water 8.85 -> 8.67 (0.979); OMAT-M gen 10.56 -> 10.30 (0.976) | this file |
| [gen-conv-bwd-lazy-q](#gen-conv-bwd-lazy-q) | Lazy `s_q` loads + register-cap launch bounds on `gconv_bwd_L0/L1/L2` | – | gen | REJECTED | S11 | `gconv_bwd_L1`: 2.67-2.78 -> 2.88-2.95 ms (`STACK:64` spill, or stalled `LDS.128` at 128 regs); `gconv_bwd_L0` 512 threads 1.26 -> 1.31 ms | this file |
| [gen-first-bwd-edge](#gen-first-bwd-edge) | Edge-parallel layer-0 backward + ZBL + force/virial assembly (`tabled_outer_sum0_backward_kernel` design) | `first_bwd_edge_kernel` | gen (`first_hidden`, `J - 1 <= 96`) | DEFAULT (5060 Ti only) | S12 | OMAT-S `SrTiO3` 5.62 -> 5.13 ms/step (0.913); OFF23-M water 8.65 -> 8.28 (0.957); OMAT-M gen 10.30 -> 9.80 (0.951); MH-1 0.990, POLAR-1-M 0.995 | this file |
| [gen-gprod-ro1-fused](#gen-gprod-ro1-fused) | Fused product fwd + linear/bias/silu/linear readout (`ro_H = 16`) + product bwd (`sc1_fused` generalisation) + 128-thread `gprod_fwd`/`bwd` | `gprod_ro1_fused_<sig>`, `launch_prod` | gen (`ro_folded`, `readout == 1`, `ro_H == 16`; `C % 128 == 0`) | DEFAULT (5060 Ti only) | S13 | OMAT-S `SrTiO3` 5.12 -> 4.99 ms/step (0.976); OFF23-M water 8.29 -> 8.16 (0.984); OMAT-M gen 9.81 -> 9.67 (0.986); POLAR-1-M `NbCl5` 27.28 -> 27.14 (0.995) | this file |
| [gen-gconv-cfix-preloc](#gen-gconv-cfix-preloc) | `C_FIX` (`96/128/224`) template on `gconv_bwd_L0/L1` (`gconv_bwd_L1<true,128>`: `REG:128 -> 62`) + prelocated `act_slab`/`act_t_sx` from `scatter_active_edges_kernel` (`pair_sym && C_x <= 256`) + 256 thr on `L1` | `C_FIX`, `PRELOC`, `preloc_grid()` | gen (`C = 96/128/224`; preloc `pair_sym && C_x <= 256`) | DEFAULT (5060 Ti only) | S14 | OMAT-S `SrTiO3` 4.91 -> 4.73 ms/step (0.964); OFF23-M water 8.00 -> 7.43 (0.928); MH-1 `NbCl5` 12.82 -> 12.50 (0.976); POLAR-1-M 27.20 -> 27.06 (0.995); OMAT-M gen 9.67 -> 8.91 (0.921, `1.051x` spec) | this file |
| [gen-conv-fwd-cg-fold](#gen-conv-fwd-cg-fold) | Staged per-edge CG linear forms in `s_sh` (`ED = 16/36/72`) + per-edge `s_pw` polynomial scalars in `gconv_fwd` | `emit_conv_fwd` | gen | DEFAULT (5060 Ti only) | S15 | `gconv_fwd_L1` `1,688 -> 1,572 us` (`OFF23-M`), `L2b` `3,029 -> 2,898 us` (`OFF23-L`); ABBA OMAT-S `SrTiO3` `0.989` (`4.82 -> 4.77 ms`), `OFF23-L` `0.989` (`28.30 -> 28.00`), `OFF23-M` tie (`0.997`), `POLAR-1-M` tie (`0.999`), `MH-1` `0.981` inconclusive (`1.682/1.746` spread) | this file |
| [gen-conv-bwd-128-regq](#gen-conv-bwd-128-regq) | `128`-thread register-`Q` edge-staged `gconv_bwd_L0/L1_128` (`PAIR = false` / `true`) | – | gen | REJECTED | S16 | `PAIR=false`: `L1` `2,216 -> 2,147 us`, MD `OFF23-M` tie (`0.999`), `OMAT-0-S` `0.997`; `PAIR=true` (`REG:128`): MD `+0.4%`/`+0.8%` | this file |
| [gen-conv-l1-carve0](#gen-conv-l1-carve0) | `PreferredSharedMemoryCarveout = 0` on generic conv & `first_bwd_edge` kernels | – | gen | REJECTED | S16 | capped `gconv_fwd_L1` to `8.19 KB` & `gconv_bwd_L1` to `32.77 KB` (`1` blk/SM); `OFF23-M` `7,372 -> 11,592 us/eval` | this file |
| [gen-conv-fwd-l1-r4](#gen-conv-fwd-l1-r4) | `gconv_fwd_L1<4, true, 512, ..., 128, 16>` (`4` receivers/block, `TILE = 16`) | – | gen | REJECTED | S16 | `OMAT-0-M` generic `SrTiO3` `8.79 -> 8.66 ms` (`0.985`), `OFF23-M` water `7.44 -> 7.54` (`1.013`, `REG:124`) | this file |

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
| [pol-sf-f32](#pol-sf-f32) | Structure factors: float32 chunk sums folded into double, coalesced `float4` loads | `sf_kernel<NC>` | POLAR | DEFAULT (5060 Ti only) | S7 | ABBA b/a 0.833 (M `NbCl5`) to 0.987 (L NaCl) | this file |
| [pol-contig-weights](#pol-contig-weights) | Per-degree contiguous `Wpot` / `Wq` / `le_W*` slices, fused `qnew = qin + out` | – | POLAR | REJECTED | S7 | ABBA b/a 0.952-1.045, mixed | this file |
| [pol-kpass-32](#pol-kpass-32) | 32-thread blocks for the per-atom k-space passes when N <= 512 | – | POLAR | REJECTED | S7 | NaCl 216 (with pol-contig-weights): +7.4% (S), +6.0% (M) | this file |
| [pol-k-split](#pol-k-split) | Per-atom k passes split over k chunks (`blockIdx.y`), chunk partials reduced in fixed order | `k_chunk`, `k_reduce_kernel` | POLAR | DEFAULT (5060 Ti only) | S8 | ABBA b/a 0.876 (S NaCl 1,728) to 0.978 (L NaCl 216); M `NbCl5` 0.928 | this file |
| [pol-2d3d-nodiv](#pol-2d3d-nodiv) | 2D/3D div-free grids for `add_bias`/`bcast_mul`/`inv` + fused `inv2_kernel` | – | POLAR | REJECTED | S15 | `POLAR-1-M` `NbCl5` quick ABBA `b/a = 1.0004` (27.06 -> 27.07 ms; DRAM-BW bound) | this file |
| [pol-sf-tiled](#pol-sf-tiled) | Multi-`k`-vector warp tiling (`8` warps = `8` `k`-vectors) sharing SoA `s_X[NC][129]` + `s_frac[3][129]` (`TILE = 128`, `float4` `X` loads) | `sf_kernel_tiled<NC, TK>`, `launch_sf` | POLAR | DEFAULT (5060 Ti only) | S15 | `sf_kernel<16/8/4>` 900/455/233 -> 316/150/84 us; ABBA (fresh plans) `POLAR-1-M` `NbCl5` 27.00 -> 25.00 ms (`0.926`), `NaCl` 4.41 -> 4.36 (`0.990`), `S` `NaCl` 1.92 -> 1.86 (`0.966`), `L` `NaCl` 8.36 -> 8.31 (`0.993`) | this file |

### Measurements without a switch

| ID | What | Status | Result | Evidence |
| --- | --- | --- | --- | --- |
| [meas-stage-spans](#meas-stage-spans) | Stage spans and top kernels, both GPUs | MEASUREMENT | `w'` 36%, spline 23% of the `SrTiO3` step (5060 Ti) | [§2][s2] |
| [meas-ncu](#meas-ncu) | Nsight Compute facts, 5060 Ti | MEASUREMENT | `w'` issue-bound (80% issue slots); spline L2-bound (83%) | [§2][s2] |
| [meas-size-scaling](#meas-size-scaling) | System-size scaling, `SrTiO3` 135-40,000 atoms | MEASUREMENT | linear above ~1,000 atoms: 1.66-1.72 us/atom-step | [§1][s1] |
| [meas-nve-drift](#meas-nve-drift) | NVE drift, `SrTiO3`, 1 ps | MEASUREMENT | -0.00011 / -0.00012 meV/atom/ps | [§1][s1] |
| [meas-ada-counters](#meas-ada-counters) | Ada performance counters | BLOCKED | `ERR_NVGPUCTRPERM`, needs admin | [§4.1 #2][s41] |
| [meas-family-bench](#meas-family-bench) | MD speed of every family, both GPUs, against FastMACE's documented numbers | MEASUREMENT | POLAR-1 level to 20% slower than FastMACE (11 Nov lines) | [§5][s5] |
| [meas-plan-variance](#meas-plan-variance) | GEMM plans differ between library builds | MEASUREMENT | POLAR-1-M NaCl 216 S8/S7 b/a 1.013 with each build's cached plans, 0.969 after both retuned | this file |

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
| Receiver / sublattice ordering in `spline` and `w'` | untried: group receivers by species / oxygen sublattice / spatial cell so co-resident SMs share table slabs (`Morton` is off at `N=5,000`) | [spl-receiver-order](#spl-receiver-order) |
| Unpaired `R=6` or `R=8` spline blocks | untried: check if trimming `acc[40]`/`she[16]` register pressure fits <= 80 regs | [spl-r6-r8](#spl-r6-r8) |
| CUDA Graph L2 persistence (`AccessPolicyWindow`) on `x0+x1` | untried: ~10-12 MiB persistent window (`SrTiO3` `x0+x1` = 9.77 MiB), check `ncu` eviction first | [mem-l2-persist-x](#mem-l2-persist-x) |
| Cold-L2 rule for Ada `lin1_fwd l=1` + wider 3xTF32 `mma` tiles | untried: `22.0 MiB < 24 MiB` (`L2/2`) times `lin1_fwd l=1` hot on Ada; check `LOUP_DEBUG_GEMM=1` and `128x64`/`64x128` 3xTF32 tiles on 5060 Ti | [gemm-cold-l2-ada-fix](#gemm-cold-l2-ada-fix) |
| Warp-aggregate `rcv` `atomicAdd`s in `i0_bwd` | untried: 1-2 receivers per 32-edge warp (`SrTiO3`); check `ncu` atomic stall counters first | [i0-bwd-rcv-warp-agg](#i0-bwd-rcv-warp-agg) |
| Active-edge compaction (`101 us`) & `tune_l1` re-check | untried: `compute_and_count_active` + `scatter_active_edges` (`tmp_vec_len` round-trip); re-check `w'`/`i0_bwd` carveouts | [comp-active-edge-tighten](#comp-active-edge-tighten), [mem-l1-carveout-retune](#mem-l1-carveout-retune) |
| Gate reference cache | done (tooling): `python/test_gate.py` caches the MACE float64 / float32 references and PolarMACE's finite-difference stress, keyed by model sha256, head, set, reference device (GPU model, compute capability, CUDA version; a reference from another GPU is not reused), MACE / torch versions, stress step and every configuration's atoms; `--no-ref-cache` recomputes, an unreadable entry is recomputed. POLAR-1-S crystals gate 40 -> 6 s on a repeat; the float32 bound is then fixed between runs. Test: `tests/test_gate_cache.py` | `python/test_gate.py` |
| GEMM plans timed in the captured graph | not built: isolated timings mis-rank plans under concurrent lanes ([meas-plan-variance](#meas-plan-variance): up to ~4% on POLAR NaCl 216); flip near-tie shapes one at a time, recapture, time replays | this file |

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

<a id="spl-receiver-order"></a>
#### spl-receiver-order: receiver / sublattice ordering for slab locality
- Status: NOT BUILT.
- What: at `N = 5,000`, `mem-reorder` is off (`N * 4 KB < L2`). In `SrTiO3`, the 3,000 oxygen atoms sit in three interleaved sublattices `(0.5,0.5,0)`, `(0.5,0,0.5)`, `(0,0.5,0.5)` in caller order, so consecutive receivers in `R=4/5` blocks mix sublattices. Ordering receivers (or block walk order) by species/sublattice or spatial cell keeps co-resident SMs on the same `(pair, r)` slabs (cf. FastMACE `BLOCK_ORDER`). Do **not** replace the `(sym_pair, r)` edge sort with `sender_idx` (that breaks `PAIR` and slab reuse).
- Sources: `OMAT0_SPEEDUP_BRIEF.md`.

<a id="spl-r6-r8"></a>
#### spl-r6-r8: unpaired `R=6` / `R=8` spline blocks
- Status: NOT BUILT.
- What: `R=5` unpaired (96 regs, 640 threads = 20 warps/block) already won `SrTiO3` on both GPUs. Check whether trimming `acc[40]` / `she[16]` register pressure lets `R=6` or `R=8` fit at <= 80 registers for higher occupancy and L1 slab sharing.
- Sources: `OMAT0_SPEEDUP_BRIEF.md`.

<a id="spl-v6-cubic-cg-fold"></a>
#### spl-v6-cubic-cg-fold: `w'` cubic decode and staged CG linear forms in `spline_forward_inter1`
- Stage: S15 (14 November 2026).
- What: switched `spline_forward_inter1` (`python/gen_kernels.py`, `src/engine.cu`) from the 3-coefficient `d_packed8_table_1` + unfolded decode to `d_cubic_packed8_1` (`cubic_bias_shift`) with the `w'` bias-folded cubic decode and warp-staged `s_pw` (`float4`) polynomial factors; grouped per-edge Clebsch-Gordan `CONST * she[m]` linear forms into 18 shared-memory slots (`ED = 36`) computed once per edge in warp 1, applied accumulator scales at store, and narrowed `TILE` from 32 to 16 so `R = 5` stays at `REG:96, STACK:0, SHARED:14784` inside the 16 KB carveout.
- Status: DEFAULT (5060 Ti only).
- Measured (5060 Ti): frozen `SrTiO3` (`R1p`) `1,403.4 -> 1,305.9 us`, all kernels `8,075 -> 7,860 us/eval`; ABBA vs `c24b6c4`: `SrTiO3` `8.4024 -> 8.2752 ms/step` (`0.9848`; with `LOUP_DEBUG_SPLINE`: `8.4259 -> 8.2714`, `0.9817`), `NbCl5` `1.8630 -> 1.8525` (`0.9944`). `mace-omat-0-medium` oxide gate passes on default plus every forced `LOUP_SPLINE_VARIANT` (`R1, R1p, R4, R4p, R5`).
- Sources: this file; `.internal/research/2026-11-14-omat0-medium-speedup-candidates.md` (v6); commit `ca23b3c`.

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
- Switch: `LOUP_GEN_DECODE_FOLD` when running `python/gen_kernels.py` (on by default since 13 November; `=0`
  regenerates the unfolded form). It changes both `w'` kernels.
- What: each coefficient is decoded as `raw_k - hb` with `hb = 1.5 * 2^e`. Substituting into `w` and `radial_dt`
  gives `w = raw0 + s1 raw1 + s2 raw2 + s3 raw3 - 1.5 (1 + s1 + s2 + s3) 2^e` (and likewise the slope), with the two
  bias factors computed once per edge: 4 FADD + 1 LOP3 per path become 2 FFMA. External suggestion, 13 November.
- Status: DEFAULT since 13 November. A win on the 5060 Ti in all three runs, a tie on the Ada, the gate passes on
  both: the adoption bar of `OMAT0_SPEEDUP_BRIEF.md`.
- Measured (13 November, libraries built from the same tree, differing only in this switch; `bench_md.py`, order
  a b b a, twice per GPU):
  - SASS, `wprime_backward_inter1_rcv`: `sm_120` 1,000 -> 976 instructions and the 8-byte stack spill disappears;
    `sm_89` 1,016 -> 1,000; 64 registers on both, before and after.
  - RTX 5060 Ti: `SrTiO3` 8.241 -> 8.183 ms (0.9929) and 8.308 -> 8.265 ms (0.9948); `NbCl5` 1.799 -> 1.790 (0.9951)
    and 1.807 -> 1.796 (0.9938). In all four neighbouring `SrTiO3` pairs the folded side is faster (0.3 to 0.9%).
    A third run after the final rebuild: `SrTiO3` 8.285 -> 8.223 ms (0.9925), `NbCl5` 1.803 -> 1.797 (0.9965).
  - RTX 4000 Ada: `SrTiO3` 9.316 -> 9.331 ms (1.0015) and 9.455 -> 9.413 ms (0.9955); `NbCl5` 1.0002 and 0.9925: a
    tie.
  - Accuracy: the default gate passes all 30 rows on both GPUs. On the 5060 Ti, two gate runs per library: mean
    stress ratio 1.385 / 1.187 unfolded against 1.080 / 1.210 folded (max 2.36 / 2.32 against 1.77 / 1.80); mean
    force ratio 0.702 / 0.670 against 0.698 / 0.696. The gate's ratios vary by about 15% from run to run, more than
    any difference between the two. (A single run each on the Ada, 0.764 against 0.940 for stress, had looked like
    a cost; it is inside that spread.)
  - Ada: 46 tests pass with the folded library (twice).
- Verified with the fold on: 5060 Ti 47 tests and the full gate for all eight model sets (78 rows, worst force
  ratio 1.45, stress 2.27); Ada 46 tests (OMAT-0 small skipped) and the default gate. The full family gate was not
  rerun on the Ada with the fold (the workstations were closed for a power outage); the fold touches only the
  OMAT-0 medium kernels, which the default gate covers.
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

<a id="i0-bwd-rcv-warp-agg"></a>
#### i0-bwd-rcv-warp-agg: warp-aggregated `rcv` force atomics in `i0_bwd`
- Status: NOT BUILT.
- What: `tabled_outer_sum0_backward_kernel` (`src/interaction0.cu:421-426`) performs 3 `double atomicAdd`s for `rcv` and 3 for `snd` on every active edge. Because active edges are receiver-sorted (~75 edges/atom on `SrTiO3`), a 32-edge warp spans 1-2 receivers; warp-aggregating `rcv` contributions before `atomicAdd` roughly halves `forces_f64` atomic traffic. Falsifier: check `ncu` atomic stall counters first; stop if no material stall or < 0.5% MD gain.
- Sources: `OMAT0_SPEEDUP_BRIEF.md`.

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

<a id="gemm-own-n144"></a>
#### gemm-own-n144: partial-`N` `128 x 64` tiles in `loup::sgemm128`
- Stage: S16 (14 November 2026).
- What: extended `loup::sgemm128` (`src/sgemm.cu`) to launch `sgemm128_kernel<4, false>` (`128 x 64` tiles, `REG:105, STACK:0`) whenever `(N % 128) != 0 && (N % 128) <= 64` (such as `N = 144` padded `K0` layers), avoiding nearly half-empty `128 x 128` tiles.
- Status: DEFAULT (5060 Ti only).
- Measured (5060 Ti): in `POLAR-1-M` `NbCl5` `prof_k`, partial-`N` `sgemm128` cut that shape from `170.7` to `76.6 us`. (A paired `L2/4` cold-L2 threshold test in `d906545` was reverted in `0a240ec` after mixed `OMAT-0-S` ordering.)
- Sources: this file; commits `d906545`, `0a240ec`.

<a id="gemm-own-sgemm64"></a>
#### gemm-own-sgemm64: `64 x 64 x 16` `cp.async` SIMT SGEMM for `M <= 1600`
- Switch: `loup::sgemm64` (`sgemm64_kernel<FULL_N>` in `src/sgemm.cu`), autotuned when `is_sm120 && M <= 1600`, stored in `DiskPlan::use_own == 2`.
- Stage: S16 (14 November 2026).
- What: on small cells (`N = 216..648`, `M <= 1512`), `128 x 128` tiles launched only 2–12 blocks on 36 SMs and cuBLASLt fell back to `magma_sgemmEx_kernel` (e.g. `124 us` on `POLAR-1-S` `M = 216, N = 144, K = 2048`). `sgemm64_kernel` uses `128` threads (`8 x 4` outputs/thread, `REG:103, STACK:0, SHARED:25600`), giving `4x` more threadblocks across SMs. Unscoped `sgemm64` won hot-L2 autotuning on `M = 1944..5832` (`nbcl5`) where cold `A` streams hurt (`+0.68%` `POLAR-1-M`, `+0.25%` `MH-1`), and Ada was not tested, so it is scoped to `sm_120` and `M <= 1600`.
- Status: DEFAULT (5060 Ti only).
- Measured (5060 Ti, fresh plans): `POLAR-1-S` `NaCl216` `1.8458 -> 1.6105 ms` (`b/a = 0.8726`), `POLAR-1-M` `NaCl216` `4.2996 -> 3.6069` (`0.8389`), `POLAR-1-L` `NaCl216` `8.2416 -> 7.2164` (`0.8756`), `MH-1` `NaCl216` `1.7089 -> 1.5700` (`0.9187`), `OFF23-M` `water216` `1.3912 -> 1.3047` (`0.9378`), `MH-1` `NbCl5` tie (`12.5860 -> 12.5888`, `1.0002`). Gates pass on `POLAR-1-S/M`, `MH-1` (crystals + organic), `OMAT-0-S/M`, `OFF23-M`; `POLAR-1-L` `cmp_libs` `dF = 2.07e-7` (`nacl216`); pytest suite passes.
- Sources: this file.

<a id="gemm-own-sgemm32"></a>
#### gemm-own-sgemm32: `32 x 32 x 16` `cp.async` SIMT SGEMM + deterministic split-K for `sm_120` `M <= 1600`
- Switch: `loup::sgemm32` (`sgemm32_kernel<FULL_N>` + `splitk_reduce_kernel` in `src/sgemm.cu`), autotuned for `split_k` in `{1, 2, 4}` when `is_sm120 && M <= 1600` (with `K >= 64 * sk` and `sk * M * N * 4 <= cublas_ws_bytes`), stored in `DiskPlan::use_own` in `{3, 4, 5}`.
- Stage: S17 (14 November 2026).
- What: even with `sgemm64`, `M = 216, N = 8..256` GEMMs in `POLAR-1-S/M/L` and `MH-1` `NaCl216` (`63–71%` of GPU kernel time) launched only `4..16` blocks on 36 SMs (`4` `M`-tiles x `1..4` `N`-tiles), leaving 55–89% of SMs idle (`M=216, N=144, K=2048` took `53.2 us`; `N=16, K=512` took `14.7 us`). `32 x 32` tiles + deterministic split-K (`ws_p` per-stream 4 MB workspace + in-order `splitk_reduce_kernel`, zero atomics/memsets) cut those to `20.8 us` / `7.6 us`.
- Status: DEFAULT (5060 Ti only).
- Measured (5060 Ti, fresh plans): `POLAR-1-S` `NaCl216` `1.6451 -> 1.4958 ms` (`b/a = 0.9092`), `POLAR-1-M` `NaCl216` `3.6966 -> 3.4591` (`0.9357`), `POLAR-1-L` `NaCl216` `7.2813 -> 6.8949` (`0.9469`), `MH-1` `NaCl216` `1.6097 -> 1.5264` (`0.9483`), `OFF23-M` `water216` tie (`1.3436 -> 1.3520`, `1.0063`), `MH-1` `NbCl5` tie (`12.5460 -> 12.5867`, `1.0032`). Cached 3x gates pass on `POLAR-1-S/M`, `MH-1` (crystals + organic), `OMAT-0-S/M`, `OFF23-M`; `cmp_libs` `dF <= 3.1e-6`; pytest suite passes (`67 passed`).
- Sources: this file.
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

<a id="gemm-cold-l2-ada-fix"></a>
#### gemm-cold-l2-ada-fix: cold-L2 classification on Ada `lin1_fwd l=1` + wider 3xTF32 `mma` tiles
- Status: NOT BUILT.
- What: on Ada (`l2_bytes = 48 MiB`), `sgemm_rm`'s `M * K * 4 > l2_bytes / 2` threshold (`24 MiB`) times `lin1_fwd l=1` (`15000 * 384 * 4 = 22.0 MiB`) **hot**, even though `m1_0..3` total 97 MB and stream from DRAM; also check `lin1_bwd` (`db_l` vs `m1_l`) and `LOUP_DEBUG_GEMM=1` picks on both GPUs. Separately, `HybCubin` only provides `64x64`, `32x64`, `64x32` tiles (beaten by `sgemm128` on 5060 Ti `lin1`); test `128x64` / `64x128` 3xTF32 `mma` tiles.
- Sources: `OMAT0_SPEEDUP_BRIEF.md`.

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

<a id="mem-l2-persist-x"></a>
#### mem-l2-persist-x: CUDA Graph L2 persistence (`AccessPolicyWindow`) for `x0 + x1`
- Status: NOT BUILT.
- What: attach `cudaKernelNodeAttributeAccessPolicyWindow` to the `spline_forward_inter1` and `wprime_backward_inter1_rcv` graph nodes for `x0 + x1` (`9.77 MiB` on `SrTiO3`), with `cudaLimitPersistingL2CacheSize` around `10-12 MiB` (leaving >= 20 MiB on 5060 Ti for table slabs). Gate on checking `ncu` L2 hit rate on `x0`/`x1` first (`spline` writes `m1` after the edge loop, not during it, so `x0`/`x1` may already stay L2-resident).
- Sources: `OMAT0_SPEEDUP_BRIEF.md`.

<a id="mem-l1-carveout-retune"></a>
#### mem-l1-carveout-retune: re-check `tune_l1` carveouts after items 23-31
- Status: NOT BUILT.
- What: `tune_l1` shared/L1 carveouts in `src/engine.cu` were measured before items 23-31 (`i0_bwd` staging, `w'` `s_q` / `wp_threads`); re-checking `w'` and `i0_bwd` carveouts is cheap.
- Sources: `OMAT0_SPEEDUP_BRIEF.md`.

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

<a id="comp-active-edge-tighten"></a>
#### comp-active-edge-tighten: active-edge compaction pass
- Status: NOT BUILT.
- What: `compute_and_count_active` (47 us) + `DeviceScan::ExclusiveSum` + `scatter_active_edges` (54 us) take ~101 us/step on 5060 Ti `SrTiO3`, writing and re-reading `d_tmp_vec_len`. Note single-pass decoupled look-back compaction (`edge-compaction-1pass`) was already rejected as an MD tie.
- Sources: `OMAT0_SPEEDUP_BRIEF.md`.

### Generic path

Families: MACE-OMAT-0 small, MACE-OFF23 S / M / L, OFF23b M, OFF24 M, MACE-MH-1 (any head), MACE-POLAR-1 S / M / L.
The first entries below are design decisions of the first version (12-13 November 2026), not A/B-timed against an
alternative; the S7 entries from gen-fork-zero-slab on carry ABBA measurements (5060 Ti only).

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

The entries below are speed work of 13 November 2026 (stage S7), A/B-timed on the RTX 5060 Ti only: the owner
ruled the workstations out (planned power outage), so none of them has an Ada measurement. Method:
`.tmp_o/ab_fam.py` (`bench_family.py` protocol, one process per side, order a b b a; b/a < 1 is faster).
Cells: OMAT-0 small `SrTiO3` 5,000 atoms; OFF23 medium `water_216` (648) and water 2x2x2 (5,184); MH-1 `SrTiO3`;
POLAR-1 NaCl 3x3x3 (216) and POLAR-1-M `NbCl5` 3x3x3 (1,944).

<a id="gen-fork-zero-slab"></a>
#### gen-fork-zero-slab: GEMM lanes, one zero slab, dead stores
- Switch: `GenericPassInputs::gemm_fork / gemm_lane / gemm_join` (engine side streams), passed into `PolarLR::record`; `GenericModel::d_zero_slab_`, `PolarLR` `bwd_zero_slab`.
- Stage: S7.
- What: the per-degree and per-species GEMMs of each layer, and the source / mixer / readout GEMMs of the long-range part, run on the engine's side streams; the per-step gradient accumulators sit in one slab cleared by one memset; `gate_bwd_kernel` no longer stores `dp` buffers that are never read (layer 0, l = 3) and the dead `dnw[2]` is gone.
- Status: DEFAULT.
- Measured (b/a against the build before, which already had pol-sf-f32): OMAT-S 0.987, OFF23-M 0.961 / 0.990, MH-1 0.975, POLAR-1-S 0.918, POLAR-1-M 0.974 / 0.963 (NaCl / `NbCl5`), POLAR-1-L 0.914. Gates: all families PASS; 47 tests pass.
- Sources: this file (commit `1310ddc`).

<a id="gen-cublas-stream"></a>
#### gen-cublas-stream: cuBLAS fallback on the lane's stream
- Switch: `sgemm_rm` in `src/engine.cu`.
- Stage: S7.
- What: GEMMs with a non-packed layout (POLAR's strided `Wpot` / `Wq` column blocks) fall back to `cublasSgemm`, which ran on the main stream even when a fork lane was selected, so it raced the side-stream GEMMs writing its operands.
- Status: FIX (found by the gate after gen-fork-zero-slab).
- Measured: POLAR-1-M crystals gate force / stress ratios 1.4e4-4e5 (energies off by 15-145 meV/atom) -> every row PASS (worst 0.72 / 0.72).
- Sources: this file.

<a id="gen-skip-fold"></a>
#### gen-skip-fold: skip-message and readout-tail folds
- Switch: `GenericLayer::skip_folded` (blocks of kind agnostic / density), `GenericLayer::ro_folded` (last layer, scalar output, no long range, MLP readout).
- Stage: S7.
- What: the per-species `W_skip_msg` that follows the message linear is folded at load (in float64) into a per-species `W_lin` (and `C0` for the first layer), removing one species GEMM per degree in each direction; on the last layer `hout = y W_p + h_in W_sc` feeds only the readout's first linear, so `W_p ro_W1` and `W_sc ro_W1` replace three GEMMs by two narrower ones. Exactness checked on CPU against MACE float64 (OMAT-S 4.9e-6 meV/atom, 4.1e-8 eV/A; OFF23-M 1.3e-5, 4.2e-8).
- Status: DEFAULT.
- Measured: OMAT-S 0.947, OFF23-M 0.939 / 0.956, MH-1 0.991. Gates PASS (OMAT-S worst force 0.91, OFF23-M 0.21, MH-1 0.64); 47 tests pass.
- Sources: this file (commit `7fff54d`).

<a id="gen-conv-v2"></a>
#### gen-conv-v2: conv and first-layer kernels rewritten
- Switch: `python/gen_generic.py` -> `gconv_fwd_L{0,1,2}<R, PAIR>`, `gconv_bwd_L{0,1,2}<WANT_DX>`; `first_fwd_kernel`, `first_bwd_kernel` in `src/generic.cu`.
- Stage: S7.
- What: (1) the conv tables are re-packed at load into the specialised path's 8-byte cubic format (`CUBIC_PACKED8`, 5-bit shared exponent; error 1e-4 to 1.8e-3 of each interval's largest value on every cached bundle, CPU check), with the edge density in its own `float4` table; (2) the forward stages each 32-edge tile's sender, spline offset and SH in shared memory, 4 (lx 0) or 2 (lx 1) receivers per block at C = 128, paired decode of edges sharing a slab; (3) the backward of lx 0 / 1 uses the late pull-back of FastMACE's `wprime_pullback` (q = M g staged per channel, geometry differentiated once per edge from 4 warp-reduced scalars; formulas checked against e3nn autograd to 1e-15 for all 14 paths), lx 2 keeps the per-channel SH gradient with a 16-shuffle fold, one warp per edge, no atomics on `edge_dvec`; (4) the first layer stages SH per 32-edge tile in the forward and walks edges per warp in the backward. Zero spills at the chosen bounds (`ptxas -v`, `sm_120`).
- Status: DEFAULT.
- Measured: OMAT-S 0.688, OFF23-M 0.789 / 0.753, MH-1 0.907, POLAR-1-S 0.987, POLAR-1-M 0.968 / 0.982, POLAR-1-L 0.971. Gates PASS for OMAT-S (worst force 1.23), OFF23-S, OFF23-M, MH-1, POLAR-1-S / -M (worst 1.71); 47 tests pass.
- Sources: this file (commit `3e526d5`).

<a id="gen-skinny-gemm"></a>
#### gen-skinny-gemm: own kernels for skinny products
- Stage: S7.
- What: `sgemm_rm` routed every product with N <= 16 or K <= 16 (any strides) to two own kernels: N <= 16 with eight warps splitting K over 4 rows and a fixed-order reduction; K <= 16 with one thread per output. Target: the strided POLAR `Wpot` / `Wq` products (`cublasSgemm` fallback, `magma_sgemmEx` 55-93 us each at 216 atoms; a 216 x 8 output with K = 2560 got a handful of CTAs).
- Status: REJECTED.
- Measured: the POLAR launches dropped to ~7 us each (nsys), but OMAT-S's packed N = 16 / K = 16 products at 5,000 rows ran 32-45 us against ~16 us for the cuBLAS / hybrid tiles. ABBA b/a: OMAT-S `SrTiO3` 1.040, POLAR-1-M `NbCl5` 1.009, OMAT-M `SrTiO3` 1.016 (the specialised path's two `skip16` products took the route; it never runs `first_bwd`). POLAR-1-M crystals gate passed (worst 1.27).
- Why: unconditional routing loses on large-M shapes; a per-call tuner candidate repeats [gemm-group-policy](#gemm-group-policy). A retry should be one static rule for the strided long-K shapes only, judged by MD A/B.
- Sources: this file.

<a id="gen-first-bwd-tail"></a>
#### gen-first-bwd-tail: batched first-layer backward tail
- Stage: S7.
- What: `first_bwd_kernel` ran `sh16_grad` and the force projection on lane 0 once per edge; each warp instead buffered 32 edges' reduced `dY[16]` and `er` in shared memory (17 KB per block) and finished them one edge per lane. Arithmetic unchanged.
- Status: REJECTED.
- Measured: ABBA b/a against gen-skinny-gemm: OMAT-S `SrTiO3` 1.040, OFF23-M water 2x2x2 1.017.
- Why: [INFERENCE] the kernel's `G` and table reads rely on L1, and the 17 KB buffer costs carveout and resident blocks; the serial tail was not the limiter.
- Sources: this file.

<a id="gen-conv-l2-split"></a>
#### gen-conv-l2-split: late pull-back for the lx 2 backward's degree <= 1 paths
- Switch: `emit_conv_bwd_lp(2)` -> `gconv_bwd_L2lp`, `emit_conv_bwd_L2(groups=(2,))` -> `gconv_bwd_L2` in `python/gen_generic.py`; `launch_conv_bwd` runs both (the first owns the density term, the second gets `dD = nullptr`).
- Stage: S8.
- What: the ten paths of x degree <= 1 are exactly lx 1's path set, so lx 1's late pull-back kernel indexed into the 17-path table covers them; the seven x degree 2 paths (no Cartesian form, as in FastMACE's `wprime_pullback`) keep the per-channel SH gradient. Falsifier first: with either group removed the old kernel took 7.15 ms (degree <= 1) and 6.92 ms (degree 2) of 13.9 ms on OFF23-L water.
- Status: DEFAULT.
- Measured: OFF23-L water backward 13.59 ms -> 4.46 + 7.47 ms per call (nsys, frozen cell); ABBA b/a 0.925 (33.91 -> 31.36 ms/step). Forces vs S7: 9.8e-7 eV/A (run-to-run 2.1e-6). OFF23-L organic gate PASS (worst stress 1.94).
- Sources: this file.

<a id="gen-first-fwd-race"></a>
#### gen-first-fwd-race: species-segment end in parity slots
- Stage: S8 (review T2, `loupmd-review-20261112.md`).
- What: `first_fwd_kernel` kept a receiver's species-segment end in one shared int; when a species had no edge no barrier separated the other warps' read from thread 0's write for the next species. Now `s_ee[s & 1]`: the next barrier bounds thread 0's lead to one slot.
- Status: FIX.
- Measured: forces unchanged at noise level (OMAT-S, MH-1, OFF23-M, POLAR-1-S). `tests/test_families.py::test_receiver_with_an_empty_species_segment_matches_mace` (OMAT-S, MH-1) covers the path, receivers with an empty species segment between non-empty ones, against MACE float64; it cannot force the warp timing of the race itself.
- Sources: this file.

<a id="gen-first-bwd-v2"></a>
#### gen-first-bwd-v2: staged gradient rows and half-warp edges in the first-layer backward
- Switch: `first_bwd_kernel<STAGE_G>` (staged when 16 S (J - 1) floats fit in 46 KB of dynamic shared memory).
- Stage: S8.
- What: ncu (OMAT-S `SrTiO3`): issue slots 79% busy, 17.2 active threads per warp, L1 hit 97%: instruction-bound on half-empty warps. (1) the receiver's `G[m][s Jp + j]` (identical for all its edges) staged once, m innermost in float4 chunks swizzled by `(k >> 1) & 3`: 4 conflict-free shared loads per column instead of 16 global ones; (2) each half-warp owns an edge and strides the J = 66 columns by 16 (5 passes per two edges instead of 3 per edge, the third with 2 lanes), the `sh16_grad` tail on lanes 0 and 16 together. Arithmetic per column unchanged.
- Status: DEFAULT.
- Measured: kernel 1,114 -> ~809 us (OMAT-S `SrTiO3`), 929 -> 723 us (OFF23-M water). ABBA b/a: OMAT-S `SrTiO3` 0.948, OFF23-M water 0.982, OMAT-S `NbCl5` 0.986, MH-1 `SrTiO3` 0.996. Forces at noise level.
- Sources: this file.

<a id="gen-conv-fwd-r1"></a>
#### gen-conv-fwd-r1: one receiver per conv-forward block
- Stage: S8.
- What: gen-conv-v2 packed 4 (lx 0) / 2 (lx 1) receivers per block at C = 128; every block lasts as long as its busiest receiver, and ncu showed achieved occupancy at half the theoretical (33% of 66%, 16% of 33%).
- Status: DEFAULT.
- Measured: kernels 676 -> 495 us (OMAT-S `gconv_fwd_L0`; R = 2: 1,060 us), 2,054 -> 1,664 us (OFF23-M `gconv_fwd_L1`; R = 4: 1,772 us). ABBA b/a: OMAT-S `SrTiO3` 0.981, OFF23-M water 0.956, water_216 0.949, POLAR-1-M NaCl 216 0.989.
- Sources: this file.

<a id="gen-conv-fwd-l2-split"></a>
#### gen-conv-fwd-l2-split: lx 2 conv forward by x degree
- Switch: `emit_conv_fwd(2, groups=...)` -> `gconv_fwd_L2a` (x degree <= 1, owns the density) and `gconv_fwd_L2b` (x degree 2, `density_table = nullptr`).
- Stage: S8.
- What: the single kernel held 71 accumulators at 174 registers (one 256-thread block per SM); the halves hold 40 / 31 at 122 / 102 registers.
- Status: DEFAULT.
- Measured: OFF23-L water forward 7.59-7.69 -> 3.10 + 3.13 ms (nsys, frozen cell); forces vs before 9.8e-7 eV/A.
- Sources: this file.

<a id="gen-conv-bwd-128"></a>
#### gen-conv-bwd-128: 128-thread conv backward blocks
- Stage: S8. Status: REJECTED.
- Measured: kernels -4% (lx 0 and 1, nsys); ABBA b/a 0.997 (OMAT-S `SrTiO3`), 0.992 (OFF23-M water). 64 threads: +17% / +22%. Below resolution, and it halves the warps per receiver on 216-atom POLAR cells. Sources: this file.

<a id="gen-conv-fwd-l1-split"></a>
#### gen-conv-fwd-l1-split: lx 1 conv forward by x degree
- Stage: S8. Status: REJECTED.
- Measured: 1.70 ms -> 0.58 + 1.20-1.29 ms (OFF23-M water); the x degree 1 kernel alone needs 128 registers, more than the joint 122. Sources: this file.

<a id="gen-zbl-in-assembly"></a>
#### gen-zbl-in-assembly: ZBL in the force-assembly pass
- Stage: S10.
- What: the separate `zbl_kernel` read every active edge and its species, did double-precision `exp` and divisions on the close ones (FP64 at 1/64 rate on GeForce) and wrote back dE/dvec, before the backward. The assembly pass now adds ZBL to each edge's total dE/dvec and sums its energy (warp reduction, double), skipping edges beyond the largest ZBL cutoff (the specialised path's `zbl_r_max_max` early-out); the per-edge terms are float32, as in MACE's float32 model, which is the gate's yardstick.
- Measured (OMAT-S `SrTiO3`, nsys, frozen cell): `zbl_kernel` 188 + `assemble_forces_kernel` 201 -> 375 us merged in double, 221 us in float32. Forces against the double version at run-to-run noise (1e-6 eV/A).
- Sources: this file.

<a id="gen-first-bwd-assembly"></a>
#### gen-first-bwd-assembly: assembly in the first-layer backward
- Stage: S10. Status: REJECTED.
- What: `first_bwd_kernel` is the last per-edge kernel, so it could finish each edge: total dE/dvec, sender force by atomics, receiver force and virial reduced per block. The specialised path does this, but its layer-0 backward is edge-parallel (one edge per lane); the generic one finishes an edge on one lane per half-warp.
- Measured (OMAT-S `SrTiO3`): with ZBL in the tail 2,243 us (128 registers, FP64 on two lanes per warp), without it 1,113 us, with `__launch_bounds__(256, 3)` (80 registers) 1,142 us, against 790 + 201 (separate assembly) = 990 us. The serial tail, not occupancy, is the cost.
- Sources: this file.

<a id="gen-first-hidden-k0-align"></a>
#### gen-first-hidden-k0-align: layer-0 `K` alignment (`sgemm128` + `3xTF32`)
- Stage: S11.
- What: `export_generic.py` wrote `J - 1 = 65` columns (64 radial + 1 bias) into `first_C0`, making `K = 195` (`S=3`) or `130` (`S=2`) and blocking both `sgemm128` (`K % 16 == 0, N % 4 == 0`) and `HybCubin` (`K % 4 == 0, N % 4 == 0`) on all layer-0 `S0 @ C0` / `dS0 = dlin @ C0^T` GEMMs. In `GenericModel::load()`, if column `J - 2` of `first_C0` is identically zero across all species and degrees (all `ScaleShiftMACE` models: OMAT-0, OFF23/23b/24, MP), it is stripped from `C0` and `table` (`J: 66 -> 65, K = 64 S`). Otherwise (nonlinear models: POLAR-1, MH-1), `K0` is padded to a multiple of 16 with zero rows in `C0` and trailing zeros written by `first_fwd_kernel`.
- Measured (5060 Ti, ABBA): OMAT-0 small `SrTiO3` 5.88 -> 5.62 ms/step (`0.956`), OMAT-0 medium generic `SrTiO3` 10.56 -> 10.30 (`0.976`), OFF23-M water 8.85 -> 8.67 (`0.979`), MH-1 `NbCl5` 13.27 -> 13.16 (`0.992`), POLAR-1-M `NbCl5` 27.37 -> 27.35 (`0.999`). Oxide, crystals, organic gates pass.
- Sources: this file.

<a id="gen-conv-bwd-lazy-q"></a>
#### gen-conv-bwd-lazy-q: lazy `s_q` loads and launch bounds in `gconv_bwd`
- Stage: S11. Status: REJECTED.
- What: `gconv_bwd_L1<true>` uses 128 registers (`REG:128, STACK:0`, 256 threads) vs `wprime_backward_inter1_rcv`'s 64 registers (`STACK:0`, 512 threads). Tried loading `sq4[0..9]` lazily per path instead of all upfront, with `__launch_bounds__(512, 2)` and `(256, 3)`, and 512 threads on `L0`.
- Measured (5060 Ti, `OFF23-M` water / `OMAT-S` `SrTiO3`): `(512, 2)` and `(256, 3)` spilled (`STACK:64`, 2,875-2,948 vs 2,671-2,776 us), and even uncapped at 128 registers lazy `sq4` loads ran at 2,942 us because preloading all `sq4` lets `nvcc` batch the `LDS.128` reads. `gconv_bwd_L0` at 512 threads ran 1,313 vs 1,257 us. Reverted.
- Sources: this file.

<a id="gen-first-bwd-edge"></a>
#### gen-first-bwd-edge: edge-parallel layer-0 backward + ZBL + force/virial assembly
- Stage: S12.
- What: `first_bwd_kernel` was receiver-parallel (1 block/atom, dynamic `s_G` staging, half-warp serial tail + separate `assemble_forces_kernel`). Ported the `interaction0.cu` `tabled_outer_sum0_backward_kernel` architecture (`first_bwd_edge_kernel<COLS, HAS_ZBL, STRESS>`, 4 warps x 32 edges/warp, `COLS = 2` for `J - 1 <= 64` with `float2` `dS0` loads and `COLS = 3` for `J - 1 <= 96`): caches `dS0` in registers across consecutive sorted active edges with the same `(rcv, snd_sp)`, folds SH gradients in `s_edge[4][32][20]`, then all 32 lanes run `sh16_grad`, `edge_dvec` sum, float32 ZBL, forces and virial in parallel. Replaces both `first_bwd_kernel` and `assemble_forces_kernel` when `J - 1 <= 96`.
- Measured (5060 Ti, ABBA): OMAT-0 small `SrTiO3` 5.62 -> 5.13 ms/step (`0.913`, kernel `779+223 -> 526 us`), OFF23-M water 8.65 -> 8.28 (`0.957`, kernel `685+149 -> 362 us`), OMAT-0 medium generic `SrTiO3` 10.30 -> 9.80 (`0.951`), MH-1 `NbCl5` 13.20 -> 13.07 (`0.990`), POLAR-1-M `NbCl5` 27.40 -> 27.27 (`0.995`). Oxide, crystals, organic gates pass.
- Sources: this file.

<a id="gen-gprod-ro1-fused"></a>
#### gen-gprod-ro1-fused: fused product + nonlinear readout and multi-warp `gprod`
- Stage: S13.
- What: generalised `gen_kernels.py`'s `sc1_fused` to `gen_generic.py` (`gprod_ro1_fused_<sig><MAX_B, MIN_BLK>`, `blockDim = C <= 512`, `ro_H = 16`): runs product forward, `s_y @ ro_W1 + sc_skip16 + ro_b1`, `c * silu` energy + derivative, `dhid @ ro_W1^T`, and product backward in one kernel without writing `y0`/`dy0` or launching `ro_W1` GEMMs or `ro_mlp_kernel`. Also switched `gprod_fwd`/`gprod_bwd` from 32 threads/block to `128` threads/block when `C % 128 == 0` (with `(threadIdx.x & 31) == 0` for warp `dD` atomics), helping `POLAR-1-M` (`C = 512`, `gprod_bwd` 435 -> 387 us/call) and layer-0 `gprod`.
- Measured (5060 Ti, ABBA): OMAT-0 small `SrTiO3` 5.12 -> 4.99 ms/step (`0.976`), OFF23-M water 8.29 -> 8.16 (`0.984`), OMAT-0 medium generic `SrTiO3` 9.81 -> 9.67 (`0.986`, generic-vs-specialised gap now +913 us vs +2,144 us before S10), POLAR-1-M `NbCl5` 27.28 -> 27.14 (`0.995`). Oxide, OFF23-S/M organic, POLAR-1-M crystals gates pass.
- Sources: this file.

<a id="gen-gconv-cfix-preloc"></a>
#### gen-gconv-cfix-preloc: `C_FIX` template and prelocated spline indices in `gconv_fwd`/`gconv_bwd`
- Stage: S14.
- What: `gconv_bwd_L1<true>` previously held 128 registers (`REG:128, STACK:0`) with runtime `C` and per-edge `locate_spline(grid)` + `species`/`sym_pair` inside the loop. Templated `C_FIX` (`96, 128, 224, 0` on `gconv_bwd_L0/L1` and `128` on `gconv_fwd_L0/L1`) and passed prelocated `act_slab` and `act_t_sx` from `scatter_active_edges_kernel` via `GenericPassInputs` when `pair_sym && C_x <= 256` and `L->grid == preloc_grid()`. `gconv_bwd_L1<true, 128, true>` drops to `REG:62, STACK:0`. On `gconv_bwd_L1<..., 128>`, 256 threads/block beat 512 threads (`OFF23-M` water 2,158 vs 2,297 us; `OMAT-M` generic `SrTiO3` 2,682 vs 2,712 us). (Separately tested `LOUP_REORDER=1` on OMAT-S, OFF23-M, OMAT-M: no win observed at `N <= 5,184`.)
- Measured (5060 Ti, clean ABBA vs S13): OMAT-0 small `SrTiO3` 4.91 -> 4.73 ms/step (`0.964`), OFF23-M water 8.00 -> 7.43 (`0.928`), MH-1 `NbCl5` 12.82 -> 12.50 (`0.976`), POLAR-1-M `NbCl5` 27.20 -> 27.06 (`0.995`), OMAT-0 medium generic `SrTiO3` 9.67 -> 8.91 (`0.921`; `ab_mode.py`: `8.73 ms` vs specialised `8.31 ms` = `1.051x`, down from `1.273x` pre-S10). Oxide, OFF23-S/M organic, MH-1, POLAR-1-M crystals gates pass; `cmp_libs.py` passes on OMAT-S, OFF23-S/M/L, MH-1, POLAR-1-M.
- Sources: this file.

<a id="gen-conv-fwd-cg-fold"></a>
#### gen-conv-fwd-cg-fold: staged CG linear forms and `s_pw` in `gconv_fwd`
- Stage: S15 (14 November 2026).
- What: applied the same per-edge Clebsch-Gordan linear-form staging (`ED = 16/36/72` for `lx = 0/1/2` in `s_sh`, computed once per edge where `edge_sh16` runs) and warp-0 `s_pw` (`float4`) polynomial scaling to `emit_conv_fwd` in `python/gen_generic.py`.
- Status: DEFAULT (5060 Ti only).
- Measured (5060 Ti): kernel profiles `OFF23-M` water `gconv_fwd_L1` `1,687.8 -> 1,572.4 us` (`-6.8%`), `OFF23-L` water `gconv_fwd_L2b` `3,029.3 -> 2,897.5 us` (`-4.4%`); ABBA `OMAT-0` small `SrTiO3` `4.8219 -> 4.7672 ms` (`0.9886`), `OFF23-L` water `28.3006 -> 27.9964 ms` (`0.9893`), `OFF23-M` water tie (`7.5261 -> 7.5063`, `0.9974`), `POLAR-1-M` `NbCl5` tie (`25.0199 -> 24.9920`, `0.9989`), `MH-1` `NaCl` `1.7464 -> 1.7138` (`0.9813`, inconclusive: candidate reps spanned `1.6816` vs `1.7460 ms`). Cached 3x gates pass on `OMAT-0` medium/small, `OFF23-S/M/L`, `OFF24-M`, `MH-1` (crystals + organic), `POLAR-1-S/M`; pytest suite passes.
- Sources: this file.

<a id="gen-conv-bwd-128-regq"></a>
#### gen-conv-bwd-128-regq: `128`-thread register-`Q` edge-staged `gconv_bwd`
- Stage: S16 (14 November 2026).
- What: replaced `s_q` shared-memory staging and `4`-iteration `ch` loop in `gconv_bwd_L0/L1` (`C = 128`) with `128` threads (`1` channel/thread), `Q` in registers, `32`-edge tile staging, and `s_pull[4][32]` reduction (`PAIR = false` and `true`).
- Status: REJECTED.
- Measured (5060 Ti): `PAIR = false` (`L1 REG:80`, `L0 REG:56`, `STACK:0`): `gconv_bwd_L1` `2,216 -> 2,147 us` (`ncu` `L1 Hit 8.51%`, `L2 79.9%`), `OFF23-M` water ABBA `7.405 -> 7.398 ms` (`0.9991` tie), `OMAT-0-S` `SrTiO3` `4.725 -> 4.712` (`0.9972`). Adding staged `s_n/s_pw/s_dpw` + `PAIR = true` hit `REG:128` (`L1`) / `80` (`L0`) and regressed MD (`7.43 -> 7.46 ms`, `4.73 -> 4.77 ms`). Reverted.
- Sources: this file.

<a id="gen-conv-l1-carve0"></a>
#### gen-conv-l1-carve0: `PreferredSharedMemoryCarveout = 0` on generic conv & `first_bwd_edge`
- Stage: S16 (14 November 2026).
- Status: REJECTED.
- Measured (5060 Ti, `OFF23-M` water): `ncu` confirmed driver dropped shared config to `8.19 KB` (`gconv_fwd_L1`) and `32.77 KB` (`gconv_bwd_L1`), raising `L1 Hit` `4.5 -> 6.8%` and `9.1 -> 15.1%` but capping each SM to `1` block (`gconv_fwd_L1` `1,537 -> 2,805 us`, `gconv_bwd_L1` `2,242 -> 3,892 us`, `first_bwd_edge` `364 -> 1,140 us`). Reverted.
- Sources: this file.

<a id="gen-conv-fwd-l1-r4"></a>
#### gen-conv-fwd-l1-r4: `R = 4, TILE = 16` in `gconv_fwd_L1`
- Stage: S16 (14 November 2026).
- Status: REJECTED.
- Measured (5060 Ti, `REG:124, SHARED:12032`, `L1 Hit 4.5 -> 18.3%`): `OMAT-0-M` generic `SrTiO3` `8.79 -> 8.66 ms` (`0.985`), `OFF23-M` water `7.44 -> 7.54` (`1.013`, `1,571 -> 1,700 us`). Interleaved water molecules lose to higher register count. Reverted (needs per-system selection if pursued).
- Sources: this file.

<a id="rv8-fixes"></a>
#### rv8-fixes: eighth review pass (`loupmd-review-20261112.md`, U1-U9)
- Stage: S9.
- U1, packed tables: both loaders (generic conv layers, specialised interaction 1) now measure the CUBIC_PACKED8 error per column, exactly (maxima over each interval from the derivative's roots / the vertex), for the value and for d/dr (dv/dt times the band's inv_h), over the whole table and separately over r >= 1 A (an interval straddling 1 A contributes its part above, extrema on the clipped range) (the export certificate's second region), relative to the column's largest value / slope; load refuses above `kPackedTableMaxRelErr` = 5e-4. Measured on every current bundle (`LOUP_DEBUG_PACK=1`): 2.7e-5 (POLAR-1-L) to 1.7e-4 (OFF23 medium); OMAT-0 medium 8.0e-5, OMAT-0 small 1.0e-4. A guard against tables far outside what has been gated, not a certificate. Test: a bundle with one entry 2^40 larger is refused.
- U4, cuBLAS: the legacy handle's workspace is set with every `cublasSetStream` (`cublas_bind`: the lane's own workspace, as cuBLAS requires for reproducibility with concurrent streams). Whether it is behind POLAR-1-S NaNbO3_c0's per-process energies is not settled: five fresh processes each before and after gave two or three distinct values either way, but each process re-timed the near-tie GEMM plans, so plan choice was not held fixed (inconclusive; needs one complete fixed plan file).
- U2: the atom reorder also gathers `d_rnd_vel`, so the second Langevin half-kick after a rebuild uses the atom's own noise.
- U3: `loup_run_*` and `loup_profile_force` return -1 unless a neighbour list exists (`nl_valid`); `loup_set_system` zeroes velocities, so a run after only `evaluate` starts from rest instead of the previous system's velocities; non-finite `dt_fs` and `n_calls <= 0` are refused. Test: the call orders.
- U6: `ensure_edge_capacity` and `GenericModel::ensure_edges` drop the capacity to 0 and the graphs before reallocating, so a failed allocation leaves no stale capacity on freed arrays. U7: the specialised variant timing waits for a list with edges. U8: spline bands must be contiguous and end at r_max (both loaders); a plan file's `hyb_tile` must be < 3.
- U5: POLAR-1-S NaNbO3_c0 is left out of the gate unless `--include-unstable` (FastMACE excludes it too).
- Symmetrix-XL checks: a cell change followed by a pair crossing r_max on the reused list matches a fresh engine (test, OMAT-0 medium and small). MACE's e3nn -> cuEquivariance conversion, which the exporter probes, reproduces the original model in float64 (OMAT-0 small, OFF23 medium: 1e-14 eV / eV/A; POLAR-1-S: 1e-10 / 3e-9), and every family's gate compares LoupMD with the original model file, so a sign flip in the conversion (their PR 34) would show there. Not covered: a model that has not been gated; a synthetic sign-flipped U-column test is still open.
- Sources: this file; `tests/test_engine_guards.py`.

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

<a id="pol-sf-f32"></a>
#### pol-sf-f32: structure factors in float32 chunks
- Switch: `sf_kernel<NC>` (128 threads per k).
- Stage: S7.
- What: each thread sums its atoms' `X cos` / `X sin` in float32 over 1,024-atom chunks and folds them into double per chunk (as the feature and gradient passes already did), with coalesced `float4` loads of X; before, every term was a 1/64-rate FP64 add.
- Status: DEFAULT.
- Measured: POLAR-1-S NaCl 0.942, POLAR-1-M NaCl 0.973, `NbCl5` 0.833, POLAR-1-L NaCl 0.987. Gates POLAR-1-S / -M / -L PASS.
- Sources: this file (commit `9417f4a`).

<a id="pol-contig-weights"></a>
#### pol-contig-weights: per-degree contiguous SCF weights
- Stage: S7.
- What: `Wpot` / `Wq` (and the electron-energy `le_Wq` / `le_Wv`) sliced at load into one contiguous matrix per degree so their GEMMs take the packed `cublasLt` / own-SGEMM path instead of the strided `cublasSgemm` fallback; `qnew = qin + out` in one kernel instead of a copy and an add.
- Status: REJECTED.
- Measured (b/a against gen-conv-v2): POLAR-1-S NaCl 1.008, POLAR-1-M NaCl 0.952, `NbCl5` 0.993, POLAR-1-L NaCl 1.045. Gates passed. Mixed on one GPU; not kept.
- Sources: this file.

<a id="pol-kpass-32"></a>
#### pol-kpass-32: small blocks for the per-atom k-space passes
- Stage: S7.
- What: `feat_pass_kernel` / `grad_pass_kernel` with 32 threads per block for N <= 512, to spread 216 atoms over more SMs.
- Status: REJECTED.
- Measured (together with pol-contig-weights): POLAR-1-S NaCl 1.074, POLAR-1-M NaCl 1.060; each block then re-stages the same k tiles, which costs more than the extra SMs gain.
- Sources: this file.

<a id="pol-k-split"></a>
#### pol-k-split: k chunks for the per-atom k-space passes
- Switch: `k_chunk(N, K)` / `k_splits_max(N)` in `src/polar.cu`; `feat_pass_kernel` and `grad_pass_kernel` take the chunk length and range over `blockIdx.y`; `k_reduce_kernel` adds the chunks.
- Stage: S8.
- What: one thread per atom over all k left 1,944 atoms in 16 blocks (< 1 warp per scheduler on 36 SMs). The k sum is now split into enough chunks for ~2 blocks per SM; each chunk keeps its double sums and writes them to `kpart[chunk][atom]`, and `k_reduce_kernel` adds the chunks in chunk order (f assigned; dq added; forces by one atomic per atom and component, as before). One chunk keeps the direct path. A first version adding the chunks with float atomics moved POLAR-1-M `NbCl5` forces by 1.05e-4 eV/A (18x run-to-run) and was replaced; the fixed-order version differs from S8 by 8.1e-6 eV/A against 6.9e-6 run-to-run.
- Measured: kernels (POLAR-1-M `NbCl5`, nsys): `grad_pass<2>` 924-964 -> 225 us, `feat_pass` 402-449 -> 107 us, `grad_pass<1>` 250-267 -> 72 us. ABBA b/a against S8, fresh GEMM plans on both sides: POLAR-1-M `NbCl5` 0.928 (29.92 -> 27.75 ms), NaCl 216 0.958; POLAR-1-S NaCl 216 0.906, NaCl 1,728 0.876; POLAR-1-L NaCl 216 0.978. Gates PASS: POLAR-1-S (NaNbO3_c0 energy row at 0.679 meV/atom, see STATUS 5.2), POLAR-1-M (worst force 1.32), POLAR-1-L (CPU refs, worst 1.15).
- Status: DEFAULT.
- Sources: this file.

<a id="pol-2d3d-nodiv"></a>
#### pol-2d3d-nodiv: 2D/3D div-free elementwise grids and fused `inv2_kernel`
- Stage: S15 (14 November 2026).
- What: replaced `int64` div/mod indexing in `add_bias_kernel`, `bcast_mul_kernel`, `bcast_mul_bwd_kernel`, `inv_kernel`, `inv_bwd_kernel` with 2D/3D grid coordinates and fused two `inv_kernel` calls into `inv2_kernel`.
- Status: REJECTED.
- Measured (5060 Ti, `POLAR-1-M` `NbCl5`): `inv2` cut `2 * 80 -> 116 us`, while `bcast_mul` (`36.5 -> 36.1 us`), `bcast_mul_bwd` (`86.8 -> 90.2 us`), and `inv_bwd` (`112.0 -> 110.2 us`) stayed DRAM-bandwidth bound; quick 2x128 ABBA `27.0598 -> 27.0719 ms` (`b/a = 1.0004`). Reverted.
- Sources: this file.

<a id="pol-sf-tiled"></a>
#### pol-sf-tiled: multi-`k`-vector warp tiling around SoA shared-`X` in `sf_kernel`
- Switch: `sf_kernel_tiled<NC, TK>`, `launch_sf` in `src/polar.cu` (`TK = 8` for `K >= 288`, `TK = 4` for `K >= 144`, fallback `sf_kernel_1`).
- Stage: S15 (14 November 2026).
- What: `sf_kernel` previously ran 1 block per `k`, so all `K` blocks re-read `X[N, NC]` and `frac` through L1/L2 (`sf_kernel<16/8/4>` scaled linearly `900 / 455 / 233 us` on `NbCl5`). Now `TK` warps (`8` `k`-vectors per 256-thread block) share one SoA tile `s_X[NC][129]` (`float4` global loads) and `s_frac[3][129]` (`TILE = 128`), keeping `STACK:0` via `__launch_bounds__(TK * 32)` without a `minBlocks` cap.
- Status: DEFAULT (5060 Ti only).
- Measured (5060 Ti): `POLAR-1-M` `NbCl5` kernel avg `sf_kernel<16/8/4>` `899.9 / 454.9 / 232.8 -> 316.1 / 150.4 / 83.5 us`; fresh-plan ABBA vs `1e110f4`: `POLAR-1-M` `NbCl5` `27.0013 -> 24.9966 ms/step` (`b/a = 0.9258`), `NaCl` 216 `4.4061 -> 4.3599` (`0.9895`), `POLAR-1-S` `NaCl` 216 `1.9230 -> 1.8569` (`0.9657`), `POLAR-1-L` `NaCl` 216 `8.3614 -> 8.3058` (`0.9934`). `POLAR-1-S/M` crystals gates pass; `POLAR-1-L` exactness verified vs `1e110f4` via `cmp_libs.py` (`nacl216` `dF = 4.33e-7`, `nbcl5` `dF = 1.21e-3`; CPU float64 stress ref for `POLAR-1-L` `NbCl5` not re-run due to >6 min CPU autograd runtime).
- Sources: this file; commit `c24b6c4`.

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

<a id="meas-plan-variance"></a>
#### meas-plan-variance: GEMM plan choice between builds
- What: the plan cache file is keyed by a hash of the whole library, so every build tunes its own plans; winners within 3% of the runner-up are re-timed in each process. On GEMM-bound small POLAR cells the plans one build pins can differ from another's by more than the code change under test.
- Measured (5060 Ti, S8 against S7): POLAR-1-S NaCl 216 1.019 and POLAR-1-M NaCl 216 1.013 with each build's existing plans; after moving all plan files aside 0.998 / 1.006 (S) and 0.969 (M). nsys of one POLAR-1-S evaluate: every non-GEMM kernel identical, one shape at 80 us (`sgemm128`, S7's plan) against 54 us (cuBLAS, S8's).
- Why it matters: A/B rows of GEMM-bound cells need fresh plans on both sides; and a tuner that times candidates inside the captured graph (rather than isolated) could recover the spread for users. Not built.
- Sources: this file.

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
